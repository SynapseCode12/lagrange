#!/usr/bin/env python
"""Project Lagrange - Milestone 1 proof of concept.

Queries the IRSA SPHEREx archive (DR1 + QR3 only) for two test regions,
characterises the real observation coverage, determines the wavelength
actually sampled at a target star in each exposure, groups exposures into
wavelength-matched sets, and measures the star's brightness over time.

Nothing in this script fabricates data. If the archive cannot be reached the
failure is logged and reported in summary.md / summary.json, and no
measurement tables are produced for that region.

Run `python inspect_spherex.py --help` for options; see README.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Import astroquery before configuring logging: astroquery installs its own
# logger class, which breaks if logging.getLogger("astroquery") ran first.
import astroquery.ipac.irsa  # noqa: F401,E402

from spx_poc import __version__, archive, quality, report
from spx_poc.measure import measure_cutout
from spx_poc.photometry import PhotConfig
from spx_poc.spectral import CalSpectralWCS, group_by_wavelength, pairwise_violations

log = logging.getLogger("spx_poc")


# --------------------------------------------------------------------------
# Regions
# --------------------------------------------------------------------------
@dataclass
class Region:
    key: str
    name: str
    ra: float
    dec: float
    star_search_arcmin: float
    star_exclude_arcmin: float = 0.0
    note: str = ""
    target_override: tuple[float, float] | None = None


REGIONS = {
    # North Ecliptic Pole, J2000 (RA 18h00m00s, Dec +66d33m38.55s). The SPHEREx
    # northern deep field is located at the NEP; its exact footprint is not
    # assumed here - coverage is measured from the archive.
    "nep": Region("nep", "North Ecliptic Pole deep field", 270.0, 66.5607083,
                  star_search_arcmin=10.0, note="primary candidate"),
    # M101 position as used in IRSA's SPHEREx cutout tutorial. Photometry uses a
    # field star outside the galaxy disk (r > 15'), because M101 itself is extended.
    "m101": Region("m101", "M101 field (control)", 210.80227, 54.34895,
                   star_search_arcmin=25.0, star_exclude_arcmin=15.0, note="control"),
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def setup_logging(outdir: Path) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(outdir / "run_log.txt", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(ch)
    root.addHandler(fh)
    for noisy in ("urllib3", "astroquery", "pyvo", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def environment_info() -> dict:
    import astropy, astroquery, matplotlib, pyvo, requests, scipy  # noqa: E401
    return {
        "poc_version": __version__, "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
        "astropy": astropy.__version__, "astroquery": astroquery.__version__,
        "pyvo": pyvo.__version__, "requests": requests.__version__,
        "matplotlib": matplotlib.__version__,
        "run_started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def angsep_arcsec(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600


# --------------------------------------------------------------------------
# Target star selection (2MASS PSC via IRSA TAP)
# --------------------------------------------------------------------------
def select_star(region: Region, kmin: float, kmax: float, net: archive.NetConfig) -> dict:
    r_deg = region.star_search_arcmin / 60
    adql = ("SELECT designation, ra, dec, j_m, h_m, k_m, ph_qual, cc_flg FROM fp_psc "
            f"WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {region.ra}, {region.dec}, {r_deg}))=1")
    cat = archive.tap_query(adql, net, f"2MASS PSC around {region.key}")
    cat = cat[np.isfinite(cat["k_m"].astype(float))].reset_index(drop=True)
    sep_c = angsep_arcsec(cat["ra"], cat["dec"], region.ra, region.dec)
    cand = cat[(cat["k_m"] >= kmin) & (cat["k_m"] <= kmax) & (cat["ph_qual"] == "AAA")
               & (cat["cc_flg"] == "000") & (sep_c >= region.star_exclude_arcmin * 60)].copy()
    cand["sep_center_arcsec"] = sep_c[cand.index]
    chosen = None
    for _, s in cand.sort_values("sep_center_arcsec").iterrows():
        d = angsep_arcsec(cat["ra"], cat["dec"], s["ra"], s["dec"])
        others = cat[d > 0.5]
        d = d[d > 0.5]
        if ((d < 40) & (others["k_m"] < s["k_m"] + 3)).any():
            continue  # neighbour within ~6.5 px brighter than 1/16 of the star
        if ((d < 90) & (others["k_m"] < s["k_m"])).any():
            continue  # brighter star within ~15 px
        chosen = s
        break
    info = {"catalog": "2MASS PSC (IRSA TAP table fp_psc)", "n_catalog_sources": int(len(cat)),
            "n_candidates_mag_quality": int(len(cand)),
            "criteria": f"{kmin}<=Ks<={kmax}, ph_qual=AAA, cc_flg=000, no neighbour Ks<K+3 within 40\", "
                        f"no brighter star within 90\", >= {region.star_exclude_arcmin}' from region centre"}
    if chosen is None:
        raise RuntimeError("no isolated 2MASS star satisfies the selection criteria; "
                           "supply --<region>-target RA DEC")
    info.update({k: (float(chosen[k]) if k not in ("designation", "ph_qual", "cc_flg") else str(chosen[k]))
                 for k in ("designation", "ra", "dec", "j_m", "h_m", "k_m", "ph_qual", "cc_flg",
                           "sep_center_arcsec")})
    return info


# --------------------------------------------------------------------------
# Archive queries for one position
# --------------------------------------------------------------------------
def sia_all(ra: float, dec: float, releases: list[str], radius: float,
            net: archive.NetConfig) -> tuple[pd.DataFrame, dict]:
    frames, status = [], {}
    for rel in releases:
        for coll in archive.RELEASE_COLLECTIONS[rel]:
            try:
                df = archive.query_sia(ra, dec, radius, coll, net)
                status[coll] = {"ok": True, "rows": int(len(df))}
                log.info("  SIA %-18s -> %d rows", coll, len(df))
                frames.append(archive.normalise_sia(df, rel))
            except Exception as exc:  # noqa: BLE001
                status[coll] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    frames = [f for f in frames if not f.empty]
    obs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not obs.empty:
        n0 = len(obs)
        obs = obs.drop_duplicates(subset=["access_url"])
        if len(obs) < n0:
            log.info("  removed %d rows with identical access_url", n0 - len(obs))
    return obs, status


def sample_evenly(df: pd.DataFrame, nmax: int) -> pd.DataFrame:
    d = df.sort_values("t_min")
    if len(d) <= nmax:
        return d
    idx = np.unique(np.round(np.linspace(0, len(d) - 1, nmax)).astype(int))
    return d.iloc[idx]


def load_cal_wcs(detectors, releases, cache: Path, session, net) -> tuple[dict, dict]:
    cal, status = {}, {}
    for det in detectors:
        for rel in releases:  # DR1 and QR3 share calibrations per IRSA; try in order
            try:
                url = archive.find_spectral_wcs_url(session, rel, det, net)
            except Exception as exc:  # noqa: BLE001
                status[f"D{det}"] = f"discovery error: {exc}"[:200]
                continue
            if not url:
                status[f"D{det}"] = f"no spectral_wcs file found under {rel}/spectral_wcs/"
                continue
            try:
                p = archive.download_file(session, url, cache / "cal" / Path(url).name, net,
                                          validate_fits=False)
                cal[det] = CalSpectralWCS(p)
                status[f"D{det}"] = f"ok: {url}"
                break
            except Exception as exc:  # noqa: BLE001
                status[f"D{det}"] = f"download/read error ({url}): {exc}"[:300]
    return cal, status


# --------------------------------------------------------------------------
# One region end-to-end
# --------------------------------------------------------------------------
def run_region(region: Region, args, net, session, outdir: Path, cache: Path) -> dict:
    res: dict = {"region": region.name, "key": region.key, "note": region.note,
                 "center_ra": region.ra, "center_dec": region.dec, "status": "started"}
    tag = region.key
    log.info("=== Region %s: %s (RA %.5f, Dec %+.5f) ===", tag, region.name, region.ra, region.dec)

    # 1. Coverage at the region centre ---------------------------------------
    obs_c, st_c = sia_all(region.ra, region.dec, args.releases, args.search_radius_arcsec, net)
    res["sia_center_status"] = st_c
    if not any(v["ok"] for v in st_c.values()):
        res["status"] = "ARCHIVE_UNREACHABLE (SIA failed for every collection)"
        return res
    obs_c, dup_c = archive.deduplicate(obs_c)
    res["center_observations"] = report.observation_summary(obs_c, len(dup_c))
    if not obs_c.empty:
        obs_c.to_csv(outdir / f"{tag}_center_observations.csv", index=False)

    # 2. Photometry target ---------------------------------------------------
    if region.target_override:
        tra, tdec = region.target_override
        res["target"] = {"source": "user override", "ra": tra, "dec": tdec}
    else:
        try:
            res["target"] = select_star(region, args.kmag[0], args.kmag[1], net)
            tra, tdec = res["target"]["ra"], res["target"]["dec"]
        except Exception as exc:  # noqa: BLE001
            res["target"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            res["status"] = "NO_TARGET (coverage reported; photometry skipped)"
            return res
    log.info("  target: RA %.6f Dec %+.6f  %s", tra, tdec,
             res["target"].get("designation", res["target"].get("source", "")))

    # 3. Coverage at the target ----------------------------------------------
    obs, st_t = sia_all(tra, tdec, args.releases, args.search_radius_arcsec, net)
    res["sia_target_status"] = st_t
    obs, dup_t = archive.deduplicate(obs)
    res["target_observations"] = report.observation_summary(obs, len(dup_t))
    if obs.empty:
        res["status"] = "NO_OBSERVATIONS at target"
        return res
    obs["t_mid_sia"] = (obs["t_min"].astype(float) + obs["t_max"].astype(float)) / 2
    obs.to_csv(outdir / f"{tag}_target_observations.csv", index=False)
    report.plot_observation_timeline(obs, region.name, outdir / f"{tag}_observation_timeline.png")

    # 4. Independent TAP cross-check -----------------------------------------
    if not args.skip_tap:
        try:
            tap = archive.tap_images_at(tra, tdec, net)
            tap = tap[tap["release"].isin(args.releases)]
            res["tap_crosscheck"] = {
                "rows_dr1_qr3": int(len(tap)),
                "per_release": {k: int(v) for k, v in tap["release"].value_counts().items()},
                "per_band": {str(k): int(v) for k, v in tap["energy_bandpassname"].value_counts().sort_index().items()},
                "sia_rows_before_dedup": int(len(obs) + len(dup_t)),
            }
            tap.to_csv(outdir / f"{tag}_tap_images.csv", index=False)
        except Exception as exc:  # noqa: BLE001
            res["tap_crosscheck"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}

    # 5. Cutouts for the selected detectors -----------------------------------
    sel_all = obs[obs["detector"].isin(args.detectors)]
    parts = [sample_evenly(g, args.max_cutouts) for _, g in sel_all.groupby("detector")]
    sel = pd.concat(parts) if parts else sel_all
    res["cutout_selection"] = {
        f"D{d}": {"available": int((sel_all["detector"] == d).sum()),
                  "processed": int((sel["detector"] == d).sum())} for d in args.detectors}
    if any(v["processed"] < v["available"] for v in res["cutout_selection"].values()):
        res["cutout_selection"]["note"] = ("SAMPLED: only an evenly time-spaced subset was processed "
                                           "(raise --max-cutouts to process all).")
    log.info("  cutouts to process: %d (of %d on selected detectors)", len(sel), len(sel_all))

    ccache = cache / "cutouts"
    jobs, dest_by_idx = {}, {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for idx, r in sel.iterrows():
            url = archive.cutout_url(r["access_url"], tra, tdec, args.cutout_size_deg)
            dest = archive.cutout_cache_path(ccache, r["access_url"], tra, tdec, args.cutout_size_deg)
            dest_by_idx[idx] = dest
            jobs[ex.submit(archive.download_file, session, url, dest, net)] = (idx, dest)
        done, failed, t0 = 0, [], time.time()
        for fut in as_completed(jobs):
            idx, dest = jobs[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                failed.append((idx, f"{type(exc).__name__}: {exc}"[:200]))
            done += 1
            if done % 25 == 0 or done == len(jobs):
                log.info("  downloaded %d/%d (%d failed, %.0f s)", done, len(jobs), len(failed), time.time() - t0)
    res["cutout_download_failures"] = len(failed)

    # 6. Calibration spectral WCS --------------------------------------------
    cal = {}
    if not args.no_cal_wcs:
        cal, res["cal_spectral_wcs"] = load_cal_wcs(args.detectors, args.releases, cache, session, net)

    # 7. Measure ---------------------------------------------------------------
    cfg = PhotConfig(psf_method=args.psf_method)
    failed_idx = dict(failed)
    rows = []
    meta_cols = ["release", "queried_collection", "survey", "obs_id", "file_obsid", "detector",
                 "pipeline_version", "processing_date", "t_min", "t_max", "t_mid_sia",
                 "energy_bandpassname", "access_url"]
    for idx, r in sel.iterrows():
        base = {c: r.get(c) for c in meta_cols}
        if idx in failed_idx:
            rows.append({**base, "usable": False, "reason": f"download failed: {failed_idx[idx]}"})
            continue
        m = measure_cutout(dest_by_idx[idx], tra, tdec, cfg, cal, args.strict_flags, detector=int(r["detector"]))
        m.pop("detector_expected", None)
        hdr_det = m.pop("detector", None)
        rows.append({**base, **m, "detector_header": hdr_det})
    meas = pd.DataFrame(rows)
    if "mjd_avg" not in meas:
        meas["mjd_avg"] = np.nan
    meas["mjd_avg"] = pd.to_numeric(meas["mjd_avg"], errors="coerce").fillna(meas["t_mid_sia"].astype(float))
    for c in ("cwave_um", "cband_um"):
        if c not in meas:
            meas[c] = np.nan
    meas = report.classify_fit_quality(meas, args.fitq_pvalue, args.fitq_mad)
    meas["wave_group"] = group_by_wavelength(meas, args.tol_frac)
    groups = report.group_table(meas)

    meas.to_csv(outdir / f"{tag}_measurements.csv", index=False)
    groups.to_csv(outdir / f"{tag}_wavelength_groups.csv", index=False)
    res["measurements"] = report.measurement_summary(meas, groups)
    res["measurements"]["group_pairwise_violations"] = pairwise_violations(meas, args.tol_frac)
    res["measurements"]["flag_policy"] = quality.policy(args.strict_flags)
    rec_only = sorted({n for s in meas.get("flag_names_record_only", pd.Series(dtype=str)).dropna()
                       for n in str(s).split(",") if n})
    res["measurements"]["flag_names_seen_record_only"] = rec_only
    res["top_groups"] = groups.head(8).to_dict(orient="records") if not groups.empty else []

    report.plot_wavelength_coverage(meas, region.name, outdir / f"{tag}_wavelength_vs_time.png")
    report.plot_matched_lightcurves(meas, groups, region.name, outdir / f"{tag}_lightcurve_matched.png")
    report.plot_raw_chronological(meas, region.name, outdir / f"{tag}_raw_chronological_EXPLORATORY.png")
    try:
        report.plot_truncation_diagnostics(meas, region.name, outdir / f"{tag}_truncation_diagnostics.png")
    except Exception as exc:  # noqa: BLE001
        log.warning("truncation diagnostics plot failed: %s", exc)
    try:
        report.plot_stamps(meas, groups, ccache, region.name, outdir / f"{tag}_stamps.png")
    except Exception as exc:  # noqa: BLE001
        log.warning("stamp plot failed: %s", exc)
    res["status"] = "COMPLETED"
    return res


# --------------------------------------------------------------------------
# Summary document
# --------------------------------------------------------------------------
def write_summary_md(summary: dict, path: Path) -> None:
    L = ["# SPHEREx PoC run summary", "",
         f"- Run started (UTC): {summary['environment']['run_started_utc']}",
         f"- Releases: {', '.join(summary['config']['releases'])}; detectors: "
         f"{summary['config']['detectors']}; tolerance: |dCWAVE| <= {summary['config']['tol_frac']} x CBAND",
         f"- SPHEREx SIA collections visible: {', '.join(summary.get('sia_collections', [])) or 'n/a'}",
         "", "All numbers below were produced by this run's archive queries and measurements.", ""]
    for r in summary["regions"]:
        L += [f"## {r['region']} ({r['key']})", "", f"**Status:** {r['status']}", ""]
        t = r.get("target", {})
        if t:
            L.append("**Target:** " + ", ".join(f"{k}={v}" for k, v in t.items()))
            L.append("")
        for label, key in (("Coverage at region centre", "center_observations"),
                           ("Coverage at target", "target_observations")):
            o = r.get(key)
            if not o:
                continue
            L += [f"### {label}", "", f"- total observations: {o.get('total')}"]
            for k in ("duplicates_removed_across_releases", "per_release", "per_collection",
                      "per_survey_type", "per_detector", "per_release_detector",
                      "per_pipeline_version", "first_obs_utc", "last_obs_utc",
                      "n_distinct_utc_days", "n_seasons_gap30d", "per_month"):
                if k in o:
                    L.append(f"- {k}: {o[k]}")
            L.append("")
        for k in ("tap_crosscheck", "cutout_selection", "cutout_download_failures",
                  "cal_spectral_wcs", "measurements"):
            if k in r:
                L += [f"### {k}", "", "```", json.dumps(to_jsonable(r[k]), indent=2), "```", ""]
        if r.get("top_groups"):
            g = pd.DataFrame(r["top_groups"])
            cols = [c for c in ("wave_group", "cwave_min_um", "cwave_max_um", "cband_median_um",
                                "n_usable", "n_seasons_gap30d", "first_utc", "last_utc",
                                "wmean_ujy", "wmean_err_ujy", "chi2_dof_const", "n_fit_warning",
                                "chi2_dof_const_fitq_ok_only", "median_snr")
                    if c in g]
            L += ["### Top wavelength-matched groups", "", g[cols].to_markdown(index=False, floatfmt=".4g")
                  if _has_tabulate() else g[cols].to_string(index=False), ""]
        for k in ("sia_center_status", "sia_target_status"):
            if k in r:
                bad = {c: v for c, v in r[k].items() if not v.get("ok")}
                if bad:
                    L += [f"**{k} errors:** {bad}", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regions", nargs="+", default=["nep", "m101"], choices=list(REGIONS))
    p.add_argument("--releases", nargs="+", default=["dr1", "qr3"], choices=list(archive.RELEASE_COLLECTIONS))
    p.add_argument("--detectors", nargs="+", type=int, default=[2], choices=range(1, 7),
                   help="detectors (bands) to measure; coverage is reported for all six")
    p.add_argument("--max-cutouts", type=int, default=150,
                   help="max cutouts per region per detector (evenly spaced in time)")
    p.add_argument("--cutout-size-deg", type=float, default=0.05)
    p.add_argument("--tol-frac", type=float, default=0.25,
                   help="wavelength match tolerance as a fraction of CBAND")
    p.add_argument("--search-radius-arcsec", type=float, default=1.0)
    p.add_argument("--kmag", nargs=2, type=float, default=[11.0, 13.0], metavar=("KMIN", "KMAX"),
                   help="2MASS Ks range for automatic target star selection")
    p.add_argument("--nep-target", nargs=2, type=float, metavar=("RA", "DEC"))
    p.add_argument("--m101-target", nargs=2, type=float, metavar=("RA", "DEC"))
    p.add_argument("--no-cal-wcs", action="store_true",
                   help="skip calibration spectral WCS (~32 MB per detector); use header W WCS only")
    p.add_argument("--strict-flags", action="store_true",
                   help="also mask " + "/".join(quality.WARN_STRICT) + " pixels")
    p.add_argument("--psf-method", choices=["photutils", "linear"], default="photutils",
                   help="primary ePSF evaluation (the other is recorded as a cross-check)")
    p.add_argument("--fitq-pvalue", type=float, default=1e-3,
                   help="PROVISIONAL: warn if PSF-fit chi2 p-value is below this (warning only)")
    p.add_argument("--fitq-mad", type=float, default=5.0,
                   help="PROVISIONAL: warn if chi2/dof or fit_ql is > N robust sigmas above "
                        "the per-detector median (warning only)")
    p.add_argument("--skip-tap", action="store_true", help="skip TAP cross-check")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--outdir", type=Path, default=Path("output"))
    p.add_argument("--cache-dir", type=Path, default=Path("cache"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir / f"run_{run_id}"
    outdir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(outdir)

    net = archive.NetConfig(timeout_s=args.timeout, retries=args.retries)
    archive.configure_astropy_timeouts(net)
    session = archive.make_session(net)
    if args.nep_target:
        REGIONS["nep"].target_override = tuple(args.nep_target)
    if args.m101_target:
        REGIONS["m101"].target_override = tuple(args.m101_target)

    summary = {"environment": environment_info(),
               "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "regions": []}
    log.info("Output directory: %s", outdir.resolve())

    try:
        summary["sia_collections"] = archive.list_spherex_collections(net)
        log.info("SPHEREx SIA collections: %s", summary["sia_collections"])
    except Exception as exc:  # noqa: BLE001
        summary["sia_collections_error"] = f"{type(exc).__name__}: {exc}"[:300]
    if not args.skip_tap:
        try:
            archive.tap_spherex_schema(net).to_csv(outdir / "tap_schema_spherex.csv", index=False)
        except Exception as exc:  # noqa: BLE001
            summary["tap_schema_error"] = f"{type(exc).__name__}: {exc}"[:300]

    exit_code = 0
    for key in args.regions:
        try:
            res = run_region(REGIONS[key], args, net, session, outdir, args.cache_dir)
        except Exception as exc:  # noqa: BLE001
            log.exception("Region %s failed", key)
            res = {"region": REGIONS[key].name, "key": key, "status": f"FAILED: {type(exc).__name__}: {exc}"[:300]}
        if res["status"] != "COMPLETED":
            exit_code = 2
        summary["regions"].append(res)

    summary["environment"]["run_finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (outdir / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2, default=str),
                                         encoding="utf-8")
    write_summary_md(summary, outdir / "summary.md")
    log.info("Done. Status: %s", "; ".join(f"{r['key']}={r['status']}" for r in summary["regions"]))
    for h in logging.getLogger().handlers:
        h.flush()
    bundle = shutil.make_archive(str(outdir.parent / f"review_bundle_{run_id}"), "zip", outdir)
    print(f"\nSend back this file for review: {Path(bundle).resolve()}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
