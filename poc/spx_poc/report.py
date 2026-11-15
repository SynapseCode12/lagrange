"""Summaries, tables and plots. Everything here is derived from real query and
measurement tables; no values are invented. Empty inputs produce explicit
"no data" statements."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.time import Time  # noqa: E402
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval  # noqa: E402

from .spectral import count_seasons  # noqa: E402


def mjd_to_iso(m) -> str:
    try:
        return Time(float(m), format="mjd", scale="utc").isot[:19]
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------
# Observation-level (metadata) summary
# --------------------------------------------------------------------------
def observation_summary(obs: pd.DataFrame, dropped_dups: int) -> dict:
    if obs.empty:
        return {"total": 0}
    t = obs["t_min"].astype(float)
    days = pd.Series(pd.to_datetime(Time(t.values, format="mjd").isot).date)
    month = pd.to_datetime(days.astype(str)).dt.strftime("%Y-%m")
    by = lambda col: {str(k): int(v) for k, v in obs[col].value_counts().sort_index().items()}  # noqa: E731
    return {
        "total": int(len(obs)),
        "duplicates_removed_across_releases": int(dropped_dups),
        "per_release": by("release"),
        "per_collection": by("queried_collection"),
        "per_survey_type": by("survey"),
        "per_detector": {f"D{int(k)}": int(v) for k, v in obs["detector"].value_counts().sort_index().items()},
        "per_release_detector": {f"{r}/D{int(d)}": int(n) for (r, d), n in
                                 obs.groupby(["release", "detector"]).size().items()},
        "per_pipeline_version": {f"v{int(k)}": int(v) for k, v in
                                 obs["pipeline_version"].dropna().value_counts().sort_index().items()},
        "first_obs_utc": mjd_to_iso(t.min()),
        "last_obs_utc": mjd_to_iso(t.max()),
        "n_distinct_utc_days": int(days.nunique()),
        "n_seasons_gap30d": count_seasons(t.values, 30.0),
        "per_month": {k: int(v) for k, v in month.value_counts().sort_index().items()},
        "detector_bandpass_mismatches": int(obs.get("detector_mismatch", pd.Series(False)).sum()),
    }


# --------------------------------------------------------------------------
# Fit-quality classification (warning only; never changes `usable`)
# --------------------------------------------------------------------------
def classify_fit_quality(meas: pd.DataFrame, p_alpha: float, z_mad: float) -> pd.DataFrame:
    """Add `fit_quality` ("ok" / "warning" / "n/a") and `fit_quality_notes`.

    Two kinds of evidence, both PROVISIONAL and configurable:
      * absolute: chi2 p-value < p_alpha, i.e. the residuals are unlikely if
        the ePSF model and the VARIANCE layer were both correct;
      * relative: robust outlier in log10(chi2/dof) or fit_ql, more than
        z_mad scaled MADs above the median of usable observations on the
        same detector in this run.
    The thresholds are conventional statistical levels to *surface* cases for
    inspection; criteria for rejection are to be set from real-data
    distributions, which the summary reports.
    """
    m = meas.copy()
    m["fit_quality"] = "n/a"
    m["fit_quality_notes"] = ""
    m["chi2_dof_robust_z"] = np.nan
    m["fit_ql_robust_z"] = np.nan
    if m.empty or "psf_chi2_dof" not in m:
        return m
    has = m["usable"].astype(bool) & np.isfinite(m["psf_chi2_dof"].astype(float))
    for det, g in m[has].groupby("detector"):
        for col, out, tf in (("psf_chi2_dof", "chi2_dof_robust_z", np.log10),
                             ("fit_ql", "fit_ql_robust_z", lambda v: v)):
            v = tf(g[col].astype(float).clip(lower=1e-12))
            med = np.median(v)
            mad = 1.4826 * np.median(np.abs(v - med))
            if len(v) >= 5 and mad > 0:
                m.loc[g.index, out] = (v - med) / mad
    for idx in m.index[has]:
        notes = []
        p = float(m.at[idx, "psf_chi2_pvalue"]) if "psf_chi2_pvalue" in m else np.nan
        if np.isfinite(p) and p < p_alpha:
            notes.append(f"chi2_p<{p_alpha:g}")
        for out, lab in (("chi2_dof_robust_z", "chi2_dof"), ("fit_ql_robust_z", "fit_ql")):
            z = m.at[idx, out]
            if np.isfinite(z) and z > z_mad:
                notes.append(f"{lab}_outlier(z>{z_mad:g})")
        m.at[idx, "fit_quality"] = "warning" if notes else "ok"
        m.at[idx, "fit_quality_notes"] = ";".join(notes)
    return m


def _pct(x: pd.Series) -> dict:
    v = pd.to_numeric(x, errors="coerce").dropna()
    if v.empty:
        return {}
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return {"n": int(len(v)), "p05": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4]}


# --------------------------------------------------------------------------
# ePSF truncation diagnostics (v0.3.0) - descriptive only
# --------------------------------------------------------------------------
def _cols(df, prefix):
    return [c for c in df.columns if c.startswith(prefix)]


def _radius_of(col: str) -> float:
    return float(col.rsplit("_r", 1)[1])


def truncation_summary(u: pd.DataFrame) -> dict:
    """Distributions / stacks of the truncation diagnostics over usable observations."""
    if u.empty:
        return {}
    out: dict = {}
    for c in ("epsf_raw_sum", "epsf_raw_sum_over_ovs2", "epsf_norm_factor", "epsf_edge_frac",
              "epsf_allzones_rawsum_min", "epsf_allzones_rawsum_median", "epsf_allzones_rawsum_max",
              "epsf_allzones_edge_frac_median", "epsf_allzones_edge_frac_max",
              "epsf_half_support_x_px", "trunc_n_fit_pixels", "trunc_n_fit_outside_support",
              "trunc_frac_fit_outside_support", "trunc_n_fit_model_zero", "trunc_ring_excess_frac",
              "psf_over_ap3", "psf_over_ap5"):
        if c in u:
            out[c] = _pct(u[c])
    if "epsf_raw_sum" in u and u["epsf_raw_sum"].notna().sum() > 1:
        v = u["epsf_raw_sum"].astype(float).dropna()
        out["epsf_raw_sum_relative_std_across_observed_zones"] = float(v.std() / v.mean())
    out["flux_ratio_vs_radius"] = {c.replace("trunc_flux_ratio_", ""): _pct(u[c])
                                   for c in _cols(u, "trunc_flux_ratio_")}
    stack = {}
    for c in _cols(u, "trunc_resid_z_"):
        v = pd.to_numeric(u[c], errors="coerce").dropna()
        if len(v):
            stack[c.replace("trunc_resid_z_", "")] = {
                "mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
                "n": int(len(v))}
    out["stacked_resid_z_by_annulus_px"] = stack
    cog = {}
    for c in _cols(u, "cog_r"):
        if c.startswith("cog_err"):
            continue
        R = c.replace("cog_r", "")
        m = f"trunc_cogm_r{R}"
        d = pd.to_numeric(u[c], errors="coerce")
        if m in u and d.notna().any():
            diff = (d - pd.to_numeric(u[m], errors="coerce")).dropna()
            e = pd.to_numeric(u.get(f"cog_err_r{R}", pd.Series(dtype=float)), errors="coerce").dropna()
            cog[R] = {"data_median": float(d.median()), "model_median": float(u[m].median()),
                      "data_minus_model_median": float(diff.median()) if len(diff) else None,
                      "data_minus_model_mean": float(diff.mean()) if len(diff) else None,
                      "data_minus_model_sem_scatter": float(diff.std(ddof=1) / np.sqrt(len(diff)))
                      if len(diff) > 1 else None,
                      "data_minus_model_sem_propagated": float(np.sqrt(np.sum(e**2)) / len(e))
                      if len(e) else None,
                      "single_epoch_err_median": float(e.median()) if len(e) else None,
                      "n": int(d.notna().sum())}
    out["curve_of_growth_data_vs_model"] = cog
    out["phase_dependence"] = phase_dependence(u)
    if "cog_note" in u:
        out["cog_notes"] = {str(k): int(v) for k, v in u["cog_note"].value_counts().items()}
    return out


def phase_dependence(u: pd.DataFrame) -> dict:
    """Does the flux residual within wavelength groups track truncation geometry?

    resid = (flux - group weighted mean) / err, for groups with >= 3 usable
    epochs. Spearman correlation against the number of fit pixels outside the
    ePSF support and against the radial sub-pixel offset. In the synthetic
    study the 5 px fit has a phase-dependent bias; a significant correlation
    on real data would be the corresponding signature (other causes possible).
    """
    from scipy.stats import spearmanr

    need = {"wave_group", "psf_flux_ujy", "psf_err_ujy", "trunc_n_fit_outside_support", "subpix_x"}
    if u.empty or not need <= set(u.columns):
        return {}
    parts = []
    for _, g in u[u["wave_group"].astype(str) != ""].groupby("wave_group"):
        if len(g) < 3:
            continue
        w = 1 / g["psf_err_ujy"].astype(float) ** 2
        mu = np.sum(w * g["psf_flux_ujy"]) / np.sum(w)
        parts.append(pd.DataFrame({
            "resid": (g["psf_flux_ujy"] - mu) / g["psf_err_ujy"],
            "n_out": g["trunc_n_fit_outside_support"].astype(float),
            "phase_r": np.hypot(g["subpix_x"].astype(float), g["subpix_y"].astype(float))}))
    if not parts:
        return {"note": "no wavelength group with >= 3 usable epochs"}
    d = pd.concat(parts).dropna()
    out = {"n": int(len(d))}
    for col in ("n_out", "phase_r"):
        if d[col].nunique() > 2 and len(d) >= 8:
            rho, p = spearmanr(d["resid"], d[col])
            out[f"spearman_resid_vs_{col}"] = {"rho": float(rho), "p": float(p)}
    return out


def plot_truncation_diagnostics(meas: pd.DataFrame, title: str, path: Path) -> None:
    u = meas[meas["usable"]] if not meas.empty else meas
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    (a1, a2), (a3, a4) = axes
    if u.empty or not _cols(u, "trunc_resid_z_"):
        a1.text(0.5, 0.5, "No usable measurements with diagnostics", ha="center")
    else:
        rc = _cols(u, "trunc_resid_z_")
        mids, mean, sem = [], [], []
        for c in rc:
            a, b = (float(t) for t in c.replace("trunc_resid_z_", "").split("-"))
            v = pd.to_numeric(u[c], errors="coerce").dropna()
            mids.append((a + b) / 2)
            mean.append(v.mean())
            sem.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan)
        a1.errorbar(mids, mean, yerr=sem, fmt="o-", capsize=3)
        hs = u["epsf_half_support_x_px"].median() if "epsf_half_support_x_px" in u else np.nan
        a1.axvline(hs, color="grey", ls=":", label=f"ePSF support {hs:.1f} px")
        a1.axvline(5.0, color="k", ls="--", lw=0.8, label="fit radius 5 px")
        a1.axhline(0, color="k", lw=0.5)
        a1.set_xlabel("radius [px]")
        a1.set_ylabel("stacked mean (data-model)/sigma")
        a1.set_title(f"Residual profile, stacked over {len(u)} usable obs.", fontsize=9)
        a1.legend(fontsize=7)

        fc = sorted(_cols(u, "trunc_flux_ratio_"), key=_radius_of)
        R = [_radius_of(c) for c in fc] + [5.0]
        med = [u[c].median() for c in fc] + [1.0]
        q1 = [u[c].quantile(0.25) for c in fc] + [1.0]
        q3 = [u[c].quantile(0.75) for c in fc] + [1.0]
        order = np.argsort(R)
        R, med, q1, q3 = (np.array(x)[order] for x in (R, med, q1, q3))
        a2.plot(R, med, "o-", label="median")
        a2.fill_between(R, q1, q3, alpha=0.3, label="IQR")
        a2.axvline(hs, color="grey", ls=":")
        a2.axhline(1, color="k", lw=0.5)
        a2.set_xlabel("fit radius [px]")
        a2.set_ylabel("flux(R) / flux(5 px)")
        a2.set_title("Refit flux vs fit radius (primary unchanged)", fontsize=9)
        a2.legend(fontsize=7)

        cc = sorted([c for c in _cols(u, "cog_r")], key=_radius_of)
        if cc and u[cc].notna().any().any():
            Rc = [_radius_of(c) for c in cc]
            means, sems = [], []
            for c in cc:
                v = pd.to_numeric(u[c], errors="coerce").dropna()
                means.append(v.mean())
                sems.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan)
            a3.errorbar(Rc, means, yerr=sems, fmt="o-", capsize=3,
                        label=f"data: mean +- s.e.m. (n={int(u[cc[0]].notna().sum())})")
            a3.plot(Rc, [u[f"trunc_cogm_r{_radius_of(c):g}"].mean() for c in cc], "s--", label="ePSF model (mean)")
            a3.axvline(hs, color="grey", ls=":")
            a3.set_xlabel("aperture radius [px]")
            a3.set_ylabel("flux(<R) / flux(<8 px)")
            a3.legend(fontsize=7)
        else:
            a3.text(0.5, 0.5, "curve of growth unavailable\n(cutout too small: use --cutout-size-deg 0.1)",
                    ha="center", va="center", transform=a3.transAxes, fontsize=8)
        a3.set_title("Curve of growth: data vs ePSF model", fontsize=9)

        if "epsf_raw_sum_over_ovs2" in u:
            a4.hist(u["epsf_raw_sum_over_ovs2"].dropna(), bins=30, label="observed zones")
            for c, ls in (("epsf_allzones_rawsum_min", ":"), ("epsf_allzones_rawsum_max", "--")):
                if c in u and u[c].notna().any():
                    ov = u["epsf_oversample"].median() ** 2
                    a4.axvline(u[c].median() / ov, color="r", ls=ls, label=c.replace("epsf_allzones_", "all zones ") + " (median over files)")
            a4.set_xlabel("raw ePSF array sum / ovs^2  (as stored in file)")
            a4.legend(fontsize=7)
        a4.set_title("Stored ePSF normalisation", fontsize=9)
    fig.suptitle(f"{title}: ePSF truncation diagnostics (descriptive; primary photometry unchanged)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Measurement-level summary
# --------------------------------------------------------------------------
def weighted_constancy(flux: np.ndarray, err: np.ndarray) -> dict:
    ok = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    f, e = flux[ok], err[ok]
    if f.size < 2:
        return {"wmean_ujy": float(f[0]) if f.size else np.nan, "wmean_err_ujy": np.nan,
                "chi2_dof_const": np.nan, "rms_over_median_err": np.nan}
    w = 1 / e**2
    mu = np.sum(w * f) / np.sum(w)
    chi2 = np.sum(w * (f - mu) ** 2)
    return {"wmean_ujy": float(mu), "wmean_err_ujy": float(np.sqrt(1 / np.sum(w))),
            "chi2_dof_const": float(chi2 / (f.size - 1)),
            "rms_over_median_err": float(np.std(f, ddof=1) / np.median(e))}


def group_table(meas: pd.DataFrame) -> pd.DataFrame:
    if meas.empty or "wave_group" not in meas:
        return pd.DataFrame()
    rows = []
    for gid, g in meas[meas["wave_group"] != ""].groupby("wave_group"):
        u = g[g["usable"]]
        mj = u["mjd_avg"].astype(float).values if len(u) else np.array([])
        rec = {
            "wave_group": gid, "detector": int(g["detector"].iloc[0]),
            "cwave_min_um": g["cwave_um"].min(), "cwave_max_um": g["cwave_um"].max(),
            "cwave_span_nm": 1e3 * (g["cwave_um"].max() - g["cwave_um"].min()),
            "cband_median_um": g["cband_um"].median(),
            "n_measured": len(g), "n_usable": len(u),
            "n_distinct_epochs": int(np.unique(np.round(mj, 4)).size),
            "n_seasons_gap30d": count_seasons(mj, 30.0),
            "first_utc": mjd_to_iso(mj.min()) if mj.size else "",
            "last_utc": mjd_to_iso(mj.max()) if mj.size else "",
            "span_days": float(mj.max() - mj.min()) if mj.size else np.nan,
            "releases": ",".join(sorted(g["release"].dropna().astype(str).unique())),
            "pipeline_versions": ",".join(sorted({str(int(v)) for v in g["pipeline_version"].dropna()})),
        }
        if len(u):
            rec.update(weighted_constancy(u["psf_flux_ujy"].values.astype(float),
                                          u["psf_err_ujy"].values.astype(float)))
            rec["median_snr"] = float(np.nanmedian(u["psf_snr"]))
            if "fit_quality" in u:
                okq = u[u["fit_quality"] == "ok"]
                rec["n_fit_warning"] = int((u["fit_quality"] == "warning").sum())
                rec["chi2_dof_const_fitq_ok_only"] = weighted_constancy(
                    okq["psf_flux_ujy"].values.astype(float),
                    okq["psf_err_ujy"].values.astype(float))["chi2_dof_const"]
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["n_usable", "wave_group"], ascending=[False, True])


def measurement_summary(meas: pd.DataFrame, groups: pd.DataFrame) -> dict:
    if meas.empty:
        return {"n_measured": 0}
    usable = meas[meas["usable"]]
    reasons = meas.loc[~meas["usable"], "reason"].fillna("").map(
        lambda s: s.split(":")[0][:80]).value_counts()
    warn = {}
    for s in usable.get("core_flags", pd.Series(dtype=str)).fillna(""):
        for n in filter(None, s.split(";")):
            warn[n] = warn.get(n, 0) + 1
    out = {
        "n_measured": int(len(meas)), "n_usable": int(len(usable)),
        "rejection_reasons": {k: int(v) for k, v in reasons.items()},
        "flags_near_target_in_usable_obs": dict(sorted(warn.items())),
        "wave_source_counts": {k: int(v) for k, v in meas["wave_source"].value_counts().items()}
        if "wave_source" in meas else {},
    }
    if "cwave_cal_minus_hdr_nm" in meas:
        d = meas["cwave_cal_minus_hdr_nm"].astype(float).dropna()
        if len(d):
            out["cwave_cal_minus_header_nm"] = {"median": float(d.median()),
                                                "max_abs": float(d.abs().max()), "n": int(len(d))}
    if "mjd_avg" in meas and "t_mid_sia" in meas:
        dt = (meas["mjd_avg"].astype(float) - meas["t_mid_sia"].astype(float)) * 86400
        dt = dt.dropna()
        if len(dt):
            out["header_mjd_avg_minus_sia_mid_s"] = {"median": float(dt.median()),
                                                     "max_abs": float(dt.abs().max())}
    if "omega_median_arcsec2" in meas:
        out["pixel_solid_angle_median_arcsec2"] = float(meas["omega_median_arcsec2"].median())
    if "parent_check" in meas:
        out["parent_pixel_check"] = {
            "counts": {str(k): int(v) for k, v in meas["parent_check"].value_counts().items()},
            "max_abs_dx_awcs_minus_irsa_px": float(np.nanmax(np.abs(meas["parent_dx_awcs_minus_irsa"].astype(float))))
            if meas["parent_dx_awcs_minus_irsa"].notna().any() else None,
            "max_abs_dy_awcs_minus_irsa_px": float(np.nanmax(np.abs(meas["parent_dy_awcs_minus_irsa"].astype(float))))
            if meas["parent_dy_awcs_minus_irsa"].notna().any() else None,
        }
    if "fit_quality" in meas:
        out["fit_quality"] = {
            "counts_usable": {str(k): int(v) for k, v in usable["fit_quality"].value_counts().items()},
            "notes_counts": dict(pd.Series([n for s in usable["fit_quality_notes"].fillna("")
                                            for n in s.split(";") if n]).value_counts().astype(int)),
            "psf_chi2_dof_distribution": _pct(usable.get("psf_chi2_dof", pd.Series(dtype=float))),
            "fit_ql_distribution": _pct(usable.get("fit_ql", pd.Series(dtype=float))),
            "chi2_pvalue_distribution": _pct(usable.get("psf_chi2_pvalue", pd.Series(dtype=float))),
            "fit_ql_expected_gaussian": float(np.sqrt(2 / np.pi)),
        }
    if "epsf_edge_frac" in meas:
        out["epsf_edge_frac_distribution"] = _pct(meas["epsf_edge_frac"])
    out["truncation_diagnostics"] = truncation_summary(usable)
    alt = [c for c in meas.columns if c.endswith("_minus_primary_sigma")]
    for c in alt:
        out[f"{c}_distribution"] = _pct(usable[c])
    if not groups.empty:
        nu = groups["n_usable"]
        out["matched_groups"] = {
            "n_groups": int(len(groups)),
            "max_usable_epochs_in_one_group": int(nu.max()),
            "groups_with_>=2_usable": int((nu >= 2).sum()),
            "groups_with_>=5_usable": int((nu >= 5).sum()),
            "groups_with_>=10_usable": int((nu >= 10).sum()),
            "groups_with_>=2_seasons": int((groups["n_seasons_gap30d"] >= 2).sum()),
        }
    return out


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _dates(m):
    return pd.to_datetime(Time(np.asarray(m, float), format="mjd").isot)


def plot_observation_timeline(obs: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    if obs.empty:
        ax.text(0.5, 0.5, "No observations returned by the archive", ha="center")
    else:
        for det, g in obs.groupby("detector"):
            ax.plot(_dates(g["t_min"]), np.full(len(g), det) + np.random.default_rng(0).uniform(-0.25, 0.25, len(g)),
                    "|", ms=8, alpha=0.6, label=f"D{int(det)}")
        ax.set_ylabel("detector")
        ax.set_yticks(range(1, 7))
    ax.set_title(f"{title}: every archive observation containing the target (SIA2, DR1+QR3)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_wavelength_coverage(meas: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    m = meas[np.isfinite(meas.get("cwave_um", pd.Series(dtype=float)).astype(float))] if not meas.empty else meas
    if m.empty:
        ax.text(0.5, 0.5, "No wavelength measurements", ha="center")
    else:
        for usable, mk, lab in ((True, "o", "usable"), (False, "x", "rejected")):
            s = m[m["usable"] == usable]
            if len(s):
                ax.scatter(_dates(s["mjd_avg"]), s["cwave_um"], marker=mk, s=18,
                           c=s["detector"], cmap="viridis", vmin=1, vmax=6, label=lab)
        ax.set_ylabel("CWAVE at target [um]")
        ax.legend()
    ax.set_title(f"{title}: wavelength sampled at the target vs time (LVF)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_matched_lightcurves(meas: pd.DataFrame, groups: pd.DataFrame, title: str,
                             path: Path, n_panels: int = 4) -> None:
    top = groups[groups["n_usable"] >= 2].head(n_panels) if not groups.empty else groups
    n = max(len(top), 1)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), squeeze=False)
    if top.empty:
        axes[0, 0].text(0.5, 0.5, "No wavelength-matched group has >= 2 usable epochs",
                        ha="center", transform=axes[0, 0].transAxes)
    for ax, (_, g) in zip(axes[:, 0], top.iterrows()):
        s = meas[(meas["wave_group"] == g["wave_group"]) & meas["usable"]].sort_values("mjd_avg")
        ax.errorbar(_dates(s["mjd_avg"]), s["psf_flux_ujy"], yerr=s["psf_err_ujy"],
                    fmt="o", ms=4, capsize=2, label="ePSF forced")
        if "fit_quality" in s and (s["fit_quality"] == "warning").any():
            w = s[s["fit_quality"] == "warning"]
            ax.plot(_dates(w["mjd_avg"]), w["psf_flux_ujy"], "o", mfc="none", mec="red",
                    ms=9, label="fit-quality warning (still usable)")
        if np.isfinite(s["ap_flux_ujy"]).any():
            ax.plot(_dates(s["mjd_avg"]), s["ap_flux_ujy"], "s", mfc="none", ms=4,
                    alpha=0.6, label="aperture r=3px (no ap. corr.)")
        if np.isfinite(g.get("wmean_ujy", np.nan)):
            ax.axhline(g["wmean_ujy"], color="k", lw=0.8, ls="--")
        ax.set_ylabel("flux [uJy]")
        ax.set_title(f"{g['wave_group']}: CWAVE {g['cwave_min_um']:.4f}-{g['cwave_max_um']:.4f} um, "
                     f"n={int(g['n_usable'])}, seasons={int(g['n_seasons_gap30d'])}, "
                     f"chi2/dof(const)={g.get('chi2_dof_const', np.nan):.2f}", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"{title}: brightness vs time within wavelength-matched groups")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_raw_chronological(meas: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    s = meas[meas["usable"]] if not meas.empty else meas
    if s.empty:
        ax.text(0.5, 0.5, "No usable measurements", ha="center")
    else:
        sc = ax.scatter(_dates(s["mjd_avg"]), s["psf_flux_ujy"], c=s["cwave_um"], s=14, cmap="plasma")
        fig.colorbar(sc, ax=ax, label="CWAVE [um]")
        ax.set_ylabel("flux [uJy]")
    ax.set_title(f"{title}: EXPLORATORY raw chronology - mixes wavelengths, NOT a light curve")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_stamps(meas: pd.DataFrame, groups: pd.DataFrame, cache_dir: Path, title: str,
                path: Path, n: int = 12) -> None:
    if groups.empty or meas.empty:
        return
    gid = groups.iloc[0]["wave_group"]
    s = meas[(meas["wave_group"] == gid) & meas["usable"]].sort_values("mjd_avg").head(n)
    if s.empty:
        return
    cols = min(6, len(s))
    rows = int(np.ceil(len(s) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.9 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), s.iterrows()):
        with fits.open(cache_dir / r["cutout_file"]) as hdul:
            img = hdul["IMAGE"].data.astype(float)  # original IMAGE, ZODI not subtracted
        norm = ImageNormalize(img, interval=ZScaleInterval(), stretch=AsinhStretch())
        ax.imshow(img, origin="lower", cmap="gray", norm=norm)
        ax.plot(r["x"], r["y"], "+", color="cyan", ms=10)
        ax.set_title(f"{mjd_to_iso(r['mjd_avg'])[:10]}\n{r['cwave_um']:.4f} um", fontsize=7)
    fig.suptitle(f"{title}: IMAGE cutouts in group {gid} (no zodi subtraction)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
