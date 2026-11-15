#!/usr/bin/env python
"""Offline self-test of the PoC code paths using SYNTHETIC data.

!! Everything in this file is synthetic. It tests the code (parsing, masking,
!! photometry statistics, wavelength grouping); it says nothing about SPHEREx
!! data. No output of this script may be presented as an observation.

Independence: synthetic images are generated from an ANALYTIC PSF (exact
erf pixel integrals, see validate_epsf.py), not from the code's own ePSF
sampler; the EPSF table handed to the code is built from the same analytic
PSF by the ePSF definition. The header 'W' spectral WCS (-TAB) is not
synthesised; a synthetic calibration CWAVE/CBAND file is used instead.

Usage:  python selftest_synthetic.py      (exit 0 if every check passes)
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

import validate_epsf as ve
from spx_poc import quality
from spx_poc.measure import measure_cutout
from spx_poc.photometry import MJY_TO_UJY, SR_TO_ARCSEC2, PhotConfig, pixel_solid_angle
from spx_poc.report import classify_fit_quality
from spx_poc.spectral import (CalSpectralWCS, _wcs, comparable, group_by_wavelength,
                              pairwise_violations)

RNG = np.random.default_rng(12345)
PIX_DEG = 6.15 / 3600
PSF_TRUE = [(1.0, 0.8, 0.8)]          # analytic PSF used to make images
FLAG_NAMES = ["TRANSIENT", "OVERFLOW", "SUR_ERROR", "NONFUNC", "DICHROIC", "MISSING_DATA",
              "HOT", "COLD", "FULLSAMPLE", "PHANMISS", "NONLINEAR", "PERSIST", "OUTLIER",
              "CROSSTALK", "SOURCE", "GHOST", "STREAK", "BLOOM", "SNOWBALL", "HALO",
              "SATELLITE_HALO"]
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


# --------------------------------------------------------------------------
# Synthetic MEF
# --------------------------------------------------------------------------
def make_mef(path: Path, ra0: float, dec0: float, flux_ujy: float, bkg: float,
             noise_var: float, flag_center: str | None = None,
             parent_origin=(1000.0, 500.0), awcs_consistent: bool = True,
             psf_image=PSF_TRUE, psf_table=PSF_TRUE, n: int = 31,
             subpixel: bool = True) -> None:
    """MEF following the documented layout. 'A' keywords follow IRSA's
    convention: parent0 = 1 + x0 - CRPIXnA  (so CRVALnA = 0, CDELTnA = 1).
    awcs_consistent=False sets CRVAL1A = 5 to create a deliberate mismatch."""
    h = fits.Header()
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    # random sub-pixel placement of the target so all interpolation phases are exercised
    dx, dy = (RNG.uniform(-0.5, 0.5, 2) if subpixel else (0.0, 0.0))
    h["CRPIX1"], h["CRPIX2"] = (n + 1) / 2 + dx, (n + 1) / 2 + dy
    h["CRVAL1"], h["CRVAL2"] = ra0, dec0
    h["CDELT1"], h["CDELT2"] = -PIX_DEG, PIX_DEG
    h["CUNIT1"] = h["CUNIT2"] = "deg"
    h["CTYPE1A"], h["CTYPE2A"] = "LINEAR", "LINEAR"
    h["CRPIX1A"], h["CRPIX2A"] = 1.0 - parent_origin[0], 1.0 - parent_origin[1]
    h["CRVAL1A"], h["CRVAL2A"] = (0.0 if awcs_consistent else 5.0), 0.0
    h["CDELT1A"] = h["CDELT2A"] = 1.0
    h["MJD-AVG"], h["DATE-AVG"], h["XPOSURE"] = 61000.5, "2025-09-01T12:00:00", 113.58
    h["DETECTOR"], h["BUNIT"], h["OBSID"] = 2, "MJy/sr", "SYNTHETIC"
    wcs = _wcs(h)
    omega = pixel_solid_angle(wcs, (n, n))
    x0, y0 = wcs.all_world2pix([[ra0, dec0]], 0)[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    T = ve.pixel_integrals(psf_image, xx - x0, yy - y0)   # analytic truth
    var = np.full((n, n), noise_var)
    img = bkg + (flux_ujy / MJY_TO_UJY) * T / omega + RNG.normal(0, np.sqrt(noise_var), (n, n))
    bits = {nm: i for i, nm in enumerate(FLAG_NAMES)}
    flags = np.zeros((n, n), np.int32)
    flags[int(round(y0)), int(round(x0))] |= 1 << bits["SOURCE"]
    if flag_center:
        flags[int(round(y0)), int(round(x0))] |= 1 << bits[flag_center]
    fh = fits.Header()
    for nm, b in bits.items():
        fh[f"MSKN{b:04d}"] = nm
        fh[f"MSKD{b:04d}"] = f"synthetic {nm}"
    table = ve.make_epsf_table(psf_table)
    xc = np.repeat([340.0, 1020.0, 1700.0], 3)
    yc = np.tile([340.0, 1020.0, 1700.0], 3)
    cols = [fits.Column("XCENTER", "E", array=xc), fits.Column("YCENTER", "E", array=yc),
            fits.Column("CWAVE", "E", array=np.full(9, 1.3)),
            fits.Column("CBAND", "E", array=np.full(9, 0.03)),
            fits.Column("EPSF", "1089E", dim="(33,33)", array=np.repeat(table[None], 9, 0))]
    ep = fits.BinTableHDU.from_columns(cols, name="EPSF")
    ep.header["OVSMPX"] = ep.header["OVSMPY"] = 5
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(img.astype(np.float32), h, name="IMAGE"),
                  fits.CompImageHDU(flags, fh, name="FLAGS"),
                  fits.ImageHDU(var.astype(np.float32), name="VARIANCE"),
                  fits.ImageHDU(np.full((n, n), bkg * 0.9, np.float32), name="ZODI"),
                  ep]).writeto(path, overwrite=True)


def make_cal(path: Path) -> None:
    yy, _ = np.mgrid[0:2040, 0:2040]
    cw = (1.10 + 0.52 * yy / 2039).astype(np.float32)  # LVF gradient along y
    cb = (cw / 41).astype(np.float32)
    hw, hb = fits.Header(), fits.Header()
    hw["BUNIT"] = hb["BUNIT"] = "um"
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(cw, hw, name="CWAVE"),
                  fits.ImageHDU(cb, hb, name="CBAND")]).writeto(path, overwrite=True)


def legacy_seed_only_grouping(df, tol):
    """The v0.1.0 algorithm (compares only with the seed's CBAND); kept here
    only to demonstrate that the new test detects its flaw."""
    gid = pd.Series("", index=df.index, dtype=object)
    for det, sub in df.sort_values("cwave_um").groupby("detector"):
        k, sw, sb = -1, None, None
        for idx, r in sub.iterrows():
            if sw is None or (r.cwave_um - sw) > tol * sb:
                k += 1
                sw, sb = r.cwave_um, r.cband_um
            gid[idx] = f"D{det}-G{k:03d}"
    return gid


# --------------------------------------------------------------------------
def main() -> int:
    print("SYNTHETIC SELF-TEST - not SPHEREx data\n")
    tmp = Path(tempfile.mkdtemp(prefix="spx_selftest_"))
    ra0, dec0 = 270.0, 66.56
    cfg = PhotConfig()
    truth = 5000.0

    print("-- geometry & flags")
    h = fits.Header({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRPIX1": 16, "CRPIX2": 16,
                     "CRVAL1": ra0, "CRVAL2": dec0, "CDELT1": -PIX_DEG, "CDELT2": PIX_DEG})
    om = np.median(pixel_solid_angle(_wcs(h), (31, 31))) * SR_TO_ARCSEC2
    check("pixel solid angle ~ 6.15^2 arcsec^2", abs(om / 6.15**2 - 1) < 1e-3, f"({om:.4f})")
    fh = fits.Header()
    for b, nm in enumerate(FLAG_NAMES):
        fh[f"MSKN{b:04d}"] = nm
    bits = quality.parse_flag_bits(fh)
    _, missing = quality.mask_for(quality.EXCLUDE, bits)
    check("flag names parsed", len(bits) == len(FLAG_NAMES) and not missing)
    check("decode bitmask", quality.decode((1 << bits["HOT"]) | (1 << bits["SOURCE"]), bits) == ["HOT", "SOURCE"])

    cal_path = tmp / "spectral_wcs_D2_synthetic.fits"
    make_cal(cal_path)
    cal = {2: CalSpectralWCS(cal_path)}

    print("\n-- item 4: DICHROIC / strict-flag policy")
    check("DICHROIC masked with --strict-flags", "DICHROIC" in quality.masked_names(True))
    check("DICHROIC not masked by default", "DICHROIC" not in quality.masked_names(False))
    check("DICHROIC listed as record-only by default",
          "DICHROIC" in quality.record_only_names(bits, strict=False)
          and "DICHROIC" not in quality.record_only_names(bits, strict=True))
    check("SOURCE never masked", "SOURCE" not in quality.masked_names(True))
    readme = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")
    m = re.search(r"`--strict-flags` additionally masks exactly:\s*([A-Z_, ]+)", readme)
    doc_set = {t.strip() for t in m.group(1).split(",")} if m else set()
    check("README strict-flag list == code (quality.WARN_STRICT)", doc_set == set(quality.WARN_STRICT),
          f"(README: {sorted(doc_set)})")
    p = tmp / "mef_dichroic.fits"
    make_mef(p, ra0, dec0, truth, 0.3, 2.5e-3, flag_center="DICHROIC")
    r_def = measure_cutout(p, ra0, dec0, cfg, cal, detector=2)
    r_str = measure_cutout(p, ra0, dec0, cfg, cal, detector=2, strict_flags=True)
    check("DICHROIC target: usable by default (warning), rejected with --strict-flags",
          r_def["usable"] and "DICHROIC" in r_def["core_flags"] and not r_str["usable"])

    print("\n-- item 5: parent-pixel cross-check (IRSA CRPIXnA formula vs A-WCS)")
    p = tmp / "mef_first.fits"
    make_mef(p, ra0, dec0, truth, 0.3, 2.5e-3)
    first = measure_cutout(p, ra0, dec0, cfg, cal, detector=2)
    check("IRSA formula gives expected parent pixel",
          abs(first["x_parent"] - (1000 + first["x"])) < 1e-9 and abs(first["y_parent"] - (500 + first["y"])) < 1e-9
          and first["parent_method"] == "irsa_crpixA", f"(x={first['x']:.3f}->{first['x_parent']:.3f})")
    check("consistent header -> parent_check=agree",
          first["parent_check"] == "agree" and abs(first["parent_dx_awcs_minus_irsa"]) < 1e-9)
    p = tmp / "mef_bad_awcs.fits"
    make_mef(p, ra0, dec0, truth, 0.3, 2.5e-3, awcs_consistent=False)
    bad = measure_cutout(p, ra0, dec0, cfg, cal, detector=2)
    check("inconsistent header -> DISAGREE flagged and difference recorded",
          bad["parent_check"] == "DISAGREE" and abs(bad["parent_dx_awcs_minus_irsa"] - 5.0) < 1e-9,
          f"(dx={bad['parent_dx_awcs_minus_irsa']:+.3f} px)")
    expected_w = 1.10 + 0.52 * round(first["y_parent"]) / 2039
    check("CWAVE from calibration map at parent pixel", first["wave_source"] == "calibration"
          and abs(first["cwave_um"] - expected_w) < 1e-5, f"({first['cwave_um']:.5f} um)")
    check("header W WCS absent in synthetic file -> recorded, not invented",
          not np.isfinite(first["cwave_hdr_um"]) and "wave_hdr_error" in first)
    xc = np.repeat([340.0, 1020.0, 1700.0], 3)
    yc = np.tile([340.0, 1020.0, 1700.0], 3)
    nearest = int(np.argmin(np.hypot(xc - first["x_parent"], yc - first["y_parent"])))
    check("ePSF zone = nearest zone in parent frame", first.get("epsf_frame") == "parent"
          and first.get("epsf_zone") == nearest, f"(zone={first.get('epsf_zone')})")

    print("\n-- item 2: ePSF vs analytic truth and photutils (independent)")
    r = ve.evaluate(PSF_TRUE, "photutils", noise_trials=200)
    check("photutils ImagePSF: shape error < 0.5% of peak", r["shape_err_max"] < 0.005, f"({r['shape_err_max']:.4f})")
    check("photutils ImagePSF: native samples sum to 1 at all phases",
          abs(r["sum_min"] - 1) < 2e-3 and abs(r["sum_max"] - 1) < 2e-3, f"({r['sum_min']:.4f}-{r['sum_max']:.4f})")
    check("photutils ImagePSF: noiseless flux bias < 0.1% at all phases",
          max(abs(r["bias_noiseless_min"]), abs(r["bias_noiseless_max"])) < 1e-3)
    rl = ve.evaluate(PSF_TRUE, "linear", noise_trials=50)
    check("linear interpolation is measurably worse (documents why it is not primary)",
          rl["shape_err_max"] > 5 * r["shape_err_max"], f"({rl['shape_err_max']:.4f} vs {r['shape_err_max']:.4f})")
    check("axis convention (y-elongated stays y-elongated), both methods", all(ve.axis_convention_check().values()))

    print("\n-- end-to-end flux recovery on analytic-truth images (200 noise realisations)")
    pulls, frac_ap, pvals, alt_sig = [], [], [], []
    for i in range(200):
        p = tmp / f"mef_{i}.fits"
        make_mef(p, ra0, dec0, truth, bkg=0.3, noise_var=2.5e-3)
        row = measure_cutout(p, ra0, dec0, cfg, cal, detector=2)
        if row["usable"]:
            pulls.append((row["psf_flux_ujy"] - truth) / row["psf_err_ujy"])
            pvals.append(row["psf_chi2_pvalue"])
            alt_sig.append(row.get("psf_flux_linear_minus_primary_sigma", np.nan))
            if np.isfinite(row.get("ap_flux_ujy", np.nan)):
                frac_ap.append(row["ap_flux_ujy"] / truth)
        p.unlink()
    pulls, pvals = np.array(pulls), np.array(pvals)
    check("all synthetic measurements usable", len(pulls) == 200, f"({len(pulls)}/200)")
    check("primary method is photutils", row.get("psf_method") == "photutils")
    check("flux unbiased (|mean pull| < 0.2)", abs(pulls.mean()) < 0.2, f"(mean={pulls.mean():+.3f})")
    check("errors calibrated (0.85 < std pull < 1.15)", 0.85 < pulls.std(ddof=1) < 1.15, f"(std={pulls.std(ddof=1):.3f})")
    check("aperture r=3px captures most flux (no ap. corr.)", 0.9 < np.median(frac_ap) <= 1.02,
          f"(median frac={np.median(frac_ap):.3f})")
    alt_sig = np.array(alt_sig)
    check("linear cross-check recorded", np.isfinite(alt_sig).all(),
          f"(linear - photutils: median {np.median(alt_sig):+.3f} sigma, max |{np.max(np.abs(alt_sig)):.3f}| sigma)")

    print("\n-- item 3: fit-quality metrics & classification (warning only)")
    ks = np.mean(pvals < 0.1)
    check("chi2 p-values ~uniform under correct model (frac<0.1 in 0.04-0.17)", 0.04 < ks < 0.17,
          f"({ks:.3f})")
    p = tmp / "mef_wrongpsf.fits"
    make_mef(p, ra0, dec0, 2e5, 0.3, 2.5e-3, psf_image=[(1.0, 1.05, 1.05)], psf_table=PSF_TRUE)
    wrong = measure_cutout(p, ra0, dec0, cfg, cal, detector=2)
    check("mis-specified PSF at high S/N -> tiny chi2 p-value, still 'usable'",
          wrong["usable"] and wrong["psf_chi2_pvalue"] < 1e-6, f"(p={wrong['psf_chi2_pvalue']:.1e}, "
          f"chi2/dof={wrong['psf_chi2_dof']:.1f}, fit_ql={wrong['fit_ql']:.2f})")
    df = pd.DataFrame({"usable": True, "detector": 2,
                       "psf_chi2_dof": np.r_[RNG.normal(1, 0.1, 30).clip(0.5), 8.0],
                       "fit_ql": np.r_[RNG.normal(0.8, 0.05, 30), 2.5],
                       "psf_chi2_pvalue": np.r_[RNG.uniform(0.01, 1, 30), 1e-9]})
    df.loc[0, "usable"] = False
    out = classify_fit_quality(df, 1e-3, 5.0)
    check("classification flags the outlier", out.iloc[-1]["fit_quality"] == "warning"
          and "chi2_p" in out.iloc[-1]["fit_quality_notes"], f"({out.iloc[-1]['fit_quality_notes']})")
    check("classification never changes 'usable'", (out["usable"] == df["usable"]).all())
    check("unusable rows get fit_quality n/a", out.iloc[0]["fit_quality"] == "n/a")
    check("well-behaved rows mostly ok", (out.iloc[1:-1]["fit_quality"] == "ok").mean() > 0.9)

    print("\n-- HOT / TRANSIENT masking")
    p = tmp / "mef_hot.fits"
    make_mef(p, ra0, dec0, truth, 0.3, 2.5e-3, flag_center="HOT")
    row = measure_cutout(p, ra0, dec0, cfg, cal, detector=2)
    check("HOT target pixel -> unusable", not row["usable"] and "HOT" in row["reason"])
    p = tmp / "mef_transient.fits"
    make_mef(p, ra0, dec0, truth, 0.3, 2.5e-3, flag_center="TRANSIENT")
    row = measure_cutout(p, ra0, dec0, cfg, cal, detector=2)
    check("TRANSIENT recorded as warning only (default policy)", row["usable"] and "TRANSIENT" in row["core_flags"])
    row = measure_cutout(p, ra0, dec0, cfg, cal, detector=2, strict_flags=True)
    check("TRANSIENT rejected with --strict-flags", not row["usable"])

    print("\n-- item 1: wavelength grouping (pairwise min-CBAND criterion)")
    df = pd.DataFrame({"detector": [2, 2, 2, 2, 3],
                       "cwave_um": [1.300, 1.305, 1.309, 1.330, 1.700],
                       "cband_um": [0.032] * 4 + [0.041]})
    g = group_by_wavelength(df, 0.25)
    check("constant CBAND: tolerance 8 nm", list(g) == ["D2-G000", "D2-G000", "D2-G001", "D2-G002", "D3-G000"],
          f"({list(g)})")
    # seed has wide CBAND, candidate narrow: 8 nm <= 0.25*40 nm (seed) but > 0.25*20 nm (min)
    df = pd.DataFrame({"detector": [2, 2], "cwave_um": [1.300, 1.308], "cband_um": [0.040, 0.020]})
    g = group_by_wavelength(df, 0.25)
    check("varying CBAND: uses min(CBAND) of the pair, not the seed's", g.iloc[0] != g.iloc[1],
          f"({list(g)}; legacy: {list(legacy_seed_only_grouping(df, 0.25))})")
    # violation only against a NON-seed member
    df = pd.DataFrame({"detector": [2, 2, 2], "cwave_um": [1.300, 1.301, 1.306],
                       "cband_um": [0.040, 0.016, 0.040]})
    g = group_by_wavelength(df, 0.25)
    check("varying CBAND: candidate checked against every member",
          g.iloc[0] == g.iloc[1] and g.iloc[2] != g.iloc[1], f"({list(g)})")
    check("pairwise criterion function", comparable(2, 1.300, 0.032, 2, 1.307, 0.032, 0.25)
          and not comparable(2, 1.300, 0.032, 3, 1.300, 0.032, 0.25)
          and not comparable(2, 1.300, 0.040, 2, 1.308, 0.020, 0.25))
    n = 3000
    df = pd.DataFrame({"detector": RNG.integers(1, 4, n),
                       "cwave_um": RNG.uniform(1.10, 1.62, n),
                       "cband_um": RNG.uniform(0.015, 0.050, n)})
    df["wave_group"] = group_by_wavelength(df, 0.25)
    nv = pairwise_violations(df, 0.25)
    df_leg = df.copy()
    df_leg["wave_group"] = legacy_seed_only_grouping(df, 0.25)
    nv_leg = pairwise_violations(df_leg, 0.25)
    check("random varying-CBAND set: 0 pairwise violations", nv == 0,
          f"({df['wave_group'].nunique()} groups; legacy algorithm: {nv_leg} violations)")
    check("validator detects violations (sensitivity)", nv_leg > 0)
    for tol in (0.1, 0.25, 0.5):
        df["wave_group"] = group_by_wavelength(df, tol)
        if pairwise_violations(df, tol):
            break
    check("0 violations at tol_frac 0.1/0.25/0.5", pairwise_violations(df, tol) == 0)

    print("\n-- v0.3.0: ePSF truncation diagnostics (do not change primary photometry)")
    p = tmp / "mef_trunc.fits"
    make_mef(p, ra0, dec0, truth, 0.3, 2.5e-3, subpixel=False)
    base = measure_cutout(p, ra0, dec0, cfg, cal, detector=2)
    tab_sum = float(ve.make_epsf_table(PSF_TRUE).sum())
    check("raw ePSF sum recorded as stored", abs(base["epsf_raw_sum"] - tab_sum) < 1e-4 * tab_sum,
          f"({base['epsf_raw_sum']:.4f}; /25 = {base['epsf_raw_sum_over_ovs2']:.4f})")
    check("normalisation factor = 25 / raw sum (float32 storage)",
          abs(base["epsf_norm_factor"] * tab_sum / 25 - 1) < 1e-6)
    check("all-zone raw-sum statistics recorded", base.get("epsf_allzones_n") == 9
          and abs(base["epsf_allzones_rawsum_max"] - tab_sum) < 1e-3)
    # rescale the stored ePSF by 0.9 in an otherwise identical file
    p2 = tmp / "mef_trunc_scaled.fits"
    with fits.open(p) as hd:
        hd["EPSF"].data["EPSF"] = hd["EPSF"].data["EPSF"] * 0.9
        hd.writeto(p2, overwrite=True)
    scaled = measure_cutout(p2, ra0, dec0, cfg, cal, detector=2)
    check("stored-array scale changes raw sum/norm factor only, not the (renormalised) primary flux",
          abs(scaled["epsf_raw_sum"] / base["epsf_raw_sum"] - 0.9) < 1e-6
          and abs(scaled["psf_flux_ujy"] / base["psf_flux_ujy"] - 1) < 1e-6,
          f"(flux ratio {scaled['psf_flux_ujy'] / base['psf_flux_ujy']:.9f})")
    # independent count from the recorded star position (lattice of pixel centres)
    yy, xx = np.mgrid[0:31, 0:31]
    rr = np.hypot(xx - base["x"], yy - base["y"])
    n_fit = int((rr <= 5).sum())
    n_out = int(((rr <= 5) & ((np.abs(xx - base["x"]) > 3.2) | (np.abs(yy - base["y"]) > 3.2))).sum())
    check("fit-pixel and outside-support counts match an independent count",
          base["trunc_n_fit_pixels"] == n_fit and base["trunc_n_fit_outside_support"] == n_out,
          f"({base['trunc_n_fit_pixels']} fit, {base['trunc_n_fit_outside_support']} outside; "
          f"expected {n_fit}, {n_out}; ~1/3 of the 5 px fit region lies beyond the support)")
    check("model is exactly 0 on every outside-support fit pixel",
          base["trunc_n_fit_model_zero"] == base["trunc_n_fit_outside_support"])
    check("half-support = (33-1)/2/5 = 3.2 px", abs(base["epsf_half_support_x_px"] - 3.2) < 1e-12)
    check("radius refits recorded (flux_ratio at r=4 and r=6)",
          np.isfinite(base["trunc_flux_ratio_r4"]) and np.isfinite(base["trunc_flux_ratio_r6"]))
    check("curve of growth computed with uncertainties", np.isfinite(base["cog_r3.2"])
          and np.isfinite(base["cog_err_r3.2"]) and np.isfinite(base["trunc_cogm_r3.2"]))

    print("\n-- v0.3.0: item 4 - normalising a truncated ePSF to 25 (noiseless, analytic truth)")
    import study_truncation as stt
    from spx_poc.photometry import forced_psf_fit, sample_epsf
    for name in ("Gauss 0.8 + 5% halo s=3", "Moffat FWHM1.8 beta=2.5"):
        pix = stt.PSFS[name]
        table = stt.table_for(pix)
        miss = 1 - table.sum() / 25
        res = {}
        for norm in ("renorm25", "raw"):
            z = stt.zone(table, norm)
            c = (stt.STAMP - 1) / 2
            yy, xx = np.mgrid[0:stt.STAMP, 0:stt.STAMP].astype(float)
            x0, y0 = c + 0.23, c - 0.31
            T = pix(xx - x0, yy - y0)
            P = sample_epsf(z, T.shape, x0, y0)
            fit = forced_psf_fit(T, np.ones_like(T), P, np.ones_like(T), np.hypot(xx - x0, yy - y0) <= 3.2)
            res[norm] = fit["A"] - 1
        check(f"{name}: renorm25 bias (r<=3.2) = -missing fraction",
              abs(res["renorm25"] + miss) < 5e-4, f"(bias {100 * res['renorm25']:+.2f}%, missing {100 * miss:.2f}%)")
        check(f"{name}: as-stored (pre-truncation normalised) bias ~ 0 within support",
              abs(res["raw"]) < 5e-4, f"({100 * res['raw']:+.3f}%)")

    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed (synthetic data only).")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
