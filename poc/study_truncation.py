#!/usr/bin/env python
"""Synthetic study of ePSF truncation: fit radius and normalisation.

!! SYNTHETIC. Analytic PSFs, not the SPHEREx PSF. This establishes how the
!! *method* responds to truncation; it cannot say how much the real SPHEREx
!! ePSF is truncated - that requires real data (see README section 5.8).

Set-up
------
* Truth: exact flux in each native pixel for a known, unit-normalised PSF
  (Gaussian mixtures: erf products; Moffat: numerical sub-pixel integration
  of the analytically normalised profile). Image = F*T + B, F = 1.
* ePSF table: built from the same PSF by the ePSF definition (flux in a whole
  pixel centred at each 1/5-pixel offset), truncated to 33x33 (+-3.2 px).
  Its sum / 25 = e_table, the fraction of the PSF's flux inside the support.
* Two normalisation hypotheses for the stored array:
    renorm25 - rescale the array to sum 25 (CURRENT PoC behaviour);
    raw      - use the array as stored, i.e. as if the archive normalised
               the PSF *before* truncation (stored sum = 25*e_table).
  Which one matches SPHEREx is NOT known; the real raw sums will tell.
* Fits use the PoC's own forced_psf_fit (flux + constant background) with
  Photutils ImagePSF sampling, uniform pixel noise, over 5x5 sub-pixel phases,
  at fit radii 2.5 ... 8 px. Noiseless fits give the systematic bias;
  covariance gives the relative statistical error.
* The same diagnostic functions as the real pipeline (spx_poc.truncation) are
  evaluated at the CURRENT configuration (renorm25, r_fit = 5 px) so their
  synthetic signatures can be compared with the first real run.

Usage:  python study_truncation.py [--outdir study_output]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf

from spx_poc import truncation
from spx_poc.photometry import EPSFZone, PhotConfig, aperture_phot, forced_psf_fit, sample_epsf

OVS, NTAB, STAMP = 5, 33, 31
HS = (NTAB - 1) / 2 / OVS  # 3.2 px
RADII = (2.5, 3.0, 3.2, 4.0, 5.0, 6.0, 8.0)
PHASES = np.linspace(-0.4, 0.4, 5)
SNR_REF = 200.0  # total S/N used to express residual signatures as z-scores


# --------------------------------------------------------------------------
# Analytic PSFs (unit total flux)
# --------------------------------------------------------------------------
def gauss_mix(comps):
    def pix(xc, yc):
        out = 0.0
        for w, sx, sy in comps:
            bx = 0.5 * (erf((xc + 0.5) / (np.sqrt(2) * sx)) - erf((xc - 0.5) / (np.sqrt(2) * sx)))
            by = 0.5 * (erf((yc + 0.5) / (np.sqrt(2) * sy)) - erf((yc - 0.5) / (np.sqrt(2) * sy)))
            out = out + w * bx * by
        return out
    return pix


def moffat(fwhm, beta, sub=15):
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    norm = (beta - 1) / (np.pi * alpha**2)
    o = (np.arange(sub) + 0.5) / sub - 0.5  # sub-pixel sample offsets

    def pix(xc, yc):
        xc, yc = np.asarray(xc, float), np.asarray(yc, float)
        acc = np.zeros(np.broadcast(xc, yc).shape)
        for dy in o:
            for dx in o:
                r2 = (xc + dx) ** 2 + (yc + dy) ** 2
                acc += norm * (1 + r2 / alpha**2) ** (-beta)
        return acc / sub**2
    return pix


def gauss_with_moffat_wing(core_sigma, wing_frac, wing_fwhm, wing_beta):
    g = gauss_mix([(1 - wing_frac, core_sigma, core_sigma)])
    m = moffat(wing_fwhm, wing_beta)
    return lambda xc, yc: g(xc, yc) + wing_frac * m(xc, yc)


PSFS = {
    "Gauss s=0.8 (compact)": gauss_mix([(1.0, 0.8, 0.8)]),
    "Gauss s=1.1 (mild trunc.)": gauss_mix([(1.0, 1.1, 1.1)]),
    "Gauss 0.8 + 5% halo s=3": gauss_mix([(0.95, 0.8, 0.8), (0.05, 3.0, 3.0)]),
    "Gauss 0.8 + 10% halo s=5": gauss_mix([(0.90, 0.8, 0.8), (0.10, 5.0, 5.0)]),
    "Moffat FWHM1.8 beta=4": moffat(1.8, 4.0),
    "Moffat FWHM1.8 beta=2.5": moffat(1.8, 2.5),
    "Gauss 0.8 + 8% Moffat(b=2) wing": gauss_with_moffat_wing(0.8, 0.08, 2.5, 2.0),
}


def table_for(pix):
    off = (np.arange(NTAB) - (NTAB - 1) / 2) / OVS
    dy, dx = np.meshgrid(off, off, indexing="ij")
    return pix(dx, dy)  # [y, x]


def zone(table, normalisation):
    if normalisation == "renorm25":
        arr = table / table.sum() * OVS**2
    elif normalisation == "raw":
        # As stored: entries are true per-pixel flux fractions of the *untruncated*
        # unit PSF, so the array sums to 25 * e_table (< 25).
        arr = table.copy()
    else:
        raise ValueError(normalisation)
    return EPSFZone(0, arr, OVS, 0, 0, np.nan, np.nan, 0, "parent",
                    half_support_x=HS, half_support_y=HS)


# --------------------------------------------------------------------------
def run(outdir: Path) -> pd.DataFrame:
    c = (STAMP - 1) / 2
    yy, xx = np.mgrid[0:STAMP, 0:STAMP].astype(float)
    omega = np.ones((STAMP, STAMP))
    rows, diag_rows = [], []
    for name, pix in PSFS.items():
        table = table_for(pix)
        e_table = table.sum() / OVS**2  # fraction of PSF flux inside the support
        ring = np.ones_like(table, bool)
        ring[1:-1, 1:-1] = False
        edge = table[ring].sum() / table.sum()
        for norm in ("renorm25", "raw"):
            z = zone(table, norm)
            per_r = {R: {"bias": [], "bkg_abs": [], "sA": []} for R in RADII}
            for fx in PHASES:
                for fy in PHASES:
                    x0, y0 = c + fx, c + fy
                    T = pix(xx - x0, yy - y0)
                    P = sample_epsf(z, T.shape, x0, y0, method="photutils")
                    r = np.hypot(xx - x0, yy - y0)
                    ones = np.ones_like(T)
                    for R in RADII:
                        reg = r <= R
                        fit = forced_psf_fit(T, ones, P, omega, reg)  # noiseless, B_true = 0
                        per_r[R]["bias"].append(fit["A"] - 1.0)
                        per_r[R]["bkg_abs"].append(fit["B"] * reg.sum())  # flux absorbed by B in fit area
                        per_r[R]["sA"].append(fit["sA"])
                    if norm == "renorm25":
                        # diagnostic signatures at the CURRENT configuration (r_fit=5)
                        sig = np.sqrt(np.ones_like(T))
                        fit5 = forced_psf_fit(T, sig**2, P, omega, r <= 5.0)
                        sigma_pix = 1.0 / (SNR_REF * fit5["sA"])  # noise giving S/N_REF
                        s2 = np.full_like(T, sigma_pix**2)
                        fit5 = forced_psf_fit(T, s2, P, omega, r <= 5.0)
                        d = truncation.diagnostics(T, s2, P, omega, np.ones_like(T, bool), r,
                                                   x0, y0, HS, HS, fit5, 5.0)
                        ap5 = aperture_phot(T, s2, omega, np.ones_like(T, bool), r,
                                            PhotConfig(r_ap=5.0, r_in=7.0, r_out=12.0))
                        d["psf_over_ap5"] = fit5["A"] / ap5["ap_flux_ujy"] * 1e12  # ap in uJy units
                        d.update({k: v for k, v in truncation.curve_of_growth(
                            T, s2, omega, np.ones_like(T, bool), r, x0, y0).items()
                            if k.startswith("cog_r")})
                        d.update(psf=name, phase_x=fx, phase_y=fy)
                        diag_rows.append(d)
            sA5 = np.mean(per_r[5.0]["sA"])
            for R in RADII:
                b = np.array(per_r[R]["bias"])
                rows.append({
                    "psf": name, "normalisation": norm, "r_fit": R,
                    "e_table": e_table, "missing_frac": 1 - e_table, "edge_frac": edge,
                    "flux_bias_mean": b.mean(), "flux_bias_phase_spread": b.max() - b.min(),
                    "bkg_absorbed_frac_mean": float(np.mean(per_r[R]["bkg_abs"])),
                    "rel_stat_err_vs_r5": float(np.mean(per_r[R]["sA"]) / sA5),
                })
    df = pd.DataFrame(rows)
    dg = pd.DataFrame(diag_rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "truncation_study.csv", index=False)
    dg.to_csv(outdir / "truncation_signatures_r5_renorm25.csv", index=False)
    return df, dg


def plot(df: pd.DataFrame, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, norm in zip(axes, ("renorm25", "raw")):
        for name, g in df[df["normalisation"] == norm].groupby("psf", sort=False):
            ax.errorbar(g["r_fit"], 100 * g["flux_bias_mean"], yerr=50 * g["flux_bias_phase_spread"],
                        marker="o", ms=4, capsize=2, label=f"{name} (missing {100 * g['missing_frac'].iloc[0]:.1f}%)")
        ax.axvline(HS, color="grey", ls=":", lw=1)
        ax.axvline(5.0, color="k", ls="--", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("fit radius [native px]  (dotted: ePSF support 3.2 px; dashed: current 5 px)")
        ax.set_title({"renorm25": "array rescaled to sum 25 (CURRENT PoC)",
                      "raw": "array used as stored (hypothesis: normalised before truncation)"}[norm],
                     fontsize=9)
    axes[0].set_ylabel("flux bias [%]  (bars: half the sub-pixel-phase spread)")
    axes[1].legend(fontsize=7, loc="lower left")
    fig.suptitle("SYNTHETIC: recovered flux vs fit radius for analytic PSFs, noiseless, F=1", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "truncation_study.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("study_output"))
    args = ap.parse_args()
    print("SYNTHETIC ePSF TRUNCATION STUDY - analytic PSFs, not SPHEREx data\n")
    df, dg = run(args.outdir)
    plot(df, args.outdir)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    print("1) PSFs: fraction of flux inside the 33x33 (+-3.2 px) ePSF support (e_table),\n"
          "   and fraction of the table's own flux in its outermost ring (edge_frac)")
    print(df.drop_duplicates("psf")[["psf", "e_table", "missing_frac", "edge_frac"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    for norm in ("renorm25", "raw"):
        print(f"\n2) Flux bias [%] vs fit radius, normalisation = {norm}"
              f"{'  (CURRENT)' if norm == 'renorm25' else '  (hypothesis)'}")
        t = df[df["normalisation"] == norm].pivot(index="psf", columns="r_fit", values="flux_bias_mean") * 100
        print(t.reindex(PSFS.keys()).to_string(float_format=lambda v: f"{v:+7.2f}"))
        print("   sub-pixel-phase spread of the bias [% peak-to-peak]:")
        t = df[df["normalisation"] == norm].pivot(index="psf", columns="r_fit", values="flux_bias_phase_spread") * 100
        print(t.reindex(PSFS.keys()).to_string(float_format=lambda v: f"{v:7.2f}"))
        print("   source flux absorbed by the background term within the fit area [% of F]:")
        t = df[df["normalisation"] == norm].pivot(index="psf", columns="r_fit", values="bkg_absorbed_frac_mean") * 100
        print(t.reindex(PSFS.keys()).to_string(float_format=lambda v: f"{v:+7.2f}"))

    print("\n3) Statistical cost: sigma_flux(r) / sigma_flux(5 px), background-limited noise")
    t = df[df["normalisation"] == "renorm25"].pivot(index="psf", columns="r_fit", values="rel_stat_err_vs_r5")
    print(t.reindex(PSFS.keys()).to_string(float_format=lambda v: f"{v:7.3f}"))

    print(f"\n4) Diagnostic signatures at the CURRENT configuration (renorm25, r_fit=5), S/N={SNR_REF:g}"
          " (means over phases)")
    cols = (["n_fit_pixels", "n_fit_outside_support", "ring_excess_frac", "psf_over_ap5",
             "flux_ratio_r3.2", "flux_ratio_r4", "flux_ratio_r8",
             "cog_r3.2", "cogm_r3.2", "cog_r5", "cogm_r5"]
            + [c for c in dg.columns if c.startswith("resid_z_")])
    print(dg.groupby("psf", sort=False)[cols].mean().reindex(PSFS.keys()).T
          .to_string(float_format=lambda v: f"{v:+.3f}"))
    print(f"\nWrote {args.outdir / 'truncation_study.csv'}, "
          f"{args.outdir / 'truncation_signatures_r5_renorm25.csv'}, {args.outdir / 'truncation_study.png'}")


if __name__ == "__main__":
    main()
