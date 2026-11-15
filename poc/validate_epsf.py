#!/usr/bin/env python
"""Independent validation of the ePSF evaluation used for photometry.

!! SYNTHETIC. This validates code against an analytic truth; it is not a
!! measurement of the real SPHEREx PSF.

Why this is independent
-----------------------
Neither interpolator under test generates the truth. For an analytic optical
PSF g(u, v) (sums of axis-aligned elliptical Gaussians), the exact flux in a
detector pixel is a product of erf differences:

  T(x_pix, y_pix; x0, y0) = integral over the pixel square of g(u - x0, v - y0)

The ePSF table given to the code is built the same way, by definition of an
ePSF (Anderson & King 2000; IRSA ePSF tutorial): the oversampled entry at
offset (dx, dy) = (k/ovs, l/ovs) is the fraction of flux in a whole pixel
centred at that offset. It is then truncated to 33x33 at ovs=5, as in
SPHEREx DR1/QR3 EPSF extensions.

Compared, at many sub-pixel source positions:
  * truth           T (exact pixel integrals, infinite PSF)
  * photutils       photutils.psf.ImagePSF.evaluate (IRSA tutorial approach)
  * linear          the PoC's hand-written map_coordinates(order=1)

Checks: shape agreement with truth (max |model - T| / peak), normalisation
(sum of native samples), x/y axis convention (elliptical PSF), and flux
recovery through the actual fitting function (noiseless bias, and pulls over
noisy realisations).

Usage: python validate_epsf.py      (exit 0 if all acceptance checks pass)
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.special import erf

from spx_poc.photometry import EPSFZone, forced_psf_fit, sample_epsf

OVS, NTAB, STAMP = 5, 33, 21
PHASES = np.round(np.arange(-0.5, 0.5001, 0.1), 3)  # sub-pixel offsets in x and y
RNG = np.random.default_rng(2026)

# (name, list of (weight, sigma_x, sigma_y)). Widths in native pixels.
# Part A: PSFs essentially contained in the +-3.2 px table -> tests the
#         interpolation code itself, with pass/fail acceptance.
PSFS_CONTAINED = [
    ("narrow   sx=sy=0.55", [(1.0, 0.55, 0.55)]),
    ("medium   sx=sy=0.70", [(1.0, 0.70, 0.70)]),
    ("medium   sx=sy=0.85", [(1.0, 0.85, 0.85)]),
    ("ellipt.  sx=0.5 sy=0.8", [(1.0, 0.50, 0.80)]),
]
# Part B: PSFs with flux beyond the table -> quantifies truncation bias, a
#         property of the 33x33 data product shared by any interpolator.
#         Reported, not judged.
PSFS_TRUNCATED = [
    ("broad    sx=sy=1.20", [(1.0, 1.20, 1.20)]),
    ("ellipt.  sx=0.6 sy=1.1", [(1.0, 0.60, 1.10)]),
    ("core+5% halo (s=3px)", [(0.95, 0.80, 0.80), (0.05, 3.0, 3.0)]),
]

# Acceptance (code-correctness) criteria. These judge the *interpolation code*
# against an exact answer; they are not SPHEREx data-quality thresholds.
ACC_SHAPE = 0.02        # max |model - truth| / truth peak
ACC_NOISELESS_BIAS = 0.005
ACC_PULL_MEAN = 0.1
ACC_PULL_STD = (0.9, 1.1)


def _box(c, s, lo, hi):
    """Integral of a unit 1-D Gaussian (centre c, sigma s) over [lo, hi]."""
    r = np.sqrt(2.0) * s
    return 0.5 * (erf((hi - c) / r) - erf((lo - c) / r))


def pixel_integrals(comps, xc, yc):
    """Exact fraction of flux in a unit pixel centred at (xc, yc) for a source at 0."""
    out = 0.0
    for w, sx, sy in comps:
        out = out + w * _box(0.0, sx, xc - 0.5, xc + 0.5) * _box(0.0, sy, yc - 0.5, yc + 0.5)
    return out


def make_epsf_table(comps):
    off = (np.arange(NTAB) - (NTAB - 1) / 2) / OVS
    dy, dx = np.meshgrid(off, off, indexing="ij")
    return pixel_integrals(comps, dx, dy)  # array[y, x]; truncated at +-3.2 px


def truth_stamp(comps, x0, y0):
    yy, xx = np.mgrid[0:STAMP, 0:STAMP].astype(float)
    return pixel_integrals(comps, xx - x0, yy - y0)


def zone_for(table):
    psf = table / table.sum() * OVS**2  # same normalisation as load_epsf_zone()
    return EPSFZone(0, psf, OVS, 0, 0, np.nan, np.nan, 0, "parent")


def evaluate(comps, method, noise_trials=300):
    table = make_epsf_table(comps)
    zone = zone_for(table)
    c = (STAMP - 1) / 2
    yy, xx = np.mgrid[0:STAMP, 0:STAMP]
    omega = np.ones((STAMP, STAMP))
    shape_err, sums, bias = [], [], []
    for fx in PHASES:
        for fy in PHASES:
            x0, y0 = c + fx, c + fy
            T = truth_stamp(comps, x0, y0)
            M = sample_epsf(zone, (STAMP, STAMP), x0, y0, method=method)
            shape_err.append(np.max(np.abs(M - T)) / T.max())
            sums.append(M.sum())
            region = np.hypot(xx - x0, yy - y0) <= 5.0
            # noiseless: flux 1, background 0.01 per pixel, unit variance
            fit = forced_psf_fit(T + 0.01 * omega, np.ones_like(T), M, omega, region)
            bias.append(fit["A"] - 1.0)
    # noisy flux recovery at random phases (S/N ~ 50), truth-generated images
    pulls = []
    flux, bkg, sig = 1000.0, 5.0, 2.0
    for _ in range(noise_trials):
        x0, y0 = c + RNG.uniform(-0.5, 0.5), c + RNG.uniform(-0.5, 0.5)
        img = flux * truth_stamp(comps, x0, y0) + bkg + RNG.normal(0, sig, (STAMP, STAMP))
        M = sample_epsf(zone, (STAMP, STAMP), x0, y0, method=method)
        region = np.hypot(xx - x0, yy - y0) <= 5.0
        fit = forced_psf_fit(img, np.full_like(img, sig**2), M, omega, region)
        pulls.append((fit["A"] - flux) / fit["sA"])
    pulls = np.array(pulls)
    enclosed_table = table.sum() / OVS**2  # fraction of flux inside the 33x33 table
    return {
        "shape_err_max": float(np.max(shape_err)),
        "sum_min": float(np.min(sums)), "sum_max": float(np.max(sums)),
        "bias_noiseless_min": float(np.min(bias)), "bias_noiseless_max": float(np.max(bias)),
        "pull_mean": float(pulls.mean()), "pull_std": float(pulls.std(ddof=1)),
        "table_enclosed_fraction": float(enclosed_table),
    }


def axis_convention_check():
    """Elliptical PSF elongated in y: both evaluators must put the extent along y."""
    comps = [(1.0, 0.5, 1.4)]
    zone = zone_for(make_epsf_table(comps))
    c = (STAMP - 1) / 2
    out = {}
    for m in ("photutils", "linear"):
        M = sample_epsf(zone, (STAMP, STAMP), c, c, method=m)
        yy, xx = np.mgrid[0:STAMP, 0:STAMP]
        var_x = np.sum(M * (xx - c) ** 2) / M.sum()
        var_y = np.sum(M * (yy - c) ** 2) / M.sum()
        out[m] = var_y > 2 * var_x
    return out


def _row(name, method, r, verdict):
    return (f"{name:24s} {method:9s} {r['shape_err_max']:7.4f} "
            f"{r['sum_min']:7.4f}-{r['sum_max']:.4f} "
            f"{r['bias_noiseless_min']:+9.4f}..{r['bias_noiseless_max']:+.4f} "
            f"{r['pull_mean']:+9.3f} {r['pull_std']:8.3f} {r['table_enclosed_fraction']:8.4f}  {verdict}")


def main() -> int:
    print("ePSF VALIDATION - SYNTHETIC analytic PSFs, not SPHEREx data\n")
    print(f"ovs={OVS}, table {NTAB}x{NTAB} (+-{(NTAB - 1) / 2 / OVS:.1f} px), "
          f"{len(PHASES)}x{len(PHASES)} sub-pixel phases {PHASES.min():+.1f}..{PHASES.max():+.1f} px, "
          f"noisy trials at S/N~50\n")
    hdr = (f"{'PSF':24s} {'method':9s} {'shape':>7s} {'sum range':>15s} "
           f"{'bias(noiseless)':>19s} {'pull mean':>9s} {'pull std':>8s} {'tab.encl':>8s}  verdict")
    verdicts = {"photutils": True, "linear": True}

    print("PART A - interpolation accuracy (PSF contained in the table)")
    print(hdr)
    print("-" * len(hdr))
    for name, comps in PSFS_CONTAINED:
        for method in ("photutils", "linear"):
            r = evaluate(comps, method)
            ok = (r["shape_err_max"] <= ACC_SHAPE
                  and max(abs(r["bias_noiseless_min"]), abs(r["bias_noiseless_max"])) <= ACC_NOISELESS_BIAS
                  and abs(r["pull_mean"]) <= ACC_PULL_MEAN
                  and ACC_PULL_STD[0] <= r["pull_std"] <= ACC_PULL_STD[1])
            verdicts[method] &= ok
            print(_row(name, method, r, "PASS" if ok else "FAIL"))

    print("\nPART B - truncation of the 33x33 table (informational; same for any interpolator)")
    print(hdr)
    print("-" * len(hdr))
    for name, comps in PSFS_TRUNCATED:
        for method in ("photutils", "linear"):
            r = evaluate(comps, method)
            print(_row(name, method, r, "reported"))

    ax = axis_convention_check()
    print(f"\nAxis convention (y-elongated PSF stays y-elongated): "
          f"{ {k: bool(v) for k, v in ax.items()} }")
    for k in verdicts:
        verdicts[k] &= bool(ax[k])

    print(f"\nPart A acceptance: shape <= {ACC_SHAPE} of peak, |noiseless bias| <= {ACC_NOISELESS_BIAS}, "
          f"|pull mean| <= {ACC_PULL_MEAN}, pull std in {ACC_PULL_STD}")
    for k, v in verdicts.items():
        print(f"  {k:9s}: {'PASS' if v else 'FAIL'}")
    print("RESULT:", "photutils (primary method) PASSES" if verdicts["photutils"]
          else "photutils (primary method) FAILS")
    return 0 if verdicts["photutils"] else 1


if __name__ == "__main__":
    sys.exit(main())
