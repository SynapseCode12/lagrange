"""ePSF-truncation diagnostics (v0.3.0). DIAGNOSTIC ONLY.

Nothing here changes the primary photometry (method, 5 px fit radius and
ovs**2 normalisation are unchanged). The same functions are used on real
cutouts (measure.py) and in the synthetic study (study_truncation.py) so the
signatures seen in synthetic data can be compared directly with real data.

Background
----------
A SPHEREx EPSF zone is a 33x33 array at 5x oversampling: its support is
+-(33-1)/2/5 = +-3.2 native pixels (a square). Photutils ImagePSF returns 0
outside the array, so pixels of the 5 px fit region beyond the support are
modelled as background only. Any source light there is absorbed by the
background term B, which then also lowers the fitted flux A.

Quantities (all dimensionless or in the image's own units)
---------------------------------------------------------
n_fit_outside_support      fit-region pixels whose centre lies outside the
                           ePSF support (model value exactly 0 there)
n_fit_model_zero           fit-region pixels where the sampled ePSF is 0
                           (cross-check of the above)
epsf_native_sum            sum of the sampled ePSF over the whole cutout
flux_ratio_r<R>            A(R) / A(r_fit) for refits with radius R
bkg_shift_r<R>             B(R) - B(r_fit)  (image units, MJy/sr on real data)
err_ratio_r<R>             sigma_A(R) / sigma_A(r_fit)
resid_z_<a>-<b>            mean of (data - model)/sigma over valid pixels in
                           the annulus a < r <= b (primary model). Unmodelled
                           wing light shows as positive means just outside the
                           support and negative means where B is overestimated.
ring_excess_frac           sum(data - model) over support < r <= r_fit,
                           divided by A (flux unexplained by the model in the
                           part of the fit region the ePSF cannot reach)
"""

from __future__ import annotations

import numpy as np

from .photometry import forced_psf_fit

DIAG_RADII = (2.5, 3.0, 3.2, 4.0, 6.0, 8.0)


def annulus_edges(half_support: float, r_fit: float) -> list[float]:
    e = sorted({0.0, 1.5, 2.5, round(half_support, 2), 4.0, r_fit, r_fit + 1.0, r_fit + 2.0})
    return [v for v in e if v <= r_fit + 2.0]


def outside_support(shape, x0: float, y0: float, hsx: float, hsy: float) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    return (np.abs(xx - x0) > hsx) | (np.abs(yy - y0) > hsy)


def diagnostics(f, s2, P, omega, valid, r, x0, y0, hsx, hsy, fit, r_fit,
                radii=DIAG_RADII) -> dict:
    """Truncation diagnostics around a primary fit `fit` made with radius r_fit."""
    out: dict = {}
    region = (r <= r_fit) & valid
    outs = outside_support(f.shape, x0, y0, hsx, hsy)
    out["n_fit_pixels"] = int(region.sum())
    out["n_fit_outside_support"] = int((region & outs).sum())
    out["frac_fit_outside_support"] = out["n_fit_outside_support"] / max(out["n_fit_pixels"], 1)
    out["n_fit_model_zero"] = int((region & (P == 0)).sum())
    out["epsf_native_sum"] = float(P.sum())
    out["epsf_half_support_px"] = float(min(hsx, hsy))

    A0, B0, sA0 = fit["A"], fit["B"], fit["sA"]
    for R in radii:
        key = f"r{R:g}"
        reg = (r <= R) & valid
        try:
            fr = forced_psf_fit(f, s2, P, omega, reg)
            out[f"flux_ratio_{key}"] = fr["A"] / A0 if A0 != 0 else np.nan
            out[f"bkg_shift_{key}"] = fr["B"] - B0
            out[f"err_ratio_{key}"] = fr["sA"] / sA0 if sA0 > 0 else np.nan
        except Exception:  # noqa: BLE001 - too few pixels / ill-conditioned
            out[f"flux_ratio_{key}"] = out[f"bkg_shift_{key}"] = out[f"err_ratio_{key}"] = np.nan

    model = A0 * P + B0 * omega
    z = (f - model) / np.sqrt(s2)
    edges = annulus_edges(min(hsx, hsy), r_fit)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r > a) & (r <= b) & valid
        out[f"resid_z_{a:g}-{b:g}"] = float(np.mean(z[m])) if m.any() else np.nan
    ring = (r > min(hsx, hsy)) & (r <= r_fit) & valid
    out["ring_excess_frac"] = float(np.sum((f - model)[ring]) / A0) if A0 != 0 else np.nan
    # Model curve of growth (same radii as curve_of_growth): what the sampled,
    # truncated, renormalised ePSF predicts. data cog < model cog at a radius
    # means the star has more flux beyond that radius than the ePSF contains.
    tot = float(np.sum(P[r <= max(COG_RADII)]))
    for R in COG_RADII:
        out[f"cogm_r{R:g}"] = float(np.sum(P[r <= R]) / tot) if tot > 0 else np.nan
    return out


COG_RADII = (2.0, 3.0, 3.2, 4.0, 5.0, 6.0, 8.0)
COG_ANNULUS = (10.0, 13.0)


def curve_of_growth(img, var, omega, valid, r, x0, y0, radii=COG_RADII,
                    annulus=COG_ANNULUS) -> dict:
    """Model-independent encircled-flux curve: aperture flux(R) / flux(R_max).

    Background: sigma-clipped median of `annulus` (px). Returns NaN for a
    radius whose aperture contains masked pixels, or if the annulus does not
    fit in the cutout (use a larger --cutout-size-deg, e.g. 0.1 deg).
    cog_r<R> is directly comparable with the fraction of PSF flux inside R,
    the quantity that ePSF truncation and normalisation depend on. Noisy per
    epoch; meant to be stacked (median) over many observations of a bright star.
    """
    from astropy.stats import sigma_clipped_stats

    out = {f"cog_r{R:g}": np.nan for R in radii}
    ny, nx = img.shape
    ann = (r >= annulus[0]) & (r <= annulus[1]) & valid
    # the full annulus must lie inside the cutout
    if (x0 - annulus[1] < 0 or y0 - annulus[1] < 0 or x0 + annulus[1] > nx - 1
            or y0 + annulus[1] > ny - 1 or ann.sum() < 30):
        out["cog_note"] = "annulus outside cutout"
        return out
    _, med, _ = sigma_clipped_stats(img[ann], sigma=3.0)
    rmax = max(radii)
    if np.any((r <= rmax) & ~valid):
        out["cog_note"] = "masked pixel within largest aperture"
        return out
    w = var * omega**2
    big = r <= rmax
    b = float(np.sum((img[big] - med) * omega[big]))
    vb_total = float(np.sum(w[big]))
    if b <= 0:
        out["cog_note"] = "non-positive flux in largest aperture"
        return out
    for R in radii:
        ap = r <= R
        a_ = float(np.sum((img[ap] - med) * omega[ap]))
        va = float(np.sum(w[ap]))
        vc = vb_total - va  # ring R < r <= rmax, independent of the inner aperture
        ratio = a_ / b
        out[f"cog_r{R:g}"] = ratio
        # ratio = a / (a + c): var ~ [(1-ratio)^2 var_a + ratio^2 var_c] / b^2
        # (pixel noise only; background-level error not included)
        out[f"cog_err_r{R:g}"] = float(np.sqrt((1 - ratio) ** 2 * va + ratio**2 * vc) / b)
    out["cog_bkg"] = float(med)
    return out
