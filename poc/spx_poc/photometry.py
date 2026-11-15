"""Source photometry on a single SPHEREx Spectral Image cutout.

Inputs are the unmodified IMAGE, VARIANCE, FLAGS and ZODI arrays. Nothing is
written back to the cutout; all derived quantities are returned as numbers.

Units
-----
IMAGE is surface brightness in MJy/sr. To obtain a flux density we multiply
each pixel by its solid angle Omega_k (sr), computed here from the image's own
celestial WCS including SIP distortion (finite differences of the pixel
corners on the unit sphere). IRSA's spectrophotometry tool uses the official
Solid Angle Pixel Map calibration product instead; the WCS-derived value is an
approximation documented in SCIENTIFIC NOTES and checked against the nominal
6.15" pixel in the output (`omega_median_arcsec2`).

    f_k  [MJy]   = I_k * Omega_k
    s2_k [MJy^2] = VARIANCE_k * Omega_k^2
    1 MJy = 1e12 uJy

Background / zodiacal light
---------------------------
The ZODI layer is NOT subtracted. Both estimators fit or measure a *local*
constant surface-brightness background around the source. Zodiacal light is
smooth on arc-minute scales, so a local constant absorbs it together with any
other diffuse emission (and any error in the ZODI model). The ZODI median at
the target and the fitted background are both reported so the difference can
be inspected. This mirrors IRSA's spectrophotometry tool, which also fits a
local background and does not use ZODI.

Estimator 1: ePSF forced photometry (primary)
---------------------------------------------
Model in the fit region (r <= r_fit pixels, unmasked):
    f_k = A * P_k + B * Omega_k
P_k is the effective PSF of the nearest EPSF zone, shifted to the source's
sub-pixel position and sampled at pixel centres. A (flux, MJy) and B
(background, MJy/sr) are solved by weighted linear least squares with
weights 1/s2_k; their uncertainties come from the covariance matrix.
Position is held fixed at the catalogue position (forced photometry).

Estimator 2: circular aperture (cross-check)
--------------------------------------------
Sum of (I_k - B_ann) * Omega_k within r_ap, background from the sigma-clipped
median of an annulus. No aperture correction is applied, so this is suitable
for *relative* comparisons on the same detector only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates
from scipy.stats import chi2 as chi2_dist

MJY_TO_UJY = 1e12
SR_TO_ARCSEC2 = (180.0 / np.pi * 3600.0) ** 2


@dataclass
class PhotConfig:
    r_fit: float = 5.0        # px, ePSF fit region
    r_quality: float = 2.5    # px, core used for coverage & fit-quality metric
    r_ap: float = 3.0         # px, aperture radius
    r_in: float = 7.0         # px, annulus inner radius
    r_out: float = 12.0       # px, annulus outer radius
    min_core_valid: float = 0.8
    psf_method: str = "photutils"   # primary ePSF evaluation; "linear" also recorded


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def _unit_vectors(wcs: WCS, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ra, dec = wcs.all_pix2world(x, y, 0)
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def pixel_solid_angle(wcs: WCS, shape: tuple[int, int]) -> np.ndarray:
    """Per-pixel solid angle [sr] from the celestial WCS (SIP included)."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    dx = _unit_vectors(wcs, xx + 0.5, yy) - _unit_vectors(wcs, xx - 0.5, yy)
    dy = _unit_vectors(wcs, xx, yy + 0.5) - _unit_vectors(wcs, xx, yy - 0.5)
    return np.linalg.norm(np.cross(dx, dy), axis=-1)


# --------------------------------------------------------------------------
# ePSF
# --------------------------------------------------------------------------
@dataclass
class EPSFZone:
    index: int
    psf: np.ndarray          # oversampled 2-D array (normalised, see below)
    oversample: int
    xcenter: float
    ycenter: float
    cwave: float
    cband: float
    distance_px: float
    coord_frame: str         # "parent" or "cutout"
    edge_frac: float = float("nan")  # flux fraction in the table's outer ring (truncation diagnostic)
    raw_sum: float = float("nan")    # sum of the zone array exactly as stored in the file
    norm_factor: float = float("nan")  # ovs**2 / raw_sum, multiplied onto the array
    half_support_x: float = float("nan")  # native px from centre to the array edge
    half_support_y: float = float("nan")
    zones: dict | None = None        # raw-sum / edge-fraction statistics over ALL zones in the file


def _col(names, wanted):
    for n in names:
        if n.upper() == wanted:
            return n
    return None


def load_epsf_zone(hdul: fits.HDUList, x_parent: float, y_parent: float,
                   x_cut: float, y_cut: float) -> EPSFZone:
    """Pick the EPSF zone whose centre is nearest the target.

    IRSA's ePSF tutorial describes XCENTER/YCENTER as 0-indexed zone centres
    and selects the zone by distance to the parent-detector pixel. We still
    check the frame from the value range (zone centres spread over ~2040 px =>
    parent frame) and record it, rather than assume it.
    """
    if "EPSF" not in [h.name for h in hdul]:
        raise KeyError("no EPSF extension")
    hdu = hdul["EPSF"]
    tab, names = hdu.data, hdu.columns.names
    cx, cy = _col(names, "XCENTER"), _col(names, "YCENTER")
    cw, cb = _col(names, "CWAVE"), _col(names, "CBAND")
    if cx is None or cy is None:
        raise KeyError(f"EPSF table lacks XCENTER/YCENTER (columns: {names})")
    arr = None
    for n in names:
        a = np.asarray(tab[n])
        if a.ndim == 3 and a.shape[1] == a.shape[2] and a.shape[1] > 5:
            arr = a
            break
        if a.ndim == 2 and a.shape[1] > 25 and int(np.sqrt(a.shape[1])) ** 2 == a.shape[1]:
            s = int(np.sqrt(a.shape[1]))
            arr = a.reshape(-1, s, s)
            break
    if arr is None:
        raise KeyError(f"EPSF table: no square array column found (columns: {names})")

    xc = np.asarray(tab[cx], float)
    yc = np.asarray(tab[cy], float)
    frame = "parent" if (np.nanmax(xc) - np.nanmin(xc)) > 500 else "cutout"
    tx, ty = (x_parent, y_parent) if frame == "parent" else (x_cut, y_cut)
    d = np.hypot(xc - tx, yc - ty)
    i = int(np.nanargmin(d))
    ovs = int(hdu.header.get("OVSMPX", 5))
    ovs_y = int(hdu.header.get("OVSMPY", ovs))
    if ovs_y != ovs:
        raise ValueError(f"unequal oversampling OVSMPX={ovs} OVSMPY={ovs_y} not supported")
    raw = np.asarray(arr[i], float)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    total = raw.sum()
    if total <= 0:
        raise ValueError("EPSF zone has non-positive sum")
    # Current (v0.2.0) convention, unchanged: rescale so the oversampled array
    # sums to ovs^2, i.e. native-grid samples sum to ~1. If the stored array is
    # already normalised to the *untruncated* PSF (unknown), raw_sum < ovs^2
    # and this rescaling inflates the model by norm_factor, biasing fluxes
    # low by the same factor (see study_truncation.py).
    norm_factor = ovs**2 / total
    psf = raw / total * ovs**2  # same operation order as v0.2.0 (bit-identical results)
    ring = np.ones_like(psf, bool)
    ring[1:-1, 1:-1] = False
    edge_frac = float(raw[ring].sum() / total)
    # statistics over every zone in this file (cheap; shows zone-to-zone variation)
    all_raw = np.where(np.isfinite(arr), arr, 0.0).astype(float)
    sums = all_raw.reshape(all_raw.shape[0], -1).sum(1)
    rings = sums - all_raw[:, 1:-1, 1:-1].reshape(all_raw.shape[0], -1).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        edge_all = np.where(sums > 0, rings / sums, np.nan)
    zones = {"n": int(len(sums)), "rawsum_min": float(np.min(sums)),
             "rawsum_median": float(np.median(sums)), "rawsum_max": float(np.max(sums)),
             "edge_frac_median": float(np.nanmedian(edge_all)),
             "edge_frac_max": float(np.nanmax(edge_all))}
    ny, nx = psf.shape
    return EPSFZone(i, psf, ovs, float(xc[i]), float(yc[i]),
                    float(tab[cw][i]) if cw else np.nan,
                    float(tab[cb][i]) if cb else np.nan,
                    float(d[i]), frame, edge_frac, float(total), float(norm_factor),
                    (nx - 1) / 2 / ovs, (ny - 1) / 2 / ovs, zones)


PSF_METHODS = ("photutils", "linear")


def sample_epsf(zone: EPSFZone, shape: tuple[int, int], x0: float, y0: float,
                method: str = "photutils") -> np.ndarray:
    """ePSF value at each native pixel centre for a source at (x0, y0) (0-based).

    The SPHEREx EPSF is a pixel-convolved effective PSF (IRSA ePSF tutorial),
    so it is *evaluated* (interpolated) at pixel centres, not integrated over
    pixels. Conventions (same as photutils.psf.ImagePSF): array axis 0 = y,
    axis 1 = x; the PSF reference point is the middle element (n-1)/2; an
    oversampled array normalised to sum ovs**2 gives native-grid samples
    summing to ~1.

    method="photutils": photutils.psf.ImagePSF (cubic spline), the approach
                        used in IRSA's SPHEREx ePSF tutorial. Default.
    method="linear":    scipy map_coordinates order=1 (kept as a cross-check).
    """
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    if method == "photutils":
        from photutils.psf import ImagePSF
        model = ImagePSF(zone.psf, oversampling=zone.oversample, fill_value=0.0)
        return np.asarray(model.evaluate(xx, yy, 1.0, x0, y0), float).reshape(shape)
    if method == "linear":
        n = zone.psf.shape[0]
        c = (n - 1) / 2.0
        rows = c + (yy - y0) * zone.oversample
        cols = c + (xx - x0) * zone.oversample
        return map_coordinates(zone.psf, [rows, cols], order=1, mode="constant", cval=0.0)
    raise ValueError(f"unknown PSF method {method!r}")


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------
def forced_psf_fit(f: np.ndarray, s2: np.ndarray, P: np.ndarray, omega: np.ndarray,
                   region: np.ndarray) -> dict:
    """Weighted least squares for f = A*P + B*omega over `region`."""
    w = 1.0 / s2[region]
    # Columns differ by ~1e9 in scale (P ~ 0.1, Omega ~ 1e-9 sr); rescale them
    # to unit RMS before solving so the conditioning test is meaningful.
    X = np.column_stack([P[region], omega[region]])
    scale = np.sqrt(np.mean(X**2, axis=0))
    if np.any(scale == 0):
        raise np.linalg.LinAlgError("empty PSF or solid-angle column in fit region")
    Xs = X / scale
    y = f[region]
    XtW = Xs.T * w
    N = XtW @ Xs
    if np.linalg.cond(N) > 1e8:
        raise np.linalg.LinAlgError("ill-conditioned PSF fit")
    cov_s = np.linalg.inv(N)
    coef_s = cov_s @ (XtW @ y)
    A, B = coef_s / scale
    cov = cov_s / np.outer(scale, scale)
    resid = y - X @ np.array([A, B])
    chi2 = float(np.sum(resid**2 * w))
    dof = int(region.sum() - 2)
    # Probability of a chi2 at least this large if the model and VARIANCE are
    # correct. A diagnostic only; never used to reject an observation.
    pval = float(chi2_dist.sf(chi2, dof)) if dof > 0 else float("nan")
    return {"A": float(A), "B": float(B), "sA": float(np.sqrt(cov[0, 0])),
            "sB": float(np.sqrt(cov[1, 1])), "chi2": chi2, "dof": dof, "chi2_pvalue": pval}


def aperture_phot(img: np.ndarray, var: np.ndarray, omega: np.ndarray, valid: np.ndarray,
                  r: np.ndarray, cfg: PhotConfig) -> dict:
    ap = r <= cfg.r_ap
    ann = (r >= cfg.r_in) & (r <= cfg.r_out) & valid
    out = {"ap_flux_ujy": np.nan, "ap_err_ujy": np.nan, "ap_bkg_mjysr": np.nan,
           "ap_n_masked": int(np.sum(ap & ~valid)), "ap_n_annulus": int(ann.sum())}
    if ann.sum() < 20 or out["ap_n_masked"] > 0:
        return out  # do not interpolate over masked pixels in the aperture
    _, med, std = sigma_clipped_stats(img[ann], sigma=3.0)
    n_eff = max(int(ann.sum()), 1)
    flux = np.sum((img[ap] - med) * omega[ap])
    var_src = np.sum(var[ap] * omega[ap] ** 2)
    var_bkg = (np.sum(omega[ap]) * std / np.sqrt(n_eff)) ** 2
    out.update(ap_flux_ujy=float(flux * MJY_TO_UJY),
               ap_err_ujy=float(np.sqrt(var_src + var_bkg) * MJY_TO_UJY),
               ap_bkg_mjysr=float(med))
    return out
