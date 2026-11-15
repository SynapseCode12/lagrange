"""Wavelength at the target position and wavelength matching.

Why this matters
----------------
Each SPHEREx detector sits behind a linear variable filter (LVF): the central
wavelength (CWAVE) and bandwidth (CBAND) change with pixel position. A sky
position therefore lands at a different wavelength in almost every exposure.
Comparing two exposures in time is only meaningful if the target was observed
at (nearly) the same wavelength, on the same detector.

Two wavelength sources are evaluated for every cutout:

1. Header spectral WCS ("W" alternate WCS, backed by the WCS-WAVE lookup table).
   IRSA: bilinear interpolation of the control grid is accurate to ~1 nm and
   intended for visualisation.
2. Calibration spectral WCS file (spectral_wcs_D<det>_*.fits, CWAVE/CBAND
   2040x2040 per-pixel maps). IRSA recommends these for science. Indexed with
   the target's 0-based parent-detector pixel from the "A" alternate WCS.

The calibration value is preferred when available; the header value is kept
as a cross-check and as the fallback (recorded in `wave_source`).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

log = logging.getLogger("spx_poc.spectral")


# Parent pixels from the two methods are expected to agree to numerical
# precision; anything larger indicates a header-convention problem.
PARENT_PIXEL_TOL_PX = 1e-3


@dataclass
class TargetPixel:
    x: float              # 0-based pixel in the cutout (SIP celestial WCS)
    y: float
    x_parent: float       # 0-based parent-detector pixel used downstream (IRSA formula)
    y_parent: float
    x_parent_awcs: float  # independent check: 'A' alternate WCS
    y_parent_awcs: float
    parent_method: str    # which method supplied x_parent/y_parent
    parent_check: str     # "agree" / "DISAGREE" / "unchecked: <reason>"

    @property
    def parent_dx(self) -> float:
        return self.x_parent_awcs - self.x_parent

    @property
    def parent_dy(self) -> float:
        return self.y_parent_awcs - self.y_parent


def _wcs(header, **kw) -> WCS:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        return WCS(header, **kw)


def parent_pixel_irsa(hdr, x: float, y: float) -> tuple[float, float]:
    """Official IRSA formula (SPHEREx PSF tutorial), 0-based in and out:
    x_parent = 1 + x_cutout - CRPIX1A,  y_parent = 1 + y_cutout - CRPIX2A."""
    return 1.0 + x - float(hdr["CRPIX1A"]), 1.0 + y - float(hdr["CRPIX2A"])


def parent_pixel_awcs(hdr, x: float, y: float) -> tuple[float, float]:
    """'A' alternate WCS ("0-based active pixel coordinates")."""
    wcs_a = _wcs(hdr, key="A")
    # Astropy may attach the primary SIP terms to lettered WCSes
    # (astropy#13105); the A WCS is a pure pixel mapping, so drop them.
    wcs_a.sip = None
    xa, ya = wcs_a.wcs_pix2world([[x, y]], 0)[0]
    return float(xa), float(ya)


def target_pixel(hdul: fits.HDUList, ra: float, dec: float) -> TargetPixel:
    """Locate (ra, dec) in the cutout and on the parent detector.

    The parent pixel is computed twice, independently: with IRSA's documented
    CRPIX1A/CRPIX2A formula (primary, as used in IRSA's ePSF zone selection)
    and with the 'A' alternate WCS. Their difference is recorded.
    """
    hdr = hdul["IMAGE"].header
    sky = _wcs(hdr)  # RA---TAN-SIP / DEC--TAN-SIP, SIP applied
    x, y = (float(v) for v in sky.all_world2pix([[ra, dec]], 0)[0])

    nan = float("nan")
    irsa = awcs = (nan, nan)
    notes = []
    if "CRPIX1A" in hdr and "CRPIX2A" in hdr:
        irsa = parent_pixel_irsa(hdr, x, y)
    else:
        notes.append("no CRPIX1A/CRPIX2A")
    if "CTYPE1A" in hdr:
        try:
            awcs = parent_pixel_awcs(hdr, x, y)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"A-WCS error {type(exc).__name__}")
    else:
        notes.append("no A WCS")

    if np.isfinite(irsa[0]):
        primary, method = irsa, "irsa_crpixA"
    elif np.isfinite(awcs[0]):
        primary, method = awcs, "a_wcs"
    else:
        raise KeyError("cannot determine parent-detector pixel: " + "; ".join(notes))

    if np.isfinite(irsa[0]) and np.isfinite(awcs[0]):
        d = max(abs(awcs[0] - irsa[0]), abs(awcs[1] - irsa[1]))
        check = "agree" if d <= PARENT_PIXEL_TOL_PX else "DISAGREE"
    else:
        check = "unchecked: " + "; ".join(notes)
    return TargetPixel(x, y, primary[0], primary[1], awcs[0], awcs[1], method, check)


def header_wavelength(hdul: fits.HDUList, x: float, y: float) -> tuple[float, float]:
    """(CWAVE, CBAND) in micron from the header 'W' spectral WCS at cutout pixel."""
    hdr = hdul["IMAGE"].header
    wcs_w = _wcs(hdr, fobj=hdul, key="W")
    wcs_w.sip = None  # wavelength mapping is independent of optical distortion
    vals = wcs_w.wcs_pix2world([[x, y]], 0)[0]
    units = [str(cu) for cu in wcs_w.wcs.cunit]
    out = []
    for v, unit in zip(vals[:2], units[:2]):
        unit = unit or "um"
        out.append(float((v * u.Unit(unit)).to_value(u.um)))
    return out[0], out[1]


class CalSpectralWCS:
    """Per-pixel CWAVE/CBAND maps from a calibration spectral_wcs file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with fits.open(self.path, memmap=False) as hdul:
            names = [h.name for h in hdul]
            if "CWAVE" not in names or "CBAND" not in names:
                raise KeyError(f"{self.path.name}: expected CWAVE and CBAND HDUs, found {names}")
            self.cwave = np.asarray(hdul["CWAVE"].data, dtype=np.float32)
            self.cband = np.asarray(hdul["CBAND"].data, dtype=np.float32)
            self.unit_w = hdul["CWAVE"].header.get("BUNIT", "um") or "um"
            self.unit_b = hdul["CBAND"].header.get("BUNIT", "um") or "um"
        self._fw = float(u.Unit(self.unit_w).to(u.um))
        self._fb = float(u.Unit(self.unit_b).to(u.um))
        log.info("Loaded calibration spectral WCS %s (shape %s, units %s/%s)",
                 self.path.name, self.cwave.shape, self.unit_w, self.unit_b)

    def at(self, x_parent: float, y_parent: float) -> tuple[float, float]:
        j, i = int(round(x_parent)), int(round(y_parent))
        ny, nx = self.cwave.shape
        if not (0 <= i < ny and 0 <= j < nx):
            return float("nan"), float("nan")
        return float(self.cwave[i, j]) * self._fw, float(self.cband[i, j]) * self._fb


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
def comparable(det1: int, w1: float, b1: float, det2: int, w2: float, b2: float,
               tol_frac: float) -> bool:
    """Pairwise criterion: same detector and |dCWAVE| <= tol_frac * min(CBAND)."""
    return det1 == det2 and abs(w1 - w2) <= tol_frac * min(b1, b2)


def group_by_wavelength(df: pd.DataFrame, tol_frac: float,
                        wave_col: str = "cwave_um", band_col: str = "cband_um",
                        det_col: str = "detector") -> pd.Series:
    """Assign wavelength-group ids within each detector.

    Greedy complete-linkage on wavelength-sorted rows: a candidate joins the
    current group only if it satisfies `comparable()` with EVERY existing
    member, i.e. CWAVE_c - CWAVE_m <= tol_frac * min(CBAND_c, CBAND_m) for all
    members m. Otherwise it seeds a new group. By construction every pair in
    a group satisfies the documented pairwise criterion. The partition is
    deterministic but not unique (a different seed order could group
    differently). Rows without a wavelength get group id "".
    """
    gid = pd.Series("", index=df.index, dtype=object)
    ok = df[wave_col].notna() & df[band_col].notna()
    sub_all = df[ok].sort_values([det_col, wave_col], kind="mergesort")
    for det, sub in sub_all.groupby(det_col, sort=True):
        k = -1
        mem_w = np.empty(0)
        mem_b = np.empty(0)
        for idx, w, b in zip(sub.index, sub[wave_col].to_numpy(float), sub[band_col].to_numpy(float)):
            fits_all = mem_w.size > 0 and np.all(np.abs(w - mem_w) <= tol_frac * np.minimum(b, mem_b))
            if not fits_all:
                k += 1
                mem_w, mem_b = np.array([w]), np.array([b])
            else:
                mem_w, mem_b = np.append(mem_w, w), np.append(mem_b, b)
            gid[idx] = f"D{int(det)}-G{k:03d}"
    return gid


def pairwise_violations(df: pd.DataFrame, tol_frac: float, group_col: str = "wave_group",
                        wave_col: str = "cwave_um", band_col: str = "cband_um",
                        det_col: str = "detector") -> int:
    """Count member pairs that violate `comparable()` (should always be 0)."""
    n = 0
    for gid, g in df[df[group_col].astype(str) != ""].groupby(group_col):
        w = g[wave_col].to_numpy(float)
        b = g[band_col].to_numpy(float)
        d = g[det_col].to_numpy()
        dw = np.abs(w[:, None] - w[None, :])
        tol = tol_frac * np.minimum(b[:, None], b[None, :])
        bad = (dw > tol * (1 + 1e-12)) | (d[:, None] != d[None, :])
        n += int(np.triu(bad, 1).sum())
    return n


def count_seasons(mjds: np.ndarray, gap_days: float = 30.0) -> int:
    """Number of clusters of epochs separated by gaps longer than gap_days."""
    m = np.sort(np.asarray(mjds, dtype=float))
    m = m[np.isfinite(m)]
    if m.size == 0:
        return 0
    return int(1 + np.sum(np.diff(m) > gap_days))
