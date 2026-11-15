"""Inspect one cached cutout: metadata, wavelength at target, quality, photometry.

The cutout file on disk is opened read-only and never modified, so the
original IMAGE, ZODI, VARIANCE and FLAGS data are preserved exactly as served
by IRSA.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from astropy.io import fits

from dataclasses import replace

from . import quality as q
from . import truncation
from .photometry import (MJY_TO_UJY, SR_TO_ARCSEC2, PhotConfig, aperture_phot,
                         forced_psf_fit, load_epsf_zone, pixel_solid_angle, sample_epsf)
from .spectral import CalSpectralWCS, _wcs, header_wavelength, target_pixel

log = logging.getLogger("spx_poc.measure")

TIME_KEYS = ("MJD-AVG", "MJD-BEG", "MJD-END", "DATE-AVG", "DATE-BEG", "XPOSURE",
             "TELAPSE", "TIMESYS", "OBSID", "DETECTOR", "BUNIT", "SPS_FLAGS",
             "NON_SURVEY")


def measure_cutout(path: Path, ra: float, dec: float, cfg: PhotConfig,
                   cal: dict[int, CalSpectralWCS] | None = None,
                   strict_flags: bool = False, detector: int | None = None,
                   ) -> dict:
    row: dict = {"cutout_file": Path(path).name, "usable": False, "reason": "",
                 "detector_expected": detector}
    try:
        with fits.open(path, memmap=False) as hdul:
            _measure(hdul, ra, dec, cfg, cal or {}, strict_flags, row)
    except Exception as exc:  # noqa: BLE001 - record and continue
        row["reason"] = row["reason"] or f"error: {type(exc).__name__}: {exc}"[:300]
        log.debug("measure failed for %s", path, exc_info=True)
    return row


def _measure(hdul, ra, dec, cfg, cal, strict_flags, row):
    names = [h.name for h in hdul]
    row["hdus"] = ",".join(names)
    for need in ("IMAGE", "VARIANCE", "FLAGS", "ZODI"):
        if need not in names:
            row["reason"] = f"missing HDU {need}"
            return

    hdr = hdul["IMAGE"].header
    for k in TIME_KEYS:
        row[k.lower().replace("-", "_")] = hdr.get(k)
    if row.get("detector") is None:
        row["detector"] = row.get("detector_expected")
    img = np.asarray(hdul["IMAGE"].data, float)
    var = np.asarray(hdul["VARIANCE"].data, float)
    flg = np.asarray(hdul["FLAGS"].data).astype(np.int64)
    zod = np.asarray(hdul["ZODI"].data, float)
    row["shape"] = f"{img.shape[0]}x{img.shape[1]}"
    if not (img.shape == var.shape == flg.shape == zod.shape):
        row["reason"] = "HDU shapes differ"
        return

    # --- position ---------------------------------------------------------
    tp = target_pixel(hdul, ra, dec)
    row.update(x=tp.x, y=tp.y, x_parent=tp.x_parent, y_parent=tp.y_parent,
               x_parent_awcs=tp.x_parent_awcs, y_parent_awcs=tp.y_parent_awcs,
               parent_dx_awcs_minus_irsa=tp.parent_dx, parent_dy_awcs_minus_irsa=tp.parent_dy,
               parent_method=tp.parent_method, parent_check=tp.parent_check,
               crpix1a=hdr.get("CRPIX1A"), crpix2a=hdr.get("CRPIX2A"),
               crval1a=hdr.get("CRVAL1A"), crval2a=hdr.get("CRVAL2A"),
               subpix_x=tp.x - round(tp.x), subpix_y=tp.y - round(tp.y))

    # --- wavelength at the target ----------------------------------------
    try:
        row["cwave_hdr_um"], row["cband_hdr_um"] = header_wavelength(hdul, tp.x, tp.y)
    except Exception as exc:  # noqa: BLE001
        row["cwave_hdr_um"] = row["cband_hdr_um"] = np.nan
        row["wave_hdr_error"] = str(exc)[:200]
    det = row.get("detector")
    calw = cal.get(int(det)) if det is not None else None
    if calw is not None:
        row["cwave_cal_um"], row["cband_cal_um"] = calw.at(tp.x_parent, tp.y_parent)
        row["cal_spectral_wcs_file"] = calw.path.name
    else:
        row["cwave_cal_um"] = row["cband_cal_um"] = np.nan
    if np.isfinite(row["cwave_cal_um"]):
        row["cwave_um"], row["cband_um"], row["wave_source"] = (
            row["cwave_cal_um"], row["cband_cal_um"], "calibration")
    else:
        row["cwave_um"], row["cband_um"], row["wave_source"] = (
            row["cwave_hdr_um"], row["cband_hdr_um"], "header_W")
    if np.isfinite(row["cwave_cal_um"]) and np.isfinite(row["cwave_hdr_um"]):
        row["cwave_cal_minus_hdr_nm"] = 1e3 * (row["cwave_cal_um"] - row["cwave_hdr_um"])

    # --- geometry & quality ----------------------------------------------
    ny, nx = img.shape
    if not (cfg.r_fit <= tp.x <= nx - 1 - cfg.r_fit and cfg.r_fit <= tp.y <= ny - 1 - cfg.r_fit):
        row["reason"] = "target too close to cutout/detector edge"
        return
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - tp.x, yy - tp.y)

    bits = q.parse_flag_bits(hdul["FLAGS"].header)
    row["flag_bits_found"] = len(bits)
    excl_mask, missing = q.mask_for(q.masked_names(strict_flags), bits)
    row["flag_names_missing"] = ",".join(missing)
    row["flag_names_record_only"] = ",".join(q.record_only_names(bits, strict_flags))
    bad = ((flg & excl_mask) != 0) | ~np.isfinite(img) | ~np.isfinite(var) | (var <= 0)
    valid = ~bad

    core = r <= cfg.r_quality
    row["core_valid_frac"] = float(valid[core].mean())
    ic, jc = int(round(tp.y)), int(round(tp.x))
    row["center_flags"] = ";".join(q.decode(flg[ic, jc], bits))
    row["core_flags"] = ";".join(q.decode(q.flags_union(flg, core), bits))
    row["zodi_median_mjysr"] = float(np.nanmedian(zod[r <= cfg.r_fit]))
    row["image_median_mjysr"] = float(np.nanmedian(img[r <= cfg.r_out]))

    sky = _wcs(hdr)
    omega = pixel_solid_angle(sky, img.shape)
    row["omega_median_arcsec2"] = float(np.median(omega) * SR_TO_ARCSEC2)

    if not valid[ic, jc]:
        row["reason"] = f"target pixel masked ({row['center_flags'] or 'non-finite'})"
        return
    if row["core_valid_frac"] < cfg.min_core_valid:
        row["reason"] = f"only {row['core_valid_frac']:.0%} of core pixels valid"
        # still try the measurement for diagnostics, but stays unusable

    # --- photometry -------------------------------------------------------
    f = img * omega
    s2 = var * omega**2
    try:
        zone = load_epsf_zone(hdul, tp.x_parent, tp.y_parent, tp.x, tp.y)
        row.update(epsf_zone=zone.index, epsf_zone_dist_px=zone.distance_px,
                   epsf_frame=zone.coord_frame, epsf_cwave_um=zone.cwave,
                   epsf_shape=f"{zone.psf.shape[0]}x{zone.psf.shape[1]}",
                   epsf_oversample=zone.oversample, epsf_edge_frac=zone.edge_frac,
                   epsf_raw_sum=zone.raw_sum, epsf_norm_factor=zone.norm_factor,
                   epsf_raw_sum_over_ovs2=zone.raw_sum / zone.oversample**2,
                   epsf_half_support_x_px=zone.half_support_x,
                   epsf_half_support_y_px=zone.half_support_y,
                   **{f"epsf_allzones_{k}": v for k, v in (zone.zones or {}).items()})
        region = (r <= cfg.r_fit) & valid
        qreg = core & valid
        # Primary method first, the other recorded as a cross-check.
        methods = [cfg.psf_method] + [m for m in ("photutils", "linear") if m != cfg.psf_method]
        for k, method in enumerate(methods):
            P = sample_epsf(zone, img.shape, tp.x, tp.y, method=method)
            fit = forced_psf_fit(f, s2, P, omega, region)
            if k == 0:
                model = fit["A"] * P + fit["B"] * omega
                row.update(
                    psf_method=method,
                    epsf_sum_in_fit=float(P[r <= cfg.r_fit].sum()),
                    psf_flux_ujy=fit["A"] * MJY_TO_UJY, psf_err_ujy=fit["sA"] * MJY_TO_UJY,
                    psf_bkg_mjysr=fit["B"], psf_bkg_err_mjysr=fit["sB"],
                    psf_chi2=fit["chi2"], psf_dof=fit["dof"],
                    psf_chi2_dof=fit["chi2"] / max(fit["dof"], 1),
                    psf_chi2_pvalue=fit["chi2_pvalue"], psf_npix=int(region.sum()),
                    # IRSA spectrophotometry fit-quality metric: mean |data-model|/sigma
                    # within 2.5 px (expected ~sqrt(2/pi)=0.80 for Gaussian noise).
                    fit_ql=float(np.mean(np.abs(f[qreg] - model[qreg]) / np.sqrt(s2[qreg]))),
                )
                row["psf_snr"] = row["psf_flux_ujy"] / row["psf_err_ujy"]
                row["bkg_minus_zodi_mjysr"] = fit["B"] - row["zodi_median_mjysr"]
                # v0.3.0 truncation diagnostics (do not alter the values above)
                try:
                    row.update({f"trunc_{k}": v for k, v in truncation.diagnostics(
                        f, s2, P, omega, valid, r, tp.x, tp.y, zone.half_support_x,
                        zone.half_support_y, fit, cfg.r_fit).items()})
                except Exception as exc:  # noqa: BLE001
                    row["trunc_error"] = f"{type(exc).__name__}: {exc}"[:200]
            else:
                row[f"psf_flux_{method}_ujy"] = fit["A"] * MJY_TO_UJY
                row[f"psf_flux_{method}_minus_primary_sigma"] = (
                    (row[f"psf_flux_{method}_ujy"] - row["psf_flux_ujy"]) / row["psf_err_ujy"])
    except Exception as exc:  # noqa: BLE001
        row["psf_error"] = f"{type(exc).__name__}: {exc}"[:200]

    row.update(aperture_phot(img, var, omega, valid, r, cfg))
    # Diagnostic: larger (r=5 px) aperture, model-independent enclosed flux.
    ap5 = aperture_phot(img, var, omega, valid, r, replace(cfg, r_ap=5.0))
    row.update(ap5_flux_ujy=ap5["ap_flux_ujy"], ap5_err_ujy=ap5["ap_err_ujy"],
               ap5_n_masked=ap5["ap_n_masked"])
    if np.isfinite(row.get("psf_flux_ujy", np.nan)) and np.isfinite(ap5["ap_flux_ujy"]) and ap5["ap_flux_ujy"] != 0:
        row["psf_over_ap5"] = row["psf_flux_ujy"] / ap5["ap_flux_ujy"]
    if np.isfinite(row.get("psf_flux_ujy", np.nan)) and np.isfinite(row.get("ap_flux_ujy", np.nan)) and row["ap_flux_ujy"] != 0:
        row["psf_over_ap3"] = row["psf_flux_ujy"] / row["ap_flux_ujy"]
    row.update(truncation.curve_of_growth(img, var, omega, valid, r, tp.x, tp.y))

    if not row["reason"]:
        if "psf_flux_ujy" not in row or not np.isfinite(row.get("psf_flux_ujy", np.nan)):
            row["reason"] = "PSF fit failed: " + row.get("psf_error", "unknown")
        elif not np.isfinite(row.get("cwave_um", np.nan)):
            row["reason"] = "no wavelength at target"
        else:
            row["usable"] = True
            row["reason"] = "ok"
