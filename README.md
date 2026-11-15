[README.md](https://github.com/user-attachments/files/34867484/README.md)
# Project Lagrange: Milestone 1 SPHEREx proof of concept

This is a self-contained Python proof of concept. It answers one question
with real archive data:

> For a given sky position, how many SPHEREx observations taken at
> *matched wavelengths* actually exist, and can we measure a source's
> brightness reliably across them?

It queries the IRSA SPHEREx archive (**DR1 and QR3 only**) for two regions:

| Region | Role | Centre (ICRS, deg) |
|---|---|---|
| North Ecliptic Pole (NEP) deep field | primary | 270.0, +66.5607 |
| M101 field | control | 210.80227, +54.34895 |

For each region the script does the following:

1. Counts every observation that contains the position (SIA2), broken down by release, collection, detector, pipeline version and date.
2. Picks an isolated, unsaturated-looking 2MASS star near the region as the photometry target. For M101 the star lies outside the galaxy disk.
3. Cross-checks the coverage with an independent TAP query.
4. Downloads small cutouts (0.05°, about 29 × 29 px) for the chosen detector(s). It never downloads the full 70 MB files.
5. For each cutout, finds the wavelength (CWAVE, CBAND) actually sampled at the star, applies the quality flags, and measures the star's flux with ePSF forced photometry plus an aperture-photometry cross-check, using the VARIANCE layer for uncertainties.
6. Groups the observations into wavelength-matched sets, counts the usable epochs, and plots brightness against time.

**No data is ever fabricated.** If IRSA can't be reached, the run records the failure in
`summary.md` / `summary.json` and exits with code 2.

---

## 1. Windows setup (PowerShell)

Requirements: Windows 10/11, **Python 3.11, 3.12 or 3.13** (64-bit, from
python.org), about 1 GB of free disk space, and internet access to
`irsa.ipac.caltech.edu`.

```powershell
# 1. Check Python (the "py" launcher comes with the python.org installer)
py -0p                      # lists installed Pythons; need 3.11+ (64-bit)

# 2. Go to the poc folder
cd C:\path\to\lagrange\poc

# 3. Create and activate a virtual environment
py -3.12 -m venv .venv      # or -3.11 / -3.13
.\.venv\Scripts\Activate.ps1
#   If PowerShell blocks the script, run once:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
#   (Command Prompt alternative: .venv\Scripts\activate.bat)

# 4. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Run, in this order

### Step A: offline checks (about 15 s, no network)

```powershell
python validate_epsf.py
python selftest_synthetic.py
python study_truncation.py      # v0.3.0: synthetic ePSF-truncation study (~5 s)
```

**Both use synthetic data and only test the code.**

- `validate_epsf.py` checks the ePSF evaluation against an *analytic truth*
  (exact pixel integrals of known PSFs) at 121 sub-pixel positions. It compares
  Photutils `ImagePSF` (the primary method) with the older linear
  interpolation, and reports how much flux is lost to truncation of the
  33 × 33 table. Expected last line:
  `RESULT: photutils (primary method) PASSES`. The linear method is
  *expected* to show FAIL; that result is why it is no longer the primary method.
- `study_truncation.py` writes `study_output\truncation_study.csv`,
  `truncation_signatures_r5_renorm25.csv` and `truncation_study.png`: flux
  and background bias against fit radius for compact and extended-wing
  analytic PSFs under two normalization hypotheses, plus the diagnostic
  signatures to compare with real data (section 5.7).
- `selftest_synthetic.py` checks flag policy, the parent-pixel cross-check,
  wavelength grouping, fit-quality classification, and end-to-end flux
  recovery on analytic-truth images. Expected last line:
  `55/55 checks passed (synthetic data only).`

### Step B: quick smoke test against the real archive (a few minutes)

```powershell
python inspect_spherex.py --regions m101 --max-cutouts 10
```

This confirms that the archive is reachable and the real files parse. Check
that the last log line says `m101=COMPLETED`.

### Step C: full proof-of-concept run

```powershell
python inspect_spherex.py
```

The defaults are both regions, DR1+QR3, detector D2 (1.10–1.62 µm), and up to 150 cutouts per region.
Expect roughly 15–60 minutes depending on your connection and IRSA load. The
cache is reused, so re-running is fast.

Optional additional runs (send these too, if you have time):

```powershell
# Another wavelength regime (D4: 2.42-3.82 um)
python inspect_spherex.py --detectors 4

# Stricter flag policy, to see how much the usable count drops
python inspect_spherex.py --strict-flags
```

**ePSF-truncation diagnostic runs (v0.3.0, M101 first).** The curve-of-growth
diagnostic needs an annulus at 10–13 px, so these runs use 0.1° cutouts.
Primary photometry is unchanged; only the cutout is larger.

```powershell
# 1. Default target star, larger cutouts
python inspect_spherex.py --regions m101 --cutout-size-deg 0.1

# 2. Brighter star (higher S/N, so the stacked diagnostics are more sensitive).
#    Check OVERFLOW/NONLINEAR in core_flags: if present, the star is too bright.
python inspect_spherex.py --regions m101 --cutout-size-deg 0.1 --kmag 9.5 11
```

## 3. What to send back

At the end, the script prints a path like:

```
Send back this file for review: C:\...\poc\output\review_bundle_YYYYMMDD_HHMMSS.zip
```

Please send:

1. **That `review_bundle_*.zip`** for each run. It contains `summary.md`,
   `summary.json`, `run_log.txt`, all CSV tables, and all PNG plots.
2. The **console output of `validate_epsf.py`, `selftest_synthetic.py` and
   `study_truncation.py`** (plus `study_output\truncation_study.png`).
3. If anything failed: the full console output of that run.

Do **not** send the `cache\` folder. It holds the raw FITS cutouts
(hundreds of MB), and the bundle already references every file by name.

---

## 4. Outputs (per run: `output\run_<timestamp>\`)

| File | Content |
|---|---|
| `summary.md` / `summary.json` | Everything below, aggregated per region |
| `run_log.txt` | Full debug log including every retry and error |
| `tap_schema_spherex.csv` | Actual columns of `spherex.plane` / `spherex.artifact` |
| `<region>_center_observations.csv` | Every DR1/QR3 observation containing the region centre |
| `<region>_target_observations.csv` | Same, at the photometry target star |
| `<region>_tap_images.csv` | Independent TAP query result (cross-check) |
| `<region>_measurements.csv` | One row per processed cutout: times, pixel positions, CWAVE/CBAND (header and calibration), flags, ePSF and aperture fluxes, usability and reason |
| `<region>_wavelength_groups.csv` | Wavelength-matched groups: span, usable epochs, seasons, constancy χ² |
| `<region>_observation_timeline.png` | All observations vs time per detector |
| `<region>_wavelength_vs_time.png` | Wavelength sampled at the star vs time (shows the LVF effect) |
| `<region>_lightcurve_matched.png` | **Brightness vs time within matched-wavelength groups** |
| `<region>_raw_chronological_EXPLORATORY.png` | All usable fluxes in time order, colored by wavelength. This mixes wavelengths and is *not* a light curve. |
| `<region>_stamps.png` | Original IMAGE cutouts of the best group, with no zodi subtraction |
| `<region>_truncation_diagnostics.png` | v0.3.0: stacked residual profile, refit flux vs radius, curve of growth (data vs ePSF model), stored ePSF normalization |

The raw cutouts are kept in `cache\cutouts\` **exactly as served by IRSA**. They
are opened read-only and never modified.

---

## 5. Method and assumptions

### 5.1 Observations counted

- SIA2 collections `spherex_dr1`, `spherex_dr1_deep`, `spherex_qr3` and
  `spherex_qr3_deep`, queried with a 1″ radius, so the image footprint must
  contain the position.
- Rows with identical URLs are dropped. The same exposure+detector appearing
  in both DR1 and QR3 is counted once, preferring DR1, and the number removed
  is reported.
- QR2 is excluded by design.
- The TAP query (`spherex.artifact` joined to `spherex.plane`, filtered with
  `CONTAINS`) is an independent count. Small differences are possible
  because IRSA documents that new data can reach SIA about a day after it
  reaches TAP.

### 5.2 Wavelength at the target (requirement A)

SPHEREx detectors sit behind linear variable filters, so CWAVE and CBAND
depend on the pixel. For each cutout:

1. The star's pixel is found with the SIP-corrected celestial WCS. It is then
   converted to the **parent-detector pixel** (0-based, full 2040 × 2040 array)
   in two independent ways:
   - **Primary:** IRSA's documented formula from its SPHEREx PSF tutorial,
     `x_parent = 1 + x_cutout − CRPIX1A` and `y_parent = 1 + y_cutout − CRPIX2A`.
   - **Check:** the `A` alternate WCS ("0-based active pixel coordinates").
   - Their difference is recorded in `parent_dx_awcs_minus_irsa` and
     `parent_dy_awcs_minus_irsa`. `parent_check` is `agree` when both are within
     0.001 px (numerical tolerance) and `DISAGREE` otherwise. The summary reports
     the counts and the largest difference. A disagreement is flagged, not
     rejected, because it points to a header-convention problem we need to
     understand rather than bad data.
2. **Calibration value (preferred):** CWAVE/CBAND are read at that parent pixel
   from the official `spectral_wcs_D<n>_spx_cal-wcs-*.fits` calibration file,
   which IRSA recommends for science. The script finds the file by walking
   IRSA's browsable directory `.../spherex/dr1/spectral_wcs/` and downloads it
   once (about 32 MB per detector). Use `--no-cal-wcs` to skip this.
3. **Header value (cross-check and fallback):** the `W` spectral WCS
   (WCS-WAVE lookup table) is evaluated at the cutout pixel. IRSA quotes about
   1 nm accuracy for it. The difference from the calibration value is reported
   (`cwave_cal_minus_hdr_nm`). The column `wave_source` records which value was
   used.

**Comparability criterion.** Two observations are treated as comparable in
wavelength when both of these hold:

- they come from the **same detector** (same LVF, PSF family and calibration),
- **|CWAVE₁ − CWAVE₂| ≤ f × min(CBAND₁, CBAND₂)**, with f = 0.25 by default
  (`--tol-frac`).

Why a quarter bandwidth: CBAND is the channel width, λ/R (about 30 nm at
1.3 µm). A shift of 0.25 × CBAND moves the bandpass by about λ/(4R), roughly
0.6% in wavelength at R ≈ 41. For a smooth continuum, a source with spectral
slope d ln F/d ln λ = 2 would then change by about 1.2%. That is below the
precision expected for faint stars, but comparable to it for bright ones. It
is **not** safe for sources with narrow spectral features (emission lines,
ice or PAH bands) near the bandpass edge. For those, f should be reduced.
Every group reports its actual CWAVE span, so the effect can be judged
afterwards.

Grouping is greedy, deterministic complete-linkage. Within each detector,
the observations are sorted by CWAVE. A candidate joins the current group
only if it satisfies the pairwise criterion with **every** existing member,
using **min(CBAND)** of each pair; otherwise it starts a new group. Every
pair in a group therefore satisfies the documented criterion. An independent
validator (`pairwise_violations`) re-checks all pairs on every run, and the
summary reports `group_pairwise_violations`, which must be 0. The partition
is not unique; a different seed order could split groups differently.
Pipeline versions and releases in each group are listed so that mixing can
be spotted. (Version 0.1.0 compared candidates only with the seed's CBAND,
which could violate the criterion when CBAND varies; this is fixed and
tested.)

### 5.3 Zodiacal light and data preservation (requirement B)

- The IMAGE, ZODI, VARIANCE and FLAGS layers are never modified, and the
  cached cutouts are the original IRSA files.
- **ZODI is not subtracted.** Both photometry methods estimate a *local*
  constant background around the star. Zodiacal light is smooth on
  arc-minute scales, so the local background absorbs it together with other
  diffuse emission and any error in the zodi model. This is the same
  approach as IRSA's Spectrophotometry Tool, which fits a local background
  and does not use ZODI.
- The **ZODI median at the star** and the **fitted background** are both
  reported (`zodi_median_mjysr`, `psf_bkg_mjysr`, `bkg_minus_zodi_mjysr`). This
  lets us check later whether subtracting the model would be justified for
  difference imaging.
- The stamp plot shows the original IMAGE.

### 5.4 Photometry (requirement C)

- **Source-centred region:** a 0.05° cutout centred on the star, served by
  IRSA's cutout service.
- **Units:** IMAGE is in MJy/sr. Each pixel is multiplied by its solid angle
  Ω, computed from the SIP-corrected WCS, giving MJy, and then converted to µJy.
  *Approximation:* IRSA's tool uses the official Solid Angle Pixel Map
  instead. The median Ω is reported and should be close to 6.15″², about
  37.8 arcsec².
- **PSF:** the cutout's EPSF table (5× oversampled effective PSFs on a grid of
  zones) is used. As in IRSA's ePSF tutorial, the zone whose 0-indexed
  `XCENTER`/`YCENTER` is nearest the star's parent-detector pixel is chosen.
  - IRSA describes the EPSF as a *pixel-convolved* effective PSF, so it is
    **evaluated (interpolated) at native pixel centres**, not integrated over
    pixels. This is how both IRSA's tutorial code and Photutils `ImagePSF`
    treat an ePSF.
  - **Primary method:** `photutils.psf.ImagePSF` (cubic spline), the approach in
    IRSA's ePSF tutorial. The table is normalised to sum to `ovs²`, so the
    native-pixel samples sum to about 1.
  - **Cross-check:** the original linear interpolation is still computed. Its
    flux is recorded in `psf_flux_linear_ujy`, and its difference from the
    primary, in units of σ, in `psf_flux_linear_minus_primary_sigma`.
  - **Validation (`validate_epsf.py`, synthetic):** against analytic truth, for
    PSFs contained in the table, `ImagePSF` matches to within 0.01–0.12% of
    the peak, with < 0.06% flux bias at every sub-pixel phase and calibrated
    pulls. Linear interpolation showed phase-dependent biases of up to 1.6%
    (+1.9σ at S/N ≈ 50), which is why it was replaced.
  - **Truncation:** the table spans ±3.2 px. Flux beyond that cannot be
    modelled by any interpolator. In the synthetic tests a PSF with 1.5% of
    its flux outside the table gave −1.7% to −2.3% bias depending on
    sub-pixel phase, which would add scatter between epochs. On real data,
    `epsf_edge_frac` (the fraction of the zone's ePSF in its outermost ring) is
    reported as a diagnostic: values near 0 mean truncation is negligible.
- **Brightness:** weighted linear least squares of *flux = A·PSF + B·Ω* over
  the pixels within 5 px. A is the source flux and B the local background.
  Position is fixed at the 2MASS position (forced photometry).
- **Uncertainty:** the per-pixel weights are 1/(VARIANCE·Ω²). The flux error
  comes from the fit covariance.
- **Cross-check:** aperture photometry (r = 3 px, local annulus background),
  with **no aperture correction**. It is only for relative comparison.
- **Brightness vs time:** `<region>_lightcurve_matched.png`, one panel per
  wavelength group with at least 2 usable epochs. Each panel shows the weighted
  mean and χ²/dof of a constant-flux model. A field star is *expected* to
  be roughly constant, which makes it a test of the photometry, not a
  variability claim.

### 5.5 Fit quality (warning only)

These metrics are reported for every fit. None of them changes `usable`.

- `psf_chi2_dof` and `psf_chi2_pvalue`: the χ² of the PSF+background fit and
  its p-value under the χ² distribution with the fit's degrees of freedom.
  If both the ePSF and VARIANCE are correct, p-values are uniform between 0 and 1.
- `fit_ql`: IRSA's spectrophotometry fit-quality metric, the mean of
  |data − model| / σ within 2.5 px. For pure Gaussian noise it is expected
  to be about √(2/π) ≈ 0.80.
- `fit_quality` is `ok`, `warning` or `n/a`. It is set to `warning` when either:
  - `psf_chi2_pvalue < --fitq-pvalue` (default 1e-3), or
  - `psf_chi2_dof` or `fit_ql` is more than `--fitq-mad` (default 5) robust σ
    (scaled MAD) above the median of usable observations on the same
    detector in the same run.
  The reasons are listed in `fit_quality_notes`.
- **These thresholds are provisional.** They are conventional statistical
  levels used only to surface cases for inspection, not to reject data.
  The summary reports the full distributions (5th–95th percentiles) of
  χ²/dof, p-value and `fit_ql`, so rejection criteria can be set from real
  data. Light-curve plots circle warned points in red. Each group also
  reports `n_fit_warning` and a constancy χ² that uses only fits marked `ok`.

### 5.6 Quality flags (usable observations)

Flag names are read from each file's own FLAGS header (`MSKNnnnn`), never
from hard-coded bit numbers.

- **Masked pixels:** these mirror IRSA's Spectrophotometry Tool exclusions:
  SUR_ERROR, NONFUNC, MISSING_DATA, HOT, COLD, NONLINEAR, PERSIST, BLOOM,
  SNOWBALL, HALO and SATELLITE_HALO.
- **Strict-only flags:** these are recorded as warnings by default.
  `--strict-flags` additionally masks exactly: TRANSIENT, OVERFLOW, OUTLIER, CROSSTALK, GHOST, STREAK, DICHROIC
  - DICHROIC marks low-throughput pixels near the dichroic edge (bands 3
    and 4 only). The flux there is calibrated, but with reduced S/N and a steep
    throughput gradient, so it belongs with the "use with caution" flags.
    Version 0.1.0 omitted it from this list by oversight, and the README
    wrongly implied `--strict-flags` masked it.
- **Recorded only, never masked:** every other flag name found in the file's
  FLAGS header, for example SOURCE (expected on a star), FULLSAMPLE, PHANMISS,
  and any name not known when this was written. Those names are listed per
  run in `flag_names_seen_record_only`.
- The active policy is written to `summary.json` under `flag_policy`. A
  self-test checks that the strict list in this README matches the code.
- An observation is **usable** when the download succeeded, the star is at least 5 px
  from the cutout edge, the star's pixel is not masked, at least 80% of the pixels within
  2.5 px are valid, the PSF fit succeeded, and a wavelength is known.
  Every rejection is recorded with its reason.

### 5.7 ePSF truncation diagnostics (v0.3.0, descriptive only)

**The primary photometry is unchanged.** It still uses Photutils `ImagePSF`, a
5 px fit radius, and an array rescaled to sum 25. On identical inputs every
v0.2.0 output value is bit-identical (3,522 values checked). The columns
below are additions only.

**Geometry.** A 33 × 33 ePSF at 5× has a square support of ±3.2 px. `ImagePSF`
returns 0 beyond it. Roughly a third of the 5 px fit region (about 27–41 of
about 76–81 pixels, depending on sub-pixel phase) is therefore modelled as
background only.

| Column(s) | Meaning |
|---|---|
| `epsf_raw_sum`, `epsf_raw_sum_over_ovs2` | Sum of the zone array **as stored**, and that sum / 25 |
| `epsf_norm_factor` | 25 / raw sum: the factor the current code multiplies onto the array |
| `epsf_edge_frac` | Fraction of the stored array's flux in its outermost ring |
| `epsf_allzones_*` | Min/median/max raw sum and edge fraction over **all** zones in the file |
| `epsf_half_support_x_px` / `_y_px` | (n−1)/2/ovs |
| `trunc_n_fit_pixels`, `trunc_n_fit_outside_support`, `trunc_frac_fit_outside_support` | Fit pixels in total, and those whose centre is beyond the support |
| `trunc_n_fit_model_zero` | Cross-check: fit pixels where the sampled model is exactly 0 |
| `trunc_flux_ratio_r<R>`, `trunc_bkg_shift_r<R>`, `trunc_err_ratio_r<R>` | Refit at R = 2.5, 3, 3.2, 4, 6, 8 px, relative to the 5 px primary |
| `trunc_resid_z_<a>-<b>` | Mean (data − model)/σ in annuli around the star |
| `trunc_ring_excess_frac` | Σ(data − model) between the support and 5 px, divided by the fitted flux |
| `cog_r<R>`, `cog_err_r<R>` | Model-independent curve of growth: aperture flux(R)/flux(8 px), annulus 10–13 px (needs `--cutout-size-deg 0.1`) |
| `trunc_cogm_r<R>` | The same ratio for the sampled ePSF model |
| `ap5_flux_ujy`, `psf_over_ap5`, `psf_over_ap3` | Larger-aperture cross-check |
| `subpix_x`, `subpix_y` | Sub-pixel phase of the star |

`summary.json` → `measurements.truncation_diagnostics` holds distributions, the
stacked residual profile with standard errors, data-versus-model curve of
growth (mean, median, standard errors from scatter and from propagation), and
`phase_dependence`. That last item is the Spearman correlation of
within-group flux residuals with `trunc_n_fit_outside_support` and with the
sub-pixel offset.

**What the synthetic study establishes** (`study_truncation.py`; analytic
PSFs, noiseless; these are properties of the method, not of SPHEREx):

1. *Normalizing a truncated array to 25 biases flux low by the missing
   fraction.* If a fraction *m* of the PSF lies outside the support, the
   renormalized model is T/(1−m). A fit confined to the support then returns
   exactly (1−m)·F, so the bias is −m. This holds to float precision for every
   test PSF (for example 2.38% missing gives −2.38%, and 6.63% gives −6.63%).
   The bias is multiplicative and specific to each zone, so it cancels between
   epochs only when they use zones with the same *m*.
2. If instead the archive normalized the PSF **before** truncation (so the
   stored sum is 25·(1−m)), using the array *as stored* gives no bias within
   the support. Which convention SPHEREx uses is not documented in the
   sources reviewed. The real `epsf_raw_sum_over_ovs2` will narrow this down.
3. Extending the fit beyond the support (the current 5 px) adds a further
   bias. The background term absorbs wing light: 0.1–0.8% extra at 5 px
   for the test PSFs, with 0.8–4.6% of the source flux absorbed into the
   background over the fit area.
4. That extra term **depends on sub-pixel phase** (up to about 0.6% peak-to-peak
   at 5 px, and 0 for fits within the support when the ePSF is otherwise perfect). It would add scatter
   between epochs that is not in the error bars.
5. Fitting only within the support (3.2 px) removes items 3–4 but raises the
   statistical error by 11–25% (background-limited).
6. **Refitting at other radii cannot detect the main bias in item 1**: flux
   ratios move by less than 1% even when 7% of the flux is missing. Detecting it
   needs a model-independent measurement (the curve of growth) or the stored
   normalization. The curve of growth is noisy: at S/N ≈ 33 a single epoch
   has about 10% error at 3.2 px. It is only informative for bright stars
   stacked over many epochs.

**What needs real SPHEREx data:** the actual missing fraction *m* and how it
varies between zones and wavelengths; the archive's normalization
convention; whether errors in the real ePSF itself dominate over
truncation; contamination by neighbours and background structure; and
whether a usable star reaches the S/N needed for the stacked diagnostics.

**What to inspect in the first M101 run, before changing anything:**

| # | Inspect | Interpretation (provisional) |
|---|---|---|
| 1 | `epsf_raw_sum_over_ovs2`, `epsf_allzones_rawsum_min/max`, `epsf_raw_sum_relative_std_across_observed_zones` | Equal to 1 for every zone, to float precision: the archive normalized after truncation (or wings are negligible), so renormalizing is a no-op and the table says nothing about *m*. Below 1 and varying between zones: consistent with normalization before truncation, so the current renormalization biases each zone by (raw/25 − 1) and adds that zone-to-zone spread between epochs. Any other constant (for example 1): a different convention, to check with IRSA documentation. |
| 2 | `epsf_edge_frac`, `epsf_allzones_edge_frac_max` | Small values are *necessary but not sufficient*. The study's halo PSFs had edge fractions of only 0.2–0.3% while missing 2.4–7.6% of flux. |
| 3 | `curve_of_growth_data_vs_model` at 3.2 and 5 px, with `data_minus_model_sem_*` | Data significantly below the model means the star has flux beyond the support. The size of the gap indicates *m*, as a lower bound, because flux beyond 8 px is lost in the normalization. Only meaningful if the standard error is ≲ 1%. |
| 4 | `stacked_resid_z_by_annulus_px` and its plot | Unmodelled wings show negative residuals at 1.5–3.2 px and positive at 3.2–5 px (study, section 4 output). Compare the shape with the study's signature table, not the amplitude, which scales with S/N. |
| 5 | `phase_dependence` | A significant correlation of flux residuals with `n_fit_outside_support` or sub-pixel phase means the phase-dependent term (item 4 above) matters at our precision. |
| 6 | `flux_ratio_vs_radius` | Expected within about 1%. Larger or erratic changes point to something other than truncation: a neighbour, background structure, or position or PSF mismatch. |
| 7 | `fit_quality` distributions, `psf_over_ap5` | χ²/dof and `fit_ql` above the Gaussian expectation, together with items 3–4, strengthen the case. On their own they are not specific to truncation. |
| 8 | Manual comparison with IRSA's Spectrophotometry Tool for the same star and epochs | The only absolute cross-check available to us. |

Options to *consider* after that run (not implemented as defaults):
(a) a fit radius equal to the support, if items 4–5 show phase-dependent
effects; (b) using the stored normalization, if item 1 indicates
normalization before truncation; (c) an aperture correction from the
stacked curve of growth, if *m* is significant and consistent across zones.

### 5.8 What the counts mean

- `n_usable` per group = **usable matched-wavelength epochs**.
- `n_seasons_gap30d` = clusters of epochs separated by more than 30 days. The
  all-sky survey revisits a given channel about every 6 months, while the deep
  fields are revisited far more often.
- If a detector has more observations than `--max-cutouts`, an **evenly
  time-spaced subset** is processed and the summary says `SAMPLED`. Raise
  `--max-cutouts` to process all of them.

---

## 6. Known limitations and unverified assumptions

These are written down deliberately. The real run is what confirms or
refutes them.

1. **EPSF details.** IRSA's ePSF tutorial gives `XCENTER`/`YCENTER` as 0-indexed
   zone centres on the parent detector. The script still checks the frame
   from the value range (`epsf_frame`). The (y, x) axis order and centring
   follow the Photutils `ImagePSF` convention and are verified against
   analytic truth with an elongated PSF, but not yet against a real SPHEREx
   star. The χ² p-value and `fit_ql` distributions from the real run are the
   test of that.
2. **ePSF truncation and normalization.** The 33 × 33 array at 5× covers
   ±3.2 px. With the current renormalization to 25, the flux is biased low by
   the PSF's missing fraction, and the 5 px fit adds a further component that
   depends on sub-pixel phase. Both effects are quantified synthetically in
   section 5.7. Their size for SPHEREx is unknown until the diagnostics are
   inspected on real data.
3. **Not yet scientifically validated.** Everything above is validated only
   against synthetic data. The ePSF photometry counts as validated for
   SPHEREx only after the real-data checks hold: χ² p-values roughly
   uniform, `fit_ql` near 0.8, the field star constant within errors in
   matched groups, and agreement with IRSA's Spectrophotometry Tool for
   the same star.
4. **Pixel solid angle** comes from the WCS, not the official map.
5. **Header `W` spectral WCS.** The FITS `-TAB` lookup is read with astropy.
   It is exercised only on real data, since the self-test can't synthesise it.
6. **Calibration file discovery.** The layout below `spectral_wcs/` is not
   documented, so the directory walk may fail. The fallback is the header
   value, and the run reports which was used.
7. **Target selection.** 2MASS positions are from epoch ~2000, so proper
   motion is ignored (normally ≪ 6″ pixels). The SPHEREx saturation level is
   not documented in the sources reviewed; Ks 11–13 was chosen to stay well
   clear of it, and OVERFLOW/NONLINEAR flags near the star are reported.
8. **Cutout size.** If cutouts include the full EPSF table, each is about 2 MB.
   The defaults then need up to about 0.6 GB of cache.

## 7. Troubleshooting

| Symptom | What to do |
|---|---|
| `ARCHIVE_UNREACHABLE` or repeated timeouts | Check that https://irsa.ipac.caltech.edu opens in a browser; try `--timeout 300 --workers 2`; try again later (IRSA maintenance). |
| `NO_TARGET` | The automatic 2MASS selection found no isolated star. Pass one: `--nep-target RA DEC` or `--m101-target RA DEC`. |
| Corporate proxy / SSL errors | Set `HTTPS_PROXY` in PowerShell (`$env:HTTPS_PROXY="http://proxy:port"`). Do not disable certificate checks. |
| Need a clean start | Delete `output\` (results). Deleting `cache\` forces all downloads again. |

## 8. Sources

- IRSA SPHEREx data access: https://caltech-ipac.github.io/spherex-archive-documentation/spherex-data-access
- IRSA SPHEREx data products: https://caltech-ipac.github.io/spherex-archive-documentation/spherex-data-products
- IRSA SPHEREx data releases: https://caltech-ipac.github.io/spherex-archive-documentation/spherex-data-releases
- SPHEREx Explanatory Supplement: https://irsa.ipac.caltech.edu/data/SPHEREx/docs/SPHEREx_Expsupp_QR.pdf
- IRSA tutorials: spherex-intro, spherex-cutouts, spherex-psf at https://caltech-ipac.github.io/irsa-tutorials/
- IRSA SPHEREx Spectrophotometry Tool (flag policy, local background, fit_ql): https://irsa.ipac.caltech.edu/onlinehelp/spherex/spherex/sp.html
- IRSA SPHEREx ePSF tutorial (ImagePSF, 0-indexed zone centres, CRPIXnA formula): https://caltech-ipac.github.io/irsa-tutorials/spherex-psf/
- Photutils `ImagePSF` (ePSF evaluated, not integrated): https://photutils.readthedocs.io/

## 9. Change log

**v0.3.0** (ePSF truncation investigation; primary photometry unchanged)
1. Each fit now records the stored ePSF array sum, the normalization factor
   applied, the edge-ring fraction, the support size, and all-zone statistics.
2. The number of fit-region pixels outside the ePSF support is recorded, with a
   model-zero cross-check.
3. New diagnostics: refits at 2.5–8 px, residual annuli, ring excess,
   model-independent curve of growth with uncertainties against the model
   curve, r = 5 px aperture, sub-pixel phase, and a phase-dependence
   correlation. These come with a summary block and a four-panel plot.
4. `study_truncation.py`: a synthetic study (compact, halo and Moffat PSFs; two
   normalization hypotheses; seven fit radii; sub-pixel phases).
5. Tests: 13 new self-test checks (55 total), including the −(missing
   fraction) bias law. A regression run shows all v0.2.0 outputs are
   bit-identical.

**v0.2.0** (after code review)
1. Wavelength grouping now enforces the pairwise criterion with min(CBAND) against
   every group member (complete linkage). There is a run-time validator
   (`group_pairwise_violations`, must be 0) and new tests with varying CBAND.
2. ePSF evaluation now uses Photutils `ImagePSF` (IRSA tutorial approach) as the
   primary method, with linear interpolation kept as a recorded cross-check.
   `validate_epsf.py` tests both against analytic truth (independent of the
   code under test). Self-test images are now generated from analytic truth.
   A truncation diagnostic (`epsf_edge_frac`) was added.
3. Added the χ² p-value and a provisional, warning-only fit-quality
   classification (`fit_quality`, `fit_quality_notes`), plus summary
   distributions of χ²/dof, p-value and `fit_ql`.
4. DICHROIC added to the `--strict-flags` set. The README now lists the
   exact masked sets, and a test keeps the README and code in sync.
5. The parent-detector pixel now comes from IRSA's CRPIXnA formula, independently
   cross-checked against the A-WCS, with the difference and agree/DISAGREE
   recorded.

The archive query and download architecture is unchanged.
