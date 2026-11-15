"""Access to the IRSA SPHEREx archive.

Everything here talks to real IRSA services. Nothing in this module produces
synthetic data: if a request fails after its retries, the error is raised (or
recorded by the caller) rather than replaced with placeholder values.

Documented interfaces used (see README for links):
  * SIA2 via astroquery: Irsa.query_sia(pos=(coord, radius), collection=...)
      collections: spherex_dr1, spherex_dr1_deep, spherex_qr3, spherex_qr3_deep
  * TAP (ADQL) tables spherex.artifact / spherex.plane, as used in IRSA's
    cutout and PSF tutorials (columns a.uri, a.planeid, p.planeid, p.poly,
    p.energy_bandpassname, p.time_bounds_lower).
  * IBE cutout service: append ?center=RA,Dec&size=SIZE_DEG to an image URL.
  * Browsable calibration directories under
    https://irsa.ipac.caltech.edu/ibe/data/spherex/<release>/spectral_wcs/
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("spx_poc.archive")

IRSA_BASE = "https://irsa.ipac.caltech.edu/"
IBE_SPHEREX = "https://irsa.ipac.caltech.edu/ibe/data/spherex/"

# Only the current releases. QR2 is deliberately excluded (project decision:
# calibrations differ and DR1/QR3 supersede it).
RELEASE_COLLECTIONS: dict[str, list[str]] = {
    "dr1": ["spherex_dr1", "spherex_dr1_deep"],
    "qr3": ["spherex_qr3", "spherex_qr3_deep"],
}

# level2_2025W18_2B_0001_1D1_spx_l2b-v12-2025-164.fits
FILENAME_RE = re.compile(
    r"level2_(?P<obsid>.+?)D(?P<det>\d)_spx_l2b-v(?P<ver>\d+)-(?P<pdate>\d{4}-\d{3})"
)
RELEASE_IN_PATH_RE = re.compile(r"/spherex/(?P<rel>[a-z]+\d*)/")

T = TypeVar("T")


@dataclass
class NetConfig:
    timeout_s: float = 120.0
    retries: int = 4
    backoff_s: float = 2.0


# --------------------------------------------------------------------------
# Retry helpers
# --------------------------------------------------------------------------
def with_retries(fn: Callable[[], T], what: str, net: NetConfig) -> T:
    """Call fn(), retrying with exponential backoff. Re-raises the last error."""
    last: Exception | None = None
    for attempt in range(1, net.retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise after retries
            last = exc
            if attempt == net.retries:
                break
            delay = net.backoff_s * 2 ** (attempt - 1)
            log.warning("%s failed (attempt %d/%d): %s -- retrying in %.0f s",
                        what, attempt, net.retries, _short(exc), delay)
            time.sleep(delay)
    assert last is not None
    log.error("%s failed after %d attempts: %s", what, net.retries, _short(last))
    raise last


def _short(exc: Exception, n: int = 300) -> str:
    s = f"{type(exc).__name__}: {exc}"
    return s if len(s) <= n else s[:n] + "..."


def make_session(net: NetConfig) -> requests.Session:
    """requests.Session with transport-level retries on transient HTTP errors."""
    retry = Retry(
        total=net.retries,
        connect=net.retries,
        read=net.retries,
        backoff_factor=net.backoff_s,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    s = requests.Session()
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers["User-Agent"] = "ProjectLagrange-PoC/0.1 (astropy; research use)"
    return s


def configure_astropy_timeouts(net: NetConfig) -> None:
    # SPHEREx files can be slow to serve; IRSA's own tutorials raise this limit.
    from astropy.utils.data import conf
    conf.remote_timeout = net.timeout_s


# --------------------------------------------------------------------------
# Metadata queries
# --------------------------------------------------------------------------
def list_spherex_collections(net: NetConfig) -> list[str]:
    from astroquery.ipac.irsa import Irsa

    tab = with_retries(lambda: Irsa.list_collections(servicetype="SIA", filter="spherex"),
                       "list SIA collections", net)
    col = "collection" if "collection" in tab.colnames else tab.colnames[0]
    return sorted(str(v) for v in tab[col])


def query_sia(ra: float, dec: float, radius_arcsec: float, collection: str,
              net: NetConfig, maxrec: int = 200_000) -> pd.DataFrame:
    """SIA2 query for one collection. Returns a DataFrame (possibly empty)."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.ipac.irsa import Irsa

    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    tab = with_retries(
        lambda: Irsa.query_sia(pos=(coord, radius_arcsec * u.arcsec),
                               collection=collection, maxrec=maxrec),
        f"SIA query {collection} @ ({ra:.5f},{dec:+.5f})", net)
    if len(tab) >= maxrec:
        log.warning("SIA result for %s hit maxrec=%d -- result may be truncated",
                    collection, maxrec)
    keep = [c for c in (
        "obs_id", "obs_collection", "t_min", "t_max", "t_exptime",
        "energy_bandpassname", "em_min", "em_max", "access_url", "access_format",
        "access_estsize", "s_ra", "s_dec", "dataproduct_subtype", "calib_level",
        "obs_release_date", "cloud_access",
    ) if c in tab.colnames]
    df = tab[keep].to_pandas() if len(tab) else pd.DataFrame(columns=keep)
    for c in df.columns:  # decode any bytes columns
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.decode() if isinstance(v, bytes) else v)
    df["queried_collection"] = collection
    df["sia_columns_available"] = ",".join(tab.colnames)
    return df


def tap_query(adql: str, net: NetConfig, what: str) -> pd.DataFrame:
    from astroquery.ipac.irsa import Irsa

    def run():
        try:
            return Irsa.query_tap(adql).to_table()
        except Exception as exc:  # fall back to an async job for slow queries
            log.info("%s: sync TAP failed (%s); trying async", what, _short(exc, 120))
            return Irsa.query_tap(adql, async_job=True).to_table()

    tab = with_retries(run, what, net)
    df = tab.to_pandas()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.decode() if isinstance(v, bytes) else v)
    return df


def tap_spherex_schema(net: NetConfig) -> pd.DataFrame:
    adql = ("SELECT table_name, column_name, datatype, unit, description "
            "FROM TAP_SCHEMA.columns "
            "WHERE table_name IN ('spherex.plane', 'spherex.artifact') "
            "ORDER BY table_name, column_name")
    return tap_query(adql, net, "TAP schema for spherex.plane/artifact")


def tap_images_at(ra: float, dec: float, net: NetConfig) -> pd.DataFrame:
    """All SPHEREx images whose footprint contains (ra, dec), via TAP.

    The release is not a documented plane column, so it is derived from the
    artifact URI path (.../spherex/<release>/level2/...).
    """
    adql = (
        "SELECT a.uri, p.energy_bandpassname, p.time_bounds_lower "
        "FROM spherex.artifact a JOIN spherex.plane p ON a.planeid = p.planeid "
        f"WHERE 1 = CONTAINS(POINT('ICRS', {ra:.7f}, {dec:.7f}), p.poly)"
    )
    df = tap_query(adql, net, f"TAP image search @ ({ra:.5f},{dec:+.5f})")
    df["release"] = df["uri"].map(release_from_path)
    return df


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------
def release_from_path(path: str) -> str:
    m = RELEASE_IN_PATH_RE.search(str(path))
    return m.group("rel") if m else "unknown"


def parse_filename(url: str) -> dict:
    m = FILENAME_RE.search(str(url))
    if not m:
        return {"file_obsid": None, "detector": None, "pipeline_version": None,
                "processing_date": None}
    return {"file_obsid": m.group("obsid"), "detector": int(m.group("det")),
            "pipeline_version": int(m.group("ver")),
            "processing_date": m.group("pdate")}


def detector_from_bandpass(name) -> int | None:
    m = re.search(r"D(\d)", str(name or ""))
    return int(m.group(1)) if m else None


def normalise_sia(df: pd.DataFrame, release: str) -> pd.DataFrame:
    """Add release, detector, survey type, and filename-derived fields."""
    if df.empty:
        return df
    out = df.copy()
    out["release"] = release
    out["survey"] = np.where(out["queried_collection"].str.endswith("_deep"), "deep", "wide")
    parsed = out["access_url"].map(parse_filename).apply(pd.Series)
    out = pd.concat([out, parsed], axis=1)
    det_bp = out.get("energy_bandpassname", pd.Series(index=out.index, dtype=object)).map(
        detector_from_bandpass)
    out["detector"] = out["detector"].fillna(det_bp)
    out["detector_mismatch"] = (det_bp.notna() & out["detector"].notna()
                                & (det_bp != out["detector"]))
    return out


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop the same exposure+detector appearing in more than one release.

    Preference: DR1 over QR3 (DR1 is the recommended reprocessing for surveys
    1-2). Returns (kept, dropped).
    """
    if df.empty:
        return df, df
    pref = {"dr1": 0, "qr3": 1}
    d = df.copy()
    d["_key_obs"] = d["file_obsid"].fillna(d.get("obs_id"))
    d["_pref"] = d["release"].map(pref).fillna(9)
    d = d.sort_values(["_key_obs", "detector", "_pref"])
    dup = d.duplicated(subset=["_key_obs", "detector"], keep="first")
    return (d[~dup].drop(columns=["_pref"]), d[dup].drop(columns=["_pref"]))


# --------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------
def cutout_url(access_url: str, ra: float, dec: float, size_deg: float) -> str:
    sep = "&" if "?" in access_url else "?"
    return f"{access_url}{sep}center={ra:.7f},{dec:.7f}&size={size_deg:g}"


def cutout_cache_path(cache_dir: Path, access_url: str, ra: float, dec: float,
                      size_deg: float) -> Path:
    stem = Path(access_url.split("?")[0]).stem
    tag = hashlib.sha1(f"{ra:.7f},{dec:.7f},{size_deg:g}".encode()).hexdigest()[:10]
    return cache_dir / f"{stem}__cut_{tag}.fits"


def download_file(session: requests.Session, url: str, dest: Path, net: NetConfig,
                  validate_fits: bool = True) -> Path:
    """Download url -> dest atomically. Reuses a valid cached file.

    Raises on failure; never writes a partial file under the final name.
    """
    from astropy.io import fits

    if dest.exists() and dest.stat().st_size > 0:
        if not validate_fits or _is_valid_fits(dest):
            return dest
        log.warning("Cached file is invalid, re-downloading: %s", dest.name)
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def fetch():
        with session.get(url, stream=True, timeout=(30, net.timeout_s)) as r:
            if r.status_code == 404:
                raise FileNotFoundError(f"HTTP 404 for {url}")
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
        if validate_fits:
            with fits.open(tmp) as hdul:
                names = [h.name for h in hdul]
            if "IMAGE" not in names and not dest.name.startswith("spectral_wcs"):
                raise ValueError(f"Downloaded FITS lacks IMAGE HDU (HDUs: {names})")
        tmp.replace(dest)
        return dest

    try:
        return with_retries(fetch, f"download {dest.name}", net)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _is_valid_fits(path: Path) -> bool:
    from astropy.io import fits
    try:
        with fits.open(path) as hdul:
            _ = [h.name for h in hdul]
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------
# Calibration: spectral WCS (CWAVE / CBAND) file discovery
# --------------------------------------------------------------------------
_HREF_RE = re.compile(r'href="([^"?#]+)"', re.IGNORECASE)


def find_spectral_wcs_url(session: requests.Session, release: str, detector: int,
                          net: NetConfig, max_depth: int = 3) -> str | None:
    """Search the browsable IBE directory for spectral_wcs_D<det>_*.fits.

    The directory layout below spectral_wcs/ is not documented in detail, so
    this walks HTML index pages (breadth-first, bounded depth) and picks the
    highest version / latest date. Returns None if nothing is found.
    """
    pattern = re.compile(
        rf"spectral_wcs_D{detector}_spx_cal-wcs-v(\d+)-(\d{{4}}-\d{{3}})\.fits$")
    root = urljoin(IBE_SPHEREX, f"{release}/spectral_wcs/")
    frontier, seen, hits = [(root, 0)], set(), []
    while frontier:
        url, depth = frontier.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            r = session.get(url, timeout=(30, net.timeout_s))
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            log.info("Could not list %s: %s", url, _short(exc, 150))
            continue
        for href in _HREF_RE.findall(r.text):
            full = urljoin(url, href)
            if not full.startswith(root):
                continue  # parent links etc.
            m = pattern.search(full)
            if m:
                hits.append((int(m.group(1)), m.group(2), full))
            elif full.endswith("/") and depth < max_depth:
                frontier.append((full, depth + 1))
    if not hits:
        return None
    hits.sort()
    log.info("Spectral WCS candidates for D%d in %s: %s", detector, release,
             ", ".join(Path(h[2]).name for h in hits))
    return hits[-1][2]
