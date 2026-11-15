"""FLAGS extension handling.

Bit definitions are read from each file's own FLAGS header (MSKNnnnn = name,
MSKDnnnn = description, nnnn = bit number), as shown in IRSA's introductory
SPHEREx tutorial. Nothing is hard-coded by bit number.

Masking policy (per pixel)
--------------------------
Pixels carrying any EXCLUDE flag are removed from the photometric fit. This
list mirrors the flags that IRSA's own SPHEREx Spectrophotometry Tool ignores:
SUR_ERROR, NONFUNC, MISSING_DATA, HOT, COLD, NONLINEAR, PERSIST, BLOOM,
SNOWBALL, HALO, SATELLITE_HALO.

WARN_STRICT flags are recorded as warnings by default and masked only with
`--strict-flags`: TRANSIENT, OVERFLOW, OUTLIER, CROSSTALK, GHOST, STREAK,
DICHROIC.

DICHROIC marks low-throughput pixels near the dichroic edge in bands 3 and 4
(Explanatory Supplement flag table; >=50% attenuation per the dichroic
calibration product). The flux there is calibrated but at reduced S/N and
with a steep throughput gradient, so it belongs with the other "use with
caution" flags. (It was omitted from WARN_STRICT in v0.1.0 by oversight.)

Every other flag name found in a file (e.g. SOURCE, FULLSAMPLE, PHANMISS,
GHOST_FPA, GHOST_EXT, or names not known when this was written) is RECORD_ONLY:
reported near the target, never masked. SOURCE (known-source mask) is
expected on a real star.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits

EXCLUDE = ("SUR_ERROR", "NONFUNC", "MISSING_DATA", "HOT", "COLD", "NONLINEAR",
           "PERSIST", "BLOOM", "SNOWBALL", "HALO", "SATELLITE_HALO")
WARN_STRICT = ("TRANSIENT", "OVERFLOW", "OUTLIER", "CROSSTALK", "GHOST", "STREAK",
               "DICHROIC")
NEVER_REJECT = ("SOURCE",)


def masked_names(strict: bool) -> tuple[str, ...]:
    """Exactly the flag names whose pixels are excluded from the fit."""
    return EXCLUDE + (WARN_STRICT if strict else ())


def policy(strict: bool) -> dict:
    return {"masked": list(masked_names(strict)),
            "recorded_only": "all other flag names in each file's FLAGS header"
                             + ("" if strict else f" (incl. {', '.join(WARN_STRICT)})")}


def record_only_names(bits: dict[str, int], strict: bool) -> list[str]:
    m = set(masked_names(strict))
    return sorted(n for n in bits if n not in m)


def parse_flag_bits(header: fits.Header) -> dict[str, int]:
    bits: dict[str, int] = {}
    for key in header:
        if key.startswith("MSKN") and key[4:].isdigit():
            bits[str(header[key]).strip()] = int(key[4:])
    return bits


def mask_for(names, bits: dict[str, int]) -> tuple[int, list[str]]:
    """Bitmask for the given flag names, plus names absent from this file."""
    m, missing = 0, []
    for n in names:
        if n in bits:
            m |= 1 << bits[n]
        else:
            missing.append(n)
    return m, missing


def decode(value: int, bits: dict[str, int]) -> list[str]:
    value = int(value)
    return sorted(n for n, b in bits.items() if value & (1 << b))


def flags_union(flags: np.ndarray, region: np.ndarray) -> int:
    vals = flags[region].astype(np.int64)
    return int(np.bitwise_or.reduce(vals)) if vals.size else 0
