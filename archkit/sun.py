"""Solar position (NOAA General Solar Position algorithm, ~0.01 deg accuracy 1950-2050).

Used to light renders with the true sun for a site/date/time and to produce
shadow studies (e.g. Mar 21 / Jun 21 / Dec 21 at 9am, noon, 3pm).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone


def _julian_day(dt: datetime) -> float:
    dt = dt.astimezone(timezone.utc)
    y, m = dt.year, dt.month
    d = dt.day + (dt.hour + (dt.minute + dt.second / 60) / 60) / 24
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5


def solar_position(lat: float, lon: float, when: datetime) -> tuple[float, float]:
    """Return (azimuth_deg clockwise from true north, elevation_deg incl. refraction)."""
    if when.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    jd = _julian_day(when)
    t = (jd - 2451545.0) / 36525.0
    l0 = (280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360
    mean_anom = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    ecc = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    mr = math.radians(mean_anom)
    ctr = (math.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t))
           + math.sin(2 * mr) * (0.019993 - 0.000101 * t) + math.sin(3 * mr) * 0.000289)
    true_long = l0 + ctr
    omega = 125.04 - 1934.136 * t
    app_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    eps0 = 23 + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60) / 60
    eps = eps0 + 0.00256 * math.cos(math.radians(omega))
    decl = math.degrees(math.asin(math.sin(math.radians(eps)) * math.sin(math.radians(app_long))))
    yv = math.tan(math.radians(eps / 2)) ** 2
    l0r = math.radians(l0)
    eqtime = 4 * math.degrees(yv * math.sin(2 * l0r) - 2 * ecc * math.sin(mr)
                              + 4 * ecc * yv * math.sin(mr) * math.cos(2 * l0r)
                              - 0.5 * yv * yv * math.sin(4 * l0r) - 1.25 * ecc * ecc * math.sin(2 * mr))
    utc = when.astimezone(timezone.utc)
    minutes = utc.hour * 60 + utc.minute + utc.second / 60
    tst = (minutes + eqtime + 4 * lon) % 1440
    ha = tst / 4 - 180
    if ha < -180:
        ha += 360
    latr, declr, har = map(math.radians, (lat, decl, ha))
    cz = math.sin(latr) * math.sin(declr) + math.cos(latr) * math.cos(declr) * math.cos(har)
    cz = max(-1.0, min(1.0, cz))
    zen = math.degrees(math.acos(cz))
    elev = 90 - zen
    # atmospheric refraction (NOAA)
    if elev > 85:
        refr = 0.0
    elif elev > 5:
        te = math.tan(math.radians(elev))
        refr = 58.1 / te - 0.07 / te ** 3 + 0.000086 / te ** 5
    elif elev > -0.575:
        refr = 1735 + elev * (-518.2 + elev * (103.4 + elev * (-12.79 + elev * 0.711)))
    else:
        refr = -20.774 / math.tan(math.radians(elev))
    elev += refr / 3600
    sz = math.sin(math.radians(zen))
    if abs(sz) < 1e-9:
        az = 180.0
    else:
        caz = (math.sin(latr) * cz - math.sin(declr)) / (math.cos(latr) * sz)
        caz = max(-1.0, min(1.0, caz))
        az = math.degrees(math.acos(caz))
        az = (az + 180) % 360 if ha > 0 else (540 - az) % 360
    return az, elev


def sun_vector(az_deg: float, el_deg: float) -> tuple[float, float, float]:
    """Unit vector pointing TO the sun in site frame (+x east, +y north, +z up)."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    return (math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e))


def parse_when(s: str, utc_offset_h: float) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone(timedelta(hours=utc_offset_h)))
    return dt


def guess_utc_offset(lon: float) -> float:
    """Crude solar-time offset; specify `utc_offset_h` in the spec for accuracy (DST!)."""
    return round(lon / 15)
