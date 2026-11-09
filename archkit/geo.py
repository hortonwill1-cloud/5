"""Geodesy: geocoding and the project-local coordinate frame.

The local frame is a transverse-Mercator projection centred on the site
reference point (scale factor 1).  Distances measured in it match ground
(ellipsoid) distances to better than 1 part in 10^6 within ~5 km, so
drawings produced from it are dimensionally faithful to the real world.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import requests

USER_AGENT = "archkit/0.1 (architectural site coordination tool)"
CACHE_DIR = Path(".cache/archkit")


def _cache_get(key: str):
    p = CACHE_DIR / (hashlib.sha1(key.encode()).hexdigest() + ".json")
    if p.exists():
        return json.loads(p.read_text())
    return None


def _cache_put(key: str, value) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / (hashlib.sha1(key.encode()).hexdigest() + ".json")
    p.write_text(json.dumps(value))


def http_json(url: str, *, params=None, data=None, timeout=90, retries=4):
    """GET/POST with on-disk cache and polite exponential backoff."""
    key = json.dumps([url, params, data], sort_keys=True)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    delay = 2.0
    last = None
    for _ in range(retries):
        try:
            if data is not None:
                r = requests.post(url, data=data, timeout=timeout,
                                  headers={"User-Agent": USER_AGENT})
            else:
                r = requests.get(url, params=params, timeout=timeout,
                                 headers={"User-Agent": USER_AGENT})
            if r.status_code in (429, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} from {url}")
            r.raise_for_status()
            out = r.json()
            _cache_put(key, out)
            return out
        except (requests.RequestException, ValueError) as e:  # noqa: PERF203
            last = e
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"request failed after {retries} tries: {url}: {last}")


def geocode(address: str) -> tuple[float, float, str]:
    """Address -> (lat, lon, display_name) via OSM Nominatim."""
    res = http_json("https://nominatim.openstreetmap.org/search",
                    params={"q": address, "format": "json", "limit": 1})
    if not res:
        raise ValueError(f"address not found: {address!r}")
    return float(res[0]["lat"]), float(res[0]["lon"]), res[0]["display_name"]


@dataclass
class LocalFrame:
    """Project-local metric frame centred on (lat0, lon0)."""

    lat0: float
    lon0: float

    def __post_init__(self):
        try:
            from pyproj import Transformer

            proj = (f"+proj=tmerc +lat_0={self.lat0} +lon_0={self.lon0} +k=1 "
                    "+x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs")
            self._fwd = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
            self._inv = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)
        except ImportError:  # pragma: no cover - fallback path
            self._fwd = self._inv = None

    # (lon, lat) order mirrors GeoJSON
    def to_local(self, lon: float, lat: float) -> tuple[float, float]:
        if self._fwd is not None:
            x, y = self._fwd.transform(lon, lat)
            return float(x), float(y)
        # Fallback: local equirectangular on WGS84 radii of curvature.
        a, e2 = 6378137.0, 6.69437999014e-3
        phi = math.radians(self.lat0)
        s = math.sin(phi)
        n = a / math.sqrt(1 - e2 * s * s)
        m = a * (1 - e2) / (1 - e2 * s * s) ** 1.5
        return (math.radians(lon - self.lon0) * n * math.cos(phi),
                math.radians(lat - self.lat0) * m)

    def to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        if self._inv is not None:
            lon, lat = self._inv.transform(x, y)
            return float(lon), float(lat)
        a, e2 = 6378137.0, 6.69437999014e-3
        phi = math.radians(self.lat0)
        s = math.sin(phi)
        n = a / math.sqrt(1 - e2 * s * s)
        m = a * (1 - e2) / (1 - e2 * s * s) ** 1.5
        return (self.lon0 + math.degrees(x / (n * math.cos(phi))),
                self.lat0 + math.degrees(y / m))

    def grid_convergence_deg(self) -> float:
        """Angle between grid north and true north at the origin (0 by construction)."""
        return 0.0

    def bbox_lonlat(self, radius_m: float) -> tuple[float, float, float, float]:
        corners = [self.to_lonlat(sx * radius_m, sy * radius_m)
                   for sx in (-1, 1) for sy in (-1, 1)]
        lons = [c[0] for c in corners]
        lats = [c[1] for c in corners]
        return min(lats), min(lons), max(lats), max(lons)  # S, W, N, E


def magnetic_declination_note() -> str:
    return ("North arrow shows TRUE north. For magnetic north, look up the "
            "declination for the site date at https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml")
