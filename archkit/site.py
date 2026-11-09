"""Real-world site context: existing buildings, roads, terrain, parcels.

Sources (in order of authority — always prefer the highest available):
  1. Surveyor's ALTA/NSPS or boundary/topographic survey  (import as GeoJSON / DXF)
  2. County / municipal GIS parcels & building footprints (import as GeoJSON)
  3. OpenStreetMap via Overpass API  (footprints typically +/-1-3 m, heights often estimated)
  4. USGS 3DEP 1 m DEM (US) or SRTM/Copernicus 30 m (global) for terrain

Every feature records its `source` so drawings and reports can state accuracy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import LineString, Polygon, mapping, shape
from shapely.ops import unary_union

from .geo import LocalFrame, http_json

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
]

# Typical paved widths (m) by OSM highway class, used only when no width/lanes tag.
ROAD_WIDTH_M = {
    "motorway": 15.0, "trunk": 14.0, "primary": 12.0, "secondary": 10.5,
    "tertiary": 9.0, "unclassified": 7.0, "residential": 7.5, "service": 4.5,
    "living_street": 6.0, "pedestrian": 5.0, "footway": 2.0, "path": 1.5,
    "cycleway": 2.0, "track": 3.5, "motorway_link": 6.0, "trunk_link": 6.0,
    "primary_link": 6.0, "secondary_link": 6.0, "tertiary_link": 6.0,
}
LANE_WIDTH_M = 3.35  # ~11 ft
GREEN_LANDUSE = {"grass", "forest", "meadow", "recreation_ground", "village_green", "cemetery"}
GREEN_LEISURE = {"park", "garden", "pitch", "stadium", "golf_course", "playground", "sports_centre"}
LEVEL_HEIGHT_M = 3.2


def _parse_length(v: str | None) -> float | None:
    """Parse OSM length tags: '12', '12 m', '40 ft', "40'", "12'6\"'."""
    if not v:
        return None
    s = v.strip().lower().replace(",", ".")
    try:
        if "'" in s or "ft" in s:
            s = s.replace("ft", "'").replace('"', "").replace(" ", "")
            ft, _, inch = s.partition("'")
            return float(ft or 0) * 0.3048 + float(inch or 0) * 0.0254
        return float(s.replace("m", "").strip())
    except ValueError:
        return None


@dataclass
class Feature:
    kind: str                     # building | road | parcel | water | green | tree | rail
    geom: object                  # shapely geometry in local metres
    props: dict = field(default_factory=dict)
    source: str = "osm"

    @property
    def height(self) -> float:
        return float(self.props.get("height_m", 0.0))

    @property
    def base(self) -> float:
        return float(self.props.get("base_m", 0.0))


@dataclass
class SiteContext:
    frame: LocalFrame
    radius_m: float
    features: list[Feature] = field(default_factory=list)
    elevation: dict | None = None      # {"origin_m": float, "grid": [[x,y,z]...], "source": str}
    attribution: list[str] = field(default_factory=list)

    def of(self, kind: str) -> list[Feature]:
        return [f for f in self.features if f.kind == kind]

    # ---- terrain -------------------------------------------------------
    def ground_z(self, x: float, y: float) -> float:
        """Terrain height relative to site origin elevation (inverse-distance weighted)."""
        if not self.elevation or not self.elevation.get("grid"):
            return 0.0
        z0 = self.elevation["origin_m"]
        lut = self._grid_lut()
        if lut is not None:
            xs, ys, Z = lut
            import numpy as np

            x = min(max(x, xs[0]), xs[-1])
            y = min(max(y, ys[0]), ys[-1])
            i = min(int(np.searchsorted(xs, x, side="right")) - 1, len(xs) - 2)
            j = min(int(np.searchsorted(ys, y, side="right")) - 1, len(ys) - 2)
            tx = (x - xs[i]) / (xs[i + 1] - xs[i])
            ty = (y - ys[j]) / (ys[j + 1] - ys[j])
            z = (Z[j, i] * (1 - tx) * (1 - ty) + Z[j, i + 1] * tx * (1 - ty)
                 + Z[j + 1, i] * (1 - tx) * ty + Z[j + 1, i + 1] * tx * ty)
            return float(z) - z0
        num = den = 0.0
        for gx, gy, gz in self.elevation["grid"]:
            d2 = (gx - x) ** 2 + (gy - y) ** 2
            if d2 < 1e-6:
                return gz - z0
            w = 1.0 / d2
            num += w * gz
            den += w
        return num / den - z0

    def _grid_lut(self):
        """Regular-grid lookup (xs, ys, Z) for bilinear interpolation, or None."""
        cached = getattr(self, "_lut", "unset")
        if cached != "unset":
            return cached
        import numpy as np

        g = np.asarray(self.elevation["grid"], float)
        xs, ys = np.unique(np.round(g[:, 0], 3)), np.unique(np.round(g[:, 1], 3))
        self._lut = None
        if len(xs) >= 2 and len(ys) >= 2 and len(xs) * len(ys) == len(g):
            Z = np.full((len(ys), len(xs)), np.nan)
            for x, y, z in g:
                Z[np.searchsorted(ys, round(y, 3)), np.searchsorted(xs, round(x, 3))] = z
            if not np.isnan(Z).any():
                self._lut = (xs, ys, Z)
        return self._lut

    # ---- persistence ---------------------------------------------------
    def to_json(self) -> dict:
        return {
            "origin": {"lat": self.frame.lat0, "lon": self.frame.lon0},
            "radius_m": self.radius_m,
            "elevation": self.elevation,
            "attribution": self.attribution,
            "features": [
                {"kind": f.kind, "source": f.source, "props": f.props,
                 "geometry": mapping(f.geom)} for f in self.features
            ],
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "SiteContext":
        d = json.loads(Path(path).read_text())
        ctx = cls(LocalFrame(d["origin"]["lat"], d["origin"]["lon"]), d["radius_m"],
                  elevation=d.get("elevation"), attribution=d.get("attribution", []))
        for f in d["features"]:
            ctx.features.append(Feature(f["kind"], shape(f["geometry"]), f["props"], f["source"]))
        return ctx


# ---------------------------------------------------------------------------
# OpenStreetMap
# ---------------------------------------------------------------------------

def _overpass_query(s: float, w: float, n: float, e: float) -> str:
    bb = f"({s},{w},{n},{e})"
    return f"""
[out:json][timeout:120];
(
  way["building"]{bb};
  relation["building"]{bb};
  way["building:part"]{bb};
  way["highway"]{bb};
  way["railway"~"rail|light_rail|tram"]{bb};
  way["natural"="water"]{bb};
  way["waterway"]{bb};
  way["landuse"~"grass|forest|meadow|recreation_ground"]{bb};
  way["leisure"~"park|garden|pitch"]{bb};
  way["boundary"="cadastral"]{bb};
  way["landuse"="residential"]["ref"]{bb};
  node["natural"="tree"]{bb};
);
out geom;
"""


def _building_height(tags: dict) -> tuple[float, float, str]:
    """Return (height, min_height, provenance)."""
    h = _parse_length(tags.get("height"))
    base = _parse_length(tags.get("min_height")) or 0.0
    if h:
        return h, base, "tagged height"
    lv = tags.get("building:levels")
    roof_lv = tags.get("roof:levels")
    try:
        if lv:
            levels = float(lv) + (float(roof_lv) * 0.6 if roof_lv else 0)
            min_lv = float(tags.get("building:min_level", 0))
            return levels * LEVEL_HEIGHT_M, base or min_lv * LEVEL_HEIGHT_M, "from levels"
    except ValueError:
        pass
    btype = tags.get("building", "yes")
    guess = {"garage": 3.0, "shed": 2.5, "carport": 2.8, "house": 7.0, "detached": 7.0,
             "residential": 9.5, "apartments": 12.0, "commercial": 8.0, "retail": 5.5,
             "industrial": 8.0, "church": 14.0, "school": 8.0, "office": 15.0}.get(btype, 6.0)
    return guess, base, "ESTIMATED (no height tag)"


def _road_width(tags: dict) -> tuple[float, str]:
    w = _parse_length(tags.get("width"))
    if w:
        return w, "tagged width"
    lanes = tags.get("lanes")
    if lanes:
        try:
            return float(lanes) * LANE_WIDTH_M + 1.0, "from lanes"
        except ValueError:
            pass
    return ROAD_WIDTH_M.get(tags.get("highway", ""), 6.0), "ESTIMATED (class default)"


def _closed_ring(coords: list[tuple[float, float]]) -> Polygon | None:
    if len(coords) < 4:
        return None
    p = Polygon(coords)
    if not p.is_valid:
        p = p.buffer(0)
    return p if (not p.is_empty and p.area > 0.5) else None


def parse_osm(data: dict, frame: LocalFrame) -> list[Feature]:
    feats: list[Feature] = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "node":
            if tags.get("natural") == "tree":
                x, y = frame.to_local(el["lon"], el["lat"])
                from shapely.geometry import Point
                h = _parse_length(tags.get("height")) or 8.0
                feats.append(Feature("tree", Point(x, y), {"height_m": h, "osm_id": el["id"],
                                     "crown_m": _parse_length(tags.get("diameter_crown")) or 5.0}))
            continue
        if el["type"] == "relation":
            # Multipolygon building: outer rings minus inner rings.
            outers, inners = [], []
            for m in el.get("members", []):
                if "geometry" not in m:
                    continue
                ring = [frame.to_local(p["lon"], p["lat"]) for p in m["geometry"]]
                poly = _closed_ring(ring)
                if poly is None:
                    continue
                (inners if m.get("role") == "inner" else outers).append(poly)
            if not outers:
                continue
            geom = unary_union(outers)
            if inners:
                geom = geom.difference(unary_union(inners))
            h, base, prov = _building_height(tags)
            feats.append(Feature("building", geom, {"height_m": h, "base_m": base,
                                 "height_source": prov, "osm_id": el["id"], **_keep(tags)}))
            continue
        geom_pts = el.get("geometry")
        if not geom_pts:
            continue
        pts = [frame.to_local(p["lon"], p["lat"]) for p in geom_pts]
        closed = len(pts) > 3 and pts[0] == pts[-1]
        if "building" in tags or "building:part" in tags:
            poly = _closed_ring(pts)
            if poly is None:
                continue
            h, base, prov = _building_height(tags)
            kind = "building" if "building" in tags else "building_part"
            feats.append(Feature(kind, poly, {"height_m": h, "base_m": base,
                                 "height_source": prov, "osm_id": el["id"], **_keep(tags)}))
        elif "highway" in tags:
            line = LineString(pts)
            w, prov = _road_width(tags)
            feats.append(Feature("road", line, {"width_m": w, "width_source": prov,
                                 "highway": tags["highway"], "name": tags.get("name", ""),
                                 "osm_id": el["id"]}))
        elif "railway" in tags:
            feats.append(Feature("rail", LineString(pts), {"railway": tags["railway"],
                                 "osm_id": el["id"]}))
        elif tags.get("boundary") == "cadastral" and closed:
            poly = _closed_ring(pts)
            if poly is not None:
                feats.append(Feature("parcel", poly, {"ref": tags.get("ref", ""),
                                     "osm_id": el["id"]}))
        elif tags.get("natural") == "water" and closed:
            poly = _closed_ring(pts)
            if poly is not None:
                feats.append(Feature("water", poly, {"osm_id": el["id"]}))
        elif "waterway" in tags:
            feats.append(Feature("waterway", LineString(pts), {"osm_id": el["id"]}))
        elif closed and (tags.get("landuse") in GREEN_LANDUSE or tags.get("leisure") in GREEN_LEISURE
                         or tags.get("natural") in ("wood", "grassland", "scrub")):
            poly = _closed_ring(pts)
            if poly is not None:
                feats.append(Feature("green", poly, {
                    "osm_id": el["id"], "use": tags.get("leisure") or tags.get("landuse") or tags.get("natural"),
                    "sport": tags.get("sport", ""), "name": tags.get("name", ""),
                    "surface": tags.get("surface", "")}))
    return feats


def _keep(tags: dict) -> dict:
    keys = ("name", "building", "addr:housenumber", "addr:street", "building:levels",
            "roof:shape", "building:material", "building:colour", "roof:colour")
    return {k: tags[k] for k in keys if k in tags}


def osm_xml_to_elements(xml_text: str) -> dict:
    """Convert OSM API 0.6 XML into Overpass-style JSON (`out geom`) elements."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    nodes = {}
    els = []
    for nd in root.iter("node"):
        nodes[nd.get("id")] = (float(nd.get("lat")), float(nd.get("lon")))
        tags = {t.get("k"): t.get("v") for t in nd.iter("tag")}
        if tags:
            els.append({"type": "node", "id": int(nd.get("id")), "lat": nodes[nd.get("id")][0],
                        "lon": nodes[nd.get("id")][1], "tags": tags})
    ways = {}
    for w in root.iter("way"):
        refs = [r.get("ref") for r in w.iter("nd")]
        geom = [{"lat": nodes[r][0], "lon": nodes[r][1]} for r in refs if r in nodes]
        tags = {t.get("k"): t.get("v") for t in w.iter("tag")}
        ways[w.get("id")] = geom
        if len(geom) == len(refs) and len(geom) >= 2:
            els.append({"type": "way", "id": int(w.get("id")), "tags": tags, "geometry": geom})
    for r in root.iter("relation"):
        tags = {t.get("k"): t.get("v") for t in r.iter("tag")}
        if "building" not in tags:
            continue
        members = [{"type": m.get("type"), "role": m.get("role"), "geometry": ways[m.get("ref")]}
                   for m in r.iter("member") if m.get("type") == "way" and m.get("ref") in ways]
        els.append({"type": "relation", "id": int(r.get("id")), "tags": tags, "members": members})
    return {"elements": els}


def _fetch_osm_api(frame: LocalFrame, radius_m: float) -> dict:
    """Fallback: official OSM editing API (bbox must be < 0.25 deg^2; fine for sites)."""
    import requests

    from .geo import USER_AGENT, _cache_get, _cache_put

    s, w, n, e = frame.bbox_lonlat(radius_m)
    url = f"https://api.openstreetmap.org/api/0.6/map?bbox={w:.6f},{s:.6f},{e:.6f},{n:.6f}"
    cached = _cache_get(url)
    if cached is not None:
        return cached
    r = requests.get(url, timeout=120, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    data = osm_xml_to_elements(r.text)
    _cache_put(url, data)
    return data


def fetch_osm(frame: LocalFrame, radius_m: float) -> list[Feature]:
    s, w, n, e = frame.bbox_lonlat(radius_m)
    q = _overpass_query(s, w, n, e)
    errors = []
    for url in OVERPASS_URLS:
        try:
            data = http_json(url, data={"data": q}, timeout=180, retries=2)
            return parse_osm(data, frame)
        except RuntimeError as err:
            errors.append(str(err))
    try:
        return parse_osm(_fetch_osm_api(frame, radius_m), frame)
    except Exception as err:  # noqa: BLE001
        errors.append(f"OSM API: {err}")
    raise RuntimeError("OpenStreetMap unavailable: " + " | ".join(errors))


# ---------------------------------------------------------------------------
# Elevation
# ---------------------------------------------------------------------------

def _usgs_point(lon: float, lat: float) -> float | None:
    d = http_json("https://epqs.nationalmap.gov/v1/json",
                  params={"x": f"{lon:.7f}", "y": f"{lat:.7f}", "units": "Meters", "wkid": 4326})
    try:
        v = float(d["value"])
        return None if v < -1000 else v
    except (KeyError, TypeError, ValueError):
        return None


def fetch_elevation(frame: LocalFrame, radius_m: float, n: int = 9,
                    source: str = "auto") -> dict:
    """Sample an n x n terrain grid. USGS 3DEP (1 m, US only) or OpenTopoData SRTM 30 m."""
    xs = [(-1 + 2 * i / (n - 1)) * radius_m for i in range(n)]
    pts = [(x, y) for y in xs for x in xs]
    lonlat = [frame.to_lonlat(x, y) for x, y in pts]
    grid: list[list[float]] = []
    used = None
    if source in ("auto", "usgs"):
        try:
            z0 = _usgs_point(frame.lon0, frame.lat0)
            if z0 is not None:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=8) as ex:
                    zs = list(ex.map(lambda ll: _usgs_point(*ll), lonlat))
                for (x, y), z in zip(pts, zs):
                    if z is not None:
                        grid.append([x, y, z])
                used = "USGS 3DEP (1 m DEM) via EPQS"
                return {"origin_m": z0, "grid": grid, "source": used}
        except RuntimeError:
            if source == "usgs":
                raise
    # Global fallback: OpenTopoData (max 100 locations per request).
    allpts = [(frame.lon0, frame.lat0)] + lonlat
    vals: list[float | None] = []
    for i in range(0, len(allpts), 100):
        chunk = allpts[i:i + 100]
        locs = "|".join(f"{lat:.7f},{lon:.7f}" for lon, lat in chunk)
        d = http_json("https://api.opentopodata.org/v1/srtm30m", params={"locations": locs})
        vals += [r.get("elevation") for r in d.get("results", [])]
    z0 = vals[0] or 0.0
    for (x, y), z in zip(pts, vals[1:]):
        if z is not None:
            grid.append([x, y, float(z)])
    return {"origin_m": z0, "grid": grid, "source": "SRTM 30 m via OpenTopoData (+/-~5 m vertical)"}


# ---------------------------------------------------------------------------
# Authoritative imports
# ---------------------------------------------------------------------------

def import_geojson(path: str | Path, frame: LocalFrame, kind: str, source: str) -> list[Feature]:
    """Import county GIS / survey GeoJSON (WGS84 lon/lat) as features of `kind`."""
    d = json.loads(Path(path).read_text())
    feats = d["features"] if d.get("type") == "FeatureCollection" else [d]
    out = []
    for f in feats:
        g = shape(f["geometry"])
        from shapely.ops import transform
        g_local = transform(lambda x, y, z=None: _xy_vec(frame, x, y), g)
        props = dict(f.get("properties") or {})
        if kind == "building" and "height_m" not in props:
            props["height_m"] = float(props.get("height", 6.0))
        out.append(Feature(kind, g_local, props, source))
    return out


def _xy_vec(frame: LocalFrame, xs, ys):
    try:
        pairs = [frame.to_local(x, y) for x, y in zip(xs, ys)]
        return tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)
    except TypeError:
        return frame.to_local(xs, ys)


def build_context(lat: float, lon: float, radius_m: float = 200.0, *, osm: bool = True,
                  terrain: bool = True, imports: list[dict] | None = None,
                  osm_json: str | None = None) -> SiteContext:
    frame = LocalFrame(lat, lon)
    ctx = SiteContext(frame, radius_m)
    if osm_json:
        ctx.features += parse_osm(json.loads(Path(osm_json).read_text()), frame)
        ctx.attribution.append("© OpenStreetMap contributors (ODbL)")
    elif osm:
        ctx.features += fetch_osm(frame, radius_m)
        ctx.attribution.append("© OpenStreetMap contributors (ODbL)")
    for imp in imports or []:
        new = import_geojson(imp["path"], frame, imp["kind"], imp.get("source", imp["path"]))
        if imp.get("replace_osm"):
            ctx.features = [f for f in ctx.features if f.kind != imp["kind"]]
        ctx.features += new
        ctx.attribution.append(imp.get("source", imp["path"]))
    if terrain:
        try:
            ctx.elevation = fetch_elevation(frame, radius_m)
            ctx.attribution.append(ctx.elevation["source"])
        except RuntimeError as e:
            ctx.elevation = None
            ctx.attribution.append(f"terrain unavailable ({e}); site assumed flat")
    for f in ctx.features:
        if f.kind in ("building", "building_part"):
            c = f.geom.representative_point()
            f.props["ground_z"] = round(ctx.ground_z(c.x, c.y), 3)
    return ctx


def road_polygons(ctx: SiteContext):
    """Buffer road centrelines to their (estimated) paved width."""
    out = []
    for r in ctx.of("road"):
        out.append((r, r.geom.buffer(r.props.get("width_m", 6.0) / 2, cap_style="flat")))
    return out


def nearest_road(ctx: SiteContext, geom, include_minor: bool = False):
    best, best_d = None, math.inf
    for r in ctx.of("road"):
        if not include_minor and r.props.get("highway") in ("footway", "path", "cycleway",
                                                             "steps", "track", "service"):
            continue
        d = r.geom.distance(geom)
        if d < best_d:
            best, best_d = r, d
    return best, best_d
