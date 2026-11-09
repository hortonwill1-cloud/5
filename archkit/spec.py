"""Design specification: the single source of truth for a project.

A project YAML declares the site, zoning envelope, applicable code profile,
and the building (levels, walls, openings, rooms, roof, materials).  All
lengths in the YAML are in `project.units` ("ft" or "m"); they are converted
to metres on load.  See examples/ for a complete annotated spec.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Polygon

from .units import UNIT_TO_M


@dataclass
class Opening:
    kind: str           # door | window | garage_door | storefront
    wall: str
    segment: int
    offset: float       # from segment start to near edge of opening (m)
    width: float
    height: float
    sill: float = 0.0
    tag: str = ""
    swing: str = "in-left"   # in-left|in-right|out-left|out-right|slide|none
    egress: bool = False
    operable: bool = True


@dataclass
class Wall:
    id: str
    path: list[tuple[float, float]]   # centreline, building frame (m)
    thickness: float
    closed: bool = False
    exterior: bool = False
    height: float | None = None       # default: level floor-to-floor
    material: str = "plaster"
    fire_rating_hr: float = 0.0

    def segments(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        pts = list(self.path)
        if self.closed:
            pts = pts + [pts[0]]
        return list(zip(pts[:-1], pts[1:]))


@dataclass
class Room:
    name: str
    number: str
    polygon: list[tuple[float, float]]
    use: str = "habitable"   # habitable | sleeping | bath | circulation | storage | garage | other
    finish: str = ""

    @property
    def poly(self) -> Polygon:
        return Polygon(self.polygon)


@dataclass
class Stair:
    start: tuple[float, float]
    direction_deg: float
    width: float
    risers: int
    riser: float
    tread: float


@dataclass
class Level:
    name: str
    elevation: float          # top of slab, relative to project datum (m)
    floor_to_floor: float
    ceiling: float
    slab: float = 0.3
    walls: list[Wall] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    stairs: list[Stair] = field(default_factory=list)

    def wall(self, wid: str) -> Wall:
        for w in self.walls:
            if w.id == wid:
                return w
        raise KeyError(f"level {self.name!r}: no wall {wid!r}")

    def footprint(self) -> Polygon:
        """Outer face of exterior walls."""
        for w in self.walls:
            if w.exterior and w.closed:
                return Polygon(w.path).buffer(w.thickness / 2, join_style="mitre")
        raise ValueError(f"level {self.name!r} has no closed exterior wall")


@dataclass
class Roof:
    type: str = "flat"         # flat | gable | hip | shed
    pitch: float = 0.0         # rise/run
    overhang: float = 0.0
    parapet: float = 0.0
    thickness: float = 0.3
    ridge_axis: str = "auto"   # auto | x | y
    material: str = "roof_membrane"


@dataclass
class Building:
    placement: tuple[float, float]   # building-frame origin in site frame (m)
    rotation_deg: float              # CCW from site +x
    levels: list[Level]
    roof: Roof
    finish_floor_elev: float | None = None  # level-1 FF vs site-origin grade (m); None = auto
    materials: dict = field(default_factory=dict)
    occupancy: str = "R-3"
    construction_type: str = "V-B"
    sprinklered: bool = False

    def to_site(self, geom):
        g = rotate(geom, self.rotation_deg, origin=(0, 0), use_radians=False)
        return translate(g, *self.placement)

    def pt_to_site(self, x: float, y: float) -> tuple[float, float]:
        a = math.radians(self.rotation_deg)
        return (self.placement[0] + x * math.cos(a) - y * math.sin(a),
                self.placement[1] + x * math.sin(a) + y * math.cos(a))

    def footprint(self) -> Polygon:
        return self.levels[0].footprint()

    def top_of_wall(self) -> float:
        top = self.levels[-1]
        return top.elevation + top.floor_to_floor

    def roof_peak(self) -> float:
        """Highest point above level-1 finished floor."""
        tow = self.top_of_wall()
        if self.roof.type == "flat":
            return tow + self.roof.parapet
        # Roof surface sits `thickness` (heel/rafter depth) above top of wall at the
        # wall line; the overhang lowers the eave but does not raise the ridge.
        span = self._roof_span()
        rise = self.roof.pitch * span / 2
        if self.roof.type == "shed":
            rise = self.roof.pitch * (span + self.roof.overhang)
        return tow + self.roof.thickness + rise

    def _roof_span(self) -> float:
        fp = self.footprint()
        minx, miny, maxx, maxy = fp.bounds
        axis = self.ridge_axis()
        return (maxy - miny) if axis == "x" else (maxx - minx)

    def ridge_axis(self) -> str:
        if self.roof.ridge_axis in ("x", "y"):
            return self.roof.ridge_axis
        minx, miny, maxx, maxy = self.footprint().bounds
        return "x" if (maxx - minx) >= (maxy - miny) else "y"

    def gross_area(self) -> float:
        return sum(lv.footprint().area for lv in self.levels)


@dataclass
class View:
    name: str
    eye: tuple[float, float, float]      # site frame m; z above grade at eye
    target: tuple[float, float, float]
    focal_mm: float = 35.0
    sensor_mm: float = 36.0
    resolution: tuple[int, int] = (3840, 2160)
    datetime: str = "2026-06-21T15:00:00"
    utc_offset_h: float = 0.0
    background_photo: str | None = None
    camera_solve: dict | None = None     # {"photo":..., "points":[{"px":[u,v],"site":[x,y,z]}...]}
    samples: int = 512


@dataclass
class Project:
    name: str
    number: str
    client: str
    architect: str
    address: str
    units: str          # "ft" | "m"
    system: str         # "imperial" | "metric"
    lat: float | None
    lon: float | None
    context_radius: float
    parcel: Polygon | None
    parcel_source: str
    imports: list[dict]
    zoning: dict
    code: dict
    building: Building
    views: list[View]
    sheet: dict
    root: Path
    osm_json: str | None = None
    elevation_source: str = "auto"
    timezone: str | None = None   # IANA name, e.g. America/Denver (DST-aware)


def _len(v, k):
    return None if v is None else float(v) * k


def _pts(seq, k):
    return [(float(x) * k, float(y) * k) for x, y in seq]


def _pitch(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, str) and ":" in v:
        a, b = v.split(":")
        return float(a) / float(b)
    if isinstance(v, str) and v.endswith("deg"):
        return math.tan(math.radians(float(v[:-3])))
    return float(v)


def load(path: str | Path) -> Project:
    path = Path(path)
    d = yaml.safe_load(path.read_text())
    p = d.get("project", {})
    units = p.get("units", "ft")
    k = UNIT_TO_M[units]
    system = "imperial" if units in ("ft", "in") else "metric"

    site = d.get("site", {})
    loc = site.get("location", {})
    lat, lon = loc.get("lat"), loc.get("lon")
    parcel = None
    parcel_src = "none"
    pd = site.get("parcel")
    if pd and "local" in pd:
        parcel = Polygon(_pts(pd["local"], k))
        parcel_src = pd.get("source", "spec (local coordinates)")
    elif pd and "latlon" in pd:
        parcel = ("latlon", pd["latlon"])  # resolved once a frame exists
        parcel_src = pd.get("source", "spec (lat/lon)")
    elif pd and "geojson" in pd:
        parcel = ("geojson", str((path.parent / pd["geojson"]).resolve()))
        parcel_src = pd.get("source", pd["geojson"])

    b = d["building"]
    levels = []
    for lv in b["levels"]:
        walls = []
        for w in lv.get("walls", []):
            walls.append(Wall(
                id=str(w["id"]), path=_pts(w["path"], k), thickness=float(w["thickness"]) * k,
                closed=bool(w.get("closed", False)), exterior=bool(w.get("exterior", False)),
                height=_len(w.get("height"), k), material=w.get("material",
                                                               "facade" if w.get("exterior") else "plaster"),
                fire_rating_hr=float(w.get("fire_rating_hr", 0))))
        ops = []
        for o in lv.get("openings", []):
            kind = o.get("kind", "window")
            ops.append(Opening(
                kind=kind, wall=str(o["wall"]), segment=int(o.get("segment", 0)),
                offset=float(o["offset"]) * k, width=float(o["width"]) * k,
                height=float(o.get("height", 6.6667 if units == "ft" else 2.03)) * k,
                sill=float(o.get("sill", 0 if kind in ("door", "garage_door") else
                                 (3.0 if units == "ft" else 0.9))) * k,
                tag=str(o.get("tag", "")), swing=o.get("swing", "in-left"),
                egress=bool(o.get("egress", False)), operable=bool(o.get("operable", True))))
        rooms = [Room(str(r["name"]), str(r.get("number", "")), _pts(r["polygon"], k),
                      r.get("use", "habitable"), r.get("finish", "")) for r in lv.get("rooms", [])]
        stairs = [Stair(tuple(_pts([s["start"]], k)[0]), float(s.get("direction_deg", 0)),
                        float(s["width"]) * k, int(s["risers"]), float(s["riser"]) * k,
                        float(s["tread"]) * k) for s in lv.get("stairs", [])]
        levels.append(Level(
            name=lv["name"], elevation=float(lv.get("elevation", 0)) * k,
            floor_to_floor=float(lv["floor_to_floor"]) * k,
            ceiling=float(lv.get("ceiling", lv["floor_to_floor"])) * k,
            slab=float(lv.get("slab", 1.0 if units == "ft" else 0.3)) * k,
            walls=walls, openings=ops, rooms=rooms, stairs=stairs))
    # Upper levels inherit exterior wall if omitted.
    for i, lv in enumerate(levels[1:], 1):
        if not any(w.exterior for w in lv.walls):
            for w in levels[i - 1].walls:
                if w.exterior:
                    lv.walls.insert(0, Wall(w.id, list(w.path), w.thickness, w.closed, True,
                                            None, w.material, w.fire_rating_hr))
    r = b.get("roof", {})
    roof = Roof(type=r.get("type", "flat"), pitch=_pitch(r.get("pitch")),
                overhang=float(r.get("overhang", 0)) * k, parapet=float(r.get("parapet", 0)) * k,
                thickness=float(r.get("thickness", 1.0 if units == "ft" else 0.3)) * k,
                ridge_axis=r.get("ridge_axis", "auto"), material=r.get("material", "roof_membrane"))
    pl = b.get("placement", {})
    building = Building(
        placement=(float(pl.get("x", 0)) * k, float(pl.get("y", 0)) * k),
        rotation_deg=float(pl.get("rotation_deg", 0)), levels=levels, roof=roof,
        finish_floor_elev=(None if str(b.get("finish_floor", "auto")) == "auto"
                           else float(b["finish_floor"]) * k),
        materials=b.get("materials", {}), occupancy=b.get("occupancy", "R-3"),
        construction_type=b.get("construction_type", "V-B"),
        sprinklered=bool(b.get("sprinklered", False)))

    views = []
    for v in d.get("views", []):
        views.append(View(
            name=v["name"], eye=tuple(float(c) * k for c in v["eye"]),
            target=tuple(float(c) * k for c in v["target"]),
            focal_mm=float(v.get("focal_mm", 35)), sensor_mm=float(v.get("sensor_mm", 36)),
            resolution=tuple(v.get("resolution", [3840, 2160])),
            datetime=str(v.get("datetime", "2026-06-21T15:00:00")),
            utc_offset_h=float(v.get("utc_offset_h", 0)),
            background_photo=(str((path.parent / v["background_photo"]).resolve())
                              if v.get("background_photo") else None),
            camera_solve=v.get("camera_solve"), samples=int(v.get("samples", 512))))

    zoning = dict(d.get("zoning", {}))
    for key in ("max_height",):
        if key in zoning:
            zoning[key] = float(zoning[key]) * k
    if "setbacks" in zoning:
        zoning["setbacks"] = {s: float(v) * k for s, v in zoning["setbacks"].items()}

    imports = []
    for imp in site.get("imports", []):
        imp = dict(imp)
        imp["path"] = str((path.parent / imp["path"]).resolve())
        imports.append(imp)

    return Project(
        name=p.get("name", path.stem), number=str(p.get("number", "")),
        client=p.get("client", ""), architect=p.get("architect", ""),
        address=site.get("address", ""), units=units, system=system, lat=lat, lon=lon,
        context_radius=float(site.get("context_radius_m", 150)),
        parcel=parcel, parcel_source=parcel_src, imports=imports, zoning=zoning,
        code=d.get("code", {"profile": "IRC-2021"}), building=building, views=views,
        sheet=d.get("sheet", {}), root=path.parent,
        osm_json=(str((path.parent / site["osm_json"]).resolve()) if site.get("osm_json") else None),
        elevation_source=site.get("elevation_source", "auto"),
        timezone=site.get("timezone"))


def resolve_parcel(project: Project, frame) -> None:
    """Convert a lat/lon or GeoJSON parcel into the local frame once the frame is known."""
    import json

    from shapely.geometry import shape
    from shapely.ops import transform

    if isinstance(project.parcel, tuple):
        kind, val = project.parcel
        if kind == "latlon":
            project.parcel = Polygon([frame.to_local(lon, lat) for lat, lon in val])
        else:
            g = json.loads(Path(val).read_text())
            if g.get("type") == "FeatureCollection":
                g = g["features"][0]
            geom = shape(g.get("geometry", g))
            project.parcel = transform(lambda x, y, z=None: (
                tuple(frame.to_local(a, b)[0] for a, b in zip(x, y)),
                tuple(frame.to_local(a, b)[1] for a, b in zip(x, y))), geom)


def wall_segment_geom(w: Wall, seg: int) -> LineString:
    a, b = w.segments()[seg]
    return LineString([a, b])
