"""3D model generation (site frame, metres, z = height above site-origin grade).

Produces a list of planar faces tagged with material and group, from which
we export OBJ/MTL (Blender, Rhino, SketchUp, Revit import) and drive the
built-in preview renderer and elevation drawings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon

from . import materials
from .site import SiteContext, road_polygons
from .spec import Building, Level, Opening, Project, Wall


@dataclass
class Face:
    verts: np.ndarray        # (n, 3)
    material: str
    group: str


@dataclass
class Model:
    faces: list[Face] = field(default_factory=list)

    def add(self, verts, material: str, group: str) -> None:
        v = np.asarray(verts, dtype=float)
        if len(v) >= 3:
            self.faces.append(Face(v, material, group))

    def bounds(self, groups: set[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        vs = [f.verts for f in self.faces if groups is None or f.group in groups]
        allv = np.vstack(vs)
        return allv.min(0), allv.max(0)


# ---------------------------------------------------------------------------
# Primitive builders
# ---------------------------------------------------------------------------

def triangulate(poly: Polygon) -> list[np.ndarray]:
    """Constrained triangulation of a polygon (holes respected) -> list of (3,2) arrays."""
    import shapely

    poly = poly.buffer(0)
    polys = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
    out = []
    for p in polys:
        if p.is_empty:
            continue
        try:
            tris = shapely.constrained_delaunay_triangles(p)
            geoms = list(tris.geoms)
        except AttributeError:  # shapely < 2.1
            from shapely.ops import triangulate as tri
            geoms = [t for t in tri(p) if p.contains(t.representative_point())]
        for t in geoms:
            c = np.asarray(t.exterior.coords)[:3]
            # enforce CCW
            if (c[1, 0] - c[0, 0]) * (c[2, 1] - c[0, 1]) - (c[2, 0] - c[0, 0]) * (c[1, 1] - c[0, 1]) < 0:
                c = c[::-1]
            out.append(c)
    return out


def add_prism(m: Model, poly: Polygon, z0: float, z1: float, mat: str, group: str,
              top_mat: str | None = None, zfun=None) -> None:
    """Extrude polygon from z0 to z1. zfun(x, y) optionally offsets base per-vertex."""
    if poly.is_empty or z1 <= z0:
        return
    for tri in triangulate(poly):
        m.add([[x, y, z1] for x, y in tri], top_mat or mat, group)
        m.add([[x, y, z0] for x, y in tri[::-1]], mat, group)
    polys = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            c = list(ring.coords)
            for (ax, ay), (bx, by) in zip(c[:-1], c[1:]):
                za = z0 + (zfun(ax, ay) if zfun else 0)
                zb = z0 + (zfun(bx, by) if zfun else 0)
                m.add([[ax, ay, za], [bx, by, zb], [bx, by, z1], [ax, ay, z1]], mat, group)


def add_box_along(m: Model, a, u, n, s0, s1, w0, w1, z0, z1, mat, group):
    """Box spanning [s0,s1] along unit u, [w0,w1] along normal n, [z0,z1] vertical."""
    if s1 - s0 < 1e-4 or z1 - z0 < 1e-4:
        return
    a, u, n = np.asarray(a), np.asarray(u), np.asarray(n)

    def P(s, w, z):
        xy = a + u * s + n * w
        return [xy[0], xy[1], z]

    c = [P(s0, w0, z0), P(s1, w0, z0), P(s1, w1, z0), P(s0, w1, z0),
         P(s0, w0, z1), P(s1, w0, z1), P(s1, w1, z1), P(s0, w1, z1)]
    for idx in ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7), (3, 2, 1, 0)):
        m.add([c[i] for i in idx], mat, group)


def add_sphere(m: Model, cx, cy, cz, r, mat, group, nu=10, nv=6, squash=1.0):
    for i in range(nv):
        t0, t1 = math.pi * i / nv, math.pi * (i + 1) / nv
        for j in range(nu):
            p0, p1 = 2 * math.pi * j / nu, 2 * math.pi * (j + 1) / nu

            def P(t, p):
                return [cx + r * math.sin(t) * math.cos(p), cy + r * math.sin(t) * math.sin(p),
                        cz + r * squash * math.cos(t)]
            m.add([P(t0, p0), P(t1, p0), P(t1, p1), P(t0, p1)], mat, group)


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def _xf(b: Building, xy):
    return np.array(b.pt_to_site(*xy))


def _rot(b: Building, v):
    a = math.radians(b.rotation_deg)
    return np.array([v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)])


def wall_openings(level: Level, wall: Wall, seg: int) -> list[Opening]:
    return sorted([o for o in level.openings if o.wall == wall.id and o.segment == seg],
                  key=lambda o: o.offset)


def add_wall(m: Model, b: Building, level: Level, wall: Wall, z_base: float, mats: dict):
    height = wall.height or level.floor_to_floor
    mat = wall.material
    grp = f"{level.name}/walls"
    segs = wall.segments()
    for si, (pa, pb) in enumerate(segs):
        a, bb = np.array(pa), np.array(pb)
        L = float(np.linalg.norm(bb - a))
        if L < 1e-6:
            continue
        u_local = (bb - a) / L
        n_local = np.array([-u_local[1], u_local[0]])
        A = _xf(b, a)
        u = _rot(b, u_local)
        n = _rot(b, n_local)
        t = wall.thickness
        # Butt-joint corners: each segment owns the corner column at its START and stops
        # short at its END, so no two faces are coplanar (no z-fighting seams in renders).
        ext0 = t / 2 if (wall.closed or si > 0) else 0.0
        ext1 = -t / 2 if (wall.closed or si < len(segs) - 1) else 0.0
        ops = wall_openings(level, wall, si)
        cursor = -ext0
        for o in ops:
            s0, s1 = o.offset, o.offset + o.width
            add_box_along(m, A, u, n, cursor, s0, -t / 2, t / 2, z_base, z_base + height, mat, grp)
            # below sill / above head
            add_box_along(m, A, u, n, s0, s1, -t / 2, t / 2, z_base, z_base + o.sill, mat, grp)
            add_box_along(m, A, u, n, s0, s1, -t / 2, t / 2, z_base + o.sill + o.height,
                          z_base + height, mat, grp)
            _add_opening_fill(m, A, u, n, o, t, z_base, level, mats)
            cursor = s1
        add_box_along(m, A, u, n, cursor, L + ext1, -t / 2, t / 2, z_base, z_base + height, mat, grp)


def _add_opening_fill(m, A, u, n, o: Opening, t, z_base, level, mats):
    grp = f"{level.name}/openings"
    fw = 0.06  # frame face width
    z0, z1 = z_base + o.sill, z_base + o.sill + o.height
    s0, s1 = o.offset, o.offset + o.width
    frame = mats.get("frame", "frame_black")
    if o.kind in ("window", "storefront") or (o.kind == "door" and "glass" in o.tag.lower()):
        glass = mats.get("glazing", "glass")
        # frame ring
        d0, d1 = -0.05, 0.05
        add_box_along(m, A, u, n, s0, s1, d0, d1, z0, z0 + fw, frame, grp)
        add_box_along(m, A, u, n, s0, s1, d0, d1, z1 - fw, z1, frame, grp)
        add_box_along(m, A, u, n, s0, s0 + fw, d0, d1, z0, z1, frame, grp)
        add_box_along(m, A, u, n, s1 - fw, s1, d0, d1, z0, z1, frame, grp)
        # mullions every ~1.2 m
        nmul = int(o.width // 1.25)
        for k in range(1, nmul + 1):
            s = s0 + o.width * k / (nmul + 1)
            add_box_along(m, A, u, n, s - fw / 2, s + fw / 2, d0, d1, z0, z1, frame, grp)
        add_box_along(m, A, u, n, s0 + fw, s1 - fw, -0.006, 0.006, z0 + fw, z1 - fw, glass, grp)
    elif o.kind == "garage_door":
        add_box_along(m, A, u, n, s0, s1, t / 2 - 0.08, t / 2 - 0.04, z0, z1,
                      mats.get("garage_door", "metal_panel"), grp)
    else:  # door
        add_box_along(m, A, u, n, s0, s1, t / 2 - 0.09, t / 2 - 0.04, z0, z1,
                      mats.get("door", "door"), grp)


def roof_faces(m: Model, b: Building, z_tow: float):
    r = b.roof
    fp = b.footprint()
    grp = "roof"
    mat = r.material
    facade = b.levels[-1].walls[0].material if b.levels[-1].walls else "facade"
    if r.type == "flat":
        add_prism(m, b.to_site(fp), z_tow, z_tow + r.thickness * 0.2, "slab", grp, top_mat=mat)
        if r.parapet > 0:
            inner = fp.buffer(-0.2, join_style="mitre")
            ring = fp.difference(inner)
            add_prism(m, b.to_site(ring), z_tow, z_tow + r.parapet, facade, grp)
        return
    minx, miny, maxx, maxy = fp.bounds
    oh = r.overhang
    axis = b.ridge_axis()
    # frieze band closing the heel between top of wall and roof surface
    if r.thickness > 0:
        ring = fp.difference(fp.buffer(-0.2, join_style="mitre"))
        add_prism(m, b.to_site(ring), z_tow, z_tow + r.thickness, facade, grp)
    z_tow = z_tow + r.thickness
    # Work in a frame where the ridge runs along local "s" and slopes across "w".
    if axis == "x":
        s0, s1, w0, w1 = minx, maxx, miny, maxy

        def P(s, w, z):
            return [*b.pt_to_site(s, w), z]
    else:
        s0, s1, w0, w1 = miny, maxy, minx, maxx

        def P(s, w, z):
            return [*b.pt_to_site(w, s), z]
    half = (w1 - w0) / 2
    wc = (w0 + w1) / 2
    pitch = r.pitch
    z_eave = z_tow - pitch * oh
    if r.type == "shed":
        z_hi = z_tow + pitch * (w1 - w0 + oh)
        m.add([P(s0 - oh, w0 - oh, z_eave), P(s1 + oh, w0 - oh, z_eave),
               P(s1 + oh, w1 + oh, z_hi), P(s0 - oh, w1 + oh, z_hi)], mat, grp)
        for s, sgn in ((s0, -1), (s1, 1)):
            tri = [P(s, w0, z_tow), P(s, w1, z_tow), P(s, w1, z_tow + pitch * (w1 - w0))]
            m.add(tri if sgn > 0 else tri[::-1], facade, grp)
        add_prism_soffit(m, P, s0, s1, w0, w1, oh, z_eave)
        return
    z_ridge = z_tow + pitch * half
    if r.type == "gable":
        for side in (-1, 1):
            we = wc + side * (half + oh)
            quad = [P(s0 - oh, we, z_eave), P(s1 + oh, we, z_eave),
                    P(s1 + oh, wc, z_ridge), P(s0 - oh, wc, z_ridge)]
            m.add(quad if side < 0 else quad[::-1], mat, grp)
            # fascia
            m.add([P(s0 - oh, we, z_eave - 0.2), P(s1 + oh, we, z_eave - 0.2),
                   P(s1 + oh, we, z_eave), P(s0 - oh, we, z_eave)], "frame_white", grp)
        for s, sgn in ((s0, -1), (s1, 1)):  # gable-end wall triangles, outward-facing
            tri = [P(s, w0, z_tow), P(s, w1, z_tow), P(s, wc, z_ridge)]
            m.add(tri if sgn > 0 else tri[::-1], facade, grp)
        add_prism_soffit(m, P, s0, s1, w0, w1, oh, z_eave)
    elif r.type == "hip":
        S0, S1, W0, W1 = s0 - oh, s1 + oh, w0 - oh, w1 + oh
        H = half + oh
        z_r = z_eave + pitch * H
        rs0, rs1 = S0 + H, S1 - H
        if rs1 < rs0:  # square-ish: pyramid
            rs0 = rs1 = (S0 + S1) / 2
        wcm = (W0 + W1) / 2
        m.add([P(S0, W0, z_eave), P(S1, W0, z_eave), P(rs1, wcm, z_r), P(rs0, wcm, z_r)], mat, grp)
        m.add([P(S1, W1, z_eave), P(S0, W1, z_eave), P(rs0, wcm, z_r), P(rs1, wcm, z_r)], mat, grp)
        m.add([P(S1, W0, z_eave), P(S1, W1, z_eave), P(rs1, wcm, z_r)], mat, grp)
        m.add([P(S0, W1, z_eave), P(S0, W0, z_eave), P(rs0, wcm, z_r)], mat, grp)
        add_prism_soffit(m, P, s0, s1, w0, w1, oh, z_eave)
    else:
        raise ValueError(f"unknown roof type {r.type!r}")


def add_prism_soffit(m, P, s0, s1, w0, w1, oh, z):
    if oh <= 0:
        return
    for a, b_, c, d in (((s0 - oh, w0 - oh), (s1 + oh, w0 - oh), (s1 + oh, w0), (s0 - oh, w0)),
                        ((s0 - oh, w1), (s1 + oh, w1), (s1 + oh, w1 + oh), (s0 - oh, w1 + oh)),
                        ((s0 - oh, w0), (s0, w0), (s0, w1), (s0 - oh, w1)),
                        ((s1, w0), (s1 + oh, w0), (s1 + oh, w1), (s1, w1))):
        m.add([P(*a, z), P(*d, z), P(*c, z), P(*b_, z)], "frame_white", "roof")


def resolve_finish_floor(project: Project, ctx: SiteContext | None) -> float:
    b = project.building
    if b.finish_floor_elev is not None:
        return b.finish_floor_elev
    if ctx is None:
        return 0.15
    fp = b.to_site(b.footprint())
    zs = [ctx.ground_z(x, y) for x, y in fp.exterior.coords]
    return max(zs) + 0.15  # 6" above highest adjacent grade (IRC R401.3 drainage)


def building_model(project: Project, ctx: SiteContext | None, m: Model | None = None) -> Model:
    m = m or Model()
    b = project.building
    ff = resolve_finish_floor(project, ctx)
    mats = b.materials
    for lv in b.levels:
        z = ff + lv.elevation
        # slab bears inside the exterior wall (edge hidden), like platform framing
        inner = lv.footprint().buffer(-max(w.thickness for w in lv.walls if w.exterior) + 0.01,
                                      join_style="mitre")
        i = b.levels.index(lv)
        if i > 0 and b.levels[i - 1].stairs:  # floor opening over stairs from below
            from .drawings.plans import stair_rect
            for st in b.levels[i - 1].stairs:
                inner = inner.difference(stair_rect(st).buffer(0.05, join_style="mitre"))
        add_prism(m, b.to_site(inner), z - lv.slab, z, "slab", f"{lv.name}/slab")
        for st in lv.stairs:  # stepped stair mass
            ang = math.radians(st.direction_deg)
            u = np.array([math.cos(ang), math.sin(ang)])
            nn = np.array([-u[1], u[0]])
            A = _xf(b, st.start)
            for k in range(st.risers - 1):
                add_box_along(m, A, _rot(b, u), _rot(b, nn), k * st.tread, (k + 1) * st.tread,
                              0, st.width, z, z + (k + 1) * st.riser, "wood_cedar", f"{lv.name}/stairs")
        for w in lv.walls:
            add_wall(m, b, lv, w, z, mats)
    z_tow = ff + b.top_of_wall()
    roof_faces(m, b, z_tow)
    # foundation skirt down to grade
    if ctx is not None:
        fp = b.to_site(b.footprint())
        low = min(ctx.ground_z(x, y) for x, y in fp.exterior.coords)
        if ff - b.levels[0].slab > low:
            add_prism(m, fp, low - 0.3, ff - b.levels[0].slab, "concrete", "foundation")
    return m


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

def terrain_model(ctx: SiteContext, m: Model, exclude=None) -> None:
    r = ctx.radius_m
    pts = MultiPoint([(gx, gy) for gx, gy, _ in (ctx.elevation or {}).get("grid", [])] or
                     [(-r, -r), (r, -r), (r, r), (-r, r)])
    # densify with a regular 25 m lattice so draping follows interpolated surface
    step = max(10.0, r / 12)
    xs = np.arange(-r, r + 1e-6, step)
    lattice = [(x, y) for x in xs for y in xs]
    from shapely.ops import triangulate as dtri
    allpts = MultiPoint(list(pts.geoms) + [Point(p) for p in lattice])
    for t in dtri(allpts):
        c = np.asarray(t.exterior.coords)[:3]
        if (c[1, 0] - c[0, 0]) * (c[2, 1] - c[0, 1]) - (c[2, 0] - c[0, 0]) * (c[1, 1] - c[0, 1]) < 0:
            c = c[::-1]
        if exclude is not None and exclude.contains(Polygon(c).centroid):
            pass
        m.add([[x, y, ctx.ground_z(x, y) - 0.02] for x, y in c], "terrain", "terrain")


def _drape(m: Model, ctx: SiteContext, poly: Polygon, lift: float, mat: str, group: str,
           max_edge: float = 15.0):
    poly = poly.segmentize(max_edge) if hasattr(poly, "segmentize") else poly
    for tri in triangulate(poly):
        m.add([[x, y, ctx.ground_z(x, y) + lift] for x, y in tri], mat, group)


def context_model(ctx: SiteContext, project: Project | None = None, m: Model | None = None,
                  terrain: bool = True) -> Model:
    m = m or Model()
    site_fp = None
    if project is not None:
        b = project.building
        site_fp = b.to_site(b.footprint()).buffer(0.5)
    if terrain:
        terrain_model(ctx, m)
    clip = Point(0, 0).buffer(ctx.radius_m)
    for f in ctx.of("green"):
        g = f.geom.intersection(clip)
        if not g.is_empty and g.geom_type in ("Polygon", "MultiPolygon"):
            _drape(m, ctx, g, 0.02, "grass", "context/green")
    for f in ctx.of("water"):
        g = f.geom.intersection(clip)
        if not g.is_empty and g.geom_type in ("Polygon", "MultiPolygon"):
            _drape(m, ctx, g, 0.03, "water", "context/water")
    for r, poly in road_polygons(ctx):
        g = poly.intersection(clip)
        if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        hw = r.props.get("highway")
        mat = "sidewalk" if hw in ("footway", "path", "pedestrian", "cycleway", "steps") else "asphalt"
        _drape(m, ctx, g, 0.05 if mat == "asphalt" else 0.08, mat, "context/roads")
    for f in ctx.of("building") + ctx.of("building_part"):
        g = f.geom
        if not g.intersects(clip):
            continue
        if site_fp is not None and g.intersects(site_fp) and f.props.get("demolish", True):
            continue  # existing building on our footprint: assume demolition (flagged by checks)
        gz = f.props.get("ground_z", 0.0)
        if f.props.get("roof_grids"):
            lidar_building(m, f, ctx)
            continue
        add_prism(m, g, gz + f.base - (0.5 if f.base == 0 else 0), gz + f.height, "context",
                  "context/buildings", top_mat="context_roof")
    for t in ctx.of("tree"):
        x, y = t.geom.x, t.geom.y
        if site_fp is not None and site_fp.contains(t.geom):
            continue
        gz = ctx.ground_z(x, y)
        h = t.props.get("height_m", 8.0)
        cr = t.props.get("crown_m", 5.0) / 2
        add_box_along(m, (x, y), (1, 0), (0, 1), -0.15, 0.15, -0.15, 0.15, gz, gz + h * 0.5,
                      "tree_trunk", "context/trees")
        add_sphere(m, x, y, gz + h - cr, cr, "tree_crown", "context/trees", squash=1.1)
    return m


def _grid_sampler(r: dict, datum: float):
    Z = np.asarray(r["z"], float).reshape(r["ny"], r["nx"])
    x0, y0, c = r["x0"], r["y0"], r["cell"]

    def f(x, y):
        fx = min(max((x - x0) / c, 0), r["nx"] - 1.0001)
        fy = min(max((y - y0) / c, 0), r["ny"] - 1.0001)
        i, j = int(fx), int(fy)
        tx, ty = fx - i, fy - j
        return float(Z[j, i] * (1 - tx) * (1 - ty) + Z[j, i + 1] * tx * (1 - ty)
                     + Z[j + 1, i] * (1 - tx) * ty + Z[j + 1, i + 1] * tx * ty) - datum
    return f


def lidar_building(m: Model, f, ctx: SiteContext, mat: str = "context",
                   roof_mat: str = "context_roof", group: str = "context/buildings"):
    """Building from measured LiDAR roof grid: true roof surface + walls to grade."""
    from shapely.ops import triangulate as dtri

    g = f.geom
    parts = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
    datum = f.props.get("z_datum", 0.0)
    for r in f.props["roof_grids"]:
        poly = parts[r.get("part", 0)]
        zf = _grid_sampler(r, datum)
        c = r["cell"]
        ring_pts = []
        for ring in [poly.exterior, *poly.interiors]:
            ring_pts += list(ring.segmentize(c).coords)[:-1]
        inner = poly.buffer(-c * 0.5)
        grid_pts = []
        if not inner.is_empty:
            import shapely
            xs = r["x0"] + np.arange(r["nx"]) * c
            ys = r["y0"] + np.arange(r["ny"]) * c
            X, Y = np.meshgrid(xs, ys)
            msk = shapely.contains_xy(inner, X.ravel(), Y.ravel())
            grid_pts = list(zip(X.ravel()[msk], Y.ravel()[msk]))
        tris = dtri(MultiPoint(ring_pts + grid_pts))
        for t in tris:
            if not poly.contains(t.centroid):
                continue
            cc = np.asarray(t.exterior.coords)[:3]
            if (cc[1, 0] - cc[0, 0]) * (cc[2, 1] - cc[0, 1]) - (cc[2, 0] - cc[0, 0]) * (cc[1, 1] - cc[0, 1]) < 0:
                cc = cc[::-1]
            m.add([[x, y, zf(x, y)] for x, y in cc], roof_mat, group)
        for ring in [poly.exterior, *poly.interiors]:
            co = list(ring.segmentize(c).coords)
            for (ax, ay), (bx, by) in zip(co[:-1], co[1:]):
                ga, gb = ctx.ground_z(ax, ay) - 0.3, ctx.ground_z(bx, by) - 0.3
                m.add([[ax, ay, ga], [bx, by, gb], [bx, by, zf(bx, by)], [ax, ay, zf(ax, ay)]], mat, group)


def full_model(project: Project, ctx: SiteContext | None) -> Model:
    m = Model()
    if ctx is not None:
        context_model(ctx, project, m)
    building_model(project, ctx, m)
    return m


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_obj(m: Model, path: str | Path, overrides: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mtl = path.with_suffix(".mtl")
    used = sorted({f.material for f in m.faces})
    with mtl.open("w") as fh:
        for name in used:
            p = materials.resolve(name, overrides)
            r, g, b = p["base"]
            fh.write(f"newmtl {name}\nKd {r:.4f} {g:.4f} {b:.4f}\nNs {int((1 - p.get('rough', .5)) * 500)}\n")
            if p.get("transmission"):
                fh.write(f"d {1 - 0.9 * p['transmission']:.3f}\nNi {p.get('ior', 1.5)}\n")
            fh.write("\n")
    by_group: dict[str, list[Face]] = {}
    for f in m.faces:
        by_group.setdefault(f.group, []).append(f)
    vi = 1
    with path.open("w") as fh:
        fh.write(f"# archkit model — metres, Z up, site-local frame\nmtllib {mtl.name}\n")
        for grp, faces in by_group.items():
            fh.write(f"o {grp.replace(' ', '_')}\n")
            cur = None
            for f in faces:
                if f.material != cur:
                    fh.write(f"usemtl {f.material}\n")
                    cur = f.material
                for v in f.verts:
                    fh.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
                fh.write("f " + " ".join(str(vi + i) for i in range(len(f.verts))) + "\n")
                vi += len(f.verts)
    return path
