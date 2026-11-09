"""Site plan and floor plan drawings."""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from ..checks import setback_envelope
from ..site import SiteContext, road_polygons
from ..spec import Level, Opening, Project, Wall
from ..units import fmt_area, fmt_len
from .core import Drawing


# ---------------------------------------------------------------------------
# Site plan (true north up, site frame)
# ---------------------------------------------------------------------------

def site_plan(project: Project, ctx: SiteContext | None, extent: float | None = None) -> Drawing:
    dr = Drawing()
    b = project.building
    sysu = project.system
    fp = b.to_site(b.footprint())
    focus = project.parcel if project.parcel is not None else fp
    if extent is None:
        minx, miny, maxx, maxy = focus.bounds
        extent = max(maxx - minx, maxy - miny) * 0.5 + 25
    cx, cy = focus.centroid.x, focus.centroid.y
    clip = box(cx - extent, cy - extent, cx + extent, cy + extent)
    dr.poly(list(clip.exterior.coords), "G-ANNO-TTLB", closed=True, lw=0.0, edge=False)

    if ctx is not None:
        # terrain contours
        if ctx.elevation and ctx.elevation.get("grid"):
            _contours(dr, ctx, clip, project)
        for f in ctx.of("green"):
            dr.shape(f.geom.intersection(clip), "L-PLNT-PATT", fill="#eef4ea", edge=False, z=0)
        for f in ctx.of("water"):
            dr.shape(f.geom.intersection(clip), "C-WATR", fill="#d6e9f5", z=0)
        rp = [(r, p.intersection(clip)) for r, p in road_polygons(ctx)]
        pav = unary_union([p for r, p in rp if r.props.get("highway") not in
                           ("footway", "path", "cycleway", "steps")] or [Point(0, 0).buffer(0)])
        walk = unary_union([p for r, p in rp if r.props.get("highway") in
                            ("footway", "path", "cycleway", "steps")] or [Point(0, 0).buffer(0)])
        dr.shape(walk, "C-ROAD", fill="#f4f4f4", z=1, lw=0.13)
        dr.shape(pav, "C-ROAD-PATT", fill="#e4e4e4", edge=True, z=1)
        named = set()
        for r, _ in rp:
            g = r.geom.intersection(clip)
            if g.is_empty:
                continue
            if r.props.get("highway") not in ("footway", "path", "cycleway", "steps", "service"):
                dr.shape(g, "C-ROAD-CNTR", z=2)
            nm = r.props.get("name")
            if nm and nm not in named and g.length > 30:
                named.add(nm)
                line = g if g.geom_type == "LineString" else max(g.geoms, key=lambda x: x.length)
                p = line.interpolate(0.5, normalized=True)
                p2 = line.interpolate(min(1.0, 0.5 + 2 / max(line.length, 1)), normalized=True)
                ang = math.degrees(math.atan2(p2.y - p.y, p2.x - p.x))
                if ang > 90 or ang < -90:
                    ang += 180
                dr.text((p.x, p.y), nm.upper(), h_mm=2.5, rot=ang, weight="bold", z=6)
        for f in ctx.of("building"):
            g = f.geom.intersection(clip)
            if g.is_empty or g.intersects(fp):
                continue
            dr.shape(g, "C-BLDG-EXST-PATT", fill="#ececec", edge=True, z=3, lw=0.35)
            if g.area > 60:
                c = g.representative_point()
                hs = f.props.get("height_source", "")
                lab = f"{fmt_len(f.height, sysu)}" + ("*" if "ESTIMATED" in hs else "")
                num = f.props.get("addr:housenumber")
                dr.text((c.x, c.y), f"{num}\n{lab}" if num else lab, h_mm=1.8, z=6)
        for t in ctx.of("tree"):
            if clip.contains(t.geom):
                r = t.props.get("crown_m", 5) / 2
                dr.circle((t.geom.x, t.geom.y), r, "L-PLNT", fill="#dfeedd", z=4)
                dr.circle((t.geom.x, t.geom.y), 0.15, "L-PLNT", z=4)

    if project.parcel is not None:
        parcel = project.parcel
        dr.shape(parcel, "C-PROP", fill="none", z=7)
        if "setbacks" in project.zoning:
            env, labels = setback_envelope(project, ctx)
            dr.shape(env, "C-PROP-SBCK", fill="none", z=7)
            # lot line bearings & distances
        coords = list(parcel.exterior.coords)
        ccw = parcel.exterior.is_ccw
        for i, (a, c) in enumerate(zip(coords[:-1], coords[1:])):
            L = math.dist(a, c)
            off = -2.5 if ccw else 2.5   # dim offset is to the LEFT of a->c; outside = right for CCW
            dr.dim(a, c, off, f"{_bearing(a, c)}  {fmt_len(L, sysu)}")
        # building to lot-line dimensions (perpendicular)
        for i, (a, c) in enumerate(zip(coords[:-1], coords[1:])):
            edge = LineString([a, c])
            from shapely.ops import nearest_points
            p_fp, p_edge = nearest_points(fp, edge)
            d = p_fp.distance(p_edge)
            if d > 0.3:
                dr.line((p_fp.x, p_fp.y), (p_edge.x, p_edge.y), "A-ANNO-DIMS", z=8)
                mid = ((p_fp.x + p_edge.x) / 2, (p_fp.y + p_edge.y) / 2)
                ang = math.degrees(math.atan2(p_edge.y - p_fp.y, p_edge.x - p_fp.x))
                if ang > 90 or ang < -90:
                    ang += 180
                dr.text(mid, fmt_len(d, sysu), h_mm=2.0, rot=ang, va="bottom", z=8)
    # proposed building
    dr.shape(fp, "A-BLDG-PROP-PATT", fill="#9e9e9e", edge=True, z=9, lw=0.7)
    if b.roof.overhang > 0:
        dr.shape(b.to_site(b.footprint().envelope.buffer(b.roof.overhang, join_style="mitre")),
                 "A-ROOF-OTLN", fill="none", z=9)
    c = fp.representative_point()
    from ..model3d import resolve_finish_floor
    ff = resolve_finish_floor(project, ctx)
    z_abs = (ctx.elevation["origin_m"] if ctx and ctx.elevation else 0.0) + ff
    elev_str = (f"{z_abs / 0.3048:,.2f}'" if sysu == "imperial" else f"{z_abs:,.2f} m")
    dr.text((c.x, c.y), f"PROPOSED\n{project.building.occupancy}  {len(b.levels)} STORY\n"
                        f"FF = {elev_str}", h_mm=2.4, weight="bold", z=10)
    # site origin benchmark
    dr.circle((0, 0), 0.6, "A-ANNO-SYMB", z=10)
    dr.line((-1.2, 0), (1.2, 0), "A-ANNO-SYMB", z=10)
    dr.line((0, -1.2), (0, 1.2), "A-ANNO-SYMB", z=10)
    if project.lat is not None:
        dr.text((1.5, -1.5), f"ORIGIN {project.lat:.6f}, {project.lon:.6f}", h_mm=1.6, ha="left",
                va="top", z=10)
    return dr


def _bearing(a, c) -> str:
    """Quadrant bearing (grid = true north at origin), e.g. N 12°34'56\" E."""
    dx, dy = c[0] - a[0], c[1] - a[1]
    az = math.degrees(math.atan2(dx, dy)) % 360
    if az <= 90:
        ns, ang, ew = "N", az, "E"
    elif az <= 180:
        ns, ang, ew = "S", 180 - az, "E"
    elif az <= 270:
        ns, ang, ew = "S", az - 180, "W"
    else:
        ns, ang, ew = "N", 360 - az, "W"
    d = int(ang)
    m = int((ang - d) * 60)
    s = round(((ang - d) * 60 - m) * 60)
    if s == 60:
        m, s = m + 1, 0
    return f"{ns} {d:02d}°{m:02d}'{s:02d}\" {ew}"


def _contours(dr: Drawing, ctx: SiteContext, clip: Polygon, project: Project):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    minx, miny, maxx, maxy = clip.bounds
    n = 60
    xs = np.linspace(minx, maxx, n)
    ys = np.linspace(miny, maxy, n)
    z0 = ctx.elevation["origin_m"]
    Z = np.array([[ctx.ground_z(x, y) + z0 for x in xs] for y in ys])
    imperial = project.system == "imperial"
    unit = 0.3048 if imperial else 1.0
    Zu = Z / unit
    minor = 1.0 if (Zu.max() - Zu.min()) < 30 else 2.0
    major = minor * 5
    lv = np.arange(math.floor(Zu.min() / minor) * minor, Zu.max() + minor, minor)
    if len(lv) < 2:
        return
    fig = plt.figure()
    cs = plt.contour(xs, ys, Zu, levels=lv)
    for level, segs in zip(cs.levels, cs.allsegs):
        is_major = abs(level / major - round(level / major)) < 1e-6
        for seg in segs:
            if len(seg) < 2:
                continue
            dr.poly(seg, "C-TOPO-MAJR" if is_major else "C-TOPO-MINR", closed=False, z=0)
            if is_major and len(seg) > 10:
                p = seg[len(seg) // 2]
                dr.text((p[0], p[1]), f"{level:g}", h_mm=1.6, z=0)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Floor plans (building frame)
# ---------------------------------------------------------------------------

def _seg_frame(a, c):
    a, c = np.asarray(a, float), np.asarray(c, float)
    L = float(np.linalg.norm(c - a))
    u = (c - a) / L
    n = np.array([-u[1], u[0]])
    return a, u, n, L


def wall_polygon(w: Wall) -> Polygon:
    line = LineString(list(w.path) + ([w.path[0]] if w.closed else []))
    if w.closed:
        return Polygon(w.path).buffer(w.thickness / 2, join_style="mitre").difference(
            Polygon(w.path).buffer(-w.thickness / 2, join_style="mitre"))
    return line.buffer(w.thickness / 2, cap_style="flat", join_style="mitre")


def opening_box(w: Wall, seg: int, o: Opening, extra: float = 0.01) -> Polygon:
    a, u, n, _ = _seg_frame(*w.segments()[seg])
    t = w.thickness / 2 + extra
    p = [a + u * o.offset + n * t, a + u * (o.offset + o.width) + n * t,
         a + u * (o.offset + o.width) - n * t, a + u * o.offset - n * t]
    return Polygon(p)


def floor_plan(project: Project, level: Level, cut_height: float = 1.2) -> Drawing:
    dr = Drawing()
    sysu = project.system
    walls_by_id = {w.id: w for w in level.walls}
    solid = unary_union([wall_polygon(w) for w in level.walls])
    cuts = []
    for o in level.openings:
        w = walls_by_id.get(o.wall)
        if w is None:
            continue
        if o.sill < cut_height < o.sill + o.height:  # opening is cut at plan cut plane
            cuts.append(opening_box(w, o.segment, o))
    cut_walls = solid.difference(unary_union(cuts)) if cuts else solid
    dr.shape(cut_walls, "A-WALL-PATT", fill="#4a4a4a", edge=True, z=5, lw=0.7)

    # rooms
    for r in level.rooms:
        p = r.poly
        c = p.representative_point() if not p.contains(p.centroid) else p.centroid
        dr.text((c.x, c.y + 0.35), r.name.upper(), "A-FLOR-IDEN", h_mm=2.6, weight="bold", z=6)
        dr.text((c.x, c.y - 0.15), r.number, "A-FLOR-IDEN", h_mm=2.0, z=6)
        dr.text((c.x, c.y - 0.6), fmt_area(p.area, sysu), "A-FLOR-IDEN", h_mm=1.8, z=6)

    # openings
    for o in level.openings:
        w = walls_by_id.get(o.wall)
        if w is None:
            continue
        a, u, n, L = _seg_frame(*w.segments()[o.segment])
        t = w.thickness / 2
        s0, s1 = o.offset, o.offset + o.width
        P = lambda s, d: tuple(a + u * s + n * d)  # noqa: E731
        # jamb lines
        dr.line(P(s0, -t), P(s0, t), "A-WALL", z=6)
        dr.line(P(s1, -t), P(s1, t), "A-WALL", z=6)
        if o.kind in ("window", "storefront"):
            dr.line(P(s0, -t), P(s1, -t), "A-GLAZ-SILL", z=6)
            dr.line(P(s0, t), P(s1, t), "A-GLAZ-SILL", z=6)
            dr.line(P(s0, -0.02), P(s1, -0.02), "A-GLAZ", z=6)
            dr.line(P(s0, 0.02), P(s1, 0.02), "A-GLAZ", z=6)
            tag_pt = P((s0 + s1) / 2, (t + 0.55) * _outside_sign(level, w, o, a, u, n))
            dr.text(tag_pt, o.tag or "W", "A-ANNO-SYMB", h_mm=1.8, z=7, box=True)
        elif o.kind == "garage_door":
            dr.line(P(s0, 0), P(s1, 0), "A-DOOR", z=6)
            inside = -_outside_sign(level, w, o, a, u, n)
            dr.poly([P(s0, inside * t), P(s1, inside * t), P(s1, inside * (t + 2.2)),
                     P(s0, inside * (t + 2.2))], "A-FLOR-OVHD", closed=True, z=6)
        else:
            _door(dr, o, P, t, level, w, a, u, n)
            tag_pt = P((s0 + s1) / 2, (t + 0.45) * -_outside_sign(level, w, o, a, u, n) * -1)
            dr.text(tag_pt, o.tag or "D", "A-ANNO-SYMB", h_mm=1.8, z=7, box=True)

    # stairs
    for s in level.stairs:
        _stair(dr, s, level)

    # exterior dimension strings
    ext = [w for w in level.walls if w.exterior]
    for w in ext:
        for si, (pa, pb) in enumerate(w.segments()):
            a, u, n, L = _seg_frame(pa, pb)
            out_sign = _outward(w, a, u, n, L)
            t = w.thickness / 2
            base_off = out_sign * (t + 1.8)
            # overall (outside face to outside face)
            ea = a - u * t + n * out_sign * t
            eb = a + u * (L + t) + n * out_sign * t
            dr.dim(ea, eb, out_sign * 2.6, fmt_len(L + 2 * t, sysu))
            # openings string: centre-line of openings
            ops = sorted([o for o in level.openings if o.wall == w.id and o.segment == si],
                         key=lambda o: o.offset)
            if ops:
                stops = [ea] + [a + u * (o.offset + o.width / 2) + n * out_sign * t for o in ops] + [eb]
                for p0, p1 in zip(stops[:-1], stops[1:]):
                    dr.dim(p0, p1, out_sign * 1.6, fmt_len(float(np.linalg.norm(p1 - p0)), sysu))
    return dr


def _outward(w: Wall, a, u, n, L) -> int:
    """+1 if the segment normal n points outside the closed exterior wall, else -1."""
    if not w.closed:
        return 1
    poly = Polygon(w.path)
    mid = a + u * L / 2
    return 1 if not poly.contains(Point(*(mid + n * (w.thickness + 0.05)))) else -1


def _outside_sign(level, w, o, a, u, n) -> int:
    seglen = float(np.linalg.norm(np.asarray(w.segments()[o.segment][1]) - a))
    return _outward(w, a, u, n, seglen)


def _door(dr: Drawing, o: Opening, P, t, level, w, a, u, n):
    s0, s1 = o.offset, o.offset + o.width
    out = _outside_sign(level, w, o, a, u, n)
    swing = o.swing
    if swing in ("none", "slide"):
        dr.line(P(s0, 0.03), P((s0 + s1) / 2 + 0.05, 0.03), "A-DOOR", z=6)
        dr.line(P((s0 + s1) / 2 - 0.05, -0.03), P(s1, -0.03), "A-DOOR", z=6)
        return
    side = out if swing.startswith("out") else -out          # which face the leaf opens toward
    hinge_s = s0 if swing.endswith("left") else s1
    free_s = s1 if hinge_s == s0 else s0
    face = side * t
    hinge = np.array(P(hinge_s, face))
    leaf_tip = hinge + (np.asarray(n) * side) * o.width
    dr.line(tuple(hinge), tuple(leaf_tip), "A-DOOR", z=6)
    closed_tip = np.array(P(free_s, face))
    a0 = math.degrees(math.atan2(*(closed_tip - hinge)[::-1]))
    a1 = math.degrees(math.atan2(*(leaf_tip - hinge)[::-1]))
    lo, hi = sorted([a0, a1])
    if hi - lo > 180:
        lo, hi = hi, lo + 360
    dr.arc(tuple(hinge), o.width, lo, hi, "A-DOOR-SWNG", z=6)


def _stair(dr: Drawing, s, level: Level):
    ang = math.radians(s.direction_deg)
    u = np.array([math.cos(ang), math.sin(ang)])
    n = np.array([-u[1], u[0]])
    a = np.asarray(s.start, float)
    treads = s.risers - 1
    run = treads * s.tread
    corners = [a, a + u * run, a + u * run + n * s.width, a + n * s.width]
    dr.poly(corners, "A-FLOR-STRS", closed=True, z=6)
    for i in range(1, treads):
        p0 = a + u * s.tread * i
        dr.line(tuple(p0), tuple(p0 + n * s.width), "A-FLOR-STRS", z=6)
    mid0 = a + n * s.width / 2 + u * s.tread * 0.5
    mid1 = a + n * s.width / 2 + u * (run - s.tread * 0.5)
    dr.line(tuple(mid0), tuple(mid1), "A-FLOR-STRS", z=6)
    head = mid1
    for side in (-1, 1):
        dr.line(tuple(head), tuple(head - u * 0.3 + n * side * 0.15), "A-FLOR-STRS", z=6)
    dr.text(tuple(mid0 - u * 0.35), f"UP {s.risers}R", h_mm=1.8, z=7)
    # break line at cut height (~ 1/3 run)
    k = max(1, int(treads * 0.45))
    p = a + u * s.tread * k
    dr.line(tuple(p + n * -0.05), tuple(p + u * 0.6 + n * (s.width + 0.05)), "A-FLOR-STRS", z=6)
