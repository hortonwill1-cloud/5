"""Exterior elevations and building sections, projected from the 3D model.

Because they are generated from the same model as the plans and renders,
the drawing set is internally coordinated by construction.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from .. import materials
from ..model3d import Model, building_model, resolve_finish_floor
from ..site import SiteContext
from ..spec import Project
from ..units import fmt_len
from .core import Drawing


def _hex(rgb) -> str:
    return "#" + "".join(f"{int(max(0, min(1, c)) * 255):02x}" for c in rgb)


def _tint(mat: str, overrides, shade: float) -> str:
    base = materials.srgb(materials.resolve(mat, overrides)["base"])
    # architectural rendering: wash towards white, keep material hue
    c = [0.55 + 0.45 * v for v in base]
    return _hex([v * shade for v in c])


def _face_normal(v: np.ndarray) -> np.ndarray:
    n = np.zeros(3)
    for i in range(len(v)):
        a, b = v[i], v[(i + 1) % len(v)]
        n += np.cross(a, b)
    ln = np.linalg.norm(n)
    return n / ln if ln > 0 else n


def _clip_plane(v: np.ndarray, d: np.ndarray, d0: float) -> np.ndarray:
    """Keep part of polygon with dot(p, d) >= d0 (Sutherland-Hodgman, one plane)."""
    out = []
    n = len(v)
    for i in range(n):
        a, b = v[i], v[(i + 1) % n]
        da, db = a @ d - d0, b @ d - d0
        if da >= 0:
            out.append(a)
        if (da >= 0) != (db >= 0):
            t = da / (da - db)
            out.append(a + (b - a) * t)
    return np.asarray(out)


def _project_model(dr: Drawing, m: Model, right: np.ndarray, view: np.ndarray, overrides,
                   origin: np.ndarray, beyond_of: float | None = None, light=None):
    """Painter's-algorithm projection of model faces into the drawing's (s, z) plane."""
    light = light if light is not None else np.array([-0.4, -0.6, 0.7]) / np.linalg.norm([-0.4, -0.6, 0.7])
    items = []
    for f in m.faces:
        v = f.verts
        if beyond_of is not None:
            v = _clip_plane(v, view, beyond_of)
            if len(v) < 3:
                continue
        nrm = _face_normal(v)
        if abs(nrm @ view) < 0.03 and abs(nrm[2]) < 0.97:
            continue  # edge-on vertical face
        if nrm @ view > 0.03:
            continue  # back-facing
        s = (v - origin) @ right
        z = v[:, 2]
        if np.ptp(s) < 1e-4 or np.ptp(z) < 1e-4:
            continue
        depth = float(((v - origin) @ view).mean())
        shade = 0.82 + 0.18 * max(0.0, float(-nrm @ view)) * (0.6 + 0.4 * max(0, nrm @ light))
        items.append((depth, np.c_[s, z], f.material, f.group, shade))
    items.sort(key=lambda t: -t[0])
    for depth, pts, mat, grp, shade in items:
        if "openings" in grp:
            dr.poly(pts, "A-GLAZ", fill=_tint(mat, overrides, shade), lw=0.13, z=2)
        elif grp == "roof":
            dr.poly(pts, "A-ROOF", fill=_tint(mat, overrides, shade * 0.95), lw=0.18, z=2)
        elif grp.startswith("context"):
            dr.poly(pts, "A-ELEV-BEYD", fill="#f2f2f2", lw=0.13, z=1)
        else:
            col = _tint(mat, overrides, shade)
            dr.poly(pts, "A-ELEV-FILL", fill=col, edge=False, z=2)
    return items


def elevation(project: Project, ctx: SiteContext | None, facing: str) -> tuple[Drawing, str]:
    """facing: 'south'|'north'|'east'|'west' — compass side of the BUILDING being drawn
    (in building frame: south = -y side). Returns drawing and title."""
    b = project.building
    rot = math.radians(b.rotation_deg)
    dirs = {"south": (0, -1), "north": (0, 1), "east": (1, 0), "west": (-1, 0)}
    ox, oy = dirs[facing]
    # outward normal of that facade in site frame; we look opposite to it
    nx = ox * math.cos(rot) - oy * math.sin(rot)
    ny = ox * math.sin(rot) + oy * math.cos(rot)
    view = np.array([-nx, -ny, 0.0])
    right = np.cross(view, [0, 0, 1])  # screen-right when looking along `view`, z up
    m = building_model(project, ctx)
    fp = b.to_site(b.footprint())
    origin = np.array([fp.centroid.x, fp.centroid.y, 0.0])
    dr = Drawing()
    items = _project_model(dr, m, right, view, b.materials, origin)
    # silhouette
    polys = [Polygon(p) for _, p, _, _, _ in items]
    sil = unary_union([p.buffer(0) for p in polys if p.is_valid or p.buffer(0).area > 0])
    dr.shape(sil, "A-ELEV-OTLN", fill="none", z=4)
    smin, zmin, smax, zmax = sil.bounds
    ff = resolve_finish_floor(project, ctx)
    # grade line (terrain profile along facade)
    if ctx is not None:
        ss = np.linspace(smin - 4, smax + 4, 40)
        gpts = []
        face_off = np.array([nx, ny, 0]) * 2.0
        for s in ss:
            p = origin + right * s + face_off
            gpts.append((s, ctx.ground_z(p[0], p[1])))
    else:
        gpts = [(smin - 4, 0.0), (smax + 4, 0.0)]
    gp = np.array(gpts)
    earth = Polygon(list(gp) + [(gp[-1, 0], gp[:, 1].min() - 0.6), (gp[0, 0], gp[:, 1].min() - 0.6)])
    dr.shape(earth, "G-GRND-PATT", fill="#e9e3df", edge=False, z=3)
    dr.poly(gp, "G-GRND", closed=False, z=5)
    _datums(dr, project, ff, smin, smax)
    return dr, f"{facing.upper()} ELEVATION"


def _datums(dr: Drawing, project: Project, ff: float, smin: float, smax: float):
    b = project.building
    sysu = project.system
    x0 = smax + 1.0
    marks = [(lv.name, ff + lv.elevation) for lv in b.levels]
    marks.append(("T.O. WALL", ff + b.top_of_wall()))
    if b.roof.type != "flat":
        marks.append(("RIDGE" if b.roof.type != "shed" else "HIGH POINT", ff + b.roof_peak()))
    elif b.roof.parapet > 0:
        marks.append(("T.O. PARAPET", ff + b.roof_peak()))
    for name, z in marks:
        dr.line((smin - 1.0, z), (x0 + 2.4, z), "A-ANNO-LEVL", z=6)
        tri = [(x0 + 0.5, z), (x0 + 0.25, z + 0.3), (x0 + 0.75, z + 0.3)]
        dr.poly(tri, "A-ANNO-SYMB", fill="#000000", z=6)
        rel = z - ff
        dr.text((x0 + 1.0, z + 0.12), name.upper(), h_mm=2.0, ha="left", va="bottom", weight="bold", z=7)
        lab = ("+" if rel >= 0 else "") + fmt_len(rel, sysu) if sysu == "imperial" else f"{rel:+.3f}"
        dr.text((x0 + 1.0, z - 0.1), lab, h_mm=1.8, ha="left", va="top", z=7)
    if project.zoning.get("max_height"):
        # zoning height plane relative to finish floor (approx. grade)
        zh = project.zoning["max_height"]
        dr.line((smin - 1.0, zh), (x0 + 2.4, zh), "C-PROP-SBCK", z=6)
        dr.text((smin - 0.8, zh + 0.12), "MAX. HEIGHT (ZONING)", "A-ANNO-TEXT", h_mm=1.8, ha="left",
                va="bottom", z=7)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section(project: Project, ctx: SiteContext | None, axis: str = "x", at: float | None = None,
            look: int = 1) -> tuple[Drawing, str]:
    """Cut a section along building-frame `axis` ('x' cut line runs along x) at coordinate `at`.
    look = +1 looks toward +normal of the cut line."""
    b = project.building
    fp_l = b.footprint()
    minx, miny, maxx, maxy = fp_l.bounds
    if axis == "x":
        at = (miny + maxy) / 2 if at is None else at
        a_l, c_l = (minx - 10, at), (maxx + 10, at)
        nrm_l = np.array([0, look])
    else:
        at = (minx + maxx) / 2 if at is None else at
        a_l, c_l = (at, miny - 10), (at, maxy + 10)
        nrm_l = np.array([-look, 0])
    A = np.array(b.pt_to_site(*a_l))
    C = np.array(b.pt_to_site(*c_l))
    u = (C - A) / np.linalg.norm(C - A)
    rot = math.radians(b.rotation_deg)
    n = np.array([nrm_l[0] * math.cos(rot) - nrm_l[1] * math.sin(rot),
                  nrm_l[0] * math.sin(rot) + nrm_l[1] * math.cos(rot)])
    view = np.array([n[0], n[1], 0.0])
    right = np.cross(view, [0, 0, 1])
    origin = np.array([A[0], A[1], 0.0]) + np.array([u[0], u[1], 0]) * 10
    d0 = float(origin @ view)
    m = building_model(project, ctx)
    dr = Drawing()
    _project_model(dr, m, right, view, b.materials, origin, beyond_of=d0)
    ff = resolve_finish_floor(project, ctx)
    cut_line_site = LineString([A, C])
    s_of = lambda p: float((np.array([p[0], p[1], 0]) - origin) @ right)  # noqa: E731
    poche = []
    from .plans import opening_box, wall_polygon
    for lv in b.levels:
        z = ff + lv.elevation
        # slab
        slab = b.to_site(lv.footprint()).intersection(cut_line_site)
        for seg in getattr(slab, "geoms", [slab]):
            if seg.is_empty or seg.length < 1e-3:
                continue
            s_a, s_b = sorted(s_of(p) for p in seg.coords)
            poche.append(Polygon([(s_a, z - lv.slab), (s_b, z - lv.slab), (s_b, z), (s_a, z)]))
        for w in lv.walls:
            wp = b.to_site(wall_polygon(w))
            hit = wp.intersection(cut_line_site)
            for seg in getattr(hit, "geoms", [hit]):
                if seg.is_empty or seg.length < 1e-3:
                    continue
                s_a, s_b = sorted(s_of(p) for p in seg.coords)
                h = w.height or lv.floor_to_floor
                # openings crossing the cut
                gaps = []
                for o in lv.openings:
                    if o.wall != w.id:
                        continue
                    if b.to_site(opening_box(w, o.segment, o, 0.0)).intersects(seg):
                        gaps.append((o.sill, o.sill + o.height))
                zs = [0.0]
                for g0, g1 in sorted(gaps):
                    poche.append(Polygon([(s_a, z + zs[-1]), (s_b, z + zs[-1]), (s_b, z + g0), (s_a, z + g0)]))
                    zs.append(g1)
                poche.append(Polygon([(s_a, z + zs[-1]), (s_b, z + zs[-1]), (s_b, z + h), (s_a, z + h)]))
    # roof cut: intersect roof faces with plane
    for f in m.faces:
        if f.group != "roof":
            continue
        v = f.verts
        dv = v @ view - d0
        pts = []
        for i in range(len(v)):
            p, q = v[i], v[(i + 1) % len(v)]
            if (dv[i] >= 0) != (dv[(i + 1) % len(v)] >= 0):
                t = dv[i] / (dv[i] - dv[(i + 1) % len(v)])
                r = p + (q - p) * t
                pts.append((float((r - origin) @ right), float(r[2])))
        if len(pts) == 2:
            ls = LineString(pts)
            poche.append(ls.buffer(max(b.roof.thickness, 0.15) / 2, cap_style="flat"))
    cut = unary_union([p for p in poche if not p.is_empty])
    dr.shape(cut, "A-SECT-PATT", fill="#1f1f1f", z=6, lw=1.0)
    # ground
    smin, _, smax, _ = cut.bounds
    ss = np.linspace(smin - 5, smax + 5, 50)
    gp = np.array([(s, (ctx.ground_z(*(origin + right * s)[:2]) if ctx else 0.0)) for s in ss])
    earth = Polygon(list(gp) + [(gp[-1, 0], gp[:, 1].min() - 1.2), (gp[0, 0], gp[:, 1].min() - 1.2)])
    earth = earth.difference(cut)
    dr.shape(earth, "G-GRND-PATT", fill="#d7ccc8", edge=False, z=3)
    dr.poly(gp, "G-GRND", closed=False, z=7)
    _datums(dr, project, ff, smin, smax)
    # room names cut through
    for lv in b.levels:
        for r in lv.rooms:
            hit = b.to_site(r.poly).intersection(cut_line_site)
            if not hit.is_empty and hit.length > 1.0:
                s_mid = s_of(hit.interpolate(0.5, normalized=True).coords[0])
                dr.text((s_mid, ff + lv.elevation + lv.ceiling / 2), r.name.upper(), h_mm=2.2, z=8)
                dr.line((s_mid - 0.3, ff + lv.elevation + lv.ceiling), (s_mid + 0.3, ff + lv.elevation + lv.ceiling),
                        "A-ANNO-SYMB", z=8)
    return dr, f"BUILDING SECTION {'A' if axis == 'x' else 'B'}"
