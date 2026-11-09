"""Shadow study: cast shadows of proposed + existing buildings on the ground plane
for the equinoxes/solstices at 9am, 12pm, 3pm local standard time (common
planning-submission format).  Shadows are exact projections of the 3D massing
along the true solar vector (flat-ground approximation at the site datum).
"""

from __future__ import annotations

import math
from datetime import datetime

from shapely.affinity import translate
from shapely.geometry import MultiPoint, box
from shapely.ops import unary_union

from .drawings.core import Drawing
from .drawings.sheet import IN, Sheet
from .site import SiteContext, road_polygons
from .spec import Project
from .sun import guess_utc_offset, solar_position

DATES = [("MAR 20 (EQUINOX)", 3, 20), ("JUN 21 (SUMMER SOLSTICE)", 6, 21), ("DEC 21 (WINTER SOLSTICE)", 12, 21)]
TIMES = [9, 12, 15]


def prism_shadow(poly, h_base: float, h_top: float, az: float, el: float):
    """Ground shadow of a vertical prism (footprint poly, heights in m above ground)."""
    if el <= 0.5:
        return None
    L = 1.0 / math.tan(math.radians(el))
    dx, dy = -math.sin(math.radians(az)) * L, -math.cos(math.radians(az)) * L
    top = translate(poly, dx * h_top, dy * h_top)
    base = translate(poly, dx * h_base, dy * h_base) if h_base > 0 else poly
    hull_pts = list(top.exterior.coords) + list(base.exterior.coords)
    sweep = MultiPoint(hull_pts).convex_hull if poly.convex_hull.equals(poly) else unary_union(
        [MultiPoint([a, b, (a[0] + dx * h_top, a[1] + dy * h_top), (b[0] + dx * h_top, b[1] + dy * h_top)]).convex_hull
         for a, b in zip(list(poly.exterior.coords)[:-1], list(poly.exterior.coords)[1:])] + [top])
    return sweep.difference(poly) if h_base == 0 else sweep


def shadow_study(project: Project, ctx: SiteContext | None, out, tz: float | None = None, year=2026):
    from matplotlib.backends.backend_pdf import PdfPages

    from .model3d import resolve_finish_floor

    from .sun import local_tz

    if tz is None and not project.timezone:
        tz = project.views[0].utc_offset_h if project.views else guess_utc_offset(project.lon)
    tzinfo = local_tz(project.timezone if tz is None else None, tz)
    b = project.building
    fp = b.to_site(b.footprint())
    ff = resolve_finish_floor(project, ctx)
    h = ff + b.roof_peak() - (ctx.ground_z(fp.centroid.x, fp.centroid.y) if ctx else 0)
    h_eave = ff + b.top_of_wall()
    focus = project.parcel if project.parcel is not None else fp
    c = focus.centroid
    R = 60.0
    clip = box(c.x - R, c.y - R, c.x + R, c.y + R)
    sh = Sheet(project, "A9.0", "Shadow Study")
    x0, y0, W, H = sh.area
    cw, ch = W / 3, H / 3
    for i, (dlabel, mo, dy) in enumerate(DATES):
        for j, hr in enumerate(TIMES):
            when = datetime(year, mo, dy, hr, 0, tzinfo=tzinfo)
            az, el = solar_position(project.lat, project.lon, when)
            dr = Drawing()
            dr.poly(list(clip.exterior.coords), "G-ANNO-TTLB", lw=0.13)
            if ctx is not None:
                for r, p in road_polygons(ctx):
                    dr.shape(p.intersection(clip), "C-ROAD-PATT", fill="#eeeeee", edge=False, z=0)
            exist_sh, new_sh = [], []
            if ctx is not None:
                for f in ctx.of("building"):
                    if f.geom.intersects(fp) or not f.geom.intersects(clip.buffer(40)):
                        continue
                    s = prism_shadow(f.geom, 0, f.height, az, el)
                    if s is not None:
                        exist_sh.append(s)
                    dr.shape(f.geom.intersection(clip), "C-BLDG-EXST-PATT", fill="#ffffff", z=4, lw=0.25)
            # proposed: pitched roofs approximated by eave prism + ridge-height prism of a narrowed footprint
            s = prism_shadow(fp, 0, h_eave if b.roof.type != "flat" else h, az, el)
            if s is not None:
                new_sh.append(s)
            if b.roof.type in ("gable", "hip"):
                core = fp.buffer(-min(fp.bounds[2] - fp.bounds[0], fp.bounds[3] - fp.bounds[1]) * 0.35)
                if not core.is_empty:
                    s2 = prism_shadow(core, 0, h, az, el)
                    if s2 is not None:
                        new_sh.append(s2)
            ex = unary_union(exist_sh).intersection(clip) if exist_sh else None
            nw = unary_union(new_sh).intersection(clip) if new_sh else None
            if ex is not None:
                dr.shape(ex, "A-ELEV-FILL", fill="#9e9e9e", edge=False, z=1, alpha=0.8)
            if nw is not None:
                net = nw.difference(ex) if ex is not None else nw
                dr.shape(net, "A-ELEV-FILL", fill="#d35400", edge=False, z=2, alpha=0.75)
            if project.parcel is not None:
                dr.shape(project.parcel, "C-PROP", fill="none", z=5)
            dr.shape(fp, "A-BLDG-PROP-PATT", fill="#555555", z=6)
            title = (f"{dlabel.split(' (')[0]} {hr:02d}:00 {when.tzname()}  "
                     f"(az {az:.0f}°, alt {el:.0f}°)")
            sh.viewport(dr, (x0 + j * cw, y0 + (2 - i) * ch, cw, ch), title=title,
                        kind="site", number=i * 3 + j + 1)
    sh.paper.text(x0, y0 + H + 0.1 * IN,
                  f"Local clock time ({project.timezone or f'UTC{tz:+g}'}). Grey = existing shadow; orange = NEW shadow from proposal. "
                  "Flat-ground projection.", fontsize=7)
    with PdfPages(out) as pdf:
        pdf.savefig(sh.fig)
    sh.fig.savefig(str(out).replace(".pdf", ".png"), dpi=120)
    import matplotlib.pyplot as plt
    plt.close(sh.fig)
    return out
