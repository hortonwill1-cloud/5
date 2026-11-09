"""Site coordination and preliminary code checks.

Every check returns a Finding with PASS / WARN / FAIL, the measured value,
the requirement, and the reference.  These are PRELIMINARY design-phase
checks — the Authority Having Jurisdiction (AHJ), local amendments and a
licensed professional govern.  Values come from the 2021 I-codes unless the
project's `code:` block overrides them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from .site import SiteContext, nearest_road, road_polygons
from .spec import Project
from .units import FT, INCH, SQFT, fmt_area, fmt_len


@dataclass
class Finding:
    status: str        # PASS | WARN | FAIL | INFO
    category: str      # site | zoning | code | model | data
    item: str
    measured: str
    required: str
    ref: str = ""

    def row(self) -> str:
        return f"[{self.status:4}] {self.category:7} {self.item}: {self.measured} (req. {self.required}) {self.ref}"


# Defaults (metres).  Override in spec: code: {min_door_clear: 0.81, ...}
CODE_DEFAULTS = {
    "IRC-2021": {
        "min_egress_door_width": 36 * INCH,       # R311.2: 32" clear ~ 36" leaf
        "min_door_width": 30 * INCH,
        "min_habitable_area": 70 * SQFT,          # R304.1
        "min_habitable_dim": 7 * FT,              # R304.2
        "min_ceiling": 7 * FT,                    # R305.1
        "max_riser": 7.75 * INCH,                 # R311.7.5.1
        "min_tread": 10 * INCH,                   # R311.7.5.2
        "min_stair_width": 36 * INCH,             # R311.7.1
        "egress_min_area": 5.7 * SQFT,            # R310.2.1 (5.0 at grade floor)
        "egress_min_width": 20 * INCH,
        "egress_min_height": 24 * INCH,
        "egress_max_sill": 44 * INCH,
        "fire_sep_unrated": 5 * FT,               # R302.1 walls <5 ft need 1-hr
        "glazing_net_factor": 0.80,               # rough->net clear for casements
    },
    "IBC-2021": {
        "min_egress_door_width": 36 * INCH,       # 1010.1.1: 32" clear
        "min_door_width": 36 * INCH,
        "min_habitable_area": 70 * SQFT,
        "min_habitable_dim": 7 * FT,
        "min_ceiling": 7.5 * FT,                  # 1208.2
        "max_riser": 7 * INCH,                    # 1011.5.2
        "min_tread": 11 * INCH,
        "min_stair_width": 44 * INCH,             # 1011.2
        "egress_min_area": 5.7 * SQFT,            # 1031.3 (R occupancies)
        "egress_min_width": 20 * INCH,
        "egress_min_height": 24 * INCH,
        "egress_max_sill": 44 * INCH,
        "fire_sep_unrated": 10 * FT,              # Table 705.5 (varies by type/occupancy)
        "glazing_net_factor": 0.80,
    },
}


def _rules(project: Project) -> dict:
    prof = project.code.get("profile", "IRC-2021")
    r = dict(CODE_DEFAULTS.get(prof, CODE_DEFAULTS["IRC-2021"]))
    for k, v in project.code.items():
        if k in r:
            r[k] = float(v) * (FT if project.units == "ft" else 1.0)
    r["profile"] = prof
    return r


def front_lot_lines(parcel: Polygon, ctx: SiteContext | None) -> list[int]:
    """Indices of parcel edges facing the nearest public road (front lot lines)."""
    coords = list(parcel.exterior.coords)
    edges = [LineString([a, b]) for a, b in zip(coords[:-1], coords[1:])]
    if ctx is None or not ctx.of("road"):
        # fall back to the southern-most edge
        return [min(range(len(edges)), key=lambda i: edges[i].centroid.y)]
    road, _ = nearest_road(ctx, parcel)
    if road is None:
        return [0]
    d = [e.centroid.distance(road.geom) for e in edges]
    dmin = min(d)
    # every edge within 3 m of the closest qualifies (corner lots -> multiple fronts)
    return [i for i, di in enumerate(d) if di <= dmin + 3.0 and edges[i].length > 3.0]


def setback_envelope(project: Project, ctx: SiteContext | None) -> tuple[Polygon, dict]:
    """Buildable area: parcel minus front/side/rear setback strips."""
    parcel = project.parcel
    sb = project.zoning.get("setbacks", {})
    front_i = front_lot_lines(parcel, ctx)
    coords = list(parcel.exterior.coords)
    edges = [LineString([a, b]) for a, b in zip(coords[:-1], coords[1:])]
    fronts = [edges[i] for i in front_i]
    front_dir = None
    if fronts:
        f = max(fronts, key=lambda e: e.length)
        (x0, y0), (x1, y1) = f.coords
        front_dir = math.atan2(y1 - y0, x1 - x0)
    strips = []
    labels = {}
    for i, e in enumerate(edges):
        if i in front_i:
            d, lab = sb.get("front", 0.0), "front"
        else:
            (x0, y0), (x1, y1) = e.coords
            ang = math.atan2(y1 - y0, x1 - x0)
            parallel = front_dir is not None and abs(math.sin(ang - front_dir)) < 0.5
            if parallel:
                d, lab = sb.get("rear", 0.0), "rear"
            else:
                d, lab = sb.get("side", 0.0), "side"
        labels[i] = lab
        if d > 0:
            strips.append(e.buffer(d, cap_style="flat").intersection(parcel))
            strips.append(e.buffer(d))
    env = parcel.difference(unary_union(strips)) if strips else parcel
    if env.geom_type == "MultiPolygon":
        env = max(env.geoms, key=lambda g: g.area)
    return env, labels


def _provenance(project: Project, ctx: SiteContext | None) -> list[Finding]:
    out = []
    if ctx is not None:
        blds = ctx.of("building")
        est_b = [f for f in blds if "ESTIMATED" in f.props.get("height_source", "")]
        lid = [f for f in blds if f.props.get("height_source", "").startswith("LiDAR")]
        if lid:
            out.append(Finding("INFO", "data", "context building heights measured (LiDAR)",
                               f"{len(lid)}/{len(blds)}", "-"))
        if est_b:
            out.append(Finding("INFO", "data", "context building heights estimated",
                               f"{len(est_b)}/{len(blds)}", "tagged heights, LiDAR or survey",
                               "verify by site visit / LiDAR before publishing views"))
        if ctx.elevation:
            out.append(Finding("INFO", "data", "terrain source", ctx.elevation["source"], "-"))
        else:
            out.append(Finding("WARN", "data", "terrain", "flat-site assumption", "DEM or topo survey"))
    src = project.parcel_source.lower()
    authoritative = ("survey" in src or "gis" in src or "parcel" in src) and "approx" not in src
    out.append(Finding("INFO" if authoritative else "WARN", "data", "parcel boundary source",
                       project.parcel_source, "boundary survey for permit documents"))
    return out


def run_checks(project: Project, ctx: SiteContext | None) -> list[Finding]:
    from .model3d import resolve_finish_floor

    out: list[Finding] = []
    sysu = project.system
    L = lambda v: fmt_len(v, sysu)  # noqa: E731
    A = lambda v: fmt_area(v, sysu)  # noqa: E731
    b = project.building
    rules = _rules(project)
    if b is None:
        out += _provenance(project, ctx)
        if project.stadium:
            from .stadium import stadium_checks
            out += stadium_checks(project, ctx)
        return out
    fp_local = b.footprint()
    fp = b.to_site(fp_local)

    out += _provenance(project, ctx)
    if project.stadium:
        from .stadium import stadium_checks
        out += stadium_checks(project, ctx)

    # ---------------- model sanity ----------------------------------------
    for lv in b.levels:
        for o in lv.openings:
            try:
                w = lv.wall(o.wall)
                a, c = w.segments()[o.segment]
            except (KeyError, IndexError):
                out.append(Finding("FAIL", "model", f"{lv.name} opening {o.tag or o.kind}",
                                   f"wall {o.wall} seg {o.segment} not found", "valid host"))
                continue
            seglen = math.dist(a, c)
            if o.offset < w.thickness / 2 or o.offset + o.width > seglen - w.thickness / 2:
                out.append(Finding("FAIL", "model", f"{lv.name} {o.kind} {o.tag}",
                                   f"{L(o.offset)}..{L(o.offset + o.width)} on {L(seglen)} wall",
                                   "opening clear of wall ends/corners"))
            head = o.sill + o.height
            if head > (w.height or lv.floor_to_floor) - 0.15:
                out.append(Finding("FAIL", "model", f"{lv.name} {o.kind} {o.tag} head",
                                   L(head), f"< {L((w.height or lv.floor_to_floor) - 0.15)} (header)"))
        # overlapping openings
        for w in lv.walls:
            for si in range(len(w.segments())):
                ops = sorted([o for o in lv.openings if o.wall == w.id and o.segment == si],
                             key=lambda o: o.offset)
                for o1, o2 in zip(ops, ops[1:]):
                    if o1.offset + o1.width > o2.offset - 0.1:
                        out.append(Finding("FAIL", "model", f"{lv.name} openings {o1.tag}/{o2.tag}",
                                           "overlap or <4\" pier", "min. 4\" pier between openings"))
    if b.roof.type in ("gable", "hip", "shed"):
        ratio = fp_local.area / fp_local.envelope.area
        if ratio < 0.98:
            out.append(Finding("WARN", "model", "pitched roof on non-rectangular footprint",
                               f"fill {ratio:.0%}", "rectangular", "roof generated on bounding box"))

    # ---------------- zoning ----------------------------------------------
    z = project.zoning
    if project.parcel is not None:
        parcel = project.parcel
        out.append(Finding("INFO", "zoning", "lot area", A(parcel.area), "-"))
        if not parcel.contains(fp):
            over = fp.difference(parcel).area
            out.append(Finding("FAIL", "zoning", "building within lot", f"{A(over)} outside lot",
                               "entirely within property lines"))
        if "setbacks" in z:
            env, labels = setback_envelope(project, ctx)
            outside = fp.difference(env)
            if outside.area > 0.01:
                out.append(Finding("FAIL", "zoning", "setbacks", f"{A(outside.area)} encroaches",
                                   ", ".join(f"{k} {L(v)}" for k, v in z["setbacks"].items())))
            else:
                out.append(Finding("PASS", "zoning", "setbacks", "within buildable envelope",
                                   ", ".join(f"{k} {L(v)}" for k, v in z["setbacks"].items())))
            # report actual distance per side
            coords = list(parcel.exterior.coords)
            for i, (a, c) in enumerate(zip(coords[:-1], coords[1:])):
                d = LineString([a, c]).distance(fp)
                req = z["setbacks"].get(labels[i], 0)
                out.append(Finding("PASS" if d + 1e-3 >= req else "FAIL", "zoning",
                                   f"lot line {i} ({labels[i]})", L(d), f">= {L(req)}"))
        cov = fp.area / parcel.area
        if "max_lot_coverage" in z:
            mx = float(z["max_lot_coverage"])
            out.append(Finding("PASS" if cov <= mx else "FAIL", "zoning", "lot coverage",
                               f"{cov:.1%}", f"<= {mx:.1%}"))
        far = b.gross_area() / parcel.area
        if "max_far" in z:
            mx = float(z["max_far"])
            out.append(Finding("PASS" if far <= mx else "FAIL", "zoning", "FAR",
                               f"{far:.2f}", f"<= {mx:.2f}"))
        # fire separation distance (to lot line)
        fsd = min(LineString([a, c]).distance(fp)
                  for a, c in zip(list(parcel.exterior.coords)[:-1], list(parcel.exterior.coords)[1:]))
        rated = any(w.fire_rating_hr >= 1 for w in b.levels[0].walls if w.exterior)
        ok = fsd >= rules["fire_sep_unrated"] or rated
        out.append(Finding("PASS" if ok else "FAIL", "code", "fire separation distance",
                           L(fsd) + (" (rated wall)" if rated else ""),
                           f">= {L(rules['fire_sep_unrated'])} or 1-hr rated wall",
                           "IRC R302.1 / IBC 705"))
    if "max_height" in z:
        ff = resolve_finish_floor(project, ctx)
        # height measured from average grade (common definition) to highest point
        if ctx is not None:
            grade = sum(ctx.ground_z(x, y) for x, y in fp.exterior.coords[:-1]) / (len(fp.exterior.coords) - 1)
        else:
            grade = 0.0
        if b.roof.type in ("gable", "hip"):
            # many codes measure pitched roofs to mean roof height; report both
            tow = ff + b.top_of_wall()
            peak = ff + b.roof_peak()
            mean_h = (tow + peak) / 2 - grade
            out.append(Finding("PASS" if mean_h <= z["max_height"] else "FAIL", "zoning",
                               "building height (mean roof)", L(mean_h), f"<= {L(z['max_height'])}"))
        h = ff + b.roof_peak() - grade
        out.append(Finding("PASS" if h <= z["max_height"] else
                           ("WARN" if b.roof.type in ("gable", "hip") else "FAIL"),
                           "zoning", "building height (highest point)", L(h), f"<= {L(z['max_height'])}",
                           "confirm local height definition"))

    # ---------------- context coordination --------------------------------
    if ctx is not None:
        for f in ctx.of("building"):
            if f.geom.intersects(fp):
                nm = f.props.get("name") or " ".join(
                    x for x in (f.props.get("addr:housenumber"), f.props.get("addr:street")) if x
                ) or f"OSM way {f.props.get('osm_id')}"
                out.append(Finding("WARN", "site", f"existing building on footprint ({nm})",
                                   A(f.geom.intersection(fp).area), "demolition or redesign",
                                   "confirm demolition scope"))
        others = [f for f in ctx.of("building") if not f.geom.intersects(fp)]
        if others:
            nb = min(others, key=lambda f: f.geom.distance(fp))
            d = nb.geom.distance(fp)
            out.append(Finding("PASS" if d >= 2 * rules["fire_sep_unrated"] else "WARN", "site",
                               "nearest neighbouring building", L(d),
                               f">= {L(2 * rules['fire_sep_unrated'])} typical", "verify on survey"))
        for r, poly in road_polygons(ctx):
            if poly.intersects(fp) and r.props.get("highway") not in ("footway", "path", "service"):
                out.append(Finding("FAIL", "site", f"building intersects road {r.props.get('name') or r.props.get('highway')}",
                                   A(poly.intersection(fp).area), "no encroachment into ROW"))
        road, d = nearest_road(ctx, fp)
        if road is not None:
            edge = d - road.props.get("width_m", 6) / 2
            out.append(Finding("INFO", "site", f"nearest road: {road.props.get('name') or road.props.get('highway')}",
                               f"{L(edge)} to pavement edge ({road.props['width_source']})", "-"))

    # ---------------- building code ---------------------------------------
    egress_door = False
    for lv in b.levels:
        if lv.ceiling < rules["min_ceiling"]:
            out.append(Finding("FAIL", "code", f"{lv.name} ceiling height", L(lv.ceiling),
                               f">= {L(rules['min_ceiling'])}", f"{rules['profile']} R305/1208.2"))
        for r in lv.rooms:
            if r.use in ("habitable", "sleeping"):
                ar = r.poly.area
                mrr = r.poly.minimum_rotated_rectangle
                xs = list(mrr.exterior.coords)
                dmin = min(math.dist(xs[0], xs[1]), math.dist(xs[1], xs[2]))
                st = "PASS" if ar >= rules["min_habitable_area"] and dmin >= rules["min_habitable_dim"] else "FAIL"
                out.append(Finding(st, "code", f"{lv.name} {r.number} {r.name} size",
                                   f"{A(ar)}, min dim {L(dmin)}",
                                   f">= {A(rules['min_habitable_area'])}, {L(rules['min_habitable_dim'])}",
                                   "IRC R304 / IBC 1208"))
            if r.use == "sleeping":
                ok_win = False
                for o in lv.openings:
                    if o.kind != "window" or not o.operable:
                        continue
                    w = lv.wall(o.wall)
                    a, c = w.segments()[o.segment]
                    mid_t = (o.offset + o.width / 2) / math.dist(a, c)
                    mx, my = a[0] + (c[0] - a[0]) * mid_t, a[1] + (c[1] - a[1]) * mid_t
                    from shapely.geometry import Point
                    if r.poly.buffer(w.thickness).contains(Point(mx, my)):
                        f = rules["glazing_net_factor"]
                        net_w, net_h = o.width * f, o.height * f
                        net_a = net_w * net_h
                        if (net_a >= rules["egress_min_area"] and net_w >= rules["egress_min_width"]
                                and net_h >= rules["egress_min_height"] and o.sill <= rules["egress_max_sill"]):
                            ok_win = True
                out.append(Finding("PASS" if ok_win else "FAIL", "code",
                                   f"{lv.name} {r.number} {r.name} emergency escape opening",
                                   "compliant window" if ok_win else "none found",
                                   f"net {A(rules['egress_min_area'])}, sill <= {L(rules['egress_max_sill'])}",
                                   "IRC R310 / IBC 1031 (net clear estimated from rough opening)"))
        for o in lv.openings:
            if o.kind == "door":
                req = rules["min_egress_door_width"] if o.egress else rules["min_door_width"]
                out.append(Finding("PASS" if o.width + 1e-3 >= req else "FAIL", "code",
                                   f"{lv.name} door {o.tag} width", L(o.width), f">= {L(req)}",
                                   "IRC R311.2 / IBC 1010.1.1"))
                if o.egress:
                    egress_door = True
        for i, s in enumerate(lv.stairs):
            ok = s.riser <= rules["max_riser"] + 1e-4 and s.tread >= rules["min_tread"] - 1e-4
            out.append(Finding("PASS" if ok else "FAIL", "code", f"{lv.name} stair {i + 1} geometry",
                               f"riser {L(s.riser)}, tread {L(s.tread)}",
                               f"riser <= {L(rules['max_riser'])}, tread >= {L(rules['min_tread'])}",
                               "IRC R311.7.5 / IBC 1011.5"))
            out.append(Finding("PASS" if s.width >= rules["min_stair_width"] else "FAIL", "code",
                               f"{lv.name} stair {i + 1} width", L(s.width),
                               f">= {L(rules['min_stair_width'])}"))
            rise_total = s.risers * s.riser
            if abs(rise_total - lv.floor_to_floor) > 0.01:
                out.append(Finding("FAIL", "code", f"{lv.name} stair {i + 1} total rise",
                                   L(rise_total), f"= floor-to-floor {L(lv.floor_to_floor)}"))
    out.append(Finding("PASS" if egress_door else "FAIL", "code", "egress door provided",
                       "yes" if egress_door else "no door marked egress: true", ">= 1",
                       "IRC R311.2 / IBC 1006"))
    return out


def summarize(findings: list[Finding]) -> str:
    c = {s: sum(f.status == s for f in findings) for s in ("FAIL", "WARN", "PASS", "INFO")}
    lines = [f.row() for f in sorted(findings, key=lambda f: ["FAIL", "WARN", "PASS", "INFO"].index(f.status))]
    head = (f"{c['FAIL']} FAIL · {c['WARN']} WARN · {c['PASS']} PASS · {c['INFO']} INFO\n"
            "Preliminary checks only — verify with the AHJ and a licensed professional.\n")
    return head + "\n".join(lines)
