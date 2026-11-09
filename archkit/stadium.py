"""Stadium design tools: seating-bowl sightlines, assembly egress, seat counts, playing field.

Sightlines use the C-value method (as in the UK Guide to Safety at Sports
Grounds and FIFA/NFL practice): each row's eye point must clear the eye of the
row in front by C (mm) on the line to the focal point.

    E[n+1] = (E[n] + C) * D[n+1] / D[n]

with D horizontal distance eye->focal point and E eye height above the focal
point.  Risers are rounded UP to a construction increment and the achieved C
is reported per row.  Typical targets: 90 mm good, 120 mm excellent, 60 mm
minimum.  Bowls steeper than 34° are flagged.

Egress follows IBC 2021 §1029 (assembly seating).  Values are design-phase
defaults — confirm against the adopted code edition (Kentucky Building Code
for Kroger Field), NFPA 101 §12.4 and the AHJ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Point, Polygon, box

from .checks import Finding
from .units import FT, INCH

YD = 0.9144


# ---------------------------------------------------------------------------
# Sightlines
# ---------------------------------------------------------------------------

@dataclass
class BowlSection:
    name: str
    side: str                 # west | east | north | south (relative to the field axis frame)
    first_row_offset: float   # m, horizontal from focal line (sideline / end line) to row-1 eye
    first_row_height: float   # m, row-1 tread above field level
    rows: int
    tread: float = 0.80       # m row depth (US typical 33" = 0.84)
    eye_height: float = 1.20  # m seated eye above tread
    c_target: float = 0.090   # m
    riser_increment: float = 0.010
    min_riser: float = 0.0
    length: float = 100.0     # m along the field axis (seating zone length)
    aisle_spacing: float = 0.0  # m between aisles (0 -> computed from seats/row limit)
    seat_width: float = 0.508   # m (20")
    aisle_width: float = 1.22   # m (48")
    focal_offset: float = 0.0   # m beyond the near line (e.g. 0 = sideline)
    center: float = 0.0         # m along axis from field centre
    smoke_protected: bool = True
    rows_data: list = field(default_factory=list)

    def solve(self) -> list[dict]:
        D = self.first_row_offset + self.focal_offset
        E = self.first_row_height + self.eye_height
        out = [{"row": 1, "D": D, "E": E, "tread_z": self.first_row_height, "riser": None, "C": None}]
        for n in range(2, self.rows + 1):
            D1 = D + self.tread
            E1_req = (E + self.c_target) * D1 / D
            riser = max(E1_req - E, self.min_riser)
            riser = math.ceil(riser / self.riser_increment - 1e-9) * self.riser_increment
            E1 = E + riser
            C = E1 * D / D1 - E
            out.append({"row": n, "D": D1, "E": E1, "tread_z": E1 - self.eye_height,
                        "riser": riser, "C": C})
            D, E = D1, E1
        self.rows_data = out
        return out

    @property
    def rake_deg(self) -> float:
        r = self.rows_data or self.solve()
        risers = [x["riser"] for x in r[1:]]
        return math.degrees(math.atan(max(risers) / self.tread)) if risers else 0.0

    def seats(self) -> dict:
        """Seats per row and total given aisle layout along the section length."""
        if self.aisle_spacing > 0:
            n_aisles = int(self.length // self.aisle_spacing) + 1
        else:
            # aisles both ends, max 100 seats between aisles (IBC 1029.13.1 smoke-protected) -
            # use a practical 26-28 seat bay (typical stadium)
            n_aisles = max(2, math.ceil(self.length / (28 * self.seat_width + self.aisle_width)) + 1)
        seat_len = self.length - n_aisles * self.aisle_width
        per_row = max(0, int(seat_len // self.seat_width))
        bays = n_aisles - 1
        return {"aisles": n_aisles, "bays": bays, "seats_per_row": per_row,
                "seats_per_bay_row": per_row / max(bays, 1), "total": per_row * self.rows}


# ---------------------------------------------------------------------------
# Egress (IBC 2021 §1029)
# ---------------------------------------------------------------------------

# Table 1029.6.2 smoke-protected assembly: occupant load -> (stairs in/occ, level in/occ)
SMOKE_PROTECTED = [(5000, 0.200, 0.150), (10000, 0.130, 0.100), (15000, 0.096, 0.070),
                   (20000, 0.076, 0.056), (25000, 0.060, 0.044)]
OUTDOOR = (0.08, 0.06)       # §1029.6.3 outdoor smoke-protected assembly
GENERAL = (0.30, 0.20)       # §1005.3.1 / 1029.6.1


def capacity_factors(occupants: int, smoke_protected: bool, outdoor: bool) -> tuple[float, float, str]:
    if outdoor:
        return OUTDOOR[0], OUTDOOR[1], "IBC 1029.6.3 (outdoor smoke-protected)"
    if smoke_protected:
        for lim, st, lv in SMOKE_PROTECTED:
            if occupants <= lim:
                return st, lv, "IBC Table 1029.6.2"
        return SMOKE_PROTECTED[-1][1], SMOKE_PROTECTED[-1][2], "IBC Table 1029.6.2"
    return GENERAL[0], GENERAL[1], "IBC 1005.3.1"


def egress_checks(sec: BowlSection, *, outdoor: bool = True, vomitories: int = 0,
                  vom_width: float = 0.0, total_occupants: int | None = None) -> list[Finding]:
    f: list[Finding] = []
    rows = sec.rows_data or sec.solve()
    s = sec.seats()
    occ = s["total"]
    total = total_occupants or occ
    st_f, lv_f, ref = capacity_factors(total, sec.smoke_protected, outdoor)
    L = lambda m: f"{m / INCH:.1f}\""  # noqa: E731
    # aisle widths
    need = 48 * INCH  # aisle stairs serving seating both sides (1029.9.1)
    f.append(Finding("PASS" if sec.aisle_width + 1e-4 >= need else "FAIL", "egress",
                     f"{sec.name} aisle stair width", L(sec.aisle_width), f">= {L(need)}",
                     "IBC 1029.9.1"))
    # seats between aisles
    per_bay = s["seats_per_bay_row"]
    max_seats = 100 if sec.smoke_protected else 32
    f.append(Finding("PASS" if per_bay <= max_seats else "FAIL", "egress",
                     f"{sec.name} seats per row between aisles", f"{per_bay:.0f}",
                     f"<= {max_seats}", "IBC 1029.13.1 / Table 1029.13.1.1"))
    # row clear width (smoke-protected table simplified: 12" up to 14 seats + 0.3"/seat beyond)
    tread_clear = sec.tread - 0.45  # seat depth when occupied/unoccupied ~ 18" (tip-up ~ 12")
    req = 12 * INCH + max(0, per_bay - 14) * 0.3 * INCH * (0.5 if sec.smoke_protected else 1.0)
    f.append(Finding("PASS" if tread_clear + 1e-4 >= req else "FAIL", "egress",
                     f"{sec.name} row clear width (est.)", L(tread_clear), f">= {L(req)}",
                     "IBC 1029.13.1 (verify with seat product)"))
    # aisle stair risers
    risers = [r["riser"] for r in rows[1:]]
    big = max(risers) if risers else 0
    steps = math.ceil(big / (8 * INCH)) if big else 0
    f.append(Finding("PASS" if big <= 9 * INCH or steps >= 2 else "WARN", "egress",
                     f"{sec.name} max riser", f"{big * 1000:.0f} mm",
                     "aisle risers 4\"-8\" (split rows into steps)", "IBC 1029.13.2.2"))
    f.append(Finding("INFO", "egress", f"{sec.name} aisle steps per row (max)", str(steps),
                     "uniform within each row", "IBC 1029.13.2.2.1"))
    # rake
    f.append(Finding("PASS" if sec.rake_deg <= 34 else "WARN", "sightline",
                     f"{sec.name} maximum rake", f"{sec.rake_deg:.1f}°", "<= 34°",
                     "Guide to Safety at Sports Grounds 12.13 (comfort/vertigo)"))
    cs = [r["C"] for r in rows[1:]]
    if cs:
        cmin = min(cs)
        f.append(Finding("PASS" if cmin >= 0.060 - 1e-6 else "FAIL", "sightline",
                         f"{sec.name} C-value min / target", f"{cmin * 1000:.0f} mm / "
                         f"{sec.c_target * 1000:.0f} mm", ">= 60 mm (90 good, 120 excellent)"))
    # guards at cross-aisles: rows whose tread is > 30" above the row in front
    f.append(Finding("INFO", "egress", f"{sec.name} occupant load", f"{occ:,}",
                     f"egress width @ {st_f}\"/occ stairs, {lv_f}\"/occ level", ref))
    req_w = occ * st_f * INCH
    f.append(Finding("INFO", "egress", f"{sec.name} required stair egress width",
                     f"{req_w / FT:.1f} ft", "provide via aisles/vomitories", ref))
    prov_aisle = s["aisles"] * sec.aisle_width
    f.append(Finding("PASS" if prov_aisle >= req_w else "FAIL", "egress",
                     f"{sec.name} aisle egress capacity", f"{prov_aisle / FT:.1f} ft "
                     f"({s['aisles']} aisles)", f">= {req_w / FT:.1f} ft",
                     ref + " — add aisles/widen, or count vomitories"))
    if vomitories:
        prov = vomitories * vom_width
        f.append(Finding("PASS" if prov >= occ * lv_f * INCH else "FAIL", "egress",
                         f"{sec.name} vomitory width", f"{prov / FT:.1f} ft",
                         f">= {occ * lv_f * INCH / FT:.1f} ft (level)", ref))
    # travel distance proxy: farthest seat to aisle end along row + aisle run
    bay_len = per_bay * sec.seat_width
    run = rows[-1]["D"] - rows[0]["D"]
    rise = rows[-1]["E"] - rows[0]["E"]
    travel = bay_len / 2 + math.hypot(run, rise)
    lim = 400 * FT if sec.smoke_protected else 200 * FT
    f.append(Finding("PASS" if travel <= lim else "WARN", "egress",
                     f"{sec.name} travel to section exit (est.)", f"{travel / FT:.0f} ft",
                     f"<= {lim / FT:.0f} ft", "IBC 1029.7 (outdoor smoke-protected may be unlimited)"))
    return f


# ---------------------------------------------------------------------------
# Field geometry (field frame: x along the long axis, y across; origin at field centre)
# ---------------------------------------------------------------------------

@dataclass
class Field:
    center: tuple[float, float]
    axis_deg: float                 # direction of the long (goal-to-goal) axis, CCW from +x (east)
    kind: str = "american_football_ncaa"
    turf: str = "synthetic"

    @property
    def length(self) -> float:
        return 120 * YD             # incl. end zones

    @property
    def width(self) -> float:
        return 160 * FT             # 53 1/3 yd

    def to_site(self, g):
        return translate(rotate(g, self.axis_deg, origin=(0, 0)), *self.center)

    def pt(self, x, y):
        a = math.radians(self.axis_deg)
        return (self.center[0] + x * math.cos(a) - y * math.sin(a),
                self.center[1] + x * math.sin(a) + y * math.cos(a))

    def markings(self) -> list[tuple[str, Polygon]]:
        """(material, polygon) list in site frame: stripes, end zones, lines, hashes."""
        L, W = self.length, self.width
        out = []
        lw = 4 * INCH
        out.append(("turf_dark", box(-L / 2 - 4 * YD, -W / 2 - 6 * YD, L / 2 + 4 * YD, W / 2 + 6 * YD)))
        for k in range(20):  # 5-yd mowing stripes between goal lines
            x0 = -50 * YD + k * 5 * YD
            out.append(("turf_light" if k % 2 == 0 else "turf_dark", box(x0, -W / 2, x0 + 5 * YD, W / 2)))
        out.append(("endzone", box(-L / 2, -W / 2, -50 * YD, W / 2)))
        out.append(("endzone", box(50 * YD, -W / 2, L / 2, W / 2)))
        # boundary (6 ft wide white border outside sideline per NCAA rule 1-2-3 limit lines simplified)
        out.append(("paint_white", box(-L / 2, -W / 2 - 4 * INCH, L / 2, -W / 2)))
        out.append(("paint_white", box(-L / 2, W / 2, L / 2, W / 2 + 4 * INCH)))
        out.append(("paint_white", box(-L / 2 - 4 * INCH, -W / 2, -L / 2, W / 2)))
        out.append(("paint_white", box(L / 2, -W / 2, L / 2 + 4 * INCH, W / 2)))
        for yd in range(-50, 51, 5):  # yard lines
            x = yd * YD
            w = 8 * INCH if yd in (-50, 50) else lw   # goal lines 8"
            out.append(("paint_white", box(x - w / 2, -W / 2, x + w / 2, W / 2)))
        hash_y = W / 2 - 60 * FT   # NCAA hashes 60' from sidelines
        for yd in range(-49, 50):
            if yd % 5 == 0:
                continue
            x = yd * YD
            for yy in (-W / 2, hash_y, -hash_y, W / 2 - 2 * FT):
                out.append(("paint_white", box(x - lw / 2, yy, x + lw / 2, yy + 2 * FT)))
        return [(m, self.to_site(p)) for m, p in out]


def detect_field(ctx) -> Field | None:
    """Find the American-football pitch in OSM context and return its pose."""
    best = None
    stadia = [f.geom for f in ctx.of("green") if f.props.get("use") == "stadium"]
    stadia += [f.geom for f in ctx.of("building") if f.props.get("building") == "stadium"]
    for f in ctx.of("green"):
        use = (f.props.get("use") or "")
        if use != "pitch":
            continue
        g = f.geom
        mrr = g.minimum_rotated_rectangle
        c = list(mrr.exterior.coords)
        e1 = np.subtract(c[1], c[0])
        e2 = np.subtract(c[2], c[1])
        long_e = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
        Lm = max(np.linalg.norm(e1), np.linalg.norm(e2))
        Wm = min(np.linalg.norm(e1), np.linalg.norm(e2))
        score = abs(Lm - 110) + abs(Wm - 49)   # NCAA field incl. end zones
        sport = f.props.get("sport", "")
        if "football" in sport:
            score -= 50
        if any(st_.contains(g.representative_point()) for st_ in stadia):
            score -= 100   # the game field sits inside the stadium; practice fields don't
        score += math.hypot(*mrr.centroid.coords[0]) / 1000.0  # tie-break: nearest to site origin
        if best is None or score < best[0]:
            ang = math.degrees(math.atan2(long_e[1], long_e[0]))
            best = (score, Field((mrr.centroid.x, mrr.centroid.y), ang))
    return best[1] if best else None


def bowl_polygon(fieldg: Field, sec: BowlSection) -> Polygon:
    """Plan footprint of a seating section in site frame."""
    rows = sec.rows_data or sec.solve()
    depth = rows[-1]["D"] - rows[0]["D"] + sec.tread
    L, W = fieldg.length, fieldg.width
    side = sec.side
    if side in ("west", "east"):   # long sides: along the axis
        y0 = (W / 2 + sec.focal_offset * 0) + sec.first_row_offset - sec.tread / 2
        g = box(sec.center - sec.length / 2, y0, sec.center + sec.length / 2, y0 + depth)
        if side == "east":
            g = rotate(g, 180, origin=(0, 0))
    else:                          # end zones
        x0 = L / 2 + sec.first_row_offset - sec.tread / 2
        g = box(x0, sec.center - sec.length / 2, x0 + depth, sec.center + sec.length / 2)
        if side == "north":
            pass
        else:
            g = rotate(g, 180, origin=(0, 0))
    return fieldg.to_site(g)


def section_drawing(sec: BowlSection, system: str = "imperial"):
    """Bowl section with treads, risers, eye points and sightlines to the focal point."""
    from .drawings.core import Drawing

    rows = sec.rows_data or sec.solve()
    dr = Drawing()
    # field level & focal point at x=0
    fx = -sec.focal_offset
    dr.line((fx - 8, 0), (rows[-1]["D"] + 3, 0), "G-GRND")
    dr.circle((0, 0), 0.25, "A-ANNO-SYMB", fill="#c0392b")
    dr.text((0, -1.0), "FOCAL POINT", h_mm=2.0)
    prof = [(rows[0]["D"] - sec.tread / 2, 0.0)]
    for r in rows:
        x0, x1 = r["D"] - sec.tread / 2, r["D"] + sec.tread / 2
        prof += [(x0, r["tread_z"]), (x1, r["tread_z"])]
    prof += [(prof[-1][0], 0.0)]
    dr.poly(prof, "A-SECT-PATT", fill="#2b2b2b", z=3)
    for i, r in enumerate(rows):
        dr.circle((r["D"], r["E"]), 0.08, "A-ANNO-SYMB", fill="#1f77b4", z=5)
        if i % max(1, len(rows) // 8) == 0 or i == len(rows) - 1:
            dr.line((r["D"], r["E"]), (0, 0), "A-ELEV-BEYD", z=2)
            c = "" if r["C"] is None else f"C={r['C'] * 1000:.0f}"
            dr.text((r["D"] + 0.6, r["E"] + 0.25), f"R{r['row']}  {c}", h_mm=1.8, ha="left", z=6)
    return dr


def section_table(sec: BowlSection, system: str = "imperial") -> list[tuple]:
    rows = sec.rows_data or sec.solve()
    out = [("Row", "Dist", "Eye ht", "Riser", "C (mm)")]
    for r in rows:
        if system == "imperial":
            d = f"{r['D'] / FT:.1f}'"
            e = f"{r['E'] / FT:.2f}'"
            rs = "" if r["riser"] is None else f"{r['riser'] / INCH:.2f}\""
        else:
            d, e = f"{r['D']:.2f}", f"{r['E']:.3f}"
            rs = "" if r["riser"] is None else f"{r['riser'] * 1000:.0f}"
        out.append((r["row"], d, e, rs, "" if r["C"] is None else f"{r['C'] * 1000:.0f}"))
    return out


def stand_model(m, fieldg: Field, sec: BowlSection, base_z: float = 0.0,
                seat_mat: str = "seat_blue", conc_mat: str = "concrete"):
    """Add stepped seating rows (concrete treads + seat band) to a Model."""
    from .model3d import add_box_along

    rows = sec.rows_data or sec.solve()
    L, W = fieldg.length, fieldg.width
    a = math.radians(fieldg.axis_deg)
    ax = np.array([math.cos(a), math.sin(a)])
    ay = np.array([-ax[1], ax[0]])
    C = np.array(fieldg.center)
    if sec.side in ("west", "east"):
        sgn = 1 if sec.side == "west" else -1
        u = ax * sgn                     # along the rows
        n = ay * sgn                     # away from the field
        origin = C + n * (W / 2) + ax * sec.center - u * sec.length / 2
    else:
        sgn = 1 if sec.side == "north" else -1
        u = -ay * sgn
        n = ax * sgn
        origin = C + n * (L / 2) + (-ay * sgn) * 0 + ay * sec.center - u * sec.length / 2
    for r in rows:
        w0 = r["D"] - sec.tread / 2
        w1 = r["D"] + sec.tread / 2
        add_box_along(m, origin, u, n, 0, sec.length, w0, w1, base_z, base_z + r["tread_z"],
                      conc_mat, f"stadium/{sec.name}/bowl")
        # seat band (simplified chair row: back + seat volume)
        add_box_along(m, origin, u, n, 0, sec.length, w0 + 0.30, w0 + 0.75, base_z + r["tread_z"],
                      base_z + r["tread_z"] + 0.45, seat_mat, f"stadium/{sec.name}/seats")
    # back wall
    last = rows[-1]
    add_box_along(m, origin, u, n, 0, sec.length, last["D"] + sec.tread / 2,
                  last["D"] + sec.tread / 2 + 0.3, base_z, base_z + last["tread_z"] + 1.1,
                  conc_mat, f"stadium/{sec.name}/bowl")


def field_model(m, fieldg: Field, z: float):
    for i, (mat, poly) in enumerate(fieldg.markings()):
        lift = {"turf_dark": 0.004, "turf_light": 0.004, "endzone": 0.006, "paint_white": 0.01}[mat]
        from .model3d import triangulate
        for tri in triangulate(poly):
            m.add([[x, y, z + (0.0 if i == 0 else lift)] for x, y in tri], mat, "stadium/field")


# ---------------------------------------------------------------------------
# Spec integration
# ---------------------------------------------------------------------------

def from_spec(project, ctx) -> tuple[Field | None, list[BowlSection]]:
    """Parse the project's `stadium:` block (lengths in project units; C in mm)."""
    sd = project.stadium or {}
    if not sd:
        return None, []
    k = FT if project.units == "ft" else 1.0
    fd = sd.get("field", "auto")
    fieldg = None
    if isinstance(fd, dict):
        fieldg = Field((float(fd["center"][0]) * k, float(fd["center"][1]) * k),
                       float(fd.get("axis_deg", 90)))
    elif ctx is not None:
        fieldg = detect_field(ctx)
        if fieldg is not None and "axis_deg" in (sd.get("field_hint") or {}):
            fieldg.axis_deg = float(sd["field_hint"]["axis_deg"])
    if fieldg is not None:
        # canonicalise: long axis pointing into the northern half-plane
        if math.sin(math.radians(fieldg.axis_deg)) < 0:
            fieldg.axis_deg = (fieldg.axis_deg + 180) % 360
    secs = []
    for s in sd.get("sections", []):
        secs.append(BowlSection(
            name=s["name"], side=s["side"],
            first_row_offset=float(s["first_row_offset"]) * k,
            first_row_height=float(s.get("first_row_height", 0)) * k,
            rows=int(s["rows"]), tread=float(s.get("tread", 33 / 12 if k == FT else 0.84)) * k,
            eye_height=float(s.get("eye_height", 3.9 if k == FT else 1.2)) * k,
            c_target=float(s.get("c_target_mm", 90)) / 1000,
            riser_increment=float(s.get("riser_increment_mm", 10)) / 1000,
            length=float(s["length"]) * k, center=float(s.get("center", 0)) * k,
            seat_width=float(s.get("seat_width", 20 / 12 if k == FT else 0.508)) * k,
            aisle_width=float(s.get("aisle_width", 4 if k == FT else 1.22)) * k,
            aisle_spacing=float(s.get("aisle_spacing", 0)) * k,
            focal_offset=float(s.get("focal_offset", 0)) * k,
            smoke_protected=bool(s.get("smoke_protected", True))))
        secs[-1].solve()
    return fieldg, secs


def stadium_checks(project, ctx) -> list[Finding]:
    fieldg, secs = from_spec(project, ctx)
    sd = project.stadium or {}
    out: list[Finding] = []
    if fieldg is None:
        out.append(Finding("FAIL", "stadium", "playing field", "not found in OSM",
                           "set stadium.field: {center, axis_deg}"))
        return out
    out.append(Finding("INFO", "stadium", "field pose",
                       f"centre ({fieldg.center[0]:.1f}, {fieldg.center[1]:.1f}) m, axis {fieldg.axis_deg:.2f}°",
                       "-", "from OSM pitch polygon" if sd.get("field", "auto") == "auto" else "spec"))
    total = sum(s.seats()["total"] for s in secs)
    for s in secs:
        st = s.seats()
        out.append(Finding("INFO", "stadium", f"{s.name} seats",
                           f"{st['total']:,} ({s.rows} rows x {st['seats_per_row']})",
                           f"{st['aisles']} aisles"))
        out += egress_checks(s, outdoor=bool(sd.get("outdoor", True)),
                             total_occupants=int(sd.get("occupants_total", total)),
                             vomitories=int(sd.get("vomitories_per_section", 0)),
                             vom_width=float(sd.get("vomitory_width", 0)) * (FT if project.units == "ft" else 1))
        poly = bowl_polygon(fieldg, s)
        if ctx is not None:
            for r in ctx.of("road"):
                if r.props.get("highway") in ("footway", "path", "service", "steps", "cycleway"):
                    continue
                if r.geom.intersects(poly):
                    out.append(Finding("FAIL", "site", f"{s.name} footprint crosses road",
                                       r.props.get("name") or r.props.get("highway"), "no encroachment"))
            if project.parcel is not None and not project.parcel.contains(poly):
                out.append(Finding("FAIL", "site", f"{s.name} outside property", "-", "within parcel"))
    out.append(Finding("INFO", "stadium", "new seats (all sections)", f"{total:,}", "-"))
    return out
