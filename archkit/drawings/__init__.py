"""Drawing set assembly: PDF sheet set + per-view DXF files."""

from __future__ import annotations

from pathlib import Path

from ..checks import Finding
from ..site import SiteContext
from ..spec import Project
from ..units import fmt_area
from .core import write_dxf
from .elevations import elevation, section
from .plans import floor_plan, site_plan
from .sheet import IN, Sheet


def build_set(project: Project, ctx: SiteContext | None, outdir: str | Path,
              findings: list[Finding] | None = None) -> dict:
    from matplotlib.backends.backend_pdf import PdfPages

    outdir = Path(outdir)
    (outdir / "dxf").mkdir(parents=True, exist_ok=True)
    (outdir / "sheets").mkdir(parents=True, exist_ok=True)
    produced: dict[str, list[str]] = {"pdf": [], "dxf": [], "png": []}
    sheets: list[Sheet] = []
    b = project.building
    georef = {"lat": project.lat, "lon": project.lon} if project.lat is not None else None

    if ctx is not None:
        from ..model3d import apply_clips
        apply_clips(project, ctx)   # existing construction removed by the proposal

    # ---- C1.0 site plan --------------------------------------------------
    sp = site_plan(project, ctx)
    sh = Sheet(project, "C1.0", "Site Plan")
    x, y, w, h = sh.area
    _, scale = sh.viewport(sp, (x, y, w * 0.78, h), title="Architectural Site Plan", kind="site",
                           north_deg=0.0)
    _site_tables(sh, project, ctx, findings, x + w * 0.79, y + h - 0.3 * IN)
    sheets.append(sh)
    from ..units import SCALES
    write_dxf(sp, outdir / "dxf" / "C1.0-site-plan.dxf", project.units, SCALES[scale],
              project.name, georef)
    produced["dxf"].append("dxf/C1.0-site-plan.dxf")

    if project.stadium:
        _stadium_sheets(project, ctx, outdir, sheets, produced)
    if b is None:
        return _finish(project, outdir, sheets, produced)

    # ---- A1.x floor plans --------------------------------------------------
    for i, lv in enumerate(b.levels):
        fp = floor_plan(project, lv)
        num = f"A1.{i + 1}"
        sh = Sheet(project, num, f"{lv.name} Floor Plan")
        _, scale = sh.viewport(fp, sh.area, title=f"{lv.name} Floor Plan",
                               scale=project.sheet.get("plan_scale"), kind="plan",
                               north_deg=-b.rotation_deg)
        sheets.append(sh)
        fn = f"{num}-{lv.name.lower().replace(' ', '-')}-plan.dxf"
        write_dxf(fp, outdir / "dxf" / fn, project.units, SCALES[scale], project.name)
        produced["dxf"].append(f"dxf/{fn}")

    # ---- A2.x elevations (two per sheet, common scale) ---------------------
    from ..units import pick_scale
    els = [elevation(project, ctx, f) for f in ("east", "south", "west", "north")]
    probe = Sheet(project, "A2.0", "Exterior Elevations")
    x, y, w, h = probe.area
    import matplotlib.pyplot as plt
    plt.close(probe.fig)
    ext = [d.bounds() for d, _ in els]
    common = project.sheet.get("elevation_scale") or pick_scale(
        max(e[2] - e[0] for e in ext), max(e[3] - e[1] for e in ext),
        w - 0.6 * IN, h / 2 - 0.9 * IN, project.system)
    for sheet_i in range(2):
        sh = Sheet(project, f"A2.{sheet_i}", "Exterior Elevations")
        for k in range(2):
            d, t = els[sheet_i * 2 + k]
            sh.viewport(d, (x, y + (1 - k) * h / 2, w, h / 2), title=t, scale=common, number=k + 1)
            fn = f"A2.{sheet_i}-{t.split()[0].lower()}-elevation.dxf"
            write_dxf(d, outdir / "dxf" / fn, project.units, SCALES[common], project.name)
            produced["dxf"].append(f"dxf/{fn}")
        sheets.append(sh)

    # ---- A3.0 sections ----------------------------------------------------
    sh = Sheet(project, "A3.0", "Building Sections")
    x, y, w, h = sh.area
    secs = [section(project, ctx, "x"), section(project, ctx, "y")]
    ext = [d.bounds() for d, _ in secs]
    common = pick_scale(max(e[2] - e[0] for e in ext), max(e[3] - e[1] for e in ext),
                        w - 0.6 * IN, h / 2 - 0.9 * IN, project.system)
    for k, (d, t) in enumerate(secs):
        sh.viewport(d, (x, y + (1 - k) * h / 2, w, h / 2), title=t, scale=common, number=k + 1)
        fn = f"A3.0-section-{'a' if k == 0 else 'b'}.dxf"
        write_dxf(d, outdir / "dxf" / fn, project.units, SCALES[common], project.name)
        produced["dxf"].append(f"dxf/{fn}")
    sheets.append(sh)

    return _finish(project, outdir, sheets, produced)


def _stadium_sheets(project, ctx, outdir, sheets, produced):
    from .. import stadium as st
    from ..units import SCALES
    fieldg, secs = st.from_spec(project, ctx)
    for i, sec in enumerate(secs):
        sh = Sheet(project, f"S1.{i + 1}", f"Seating Bowl — {sec.name}")
        x, y, w, h = sh.area
        dr = st.section_drawing(sec, project.system)
        _, scale = sh.viewport(dr, (x, y + h * 0.42, w, h * 0.58),
                               title=f"{sec.name} — sightline section", number=1)
        rows = st.section_table(sec, project.system)
        n = len(rows)
        per_col = 34
        for c0 in range(0, n - 1, per_col):
            chunk = [rows[0]] + rows[1 + c0:1 + c0 + per_col]
            sh.table(x + (c0 // per_col) * 3.3 * IN, y + h * 0.40, chunk,
                     col_w=(0.45 * IN, 0.75 * IN, 0.75 * IN, 0.7 * IN, 0.6 * IN),
                     title="SIGHTLINE SCHEDULE" if c0 == 0 else None)
        seats = sec.seats()
        summ = [("Rows", sec.rows), ("Tread", f"{sec.tread / 0.0254:.1f}\""),
                ("Target C", f"{sec.c_target * 1000:.0f} mm"),
                ("Min C achieved", f"{min(r['C'] for r in sec.rows_data[1:]) * 1000:.0f} mm"),
                ("Max rake", f"{sec.rake_deg:.1f}°"), ("Seats", f"{seats['total']:,}"),
                ("Seats/row", seats["seats_per_row"]), ("Aisles", seats["aisles"])]
        sh.table(x + w - 2.6 * IN, y + h * 0.40, summ, col_w=(1.3 * IN, 1.2 * IN), title="SECTION DATA")
        sheets.append(sh)
        fn = f"S1.{i + 1}-{_slug(sec.name)}-section.dxf"
        write_dxf(dr, outdir / "dxf" / fn, project.units, SCALES[scale], project.name)
        produced["dxf"].append(f"dxf/{fn}")


def _finish(project, outdir, sheets, produced):
    from matplotlib.backends.backend_pdf import PdfPages

    set_path = outdir / f"{_slug(project.name)}-drawing-set.pdf"
    with PdfPages(set_path) as pdf:
        for s in sheets:
            pdf.savefig(s.fig)
    produced["pdf"].append(set_path.name)
    for s in sheets:
        png = outdir / "sheets" / f"{s.number}.png"
        s.fig.savefig(png, dpi=project.sheet.get("png_dpi", 150))
        produced["png"].append(f"sheets/{png.name}")
        import matplotlib.pyplot as plt
        plt.close(s.fig)
    return produced


def _site_tables(sh: Sheet, project: Project, ctx, findings, x, y):
    b = project.building
    rows = []
    sysu = project.system
    if b is None:
        if project.parcel is not None:
            rows.append(("Site area", fmt_area(project.parcel.area, sysu)))
        if project.stadium:
            from .. import stadium as st
            fieldg, secs = st.from_spec(project, ctx)
            for sec in secs:
                rows.append((sec.name[:24], f"{sec.seats()['total']:,} seats"))
            rows.append(("New seats total", f"{sum(x.seats()['total'] for x in secs):,}"))
        rows.append(("Code profile", project.code.get("profile", "IBC-2021")))
        y = sh.table(x, y, rows, col_w=(1.35 * IN, 1.15 * IN), title="SITE DATA")
        _findings_and_sources(sh, ctx, findings, x, y)
        return
    fp = b.to_site(b.footprint())
    if project.parcel is not None:
        rows.append(("Lot area", fmt_area(project.parcel.area, sysu)))
        rows.append(("Building footprint", fmt_area(fp.area, sysu)))
        rows.append(("Lot coverage", f"{fp.area / project.parcel.area:.1%}"))
        rows.append(("Gross floor area", fmt_area(b.gross_area(), sysu)))
        rows.append(("FAR", f"{b.gross_area() / project.parcel.area:.2f}"))
    rows.append(("Occupancy / Type", f"{b.occupancy} / {b.construction_type}"))
    rows.append(("Code profile", project.code.get("profile", "IRC-2021")))
    for k, v in project.zoning.items():
        if k == "setbacks":
            for s, d in v.items():
                from ..units import fmt_len
                rows.append((f"Setback {s}", fmt_len(d, sysu)))
        elif k == "max_height":
            from ..units import fmt_len
            rows.append(("Max height", fmt_len(v, sysu)))
        elif k not in ("district",):
            rows.append((k.replace("_", " ").capitalize(), str(v)))
    if "district" in project.zoning:
        rows.insert(0, ("Zoning district", project.zoning["district"]))
    y = sh.table(x, y, rows, col_w=(1.35 * IN, 1.15 * IN), title="SITE DATA")
    _findings_and_sources(sh, ctx, findings, x, y)


def _findings_and_sources(sh, ctx, findings, x, y):
    if findings:
        fails = [f for f in findings if f.status == "FAIL"]
        warns = [f for f in findings if f.status == "WARN"]
        rows = [("Checks", f"{len(fails)} fail / {len(warns)} warn")]
        for f in (fails + warns)[:14]:
            rows.append((f.status, f.item[:34]))
        y = sh.table(x, y - 0.2 * IN, rows, col_w=(0.45 * IN, 2.05 * IN), title="COORDINATION")
    if ctx is not None:
        rows = [(a[:44], "") for a in ctx.attribution]
        rows.append(("* = height estimated (untagged)", ""))
        sh.table(x, y - 0.2 * IN, rows, col_w=(2.5 * IN, 0.0), title="DATA SOURCES")


def _slug(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in s).strip("-")
