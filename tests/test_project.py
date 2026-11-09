import math

import pytest

from archkit import spec
from archkit.checks import run_checks
from archkit.model3d import building_model, export_obj, full_model
from archkit.units import FT


def test_spec_loads_in_metres(loaded):
    pr, ctx = loaded
    b = pr.building
    assert len(b.levels) == 2
    fp = b.footprint()
    # 52' x 34' centreline + 9" wall -> 52.75' x 34.75' outside
    assert math.isclose(fp.area, (52.75 * FT) * (34.75 * FT), rel_tol=1e-6)
    assert math.isclose(pr.parcel.area, 50 * 125 * FT * FT, rel_tol=1e-6)


def test_osm_parse(loaded):
    _, ctx = loaded
    kinds = sorted(f.kind for f in ctx.features)
    assert kinds.count("building") == 3 and "road" in kinds and "tree" in kinds
    tall = [f for f in ctx.of("building") if f.props.get("height_source") == "tagged height"]
    assert tall and tall[0].height == 8.0
    lv = [f for f in ctx.of("building") if f.props.get("height_source") == "from levels"]
    assert lv and math.isclose(lv[0].height, 6.4)


def _status(findings, item_prefix):
    return [f.status for f in findings if f.item.startswith(item_prefix)]


def test_example_passes_zoning_and_flags_demolition(loaded):
    pr, ctx = loaded
    f = run_checks(pr, ctx)
    assert _status(f, "setbacks") == ["PASS"]
    assert _status(f, "lot coverage") == ["PASS"]
    assert "PASS" in _status(f, "building height (highest point)")
    assert any(x.item.startswith("existing building on footprint") for x in f)
    assert _status(f, "egress door provided") == ["PASS"]
    assert all(s == "PASS" for s in _status(f, "Level 2 201 Primary Bedroom emergency"))
    model_fails = [x for x in f if x.category == "model" and x.status == "FAIL"]
    assert not model_fails, [x.row() for x in model_fails]


def test_setback_violation_detected(loaded):
    pr, ctx = loaded
    x, y = pr.building.placement
    pr.building.placement = (x + 12 * FT, y)  # push 12' toward the street
    f = run_checks(pr, ctx)
    assert _status(f, "setbacks") == ["FAIL"]


def test_bad_stair_and_door_flagged(loaded):
    pr, ctx = loaded
    lv = pr.building.levels[0]
    lv.stairs[0].riser = 8.5 * 0.0254
    lv.openings[0].width = 2.5 * FT  # egress door D1
    f = run_checks(pr, ctx)
    assert "FAIL" in _status(f, "Level 1 stair 1 geometry")
    assert "FAIL" in _status(f, "Level 1 door D1 width")


def test_model_and_obj(loaded, tmp_path):
    pr, ctx = loaded
    m = building_model(pr, ctx)
    lo, hi = m.bounds()
    # peak = FF + 20' walls + roof heel + 5/12 * 17.375' (half outside span)
    from archkit.model3d import resolve_finish_floor
    ff = resolve_finish_floor(pr, ctx)
    assert math.isclose(hi[2], ff + pr.building.roof_peak(), abs_tol=0.01)
    full = full_model(pr, ctx)
    p = export_obj(full, tmp_path / "m.obj")
    txt = p.read_text()
    assert "usemtl glass_lowe" in txt and "o context/buildings" in txt


def test_drawings_and_dxf(loaded, tmp_path):
    import ezdxf

    from archkit.drawings import build_set

    pr, ctx = loaded
    out = build_set(pr, ctx, tmp_path, run_checks(pr, ctx))
    assert (tmp_path / out["pdf"][0]).stat().st_size > 50_000
    doc = ezdxf.readfile(tmp_path / "dxf" / "A1.1-level-1-plan.dxf")
    layers = {e.dxf.layer for e in doc.modelspace()}
    assert {"A-WALL-PATT", "A-DOOR", "A-GLAZ", "A-ANNO-DIMS"} <= layers or \
        {"A-WALL", "A-DOOR", "A-GLAZ", "A-ANNO-DIMS"} <= layers
    # DXF is in real inches: the overall south-wall dimension must read 52'-9" = 633"
    dims = [d.get_measurement() for d in doc.modelspace().query("DIMENSION")]
    assert any(abs(d - 633.0) < 0.01 for d in dims)


def test_shadow_length_matches_trig():
    from shapely.geometry import box

    from archkit.shadows import prism_shadow

    s = prism_shadow(box(0, 0, 1, 1), 0, 10.0, az=180.0, el=45.0)  # sun due south
    minx, miny, maxx, maxy = s.bounds
    assert math.isclose(maxy, 1 + 10.0, abs_tol=1e-6)  # shadow falls north, length = h / tan(45)


def test_preview_render_and_metadata(loaded, tmp_path):
    import json

    from archkit.render import render_preview

    pr, ctx = loaded
    m = full_model(pr, ctx)
    v = pr.views[0]
    out = render_preview(pr, ctx, m, v, tmp_path / "p.png", scale=0.1)
    assert out.exists()
    meta = json.loads((tmp_path / f"{v.name}.view.json").read_text())
    # June 21 09:30 MDT in Denver: sun in the east, ~43 deg up
    assert 85 < meta["sun_azimuth_deg"] < 100 and 38 < meta["sun_elevation_deg"] < 48


def test_view_camera_from_photo_control_points(loaded):
    """A view defined by photo control points reproduces the camera that took the 'photo'."""
    import numpy as np

    from archkit.camera import look_at
    from archkit.render import view_camera
    from archkit.spec import View

    pr, ctx = loaded
    truth = look_at([24.0, -17.0, 1.6], [-2.0, 0.0, 4.0], 26.0, 36.0, 6000, 4000)
    fp = pr.building.to_site(pr.building.footprint())
    corners = [(x, y, z) for x, y in list(fp.exterior.coords)[:4] for z in (0.2, 6.0)]
    corners += [(19.3, -30.0, 0.0), (19.3, 30.0, 0.0)]  # kerb points
    px, _ = truth.project(np.array(corners))
    pts = [{"px": p.tolist(), "site": [c / FT for c in w]} for p, w in zip(px, corners)]
    v = View("verified", (0, 0, 0), (1, 0, 0), resolution=(3000, 2000),
             camera_solve={"image_size": [6000, 4000], "points": pts})
    cam, info = view_camera(pr, ctx, v)
    assert info["rms_px"] < 0.01
    assert np.linalg.norm(cam.eye - truth.eye) < 0.01
    assert abs(cam.focal_mm - 26.0) < 0.01
