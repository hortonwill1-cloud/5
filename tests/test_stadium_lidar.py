import math

import numpy as np
import pytest
from shapely.geometry import box

from archkit import lidar
from archkit.stadium import BowlSection, Field, capacity_factors, egress_checks


def test_sightline_solver_meets_c_target_every_row():
    s = BowlSection("t", "south", first_row_offset=9.0, first_row_height=1.8, rows=40,
                    tread=0.84, c_target=0.090, riser_increment=0.010, length=48)
    rows = s.solve()
    for prev, r in zip(rows, rows[1:]):
        assert r["C"] >= 0.090 - 1e-9                       # never below target
        assert r["C"] < 0.090 + r["riser"]                  # rounding adds < one increment's worth
        assert math.isclose(r["riser"] * 100, round(r["riser"] * 100), abs_tol=1e-9)  # 10 mm steps
        # achieved C recomputed from geometry: sightline from eye(n) to focal point clears eye(n-1)
        clear = r["E"] * prev["D"] / r["D"] - prev["E"]
        assert math.isclose(clear, r["C"], abs_tol=1e-12)
    assert all(b["riser"] >= a["riser"] - 1e-9 for a, b in zip(rows[1:], rows[2:]))  # bowl steepens
    assert 20 < s.rake_deg < 34


def test_egress_flags_insufficient_aisles():
    s = BowlSection("t", "south", first_row_offset=9.0, first_row_height=1.8, rows=40,
                    tread=0.84, length=48.8, aisle_width=1.22)
    s.solve()
    f = {x.item.split(" ", 1)[1]: x for x in egress_checks(s)}
    assert f["aisle egress capacity"].status == "FAIL"
    s.aisle_spacing = 9.1
    f = {x.item.split(" ", 1)[1]: x for x in egress_checks(s)}
    assert f["aisle egress capacity"].status == "PASS"


def test_capacity_factor_tables():
    assert capacity_factors(60000, True, True)[:2] == (0.08, 0.06)
    assert capacity_factors(4000, True, False)[:2] == (0.200, 0.150)
    assert capacity_factors(30000, True, False)[:2] == (0.060, 0.044)
    assert capacity_factors(500, False, False)[:2] == (0.30, 0.20)


def test_field_markings_dimensions():
    f = Field((10.0, -5.0), 30.0)
    marks = f.markings()
    goal_lines = [p for m, p in marks if m == "paint_white" and abs(p.area - 8 * 0.0254 * 160 * 0.3048) < 0.01]
    assert len(goal_lines) == 2
    d = goal_lines[0].centroid.distance(goal_lines[1].centroid)
    assert math.isclose(d, 100 * 0.9144, abs_tol=0.01)      # goal line to goal line = 100 yd


def _synthetic_gable(n=4000, seed=0):
    """Points on a 12 x 8 m gable roof (eave 100 m abs, 6:12) plus ground and noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 12, n)
    y = rng.uniform(0, 8, n)
    z = 100 + 0.5 * (4 - np.abs(y - 4)) + rng.normal(0, 0.03, n)
    pts = np.column_stack([x, y, z, np.ones(n), np.ones(n)])
    gx, gy = rng.uniform(-10, 22, 3000), rng.uniform(-10, 18, 3000)
    outside = ~((gx > 0) & (gx < 12) & (gy > 0) & (gy < 8))
    ground = np.column_stack([gx[outside], gy[outside], np.full(outside.sum(), 94.0),
                              np.full(outside.sum(), 2.0), np.ones(outside.sum())])
    return pts, ground


def test_roof_grid_and_fit_recover_gable():
    roof, ground = _synthetic_gable()
    poly = box(0, 0, 12, 8)
    grid = lidar.roof_surface(poly, roof, ground, cell=1.0)
    fit = lidar.fit_roof(poly, grid, 94.0)
    assert fit is not None and fit["type"] == "gable"
    assert math.isclose(fit["pitch"], 0.5, abs_tol=0.06)       # 6:12
    assert math.isclose(fit["eave"], 100.0, abs_tol=0.25)
    assert fit["rms"] < 0.2


def test_roof_fit_rejects_ground_contaminated_edges():
    roof, ground = _synthetic_gable(seed=1)
    poly = box(-1.0, -1.0, 13.0, 9.0)                           # footprint 1 m too big (OSM offset)
    grid = lidar.roof_surface(poly, roof, ground, cell=1.0)
    fit = lidar.fit_roof(poly, grid, 94.0)
    assert fit is not None and fit["type"] in ("gable", "hip")
    assert fit["inliers"] >= 0.6


def test_terrain_grid_from_ground_returns():
    rng = np.random.default_rng(3)
    x, y = rng.uniform(-50, 50, 20000), rng.uniform(-50, 50, 20000)
    z = 250 + 0.02 * x - 0.01 * y                             # 2 % / 1 % slopes
    pts = np.column_stack([x, y, z, np.full_like(x, 2), np.ones_like(x)])
    t = lidar.terrain_grid(pts, 40, 2.0)
    from archkit.geo import LocalFrame
    from archkit.site import SiteContext
    ctx = SiteContext(LocalFrame(38, -84.5), 40, elevation=t)
    assert math.isclose(t["origin_m"], 250, abs_tol=0.1)
    assert math.isclose(ctx.ground_z(20, -10), 0.02 * 20 + 0.01 * 10, abs_tol=0.1)


@pytest.mark.parametrize("name,year", [("CO_DRCOG_2_2020", 2020), ("KY_FullState", 0),
                                       ("USGS_LPC_CO_SoPlatteRiver_Lot5_2013_LAS_2015", 2015)])
def test_project_year_parse(name, year, monkeypatch):
    monkeypatch.setattr(lidar, "http_json", lambda *a, **k: {"features": [
        {"properties": {"name": name, "count": 1, "url": ""},
         "geometry": {"type": "Polygon", "coordinates": [[[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]]}}]})
    assert lidar.find_projects(0, 0)[0]["year"] == year
