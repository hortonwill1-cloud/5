import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from archkit.camera import look_at, solve_camera
from archkit.geo import LocalFrame
from archkit.sun import solar_position
from archkit.units import fmt_ft_in, pick_scale


def test_ft_in_formatting():
    assert fmt_ft_in(0.3048 * 12.5) == "12'-6\""
    assert fmt_ft_in(0.3048 * 10 + 0.0254 * 3.5) == "10'-3 1/2\""
    assert fmt_ft_in(0) == "0'-0\""


def test_local_frame_is_metrically_true():
    """Local distances must match WGS84 geodesic distances to < 1 mm over 500 m."""
    from pyproj import Geod

    f = LocalFrame(39.69, -104.98)
    g = Geod(ellps="WGS84")
    for x, y in [(500, 0), (0, 500), (350, -350), (-420, 260)]:
        lon, lat = f.to_lonlat(x, y)
        _, _, d = g.inv(f.lon0, f.lat0, lon, lat)
        assert abs(d - math.hypot(x, y)) < 0.001
        bx, by = f.to_local(lon, lat)
        assert abs(bx - x) < 1e-6 and abs(by - y) < 1e-6


def test_solar_position_reference_values():
    # Solstice solar noon elevation = 90 - (lat - 23.44)
    tz = timezone(timedelta(hours=-6))
    best = max((solar_position(39.74, -104.99, datetime(2026, 6, 21, 12, m, tzinfo=tz))
                for m in range(0, 60)), key=lambda t: t[1])
    best2 = max((solar_position(39.74, -104.99, datetime(2026, 6, 21, 13, m, tzinfo=tz))
                 for m in range(0, 60)), key=lambda t: t[1])
    el = max(best[1], best2[1])
    assert abs(el - (90 - (39.74 - 23.44))) < 0.1
    # Morning sun is in the east, afternoon in the west
    az_am, _ = solar_position(39.74, -104.99, datetime(2026, 3, 20, 9, 0, tzinfo=tz))
    az_pm, _ = solar_position(39.74, -104.99, datetime(2026, 3, 20, 17, 0, tzinfo=tz))
    assert 90 < az_am < 160 and 200 < az_pm < 270


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_camera_solve_recovers_pose(seed):
    rng = np.random.default_rng(seed)
    cam = look_at([-40 + 30 * seed, -60, 1.6], [5, 8, 6], 24 + 10 * seed, 36, 4000, 3000)
    W = rng.uniform([-10, -10, 0], [20, 20, 15], (10, 3))
    px, _ = cam.project(W)
    solved, rms = solve_camera(px + rng.normal(0, 0.5, px.shape), W, 4000, 3000)
    assert np.linalg.norm(solved.eye - cam.eye) < 1.5
    assert abs(solved.focal_mm - cam.focal_mm) < 1.0
    assert solved.forward @ cam.forward > 0.9999
    assert rms < 1.0


def test_pick_scale():
    # 20 m wide plan on a 0.6 m-wide area -> 1/4" = 1'-0" (1:48) fits, 3/8" does not
    assert pick_scale(20, 10, 0.6, 0.5, "imperial") == '1/4" = 1\'-0"'
    assert pick_scale(20, 10, 0.6, 0.5, "metric") == "1:50"
