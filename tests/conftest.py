import json
import shutil
from pathlib import Path

import pytest

from archkit.geo import LocalFrame

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "platt-park-infill" / "project.yaml"
LAT, LON = 39.6904378, -104.9795167


def _way(frame, wid, pts, tags):
    geom = []
    for x, y in pts:
        lon, lat = frame.to_lonlat(x, y)
        geom.append({"lat": lat, "lon": lon})
    return {"type": "way", "id": wid, "tags": tags, "geometry": geom}


@pytest.fixture
def osm_fixture(tmp_path):
    """Synthetic Overpass response: a street, two neighbours, the existing house, a tree."""
    f = LocalFrame(LAT, LON)
    els = [
        _way(f, 1, [(19.3, -200), (19.3, 200)], {"highway": "residential", "name": "Test Street"}),
        _way(f, 2, [(-6.9, -5.4), (6.9, -5.4), (6.9, 5.5), (-6.9, 5.5), (-6.9, -5.4)],
             {"building": "house", "addr:housenumber": "1435"}),
        _way(f, 3, [(-7, -20), (7, -20), (7, -10), (-7, -10), (-7, -20)],
             {"building": "house", "height": "8"}),
        _way(f, 4, [(-7, 12), (7, 12), (7, 22), (-7, 22), (-7, 12)],
             {"building": "house", "building:levels": "2"}),
    ]
    lon, lat = f.to_lonlat(-20, 0)
    els.append({"type": "node", "id": 9, "lat": lat, "lon": lon, "tags": {"natural": "tree"}})
    p = tmp_path / "osm.json"
    p.write_text(json.dumps({"elements": els}))
    return p


@pytest.fixture
def project_file(tmp_path, osm_fixture):
    """Copy of the example spec wired to the offline OSM fixture."""
    import yaml

    d = yaml.safe_load(EXAMPLE.read_text())
    d["site"]["osm_json"] = str(osm_fixture)
    p = tmp_path / "project.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


@pytest.fixture
def loaded(project_file):
    from archkit import spec
    from archkit.site import build_context

    pr = spec.load(project_file)
    ctx = build_context(pr.lat, pr.lon, 120, terrain=False, osm_json=pr.osm_json)
    spec.resolve_parcel(pr, ctx.frame)
    return pr, ctx
