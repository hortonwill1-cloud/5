import dataclasses

import pytest

from archkit.render.proof import check_view, contact_sheet, unproofed, view_signature


def test_signature_tracks_camera_not_quality(loaded):
    pr, _ = loaded
    v = pr.views[0]
    s0 = view_signature(v)
    assert view_signature(dataclasses.replace(v, samples=4096)) == s0            # quality only
    assert view_signature(dataclasses.replace(v, resolution=(1920, 1080))) == s0  # same aspect
    assert view_signature(dataclasses.replace(v, eye=(v.eye[0] + 1, *v.eye[1:]))) != s0
    assert view_signature(dataclasses.replace(v, datetime="2026-12-21T09:30:00")) != s0


def test_final_render_refused_until_proofed(loaded, tmp_path):
    import json

    from archkit.render import NotProofed, render_blender

    pr, ctx = loaded
    v = pr.views[0]
    with pytest.raises(NotProofed):
        render_blender(pr, ctx, None, v, tmp_path / "renders", percent=100)
    (tmp_path / "proof").mkdir()
    (tmp_path / "proof" / "proofs.json").write_text(json.dumps({v.name: {"signature": view_signature(v)}}))
    assert unproofed([v], tmp_path) == []
    moved = dataclasses.replace(v, target=(0.0, 0.0, 0.0))
    assert unproofed([moved], tmp_path) == [v.name]                               # camera changed


def test_check_view_catches_eye_inside_building(loaded):
    pr, ctx = loaded
    v = dataclasses.replace(pr.views[0], eye=(0.0, -15.0, 1.7), name="bad")      # inside 8 m neighbour
    errors, warnings, info = check_view(pr, ctx, v)
    assert errors and "inside existing building" in errors[0]
    ok = dataclasses.replace(pr.views[0], eye=(0.0, -15.0, 12.0), name="drone")  # above its roof
    assert check_view(pr, ctx, ok)[0] == []


def test_contact_sheet_renders_errors_and_images(loaded, tmp_path):
    from PIL import Image

    pr, _ = loaded
    img = tmp_path / "a.png"
    Image.new("RGB", (480, 270), "skyblue").save(img)
    tiles = [{"view": pr.views[0], "img": img, "errors": [], "warnings": ["low sun"],
              "info": {"sun": (90, 5), "hfov_deg": 73.7, "eye_agl_m": 1.7}, "secs": 12},
             {"view": pr.views[1], "img": None, "errors": ["camera inside building"],
              "warnings": [], "info": {}, "secs": 0}]
    out = contact_sheet(pr, tiles, tmp_path / "sheet.jpg")
    w, h = Image.open(out).size
    assert w > 900 and h > 300
