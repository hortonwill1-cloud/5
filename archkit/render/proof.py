"""Proof pass: fast low-resolution drafts of every view on one labelled contact sheet.

Run before any final render so framing, sun and camera problems are caught in
seconds instead of after a 5-30 minute render.  Each view is checked (camera
inside a building, line of sight to the target, sun below the horizon), drafted
at ~480 px with few samples, and stamped with a signature of its camera settings.
Final renders refuse views whose current settings were never proofed.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from ..spec import View

PROOF_DIR = "proof"


def view_signature(v: View) -> str:
    """Hash of everything that changes framing/lighting (not quality settings)."""
    key = {
        "eye": [round(c, 4) for c in v.eye], "target": [round(c, 4) for c in v.target],
        "focal_mm": v.focal_mm, "sensor_mm": v.sensor_mm,
        "aspect": round(v.resolution[0] / v.resolution[1], 4),
        "datetime": v.datetime, "utc_offset_h": v.utc_offset_h,
        "background_photo": v.background_photo, "camera_solve": v.camera_solve,
        "sky": v.sky,
    }
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _stamp_path(outdir: Path) -> Path:
    return outdir / PROOF_DIR / "proofs.json"


def load_stamps(outdir: Path) -> dict:
    p = _stamp_path(outdir)
    return json.loads(p.read_text()) if p.exists() else {}


def unproofed(views: list[View], outdir: Path) -> list[str]:
    """Names of views with no proof, or whose camera/sun settings changed since proofing."""
    stamps = load_stamps(outdir)
    return [v.name for v in views if stamps.get(v.name, {}).get("signature") != view_signature(v)]


def check_view(project, ctx, v: View) -> tuple[list[str], list[str], dict]:
    """(errors, warnings, info) for a view without rendering it."""
    from . import sun_for, view_camera

    errors, warnings = [], []
    info: dict = {}
    try:
        cam, cinfo = view_camera(project, ctx, v)
        info.update(cinfo)
        gz = ctx.ground_z(cam.eye[0], cam.eye[1]) if ctx is not None else 0.0
        info["eye_agl_m"] = round(float(cam.eye[2] - gz), 1)
        info["hfov_deg"] = round(2 * math.degrees(math.atan(cam.width / 2 / cam.focal_px)), 1)
        if cinfo.get("occluded_by"):
            warnings.append("blocked by " + ", ".join(cinfo["occluded_by"]))
        if cinfo.get("rms_px") is not None and cinfo["rms_px"] > 3:
            warnings.append(f"camera solve RMS {cinfo['rms_px']:.1f} px (> 3 px)")
    except ValueError as e:
        errors.append(str(e).split(": ", 1)[-1])
    az, el = sun_for(project, v)
    info["sun"] = (round(az, 1), round(el, 1))
    if el <= 0:
        warnings.append(f"sun below horizon ({el:.1f}°) - night render")
    elif el < 8:
        warnings.append(f"low sun ({el:.1f}°): long shadows, warm light")
    if v.focal_mm < 20:
        warnings.append(f"{v.focal_mm:g} mm lens: strong wide-angle distortion")
    return errors, warnings, info


def run_proof(project, ctx, model, views: list[View], outdir: Path, width: int = 480,
              samples: int = 16, use_blender: bool = True, progress=print) -> Path:
    from . import find_blender, render_blender, render_preview

    pdir = outdir / PROOF_DIR
    pdir.mkdir(parents=True, exist_ok=True)
    blender = find_blender() if use_blender else None
    stamps = load_stamps(outdir)
    tiles = []
    for v in views:
        t0 = time.time()
        errors, warnings, info = check_view(project, ctx, v)
        img = None
        if not errors:
            pct = max(5, min(100, int(round(100 * width / v.resolution[0]))))
            try:
                if blender:
                    img = render_blender(project, ctx, model, v, pdir, percent=pct,
                                         samples=samples, check_proof=False)
                else:
                    img = render_preview(project, ctx, model, v, pdir / f"{v.name}.png",
                                         scale=pct / 100)
            except Exception as e:  # noqa: BLE001
                errors.append(f"render failed: {str(e).splitlines()[0][:120]}")
        secs = time.time() - t0
        tiles.append({"view": v, "img": img, "errors": errors, "warnings": warnings,
                      "info": info, "secs": secs})
        status = "ERROR" if errors else ("WARN" if warnings else "ok")
        progress(f"  proof {v.name:<16} {status:<5} {secs:5.1f}s  " + "; ".join(errors + warnings))
        if not errors:
            stamps[v.name] = {"signature": view_signature(v), "proofed_at": time.strftime("%Y-%m-%d %H:%M"),
                              "warnings": warnings}
    _stamp_path(outdir).write_text(json.dumps(stamps, indent=1))
    sheet = contact_sheet(project, tiles, pdir / "contact-sheet.jpg", width)
    return sheet


def contact_sheet(project, tiles: list[dict], out: Path, width: int = 480) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    def font(sz, bold=False):
        for name in ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",):
            try:
                import matplotlib
                return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), sz)
            except OSError:
                pass
        return ImageFont.load_default()

    f_title, f_head, f_txt = font(22, True), font(15, True), font(12)
    cols = min(3, max(1, len(tiles)))
    rows = math.ceil(len(tiles) / cols)
    tile_w = width
    tile_h = max(int(width * t["view"].resolution[1] / t["view"].resolution[0]) for t in tiles)
    cap_h = 92
    pad = 16
    W = cols * (tile_w + pad) + pad
    H = 56 + rows * (tile_h + cap_h + pad) + pad
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    d.text((pad, 14), f"PROOF — {project.name}", font=f_title, fill="black")
    d.text((W - pad - 330, 20), time.strftime("%Y-%m-%d %H:%M") + "   approve framing before finals",
           font=f_txt, fill="#555")
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (tile_w + pad)
        y = 56 + r * (tile_h + cap_h + pad)
        v = t["view"]
        th = int(tile_w * v.resolution[1] / v.resolution[0])
        if t["img"] is not None and Path(t["img"]).exists():
            im = Image.open(t["img"]).convert("RGB").resize((tile_w, th))
            sheet.paste(im, (x, y))
        else:
            d.rectangle([x, y, x + tile_w, y + th], fill="#f3d6d6")
            d.text((x + 12, y + th // 2 - 8), "NOT RENDERED", font=f_head, fill="#a00")
        colour = "#b00000" if t["errors"] else ("#b36b00" if t["warnings"] else "#1d6b1d")
        d.rectangle([x, y, x + tile_w - 1, y + th - 1], outline=colour, width=3)
        info = t["info"]
        cy = y + th + 6
        d.text((x, cy), v.name, font=f_head, fill="black")
        sun = info.get("sun", ("?", "?"))
        line = (f"{v.focal_mm:g} mm · HFOV {info.get('hfov_deg', '?')}° · eye {info.get('eye_agl_m', '?')} m AGL · "
                f"sun az {sun[0]}° el {sun[1]}°")
        d.text((x, cy + 20), line, font=f_txt, fill="#333")
        d.text((x, cy + 36), f"{v.datetime}  ({t['secs']:.0f}s)", font=f_txt, fill="#333")
        msgs = t["errors"] + t["warnings"]
        msg = "; ".join(msgs) if msgs else "OK - ready for final"
        d.text((x, cy + 54), msg[:78], font=f_txt, fill=colour)
        if len(msg) > 78:
            d.text((x, cy + 70), msg[78:156], font=f_txt, fill=colour)
    sheet.save(out, quality=90)
    return out


def signature_table(views: list[View]) -> dict:
    return {v.name: view_signature(v) for v in views}


__all__ = ["run_proof", "check_view", "unproofed", "view_signature", "contact_sheet", "asdict"]
