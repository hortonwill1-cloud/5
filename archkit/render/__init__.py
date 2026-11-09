"""Rendering: Blender Cycles (photoreal) and a built-in preview renderer."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .. import materials
from ..camera import Camera, look_at, solve_camera
from ..model3d import Model, export_obj
from ..site import SiteContext
from ..spec import Project, View
from ..sun import parse_when, solar_position

BLENDER_CANDIDATES = [os.environ.get("BLENDER", ""), "blender",
                      "/opt/blender/blender-4.5.15-linux-x64/blender",
                      "/Applications/Blender.app/Contents/MacOS/Blender"]


def find_blender() -> str | None:
    for c in BLENDER_CANDIDATES:
        if not c:
            continue
        p = shutil.which(c) or (c if Path(c).exists() else None)
        if p:
            return p
    for p in sorted(Path("/opt/blender").glob("blender-*/blender")):
        return str(p)
    return None


def view_camera(project: Project, ctx: SiteContext | None, v: View) -> tuple[Camera, dict]:
    """Camera for a view: solved from photo control points, or eye/target (z above local grade)."""
    info: dict = {}
    if v.camera_solve:
        cs = v.camera_solve
        k = 0.3048 if project.units == "ft" else 1.0
        px = np.array([p["px"] for p in cs["points"]], float)
        world = np.array([[c * k for c in p["site"]] for p in cs["points"]], float)
        w, h = cs.get("image_size", v.resolution)
        cam, rms = solve_camera(px, world, int(w), int(h), v.sensor_mm)
        info.update(rms_px=rms, solved=True, n_points=len(px))
        cam.width, cam.height = v.resolution
        cam.focal_px *= v.resolution[0] / int(w)
        return cam, info
    gz = (lambda x, y: ctx.ground_z(x, y)) if ctx is not None else (lambda x, y: 0.0)
    eye = (v.eye[0], v.eye[1], gz(v.eye[0], v.eye[1]) + v.eye[2])
    tgt = (v.target[0], v.target[1], gz(v.target[0], v.target[1]) + v.target[2])
    return look_at(eye, tgt, v.focal_mm, v.sensor_mm, *v.resolution), info


def sun_for(project: Project, v: View) -> tuple[float, float]:
    when = parse_when(v.datetime, v.utc_offset_h, project.timezone)
    return solar_position(project.lat or 0.0, project.lon or 0.0, when)


def render_blender(project: Project, ctx: SiteContext | None, model: Model, v: View,
                   outdir: Path, *, percent: int = 100, samples: int | None = None,
                   save_blend: bool = False, exposure: float | None = None) -> Path:
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender not found. Install Blender 4.x and set $BLENDER, "
                           "or use `archkit render --preview`.")
    outdir.mkdir(parents=True, exist_ok=True)
    obj = export_obj(model, outdir / "model" / "scene.obj", project.materials)
    cam, info = view_camera(project, ctx, v)
    az, el = sun_for(project, v)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = cam.R[0], cam.R[1], -cam.R[2], cam.eye
    used = sorted({f.material for f in model.faces})
    job = {
        "obj": str(obj.resolve()),
        "materials": {m: materials.resolve(m, project.materials) for m in used},
        "sun_az": az, "sun_el": el,
        "altitude_m": (ctx.elevation["origin_m"] if ctx and ctx.elevation else 0.0),
        "camera": {"focal_mm": cam.focal_mm, "sensor_mm": cam.sensor_mm,
                   "matrix_world": M.tolist()},
        "resolution": list(v.resolution), "percent": percent,
        "samples": samples or v.samples,
        "photomontage": bool(v.background_photo),
        "exposure": exposure if exposure is not None else project.sheet.get("render_exposure", -4.5),
        "exposure_offset": v.exposure_offset,
        "out": str((outdir / f"{v.name}.png").resolve()),
    }
    if save_blend:
        job["save_blend"] = str((outdir / f"{v.name}.blend").resolve())
    if v.textures:
        from ..assets import textures_for
        job["textures"] = textures_for(used, "4k" if percent >= 100 else "2k")
    if v.ortho and ctx is not None and project.lat is not None:
        try:
            from ..model3d import apply_clips
            from ..ortho import fetch_ortho, mask_for
            o = fetch_ortho(ctx.frame, ctx.radius_m)
            polys = list(project.clip_context)
            if project.building is not None:
                polys.append(project.building.to_site(project.building.footprint()).buffer(1.0))
            if project.stadium:
                from .. import stadium as st
                fieldg, secs = st.from_spec(project, ctx)
                polys += [st.bowl_polygon(fieldg, s_).buffer(1.0) for s_ in secs]
            if polys:
                o["mask"] = mask_for(o, polys, outdir / "model" / "ortho_mask.png")
            # Eye-level views: an overhead photo shows tree canopy over streets and baked
            # shade, so drape it on roofs only; elevated/aerial views use it everywhere.
            eye_agl = cam.eye[2] - ctx.ground_z(cam.eye[0], cam.eye[1])
            o["materials"] = (["context_roof"] if eye_agl < 20 else
                              ["terrain", "grass", "asphalt", "sidewalk", "context_roof", "water"])
            job["ortho"] = o
            info["ortho"] = o["source"] + f" on {', '.join(o['materials'])}"
        except Exception as e:  # noqa: BLE001
            print(f"  orthoimagery unavailable ({e}); procedural ground")
    job["camera_fx"] = v.camera_fx
    if v.tree_models and ctx is not None and ctx.of("tree"):
        try:
            from ..assets import TREE_MODELS, model_gltf
            job["tree_models"] = [str(model_gltf(s_).resolve()) for s_ in TREE_MODELS]
            excl = None
            if project.building is not None:
                excl = project.building.to_site(project.building.footprint()).buffer(0.5)
            job["trees"] = [{"x": t.geom.x, "y": t.geom.y, "z": ctx.ground_z(t.geom.x, t.geom.y),
                             "h": t.props.get("height_m", 8.0), "crown": t.props.get("crown_m", 5.0)}
                            for t in ctx.of("tree") if excl is None or not excl.contains(t.geom)]
        except Exception as e:  # noqa: BLE001
            print(f"  tree models unavailable ({e}); proxy canopies")
    if v.sky != "physical" and el > 0:
        from ..assets import choose_sky
        try:
            job["sky"] = choose_sky(el, blender, v.sky, "4k" if percent >= 100 else "2k")
            job["exposure_offset"] = v.exposure_offset
        except Exception as e:  # noqa: BLE001
            print(f"  HDRI sky unavailable ({e}); using physical sky")
    jp = outdir / f"{v.name}.job.json"
    jp.write_text(json.dumps(job, indent=1))
    script = Path(__file__).with_name("blender_scene.py")
    r = subprocess.run([blender, "-b", "--factory-startup", "-P", str(script), "--", str(jp)],
                       capture_output=True, text=True)
    out = Path(job["out"])
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"Blender failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    if v.background_photo:
        out = composite(out, Path(v.background_photo), outdir / f"{v.name}-photomontage.png")
    elif v.camera_fx:
        lens_post(out, out)
    if job.get("sky"):
        info["sky_hdri"] = job["sky"]["slug"]
        info["sky_sun_elevation_mismatch_deg"] = job["sky"]["elev_mismatch_deg"]
        info["shadows"] = "exact sun lamp at computed position (HDRI sun removed)"
    _write_metadata(project, v, cam, info, az, el, outdir)
    return out


def lens_post(src: Path, dst: Path, vignette: float = 0.15, ca_px: float = 0.6,
              grain: float = 0.008, seed: int = 7) -> Path:
    """Photographic lens/sensor signature: vignetting, lateral chromatic aberration, grain."""
    from PIL import Image

    im = Image.open(src)
    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    H, W, _ = arr.shape
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = (W - 1) / 2, (H - 1) / 2
    r = np.hypot((xx - cx) / cx, (yy - cy) / cy) / math.sqrt(2)
    # lateral CA: scale red out / blue in by a fraction of a pixel at the corners
    out = arr.copy()
    for ch, k in ((0, ca_px), (2, -ca_px)):
        sc = 1 + k / max(W, H) * 2
        sx = np.clip(cx + (xx - cx) / sc, 0, W - 1).astype(np.float32)
        sy = np.clip(cy + (yy - cy) / sc, 0, H - 1).astype(np.float32)
        x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
        x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
        fx, fy = sx - x0, sy - y0
        c = arr[..., ch]
        out[..., ch] = (c[y0, x0] * (1 - fx) * (1 - fy) + c[y0, x1] * fx * (1 - fy)
                        + c[y1, x0] * (1 - fx) * fy + c[y1, x1] * fx * fy)
    out *= (1 - vignette * r ** 2.2)[..., None]                       # natural vignette
    rng = np.random.default_rng(seed)
    lum = out.mean(2, keepdims=True)
    out += rng.normal(0, grain, (H, W, 1)).astype(np.float32) * np.sqrt(np.clip(lum, 0.02, 1))
    Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)).save(dst)
    return dst


def composite(render: Path, photo: Path, out: Path) -> Path:
    from PIL import Image

    fg = Image.open(render).convert("RGBA")
    bg = Image.open(photo).convert("RGBA").resize(fg.size, Image.LANCZOS)
    Image.alpha_composite(bg, fg).convert("RGB").save(out, quality=95)
    return out


def _write_metadata(project, v, cam, info, az, el, outdir):
    meta = {
        "view": v.name, "datetime": v.datetime, "utc_offset_h": v.utc_offset_h,
        "sun_azimuth_deg": round(az, 3), "sun_elevation_deg": round(el, 3),
        "camera_eye_site_m": [round(float(c), 3) for c in cam.eye],
        "camera_forward": [round(float(c), 4) for c in cam.forward],
        "focal_mm": round(cam.focal_mm, 2), "sensor_mm": cam.sensor_mm,
        "horizontal_fov_deg": round(2 * np.degrees(np.arctan(cam.width / 2 / cam.focal_px)), 2),
        **info,
    }
    if project.lat is not None:
        from ..geo import LocalFrame
        lon, lat = LocalFrame(project.lat, project.lon).to_lonlat(cam.eye[0], cam.eye[1])
        meta["camera_latlon"] = [round(lat, 7), round(lon, 7)]
    (outdir / f"{v.name}.view.json").write_text(json.dumps(meta, indent=1))


# ---------------------------------------------------------------------------
# Built-in preview renderer (no Blender): perspective, sun-lit, painter's algorithm
# ---------------------------------------------------------------------------

def render_preview(project: Project, ctx: SiteContext | None, model: Model, v: View,
                   out: Path, scale: float = 0.5) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    cam, info = view_camera(project, ctx, v)
    W, H = int(v.resolution[0] * scale), int(v.resolution[1] * scale)
    cam = Camera(cam.eye, cam.R, cam.focal_px * scale, W, H, cam.sensor_mm)
    az, el = sun_for(project, v)
    from ..sun import sun_vector
    sv = np.array(sun_vector(az, el))
    polys, cols, depths = [], [], []
    for f in model.faces:
        p, z = cam.project(f.verts)
        if (z <= 0.2).any():
            continue
        if p[:, 0].max() < 0 or p[:, 0].min() > W or p[:, 1].max() < 0 or p[:, 1].min() > H:
            continue
        a, b, c = f.verts[0], f.verts[1], f.verts[2]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln == 0:
            continue
        n /= ln
        if n @ (cam.eye - f.verts.mean(0)) < 0:
            n = -n  # two-sided
        mp = materials.resolve(f.material, project.materials)
        base = np.array(mp["base"])
        lam = max(0.0, n @ sv) if el > 0 else 0.0
        sky = 0.35 + 0.25 * n[2]
        col = base * (1.6 * lam + sky)
        if mp.get("transmission"):
            col = np.array([0.35, 0.45, 0.55]) * (0.6 + 0.4 * n[2]) + 0.15 * lam
        polys.append(p)
        cols.append(materials.srgb(tuple(np.clip(col, 0, 1))))
        depths.append(z.mean())
    order = np.argsort(depths)[::-1]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    grad = np.linspace(0, 1, 256)[:, None]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("sky", ["#5b8fc7", "#a9c8e6", "#e8eef3"])
    ax.imshow(grad, extent=(0, W, H, 0), cmap=cmap, aspect="auto", zorder=0)
    pc = PolyCollection([polys[i] for i in order], facecolors=[cols[i] for i in order],
                        edgecolors=[cols[i] for i in order], linewidths=0.3, zorder=1)
    ax.add_collection(pc)
    ax.text(8, H - 8, f"{project.name} — {v.name} — PREVIEW  sun az {az:.1f}° el {el:.1f}°",
            fontsize=8, color="white", va="bottom",
            bbox=dict(fc="black", alpha=0.4, lw=0))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    _write_metadata(project, v, cam, info, az, el, out.parent)
    return out
