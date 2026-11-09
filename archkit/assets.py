"""Poly Haven (CC0) scanned PBR textures and HDRI skies, downloaded on demand and cached.

Textures are mapped at their published real-world size (``dimensions`` in mm),
so bricks, siding courses and paving joints render at true scale.

HDRI skies are chosen to match the true sun: every candidate sky is probed
(in Blender) for the position and irradiance of its sun; the one whose sun
elevation is closest to the computed solar elevation is used, rotated to the
true solar azimuth, its painted-in sun removed, and replaced by an exactly
placed sun lamp of the HDRI's own measured irradiance (airmass-corrected).
Shadows are therefore exact while sky, clouds and reflections are photographic.
"""

from __future__ import annotations

import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .geo import CACHE_DIR, USER_AGENT

API = "https://api.polyhaven.com"
PH_DIR = CACHE_DIR / "polyhaven"

# archkit material -> Poly Haven texture (+ optional colour treatment)
TEXTURES: dict[str, dict] = {
    "brick_red":     {"slug": "red_brick_03"},
    "brick_buff":    {"slug": "yellow_brick"},
    "stucco_white":  {"slug": "white_stucco"},
    "facade":        {"slug": "white_stucco"},
    "plaster":       {"slug": "painted_plaster_wall"},
    "concrete":      {"slug": "concrete_wall_006", "mult": (1.15, 1.15, 1.2)},
    "board_formed":  {"slug": "concrete_wall_008", "recolor": (0.42, 0.41, 0.40)},
    "fiber_cement":  {"slug": "exterior_wall_cladding_03"},
    "wood_cedar":    {"slug": "weathered_plank_siding"},
    "metal_panel":   {"slug": "box_profile_metal_sheet", "recolor": (0.16, 0.17, 0.18)},
    "standing_seam": {"slug": "box_profile_metal_sheet", "recolor": (0.09, 0.095, 0.1)},
    "roof_shingle":  {"slug": "grey_roof_01"},
    "asphalt":       {"slug": "asphalt_02"},
    "sidewalk":      {"slug": "concrete_pavement"},
    "grass":         {"slug": "leafy_grass", "mult": (0.62, 1.0, 0.42)},
    "terrain":       {"slug": "leafy_grass", "mult": (0.62, 1.0, 0.42)},
    "track":         {"slug": "rubberized_track"},
    "limestone":     {"slug": "large_sandstone_blocks", "mult": (1.25, 1.2, 1.1)},
}

# Daylight skies without ground ("puresky"), clear to partly cloudy.
SKY_CANDIDATES = {
    "clear": ["syferfontein_1d_clear_puresky", "syferfontein_6d_clear_puresky",
              "syferfontein_18d_clear_puresky", "kloofendal_43d_clear_puresky",
              "qwantani_morning_puresky", "qwantani_mid_morning_puresky", "qwantani_noon_puresky",
              "qwantani_afternoon_puresky", "qwantani_late_afternoon_puresky"],
    "partly_cloudy": ["kloofendal_28d_misty_puresky", "kloofendal_38d_partly_cloudy_puresky",
                      "kloofendal_48d_partly_cloudy_puresky", "kloppenheim_06_puresky",
                      "table_mountain_1_puresky", "lonely_road_afternoon_puresky"],
    "overcast": ["kloofendal_overcast_puresky", "overcast_soil_puresky"],
}


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300, headers={"User-Agent": USER_AGENT}) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.rename(dest)
    return dest


def _json(url: str):
    p = PH_DIR / "api" / (url.replace(API, "").strip("/").replace("/", "_").replace("?", "_") + ".json")
    if p.exists():
        return json.loads(p.read_text())
    r = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(r.text)
    return r.json()


def texture(slug: str, res: str = "2k") -> dict:
    """Download diffuse/roughness/normal(GL)/displacement maps; return paths + size (m)."""
    files = _json(f"{API}/files/{slug}")
    info = _json(f"{API}/info/{slug}")
    want = {"diffuse": "Diffuse", "rough": "Rough", "normal": "nor_gl", "disp": "Displacement"}
    out = {"slug": slug}
    jobs = []
    for key, name in want.items():
        try:
            u = files[name][res]["jpg"]["url"]
        except KeyError:
            continue
        jobs.append((key, u, PH_DIR / "tex" / slug / f"{slug}_{name}_{res}.jpg"))
    with ThreadPoolExecutor(4) as ex:
        for (key, _u, _d), p in zip(jobs, ex.map(lambda j: _download(j[1], j[2]), jobs)):
            out[key] = str(p.resolve())
    dims = info.get("dimensions") or [2000, 2000]
    out["size_m"] = [float(dims[0]) / 1000.0, float(dims[1]) / 1000.0]
    return out


def textures_for(materials_used, res: str = "2k") -> dict[str, dict]:
    out = {}
    for m in materials_used:
        spec = TEXTURES.get(m)
        if not spec:
            continue
        try:
            t = texture(spec["slug"], res)
        except Exception as e:  # noqa: BLE001 - offline -> procedural fallback
            print(f"  texture {spec['slug']} unavailable ({e}); procedural fallback")
            continue
        t.update({k: v for k, v in spec.items() if k != "slug"})
        out[m] = t
    return out


def hdri_file(slug: str, res: str = "4k") -> Path:
    files = _json(f"{API}/files/{slug}")
    u = files["hdri"][res]["hdr"]["url"]
    return _download(u, PH_DIR / "hdri" / f"{slug}_{res}.hdr")


PROBE = r'''
import bpy, json, math, sys
import numpy as np
img = bpy.data.images.load(sys.argv[sys.argv.index("--") + 1])
W, H = img.size
px = np.empty(W * H * 4, np.float32); img.pixels.foreach_get(px)
a = px.reshape(H, W, 4)[:, :, :3]
lum = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
r, c = np.unravel_index(np.argmax(lum), lum.shape)
el = (r + 0.5) / H * math.pi - math.pi / 2
phi = (0.5 - (c + 0.5) / W) * 2 * math.pi          # atan2(y, x) of the sun direction
rows = (np.arange(H) + 0.5) / H * math.pi - math.pi / 2
dOm = (2 * math.pi / W) * (math.pi / H) * np.cos(rows)[:, None]
# angular distance of every pixel from the sun
cols = (0.5 - (np.arange(W) + 0.5) / W) * 2 * math.pi
ce = np.cos(rows)[:, None]; se = np.sin(rows)[:, None]
cosd = se * math.sin(el) + ce * math.cos(el) * np.cos(cols[None, :] - phi)
ang = np.degrees(np.arccos(np.clip(cosd, -1, 1)))
ring = (ang > 6) & (ang < 12)
bg = float(np.median(lum[ring])) if ring.any() else 0.0
disc = ang < 6
e_sun = float(((lum - bg).clip(0) * dOm)[disc].sum())
up = rows[:, None] > 0
lum_nosun = np.where(disc, bg, lum)
e_sky = float((lum_nosun * dOm * np.sin(rows)[:, None] * up).sum())
print("PROBE" + json.dumps({"W": W, "H": H, "el_deg": math.degrees(el), "phi_deg": math.degrees(phi),
                            "e_sun": e_sun, "e_sky_h": e_sky, "peak": float(lum[r, c])}))
'''


def probe_hdri(path: Path, blender: str) -> dict:
    cache = path.with_suffix(".probe.json")
    if cache.exists():
        return json.loads(cache.read_text())
    script = PH_DIR / "probe_hdri.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(PROBE)
    r = subprocess.run([blender, "-b", "--factory-startup", "-P", str(script), "--", str(path)],
                       capture_output=True, text=True, timeout=600)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("PROBE")), None)
    if line is None:
        raise RuntimeError(f"HDRI probe failed: {r.stdout[-1500:]} {r.stderr[-1500:]}")
    d = json.loads(line[5:])
    cache.write_text(json.dumps(d))
    return d


def airmass(el_deg: float) -> float:
    """Kasten & Young (1989) relative optical airmass."""
    z = 90 - max(el_deg, 0.5)
    return 1 / (math.cos(math.radians(z)) + 0.50572 * (96.07995 - z) ** -1.6364)


def choose_sky(sun_el: float, blender: str, style: str = "partly_cloudy", res: str = "4k") -> dict:
    """Pick the candidate HDRI whose sun elevation best matches; return probe + path."""
    cands = SKY_CANDIDATES.get(style, SKY_CANDIDATES["partly_cloudy"])
    probes = []
    for slug in cands:
        try:
            p = hdri_file(slug, "1k")
            pr = probe_hdri(p, blender)
            probes.append((abs(pr["el_deg"] - sun_el), slug, pr))
        except Exception as e:  # noqa: BLE001
            print(f"  sky {slug} unavailable: {e}")
    if not probes:
        raise RuntimeError("no HDRI skies available")
    probes.sort(key=lambda t: t[0])
    err, slug, _ = probes[0]
    path = hdri_file(slug, res)
    pr = probe_hdri(path, blender)
    tau = 0.25  # broadband optical depth, clear continental air
    e_sun_target = pr["e_sun"] * math.exp(-tau * (airmass(sun_el) - airmass(pr["el_deg"])))
    if style == "overcast":
        e_sun_target = 0.0
    return {"slug": slug, "path": str(path.resolve()), "probe": pr, "elev_mismatch_deg": round(err, 2),
            "sun_strength": e_sun_target}
