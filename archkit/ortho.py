"""Aerial orthoimagery (USDA NAIP via USGS The National Map, public domain, 0.3-0.6 m).

Fetched for the site square, georeferenced to the project-local frame, and draped
on terrain, roads and context roofs in renders.  Areas the proposal demolishes or
builds on are masked so the photo never shows structures that will not exist.

Caveat: an orthophoto bakes in the lighting and shadows of its flight date; the
renderer multiplies it with neutral scanned detail and lets the exact sun lamp add
the design-time shadows, but faint flight-time tree shadows remain on the ground.
"""

from __future__ import annotations

import math
from pathlib import Path

import requests

from .geo import CACHE_DIR, USER_AGENT, LocalFrame

NAIP = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"


def fetch_ortho(frame: LocalFrame, radius_m: float, res_m: float = 0.3, max_px: int = 4000) -> dict:
    """Download a north-up RGB orthophoto covering [-R, R]^2 (local m).  Returns georef."""
    from pyproj import Transformer

    to3857 = Transformer.from_crs(
        f"+proj=tmerc +lat_0={frame.lat0} +lon_0={frame.lon0} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m",
        "EPSG:3857", always_xy=True)
    xs, ys = to3857.transform([-radius_m, radius_m, -radius_m, radius_m],
                              [-radius_m, -radius_m, radius_m, radius_m])
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    k = 1 / math.cos(math.radians(frame.lat0))           # web-mercator scale at the site
    px = min(max_px, int(math.ceil(2 * radius_m / res_m)))
    out = CACHE_DIR / "ortho" / f"naip_{frame.lat0:.6f}_{frame.lon0:.6f}_{int(radius_m)}_{px}.jpg"
    if not out.exists():
        r = requests.get(NAIP, timeout=300, headers={"User-Agent": USER_AGENT}, params={
            "bbox": f"{x0},{y0},{x1},{y1}", "bboxSR": 3857, "imageSR": 3857,
            "size": f"{px},{px}", "format": "jpg", "bandIds": "0,1,2",
            "interpolation": "RSP_BilinearInterpolation", "f": "image"})
        r.raise_for_status()
        if not r.headers.get("content-type", "").startswith("image"):
            raise RuntimeError(f"NAIP export failed: {r.text[:300]}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(r.content)
    # local extent of the exported 3857 square (web mercator ~ uniform scale k locally)
    back = Transformer.from_crs("EPSG:3857", to3857.source_crs, always_xy=True)
    lx, ly = back.transform([x0, x1], [y0, y1])
    return {"path": str(out.resolve()), "x0": lx[0], "y0": ly[0], "w": lx[1] - lx[0],
            "h": ly[1] - ly[0], "res_m": (lx[1] - lx[0]) / px, "scale_k": k,
            "source": "USDA NAIP via USGS The National Map (public domain)"}


def mask_for(ortho: dict, polygons, out_path: Path, feather_m: float = 1.5) -> str:
    """White = use photo, black = masked (proposal / demolition areas)."""
    from PIL import Image, ImageDraw, ImageFilter

    im = Image.open(ortho["path"])
    W, H = im.size
    m = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(m)

    def px(x, y):
        return ((x - ortho["x0"]) / ortho["w"] * W, (1 - (y - ortho["y0"]) / ortho["h"]) * H)

    for poly in polygons:
        geoms = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
        for g in geoms:
            d.polygon([px(x, y) for x, y in g.exterior.coords], fill=0)
    m = m.filter(ImageFilter.GaussianBlur(max(1, feather_m / ortho["res_m"])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(out_path)
    return str(out_path.resolve())
