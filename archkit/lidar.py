"""USGS 3DEP LiDAR (Entwine Point Tiles on AWS) -> measured site context.

Reads only the octree nodes that cover the site from
``s3-us-west-2.amazonaws.com/usgs-lidar-public/<project>/`` (public, CC0) and
derives, in the project-local frame:

* a bare-earth terrain grid from ground returns (ASPRS class 2),
* per-building roof surfaces (gridded max of structure returns), giving real
  heights and real roof shapes — pitched roofs, stadium stands, canopies,
* trees (local maxima of high-vegetation returns, class 3-5) with height and
  crown size.

Accuracy: 3DEP QL2 or better (≤ 10 cm RMSE vertical, ≥ 2 pts/m²).  Building
heights are measured, not estimated; provenance is recorded as ``LiDAR``.
"""

from __future__ import annotations

import io
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from .geo import CACHE_DIR, USER_AGENT, LocalFrame, http_json

BASE = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public"
RESOURCES = "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson"

GROUND, LOW_VEG, MED_VEG, HIGH_VEG, BUILDING, NOISE, WATER = 2, 3, 4, 5, 6, 7, 9
STRUCTURE = (0, 1, BUILDING, 17)  # unclassified/never-classified often holds stands & canopies


def find_projects(lon: float, lat: float) -> list[dict]:
    """EPT projects whose footprint contains the point, newest/densest first."""
    from shapely.geometry import Point, shape

    d = http_json(RESOURCES, timeout=120)
    p = Point(lon, lat)
    out = []
    for f in d["features"]:
        try:
            if shape(f["geometry"]).contains(p):
                out.append({"name": f["properties"]["name"], "count": f["properties"].get("count", 0),
                            "url": f["properties"]["url"]})
        except Exception:  # noqa: BLE001,S112
            continue
    # prefer the most recent acquisition (year in the project name), then the more specific one
    import re

    def year(name):
        ys = [int(y) for y in re.findall(r"(?<!\d)(19\d\d|20\d\d)(?!\d)", name)]
        return max(ys) if ys else 0
    for r in out:
        r["year"] = year(r["name"])
    out.sort(key=lambda r: (-r["year"], r["count"]))
    return out


def _get(url: str) -> bytes:
    key = CACHE_DIR / "ept" / url.replace("https://", "").replace("/", "_")
    if key.exists():
        return key.read_bytes()
    r = requests.get(url, timeout=180, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_bytes(r.content)
    return r.content


class EPT:
    def __init__(self, project: str):
        self.project = project
        self.root = f"{BASE}/{project}"
        self.meta = json.loads(_get(f"{self.root}/ept.json"))
        self.cube = self.meta["bounds"]  # [minx, miny, minz, maxx, maxy, maxz]
        self.span = self.meta["span"]
        self.srs = self.meta["srs"].get("horizontal", "3857")

    def node_bounds(self, key: str):
        d, x, y, z = map(int, key.split("-"))
        minx, miny, minz, maxx, maxy, maxz = self.cube
        w = (maxx - minx) / 2 ** d
        h = (maxy - miny) / 2 ** d
        return minx + x * w, miny + y * h, minx + (x + 1) * w, miny + (y + 1) * h

    def _hierarchy(self, key: str) -> dict:
        return json.loads(_get(f"{self.root}/ept-hierarchy/{key}.json"))

    def nodes(self, bbox, max_depth: int) -> list[str]:
        """Keys of nodes intersecting bbox (in EPT SRS) down to max_depth."""
        bx0, by0, bx1, by1 = bbox
        found = []

        def walk(hier: dict, key: str):
            if key not in hier:
                return
            d = int(key.split("-")[0])
            x0, y0, x1, y1 = self.node_bounds(key)
            if x1 < bx0 or x0 > bx1 or y1 < by0 or y0 > by1:
                return
            if hier[key] == -1:
                hier = self._hierarchy(key)
            if hier.get(key, 0) > 0:
                found.append(key)
            if d >= max_depth:
                return
            _, x, y, z = map(int, key.split("-"))
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        walk(hier, f"{d + 1}-{2 * x + dx}-{2 * y + dy}-{2 * z + dz}")

        walk(self._hierarchy("0-0-0-0"), "0-0-0-0")
        return found

    def depth_for(self, spacing_m: float, lat: float) -> int:
        width = self.cube[3] - self.cube[0]
        scale = 1 / math.cos(math.radians(lat)) if self.srs == "3857" else 1.0
        return max(0, math.ceil(math.log2(width / self.span / (spacing_m * scale))))

    def read(self, key: str) -> np.ndarray:
        import laspy

        las = laspy.read(io.BytesIO(_get(f"{self.root}/ept-data/{key}.laz")))
        return np.column_stack([las.x, las.y, las.z, np.asarray(las.classification, float),
                                np.asarray(las.number_of_returns, float)])


def fetch_points(frame: LocalFrame, radius_m: float, project: str | None = None,
                 spacing_m: float = 0.5, progress=None) -> tuple[np.ndarray, str]:
    """All returns within radius as (n, 5) [x_local, y_local, z, class, n_returns] + project."""
    from pyproj import Transformer

    if project is None:
        projs = find_projects(frame.lon0, frame.lat0)
        if not projs:
            raise RuntimeError("no USGS 3DEP EPT coverage at this location")
        project = projs[0]["name"]
    ept = EPT(project)
    to_ept = Transformer.from_crs("EPSG:4326", f"EPSG:{ept.srs}", always_xy=True)
    s, w, n, e = frame.bbox_lonlat(radius_m)
    xs, ys = to_ept.transform([w, e, w, e], [s, s, n, n])
    bbox = (min(xs), min(ys), max(xs), max(ys))
    keys = ept.nodes(bbox, ept.depth_for(spacing_m, frame.lat0))
    if progress:
        progress(f"LiDAR {project}: {len(keys)} nodes")
    with ThreadPoolExecutor(max_workers=8) as ex:
        arrays = list(ex.map(ept.read, keys))
    pts = np.vstack(arrays) if arrays else np.zeros((0, 5))
    m = (pts[:, 0] >= bbox[0]) & (pts[:, 0] <= bbox[2]) & (pts[:, 1] >= bbox[1]) & (pts[:, 1] <= bbox[3])
    pts = pts[m]
    # EPT SRS -> local tmerc
    proj = (f"+proj=tmerc +lat_0={frame.lat0} +lon_0={frame.lon0} +k=1 +x_0=0 +y_0=0 "
            "+ellps=WGS84 +units=m +no_defs")
    t = Transformer.from_crs(f"EPSG:{ept.srs}", proj, always_xy=True)
    lx, ly = t.transform(pts[:, 0], pts[:, 1])
    out = np.column_stack([lx, ly, pts[:, 2], pts[:, 3], pts[:, 4]])
    out = out[np.hypot(out[:, 0], out[:, 1]) <= radius_m * 1.42]
    return out, project


# ---------------------------------------------------------------------------
# Rasters
# ---------------------------------------------------------------------------

def _grid_stat(x, y, z, x0, y0, cell, nx, ny, how="max"):
    ix = np.clip(((x - x0) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((y - y0) / cell).astype(int), 0, ny - 1)
    out = np.full((ny, nx), np.nan)
    idx = iy * nx + ix
    order = np.argsort(z) if how == "max" else np.argsort(-z)
    flat = out.ravel()
    flat[idx[order]] = z[order]  # last write wins -> max (or min)
    return out


def _grid_median(x, y, z, x0, y0, cell, nx, ny):
    """Per-cell median (robust to stray foliage / bird returns)."""
    ix = np.clip(((x - x0) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((y - y0) / cell).astype(int), 0, ny - 1)
    idx = iy * nx + ix
    order = np.lexsort((z, idx))
    idx_s, z_s = idx[order], z[order]
    starts = np.r_[0, np.flatnonzero(np.diff(idx_s)) + 1]
    ends = np.r_[starts[1:], len(idx_s)]
    out = np.full(nx * ny, np.nan)
    # upper-middle element: biased slightly high so eaves are not eroded
    out[idx_s[starts]] = z_s[np.minimum(starts + (ends - starts) * 2 // 3, ends - 1)]
    return out.reshape(ny, nx)


def _despike(a: np.ndarray, tol: float = 1.5) -> np.ndarray:
    p = np.pad(a, 2, mode="edge")
    stack = np.stack([p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(5) for j in range(5)])
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(stack, 0)
    return np.where(a > med + tol, med, a)


def _fill_nearest(a: np.ndarray, iters: int = 200) -> np.ndarray:
    a = a.copy()
    for _ in range(iters):
        nan = np.isnan(a)
        if not nan.any():
            break
        p = np.pad(a, 1, constant_values=np.nan)
        stack = np.stack([p[1:-1, :-2], p[1:-1, 2:], p[:-2, 1:-1], p[2:, 1:-1]])
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(stack, 0)
        a[nan] = m[nan]
    return a


def _median3(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    stack = np.stack([p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)])
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stack, 0)


def terrain_grid(pts: np.ndarray, radius_m: float, cell: float) -> dict:
    g = pts[pts[:, 3] == GROUND]
    if len(g) < 100:
        raise RuntimeError("too few ground returns")
    n = int(2 * radius_m / cell) + 1
    x0 = -radius_m
    lo = _grid_stat(g[:, 0], g[:, 1], g[:, 2], x0 - cell / 2, x0 - cell / 2, cell, n, n, how="min")
    z = _median3(_fill_nearest(lo))
    xs = x0 + np.arange(n) * cell
    grid = [[float(xs[i]), float(xs[j]), round(float(z[j, i]), 3)] for j in range(n) for i in range(n)]
    oi = int(round(-x0 / cell))
    return {"origin_m": float(z[oi, oi]), "grid": grid, "cell_m": cell,
            "source": "USGS 3DEP LiDAR ground returns (class 2)"}


def roof_surface(poly, pts_struct: np.ndarray, pts_ground: np.ndarray, cell: float = 1.0):
    """Gridded roof heights over a footprint polygon (absolute z). None if no returns."""
    from shapely import contains_xy

    minx, miny, maxx, maxy = poly.bounds
    nx = max(2, int(math.ceil((maxx - minx) / cell)) + 1)
    ny = max(2, int(math.ceil((maxy - miny) / cell)) + 1)
    sel = (pts_struct[:, 0] >= minx - cell) & (pts_struct[:, 0] <= maxx + cell) & \
          (pts_struct[:, 1] >= miny - cell) & (pts_struct[:, 1] <= maxy + cell)
    s = pts_struct[sel]
    if len(s):
        s = s[contains_xy(poly.buffer(0.3), s[:, 0], s[:, 1])]
    if len(s) < 4:
        return None
    hi = _grid_median(s[:, 0], s[:, 1], s[:, 2], minx - cell / 2, miny - cell / 2, cell, nx, ny)
    # where only ground was returned (open courtyards, the playing field) keep ground level
    gsel = (pts_ground[:, 0] >= minx - cell) & (pts_ground[:, 0] <= maxx + cell) & \
           (pts_ground[:, 1] >= miny - cell) & (pts_ground[:, 1] <= maxy + cell)
    g = pts_ground[gsel]
    if len(g):
        gl = _grid_stat(g[:, 0], g[:, 1], g[:, 2], minx - cell / 2, miny - cell / 2, cell, nx, ny, "min")
        hi = np.where(np.isnan(hi) & ~np.isnan(gl), gl, hi)
    raw = _despike(_fill_nearest(hi))
    z = _median3(raw)   # display surface; the median filter shaves ridges, so fit on `raw`
    return {"x0": minx, "y0": miny, "cell": cell, "nx": nx, "ny": ny,
            "z": [round(float(v), 2) for v in z.ravel()],
            "z_raw": [round(float(v), 2) for v in raw.ravel()]}


def fit_roof(poly, grid: dict, ground_abs: float, max_rms: float = 0.35,
             min_inliers: float = 0.6) -> dict | None:
    """Fit flat / gable / hip / shed roof to a LiDAR roof grid.  Returns params or None
    (complex roof: keep the measured surface).  Heights absolute (same datum as grid)."""
    import shapely

    mrr = poly.minimum_rotated_rectangle
    c = list(mrr.exterior.coords)[:4]
    e1 = np.subtract(c[1], c[0])
    e2 = np.subtract(c[2], c[1])
    if np.linalg.norm(e1) < np.linalg.norm(e2):
        e1, e2 = e2, e1
    L, W = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    if W < 3 or poly.area / mrr.area < 0.8:
        return None
    u = e1 / L
    v = np.array([-u[1], u[0]])
    cx, cy = mrr.centroid.x, mrr.centroid.y
    nx, ny, cell = grid["nx"], grid["ny"], grid["cell"]
    xs = grid["x0"] + np.arange(nx) * cell
    ys = grid["y0"] + np.arange(ny) * cell
    X, Y = np.meshgrid(xs, ys)
    Z = np.asarray(grid.get("z_raw", grid["z"])).reshape(ny, nx)
    inner = poly.buffer(-0.6)
    if inner.is_empty:
        return None
    m = shapely.contains_xy(inner, X.ravel(), Y.ravel())
    if m.sum() < 8:
        return None
    px, py, pz = X.ravel()[m] - cx, Y.ravel()[m] - cy, Z.ravel()[m]
    s_, t_ = px * u[0] + py * u[1], px * v[0] + py * v[1]
    d_gable = W / 2 - np.abs(t_)
    d_hip = np.minimum(W / 2 - np.abs(t_), L / 2 - np.abs(s_))
    feats = {"flat": None, "gable": d_gable, "hip": d_hip, "shed": t_ + W / 2}
    best = None
    for name, f in feats.items():
        A = np.ones((len(pz), 1)) if f is None else np.column_stack([np.ones(len(pz)), f])
        keep = np.ones(len(pz), bool)
        for _ in range(6):  # trimmed least squares: drop ground-hit edge cells & foliage
            coef, *_ = np.linalg.lstsq(A[keep], pz[keep], rcond=None)
            res = np.abs(A @ coef - pz)
            thr = max(0.3, 2.5 * 1.4826 * np.median(res[keep]))
            new_keep = res < thr
            if new_keep.sum() < 8 or (new_keep == keep).all():
                break
            keep = new_keep
        inl = keep.mean()
        rms = float(np.sqrt(np.mean(res[keep] ** 2)))
        k = float(coef[1]) if f is not None else 0.0
        if f is not None and not (0.05 < k < 1.6):  # 0.6:12 .. 19:12
            continue
        if inl < min_inliers:
            continue
        penalty = {"flat": 0.0, "gable": 0.04, "hip": 0.06, "shed": 0.08}[name] + 0.3 * (1 - inl)
        if best is None or rms + penalty < best[0]:
            best = (rms + penalty, rms, name, float(coef[0]), k, inl)
    if best is None or best[1] > max_rms or best[3] - ground_abs < 2.0:
        return None
    _, rms, name, e, k, inl = best
    return {"type": name, "eave": e, "pitch": k, "u": u.tolist(), "center": [cx, cy],
            "L": L, "W": W, "rms": round(rms, 3), "inliers": round(float(inl), 2)}


def detect_trees(pts: np.ndarray, ground_fn, radius_m: float, cell: float = 1.5,
                 min_h: float = 4.0) -> list[dict]:
    v = pts
    if len(v) < 20:
        return []
    n = int(2 * radius_m / cell) + 1
    x0 = -radius_m
    hi = _grid_stat(v[:, 0], v[:, 1], v[:, 2], x0, x0, cell, n, n, "max")
    gx = x0 + (np.arange(n) + 0.5) * cell
    out = []
    h = np.full_like(hi, np.nan)
    for j in range(n):
        for i in range(n):
            if not np.isnan(hi[j, i]):
                h[j, i] = hi[j, i] - ground_fn(gx[i], gx[j])
    w = 3  # local-max window radius in cells (~4.5 m)
    hp = np.pad(np.nan_to_num(h, nan=-1), w, constant_values=-1)
    for j in range(n):
        for i in range(n):
            hv = h[j, i]
            if np.isnan(hv) or hv < min_h:
                continue
            win = hp[j:j + 2 * w + 1, i:i + 2 * w + 1]
            if hv >= win.max():
                # crown: canopy cells above 60% of tree height within the window
                crown = np.count_nonzero(win > hv * 0.6) * cell * cell
                d = 2 * math.sqrt(crown / math.pi)
                out.append({"x": float(gx[i]), "y": float(gx[j]), "height_m": round(float(hv), 1),
                            "crown_m": round(float(min(max(d, 3.0), min(14.0, hv * 0.9))), 1)})
    return out


def apply_to_context(ctx, spacing_m: float = 0.5, project: str | None = None,
                     terrain: bool = True, trees: bool = True, progress=print) -> dict:
    """Measure building heights/roofs, terrain and trees for a SiteContext in place."""
    from shapely.geometry import Point

    from .site import Feature

    pts, proj = fetch_points(ctx.frame, ctx.radius_m, project, spacing_m, progress)
    progress(f"LiDAR: {len(pts):,} returns")
    stats = {"project": proj, "points": int(len(pts))}
    if terrain:
        cell = max(1.0, ctx.radius_m / 120)
        ctx.elevation = terrain_grid(pts, ctx.radius_m, cell)
        ctx.elevation["source"] += f" [{proj}]"
        if hasattr(ctx, "_lut"):
            del ctx._lut
    ground = pts[pts[:, 3] == GROUND]
    classified = np.count_nonzero(np.isin(pts[:, 3], (BUILDING, HIGH_VEG, MED_VEG))) > 0.02 * len(pts)
    stats["classification"] = "ASPRS classes" if classified else "ground-only; single-return heuristic"
    if classified:
        struct = pts[np.isin(pts[:, 3], STRUCTURE)]
        veg = pts[np.isin(pts[:, 3], (MED_VEG, HIGH_VEG))]
    else:
        # Unclassified surveys: solid surfaces return one echo per pulse; foliage scatters
        # into several.  Roofs = single-return non-ground; vegetation = multi-return.
        other = pts[np.isin(pts[:, 3], (0, 1))]
        struct = other[other[:, 4] <= 1]
        veg = other[other[:, 4] > 1]
    z0 = ctx.elevation["origin_m"] if ctx.elevation else float(np.median(ground[:, 2]))
    n_meas = 0
    for f in ctx.of("building") + ctx.of("building_part"):
        g = f.geom
        cell = 1.0 if g.area < 60000 else 2.0   # 1 m keeps stadium tiers crisp
        polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
        grids = []
        for k, p in enumerate(polys):
            r = roof_surface(p, struct, ground, cell)
            if r:
                r["part"] = k
                grids.append(r)
        if not grids:
            continue
        c = g.representative_point()
        gz_abs = ctx.ground_z(c.x, c.y) + z0
        zs = np.concatenate([np.asarray(r["z"]) for r in grids])
        top = float(np.nanpercentile(zs, 98)) - gz_abs
        if top < 1.5:
            continue
        if len(grids) == 1 and len(polys) == 1:
            fit = fit_roof(polys[0], grids[0], gz_abs)
            if fit is None and g.area < 600:
                # houses/garages: a clean approximate roof reads better than noisy truth
                fit = fit_roof(polys[0], grids[0], gz_abs, max_rms=1.5, min_inliers=0.35)
                if fit is None:
                    z = np.asarray(grids[0]["z"])
                    fit = {"type": "flat", "eave": float(np.nanpercentile(z, 60)), "pitch": 0.0,
                           "u": [1.0, 0.0], "center": [g.centroid.x, g.centroid.y],
                           "L": 1.0, "W": 1.0, "rms": None, "inliers": None}
                fit["approximate"] = True
            if fit:
                f.props["roof_fit"] = fit
                peak = fit["eave"] + fit["pitch"] * (fit["W"] if fit["type"] == "shed" else
                                                     fit["W"] / 2 if fit["type"] != "flat" else 0)
                top = peak - gz_abs
        f.props["height_m"] = round(top, 2)
        f.props["height_source"] = f"LiDAR (3DEP {proj})"
        f.props["roof_grids"] = grids
        f.props["ground_z"] = round(gz_abs - z0, 3)
        f.props["z_datum"] = z0
        n_meas += 1
    stats["buildings_measured"] = n_meas
    if trees:
        ctx.features = [f for f in ctx.features if f.kind != "tree"]
        tr = detect_trees(veg, lambda x, y: ctx.ground_z(x, y) + z0, ctx.radius_m)
        bfp = [f.geom for f in ctx.of("building")]
        # no trees inside stadia / on pitches (light standards, rigging scatter like foliage)
        bfp += [f.geom for f in ctx.of("green") if f.props.get("use") in ("stadium", "pitch")]
        from shapely.strtree import STRtree
        tree_idx = STRtree(bfp) if bfp else None
        for t in tr:
            p = Point(t["x"], t["y"])
            if tree_idx is not None and len(tree_idx.query(p, predicate="intersects")):
                continue
            if t["height_m"] > 14 and t["crown_m"] < 4.0:
                continue  # tall & narrow: light pole / mast, not a tree
            ctx.features.append(Feature("tree", p, {"height_m": t["height_m"], "crown_m": t["crown_m"]},
                                        f"LiDAR (3DEP {proj})"))
        stats["trees"] = len(ctx.of("tree"))
    ctx.attribution.append(f"USGS 3DEP LiDAR {proj} (public domain): {stats}")
    return stats
