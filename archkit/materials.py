"""Physically-based material library (Principled BSDF parameters).

Values are measured/standard albedos (linear sRGB) so renders read correctly
under a physical sky: e.g. fresh concrete ~0.35-0.45, asphalt ~0.05-0.10,
grass ~0.20, clear float glass IOR 1.52.  Override per project in
`building.materials:` with any of these keys.
"""

from __future__ import annotations

LIB: dict[str, dict] = {
    # --- facades ---------------------------------------------------------
    "facade":        {"base": (0.78, 0.76, 0.72), "rough": 0.85, "bump": 0.15, "noise": 0.04},
    "stucco_white":  {"base": (0.80, 0.79, 0.76), "rough": 0.90, "bump": 0.25, "noise": 0.05},
    "brick_red":     {"base": (0.36, 0.13, 0.08), "rough": 0.90, "bump": 0.6, "brick": True,
                      "mortar": (0.55, 0.53, 0.50)},
    "brick_buff":    {"base": (0.62, 0.50, 0.36), "rough": 0.90, "bump": 0.6, "brick": True,
                      "mortar": (0.65, 0.63, 0.60)},
    "concrete":      {"base": (0.42, 0.41, 0.39), "rough": 0.80, "bump": 0.2, "noise": 0.08},
    "board_formed":  {"base": (0.45, 0.44, 0.42), "rough": 0.85, "bump": 0.5, "noise": 0.08,
                      "boards": 0.15},
    "wood_cedar":    {"base": (0.42, 0.24, 0.13), "rough": 0.70, "bump": 0.3, "boards": 0.14,
                      "noise": 0.10},
    "fiber_cement":  {"base": (0.30, 0.31, 0.31), "rough": 0.75, "bump": 0.2, "boards": 0.20},
    "metal_panel":   {"base": (0.20, 0.21, 0.22), "rough": 0.35, "metal": 0.9, "boards": 0.40},
    "zinc":          {"base": (0.48, 0.50, 0.52), "rough": 0.40, "metal": 1.0},
    "limestone":     {"base": (0.70, 0.66, 0.57), "rough": 0.80, "bump": 0.3, "noise": 0.06},
    "plaster":       {"base": (0.85, 0.84, 0.81), "rough": 0.90},
    # --- openings --------------------------------------------------------
    "glass":         {"base": (0.95, 0.97, 0.97), "rough": 0.0, "transmission": 1.0, "ior": 1.52},
    "glass_lowe":    {"base": (0.80, 0.88, 0.90), "rough": 0.0, "transmission": 0.95, "ior": 1.52},
    "frame_black":   {"base": (0.03, 0.03, 0.03), "rough": 0.40, "metal": 0.6},
    "frame_white":   {"base": (0.85, 0.85, 0.85), "rough": 0.40},
    "door":          {"base": (0.25, 0.15, 0.08), "rough": 0.50, "boards": 0.0},
    # --- roofs -----------------------------------------------------------
    "roof_membrane": {"base": (0.55, 0.55, 0.54), "rough": 0.80},
    "roof_shingle":  {"base": (0.10, 0.10, 0.10), "rough": 0.95, "bump": 0.7, "noise": 0.1},
    "standing_seam": {"base": (0.12, 0.13, 0.14), "rough": 0.35, "metal": 1.0, "boards": 0.45},
    "slab":          {"base": (0.50, 0.49, 0.47), "rough": 0.8},
    # --- site / context --------------------------------------------------
    "context":       {"base": (0.62, 0.61, 0.59), "rough": 0.85},
    "context_roof":  {"base": (0.45, 0.45, 0.45), "rough": 0.85},
    "asphalt":       {"base": (0.07, 0.07, 0.075), "rough": 0.90, "bump": 0.3, "noise": 0.06},
    "sidewalk":      {"base": (0.50, 0.49, 0.47), "rough": 0.85, "noise": 0.05},
    "grass":         {"base": (0.10, 0.20, 0.05), "rough": 0.95, "bump": 0.4, "noise": 0.15},
    "terrain":       {"base": (0.09, 0.16, 0.05), "rough": 0.95, "bump": 0.4, "noise": 0.15},
    "water":         {"base": (0.02, 0.05, 0.06), "rough": 0.05, "transmission": 0.3, "ior": 1.33},
    "tree_crown":    {"base": (0.05, 0.13, 0.03), "rough": 0.9, "sss": 0.2},
    "tree_trunk":    {"base": (0.15, 0.10, 0.07), "rough": 0.9},
    "rail":          {"base": (0.25, 0.22, 0.20), "rough": 0.9},
}


def resolve(name: str, overrides: dict | None = None) -> dict:
    """Material params for `name`, allowing project aliases (e.g. facade: brick_red)."""
    overrides = overrides or {}
    seen = set()
    while name in overrides and isinstance(overrides[name], str) and name not in seen:
        seen.add(name)
        name = overrides[name]
    base = dict(LIB.get(name, LIB["facade"]))
    if isinstance(overrides.get(name), dict):
        base.update(overrides[name])
    base["name"] = name
    return base


def srgb(c: tuple[float, float, float]) -> tuple[float, float, float]:
    """Linear -> display sRGB (for the matplotlib preview)."""
    def f(v):
        v = max(0.0, min(1.0, v))
        return 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055
    return tuple(f(v) for v in c)
