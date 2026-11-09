"""Unit conversion and architectural formatting."""

from __future__ import annotations

from fractions import Fraction

FT = 0.3048
INCH = 0.0254
SQFT = FT * FT

UNIT_TO_M = {"m": 1.0, "mm": 0.001, "cm": 0.01, "ft": FT, "in": INCH}


def to_m(value: float, unit: str) -> float:
    return float(value) * UNIT_TO_M[unit]


def fmt_ft_in(meters: float, precision: int = 16) -> str:
    """Format a length as architectural feet-inches, e.g. 12'-6 1/2"."""
    sign = "-" if meters < 0 else ""
    total_in = abs(meters) / INCH
    frac = Fraction(round(total_in * precision), precision)
    feet = int(frac // 12)
    rem = frac - feet * 12
    whole_in = int(rem)
    part = rem - whole_in
    inch_str = str(whole_in)
    if part:
        inch_str += f" {part.numerator}/{part.denominator}"
    return f"{sign}{feet}'-{inch_str}\""


def fmt_len(meters: float, system: str) -> str:
    if system == "imperial":
        return fmt_ft_in(meters)
    return f"{meters * 1000:.0f}" if meters < 10 else f"{meters:.3f}"


def fmt_area(m2: float, system: str) -> str:
    if system == "imperial":
        return f"{m2 / SQFT:,.0f} SF"
    return f"{m2:,.1f} m²"


# Drawing scales: name -> model metres per paper metre.
SCALES = {
    # imperial architectural
    '1/16" = 1\'-0"': 192,
    '3/32" = 1\'-0"': 128,
    '1/8" = 1\'-0"': 96,
    '3/16" = 1\'-0"': 64,
    '1/4" = 1\'-0"': 48,
    '3/8" = 1\'-0"': 32,
    '1/2" = 1\'-0"': 24,
    '1" = 1\'-0"': 12,
    # imperial engineering (site)
    '1" = 10\'': 120,
    '1" = 20\'': 240,
    '1" = 30\'': 360,
    '1" = 40\'': 480,
    '1" = 50\'': 600,
    '1" = 100\'': 1200,
    # metric
    "1:20": 20, "1:50": 50, "1:100": 100, "1:200": 200,
    "1:250": 250, "1:500": 500, "1:1000": 1000,
}

IMPERIAL_ARCH = ['1" = 1\'-0"', '1/2" = 1\'-0"', '3/8" = 1\'-0"', '1/4" = 1\'-0"',
                 '3/16" = 1\'-0"', '1/8" = 1\'-0"', '3/32" = 1\'-0"', '1/16" = 1\'-0"']
IMPERIAL_SITE = ['1" = 10\'', '1" = 20\'', '1" = 30\'', '1" = 40\'', '1" = 50\'', '1" = 100\'']
METRIC = ["1:20", "1:50", "1:100", "1:200", "1:250", "1:500", "1:1000"]


def pick_scale(extent_w: float, extent_h: float, area_w: float, area_h: float,
               system: str, kind: str = "arch") -> str:
    """Largest standard scale at which a model extent (m) fits a paper area (m)."""
    if system == "imperial":
        names = IMPERIAL_SITE if kind == "site" else IMPERIAL_ARCH
    else:
        names = METRIC
    for name in names:  # ordered largest scale (most detail) first
        r = SCALES[name]
        if extent_w / r <= area_w and extent_h / r <= area_h:
            return name
    return names[-1]
