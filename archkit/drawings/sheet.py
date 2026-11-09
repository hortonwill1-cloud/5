"""Drawing sheets: sizes, title block, true-scale viewports, scale bar, north arrow."""

from __future__ import annotations

import datetime as dt
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrow, Polygon as MPoly, Rectangle  # noqa: E402

from ..units import FT, SCALES, pick_scale  # noqa: E402
from .core import MM_TO_PT, Drawing, render_mpl  # noqa: E402

IN = 0.0254
SHEETS = {  # (width, height) metres, landscape
    "ARCH-D": (36 * IN, 24 * IN), "ARCH-C": (24 * IN, 18 * IN), "ARCH-E1": (42 * IN, 30 * IN),
    "ANSI-B": (17 * IN, 11 * IN), "ANSI-D": (34 * IN, 22 * IN),
    "A1": (0.841, 0.594), "A2": (0.594, 0.420), "A3": (0.420, 0.297), "A0": (1.189, 0.841),
}
TB_W = 3.5 * IN          # title block strip width (right edge)
MARGIN = 0.5 * IN
BIND = 1.5 * IN          # left binding margin


class Sheet:
    def __init__(self, project, number: str, title: str, size: str | None = None):
        self.project = project
        size = size or project.sheet.get("size", "ARCH-D" if project.system == "imperial" else "A1")
        self.size = size
        self.W, self.H = SHEETS[size]
        self.number, self.title = number, title
        self.fig = plt.figure(figsize=(self.W / IN, self.H / IN), dpi=300)
        self.paper = self.fig.add_axes([0, 0, 1, 1])
        self.paper.set_xlim(0, self.W)
        self.paper.set_ylim(0, self.H)
        self.paper.axis("off")
        self._border()
        self._title_block()

    # usable drawing area (metres on paper)
    @property
    def area(self) -> tuple[float, float, float, float]:
        x0, y0 = BIND, MARGIN
        return x0, y0, self.W - MARGIN - TB_W - 0.25 * IN - x0, self.H - 2 * MARGIN

    def _border(self):
        p = self.paper
        p.add_patch(Rectangle((BIND - 0.25 * IN, MARGIN - 0.25 * IN),
                              self.W - BIND - 0.0 * IN, self.H - 2 * MARGIN + 0.5 * IN,
                              fill=False, lw=0.7 * MM_TO_PT))

    def _title_block(self):
        pr = self.project
        p = self.paper
        x0 = self.W - MARGIN - TB_W + 0.25 * IN
        x1 = self.W - 0.25 * IN
        y0, y1 = MARGIN - 0.25 * IN, self.H - MARGIN + 0.25 * IN
        p.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.7 * MM_TO_PT))
        cx = (x0 + x1) / 2
        fs = lambda mm: mm * MM_TO_PT / 0.72  # noqa: E731

        def hline(y, lw=0.35):
            p.plot([x0, x1], [y, y], color="k", lw=lw * MM_TO_PT)

        y = y1 - 0.35 * IN
        p.text(cx, y, (pr.architect or "ARCHITECT").upper(), ha="center", va="top", fontsize=fs(4),
               weight="bold")
        y -= 0.45 * IN
        hline(y)
        rows = [("PROJECT", pr.name), ("ADDRESS", pr.address or "—"), ("CLIENT", pr.client or "—"),
                ("PROJECT NO.", pr.number or "—"),
                ("ISSUE", self.project.sheet.get("issue", "SCHEMATIC DESIGN")),
                ("DATE", self.project.sheet.get("date", dt.date.today().isoformat()))]
        for k, v in rows:
            y -= 0.18 * IN
            p.text(x0 + 0.1 * IN, y, k, fontsize=fs(1.8), va="top", color="#555")
            y -= 0.16 * IN
            for ln in _wrap(str(v), 30):
                p.text(x0 + 0.1 * IN, y, ln, fontsize=fs(2.6), va="top", weight="bold")
                y -= 0.17 * IN
            hline(y + 0.05 * IN, 0.13)
        # notes
        y -= 0.1 * IN
        notes = self.project.sheet.get("notes", []) + [
            "PRELIMINARY — NOT FOR CONSTRUCTION.",
            "Site context from public data; verify with boundary & topographic survey.",
            "Code checks are preliminary; AHJ review governs.",
        ]
        p.text(x0 + 0.1 * IN, y, "GENERAL NOTES", fontsize=fs(1.8), va="top", color="#555")
        y -= 0.2 * IN
        for i, n in enumerate(notes, 1):
            for j, ln in enumerate(_wrap(f"{i}. {n}", 42)):
                p.text(x0 + 0.1 * IN, y, ln, fontsize=fs(1.9), va="top")
                y -= 0.14 * IN
        # sheet number box
        hline(y0 + 1.6 * IN, 0.5)
        p.text(cx, y0 + 1.45 * IN, self.title.upper(), ha="center", va="top", fontsize=fs(3.2),
               weight="bold", wrap=True)
        p.text(cx, y0 + 0.75 * IN, self.number, ha="center", va="center", fontsize=fs(11),
               weight="bold")
        p.text(cx, y0 + 0.2 * IN, "archkit", ha="center", fontsize=fs(1.6), color="#888")

    # ------------------------------------------------------------------
    def viewport(self, dr: Drawing, rect, *, title: str, scale: str | None = None,
                 kind: str = "arch", number: int = 1, margin_mm: float = 18,
                 north_deg: float | None = None):
        """Place a drawing at true scale inside paper rect (x, y, w, h metres)."""
        x, y, w, h = rect
        bx0, by0, bx1, by1 = dr.bounds()
        mw, mh = bx1 - bx0, by1 - by0
        title_h = 0.5 * IN
        avail_w, avail_h = w - 2 * margin_mm / 1000, h - title_h - 2 * margin_mm / 1000
        scale = scale or pick_scale(mw, mh, avail_w, avail_h, self.project.system, kind)
        ratio = SCALES[scale]
        ax = self.fig.add_axes([x / self.W, (y + title_h) / self.H, w / self.W, (h - title_h) / self.H])
        ax.axis("off")
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        half_w, half_h = w * ratio / 2, (h - title_h) * ratio / 2
        ax.set_xlim(cx - half_w, cx + half_w)
        ax.set_ylim(cy - half_h, cy + half_h)
        ax.set_aspect("equal", adjustable="box")
        render_mpl(dr, ax, ratio)
        self._view_title(x, y, w, title, scale, number)
        if north_deg is not None:
            self.north_arrow(x + w - 0.9 * IN, y + h - 1.0 * IN, north_deg)
        return ax, scale

    def _view_title(self, x, y, w, title, scale, number):
        p = self.paper
        fs = lambda mm: mm * MM_TO_PT / 0.72  # noqa: E731
        r = 0.16 * IN
        cx, cy = x + 0.2 * IN + r, y + 0.25 * IN
        p.add_patch(plt.Circle((cx, cy), r, fill=False, lw=0.35 * MM_TO_PT))
        p.plot([cx - r, cx + r], [cy, cy], color="k", lw=0.25 * MM_TO_PT)
        p.text(cx, cy + r / 2, str(number), ha="center", va="center", fontsize=fs(2.2), weight="bold")
        p.text(cx, cy - r / 2, self.number, ha="center", va="center", fontsize=fs(1.6))
        p.text(cx + r + 0.1 * IN, cy + 0.02 * IN, title.upper(), fontsize=fs(3.2), weight="bold",
               va="bottom")
        p.plot([cx + r + 0.1 * IN, x + w * 0.65], [cy, cy], color="k", lw=0.5 * MM_TO_PT)
        p.text(cx + r + 0.1 * IN, cy - 0.04 * IN, f"SCALE: {scale}", fontsize=fs(2.2), va="top")
        self.scale_bar(x + w * 0.66, cy - 0.05 * IN, SCALES[scale])

    def scale_bar(self, x, y, ratio):
        """Graphic scale (survives reprographic resizing)."""
        p = self.paper
        imperial = self.project.system == "imperial"
        unit = FT if imperial else 1.0
        target_paper = 1.6 * IN
        model_len = target_paper * ratio / unit
        nice = [1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 80, 100, 200, 250, 400, 500, 1000]
        total = max(n for n in nice if n <= max(model_len, 1))
        seg = total / 4
        fs = 2.0 * MM_TO_PT / 0.72
        hgt = 0.05 * IN
        for i in range(4):
            sx = x + i * seg * unit / ratio
            p.add_patch(Rectangle((sx, y), seg * unit / ratio, hgt, facecolor="k" if i % 2 == 0 else "w",
                                  edgecolor="k", lw=0.25 * MM_TO_PT))
            p.text(sx, y - 0.03 * IN, f"{seg * i:g}", fontsize=fs, ha="center", va="top")
        p.text(x + total * unit / ratio, y - 0.03 * IN, f"{total:g} {'FT' if imperial else 'M'}",
               fontsize=fs, ha="center", va="top")

    def north_arrow(self, x, y, angle_deg_from_up: float = 0.0):
        """True-north arrow; angle = rotation of true north from sheet up (CCW +)."""
        p = self.paper
        r = 0.35 * IN
        a = math.radians(angle_deg_from_up)
        p.add_patch(plt.Circle((x, y), r, fill=False, lw=0.35 * MM_TO_PT))
        tip = (x - r * math.sin(a), y + r * math.cos(a))
        left = (x - 0.3 * r * math.cos(a) + 0.5 * r * math.sin(a), y - 0.3 * r * math.sin(a) - 0.5 * r * math.cos(a))
        right = (x + 0.3 * r * math.cos(a) + 0.5 * r * math.sin(a), y + 0.3 * r * math.sin(a) - 0.5 * r * math.cos(a))
        p.add_patch(MPoly([tip, left, (x, y), ], closed=True, fc="k", ec="k", lw=0.3))
        p.add_patch(MPoly([tip, right, (x, y)], closed=True, fc="w", ec="k", lw=0.3))
        p.text(x - 1.3 * r * math.sin(a), y + 1.3 * r * math.cos(a), "N", ha="center", va="center",
               fontsize=4 * MM_TO_PT / 0.72, weight="bold", rotation=math.degrees(a))
        p.text(x, y - r - 0.08 * IN, "TRUE NORTH", ha="center", va="top", fontsize=1.6 * MM_TO_PT / 0.72)

    def table(self, x, y, rows, col_w=(1.6 * IN, 1.4 * IN), title=None):
        p = self.paper
        fs = 2.0 * MM_TO_PT / 0.72
        rh = 0.17 * IN
        if title:
            p.text(x, y, title, fontsize=2.6 * MM_TO_PT / 0.72, weight="bold", va="bottom")
            y -= 0.04 * IN
        for i, row in enumerate(rows):
            yy = y - (i + 1) * rh
            p.plot([x, x + sum(col_w)], [yy, yy], color="k", lw=0.13 * MM_TO_PT)
            cx = x
            for c, wdt in zip(row, col_w):
                p.text(cx + 0.04 * IN, yy + rh * 0.3, str(c), fontsize=fs)
                cx += wdt
        return y - (len(rows) + 1) * rh

    def save(self, path):
        self.fig.savefig(path)
        plt.close(self.fig)


def _wrap(s: str, n: int) -> list[str]:
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]
