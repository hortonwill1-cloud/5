"""Backend-neutral 2D drawing entities rendered to matplotlib (true scale) and DXF.

Layer names follow the US National CAD Standard / AIA CAD Layer Guidelines.
Line weights are ISO pen sizes in mm at plotted size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MM_TO_PT = 72 / 25.4

# name: (lineweight mm, colour hex, linetype, aci colour for DXF)
LAYERS: dict[str, tuple[float, str, str, int]] = {
    "A-WALL":           (0.70, "#000000", "solid", 7),
    "A-WALL-PATT":      (0.00, "#3a3a3a", "solid", 8),
    "A-WALL-ELEV":      (0.35, "#000000", "solid", 7),
    "A-DOOR":           (0.35, "#000000", "solid", 2),
    "A-DOOR-SWNG":      (0.13, "#000000", "solid", 2),
    "A-GLAZ":           (0.25, "#000000", "solid", 4),
    "A-GLAZ-SILL":      (0.18, "#000000", "solid", 4),
    "A-FLOR-STRS":      (0.25, "#000000", "solid", 3),
    "A-FLOR-OVHD":      (0.18, "#000000", "dashed", 8),
    "A-FLOR-IDEN":      (0.18, "#000000", "solid", 7),
    "A-ROOF":           (0.50, "#000000", "solid", 6),
    "A-ROOF-OTLN":      (0.25, "#000000", "dashed", 6),
    "A-SECT-MCUT":      (1.00, "#000000", "solid", 7),
    "A-SECT-PATT":      (0.00, "#1a1a1a", "solid", 250),
    "A-ELEV-OTLN":      (0.70, "#000000", "solid", 7),
    "A-ELEV-BEYD":      (0.18, "#555555", "solid", 8),
    "A-ELEV-FILL":      (0.00, "#000000", "solid", 9),
    "A-ANNO-DIMS":      (0.18, "#000000", "solid", 1),
    "A-ANNO-TEXT":      (0.25, "#000000", "solid", 7),
    "A-ANNO-SYMB":      (0.25, "#000000", "solid", 7),
    "A-ANNO-LEVL":      (0.18, "#000000", "dashdot", 1),
    "A-BLDG-PROP":      (0.70, "#000000", "solid", 7),
    "A-BLDG-PROP-PATT": (0.00, "#bdbdbd", "solid", 252),
    "C-PROP":           (0.50, "#000000", "phantom", 1),
    "C-PROP-SBCK":      (0.25, "#c0392b", "dashed", 1),
    "C-ROAD":           (0.25, "#333333", "solid", 8),
    "C-ROAD-PATT":      (0.00, "#e2e2e2", "solid", 254),
    "C-ROAD-CNTR":      (0.13, "#555555", "center", 8),
    "C-BLDG-EXST":      (0.35, "#444444", "solid", 8),
    "C-BLDG-EXST-PATT": (0.00, "#efefef", "solid", 254),
    "C-TOPO-MAJR":      (0.25, "#8d6e63", "solid", 33),
    "C-TOPO-MINR":      (0.13, "#bcaaa4", "solid", 33),
    "C-WATR":           (0.18, "#4a90c2", "solid", 5),
    "L-PLNT":           (0.18, "#2e7d32", "solid", 3),
    "L-PLNT-PATT":      (0.00, "#e8f0e3", "solid", 3),
    "G-ANNO-TTLB":      (0.35, "#000000", "solid", 7),
    "G-ANNO-NRTH":      (0.35, "#000000", "solid", 7),
    "G-GRND":           (1.00, "#000000", "solid", 7),
    "G-GRND-PATT":      (0.00, "#d7ccc8", "solid", 34),
}

DASHES = {
    "solid": None,
    "dashed": (4, 2),
    "center": (10, 2, 2, 2),
    "phantom": (12, 2, 2, 2, 2, 2),
    "dashdot": (8, 2, 1, 2),
}


@dataclass
class Ent:
    kind: str
    layer: str
    data: dict


@dataclass
class Drawing:
    """Entities in model metres (x right, y up in the view's own 2D frame)."""

    ents: list[Ent] = field(default_factory=list)

    def poly(self, pts, layer, closed=True, fill: str | None = None, lw: float | None = None,
             alpha: float = 1.0, z: int = 1, edge: bool = True):
        self.ents.append(Ent("poly", layer, dict(pts=np.asarray(pts, float), closed=closed,
                                                 fill=fill, lw=lw, alpha=alpha, z=z, edge=edge)))

    def line(self, p0, p1, layer, lw=None, z=2):
        self.ents.append(Ent("poly", layer, dict(pts=np.asarray([p0, p1], float), closed=False,
                                                 fill=None, lw=lw, alpha=1, z=z, edge=True)))

    def shape(self, geom, layer, fill=None, edge=True, z=1, lw=None, alpha=1.0):
        """Add a shapely (Multi)Polygon / LineString."""
        if geom.is_empty:
            return
        gt = geom.geom_type
        if gt in ("MultiPolygon", "MultiLineString", "GeometryCollection"):
            for g in geom.geoms:
                self.shape(g, layer, fill, edge, z, lw, alpha)
        elif gt == "Polygon":
            self.ents.append(Ent("hatch", layer, dict(geom=geom, fill=fill, edge=edge, z=z, lw=lw,
                                                      alpha=alpha)))
        elif gt in ("LineString", "LinearRing"):
            self.poly(list(geom.coords), layer, closed=False, z=z, lw=lw)

    def arc(self, c, r, a0, a1, layer, z=2):
        self.ents.append(Ent("arc", layer, dict(c=c, r=r, a0=a0, a1=a1, z=z)))

    def circle(self, c, r, layer, fill=None, z=3):
        self.ents.append(Ent("circle", layer, dict(c=c, r=r, fill=fill, z=z)))

    def text(self, p, s, layer="A-ANNO-TEXT", h_mm=2.5, rot=0.0, ha="center", va="center",
             weight="normal", z=5, box=False):
        self.ents.append(Ent("text", layer, dict(p=p, s=s, h=h_mm, rot=rot, ha=ha, va=va,
                                                 weight=weight, z=z, box=box)))

    def dim(self, p0, p1, offset, label, layer="A-ANNO-DIMS", z=4):
        """Aligned dimension; offset (m, model) to the left of p0->p1."""
        self.ents.append(Ent("dim", layer, dict(p0=np.asarray(p0, float), p1=np.asarray(p1, float),
                                                off=offset, label=label, z=z)))

    def bounds(self):
        xs, ys = [], []
        for e in self.ents:
            if e.kind == "poly":
                xs += list(e.data["pts"][:, 0])
                ys += list(e.data["pts"][:, 1])
            elif e.kind == "hatch":
                b = e.data["geom"].bounds
                xs += [b[0], b[2]]
                ys += [b[1], b[3]]
            elif e.kind in ("arc", "circle"):
                c, r = e.data["c"], e.data["r"]
                xs += [c[0] - r, c[0] + r]
                ys += [c[1] - r, c[1] + r]
            elif e.kind == "dim":
                p0, p1 = e.data["p0"], e.data["p1"]
                v = p1 - p0
                ln = float(np.hypot(*v)) or 1.0
                nrm = np.array([-v[1], v[0]]) / ln
                for p in (p0, p1, p0 + nrm * e.data["off"] * 1.25, p1 + nrm * e.data["off"] * 1.25):
                    xs.append(p[0])
                    ys.append(p[1])
            elif e.kind == "text":
                xs.append(e.data["p"][0])
                ys.append(e.data["p"][1])
        return min(xs), min(ys), max(xs), max(ys)

    def transformed(self, fn) -> "Drawing":
        """Return a copy with all model points mapped by fn((x, y)) -> (x, y)."""
        from shapely.ops import transform

        out = Drawing()
        for e in self.ents:
            d = dict(e.data)
            if e.kind == "poly":
                d["pts"] = np.array([fn(p) for p in d["pts"]])
            elif e.kind == "hatch":
                d["geom"] = transform(lambda x, y, z=None: tuple(zip(*[fn(p) for p in zip(x, y)])), d["geom"])
            elif e.kind in ("arc", "circle"):
                d["c"] = fn(d["c"])
            elif e.kind == "text":
                d["p"] = fn(d["p"])
            elif e.kind == "dim":
                d["p0"], d["p1"] = np.array(fn(d["p0"])), np.array(fn(d["p1"]))
            out.ents.append(Ent(e.kind, e.layer, d))
        return out


# ---------------------------------------------------------------------------
# matplotlib renderer (true scale)
# ---------------------------------------------------------------------------

def _lw(layer, override=None):
    mm = override if override is not None else LAYERS.get(layer, (0.25,))[0]
    return mm * MM_TO_PT


def _ls(layer, lw_pt):
    pattern = DASHES.get(LAYERS.get(layer, (0, "", "solid"))[2])
    if not pattern:
        return "solid"
    return (0, tuple(max(1.0, p * 1.0) for p in pattern))


def render_mpl(dr: Drawing, ax, scale_ratio: float, dim_style: str = "arch"):
    """Draw entities on an axes whose data units are model metres at 1:scale_ratio."""
    from matplotlib.patches import Arc, Circle, PathPatch
    from matplotlib.patches import Polygon as MPolygon
    from matplotlib.path import Path as MPath

    # model metres per paper mm
    m_per_mm = scale_ratio / 1000.0
    for e in sorted(dr.ents, key=lambda e: e.data.get("z", 1)):
        lay = LAYERS.get(e.layer, (0.25, "#000000", "solid", 7))
        col = lay[1]
        d = e.data
        if e.kind == "poly":
            pts = d["pts"]
            if d["fill"] is not None and d["closed"]:
                # fill + outline as ONE patch so painter's order (hidden-line removal) holds
                lw = _lw(e.layer, d["lw"]) if d["edge"] else 0.25
                ax.add_patch(MPolygon(pts, closed=True, facecolor=d["fill"],
                                      edgecolor=col if d["edge"] else d["fill"], lw=lw,
                                      alpha=d["alpha"], zorder=d["z"], joinstyle="miter"))
                continue
            if d["edge"]:
                p = np.vstack([pts, pts[:1]]) if d["closed"] else pts
                lw = _lw(e.layer, d["lw"])
                ax.plot(p[:, 0], p[:, 1], color=col, lw=lw, linestyle=_ls(e.layer, lw),
                        solid_capstyle="butt", solid_joinstyle="miter", zorder=d["z"] + 0.1)
        elif e.kind == "hatch":
            g = d["geom"]
            verts, codes = [], []
            for ring in [g.exterior, *g.interiors]:
                c = list(ring.coords)
                verts += c
                codes += [MPath.MOVETO] + [MPath.LINETO] * (len(c) - 2) + [MPath.CLOSEPOLY]
            path = MPath(verts, codes)
            fill = d["fill"] if d["fill"] is not None else lay[1]
            lw = _lw(e.layer, d["lw"]) if d["edge"] else 0
            edgecol = "#000000" if e.layer.endswith("PATT") else col
            ax.add_patch(PathPatch(path, facecolor=fill if d["fill"] != "none" else "none",
                                   edgecolor=edgecol if d["edge"] else "none", lw=lw,
                                   alpha=d["alpha"], zorder=d["z"], joinstyle="miter"))
        elif e.kind == "arc":
            ax.add_patch(Arc(d["c"], 2 * d["r"], 2 * d["r"], theta1=d["a0"], theta2=d["a1"],
                             color=col, lw=_lw(e.layer), zorder=d["z"]))
        elif e.kind == "circle":
            ax.add_patch(Circle(d["c"], d["r"], facecolor=d["fill"] or "none", edgecolor=col,
                                lw=_lw(e.layer), zorder=d["z"]))
        elif e.kind == "text":
            kw = {}
            if d["box"]:
                kw["bbox"] = dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.4)
            ax.text(d["p"][0], d["p"][1], d["s"], fontsize=d["h"] * MM_TO_PT / 0.72,
                    rotation=d["rot"], ha=d["ha"], va=d["va"], weight=d["weight"],
                    family="DejaVu Sans", zorder=d["z"], rotation_mode="anchor", **kw)
        elif e.kind == "dim":
            _mpl_dim(ax, d, m_per_mm, col, _lw(e.layer))


def _mpl_dim(ax, d, m_per_mm, col, lw):
    p0, p1 = d["p0"], d["p1"]
    v = p1 - p0
    L = np.linalg.norm(v)
    if L < 1e-9:
        return
    u = v / L
    n = np.array([-u[1], u[0]])
    off = d["off"]
    q0, q1 = p0 + n * off, p1 + n * off
    gap, over = 1.5 * m_per_mm, 2.0 * m_per_mm
    sgn = 1 if off >= 0 else -1
    for p, q in ((p0, q0), (p1, q1)):
        a = p + n * sgn * gap
        b = q + n * sgn * over
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=lw, zorder=d["z"])
    e0, e1 = q0 - u * over, q1 + u * over
    ax.plot([e0[0], e1[0]], [e0[1], e1[1]], color=col, lw=lw, zorder=d["z"])
    # architectural tick (45 deg slash)
    t = (u + n) / math.sqrt(2) * 1.4 * m_per_mm
    for q in (q0, q1):
        ax.plot([q[0] - t[0], q[0] + t[0]], [q[1] - t[1], q[1] + t[1]], color=col, lw=lw * 2.5,
                zorder=d["z"])
    ang = math.degrees(math.atan2(u[1], u[0]))
    if ang > 90.01 or ang < -89.99:
        ang -= 180 * (1 if ang > 0 else -1)
    mid = (q0 + q1) / 2 + n * 1.0 * m_per_mm * (1 if ang == math.degrees(math.atan2(u[1], u[0])) else -1)
    ax.text(mid[0], mid[1], d["label"], fontsize=2.2 * MM_TO_PT / 0.72, rotation=ang,
            rotation_mode="anchor", ha="center", va="bottom", zorder=d["z"], family="DejaVu Sans")


# ---------------------------------------------------------------------------
# DXF writer
# ---------------------------------------------------------------------------

def write_dxf(dr: Drawing, path, units: str, scale_ratio: float, title: str = "",
              georef: dict | None = None):
    """Write entities to DXF in real-world units (inches if imperial, metres if metric)."""
    import ezdxf
    from ezdxf import units as dxu
    from ezdxf.enums import TextEntityAlignment

    imperial = units in ("ft", "in")
    k = 1 / 0.0254 if imperial else 1.0
    doc = ezdxf.new("R2018", setup=True)
    doc.units = dxu.IN if imperial else dxu.M
    doc.header["$MEASUREMENT"] = 0 if imperial else 1
    msp = doc.modelspace()
    for name, (lw, _c, lt, aci) in LAYERS.items():
        ltype = {"solid": "CONTINUOUS", "dashed": "DASHED", "center": "CENTER",
                 "phantom": "PHANTOM", "dashdot": "DASHDOT"}[lt]
        if ltype not in doc.linetypes:
            ltype = "CONTINUOUS"
        doc.layers.add(name, color=aci, linetype=ltype,
                       lineweight=int(round(lw * 100)) if lw > 0 else 0)
    ds = doc.dimstyles.new("ARCHKIT")
    ds.dxf.dimtsz = 1.4 / 1000 * scale_ratio * k   # tick size
    ds.dxf.dimtxt = 2.2 / 1000 * scale_ratio * k
    ds.dxf.dimexo = 1.5 / 1000 * scale_ratio * k
    ds.dxf.dimexe = 2.0 / 1000 * scale_ratio * k
    ds.dxf.dimgap = 0.8 / 1000 * scale_ratio * k
    ds.dxf.dimtad = 1
    ds.dxf.dimlunit = 4 if imperial else 2
    ds.dxf.dimdec = 0 if not imperial else 4
    ds.dxf.dimtih = 0
    ds.dxf.dimtoh = 0
    S = lambda p: (float(p[0]) * k, float(p[1]) * k)  # noqa: E731
    for e in dr.ents:
        d = e.data
        a = {"layer": e.layer}
        if e.kind == "poly":
            pts = [S(p) for p in d["pts"]]
            if d["fill"] is not None and d["closed"] and len(pts) >= 3:
                h = msp.add_hatch(color=256, dxfattribs={"layer": e.layer + "-PATT"
                                  if (e.layer + "-PATT") in LAYERS else e.layer})
                h.paths.add_polyline_path(pts, is_closed=True)
            if d["edge"]:
                msp.add_lwpolyline(pts, close=d["closed"], dxfattribs=a)
        elif e.kind == "hatch":
            g = d["geom"]
            if d["fill"] not in (None, "none") or e.layer.endswith("PATT"):
                h = msp.add_hatch(color=256, dxfattribs=a)
                h.paths.add_polyline_path([S(p) for p in g.exterior.coords], is_closed=True,
                                          flags=1)
                for ring in g.interiors:
                    h.paths.add_polyline_path([S(p) for p in ring.coords], is_closed=True, flags=0)
            if d["edge"]:
                edge_layer = e.layer[:-5] if e.layer.endswith("-PATT") else e.layer
                for ring in [g.exterior, *g.interiors]:
                    msp.add_lwpolyline([S(p) for p in ring.coords], close=True,
                                       dxfattribs={"layer": edge_layer})
        elif e.kind == "arc":
            msp.add_arc(S(d["c"]), d["r"] * k, d["a0"], d["a1"], dxfattribs=a)
        elif e.kind == "circle":
            msp.add_circle(S(d["c"]), d["r"] * k, dxfattribs=a)
        elif e.kind == "text":
            h = d["h"] / 1000 * scale_ratio * k
            for i, line in enumerate(str(d["s"]).split("\n")):
                p = S(d["p"])
                t = msp.add_text(line, height=h, rotation=d["rot"], dxfattribs=a)
                t.set_placement((p[0], p[1] - i * h * 1.5), align=TextEntityAlignment.MIDDLE_CENTER)
        elif e.kind == "dim":
            p0, p1 = S(d["p0"]), S(d["p1"])
            dim = msp.add_aligned_dim(p1=p0, p2=p1, distance=d["off"] * k, dimstyle="ARCHKIT",
                                      dxfattribs=a)
            dim.render()
    if georef:
        doc.header["$LATITUDE"] = georef["lat"]
        doc.header["$LONGITUDE"] = georef["lon"]
        doc.header["$NORTHDIRECTION"] = 0.0
    if title:
        doc.header["$PROJECTNAME"] = title[:255]
    doc.saveas(path)
    return path
