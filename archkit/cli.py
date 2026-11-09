"""archkit command line.

  archkit site     PROJECT.yaml            fetch & cache real-world context
  archkit check    PROJECT.yaml            coordination + code checks
  archkit drawings PROJECT.yaml            PDF set, sheet PNGs, DXF files
  archkit model    PROJECT.yaml            OBJ/MTL of building + context
  archkit render   PROJECT.yaml [--view N] [--preview] [--percent 25]
  archkit shadows  PROJECT.yaml            solstice/equinox shadow study
  archkit build    PROJECT.yaml            everything
  archkit locate   "ADDRESS"               geocode an address -> lat/lon
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import spec as spec_mod
from .site import SiteContext, build_context


def _out(project, args) -> Path:
    p = Path(args.out) if getattr(args, "out", None) else project.root / "out"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_all(path: str, args, need_ctx: bool = True):
    project = spec_mod.load(path)
    if project.lat is None:
        if not project.address:
            raise SystemExit("site.location (lat/lon) or site.address is required")
        from .geo import geocode
        project.lat, project.lon, _ = geocode(project.address)
    out = _out(project, args)
    ctx = None
    if need_ctx:
        cache = out / "site-context.json"
        if cache.exists() and not getattr(args, "refresh", False):
            ctx = SiteContext.load(cache)
        elif not getattr(args, "offline", False):
            ctx = build_context(project.lat, project.lon, project.context_radius,
                                imports=project.imports, osm_json=project.osm_json)
            ctx.save(cache)
        else:
            from .geo import LocalFrame
            ctx = SiteContext(LocalFrame(project.lat, project.lon), project.context_radius)
        spec_mod.resolve_parcel(project, ctx.frame)
        parcels = [f for f in ctx.features if f.kind == "parcel" and f.source != "osm"]
        if project.parcel is None and parcels:
            from shapely.geometry import Point
            hit = [f for f in parcels if f.geom.contains(Point(*project.building.placement))]
            if hit:
                project.parcel = hit[0].geom
                project.parcel_source = hit[0].source
    return project, ctx, out


def cmd_locate(args):
    from .geo import geocode
    lat, lon, name = geocode(args.address)
    print(json.dumps({"lat": lat, "lon": lon, "display_name": name}, indent=1))


def cmd_site(args):
    args.refresh = True
    project, ctx, out = load_all(args.project, args)
    from collections import Counter
    print(f"site context -> {out / 'site-context.json'}")
    print(dict(Counter(f.kind for f in ctx.features)))
    for a in ctx.attribution:
        print("  source:", a)


def cmd_check(args):
    from .checks import run_checks, summarize
    project, ctx, out = load_all(args.project, args)
    f = run_checks(project, ctx)
    txt = summarize(f)
    (out / "checks.txt").write_text(txt)
    (out / "checks.json").write_text(json.dumps([x.__dict__ for x in f], indent=1))
    print(txt)
    return 1 if any(x.status == "FAIL" for x in f) and args.strict else 0


def cmd_drawings(args):
    from .checks import run_checks
    from .drawings import build_set
    project, ctx, out = load_all(args.project, args)
    produced = build_set(project, ctx, out, run_checks(project, ctx))
    print(json.dumps(produced, indent=1))


def cmd_model(args):
    from .model3d import export_obj, full_model
    project, ctx, out = load_all(args.project, args)
    m = full_model(project, ctx)
    p = export_obj(m, out / "model" / "scene.obj", project.building.materials)
    print(f"{len(m.faces)} faces -> {p}")


def cmd_render(args):
    from .model3d import full_model
    from .render import render_blender, render_preview
    project, ctx, out = load_all(args.project, args)
    m = full_model(project, ctx)
    views = [v for v in project.views if not args.view or v.name in args.view]
    if not views:
        raise SystemExit("no matching views in spec")
    for v in views:
        if args.preview:
            p = render_preview(project, ctx, m, v, out / "renders" / f"{v.name}-preview.png")
        else:
            p = render_blender(project, ctx, m, v, out / "renders", percent=args.percent,
                               samples=args.samples, save_blend=args.blend, exposure=args.exposure)
        print(f"{v.name} -> {p}")


def cmd_shadows(args):
    from .shadows import shadow_study
    project, ctx, out = load_all(args.project, args)
    p = shadow_study(project, ctx, out / "shadow-study.pdf", tz=args.utc_offset)
    print(p)


def cmd_build(args):
    for fn in (cmd_check, cmd_drawings, cmd_model, cmd_shadows):
        if fn is cmd_check:
            args.strict = False
        if fn is cmd_shadows:
            args.utc_offset = None
        fn(args)
        args.refresh = False
    args.view = None
    cmd_render(args)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="archkit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("locate")
    s.add_argument("address")
    s.set_defaults(fn=cmd_locate)
    for name, fn in (("site", cmd_site), ("check", cmd_check), ("drawings", cmd_drawings),
                     ("model", cmd_model), ("render", cmd_render), ("shadows", cmd_shadows),
                     ("build", cmd_build)):
        s = sub.add_parser(name)
        s.add_argument("project")
        s.add_argument("--out", help="output dir (default: <project dir>/out)")
        s.add_argument("--refresh", action="store_true", help="re-fetch site context")
        s.add_argument("--offline", action="store_true", help="no network; flat empty context")
        if name == "check":
            s.add_argument("--strict", action="store_true", help="exit 1 on any FAIL")
        if name in ("render", "build"):
            s.add_argument("--view", action="append")
            s.add_argument("--preview", action="store_true", help="fast built-in renderer")
            s.add_argument("--percent", type=int, default=100, help="resolution percent")
            s.add_argument("--samples", type=int)
            s.add_argument("--exposure", type=float)
            s.add_argument("--blend", action="store_true", help="also save .blend scene")
        if name == "shadows":
            s.add_argument("--utc-offset", type=float, dest="utc_offset")
        s.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    rc = args.fn(args)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
