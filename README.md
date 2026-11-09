# archkit

Site-coordinated architectural drawings, 3D models and photoreal renders from one design spec.

```
project.yaml ──► real-world site context ──► coordination + code checks
                 (OSM, USGS 3DEP, county GIS)      (zoning, setbacks, IRC/IBC)
             ──► drawing set: site plan, floor plans, elevations, sections, shadow study
                 (true-scale PDF on ARCH-D/A1 sheets + DXF in real units, AIA layers)
             ──► 3D model (OBJ/MTL)  ──► Blender Cycles renders / photomontages
```

## Quick start

```bash
pip install -e '.[dev]'
archkit build examples/platt-park-infill/project.yaml --percent 50
```

Outputs land in `examples/platt-park-infill/out/`:

| file | what |
|---|---|
| `*-drawing-set.pdf`, `sheets/*.png` | C1.0 site plan · A1.x floor plans · A2.x elevations · A3.0 sections |
| `dxf/*.dxf` | CAD files in real inches/metres, NCS/AIA layers, true dimension entities |
| `checks.txt/json` | zoning, site coordination and building-code findings with references |
| `model/scene.obj` | building + terrain + context massing (open in Blender/Rhino/SketchUp/Revit) |
| `renders/*.png` | Cycles renders; `*.view.json` records sun position, camera, lat/lon |
| `shadow-study.pdf/png` | equinox/solstice × 9am/12pm/3pm, new vs existing shadow |

## What makes it accurate *and* photoreal

* **Measured context**: USGS 3DEP LiDAR → real building heights and roof forms, bare-earth
  terrain and individual trees, merged with OpenStreetMap footprints and streets.
* **Photographic lighting**: real HDRI skies chosen and rotated to match the true sun; the
  sky's own sun is replaced by an exact sun lamp, so shadows are correct to the minute.
* **Scanned materials** (Poly Haven, CC0) mapped at their published real-world size.
* **Stadium tools**: C-value seating-bowl solver, IBC §1029 egress, seat counts, NCAA field.

Examples: `examples/platt-park-infill` (Denver infill house) and
`examples/kroger-field-tool-demo` (stadium tools exercised on Kroger Field's real existing
conditions; the seating sections are test inputs, **not** a renovation proposal).

archkit is a reference and analysis tool: it measures existing conditions and checks,
draws and renders the design *you* specify. It does not author or replace your plans.

## Accuracy

* **Geometry**: site-local transverse-Mercator frame — distances agree with WGS84 geodesics
  to < 1 mm over 500 m (tested). Drawings are plotted at true architectural/engineering scale
  with graphic scale bars; DXF dimensions measure in real units.
* **Sun**: NOAA solar-position algorithm (~0.01°), DST-aware via IANA time zones. Renders use
  an exact sun lamp at that position (HDRI skies are rotated to match; `sky: physical` uses
  Blender's Nishita sky).
* **Camera matching**: `camera_solve` recovers a photo's camera from ≥ 6 surveyed points
  (normalised DLT + Gauss-Newton), reporting reprojection RMS — the basis for verified views.
* **Context data**: OpenStreetMap footprints are typically ±1–3 m and heights are often
  estimated (flagged `*`). Import a boundary/topographic survey or county GIS GeoJSON via
  `site.imports` for permit-grade work.

Code and zoning checks are **preliminary design aids**; the Authority Having Jurisdiction and a
licensed architect/engineer govern. Nothing here is stamped construction documentation.

## Commands

```
archkit locate "ADDRESS"
archkit site | check | drawings | model | shadows | build  PROJECT.yaml
archkit proof  PROJECT.yaml [--view NAME]       # ~15 s/view drafts + checks -> contact sheet (first!)
archkit render PROJECT.yaml [--view NAME] [--percent 25] [--samples 64] [--blend] [--force]
```

See `examples/platt-park-infill/project.yaml` for a fully annotated spec and `CLAUDE.md`
for the working rules used by Claude in this repo.
