# archkit — working rules for Claude

This repo is a toolkit for **site-coordinated architectural deliverables**:
a single YAML spec → real-world site context → coordination/code checks →
true-scale drawing set (PDF + DXF) → 3D model (OBJ) → photoreal renders (Blender Cycles).

## Setup (fresh container)

```bash
pip install -e '.[dev]'
# Blender 4.5 LTS for photoreal renders (not needed for drawings/tests):
mkdir -p /opt/blender && cd /opt/blender && \
  curl -sSL https://download.blender.org/release/Blender4.5/blender-4.5.15-linux-x64.tar.xz | tar xJ
python -m pytest -q          # offline, ~10 s
```

Network hosts used (cloud environment `5-archkit` allows them; its setup script installs
the package and Blender): nominatim.openstreetmap.org (geocode), overpass-api.de and
api.openstreetmap.org (OSM context; the OSM API is the automatic fallback),
epqs.nationalmap.gov (USGS 3DEP 1 m elevation), api.opentopodata.org (SRTM, global),
s3-us-west-2.amazonaws.com/usgs-lidar-public + raw.githubusercontent.com (USGS 3DEP LiDAR
EPT index), api.polyhaven.com + dl.polyhaven.org (CC0 textures/HDRIs), download.blender.org.

## Workflow

1. `archkit locate "ADDRESS"` → lat/lon. Start a spec from `examples/platt-park-infill/project.yaml`.
2. `archkit site P.yaml` → fetch context into `out/site-context.json` (cached; `--refresh` to refetch).
3. Design: edit walls/openings/rooms/roof in the spec.
4. `archkit check P.yaml` → fix every FAIL; explain every WARN to the user.
5. `archkit drawings P.yaml` → `out/*-drawing-set.pdf`, `out/sheets/*.png`, `out/dxf/*.dxf`.
6. `archkit render P.yaml --preview` for framing, then `archkit render P.yaml` (Cycles).
   Use `--percent 25 --samples 32` for drafts; finals at 100% / 256–512 samples.
7. `archkit shadows P.yaml` → solstice/equinox shadow study.
8. **Always look at the PNG sheets and renders** (Read tool) before reporting —
   check for overlapping text, clipped dimensions, seams, wrong sun side.

## Measured context (LiDAR) — use it whenever the site is in the US

`site.lidar: {spacing_m: 0.5}` makes `archkit site` read USGS 3DEP point clouds (newest
survey covering the site): bare-earth terrain, every OSM footprint's measured height and
roof (robust flat/gable/hip/shed fit; complex roofs such as stadium bowls keep the measured
surface), and real trees. Ground-only surveys are handled by single- vs multi-return
separation. Check `checks.txt` for how many heights are measured vs estimated.

## Rendering quality

Views default to `sky: partly_cloudy` (Poly Haven HDRI matched to the computed sun
elevation, rotated to the true azimuth, its sun painted out and replaced by an exact sun
lamp; auto exposure) and `textures: true` (scanned PBR maps at real-world size, see
`archkit/assets.py`). `sky: physical` = Nishita sky. Draft: `--percent 25 --samples 32`;
final: `--percent 100 --samples 256+` (4K textures/HDRI are used at 100%).
`<view>.view.json` records sun, camera, HDRI and its elevation mismatch.

## Stadiums / assembly seating

`stadium:` block (see `examples/kroger-field-renovation/project.yaml`): field auto-posed from
the OSM pitch inside the stadium; `sections` solved with the C-value method
(`archkit/stadium.py`), IBC §1029 egress checks (aisle widths/capacity, seats between
aisles, risers, travel), seat counts, S-series sightline sheets, field + stand models.
New sections automatically clip (demolish) the existing LiDAR-measured structure;
`site.clip_context` removes other areas. Sides are relative to the field axis
(canonicalised toward north): north/south = end zones ahead/behind, west/east = left/right.

## Accuracy rules (non-negotiable)

- Coordinates: site-local transverse Mercator, metres internally; tested to < 1 mm vs
  WGS84 geodesic over 500 m. Never mix in unprojected lat/lon math.
- State data provenance. OSM footprints are ±1–3 m and most heights are ESTIMATED
  (flagged `*` on the site plan and in checks). Parcel lines from anything other than a
  boundary survey or county GIS must be labelled APPROXIMATE.
- Prefer authoritative imports (`site.imports`: survey / county GIS GeoJSON, `replace_osm: true`).
- Code checks are preliminary (IRC/IBC 2021 defaults, override in `code:`); always say
  the AHJ and a licensed professional govern. Do not present output as permit-ready.
- Sun position uses the NOAA algorithm; set `site.timezone` (IANA) so DST is right.
- Photomontages (verified views): use `camera_solve` with ≥ 6 control points of known
  site coordinates; report the reprojection RMS from `out/renders/<view>.view.json`.
  Never "eyeball" a camera for a view presented as accurate.

## Code map

| module | role |
|---|---|
| `archkit/geo.py` | geocoding, `LocalFrame` projection, cached HTTP |
| `archkit/site.py` | OSM/Overpass + OSM API, USGS/SRTM terrain, GeoJSON imports |
| `archkit/spec.py` | YAML → `Project`/`Building`/`Level`/`Wall`/`Opening` (metres) |
| `archkit/checks.py` | zoning envelope, setbacks, coverage, height, fire separation, IRC/IBC checks |
| `archkit/model3d.py` | walls with real openings, slabs, roofs, terrain, context → OBJ |
| `archkit/drawings/` | entity model → true-scale matplotlib sheets + DXF (AIA layers) |
| `archkit/render/` | Blender Cycles job (`blender_scene.py` runs inside Blender) + preview |
| `archkit/camera.py` | pinhole camera, DLT + Gauss-Newton camera solve from photo points |
| `archkit/sun.py` | NOAA solar position, DST-aware local time |
| `archkit/shadows.py` | 9-panel shadow study sheet |
| `archkit/lidar.py` | USGS 3DEP EPT reader, terrain, roof grids + fits, tree detection |
| `archkit/assets.py` | Poly Haven textures/HDRIs, HDRI sun probe & sky matching |
| `archkit/stadium.py` | sightlines, egress, seats, field markings, stands |

Conventions: building frame vs site frame (`Building.to_site`); opening `offset` is from
the wall segment start to the near jamb; closed exterior walls are listed as centreline
rectangles; segment order follows the path. Elevation/section views are projected from
the same 3D model as renders, so the set stays coordinated — keep it that way (never
draw elevations independently).
