---
name: arch-design
description: Produce professional architectural deliverables coordinated with real-world sites — site plans, floor plans, elevations, sections (PDF + DXF), 3D models, photoreal Blender renders, photomontages and shadow studies — using the archkit toolkit in this repo. Use whenever the user asks for an architectural rendering, schematic, plan, elevation, section, site/massing study, zoning or setback check, or wants a design placed on a real address.
---

# Architectural design with archkit

Follow `CLAUDE.md` (setup, workflow, accuracy rules). Summary of the loop:

1. **Locate & gather**: `archkit locate "<address>"`; copy
   `examples/platt-park-infill/project.yaml` to `projects/<slug>/project.yaml`; set
   `site.location`, `site.timezone`, parcel (ask the user for a survey / county parcel
   GeoJSON — otherwise mark it APPROXIMATE), zoning district values and code profile.
   Ask the user for the zoning numbers if you cannot verify them; never invent them silently.
2. **Context**: set `site.lidar: {spacing_m: 0.5}` for US sites, then
   `archkit site projects/<slug>/project.yaml`; read the counts and sources.
   Stadium/arena work: add a `stadium:` block (copy `examples/kroger-field-renovation`).
3. **Design** in the spec: exterior wall rectangle(s), interior walls, openings
   (offset from segment start), rooms with `use`, stairs (risers × riser = floor-to-floor),
   roof, materials (see `archkit/materials.py`).
4. **Check**: `archkit check … --strict` until 0 FAIL. Report WARN/INFO honestly.
5. **Drawings**: `archkit drawings …`; open every `out/sheets/*.png` with Read and fix
   layout problems before delivering.
6. **Renders**: `--preview` to frame each view, then Cycles drafts (`--percent 25
   --samples 32`) and finals. Inspect every PNG for seams, floating objects, false trees,
   z-fighting stripes and exposure before delivering.
   Eye-level views: eye z ≈ 5.5 ft (1.7 m) above grade, 24–35 mm lens, verticals level.
7. **Photomontage** when the user supplies site photos: collect ≥ 6 well-spread control
   points (pixel ↔ site coords) into `camera_solve`; report RMS px. Context buildings
   become holdouts and terrain a shadow catcher automatically.
8. **Deliver**: PDF set, DXFs, renders, `checks.txt`, and a short note listing data
   sources, approximations, and what must be verified (survey, AHJ, heights).
