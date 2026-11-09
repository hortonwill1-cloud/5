"""archkit: site-coordinated architectural drawings, models and renders.

Pipeline:  spec (YAML)  ->  site context (OSM / elevation / parcel GIS)
           ->  coordination + code checks  ->  drawings (PDF/SVG/DXF)
           ->  3D model (OBJ)  ->  render (Blender Cycles, or built-in preview)

All internal geometry is in metres in a project-local frame:
origin at the site reference point, +x = grid east, +y = grid north, +z = up,
via a transverse-Mercator projection centred on the site (sub-millimetre
distortion over a few km).
"""

__version__ = "0.1.0"
