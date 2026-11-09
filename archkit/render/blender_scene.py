"""Runs INSIDE Blender:  blender -b -P blender_scene.py -- job.json

Builds a physically-based scene from the archkit OBJ export and renders it
with Cycles: Nishita physical sky with the true solar position, PBR materials
with procedural micro-detail, AgX tone mapping, OpenImageDenoise.  For a
photomontage it renders with a transparent film, context buildings as
holdouts (real buildings in the photo correctly occlude the proposal) and the
terrain as a shadow catcher, then composites over the photo.
"""

import json
import math
import sys

import bpy
from mathutils import Matrix, Vector

job = json.loads(open(sys.argv[sys.argv.index("--") + 1]).read())


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def set_input(node, names, value):
    for n in names if isinstance(names, (list, tuple)) else [names]:
        if n in node.inputs:
            node.inputs[n].default_value = value
            return True
    return False


def make_material(name, p):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes.get("Principled BSDF")
    base = tuple(p["base"]) + (1.0,)
    set_input(bsdf, "Base Color", base)
    set_input(bsdf, "Roughness", p.get("rough", 0.5))
    set_input(bsdf, "Metallic", p.get("metal", 0.0))
    if p.get("transmission"):
        set_input(bsdf, ["Transmission Weight", "Transmission"], p["transmission"])
        set_input(bsdf, "IOR", p.get("ior", 1.5))
    if p.get("sss"):
        set_input(bsdf, ["Subsurface Weight", "Subsurface"], p["sss"])
    texco = nodes.new("ShaderNodeTexCoord")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(texco.outputs["Object"], sep.inputs[0])
    # wall-aligned 2D coords: (x + y, z)
    add = nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    links.new(sep.outputs[0], add.inputs[0])
    links.new(sep.outputs[1], add.inputs[1])
    comb = nodes.new("ShaderNodeCombineXYZ")
    links.new(add.outputs[0], comb.inputs[0])
    links.new(sep.outputs[2], comb.inputs[1])
    height_src = None
    if p.get("brick"):
        br = nodes.new("ShaderNodeTexBrick")
        set_input(br, "Scale", 1.0)
        set_input(br, "Brick Width", 0.225)
        set_input(br, "Row Height", 0.075)
        set_input(br, "Mortar Size", 0.010)
        set_input(br, "Color1", base)
        c2 = tuple(min(1, v * 0.8) for v in p["base"]) + (1.0,)
        set_input(br, "Color2", c2)
        set_input(br, "Mortar", tuple(p.get("mortar", (0.6, 0.6, 0.6))) + (1.0,))
        br.offset = 0.5
        links.new(comb.outputs[0], br.inputs["Vector"])
        links.new(br.outputs["Color"], bsdf.inputs["Base Color"])
        height_src = br.outputs["Fac"]
    elif p.get("boards"):
        wave = nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.bands_direction = "Y" if p["boards"] < 0.3 else "X"
        set_input(wave, "Scale", 1.0 / p["boards"])
        set_input(wave, "Distortion", 0.0)
        wave.wave_profile = "SAW"
        links.new(comb.outputs[0], wave.inputs["Vector"])
        height_src = wave.outputs["Fac"]
    if p.get("noise"):
        noise = nodes.new("ShaderNodeTexNoise")
        set_input(noise, "Scale", 3.0)
        set_input(noise, "Detail", 8.0)
        links.new(texco.outputs["Object"], noise.inputs["Vector"])
        mix = nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        set_input(mix, "Factor", p["noise"] * 3)
        if not p.get("brick"):
            mix.inputs[6].default_value = base
        else:
            links.new(bsdf.inputs["Base Color"].links[0].from_socket, mix.inputs[6])
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (0.75, 0.75, 0.75, 1)
        ramp.color_ramp.elements[1].color = (1.15, 1.15, 1.15, 1)
        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], mix.inputs[7])
        links.new(mix.outputs[2], bsdf.inputs["Base Color"])
        if height_src is None:
            height_src = noise.outputs["Fac"]
    if height_src is not None and p.get("bump", 0.2) > 0:
        bump = nodes.new("ShaderNodeBump")
        set_input(bump, "Strength", min(1.0, p.get("bump", 0.2)))
        set_input(bump, "Distance", 0.005)
        links.new(height_src, bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def organic_trees(ob):
    """Turn low-poly crown spheres into irregular, leafy-looking canopies."""
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)  # weld so displacement stays closed
    bm.to_mesh(ob.data)
    bm.free()
    sub = ob.modifiers.new("Subdiv", "SUBSURF")
    sub.levels = sub.render_levels = 2
    tex = bpy.data.textures.new("crown_noise", "CLOUDS")
    tex.noise_scale = 1.6
    tex.noise_depth = 3
    disp = ob.modifiers.new("Displace", "DISPLACE")
    disp.texture = tex
    disp.texture_coords = "GLOBAL"
    disp.strength = 1.4
    disp.mid_level = 0.55
    tex2 = bpy.data.textures.new("leaf_noise", "CLOUDS")
    tex2.noise_scale = 0.45
    tex2.noise_depth = 2
    d2 = ob.modifiers.new("Leaves", "DISPLACE")
    d2.texture = tex2
    d2.texture_coords = "GLOBAL"
    d2.strength = 0.5
    for poly in ob.data.polygons:
        poly.use_smooth = True


def build_world(sun_az, sun_el):
    world = bpy.data.worlds.new("Sky")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_disc = True
    sky.sun_elevation = math.radians(max(sun_el, -2.0))
    # Nishita sun direction = (cos el sin r, -cos el cos r, sin el); azimuth is clockwise from N.
    sky.sun_rotation = math.pi - math.radians(sun_az)
    sky.altitude = job.get("altitude_m", 0.0)
    sky.air_density = job.get("air", 1.0)
    sky.dust_density = job.get("dust", 1.0)
    sky.ozone_density = 1.0
    sky.sun_intensity = 1.0
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = job.get("sky_strength", 1.0)
    return world


def setup_camera(cam_json):
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = cam_json["focal_mm"]
    cam_data.sensor_width = cam_json["sensor_mm"]
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.clip_start = 0.1
    cam_data.clip_end = 5000
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.matrix_world = Matrix(cam_json["matrix_world"])
    bpy.context.scene.camera = cam
    return cam


def main():
    reset()
    scene = bpy.context.scene
    try:
        bpy.ops.wm.obj_import(filepath=job["obj"], forward_axis="Y", up_axis="Z")
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=job["obj"], axis_forward="Y", axis_up="Z")
    mats = {name: make_material(name, p) for name, p in job["materials"].items()}
    for ob in list(scene.objects):
        if ob.type != "MESH":
            continue
        for slot in ob.material_slots:
            if slot.material is None:
                continue
            nm = slot.material.name.split(".")[0]
            if nm in mats:
                slot.material = mats[nm]
        # smooth shading on organic context, flat on architecture
        ob.data.shade_flat() if hasattr(ob.data, "shade_flat") else None
        name = ob.name.lower()
        if job.get("photomontage"):
            if name.startswith("context") and "roads" not in name and "green" not in name:
                ob.is_holdout = True
            if name.startswith("terrain") or "roads" in name or "green" in name:
                ob.is_shadow_catcher = True
        if "trees" in name:
            organic_trees(ob)
    build_world(job["sun_az"], job["sun_el"])
    setup_camera(job["camera"])

    scene.render.engine = "CYCLES"
    cy = scene.cycles
    cy.samples = job.get("samples", 512)
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.01
    cy.use_denoising = True
    try:
        cy.denoiser = "OPENIMAGEDENOISE"
    except TypeError:
        pass
    cy.max_bounces = 12
    cy.transmission_bounces = 12
    cy.glossy_bounces = 6
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    # GPU if present
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs is not None:
        cp = prefs.preferences
        for dev in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                cp.compute_device_type = dev
                cp.get_devices()
                if any(d.type == dev for d in cp.devices):
                    for d in cp.devices:
                        d.use = True
                    cy.device = "GPU"
                    break
            except TypeError:
                continue
    scene.render.resolution_x, scene.render.resolution_y = job["resolution"]
    scene.render.resolution_percentage = job.get("percent", 100)
    scene.render.film_transparent = bool(job.get("photomontage"))
    vs = scene.view_settings
    for vt in ("AgX", "Filmic"):
        try:
            vs.view_transform = vt
            break
        except TypeError:
            continue
    for look in ("AgX - Medium High Contrast", "Medium High Contrast"):
        try:
            vs.look = look
            break
        except TypeError:
            continue
    vs.exposure = job.get("exposure", 0.0)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if job.get("photomontage") else "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.render.filepath = job["out"]
    if job.get("save_blend"):
        bpy.ops.wm.save_as_mainfile(filepath=job["save_blend"])
    bpy.ops.render.render(write_still=True)


main()
