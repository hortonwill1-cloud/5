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


def make_textured(name, p, t):
    """PBR material from scanned maps, box-projected at true real-world scale."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes.get("Principled BSDF")
    texco = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    sx, sy = t["size_m"]
    mapping.inputs["Scale"].default_value = (1.0 / sx, 1.0 / sx, 1.0 / sy)
    links.new(texco.outputs["Object"], mapping.inputs["Vector"])

    def img(path, noncolor=False):
        n = nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(path, check_existing=True)
        if noncolor:
            n.image.colorspace_settings.name = "Non-Color"
        n.projection = "BOX"
        n.projection_blend = 0.25
        links.new(mapping.outputs["Vector"], n.inputs["Vector"])
        return n

    col = img(t["diffuse"]).outputs["Color"]
    if t.get("recolor"):
        bw = nodes.new("ShaderNodeRGBToBW")
        links.new(col, bw.inputs[0])
        mul = nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = 1 / 0.45
        links.new(bw.outputs[0], mul.inputs[0])
        mix = nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        mix.inputs[6].default_value = tuple(t["recolor"]) + (1.0,)
        comb = nodes.new("ShaderNodeCombineColor")
        for i in range(3):
            links.new(mul.outputs[0], comb.inputs[i])
        links.new(comb.outputs[0], mix.inputs[7])
        col = mix.outputs[2]
    if t.get("mult"):
        mix = nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        links.new(col, mix.inputs[6])
        mix.inputs[7].default_value = tuple(t["mult"]) + (1.0,)
        col = mix.outputs[2]
    links.new(col, bsdf.inputs["Base Color"])
    set_input(bsdf, "Metallic", p.get("metal", 0.0))
    if t.get("rough"):
        links.new(img(t["rough"], True).outputs["Color"], bsdf.inputs["Roughness"])
    else:
        set_input(bsdf, "Roughness", p.get("rough", 0.6))
    normal_out = None
    if t.get("normal"):
        nm = nodes.new("ShaderNodeNormalMap")
        links.new(img(t["normal"], True).outputs["Color"], nm.inputs["Color"])
        normal_out = nm.outputs["Normal"]
    if t.get("disp"):
        bump = nodes.new("ShaderNodeBump")
        set_input(bump, "Strength", 0.35)
        set_input(bump, "Distance", 0.01)
        links.new(img(t["disp"], True).outputs["Color"], bump.inputs["Height"])
        if normal_out is not None:
            links.new(normal_out, bump.inputs["Normal"])
        normal_out = bump.outputs["Normal"]
    if normal_out is not None:
        links.new(normal_out, bsdf.inputs["Normal"])
    return mat


ORTHO_MATS = {"terrain", "grass", "asphalt", "sidewalk", "context_roof", "water"}


def drape_ortho(mat):
    """Replace base colour with the georeferenced aerial photo (soft-light blended with the
    material's own fine detail), falling back to the material where the mask is black."""
    o = job.get("ortho")
    if not o or mat.name.split(".")[0] not in set(o.get("materials", ORTHO_MATS)):
        return
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes.get("Principled BSDF")
    base_in = bsdf.inputs["Base Color"]
    detail = base_in.links[0].from_socket if base_in.links else None
    if detail is None:
        rgb = nodes.new("ShaderNodeRGB")
        rgb.outputs[0].default_value = tuple(base_in.default_value)
        detail = rgb.outputs[0]
    texco = nodes.new("ShaderNodeTexCoord")
    mp = nodes.new("ShaderNodeMapping")
    mp.inputs["Location"].default_value = (-o["x0"] / o["w"], -o["y0"] / o["h"], 0)
    mp.inputs["Scale"].default_value = (1 / o["w"], 1 / o["h"], 1)
    links.new(texco.outputs["Object"], mp.inputs["Vector"])

    def img(path, noncolor=False):
        n = nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(path, check_existing=True)
        if noncolor:
            n.image.colorspace_settings.name = "Non-Color"
        n.extension = "EXTEND"
        n.interpolation = "Cubic"
        links.new(mp.outputs["Vector"], n.inputs["Vector"])
        return n

    photo = img(o["path"]).outputs["Color"]
    # photo carries colour; the scanned material adds high-frequency detail (luminance ratio)
    bw = nodes.new("ShaderNodeRGBToBW")
    links.new(detail, bw.inputs[0])
    blur = nodes.new("ShaderNodeMath")
    blur.operation = "MULTIPLY"
    blur.inputs[1].default_value = 1.0 / max(o.get("detail_mean", 0.25), 0.05)
    links.new(bw.outputs[0], blur.inputs[0])
    lerp = nodes.new("ShaderNodeMapRange")       # keep detail subtle: 0.75..1.25
    lerp.inputs["From Min"].default_value = 0.0
    lerp.inputs["From Max"].default_value = 2.0
    lerp.inputs["To Min"].default_value = 0.75
    lerp.inputs["To Max"].default_value = 1.25
    links.new(blur.outputs[0], lerp.inputs["Value"])
    mul = nodes.new("ShaderNodeMix")
    mul.data_type = "RGBA"
    mul.blend_type = "MULTIPLY"
    mul.inputs["Factor"].default_value = 1.0
    links.new(photo, mul.inputs[6])
    comb = nodes.new("ShaderNodeCombineColor")
    for i in range(3):
        links.new(lerp.outputs["Result"], comb.inputs[i])
    links.new(comb.outputs[0], mul.inputs[7])
    out = mul.outputs[2]
    # only on up-facing surfaces: a vertical photo smears down walls and steep slopes
    geo = nodes.new("ShaderNodeNewGeometry")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(geo.outputs["Normal"], sep.inputs[0])
    up = nodes.new("ShaderNodeMapRange")
    up.inputs["From Min"].default_value = 0.45
    up.inputs["From Max"].default_value = 0.7
    links.new(sep.outputs["Z"], up.inputs["Value"])
    fac = up.outputs["Result"]
    if o.get("mask"):
        mm = nodes.new("ShaderNodeMath")
        mm.operation = "MULTIPLY"
        links.new(img(o["mask"], True).outputs["Color"], mm.inputs[0])
        links.new(fac, mm.inputs[1])
        fac = mm.outputs[0]
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    links.new(fac, mix.inputs["Factor"])
    links.new(detail, mix.inputs[6])
    links.new(out, mix.inputs[7])
    links.new(mix.outputs[2], base_in)


def add_bevel(mat, radius=0.012):
    """Rounded-edge shading: real edges catch highlights; CG's razor edges read as fake."""
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is None:
        return
    bev = nt.nodes.new("ShaderNodeBevel")
    bev.samples = 6
    bev.inputs["Radius"].default_value = radius
    nin = bsdf.inputs["Normal"]
    if nin.links:
        src = nin.links[0].from_node
        if src.type == "BUMP" and not src.inputs["Normal"].links:
            nt.links.new(bev.outputs["Normal"], src.inputs["Normal"])
    else:
        nt.links.new(bev.outputs["Normal"], nin)


ARCH_MATS = {"facade", "stucco_white", "brick_red", "brick_buff", "concrete", "board_formed",
             "wood_cedar", "fiber_cement", "metal_panel", "zinc", "limestone", "plaster",
             "frame_black", "frame_white", "door", "slab", "context", "seat_blue", "seat_white"}


def make_material(name, p):
    mat = _make_material(name, p)
    drape_ortho(mat)
    if name in ARCH_MATS:
        add_bevel(mat)
    return mat


def _make_material(name, p):
    tex = job.get("textures", {}).get(name)
    if tex:
        return make_textured(name, p, tex)
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
    vg = ob.vertex_groups.new(name="crown")
    crown_idx = {i for i, sl in enumerate(ob.material_slots) if sl.material and "crown" in sl.material.name}
    verts = {v for poly in ob.data.polygons if poly.material_index in crown_idx for v in poly.vertices}
    vg.add(list(verts), 1.0, "REPLACE")
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
    disp.vertex_group = "crown"
    tex2 = bpy.data.textures.new("leaf_noise", "CLOUDS")
    tex2.noise_scale = 0.45
    tex2.noise_depth = 2
    d2 = ob.modifiers.new("Leaves", "DISPLACE")
    d2.texture = tex2
    d2.texture_coords = "GLOBAL"
    d2.strength = 0.5
    d2.vertex_group = "crown"
    for poly in ob.data.polygons:
        poly.use_smooth = True


def remove_sun(path, probe, out_path):
    """Paint the sun out of an equirect HDRI (replaced by surrounding sky radiance)."""
    import os

    import numpy as np
    if os.path.exists(out_path):
        return bpy.data.images.load(out_path)
    img = bpy.data.images.load(path)
    W, H = img.size
    px = np.empty(W * H * 4, np.float32)
    img.pixels.foreach_get(px)
    a = px.reshape(H, W, 4)
    el, phi = math.radians(probe["el_deg"]), math.radians(probe["phi_deg"])
    rows = (np.arange(H) + 0.5) / H * math.pi - math.pi / 2
    cols = (0.5 - (np.arange(W) + 0.5) / W) * 2 * math.pi
    cosd = (np.sin(rows)[:, None] * math.sin(el)
            + np.cos(rows)[:, None] * math.cos(el) * np.cos(cols[None, :] - phi))
    ang = np.degrees(np.arccos(np.clip(cosd, -1, 1)))
    ring = (ang > 6) & (ang < 12)
    bg = np.median(a[ring][:, :3], 0)
    disc = ang < 6
    w = np.clip((6 - ang) / 2, 0, 1)[..., None]   # feathered edge
    a[..., :3] = np.where(disc[..., None], a[..., :3] * (1 - w) + bg * w, a[..., :3])
    img.pixels.foreach_set(a.ravel())
    img.filepath_raw = out_path
    img.file_format = "HDR"
    img.save()
    return img


def build_hdri_world(sky, sun_az, sun_el):
    world = bpy.data.worlds.new("HDRI")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = remove_sun(sky["path"], sky["probe"], sky["path"].replace(".hdr", "_nosun.hdr"))
    texco = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.vector_type = "POINT"
    # world direction with atan2(y,x) = phi_t must look up the image sun at phi_img
    phi_t = math.radians(90.0 - sun_az)
    mapping.inputs["Rotation"].default_value = (0, 0, math.radians(sky["probe"]["phi_deg"]) - phi_t)
    nt.links.new(texco.outputs["Generated"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0
    # exact sun
    if sky["sun_strength"] > 0 and sun_el > 0:
        ld = bpy.data.lights.new("Sun", "SUN")
        ld.energy = sky["sun_strength"]
        ld.angle = math.radians(0.53)
        sun = bpy.data.objects.new("Sun", ld)
        bpy.context.collection.objects.link(sun)
        a, e = math.radians(sun_az), math.radians(sun_el)
        v = Vector((math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)))
        sun.rotation_euler = v.to_track_quat("Z", "Y").to_euler()
    e_h = sky["sun_strength"] * max(math.sin(math.radians(sun_el)), 0) + sky["probe"]["e_sky_h"]
    return math.log2(math.pi / max(e_h, 1e-6))


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


def scatter_tree_models():
    """Replace proxy canopies with instanced scanned tree models sized to LiDAR trees."""
    import random

    trees, models = job.get("trees"), job.get("tree_models")
    if not trees or not models:
        return
    for ob in list(bpy.data.objects):
        if "trees" in ob.name.lower() and ob.type == "MESH":
            bpy.data.objects.remove(ob, do_unlink=True)
    lib = []
    for i, path in enumerate(models):
        coll = bpy.data.collections.new(f"tree_model_{i}")
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=path)
        new = [o for o in bpy.data.objects if o not in before]
        for o in new:
            for c in list(o.users_collection):
                c.objects.unlink(o)
            coll.objects.link(o)
        meshes = [o for o in new if o.type == "MESH"]
        pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
        h = max(p.z for p in pts) - min(p.z for p in pts)
        wdt = max(max(p.x for p in pts) - min(p.x for p in pts), max(p.y for p in pts) - min(p.y for p in pts))
        coll.instance_offset = (0, 0, min(p.z for p in pts))
        lib.append((coll, h, wdt))
    rnd = random.Random(11)
    parent = bpy.data.collections.new("trees_instanced")
    bpy.context.scene.collection.children.link(parent)
    for t in trees:
        coll, h, wdt = lib[rnd.randrange(len(lib))]
        e = bpy.data.objects.new("tree", None)
        e.instance_type = "COLLECTION"
        e.instance_collection = coll
        e.location = (t["x"], t["y"], t["z"])
        e.rotation_euler = (0, 0, rnd.uniform(0, 2 * math.pi))
        sz = t["h"] / h
        sxy = max(t["crown"] / wdt, sz * 0.6)
        e.scale = (sxy * rnd.uniform(0.9, 1.1), sxy * rnd.uniform(0.9, 1.1), sz)
        parent.objects.link(e)


def setup_compositor(scene):
    """Glare on specular highlights + aerial perspective (haze increasing with distance)."""
    vl = scene.view_layers[0]
    vl.use_pass_mist = True
    w = scene.world
    w.mist_settings.start = job.get("haze_start_m", 150.0)
    w.mist_settings.depth = job.get("haze_depth_m", 4000.0)
    w.mist_settings.falloff = "LINEAR"
    scene.use_nodes = True
    nt = scene.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    rl = nt.nodes.new("CompositorNodeRLayers")
    glare = nt.nodes.new("CompositorNodeGlare")
    try:
        glare.glare_type = "FOG_GLOW"
        glare.quality = "HIGH"
        glare.threshold = 1.2
        glare.mix = -0.92
        glare.size = 8
    except (AttributeError, TypeError):
        pass
    nt.links.new(rl.outputs["Image"], glare.inputs[0])
    haze = nt.nodes.new("CompositorNodeMixRGB")
    haze.blend_type = "MIX"
    haze.inputs[2].default_value = tuple(job.get("haze_rgb", (0.62, 0.70, 0.80))) + (1.0,)
    mul = nt.nodes.new("CompositorNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = job.get("haze_amount", 0.12)
    nt.links.new(rl.outputs["Mist"], mul.inputs[0])
    nt.links.new(mul.outputs[0], haze.inputs[0])
    nt.links.new(glare.outputs[0], haze.inputs[1])
    alpha = nt.nodes.new("CompositorNodeSetAlpha")
    nt.links.new(haze.outputs[0], alpha.inputs["Image"])
    nt.links.new(rl.outputs["Alpha"], alpha.inputs["Alpha"])
    comp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(alpha.outputs[0], comp.inputs["Image"])


def main():
    reset()
    scene = bpy.context.scene
    try:
        bpy.ops.wm.obj_import(filepath=job["obj"], forward_axis="Y", up_axis="Z")
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=job["obj"], axis_forward="Y", axis_up="Z")
    scatter_tree_models()
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
        if "trees" in name and not job.get("tree_models"):
            organic_trees(ob)
    auto_ev = None
    if job.get("sky"):
        auto_ev = build_hdri_world(job["sky"], job["sun_az"], job["sun_el"])
    else:
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
    if auto_ev is not None:
        vs.exposure = auto_ev + job.get("exposure_offset", 0.0)
        print("AUTO_EV", vs.exposure)
    else:
        vs.exposure = job.get("exposure", 0.0)
    if job.get("camera_fx", True):
        setup_compositor(scene)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if job.get("photomontage") else "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.render.filepath = job["out"]
    if job.get("save_blend"):
        bpy.ops.wm.save_as_mainfile(filepath=job["save_blend"])
    bpy.ops.render.render(write_still=True)


main()
