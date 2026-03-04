import argparse
import bpy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--texture", required=True)
    parser.add_argument("--normal", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=["glb", "fbx", "obj"], default="glb")
    parser.add_argument("--remesh_voxel", type=float, default=0.02)
    parser.add_argument("--decimate", type=float, default=0.35)
    argv = []
    if "--" in __import__("sys").argv:
        argv = __import__("sys").argv[__import__("sys").argv.index("--") + 1 :]
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_mesh(mesh_path: str):
    if mesh_path.lower().endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=mesh_path)
    elif mesh_path.lower().endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=mesh_path)
    else:
        raise RuntimeError(f"Unsupported mesh format: {mesh_path}")
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    return obj


def apply_modifiers(obj, remesh_voxel: float, decimate_ratio: float):
    remesh = obj.modifiers.new("AutoRemesh", "REMESH")
    remesh.mode = "VOXEL"
    remesh.voxel_size = max(remesh_voxel, 1e-4)
    bpy.ops.object.modifier_apply(modifier=remesh.name)

    smooth = obj.modifiers.new("Smooth", "SMOOTH")
    smooth.iterations = 5
    smooth.factor = 0.35
    bpy.ops.object.modifier_apply(modifier=smooth.name)

    dec = obj.modifiers.new("Decimate", "DECIMATE")
    dec.ratio = max(0.02, min(1.0, 1.0 - decimate_ratio))
    bpy.ops.object.modifier_apply(modifier=dec.name)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def uv_unwrap(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.1519, island_margin=0.03)
    bpy.ops.object.mode_set(mode="OBJECT")


def create_material(obj, texture_path: str, normal_path: str):
    mat = bpy.data.materials.new(name="Pic2MeshMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    tex = nodes.new(type="ShaderNodeTexImage")
    normal_tex = nodes.new(type="ShaderNodeTexImage")
    normal_map = nodes.new(type="ShaderNodeNormalMap")

    tex.image = bpy.data.images.load(texture_path)
    normal_tex.image = bpy.data.images.load(normal_path)
    normal_tex.image.colorspace_settings.name = "Non-Color"

    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(normal_tex.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def export_model(output_path: str, export_format: str):
    if export_format == "glb":
        bpy.ops.export_scene.gltf(filepath=output_path, export_format="GLB", use_selection=True)
    elif export_format == "fbx":
        bpy.ops.export_scene.fbx(filepath=output_path, use_selection=True, path_mode="COPY", embed_textures=True)
    elif export_format == "obj":
        bpy.ops.wm.obj_export(filepath=output_path, export_selected_objects=True)


def main():
    args = parse_args()
    clear_scene()
    obj = import_mesh(args.mesh)
    apply_modifiers(obj, args.remesh_voxel, args.decimate)
    uv_unwrap(obj)
    create_material(obj, args.texture, args.normal)

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    export_model(args.output, args.format)


if __name__ == "__main__":
    main()
