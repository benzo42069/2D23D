from pathlib import Path

from core.blender_runner import build_blender_command


def test_blender_command_format():
    cmd = build_blender_command(
        blender_path=Path(r"C:\\Program Files\\Blender\\blender.exe"),
        script_path=Path("blender/blender_pipeline.py"),
        mesh_path=Path("out/mesh.obj"),
        texture_path=Path("out/tex.png"),
        normal_map_path=Path("out/n.png"),
        output_path=Path("out/model.glb"),
        export_format="glb",
        remesh_voxel=0.02,
        decimate_ratio=0.3,
    )
    assert cmd[1:4] == ["-b", "-P", "blender/blender_pipeline.py"]
    assert "--mesh" in cmd
    assert cmd[-2:] == ["--decimate", "0.3"]
