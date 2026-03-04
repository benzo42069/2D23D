from __future__ import annotations

import subprocess
from pathlib import Path


class BlenderInvocationError(RuntimeError):
    pass


def build_blender_command(
    blender_path: Path,
    script_path: Path,
    mesh_path: Path,
    texture_path: Path,
    normal_map_path: Path,
    output_path: Path,
    export_format: str,
    remesh_voxel: float,
    decimate_ratio: float,
) -> list[str]:
    return [
        str(blender_path),
        "-b",
        "-P",
        str(script_path),
        "--",
        "--mesh",
        str(mesh_path),
        "--texture",
        str(texture_path),
        "--normal",
        str(normal_map_path),
        "--output",
        str(output_path),
        "--format",
        export_format,
        "--remesh_voxel",
        str(remesh_voxel),
        "--decimate",
        str(decimate_ratio),
    ]


def run_blender(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise BlenderInvocationError(
            f"Blender failed ({proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
