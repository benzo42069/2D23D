from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QualityPreset:
    name: str
    depth_size: int
    bilateral_sigma: float
    smooth_iterations: int
    decimate_ratio: float
    remesh_voxel: float


PRESETS: dict[str, QualityPreset] = {
    "fast": QualityPreset(
        name="fast",
        depth_size=256,
        bilateral_sigma=1.0,
        smooth_iterations=1,
        decimate_ratio=0.55,
        remesh_voxel=0.04,
    ),
    "balanced": QualityPreset(
        name="balanced",
        depth_size=384,
        bilateral_sigma=1.4,
        smooth_iterations=2,
        decimate_ratio=0.35,
        remesh_voxel=0.025,
    ),
    "high": QualityPreset(
        name="high",
        depth_size=512,
        bilateral_sigma=1.8,
        smooth_iterations=3,
        decimate_ratio=0.2,
        remesh_voxel=0.015,
    ),
}


@dataclass
class PipelineConfig:
    input_path: Path
    output_dir: Path
    blender_path: Path | None
    mode: str
    preset: str
    scale: float
    thickness: float
    decimate: float | None
    remesh_voxel: float | None
    export_format: str
    no_blender: bool

    @property
    def quality(self) -> QualityPreset:
        return PRESETS[self.preset]
