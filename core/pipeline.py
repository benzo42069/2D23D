from __future__ import annotations

from pathlib import Path

from core.blender_runner import build_blender_command, run_blender
from core.config import PipelineConfig
from core.depth import DepthEstimator, save_depth
from core.mesh import depth_to_mesh, export_mesh
from core.multi_view import refine_depth_multi
from core.preprocess import preprocess_image, save_preprocess_outputs
from core.texture import write_diffuse_texture, write_normal_map_from_depth
from core.utils import ensure_dir, logger, timed_step


class PipelineError(RuntimeError):
    pass


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise PipelineError(f"Input path does not exist: {input_path}")
    imgs = [
        p
        for p in sorted(input_path.iterdir())
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    ]
    if not imgs:
        raise PipelineError(f"No supported images in: {input_path}")
    return imgs


def run_pipeline(cfg: PipelineConfig) -> Path:
    images = collect_images(cfg.input_path)
    model_name = cfg.input_path.stem if cfg.input_path.is_file() else cfg.input_path.name
    model_dir = ensure_dir(cfg.output_dir / model_name)
    interm_dir = ensure_dir(model_dir / "intermediate")

    with timed_step("Preprocess + depth"):
        if cfg.mode == "multi" and len(images) > 1:
            image_rgb, mask, depth = refine_depth_multi(images, cfg.quality)
        else:
            pre = preprocess_image(images[0])
            save_preprocess_outputs(pre, interm_dir)
            estimator = DepthEstimator()
            d = estimator.estimate(pre.normalized_rgb, pre.mask, cfg.quality)
            logger.info("Depth source: %s", d.source)
            image_rgb, mask, depth = pre.image_rgb, pre.mask, d.depth

        save_depth(depth, interm_dir / "depth.png")

    with timed_step("Texture prep"):
        tex_path = write_diffuse_texture(image_rgb, mask, interm_dir / "texture_diffuse.png")
        normal_path = write_normal_map_from_depth(depth, mask, interm_dir / "texture_normal.png")

    with timed_step("Mesh generation"):
        decimate = cfg.decimate if cfg.decimate is not None else cfg.quality.decimate_ratio
        remesh_voxel = cfg.remesh_voxel if cfg.remesh_voxel is not None else cfg.quality.remesh_voxel
        mesh = depth_to_mesh(
            depth=depth,
            mask=mask,
            scale=cfg.scale,
            thickness=cfg.thickness,
            smooth_iterations=cfg.quality.smooth_iterations,
        )
        mesh_path = export_mesh(mesh, interm_dir / "mesh_raw.obj")
        logger.info("Raw mesh: vertices=%s faces=%s", len(mesh.vertices), len(mesh.faces))

    if cfg.no_blender:
        logger.info("Skipping Blender stage due to --no_blender")
        return mesh_path

    if not cfg.blender_path or not cfg.blender_path.exists():
        raise PipelineError("Blender executable not found. Provide --blender path or use --no_blender")

    with timed_step("Blender cleanup + export"):
        final_dir = ensure_dir(model_dir / "final")
        export_path = final_dir / f"{model_name}.{cfg.export_format}"
        script_path = Path(__file__).resolve().parent.parent / "blender" / "blender_pipeline.py"
        cmd = build_blender_command(
            blender_path=cfg.blender_path,
            script_path=script_path,
            mesh_path=mesh_path,
            texture_path=tex_path,
            normal_map_path=normal_path,
            output_path=export_path,
            export_format=cfg.export_format,
            remesh_voxel=remesh_voxel,
            decimate_ratio=decimate,
        )
        logger.info("Running Blender command")
        run_blender(cmd)

    return export_path
