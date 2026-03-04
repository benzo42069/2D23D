from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.config import PRESETS, PipelineConfig
from core.pipeline import PipelineError, run_pipeline
from core.utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2D image(s) to plausible 3D mesh + Blender export pipeline")
    parser.add_argument("--input", required=True, help="Input image file or folder")
    parser.add_argument("--output", required=True, help="Output root folder")
    parser.add_argument("--blender", help="Path to blender.exe")
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="balanced")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--thickness", type=float, default=0.08)
    parser.add_argument("--decimate", type=float, default=None)
    parser.add_argument("--remesh_voxel", type=float, default=None)
    parser.add_argument("--export", dest="export_format", choices=["glb", "fbx", "obj"], default="glb")
    parser.add_argument("--no_blender", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = PipelineConfig(
        input_path=Path(args.input).resolve(),
        output_dir=Path(args.output).resolve(),
        blender_path=Path(args.blender).resolve() if args.blender else None,
        mode=args.mode,
        preset=args.preset,
        scale=args.scale,
        thickness=args.thickness,
        decimate=args.decimate,
        remesh_voxel=args.remesh_voxel,
        export_format=args.export_format,
        no_blender=args.no_blender,
    )

    try:
        result = run_pipeline(cfg)
    except PipelineError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # broad for CLI surface
        print(f"[FATAL] Unexpected error: {exc}", file=sys.stderr)
        return 3

    print(f"Done. Output: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
