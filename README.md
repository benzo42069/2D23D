# Pic2Mesh (Windows, Python 3.11+, Blender 4.x)

Deterministic 2D-to-3D pipeline that converts one image (or multiple photos for refinement) into a plausible 3D mesh, then automates Blender cleanup and export.

## Final default choices

- **Preprocess**: classical automatic foreground extraction (Otsu + morphology + largest component), with optional manual mask injection in code.
- **Depth**: MiDaS (`MiDaS_small`) through PyTorch `torch.hub`, with CPU fallback and a deterministic heuristic fallback when ML runtime fails.
- **Mesh**: heightfield front surface + closed back surface + side walls, then hole fill and Laplacian smoothing.
- **Texture**: original foreground used as diffuse map, Sobel-derived normal map from depth.
- **Blender**: headless script applies voxel remesh, smooth, decimate, normal recalculation, Smart UV, material setup, and export (GLB default).
- **GUI**: Tkinter launcher (`app/gui.py`) for quick Windows usage.

## Project structure

```text
app/
  main.py
  gui.py
core/
  config.py
  utils.py
  preprocess.py
  depth.py
  multi_view.py
  mesh.py
  texture.py
  blender_runner.py
  pipeline.py
blender/
  blender_pipeline.py
tests/
  test_paths.py
  test_mesh.py
  test_blender_cmd.py
requirements.txt
README.md
```

## Setup (Windows)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> First MiDaS run downloads model weights into torch cache automatically.

## CLI usage

Single-image full pipeline:

```powershell
python -m app.main `
  --input .\samples\shoe.jpg `
  --output .\output `
  --blender "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" `
  --mode single `
  --preset balanced `
  --export glb
```

Multi-image refinement:

```powershell
python -m app.main --input .\samples\turntable_set --output .\output --blender "C:\...\blender.exe" --mode multi --preset high --export glb
```

Mesh-only dry run (skip Blender):

```powershell
python -m app.main --input .\samples\shoe.jpg --output .\output --mode single --no_blender
```

### Options

- `--preset fast|balanced|high`
- `--scale <float>` model XY extent scalar (default `1.0`)
- `--thickness <float>` rear shell thickness (default `0.08`)
- `--decimate <0..1>` cleanup aggressiveness override
- `--remesh_voxel <float>` Blender remesh voxel size override
- `--export glb|fbx|obj`

## GUI usage

```powershell
python -m app.gui
```

Pick inputs and click **Run**.

## Output layout

For input `shoe.jpg`:

```text
output/shoe/
  intermediate/
    image_rgb.png
    mask.png
    normalized.png
    foreground.png
    depth.png
    texture_diffuse.png
    texture_normal.png
    mesh_raw.obj
  final/
    shoe.glb
```

## Troubleshooting

- **Blender missing**: pass `--blender` path or use `--no_blender`.
- **Torch/MiDaS unavailable**: pipeline falls back to deterministic heuristic depth.
- **Poor silhouette**: provide cleaner source image, use high contrast background, or extend code with manual mask input.
- **Slow CPU inference**: use `--preset fast`.

## Limitations

- Single-image 3D is ambiguous: result is plausible relief-like geometry, not photogrammetric truth.
- Multi-view refinement is 2D homography-based and best for small viewpoint changes, not full SfM.
- Very thin or reflective objects may produce weak masks/depth.
