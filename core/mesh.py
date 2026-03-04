from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class MeshBuildResult:
    mesh: trimesh.Trimesh
    obj_path: Path


def depth_to_mesh(
    depth: np.ndarray,
    mask: np.ndarray,
    scale: float,
    thickness: float,
    smooth_iterations: int,
) -> trimesh.Trimesh:
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]
    x = (xs / max(w - 1, 1) - 0.5) * scale
    y = (0.5 - ys / max(h - 1, 1)) * scale
    z = depth * scale

    front_vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    back_vertices = np.column_stack([x.ravel(), y.ravel(), np.full(h * w, -thickness, dtype=np.float32)])
    vertices = np.vstack([front_vertices, back_vertices]).astype(np.float32)

    def vid(ix: int, iy: int, back: bool = False) -> int:
        base = iy * w + ix
        return base + (h * w if back else 0)

    faces = []
    valid = mask.astype(bool)

    for iy in range(h - 1):
        for ix in range(w - 1):
            cell = [valid[iy, ix], valid[iy, ix + 1], valid[iy + 1, ix], valid[iy + 1, ix + 1]]
            if all(cell):
                faces.append([vid(ix, iy), vid(ix + 1, iy), vid(ix, iy + 1)])
                faces.append([vid(ix + 1, iy), vid(ix + 1, iy + 1), vid(ix, iy + 1)])
                faces.append([vid(ix, iy, True), vid(ix, iy + 1, True), vid(ix + 1, iy, True)])
                faces.append([vid(ix + 1, iy, True), vid(ix, iy + 1, True), vid(ix + 1, iy + 1, True)])

    # Side walls around mask boundaries.
    for iy in range(h - 1):
        for ix in range(w - 1):
            if valid[iy, ix] and not valid[iy, ix + 1]:
                f0, f1 = vid(ix, iy), vid(ix, iy + 1)
                b0, b1 = vid(ix, iy, True), vid(ix, iy + 1, True)
                faces.extend([[f0, f1, b0], [f1, b1, b0]])
            if valid[iy, ix + 1] and not valid[iy, ix]:
                f0, f1 = vid(ix + 1, iy), vid(ix + 1, iy + 1)
                b0, b1 = vid(ix + 1, iy, True), vid(ix + 1, iy + 1, True)
                faces.extend([[f0, b0, f1], [f1, b0, b1]])
            if valid[iy, ix] and not valid[iy + 1, ix]:
                f0, f1 = vid(ix, iy), vid(ix + 1, iy)
                b0, b1 = vid(ix, iy, True), vid(ix + 1, iy, True)
                faces.extend([[f0, b0, f1], [f1, b0, b1]])
            if valid[iy + 1, ix] and not valid[iy, ix]:
                f0, f1 = vid(ix, iy + 1), vid(ix + 1, iy + 1)
                b0, b1 = vid(ix, iy + 1, True), vid(ix + 1, iy + 1, True)
                faces.extend([[f0, f1, b0], [f1, b1, b0]])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int64), process=False)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fill_holes()
    for _ in range(max(smooth_iterations, 0)):
        trimesh.smoothing.filter_laplacian(mesh, lamb=0.35, iterations=1)
    return mesh


def export_mesh(mesh: trimesh.Trimesh, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path
