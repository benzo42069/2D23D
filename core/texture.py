from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def write_diffuse_texture(image_rgb: np.ndarray, mask: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tex = image_rgb.copy()
    tex[mask == 0] = 0
    Image.fromarray(tex).save(path)
    return path


def write_normal_map_from_depth(depth: np.ndarray, mask: np.ndarray, path: Path, strength: float = 2.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dzdx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dzdy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)

    nx = -dzdx * strength
    ny = -dzdy * strength
    nz = np.ones_like(depth)

    n = np.stack([nx, ny, nz], axis=-1)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.clip(norm, 1e-6, None)
    normal_rgb = ((n * 0.5 + 0.5) * 255).astype(np.uint8)
    normal_rgb[mask == 0] = np.array([128, 128, 255], dtype=np.uint8)
    Image.fromarray(normal_rgb).save(path)
    return path
