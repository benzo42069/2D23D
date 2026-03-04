from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.config import QualityPreset


@dataclass
class DepthResult:
    depth: np.ndarray
    source: str


class DepthEstimator:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or Path(os.getenv("PIC2MESH_CACHE", Path.home() / ".pic2mesh_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def estimate(self, image_rgb: np.ndarray, mask: np.ndarray, quality: QualityPreset) -> DepthResult:
        try:
            depth = self._estimate_midas(image_rgb, quality)
            source = "midas"
        except Exception:
            depth = self._estimate_heuristic(image_rgb)
            source = "heuristic"

        depth = self._postprocess(depth, mask, quality)
        return DepthResult(depth=depth, source=source)

    def _estimate_midas(self, image_rgb: np.ndarray, quality: QualityPreset) -> np.ndarray:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_type = "MiDaS_small"
        midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        midas.to(device).eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = transforms.small_transform

        h, w, _ = image_rgb.shape
        scale = quality.depth_size / max(h, w)
        resized = cv2.resize(image_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        input_batch = transform(resized).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=resized.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
        depth = depth - depth.min()
        depth /= max(depth.max(), 1e-6)
        return depth.astype(np.float32)

    def _estimate_heuristic(self, image_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gray = cv2.GaussianBlur(gray, (0, 0), 1.2)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(gx * gx + gy * gy)
        inv_grad = 1.0 - np.clip(gradient * 1.5, 0.0, 1.0)
        center_bias = self._radial_center_bias(gray.shape)
        depth = 0.75 * gray + 0.25 * center_bias
        depth = 0.65 * depth + 0.35 * inv_grad
        depth = cv2.GaussianBlur(depth, (0, 0), 2.0)
        depth -= depth.min()
        depth /= max(depth.max(), 1e-6)
        return depth.astype(np.float32)

    @staticmethod
    def _radial_center_bias(shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        ys, xs = np.mgrid[0:h, 0:w]
        cx, cy = w / 2.0, h / 2.0
        dist = np.sqrt(((xs - cx) / max(w, 1)) ** 2 + ((ys - cy) / max(h, 1)) ** 2)
        bias = 1.0 - np.clip(dist / 0.6, 0.0, 1.0)
        return bias.astype(np.float32)

    def _postprocess(self, depth: np.ndarray, mask: np.ndarray, quality: QualityPreset) -> np.ndarray:
        d = depth.copy()
        d = cv2.bilateralFilter(d.astype(np.float32), d=7, sigmaColor=quality.bilateral_sigma, sigmaSpace=5.0)
        d *= mask.astype(np.float32)
        if mask.sum() > 0:
            fg = d[mask > 0]
            p1, p99 = np.percentile(fg, [1, 99])
            d = np.clip((d - p1) / max(p99 - p1, 1e-6), 0.0, 1.0)
            d *= mask
        return d.astype(np.float32)


def save_depth(depth: np.ndarray, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(vis).save(out_path)
    return out_path
