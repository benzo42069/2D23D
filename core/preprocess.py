from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class PreprocessResult:
    image_rgb: np.ndarray
    mask: np.ndarray
    normalized_rgb: np.ndarray


def _auto_mask(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Pick best foreground polarity by preferring central object occupancy.
    h, w = gray.shape
    center = otsu[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    if np.mean(center) < 127:
        otsu = 255 - otsu

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if num_labels <= 1:
        return (cleaned > 0).astype(np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    mask = (labels == largest).astype(np.uint8)
    return mask


def preprocess_image(image_path: Path, manual_mask: Path | None = None) -> PreprocessResult:
    image = Image.open(image_path).convert("RGB")
    image_rgb = np.asarray(image)

    if manual_mask:
        m = Image.open(manual_mask).convert("L")
        mask = (np.asarray(m) > 127).astype(np.uint8)
    else:
        mask = _auto_mask(image_rgb)

    foreground = image_rgb * mask[..., None]

    # Illumination normalization
    fg_float = foreground.astype(np.float32) / 255.0
    if mask.sum() > 0:
        mean = fg_float[mask > 0].mean(axis=0)
        gain = 0.5 / np.clip(mean, 1e-3, None)
        normalized = np.clip(fg_float * gain, 0.0, 1.0)
    else:
        normalized = fg_float

    return PreprocessResult(
        image_rgb=image_rgb,
        mask=mask,
        normalized_rgb=(normalized * 255).astype(np.uint8),
    )


def save_preprocess_outputs(result: PreprocessResult, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "image_rgb.png"
    mask_path = out_dir / "mask.png"
    norm_path = out_dir / "normalized.png"
    fg_path = out_dir / "foreground.png"

    Image.fromarray(result.image_rgb).save(raw_path)
    Image.fromarray((result.mask * 255).astype(np.uint8)).save(mask_path)
    Image.fromarray(result.normalized_rgb).save(norm_path)
    Image.fromarray(result.image_rgb * result.mask[..., None]).save(fg_path)
    return {
        "raw": raw_path,
        "mask": mask_path,
        "normalized": norm_path,
        "foreground": fg_path,
    }
