from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.config import QualityPreset
from core.depth import DepthEstimator
from core.preprocess import preprocess_image


def refine_depth_multi(image_paths: list[Path], quality: QualityPreset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(image_paths) == 1:
        result = preprocess_image(image_paths[0])
        estimator = DepthEstimator()
        depth = estimator.estimate(result.normalized_rgb, result.mask, quality).depth
        return result.image_rgb, result.mask, depth

    pre = [preprocess_image(p) for p in image_paths]
    base = pre[0]
    estimator = DepthEstimator()

    base_depth = estimator.estimate(base.normalized_rgb, base.mask, quality).depth
    depth_accum = base_depth.copy()
    weight = np.ones_like(base_depth, dtype=np.float32)

    orb = cv2.ORB_create(1200)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    base_gray = cv2.cvtColor(base.normalized_rgb, cv2.COLOR_RGB2GRAY)
    kp1, des1 = orb.detectAndCompute(base_gray, None)

    for r in pre[1:]:
        d = estimator.estimate(r.normalized_rgb, r.mask, quality).depth
        img_gray = cv2.cvtColor(r.normalized_rgb, cv2.COLOR_RGB2GRAY)
        kp2, des2 = orb.detectAndCompute(img_gray, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            continue

        matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)[:300]
        if len(matches) < 8:
            continue

        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
        if H is None:
            continue

        warped_depth = cv2.warpPerspective(d, H, (base_depth.shape[1], base_depth.shape[0]))
        warped_mask = cv2.warpPerspective(r.mask.astype(np.float32), H, (base_depth.shape[1], base_depth.shape[0]))
        valid = (warped_mask > 0.5).astype(np.float32)
        depth_accum += warped_depth * valid
        weight += valid

    fused = depth_accum / np.clip(weight, 1e-6, None)
    fused *= base.mask
    if base.mask.sum() > 0:
        vals = fused[base.mask > 0]
        fused = np.clip((fused - vals.min()) / max(vals.max() - vals.min(), 1e-6), 0.0, 1.0)
    return base.image_rgb, base.mask, fused.astype(np.float32)
