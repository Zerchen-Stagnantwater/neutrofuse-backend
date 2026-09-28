"""
Sharpness / focus-measure operators.

These operate on a single 2D grayscale patch and return a scalar
"in-focus" score. Higher = sharper. This is the raw signal that the
neutrosophic Truth (T) component is derived from.
"""
from __future__ import annotations

import numpy as np
import cv2


def laplacian_variance(patch: np.ndarray) -> float:
    """Variance of the Laplacian. Standard, fast, robust focus measure."""
    lap = cv2.Laplacian(patch, cv2.CV_64F)
    return float(lap.var())


def tenengrad(patch: np.ndarray, ksize: int = 3) -> float:
    """Mean squared gradient magnitude (Sobel-based)."""
    gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=ksize)
    return float(np.mean(gx ** 2 + gy ** 2))


SHARPNESS_METRICS = {
    "laplacian_variance": laplacian_variance,
    "tenengrad": tenengrad,
}


def compute_sharpness_map(
    image: np.ndarray,
    patch_size: int,
    metric: str = "laplacian_variance",
) -> np.ndarray:
    """
    Compute a per-patch sharpness score over a grayscale image.

    Image dimensions must already be multiples of `patch_size`
    (use core.patches.pad_to_multiple beforehand).

    Returns
    -------
    np.ndarray of shape (H // patch_size, W // patch_size)
    """
    if metric not in SHARPNESS_METRICS:
        raise ValueError(
            f"Unknown sharpness metric '{metric}'. "
            f"Available: {list(SHARPNESS_METRICS)}"
        )
    fn = SHARPNESS_METRICS[metric]

    h, w = image.shape[:2]
    if h % patch_size or w % patch_size:
        raise ValueError(
            f"Image shape {(h, w)} is not a multiple of patch_size={patch_size}. "
            "Call patches.pad_to_multiple first."
        )

    gh, gw = h // patch_size, w // patch_size
    sharpness = np.empty((gh, gw), dtype=np.float64)

    for i in range(gh):
        for j in range(gw):
            patch = image[
                i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size,
            ]
            sharpness[i, j] = fn(patch)

    return sharpness
