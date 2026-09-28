"""
Trivial pixel-averaging baseline.

No decomposition, no weighting -- just (A + B) / 2. This is the floor
every other method (including ours) must clear; it exists purely as
a reference point in the ablation, not because it's a credible fusion
method on its own.
"""
from __future__ import annotations

import numpy as np


def fuse_naive_average(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    image_a, image_b : same shape, dtype uint8, grayscale or color.

    Returns
    -------
    uint8 array, same shape: (image_a + image_b) / 2, rounded.
    """
    if image_a.shape != image_b.shape:
        raise ValueError(f"Images must match shape: {image_a.shape} vs {image_b.shape}")

    fused = (image_a.astype(np.float64) + image_b.astype(np.float64)) / 2.0
    return np.clip(np.round(fused), 0, 255).astype(np.uint8)
