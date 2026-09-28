"""
Structural Similarity Index (SSIM) against a ground-truth all-in-focus
image.

Unlike MI, SF, and Q^abf, SSIM requires an actual all-in-focus
reference image -- it cannot be computed from the two defocused
sources alone. Most multi-focus datasets (Lytro, MFI-WHU) do NOT ship
a ground truth, so this metric is optional: callers should check
availability before calling, or rely on metrics.evaluate.evaluate_fusion
which handles the absence gracefully.

This wraps skimage's validated implementation rather than
reimplementing the windowed-Gaussian SSIM math, since that
implementation is the field-standard reference.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def ssim(fused: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    SSIM(fused, ground_truth) in [-1, 1], higher is better (1.0 = identical).

    Both images must be grayscale, same shape. Pixel range is inferred
    as the dtype's natural range via skimage's data_range handling for
    uint8 input.
    """
    if fused.shape != ground_truth.shape:
        raise ValueError(
            f"fused and ground_truth must match shape: {fused.shape} vs {ground_truth.shape}"
        )

    return float(structural_similarity(fused, ground_truth, data_range=255))
