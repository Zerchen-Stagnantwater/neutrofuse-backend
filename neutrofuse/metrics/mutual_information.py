"""
Mutual Information (MI) fusion metric.

MI(fused, source) measures how much information about a source image
is preserved in the fused output, via joint histogram entropy. The
standard fusion-quality convention is to report:

    MI_total = MI(fused, A) + MI(fused, B)

Higher is better -- it means the fused image retains information from
both sources rather than discarding one in favor of the other.

This is computed on grayscale images. If color images are passed in,
convert to grayscale before calling (see metrics.common.to_grayscale).
"""
from __future__ import annotations

import numpy as np


def _joint_histogram(img_x: np.ndarray, img_y: np.ndarray, bins: int = 256) -> np.ndarray:
    """2D joint histogram of two same-shape uint8 images, normalized to a
    joint probability distribution."""
    hist_2d, _, _ = np.histogram2d(
        img_x.ravel(), img_y.ravel(), bins=bins, range=[[0, 255], [0, 255]]
    )
    total = hist_2d.sum()
    return hist_2d / total if total > 0 else hist_2d


def mutual_information(img_x: np.ndarray, img_y: np.ndarray, bins: int = 256) -> float:
    """
    MI(X; Y) = sum_{x,y} p(x,y) * log( p(x,y) / (p(x)*p(y)) )

    Computed via joint and marginal histograms. img_x, img_y must be
    grayscale, same shape, dtype uint8 (or castable to it).
    """
    if img_x.shape != img_y.shape:
        raise ValueError(f"Images must match shape: {img_x.shape} vs {img_y.shape}")

    p_xy = _joint_histogram(img_x, img_y, bins)
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    # Avoid log(0): only sum over nonzero joint entries.
    nonzero = p_xy > 0
    px_grid = p_x[:, np.newaxis] * np.ones((1, bins))
    py_grid = np.ones((bins, 1)) * p_y[np.newaxis, :]
    denom = px_grid * py_grid

    mi = np.sum(p_xy[nonzero] * np.log(p_xy[nonzero] / denom[nonzero]))
    return float(mi)


def mi_total(fused: np.ndarray, source_a: np.ndarray, source_b: np.ndarray, bins: int = 256) -> float:
    """MI(fused, A) + MI(fused, B). Higher = more information from both
    sources preserved in the fusion result."""
    return mutual_information(fused, source_a, bins) + mutual_information(fused, source_b, bins)
