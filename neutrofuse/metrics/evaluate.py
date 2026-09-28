"""
Single entry point for scoring a fusion result against its sources
(and optionally a ground-truth all-in-focus image).

This is what experiments/ should call -- individual metric modules
are the expansion points if a new metric is needed later (e.g. VIFF,
Qcb), but day-to-day usage should go through evaluate_fusion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neutrofuse.metrics.common import to_grayscale
from neutrofuse.metrics.mutual_information import mi_total
from neutrofuse.metrics.spatial_frequency import spatial_frequency
from neutrofuse.metrics.qabf import qabf
from neutrofuse.metrics.ssim_metric import ssim


@dataclass(frozen=True)
class FusionMetrics:
    mi_total: float
    spatial_frequency: float
    qabf: float
    ssim: float | None  # None when no ground_truth was supplied


def evaluate_fusion(
    fused: np.ndarray,
    source_a: np.ndarray,
    source_b: np.ndarray,
    ground_truth: np.ndarray | None = None,
) -> FusionMetrics:
    """
    Compute the full metric suite for a fused image.

    Parameters
    ----------
    fused, source_a, source_b : same shape, grayscale or color uint8.
        Color images are converted to grayscale internally (all four
        metrics operate on grayscale).
    ground_truth : optional all-in-focus reference image, same shape.
        If omitted, ssim is reported as None.

    Returns
    -------
    FusionMetrics
    """
    fused_gray = to_grayscale(fused)
    a_gray = to_grayscale(source_a)
    b_gray = to_grayscale(source_b)

    mi = mi_total(fused_gray, a_gray, b_gray)
    sf = spatial_frequency(fused_gray)
    q = qabf(a_gray, b_gray, fused_gray)

    ssim_score = None
    if ground_truth is not None:
        gt_gray = to_grayscale(ground_truth)
        ssim_score = ssim(fused_gray, gt_gray)

    return FusionMetrics(
        mi_total=mi,
        spatial_frequency=sf,
        qabf=q,
        ssim=ssim_score,
    )
