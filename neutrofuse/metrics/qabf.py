"""
Q^abf fusion quality metric (Xydeas & Petrovic, 2000).

Measures how well edge information (strength + orientation) from each
source is preserved in the fused image, weighted by local edge
salience so that strong edges matter more than weak ones.

Algorithm
---------
1. Sobel gradients -> per-pixel edge strength g(i,j) and orientation
   alpha(i,j), for source A, source B, and the fused image F.
2. Relative strength/orientation preservation of A in F:
       G_AF(i,j) = min(g_F, g_A) / max(g_F, g_A)
       A_AF(i,j) = 1 - |alpha_F - alpha_A| / (pi/2)
3. Edge information preservation value, with sigmoid falloff
   constants (standard values from the original paper):
       Q_AF(i,j) = Gamma_g / (1 + exp(kappa_g * (G_AF - sigma_g)))
                 * Gamma_a / (1 + exp(kappa_a * (A_AF - sigma_a)))
4. Weight each pixel's Q_AF, Q_BF by source edge strength and combine:
       Q^abf = sum( Q_AF*w_A + Q_BF*w_B ) / sum( w_A + w_B )
   where w_A = g_A^L, w_B = g_B^L (L is a strength-weighting exponent).

Output is in [0, 1]. Higher = better edge preservation from both
sources into the fused result.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QABFConstants:
    """Standard constants from Xydeas & Petrovic (2000)."""
    gamma_g: float = 1.0
    kappa_g: float = -10.0
    sigma_g: float = 0.5
    gamma_a: float = 1.0
    kappa_a: float = -20.0
    sigma_a: float = 0.8
    strength_exponent: float = 1.0  # L


def _edge_strength_and_orientation(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel-based gradient magnitude and orientation (radians)."""
    img = image.astype(np.float64)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    strength = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.arctan2(gy, gx)
    return strength, orientation


def _edge_preservation(
    g_src: np.ndarray, a_src: np.ndarray,
    g_fused: np.ndarray, a_fused: np.ndarray,
    c: QABFConstants,
) -> np.ndarray:
    """Per-pixel Q_XF: how well source X's edge is preserved in the fused image."""
    eps = 1e-12

    g_ratio = np.minimum(g_fused, g_src) / (np.maximum(g_fused, g_src) + eps)

    angle_diff = np.abs(a_fused - a_src)
    # wrap to [0, pi/2] equivalent difference (orientation is mod pi for edges)
    angle_diff = np.minimum(angle_diff, np.pi - angle_diff)
    a_ratio = 1.0 - (angle_diff / (np.pi / 2))

    q_g = c.gamma_g / (1.0 + np.exp(c.kappa_g * (g_ratio - c.sigma_g)))
    q_a = c.gamma_a / (1.0 + np.exp(c.kappa_a * (a_ratio - c.sigma_a)))

    return q_g * q_a


def qabf(
    source_a: np.ndarray,
    source_b: np.ndarray,
    fused: np.ndarray,
    constants: QABFConstants | None = None,
) -> float:
    """
    Compute Q^abf for a fused image relative to its two sources.

    All three images must be grayscale, same shape. Returns a scalar
    in [0, 1] (in practice rarely reaching the extremes).
    """
    if not (source_a.shape == source_b.shape == fused.shape):
        raise ValueError(
            f"All images must match shape: A={source_a.shape}, "
            f"B={source_b.shape}, fused={fused.shape}"
        )

    c = constants or QABFConstants()

    g_a, a_a = _edge_strength_and_orientation(source_a)
    g_b, a_b = _edge_strength_and_orientation(source_b)
    g_f, a_f = _edge_strength_and_orientation(fused)

    q_af = _edge_preservation(g_a, a_a, g_f, a_f, c)
    q_bf = _edge_preservation(g_b, a_b, g_f, a_f, c)

    w_a = g_a ** c.strength_exponent
    w_b = g_b ** c.strength_exponent

    numerator = np.sum(q_af * w_a + q_bf * w_b)
    denominator = np.sum(w_a + w_b)

    if denominator < 1e-12:
        return 0.0

    return float(numerator / denominator)
