"""
Per-patch feature vectors used for hyperedge membership.

Combines a Local Binary Pattern histogram (texture) with a gradient
orientation histogram (structure) into a single L2-normalized vector.
Two patches with similar feature vectors are candidates for sharing a
hyperedge in core.hypergraph.
"""
from __future__ import annotations

import numpy as np
import cv2
from skimage.feature import local_binary_pattern


def _lbp_histogram(patch: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    if patch.shape[0] <= radius * 2 or patch.shape[1] <= radius * 2:
        # Patch too small for the requested radius; fall back to radius=1.
        radius, n_points = 1, 8

    lbp = local_binary_pattern(patch.astype(np.uint8), n_points, radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float64)
    total = hist.sum()
    return hist / total if total > 0 else hist


def _gradient_orientation_histogram(patch: np.ndarray, n_bins: int = 8) -> np.ndarray:
    gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.arctan2(gy, gx)  # range (-pi, pi]

    hist, _ = np.histogram(
        orientation, bins=n_bins, range=(-np.pi, np.pi), weights=magnitude
    )
    total = hist.sum()
    return hist / total if total > 0 else hist


def feature_dim(lbp_points: int = 8, n_orientation_bins: int = 8) -> int:
    """Length of the feature vector produced by compute_patch_features."""
    return (lbp_points + 2) + n_orientation_bins


def compute_patch_features(
    patch: np.ndarray,
    lbp_radius: int = 1,
    lbp_points: int = 8,
    n_orientation_bins: int = 8,
) -> np.ndarray:
    """
    Returns a single feature vector for a grayscale patch:
    [LBP histogram | gradient orientation histogram], L2-normalized.
    """
    patch = patch.astype(np.float64)
    lbp_hist = _lbp_histogram(patch, lbp_radius, lbp_points)
    grad_hist = _gradient_orientation_histogram(patch, n_orientation_bins)

    feature = np.concatenate([lbp_hist, grad_hist])
    norm = np.linalg.norm(feature)
    return feature / norm if norm > 0 else feature


def compute_feature_grid(
    image: np.ndarray,
    patch_size: int,
    lbp_radius: int = 1,
    lbp_points: int = 8,
    n_orientation_bins: int = 8,
) -> np.ndarray:
    """
    Compute per-patch feature vectors over a grayscale image.

    Image H, W must be multiples of patch_size.

    Returns
    -------
    np.ndarray of shape (gh, gw, feature_dim)
    """
    h, w = image.shape[:2]
    if h % patch_size or w % patch_size:
        raise ValueError(
            f"Image shape {(h, w)} is not a multiple of patch_size={patch_size}."
        )

    gh, gw = h // patch_size, w // patch_size
    dim = feature_dim(lbp_points, n_orientation_bins)
    features = np.empty((gh, gw, dim), dtype=np.float64)

    for i in range(gh):
        for j in range(gw):
            patch = image[
                i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size,
            ]
            features[i, j] = compute_patch_features(
                patch, lbp_radius, lbp_points, n_orientation_bins
            )

    return features
