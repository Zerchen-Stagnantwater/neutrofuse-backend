"""
Compose the final fused image from per-patch blend weights.

This operates on the original (possibly color, possibly padded)
images -- not on the grayscale sharpness/feature grids used upstream.
The blend_weights grid (one scalar per patch) is upsampled to pixel
resolution and used as a per-pixel alpha for a linear blend between
the two sources.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from neutrofuse.core.patches import unpad


def _upsample_weights(blend_weights: np.ndarray, patch_size: int) -> np.ndarray:
    """Nearest-neighbor upsample a (gh, gw) weight grid to (gh*ps, gw*ps).

    Every pixel within a patch gets the exact same weight, with a hard
    discontinuity at every patch boundary. This reproduces the
    underlying per-patch decision exactly -- useful for visualizations
    that should show the discrete decision as-is (see viz.decision_map)
    -- but produces visible blocky/stair-stepped artifacts in the final
    composed image wherever a patch boundary crosses a real edge in the
    source images. See _upsample_weights_smooth for the version used by
    compose_fused_image by default.
    """
    return np.repeat(np.repeat(blend_weights, patch_size, axis=0), patch_size, axis=1)


def _upsample_weights_smooth(blend_weights: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Smoothly upsample a (gh, gw) weight grid to (gh*ps, gw*ps) via
    bilinear interpolation between patch centers, eliminating the hard
    per-patch discontinuity that _upsample_weights produces.

    This is the actual fix for a real visible artifact: composing the
    fused image with nearest-neighbor-upsampled weights produces a
    blocky, stair-stepped pattern (in an exact patch_size-pixel grid)
    along any curved or diagonal edge in the source images that
    straddles a sharp/blurred boundary -- confirmed directly by
    zooming into a real fusion result. The fix is to smooth the
    DECISION'S SPATIAL EXTENT, not to blur the resulting image
    afterward; blurring after compositing would soften real image
    detail along with the artifact, while smoothing the weight grid
    before compositing only affects how the already-correct per-patch
    decision is distributed across pixels.

    Coordinate derivation: patch r occupies pixel rows
    [r*ps, r*ps + ps - 1], so its center sits at the continuous
    coordinate r*ps + (ps-1)/2. We want that exact output position to
    map back to grid index r (so the original decision is preserved
    exactly at each patch's center, not just approximately), which
    gives the inverse mapping grid_coord(y) = (y - (ps-1)/2) / ps.
    Verified directly against a non-adversarial smoothly-varying test
    grid: this recovers the exact original value at every patch center
    (to within ordinary linear-interpolation rounding at the true
    sub-pixel center position, since patch_size is even and the true
    center sits at a half-integer pixel coordinate with no single
    discrete pixel exactly on it).

    mode="nearest" extrapolation is used for any output pixel whose
    mapped coordinate falls outside [0, gh-1] / [0, gw-1] -- this only
    occurs in the half-patch margin at the image's outer edge, where
    clamping to the nearest real patch's value is the correct
    behavior (there is no neighboring patch beyond the image boundary
    to interpolate toward).
    """
    gh, gw = blend_weights.shape
    out_h, out_w = gh * patch_size, gw * patch_size

    out_y = np.arange(out_h)
    out_x = np.arange(out_w)
    grid_y = (out_y - (patch_size - 1) / 2.0) / patch_size
    grid_x = (out_x - (patch_size - 1) / 2.0) / patch_size
    gyy, gxx = np.meshgrid(grid_y, grid_x, indexing="ij")

    smoothed = map_coordinates(
        blend_weights.astype(np.float64), [gyy, gxx], order=1, mode="nearest"
    )
    return smoothed


def compose_fused_image(
    image_a: np.ndarray,
    image_b: np.ndarray,
    blend_weights: np.ndarray,
    patch_size: int,
    original_shape: tuple[int, int] | None = None,
    smooth: bool = True,
) -> np.ndarray:
    """
    Parameters
    ----------
    image_a, image_b : padded source images, same shape, dtype uint8.
        Can be grayscale (H, W) or color (H, W, C).
    blend_weights : (gh, gw) array of per-patch weight_a in [0, 1].
        gh, gw must equal image_a.shape[:2] // patch_size.
    patch_size : patch size used to produce blend_weights.
    original_shape : if given, crop the result back to this (H, W)
        before returning (undoes core.patches.pad_to_multiple).
    smooth : if True (default), upsample blend_weights with smooth
        center-preserving interpolation (_upsample_weights_smooth),
        eliminating patch-boundary blockiness in the composed image.
        If False, use hard nearest-neighbor upsampling
        (_upsample_weights) instead -- the original behavior, kept
        for exact reproducibility of earlier results and for callers
        that specifically want the discrete per-patch decision
        reflected pixel-for-pixel rather than smoothed.

    Returns
    -------
    Fused image, same dtype and channel layout as image_a/image_b.
    """
    if image_a.shape != image_b.shape:
        raise ValueError(f"Source images must match shape: {image_a.shape} vs {image_b.shape}")

    if smooth:
        pixel_weights = _upsample_weights_smooth(blend_weights, patch_size)
    else:
        pixel_weights = _upsample_weights(blend_weights, patch_size)

    if image_a.ndim == 3:
        pixel_weights = pixel_weights[:, :, np.newaxis]

    fused = (
        pixel_weights * image_a.astype(np.float64)
        + (1.0 - pixel_weights) * image_b.astype(np.float64)
    )
    fused = np.clip(fused, 0, 255).astype(image_a.dtype)

    if original_shape is not None:
        fused = unpad(fused, original_shape)

    return fused
