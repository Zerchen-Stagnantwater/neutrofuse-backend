"""
Patch extraction and reconstruction utilities.

Vertices in the hypergraph correspond to non-overlapping image patches.
This module handles the grid <-> image conversions and padding so the
image dimensions always divide evenly into patch_size x patch_size
blocks.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def pad_to_multiple(image: np.ndarray, patch_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pad an image (edge replication) so H and W are multiples of patch_size.

    Returns
    -------
    padded_image, original_shape (H, W)
    """
    h, w = image.shape[:2]
    pad_h = (-h) % patch_size
    pad_w = (-w) % patch_size

    if pad_h == 0 and pad_w == 0:
        return image, (h, w)

    if image.ndim == 2:
        pad_width = ((0, pad_h), (0, pad_w))
    else:
        pad_width = ((0, pad_h), (0, pad_w), (0, 0))

    padded = np.pad(image, pad_width, mode="edge")
    return padded, (h, w)


def unpad(image: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
    """Crop back to original_shape (H, W)."""
    h, w = original_shape
    return image[:h, :w, ...]


def extract_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Split an image into a grid of non-overlapping patches.

    Image H, W must be multiples of patch_size.

    Returns
    -------
    np.ndarray of shape (gh, gw, patch_size, patch_size[, C])
    patches[i, j] == image[i*ps:(i+1)*ps, j*ps:(j+1)*ps]
    """
    h, w = image.shape[:2]
    if h % patch_size or w % patch_size:
        raise ValueError(
            f"Image shape {(h, w)} is not a multiple of patch_size={patch_size}."
        )

    gh, gw = h // patch_size, w // patch_size

    if image.ndim == 2:
        reshaped = image.reshape(gh, patch_size, gw, patch_size)
        patches = reshaped.transpose(0, 2, 1, 3)
    else:
        c = image.shape[2]
        reshaped = image.reshape(gh, patch_size, gw, patch_size, c)
        patches = reshaped.transpose(0, 2, 1, 3, 4)

    return patches


def reconstruct_from_patches(patches: np.ndarray) -> np.ndarray:
    """
    Inverse of extract_patches.

    patches: (gh, gw, patch_size, patch_size[, C])
    Returns image of shape (gh*patch_size, gw*patch_size[, C])
    """
    if patches.ndim == 4:
        gh, gw, ph, pw = patches.shape
        image = patches.transpose(0, 2, 1, 3).reshape(gh * ph, gw * pw)
    elif patches.ndim == 5:
        gh, gw, ph, pw, c = patches.shape
        image = patches.transpose(0, 2, 1, 3, 4).reshape(gh * ph, gw * pw, c)
    else:
        raise ValueError(f"Unexpected patches ndim={patches.ndim}")

    return image


def grid_shape_for(image: np.ndarray, patch_size: int) -> Tuple[int, int]:
    """Patch-grid shape (gh, gw) for an already-aligned image."""
    h, w = image.shape[:2]
    return h // patch_size, w // patch_size
