"""
Visualizations of the per-patch fusion decision: which source won
each patch, and where hyperedge aggregation (the indeterminate-regime
consensus mechanism) actually fired.

These make the decision rule in fusion.decision inspectable on a real
image, which matters for the ablation study (experiments/) -- e.g.
confirming that the aggregation mask isn't degenerate (all-true or
all-false) on a given image before drawing conclusions from metrics
alone.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from neutrofuse.fusion.decision import Source
from neutrofuse.viz.heatmap import upsample_to_pixels


def render_blend_weight_map(
    blend_weights: np.ndarray,
    patch_size: int,
    base_shape: tuple[int, int] | None = None,
    title: str = "Source A weight",
) -> Figure:
    """
    Render the per-patch blend_weights grid (weight on source A, in
    [0, 1]) as a diverging heatmap: blue = source B, red = source A,
    white = even blend. This is the most direct visualization of what
    decide_grid actually decided, independent of the final pixel
    composition in fusion.compose.

    Parameters
    ----------
    blend_weights : (gh, gw) float array, FusionResult.blend_weights
    patch_size : patch size used to produce blend_weights
    base_shape : if given, crop the upsampled map to this (H, W)
        rather than the full padded grid -- pass FusionResult's
        original (unpadded) source shape for an exact-size figure.
    """
    pixel_map = upsample_to_pixels(blend_weights, patch_size)
    if base_shape is not None:
        pixel_map = pixel_map[: base_shape[0], : base_shape[1]]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(pixel_map, cmap="coolwarm_r", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="weight on A")
    fig.tight_layout()

    return fig


def render_aggregation_mask(
    used_aggregation_mask: np.ndarray,
    patch_size: int,
    base_shape: tuple[int, int] | None = None,
    title: str = "Hyperedge aggregation triggered",
) -> Figure:
    """
    Binary map of where the indeterminate regime (I >= theta) caused
    the decision rule to fall back to hyperedge-aggregated confidence
    rather than direct T comparison. Bright = aggregation fired.

    This should correlate spatially with the indeterminacy_heatmap's
    bright regions, by construction -- a useful internal-consistency
    check when validating the pipeline on a new image.
    """
    pixel_mask = upsample_to_pixels(used_aggregation_mask.astype(np.float64), patch_size)
    if base_shape is not None:
        pixel_mask = pixel_mask[: base_shape[0], : base_shape[1]]

    cmap = ListedColormap(["#1a1a2e", "#f9c80e"])  # dark navy / bright yellow

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(pixel_mask, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()

    return fig


def render_decision_map(
    sources: np.ndarray,
    patch_size: int,
    base_shape: tuple[int, int] | None = None,
    title: str = "Per-patch source decision",
) -> Figure:
    """
    Categorical map of the discrete Source decision (A / B / BLEND)
    per patch, as returned by fusion.decision.decide_grid's first
    output. Distinct from render_blend_weight_map: that one shows the
    continuous weight, this one shows the discrete category, which is
    useful for counting how often BLEND (the near-tie case) actually
    triggers on a real image.

    Parameters
    ----------
    sources : (gh, gw) dtype=object array of Source enum values
    """
    gh, gw = sources.shape
    numeric = np.zeros((gh, gw), dtype=np.int64)
    for i in range(gh):
        for j in range(gw):
            s = sources[i, j]
            if s == Source.A:
                numeric[i, j] = 0
            elif s == Source.B:
                numeric[i, j] = 1
            else:
                numeric[i, j] = 2

    pixel_map = upsample_to_pixels(numeric.astype(np.float64), patch_size)
    if base_shape is not None:
        pixel_map = pixel_map[: base_shape[0], : base_shape[1]]

    cmap = ListedColormap(["#e63946", "#1d3557", "#f1faee"])  # A=red, B=navy, BLEND=cream

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(pixel_map, cmap=cmap, vmin=0, vmax=2)
    ax.set_title(title)
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Source A", "Source B", "Blend"])
    fig.tight_layout()

    return fig
