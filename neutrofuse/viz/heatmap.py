"""
Indeterminacy heatmap visualization.

This is the figure that carries the project's central qualitative
claim: the comparative-indeterminacy component I (see
core.neutrosophic) should concentrate at focus boundaries -- regions
where neither source is clearly sharper than the other -- rather than
being uniform or random. If it doesn't, that's a finding about the
method, not just a plotting concern, so this module exposes the raw
heatmap array as well as the rendered figure.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: never depends on a display being present
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def upsample_to_pixels(patch_grid: np.ndarray, patch_size: int) -> np.ndarray:
    """Nearest-neighbor upsample a (gh, gw) patch-level grid to pixel
    resolution (gh*patch_size, gw*patch_size), for overlaying on the
    original image. Shared with fusion.compose's internal upsampling
    logic conceptually, but kept separate here since viz consumers
    don't need compose's blending semantics."""
    return np.repeat(np.repeat(patch_grid, patch_size, axis=0), patch_size, axis=1)


def render_indeterminacy_heatmap(
    base_image: np.ndarray,
    indeterminacy_grid: np.ndarray,
    patch_size: int,
    alpha: float = 0.6,
    cmap: str = "inferno",
    title: str = "Indeterminacy (I)",
    auto_scale: bool = True,
) -> Figure:
    """
    Overlay the indeterminacy grid on top of the base image.

    Parameters
    ----------
    base_image : (H, W) or (H, W, 3) uint8, typically one of the
        sources or the fused result. Used as grayscale backdrop
        regardless of channel count, so the colormap reads clearly.
    indeterminacy_grid : (gh, gw) float array in [0, 1], e.g.
        FusionResult.comp_a.I from pipeline.run_fusion.
    patch_size : the patch size used to produce indeterminacy_grid,
        needed to upsample it back to pixel resolution.
    alpha : opacity of the heatmap overlay, 0=invisible, 1=opaque.
    cmap : matplotlib colormap name. "inferno" makes high-I regions
        (the boundaries we care about) visually pop as bright yellow.
    auto_scale : if True (default), set vmin/vmax to the indeterminacy
        grid's actual min/max rather than a fixed [0, 1]. I's
        theoretical range is [0, 1], but on real images (after the
        log-compression in core.neutrosophic) the floor empirically
        sits well above 0 -- e.g. ~0.2-0.3, not 0 -- since even the
        most confident patches still carry some residual ambiguity.
        Fixing vmin=0 in that regime compresses all the real
        variation into the top portion of the colormap and makes the
        overlay look uniformly bright rather than showing structure.
        Set False to force the literal [0, 1] range, e.g. for
        comparing multiple images on an identical absolute scale.

    Returns
    -------
    matplotlib Figure. Caller is responsible for saving/closing it
    (see viz.io.save_figure) -- this function never writes to disk,
    so it stays testable without touching the filesystem.
    """
    if base_image.ndim == 3:
        backdrop = base_image.mean(axis=2)
    else:
        backdrop = base_image.astype(np.float64)

    pixel_I = upsample_to_pixels(indeterminacy_grid, patch_size)

    expected_shape = backdrop.shape
    if pixel_I.shape != expected_shape:
        # Source image dims weren't an exact multiple of patch_size
        # (the pipeline pads internally); crop the upsampled grid to
        # match rather than erroring, since this is a display-only
        # mismatch, not a correctness issue in the underlying data.
        pixel_I = pixel_I[: expected_shape[0], : expected_shape[1]]

    if auto_scale:
        vmin, vmax = float(indeterminacy_grid.min()), float(indeterminacy_grid.max())
        if vmax - vmin < 1e-9:
            vmin, vmax = 0.0, 1.0  # degenerate: uniform I, fall back to full range
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(backdrop, cmap="gray")
    im = ax.imshow(pixel_I, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="I (relative to this image)")
    fig.tight_layout()

    return fig
