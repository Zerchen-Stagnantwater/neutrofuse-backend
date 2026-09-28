"""
Composite comparison panels combining sources, fused output, and
diagnostic overlays into single figures -- the layout reviewers
expect in fusion papers, rather than scattered individual plots.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from neutrofuse.viz.heatmap import upsample_to_pixels


def _to_display(image: np.ndarray) -> np.ndarray:
    """BGR (OpenCV convention, used throughout this project's data
    loaders) -> RGB for correct matplotlib display."""
    if image.ndim == 3 and image.shape[2] == 3:
        return image[:, :, ::-1]
    return image


def render_comparison_panel(
    image_a: np.ndarray,
    image_b: np.ndarray,
    fused: np.ndarray,
    indeterminacy_grid: np.ndarray | None = None,
    patch_size: int | None = None,
    ground_truth: np.ndarray | None = None,
    pair_id: str = "",
) -> Figure:
    """
    Side-by-side panel: source A, source B, fused result, and
    optionally an indeterminacy heatmap and/or ground truth.

    Parameters
    ----------
    image_a, image_b, fused : uint8, color (H,W,3, BGR) or grayscale
    indeterminacy_grid : (gh, gw) float in [0,1], pass FusionResult.comp_a.I
        to include the heatmap panel. Requires patch_size to also be given.
    patch_size : required if indeterminacy_grid is given.
    ground_truth : if given, adds a fourth/fifth image panel for direct
        visual comparison against the all-in-focus reference.
    pair_id : used as the figure's suptitle, e.g. "lytro_01".

    Returns
    -------
    matplotlib Figure with one row, N columns (N depends on which
    optional panels are included).
    """
    if indeterminacy_grid is not None and patch_size is None:
        raise ValueError("patch_size is required when indeterminacy_grid is provided")

    panels: list[tuple[np.ndarray, str, str | None]] = [
        (image_a, "Source A", None),
        (image_b, "Source B", None),
        (fused, "Fused", None),
    ]

    if ground_truth is not None:
        panels.append((ground_truth, "Ground truth", None))

    if indeterminacy_grid is not None:
        pixel_I = upsample_to_pixels(indeterminacy_grid, patch_size)
        backdrop = fused.mean(axis=2) if fused.ndim == 3 else fused.astype(np.float64)
        pixel_I = pixel_I[: backdrop.shape[0], : backdrop.shape[1]]
        panels.append((backdrop, "Indeterminacy (I)", "heatmap"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (img, label, kind) in zip(axes, panels):
        if kind == "heatmap":
            ax.imshow(img, cmap="gray")
            pixel_I_overlay = upsample_to_pixels(indeterminacy_grid, patch_size)
            pixel_I_overlay = pixel_I_overlay[: img.shape[0], : img.shape[1]]
            # Auto-scale to the indeterminacy grid's actual range rather
            # than a fixed [0,1] -- see viz.heatmap.render_indeterminacy_heatmap
            # for why a fixed range washes out real-image structure.
            vmin, vmax = float(indeterminacy_grid.min()), float(indeterminacy_grid.max())
            if vmax - vmin < 1e-9:
                vmin, vmax = 0.0, 1.0
            ax.imshow(pixel_I_overlay, cmap="inferno", alpha=0.6, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(_to_display(img), cmap="gray" if img.ndim == 2 else None)
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    if pair_id:
        fig.suptitle(pair_id, fontsize=13, y=1.02)

    fig.tight_layout()
    return fig


def render_metrics_bar_comparison(
    method_names: list[str],
    metric_values: dict[str, list[float]],
    title: str = "Fusion quality comparison",
) -> Figure:
    """
    Grouped bar chart comparing multiple methods across multiple
    metrics -- e.g. ours vs naive-averaging vs guided-filter, on
    MI/SF/Qabf. This is the standard quantitative-comparison figure
    for the ablation study.

    Parameters
    ----------
    method_names : e.g. ["Ours", "Naive averaging", "Guided filter"]
    metric_values : dict mapping metric name -> list of values, one
        per method, same order as method_names. e.g.
        {"MI": [5.5, 4.8, 5.1], "SF": [149.7, 74.9, 120.3]}

    Returns
    -------
    matplotlib Figure with one subplot per metric.
    """
    n_metrics = len(metric_values)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))

    for ax, (metric_name, values) in zip(axes, metric_values.items()):
        if len(values) != len(method_names):
            raise ValueError(
                f"metric '{metric_name}' has {len(values)} values but "
                f"{len(method_names)} method_names were given"
            )
        bars = ax.bar(method_names, values, color=colors)
        ax.set_title(metric_name)
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3g}", ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig
