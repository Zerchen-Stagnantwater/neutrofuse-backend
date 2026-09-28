"""
Figure save/cleanup helper.

Rendering functions in heatmap.py, decision_map.py, and comparison.py
deliberately return Figure objects rather than saving directly, so
they stay unit-testable without touching disk. This module is the one
place that writes PNGs and closes figures -- important in batch runs
(e.g. experiments/ rendering one figure per pair across 20+ pairs),
since unclosed matplotlib figures accumulate in memory across calls.
"""
from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure


def save_figure(fig: Figure, path: Path, dpi: int = 150, close: bool = True) -> Path:
    """
    Save a figure to disk as PNG (or whatever extension path has;
    matplotlib infers format from the suffix).

    Parameters
    ----------
    close : if True (default), closes the figure after saving to free
        memory. Set False only if the caller still needs to display
        or further modify the figure afterward.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return path
