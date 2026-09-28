import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from neutrofuse.viz.comparison import render_comparison_panel, render_metrics_bar_comparison


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    yield
    plt.close("all")


def _make_color(size=16, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


def test_render_comparison_panel_basic_three_panels():
    a, b, fused = _make_color(seed=0), _make_color(seed=1), _make_color(seed=2)
    fig = render_comparison_panel(a, b, fused)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3


def test_render_comparison_panel_with_ground_truth_adds_panel():
    a, b, fused, gt = (_make_color(seed=s) for s in range(4))
    fig = render_comparison_panel(a, b, fused, ground_truth=gt)
    assert len(fig.axes) == 4


def test_render_comparison_panel_with_heatmap_adds_panel():
    a, b, fused = _make_color(seed=0), _make_color(seed=1), _make_color(seed=2)
    I_grid = np.random.default_rng(0).uniform(0, 1, (2, 2))
    fig = render_comparison_panel(a, b, fused, indeterminacy_grid=I_grid, patch_size=8)
    assert len(fig.axes) == 4


def test_render_comparison_panel_with_both_gt_and_heatmap():
    a, b, fused, gt = (_make_color(seed=s) for s in range(4))
    I_grid = np.random.default_rng(0).uniform(0, 1, (2, 2))
    fig = render_comparison_panel(
        a, b, fused, ground_truth=gt, indeterminacy_grid=I_grid, patch_size=8
    )
    assert len(fig.axes) == 5


def test_heatmap_without_patch_size_raises():
    a, b, fused = _make_color(seed=0), _make_color(seed=1), _make_color(seed=2)
    I_grid = np.random.default_rng(0).uniform(0, 1, (2, 2))
    try:
        render_comparison_panel(a, b, fused, indeterminacy_grid=I_grid)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_render_comparison_panel_handles_grayscale():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    b = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    fused = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    fig = render_comparison_panel(a, b, fused)
    assert isinstance(fig, Figure)


def test_render_metrics_bar_comparison_basic():
    fig = render_metrics_bar_comparison(
        method_names=["Ours", "Naive avg"],
        metric_values={"MI": [5.5, 4.8], "SF": [149.7, 74.9]},
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_render_metrics_bar_comparison_mismatched_lengths_raises():
    try:
        render_metrics_bar_comparison(
            method_names=["Ours", "Naive avg", "Third method"],
            metric_values={"MI": [5.5, 4.8]},  # only 2 values for 3 methods
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_render_metrics_bar_comparison_single_metric():
    fig = render_metrics_bar_comparison(
        method_names=["Ours", "Naive avg", "Guided filter"],
        metric_values={"Qabf": [0.75, 0.60, 0.68]},
    )
    assert len(fig.axes) == 1
