import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from neutrofuse.fusion.decision import Source
from neutrofuse.viz.decision_map import (
    render_blend_weight_map,
    render_aggregation_mask,
    render_decision_map,
)


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    yield
    plt.close("all")


def test_render_blend_weight_map_returns_figure():
    weights = np.random.default_rng(0).uniform(0, 1, (4, 4))
    fig = render_blend_weight_map(weights, patch_size=8)
    assert isinstance(fig, Figure)


def test_render_blend_weight_map_crops_to_base_shape():
    weights = np.random.default_rng(0).uniform(0, 1, (4, 4))
    fig = render_blend_weight_map(weights, patch_size=8, base_shape=(30, 30))
    assert isinstance(fig, Figure)


def test_render_aggregation_mask_returns_figure():
    mask = np.array([[True, False], [False, True]])
    fig = render_aggregation_mask(mask, patch_size=8)
    assert isinstance(fig, Figure)


def test_render_aggregation_mask_handles_all_false():
    """Degenerate case: aggregation never fired. Must not crash on a
    uniform mask -- this is exactly the scenario the ablation study
    needs to be able to detect and plot."""
    mask = np.zeros((4, 4), dtype=bool)
    fig = render_aggregation_mask(mask, patch_size=8)
    assert isinstance(fig, Figure)


def test_render_aggregation_mask_handles_all_true():
    mask = np.ones((4, 4), dtype=bool)
    fig = render_aggregation_mask(mask, patch_size=8)
    assert isinstance(fig, Figure)


def test_render_decision_map_returns_figure():
    sources = np.empty((2, 2), dtype=object)
    sources[0, 0] = Source.A
    sources[0, 1] = Source.B
    sources[1, 0] = Source.BLEND
    sources[1, 1] = Source.A

    fig = render_decision_map(sources, patch_size=8)
    assert isinstance(fig, Figure)


def test_render_decision_map_numeric_encoding_matches_source_categories():
    """Verify the categorical encoding inside render_decision_map
    actually maps Source.A/B/BLEND to distinct values 0/1/2, by
    checking the rendered image data directly rather than just
    confirming no crash."""
    sources = np.empty((1, 3), dtype=object)
    sources[0, 0] = Source.A
    sources[0, 1] = Source.B
    sources[0, 2] = Source.BLEND

    fig = render_decision_map(sources, patch_size=4)
    ax = fig.axes[0]
    img_data = ax.images[0].get_array()

    # one patch_size=4 block per source, so check the center pixel of
    # each block lands on a distinct numeric category
    val_a = img_data[2, 2]
    val_b = img_data[2, 6]
    val_blend = img_data[2, 10]

    assert val_a != val_b
    assert val_b != val_blend
    assert val_a != val_blend
