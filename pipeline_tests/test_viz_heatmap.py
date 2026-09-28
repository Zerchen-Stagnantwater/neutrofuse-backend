import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from neutrofuse.viz.heatmap import upsample_to_pixels, render_indeterminacy_heatmap


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    yield
    plt.close("all")


def test_upsample_to_pixels_shape():
    grid = np.zeros((4, 5))
    result = upsample_to_pixels(grid, patch_size=8)
    assert result.shape == (32, 40)


def test_upsample_to_pixels_preserves_values_per_block():
    grid = np.array([[0.1, 0.9]])
    result = upsample_to_pixels(grid, patch_size=4)
    assert result.shape == (4, 8)
    np.testing.assert_allclose(result[:, :4], 0.1)
    np.testing.assert_allclose(result[:, 4:], 0.9)


def test_render_returns_figure():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    I_grid = rng.uniform(0, 1, (4, 4))

    fig = render_indeterminacy_heatmap(base, I_grid, patch_size=8)
    assert isinstance(fig, Figure)


def test_render_handles_color_base_image():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    I_grid = rng.uniform(0, 1, (4, 4))

    fig = render_indeterminacy_heatmap(base, I_grid, patch_size=8)
    assert isinstance(fig, Figure)


def test_render_crops_mismatched_grid_to_base_shape():
    """When base image dims aren't an exact multiple of patch_size
    (common after the pipeline's internal padding/unpadding), the
    upsampled grid will be larger than the base image -- must crop,
    not crash."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (30, 30), dtype=np.uint8)  # not a multiple of 8
    I_grid = rng.uniform(0, 1, (4, 4))  # upsamples to 32x32, base is 30x30

    fig = render_indeterminacy_heatmap(base, I_grid, patch_size=8)
    assert isinstance(fig, Figure)


def test_render_does_not_write_to_disk():
    """Rendering must be pure -- no filesystem side effects -- so it
    stays fast and testable. Saving is io.save_figure's job."""
    import os
    before = set(os.listdir("."))

    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    I_grid = rng.uniform(0, 1, (2, 2))
    render_indeterminacy_heatmap(base, I_grid, patch_size=8)

    after = set(os.listdir("."))
    assert before == after


def test_auto_scale_uses_actual_data_range():
    """Regression test for a real visual-clarity bug found rendering
    on actual Lytro data: I's theoretical range is [0,1], but after
    the log-compression fix in core.neutrosophic, real images have I
    concentrated in a much narrower empirical band (e.g. [0.2, 1.0]
    on lytro_01). With a fixed vmin=0, vmax=1, that narrow band gets
    squeezed into the top portion of the colormap and the overlay
    looks uniformly bright with no visible structure. auto_scale=True
    must set vmin/vmax to the grid's actual min/max."""
    base = np.full((16, 16), 128, dtype=np.uint8)
    I_grid = np.array([[0.6, 0.65], [0.7, 0.75]])  # narrow real-world-like band

    fig = render_indeterminacy_heatmap(base, I_grid, patch_size=8, auto_scale=True)
    overlay_im = fig.axes[0].images[1]  # [0]=grayscale backdrop, [1]=heatmap overlay
    assert np.isclose(overlay_im.norm.vmin, 0.6)
    assert np.isclose(overlay_im.norm.vmax, 0.75)


def test_auto_scale_false_forces_zero_one_range():
    base = np.full((16, 16), 128, dtype=np.uint8)
    I_grid = np.array([[0.6, 0.65], [0.7, 0.75]])

    fig = render_indeterminacy_heatmap(base, I_grid, patch_size=8, auto_scale=False)
    overlay_im = fig.axes[0].images[1]
    assert np.isclose(overlay_im.norm.vmin, 0.0)
    assert np.isclose(overlay_im.norm.vmax, 1.0)


def test_auto_scale_handles_degenerate_uniform_grid():
    """If I is exactly uniform (vmax - vmin == 0), auto_scale must
    fall back to [0, 1] rather than passing a zero-width norm range
    to matplotlib (which would raise or produce a blank colorbar)."""
    base = np.full((16, 16), 128, dtype=np.uint8)
    I_grid = np.full((2, 2), 0.5)

    fig = render_indeterminacy_heatmap(base, I_grid, patch_size=8, auto_scale=True)
    overlay_im = fig.axes[0].images[1]
    assert np.isclose(overlay_im.norm.vmin, 0.0)
    assert np.isclose(overlay_im.norm.vmax, 1.0)


def test_render_includes_colorbar():
    """A colorbar is required for the heatmap to be self-documenting
    about what color maps to what I value, especially now that the
    scale is data-dependent rather than a fixed [0,1]."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    I_grid = rng.uniform(0, 1, (2, 2))

    fig = render_indeterminacy_heatmap(base, I_grid, patch_size=8)
    # a colorbar adds a second Axes to the figure beyond the main plot
    assert len(fig.axes) == 2
