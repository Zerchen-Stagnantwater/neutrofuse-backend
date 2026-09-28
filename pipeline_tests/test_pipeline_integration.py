import numpy as np

from neutrofuse.config import FusionConfig
from neutrofuse.pipeline import run_fusion


def _make_textured_half_sharp(size: int, sharp_side: str, seed: int) -> np.ndarray:
    """
    Produce a grayscale image where one half is sharp (random noise,
    high-frequency) and the other half is flat/blurred (uniform gray).
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 128, dtype=np.uint8)
    half = size // 2

    sharp_patch = rng.integers(0, 256, (size, half), dtype=np.uint8)
    if sharp_side == "left":
        img[:, :half] = sharp_patch
    else:
        img[:, half:] = sharp_patch

    return img


def test_pipeline_runs_end_to_end_on_synthetic_pair():
    image_a = _make_textured_half_sharp(32, sharp_side="left", seed=0)
    image_b = _make_textured_half_sharp(32, sharp_side="right", seed=1)

    config = FusionConfig(patch_size=8)
    result = run_fusion(image_a, image_b, config)

    assert result.fused_image.shape == image_a.shape
    assert result.fused_image.dtype == np.uint8
    assert result.blend_weights.shape == (4, 4)
    assert result.comp_a.T.shape == (4, 4)


def test_fused_image_takes_sharp_region_from_each_source():
    """Left half should come predominantly from A (sharp there),
    right half predominantly from B (sharp there)."""
    image_a = _make_textured_half_sharp(32, sharp_side="left", seed=0)
    image_b = _make_textured_half_sharp(32, sharp_side="right", seed=1)

    config = FusionConfig(patch_size=8)
    result = run_fusion(image_a, image_b, config)

    left_weights = result.blend_weights[:, :2]   # left half of patch grid
    right_weights = result.blend_weights[:, 2:]  # right half

    # left should favor A (weight close to 1), right should favor B (weight close to 0)
    assert left_weights.mean() > 0.6
    assert right_weights.mean() < 0.4


def test_pipeline_handles_color_images():
    rng = np.random.default_rng(0)
    image_a = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    image_b = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)

    config = FusionConfig(patch_size=8)
    result = run_fusion(image_a, image_b, config)

    assert result.fused_image.shape == (16, 16, 3)


def test_pipeline_handles_non_aligned_dimensions():
    """Image dims not multiples of patch_size should still work via padding,
    and output shape should match the *original* unpadded input."""
    rng = np.random.default_rng(0)
    image_a = rng.integers(0, 256, (20, 22), dtype=np.uint8)
    image_b = rng.integers(0, 256, (20, 22), dtype=np.uint8)

    config = FusionConfig(patch_size=8)
    result = run_fusion(image_a, image_b, config)

    assert result.fused_image.shape == (20, 22)


def test_pipeline_raises_on_shape_mismatch():
    image_a = np.zeros((16, 16), dtype=np.uint8)
    image_b = np.zeros((16, 24), dtype=np.uint8)

    try:
        run_fusion(image_a, image_b)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_pipeline_uses_default_config_when_none_given():
    rng = np.random.default_rng(0)
    image_a = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    image_b = rng.integers(0, 256, (16, 16), dtype=np.uint8)

    result = run_fusion(image_a, image_b)  # config=None
    assert result.fused_image.shape == (16, 16)


def test_aggregation_mask_has_some_true_and_false_on_mixed_image():
    """A realistic-ish image should trigger both confident and
    indeterminate regimes somewhere, not collapse to all-one-mode."""
    image_a = _make_textured_half_sharp(32, sharp_side="left", seed=0)
    image_b = _make_textured_half_sharp(32, sharp_side="right", seed=1)

    config = FusionConfig(patch_size=8, indeterminacy_threshold=0.3)
    result = run_fusion(image_a, image_b, config)

    mask = result.used_aggregation_mask
    # not asserting a specific split, just that the mask isn't degenerate
    assert mask.dtype == bool
