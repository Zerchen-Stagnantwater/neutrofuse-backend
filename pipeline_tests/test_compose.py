import numpy as np

from neutrofuse.fusion.compose import compose_fused_image


def test_full_weight_a_recovers_image_a():
    image_a = np.full((8, 8), 50, dtype=np.uint8)
    image_b = np.full((8, 8), 200, dtype=np.uint8)
    blend_weights = np.ones((2, 2))  # patch_size=4 -> grid 2x2

    fused = compose_fused_image(image_a, image_b, blend_weights, patch_size=4)
    np.testing.assert_array_equal(fused, image_a)


def test_full_weight_b_recovers_image_b():
    image_a = np.full((8, 8), 50, dtype=np.uint8)
    image_b = np.full((8, 8), 200, dtype=np.uint8)
    blend_weights = np.zeros((2, 2))

    fused = compose_fused_image(image_a, image_b, blend_weights, patch_size=4)
    np.testing.assert_array_equal(fused, image_b)


def test_half_weight_averages():
    image_a = np.full((4, 4), 100, dtype=np.uint8)
    image_b = np.full((4, 4), 200, dtype=np.uint8)
    blend_weights = np.full((1, 1), 0.5)

    fused = compose_fused_image(image_a, image_b, blend_weights, patch_size=4)
    np.testing.assert_array_equal(fused, np.full((4, 4), 150, dtype=np.uint8))


def test_per_patch_weights_applied_independently_hard_mode():
    """With smooth=False, patch boundaries are hard cuts -- the
    original nearest-neighbor behavior, preserved for callers that
    want it (e.g. exact reproducibility of earlier results, or
    visualizations that should show the discrete decision as-is)."""
    image_a = np.full((4, 8), 0, dtype=np.uint8)
    image_b = np.full((4, 8), 100, dtype=np.uint8)
    blend_weights = np.array([[1.0, 0.0]])  # left patch=A, right patch=B

    fused = compose_fused_image(image_a, image_b, blend_weights, patch_size=4, smooth=False)
    np.testing.assert_array_equal(fused[:, :4], np.full((4, 4), 0, dtype=np.uint8))
    np.testing.assert_array_equal(fused[:, 4:], np.full((4, 4), 100, dtype=np.uint8))


def test_smooth_mode_blends_across_patch_boundary():
    """With smooth=True (the default), the same input should NOT
    produce a hard cut at the patch boundary -- pixels near the
    boundary should show intermediate values, which is the actual fix
    for the patch-blockiness artifact found on real fusion output."""
    image_a = np.full((4, 8), 0, dtype=np.uint8)
    image_b = np.full((4, 8), 100, dtype=np.uint8)
    blend_weights = np.array([[1.0, 0.0]])

    fused = compose_fused_image(image_a, image_b, blend_weights, patch_size=4, smooth=True)

    # pixels right at the boundary (columns 3 and 4) should be
    # somewhere between 0 and 100, not exactly 0 or exactly 100
    boundary_pixels = fused[:, 3:5].astype(np.float64)
    assert np.all(boundary_pixels > 0)
    assert np.all(boundary_pixels < 100)


def test_smooth_mode_preserves_patch_center_value_on_smooth_grid():
    """The core correctness property of smooth upsampling: at each
    patch's center pixel, the original per-patch weight should still
    be recovered (to within ordinary linear-interpolation rounding),
    not just 'somewhere in the blended range'. Verified on a smoothly-
    varying weight grid where exact recovery is meaningful to check
    (a checkerboard grid is an adversarial worst-case for any linear
    interpolator and won't recover exactly even with correct math)."""
    patch_size = 8
    weights = np.array([[0.1, 0.3, 0.5, 0.7],
                         [0.2, 0.4, 0.6, 0.8],
                         [0.15, 0.35, 0.55, 0.75],
                         [0.25, 0.45, 0.65, 0.85]])

    image_a = np.full((32, 32), 255, dtype=np.uint8)
    image_b = np.full((32, 32), 0, dtype=np.uint8)

    fused = compose_fused_image(image_a, image_b, weights, patch_size=patch_size, smooth=True)

    center_offset = patch_size // 2  # nearest discrete pixel to the true half-integer center
    for r in range(4):
        for c in range(4):
            py, px = r * patch_size + center_offset, c * patch_size + center_offset
            expected_weight = weights[r, c]
            # fused = weight*255 + (1-weight)*0 = weight*255
            expected_value = expected_weight * 255
            actual_value = float(fused[py, px])
            assert abs(actual_value - expected_value) < 8, (
                f"patch ({r},{c}): expected ~{expected_value:.1f}, got {actual_value}"
            )


def test_smooth_is_default():
    """smooth=True must be the default -- this is the fix, and it
    should apply automatically to every caller (including
    pipeline.run_fusion) without requiring an explicit opt-in."""
    image_a = np.full((4, 8), 0, dtype=np.uint8)
    image_b = np.full((4, 8), 100, dtype=np.uint8)
    blend_weights = np.array([[1.0, 0.0]])

    fused_default = compose_fused_image(image_a, image_b, blend_weights, patch_size=4)
    fused_explicit_smooth = compose_fused_image(
        image_a, image_b, blend_weights, patch_size=4, smooth=True
    )
    np.testing.assert_array_equal(fused_default, fused_explicit_smooth)


def test_color_image_blending():
    image_a = np.zeros((4, 4, 3), dtype=np.uint8)
    image_b = np.full((4, 4, 3), 255, dtype=np.uint8)
    blend_weights = np.full((1, 1), 0.5)

    fused = compose_fused_image(image_a, image_b, blend_weights, patch_size=4)
    assert fused.shape == (4, 4, 3)
    np.testing.assert_array_equal(fused, np.full((4, 4, 3), 127, dtype=np.uint8))


def test_original_shape_crops_padded_result():
    image_a = np.full((8, 8), 10, dtype=np.uint8)
    image_b = np.full((8, 8), 20, dtype=np.uint8)
    blend_weights = np.ones((2, 2))

    fused = compose_fused_image(
        image_a, image_b, blend_weights, patch_size=4, original_shape=(5, 6)
    )
    assert fused.shape == (5, 6)


def test_shape_mismatch_raises():
    image_a = np.zeros((4, 4), dtype=np.uint8)
    image_b = np.zeros((4, 8), dtype=np.uint8)
    blend_weights = np.ones((1, 1))

    try:
        compose_fused_image(image_a, image_b, blend_weights, patch_size=4)
        assert False, "expected ValueError"
    except ValueError:
        pass
