import numpy as np

from neutrofuse.core.features import (
    compute_patch_features,
    compute_feature_grid,
    feature_dim,
)


def test_feature_vector_length_matches_feature_dim():
    patch = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
    feat = compute_patch_features(patch, lbp_radius=1, lbp_points=8, n_orientation_bins=8)
    assert feat.shape[0] == feature_dim(lbp_points=8, n_orientation_bins=8)


def test_feature_vector_is_unit_norm():
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    feat = compute_patch_features(patch)
    norm = np.linalg.norm(feat)
    assert np.isclose(norm, 1.0) or np.isclose(norm, 0.0)


def test_flat_patch_produces_zero_gradient_component():
    """A flat patch has no gradient, so the orientation-histogram half is all zero."""
    flat = np.full((8, 8), 100, dtype=np.uint8)
    feat = compute_patch_features(flat, lbp_points=8, n_orientation_bins=8)
    grad_part = feat[-8:]
    np.testing.assert_allclose(grad_part, 0.0)


def test_small_patch_falls_back_gracefully():
    """Patch smaller than 2*radius should not crash; falls back to radius=1."""
    tiny = np.random.randint(0, 256, (2, 2), dtype=np.uint8)
    feat = compute_patch_features(tiny, lbp_radius=3, lbp_points=8)
    assert feat.shape[0] == feature_dim(lbp_points=8, n_orientation_bins=8)
    assert np.all(np.isfinite(feat))


def test_compute_feature_grid_shape():
    img = np.random.randint(0, 256, (16, 24), dtype=np.uint8)
    grid = compute_feature_grid(img, patch_size=8)
    assert grid.shape == (2, 3, feature_dim())


def test_compute_feature_grid_rejects_unaligned_image():
    img = np.zeros((10, 10), dtype=np.uint8)
    try:
        compute_feature_grid(img, patch_size=8)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_similar_patches_have_high_cosine_similarity():
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    patch_noisy = np.clip(
        patch.astype(np.int32) + rng.integers(-2, 3, patch.shape), 0, 255
    ).astype(np.uint8)

    feat_a = compute_patch_features(patch)
    feat_b = compute_patch_features(patch_noisy)

    cos_sim = np.dot(feat_a, feat_b) / (np.linalg.norm(feat_a) * np.linalg.norm(feat_b))
    assert cos_sim > 0.8
