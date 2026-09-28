import numpy as np
import cv2

from neutrofuse.core.sharpness import compute_sharpness_map, laplacian_variance, tenengrad


def test_laplacian_variance_zero_for_flat_patch():
    flat = np.full((8, 8), 128, dtype=np.uint8)
    assert laplacian_variance(flat) == 0.0


def test_laplacian_variance_positive_for_textured_patch():
    rng = np.random.default_rng(0)
    textured = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    assert laplacian_variance(textured) > 0.0


def test_sharp_patch_scores_higher_than_blurred():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    blurred = cv2.blur(sharp, (5, 5))

    assert laplacian_variance(sharp) > laplacian_variance(blurred)
    assert tenengrad(sharp) > tenengrad(blurred)


def test_compute_sharpness_map_shape():
    img = np.random.randint(0, 256, (32, 24), dtype=np.uint8)
    sharpness = compute_sharpness_map(img, patch_size=8)
    assert sharpness.shape == (4, 3)


def test_compute_sharpness_map_tenengrad_metric():
    img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
    sharpness = compute_sharpness_map(img, patch_size=8, metric="tenengrad")
    assert sharpness.shape == (2, 2)
    assert np.all(sharpness >= 0)


def test_compute_sharpness_map_rejects_unaligned_image():
    img = np.zeros((10, 10), dtype=np.uint8)
    try:
        compute_sharpness_map(img, patch_size=8)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_compute_sharpness_map_rejects_unknown_metric():
    img = np.zeros((8, 8), dtype=np.uint8)
    try:
        compute_sharpness_map(img, patch_size=8, metric="not_a_metric")
        assert False, "expected ValueError"
    except ValueError:
        pass
