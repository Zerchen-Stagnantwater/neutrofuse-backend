import numpy as np

from neutrofuse.metrics.ssim_metric import ssim


def test_identical_images_give_ssim_one():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    assert np.isclose(ssim(img, img), 1.0)


def test_unrelated_images_give_lower_ssim_than_identical():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    other = rng.integers(0, 256, (32, 32), dtype=np.uint8)

    assert ssim(img, other) < ssim(img, img)


def test_ssim_in_valid_range():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    other = rng.integers(0, 256, (32, 32), dtype=np.uint8)

    score = ssim(img, other)
    assert -1.0 <= score <= 1.0


def test_shape_mismatch_raises():
    a = np.zeros((8, 8), dtype=np.uint8)
    b = np.zeros((8, 16), dtype=np.uint8)
    try:
        ssim(a, b)
        assert False, "expected ValueError"
    except ValueError:
        pass
