import numpy as np

from experiments.baselines.naive_average import fuse_naive_average


def test_average_of_two_constants():
    a = np.full((8, 8), 100, dtype=np.uint8)
    b = np.full((8, 8), 200, dtype=np.uint8)
    fused = fuse_naive_average(a, b)
    np.testing.assert_array_equal(fused, np.full((8, 8), 150, dtype=np.uint8))


def test_identical_images_recover_same_image():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    fused = fuse_naive_average(a, a)
    np.testing.assert_array_equal(fused, a)


def test_handles_color_images():
    a = np.full((4, 4, 3), 50, dtype=np.uint8)
    b = np.full((4, 4, 3), 150, dtype=np.uint8)
    fused = fuse_naive_average(a, b)
    assert fused.shape == (4, 4, 3)
    np.testing.assert_array_equal(fused, np.full((4, 4, 3), 100, dtype=np.uint8))


def test_rounds_rather_than_truncates():
    a = np.array([[1]], dtype=np.uint8)
    b = np.array([[2]], dtype=np.uint8)
    fused = fuse_naive_average(a, b)
    # (1+2)/2 = 1.5 -> rounds to 2, not truncates to 1
    assert fused[0, 0] == 2


def test_shape_mismatch_raises():
    a = np.zeros((8, 8), dtype=np.uint8)
    b = np.zeros((8, 16), dtype=np.uint8)
    try:
        fuse_naive_average(a, b)
        assert False, "expected ValueError"
    except ValueError:
        pass
