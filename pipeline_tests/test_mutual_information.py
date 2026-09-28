import numpy as np

from neutrofuse.metrics.mutual_information import mutual_information, mi_total


def test_identical_images_have_high_mi():
    """MI(X, X) should be substantially higher than MI(X, random noise)."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    noise = rng.integers(0, 256, (32, 32), dtype=np.uint8)

    mi_self = mutual_information(img, img)
    mi_noise = mutual_information(img, noise)

    assert mi_self > mi_noise


def test_mi_is_symmetric():
    rng = np.random.default_rng(0)
    img_x = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    img_y = rng.integers(0, 256, (16, 16), dtype=np.uint8)

    mi_xy = mutual_information(img_x, img_y)
    mi_yx = mutual_information(img_y, img_x)

    assert np.isclose(mi_xy, mi_yx)


def test_mi_is_nonnegative():
    rng = np.random.default_rng(0)
    img_x = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    img_y = rng.integers(0, 256, (16, 16), dtype=np.uint8)

    assert mutual_information(img_x, img_y) >= -1e-9  # allow tiny float error


def test_uniform_image_has_zero_mi_with_anything():
    """A constant image carries no information, so joint entropy collapses
    and MI should be ~0 regardless of the other image."""
    flat = np.full((16, 16), 100, dtype=np.uint8)
    rng = np.random.default_rng(0)
    other = rng.integers(0, 256, (16, 16), dtype=np.uint8)

    mi = mutual_information(flat, other)
    assert np.isclose(mi, 0.0, atol=1e-9)


def test_shape_mismatch_raises():
    img_x = np.zeros((8, 8), dtype=np.uint8)
    img_y = np.zeros((8, 16), dtype=np.uint8)
    try:
        mutual_information(img_x, img_y)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_mi_total_sums_both_sources():
    rng = np.random.default_rng(0)
    fused = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    a = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    b = rng.integers(0, 256, (16, 16), dtype=np.uint8)

    expected = mutual_information(fused, a) + mutual_information(fused, b)
    assert np.isclose(mi_total(fused, a, b), expected)


def test_fused_equal_to_a_gives_higher_mi_with_a_than_random_fused():
    """If fused == source_a exactly, MI(fused, A) should be at its
    self-information maximum, higher than some unrelated random image."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    b = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    unrelated = rng.integers(0, 256, (32, 32), dtype=np.uint8)

    mi_fused_is_a = mutual_information(a, a)
    mi_unrelated_with_a = mutual_information(unrelated, a)

    assert mi_fused_is_a > mi_unrelated_with_a
