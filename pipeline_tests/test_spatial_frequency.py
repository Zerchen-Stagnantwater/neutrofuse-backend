import numpy as np

from neutrofuse.metrics.spatial_frequency import spatial_frequency


def test_flat_image_has_zero_spatial_frequency():
    flat = np.full((16, 16), 128, dtype=np.uint8)
    assert spatial_frequency(flat) == 0.0


def test_textured_image_has_positive_spatial_frequency():
    rng = np.random.default_rng(0)
    textured = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    assert spatial_frequency(textured) > 0.0


def test_sharp_image_has_higher_sf_than_blurred():
    import cv2
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    blurred = cv2.blur(sharp, (5, 5))

    assert spatial_frequency(sharp) > spatial_frequency(blurred)


def test_sf_matches_manual_rf_cf_computation():
    img = np.array([[1, 2, 4], [8, 16, 32], [1, 1, 1]], dtype=np.float64)

    row_diff = img[:, 1:] - img[:, :-1]
    col_diff = img[1:, :] - img[:-1, :]
    rf = np.sqrt(np.mean(row_diff ** 2))
    cf = np.sqrt(np.mean(col_diff ** 2))
    expected = np.sqrt(rf ** 2 + cf ** 2)

    assert np.isclose(spatial_frequency(img), expected)


def test_single_pixel_row_or_column_does_not_crash():
    """A 1xN or Nx1 image has no column-diff or row-diff respectively;
    should degrade gracefully rather than raising."""
    row_image = np.array([[1, 5, 2, 8]], dtype=np.uint8)
    col_image = np.array([[1], [5], [2], [8]], dtype=np.uint8)

    assert np.isfinite(spatial_frequency(row_image))
    assert np.isfinite(spatial_frequency(col_image))
