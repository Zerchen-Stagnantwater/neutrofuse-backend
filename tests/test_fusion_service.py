import io

import cv2
import numpy as np
import pytest
from PIL import Image

from app.fusion_service import (
    FusionError,
    _match_shapes,
    _resize_to_long_edge,
    fuse_images,
)


def _make(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


# --- _resize_to_long_edge ---

def test_resize_does_not_upscale():
    img = _make(100, 80)
    result = _resize_to_long_edge(img, max_long_edge=200)
    assert result.shape == (100, 80, 3)


def test_resize_caps_long_edge():
    img = _make(1200, 900)
    result = _resize_to_long_edge(img, max_long_edge=800)
    assert max(result.shape[:2]) == 800


def test_resize_preserves_aspect_ratio():
    img = _make(1200, 900)  # 4:3
    result = _resize_to_long_edge(img, max_long_edge=800)
    aspect = result.shape[1] / result.shape[0]
    assert abs(aspect - 900 / 1200) < 0.01


def test_resize_uses_inter_area_no_upscale():
    # INTER_AREA should never expand the image
    img = _make(400, 400)
    result = _resize_to_long_edge(img, max_long_edge=400)
    assert result.shape[:2] == (400, 400)


# --- _match_shapes ---

def test_identical_shapes_returned_unchanged():
    a = _make(100, 100)
    b = _make(100, 100)
    ra, rb = _match_shapes(a, b)
    np.testing.assert_array_equal(ra, a)
    assert rb.shape == b.shape


def test_slightly_different_shapes_get_matched():
    a = _make(100, 133)
    b = _make(101, 134)  # tiny difference, same 4:3 aspect ratio
    ra, rb = _match_shapes(a, b)
    assert ra.shape == rb.shape


def test_different_aspect_ratios_raise_fusion_error():
    a = _make(100, 200)   # 2:1
    b = _make(100, 100)   # 1:1
    with pytest.raises(FusionError, match="aspect"):
        _match_shapes(a, b)


def test_aspect_difference_exactly_at_threshold():
    # 5% threshold: <5% diff should pass, >5% should fail.
    a = _make(100, 200)  # aspect = 2.0
    b_pass = _make(100, 209)  # aspect 2.09, diff 4.5% — below threshold, should pass
    b_fail = _make(100, 212)  # aspect 2.12, diff 6.0% — above threshold, should fail

    ra, _ = _match_shapes(a, b_pass)
    assert ra.shape[:2] == a.shape[:2]

    with pytest.raises(FusionError):
        _match_shapes(a, b_fail)


# --- fuse_images ---

def test_fuse_images_returns_jpeg_bytes():
    a = _make(100, 100)
    b = _make(100, 100, seed=1)
    result = fuse_images(a, b)
    assert isinstance(result, bytes)
    assert len(result) > 0
    # verify it's a decodable JPEG
    pil_img = Image.open(io.BytesIO(result))
    assert pil_img.format == "JPEG"


def test_fuse_images_downsizes_large_inputs():
    # 1200x1200 should be downsized to 800x800 by the service
    a = _make(1200, 1200)
    b = _make(1200, 1200, seed=1)
    result = fuse_images(a, b)
    decoded = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) <= 800


def test_fuse_images_rejects_incompatible_aspect_ratios():
    a = _make(100, 200)
    b = _make(100, 100)
    with pytest.raises(FusionError):
        fuse_images(a, b)


def test_fuse_images_handles_portrait_inputs():
    a = _make(133, 100)
    b = _make(133, 100, seed=1)
    result = fuse_images(a, b)
    assert isinstance(result, bytes)
    pil_img = Image.open(io.BytesIO(result))
    assert pil_img.format == "JPEG"
