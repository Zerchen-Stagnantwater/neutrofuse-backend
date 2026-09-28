import numpy as np

from neutrofuse.data.types import ImagePair, default_cache_dir


def test_valid_pair_constructs():
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = np.ones((8, 8, 3), dtype=np.uint8)
    pair = ImagePair(pair_id="test_01", image_a=a, image_b=b)
    assert pair.pair_id == "test_01"
    assert pair.ground_truth is None


def test_mismatched_shapes_raise():
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = np.zeros((8, 16, 3), dtype=np.uint8)
    try:
        ImagePair(pair_id="test_01", image_a=a, image_b=b)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_ground_truth_shape_mismatch_raises():
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = np.zeros((8, 8, 3), dtype=np.uint8)
    gt = np.zeros((16, 16, 3), dtype=np.uint8)
    try:
        ImagePair(pair_id="test_01", image_a=a, image_b=b, ground_truth=gt)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_ground_truth_matching_shape_is_valid():
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = np.zeros((8, 8, 3), dtype=np.uint8)
    gt = np.zeros((8, 8, 3), dtype=np.uint8)
    pair = ImagePair(pair_id="test_01", image_a=a, image_b=b, ground_truth=gt)
    assert pair.ground_truth is not None


def test_default_cache_dir_is_under_home():
    cache = default_cache_dir()
    assert "neutrofuse" in str(cache)
    assert "datasets" in str(cache)
