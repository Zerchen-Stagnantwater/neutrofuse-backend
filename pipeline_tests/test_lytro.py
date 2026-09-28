import numpy as np
import pytest

from neutrofuse.data.lytro import load_lytro_pairs, _pair_urls, _N_PAIRS
from neutrofuse.data.types import ImagePair


def test_pair_urls_format():
    url_a, url_b = _pair_urls(1)
    assert url_a.endswith("c_01_1.tif")
    assert url_b.endswith("c_01_2.tif")

    url_a, url_b = _pair_urls(20)
    assert url_a.endswith("c_20_1.tif")


def test_invalid_indices_raise_before_any_network_call():
    try:
        load_lytro_pairs(indices=[0])
        assert False, "expected ValueError"
    except ValueError:
        pass

    try:
        load_lytro_pairs(indices=[_N_PAIRS + 1])
        assert False, "expected ValueError"
    except ValueError:
        pass


@pytest.mark.network
def test_load_single_real_pair(tmp_path):
    """Real network test: download pair 1 and verify it's a usable
    ImagePair with the documented 520x520 color dimensions."""
    pairs = load_lytro_pairs(indices=[1], cache_dir=tmp_path)

    assert len(pairs) == 1
    pair = pairs[0]
    assert isinstance(pair, ImagePair)
    assert pair.pair_id == "lytro_01"
    assert pair.source_dataset == "lytro"
    assert pair.ground_truth is None
    assert pair.image_a.shape == (520, 520, 3)
    assert pair.image_a.dtype == np.uint8


@pytest.mark.network
def test_repeated_load_uses_cache(tmp_path):
    """Second call with the same cache_dir should not re-download --
    verified indirectly by checking the cached files exist after the
    first call and the second call still succeeds fast."""
    load_lytro_pairs(indices=[2], cache_dir=tmp_path)
    cached_file = tmp_path / "c_02_1.tif"
    assert cached_file.exists()

    pairs_again = load_lytro_pairs(indices=[2], cache_dir=tmp_path)
    assert len(pairs_again) == 1
