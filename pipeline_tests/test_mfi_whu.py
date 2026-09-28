import numpy as np
import pytest

from neutrofuse.data.mfi_whu import load_mfiwhu_pairs, _N_PAIRS
from neutrofuse.data.types import ImagePair


def test_invalid_indices_raise_before_any_network_call():
    try:
        load_mfiwhu_pairs(indices=[0])
        assert False, "expected ValueError"
    except ValueError:
        pass

    try:
        load_mfiwhu_pairs(indices=[_N_PAIRS + 1])
        assert False, "expected ValueError"
    except ValueError:
        pass


@pytest.mark.network
def test_load_single_real_pair_with_ground_truth(tmp_path):
    """Real network + extraction test: download the rar, extract pair
    1 (source_1, source_2, full_clear), verify it's a usable ImagePair
    WITH ground truth populated -- the key difference from Lytro."""
    pairs = load_mfiwhu_pairs(indices=[1], cache_dir=tmp_path)

    assert len(pairs) == 1
    pair = pairs[0]
    assert isinstance(pair, ImagePair)
    assert pair.pair_id == "mfiwhu_001"
    assert pair.source_dataset == "mfi-whu"
    assert pair.ground_truth is not None
    assert pair.image_a.dtype == np.uint8
    assert pair.image_a.shape == pair.ground_truth.shape


@pytest.mark.network
def test_load_multiple_pairs_reuses_cached_archive(tmp_path):
    """Loading a second, different index after the first call should
    reuse the cached .rar (not re-download 18MB again) and only
    extract the newly requested files."""
    load_mfiwhu_pairs(indices=[1], cache_dir=tmp_path)
    archive_path = tmp_path / "MFI-WHU.rar"
    assert archive_path.exists()
    archive_size_after_first = archive_path.stat().st_size

    pairs = load_mfiwhu_pairs(indices=[2], cache_dir=tmp_path)
    assert len(pairs) == 1
    assert pairs[0].pair_id == "mfiwhu_002"
    # archive should be untouched/identical, not re-fetched
    assert archive_path.stat().st_size == archive_size_after_first
