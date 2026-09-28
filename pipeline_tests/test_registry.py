import pytest

from neutrofuse.data.registry import load_dataset, available_datasets
from neutrofuse.data.types import ImagePair


def test_available_datasets_lists_both():
    names = available_datasets()
    assert "lytro" in names
    assert "mfi-whu" in names


def test_unknown_dataset_name_raises_before_any_loader_call():
    try:
        load_dataset("not_a_real_dataset")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "not_a_real_dataset" in str(e)
        assert "lytro" in str(e)  # error should list valid options


@pytest.mark.network
def test_load_dataset_lytro_dispatches_correctly(tmp_path):
    pairs = load_dataset("lytro", indices=[1], cache_dir=tmp_path)
    assert len(pairs) == 1
    assert isinstance(pairs[0], ImagePair)
    assert pairs[0].source_dataset == "lytro"


@pytest.mark.network
def test_load_dataset_mfiwhu_dispatches_correctly(tmp_path):
    pairs = load_dataset("mfi-whu", indices=[1], cache_dir=tmp_path)
    assert len(pairs) == 1
    assert isinstance(pairs[0], ImagePair)
    assert pairs[0].source_dataset == "mfi-whu"
    assert pairs[0].ground_truth is not None
