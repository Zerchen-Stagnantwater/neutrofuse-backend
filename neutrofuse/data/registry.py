"""
Dataset registry: single entry point for loading any supported
dataset by name, so experiments/ doesn't need to import individual
loader modules or know their function signatures differ.

Adding a third dataset later: write a new loader module exposing
load_X_pairs(indices, cache_dir) -> list[ImagePair], then add one line
to _LOADERS below.
"""
from __future__ import annotations

from pathlib import Path

from neutrofuse.data.types import ImagePair
from neutrofuse.data.lytro import load_lytro_pairs
from neutrofuse.data.mfi_whu import load_mfiwhu_pairs

_LOADERS = {
    "lytro": load_lytro_pairs,
    "mfi-whu": load_mfiwhu_pairs,
}


def available_datasets() -> list[str]:
    return list(_LOADERS.keys())


def load_dataset(
    name: str,
    indices: list[int] | None = None,
    cache_dir: Path | None = None,
) -> list[ImagePair]:
    """
    Load pairs from a named dataset.

    Parameters
    ----------
    name : "lytro" or "mfi-whu"
    indices : 1-based pair indices, dataset-specific range. None loads
        every pair.
    cache_dir : per-dataset cache override. If None, each loader uses
        its own default subfolder under ~/.cache/neutrofuse/datasets/.

    Returns
    -------
    list[ImagePair]
    """
    if name not in _LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {available_datasets()}"
        )
    return _LOADERS[name](indices, cache_dir)
