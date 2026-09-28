"""
Core data types shared by all dataset loaders.

ImagePair is the unit every loader (Lytro, MFI-WHU, future additions)
produces, and the unit experiments/ consumes. Keeping this type
loader-agnostic is the actual expansion point: adding a third dataset
later means writing one new loader module that returns a list of
ImagePair, nothing else has to change.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ImagePair:
    """One multi-focus source pair, optionally with ground truth."""
    pair_id: str          # e.g. "lytro_01", "mfiwhu_037"
    image_a: np.ndarray    # near-focus or first source, uint8, color or grayscale
    image_b: np.ndarray    # far-focus or second source, same shape/dtype as image_a
    ground_truth: np.ndarray | None = None  # all-in-focus reference, if the dataset provides one
    source_dataset: str = ""  # "lytro" or "mfi-whu"

    def __post_init__(self):
        if self.image_a.shape != self.image_b.shape:
            raise ValueError(
                f"[{self.pair_id}] image_a/image_b shape mismatch: "
                f"{self.image_a.shape} vs {self.image_b.shape}"
            )
        if self.ground_truth is not None and self.ground_truth.shape != self.image_a.shape:
            raise ValueError(
                f"[{self.pair_id}] ground_truth shape {self.ground_truth.shape} "
                f"does not match source shape {self.image_a.shape}"
            )


def default_cache_dir() -> Path:
    """Where downloaded datasets are cached, so repeated runs don't re-fetch."""
    return Path.home() / ".cache" / "neutrofuse" / "datasets"
