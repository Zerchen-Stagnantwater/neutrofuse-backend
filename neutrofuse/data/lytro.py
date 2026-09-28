"""
Lytro multi-focus dataset loader.

Source: 20 color pairs from the Lytro camera, mirrored as individual
.tif files at github.com/yuliu316316/MFIF (sourceimages/color/),
itself redistributing the original Nejati et al. (2015) dataset.

No ground truth (all-in-focus reference) is available for this
dataset -- this is a known property of Lytro, not a loader bug. SSIM
evaluation will report None for every Lytro pair (see metrics.evaluate).

Files are fetched individually via raw.githubusercontent.com rather
than downloading the full repository tarball (~260MB, mostly
pre-computed comparison-method outputs this project doesn't need).
"""
from __future__ import annotations

from pathlib import Path

import cv2

from neutrofuse.data.types import ImagePair, default_cache_dir
from neutrofuse.data.download import download_file, DownloadError

_BASE_URL = "https://raw.githubusercontent.com/yuliu316316/MFIF/master/sourceimages/color"
_N_PAIRS = 20


def _pair_urls(index: int) -> tuple[str, str]:
    """index is 1-based, matching the dataset's own c_01..c_20 naming."""
    stem = f"c_{index:02d}"
    return f"{_BASE_URL}/{stem}_1.tif", f"{_BASE_URL}/{stem}_2.tif"


def _load_image(path: Path) -> "object":
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise DownloadError(
            f"Downloaded file at {path} could not be read as an image "
            "(file may be corrupted or incomplete -- try deleting it and re-running)."
        )
    return img


def load_lytro_pairs(
    indices: list[int] | None = None,
    cache_dir: Path | None = None,
) -> list[ImagePair]:
    """
    Load Lytro multi-focus pairs.

    Parameters
    ----------
    indices : 1-based pair indices to load, e.g. [1, 5, 12]. Defaults
        to all 20 pairs if not given.
    cache_dir : where to cache downloaded files. Defaults to
        ~/.cache/neutrofuse/datasets/lytro.

    Returns
    -------
    list[ImagePair], ground_truth always None, source_dataset="lytro".
    """
    if indices is None:
        indices = list(range(1, _N_PAIRS + 1))

    invalid = [i for i in indices if not (1 <= i <= _N_PAIRS)]
    if invalid:
        raise ValueError(f"Lytro pair indices must be in [1, {_N_PAIRS}], got invalid: {invalid}")

    cache = cache_dir or (default_cache_dir() / "lytro")
    pairs: list[ImagePair] = []

    for idx in indices:
        url_a, url_b = _pair_urls(idx)
        stem = f"c_{idx:02d}"

        path_a = download_file(url_a, cache / f"{stem}_1.tif")
        path_b = download_file(url_b, cache / f"{stem}_2.tif")

        img_a = _load_image(path_a)
        img_b = _load_image(path_b)

        pairs.append(
            ImagePair(
                pair_id=f"lytro_{idx:02d}",
                image_a=img_a,
                image_b=img_b,
                ground_truth=None,
                source_dataset="lytro",
            )
        )

    return pairs
