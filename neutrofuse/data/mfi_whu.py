"""
MFI-WHU multi-focus dataset loader.

Source: github.com/HaoZhang1018/MFI-WHU, 120 pairs distributed as a
single MFI-WHU.rar containing three folders with matching numeric
filenames (1.jpg .. 120.jpg, no zero-padding):

    MFI-WHU/source_1/<n>.jpg    -- one defocus variant
    MFI-WHU/source_2/<n>.jpg    -- the other defocus variant
    MFI-WHU/full_clear/<n>.jpg  -- all-in-focus ground truth

Unlike Lytro, ground truth IS available here, so SSIM evaluation works
for every MFI-WHU pair (see metrics.evaluate).

Extraction requires a rar-capable binary on PATH (unrar, unrar-free,
7z, or bsdtar). See data.rar_extract for the extraction logic and an
important caveat about unrar-free's silent-skip behavior on missing
internal paths.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from neutrofuse.data.types import ImagePair, default_cache_dir
from neutrofuse.data.download import download_file, DownloadError
from neutrofuse.data.rar_extract import extract_files, ExtractionError

_RAR_URL = "https://raw.githubusercontent.com/HaoZhang1018/MFI-WHU/master/MFI-WHU.rar"
_N_PAIRS = 120
_ARCHIVE_ROOT = "MFI-WHU"


def _load_image(path: Path) -> "object":
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise DownloadError(
            f"Extracted file at {path} could not be read as an image "
            "(extraction may have produced a truncated file)."
        )
    return img


def load_mfiwhu_pairs(
    indices: list[int] | None = None,
    cache_dir: Path | None = None,
) -> list[ImagePair]:
    """
    Load MFI-WHU multi-focus pairs, with ground truth.

    Parameters
    ----------
    indices : 1-based pair indices to load (matches the dataset's own
        1..120 filenames). Defaults to all 120 pairs if not given.
    cache_dir : where to cache the downloaded .rar and extracted
        files. Defaults to ~/.cache/neutrofuse/datasets/mfi-whu.

    Returns
    -------
    list[ImagePair], ground_truth populated, source_dataset="mfi-whu".

    Raises
    ------
    DownloadError if the .rar can't be fetched.
    ExtractionError if no rar tool is available, or extraction fails
        (see data.rar_extract.extract_files for the failure modes it
        distinguishes).
    """
    if indices is None:
        indices = list(range(1, _N_PAIRS + 1))

    invalid = [i for i in indices if not (1 <= i <= _N_PAIRS)]
    if invalid:
        raise ValueError(f"MFI-WHU pair indices must be in [1, {_N_PAIRS}], got invalid: {invalid}")

    cache = cache_dir or (default_cache_dir() / "mfi-whu")
    archive_path = download_file(_RAR_URL, cache / "MFI-WHU.rar", timeout=120)

    internal_paths = []
    for idx in indices:
        internal_paths.append(f"{_ARCHIVE_ROOT}/source_1/{idx}.jpg")
        internal_paths.append(f"{_ARCHIVE_ROOT}/source_2/{idx}.jpg")
        internal_paths.append(f"{_ARCHIVE_ROOT}/full_clear/{idx}.jpg")

    extracted_dir = cache / "extracted"
    extracted = extract_files(archive_path, internal_paths, extracted_dir)

    pairs: list[ImagePair] = []
    for idx in indices:
        path_a = extracted[f"{_ARCHIVE_ROOT}/source_1/{idx}.jpg"]
        path_b = extracted[f"{_ARCHIVE_ROOT}/source_2/{idx}.jpg"]
        path_gt = extracted[f"{_ARCHIVE_ROOT}/full_clear/{idx}.jpg"]

        img_a = _load_image(path_a)
        img_b = _load_image(path_b)
        img_gt = _load_image(path_gt)

        pairs.append(
            ImagePair(
                pair_id=f"mfiwhu_{idx:03d}",
                image_a=img_a,
                image_b=img_b,
                ground_truth=img_gt,
                source_dataset="mfi-whu",
            )
        )

    return pairs
