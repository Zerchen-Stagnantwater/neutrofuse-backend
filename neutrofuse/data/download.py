"""
Shared download utility with on-disk caching.

Every loader needs "fetch this URL to this path, skip if already
cached, raise something actionable on failure" -- that logic lives
here once rather than duplicated per-dataset.
"""
from __future__ import annotations

from pathlib import Path

import requests


class DownloadError(RuntimeError):
    """Raised when a dataset file could not be fetched."""


def download_file(url: str, dest_path: Path, timeout: int = 30) -> Path:
    """
    Download url to dest_path if dest_path doesn't already exist.

    Returns dest_path. Raises DownloadError with the URL and HTTP
    status on failure, rather than letting a raw requests exception
    surface -- callers (and their error messages to users) shouldn't
    need to know this uses requests internally.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        raise DownloadError(f"Failed to download {url}: {e}") from e

    if len(response.content) == 0:
        raise DownloadError(f"Downloaded empty file from {url}")

    dest_path.write_bytes(response.content)
    return dest_path
