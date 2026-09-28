"""
Rar archive extraction, used only by the MFI-WHU loader (the dataset
is distributed as a single .rar with no alternative mirror).

IMPORTANT: unrar-free (the apt-installable open implementation, as
opposed to the nonfree `unrar` binary) returns exit code 0 and prints
"All OK" even when asked to extract a path that does not exist inside
the archive -- it silently skips it rather than failing. Exit-code
checking alone is therefore not sufficient; every extraction here is
followed by an explicit check that the expected output file landed on
disk. A genuinely corrupt/unreadable archive does still produce a
nonzero exit code, so that failure mode is caught up front.

This module shells out to whichever extractor is on PATH (checked in
order: unrar, unrar-free, 7z, bsdtar) rather than bundling one, since
none of the pure-Python options on PyPI (e.g. `rarfile`) implement
decompression themselves -- they all wrap one of these same binaries.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class ExtractionError(RuntimeError):
    """Raised when rar extraction fails or produces no usable extractor."""


_CANDIDATE_TOOLS = ["unrar", "unrar-free", "7z", "bsdtar"]


def find_extractor() -> str | None:
    """Return the first available extraction binary's name, or None."""
    for tool in _CANDIDATE_TOOLS:
        if shutil.which(tool):
            return tool
    return None


def _build_command(tool: str, archive_path: Path, internal_paths: list[str], dest_dir: Path) -> list[str]:
    if tool in ("unrar", "unrar-free"):
        # Both accept: <tool> -f <archive> [files...] <dest>
        # (unrar-free's flag set is a strict subset of unrar's for this usage)
        return [tool, "-f", str(archive_path), *internal_paths, str(dest_dir)]
    if tool == "7z":
        return ["7z", "x", f"-o{dest_dir}", "-y", str(archive_path), *internal_paths]
    if tool == "bsdtar":
        return ["bsdtar", "-xf", str(archive_path), "-C", str(dest_dir), *internal_paths]
    raise ExtractionError(f"No command builder for tool '{tool}'")


def extract_files(
    archive_path: Path,
    internal_paths: list[str],
    dest_dir: Path,
) -> dict[str, Path]:
    """
    Extract specific files from a rar archive, preserving their
    internal directory structure under dest_dir.

    Parameters
    ----------
    archive_path : path to the .rar file on disk
    internal_paths : paths inside the archive, e.g. "MFI-WHU/source_1/1.jpg"
    dest_dir : extraction root; files land at dest_dir/<internal_path>

    Returns
    -------
    dict mapping each requested internal_path -> its extracted Path on disk.

    Raises
    ------
    ExtractionError if no extractor binary is found, the tool exits
    nonzero, or (critically) the tool exits 0 but a requested file did
    not actually appear on disk -- see module docstring on unrar-free.
    """
    tool = find_extractor()
    if tool is None:
        raise ExtractionError(
            "No rar extraction tool found on PATH (checked: "
            f"{', '.join(_CANDIDATE_TOOLS)}). Install one, e.g.:\n"
            "  sudo apt-get install unrar-free\n"
            "then re-run."
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    command = _build_command(tool, archive_path, internal_paths, dest_dir)

    result = subprocess.run(command, capture_output=True, text=False)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace") if result.stderr else ""
        raise ExtractionError(
            f"'{tool}' exited with code {result.returncode} extracting {archive_path}: {stderr}"
        )

    extracted: dict[str, Path] = {}
    missing: list[str] = []
    for internal_path in internal_paths:
        candidate = dest_dir / internal_path
        if candidate.exists() and candidate.stat().st_size > 0:
            extracted[internal_path] = candidate
        else:
            missing.append(internal_path)

    if missing:
        # This is the unrar-free silent-skip case: exit code was 0 above,
        # but the file genuinely isn't there. Surface it as a hard error
        # rather than letting callers receive a partial/empty ImagePair.
        raise ExtractionError(
            f"'{tool}' reported success but the following paths were not "
            f"found in {archive_path} after extraction: {missing}. "
            "The internal archive layout may have changed, or the archive "
            "is incomplete -- try deleting the cached .rar and re-downloading."
        )

    return extracted
