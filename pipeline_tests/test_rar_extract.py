import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from neutrofuse.data.rar_extract import (
    find_extractor,
    extract_files,
    ExtractionError,
    _build_command,
)


def test_find_extractor_returns_none_when_nothing_on_path():
    with patch("shutil.which", return_value=None):
        assert find_extractor() is None


def test_find_extractor_returns_first_match():
    def fake_which(name):
        return "/usr/bin/unrar-free" if name == "unrar-free" else None

    with patch("shutil.which", side_effect=fake_which):
        assert find_extractor() == "unrar-free"


def test_build_command_unrar_free():
    cmd = _build_command("unrar-free", Path("/a.rar"), ["x/1.jpg"], Path("/dest"))
    assert cmd == ["unrar-free", "-f", "/a.rar", "x/1.jpg", "/dest"]


def test_build_command_7z():
    cmd = _build_command("7z", Path("/a.rar"), ["x/1.jpg"], Path("/dest"))
    assert cmd[:2] == ["7z", "x"]
    assert "/a.rar" in cmd
    assert "x/1.jpg" in cmd


def test_no_extractor_available_raises_with_install_hint():
    with patch("neutrofuse.data.rar_extract.find_extractor", return_value=None):
        try:
            extract_files(Path("/fake.rar"), ["a.jpg"], Path("/dest"))
            assert False, "expected ExtractionError"
        except ExtractionError as e:
            assert "unrar" in str(e).lower()


def test_nonzero_exit_code_raises_extraction_error(tmp_path):
    fake_proc = subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=b"corrupt archive")

    with patch("neutrofuse.data.rar_extract.find_extractor", return_value="unrar-free"), \
         patch("subprocess.run", return_value=fake_proc):
        try:
            extract_files(tmp_path / "fake.rar", ["a.jpg"], tmp_path / "dest")
            assert False, "expected ExtractionError"
        except ExtractionError as e:
            assert "1" in str(e) or "corrupt" in str(e).lower()


def test_silent_skip_detected_when_exit_zero_but_file_missing(tmp_path):
    """This is the unrar-free-specific bug found during investigation:
    exit code 0, 'All OK', but the requested internal path never
    actually appears on disk because it doesn't exist in the archive.
    extract_files must catch this rather than returning a broken result."""
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"All OK", stderr=b"")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    # deliberately do NOT create dest_dir/missing.jpg, simulating the silent-skip

    with patch("neutrofuse.data.rar_extract.find_extractor", return_value="unrar-free"), \
         patch("subprocess.run", return_value=fake_proc):
        try:
            extract_files(tmp_path / "fake.rar", ["missing.jpg"], dest_dir)
            assert False, "expected ExtractionError"
        except ExtractionError as e:
            assert "missing.jpg" in str(e)


def test_successful_extraction_returns_correct_paths(tmp_path):
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    expected_file = dest_dir / "sub" / "1.jpg"
    expected_file.parent.mkdir(parents=True)
    expected_file.write_bytes(b"fake jpg content")

    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"All OK", stderr=b"")

    with patch("neutrofuse.data.rar_extract.find_extractor", return_value="unrar-free"), \
         patch("subprocess.run", return_value=fake_proc):
        result = extract_files(tmp_path / "fake.rar", ["sub/1.jpg"], dest_dir)

    assert result == {"sub/1.jpg": expected_file}


def test_zero_byte_extracted_file_counts_as_missing(tmp_path):
    """A zero-byte file on disk is as bad as no file -- likely a
    truncated/failed extraction, not a valid image."""
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    empty_file = dest_dir / "1.jpg"
    empty_file.write_bytes(b"")  # zero bytes

    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"All OK", stderr=b"")

    with patch("neutrofuse.data.rar_extract.find_extractor", return_value="unrar-free"), \
         patch("subprocess.run", return_value=fake_proc):
        try:
            extract_files(tmp_path / "fake.rar", ["1.jpg"], dest_dir)
            assert False, "expected ExtractionError"
        except ExtractionError:
            pass
