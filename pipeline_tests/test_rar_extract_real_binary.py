"""
Integration test against the REAL unrar-free binary (not mocked), to
confirm the unit tests in test_rar_extract.py (which mock subprocess)
accurately reflect actual tool behavior. Skipped if no rar tool is
present on the test machine.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from neutrofuse.data.rar_extract import extract_files, find_extractor, ExtractionError

pytestmark = pytest.mark.skipif(
    find_extractor() is None, reason="no rar extraction tool available on this machine"
)


def _make_test_rar(tmp_path: Path) -> Path:
    """Build a minimal real .rar containing one known file, using
    whatever rar-creation capability is available. If creation isn't
    possible (most rar tools are extract-only, by design -- RAR is a
    proprietary format), skip rather than fail."""
    rar_path = tmp_path / "test.rar"
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "real.txt").write_text("actual content")

    if shutil.which("rar"):
        subprocess.run(
            ["rar", "a", str(rar_path), "real.txt"],
            cwd=src_dir, capture_output=True, check=True,
        )
        return rar_path

    pytest.skip("no rar archive-creation tool available; cannot build a real test fixture")


def test_real_extraction_of_existing_file(tmp_path):
    rar_path = _make_test_rar(tmp_path)
    dest = tmp_path / "dest"
    result = extract_files(rar_path, ["real.txt"], dest)
    assert result["real.txt"].read_text() == "actual content"


def test_real_unrar_free_silent_skip_on_missing_path(tmp_path):
    """Confirms the actual documented unrar-free behavior: requesting
    a nonexistent internal path exits 0 but extracts nothing, and our
    wrapper must turn that into ExtractionError."""
    rar_path = _make_test_rar(tmp_path)
    dest = tmp_path / "dest"
    try:
        extract_files(rar_path, ["does_not_exist.txt"], dest)
        assert False, "expected ExtractionError"
    except ExtractionError as e:
        assert "does_not_exist.txt" in str(e)
