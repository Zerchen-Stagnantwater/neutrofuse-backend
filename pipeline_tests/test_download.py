import http.server
import threading
import socketserver
import tempfile
from pathlib import Path

import pytest

from neutrofuse.data.download import download_file, DownloadError


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ok.bin":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(b"hello world")
        elif self.path == "/empty.bin":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"")
        elif self.path == "/notfound.bin":
            self.send_response(404)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # silence test output


@pytest.fixture(scope="module")
def local_server():
    server = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_download_succeeds_and_writes_content(local_server, tmp_path):
    dest = tmp_path / "out.bin"
    result = download_file(f"{local_server}/ok.bin", dest)
    assert result == dest
    assert dest.read_bytes() == b"hello world"


def test_download_skips_if_already_cached(local_server, tmp_path):
    dest = tmp_path / "out.bin"
    dest.write_bytes(b"already here")
    # point at a URL that would 404 if actually requested -- if the cache
    # check works, this should never hit the network and should return
    # the existing content untouched.
    result = download_file(f"{local_server}/notfound.bin", dest)
    assert result == dest
    assert dest.read_bytes() == b"already here"


def test_download_404_raises_download_error(local_server, tmp_path):
    dest = tmp_path / "out.bin"
    try:
        download_file(f"{local_server}/notfound.bin", dest)
        assert False, "expected DownloadError"
    except DownloadError:
        pass
    assert not dest.exists()


def test_download_empty_response_raises(local_server, tmp_path):
    dest = tmp_path / "out.bin"
    try:
        download_file(f"{local_server}/empty.bin", dest)
        assert False, "expected DownloadError"
    except DownloadError:
        pass


def test_download_creates_parent_directories(local_server, tmp_path):
    dest = tmp_path / "nested" / "dirs" / "out.bin"
    download_file(f"{local_server}/ok.bin", dest)
    assert dest.exists()
