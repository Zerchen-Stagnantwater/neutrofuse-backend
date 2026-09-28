"""
Tests for the /fuse endpoint.

The primary concern here is the HTTP contract: status codes, response
shapes, and error messages. The fusion quality and pipeline correctness
are covered in pipeline_tests/ (the real 190-test suite). These tests
focus on what happens at the API boundary when inputs are malformed,
missing, or valid.
"""
import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def _make_jpeg_bytes(h=100, w=100, seed=0):
    arr = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# --- Liveness and health ---

def test_root_returns_200():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "neutrofuse-api"


def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# --- /fuse happy path ---

def test_fuse_returns_jpeg_on_valid_inputs():
    files = {
        "image_a": ("a.jpg", _make_jpeg_bytes(), "image/jpeg"),
        "image_b": ("b.jpg", _make_jpeg_bytes(seed=1), "image/jpeg"),
    }
    r = client.post("/fuse", files=files)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    img = Image.open(io.BytesIO(r.content))
    assert img.format == "JPEG"


def test_fuse_result_is_not_empty():
    files = {
        "image_a": ("a.jpg", _make_jpeg_bytes(), "image/jpeg"),
        "image_b": ("b.jpg", _make_jpeg_bytes(seed=1), "image/jpeg"),
    }
    r = client.post("/fuse", files=files)
    assert len(r.content) > 1000  # real JPEG is at least a few KB


def test_fuse_cache_control_is_no_store():
    """User-uploaded content must never be cached anywhere."""
    files = {
        "image_a": ("a.jpg", _make_jpeg_bytes(), "image/jpeg"),
        "image_b": ("b.jpg", _make_jpeg_bytes(seed=1), "image/jpeg"),
    }
    r = client.post("/fuse", files=files)
    assert "no-store" in r.headers.get("cache-control", "")


# --- /fuse validation errors ---

def test_fuse_missing_image_b_returns_422():
    files = {"image_a": ("a.jpg", _make_jpeg_bytes(), "image/jpeg")}
    r = client.post("/fuse", files=files)
    assert r.status_code == 422


def test_fuse_missing_both_images_returns_422():
    r = client.post("/fuse")
    assert r.status_code == 422


def test_fuse_wrong_content_type_returns_400():
    txt = b"this is not an image"
    files = {
        "image_a": ("a.txt", txt, "text/plain"),
        "image_b": ("b.jpg", _make_jpeg_bytes(), "image/jpeg"),
    }
    r = client.post("/fuse", files=files)
    assert r.status_code == 400
    assert "error" in r.json()


def test_fuse_empty_file_returns_400():
    files = {
        "image_a": ("a.jpg", b"", "image/jpeg"),
        "image_b": ("b.jpg", _make_jpeg_bytes(), "image/jpeg"),
    }
    r = client.post("/fuse", files=files)
    assert r.status_code == 400
    assert "error" in r.json()


def test_fuse_corrupt_image_bytes_returns_400():
    files = {
        "image_a": ("a.jpg", b"\xff\xd8garbage not a jpeg", "image/jpeg"),
        "image_b": ("b.jpg", _make_jpeg_bytes(), "image/jpeg"),
    }
    r = client.post("/fuse", files=files)
    assert r.status_code == 400
    assert "error" in r.json()


def test_fuse_incompatible_aspect_ratios_returns_422():
    landscape = _make_jpeg_bytes(h=100, w=300)  # 3:1
    portrait = _make_jpeg_bytes(h=300, w=100)    # 1:3
    files = {
        "image_a": ("a.jpg", landscape, "image/jpeg"),
        "image_b": ("b.jpg", portrait, "image/jpeg"),
    }
    r = client.post("/fuse", files=files)
    assert r.status_code == 422
    assert "error" in r.json()


# --- Error response shape compliance ---

def test_error_responses_use_error_key_not_detail():
    """
    FastAPI's default exception format is {"detail": "..."}.
    neutrofuse-web's proxy (src/app/api/fuse/route.ts) reads
    response.json().error, not .detail. All error responses must
    use "error" as the key, which app.main.http_exception_handler
    provides. Verified here so a refactor doesn't silently break
    the frontend-backend contract.
    """
    files = {"image_a": ("a.jpg", b"", "image/jpeg"),
             "image_b": ("b.jpg", _make_jpeg_bytes(), "image/jpeg")}
    r = client.post("/fuse", files=files)
    body = r.json()
    assert "error" in body, f"expected 'error' key, got keys: {list(body.keys())}"
    assert "detail" not in body, "used 'detail' instead of 'error' -- will break the frontend"


def test_get_fuse_returns_405():
    """POST-only endpoint. GET should return 405, not 404 or 500."""
    r = client.get("/fuse")
    assert r.status_code == 405
