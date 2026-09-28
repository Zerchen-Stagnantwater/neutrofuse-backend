"""
Wraps neutrofuse.pipeline.run_fusion for API use: resizes inputs to a
shared, bounded resolution, runs the real pipeline, encodes the
result as JPEG bytes.

This is intentionally a thin layer. The actual fusion logic lives in
the neutrofuse package, unmodified and import-as-is, so this API
always reflects exactly what the research report measured -- no
reimplementation, no drift.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from neutrofuse.config import FusionConfig
from neutrofuse.pipeline import run_fusion


class FusionError(Exception):
    """Raised for fusion failures that are the client's fault (e.g.
    mismatched aspect ratios making a sensible shared resize
    impossible) -- distinct from unexpected server errors, so
    app.main can return a 422 rather than a 500 for these."""


def _resize_to_long_edge(image: np.ndarray, max_long_edge: int) -> np.ndarray:
    """Downscale (never upscale) so the longer dimension is at most
    max_long_edge, preserving aspect ratio. INTER_AREA is the correct
    choice for downscaling -- it averages source pixels into each
    destination pixel rather than naively sampling, which matters
    here since the fusion pipeline's sharpness measurements are
    sensitive to aliasing artifacts a naive resize would introduce."""
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    new_w, new_h = round(w * scale), round(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _match_shapes(image_a: np.ndarray, image_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    run_fusion requires image_a.shape == image_b.shape exactly (see
    neutrofuse.pipeline.run_fusion's ValueError). Real user uploads of
    "the same scene" are routinely off by a handful of pixels even
    when shot back-to-back on a tripod (different JPEG encoders,
    slightly different crop, etc.), so this is a normal case to
    handle, not an edge case to reject outright.

    If the two images have a similar aspect ratio (within 5%), resize
    image_b to match image_a's exact dimensions -- a small resize is
    visually inconsequential and lets the request succeed. If the
    aspect ratios differ substantially, the photos likely aren't of
    the same framing, and forcing a resize would distort one of them;
    raise FusionError instead so the client gets an actionable message
    rather than a confusing or visually wrong result.
    """
    h_a, w_a = image_a.shape[:2]
    h_b, w_b = image_b.shape[:2]

    if (h_a, w_a) == (h_b, w_b):
        return image_a, image_b

    aspect_a = w_a / h_a
    aspect_b = w_b / h_b
    aspect_diff = abs(aspect_a - aspect_b) / aspect_a

    if aspect_diff > 0.05:
        raise FusionError(
            "Those two photos don't look like the same framing (very different "
            "aspect ratios). Use two shots of the same scene from the same position."
        )

    image_b_resized = cv2.resize(image_b, (w_a, h_a), interpolation=cv2.INTER_AREA)
    return image_a, image_b_resized


def _encode_jpeg(image: np.ndarray, quality: int = 92) -> bytes:
    """
    Encode via Pillow rather than cv2.imencode for the final output
    step specifically -- Pillow's JPEG encoder handles ICC/orientation
    metadata more predictably across the variety of source formats
    real uploads arrive in, and keeps this one conversion point
    decoupled from cv2's BGR assumption (explicit channel swap below).
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def fuse_images(image_a: np.ndarray, image_b: np.ndarray) -> bytes:
    """
    Resize, shape-match, fuse, and encode. Returns JPEG bytes.

    Raises FusionError for client-attributable failures (aspect ratio
    mismatch). Lets any other exception from run_fusion propagate
    uncaught -- app.main's broad except clause logs and returns a
    generic 500 for those, since an unexpected failure deep in the
    pipeline is a server-side concern, not something to explain to
    the client in detail.
    """
    resized_a = _resize_to_long_edge(image_a, settings.max_long_edge)
    resized_b = _resize_to_long_edge(image_b, settings.max_long_edge)

    matched_a, matched_b = _match_shapes(resized_a, resized_b)

    config = FusionConfig(patch_size=settings.patch_size)
    result = run_fusion(matched_a, matched_b, config)

    return _encode_jpeg(result.fused_image)
