"""
NeutroFuse API: a thin, careful FastAPI wrapper around the verified
neutrofuse.pipeline.run_fusion pipeline.

Design principles this module follows, and why:

1. Never trust client-side limits as the only defense. The web
   frontend (neutrofuse-web) downsizes images to 1600px before
   upload, but this API is public and may be called directly --
   the server-side cap here is the real enforcement point.

2. Match the verified runtime envelope. Direct testing during
   development found run_fusion takes ~112s at full 4000x3000 phone-
   camera resolution versus ~4s at 800px on the long edge. The
   default cap here (MAX_LONG_EDGE) is chosen from that same
   measurement, not a guess.

3. Decode images with cv2, not PIL. The pipeline's own dataset
   loaders (neutrofuse.data.lytro, neutrofuse.data.mfi_whu) use
   cv2.imread throughout, which reads BGR channel order. Using PIL
   here instead would silently introduce an RGB/BGR mismatch between
   user-uploaded images and the channel order every internal test
   and the dataset loaders assume. cv2.imdecode keeps this consistent.

4. Never leak internal error detail to the client. Stack traces,
   library version strings, and file-system paths are logged
   server-side and never included in the JSON error body.
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app.config import settings
from app.fusion_service import FusionError, fuse_images

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neutrofuse-api")

app = FastAPI(
    title="NeutroFuse API",
    description=(
        "Multi-focus image fusion using neutrosophic hypergraph "
        "decision-making. Wraps the verified neutrofuse research package."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Liveness/identity check -- not the health check used by deploy
    platforms (see /health), just a friendly response for anyone who
    hits the bare API URL in a browser."""
    return {
        "service": "neutrofuse-api",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Health check endpoint for uptime monitoring / deploy platforms
    (Render, Railway, Fly.io all expect something like this to exist
    and return 200 quickly, without doing real work)."""
    return {"status": "ok"}


def _decode_upload(raw: bytes, field_name: str) -> np.ndarray:
    """Decode uploaded image bytes into a BGR uint8 array via cv2,
    matching the channel convention every other part of the pipeline
    assumes. Raises HTTPException with a client-safe message on
    failure -- corrupted/non-image uploads are a normal occurrence
    for a public endpoint, not a server error."""
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} could not be read as an image. Use a JPEG, PNG, or WebP file.",
        )
    return image


def _validate_size(file: UploadFile, raw: bytes, field_name: str) -> None:
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty.")
    if len(raw) > settings.max_upload_bytes:
        max_mb = settings.max_upload_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is too large. Keep uploads under {max_mb:.0f}MB.",
        )
    if file.content_type not in settings.accepted_content_types:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be a JPEG, PNG, or WebP image.",
        )


@app.post("/fuse")
async def fuse(
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
):
    """
    Fuse two multi-focus source images into one all-in-focus result.

    Returns the fused image as image/jpeg on success. On failure,
    returns a JSON body {"error": "..."} with an appropriate status
    code -- this shape is what neutrofuse-web's proxy route expects
    and forwards to the browser (see its src/app/api/fuse/route.ts).
    """
    raw_a = await image_a.read()
    raw_b = await image_b.read()

    _validate_size(image_a, raw_a, "image_a")
    _validate_size(image_b, raw_b, "image_b")

    array_a = _decode_upload(raw_a, "image_a")
    array_b = _decode_upload(raw_b, "image_b")

    start = time.monotonic()
    try:
        fused_jpeg_bytes = fuse_images(array_a, array_b)
    except FusionError as e:
        logger.warning("Fusion failed for a request: %s", e)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception:
        # Anything unexpected: log full detail server-side, tell the
        # client nothing beyond "something went wrong" -- see module
        # docstring point 4.
        logger.exception("Unexpected error during fusion")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while fusing those photos. Try again.",
        )
    elapsed = time.monotonic() - start
    logger.info("Fused request in %.2fs", elapsed)

    return Response(
        content=fused_jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Reshape FastAPI's default {"detail": "..."} into {"error": "..."}
    to match the contract neutrofuse-web's proxy route expects (see
    its badRequest() helper and error-forwarding logic)."""
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
