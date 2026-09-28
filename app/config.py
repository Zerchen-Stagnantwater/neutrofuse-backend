"""
Centralized configuration, read from environment variables with
sensible defaults. Kept in one place so the runtime/quality tradeoffs
established during development (see fusion_service.py) are easy to
find and adjust without hunting through the request-handling code.
"""
from __future__ import annotations

import os


class Settings:
    # Server-side resize ceiling. Verified directly during development:
    # ~4s at 800px on the long edge, ~112s at full 4000x3000 phone-camera
    # resolution. 800 is the default; raise it only with awareness of
    # that cost curve and the consequences for request timeouts on
    # whatever platform this is deployed to.
    max_long_edge: int = int(os.environ.get("NEUTROFUSE_MAX_LONG_EDGE", "800"))

    max_upload_bytes: int = int(os.environ.get("NEUTROFUSE_MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))

    accepted_content_types: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
    )

    # Patch size for the fusion pipeline. 8 matches every test and
    # the real-data ablation results in the research report -- changing
    # this changes the quality/runtime tradeoff in ways not re-verified
    # against real data, so it's deliberately not exposed as an
    # easily-tweaked env var without that caveat in mind.
    patch_size: int = 8

    allowed_origins: list[str] = (
        os.environ.get("NEUTROFUSE_ALLOWED_ORIGINS", "*").split(",")
    )


settings = Settings()
