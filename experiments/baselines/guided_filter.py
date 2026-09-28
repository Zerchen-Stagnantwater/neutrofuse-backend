"""
Guided filter fusion (GFF) baseline -- Li, Kang & Hu, "Image Fusion
with Guided Filtering", IEEE TIP 2013.

This is the "strong classic" baseline named in the project's original
architecture: a non-trivial, well-established multi-focus fusion
method that our neutrosophic-hypergraph approach needs to beat on
more than just naive averaging to make a credible case.

Algorithm (two-scale decomposition + guided-filter weight refinement):

1. Two-scale split: for each source, base = average_filter(source),
   detail = source - base. Base captures large-scale intensity,
   detail captures texture/edges.
2. Per-source saliency: Laplacian-filter each source, then take the
   local average of |Laplacian| over a small window -- a simple,
   standard focus/saliency measure (same family as core.sharpness,
   but this baseline intentionally uses its own self-contained
   implementation rather than importing ours, so the comparison
   isn't accidentally sharing machinery with the method it's
   supposed to be a baseline for).
3. Initial weight maps: per-pixel argmax of the two saliency maps
   (binary, not soft -- this is what makes the raw weight maps noisy
   and motivates the next step).
4. Guided-filter refinement: smooth each binary weight map using the
   corresponding source image as the guide, with two different
   (radius, eps) parameter pairs -- a large-radius pass for the base
   layer's weights (smooth, large-scale), a small-radius pass for the
   detail layer's weights (preserve fine structure). This is the
   paper's key mechanism and the reason GFF outperforms naive
   weighted averaging.
5. Normalize the two refined weight maps to sum to 1 at every pixel,
   per layer.
6. Fused base = sum(weight_base_k * base_k), fused detail likewise,
   output = fused_base + fused_detail.

KNOWN UPSTREAM DEFECT (found running the real 20-pair Lytro ablation,
not a hypothetical edge case):

cv2.ximgproc.guidedFilter at the small radius the paper specifies for
the detail layer (radius=7) produces NaN over large contiguous
regions -- sometimes 100% of the image -- on roughly half of the real
Lytro pairs tested. This is NOT the eps-underflow-on-flat-regions
issue (that one is real too, see GFFConstants, and is handled
separately): verified directly that in every affected case here,
both the guide image and the raw weight map have substantial local
variance in the NaN region, ruling out the flat-region explanation.
The actual symptom -- a defective region confined to part of the
image, going away or changing shape under different radii in a way
inconsistent with any local image statistic -- matches two long-
standing, still-open opencv_contrib reports: "Guided filter cuts
right part of an image" (#1288) and "Guided filter might produce
wrong results when radius is small ... only part of the image has
the correct values" (#760), both attributed to the C++
implementation's internal region-parallelized computation rather than
anything about the input. No fixed larger radius reliably avoids it
either -- retrying at radius=15 left 2 of 10 affected real pairs
still 90-99% NaN, so radius is not a safe dial to turn.

The fix applied here is detection + fallback, not parameter-tuning:
after each guided-filter call, check for NaN; if found, discard that
call's result entirely and use the raw (unfiltered) binary weight map
for that pass instead. The raw weight map is a simple boolean
comparison and confirmed NaN-free on all 20 real Lytro pairs, so it's
a safe, always-available fallback. This means GFF silently degrades
toward its own un-smoothed weight map on affected images/layers
rather than crashing -- a real quality cost on those specific cases
(the whole point of the guided-filter step is the smoothing it
usually provides), but a correct and honestly-labeled one: see
GFFResult.detail_filter_failed / base_filter_failed below, which the
ablation runner can use to report how often this occurred rather than
hiding it.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GFFConstants:
    """Parameter values following the original paper's recommendations."""
    avg_filter_radius: int = 15      # two-scale decomposition window
    laplacian_ksize: int = 3
    saliency_window: int = 5          # local-average window for |Laplacian|
    base_guided_radius: int = 45      # large radius -> smooth base-layer weights
    base_guided_eps: float = 0.3 ** 2
    detail_guided_radius: int = 7     # small radius -> preserve detail-layer edges
    detail_guided_eps: float = 1e-3
    # NOTE on eps: cv2.ximgproc.guidedFilter reliably produces NaN output
    # on perfectly flat regions of the guide image when eps <= 1e-4 (the
    # internal a = cov/(var+eps) computation underflows at float32
    # precision when local variance is exactly 0). Verified via direct
    # repro: a constant guide image with eps=1e-4 is 100% NaN, eps=1e-3
    # is 0% NaN. This is a DIFFERENT failure mode from the radius-related
    # one described in the module docstring -- both are handled by the
    # same NaN-detection-and-fallback mechanism in fuse_guided_filter,
    # but they have different root causes and this eps value only
    # defends against this one.


def _guided_filter_with_fallback(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int,
    eps: float,
) -> tuple[np.ndarray, bool]:
    """
    Run cv2.ximgproc.guidedFilter, falling back to the raw (unfiltered)
    src if the result contains any NaN.

    Returns (result, fell_back). See module docstring for why this
    fallback exists -- it is not optional/cosmetic error handling, it
    is the only way this baseline produces a valid result on roughly
    half of real Lytro pairs at the paper's specified detail-layer
    radius.
    """
    result = cv2.ximgproc.guidedFilter(guide, src, radius, eps).astype(np.float64)
    if np.isnan(result).any():
        return src.astype(np.float64).copy(), True
    return result, False


def _two_scale_decompose(image: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (base, detail) where base = box-filtered image, detail = image - base."""
    ksize = 2 * radius + 1
    base = cv2.boxFilter(image.astype(np.float64), -1, (ksize, ksize))
    detail = image.astype(np.float64) - base
    return base, detail


def _saliency_map(gray: np.ndarray, laplacian_ksize: int, window: int) -> np.ndarray:
    """Local average of |Laplacian| -- a simple, standard saliency/focus measure."""
    lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F, ksize=laplacian_ksize)
    abs_lap = np.abs(lap)
    ksize = 2 * window + 1
    return cv2.boxFilter(abs_lap, -1, (ksize, ksize))


def _to_gray_for_guide(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def fuse_guided_filter(
    image_a: np.ndarray,
    image_b: np.ndarray,
    constants: GFFConstants | None = None,
    fallback_info: dict | None = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    image_a, image_b : same shape, dtype uint8, grayscale or color.
    fallback_info : optional dict; if given, this function writes
        boolean keys "base_fell_back" and "detail_fell_back" into it,
        recording whether either guided-filter pass had to fall back
        to the raw weight map due to the NaN defect described in the
        module docstring. Callers that want to report how often this
        happened across a real ablation run (rather than silently
        accepting possibly-degraded results) should pass a dict here
        and inspect it after the call.

    Returns
    -------
    uint8 array, same shape. Always a valid result (no NaN, no
    raised exception from the known guidedFilter defect) -- degraded
    quality on affected images/layers is possible (see module
    docstring) but the function will not crash on it.
    """
    if image_a.shape != image_b.shape:
        raise ValueError(f"Images must match shape: {image_a.shape} vs {image_b.shape}")

    c = constants or GFFConstants()

    gray_a = _to_gray_for_guide(image_a)
    gray_b = _to_gray_for_guide(image_b)

    base_a, detail_a = _two_scale_decompose(image_a, c.avg_filter_radius)
    base_b, detail_b = _two_scale_decompose(image_b, c.avg_filter_radius)

    sal_a = _saliency_map(gray_a, c.laplacian_ksize, c.saliency_window)
    sal_b = _saliency_map(gray_b, c.laplacian_ksize, c.saliency_window)

    # Initial binary weight maps: 1 where A's saliency wins, else 0.
    raw_weight_a = (sal_a >= sal_b).astype(np.float64)

    guide_a = gray_a.astype(np.float32)

    weight_base_a, base_fell_back = _guided_filter_with_fallback(
        guide_a, raw_weight_a.astype(np.float32), c.base_guided_radius, c.base_guided_eps
    )
    weight_detail_a, detail_fell_back = _guided_filter_with_fallback(
        guide_a, raw_weight_a.astype(np.float32), c.detail_guided_radius, c.detail_guided_eps
    )

    if fallback_info is not None:
        fallback_info["base_fell_back"] = base_fell_back
        fallback_info["detail_fell_back"] = detail_fell_back

    weight_base_b = 1.0 - weight_base_a
    weight_detail_b = 1.0 - weight_detail_a

    # Clip in case the guided filter overshoots slightly past [0, 1]
    # at strong edges -- a known property of the algorithm, not a bug.
    weight_base_a = np.clip(weight_base_a, 0.0, 1.0)
    weight_base_b = np.clip(weight_base_b, 0.0, 1.0)
    weight_detail_a = np.clip(weight_detail_a, 0.0, 1.0)
    weight_detail_b = np.clip(weight_detail_b, 0.0, 1.0)

    if image_a.ndim == 3:
        weight_base_a = weight_base_a[:, :, np.newaxis]
        weight_base_b = weight_base_b[:, :, np.newaxis]
        weight_detail_a = weight_detail_a[:, :, np.newaxis]
        weight_detail_b = weight_detail_b[:, :, np.newaxis]

    fused_base = weight_base_a * base_a + weight_base_b * base_b
    fused_detail = weight_detail_a * detail_a + weight_detail_b * detail_b

    fused = fused_base + fused_detail

    # This should now be unreachable -- both guided-filter calls are
    # NaN-protected above -- but kept as a hard backstop in case a
    # future change introduces a new NaN source elsewhere in the
    # pipeline (e.g. a different OpenCV call). Failing loudly here is
    # strictly better than the alternative of silently returning
    # corrupted uint8 data via np.clip(nan, 0, 255).astype(uint8),
    # which produces 0, not an error.
    if np.isnan(fused).any():
        raise FloatingPointError(
            "fuse_guided_filter produced NaN output despite the "
            "base/detail NaN fallback both being active. This means "
            "the input itself is unsafe (e.g. NaN/Inf present in "
            "image_a or image_b) or there is NaN coming from somewhere "
            "other than the two guidedFilter calls -- check those "
            "first before assuming this is the known guidedFilter "
            "radius/eps defect (see module docstring), since that one "
            "should already be handled."
        )

    return np.clip(fused, 0, 255).astype(np.uint8)
