import numpy as np
import cv2
import pytest

from experiments.baselines.guided_filter import fuse_guided_filter, GFFConstants
from experiments.baselines.naive_average import fuse_naive_average


def _make_textured_half_sharp(size: int, sharp_side: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 128, dtype=np.uint8)
    half = size // 2
    sharp_patch = rng.integers(0, 256, (size, half), dtype=np.uint8)
    if sharp_side == "left":
        img[:, :half] = sharp_patch
    else:
        img[:, half:] = sharp_patch
    return img


def test_output_shape_and_dtype():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    b = rng.integers(0, 256, (64, 64), dtype=np.uint8)

    fused = fuse_guided_filter(a, b)
    assert fused.shape == (64, 64)
    assert fused.dtype == np.uint8


def test_handles_color_images():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

    fused = fuse_guided_filter(a, b)
    assert fused.shape == (64, 64, 3)


def test_output_values_in_valid_range():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    b = rng.integers(0, 256, (64, 64), dtype=np.uint8)

    fused = fuse_guided_filter(a, b)
    assert fused.min() >= 0
    assert fused.max() <= 255


def test_identical_sources_reproduce_input():
    """If A == B exactly, every weight map should be degenerate
    (saliency ties, base+detail reconstruct exactly), so fusing A
    with itself should reproduce A closely."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (64, 64), dtype=np.uint8)

    fused = fuse_guided_filter(a, a)
    # allow small numerical tolerance from box-filter float roundtrip
    diff = np.abs(fused.astype(np.int32) - a.astype(np.int32))
    assert diff.mean() < 2.0


def test_shape_mismatch_raises():
    a = np.zeros((8, 8), dtype=np.uint8)
    b = np.zeros((8, 16), dtype=np.uint8)
    try:
        fuse_guided_filter(a, b)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_sharper_than_naive_averaging_on_synthetic_pair():
    """Directional sanity check: GFF should preserve more sharpness
    than naive pixel averaging on a clean sharp-vs-blurred synthetic
    pair, since that's the entire point of weight-map-based fusion
    over blind averaging. Uses spatial frequency (same metric the
    real evaluation pipeline uses) as the sharpness proxy."""
    from neutrofuse.metrics.spatial_frequency import spatial_frequency

    image_a = _make_textured_half_sharp(64, sharp_side="left", seed=0)
    image_b = _make_textured_half_sharp(64, sharp_side="right", seed=1)

    gff_fused = fuse_guided_filter(image_a, image_b)
    naive_fused = fuse_naive_average(image_a, image_b)

    sf_gff = spatial_frequency(gff_fused)
    sf_naive = spatial_frequency(naive_fused)

    assert sf_gff > sf_naive


def test_custom_constants_are_respected():
    """Changing avg_filter_radius should change the output (confirms
    the constants parameter actually flows through, not silently
    ignored in favor of hardcoded values)."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    b = rng.integers(0, 256, (64, 64), dtype=np.uint8)

    default_result = fuse_guided_filter(a, b)
    custom_result = fuse_guided_filter(a, b, constants=GFFConstants(avg_filter_radius=3))

    assert not np.array_equal(default_result, custom_result)


def test_flat_guide_region_does_not_produce_nan():
    """Regression test for a real bug found via direct investigation:
    cv2.ximgproc.guidedFilter reliably returns NaN over perfectly flat
    regions of the guide image when eps <= 1e-4 (float32 underflow in
    the internal cov/(var+eps) computation when local variance is
    exactly 0). Reproduced directly: a fully-constant 64x64 guide
    image with eps=1e-4 came back 100% NaN; eps=1e-3 came back 0% NaN.
    Real photographs routinely contain flat regions (sky, walls), so
    this test uses a synthetic image with a large flat half plus a
    textured half -- exactly the structure that triggered it."""
    img = np.full((64, 64), 128, dtype=np.uint8)
    img[:, :32] = np.random.default_rng(0).integers(0, 256, (64, 32), dtype=np.uint8)

    other = np.random.default_rng(1).integers(0, 256, (64, 64), dtype=np.uint8)

    fused = fuse_guided_filter(img, other)
    assert not np.isnan(fused.astype(np.float64)).any()
    assert fused.dtype == np.uint8


def test_known_nan_eps_falls_back_instead_of_raising():
    """
    This test's contract intentionally changed from an earlier version
    of this module. The earlier version raised FloatingPointError on
    any NaN reaching the output, treating eps=1e-6 (known to NaN on
    flat guide regions) as something that should fail loudly.

    That was reasonable defensive design for the failure mode it knew
    about, but running the real 20-pair Lytro ablation surfaced a
    SECOND, more common NaN source (see module docstring: an upstream
    cv2.ximgproc.guidedFilter defect at small radii, independent of
    eps, hitting 10/20 real pairs) which the same raise-on-NaN design
    would also trigger -- making the baseline crash on half of any
    real dataset. Since both NaN sources are now handled by the same
    detect-and-fall-back-to-raw-weights mechanism, deliberately
    feeding a known-bad eps must now produce a valid (degraded, but
    correct) result with detail_fell_back=True reported, not a raise.
    """
    flat = np.full((32, 32), 100, dtype=np.uint8)
    other = np.random.default_rng(0).integers(0, 256, (32, 32), dtype=np.uint8)

    bad_constants = GFFConstants(detail_guided_eps=1e-6)  # known to produce NaN on flat regions
    info = {}
    result = fuse_guided_filter(flat, other, constants=bad_constants, fallback_info=info)

    assert not np.isnan(result.astype(np.float64)).any()
    assert result.dtype == np.uint8
    assert info["detail_fell_back"] is True


def test_fallback_info_reports_no_fallback_on_normal_input():
    """On well-behaved input with default constants, neither guided-
    filter pass should need to fall back -- fallback_info should
    reflect that explicitly rather than the dict being left empty."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    b = rng.integers(0, 256, (64, 64), dtype=np.uint8)

    info = {}
    fuse_guided_filter(a, b, fallback_info=info)

    assert info == {"base_fell_back": False, "detail_fell_back": False}


def test_fallback_info_is_optional():
    """Callers that don't care about fallback diagnostics shouldn't
    need to pass fallback_info at all -- the parameter must be
    genuinely optional, not silently required."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    b = rng.integers(0, 256, (32, 32), dtype=np.uint8)

    result = fuse_guided_filter(a, b)  # no fallback_info passed
    assert result.shape == (32, 32)


@pytest.mark.network
def test_real_lytro_pair_with_known_upstream_nan_defect(tmp_path):
    """
    Regression test for a real bug found running the full 20-pair
    Lytro ablation, not a synthetic edge case: lytro_15 triggers
    cv2.ximgproc.guidedFilter's known upstream defect (see module
    docstring; matches opencv_contrib issues #1288 and #760) at
    100% NaN in the detail layer, at the paper's documented radius=7,
    with default (safe, non-degenerate) eps. This is exactly the
    input that the old raise-on-NaN design would have crashed on, and
    exactly why the fallback mechanism exists rather than being
    purely defensive/theoretical."""
    from neutrofuse.data.lytro import load_lytro_pairs

    pairs = load_lytro_pairs(indices=[15], cache_dir=tmp_path)
    pair = pairs[0]

    info = {}
    result = fuse_guided_filter(pair.image_a, pair.image_b, fallback_info=info)

    assert not np.isnan(result.astype(np.float64)).any()
    assert result.shape == pair.image_a.shape
    assert result.dtype == np.uint8
    # this specific real pair is confirmed (via direct investigation)
    # to trigger the defect in the detail layer; assert the fallback
    # mechanism actually engaged rather than the test accidentally
    # passing because the upstream bug stopped reproducing
    assert info["detail_fell_back"] is True


@pytest.mark.network
def test_real_lytro_dataset_fallback_rate(tmp_path):
    """
    Broader real-data check: confirms the fallback mechanism handles
    EVERY real Lytro pair without crashing (the property that matters
    for being able to run the full ablation at all), and reports how
    many needed it -- direct investigation found 10/20 real pairs
    affected in the detail layer, 0/20 in the base layer. This test
    doesn't hard-assert that exact count (OpenCV version differences
    across environments could shift it), but does assert the
    mechanism is doing real work, not just defined-but-never-triggered."""
    from neutrofuse.data.lytro import load_lytro_pairs

    pairs = load_lytro_pairs(indices=list(range(1, 21)), cache_dir=tmp_path)

    detail_fallback_count = 0
    base_fallback_count = 0

    for pair in pairs:
        info = {}
        result = fuse_guided_filter(pair.image_a, pair.image_b, fallback_info=info)
        assert not np.isnan(result.astype(np.float64)).any(), (
            f"{pair.pair_id} produced NaN despite fallback mechanism"
        )
        if info["detail_fell_back"]:
            detail_fallback_count += 1
        if info["base_fell_back"]:
            base_fallback_count += 1

    # the real, specific finding from direct investigation -- documented
    # here as a number, not just prose, so a future change that makes
    # this defect worse (or better) is visible rather than silently
    # drifting
    print(f"\nReal Lytro fallback rate: detail={detail_fallback_count}/20, base={base_fallback_count}/20")
    assert detail_fallback_count > 0  # confirms the mechanism is doing real work on real data
