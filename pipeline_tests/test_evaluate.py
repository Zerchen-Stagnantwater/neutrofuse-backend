import numpy as np

from neutrofuse.metrics.evaluate import evaluate_fusion, FusionMetrics


def test_evaluate_fusion_without_ground_truth():
    rng = np.random.default_rng(0)
    fused = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    a = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    b = rng.integers(0, 256, (32, 32), dtype=np.uint8)

    metrics = evaluate_fusion(fused, a, b)

    assert isinstance(metrics, FusionMetrics)
    assert metrics.ssim is None
    assert np.isfinite(metrics.mi_total)
    assert np.isfinite(metrics.spatial_frequency)
    assert np.isfinite(metrics.qabf)


def test_evaluate_fusion_with_ground_truth():
    rng = np.random.default_rng(0)
    fused = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    a = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    b = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    gt = rng.integers(0, 256, (32, 32), dtype=np.uint8)

    metrics = evaluate_fusion(fused, a, b, ground_truth=gt)

    assert metrics.ssim is not None
    assert -1.0 <= metrics.ssim <= 1.0


def test_evaluate_fusion_handles_color_images():
    rng = np.random.default_rng(0)
    fused = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    a = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)

    metrics = evaluate_fusion(fused, a, b)
    assert np.isfinite(metrics.mi_total)
    assert np.isfinite(metrics.qabf)


def test_fused_equal_to_a_scores_higher_qabf_for_checkerboard_source():
    """Integration-level sanity check: a fused image that exactly copies
    the more-detailed source should score better on qabf than a fused
    image that's pure noise unrelated to either source."""
    size = 32
    a = np.zeros((size, size), dtype=np.uint8)
    a[:, ::4] = 255  # vertical stripes -> strong, well-defined edges
    rng = np.random.default_rng(0)
    b = rng.integers(0, 256, (size, size), dtype=np.uint8)
    unrelated_fused = rng.integers(0, 256, (size, size), dtype=np.uint8)

    metrics_good = evaluate_fusion(a, a, b)         # fused == A
    metrics_bad = evaluate_fusion(unrelated_fused, a, b)

    assert metrics_good.qabf > metrics_bad.qabf
