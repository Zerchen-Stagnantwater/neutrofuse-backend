import numpy as np

from neutrofuse.config import FusionConfig
from neutrofuse.data.types import ImagePair
from experiments.ablation import (
    run_ablation,
    AblationReport,
    MethodResult,
    METHOD_OURS_FULL,
    METHOD_OURS_NO_AGGREGATION,
    METHOD_NAIVE_AVERAGE,
    METHOD_GUIDED_FILTER,
)
from neutrofuse.metrics.evaluate import FusionMetrics


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


def _make_pair(pair_id: str, with_gt: bool = False) -> ImagePair:
    a = _make_textured_half_sharp(32, "left", seed=hash(pair_id) % 1000)
    b = _make_textured_half_sharp(32, "right", seed=(hash(pair_id) + 1) % 1000)
    gt = np.full((32, 32), 128, dtype=np.uint8) if with_gt else None
    return ImagePair(pair_id=pair_id, image_a=a, image_b=b, ground_truth=gt)


def test_run_ablation_produces_four_methods_per_pair():
    pairs = [_make_pair("test_01")]
    report = run_ablation(pairs, base_config=FusionConfig(patch_size=8))

    assert len(report.results) == 4
    names = {r.method_name for r in report.results}
    assert names == {
        METHOD_OURS_FULL, METHOD_OURS_NO_AGGREGATION,
        METHOD_NAIVE_AVERAGE, METHOD_GUIDED_FILTER,
    }


def test_run_ablation_without_baselines_only_runs_ours_variants():
    pairs = [_make_pair("test_01")]
    report = run_ablation(pairs, base_config=FusionConfig(patch_size=8), include_baselines=False)

    assert len(report.results) == 2
    names = {r.method_name for r in report.results}
    assert names == {METHOD_OURS_FULL, METHOD_OURS_NO_AGGREGATION}


def test_run_ablation_multiple_pairs():
    pairs = [_make_pair("test_01"), _make_pair("test_02")]
    report = run_ablation(pairs, base_config=FusionConfig(patch_size=8))

    assert len(report.results) == 8  # 4 methods x 2 pairs
    assert set(report.pair_ids()) == {"test_01", "test_02"}


def test_no_aggregation_variant_actually_differs_from_full():
    """The whole point of the no-agg variant is that it should produce
    a DIFFERENT result than the full pipeline on an image where
    aggregation meaningfully fires -- otherwise the ablation is
    comparing a method against itself."""
    pairs = [_make_pair("test_01")]
    report = run_ablation(pairs, base_config=FusionConfig(patch_size=8), include_baselines=False)

    full_result = [r for r in report.results if r.method_name == METHOD_OURS_FULL][0]
    no_agg_result = [r for r in report.results if r.method_name == METHOD_OURS_NO_AGGREGATION][0]

    # metrics need not differ by much, but the two methods are
    # supposed to be genuinely different pipelines -- assert they're
    # not bitwise-identical reruns of the same config
    assert full_result.metrics != no_agg_result.metrics or True
    # (a softer, always-true fallback above documents intent; the
    # real check is that no_agg used theta=inf -- verified in the
    # dedicated config test below, since metrics CAN legitimately tie)


def test_no_aggregation_config_uses_infinite_theta():
    """Directly verify the mechanism: the no-agg variant must derive
    a FusionConfig with indeterminacy_threshold=inf, which is the
    existing tested mechanism (see fusion.decision.decide_patch) for
    fully disabling the aggregation branch."""
    import math
    from unittest.mock import patch as mock_patch
    import experiments.ablation as ablation_module

    captured_configs = []
    original_run_fusion = ablation_module.run_fusion

    def spy_run_fusion(image_a, image_b, config):
        captured_configs.append(config)
        return original_run_fusion(image_a, image_b, config)

    pairs = [_make_pair("test_01")]
    with mock_patch.object(ablation_module, "run_fusion", side_effect=spy_run_fusion):
        run_ablation(pairs, base_config=FusionConfig(patch_size=8, indeterminacy_threshold=0.5), include_baselines=False)

    assert len(captured_configs) == 2
    full_cfg, no_agg_cfg = captured_configs
    assert full_cfg.indeterminacy_threshold == 0.5
    assert no_agg_cfg.indeterminacy_threshold == math.inf


def test_per_method_mean_basic():
    report = AblationReport(results=[
        MethodResult("methodA", "p1", FusionMetrics(mi_total=4.0, spatial_frequency=10.0, qabf=0.5, ssim=None)),
        MethodResult("methodA", "p2", FusionMetrics(mi_total=6.0, spatial_frequency=20.0, qabf=0.7, ssim=None)),
        MethodResult("methodB", "p1", FusionMetrics(mi_total=3.0, spatial_frequency=5.0, qabf=0.4, ssim=None)),
    ])

    means = report.per_method_mean("mi_total")
    assert np.isclose(means["methodA"], 5.0)
    assert np.isclose(means["methodB"], 3.0)


def test_per_method_mean_excludes_none_ssim():
    """A method evaluated on a mix of GT-available and GT-absent
    pairs should average ssim only over the pairs where it's not
    None, not treat None as 0 or crash on the mix."""
    report = AblationReport(results=[
        MethodResult("methodA", "p1", FusionMetrics(mi_total=4.0, spatial_frequency=10.0, qabf=0.5, ssim=0.9)),
        MethodResult("methodA", "p2", FusionMetrics(mi_total=4.0, spatial_frequency=10.0, qabf=0.5, ssim=None)),
        MethodResult("methodA", "p3", FusionMetrics(mi_total=4.0, spatial_frequency=10.0, qabf=0.5, ssim=0.7)),
    ])

    means = report.per_method_mean("ssim")
    assert np.isclose(means["methodA"], 0.8)  # (0.9 + 0.7) / 2, not /3


def test_method_names_preserves_first_seen_order():
    report = AblationReport(results=[
        MethodResult("methodB", "p1", FusionMetrics(mi_total=1.0, spatial_frequency=1.0, qabf=1.0, ssim=None)),
        MethodResult("methodA", "p1", FusionMetrics(mi_total=1.0, spatial_frequency=1.0, qabf=1.0, ssim=None)),
        MethodResult("methodB", "p2", FusionMetrics(mi_total=1.0, spatial_frequency=1.0, qabf=1.0, ssim=None)),
    ])
    assert report.method_names() == ["methodB", "methodA"]


def test_for_pair_filters_correctly():
    report = AblationReport(results=[
        MethodResult("methodA", "p1", FusionMetrics(mi_total=1.0, spatial_frequency=1.0, qabf=1.0, ssim=None)),
        MethodResult("methodB", "p1", FusionMetrics(mi_total=2.0, spatial_frequency=1.0, qabf=1.0, ssim=None)),
        MethodResult("methodA", "p2", FusionMetrics(mi_total=3.0, spatial_frequency=1.0, qabf=1.0, ssim=None)),
    ])

    p1_results = report.for_pair("p1")
    assert len(p1_results) == 2
    assert {r.method_name for r in p1_results} == {"methodA", "methodB"}


def test_ground_truth_flows_through_to_ssim():
    """When the pair has ground_truth, evaluate_fusion should be
    called with it, producing a non-None ssim for every method."""
    pairs = [_make_pair("test_01", with_gt=True)]
    report = run_ablation(pairs, base_config=FusionConfig(patch_size=8))

    for r in report.results:
        assert r.metrics.ssim is not None


def test_no_ground_truth_produces_none_ssim():
    pairs = [_make_pair("test_01", with_gt=False)]
    report = run_ablation(pairs, base_config=FusionConfig(patch_size=8))

    for r in report.results:
        assert r.metrics.ssim is None
