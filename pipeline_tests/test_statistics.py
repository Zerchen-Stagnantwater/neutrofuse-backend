import numpy as np
import pytest

from experiments.ablation import AblationReport, MethodResult
from experiments.statistics import compare_methods, compare_all_metrics, PairedComparison
from neutrofuse.metrics.evaluate import FusionMetrics


def _make_report(diffs: list[float], metric: str = "qabf") -> AblationReport:
    """Build a minimal report where method_a's metric value is
    method_b's plus a known, controlled diff, for each pair."""
    results = []
    base = 0.5
    for i, d in enumerate(diffs):
        pid = f"p{i}"
        kwargs_a = {"mi_total": 1.0, "spatial_frequency": 1.0, "qabf": 1.0, "ssim": None}
        kwargs_b = {"mi_total": 1.0, "spatial_frequency": 1.0, "qabf": 1.0, "ssim": None}
        kwargs_a[metric] = base + d
        kwargs_b[metric] = base
        results.append(MethodResult("method_a", pid, FusionMetrics(**kwargs_a)))
        results.append(MethodResult("method_b", pid, FusionMetrics(**kwargs_b)))
    return AblationReport(results=results)


def test_compare_methods_basic_mean_diff():
    report = _make_report([0.1, 0.2, 0.3])
    comparison = compare_methods(report, "method_a", "method_b", "qabf")

    assert np.isclose(comparison.mean_diff, 0.2)
    assert comparison.n_pairs == 3
    assert comparison.wins_a == 3
    assert comparison.wins_b == 0
    assert comparison.ties == 0


def test_compare_methods_mixed_wins():
    report = _make_report([0.1, -0.1, 0.2, -0.05])
    comparison = compare_methods(report, "method_a", "method_b", "qabf")

    assert comparison.wins_a == 2
    assert comparison.wins_b == 2


def test_compare_methods_significant_consistent_small_effect():
    """Mirrors the real finding: a small but highly consistent
    positive diff across many pairs should be statistically
    significant, even though the mean difference itself is tiny."""
    rng = np.random.default_rng(0)
    # 20 pairs, mean diff +0.004, small noise, same direction almost always
    diffs = list(0.004 + rng.normal(0, 0.001, 20))
    report = _make_report(diffs)
    comparison = compare_methods(report, "method_a", "method_b", "qabf")

    assert comparison.significant_at_05
    assert comparison.p_value < 0.05


def test_compare_methods_not_significant_when_noisy_and_small_n():
    """A genuinely noisy, inconsistent difference across few pairs
    should NOT come back as significant -- this is the contrast case
    to the consistent-small-effect test above."""
    diffs = [0.1, -0.15, 0.05, -0.08]
    report = _make_report(diffs)
    comparison = compare_methods(report, "method_a", "method_b", "qabf")

    assert not comparison.significant_at_05


def test_compare_methods_raises_on_zero_valid_pairs():
    """If every pair has ssim=None for both methods (e.g. an all-Lytro
    report), comparing on ssim must raise rather than silently
    returning a meaningless 0/NaN result."""
    results = [
        MethodResult("method_a", "p1", FusionMetrics(mi_total=1, spatial_frequency=1, qabf=1, ssim=None)),
        MethodResult("method_b", "p1", FusionMetrics(mi_total=1, spatial_frequency=1, qabf=1, ssim=None)),
    ]
    report = AblationReport(results=results)

    try:
        compare_methods(report, "method_a", "method_b", "ssim")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_compare_methods_excludes_pairs_with_none_ssim_but_keeps_others():
    results = [
        MethodResult("method_a", "p1", FusionMetrics(mi_total=1, spatial_frequency=1, qabf=1, ssim=0.9)),
        MethodResult("method_b", "p1", FusionMetrics(mi_total=1, spatial_frequency=1, qabf=1, ssim=0.8)),
        MethodResult("method_a", "p2", FusionMetrics(mi_total=1, spatial_frequency=1, qabf=1, ssim=None)),
        MethodResult("method_b", "p2", FusionMetrics(mi_total=1, spatial_frequency=1, qabf=1, ssim=None)),
    ]
    report = AblationReport(results=results)

    comparison = compare_methods(report, "method_a", "method_b", "ssim")
    assert comparison.n_pairs == 1  # only p1 has valid ssim for both


def test_compare_methods_handles_zero_variance_diff_gracefully():
    """If every pair has the exact same diff (zero variance), a
    standard t-test's denominator is zero -- must report NaN p-value
    rather than crashing or returning a bogus number."""
    report = _make_report([0.05, 0.05, 0.05, 0.05])
    comparison = compare_methods(report, "method_a", "method_b", "qabf")

    assert np.isclose(comparison.mean_diff, 0.05)
    assert np.isnan(comparison.p_value)


def test_compare_methods_single_pair_returns_nan_pvalue():
    """A single pair has no variance to test against -- t-test needs
    at least 2 points. Should report NaN, not crash."""
    report = _make_report([0.1])
    comparison = compare_methods(report, "method_a", "method_b", "qabf")

    assert comparison.n_pairs == 1
    assert np.isnan(comparison.p_value)


def test_compare_methods_skips_pairs_missing_one_method():
    """If a pair was only evaluated under one of the two methods
    (e.g. include_baselines=False on some run), it should be silently
    excluded, not crash with an index error."""
    results = [
        MethodResult("method_a", "p1", FusionMetrics(mi_total=1, spatial_frequency=1, qabf=0.6, ssim=None)),
        MethodResult("method_b", "p1", FusionMetrics(mi_total=1, spatial_frequency=1, qabf=0.5, ssim=None)),
        MethodResult("method_a", "p2", FusionMetrics(mi_total=1, spatial_frequency=1, qabf=0.7, ssim=None)),
        # p2 has no method_b result
    ]
    report = AblationReport(results=results)

    comparison = compare_methods(report, "method_a", "method_b", "qabf")
    assert comparison.n_pairs == 1


def test_summary_line_contains_key_numbers():
    report = _make_report([0.1, 0.2, 0.3])
    comparison = compare_methods(report, "method_a", "method_b", "qabf")
    line = comparison.summary_line()

    assert "method_a" in line
    assert "method_b" in line
    assert "qabf" in line
    assert "wins" in line


def test_compare_all_metrics_returns_dict_for_default_metrics():
    report = _make_report([0.1, 0.2, 0.3], metric="mi_total")
    # _make_report only varies one metric at a time in this helper,
    # but compare_all_metrics should still run against all three
    # default metrics without raising (qabf/spatial_frequency diffs
    # will just be 0 here, which is a valid, if uninteresting, result)
    results = compare_all_metrics(report, "method_a", "method_b")

    assert set(results.keys()) == {"mi_total", "spatial_frequency", "qabf"}
    assert all(isinstance(v, PairedComparison) for v in results.values())


def test_compare_all_metrics_respects_custom_metric_list():
    report = _make_report([0.1, 0.2, 0.3], metric="mi_total")
    results = compare_all_metrics(report, "method_a", "method_b", metric_names=["mi_total"])

    assert set(results.keys()) == {"mi_total"}


def test_real_finding_reproduced_aggregation_helps_significantly():
    """
    Direct regression test reproducing the real result found running
    the full 20-pair Lytro ablation: 'ours_full' vs
    'ours_no_aggregation' on qabf showed mean_diff=+0.0043,
    18/20 pairs favoring aggregation, p=0.000013 (paired t-test).
    This hardcodes that exact per-pair diff data (captured from the
    real run) so a future change to compare_methods' math can be
    checked against a known-correct real-world answer, not just
    synthetic toy data."""
    real_diffs = [
        0.0060, 0.0051, 0.0078, 0.0063, 0.0123, 0.0018, 0.0044, 0.0003,
        0.0024, -0.0002, 0.0030, 0.0050, -0.0005, 0.0022, 0.0009, 0.0047,
        0.0070, 0.0050, 0.0055, 0.0081,
    ]
    report = _make_report(real_diffs)
    comparison = compare_methods(report, "method_a", "method_b", "qabf")

    assert comparison.wins_a == 18
    assert comparison.wins_b == 2
    assert comparison.significant_at_05
    assert comparison.p_value < 0.001
    assert np.isclose(comparison.mean_diff, np.mean(real_diffs), atol=1e-6)
