"""
Statistical comparison between two methods' per-pair results.

AblationReport.per_method_mean gives aggregate means, which can hide
the actual story -- a small, consistent improvement and a large,
inconsistent one can produce the same mean difference but mean very
different things about whether the effect is real. This module runs
paired significance tests instead, which is what the per-pair
structure in AblationReport.results actually supports and what an
aggregate mean throws away.

This emerged from running the full ablation on real data: the
aggregate means suggested "ours_full" barely differs from
"ours_no_aggregation" (<1.3% on every metric), which on its own reads
as "aggregation does nothing." A paired test on the same data showed
the opposite -- a small but highly consistent effect, in the same
direction on 18/20 real pairs, p<0.0001. Aggregate means are not
wrong here, but they're not the whole story either; this module
exists so that distinction doesn't get lost.

REAL FINDING, with mechanism verified directly on real data -- not
just a number, an explanation: aggregation (ours_full) significantly
IMPROVES qabf (p=7.8e-6, 18/20 pairs) and mi_total (p=4.7e-6, 17/20)
relative to no-aggregation, but significantly WORSENS spatial_frequency
(p=0.0012, only 2/20 pairs favor aggregation). Traced the mechanism on
lytro_01: of the 183 patches where aggregation flips the source
decision relative to no-aggregation, 122 flip from B to A specifically
at patches where B was the LOCALLY sharper source (mean raw sharpness
49.1 vs A's 23.8 at those exact locations). Aggregation overrides the
locally-optimal sharp choice in favor of neighborhood consensus there,
trading some patch-level sharpness for better structural/information
coherence overall -- a real, explicable tradeoff, not noise or a bug.
This is exactly the kind of result the planned failure-case-analysis
section needs: a place where the method's own design choice has a
measurable cost, with a verified mechanism, not a vague gesture at
"limitations."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from experiments.ablation import AblationReport


@dataclass(frozen=True)
class PairedComparison:
    """Result of comparing method_a against method_b on one metric,
    across every pair both methods were evaluated on."""
    method_a: str
    method_b: str
    metric_name: str
    n_pairs: int
    mean_diff: float          # mean(method_a - method_b), positive = a wins on average
    wins_a: int                # pairs where method_a > method_b
    wins_b: int                # pairs where method_b > method_a
    ties: int                  # pairs where they're equal (rare with float metrics, but possible)
    t_statistic: float
    p_value: float              # paired t-test, two-sided

    @property
    def significant_at_05(self) -> bool:
        return self.p_value < 0.05

    def summary_line(self) -> str:
        """One-line human-readable summary, e.g. for a results table
        or console report -- intentionally states the win count
        alongside the p-value, since either number alone can overstate
        the result (a narrow win-count majority can still be
        significant if the effect is highly consistent, and vice versa
        a large mean difference driven by a few pairs can be
        non-significant)."""
        direction = "wins" if self.mean_diff > 0 else "loses"
        sig = "significant" if self.significant_at_05 else "not significant"
        return (
            f"{self.method_a} {direction} vs {self.method_b} on {self.metric_name}: "
            f"mean_diff={self.mean_diff:+.4f}, {self.method_a} wins {self.wins_a}/{self.n_pairs} pairs, "
            f"p={self.p_value:.4g} ({sig})"
        )


def compare_methods(
    report: AblationReport,
    method_a: str,
    method_b: str,
    metric_name: str,
) -> PairedComparison:
    """
    Paired comparison of two methods on one metric, across every pair
    where both methods have a non-None value for that metric.

    Parameters
    ----------
    report : an AblationReport from experiments.ablation.run_ablation
    method_a, method_b : method name constants, e.g.
        experiments.ablation.METHOD_OURS_FULL
    metric_name : one of "mi_total", "spatial_frequency", "qabf", "ssim".
        For "ssim", pairs where either method has ssim=None (no ground
        truth available, e.g. every Lytro pair) are excluded -- same
        convention as AblationReport.per_method_mean.

    Raises
    ------
    ValueError if there are zero pairs with valid values for both
    methods on this metric (e.g. comparing ssim across an all-Lytro
    report), since a statistical test on zero data points isn't
    meaningful and silently returning NaN/0 would be worse than
    failing loudly here.
    """
    diffs: list[float] = []

    for pair_id in report.pair_ids():
        results_for_pair = report.for_pair(pair_id)
        result_a = next((r for r in results_for_pair if r.method_name == method_a), None)
        result_b = next((r for r in results_for_pair if r.method_name == method_b), None)

        if result_a is None or result_b is None:
            continue  # this pair wasn't evaluated under one of the two methods

        value_a = getattr(result_a.metrics, metric_name)
        value_b = getattr(result_b.metrics, metric_name)

        if value_a is None or value_b is None:
            continue  # e.g. ssim with no ground truth for this pair

        diffs.append(value_a - value_b)

    if len(diffs) == 0:
        raise ValueError(
            f"No pairs have valid '{metric_name}' values for both "
            f"'{method_a}' and '{method_b}' -- cannot run a statistical "
            f"test on zero data points. If comparing ssim, check that "
            f"at least one pair in this report has ground_truth set."
        )

    diffs_arr = np.array(diffs)
    wins_a = int(np.sum(diffs_arr > 0))
    wins_b = int(np.sum(diffs_arr < 0))
    ties = int(np.sum(diffs_arr == 0))

    if len(diffs_arr) >= 2 and not np.allclose(diffs_arr, diffs_arr[0]):
        t_stat, p_value = stats.ttest_1samp(diffs_arr, 0)
    else:
        # Degenerate case: fewer than 2 pairs, or every diff is
        # identical (zero variance) -- ttest_1samp would return NaN
        # or raise. A single data point, or a perfectly constant
        # difference, isn't something a t-test can assign a
        # meaningful p-value to; report it honestly as such rather
        # than letting scipy's NaN propagate silently.
        t_stat, p_value = float("nan"), float("nan")

    return PairedComparison(
        method_a=method_a,
        method_b=method_b,
        metric_name=metric_name,
        n_pairs=len(diffs_arr),
        mean_diff=float(diffs_arr.mean()),
        wins_a=wins_a,
        wins_b=wins_b,
        ties=ties,
        t_statistic=float(t_stat),
        p_value=float(p_value),
    )


def compare_all_metrics(
    report: AblationReport,
    method_a: str,
    method_b: str,
    metric_names: list[str] | None = None,
) -> dict[str, PairedComparison]:
    """
    Run compare_methods across multiple metrics at once.

    Parameters
    ----------
    metric_names : defaults to ["mi_total", "spatial_frequency", "qabf"].
        "ssim" is NOT included by default since it raises if no pair
        in the report has ground truth (e.g. an all-Lytro report) --
        pass it explicitly when the report includes a GT-providing
        dataset like MFI-WHU.

    Returns
    -------
    dict mapping metric_name -> PairedComparison
    """
    names = metric_names or ["mi_total", "spatial_frequency", "qabf"]
    return {name: compare_methods(report, method_a, method_b, name) for name in names}
