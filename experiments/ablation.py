"""
Ablation runner: compares our method (full pipeline, and a
no-aggregation variant) against baseline fusion methods, across a set
of real ImagePairs, reporting per-pair and aggregate metrics.

This is the central evidentiary artifact for the project: it answers
two separate questions in one pass --
  1. Does hyperedge aggregation help? (ours-full vs ours-no-agg)
  2. Does the overall method beat established baselines? (ours-full
     vs naive averaging vs guided filter fusion)

Kept independent of any single dataset -- callers pass in a
list[ImagePair] from data.registry.load_dataset, so the same runner
works for Lytro, MFI-WHU, or any future addition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from neutrofuse.config import FusionConfig
from neutrofuse.data.types import ImagePair
from neutrofuse.pipeline import run_fusion
from neutrofuse.metrics.evaluate import evaluate_fusion, FusionMetrics
from experiments.baselines.naive_average import fuse_naive_average
from experiments.baselines.guided_filter import fuse_guided_filter


@dataclass(frozen=True)
class MethodResult:
    """One method's result for one pair."""
    method_name: str
    pair_id: str
    metrics: FusionMetrics


@dataclass
class AblationReport:
    """Full results: every (method, pair) combination, plus aggregate
    means per method for quick comparison."""
    results: list[MethodResult] = field(default_factory=list)

    def per_method_mean(self, metric_name: str) -> dict[str, float]:
        """
        Mean of a given metric (mi_total, spatial_frequency, qabf, or
        ssim) across all pairs, grouped by method name.

        ssim entries that are None (e.g. every Lytro pair) are
        excluded from that metric's mean rather than treated as 0 or
        raising -- a method evaluated on a mix of GT-available and
        GT-absent pairs should be scored only on the pairs where SSIM
        is actually meaningful.
        """
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}

        for r in self.results:
            value = getattr(r.metrics, metric_name)
            if value is None:
                continue
            sums[r.method_name] = sums.get(r.method_name, 0.0) + value
            counts[r.method_name] = counts.get(r.method_name, 0) + 1

        return {name: sums[name] / counts[name] for name in sums}

    def method_names(self) -> list[str]:
        seen = []
        for r in self.results:
            if r.method_name not in seen:
                seen.append(r.method_name)
        return seen

    def pair_ids(self) -> list[str]:
        seen = []
        for r in self.results:
            if r.pair_id not in seen:
                seen.append(r.pair_id)
        return seen

    def for_pair(self, pair_id: str) -> list[MethodResult]:
        return [r for r in self.results if r.pair_id == pair_id]


# Method names as constants, so callers comparing against specific
# methods (e.g. in viz or a results table) don't risk a typo'd string
# silently producing an empty filter.
METHOD_OURS_FULL = "ours_full"
METHOD_OURS_NO_AGGREGATION = "ours_no_aggregation"
METHOD_NAIVE_AVERAGE = "naive_average"
METHOD_GUIDED_FILTER = "guided_filter"


def run_ablation(
    pairs: list[ImagePair],
    base_config: FusionConfig | None = None,
    include_baselines: bool = True,
) -> AblationReport:
    """
    Run every method against every pair and collect metrics.

    Parameters
    ----------
    pairs : ImagePairs to evaluate on, e.g. from data.registry.load_dataset.
    base_config : FusionConfig for "ours_full". The no-aggregation
        variant is derived from this by setting indeterminacy_threshold
        to infinity (see fusion.decision.decide_patch: I < theta is
        always true for finite I when theta=inf, so the decision rule
        never enters the hyperedge-aggregation branch -- this is the
        existing, tested mechanism for disabling aggregation, not a
        separate code path).
    include_baselines : if False, only runs the two "ours" variants
        (faster, useful when iterating on the aggregation question
        alone without re-running the slower baselines).

    Returns
    -------
    AblationReport
    """
    cfg_full = base_config or FusionConfig()
    cfg_no_agg = FusionConfig(
        patch_size=cfg_full.patch_size,
        sharpness_metric=cfg_full.sharpness_metric,
        hyperedge_radius=cfg_full.hyperedge_radius,
        similarity_threshold=cfg_full.similarity_threshold,
        indeterminacy_threshold=math.inf,
        clip_percentile=cfg_full.clip_percentile,
    )

    report = AblationReport()

    for pair in pairs:
        result_full = run_fusion(pair.image_a, pair.image_b, cfg_full)
        metrics_full = evaluate_fusion(
            result_full.fused_image, pair.image_a, pair.image_b, pair.ground_truth
        )
        report.results.append(MethodResult(METHOD_OURS_FULL, pair.pair_id, metrics_full))

        result_no_agg = run_fusion(pair.image_a, pair.image_b, cfg_no_agg)
        metrics_no_agg = evaluate_fusion(
            result_no_agg.fused_image, pair.image_a, pair.image_b, pair.ground_truth
        )
        report.results.append(MethodResult(METHOD_OURS_NO_AGGREGATION, pair.pair_id, metrics_no_agg))

        if include_baselines:
            naive_fused = fuse_naive_average(pair.image_a, pair.image_b)
            metrics_naive = evaluate_fusion(naive_fused, pair.image_a, pair.image_b, pair.ground_truth)
            report.results.append(MethodResult(METHOD_NAIVE_AVERAGE, pair.pair_id, metrics_naive))

            gff_fused = fuse_guided_filter(pair.image_a, pair.image_b)
            metrics_gff = evaluate_fusion(gff_fused, pair.image_a, pair.image_b, pair.ground_truth)
            report.results.append(MethodResult(METHOD_GUIDED_FILTER, pair.pair_id, metrics_gff))

    return report
