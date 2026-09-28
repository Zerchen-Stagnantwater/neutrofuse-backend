"""
Fusion decision rule.

For each patch, decide whether to take it from source A, source B, or
a blend, based on:

  1. Direct comparison of T_A vs T_B when indeterminacy I is below
     the threshold theta -- the evidence is locally unambiguous.
  2. Hyperedge-aggregated confidences (T_A_hat, T_B_hat) when I >= theta
     -- local evidence is ambiguous, so the decision is deferred to
     the neighborhood consensus.

This module is the seam between core/ (neutrosophic math, hypergraph
math) and the rest of the pipeline. Swapping in a different decision
rule (e.g. soft blending instead of hard selection) only requires
changing `decide_patch` and `fuse_patch`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np

from neutrofuse.core.hypergraph import Hypergraph, aggregate_confidence_grids


class Source(Enum):
    A = "A"
    B = "B"
    BLEND = "BLEND"


@dataclass(frozen=True)
class PatchDecision:
    source: Source
    blend_weight_a: float  # in [0, 1]; meaningful only when source == BLEND
    used_aggregation: bool  # True if this decision came from hyperedge consensus


def decide_patch(
    T_a: float,
    T_b: float,
    I: float,
    T_a_hat: float,
    T_b_hat: float,
    theta: float,
) -> PatchDecision:
    """
    Decision rule for a single patch.

    I < theta  -> direct comparison of T_a vs T_b (confident regime)
    I >= theta -> use aggregated T_a_hat vs T_b_hat (consensus regime)

    In both regimes, if the winning margin is small (<0.05) the patch
    is blended rather than hard-selected, to avoid visible seams at
    near-tie boundaries.
    """
    if I < theta:
        eff_a, eff_b = T_a, T_b
        used_aggregation = False
    else:
        eff_a, eff_b = T_a_hat, T_b_hat
        used_aggregation = True

    margin = eff_a - eff_b
    total = eff_a + eff_b

    if abs(margin) < 0.05:
        weight_a = 0.5 if total < 1e-12 else eff_a / total
        return PatchDecision(Source.BLEND, weight_a, used_aggregation)

    if margin > 0:
        return PatchDecision(Source.A, 1.0, used_aggregation)
    return PatchDecision(Source.B, 0.0, used_aggregation)


def decide_grid(
    T_a: np.ndarray,
    T_b: np.ndarray,
    I: np.ndarray,
    hypergraph: Hypergraph,
    theta: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized-ish decision pass over the full patch grid.

    Returns three (gh, gw) arrays:
      sources       : dtype object, holds Source enum per patch
      blend_weights : float, weight_a per patch (1.0 for pure A, 0.0 for pure B)
      used_agg_mask : bool, True where hyperedge aggregation was used
    """
    gh, gw = T_a.shape
    T_a_hat, T_b_hat = aggregate_confidence_grids(hypergraph, T_a, T_b)

    sources = np.empty((gh, gw), dtype=object)
    blend_weights = np.empty((gh, gw), dtype=np.float64)
    used_agg_mask = np.empty((gh, gw), dtype=bool)

    for i in range(gh):
        for j in range(gw):
            decision = decide_patch(
                T_a[i, j], T_b[i, j], I[i, j],
                T_a_hat[i, j], T_b_hat[i, j],
                theta,
            )
            sources[i, j] = decision.source
            blend_weights[i, j] = decision.blend_weight_a
            used_agg_mask[i, j] = decision.used_aggregation

    return sources, blend_weights, used_agg_mask
