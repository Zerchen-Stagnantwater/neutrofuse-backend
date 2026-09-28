"""Central configuration for the fusion pipeline."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FusionConfig:
    patch_size: int = 8
    sharpness_metric: str = "laplacian_variance"   # see core.sharpness.SHARPNESS_METRICS
    hyperedge_radius: int = 1                       # Chebyshev radius in patch-grid units
    similarity_threshold: float = 0.7               # cosine similarity for hyperedge membership
    indeterminacy_threshold: float = 0.75            # theta: I >= theta triggers aggregation.
    # Calibrated against the log-compressed I distribution (see
    # core.neutrosophic): on real Lytro/MFI-WHU data, I's bulk now
    # sits around 0.6-0.8 (median ~0.74 on lytro_01), so 0.3 from an
    # earlier uncalibrated default left aggregation firing on ~99% of
    # patches -- technically not wrong, but it made the aggregation
    # mask useless as a diagnostic (see viz.decision_map), since it
    # barely discriminated anything. 0.75 gives ~50% aggregation rate
    # on real data with no measurable difference in MI/SF/Qabf across
    # the swept range, so it's chosen for interpretability of the
    # mask, not because metrics required it.
    clip_percentile: float = 1.0                     # percentile clip before T/I normalization
