"""
Neutrosophic component computation.

Given per-patch sharpness maps for two co-registered source images,
compute the Truth (T), Indeterminacy (I) and Falsity (F) components
for each source.

T_X(p): normalized sharpness confidence for source X at patch p,
        normalized jointly across both sources so T_A and T_B are
        directly comparable on the same scale.

F_X(p) = 1 - T_X(p)

I(p):   comparative indeterminacy, shared between sources.
        I(p) = 1 - |T_A(p) - T_B(p)|
        High when the two sources are nearly equally (un)sharp at p,
        i.e. the decision of which source to trust is genuinely
        ambiguous from local evidence alone. This is the quantity
        that triggers hyperedge aggregation downstream.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NeutrosophicComponents:
    """Per-patch (T, I, F) grids for a single source image."""
    T: np.ndarray
    I: np.ndarray
    F: np.ndarray

    @property
    def shape(self):
        return self.T.shape


def _normalize_jointly(
    sharpness_a: np.ndarray,
    sharpness_b: np.ndarray,
    clip_percentile: float = 1.0,
    use_log_compression: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize both maps to a shared [0, 1] range.

    Real focus-measure distributions (Laplacian variance, Tenengrad)
    are not just heavy-tailed but log-normal-shaped: across a typical
    photograph, sharpness values span 2-4 orders of magnitude between
    flat regions (sky, walls) and strong edges, with the bulk of
    patches sitting far closer to the low end on a LINEAR scale than
    a 0.5 midpoint would suggest. Concretely, on real Lytro data the
    median sharpness sat at ~0.6% of the 99th percentile -- meaning
    even after percentile clipping removes outliers, a linear min-max
    still compresses nearly the entire image toward T=0 for both
    sources, which collapses |T_a - T_b| near 0 almost everywhere and
    makes I read as "ambiguous" when the real situation is "both
    confidences underflowed together".

    log1p compression first turns that multiplicative spread into an
    additive one (median/p99 ratio went from 0.006 to 0.39 on the same
    test image), so the subsequent percentile-clip + min-max actually
    spreads the bulk of real patches across [0, 1] instead of pinning
    them near the floor. Percentile clipping is still applied in log
    space afterward, since outliers remain possible even post-log.
    """
    if use_log_compression:
        sharpness_a = np.log1p(sharpness_a)
        sharpness_b = np.log1p(sharpness_b)

    combined = np.concatenate([sharpness_a.ravel(), sharpness_b.ravel()])
    lo = np.percentile(combined, clip_percentile)
    hi = np.percentile(combined, 100.0 - clip_percentile)

    if hi - lo < 1e-12:
        # Degenerate case: both sources uniformly (un)sharp everywhere.
        # No information to discriminate -> mid confidence everywhere.
        flat_a = np.full_like(sharpness_a, 0.5, dtype=np.float64)
        flat_b = np.full_like(sharpness_b, 0.5, dtype=np.float64)
        return flat_a, flat_b

    clipped_a = np.clip(sharpness_a, lo, hi)
    clipped_b = np.clip(sharpness_b, lo, hi)

    norm_a = (clipped_a - lo) / (hi - lo)
    norm_b = (clipped_b - lo) / (hi - lo)
    return norm_a, norm_b


def compute_neutrosophic_pair(
    sharpness_a: np.ndarray,
    sharpness_b: np.ndarray,
    clip_percentile: float = 1.0,
    use_log_compression: bool = True,
) -> tuple[NeutrosophicComponents, NeutrosophicComponents]:
    """
    Compute (T, I, F) for both sources from their raw per-patch
    sharpness maps. sharpness_a/b must have the same shape.

    clip_percentile : passed through to _normalize_jointly. Default
        of 1.0 clips to [p1, p99] (in log space, if use_log_compression
        is True) before min-max normalizing.
    use_log_compression : apply log1p before normalizing. Default True,
        since real focus-measure distributions are log-normal-shaped
        (see _normalize_jointly docstring). Set False only if sharpness
        is already on a roughly linear/bounded scale (e.g. synthetic
        test data, or a custom sharpness_metric designed differently).
    """
    if sharpness_a.shape != sharpness_b.shape:
        raise ValueError(
            f"Sharpness maps must match: {sharpness_a.shape} vs {sharpness_b.shape}"
        )

    T_a, T_b = _normalize_jointly(sharpness_a, sharpness_b, clip_percentile, use_log_compression)

    I = 1.0 - np.abs(T_a - T_b)

    F_a = 1.0 - T_a
    F_b = 1.0 - T_b

    comp_a = NeutrosophicComponents(T=T_a, I=I, F=F_a)
    comp_b = NeutrosophicComponents(T=T_b, I=I, F=F_b)
    return comp_a, comp_b
