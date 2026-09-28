"""
End-to-end multi-focus fusion pipeline.

Wires together: padding -> sharpness -> neutrosophic components ->
feature extraction -> hypergraph construction -> decision -> compose.

This is the single entry point most callers (experiments/, future CLI,
notebooks) should use. It also returns the intermediate grids needed
for visualization and ablation (with/without aggregation), since those
are central to the research deliverables, not an afterthought.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from neutrofuse.config import FusionConfig
from neutrofuse.core.patches import pad_to_multiple
from neutrofuse.core.sharpness import compute_sharpness_map
from neutrofuse.core.neutrosophic import compute_neutrosophic_pair, NeutrosophicComponents
from neutrofuse.core.features import compute_feature_grid
from neutrofuse.core.hypergraph import Hypergraph, build_hypergraph
from neutrofuse.fusion.decision import decide_grid
from neutrofuse.fusion.compose import compose_fused_image


@dataclass
class FusionResult:
    fused_image: np.ndarray
    comp_a: NeutrosophicComponents
    comp_b: NeutrosophicComponents
    hypergraph: Hypergraph
    blend_weights: np.ndarray      # (gh, gw), weight on source A
    used_aggregation_mask: np.ndarray  # (gh, gw), bool


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def run_fusion(
    image_a: np.ndarray,
    image_b: np.ndarray,
    config: FusionConfig | None = None,
) -> FusionResult:
    """
    Fuse two co-registered, differently-focused images of the same scene.

    Parameters
    ----------
    image_a, image_b : np.ndarray, same shape, dtype uint8.
        Grayscale (H, W) or color (H, W, C). Color images are fused
        in color (blend weights derived from grayscale, applied to
        all channels); grayscale derivatives are used only for
        sharpness/feature computation.
    config : FusionConfig, defaults to FusionConfig() if not given.

    Returns
    -------
    FusionResult
    """
    if image_a.shape != image_b.shape:
        raise ValueError(f"Source images must match shape: {image_a.shape} vs {image_b.shape}")

    cfg = config or FusionConfig()

    padded_a, original_shape = pad_to_multiple(image_a, cfg.patch_size)
    padded_b, _ = pad_to_multiple(image_b, cfg.patch_size)

    gray_a = _to_grayscale(padded_a)
    gray_b = _to_grayscale(padded_b)

    sharpness_a = compute_sharpness_map(gray_a, cfg.patch_size, cfg.sharpness_metric)
    sharpness_b = compute_sharpness_map(gray_b, cfg.patch_size, cfg.sharpness_metric)

    comp_a, comp_b = compute_neutrosophic_pair(sharpness_a, sharpness_b, cfg.clip_percentile)

    # Features computed on whichever source is locally sharper, patch-wise,
    # so hyperedges reflect the best available structural evidence rather
    # than being biased toward one source's blur characteristics.
    sharper_mask = sharpness_a >= sharpness_b
    combined_gray = np.where(
        np.repeat(np.repeat(sharper_mask, cfg.patch_size, axis=0), cfg.patch_size, axis=1),
        gray_a, gray_b,
    )
    feature_grid = compute_feature_grid(combined_gray, cfg.patch_size)

    hypergraph = build_hypergraph(
        feature_grid, radius=cfg.hyperedge_radius, similarity_threshold=cfg.similarity_threshold
    )

    _, blend_weights, used_aggregation_mask = decide_grid(
        comp_a.T, comp_b.T, comp_a.I, hypergraph, cfg.indeterminacy_threshold
    )

    fused = compose_fused_image(
        padded_a, padded_b, blend_weights, cfg.patch_size, original_shape=original_shape
    )

    return FusionResult(
        fused_image=fused,
        comp_a=comp_a,
        comp_b=comp_b,
        hypergraph=hypergraph,
        blend_weights=blend_weights,
        used_aggregation_mask=used_aggregation_mask,
    )
