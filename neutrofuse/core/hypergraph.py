"""
Hypergraph construction over the patch grid, and hyperedge-based
aggregation for indeterminate vertices.

Vertex indexing: vertex (i, j) in the patch grid (shape gh x gw) is
flattened to index i * gw + j, matching np.ndarray.reshape order.

Hyperedge construction (neighborhood-hyperedge scheme)
-------------------------------------------------------
For each vertex p, its hyperedge e_p consists of p itself plus every
vertex q within Chebyshev distance `radius` in the patch grid whose
feature vector has cosine similarity >= `similarity_threshold` with
p's feature vector.

This gives exactly one hyperedge per vertex (|hyperedges| == |vertices|),
but hyperedges overlap heavily -- a vertex typically belongs to several
hyperedges (its own, plus those centered on perceptually similar
neighbors). That overlap is what makes aggregation meaningful: an
indeterminate vertex pools sharpness evidence from every hyperedge it
participates in, not just its own immediate neighborhood.

This is the simplest hyperedge-formation strategy and the primary
expansion point of the module -- e.g. swapping in superpixel-derived
regions or multi-scale hyperedges only requires producing a different
list[Set[int]] for `Hypergraph.hyperedges`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set, Tuple

import numpy as np


@dataclass
class Hypergraph:
    grid_shape: Tuple[int, int]              # (gh, gw)
    hyperedges: List[Set[int]]               # one set of vertex indices per hyperedge
    vertex_to_edges: List[List[int]] = field(default_factory=list)  # vertex idx -> hyperedge idxs

    @property
    def n_vertices(self) -> int:
        return self.grid_shape[0] * self.grid_shape[1]

    def __post_init__(self):
        if not self.vertex_to_edges:
            self.vertex_to_edges = [[] for _ in range(self.n_vertices)]
            for e_idx, edge in enumerate(self.hyperedges):
                for v in edge:
                    self.vertex_to_edges[v].append(e_idx)


def vertex_index(i: int, j: int, grid_shape: Tuple[int, int]) -> int:
    _, gw = grid_shape
    return i * gw + j


def vertex_coords(idx: int, grid_shape: Tuple[int, int]) -> Tuple[int, int]:
    _, gw = grid_shape
    return idx // gw, idx % gw


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_hypergraph(
    feature_grid: np.ndarray,
    radius: int = 1,
    similarity_threshold: float = 0.7,
) -> Hypergraph:
    """
    Build a neighborhood hypergraph from a per-patch feature grid.

    Parameters
    ----------
    feature_grid : (gh, gw, feature_dim)
    radius : Chebyshev distance defining the candidate spatial neighborhood
    similarity_threshold : minimum cosine similarity for a neighbor to
        join the hyperedge

    Returns
    -------
    Hypergraph with exactly gh*gw hyperedges (one per vertex).
    Every hyperedge contains at least its center vertex.
    """
    gh, gw, _ = feature_grid.shape
    grid_shape = (gh, gw)

    hyperedges: List[Set[int]] = []

    for i in range(gh):
        for j in range(gw):
            center_idx = vertex_index(i, j, grid_shape)
            center_feat = feature_grid[i, j]
            edge = {center_idx}

            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < gh and 0 <= nj < gw):
                        continue

                    neighbor_feat = feature_grid[ni, nj]
                    sim = _cosine_similarity(center_feat, neighbor_feat)
                    if sim >= similarity_threshold:
                        edge.add(vertex_index(ni, nj, grid_shape))

            hyperedges.append(edge)

    return Hypergraph(grid_shape=grid_shape, hyperedges=hyperedges)


def hyperedge_confidence(edge: Set[int], T_a_flat: np.ndarray, T_b_flat: np.ndarray) -> float:
    """
    Confidence weight w(e) for a hyperedge: mean over its members of
    max(T_A, T_B) -- i.e. how sharp the *better* source is at each
    member, averaged. High-confidence hyperedges dominate aggregation.
    """
    vals = [max(T_a_flat[v], T_b_flat[v]) for v in edge]
    return float(np.mean(vals))


def aggregate_confidences(
    vertex_idx: int,
    hypergraph: Hypergraph,
    T_a_flat: np.ndarray,
    T_b_flat: np.ndarray,
) -> Tuple[float, float]:
    """
    For a single vertex, aggregate T_A and T_B over every hyperedge
    containing that vertex, weighted by hyperedge confidence.

    Returns (T_A_hat, T_B_hat).
    """
    edge_indices = hypergraph.vertex_to_edges[vertex_idx]

    weighted_a = 0.0
    weighted_b = 0.0
    total_weight = 0.0

    for e_idx in edge_indices:
        edge = hypergraph.hyperedges[e_idx]
        w = hyperedge_confidence(edge, T_a_flat, T_b_flat)
        mean_a = float(np.mean([T_a_flat[v] for v in edge]))
        mean_b = float(np.mean([T_b_flat[v] for v in edge]))

        weighted_a += w * mean_a
        weighted_b += w * mean_b
        total_weight += w

    if total_weight < 1e-12:
        return T_a_flat[vertex_idx], T_b_flat[vertex_idx]

    return weighted_a / total_weight, weighted_b / total_weight


def aggregate_confidence_grids(
    hypergraph: Hypergraph,
    T_a: np.ndarray,
    T_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate over the full grid.

    T_a, T_b : (gh, gw)
    Returns T_a_hat, T_b_hat of the same shape.

    NOTE: O(n_vertices * avg_edges_per_vertex * avg_edge_size), pure
    Python. For grid sizes beyond a few thousand vertices this is the
    first place to optimize -- e.g. precompute a sparse vertex-by-edge
    incidence matrix and replace the inner loops with two sparse
    matmuls.
    """
    gh, gw = T_a.shape
    T_a_flat = T_a.ravel()
    T_b_flat = T_b.ravel()

    T_a_hat = np.empty_like(T_a_flat)
    T_b_hat = np.empty_like(T_b_flat)

    for idx in range(gh * gw):
        T_a_hat[idx], T_b_hat[idx] = aggregate_confidences(idx, hypergraph, T_a_flat, T_b_flat)

    return T_a_hat.reshape(gh, gw), T_b_hat.reshape(gh, gw)
