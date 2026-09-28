import numpy as np

from neutrofuse.core.hypergraph import (
    Hypergraph,
    vertex_index,
    vertex_coords,
    build_hypergraph,
    hyperedge_confidence,
    aggregate_confidences,
    aggregate_confidence_grids,
)


def test_vertex_index_and_coords_are_inverses():
    grid_shape = (4, 5)
    for i in range(4):
        for j in range(5):
            idx = vertex_index(i, j, grid_shape)
            assert vertex_coords(idx, grid_shape) == (i, j)


def test_vertex_index_matches_reshape_order():
    """Index must match np.ndarray.reshape(-1) flattening order."""
    grid = np.arange(12).reshape(3, 4)
    flat = grid.ravel()
    for i in range(3):
        for j in range(4):
            idx = vertex_index(i, j, (3, 4))
            assert flat[idx] == grid[i, j]


def test_every_hyperedge_contains_its_center():
    rng = np.random.default_rng(0)
    feature_grid = rng.normal(size=(3, 3, 10))
    hg = build_hypergraph(feature_grid, radius=1, similarity_threshold=0.99)

    for idx, edge in enumerate(hg.hyperedges):
        assert idx in edge


def test_n_hyperedges_equals_n_vertices():
    rng = np.random.default_rng(0)
    feature_grid = rng.normal(size=(4, 5, 10))
    hg = build_hypergraph(feature_grid, radius=1, similarity_threshold=0.5)
    assert len(hg.hyperedges) == hg.n_vertices == 20


def test_identical_features_join_same_hyperedge():
    """If every patch has the identical feature vector, similarity=1 >= any threshold,
    so each hyperedge should include all spatial neighbors within radius."""
    feature_grid = np.ones((3, 3, 4))
    hg = build_hypergraph(feature_grid, radius=1, similarity_threshold=0.99)

    center_idx = vertex_index(1, 1, (3, 3))
    center_edge = hg.hyperedges[center_idx]
    # center (1,1) has 8 neighbors within radius=1 in a 3x3 grid, plus itself
    assert len(center_edge) == 9


def test_high_threshold_isolates_dissimilar_patches():
    """Orthogonal feature vectors have cosine similarity 0, should not join
    a hyperedge under a high similarity threshold."""
    feature_grid = np.zeros((2, 2, 2))
    feature_grid[0, 0] = [1, 0]
    feature_grid[0, 1] = [0, 1]
    feature_grid[1, 0] = [1, 0]
    feature_grid[1, 1] = [0, 1]

    hg = build_hypergraph(feature_grid, radius=1, similarity_threshold=0.9)
    idx_00 = vertex_index(0, 0, (2, 2))
    edge = hg.hyperedges[idx_00]
    # (0,0)=[1,0] is orthogonal to (0,1)=[0,1] and (1,1)=[0,1], but parallel to (1,0)=[1,0]
    idx_01 = vertex_index(0, 1, (2, 2))
    idx_10 = vertex_index(1, 0, (2, 2))
    assert idx_01 not in edge
    assert idx_10 in edge


def test_vertex_to_edges_consistency():
    rng = np.random.default_rng(1)
    feature_grid = rng.normal(size=(3, 3, 6))
    hg = build_hypergraph(feature_grid, radius=1, similarity_threshold=0.3)

    for v in range(hg.n_vertices):
        for e_idx in hg.vertex_to_edges[v]:
            assert v in hg.hyperedges[e_idx]


def test_hyperedge_confidence_is_mean_of_max():
    edge = {0, 1, 2}
    T_a = np.array([0.2, 0.8, 0.5])
    T_b = np.array([0.6, 0.1, 0.9])
    # max per member: 0.6, 0.8, 0.9 -> mean = 0.766...
    expected = np.mean([0.6, 0.8, 0.9])
    assert np.isclose(hyperedge_confidence(edge, T_a, T_b), expected)


def test_aggregate_confidences_matches_self_when_isolated():
    """A vertex whose only hyperedge is itself should aggregate to its own value."""
    hg = Hypergraph(grid_shape=(1, 1), hyperedges=[{0}])
    T_a = np.array([0.7])
    T_b = np.array([0.3])

    agg_a, agg_b = aggregate_confidences(0, hg, T_a, T_b)
    assert np.isclose(agg_a, 0.7)
    assert np.isclose(agg_b, 0.3)


def test_aggregate_confidence_grids_shape_and_range():
    rng = np.random.default_rng(0)
    T_a = rng.uniform(0, 1, (4, 4))
    T_b = rng.uniform(0, 1, (4, 4))
    feature_grid = rng.normal(size=(4, 4, 8))

    hg = build_hypergraph(feature_grid, radius=1, similarity_threshold=0.5)
    T_a_hat, T_b_hat = aggregate_confidence_grids(hg, T_a, T_b)

    assert T_a_hat.shape == (4, 4)
    assert T_b_hat.shape == (4, 4)
    assert np.all(T_a_hat >= 0) and np.all(T_a_hat <= 1)
    assert np.all(T_b_hat >= 0) and np.all(T_b_hat <= 1)


def test_aggregation_pulls_extreme_value_toward_neighborhood_mean():
    """A vertex with an outlier T value, surrounded by similar-feature
    neighbors with a different consensus T, should move toward that
    consensus after aggregation (since its hyperedge includes those
    neighbors due to identical features)."""
    feature_grid = np.ones((3, 3, 4))  # identical features -> fully connected within radius
    T_a = np.full((3, 3), 0.9)
    T_a[1, 1] = 0.1  # center is an outlier
    T_b = np.full((3, 3), 0.1)

    hg = build_hypergraph(feature_grid, radius=1, similarity_threshold=0.99)
    T_a_hat, _ = aggregate_confidence_grids(hg, T_a, T_b)

    center_idx = vertex_index(1, 1, (3, 3))
    flat_hat = T_a_hat.ravel()
    # aggregated value should be pulled up from 0.1 toward the 0.9 consensus
    assert flat_hat[center_idx] > 0.1
