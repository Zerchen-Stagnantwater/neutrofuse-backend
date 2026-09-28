import numpy as np

from neutrofuse.core.hypergraph import Hypergraph
from neutrofuse.fusion.decision import decide_patch, decide_grid, Source


def test_confident_regime_picks_higher_T():
    decision = decide_patch(T_a=0.9, T_b=0.2, I=0.1, T_a_hat=0.5, T_b_hat=0.5, theta=0.3)
    assert decision.source == Source.A
    assert decision.used_aggregation is False


def test_confident_regime_picks_B():
    decision = decide_patch(T_a=0.1, T_b=0.85, I=0.1, T_a_hat=0.5, T_b_hat=0.5, theta=0.3)
    assert decision.source == Source.B
    assert decision.used_aggregation is False


def test_indeterminate_regime_uses_aggregated_values():
    # Direct T's would favor A, but I is high so aggregated values (favoring B) should win.
    decision = decide_patch(T_a=0.9, T_b=0.1, I=0.8, T_a_hat=0.2, T_b_hat=0.9, theta=0.3)
    assert decision.source == Source.B
    assert decision.used_aggregation is True


def test_near_tie_produces_blend():
    decision = decide_patch(T_a=0.51, T_b=0.50, I=0.1, T_a_hat=0.5, T_b_hat=0.5, theta=0.3)
    assert decision.source == Source.BLEND
    assert 0.0 <= decision.blend_weight_a <= 1.0


def test_blend_weight_reflects_relative_confidence():
    decision = decide_patch(T_a=0.501, T_b=0.499, I=0.1, T_a_hat=0.5, T_b_hat=0.5, theta=0.3)
    assert decision.source == Source.BLEND
    assert np.isclose(decision.blend_weight_a, 0.501 / (0.501 + 0.499))


def test_zero_total_confidence_blends_evenly():
    decision = decide_patch(T_a=0.0, T_b=0.0, I=0.1, T_a_hat=0.0, T_b_hat=0.0, theta=0.3)
    assert decision.source == Source.BLEND
    assert np.isclose(decision.blend_weight_a, 0.5)


def test_decide_grid_shapes_and_types():
    rng = np.random.default_rng(0)
    T_a = rng.uniform(0, 1, (3, 3))
    T_b = rng.uniform(0, 1, (3, 3))
    I = rng.uniform(0, 1, (3, 3))

    hyperedges = [{i} for i in range(9)]  # trivial: each vertex its own hyperedge
    hg = Hypergraph(grid_shape=(3, 3), hyperedges=hyperedges)

    sources, blend_weights, used_agg_mask = decide_grid(T_a, T_b, I, hg, theta=0.3)

    assert sources.shape == (3, 3)
    assert blend_weights.shape == (3, 3)
    assert used_agg_mask.shape == (3, 3)
    assert np.all((blend_weights >= 0) & (blend_weights <= 1))
    assert all(isinstance(s, Source) for s in sources.ravel())


def test_decide_grid_aggregation_mask_matches_I_threshold():
    T_a = np.array([[0.9, 0.9]])
    T_b = np.array([[0.1, 0.1]])
    I = np.array([[0.1, 0.9]])  # first patch confident, second indeterminate

    hyperedges = [{0}, {1}]
    hg = Hypergraph(grid_shape=(1, 2), hyperedges=hyperedges)

    _, _, used_agg_mask = decide_grid(T_a, T_b, I, hg, theta=0.3)
    assert used_agg_mask[0, 0] == False
    assert used_agg_mask[0, 1] == True
