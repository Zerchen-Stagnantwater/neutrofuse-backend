import numpy as np

from neutrofuse.metrics.qabf import qabf, QABFConstants, _edge_strength_and_orientation, _edge_preservation


def _checkerboard(size: int, block: int) -> np.ndarray:
    """Strong, unambiguous edges for sanity-checking gradient computation."""
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(0, size, block):
        for j in range(0, size, block):
            if ((i // block) + (j // block)) % 2 == 0:
                img[i:i + block, j:j + block] = 255
    return img


def test_qabf_in_unit_range():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    b = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    fused = ((a.astype(np.int32) + b.astype(np.int32)) // 2).astype(np.uint8)

    score = qabf(a, b, fused)
    assert 0.0 <= score <= 1.0


def test_fused_equal_to_source_a_gives_high_preservation_of_a():
    """If fused == A exactly, Q_AF should be near its maximum everywhere
    edges exist, so qabf(A, B, A) > qabf(A, B, random_unrelated)."""
    img_a = _checkerboard(32, 8)
    rng = np.random.default_rng(0)
    img_b = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    unrelated = rng.integers(0, 256, (32, 32), dtype=np.uint8)

    score_fused_is_a = qabf(img_a, img_b, img_a)
    score_fused_is_unrelated = qabf(img_a, img_b, unrelated)

    assert score_fused_is_a > score_fused_is_unrelated


def test_flat_images_give_zero_score():
    """No edges anywhere -> denominator (sum of weights) is ~0 -> score
    should be 0, not NaN or a crash."""
    flat_a = np.full((16, 16), 100, dtype=np.uint8)
    flat_b = np.full((16, 16), 150, dtype=np.uint8)
    flat_fused = np.full((16, 16), 125, dtype=np.uint8)

    score = qabf(flat_a, flat_b, flat_fused)
    assert score == 0.0


def test_shape_mismatch_raises():
    a = np.zeros((8, 8), dtype=np.uint8)
    b = np.zeros((8, 16), dtype=np.uint8)
    fused = np.zeros((8, 8), dtype=np.uint8)
    try:
        qabf(a, b, fused)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_edge_preservation_perfect_match_gives_max_value():
    """When source and fused have identical strength/orientation,
    Q should equal gamma_g * gamma_a (the sigmoid ceiling)."""
    c = QABFConstants()
    g = np.array([[1.0, 2.0]])
    a = np.array([[0.5, 1.0]])

    q = _edge_preservation(g, a, g, a, c)
    expected_max = c.gamma_g * c.gamma_a
    # ratio=1.0 with these sigmoid constants should land very close to the ceiling
    assert np.all(q > 0.9 * expected_max)


def test_orientation_wraps_correctly_near_pi():
    """Orientation difference near pi should wrap to a small effective
    difference (edges at +89 deg and -89 deg are nearly the same edge)."""
    c = QABFConstants()
    g = np.array([[1.0]])
    a_near_pos_pi = np.array([[np.pi / 2 - 0.01]])
    a_near_neg_pi = np.array([[-(np.pi / 2 - 0.01)]])

    # naive |diff| would be ~pi-0.02 (large); wrapped diff should be small
    q_wrapped = _edge_preservation(g, a_near_pos_pi, g, a_near_neg_pi, c)
    q_unwrapped_equivalent = _edge_preservation(g, a_near_pos_pi, g, a_near_pos_pi, c)

    # both orientations are physically close, so preservation scores should
    # be in the same ballpark, not wildly different
    assert q_wrapped[0, 0] > 0.3


def test_edge_strength_and_orientation_shapes():
    img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
    strength, orientation = _edge_strength_and_orientation(img)
    assert strength.shape == (16, 16)
    assert orientation.shape == (16, 16)
    assert np.all(strength >= 0)
