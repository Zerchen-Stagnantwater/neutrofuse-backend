import numpy as np

from neutrofuse.core.neutrosophic import compute_neutrosophic_pair


def test_components_in_unit_range():
    rng = np.random.default_rng(0)
    sharp_a = rng.uniform(0, 100, (4, 4))
    sharp_b = rng.uniform(0, 100, (4, 4))

    comp_a, comp_b = compute_neutrosophic_pair(sharp_a, sharp_b)

    for comp in (comp_a, comp_b):
        assert np.all(comp.T >= 0) and np.all(comp.T <= 1)
        assert np.all(comp.I >= 0) and np.all(comp.I <= 1)
        assert np.all(comp.F >= 0) and np.all(comp.F <= 1)


def test_F_is_complement_of_T():
    rng = np.random.default_rng(0)
    sharp_a = rng.uniform(0, 100, (4, 4))
    sharp_b = rng.uniform(0, 100, (4, 4))

    comp_a, comp_b = compute_neutrosophic_pair(sharp_a, sharp_b)

    np.testing.assert_allclose(comp_a.F, 1 - comp_a.T)
    np.testing.assert_allclose(comp_b.F, 1 - comp_b.T)


def test_indeterminacy_shared_across_sources():
    rng = np.random.default_rng(0)
    sharp_a = rng.uniform(0, 100, (4, 4))
    sharp_b = rng.uniform(0, 100, (4, 4))

    comp_a, comp_b = compute_neutrosophic_pair(sharp_a, sharp_b)
    np.testing.assert_array_equal(comp_a.I, comp_b.I)


def test_indeterminacy_high_when_sources_equal():
    sharp_a = np.array([[10.0, 50.0]])
    sharp_b = np.array([[10.0, 50.0]])  # identical -> max ambiguity everywhere

    comp_a, comp_b = compute_neutrosophic_pair(sharp_a, sharp_b)
    np.testing.assert_allclose(comp_a.I, 1.0)


def test_indeterminacy_low_when_sources_differ_maximally():
    sharp_a = np.array([[0.0, 100.0]])
    sharp_b = np.array([[100.0, 0.0]])

    comp_a, comp_b = compute_neutrosophic_pair(sharp_a, sharp_b)
    np.testing.assert_allclose(comp_a.I, 0.0)


def test_degenerate_equal_sharpness_everywhere():
    sharp_a = np.full((3, 3), 5.0)
    sharp_b = np.full((3, 3), 5.0)

    comp_a, comp_b = compute_neutrosophic_pair(sharp_a, sharp_b)
    np.testing.assert_allclose(comp_a.T, 0.5)
    np.testing.assert_allclose(comp_b.T, 0.5)


def test_heavy_tailed_outlier_does_not_collapse_bulk_distribution():
    """Regression test for a real bug found running on actual Lytro
    data: Laplacian-variance sharpness is heavy-tailed (a handful of
    patches can sit at 50-400x the median), and naive global min-max
    normalization against that single outlier compressed the entire
    rest of the distribution near T=0 for both sources, making I
    read as 'ambiguous' almost everywhere even though most patches
    had a clear, discriminable sharpness difference. Percentile
    clipping should keep the bulk of the distribution spread out."""
    rng = np.random.default_rng(0)
    # 99 patches with normal-range sharpness, 1 extreme outlier --
    # mirrors the real distribution shape found in practice (p99 was
    # ~13x the median, max was ~400x the median on lytro_01).
    sharp_a = np.concatenate([rng.uniform(10, 100, 99), [9000.0]])
    sharp_b = np.concatenate([rng.uniform(5, 50, 99), [50.0]])

    comp_a, comp_b = compute_neutrosophic_pair(sharp_a, sharp_b, clip_percentile=1.0)

    # excluding the outlier, T_a should clearly dominate T_b for most
    # patches (since sharp_a's bulk range is higher), and the mean I
    # over those patches should NOT be pushed up near 1.0 by the
    # single outlier the way global min-max would.
    bulk_I = comp_a.I[:99]
    assert bulk_I.mean() < 0.8


def test_clip_percentile_zero_reduces_to_global_minmax_without_log():
    """clip_percentile=0 clips to [p0, p100] = [min, max]. Verified
    with use_log_compression=False since the default pipeline now
    applies log1p first (see test_log_compression_spreads_real_world_distribution)."""
    sharp_a = np.array([1.0, 5.0, 1000.0])
    sharp_b = np.array([2.0, 4.0, 800.0])

    comp_a, comp_b = compute_neutrosophic_pair(
        sharp_a, sharp_b, clip_percentile=0.0, use_log_compression=False
    )

    combined = np.concatenate([sharp_a, sharp_b])
    lo, hi = combined.min(), combined.max()
    expected_T_a = (sharp_a - lo) / (hi - lo)

    np.testing.assert_allclose(comp_a.T, expected_T_a)


def test_log_compression_spreads_real_world_distribution():
    """Regression test for the actual root cause found on real Lytro
    data: focus-measure sharpness is log-normal-shaped (median sat at
    ~0.6% of p99 on lytro_01), so even after percentile clipping a
    LINEAR min-max still compresses nearly every patch toward T=0 for
    both sources, collapsing |T_a - T_b| near 0 and making I read as
    'ambiguous' almost everywhere. With log1p applied first, the bulk
    of a log-normal-shaped distribution should land away from the
    floor instead of pinned near it."""
    rng = np.random.default_rng(0)
    # Mirrors the real shape: bulk of values low, with a multiplicative
    # (not additive) spread up to a much larger tail value.
    bulk = rng.uniform(10, 100, 200)
    tail = np.array([9000.0, 7000.0])
    sharp_a = np.concatenate([bulk, tail])
    sharp_b = np.concatenate([bulk * 0.3, tail * 0.8])  # correlated but dimmer

    comp_a, _ = compute_neutrosophic_pair(sharp_a, sharp_b, use_log_compression=True)

    # the bulk (first 200 patches) should NOT be pinned near the floor;
    # median T over the bulk should sit at a meaningfully spread value,
    # not collapse toward 0 the way linear normalization does.
    bulk_T = comp_a.T[:200]
    assert np.median(bulk_T) > 0.2


def test_log_compression_can_be_disabled():
    """use_log_compression=False should skip the log1p step entirely,
    reproducing the original (pre-fix) linear-normalization behavior
    for callers who explicitly want it (e.g. a custom sharpness_metric
    already on a bounded linear scale)."""
    sharp_a = np.array([1.0, 10.0, 100.0])
    sharp_b = np.array([1.0, 10.0, 100.0])

    comp_with_log, _ = compute_neutrosophic_pair(sharp_a, sharp_b, use_log_compression=True)
    comp_without_log, _ = compute_neutrosophic_pair(sharp_a, sharp_b, use_log_compression=False)

    # both are valid normalizations but should differ in general,
    # confirming the flag actually changes behavior rather than being
    # silently ignored
    assert not np.allclose(comp_with_log.T, comp_without_log.T)


def test_shape_mismatch_raises():
    sharp_a = np.zeros((2, 2))
    sharp_b = np.zeros((3, 3))
    try:
        compute_neutrosophic_pair(sharp_a, sharp_b)
        assert False, "expected ValueError"
    except ValueError:
        pass
