import numpy as np
import pytest

from noether_replay.conditioning import (
    compose_operators,
    decimation_operator,
    fir_operator,
    propagate_covariance,
    selection_operator,
    validate_covariance,
)


def test_ar1_pair_average_matches_analytic_variance_and_cross_covariance():
    rho, variance = 0.8, 4e-18
    indices = np.arange(6)
    covariance = variance * rho ** np.abs(indices[:, None] - indices[None, :])
    filtering = fir_operator(6, np.array([0.5, 0.5]))
    operator = compose_operators(filtering, decimation_operator(5, 2))
    actual = propagate_covariance(covariance, operator)
    expected_variance = variance * (1 + rho) / 2
    expected_adjacent = variance * (rho + 2 * rho**2 + rho**3) / 4
    np.testing.assert_allclose(np.diag(actual), expected_variance, rtol=1e-13, atol=0)
    np.testing.assert_allclose(actual[0, 1], expected_adjacent, rtol=1e-13, atol=0)
    assert actual[0, 0] > propagate_covariance(np.diag(np.diag(covariance)), operator)[0, 0]


def test_fir_uses_valid_causal_support_and_known_alignment():
    values = np.array([1.0, 2.0, 4.0, 8.0])
    operator = fir_operator(4, np.array([1.0, -0.5]))
    np.testing.assert_array_equal(operator @ values, [1.5, 3.0, 6.0])
    assert operator.shape == (3, 4)


def test_selection_repeats_preserve_perfect_correlation():
    operator = selection_operator(3, [2, 0, 2])
    covariance = propagate_covariance(np.diag([1.0, 2.0, 3.0]), operator)
    np.testing.assert_allclose(
        covariance, [[3.0, 0.0, 3.0], [0.0, 1.0, 0.0], [3.0, 0.0, 3.0]], atol=1e-14
    )


def test_sequential_and_composed_conditioning_agree():
    covariance = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 1.0]])
    first = fir_operator(3, np.array([0.2, 0.8]))
    second = selection_operator(2, [1])
    np.testing.assert_allclose(
        propagate_covariance(covariance, compose_operators(first, second)),
        propagate_covariance(propagate_covariance(covariance, first), second),
        rtol=1e-13,
    )


@pytest.mark.parametrize("scale", [1.0, 1e-24, 1e24])
@pytest.mark.parametrize(
    "bad", [np.array([[1.0, 2.0], [2.0, 1.0]]), np.array([[1.0, 0.0], [0.1, 1.0]])]
)
def test_physical_invalidity_is_not_hidden_by_si_scale(scale, bad):
    with pytest.raises(ValueError):
        validate_covariance(scale * bad)


def test_zero_and_singular_covariance_do_not_get_diagonal_jitter():
    np.testing.assert_array_equal(validate_covariance(np.zeros((2, 2))), np.zeros((2, 2)))
    rank_one = np.ones((3, 3)) * 1e-20
    result = validate_covariance(rank_one)
    np.testing.assert_allclose(result, rank_one, rtol=1e-14, atol=0)
    assert np.linalg.matrix_rank(result) == 1


@pytest.mark.parametrize(
    "bad",
    [np.array([[np.nan]]), np.array([[np.inf]]), np.ones((2, 3)), np.eye(513), np.array([["1"]])],
)
def test_invalid_or_unbounded_covariance_rejected(bad):
    with pytest.raises(ValueError):
        validate_covariance(bad)


def test_invalid_operator_dimensions_and_indices_rejected():
    for indices in [[], [-1], [3], [True], [1.2]]:
        with pytest.raises(ValueError):
            selection_operator(3, indices)
    with pytest.raises(ValueError):
        propagate_covariance(np.eye(3), np.ones((2, 4)))
    with pytest.raises(ValueError):
        compose_operators(np.ones((2, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        decimation_operator(5, 2, phase=2)
    with pytest.raises(ValueError):
        fir_operator(2, np.ones(3))


def test_gaussian_samples_independently_match_transformed_covariance():
    rng = np.random.default_rng(24010)
    covariance = 0.7 ** np.abs(np.arange(8)[:, None] - np.arange(8)[None, :])
    operator = compose_operators(
        fir_operator(8, np.array([0.25, 0.5, 0.25])), decimation_operator(6, 2)
    )
    draws = rng.multivariate_normal(np.zeros(8), covariance, size=40000)
    measured = np.cov(draws @ operator.T, rowvar=False)
    expected = propagate_covariance(covariance, operator)
    np.testing.assert_allclose(measured, expected, rtol=0.04, atol=0.008)


def test_psd_projection_cannot_return_nonfinite_covariance():
    covariance = np.full((5, 5), np.finfo(float).max)
    # At this magnitude projection can exceed float64 by roundoff. Either a
    # finite matrix or explicit numerical-range rejection is valid, never inf.
    try:
        result = validate_covariance(covariance)
    except ValueError as exc:
        assert "numerical range" in str(exc)
    else:
        assert np.isfinite(result).all()


@pytest.mark.skipif(
    np.finfo(np.longdouble).max <= np.finfo(float).max,
    reason="longdouble has no wider exponent range on this platform",
)
def test_extended_precision_fir_overflow_is_rejected():
    taps = np.array([np.longdouble(np.finfo(float).max) * 2], dtype=np.longdouble)
    with pytest.raises(ValueError, match="numerical range"):
        fir_operator(2, taps)
