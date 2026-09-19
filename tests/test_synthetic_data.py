import numpy as np

from synthetic_data import (
    GBMParams,
    MertonParams,
    RegimeSwitchingParams,
    compute_accuracy_scores,
    generate_regime_switching_gbm,
    get_theoretical_moments_gbm,
    get_theoretical_moments_merton,
    simulate_gbm,
    simulate_merton_jump_diffusion,
)


def test_simulate_gbm_is_reproducible_with_a_seed():
    first_prices, first_times = simulate_gbm(100.0, 0.05, 0.2, 1.0, 12, random_state=7)
    second_prices, second_times = simulate_gbm(100.0, 0.05, 0.2, 1.0, 12, random_state=7)

    np.testing.assert_array_equal(first_prices, second_prices)
    np.testing.assert_array_equal(first_times, second_times)


def test_simulate_gbm_rejects_invalid_time_parameters():
    with np.testing.assert_raises_regex(ValueError, "positive"):
        simulate_gbm(100.0, 0.05, 0.2, 1.0, 0)


def test_simulate_merton_rejects_invalid_jump_parameters():
    with np.testing.assert_raises_regex(ValueError, "lambda_"):
        simulate_merton_jump_diffusion(100.0, 0.05, 0.2, -1.0, 0.0, 0.1, 1.0, 12)


def test_simulated_price_paths_have_expected_lengths_and_start_value():
    prices, times = simulate_merton_jump_diffusion(
        100.0, 0.05, 0.2, 2.0, -0.02, 0.05, 1.0, 12, random_state=7
    )

    assert len(prices) == 13
    assert len(times) == 13
    np.testing.assert_allclose(prices[0], 100.0)
    assert np.all(prices > 0)


def test_regime_switching_gbm_returns_consistent_shapes():
    params = RegimeSwitchingParams(
        timesteps_per_year=12,
        n_years=2,
        n_regime_changes=1,
        regime_length=3,
        random_state=7,
    )

    prices, times, labels, intervals = generate_regime_switching_gbm(
        params,
        GBMParams(mu=0.02, sigma=0.2),
        GBMParams(mu=-0.02, sigma=0.3),
    )

    assert len(prices) == len(times) == len(labels) + 1
    assert set(np.unique(labels)).issubset({0, 1})
    assert len(intervals) == 1


def test_regime_switching_rejects_invalid_configuration():
    params = RegimeSwitchingParams(timesteps_per_year=0)

    with np.testing.assert_raises_regex(ValueError, "timesteps_per_year"):
        generate_regime_switching_gbm(params, GBMParams(0.1, 0.2), GBMParams(-0.1, 0.3))


def test_theoretical_moments_scale_with_time_step():
    mean, variance = get_theoretical_moments_gbm(GBMParams(mu=0.1, sigma=0.2), 0.5)
    jump_mean, jump_variance = get_theoretical_moments_merton(
        MertonParams(mu=0.1, sigma=0.2, lambda_=2.0, gamma=0.01, delta=0.05),
        0.5,
    )

    assert mean == (0.1 - 0.2**2 / 2) * 0.5
    assert variance == 0.2**2 * 0.5
    assert jump_mean > mean
    assert jump_variance > variance


def test_accuracy_scores_are_perfect_for_matching_window_labels():
    predicted = np.array([0, 1])
    true_labels = np.array([0, 0, 1, 1])

    total, regime_on, regime_off = compute_accuracy_scores(predicted, true_labels, 2, 2)

    assert (total, regime_on, regime_off) == (1.0, 1.0, 1.0)


def test_accuracy_scores_reject_mismatched_window_labels():
    with np.testing.assert_raises_regex(ValueError, "does not match"):
        compute_accuracy_scores(np.array([0]), np.array([0, 1, 0, 1]), 2, 2)