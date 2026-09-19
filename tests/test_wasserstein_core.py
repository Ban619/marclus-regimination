import numpy as np
import pytest

from wasserstein_kmeans import (
    compute_log_returns,
    create_sliding_windows,
    wasserstein_barycenter_1d,
    wasserstein_distance_1d,
    compute_mmd_biased,
    compute_mmd_fast,
    compute_self_similarity,
    compute_between_cluster_mmd,
    order_clusters_by_variance,
    MomentKMeans,
    WassersteinKMeans,
)


def test_compute_log_returns_uses_consecutive_prices():
    prices = np.array([100.0, 110.0, 99.0])

    result = compute_log_returns(prices)

    np.testing.assert_allclose(result, np.log([1.1, 0.9]))


def test_compute_log_returns_rejects_non_positive_prices():
    with pytest.raises(ValueError, match="positive"):
        compute_log_returns(np.array([100.0, 0.0, 101.0]))


def test_create_sliding_windows_keeps_expected_overlap():
    returns = np.arange(7, dtype=float)

    windows = create_sliding_windows(returns, h1=4, h2=2)

    assert len(windows) == 2
    np.testing.assert_array_equal(windows[0], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(windows[1], [2.0, 3.0, 4.0, 5.0])


def test_create_sliding_windows_rejects_invalid_window_parameters():
    with pytest.raises(ValueError, match="h1"):
        create_sliding_windows(np.arange(4, dtype=float), h1=0, h2=1)
    with pytest.raises(ValueError, match="negative"):
        create_sliding_windows(np.arange(4, dtype=float), h1=2, h2=-1)


def test_wasserstein_distance_is_zero_for_identical_distributions():
    distribution = np.array([3.0, 1.0, 2.0])

    assert wasserstein_distance_1d(distribution, distribution, p=1) == 0.0


def test_wasserstein_distance_rejects_empty_distributions():
    with pytest.raises(ValueError, match="empty"):
        wasserstein_distance_1d(np.array([]), np.array([1.0]))


def test_wasserstein_distance_supports_different_sample_sizes():
    left = np.array([0.0, 2.0])
    right = np.array([0.0, 1.0, 2.0])

    assert wasserstein_distance_1d(left, right, p=1) == 0.0


def test_barycenter_uses_median_for_first_order_distance():
    distributions = [
        np.array([0.0, 2.0]),
        np.array([1.0, 3.0]),
        np.array([2.0, 4.0]),
    ]

    result = wasserstein_barycenter_1d(distributions, p=1)

    np.testing.assert_array_equal(result, [2.0, 3.0])


def test_mmd_implementations_agree_for_one_dimensional_samples():
    left = np.array([-1.0, 0.0, 1.0])
    right = np.array([0.0, 1.0, 2.0])

    assert compute_mmd_fast(left, right) == compute_mmd_biased(left, right)


def test_fast_mmd_preserves_features_for_matrix_samples():
    left = np.array([[0.0, 0.0], [1.0, 1.0]])
    right = np.array([[0.0, 1.0], [1.0, 2.0]])

    result = compute_mmd_fast(left, right)

    np.testing.assert_allclose(result, compute_mmd_biased(left, right))


def test_fast_mmd_rejects_incompatible_samples():
    with pytest.raises(ValueError, match="same number of features"):
        compute_mmd_fast(np.ones((2, 1)), np.ones((2, 2)))


def test_mmd_metrics_reject_non_positive_bandwidth():
    with pytest.raises(ValueError, match="sigma"):
        compute_mmd_fast(np.array([0.0]), np.array([1.0]), sigma=0)
    with pytest.raises(ValueError, match="sigma"):
        compute_self_similarity([np.array([0.0]), np.array([1.0])], sigma=-1)


def test_between_cluster_mmd_rejects_empty_clusters():
    with pytest.raises(ValueError, match="clusters"):
        compute_between_cluster_mmd([], [np.array([0.0])])


def test_self_similarity_is_reproducible_with_a_seed():
    distributions = [
        np.array([-1.0, 0.0, 1.0]),
        np.array([-0.5, 0.0, 0.5]),
        np.array([0.0, 0.5, 1.0]),
    ]

    first = compute_self_similarity(distributions, n_samples=4, random_state=11)
    second = compute_self_similarity(distributions, n_samples=4, random_state=11)

    assert first == second


def test_between_cluster_mmd_returns_requested_number_of_scores():
    cluster1 = [np.array([0.0, 1.0]), np.array([0.0, 2.0])]
    cluster2 = [np.array([2.0, 3.0]), np.array([3.0, 4.0])]

    scores = compute_between_cluster_mmd(cluster1, cluster2, n_samples=7, random_state=11)

    assert scores.shape == (7,)
    assert np.all(scores >= 0)


def test_wasserstein_kmeans_rejects_impossible_cluster_count():
    model = WassersteinKMeans(n_clusters=3, n_init=1)

    with pytest.raises(ValueError, match="exceed"):
        model.fit([np.array([0.0]), np.array([1.0])])


def test_moment_kmeans_rejects_empty_input():
    model = MomentKMeans(n_clusters=2, n_init=1)

    with pytest.raises(ValueError, match="empty"):
        model.fit([])


def test_wasserstein_kmeans_is_reproducible_with_a_seed():
    distributions = [
        np.array([-2.0, -1.0, 0.0]),
        np.array([-1.5, -0.5, 0.5]),
        np.array([4.0, 5.0, 6.0]),
        np.array([4.5, 5.5, 6.5]),
    ]

    first = WassersteinKMeans(n_clusters=2, n_init=3, random_state=19).fit(distributions)
    second = WassersteinKMeans(n_clusters=2, n_init=3, random_state=19).fit(distributions)

    np.testing.assert_array_equal(first.labels_, second.labels_)
    assert first.inertia_ == second.inertia_


def test_wasserstein_kmeans_predict_requires_fit():
    model = WassersteinKMeans(n_clusters=2)

    with pytest.raises(ValueError, match="fitted"):
        model.predict([np.array([0.0])])


def test_cluster_ordering_rejects_inconsistent_lengths():
    with pytest.raises(ValueError, match="same length"):
        order_clusters_by_variance([np.array([0.0])], np.array([0, 1]), [np.array([0.0])])


def test_cluster_ordering_rejects_unknown_labels():
    with pytest.raises(ValueError, match="existing centroid"):
        order_clusters_by_variance(
            [np.array([0.0])], np.array([2]), [np.array([0.0]), np.array([1.0])]
        )