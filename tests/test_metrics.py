import numpy as np

from fsq.metrics import approx_ratio, feasible_rate, good_schedules, mean_pairwise_hamming, sample_stats


def test_metrics_toy():
    w = [1.0, 1.0, 0.5]
    e = [[0, 1]]
    X = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1]])
    assert feasible_rate(X, e) == 0.75
    G = good_schedules(X, w, e, optimum=1.5)
    assert len(G) == 2                     # two distinct optimal schedules
    assert mean_pairwise_hamming(G) == 2.0
    assert abs(approx_ratio(1.2, 1.5) - 0.8) < 1e-12
    s = sample_stats(X, w, e, 1.5)
    assert s["n_good"] == 2 and s["best_ratio"] == 1.0
