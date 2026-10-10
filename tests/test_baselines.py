import numpy as np

from fsq.baselines import exact, exact_bruteforce, greedy, simulated_annealing, _exact_bnb
from fsq.formulate import is_feasible


def random_graph(n, p, seed):
    rng = np.random.default_rng(seed)
    w = list(rng.random(n))
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    return w, edges


def test_exact_matches_bruteforce():
    for seed in range(15):
        n = 6 + seed % 11
        w, e = random_graph(n, 0.3 + 0.03 * (seed % 5), seed)
        xb = exact_bruteforce(w, e)
        xm, _ = exact(w, e)
        assert is_feasible(xm, e)
        assert abs(np.dot(xb, w) - np.dot(xm, w)) < 1e-9


def test_branch_and_bound_matches_bruteforce():
    for seed in range(8):
        w, e = random_graph(14, 0.35, 100 + seed)
        adj = [set() for _ in range(14)]
        for i, j in e:
            adj[i].add(j); adj[j].add(i)
        sel = _exact_bnb(np.asarray(w), adj)
        assert abs(sum(w[i] for i in sel) - np.dot(exact_bruteforce(w, e), w)) < 1e-9


def test_greedy_and_sa_feasible():
    for seed in range(10):
        w, e = random_graph(16, 0.4, seed)
        assert is_feasible(greedy(w, e), e)
        X = simulated_annealing(w, e, restarts=64, seed=seed, sweeps=60)
        assert X.shape == (64, 16)
        assert all(is_feasible(x, e) for x in X)
