import numpy as np

from fsq.baselines import adjacency, exact_bruteforce, greedy
from fsq.formulate import is_feasible
from fsq.lns import (SOLVERS, best_compatible, conditioned_free, local_problem, release_neighbourhoods,
                     run_lns)
from tests.test_baselines import random_graph


def test_conditioned_subproblem_excludes_conflicting_candidates():
    # path 0-1-2-3, with 3 selected outside cluster {0,1,2}: 2 must be excluded
    adj = adjacency(4, [[0, 1], [1, 2], [2, 3]])
    x = np.array([0, 0, 0, 1], dtype=np.int8)
    assert conditioned_free([0, 1, 2], x, adj) == [0, 1]


def test_lns_never_worse_and_always_feasible():
    for seed in range(8):
        w, e = random_graph(40, 0.12, seed)
        x0 = greedy(w, e)
        for solver in ("exact", "random", "sa"):
            r = run_lns(w, e, "release", x0, SOLVERS[solver], sweeps=3, budget=16, seed=seed, max_free=10)
            assert is_feasible(r["x"], e)
            assert all(b >= a - 1e-12 for a, b in zip(r["values"], r["values"][1:]))
            assert r["values"][-1] >= np.dot(x0, w) - 1e-12


def test_lns_deterministic_with_fixed_seed():
    w, e = random_graph(30, 0.15, 3)
    x0 = greedy(w, e)
    a = run_lns(w, e, "release", x0, SOLVERS["random"], sweeps=3, budget=8, seed=5, max_free=8)
    b = run_lns(w, e, "release", x0, SOLVERS["random"], sweeps=3, budget=8, seed=5, max_free=8)
    assert a["values"] == b["values"] and (a["x"] == b["x"]).all()


def test_release_neighbourhoods_respect_budget():
    w, e = random_graph(50, 0.1, 1)
    x = greedy(w, e)
    adj = adjacency(50, e)
    for hood in release_neighbourhoods(x, adj, 9, seed=2):
        assert len(hood) <= 9


def test_best_compatible_rejects_outside_clash():
    # nodes 0,1 in the neighbourhood; node 2 outside is selected and conflicts with 1
    w = [1.0, 2.0, 1.0]
    adj = adjacency(3, [[1, 2]])
    x = np.array([1, 0, 1], dtype=np.int8)
    samples = np.array([[0, 1], [1, 0]], dtype=np.int8)   # local order = free [0, 1]
    out = best_compatible([0, 1], [0, 1], samples, x, w, adj)
    assert out.tolist() == [1, 0]


def test_exact_lns_reaches_optimum_on_small_graph():
    w, e = random_graph(14, 0.3, 4)
    opt = np.dot(exact_bruteforce(w, e), w)
    r = run_lns(w, e, "release", greedy(w, e), SOLVERS["exact"], sweeps=5, budget=1, seed=0, max_free=14)
    assert abs(r["values"][-1] - opt) < 1e-9
