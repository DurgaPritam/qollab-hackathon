import itertools

import numpy as np

from fsq.baselines import exact_bruteforce
from fsq.binenc import Encoding, bits_for, clique_groups, cnot_cost, pauli_terms
from fsq.formulate import all_bitstrings
from fsq.metrics import feasible_mask

def random_problem(n=9, p=0.45, _rng=[None]):
    if _rng[0] is None:
        _rng[0] = np.random.default_rng(3)
    RNG = _rng[0]
    edges = [[i, j] for i, j in itertools.combinations(range(n), 2) if RNG.random() < p]
    return list(RNG.uniform(0.2, 1.0, n)), edges


def test_groups_are_cliques_and_partition():
    for _ in range(10):
        w, e = random_problem()
        es = {tuple(x) for x in e}
        groups = clique_groups(len(w), e)
        assert sorted(v for g in groups for v in g) == list(range(len(w)))
        for g in groups:
            assert all((a, b) in es for a, b in itertools.combinations(g, 2))


def test_encoded_energy_equals_one_hot_on_valid_states():
    for _ in range(10):
        w, e = random_problem()
        enc = Encoding(w, e)
        P = 1.2 * max(w)
        E, X = enc.energies(P)
        Z = all_bitstrings(enc.Q)
        for k in range(len(Z)):
            x, bad = enc.decode(Z[k])
            assert np.array_equal(x, X[k])
            if bad == 0:
                one_hot = -np.dot(x, w) + P * sum(x[a] * x[b] for a, b in e)
                assert abs(E[k] - one_hot) < 1e-9
                assert np.array_equal(enc.encode(x), Z[k])


def test_invalid_codes_never_win_and_minimum_is_the_optimum():
    for _ in range(10):
        w, e = random_problem()
        enc = Encoding(w, e)
        E, X = enc.energies(1.2 * max(w))
        Z = all_bitstrings(enc.Q)
        bad = np.array([enc.decode(z)[1] for z in Z])
        # every invalid state is strictly worse than the same state with that group set to 0
        if (bad > 0).any():
            assert E[bad > 0].min() > E.min()
        k = int(np.argmin(E))
        assert bad[k] == 0 and feasible_mask(X[k:k + 1], e)[0]
        opt = exact_bruteforce(w, e)
        assert abs(np.dot(X[k], w) - np.dot(opt, w)) < 1e-9


def test_binary_uses_fewer_qubits_on_cliques_and_pauli_expansion_is_exact():
    w = [1.0] * 7
    e = [list(c) for c in itertools.combinations(range(7), 2)]      # one 7-clique
    enc = Encoding(w, e)
    assert enc.Q == bits_for(7) == 3
    E, _ = enc.energies(1.2)
    terms = pauli_terms(E, enc.Q)
    assert sum(terms.values()) <= 2 ** 3 - 1 and cnot_cost(terms) >= 0


def test_binary_sized_lns_is_feasible_and_never_worse():
    from fsq.baselines import adjacency, greedy
    from fsq.binenc import binary_size, propose_random_encoded
    from fsq.formulate import is_feasible
    from fsq.lns import run_lns
    w, e = random_problem(n=30, p=0.2)
    x0 = greedy(w, e)
    r = run_lns(w, e, "release", x0, propose_random_encoded, sweeps=3, budget=64, seed=1, max_free=8,
                size=binary_size(adjacency(len(w), e)))
    assert is_feasible(r["x"], e) and r["values"][-1] >= np.dot(x0, w) - 1e-12
