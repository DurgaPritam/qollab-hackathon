import itertools

import numpy as np

from fsq.formulate import (all_bitstrings, conflict_edges, energies_batch, ising_operator,
                           ising_terms, qubo_energy, qubo_matrix)

W = [0.5, 1.0, 0.3, 0.8, 0.6]
EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 3]]
P = 1.2


def test_qubo_energy_by_hand():
    assert qubo_energy([1, 0, 1, 0, 1], W, EDGES, P) == -(0.5 + 0.3 + 0.6)
    # x1 and x3 conflict: -(1.0 + 0.8) + 1.2
    assert abs(qubo_energy([0, 1, 0, 1, 0], W, EDGES, P) - (-1.8 + 1.2)) < 1e-12


def test_matrix_and_batch_match_scalar():
    Q = qubo_matrix(W, EDGES, P)
    X = all_bitstrings(5)
    batch = energies_batch(X, W, EDGES, P)
    for x, e in zip(X, batch):
        assert abs(x @ Q @ x - qubo_energy(x, W, EDGES, P)) < 1e-12
        assert abs(e - qubo_energy(x, W, EDGES, P)) < 1e-12


def test_ising_reproduces_qubo_for_all_bitstrings():
    h, J, off = ising_terms(W, EDGES, P)
    op, off2 = ising_operator(W, EDGES, P)
    diag = np.real(np.diag(op.to_matrix()))
    for k, x in enumerate(all_bitstrings(5)):
        z = 1 - 2 * x
        e = off + h @ z + sum(v * z[a] * z[b] for (a, b), v in J.items())
        assert abs(e - qubo_energy(x, W, EDGES, P)) < 1e-9
        # SparsePauliOp diagonal is indexed with qubit 0 = least significant bit
        assert abs(diag[k] + off2 - qubo_energy(x, W, EDGES, P)) < 1e-9


def test_conflict_rules():
    cfg = {"dwell_s": 10, "slew_s": 60}
    c = [
        {"target_id": "A", "satellite": "S1", "t": 0},
        {"target_id": "B", "satellite": "S1", "t": 50},    # same sat, too close -> conflict
        {"target_id": "C", "satellite": "S1", "t": 200},   # same sat, far enough
        {"target_id": "A", "satellite": "S2", "t": 9000},  # same target -> conflict
    ]
    assert conflict_edges(c, cfg) == [[0, 1], [0, 3]]
