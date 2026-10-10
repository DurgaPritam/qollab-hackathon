import numpy as np

from fsq.formulate import all_bitstrings, energies_batch
from fsq.qaoa import (counts_to_samples, cvar, exact_probabilities, lp_relaxation, optimize,
                      qaoa_circuit, qaoa_statevector, warm_thetas)

W = [1.0, 0.6, 0.9, 0.5]      # path 0-1-2-3; optimum {0, 2}
E = [[0, 1], [1, 2], [2, 3]]
P = 1.2


def _qiskit_probs(qc):
    from qiskit.quantum_info import Statevector
    return Statevector(qc).probabilities()


def test_numpy_sim_matches_qiskit():
    th = warm_thetas(np.array([0.9, 0.1, 0.8, 0.3]), 0.25)
    for thetas in (None, th):
        qc = qaoa_circuit(W, E, P, [0.7, 0.3], [0.4, 0.2], thetas, measure=False)
        En = energies_batch(all_bitstrings(4), W, E, P)
        ours = np.abs(qaoa_statevector(En, 4, [0.7, 0.3], [0.4, 0.2], thetas)) ** 2
        assert np.allclose(ours, _qiskit_probs(qc), atol=1e-10)


def test_warm_start_marginals():
    c = np.array([0.9, 0.1, 0.6, 0.3])
    th = warm_thetas(c, 0.25)
    qc = qaoa_circuit(W, E, P, [], [], th, measure=False)
    probs = _qiskit_probs(qc)
    X = all_bitstrings(4)
    marg = probs @ X
    assert np.allclose(marg, np.clip(c, 0.25, 0.75), atol=1e-10)


def test_tuned_qaoa_mode_is_optimal():
    th = warm_thetas(lp_relaxation(W, E), 0.25)
    # warm start at p=1 and cold start at p=2 (cold p=1 is not enough on this graph:
    # CVaR with alpha=0.2 plateaus before the optimum becomes the mode)
    for thetas, p in ((th, 1), (None, 2)):
        prm = optimize(W, E, P, p=p, alpha=0.2, thetas=thetas, maxiter=150)
        probs = exact_probabilities(W, E, P, prm["gammas"], prm["betas"], thetas)
        best = all_bitstrings(4)[int(np.argmax(probs))]
        assert list(best) == [1, 0, 1, 0]


def test_cvar_and_counts():
    e = np.array([-2.0, -1.0, 0.0])
    p = np.array([0.1, 0.3, 0.6])
    assert abs(cvar(e, p, 0.2) - (-1.5)) < 1e-12
    s = counts_to_samples({"0101": 2, "0x1": 1}, 4)
    assert s.tolist() == [[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 0, 0]]
