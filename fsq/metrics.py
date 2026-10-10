"""Metrics used in every results table."""

from __future__ import annotations

import numpy as np


def approx_ratio(value: float, optimum: float) -> float:
    """value(schedule) / value(exact optimum)."""
    return float(value / optimum) if optimum > 0 else 1.0


def feasible_mask(X: np.ndarray, edges: list) -> np.ndarray:
    """Boolean mask: rows of X that violate no conflict."""
    X = np.asarray(X)
    if len(X) == 0:
        return np.zeros(0, dtype=bool)
    if not len(edges):
        return np.ones(len(X), dtype=bool)
    E = np.asarray(edges)
    return ~((X[:, E[:, 0]] & X[:, E[:, 1]]).astype(bool).any(axis=1))


def feasible_rate(X: np.ndarray, edges: list) -> float:
    """Share of samples with no violated conflict."""
    m = feasible_mask(X, edges)
    return float(m.mean()) if len(m) else 0.0


def mean_pairwise_hamming(X: np.ndarray, max_pairs: int = 4000, seed: int = 0) -> float:
    """Mean Hamming distance over pairs of distinct rows (sampled if there are many pairs)."""
    X = np.asarray(X)
    k = len(X)
    if k < 2:
        return 0.0
    total = k * (k - 1) // 2
    if total <= max_pairs:
        i, j = np.triu_indices(k, 1)
    else:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, k, size=max_pairs * 2)
        j = rng.integers(0, k, size=max_pairs * 2)
        keep = i != j
        i, j = i[keep][:max_pairs], j[keep][:max_pairs]
    return float((X[i] != X[j]).sum(axis=1).mean())


def good_schedules(X: np.ndarray, w: list[float], edges: list, optimum: float,
                   threshold: float = 0.95) -> np.ndarray:
    """Distinct feasible schedules with value >= threshold * optimum."""
    X = np.asarray(X)
    if len(X) == 0:
        return X
    F = X[feasible_mask(X, edges)]
    if len(F) == 0:
        return F
    vals = F @ np.asarray(w, dtype=float)
    G = F[vals >= threshold * optimum - 1e-12]
    if len(G) == 0:
        return G
    return np.unique(G, axis=0)


def sample_stats(X: np.ndarray, w: list[float], edges: list, optimum: float,
                 threshold: float = 0.95, seed: int = 0) -> dict:
    """n_good, diversity and feasible_rate for one batch of samples (§6.2)."""
    G = good_schedules(X, w, edges, optimum, threshold)
    F = np.asarray(X)[feasible_mask(X, edges)]
    best = float((F @ np.asarray(w, dtype=float)).max()) if len(F) else 0.0
    return {
        "samples": int(len(X)),
        "feasible_rate": feasible_rate(X, edges),
        "n_good": int(len(G)),
        "diversity": mean_pairwise_hamming(G, seed=seed),
        "best_ratio": approx_ratio(best, optimum),
    }


def two_qubit_count(circuit) -> int:
    """Number of two-qubit gates in a (transpiled) circuit."""
    return sum(1 for inst in circuit.data
               if inst.operation.num_qubits == 2 and inst.operation.name != "barrier")


def est_success(n_2q: int, e2: float = 0.005) -> float:
    """Simple model: (1 - e2)^(n_2q). Not a hardware measurement."""
    return float((1 - e2) ** n_2q)
