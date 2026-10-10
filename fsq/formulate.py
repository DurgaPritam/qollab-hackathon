"""Candidates, values, conflict graph, QUBO and Ising forms.

A candidate i = (target, satellite pass, look side). Selecting a set of candidates with no
conflicting pair is a maximum-weight independent set (MWIS) problem:

    minimize  -sum_i w_i x_i + P * sum_{(i,j) in E} x_i x_j ,   x in {0,1}^n

Bit convention used everywhere: x_i = 1 means "image candidate i". In Qiskit bitstrings
qubit 0 is the rightmost character, and measured bit 1 corresponds to x_i = 1.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .config import CONFIG


# --------------------------------------------------------------------------- values

def urgency_decay(t_hours: float, tau_h: float) -> float:
    """exp(-t/tau): images taken sooner after the event start are worth more."""
    return math.exp(-max(t_hours, 0.0) / tau_h)


def image_quality(incidence_deg: float, best: float, width: float) -> float:
    """Gaussian bump peaking at `best` degrees; ~0.2 at the edges of a 20-45 deg range."""
    return math.exp(-((incidence_deg - best) / width) ** 2)


def candidate_value(priority: float, population: float, t_hours: float,
                    incidence_deg: float, cfg: dict = CONFIG) -> float:
    """w_i = priority * population * urgency_decay(t) * image_quality(incidence)."""
    return (priority * population * urgency_decay(t_hours, cfg["tau_h"])
            * image_quality(incidence_deg, cfg["best_inc_deg"], cfg["quality_width_deg"]))


# --------------------------------------------------------------------------- instance

def build_instance(targets: list[dict], windows: list[dict], t0: float,
                   instance_id: str = "real", cfg: dict = CONFIG) -> dict:
    """Build a JSON-serialisable scheduling instance.

    targets: dicts with target_id, name, lat, lon, population, priority.
    windows: dicts with satellite, target_id, t_broadside (unix s), best_incidence_deg,
             look_side, pass_id.
    t0: event start (unix seconds); urgency decays from here.
    """
    tmap = {t["target_id"]: t for t in targets}
    cands = []
    for wdw in sorted(windows, key=lambda w: (w["t_broadside"], w["satellite"], w["target_id"])):
        tg = tmap.get(wdw["target_id"])
        if tg is None:
            continue
        t_h = (wdw["t_broadside"] - t0) / 3600.0
        raw = candidate_value(float(tg["priority"]), float(tg["population"]), t_h,
                              float(wdw["best_incidence_deg"]), cfg)
        cands.append({
            "id": len(cands),
            "target_id": wdw["target_id"],
            "satellite": wdw["satellite"],
            "pass_id": wdw.get("pass_id", ""),
            "t": float(wdw["t_broadside"]),
            "t_hours": round(t_h, 4),
            "incidence": float(wdw["best_incidence_deg"]),
            "look_side": wdw.get("look_side", ""),
            "value_raw": raw,
        })
    wmax = max((c["value_raw"] for c in cands), default=1.0) or 1.0
    w = [c["value_raw"] / wmax for c in cands]
    edges = conflict_edges(cands, cfg)
    return {"id": instance_id, "t0": t0, "candidates": cands, "w": w, "edges": edges,
            "config": {k: cfg[k] for k in ("dwell_s", "slew_s", "tau_h", "penalty_mult")}}


def conflict_edges(cands: list[dict], cfg: dict = CONFIG) -> list[list[int]]:
    """(i, j) conflict if same target, or same satellite with imaging slots closer than slew.

    Each image occupies [t - dwell/2, t + dwell/2]; the satellite then needs `slew_s`
    seconds before the next slot may start. So two slots on one satellite conflict when
    |t_i - t_j| < dwell + slew.
    """
    gap = cfg["dwell_s"] + cfg["slew_s"]
    edges = []
    n = len(cands)
    for i in range(n):
        ci = cands[i]
        for j in range(i + 1, n):
            cj = cands[j]
            if ci["target_id"] == cj["target_id"]:
                edges.append([i, j])
            elif ci["satellite"] == cj["satellite"] and abs(ci["t"] - cj["t"]) < gap:
                edges.append([i, j])
    return edges


def graph_density(n: int, edges: list) -> float:
    """|E| / (n choose 2)."""
    return 0.0 if n < 2 else len(edges) / (n * (n - 1) / 2)


# --------------------------------------------------------------------------- QUBO

def penalty(w: Iterable[float], mult: float = CONFIG["penalty_mult"]) -> float:
    """P = mult * max(w). With mult > 1, P > max(w_i, w_j) on every edge."""
    w = list(w)
    return mult * (max(w) if w else 1.0)


def qubo_matrix(w: list[float], edges: list, P: float) -> np.ndarray:
    """Upper-triangular Q with E(x) = x^T Q x (diagonal holds the linear terms)."""
    n = len(w)
    Q = np.zeros((n, n))
    Q[np.arange(n), np.arange(n)] = -np.asarray(w, dtype=float)
    for i, j in edges:
        a, b = (i, j) if i < j else (j, i)
        Q[a, b] += P
    return Q


def qubo_energy(x, w: list[float], edges: list, P: float) -> float:
    """QUBO energy of one 0/1 vector."""
    x = np.asarray(x, dtype=float)
    e = -float(np.dot(w, x))
    for i, j in edges:
        e += P * x[i] * x[j]
    return e


def energies_batch(X: np.ndarray, w: list[float], edges: list, P: float) -> np.ndarray:
    """QUBO energies for each row of a 0/1 matrix X (shape k x n)."""
    X = np.asarray(X, dtype=float)
    e = -X @ np.asarray(w, dtype=float)
    if len(edges):
        E = np.asarray(edges)
        e = e + P * (X[:, E[:, 0]] * X[:, E[:, 1]]).sum(axis=1)
    return e


def all_bitstrings(n: int) -> np.ndarray:
    """All 2^n 0/1 vectors; row k has x_i = bit i of k (qubit-0-rightmost convention)."""
    k = np.arange(2 ** n)
    return ((k[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)


def schedule_value(x, w: list[float]) -> float:
    """Sum of values of selected candidates (ignores feasibility)."""
    return float(np.dot(np.asarray(x, dtype=float), w))


def is_feasible(x, edges: list) -> bool:
    """True when no selected pair conflicts."""
    return not any(x[i] and x[j] for i, j in edges)


# --------------------------------------------------------------------------- Ising

def ising_terms(w: list[float], edges: list, P: float):
    """Map the QUBO to Ising form with x_i = (1 - z_i) / 2.

    Returns (h, J, offset) such that E(x) = offset + sum_i h_i z_i + sum_(ij) J_ij z_i z_j.
    """
    n = len(w)
    h = np.array([wi / 2.0 for wi in w], dtype=float)
    J: dict[tuple[int, int], float] = {}
    offset = -sum(w) / 2.0
    for i, j in edges:
        a, b = (i, j) if i < j else (j, i)
        h[a] -= P / 4.0
        h[b] -= P / 4.0
        J[(a, b)] = J.get((a, b), 0.0) + P / 4.0
        offset += P / 4.0
    return h, J, offset


def ising_operator(w: list[float], edges: list, P: float):
    """Ising Hamiltonian as a Qiskit SparsePauliOp plus the constant offset."""
    from qiskit.quantum_info import SparsePauliOp

    n = len(w)
    h, J, offset = ising_terms(w, edges, P)
    terms = []
    for i in range(n):
        if h[i] != 0:
            terms.append(("Z", [i], float(h[i])))
    for (a, b), v in J.items():
        terms.append(("ZZ", [a, b], float(v)))
    if not terms:
        terms = [("I", [0], 0.0)]
    op = SparsePauliOp.from_sparse_list(terms, num_qubits=n)
    return op, offset
