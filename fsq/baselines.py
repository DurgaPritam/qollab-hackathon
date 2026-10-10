"""Strong classical baselines: greedy, simulated annealing, exact optimum."""

from __future__ import annotations

import time

import numpy as np

from .formulate import all_bitstrings, energies_batch, penalty
from .config import CONFIG


def adjacency(n: int, edges: list) -> list[list[int]]:
    """Neighbour lists."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    return adj


# --------------------------------------------------------------------------- greedy

def greedy(w: list[float], edges: list) -> np.ndarray:
    """Greedy by value density w_i / (1 + degree_i); add a candidate if it has no conflict."""
    n = len(w)
    adj = adjacency(n, edges)
    order = sorted(range(n), key=lambda i: (-w[i] / (1 + len(adj[i])), i))
    x = np.zeros(n, dtype=np.int8)
    blocked = np.zeros(n, dtype=bool)
    for i in order:
        if not blocked[i]:
            x[i] = 1
            blocked[i] = True
            for j in adj[i]:
                blocked[j] = True
    return x


# --------------------------------------------------------------------------- SA

def simulated_annealing(w: list[float], edges: list, restarts: int, seed: int,
                        sweeps: int = 100, P: float | None = None,
                        t_start: float | None = None, t_end: float = 1e-3) -> np.ndarray:
    """Single-bit-flip SA on the QUBO with geometric cooling.

    Runs `restarts` independent chains in parallel (vectorised) and returns every final
    state (shape restarts x n), not only the best, so the sample budget equals `restarts`.
    Each chain starts from a uniformly random bitstring; one sweep visits every bit once
    in a fresh random order.
    """
    rng = np.random.default_rng(seed)
    n = len(w)
    if n == 0:
        return np.zeros((restarts, 0), dtype=np.int8)
    w = np.asarray(w, dtype=float)
    P = penalty(w, CONFIG["penalty_mult"]) if P is None else P
    adj = [np.asarray(a, dtype=int) for a in adjacency(n, edges)]
    X = rng.integers(0, 2, size=(restarts, n)).astype(np.int8)
    # local field f_k = P * (number of selected neighbours of k)
    F = np.zeros((restarts, n))
    for k in range(n):
        if len(adj[k]):
            F[:, k] = P * X[:, adj[k]].sum(axis=1)
    t_start = P if t_start is None else t_start
    temps = t_start * (t_end / t_start) ** (np.arange(sweeps) / max(sweeps - 1, 1))
    rows = np.arange(restarts)
    for T in temps:
        for k in rng.permutation(n):
            xk = X[:, k]
            s = 1 - 2 * xk.astype(float)        # +1 if turning on, -1 if turning off
            delta = s * (-w[k] + F[:, k])
            accept = (delta <= 0) | (rng.random(restarts) < np.exp(-np.maximum(delta, 0) / T))
            if not accept.any():
                continue
            idx = rows[accept]
            X[idx, k] = 1 - X[idx, k]
            if len(adj[k]):
                F[np.ix_(idx, adj[k])] += (s[idx] * P)[:, None]
    return X


# --------------------------------------------------------------------------- exact

def _components(n: int, edges: list) -> list[list[int]]:
    adj = adjacency(n, edges)
    seen = [False] * n
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def exact_bruteforce(w: list[float], edges: list) -> np.ndarray:
    """Enumerate all 2^n bitstrings (n <= 20). Used for tests and as a fallback."""
    n = len(w)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    if n > 20:
        raise ValueError("brute force limited to 20 variables")
    X = all_bitstrings(n)
    vals = X @ np.asarray(w, dtype=float)
    if len(edges):
        E = np.asarray(edges)
        feas = ~((X[:, E[:, 0]] & X[:, E[:, 1]]).astype(bool).any(axis=1))
        vals = np.where(feas, vals, -np.inf)
    return X[int(np.argmax(vals))].copy()


def _exact_bnb(w: np.ndarray, adj: list[set]) -> list[int]:
    """Branch and bound for MWIS on one component (pure Python; no SciPy needed)."""
    best = {"val": -1.0, "set": []}

    def rec(cand: list[int], chosen: list[int], val: float):
        if val + sum(w[c] for c in cand) <= best["val"] + 1e-12:
            return
        if not cand:
            best["val"], best["set"] = val, chosen[:]
            return
        v = max(cand, key=lambda c: (len(adj[c] & set(cand)), w[c]))
        # branch 1: take v
        rest = [c for c in cand if c != v and c not in adj[v]]
        rec(rest, chosen + [v], val + w[v])
        # branch 2: skip v
        rec([c for c in cand if c != v], chosen, val)

    rec(sorted(range(len(w)), key=lambda i: -w[i]), [], 0.0)
    return best["set"]


def exact(w: list[float], edges: list) -> tuple[np.ndarray, str]:
    """Exact MWIS. Uses scipy.optimize.milp (HiGHS) when available, else per-component
    brute force / branch and bound. Returns (x, method_name)."""
    n = len(w)
    try:
        from scipy.optimize import LinearConstraint, Bounds, milp
        if n == 0:
            return np.zeros(0, dtype=np.int8), "milp"
        c = -np.asarray(w, dtype=float)
        cons = []
        if len(edges):
            A = np.zeros((len(edges), n))
            for r, (i, j) in enumerate(edges):
                A[r, i] = A[r, j] = 1
            cons.append(LinearConstraint(A, -np.inf, 1))
        res = milp(c, constraints=cons, integrality=np.ones(n), bounds=Bounds(0, 1))
        if res.status == 0:
            return np.round(res.x).astype(np.int8), "milp"
    except Exception:
        pass
    x = np.zeros(n, dtype=np.int8)
    wa = np.asarray(w, dtype=float)
    for comp in _components(n, edges):
        idx = {v: k for k, v in enumerate(comp)}
        sub_e = [[idx[i], idx[j]] for i, j in edges if i in idx and j in idx]
        if len(comp) <= 18:
            xs = exact_bruteforce(list(wa[comp]), sub_e)
            for k, v in enumerate(comp):
                x[v] = xs[k]
        else:
            sadj = [set() for _ in comp]
            for i, j in sub_e:
                sadj[i].add(j)
                sadj[j].add(i)
            for k in _exact_bnb(wa[comp], sadj):
                x[comp[k]] = 1
    return x, "branch-and-bound"


def timed(fn, *args, **kwargs):
    """Run fn and return (result, seconds)."""
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t
