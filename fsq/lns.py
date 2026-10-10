"""Conditioned large-neighbourhood search (LNS) over the clusters.

v1 solved every cluster independently, then repaired the conflicts the partition cut
(233 of 439 in the default run). Here each cluster is solved *given* the rest of the
schedule instead:

  1. incumbent = a feasible schedule (greedy);
  2. for cluster C, fix every candidate outside C to the incumbent and drop from C every
     candidate that conflicts with a selected outside candidate ("conditioned subproblem":
     smaller, and no cut conflict can be violated);
  3. a sub-solver proposes schedules for the conditioned subproblem (QAOA samples, or a
     control: exact, SA, uniform random feasible);
  4. the best proposal that is still feasible against the *current* schedule replaces C
     if it raises the total value;
  5. repeat for K sweeps.

The result is never worse than the incumbent and always feasible (tested).
"""

from __future__ import annotations

import numpy as np

from .baselines import adjacency, exact_bruteforce, simulated_annealing
from .formulate import all_bitstrings
from .metrics import feasible_mask


# --------------------------------------------------------------------------- subproblem

def conditioned_free(nodes: list[int], x: np.ndarray, adj: list[list[int]]) -> list[int]:
    """Candidates of cluster `nodes` that do not conflict with any selected candidate
    outside the cluster (under schedule x)."""
    inside = set(nodes)
    return [u for u in nodes if not any(x[v] and v not in inside for v in adj[u])]


def local_problem(free: list[int], w, edges_adj: list[list[int]]):
    """Re-indexed weights and edges for the free candidates."""
    idx = {v: k for k, v in enumerate(free)}
    wl = [float(w[v]) for v in free]
    el = sorted({(min(idx[u], idx[v]), max(idx[u], idx[v]))
                 for u in free for v in edges_adj[u] if v in idx})
    return wl, [list(e) for e in el]


def release_neighbourhoods(x: np.ndarray, adj: list[list[int]], max_free: int,
                           seed: int, randomized: bool = True) -> list[list[int]]:
    """Neighbourhoods for one sweep, built around the current schedule (not a fixed
    partition, which cannot move a district's image from one cluster to another).

    For each selected candidate s (in random order): release s, then keep releasing the
    selected candidate that unlocks the most still-blocked candidates, while the free set
    (released candidates + every unselected candidate blocked only by released ones) has at
    most `max_free` members. Each neighbourhood is a list of free candidates; re-optimising
    it can swap a released image for other images it was blocking.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    selected = [v for v in range(n) if x[v]]
    blockers = {u: {v for v in adj[u] if x[v]} for u in range(n) if not x[u]}
    out, seen = [], set()
    for s in rng.permutation(selected):
        released = {int(s)}
        while True:
            free = set(released) | {u for u, b in blockers.items() if b and b <= released}
            # next release: another blocker of a candidate that `released` already partly blocks
            cand = {v for u, b in blockers.items() if b & released for v in b} - released
            fits = []
            for v in sorted(cand):
                r2 = released | {v}
                f2 = set(r2) | {u for u, b in blockers.items() if b and b <= r2}
                if len(f2) <= max_free:
                    fits.append((len(f2), v))
            if not fits:
                break
            if randomized:
                released.add(fits[int(rng.integers(len(fits)))][1])
            else:
                released.add(max(fits)[1])
        free = sorted(set(released) | {u for u, b in blockers.items() if b and b <= released})
        if len(free) > max_free:          # a single release can already be too big: trim
            free = sorted(released) + [u for u in free if u not in released][: max_free - len(released)]
        key = tuple(free)
        if key not in seen and len(free) > 1:
            seen.add(key)
            out.append(free)
    return out


# --------------------------------------------------------------------------- sub-solvers
# Each returns candidate local bitstrings (rows of 0/1). "Exact" returns one row.

def propose_exact(wl, el, budget, seed, **_):
    return exact_bruteforce(wl, el)[None, :]


def propose_sa(wl, el, budget, seed, **_):
    m = max(wl) or 1.0
    return simulated_annealing([v / m for v in wl], el, restarts=budget, seed=seed, sweeps=100, P=1.2)


def propose_random(wl, el, budget, seed, **_):
    """Uniform random FEASIBLE schedules (all feasible sets enumerated, then sampled)."""
    n = len(wl)
    X = all_bitstrings(n)
    F = X[feasible_mask(X, el)]
    rng = np.random.default_rng(seed)
    return F[rng.integers(0, len(F), size=budget)]


def qaoa_distribution(wl, el, p: int = 1, alpha: float = 0.2, eps: float = 0.25,
                      penalty_mult: float = 1.2, grid: int = 8, maxiter: int = 60):
    """Warm-start CVaR-QAOA for one subproblem: tuned angles and the exact output
    distribution (offline / local simulation only)."""
    from .qaoa import exact_probabilities, lp_relaxation, optimize, scaled_cluster_qubo, warm_thetas
    ws, P = scaled_cluster_qubo(wl, penalty_mult)
    thetas = warm_thetas(lp_relaxation(ws, el), eps)
    prm = optimize(ws, el, P, p, alpha, thetas=thetas, maxiter=maxiter, grid=grid)
    probs = exact_probabilities(ws, el, P, prm["gammas"], prm["betas"], thetas)
    return probs, prm, thetas


def propose_qaoa(wl, el, budget, seed, **_):
    from .qaoa import sample_from_probs
    if len(wl) == 1:
        return np.ones((1, 1), dtype=np.int8)
    probs, _, _ = qaoa_distribution(wl, el)
    return sample_from_probs(probs, len(wl), budget, seed)


SOLVERS = {"exact": propose_exact, "qaoa": propose_qaoa, "sa": propose_sa, "random": propose_random}


# --------------------------------------------------------------------------- acceptance

def best_compatible(nodes: list[int], free: list[int], samples: np.ndarray, x: np.ndarray,
                    w, adj) -> np.ndarray | None:
    """Best proposal (by value) that is feasible inside and compatible with the current
    schedule outside the cluster. Returns the new 0/1 assignment for `nodes`."""
    inside = set(nodes)
    best, best_val = None, -1.0
    seen = set()
    for row in samples:
        key = row.tobytes()
        if key in seen:
            continue
        seen.add(key)
        chosen = [free[k] for k, b in enumerate(row) if b]
        cs = set(chosen)
        if any(v in cs for u in chosen for v in adj[u]):                     # infeasible inside
            continue
        if any(x[v] and v not in inside for u in chosen for v in adj[u]):    # clashes outside
            continue
        val = sum(w[u] for u in chosen)
        if val > best_val:
            best, best_val = cs, val
    if best is None:
        return None
    return np.array([1 if u in best else 0 for u in nodes], dtype=np.int8)


def apply_move(x: np.ndarray, nodes: list[int], new_local: np.ndarray, w) -> bool:
    """Replace cluster `nodes` in x by new_local if that strictly raises the value."""
    old = sum(w[u] for u in nodes if x[u])
    new = sum(w[u] for u, b in zip(nodes, new_local) if b)
    if new > old + 1e-12:
        for u, b in zip(nodes, new_local):
            x[u] = b
        return True
    return False


# --------------------------------------------------------------------------- LNS driver

def run_lns(w, edges, clusters, x0, propose, sweeps: int, budget: int,
            seed: int, max_free: int = 12, randomized: bool = True) -> dict:
    """Sequential LNS: each neighbourhood is conditioned on the current schedule.

    clusters: a fixed list of node lists, or "release" for neighbourhoods rebuilt around
    the current schedule every sweep (release_neighbourhoods, at most max_free qubits).
    propose(wl, el, budget, seed, nodes=...) -> local proposals. Returns the final
    schedule, the value after each sweep and move statistics.
    """
    adj = adjacency(len(w), edges)
    x = np.array(x0, dtype=np.int8).copy()
    values = [float(np.dot(x, w))]
    moves = 0
    calls = 0
    for s in range(sweeps):
        hoods = (release_neighbourhoods(x, adj, max_free, seed * 7919 + s, randomized)
                 if isinstance(clusters, str) else clusters)
        for k, nodes in enumerate(hoods):
            free = conditioned_free(nodes, x, adj)
            if not free:
                continue
            wl, el = local_problem(free, w, adj)
            props = propose(wl, el, budget, seed * 100003 + s * 1009 + k, nodes=nodes, free=free)
            calls += 1
            new_local = best_compatible(nodes, free, props, x, w, adj)
            if new_local is not None and apply_move(x, nodes, new_local, w):
                moves += 1
        values.append(float(np.dot(x, w)))
    return {"x": x, "values": values, "moves": moves, "calls": calls}
