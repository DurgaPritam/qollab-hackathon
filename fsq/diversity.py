"""Fair diversity comparison: QAOA sampling vs simulated annealing at the same budget.

Two levels, both with an equal sample budget S:
  * cluster level: the same cluster QUBO; QAOA gives S shots, SA gives S independent
    restarts. This compares the two samplers directly.
  * pipeline level: the full schedule. Shot k of every cluster is stitched into one
    schedule and repaired (the hybrid pipeline); SA runs S restarts on the full QUBO.
"""

from __future__ import annotations

import numpy as np

from .baselines import exact, simulated_annealing
from .decompose import cluster_subproblem, repair, stitch
from .metrics import sample_stats
from .qaoa import exact_probabilities, sample_from_probs, scaled_cluster_qubo, warm_thetas, lp_relaxation


def cluster_level(w_local, edges_local, params: dict, shots: int, seed: int,
                  penalty_mult: float, eps: float, sa_sweeps: int = 100) -> dict:
    """QAOA vs SA on one cluster QUBO with `shots` samples each."""
    ws, P = scaled_cluster_qubo(w_local, penalty_mult)
    n = len(ws)
    xopt, _ = exact(ws, edges_local)
    opt = float(np.dot(xopt, ws))
    thetas = None
    if params.get("warm", True):
        thetas = warm_thetas(lp_relaxation(ws, edges_local), eps)
    probs = exact_probabilities(ws, edges_local, P, params["gammas"], params["betas"], thetas)
    Xq = sample_from_probs(probs, n, shots, seed)
    Xs = simulated_annealing(ws, edges_local, restarts=shots, seed=seed, sweeps=sa_sweeps, P=P)
    return {"n": n, "edges": len(edges_local),
            "qaoa": sample_stats(Xq, ws, edges_local, opt, seed=seed),
            "sa": sample_stats(Xs, ws, edges_local, opt, seed=seed)}


def pipeline_samples(w, edges, clusters, cluster_samples: list[np.ndarray]) -> np.ndarray:
    """Stitch shot k of every cluster into a full schedule, then repair each one."""
    n = len(w)
    shots = min(len(s) for s in cluster_samples)
    out = np.zeros((shots, n), dtype=np.int8)
    for k in range(shots):
        x = stitch(clusters, [s[k] for s in cluster_samples], n)
        out[k] = repair(x, w, edges)
    return out
