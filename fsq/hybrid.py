"""The hybrid pipeline: decompose -> warm-start QAOA per cluster -> stitch -> repair.

Shared by the offline experiments and the live Qollab run, so both run the same code.
"""

from __future__ import annotations

import numpy as np

from .config import CONFIG
from .decompose import cluster_subproblem, cut_edges, partition, repair, stitch
from .formulate import is_feasible
from .metrics import feasible_mask
from .qaoa import (lp_relaxation, optimize, qaoa_circuit, scaled_cluster_qubo, warm_thetas)


def prepare_clusters(inst: dict, cfg: dict = CONFIG, max_cluster: int | None = None,
                     seed: int = 0) -> list[dict]:
    """Split the instance into clusters and compute each cluster's scaled QUBO and
    warm-start angles (LP relaxation with clique constraints, regularised by eps)."""
    w, edges = inst["w"], inst["edges"]
    mc = max_cluster or cfg["max_cluster"]
    out = []
    for nodes in partition(len(w), edges, mc, seed):
        wl, el = cluster_subproblem(nodes, w, edges)
        ws, P = scaled_cluster_qubo(wl, cfg["penalty_mult"])
        c = lp_relaxation(ws, el)
        out.append({"nodes": nodes, "w": ws, "edges": el, "P": P,
                    "lp": [round(float(v), 6) for v in c],
                    "thetas": [float(t) for t in warm_thetas(c, cfg["eps"])]})
    return out


def tune_cluster(cl: dict, p: int, warm: bool, cfg: dict = CONFIG) -> dict:
    """CVaR-optimised QAOA angles for one cluster (offline / local only)."""
    if len(cl["nodes"]) == 1:
        return {"gammas": [0.0] * p, "betas": [0.0] * p, "cvar": None, "evals": 0}
    return optimize(cl["w"], cl["edges"], cl["P"], p, cfg["alpha"],
                    thetas=cl["thetas"] if warm else None)


def cluster_circuit(cl: dict, params: dict, warm: bool = True, measure: bool = True):
    """Measured QAOA circuit for one cluster with fixed parameters."""
    return qaoa_circuit(cl["w"], cl["edges"], cl["P"], params["gammas"], params["betas"],
                        thetas=cl["thetas"] if warm else None, measure=measure)


def best_sample(cl: dict, X: np.ndarray) -> tuple[np.ndarray, bool]:
    """Highest-value FEASIBLE sample of a cluster (no violated conflict).

    If no sample is feasible, return the most frequent sample (repair fixes it later).
    Returns (bitstring, found_feasible).
    """
    X = np.asarray(X)
    feas = feasible_mask(X, cl["edges"])
    if feas.any():
        F = X[feas]
        return F[int(np.argmax(F @ np.asarray(cl["w"], dtype=float)))], True
    uniq, cnt = np.unique(X, axis=0, return_counts=True)
    return uniq[int(np.argmax(cnt))], False


def assemble(inst: dict, clusters: list[dict], cluster_samples: list[np.ndarray]) -> dict:
    """Pick each cluster's best feasible sample, stitch, repair, and report what happened."""
    w, edges = inst["w"], inst["edges"]
    picks = [best_sample(cl, X) for cl, X in zip(clusters, cluster_samples)]
    raw = stitch([cl["nodes"] for cl in clusters], [p[0] for p in picks], len(w))
    log: list = []
    x = repair(raw, w, edges, log)
    feas_rates = [float(feasible_mask(X, cl["edges"]).mean()) for cl, X in zip(clusters, cluster_samples)]
    return {"x": x, "raw_feasible": bool(is_feasible(raw, edges)), "repair_log": log,
            "value": float(np.dot(x, w)), "cluster_feasible_rates": feas_rates,
            "clusters_with_feasible": int(sum(p[1] for p in picks)),
            "cut_conflicts": len(cut_edges([cl["nodes"] for cl in clusters], edges))}
