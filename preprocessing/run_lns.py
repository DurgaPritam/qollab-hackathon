"""Step 1 experiments: conditioned LNS with QAOA and its controls. Offline only.

    .venv/bin/python preprocessing/run_lns.py           # ~10 min on 8 cores
    .venv/bin/python preprocessing/run_lns.py --quick   # 3 instances

Part A (pipeline): from the greedy schedule, K sweeps of conditioned LNS over release
neighbourhoods of at most MAX_FREE qubits. Same instances (real + 20 perturbed), same
seeds and same sample budget for every sub-solver:
    exact (ceiling), QAOA (warm start, CVaR, p=1, angles tuned per subproblem on an exact
    statevector), SA (budget restarts x 100 sweeps), uniform random feasible schedules.
Part B (sampler quality): identical subproblems for every sampler; probability mass on the
subproblem optimum and the chance of hitting it within B shots.

Writes results/e_lns.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from fsq.baselines import adjacency, exact_bruteforce, greedy, simulated_annealing  # noqa: E402
from fsq.data_load import load_targets, load_windows, make_instance, perturb  # noqa: E402
from fsq.formulate import all_bitstrings  # noqa: E402
from fsq.lns import SOLVERS, local_problem, qaoa_distribution, release_neighbourhoods, run_lns  # noqa: E402
from fsq.metrics import feasible_mask  # noqa: E402

BUDGET = 1024
SWEEPS = 10
MAX_FREE = 12
LARGER_FREE = 16           # exact / SA / random only (QAOA tuning at 16 qubits is too slow offline)
BUDGETS_B = [8, 32, 128, 1024]
SMALL_BUDGET = 32         # shot budget where the sampler's distribution can matter


def instances(n_pert):
    T, W = load_targets(), load_windows()
    out = [(make_instance(T, W, instance_id="real"), 7)]
    out += [(perturb(T, W, s), s) for s in range(1, n_pert + 1)]
    return out


def job_a(args):
    inst, seed, solver, max_free, budget = args
    t = time.perf_counter()
    xg = greedy(inst["w"], inst["edges"])
    r = run_lns(inst["w"], inst["edges"], "release", xg, SOLVERS[solver], sweeps=SWEEPS, budget=budget,
                seed=seed, max_free=max_free)
    return {"instance": inst["id"], "solver": solver, "max_free": max_free, "budget": budget,
            "values": r["values"], "moves": r["moves"], "calls": r["calls"],
            "runtime_s": time.perf_counter() - t}


def subproblems_for_b(inst, seed, k_max):
    """Subproblems (>= 6 free candidates) met along the exact-solver LNS trajectory."""
    w, edges = inst["w"], inst["edges"]
    adj = adjacency(len(w), edges)
    x = greedy(w, edges)
    out = []
    for s in range(3):
        for free in release_neighbourhoods(x, adj, MAX_FREE, seed * 7919 + s):
            if len(free) >= 6:
                out.append(local_problem(free, w, adj))
        x = run_lns(w, edges, "release", x, SOLVERS["exact"], sweeps=1, budget=1, seed=seed + s,
                    max_free=MAX_FREE)["x"]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(out))[:k_max]
    return [out[i] for i in idx]


def job_b(args):
    wl, el, seed = args
    n = len(wl)
    X = all_bitstrings(n)
    feas = feasible_mask(X, el)
    vals = np.where(feas, X @ np.asarray(wl), -1.0)
    opt = vals.max()
    is_opt = vals >= opt - 1e-12
    probs, prm, _ = qaoa_distribution(wl, el)
    p_qaoa = float(probs[is_opt].sum())
    p_rand = float(is_opt.sum() / feas.sum())
    m = max(wl)
    Xs = simulated_annealing([v / m for v in wl], el, restarts=BUDGET, seed=seed, sweeps=100, P=1.2)
    sv = np.where(feasible_mask(Xs, el), Xs @ np.asarray(wl), -1.0)
    p_sa = float(np.mean(sv >= opt - 1e-12))
    return {"n": n, "edges": len(el), "n_feasible": int(feas.sum()), "n_opt": int(is_opt.sum()),
            "p_qaoa": p_qaoa, "p_random": p_rand, "p_sa": p_sa,
            "qaoa_feasible": float(probs[feas].sum())}


def ms(v):
    return [float(np.mean(v)), float(np.std(v))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    n_pert = 2 if args.quick else 20
    insts = instances(n_pert)
    exact = {r["instance"]: r["value"] for r in json.load(open("results/exact.json"))["instances"]}
    gre = {}
    for inst, _ in insts:
        gre[inst["id"]] = float(np.dot(greedy(inst["w"], inst["edges"]), inst["w"]))

    jobs = [(inst, seed, s, MAX_FREE, BUDGET) for inst, seed in insts for s in ("exact", "qaoa", "sa", "random")]
    jobs += [(inst, seed, s, LARGER_FREE, BUDGET) for inst, seed in insts for s in ("exact", "sa", "random")]
    jobs += [(inst, seed, s, MAX_FREE, SMALL_BUDGET) for inst, seed in insts for s in ("qaoa", "sa", "random")]
    with Pool(min(8, os.cpu_count() or 2)) as pool:
        res_a = []
        for r in pool.imap_unordered(job_a, jobs):
            res_a.append(r)
            print(f"  A {r['instance']:>14} {r['solver']:>6} q{r['max_free']} b{r['budget']} "
                  f"ratio {r['values'][-1] / exact[r['instance']]:.4f} moves {r['moves']} {r['runtime_s']:.0f}s", flush=True)
        sub = []
        for inst, seed in insts[: (2 if args.quick else 6)]:
            sub += [(wl, el, seed * 31 + k) for k, (wl, el) in enumerate(subproblems_for_b(inst, seed, 60))]
        res_b = pool.map(job_b, sub)

    # ---- aggregate part A
    table = {}
    for r in res_a:
        key = f"{r['solver']}_q{r['max_free']}" + ("" if r["budget"] == BUDGET else f"_b{r['budget']}")
        e = exact[r["instance"]]
        d = table.setdefault(key, {"solver": r["solver"], "max_free": r["max_free"], "budget": r["budget"],
                                   "ratios_by_sweep": [],
                                   "moves": [], "calls": [], "runtime_s": [], "instances": []})
        d["ratios_by_sweep"].append([v / e for v in r["values"]])
        d["moves"].append(r["moves"]); d["calls"].append(r["calls"]); d["runtime_s"].append(r["runtime_s"])
        d["instances"].append(r["instance"])
    summary = {}
    for key, d in table.items():
        a = np.array(d["ratios_by_sweep"])
        summary[key] = {"solver": d["solver"], "max_free": d["max_free"], "budget": d["budget"],
                        "final": ms(a[:, -1]), "worst": float(a[:, -1].min()),
                        "optimal": int(np.sum(a[:, -1] > 1 - 1e-9)), "by_sweep_mean": a.mean(axis=0).tolist(),
                        "moves": ms(d["moves"]), "calls": ms(d["calls"]), "runtime_s": ms(d["runtime_s"]),
                        "per_instance_final": dict(zip(d["instances"], a[:, -1].tolist()))}
    g = np.array([gre[i] / exact[i] for i in gre])
    summary["greedy"] = {"final": ms(g), "worst": float(g.min())}

    # ---- aggregate part B
    pb = {}
    for k in ("p_qaoa", "p_random", "p_sa"):
        p = np.array([r[k] for r in res_b])
        pb[k] = {"mean": float(p.mean()), "median": float(np.median(p)),
                 "hit_within": {str(B): float(np.mean(1 - (1 - p) ** B)) for B in BUDGETS_B}}
    q, rnd = np.array([r["p_qaoa"] for r in res_b]), np.array([r["p_random"] for r in res_b])
    pb["qaoa_over_random"] = {"mean_ratio": float(np.mean(q / rnd)), "median_ratio": float(np.median(q / rnd)),
                              "qaoa_higher": int(np.sum(q > rnd)), "n": len(res_b)}
    pb["sizes"] = ms([r["n"] for r in res_b])
    pb["qaoa_feasible_mass"] = ms([r["qaoa_feasible"] for r in res_b])

    out = {"meta": {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "python": platform.python_version(), "budget": BUDGET, "sweeps": SWEEPS,
                    "max_free": MAX_FREE, "larger_free": LARGER_FREE, "small_budget": SMALL_BUDGET, "neighbourhoods": "release, randomized",
                    "incumbent": "greedy", "n_instances": len(insts),
                    "qaoa": "warm start (clique LP, eps 0.25), CVaR alpha 0.2, p=1, grid 8x8 + COBYLA 60, exact statevector",
                    "sa_proposer": f"{BUDGET} restarts x 100 sweeps",
                    "random_proposer": "uniform over all feasible schedules of the subproblem"},
           "part_a": summary, "part_b": pb, "part_b_subproblems": res_b}
    json.dump(out, open("results/e_lns.json", "w"))
    print(f"\nPart A (ratio to exact after {SWEEPS} sweeps, {len(insts)} instances):")
    print(f"  greedy                {g.mean():.4f} ± {g.std():.4f}")
    for key in sorted(k for k in summary if k != "greedy"):
        s_ = summary[key]
        print(f"  {key:<20}  {s_['final'][0]:.4f} ± {s_['final'][1]:.4f}  worst {s_['worst']:.4f}  "
              f"optimal {s_['optimal']}/{len(insts)}  moves {s_['moves'][0]:.0f}  {s_['runtime_s'][0]:.0f}s")
    print(f"\nPart B ({len(res_b)} subproblems, size {pb['sizes'][0]:.1f}):")
    for k in ("p_qaoa", "p_random", "p_sa"):
        print(f"  {k:<9} P(opt) mean {pb[k]['mean']:.3f}  hit within " +
              ", ".join(f"{B}: {pb[k]['hit_within'][str(B)]:.3f}" for B in BUDGETS_B))
    print(f"  QAOA / random P(opt): median x{pb['qaoa_over_random']['median_ratio']:.2f}, "
          f"QAOA higher on {pb['qaoa_over_random']['qaoa_higher']}/{len(res_b)}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
