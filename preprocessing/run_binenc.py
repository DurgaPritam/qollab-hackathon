"""Step 4: binary (qubit-efficient) encoding inside conditioned LNS. Offline only.

    .venv/bin/python preprocessing/run_binenc.py

Neighbourhoods are grown until their BINARY encoding needs at most 12 qubits (about twice
as many candidates as one-hot at 12 qubits). Same 21 instances, seeds and 10 sweeps as
run_lns.py. Sub-solvers on the same neighbourhoods:
  exact (MILP, ceiling); QAOA on the encoded circuit (warm start, CVaR, p=1, exact
  statevector); SA on the one-hot subproblem; uniform random ENCODED bitstrings (decoded,
  invalid codes -> none). Budgets 1,024 and 32 per neighbourhood, each in its own units.
Part B: identical binary-sized subproblems; P(optimum) per sample, and the circuit cost of
the encoded cost layer (Pauli-Z terms by weight, CNOTs as parity ladders on all-to-all
hardware) vs the one-hot circuit for the same subproblem (2 CNOTs per conflict edge).

Writes results/e_binenc.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from fsq.baselines import adjacency, exact, greedy, simulated_annealing  # noqa: E402
from fsq.binenc import (Encoding, binary_qaoa_distribution, binary_size, cnot_cost, pauli_terms,  # noqa: E402
                        propose_exact_large, propose_qaoa_binary, propose_random_encoded)
from fsq.lns import local_problem, propose_sa, release_neighbourhoods, run_lns  # noqa: E402
from fsq.metrics import feasible_mask  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "preprocessing"))
from run_lns import instances  # noqa: E402

QB = 12
SWEEPS = 10
SOLVERS = {"exact": propose_exact_large, "qaoa_bin": propose_qaoa_binary, "sa": propose_sa,
           "random_enc": propose_random_encoded}


def job_a(args):
    inst, seed, solver, budget = args
    t = time.perf_counter()
    w, edges = inst["w"], inst["edges"]
    adj = adjacency(len(w), edges)
    r = run_lns(w, edges, "release", greedy(w, edges), SOLVERS[solver], sweeps=SWEEPS, budget=budget,
                seed=seed, max_free=QB, size=binary_size(adj))
    return {"instance": inst["id"], "solver": solver, "budget": budget, "values": r["values"],
            "moves": r["moves"], "runtime_s": time.perf_counter() - t}


def job_b(args):
    wl, el, seed = args
    probs, X, enc, _ = binary_qaoa_distribution(wl, el)
    ws = np.asarray(wl)
    opt = float(np.dot(exact(wl, el)[0], ws))
    feas = feasible_mask(X, el)
    is_opt = feas & (X @ ws >= opt - 1e-9)
    m = max(wl)
    Xs = simulated_annealing([v / m for v in wl], el, restarts=1024, seed=seed, sweeps=100, P=1.2)
    p_sa = float(np.mean(feasible_mask(Xs, el) & (Xs @ ws >= opt - 1e-9)))
    from fsq.qaoa import scaled_cluster_qubo
    E, _ = Encoding(*scaled_cluster_qubo(wl, 1.2)[:1], el).energies(1.2)
    terms = pauli_terms(E, enc.Q)
    return {"n": len(wl), "Q": enc.Q, "edges": len(el), "p_qaoa": float(probs[is_opt].sum()),
            "p_random_enc": float(is_opt.mean()), "p_sa": p_sa,
            "terms_by_weight": {str(k): v for k, v in sorted(terms.items())},
            "cnot_binary_p1": cnot_cost(terms), "cnot_one_hot_p1": 2 * len(el)}


def subproblems(inst, seed, k):
    w, edges = inst["w"], inst["edges"]
    adj = adjacency(len(w), edges)
    hoods = release_neighbourhoods(greedy(w, edges), adj, QB, seed, size=binary_size(adj))
    out = [local_problem(f, w, adj) for f in hoods if len(f) >= 6]
    return out[:k]


def ms(v):
    return [float(np.mean(v)), float(np.std(v))]


def main():
    t0 = time.time()
    insts = instances(20)
    exact_opt = {r["instance"]: r["value"] for r in json.load(open("results/exact.json"))["instances"]}
    jobs = [(inst, seed, s, b) for inst, seed in insts for s in SOLVERS for b in (1024, 32)
            if not (s == "exact" and b == 32)]
    with Pool(min(8, os.cpu_count() or 2)) as pool:
        res_a = []
        for r in pool.imap_unordered(job_a, jobs):
            res_a.append(r)
            print(f"  A {r['instance']:>14} {r['solver']:>10} b{r['budget']:<5} "
                  f"ratio {r['values'][-1] / exact_opt[r['instance']]:.4f} {r['runtime_s']:.0f}s", flush=True)
        sub = [(wl, el, seed * 31 + k) for inst, seed in insts[:6] for k, (wl, el) in enumerate(subproblems(inst, seed, 12))]
        res_b = pool.map(job_b, sub)
    summary = {}
    for r in res_a:
        key = f"{r['solver']}_b{r['budget']}"
        d = summary.setdefault(key, {"solver": r["solver"], "budget": r["budget"], "final": [], "runtime_s": [],
                                     "per_instance": {}})
        v = r["values"][-1] / exact_opt[r["instance"]]
        d["final"].append(v); d["runtime_s"].append(r["runtime_s"]); d["per_instance"][r["instance"]] = v
    for d in summary.values():
        f = np.array(d["final"])
        d.update(mean_sd=ms(f), worst=float(f.min()), optimal=int(np.sum(f > 1 - 1e-9)), runtime=ms(d["runtime_s"]))
        del d["final"], d["runtime_s"]
    pb = {k: {"mean": float(np.mean([r[k] for r in res_b])), "median": float(np.median([r[k] for r in res_b]))}
          for k in ("p_qaoa", "p_random_enc", "p_sa")}
    q, u = np.array([r["p_qaoa"] for r in res_b]), np.array([r["p_random_enc"] for r in res_b])
    pb["qaoa_over_random_enc"] = {"median_ratio": float(np.median(q / np.maximum(u, 1e-12))),
                                  "qaoa_higher": int(np.sum(q > u)), "n": len(res_b)}
    pb["n_candidates"] = ms([r["n"] for r in res_b]); pb["Q"] = ms([r["Q"] for r in res_b])
    pb["cnot_binary_p1"] = ms([r["cnot_binary_p1"] for r in res_b])
    pb["cnot_one_hot_p1"] = ms([r["cnot_one_hot_p1"] for r in res_b])
    out = {"meta": {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "qubit_budget": QB,
                    "sweeps": SWEEPS, "n_instances": len(insts),
                    "encoding": "greedy clique cover; group of k -> ceil(log2(k+1)) qubits; invalid codes penalised",
                    "qaoa": "warm start from clique LP mapped to code bits (eps 0.25), CVaR 0.2, p=1, grid 8x8 + COBYLA 60",
                    "costs": {"qaoa_bin": "shots of one encoded circuit", "sa": "restarts x 100 sweeps (one-hot)",
                              "random_enc": "uniform encoded bitstrings", "exact": "MILP (ceiling)"},
                    "cnot_model": "2(k-1) CNOTs per weight-k Z term (parity ladder), all-to-all, cost layer only"},
           "part_a": summary, "part_b": pb, "part_b_subproblems": res_b}
    json.dump(out, open("results/e_binenc.json", "w"))
    print(f"\nBinary-encoded LNS ({QB} encoded qubits, {len(insts)} instances, {SWEEPS} sweeps):")
    for k in sorted(summary):
        s = summary[k]
        print(f"  {k:<16} {s['mean_sd'][0]:.4f} ± {s['mean_sd'][1]:.4f}  worst {s['worst']:.4f}  "
              f"optimal {s['optimal']}/{len(insts)}  {s['runtime'][0]:.0f}s")
    print(f"Part B ({len(res_b)} subproblems, {pb['n_candidates'][0]:.1f} candidates on {pb['Q'][0]:.1f} qubits): "
          f"P(opt) QAOA {pb['p_qaoa']['mean']:.3f}, SA {pb['p_sa']['mean']:.3f}, random encoded "
          f"{pb['p_random_enc']['mean']:.4f}; QAOA higher on {pb['qaoa_over_random_enc']['qaoa_higher']}/{len(res_b)}; "
          f"CNOTs p=1 binary {pb['cnot_binary_p1'][0]:.0f} vs one-hot {pb['cnot_one_hot_p1'][0]:.0f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
