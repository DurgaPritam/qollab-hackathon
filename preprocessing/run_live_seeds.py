"""How much does the live run's outcome depend on the project seed? Offline only.

    .venv/bin/python preprocessing/run_live_seeds.py

For seeds 1..20, repeat the live default run's recipe on the default instance:
greedy incumbent -> release neighbourhoods built with that seed (<= lns_max_free qubits) ->
warm-start QAOA angles tuned for each neighbourhood -> 1,024 shots per neighbourhood sampled
from the exact output distribution (seeded) -> ONE sequential sweep of improving moves ->
lns_classical_sweeps classical LNS sweeps with an exact sub-solver.

Seed 7 is the project's fixed seed (CONFIG["seed"], CONFIG["lns_seed"]); it was chosen before
this sweep and is not changed by it. Writes results/e_live_seeds.json.
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

from fsq.baselines import adjacency, greedy  # noqa: E402
from fsq.config import CONFIG  # noqa: E402
from fsq.lns import (SOLVERS, apply_move, best_compatible, local_problem, qaoa_distribution,  # noqa: E402
                     release_neighbourhoods, run_lns)
from fsq.qaoa import sample_from_probs  # noqa: E402

SEEDS = list(range(1, 21))
SHOTS = 1024


def one_seed(seed):
    inst = json.load(open("data/instance_default.json"))
    w, edges = inst["w"], inst["edges"]
    opt = float(inst["exact"]["value"])
    adj = adjacency(len(w), edges)
    xg = greedy(w, edges)
    hoods = release_neighbourhoods(xg, adj, CONFIG["lns_max_free"], seed)
    x = xg.copy()
    moves = 0
    for j, f in enumerate(hoods):
        wl, el = local_problem(f, w, adj)
        probs, _, _ = qaoa_distribution(wl, el)
        X = sample_from_probs(probs, len(f), SHOTS, seed * 101 + j)
        new = best_compatible(f, f, X, x, w, adj)
        if new is not None and apply_move(x, f, new, w):
            moves += 1
    xq = x.copy()
    rc = run_lns(w, edges, "release", xq, SOLVERS["exact"], sweeps=CONFIG["lns_classical_sweeps"], budget=1,
                 seed=seed + 1000, max_free=CONFIG["lns_max_free"])
    return {"seed": seed, "greedy": float(np.dot(xg, w)) / opt, "quantum_sweep": float(np.dot(xq, w)) / opt,
            "after_classical": float(np.dot(rc["x"], w)) / opt, "moves": moves,
            "neighbourhoods": [len(f) for f in hoods]}


def main():
    t0 = time.time()
    with Pool(min(8, os.cpu_count() or 2)) as pool:
        rows = sorted(pool.map(one_seed, SEEDS), key=lambda r: r["seed"])
    q = np.array([r["quantum_sweep"] for r in rows])
    c = np.array([r["after_classical"] for r in rows])
    summary = {"n_seeds": len(rows), "project_seed": CONFIG["lns_seed"],
               "quantum_sweep": {"mean": float(q.mean()), "min": float(q.min()), "max": float(q.max()),
                                 "optimal": int(np.sum(q > 1 - 1e-9)),
                                 "values": sorted({round(v, 3) for v in q.tolist()})},
               "after_classical": {"mean": float(c.mean()), "min": float(c.min()),
                                   "optimal": int(np.sum(c > 1 - 1e-9))},
               "greedy": rows[0]["greedy"]}
    out = {"meta": {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shots": SHOTS,
                    "sampling": "exact QAOA output distribution, seeded draws (live run samples on a backend)",
                    "seed_controls": "neighbourhood construction and shot sampling"},
           "summary": summary, "seeds": rows}
    json.dump(out, open("results/e_live_seeds.json", "w"), indent=1)
    for r in rows:
        print(f"  seed {r['seed']:>2}  greedy {r['greedy']:.3f}  QAOA sweep {r['quantum_sweep']:.3f}  "
              f"+classical {r['after_classical']:.3f}  moves {r['moves']}")
    print(f"QAOA-sampled sweep: mean {q.mean():.4f}, optimal on {summary['quantum_sweep']['optimal']}/{len(rows)}, "
          f"values {summary['quantum_sweep']['values']}; after classical: optimal on "
          f"{summary['after_classical']['optimal']}/{len(rows)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
