"""Entry point used by the Qollab Run button (called from main.js).

Default run (fast, no optimisation in the browser):
  1. load data/instance_default.json (candidates, conflicts, clusters, tuned QAOA angles)
  2. build one warm-start QAOA circuit per cluster
  3. send ALL cluster circuits to the selected backend in ONE job (IonQ allows one job)
  4. stitch the best sample of each cluster and repair conflicts
  5. run greedy and simulated annealing live; compare with the precomputed exact optimum
"""

from __future__ import annotations

import json
import time

import numpy as np

from .baselines import exact_bruteforce, greedy, simulated_annealing
from .data_load import load_default_instance
from .formulate import is_feasible
from .hybrid import assemble, cluster_circuit
from .metrics import approx_ratio, feasible_mask, sample_stats
from .qaoa import counts_to_samples, run_one_job

VERDICT = ("Honest verdict: at this size, classical methods find schedules at least as good "
           "as the hybrid. The quantum part is measured fairly below; it is not an advantage.")


def _backend_name(backend) -> str:
    for attr in ("name", "backend_name"):
        try:
            v = getattr(backend, attr)
            v = v() if callable(v) else v
            if v:
                return str(v)
        except Exception:
            pass
    return "unknown" if backend is not None else "none (local)"


def _line(char="-", n=72):
    print(char * n)


def run_default(backend=None, shots: int = 1024, mode: str = "list", seed: int = 7) -> str:
    """Run the default scenario and return a JSON string for the page."""
    t_all = time.perf_counter()
    bname = _backend_name(backend)
    _line("=")
    print("FloodSight Q - SAR imaging schedule for flood response (hybrid classical-quantum)")
    _line("=")
    inst = load_default_instance()
    w, edges = inst["w"], inst["edges"]
    n = len(w)
    opt = float(inst["exact"]["value"])
    clusters = inst["clusters"]
    qclusters = [cl for cl in clusters if len(cl["nodes"]) > 1]
    print(f"Instance: {inst['description']}")
    print(f"  {n} candidate images, {len(edges)} conflicts, {len(clusters)} clusters "
          f"({len(qclusters)} need qubits; largest {max(len(c['nodes']) for c in clusters)} qubits)")
    print(f"Backend: {bname}   shots: {shots}   mode: {mode}")

    # ---- quantum: one job --------------------------------------------------------------
    _line()
    print("[1/3] Final QAOA sampling (precomputed warm-start angles, p=1), one job")
    t = time.perf_counter()
    circuits = [cluster_circuit(cl, cl["params"]) for cl in qclusters]
    counts, where, note = run_one_job(backend, circuits, shots, seed, mode)
    print(f"  {note}")
    samples, qi = [], 0
    for cl in clusters:
        if len(cl["nodes"]) == 1:
            samples.append(np.ones((shots, 1), dtype=np.int8))   # isolated candidate: take it
        else:
            samples.append(counts_to_samples(counts[qi], len(cl["nodes"])))
            qi += 1
    res = assemble(inst, clusters, samples)
    t_q = time.perf_counter() - t
    drops = sum(1 for a in res["repair_log"] if a["action"] == "drop")
    adds = len(res["repair_log"]) - drops
    print(f"  stitched schedule feasible before repair: {res['raw_feasible']}; "
          f"repair dropped {drops}, added {adds}")
    for a in res["repair_log"]:
        c = inst["candidates"][a["candidate"]]
        print(f"    repair {a['action']:>4}: {c['target_id']} by {c['satellite']} at +{c['t_hours']:.1f} h")
    cl_stats = []
    for cl, X in zip(clusters, samples):
        if 1 < len(cl["nodes"]) <= 16:   # brute force: NumPy only, no SciPy load in the browser
            copt = float(np.dot(exact_bruteforce(cl["w"], cl["edges"]), cl["w"]))
            cl_stats.append(sample_stats(X, cl["w"], cl["edges"], copt))

    # ---- classical baselines live --------------------------------------------------------
    _line()
    print("[2/3] Classical baselines (live)")
    t = time.perf_counter(); xg = greedy(w, edges); t_g = time.perf_counter() - t
    t = time.perf_counter()
    Xs = simulated_annealing(w, edges, restarts=shots, seed=seed, sweeps=100)
    vals = Xs @ np.asarray(w)
    vals[~feasible_mask(Xs, edges)] = -1
    xs = Xs[int(np.argmax(vals))]
    t_s = time.perf_counter() - t
    x_exact = np.asarray(inst["exact"]["x"], dtype=np.int8)

    # ---- results -------------------------------------------------------------------------
    _line()
    print("[3/3] Results (value = weighted people-priority-urgency-quality, 1.0 = exact optimum)")
    rows = [("Exact optimum (MILP, precomputed)", x_exact, None),
            ("Greedy (value density)", xg, t_g),
            (f"Simulated annealing ({shots} restarts)", xs, t_s),
            (f"Hybrid QAOA (this run, {bname})", res["x"], t_q)]
    print(f"  {'method':<44}{'ratio':>7}{'images':>8}{'feasible':>10}{'time s':>8}")
    out_methods = {}
    for name, x, rt in rows:
        val = float(np.dot(x, w))
        print(f"  {name:<44}{approx_ratio(val, opt):>7.3f}{int(np.sum(x)):>8d}"
              f"{str(is_feasible(x, edges)):>10}{('-' if rt is None else f'{rt:.2f}'):>8}")
        key = name.split(" ")[0].lower()
        out_methods[key] = {"ratio": approx_ratio(val, opt), "x": [int(v) for v in x],
                            "runtime_s": rt}
    if cl_stats:
        fr = np.mean([s["feasible_rate"] for s in cl_stats])
        ng = np.mean([s["n_good"] for s in cl_stats])
        print(f"  per-cluster QAOA samples: mean feasible rate {fr:.2f}, "
              f"mean distinct near-optimal (>=95%) schedules {ng:.1f}")
    _line()
    print(VERDICT)
    print(f"Total runtime: {time.perf_counter() - t_all:.1f} s")
    _line("=")
    return json.dumps({"backend": bname, "shots": shots, "note": note, "where": where,
                       "methods": out_methods, "repair_log": res["repair_log"],
                       "raw_feasible": res["raw_feasible"], "cluster_stats": cl_stats,
                       "runtime_s": time.perf_counter() - t_all})


if __name__ == "__main__":
    run_default(None)
