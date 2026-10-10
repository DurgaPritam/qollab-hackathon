"""Entry point used by the Qollab Run button (called from main.js).

Default run (fast, no optimisation in the browser):
  1. load data/instance_default.json (candidates, conflicts, clusters, tuned QAOA angles)
  2. build one warm-start QAOA circuit per cluster
  3. sample with at most ONE backend job. IonQ accepts one circuit per job, so there the
     clusters are packed side by side into one circuit; local simulators get the list.
  4. keep each cluster's best feasible sample, stitch, repair conflicts
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
from .qaoa import counts_to_samples, sample_clusters
from .config import CONFIG

E1_JSON = "results/e1.json"

VERDICT = ("Honest verdict: at this size, classical methods find schedules at least as good "
           "as the hybrid. The quantum part is measured fairly; it is not an advantage.")


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


def hybrid_label(bname: str, info: dict, n_clusters: int, sizes: list[int]) -> str:
    """Results-row label that states where sampling actually happened."""
    if info["mode"] == "none":
        return "Hybrid QAOA (sampled locally - no backend)"
    if info["error"]:
        who = "IonQ" if info["mode"] == "packed" else bname
        return f"Hybrid QAOA (sampled locally - {who} job failed)"
    r, l = info["remote"], info["local"]
    if info["mode"] == "packed":
        cl = lambda k: f"{k} cluster" + ("" if k == 1 else "s")  # noqa: E731
        lab = f"IonQ: {cl(len(r))} / {info['remote_qubits']} qubits"
        return f"Hybrid QAOA ({lab}; local: {cl(len(l))})" if l else f"Hybrid QAOA ({lab})"
    return f"Hybrid QAOA ({bname}: all {n_clusters} clusters)"


def gap_insight(path: str = E1_JSON, tol: float = 0.005) -> str:
    """Hybrid vs 'same split, exact per cluster' on the 21 precomputed instances."""
    e1 = json.load(open(path, encoding="utf-8"))
    bench = [r for r in e1["instances"] if r["instance"] != "default"]
    h = [r["methods"]["hybrid_warm_p1"]["ratio"] for r in bench]
    c = [r["methods"]["decomp_exact"]["ratio"] for r in bench]
    same = sum(abs(a - b) <= 1e-9 for a, b in zip(h, c))
    line = (f"Precomputed, {len(bench)} instances: hybrid QAOA mean ratio {np.mean(h):.3f} vs same split, "
            f"exact per cluster {np.mean(c):.3f} (identical on {same} of {len(bench)}).")
    if abs(np.mean(h) - np.mean(c)) <= tol:
        line += ("\n  Hybrid QAOA matches the exact answer within each cluster; the gap to the global "
                 "optimum comes from splitting the problem, not from QAOA.")
    else:
        line += f"\n  QAOA loses {np.mean(c) - np.mean(h):.3f} on top of the decomposition."
    return line


def run_default(backend=None, shots: int = 1024, seed: int = 7) -> str:
    """Run the default scenario and return a JSON summary string."""
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
    qidx = [k for k, cl in enumerate(clusters) if len(cl["nodes"]) > 1]
    sizes_all = [len(cl["nodes"]) for cl in clusters]
    print(f"Instance: {inst['description']}")
    print(f"  {n} candidate images, {len(edges)} conflicts, {len(clusters)} clusters "
          f"(sizes {sizes_all}; largest {max(sizes_all)} qubits)")
    print(f"Backend: {bname}   shots: {shots}")

    # ---- quantum: at most one job -------------------------------------------------------
    _line()
    print("[1/3] Final QAOA sampling (precomputed warm-start angles, p=1), at most one job")
    t = time.perf_counter()
    circuits = [cluster_circuit(clusters[k], clusters[k]["params"]) for k in qidx]
    counts, info = sample_clusters(backend, circuits, shots, seed, CONFIG["ionq_qubit_budget"])
    qsizes = [c.num_qubits for c in circuits]
    cid = lambda j: qidx[j] + 1  # noqa: E731  (1-based cluster ids, as printed)
    if info["mode"] == "packed" and not info["error"]:
        k = len(info["remote"])
        print(f"  IonQ (one circuit per job): {k} cluster{'' if k == 1 else 's'} in ONE "
              f"{info['remote_qubits']}-qubit circuit (budget {info['budget']} qubits = min(backend size, "
              f"IONQ_QUBIT_BUDGET)), ONE job")
    elif info["mode"] == "list" and not info["error"]:
        print(f"  one job on {bname} with {len(circuits)} circuits (one per cluster)")
    if info["error"]:
        print(f"  backend job FAILED ({info['error']}); nothing else was sent to the backend")
    if info["remote"]:
        print("  sampled on backend: " + ", ".join(f"cluster {cid(j)} ({qsizes[j]}q)" for j in sorted(info["remote"])))
    if info["local"]:
        print("  sampled locally:    " + ", ".join(f"cluster {cid(j)} ({qsizes[j]}q)" for j in sorted(info["local"])))
    samples, qi = [], 0
    for cl in clusters:
        if len(cl["nodes"]) == 1:
            samples.append(np.ones((shots, 1), dtype=np.int8))   # isolated candidate: take it
        else:
            samples.append(counts_to_samples(counts[qi], len(cl["nodes"])))
            qi += 1
    res = assemble(inst, clusters, samples)
    t_q = time.perf_counter() - t
    drops = [a for a in res["repair_log"] if a["action"] == "drop"]
    adds = [a for a in res["repair_log"] if a["action"] == "add"]
    print(f"  clusters with at least one feasible sample: {res['clusters_with_feasible']} of {len(clusters)} "
          f"(best feasible sample kept per cluster)")
    print(f"  cut conflicts (between clusters, from the partition): {res['cut_conflicts']} of {len(edges)}")
    print(f"  stitched schedule feasible before repair: {res['raw_feasible']}; repair: dropped {len(drops)}, "
          f"added {len(adds)} (net {len(adds) - len(drops):+d} images)")
    for a in res["repair_log"]:
        c = inst["candidates"][a["candidate"]]
        print(f"    {a['action']:>4}: {c['target_id']} by {c['satellite']} at +{c['t_hours']:.1f} h")
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
    print(f"  greedy {t_g:.2f} s; simulated annealing {shots} restarts x 100 sweeps {t_s:.2f} s")
    x_exact = np.asarray(inst["exact"]["x"], dtype=np.int8)

    # ---- results -------------------------------------------------------------------------
    _line()
    print("[3/3] Results (value = priority x people x urgency x image quality; 1.000 = exact optimum)")
    hlabel = hybrid_label(bname, info, len(circuits), qsizes)
    rows = [("exact", "Exact optimum (MILP, precomputed)", x_exact, None),
            ("greedy", "Greedy (value density)", xg, t_g),
            ("sa", f"Simulated annealing ({shots} restarts)", xs, t_s),
            ("hybrid", hlabel, res["x"], t_q)]
    width = max(46, max(len(r[1]) for r in rows) + 2)
    print(f"  {'method':<{width}}{'ratio':>7}{'images':>8}{'feasible':>10}{'time s':>8}")
    out_methods = {}
    for key, name, x, rt in rows:
        val = float(np.dot(x, w))
        print(f"  {name:<{width}}{approx_ratio(val, opt):>7.3f}{int(np.sum(x)):>8d}"
              f"{str(is_feasible(x, edges)):>10}{('-' if rt is None else f'{rt:.2f}'):>8}")
        out_methods[key] = {"label": name, "ratio": approx_ratio(val, opt), "x": [int(v) for v in x],
                            "runtime_s": rt}
    if cl_stats:
        fr = np.mean([s["feasible_rate"] for s in cl_stats])
        ng = np.mean([s["n_good"] for s in cl_stats])
        print(f"  per-cluster QAOA samples: mean feasible rate {fr:.2f}, "
              f"mean distinct near-optimal (>=95%) schedules {ng:.1f}")
    _line()
    try:
        print("Where the gap comes from: " + gap_insight())
    except Exception as e:
        print(f"(decomposition insight unavailable: {type(e).__name__})")
    if info["panel"]:
        print(f"Probabilities panel: shows {info['panel']}. All clusters are summarised above.")
    else:
        print("Probabilities panel: no circuit ran on a backend in this run (all sampled locally).")
    _line()
    print(VERDICT)
    print(f"Total runtime: {time.perf_counter() - t_all:.1f} s")
    _line("=")
    return json.dumps({"backend": bname, "shots": shots, "sampling": info, "hybrid_label": hlabel,
                       "methods": out_methods, "repair_log": res["repair_log"],
                       "raw_feasible": res["raw_feasible"], "cluster_stats": cl_stats,
                       "runtime_s": time.perf_counter() - t_all})


if __name__ == "__main__":
    run_default(None)
