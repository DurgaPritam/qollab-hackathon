"""Run every offline experiment and write results/*.json and the default Run instance.

    .venv/bin/python preprocessing/run_experiments.py            # everything
    .venv/bin/python preprocessing/run_experiments.py --quick    # 3 perturbed instances

Experiments (see CLAUDE.md §7):
  E1     all methods on the real event + 20 seeded perturbed instances
  E-DIV  fair diversity comparison, QAOA vs SA at the same sample budget
  E-CONN connectivity: all-to-all vs grid vs heavy-hex compile cost
  E2     QAOA depth p = 1, 2 (warm start)
  E3     warm start vs cold start
  E5     qubits needed vs instance size
All seeds are fixed and recorded. Every approximation ratio uses the exact (MILP) optimum.
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

from fsq.baselines import exact, greedy, simulated_annealing  # noqa: E402
from fsq.config import CONFIG  # noqa: E402
from fsq.connectivity import compare  # noqa: E402
from fsq.data_load import load_targets, load_windows, make_instance, perturb  # noqa: E402
from fsq.decompose import repair, stitch, cluster_subproblem  # noqa: E402
from fsq.diversity import pipeline_samples  # noqa: E402
from fsq.formulate import graph_density, penalty  # noqa: E402
from fsq.hybrid import assemble, prepare_clusters, tune_cluster  # noqa: E402
from fsq.metrics import approx_ratio, sample_stats  # noqa: E402
from fsq.qaoa import exact_probabilities, sample_from_probs  # noqa: E402

SHOTS = CONFIG["shots"]
SA_SWEEPS = 100          # cluster-level SA (<= 12 variables)
SA_SWEEPS_FULL = 1000    # full-schedule SA (~270 variables; tuned so SA is a strong baseline)
N_PERTURBED = 20
DEFAULT_RUN = {"horizon_h": 24, "n_targets": 10}   # small instance for the live Run button


def r4(v):
    return None if v is None else round(float(v), 4)


def default_target_ids(targets, n):
    """Top-n targets by priority x population (the default live instance)."""
    ranked = sorted(targets, key=lambda t: (-t["priority"] * t["population"], t["target_id"]))
    return [t["target_id"] for t in ranked[:n]]


def solve_instance(job):
    """E1 + E-DIV (+ E2/E3 when job['extra']) for one instance."""
    inst, seed, extra = job["inst"], job["seed"], job["extra"]
    w, edges = inst["w"], inst["edges"]
    n = len(w)
    out = {"instance": inst["id"], "seed": seed, "n": n, "edges": len(edges),
           "density": r4(graph_density(n, edges)), "methods": {}}
    t = time.perf_counter()
    xopt, how = exact(w, edges)
    opt = float(np.dot(xopt, w))
    out["exact"] = {"value": opt, "solver": how, "runtime_s": r4(time.perf_counter() - t),
                    "x": [int(v) for v in xopt]}

    def record(name, x, rt, **kw):
        out["methods"][name] = dict({"value": r4(np.dot(x, w)), "ratio": r4(approx_ratio(np.dot(x, w), opt)),
                                     "runtime_s": r4(rt), "x": [int(v) for v in x]}, **kw)

    t = time.perf_counter(); xg = greedy(w, edges); record("greedy", xg, time.perf_counter() - t)
    t = time.perf_counter()
    Xsa = simulated_annealing(w, edges, restarts=SHOTS, seed=seed, sweeps=SA_SWEEPS_FULL)
    vals = Xsa @ np.asarray(w)
    from fsq.metrics import feasible_mask
    vals[~feasible_mask(Xsa, edges)] = -1
    record("sa", Xsa[int(np.argmax(vals))], time.perf_counter() - t, restarts=SHOTS, sweeps=SA_SWEEPS_FULL)

    # hybrid
    t = time.perf_counter()
    clusters = prepare_clusters(inst, seed=seed)
    t_prep = time.perf_counter() - t
    variants = [("warm", 1)] + ([("warm", 2), ("cold", 1), ("cold", 2)] if extra else [])
    cl_samples = {}
    for kind, p in variants:
        t = time.perf_counter()
        params = [tune_cluster(cl, p, kind == "warm") for cl in clusters]
        t_tune = time.perf_counter() - t
        t = time.perf_counter()
        samples = []
        p_opt = []
        for k, (cl, prm) in enumerate(zip(clusters, params)):
            th = cl["thetas"] if kind == "warm" else None
            probs = exact_probabilities(cl["w"], cl["edges"], cl["P"], prm["gammas"], prm["betas"], th)
            samples.append(sample_from_probs(probs, len(cl["w"]), SHOTS, seed * 1000 + k))
            xc, _ = exact(cl["w"], cl["edges"])
            kopt = int(sum(int(b) << i for i, b in enumerate(xc)))
            p_opt.append(float(probs[kopt]))
        res = assemble(inst, clusters, samples)
        name = f"hybrid_{kind}_p{p}"
        record(name, res["x"], time.perf_counter() - t + t_prep, tune_s=r4(t_tune),
               repairs=len(res["repair_log"]), raw_feasible=res["raw_feasible"],
               mean_cluster_feasible=r4(np.mean(res["cluster_feasible_rates"])),
               mean_p_opt_cluster=r4(np.mean([p for p, cl in zip(p_opt, clusters) if len(cl["w"]) > 1] or [1.0])))
        cl_samples[name] = samples
        if name == "hybrid_warm_p1":
            out["clusters"] = [{"nodes": cl["nodes"], "size": len(cl["nodes"]),
                                "edges": len(cl["edges"]), "params": prm, "p_opt": r4(po)}
                               for cl, prm, po in zip(clusters, params, p_opt)]
            out["params_warm_p1"] = params
    # control: same decomposition, exact solve per cluster (isolates decomposition loss)
    t = time.perf_counter()
    loc = [exact(cl["w"], cl["edges"])[0] for cl in clusters]
    xd = repair(stitch([cl["nodes"] for cl in clusters], loc, n), w, edges)
    record("decomp_exact", xd, time.perf_counter() - t)

    # E-DIV, cluster level (clusters with >= 4 nodes; tiny clusters are trivial)
    div_clusters = []
    for k, cl in enumerate(clusters):
        if len(cl["w"]) < 4:
            continue
        xc, _ = exact(cl["w"], cl["edges"])
        copt = float(np.dot(xc, cl["w"]))
        Xq = cl_samples["hybrid_warm_p1"][k]
        Xs = simulated_annealing(cl["w"], cl["edges"], restarts=SHOTS, seed=seed * 1000 + k,
                                 sweeps=SA_SWEEPS, P=cl["P"])
        div_clusters.append({"cluster": k, "n": len(cl["w"]), "edges": len(cl["edges"]),
                             "qaoa": sample_stats(Xq, cl["w"], cl["edges"], copt, seed=seed),
                             "sa": sample_stats(Xs, cl["w"], cl["edges"], copt, seed=seed)})
    # E-DIV, pipeline level: both get the same repair step
    Xh = pipeline_samples(w, edges, [cl["nodes"] for cl in clusters], cl_samples["hybrid_warm_p1"])
    Xsa_rep = np.array([repair(x, w, edges) for x in Xsa])
    out["diversity"] = {
        "budget": SHOTS, "sa_sweeps": SA_SWEEPS, "sa_sweeps_full": SA_SWEEPS_FULL, "clusters": div_clusters,
        "pipeline": {"hybrid": sample_stats(Xh, w, edges, opt, seed=seed),
                     "sa_raw": sample_stats(Xsa, w, edges, opt, seed=seed),
                     "sa_repaired": sample_stats(Xsa_rep, w, edges, opt, seed=seed)}}
    out["n_clusters"] = len(clusters)
    out["max_cluster"] = max(len(cl["nodes"]) for cl in clusters)
    return out


def default_instance_file(inst, res, clusters):
    """Everything the live Run needs, so the browser does no optimisation."""
    return {
        "id": inst["id"], "description": "Default live instance: top-10 targets, first 24 h",
        "t0": inst["t0"], "candidates": inst["candidates"], "w": inst["w"], "edges": inst["edges"],
        "config": CONFIG,
        "clusters": [{"nodes": cl["nodes"], "w": cl["w"], "edges": cl["edges"], "P": cl["P"],
                      "thetas": cl["thetas"], "params": prm}
                     for cl, prm in zip(clusters, res["params_warm_p1"])],
        "exact": res["exact"], "precomputed": {k: {kk: v[kk] for kk in ("value", "ratio")}
                                                for k, v in res["methods"].items()},
    }


def conn_experiment(inst):
    """E-CONN on connected subgraphs of the real conflict graph, grown by BFS from the
    highest-degree candidate, plus each size's density."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(len(inst["w"])))
    G.add_edges_from(map(tuple, inst["edges"]))
    start = max(G.degree, key=lambda kv: kv[1])[0]
    order = list(nx.bfs_tree(G, start))
    rows = []
    for size in (4, 6, 8, 10, 12, 14, 16, 20, 24):
        if size > len(order):
            break
        nodes = order[:size]
        wl, el = cluster_subproblem(nodes, inst["w"], inst["edges"])
        for p in (1, 2):
            t = time.perf_counter()
            r = compare(wl, el, penalty(wl), p, seed=42, heavy_hex=True)
            r["compile_s"] = r4(time.perf_counter() - t)
            rows.append(r)
            print(f"  E-CONN n={size} p={p} density={r['density']:.2f} "
                  f"2q all={r['all_to_all']['two_qubit']} grid={r['grid']['two_qubit']} "
                  f"hex={r['heavy_hex']['two_qubit']}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-conn", action="store_true")
    args = ap.parse_args()
    t_all = time.perf_counter()
    targets, wdata = load_targets(), load_windows()
    n_pert = 3 if args.quick else N_PERTURBED
    real = make_instance(targets, wdata, instance_id="real")
    dflt_ids = default_target_ids(targets, DEFAULT_RUN["n_targets"])
    dflt = make_instance(targets, wdata, DEFAULT_RUN["horizon_h"], dflt_ids, "default")
    jobs = [{"inst": real, "seed": CONFIG["seed"], "extra": True},
            {"inst": dflt, "seed": CONFIG["seed"], "extra": True}]
    jobs += [{"inst": perturb(targets, wdata, s), "seed": s, "extra": s <= 5} for s in range(1, n_pert + 1)]
    print(f"real: n={len(real['w'])} edges={len(real['edges'])}; default: n={len(dflt['w'])}; "
          f"{n_pert} perturbed", flush=True)
    with Pool(min(8, os.cpu_count() or 2)) as pool:
        results = []
        for r in pool.imap(solve_instance, jobs):
            m = r["methods"]
            print(f"  {r['instance']:>14} n={r['n']:3d} clusters={r['n_clusters']:2d} "
                  + " ".join(f"{k}={v['ratio']:.3f}" for k, v in m.items()), flush=True)
            results.append(r)
    meta = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": platform.python_version(), "config": CONFIG, "shots": SHOTS,
            "sa_sweeps": SA_SWEEPS, "sa_sweeps_full": SA_SWEEPS_FULL, "n_perturbed": n_pert}
    try:
        import qiskit, scipy, numpy
        meta["versions"] = {"qiskit": qiskit.__version__, "scipy": scipy.__version__, "numpy": numpy.__version__}
    except Exception:
        pass
    os.makedirs("results", exist_ok=True)
    real_r, dflt_r = results[0], results[1]
    e1 = {"meta": meta, "instances": [{k: v for k, v in r.items() if k not in ("diversity", "params_warm_p1")}
                                      for r in results]}
    json.dump(e1, open("results/e1.json", "w"))
    json.dump({"meta": meta, "instances": [{"instance": r["instance"], "seed": r["seed"], "n": r["n"],
                                            "exact_value": r["exact"]["value"], **r["diversity"]}
                                           for r in results]}, open("results/e_div.json", "w"))
    json.dump({"meta": meta, "instances": [{"instance": r["instance"], "value": r["exact"]["value"],
                                            "solver": r["exact"]["solver"], "x": r["exact"]["x"]}
                                           for r in results]}, open("results/exact.json", "w"))
    # default instance for the live run
    clusters = prepare_clusters(dflt, seed=CONFIG["seed"])
    json.dump(default_instance_file(dflt, dflt_r, clusters), open("data/instance_default.json", "w"))
    json.dump({"instance": "default", "p": 1, "warm": True,
               "clusters": [{"nodes": c["nodes"], **c["params"]} for c in dflt_r["clusters"]]},
              open("data/params.json", "w"), indent=1)
    # E5: qubits needed vs size
    e5 = []
    for h in (6, 12, 24, 36, 48):
        inst = make_instance(targets, wdata, h, instance_id=f"h{h}")
        cl = prepare_clusters(inst, seed=CONFIG["seed"]) if inst["w"] else []
        from fsq.decompose import connected_components
        comps = connected_components(len(inst["w"]), inst["edges"])
        e5.append({"horizon_h": h, "candidates": len(inst["w"]), "edges": len(inst["edges"]),
                   "density": r4(graph_density(len(inst["w"]), inst["edges"])),
                   "largest_component": max((len(c) for c in comps), default=0),
                   "clusters_at_12": len(cl)})
    json.dump({"meta": meta, "rows": e5}, open("results/e5.json", "w"))
    if not args.skip_conn:
        print("E-CONN ...", flush=True)
        json.dump({"meta": dict(meta, basis=["rz", "sx", "x", "cx"], optimization_level=3,
                                seed_transpiler=42, e2_model=0.005),
                   "rows": conn_experiment(real)}, open("results/e_conn.json", "w"))
    print(f"done in {time.perf_counter() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
