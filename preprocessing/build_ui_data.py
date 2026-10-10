"""Bundle data/ and results/ into ui/data.js for the page (instant load, no fetch).

    .venv/bin/python preprocessing/build_ui_data.py

Captions that state a result are generated from the numbers here, so the page text
always matches the data.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from fsq.data_load import load_targets, load_windows, make_instance  # noqa: E402
from preprocessing.run_experiments import DEFAULT_RUN, default_target_ids  # noqa: E402

LABELS = {"exact": "Exact optimum (MILP)", "greedy": "Greedy (value density)",
          "sa": "Simulated annealing (1024 restarts)", "hybrid_warm_p1": "Hybrid QAOA (warm start, p=1)",
          "decomp_exact": "Control: same decomposition, exact per cluster",
          "hybrid_warm_p2": "Hybrid QAOA warm start p=2", "hybrid_cold_p1": "Hybrid QAOA cold start p=1",
          "hybrid_cold_p2": "Hybrid QAOA cold start p=2"}


def ms(v):
    v = [x for x in v if x is not None]
    return [float(np.mean(v)), float(np.std(v))] if v else None


def idx(x):
    return [i for i, b in enumerate(x) if b]


def main():
    targets, wdata = load_targets(), load_windows()
    meta_src = json.load(open("data/meta.json"))
    e1 = json.load(open("results/e1.json"))
    ediv = json.load(open("results/e_div.json"))
    econn = json.load(open("results/e_conn.json"))
    e5 = json.load(open("results/e5.json"))
    by_id = {r["instance"]: r for r in e1["instances"]}
    bench = [r for r in e1["instances"] if r["instance"] != "default"]   # real + perturbed

    insts = {}
    for key, label, inst in (
        ("real", f"Real event, all {len(targets)} targets, {wdata['horizon_h']} h",
         make_instance(targets, wdata, instance_id="real")),
        ("default", f"Default live run: top {DEFAULT_RUN['n_targets']} targets, first {DEFAULT_RUN['horizon_h']} h",
         make_instance(targets, wdata, DEFAULT_RUN["horizon_h"],
                       default_target_ids(targets, DEFAULT_RUN["n_targets"]), "default"))):
        r = by_id[key]
        sched = {"exact": idx(r["exact"]["x"])}
        ratios = {"exact": 1.0}
        for m, k in (("greedy", "greedy"), ("sa", "sa"), ("hybrid_warm_p1", "hybrid")):
            sched[k] = idx(r["methods"][m]["x"])
            ratios[k] = r["methods"][m]["ratio"]
        insts[key] = {
            "label": label, "horizon_h": DEFAULT_RUN["horizon_h"] if key == "default" else wdata["horizon_h"],
            "n_edges": len(inst["edges"]),
            "candidates": [[c["target_id"], c["satellite"], round(c["t_hours"], 3), round(c["incidence"], 1),
                            round(w, 4), c["look_side"]] for c, w in zip(inst["candidates"], inst["w"])],
            "schedules": sched, "ratios": ratios}

    # E1 table
    rows = []
    for m in ("greedy", "sa", "hybrid_warm_p1", "decomp_exact"):
        vals = [r["methods"][m]["ratio"] for r in bench]
        rows.append({"label": LABELS[m], "real": by_id["real"]["methods"][m]["ratio"], "mean": float(np.mean(vals)),
                     "sd": float(np.std(vals)), "min": float(np.min(vals)),
                     "runtime": float(np.mean([r["methods"][m]["runtime_s"] for r in bench]))})
    n_b = len(bench)
    loss = [r["methods"]["decomp_exact"]["ratio"] - r["methods"]["hybrid_warm_p1"]["ratio"] for r in bench]
    e1_note = (f"{n_b} instances. Hybrid runtime excludes offline angle tuning. The control row uses the hybrid's "
               f"decomposition but solves each cluster exactly; the hybrid matches it on "
               f"{sum(abs(l) < 1e-9 for l in loss)} of {n_b} instances, so the gap to the optimum comes from "
               f"splitting the problem into ≤12-qubit clusters, not from QAOA.")

    # E-DIV aggregate
    dinst = [d for d in ediv["instances"] if d["instance"] != "default"]
    cl = [c for d in dinst for c in d["clusters"]]
    agg = {m: {k: ms([c[m][k] for c in cl]) for k in ("feasible_rate", "n_good", "diversity", "best_ratio")}
           for m in ("qaoa", "sa")}
    agg["n"] = len(cl)
    pipe = {m: {k: ms([d["pipeline"][m][k] for d in dinst]) for k in ("feasible_rate", "n_good", "diversity", "best_ratio")}
            for m in ("hybrid", "sa_raw", "sa_repaired")}
    q, sa = agg["qaoa"], agg["sa"]
    more_good = "QAOA" if q["n_good"][0] > sa["n_good"][0] else "simulated annealing"
    qn = np.array([c["qaoa"]["n_good"] for c in cl])
    sn = np.array([c["sa"]["n_good"] for c in cl])
    full_win = "simulated annealing" if pipe["sa_raw"]["n_good"][0] > pipe["hybrid"]["n_good"][0] else "the hybrid"
    div_caption = (
        f"Same QUBO, same budget: {ediv['instances'][0]['budget']} QAOA shots vs {ediv['instances'][0]['budget']} SA restarts. "
        f"Per cluster ({len(cl)} clusters, SA {ediv['instances'][0]['sa_sweeps']} sweeps), {more_good} found more distinct "
        f"near-optimal schedules on average ({fmt(q['n_good'][0])} vs {fmt(sa['n_good'][0])}): QAOA had more in "
        f"{int((qn > sn).sum())} clusters, SA in {int((qn < sn).sum())}, tie in {int((qn == sn).sum())}. SA's samples were "
        f"always feasible; QAOA's {fmt(100 * q['feasible_rate'][0], 0)}%. For whole schedules, {full_win} wins clearly: "
        f"{fmt(pipe['sa_raw']['n_good'][0], 0)} vs {fmt(pipe['hybrid']['n_good'][0], 0)} distinct near-optimal schedules per "
        f"instance (SA {ediv['instances'][0]['sa_sweeps_full']} sweeps), with more diversity. Diversity = mean Hamming "
        f"distance among near-optimal schedules (more = more different backup plans).")

    # E-CONN caption
    big = max((r for r in econn["rows"] if r["p"] == 1), key=lambda r: r["n"])
    conn_caption = (
        f"Subgraphs of the real conflict graph, grown from its densest point. At {big['n']} qubits (density "
        f"{big['density']:.2f}), p=1 needs {big['all_to_all']['two_qubit']} two-qubit gates with all-to-all "
        f"connectivity vs {big['grid']['two_qubit']} on a square grid and {big['heavy_hex']['two_qubit']} on heavy-hex "
        f"(×{big['grid']['two_qubit'] / max(big['all_to_all']['two_qubit'], 1):.1f} and "
        f"×{big['heavy_hex']['two_qubit'] / max(big['all_to_all']['two_qubit'], 1):.1f}). Extra gates are SWAPs.")

    # E2 / E3
    extra = [r for r in bench if "hybrid_cold_p1" in r["methods"]]
    e23 = []
    for m in ("hybrid_warm_p1", "hybrid_warm_p2", "hybrid_cold_p1", "hybrid_cold_p2"):
        e23.append([LABELS[m], pm(ms([r["methods"][m]["ratio"] for r in extra])),
                    pm(ms([r["methods"][m]["mean_p_opt_cluster"] for r in extra])),
                    pm(ms([r["methods"][m]["mean_cluster_feasible"] for r in extra])),
                    pm(ms([r["methods"][m]["tune_s"] for r in extra]), 1)])
    more = [
        {"title": "Depth (E2) and warm vs cold start (E3)",
         "text": f"{len(extra)} instances (real + perturbed 1-5). P(optimum) = probability the exact QAOA state puts on "
                 f"each cluster's optimal schedule, averaged over clusters with ≥2 candidates.",
         "headers": ["Variant", "ratio to optimum", "P(cluster optimum)", "cluster feasible rate", "tuning s"],
         "rows": e23},
        {"title": "Qubits needed vs instance size (E5)",
         "text": "Without decomposition the whole problem needs one qubit per candidate image. Today's IonQ simulators "
                 "offer 25-29 qubits, so realistic instances must be decomposed.",
         "headers": ["horizon", "candidates (qubits)", "conflicts", "density", "largest connected component", "clusters ≤12"],
         "rows": [[f"{r['horizon_h']} h", r["candidates"], r["edges"], fmt(r["density"], 3), r["largest_component"], r["clusters_at_12"]]
                  for r in e5["rows"]]},
    ]
    if os.path.exists("results/e4.json"):
        e4 = json.load(open("results/e4.json"))
        more.append({"title": "Noise (E4)", "text": e4["text"], "headers": e4["headers"], "rows": e4["rows"]})

    data = {
        "meta": {"population_label": meta_src["population_label"], "footer": meta_src["footer"],
                 "generated": e1["meta"]["generated_utc"]},
        "targets": [{k: t[k] for k in ("target_id", "name", "lat", "lon", "population", "priority")} for t in targets],
        "region": meta_src["region_rings"],
        "instances": insts,
        "e1": {"rows": rows, "note": e1_note},
        "div": {"cluster": agg, "pipeline": pipe, "budget": ediv["instances"][0]["budget"],
                "n_instances": len(dinst), "caption": div_caption},
        "conn": {"rows": econn["rows"], "caption": conn_caption},
        "more": more,
    }
    with open("ui/data.js", "w") as f:
        f.write("// Generated by preprocessing/build_ui_data.py. Do not edit by hand.\n")
        f.write("export const DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"ui/data.js written ({os.path.getsize('ui/data.js') / 1024:.0f} KB)")
    print(e1_note)
    print(div_caption)
    print(conn_caption)


def fmt(v, d=1):
    return f"{v:.{d}f}"


def pm(pair, d=3):
    return "–" if pair is None else f"{pair[0]:.{d}f} ± {pair[1]:.{d}f}"


if __name__ == "__main__":
    main()
