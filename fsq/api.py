"""Entry point used by the Qollab Run button (called from main.py).

Default run (fast, no optimisation in the browser), conditioned large-neighbourhood search:
  1. load data/instance_default.json (candidates, conflicts, precomputed LNS angles)
  2. incumbent = greedy schedule; build neighbourhoods (<= 12 qubits) around it, each
     conditioned on the rest of the schedule, so no conflict outside can be violated
  3. one warm-start QAOA circuit per neighbourhood, sampled with at most ONE backend job
     (IonQ: one circuit per job, so neighbourhoods are packed onto disjoint qubits within
     IONQ_QUBIT_BUDGET; local simulators get the list)
  4. apply each neighbourhood's best compatible sample if it raises the schedule value
  5. optional classical LNS sweeps (exact sub-solver), labelled as classical
  6. greedy and simulated annealing live; compare with the precomputed exact optimum
"""

from __future__ import annotations

import json
import textwrap
import time

import numpy as np

from .baselines import adjacency, exact_bruteforce, greedy, simulated_annealing
from .config import CONFIG
from .data_load import load_default_instance
from .formulate import is_feasible
from .lns import SOLVERS, apply_move, best_compatible, local_problem, release_neighbourhoods, run_lns
from .metrics import approx_ratio, feasible_mask
from .qaoa import counts_to_samples, qaoa_circuit, sample_clusters, scaled_cluster_qubo, warm_thetas

E1_JSON = "results/e1.json"
E_LNS_JSON = "results/e_lns.json"
E_SEEDS_JSON = "results/e_live_seeds.json"

VERDICT = ("Honest verdict: no quantum advantage. The gain over greedy comes from the conditioned LNS "
           "neighbourhood structure: with 1,024 samples per neighbourhood, exact, SA and "
           "uniform-random sub-solvers reach the same schedules, "
           "and full-problem SA is still best. QAOA's measured contribution is sample efficiency "
           "(more probability on each subproblem's optimum than uniform random), not final quality.")


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


def _say(text: str, indent: str = "  ", width: int = 88):
    """Print prose wrapped to the console width (each source line kept as a paragraph)."""
    for para in text.split("\n"):
        lead = para[:len(para) - len(para.lstrip())]
        print(textwrap.fill(para.strip(), width, initial_indent=lead or indent,
                            subsequent_indent=(lead or indent) + "  "))


def _line(char="-", n=72):
    print(char * n)


def hybrid_label(bname: str, info: dict, n_units: int, unit: str = "nbhd") -> str:
    """Label of the QAOA stage row; states where sampling actually happened. The QAOA row
    only ever carries the value after this one sweep, never the classical refinement."""
    head = "+ QAOA LNS sweep, 1 job"
    bname = bname if len(bname) <= 18 else bname[:17] + "~"
    if info["mode"] == "none":
        return "+ QAOA LNS sweep (local, no backend)"
    if info["error"]:
        who = "IonQ" if info["mode"] == "packed" else bname
        return f"+ QAOA LNS sweep (local; {who} job failed)"
    r, l = info["remote"], info["local"]
    if info["mode"] == "packed":
        lab = f"IonQ {len(r)} {unit}/{info['remote_qubits']}q"
        return f"{head} ({lab}; local {len(l)})" if l else f"{head} ({lab})"
    return f"{head} ({bname}, {n_units} {unit}s)"


def gap_insight(e1_path: str = E1_JSON, lns_path: str = E_LNS_JSON) -> str:
    """What the precomputed 21-instance results say, computed from the result files."""
    e1 = json.load(open(e1_path, encoding="utf-8"))
    bench = [r for r in e1["instances"] if r["instance"] != "default"]
    v1 = np.mean([r["methods"]["hybrid_warm_p1"]["ratio"] for r in bench])
    sa = np.mean([r["methods"]["sa"]["ratio"] for r in bench])
    d = json.load(open(lns_path, encoding="utf-8"))
    a, b = d["part_a"], d["part_b"]
    f = lambda k: a[k]["final"][0] if k in a else float("nan")  # noqa: E731
    lines = [f"Precomputed, {len(bench)} instances (ratio to exact optimum): greedy {f('greedy'):.3f}; v1 hybrid "
             f"(independent clusters + repair) {v1:.3f}; LNS {f('qaoa_q12'):.3f}; full-problem SA {sa:.3f}.",
             f"  The gain from {v1:.3f} to {f('qaoa_q12'):.3f} comes from the LNS neighbourhood structure: with 1,024 "
             f"samples per neighbourhood, QAOA {f('qaoa_q12'):.3f}, exact {f('exact_q12'):.3f}, SA {f('sa_q12'):.3f} "
             f"and uniform-random-feasible {f('random_q12'):.3f} sub-solvers give the same result."]
    if "exact_q16" in a and f("exact_q16") > f("exact_q12") + 5e-4:
        lines.append(f"  16-qubit neighbourhoods (classical sub-solvers) reach {f('exact_q16'):.3f}: "
                     "more qubits per neighbourhood gives better schedules.")
    if "qaoa_q12_b32" in a:
        lines.append(f"  With a budget of 32 per neighbourhood: QAOA {f('qaoa_q12_b32'):.3f} (32 shots), "
                     f"SA {f('sa_q12_b32'):.3f} (32 restarts x 100 sweeps), uniform random feasible "
                     f"{f('random_q12_b32'):.3f} (32 draws), uniform random bitstrings {f('random_all_q12_b32'):.3f} "
                     f"(32 draws), exact ceiling {f('exact_q12_b32'):.3f}. Costs are in different units.")
    o = b["qaoa_over_random"]
    lines.append(f"  QAOA's measured contribution is sample efficiency: x{o['median_ratio']:.0f} (median) more "
                 f"probability on each subproblem's optimum than uniform random feasible "
                 f"({o['qaoa_higher']}/{o['n']} subproblems); one SA restart (100 sweeps) finds it with "
                 f"probability {b['p_sa']['mean']:.2f}, one QAOA shot {b['p_qaoa']['mean']:.2f}.")
    return "\n".join(lines)


def seed_insight(path: str = E_SEEDS_JSON) -> str:
    s = json.load(open(path, encoding="utf-8"))["summary"]
    q, c = s["quantum_sweep"], s["after_classical"]
    others = [v for v in q["values"] if v < 1 - 1e-9]
    return (f"Seed {s['project_seed']} is the project's fixed seed. Across {s['n_seeds']} seeds the one-job "
            f"QAOA-sampled sweep reaches the optimum on {q['optimal']} and "
            f"{'/'.join(f'{v:.3f}' for v in others)} on the rest; with the classical sweeps "
            f"{c['optimal']}/{s['n_seeds']} reach 1.000.")


def _neighbourhood_params(inst, hoods, adj):
    """Precomputed angles if the live neighbourhoods match the offline ones, else a fallback."""
    pre = inst.get("lns", {}).get("neighbourhoods", [])
    by_key = {tuple(h["free"]): h for h in pre}
    if pre and all(tuple(f) in by_key for f in hoods):
        return [by_key[tuple(f)] for f in hoods], True
    g = float(np.median([h["gammas"][0] for h in pre])) if pre else 0.5
    b = float(np.median([h["betas"][0] for h in pre])) if pre else 0.3
    inc = inst.get("lns", {}).get("incumbent", [])
    out = []
    for f in hoods:   # warm start from the incumbent: selected -> 0.75, others -> 0.25 (after clipping)
        c = np.array([1.0 if inc and inc[v] else 0.0 for v in f])
        out.append({"free": f, "gammas": [g], "betas": [b], "thetas": list(warm_thetas(c, CONFIG["eps"]))})
    return out, False


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
    adj = adjacency(n, edges)
    print(f"Instance: {inst['description']}")
    print(f"  {n} candidate images, {len(edges)} conflicts")
    print(f"Backend: {bname}   shots: {shots}")

    # ---- incumbent and neighbourhoods -----------------------------------------------------
    _line()
    print("[1/4] Incumbent and neighbourhoods (classical)")
    t = time.perf_counter(); xg = greedy(w, edges); t_g = time.perf_counter() - t
    hoods = release_neighbourhoods(xg, adj, CONFIG["lns_max_free"], CONFIG["lns_seed"])
    params, matched = _neighbourhood_params(inst, hoods, adj)
    print(f"  greedy schedule: {int(xg.sum())} images, ratio {approx_ratio(np.dot(xg, w), opt):.3f}")
    print(f"  {len(hoods)} neighbourhoods around it (sizes {[len(f) for f in hoods]}, <= {CONFIG['lns_max_free']} "
          f"qubits), each conditioned on the rest of the schedule")
    if not matched:
        print("  (neighbourhoods differ from the precomputed ones: using median angles - expect weaker samples)")

    # ---- quantum: at most one job -------------------------------------------------------------
    _line()
    print("[2/4] QAOA sampling of every neighbourhood (precomputed warm-start angles, p=1), at most one job")
    t = time.perf_counter()
    circuits, locals_ = [], []
    for f, prm in zip(hoods, params):
        wl, el = local_problem(f, w, adj)
        ws, P = scaled_cluster_qubo(wl, CONFIG["penalty_mult"])
        circuits.append(qaoa_circuit(ws, el, P, prm["gammas"], prm["betas"], thetas=prm["thetas"]))
        locals_.append((wl, el))
    counts, info = sample_clusters(backend, circuits, shots, seed, CONFIG["ionq_qubit_budget"],
                                   unit="neighbourhood")
    qsizes = [c.num_qubits for c in circuits]
    if info["mode"] == "packed" and not info["error"]:
        k = len(info["remote"])
        print(f"  IonQ (one circuit per job): {k} neighbourhood{'' if k == 1 else 's'} in ONE "
              f"{info['remote_qubits']}-qubit circuit (budget {info['budget']} qubits = min(backend size, "
              f"IONQ_QUBIT_BUDGET)), ONE job")
    elif info["mode"] == "list" and not info["error"]:
        print(f"  one job on {bname} with {len(circuits)} circuits (one per neighbourhood)")
    if info["error"]:
        print(f"  backend job FAILED ({info['error']}); nothing else was sent to the backend")
    if info["remote"]:
        print("  sampled on backend: " + ", ".join(f"N{j + 1} ({qsizes[j]}q)" for j in sorted(info["remote"])))
    if info["local"]:
        print("  sampled locally:    " + ", ".join(f"N{j + 1} ({qsizes[j]}q)" for j in sorted(info["local"])))

    x = xg.copy()
    hits, moves = 0, []
    for j, (f, (wl, el), c) in enumerate(zip(hoods, locals_, counts)):
        X = counts_to_samples(c, len(f))
        sub_opt = float(np.dot(exact_bruteforce(wl, el), wl))
        vals = np.where(feasible_mask(X, el), X @ np.asarray(wl), -1.0)
        hits += int(vals.max() >= sub_opt - 1e-12)
        new = best_compatible(f, f, X, x, w, adj)
        if new is not None and apply_move(x, f, new, w):
            moves.append((j, "backend" if j in info["remote"] else "local"))
    x_q = x.copy()
    t_q = time.perf_counter() - t
    print(f"  neighbourhoods whose samples contain their optimum: {hits} of {len(hoods)}")
    print(f"  improving moves applied: {len(moves)}" +
          (" (" + ", ".join(f"N{j + 1} from {src} samples" for j, src in moves) + ")" if moves else ""))
    print(f"  schedule after the QAOA-sampled sweep: ratio {approx_ratio(np.dot(x_q, w), opt):.3f} "
          f"(greedy {approx_ratio(np.dot(xg, w), opt):.3f})")

    # ---- classical refinement ---------------------------------------------------------------
    _line()
    k_cl = CONFIG["lns_classical_sweeps"]
    print(f"[3/4] Classical refinement: {k_cl} more LNS sweeps with an exact sub-solver (classical, labelled)")
    t = time.perf_counter()
    rc = run_lns(w, edges, "release", x_q, SOLVERS["exact"], sweeps=k_cl, budget=1, seed=seed + 1000,
                 max_free=CONFIG["lns_max_free"])
    x_c = rc["x"]
    t_c = time.perf_counter() - t
    print(f"  {rc['moves']} further improving moves; ratio {approx_ratio(np.dot(x_c, w), opt):.3f}")

    # ---- classical baselines ------------------------------------------------------------------
    t = time.perf_counter()
    Xs = simulated_annealing(w, edges, restarts=shots, seed=seed, sweeps=100)
    vals = Xs @ np.asarray(w)
    vals[~feasible_mask(Xs, edges)] = -1
    xs = Xs[int(np.argmax(vals))]
    t_s = time.perf_counter() - t
    x_exact = np.asarray(inst["exact"]["x"], dtype=np.int8)

    # ---- results ---------------------------------------------------------------------------
    _line()
    print("[4/4] Results (value = priority x people x urgency x image quality; 1.000 = exact optimum)")
    hlabel = hybrid_label(bname, info, len(circuits))
    rows = [("exact", "Exact optimum (MILP, precomputed)", x_exact, None),
            ("sa", f"Simulated annealing ({shots} restarts)", xs, t_s),
            (None, "Hybrid LNS, stage by stage (nbhd = neighbourhood):", None, None),
            ("greedy", "  Greedy incumbent", xg, t_g),
            ("hybrid_quantum_sweep", "  " + hlabel, x_q, t_q),
            ("hybrid", f"  + {k_cl} classical LNS sweeps (exact)", x_c, t_c)]
    width = max(40, max(len(r[1]) for r in rows if r[0]) + 1)
    print(f"  {'method':<{width}}{'ratio':>7}{'images':>8}{'feasible':>10}{'time s':>8}")
    out_methods = {}
    for key, name, xx, rt in rows:
        if key is None:
            print(f"  {name}")
            continue
        val = float(np.dot(xx, w))
        print(f"  {name:<{width}}{approx_ratio(val, opt):>7.3f}{int(np.sum(xx)):>8d}"
              f"{str(is_feasible(xx, edges)):>10}{('-' if rt is None else f'{rt:.2f}'):>8}")
        out_methods[key] = {"label": name.strip(), "ratio": approx_ratio(val, opt), "x": [int(v) for v in xx],
                            "runtime_s": rt}
    try:
        _say(seed_insight())
    except Exception as e:
        print(f"  (seed summary unavailable: {type(e).__name__})")
    _line()
    try:
        _say("What the benchmarks say: " + gap_insight(), indent="")
    except Exception as e:
        print(f"(benchmark insight unavailable: {type(e).__name__})")
    if info["panel"]:
        _say(f"Probabilities panel: shows {info['panel']}. All neighbourhoods are summarised above.", indent="")
    else:
        print("Probabilities panel: no circuit ran on a backend in this run (all sampled locally).")
    _line()
    _say(VERDICT, indent="")
    print(f"Total runtime: {time.perf_counter() - t_all:.1f} s")
    _line("=")
    return json.dumps({"backend": bname, "shots": shots, "sampling": info, "hybrid_label": hlabel,
                       "methods": out_methods, "neighbourhoods": [len(f) for f in hoods],
                       "precomputed_angles": matched, "subproblem_hits": hits,
                       "moves": [{"neighbourhood": j + 1, "source": src} for j, src in moves],
                       "runtime_s": time.perf_counter() - t_all})


if __name__ == "__main__":
    run_default(None)
