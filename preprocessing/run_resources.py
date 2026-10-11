"""Step 2: resource estimate for the full problem on one QAOA circuit. Offline only.

    .venv/bin/python preprocessing/run_resources.py

Instances from the real data: 6 h, 24 h and the full 48 h event (268 candidates), plus
the default live instance (68 candidates) for reference. For each:
  * qubits, one-hot (current): one qubit per candidate image (exact);
  * qubits, binary (analytic, step 4 not used): conflict cliques are "at most one" groups;
    a greedy clique cover groups the candidates and a group of size k needs ceil(log2(k+1))
    qubits. Counted only; the extra multi-qubit terms of that encoding are NOT compiled here;
  * two-qubit gates and depth for the one-hot QAOA circuit at p = 1 and p = 2, transpiled
    directly (Qiskit, optimization_level 3, seed 42, basis rz/sx/x/cx) for all-to-all,
    square-grid and heavy-hex coupling maps. No extrapolation: every size is compiled;
  * a simple success model (1 - e2)^n2q at e2 = 1e-2 .. 1e-5, and the e2 at which the model
    gives 0.5. This is a gate-error-only model, not a hardware prediction (it ignores
    single-qubit errors, readout, crosstalk, decoherence over the circuit depth and the
    number of shots needed).

Writes results/e_resources.json.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from fsq.baselines import adjacency  # noqa: E402
from fsq.connectivity import BASIS, grid_shape  # noqa: E402
from fsq.data_load import load_default_instance, load_targets, load_windows, make_instance  # noqa: E402
from fsq.metrics import two_qubit_count  # noqa: E402
from fsq.qaoa import qaoa_circuit  # noqa: E402

E2 = [1e-2, 1e-3, 1e-4, 1e-5]
TARGET_SUCCESS = 0.5


def clique_cover(n, edges):
    """Greedy clique partition: repeatedly take the highest-degree unassigned vertex and add
    unassigned vertices adjacent to every member. Each clique is an at-most-one group."""
    adj = [set(a) for a in adjacency(n, edges)]
    left = set(range(n))
    groups = []
    while left:
        v = max(left, key=lambda u: (len(adj[u] & left), -u))
        g = [v]
        for u in sorted(adj[v] & left, key=lambda u: -len(adj[u] & left)):
            if all(u in adj[x] for x in g):
                g.append(u)
        groups.append(g)
        left -= set(g)
    return groups


def compile_one(qc, cmap):
    from qiskit import transpile
    t = time.perf_counter()
    tc = transpile(qc, basis_gates=BASIS, coupling_map=cmap, optimization_level=3, seed_transpiler=42)
    return {"two_qubit": two_qubit_count(tc), "depth": tc.depth(), "compile_s": round(time.perf_counter() - t, 2)}


def maps(n):
    from qiskit.transpiler import CouplingMap
    d = 3
    while CouplingMap.from_heavy_hex(d).size() < n:
        d += 2
    r, c = grid_shape(n)
    return {"all_to_all": (None, {}), "grid": (CouplingMap.from_grid(r, c), {"shape": [r, c]}),
            "heavy_hex": (CouplingMap.from_heavy_hex(d), {"distance": d, "device_qubits": CouplingMap.from_heavy_hex(d).size()})}


def success(n2q):
    return {f"{e:g}": (1 - e) ** n2q for e in E2}


def e2_needed(n2q):
    return 1 - TARGET_SUCCESS ** (1 / n2q) if n2q else None


def main():
    t0 = time.time()
    T, W = load_targets(), load_windows()
    insts = [("6 h", make_instance(T, W, 6)), ("default live (10 districts, 24 h)", load_default_instance()),
             ("24 h", make_instance(T, W, 24)), ("full event, 48 h", make_instance(T, W, 48))]
    rows = []
    for name, inst in insts:
        w, edges = inst["w"], inst["edges"]
        n = len(w)
        groups = clique_cover(n, edges)
        row = {"instance": name, "candidates": n, "edges": len(edges),
               "qubits_one_hot": n,
               "qubits_binary_analytic": int(sum(math.ceil(math.log2(len(g) + 1)) for g in groups)),
               "clique_groups": len(groups), "largest_group": max(len(g) for g in groups),
               "two_qubit_lower_bound_p1": 2 * len(edges), "by_p": {}}
        for p in (1, 2):
            qc = qaoa_circuit(w, edges, 1.2 * max(w), [0.4] * p, [0.3] * p, thetas=[1.1] * n, measure=False)
            out = {}
            for key, (cm, extra) in maps(n).items():
                r = compile_one(qc, cm)
                r.update(extra)
                r["est_success"] = success(r["two_qubit"])
                r["e2_for_success_0.5"] = e2_needed(r["two_qubit"])
                out[key] = r
                print(f"  {name:<34} n={n:<4} p={p} {key:<10} 2q={r['two_qubit']:<7} depth={r['depth']:<6} "
                      f"e2 for 0.5: {r['e2_for_success_0.5']:.1e}  ({r['compile_s']}s)", flush=True)
            row["by_p"][str(p)] = out
        rows.append(row)
    meta = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "transpile": "Qiskit, optimization_level 3, seed_transpiler 42, basis rz/sx/x/cx",
            "circuit": "one-hot warm-start QAOA cost + warm-start mixer, no measurement; placeholder angles",
            "binary_qubits": "analytic count from a greedy clique cover (not compiled)",
            "success_model": "(1 - e2)^n2q, two-qubit gate errors only - a simple model, not a hardware prediction",
            "target_success": TARGET_SUCCESS}
    json.dump({"meta": meta, "rows": rows}, open("results/e_resources.json", "w"), indent=1)
    print(f"wrote results/e_resources.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
