"""Connectivity comparison: QAOA circuits compiled for all-to-all vs grid vs heavy-hex.

Trapped-ion processors (such as IonQ's) connect every qubit pair, so a dense conflict
graph needs no SWAPs. Nearest-neighbour layouts need SWAPs, each costing three CX gates.
"""

from __future__ import annotations

import math

from .formulate import graph_density
from .metrics import est_success, two_qubit_count
from .qaoa import qaoa_circuit

BASIS = ["rz", "sx", "x", "cx"]


def grid_shape(n: int) -> tuple[int, int]:
    """Smallest near-square grid with at least n sites."""
    r = math.ceil(math.sqrt(n))
    c = math.ceil(n / r)
    return r, c


def compile_stats(circuit, coupling_map, seed: int = 42) -> dict:
    """Transpile with optimization_level=3 and report 2q gate count and depth."""
    from qiskit import transpile
    t = transpile(circuit, basis_gates=BASIS, coupling_map=coupling_map,
                  optimization_level=3, seed_transpiler=seed)
    n2 = two_qubit_count(t)
    return {"two_qubit": n2, "depth": t.depth(), "est_success": est_success(n2)}


def compare(w: list[float], edges: list, P: float, p: int, seed: int = 42,
            heavy_hex: bool = True) -> dict:
    """All-to-all vs grid (and heavy-hex) for one instance at QAOA depth p.

    Angles are fixed placeholders (0.4, 0.3): gate counts do not depend on their values,
    except that exact zeros could be optimised away, which these avoid.
    """
    from qiskit.transpiler import CouplingMap

    n = len(w)
    qc = qaoa_circuit(w, edges, P, [0.4] * p, [0.3] * p, thetas=[1.1] * n, measure=False)
    r, c = grid_shape(n)
    out = {"n": n, "edges": len(edges), "density": graph_density(n, edges), "p": p,
           "all_to_all": compile_stats(qc, None, seed),
           "grid": dict(compile_stats(qc, CouplingMap.from_grid(r, c), seed), shape=[r, c])}
    if heavy_hex:
        d = 3
        while CouplingMap.from_heavy_hex(d).size() < n:
            d += 2
        out["heavy_hex"] = dict(compile_stats(qc, CouplingMap.from_heavy_hex(d), seed), distance=d)
    return out
