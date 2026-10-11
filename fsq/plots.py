"""Matplotlib figures, shared by the Qollab run (plt.show -> SVG in the console) and the
offline script that saves PNGs for the project page.

Colours: a validated colour-blind-safe categorical order (blue, orange, aqua) and a single-
hue blue ramp for priority.
"""

from __future__ import annotations

import json
import math

import numpy as np

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
PRIORITY = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

E1_JSON = "results/e1.json"
E_DIV_JSON = "results/e_div.json"
E_CONN_JSON = "results/e_conn.json"
META_JSON = "data/meta.json"


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def schedule_map(inst: dict, targets: list[dict], x, title: str):
    """Assam outline, districts sized by people affected and coloured by priority;
    districts imaged in schedule x get an orange ring."""
    import matplotlib.pyplot as plt

    meta = json.load(open(META_JSON, encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for ring in meta["region_rings"]:
        r = np.asarray(ring)
        ax.fill(r[:, 0], r[:, 1], color="#f0efec", edgecolor="#c3c2b7", linewidth=1)
    imaged = {inst["candidates"][i]["target_id"] for i, v in enumerate(x) if v}
    pmax = max(t["population"] for t in targets)
    for t in sorted(targets, key=lambda t: -t["population"]):
        size = 30 + 600 * t["population"] / pmax
        ax.scatter(t["lon"], t["lat"], s=size, color=PRIORITY[t["priority"] - 1], edgecolor="white",
                   linewidth=1.2, zorder=3)
        if t["target_id"] in imaged:
            ax.scatter(t["lon"], t["lat"], s=size * 1.9, facecolor="none", edgecolor=ORANGE, linewidth=2, zorder=4)
    placed = []   # label the biggest districts first, skipping any too close to one already labelled
    for t in sorted(targets, key=lambda t: -t["population"]):
        if all(abs(t["lon"] - a) > 0.75 or abs(t["lat"] - b) > 0.3 for a, b in placed):
            placed.append((t["lon"], t["lat"]))
            ax.annotate(t["name"], (t["lon"], t["lat"]), xytext=(8, 5), textcoords="offset points", fontsize=8,
                        color=INK, bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.7})
    ax.set_aspect(1 / math.cos(math.radians(26)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    handles = [plt.Line2D([], [], marker="o", ls="", color=PRIORITY[p - 1], markersize=7, label=f"priority {p}")
               for p in range(1, 6)]
    handles.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=ORANGE,
                              markeredgewidth=2, markersize=9, label="imaged in this schedule"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def connectivity_chart(p: int = 1):
    """Two-qubit gates vs qubits for all-to-all, grid and heavy-hex (precomputed E-CONN)."""
    import matplotlib.pyplot as plt

    rows = [r for r in json.load(open(E_CONN_JSON))["rows"] if r["p"] == p]
    n = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for key, label, c in (("all_to_all", "All-to-all (trapped ion)", BLUE), ("grid", "Square grid", ORANGE),
                          ("heavy_hex", "Heavy-hex", AQUA)):
        y = [r[key]["two_qubit"] for r in rows]
        ax.plot(n, y, color=c, linewidth=2, marker="o", markersize=5, label=label)
    last = rows[-1]
    ax.annotate(f"{last['all_to_all']['two_qubit']}", (last["n"], last["all_to_all"]["two_qubit"]),
                xytext=(6, -4), textcoords="offset points", fontsize=9, color=INK)
    ax.annotate(f"{last['grid']['two_qubit']} / {last['heavy_hex']['two_qubit']}", (last["n"], last["grid"]["two_qubit"]),
                xytext=(6, 2), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xticks(n)
    _style(ax, f"Two-qubit gates after compiling QAOA (p={p}) for real flood conflict subgraphs")
    ax.set_xlabel("qubits (candidate images in the cluster)", color=MUTED)
    ax.set_ylabel("two-qubit gates after transpile", color=MUTED)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def diversity_chart():
    """QAOA vs SA at equal sample budget: cluster-level feasible rate, distinct near-optimal
    schedules and diversity, plus whole-schedule near-optimal count."""
    import matplotlib.pyplot as plt

    d = [i for i in json.load(open(E_DIV_JSON))["instances"] if i["instance"] != "default"]
    cl = [c for i in d for c in i["clusters"]]
    panels = [
        ("Feasible rate\n(per cluster)", [c["qaoa"]["feasible_rate"] for c in cl], [c["sa"]["feasible_rate"] for c in cl]),
        ("Distinct ≥95% schedules\n(per cluster)", [c["qaoa"]["n_good"] for c in cl], [c["sa"]["n_good"] for c in cl]),
        ("Distinct ≥95% schedules\n(whole schedule)", [i["pipeline"]["hybrid"]["n_good"] for i in d],
         [i["pipeline"]["sa_raw"]["n_good"] for i in d]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.6))
    for ax, (title, q, s) in zip(axes, panels):
        means = [np.mean(q), np.mean(s)]
        sds = [np.std(q), np.std(s)]
        bars = ax.bar([0, 1], means, yerr=sds, color=[BLUE, ORANGE], width=0.6, capsize=3,
                      error_kw={"ecolor": MUTED, "linewidth": 1})
        for b, m in zip(bars, means):   # value inside the bar, clear of the error bar
            ax.text(b.get_x() + b.get_width() / 2, m / 2, f"{m:.2f}" if m < 10 else f"{m:.0f}",
                    ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        ax.set_xticks([0, 1], ["QAOA", "SA"])
        _style(ax, title)
    fig.suptitle(f"Same QUBO, same budget (1,024 samples): {len(cl)} clusters, {len(d)} instances; mean ± sd",
                 fontsize=10, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def ratio_chart(lns_path: str = "results/e_lns.json"):
    """Approximation ratio per method over the 21 benchmark instances."""
    import matplotlib.pyplot as plt

    e = [i for i in json.load(open(E1_JSON))["instances"] if i["instance"] != "default"]
    lns = json.load(open(lns_path))["part_a"]["qaoa_q12"]["per_instance_final"]
    cols = [("Greedy", [i["methods"]["greedy"]["ratio"] for i in e]),
            ("v1 hybrid QAOA\n(clusters + repair)", [i["methods"]["hybrid_warm_p1"]["ratio"] for i in e]),
            ("LNS + QAOA\n(12 qubits)", [lns[i["instance"]] for i in e]),
            ("Simulated annealing\n(full problem)", [i["methods"]["sa"]["ratio"] for i in e])]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for k, (label, v) in enumerate(cols):
        ax.scatter(np.full(len(v), k) + np.linspace(-0.15, 0.15, len(v)), v, s=18, color=BLUE, alpha=0.7, zorder=3)
        ax.plot([k - 0.25, k + 0.25], [np.mean(v)] * 2, color=INK, linewidth=2, zorder=4)
        ax.annotate(f"mean {np.mean(v):.3f}", (k, np.mean(v)), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(cols)), [l for l, _ in cols])
    ax.set_ylabel("value / exact optimum", color=MUTED)
    _style(ax, f"Schedule quality on {len(e)} instances (dots = instances, bar = mean)")
    fig.text(0.01, 0.01, "LNS with exact, SA or uniform-random sub-solvers gives the same 0.993: "
             "the gain comes from the neighbourhood structure.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def resource_chart(path: str = "results/e_resources.json"):
    """Step 2: one-circuit QAOA cost for the real instances (precomputed, directly compiled)."""
    import matplotlib.pyplot as plt

    d = json.load(open(path))
    rows = sorted(d["rows"], key=lambda r: r["candidates"])
    topo = (("all_to_all", "All-to-all (trapped ion)", BLUE), ("grid", "Square grid", ORANGE),
            ("heavy_hex", "Heavy-hex", AQUA))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    n = [r["candidates"] for r in rows]
    for key, label, c in topo:
        y = [r["by_p"]["1"][key]["two_qubit"] for r in rows]
        a1.plot(n, y, color=c, linewidth=2, marker="o", markersize=6, label=label)
        a1.annotate(f"{y[-1]:,}", (n[-1], y[-1]), xytext=(6, -3), textcoords="offset points", fontsize=9, color=INK)
    a1.set_yscale("log")
    a1.set_xticks(n)
    a1.set_xlim(right=n[-1] * 1.18)
    _style(a1, "Two-qubit gates, one QAOA circuit (p=1)")
    a1.set_xlabel("qubits (one-hot: candidate images)", color=MUTED)
    a1.set_ylabel("two-qubit gates after transpile (log)", color=MUTED)
    a1.legend(frameon=False, fontsize=9)

    full = rows[-1]
    e2 = np.logspace(-5, -2, 61)
    for key, label, c in topo:
        g = full["by_p"]["1"][key]["two_qubit"]
        a2.plot(e2, (1 - e2) ** g, color=c, linewidth=2, label=label)
    a2.axhline(d["meta"]["target_success"], color=MUTED, linewidth=1, linestyle="--")
    a2.set_xscale("log")
    a2.set_ylim(0, 1.02)
    _style(a2, f"Full event ({full['candidates']} qubits): simple model (1-e2)^gates")
    a2.set_xlabel("two-qubit gate error e2 (log)", color=MUTED)
    a2.set_ylabel("estimated circuit success", color=MUTED)
    fig.text(0.01, 0.01, "Gate-error-only model, not a hardware prediction. Compiled with Qiskit "
             "(level 3); no extrapolation.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig
