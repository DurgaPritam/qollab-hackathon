"""Precompute the live run's LNS neighbourhoods and QAOA angles into data/instance_default.json.

    .venv/bin/python preprocessing/build_live_lns.py    (run after run_experiments.py)

The live run rebuilds the same neighbourhoods (greedy incumbent, fixed seed) and checks
they match; the browser then only builds circuits and samples, no optimisation.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from fsq.baselines import adjacency, greedy  # noqa: E402
from fsq.config import CONFIG  # noqa: E402
from fsq.lns import local_problem, qaoa_distribution, release_neighbourhoods  # noqa: E402

PATH = "data/instance_default.json"


def build(inst: dict) -> dict:
    w, edges = inst["w"], inst["edges"]
    adj = adjacency(len(w), edges)
    x0 = greedy(w, edges)
    hoods = release_neighbourhoods(x0, adj, CONFIG["lns_max_free"], CONFIG["lns_seed"])
    out = []
    for free in hoods:
        wl, el = local_problem(free, w, adj)
        _, prm, thetas = qaoa_distribution(wl, el)
        out.append({"free": free, "gammas": prm["gammas"], "betas": prm["betas"],
                    "thetas": [float(t) for t in thetas]})
    return {"seed": CONFIG["lns_seed"], "max_free": CONFIG["lns_max_free"],
            "incumbent": [int(v) for v in x0], "neighbourhoods": out}


def main():
    inst = json.load(open(PATH))
    inst["lns"] = build(inst)
    json.dump(inst, open(PATH, "w"))
    sizes = [len(h["free"]) for h in inst["lns"]["neighbourhoods"]]
    print(f"{PATH}: {len(sizes)} LNS neighbourhoods, sizes {sizes}")


if __name__ == "__main__":
    main()
