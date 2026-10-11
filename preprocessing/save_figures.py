"""Save the project-page figures to docs/figures/*.png (same code as the live run).

    .venv/bin/python preprocessing/save_figures.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from fsq import plots  # noqa: E402
from fsq.data_load import load_targets, load_windows, make_instance  # noqa: E402


def main():
    os.makedirs("docs/figures", exist_ok=True)
    e1 = json.load(open("results/e1.json"))
    real = [r for r in e1["instances"] if r["instance"] == "real"][0]
    inst = make_instance(load_targets(), load_windows(), instance_id="real")
    figs = {
        "map_exact.png": plots.schedule_map(inst, load_targets(), real["exact"]["x"],
                                            "Exact optimal schedule, real event: 25 districts, 48 h, 8 SAR satellites"),
        "connectivity_p1.png": plots.connectivity_chart(1),
        "connectivity_p2.png": plots.connectivity_chart(2),
        "diversity.png": plots.diversity_chart(),
        "ratios.png": plots.ratio_chart(),
        "resources.png": plots.resource_chart(),
    }
    for name, fig in figs.items():
        fig.savefig(f"docs/figures/{name}", dpi=150, facecolor="white")
        print("wrote docs/figures/" + name)


if __name__ == "__main__":
    main()
