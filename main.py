"""FloodSight Q - Qollab entry point (Python project).

Press Run and pick a backend in "Select QPU". Qollab injects `backend` and calls main().
The default run:
  1. samples every warm-start QAOA cluster in ONE backend job (IonQ allows one job per run),
  2. stitches and repairs the schedule, runs greedy and simulated annealing live,
  3. compares everything with the precomputed exact optimum,
  4. draws the schedule map and the precomputed results (connectivity, diversity, quality).

Data and results ship as project files: data/instance_default.json, data/targets.csv,
data/meta.json, results/e1.json, results/e_div.json, results/e_conn.json.
"""

import json

from fsq.api import run_default

# Qollab ships only data files whose path appears as a plain string in the code, so every
# file the run reads is listed here literally (the fsq modules also use these exact strings).
DATA_FILES = (
    "data/instance_default.json",
    "data/targets.csv",
    "data/meta.json",
    "results/e1.json",
    "results/e_div.json",
    "results/e_conn.json",
)


def main(shots: int = 1024, charts: bool = True):
    be = globals().get("backend")
    out = json.loads(run_default(be, shots=shots))
    if not charts:
        return
    try:
        import matplotlib.pyplot as plt
        from fsq import plots
        from fsq.data_load import load_default_instance, load_targets

        inst = load_default_instance()
        print("\nFigures (the first uses this run's schedule; the rest are precomputed offline results):")
        plots.schedule_map(inst, load_targets(), out["methods"]["hybrid"]["x"],
                           f"This run's hybrid schedule: top 10 districts, first 24 h ({out['backend']})")
        plt.show()
        plots.connectivity_chart(1)
        plt.show()
        plots.diversity_chart()
        plt.show()
        plots.ratio_chart()
        plt.show()
    except Exception as e:  # charts are a bonus; never fail the run over them
        print(f"(charts skipped: {type(e).__name__}: {e})")
