"""E4: one simple noise model vs ideal sampling, on the real-event clusters. Offline only.

    .venv/bin/python preprocessing/run_noise.py      (needs qiskit-aer; run after run_experiments.py)

Noise model (stated, simple, not a calibrated device model): depolarizing error 0.1% on
every 1-qubit gate, 0.5% on every CX, 1% symmetric readout error. Circuits are compiled
all-to-all (trapped-ion style), basis rz/sx/x/cx.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from fsq.baselines import exact  # noqa: E402
from fsq.config import CONFIG  # noqa: E402
from fsq.data_load import load_targets, load_windows, make_instance  # noqa: E402
from fsq.hybrid import assemble, cluster_circuit, prepare_clusters  # noqa: E402
from fsq.metrics import approx_ratio, sample_stats  # noqa: E402
from fsq.qaoa import counts_to_samples  # noqa: E402

E1Q, E2Q, EREAD = 0.001, 0.005, 0.01


def main():
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(E1Q, 1), ["rz", "sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(E2Q, 2), ["cx"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - EREAD, EREAD], [EREAD, 1 - EREAD]]))
    sims = {"ideal": AerSimulator(seed_simulator=11), "noisy": AerSimulator(noise_model=nm, seed_simulator=11)}

    inst = make_instance(load_targets(), load_windows(), instance_id="real")
    clusters = prepare_clusters(inst, seed=CONFIG["seed"])
    e1 = json.load(open("results/e1.json"))
    real = [r for r in e1["instances"] if r["instance"] == "real"][0]
    params = [c["params"] for c in real["clusters"]]
    opt = real["exact"]["value"]
    shots = CONFIG["shots"]
    out = {}
    for name, sim in sims.items():
        samples, stats = [], []
        for cl, prm in zip(clusters, params):
            k = len(cl["nodes"])
            if k == 1:
                samples.append(np.ones((shots, 1), dtype=np.int8))
                continue
            qc = transpile(cluster_circuit(cl, prm), basis_gates=["rz", "sx", "x", "cx"], optimization_level=3,
                           seed_transpiler=42)
            X = counts_to_samples(sim.run(qc, shots=shots).result().get_counts(), k)
            samples.append(X)
            if k >= 4:
                copt = float(np.dot(exact(cl["w"], cl["edges"])[0], cl["w"]))
                stats.append(sample_stats(X, cl["w"], cl["edges"], copt))
        res = assemble(inst, clusters, samples)
        out[name] = {"ratio": approx_ratio(res["value"], opt),
                     "feasible_rate": float(np.mean([s["feasible_rate"] for s in stats])),
                     "n_good": float(np.mean([s["n_good"] for s in stats])),
                     "best_ratio": float(np.mean([s["best_ratio"] for s in stats])),
                     "repairs": len(res["repair_log"]), "clusters": len(stats)}
        print(name, out[name])
    rows = [[k, f"{v['ratio']:.3f}", f"{v['feasible_rate']:.3f}", f"{v['n_good']:.1f}", f"{v['best_ratio']:.3f}", v["repairs"]]
            for k, v in out.items()]
    json.dump({"model": {"depol_1q": E1Q, "depol_2q": E2Q, "readout": EREAD}, "shots": shots, "results": out,
               "title": "Noise (E4)",
               "text": (f"Real event, warm-start p=1, {shots} shots per cluster on Qiskit Aer. Simple model: depolarizing "
                        f"{E1Q:.1%} per 1-qubit gate, {E2Q:.1%} per CX, {EREAD:.0%} readout. Per-cluster columns average "
                        f"the {out['ideal']['clusters']} clusters with at least 4 candidates."),
               "headers": ["simulation", "full schedule ratio", "cluster feasible rate", "cluster distinct ≥95% schedules",
                           "cluster best sample ratio", "repair actions"],
               "rows": rows}, open("results/e4.json", "w"))


if __name__ == "__main__":
    main()
