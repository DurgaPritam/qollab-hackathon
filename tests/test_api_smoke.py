import json
import os

import numpy as np
import pytest

from fsq.formulate import is_feasible

HAVE_DEFAULT = os.path.exists("data/instance_default.json")
needs_default = pytest.mark.skipif(not HAVE_DEFAULT, reason="run preprocessing first")


class CountingBackend:
    """Local simulator that fails like IonQ if a second job is sent."""
    def __init__(self, name="counting_basic_simulator"):
        from qiskit.providers.basic_provider import BasicSimulator
        self.inner, self.jobs, self.name, self.sent = BasicSimulator(), 0, name, []

    def run(self, circuits, **kw):
        self.jobs += 1
        assert self.jobs == 1, "only one job per run is allowed"
        self.sent.append(circuits)
        return self.inner.run(circuits, **kw)


class FakeIonQ(CountingBackend):
    """Behaves like Qollab's IonQ backend: one job per run, ONE circuit per job."""
    def __init__(self, num_qubits=17):
        super().__init__("ionq_simulator")
        self.num_qubits = num_qubits

    def run(self, circuits, **kw):
        if isinstance(circuits, (list, tuple)):
            self.jobs += 1
            raise RuntimeError('IonQAPIError 400: "Only job type ionq.circuit.v1 is supported"')
        return super().run(circuits, **kw)


@needs_default
def test_default_run_end_to_end_local_simulator():
    from fsq.api import run_default
    be = CountingBackend()
    r = json.loads(run_default(be, shots=256))
    assert be.jobs == 1
    assert r["sampling"]["mode"] == "list" and not r["sampling"]["local"]
    assert "counting_basic_simulator: all" in r["hybrid_label"]
    # largest cluster submitted last (the Probabilities panel shows the last circuit)
    sizes = [c.num_qubits for c in be.sent[0]]
    assert sizes[-1] == max(sizes)
    inst = json.load(open("data/instance_default.json"))
    for m in r["methods"].values():
        assert is_feasible(m["x"], inst["edges"])
        assert m["ratio"] <= 1.0 + 1e-9


@needs_default
def test_ionq_packs_clusters_into_one_circuit():
    from fsq.api import run_default
    be = FakeIonQ(num_qubits=17)          # 12 + 5 qubits fit; the rest run locally
    r = json.loads(run_default(be, shots=128))
    s = r["sampling"]
    assert be.jobs == 1 and not isinstance(be.sent[0], list)
    assert s["mode"] == "packed" and s["error"] is None
    assert s["remote_qubits"] <= 17 and len(s["remote"]) == 2
    assert sorted(s["remote"] + s["local"]) == list(range(len(s["remote"]) + len(s["local"])))
    assert r["hybrid_label"].startswith("Hybrid QAOA (IonQ: 2 clusters / 17 qubits; local:")


@needs_default
def test_ionq_failure_is_labelled_local():
    from fsq.api import run_default

    class Broken(FakeIonQ):
        def run(self, *a, **k):
            self.jobs += 1
            raise RuntimeError("simulated IonQ failure")

    be = Broken()
    r = json.loads(run_default(be, shots=64))
    assert be.jobs == 1
    assert r["hybrid_label"] == "Hybrid QAOA (sampled locally - IonQ job failed)"
    assert "ionq" not in r["hybrid_label"].lower().replace("ionq job failed", "")


def test_pack_respects_budget_and_assigns_once():
    from fsq.qaoa import pack_clusters
    sizes = [12, 9, 9, 9, 8, 8, 8, 5]
    for budget in (5, 12, 17, 25, 29, 36, 100):
        packed, offsets = pack_clusters(sizes, budget)
        assert sum(sizes[k] for k in packed) <= budget
        assert len(set(packed)) == len(packed)
        # offsets are contiguous, disjoint qubit ranges
        o = 0
        for k, off in zip(packed, offsets):
            assert off == o
            o += sizes[k]
    assert pack_clusters(sizes, 25)[0] == [0, 1]        # 12 + 9, then nothing else fits but...
    assert pack_clusters(sizes, 26)[0] == [0, 1, 7]     # ...a 5 fits into the remaining space


def test_marginal_counts_bit_order():
    from fsq.qaoa import marginal_counts
    # cluster A: 2 qubits at offset 0; cluster B: 3 qubits at offset 2 (qubit 0 = rightmost char)
    joint = {"10101": 3, "00110": 1, "0x1f": 2}
    assert marginal_counts(joint, 0, 2) == {"01": 3, "10": 1, "11": 2}
    assert marginal_counts(joint, 2, 3) == {"101": 3, "001": 1, "111": 2}


def test_combined_circuit_marginals_match_each_cluster():
    from qiskit import QuantumCircuit
    from qiskit.providers.basic_provider import BasicSimulator
    from fsq.qaoa import combine_circuits, marginal_counts
    a = QuantumCircuit(2); a.x(0)                  # cluster A alone -> "01"
    b = QuantumCircuit(3); b.x(1); b.x(2)          # cluster B alone -> "110"
    qc = combine_circuits([a, b], [0, 2])
    counts = BasicSimulator().run(qc, shots=50).result().get_counts()
    assert marginal_counts(counts, 0, 2) == {"01": 50}
    assert marginal_counts(counts, 2, 3) == {"110": 50}


def test_best_sample_prefers_highest_value_feasible():
    from fsq.hybrid import best_sample
    cl = {"w": [1.0, 0.6, 0.9], "edges": [[0, 1], [1, 2]]}
    X = np.array([[1, 1, 1]] * 5 + [[0, 1, 0], [1, 0, 1], [1, 0, 0]])
    x, ok = best_sample(cl, X)
    assert ok and x.tolist() == [1, 0, 1]
    x, ok = best_sample(cl, np.array([[1, 1, 1]] * 3 + [[1, 1, 0]]))
    assert not ok and x.tolist() == [1, 1, 1]        # most frequent when none is feasible


@needs_default
def test_main_py_entry_point(capsys):
    import main
    main.main(shots=128, charts=False)
    out = capsys.readouterr().out
    assert "Honest verdict" in out and "Total runtime" in out
    assert "Where the gap comes from" in out and "Probabilities panel" in out
