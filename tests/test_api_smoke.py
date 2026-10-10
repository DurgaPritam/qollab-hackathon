import json
import os

import pytest

from fsq.formulate import is_feasible


@pytest.mark.skipif(not os.path.exists("data/instance_default.json"), reason="run preprocessing first")
def test_default_run_end_to_end_one_job():
    from qiskit.providers.basic_provider import BasicSimulator

    class CountingBackend:
        """Wraps a simulator and fails like IonQ if a second job is sent."""
        def __init__(self):
            self.inner, self.jobs = BasicSimulator(), 0
            self.name = "counting-basic-simulator"

        def run(self, circuits, **kw):
            self.jobs += 1
            assert self.jobs == 1, "only one job per run is allowed"
            return self.inner.run(circuits, **kw)

    be = CountingBackend()
    r = json.loads(__import__("fsq.api", fromlist=["run_default"]).run_default(be, shots=256))
    assert be.jobs == 1
    assert all(w == "backend" for w in r["where"])
    inst = json.load(open("data/instance_default.json"))
    for m in r["methods"].values():
        assert is_feasible(m["x"], inst["edges"])
        assert m["ratio"] <= 1.0 + 1e-9


def test_failing_backend_falls_back_locally_without_second_job():
    from fsq import api

    class Broken:
        name = "broken"
        calls = 0

        def run(self, *a, **k):
            Broken.calls += 1
            raise RuntimeError("simulated backend failure")

    if not os.path.exists("data/instance_default.json"):
        pytest.skip("run preprocessing first")
    r = json.loads(api.run_default(Broken(), shots=128))
    assert Broken.calls == 1
    assert "sampled locally" in r["note"]
