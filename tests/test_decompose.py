import numpy as np

from fsq.decompose import cluster_subproblem, partition, repair, stitch
from fsq.formulate import is_feasible
from tests.test_baselines import random_graph


def test_partition_covers_all_nodes_once_and_respects_size():
    for seed in range(6):
        w, e = random_graph(40, 0.12, seed)
        clusters = partition(40, e, max_cluster=8, seed=seed)
        flat = sorted(v for c in clusters for v in c)
        assert flat == list(range(40))
        assert max(len(c) for c in clusters) <= 8


def test_stitch_repair_always_feasible():
    rng = np.random.default_rng(0)
    for seed in range(10):
        w, e = random_graph(30, 0.15, seed)
        clusters = partition(30, e, max_cluster=6, seed=seed)
        local = [rng.integers(0, 2, len(c)) for c in clusters]  # deliberately random
        x = stitch(clusters, local, 30)
        log = []
        y = repair(x, w, e, log)
        assert is_feasible(y, e)
        # maximal: nothing else can be added
        for i in range(30):
            if not y[i]:
                assert any(y[j] for a, b in e for j in ((b,) if a == i else (a,) if b == i else ()))


def test_cluster_subproblem_reindexes():
    w = [0.1, 0.2, 0.3, 0.4]
    sw, se = cluster_subproblem([1, 3], w, [[1, 3], [0, 1]])
    assert sw == [0.2, 0.4] and se == [[0, 1]]
