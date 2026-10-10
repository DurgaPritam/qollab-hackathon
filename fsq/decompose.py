"""Decomposition into qubit-sized clusters, then stitching and repair."""

from __future__ import annotations

import numpy as np

from .baselines import _components, adjacency


def connected_components(n: int, edges: list) -> list[list[int]]:
    """Connected components of the conflict graph (sorted node lists)."""
    return _components(n, edges)


def _bisect(nodes: list[int], edges: list, seed: int) -> tuple[list[int], list[int]]:
    """Kernighan-Lin bisection (networkx) with a spectral fallback."""
    node_set = set(nodes)
    sub = [(i, j) for i, j in edges if i in node_set and j in node_set]
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(sub)
        a, b = nx.algorithms.community.kernighan_lin_bisection(G, seed=seed)
        return sorted(a), sorted(b)
    except ImportError:
        idx = {v: k for k, v in enumerate(nodes)}
        L = np.zeros((len(nodes), len(nodes)))
        for i, j in sub:
            a, b = idx[i], idx[j]
            L[a, b] = L[b, a] = -1
        L[np.diag_indices_from(L)] = -L.sum(axis=1)
        _, vecs = np.linalg.eigh(L)
        order = np.argsort(vecs[:, 1], kind="stable")
        half = len(nodes) // 2
        return sorted(nodes[k] for k in order[:half]), sorted(nodes[k] for k in order[half:])


def partition(n: int, edges: list, max_cluster: int, seed: int = 0) -> list[list[int]]:
    """Components, with any component larger than max_cluster split by repeated bisection.

    Every node appears in exactly one cluster.
    """
    out = []
    stack = connected_components(n, edges)
    while stack:
        comp = stack.pop()
        if len(comp) <= max_cluster:
            out.append(comp)
        else:
            a, b = _bisect(comp, edges, seed)
            # a bisection can leave a part disconnected; split it into components again
            for part in (a, b):
                ps = set(part)
                pe = [(i, j) for i, j in edges if i in ps and j in ps]
                idx = {v: k for k, v in enumerate(part)}
                for c in connected_components(len(part), [[idx[i], idx[j]] for i, j in pe]):
                    stack.append([part[k] for k in c])
    out = merge_clusters(out, edges, max_cluster)
    out.sort(key=lambda c: (-len(c), c[0]))
    return out


def merge_clusters(clusters: list[list[int]], edges: list, max_cluster: int) -> list[list[int]]:
    """Repeatedly merge the pair of clusters joined by the most cut edges, while the merged
    size stays <= max_cluster. Bisection overshoots (a 13-node part becomes 6 + 7); this
    gives the qubits back and removes cut edges, so less is left for repair."""
    clusters = [list(c) for c in clusters]
    while True:
        where = {v: k for k, c in enumerate(clusters) for v in c}
        links: dict[tuple[int, int], int] = {}
        for i, j in edges:
            a, b = where[i], where[j]
            if a != b:
                key = (min(a, b), max(a, b))
                links[key] = links.get(key, 0) + 1
        best = None
        for (a, b), cnt in links.items():
            if len(clusters[a]) + len(clusters[b]) <= max_cluster:
                cand = (cnt, -(len(clusters[a]) + len(clusters[b])), -a, -b)
                if best is None or cand > best[0]:
                    best = (cand, a, b)
        if best is None:
            return clusters
        _, a, b = best
        clusters[a] = sorted(clusters[a] + clusters[b])
        del clusters[b]


def cluster_subproblem(cluster: list[int], w: list[float], edges: list):
    """Local weights and edges (re-indexed 0..k-1) for one cluster."""
    idx = {v: k for k, v in enumerate(cluster)}
    sw = [w[v] for v in cluster]
    se = [[idx[i], idx[j]] for i, j in edges if i in idx and j in idx]
    return sw, se


def cut_edges(clusters: list[list[int]], edges: list) -> list[list[int]]:
    """Edges whose endpoints lie in different clusters."""
    where = {}
    for c, nodes in enumerate(clusters):
        for v in nodes:
            where[v] = c
    return [[i, j] for i, j in edges if where[i] != where[j]]


def repair(x, w: list[float], edges: list, log: list | None = None) -> np.ndarray:
    """Make x feasible, then greedily fill.

    1. While any conflict is violated, drop the lowest-value selected node that sits on a
       violated edge.
    2. Add unused candidates in order of value density w/(1+deg) if they fit.
    Every action is appended to `log` (when given) as a dict.
    """
    x = np.array(x, dtype=np.int8).copy()
    n = len(w)
    adj = adjacency(n, edges)
    while True:
        bad = {v for i, j in edges if x[i] and x[j] for v in (i, j)}
        if not bad:
            break
        v = min(bad, key=lambda k: (w[k], k))
        x[v] = 0
        if log is not None:
            log.append({"action": "drop", "candidate": int(v), "value": float(w[v])})
    order = sorted(range(n), key=lambda i: (-w[i] / (1 + len(adj[i])), i))
    for i in order:
        if not x[i] and not any(x[j] for j in adj[i]):
            x[i] = 1
            if log is not None:
                log.append({"action": "add", "candidate": int(i), "value": float(w[i])})
    return x


def stitch(clusters: list[list[int]], local_solutions: list, n: int) -> np.ndarray:
    """Place each cluster's local 0/1 solution into a global vector."""
    x = np.zeros(n, dtype=np.int8)
    for nodes, xs in zip(clusters, local_solutions):
        for k, v in enumerate(nodes):
            x[v] = xs[k]
    return x
