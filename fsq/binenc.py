"""Qubit-efficient (binary) encoding of a scheduling subproblem (Step 4, offline analysis).

One-hot: one qubit per candidate image. Binary: candidates that pairwise conflict form an
"at most one" group (a clique of the conflict graph). A group of k candidates is encoded in
m = ceil(log2(k + 1)) qubits: code 0 = none chosen, codes 1..k = that candidate, codes
above k are invalid and penalised. Conflicts between groups become terms on the qubits of
both groups.

The cost is built exactly as a diagonal over all 2^Q encoded bitstrings (Q <= ~16 here), and
expanded into Pauli-Z terms with a Walsh-Hadamard transform to count the gates a circuit
would need (a weight-k Z term costs 2(k-1) CNOTs as a parity ladder on all-to-all hardware).
"""

from __future__ import annotations

import math

import numpy as np

from .baselines import adjacency
from .formulate import all_bitstrings


def clique_groups(n: int, edges) -> list[list[int]]:
    """Greedy clique partition (each group is an at-most-one set of candidates)."""
    adj = [set(a) for a in adjacency(n, edges)]
    left = set(range(n))
    groups = []
    while left:
        v = max(left, key=lambda u: (len(adj[u] & left), -u))
        g = [v]
        for u in sorted(adj[v] & left, key=lambda u: (-len(adj[u] & left), u)):
            if all(u in adj[x] for x in g):
                g.append(u)
        groups.append(sorted(g))
        left -= set(g)
    return groups


def bits_for(k: int) -> int:
    return max(1, math.ceil(math.log2(k + 1)))


def encoded_qubits(n: int, edges) -> int:
    return sum(bits_for(len(g)) for g in clique_groups(n, edges))


class Encoding:
    """Layout of the binary encoding for a local problem (w, edges)."""

    def __init__(self, w, edges):
        self.w = [float(v) for v in w]
        self.edges = [tuple(e) for e in edges]
        self.n = len(w)
        self.groups = clique_groups(self.n, edges)
        self.offsets, q = [], 0
        for g in self.groups:
            self.offsets.append(q)
            q += bits_for(len(g))
        self.Q = q

    def decode(self, z: np.ndarray):
        """Encoded bits -> (one-hot x, number of invalid groups)."""
        x = np.zeros(self.n, dtype=np.int8)
        bad = 0
        for g, off in zip(self.groups, self.offsets):
            m = bits_for(len(g))
            code = int(sum(int(z[off + i]) << i for i in range(m)))
            if code > len(g):
                bad += 1
            elif code:
                x[g[code - 1]] = 1
        return x, bad

    def encode(self, x) -> np.ndarray:
        """One-hot x (at most one per group) -> encoded bits."""
        z = np.zeros(self.Q, dtype=np.int8)
        for g, off in zip(self.groups, self.offsets):
            chosen = [j for j, v in enumerate(g) if x[v]]
            assert len(chosen) <= 1, "x picks two members of one at-most-one group"
            code = chosen[0] + 1 if chosen else 0
            for i in range(bits_for(len(g))):
                z[off + i] = (code >> i) & 1
        return z

    def energies(self, P: float, P_invalid: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Energy of every encoded bitstring (rows of all_bitstrings(Q)) and the decoded
        one-hot schedules. E = -value + P * conflicts + P_invalid * invalid groups."""
        P_invalid = P if P_invalid is None else P_invalid
        Z = all_bitstrings(self.Q)
        X = np.zeros((len(Z), self.n), dtype=np.int8)
        bad = np.zeros(len(Z))
        for g, off in zip(self.groups, self.offsets):
            m = bits_for(len(g))
            code = sum(Z[:, off + i].astype(int) << i for i in range(m))
            bad += code > len(g)
            for j, v in enumerate(g):
                X[:, v] = code == j + 1
        w = np.asarray(self.w)
        conf = np.zeros(len(Z))
        for a, b in self.edges:
            conf += X[:, a] * X[:, b]
        return -(X @ w) + P * conf + P_invalid * bad, X

    def warm_probabilities(self, c: np.ndarray) -> np.ndarray:
        """Per-qubit probability of 1 from a one-hot fractional solution c (e.g. the LP):
        P(bit i of group g = 1) = sum of c over the members whose code has bit i set."""
        p = np.zeros(self.Q)
        for g, off in zip(self.groups, self.offsets):
            for i in range(bits_for(len(g))):
                p[off + i] = sum(c[v] for j, v in enumerate(g) if ((j + 1) >> i) & 1)
        return np.clip(p, 0, 1)


def pauli_terms(E: np.ndarray, Q: int, tol: float = 1e-9) -> dict[int, int]:
    """Number of nonzero Z-string terms by weight in E = sum_S c_S prod_{i in S} Z_i."""
    v = E.astype(float).copy()
    h = 1
    while h < len(v):
        for i in range(0, len(v), 2 * h):
            a, b = v[i:i + h].copy(), v[i + h:i + 2 * h].copy()
            v[i:i + h], v[i + h:i + 2 * h] = a + b, a - b
        h *= 2
    v /= len(v)
    out: dict[int, int] = {}
    for idx in np.nonzero(np.abs(v) > tol * max(1.0, np.abs(v).max()))[0]:
        k = bin(int(idx)).count("1")
        if k:
            out[k] = out.get(k, 0) + 1
    return out


def cnot_cost(terms: dict[int, int], p: int = 1) -> int:
    """CNOTs for the cost layers on all-to-all hardware: 2(k-1) per weight-k term per layer."""
    return p * sum(cnt * 2 * (k - 1) for k, cnt in terms.items() if k >= 2)


def binary_size(adj: list[list[int]]):
    """size() for release_neighbourhoods: encoded qubits of the free set's local problem."""
    def size(free) -> int:
        idx = {v: k for k, v in enumerate(free)}
        el = [(idx[u], idx[v]) for u in free for v in adj[u] if v in idx and idx[u] < idx[v]]
        return encoded_qubits(len(free), el)
    return size


# --------------------------------------------------------------------------- QAOA on the encoding

def binary_qaoa_distribution(wl, el, p: int = 1, alpha: float = 0.2, eps: float = 0.25,
                             penalty_mult: float = 1.2, grid: int = 8, maxiter: int = 60):
    """Warm-start CVaR-QAOA on the binary encoding of one subproblem (exact statevector).

    Warm start: each encoded qubit starts with P(1) = LP mass of the candidates whose code
    has that bit set (clique LP of the one-hot problem), clipped to [eps, 1 - eps].
    Returns (probabilities over encoded bitstrings, decoded one-hot rows, encoding, params).
    """
    from scipy.optimize import minimize

    from .qaoa import cvar, lp_relaxation, qaoa_statevector, scaled_cluster_qubo

    ws, P = scaled_cluster_qubo(wl, penalty_mult)
    enc = Encoding(ws, el)
    E, X = enc.energies(P)
    c = enc.warm_probabilities(lp_relaxation(ws, el))
    thetas = 2 * np.arcsin(np.sqrt(np.clip(c, eps, 1 - eps)))
    Q = enc.Q

    def f(prm):
        probs = np.abs(qaoa_statevector(E, Q, prm[:p], prm[p:], thetas)) ** 2
        return cvar(E, probs, alpha)

    best = min(((f([g, b]), [g, b]) for g in np.linspace(0.1, math.pi, grid)
                for b in np.linspace(0.05, math.pi / 2, grid)), key=lambda t: t[0])
    x0 = best[1] if p == 1 else best[1][:1] * p + best[1][1:] * p
    res = minimize(f, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.2})
    prm = res.x if res.fun <= f(x0) else np.asarray(x0)
    probs = np.abs(qaoa_statevector(E, Q, prm[:p], prm[p:], thetas)) ** 2
    return probs, X, enc, {"gammas": list(map(float, prm[:p])), "betas": list(map(float, prm[p:]))}


def propose_qaoa_binary(wl, el, budget, seed, **_):
    if len(wl) == 1:
        return np.ones((1, 1), dtype=np.int8)
    probs, X, _, _ = binary_qaoa_distribution(wl, el)
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(probs), size=budget, p=probs / probs.sum())]


def propose_random_encoded(wl, el, budget, seed, **_):
    """Uniform random encoded bitstrings, decoded (invalid groups -> none chosen)."""
    enc = Encoding(wl, el)
    rng = np.random.default_rng(seed)
    out = np.zeros((budget, len(wl)), dtype=np.int8)
    for r, z in enumerate(rng.integers(0, 2, size=(budget, enc.Q))):
        out[r] = enc.decode(z)[0]
    return out


def propose_exact_large(wl, el, budget, seed, **_):
    from .baselines import exact
    return exact(wl, el)[0][None, :]
