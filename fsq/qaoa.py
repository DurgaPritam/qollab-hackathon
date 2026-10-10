"""Warm-start and cold-start QAOA with a CVaR objective, plus one-job final sampling.

Methods:
  * Warm-start QAOA: Egger, Marecek & Woerner, "Warm-starting quantum optimization",
    Quantum 5, 479 (2021). Initial state RY(theta_i)|0>, theta_i = 2 arcsin(sqrt(c_i)),
    mixer RY(theta_i) RZ(-2 beta) RY(-theta_i) per qubit.
  * CVaR objective: Barkoutsos et al., "Improving Variational Quantum Optimization using
    CVaR", Quantum 4, 256 (2020).

Parameter optimisation runs offline (or on local simulators) with an exact statevector
computed by a small NumPy simulator. `tests/test_qaoa.py` checks that simulator against
Qiskit's Statevector for the same circuit. The default Qollab run only samples, using
precomputed parameters.
"""

from __future__ import annotations

import math

import numpy as np

from .formulate import all_bitstrings, energies_batch, ising_terms


# --------------------------------------------------------------------------- scaling

def scaled_cluster_qubo(w_local: list[float], penalty_mult: float):
    """Scale a cluster's weights so max w = 1 and use P = penalty_mult (> 1 = max w).

    Scaling by a positive constant does not change which bitstrings are optimal, and it
    keeps the QAOA angles in a similar range for every cluster.
    """
    m = max(w_local) if w_local else 1.0
    m = m if m > 0 else 1.0
    return [wi / m for wi in w_local], penalty_mult


# --------------------------------------------------------------------------- warm start

def lp_relaxation(w: list[float], edges: list) -> np.ndarray:
    """LP relaxation of MWIS with clique constraints (sum over each maximal clique <= 1).

    Plain edge constraints give the uninformative all-1/2 solution on dense graphs; the
    clique form is the standard tighter relaxation. Falls back to greedy-based 0/1 values
    if SciPy's linprog is unavailable.
    """
    n = len(w)
    if n == 0:
        return np.zeros(0)
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(map(tuple, edges))
        cliques = [c for c in nx.find_cliques(G) if len(c) > 1]
    except ImportError:
        cliques = [list(e) for e in edges]
    try:
        from scipy.optimize import linprog
        if not cliques:
            return np.ones(n)
        A = np.zeros((len(cliques), n))
        for r, c in enumerate(cliques):
            A[r, c] = 1
        res = linprog(-np.asarray(w, dtype=float), A_ub=A, b_ub=np.ones(len(cliques)),
                      bounds=[(0, 1)] * n, method="highs")
        if res.status == 0:
            return np.clip(res.x, 0, 1)
    except Exception:
        pass
    from .baselines import greedy
    return greedy(w, edges).astype(float)


def warm_thetas(c: np.ndarray, eps: float) -> np.ndarray:
    """theta_i = 2 arcsin(sqrt(clip(c_i, eps, 1 - eps)))."""
    c = np.clip(np.asarray(c, dtype=float), eps, 1 - eps)
    return 2 * np.arcsin(np.sqrt(c))


# --------------------------------------------------------------------------- circuits

def qaoa_circuit(w: list[float], edges: list, P: float, gammas, betas,
                 thetas=None, measure: bool = True):
    """Build the QAOA circuit as a Qiskit QuantumCircuit.

    thetas=None gives standard (cold) QAOA: H initial state and RX(2 beta) mixer.
    """
    from qiskit import QuantumCircuit

    n = len(w)
    h, J, _ = ising_terms(w, edges, P)
    qc = QuantumCircuit(n)
    for i in range(n):
        if thetas is None:
            qc.h(i)
        else:
            qc.ry(float(thetas[i]), i)
    for g, b in zip(gammas, betas):
        for i in range(n):
            if h[i] != 0:
                qc.rz(2 * g * h[i], i)
        for (a, c), v in J.items():
            qc.rzz(2 * g * v, a, c)
        for i in range(n):
            if thetas is None:
                qc.rx(2 * b, i)
            else:
                qc.ry(-float(thetas[i]), i)
                qc.rz(-2 * b, i)
                qc.ry(float(thetas[i]), i)
    if measure:
        qc.measure_all()
    return qc


# --------------------------------------------------------------------------- fast sim

def _ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(t):
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]], dtype=complex)


def _rx(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _apply_1q(psi: np.ndarray, U: np.ndarray, i: int, n: int) -> np.ndarray:
    v = psi.reshape(2 ** (n - 1 - i), 2, 2 ** i)
    return np.einsum("ab,xbz->xaz", U, v).reshape(-1)


def qaoa_statevector(energies: np.ndarray, n: int, gammas, betas, thetas=None) -> np.ndarray:
    """Exact QAOA statevector (index k: qubit i = bit i of k). Matches qaoa_circuit up to
    a global phase; `energies` are QUBO energies of all 2^n bitstrings."""
    if thetas is None:
        psi = np.full(2 ** n, 1 / math.sqrt(2 ** n), dtype=complex)
    else:
        psi = np.array([1.0 + 0j])
        for i in range(n):  # build product state, qubit 0 = least significant
            a = np.array([math.cos(thetas[i] / 2), math.sin(thetas[i] / 2)], dtype=complex)
            psi = np.kron(a, psi)
    mixers = None
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * energies)
        if thetas is None:
            U = _rx(2 * b)
            for i in range(n):
                psi = _apply_1q(psi, U, i, n)
        else:
            mixers = [(_ry(thetas[i]) @ _rz(-2 * b) @ _ry(-thetas[i])) for i in range(n)]
            for i in range(n):
                psi = _apply_1q(psi, mixers[i], i, n)
    return psi


def cvar(energies: np.ndarray, probs: np.ndarray, alpha: float) -> float:
    """Mean energy of the best (lowest) alpha probability mass."""
    order = np.argsort(energies, kind="stable")
    e, p = energies[order], probs[order]
    cum = np.cumsum(p)
    take = np.minimum(p, np.maximum(alpha - (cum - p), 0))
    return float(np.dot(e, take) / alpha)


# --------------------------------------------------------------------------- optimise

def optimize(w: list[float], edges: list, P: float, p: int, alpha: float,
             thetas=None, maxiter: int = 200, grid: int = 12) -> dict:
    """Optimise (gammas, betas) for the CVaR objective with COBYLA.

    p = 1: coarse grid over (gamma, beta) then COBYLA from the best grid point.
    p > 1: start from the optimised (p-1) angles with a small extra layer, then COBYLA.
    Fully deterministic.
    """
    from scipy.optimize import minimize

    n = len(w)
    X = all_bitstrings(n)
    E = energies_batch(X, w, edges, P)

    def f(params):
        g, b = params[:len(params) // 2], params[len(params) // 2:]
        probs = np.abs(qaoa_statevector(E, n, g, b, thetas)) ** 2
        return cvar(E, probs, alpha)

    evals = 0
    if p == 1:
        best = None
        for g in np.linspace(0.1, 2 * math.pi / 2, grid):
            for b in np.linspace(0.05, math.pi / 2, grid):
                val = f([g, b])
                evals += 1
                if best is None or val < best[0]:
                    best = (val, [g, b])
        x0 = best[1]
    else:
        prev = optimize(w, edges, P, p - 1, alpha, thetas, maxiter, grid)
        evals += prev["evals"]
        x0 = list(prev["gammas"]) + [prev["gammas"][-1] * 0.5] + list(prev["betas"]) + [prev["betas"][-1] * 0.5]
    res = minimize(f, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.2})
    evals += int(res.nfev)
    params = res.x if res.fun <= f(x0) else np.asarray(x0)
    g, b = list(map(float, params[:p])), list(map(float, params[p:]))
    probs = np.abs(qaoa_statevector(E, n, g, b, thetas)) ** 2
    return {"gammas": g, "betas": b, "cvar": cvar(E, probs, alpha),
            "expected_energy": float(np.dot(E, probs)), "evals": evals}


def exact_probabilities(w, edges, P, gammas, betas, thetas=None) -> np.ndarray:
    """Probabilities of all 2^n bitstrings under tuned parameters."""
    n = len(w)
    E = energies_batch(all_bitstrings(n), w, edges, P)
    return np.abs(qaoa_statevector(E, n, gammas, betas, thetas)) ** 2


def sample_from_probs(probs: np.ndarray, n: int, shots: int, seed: int) -> np.ndarray:
    """Draw `shots` bitstrings (rows of 0/1) from an exact distribution."""
    rng = np.random.default_rng(seed)
    probs = probs / probs.sum()
    ks = rng.choice(len(probs), size=shots, p=probs)
    return ((ks[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)


# --------------------------------------------------------------------------- backend

def counts_to_samples(counts: dict, n: int) -> np.ndarray:
    """Expand a counts dict into one 0/1 row per shot (qubit 0 = rightmost bit)."""
    rows = []
    for key, c in counts.items():
        k = str(key).replace(" ", "")
        if k.startswith("0x"):
            val = int(k, 16)
        else:
            val = int(k, 2)
        bits = [(val >> i) & 1 for i in range(n)]
        rows.extend([bits] * int(c))
    return np.asarray(rows, dtype=np.int8).reshape(-1, n)


def _transpile_portable(circs):
    """Rewrite to a small gate set that every simulator and IonQ accept."""
    from qiskit import transpile
    return transpile(circs, basis_gates=["rx", "ry", "rz", "h", "cx"], optimization_level=1,
                     seed_transpiler=11)


def local_counts(circuit, shots: int, seed: int) -> dict:
    """Sample a measured circuit locally with Qiskit's exact statevector."""
    from qiskit.quantum_info import Statevector
    sv = Statevector(circuit.remove_final_measurements(inplace=False))
    sv.seed(seed)
    return dict(sv.sample_counts(shots))


def is_ionq(backend) -> bool:
    """True for IonQ backends (name or provider class mentions IonQ)."""
    if backend is None:
        return False
    names = [type(backend).__name__, type(backend).__module__]
    for attr in ("name", "backend_name"):
        try:
            v = getattr(backend, attr)
            names.append(str(v() if callable(v) else v))
        except Exception:
            pass
    return any("ionq" in n.lower() for n in names)


def qubit_budget(backend, default: int) -> int:
    """backend.num_qubits when the backend reports it, else `default`."""
    try:
        n = getattr(backend, "num_qubits", None)
        n = n() if callable(n) else n
        if n:
            return int(n)
    except Exception:
        pass
    return int(default)


def pack_clusters(sizes: list[int], budget: int) -> tuple[list[int], list[int]]:
    """Greedy bin-pack into one circuit: largest clusters first, add each one that still fits.

    Returns (packed cluster indices, their qubit offsets). Clusters not packed are sampled
    locally.
    """
    order = sorted(range(len(sizes)), key=lambda k: (-sizes[k], k))
    packed, offsets, used = [], [], 0
    for k in order:
        if used + sizes[k] <= budget:
            packed.append(k)
            offsets.append(used)
            used += sizes[k]
    return packed, offsets


def combine_circuits(circuits: list, offsets: list[int]):
    """Place unmeasured cluster circuits side by side on disjoint qubits, then measure all.

    Cluster j occupies qubits offsets[j] .. offsets[j] + n_j - 1, so in Qiskit's
    little-endian bitstrings its bits are the slice counted from the right.
    """
    from qiskit import QuantumCircuit

    total = max((o + c.num_qubits for c, o in zip(circuits, offsets)), default=0)
    qc = QuantumCircuit(total)
    for c, o in zip(circuits, offsets):
        qc.compose(c.remove_final_measurements(inplace=False), qubits=list(range(o, o + c.num_qubits)),
                   inplace=True)
    qc.measure_all()
    return qc


def marginal_counts(counts: dict, offset: int, n: int) -> dict:
    """Counts of the n qubits starting at `offset`, from counts over a wider register.

    Keys use Qiskit's order: qubit 0 is the rightmost character. Hex keys ("0x..") are
    accepted. Output keys are n-character bitstrings in the same order.
    """
    out: dict[str, int] = {}
    mask = (1 << n) - 1
    for key, c in counts.items():
        k = str(key).replace(" ", "")
        val = int(k, 16) if k.startswith("0x") else int(k, 2)
        sub = format((val >> offset) & mask, f"0{n}b")
        out[sub] = out.get(sub, 0) + int(c)
    return out


def _counts_from_result(res, pos: int, n_circuits: int) -> dict:
    try:
        return dict(res.get_counts(pos))
    except Exception:
        allc = res.get_counts()
        return dict(allc[pos] if isinstance(allc, list) else allc)


def sample_clusters(backend, circuits: list, shots: int, seed: int, ionq_budget: int = 25) -> tuple[list[dict], dict]:
    """Final sampling with at most ONE backend job, and an exact record of where each
    cluster was sampled.

    * IonQ (one circuit per job): bin-pack clusters into one combined circuit on disjoint
      qubits (budget = backend.num_qubits, else `ionq_budget`); the rest run locally.
    * Other backends (local simulators): one job with the list of circuits, largest cluster
      submitted last (Qollab's Probabilities panel shows the last circuit). No packing: a
      ~24-qubit statevector is too heavy for the browser.
    * Any backend error: nothing else is sent; every cluster is sampled locally.

    Returns (counts per cluster, info). info["remote"] lists clusters sampled on the backend.
    """
    n = len(circuits)
    sizes = [c.num_qubits for c in circuits]
    counts: list = [None] * n
    info = {"mode": "local", "remote": [], "local": list(range(n)), "remote_qubits": 0,
            "error": None, "panel": None, "budget": None}
    if backend is None or n == 0:
        info["mode"] = "none"
    elif is_ionq(backend):
        budget = qubit_budget(backend, ionq_budget)
        packed, offsets = pack_clusters(sizes, budget)
        info.update(mode="packed", budget=budget)
        try:
            combined = _transpile_portable(combine_circuits([circuits[k] for k in packed], offsets))
            res = backend.run(combined, shots=shots).result()
            joint = _counts_from_result(res, 0, 1)
            for k, o in zip(packed, offsets):
                counts[k] = marginal_counts(joint, o, sizes[k])
            info.update(remote=packed, remote_qubits=sum(sizes[k] for k in packed),
                        panel=f"the single packed circuit ({sum(sizes[k] for k in packed)} qubits, "
                              f"clusters {', '.join(str(k + 1) for k in sorted(packed))})")
        except Exception as e:  # never send a second job
            info["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    else:
        order = sorted(range(n), key=lambda k: (sizes[k], -k))     # largest last
        info["mode"] = "list"
        try:
            tc = _transpile_portable([circuits[k] for k in order])
            res = backend.run(tc if len(tc) > 1 else tc[0], shots=shots).result()
            for pos, k in enumerate(order):
                counts[k] = _counts_from_result(res, pos, n)
            last = order[-1]
            info.update(remote=list(range(n)), remote_qubits=sum(sizes),
                        panel=f"the last circuit in the job: cluster {last + 1} ({sizes[last]} qubits, the largest)")
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    if info["error"]:
        info["remote"], info["remote_qubits"] = [], 0
        counts = [None] * n
    info["local"] = [k for k in range(n) if k not in info["remote"]]
    for k in info["local"]:
        counts[k] = local_counts(circuits[k], shots, seed + k)
    return counts, info
