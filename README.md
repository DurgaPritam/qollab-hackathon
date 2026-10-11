# FloodSight Q

Which flooded places should radar satellites image first? A hybrid classical–quantum scheduler for SAR imaging after a flood, run on real 2022 Assam flood data and real satellite orbits, and benchmarked honestly against strong classical baselines.

Built for Qollab × IonQ, theme Quantum Impact. Qiskit · MIT licence.

![Exact optimal schedule on the real event](https://raw.githubusercontent.com/DurgaPritam/qollab-hackathon/main/docs/figures/map_exact.png)

*Exact optimal schedule for the real event: districts sized by people affected, coloured by priority; orange rings mark imaged districts.*

## At a glance

- Our hybrid (conditioned LNS + QAOA) averages **0.993** of the exact optimum on 21 instances, vs greedy 0.984 and our first version 0.949.
- With the same neighbourhoods, QAOA ties the exact and SA sub-solvers. Its edge is sampling: **17×** more probability on each subproblem's optimum than uniform random (354/354).
- On all-to-all (trapped-ion) hardware, QAOA at 24 qubits needs **~40%** fewer two-qubit gates than on a grid or heavy-hex (372 vs 630 / 625).
- The whole flood as one circuit needs **268** qubits and two-qubit error near 1e-4, about ten times better than today's hardware.
- The live run shows each stage on its own row: greedy 0.988 → one QAOA-sampled sweep **0.995** → classical sweeps 1.000.

![Schedule quality on 21 instances](https://raw.githubusercontent.com/DurgaPritam/qollab-hackathon/main/docs/figures/ratios.png)

*Schedule quality on 21 instances (dots = instances, bars = means); full-problem simulated annealing is still best.*

## How to run

1. Open the project in Chrome or Edge (Brave, Opera and Firefox 153+ also work). Safari cannot run Qollab projects.
2. Click **Run Project** and choose a backend in *Select QPU*. *Run settings* offers `shots` (default 1024) and `charts` (default on).
3. **Built-in simulator (24q):** about 25 s on the first run (Qollab loads its packages), about 9 s after. Our Python code takes about 1 s.
4. **IonQ Aria 1 (25q):** remote noise-model simulator, about 20–30 s plus any queue. IonQ accepts one circuit per job, so the largest neighbourhood (12 qubits) runs on IonQ and the other 6 locally.
5. Verified on Qollab (Oct 11, 2026): built-in simulator 1.1 s of Python, IonQ Aria 1 17.7 s, no errors.
6. The default scenario is the top 10 districts over the first 24 h: 68 candidate images, 7 neighbourhoods of at most 12 qubits, angles precomputed offline.
7. Look for where each neighbourhood was sampled (backend or local), then the results table, stage by stage: `Greedy incumbent 0.988`, `+ QAOA LNS sweep, 1 job (IonQ 1 nbhd/12q; local 6) 0.995`, `+ 5 classical LNS sweeps (exact) 1.000`. The 1.000 comes from the classical sweeps, not from QAOA.
8. The seed line: 7 is the project's fixed seed. Across 20 seeds the QAOA-sampled sweep reaches the optimum on 3 and 0.995 on 17; after the classical sweeps, 8 of 20 reach 1.000.
9. Then the benchmark lines and the honest verdict. Qollab's Probabilities panel shows one circuit only: the largest neighbourhood locally, or the single packed circuit on IonQ. The console says which.

## Honest verdict

**No quantum advantage.** On 21 instances: v1 hybrid 0.949, greedy 0.984, conditioned LNS 0.993, full-problem simulated annealing 0.997.

- The gain from 0.949 to 0.993 comes from the LNS neighbourhood structure, not from QAOA: with 1,024 samples per neighbourhood, exact, SA and uniform-random sub-solvers reach the same 0.993.
- At 32 per neighbourhood, QAOA and SA both still reach 0.993; uniform random falls to 0.988 (feasible sets) and 0.984 (all bitstrings).
- What QAOA adds: sample efficiency (17× uniform random, 354/354), though one SA restart still finds the optimum more often (0.70 vs 0.19 per sample).
- All-to-all hardware needs far fewer two-qubit gates: 6,760 vs 22,905 (grid) and 32,124 (heavy-hex) for the full event at p = 1.

---

## The problem

The 2022 Assam floods affected **8.85 million** people in 35 districts (6 Apr – 28 Oct 2022, Govt. of Assam / ASDMA). Synthetic-aperture radar (SAR) sees through monsoon cloud and works at night; optical satellites cannot.

Each radar satellite passes for only a few minutes, needs time to slew between targets, and images well only within a band of incidence angles. Someone has to decide which places get imaged on which pass.

IonQ completed its acquisition of SAR company Capella Space in July 2025 ([IonQ press release, 15 Jul 2025](https://ionq.com/news/ionq-completes-acquisition-of-capella-space-advancing-vision-for-space-based)). Three Capella satellites are in our constellation.

![Humanitarian response in Assam flood 2022](https://raw.githubusercontent.com/DurgaPritam/qollab-hackathon/main/docs/figures/flood_relief.jpg)

*Humanitarian response in Assam flood 2022. Photo: NBalgovind / [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Flood_Relief.jpg), CC BY-SA 4.0.*

### Data used

- **Targets:** the 25 most-affected districts. People affected, villages, deaths and relief-camp inmates come from the official *Assam Floods 2022* memorandum.
- **Priority (1–5):** the quintile of each district's displacement share (relief-camp inmates ÷ people affected), plus 1 if the district had 10 or more deaths, capped at 5.
- **Satellites:** Sentinel-1A, 1C, 1D, NISAR, EOS-4 (RISAT-1A) and Capella-13, 14, 15, from CelesTrak TLEs downloaded on 2026-10-10 and propagated with SGP4 over 48 h.
- **Caveat:** CelesTrak serves only current orbital elements, so this is today's constellation replayed over the 2022 flood locations, not the passes that existed in June 2022.

## Formulation

A **candidate** is one (district, satellite pass, look side). It is valid when the incidence angle at broadside (zero Doppler, closest approach) lies within 20°–45°. The real event has **268** candidates.

The **value** of a candidate, normalised to [0, 1] afterwards:

$$w_i = \text{priority} \times \text{people affected} \times e^{-t_i/24\,\text{h}} \times q(\theta_i), \qquad q(\theta) = e^{-((\theta - 32.5^\circ)/10^\circ)^2}$$

Two candidates **conflict** if they image the same district (one image per district), or are on the same satellite with imaging slots closer than dwell + slew (10 s + 60 s). The real event has **3,380** conflicts (graph density 0.095).

Choosing a schedule is a maximum-weight independent set problem, written as a QUBO:

$$\min_x \; -\sum_i w_i x_i + P \sum_{(i,j)\in E} x_i x_j, \qquad P = 1.2 \max_i w_i$$

We map it to Ising form with $x_i = (1 - z_i)/2$, keeping the constant offset; tests check that Ising and QUBO energies agree on every bitstring.

### Classical baselines

We made these deliberately strong, because the comparison is only credible if they are.

- **Greedy:** sort by value density $w_i / (1 + \deg_i)$ and add each candidate that does not conflict.
- **Simulated annealing (SA):** single-bit flips, geometric cooling, 1,024 seeded restarts of 1,000 sweeps. We first tried 100 sweeps, which fell below greedy, so we tuned SA up before comparing anything; the hybrid's settings were never tuned against SA.
- **Exact optimum:** `scipy.optimize.milp` (HiGHS). Every approximation ratio on this page divides by this optimum.

## Method

The live pipeline is conditioned large-neighbourhood search (LNS) with QAOA as the sub-solver ([`fsq/lns.py`](https://github.com/DurgaPritam/qollab-hackathon/blob/main/fsq/lns.py), [`fsq/api.py`](https://github.com/DurgaPritam/qollab-hackathon/blob/main/fsq/api.py)).

1. **Incumbent:** the greedy schedule.
2. **Neighbourhoods:** release a few selected images plus every unselected image blocked only by them, up to 12 qubits. Each neighbourhood is conditioned on the rest of the schedule, so no outside conflict can be broken.
3. **Warm-start QAOA** (Egger, Mareček & Woerner, *Quantum* 5, 479, 2021): the clique-constrained LP relaxation, clipped to $c_i \in [0.25, 0.75]$, sets each qubit's start state $R_Y(\theta_i)\lvert 0\rangle$ with $\theta_i = 2\arcsin\sqrt{c_i}$ and the mixer $R_Y(\theta_i)\,R_Z(-2\beta)\,R_Y(-\theta_i)$.
4. **CVaR objective** (Barkoutsos et al., *Quantum* 4, 256, 2020), α = 0.2, p = 1. Angles are optimised offline only with COBYLA on an exact statevector simulator, which tests check against Qiskit's `Statevector`.
5. **One job:** all neighbourhood circuits are sampled in one backend job (details below).
6. **Accept moves:** walk through the neighbourhoods and apply the best sample still compatible with the current schedule if it raises the total value. The result is never worse than greedy and always feasible (both tested).
7. **Classical refinement:** 5 more LNS sweeps with an exact sub-solver, labelled classical in the output.

### Running on IonQ: one circuit per job

IonQ allows one job per run and one circuit per job; a multi-circuit job was rejected on Qollab with `IonQAPIError 400: "Only job type ionq.circuit.v1 is supported"`.

- **Packing:** neighbourhoods are independent, so several can sit side by side on disjoint qubits of one circuit, bin-packed largest first up to a qubit budget. Counts are split back per neighbourhood by marginalising over each qubit range (Qiskit little-endian order, unit-tested).
- **Width check:** before submitting, the run prints the circuit width and checks it has exactly the packed number of qubits and classical bits. The circuit is only rewritten to basis gates, with no backend layout, so nothing pads it.
- **Qubit budget:** `min(backend.num_qubits, IONQ_QUBIT_BUDGET)`, default **12**. A 21-qubit packed circuit was accepted by Qollab's IonQ Aria 1 simulator but failed with `NotEnoughQubits` (21 < 25; the cause is on the IonQ side and still unknown), and a failed job cannot be retried within one run.
- **Raising it:** set `"ionq_qubit_budget"` in [`fsq/config.py`](https://github.com/DurgaPritam/qollab-hackathon/blob/main/fsq/config.py) once a larger size has been verified on the selected backend.
- **Local simulators:** circuits go as a list in one job, not packed, because a ~24-qubit statevector is too heavy for the browser.
- **Failures:** if a backend job fails, nothing more is sent, everything is sampled locally, and the label says `local; … job failed`. A label never names a backend that was not used.

## Results

The benchmark has 21 instances: the real event plus 20 seeded perturbations. Each perturbation shifts every priority by −1, 0 or +1, jitters each pass time by N(0, 20 s) and each incidence angle by N(0, 1°).

### Schedule quality

| Method | Real event | Mean ± sd (21) | Worst |
| --- | ---: | ---: | ---: |
| Greedy | 0.994 | 0.984 ± 0.024 | 0.899 |
| Simulated annealing (1,024 restarts) | 0.996 | 0.997 ± 0.002 | 0.993 |
| v1 hybrid QAOA (fixed clusters + repair) | 0.966 | 0.949 ± 0.038 | 0.886 |
| LNS + QAOA, 12 qubits (this version) | 0.995 | 0.993 ± 0.009 | 0.960 |

### Conditioned LNS and its controls

Same 21 instances, same seeds, 10 sweeps, every sub-solver at the same budget per neighbourhood ([`preprocessing/run_lns.py`](https://github.com/DurgaPritam/qollab-hackathon/blob/main/preprocessing/run_lns.py) → `results/e_lns.json`).

| Sub-solver inside LNS | Budget per neighbourhood | Mean ± sd | Worst | Optimal |
| --- | --- | ---: | ---: | ---: |
| Exact (ceiling), 12 qubits | brute force over 2ⁿ | 0.993 ± 0.009 | 0.960 | 4/21 |
| QAOA (warm start, CVaR, p=1), 12 qubits | 1,024 shots | 0.993 ± 0.009 | 0.960 | 4/21 |
| SA, 12 qubits | 1,024 restarts × 100 sweeps | 0.993 ± 0.009 | 0.960 | 4/21 |
| Uniform random feasible, 12 qubits | 1,024 draws | 0.993 ± 0.009 | 0.960 | 4/21 |
| Exact / SA / random feasible, 16 qubits | as above | 0.995 ± 0.009 | 0.960 | 7/21 |

- **The neighbourhood structure is the limit, not the sub-solver:** at 1,024 samples, all four sub-solvers are identical.
- **Larger neighbourhoods help:** 16 qubits beat 12, so more candidates per qubit is the clearest path to better schedules (see binary encoding below).
- A fixed partition gave zero gain even with an exact sub-solver: moving a district's image means dropping a pick in one cluster and adding one in another, which no single fixed cluster can see.

**Live run, stage by stage** (default instance, seed 7, the project's fixed seed): greedy incumbent 0.988 → one QAOA-sampled LNS sweep in one job **0.995** → 5 classical LNS sweeps 1.000. Across 20 seeds ([`preprocessing/run_live_seeds.py`](https://github.com/DurgaPritam/qollab-hackathon/blob/main/preprocessing/run_live_seeds.py)), the QAOA-sampled sweep gives 0.995 on 17 and the optimum on 3; after the classical sweeps, 8 of 20 reach 1.000.

### Resource estimate: the whole problem as one circuit

Every circuit is compiled directly (Qiskit, level 3, seed 42, basis rz/sx/x/cx), with no extrapolation ([`preprocessing/run_resources.py`](https://github.com/DurgaPritam/qollab-hackathon/blob/main/preprocessing/run_resources.py) → `results/e_resources.json`).

| Instance | Qubits, one-hot | Qubits, binary (analytic) | Depth p=1, all-to-all |
| --- | ---: | ---: | ---: |
| 6 h | 44 | 14 | 177 |
| Default live (10 districts, 24 h) | 68 | 34 | 292 |
| 24 h | 151 | 45 | 668 |
| Full event, 48 h | 268 | 92 | 1,153 |

| Instance | 2q gates p=1: all-to-all | Grid | Heavy-hex | e2 for success 0.5 (all-to-all / grid) |
| --- | ---: | ---: | ---: | ---: |
| 6 h | 714 | 1,379 | 1,687 | 9.7e-4 / 5.0e-4 |
| Default live | 878 | 2,175 | 2,996 | 7.9e-4 / 3.2e-4 |
| 24 h | 3,328 | 9,883 | 13,468 | 2.1e-4 / 7.0e-5 |
| Full event, 48 h | 6,760 | 22,905 | 32,124 | 1.0e-4 / 3.0e-5 |

At p = 2 the full event needs 13,520 / 50,960 / 70,590 two-qubit gates (all-to-all / grid / heavy-hex). The binary column is a count only: a greedy clique cover gives 25 "at most one" groups for the full event, and a group of k candidates needs ⌈log₂(k+1)⌉ qubits; its extra multi-qubit gates are not included.

![Resource estimate](https://raw.githubusercontent.com/DurgaPritam/qollab-hackathon/main/docs/figures/resources.png)

*Two-qubit gates per topology for the real instances, and the simple success model for the full event.*

The success column uses a gate-error-only model, $(1 - e_2)^{\text{gates}}$, and is not a hardware prediction: it ignores single-qubit and readout errors, decoherence over the depth, and the number of shots needed. Current commercial two-qubit errors are around 1e-3 or higher, so whole-problem QAOA is out of reach; LNS sidesteps this with ≤12-qubit circuits.

### Connectivity: what all-to-all buys

QAOA circuits for subgraphs of the real conflict graph, compiled with Qiskit (`optimization_level=3`, basis rz/sx/x/cx, `seed_transpiler=42`) for all-to-all (trapped ion), square-grid and heavy-hex layouts, p = 1.

| Qubits | Density | 2q gates, all-to-all | Grid | Heavy-hex |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.64 | 36 | 46 | 63 |
| 12 | 0.55 | 72 | 107 | 111 |
| 16 | 0.58 | 140 | 217 | 259 |
| 20 | 0.63 | 240 | 397 | 388 |
| 24 | 0.67 | 372 | 630 | 625 |

| Qubits | Est. success, all-to-all | Grid | Heavy-hex |
| ---: | ---: | ---: | ---: |
| 8 | 0.83 | 0.79 | 0.73 |
| 12 | 0.70 | 0.58 | 0.57 |
| 16 | 0.50 | 0.34 | 0.27 |
| 20 | 0.30 | 0.14 | 0.14 |
| 24 | 0.15 | 0.04 | 0.04 |

![Two-qubit gates: all-to-all vs grid vs heavy-hex](https://raw.githubusercontent.com/DurgaPritam/qollab-hackathon/main/docs/figures/connectivity_p1.png)

*Two-qubit gates after compiling QAOA (p = 1) for real conflict subgraphs, by hardware layout.*

At p = 2, 24 qubits need 744 / 1,271 / 1,178 two-qubit gates. The extra gates on grid and heavy-hex are SWAPs, and the penalty grows with size and density. *Est. success* = $(1 - 0.005)^{n_{2q}}$, a simple model with a stated error rate, not a hardware measurement.

### Binary (qubit-efficient) encoding

Candidates that pairwise conflict (a clique) can have at most one selected. A greedy clique cover groups them, and a group of k is stored in ⌈log₂(k+1)⌉ qubits (code 0 = none, codes 1..k = one candidate, higher codes penalised). With 12 encoded qubits, a neighbourhood holds 17–25 candidates instead of ≤12 ([`fsq/binenc.py`](https://github.com/DurgaPritam/qollab-hackathon/blob/main/fsq/binenc.py), [`preprocessing/run_binenc.py`](https://github.com/DurgaPritam/qollab-hackathon/blob/main/preprocessing/run_binenc.py) → `results/e_binenc.json`).

| Sub-solver, 12 encoded qubits | Budget | Mean ± sd | Worst | Optimal |
| --- | --- | ---: | ---: | ---: |
| One-hot exact, 12 qubits (reference) | – | 0.993 ± 0.009 | 0.960 | 4/21 |
| Binary, exact (MILP, ceiling) | – | 0.995 ± 0.009 | 0.960 | 7/21 |
| Binary, QAOA on the encoded circuit | 1,024 or 32 shots | 0.995 ± 0.009 | 0.960 | 7/21 |
| SA on the same neighbourhoods | 1,024 or 32 restarts | 0.995 ± 0.009 | 0.960 | 7/21 |
| Uniform random encoded bitstrings | 1,024 / 32 draws | 0.994 / 0.990 | 0.960 / 0.946 | 3/21, 0/21 |

- Twice the candidates per qubit raises the ceiling from 0.9928 to **0.9954** (better on 12 instances, worse on 2), matching 16 one-hot qubits (0.9949).
- SA at 32 restarts differs from QAOA on one instance only (0.9922 vs 0.9955 on perturbed-9), so this is not a trend.
- The price is gates: the encoded cost layer averages **760** CNOTs vs 120 one-hot for the same subproblem (all-to-all estimate). The live run therefore stays one-hot; the encoding is an offline analysis.

Tests check that the encoded energy equals the one-hot energy on every valid state, that invalid codes never win, and that the minimum is the true optimum.

---

## Detailed results

### Low budget: 32 samples per neighbourhood

Costs are in different units: a QAOA shot is one run of a p = 1 circuit (angles tuned offline); an SA restart is 100 sweeps of n single-spin updates; "uniform random feasible" first enumerates all 2ⁿ assignments to list the feasible ones, while "uniform random bitstrings" draws from all 2ⁿ without enumeration and rejects infeasible ones.

| Sub-solver inside LNS, 12 qubits | Budget per neighbourhood | Mean ± sd | Worst | Optimal |
| --- | --- | ---: | ---: | ---: |
| Exact (ceiling), rerun beside the 32-budget rows | – | 0.993 ± 0.009 | 0.960 | 4/21 |
| QAOA | 32 shots | 0.993 ± 0.009 | 0.960 | 4/21 |
| SA | 32 restarts × 100 sweeps | 0.993 ± 0.009 | 0.960 | 4/21 |
| Uniform random feasible | 32 draws | 0.988 ± 0.016 | 0.948 | 2/21 |
| Uniform random bitstrings (all 2ⁿ) | 32 draws | 0.984 ± 0.024 | 0.900 | 0/21 |

### Sampler quality on identical subproblems

354 subproblems of 9–12 candidates, from the same LNS runs.

| Sampler | P(optimum) per sample | Hit within 8 | Hit within 32 |
| --- | ---: | ---: | ---: |
| QAOA, one shot | 0.191 | 0.808 | 0.998 |
| SA, one restart (100 sweeps) | 0.702 | 0.995 | 1.000 |
| Uniform random feasible, one draw | 0.012 | 0.094 | 0.318 |
| Uniform random bitstring, one draw | 0.0004 | 0.003 | 0.012 |

QAOA beats uniform random feasible sampling on all 354 subproblems (median **17×**). SA beats QAOA per sample. On 72 binary-encoded subproblems (19.5 candidates on 11.7 qubits), P(optimum) per sample is 0.13 for QAOA, 0.0004 for uniform random encoded strings (QAOA higher on 72/72) and 0.48 for an SA restart.

### Diversity (v1 pipeline)

Does QAOA sampling give more distinct good backup plans? Same QUBO, same 1,024-sample budget; run on the v1 pipeline and not rerun for LNS.

| Level / method | Feasible rate | Distinct ≥95% of optimum | Mean Hamming |
| --- | ---: | ---: | ---: |
| Per cluster: QAOA (1,024 shots) | 0.70 ± 0.05 | 1.37 ± 1.00 | 0.37 ± 0.77 |
| Per cluster: SA (1,024 restarts, 100 sweeps) | 1.00 | 1.26 ± 0.72 | 0.36 ± 0.81 |
| Full schedule: v1 hybrid (stitched + repaired) | 1.00 | 136 ± 96 | 18.8 ± 1.8 |
| Full schedule: SA (1,024 restarts, 1,000 sweeps) | 1.00 | 303 ± 96 | 26.4 ± 1.4 |

![QAOA vs SA at the same sample budget](https://raw.githubusercontent.com/DurgaPritam/qollab-hackathon/main/docs/figures/diversity.png)

*Feasibility and distinct near-optimal schedules, QAOA vs SA, at the same 1,024-sample budget (v1 pipeline).*

Per cluster (670 clusters with ≥4 candidates), QAOA's 1.37 vs SA's 1.26 is a small edge at best: error bars overlap (QAOA more in 59 clusters, fewer in 1, equal in 610), and only 70% of QAOA samples are feasible. For whole schedules, SA wins: **303** distinct near-optimal schedules vs 136, and more varied (Hamming 26.4 vs 18.8).

### Warm vs cold start and depth (v1 pipeline)

Six instances. P(opt) is the probability the QAOA state puts on each cluster's optimal schedule; the final ratio is the v1 schedule.

| Variant | P(cluster optimum) | Cluster feasible rate | Final ratio (v1) |
| --- | ---: | ---: | ---: |
| Warm start, p=1 | 0.34 | 0.70 | 0.960 |
| Warm start, p=2 | 0.30 | 0.62 | 0.960 |
| Cold start, p=1 | 0.14 | 0.25 | 0.960 |
| Cold start, p=2 | 0.15 | 0.27 | 0.960 |

Warm start puts **2.5×** more probability on the optimum. p = 2 adds nothing, because CVaR with α = 0.2 plateaus once the optimum holds 20% of the probability. The LNS warm start uses the same p = 1 recipe.

### Noise (v1 pipeline)

Real event on Qiskit Aer with 0.1% depolarizing error per 1-qubit gate, 0.5% per CX and 1% readout error. Cluster feasibility drops from 0.70 to 0.57, but the final v1 schedule is unchanged at 0.966, because 1,024 shots still contain each cluster's optimum.

### Qubits vs size

| Horizon | Candidates (qubits) | Conflicts | v1 clusters ≤12 |
| ---: | ---: | ---: | ---: |
| 6 h | 44 | 357 | 5 |
| 12 h | 63 | 561 | 8 |
| 24 h | 151 | 1,664 | 16 |
| 48 h | 268 | 3,380 | 32 |

Without decomposition, one qubit per candidate is far beyond today's 25–29-qubit IonQ simulators.

## Earlier version (v1) and what we learned

v1 split the conflict graph into fixed clusters, solved each with QAOA, then stitched the clusters together and repaired conflicts. It averaged **0.949**, below greedy, which is what led to conditioned LNS.

- **Decompose:** split into connected components; any component over 12 nodes is split by repeated Kernighan–Lin bisection, and the most-linked pieces are merged back up to 12 qubits ([`fsq/decompose.py`](https://github.com/DurgaPritam/qollab-hackathon/blob/main/fsq/decompose.py), now used only by the benchmarks).
- **Stitch:** take each cluster's highest-value feasible sample (or the most frequent, if none is feasible) and join them. Conflicts can occur only on edges cut by the partition (233 of 439 in the default run).
- **Repair:** repeatedly drop the lowest-value candidate in a violated conflict, then greedily fill by value density; the console logged the net change.

| v1 method | Real event | Mean ± sd (21) | Worst |
| --- | ---: | ---: | ---: |
| v1 hybrid QAOA (warm start, p=1) | 0.966 | 0.949 ± 0.038 | 0.886 |
| v1 control: same fixed clusters, exact per cluster | 0.966 | 0.949 ± 0.038 | 0.886 |

**Decomposition insight.** The v1 hybrid matched its control on all 21 instances: QAOA found the best answer inside every cluster, so v1's gap came from splitting the problem, not from QAOA. The real conflict graph is one connected component, and **75%** of conflicts (2,534 of 3,380) crossed cluster boundaries; repair handled them but lost value.

**Better partitions are not just fewer cut conflicts.** A partition filling clusters up to 12 qubits (uneven Kernighan–Lin splits, 5–10 restarts) cut fewer conflicts on 8 instances (2,127 vs 2,461) but scored worse with exact per-cluster solves (0.955 vs 0.975). Which conflicts are cut matters more than how many; conditioned LNS then replaced the fixed partition altogether.

---

## Limitations, and what would change at realistic scale

- **Not modelled:** onboard memory and power, downlink scheduling and ground-station contacts, tasking latency, and satellite agility beyond a fixed 60 s slew.
- **Simplifications:** each image is a single spot, whereas real SAR swaths cover several districts. Cloud cover is ignored, which is fine for radar.
- **Data caveats:** district points are headquarters towns; "people affected" is cumulative over three flood waves, so some people may be counted twice; the pass geometry is from 2026 orbits, not 2022.

**When quantum could matter:** conditioned LNS closes most of the decomposition gap with any sub-solver, and larger neighbourhoods help, so the lever is more candidates per circuit (larger devices or a denser encoding), not a better sampler. That needs devices holding whole connected components (hundreds of qubits) and error rates low enough for deep, dense circuits, where all-to-all connectivity saves the most gates.

We have not shown that QAOA beats classical methods at any size.

## Adapting it

- **Another disaster:** replace `data/targets.csv` (or [`preprocessing/build_targets.py`](https://github.com/DurgaPritam/qollab-hackathon/blob/main/preprocessing/build_targets.py)) with the new locations and severities.
- **Another constellation:** replace `data/tles.txt`.
- Then run `preprocessing/compute_windows.py`, `run_experiments.py` and `build_ui_data.py`.
- **Tuning:** every setting is in [`fsq/config.py`](https://github.com/DurgaPritam/qollab-hackathon/blob/main/fsq/config.py) (incidence range, slew, urgency τ, penalty, cluster size, ε, α, shots).

## Data sources and licences

- **Flood figures:** Govt. of Assam / ASDMA, *Memorandum on Assam Floods 2022*, tables 2.5–2.7 ([PDF](https://igr.assam.gov.in/sites/default/files/swf_utility_folder/departments/asdma_revenue_uneecopscloud_com_oid_70/menu/document/assam_flood_memorandum_2022.pdf)). A public government document, cited with attribution.
- **Orbits:** [CelesTrak](https://celestrak.org) GP/TLE data, snapshot 2026-10-10 12:52 UTC.
- **District coordinates:** Wikipedia (CC BY-SA 4.0) and Wikidata (CC0).
- **Assam outline:** [Natural Earth](https://www.naturalearthdata.com/) 1:50m admin-1 (public domain).
- **Photo:** NBalgovind, Wikimedia Commons, CC BY-SA 4.0 (a resized copy is served from this repository).

[`data/README.md`](https://github.com/DurgaPritam/qollab-hackathon/blob/main/data/README.md) gives full details per file.

## Prior work, attribution and AI use

- **Methods:** warm-start QAOA (Egger, Mareček & Woerner, *Quantum* 5, 479, 2021); CVaR-QAOA (Barkoutsos, Nannicini, Robert, Tavernelli & Woerner, *Quantum* 4, 256, 2020); Kernighan–Lin partitioning via `networkx`, offline only.
- **Prior project, "Quantum Twin":** the connectivity comparison follows the approach of our earlier project. `fsq/connectivity.py` was written during this hackathon.
- **Everything else** in this repository was written during the hackathon window.
- **AI assistance:** an AI coding assistant was used during development, for code, tests and drafting documentation. The team reviewed and ran everything, and all results come from the code in this repository.

## Reproduce

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r preprocessing/requirements-offline.txt qiskit-aer matplotlib
.venv/bin/python preprocessing/compute_windows.py     # orbits -> data/windows.json
.venv/bin/python preprocessing/run_experiments.py     # E1, E-DIV, E-CONN, E2, E3, E5 -> results/  (~5 min, 8 cores)
.venv/bin/python preprocessing/run_noise.py           # E4 -> results/e4.json
.venv/bin/python preprocessing/run_lns.py             # conditioned LNS + controls -> results/e_lns.json (~5 min)
.venv/bin/python preprocessing/run_live_seeds.py      # live recipe over 20 seeds -> results/e_live_seeds.json
.venv/bin/python preprocessing/run_resources.py       # resource estimate -> results/e_resources.json
.venv/bin/python preprocessing/run_binenc.py          # binary encoding inside LNS -> results/e_binenc.json (~6 min)
.venv/bin/python preprocessing/save_figures.py        # -> docs/figures/*.png
.venv/bin/python preprocessing/build_ui_data.py       # -> ui/data.js (optional interactive page)
.venv/bin/python -m pytest -q tests                   # all tests
.venv/bin/python preprocessing/package_qollab.py      # -> dist/floodsight-q.zip for Qollab
```

## Repository layout

| Path | Contents |
| --- | --- |
| `main.py` | Qollab entry point: Run calls `main(shots, charts)` |
| `fsq/api.py` | Run-button pipeline (live LNS run) |
| `fsq/lns.py` | Conditioned LNS: neighbourhoods, sub-solvers, acceptance |
| `fsq/qaoa.py` | Warm/cold QAOA, CVaR, one-job sampling and IonQ packing |
| `fsq/formulate.py` | Candidates, values, conflicts, QUBO and Ising forms |
| `fsq/baselines.py` | Greedy, SA, exact |
| `fsq/binenc.py` | Binary (clique) encoding, offline analysis |
| `fsq/connectivity.py` | Connectivity comparison |
| `fsq/decompose.py` | v1 clustering, stitch and repair (benchmarks only) |
| `fsq/diversity.py`, `fsq/metrics.py` | Diversity comparison and metrics |
| `fsq/plots.py` | Figures, used live and saved to `docs/figures/` |
| `data/` | Shipped data and its README |
| `results/` | Experiment outputs (JSON) |
| `preprocessing/` | Offline scripts, CPython only |
| `tests/` | pytest suite |
| `index.html`, `main.js`, `main.css`, `ui/` | Optional interactive web page (GitHub only), left out of the Qollab upload so Qollab runs `main.py` as a Python project |

## Links

- Demo video: *link to be added*
- Repository: [github.com/DurgaPritam/qollab-hackathon](https://github.com/DurgaPritam/qollab-hackathon)

MIT licence: see [`LICENSE`](https://github.com/DurgaPritam/qollab-hackathon/blob/main/LICENSE).
