# FloodSight Q

**Which flooded places should radar satellites image first?** FloodSight Q is a hybrid classical–quantum scheduler for SAR (radar) imaging after a flood. It runs warm-start QAOA on real 2022 Assam flood data and real satellite orbits, and benchmarks the result honestly against strong classical baselines.

Built for Qollab × IonQ, theme **Quantum Impact**. Qiskit · MIT licence.

---

> ### How to run
> 1. Open the project in a **recent Chrome, Edge, Brave or Opera, or Firefox 153+**. **Safari cannot run Qollab projects.**
> 2. Press **Run**. In *Select QPU*, choose the **built-in simulator** (fastest) or an **IonQ simulator**. Either one gets **exactly one job**.
> 3. Precomputed results (map, schedules, charts) draw at once. The live run then samples every QAOA cluster in a single backend job (1,024 shots) and repairs the stitched schedule. Greedy and simulated annealing run live next to it. Expect well under a minute on the built-in simulator; IonQ's noise-model simulators queue remotely and take longer.
> 4. What to look for:
>    - **Live run** table: the ratio of each method to the exact optimum.
>    - **Console**: one line saying `one job sent to backend with 9 circuit(s)`, followed by every repair action.
>    - **Map**: switches to this run's schedule when the run finishes.

> ### Honest verdict
> **At this size, classical methods find schedules at least as good as the hybrid**:
> - simulated annealing averages 0.997 of the optimum and greedy 0.984, against the hybrid's 0.949;
> - the hybrid's whole gap comes from cutting the problem into qubit-sized pieces, not from QAOA.
>
> What the quantum part adds, measured fairly:
> - **a little more variety** among near-optimal plans per cluster;
> - **far fewer two-qubit gates** on all-to-all trapped-ion hardware than on nearest-neighbour layouts: 1.7× fewer at 24 qubits.
>
> We make **no quantum-advantage claim**.

---

## 1. The problem

The 2022 Assam floods affected **8.85 million people in 35 districts** (6 Apr – 28 Oct 2022, Govt. of Assam / ASDMA).

**Why radar:** synthetic-aperture radar (SAR) sees through monsoon cloud and works at night. Optical satellites cannot do either during a flood.

**Why it's a scheduling problem:** each radar satellite passes for only a few minutes, needs time to slew between targets, and images well only within a band of incidence angles. Someone has to decide which places get imaged on which pass.

One commercial link to this track: **IonQ completed its acquisition of Capella Space**, a SAR satellite company, in July 2025 ([IonQ press release, 15 Jul 2025](https://ionq.com/news/ionq-completes-acquisition-of-capella-space-advancing-vision-for-space-based)). Three Capella satellites are in our constellation.

![Humanitarian response in Assam flood 2022](https://commons.wikimedia.org/wiki/Special:FilePath/Flood_Relief.jpg?width=900)

*Humanitarian response in Assam flood 2022. Photo: NBalgovind / [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Flood_Relief.jpg), CC BY-SA 4.0.*

### Data used

**Targets: the 25 most-affected districts.**
- People affected, villages, deaths and relief-camp inmates come from the official *Assam Floods 2022* memorandum.
- Priority runs from 1 to 5. It is the quintile of each district's displacement share (relief-camp inmates ÷ people affected), plus 1 if the district had 10 or more deaths, capped at 5.

**Satellites:** Sentinel-1A, 1C, 1D, NISAR, EOS-4 (RISAT-1A) and Capella-13, 14, 15.
- Orbits come from **CelesTrak TLEs downloaded on 2026-10-10**.
- Propagated with SGP4 over **48 h**.

**Important caveat:** CelesTrak serves only current orbital elements, so this is today's constellation replayed over the 2022 flood locations. It is not a reconstruction of the passes that existed in June 2022.

## 2. Formulation

**Candidates.** A candidate is one *(district, satellite pass, look side)*.
- It is valid when the incidence angle at broadside (zero Doppler, closest approach) lies within **20°–45°**.
- The real event has **268 candidates**.

**Value of a candidate.**

$$w_i = \text{priority} \times \text{people affected} \times e^{-t_i/24\,\text{h}} \times q(\theta_i), \qquad q(\theta) = e^{-((\theta - 32.5^\circ)/10^\circ)^2}$$

The values are then normalised to [0, 1].

**Conflicts (edges).** Two candidates conflict if either:
- they image the **same district** (one image per district), or
- they are on the **same satellite** with imaging slots closer than dwell + slew, i.e. 10 s + 60 s.

The real event has **3,380 conflicts**, a graph density of 0.095.

**QUBO.** Choosing a schedule is a maximum-weight independent set problem:

$$\min_x \; -\sum_i w_i x_i + P \sum_{(i,j)\in E} x_i x_j, \qquad P = 1.2 \max_i w_i$$

**Ising form.** We map the QUBO to Ising form with $x_i = (1 - z_i)/2$, keeping the constant offset. Tests check that the Ising energies equal the QUBO energies for every bitstring.

## 3. Classical baselines

We made these deliberately strong, because the comparison is only credible if they are.

- **Greedy:** sort by value density $w_i / (1 + \deg_i)$ and add each candidate if it does not conflict.
- **Simulated annealing** on the same QUBO:
  - single-bit flips with geometric cooling;
  - **1,024 seeded restarts**, 1,000 sweeps on full schedules.
  - We first tried 100 sweeps. That fell below greedy, so we tuned SA up before comparing anything. The hybrid's settings were never tuned against SA.
- **Exact optimum:** `scipy.optimize.milp` (HiGHS). **Every approximation ratio on this page divides by this optimum.**

## 4. Hybrid method

1. **Decompose.** Split the conflict graph into connected components. Any component over 12 nodes is split by repeated Kernighan–Lin bisection, and the most-linked pieces are merged back up to 12 qubits.
2. **Warm start** (Egger, Mareček & Woerner, *Quantum* 5, 479, 2021).
   - Solve the LP relaxation with clique constraints. Plain edge constraints give an uninformative all-½ solution.
   - Clip each value to $c_i \in [0.25, 0.75]$.
   - Prepare each qubit with $R_Y(\theta_i)\lvert 0\rangle$, where $\theta_i = 2\arcsin\sqrt{c_i}$.
   - Use the mixer $R_Y(\theta_i)\,R_Z(-2\beta)\,R_Y(-\theta_i)$.
3. **CVaR objective** (Barkoutsos et al., *Quantum* 4, 256, 2020), with α = 0.2. Angles are optimised with COBYLA **offline only**, on an exact statevector simulator that the tests check against Qiskit's `Statevector`.
4. **One-job sampling.** Every cluster's circuit goes to the selected backend in **one `backend.run` call**, as IonQ allows one job per run.
   - If the backend rejects the call, nothing more is sent; the clusters are sampled locally and the console says so.
5. **Stitch and repair.**
   - Take each cluster's lowest-energy sample.
   - Drop the lower-value candidate on every conflict between clusters.
   - Greedily add back any compatible unused candidate.
   - Every action is logged.

The **default Run** uses a small instance: the top 10 districts over the first 24 h. It has 68 candidates and 9 clusters of at most 9 qubits, with precomputed angles, so the browser does no optimisation.

## 5. Results

The benchmark has 21 instances: the real event plus 20 seeded perturbations. Each perturbation shifts every priority by −1, 0 or +1, jitters each pass time by N(0, 20 s) and jitters each incidence angle by N(0, 1°).

### Approximation ratio (schedule value / exact optimum)

| Method | Real event | Mean ± sd (21) | Worst |
| --- | --- | --- | --- |
| Greedy | 0.994 | 0.984 ± 0.024 | 0.899 |
| Simulated annealing (1,024 restarts) | 0.996 | **0.997 ± 0.002** | 0.993 |
| Hybrid QAOA (warm start, p=1) | 0.966 | 0.949 ± 0.038 | 0.886 |
| *Control: same decomposition, exact per cluster* | 0.966 | 0.949 ± 0.038 | 0.886 |

The hybrid matches the control on **all 21 instances**: with 1,024 shots, QAOA finds every cluster's optimum. The loss comes from decomposition. The real conflict graph is one connected component, and in our partition into ≤12-qubit clusters **75% of conflicts (2,534 of 3,380) cross cluster boundaries**. Repair handles those conflicts but loses value in doing so.

### Fair diversity comparison: same QUBO, same 1,024-sample budget

The question: does QAOA sampling give more distinct good backup plans, useful if priorities change?

| Level / method | Feasible rate | Distinct schedules ≥95% of optimum | Diversity (mean Hamming) |
| --- | --- | --- | --- |
| Per cluster: QAOA (1,024 shots) | 0.70 ± 0.05 | **1.37 ± 1.00** | 0.37 ± 0.77 |
| Per cluster: SA (1,024 restarts, 100 sweeps) | **1.00** | 1.26 ± 0.72 | 0.36 ± 0.81 |
| Full schedule: hybrid (stitched + repaired) | 1.00 | 136 ± 96 | 18.8 ± 1.8 |
| Full schedule: SA (1,024 restarts, 1,000 sweeps) | 1.00 | **303 ± 96** | **26.4 ± 1.4** |

**Per cluster** (670 clusters with ≥4 candidates), QAOA found more distinct near-optimal schedules than SA in 59 clusters, fewer in 1, and the same number in 610. That is a small, consistent edge, bought with 30% infeasible samples.

**For whole schedules, SA clearly wins**: it gives about twice as many good plans, and more varied ones. We report both levels because the per-cluster result alone would overstate what QAOA delivers.

### Connectivity: what IonQ's all-to-all connectivity buys

We took subgraphs of the real conflict graph and compiled the QAOA circuit with Qiskit (`optimization_level=3`, basis rz/sx/x/cx, `seed_transpiler=42`) for three layouts:
- **all-to-all**, as on trapped-ion hardware;
- a **square grid**;
- **heavy-hex**.

| Qubits | Density | 2q gates, all-to-all | 2q gates, grid | 2q gates, heavy-hex | Est. success, all / grid / hex |
| --- | --- | --- | --- | --- | --- |
| 8 | 0.64 | 36 | 46 | 63 | 0.84 / 0.79 / 0.73 |
| 12 | 0.55 | 72 | 107 | 111 | 0.70 / 0.59 / 0.57 |
| 16 | 0.58 | 140 | 217 | 259 | 0.50 / 0.34 / 0.27 |
| 20 | 0.63 | 240 | 397 | 388 | 0.30 / 0.14 / 0.14 |
| 24 | 0.67 | **372** | 630 | 625 | **0.16** / 0.04 / 0.04 |

This is QAOA at p=1; at p=2, 24 qubits need 744 / 1,271 / 1,178 two-qubit gates. The extra gates on grid and heavy-hex are SWAPs, and the penalty grows with size and density. Dense disaster-zone conflict graphs are the case where all-to-all connectivity matters most.

*Est. success* = $(1 - 0.005)^{n_{2q}}$. This is a simple model with a stated error rate, **not a hardware measurement**.

### Other experiments

**Warm vs cold start (E3) and depth (E2)**, on 6 instances. P(opt) is the probability the QAOA state puts on each cluster's optimal schedule.

| Variant | P(cluster optimum) | Cluster feasible rate | Final ratio |
| --- | --- | --- | --- |
| Warm start, p=1 | **0.34** | **0.70** | 0.960 |
| Warm start, p=2 | 0.30 | 0.62 | 0.960 |
| Cold start, p=1 | 0.14 | 0.25 | 0.960 |
| Cold start, p=2 | 0.15 | 0.27 | 0.960 |

- **Warm start puts 2.5× more probability on the optimum.** That is a real, measurable effect.
- p=2 adds nothing, because CVaR with α=0.2 plateaus once the optimum holds 20% of the probability.
- All four variants reach the same final schedule, because decomposition, not sampling, is the bottleneck.

**Noise (E4).** We ran the real event on Qiskit Aer with:
- 0.1% depolarizing error on each 1-qubit gate;
- 0.5% on each CX;
- 1% readout error.

Cluster feasibility drops from 0.70 to 0.57. The final schedule is unchanged at 0.966, because 1,024 shots still contain each cluster's optimum.

**Qubits vs size (E5).**

| Horizon | Candidates (qubits) | Conflicts | Clusters ≤12 |
| --- | --- | --- | --- |
| 6 h | 44 | 357 | 5 |
| 12 h | 63 | 561 | 8 |
| 24 h | 151 | 1,664 | 16 |
| 48 h | 268 | 3,380 | 32 |

Without decomposition, one qubit per candidate is far beyond today's 25–29-qubit IonQ simulators.

## 6. Limitations, and what would change at realistic scale

**Not modelled:**
- onboard memory and power;
- downlink scheduling and ground-station contacts;
- tasking latency;
- satellite agility beyond a fixed 60 s slew.

Each image is treated as a single spot, whereas real SAR swaths cover several districts at once. Cloud cover is ignored, which is fine because radar is unaffected by it.

**Data caveats:**
- District points are headquarters towns.
- "People affected" is cumulative over three flood waves, so some people may be counted twice.
- The pass geometry is from 2026 orbits, not 2022.

**When quantum could matter:** our results suggest the bottleneck is the decomposition, not the sampler. Quantum help would need:
- devices large enough to hold whole connected components (hundreds of qubits);
- error rates low enough that deep, dense circuits survive, which is where all-to-all connectivity saves the most gates.

We have not shown that QAOA beats classical methods at any size.

## 7. Adapting it

- **Another disaster:** replace `data/targets.csv` (or `preprocessing/build_targets.py`) with the new locations and severities.
- **Another constellation:** replace `data/tles.txt`.
- Then run `preprocessing/compute_windows.py`, `run_experiments.py` and `build_ui_data.py`.
- **Tuning:** every setting is in `fsq/config.py` (incidence range, slew, urgency τ, penalty, cluster size, ε, α, shots).

## 8. Data sources and licences

- **Flood figures:** Govt. of Assam / ASDMA, *Memorandum on Assam Floods 2022*, tables 2.5–2.7 ([PDF](https://igr.assam.gov.in/sites/default/files/swf_utility_folder/departments/asdma_revenue_uneecopscloud_com_oid_70/menu/document/assam_flood_memorandum_2022.pdf)). A public government document, cited with attribution.
- **Orbits:** [CelesTrak](https://celestrak.org) GP/TLE data, **snapshot 2026-10-10 12:52 UTC**.
- **District coordinates:** Wikipedia (CC BY-SA 4.0) and Wikidata (CC0).
- **Assam outline:** [Natural Earth](https://www.naturalearthdata.com/) 1:50m admin-1 (public domain).
- **Photo:** NBalgovind, Wikimedia Commons, CC BY-SA 4.0.

`data/README.md` gives full details per file.

## 9. Prior work, attribution and AI use

- **Methods:**
  - warm-start QAOA: Egger, Mareček & Woerner, *Quantum* 5, 479 (2021);
  - CVaR-QAOA: Barkoutsos, Nannicini, Robert, Tavernelli & Woerner, *Quantum* 4, 256 (2020);
  - Kernighan–Lin partitioning: via `networkx`, offline only.
- **Prior project, "Quantum Twin":** the connectivity comparison follows the approach of our earlier project. `fsq/connectivity.py` was written during this hackathon.
- **Everything else** in this repository was written during the hackathon window.
- **AI assistance:** an AI coding assistant was used during development, for code, tests and drafting documentation. The team reviewed and ran everything, and all results come from the code in this repository.

## 10. Reproduce

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r preprocessing/requirements-offline.txt qiskit-aer
.venv/bin/python preprocessing/compute_windows.py     # orbits -> data/windows.json
.venv/bin/python preprocessing/run_experiments.py     # E1, E-DIV, E-CONN, E2, E3, E5 -> results/  (~4 min, 8 cores)
.venv/bin/python preprocessing/run_noise.py           # E4 -> results/e4.json
.venv/bin/python preprocessing/build_ui_data.py       # -> ui/data.js
.venv/bin/python -m pytest -q tests                   # 17 tests
```

## Repository layout

| Path | Contents |
| --- | --- |
| `index.html`, `main.js`, `main.css`, `ui/` | Page; plain SVG, no external libraries |
| `fsq/` | Python package |
| `fsq/api.py` | Run-button entry point |
| `fsq/formulate.py` | Candidates, values, conflicts, QUBO and Ising forms |
| `fsq/baselines.py` | Greedy, SA, exact |
| `fsq/decompose.py` | Clustering, stitch and repair |
| `fsq/qaoa.py` | Warm/cold QAOA, CVaR, one-job sampling |
| `fsq/connectivity.py` | Connectivity comparison |
| `fsq/diversity.py`, `fsq/metrics.py` | Diversity comparison and metrics |
| `data/` | Shipped data and its README |
| `results/` | Experiment outputs (JSON) |
| `preprocessing/` | Offline scripts, CPython only |
| `tests/` | pytest suite |

**Links:** demo video, *to be added* · repository: <https://github.com/DurgaPritam/qollab-hackathon>

MIT licence: see `LICENSE`.
