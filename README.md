# FloodSight Q

**Which flooded places should radar satellites image first?** FloodSight Q is a hybrid classical–quantum scheduler for SAR (radar) imaging after a flood. It runs warm-start QAOA on real 2022 Assam flood data and real satellite orbits, and benchmarks the result honestly against strong classical baselines.

Built for Qollab × IonQ, theme **Quantum Impact**. Qiskit · MIT licence.

---

> ### How to run
> 1. Open the project in **Chrome or Edge** (Brave, Opera and Firefox 153+ also work). **Safari cannot run Qollab projects.**
> 2. Click **Run Project**. In *Select QPU*, choose one of:
>    - **Built-in simulator (24q):** about **23 s** on the first run (Qollab loads its packages) and about **9 s** after; our Python code itself takes 1–3 s.
>    - **IonQ Aria 1 (25q):** a remote noise-model simulator that may queue. IonQ accepts **one circuit per job**. By default the run sends the **largest cluster (12 qubits)** to IonQ as that one circuit and samples the other 7 clusters locally; the console lists which ran where. See *IonQ qubit budget* below. Verified on Qollab (Oct 10, 2026): the 12-qubit circuit ran on IonQ Aria 1 in one job, and the whole run took about 14 s, mostly the remote IonQ job.
>
>    *Run settings* offers `shots` (default 1024) and `charts` (default on).
> 3. What the run does (the default scenario: top 10 districts over the first 24 h, 68 candidate images):
>    - builds the greedy schedule (the incumbent) and 7 neighbourhoods of at most 12 candidates around it, each conditioned on the rest of the schedule;
>    - samples one warm-start QAOA circuit per neighbourhood in **one job**, and applies each neighbourhood's best compatible sample if it raises the schedule value;
>    - runs 5 more LNS sweeps with an exact classical sub-solver (labelled classical);
>    - runs simulated annealing live and compares everything with the exact optimum;
>    - draws four figures.
> 4. What to look for:
>    - **where each neighbourhood was sampled** (backend or local);
>    - **the results table, stage by stage**: `Greedy incumbent 0.988`, `+ one QAOA-sampled LNS sweep (one job) 0.995`, `+ 5 classical LNS sweeps 1.000`. The 1.000 comes from the classical sweeps, not from QAOA;
>    - **the seed line**: 7 is the project's fixed seed. Across 20 seeds the one-job QAOA-sampled sweep reaches the optimum on 3 and 0.995 on the other 17; with the classical sweeps, 8 of 20 reach 1.000 (`results/e_live_seeds.json`);
>    - **the benchmark lines and the honest verdict**.
>
>    Qollab's **Probabilities panel shows one circuit only**. On local simulators it is the largest neighbourhood, which is submitted last; on IonQ it is the single packed circuit. The console says which.

> ### Honest verdict
> **No quantum advantage.** On 21 instances (ratio to the exact optimum):
> - v1 hybrid (independent clusters + repair) **0.949**, greedy **0.984**, conditioned LNS **0.993**, full-problem simulated annealing **0.997**.
> - **The gain from 0.949 to 0.993 comes from the LNS neighbourhood structure, not from QAOA.** With 1,024 samples per neighbourhood, the exact, SA and uniform-random sub-solvers reach the identical 0.993.
> - With 32 per neighbourhood, QAOA (32 shots) and SA (32 restarts × 100 sweeps) both still reach 0.993, the exact ceiling. Uniform random falls to 0.988 over feasible sets and 0.984 over all bitstrings.
>
> What the quantum part adds, measured fairly:
> - **sample efficiency**: QAOA puts **17× (median) more probability** on each subproblem's optimum than uniform random feasible sampling, on **354 of 354** subproblems. One SA restart still finds the optimum more often (0.70 vs 0.19 per sample), at the cost of 100 sweeps;
> - **far fewer two-qubit gates** on all-to-all trapped-ion hardware: the full event as one QAOA circuit needs 6,760 two-qubit gates all-to-all against 22,905 on a grid and 32,124 on heavy-hex (p = 1).
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

![Exact optimal schedule on the real event](https://raw.githubusercontent.com/DurgaPritam/qollab-hackathon/main/docs/figures/map_exact.png)

*Districts sized by people affected and coloured by priority; orange rings mark districts imaged in the exact optimal schedule.*

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
4. **One-job sampling.** IonQ allows **one job per run and one circuit per job**; we confirmed the second limit on Qollab, where a multi-circuit job was rejected with `IonQAPIError 400: "Only job type ionq.circuit.v1 is supported"`.
   - **On IonQ:** clusters are independent subproblems, so several can sit side by side on **disjoint qubits of one circuit**. We bin-pack largest first, up to a qubit budget. A joint measurement of that circuit gives independent samples for each cluster. We split the counts back per cluster by marginalising over each cluster's qubit range, using Qiskit's little-endian bit order; a unit test covers this. Clusters that do not fit are sampled locally.
   - **Before submitting**, the run prints the circuit's width and checks it has exactly the packed number of qubits and classical bits. The circuit is only rewritten to basis gates, with no backend layout, so nothing pads it; IonQ compiles it afterwards.
   - **IonQ qubit budget:** the budget is `min(backend.num_qubits, IONQ_QUBIT_BUDGET)`, and the default `IONQ_QUBIT_BUDGET` is **12**, i.e. the largest cluster alone.
     - Why: on Qollab's IonQ Aria 1 simulator, a 21-qubit packed circuit (clusters 1 and 2) was accepted but failed with `NotEnoughQubits`, although 21 < 25. We verified that the submitted circuit had exactly 21 qubits and 21 classical bits; the cause is on the IonQ side and still unknown.
     - Because IonQ allows one job per run, a failed job cannot be retried, so the default is the size known to work.
     - To raise it, set `"ionq_qubit_budget"` in `fsq/config.py`, e.g. to 21 for clusters 1 and 2, once that size has been verified on the selected IonQ backend.
   - **On local simulators:** clusters go as a list of separate circuits in one job. They are not packed, because a ~24-qubit statevector is too heavy for the browser.
   - **If a backend job fails:** nothing more is sent. Everything is sampled locally, and the results row says `sampled locally - … job failed`. The label never names a backend that was not used.
5. **Stitch and repair.**
   - For each cluster, take the **highest-value feasible** sample; if no sample is feasible, take the most frequent.
   - Stitch the clusters together. Conflicts between clusters can only occur on the edges cut by the partition (233 of 439 in the default run).
   - **Repair:** repeatedly drop the lowest-value candidate in a violated conflict, then greedily fill by value density.
   - The console logs the **net change**: drops and adds versus the stitched schedule.

The **default Run** uses a small instance: the top 10 districts over the first 24 h. It has 68 candidates and 8 clusters of 5–12 qubits, with precomputed angles, so the browser does no optimisation.

## 5. Results

The benchmark has 21 instances: the real event plus 20 seeded perturbations. Each perturbation shifts every priority by −1, 0 or +1, jitters each pass time by N(0, 20 s) and jitters each incidence angle by N(0, 1°).

### Approximation ratio (schedule value / exact optimum)

| Method | Real event | Mean ± sd (21) | Worst |
| --- | --- | --- | --- |
| Greedy | 0.994 | 0.984 ± 0.024 | 0.899 |
| Simulated annealing (1,024 restarts) | 0.996 | **0.997 ± 0.002** | 0.993 |
| Hybrid QAOA (warm start, p=1) | 0.966 | 0.949 ± 0.038 | 0.886 |
| *Control: same decomposition, exact per cluster* | 0.966 | 0.949 ± 0.038 | 0.886 |

![Schedule quality on 21 instances](https://raw.githubusercontent.com/DurgaPritam/qollab-hackathon/main/docs/figures/ratios.png)

**Decomposition insight.** The hybrid matches the control (same split, exact answer per cluster) on **all 21 instances**: 0.949 vs 0.949. QAOA finds the best answer inside every cluster, so the **gap to the global optimum comes from splitting the problem, not from QAOA**. The live run computes and prints this from `results/e1.json`.

With 1,024 shots, QAOA finds every cluster's optimum, so the loss comes from decomposition. The real conflict graph is one connected component, and in our partition into ≤12-qubit clusters **75% of conflicts (2,534 of 3,380) cross cluster boundaries**. Repair handles those conflicts but loses value in doing so.

### Conditioned large-neighbourhood search (LNS): the fix for the decomposition gap

**Method.**
- Start from the greedy schedule.
- Each sweep builds neighbourhoods around the current schedule: release a few selected images, plus every unselected image blocked only by them, keeping each neighbourhood at most 12 qubits.
- Each neighbourhood is solved *given* the rest of the schedule, so no conflict outside it can be violated.
- A proposal is accepted only if it raises the total value, so the result is never worse than greedy and always feasible (both tested).
- A fixed partition gave **zero** gain even with an exact sub-solver: moving a district's image means dropping a pick in one cluster and adding one in another, which no single fixed cluster can see.

Same 21 instances, same seeds, 10 sweeps, every sub-solver at the same budget per neighbourhood (`preprocessing/run_lns.py` → `results/e_lns.json`):

| Sub-solver inside LNS | Budget per neighbourhood (its own units) | Mean ± sd | Worst | Optimal |
| --- | --- | --- | --- | --- |
| Greedy (no LNS) | – | 0.984 ± 0.024 | 0.899 | – |
| Exact (ceiling), 12 qubits | brute force over 2ⁿ | 0.993 ± 0.009 | 0.960 | 4/21 |
| QAOA (warm start, CVaR, p=1), 12 qubits | 1,024 shots | 0.993 ± 0.009 | 0.960 | 4/21 |
| SA, 12 qubits | 1,024 restarts × 100 sweeps | 0.993 ± 0.009 | 0.960 | 4/21 |
| Uniform random **feasible** schedules, 12 qubits | 1,024 draws | 0.993 ± 0.009 | 0.960 | 4/21 |
| Exact (ceiling), 12 qubits, rerun beside the 32-budget rows | – | 0.993 ± 0.009 | 0.960 | 4/21 |
| QAOA, 12 qubits | **32 shots** | 0.993 ± 0.009 | 0.960 | 4/21 |
| SA, 12 qubits | **32 restarts** × 100 sweeps | 0.993 ± 0.009 | 0.960 | 4/21 |
| Uniform random feasible, 12 qubits | **32 draws** | 0.988 ± 0.016 | 0.948 | 2/21 |
| Uniform random **bitstrings** (all 2ⁿ), 12 qubits | **32 draws** | 0.984 ± 0.024 | 0.900 | 0/21 |
| Exact / SA / random feasible, **16 qubits** | as above, 1,024 | 0.995 ± 0.009 | 0.960 | 7/21 |

What the table says:
- **The neighbourhood structure is the limit, not the sub-solver.** At 1,024 samples, QAOA, exact, SA and random feasible are identical. At 32, QAOA and SA still match the exact ceiling.
- **Larger neighbourhoods help.** 16 qubits beat 12, so more candidates per qubit is the clearest path to better schedules (Step 4: binary encoding).
- **Costs are not comparable one-to-one.**
  - A QAOA shot is one run of a p=1 circuit, whose angles are tuned offline on a statevector.
  - An SA restart is 100 sweeps of n single-spin updates.
  - "Uniform random feasible" first enumerates all 2ⁿ assignments to list the feasible ones, which is itself an exponential classical step. "Uniform random bitstrings" draws from all 2ⁿ assignments without enumeration and rejects infeasible ones.

**Sampler quality on identical subproblems** (354 subproblems of 9–12 candidates):

| Sampler | P(optimum) per sample, mean | Hit within 8 | Hit within 32 |
| --- | --- | --- | --- |
| QAOA, one shot | 0.191 | 0.81 | 0.998 |
| SA, one restart (100 sweeps) | 0.702 | 0.995 | 1.000 |
| Uniform random feasible, one draw | 0.012 | 0.09 | 0.32 |
| Uniform random bitstring, one draw | 0.0004 | 0.003 | 0.012 |

QAOA beats uniform random feasible sampling on all 354 subproblems (median 17×). SA beats QAOA per sample.

**Live run, stage by stage** (default instance, seed 7, the project's fixed seed): greedy incumbent **0.988** → one QAOA-sampled LNS sweep in one job **0.995** → 5 classical LNS sweeps **1.000**. Across 20 seeds (`preprocessing/run_live_seeds.py`), the QAOA-sampled sweep gives 0.995 on 17 and the optimum on 3; after the classical sweeps, 8 of 20 reach 1.000.

### Resource estimate: the whole problem as one QAOA circuit

`preprocessing/run_resources.py` → `results/e_resources.json`. Every circuit is compiled directly (Qiskit, level 3, seed 42, basis rz/sx/x/cx), with no extrapolation. The success column uses a simple gate-error-only model, (1 − e2)^gates, and is **not a hardware prediction**: it ignores single-qubit and readout errors, decoherence over the depth, and the number of shots needed.

| Instance | Qubits, one-hot | Qubits, binary (analytic) | 2q gates p=1: all-to-all / grid / heavy-hex | Depth p=1 (all-to-all) | e2 for success 0.5 (all-to-all / grid) |
| --- | --- | --- | --- | --- | --- |
| 6 h | 44 | 14 | 714 / 1,379 / 1,687 | 177 | 9.7e-4 / 5.0e-4 |
| Default live (10 districts, 24 h) | 68 | 34 | 878 / 2,175 / 2,996 | 292 | 7.9e-4 / 3.2e-4 |
| 24 h | 151 | 45 | 3,328 / 9,883 / 13,468 | 668 | 2.1e-4 / 7.0e-5 |
| **Full event, 48 h** | **268** | **92** | **6,760 / 22,905 / 32,124** | 1,153 | **1.0e-4 / 3.0e-5** |

At p = 2, the full event needs 13,520 / 50,960 / 70,590 two-qubit gates (all-to-all / grid / heavy-hex).

The binary column is a **count only**:
- Cliques of the conflict graph are "at most one" groups. A greedy clique cover gives 25 groups for the full event, and a group of k candidates needs ⌈log₂(k+1)⌉ qubits.
- That encoding adds multi-qubit terms whose gate cost is not included here.

![Resource estimate](https://raw.githubusercontent.com/DurgaPritam/qollab-hackathon/improve/docs/figures/resources.png)

**Reading it.** The full event as a single circuit needs 268 qubits and about 6,800 two-qubit gates on all-to-all hardware. Under the simple model, a useful single-shot success (0.5) needs two-qubit error near **1e-4** on all-to-all and near **3e-5** on a grid, about 3× lower because of SWAP overhead. Current commercial two-qubit errors are around 1e-3 or higher, so whole-problem QAOA is out of reach. The LNS pipeline sidesteps this by using ≤12-qubit circuits.

### Fair diversity comparison: same QUBO, same 1,024-sample budget

The question: does QAOA sampling give more distinct good backup plans, useful if priorities change?

| Level / method | Feasible rate | Distinct schedules ≥95% of optimum | Diversity (mean Hamming) |
| --- | --- | --- | --- |
| Per cluster: QAOA (1,024 shots) | 0.70 ± 0.05 | **1.37 ± 1.00** | 0.37 ± 0.77 |
| Per cluster: SA (1,024 restarts, 100 sweeps) | **1.00** | 1.26 ± 0.72 | 0.36 ± 0.81 |
| Full schedule: hybrid (stitched + repaired) | 1.00 | 136 ± 96 | 18.8 ± 1.8 |
| Full schedule: SA (1,024 restarts, 1,000 sweeps) | 1.00 | **303 ± 96** | **26.4 ± 1.4** |

![QAOA vs SA at the same sample budget](https://raw.githubusercontent.com/DurgaPritam/qollab-hackathon/main/docs/figures/diversity.png)

**Per cluster** (670 clusters with ≥4 candidates), QAOA averages **1.37** distinct schedules within 95% of optimal against SA's **1.26**, and the **error bars overlap**. QAOA had more in 59 clusters, fewer in 1, and the same number in 610. That is a small edge at best, and it costs feasibility: only **70%** of QAOA samples are feasible, against **100%** for SA.

**For whole schedules, SA wins on diversity**: **303** distinct near-optimal schedules against **136** for the hybrid, and they are more varied (Hamming 26.4 vs 18.8). We report both levels because the per-cluster result alone would overstate what QAOA delivers.

### Connectivity: what IonQ's all-to-all connectivity buys

We took subgraphs of the real conflict graph and compiled the QAOA circuit with Qiskit (`optimization_level=3`, basis rz/sx/x/cx, `seed_transpiler=42`) for three layouts:
- **all-to-all**, as on trapped-ion hardware;
- a **square grid**;
- **heavy-hex**.

![Two-qubit gates: all-to-all vs grid vs heavy-hex](https://raw.githubusercontent.com/DurgaPritam/qollab-hackathon/main/docs/figures/connectivity_p1.png)

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

**Better partitions are not just fewer cut conflicts.** We tried a partition that fills clusters up to 12 qubits, with uneven Kernighan–Lin splits and 5–10 restarts. On 8 instances it cut fewer conflicts (2,127 vs 2,461 on average), but schedule quality got *worse*: 0.955 vs 0.975 with exact per-cluster solves. Which conflicts are cut matters more than how many, so we kept the original partition. A value-aware partition is the obvious next step.

**When quantum could matter:** conditioned LNS closes most of the decomposition gap with any sub-solver, and larger neighbourhoods help, so the lever is more candidates per circuit (larger devices or a denser encoding), not a better sampler. Quantum help would need:
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
python3.12 -m venv .venv && .venv/bin/pip install -r preprocessing/requirements-offline.txt qiskit-aer matplotlib
.venv/bin/python preprocessing/compute_windows.py     # orbits -> data/windows.json
.venv/bin/python preprocessing/run_experiments.py     # E1, E-DIV, E-CONN, E2, E3, E5 -> results/  (~5 min, 8 cores)
.venv/bin/python preprocessing/run_noise.py           # E4 -> results/e4.json
.venv/bin/python preprocessing/run_lns.py             # conditioned LNS + controls -> results/e_lns.json (~5 min)
.venv/bin/python preprocessing/run_live_seeds.py      # live recipe over 20 seeds -> results/e_live_seeds.json
.venv/bin/python preprocessing/run_resources.py       # resource estimate -> results/e_resources.json
.venv/bin/python preprocessing/save_figures.py        # -> docs/figures/*.png
.venv/bin/python preprocessing/build_ui_data.py       # -> ui/data.js (optional interactive page)
.venv/bin/python -m pytest -q tests                   # all tests
.venv/bin/python preprocessing/package_qollab.py      # -> dist/floodsight-q.zip for Qollab
```

## Repository layout

| Path | Contents |
| --- | --- |
| `main.py` | Qollab entry point: Run calls `main(shots, charts)` |
| `fsq/` | Python package |
| `fsq/api.py` | Run-button entry point |
| `fsq/formulate.py` | Candidates, values, conflicts, QUBO and Ising forms |
| `fsq/baselines.py` | Greedy, SA, exact |
| `fsq/decompose.py` | Clustering, stitch and repair |
| `fsq/qaoa.py` | Warm/cold QAOA, CVaR, one-job sampling |
| `fsq/connectivity.py` | Connectivity comparison |
| `fsq/diversity.py`, `fsq/metrics.py` | Diversity comparison and metrics |
| `fsq/plots.py` | Figures, used live and saved to `docs/figures/` |
| `data/` | Shipped data and its README |
| `results/` | Experiment outputs (JSON) |
| `preprocessing/` | Offline scripts, CPython only |
| `tests/` | pytest suite |
| `index.html`, `main.js`, `main.css`, `ui/` | Optional interactive web page (GitHub only). It is left out of the Qollab upload so Qollab runs `main.py` as a Python project. |

**Links:** demo video, *to be added* · repository: <https://github.com/DurgaPritam/qollab-hackathon>

MIT licence: see `LICENSE`.
