"""All tunable settings in one place. Every experiment records the config it used."""

CONFIG = {
    # Geometry / windows (used offline by preprocessing/compute_windows.py)
    "inc_min_deg": 20.0,          # usable SAR incidence range
    "inc_max_deg": 45.0,
    "best_inc_deg": 32.5,         # image quality peaks here
    "quality_width_deg": 10.0,    # Gaussian width of the quality curve
    # Scheduling
    "dwell_s": 10.0,              # imaging slot length, centred on broadside time
    "slew_s": 60.0,               # time a satellite needs between two images
    "tau_h": 24.0,                # urgency decay constant (hours)
    "penalty_mult": 1.2,          # P = penalty_mult * max(w)
    # Hybrid quantum
    "max_cluster": 12,            # qubits per cluster in live runs
    "eps": 0.25,                  # warm-start regularisation
    "alpha": 0.2,                 # CVaR fraction
    "p": 1,                       # QAOA depth for the default run
    "shots": 1024,
    "ionq_qubit_budget": 25,      # IONQ_QUBIT_BUDGET: used when the backend does not report num_qubits
    "seed": 7,
}
