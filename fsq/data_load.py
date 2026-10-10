"""Load shipped data files and build (optionally perturbed) instances.

File paths are plain string literals on purpose: the Qollab runtime only ships files
whose path appears literally in the code.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from .config import CONFIG
from .formulate import build_instance

TARGETS_CSV = "data/targets.csv"
WINDOWS_JSON = "data/windows.json"
DEFAULT_INSTANCE_JSON = "data/instance_default.json"


def load_targets() -> list[dict]:
    """Rows of data/targets.csv with numeric fields converted."""
    rows = list(csv.DictReader(open(TARGETS_CSV, encoding="utf-8")))
    for r in rows:
        r["lat"], r["lon"] = float(r["lat"]), float(r["lon"])
        r["population"], r["priority"] = float(r["population"]), int(r["priority"])
    return rows


def load_windows() -> dict:
    """Parsed data/windows.json."""
    return json.load(open(WINDOWS_JSON, encoding="utf-8"))


def load_default_instance() -> dict:
    """The precomputed default instance used by the Run button."""
    return json.load(open(DEFAULT_INSTANCE_JSON, encoding="utf-8"))


def make_instance(targets, wdata, horizon_h: float | None = None, target_ids=None,
                  instance_id: str = "real", cfg: dict = CONFIG) -> dict:
    """Instance from the real data, optionally restricted in time and to some targets."""
    t0 = wdata["event_start_unix"]
    wins = wdata["windows"]
    if horizon_h is not None:
        wins = [w for w in wins if w["t_broadside"] - t0 <= horizon_h * 3600]
    tg = targets
    if target_ids is not None:
        keep = set(target_ids)
        tg = [t for t in targets if t["target_id"] in keep]
    return build_instance(tg, wins, t0, instance_id, cfg)


def perturb(targets, wdata, seed: int, horizon_h: float | None = None, target_ids=None,
            cfg: dict = CONFIG) -> dict:
    """Seeded perturbed instance: each priority moves by -1/0/+1 (kept in 1..5),
    each broadside time is jittered by N(0, 20 s), each incidence by N(0, 1 deg)."""
    rng = np.random.default_rng(seed)
    tg = [dict(t, priority=int(np.clip(t["priority"] + rng.integers(-1, 2), 1, 5))) for t in targets]
    wins = []
    for w in wdata["windows"]:
        w2 = dict(w)
        w2["t_broadside"] = w["t_broadside"] + float(rng.normal(0, 20))
        w2["best_incidence_deg"] = float(np.clip(w["best_incidence_deg"] + rng.normal(0, 1),
                                                 cfg["inc_min_deg"], cfg["inc_max_deg"]))
        wins.append(w2)
    return make_instance(tg, dict(wdata, windows=wins), horizon_h, target_ids,
                         f"perturbed-{seed}", cfg)
