"""Compute SAR imaging windows for each (target, satellite pass). Offline, CPython only.

Reproduce:
    python3.12 -m venv .venv && .venv/bin/pip install -r preprocessing/requirements-offline.txt
    .venv/bin/python preprocessing/compute_windows.py
Tested with Python 3.12, skyfield 1.55, sgp4 2.27, numpy 2.5.3.

Inputs:  data/tles.txt (CelesTrak snapshot, see data/README.md), data/targets.csv
Output:  data/windows.json

Geometry (skyfield SGP4 propagation; WGS84 ground points):
  * incidence angle at the target ~= 90 deg - elevation of the satellite seen from the
    target (exact for a flat local horizon; Earth-curvature correction at the ground point
    is already included because elevation is measured from the local WGS84 horizon).
  * a SAR images a point at broadside (zero Doppler), i.e. at closest approach. We keep a
    pass when the incidence at broadside lies in [inc_min, inc_max] (default 20-45 deg).
  * t_start / t_end: the part of the pass where incidence is inside the usable range.
  * look_side: 'right' or 'left' of the ground track in the direction of motion.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from skyfield.api import EarthSatellite, load, wgs84

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fsq.config import CONFIG  # noqa: E402

EVENT_START_UTC = datetime(2026, 10, 11, 0, 0, 0, tzinfo=timezone.utc)
HORIZON_H = 48
STEP_S = 10


def load_tles(path):
    lines = [l.rstrip() for l in open(path) if l.strip()]
    ts = load.timescale()
    return [EarthSatellite(lines[i + 1], lines[i + 2], lines[i].strip(), ts) for i in range(0, len(lines), 3)]


def main():
    ts = load.timescale()
    sats = load_tles("data/tles.txt")
    targets = list(csv.DictReader(open("data/targets.csv")))
    t0 = EVENT_START_UTC.timestamp()
    secs = np.arange(0, HORIZON_H * 3600 + 1, STEP_S)
    times = ts.utc(EVENT_START_UTC.year, EVENT_START_UTC.month, EVENT_START_UTC.day, 0, 0, secs)
    lo, hi = CONFIG["inc_min_deg"], CONFIG["inc_max_deg"]
    windows = []
    for sat in sats:
        sat_name = sat.name.split(" (")[0]
        for tg in targets:
            site = wgs84.latlon(float(tg["lat"]), float(tg["lon"]))
            alt, _, dist = (sat - site).at(times).altaz()
            elev = alt.degrees
            up = elev > 0
            # contiguous above-horizon runs = passes
            edges = np.flatnonzero(np.diff(up.astype(int)))
            starts = list(np.flatnonzero(up[:1])) + list(edges[up[edges + 1]] + 1)
            for s in starts:
                e = s
                while e + 1 < len(up) and up[e + 1]:
                    e += 1
                k = s + int(np.argmin(dist.km[s:e + 1]))
                # refine closest approach to 1 s
                fine = ts.utc(EVENT_START_UTC.year, EVENT_START_UTC.month, EVENT_START_UTC.day, 0, 0,
                              np.arange(secs[k] - STEP_S, secs[k] + STEP_S + 1, 1.0))
                a2, _, d2 = (sat - site).at(fine).altaz()
                j = int(np.argmin(d2.km))
                inc_b = 90.0 - a2.degrees[j]
                if not (lo <= inc_b <= hi):
                    continue
                t_b = t0 + float(secs[k] - STEP_S + j)
                inc = 90.0 - elev[s:e + 1]
                ok = np.flatnonzero((inc >= lo) & (inc <= hi))
                t_start, t_end = t0 + float(secs[s + ok[0]]), t0 + float(secs[s + ok[-1]])
                # look side
                tb = ts.from_datetime(datetime.fromtimestamp(t_b, tz=timezone.utc))
                gs = sat.at(tb)
                r, v = gs.position.km, gs.velocity.km_per_s
                tgt = site.at(tb).position.km
                right = np.cross(v, r / np.linalg.norm(r))
                side = "right" if np.dot(tgt - r, right) > 0 else "left"
                windows.append({
                    "satellite": sat_name, "target_id": tg["target_id"],
                    "t_start": t_start, "t_end": t_end, "t_broadside": t_b,
                    "best_incidence_deg": round(float(inc_b), 2), "look_side": side,
                })
    # pass ids: per satellite, broadside times more than 20 min apart start a new pass
    windows.sort(key=lambda w: (w["satellite"], w["t_broadside"]))
    last, idx = {}, {}
    for w in windows:
        s = w["satellite"]
        if s not in last or w["t_broadside"] - last[s] > 1200:
            idx[s] = idx.get(s, 0) + 1
        last[s] = w["t_broadside"]
        w["pass_id"] = f"{s}#{idx[s]}"
    windows.sort(key=lambda w: w["t_broadside"])
    out = {"event_start_utc": EVENT_START_UTC.isoformat(), "event_start_unix": t0,
           "horizon_h": HORIZON_H, "incidence_range_deg": [lo, hi],
           "tle_file": "data/tles.txt", "windows": windows}
    json.dump(out, open("data/windows.json", "w"), indent=1)
    sats_used = sorted({w["satellite"] for w in windows})
    print(f"{len(windows)} windows, {len({w['pass_id'] for w in windows})} passes, satellites: {sats_used}")


if __name__ == "__main__":
    main()
