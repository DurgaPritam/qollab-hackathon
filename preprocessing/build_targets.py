"""Write data/targets.csv, data/region.geojson and data/meta.json. Offline, CPython only.

    curl -o /tmp/ne.geojson https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces.geojson
    .venv/bin/python preprocessing/build_targets.py /tmp/ne.geojson

District numbers are typed in from the Govt. of Assam Flood Memorandum 2022, tables 2.5-2.7
(PDF pages 17-21); see data/README.md. Coordinates: HQ town, Wikipedia (W) or Wikidata (D).
"""

import csv
import json
import os
import sys

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# district, HQ town, lat, lon, coord source, people affected, villages, deaths (None = not in table 2.6), relief-camp inmates
DISTRICTS = [
    ("Cachar", "Silchar", 24.82, 92.80, "W", 1434750, 826, 48, 303694),
    ("Barpeta", "Barpeta", 26.32, 91.00, "W", 1213060, 660, 18, 118713),
    ("Kamrup", "Amingaon", 26.1883, 91.6783, "D", 897119, 591, 8, 15978),
    ("Dhubri", "Dhubri", 26.02, 90.00, "W", 605805, 891, 5, 10307),
    ("Nagaon", "Nagaon", 26.3504, 92.6796, "W", 567942, 381, 29, 23663),
    ("Darrang", "Mangaldoi", 26.43, 92.03, "D", 557901, 498, 9, 8766),
    ("Nalbari", "Nalbari", 26.445, 91.44, "W", 499090, 425, 6, 33982),
    ("Bajali", "Pathsala", 26.4994, 91.1793, "D", 356000, 173, 3, 6909),
    ("Bongaigaon", "Bongaigaon", 26.4769, 90.5583, "W", 339481, 351, None, 10916),
    ("Karimganj", "Karimganj", 24.87, 92.35, "W", 308581, 729, 10, 29332),
    ("Goalpara", "Goalpara", 26.17, 90.62, "W", 284368, 465, 2, 38286),
    ("Hailakandi", "Hailakandi", 24.68, 92.57, "W", 256019, 186, 5, 6444),
    ("Morigaon", "Morigaon", 26.2522, 92.3422, "W", 196821, 420, 6, 32559),
    ("Lakhimpur", "North Lakhimpur", 27.2414, 94.1056, "D", 181029, 379, 7, 1268),
    ("South Salmara", "Hatsingimari", 25.68, 89.98, "D", 178954, 94, None, 474),
    ("Hojai", "Hojai", 26.0028, 92.8528, "D", 154480, 133, 9, 56952),
    ("Dhemaji", "Dhemaji", 27.48, 94.58, "W", 115938, 308, 2, 483),
    ("Kokrajhar", "Kokrajhar", 26.40, 90.27, "W", 97442, 442, 2, 20696),
    ("Sonitpur", "Tezpur", 26.6283, 92.7997, "W", 96167, 272, 2, 622),
    ("Udalguri", "Udalguri", 26.7452, 92.0962, "D", 80121, 323, 6, 6850),
    ("Biswanath", "Biswanath Chariali", 26.7278, 93.1517, "W", 77063, 224, 2, 637),
    ("Baksa", "Mushalpur", 26.6620, 91.3411, "D", 76625, 161, None, 2941),
    ("Kamrup Metro", "Guwahati", 26.1722, 91.7458, "W", 47532, 86, 7, 935),
    ("Karbi Anglong West", "Hamren", 25.9233, 92.6083, "W", 43432, 37, None, 275),
    ("Golaghat", "Golaghat", 26.52, 93.97, "W", 27952, 56, 2, 0),
]


def priority_rule(rows):
    """Quintile (1-5) of displacement share (camp inmates / people affected), +1 if >= 10 deaths, max 5."""
    share = np.array([r[8] / r[5] for r in rows])
    quint = 1 + (share.argsort().argsort() * 5) // len(rows)
    return [int(min(5, q + (1 if (r[7] or 0) >= 10 else 0))) for r, q in zip(rows, quint)], share


def main(ne_path):
    prio, share = priority_rule(DISTRICTS)
    out = []
    for r, p, sh in zip(DISTRICTS, prio, share):
        out.append({"target_id": "AS-" + "".join(w[:3].upper() for w in r[0].split())[:9], "name": r[0],
                    "lat": r[2], "lon": r[3], "population": r[5], "priority": p,
                    "source": f"pop/villages/deaths/camps: Assam Flood Memorandum 2022 tables 2.5-2.7; coords: "
                              f"{'Wikipedia' if r[4] == 'W' else 'Wikidata'} ({r[1]})",
                    "hq": r[1], "villages": r[6], "deaths": "" if r[7] is None else r[7],
                    "camp_inmates": r[8], "displacement_share": round(float(sh), 4)})
    with open("data/targets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    ne = json.load(open(ne_path))
    g = [f for f in ne["features"] if f["properties"].get("name") == "Assam"][0]["geometry"]
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    rings = [[[round(x, 4), round(y, 4)] for x, y in poly[0]] for poly in polys]
    json.dump({"type": "FeatureCollection", "features": [{"type": "Feature",
               "properties": {"name": "Assam", "source": "Natural Earth 50m admin-1 (public domain)"},
               "geometry": {"type": "MultiPolygon", "coordinates": [[r] for r in rings]}}]},
              open("data/region.geojson", "w"))
    json.dump({"population_label": "people affected (cumulative, 6 Apr-28 Oct 2022)",
               "footer": "Sources: Govt. of Assam / ASDMA Flood Memorandum 2022 (district tables); CelesTrak TLE "
                         "snapshot 2026-10-10; Natural Earth (public domain); Wikipedia/Wikidata coordinates. MIT licence. "
                         "Pass geometry is from the 2026 TLE snapshot replayed over the 2022 flood locations, "
                         "not the 2022 passes.",
               "region_rings": rings}, open("data/meta.json", "w"))
    print(f"{len(out)} targets written")


if __name__ == "__main__":
    main(sys.argv[1])
