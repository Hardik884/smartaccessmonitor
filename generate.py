"""
Generate a synthetic access log for training and evaluating the ML models.

    python generate.py                # 180 days -> access_log.csv
    python generate.py --days 180 --seed 7

Each person in config.AUTHORIZED_KEYS gets a weekly routine (arrival time,
stay length, which days they come, occasional second visit). A small share of
visits are labeled anomalies of a specific type so the models can be scored:

    odd_hour     entry far outside the person's usual arrival time
    odd_day      entry on a day the person essentially never comes
    odd_stay     exit after a stay far shorter or longer than usual
    weak_signal  key detected at the very edge of range (key not on the person)
    no_key       someone crossed without any authorized key (unauthorized)
"""

import argparse
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import AUTHORIZED_KEYS, TRAINING_CSV, UNKNOWN_PERSON

NORMAL_RSSI = (-56, 6)       # mean, std in dBm
EDGE_RSSI = (-75, -70)       # just above the firmware's cut-off


def _rssi(rng):
    return int(np.clip(rng.normal(*NORMAL_RSSI), -69, -38))


def _visit(rng, day, key, profile, arrival_hour, anomaly_type=None):
    """Return the (entry, exit) records for one visit."""
    person = AUTHORIZED_KEYS[key]["person"]
    arrival = day + timedelta(hours=float(arrival_hour))

    if anomaly_type == "odd_stay":
        stay = rng.choice([rng.uniform(2, 8), profile["stay_min"] * rng.uniform(3, 4.5)])
    else:
        stay = max(10.0, rng.normal(profile["stay_min"], profile["stay_std"]))
    leave = arrival + timedelta(minutes=float(stay))

    entry_rssi = rng.integers(*EDGE_RSSI) if anomaly_type == "weak_signal" else _rssi(rng)
    entry_label = anomaly_type if anomaly_type in ("odd_hour", "odd_day", "weak_signal") else ""
    exit_label = "odd_stay" if anomaly_type == "odd_stay" else ""

    base = {"person": person, "ble_name": key}
    return [
        {**base, "timestamp": arrival, "action": "ENTER", "rssi": int(entry_rssi),
         "stay_duration_min": None, "anomaly_type": entry_label},
        {**base, "timestamp": leave, "action": "EXIT", "rssi": _rssi(rng),
         "stay_duration_min": round(float(stay), 1), "anomaly_type": exit_label},
    ]


def generate_dataset(days=180, anomaly_rate=0.08, seed=42, start=datetime(2026, 1, 5)):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    records = []

    for offset in range(days):
        day = start + timedelta(days=offset)
        dow = day.weekday()

        for key, cfg in AUTHORIZED_KEYS.items():
            p = cfg["profile"]
            comes = rng.random() < p["attendance"][dow]
            anomaly = rng.random() < anomaly_rate

            if anomaly:
                kind = rng.choice(["odd_hour", "odd_day", "odd_stay", "weak_signal"])
                if kind == "odd_day":
                    unusual = [d for d in range(7) if p["attendance"][d] < 0.05]
                    if not unusual or dow not in unusual:
                        kind = "odd_hour"
                if kind == "odd_hour":
                    hour = (p["arrival_hour"] + rng.choice([-1, 1]) * rng.uniform(6, 10)) % 24
                else:
                    hour = rng.normal(p["arrival_hour"], p["arrival_std"])
                if kind == "odd_day" or comes:
                    records += _visit(rng, day, key, p, hour, kind)
                    continue

            if not comes:
                continue

            hour = rng.normal(p["arrival_hour"], p["arrival_std"])
            first = _visit(rng, day, key, p, hour)
            records += first
            if rng.random() < p["second_visit"]:
                back = first[1]["timestamp"] + timedelta(minutes=float(rng.uniform(30, 150)))
                back_hour = back.hour + back.minute / 60
                if back_hour < 21:
                    records += _visit(rng, day, key, p, back_hour)

        # Unauthorized crossings: no key in range
        if rng.random() < 0.12:
            when = day + timedelta(hours=float(rng.uniform(0, 24)))
            records.append({
                "timestamp": when, "person": UNKNOWN_PERSON, "ble_name": "",
                "action": "ENTER", "rssi": None, "stay_duration_min": None,
                "anomaly_type": "no_key",
            })

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    df["is_anomaly"] = (df["anomaly_type"] != "").astype(int)

    occupancy, inside = [], 0
    for action in df["action"]:
        inside = inside + 1 if action == "ENTER" else max(0, inside - 1)
        occupancy.append(inside)
    df["people_inside"] = occupancy
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--anomaly-rate", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=TRAINING_CSV)
    args = ap.parse_args()

    df = generate_dataset(args.days, args.anomaly_rate, args.seed)
    df.to_csv(args.out, index=False)

    print(f"Generated {len(df)} records over {args.days} days -> {args.out}")
    print(df.groupby(["person", "action"]).size().unstack(fill_value=0).to_string())
    counts = df.loc[df["is_anomaly"] == 1, "anomaly_type"].value_counts()
    print("\nLabeled anomalies:")
    print(counts.to_string())
