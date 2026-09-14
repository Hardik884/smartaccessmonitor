"""
Smart Access Monitor - anomaly and behavioral models
====================================================

    python generate.py                  # synthetic training data (access_log.csv)
    python ml_models.py                 # evaluate on a time-based holdout, then train and save
    python ml_models.py --db access_monitor.db   # also learn from real logged events

Two models, both unsupervised (they never see the anomaly labels while training):

1. Site model - Isolation Forest over every authorized entry. Features:
   time of day and day of week as sin/cos pairs (so 23:00 sits next to 00:00
   and Sunday next to Monday), a weekend flag and the key's RSSI at the door.
   It flags entries that don't look like anything the building normally sees.

2. Person profiles - one per key holder:
   - which days they come (smoothed day-of-week frequencies)
   - when they arrive (Gaussian mixture on the circular hour, component count
     picked by BIC, so someone with a morning and an afternoon slot gets two)
   - how strong their key's signal usually is at the door (robust median/MAD,
     low side only), since every tag and phone transmits at a different power
   - how long they stay (robust log-normal: median and MAD of log minutes),
     scored on EXIT using the real time since that person's entry

Raw model outputs are calibrated against the training data so the 0-100
numbers mean the same thing for every model: a typical event scores about 10,
an event just past the rarest 1% of normal training events reaches 50, and
anything well beyond that climbs toward 100. The overall risk is the strongest
of the scores, and the dashboard's rule checks (no key, tailgating, re-entry
without exit) can raise it further.

`python ml_models.py` prints a holdout evaluation (ROC-AUC, precision/recall,
false alarm rate and recall per anomaly type) before saving the models.
"""

import argparse
import math
import os
import pickle
import sqlite3
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from config import MODEL_FILE, RISK_HIGH, RISK_MEDIUM, TRAINING_CSV, UNKNOWN_PERSON

warnings.filterwarnings("ignore", category=UserWarning)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MIN_VISITS_FOR_PROFILE = 8
# Extra room beyond the training 1st percentile before a score reaches 50.
# 0.25 gave ~2% false alarms and ~92% recall on the synthetic holdout.
CALIBRATION_MARGIN = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_datetime(ts):
    if isinstance(ts, datetime):
        return ts
    return pd.Timestamp(ts).to_pydatetime()


def hour_of(ts):
    return ts.hour + ts.minute / 60 + ts.second / 3600


def cyclic(value, period):
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def site_features(ts, rssi):
    hs, hc = cyclic(hour_of(ts), 24)
    ds, dc = cyclic(ts.weekday(), 7)
    return [hs, hc, ds, dc, float(ts.weekday() >= 5), float(rssi)]


def fmt_hour(h):
    minutes = int(round(h * 60)) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# ─────────────────────────────────────────────────────────────────────────────
# Calibration: normality statistic (higher = more normal) -> 0-100 risk
# ─────────────────────────────────────────────────────────────────────────────

def fit_calibration(normality):
    q01, q50 = np.percentile(normality, [1, 50])
    return {"q01": float(q01), "spread": float(max(q50 - q01, 1e-6))}


def calibrated_risk(calib, value):
    # logistic through (median -> 10) and (1st percentile minus margin -> 50)
    center = calib["q01"] - CALIBRATION_MARGIN * calib["spread"]
    z = 2.197 * (value - center) / ((1 + CALIBRATION_MARGIN) * calib["spread"])
    return 100.0 / (1.0 + math.exp(max(-50.0, min(50.0, z))))


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def prepare(df):
    df = df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"])
    if "is_anomaly" not in df:
        df["is_anomaly"] = 0
    df["ble_name"] = df.get("ble_name", pd.Series("", index=df.index)).fillna("")
    return df


def fit_site_model(entries):
    rssi = entries["rssi"].astype(float)
    rssi_median = float(rssi.median()) if rssi.notna().any() else -60.0
    X = np.array([site_features(t, r) for t, r in zip(entries["ts"], rssi.fillna(rssi_median))])
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = IsolationForest(n_estimators=300, contamination="auto", random_state=42).fit(Xs)
    return {
        "model": model,
        "scaler": scaler,
        "calib": fit_calibration(model.score_samples(Xs)),
        "rssi_median": rssi_median,
        "n": len(entries),
    }


def fit_hour_mixture(hours):
    X = np.array([cyclic(h, 24) for h in hours])
    best, best_bic = None, np.inf
    for k in range(1, 4):
        if len(X) < 6 * k:
            break
        gmm = GaussianMixture(k, covariance_type="full", reg_covar=1e-3, n_init=3, random_state=0).fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best, best_bic = gmm, bic
    return best


def fit_person_profile(entries, exits):
    hours = np.array([hour_of(t) for t in entries["ts"]])
    dows = np.array([t.weekday() for t in entries["ts"]])

    day_counts = np.bincount(dows, minlength=7).astype(float)
    day_prob = (day_counts + 0.5) / (day_counts.sum() + 3.5)

    gmm = fit_hour_mixture(hours)
    X = np.array([cyclic(h, 24) for h in hours])
    loglik = np.log(day_prob[dows]) + gmm.score_samples(X)

    # typical arrival window, measured around the circular mean so it survives midnight
    center = (math.degrees(math.atan2(X[:, 0].mean(), X[:, 1].mean())) / 15) % 24
    offsets = (hours - center + 12) % 24 - 12
    lo, hi = np.percentile(offsets, [2.5, 97.5])

    profile = {
        "day_prob": day_prob,
        "day_counts": day_counts,
        "hour_gmm": gmm,
        "entry_calib": fit_calibration(loglik),
        "hour_center": float(center),
        "hour_window": (float(lo), float(hi)),
        "n_entries": int(len(entries)),
        "signal": None,
        "stay": None,
    }

    rssi = entries["rssi"].dropna().astype(float)
    if len(rssi) >= MIN_VISITS_FOR_PROFILE:
        median = float(rssi.median())
        sigma = float(max(1.4826 * np.median(np.abs(rssi - median)), 2.0))
        profile["signal"] = {
            "median": median,
            "sigma": sigma,
            "calib": fit_calibration(np.minimum(0.0, (rssi - median) / sigma)),
        }

    stays = exits["stay_duration_min"].dropna().astype(float)
    stays = stays[stays > 0]
    if len(stays) >= MIN_VISITS_FOR_PROFILE:
        logs = np.log(stays)
        mu = float(np.median(logs))
        sigma = float(max(1.4826 * np.median(np.abs(logs - mu)), 0.08))
        profile["stay"] = {
            "mu": mu,
            "sigma": sigma,
            "calib": fit_calibration(-np.abs(logs - mu) / sigma),
            "typical": (float(math.exp(mu - 1.5 * sigma)), float(math.exp(mu + 1.5 * sigma))),
        }
    return profile


def train_models(df, verbose=True):
    """Fit both models on the normal, authorized events in df."""
    df = prepare(df)
    normal = df[(df["is_anomaly"] == 0) & (df["person"] != UNKNOWN_PERSON) & (df["ble_name"] != "")]
    entries = normal[normal["action"] == "ENTER"]
    exits = normal[normal["action"] == "EXIT"]

    bundle = {"site": fit_site_model(entries), "profiles": {}, "trained_at": datetime.now().isoformat(timespec="seconds")}
    if verbose:
        print(f"  Site model: Isolation Forest on {len(entries)} entries")

    for person, person_entries in entries.groupby("person"):
        if len(person_entries) < MIN_VISITS_FOR_PROFILE:
            if verbose:
                print(f"  {person}: only {len(person_entries)} entries, skipping profile")
            continue
        profile = fit_person_profile(person_entries, exits[exits["person"] == person])
        bundle["profiles"][person] = profile
        if verbose:
            lo, hi = profile["hour_window"]
            c = profile["hour_center"]
            stay = profile["stay"]
            stay_txt = f"stay {stay['typical'][0]:.0f}-{stay['typical'][1]:.0f} min" if stay else "no stay model"
            print(f"  {person}: {profile['n_entries']} entries, arrives {fmt_hour(c + lo)}-{fmt_hour(c + hi)}, "
                  f"{profile['hour_gmm'].n_components} arrival cluster(s), {stay_txt}")
    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

class AccessScorer:
    """Scores one access event. Works without trained models (rules only)."""

    def __init__(self, bundle=None):
        self.bundle = bundle

    @classmethod
    def load(cls, path=MODEL_FILE):
        if not os.path.exists(path):
            return cls(None)
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    @property
    def has_models(self):
        return self.bundle is not None

    def score(self, event, rule_hits=()):
        """
        event: {timestamp, person, action ("ENTER"/"EXIT"), authorized (bool),
                rssi (dBm or None), stay_duration_min (EXIT only, or None),
                reason ("key" / "no_key" / "tailgate", optional)}
        rule_hits: iterable of (risk_floor, reason) from stateful checks in the caller.
        """
        ts = to_datetime(event["timestamp"])
        person = event.get("person") or UNKNOWN_PERSON
        action = event.get("action", "ENTER")
        rssi = event.get("rssi")
        anomaly = behavioral = 0.0
        floor = 0.0
        reasons = []

        if not event.get("authorized"):
            no_key = event.get("reason", "no_key") == "no_key"   # tailgating explains itself via rule_hits
            if action == "ENTER":
                floor = 100.0
                if no_key:
                    reasons.append("No authorized key in range")
            elif no_key:
                reasons.append("Left without a key in range")
        elif self.bundle:
            site = self.bundle["site"]
            profile = self.bundle["profiles"].get(person)

            if action == "ENTER":
                x = site_features(ts, rssi if rssi is not None else site["rssi_median"])
                normality = site["model"].score_samples(site["scaler"].transform([x]))[0]
                anomaly = calibrated_risk(site["calib"], normality)

                if profile:
                    behavioral, why = self._entry_behavior(profile, person, ts)
                    reasons += why
                    if profile["signal"] and rssi is not None:
                        weak, why = self._signal_behavior(profile["signal"], rssi)
                        behavioral = max(behavioral, weak)
                        reasons += why
                else:
                    reasons.append(f"No behavioral profile for {person} yet")

            elif profile and profile["stay"] and event.get("stay_duration_min") is not None:
                behavioral, why = self._stay_behavior(profile["stay"], event["stay_duration_min"])
                reasons += why

        for rule_floor, reason in rule_hits:
            floor = max(floor, rule_floor)
            reasons.append(reason)

        risk = max(anomaly, behavioral, floor)
        level = "HIGH" if risk >= RISK_HIGH else "MEDIUM" if risk >= RISK_MEDIUM else "NORMAL"
        return {
            "anomaly_score": int(round(anomaly)),
            "behavioral_score": int(round(behavioral)),
            "risk": int(round(risk)),
            "alert_level": level,
            "reasons": reasons,
        }

    @staticmethod
    def _entry_behavior(profile, person, ts):
        h, dow = hour_of(ts), ts.weekday()
        loglik = math.log(profile["day_prob"][dow]) + profile["hour_gmm"].score_samples([cyclic(h, 24)])[0]
        score = calibrated_risk(profile["entry_calib"], loglik)
        reasons = []

        day_visits = int(profile["day_counts"][dow])
        if day_visits == 0 and profile["n_entries"] >= 20:
            # never seen on this day across a meaningful history: always worth a look
            score = max(score, RISK_MEDIUM + 10)
            reasons.append(f"{person} has not come in on a {DAYS[dow]} before "
                           f"(0 of {profile['n_entries']} visits)")
        elif day_visits / profile["n_entries"] < 0.03:
            reasons.append(f"{person} rarely comes in on {DAYS[dow]}s "
                           f"({day_visits} of {profile['n_entries']} visits)")

        c = profile["hour_center"]
        lo, hi = profile["hour_window"]
        offset = (h - c + 12) % 24 - 12
        if offset < lo - 0.5 or offset > hi + 0.5:
            reasons.append(f"Arrival at {fmt_hour(h)} is outside {person}'s usual "
                           f"{fmt_hour(c + lo)}-{fmt_hour(c + hi)}")
        return score, reasons

    @staticmethod
    def _signal_behavior(signal, rssi):
        z = (rssi - signal["median"]) / signal["sigma"]
        score = calibrated_risk(signal["calib"], min(0.0, z))
        reasons = []
        if score >= RISK_MEDIUM:
            reasons.append(f"Key signal weak for this key ({rssi} dBm, usually around "
                           f"{signal['median']:.0f}); key may not be on the person")
        return score, reasons

    @staticmethod
    def _stay_behavior(stay, minutes):
        minutes = max(float(minutes), 0.5)
        normality = -abs(math.log(minutes) - stay["mu"]) / stay["sigma"]
        score = calibrated_risk(stay["calib"], normality)
        lo, hi = stay["typical"]
        reasons = []
        if score >= 40 and (minutes < lo or minutes > hi):
            reasons.append(f"Stayed {minutes:.0f} min (usually {lo:.0f}-{hi:.0f})")
        return score, reasons


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def score_frame(scorer, df):
    rows = []
    for r in df.itertuples():
        stay = r.stay_duration_min if r.action == "EXIT" and pd.notna(r.stay_duration_min) else None
        result = scorer.score({
            "timestamp": r.ts, "person": r.person, "action": r.action,
            "authorized": bool(r.ble_name), "rssi": None if pd.isna(r.rssi) else int(r.rssi),
            "stay_duration_min": stay,
        })
        rows.append(result)
    return pd.DataFrame(rows, index=df.index)


def evaluate(df, holdout=0.3):
    """Train on the earliest (1 - holdout) of the timeline, test on the rest."""
    df = prepare(df)
    cutoff = df["ts"].quantile(1 - holdout)
    train, test = df[df["ts"] < cutoff], df[df["ts"] >= cutoff]
    test = test[test["ble_name"] != ""]            # no-key events are caught by rule, not ML
    if test["is_anomaly"].nunique() < 2:
        print("  Not enough labeled anomalies in the holdout to evaluate.")
        return

    print(f"  Train: {len(train)} events before {cutoff:%Y-%m-%d}   Test: {len(test)} authorized events after")
    scores = score_frame(AccessScorer(train_models(train, verbose=False)), test)
    y = test["is_anomaly"].values

    print("\n  ROC-AUC (1.0 = perfect ranking, 0.5 = random)")
    for col, label in [("anomaly_score", "Site model (Isolation Forest)"),
                       ("behavioral_score", "Person profiles"),
                       ("risk", "Combined risk")]:
        print(f"    {label:<32} {roc_auc_score(y, scores[col]):.3f}")

    flagged = (scores["risk"] >= RISK_MEDIUM).astype(int).values
    p, r, f1, _ = precision_recall_fscore_support(y, flagged, average="binary", zero_division=0)
    print(f"\n  At the MEDIUM threshold (risk >= {RISK_MEDIUM}):")
    print(f"    precision {p:.2f}   recall {r:.2f}   F1 {f1:.2f}   "
          f"false alarms {int(((flagged == 1) & (y == 0)).sum())} of {int((y == 0).sum())} normal events")

    print("\n  Caught by anomaly type:")
    for kind, group in test[test["is_anomaly"] == 1].groupby("anomaly_type"):
        caught = int((scores.loc[group.index, "risk"] >= RISK_MEDIUM).sum())
        print(f"    {kind:<12} {caught}/{len(group)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def load_db_events(path):
    conn = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(
            "SELECT timestamp, person, ble_name, action, rssi, stay_min AS stay_duration_min "
            "FROM events WHERE source = 'live' AND authorized = 1", conn)
    finally:
        conn.close()
    df["is_anomaly"] = 0
    df["anomaly_type"] = ""
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the Smart Access Monitor models.")
    ap.add_argument("--csv", default=TRAINING_CSV, help="training data from generate.py")
    ap.add_argument("--db", help="also train on real events logged by the dashboard")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    frames = []
    if os.path.exists(args.csv):
        frames.append(pd.read_csv(args.csv))
        print(f"Loaded {len(frames[-1])} events from {args.csv}")
    if args.db:
        frames.append(load_db_events(args.db))
        print(f"Loaded {len(frames[-1])} live authorized events from {args.db}")
    if not frames:
        raise SystemExit(f"No training data. Run `python generate.py` first or pass --db.")
    data = pd.concat(frames, ignore_index=True)

    if not args.no_eval and os.path.exists(args.csv):
        print("\nEvaluation (time-based holdout on labeled data)")
        evaluate(pd.read_csv(args.csv))

    print("\nTraining on all normal events")
    bundle = train_models(data)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved {MODEL_FILE}")

    print("\nExample events")
    scorer = AccessScorer(bundle)
    examples = [
        ("Hardik enters Monday 09:40", {"timestamp": "2026-03-02 09:40", "person": "Hardik", "action": "ENTER", "authorized": True, "rssi": -55}),
        ("Hardik enters Sunday 02:10", {"timestamp": "2026-03-08 02:10", "person": "Hardik", "action": "ENTER", "authorized": True, "rssi": -55}),
        ("Prof enters with a weak key signal", {"timestamp": "2026-03-03 08:30", "person": "Prof", "action": "ENTER", "authorized": True, "rssi": -74}),
        ("Ekaansh leaves after 5 hours", {"timestamp": "2026-03-04 16:00", "person": "Ekaansh", "action": "EXIT", "authorized": True, "rssi": -58, "stay_duration_min": 300}),
        ("Someone enters with no key", {"timestamp": "2026-03-04 14:00", "person": UNKNOWN_PERSON, "action": "ENTER", "authorized": False, "rssi": None}),
    ]
    for label, event in examples:
        s = scorer.score(event)
        print(f"  {label:<38} risk {s['risk']:>3}  {s['alert_level']:<6}  "
              f"(site {s['anomaly_score']}, person {s['behavioral_score']})")
        for reason in s["reasons"]:
            print(f"      - {reason}")
