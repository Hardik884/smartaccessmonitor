import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# MODEL 1: ISOLATION FOREST (Anomaly Detection)
# ─────────────────────────────────────────────

def train_isolation_forest(df):
    print("\n📊 Training Isolation Forest (Anomaly Detection)...")

    # Features for anomaly detection
    # - hour: what time of day
    # - day_of_week: which day
    # - stay_duration_min: how long they stayed
    # - rssi: signal strength (proxy for how close they were)
    features = ["hour", "day_of_week", "stay_duration_min", "rssi"]

    # Only train on normal entries (not unknown/unauthorized)
    train_df = df[df["person"] != "UNKNOWN"][features].copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df)

    # contamination = expected fraction of anomalies in real data
    model = IsolationForest(
        n_estimators=100,
        contamination=0.08,
        random_state=42
    )
    model.fit(X_train)

    # Test on full dataset
    X_all = scaler.transform(df[features])
    scores = model.decision_function(X_all)   # more negative = more anomalous
    predictions = model.predict(X_all)        # -1 = anomaly, 1 = normal

    # Normalize score to 0-100 (higher = more anomalous)
    anomaly_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min())
    anomaly_scores = (anomaly_scores * 100).round(1)

    df = df.copy()
    df["anomaly_score"] = anomaly_scores
    df["if_prediction"] = predictions  # -1 = anomaly, 1 = normal

    # Evaluate
    known_anomalies = df[df["is_anomaly"] == 1]
    caught = known_anomalies[known_anomalies["if_prediction"] == -1]
    precision = len(caught) / len(known_anomalies) * 100 if len(known_anomalies) > 0 else 0

    print(f"   ✅ Model trained on {len(train_df)} normal records")
    print(f"   ✅ Anomaly detection rate: {precision:.1f}% of known anomalies caught")
    print(f"   ✅ High anomaly score (>70) events: {len(df[df['anomaly_score'] > 70])}")

    # Save model
    with open("isolation_forest.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "features": features}, f)
    print("   ✅ Saved: isolation_forest.pkl")

    return df, model, scaler

# ─────────────────────────────────────────────────
# MODEL 2: KMEANS (Behavioral Profiling per Person)
# ─────────────────────────────────────────────────

def train_behavioral_profiles(df):
    print("\n🧠 Training KMeans Behavioral Profiles...")

    profiles = {}
    authorized_people = [p for p in df["person"].unique() if p != "UNKNOWN"]

    for person in authorized_people:
        person_df = df[
            (df["person"] == person) &
            (df["action"] == "ENTER") &
            (df["is_anomaly"] == 0)  # only train on normal behavior
        ].copy()

        if len(person_df) < 3:
            print(f"   ⚠️  Not enough data for {person}, skipping")
            continue

        # Features that define a person's behavioral fingerprint
        features = ["hour", "day_of_week", "stay_duration_min"]
        X = person_df[features].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Cluster their behavior into patterns
        # e.g., Hardik has "morning session" and "afternoon session" clusters
        n_clusters = min(2, len(person_df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X_scaled)

        # Calculate normal deviation range
        distances = []
        for i, point in enumerate(X_scaled):
            cluster = kmeans.labels_[i]
            center = kmeans.cluster_centers_[cluster]
            dist = np.linalg.norm(point - center)
            distances.append(dist)

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        # Threshold: anything more than 2.5 std devs from cluster center = suspicious
        threshold = mean_dist + 2.5 * std_dist

        profiles[person] = {
            "model": kmeans,
            "scaler": scaler,
            "features": features,
            "threshold": threshold,
            "mean_dist": mean_dist,
            "std_dist": std_dist,
            "normal_hours": (int(person_df["hour"].min()), int(person_df["hour"].max())),
            "usual_days": sorted(person_df["day_of_week"].unique().tolist()),
            "avg_stay": float(person_df["stay_duration_min"].mean()),
        }

        print(f"   ✅ {person}: normal hours {profiles[person]['normal_hours']}, "
              f"avg stay {profiles[person]['avg_stay']:.0f} min, "
              f"threshold {threshold:.2f}")

    # Save profiles
    with open("behavioral_profiles.pkl", "wb") as f:
        pickle.dump(profiles, f)
    print("   ✅ Saved: behavioral_profiles.pkl")

    return profiles

# ─────────────────────────────────────────────────────────────────
# COMBINED SCORING: Run both models on a new event
# ─────────────────────────────────────────────────────────────────

def score_event(event: dict, if_bundle=None, profiles=None):
    """
    Score a new incoming event using both models.
    event = {
        "person": "Hardik",
        "mac": "79:47:0e:74:97:a1",
        "action": "ENTER",
        "hour": 14,
        "day_of_week": 2,
        "stay_duration_min": 90,
        "rssi": -55
    }
    Returns: dict with anomaly_score, behavioral_score, alert_level, reason
    """
    result = {
        "anomaly_score": 0,
        "behavioral_score": 0,
        "alert_level": "NORMAL",
        "reasons": []
    }

    # Load models if not passed
    if if_bundle is None:
        with open("isolation_forest.pkl", "rb") as f:
            if_bundle = pickle.load(f)
    if profiles is None:
        with open("behavioral_profiles.pkl", "rb") as f:
            profiles = pickle.load(f)

    # ── MODEL 1: Isolation Forest score ──
    features = if_bundle["features"]
    X = np.array([[event[f] for f in features]])
    X_scaled = if_bundle["scaler"].transform(X)
    score = if_bundle["model"].decision_function(X_scaled)[0]
    # Normalize: more negative = higher anomaly score
    anomaly_score = min(100, max(0, int((-score + 0.3) * 100)))
    result["anomaly_score"] = anomaly_score

    if anomaly_score > 70:
        result["reasons"].append(f"Unusual pattern detected (score: {anomaly_score})")

    # ── MODEL 2: Behavioral Profile check ──
    person = event.get("person", "UNKNOWN")

    if person == "UNKNOWN":
        result["behavioral_score"] = 100
        result["reasons"].append("Unknown/unauthorized device")
    elif person in profiles:
        profile = profiles[person]
        features_bp = profile["features"]
        X_bp = np.array([[event[f] for f in features_bp]])
        X_bp_scaled = profile["scaler"].transform(X_bp)

        # Find closest cluster center
        distances = [np.linalg.norm(X_bp_scaled[0] - center)
                     for center in profile["model"].cluster_centers_]
        min_dist = min(distances)

        # Normalize to 0-100
        behavioral_score = min(100, int((min_dist / profile["threshold"]) * 50))
        result["behavioral_score"] = behavioral_score

        # Specific reason detection
        hour = event.get("hour", 12)
        normal_start, normal_end = profile["normal_hours"]
        if hour < normal_start - 2 or hour > normal_end + 2:
            result["reasons"].append(
                f"{person} usually enters between {normal_start}:00-{normal_end}:00, "
                f"but this entry is at {hour}:00"
            )

        dow = event.get("day_of_week", 0)
        if dow not in profile["usual_days"]:
            days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            result["reasons"].append(
                f"{person} doesn't usually come on {days[dow]}"
            )

        if behavioral_score > 60:
            result["reasons"].append(
                f"Behavioral pattern mismatch for {person} — possible stolen credential"
            )

    # ── Combined alert level ──
    combined = max(result["anomaly_score"], result["behavioral_score"])
    if person == "UNKNOWN" or combined >= 80:
        result["alert_level"] = "HIGH"
    elif combined >= 50:
        result["alert_level"] = "MEDIUM"
    else:
        result["alert_level"] = "NORMAL"

    return result

# ─────────────────────────────────────────────
# MAIN: Train everything and show sample output
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Smart Access Monitor — ML Training")
    print("=" * 55)

    # Load dataset
    if not os.path.exists("access_log.csv"):
        print("❌ access_log.csv not found. Run generate_data.py first!")
        exit()

    df = pd.read_csv("access_log.csv")
    print(f"\n✅ Loaded {len(df)} records from access_log.csv")

    # Train models
    df, if_model, if_scaler = train_isolation_forest(df)
    profiles = train_behavioral_profiles(df)

    # Save enriched dataset
    df.to_csv("access_log_scored.csv", index=False)
    print("\n✅ Saved scored dataset: access_log_scored.csv")

    # ── Demo: score some test events ──
    print("\n" + "=" * 55)
    print("  DEMO: Scoring test events")
    print("=" * 55)

    test_events = [
        {
            "label": "✅ Normal — Hardik enters at 10am Monday",
            "person": "Hardik", "mac": "79:47:0e:74:97:a1",
            "action": "ENTER", "hour": 10, "day_of_week": 0,
            "stay_duration_min": 90, "rssi": -55
        },
        {
            "label": "🚨 Anomaly — Hardik enters at 2am Sunday",
            "person": "Hardik", "mac": "79:47:0e:74:97:a1",
            "action": "ENTER", "hour": 2, "day_of_week": 6,
            "stay_duration_min": 90, "rssi": -55
        },
        {
            "label": "🚨 Anomaly — Unknown device tries to enter",
            "person": "UNKNOWN", "mac": "ff:12:34:56:78:90",
            "action": "UNAUTHORIZED", "hour": 14, "day_of_week": 1,
            "stay_duration_min": 0, "rssi": -70
        },
        {
            "label": "⚠️  Suspicious — Ekaansh enters Saturday at midnight",
            "person": "Ekaansh", "mac": "aa:bb:cc:dd:ee:01",
            "action": "ENTER", "hour": 0, "day_of_week": 5,
            "stay_duration_min": 200, "rssi": -60
        },
        {
            "label": "✅ Normal — Prof enters at 9am Tuesday",
            "person": "Prof", "mac": "aa:bb:cc:dd:ee:02",
            "action": "ENTER", "hour": 9, "day_of_week": 1,
            "stay_duration_min": 120, "rssi": -50
        },
    ]

    if_bundle = {"model": if_model, "scaler": if_scaler, "features": ["hour", "day_of_week", "stay_duration_min", "rssi"]}

    for event in test_events:
        print(f"\n{event['label']}")
        result = score_event(event, if_bundle=if_bundle, profiles=profiles)
        print(f"   Anomaly Score:    {result['anomaly_score']}/100")
        print(f"   Behavioral Score: {result['behavioral_score']}/100")
        print(f"   Alert Level:      {result['alert_level']}")
        if result["reasons"]:
            for r in result["reasons"]:
                print(f"   Reason:           {r}")

    print("\n✅ All models ready. Run dashboard.py to start live monitoring.")