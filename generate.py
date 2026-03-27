import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Authorized people
PEOPLE = {
    "Hardik":  {"usual_hours": (9, 18),  "usual_days": [0,1,2,3,4],     "avg_stay_min": 90,  "mac": "79:47:0e:74:97:a1"},
    "Ekaansh": {"usual_hours": (10, 17), "usual_days": [0,1,2,3,4],     "avg_stay_min": 60,  "mac": "aa:bb:cc:dd:ee:01"},
    "Prof":    {"usual_hours": (8, 16),  "usual_days": [0,1,2,3,4,5],   "avg_stay_min": 120, "mac": "aa:bb:cc:dd:ee:02"},
}

def random_time_for_person(person, anomaly=False):
    profile = PEOPLE[person]
    if anomaly:
        # Pick an unusual hour outside their normal range
        normal_start, normal_end = profile["usual_hours"]
        unusual_hours = list(range(0, normal_start - 2)) + list(range(normal_end + 2, 24))
        hour = random.choice(unusual_hours) if unusual_hours else 2
    else:
        start, end = profile["usual_hours"]
        hour = random.randint(start, end - 1)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return hour, minute, second

def generate_dataset(days=21, anomaly_rate=0.08):
    records = []
    base_date = datetime(2025, 2, 1)

    for day_offset in range(days):
        current_date = base_date + timedelta(days=day_offset)
        weekday = current_date.weekday()

        for person, profile in PEOPLE.items():
            # Skip if person doesn't come this day
            if weekday not in profile["usual_days"]:
                continue
            # Random chance they skip a day (10%)
            if random.random() < 0.10:
                continue

            # Normal entry
            is_anomaly = random.random() < anomaly_rate
            hour, minute, second = random_time_for_person(person, anomaly=is_anomaly)
            entry_time = current_date.replace(hour=hour, minute=minute, second=second)

            # Stay duration
            if is_anomaly:
                stay_min = random.choice([5, 200, 240])  # too short or too long
            else:
                stay_min = int(np.random.normal(profile["avg_stay_min"], 15))
                stay_min = max(15, stay_min)

            exit_time = entry_time + timedelta(minutes=stay_min)

            # Entry record
            records.append({
                "timestamp": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                "person": person,
                "mac": profile["mac"],
                "action": "ENTER",
                "hour": hour,
                "day_of_week": weekday,
                "stay_duration_min": stay_min,
                "rssi": random.randint(-65, -45),
                "people_inside": None,  # filled below
                "is_anomaly": int(is_anomaly)
            })

            # Exit record
            records.append({
                "timestamp": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "person": person,
                "mac": profile["mac"],
                "action": "EXIT",
                "hour": exit_time.hour,
                "day_of_week": weekday,
                "stay_duration_min": stay_min,
                "rssi": random.randint(-65, -45),
                "people_inside": None,
                "is_anomaly": int(is_anomaly)
            })

        # Add some unauthorized attempts (no name, unknown MAC)
        if random.random() < 0.15:
            unauth_hour = random.randint(0, 23)
            unauth_time = current_date.replace(
                hour=unauth_hour,
                minute=random.randint(0, 59),
                second=random.randint(0, 59)
            )
            records.append({
                "timestamp": unauth_time.strftime("%Y-%m-%d %H:%M:%S"),
                "person": "UNKNOWN",
                "mac": f"ff:{random.randint(10,99):02x}:{random.randint(10,99):02x}:{random.randint(10,99):02x}:{random.randint(10,99):02x}:{random.randint(10,99):02x}",
                "action": "UNAUTHORIZED",
                "hour": unauth_hour,
                "day_of_week": weekday,
                "stay_duration_min": 0,
                "rssi": random.randint(-80, -55),
                "people_inside": 0,
                "is_anomaly": 1
            })

    # Sort by timestamp
    records.sort(key=lambda x: x["timestamp"])

    # Fill people_inside count
    count = 0
    for r in records:
        if r["action"] == "ENTER":
            count += 1
        elif r["action"] == "EXIT":
            count = max(0, count - 1)
        r["people_inside"] = count

    df = pd.DataFrame(records)
    df.to_csv("access_log.csv", index=False)
    print(f"✅ Generated {len(df)} records over {days} days")
    print(f"   Normal entries: {len(df[df['is_anomaly']==0])}")
    print(f"   Anomalies:      {len(df[df['is_anomaly']==1])}")
    print(f"   Saved to: access_log.csv")
    return df

if __name__ == "__main__":
    df = generate_dataset(days=21, anomaly_rate=0.08)
    print("\nSample records:")
    print(df.head(10).to_string())