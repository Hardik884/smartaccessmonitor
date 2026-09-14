"""
Shared configuration for the dashboard, the data generator and the ML models.

AUTHORIZED_KEYS maps the BLE advertised name of each key (exactly as it is
listed in `authorizedKeys[]` in the ESP32 firmware) to the person who carries it.
The dashboard shows the person's name; the ML models build one behavioral
profile per person.

The "profile" block is only used by generate.py to create synthetic training
data. Real behavior is learned from logged events once you retrain with --db.
"""

AUTHORIZED_KEYS = {
    "Hardik-Key": {
        "person": "Hardik",
        "profile": {
            "arrival_hour": 9.5, "arrival_std": 0.9,
            "stay_min": 95, "stay_std": 20,
            # attendance probability Mon..Sun
            "attendance": [0.9, 0.9, 0.85, 0.9, 0.8, 0.1, 0.0],
            "second_visit": 0.25,
        },
    },
    "Ekaansh-Key": {
        "person": "Ekaansh",
        "profile": {
            "arrival_hour": 11.0, "arrival_std": 1.2,
            "stay_min": 60, "stay_std": 15,
            "attendance": [0.85, 0.8, 0.85, 0.8, 0.7, 0.0, 0.0],
            "second_visit": 0.35,
        },
    },
    "Prof-Key": {
        "person": "Prof",
        "profile": {
            "arrival_hour": 8.5, "arrival_std": 0.6,
            "stay_min": 130, "stay_std": 25,
            "attendance": [0.95, 0.95, 0.95, 0.95, 0.9, 0.5, 0.0],
            "second_visit": 0.15,
        },
    },
}

UNKNOWN_PERSON = "Unidentified"

# File locations
LIVE_DB = "access_monitor.db"
DEMO_DB = "access_monitor_demo.db"
TRAINING_CSV = "access_log.csv"
MODEL_FILE = "access_models.pkl"

# Alert thresholds on the 0-100 risk score
RISK_HIGH = 75
RISK_MEDIUM = 50

# Same key crossing the door again within this window is treated as possible
# tailgating / key sharing by the dashboard (the firmware has its own check too).
TAILGATE_WINDOW_S = 5


def person_for_key(ble_name):
    """Return the display name for a BLE key, or the raw name if it isn't configured."""
    if not ble_name:
        return UNKNOWN_PERSON
    entry = AUTHORIZED_KEYS.get(ble_name)
    return entry["person"] if entry else ble_name
