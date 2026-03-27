"""
Smart Access Monitor — Live Dashboard
======================================
Run:  python dashboard.py
Open: http://localhost:5000

DEMO MODE (no hardware needed):
  python dashboard.py --demo

LIVE MODE (ESP32 connected):
  python dashboard.py --port /dev/cu.usbserial-0001
"""

import argparse
import json
import os
import pickle
import random
import sqlite3
import threading
import time
from datetime import datetime

import numpy as np
from flask import Flask, Response, jsonify, render_template_string

# ── Try importing serial (only needed in live mode) ──────────────────────────
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

app = Flask(__name__)
DB_FILE = "access_monitor.db"
DEMO_MODE = False
SERIAL_PORT = None

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            person TEXT,
            mac TEXT,
            action TEXT,
            direction TEXT,
            people_inside INTEGER,
            anomaly_score INTEGER,
            behavioral_score INTEGER,
            alert_level TEXT,
            reasons TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_event(event):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO events
        (timestamp, person, mac, action, direction, people_inside,
         anomaly_score, behavioral_score, alert_level, reasons)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        event["timestamp"], event["person"], event["mac"],
        event["action"], event["direction"], event["people_inside"],
        event["anomaly_score"], event["behavioral_score"],
        event["alert_level"], json.dumps(event["reasons"])
    ))
    conn.commit()
    conn.close()

def get_recent_events(limit=50):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        SELECT * FROM events ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    cols = ["id","timestamp","person","mac","action","direction",
            "people_inside","anomaly_score","behavioral_score","alert_level","reasons"]
    return [dict(zip(cols, row)) for row in rows]

def get_stats():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT people_inside FROM events ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    people_inside = row[0] if row else 0

    c.execute("SELECT COUNT(*) FROM events WHERE action='UNAUTHORIZED' OR alert_level='HIGH'")
    alerts = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM events WHERE action='ENTER'")
    total_entries = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM events WHERE DATE(timestamp) = DATE('now')")
    today = c.fetchone()[0]

    conn.close()
    return {
        "people_inside": people_inside,
        "total_alerts": alerts,
        "total_entries": total_entries,
        "today_events": today
    }

# ─────────────────────────────────────────────────────────────────────────────
# ML SCORING
# ─────────────────────────────────────────────────────────────────────────────

if_bundle = None
profiles = None

def load_models():
    global if_bundle, profiles
    try:
        with open("isolation_forest.pkl", "rb") as f:
            if_bundle = pickle.load(f)
        with open("behavioral_profiles.pkl", "rb") as f:
            profiles = pickle.load(f)
        print("✅ ML models loaded")
    except FileNotFoundError:
        print("⚠️  ML models not found — run ml_models.py first. Running without ML scoring.")

def score_event(event):
    if if_bundle is None or profiles is None:
        return {"anomaly_score": 0, "behavioral_score": 0, "alert_level": "NORMAL", "reasons": []}

    result = {"anomaly_score": 0, "behavioral_score": 0, "alert_level": "NORMAL", "reasons": []}
    features = if_bundle["features"]

    try:
        X = np.array([[event.get(f, 0) for f in features]])
        X_scaled = if_bundle["scaler"].transform(X)
        score = if_bundle["model"].decision_function(X_scaled)[0]
        anomaly_score = min(100, max(0, int((-score + 0.3) * 100)))
        result["anomaly_score"] = anomaly_score
        if anomaly_score > 70:
            result["reasons"].append(f"Unusual pattern (score: {anomaly_score})")
    except Exception:
        pass

    person = event.get("person", "UNKNOWN")
    if person == "UNKNOWN":
        result["behavioral_score"] = 100
        result["reasons"].append("Unknown/unauthorized device")
    elif profiles and person in profiles:
        profile = profiles[person]
        try:
            X_bp = np.array([[event.get(f, 0) for f in profile["features"]]])
            X_bp_scaled = profile["scaler"].transform(X_bp)
            distances = [np.linalg.norm(X_bp_scaled[0] - c) for c in profile["model"].cluster_centers_]
            min_dist = min(distances)
            behavioral_score = min(100, int((min_dist / profile["threshold"]) * 50))
            result["behavioral_score"] = behavioral_score

            hour = event.get("hour", 12)
            ns, ne = profile["normal_hours"]
            if hour < ns - 2 or hour > ne + 2:
                result["reasons"].append(f"{person} unusual hour: {hour}:00 (normal: {ns}:00–{ne}:00)")

            dow = event.get("day_of_week", 0)
            days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            if dow not in profile["usual_days"]:
                result["reasons"].append(f"{person} doesn't usually come on {days[dow]}")

            if behavioral_score > 60:
                result["reasons"].append(f"Behavioral mismatch — possible stolen credential")
        except Exception:
            pass

    combined = max(result["anomaly_score"], result["behavioral_score"])
    if person == "UNKNOWN" or combined >= 80:
        result["alert_level"] = "HIGH"
    elif combined >= 50:
        result["alert_level"] = "MEDIUM"
    else:
        result["alert_level"] = "NORMAL"

    return result

# ─────────────────────────────────────────────────────────────────────────────
# DEMO MODE — generates fake events automatically
# ─────────────────────────────────────────────────────────────────────────────

DEMO_PEOPLE = [
    {"person": "Hardik",  "mac": "79:47:0e:74:97:a1"},
    {"person": "Ekaansh", "mac": "aa:bb:cc:dd:ee:01"},
    {"person": "Prof",    "mac": "aa:bb:cc:dd:ee:02"},
]
people_count = 0

def make_demo_event():
    global people_count
    now = datetime.now()
    hour = now.hour
    dow = now.weekday()

    # 15% chance of anomaly / unauthorized
    roll = random.random()

    if roll < 0.12:
        # Unauthorized
        event = {
            "person": "UNKNOWN",
            "mac": f"ff:{random.randint(10,99):02x}:{random.randint(10,99):02x}:xx:xx:xx",
            "action": "UNAUTHORIZED",
            "direction": "ENTERING",
            "hour": hour,
            "day_of_week": dow,
            "stay_duration_min": 0,
            "rssi": random.randint(-80, -60),
        }
    elif roll < 0.20:
        # Anomalous authorized (e.g., 2am entry)
        p = random.choice(DEMO_PEOPLE)
        event = {
            "person": p["person"],
            "mac": p["mac"],
            "action": "ENTER",
            "direction": "ENTERING",
            "hour": random.choice([1, 2, 3, 23]),
            "day_of_week": 6,  # Sunday
            "stay_duration_min": random.choice([5, 180, 240]),
            "rssi": random.randint(-65, -45),
        }
    elif roll < 0.45:
        # Normal exit
        if people_count > 0:
            p = random.choice(DEMO_PEOPLE)
            people_count = max(0, people_count - 1)
            event = {
                "person": p["person"],
                "mac": p["mac"],
                "action": "EXIT",
                "direction": "EXITING",
                "hour": hour,
                "day_of_week": dow,
                "stay_duration_min": random.randint(30, 120),
                "rssi": random.randint(-65, -45),
            }
        else:
            p = random.choice(DEMO_PEOPLE)
            people_count += 1
            event = {
                "person": p["person"],
                "mac": p["mac"],
                "action": "ENTER",
                "direction": "ENTERING",
                "hour": hour,
                "day_of_week": dow,
                "stay_duration_min": random.randint(45, 120),
                "rssi": random.randint(-65, -45),
            }
    else:
        # Normal entry
        p = random.choice(DEMO_PEOPLE)
        people_count += 1
        event = {
            "person": p["person"],
            "mac": p["mac"],
            "action": "ENTER",
            "direction": "ENTERING",
            "hour": hour,
            "day_of_week": dow,
            "stay_duration_min": random.randint(45, 120),
            "rssi": random.randint(-65, -45),
        }

    scores = score_event(event)
    event.update({
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "people_inside": people_count,
        "anomaly_score": scores["anomaly_score"],
        "behavioral_score": scores["behavioral_score"],
        "alert_level": scores["alert_level"],
        "reasons": scores["reasons"],
    })
    return event

def demo_loop():
    print("🎬 Demo mode running — generating events every 4 seconds")
    while True:
        event = make_demo_event()
        insert_event(event)
        level_icon = {"HIGH": "🚨", "MEDIUM": "⚠️", "NORMAL": "✅"}.get(event["alert_level"], "")
        print(f"{level_icon} [{event['timestamp']}] {event['person']} {event['action']} | "
              f"Alert: {event['alert_level']} | People: {event['people_inside']}")
        time.sleep(4)

# ─────────────────────────────────────────────────────────────────────────────
# LIVE MODE — reads from ESP32 serial
# ─────────────────────────────────────────────────────────────────────────────

def parse_serial_line(line, people_count_ref):
    """Parse ESP32 serial output into an event dict."""
    event = None
    now = datetime.now()
    hour = now.hour
    dow = now.weekday()

    if "AUTHORIZED" in line and "UNAUTHORIZED" not in line:
        direction = "ENTERING" if "ENTERING" in line else "EXITING"
        action = "ENTER" if direction == "ENTERING" else "EXIT"
        if direction == "ENTERING":
            people_count_ref[0] += 1
        else:
            people_count_ref[0] = max(0, people_count_ref[0] - 1)

        event = {
            "person": "Authorized User",
            "mac": "known",
            "action": action,
            "direction": direction,
            "hour": hour,
            "day_of_week": dow,
            "stay_duration_min": 60,
            "rssi": -55,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "people_inside": people_count_ref[0],
        }

    elif "UNAUTHORIZED" in line:
        event = {
            "person": "UNKNOWN",
            "mac": "unknown",
            "action": "UNAUTHORIZED",
            "direction": "ENTERING",
            "hour": hour,
            "day_of_week": dow,
            "stay_duration_min": 0,
            "rssi": -70,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "people_inside": people_count_ref[0],
        }

    if event:
        scores = score_event(event)
        event.update({
            "anomaly_score": scores["anomaly_score"],
            "behavioral_score": scores["behavioral_score"],
            "alert_level": scores["alert_level"],
            "reasons": scores["reasons"],
        })

    return event

def live_loop(port):
    if not SERIAL_AVAILABLE:
        print("❌ pyserial not installed. Run: pip install pyserial")
        return

    people_count_ref = [0]
    print(f"🔌 Connecting to ESP32 on {port}...")
    try:
        ser = serial.Serial(port, 115200, timeout=2)
        print(f"✅ Connected to {port}")
        while True:
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    print(f"ESP32: {line}")
                    event = parse_serial_line(line, people_count_ref)
                    if event:
                        insert_event(event)
            except Exception as e:
                print(f"Serial read error: {e}")
                time.sleep(1)
    except serial.SerialException as e:
        print(f"❌ Could not open port {port}: {e}")
        print("   Falling back to demo mode...")
        demo_loop()

# ─────────────────────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/events")
def api_events():
    events = get_recent_events(50)
    for e in events:
        if isinstance(e["reasons"], str):
            try:
                e["reasons"] = json.loads(e["reasons"])
            except Exception:
                e["reasons"] = []
    return jsonify(events)

@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())

@app.route("/api/stream")
def stream():
    """Server-Sent Events for real-time push to browser."""
    def generate():
        last_id = 0
        while True:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT * FROM events WHERE id > ? ORDER BY id ASC", (last_id,))
            rows = c.fetchall()
            conn.close()

            for row in rows:
                cols = ["id","timestamp","person","mac","action","direction",
                        "people_inside","anomaly_score","behavioral_score","alert_level","reasons"]
                event = dict(zip(cols, row))
                last_id = event["id"]
                try:
                    event["reasons"] = json.loads(event["reasons"]) if event["reasons"] else []
                except Exception:
                    event["reasons"] = []
                yield f"data: {json.dumps(event)}\n\n"

            time.sleep(1)

    return Response(generate(), mimetype="text/event-stream")

# ─────────────────────────────────────────────────────────────────────────────
# HTML DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart Access Monitor</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0d1117; color: #e2e8f0; font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }

  .header {
    background: #0d2137;
    border-bottom: 2px solid #0e7490;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .header h1 { font-size: 20px; font-weight: 700; color: #22d3ee; letter-spacing: 1px; }
  .header .subtitle { font-size: 12px; color: #64748b; margin-top: 2px; }
  .mode-badge {
    background: #1e3a5f;
    border: 1px solid #22d3ee;
    color: #22d3ee;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
  }
  .live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #10b981;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    padding: 20px 24px;
  }
  .stat-card {
    background: #1e293b;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 16px 20px;
  }
  .stat-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
  .stat-value { font-size: 36px; font-weight: 700; margin-top: 4px; }
  .stat-card.people .stat-value { color: #22d3ee; }
  .stat-card.alerts .stat-value { color: #ef4444; }
  .stat-card.entries .stat-value { color: #10b981; }
  .stat-card.today .stat-value { color: #f59e0b; }

  .main { display: grid; grid-template-columns: 1fr 340px; gap: 0; padding: 0 24px 24px; }

  .feed-section { }
  .feed-title {
    font-size: 13px; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 1px;
    padding: 12px 0 10px;
    border-bottom: 1px solid #1e3a5f;
    margin-bottom: 12px;
  }

  .event-card {
    background: #1e293b;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 8px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    animation: slideIn 0.3s ease;
    border-left: 3px solid #1e3a5f;
  }
  .event-card.HIGH { border-left-color: #ef4444; background: #1e1a1a; }
  .event-card.MEDIUM { border-left-color: #f59e0b; background: #1e1b14; }
  .event-card.NORMAL { border-left-color: #10b981; }
  @keyframes slideIn { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: translateY(0); } }

  .event-icon { font-size: 20px; line-height: 1; margin-top: 2px; }
  .event-body { flex: 1; }
  .event-main { display: flex; align-items: center; gap: 8px; }
  .event-person { font-weight: 700; font-size: 14px; }
  .event-person.UNKNOWN { color: #ef4444; }
  .event-action { font-size: 11px; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
  .event-action.ENTER { background: #064e3b; color: #34d399; }
  .event-action.EXIT { background: #1e293b; color: #94a3b8; border: 1px solid #334155; }
  .event-action.UNAUTHORIZED { background: #450a0a; color: #ef4444; }
  .event-direction { font-size: 11px; color: #64748b; }
  .event-time { font-size: 11px; color: #475569; margin-top: 3px; }
  .event-reasons { margin-top: 5px; }
  .event-reason { font-size: 11px; color: #f59e0b; margin-top: 2px; }
  .event-reason::before { content: "⚠ "; }

  .scores { display: flex; gap: 8px; margin-top: 6px; }
  .score-pill {
    font-size: 10px; padding: 2px 8px; border-radius: 10px;
    background: #0d2137; color: #94a3b8;
  }
  .score-pill.high { background: #450a0a; color: #ef4444; }
  .score-pill.medium { background: #451a03; color: #f59e0b; }

  .sidebar { padding-left: 20px; }
  .people-display {
    background: #1e293b;
    border: 1px solid #0e7490;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin-top: 41px;
    margin-bottom: 16px;
  }
  .people-display .big-count {
    font-size: 72px;
    font-weight: 900;
    color: #22d3ee;
    line-height: 1;
  }
  .people-display .people-label { color: #64748b; font-size: 12px; margin-top: 6px; text-transform: uppercase; letter-spacing: 1px; }

  .alert-box {
    background: #1e293b;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
  }
  .alert-box h3 { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
  .alert-item {
    background: #450a0a;
    border-left: 3px solid #ef4444;
    border-radius: 4px;
    padding: 8px 10px;
    margin-bottom: 6px;
    font-size: 12px;
    color: #fca5a5;
  }
  .alert-item.medium {
    background: #451a03;
    border-left-color: #f59e0b;
    color: #fcd34d;
  }
  .alert-item .alert-time { font-size: 10px; color: #6b7280; margin-top: 3px; }
  .no-alerts { color: #475569; font-size: 12px; }

  .empty-state { text-align: center; color: #475569; padding: 40px 20px; font-size: 14px; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>⚡ Smart Access Monitor</h1>
    <div class="subtitle">Dual Ultrasonic + BLE + ML Anomaly Detection</div>
  </div>
  <div class="mode-badge" id="modeBadge">
    <span class="live-dot"></span> LOADING...
  </div>
</div>

<div class="stats-grid">
  <div class="stat-card people">
    <div class="stat-label">People Inside</div>
    <div class="stat-value" id="statPeople">0</div>
  </div>
  <div class="stat-card alerts">
    <div class="stat-label">Total Alerts</div>
    <div class="stat-value" id="statAlerts">0</div>
  </div>
  <div class="stat-card entries">
    <div class="stat-label">Total Entries</div>
    <div class="stat-value" id="statEntries">0</div>
  </div>
  <div class="stat-card today">
    <div class="stat-label">Today's Events</div>
    <div class="stat-value" id="statToday">0</div>
  </div>
</div>

<div class="main">
  <div class="feed-section">
    <div class="feed-title">📡 Live Event Feed</div>
    <div id="eventFeed">
      <div class="empty-state">Waiting for events...</div>
    </div>
  </div>

  <div class="sidebar">
    <div class="people-display">
      <div class="big-count" id="bigCount">0</div>
      <div class="people-label">People Inside Right Now</div>
    </div>

    <div class="alert-box">
      <h3>🚨 Recent Alerts</h3>
      <div id="alertList"><div class="no-alerts">No alerts yet</div></div>
    </div>
  </div>
</div>

<script>
  const MAX_EVENTS = 30;
  let allEvents = [];
  let recentAlerts = [];

  function getIcon(event) {
    if (event.action === "UNAUTHORIZED") return "🚫";
    if (event.alert_level === "HIGH") return "🚨";
    if (event.alert_level === "MEDIUM") return "⚠️";
    if (event.action === "EXIT") return "👋";
    return "✅";
  }

  function formatTime(ts) {
    return ts.split(" ")[1] || ts;
  }

  function scoreClass(score) {
    if (score >= 70) return "high";
    if (score >= 40) return "medium";
    return "";
  }

  function renderEvent(event) {
    const reasons = Array.isArray(event.reasons) ? event.reasons : [];
    const reasonHtml = reasons.map(r => `<div class="event-reason">${r}</div>`).join("");
    const aScore = event.anomaly_score || 0;
    const bScore = event.behavioral_score || 0;

    return `
      <div class="event-card ${event.alert_level}">
        <div class="event-icon">${getIcon(event)}</div>
        <div class="event-body">
          <div class="event-main">
            <span class="event-person ${event.person === 'UNKNOWN' ? 'UNKNOWN' : ''}">${event.person}</span>
            <span class="event-action ${event.action}">${event.action}</span>
            <span class="event-direction">${event.direction || ""}</span>
          </div>
          <div class="event-time">${formatTime(event.timestamp)} &nbsp;·&nbsp; ${event.people_inside} inside &nbsp;·&nbsp; MAC: ${(event.mac || "").substring(0,17)}</div>
          <div class="scores">
            <span class="score-pill ${scoreClass(aScore)}">Anomaly: ${aScore}</span>
            <span class="score-pill ${scoreClass(bScore)}">Behavioral: ${bScore}</span>
          </div>
          ${reasonHtml ? `<div class="event-reasons">${reasonHtml}</div>` : ""}
        </div>
      </div>
    `;
  }

  function renderAlerts() {
    const alertDiv = document.getElementById("alertList");
    const alerts = recentAlerts.slice(0, 6);
    if (alerts.length === 0) {
      alertDiv.innerHTML = '<div class="no-alerts">No alerts yet</div>';
      return;
    }
    alertDiv.innerHTML = alerts.map(e => {
      const reasons = Array.isArray(e.reasons) ? e.reasons : [];
      const cls = e.alert_level === "HIGH" ? "" : "medium";
      return `
        <div class="alert-item ${cls}">
          <strong>${e.person}</strong> — ${e.action}
          ${reasons.length ? "<br>" + reasons[0] : ""}
          <div class="alert-time">${formatTime(e.timestamp)}</div>
        </div>
      `;
    }).join("");
  }

  function updateStats() {
    fetch("/api/stats")
      .then(r => r.json())
      .then(stats => {
        document.getElementById("statPeople").textContent = stats.people_inside;
        document.getElementById("statAlerts").textContent = stats.total_alerts;
        document.getElementById("statEntries").textContent = stats.total_entries;
        document.getElementById("statToday").textContent = stats.today_events;
        document.getElementById("bigCount").textContent = stats.people_inside;
      });
  }

  // Load initial events
  fetch("/api/events")
    .then(r => r.json())
    .then(events => {
      events.reverse().forEach(e => {
        allEvents.unshift(e);
        if (e.alert_level !== "NORMAL") recentAlerts.unshift(e);
      });
      const feed = document.getElementById("eventFeed");
      if (allEvents.length > 0) {
        feed.innerHTML = allEvents.slice(0, MAX_EVENTS).map(renderEvent).join("");
      }
      renderAlerts();
      updateStats();
    });

  // Server-sent events for real-time updates
  const evtSource = new EventSource("/api/stream");
  evtSource.onmessage = function(e) {
    const event = JSON.parse(e.data);
    allEvents.unshift(event);
    if (allEvents.length > MAX_EVENTS) allEvents.pop();

    if (event.alert_level !== "NORMAL") {
      recentAlerts.unshift(event);
      if (recentAlerts.length > 10) recentAlerts.pop();
    }

    const feed = document.getElementById("eventFeed");
    feed.innerHTML = allEvents.slice(0, MAX_EVENTS).map(renderEvent).join("");
    renderAlerts();
    updateStats();

    document.getElementById("modeBadge").innerHTML =
      '<span class="live-dot"></span> LIVE';
  };

  evtSource.onerror = function() {
    document.getElementById("modeBadge").innerHTML = "⚡ RECONNECTING...";
  };

  setInterval(updateStats, 5000);
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run in demo mode (no hardware)")
    parser.add_argument("--port", type=str, default=None, help="Serial port for ESP32 e.g. /dev/cu.usbserial-0001")
    args = parser.parse_args()

    init_db()
    load_models()

    if args.demo or args.port is None:
        print("\n🎬 Starting in DEMO MODE (no hardware needed)")
        print("   Run with --port /dev/cu.usbserial-XXXX for live ESP32 mode\n")
        t = threading.Thread(target=demo_loop, daemon=True)
    else:
        print(f"\n🔌 Starting in LIVE MODE on port {args.port}\n")
        t = threading.Thread(target=live_loop, args=(args.port,), daemon=True)

    t.start()
    print("🌐 Dashboard running at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)