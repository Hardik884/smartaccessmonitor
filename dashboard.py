"""
Smart Access Monitor - dashboard
================================

    python dashboard.py                  live: finds the ESP32 serial port by itself
    python dashboard.py --port COM5      live on a specific port (/dev/cu.usbserial-0001 on macOS)
    python dashboard.py --demo           simulated events, kept in a separate database

Then open http://127.0.0.1:5001

Live mode never invents events. If the board isn't found or the cable is pulled,
the dashboard says so and keeps retrying. Demo mode replays simulated working days
(from the profiles in config.py) on a fast clock into access_monitor_demo.db, which
is wiped at the start of each demo run, so it can't mix with real data.

The reader understands both the JSON lines from firmware/smart_access_monitor
and the plain-text output of the original sketch.
"""

import argparse
import json
import os
import random
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta

from flask import Flask, Response, jsonify, request

from config import (AUTHORIZED_KEYS, DEMO_DB, LIVE_DB, TAILGATE_WINDOW_S, UNKNOWN_PERSON,
                    RISK_MEDIUM, person_for_key)
from ml_models import AccessScorer, MIN_LIVE_DAYS, retrain

RETRAIN_EVERY = 10   # authorized live events between automatic retrains

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None

app = Flask(__name__, static_folder="static", static_url_path="/static")

# USB-serial bridges found on ESP32 boards: CP210x, CH340/CH9102, FTDI, native USB (S2/S3/C3)
ESP32_USB_VIDS = {0x10C4, 0x1A86, 0x0403, 0x303A}
STALE_INSIDE_HOURS = 16


# ─────────────────────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────────────────────

EVENT_COLUMNS = ["id", "timestamp", "source", "person", "ble_name", "mac", "action", "authorized",
                 "reason", "rssi", "stay_min", "people_inside", "anomaly_score",
                 "behavioral_score", "risk", "alert_level", "reasons"]


class EventStore:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self):
        with self.lock:
            cols = [r["name"] for r in self.conn.execute("PRAGMA table_info(events)")]
            if cols and "source" not in cols:
                # schema from the first version: keep the rows, but out of the way
                legacy = f"events_legacy_{datetime.now():%Y%m%d%H%M%S}"
                self.conn.execute(f"ALTER TABLE events RENAME TO {legacy}")
                print(f"Old events table moved to {legacy}")
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    person TEXT NOT NULL,
                    ble_name TEXT NOT NULL DEFAULT '',
                    mac TEXT NOT NULL DEFAULT '',
                    action TEXT NOT NULL,
                    authorized INTEGER NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    rssi INTEGER,
                    stay_min REAL,
                    people_inside INTEGER,
                    anomaly_score INTEGER,
                    behavioral_score INTEGER,
                    risk INTEGER,
                    alert_level TEXT,
                    reasons TEXT
                )""")
            self.conn.execute("CREATE INDEX IF NOT EXISTS events_ts ON events(timestamp)")
            self.conn.commit()

    def insert(self, event):
        cols = [c for c in EVENT_COLUMNS if c != "id"]
        row = {**event, "reasons": json.dumps(event["reasons"])}
        with self.lock:
            cur = self.conn.execute(
                f"INSERT INTO events ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [row[c] for c in cols])
            self.conn.commit()
            return cur.lastrowid

    def query(self, sql, params=()):
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    @staticmethod
    def decode(row):
        row["authorized"] = bool(row["authorized"])
        try:
            row["reasons"] = json.loads(row["reasons"] or "[]")
        except ValueError:
            row["reasons"] = []
        return row

    def recent(self, limit=100, alerts_only=False):
        where = "WHERE alert_level != 'NORMAL'" if alerts_only else ""
        rows = self.query(f"SELECT * FROM events {where} ORDER BY id DESC LIMIT ?", (limit,))
        return [self.decode(r) for r in rows]

    def since(self, after, limit=200):
        rows = self.query("SELECT * FROM events WHERE id > ? ORDER BY id ASC LIMIT ?", (after, limit))
        return [self.decode(r) for r in rows]

    def day_summary(self, now):
        day = (now.strftime("%Y-%m-%d"), (now + timedelta(days=1)).strftime("%Y-%m-%d"))
        s = self.query("""
            SELECT
              SUM(action = 'ENTER')                    AS entries,
              SUM(action = 'EXIT')                     AS exits,
              SUM(alert_level != 'NORMAL')             AS alerts,
              SUM(alert_level = 'HIGH')                AS high_alerts,
              SUM(action = 'ENTER' AND authorized = 0) AS unidentified_entries
            FROM events WHERE timestamp >= ? AND timestamp < ?""", day)[0]
        hourly = [0] * 24
        for r in self.query("""
                SELECT CAST(substr(timestamp, 12, 2) AS INTEGER) AS h, COUNT(*) AS n
                FROM events WHERE timestamp >= ? AND timestamp < ? AND action = 'ENTER'
                GROUP BY h""", day):
            hourly[r["h"]] = r["n"]
        last = self.query("SELECT * FROM events ORDER BY id DESC LIMIT 1")
        return {
            **{k: int(v or 0) for k, v in s.items()},
            "hourly_entries": hourly,
            "last_event": self.decode(last[0]) if last else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Event processing (shared by live and demo)
# ─────────────────────────────────────────────────────────────────────────────

class Monitor:
    def __init__(self, store, scorer, source, clock=datetime.now):
        self.store = store
        self.scorer = scorer
        self.source = source
        self.clock = clock          # demo mode swaps in a simulated clock
        self.lock = threading.RLock()
        self.serial = None
        self.link = {"mode": source, "connected": source == "demo", "port": None,
                     "firmware": None, "message": None, "last_message": None}
        self.nearby = []            # [{key, person, rssi, age_ms}] from the last status line
        self.nearby_at = 0.0
        self.inside = {}            # person -> {"since": datetime, "ble_name": str}
        self.last_key_entry = {}    # ble_name -> datetime, for tailgating on legacy firmware
        self.count = 0
        self.verbose = False
        self.since_retrain = 0
        self.retraining = False
        self._restore()

    def retrain_async(self):
        """Refit the models with the real events logged so far, without blocking the reader."""
        if self.source != "live" or self.retraining:
            return
        self.retraining = True

        def run():
            try:
                bundle = retrain(db_path=self.store.path, verbose=False)
                if bundle:
                    self.scorer.bundle = bundle
                    real = [p for p, prof in bundle["profiles"].items() if prof.get("from_live_data")]
                    print(f"Models retrained. Profiles from real visits: {', '.join(real) or 'none yet'}")
            except Exception as exc:
                print(f"Retraining failed: {exc}")
            finally:
                self.retraining = False

        threading.Thread(target=run, daemon=True).start()

    def _restore(self):
        """Rebuild who is inside from the log, so a dashboard restart doesn't forget."""
        cutoff = (datetime.now() - timedelta(hours=STALE_INSIDE_HOURS)).strftime("%Y-%m-%d %H:%M:%S")
        rows = self.store.query("""
            SELECT e.person, e.ble_name, e.action, e.timestamp FROM events e
            JOIN (SELECT person, MAX(id) AS id FROM events WHERE authorized = 1 GROUP BY person) last
              ON e.id = last.id
            WHERE e.timestamp >= ?""", (cutoff,))
        for r in rows:
            if r["action"] == "ENTER":
                self.inside[r["person"]] = {"since": datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S"),
                                            "ble_name": r["ble_name"]}
        last = self.store.query("SELECT people_inside FROM events ORDER BY id DESC LIMIT 1")
        self.count = last[0]["people_inside"] if last else len(self.inside)

    # ── link status ──
    def set_link(self, **fields):
        with self.lock:
            self.link.update(fields)

    def attach_serial(self, ser):
        with self.lock:
            self.serial = ser

    def send_command(self, text):
        with self.lock:
            if self.serial is not None:
                try:
                    self.serial.write((text + "\n").encode())
                except Exception as exc:
                    print(f"Could not send '{text}' to the board: {exc}")

    # ── input ──
    def handle_line(self, line, legacy):
        with self.lock:
            self.link["last_message"] = time.time()
        if line.startswith("{"):
            try:
                msg = json.loads(line)
            except ValueError:
                return
            self.handle_message(msg)
        else:
            msg = legacy.feed(line)
            if msg:
                self.set_link(firmware="original sketch (text output)")
                self.handle_message(msg)

    def handle_message(self, msg):
        kind = msg.get("evt")
        if kind == "access":
            self.handle_access(msg)
        elif kind == "status":
            with self.lock:
                if msg.get("count") is not None:
                    self.count = int(msg["count"])
                if "keys" in msg:
                    self.nearby = [{"key": k["key"], "person": person_for_key(k["key"]),
                                    "rssi": k.get("rssi"), "age_ms": k.get("age", 0)} for k in msg["keys"]]
                    self.nearby_at = time.time()
        elif kind == "boot":
            self.set_link(firmware=f"v{msg.get('fw', '?')}")
            missing = [k for k in msg.get("keys", []) if k not in AUTHORIZED_KEYS]
            if missing:
                print(f"Keys on the board but not in config.py: {', '.join(missing)}")
            with self.lock:
                count = self.count
            if count > 0:
                # the board restarted (it does when the port opens); give it back the occupancy
                self.send_command(f"COUNT {count}")

    def handle_access(self, msg):
        now = self.clock()
        entering = msg.get("dir") == "ENTER"
        action = "ENTER" if entering else "EXIT"
        ble_name = msg.get("key") or ""
        authorized = bool(msg.get("auth"))
        reason = msg.get("reason") or ("key" if authorized else "no_key")
        rssi = msg.get("rssi")
        rules = []

        with self.lock:
            owner = person_for_key(ble_name) if ble_name else UNKNOWN_PERSON

            # tailgating: the firmware checks this too, but the original sketch doesn't
            last = self.last_key_entry.get(ble_name)
            gap = (now - last).total_seconds() if last else None
            if authorized and entering and gap is not None and gap < TAILGATE_WINDOW_S:
                authorized, reason = False, "tailgate"
            if reason == "tailgate":
                when = f"{gap:.0f} s after" if gap is not None and gap < 60 else "right after"
                rules.append((100, f"Second entry on {owner}'s key {when} the first (tailgating or shared key)"))

            person = owner if authorized else UNKNOWN_PERSON
            stay = None
            if authorized and entering:
                self.last_key_entry[ble_name] = now
                if person in self.inside:
                    since = self.inside[person]["since"]
                    rules.append((RISK_MEDIUM + 5, f"{person} entered again without an exit since {since:%H:%M}"))
                self.inside[person] = {"since": now, "ble_name": ble_name}
            elif authorized and not entering:
                self.last_key_entry.pop(ble_name, None)
                visit = self.inside.pop(person, None)
                if visit:
                    stay = round((now - visit["since"]).total_seconds() / 60, 1)
                else:
                    rules.append((0, f"No entry recorded for {person} before this exit"))

            if msg.get("count") is not None:
                self.count = int(msg["count"])
            elif authorized:   # original sketch: only keyed crossings change the count
                self.count = self.count + 1 if entering else max(0, self.count - 1)

            scores = self.scorer.score({
                "timestamp": now, "person": owner if reason == "tailgate" else person, "action": action,
                "authorized": authorized, "reason": reason, "rssi": rssi, "stay_duration_min": stay,
            }, rule_hits=rules)

            event = {
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), "source": self.source,
                "person": person, "ble_name": ble_name, "mac": msg.get("mac") or "",
                "action": action, "authorized": int(authorized), "reason": reason, "rssi": rssi,
                "stay_min": stay, "people_inside": self.count, **scores,
            }
            event["id"] = self.store.insert(event)
            if authorized:
                self.since_retrain += 1
                if self.since_retrain >= RETRAIN_EVERY:
                    self.since_retrain = 0
                    self.retrain_async()

        key_txt = f" ({ble_name}, {rssi} dBm)" if ble_name else ""
        print(f"{now:%H:%M:%S}  {action:<5}  {person}{key_txt}  risk {scores['risk']} {scores['alert_level']}"
              + "".join(f"\n            - {r}" for r in scores["reasons"]))
        return event

    # ── output ──
    def snapshot(self):
        with self.lock:
            now = self.clock()
            inside = [{"person": p, "ble_name": v["ble_name"], "since": v["since"].strftime("%Y-%m-%d %H:%M:%S"),
                       "minutes": int((now - v["since"]).total_seconds() // 60)}
                      for p, v in sorted(self.inside.items(), key=lambda kv: kv[1]["since"])]
            link = dict(self.link)
            if link["last_message"]:
                link["seconds_since_message"] = round(time.time() - link["last_message"], 1)
            nearby = self.nearby if time.time() - self.nearby_at < 15 else []
            return {
                "now": now.strftime("%Y-%m-%d %H:%M:%S"),
                "link": link,
                "occupancy": {"total": max(self.count, len(inside)), "inside": inside,
                              "unidentified": max(0, self.count - len(inside))},
                "keys_nearby": nearby,
                "models": self._model_info(),
                "today": self.store.day_summary(now),
                "authorized_keys": [{"key": k, "person": v["person"]} for k, v in AUTHORIZED_KEYS.items()],
            }

    def _model_info(self):
        if not self.scorer.has_models:
            return {"loaded": False}
        b = self.scorer.bundle
        return {"loaded": True, "trained_at": b.get("trained_at"), "profiles": sorted(b["profiles"]),
                "sample_profiles": sorted(p for p, prof in b["profiles"].items() if not prof.get("from_live_data")),
                "min_live_days": MIN_LIVE_DAYS}


# ─────────────────────────────────────────────────────────────────────────────
# Original sketch's text output
# ─────────────────────────────────────────────────────────────────────────────

class LegacyParser:
    """
    Turns the original sketch's prints into access messages. That sketch never says
    which key matched, so the key is the strongest configured name printed in the
    "Device: <name> | RSSI: <n>" lines of the same scan.
    """
    DEVICE = re.compile(r"Device:\s*(.*?)\s*\|\s*RSSI:\s*(-?\d+)")
    COUNT = re.compile(r"People inside:\s*(\d+)")

    def __init__(self, rssi_min=-70):
        self.rssi_min = rssi_min
        self.direction = None
        self.devices = {}

    def feed(self, line):
        if "Direction:" in line:
            self.direction = "ENTER" if "ENTERING" in line else "EXIT" if "EXITING" in line else None
            self.devices = {}
            return None

        m = self.DEVICE.search(line)
        if m:
            name, rssi = m.group(1), int(m.group(2))
            self.devices[name] = max(rssi, self.devices.get(name, -999))
            return None

        if line.startswith("Sensor A:"):
            m = self.COUNT.search(line)
            return {"evt": "status", "count": int(m.group(1))} if m else None

        if "UNAUTHORIZED" in line:
            msg = {"evt": "access", "dir": self.direction or "ENTER", "auth": False, "key": "",
                   "rssi": None, "reason": "no_key", "count": None}
            self.direction, self.devices = None, {}
            return msg

        if "AUTHORIZED" in line:
            direction = "ENTER" if "Welcome" in line else "EXIT" if "Goodbye" in line else self.direction
            in_range = {n: r for n, r in self.devices.items() if r > self.rssi_min}
            known = {n: r for n, r in in_range.items() if n in AUTHORIZED_KEYS}
            pool = known or in_range
            key = max(pool, key=pool.get) if pool else ""
            m = self.COUNT.search(line)
            msg = {"evt": "access", "dir": direction or "ENTER", "auth": True, "key": key,
                   "rssi": pool.get(key), "reason": "key", "count": int(m.group(1)) if m else None}
            self.direction, self.devices = None, {}
            return msg
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Serial reader
# ─────────────────────────────────────────────────────────────────────────────

def find_esp32_port():
    ports = list(list_ports.comports())
    for p in ports:
        if p.vid in ESP32_USB_VIDS:
            return p.device, ports
    return None, ports


def serial_worker(monitor, port_arg):
    if serial is None:
        monitor.set_link(message="pyserial is not installed: pip install pyserial")
        return

    announced = None
    while True:
        port, ports = (port_arg, None) if port_arg else find_esp32_port()
        if not port:
            names = ", ".join(p.device for p in ports) or "none"
            msg = f"No ESP32 found (serial ports: {names}). Plug it in or pass --port."
            if msg != announced:
                print(msg)
                announced = msg
            monitor.set_link(connected=False, port=None, message=msg)
            time.sleep(3)
            continue

        ser = serial.Serial()
        ser.port, ser.baudrate, ser.timeout = port, 115200, 1
        ser.dtr = ser.rts = False    # avoid resetting the board when the port opens
        try:
            ser.open()
        except (serial.SerialException, OSError) as exc:
            if "denied" in str(exc).lower() or "busy" in str(exc).lower():
                msg = (f"{port} is in use by another program. Close the Arduino Serial Monitor "
                       f"(only one program can read the port); the dashboard will connect on its own.")
            else:
                msg = f"Could not open {port}: {exc}"
            if msg != announced:
                print(msg)
                announced = msg
            monitor.set_link(connected=False, port=port, message=msg)
            time.sleep(3)
            continue

        print(f"Connected to {port}")
        announced = None
        monitor.attach_serial(ser)
        monitor.set_link(connected=True, port=port, message=None)
        legacy = LegacyParser()
        try:
            while True:
                raw = ser.readline()
                if raw:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line:
                        if monitor.verbose:
                            print(f"  board> {line}")
                        monitor.handle_line(line, legacy)
        except (serial.SerialException, OSError) as exc:
            print(f"Lost connection to {port}: {exc}")
            monitor.set_link(connected=False, message=f"Lost connection to {port}. Retrying.")
        finally:
            monitor.attach_serial(None)
            try:
                ser.close()
            except Exception:
                pass
        time.sleep(2)


# ─────────────────────────────────────────────────────────────────────────────
# Demo: simulated working days, compressed, pushed through the same pipeline
# ─────────────────────────────────────────────────────────────────────────────

def plan_demo_day(day, rng):
    """One day of firmware-style access messages built from the profiles in config.py."""
    def keyed(direction, key, rssi=None):
        return {"dir": direction, "auth": True, "key": key, "reason": "key",
                "rssi": rssi if rssi is not None else rng.randint(-64, -48)}

    def no_key(direction):
        return {"dir": direction, "auth": False, "key": "", "rssi": None, "reason": "no_key"}

    plan = []
    for key, cfg in AUTHORIZED_KEYS.items():
        p = cfg["profile"]
        if rng.random() >= p["attendance"][day.weekday()]:
            continue
        t = day + timedelta(hours=rng.gauss(p["arrival_hour"], p["arrival_std"]))
        for visit in range(2):
            weak = rng.random() < 0.05
            plan.append((t, keyed("ENTER", key, rng.randint(-75, -71) if weak else None)))
            if rng.random() < 0.06:   # someone slips in behind them
                plan.append((t + timedelta(seconds=rng.randint(2, 5)),
                             {**keyed("ENTER", key), "auth": False, "reason": "tailgate"}))
                plan.append((t + timedelta(minutes=rng.uniform(5, 40)), no_key("EXIT")))
            t += timedelta(minutes=max(10, rng.gauss(p["stay_min"], p["stay_std"])))
            plan.append((t, keyed("EXIT", key)))
            if visit == 1 or rng.random() >= p["second_visit"]:
                break
            t += timedelta(minutes=rng.uniform(30, 120))

    if rng.random() < 0.4:            # a visitor without a key
        t = day + timedelta(hours=rng.uniform(9, 18))
        plan += [(t, no_key("ENTER")), (t + timedelta(minutes=rng.uniform(2, 25)), no_key("EXIT"))]
    if rng.random() < 0.2:            # a key holder at an odd hour
        key = rng.choice(list(AUTHORIZED_KEYS))
        t = day + timedelta(hours=rng.uniform(21, 23))
        plan += [(t, keyed("ENTER", key)), (t + timedelta(minutes=rng.uniform(15, 50)), keyed("EXIT", key))]
    return sorted(plan, key=lambda item: item[0])


def demo_worker(monitor, interval, sim):
    rng = random.Random()
    day = sim["now"].replace(hour=0, minute=0, second=0, microsecond=0)
    count = 0
    print(f"Demo mode: simulated days starting {sim['now']:%d %b %H:%M}, one event every {interval:g} s, "
          f"stored in {DEMO_DB}")
    while True:
        for when, msg in plan_demo_day(day, rng):
            if when < sim["now"]:
                continue
            time.sleep(interval)
            sim["now"] = when
            count = count + 1 if msg["dir"] == "ENTER" else max(0, count - 1)
            monitor.handle_message({"evt": "access", "mac": "", "count": count, **msg})
            with monitor.lock:
                monitor.link["last_message"] = time.time()
            heard = ({msg["key"]} if msg["key"] else set()) | {k for k in AUTHORIZED_KEYS if rng.random() < 0.2}
            monitor.handle_message({"evt": "status", "count": count, "keys": [
                {"key": k, "rssi": msg["rssi"] if k == msg["key"] else rng.randint(-74, -60),
                 "age": rng.randint(100, 3000)} for k in sorted(heard)]})
        day += timedelta(days=1)
        sim["now"] = day + timedelta(hours=7)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────────────────────────────────────

monitor = None


@app.route("/")
def index():
    return app.send_static_file("dashboard.html")


@app.route("/api/state")
def api_state():
    return jsonify(monitor.snapshot())


@app.route("/api/events")
def api_events():
    limit = min(int(request.args.get("limit", 100)), 500)
    alerts = request.args.get("alerts") == "1"
    return jsonify(monitor.store.recent(limit=limit, alerts_only=alerts))


@app.route("/api/stream")
def api_stream():
    after = int(request.headers.get("Last-Event-ID") or request.args.get("after") or 0)

    def generate(last_id):
        yield "retry: 3000\n\n"
        while True:
            for event in monitor.store.since(last_id):
                last_id = event["id"]
                yield f"id: {last_id}\ndata: {json.dumps(event)}\n\n"
            time.sleep(1)

    return Response(generate(after), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Smart Access Monitor dashboard")
    ap.add_argument("--demo", action="store_true", help="simulated events (separate database)")
    ap.add_argument("--port", help="serial port, e.g. COM5 or /dev/cu.usbserial-0001 (auto-detected if omitted)")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open the dashboard to your network")
    ap.add_argument("--http-port", type=int, default=5001)
    ap.add_argument("--interval", type=float, default=4.0, help="demo: seconds between events")
    ap.add_argument("--verbose", action="store_true", help="print every line from the board")
    args = ap.parse_args()

    source = "demo" if args.demo else "live"
    scorer = AccessScorer.load()
    print("Models loaded" if scorer.has_models else "No trained models found (run ml_models.py). Using rule checks only.")

    if args.demo:
        if os.path.exists(DEMO_DB):
            os.remove(DEMO_DB)     # each demo run starts clean; live data is never touched
        sim = {"now": datetime.now().replace(hour=7, minute=0, second=0, microsecond=0)}
        monitor = Monitor(EventStore(DEMO_DB), scorer, source, clock=lambda: sim["now"])
        worker = threading.Thread(target=demo_worker, args=(monitor, args.interval, sim), daemon=True)
    else:
        monitor = Monitor(EventStore(LIVE_DB), scorer, source)
        worker = threading.Thread(target=serial_worker, args=(monitor, args.port), daemon=True)
        monitor.retrain_async()   # pick up real events logged since the last training
    # the demo replays the sample schedules, so those profiles are the ground truth there
    scorer.trust_sample_data = args.demo
    monitor.verbose = args.verbose
    worker.start()

    print(f"Dashboard: http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.http_port}")
    app.run(host=args.host, port=args.http_port, debug=False, threaded=True)
