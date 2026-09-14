/*
 * Smart Access Monitor - ESP32 firmware
 *
 * Two HC-SR04 ultrasonic sensors across the doorway work out direction
 * (A = outside, B = inside; A then B is an entry). A background task keeps
 * scanning BLE so the key is usually already known when someone reaches the door.
 *
 * Serial output, 115200 baud. Every machine-readable line is one JSON object:
 *   {"evt":"boot","fw":"2.0","keys":["Hardik-Key"]}
 *   {"evt":"access","dir":"ENTER","auth":true,"key":"Hardik-Key","mac":"aa:bb:..","rssi":-58,
 *    "reason":"key","alarm":false,"count":1,"ms":123456}
 *   {"evt":"status","count":1,"ms":123456,"keys":[{"key":"Hardik-Key","rssi":-60,"age":1200}]}
 * reason is "key", "no_key" or "tailgate". Anything else printed is debug text.
 *
 * Commands (send a line over serial): RESET, or COUNT <n> to correct occupancy.
 *
 * Wiring note: HC-SR04 ECHO is 5 V. Put a divider (e.g. 1k / 2k) on ECHO_A and
 * ECHO_B, or use 3.3 V sensors such as the HC-SR04P / RCWL-1601.
 */

#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>

// ---- Pins ----
#define TRIG_A 22   // outside sensor
#define ECHO_A 18
#define TRIG_B 19   // inside sensor
#define ECHO_B 21
#define BUZZER 26

// ---- Authorized keys: BLE advertised names, must match config.py ----
const char* AUTHORIZED_KEYS[] = {
  "Hardik-Key",
};
const int KEY_COUNT = sizeof(AUTHORIZED_KEYS) / sizeof(AUTHORIZED_KEYS[0]);

// ---- Tuning ----
const long          DETECT_CM          = 50;     // closer than this = someone in the beam
const int           DEBOUNCE_SAMPLES   = 2;      // consecutive readings to change beam state
const int           RSSI_MIN           = -75;    // ignore keys weaker than this
const unsigned long KEY_FRESH_MS       = 4000;   // key must have been heard this recently
const unsigned long KEY_GRACE_MS       = 1500;   // after a crossing, wait this long for a key
const unsigned long CROSS_TIMEOUT_MS   = 3000;   // first beam broken but second never was
const unsigned long STUCK_TIMEOUT_MS   = 15000;  // a beam blocked this long is ignored
const unsigned long TAILGATE_MS        = 5000;   // same key entering twice inside this window
const unsigned long CLEAR_HOLD_MS      = 600;    // both beams must stay clear this long before re-arming
const unsigned long STATUS_MS          = 5000;
const bool          REQUIRE_KEY_TO_EXIT = false;
const bool          DEBUG_DISTANCE      = false;

// ---- BLE sightings (written by the scan task, read by loop) ----
struct Sighting {
  int rssi;
  unsigned long seenAt;   // millis, 0 = never
  char mac[18];
  unsigned long enteredAt; // last authorized entry with this key, for tailgating
};
Sighting sightings[KEY_COUNT];
portMUX_TYPE keyMux = portMUX_INITIALIZER_UNLOCKED;
BLEScan* pBLEScan;

class KeyCallbacks : public BLEAdvertisedDeviceCallbacks {
  void onResult(BLEAdvertisedDevice device) {
    if (!device.haveName()) return;
    String name = device.getName().c_str();
    for (int i = 0; i < KEY_COUNT; i++) {
      if (name == AUTHORIZED_KEYS[i]) {
        String mac = device.getAddress().toString().c_str();
        int rssi = device.getRSSI();
        portENTER_CRITICAL(&keyMux);
        sightings[i].rssi = rssi;
        sightings[i].seenAt = millis();
        strncpy(sightings[i].mac, mac.c_str(), sizeof(sightings[i].mac) - 1);
        portEXIT_CRITICAL(&keyMux);
      }
    }
  }
};

void bleScanTask(void*) {
  for (;;) {
    pBLEScan->start(1, false);   // blocks ~1 s; results arrive through KeyCallbacks
    pBLEScan->clearResults();
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

// ---- Ultrasonic ----
long readCm(int trig, int echo) {
  digitalWrite(trig, LOW);
  delayMicroseconds(2);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);
  unsigned long us = pulseIn(echo, HIGH, 25000UL);  // ~4 m max
  if (us == 0) return -1;                           // no echo = nothing in range
  return us / 58;
}

struct Beam {
  bool active = false;
  int streak = 0;
  unsigned long activeSince = 0;

  void update(long cm, unsigned long now) {
    bool near = cm > 0 && cm < DETECT_CM;
    if (near == active) { streak = 0; return; }
    if (++streak >= DEBOUNCE_SAMPLES) {
      active = near;
      streak = 0;
      if (active) activeSince = now;
    }
  }
  // a beam blocked for ages (door left open, bag on the floor) shouldn't freeze counting
  bool blocked(unsigned long now) const { return active && now - activeSince < STUCK_TIMEOUT_MS; }
};
Beam beamA, beamB;

// ---- Crossing state machine ----
enum CrossState { IDLE, A_FIRST, B_FIRST, BOTH_FIRST, WAIT_CLEAR };
CrossState crossState = IDLE;
unsigned long stateSince = 0;

// ---- Pending decision (waiting briefly for a key after a crossing) ----
struct Pending {
  bool active = false;
  bool entering = false;
  unsigned long since = 0;
} pending;

int peopleCount = 0;
unsigned long lastStatus = 0;

// ---- Buzzer (non-blocking) ----
struct {
  int remaining = 0;
  unsigned onMs = 0, offMs = 0;
  bool on = false;
  unsigned long changedAt = 0;
} buzz;

void startBeeps(int n, unsigned onMs, unsigned offMs) {
  buzz.remaining = n;
  buzz.onMs = onMs;
  buzz.offMs = offMs;
  buzz.on = true;
  buzz.changedAt = millis();
  digitalWrite(BUZZER, HIGH);
}

void updateBuzzer(unsigned long now) {
  if (buzz.remaining <= 0) return;
  unsigned long elapsed = now - buzz.changedAt;
  if (buzz.on && elapsed >= buzz.onMs) {
    digitalWrite(BUZZER, LOW);
    buzz.on = false;
    buzz.changedAt = now;
    buzz.remaining--;
  } else if (!buzz.on && elapsed >= buzz.offMs) {
    digitalWrite(BUZZER, HIGH);
    buzz.on = true;
    buzz.changedAt = now;
  }
}

// ---- Output ----
void printStatus(unsigned long now) {
  Serial.printf("{\"evt\":\"status\",\"count\":%d,\"ms\":%lu,\"keys\":[", peopleCount, now);
  bool first = true;
  for (int i = 0; i < KEY_COUNT; i++) {
    portENTER_CRITICAL(&keyMux);
    Sighting s = sightings[i];
    unsigned long t = millis();
    portEXIT_CRITICAL(&keyMux);
    if (s.seenAt == 0 || t - s.seenAt > 10000) continue;
    Serial.printf("%s{\"key\":\"%s\",\"rssi\":%d,\"age\":%lu}", first ? "" : ",",
                  AUTHORIZED_KEYS[i], s.rssi, t - s.seenAt);
    first = false;
  }
  Serial.println("]}");
}

// Strongest fresh authorized key, or -1
int bestKey(int& rssiOut, char* macOut) {
  int best = -1;
  portENTER_CRITICAL(&keyMux);
  unsigned long t = millis();
  for (int i = 0; i < KEY_COUNT; i++) {
    const Sighting& s = sightings[i];
    if (s.seenAt == 0 || t - s.seenAt > KEY_FRESH_MS || s.rssi < RSSI_MIN) continue;
    if (best < 0 || s.rssi > sightings[best].rssi) best = i;
  }
  if (best >= 0) {
    rssiOut = sightings[best].rssi;
    memcpy(macOut, sightings[best].mac, 18);
  }
  portEXIT_CRITICAL(&keyMux);
  return best;
}

void decide(bool entering, unsigned long now) {
  int rssi = 0;
  char mac[18] = "";
  int key = bestKey(rssi, mac);

  const char* reason = key >= 0 ? "key" : "no_key";
  bool auth = key >= 0;
  if (auth && entering && sightings[key].enteredAt != 0 && now - sightings[key].enteredAt < TAILGATE_MS) {
    reason = "tailgate";   // one key, two entries back to back
    auth = false;
  }
  if (auth && entering) sightings[key].enteredAt = now;
  if (auth && !entering) sightings[key].enteredAt = 0;   // left again: next entry is a fresh one

  bool alarm = entering ? !auth : (REQUIRE_KEY_TO_EXIT && !auth);

  // Occupancy counts bodies through the door, keyed or not
  if (entering) peopleCount++;
  else if (peopleCount > 0) peopleCount--;

  Serial.printf("{\"evt\":\"access\",\"dir\":\"%s\",\"auth\":%s,\"key\":\"%s\",\"mac\":\"%s\",",
                entering ? "ENTER" : "EXIT", auth ? "true" : "false",
                key >= 0 ? AUTHORIZED_KEYS[key] : "", mac);
  if (key >= 0) Serial.printf("\"rssi\":%d,", rssi);
  else Serial.print("\"rssi\":null,");
  Serial.printf("\"reason\":\"%s\",\"alarm\":%s,\"count\":%d,\"ms\":%lu}\n",
                reason, alarm ? "true" : "false", peopleCount, now);

  if (alarm) startBeeps(3, 200, 200);
  else startBeeps(1, 60, 0);
}

void onCrossing(bool entering, unsigned long now) {
  if (pending.active) decide(pending.entering, now);   // don't lose the previous one
  pending.active = true;
  pending.entering = entering;
  pending.since = now;
}

void updatePending(unsigned long now) {
  if (!pending.active) return;
  int rssi;
  char mac[18];
  if (bestKey(rssi, mac) >= 0 || now - pending.since >= KEY_GRACE_MS) {
    pending.active = false;
    decide(pending.entering, now);
  }
}

void updateCrossing(unsigned long now) {
  bool a = beamA.blocked(now);
  bool b = beamB.blocked(now);
  unsigned long inState = now - stateSince;

  switch (crossState) {
    case IDLE:
      if (a && b)      { crossState = BOTH_FIRST; stateSince = now; }
      else if (a)      { crossState = A_FIRST;    stateSince = now; }
      else if (b)      { crossState = B_FIRST;    stateSince = now; }
      break;

    case A_FIRST:
      if (b)                                   { onCrossing(true, now);  crossState = WAIT_CLEAR; stateSince = now; }
      else if (inState > CROSS_TIMEOUT_MS)     { crossState = a ? WAIT_CLEAR : IDLE; stateSince = now; }
      break;

    case B_FIRST:
      if (a)                                   { onCrossing(false, now); crossState = WAIT_CLEAR; stateSince = now; }
      else if (inState > CROSS_TIMEOUT_MS)     { crossState = b ? WAIT_CLEAR : IDLE; stateSince = now; }
      break;

    case BOTH_FIRST:
      // both beams broke together: direction is whichever one is still blocked last
      if (!a && b)                             { onCrossing(true, now);  crossState = WAIT_CLEAR; stateSince = now; }
      else if (a && !b)                        { onCrossing(false, now); crossState = WAIT_CLEAR; stateSince = now; }
      else if (!a && !b)                       { crossState = IDLE; }
      break;

    case WAIT_CLEAR:
      // a body between the sensors flickers the beams; only re-arm after a steady clear
      if (a || b)                              { stateSince = now; }
      else if (inState >= CLEAR_HOLD_MS)       { crossState = IDLE; }
      break;
  }
}

void readCommands() {
  static String line;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      line.trim();
      if (line == "RESET") peopleCount = 0;
      else if (line.startsWith("COUNT ")) peopleCount = max(0L, line.substring(6).toInt());
      if (line.length()) printStatus(millis());
      line = "";
    } else if (line.length() < 32) {
      line += c;
    }
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(TRIG_A, OUTPUT);
  pinMode(ECHO_A, INPUT);
  pinMode(TRIG_B, OUTPUT);
  pinMode(ECHO_B, INPUT);
  pinMode(BUZZER, OUTPUT);

  memset(sightings, 0, sizeof(sightings));

  BLEDevice::init("");
  pBLEScan = BLEDevice::getScan();
  pBLEScan->setAdvertisedDeviceCallbacks(new KeyCallbacks(), false);
  pBLEScan->setActiveScan(true);   // names often live in the scan response
  pBLEScan->setInterval(100);
  pBLEScan->setWindow(99);
  xTaskCreatePinnedToCore(bleScanTask, "ble_scan", 6144, nullptr, 1, nullptr, 0);

  Serial.print("{\"evt\":\"boot\",\"fw\":\"2.0\",\"keys\":[");
  for (int i = 0; i < KEY_COUNT; i++) Serial.printf("%s\"%s\"", i ? "," : "", AUTHORIZED_KEYS[i]);
  Serial.println("]}");

  startBeeps(1, 200, 0);
}

void loop() {
  unsigned long now = millis();

  long distA = readCm(TRIG_A, ECHO_A);
  delay(5);                         // let the first ping die out before the second
  long distB = readCm(TRIG_B, ECHO_B);
  now = millis();

  beamA.update(distA, now);
  beamB.update(distB, now);
  if (DEBUG_DISTANCE) Serial.printf("A %ld cm  B %ld cm  state %d\n", distA, distB, crossState);

  updateCrossing(now);
  updatePending(now);
  updateBuzzer(now);
  readCommands();

  if (now - lastStatus >= STATUS_MS) {
    lastStatus = now;
    printStatus(now);
  }

  delay(15);
}
