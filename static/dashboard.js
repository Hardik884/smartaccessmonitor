(() => {
  const $ = (id) => document.getElementById(id);
  const ENTITIES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  // BLE names come from the air; never put them in the page unescaped
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ENTITIES[c]);

  const MAX_ROWS = 150;
  const LEVEL_LABEL = { HIGH: "High", MEDIUM: "Medium" };
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  let events = [];            // newest first
  const seen = new Set();
  const fresh = new Set();
  let filter = "all";
  let lastId = 0;

  // ── formatting ──
  const pad = (n) => String(n).padStart(2, "0");
  const parseTs = (ts) => new Date(ts.replace(" ", "T"));
  const hms = (d) => `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  const hm = (d) => `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  let serverNow = new Date();   // demo mode runs on a simulated clock, so trust the server's time
  const isToday = (d) => d.toDateString() === serverNow.toDateString();
  const day = (d) => `${d.getDate()} ${MONTHS[d.getMonth()]}`;

  function duration(min) {
    min = Math.max(0, Math.round(min));
    if (min < 60) return `${min} min`;
    const h = Math.floor(min / 60), m = min % 60;
    return m ? `${h} h ${m} min` : `${h} h`;
  }

  function ago(seconds) {
    if (seconds < 5) return "just now";
    if (seconds < 60) return `${Math.round(seconds)} s ago`;
    if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
    return `${Math.round(seconds / 3600)} h ago`;
  }

  function signalBars(rssi) {
    const level = rssi == null ? 0 : rssi >= -55 ? 4 : rssi >= -62 ? 3 : rssi >= -70 ? 2 : 1;
    const bars = [4, 7, 10, 12].map((h, i) => `<i class="${i < level ? "on" : ""}" style="height:${h}px"></i>`).join("");
    return `<span class="signal" aria-hidden="true">${bars}</span>`;
  }

  // ── event log ──
  function rowHtml(e) {
    const d = parseTs(e.timestamp);
    const known = e.authorized;
    let person = `<span class="person${known ? "" : " unknown"}">${esc(e.person)}</span>`;
    if (e.reason === "tailgate") person += `<span class="tag">Tailgate</span>`;
    else if (!known && e.action === "ENTER") person += `<span class="tag">No key</span>`;

    const door = (e.action === "ENTER" ? "In" : "Out") +
      (e.stay_min != null ? `<span class="sub">after ${duration(e.stay_min)}</span>` : "");

    const key = e.ble_name
      ? `${esc(e.ble_name)}<span class="sub">${e.rssi != null ? `${e.rssi} dBm` : "signal unknown"}</span>`
      : `<span class="sub">none</span>`;

    const risk = e.risk ?? 0;
    const notes = (e.reasons || []).map((r) => `<div>${esc(r)}</div>`).join("");
    const time = `<span class="time">${hms(d)}${isToday(d) ? "" : `<span class="date">${day(d)}</span>`}</span>`;

    return `<tr class="lvl-${esc(e.alert_level)}${fresh.has(e.id) ? " fresh" : ""}">
      <td>${time}</td>
      <td>${person}</td>
      <td class="door">${door}</td>
      <td class="key">${key}</td>
      <td><div class="risk" title="Site model ${e.anomaly_score ?? 0}, person profile ${e.behavioral_score ?? 0}">
        <span class="meter"><span style="width:${Math.max(4, risk)}%"></span></span>
        <span class="risk-num">${risk}</span>
        <span class="risk-label">${LEVEL_LABEL[e.alert_level] || ""}</span>
      </div></td>
      <td class="notes">${notes}</td>
    </tr>`;
  }

  function renderLog() {
    const rows = filter === "alerts" ? events.filter((e) => e.alert_level !== "NORMAL") : events;
    $("eventRows").innerHTML = rows.slice(0, MAX_ROWS).map(rowHtml).join("");
    $("emptyLog").hidden = rows.length > 0;
    $("emptyLog").textContent = filter === "alerts" ? "No alerts." : "No events recorded yet.";
    fresh.clear();
  }

  function addEvent(e, isNew) {
    if (seen.has(e.id)) return false;
    seen.add(e.id);
    lastId = Math.max(lastId, e.id);
    if (isNew) { events.unshift(e); fresh.add(e.id); } else events.push(e);
    if (events.length > MAX_ROWS * 2) seen.delete(events.pop().id);
    return true;
  }

  document.querySelectorAll(".segmented button").forEach((btn) => {
    btn.addEventListener("click", () => {
      filter = btn.dataset.filter;
      document.querySelectorAll(".segmented button").forEach((b) => b.setAttribute("aria-selected", b === btn));
      renderLog();
    });
  });

  // ── state panels ──
  function renderLink(s) {
    const link = s.link, el = $("link"), notice = $("notice");
    let state, text, message = null;
    if (link.mode === "demo") {
      state = "demo";
      text = "Demo data (simulated events)";
    } else if (link.connected) {
      const quiet = link.seconds_since_message;
      state = quiet != null && quiet > 15 ? "pending" : "ok";
      text = `Connected to ${link.port}`;
      if (quiet == null) text += ", waiting for data";
      else if (quiet > 15) text += `, no data for ${Math.round(quiet)} s`;
      if (link.firmware) text += ` · firmware ${link.firmware}`;
      if (link.firmware && link.firmware.startsWith("original")) {
        message = "The board is running the original sketch. Names are matched from its debug output; flash firmware/smart_access_monitor for exact key reporting, tailgating checks and keys-in-range.";
      }
    } else {
      state = "down";
      text = link.port ? `Disconnected (${link.port})` : "No board connected";
      message = link.message;
    }
    el.dataset.state = state;
    $("linkText").textContent = text;
    notice.hidden = !message;
    notice.textContent = message || "";
  }

  function renderSummary(s) {
    const occ = s.occupancy, t = s.today;
    $("occTotal").textContent = occ.total;
    const parts = [];
    if (occ.inside.length) parts.push(`${occ.inside.length} identified`);
    if (occ.unidentified) parts.push(`${occ.unidentified} unidentified`);
    $("occMeta").textContent = parts.join(", ") || "Nobody inside";

    $("entries").textContent = t.entries;
    $("exits").textContent = `${t.exits} exit${t.exits === 1 ? "" : "s"}` +
      (t.unidentified_entries ? `, ${t.unidentified_entries} without a key` : "");

    $("alerts").textContent = t.alerts;
    $("alertsMeta").textContent = t.alerts ? `${t.high_alerts} high` : "None";

    const last = t.last_event;
    if (last) {
      const d = parseTs(last.timestamp);
      $("lastWhen").textContent = isToday(d) ? hm(d) : `${day(d)}, ${hm(d)}`;
      const verb = last.action === "ENTER" ? "entered" : "left";
      $("lastWhat").textContent = `${last.person} ${verb}, ${ago((serverNow - d) / 1000)}`;
    } else {
      $("lastWhen").textContent = "None yet";
      $("lastWhat").innerHTML = "&nbsp;";
    }
  }

  function renderInside(s) {
    const occ = s.occupancy;
    const items = occ.inside.map((p) => {
      const since = parseTs(p.since);
      return `<li><div><div class="primary">${esc(p.person)}</div><div class="secondary">${esc(p.ble_name)}</div></div>
        <div class="value">since ${hm(since)} · ${duration(p.minutes)}</div></li>`;
    });
    if (occ.unidentified) {
      items.push(`<li><div><div class="primary">Unidentified</div><div class="secondary">entered without a key</div></div>
        <div class="value">${occ.unidentified}</div></li>`);
    }
    $("insideList").innerHTML = items.join("") || `<li class="none">Nobody inside</li>`;
  }

  function renderKeys(s) {
    const legacy = s.link.firmware && s.link.firmware.startsWith("original");
    $("keysNote").textContent = `${s.authorized_keys.length} configured`;
    if (legacy) {
      $("keysList").innerHTML = `<li class="none">Needs the updated firmware</li>`;
      return;
    }
    const items = s.keys_nearby.map((k) => `<li>
        <div><div class="primary">${esc(k.person)}</div><div class="secondary">${esc(k.key)}</div></div>
        <div class="value">${signalBars(k.rssi)}${k.rssi} dBm</div></li>`);
    $("keysList").innerHTML = items.join("") || `<li class="none">No keys heard in the last 15 s</li>`;
  }

  function renderModels(s) {
    const m = s.models;
    if (!m.loaded) {
      $("models").textContent = "No trained models, so only the rule checks are active (no key, tailgating, re-entry without exit). Run python ml_models.py to add anomaly scoring.";
      return;
    }
    const trained = m.trained_at ? parseTs(m.trained_at.replace("T", " ")) : null;
    const who = m.profiles.length ? m.profiles.join(", ") : "nobody yet";
    $("models").textContent = `Isolation Forest site model, plus behavioral profiles for ${who}.` +
      (trained ? ` Trained ${day(trained)} ${trained.getFullYear()}, ${hm(trained)}.` : "");
  }

  // ── chart: entries per hour, single series ──
  const tooltip = $("tooltip");

  function renderChart(hourly) {
    const W = 288, H = 128, left = 22, right = 4, top = 8, bottom = 18;
    const plotW = W - left - right, plotH = H - top - bottom;
    const max = Math.max(...hourly);
    const yMax = max <= 4 ? Math.max(max, 2) : Math.ceil(max / 5) * 5;
    const ticks = yMax <= 4 ? [0, yMax] : [0, yMax / 2, yMax];
    const slot = plotW / 24;
    const bw = Math.min(24, slot - 2);
    const r = Math.min(3, bw / 2);
    const nowHour = serverNow.getHours();
    const y = (v) => top + plotH - (v / yMax) * plotH;

    let svg = "";
    for (const t of ticks) {
      svg += `<line class="${t === 0 ? "base" : "grid"}" x1="${left}" x2="${W - right}" y1="${y(t)}" y2="${y(t)}"/>`;
      svg += `<text class="tick" x="${left - 6}" y="${y(t) + 3}" text-anchor="end">${t}</text>`;
    }
    hourly.forEach((v, h) => {
      const x = left + h * slot + (slot - bw) / 2;
      if (v > 0) {
        const yTop = y(v), base = y(0);
        const path = `M${x},${base} V${yTop + r} Q${x},${yTop} ${x + r},${yTop} H${x + bw - r} Q${x + bw},${yTop} ${x + bw},${yTop + r} V${base} Z`;
        svg += `<path class="bar${h > nowHour ? " dim" : ""}" d="${path}"/>`;
      }
      svg += `<rect class="hit" data-h="${h}" data-v="${v}" x="${left + h * slot}" y="${top}" width="${slot}" height="${plotH + bottom}"/>`;
    });
    for (const h of [0, 6, 12, 18]) {
      svg += `<text class="tick" x="${left + h * slot + slot / 2}" y="${H - 4}" text-anchor="middle">${pad(h)}:00</text>`;
    }

    const busiest = hourly.indexOf(max);
    const label = max ? `Entries by hour today. Busiest hour ${pad(busiest)}:00 with ${max}.` : "No entries yet today.";
    $("chart").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${label}">${svg}</svg>`;

    $("chart").querySelectorAll(".hit").forEach((hit) => {
      hit.addEventListener("mousemove", (ev) => {
        const h = +hit.dataset.h, v = +hit.dataset.v;
        tooltip.textContent = `${pad(h)}:00 to ${pad((h + 1) % 24)}:00 · ${v} ${v === 1 ? "entry" : "entries"}`;
        tooltip.hidden = false;
        tooltip.style.left = `${ev.clientX + 12}px`;
        tooltip.style.top = `${ev.clientY - 30}px`;
      });
      hit.addEventListener("mouseleave", () => { tooltip.hidden = true; });
    });
  }

  let lastHourly = "";
  async function refreshState() {
    try {
      const s = await (await fetch("/api/state")).json();
      serverNow = parseTs(s.now);
      renderLink(s);
      renderSummary(s);
      renderInside(s);
      renderKeys(s);
      renderModels(s);
      const key = JSON.stringify(s.today.hourly_entries) + serverNow.getHours();
      if (key !== lastHourly) { renderChart(s.today.hourly_entries); lastHourly = key; }
    } catch (err) {
      $("link").dataset.state = "down";
      $("linkText").textContent = "Dashboard server not responding";
    }
  }

  let stateTimer = null;
  const refreshSoon = () => { clearTimeout(stateTimer); stateTimer = setTimeout(refreshState, 250); };

  async function start() {
    const initial = await (await fetch(`/api/events?limit=${MAX_ROWS}`)).json();
    initial.forEach((e) => addEvent(e, false));
    renderLog();
    refreshState();
    setInterval(refreshState, 2000);

    const stream = new EventSource(`/api/stream?after=${lastId}`);
    stream.onmessage = (msg) => {
      if (addEvent(JSON.parse(msg.data), true)) {
        // keep the reader's place when rows are added above them
        const before = document.documentElement.scrollHeight;
        renderLog();
        if (window.scrollY > 200) window.scrollBy(0, document.documentElement.scrollHeight - before);
        refreshSoon();
      }
    };
  }

  start();
})();
