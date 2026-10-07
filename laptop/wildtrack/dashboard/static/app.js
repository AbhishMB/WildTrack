"use strict";
// wildtrack ground station. Talks to the laptop over one WebSocket:
//   in:  hello, state (5 Hz), animal (1 Hz), frame meta + JPEG (binary), alert, reply
//   out: {type: "cmd", action: "lock" | "engage" | "release" | "sim_pause", track_id?}

const $ = (id) => document.getElementById(id);
const canvas = $("video");
const ctx = canvas.getContext("2d");

let ws = null;
let control = false;
let cfg = { stale_pin_s: 30, map_center: [12.2958, 76.6394], map_zoom: 15, map_max_zoom: 19 };
let state = null;
let animal = null;
let pendingMeta = null;
let lastMeta = null;
let audio = null;

// ------------------------------------------------------------------ map

const map = L.map("map", { zoomControl: true, attributionControl: false }).setView(cfg.map_center, cfg.map_zoom);
let tiles = L.tileLayer("/tiles/{z}/{x}/{y}.png", { maxZoom: cfg.map_max_zoom, maxNativeZoom: cfg.map_max_zoom }).addTo(map);
const droneMarker = L.marker([0, 0], {
  icon: L.divIcon({ className: "", html: '<div class="drone-icon" id="drone-arrow">&#x27A4;</div>', iconSize: [26, 26], iconAnchor: [13, 13] }),
  interactive: false,
});
const homeMarker = L.marker([0, 0], { icon: L.divIcon({ className: "", html: '<div class="home-icon">H</div>', iconSize: [20, 20], iconAnchor: [10, 10] }), interactive: false });
const animalPin = L.circleMarker([0, 0], { radius: 8, weight: 2, color: "#7a1d00", fillColor: "#e8590c", fillOpacity: 1 });
const sigma1 = L.circle([0, 0], { radius: 1, weight: 2, color: "#e8590c", fillOpacity: 0.15 });
const sigma2 = L.circle([0, 0], { radius: 1, weight: 1, color: "#e8590c", dashArray: "4 4", fill: false });
const trail = L.polyline([], { color: "#e8590c", weight: 3, opacity: 0.7 });
const estMarker = L.circleMarker([0, 0], { radius: 4, weight: 1, color: "#333", fillColor: "#fff", fillOpacity: 1 });
let mapCentred = false;

function setOnMap(layer, on) {
  if (on && !map.hasLayer(layer)) layer.addTo(map);
  if (!on && map.hasLayer(layer)) map.removeLayer(layer);
}

// ------------------------------------------------------------------ socket

function connect() {
  ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.binaryType = "blob";
  ws.onmessage = async (ev) => {
    if (typeof ev.data !== "string") {
      const meta = pendingMeta;
      pendingMeta = null;
      if (!meta) return;
      try {
        const bmp = await createImageBitmap(ev.data);
        drawFrame(bmp, meta);
      } catch (e) { /* a corrupt frame: skip it */ }
      return;
    }
    const m = JSON.parse(ev.data);
    switch (m.type) {
      case "hello": onHello(m); break;
      case "state": state = m; renderState(); break;
      case "animal": animal = m; renderAnimal(); break;
      case "frame": pendingMeta = m; break;
      case "alert": addAlert(m); break;
      case "reply": $("reply").textContent = (m.ok ? "" : "⚠ ") + m.message; break;
    }
  };
  ws.onclose = () => {
    $("s-link").className = "pill bad";
    $("s-link").textContent = "laptop offline";
    setTimeout(connect, 1000);
  };
}

function send(action, extra = {}) {
  unlockAudio();
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "cmd", action, ...extra }));
}

function onHello(m) {
  control = m.control;
  cfg = Object.assign(cfg, m.config);
  $("viewonly").hidden = control;
  $("sim-panel").hidden = !cfg.sim;
  $("tello-panel").hidden = cfg.backend !== "tello" || !control;
  tiles.options.maxZoom = cfg.map_max_zoom;
  if (!mapCentred) map.setView(cfg.map_center, cfg.map_zoom);
  renderButtons();
}

// ------------------------------------------------------------------ video

function boxStyle(d) {
  if (d.locked) return { color: "#22e36b", width: 4 };
  if (d.candidate) return { color: "#ffb000", width: 2 };
  if (d.lockable) return { color: "#ffe066", width: 1 };
  return { color: "#bbbbbb", width: 1 };
}

function drawFrame(bmp, meta) {
  if (canvas.width !== meta.w || canvas.height !== meta.h) {
    canvas.width = meta.w;
    canvas.height = meta.h;
  }
  ctx.drawImage(bmp, 0, 0);
  ctx.font = "bold 13px system-ui, sans-serif";
  for (const d of meta.dets) {
    const [x1, y1, x2, y2] = d.box;
    const s = boxStyle(d);
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (d.locked || d.candidate || d.lockable) {
      const label = `#${d.id} ${d.cls} ${d.conf.toFixed(2)}`;
      const w = ctx.measureText(label).width + 6;
      ctx.fillStyle = "rgba(0,0,0,0.65)";
      ctx.fillRect(x1, Math.max(0, y1 - 17), w, 16);
      ctx.fillStyle = s.color;
      ctx.fillText(label, x1 + 3, Math.max(12, y1 - 4));
    }
  }
  lastMeta = meta;
  $("novideo").hidden = true;
}

canvas.addEventListener("click", (ev) => {
  if (!lastMeta || !control) return;
  const r = canvas.getBoundingClientRect();
  const x = ((ev.clientX - r.left) / r.width) * canvas.width;
  const y = ((ev.clientY - r.top) / r.height) * canvas.height;
  const pad = 8; // small boxes are hard to hit
  const hits = lastMeta.dets.filter((d) => x >= d.box[0] - pad && x <= d.box[2] + pad && y >= d.box[1] - pad && y <= d.box[3] + pad);
  if (!hits.length) return;
  hits.sort((a, b) => (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]) - (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]));
  send("lock", { track_id: hits[0].id });
});

// ------------------------------------------------------------------ state

const fmt = (x, d = 0, unit = "") => (x === null || x === undefined ? "–" : x.toFixed(d) + unit);

function renderState() {
  const s = state;
  const mode = $("mode");
  mode.textContent = s.mode;
  mode.className = "mode " + s.mode;
  $("s-link").className = "pill " + (s.link_alive ? "ok" : "bad");
  $("s-link").textContent = s.link_alive ? "phone link" : "phone link DOWN";
  $("s-video").className = "pill " + (s.video.connected ? "ok" : "bad");
  $("s-video").textContent = s.video.connected ? `video ${s.video.w}×${s.video.h} ${s.video.fps}fps` : "video DOWN";
  $("s-lag").textContent = fmt(s.lag_s * 1000, 0, " ms");
  $("s-infer").textContent = fmt(s.infer_ms, 0, " ms") + (s.tile_search ? " (tiled)" : "");
  $("s-rate").textContent = s.mode === "AUTO" || s.mode === "LOST" ? fmt(s.cmd_rate_hz, 1, " Hz") : "–";
  const d = s.drone;
  $("s-batt").textContent = d ? fmt(d.battery_pct, 0, "%") : "–";
  $("s-sats").textContent = d ? String(d.gps_sats) : "–";
  $("s-agl").textContent = d ? fmt(d.agl_m, 0, " m") : "–";
  $("s-home").textContent = d ? fmt(d.dist_home_m, 0, " m") : "–";
  $("s-vs").textContent = d ? (d.vs_enabled ? "ON" : "off") : "–";

  if (d && d.lat !== null) {
    droneMarker.setLatLng([d.lat, d.lon]);
    setOnMap(droneMarker, true);
    const arrow = document.getElementById("drone-arrow");
    if (arrow) arrow.style.transform = `rotate(${d.yaw_deg - 90}deg)`; // the glyph points east
    if (!mapCentred) { map.setView([d.lat, d.lon], Math.max(map.getZoom(), 16)); mapCentred = true; }
    if ($("follow").checked && !map.getBounds().pad(-0.2).contains([d.lat, d.lon])) map.panTo([d.lat, d.lon]);
  }
  if (s.home) { homeMarker.setLatLng([s.home.lat, s.home.lon]); setOnMap(homeMarker, true); }
  if (s.estimate) { estMarker.setLatLng([s.estimate.lat, s.estimate.lon]); }
  setOnMap(estMarker, !!s.estimate);
  renderButtons();
  renderAnimal();
}

function renderButtons() {
  const s = state;
  const mode = s ? s.mode : "MANUAL";
  const refusals = s ? s.engage_refusals : ["no state yet"];
  $("b-lock").disabled = !control || mode !== "MANUAL" || !(s && s.candidates.length);
  $("b-engage").disabled = !control || refusals.length > 0;
  $("b-release").disabled = false; // RELEASE always works
  const ul = $("refusals");
  ul.innerHTML = "";
  if (mode === "LOCKED") {
    for (const r of refusals) {
      const li = document.createElement("li");
      li.textContent = "ENGAGE blocked: " + r;
      ul.appendChild(li);
    }
  }
}

// ------------------------------------------------------------------ animal position

function dms(deg, pos, neg) {
  const h = deg >= 0 ? pos : neg;
  let a = Math.abs(deg);
  let d = Math.floor(a);
  let m = Math.floor((a - d) * 60);
  let s = ((a - d) * 60 - m) * 60;
  if (s >= 59.995) { s = 0; m += 1; }
  if (m === 60) { m = 0; d += 1; }
  return `${d}°${String(m).padStart(2, "0")}'${s.toFixed(2).padStart(5, "0")}"${h}`;
}

const COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
const label = (cls) => (cls || "animal").replace("_", " ");

function positionText(a) {
  const t = new Date(a.last_seen_ms).toLocaleTimeString();
  return `${label(a.cls)} at ${a.lat.toFixed(6)}, ${a.lon.toFixed(6)} (±${Math.round(a.sigma_m)} m) seen ${t}. ` +
    `Maps: https://maps.google.com/?q=${a.lat.toFixed(6)},${a.lon.toFixed(6)}`;
}

function renderAnimal() {
  const a = animal;
  $("c-none").hidden = !!a;
  $("c-body").hidden = !a;
  if (!a) return;
  const nowMs = state ? state.wall_ms : Date.now();
  const age = Math.max(0, (nowMs - a.last_seen_ms) / 1000);
  const stale = age > cfg.stale_pin_s;
  $("coords-panel").classList.toggle("stale", stale);
  $("c-class").textContent = `— ${label(a.cls)}${a.track_id !== null && a.locked ? " #" + a.track_id : ""}`;
  $("c-dec").textContent = `${a.lat.toFixed(6)}, ${a.lon.toFixed(6)}`;
  $("c-dms").textContent = `${dms(a.lat, "N", "S")}  ${dms(a.lon, "E", "W")}`;
  $("c-sigma").textContent = `±${a.sigma_m.toFixed(1)} m (1σ), ±${(2 * a.sigma_m).toFixed(0)} m (2σ)`;
  $("c-seen").textContent = `${new Date(a.last_seen_ms).toLocaleTimeString()} (${age < 2 ? "now" : Math.round(age) + " s ago"})` + (stale ? " — STALE" : "");
  $("c-move").textContent = a.speed_mps < 0.3 ? "stationary" :
    `${a.speed_mps.toFixed(1)} m/s (${(a.speed_mps * 3.6).toFixed(0)} km/h) toward ${COMPASS[Math.round(a.heading_deg / 45) % 8]} (${Math.round(a.heading_deg)}°)`;
  const q = `${a.lat.toFixed(6)},${a.lon.toFixed(6)}`;
  $("a-gmaps").href = `https://www.google.com/maps/search/?api=1&query=${q}`;
  $("a-geo").href = `geo:${q}?q=${q}(${encodeURIComponent(label(a.cls))})`;
  $("a-sms").href = `sms:?&body=${encodeURIComponent(positionText(a))}`;

  const ll = [a.lat, a.lon];
  const colour = stale ? "#888888" : "#e8590c";
  animalPin.setLatLng(ll).setStyle({ fillColor: colour, color: stale ? "#555" : "#7a1d00" });
  sigma1.setLatLng(ll).setRadius(a.sigma_m).setStyle({ color: colour });
  sigma2.setLatLng(ll).setRadius(2 * a.sigma_m).setStyle({ color: colour });
  trail.setLatLngs(a.trail || []).setStyle({ color: colour });
  for (const l of [trail, sigma2, sigma1, animalPin]) setOnMap(l, true);
}

// ------------------------------------------------------------------ coordinate sharing

function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
  const ta = document.createElement("textarea"); // http on a phone: no clipboard API
  ta.value = text;
  document.body.appendChild(ta);
  ta.select();
  document.execCommand("copy");
  ta.remove();
  return Promise.resolve();
}

$("b-copy").onclick = () => {
  if (!animal) return;
  copyText(`${animal.lat.toFixed(6)}, ${animal.lon.toFixed(6)}`).then(() => ($("reply").textContent = "coordinates copied"));
};

function qrPayload(kind) {
  const q = `${animal.lat.toFixed(6)},${animal.lon.toFixed(6)}`;
  if (kind === "geo") return `geo:${q}?q=${q}`;
  if (kind === "text") return positionText(animal);
  return `https://www.google.com/maps/search/?api=1&query=${q}`;
}

function showQr() {
  if (!animal) return;
  const kind = document.querySelector('input[name="qrkind"]:checked').value;
  const text = qrPayload(kind);
  $("qr-img").src = "/api/qr.svg?text=" + encodeURIComponent(text);
  $("qr-text").textContent = text;
  $("qr-modal").hidden = false;
}
$("b-qr").onclick = showQr;
for (const r of document.querySelectorAll('input[name="qrkind"]')) r.onchange = showQr;
$("b-qr-close").onclick = () => ($("qr-modal").hidden = true);

// ------------------------------------------------------------------ alerts

function unlockAudio() {
  if (!audio) {
    try { audio = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { audio = null; }
  }
}

function beep() {
  if (!audio) return;
  for (let i = 0; i < 3; i++) {
    const o = audio.createOscillator();
    const g = audio.createGain();
    o.frequency.value = 880;
    g.gain.value = 0.2;
    o.connect(g).connect(audio.destination);
    o.start(audio.currentTime + i * 0.3);
    o.stop(audio.currentTime + i * 0.3 + 0.18);
  }
}

let alarmTimer = null;
function addAlert(m) {
  const li = document.createElement("li");
  li.className = m.level;
  const t = document.createElement("time");
  t.textContent = new Date(m.wall_ms).toLocaleTimeString();
  li.append(t, m.text);
  $("alerts").prepend(li);
  if (m.level === "alarm") {
    $("alarm").textContent = m.text;
    $("alarm").hidden = false;
    beep();
    clearTimeout(alarmTimer);
    alarmTimer = setTimeout(() => ($("alarm").hidden = true), 15000);
  }
}
$("alarm").onclick = () => ($("alarm").hidden = true);

// ------------------------------------------------------------------ controls

$("b-lock").onclick = () => send("lock");
$("b-engage").onclick = () => send("engage");
$("b-release").onclick = () => send("release");
$("b-pause").onclick = () => send("sim_pause");
$("b-takeoff").onclick = () => send("takeoff");
$("b-land").onclick = () => send("land");
$("b-emergency").onclick = () => send("emergency");

// Manual flight (Tello): hold keys, sticks are re-sent at 10 Hz; the link hovers if they stop.
const STICK_KEYS = { w: ["fwd", 1], s: ["fwd", -1], d: ["right", 1], a: ["right", -1], r: ["up", 1], f: ["up", -1], ".": ["yaw", 1], ",": ["yaw", -1] };
const held = new Set();
let sticksSent = false;
document.addEventListener("keydown", (ev) => { if (STICK_KEYS[ev.key.toLowerCase()] && !ev.repeat) held.add(ev.key.toLowerCase()); });
document.addEventListener("keyup", (ev) => held.delete(ev.key.toLowerCase()));
window.addEventListener("blur", () => held.clear());
setInterval(() => {
  if (cfg.backend !== "tello" || !control || !state || state.mode !== "MANUAL") return;
  const sticks = { fwd: 0, right: 0, up: 0, yaw: 0 };
  for (const k of held) { const [axis, v] = STICK_KEYS[k]; sticks[axis] += v; }
  const any = held.size > 0;
  if (any || sticksSent) send("stick", sticks);
  sticksSent = any;
}, 100);

document.addEventListener("keydown", (ev) => {
  if (ev.target.tagName === "INPUT" || ev.target.tagName === "TEXTAREA") return;
  if (ev.key === "Escape") { send("release"); $("qr-modal").hidden = true; ev.preventDefault(); }
  else if (!control) return;
  else if (ev.key === "l" || ev.key === "L") send("lock");
  else if (ev.key === "e" || ev.key === "E") send("engage");
});
document.addEventListener("click", unlockAudio, { once: true });

connect();
// Age counters keep moving between animal messages.
setInterval(() => { if (animal && state) renderAnimal(); }, 1000);
