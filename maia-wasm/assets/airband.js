// Airband channel configuration page.
//
// Standalone static page (no build step) served by maia-httpd at /airband.html.
// It renders a live Canvas2D spectrum + scrolling waterfall fed by the existing
// /waterfall WebSocket, overlays the channel plan, and edits /api/airband.
// Changes are persisted to the config file and applied on receiver restart.

"use strict";

const WF_ROWS = 220;            // waterfall history rows kept off-screen
const MIN_SPAN_HZ = 60e3;       // tightest zoom span
const CHAN_HALF_BW_HZ = 6e3;    // half-width used for per-channel signal power
const SIGNAL_PRESENT_DB = 6;    // SNR above floor considered "present"
const SIGNAL_WEAK_DB = 3;
const SPUR_HALF_HZ = 12e3;      // drawn half-width of spur / DC avoid bands
// Waterfall feed rate to request so the live displays (and per-channel signal
// bars) stay responsive. The feed is shared with the main :8000 waterfall, so
// we only ever raise it, never lower a higher manual setting.
const WF_MIN_RATE_HZ = 20;

// Manual RX gain limits accepted by the backend (0..=77 dB).
const GAIN_MIN_DB = 0;
const GAIN_MAX_DB = 77;

// Wheel-zoom sensitivity. The zoom factor is exp(deltaY * k), so smaller k means
// gentler zoom. deltaY is normalized for line/page wheel modes and clamped so a
// single touchpad/touchscreen flick cannot zoom wildly.
const ZOOM_WHEEL_K = 0.0015;

// Known fixed spurs (Hz). 120.000 MHz is the 3rd harmonic of the Pluto 40 MHz
// reference; it shows up as a steady birdie regardless of tuning/gain.
const KNOWN_SPURS = [
  { hz: 120_000_000, label: "120.000 (ref 3x)" },
];

const els = {};
function $(id) { return document.getElementById(id); }

// ---- application state -----------------------------------------------------

const radio = { centerHz: 123.438e6, spanHz: 14e6, source: "AD9361" };
const caps = { maxChannels: 21, sampRate: 14e6, sampRateLocked: true };

// Editable working copy of the configuration.
let plan = {
  centerHz: 123_438_000,
  sampRate: 14_000_000,
  rfBandwidth: null,
  gainDb: 71,
  agc: "manual",
  pollMs: 20,
  channels: [], // { freq, label, enabled }
};
let savedKey = "";           // serialized snapshot of the last saved plan
let selected = -1;           // selected channel index

// waterfall / spectrum buffers
let nbins = 0;
let lineDb = null;           // Float32Array(nbins), latest spectrum in dB
let wfCanvas = null;         // offscreen full-band waterfall (nbins x WF_ROWS)
let wfCtx = null;
let wfRow = new ImageData(1, 1);
// Color scale in dB. Seeded to the same window the main maia waterfall uses
// (min 35 / max 85) and then auto-tracked to the measured noise floor below.
let colorMin = 35, colorMax = 85;
let noiseFloorDb = 45;

// dB window placed around the measured floor to reproduce the main waterfall's
// contrast: the floor sits ~64% up the scale, leaving headroom for signals.
const WF_FLOOR_BELOW_DB = 32;  // colorMin = floor - this
const WF_FLOOR_ABOVE_DB = 18;  // colorMax = floor + this

// view (zoom) state in Hz
const view = { centerHz: 123.438e6, spanHz: 14e6 };

// ---- formatting helpers ----------------------------------------------------

function fmtMhz(hz, digits = 4) { return (hz / 1e6).toFixed(digits); }
function parseMhz(str) {
  const v = parseFloat(String(str).trim());
  return Number.isFinite(v) ? v * 1e6 : NaN;
}
function clamp(x, lo, hi) { return x < lo ? lo : x > hi ? hi : x; }

function viewStart() { return view.centerHz - view.spanHz / 2; }
function viewEnd() { return view.centerHz + view.spanHz / 2; }
function bandStart() { return radio.centerHz - radio.spanHz / 2; }
function bandEnd() { return radio.centerHz + radio.spanHz / 2; }

function binToFreq(i) { return bandStart() + (i + 0.5) * radio.spanHz / nbins; }
function freqToBin(f) { return Math.round((f - bandStart()) / radio.spanHz * nbins - 0.5); }

function freqToX(f, w) { return (f - viewStart()) / view.spanHz * w; }
function xToFreq(x, w) { return viewStart() + (x / w) * view.spanHz; }

function setStatus(text, cls) {
  els.status.textContent = text;
  els.status.className = "status" + (cls ? " " + cls : "");
}

// ---- colormap (inferno-like) -----------------------------------------------

const CMAP = [
  [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
  [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164],
];
function colormap(t) {
  t = clamp(t, 0, 1) * (CMAP.length - 1);
  const i = Math.floor(t), f = t - i;
  const a = CMAP[i], b = CMAP[Math.min(i + 1, CMAP.length - 1)];
  return [
    (a[0] + (b[0] - a[0]) * f) | 0,
    (a[1] + (b[1] - a[1]) * f) | 0,
    (a[2] + (b[2] - a[2]) * f) | 0,
  ];
}

// dB at an arbitrary frequency (nearest bin).
function dbAt(f) {
  if (!lineDb) return -120;
  const i = freqToBin(f);
  if (i < 0 || i >= nbins) return -120;
  return lineDb[i];
}

// Peak dB within +/- CHAN_HALF_BW_HZ of a channel.
function channelPeakDb(f) {
  if (!lineDb) return -120;
  const lo = Math.max(0, freqToBin(f - CHAN_HALF_BW_HZ));
  const hi = Math.min(nbins - 1, freqToBin(f + CHAN_HALF_BW_HZ));
  let peak = -120;
  for (let i = lo; i <= hi; i++) if (lineDb[i] > peak) peak = lineDb[i];
  return peak;
}

// ---- waterfall WebSocket ---------------------------------------------------

function ensureBuffers(n) {
  if (n === nbins && wfCanvas) return;
  nbins = n;
  lineDb = new Float32Array(n);
  wfCanvas = document.createElement("canvas");
  wfCanvas.width = n;
  wfCanvas.height = WF_ROWS;
  wfCtx = wfCanvas.getContext("2d");
  wfCtx.fillStyle = "#05070a";
  wfCtx.fillRect(0, 0, n, WF_ROWS);
  wfRow = wfCtx.createImageData(n, 1);
}

function percentile(arr, p) {
  // Cheap approximate percentile via a histogram. The range is derived from the
  // actual data each frame so it works regardless of the absolute power scale
  // (the /waterfall feed is linear power, so dB lands wherever the front-end
  // gain/levels put it -- often well above 0 dB).
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (v <= -119) continue; // skip the zero-bin sentinel
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!(hi > lo)) return Number.isFinite(lo) ? lo : 0;
  const bins = 128;
  const hist = new Int32Array(bins);
  let cnt = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (v <= -119) continue;
    let b = Math.floor((v - lo) / (hi - lo) * bins);
    if (b < 0) b = 0; else if (b >= bins) b = bins - 1;
    hist[b]++; cnt++;
  }
  const target = cnt * p;
  let acc = 0;
  for (let b = 0; b < bins; b++) {
    acc += hist[b];
    if (acc >= target) return lo + (b + 0.5) / bins * (hi - lo);
  }
  return hi;
}

function onSpectrum(linear) {
  ensureBuffers(linear.length);
  const data = wfRow.data;
  for (let i = 0; i < nbins; i++) {
    const v = linear[i];
    const db = v > 0 ? 10 * Math.log10(v) : -120;
    lineDb[i] = db;
    const t = (db - colorMin) / (colorMax - colorMin);
    const c = colormap(t);
    const o = i * 4;
    data[o] = c[0]; data[o + 1] = c[1]; data[o + 2] = c[2]; data[o + 3] = 255;
  }
  // scroll down one row and stamp the new line on top
  wfCtx.drawImage(wfCanvas, 0, 0, nbins, WF_ROWS - 1, 0, 1, nbins, WF_ROWS - 1);
  wfCtx.putImageData(wfRow, 0, 0);

  // Track the measured noise floor and place a fixed-width dB window around it
  // (smoothed) so the colormap keeps the main waterfall's contrast.
  const floor = percentile(lineDb, 0.30);
  noiseFloorDb = noiseFloorDb * 0.9 + floor * 0.1;
  colorMin = colorMin * 0.9 + (floor - WF_FLOOR_BELOW_DB) * 0.1;
  colorMax = colorMax * 0.9 + (floor + WF_FLOOR_ABOVE_DB) * 0.1;

  // Drive the per-channel meters off every frame (not a slow timer) so the bars
  // track the live signal and fall as soon as a transmission ends.
  updateSignalMeters();
}

function connectWaterfall() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.hostname}:${location.port}/waterfall`;
  let ws;
  const open = () => {
    ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    ws.onmessage = (e) => onSpectrum(new Float32Array(e.data));
    ws.onclose = () => setTimeout(open, 1000);
    ws.onerror = () => ws.close();
  };
  open();
}

// ---- rendering -------------------------------------------------------------

function resizeCanvases() {
  const w = els.spectrum.clientWidth | 0;
  for (const c of [els.spectrum, els.waterfall]) {
    if (c.width !== w) c.width = w;
  }
  const oh = els.spectrum.height + els.waterfall.height;
  if (els.overlay.width !== w) els.overlay.width = w;
  if (els.overlay.height !== oh) els.overlay.height = oh;
  const mw = els.minimap.clientWidth | 0;
  if (els.minimap.width !== mw) els.minimap.width = mw;
}

function niceStep(span) {
  const raw = span / 8;
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / pow;
  const step = m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10;
  return step * pow;
}

function drawSpectrum() {
  const c = els.spectrum, ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.fillStyle = "#05070a";
  ctx.fillRect(0, 0, w, h);
  if (!lineDb) return;

  // frequency grid + labels
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.fillStyle = "#6b7785";
  ctx.font = "11px ui-monospace, monospace";
  ctx.lineWidth = 1;
  const step = niceStep(view.spanHz);
  const f0 = Math.ceil(viewStart() / step) * step;
  for (let f = f0; f <= viewEnd(); f += step) {
    const x = Math.round(freqToX(f, w)) + 0.5;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    ctx.fillText((f / 1e6).toFixed(step < 1e6 ? 3 : 1), x + 3, h - 4);
  }

  // dB grid
  const dbTop = colorMax, dbBot = colorMin;
  ctx.strokeStyle = "rgba(255,255,255,0.05)";
  for (let k = 0; k <= 4; k++) {
    const y = Math.round(h * k / 4) + 0.5;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  const yOf = (db) => h - (db - dbBot) / (dbTop - dbBot) * h;

  // spectrum trace
  ctx.strokeStyle = "#7fe0a0";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < w; x++) {
    const f = xToFreq(x, w);
    const y = clamp(yOf(dbAt(f)), 0, h);
    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // noise floor reference
  ctx.strokeStyle = "rgba(240,180,41,0.5)";
  ctx.setLineDash([4, 4]);
  const yf = yOf(noiseFloorDb);
  ctx.beginPath(); ctx.moveTo(0, yf); ctx.lineTo(w, yf); ctx.stroke();
  ctx.setLineDash([]);
}

function drawWaterfall() {
  const c = els.waterfall, ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.fillStyle = "#05070a";
  ctx.fillRect(0, 0, w, h);
  if (!wfCanvas) return;
  const sx = clamp(freqToBin(viewStart()), 0, nbins - 1);
  const ex = clamp(freqToBin(viewEnd()), sx + 1, nbins);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(wfCanvas, sx, 0, ex - sx, WF_ROWS, 0, 0, w, h);
}

function signalClass(snr) {
  if (snr >= SIGNAL_PRESENT_DB) return "ok";
  if (snr >= SIGNAL_WEAK_DB) return "weak";
  return "none";
}

function drawOverlay() {
  const c = els.overlay, ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);

  // spur / DC avoid bands
  if (els.show_spurs.checked) {
    const bands = KNOWN_SPURS.map(s => ({ hz: s.hz, label: s.label }));
    bands.push({ hz: radio.centerHz, label: "DC" });
    for (const b of bands) {
      const x0 = freqToX(b.hz - SPUR_HALF_HZ, w);
      const x1 = freqToX(b.hz + SPUR_HALF_HZ, w);
      if (x1 < 0 || x0 > w) continue;
      ctx.fillStyle = "rgba(240,99,99,0.16)";
      ctx.fillRect(x0, 0, x1 - x0, h);
      ctx.fillStyle = "rgba(240,99,99,0.8)";
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillText(b.label, x1 + 3, 12);
    }
  }

  // channel markers
  ctx.font = "11px ui-monospace, monospace";
  plan.channels.forEach((ch, idx) => {
    const x = freqToX(ch.freq, w);
    if (x < -20 || x > w + 20) return;
    const sel = idx === selected;
    const en = ch.enabled;
    ctx.strokeStyle = sel ? "#4aa8ff" : en ? "rgba(127,224,160,0.9)" : "rgba(154,167,180,0.6)";
    ctx.lineWidth = sel ? 2 : 1;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    // handle + label
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillRect(x - 4, 0, 8, 8);
    const name = ch.label || fmtMhz(ch.freq, 3);
    ctx.fillText(name, x + 6, 20 + (idx % 3) * 12);
    // signal pip
    const snr = channelPeakDb(ch.freq) - noiseFloorDb;
    const cls = signalClass(snr);
    ctx.fillStyle = cls === "ok" ? "#41d18a" : cls === "weak" ? "#f0b429" : "#f06363";
    ctx.beginPath(); ctx.arc(x, 14, 3, 0, Math.PI * 2); ctx.fill();
  });
}

function drawMinimap() {
  const c = els.minimap, ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.fillStyle = "#05070a";
  ctx.fillRect(0, 0, w, h);
  if (!lineDb) return;
  const xOfBand = (f) => (f - bandStart()) / radio.spanHz * w;

  // downsampled full-band trace
  ctx.strokeStyle = "rgba(127,224,160,0.7)";
  ctx.beginPath();
  for (let x = 0; x < w; x++) {
    const i0 = Math.floor(x / w * nbins);
    const i1 = Math.max(i0 + 1, Math.floor((x + 1) / w * nbins));
    let peak = -120;
    for (let i = i0; i < i1 && i < nbins; i++) if (lineDb[i] > peak) peak = lineDb[i];
    const y = h - clamp((peak - colorMin) / (colorMax - colorMin), 0, 1) * h;
    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // spur ticks
  ctx.fillStyle = "rgba(240,99,99,0.8)";
  for (const s of KNOWN_SPURS) {
    const x = xOfBand(s.hz);
    if (x >= 0 && x <= w) ctx.fillRect(x - 1, 0, 2, h);
  }
  // channel ticks
  plan.channels.forEach((ch) => {
    const x = xOfBand(ch.freq);
    if (x < 0 || x > w) return;
    ctx.fillStyle = ch.enabled ? "rgba(74,168,255,0.9)" : "rgba(154,167,180,0.6)";
    ctx.fillRect(x - 1, h - 10, 2, 10);
  });

  // viewport rectangle
  const vx0 = xOfBand(viewStart());
  const vx1 = xOfBand(viewEnd());
  ctx.strokeStyle = "#4aa8ff";
  ctx.fillStyle = "rgba(74,168,255,0.12)";
  ctx.lineWidth = 1;
  ctx.fillRect(vx0, 0, vx1 - vx0, h);
  ctx.strokeRect(vx0 + 0.5, 0.5, vx1 - vx0 - 1, h - 1);
}

function render() {
  resizeCanvases();
  drawSpectrum();
  drawWaterfall();
  drawOverlay();
  drawMinimap();
  requestAnimationFrame(render);
}

// ---- view manipulation -----------------------------------------------------

function clampView() {
  view.spanHz = clamp(view.spanHz, MIN_SPAN_HZ, radio.spanHz);
  const half = view.spanHz / 2;
  view.centerHz = clamp(view.centerHz, bandStart() + half, bandEnd() - half);
  syncSpanSlider();
}

function setView(centerHz, spanHz) {
  if (spanHz != null) view.spanHz = spanHz;
  view.centerHz = centerHz;
  clampView();
}

// Convert a wheel event to a gentle zoom factor (>1 zooms out, <1 zooms in).
function wheelZoomFactor(e) {
  const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1; // lines/pages -> px
  const dy = clamp(e.deltaY * unit, -50, 50);
  return Math.exp(dy * ZOOM_WHEEL_K);
}

function zoomAt(freq, factor) {
  const newSpan = clamp(view.spanHz * factor, MIN_SPAN_HZ, radio.spanHz);
  // keep `freq` under the cursor
  const rel = (freq - viewStart()) / view.spanHz;
  view.spanHz = newSpan;
  view.centerHz = freq - (rel - 0.5) * newSpan;
  clampView();
}

function syncSpanSlider() {
  // logarithmic slider 0..1000 over [MIN_SPAN_HZ, full]
  const t = Math.log(view.spanHz / MIN_SPAN_HZ) / Math.log(radio.spanHz / MIN_SPAN_HZ);
  els.span_slider.value = String(Math.round(clamp(t, 0, 1) * 1000));
  els.span_label.textContent = view.spanHz >= radio.spanHz * 0.999
    ? "full" : (view.spanHz / 1e6).toFixed(view.spanHz < 1e6 ? 3 : 2) + " MHz";
}

function nearestChannel(freq, pxTolHz) {
  let best = -1, bestD = Infinity;
  plan.channels.forEach((ch, i) => {
    const d = Math.abs(ch.freq - freq);
    if (d < bestD) { bestD = d; best = i; }
  });
  return bestD <= pxTolHz ? best : -1;
}

function setupOverlayInteraction() {
  const c = els.overlay;
  let mode = null;        // "pan" | "drag-channel"
  let dragIdx = -1;
  let last = 0;

  c.addEventListener("pointerdown", (e) => {
    const w = c.width;
    const f = xToFreq(e.offsetX * w / c.clientWidth, w);
    const tol = view.spanHz / c.clientWidth * 8; // ~8px
    const hit = nearestChannel(f, tol);
    if (hit >= 0) {
      mode = "drag-channel"; dragIdx = hit; selectChannel(hit);
    } else {
      mode = "pan"; last = e.clientX;
    }
    c.setPointerCapture(e.pointerId);
  });

  c.addEventListener("pointermove", (e) => {
    const w = c.width;
    if (mode === "pan") {
      const dx = (e.clientX - last) / c.clientWidth * w;
      last = e.clientX;
      view.centerHz -= dx / w * view.spanHz;
      clampView();
    } else if (mode === "drag-channel" && dragIdx >= 0) {
      const f = xToFreq(e.offsetX * w / c.clientWidth, w);
      plan.channels[dragIdx].freq = Math.round(f);
      refreshTableSoft();
      markDirty();
    }
  });

  const end = (e) => {
    if (mode) c.releasePointerCapture(e.pointerId);
    mode = null; dragIdx = -1;
  };
  c.addEventListener("pointerup", end);
  c.addEventListener("pointercancel", end);

  c.addEventListener("dblclick", (e) => {
    const w = c.width;
    const f = Math.round(xToFreq(e.offsetX * w / c.clientWidth, w));
    addChannel(f);
  });

  c.addEventListener("wheel", (e) => {
    e.preventDefault();
    const w = c.width;
    const f = xToFreq(e.offsetX * w / c.clientWidth, w);
    zoomAt(f, wheelZoomFactor(e));
  }, { passive: false });
}

function setupMinimapInteraction() {
  const c = els.minimap;
  let dragging = false;
  const toFreq = (e) => bandStart() + (e.offsetX / c.clientWidth) * radio.spanHz;
  c.addEventListener("pointerdown", (e) => {
    dragging = true; setView(toFreq(e), null); c.setPointerCapture(e.pointerId);
  });
  c.addEventListener("pointermove", (e) => { if (dragging) setView(toFreq(e), null); });
  c.addEventListener("pointerup", (e) => { dragging = false; c.releasePointerCapture(e.pointerId); });
  c.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoomAt(toFreq(e), wheelZoomFactor(e));
  }, { passive: false });
}

function setupViewControls() {
  els.span_slider.addEventListener("input", () => {
    const t = els.span_slider.value / 1000;
    view.spanHz = MIN_SPAN_HZ * Math.pow(radio.spanHz / MIN_SPAN_HZ, t);
    clampView();
  });
  els.fit_band.addEventListener("click", () => setView(radio.centerHz, radio.spanHz));
  els.tune_input.addEventListener("input", () => {
    const f = parseMhz(els.tune_input.value);
    if (!Number.isFinite(f)) return;
    // auto-zoom in to give context around the typed frequency
    const span = Math.min(view.spanHz, 0.8e6);
    setView(f, span);
  });
}

// ---- channel model + table -------------------------------------------------

function enabledCount() { return plan.channels.filter(c => c.enabled).length; }

function addChannel(freqHz, label) {
  if (!Number.isFinite(freqHz)) freqHz = Math.round(view.centerHz);
  if (enabledCount() >= caps.maxChannels) {
    alert(`Channel limit reached (${caps.maxChannels} active). Remove or disable one first.`);
    return;
  }
  plan.channels.push({ freq: Math.round(freqHz), label: label || "", enabled: true });
  selected = plan.channels.length - 1;
  refreshTable();
  markDirty();
}

function removeChannel(idx) {
  plan.channels.splice(idx, 1);
  if (selected >= plan.channels.length) selected = plan.channels.length - 1;
  refreshTable();
  markDirty();
}

function moveChannel(idx, dir) {
  const j = idx + dir;
  if (j < 0 || j >= plan.channels.length) return;
  const tmp = plan.channels[idx];
  plan.channels[idx] = plan.channels[j];
  plan.channels[j] = tmp;
  if (selected === idx) selected = j; else if (selected === j) selected = idx;
  refreshTable();
  markDirty();
}

function selectChannel(idx) {
  selected = idx;
  for (const tr of els.channel_rows.children) {
    tr.classList.toggle("selected", Number(tr.dataset.idx) === idx);
  }
}

function addPeakChannel() {
  if (!lineDb) return;
  const lo = clamp(freqToBin(viewStart()), 0, nbins - 1);
  const hi = clamp(freqToBin(viewEnd()), lo + 1, nbins);
  let peak = -120, pi = lo;
  for (let i = lo; i < hi; i++) {
    // ignore spur / DC bins
    const f = binToFreq(i);
    if (Math.abs(f - radio.centerHz) < SPUR_HALF_HZ) continue;
    if (KNOWN_SPURS.some(s => Math.abs(f - s.hz) < SPUR_HALF_HZ)) continue;
    if (lineDb[i] > peak) { peak = lineDb[i]; pi = i; }
  }
  addChannel(Math.round(binToFreq(pi)));
}

function refreshTable() {
  const tb = els.channel_rows;
  tb.textContent = "";
  plan.channels.forEach((ch, idx) => {
    const tr = document.createElement("tr");
    tr.dataset.idx = idx;
    if (idx === selected) tr.classList.add("selected");
    if (!ch.enabled) tr.classList.add("disabled");

    const on = document.createElement("input");
    on.type = "checkbox"; on.checked = ch.enabled;
    on.addEventListener("change", () => { ch.enabled = on.checked; refreshTable(); markDirty(); });

    const label = document.createElement("input");
    label.type = "text"; label.value = ch.label; label.placeholder = "(no label)";
    label.addEventListener("input", () => { ch.label = label.value; markDirty(); });

    const freq = document.createElement("input");
    freq.type = "text"; freq.value = fmtMhz(ch.freq, 4); freq.className = "freq";
    freq.addEventListener("change", () => {
      const f = parseMhz(freq.value);
      if (Number.isFinite(f)) { ch.freq = Math.round(f); setView(f, Math.min(view.spanHz, 0.8e6)); markDirty(); }
      refreshTable();
    });

    const meterWrap = document.createElement("div");
    meterWrap.className = "meter"; meterWrap.dataset.idx = idx;
    meterWrap.innerHTML = "<span></span>";

    const mk = (txt, fn, cls) => {
      const b = document.createElement("button");
      b.textContent = txt; b.className = "iconbtn" + (cls ? " " + cls : "");
      b.addEventListener("click", fn); return b;
    };
    const actions = document.createElement("div");
    actions.append(
      mk("\u2191", () => moveChannel(idx, -1), "ghost"),
      mk("\u2193", () => moveChannel(idx, 1), "ghost"),
      mk("\u2715", () => removeChannel(idx)),
    );

    const tdOn = document.createElement("td"); tdOn.append(on);
    const tdIdx = document.createElement("td"); tdIdx.textContent = idx; tdIdx.className = "col-idx";
    const tdLabel = document.createElement("td"); tdLabel.className = "col-label"; tdLabel.append(label);
    const tdFreq = document.createElement("td"); tdFreq.className = "col-freq"; tdFreq.append(freq);
    const tdSig = document.createElement("td"); tdSig.className = "col-signal"; tdSig.append(meterWrap);
    const tdAct = document.createElement("td"); tdAct.className = "col-actions"; tdAct.append(actions);
    tr.append(tdOn, tdIdx, tdLabel, tdFreq, tdSig, tdAct);
    tr.addEventListener("click", (e) => { if (e.target === tr || e.target.tagName === "TD") selectChannel(idx); });
    tb.append(tr);
  });
  updateSlotBadge();
}

// lightweight update of freq cells while dragging (avoids full rebuild)
function refreshTableSoft() {
  for (const tr of els.channel_rows.children) {
    const idx = Number(tr.dataset.idx);
    const ch = plan.channels[idx];
    if (!ch) continue;
    const freq = tr.querySelector("input.freq");
    if (freq && document.activeElement !== freq) freq.value = fmtMhz(ch.freq, 4);
  }
}

function updateSignalMeters() {
  for (const wrap of els.channel_rows.querySelectorAll(".meter")) {
    const idx = Number(wrap.dataset.idx);
    const ch = plan.channels[idx];
    if (!ch) continue;
    const snr = channelPeakDb(ch.freq) - noiseFloorDb;
    wrap.className = "meter " + signalClass(snr);
    wrap.firstChild.style.width = clamp(snr / 30, 0, 1) * 100 + "%";
  }
}

function updateSlotBadge() {
  const n = enabledCount();
  els.slot_usage.textContent = `${n} / ${caps.maxChannels}`;
  els.slot_usage.classList.toggle("full", n >= caps.maxChannels);
}

// ---- dirty tracking + front-end form --------------------------------------

function planKey() {
  return JSON.stringify({
    c: plan.centerHz, b: plan.rfBandwidth, g: plan.gainDb, a: plan.agc, p: plan.pollMs,
    ch: plan.channels.filter(c => c.enabled).map(c => [Math.round(c.freq), c.label || ""]),
  });
}
function isDirty() { return planKey() !== savedKey; }
function markDirty() {
  els.dirty_label.textContent = isDirty() ? "unsaved changes" : "";
  updateSlotBadge();
}

function readFrontEndForm() {
  // Center is locked (read-only); plan.centerHz stays whatever the device
  // reported so saved channels remain inside the capture window.
  const bw = parseMhz(els.rf_bw_input.value);
  plan.rfBandwidth = Number.isFinite(bw) ? Math.round(bw) : null;
  const g = parseFloat(els.gain_input.value);
  if (Number.isFinite(g)) {
    plan.gainDb = clamp(g, GAIN_MIN_DB, GAIN_MAX_DB);
    if (plan.gainDb !== g) els.gain_input.value = plan.gainDb; // reflect clamp
  }
  plan.agc = els.agc_select.value;
  const p = parseInt(els.poll_input.value, 10);
  if (Number.isFinite(p)) plan.pollMs = p;
}

function writeFrontEndForm() {
  els.center_input.value = fmtMhz(plan.centerHz, 4);
  els.samp_rate_input.value = (plan.sampRate / 1e6).toFixed(3) + " Msps";
  els.rf_bw_input.value = plan.rfBandwidth ? fmtMhz(plan.rfBandwidth, 3) : "";
  els.gain_input.value = plan.gainDb;
  els.agc_select.value = plan.agc;
  els.poll_input.value = plan.pollMs;
}

function setupFrontEndForm() {
  // Center and sample rate are locked (read-only), so they are not wired here.
  for (const id of ["rf_bw_input", "gain_input", "agc_select", "poll_input"]) {
    els[id].addEventListener("change", () => { readFrontEndForm(); markDirty(); });
  }
}

// ---- API -------------------------------------------------------------------

async function getJson(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

async function loadAll() {
  setStatus("loading\u2026");
  const api = await getJson("/api");
  // radio tuning for the waterfall axis
  radio.centerHz = api.ad9361.rx_lo_frequency;
  radio.spanHz = api.ad9361.sampling_frequency;
  try {
    const spec = await getJson("/api/spectrometer");
    radio.source = spec.input;
    radio.spanHz = spec.input_sampling_frequency || radio.spanHz;
    if (spec.input === "DDC" && api.ddc) radio.centerHz += api.ddc.frequency;
    // Bump a slow feed up to a responsive rate (best effort; shared resource).
    if ((spec.output_sampling_frequency || 0) < WF_MIN_RATE_HZ - 0.5) {
      fetch("/api/spectrometer", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ output_sampling_frequency: WF_MIN_RATE_HZ }),
      }).catch(() => { /* non-fatal */ });
    }
  } catch (_) { /* spectrometer optional */ }

  const ab = await getJson("/api/airband");
  caps.maxChannels = ab.max_channels;
  caps.sampRate = ab.samp_rate;
  caps.sampRateLocked = ab.samp_rate_locked;
  plan.centerHz = ab.center_hz;
  plan.sampRate = ab.samp_rate;
  plan.rfBandwidth = ab.rf_bandwidth ?? null;
  plan.gainDb = ab.gain_db;
  plan.agc = ab.agc; // snake_case from the API ("manual" / "slow_attack" / ...)
  plan.pollMs = ab.poll_ms;
  plan.channels = ab.channels.map(c => ({ freq: c.freq_hz, label: c.label || "", enabled: true }));
  savedKey = planKey();

  setView(radio.centerHz, radio.spanHz);
  writeFrontEndForm();
  refreshTable();
  markDirty();
  showRestartBanner(ab.needs_restart);
  setStatus(ab.enabled ? "receiver running" : "receiver disabled",
    ab.enabled ? "ok" : "");
}

async function save() {
  readFrontEndForm();
  const body = {
    center_hz: plan.centerHz,
    gain_db: plan.gainDb,
    agc: plan.agc,
    poll_ms: plan.pollMs,
    channels: plan.channels.filter(c => c.enabled)
      .map(c => ({ freq_hz: c.freq, label: c.label ? c.label : undefined })),
  };
  if (plan.rfBandwidth) body.rf_bandwidth = plan.rfBandwidth;
  els.save_btn.disabled = true;
  try {
    const r = await fetch("/api/airband", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { msg = (await r.json()).error_description || msg; } catch (_) {}
      alert("Save failed: " + msg);
      return;
    }
    const ab = await r.json();
    savedKey = planKey();
    markDirty();
    showRestartBanner(ab.needs_restart);
    setStatus("saved", "ok");
  } catch (e) {
    alert("Save failed: " + e.message);
  } finally {
    els.save_btn.disabled = false;
  }
}

function showRestartBanner(needs) {
  els.restart_banner.classList.toggle("hidden", !needs);
}

async function restart() {
  if (!confirm("Restart the airband receiver now to apply the saved configuration?")) return;
  try {
    const r = await fetch("/api/system/restart", { method: "POST" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
  } catch (e) {
    alert("Could not trigger restart automatically (" + e.message +
      "). Reboot the device manually to apply.");
    return;
  }
  setStatus("restarting\u2026 reconnecting", "");
  els.restart_banner.classList.add("hidden");
  // poll until the service comes back
  const t0 = Date.now();
  const poll = async () => {
    try { await getJson("/api"); await loadAll(); setStatus("receiver running", "ok"); }
    catch (_) {
      if (Date.now() - t0 < 60000) setTimeout(poll, 2000);
      else setStatus("still waiting\u2014reload the page", "err");
    }
  };
  setTimeout(poll, 4000);
}

// ---- presets / import / export ---------------------------------------------

const DEFAULT_PLAN = [
  118.05, 119.2, 119.9, 120.1, 120.4, 120.95, 121.5, 121.6, 121.7, 122.275,
  122.95, 122.975, 123.9, 124.7, 125.6, 125.9, 126.25, 126.5, 126.875, 127.1, 128.5,
];

function loadDefaultPlan() {
  if (isDirty() && !confirm("Replace the current channel list with the built-in default plan?")) return;
  plan.channels = DEFAULT_PLAN.map(mhz => ({ freq: Math.round(mhz * 1e6), label: "", enabled: true }));
  selected = -1;
  refreshTable();
  markDirty();
}

function exportJson() {
  const out = {
    center_hz: plan.centerHz,
    samp_rate: plan.sampRate,
    rf_bandwidth: plan.rfBandwidth,
    gain_db: plan.gainDb,
    agc: plan.agc,
    poll_ms: plan.pollMs,
    channels_hz: plan.channels.filter(c => c.enabled).map(c => c.freq),
    channel_labels: plan.channels.filter(c => c.enabled).map(c => c.label || ""),
  };
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "airband.json";
  a.click();
  URL.revokeObjectURL(a.href);
}

function importJson(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const j = JSON.parse(reader.result);
      if (typeof j.center_hz === "number") plan.centerHz = j.center_hz;
      if (typeof j.gain_db === "number") plan.gainDb = j.gain_db;
      if (typeof j.agc === "string") plan.agc = j.agc;
      if (typeof j.poll_ms === "number") plan.pollMs = j.poll_ms;
      plan.rfBandwidth = typeof j.rf_bandwidth === "number" ? j.rf_bandwidth : null;
      if (Array.isArray(j.channels_hz)) {
        const labels = Array.isArray(j.channel_labels) ? j.channel_labels : [];
        plan.channels = j.channels_hz.map((f, i) => ({ freq: Math.round(f), label: labels[i] || "", enabled: true }));
      }
      selected = -1;
      writeFrontEndForm();
      refreshTable();
      markDirty();
    } catch (e) {
      alert("Import failed: " + e.message);
    }
  };
  reader.readAsText(file);
}

async function setWaterfallFullBand() {
  try {
    await fetch("/api/spectrometer", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: "AD9361" }),
    });
    setTimeout(loadAll, 500);
  } catch (e) { alert("Could not change waterfall source: " + e.message); }
}

// ---- init ------------------------------------------------------------------

function cacheEls() {
  for (const id of [
    "status", "restart_banner", "restart_btn", "dismiss_banner", "spectrum",
    "waterfall", "overlay", "minimap", "tune_input", "span_slider", "span_label",
    "fit_band", "show_spurs", "floor_label", "slot_usage", "add_channel",
    "snap_peak", "sort_channels", "channel_rows", "center_input", "samp_rate_input",
    "rf_bw_input", "gain_input", "agc_select", "poll_input",
    "save_btn", "reload_btn", "dirty_label", "preset_default", "export_json",
    "import_json", "set_source_ad9361",
  ]) els[id] = $(id);
}

function setupButtons() {
  els.add_channel.addEventListener("click", () => addChannel(Math.round(view.centerHz)));
  els.snap_peak.addEventListener("click", addPeakChannel);
  els.sort_channels.addEventListener("click", () => {
    plan.channels.sort((a, b) => a.freq - b.freq);
    selected = -1; refreshTable(); markDirty();
  });
  els.save_btn.addEventListener("click", save);
  els.reload_btn.addEventListener("click", () => {
    if (!isDirty() || confirm("Discard unsaved changes and reload from the device?")) loadAll();
  });
  els.restart_btn.addEventListener("click", restart);
  els.dismiss_banner.addEventListener("click", () => els.restart_banner.classList.add("hidden"));
  els.preset_default.addEventListener("click", loadDefaultPlan);
  els.export_json.addEventListener("click", exportJson);
  els.import_json.addEventListener("change", (e) => { if (e.target.files[0]) importJson(e.target.files[0]); });
  els.set_source_ad9361.addEventListener("click", setWaterfallFullBand);
}

function updateFloorLabel() {
  els.floor_label.textContent = lineDb
    ? `noise floor ~${noiseFloorDb.toFixed(0)} dB \u00b7 suggested squelch ~${(noiseFloorDb + SIGNAL_PRESENT_DB).toFixed(0)} dB`
    : "";
}

async function init() {
  cacheEls();
  setupButtons();
  setupViewControls();
  setupFrontEndForm();
  setupOverlayInteraction();
  setupMinimapInteraction();
  connectWaterfall();
  requestAnimationFrame(render);
  // Meters update per waterfall frame (see onSpectrum); the floor label only
  // needs a lazy refresh.
  setInterval(updateFloorLabel, 250);
  window.addEventListener("beforeunload", (e) => {
    if (isDirty()) { e.preventDefault(); e.returnValue = ""; }
  });
  try { await loadAll(); }
  catch (e) { setStatus("error: " + e.message, "err"); }
}

init();
