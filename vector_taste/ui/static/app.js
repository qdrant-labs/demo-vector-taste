/* Vector Taste UI. No framework, no build step — the whole app is this file.
   Keyboard-driven because a presenter should never hunt for a mouse on stage. */

const $ = (s) => document.querySelector(s);
const state = {
  hits: [],
  pos: new Map(),   // point_id -> hit
  neg: new Map(),
  cursor: -1,
  diff: null,
  playing: null,
};

/* ------------------------------------------------------------------ audio + equalizer */
/* One AudioContext for the page. Created lazily on first play: Chrome and Safari refuse to
   start one without a user gesture, so doing this at load time yields a suspended context
   and a dead visualizer. */
const audioEl = new Audio();
audioEl.crossOrigin = "anonymous";
let ctx, analyser, srcNode, bins, rafId;

function initAnalyser() {
  if (ctx) return true;
  try {
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = ctx.createAnalyser();
    analyser.fftSize = 256;          // 128 bins reads well as bars at projector size
    analyser.smoothingTimeConstant = 0.75;
    srcNode = ctx.createMediaElementSource(audioEl);
    srcNode.connect(analyser);
    analyser.connect(ctx.destination);
    bins = new Uint8Array(analyser.frequencyBinCount);
    return true;
  } catch (e) {
    // Never let a decorative visualizer break playback.
    console.warn("analyser unavailable:", e);
    $("#eqnote").textContent = "visualizer unavailable";
    return false;
  }
}

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function drawEq() {
  const cv = $("#eq");
  const g = cv.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  if (!analyser) return;
  analyser.getByteFrequencyData(bins);

  /* Log-scale the frequency axis. Linear bins put almost everything in the bass and the
     display becomes one lump on the left. */
  const nBars = 64;
  const gap = 2;
  const bw = (w - gap * (nBars - 1)) / nBars;
  const style = getComputedStyle(document.body);
  const accent = style.getPropertyValue("--accent").trim() || "#DC244C";
  const violet = style.getPropertyValue("--violet").trim() || "#6047FF";
  const dim = style.getPropertyValue("--line").trim() || "#4E5366";

  /* Qdrant's signature red-to-violet gradient across the frequency axis. The brand
     reserves this gradient for hero moments; a live spectrum of the music being searched
     is the one place in this UI that qualifies. */
  const grad = g.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, accent);
  grad.addColorStop(1, violet);

  for (let i = 0; i < nBars; i++) {
    const lo = Math.floor(Math.pow(i / nBars, 2) * bins.length);
    const hi = Math.max(lo + 1, Math.floor(Math.pow((i + 1) / nBars, 2) * bins.length));
    let peak = 0;
    for (let j = lo; j < hi && j < bins.length; j++) peak = Math.max(peak, bins[j]);
    const bh = Math.max(2, (peak / 255) * (h - 4));
    g.fillStyle = peak > 8 ? grad : dim;
    g.globalAlpha = peak > 8 ? 0.55 + (peak / 255) * 0.45 : 1;
    g.fillRect(i * (bw + gap), h - bh, bw, bh);
  }
  g.globalAlpha = 1;
  rafId = requestAnimationFrame(drawEq);
}

function startViz() {
  $("#viz").hidden = false;
  if (reduceMotion) { $("#eqnote").textContent = "motion reduced"; return; }
  if (!rafId) rafId = requestAnimationFrame(drawEq);
}
function stopViz() {
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  const cv = $("#eq"), g = cv.getContext("2d");
  g.clearRect(0, 0, cv.width, cv.height);
}

function play(url, label) {
  const same = state.playing === url;
  if (same && !audioEl.paused) { audioEl.pause(); stopViz(); return; }
  initAnalyser();
  if (ctx && ctx.state === "suspended") ctx.resume();
  if (!same) { audioEl.src = url; state.playing = url; }
  audioEl.play().then(() => {
    $("#nowplaying").textContent = label || "playing";
    startViz();
    render();
  }).catch((e) => toast("playback failed: " + e.message));
}

audioEl.addEventListener("ended", () => { stopViz(); state.playing = null; syncTransport(); render(); });
audioEl.addEventListener("pause", () => { stopViz(); syncTransport(); render(); });
audioEl.addEventListener("play", () => { syncTransport(); render(); });

/* ------------------------------------------------------------------- generator toggle */
/* Which backend composes. Sent per-request, so switching takes effect on the NEXT compose
   with no server restart.

   Null until boot reads the server's GEN_BACKEND: the request ALWAYS carries a backend, so
   defaulting to a guess here would silently override the stage config (GEN_BACKEND=bank)
   with whatever this file happened to hardcode. */
let backend = null;
let backendsAvailable = {};

const BACKEND_NOTE = {
  bank: "generator: pre-baked bank (instant)",
  local: "generator: Local ACE-Step (~2 min)",
  elevenlabs: "generator: ElevenLabs (seconds)",
};

/* The header names the ACTIVE generator, so it has to be rebuilt whenever the toggle moves,
   not only on the status poll. Kept in one place so switching never drops the worker note. */
function syncStatus() {
  const el = $("#status");
  if (!el.dataset.base) return;                 // boot has not answered yet
  const worker = backend === "bank" ? "" : (el.dataset.worker || "");
  el.textContent = `${el.dataset.base} · ${backend}${worker}`;
}

function renderBackends() {
  for (const b of document.querySelectorAll("#backends button")) {
    const name = b.dataset.backend;
    const ok = backendsAvailable[name] !== false;
    b.disabled = !ok;
    b.setAttribute("aria-checked", String(name === backend));
    if (!ok) {
      b.title = name === "elevenlabs"
        ? "Set ELEVENLABS_API_KEY in .env.local to enable"
        : "Not available on this machine";
    }
  }
}

function setBackend(name, quiet) {
  if (backendsAvailable[name] === false) {
    toast(name === "elevenlabs"
      ? "ElevenLabs needs ELEVENLABS_API_KEY in .env.local" : `${name} is unavailable`, 5000);
    return;
  }
  backend = name;
  try { localStorage.setItem("vt-backend", name); } catch { /* private mode */ }
  renderBackends();
  if (!quiet) toast(BACKEND_NOTE[name] || `generator: ${name}`, 2500);
  syncStatus();
}

document.querySelector("#backends").addEventListener("click", (e) => {
  const b = e.target.closest("[data-backend]");
  if (b) setBackend(b.dataset.backend);
});

/* ------------------------------------------------------------------------- transport */
const SKIP = 15;
const fmtTime = (s) =>
  Number.isFinite(s) ? `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}` : "0:00";

/* Clamp rather than letting currentTime go negative or past the end: seeking past the end
   fires `ended` and drops the track, which mid-demo looks like a crash. */
function skip(delta) {
  if (!audioEl.src) return;
  const dur = Number.isFinite(audioEl.duration) ? audioEl.duration : Infinity;
  audioEl.currentTime = Math.max(0, Math.min(dur - 0.05, audioEl.currentTime + delta));
  syncTransport();
}

function togglePlay() {
  if (!audioEl.src) {                       // nothing loaded yet: start the cursor row
    const cur = state.hits[state.cursor] || state.hits[0];
    if (cur) play(cur.audio_url, `${cur.artist} — ${cur.title}`);
    return;
  }
  if (audioEl.paused) {
    initAnalyser();
    if (ctx && ctx.state === "suspended") ctx.resume();
    audioEl.play().then(startViz).catch((e) => toast("playback failed: " + e.message));
  } else {
    audioEl.pause();
  }
}

let seeking = false;
function syncTransport() {
  const playing = audioEl.src && !audioEl.paused;
  $("#playpause").textContent = playing ? "Pause" : "Play";
  const cur = audioEl.currentTime || 0;
  const dur = Number.isFinite(audioEl.duration) ? audioEl.duration : 0;
  $("#time").textContent = `${fmtTime(cur)} / ${fmtTime(dur)}`;
  if (!seeking && dur) $("#seek").value = Math.round((cur / dur) * 1000);
}

audioEl.addEventListener("timeupdate", syncTransport);
audioEl.addEventListener("loadedmetadata", syncTransport);
$("#playpause").onclick = togglePlay;
$("#back15").onclick = () => skip(-SKIP);
$("#fwd15").onclick = () => skip(SKIP);
$("#seek").addEventListener("input", () => { seeking = true; });
$("#seek").addEventListener("change", (e) => {
  seeking = false;
  if (Number.isFinite(audioEl.duration)) {
    audioEl.currentTime = (e.target.value / 1000) * audioEl.duration;
  }
});

/* --------------------------------------------------------------------------- rendering */
function tagFor(h) {
  const d = state.diff;
  if (!d) return "";
  if (d.added && d.added.includes(h.segment_id)) return '<span class="tag new">NEW</span>';
  if (d.moved && d.moved[h.segment_id]) {
    const [o, n] = d.moved[h.segment_id];
    const up = n < o;
    return `<span class="tag ${up ? "up" : "down"}">${up ? "UP" : "DOWN"} ${Math.abs(o - n)}</span>`;
  }
  return "";
}

/* Marks are stored by point_id — the specific 10s chunk that surfaced, which is what the
   API needs (one vector per gesture). But the chunk that wins can change between a search
   and the recommend that follows it, so matching the DISPLAY state on point_id alone makes
   a track you just marked render as unmarked. Match on segment for display. */
const markedSegments = (map) => new Set([...map.values()].map((h) => h.segment_id));

function isMarked(map, h) {
  return map.has(h.point_id) || markedSegments(map).has(h.segment_id);
}

function rowHtml(h, i, dropped) {
  const isPos = isMarked(state.pos, h);
  const isNeg = isMarked(state.neg, h);
  const cls = [
    "row",
    dropped ? "dropped" : "",
    isPos ? "is-pos" : "",
    isNeg ? "is-neg" : "",
    state.playing === h.audio_url && !audioEl.paused ? "playing" : "",
  ].join(" ");
  const isPlaying = state.playing === h.audio_url && !audioEl.paused;
  const sub = [h.bpm ? h.bpm + " BPM" : null, h.key, h.license, (h.tags || []).slice(0, 2).join(", ")]
    .filter(Boolean).join(" · ");
  return `
  <div class="${cls}" data-i="${i}" data-id="${h.point_id}" data-url="${h.audio_url}"
       tabindex="0" role="button" aria-current="${state.cursor === i}"
       aria-label="${h.artist} — ${h.title}">
    <button class="rowplay" data-play="1" aria-label="${isPlaying ? "Pause" : "Play"} ${esc(h.title)}"
            title="${isPlaying ? "Pause" : "Play"}">${isPlaying ? "❚❚" : "▶"}</button>
    <div class="rank">${dropped ? "—" : i + 1}</div>
    <div class="meta">
      <div class="name">${esc(h.artist)} — ${esc(h.title)} ${tagFor(h)}</div>
      <div class="sub">${esc(sub)}</div>
    </div>
    <div class="score">${dropped ? "" : h.score.toFixed(4)}</div>
    <div class="mark">
      <button class="mk ${isPos ? "on-pos" : ""}" data-mark="pos" aria-label="More like this">+</button>
      <button class="mk ${isNeg ? "on-neg" : ""}" data-mark="neg" aria-label="Less like this">−</button>
    </div>
  </div>`;
}

function render() {
  const box = $("#results");
  const has = state.hits.length > 0;
  $("#empty").hidden = has;
  box.innerHTML = state.hits.map((h, i) => rowHtml(h, i, false)).join("");

  if (state.diff && state.diff.dropped && state.diff.dropped.length) {
    box.insertAdjacentHTML("beforeend",
      `<div class="sub dim" style="margin-top:.6rem">dropped out after the negative:</div>` +
      state.diff.dropped.map((h) => rowHtml(h, -1, true)).join(""));
  }

  const chips = [...state.pos.values()].map((h) => chip(h, "pos"))
    .concat([...state.neg.values()].map((h) => chip(h, "neg"))).join("");
  $("#chips").innerHTML = chips || '<span class="dim">no examples marked yet</span>';
  $("#tastebar").hidden = state.pos.size === 0 && state.neg.size === 0;
}

const chip = (h, kind) => `<span class="chip ${kind}">
  <span class="sign">${kind === "pos" ? "+" : "−"}</span>${esc(h.artist)} — ${esc(h.title)}
  <button data-unchip="${h.point_id}" aria-label="Remove">×</button></span>`;

const esc = (s) => String(s ?? "").replace(/[<>&"]/g, (c) =>
  ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;" }[c]));

/* "72nd", not "72th". This number is the last thing the audience reads. */
function ordinal(n) {
  const i = Math.round(n);
  const mod100 = i % 100;
  if (mod100 >= 11 && mod100 <= 13) return `${i}th`;
  return `${i}${["th", "st", "nd", "rd"][i % 10] || "th"}`;
}

function toast(msg, ms = 3200) {
  const t = $("#toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(t._t); t._t = setTimeout(() => (t.hidden = true), ms);
}

/* ------------------------------------------------------------------------------- api */
async function api(path, body) {
  const r = await fetch(path, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    // Attach the status: callers need to tell an abort (499) from a real failure.
    throw Object.assign(new Error((await r.text()).slice(0, 160)), { status: r.status });
  }
  return r.json();
}

async function doSearch() {
  const text = $("#q").value.trim();
  if (!text) return;
  $("#go").disabled = true; $("#go").textContent = "…";
  try {
    const { hits } = await api("/api/search", { text, limit: 12 });
    state.hits = hits; state.diff = null; state.cursor = hits.length ? 0 : -1;
    render();
    if (!hits.length) toast("no results");
  } catch (e) { toast("search failed: " + e.message); }
  finally { $("#go").disabled = false; $("#go").textContent = "Search"; }
}

async function refreshTaste() {
  if (!state.pos.size && !state.neg.size) return;
  try {
    const r = await api("/api/taste", {
      positives: [...state.pos.keys()], negatives: [...state.neg.keys()],
      steer: $("#steer").value, limit: 12,
    });
    state.hits = r.hits; state.diff = r.diff;
    if (state.cursor >= state.hits.length) state.cursor = state.hits.length - 1;
    render();
    if (r.diff && !r.diff.changed) toast("that negative changed nothing — try a stronger contrast");
  } catch (e) { toast("taste failed: " + e.message); }
}

/* Compose really generates (~100-140s on an M4), so a silent spinner would read as a hang.
   The ring is driven by the server's actual progress, not by elapsed time. */
const RING_CIRCUMFERENCE = 326.7;          // 2πr for r=52, matches the CSS dasharray
let composeTimer = null;

function renderProgress(p) {
  const box = $("#progress");
  // The model load reports nothing at all, so that phase is honestly indeterminate.
  const indeterminate = p.worker === "warming" || p.phase === "loading model";
  box.classList.toggle("indeterminate", indeterminate);

  const frac = Math.max(0, Math.min(1, p.frac || 0));
  $("#ringfill").style.strokeDashoffset = indeterminate
    ? 0
    : RING_CIRCUMFERENCE * (1 - frac);
  $("#progpct").textContent = indeterminate ? "…" : `${Math.round(frac * 100)}%`;
  $("#progphase").textContent = indeterminate
    ? "loading model (first run)"
    : (p.desc || p.phase || "working");
  $("#prognote").textContent =
    p.step && p.total ? `step ${p.step} of ${p.total} · ${p.elapsed ?? 0}s`
                      : `${p.elapsed ?? 0}s elapsed`;
}

function startComposeProgress() {
  $("#gen").disabled = true;
  $("#abort").disabled = false;
  $("#acts").hidden = true;
  $("#progress").hidden = false;
  renderProgress({ frac: 0, phase: "starting", elapsed: 0 });

  const poll = async () => {
    try {
      renderProgress(await fetch("/api/progress").then((r) => r.json()));
    } catch { /* a dropped poll is not worth surfacing; the next one will land */ }
  };
  poll();
  composeTimer = setInterval(poll, 500);
}

/* ACE-Step cannot be cancelled cooperatively, so the server kills the worker. The model
   reload it costs is started immediately in the background -- the header shows "warming
   model" while you carry on marking tracks. */
async function abortCompose() {
  if (!composeTimer) return;                 // nothing in flight
  $("#abort").disabled = true;
  $("#progphase").textContent = "stopping…";
  try {
    const r = await fetch("/api/abort", { method: "POST" }).then((x) => x.json());
    toast(r.aborted ? "stopped — reloading the model in the background"
                    : "nothing was generating", 5000);
  } catch (e) { toast("could not stop: " + e.message); }
}

$("#abort").onclick = abortCompose;

function stopComposeProgress() {
  clearInterval(composeTimer);
  composeTimer = null;
  $("#progress").hidden = true;
  $("#acts").hidden = false;
  $("#gen").disabled = false;
  $("#gen").innerHTML = 'Compose <kbd>G</kbd>';
}

async function compose() {
  if (!state.pos.size) { toast("mark at least one + first"); return; }
  if (composeTimer) { toast("already composing"); return; }   // no double-submit
  startComposeProgress();
  try {
    const g = await api("/api/generate", {
      positives: [...state.pos.keys()], negatives: [...state.neg.keys()],
      steer: $("#steer").value, backend,
    });
    let loop = null;
    try {
      loop = await api("/api/loop", {
        positives: [...state.pos.keys()], negatives: [...state.neg.keys()],
        steer: $("#steer").value,
      });
    } catch (e) { console.warn("loop failed", e); }
    showGenerated(g, loop);
  } catch (e) {
    // The server answers 499 when the user stopped it; that is not a failure.
    if (e.status !== 499) toast("compose failed: " + e.message, 8000);
  } finally {
    stopComposeProgress();
    boot();                                  // refresh the header: model is warming again
  }
}

function showGenerated(g, loop) {
  const p = $("#genpanel");
  const pct = loop ? loop.percentile : null;
  const banner = g.note
    ? `<div class="banner">fallback: ${esc(g.note)}</div>` : "";

  p.innerHTML = `<div class="genhead"><h2>Composed</h2>
      <button id="closegen" class="ghost">Close</button></div>
    <div id="genbody">
      ${banner}
      ${pct !== null ? `
      <div class="bigstat">${ordinal(pct).replace(/([a-z]+)$/, '<span class="unit">$1</span>')}<span class="unit"> percentile</span></div>
      <div class="dim">closer to your taste than ${pct.toFixed(0)}% of ${loop.population} segments</div>
      <div class="bar"><i style="width:${Math.max(1, pct)}%"></i></div>
      <div class="bar baseline"><i style="width:${Math.max(1, loop.baseline_percentile)}%"></i></div>
      <div class="dim" style="font-size:.82em">
        generated cosine ${loop.cosine} · best human neighbour ${loop.baseline_cosine}
        (${loop.baseline_percentile.toFixed(0)}th) — grey bar
      </div>` : `<div class="dim">score unavailable</div>`}
      <dl class="kv">
        <dt>prompt</dt><dd>${esc(g.prompt)}</dd>
        <dt>bpm / key</dt><dd class="mono">${g.bpm ?? "—"} · ${esc(g.keyscale ?? "—")}</dd>
        <dt>backend</dt><dd class="mono">${esc(g.backend)}${g.from_bank ? " (pre-baked)" : ""}</dd>
        <dt>style ref</dt><dd>${g.reference ? esc(g.reference.artist + " — " + g.reference.title) : "—"}</dd>
      </dl>
      <audio controls src="${g.audio_url}" id="genaudio"></audio>
    </div>`;
  p.hidden = false;
  $("#closegen").onclick = () => (p.hidden = true);
  // Route the generated track through the same analyser so the A/B comparison is visible.
  $("#genaudio").addEventListener("play", () => play(g.audio_url, "generated track"));
  p.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "nearest" });
}

/* ------------------------------------------------------------------------ interaction */
/* Remove any entry for this hit's SEGMENT, whichever chunk it was stored under.
   Returns true if something was removed, so the caller can treat a second click as a
   toggle-off rather than stacking a near-duplicate vector onto the taste profile. */
function unmarkSegment(map, hit) {
  let removed = false;
  for (const [id, h] of [...map]) {
    if (id === hit.point_id || h.segment_id === hit.segment_id) {
      map.delete(id);
      removed = true;
    }
  }
  return removed;
}

function mark(pointId, kind) {
  const hit = state.hits.find((h) => h.point_id === pointId)
    || [...state.pos.values(), ...state.neg.values()].find((h) => h.point_id === pointId);
  if (!hit) return;
  const [add, other] = kind === "pos" ? [state.pos, state.neg] : [state.neg, state.pos];
  unmarkSegment(other, hit);                   // a result is either + or −, never both
  if (!unmarkSegment(add, hit)) add.set(pointId, hit);
  render();
  refreshTaste();
}

$("#results").addEventListener("click", (e) => {
  const row = e.target.closest(".row");
  if (!row) return;
  const mk = e.target.closest("[data-mark]");
  if (mk) { mark(row.dataset.id, mk.dataset.mark); return; }
  const i = +row.dataset.i;
  if (i >= 0) state.cursor = i;
  // The row's own play button toggles the CURRENT track rather than restarting it.
  if (e.target.closest("[data-play]") && state.playing === row.dataset.url) {
    togglePlay();
    return;
  }
  play(row.dataset.url, row.getAttribute("aria-label"));
});

$("#chips").addEventListener("click", (e) => {
  const b = e.target.closest("[data-unchip]");
  if (!b) return;
  state.pos.delete(b.dataset.unchip); state.neg.delete(b.dataset.unchip);
  render(); refreshTaste();
});

/* ------------------------------------------------------------------------------ theme */
/* Dark is the brand default. The initial value is set in <head> before first paint;
   this only handles toggling and persistence. */
function toggleTheme() {
  const root = document.documentElement;
  const next = root.dataset.theme === "light" ? "dark" : "light";
  root.dataset.theme = next;
  try { localStorage.setItem("vt-theme", next); } catch { /* private mode */ }
  toast(`${next} theme`, 1200);
}

$("#theme").onclick = toggleTheme;

$("#go").onclick = doSearch;
$("#q").addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });
$("#gen").onclick = compose;
$("#clear").onclick = () => {
  state.pos.clear(); state.neg.clear(); state.diff = null;
  render(); doSearch();
};

document.addEventListener("keydown", (e) => {
  const typing = ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName);
  if (e.key === "/" && !typing) { e.preventDefault(); $("#q").focus(); $("#q").select(); return; }
  if (e.key === "Escape") {
    if (composeTimer) { abortCompose(); return; }
    document.activeElement.blur();
    return;
  }
  if (typing) return;

  // Theme is handled before the "no results" guard below — otherwise it would be dead on
  // the empty state, which is exactly when someone first reaches for it.
  if (e.key === "t" || e.key === "T") { toggleTheme(); return; }
  if (e.key === "b" || e.key === "B") {
    // Cycle through the ones this server can actually do, so B never lands on a dead option.
    const usable = [...document.querySelectorAll("#backends button")]
      .map((b) => b.dataset.backend).filter((n) => backendsAvailable[n] !== false);
    if (usable.length) setBackend(usable[(usable.indexOf(backend) + 1) % usable.length]);
    return;
  }

  const n = state.hits.length;
  if (!n) return;
  const cur = state.hits[state.cursor];

  switch (e.key) {
    case "ArrowDown": case "j":
      e.preventDefault(); state.cursor = Math.min(n - 1, state.cursor + 1); render();
      document.querySelector(`[data-i="${state.cursor}"]`)?.scrollIntoView({ block: "nearest" });
      break;
    case "ArrowUp": case "k":
      e.preventDefault(); state.cursor = Math.max(0, state.cursor - 1); render();
      document.querySelector(`[data-i="${state.cursor}"]`)?.scrollIntoView({ block: "nearest" });
      break;
    case " ":
      e.preventDefault();
      // Toggle whatever is loaded; only start the cursor row if nothing is.
      if (audioEl.src) togglePlay();
      else if (cur) play(cur.audio_url, `${cur.artist} — ${cur.title}`);
      break;
    case "ArrowLeft":
      e.preventDefault(); skip(-SKIP); break;
    case "ArrowRight":
      e.preventDefault(); skip(SKIP); break;
    case "+": case "=":
      if (cur) mark(cur.point_id, "pos"); break;
    case "-": case "_":
      if (cur) mark(cur.point_id, "neg"); break;
    case "g": case "G":
      compose(); break;
  }
});

/* ------------------------------------------------------------------------------ boot */
(async function boot() {
  try {
    const s = await fetch("/api/status").then((r) => r.json());
    backendsAvailable = s.backends || {};
    let stored = null;
    try { stored = localStorage.getItem("vt-backend"); } catch { /* private mode */ }
    // A stored choice only survives if this server can still do it -- otherwise a key that
    // has since been removed would make every Compose fail instead of falling back.
    setBackend(backendsAvailable[stored] === true ? stored : (s.backend || "local"), true);
    $("#status").dataset.worker =
      s.worker === "warming" ? " · warming model…"
      : s.worker === "ready" ? " · model ready"
      : s.worker === "unavailable" ? " · no generator" : "";
    $("#status").dataset.base = `${s.points.toLocaleString()} points · ${s.target}`;
    syncStatus();
    // Poll while warming so the header flips to "model ready" on its own.
    if (s.worker === "warming") setTimeout(boot, 5000);
  } catch {
    $("#status").textContent = "qdrant unreachable — run ./scripts/qdrant_up.sh";
  }
  render();
  $("#q").focus();
})();
