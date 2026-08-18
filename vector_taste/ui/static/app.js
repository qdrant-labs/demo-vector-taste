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
  gen: null,        // last /api/generate result, kept so the tile can re-render on play/pause
  loop: null,       // its /api/loop scoring
  prevOrder: null,  // segment_ids of the list on screen BEFORE the current one
  moves: null,      // what the last gesture did to the ranking; drives the NEW/UP/DOWN tags
  clips: [],        // your uploads, cleared server-side on every start
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

/* --------------------------------------------------------------------- icons + mode */
/* One place that builds an icon, so the sprite ids never drift across templates. */
const icon = (name, cls = "") => `<svg class="i ${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;

/* Density. Desktop by default -- <head> already applied it before first paint, so this only
   handles switching and persistence, exactly like toggleTheme(). */
function currentMode() {
  return document.documentElement.dataset.mode === "present" ? "present" : "desktop";
}

function renderModes() {
  const m = currentMode();
  for (const b of document.querySelectorAll("#modes button")) {
    b.setAttribute("aria-checked", String(b.dataset.mode === m));
  }
}

function setMode(name, quiet) {
  document.documentElement.dataset.mode = name === "present" ? "present" : "desktop";
  try { localStorage.setItem("vt-mode", currentMode()); } catch { /* private mode */ }
  renderModes();
  if (!quiet) {
    toast(currentMode() === "present"
      ? "presentation mode — large, stripped back" : "desktop mode — dense, full detail", 2000);
  }
}

document.querySelector("#modes").addEventListener("click", (e) => {
  const b = e.target.closest("[data-mode]");
  if (b) setMode(b.dataset.mode);
});

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
  // Switching to a generator that cannot sing has to drop vocals with it, or Compose would
  // send a combination the server rejects.
  if (vocals && !canSing()) vocals = false;
  renderVocals();
  if (!quiet) toast(BACKEND_NOTE[name] || `generator: ${name}`, 2500);
  syncStatus();
}

document.querySelector("#backends").addEventListener("click", (e) => {
  const b = e.target.closest("[data-backend]");
  if (b) setBackend(b.dataset.backend);
});

/* --------------------------------------------------------------------------- vocals */
/* Instrumental everywhere except ElevenLabs: ACE-Step has no lyrics source and would sing
   wordless syllables, so the option disables itself rather than lying about what it does.
   The server rejects the combination too — this toggle is convenience, not the guard. */
let vocals = false;
let vocalsBackends = ["elevenlabs"];

const canSing = () => vocalsBackends.includes(backend);

function renderVocals() {
  const ok = canSing();
  for (const b of document.querySelectorAll("#vocals button")) {
    b.disabled = !ok;
    b.setAttribute("aria-checked", String((b.dataset.vocals === "1") === vocals));
    b.title = ok
      ? (b.dataset.vocals === "1" ? "Sung vocals" : "No singing")
      : `Vocals need the ${vocalsBackends.join(" or ")} generator`;
  }
}

function setVocals(on, quiet) {
  if (on && !canSing()) {
    toast(`the ${backend} generator cannot sing — switch generator with B`, 5000);
    return;
  }
  vocals = on;
  try { localStorage.setItem("vt-vocals", on ? "1" : "0"); } catch { /* private mode */ }
  renderVocals();
  if (!quiet) toast(on ? "vocals: sung" : "vocals: instrumental", 2200);
}

document.querySelector("#vocals").addEventListener("click", (e) => {
  const b = e.target.closest("[data-vocals]");
  if (b) setVocals(b.dataset.vocals === "1");
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
  // Swap the sprite reference rather than the text: this button is icon-only now.
  $("#playpause").querySelector("use")
    .setAttribute("href", playing ? "#i-pause" : "#i-play");
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
/* Rank movement caused by the LAST gesture, computed against the list that was on screen
   before it.

   The server only diffs when a negative is marked, and only ever against positives-only --
   so marking a positive, marking a second one, or unmarking moved the list with nothing on
   screen to say so. Measured: those gestures shift 2-8 of 12 rows each.

   Shaped exactly like the server's diff.added/diff.moved so tagFor() reads either. When a
   negative IS the gesture, the previous list is the positives-only list, so this agrees with
   the server exactly -- the centerpiece moment is unchanged. */
function computeMoves(prev, hits) {
  if (!prev || !prev.length) return null;      // nothing to compare a fresh search against
  const was = new Map(prev.map((sid, i) => [sid, i]));
  const added = [];
  const moved = {};
  hits.forEach((h, i) => {
    if (!was.has(h.segment_id)) added.push(h.segment_id);
    else if (was.get(h.segment_id) !== i) moved[h.segment_id] = [was.get(h.segment_id), i];
  });
  return { added, moved };
}

const orderOf = (hits) => hits.map((h) => h.segment_id);

function tagFor(h) {
  const d = state.moves;
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
    <button class="rowplay icon" data-play="1" aria-label="${isPlaying ? "Pause" : "Play"} ${esc(h.title)}"
            title="${isPlaying ? "Pause" : "Play"}">${icon(isPlaying ? "pause" : "play")}</button>
    <div class="rank">${dropped ? "—" : i + 1}</div>
    <div class="meta">
      <div class="name">${esc(h.artist)} — ${esc(h.title)}
        ${h.is_upload ? '<span class="tag yours">YOURS</span>' : ""} ${tagFor(h)}</div>
      <div class="sub">${esc(sub)}</div>
    </div>
    <div class="score">${dropped ? "" : h.score.toFixed(4)}</div>
    <div class="mark">
      <button class="mk ${isPos ? "on-pos" : ""}" data-mark="pos" aria-label="More like this"
              title="More like this">+</button>
      <button class="mk ${isNeg ? "on-neg" : ""}" data-mark="neg" aria-label="Less like this"
              ${canMarkNegative() ? "" : "disabled"}
              title="${canMarkNegative() ? "Less like this" : NEG_NEEDS_ANCHOR}">−</button>
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

  const bySegment = (map, kind) => {
    const seen = new Set();
    return [...map.values()].filter((h) => {
      if (seen.has(h.segment_id)) return false;
      seen.add(h.segment_id);
      return true;
    }).map((h) => chip(h, kind));
  };
  const chips = bySegment(state.pos, "pos").concat(bySegment(state.neg, "neg")).join("");
  $("#chips").innerHTML = chips || '<span class="dim">no examples marked yet</span>';
  $("#tastebar").hidden = state.pos.size === 0 && state.neg.size === 0;
  renderGenPanel();
  renderClips();
}

const chip = (h, kind) => `<span class="chip ${kind}">
  <span class="sign">${kind === "pos" ? "+" : "−"}</span>${esc(h.artist)} — ${esc(h.title)}
  <button data-unchip="${h.point_id}" class="icon" aria-label="Remove ${esc(h.title)}"
          title="Remove">${icon("x")}</button></span>`;

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

/* --------------------------------------------------------------------------- uploads */
/* Your own audio, embedded with the same model as the corpus and upserted into the same
   collection -- so an upload is searchable, markable and recommendable with no special
   cases anywhere in retrieval. The server clears these on every start. */

function renderClips() {
  const box = $("#clips");
  box.hidden = state.clips.length === 0;
  $("#cliprows").innerHTML = state.clips.map((c) => {
    const on = state.playing === c.audio_url && !audioEl.paused;
    return `
    <div class="playrow clip${on ? " playing" : ""}">
      <button class="rowplay icon" data-clip-play="${esc(c.audio_url)}"
              aria-label="${on ? "Pause" : "Play"} ${esc(c.title)}">${icon(on ? "pause" : "play")}</button>
      <span class="playrow-meta">
        <span class="playrow-kind">${esc(c.title)}</span>
        <span class="playrow-label">${c.points} chunks · ${c.bpm ?? "—"} BPM · ${esc(c.key ?? "—")}</span>
      </span>
      <span class="clip-acts">
        <button data-clip-similar="${esc(c.track_id)}" class="ghost">Find similar</button>
        <button data-clip-remove="${esc(c.track_id)}" class="ghost icon"
                aria-label="Remove ${esc(c.title)}" title="Remove">${icon("x")}</button>
      </span>
    </div>`;
  }).join("");
}

async function loadClips() {
  try {
    state.clips = (await fetch("/api/uploads").then((r) => r.json())).uploads || [];
  } catch { state.clips = []; }
  renderClips();
}

/* Every chunk of the clip becomes a positive, so "find similar" matches on the strongest
   moment of the track rather than on whatever its first ten seconds happen to be. Qdrant
   excludes the positives themselves from the results, so the clip never echoes back. */
function findSimilar(trackId) {
  const c = state.clips.find((x) => x.track_id === trackId);
  if (!c) return;
  state.pos.clear(); state.neg.clear();
  for (const pid of c.point_ids) {
    state.pos.set(pid, { point_id: pid, segment_id: c.segment_id, artist: "You",
                         title: c.title, audio_url: c.audio_url });
  }
  render();
  refreshTaste();
}

async function uploadFiles(files) {
  const list = [...files].filter((f) => f.size);
  if (!list.length) return;
  const btn = $("#uploadbtn");
  btn.disabled = true;
  btn.classList.add("busy");
  let last = null;
  for (const f of list) {
    btn.title = `embedding ${f.name.slice(0, 24)}…`;
    const body = new FormData();
    body.append("file", f);
    try {
      const r = await fetch("/api/upload", { method: "POST", body });
      const j = await r.json();
      if (!r.ok) { toast(`${f.name}: ${j.detail || "upload failed"}`, 6000); continue; }
      last = j;
      toast(`${j.title} · ${j.points} chunks embedded`
            + (j.truncated ? " (truncated to 5 min)" : ""), 4000);
    } catch (e) { toast(`${f.name}: ${e.message}`, 6000); }
  }
  btn.disabled = false;
  btn.classList.remove("busy");
  btn.title = "Add your own audio and find its neighbors";
  await loadClips();
  // Answer the question the upload was asking, without making them click again.
  if (last) findSimilar(last.track_id);
}

$("#uploadbtn").onclick = () => $("#uploadinput").click();
$("#uploadinput").addEventListener("change", (e) => {
  uploadFiles(e.target.files);
  e.target.value = "";                       // so re-picking the same file still fires
});

$("#clips").addEventListener("click", (e) => {
  const play = e.target.closest("[data-clip-play]");
  if (play) { play_(play.dataset.clipPlay); return; }
  const sim = e.target.closest("[data-clip-similar]");
  if (sim) { findSimilar(sim.dataset.clipSimilar); return; }
  const rm = e.target.closest("[data-clip-remove]");
  if (rm) removeClip(rm.dataset.clipRemove);
});

function play_(url) {
  const c = state.clips.find((x) => x.audio_url === url);
  play(url, c ? c.title : "your clip");
}

async function removeClip(trackId) {
  const c = state.clips.find((x) => x.track_id === trackId);
  try {
    await fetch(`/api/uploads/${encodeURIComponent(trackId)}`, { method: "DELETE" });
  } catch (e) { toast("could not remove: " + e.message); return; }
  // Its points are gone, so any mark pointing at them has to go too or the next query 400s.
  if (c) for (const pid of c.point_ids) { state.pos.delete(pid); state.neg.delete(pid); }
  await loadClips();
  render();
  if (state.pos.size || state.neg.size) refreshTaste();
}

/* Drag anywhere on the window. dragleave fires constantly while moving over children, so
   the veil is driven by a counter rather than by the last event seen. */
let dragDepth = 0;
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("dragenter", (e) => {
  e.preventDefault();
  if (++dragDepth === 1) $("#dropveil").hidden = false;
});
window.addEventListener("dragleave", () => {
  if (--dragDepth <= 0) { dragDepth = 0; $("#dropveil").hidden = true; }
});
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0; $("#dropveil").hidden = true;
  if (e.dataTransfer && e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
});

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
  $("#go").disabled = true; $("#go").classList.add("busy");
  try {
    const { hits } = await api("/api/search", { text, limit: 12 });
    state.hits = hits; state.diff = null; state.cursor = hits.length ? 0 : -1;
    // No tags on a bare search -- but arm the baseline, so the FIRST mark tags against it.
    state.moves = null;
    state.prevOrder = orderOf(hits);
    render();
    if (!hits.length) toast("no results");
  } catch (e) { toast("search failed: " + e.message); }
  finally { $("#go").disabled = false; $("#go").classList.remove("busy"); }
}

async function refreshTaste() {
  if (!state.pos.size && !state.neg.size) return;
  try {
    const r = await api("/api/taste", {
      positives: [...state.pos.keys()], negatives: [...state.neg.keys()],
      steer: $("#steer").value, limit: 12,
    });
    // Order matters: compare against the old list BEFORE overwriting it, or every diff
    // comes back empty.
    state.moves = computeMoves(state.prevOrder, r.hits);
    state.prevOrder = orderOf(r.hits);
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

/* ACE-Step cannot be canceled cooperatively, so the server kills the worker. The model
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
  $("#gen").textContent = "Compose";
}

async function compose() {
  if (!state.pos.size) { toast("mark at least one + first"); return; }
  if (composeTimer) { toast("already composing"); return; }   // no double-submit
  startComposeProgress();
  try {
    const g = await api("/api/generate", {
      positives: [...state.pos.keys()], negatives: [...state.neg.keys()],
      steer: $("#steer").value, backend, vocals,
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

/* One play row. Deliberately NOT an <audio controls> element: the page already has a
   transport, and a second element on the same URL played the track twice — once itself and
   once through the listener that fed the analyser. Everything routes through play(). */
function playRow(url, kind, label, score, note) {
  if (!url) return "";
  const on = state.playing === url && !audioEl.paused;
  return `
  <button class="playrow${on ? " playing" : ""}" data-play-url="${esc(url)}"
          data-play-label="${esc(label)}" aria-label="${on ? "Pause" : "Play"} ${esc(label)}">
    <span class="rowplay" aria-hidden="true">${icon(on ? "pause" : "play")}</span>
    <span class="playrow-meta">
      <span class="playrow-kind">${esc(kind)}</span>
      <span class="playrow-label">${esc(label)}</span>
    </span>
    <span class="playrow-score mono">${score ?? ""}<span class="dim"> ${esc(note || "")}</span></span>
  </button>`;
}

/* Split out of showGenerated so render() can redraw it: the play/pause icon on each row has
   to follow playback, and render() already runs on every play, pause and ended. */
function renderGenPanel() {
  const p = $("#genpanel");
  const g = state.gen;
  if (!g) { p.hidden = true; return; }
  const loop = state.loop;
  const pct = loop ? loop.percentile : null;
  const banner = g.note ? `<div class="banner">fallback: ${esc(g.note)}</div>` : "";
  const ref = g.reference;
  const base = loop && loop.baseline;
  /* These can be the same track — the style reference is the top hit, the baseline is the
     top hit whose track you did not mark, and those coincide whenever your top hit is one
     you left unmarked. Two rows for one file is noise, and both would light up as playing
     at once, so collapse them and say it does both jobs. */
  const same = base && ref && base.segment_id === ref.segment_id;

  p.innerHTML = `<div class="genhead"><h2>Composed</h2>
      <button id="closegen" class="ghost icon" aria-label="Close" title="Close">${icon("x")}</button></div>
    <div id="genbody">
      ${banner}
      ${pct !== null ? `
      <div class="bigstat">${ordinal(pct).replace(/([a-z]+)$/, '<span class="unit">$1</span>')}<span class="unit"> percentile</span></div>
      <div class="dim">closer to your taste than ${pct.toFixed(0)}% of ${loop.population} corpus segments<span class="dim"> — your uploads are not counted</span></div>
      <div class="bar"><i style="width:${Math.max(1, pct)}%"></i></div>
      <div class="bar baseline"><i style="width:${Math.max(1, loop.baseline_percentile)}%"></i></div>
      <div class="dim" style="font-size:.82em">
        gray bar: the closest human track, ${ordinal(loop.baseline_percentile)} percentile
      </div>` : `<div class="dim">score unavailable</div>`}

      <div class="playrows">
        ${playRow(g.audio_url, "Generated", g.vocals ? "this take, with vocals" : "this take",
                  loop ? loop.cosine.toFixed(4) : "", "cosine")}
        ${base ? playRow(base.audio_url,
                  same ? "Closest human · also the style reference" : "Closest human",
                  `${base.artist} — ${base.title}`,
                  loop.baseline_cosine.toFixed(4), "cosine") : ""}
        ${ref && !same ? playRow(ref.audio_url, "Style reference",
                  `${ref.artist} — ${ref.title}`, ref.score.toFixed(4), "to your search") : ""}
      </div>

      <dl class="kv">
        <dt>prompt</dt><dd>${esc(g.prompt)}</dd>
        <dt>bpm / key</dt><dd class="mono">${g.bpm ?? "—"} · ${esc(g.keyscale ?? "—")}</dd>
        <dt>generator</dt><dd class="mono">${esc(g.backend)}${g.from_bank ? " (pre-baked)" : ""}${g.vocals ? " · vocals" : ""}</dd>
      </dl>
    </div>`;
  p.hidden = false;
  $("#closegen").onclick = () => { state.gen = null; p.hidden = true; };
}

$("#genpanel").addEventListener("click", (e) => {
  const b = e.target.closest("[data-play-url]");
  if (b) play(b.dataset.playUrl, b.dataset.playLabel);
});

function showGenerated(g, loop) {
  state.gen = g;
  state.loop = loop;
  renderGenPanel();
  $("#genpanel").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "nearest" });
}

/* ------------------------------------------------------------------------ interaction */
/* Remove any entry for this hit's SEGMENT, whichever chunk it was stored under.
   Returns true if something was removed, so the caller can treat a second click as a
   toggle-off rather than stacking a near-duplicate vector onto the taste profile. */
/* Dropping the last positive leaves any negatives unanchored, which is the same broken
   state reached from the other direction. Clear them together. */
function dropStrandedNegatives() {
  if (state.pos.size === 0 && state.neg.size) {
    state.neg.clear();
    toast("cleared the − marks: they need a + to push against", 3500);
    return true;
  }
  return false;
}

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

/* A negative needs a positive to push against. On its own the recommend query heads for the
   far side of the space -- measured: none of the twelve results you were looking at survive,
   and the scores go negative. So "-" stays disabled until something is marked "+". */
const NEG_NEEDS_ANCHOR = "Mark a + first — a − on its own jumps to unrelated music";
const canMarkNegative = () => state.pos.size > 0;

function mark(pointId, kind) {
  const hit = state.hits.find((h) => h.point_id === pointId)
    || [...state.pos.values(), ...state.neg.values()].find((h) => h.point_id === pointId);
  if (!hit) return;
  if (kind === "neg" && !canMarkNegative()) { toast(NEG_NEEDS_ANCHOR, 4000); return; }
  const [add, other] = kind === "pos" ? [state.pos, state.neg] : [state.neg, state.pos];
  unmarkSegment(other, hit);                   // a result is either + or −, never both
  if (!unmarkSegment(add, hit)) add.set(pointId, hit);
  dropStrandedNegatives();                     // un-marking the last + strands the − marks
  render();
  if (state.pos.size || state.neg.size) refreshTaste(); else doSearch();
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
  const hit = state.pos.get(b.dataset.unchip) || state.neg.get(b.dataset.unchip);
  if (hit) { unmarkSegment(state.pos, hit); unmarkSegment(state.neg, hit); }
  else { state.pos.delete(b.dataset.unchip); state.neg.delete(b.dataset.unchip); }
  dropStrandedNegatives();
  render();
  if (state.pos.size) refreshTaste(); else doSearch();
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
  if (e.key === "m" || e.key === "M") {
    setMode(currentMode() === "present" ? "desktop" : "present");
    return;
  }
  if (e.key === "v" || e.key === "V") { setVocals(!vocals); return; }
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
    // If the page was loaded from a stale cache its code will not match what the server is
    // serving, and every fix will look like it did not land. Say so instead of leaving
    // someone debugging a version they are not running.
    const mine = document.body.dataset.build;
    if (s.build && mine && s.build !== mine) {
      toast("This page is out of date — reload to get the current version", 60000);
      $("#status").dataset.stale = "1";
    }
    backendsAvailable = s.backends || {};
    let stored = null;
    try { stored = localStorage.getItem("vt-backend"); } catch { /* private mode */ }
    // A stored choice only survives if this server can still do it -- otherwise a key that
    // has since been removed would make every Compose fail instead of falling back.
    vocalsBackends = s.vocals_backends || vocalsBackends;
    setBackend(backendsAvailable[stored] === true ? stored : (s.backend || "local"), true);
    let storedVocals = null;
    try { storedVocals = localStorage.getItem("vt-vocals"); } catch { /* private mode */ }
    setVocals(storedVocals === "1" && canSing(), true);
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
