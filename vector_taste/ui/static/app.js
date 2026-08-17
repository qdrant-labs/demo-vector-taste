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
  const accent = style.getPropertyValue("--accent").trim() || "#38bdf8";
  const dim = style.getPropertyValue("--line").trim() || "#262d36";

  for (let i = 0; i < nBars; i++) {
    const lo = Math.floor(Math.pow(i / nBars, 2) * bins.length);
    const hi = Math.max(lo + 1, Math.floor(Math.pow((i + 1) / nBars, 2) * bins.length));
    let peak = 0;
    for (let j = lo; j < hi && j < bins.length; j++) peak = Math.max(peak, bins[j]);
    const bh = Math.max(2, (peak / 255) * (h - 4));
    g.fillStyle = peak > 8 ? accent : dim;
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

audioEl.addEventListener("ended", () => { stopViz(); state.playing = null; render(); });
audioEl.addEventListener("pause", () => stopViz());

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

function rowHtml(h, i, dropped) {
  const isPos = state.pos.has(h.point_id);
  const isNeg = state.neg.has(h.point_id);
  const cls = [
    "row",
    dropped ? "dropped" : "",
    isPos ? "is-pos" : "",
    isNeg ? "is-neg" : "",
    state.playing === h.audio_url && !audioEl.paused ? "playing" : "",
  ].join(" ");
  const sub = [h.bpm ? h.bpm + " BPM" : null, h.key, h.license, (h.tags || []).slice(0, 2).join(", ")]
    .filter(Boolean).join(" · ");
  return `
  <div class="${cls}" data-i="${i}" data-id="${h.point_id}" data-url="${h.audio_url}"
       tabindex="0" role="button" aria-current="${state.cursor === i}"
       aria-label="${h.artist} — ${h.title}">
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
  if (!r.ok) throw new Error((await r.text()).slice(0, 160));
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

async function compose() {
  if (!state.pos.size) { toast("mark at least one + first"); return; }
  $("#gen").disabled = true; $("#gen").textContent = "composing…";
  try {
    const g = await api("/api/generate", {
      positives: [...state.pos.keys()], negatives: [...state.neg.keys()],
      steer: $("#steer").value,
    });
    let loop = null;
    try {
      loop = await api("/api/loop", {
        positives: [...state.pos.keys()], negatives: [...state.neg.keys()],
        steer: $("#steer").value,
      });
    } catch (e) { console.warn("loop failed", e); }
    showGenerated(g, loop);
  } catch (e) { toast("compose failed: " + e.message); }
  finally { $("#gen").disabled = false; $("#gen").innerHTML = 'Compose <kbd>G</kbd>'; }
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
      <div class="bigstat">${pct.toFixed(0)}<span class="unit">th percentile</span></div>
      <div class="dim">closer to your taste than ${pct.toFixed(0)}% of ${loop.population} segments</div>
      <div class="bar"><i style="width:${Math.max(1, pct)}%"></i></div>
      <div class="bar baseline"><i style="width:${Math.max(1, loop.baseline_percentile)}%"></i></div>
      <div class="dim" style="font-size:.82em">
        generated cosine ${loop.cosine} · best human neighbour ${loop.baseline_cosine}
        (${loop.baseline_percentile.toFixed(0)}th) — grey bar
      </div>` : `<div class="dim">score unavailable</div>`}
      <dl class="kv">
        <dt>prompt</dt><dd>${esc(g.prompt)}</dd>
        <dt>bpm / key</dt><dd>${g.bpm ?? "—"} · ${esc(g.keyscale ?? "—")}</dd>
        <dt>backend</dt><dd>${esc(g.backend)}${g.from_bank ? " (pre-baked)" : ""}</dd>
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
function mark(pointId, kind) {
  const hit = state.hits.find((h) => h.point_id === pointId)
    || [...state.pos.values(), ...state.neg.values()].find((h) => h.point_id === pointId);
  if (!hit) return;
  const [add, other] = kind === "pos" ? [state.pos, state.neg] : [state.neg, state.pos];
  other.delete(pointId);                       // a result is either + or −, never both
  if (add.has(pointId)) add.delete(pointId); else add.set(pointId, hit);
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
  play(row.dataset.url, row.getAttribute("aria-label"));
});

$("#chips").addEventListener("click", (e) => {
  const b = e.target.closest("[data-unchip]");
  if (!b) return;
  state.pos.delete(b.dataset.unchip); state.neg.delete(b.dataset.unchip);
  render(); refreshTaste();
});

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
  if (e.key === "Escape") { document.activeElement.blur(); return; }
  if (typing) return;

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
      if (cur) { e.preventDefault(); play(cur.audio_url, `${cur.artist} — ${cur.title}`); }
      break;
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
    $("#status").textContent = `${s.points.toLocaleString()} points · ${s.target} · ${s.backend}`;
  } catch {
    $("#status").textContent = "qdrant unreachable — run ./scripts/qdrant_up.sh";
  }
  render();
  $("#q").focus();
})();
