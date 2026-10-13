'use strict';
/* Erklär-Website zum neuronalen Netz. Kein Framework, keine externen Bibliotheken. */

// ===========================================================================
// Hilfsfunktionen
// ===========================================================================
const $ = (s) => document.querySelector(s);
const SVGNS = 'http://www.w3.org/2000/svg';
const C1 = '#2a78d6', C2 = '#eb6834', INK = '#0b0b0b', INK2 = '#52514e', MUTED = '#898781', LINE = '#e4e3df';
const GOOD = '#0ca30c', BAD = '#d03b3b', SEGOFF = '#ecebe7';

function svg(tag, attrs = {}, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function html(tag, attrs = {}, parent, text) {
  const e = document.createElement(tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const sig = (z) => 1 / (1 + Math.exp(-z));
const f = (v, d = 3) => {
  const s = Number(v).toFixed(d);
  return s.startsWith('-') && Number(s) === 0 ? s.slice(1) : s;
};
function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function mix(a, b, t) {
  t = Math.max(0, Math.min(1, t));
  const A = hexToRgb(a), B = hexToRgb(b);
  return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(',')})`;
}
function animate(ms, fn) {
  return new Promise((res) => {
    const t0 = performance.now();
    let fin = false;
    const end = () => { if (!fin) { fin = true; fn(1); res(); } };
    function frame(now) {
      if (fin) return;
      const t = Math.min(1, (now - t0) / ms);
      fn(t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);   // weich anfahren und bremsen
      if (t < 1) requestAnimationFrame(frame); else end();
    }
    requestAnimationFrame(frame);
    setTimeout(end, ms + 200);   // falls der Tab im Hintergrund ist und keine Frames zeichnet
  });
}
function mulberry32(seed) {   // kleiner Zufallsgenerator mit Startwert -> gleiche Startgewichte bei gleicher Nummer
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const tip = $('#tip');
function showTip(ev, text) { tip.textContent = text; tip.style.display = 'block'; tip.style.left = ev.clientX + 12 + 'px'; tip.style.top = ev.clientY + 12 + 'px'; }
function hideTip() { tip.style.display = 'none'; }

// Trainingsdaten (wie in part2_train.py)
const XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]];
const XOR_Y = [[0], [1], [1], [0]];
const SEG_X = [...Array(10).keys()].map((d) => [(d >> 3) & 1, (d >> 2) & 1, (d >> 1) & 1, d & 1]);
const SEG_Y = [
  [1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 1], [1, 1, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1, 1],
  [1, 0, 1, 1, 0, 1, 1], [1, 0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1],
];
const SEGNAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G'];

// ===========================================================================
// Das Netz in JavaScript – derselbe Algorithmus wie python/nn.py
// ===========================================================================
class Net {
  constructor(sizes, seed = 1) {
    const rnd = mulberry32(seed);
    const u = () => rnd() * 2 - 1;          // wie random.uniform(-1, 1)
    this.sizes = sizes;
    this.L = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const W = [], b = [];
      for (let j = 0; j < sizes[l + 1]; j++) {
        W.push(Array.from({ length: sizes[l] }, u));
        b.push(u());
      }
      this.L.push({ W, b });
    }
  }
  static fromPython(json) {
    const n = new Net(json.layer_sizes);
    n.L = json.layers.map((layer) => ({ W: layer.map((x) => x.weights.slice()), b: layer.map((x) => x.bias) }));
    return n;
  }
  clone() { const n = new Net(this.sizes); n.L = this.L.map((l) => ({ W: l.W.map((r) => r.slice()), b: l.b.slice() })); return n; }
  forward(x) {
    const acts = [x];
    let a = x;
    for (const { W, b } of this.L) {
      a = W.map((w, j) => sig(w.reduce((s, wi, i) => s + wi * a[i], b[j])));
      acts.push(a);
    }
    return acts;
  }
  predict(x) { const a = this.forward(x); return a[a.length - 1]; }
  trainStep(x, y, lr) {
    const acts = this.forward(x);
    const out = acts[acts.length - 1];
    let err = out.map((o, j) => o - y[j]);                 // dJ/dy_hat
    const mse = err.reduce((s, e) => s + e * e, 0) / err.length;
    for (let l = this.L.length - 1; l >= 0; l--) {
      const { W, b } = this.L[l], inp = acts[l], a = acts[l + 1];
      const prev = new Array(inp.length).fill(0);
      for (let j = 0; j < W.length; j++) {
        const delta = err[j] * a[j] * (1 - a[j]);            // Fehler · Sigmoid-Ableitung
        for (let i = 0; i < inp.length; i++) {
          prev[i] += delta * W[j][i];                        // vor dem Update (alte Gewichte)
          W[j][i] -= lr * delta * inp[i];
        }
        b[j] -= lr * delta;
      }
      err = prev;
    }
    return mse;
  }
  epoch(X, Y, lr) { let s = 0; for (let k = 0; k < X.length; k++) s += this.trainStep(X[k], Y[k], lr); return s / X.length; }
  correctBits(X, Y) {
    let c = 0;
    X.forEach((x, k) => this.predict(x).forEach((o, j) => { if ((o >= .5 ? 1 : 0) === Y[k][j]) c++; }));
    return c;
  }
}

// ===========================================================================
// Netz als SVG zeichnen und animieren
// ===========================================================================
class NetView {
  constructor(el, sizes, opt = {}) {
    this.el = el; this.sizes = sizes; this.opt = opt;
    el.innerHTML = '';
    const vb = el.viewBox.baseVal, W = vb.width, H = vb.height;
    const big = Math.max(...sizes);
    this.r = opt.r || (big > 10 ? 10 : 24);
    const padX = opt.padX || 80, padY = opt.padY || 34;
    this.pos = sizes.map((n, l) => {
      const x = padX + l * (W - 2 * padX) / (sizes.length - 1);
      const gap = Math.min(n > 1 ? (H - 2 * padY) / (n - 1) : 0, opt.maxGap || 120);
      return Array.from({ length: n }, (_, i) => ({ x, y: H / 2 + (i - (n - 1) / 2) * gap }));
    });
    this.gEdges = svg('g', {}, el);
    this.gNodes = svg('g', {}, el);
    this.gDots = svg('g', {}, el);
    this.edges = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      this.edges.push(this.pos[l + 1].map((p2, j) => this.pos[l].map((p1, i) => {
        const e = svg('line', { x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y, class: 'edge', stroke: LINE, 'stroke-width': 1.5 }, this.gEdges);
        e.addEventListener('mousemove', (ev) => showTip(ev, `Gewicht = ${f(e.__w ?? 0)}`));
        e.addEventListener('mouseleave', hideTip);
        return e;
      })));
    }
    this.nodes = this.pos.map((layer, l) => layer.map((p, i) => {
      const g = svg('g', { class: 'node' }, this.gNodes);
      const c = svg('circle', { cx: p.x, cy: p.y, r: this.r, fill: '#fff' }, g);
      const t = svg('text', { x: p.x, y: p.y }, g);
      return { g, c, t, l, i, ...p };
    }));
    const lab = (arr, side) => arr && arr.forEach((txt, i) => {
      const layer = side < 0 ? this.pos[0] : this.pos[sizes.length - 1];
      const p = layer[i];
      svg('text', { x: p.x + side * (this.r + 10), y: p.y, class: 'lbl', 'text-anchor': side < 0 ? 'end' : 'start', 'dominant-baseline': 'central' }, el).textContent = txt;
    });
    lab(opt.inLabels, -1);
    lab(opt.outLabels, 1);
    if (opt.layerNames) opt.layerNames.forEach((txt, l) => {
      svg('text', { x: this.pos[l][0].x, y: 16, class: 'lbl', 'text-anchor': 'middle' }, el).textContent = txt;
    });
  }
  setWeights(net) {
    const scale = this.opt.wScale || 0.9;
    net.L.forEach((layer, l) => layer.W.forEach((row, j) => row.forEach((w, i) => {
      const e = this.edges[l][j][i];
      e.__w = w;
      e.setAttribute('stroke', w >= 0 ? C1 : C2);
      e.setAttribute('stroke-width', 0.6 + Math.min(Math.abs(w), 8) * scale);
    })));
  }
  setActs(acts, upto = acts.length - 1) {
    this.nodes.forEach((layer, l) => layer.forEach((n, i) => {
      if (l > upto || !acts[l]) return this.setNode(n, null);
      this.setNode(n, acts[l][i]);
    }));
  }
  setNode(n, a, label) {
    if (a === null || a === undefined) {
      n.c.setAttribute('fill', '#fff'); n.t.textContent = ''; return;
    }
    n.c.setAttribute('fill', mix('#ffffff', C1, a));
    n.t.setAttribute('fill', a > .55 ? '#fff' : INK);
    if (this.r >= 18) n.t.textContent = label ?? f(a, 2);
  }
  clear() { this.nodes.flat().forEach((n) => { this.setNode(n, null); n.g.classList.remove('sel'); n.c.style.stroke = ''; n.c.style.strokeWidth = ''; }); }
  ring(l, i, color = C2) { const n = this.nodes[l][i]; n.c.style.stroke = color; n.c.style.strokeWidth = '4'; }
  // Punkte wandern über alle Kanten von Schicht l nach l+1 (oder rückwärts)
  async flow(l, { reverse = false, color = C1, ms = 800, only } = {}) {
    const dots = [];
    this.edges[l].forEach((row, j) => row.forEach((e, i) => {
      if (only && !only(j, i)) return;
      const a = this.pos[l][i], b = this.pos[l + 1][j];
      const [s, t] = reverse ? [b, a] : [a, b];
      dots.push({ c: svg('circle', { r: this.r > 15 ? 6 : 3.5, fill: color, cx: s.x, cy: s.y }, this.gDots), s, t });
    }));
    await animate(ms, (k) => dots.forEach((d) => {
      d.c.setAttribute('cx', d.s.x + (d.t.x - d.s.x) * k);
      d.c.setAttribute('cy', d.s.y + (d.t.y - d.s.y) * k);
    }));
    dots.forEach((d) => d.c.remove());
  }
  flash(l) {
    this.edges[l].flat().forEach((e) => { e.classList.remove('flash'); void e.getBBox(); e.classList.add('flash'); });
  }
}

// ===========================================================================
// Liniendiagramm für den Fehler (logarithmische y-Achse, Hover)
// ===========================================================================
class LossChart {
  constructor(canvas, { xMax = 20000, yMin = 1e-4, yMax = 1, series = [{ name: 'MSE', color: C1 }] } = {}) {
    Object.assign(this, { canvas, xMax, yMin, yMax, series });
    this.ctx = canvas.getContext('2d');
    this.data = series.map(() => []);
    this.hoverX = null;
    canvas.addEventListener('mousemove', (ev) => {
      const r = canvas.getBoundingClientRect();
      this.hoverX = (ev.clientX - r.left) * canvas.width / r.width;
      this.draw();
      const ep = this.xToEpoch(this.hoverX);
      if (ep === null) return hideTip();
      const parts = this.series.map((s, si) => {
        const p = this.nearest(si, ep);
        return p ? `${s.name}: ${f(p[1], 5)}` : null;
      }).filter(Boolean);
      if (parts.length) showTip(ev, `Epoche ${Math.round(ep)} · ` + parts.join(' · ')); else hideTip();
    });
    canvas.addEventListener('mouseleave', () => { this.hoverX = null; hideTip(); this.draw(); });
    this.draw();
  }
  get m() { return { l: 62, r: 14, t: 12, b: 38 }; }
  xToEpoch(px) {
    const { l, r } = this.m, w = this.canvas.width - l - r;
    if (px < l || px > l + w) return null;
    return (px - l) / w * this.xMax;
  }
  nearest(si, ep) {
    const d = this.data[si];
    if (!d.length) return null;
    let best = d[0];
    for (const p of d) if (Math.abs(p[0] - ep) < Math.abs(best[0] - ep)) best = p;
    return Math.abs(best[0] - ep) < this.xMax * 0.03 ? best : null;
  }
  clear() { this.data = this.series.map(() => []); this.draw(); }
  push(si, x, y) { this.data[si].push([x, y]); }
  draw() {
    const { ctx, canvas } = this, { l, r, t, b } = this.m;
    const W = canvas.width - l - r, H = canvas.height - t - b;
    const lyMin = Math.log10(this.yMin), lyMax = Math.log10(this.yMax);
    const X = (x) => l + x / this.xMax * W;
    const Y = (y) => t + (1 - (Math.log10(Math.max(y, this.yMin)) - lyMin) / (lyMax - lyMin)) * H;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = '13px system-ui, sans-serif';
    ctx.lineWidth = 1;
    for (let e = Math.ceil(lyMin); e <= lyMax; e++) {            // Gitterlinien je Zehnerpotenz
      const y = Y(10 ** e);
      ctx.strokeStyle = '#efeeea'; ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(l + W, y); ctx.stroke();
      ctx.fillStyle = MUTED; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText(e >= 0 ? String(10 ** e) : (10 ** e).toFixed(-e), l - 8, y);
    }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let i = 0; i <= 4; i++) {
      const x = this.xMax * i / 4;
      ctx.fillStyle = MUTED; ctx.fillText(String(Math.round(x)), X(x), t + H + 8);
    }
    ctx.strokeStyle = LINE; ctx.beginPath(); ctx.moveTo(l, t + H); ctx.lineTo(l + W, t + H); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.textAlign = 'right'; ctx.fillText('Epoche', l + W, t + H + 22);
    this.series.forEach((s, si) => {
      const d = this.data[si];
      if (!d.length) return;
      const step = Math.max(1, Math.floor(d.length / 1500));
      ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let k = 0; k < d.length; k += step) { const [x, y] = d[k]; k ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)); }
      const last = d[d.length - 1]; ctx.lineTo(X(last[0]), Y(last[1]));
      ctx.stroke();
      if (d.length < 40) d.forEach(([x, y]) => { ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(X(x), Y(y), 4, 0, 7); ctx.fill(); });
    });
    if (this.hoverX !== null && this.hoverX >= l && this.hoverX <= l + W) {
      ctx.strokeStyle = MUTED; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(this.hoverX, t); ctx.lineTo(this.hoverX, t + H); ctx.stroke(); ctx.setLineDash([]);
    }
  }
}

// ===========================================================================
// 7-Segment-Anzeige als SVG
// ===========================================================================
function makeSegDisplay(el) {
  el.innerHTML = '';
  el.setAttribute('viewBox', '0 0 200 320');
  const t = 20;
  const hz = (x1, x2, y) => [[x1, y], [x1 + t / 2, y - t / 2], [x2 - t / 2, y - t / 2], [x2, y], [x2 - t / 2, y + t / 2], [x1 + t / 2, y + t / 2]];
  const vt = (x, y1, y2) => [[x, y1], [x + t / 2, y1 + t / 2], [x + t / 2, y2 - t / 2], [x, y2], [x - t / 2, y2 - t / 2], [x - t / 2, y1 + t / 2]];
  const shapes = [hz(38, 162, 30), vt(170, 38, 152), vt(170, 168, 282), hz(38, 162, 290), vt(30, 168, 282), vt(30, 38, 152), hz(38, 162, 160)];
  const segs = shapes.map((pts) => svg('polygon', { points: pts.map((p) => p.join(',')).join(' '), class: 'seg' }, el));
  return (vals) => segs.forEach((s, i) => { s.style.fill = mix(SEGOFF, C1, vals ? vals[i] : 0); });
}
function segFigure(parent, caption) {
  const fig = html('figure', {}, parent);
  const s = svg('svg', {}, fig);
  const set = makeSegDisplay(s);
  const cap = html('figcaption', {}, fig, caption);
  return { set, cap, fig };
}

// ===========================================================================
// Skripte über den lokalen Server starten
// ===========================================================================
let serverOk = false;
async function runScript(name, out, onLine) {
  if (!serverOk) {
    out.textContent += 'Der Server läuft nicht. Starte die Website mit:  python web/server.py\n';
    return 1;
  }
  out.textContent += `$ ${({ part2: 'python part2_train.py', part3: 'python part3_tensorflow.py', build: 'c\\build.bat', part4: 'part4_txt.exe', part5: 'part5_header.exe' })[name]}\n`;
  let code = 1, buf = '';
  try {
    const res = await fetch('/api/run?name=' + name);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let k;
      while ((k = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, k).replace(/\r$/, '');
        buf = buf.slice(k + 1);
        const m = line.match(/^\[EXIT (-?\d+)\]$/);
        if (m) { code = Number(m[1]); continue; }
        out.textContent += line + '\n';
        out.scrollTop = out.scrollHeight;
        if (onLine) onLine(line);
      }
    }
  } catch (e) {
    out.textContent += 'Fehler: ' + e.message + '\n';
  }
  out.textContent += code === 0 ? '✓ fertig\n\n' : `✗ beendet mit Code ${code}\n\n`;
  out.scrollTop = out.scrollHeight;
  return code;
}
function setChip(el, cls, text) { el.className = 'chip ' + cls; el.textContent = text; }

async function fetchText(path) {
  try { const r = await fetch(path); if (!r.ok) throw 0; return (await r.text()).replace(/\r/g, ''); } catch { return null; }
}

// Python-Gewichte laden; ohne Server wird im Browser trainiert (gleicher Algorithmus)
const nets = {};
async function loadNets() {
  const txt = await fetchText('/output/xor_net.json');
  const txt2 = await fetchText('/output/7seg_net.json');
  // Skripte gehen nur, wenn die Seite über server.py (http) läuft und output/ erreichbar ist
  serverOk = location.protocol.startsWith('http') && txt !== null;
  if (txt) nets.xor = Net.fromPython(JSON.parse(txt));
  else { nets.xor = trainFallback([2, 2, 1], XOR_X, XOR_Y); }
  if (txt2) nets.seg = Net.fromPython(JSON.parse(txt2));
  else { nets.seg = trainFallback([4, 16, 7], SEG_X, SEG_Y); }
  nets.fromFiles = !!(txt && txt2);
  const st = $('#serverState');
  if (serverOk) setChip(st, 'ok', 'Server verbunden – Skripte startbar');
  else setChip(st, 'err', 'Ohne Server: python web/server.py starten');
}
function trainFallback(sizes, X, Y) {
  for (let seed = 1; seed < 50; seed++) {
    const n = new Net(sizes, seed);
    for (let e = 0; e < 20000; e++) n.epoch(X, Y, 0.1);
    if (n.correctBits(X, Y) === X.length * Y[0].length) return n;
  }
  return new Net(sizes, 1);
}

// ===========================================================================
// Kapitel-Navigation
// ===========================================================================
const chapters = [...document.querySelectorAll('.chapter')];
const inits = {};
const done = new Set();
let current = 0;
function buildToc() {
  chapters.forEach((c, i) => {
    const li = html('li', {}, $('#toc'));
    const a = html('a', { href: '#' + c.id }, li, c.dataset.title);
    a.addEventListener('click', (e) => { e.preventDefault(); show(i); $('#sidebar').classList.remove('open'); });
  });
}
function show(i) {
  current = Math.max(0, Math.min(chapters.length - 1, i));
  chapters.forEach((c, k) => c.classList.toggle('on', k === current));
  document.querySelectorAll('#toc a').forEach((a, k) => a.classList.toggle('on', k === current));
  $('#pageInfo').textContent = `${current + 1} / ${chapters.length} · ${chapters[current].dataset.title}`;
  $('#prevBtn').disabled = current === 0;
  $('#nextBtn').disabled = current === chapters.length - 1;
  history.replaceState(null, '', '#' + chapters[current].id);
  window.scrollTo({ top: 0 });
  const id = chapters[current].id;
  if (!done.has(id) && inits[id]) { done.add(id); inits[id](); }
}
$('#prevBtn').onclick = () => show(current - 1);
$('#nextBtn').onclick = () => show(current + 1);
$('#menuBtn').onclick = () => $('#sidebar').classList.toggle('open');
document.addEventListener('keydown', (e) => {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
  if (e.key === 'ArrowRight') show(current + 1);
  if (e.key === 'ArrowLeft') show(current - 1);
});

// ===========================================================================
// Kapitel: Überblick
// ===========================================================================
inits['k-ueberblick'] = () => {
  const el = $('#pipeSvg');
  const defs = svg('defs', {}, el);
  const mk = svg('marker', { id: 'arr', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto' }, defs);
  svg('path', { d: 'M0,0 L10,5 L0,10 z', fill: MUTED }, mk);
  const B = [
    { x: 20, y: 30, t: 'Python-Netz', s: 'Teil 1 · nn.py' },
    { x: 210, y: 30, t: 'Training', s: 'Teil 2 · XOR, 7-Seg' },
    { x: 400, y: 30, t: 'Speichern', s: '.json · .txt · .h' },
    { x: 590, y: 30, t: 'C-Programm', s: 'Teil 4 + 5' },
    { x: 210, y: 150, t: 'TensorFlow', s: 'Teil 3 · Vergleich' },
  ];
  const boxes = B.map((b) => {
    const g = svg('g', { class: 'box' }, el);
    svg('rect', { x: b.x, y: b.y, width: 160, height: 66, rx: 10 }, g);
    svg('text', { x: b.x + 80, y: b.y + 24 }, g).textContent = b.t;
    svg('text', { x: b.x + 80, y: b.y + 46, class: 's' }, g).textContent = b.s;
    return g;
  });
  const links = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5]];
  const path = ([a, b]) => {
    const A = B[a], Bb = B[b];
    if (Bb.y > A.y) return [A.x + 80, A.y + 66, Bb.x + 80, Bb.y];
    return [A.x + 160, A.y + 33, Bb.x, Bb.y + 33];
  };
  links.forEach((lk) => { const [x1, y1, x2, y2] = path(lk); svg('line', { x1, y1, x2: x2 - 2, y2, stroke: MUTED, 'stroke-width': 2, 'marker-end': 'url(#arr)' }, el); });
  const token = svg('circle', { r: 9, fill: C2, opacity: 0 }, el);
  const texts = [
    'Teil 1: Wir schreiben die Klassen Neuron, Layer und NeuralNetwork selbst in Python.',
    'Teil 2: Das Netz lernt XOR und den 7-Segment-Decoder (20000 Epochen).',
    'Danach speichern wir alle Gewichte: .json für Python, .txt und .h für C.',
    'Teil 4 + 5: Ein C-Programm lädt die Gewichte und rechnet nur noch vorwärts.',
    'Teil 3: Zum Vergleich trainieren wir dieselben Netze mit TensorFlow.',
  ];
  let busy = false;
  $('#pipePlay').onclick = async () => {
    if (busy) return; busy = true;
    boxes.forEach((b) => b.classList.remove('on'));
    const order = [[null, 0], [[0, 1], 1], [[1, 5], 5], [[1, 2], 2], [[2, 3], 3], [[3, 4], 4]];
    for (const [lk, bi] of order) {
      if (lk) {
        const [x1, y1, x2, y2] = path(lk);
        token.setAttribute('opacity', 1);
        await animate(700, (k) => { token.setAttribute('cx', x1 + (x2 - x1) * k); token.setAttribute('cy', y1 + (y2 - y1) * k); });
        token.setAttribute('opacity', 0);
      }
      boxes[bi].classList.add('on');
      $('#pipeText').textContent = texts[bi];
      await sleep(1600);
    }
    busy = false;
  };

  $('#runAll').onclick = async () => {
    const btn = $('#runAll'); btn.disabled = true;
    const list = [['part2', 'Teil 2'], ...($('#allPart3').checked ? [['part3', 'Teil 3']] : []), ['build', 'C bauen'], ['part4', 'Teil 4'], ['part5', 'Teil 5']];
    const steps = $('#runAllSteps'); steps.innerHTML = '';
    const chips = list.map(([, t]) => html('span', { class: 'chip' }, steps, t));
    const out = $('#runAllOut'); out.textContent = '';
    for (let k = 0; k < list.length; k++) {
      setChip(chips[k], 'run', list[k][1] + ' läuft …');
      const code = await runScript(list[k][0], out);
      setChip(chips[k], code === 0 ? 'ok' : 'err', list[k][1] + (code === 0 ? ' ✓' : ' ✗'));
      if (code !== 0) break;
    }
    btn.disabled = false;
  };
};

// ===========================================================================
// Kapitel 1: Das Neuron
// ===========================================================================
inits['k-neuron'] = () => {
  const el = $('#neuronSvg');
  const P = { x1: { x: 70, y: 90 }, x2: { x: 70, y: 210 }, n: { x: 320, y: 150 }, o: { x: 500, y: 150 }, b: { x: 320, y: 30 } };
  const e1 = svg('line', { x1: P.x1.x, y1: P.x1.y, x2: P.n.x, y2: P.n.y, class: 'edge' }, el);
  const e2 = svg('line', { x1: P.x2.x, y1: P.x2.y, x2: P.n.x, y2: P.n.y, class: 'edge' }, el);
  svg('line', { x1: P.b.x, y1: P.b.y + 14, x2: P.n.x, y2: P.n.y - 50, stroke: MUTED, 'stroke-width': 2, 'stroke-dasharray': '4 4' }, el);
  svg('line', { x1: P.n.x + 50, y1: P.n.y, x2: P.o.x - 26, y2: P.o.y, stroke: MUTED, 'stroke-width': 2 }, el);
  const l1 = svg('text', { x: 190, y: 100, class: 'lblBig', 'text-anchor': 'middle' }, el);
  const l2 = svg('text', { x: 190, y: 215, class: 'lblBig', 'text-anchor': 'middle' }, el);
  const lb = svg('text', { x: P.b.x, y: P.b.y, class: 'lblBig', 'text-anchor': 'middle' }, el);
  const mkNode = (p, r, label) => {
    const g = svg('g', { class: 'node' }, el);
    const c = svg('circle', { cx: p.x, cy: p.y, r, fill: '#fff' }, g);
    const t = svg('text', { x: p.x, y: p.y }, g);
    if (label) svg('text', { x: p.x, y: p.y + r + 16, class: 'lbl', 'text-anchor': 'middle' }, el).textContent = label;
    return { c, t };
  };
  const nx1 = mkNode(P.x1, 24, 'x₁'), nx2 = mkNode(P.x2, 24, 'x₂'), nn = mkNode(P.n, 50, 'Neuron'), no = mkNode(P.o, 26, 'Ausgang a');
  const dot = svg('circle', { r: 7, fill: C2, opacity: 0 }, el);
  const ids = ['nx1', 'nx2', 'nw1', 'nw2', 'nb'];
  const val = () => Object.fromEntries(ids.map((k) => [k, Number($('#' + k).value)]));
  let target = null;   // null | 'UND' | 'ODER' | 'XOR'
  const targets = { UND: [0, 0, 0, 1], ODER: [0, 1, 1, 1], XOR: [0, 1, 1, 0] };

  function refresh() {
    const v = val();
    ids.forEach((k) => { $('#' + k).nextElementSibling.textContent = f(v[k], k.startsWith('nx') ? 0 : 1); });
    [[e1, v.nw1], [e2, v.nw2]].forEach(([e, w]) => { e.setAttribute('stroke', w >= 0 ? C1 : C2); e.setAttribute('stroke-width', 1 + Math.min(Math.abs(w), 8) * 1.1); });
    const z = v.nx1 * v.nw1 + v.nx2 * v.nw2 + v.nb, a = sig(z);
    nx1.t.textContent = v.nx1; nx2.t.textContent = v.nx2;
    nx1.c.setAttribute('fill', mix('#fff', C1, v.nx1)); nx1.t.setAttribute('fill', v.nx1 ? '#fff' : INK);
    nx2.c.setAttribute('fill', mix('#fff', C1, v.nx2)); nx2.t.setAttribute('fill', v.nx2 ? '#fff' : INK);
    l1.textContent = `w₁ = ${f(v.nw1, 1)}`; l2.textContent = `w₂ = ${f(v.nw2, 1)}`; lb.textContent = `b = ${f(v.nb, 1)}`;
    nn.t.textContent = 'z = ' + f(z, 2); nn.t.setAttribute('fill', INK);
    no.c.setAttribute('fill', mix('#fff', C1, a)); no.t.textContent = f(a, 2); no.t.setAttribute('fill', a > .55 ? '#fff' : INK);
    $('#neuronFormula').textContent =
      `z = x₁·w₁ + x₂·w₂ + b\n  = ${v.nx1}·${f(v.nw1, 1)} + ${v.nx2}·${f(v.nw2, 1)} + ${f(v.nb, 1)} = ${f(z, 2)}\n` +
      `a = σ(z) = 1 / (1 + e^(−${f(z, 2)})) = ${f(a, 3)}`;
    // Wahrheitstabelle
    const tb = $('#neuronTable');
    tb.innerHTML = '<tr><th>x₁</th><th>x₂</th><th>z</th><th>a</th><th>Bit (a ≥ 0,5)</th>' + (target ? `<th>Soll (${target})</th>` : '') + '</tr>';
    let ok = 0;
    XOR_X.forEach(([p, q], k) => {
      const zz = p * v.nw1 + q * v.nw2 + v.nb, aa = sig(zz), bit = aa >= .5 ? 1 : 0;
      const tr = html('tr', { class: p === v.nx1 && q === v.nx2 ? 'hl' : '' }, tb);
      [p, q, f(zz, 2), f(aa, 3), bit].forEach((x) => html('td', {}, tr, String(x)));
      if (target) {
        const good = bit === targets[target][k]; if (good) ok++;
        html('td', { class: good ? 'ok' : 'bad' }, tr, `${targets[target][k]} ${good ? '✓' : '✗'}`);
      }
    });
    const msg = $('#neuronMsg');
    if (!target) msg.textContent = 'Tipp: drück „als UND“ oder „als ODER“.';
    else if (target === 'XOR') msg.innerHTML = ok === 4 ? 'Wow – das sollte nicht gehen!' :
      `<b>${ok}/4 richtig.</b> Egal wie du die Regler stellst: ein einzelnes Neuron zieht nur eine <b>gerade Grenze</b>. XOR braucht zwei – darum gibt es die versteckte Schicht.`;
    else msg.innerHTML = `<b>${ok}/4 richtig</b> – ein Neuron kann ${target}.`;
  }
  ids.forEach((k) => $('#' + k).addEventListener('input', refresh));
  const preset = (w1, w2, b, t) => { $('#nw1').value = w1; $('#nw2').value = w2; $('#nb').value = b; target = t; refresh(); };
  $('#presetAnd').onclick = () => preset(5, 5, -7.5, 'UND');
  $('#presetOr').onclick = () => preset(5, 5, -2.5, 'ODER');
  $('#presetXor').onclick = () => preset(5, 5, -2.5, 'XOR');
  refresh();

  let busy = false;
  const move = (from, to, ms = 800) => {
    dot.setAttribute('opacity', 1);
    return animate(ms, (k) => { dot.setAttribute('cx', from.x + (to.x - from.x) * k); dot.setAttribute('cy', from.y + (to.y - from.y) * k); })
      .then(() => dot.setAttribute('opacity', 0));
  };
  $('#neuronPlay').onclick = async () => {
    if (busy) return; busy = true;
    const v = val(), cap = $('#neuronStep');
    const z = v.nx1 * v.nw1 + v.nx2 * v.nw2 + v.nb, a = sig(z);
    no.c.setAttribute('fill', '#fff'); no.t.textContent = ''; nn.t.textContent = '';
    cap.innerHTML = `<b>1. Malnehmen:</b> x₁ · w₁ = ${v.nx1} · ${f(v.nw1, 1)} = <b>${f(v.nx1 * v.nw1, 2)}</b>`;
    e1.classList.add('flash'); await move(P.x1, P.n); e1.classList.remove('flash');
    nn.t.textContent = f(v.nx1 * v.nw1, 2);
    cap.innerHTML = `<b>1. Malnehmen:</b> x₂ · w₂ = ${v.nx2} · ${f(v.nw2, 1)} = <b>${f(v.nx2 * v.nw2, 2)}</b>`;
    e2.classList.add('flash'); await move(P.x2, P.n); e2.classList.remove('flash');
    nn.t.textContent = f(v.nx1 * v.nw1 + v.nx2 * v.nw2, 2);
    cap.innerHTML = `<b>2. Zusammenzählen:</b> ${f(v.nx1 * v.nw1, 2)} + ${f(v.nx2 * v.nw2, 2)} + Bias ${f(v.nb, 1)}`;
    await move({ x: P.b.x, y: P.b.y + 14 }, { x: P.n.x, y: P.n.y - 50 }, 600);
    nn.t.textContent = 'z = ' + f(z, 2);
    cap.innerHTML = `<b>2. Zusammenzählen:</b> z = <b>${f(z, 2)}</b>`;
    await sleep(900);
    nn.t.textContent = 'σ(' + f(z, 1) + ')';
    cap.innerHTML = `<b>3. Quetschen:</b> σ(${f(z, 2)}) = <b>${f(a, 3)}</b> – ${a >= .5 ? 'über 0,5, also „an“ (1)' : 'unter 0,5, also „aus“ (0)'}`;
    await sleep(700);
    await move({ x: P.n.x + 50, y: P.n.y }, { x: P.o.x - 26, y: P.o.y }, 600);
    refresh();
    busy = false;
  };
};

// ===========================================================================
// Kapitel 2: Sigmoid
// ===========================================================================
inits['k-sigmoid'] = () => {
  const cv = $('#sigCanvas'), ctx = cv.getContext('2d');
  const m = { l: 56, r: 20, t: 16, b: 40 }, W = cv.width - m.l - m.r, H = cv.height - m.t - m.b;
  const X = (z) => m.l + (z + 8) / 16 * W, Y = (y) => m.t + (1 - y) * H;
  function draw() {
    const z = Number($('#sigZ').value), a = sig(z), d = a * (1 - a);
    $('#sigZo').textContent = f(z, 2);
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.font = '13px system-ui, sans-serif';
    for (const y of [0, .25, .5, .75, 1]) {
      ctx.strokeStyle = '#efeeea'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(m.l, Y(y)); ctx.lineTo(m.l + W, Y(y)); ctx.stroke();
      ctx.fillStyle = MUTED; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(String(y), m.l - 8, Y(y));
    }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let k = -8; k <= 8; k += 2) ctx.fillText(String(k), X(k), m.t + H + 8);
    ctx.strokeStyle = LINE; ctx.beginPath(); ctx.moveTo(X(0), m.t); ctx.lineTo(X(0), m.t + H); ctx.stroke();
    ctx.fillText('z', m.l + W - 6, m.t + H + 22);
    const curve = (fn, color) => {
      ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.beginPath();
      for (let px = 0; px <= W; px++) { const zz = px / W * 16 - 8; const y = fn(zz); px ? ctx.lineTo(m.l + px, Y(y)) : ctx.moveTo(m.l + px, Y(y)); }
      ctx.stroke();
    };
    curve(sig, C1);
    if ($('#sigDeriv').checked) curve((zz) => sig(zz) * (1 - sig(zz)), C2);
    ctx.fillStyle = C1; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText('σ(z)', X(5.2), Y(sig(5.2)) + 16);
    if ($('#sigDeriv').checked) { ctx.fillStyle = C2; ctx.fillText("σ'(z)", X(2.6), Y(.2)); }
    // aktueller Punkt
    ctx.setLineDash([5, 5]); ctx.strokeStyle = MUTED; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(z), m.t + H); ctx.lineTo(X(z), Y(a)); ctx.lineTo(m.l, Y(a)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C1; ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(X(z), Y(a), 8, 0, 7); ctx.fill(); ctx.stroke();
    if ($('#sigDeriv').checked) { ctx.fillStyle = C2; ctx.beginPath(); ctx.arc(X(z), Y(d), 7, 0, 7); ctx.fill(); ctx.stroke(); }
    $('#sigText').innerHTML = `z = <b>${f(z, 2)}</b> → σ(z) = <b class="c1">${f(a, 3)}</b>` +
      ($('#sigDeriv').checked ? ` · Ableitung a·(1−a) = <b class="c2">${f(d, 3)}</b>` : '') +
      (Math.abs(z) > 5 ? ' – hier ist die Kurve fast flach: das Neuron lernt an dieser Stelle kaum noch.' : z === 0 ? ' – steilste Stelle, hier lernt das Neuron am schnellsten.' : '');
  }
  $('#sigZ').addEventListener('input', draw);
  $('#sigDeriv').addEventListener('change', draw);
  $('#sigSweep').onclick = async () => {
    await animate(5000, (k) => { $('#sigZ').value = -8 + 16 * k; draw(); });
  };
  draw();
};

// ===========================================================================
// Kapitel 3: Vorwärts rechnen
// ===========================================================================
inits['k-forward'] = () => {
  const net = nets.xor;
  const view = new NetView($('#fwdSvg'), [2, 2, 1], { inLabels: ['x₁', 'x₂'], outLabels: ['ŷ'], layerNames: ['Eingang', 'versteckt (h₁, h₂)', 'Ausgang'], padX: 110 });
  view.setWeights(net);
  const results = {};
  const table = () => {
    const tb = $('#fwdTable');
    tb.innerHTML = '<tr><th>x₁</th><th>x₂</th><th>h₁</th><th>h₂</th><th>ŷ</th><th>Bit</th><th>Soll</th><th></th></tr>';
    XOR_X.forEach((x, k) => {
      const r = results[k], tr = html('tr', {}, tb);
      [x[0], x[1]].forEach((v) => html('td', {}, tr, String(v)));
      if (!r) { html('td', { colspan: 6 }, tr, '–'); return; }
      const bit = r[2][0] >= .5 ? 1 : 0;
      [f(r[1][0], 3), f(r[1][1], 3), f(r[2][0], 3), bit, XOR_Y[k][0]].forEach((v) => html('td', {}, tr, String(v)));
      html('td', { class: bit === XOR_Y[k][0] ? 'ok' : 'bad' }, tr, bit === XOR_Y[k][0] ? '✓' : '✗');
    });
  };
  table();
  let busy = false;
  async function play(k) {
    if (busy) return; busy = true;
    const x = XOR_X[k], acts = net.forward(x), cap = $('#fwdText');
    const L = net.L;
    view.clear();
    view.setActs(acts, 0);
    cap.innerHTML = `Eingabe <b>[${x}]</b> wird an die versteckte Schicht geschickt …`;
    await view.flow(0);
    view.setActs(acts, 1);
    cap.innerHTML = `h₁ = σ(${f(L[0].W[0][0], 2)}·${x[0]} + ${f(L[0].W[0][1], 2)}·${x[1]} + ${f(L[0].b[0], 2)}) = <b>${f(acts[1][0])}</b><br>` +
      `h₂ = σ(${f(L[0].W[1][0], 2)}·${x[0]} + ${f(L[0].W[1][1], 2)}·${x[1]} + ${f(L[0].b[1], 2)}) = <b>${f(acts[1][1])}</b>`;
    await sleep(1600);
    await view.flow(1);
    view.setActs(acts, 2);
    const bit = acts[2][0] >= .5 ? 1 : 0;
    cap.innerHTML = `ŷ = σ(${f(L[1].W[0][0], 2)}·${f(acts[1][0], 2)} + ${f(L[1].W[0][1], 2)}·${f(acts[1][1], 2)} + ${f(L[1].b[0], 2)}) = <b>${f(acts[2][0])}</b> → Bit <b>${bit}</b> ` +
      (bit === XOR_Y[k][0] ? '<span class="ok">✓ richtig</span>' : '<span class="bad">✗ falsch</span>');
    results[k] = acts; table();
    busy = false;
  }
  XOR_X.forEach((x, k) => { const b = html('button', { class: 'ghost' }, $('#fwdBtns'), `[${x}]`); b.onclick = () => play(k); });
  $('#fwdAll').onclick = async () => { for (let k = 0; k < 4; k++) { await play(k); await sleep(1400); } };
  if (!nets.fromFiles) $('#fwdText').textContent = 'Hinweis: Ohne Server wurden die Gewichte im Browser trainiert.';
};

// ===========================================================================
// Kapitel 4: Gradientenabstieg
// ===========================================================================
inits['k-gradient'] = () => {
  const cv = $('#gdCanvas'), ctx = cv.getContext('2d');
  const J = (w) => 0.025 * w ** 4 - 0.4 * w * w + 0.3 * w + 2.5;   // Fehlerkurve mit 2 Tälern
  const dJ = (w) => 0.1 * w ** 3 - 0.8 * w + 0.3;
  const m = { l: 50, r: 20, t: 16, b: 40 }, W = cv.width - m.l - m.r, H = cv.height - m.t - m.b;
  const X = (w) => m.l + (w + 5.5) / 11 * W, Y = (j) => m.t + (1 - j / 6) * H;
  const starts = [4.6, -4.8, 1.2];
  let si = 0, w = starts[0], trail = [], timer = null, gone = false;
  const lr = () => Number($('#gdLr').value);
  function draw() {
    $('#gdLro').textContent = f(lr(), 2);
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.font = '13px system-ui, sans-serif';
    ctx.strokeStyle = LINE; ctx.beginPath(); ctx.moveTo(m.l, m.t + H); ctx.lineTo(m.l + W, m.t + H); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let k = -5; k <= 5; k++) ctx.fillText(String(k), X(k), m.t + H + 8);
    ctx.fillText('Gewicht w', m.l + W - 40, m.t + H + 22);
    ctx.save(); ctx.translate(16, m.t + H / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('Fehler', 0, 0); ctx.restore();
    ctx.strokeStyle = INK2; ctx.lineWidth = 2.5; ctx.beginPath();
    for (let px = 0; px <= W; px++) { const ww = px / W * 11 - 5.5; const y = Y(J(ww)); px ? ctx.lineTo(m.l + px, y) : ctx.moveTo(m.l + px, y); }
    ctx.stroke();
    ctx.fillStyle = MUTED; ctx.textAlign = 'center';
    ctx.fillText('tiefstes Tal (globales Minimum)', X(-3), Y(J(-3)) + 14);
    ctx.fillText('kleines Tal (lokales Minimum)', X(2.62), Y(J(2.62)) + 14);
    trail.forEach((p, k) => { ctx.fillStyle = mix('#ffffff', C1, .25 + .5 * k / Math.max(1, trail.length)); ctx.beginPath(); ctx.arc(X(p), Y(J(p)), 4, 0, 7); ctx.fill(); });
    if (gone) return;
    // Tangente
    const g = dJ(w), y0 = J(w);
    ctx.strokeStyle = C2; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(X(w - 1.2), Y(y0 - 1.2 * g)); ctx.lineTo(X(w + 1.2), Y(y0 + 1.2 * g)); ctx.stroke();
    // nächster Schritt als Pfeil
    const wn = w - lr() * g;
    if (Math.abs(wn - w) > 0.02 && Math.abs(wn) < 5.5) {
      ctx.strokeStyle = C1; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(X(w), Y(y0) - 22); ctx.lineTo(X(wn), Y(y0) - 22); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C1; ctx.beginPath(); const dir = Math.sign(wn - w);
      ctx.moveTo(X(wn), Y(y0) - 22); ctx.lineTo(X(wn) - dir * 9, Y(y0) - 28); ctx.lineTo(X(wn) - dir * 9, Y(y0) - 16); ctx.fill();
    }
    ctx.fillStyle = C1; ctx.strokeStyle = '#fff'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(X(w), Y(y0), 11, 0, 7); ctx.fill(); ctx.stroke();
    $('#gdText').innerHTML = `w = <b>${f(w, 2)}</b> · Fehler = ${f(y0, 3)} · <span class="c2">Steigung = ${f(g, 3)}</span><br>` +
      `w_neu = ${f(w, 2)} − ${f(lr(), 2)} · ${f(g, 3)} = <b>${f(wn, 2)}</b>` +
      (Math.abs(g) < 0.01 ? ' – Steigung ≈ 0: <b>im Tal angekommen</b>.' + (w > 0 ? ' Aber nur im kleinen Tal! Das ist ein <b>lokales Minimum</b>.' : '') : '');
  }
  async function step() {
    if (gone) return;
    const g = dJ(w), wn = w - lr() * g;
    trail.push(w);
    if (Math.abs(wn) > 5.5) {
      gone = true; stop(); draw();
      $('#gdText').innerHTML = '<b class="bad">Weg!</b> Die Lernrate war zu groß – die Kugel ist aus dem Bild gesprungen. Lernrate kleiner stellen und „Neu starten“.';
      return;
    }
    const w0 = w;
    await animate(350, (k) => { w = w0 + (wn - w0) * k; draw(); });
  }
  function stop() { clearInterval(timer); timer = null; $('#gdAuto').textContent = '▶ automatisch'; }
  $('#gdStep').onclick = step;
  $('#gdAuto').onclick = () => {
    if (timer) return stop();
    $('#gdAuto').textContent = '⏸ anhalten';
    timer = setInterval(() => { if (Math.abs(dJ(w)) < 0.005 || gone) stop(); else step(); }, 450);
  };
  $('#gdReset').onclick = () => { stop(); si = (si + 1) % starts.length; w = starts[si]; trail = []; gone = false; draw(); };
  $('#gdLr').addEventListener('input', draw);
  draw();
};

// ===========================================================================
// Kapitel 5: Backpropagation Schritt für Schritt
// ===========================================================================
inits['k-backprop'] = () => {
  const view = new NetView($('#bpSvg'), [2, 2, 1], { inLabels: ['x₁', 'x₂'], outLabels: ['ŷ'], layerNames: ['Eingang', 'versteckt', 'Ausgang'], padX: 110 });
  const x = [1, 0], y = 1, lr = 0.5;
  let seed = 3, S, step = 0;

  function compute() {
    const net = new Net([2, 2, 1], seed);
    const W1 = net.L[0].W.map((r) => r.slice()), b1 = net.L[0].b.slice(), W2 = net.L[1].W[0].slice(), b2 = net.L[1].b[0];
    const h = W1.map((w, j) => sig(w[0] * x[0] + w[1] * x[1] + b1[j]));
    const o = sig(W2[0] * h[0] + W2[1] * h[1] + b2);
    const e = o - y, d2 = e * o * (1 - o);
    const W2n = W2.map((w, i) => w - lr * d2 * h[i]), b2n = b2 - lr * d2;
    const eh = W2.map((w) => d2 * w);                      // alte Gewichte!
    const d1 = eh.map((v, j) => v * h[j] * (1 - h[j]));
    const W1n = W1.map((w, j) => w.map((wi, i) => wi - lr * d1[j] * x[i])), b1n = b1.map((b, j) => b - lr * d1[j]);
    const h2 = W1n.map((w, j) => sig(w[0] * x[0] + w[1] * x[1] + b1n[j]));
    const o2 = sig(W2n[0] * h2[0] + W2n[1] * h2[1] + b2n);
    const mk = (A1, B1, A2, B2) => { const n = new Net([2, 2, 1]); n.L = [{ W: A1, b: B1 }, { W: [A2], b: [B2] }]; return n; };
    S = { W1, b1, W2, b2, h, o, e, d2, W2n, b2n, eh, d1, W1n, b1n, h2, o2,
      netOld: mk(W1, b1, W2, b2), netMid: mk(W1, b1, W2n, b2n), netNew: mk(W1n, b1n, W2n, b2n) };
  }
  const F = (v) => f(v, 3);
  const steps = [
    { t: 'Ausgangslage', w: 'old', a: -1, body: () => `<p>Die Gewichte sind am Anfang <b>zufällig</b>. Wir wollen, dass für die Eingabe <b>[1, 0]</b> eine <b>1</b> herauskommt.</p><p>Ein Trainingsschritt hat zwei Hälften: <span class="c1">vorwärts rechnen</span> und <span class="c2">rückwärts korrigieren</span>.</p>` },
    { t: '① Vorwärts: versteckte Schicht', w: 'old', a: 1, anim: () => view.flow(0), body: () => `<div class="formula">h₁ = σ(${F(S.W1[0][0])}·1 + ${F(S.W1[0][1])}·0 + ${F(S.b1[0])}) = ${F(S.h[0])}\nh₂ = σ(${F(S.W1[1][0])}·1 + ${F(S.W1[1][1])}·0 + ${F(S.b1[1])}) = ${F(S.h[1])}</div><p>Genau wie in Kapitel 3.</p>` },
    { t: '② Vorwärts: Ausgang', w: 'old', a: 2, anim: () => view.flow(1), body: () => `<div class="formula">ŷ = σ(${F(S.W2[0])}·${F(S.h[0])} + ${F(S.W2[1])}·${F(S.h[1])} + ${F(S.b2)}) = ${F(S.o)}</div><p>Das Netz sagt <b>${F(S.o)}</b>, richtig wäre <b>1</b>.</p>` },
    { t: '③ Fehler am Ausgang', w: 'old', a: 2, ring: [[2, 0]], body: () => `<div class="formula">Fehler e = ŷ − Soll = ${F(S.o)} − 1 = ${F(S.e)}</div><p>Der Fehler ist <b>${S.e < 0 ? 'negativ' : 'positiv'}</b>: der Ausgang ist zu ${S.e < 0 ? 'klein' : 'groß'}. Im Code: <code>errors = [p - t ...]</code>.</p>` },
    { t: '④ Delta des Ausgangsneurons', w: 'old', a: 2, ring: [[2, 0]], body: () => `<div class="formula">δ = e · ŷ·(1−ŷ)\n  = ${F(S.e)} · ${F(S.o)}·${F(1 - S.o)} = ${f(S.d2, 4)}</div><p>δ sagt: „So stark und in diese Richtung muss sich z am Ausgang ändern.“ Die Sigmoid-Ableitung aus Kapitel 2 steckt hier drin.</p>` },
    { t: '⑤ Gewichte der Ausgangsschicht anpassen', w: 'mid', a: 2, flash: 1, body: () => `<p>Regel: <code>w = w − α · δ · Eingang</code> (α = ${lr})</p><div class="formula">w(h₁→ŷ): ${F(S.W2[0])} − ${lr}·${f(S.d2, 4)}·${F(S.h[0])} = ${F(S.W2n[0])}\nw(h₂→ŷ): ${F(S.W2[1])} − ${lr}·${f(S.d2, 4)}·${F(S.h[1])} = ${F(S.W2n[1])}\nBias:     ${F(S.b2)} − ${lr}·${f(S.d2, 4)} = ${F(S.b2n)}</div><p>Ein Gewicht ändert sich stärker, wenn sein Eingang groß war.</p>` },
    { t: '⑥ Fehler nach hinten schicken', w: 'mid', a: 2, anim: () => view.flow(1, { reverse: true, color: C2 }), ring: [[1, 0], [1, 1]], body: () => `<p>Jedes versteckte Neuron bekommt seinen <b>Anteil am Fehler</b> – je größer sein Gewicht zum Ausgang war, desto mehr (Kettenregel):</p><div class="formula">Fehler h₁ = δ · w(h₁→ŷ) = ${f(S.d2, 4)} · ${F(S.W2[0])} = ${f(S.eh[0], 4)}\nFehler h₂ = δ · w(h₂→ŷ) = ${f(S.d2, 4)} · ${F(S.W2[1])} = ${f(S.eh[1], 4)}</div><p><b>Wichtig:</b> mit den <b>alten</b> Gewichten rechnen – deshalb steht im Code <code>prev_errors</code> vor <code>update</code>.</p>` },
    { t: '⑦ Gewichte der versteckten Schicht anpassen', w: 'new', a: 2, flash: 0, body: () => `<div class="formula">δ₁ = ${f(S.eh[0], 4)} · ${F(S.h[0])}·${F(1 - S.h[0])} = ${f(S.d1[0], 5)}\nδ₂ = ${f(S.eh[1], 4)} · ${F(S.h[1])}·${F(1 - S.h[1])} = ${f(S.d1[1], 5)}\n\nw(x₁→h₁): ${F(S.W1[0][0])} → ${F(S.W1n[0][0])}\nw(x₁→h₂): ${F(S.W1[1][0])} → ${F(S.W1n[1][0])}\nw(x₂→h₁): ${F(S.W1[0][1])} → ${F(S.W1n[0][1])}  (x₂ = 0!)\nw(x₂→h₂): ${F(S.W1[1][1])} → ${F(S.W1n[1][1])}  (x₂ = 0!)</div><p>Die Gewichte von x₂ ändern sich <b>gar nicht</b>, weil x₂ = 0 war: <code>α · δ · 0 = 0</code>.</p>` },
    { t: '⑧ Hat es geholfen?', w: 'new', a: 2, anim: async () => { await view.flow(0); await view.flow(1); }, body: () => `<p>Mit den neuen Gewichten noch einmal vorwärts rechnen:</p><div class="formula">vorher:  ŷ = ${F(S.o)}   Fehler ${F(S.o - 1)}\nnachher: ŷ = ${F(S.o2)}   Fehler ${F(S.o2 - 1)}</div><p>Der Ausgang ist ein Stück näher an 1. <b>Ein</b> Schritt bringt wenig – darum wiederholen wir das für alle Beispiele, <b>20000 Epochen</b> lang.</p>` },
  ];
  let busy = false;
  async function render(animateIt) {
    const s = steps[step];
    const net = s.w === 'old' ? S.netOld : s.w === 'mid' ? S.netMid : S.netNew;
    view.setWeights(net);
    $('#bpTitle').textContent = s.t;
    $('#bpBody').innerHTML = s.body();
    $('#bpBar').style.width = (step / (steps.length - 1) * 100) + '%';
    $('#bpPrev').disabled = step === 0;
    $('#bpNext').disabled = step === steps.length - 1;
    view.clear();
    const acts = step === 8 ? [x, S.h2, [S.o2]] : [x, S.h, [S.o]];
    if (animateIt && s.anim) {
      busy = true;
      view.setActs(acts, s.a === 1 || step === 8 ? 0 : step === 6 ? 2 : s.a - 1);
      if (step === 8) { await view.flow(0); view.setActs(acts, 1); await view.flow(1); }
      else await s.anim();
      busy = false;
    }
    if (s.a >= 0) view.setActs(acts, s.a);
    if (step === 6) S.eh.forEach((v, i) => view.setNode(view.nodes[1][i], S.h[i], f(v, 3)));
    (s.ring || []).forEach(([l, i]) => view.ring(l, i, C2));
    if (animateIt && s.flash !== undefined) view.flash(s.flash);
  }
  $('#bpNext').onclick = () => { if (!busy && step < steps.length - 1) { step++; render(true); } };
  $('#bpPrev').onclick = () => { if (!busy && step > 0) { step--; render(false); } };
  $('#bpReset').onclick = () => { if (busy) return; seed++; compute(); step = 0; render(false); };
  compute(); render(false);
};

// ===========================================================================
// Kapitel 6: Training live (XOR)
// ===========================================================================
inits['k-train'] = () => {
  const view = new NetView($('#trNet'), [2, 2, 1], { inLabels: ['x₁', 'x₂'], outLabels: ['ŷ'], padX: 110 });
  const chart = new LossChart($('#trLoss'), { yMin: 1e-4, yMax: 1, series: [{ name: 'MSE', color: C1 }] });
  const map = $('#trMap'), mctx = map.getContext('2d');
  let seed = 1, net, epoch, running = false, last;
  const speed = () => Number($('#trSpeed').value), lr = () => Number($('#trLr').value);
  function reset() {
    net = new Net([2, 2, 1], seed); epoch = 0; chart.clear(); last = null;
    $('#trSeed').textContent = seed; $('#trEpoch').textContent = 0; $('#trMse').textContent = '–';
    $('#trText').textContent = 'Drück auf Start.';
    paint();
  }
  function paint() {
    view.setWeights(net);
    view.setActs(net.forward([1, 0]));
    const n = 34, s = map.width / n;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const o = net.predict([i / (n - 1), 1 - j / (n - 1)])[0];
      mctx.fillStyle = mix('#ffffff', C1, o); mctx.fillRect(i * s, j * s, s + 1, s + 1);
    }
    XOR_X.forEach(([a, b], k) => {
      const px = 14 + a * (map.width - 28), py = 14 + (1 - b) * (map.height - 28);
      mctx.fillStyle = XOR_Y[k][0] ? INK : '#fff'; mctx.strokeStyle = XOR_Y[k][0] ? '#fff' : INK; mctx.lineWidth = 2.5;
      mctx.beginPath(); mctx.arc(px, py, 10, 0, 7); mctx.fill(); mctx.stroke();
    });
    chart.draw();
  }
  function frame() {
    if (!running) return;
    for (let k = 0; k < speed() && epoch < 20000; k++) {
      last = net.epoch(XOR_X, XOR_Y, lr());
      epoch++;
      if (epoch % 20 === 0 || epoch < 200) chart.push(0, epoch, last);
    }
    $('#trEpoch').textContent = epoch; $('#trMse').textContent = f(last, 5);
    const bits = net.correctBits(XOR_X, XOR_Y);
    if (epoch >= 20000) {
      running = false; $('#trStart').textContent = '▶ Start';
      $('#trText').innerHTML = bits === 4 && last < 0.05 ?
        `<span class="ok">Gelernt!</span> Alle 4 Fälle richtig, MSE ${f(last, 5)}.` :
        `<span class="bad">Hängen geblieben</span> (${bits}/4 richtig): <b>lokales Minimum</b>. Die Startgewichte waren ungünstig – genau wie bei TensorFlow in Teil 3. Probier „Neue Startgewichte“.`;
    } else {
      $('#trText').innerHTML = last > 0.2 ? `Plateau: der Fehler bleibt bei ca. 0,25 – das Netz „sucht“ noch. ${bits}/4 richtig.` :
        `Der Fehler fällt – das Netz hat das Muster gefunden. ${bits}/4 richtig.`;
    }
    paint();
    if (running) requestAnimationFrame(frame);
  }
  $('#trStart').onclick = () => {
    if (epoch >= 20000) reset();
    running = !running;
    $('#trStart').textContent = running ? '⏸ Pause' : '▶ Weiter';
    if (running) requestAnimationFrame(frame);
  };
  $('#trReset').onclick = () => { running = false; $('#trStart').textContent = '▶ Start'; seed++; reset(); };
  const out = () => { $('#trSpeedo').textContent = speed(); $('#trLro').textContent = f(lr(), 2); };
  $('#trSpeed').addEventListener('input', out); $('#trLr').addEventListener('input', out); out();
  reset();
};

// ===========================================================================
// Kapitel 7: 7-Segment-Decoder
// ===========================================================================
inits['k-7seg'] = () => {
  let net = nets.seg;
  const view = new NetView($('#segNet'), [4, 16, 7], {
    inLabels: ['Bit 8', 'Bit 4', 'Bit 2', 'Bit 1'], outLabels: SEGNAMES, layerNames: ['4 Bits', '16 versteckte', '7 Segmente'],
    r: 11, padX: 90, padY: 30, maxGap: 70, wScale: .35,
  });
  view.setWeights(net);
  // Eingangs- und Ausgangsknoten etwas größer mit Zahl
  const setDisp = makeSegDisplay($('#segDisp'));
  setDisp(null);
  const bars = SEGNAMES.map((n) => { const d = html('div', {}, $('#segBars')); d.style.height = '2px'; html('span', {}, d, n); html('b', {}, d, ''); return d; });
  const showBars = (o) => bars.forEach((b, i) => { b.style.height = Math.max(2, o[i] * 100) + 'px'; b.querySelector('b').textContent = o ? f(o[i], 2) : ''; });
  let busy = false;
  const btns = [];
  async function play(d) {
    if (busy) return; busy = true;
    btns.forEach((b, k) => b.classList.toggle('sel', k === d));
    const acts = net.forward(SEG_X[d]);
    view.clear(); view.setActs(acts, 0); setDisp(null); showBars([0, 0, 0, 0, 0, 0, 0]);
    $('#segText').innerHTML = `Ziffer <b>${d}</b> = binär <b>${SEG_X[d].join('')}</b>`;
    await view.flow(0, { ms: 700 });
    view.setActs(acts, 1);
    await sleep(300);
    await view.flow(1, { ms: 700 });
    view.setActs(acts, 2);
    const o = acts[2];
    setDisp(o); showBars(o);
    const bits = o.map((v) => (v >= .5 ? 1 : 0)), ok = bits.every((b, i) => b === SEG_Y[d][i]);
    $('#segText').innerHTML = `Ziffer <b>${d}</b> = <b>${SEG_X[d].join('')}</b> → Segmente <b>${bits.join('')}</b> ` +
      (ok ? '<span class="ok">✓ alle 7 richtig</span>' : '<span class="bad">✗ Fehler</span>');
    busy = false;
  }
  for (let d = 0; d < 10; d++) { const b = html('button', { class: 'ghost' }, $('#segBtns'), String(d)); b.onclick = () => play(d); btns.push(b); }
  $('#segAll').onclick = async () => { for (let d = 0; d < 10; d++) { await play(d); await sleep(900); } };

  // Gitter mit allen Ziffern + Training im Browser
  const grid = SEG_X.map((_, d) => segFigure($('#segGrid'), String(d)));
  function paintGrid() {
    let correct = 0;
    grid.forEach((g, d) => {
      const o = net.predict(SEG_X[d]);
      g.set(o);
      const ok = o.every((v, i) => (v >= .5 ? 1 : 0) === SEG_Y[d][i]);
      o.forEach((v, i) => { if ((v >= .5 ? 1 : 0) === SEG_Y[d][i]) correct++; });
      g.cap.innerHTML = `${d} <span class="${ok ? 'ok' : 'bad'}">${ok ? '✓' : '✗'}</span>`;
    });
    $('#segBits').textContent = correct + '/70';
  }
  paintGrid();
  let training = false, seed = 7;
  $('#segTrain').onclick = () => {
    if (training) { training = false; $('#segTrain').textContent = '▶ Training starten'; return; }
    net = new Net([4, 16, 7], seed++);
    let ep = 0, mse = 0;
    training = true; $('#segTrain').textContent = '⏸ anhalten';
    const frame = () => {
      if (!training) return;
      for (let k = 0; k < (ep < 300 ? 3 : 40) && ep < 20000; k++) { mse = net.epoch(SEG_X, SEG_Y, 0.1); ep++; }
      $('#segEpoch').textContent = ep; $('#segMse').textContent = f(mse, 5);
      view.setWeights(net); paintGrid();
      if (ep >= 20000) { training = false; $('#segTrain').textContent = '▶ Neu trainieren'; return; }
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  };
  $('#segUsePy').onclick = () => {
    training = false; $('#segTrain').textContent = '▶ Training starten';
    net = nets.seg; view.setWeights(net); paintGrid();
    $('#segEpoch').textContent = '20000 (Python)'; $('#segMse').textContent = '–';
  };
};

// ===========================================================================
// Kapitel 8: Export
// ===========================================================================
function genTxt(net) {
  let s = `layers ${net.L.length}\ninputs ${net.sizes[0]}\n`;
  net.L.forEach((l, li) => {
    s += `\nlayer ${li + 1} neurons ${l.W.length} activation sigmoid\n`;
    l.W.forEach((w, j) => { s += `neuron ${j + 1} bias ${l.b[j].toFixed(10)} weights ${w.map((v) => v.toFixed(10)).join(' ')}\n`; });
  });
  return s;
}
function genJson(net) {
  return JSON.stringify({ layer_sizes: net.sizes, layers: net.L.map((l) => l.W.map((w, j) => ({ activation: 'sigmoid', bias: l.b[j], weights: w }))) }, null, 2);
}
function genH(net, name = 'XOR_NET') {
  let s = `#ifndef ${name}_H\n#define ${name}_H\n\n#include "nn.h"\n\n#define ${name}_NUM_LAYERS ${net.L.length}\n\n`;
  net.L.forEach((l, li) => {
    s += `static const float ${name}_L${li + 1}_BIAS[${l.W.length}] = {${l.b.map((v) => v.toFixed(10) + 'f').join(', ')}};\n`;
    s += `static const float ${name}_L${li + 1}_WEIGHTS[${l.W.length}][${l.W[0].length}] = {\n`;
    l.W.forEach((w) => { s += `    {${w.map((v) => v.toFixed(10) + 'f').join(', ')}},\n`; });
    s += '};\n\n';
  });
  return s + `#endif\n`;
}
inits['k-export'] = async () => {
  const net = nets.xor;
  const view = new NetView($('#exNet'), [2, 2, 1], { inLabels: ['x₁', 'x₂'], outLabels: ['ŷ'], layerNames: ['Eingang', 'Schicht 1', 'Schicht 2'], padX: 110 });
  view.setWeights(net);
  const files = {
    txt: (await fetchText('/output/xor_net.txt')) || genTxt(net),
    json: (await fetchText('/output/xor_net.json')) || genJson(net),
    h: (await fetchText('/output/xor_net.h')) || genH(net),
  };
  let fmt = 'txt', sel = null;
  // Welche Zeilen gehören zu Neuron n in Schicht l (l = 1 oder 2)?
  function linesFor(text, l, n) {
    const lines = text.split('\n');
    if (fmt === 'txt') {
      const start = lines.findIndex((s) => s.startsWith(`layer ${l} `));
      const k = lines.findIndex((s, i) => i > start && s.startsWith(`neuron ${n} `));
      return k >= 0 ? [k] : [];
    }
    if (fmt === 'json') {
      const flat = (l === 1 ? 0 : net.sizes[1]) + (n - 1);   // n-tes Neuron insgesamt
      let count = -1, res = [];
      for (let i = 0; i < lines.length; i++) {
        if (lines[i].includes('"activation"')) count++;
        if (count === flat && /"(bias|activation)"|^\s*-?\d|weights/.test(lines[i])) res.push(i);
        if (count > flat) break;
      }
      return res;
    }
    const wi = lines.findIndex((s) => s.includes(`_L${l}_WEIGHTS[`));
    const bi = lines.findIndex((s) => s.includes(`_L${l}_BIAS[`));
    return [bi, wi + n].filter((i) => i >= 0);
  }
  function renderFile() {
    const text = files[fmt], pre = $('#exFile');
    const hl = sel ? linesFor(text, sel[0], sel[1]) : [];
    pre.innerHTML = '';
    text.split('\n').forEach((line, i) => {
      const sp = html('span', hl.includes(i) ? { class: 'hl' } : {}, pre, line);
      pre.appendChild(document.createTextNode('\n'));
      if (i === hl[0]) setTimeout(() => sp.scrollIntoView({ block: 'nearest' }), 0);
    });
  }
  function select(l, n) {
    view.nodes.flat().forEach((nd) => nd.g.classList.remove('sel'));
    sel = [l, n];
    view.nodes[l][n - 1].g.classList.add('sel');
    const L = net.L[l - 1];
    $('#exText').innerHTML = `Schicht ${l}, Neuron ${n}: Bias <b>${f(L.b[n - 1], 4)}</b>, Gewichte <b>${L.W[n - 1].map((v) => f(v, 4)).join(', ')}</b>`;
    renderFile();
  }
  view.nodes.forEach((layer, l) => l > 0 && layer.forEach((nd, i) => {
    nd.g.classList.add('click');
    nd.c.style.pointerEvents = 'all';
    nd.g.addEventListener('click', () => select(l, i + 1));
  }));
  document.querySelectorAll('#exTabs button').forEach((b) => b.onclick = () => {
    document.querySelectorAll('#exTabs button').forEach((x) => x.classList.toggle('on', x === b));
    fmt = b.dataset.f; renderFile();
  });
  renderFile();
  let busy = false;
  $('#exPlay').onclick = async () => {
    if (busy) return; busy = true;
    for (const [l, n] of [[1, 1], [1, 2], [2, 1]]) {
      view.flash(l - 1);
      select(l, n);
      await sleep(1500);
    }
    busy = false;
  };
};

// ===========================================================================
// Kapitel 9: Teil 2 starten (Ausgabe live auswerten)
// ===========================================================================
inits['k-part2'] = async () => {
  const chart = new LossChart($('#p2Chart'), { yMin: 1e-4, yMax: 1, series: [{ name: 'XOR', color: C1 }, { name: '7-Segment', color: C2 }] });
  let cur = -1;
  const res = {};
  function onLine(line) {
    if (line.startsWith('=== XOR')) cur = 0;
    else if (line.startsWith('=== 7-Segment')) cur = 1;
    let m = line.match(/^Epoche\s+(\d+)\s+MSE\s+([\d.]+)/);
    if (m && cur >= 0) { chart.push(cur, Number(m[1]), Number(m[2])); chart.draw(); }
    m = line.match(/^Richtige Bits: (\d+)\/(\d+)/);
    if (m && cur >= 0) res[cur] = { ...(res[cur] || {}), bits: `${m[1]}/${m[2]}` };
    m = line.match(/^Export\/Import identisch: (\w+)/);
    if (m && cur >= 0) res[cur] = { ...(res[cur] || {}), same: m[1] === 'True' };
    showRes();
  }
  function showRes() {
    const box = $('#p2Result'); box.innerHTML = '';
    ['XOR', '7-Segment'].forEach((n, i) => {
      if (!res[i]) return;
      html('p', {}, box).innerHTML = `<b>${n}:</b> richtige Bits <b>${res[i].bits ?? '…'}</b>` +
        (res[i].same !== undefined ? ` · Export → Import gleich: ${res[i].same ? '<span class="ok">ja ✓</span>' : '<span class="bad">nein</span>'}` : '');
    });
  }
  const old = await fetchText('/output/part2_ergebnis.txt');
  if (old) { old.split('\n').forEach((l) => onLine(l.replace(/\r$/, ''))); $('#p2State').textContent = 'Anzeige: letzte Ausführung'; }
  $('#p2Run').onclick = async () => {
    $('#p2Run').disabled = true; chart.clear(); cur = -1; for (const k in res) delete res[k]; showRes();
    setChip($('#p2State'), 'run', 'läuft …');
    $('#p2Out').textContent = '';
    const code = await runScript('part2', $('#p2Out'), onLine);
    setChip($('#p2State'), code === 0 ? 'ok' : 'err', code === 0 ? 'fertig ✓' : 'Fehler');
    nets.xor = Net.fromPython(JSON.parse(await fetchText('/output/xor_net.json') || genJson(nets.xor)));
    $('#p2Run').disabled = false;
  };
};

// ===========================================================================
// Kapitel 10: TensorFlow-Vergleich
// ===========================================================================
inits['k-tf'] = async () => {
  // Fallback = Werte aus der letzten Ausführung (README)
  let R = {
    XOR: { own: { mse: 0.001269, bits: '4/4', time: 0.9, conv: '10105' }, tf: { mse: 0.167848, bits: '3/4', time: 858.5, conv: 'nie' },
      rows: [[[0, 0], 0, .04, .33], [[0, 1], 1, .97, .97], [[1, 0], 1, .96, .33], [[1, 1], 0, .03, .33]] },
    SEG: { own: { mse: 0.000122, bits: '70/70', time: 20.3, conv: '1270' }, tf: { mse: 0.000705, bits: '70/70', time: 1085.3, conv: '5633' } },
  };
  const txt = await fetchText('/output/part3_ergebnis.txt');
  if (txt) {
    let sec = null; const rows = [];
    txt.split('\n').forEach((line) => {
      if (line.startsWith('=== XOR')) sec = 'XOR'; else if (line.startsWith('=== 7-Segment')) sec = 'SEG';
      const m = line.match(/^(eigenes Netz|TensorFlow)\s+MSE ([\d.]+)\s+Bits (\S+)\s+Zeit\s+([\d.]+) s\s+MSE<0.01 ab Epoche (\S+)/);
      if (m && sec) R[sec][m[1] === 'TensorFlow' ? 'tf' : 'own'] = { mse: Number(m[2]), bits: m[3], time: Number(m[4]), conv: m[5] };
      const r = sec === 'XOR' && line.match(/^\[(\d), (\d)\]\s+\[(\d)\]\s+\[([\d.]+)\]\s+\[([\d.]+)\]/);
      if (r) rows.push([[+r[1], +r[2]], +r[3], +r[4], +r[5]]);
    });
    if (rows.length === 4) R.XOR.rows = rows;
  }
  function hbars(el, key, unit, fmtv) {
    el.innerHTML = '';
    [['XOR', 'XOR'], ['SEG', '7-Segment']].forEach(([k, name]) => {
      const max = Math.max(R[k].own[key], R[k].tf[key]);
      [['own', 'eigenes Netz', 'c1'], ['tf', 'TensorFlow', 'c2']].forEach(([w, label, cls]) => {
        const r = html('div', { class: 'r' }, el);
        html('div', { class: 't' }, r, `${name} · ${label}`);
        const wrap = html('div', {}, r);
        const b = html('div', { class: 'b ' + cls }, wrap);
        html('span', {}, b, fmtv(R[k][w][key]) + unit);
        b.dataset.w = Math.max(0.3, R[k][w][key] / max * 80);
      });
      const fac = R[k].tf[key] / R[k].own[key];
      html('p', { class: 'small' }, el, `${name}: TensorFlow ${fac >= 1 ? '≈ ' + Math.round(fac) + '× so viel' : '≈ ' + f(1 / fac, 1) + '× weniger'}. Balken je Netz auf den größeren Wert skaliert.`);
    });
  }
  hbars($('#tfTime'), 'time', ' s', (v) => f(v, 1));
  hbars($('#tfMse'), 'mse', '', (v) => f(v, 6));
  const grow = () => document.querySelectorAll('#k-tf .hbars .b').forEach((b) => { b.style.width = '0'; requestAnimationFrame(() => requestAnimationFrame(() => { b.style.width = b.dataset.w + '%'; })); });
  // XOR-Vergleich
  const box = $('#tfXor'); box.innerHTML = '';
  ['Eingabe', 'Soll', 'eigenes Netz', 'TensorFlow'].forEach((h) => html('div', { class: 'h' }, box, h));
  const cells = [];
  R.XOR.rows.forEach(([x, soll, own, tf]) => {
    html('div', {}, box, `[${x}]`);
    [[soll, INK2], [own, C1], [tf, C2]].forEach(([v, col], k) => {
      const c = html('div', { class: 'cell' }, box);
      const i = html('i', {}, c); i.style.background = mix('#ffffff', col, .55);
      html('b', {}, c, k === 0 ? String(v) : f(v, 2));
      const wrong = k > 0 && (v >= .5 ? 1 : 0) !== soll;
      if (wrong) c.style.outline = `2px solid ${BAD}`;
      cells.push([i, v]);
    });
  });
  const play = () => { grow(); cells.forEach(([i]) => { i.style.width = '0'; }); requestAnimationFrame(() => requestAnimationFrame(() => cells.forEach(([i, v]) => { i.style.width = v * 100 + '%'; }))); };
  $('#tfPlay').onclick = play;
  play();
  $('#p3Run').onclick = async () => {
    if (!confirm('Teil 3 dauert ca. 30 Minuten. Wirklich starten?')) return;
    $('#p3Run').disabled = true; setChip($('#p3State'), 'run', 'läuft … (ca. 30 min)');
    const code = await runScript('part3', $('#p3Out'));
    setChip($('#p3State'), code === 0 ? 'ok' : 'err', code === 0 ? 'fertig ✓' : 'Fehler');
    $('#p3Run').disabled = false;
  };
};

// ===========================================================================
// Kapitel 11: C-Programm
// ===========================================================================
inits['k-c'] = async () => {
  const el = $('#cSvg');
  const defs = svg('defs', {}, el);
  const mk = svg('marker', { id: 'arr2', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto' }, defs);
  svg('path', { d: 'M0,0 L10,5 L0,10 z', fill: MUTED }, mk);
  const B = {
    txt: { x: 20, y: 40, w: 180, t: '7seg_net.txt', s: 'aus Python (dump_txt)' },
    h: { x: 20, y: 194, w: 180, t: '7seg_net.h', s: 'aus Python (dump_initfile)' },
    load: { x: 260, y: 40, w: 200, t: 'net_load_txt()', s: 'fscanf, Zeile für Zeile' },
    init: { x: 260, y: 194, w: 200, t: 'SEG_NET_init()', s: 'Konstanten kopieren' },
    net: { x: 520, y: 117, w: 200, t: 'Network → Layer → Neuron', s: 'Structs im Speicher' },
    pred: { x: 770, y: 117, w: 170, t: 'net_predict()', s: '7 Ausgänge → Segmente' },
  };
  const g = {};
  for (const k in B) {
    const b = B[k]; g[k] = svg('g', { class: 'box' }, el);
    svg('rect', { x: b.x, y: b.y, width: b.w, height: 66, rx: 10 }, g[k]);
    svg('text', { x: b.x + b.w / 2, y: b.y + 24 }, g[k]).textContent = b.t;
    svg('text', { x: b.x + b.w / 2, y: b.y + 46, class: 's' }, g[k]).textContent = b.s;
  }
  const mid = (k, side) => { const b = B[k]; return side === 'r' ? [b.x + b.w, b.y + 33] : [b.x, b.y + 33]; };
  const links = { a: ['txt', 'load'], b: ['h', 'init'], c: ['load', 'net'], d: ['init', 'net'], e: ['net', 'pred'] };
  for (const k in links) { const [p, q] = links[k]; const [x1, y1] = mid(p, 'r'), [x2, y2] = mid(q, 'l'); svg('line', { x1, y1, x2: x2 - 2, y2, stroke: MUTED, 'stroke-width': 2, 'marker-end': 'url(#arr2)' }, el); }
  svg('text', { x: 110, y: 160, class: 'lbl', 'text-anchor': 'middle' }, el).textContent = 'Teil 4 oben · Teil 5 unten';
  const tok = svg('circle', { r: 8, fill: C2, opacity: 0 }, el);
  const go = async (lk) => {
    const [p, q] = links[lk]; const [x1, y1] = mid(p, 'r'), [x2, y2] = mid(q, 'l');
    tok.setAttribute('opacity', 1);
    await animate(700, (k) => { tok.setAttribute('cx', x1 + (x2 - x1) * k); tok.setAttribute('cy', y1 + (y2 - y1) * k); });
    tok.setAttribute('opacity', 0);
  };
  let busy = false;
  $('#cPlay').onclick = async () => {
    if (busy) return; busy = true;
    Object.values(g).forEach((x) => x.classList.remove('on'));
    const cap = $('#cText');
    const seq = [
      ['txt', null, '<b>Teil 4:</b> Python hat die Gewichte als Text gespeichert.'],
      ['load', 'a', 'net_load_txt() liest Zeile für Zeile: <code>fscanf(f, " neuron %*d bias %f weights", ...)</code>'],
      ['net', 'c', 'Die Zahlen landen in den Structs Network → Layer → Neuron.'],
      ['pred', 'e', 'net_predict() rechnet vorwärts – nur Malnehmen, Zusammenzählen, Sigmoid. Kein Training.'],
      ['h', null, '<b>Teil 5:</b> Die Gewichte stehen schon im Header als <code>static const float</code>-Arrays.'],
      ['init', 'b', 'SEG_NET_init() reserviert Speicher und kopiert die Konstanten – keine Datei nötig.'],
      ['net', 'd', 'Ab hier ist alles gleich wie in Teil 4.'],
      ['pred', 'e', 'Gleiches Netz, gleiche Ergebnisse: 70/70 Bits richtig.'],
    ];
    for (const [box, lk, text] of seq) {
      if (box === 'h') Object.values(g).forEach((x) => x.classList.remove('on'));
      if (lk) await go(lk);
      g[box].classList.add('on');
      cap.innerHTML = text;
      await sleep(1700);
    }
    busy = false;
  };

  // Ergebnis-Anzeige aus der C-Ausgabe
  const grid = SEG_X.map((_, d) => segFigure($('#cDigits'), String(d)));
  function parseC(text) {
    text.split('\n').forEach((line) => {
      const m = line.match(/^(\d)\s+\[[\d,]+\]\s+((?:[\d.]+\s+){7})\s*([01]{7})\s+([01]{7})\s+(OK|FALSCH)/);
      if (!m) return;
      const d = +m[1], vals = m[2].trim().split(/\s+/).map(Number);
      grid[d].set(vals);
      grid[d].cap.innerHTML = `${d} <span class="${m[5] === 'OK' ? 'ok' : 'bad'}">${m[5] === 'OK' ? '✓' : '✗'}</span>`;
    });
  }
  const old = await fetchText('/output/part4_ergebnis.txt');
  if (old) parseC(old);
  const run = async (name, label) => {
    document.querySelectorAll('#k-c .row button').forEach((b) => (b.disabled = true));
    setChip($('#cState'), 'run', label + ' läuft …');
    let buf = '';
    if (name !== 'build') grid.forEach((x, d) => { x.set(null); x.cap.textContent = String(d); });
    const code = await runScript(name, $('#cOut'), (l) => { buf += l + '\n'; parseC(l); });
    setChip($('#cState'), code === 0 ? 'ok' : 'err', label + (code === 0 ? ' ✓' : ' ✗'));
    document.querySelectorAll('#k-c .row button').forEach((b) => (b.disabled = false));
  };
  $('#cBuild').onclick = () => run('build', 'C bauen');
  $('#cP4').onclick = () => run('part4', 'Teil 4');
  $('#cP5').onclick = () => run('part5', 'Teil 5');
};

// ===========================================================================
// Start
// ===========================================================================
(async () => {
  buildToc();
  await loadNets();
  const idx = chapters.findIndex((c) => '#' + c.id === location.hash);
  show(idx >= 0 ? idx : 0);
})();
