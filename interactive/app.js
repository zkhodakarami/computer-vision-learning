'use strict';

/* ============================================================
   1. Small helpers for building the page
   ============================================================ */
function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') node.className = v;
      else if (k === 'html') node.innerHTML = v;
      else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
  }
  for (const c of children) if (c !== null && c !== undefined) node.append(c);
  return node;
}
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const fmt = (v, d = 2) => Number(v).toFixed(d);
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const std = a => { const m = mean(a); return Math.sqrt(mean(a.map(v => (v - m) * (v - m)))); };
const median = a => { const s = [...a].sort((x, y) => x - y); const n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };

function makeSlider(label, min, max, step, value, onChange, show = v => fmt(v, 2)) {
  const input = el('input', { type: 'range', min, max, step, value });
  const readout = el('span', { class: 'readout' }, show(value));
  const wrap = el('label', { class: 'slider' }, el('span', { class: 'slider-label' }, label), input, readout);
  input.addEventListener('input', () => { const v = parseFloat(input.value); readout.textContent = show(v); onChange(v); });
  wrap.set = v => { input.value = v; readout.textContent = show(v); };
  wrap.get = () => parseFloat(input.value);
  wrap.input = input;
  return wrap;
}
function makeCheckbox(label, checked, onChange) {
  const input = el('input', { type: 'checkbox' });
  input.checked = checked;
  input.addEventListener('change', () => onChange(input.checked));
  const wrap = el('label', { class: 'check' }, input, el('span', null, label));
  wrap.set = v => { input.checked = v; };
  return wrap;
}
function makeSelect(label, options, value, onChange) {
  const select = el('select');
  for (const [v, text] of options) select.append(el('option', { value: v }, text));
  select.value = value;
  select.addEventListener('change', () => onChange(select.value));
  const wrap = el('label', { class: 'select' }, el('span', { class: 'slider-label' }, label), select);
  wrap.set = v => { select.value = v; };
  return wrap;
}
function button(text, onClick, cls = 'btn') { return el('button', { class: cls, onclick: onClick }, text); }
function captioned(text, ...nodes) { return el('div', { class: 'col' }, el('div', { class: 'caption' }, text), ...nodes); }
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
const C = { blue: '#2b78c2', orange: '#e8912d', red: '#d64545', green: '#3a9d5d', purple: '#8e5bd6', gray: '#8a8f98' };

/* ---------- canvas helpers ---------- */
function makeCanvas(w, h) { return el('canvas', { width: w, height: h, class: 'canvas' }); }
function clearCanvas(cv) { const ctx = cv.getContext('2d'); ctx.setLineDash([]); ctx.fillStyle = cssVar('--canvas-bg'); ctx.fillRect(0, 0, cv.width, cv.height); return ctx; }
function grayImage(arr, w, h, vmin = 0, vmax = 255) {
  const off = document.createElement('canvas'); off.width = w; off.height = h;
  const octx = off.getContext('2d'); const img = octx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const v = clamp(Math.round((arr[i] - vmin) / (vmax - vmin) * 255), 0, 255);
    img.data[4 * i] = v; img.data[4 * i + 1] = v; img.data[4 * i + 2] = v; img.data[4 * i + 3] = 255;
  }
  octx.putImageData(img, 0, 0); return off;
}
function paintGray(cv, arr, w, h, vmin = 0, vmax = 255) {
  const ctx = cv.getContext('2d'); ctx.imageSmoothingEnabled = false;
  ctx.drawImage(grayImage(arr, w, h, vmin, vmax), 0, 0, cv.width, cv.height);
}
function diverging(t) {
  t = clamp(t, -1, 1);
  const lo = [33, 102, 172], mid = [246, 246, 246], hi = [178, 24, 43];
  const a = t < 0 ? lo : hi, k = Math.abs(t);
  return [0, 1, 2].map(i => Math.round(mid[i] + (a[i] - mid[i]) * k));
}
function heatImage(arr, w, h, limit, center = 0) {
  const off = document.createElement('canvas'); off.width = w; off.height = h;
  const octx = off.getContext('2d'); const img = octx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const col = diverging((arr[i] - center) / limit);
    img.data[4 * i] = col[0]; img.data[4 * i + 1] = col[1]; img.data[4 * i + 2] = col[2]; img.data[4 * i + 3] = 255;
  }
  octx.putImageData(img, 0, 0); return off;
}
function paintHeat(cv, arr, w, h, limit, center = 0) {
  const ctx = cv.getContext('2d'); ctx.imageSmoothingEnabled = false;
  ctx.drawImage(heatImage(arr, w, h, limit, center), 0, 0, cv.width, cv.height);
}
function pointer(cv, e) {
  const rect = cv.getBoundingClientRect(); const t = e.touches ? e.touches[0] : e;
  return { x: (t.clientX - rect.left) * cv.width / rect.width, y: (t.clientY - rect.top) * cv.height / rect.height };
}
function onDrag(cv, handler) {
  let down = false;
  cv.addEventListener('mousedown', e => { down = true; handler(pointer(cv, e)); });
  window.addEventListener('mousemove', e => { if (down) handler(pointer(cv, e)); });
  window.addEventListener('mouseup', () => { down = false; });
  cv.addEventListener('touchstart', e => { handler(pointer(cv, e)); e.preventDefault(); }, { passive: false });
  cv.addEventListener('touchmove', e => { handler(pointer(cv, e)); e.preventDefault(); }, { passive: false });
}
function arrow(ctx, x0, y0, x1, y1, color, width = 2) {
  const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy); if (len < 1e-6) return;
  const head = Math.min(10, len / 2), ux = dx / len, uy = dy / len;
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width; ctx.setLineDash([]);
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - ux * head, y1 - uy * head); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - ux * head - uy * head * 0.5, y1 - uy * head + ux * head * 0.5);
  ctx.lineTo(x1 - ux * head + uy * head * 0.5, y1 - uy * head - ux * head * 0.5);
  ctx.closePath(); ctx.fill();
}
function polyline(ctx, pts, color, width = 1.5, dash = []) {
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
  ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.stroke(); ctx.setLineDash([]);
}
function dot(ctx, x, y, r, color) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); }
function text(ctx, s, x, y, color, align = 'left', size = 12, base = 'middle') {
  ctx.fillStyle = color; ctx.font = `${size}px system-ui, sans-serif`; ctx.textAlign = align; ctx.textBaseline = base; ctx.fillText(s, x, y);
}
function plane(cv, x0, x1, y0, y1, pad = 34) {
  const W = cv.width, H = cv.height;
  return {
    x0, x1, y0, y1, W, H, pad,
    X: v => pad + (v - x0) / (x1 - x0) * (W - 2 * pad),
    Y: v => H - pad - (v - y0) / (y1 - y0) * (H - 2 * pad),
    invX: px => x0 + (px - pad) / (W - 2 * pad) * (x1 - x0),
    invY: py => y0 + (H - pad - py) / (H - 2 * pad) * (y1 - y0),
  };
}
function niceStep(raw) { const p = Math.pow(10, Math.floor(Math.log10(raw))); const r = raw / p; return (r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10) * p; }
function ticks(a, b, n = 6) { const step = niceStep((b - a) / n); const out = []; for (let v = Math.ceil(a / step - 1e-9) * step; v <= b + 1e-9; v += step) out.push(Math.round(v * 1e6) / 1e6); return out; }
function tickLabel(v) { return Math.abs(v) >= 100 ? String(Math.round(v)) : String(parseFloat(v.toFixed(2))); }
function drawAxes(ctx, m, opts = {}) {
  const muted = cssVar('--text-muted');
  ctx.strokeStyle = cssVar('--grid'); ctx.lineWidth = 1; ctx.setLineDash([]);
  const xt = opts.xticks || ticks(m.x0, m.x1), yt = opts.yticks || ticks(m.y0, m.y1);
  for (const v of xt) { ctx.beginPath(); ctx.moveTo(m.X(v), m.Y(m.y0)); ctx.lineTo(m.X(v), m.Y(m.y1)); ctx.stroke(); text(ctx, tickLabel(v), m.X(v), m.Y(m.y0) + 4, muted, 'center', 11, 'top'); }
  for (const v of yt) { ctx.beginPath(); ctx.moveTo(m.X(m.x0), m.Y(v)); ctx.lineTo(m.X(m.x1), m.Y(v)); ctx.stroke(); text(ctx, tickLabel(v), m.X(m.x0) - 4, m.Y(v), muted, 'right', 11); }
  ctx.strokeStyle = muted; ctx.strokeRect(m.X(m.x0), m.Y(m.y1), m.X(m.x1) - m.X(m.x0), m.Y(m.y0) - m.Y(m.y1));
  if (opts.xlabel) text(ctx, opts.xlabel, (m.X(m.x0) + m.X(m.x1)) / 2, m.H - 3, muted, 'center', 11, 'bottom');
  if (opts.ylabel) { ctx.save(); ctx.translate(11, (m.Y(m.y0) + m.Y(m.y1)) / 2); ctx.rotate(-Math.PI / 2); text(ctx, opts.ylabel, 0, 0, muted, 'center', 11); ctx.restore(); }
}
function rng(seed) { let s = seed >>> 0; return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function gaussPair(r) { const u = Math.max(r(), 1e-12), v = r(); const m = Math.sqrt(-2 * Math.log(u)); return [m * Math.cos(2 * Math.PI * v), m * Math.sin(2 * Math.PI * v)]; }

/* ============================================================
   2. Progress, saved in this browser only
   ============================================================ */
const PROGRESS_KEY = 'cv-learning-progress';
function loadProgress() { try { return JSON.parse(localStorage.getItem(PROGRESS_KEY) || '{}'); } catch (e) { return {}; } }
function saveProgress() { try { localStorage.setItem(PROGRESS_KEY, JSON.stringify(progress)); } catch (e) { /* private window: nothing is saved, the page still works */ } }
let progress = loadProgress();
progress.visited = progress.visited || {};
progress.quiz = progress.quiz || {};
progress.tried = progress.tried || {};

/* ============================================================
   3. The lessons, concepts and quizzes
   ============================================================ */
const LESSONS = [
  { id: 'l1', title: 'Lesson 1: Images are numbers' },
  { id: 'l2', title: 'Lesson 2: Changing images' },
  { id: 'l3', title: 'Lesson 3: Math you will lean on' },
];
const CONCEPTS = [];
const QUIZZES = { l1: [], l2: [], l3: [] };
function concept(def) { CONCEPTS.push(def); }
function quiz(lesson, questions) { questions.forEach((q, i) => { q.id = `${lesson}q${i + 1}`; QUIZZES[lesson].push(q); }); }

/* ------------------------------------------------------------
   Lesson 1, concept: a picture is a grid of numbers
   ------------------------------------------------------------ */
concept({
  id: 'pixels', lesson: 'l1', title: 'A picture is a grid of numbers',
  idea: 'Move your mouse over the X-ray (or tap it on a phone). Each little square is one pixel, and each pixel is just a number from 0 (black) to 255 (white). The magnifier shows the numbers around your mouse. Switch to the tissue image to see a colour picture: there every pixel is three numbers, one each for red, green and blue.',
  tries: [
    'Find the brightest spot in the pneumonia X-ray. What number is it close to?',
    'Switch to the tissue image and show only the blue layer. Is it brighter or darker than the red layer?',
    'Turn on "show all the numbers" and find a place where the numbers change quickly from one pixel to the next. What does that place look like in the picture?',
  ],
  build(root) {
    const N = 28;
    const images = {
      pneumonia: { name: 'chest X-ray with pneumonia', gray: DATA.xray28_pneumonia },
      normal: { name: 'normal chest X-ray', gray: DATA.xray28_normal },
      tissue: { name: 'colon tissue under a microscope (' + DATA.tissue28_label + ')', rgb: DATA.tissue28 },
    };
    const st = { image: 'pneumonia', channel: 'all', numbers: false, r: 14, c: 14 };
    const main = makeCanvas(N * 12, N * 12);
    const mag = makeCanvas(7 * 46, 7 * 46);
    const info = el('div', { class: 'readout-box' });
    const channelSel = makeSelect('layer', [['all', 'all three (full colour)'], ['r', 'red only'], ['g', 'green only'], ['b', 'blue only']], 'all', v => { st.channel = v; draw(); });
    channelSel.style.display = 'none';
    const imageSel = makeSelect('image', [['pneumonia', 'chest X-ray, pneumonia'], ['normal', 'chest X-ray, normal'], ['tissue', 'colon tissue (colour)']], 'pneumonia', v => { st.image = v; channelSel.style.display = v === 'tissue' ? '' : 'none'; draw(); });
    const numbersBox = makeCheckbox('show all the numbers', false, v => { st.numbers = v; draw(); });
    root.append(el('div', { class: 'controls' }, imageSel, channelSel, numbersBox));
    root.append(el('div', { class: 'row' }, captioned('hover or tap a pixel', main), captioned('magnifier: the 7 by 7 pixels around your mouse', mag, info)));

    function pixel(r, c) {
      const im = images[st.image];
      if (im.gray) { const v = im.gray[r * N + c]; return { fill: [v, v, v], text: String(v), label: 'value ' + v }; }
      const i = (r * N + c) * 3; const R = im.rgb[i], G = im.rgb[i + 1], B = im.rgb[i + 2];
      if (st.channel === 'all') return { fill: [R, G, B], text: '', triple: [R, G, B], label: `red ${R}, green ${G}, blue ${B}` };
      const v = { r: R, g: G, b: B }[st.channel];
      const fill = st.channel === 'r' ? [v, 0, 0] : st.channel === 'g' ? [0, v, 0] : [0, 0, v];
      return { fill, text: String(v), label: `${({ r: 'red', g: 'green', b: 'blue' })[st.channel]} ${v}` };
    }
    const lum = f => 0.299 * f[0] + 0.587 * f[1] + 0.114 * f[2];
    function draw() {
      const S = st.numbers ? 20 : 12;
      if (main.width !== N * S) { main.width = N * S; main.height = N * S; }
      const ctx = main.getContext('2d');
      ctx.font = '9px ui-monospace, Menlo, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
        const p = pixel(r, c);
        ctx.fillStyle = `rgb(${p.fill[0]},${p.fill[1]},${p.fill[2]})`; ctx.fillRect(c * S, r * S, S, S);
        if (st.numbers && p.text) { ctx.fillStyle = lum(p.fill) > 128 ? '#000' : '#fff'; ctx.fillText(p.text, c * S + S / 2, r * S + S / 2); }
      }
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2; ctx.strokeRect(st.c * S + 1, st.r * S + 1, S - 2, S - 2);
      const M = 46, mctx = mag.getContext('2d');
      const r0 = clamp(st.r - 3, 0, N - 7), c0 = clamp(st.c - 3, 0, N - 7);
      mctx.textAlign = 'center'; mctx.textBaseline = 'middle';
      for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++) {
        const p = pixel(r0 + i, c0 + j);
        mctx.fillStyle = `rgb(${p.fill[0]},${p.fill[1]},${p.fill[2]})`; mctx.fillRect(j * M, i * M, M, M);
        mctx.fillStyle = lum(p.fill) > 128 ? '#000' : '#fff';
        if (p.text) { mctx.font = '13px ui-monospace, Menlo, monospace'; mctx.fillText(p.text, j * M + M / 2, i * M + M / 2); }
        else { mctx.font = '10px ui-monospace, Menlo, monospace'; mctx.fillText('R ' + p.triple[0], j * M + M / 2, i * M + 11); mctx.fillText('G ' + p.triple[1], j * M + M / 2, i * M + M / 2); mctx.fillText('B ' + p.triple[2], j * M + M / 2, i * M + M - 11); }
        if (r0 + i === st.r && c0 + j === st.c) { mctx.strokeStyle = C.orange; mctx.lineWidth = 3; mctx.strokeRect(j * M + 1.5, i * M + 1.5, M - 3, M - 3); }
      }
      const p = pixel(st.r, st.c);
      info.innerHTML = `<b>row ${st.r}, column ${st.c}</b>: ${p.label}<br>${images[st.image].name}<br>` + (images[st.image].gray ? '28 rows x 28 columns = 784 numbers in total' : '28 rows x 28 columns x 3 layers = 2352 numbers in total');
    }
    const move = e => { const q = pointer(main, e); const S = main.width / N; st.c = clamp(Math.floor(q.x / S), 0, N - 1); st.r = clamp(Math.floor(q.y / S), 0, N - 1); draw(); };
    main.addEventListener('mousemove', move); main.addEventListener('click', move);
    main.addEventListener('touchstart', e => { move(e); e.preventDefault(); }, { passive: false });
    main.addEventListener('touchmove', e => { move(e); e.preventDefault(); }, { passive: false });
    draw();
  },
});

/* ------------------------------------------------------------
   Lesson 1, concept: mixing red, green and blue
   ------------------------------------------------------------ */
concept({
  id: 'channels', lesson: 'l1', title: 'Mixing red, green and blue',
  idea: 'A colour image is three grids stacked on top of each other: one for red, one for green and one for blue. Every colour a screen can show is just three numbers between 0 and 255. Mix a colour with the sliders, then take the tissue image apart layer by layer.',
  tries: [
    'Make yellow using only two sliders.',
    'Set all three sliders to the same number. What kind of colour do you always get?',
    'Turn off the red layer of the tissue image. Why does the picture turn blue-green?',
  ],
  build(root) {
    const st = { r: 200, g: 120, b: 60, use: { r: true, g: true, b: true } };
    const swatch = el('div', { class: 'swatch' }); const label = el('div', { class: 'readout-box' });
    const sR = makeSlider('red', 0, 255, 1, st.r, v => { st.r = v; paint(); }, v => String(v));
    const sG = makeSlider('green', 0, 255, 1, st.g, v => { st.g = v; paint(); }, v => String(v));
    const sB = makeSlider('blue', 0, 255, 1, st.b, v => { st.b = v; paint(); }, v => String(v));
    const presets = [['white', 255, 255, 255], ['black', 0, 0, 0], ['pure red', 255, 0, 0], ['yellow', 255, 255, 0], ['sky blue', 135, 206, 235], ['middle grey', 128, 128, 128]];
    const btns = el('div', { class: 'btn-row' }, ...presets.map(([n, r, g, b]) => button(n, () => { st.r = r; st.g = g; st.b = b; sR.set(r); sG.set(g); sB.set(b); paint(); }, 'btn btn-small')));
    function paint() {
      swatch.style.background = `rgb(${st.r},${st.g},${st.b})`;
      label.innerHTML = `<b>rgb(${st.r}, ${st.g}, ${st.b})</b><br>as fractions of 255: (${fmt(st.r / 255)}, ${fmt(st.g / 255)}, ${fmt(st.b / 255)})`;
    }
    const N = 28, S = 6; const full = makeCanvas(N * S, N * S); const layers = [0, 1, 2].map(() => makeCanvas(N * S, N * S));
    const names = ['red', 'green', 'blue'];
    const checks = ['r', 'g', 'b'].map((k, i) => makeCheckbox('use the ' + names[i] + ' layer', true, v => { st.use[k] = v; paintTissue(); }));
    function paintTissue() {
      const ctx = full.getContext('2d'); const lctx = layers.map(c => c.getContext('2d')); const rgb = DATA.tissue28;
      for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
        const i = (r * N + c) * 3; const v = [rgb[i], rgb[i + 1], rgb[i + 2]];
        const used = [st.use.r ? v[0] : 0, st.use.g ? v[1] : 0, st.use.b ? v[2] : 0];
        ctx.fillStyle = `rgb(${used.join(',')})`; ctx.fillRect(c * S, r * S, S, S);
        lctx.forEach((lc, k) => { const col = [0, 0, 0]; col[k] = v[k]; lc.fillStyle = `rgb(${col.join(',')})`; lc.fillRect(c * S, r * S, S, S); });
      }
    }
    root.append(el('div', { class: 'row' }, el('div', { class: 'col grow' }, sR, sG, sB, btns), el('div', { class: 'col' }, swatch, label)));
    root.append(el('h3', null, 'Take a real colour image apart'));
    root.append(el('div', { class: 'controls' }, ...checks));
    root.append(el('div', { class: 'row' }, captioned('red layer', layers[0]), captioned('green layer', layers[1]), captioned('blue layer', layers[2]), captioned('the layers you ticked, put together', full)));
    paint(); paintTissue();
  },
});

/* ------------------------------------------------------------
   Lesson 2, concept: changing an image means changing its numbers
   ------------------------------------------------------------ */
function boxBlur(img, W, r) {
  const tmp = new Float64Array(W * W), out = new Float64Array(W * W);
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) { let s = 0, n = 0; for (let k = -r; k <= r; k++) { const xx = x + k; if (xx >= 0 && xx < W) { s += img[y * W + xx]; n++; } } tmp[y * W + x] = s / n; }
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) { let s = 0, n = 0; for (let k = -r; k <= r; k++) { const yy = y + k; if (yy >= 0 && yy < W) { s += tmp[yy * W + x]; n++; } } out[y * W + x] = s / n; }
  return out;
}
function sobel(img, W) {
  const out = new Float64Array(W * W); let mx = 1e-9;
  const at = (x, y) => img[clamp(y, 0, W - 1) * W + clamp(x, 0, W - 1)];
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
    const gx = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1));
    const gy = (at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1));
    out[y * W + x] = Math.hypot(gx, gy); if (out[y * W + x] > mx) mx = out[y * W + x];
  }
  return out.map(v => v / mx * 255);
}
function pipeline(src, W, p) {
  let img = Float64Array.from(src);
  if (p.flip) { const o = new Float64Array(W * W); for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) o[y * W + x] = img[y * W + (W - 1 - x)]; img = o; }
  if (p.angle !== 0) {
    const o = new Float64Array(W * W), th = p.angle * Math.PI / 180, cs = Math.cos(th), sn = Math.sin(th), c = (W - 1) / 2;
    for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
      const dx = x - c, dy = y - c; const sx = Math.round(c + dx * cs - dy * sn), sy = Math.round(c + dx * sn + dy * cs);
      o[y * W + x] = (sx >= 0 && sx < W && sy >= 0 && sy < W) ? img[sy * W + sx] : 0;
    }
    img = o;
  }
  if (p.brightness !== 0 || p.contrast !== 1) img = img.map(v => clamp((v - 128) * p.contrast + 128 + p.brightness, 0, 255));
  if (p.blur > 0) img = boxBlur(img, W, p.blur);
  if (p.edges) img = sobel(img, W);
  if (p.useThreshold) img = img.map(v => (v >= p.threshold ? 255 : 0));
  return img;
}
concept({
  id: 'ops', lesson: 'l2', title: 'Changing an image means changing its numbers',
  idea: 'Every control below is arithmetic on the grid of numbers. Brightness adds a constant to every pixel. Contrast multiplies each pixel\'s distance from middle grey. Blur replaces each pixel with the average of its neighbours. Edges light up where neighbours differ a lot. Watch the histogram (how many pixels have each brightness) move as you play.',
  tries: [
    'Push brightness up until the histogram piles up against 255. Those pixels are now "clipped": the differences between them are gone for good, even if you bring brightness back down.',
    'Turn on edges with blur at 0, then set blur to 3. Which gives cleaner edges, and why?',
    'Turn on threshold and find the value that separates the dark lungs from everything else.',
    'Set contrast to 0.2. Look at the table: what happened to the spread (std)?',
  ],
  build(root) {
    const W = 64, S = 5; const src = Float64Array.from(DATA.xray64);
    const defaults = { flip: false, angle: 0, brightness: 0, contrast: 1, blur: 0, edges: false, useThreshold: false, threshold: 128 };
    const p = { ...defaults };
    const orig = makeCanvas(W * S, W * S), out = makeCanvas(W * S, W * S), hist = makeCanvas(W * S, 150);
    const stats = el('div', { class: 'readout-box' });
    const sl = {
      brightness: makeSlider('brightness (add)', -120, 120, 1, 0, v => { p.brightness = v; draw(); }, v => (v > 0 ? '+' : '') + v),
      contrast: makeSlider('contrast (multiply)', 0.2, 3, 0.05, 1, v => { p.contrast = v; draw(); }, v => 'x ' + fmt(v, 2)),
      blur: makeSlider('blur radius', 0, 4, 1, 0, v => { p.blur = v; draw(); }, v => (v === 0 ? 'off' : v + ' px')),
      angle: makeSlider('rotate', -45, 45, 1, 0, v => { p.angle = v; draw(); }, v => v + ' deg'),
      threshold: makeSlider('threshold value', 0, 255, 1, 128, v => { p.threshold = v; draw(); }, v => String(v)),
    };
    const ck = {
      flip: makeCheckbox('flip left-right', false, v => { p.flip = v; draw(); }),
      edges: makeCheckbox('find edges', false, v => { p.edges = v; draw(); }),
      useThreshold: makeCheckbox('threshold: black or white only', false, v => { p.useThreshold = v; draw(); }),
    };
    const reset = button('reset everything', () => {
      Object.assign(p, defaults);
      sl.brightness.set(0); sl.contrast.set(1); sl.blur.set(0); sl.angle.set(0); sl.threshold.set(128);
      ck.flip.set(false); ck.edges.set(false); ck.useThreshold.set(false); draw();
    });
    root.append(el('div', { class: 'row' },
      el('div', { class: 'col grow' }, sl.brightness, sl.contrast, sl.blur, sl.angle, el('div', { class: 'controls' }, ck.flip, ck.edges), el('div', { class: 'controls' }, ck.useThreshold), sl.threshold, el('div', null, reset)),
      captioned('original', orig), captioned('after your changes', out)));
    root.append(el('div', { class: 'row' }, captioned('histogram of the result: how many pixels have each brightness', hist), captioned('the same numbers, three ways of writing them (this is what "normalize" means)', stats)));
    function drawHist(img) {
      const ctx = clearCanvas(hist); const bins = new Array(32).fill(0);
      for (const v of img) bins[clamp(Math.floor(v / 8), 0, 31)]++;
      const mx = Math.max(...bins); const m = plane(hist, 0, 256, 0, mx * 1.05, 30);
      drawAxes(ctx, m, { xticks: [0, 64, 128, 192, 255], yticks: [], xlabel: 'brightness (0 = black, 255 = white)' });
      ctx.fillStyle = C.blue;
      bins.forEach((n, i) => { const x0 = m.X(i * 8), x1 = m.X((i + 1) * 8); ctx.fillRect(x0 + 1, m.Y(n), x1 - x0 - 2, m.Y(0) - m.Y(n)); });
    }
    function draw() {
      const img = pipeline(src, W, p);
      paintGray(orig, src, W, W); paintGray(out, img, W, W); drawHist(img);
      const arr = Array.from(img); const mn = Math.min(...arr), mx = Math.max(...arr), av = mean(arr), sd = std(arr);
      const sc = arr.map(v => v / 255); const stn = sd > 0 ? sc.map(v => (v - av / 255) / (sd / 255)) : sc.map(() => 0);
      const row = (name, a) => `<tr><td>${name}</td><td>${fmt(Math.min(...a), 3)}</td><td>${fmt(Math.max(...a), 3)}</td><td>${fmt(mean(a), 3)}</td><td>${fmt(std(a), 3)}</td></tr>`;
      stats.innerHTML = `<table class="stats"><tr><th></th><th>smallest</th><th>largest</th><th>average</th><th>spread (std)</th></tr>
        <tr><td>raw, 0 to 255</td><td>${fmt(mn, 0)}</td><td>${fmt(mx, 0)}</td><td>${fmt(av, 1)}</td><td>${fmt(sd, 1)}</td></tr>
        ${row('scaled: divide by 255', sc)}${row('standardized: subtract average, divide by spread', stn)}</table>
        <p class="caption">All three draw the exact same picture. Neural networks train best on the last two.</p>`;
    }
    draw();
  },
});

/* ------------------------------------------------------------
   Lesson 3, concept: what a matrix does (SVD, range, null space)
   ------------------------------------------------------------ */
function svd2(a, b, c, d) {
  const p = a * a + c * c, q = a * b + c * d, r = b * b + d * d;
  const tr = p + r, det = p * r - q * q, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l1 = tr / 2 + disc, l2 = Math.max(0, tr / 2 - disc);
  let v1 = Math.abs(q) > 1e-9 ? [l1 - r, q] : (p >= r ? [1, 0] : [0, 1]);
  const n = Math.hypot(v1[0], v1[1]) || 1; v1 = [v1[0] / n, v1[1] / n];
  const v2 = [-v1[1], v1[0]];
  const s1 = Math.sqrt(l1), s2 = Math.sqrt(l2);
  const Av = v => [a * v[0] + b * v[1], c * v[0] + d * v[1]];
  const u1 = s1 > 1e-9 ? Av(v1).map(x => x / s1) : [1, 0];
  const u2 = s2 > 1e-6 ? Av(v2).map(x => x / s2) : [-u1[1], u1[0]];
  return { s: [s1, s2], v: [v1, v2], u: [u1, u2], Av };
}
concept({
  id: 'svd', lesson: 'l3', title: 'What a matrix does: turn, stretch, turn (the SVD)',
  idea: 'A matrix is a machine: a vector goes in, a vector comes out. Feed it every point of a circle and see what shape comes out. The SVD says the answer is always an ellipse (possibly squashed flat): the matrix turns the input, stretches it along two special directions, and turns it again. The stretch amounts are the singular values. A stretch of zero means that input direction is erased: that is the null space. Everything the matrix can output is the range.',
  tries: [
    'Click "flat (rank 1)". Which input direction is erased? Where do all outputs land?',
    'Starting from the flat matrix, nudge one slider by 0.1. The null space disappears, but one singular value stays tiny. Is that matrix safe to undo?',
    'Click "pure turn". What are the singular values of a matrix that only rotates?',
    'Make a matrix where both singular values are 2 and the output is a circle. How many ways can you find?',
  ],
  build(root) {
    const st = { a: 2, b: 1, c: 0.5, d: 1.5 };
    const cvIn = makeCanvas(300, 300), cvOut = makeCanvas(300, 300); const info = el('div', { class: 'readout-box' });
    const sl = {};
    for (const k of ['a', 'b', 'c', 'd']) sl[k] = makeSlider(k, -3, 3, 0.1, st[k], v => { st[k] = v; draw(); }, v => fmt(v, 1));
    const presets = [['healthy', 2, 1, 0.5, 1.5], ['flat (rank 1)', 1, 2, 0.5, 1], ['pure stretch', 2, 0, 0, 0.5], ['pure turn', 0.7, -0.7, 0.7, 0.7], ['erase everything', 0, 0, 0, 0]];
    const btns = el('div', { class: 'btn-row' }, ...presets.map(([n, a, b, c, d]) => button(n, () => { Object.assign(st, { a, b, c, d }); sl.a.set(a); sl.b.set(b); sl.c.set(c); sl.d.set(d); draw(); }, 'btn btn-small')));
    const matrix = el('div', { class: 'matrix' });
    root.append(el('div', { class: 'row' },
      el('div', { class: 'col grow' }, el('div', { class: 'caption' }, 'the four numbers of the matrix: top row a b, bottom row c d'), el('div', { class: 'matrix-sliders' }, sl.a, sl.b, sl.c, sl.d), btns),
      el('div', { class: 'col' }, el('div', { class: 'caption' }, 'the matrix'), matrix, info)));
    root.append(el('div', { class: 'row' }, captioned('input: the unit circle, with the input directions v1 (blue) and v2 (orange)', cvIn), captioned('output: where every point of the circle lands', cvOut)));
    function draw() {
      const { a, b, c, d } = st; const r = svd2(a, b, c, d); const [s1, s2] = r.s;
      matrix.innerHTML = `<span>${fmt(a, 1)}</span><span>${fmt(b, 1)}</span><span>${fmt(c, 1)}</span><span>${fmt(d, 1)}</span>`;
      const erased2 = s2 < 1e-3, tiny2 = s2 < 0.15, erased1 = s1 < 1e-3;
      const rank = erased1 ? 0 : erased2 ? 1 : 2;
      const pts = []; for (let i = 0; i <= 200; i++) { const t = i / 200 * Math.PI * 2; pts.push([Math.cos(t), Math.sin(t)]); }
      // input side
      let ctx = clearCanvas(cvIn); let m = plane(cvIn, -3.5, 3.5, -3.5, 3.5, 28); drawAxes(ctx, m, { xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [-3, -2, -1, 0, 1, 2, 3] });
      polyline(ctx, pts.map(p => [m.X(p[0]), m.Y(p[1])]), C.gray, 1.5);
      if (rank === 1) { const v = r.v[1]; polyline(ctx, [[m.X(-4 * v[0]), m.Y(-4 * v[1])], [m.X(4 * v[0]), m.Y(4 * v[1])]], C.orange, 1.5, [6, 4]); text(ctx, 'null space (erased inputs)', m.X(-3.3), m.Y(3.1), C.orange, 'left', 11); }
      if (rank === 0) text(ctx, 'null space: the whole plane', m.X(-3.3), m.Y(3.1), C.orange, 'left', 11);
      if (rank > 0) { arrow(ctx, m.X(0), m.Y(0), m.X(r.v[0][0]), m.Y(r.v[0][1]), C.blue, 2.5); text(ctx, 'v1', m.X(r.v[0][0] * 1.25), m.Y(r.v[0][1] * 1.25), C.blue, 'center', 13); }
      if (rank > 0) { arrow(ctx, m.X(0), m.Y(0), m.X(r.v[1][0]), m.Y(r.v[1][1]), C.orange, 2.5); text(ctx, 'v2', m.X(r.v[1][0] * 1.25), m.Y(r.v[1][1] * 1.25), C.orange, 'center', 13); }
      // output side
      ctx = clearCanvas(cvOut); m = plane(cvOut, -3.5, 3.5, -3.5, 3.5, 28); drawAxes(ctx, m, { xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [-3, -2, -1, 0, 1, 2, 3] });
      if (rank === 1) { const u = r.u[0]; polyline(ctx, [[m.X(-4 * u[0]), m.Y(-4 * u[1])], [m.X(4 * u[0]), m.Y(4 * u[1])]], C.blue, 1, [3, 4]); text(ctx, 'range: every output lands on this line', m.X(-3.3), m.Y(3.1), C.blue, 'left', 11); }
      if (rank === 0) text(ctx, 'range: only the point 0', m.X(-3.3), m.Y(3.1), C.blue, 'left', 11);
      polyline(ctx, pts.map(p => { const q = r.Av(p); return [m.X(q[0]), m.Y(q[1])]; }), cssVar('--text'), 2);
      if (!erased1) { arrow(ctx, m.X(0), m.Y(0), m.X(s1 * r.u[0][0]), m.Y(s1 * r.u[0][1]), C.blue, 2.5); text(ctx, fmt(s1, 2) + ' x u1', m.X(s1 * r.u[0][0] * 1.15), m.Y(s1 * r.u[0][1] * 1.15) - 8, C.blue, 'center', 12); }
      if (!erased2) { arrow(ctx, m.X(0), m.Y(0), m.X(s2 * r.u[1][0]), m.Y(s2 * r.u[1][1]), C.orange, 2.5); text(ctx, fmt(s2, 2) + ' x u2', m.X(s2 * r.u[1][0] * 1.15), m.Y(s2 * r.u[1][1] * 1.15) - 8, C.orange, 'center', 12); }
      if (rank === 0) dot(ctx, m.X(0), m.Y(0), 5, C.red);
      let story;
      if (rank === 2 && !tiny2) story = 'Rank 2. Nothing is erased, so this matrix can be undone (it has an inverse).';
      else if (rank === 2) story = `Rank 2, but barely: the second stretch is only ${fmt(s2, 2)}. Undoing it means multiplying by 1 / ${fmt(s2, 2)} = ${fmt(1 / s2, 0)}, which blows up any small noise. Mathematically fine, practically dangerous.`;
      else if (rank === 1) story = 'Rank 1. The input direction v2 is erased completely. Two different inputs that differ only along v2 give the same output, so the matrix cannot be undone.';
      else story = 'Rank 0. Every input becomes zero. Nothing survives.';
      info.innerHTML = `<b>singular values</b>: ${fmt(s1, 2)} and ${fmt(s2, 2)}<br><b>rank</b>: ${rank}<br>${story}`;
    }
    draw();
  },
});

/* ------------------------------------------------------------
   Lesson 3, concept: rebuild an X-ray from k pieces
   ------------------------------------------------------------ */
concept({
  id: 'lowrank', lesson: 'l3', title: 'Rebuild an X-ray from its biggest pieces',
  idea: 'An image is a matrix, so it has an SVD. The SVD splits it into 64 simple "pieces" (each piece is one column times one row, scaled by a singular value). The first pieces carry the big shapes, the last ones carry fine detail and noise. Drag the slider to choose how many pieces to keep. This is the idea behind compression, and behind cheaply adapting huge models with low-rank updates.',
  tries: [
    'Find the smallest k where you can tell the lungs are lungs.',
    'Look at "piece number k on its own" as you move the slider. Early pieces are broad shapes, later ones are fine stripes. Why?',
    'At what k does the number of stored values become larger than the original 4096? Why is that not a problem for the k we actually use?',
  ],
  build(root) {
    const W = 64, S = 4; const { U, S: sv, Vt } = DATA.svd64; const src = DATA.xray64;
    let k = 5;
    const cvOrig = makeCanvas(W * S, W * S), cvRe = makeCanvas(W * S, W * S), cvPiece = makeCanvas(W * S, W * S), cvPlot = makeCanvas(520, 170);
    const info = el('div', { class: 'readout-box' });
    const slider = makeSlider('pieces kept (k)', 1, 64, 1, k, v => { k = v; draw(); }, v => String(v));
    root.append(el('div', { class: 'controls' }, slider));
    root.append(el('div', { class: 'row' }, captioned('original (64 by 64 = 4096 numbers)', cvOrig), captioned('rebuilt from the first k pieces', cvRe), captioned('piece number k on its own (red = adds brightness, blue = removes)', cvPiece)));
    root.append(el('div', { class: 'row' }, captioned('the 64 singular values (log scale). Kept ones in blue.', cvPlot), info));
    const norm = a => Math.sqrt(a.reduce((s, v) => s + v * v, 0));
    const srcNorm = norm(src);
    function rebuild(k) { const o = new Float64Array(W * W); for (let t = 0; t < k; t++) { const s = sv[t]; for (let i = 0; i < W; i++) { const u = U[i * W + t] * s; if (u === 0) continue; for (let j = 0; j < W; j++) o[i * W + j] += u * Vt[t * W + j]; } } return o; }
    function draw() {
      const re = rebuild(k); paintGray(cvOrig, src, W, W); paintGray(cvRe, re, W, W);
      const pc = new Float64Array(W * W); let lim = 1e-9; const t = k - 1;
      for (let i = 0; i < W; i++) for (let j = 0; j < W; j++) { pc[i * W + j] = sv[t] * U[i * W + t] * Vt[t * W + j]; lim = Math.max(lim, Math.abs(pc[i * W + j])); }
      paintHeat(cvPiece, pc, W, W, lim);
      const err = norm(src.map((v, i) => v - re[i])) / srcNorm;
      const stored = k * (64 + 64 + 1);
      info.innerHTML = `<b>k = ${k}</b><br>numbers stored: ${stored} instead of 4096 (${fmt(stored / 4096 * 100, 0)}%)<br>error of the rebuilt image: ${fmt(err * 100, 1)}%<br>singular value of piece ${k}: ${fmt(sv[t], 1)} (the first is ${fmt(sv[0], 0)})`;
      const ctx = clearCanvas(cvPlot); const m = plane(cvPlot, 0, 65, -2, 4.2, 34);
      drawAxes(ctx, m, { xticks: [1, 10, 20, 30, 40, 50, 64], yticks: [-2, -1, 0, 1, 2, 3, 4], xlabel: 'piece number', ylabel: 'log10 of size' });
      sv.forEach((s, i) => { const x0 = m.X(i + 0.6), x1 = m.X(i + 1.4); const y = m.Y(Math.log10(Math.max(s, 0.01))); ctx.fillStyle = i < k ? C.blue : C.gray; ctx.fillRect(x0, y, x1 - x0, m.Y(-2) - y); });
    }
    draw();
  },
});

/* ------------------------------------------------------------
   Lesson 3, concept: the main directions of a whole dataset (PCA)
   ------------------------------------------------------------ */
concept({
  id: 'pca', lesson: 'l3', title: 'Walk along the main directions of 3000 X-rays',
  idea: 'Stack 3000 X-rays as rows of one tall matrix and take its SVD. The input directions are now whole images: the main ways X-rays differ from each other. This is principal component analysis (PCA). Here you can build an X-ray yourself: start from the average X-ray and add some amount of direction 1 and direction 2. Click on the scatter plot to jump to where a real X-ray sits.',
  tries: [
    'Set both sliders to 0. What image do you get?',
    'Slide direction 1 from -8 to +8. What changes? (Hint: look at the chest size and brightness.)',
    'Click on a green dot and then on a red dot far from it. Does the built image look more "normal" or more "pneumonia"? Why is this separation only rough?',
  ],
  build(root) {
    const { mean: mu, directions, stretch, coords, coord_labels } = DATA.pca; const N = 28;
    const st = { a1: 0, a2: 0 };
    const synth = makeCanvas(N * 8, N * 8), scatter = makeCanvas(340, 340), meanCv = makeCanvas(N * 3, N * 3);
    const dirCvs = directions.map(() => makeCanvas(N * 3, N * 3));
    const s1 = makeSlider('amount of direction 1', -9, 9, 0.1, 0, v => { st.a1 = v; draw(); }, v => fmt(v, 1));
    const s2 = makeSlider('amount of direction 2', -9, 9, 0.1, 0, v => { st.a2 = v; draw(); }, v => fmt(v, 1));
    const info = el('div', { class: 'readout-box' });
    root.append(el('div', { class: 'row' }, el('div', { class: 'col grow' }, s1, s2, info), captioned('average + a1 x direction 1 + a2 x direction 2', synth), captioned('400 real X-rays placed by their first two amounts. Click to copy one.', scatter)));
    root.append(el('div', { class: 'row' }, captioned('average X-ray', meanCv), ...directions.map((d, i) => captioned(`direction ${i + 1} (stretch ${fmt(stretch[i], 0)})`, dirCvs[i]))));
    paintGray(meanCv, mu, N, N, 0, 1);
    directions.forEach((d, i) => { const lim = Math.max(...d.map(Math.abs)); paintHeat(dirCvs[i], d, N, N, lim); });
    const m = plane(scatter, -9, 9, -9, 9, 30);
    function draw() {
      const img = mu.map((v, i) => clamp(v + st.a1 * directions[0][i] + st.a2 * directions[1][i], 0, 1));
      paintGray(synth, img, N, N, 0, 1);
      const ctx = clearCanvas(scatter); drawAxes(ctx, m, { xlabel: 'amount of direction 1', ylabel: 'amount of direction 2' });
      coords.forEach((c, i) => { ctx.globalAlpha = 0.55; dot(ctx, m.X(c[0]), m.Y(c[1]), 3, coord_labels[i] ? C.red : C.green); });
      ctx.globalAlpha = 1; dot(ctx, m.X(st.a1), m.Y(st.a2), 7, C.orange); ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(m.X(st.a1), m.Y(st.a2), 7, 0, Math.PI * 2); ctx.stroke();
      text(ctx, 'green = normal, red = pneumonia', m.X(-8.6), m.Y(8.3), cssVar('--text-muted'), 'left', 11);
      info.innerHTML = `<b>a1 = ${fmt(st.a1, 1)}, a2 = ${fmt(st.a2, 1)}</b><br>Each of the 784 pixels of the built image is: average pixel + a1 x (direction 1 pixel) + a2 x (direction 2 pixel).`;
    }
    onDrag(scatter, q => { st.a1 = clamp(m.invX(q.x), -9, 9); st.a2 = clamp(m.invY(q.y), -9, 9); s1.set(fmt(st.a1, 1)); s2.set(fmt(st.a2, 1)); draw(); });
    draw();
  },
});

/* ------------------------------------------------------------
   Lesson 3, concept: the Gaussian hill and conditioning
   ------------------------------------------------------------ */
concept({
  id: 'gauss', lesson: 'l3', title: 'Slice the Gaussian hill (conditioning)',
  idea: 'Two related numbers, x1 and x2, with a Gaussian hill over them. The covariance decides the hill\'s shape: the spreads set how wide it is each way, the correlation tilts it. Now observe x1 (drag the red line). Slicing the hill there gives a new, narrower Gaussian for x2 with a moved centre. That is conditioning, and the two formulas on the right are the whole story.',
  tries: [
    'Set correlation to 0. Move the observed x1 around. Does the guess for x2 change at all?',
    'Set correlation to 0.95. How much narrower does the slice get compared with the full spread of x2?',
    'Keep correlation fixed and move the observed x1 from -3 to +3. The centre moves, but does the width of the slice change? Why not?',
    'Make the spread of x1 small and the spread of x2 large. The tilt of the ellipse changes even though correlation stayed the same. What does the ellipse axis direction depend on?',
  ],
  build(root) {
    const st = { s1: 1, s2: 1, rho: 0.8, x1: 1.5 };
    const cvHill = makeCanvas(360, 360), cvSlice = makeCanvas(360, 230); const info = el('div', { class: 'readout-box' });
    const sl = {
      s1: makeSlider('spread of x1', 0.3, 2, 0.05, 1, v => { st.s1 = v; draw(); }),
      s2: makeSlider('spread of x2', 0.3, 2, 0.05, 1, v => { st.s2 = v; draw(); }),
      rho: makeSlider('correlation', -0.95, 0.95, 0.05, 0.8, v => { st.rho = v; draw(); }),
      x1: makeSlider('observed x1', -3, 3, 0.1, 1.5, v => { st.x1 = v; draw(); }, v => fmt(v, 1)),
    };
    root.append(el('div', { class: 'row' }, el('div', { class: 'col grow' }, sl.s1, sl.s2, sl.rho, sl.x1, info), captioned('the hill from above (contours at 1, 2, 3 spreads) and 500 samples. Drag the red line.', cvHill), captioned('x2 before and after seeing x1', cvSlice)));
    const m = plane(cvHill, -4, 4, -4, 4, 30);
    function draw() {
      const { s1, s2, rho, x1 } = st;
      const cm = rho * s2 / s1 * x1, cs = s2 * Math.sqrt(1 - rho * rho);
      const toXY = (z1, z2) => [s1 * z1, s2 * (rho * z1 + Math.sqrt(1 - rho * rho) * z2)];
      let ctx = clearCanvas(cvHill); drawAxes(ctx, m, { xlabel: 'x1', ylabel: 'x2' });
      const r = rng(11); ctx.globalAlpha = 0.35;
      for (let i = 0; i < 500; i++) { const [z1, z2] = gaussPair(r); const [x, y] = toXY(z1, z2); dot(ctx, m.X(x), m.Y(y), 2.2, C.blue); }
      ctx.globalAlpha = 1;
      for (const k of [1, 2, 3]) { const pts = []; for (let i = 0; i <= 120; i++) { const t = i / 120 * Math.PI * 2; const [x, y] = toXY(k * Math.cos(t), k * Math.sin(t)); pts.push([m.X(x), m.Y(y)]); } polyline(ctx, pts, cssVar('--text'), 1); }
      polyline(ctx, [[m.X(x1), m.Y(-4)], [m.X(x1), m.Y(4)]], C.red, 2.5);
      polyline(ctx, [[m.X(x1), m.Y(cm - 2 * cs)], [m.X(x1), m.Y(cm + 2 * cs)]], C.red, 6);
      dot(ctx, m.X(x1), m.Y(cm), 7, C.red); dot(ctx, m.X(x1), m.Y(cm), 3, '#fff');
      text(ctx, 'we saw x1 = ' + fmt(x1, 1), m.X(x1) + (x1 < 1.5 ? 6 : -6), m.Y(3.7), C.red, x1 < 1.5 ? 'left' : 'right', 12);
      // slice canvas
      ctx = clearCanvas(cvSlice); const ms = plane(cvSlice, -4, 4, 0, Math.max(0.45, 1 / (cs * Math.sqrt(2 * Math.PI)) * 1.1), 34);
      drawAxes(ctx, ms, { xlabel: 'x2', ylabel: 'how likely', yticks: [] });
      const dens = (x, mu, s) => Math.exp(-0.5 * ((x - mu) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
      const curve = (mu, s) => { const p = []; for (let i = 0; i <= 160; i++) { const x = -4 + i / 20; p.push([ms.X(x), ms.Y(dens(x, mu, s))]); } return p; };
      polyline(ctx, curve(0, s2), C.blue, 2.5); polyline(ctx, curve(cm, cs), C.red, 2.5);
      text(ctx, `before: centre 0, spread ${fmt(s2)}`, ms.X(-3.9), ms.Y(ms.y1) + 14, C.blue, 'left', 12);
      text(ctx, `after seeing x1: centre ${fmt(cm)}, spread ${fmt(cs)}`, ms.X(-3.9), ms.Y(ms.y1) + 30, C.red, 'left', 12);
      info.innerHTML = `<b>new centre</b> = 0 + correlation x (spread of x2 / spread of x1) x (observed x1 - 0)<br>= ${fmt(rho)} x (${fmt(s2)} / ${fmt(s1)}) x ${fmt(x1, 1)} = <b>${fmt(cm)}</b><br><br>
        <b>new spread</b> = spread of x2 x sqrt(1 - correlation<sup>2</sup>)<br>= ${fmt(s2)} x sqrt(1 - ${fmt(rho * rho)}) = <b>${fmt(cs)}</b><br><br>
        The new spread does not depend on the observed value, only on the covariance. Knowing x1 removes ${fmt((1 - cs / s2) * 100, 0)}% of the spread of x2.`;
    }
    onDrag(cvHill, q => { st.x1 = clamp(Math.round(m.invX(q.x) * 10) / 10, -3, 3); sl.x1.set(st.x1); draw(); });
    draw();
  },
});

/* ------------------------------------------------------------
   Lesson 3, concept: the Jacobian
   ------------------------------------------------------------ */
concept({
  id: 'jacobian', lesson: 'l3', title: 'Zoom in on a bendy map (the Jacobian)',
  idea: 'The grid is pushed through a map that bends space. Around any single point, the bending is almost a flat stretch-and-turn, which is a 2 by 2 matrix: the Jacobian. Click anywhere to move the point. The blue shape is where a small square really goes; the red dashed shape is where the Jacobian says it goes. Make the square smaller and watch the two agree. The determinant tells you by what factor area changed.',
  tries: [
    'Set the square size to the largest value. Now the flat guess is visibly wrong. Shrink the square until they match. This is what "the derivative is a local thing" means.',
    'Turn on "colour by area change". Find a red region and a blue region. Click in each and compare the determinants.',
    'Set bend strength to 0. What is the Jacobian of a map that does nothing?',
    'Compare the "by hand" and "by nudging" matrices. Why do they match so closely?',
  ],
  build(root) {
    const st = { a: 0.3, px: 0.8, py: -0.6, h: 0.35, heat: false };
    const cv = makeCanvas(420, 420), zoom = makeCanvas(240, 240); const info = el('div', { class: 'readout-box' });
    const warp = (x, y) => [x + st.a * Math.sin(1.2 * x) + st.a / 2 * Math.sin(1.2 * y), y + st.a * Math.sin(1.2 * y) + st.a / 2 * Math.sin(1.2 * x)];
    const jac = (x, y) => [[1 + 1.2 * st.a * Math.cos(1.2 * x), 0.6 * st.a * Math.cos(1.2 * y)], [0.6 * st.a * Math.cos(1.2 * x), 1 + 1.2 * st.a * Math.cos(1.2 * y)]];
    const sl = {
      a: makeSlider('bend strength', 0, 0.5, 0.01, 0.3, v => { st.a = v; draw(); }),
      h: makeSlider('square size', 0.05, 0.9, 0.05, 0.35, v => { st.h = v; draw(); }),
    };
    const heat = makeCheckbox('colour by area change (determinant)', false, v => { st.heat = v; draw(); });
    root.append(el('div', { class: 'row' }, el('div', { class: 'col grow' }, sl.a, sl.h, el('div', { class: 'controls' }, heat), info), captioned('click or drag to move the point', cv), captioned('zoomed in on the point', zoom)));
    const m = plane(cv, -3.2, 3.2, -3.2, 3.2, 26);
    function square(h) { const pts = []; const corners = [[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]]; for (let k = 0; k < 4; k++) for (let i = 0; i < 25; i++) { const t = i / 25; pts.push([corners[k][0] + (corners[k + 1][0] - corners[k][0]) * t, corners[k][1] + (corners[k + 1][1] - corners[k][1]) * t]); } pts.push(corners[0]); return pts; }
    function draw() {
      const { px, py, h } = st; const J = jac(px, py); const det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
      const fp = warp(px, py);
      const ctx = clearCanvas(cv);
      if (st.heat) {
        const n = 100; const arr = new Float64Array(n * n);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const x = -3.2 + (j + 0.5) / n * 6.4, y = 3.2 - (i + 0.5) / n * 6.4; const Jg = jac(x, y); arr[i * n + j] = Jg[0][0] * Jg[1][1] - Jg[0][1] * Jg[1][0]; }
        ctx.imageSmoothingEnabled = true; ctx.drawImage(heatImage(arr, n, n, 1.0, 1), m.X(-3.2), m.Y(3.2), m.X(3.2) - m.X(-3.2), m.Y(-3.2) - m.Y(3.2));
      }
      drawAxes(ctx, m, { xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [-3, -2, -1, 0, 1, 2, 3] });
      for (let v = -2.5; v <= 2.51; v += 0.25) {
        const a = [], b = [];
        for (let i = 0; i <= 60; i++) { const t = -2.5 + i / 12; const p1 = warp(v, t), p2 = warp(t, v); a.push([m.X(p1[0]), m.Y(p1[1])]); b.push([m.X(p2[0]), m.Y(p2[1])]); }
        polyline(ctx, a, C.gray, 0.7); polyline(ctx, b, C.gray, 0.7);
      }
      const sq = square(h);
      const truePts = sq.map(p => warp(px + p[0], py + p[1]));
      const flatPts = sq.map(p => [fp[0] + J[0][0] * p[0] + J[0][1] * p[1], fp[1] + J[1][0] * p[0] + J[1][1] * p[1]]);
      polyline(ctx, truePts.map(p => [m.X(p[0]), m.Y(p[1])]), C.blue, 2.5);
      polyline(ctx, flatPts.map(p => [m.X(p[0]), m.Y(p[1])]), C.red, 2, [6, 4]);
      dot(ctx, m.X(fp[0]), m.Y(fp[1]), 5, C.red);
      text(ctx, 'blue: where the square really goes', m.X(-3.1), m.Y(-2.85), C.blue, 'left', 11);
      text(ctx, 'red dashed: the Jacobian\'s flat guess', m.X(-3.1), m.Y(-3.05), C.red, 'left', 11);
      if (st.heat) { text(ctx, 'red: area grew (det > 1)', m.X(0.4), m.Y(-2.85), C.red, 'left', 11); text(ctx, 'blue: area shrank (det < 1)', m.X(0.4), m.Y(-3.05), C.blue, 'left', 11); }
      // zoom
      const zctx = clearCanvas(zoom); const R = h * 2.0; const mz = plane(zoom, fp[0] - R, fp[0] + R, fp[1] - R, fp[1] + R, 8);
      polyline(zctx, truePts.map(p => [mz.X(p[0]), mz.Y(p[1])]), C.blue, 2.5);
      polyline(zctx, flatPts.map(p => [mz.X(p[0]), mz.Y(p[1])]), C.red, 2, [6, 4]);
      dot(zctx, mz.X(fp[0]), mz.Y(fp[1]), 5, C.red);
      // nudging
      const eps = 1e-5; const f0 = warp(px, py), fx = warp(px + eps, py), fy = warp(px, py + eps);
      const Jn = [[(fx[0] - f0[0]) / eps, (fy[0] - f0[0]) / eps], [(fx[1] - f0[1]) / eps, (fy[1] - f0[1]) / eps]];
      const mat = (M) => `<span class="matrix"><span>${fmt(M[0][0], 3)}</span><span>${fmt(M[0][1], 3)}</span><span>${fmt(M[1][0], 3)}</span><span>${fmt(M[1][1], 3)}</span></span>`;
      info.innerHTML = `<b>point</b> (${fmt(px, 2)}, ${fmt(py, 2)}) lands at (${fmt(fp[0], 2)}, ${fmt(fp[1], 2)})<br>
        <div class="matrix-row">Jacobian by hand: ${mat(J)}</div><div class="matrix-row">by nudging each input by 0.00001: ${mat(Jn)}</div>
        <b>determinant</b> = ${fmt(det, 3)}: a tiny patch here ${det > 1.02 ? 'grows to ' + fmt(det, 2) + ' times its area' : det < 0.98 ? 'shrinks to ' + fmt(det, 2) + ' of its area' : 'keeps its area'}.<br>
        Row 1 says how much the output x moves per nudge of input x, then input y. Row 2 is the same for output y.`;
    }
    onDrag(cv, q => { st.px = clamp(m.invX(q.x), -2.5, 2.5); st.py = clamp(m.invY(q.y), -2.5, 2.5); draw(); });
    draw();
  },
});

/* ------------------------------------------------------------
   Lesson 3, concept: divergence of a vector field
   ------------------------------------------------------------ */
const FIELDS = {
  source: { name: 'source: everything flows outward', v: (x, y) => [x, y], dvx: () => 1, dvy: () => 1 },
  sink: { name: 'sink: everything flows inward', v: (x, y) => [-x, -y], dvx: () => -1, dvy: () => -1 },
  swirl: { name: 'swirl: everything goes around', v: (x, y) => [-y, x], dvx: () => 0, dvy: () => 0 },
  saddle: { name: 'saddle: stretched one way, squeezed the other', v: (x, y) => [x, -y], dvx: () => 1, dvy: () => -1 },
  speedup: { name: 'speeding up to the right', v: (x, y) => [x * x / 2 + 1, 0], dvx: x => x, dvy: () => 0 },
  warp: { name: 'how far each point moved in the Jacobian playground', v: (x, y) => [0.3 * Math.sin(1.2 * x) + 0.15 * Math.sin(1.2 * y), 0.3 * Math.sin(1.2 * y) + 0.15 * Math.sin(1.2 * x)], dvx: x => 0.36 * Math.cos(1.2 * x), dvy: (x, y) => 0.36 * Math.cos(1.2 * y) },
};
concept({
  id: 'divergence', lesson: 'l3', title: 'Where does the flow spread or squeeze? (divergence)',
  idea: 'A vector field attaches an arrow to every point: think of water moving. The divergence at a point asks: is more water leaving this spot than arriving (spreading, positive), the reverse (squeezing, negative), or neither (zero)? It is computed as "how fast the x-arrow grows along x" plus "how fast the y-arrow grows along y", which is the trace of the field\'s Jacobian. Hover to read it off at any point.',
  tries: [
    'Pick the swirl. The arrows clearly move, yet divergence is 0 everywhere. Why? (Think about whether any spot gains or loses water.)',
    'Pick the saddle. Divergence is 0 but the shape is not a swirl. What are the two parts that cancel?',
    'Pick "speeding up to the right". Where is divergence negative, and what is piling up there?',
    'Pick the warp field and compare its colours with the area-change colours in the Jacobian playground. They should look alike. Why are they not exactly the same?',
  ],
  build(root) {
    let field = 'source';
    const cv = makeCanvas(420, 420); const info = el('div', { class: 'readout-box' });
    const sel = makeSelect('field', Object.entries(FIELDS).map(([k, f]) => [k, f.name]), field, v => { field = v; draw(); });
    root.append(el('div', { class: 'controls' }, sel));
    root.append(el('div', { class: 'row' }, captioned('arrows: the field. Colour: divergence (red = spreading, blue = squeezing). Hover for numbers.', cv), el('div', { class: 'col grow' }, info)));
    const m = plane(cv, -2.6, 2.6, -2.6, 2.6, 26);
    let hover = { x: 1, y: 1 };
    function draw() {
      const f = FIELDS[field]; const ctx = clearCanvas(cv);
      const n = 90; const arr = new Float64Array(n * n); let lim = 0.5;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const x = -2.6 + (j + 0.5) / n * 5.2, y = 2.6 - (i + 0.5) / n * 5.2; arr[i * n + j] = f.dvx(x, y) + f.dvy(x, y); lim = Math.max(lim, Math.abs(arr[i * n + j])); }
      ctx.imageSmoothingEnabled = true; ctx.drawImage(heatImage(arr, n, n, lim, 0), m.X(-2.6), m.Y(2.6), m.X(2.6) - m.X(-2.6), m.Y(-2.6) - m.Y(2.6));
      drawAxes(ctx, m, { xticks: [-2, -1, 0, 1, 2], yticks: [-2, -1, 0, 1, 2] });
      let vmax = 1e-9; const g = []; for (let i = 0; i < 13; i++) for (let j = 0; j < 13; j++) { const x = -2.4 + j * 0.4, y = -2.4 + i * 0.4; const v = f.v(x, y); vmax = Math.max(vmax, Math.hypot(v[0], v[1])); g.push([x, y, v]); }
      const scale = 0.36 / vmax;
      for (const [x, y, v] of g) arrow(ctx, m.X(x), m.Y(y), m.X(x + v[0] * scale), m.Y(y + v[1] * scale), '#111', 1.3);
      const v = f.v(hover.x, hover.y); dot(ctx, m.X(hover.x), m.Y(hover.y), 5, C.orange);
      const d1 = f.dvx(hover.x, hover.y), d2 = f.dvy(hover.x, hover.y);
      info.innerHTML = `<b>${f.name}</b><br><br>at (${fmt(hover.x, 1)}, ${fmt(hover.y, 1)}): arrow = (${fmt(v[0])}, ${fmt(v[1])})<br>
        how fast the x-arrow grows along x: ${fmt(d1)}<br>how fast the y-arrow grows along y: ${fmt(d2)}<br><b>divergence = ${fmt(d1)} + ${fmt(d2)} = ${fmt(d1 + d2)}</b><br><br>
        ${d1 + d2 > 0.05 ? 'More flow leaves this spot than arrives: spreading.' : d1 + d2 < -0.05 ? 'More flow arrives than leaves: squeezing (piling up).' : 'What arrives also leaves: no net spreading here.'}`;
    }
    const move = e => { const q = pointer(cv, e); hover = { x: clamp(m.invX(q.x), -2.5, 2.5), y: clamp(m.invY(q.y), -2.5, 2.5) }; draw(); };
    cv.addEventListener('mousemove', move); cv.addEventListener('click', move);
    cv.addEventListener('touchstart', e => { move(e); e.preventDefault(); }, { passive: false }); cv.addEventListener('touchmove', e => { move(e); e.preventDefault(); }, { passive: false });
    draw();
  },
});

/* ------------------------------------------------------------
   Lesson 3, concept: conditional expectation
   ------------------------------------------------------------ */
concept({
  id: 'condexp', lesson: 'l3', title: 'The best guess curve (conditional expectation)',
  idea: 'Each dot is a real X-ray. You know X (the brightness of its top half) and want to guess Y (the brightness of its bottom half). The conditional expectation E[Y | X] is simply the average Y among X-rays with a similar X: cut the X axis into bins and average inside each bin. It is the best guess if "best" means smallest squared error. Switch Y to the pneumonia label and the same curve becomes a probability.',
  tries: [
    'Set the number of bins to 3, then to 30. Few bins: smooth but crude. Many bins: detailed but noisy. This trade-off is the whole game in machine learning.',
    'In the error panel, slide your guess until squared error is smallest. Where did you land? Now do the same for absolute error.',
    'Switch Y to "pneumonia yes/no". Pick a bin and find the guess with the smallest absolute error. Why is it always 0 or 1, and why is that a bad thing for a medical model?',
  ],
  build(root) {
    const pts = DATA.ce_points;
    const st = { mode: 'bottom', nbins: 12, line: true, bin: 6, guess: null };
    const cv = makeCanvas(520, 340), cvErr = makeCanvas(520, 220); const info = el('div', { class: 'readout-box' }), info2 = el('div', { class: 'readout-box' });
    const modeSel = makeSelect('Y is', [['bottom', 'brightness of the bottom half'], ['label', 'pneumonia yes (1) or no (0)']], 'bottom', v => { st.mode = v; st.guess = null; draw(); });
    const nb = makeSlider('number of bins', 3, 30, 1, 12, v => { st.nbins = v; st.bin = Math.min(st.bin, v); binSl.input.max = v; binSl.set(st.bin); st.guess = null; draw(); }, v => String(v));
    const lineCk = makeCheckbox('show the straight line a Gaussian would predict', true, v => { st.line = v; draw(); });
    const binSl = makeSlider('look inside bin number', 1, 12, 1, 6, v => { st.bin = v; st.guess = null; draw(); }, v => String(v));
    const guessSl = makeSlider('your single guess for Y', 0, 1, 0.005, 0.5, v => { st.guess = v; drawErr(); }, v => fmt(v, 2));
    root.append(el('div', { class: 'controls' }, modeSel, nb, lineCk));
    root.append(el('div', { class: 'row' }, captioned('dots: 1500 real X-rays. Red: average Y inside each bin, with a bar for its spread.', cv), el('div', { class: 'col grow' }, info)));
    root.append(el('h3', null, 'Why is the average the best guess?'));
    root.append(el('div', { class: 'controls' }, binSl, guessSl));
    root.append(el('div', { class: 'row' }, captioned('the error you would make with one single guess for every X-ray in this bin', cvErr), el('div', { class: 'col grow' }, info2)));
    let bins = [];
    function compute() {
      const X = pts.map(p => (st.mode === 'bottom' ? p[0] : p[2])), Y = pts.map(p => (st.mode === 'bottom' ? p[1] : p[3]));
      const order = X.map((_, i) => i).sort((i, j) => X[i] - X[j]); bins = [];
      for (let b = 0; b < st.nbins; b++) {
        const idx = order.slice(Math.floor(b * order.length / st.nbins), Math.floor((b + 1) * order.length / st.nbins));
        const ys = idx.map(i => Y[i]); bins.push({ xc: mean(idx.map(i => X[i])), ym: mean(ys), ysd: std(ys), n: ys.length, ys });
      }
      return { X, Y };
    }
    function draw() {
      const { X, Y } = compute(); const ctx = clearCanvas(cv);
      const lab = st.mode === 'bottom';
      const m = plane(cv, Math.min(...X) - 5, Math.max(...X) + 5, lab ? Math.min(...Y) - 5 : -0.1, lab ? Math.max(...Y) + 5 : 1.1, 40);
      drawAxes(ctx, m, { xlabel: lab ? 'X: brightness of the top half' : 'X: brightness of the whole X-ray', ylabel: lab ? 'Y: brightness of the bottom half' : 'Y: 1 = pneumonia, 0 = normal' });
      const r = rng(3); ctx.globalAlpha = 0.25;
      X.forEach((x, i) => dot(ctx, m.X(x), m.Y(lab ? Y[i] : Y[i] + (r() - 0.5) * 0.08), 2.2, lab ? C.blue : (Y[i] ? C.red : C.green)));
      ctx.globalAlpha = 1;
      if (lab && st.line) {
        const mx = mean(X), my = mean(Y); const cov = mean(X.map((x, i) => (x - mx) * (Y[i] - my))); const slope = cov / (std(X) ** 2);
        polyline(ctx, [[m.X(m.x0), m.Y(my + slope * (m.x0 - mx))], [m.X(m.x1), m.Y(my + slope * (m.x1 - mx))]], cssVar('--text'), 1.5, [6, 4]);
      }
      if (!lab) { const p = mean(Y); polyline(ctx, [[m.X(m.x0), m.Y(p)], [m.X(m.x1), m.Y(p)]], C.gray, 1.5, [6, 4]); text(ctx, 'overall share with pneumonia: ' + fmt(p), m.X(m.x0) + 6, m.Y(p) - 10, C.gray, 'left', 11); }
      bins.forEach((b, i) => {
        const bar = lab ? b.ysd : 2 * Math.sqrt(b.ym * (1 - b.ym) / b.n);
        polyline(ctx, [[m.X(b.xc), m.Y(b.ym - bar)], [m.X(b.xc), m.Y(b.ym + bar)]], C.red, 2);
        if (i === st.bin - 1) { ctx.strokeStyle = C.orange; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(m.X(b.xc), m.Y(b.ym), 9, 0, Math.PI * 2); ctx.stroke(); }
      });
      polyline(ctx, bins.map(b => [m.X(b.xc), m.Y(b.ym)]), C.red, 2.5);
      bins.forEach(b => dot(ctx, m.X(b.xc), m.Y(b.ym), 4, C.red));
      const b = bins[st.bin - 1];
      info.innerHTML = lab
        ? `<b>${st.nbins} bins</b>, about ${b.n} X-rays each.<br>Spread of Y on its own: ${fmt(std(Y), 1)}<br>Typical spread of Y inside a bin: ${fmt(mean(bins.map(q => q.ysd)), 1)}<br><br>Knowing X removes part of the uncertainty about Y. The red curve is E[Y | X]. ${st.line ? 'The dashed line is what you would get if X and Y were jointly Gaussian: a straight line. The real curve is close to it.' : ''}`
        : `<b>${st.nbins} bins</b>, about ${b.n} X-rays each.<br>Inside each bin the average of the 0s and 1s is the <b>share with pneumonia</b>. So E[Y | X] = P(pneumonia | X).<br><br>Notice the curve bends: very dark X-rays are also mostly pneumonia. The best guess curve is whatever the data says, not a straight line. That is why flexible models are needed later.`;
      drawErr();
    }
    function drawErr() {
      const b = bins[st.bin - 1]; const lab = st.mode === 'bottom';
      const lo = lab ? Math.min(...b.ys) : 0, hi = lab ? Math.max(...b.ys) : 1;
      guessSl.input.min = lo; guessSl.input.max = hi; guessSl.input.step = (hi - lo) / 200;
      if (st.guess === null || st.guess < lo || st.guess > hi) { st.guess = (lo + hi) / 2; guessSl.set(st.guess); }
      const mse = g => mean(b.ys.map(y => (y - g) ** 2)), mae = g => mean(b.ys.map(y => Math.abs(y - g)));
      const gs = []; for (let i = 0; i <= 120; i++) gs.push(lo + (hi - lo) * i / 120);
      const e2 = gs.map(mse), e1 = gs.map(mae); const n2 = Math.max(...e2), n1 = Math.max(...e1);
      const ctx = clearCanvas(cvErr); const m = plane(cvErr, lo, hi, 0, 1.08, 40);
      drawAxes(ctx, m, { yticks: [], xlabel: 'your guess for Y', ylabel: 'error (each curve scaled to fit)' });
      polyline(ctx, gs.map((g, i) => [m.X(g), m.Y(e2[i] / n2)]), C.blue, 2.5); polyline(ctx, gs.map((g, i) => [m.X(g), m.Y(e1[i] / n1)]), C.purple, 2.5);
      const bm = mean(b.ys), bmed = median(b.ys);
      polyline(ctx, [[m.X(st.guess), m.Y(0)], [m.X(st.guess), m.Y(1.05)]], C.orange, 2);
      dot(ctx, m.X(bm), m.Y(mse(bm) / n2), 6, C.blue); dot(ctx, m.X(bmed), m.Y(mae(bmed) / n1), 6, C.purple);
      text(ctx, 'squared error (lowest at the mean)', m.X(lo) + 6, m.Y(1.0), C.blue, 'left', 11); text(ctx, 'absolute error (lowest at the median)', m.X(lo) + 6, m.Y(0.93), C.purple, 'left', 11);
      info2.innerHTML = `<b>bin ${st.bin}</b>: ${b.n} X-rays with X around ${fmt(b.xc, 0)}<br>mean of their Y: <b>${fmt(bm, lab ? 1 : 3)}</b>, median: <b>${fmt(bmed, lab ? 1 : 3)}</b><br><br>
        your guess ${fmt(st.guess, lab ? 1 : 2)}: squared error ${fmt(mse(st.guess), lab ? 1 : 3)} (best possible ${fmt(mse(bm), lab ? 1 : 3)}), absolute error ${fmt(mae(st.guess), lab ? 1 : 3)} (best possible ${fmt(mae(bmed), lab ? 1 : 3)})<br><br>
        ${lab ? 'The loss you train with decides what the model learns to output: squared error gives the mean, absolute error gives the median.' : 'With yes/no labels, squared error is lowest at the share of 1s (a probability). Absolute error is lowest at 0 or 1: the model would always answer the majority and never say "I am 60% sure".'}`;
    }
    draw();
  },
});

/* ------------------------------------------------------------
   Quizzes
   ------------------------------------------------------------ */
quiz('l1', [
  { q: 'A black and white image with shape (28, 28) holds how many numbers?', options: ['28', '784', '56', '2352'], answer: 1, why: '28 rows times 28 columns is 784 pixels, and each pixel is one number.' },
  { q: 'In an ordinary 8-bit image, which value is pure white?', options: ['0', '1', '255', '100'], answer: 2, why: 'Values run from 0 (black) to 255 (white). The 255 comes from 8 bits: 2 to the power 8 is 256 possible values.' },
  { q: 'A colour image of 28 by 28 pixels has shape...', options: ['(28, 28)', '(28, 28, 3)', '(3, 28)', '(28, 3)'], answer: 1, why: 'Three layers (red, green, blue), each 28 by 28. The shape is written (rows, columns, layers).' },
  { q: 'About 3 of every 4 training X-rays show pneumonia. Why can that be a problem?', options: ['It makes the images darker', 'A lazy model that always says "pneumonia" already scores 74%, so accuracy alone can fool you', 'The files get larger', 'It is not a problem, more pneumonia examples is always better'], answer: 1, why: 'When one class is much more common, a model can score well by ignoring the image. Lesson 9 covers the scores that catch this.' },
]);
quiz('l2', [
  { q: 'Flipping an image left to right does what to its numbers?', options: ['Adds 255 to each one', 'Multiplies each one by -1', 'Reverses the order of the numbers in every row', 'Nothing, only the picture changes'], answer: 2, why: 'The picture is the numbers. A flip reorders each row back to front.' },
  { q: 'After dividing every pixel by 255, the largest possible value is...', options: ['255', '1', '0', '128'], answer: 1, why: '255 divided by 255 is 1. This is the "scaled 0..1" version that networks like.' },
  { q: 'Which of these can NOT be undone exactly?', options: ['Flip left to right', 'Add 10 to every pixel, then subtract 10', 'Blur', 'Rotate by 180 degrees'], answer: 2, why: 'Blur averages neighbours together, so different inputs can give the same output. The lost detail cannot be recovered. The SVD playground shows why: blur has a near null space.' },
  { q: 'Why does resizing a 64 by 64 image up to 256 by 256 add no real detail?', options: ['Because 256 is too large for a screen', 'Because the new pixels are made up from the old ones, no new information arrives', 'Because the file gets bigger', 'It does add real detail'], answer: 1, why: 'The new pixels are blends (or copies) of the old ones. Information can only be lost or kept, never created, by resizing.' },
  { q: 'Pushing brightness up until many pixels hit 255 is called clipping. What is lost?', options: ['Nothing, you can bring brightness back down', 'The differences between those pixels: they all became 255', 'The colour layers', 'The image shape'], answer: 1, why: 'Once two pixels both read 255, nobody can tell they were once 230 and 250. Bringing brightness back down leaves them equal.' },
]);
quiz('l3', [
  { q: 'A matrix has singular values 5, 2 and 0. What is its rank?', options: ['3', '2', '1', '0'], answer: 1, why: 'Rank counts the non-zero singular values. Two of them are non-zero.' },
  { q: 'The null space of a matrix contains...', options: ['Every output the matrix can make', 'Every input that comes out as exactly zero', 'The biggest input direction', 'Only the zero matrix'], answer: 1, why: 'Null space is on the input side: directions the matrix erases. The range is on the output side.' },
  { q: 'Two pixels have correlation 0.95. You learn one of them is much darker than usual. Your guess for the other...', options: ['Moves darker and becomes more certain', 'Moves brighter', 'Stays the same', 'Moves darker but becomes less certain'], answer: 0, why: 'Positive correlation moves the guess the same way. Conditioning always shrinks the spread, by a factor of sqrt(1 - 0.95 squared), which is about 0.31.' },
  { q: 'In the Gaussian conditioning formula, the new covariance depends on...', options: ['The values you observed', 'Only the covariance table, never the observed values', 'The mean only', 'Nothing, it never changes'], answer: 1, why: 'The new spread is the old spread times sqrt(1 - correlation squared). The observed value moves the centre but not the width. The playground shows this: the red slice keeps its width as you drag.' },
  { q: 'A warp has Jacobian determinant 0.5 at some point. What happened to a tiny patch there?', options: ['It doubled in area', 'It shrank to half its area', 'It was flipped', 'It became a line'], answer: 1, why: 'The determinant is the area scale factor. 0.5 means half the area.' },
  { q: 'In a medical registration warp, a negative Jacobian determinant means...', options: ['The tissue grew', 'The tissue shrank', 'The tissue folded over itself, which is physically impossible and flags a bad warp', 'Nothing special'], answer: 2, why: 'A negative area factor means the patch was turned inside out. Real tissue cannot do that, so registration tools check for it.' },
  { q: 'A field of arrows has divergence zero everywhere. If it describes water flow...', options: ['The water is standing still', 'No spot is gaining or losing water: what flows in also flows out', 'The water flows in straight lines', 'The water is draining away'], answer: 1, why: 'Zero divergence means no spreading and no piling up anywhere, like a swirl. The water can still move a lot.' },
  { q: 'You train a model with absolute error instead of squared error. What is it learning to output?', options: ['The mean of Y given X', 'The median of Y given X', 'The largest Y', 'A probability'], answer: 1, why: 'Squared error is lowest at the mean, absolute error at the median. The loss decides what the model learns.' },
  { q: 'A classifier outputs 0.7 for an X-ray. If the model is honest, then among all X-rays that look like this one...', options: ['70% of the pixels are bright', 'About 70% have pneumonia', 'The model is 70% finished', 'About 30% have pneumonia'], answer: 1, why: 'A probability output is an estimate of E[Y | X], the share of pneumonia among look-alikes. Checking this is called calibration.' },
  { q: 'Why can a denoising model trained with squared error produce slightly blurry pictures?', options: ['It blurs on purpose', 'It outputs the average of many possible clean images, and an average of sharp pictures is smooth', 'It has too few pixels', 'Squared error is a bug'], answer: 1, why: 'Squared error pushes the output toward the conditional mean. When several sharp answers are possible, their average is blurry.' },
]);

/* ============================================================
   4. Pages: home, concept, quiz
   ============================================================ */
function pageList() {
  const pages = [{ href: '#home', title: 'Start here' }];
  for (const L of LESSONS) {
    for (const c of CONCEPTS.filter(c => c.lesson === L.id)) pages.push({ href: '#concept/' + c.id, title: c.title });
    pages.push({ href: '#quiz/' + L.id, title: 'Quiz for ' + L.title.split(':')[0] });
  }
  return pages;
}
function pager(hash) {
  const pages = pageList(); const i = pages.findIndex(p => p.href === hash);
  const prev = i > 0 ? pages[i - 1] : null, next = i < pages.length - 1 ? pages[i + 1] : null;
  return el('div', { class: 'pager' }, prev ? el('a', { href: prev.href }, '< ' + prev.title) : el('span'), next ? el('a', { href: next.href }, next.title + ' >') : el('span'));
}
function showHome(main) {
  const nConcepts = CONCEPTS.length, nVisited = CONCEPTS.filter(c => progress.visited[c.id]).length;
  const allQ = Object.values(QUIZZES).flat(); const nRight = allQ.filter(q => progress.quiz[q.id]).length;
  const nTries = CONCEPTS.reduce((s, c) => s + c.tries.length, 0), nTried = Object.values(progress.tried).filter(Boolean).length;
  main.append(el('h1', null, 'Learn by playing'));
  main.append(el('p', { class: 'idea' }, 'Every concept from the lessons, as something you can touch. Drag a slider, click a picture, watch the numbers change. Then try the small experiments and take the quiz. Your progress is saved in this browser.'));
  main.append(el('div', { class: 'progress-grid' },
    el('div', { class: 'stat' }, el('div', { class: 'big' }, `${nVisited} / ${nConcepts}`), el('div', { class: 'small' }, 'playgrounds opened')),
    el('div', { class: 'stat' }, el('div', { class: 'big' }, `${nTried} / ${nTries}`), el('div', { class: 'small' }, 'experiments ticked off')),
    el('div', { class: 'stat' }, el('div', { class: 'big' }, `${nRight} / ${allQ.length}`), el('div', { class: 'small' }, 'quiz questions answered right'))));
  main.append(el('h2', null, 'How to use this'));
  main.append(el('ol', null,
    el('li', null, 'Read the lesson folder first (its README and script). This page is for playing with the ideas, not for meeting them the first time.'),
    el('li', null, 'Open a playground and do the "Try this" experiments underneath it. Tick them off as you go.'),
    el('li', null, 'Take the lesson quiz. A wrong answer costs nothing, you can try again.'),
    el('li', null, 'If something feels fuzzy, go back to the lesson README. The playgrounds use the same words.')));
  main.append(el('h2', null, 'The lessons'));
  for (const L of LESSONS) {
    const card = el('div', { class: 'lesson-card' }, el('b', null, L.title));
    const ul = el('ul');
    for (const c of CONCEPTS.filter(c => c.lesson === L.id)) ul.append(el('li', null, el('a', { href: '#concept/' + c.id }, c.title)));
    ul.append(el('li', null, el('a', { href: '#quiz/' + L.id }, `Quiz (${QUIZZES[L.id].length} questions)`)));
    card.append(ul); main.append(card);
  }
}
function showConcept(id, main) {
  const c = CONCEPTS.find(c => c.id === id); if (!c) { location.hash = '#home'; return; }
  progress.visited[c.id] = true; saveProgress();
  const L = LESSONS.find(L => L.id === c.lesson);
  main.append(el('div', { class: 'crumb' }, L.title));
  main.append(el('h1', null, c.title));
  main.append(el('p', { class: 'idea' }, c.idea));
  const box = el('section', { class: 'playground' });
  main.append(box);
  try { c.build(box); } catch (err) { box.append(el('p', null, 'This playground hit an error: ' + err.message)); console.error(err); }
  const tries = el('section', { class: 'tries' }, el('h3', null, 'Try this'));
  c.tries.forEach((t, i) => {
    const key = c.id + ':' + i; const input = el('input', { type: 'checkbox' }); input.checked = !!progress.tried[key];
    input.addEventListener('change', () => { progress.tried[key] = input.checked; saveProgress(); renderNav(); });
    tries.append(el('label', null, input, el('span', null, t)));
  });
  main.append(tries);
  main.append(pager('#concept/' + id));
}
function showQuiz(lessonId, main) {
  const L = LESSONS.find(L => L.id === lessonId); const qs = QUIZZES[lessonId]; if (!L) { location.hash = '#home'; return; }
  main.append(el('div', { class: 'crumb' }, L.title));
  main.append(el('h1', null, 'Quiz: ' + L.title.split(': ')[1]));
  const score = el('p', { class: 'caption' });
  const updateScore = () => { const n = qs.filter(q => progress.quiz[q.id]).length; score.textContent = `${n} of ${qs.length} answered right so far. Wrong answers cost nothing: try again.`; };
  main.append(score); updateScore();
  for (const q of qs) {
    const card = el('div', { class: 'question' }, el('div', { class: 'q' }, q.q));
    const opts = el('div', { class: 'options' }); const feedback = el('div');
    q.options.forEach((o, i) => {
      const b = el('button', { class: 'option' + (progress.quiz[q.id] && i === q.answer ? ' right' : '') }, o);
      b.addEventListener('click', () => {
        feedback.innerHTML = ''; [...opts.children].forEach(x => x.classList.remove('wrong'));
        if (i === q.answer) { b.classList.add('right'); progress.quiz[q.id] = true; saveProgress(); feedback.append(el('div', { class: 'why' }, 'Right. ' + q.why)); updateScore(); renderNav(); }
        else { b.classList.add('wrong'); feedback.append(el('div', { class: 'nope' }, 'Not quite. Think about it once more and pick again.')); }
      });
      opts.append(b);
    });
    if (progress.quiz[q.id]) feedback.append(el('div', { class: 'why' }, 'Right. ' + q.why));
    card.append(opts, feedback); main.append(card);
  }
  main.append(pager('#quiz/' + lessonId));
}

/* ============================================================
   5. Navigation and routing
   ============================================================ */
function navLink(hash, title, active, done) {
  return el('a', { href: '#' + hash, class: 'nav-link' + (active ? ' active' : '') + (done ? ' visited' : '') }, el('span', { class: 'mark' }, done ? '✓' : ''), el('span', null, title));
}
function renderNav() {
  const nav = document.getElementById('nav'); nav.innerHTML = '';
  const current = location.hash.slice(1) || 'home';
  nav.append(navLink('home', 'Start here', current === 'home', false));
  for (const L of LESSONS) {
    nav.append(el('div', { class: 'nav-lesson' }, L.title));
    for (const c of CONCEPTS.filter(c => c.lesson === L.id)) nav.append(navLink('concept/' + c.id, c.title, current === 'concept/' + c.id, !!progress.visited[c.id]));
    const qs = QUIZZES[L.id]; const done = qs.filter(q => progress.quiz[q.id]).length;
    nav.append(navLink('quiz/' + L.id, `Quiz (${done} of ${qs.length})`, current === 'quiz/' + L.id, done === qs.length));
  }
}
function route() {
  const hash = location.hash.slice(1) || 'home'; const [kind, id] = hash.split('/');
  const main = document.getElementById('main'); main.innerHTML = '';
  if (kind === 'concept') showConcept(id, main); else if (kind === 'quiz') showQuiz(id, main); else showHome(main);
  renderNav(); window.scrollTo(0, 0); document.body.classList.remove('nav-open');
}
window.addEventListener('hashchange', route);
document.getElementById('menu-toggle').addEventListener('click', () => document.body.classList.toggle('nav-open'));
document.getElementById('reset-progress').addEventListener('click', () => {
  if (confirm('Forget all ticks and quiz answers in this browser?')) { progress = { visited: {}, quiz: {}, tried: {} }; saveProgress(); route(); }
});
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', route);
route();
