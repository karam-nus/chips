/* ============================================================
   Chips Guide — interactive labs (vanilla JS, no dependencies)
   A page opts in with  <div class="lab" data-lab="NAME"></div>
   NAME ∈ roofline | bits | fit | systolic | wafer | sparsity | quant | atlas
   The arithmetic lives in small pure functions (exported for Node so
   tools/test_labs.js can check them against hand-derived values).
   Accelerator facts come from assets/data/accelerators.json.
   ============================================================ */
(function () {
  'use strict';

  var C = {
    bg: '#0b0b0b', grid: 'rgba(255,255,255,0.07)', grid2: 'rgba(255,255,255,0.14)', text: '#e8e8e8', dim: '#666', mid: '#9a9a9a',
    acc: '#e8916a', accHi: '#f6c39f', red: '#ef4444', green: '#22c55e', blue: '#60a5fa', purple: '#a78bfa',
    yellow: '#fbbf24', pink: '#f472b6', cyan: '#22d3ee', orange: '#fb923c', teal: '#2dd4bf'
  };
  var VENDOR_COLOR = { NVIDIA: '#76b900', AMD: '#ef4444', Google: '#60a5fa', AWS: '#fbbf24', Intel: '#22d3ee', Cerebras: '#f472b6', Groq: '#e8916a' };

  // ============================================================ pure functions
  // ---------- number formats
  var FORMATS = {
    fp32: { name: 'FP32', e: 8, m: 23, bias: 127, kind: 'ieee' },
    bf16: { name: 'BF16', e: 8, m: 7, bias: 127, kind: 'ieee' },
    fp16: { name: 'FP16', e: 5, m: 10, bias: 15, kind: 'ieee' },
    e5m2: { name: 'FP8 E5M2', e: 5, m: 2, bias: 15, kind: 'ieee' },
    e4m3: { name: 'FP8 E4M3', e: 4, m: 3, bias: 7, kind: 'fn' },
    e3m2: { name: 'FP6 E3M2', e: 3, m: 2, bias: 3, kind: 'finite' },
    e2m3: { name: 'FP6 E2M3', e: 2, m: 3, bias: 1, kind: 'finite' },
    e2m1: { name: 'FP4 E2M1', e: 2, m: 1, bias: 1, kind: 'finite' },
    int8: { name: 'INT8', bits: 8, kind: 'int' },
    int4: { name: 'INT4', bits: 4, kind: 'int' }
  };
  function fmtBits(f) { return f.kind === 'int' ? f.bits : 1 + f.e + f.m; }

  function fpDecode(f, code) {
    if (f === FORMATS.fp32) { var dv = new DataView(new ArrayBuffer(4)); dv.setUint32(0, code >>> 0); return dv.getFloat32(0); }
    var mMask = (1 << f.m) - 1, eMax = (1 << f.e) - 1;
    var s = (code >>> (f.e + f.m)) & 1, E = (code >>> f.m) & eMax, M = code & mMask;
    var sign = s ? -1 : 1;
    if (f.kind === 'ieee' && E === eMax) return M === 0 ? sign * Infinity : NaN;
    if (f.kind === 'fn' && E === eMax && M === mMask) return NaN;
    if (E === 0) return sign * Math.pow(2, 1 - f.bias) * (M / (1 << f.m));
    return sign * Math.pow(2, E - f.bias) * (1 + M / (1 << f.m));
  }

  var _tables = {};
  function fpTable(f) {           // all finite, non-negative values in code order (monotonic)
    if (_tables[f.name]) return _tables[f.name];
    var n = 1 << (f.e + f.m), out = [];
    for (var c = 0; c < n; c++) { var v = fpDecode(f, c); if (isFinite(v)) out.push({ c: c, v: v }); }
    _tables[f.name] = out; return out;
  }
  function fpStats(f) {
    var t = fpTable(f), max = t[t.length - 1].v;
    return { max: max, minNormal: Math.pow(2, 1 - f.bias), minSub: Math.pow(2, 1 - f.bias - f.m), eps: Math.pow(2, -f.m), finite: 2 * t.length - 1 };
  }
  // Round-to-nearest-even into format f. IEEE kinds overflow to ±inf; others saturate to ±max.
  function fpRound(f, x) {
    var signBit = 1 << (f.e + f.m);
    if (f === FORMATS.fp32) {
      var dv = new DataView(new ArrayBuffer(4)); dv.setFloat32(0, x); var code = dv.getUint32(0);
      return { code: code, value: dv.getFloat32(0), sat: false };
    }
    if (isNaN(x)) { var nanCode = f.kind === 'ieee' ? (((1 << f.e) - 1) << f.m) | 1 : (f.kind === 'fn' ? (1 << (f.e + f.m)) - 1 : 0); return { code: nanCode, value: NaN, sat: false }; }
    var neg = x < 0 || (x === 0 && 1 / x < 0), a = Math.abs(x), t = fpTable(f), top = t[t.length - 1], r, sat = false;
    if (a >= top.v) {
      if (f.kind === 'ieee') {
        var ulp = Math.pow(2, ((1 << f.e) - 2) - f.bias - f.m);
        if (a >= top.v + ulp / 2) { r = { c: ((1 << f.e) - 1) << f.m, v: Infinity }; } else r = top;
      } else { r = top; sat = a > top.v; }
    } else {
      var lo = 0, hi = t.length - 1;
      while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (t[mid].v <= a) lo = mid; else hi = mid; }
      var dl = a - t[lo].v, dh = t[hi].v - a;
      r = dl < dh ? t[lo] : dh < dl ? t[hi] : ((t[lo].c & 1) === 0 ? t[lo] : t[hi]);
    }
    return { code: (neg ? signBit : 0) | r.c, value: neg ? -r.v : r.v, sat: sat };
  }
  function roundHalfEven(x) { var r = Math.round(x); return (Math.abs(x % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r; }
  function intQuant(bits, x, scale) {
    var lo = -(1 << (bits - 1)), hi = (1 << (bits - 1)) - 1, q = roundHalfEven(x / scale);
    var sat = q < lo || q > hi; q = Math.max(lo, Math.min(hi, q));
    return { q: q, code: q < 0 ? q + (1 << bits) : q, value: q * scale, sat: sat };
  }

  // ---------- roofline
  function attainable(peakTF, bwTBs, I) { return Math.min(peakTF, bwTBs * I); }     // TFLOP/s; TB/s × FLOP/B = TFLOP/s
  function intensity(kind, p, bytesW) {
    // returns FLOP per byte of off-chip traffic for a named workload
    if (kind === 'decode') { var K = 8192, N = 8192, B = p; return (2 * B * K * N) / (K * N * bytesW + (B * K + B * N) * 2); }
    if (kind === 'gemm') { var n = p; return (2 * n * n * n) / (3 * n * n * bytesW); }
    if (kind === 'attn') { return 2 * p / bytesW; }           // decode attention over a GQA group of p query heads per KV head
    if (kind === 'eltwise') { return 1 / (3 * bytesW); }        // c = a + b
    return 1;
  }

  // ---------- will it fit
  var MODELS = {
    'llama3-8b':   { name: 'Llama 3.1 8B',   P: 8.03,  L: 32,  kv: 8, hd: 128, h: 4096 },
    'llama3-70b':  { name: 'Llama 3.1 70B',  P: 70.6,  L: 80,  kv: 8, hd: 128, h: 8192 },
    'llama3-405b': { name: 'Llama 3.1 405B', P: 405.9, L: 126, kv: 8, hd: 128, h: 16384 },
    'qwen25-72b':  { name: 'Qwen2.5 72B',    P: 72.7,  L: 80,  kv: 8, hd: 128, h: 8192 },
    'mistral-7b':  { name: 'Mistral 7B',     P: 7.24,  L: 32,  kv: 8, hd: 128, h: 4096 },
    'dsv3-671b':   { name: 'DeepSeek-V3 671B (MoE, MLA)', P: 671, L: 61, kvElems: 576, h: 7168 }
  };
  var GiB = Math.pow(2, 30);
  function kvCacheBytes(m, ctx, batch, kvBytes) {
    var perTokPerLayer = m.kvElems ? m.kvElems : 2 * m.kv * m.hd;     // MLA caches one compressed latent (+RoPE key)
    return perTokPerLayer * m.L * kvBytes * ctx * batch;
  }
  function memoryPlan(o) {
    // o: {m, mode, wBytes, kvBytes, ctx, batch, gpus, ckpt, shard}  → bytes per GPU by component
    var P = o.m.P * 1e9, N = o.gpus, r = { weights: 0, kv: 0, grads: 0, optim: 0, act: 0, overhead: 1.5e9 };
    var tokens = o.ctx * o.batch, h = o.m.h, L = o.m.L;
    var actFull = 34 * tokens * h * L, actCkpt = 2 * tokens * h * L + 34 * tokens * h;   // Korthikanti et al. (16-bit, FlashAttention)
    if (o.mode === 'infer') {
      r.weights = P * o.wBytes / N; r.kv = kvCacheBytes(o.m, o.ctx, o.batch, o.kvBytes) / N;
    } else if (o.mode === 'train') {
      var div = o.shard ? N : 1;
      r.weights = 2 * P / div; r.grads = 2 * P / div; r.optim = 12 * P / div;              // bf16 w + bf16 g + fp32 master, m, v
      r.act = o.ckpt ? actCkpt : actFull;
    } else {                                                                               // LoRA / QLoRA, adapters ≈ 0.5 % of P
      r.weights = P * o.wBytes / (o.shard ? N : 1); var A = 0.005 * P;
      r.grads = 2 * A; r.optim = 12 * A + 2 * A; r.act = o.ckpt ? actCkpt : actFull;
    }
    r.total = r.weights + r.kv + r.grads + r.optim + r.act + r.overhead;
    return r;
  }

  // ---------- wafer yield
  function grossDies(waferMM, edgeMM, w, h) {
    // count whole w×h dies inside the usable circle; try a few grid offsets, keep the best
    var R = waferMM / 2 - edgeMM, best = 0, bestOff = [0, 0];
    for (var ox = 0; ox < 1; ox += 0.25) for (var oy = 0; oy < 1; oy += 0.25) {
      var n = 0, x0 = -R + ox * w, y0 = -R + oy * h;
      for (var x = x0; x + w <= R; x += w) for (var y = y0; y + h <= R; y += h) {
        var fx = Math.max(Math.abs(x), Math.abs(x + w)), fy = Math.max(Math.abs(y), Math.abs(y + h));
        if (fx * fx + fy * fy <= R * R) n++;
      }
      if (n > best) { best = n; bestOff = [ox, oy]; }
    }
    return { n: best, off: bestOff, R: R };
  }
  function diesPerWaferFormula(waferMM, S) { var d = waferMM; return Math.PI * d * d / (4 * S) - Math.PI * d / Math.sqrt(2 * S); }
  function yieldModels(areaCm2, D0) {
    var AD = areaCm2 * D0;
    return { poisson: Math.exp(-AD), murphy: AD === 0 ? 1 : Math.pow((1 - Math.exp(-AD)) / AD, 2), negbin: Math.pow(1 + AD / 3, -3) };
  }

  // ---------- sparsity
  function pruneNM(w, rows, cols, n, m) {        // keep the n largest |w| in every group of m along a row
    var keep = new Array(w.length).fill(false);
    for (var r = 0; r < rows; r++) for (var g = 0; g < cols; g += m) {
      var idx = []; for (var k = 0; k < m; k++) idx.push(r * cols + g + k);
      idx.sort(function (a, b) { return Math.abs(w[b]) - Math.abs(w[a]); });
      for (var j = 0; j < n; j++) keep[idx[j]] = true;
    }
    return keep;
  }
  function pruneUnstructured(w, frac) {
    var order = w.map(function (v, i) { return i; }).sort(function (a, b) { return Math.abs(w[b]) - Math.abs(w[a]); });
    var keep = new Array(w.length).fill(false), k = Math.round(w.length * frac);
    for (var i = 0; i < k; i++) keep[order[i]] = true; return keep;
  }
  function energyKept(w, keep) { var a = 0, b = 0; for (var i = 0; i < w.length; i++) { b += w[i] * w[i]; if (keep[i]) a += w[i] * w[i]; } return a / b; }

  // ---------- quantization
  var E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
  function roundToGrid(x, grid) {            // grid: non-negative ascending, symmetric use; ties → even index
    var a = Math.abs(x), best = 0;
    for (var i = 1; i < grid.length; i++) {
      var d = Math.abs(grid[i] - a), db = Math.abs(grid[best] - a);
      if (d < db || (d === db && i % 2 === 0)) best = i;
    }
    return (x < 0 ? -1 : 1) * grid[best];
  }
  function quantize(x, opt) {
    // x: Float64Array flattened rows×cols; opt: {rows, cols, scheme: 'int'|'mxfp4'|'nvfp4', bits, gran: 'tensor'|'channel'|number, asym}
    var out = new Float64Array(x.length), rows = opt.rows, cols = opt.cols, g;
    var gmax = 0; for (var i = 0; i < x.length; i++) gmax = Math.max(gmax, Math.abs(x[i]));
    if (opt.scheme === 'mxfp4') g = 32; else if (opt.scheme === 'nvfp4') g = 16; else g = opt.gran === 'tensor' ? x.length : opt.gran === 'channel' ? cols : opt.gran;
    var tensorScale = gmax / (448 * 6);
    for (var start = 0; start < x.length; start += g) {
      var end = Math.min(x.length, start + g), amax = 0, mn = Infinity, mx = -Infinity;
      for (var j = start; j < end; j++) { amax = Math.max(amax, Math.abs(x[j])); mn = Math.min(mn, x[j]); mx = Math.max(mx, x[j]); }
      if (amax === 0) continue;
      if (opt.scheme === 'mxfp4') {
        var s = Math.pow(2, Math.floor(Math.log2(amax)) - 2);              // E8M0 shared exponent, emax(E2M1) = 2
        for (j = start; j < end; j++) out[j] = s * roundToGrid(Math.max(-6, Math.min(6, x[j] / s)), E2M1);
      } else if (opt.scheme === 'nvfp4') {
        var s8 = fpRound(FORMATS.e4m3, amax / 6 / tensorScale).value * tensorScale;   // E4M3 block scale × FP32 tensor scale
        if (s8 === 0) s8 = tensorScale;
        for (j = start; j < end; j++) out[j] = s8 * roundToGrid(Math.max(-6, Math.min(6, x[j] / s8)), E2M1);
      } else {
        var qmax = (1 << (opt.bits - 1)) - 1;
        if (opt.asym) {
          var levels = (1 << opt.bits) - 1, sa = (mx - mn) / levels || 1, z = Math.round(-mn / sa);
          for (j = start; j < end; j++) { var q = Math.max(0, Math.min(levels, Math.round(x[j] / sa) + z)); out[j] = (q - z) * sa; }
        } else {
          var ss = amax / qmax;
          for (j = start; j < end; j++) out[j] = Math.max(-qmax, Math.min(qmax, Math.round(x[j] / ss))) * ss;
        }
      }
    }
    var sig = 0, err = 0, maxe = 0;
    for (i = 0; i < x.length; i++) { sig += x[i] * x[i]; var e = x[i] - out[i]; err += e * e; maxe = Math.max(maxe, Math.abs(e)); }
    var bitsPer = opt.scheme === 'mxfp4' ? 4 + 8 / 32 : opt.scheme === 'nvfp4' ? 4 + 8 / 16 : opt.bits + (16 * (opt.asym ? 2 : 1)) / g;
    return { out: out, sqnr: 10 * Math.log10(sig / err), maxErr: maxe, bitsPer: bitsPer, group: g };
  }

  // ---------- systolic (output-stationary): which k does PE(i,j) consume at cycle t?
  function systolicK(t, i, j, n) { var k = t - i - j; return (k >= 0 && k < n) ? k : -1; }

  function gaussian(rand) { var u = 0, v = 0; while (u === 0) u = rand(); v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function rng(seed) { var s = seed >>> 0 || 1; return function () { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }

  var API = {
    FORMATS: FORMATS, fpDecode: fpDecode, fpRound: fpRound, fpStats: fpStats, intQuant: intQuant,
    attainable: attainable, intensity: intensity, MODELS: MODELS, kvCacheBytes: kvCacheBytes, memoryPlan: memoryPlan,
    grossDies: grossDies, diesPerWaferFormula: diesPerWaferFormula, yieldModels: yieldModels,
    pruneNM: pruneNM, pruneUnstructured: pruneUnstructured, energyKept: energyKept, quantize: quantize, systolicK: systolicK,
    rng: rng, gaussian: gaussian
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  if (typeof document === 'undefined') return;

  // ============================================================ DOM helpers
  function el(tag, attrs, html) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (html !== undefined) e.innerHTML = html;
    return e;
  }
  function canvas(w, h) {
    var c = el('canvas'), r = window.devicePixelRatio || 1;
    c.width = w * r; c.height = h * r; c.style.width = w + 'px'; c.style.maxWidth = '100%'; c.style.height = 'auto';
    c.getContext('2d').setTransform(r, 0, 0, r, 0, 0);
    c._w = w; c._h = h; return c;
  }
  function pointer(c, ev) { var b = c.getBoundingClientRect(); return { x: (ev.clientX - b.left) * c._w / b.width, y: (ev.clientY - b.top) * c._h / b.height }; }
  function frame(root, title) {
    root.innerHTML = '';
    root.appendChild(el('div', { 'class': 'lab-title' }, title));
    var body = el('div', { 'class': 'lab-body' }); root.appendChild(body); return body;
  }
  function panel(body) { var p = el('div', { 'class': 'lab-panel' }); body.appendChild(p); return p; }
  function controls(parent) { var c = el('div', { 'class': 'lab-controls' }); parent.appendChild(c); return c; }
  function select(parent, label, opts, val) {
    var l = el('label', null, label), s = el('select');
    opts.forEach(function (o) {
      var v = Array.isArray(o) ? o[0] : o, t = Array.isArray(o) ? o[1] : o;
      var op = el('option', { value: v }, t); if (String(v) === String(val)) op.selected = true; s.appendChild(op);
    });
    l.appendChild(s); parent.appendChild(l); return s;
  }
  function setOptions(s, opts, val) {
    s.innerHTML = ''; opts.forEach(function (o) { var op = el('option', { value: o[0] }, o[1]); if (String(o[0]) === String(val)) op.selected = true; s.appendChild(op); });
  }
  function slider(parent, label, min, max, step, val, show) {
    var l = el('label', null, label), s = el('input', { type: 'range', min: min, max: max, step: step, value: val });
    var out = el('span', { style: 'font-family:var(--font-mono);min-width:3.6em;display:inline-block' });
    var upd = function () { out.textContent = show ? show(+s.value) : s.value; };
    s.addEventListener('input', upd); upd();
    l.appendChild(s); l.appendChild(out); parent.appendChild(l); s._upd = upd; return s;
  }
  function checkbox(parent, label, val) { var l = el('label'), c = el('input', { type: 'checkbox' }); c.checked = !!val; l.appendChild(c); l.appendChild(document.createTextNode(label)); parent.appendChild(l); return c; }
  function button(parent, label) { var b = el('button', { type: 'button' }, label); parent.appendChild(b); return b; }
  function readout(parent) { var r = el('div', { 'class': 'lab-readout' }); parent.appendChild(r); return r; }
  function hint(parent, t) { parent.appendChild(el('div', { 'class': 'lab-hint' }, t)); }
  function fmt(x, d) { return (x === null || x === undefined || isNaN(x)) ? '—' : Number(x).toFixed(d === undefined ? 2 : d); }
  function si(x, unit) {
    if (!isFinite(x)) return String(x);
    var a = Math.abs(x), p = [[1e15, 'P'], [1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'k']];
    for (var i = 0; i < p.length; i++) if (a >= p[i][0]) return (x / p[i][0]).toPrecision(3) + ' ' + p[i][1] + (unit || '');
    return x.toPrecision(3) + ' ' + (unit || '');
  }
  function gib(b) { return (b / 1e9).toFixed(1) + ' GB'; }
  function txt(ctx, s, x, y, color, size, align, font, base) {
    ctx.fillStyle = color || C.text; ctx.font = (size || 11) + 'px ' + (font || 'JetBrains Mono, monospace');
    ctx.textAlign = align || 'left'; ctx.textBaseline = base || 'middle'; ctx.fillText(s, x, y);
  }
  function rect(ctx, x, y, w, h, fill, stroke) {
    if (fill) { ctx.fillStyle = fill; ctx.fillRect(x, y, w, h); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); }
  }
  function clear(ctx, c) { ctx.fillStyle = C.bg; ctx.fillRect(0, 0, c._w, c._h); }
  function line(ctx, x1, y1, x2, y2, color, w, dash) { ctx.strokeStyle = color; ctx.lineWidth = w || 1; ctx.setLineDash(dash || []); ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.setLineDash([]); }
  function dot(ctx, x, y, r, fill, stroke) { ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); } }
  // log-log axes; returns mapping functions
  function logAxes(ctx, box, xr, yr, xl, yl) {
    var lx = function (v) { return box.x + (Math.log10(v) - Math.log10(xr[0])) / (Math.log10(xr[1]) - Math.log10(xr[0])) * box.w; };
    var ly = function (v) { return box.y + box.h - (Math.log10(v) - Math.log10(yr[0])) / (Math.log10(yr[1]) - Math.log10(yr[0])) * box.h; };
    for (var e = Math.ceil(Math.log10(xr[0])); e <= Math.floor(Math.log10(xr[1])); e++) {
      var X = lx(Math.pow(10, e)); line(ctx, X, box.y, X, box.y + box.h, C.grid);
      txt(ctx, e >= 3 ? Math.pow(10, e / 1).toExponential(0).replace('e+', 'e') : String(Math.pow(10, e)), X, box.y + box.h + 11, C.mid, 9.5, 'center');
    }
    for (e = Math.ceil(Math.log10(yr[0])); e <= Math.floor(Math.log10(yr[1])); e++) {
      var Y = ly(Math.pow(10, e)); line(ctx, box.x, Y, box.x + box.w, Y, C.grid);
      txt(ctx, si(Math.pow(10, e) * 1e12, ''), box.x - 5, Y, C.mid, 9.5, 'right');
    }
    rect(ctx, box.x, box.y, box.w, box.h, null, C.grid2);
    txt(ctx, xl, box.x + box.w / 2, box.y + box.h + 25, C.mid, 10, 'center');
    ctx.save(); ctx.translate(box.x - 54, box.y + box.h / 2); ctx.rotate(-Math.PI / 2); txt(ctx, yl, 0, 0, C.mid, 10, 'center'); ctx.restore();
    return { x: lx, y: ly };
  }
  var DT_BYTES = { bf16: 2, fp8: 1, fp4: 0.5, int8: 1 };
  var DT_FIELD = { bf16: 'bf16_tf', fp8: 'fp8_tf', fp4: 'fp4_tf', int8: 'int8_tops' };
  var DT_NAME = { bf16: 'BF16/FP16', fp8: 'FP8', fp4: 'FP4', int8: 'INT8' };
  function dtypesOf(chip) { return ['bf16', 'fp8', 'fp4'].filter(function (d) { return chip[DT_FIELD[d]]; }); }

  // ============================================================ 1. roofline
  function labRoofline(root, data) {
    var chips = data.chips.filter(function (c) { return dtypesOf(c).length && c.kind !== 'Wafer' && c.mem_tbps < 100; });
    var body = frame(root, 'Lab · Roofline — is this operation compute-bound or memory-bound on this chip?');
    var c = canvas(560, 340); body.appendChild(c); var ctx = c.getContext('2d');
    var p = panel(body), ct = controls(p);
    var chipS = select(ct, 'chip', chips.map(function (x) { return [x.id, x.name]; }), 'h100');
    var dtS = select(ct, 'dtype', [], 'bf16');
    var cmpS = select(ct, 'compare', [['', 'none']].concat(chips.map(function (x) { return [x.id, x.name]; })), 'a100');
    var ct2 = controls(p);
    var wlS = select(ct2, 'workload', [['decode', 'LLM decode GEMM, batch B'], ['gemm', 'square GEMM n×n×n'], ['attn', 'decode attention, GQA group g'], ['eltwise', 'elementwise add']], 'decode');
    var parS = slider(ct2, 'B', 0, 10, 1, 0, function (v) { return String(Math.pow(2, v)); });
    var out = readout(p);
    hint(p, 'The roof is min(peak, bandwidth × intensity). Raise the batch or drop to FP8/FP4 weights and watch the operating point slide right. Attention decode does not move with batch: every sequence reads its own KV cache.');
    function chip(id) { for (var i = 0; i < chips.length; i++) if (chips[i].id === id) return chips[i]; return chips[0]; }
    function syncDtypes() { var ch = chip(chipS.value), ds = dtypesOf(ch), cur = dtS.value; setOptions(dtS, ds.map(function (d) { return [d, DT_NAME[d]]; }), ds.indexOf(cur) >= 0 ? cur : ds[0]); }
    function syncParam() {
      var w = wlS.value, lab = parS.parentNode.firstChild;
      if (w === 'decode') { lab.textContent = 'B'; parS.min = 0; parS.max = 10; }
      else if (w === 'gemm') { lab.textContent = 'n'; parS.min = 6; parS.max = 14; if (+parS.value < 6) parS.value = 12; }
      else if (w === 'attn') { lab.textContent = 'g'; parS.min = 0; parS.max = 4; if (+parS.value > 4) parS.value = 3; }
      else { lab.textContent = '—'; }
      parS.disabled = w === 'eltwise'; parS._upd();
    }
    function draw() {
      clear(ctx, c);
      var ch = chip(chipS.value), dt = dtS.value || 'bf16', peak = ch[DT_FIELD[dt]], bw = ch.mem_tbps, bytes = DT_BYTES[dt];
      var box = { x: 72, y: 16, w: 470, h: 270 }, ax = logAxes(ctx, box, [0.1, 10000], [0.1, 100000], 'arithmetic intensity  (FLOP per byte of HBM traffic)', 'attainable  FLOP/s');
      function roof(cx, d, color, width, label) {
        var pk = cx[DT_FIELD[d]]; if (!pk) return;
        var ridge = pk / cx.mem_tbps; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
        ctx.moveTo(ax.x(0.1), ax.y(cx.mem_tbps * 0.1)); ctx.lineTo(ax.x(ridge), ax.y(pk)); ctx.lineTo(ax.x(10000), ax.y(pk)); ctx.stroke();
        txt(ctx, label, ax.x(10000) - 4, ax.y(pk) - 8, color, 10, 'right');
        return ridge;
      }
      var cmp = cmpS.value ? chip(cmpS.value) : null;
      if (cmp && cmp.id !== ch.id) { var cd = cmp[DT_FIELD[dt]] ? dt : dtypesOf(cmp)[0]; roof(cmp, cd, 'rgba(160,160,160,0.55)', 1.5, cmp.name + ' ' + DT_NAME[cd]); }
      var ridge = roof(ch, dt, C.acc, 2.5, ch.name + ' ' + DT_NAME[dt]);
      line(ctx, ax.x(ridge), ax.y(peak), ax.x(ridge), box.y + box.h, 'rgba(232,145,106,0.45)', 1, [3, 3]);
      txt(ctx, 'ridge ' + fmt(ridge, 0), ax.x(ridge) + 4, box.y + box.h - 10, C.accHi, 9.5);
      var w = wlS.value, pv = Math.pow(2, +parS.value), I = intensity(w, pv, bytes), att = attainable(peak, bw, I);
      var xI = Math.max(0.1, Math.min(10000, I));
      dot(ctx, ax.x(xI), ax.y(att), 6, I < ridge ? C.red : C.green, '#000');
      var nm = w === 'decode' ? 'decode B=' + pv : w === 'gemm' ? 'GEMM n=' + pv : w === 'attn' ? 'attn g=' + pv : 'add';
      txt(ctx, nm, ax.x(xI) + 9, ax.y(att) + 1, I < ridge ? '#fca5a5' : '#86efac', 10.5);
      var bound = I < ridge ? 'MEMORY-bound' : 'COMPUTE-bound';
      out.textContent = ch.name + '  (' + ch.mem_type + ', ' + ch.mem_gb + ' GB)' +
        '\npeak ' + DT_NAME[dt] + '        ' + si(peak * 1e12, 'FLOP/s') +
        '\nbandwidth         ' + fmt(bw, 2) + ' TB/s' +
        '\nridge point       ' + fmt(ridge, 0) + ' FLOP/B' +
        '\nworkload I        ' + fmt(I, 2) + ' FLOP/B' +
        '\nattainable        ' + si(att * 1e12, 'FLOP/s') + '  (' + fmt(100 * att / peak, 1) + ' % of peak)' +
        '\nverdict           ' + bound;
    }
    chipS.addEventListener('change', function () { syncDtypes(); draw(); });
    [dtS, cmpS, parS].forEach(function (x) { x.addEventListener('input', draw); x.addEventListener('change', draw); });
    wlS.addEventListener('change', function () { syncParam(); draw(); });
    syncDtypes(); syncParam(); draw();
  }

  // ============================================================ 2. bits
  function labBits(root) {
    var body = frame(root, 'Lab · Bit explorer — what a number becomes in each format');
    var p = el('div', { 'class': 'lab-panel', style: 'flex-basis:100%' }); body.appendChild(p);
    var ct = controls(p);
    var fS = select(ct, 'format', Object.keys(FORMATS).map(function (k) { return [k, FORMATS[k].name]; }), 'bf16');
    var lab = el('label', null, 'value'), inp = el('input', { type: 'text', value: '3.14159265' }); lab.appendChild(inp); ct.appendChild(lab);
    var scl = el('label', null, 'scale'), sIn = el('input', { type: 'text', value: '0.05' }); scl.appendChild(sIn); ct.appendChild(scl);
    var ct2 = controls(p);
    ['0.1', '1/3', '65504', '448', '1e-8', '100000', '-2.5'].forEach(function (v) { var b = button(ct2, v); b.onclick = function () { inp.value = v; encode(); }; });
    var bitsDiv = el('div', { 'class': 'bits' }); p.appendChild(bitsDiv);
    var leg = el('div', { 'class': 'lab-legend' }, '<span><i style="background:#ef4444"></i>sign</span><span><i style="background:#f97316"></i>exponent</span><span><i style="background:#06b6d4"></i>mantissa</span><span><i style="background:#a855f7"></i>integer</span>');
    p.appendChild(leg);
    var out = readout(p);
    hint(p, 'Type a value (fractions like 1/3 work) or click any bit to flip it. FP8 E4M3, FP6 and FP4 have no infinity: out-of-range values saturate to ±max (the OCP conversion default used by most kernels). INT formats store round(x / scale).');
    var code = 0;
    function parse(s) { s = String(s).trim(); if (/^-?[\d.e+-]+\/[\d.e+-]+$/i.test(s)) { var q = s.split('/'); return parseFloat(q[0]) / parseFloat(q[1]); } return parseFloat(s); }
    function render(x) {
      var f = FORMATS[fS.value], n = fmtBits(f), isInt = f.kind === 'int', bits = [];
      scl.style.display = isInt ? '' : 'none';
      for (var i = n - 1; i >= 0; i--) bits.push((code >>> i) & 1);
      bitsDiv.innerHTML = '';
      bits.forEach(function (b, idx) {
        var cls = isInt ? 'i' : (idx === 0 ? 's' : idx <= f.e ? 'e' : 'm');
        var d = el('span', { 'class': 'bit ' + cls + (b ? ' on' : ''), title: 'bit ' + (n - 1 - idx) }, String(b));
        d.onclick = function () { code = (code ^ (1 << (n - 1 - idx))) >>> 0; if (n === 32) code = code >>> 0; render(null); };
        bitsDiv.appendChild(d);
      });
      var lines = [];
      if (isInt) {
        var s = parse(sIn.value) || 1, qv = code >= (1 << (n - 1)) ? code - (1 << n) : code;
        lines.push('stored integer q   ' + qv + '   (two\'s complement)');
        lines.push('decoded q × s      ' + (qv * s).toPrecision(8));
        if (x !== null) lines.push('input / error      ' + x + '  →  abs err ' + Math.abs(x - qv * s).toPrecision(3));
        lines.push('range              [' + (-(1 << (n - 1))) + ', ' + ((1 << (n - 1)) - 1) + '] × s   — ' + (1 << n) + ' levels, uniformly spaced');
      } else {
        var v = fpDecode(f, code), st = fpStats(f), E = (code >>> f.m) & ((1 << f.e) - 1), M = code & ((1 << f.m) - 1);
        lines.push('fields             sign ' + ((code >>> (n - 1)) & 1) + ' · exp ' + E + ' (bias ' + f.bias + ') · mantissa ' + M + '/' + (1 << f.m));
        lines.push('decoded value      ' + (isFinite(v) ? v.toPrecision(10).replace(/\.?0+$/, '') : String(v)) + (E === 0 && M !== 0 ? '   (subnormal)' : ''));
        if (x !== null) {
          var rel = x !== 0 ? Math.abs((v - x) / x) : Math.abs(v);
          lines.push('input → rel. error ' + x + '  →  ' + (isFinite(rel) ? rel.toExponential(2) : '∞') + (Math.abs(x) > st.max && f.kind !== 'ieee' ? '   SATURATED' : ''));
        }
        lines.push('max finite         ' + st.max.toPrecision(6));
        lines.push('min normal / sub   ' + st.minNormal.toExponential(2) + ' / ' + st.minSub.toExponential(2));
        lines.push('epsilon (2^-m)     ' + st.eps.toExponential(2) + '   ≈ ' + fmt(f.m * Math.log10(2) + 0.3, 1) + ' decimal digits');
        if (n <= 16) lines.push('finite values      ' + st.finite + (f.kind === 'ieee' ? '   (+ ±∞, NaNs)' : f.kind === 'fn' ? '   (no ∞; NaN = S.1111.111)' : '   (no ∞, no NaN)'));
      }
      out.textContent = lines.join('\n');
    }
    function encode() {
      var f = FORMATS[fS.value], x = parse(inp.value);
      if (isNaN(x)) { out.textContent = 'not a number'; return; }
      code = f.kind === 'int' ? intQuant(f.bits, x, parse(sIn.value) || 1).code : fpRound(f, x).code;
      render(x);
    }
    [fS, inp, sIn].forEach(function (x) { x.addEventListener('input', encode); x.addEventListener('change', encode); });
    encode();
  }

  // ============================================================ 3. will it fit
  function labFit(root, data) {
    var gpus = data.chips.filter(function (c) { return c.mem_gb >= 16 && c.mem_tbps < 100; });
    var body = frame(root, 'Lab · Will it fit? — memory per GPU for serving or training a model');
    var c = canvas(560, 230); body.appendChild(c); var ctx = c.getContext('2d');
    var p = panel(body), ct = controls(p);
    var mS = select(ct, 'model', Object.keys(MODELS).map(function (k) { return [k, MODELS[k].name]; }), 'llama3-70b');
    var modeS = select(ct, 'job', [['infer', 'inference'], ['train', 'full fine-tune (Adam, mixed precision)'], ['lora', 'LoRA / QLoRA']], 'infer');
    var ct2 = controls(p);
    var wS = select(ct2, 'weights', [['4', 'FP32'], ['2', 'BF16'], ['1', 'FP8 / INT8'], ['0.5625', 'INT4 g128 / FP4 + scales']], '2');
    var kvS = select(ct2, 'KV cache', [['2', 'BF16'], ['1', 'FP8']], '2');
    var ct3 = controls(p);
    var ctxS = slider(ct3, 'context', 9, 17, 1, 13, function (v) { var t = Math.pow(2, v); return t >= 1024 ? (t / 1024) + 'k' : String(t); });
    var bS = slider(ct3, 'batch', 0, 8, 1, 0, function (v) { return String(Math.pow(2, v)); });
    var ct4 = controls(p);
    var gS = select(ct4, 'GPU', gpus.map(function (g) { return [g.id, g.name + ' · ' + g.mem_gb + ' GB']; }), 'h100');
    var nS = slider(ct4, 'GPUs', 0, 9, 1, 0, function (v) { return String(Math.pow(2, v)); });
    var ct5 = controls(p);
    var ck = checkbox(ct5, ' activation checkpointing', true), sh = checkbox(ct5, ' shard weights+optimizer (FSDP / ZeRO-3)', true);
    var out = readout(p);
    hint(p, 'Estimates, not guarantees: weights = params × bytes; KV = 2 × layers × KV heads × head dim × bytes × tokens (DeepSeek MLA caches a 576-wide latent instead); Adam mixed precision = 16 bytes/param; activations from Korthikanti et al. with FlashAttention; 1.5 GB per GPU for the CUDA context and workspaces. All sizes in decimal GB, as vendors quote capacity.');
    function gpu() { for (var i = 0; i < gpus.length; i++) if (gpus[i].id === gS.value) return gpus[i]; return gpus[0]; }
    function plan(n) {
      return memoryPlan({ m: MODELS[mS.value], mode: modeS.value, wBytes: +wS.value, kvBytes: +kvS.value, ctx: Math.pow(2, +ctxS.value), batch: Math.pow(2, +bS.value), gpus: n, ckpt: ck.checked, shard: sh.checked });
    }
    function draw() {
      clear(ctx, c);
      var g = gpu(), cap = g.mem_gb * 1e9, N = Math.pow(2, +nS.value), r = plan(N), parts = [
        ['weights', r.weights, C.acc], ['KV cache', r.kv, C.purple], ['gradients', r.grads, C.blue], ['optimizer', r.optim, C.cyan], ['activations', r.act, C.yellow], ['runtime', r.overhead, '#777']];
      var scaleMax = Math.max(cap, r.total) * 1.08, x0 = 20, W = 520, y = 46, H = 46;
      txt(ctx, 'per-GPU memory, ' + N + ' × ' + g.name, x0, 18, C.mid, 11);
      var x = x0;
      parts.forEach(function (pt) { var w = W * pt[1] / scaleMax; if (w > 0.5) { rect(ctx, x, y, w, H, pt[2]); if (w > 46) txt(ctx, pt[0], x + 4, y + H / 2, '#111', 10); x += w; } });
      var capX = x0 + W * cap / scaleMax; line(ctx, capX, y - 12, capX, y + H + 12, r.total <= cap ? C.green : C.red, 2);
      txt(ctx, g.mem_gb + ' GB', capX, y - 18, r.total <= cap ? C.green : C.red, 10, 'center');
      var ly = 128; parts.forEach(function (pt, i) { var lx = x0 + (i % 3) * 175, yy = ly + Math.floor(i / 3) * 18; rect(ctx, lx, yy - 5, 10, 10, pt[2]); txt(ctx, pt[0] + '  ' + gib(pt[1]), lx + 16, yy, C.text, 10.5); });
      var need = -1; for (var k = 0; k <= 12; k++) { var n2 = Math.pow(2, k); if (plan(n2).total <= cap) { need = n2; break; } }
      txt(ctx, r.total <= cap ? 'FITS  (' + fmt(100 * r.total / cap, 0) + ' % of capacity)' : 'DOES NOT FIT  (' + fmt(r.total / cap, 2) + '× capacity)', x0, 192, r.total <= cap ? C.green : C.red, 13, 'left', 'Inter, sans-serif');
      txt(ctx, need > 0 ? 'smallest power-of-two count that fits: ' + need + ' × ' + g.name : 'does not fit within 4096 GPUs with these settings', x0, 214, C.mid, 10.5);
      var m = MODELS[mS.value], kvTok = kvCacheBytes(m, 1, 1, +kvS.value);
      out.textContent = m.name + ' · ' + m.P + 'B params · ' + m.L + ' layers' +
        '\nKV per token      ' + (kvTok / 1e3).toFixed(0) + ' kB  (all layers)' +
        '\ntokens in flight  ' + Math.pow(2, +ctxS.value) * Math.pow(2, +bS.value) +
        '\nper-GPU total     ' + gib(r.total) + '  of ' + g.mem_gb + ' GB' +
        '\nwhole job         ' + gib(r.total * N) + '  across ' + N + ' GPU' + (N > 1 ? 's' : '');
    }
    [mS, modeS, wS, kvS, ctxS, bS, gS, nS, ck, sh].forEach(function (x) { x.addEventListener('input', draw); x.addEventListener('change', draw); });
    draw();
  }

  // ============================================================ 4. systolic array
  function labSystolic(root) {
    var body = frame(root, 'Lab · Systolic array — a 4×4 output-stationary matrix multiply, cycle by cycle');
    var N = 4, c = canvas(560, 390); body.appendChild(c); var ctx = c.getContext('2d');
    var p = panel(body), ct = controls(p), stepB = button(ct, 'Step'), playB = button(ct, 'Play'), resetB = button(ct, 'Reset'), newB = button(ct, 'New matrices');
    var out = readout(p);
    hint(p, 'Row i of A enters from the left delayed by i cycles; column j of B enters from the top delayed by j. PE(i,j) multiplies whatever arrives and adds it to its own C(i,j). Each operand is fetched from memory once and reused N times as it marches across. (Google\'s MXUs hold weights stationary instead; the skewed timing is the same idea.)');
    var A, B, Cm, t = -1, timer = null, R = rng(7);
    function fresh() { A = []; B = []; for (var i = 0; i < N; i++) { A.push([]); B.push([]); for (var j = 0; j < N; j++) { A[i].push(1 + Math.floor(R() * 4)); B[i].push(1 + Math.floor(R() * 4)); } } reset(); }
    function reset() { Cm = []; for (var i = 0; i < N; i++) { Cm.push([]); for (var j = 0; j < N; j++) Cm[i].push(0); } t = -1; draw(); }
    function step() {
      if (t >= 3 * N - 3) { if (timer) { clearInterval(timer); timer = null; } return; }
      t++; for (var i = 0; i < N; i++) for (var j = 0; j < N; j++) { var k = systolicK(t, i, j, N); if (k >= 0) Cm[i][j] += A[i][k] * B[k][j]; }
      draw();
    }
    function draw() {
      clear(ctx, c);
      var cell = 50, gx = 230, gy = 172, active = 0, macs = 0;
      txt(ctx, 'cycle ' + (t < 0 ? '—' : t) + ' / ' + (3 * N - 3), 20, 18, C.accHi, 12);
      // A queue (left), skewed
      for (var i = 0; i < N; i++) for (var k = 0; k < N; k++) {
        var slot = (i + k) - t - 1;           // cycles until a[i][k] enters column 0
        if (slot >= 0 && slot < 7) { var X = gx - 24 - slot * 24; rect(ctx, X - 10, gy + i * cell + cell / 2 - 10, 20, 20, 'rgba(96,165,250,0.18)', C.blue); txt(ctx, String(A[i][k]), X, gy + i * cell + cell / 2, C.blue, 10, 'center'); }
      }
      for (var j = 0; j < N; j++) for (k = 0; k < N; k++) {
        slot = (j + k) - t - 1;
        if (slot >= 0 && slot < 7) { var Y = gy - 24 - slot * 21; rect(ctx, gx + j * cell + cell / 2 - 10, Y - 10, 20, 20, 'rgba(167,139,250,0.18)', C.purple); txt(ctx, String(B[k][j]), gx + j * cell + cell / 2, Y, C.purple, 10, 'center'); }
      }
      txt(ctx, 'A rows →', 20, gy + N * cell + 14, C.blue, 10); txt(ctx, 'B columns ↓', gx + N * cell + 10, 30, C.purple, 10);
      for (i = 0; i < N; i++) for (j = 0; j < N; j++) {
        var kk = t >= 0 ? systolicK(t, i, j, N) : -1, x = gx + j * cell, y = gy + i * cell, on = kk >= 0;
        if (on) active++;
        rect(ctx, x + 2, y + 2, cell - 4, cell - 4, on ? 'rgba(232,145,106,0.25)' : '#161616', on ? C.acc : '#333');
        txt(ctx, String(Cm[i][j]), x + cell / 2, y + cell / 2 + 6, on ? '#fff' : C.mid, 13, 'center');
        if (on) txt(ctx, A[i][kk] + '×' + B[kk][j], x + cell / 2, y + 13, C.accHi, 9.5, 'center');
      }
      for (var tt = 0; tt <= t; tt++) for (i = 0; i < N; i++) for (j = 0; j < N; j++) if (systolicK(tt, i, j, N) >= 0) macs++;
      var exp = [], ok = true; for (i = 0; i < N; i++) { exp.push([]); for (j = 0; j < N; j++) { var s = 0; for (k = 0; k < N; k++) s += A[i][k] * B[k][j]; exp[i].push(s); if (s !== Cm[i][j]) ok = false; } }
      var cycles = t + 1;
      out.textContent = 'active PEs this cycle   ' + active + ' / ' + (N * N) +
        '\nMACs done               ' + macs + ' / ' + (N * N * N) +
        '\nutilisation so far      ' + (cycles > 0 ? fmt(100 * macs / (cycles * N * N), 0) : '—') + ' %' +
        '\ntotal cycles            ' + (3 * N - 2) + '  (ideal ' + N + ' + fill/drain ' + (2 * N - 2) + ')' +
        '\nC = A·B checked         ' + (t >= 3 * N - 3 ? (ok ? 'yes ✓' : 'mismatch') : 'running…') +
        '\nfor a 256×256 MXU, fill/drain is ' + fmt(100 * 510 / (3 * 256 - 2), 0) + ' % of a single 256-deep tile — why big batches matter';
    }
    stepB.onclick = step;
    playB.onclick = function () { if (timer) { clearInterval(timer); timer = null; return; } if (t >= 3 * N - 3) reset(); timer = setInterval(step, 600); };
    resetB.onclick = function () { if (timer) { clearInterval(timer); timer = null; } reset(); };
    newB.onclick = function () { if (timer) { clearInterval(timer); timer = null; } fresh(); };
    fresh();
  }

  // ============================================================ 5. wafer yield
  function labWafer(root) {
    var body = frame(root, 'Lab · Wafer yield — why big dies are expensive and chiplets exist');
    var c = canvas(360, 360); body.appendChild(c); var ctx = c.getContext('2d');
    var p = panel(body), ct = controls(p);
    var aS = slider(ct, 'die mm²', 10, 850, 2, 814), dS = slider(ct, 'defects/cm²', 0.01, 0.5, 0.01, 0.10, function (v) { return v.toFixed(2); });
    var ct2 = controls(p), costS = slider(ct2, 'wafer $', 5000, 40000, 500, 20000, function (v) { return '$' + (v / 1000).toFixed(1) + 'k'; });
    var reB = button(ct2, 'Re-roll defects');
    var out = readout(p);
    p.appendChild(el('div', { 'class': 'wafer-legend' }, '<span style="--k:#22c55e">good die</span><span style="--k:#ef4444">killed by a defect</span><span style="--k:#555">partial die at the edge</span><span style="--k:#fbbf24">defect</span>'));
    hint(p, '300 mm wafer, 3 mm edge exclusion, square dies. Defects land at random (Poisson); a die with any defect is scrapped. Compare one 814 mm² die with four ~200 mm² chiplets at the same defect density.');
    var seed = 1;
    function draw() {
      clear(ctx, c);
      var area = +aS.value, side = Math.sqrt(area), D0 = +dS.value, cost = +costS.value, waferMM = 300, edge = 3;
      var gd = grossDies(waferMM, edge, side, side), R = gd.R, cx = 180, cy = 180, k = 168 / (waferMM / 2);
      ctx.beginPath(); ctx.arc(cx, cy, 150 * k, 0, Math.PI * 2); ctx.fillStyle = '#1a1f2b'; ctx.fill();
      ctx.beginPath(); ctx.arc(cx, cy, 150 * k, 0, Math.PI * 2); ctx.strokeStyle = '#3a4560'; ctx.lineWidth = 1.5; ctx.stroke();
      // notch
      ctx.beginPath(); ctx.arc(cx, cy + 150 * k, 4, 0, Math.PI * 2); ctx.fillStyle = C.bg; ctx.fill();
      var rand = rng(seed * 9973 + Math.round(area)), nDef = 0, lambda = D0 * Math.PI * 150 * 150 / 100, defects = [];
      // Poisson draw (Knuth for small λ, normal approx for large)
      if (lambda < 60) { var L = Math.exp(-lambda), pp = 1; do { nDef++; pp *= rand(); } while (pp > L); nDef--; }
      else nDef = Math.max(0, Math.round(lambda + Math.sqrt(lambda) * gaussian(rand)));
      for (var i = 0; i < nDef; i++) { var rr = 150 * Math.sqrt(rand()), th = 2 * Math.PI * rand(); defects.push([rr * Math.cos(th), rr * Math.sin(th)]); }
      var good = 0, total = 0, x0 = -R + gd.off[0] * side, y0 = -R + gd.off[1] * side;
      for (var x = x0; x + side <= R + 1e-9; x += side) for (var y = y0; y + side <= R + 1e-9; y += side) {
        var fx = Math.max(Math.abs(x), Math.abs(x + side)), fy = Math.max(Math.abs(y), Math.abs(y + side)), inside = fx * fx + fy * fy <= R * R;
        var hit = false; if (inside) for (var d = 0; d < defects.length; d++) if (defects[d][0] >= x && defects[d][0] < x + side && defects[d][1] >= y && defects[d][1] < y + side) { hit = true; break; }
        var col = !inside ? '#3a3a3a' : hit ? '#ef4444' : '#22c55e';
        if (inside) { total++; if (!hit) good++; }
        var X = cx + x * k, Y = cy + y * k, S = side * k;
        if (Math.sqrt(Math.max(Math.abs(x), Math.abs(x + side)) * Math.max(Math.abs(x), Math.abs(x + side)) + 0) < 160) rect(ctx, X + 0.5, Y + 0.5, Math.max(1, S - 1), Math.max(1, S - 1), inside ? (hit ? 'rgba(239,68,68,0.75)' : 'rgba(34,197,94,0.55)') : 'rgba(80,80,80,0.35)');
      }
      defects.forEach(function (q) { dot(ctx, cx + q[0] * k, cy + q[1] * k, 1.6, C.yellow); });
      var y = yieldModels(area / 100, D0), f = diesPerWaferFormula(waferMM, area);
      out.textContent = 'die ' + area + ' mm²  (' + fmt(side, 1) + ' mm square)' +
        '\ngross dies        ' + gd.n + '   (formula ≈ ' + fmt(f, 0) + ')' +
        '\ndefects on wafer  ' + nDef + '   (λ = ' + fmt(lambda, 0) + ')' +
        '\nthis wafer        ' + good + ' good  →  ' + fmt(100 * good / Math.max(1, total), 1) + ' % yield' +
        '\nPoisson model     ' + fmt(100 * y.poisson, 1) + ' %    Murphy ' + fmt(100 * y.murphy, 1) + ' %' +
        '\ncost / good die   $' + fmt(cost / Math.max(1, gd.n * y.poisson), 0) + '   (wafer only; no test, packaging, HBM)';
    }
    [aS, dS, costS].forEach(function (x) { x.addEventListener('input', draw); });
    reB.onclick = function () { seed++; draw(); };
    draw();
  }

  // ============================================================ 6. sparsity
  function labSparsity(root) {
    var body = frame(root, 'Lab · 2:4 structured sparsity — what the Sparse Tensor Core actually stores');
    var rows = 8, cols = 16, c = canvas(560, 300); body.appendChild(c); var ctx = c.getContext('2d');
    var p = panel(body), ct = controls(p);
    var pS = select(ct, 'pattern', [['2:4', '2:4 (NVIDIA Ampere→)'], ['4:8', '4:8'], ['1:4', '1:4'], ['u50', 'unstructured 50 %']], '2:4');
    var newB = button(ct, 'New weights');
    var out = readout(p);
    hint(p, 'Top: the weight matrix, kept values bright. Bottom: the compressed form a 2:4 kernel reads, half the values plus a 2-bit index per kept value. Unstructured 50 % keeps a little more of the weight energy but gives the tensor core nothing to exploit.');
    var seed = 3, W;
    function fresh() { var R = rng(seed); W = []; for (var i = 0; i < rows * cols; i++) W.push(gaussian(R) * 0.5); draw(); }
    function draw() {
      clear(ctx, c);
      var pat = pS.value, keep, n = 2, m = 4;
      if (pat === 'u50') keep = pruneUnstructured(W, 0.5); else { n = +pat.split(':')[0]; m = +pat.split(':')[1]; keep = pruneNM(W, rows, cols, n, m); }
      var cw = 30, chh = 15, x0 = 40, y0 = 26, maxA = 0; W.forEach(function (v) { maxA = Math.max(maxA, Math.abs(v)); });
      txt(ctx, 'dense W  (' + rows + '×' + cols + ')', x0, 12, C.mid, 10);
      for (var r = 0; r < rows; r++) for (var j = 0; j < cols; j++) {
        var v = W[r * cols + j], a = Math.abs(v) / maxA, k = keep[r * cols + j];
        var col = v >= 0 ? 'rgba(232,145,106,' + (k ? 0.25 + 0.75 * a : 0.07) + ')' : 'rgba(96,165,250,' + (k ? 0.25 + 0.75 * a : 0.07) + ')';
        rect(ctx, x0 + j * cw, y0 + r * chh, cw - 2, chh - 2, col, k ? null : '#262626');
        if (pat !== 'u50' && j % m === 0) line(ctx, x0 + j * cw - 1, y0 + r * chh, x0 + j * cw - 1, y0 + r * chh + chh - 2, '#555');
      }
      var y1 = y0 + rows * chh + 22;
      if (pat !== 'u50') {
        txt(ctx, 'compressed values (' + rows + '×' + (cols * n / m) + ')  +  ' + Math.ceil(Math.log2(m)) + '-bit indices', x0, y1 - 10, C.mid, 10);
        for (r = 0; r < rows; r++) {
          var col2 = 0;
          for (j = 0; j < cols; j++) if (keep[r * cols + j]) {
            v = W[r * cols + j];
            rect(ctx, x0 + col2 * cw, y1 + r * chh, cw - 2, chh - 2, v >= 0 ? 'rgba(232,145,106,0.55)' : 'rgba(96,165,250,0.55)');
            txt(ctx, String(j % m), x0 + col2 * cw + (cw - 2) / 2, y1 + r * chh + 6.5, '#111', 9, 'center');
            col2++;
          }
        }
      } else txt(ctx, 'no fixed structure → stored as CSR/bitmap; dense tensor cores cannot skip the zeros', x0, y1 + 20, C.red, 10.5);
      var kept = keep.filter(Boolean).length, dense = rows * cols * 2, comp = kept * 2 + (pat === 'u50' ? kept * 2 + (rows + 1) * 4 : kept * Math.ceil(Math.log2(m)) / 8);
      var e24 = energyKept(W, pruneNM(W, rows, cols, 2, 4)), eU = energyKept(W, pruneUnstructured(W, 0.5));
      out.textContent = 'kept                ' + kept + ' / ' + (rows * cols) + '  (' + fmt(100 * kept / (rows * cols), 0) + ' %)' +
        '\nweight energy kept  ' + fmt(100 * energyKept(W, keep), 1) + ' %   (2:4 ' + fmt(100 * e24, 1) + ' % vs unstructured ' + fmt(100 * eU, 1) + ' %)' +
        '\nstorage (BF16)      ' + dense + ' B dense → ' + fmt(comp, 0) + ' B  (' + fmt(dense / comp, 2) + '×)' +
        '\nsparse tensor core  ' + (pat === '2:4' ? 'yes: 2× peak math on A100/H100/B200' : pat === 'u50' ? 'no speedup on tensor cores' : 'not a hardware pattern on NVIDIA GPUs');
    }
    pS.addEventListener('change', draw);
    newB.onclick = function () { seed++; fresh(); };
    fresh();
  }

  // ============================================================ 7. quantization
  function labQuant(root) {
    var body = frame(root, 'Lab · Quantization error — bits, scale granularity and outliers');
    var rows = 64, cols = 256, c = canvas(560, 290); body.appendChild(c); var ctx = c.getContext('2d');
    var p = panel(body), ct = controls(p);
    var schS = select(ct, 'scheme', [['int', 'INT (uniform)'], ['mxfp4', 'MXFP4 (E2M1, g=32, 2^k scale)'], ['nvfp4', 'NVFP4 (E2M1, g=16, FP8 scale)']], 'int');
    var bS = select(ct, 'bits', [8, 6, 4, 3, 2], 4);
    var gS = select(ct, 'scale per', [['tensor', 'tensor'], ['channel', 'row (channel)'], ['128', 'group 128'], ['32', 'group 32']], '128');
    var ct2 = controls(p), asym = checkbox(ct2, ' asymmetric (zero-point)', false), outl = checkbox(ct2, ' outliers (0.3 % of values × 15)', true), newB = button(ct2, 'New tensor');
    var out = readout(p);
    hint(p, 'A 64×256 tensor drawn from a Gaussian, optionally heavy-tailed: 0.3 % of values scaled 15×, like the rare large weights and activations in LLMs. One outlier stretches the scale of every value that shares it, so smaller groups confine the damage. Bars compare signal-to-quantization-noise across granularities at the chosen bit width; every 6 dB is one bit of effective precision.');
    var seed = 11, X;
    function fresh() {
      var R = rng(seed); X = new Float64Array(rows * cols);
      for (var i = 0; i < rows * cols; i++) X[i] = gaussian(R) * (outl.checked && R() < 0.003 ? 15 : 1);
      draw();
    }
    function run(scheme, gran) { var g = gran === 'tensor' || gran === 'channel' ? gran : +gran; return quantize(X, { rows: rows, cols: cols, scheme: scheme, bits: +bS.value, gran: g, asym: asym.checked }); }
    function draw() {
      clear(ctx, c);
      var isInt = schS.value === 'int'; bS.disabled = !isInt; gS.disabled = !isInt; asym.disabled = !isInt;
      var cur = run(schS.value, gS.value);
      // histogram of row 0 values with quantization levels (per-tensor INT case shows global grid)
      var hx = 20, hy = 20, hw = 300, hh = 200, amax = 0; for (var i = 0; i < X.length; i++) amax = Math.max(amax, Math.abs(X[i]));
      var lim = Math.min(amax, 5), bins = new Array(60).fill(0);
      for (i = 0; i < X.length; i++) { var b = Math.floor((X[i] + lim) / (2 * lim) * 60); if (b >= 0 && b < 60) bins[b]++; }
      var bm = Math.max.apply(null, bins);
      txt(ctx, 'value histogram (|x| ≤ ' + fmt(lim, 1) + ') and reconstructed levels', hx, 10, C.mid, 10);
      bins.forEach(function (v, k) { rect(ctx, hx + k * hw / 60, hy + hh - hh * v / bm, hw / 60 - 1, hh * v / bm, '#3a3a3a'); });
      var levels = {}; for (i = 0; i < cols; i++) levels[cur.out[i].toFixed(4)] = cur.out[i];
      Object.keys(levels).forEach(function (k) { var v = levels[k]; if (Math.abs(v) <= lim) { var X0 = hx + (v + lim) / (2 * lim) * hw; line(ctx, X0, hy, X0, hy + hh, 'rgba(232,145,106,0.55)'); } });
      txt(ctx, '−' + fmt(lim, 1), hx, hy + hh + 10, C.mid, 9.5); txt(ctx, '+' + fmt(lim, 1), hx + hw, hy + hh + 10, C.mid, 9.5, 'right');
      txt(ctx, 'levels used by row 0', hx, hy + hh + 26, C.accHi, 9.5);
      // SQNR bars by granularity
      var bx = 350, bw = 190, opts = isInt ? [['tensor', 'tensor'], ['channel', 'row'], ['128', 'g128'], ['32', 'g32']] : [['mx', 'MXFP4'], ['nv', 'NVFP4']];
      var res = opts.map(function (o) { return isInt ? run('int', o[0]) : run(o[0] === 'mx' ? 'mxfp4' : 'nvfp4', 0); });
      var best = Math.max.apply(null, res.map(function (r) { return r.sqnr; })), top = Math.max(40, best + 5);
      txt(ctx, 'SQNR (dB) by scale granularity', bx, 10, C.mid, 10);
      res.forEach(function (r, k) {
        var y = 30 + k * 46, w = bw * Math.max(0, r.sqnr) / top, sel = isInt ? opts[k][0] === gS.value : (opts[k][0] === 'mx') === (schS.value === 'mxfp4');
        rect(ctx, bx, y, w, 22, sel ? C.acc : '#555'); txt(ctx, opts[k][1], bx, y + 32, C.mid, 9.5);
        txt(ctx, fmt(r.sqnr, 1) + ' dB · ' + fmt(r.bitsPer, 2) + ' b/w', bx + w + 4, y + 11, sel ? C.accHi : C.text, 9.5);
      });
      out.textContent = 'scheme             ' + (isInt ? 'INT' + bS.value + (asym.checked ? ' asym' : ' sym') + ', scale per ' + (gS.value === 'tensor' ? 'tensor' : gS.value === 'channel' ? 'row' : 'group of ' + gS.value) : schS.value.toUpperCase()) +
        '\nSQNR               ' + fmt(cur.sqnr, 1) + ' dB   (≈ ' + fmt(cur.sqnr / 6.02, 1) + ' effective bits)' +
        '\nmax |error|        ' + fmt(cur.maxErr, 3) +
        '\nstorage incl. scales ' + fmt(cur.bitsPer, 3) + ' bits/weight  →  ' + fmt(16 / cur.bitsPer, 2) + '× smaller than BF16';
    }
    [schS, bS, gS, asym].forEach(function (x) { x.addEventListener('change', draw); });
    outl.addEventListener('change', fresh);
    newB.onclick = function () { seed++; fresh(); };
    fresh();
  }

  // ============================================================ 8. atlas
  function labAtlas(root, data) {
    var body = frame(root, 'Atlas · ' + data.chips.length + ' accelerators, one table, one scatter');
    var wrap = el('div', { style: 'flex-basis:100%;min-width:0' }); body.appendChild(wrap);
    var ct = el('div', { 'class': 'atlas-controls' }); wrap.appendChild(ct);
    var dS = select(ct, 'y-axis', [['best', 'lowest supported precision'], ['bf16', 'BF16/FP16'], ['fp8', 'FP8'], ['fp4', 'FP4']], 'best');
    var stS = select(ct, 'status', [['all', 'all'], ['shipping', 'shipping'], ['next', 'ramping + announced']], 'all');
    var sram = checkbox(ct, ' include SRAM-only designs', false);
    var vend = el('div', { 'class': 'chip-row' }); wrap.appendChild(vend);
    var vendors = {}; data.chips.forEach(function (x) { vendors[x.vendor] = true; });
    var on = {}; Object.keys(vendors).forEach(function (v) { on[v] = true; var b = button(vend, v); b.style.borderColor = VENDOR_COLOR[v]; b.onclick = function () { on[v] = !on[v]; b.classList.toggle('off', !on[v]); render(); }; });
    var c = canvas(760, 420); wrap.appendChild(c); var ctx = c.getContext('2d');
    var tip = readout(wrap); tip.textContent = 'Hover a point for details.';
    var tbl = el('table', { 'class': 'atlas-table' }); var tw = el('div', { style: 'overflow-x:auto' }); tw.appendChild(tbl); wrap.appendChild(tw);
    var cols = [['name', 'Chip'], ['vendor', 'Vendor'], ['year', 'Year'], ['status', 'Status'], ['mem_gb', 'Mem GB'], ['mem_type', 'Type'], ['mem_tbps', 'TB/s'], ['bf16_tf', 'BF16 TF'], ['fp8_tf', 'FP8 TF'], ['fp4_tf', 'FP4 TF'], ['ridge', 'Ridge FLOP/B'], ['tdp_w', 'TDP W'], ['node', 'Process']];
    var sortKey = 'year', sortDir = 1, pts = [];
    function best(x) { return x.fp4_tf ? ['fp4', x.fp4_tf] : x.fp8_tf ? ['fp8', x.fp8_tf] : x.bf16_tf ? ['bf16', x.bf16_tf] : x.int8_tops ? ['int8', x.int8_tops] : [null, null]; }
    function yval(x) { return dS.value === 'best' ? best(x)[1] : x[DT_FIELD[dS.value]]; }
    function ridge(x) { var b = x.fp8_tf || x.bf16_tf || x.fp4_tf; return b && x.mem_tbps ? b / x.mem_tbps : null; }
    function visible(x) {
      if (!on[x.vendor]) return false;
      if (!sram.checked && x.mem_tbps >= 50) return false;
      if (stS.value === 'shipping' && x.status !== 'shipping') return false;
      if (stS.value === 'next' && x.status === 'shipping') return false;
      return true;
    }
    function render() {
      clear(ctx, c);
      var box = { x: 72, y: 16, w: 660, h: 350 }, xr = sram.checked ? [0.5, 50000] : [0.5, 40], yr = [50, 60000];
      var ax = logAxes(ctx, box, xr, yr, 'memory bandwidth  (TB/s)', 'dense tensor FLOP/s');
      // iso-ridge lines
      [100, 300, 1000, 3000].forEach(function (rI) {
        var x1 = Math.max(xr[0], yr[0] / rI), x2 = Math.min(xr[1], yr[1] / rI);
        if (x2 > x1) { line(ctx, ax.x(x1), ax.y(x1 * rI), ax.x(x2), ax.y(x2 * rI), 'rgba(232,145,106,0.22)', 1, [4, 4]); txt(ctx, 'ridge ' + rI, ax.x(x2) - 2, ax.y(x2 * rI) + 10, 'rgba(232,145,106,0.6)', 9, 'right'); }
      });
      pts = [];
      data.chips.forEach(function (x) {
        var y = yval(x); if (!visible(x) || !y) return;
        var X = ax.x(Math.max(xr[0], Math.min(xr[1], x.mem_tbps))), Y = ax.y(y), col = VENDOR_COLOR[x.vendor] || '#999';
        dot(ctx, X, Y, x.status === 'shipping' ? 5.5 : 5, x.status === 'shipping' ? col : null, x.status === 'shipping' ? '#000' : col);
        pts.push({ X: X, Y: Y, chip: x });
      });
      // labels: right of the dot, else left, else above/below; skip if every slot collides (hover still works)
      var placed = [];
      ctx.font = '9.5px JetBrains Mono, monospace';
      pts.slice().sort(function (a, b) { return b.chip.year - a.chip.year; }).forEach(function (pt) {
        var label = pt.chip.name.replace(/ \(.*\)/, '').replace('Instinct ', '').replace('GeForce ', ''), w = ctx.measureText(label).width, h = 11;
        var cands = [[pt.X + 8, pt.Y - h / 2], [pt.X - 8 - w, pt.Y - h / 2], [pt.X - w / 2, pt.Y - 8 - h], [pt.X - w / 2, pt.Y + 8]];
        for (var i = 0; i < cands.length; i++) {
          var r = { x: cands[i][0], y: cands[i][1], w: w, h: h }, hit = r.x < box.x || r.x + r.w > box.x + box.w;
          for (var j = 0; j < placed.length && !hit; j++) { var q = placed[j]; hit = r.x < q.x + q.w && q.x < r.x + r.w && r.y < q.y + q.h && q.y < r.y + r.h; }
          for (j = 0; j < pts.length && !hit; j++) { var d = pts[j]; hit = d !== pt && d.X > r.x - 4 && d.X < r.x + r.w + 4 && d.Y > r.y - 4 && d.Y < r.y + r.h + 4; }
          if (!hit) { placed.push(r); txt(ctx, label, r.x, r.y + h / 2, 'rgba(232,232,232,0.85)', 9.5); return; }
        }
      });
      txt(ctx, '● shipping   ○ ramping / announced   — dashed: constant ridge point (FLOP/B)', box.x, box.y + box.h + 40, C.mid, 9.5);
      // table
      var rows = data.chips.slice().sort(function (a, b) {
        var va = sortKey === 'ridge' ? ridge(a) : a[sortKey], vb = sortKey === 'ridge' ? ridge(b) : b[sortKey];
        if (va === null || va === undefined) return 1; if (vb === null || vb === undefined) return -1;
        return (va > vb ? 1 : va < vb ? -1 : 0) * sortDir;
      });
      var h = '<thead><tr>' + cols.map(function (k) { return '<th data-k="' + k[0] + '"' + (k[0] === sortKey ? ' class="sorted"' : '') + '>' + k[1] + (k[0] === sortKey ? (sortDir > 0 ? ' ▲' : ' ▼') : '') + '</th>'; }).join('') + '</tr></thead><tbody>';
      rows.forEach(function (x) {
        h += '<tr' + (visible(x) ? '' : ' class="dim"') + ' title="' + (x.notes || '').replace(/"/g, '&quot;') + '">';
        cols.forEach(function (k) {
          var v = k[0] === 'ridge' ? ridge(x) : x[k[0]];
          var num = typeof v === 'number';
          var s = v === null || v === undefined ? '—' : k[0] === 'ridge' ? Math.round(v) : k[0] === 'name' ? '<a href="' + x.source + '" target="_blank" rel="noopener">' + v + '</a>' : v;
          h += '<td' + (num ? ' class="n"' : '') + '>' + s + '</td>';
        });
        h += '</tr>';
      });
      tbl.innerHTML = h + '</tbody>';
      Array.prototype.forEach.call(tbl.querySelectorAll('th'), function (th) { th.onclick = function () { var k = th.getAttribute('data-k'); if (k === sortKey) sortDir = -sortDir; else { sortKey = k; sortDir = k === 'name' || k === 'vendor' ? 1 : -1; } render(); }; });
    }
    c.addEventListener('mousemove', function (ev) {
      var q = pointer(c, ev), bestP = null, bd = 196;
      pts.forEach(function (pt) { var d = (pt.X - q.x) * (pt.X - q.x) + (pt.Y - q.y) * (pt.Y - q.y); if (d < bd) { bd = d; bestP = pt; } });
      if (!bestP) return; var x = bestP.chip, b = best(x);
      tip.textContent = x.name + ' · ' + x.vendor + ' · ' + x.year + ' · ' + x.status + '\n' + x.mem_gb + ' GB ' + x.mem_type + ' @ ' + x.mem_tbps + ' TB/s · ' +
        (b[0] ? DT_NAME[b[0]] + ' ' + si(b[1] * 1e12, 'FLOP/s') : '') + (x.tdp_w ? ' · ' + x.tdp_w + ' W' : '') + '\n' + (x.notes || '');
    });
    [dS, stS, sram].forEach(function (x) { x.addEventListener('change', render); });
    render();
  }

  // ============================================================ boot
  var LABS = { roofline: labRoofline, bits: labBits, fit: labFit, systolic: labSystolic, wafer: labWafer, sparsity: labSparsity, quant: labQuant, atlas: labAtlas };
  var NEEDS_DATA = { roofline: true, fit: true, atlas: true };
  var dataPromise = null;
  function getData() {
    if (!dataPromise) {
      var base = (window.CHIPS_BASEURL || '').replace(/\/$/, '');
      dataPromise = fetch(base + '/assets/data/accelerators.json').then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); });
    }
    return dataPromise;
  }
  function boot() {
    var roots = document.querySelectorAll('.lab[data-lab]');
    Array.prototype.forEach.call(roots, function (root) {
      var name = root.getAttribute('data-lab'), fn = LABS[name];
      if (!fn) return;
      try {
        if (NEEDS_DATA[name]) {
          root.innerHTML = '<div class="lab-title">Loading accelerator data…</div>';
          getData().then(function (d) { fn(root, d); }, function () { root.innerHTML = '<div class="lab-noscript">Could not load assets/data/accelerators.json.</div>'; });
        } else fn(root);
      } catch (err) { root.innerHTML = '<div class="lab-noscript">This lab failed to start: ' + err.message + '</div>'; }
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
