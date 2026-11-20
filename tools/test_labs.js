// Checks the arithmetic behind assets/js/chips-labs.js against hand-derived values.
// Run:  node tools/test_labs.js
'use strict';
const assert = require('assert');
const L = require('../assets/js/chips-labs.js');
const F = L.FORMATS;
let n = 0;
function eq(a, b, msg) { assert.strictEqual(a, b, msg); n++; }
function near(a, b, tol, msg) { assert.ok(Math.abs(a - b) <= tol, `${msg}: got ${a}, want ${b} ± ${tol}`); n++; }

// ---- floating-point formats: round-to-nearest-even, ranges, saturation
eq(L.fpRound(F.bf16, Math.PI).value, 3.140625, 'BF16(pi)');
eq(L.fpRound(F.bf16, Math.PI).code, 0x4049, 'BF16(pi) bits');
eq(L.fpRound(F.fp16, 0.1).value, 0.0999755859375, 'FP16(0.1)');
eq(L.fpRound(F.fp16, 65504).value, 65504, 'FP16 max');
eq(L.fpRound(F.fp16, 65519).value, 65504, 'FP16 just below overflow threshold');
eq(L.fpRound(F.fp16, 65520).value, Infinity, 'FP16 overflow at max + ulp/2');
eq(L.fpRound(F.fp32, 0.1).value, Math.fround(0.1), 'FP32(0.1)');
eq(L.fpStats(F.e4m3).max, 448, 'E4M3 max');
eq(L.fpStats(F.e5m2).max, 57344, 'E5M2 max');
eq(L.fpStats(F.e3m2).max, 28, 'E3M2 max');
eq(L.fpStats(F.e2m3).max, 7.5, 'E2M3 max');
eq(L.fpStats(F.e2m1).max, 6, 'E2M1 max');
eq(L.fpStats(F.e4m3).minSub, Math.pow(2, -9), 'E4M3 min subnormal');
eq(L.fpStats(F.fp16).minNormal, Math.pow(2, -14), 'FP16 min normal');
eq(L.fpRound(F.e4m3, 0.1).value, 0.1015625, 'E4M3(0.1)');
eq(L.fpRound(F.e4m3, 500).value, 448, 'E4M3 saturates');
eq(L.fpRound(F.e4m3, -500).value, -448, 'E4M3 saturates negative');
eq(L.fpRound(F.e5m2, 1e6).value, Infinity, 'E5M2 overflows to inf (IEEE-like)');
eq(L.fpRound(F.e2m1, 2.5).value, 2, 'E2M1 tie 2.5 -> 2 (even code)');
eq(L.fpRound(F.e2m1, 5).value, 4, 'E2M1 tie 5 -> 4 (even code)');
eq(L.fpRound(F.e2m1, 0.74).value, 0.5, 'E2M1 0.74 -> 0.5');
eq(L.fpRound(F.e2m1, 100).value, 6, 'E2M1 saturates');
assert.ok(isNaN(L.fpDecode(F.e4m3, 0x7f)), 'E4M3 S.1111.111 is NaN'); n++;
eq(L.fpDecode(F.e4m3, 0x7e), 448, 'E4M3 S.1111.110 is 448');
eq(L.fpDecode(F.e5m2, 0x7c), Infinity, 'E5M2 has inf');
const e2m1 = []; for (let c = 0; c < 8; c++) e2m1.push(L.fpDecode(F.e2m1, c));
assert.deepStrictEqual(e2m1, [0, 0.5, 1, 1.5, 2, 3, 4, 6], 'E2M1 value set'); n++;

// ---- integer quantization
eq(L.intQuant(8, 1.0, 0.05).q, 20, 'INT8 1.0 / 0.05');
eq(L.intQuant(8, 10, 0.05).q, 127, 'INT8 clamps high');
eq(L.intQuant(8, -10, 0.05).code, 128, 'INT8 -128 two\'s complement');
eq(L.intQuant(4, 1.25, 0.5).q, 2, 'INT4 ties to even (2.5 -> 2)');
eq(L.intQuant(4, 1.75, 0.5).q, 4, 'INT4 ties to even (3.5 -> 4)');

// ---- roofline
near(L.intensity('gemm', 4096, 2), 4096 * 2 / 6, 1e-9, 'square GEMM intensity 2n/(3b)');
near(L.intensity('decode', 1, 2), 1, 0.001, 'batch-1 BF16 decode ~1 FLOP/B');
near(L.intensity('decode', 1, 0.5), 4, 0.01, 'batch-1 FP4 decode ~4 FLOP/B');
near(L.intensity('attn', 8, 2), 8, 1e-12, 'GQA-8 attention decode = 8 FLOP/B in BF16');
eq(L.attainable(989, 3.35, 1), 3.35, 'memory roof');
eq(L.attainable(989, 3.35, 1000), 989, 'compute roof');

// ---- will it fit
const M = L.MODELS;
eq(L.kvCacheBytes(M['llama3-70b'], 1, 1, 2), 327680, 'Llama 3.1 70B KV = 320 KiB/token (BF16)');
eq(L.kvCacheBytes(M['llama3-8b'], 1, 1, 2), 131072, 'Llama 3.1 8B KV = 128 KiB/token (BF16)');
eq(L.kvCacheBytes(M['dsv3-671b'], 1, 1, 2), 576 * 61 * 2, 'DeepSeek-V3 MLA latent cache');
const inf = L.memoryPlan({ m: M['llama3-70b'], mode: 'infer', wBytes: 2, kvBytes: 2, ctx: 8192, batch: 1, gpus: 1, ckpt: true, shard: true });
near(inf.weights / 1e9, 141.2, 0.01, '70B BF16 weights = 141.2 GB');
const tr = L.memoryPlan({ m: M['llama3-8b'], mode: 'train', wBytes: 2, kvBytes: 2, ctx: 4096, batch: 1, gpus: 8, ckpt: true, shard: true });
near((tr.weights + tr.grads + tr.optim) * 8 / 1e9, 16 * 8.03, 0.01, 'Adam mixed precision = 16 B/param');

// ---- wafer
const y = L.yieldModels(8.14, 0.1);
near(y.poisson, Math.exp(-0.814), 1e-12, 'Poisson yield');
near(y.poisson, 0.443, 0.001, 'H100-size die at D0 = 0.1 /cm2');
near(L.diesPerWaferFormula(300, 100), 640.2, 0.05, 'dies-per-wafer formula, 100 mm2');
const gd = L.grossDies(300, 3, Math.sqrt(814), Math.sqrt(814));
assert.ok(gd.n >= 56 && gd.n <= 70, 'gross dies for 814 mm2 near formula (' + gd.n + ')'); n++;

// ---- sparsity
const R = L.rng(5), w = []; for (let i = 0; i < 8 * 16; i++) w.push(L.gaussian(R));
const k24 = L.pruneNM(w, 8, 16, 2, 4);
for (let g = 0; g < w.length; g += 4) eq(k24.slice(g, g + 4).filter(Boolean).length, 2, '2 of every 4 kept');
assert.ok(L.energyKept(w, L.pruneUnstructured(w, 0.5)) >= L.energyKept(w, k24) - 1e-12, 'unstructured keeps at least as much energy'); n++;

// ---- quantization
const RQ = L.rng(9), x = new Float64Array(64 * 256); for (let i = 0; i < x.length; i++) x[i] = L.gaussian(RQ);
const q8 = L.quantize(x, { rows: 64, cols: 256, scheme: 'int', bits: 8, gran: 'tensor', asym: false });
near(q8.sqnr, 40, 2.5, 'INT8 per-tensor on N(0,1) ~ 40 dB');
const q4t = L.quantize(x, { rows: 64, cols: 256, scheme: 'int', bits: 4, gran: 'tensor', asym: false });
const q4g = L.quantize(x, { rows: 64, cols: 256, scheme: 'int', bits: 4, gran: 32, asym: false });
assert.ok(q4g.sqnr > q4t.sqnr, 'finer groups beat per-tensor at 4 bits'); n++;
near(q4g.bitsPer, 4.5, 1e-12, 'INT4 g32 with FP16 scale = 4.5 bits');
const mx = L.quantize(x, { rows: 64, cols: 256, scheme: 'mxfp4' }), nv = L.quantize(x, { rows: 64, cols: 256, scheme: 'nvfp4' });
eq(mx.bitsPer, 4.25, 'MXFP4 = 4.25 bits'); eq(nv.bitsPer, 4.5, 'NVFP4 = 4.5 bits');
assert.ok(nv.sqnr > mx.sqnr, 'NVFP4 (finer, non-power-of-two scale) beats MXFP4: ' + nv.sqnr.toFixed(1) + ' vs ' + mx.sqnr.toFixed(1)); n++;

// ---- systolic
let macs = 0; const N = 4;
for (let t = 0; t <= 3 * N - 3; t++) for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (L.systolicK(t, i, j, N) >= 0) macs++;
eq(macs, N * N * N, 'every MAC happens exactly once in 3N-2 cycles');

console.log(`ok — ${n} checks passed`);
console.log(`  INT8 per-tensor ${q8.sqnr.toFixed(1)} dB · INT4 tensor ${q4t.sqnr.toFixed(1)} dB · INT4 g32 ${q4g.sqnr.toFixed(1)} dB · MXFP4 ${mx.sqnr.toFixed(1)} dB · NVFP4 ${nv.sqnr.toFixed(1)} dB · gross dies(814 mm²) ${gd.n}`);
