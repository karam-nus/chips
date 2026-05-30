---
title: "Chapter 23 — Number Formats Deep Dive"
---

[← Back to Table of Contents](./README.md)

# Chapter 23 — Number Formats Deep Dive

[Chapter 3](./03_number_representation.md) built floating point from first principles: the sign/exponent/mantissa layout, the bias, the hidden bit, subnormals, and the eternal tension between **precision** (mantissa bits) and **dynamic range** (exponent bits). This chapter is where that foundation pays off. We are going to take a guided tour of *every* numeric format that matters in modern ML hardware — from the venerable FP32 your model was probably born in, down through BF16, FP16, the FP8 twins, FP6, FP4, the integer family (INT8 → INT1), all the way to the exotic block-scaled formats (MXFP, NVFP4) that the Blackwell generation is built around.

This is the **keystone chapter** of the guide. Quantization (Chapter 25), the silicon consequences of format choice (Chapter 24), and the practitioner decision framework (Chapter 30) all stand on what you learn here. By the end you should be able to look at a dtype name like `e4m3` or `mxfp4` and *immediately* know its bit layout, its maximum value, its smallest normal, its relative precision, and — most importantly — *why someone bothered to invent it.*

> **The one-sentence version:** Every ML number format is a single answer to one question — given a fixed bit budget, how do you split bits between **dynamic range** (exponent), **precision** (mantissa), and the option of not spending them at all (fewer bits = less memory, bandwidth, and silicon) — and the entire history of FP32 → FP4 → microscaling is the field finding ever-cheaper answers without breaking the model.

---

## The Eternal Triangle: Range vs Precision vs Bit-Width

Before the formats, internalize the trade space. Every format lives at a point inside a triangle whose three corners pull against each other:

<div class="diagram">
<div class="diagram-title">The Three Forces Every Format Balances</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card green">
    <div class="card-icon">📏</div>
    <div class="card-title">Dynamic Range</div>
    <div class="card-desc">Set by <strong>exponent bits</strong>. Ratio of largest to smallest representable value. Too little → gradients overflow to ∞ or underflow to 0. Costs no precision, just bits.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🔬</div>
    <div class="card-title">Precision</div>
    <div class="card-desc">Set by <strong>mantissa bits</strong>. How finely you can distinguish nearby values (ULP). Too little → rounding error swamps small weight updates and accumulations.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">🗜️</div>
    <div class="card-title">Bit-Width</div>
    <div class="card-desc">Total bits = bytes/param. Fewer bits → less memory, more bandwidth, more & cheaper silicon units (Ch 24). This is the prize everyone is chasing.</div>
  </div>
</div>
</div>

You cannot maximize all three. A 16-bit format has 16 bits to divide between exponent and mantissa (plus one sign bit). BF16 spends them on range; FP16 spends them on precision. Drop to 8 bits and the squeeze gets brutal: E4M3 sacrifices range for precision, E5M2 does the reverse. Drop to 4 bits and you barely have a number at all — which is exactly why **block scaling** (the back half of this chapter) was invented: it lets a tiny 4-bit element borrow its dynamic range from a shared scale, so the 4 bits can be spent almost entirely on precision.

Recall the value formula and two key quantities from Chapter 3, which we will compute for every format:

$$V = (-1)^S \times 1.M \times 2^{\,E - \text{bias}}, \qquad \text{bias} = 2^{k-1}-1 \ \text{for a } k\text{-bit exponent}$$

- **Max normal** $= (2 - 2^{-m}) \times 2^{\,e_{\max}}$ — set by the *exponent* (range).
- **Relative precision** $\approx 2^{-(m+1)}$ — the gap between 1.0 and the next representable value (one ULP at 1.0), set by the *mantissa*.

Keep both formulas in view; the rest of the chapter is just plugging numbers in.

---

## The Floating-Point Family, Format by Format

For each format we give the bit layout, the field split, byte size, dynamic range, smallest normal/subnormal, relative precision, and — the part that matters — what it is actually *for*.

### FP64 — Double Precision (1-11-52, 8 bytes)

The gold standard of scientific computing, and almost never used in ML.

```text
 63 62           52 51                                                  0
  S  EEEEEEEEEEE     MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
  ↑  ←  11 bits  →   ←──────────────────  52 bits  ──────────────────────→
```

| Field | Bits | Meaning |
|-------|:----:|---------|
| Sign | 1 | $(-1)^S$ |
| Exponent | 11 | $e = E - 1023$ |
| Mantissa | 52 | $1.M$ (hidden bit) |

- **Max normal:** $\approx 1.8 \times 10^{308}$  **Min normal:** $\approx 2.2 \times 10^{-308}$
- **Relative precision:** $2^{-53} \approx 1.1 \times 10^{-16}$ (~15–16 decimal digits)
- **ML use:** essentially none in the hot path. Occasionally for reference/"golden" results, numerically sensitive reductions, or scientific HPC. Tensor cores do not accelerate FP64 for ML; on a GPU it is the *slowest* path. Mentioned here mainly as the upper anchor of the range/precision scale.

### FP32 — Single Precision (1-8-23, 4 bytes)

The default `torch.float32`, and for two decades the only format anyone trained in.

```text
 31 30        23 22                                            0
  S  EEEEEEEE    MMMMMMMMMMMMMMMMMMMMMMM
  ↑  ← 8 bits →  ←─────────  23 bits  ─────────→
```

| Field | Bits | Meaning |
|-------|:----:|---------|
| Sign | 1 | $(-1)^S$ |
| Exponent | 8 | $e = E - 127$ |
| Mantissa | 23 | $1.M$ |

- **Max normal:** $\approx 3.4 \times 10^{38}$  **Min normal:** $\approx 1.18 \times 10^{-38}$  **Min subnormal:** $\approx 1.4 \times 10^{-45}$
- **Relative precision:** $2^{-24} \approx 6.0 \times 10^{-8}$ (~7 decimal digits)
- **ML use:** master weights, optimizer state, loss accumulation, the safe fallback. Modern training keeps an FP32 *master copy* of weights even when the matmuls run in BF16/FP8 — the high precision matters for accumulating millions of tiny gradient steps (Chapter 24's "FP32 accumulate" theme). FP32 is the **reference point** for "is my lower-precision run still correct?"

### TF32 — TensorFloat-32 (1-8-10, 19 bits used, NVIDIA)

The most misleadingly named format in the field. TF32 is **not** a storage format and it is **not** 32 bits of data. It is an *internal compute mode* introduced with NVIDIA Ampere (A100) for the tensor cores.

```text
        used by the multiplier (19 bits)
 31 30        23 22         13 12          0
  S  EEEEEEEE    MMMMMMMMMM    xxxxxxxxxxxxx   ← stored as a normal 32-bit word…
  ↑  ← 8 bits →  ← 10 bits →   ← 13 bits ignored by the TF32 multiplier →
```

| Field | Bits | Meaning |
|-------|:----:|---------|
| Sign | 1 | $(-1)^S$ |
| Exponent | 8 | **same 8-bit exponent as FP32** → identical range |
| Mantissa | 10 | only the top 10 mantissa bits feed the multiplier |

- **Range:** identical to FP32 ($\approx 10^{-38}$ to $10^{38}$) — it keeps all 8 exponent bits.
- **Precision:** $2^{-11} \approx 4.9 \times 10^{-4}$ — same mantissa as FP16 (~3 decimal digits).
- **ML use:** a "free" ~8× tensor-core speedup over true FP32 with minimal code change — PyTorch exposes it via `torch.backends.cuda.matmul.allow_tf32`. The insight: **the multiplier is the expensive part** (Chapter 24 shows multiplier area scales ~quadratically with mantissa width). By truncating the mantissa to 10 bits *inside the multiplier* while keeping FP32's exponent and accumulating in FP32, you get most of FP32's behavior at a fraction of the silicon cost. The data is still stored as 32-bit words; only the multiply is lossy. This is your first concrete example of **hardware/numerics co-design** (Chapter 24).

### FP16 — Half Precision (1-5-10, 2 bytes)

The original "let's go half-width" format, IEEE-754 standard, the workhorse of pre-2020 mixed-precision training.

```text
 15 14    10 9             0
  S  EEEEE    MMMMMMMMMM
  ↑  ←5 b→    ←─ 10 bits ─→
```

| Field | Bits | Meaning |
|-------|:----:|---------|
| Sign | 1 | $(-1)^S$ |
| Exponent | 5 | $e = E - 15$ |
| Mantissa | 10 | $1.M$ |

- **Max normal:** $\mathbf{65504}$  **Min normal:** $\approx 6.1 \times 10^{-5}$  **Min subnormal:** $\approx 5.96 \times 10^{-8}$
- **Relative precision:** $2^{-11} \approx 4.9 \times 10^{-4}$ (~3 decimal digits)
- **ML use:** inference and (with care) training. **The range problem:** FP16's max value is only **65504**. Activations and especially gradients in a large network routinely exceed that and overflow to `+inf`, which then poisons the whole tensor. The historical fix is **loss scaling** — multiply the loss by a big constant (e.g. 1024) before backprop to push tiny gradients up into FP16's normal range, then divide it back out. It works but it is fiddly, requires dynamic overflow detection, and is exactly the pain that BF16 was designed to eliminate. FP16 *does* have one bit more mantissa than BF16, so where range is controlled (inference, normalized activations) it can be slightly more accurate.

### BF16 — Brain Float 16 (1-8-7, 2 bytes)

The format that became the training default. The trick is almost embarrassingly simple: **take FP32 and chop off the bottom 16 mantissa bits.**

```text
 15 14     7 6        0
  S  EEEEEEEE  MMMMMMM
  ↑  ← 8 bits → ← 7 b →
```

| Field | Bits | Meaning |
|-------|:----:|---------|
| Sign | 1 | $(-1)^S$ |
| Exponent | 8 | **same 8-bit exponent as FP32** → $e = E - 127$ |
| Mantissa | 7 | $1.M$ |

- **Max normal:** $\approx 3.39 \times 10^{38}$ (same as FP32!)  **Min normal:** $\approx 1.18 \times 10^{-38}$
- **Relative precision:** $2^{-8} \approx 3.9 \times 10^{-3}$ (~2 decimal digits)
- **ML use:** **the default training dtype on every modern accelerator.** Because BF16 keeps FP32's *entire* exponent field, any value that fits in FP32 fits in BF16 — no overflow, **no loss scaling needed.** You trade away mantissa bits (7 vs FP16's 10), so each individual value is coarser, but training is remarkably robust to that: stochastic gradient descent is averaging over huge batches, and the FP32 accumulator (Chapter 24) absorbs the rounding. Converting FP32↔BF16 is almost free — it is literally truncation (plus rounding) of the low half-word.

> **Why BF16 won.** In training, **dynamic range beats precision.** Gradients span ~10–20 orders of magnitude across a network; weights and activations span fewer but still many. A format that overflows (FP16) forces you to babysit loss scaling and still risks `NaN` blowups. A format with FP32's range (BF16) just works. The lost mantissa precision is recovered by accumulating in FP32. This single insight — *range in the storage format, precision in the accumulator* — is the template for every lower-precision format that followed.

### FP8 — The Two Eight-Bit Floats (1 byte)

At 8 bits there is no single right answer, so the OCP (Open Compute Project) FP8 standard defines **two** formats, and hardware (NVIDIA Hopper/Blackwell) supports both. The choice is the range-vs-precision trade-off made naked.

#### E4M3 (1-4-3) — the precision-leaning FP8

```text
  7  6     3 2   0
  S  EEEE    MMM
  ↑  ←4 b→   ←3 b→
```

| Field | Bits | Meaning |
|-------|:----:|---------|
| Sign | 1 | $(-1)^S$ |
| Exponent | 4 | bias 7 |
| Mantissa | 3 | $1.M$ |

- **Max normal:** $\mathbf{448}$  **Min normal:** $2^{-6} \approx 0.0156$  **Min subnormal:** $2^{-9} \approx 0.00195$
- **Relative precision:** $2^{-4} \approx 6.25 \times 10^{-2}$ (only ~16 distinct values per binade)
- **Note:** the OCP `e4m3fn` variant ("finite") reclaims the would-be-infinity encodings to extend max to 448 and has only one NaN, no ∞. This is `torch.float8_e4m3fn`.
- **ML use:** **forward pass** — weights and activations, where values are reasonably bounded and you want as much precision as 8 bits allows.

#### E5M2 (1-5-2) — the range-leaning FP8

```text
  7  6      2 1 0
  S  EEEEE    MM
  ↑  ← 5 b →  ←2b→
```

| Field | Bits | Meaning |
|-------|:----:|---------|
| Sign | 1 | $(-1)^S$ |
| Exponent | 5 | bias 15 (same exponent field as FP16) |
| Mantissa | 2 | $1.M$ |

- **Max normal:** $\mathbf{57344}$  **Min normal:** $\approx 6.1 \times 10^{-5}$  **Min subnormal:** $\approx 1.5 \times 10^{-5}$
- **Relative precision:** $2^{-3} = 0.125$ (only ~8 distinct values per binade)
- **ML use:** **gradients in the backward pass**, where dynamic range matters more than precision (same reasoning as BF16 vs FP16, now at 8 bits). `torch.float8_e5m2`.

<div class="diagram">
<div class="diagram-title">The FP8 Trade-off: Same 8 Bits, Opposite Bets</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">E4M3 — precision</div>
    <ul>
      <li>4 exp / 3 mantissa</li>
      <li>Max 448, finer steps</li>
      <li>Forward pass (weights, activations)</li>
      <li>Overflows easily → needs scaling</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">E5M2 — range</div>
    <ul>
      <li>5 exp / 2 mantissa</li>
      <li>Max 57344, coarser steps</li>
      <li>Backward pass (gradients)</li>
      <li>Handles gradient spikes</li>
    </ul>
  </div>
</div>
</div>

> The standard FP8 training recipe uses **E4M3 forward, E5M2 backward** — precision where values are tame, range where they spike. This mirror of the BF16/FP16 lesson is no accident; it is the same trade-off pushed down another 8 bits.

### FP6 — Six-Bit Floats (E3M2 / E2M3)

A middle ground used in microscaling (MXFP6). Two variants, again range vs precision.

```text
E3M2:  S EEE MM   (1-3-2)        E2M3:  S EE MMM   (1-2-3)
```

| Format | Layout | Max normal | Min normal | Rel. precision | Leans |
|--------|:------:|:----------:|:----------:|:--------------:|-------|
| FP6 E3M2 | 1-3-2 | 28 | $2^{-2}=0.25$ | $2^{-3}=0.125$ | range |
| FP6 E2M3 | 1-2-3 | 7.5 | $2^{-1}=0.5$ | $2^{-4}=0.0625$ | precision |

- **ML use:** an accuracy/size compromise between FP8 and FP4 in microscaling formats. FP6 is awkward for memory (6 bits doesn't pack into bytes cleanly) but offers a meaningfully better accuracy/footprint point than FP4 for sensitive layers.

### FP4 — Four-Bit Float (E2M1, 0.5 bytes)

The current frontier of "how few bits can a *float* survive on." Only **16 possible values** total.

```text
  3  2 1 0
  S  EE M
  ↑  ↑  ↑
 sign | mantissa (1 bit)
   exponent (2 bits, bias 1)
```

| Field | Bits | Meaning |
|-------|:----:|---------|
| Sign | 1 | $(-1)^S$ |
| Exponent | 2 | bias 1 |
| Mantissa | 1 | $1.M$ |

- **Max normal:** $\mathbf{6.0}$  **Min normal:** $1.0$  **Min subnormal:** $0.5$
- **The complete value set** (E2M1, ignoring sign): $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$ — that's it. With sign, 15 distinct values (two zeros).
- **ML use:** essentially **never used alone** — its range is far too small (max 6.0) and its precision far too coarse for raw weights. FP4 only becomes viable inside a **block-scaled** format (MXFP4, NVFP4) where a shared scale supplies the dynamic range and the 4 bits encode only the *shape* of values within a small block. This is the punchline of the second half of the chapter.

---

## The Integer Family: INT8 → INT1 and Ternary

Integers have **no exponent** — they are pure precision with a fixed step size. Their "dynamic range" comes entirely from an external **scale factor** (the quantization scale, Chapter 25). Recall from Chapter 3 that N-bit two's complement spans $[-2^{N-1}, 2^{N-1}-1]$.

| Format | Bits | Levels | Range (signed) | Step (uniform) | ML use |
|--------|:----:|:------:|:--------------:|:--------------:|--------|
| INT8 | 8 | 256 | −128 … 127 | constant | the quantization workhorse; mature kernels everywhere |
| INT4 | 4 | 16 | −8 … 7 | constant | weight-only LLM quant (GPTQ/AWQ); 4× smaller than FP16 |
| INT2 | 2 | 4 | −2 … 1 | constant | aggressive research quant; big accuracy hit |
| INT1 (binary) | 1 | 2 | {−1, +1} | n/a | binary neural networks; matmul → XNOR+popcount |
| Ternary | ~1.6 | 3 | {−1, 0, +1} | n/a | BitNet-style; the 0 enables sparsity *and* sign |

```text
INT8 (two's complement):   S MMMMMMM        value = signed integer × scale
INT4:                      S MMM            value = signed integer × scale
Binary:                    b                value = +1 if b=1 else -1
Ternary:                   {-1, 0, +1}      ~1.58 bits of information (log2 3)
```

Key facts that drive ML decisions:

- **Uniform step.** Unlike floats, integers have a *constant* gap between values. This is great for values clustered around zero with a known max (post-activation, weights), and bad for values spanning many orders of magnitude (raw gradients). That is precisely why **weights** quantize beautifully to INT4 but **activations** (with outliers) are harder.
- **The asymmetry trap.** INT8 spans −128…127: the magnitude of the most-negative value exceeds the most-positive. Naive *symmetric* quantization wastes the −128 slot or clips it; some schemes use −127…127 for an exact symmetric zero (Chapter 3, Chapter 25).
- **Ternary {−1, 0, +1}** carries $\log_2 3 \approx 1.58$ bits. The middle value 0 is special: it lets a single representation express *both* sign and structured sparsity, which is why BitNet-style "1.58-bit" LLMs are an active frontier. Multiplies degenerate to add/subtract/skip — almost no multiplier needed at all (Chapter 24).
- **Binary (INT1)** turns the dot product into **XNOR + popcount**: multiply becomes a 1-gate XNOR, accumulate becomes counting set bits. Astonishingly cheap in silicon, but the accuracy cost is severe for general models.

---

## Master Comparison Table

The whole zoo on one page. (Subnormal/precision values are for the bare formats; block-scaled formats add a shared scale — see below.)

| Format | Bytes | Layout (S-E-M) | Exp bias | Max normal | Min normal | Rel. precision (ULP@1) | Typical ML use |
|--------|:-----:|:--------------:|:--------:|:----------:|:----------:|:----------------------:|----------------|
| FP64 | 8 | 1-11-52 | 1023 | $1.8\times10^{308}$ | $2.2\times10^{-308}$ | $1.1\times10^{-16}$ | reference / HPC |
| FP32 | 4 | 1-8-23 | 127 | $3.4\times10^{38}$ | $1.2\times10^{-38}$ | $6.0\times10^{-8}$ | master weights, accumulation |
| TF32 | (4) | 1-8-10 | 127 | $3.4\times10^{38}$ | $1.2\times10^{-38}$ | $4.9\times10^{-4}$ | A100+ tensor-core matmul mode |
| BF16 | 2 | 1-8-7 | 127 | $3.4\times10^{38}$ | $1.2\times10^{-38}$ | $3.9\times10^{-3}$ | **default training dtype** |
| FP16 | 2 | 1-5-10 | 15 | $65504$ | $6.1\times10^{-5}$ | $4.9\times10^{-4}$ | inference, legacy training |
| FP8 E4M3 | 1 | 1-4-3 | 7 | $448$ | $1.6\times10^{-2}$ | $6.25\times10^{-2}$ | FP8 forward pass |
| FP8 E5M2 | 1 | 1-5-2 | 15 | $57344$ | $6.1\times10^{-5}$ | $0.125$ | FP8 backward (gradients) |
| FP6 E3M2 | 0.75 | 1-3-2 | 3 | $28$ | $0.25$ | $0.125$ | MXFP6 (range) |
| FP6 E2M3 | 0.75 | 1-2-3 | 1 | $7.5$ | $0.5$ | $0.0625$ | MXFP6 (precision) |
| FP4 E2M1 | 0.5 | 1-2-1 | 1 | $6.0$ | $1.0$ | $0.5$ | block-scaled only (MXFP4/NVFP4) |
| INT8 | 1 | signed int | — | $127\times s$ | $s$ | $s$ (uniform) | quantized inference |
| INT4 | 0.5 | signed int | — | $7\times s$ | $s$ | $s$ (uniform) | weight-only LLM quant |
| INT2 | 0.25 | signed int | — | $1\times s$ | $s$ | $s$ (uniform) | research |
| Ternary | ~0.2 | {−1,0,1} | — | $s$ | — | — | BitNet 1.58-bit |
| INT1 | 0.125 | {−1,+1} | — | $s$ | — | — | binary nets (XNOR) |

> Read this table as a map of the triangle. Going *down* the float rows, bytes shrink (the prize) while both max-value and precision degrade. The integer rows have *no* range column worth speaking of — their range is entirely on loan from the scale $s$. Holding that thought leads directly to the most important idea in the chapter.

---

## Why Per-Tensor Scaling Breaks at Low Bit-Widths

Everything above assumed each number stands alone. In quantization (Chapter 25) you instead store integers (or tiny floats) plus a **scale** that maps them back to real values: $x \approx s \times q$. The question is: **how many values share one scale?**

The classic answer is **per-tensor** (one scale for a whole weight matrix) or **per-channel** (one per row/column). With INT8 this is fine — 256 levels is enough headroom that one scale can cover a tensor whose values span maybe 2–3 orders of magnitude.

It falls apart at 4 bits. Here is the problem in one picture: suppose a 32-element block of weights is mostly around ±0.1 but contains one **outlier** at 5.0. A single scale must be large enough to represent 5.0 — but then every small value rounds to a multiple of a coarse step, and with only 16 levels (FP4) the small values collapse to **0 or one or two coarse buckets**. The outlier "steals" all the dynamic range from its neighbors.

<div class="diagram">
<div class="diagram-title">One Scale for Many Values: the Outlier Problem</div>
<div class="flow">
  <div class="flow-node red wide">Per-tensor scale must cover the largest |value| (the outlier)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Step size = max/levels becomes huge with only 16 (FP4) or 256 (INT8) levels</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Small values round to 0 or coarse buckets → information destroyed</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Fix: give each <em>small block</em> its own scale → block / microscaling formats</div>
</div>
</div>

The fix is **block scaling** (a.k.a. **microscaling**): split the tensor into small blocks (typically **32 elements**) and give *each block its own scale*. Now the outlier only ruins its own 32-element block, and every other block gets a scale matched to its own magnitude. The dynamic range becomes *local*. This is the single idea that makes 4-bit floating point viable for real models.

---

## Block / Microscaling Formats: Shared Exponents

The core trick: a low-bit element (FP4, FP6, FP8, INT) stores only the **fine detail**; a **shared scale** stored once per block supplies the **coarse magnitude**. Conceptually, the block factors out a common exponent — a *shared exponent* — that all elements ride on top of.

$$x_i \approx \text{scale}_{\text{block}} \times q_i, \qquad q_i \in \text{small format}, \quad i = 0\ldots31$$

### MXFP — The OCP Microscaling Standard

The **Open Compute Project (OCP) Microscaling (MX)** spec standardized this in 2023, with buy-in from NVIDIA, AMD, Intel, Microsoft, Meta, and others. An MX format is defined by:

- A **block of 32 elements** (the standard block size $k = 32$).
- One **shared scale per block**, stored as **E8M0** — an 8-bit field that is *pure exponent, no mantissa, no sign*. It encodes a power of two from roughly $2^{-127}$ to $2^{127}$. (E8M0 = "just an exponent," matching FP32's exponent range so it can represent any power-of-two scale FP32 could.)
- A per-element format, which names the variant:

| MX format | Element | Element bits | + scale (E8M0) | Effective bits/element |
|-----------|:-------:|:------------:|:--------------:|:----------------------:|
| MXFP8 | FP8 (E4M3 or E5M2) | 8 | 8 bits / 32 elems | $8 + 8/32 = 8.25$ |
| MXFP6 | FP6 (E3M2 or E2M3) | 6 | 8 bits / 32 elems | $6 + 8/32 = 6.25$ |
| MXFP4 | FP4 (E2M1) | 4 | 8 bits / 32 elems | $4 + 8/32 = \mathbf{4.25}$ |
| MXINT8 | INT8 | 8 | 8 bits / 32 elems | $8.25$ |

> The memory math you must remember: **MXFP4 ≈ 4.25 bits/element.** Sixteen 4-bit elements + one 8-bit scale per 32 elements = $(32 \times 4 + 8)/32 = 136/32 = 4.25$ bits/element. That 0.25-bit "tax" buys back the dynamic range that makes 4-bit floats usable. Compare to FP16's 16 bits/element: MXFP4 is a **~3.76× memory reduction**.

<div class="diagram">
<div class="diagram-title">MXFP4 Block Layout (one block = 32 elements + 1 scale)</div>
<div class="flow-h">
  <div class="flow-node yellow">E8M0 scale<br/><small>8 bits, shared</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent">FP4 #0<br/><small>4 b</small></div>
  <div class="flow-node accent">FP4 #1<br/><small>4 b</small></div>
  <div class="flow-node accent">…<br/><small>…</small></div>
  <div class="flow-node accent">FP4 #31<br/><small>4 b</small></div>
</div>
</div>

### NVFP4 — NVIDIA Blackwell's Two-Level Scaling

NVIDIA's Blackwell generation pushes FP4 further with **NVFP4**, which differs from MXFP4 in two deliberate ways:

1. **Smaller blocks (16 elements, not 32):** finer-grained scaling → less damage from any one outlier, better accuracy.
2. **Two-level scaling:** the per-block scale is an **FP8 (E4M3)** value, not E8M0 — giving the block scale itself a *mantissa* so it can land between powers of two (E8M0 can only hit exact powers of two). A *second*, per-tensor FP32 scale then sets the overall magnitude.

$$x_i \approx \underbrace{s_{\text{tensor}}^{\text{(FP32)}}}_{\text{1 per tensor}} \times \underbrace{s_{\text{block}}^{\text{(FP8 E4M3)}}}_{\text{1 per 16 elems}} \times \underbrace{q_i^{\text{(FP4 E2M1)}}}_{\text{1 per element}}$$

| | MXFP4 (OCP) | NVFP4 (NVIDIA) |
|--|:-----------:|:--------------:|
| Element | FP4 E2M1 | FP4 E2M1 |
| Block size | 32 | 16 |
| Block scale | E8M0 (power-of-two only) | FP8 E4M3 (has mantissa) |
| Second-level scale | none | per-tensor FP32 |
| Bits/element | 4.25 | $4 + 8/16 = 4.5$ |
| Trade-off | leaner, fully standardized | slightly fatter, more accurate |

> The two formats stake out the FP4 trade space: MXFP4 is the leaner, standardized choice (4.25 bits); NVFP4 spends a little more (4.5 bits, smaller blocks, a richer scale) to claw back accuracy. Both prove the thesis — **at 4 bits, the scale structure matters as much as the element format.**

---

## Code: Reading the Bits Yourself

Talk is cheap; let's read the actual bits. All snippets are runnable with recent PyTorch/NumPy.

### Decoding sign / exponent / mantissa for any float dtype

```python
import torch

def decode_bits(x, dtype, exp_bits, mant_bits, bias):
    """Print the raw bits of a value in a given float dtype, split into fields."""
    t = torch.tensor([x], dtype=dtype)              # 1-D so we can view it as bytes
    nbytes = t.element_size()                       # 4=fp32, 2=bf16/fp16, 1=fp8
    raw = int.from_bytes(bytes(t.view(torch.uint8).tolist()), 'little')
    total = nbytes * 8
    sign = (raw >> (total - 1)) & 1
    exp  = (raw >> mant_bits) & ((1 << exp_bits) - 1)
    mant = raw & ((1 << mant_bits) - 1)
    print(f"{dtype} value={float(t[0]):<12.6g} bits={raw:0{total}b}  "
          f"S={sign} E={exp}({exp - bias:+d}) M={mant:0{mant_bits}b}")

decode_bits(1.0,  torch.float32,  8, 23, 127)   # S=0 E=127(+0) M=000...
decode_bits(1.0,  torch.bfloat16, 8,  7, 127)   # S=0 E=127(+0) M=0000000
decode_bits(1.0,  torch.float16,  5, 10,  15)   # S=0 E=15 (+0) M=0000000000
decode_bits(1.0,  torch.float8_e4m3fn, 4, 3, 7) # S=0 E=7  (+0) M=000
decode_bits(1.0,  torch.float8_e5m2,   5, 2, 15)# S=0 E=15 (+0) M=00
```

Every one of these encodes 1.0 as "sign 0, exponent = bias (so $e=0$), mantissa 0" — i.e. $1.0 \times 2^0$. The *only* thing that changes between formats is how many bits each field gets.

### BF16 vs FP16: the range demonstration

```python
import torch

big = 70000.0   # bigger than FP16's max of 65504

fp16 = torch.tensor(big, dtype=torch.float16)
bf16 = torch.tensor(big, dtype=torch.bfloat16)
print(f"FP16 of {big}: {fp16}")   # inf  ← OVERFLOW at 65504
print(f"BF16 of {big}: {bf16}")   # ~69632.0  ← fits, but coarse

# How coarse? Show BF16's step size near 70000:
import math
print("FP16 max normal :", (2 - 2**-10) * 2**15)         # 65504.0
print("BF16 max normal :", (2 - 2**-7)  * 2**127)        # 3.39e38
# ULP near 1.0:
print("FP16 ULP@1.0    :", 2**-10)   # 0.000977  (finer)
print("BF16 ULP@1.0    :", 2**-7)    # 0.0078125 (coarser by 8x)
```

This is the whole BF16-vs-FP16 story in five lines: **FP16 overflows at 65504; BF16 reaches $\sim 3.4\times10^{38}$ but with an 8× coarser step.** For training, not overflowing wins.

### Quantizing a tensor to INT8 with a scale

```python
import torch

def quantize_int8(x: torch.Tensor):
    """Symmetric per-tensor INT8 quantization. Returns (int8 codes, scale)."""
    amax = x.abs().max()
    scale = amax / 127.0                              # map [-amax, amax] -> [-127, 127]
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale

def dequantize_int8(q: torch.Tensor, scale: torch.Tensor):
    return q.to(torch.float32) * scale

x = torch.tensor([0.1, -0.4, 1.7, -2.0, 0.02, 1.95])   # shape [6], fp32
q, s = quantize_int8(x)
xhat = dequantize_int8(q, s)
print("codes (int8):", q.tolist())                  # e.g. [6, -25, 108, -127, 1, 124]
print("scale       :", float(s))                    # 2.0/127 = 0.01575
print("dequantized :", [round(v, 4) for v in xhat.tolist()])
print("max abs err :", float((x - xhat).abs().max()))  # ~half a step
```

The error per element is bounded by half a step, $s/2$ — exactly the uniform-step behavior of integers. Note the *outlier* `-2.0` sets the scale for everyone; that is the per-tensor problem we just discussed.

### Simulating an MXFP4 block (shared E8M0 scale + FP4 elements)

```python
import torch, math

# The 8 magnitudes representable by FP4 E2M1 (unsigned); with sign -> negatives too.
FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

def quantize_fp4(vals: torch.Tensor) -> torch.Tensor:
    """Round each value to the nearest signed FP4 E2M1 level."""
    sign = torch.sign(vals)
    mag  = vals.abs().unsqueeze(-1)                       # [N,1]
    idx  = (mag - FP4_LEVELS).abs().argmin(dim=-1)        # nearest level
    return sign * FP4_LEVELS[idx]

def mxfp4_block(block: torch.Tensor):
    """One MX block: pick a power-of-two (E8M0) scale, quantize 32 elems to FP4."""
    amax = block.abs().max()
    # E8M0 scale = power of two so that amax maps near FP4's max (6.0)
    exp = math.floor(math.log2(amax / 6.0)) if amax > 0 else 0
    scale = 2.0 ** exp                                   # the shared exponent
    codes = quantize_fp4(block / scale)                  # FP4 codes, shape [32]
    return codes, scale

torch.manual_seed(0)
block = torch.randn(32) * 0.1
block[7] = 5.0                                            # inject an outlier
codes, scale = mxfp4_block(block)
recon = codes * scale
print("E8M0 scale (2^k):", scale)
print("bits/element    :", (32*4 + 8) / 32, "  # 4.25, the MXFP4 tax")
print("outlier recon   :", float(recon[7]), "vs 5.0 (clamped: FP4 max 6.0 × scale 0.5)")
print("mean small err  :", float((block[:7] - recon[:7]).abs().mean()))
```

Because the scale is *per block*, the outlier no longer forces every small neighbor to round to zero. (With this power-of-two E8M0 scale of 0.5, the outlier itself even clamps a bit — $5.0/0.5 = 10$ exceeds FP4's max of 6.0, so it reconstructs to $6.0 \times 0.5 = 3.0$; a mantissa-bearing scale would land closer.) And with FP4's mere 8 magnitudes the small values are still coarsely bucketed. NVFP4's smaller 16-element blocks *and* its FP8 (mantissa-bearing) block scale exist precisely to tighten both problems.

---

## Why This Matters for Model Optimization

<div class="diagram">
<div class="diagram-title">Number Formats → Your Optimization Decisions</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Pick the format for the job</div>
    <div class="card-desc">Training → BF16 (range, no loss scaling). Inference weights → INT8/INT4. FP8 forward = E4M3, backward = E5M2. The bit layout tells you the failure mode before you hit it.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Range vs precision is the lens</div>
    <div class="card-desc">Overflowing to inf/NaN? You need exponent bits (BF16, E5M2). Losing small weight updates? You need mantissa bits or an FP32 accumulator. Diagnose by which corner of the triangle you starved.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Bits/element = your memory budget</div>
    <div class="card-desc">FP16=16, INT8=8, INT4=4, MXFP4≈4.25. A 70B model: 140 GB (FP16) → 35 GB (INT4) → ~37 GB (MXFP4). This is whether it fits on one GPU (Ch 11, Ch 29).</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Block scaling unlocks 4-bit</div>
    <div class="card-desc">Per-tensor scaling dies at 4 bits because outliers steal the range. Microscaling (32- or 16-element blocks + shared scale) is *why* MXFP4/NVFP4 work — and why Blackwell built it into silicon.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Accumulation is separate</div>
    <div class="card-desc">Low-precision storage + FP32 accumulate is the universal pattern. The format you store is not the format you sum in — Chapter 24 shows why the silicon does it this way.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">The frontier is FP4/microscaling</div>
    <div class="card-desc">FP8 is now mainstream (Hopper+); FP4 with microscaling is the bleeding edge (Blackwell). Knowing the formats tells you what next-gen hardware will accelerate — and what to quantize toward.</div>
  </div>
</div>
</div>

This chapter is the foundation for the rest of Part V. **Chapter 25 (Quantization)** turns these formats into an accuracy/speed budget — *how much* each one actually buys and where it breaks. **Chapter 30 (Datatypes Drive Optimization)** is the practitioner's decision tree: given your model, hardware, and accuracy budget, which of these formats do you reach for. But first, **Chapter 24** answers the question lurking under every format above: *why* does halving the bits roughly double the throughput? The answer is in the silicon — the multiplier, the tensor core, and the per-generation hardware that decides which of these formats run fast at all.

## Key Takeaways

- Every numeric format is a single point in the **range (exponent) vs precision (mantissa) vs bit-width (memory/throughput)** triangle; you cannot win all three at once.
- **BF16** keeps FP32's 8-bit exponent (full range, no loss scaling) and sacrifices mantissa — which is why it became the **default training dtype**. **FP16** keeps more mantissa but overflows at **65504**, forcing loss scaling.
- **FP8 comes in two flavors**: **E4M3** (max 448, precision, forward pass) and **E5M2** (max 57344, range, gradients) — the same range/precision bet as BF16/FP16, one byte down.
- **FP4 (E2M1)** has only 16 values (max 6.0) and is useless alone; it only works inside **block-scaled** formats.
- **Per-tensor scaling fails at low bit-widths** because one outlier steals the dynamic range; **microscaling** (a shared scale per 32- or 16-element block) fixes this locally.
- **MXFP** (OCP standard: shared **E8M0** scale per 32-element block) gives **MXFP4 ≈ 4.25 bits/element**; **NVFP4** (Blackwell) uses 16-element blocks with an FP8 block scale plus a per-tensor FP32 scale (≈4.5 bits) for higher accuracy.
- **Integers** (INT8→INT1, ternary) are pure precision with a *uniform* step; their dynamic range is entirely on loan from the external scale — great for weights, harder for outlier-prone activations.
- The universal pattern: **low-precision storage, high-precision (FP32) accumulation** — store cheap, sum carefully.

---

*Next: [Chapter 24 — Datatypes & Silicon](./24_datatypes_and_silicon.md), where we open up the multiplier and the tensor core to see exactly why fewer mantissa bits means quadratically cheaper hardware — and why peak throughput roughly doubles every time precision halves.*

[← Back to Table of Contents](./README.md)
