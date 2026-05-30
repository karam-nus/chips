---
title: "Chapter 24 — Datatypes & Silicon: How Formats Reshape the Chip"
---

[← Back to Table of Contents](./README.md)

# Chapter 24 — Datatypes & Silicon: How Formats Reshape the Chip

[Chapter 23](./23_number_formats_deep_dive.md) gave you the formats — FP32, BF16, FP8, FP4, INT8, microscaling, and the rest — and ended on a promise: that *why* halving the bits roughly doubles your throughput is a fact about **silicon**, not software. This chapter makes good on that. We are going to open up the chip and look at what a number format actually *costs* in transistors, area, and energy — and how that cost ripples all the way up to the peak-TFLOP/s number on a GPU spec sheet.

The central character is the **multiplier**. A neural network is, at the bottom, an ocean of multiply-accumulate (MAC) operations, and the multiplier that does each one is the single most format-sensitive piece of hardware on the chip. We will see that its area and energy scale roughly with the *square* of the mantissa width — which is the deep reason an FP8 multiply is not 4× cheaper than an FP32 multiply but more like **30–60× cheaper**. From there we build up: the tensor core / MMA unit that packs thousands of these multipliers together, the all-important trick of **low-precision multiply + FP32 accumulate**, the per-generation hardware support matrix (Volta → Blackwell), the bandwidth side of the story, and finally the **co-design** loop in which formats like BF16, TF32, FP8, and MXFP were *invented for the hardware that would run them.*

> **The one-sentence version:** A multiplier's cost grows with the *square* of its mantissa width, so every bit you remove from the mantissa makes each arithmetic unit dramatically smaller and cheaper — which lets the chip pack in more units and toggle fewer transistors per operation, and that — plus the bandwidth saved by moving fewer bytes — is the entire physical reason lower precision runs faster, captured by the rough "2× throughput per precision halving" rule.

---

## The Multiplier: Why Area Scales ~Quadratically with Mantissa Width

Start from the gate level (Chapter 2). To multiply two N-bit numbers, the schoolbook algorithm computes $N$ **partial products** (one per bit of the multiplier), each $N$ bits wide, then sums them in an adder tree. That is on the order of $N^2$ AND gates to form the partial products and an adder network that also grows super-linearly. The headline:

$$\text{multiplier area} \ \sim\ O(N^2), \qquad N = \text{operand width that actually gets multiplied}$$

The crucial subtlety for floating point: **only the significand (mantissa + hidden bit) goes through the big multiplier.** The exponents are merely *added* (an N-bit adder is cheap, $O(N)$), and the sign is a single XOR. So a float multiply decomposes as:

<div class="diagram">
<div class="diagram-title">Inside a Floating-Point Multiplier</div>
<div class="flow-h">
  <div class="flow-node accent">Sign<br/><small>1 XOR gate — trivial</small></div>
  <div class="flow-node green">Exponent<br/><small>add exponents — O(N), cheap</small></div>
  <div class="flow-node red">Significand<br/><small>multiply mantissas — O(N²), <strong>expensive</strong></small></div>
</div>
</div>

This is *the* insight of the chapter. The expensive part is the **mantissa multiply**, and it scales with the square of the mantissa width. The exponent (which gives a format its dynamic range) is nearly free to widen. **Range is cheap; precision is expensive.** This asymmetry is why BF16 (8-bit exponent, 7-bit mantissa) is *cheaper to multiply* than FP16 (5-bit exponent, 10-bit mantissa) despite having more dynamic range — fewer mantissa bits means a smaller multiplier even though the exponent is wider.

### A relative-cost table

Using significand width (mantissa + 1 hidden bit) and the $O(N^2)$ rule, normalized to FP32 = 1.0. These are first-order approximations — real designs use Booth encoding, Wallace trees, and fused units that shift the constants — but the *scaling* is what matters and it holds:

| Format | Significand bits | Multiply area $\sim N^2$ | Multiply energy (rel.) | Add (accum) cost |
|--------|:----------------:|:------------------------:|:----------------------:|:----------------:|
| FP32 | 24 | **1.00** | 1.00 | $O(N)$, ~1.0 |
| TF32 | 11 | 0.21 | ~0.2 | (accumulates in FP32) |
| FP16 | 11 | 0.21 | ~0.2 | low |
| BF16 | 8 | 0.11 | ~0.1 | low |
| INT8 | 8 (integer) | 0.11 | ~0.05–0.1* | very low |
| FP8 E4M3 | 4 | 0.028 | ~0.03 | tiny |
| FP8 E5M2 | 3 | 0.016 | ~0.02 | tiny |
| FP4 E2M1 | 2 | 0.007 | ~0.01 | negligible |

<small>*Integer multiply has no exponent/normalization logic at all, so an INT8 multiply is even cheaper than its significand width alone suggests — one reason INT8 was the first low-precision win on hardware.</small>

> A BF16 multiplier is roughly **9× smaller** than an FP32 multiplier; an FP8-E4M3 multiplier is roughly **35× smaller**; FP4 is **~140× smaller**. You do not get a clean 2×-per-bit relationship — you get something steeper because of the square law. This is why the jump from FP32 to FP8 is transformative, not incremental.

### Why fewer bits also means *less energy* per operation

Recall the CMOS switching-energy formula from [Chapter 1](./01_what_is_a_chip.md):

$$E_{\text{switch}} \approx \tfrac{1}{2}\, C V_{dd}^{2}$$

Energy is spent when transistors *toggle*. A smaller multiplier has fewer gates, fewer wires, and less switched capacitance $C$ — and it toggles fewer of them per operation. Two compounding effects:

1. **Fewer gates to toggle.** An FP8 multiply lights up ~30× fewer transistors than FP32. Less total switched capacitance → less energy.
2. **Fewer bits move.** Reading an FP8 operand from a register or SRAM toggles 8 wires, not 32. Data movement, not arithmetic, often dominates energy in modern chips — and it scales *linearly* with bit-width.

The practical upshot: low precision is not just faster, it is **more energy-efficient per operation**, which in a power-and-thermal-limited chip (every modern GPU is power-limited) directly translates into *more usable throughput*. You can fit more multipliers in the same area *and* run them within the same power envelope.

<div class="diagram">
<div class="diagram-title">Fewer Mantissa Bits → Three Compounding Wins</div>
<div class="flow">
  <div class="flow-node accent wide">Smaller mantissa (N bits)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Multiplier area ~ N² smaller → fit MORE units in the same silicon</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Less switched capacitance + fewer bits moved → LESS energy per MAC (½CV²)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Fewer bytes/param → LESS memory & bandwidth (Ch 11)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">Net: roughly 2× throughput per precision halving</div>
</div>
</div>

---

## The Tensor Core / MMA Unit: Built Around a Datatype

A single multiplier is not how modern accelerators do matmul. They use a **tensor core** (NVIDIA) / **MMA unit** (matrix-multiply-accumulate) / **systolic array** (TPU, Chapter 16): a 2D grid of MAC units that performs a small dense matrix multiply ($D = A \times B + C$) every few cycles. The grid is **physically wired for a specific input datatype.** That wiring is why a chip's supported formats are a fixed hardware property, not a software choice.

A MAC unit's datapath is exactly the decomposition above, repeated thousands of times:

<div class="diagram">
<div class="diagram-title">One MAC Lane in a Tensor Core (mixed precision)</div>
<div class="flow-h">
  <div class="flow-node accent">A (low-prec)<br/><small>BF16/FP8</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node red">multiply<br/><small>low-prec × low-prec</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">accumulate<br/><small>add into FP32</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">FP32 partial sum</div>
</div>
</div>

### Mixed-precision: low-precision multiply, FP32 accumulate

This is the single most important architectural pattern in ML hardware, and it is the silicon embodiment of Chapter 23's "store cheap, sum carefully" rule. The inputs $A$ and $B$ are low precision (BF16, FP8, INT8). The **products are summed into a wide FP32 (or INT32) accumulator.** The output may be written back in low precision, but the *running sum lives in high precision*.

Why does the accumulator need to be wide when the inputs are narrow? Because a dot product adds up hundreds or thousands of terms, and three things go wrong in low precision:

- **Magnitude growth.** A dot product of length $K$ can be ~$\sqrt{K}$ to $K$ times larger than its terms. An FP8 accumulator (max 448) would overflow almost immediately; INT8×INT8 products reaching into the thousands overflow an INT8 sum — hence **INT32** accumulators for INT8 matmul.
- **Swamping.** When you add a tiny term to a large running sum in low precision, the tiny term rounds away entirely — the "1e7 + 1 = 1e7" effect from [Chapter 3](./03_number_representation.md). Across thousands of adds the error compounds and the result drifts.
- **Catastrophic cancellation.** If positive and negative terms nearly cancel (common in trained networks), the surviving small difference is exactly where you needed your precision. Accumulating in FP32 preserves it; accumulating in BF16 can destroy it (Chapter 3, Chapter 23).

> **The rule:** the format you *multiply* in and the format you *accumulate* in are different on purpose. Tensor cores multiply in BF16/FP8/INT8 (cheap, $O(N^2)$ saved) but accumulate in FP32/INT32 (where the $O(N)$ adder cost is small and the precision is essential). This is why a "BF16 matmul" is really *BF16-multiply, FP32-accumulate* — and why you almost never lose training stability to the multiply precision alone.

### The dot-product datapath, drawn out

Zoom into a systolic array (the TPU's matrix unit, Chapter 16; tensor cores use a related dataflow). It is a 2D grid where each cell holds one MAC. Operands flow in from the edges, products accumulate as they march through, and the format choice sets how *wide* every wire and register in the grid must be:

```text
       B columns flow downward (low precision: BF16 / FP8 / INT8)
        b00   b01   b02
         │     │     │
 a00 ──►[MAC]─[MAC]─[MAC]──►   each cell: acc += a × b
         │     │     │                    ↑ multiply LOW precision
 a01 ──►[MAC]─[MAC]─[MAC]──►              acc lives in FP32/INT32 (WIDE)
         │     │     │
 a02 ──►[MAC]─[MAC]─[MAC]──►   A rows flow rightward (low precision)
         │     │     │
         ▼     ▼     ▼
      FP32 accumulators drain out the bottom  ── the running sums
```

Two format facts fall straight out of this picture:

- **The grid is sized for the *input* datatype.** Halving input width (BF16→FP8) lets the same silicon area hold ~2× more MAC cells, or feed the existing cells at 2× the rate — the structural origin of the "2× per precision halving" rule below.
- **The accumulators down the bottom are wide regardless.** They are a fixed FP32/INT32 cost, which is why the speedup from ever-smaller inputs eventually plateaus: at some point the accumulators and operand delivery, not the multipliers, set the limit.

---

## Per-Generation Hardware Support: Volta → Blackwell

A format only runs fast if the silicon has units wired for it. Here is the canonical NVIDIA progression (with TPU and AMD for context), and the throughput pattern that falls out of it. **Dense** peak figures; sparse roughly doubles these again (Chapter 26).

| Architecture (year) | New format support | Notable peak (dense) | The pattern |
|---------------------|--------------------|----------------------|-------------|
| **Volta** V100 (2017) | FP16 tensor cores | ~125 FP16 TFLOP/s | first tensor cores |
| **Turing** T4 (2018) | + INT8, INT4 | ~130 FP16 / 260 INT8 TOPS | inference int formats |
| **Ampere** A100 (2020) | + **BF16**, **TF32** | ~312 BF16 / 624 INT8 | BF16 makes training easy |
| **Hopper** H100 (2022) | + **FP8** (E4M3/E5M2) | ~990 BF16 / **1979 FP8** | FP8 doubles over BF16 |
| **Blackwell** B200 (2024) | + **FP4 / microscaling** (MXFP, NVFP4) | ~2250 FP8 / **4500 FP4** | FP4 doubles over FP8 |
| Google **TPU** v2→v5 | BF16 (native), INT8 | systolic 128×128+ MXU | BF16 was *born* on TPU |
| AMD **CDNA** MI250→MI300 | FP16/BF16/FP8 (Matrix Cores) | competitive FP8 throughput | follows the same curve |

Read the throughput column top to bottom and a rule emerges:

> **The "2× per precision halving" rule.** Each time the data width halves at the same architecture, peak throughput roughly **doubles**: H100 does ~990 BF16 vs ~1979 FP8 TFLOP/s; B200 does ~2× more in FP4 than FP8. The reason is exactly the multiplier cost: a narrower datatype lets you pack ~2× the MAC units in the same area *and* feed them with ~2× the effective bandwidth. Halve the bits → double the units → double the throughput.

### Where the 2× rule stops helping

The rule is a guide, not a guarantee. It breaks down for several concrete reasons you will hit in practice:

- **The square law over-delivers, then plateaus.** Going FP32→FP8 the multiplier shrinks ~35×, so the limit becomes *other* things — register file ports, wires, the accumulator (still FP32), and operand delivery — not the multiplier. Below FP8 you stop being multiplier-bound, so FP4 rarely gives a *clean* 2× over FP8 in real workloads.
- **Accumulation doesn't shrink.** The FP32 accumulator is fixed regardless of input precision. As inputs get tiny, accumulation becomes a larger fraction of the cost, capping the speedup.
- **Overhead grows.** Quantize/dequantize, scale handling, and (for microscaling) per-block scale logic add fixed costs that a 4-bit element can't amortize as well as an 8-bit one.
- **Memory-bound regimes (Chapter 11).** If your kernel is bandwidth-bound, not compute-bound, lower precision helps via *bytes moved*, not FLOP/s — and the win tracks bandwidth, not the 2× compute rule. (This is the usual case for LLM decode; see Chapter 29.)
- **Accuracy floors (Chapter 25).** A 2× speedup you can't use because the model's accuracy collapsed is not a speedup.

---

## The Memory & Bandwidth Side

Compute is only half the story — often the smaller half. The datatype also sets **bytes per parameter**, which drives footprint and, for memory-bound kernels, the actual runtime (the roofline of [Chapter 11](./11_memory_wall_and_bandwidth.md)).

$$\text{bytes moved} = \text{elements} \times \text{bytes per element}, \qquad \text{time}_{\text{BW-bound}} = \frac{\text{bytes moved}}{\text{bandwidth}}$$

| Format | Bytes/param | 70B model footprint | Relative bandwidth need |
|--------|:-----------:|:-------------------:|:-----------------------:|
| FP32 | 4 | 280 GB | 4× |
| BF16 / FP16 | 2 | 140 GB | 2× |
| FP8 / INT8 | 1 | 70 GB | 1× |
| INT4 / FP4 | 0.5 | 35 GB | 0.5× |
| MXFP4 | ~0.53 | ~37 GB | ~0.53× |

> For LLM **decode**, which is memory-bandwidth-bound (one token at a time, weights streamed from HBM — Chapter 29), the speedup from quantization comes almost entirely from **moving fewer bytes**, not from cheaper multiplies. Halving the bytes per weight roughly halves the time to stream the weights. This is *why* INT4 weight-only quantization speeds up token generation even when the matmul itself runs in FP16 after dequant: the bottleneck was the bytes, and you cut them.

A compact view of the two regimes the datatype affects:

| Regime | Bound by | Lower precision helps via | Example |
|--------|----------|---------------------------|---------|
| Compute-bound | tensor-core FLOP/s | more & cheaper MAC units (2× rule) | large-batch training matmul |
| Bandwidth-bound | HBM bytes/s | fewer bytes moved (linear in bit-width) | LLM single-token decode |

---

## Co-Design: Formats Invented *For* the Hardware

Here is the theme that ties Chapters 23 and 24 together. The formats that won were not handed down from a numerical-analysis committee and then implemented. They were **co-designed**: numerics and silicon shaped each other, deliberately, to land at a sweet spot of cheap-to-build and good-enough-for-models.

<div class="diagram">
<div class="diagram-title">Formats Designed Hand-in-Hand with Silicon</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">BF16 (Google Brain)</div>
    <div class="card-desc">"Truncate FP32" — chosen so FP32↔BF16 conversion is trivial and the multiplier is small, while keeping FP32's exponent so training needs no loss scaling. Born on the TPU.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">TF32 (NVIDIA)</div>
    <div class="card-desc">A *compute mode*, not a storage type: keep FP32's exponent, truncate the mantissa to 10 bits so the tensor-core multiplier is ~5× smaller — a drop-in FP32 speedup.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">FP8 E4M3 / E5M2 (OCP)</div>
    <div class="card-desc">Two formats because no single 8-bit split serves both forward (precision) and backward (range). Standardized so multiple vendors' silicon interoperates.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">MXFP / microscaling (OCP)</div>
    <div class="card-desc">A shared E8M0 scale per 32-element block — designed so the block scale costs ~0.25 bits/element and the dequant logic is cheap enough to sit inside the MMA unit.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">NVFP4 (NVIDIA Blackwell)</div>
    <div class="card-desc">16-element blocks + an FP8 block scale + FP32 tensor scale, tuned to the accuracy Blackwell's FP4 tensor cores needed to be usable for real inference.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">The OCP standardization story</div>
    <div class="card-desc">The Open Compute Project got NVIDIA, AMD, Intel, Microsoft, Meta, Arm to agree on FP8 and MX formats — so models, kernels, and silicon from different vendors speak the same bits.</div>
  </div>
</div>
</div>

The lesson for you as an optimizer: **the formats your hardware accelerates are the ones someone designed to be both cheap in silicon and survivable for models.** When a new generation adds a format (FP8 on Hopper, FP4 on Blackwell), it is a signal about where the cheap-and-good frontier just moved — and therefore where you should consider quantizing toward.

---

## Code: Quantifying the Silicon Story

### Relative multiplier cost vs mantissa width

```python
import math

# significand = mantissa bits + 1 hidden bit (integers: just the width, no hidden bit)
significand = {
    "FP32": 24, "TF32": 11, "FP16": 11, "BF16": 8,
    "FP8_E4M3": 4, "FP8_E5M2": 3, "FP4_E2M1": 2,
    "INT8": 8, "INT4": 4,
}
base = significand["FP32"]
print(f"{'format':10s} {'sig':>4s} {'mult area ~N^2':>15s} {'vs FP32':>9s}")
for name, n in significand.items():
    area = (n / base) ** 2          # O(N^2) multiplier-area model
    print(f"{name:10s} {n:4d} {area:15.4f} {1/area:8.1f}x")
# FP8_E4M3 -> ~35x smaller multiplier than FP32; FP4 -> ~144x
```

### Peak throughput scaling by precision (the 2× rule)

```python
# Model: halving the data width ~doubles peak ops, anchored on real H100 numbers.
peak = {"FP32": 67, "TF32": 495, "BF16": 990, "FP8": 1979}   # H100 dense TFLOP/s (approx)
print("H100 dense peak (TFLOP/s):")
prev = None
for fmt in ["FP32", "TF32", "BF16", "FP8"]:
    line = f"  {fmt:5s}: {peak[fmt]:6.0f}"
    if prev:
        line += f"   ({peak[fmt]/peak[prev]:.1f}x vs {prev})"
    print(line); prev = fmt
# BF16 vs FP8 ~2.0x; the 2x-per-precision-halving rule in real silicon
```

### Mixed-precision GEMM: BF16 inputs, FP32 accumulate

```python
import torch

K = 4096
A = torch.randn(2, K, dtype=torch.float32)      # [2, 4096] fp32 reference
B = torch.randn(K, 2, dtype=torch.float32)       # [4096, 2]
ref = A @ B                                      # FP32 reference result

# (a) accumulate in BF16  -> error compounds across 4096 adds
acc_bf16 = (A.bfloat16() @ B.bfloat16())                       # bf16 accumulate
# (b) tensor-core style: bf16 multiply, FP32 accumulate (cast then matmul in fp32)
acc_f32  = (A.bfloat16().float() @ B.bfloat16().float())       # fp32 accumulate

print("err, BF16 accumulate :", float((acc_bf16.float() - ref).abs().mean()))
print("err, FP32 accumulate :", float((acc_f32 - ref).abs().mean()))
# FP32-accumulate error is far smaller: same cheap bf16 multiplies, but the
# running sum keeps its precision -> this is exactly what a tensor core does.
```

The FP32-accumulate path uses the *same* low-precision multiplies (cheap, $O(N^2)$ saved) but keeps the running sum accurate — which is why every tensor core does it this way.

### Why INT8 matmul needs an INT32 accumulator

The integer version of the same lesson — and the reason every INT8 kernel accumulates in INT32:

```python
import torch

K = 64
a = torch.randint(-127, 128, (1, K), dtype=torch.int8)   # [1, 64] int8
b = torch.randint(-127, 128, (K, 1), dtype=torch.int8)   # [64, 1] int8

# correct hardware path: widen to INT32 *before* multiply/accumulate
correct = (a.to(torch.int32) @ b.to(torch.int32)).item()

# what an 8-bit accumulator would do: every partial sum wraps mod 256
acc = 0
for x, y in zip(a[0].tolist(), [r[0] for r in b.tolist()]):
    acc = (acc + x * y + 128) % 256 - 128                # simulate INT8 wraparound
print("INT32 accumulate :", correct)                     # e.g. -60929  (correct)
print("INT8  accumulate :", acc, " <- wrapped garbage")  # e.g. -1
print("one product alone:", 127 * 127, "> 127 -> already needs >8 bits")
```

A single INT8×INT8 product can reach $127 \times 127 = 16{,}129$ — already far outside INT8's range — and a length-$K$ sum of them needs more headroom still. That is the entire reason quantized inference kernels carry an **INT32 accumulator**: cheap 8-bit multiplies, wide 32-bit running sum.

---

## Why This Matters for Model Optimization

<div class="diagram">
<div class="diagram-title">Silicon Costs → Why Your Speedups Are What They Are</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Speedups are a square law</div>
    <div class="card-desc">FP32→FP8 isn't 4× cheaper, it's ~30× cheaper per multiply — because multiplier area ~ mantissa². This is *why* the low-precision payoff is so large, and why FP8/FP4 hardware exists.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Range is cheap, precision is dear</div>
    <div class="card-desc">Widening the exponent (range) costs an O(N) adder; widening the mantissa (precision) costs an O(N²) multiplier. This is the hardware reason BF16 (wide exp, narrow mantissa) is both cheap *and* training-friendly.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">The 2× rule, and its ceiling</div>
    <div class="card-desc">Peak throughput ~doubles per precision halving (BF16→FP8 on H100). But accumulation, overhead, bandwidth limits, and accuracy floors cap it — so FP4 rarely gives a clean 2× over FP8 in practice.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Accumulate in FP32, always</div>
    <div class="card-desc">Tensor cores multiply low, accumulate high — on purpose. If you build custom kernels, mirror this: low-precision storage, FP32/INT32 accumulator. It's why training survives BF16/FP8 multiplies.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Check your generation's matrix</div>
    <div class="card-desc">FP8 needs Hopper+, FP4/microscaling needs Blackwell. Quantizing to a format your silicon lacks gives you the memory win but *not* the compute win — know what your chip actually accelerates.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Two different wins to claim</div>
    <div class="card-desc">Compute-bound (training matmul) → fewer/cheaper units (2× rule). Bandwidth-bound (LLM decode) → fewer bytes moved (linear). Diagnose which you're in before predicting the speedup (Ch 11, 29).</div>
  </div>
</div>
</div>

This chapter explained the *mechanism*: lower precision is faster because each arithmetic unit is quadratically cheaper to build and cheaper to switch, so the chip packs in more of them within the same area and power, while simultaneously moving fewer bytes. **Chapter 25 (Quantization, Hardware View)** turns this mechanism into a budget — *how much* accuracy you trade for that speedup, and exactly where the curve bends from "free win" to "model breaks." That is the practitioner's real question, and it is the next chapter.

## Key Takeaways

- A multiplier's area and energy scale with the **square of the mantissa (significand) width** — so removing mantissa bits yields outsized savings (FP8 multiply ≈ **35× smaller** than FP32, not 4×).
- In a float multiply, only the **significand goes through the big $O(N^2)$ multiplier**; the exponent is just an $O(N)$ add. Hence **range is cheap, precision is expensive** — the silicon reason BF16 beats FP16 on cost.
- **Tensor cores / MMA units** are physically wired for specific datatypes and do **low-precision multiply, FP32/INT32 accumulate** — store cheap, sum carefully — to avoid overflow, swamping, and catastrophic cancellation.
- Per generation, hardware adds formats: **Volta (FP16) → Turing (+INT8/INT4) → Ampere (+BF16/TF32) → Hopper (+FP8) → Blackwell (+FP4/microscaling)**; TPUs were built around BF16.
- The **"2× per precision halving" rule** (e.g. H100 ~990 BF16 vs ~1979 FP8 TFLOP/s dense) comes from packing ~2× the MAC units in the same area; it is capped by accumulation, overhead, bandwidth, and accuracy.
- The datatype sets **bytes/param**, so for **bandwidth-bound** workloads (LLM decode) the speedup comes from moving fewer bytes — independent of the compute 2× rule.
- Modern ML formats (BF16, TF32, FP8, MXFP, NVFP4) were **co-designed** with the silicon and standardized through the OCP — a new accelerated format is a signal of where the cheap-and-good frontier moved.

---

*Next: [Chapter 25 — Quantization (Hardware View)](./25_quantization_hardware_view.md), where we turn this silicon mechanism into an accuracy/speed budget — how much low precision actually buys you, and exactly where it stops helping.*

[← Back to Table of Contents](./README.md)
