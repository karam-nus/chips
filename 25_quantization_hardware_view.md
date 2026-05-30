---
title: "Chapter 25 — Quantization: The Hardware View"
---

[← Back to Table of Contents](./README.md)

# Chapter 25 — Quantization: The Hardware View

You quantize models for a living. You know `int8` weights shrink a checkpoint, that `fp8` is the new training hotness, that `int4` lets a 70B model fit on one card. What you may *not* have a crisp mental model for is **why** each of those helps — and, more importantly, **by exactly how much, and when it does nothing at all.** That gap is where projects go wrong: someone quantizes a model to INT4, expects a 4× speedup, and gets 1.0×, because they were compute-bound on hardware with no INT4 datapath. The bytes shrank; the wall-clock didn't move.

This chapter dissects quantization strictly *from the silicon's point of view*. We build on three earlier chapters: the **memory wall** ([Chapter 11](./11_memory_wall_and_bandwidth.md)) tells us *when* fewer bytes equals more speed; the **number formats** ([Chapter 23](./23_number_formats_deep_dive.md)) tell us what INT8/FP8/FP4 actually are bit-for-bit; and **datatypes & silicon** ([Chapter 24](./24_datatypes_and_silicon.md)) tell us how a multiplier's transistor cost scales with bit-width. Here we fuse all three into a single predictive question: *given my workload and my GPU, will this quantization actually speed it up — and what will it cost me in accuracy?*

> **The one-sentence version:** Quantization always saves **memory footprint** and (for memory-bound work) **bandwidth**, but it only saves **compute time** when the hardware has a native low-precision datapath *and* you were compute-bound to begin with — so "fewer bits" is a guaranteed memory win and a *conditional* speed win.

---

## The Three Hardware Wins, Quantified

Quantization buys you three distinct things. They are governed by different hardware resources, they kick in at different times, and conflating them is the single most common mistake practitioners make. Keep them separate in your head.

<div class="diagram">
<div class="diagram-title">Three Independent Wins from Fewer Bits</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card green">
    <div class="card-icon">📦</div>
    <div class="card-title">1. Memory Footprint</div>
    <div class="card-desc">Bytes per parameter. Always wins. Decides what *fits* — model + KV cache in HBM. fp16=2B, int8=1B, int4=0.5B.</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-icon">🚚</div>
    <div class="card-title">2. Memory Bandwidth</div>
    <div class="card-desc">Bytes streamed per token. Wins *when memory-bound* (LLM decode). Halving bits ≈ halving decode time. The dominant win for serving.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">⚙️</div>
    <div class="card-title">3. Compute Throughput</div>
    <div class="card-desc">Math ops/sec. Wins *only* with a native datapath + when compute-bound (prefill, training). int8 ≈ 2× bf16; fp4 ≈ 2× fp8.</div>
  </div>
</div>
</div>

### Win 1 — Memory Footprint (always)

This one is unconditional and the easiest to reason about. The number of bytes a tensor occupies is `numel × bytes_per_element`. Drop the precision, drop the bytes. There is no hardware caveat: a tensor in `int8` is half the size of the same tensor in `fp16`, full stop.

| dtype | bits | bytes/param | 70B model weights | relative |
|-------|:----:|:-----------:|:-----------------:|:--------:|
| FP32 | 32 | 4 | 280 GB | 4× |
| FP16 / BF16 | 16 | 2 | 140 GB | 2× |
| FP8 (E4M3) | 8 | 1 | 70 GB | 1× |
| INT8 | 8 | 1 | 70 GB | 1× |
| INT4 / FP4 | 4 | 0.5 | 35 GB | 0.5× |

The footprint win decides **what fits**. A 70B model in BF16 is 140 GB — it does *not* fit on one 80 GB H100 and forces you onto two GPUs (with all the interconnect cost that implies, see [Chapter 12](./12_buses_and_interconnects.md)). In INT8 it is 70 GB — fits with ~10 GB to spare for KV cache and activations. In INT4 it is 35 GB — fits comfortably, leaving 45 GB for a large KV cache and bigger batches. **The footprint win is why INT4 exists at all:** not for speed, but to make a model *runnable* on hardware you can afford. We'll see below that the speed story is far more conditional.

> A subtle but important point: quantizing **weights** shrinks the static footprint, but the **KV cache** is often the thing that actually overflows your HBM at long context. KV-cache quantization (INT8 or FP8 K/V) is a separate, frequently larger footprint win during serving — covered below and in [Chapter 29](./29_memory_hierarchy_in_action.md).

### Win 2 — Memory Bandwidth (the dominant win for LLM decode)

Recall the central result of [Chapter 11](./11_memory_wall_and_bandwidth.md): **LLM autoregressive decode at small batch is memory-bandwidth-bound.** At batch=1, every parameter must be streamed from HBM exactly once per token generated. The tensor cores sit largely idle; the clock is set entirely by how fast HBM can deliver weights.

This makes the bandwidth win brutally simple to predict: **halving the bits per weight halves the bytes streamed, which halves the decode time** — even though you do the *same* number of multiply-accumulates (and a few *extra* dequantize ops). You are not paying for the math; you are paying for the data movement, and you cut the data movement in half.

$$t_{\text{token}} \approx \frac{\text{model bytes streamed}}{\text{HBM bandwidth}} \quad\Longrightarrow\quad t_{\text{token}} \propto \text{bytes/param}$$

For a 70B model on an H100 (HBM3 ≈ 3.35 TB/s), at batch=1:

| Weight dtype | Bytes streamed/token | Decode time/token | Tokens/s (batch=1) | Speedup vs BF16 |
|--------------|:--------------------:|:-----------------:|:------------------:|:---------------:|
| BF16 | 140 GB | ~41.8 ms | ~24 | 1.0× |
| INT8 | 70 GB | ~20.9 ms | ~48 | ~2.0× |
| INT4 | 35 GB | ~10.4 ms | ~96 | ~4.0× |

These are bandwidth-ceiling numbers (real systems hit ~60–80% of them, and there's a small dequant tax), but the *scaling* is the point: in the memory-bound regime, decode throughput scales almost perfectly inversely with bytes/param. **This is the most important quantization win in production LLM serving, and it requires no special tensor-core datapath at all** — the speedup comes purely from streaming fewer bytes. INT4 *weights* speed up decode even on a GPU that has no INT4 math units, because the multiply can happen in FP16 after a cheap on-the-fly dequant. The win is the *load*, not the *multiply*.

> This is why "weight-only quantization" (INT4 weights, FP16 activations and compute) is the workhorse of LLM serving: it captures the entire bandwidth win for decode while side-stepping the hard problem of quantizing activations. More on that distinction shortly.

### Win 3 — Compute Throughput (only with a datapath, only when compute-bound)

The third win is the one everyone *expects* and the one that most often fails to materialize. Modern tensor cores can run low-precision matmuls faster because a narrower multiplier is cheaper in silicon (the area of an integer multiplier scales roughly as the *square* of the bit-width — see [Chapter 24](./24_datatypes_and_silicon.md)), so you can pack more of them in the same area and clock them in parallel.

The published peak throughput multipliers, dense, relative to BF16:

| Format | H100 (Hopper) | B200 (Blackwell) | Notes |
|--------|:-------------:|:----------------:|-------|
| BF16 / FP16 | 1× (baseline) | 1× | The reference |
| FP8 (E4M3/E5M2) | ~2× | ~2× | Hopper added the FP8 datapath |
| INT8 | ~2× | ~2× | Available since Turing/Ampere |
| FP4 (NVFP4/MXFP4) | — (no datapath) | ~2× vs FP8 (~4× vs BF16) | Blackwell-only |
| INT4 | ~4× (Turing/Ampere had it; **dropped on Hopper**) | varies | Historically inconsistent |

Two things in that table will save you grief:

1. **A throughput multiplier only exists if the silicon has the datapath.** FP8's 2× is real *on Hopper and newer* — a Ampere A100 has no FP8 units, so FP8 there buys you footprint/bandwidth but *not* compute speed. FP4's 4× is real *on Blackwell* and nonexistent on Hopper. INT4 tensor-core math existed on Turing/Ampere, was **removed on Hopper**, so INT4 on an H100 has *no native matmul path* — its only benefit there is bandwidth/footprint.
2. **The multiplier only helps if you were compute-bound.** Prefill (long-prompt processing) and training are large dense GEMMs sitting well above the roofline ridge ([Chapter 11](./11_memory_wall_and_bandwidth.md)) — compute-bound — so doubling tensor-core throughput roughly doubles their speed. Decode at batch=1 is memory-bound, so the compute multiplier does *nothing* for it; the bandwidth win (Win 2) is what helps there.

### Which Win Dominates, and When

This is the crux of the whole chapter. Put the workload on the roofline and the answer falls out.

<div class="diagram">
<div class="diagram-title">Which Quantization Win Applies to Which Phase</div>
<div class="flow-h">
  <div class="flow-node green wide">Decode (batch=1)<br/><small>memory-bound → <b>bandwidth</b> win<br/>fewer bytes = faster, no datapath needed</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Prefill / Training<br/><small>compute-bound → <b>throughput</b> win<br/>needs native low-precision datapath</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Always (any phase)<br/><small><b>footprint</b> win<br/>decides what fits in HBM</small></div>
</div>
</div>

| Workload | Roofline regime | Win that matters | INT8 helps? | INT4-weight helps speed? |
|----------|-----------------|------------------|:-----------:|:------------------------:|
| LLM decode, batch=1 | Memory-bound | Bandwidth | Yes (~2×) | Yes (~4× via fewer bytes) |
| LLM decode, batch≫ridge | Compute-bound | Throughput | Yes if datapath | Only if INT4 datapath |
| LLM prefill (long prompt) | Compute-bound | Throughput | Yes if datapath | Rarely (no datapath on Hopper) |
| Training fwd/bwd | Compute-bound | Throughput | FP8 yes on Hopper | No (precision too low) |
| Fitting model in HBM | n/a | Footprint | Yes | Yes |

> **The mental shortcut:** *Footprint always. Bandwidth when memory-bound. Throughput when compute-bound AND the datapath exists.* If you remember nothing else from this chapter, remember those three clauses.

---

## The Mechanics at the Hardware Level

Now that we know *which* win applies *when*, let's look at *how* quantization actually executes in silicon — because the mechanics are exactly where the gains can quietly leak away.

### Affine Quantization: Scale and Zero-Point

The standard scheme is **affine (uniform) quantization**. A real value $x$ is approximated by an integer $q$ via a scale $s$ (a float) and a zero-point $z$ (an integer):

$$x \approx s\,(q - z), \qquad q = \operatorname{round}\!\left(\frac{x}{s}\right) + z$$

- $s$ (**scale**) is the size of one quantization step in real units — the resolution.
- $z$ (**zero-point**) is the integer that maps to real zero, so that $0.0$ is represented exactly (important for padding and ReLU outputs).
- $q$ is the stored low-bit integer (e.g., an `int8` in $[-128, 127]$).

For a given tensor with observed range $[\beta_{\min}, \beta_{\max}]$ mapped to integer range $[q_{\min}, q_{\max}]$:

$$s = \frac{\beta_{\max} - \beta_{\min}}{q_{\max} - q_{\min}}, \qquad z = q_{\min} - \operatorname{round}\!\left(\frac{\beta_{\min}}{s}\right)$$

### Symmetric vs Asymmetric

<div class="diagram">
<div class="diagram-title">Symmetric vs Asymmetric Quantization</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Symmetric (z = 0)</div>
    <ul>
      <li>Range is centered: <strong>[−α, +α]</strong></li>
      <li>Zero-point fixed at 0 → <strong>no z term</strong> in the matmul</li>
      <li>Cheaper hardware: integer GEMM needs no cross-terms</li>
      <li>Natural for <strong>weights</strong> (≈ zero-mean, symmetric)</li>
      <li>Wastes a code if the data is one-sided</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Asymmetric (z ≠ 0)</div>
    <ul>
      <li>Range can be lopsided: <strong>[β_min, β_max]</strong></li>
      <li>Uses the full integer range for skewed data</li>
      <li>Extra <strong>zero-point correction</strong> terms in the GEMM</li>
      <li>Natural for <strong>activations</strong> (e.g. post-ReLU ≥ 0)</li>
      <li>Slightly more compute to undo the offset</li>
    </ul>
  </div>
</div>
</div>

Why does the hardware care? Expand an asymmetric int matmul $\sum_i s_w(q_{w,i}-z_w)\,s_x(q_{x,i}-z_x)$. The cross-terms involving $z_w$ and $z_x$ produce extra reductions over the row/column that the kernel must compute and subtract. **Symmetric weights ($z_w = 0$) kill half of those terms**, which is why production INT8 kernels almost always use *symmetric* weights (and often symmetric activations too, accepting a little range waste for a simpler, faster kernel).

### Granularity: Per-Tensor vs Per-Channel vs Per-Group

A single scale for an entire tensor is cheap but crude: one outlier element stretches the range and crushes the resolution for everything else. Finer granularity gives each slice its own scale.

<div class="diagram">
<div class="diagram-title">Quantization Granularity (and its hardware cost)</div>
<div class="layer-stack">
  <div class="layer green">Per-tensor — 1 scale for the whole [N,K] weight. Cheapest; most outlier-sensitive.</div>
  <div class="layer accent">Per-channel — 1 scale per output row (N scales). Standard for weights; scale folds into the output column.</div>
  <div class="layer purple">Per-group / per-block — 1 scale per block of K (e.g. 128 or 64 elements). Best accuracy at low bits; needs blockwise dequant in the kernel.</div>
  <div class="layer orange">Microscaling (MX) — block of 32 shares a small (E8M0) power-of-two scale, baked into the format itself (Ch 23).</div>
</div>
</div>

The granularity choice interacts directly with the hardware:

- **Per-channel weight scales** are nearly free because the scale is constant down a whole output channel — you can apply it once when you write out the accumulated column, *after* the integer reduction. This is why per-channel weights + per-token activations is the standard W8A8 recipe.
- **Per-group scales** (e.g. one scale per 128 contiguous weights along K) cost more: the kernel must break the K-reduction into blocks, rescale at each block boundary, and accumulate the partial sums in higher precision. The accuracy at INT4 is much better, but the kernel is more complex and slightly slower — this is exactly what GPTQ/AWQ-style INT4 kernels do.
- **Microscaling formats (MXFP8, MXFP4, NVFP4)** push this to its logical end: the block scale is *part of the number format* — a block of 32 elements shares one tiny power-of-two (E8M0) exponent scale, decoded natively by Blackwell tensor cores. This is the bridge between "quantization" and "a number format," and it's why [Chapter 23](./23_number_formats_deep_dive.md) treats MX formats as first-class dtypes rather than a quantization scheme.

> **Granularity is a hardware/accuracy dial.** Coarser = simpler, faster kernels, more accuracy loss. Finer = better accuracy at low bits, but you pay in scale storage and kernel complexity. At INT8 per-channel is usually plenty; at INT4 you almost always need per-group.

### Weight-Only vs W8A8 vs KV-Cache Quantization

*What* you quantize matters as much as *how*. Three common schemes target three different wins:

| Scheme | What's quantized | Compute in | Primary win | Where it's used |
|--------|------------------|------------|-------------|-----------------|
| **Weight-only** (e.g. INT4 W, FP16 A) | Weights only | FP16 (dequant first) | Bandwidth + footprint | LLM decode serving (GPTQ, AWQ) |
| **W8A8** (INT8 weights + activations) | Both | INT8 (int32 accum) | Compute throughput | Prefill/training, compute-bound |
| **KV-cache quant** (INT8/FP8 K,V) | KV cache | FP16 attention math | KV footprint + bandwidth | Long-context serving |

- **Weight-only** is the decode workhorse: it captures the bandwidth win (you stream INT4 weights) but does the actual matmul in FP16, dequantizing weights on the fly. No activation-quantization headaches, no need for an INT4 datapath. The catch: every weight is *dequantized back to FP16* before the multiply, so you do **not** get the compute-throughput win — which is fine, because decode is memory-bound anyway.
- **W8A8** quantizes both operands so the matmul itself runs on INT8 tensor cores — this is the only way to get the *compute* win. But it's harder: activations have nasty outliers (next section), so W8A8 needs tricks like SmoothQuant or per-token scales to stay accurate.
- **KV-cache quantization** is its own lever. At long context the KV cache can dwarf the weights in HBM; storing K and V in INT8 or FP8 halves that and halves the bandwidth of reading it back each step. It's often the highest-ROI quantization for long-context serving and is independent of weight quantization.

### How an INT8 GEMM Actually Executes

Here is the dataflow that makes INT8 fast in silicon. The key trick is **accumulate in INT32**: each product of two INT8 values fits in 16 bits, and summing thousands of them needs more headroom, so the tensor core multiplies in INT8 but accumulates into a 32-bit integer. Only at the very end do we convert back to float.

<div class="diagram">
<div class="diagram-title">INT8 GEMM Dataflow (per output element)</div>
<div class="flow">
  <div class="flow-node green wide">INT8 weights q_w  ·  INT8 activations q_x<br/><small>1 byte each, streamed from HBM</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">INT8 × INT8 multiply<br/><small>product fits in 16 bits</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">Accumulate in INT32<br/><small>Σ over K; 32 bits avoids overflow</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Requantize / Dequantize<br/><small>multiply by s_w · s_x, add bias, cast to FP16</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">FP16 output (or re-quantize to INT8 for next layer)</div>
</div>
</div>

In equations, the integer reduction happens entirely in INT32, then a single float rescale converts the result back:

$$y \;=\; s_w\,s_x \underbrace{\sum_{k} q_{w,k}\,q_{x,k}}_{\text{INT32 accumulator}} \;+\; \text{bias}$$

The expensive part — the $K$-long reduction — runs in cheap integer arithmetic. The float multiply by $s_w s_x$ happens *once per output element*, not once per multiply-add. That amortization is precisely why INT8 GEMM is ~2× the throughput of BF16: the inner loop is integer, and the float overhead is $O(MN)$, not $O(MNK)$.

### The Dequant/Requant Tax — Why Tiny Ops Erase Gains

The accumulate-then-rescale story has a dark side. Every quantized matmul is sandwiched between **quantize** (float → int) and **dequantize/requantize** (int → float or int → int) operations. These are *elementwise* ops with arithmetic intensity ≈ 0.5–1 FLOP/byte — solidly memory-bound, the same category as LayerNorm or activation functions ([Chapter 11](./11_memory_wall_and_bandwidth.md)).

For a big compute-bound GEMM, that tax is negligible — a few $O(MN)$ elementwise passes around an $O(MNK)$ matmul. But two situations make it bite hard:

1. **Small / memory-bound matmuls.** If the matmul itself is memory-bound (decode GEMV), adding extra memory-bound quant/dequant passes can *eat the entire saving*. If your INT4 dequant kernel reads the weights, expands them to FP16 in HBM, and *then* the matmul reads them back, you've doubled traffic and lost the bandwidth win entirely.
2. **Unfused quantization.** If quant and dequant are separate kernel launches that round-trip through HBM, each one is a full memory pass. The fix is **fusion**: dequantize weights *inside* the matmul kernel, in registers/SRAM, never writing FP16 weights back to HBM. This is exactly what Marlin and CUTLASS mixed-input kernels do — and it's the difference between INT4 weight-only giving ~3–4× decode speedup versus ~1×.

> **The rule:** a quantized kernel is only as fast as its *fusion*. If the dequant isn't fused into the GEMM, you pay for the data movement twice and the "speedup" evaporates. When you benchmark a quantized model and see no gain, suspect an unfused dequant before you suspect the hardware.

---

## To What Extent It Helps — and Where It Breaks

Now the honest part. Quantization is not free accuracy and not free speed. Here is where it breaks and by how much.

### Activation Outliers: Why LLM Activations Are Hard

Weights in trained transformers are well-behaved — roughly zero-mean, roughly Gaussian, easy to quantize. **Activations are not.** In large transformers, a handful of channels develop activation values 10–100× larger than the rest (the "outlier channel" phenomenon). With a single per-tensor scale, those outliers stretch the range so far that the *typical* activations get crushed into just a few quantization levels — accuracy collapses.

```text
Activation channel magnitudes (schematic, one token):
ch:   0    1    2    3 ...  137  ... 4095
val:  0.4  0.3  0.5  0.2     58.0     0.4    ← channel 137 is a 100× outlier
                              ^^^^
      a single per-tensor scale must cover 58.0 → typical 0.4 values
      land in ~1 of 256 INT8 codes → catastrophic rounding error
```

This is *the* reason naive W8A8 fails on LLMs and why techniques exist to tame it:
- **Per-channel / per-token activation scales** give the outlier channel its own scale so it doesn't poison the rest (but per-channel *activation* scales don't fold cleanly into an integer GEMM the way per-channel *weight* scales do — they need per-token granularity to stay kernel-friendly).
- **SmoothQuant** migrates the difficulty from activations into weights by a per-channel rescaling $\hat{x} = x/s$, $\hat{W} = sW$, so neither operand has extreme outliers and both quantize cleanly. It's a software transform, but it exists *because of* the hardware constraint that GEMM kernels want simple, foldable scales.

We keep this hardware-focused, but the takeaway is: **activation quantization is the hard half of W8A8, and the difficulty is fundamentally about outliers vs. limited integer codes.**

### The Accuracy Cliff at INT4 and Below

Accuracy degradation is not linear in bits — there's a cliff. Roughly:

| Precision | Typical weight-only accuracy impact | Notes |
|-----------|-------------------------------------|-------|
| INT8 / FP8 | Near-lossless (< 0.1–0.5% on most tasks) | Per-channel weights; the safe default |
| INT4 (per-group, GPTQ/AWQ) | Small (~1–3% on hard tasks), often recoverable | Needs group size ≤ 128 + calibration |
| INT3 | Noticeable degradation | Research/edge territory; needs heavy recovery |
| INT2 / INT1 | Severe without retraining | Requires QAT/distillation; rarely production-viable |

The cliff has a clear cause: at $b$ bits you have $2^b$ representable levels. INT8 gives 256 levels — enough to capture a weight distribution finely. INT4 gives only **16 levels**; without per-group scales the quantization error swamps the signal. Below INT4 the representable set is so sparse that you must *retrain* the model to live in it (quantization-aware training, QAT) rather than just rounding a trained model (post-training quantization, PTQ).

### When INT4 Gives NO Speedup

This is worth stating bluntly because it surprises people:

<div class="diagram">
<div class="diagram-title">INT4: When It Speeds Up vs When It Doesn't</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">INT4 helps</div>
    <ul>
      <li>Decode, batch=1 → <strong>memory-bound</strong>, fewer bytes = ~4× faster</li>
      <li>Model didn't fit before → now it fits</li>
      <li>Kernel fuses dequant into the GEMM (Marlin)</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">INT4 does nothing for speed</div>
    <ul>
      <li>Compute-bound prefill on Hopper → <strong>no INT4 datapath</strong>, must dequant to FP16 to multiply</li>
      <li>Large-batch decode (compute-bound) → bandwidth win gone</li>
      <li>Unfused dequant → traffic doubled, win erased</li>
      <li>Non-matmul parts (attention, norm) dominate runtime</li>
    </ul>
  </div>
</div>
</div>

On an H100, INT4 *weight-only* is great for decode (bandwidth) but useless for prefill speed, because Hopper has no INT4 tensor-core math — the weights must be expanded to FP16 and multiplied in FP16, which runs at FP16 throughput. You got the footprint and the decode-bandwidth wins; you got *zero* compute win. That's not a bug, it's the datapath table from earlier.

### Memory Savings (Always) vs Compute Speedup (Conditional)

The cleanest summary of the whole chapter is this honest table:

| Precision | Memory save vs FP16 | Typical decode speedup (batch=1, memory-bound) | Typical compute speedup (compute-bound) | Typical accuracy impact | Hardware support |
|-----------|:-------------------:|:----------------------------------------------:|:---------------------------------------:|-------------------------|------------------|
| FP16/BF16 | 1× (baseline) | 1× | 1× | lossless | Everywhere |
| FP8 (E4M3) | 2× | ~2× | ~2× (Hopper+) | near-lossless | Hopper, Ada, Blackwell |
| INT8 (W8A8) | 2× | ~2× | ~2× | <0.5% (with SmoothQuant) | Turing+ (everywhere modern) |
| INT4 weight-only | 4× | ~3–4× | **~1× (no datapath on Hopper)** | ~1–3% (per-group) | Bandwidth win universal; math win HW-gated |
| FP4 (NVFP4/MXFP4) | 4× | ~3–4× | ~2× vs FP8 (Blackwell) | small w/ microscaling | Blackwell only |
| INT2/INT1 | 8–16× | up to 8–16× (if it works) | HW-dependent | severe w/o QAT | Niche/edge |

> **The two-column truth:** the "memory save" column is a hardware *guarantee*. The "compute speedup" column is a *promise conditional on (a) the datapath existing and (b) you being compute-bound.* Practitioners get burned by reading the compute column as if it were the memory column.

---

## Real Hardware Support

A quick map of what's actually in the silicon, because "does my GPU have the datapath" is the whole ballgame for the compute win (see [Chapter 24](./24_datatypes_and_silicon.md) for the per-generation deep dive).

<div class="diagram">
<div class="diagram-title">Low-Precision Datapath Support by Generation</div>
<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">Turing / Ampere (2018–2020)</div>
    <div class="timeline-title">INT8 + INT4 tensor cores</div>
    <div class="timeline-desc">INT8 W8A8 becomes mainstream; INT4 math path present (later dropped). No FP8.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Hopper (2022, H100)</div>
    <div class="timeline-title">FP8 (E4M3/E5M2) added; INT4 math removed</div>
    <div class="timeline-desc">FP8 ≈ 2× BF16. INT4 now bandwidth-only (dequant to FP16 to multiply). FP8 training arrives.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">Blackwell (2024, B200)</div>
    <div class="timeline-title">FP4 / microscaling (NVFP4, MXFP4, MXFP8)</div>
    <div class="timeline-desc">Native FP4 ≈ 2× FP8 ≈ 4× BF16; block scales decoded in hardware. The current frontier.</div>
  </div>
</div>
</div>

- **INT8 is everywhere** on any modern accelerator — the safe, portable choice.
- **FP8 is Hopper and newer** (and Ada). It tends to beat INT8 on accuracy because its floating-point form handles dynamic range better than fixed-point INT8, which is why FP8 has largely won for *training* in low precision.
- **FP4 / microscaling is Blackwell-only** in NVIDIA's line; the block-scale format (Chapter 23) is what makes 4-bit accurate enough to be usable.

### Kernels: Marlin, CUTLASS, and Why INT4 Packing Needs Special Code

There is no INT4 native storage type in C/CUDA — the smallest addressable unit is a byte. So **INT4 weights are packed two-per-byte**: the high nibble holds one weight, the low nibble another. A GPU load instruction fetches bytes; the kernel must *unpack* the nibbles (shift and mask) before it can use them.

```text
One byte storing two INT4 weights:
 bit:  7 6 5 4 | 3 2 1 0
       └─w[1]─┘ └─w[0]─┘
 unpack:  w0 = byte & 0x0F ;  w1 = (byte >> 4) & 0x0F   (then map 0..15 → signed)
```

This packing is exactly why INT4 needs **special kernels**:
- **Marlin** (and **Machete**, **CUTLASS** mixed-input GEMMs) are hand-tuned kernels that load packed INT4 weights, unpack and dequantize them *in registers/SRAM*, and feed FP16 tensor cores — all fused, so weights never hit HBM as FP16. This is what delivers the ~3–4× decode speedup; a naive unpack-to-HBM-then-matmul gives ~1×.
- The kernel also has to interleave the packed layout cleverly so that the unpack shifts line up with the tensor-core load pattern — which is why you can't just `tensor.to(int4)` and expect speed; you need a kernel that understands the packing. ([Chapter 24](./24_datatypes_and_silicon.md) covers how a multiplier's bit-width drives this.)

> Packing density is a footprint/bandwidth win (2 weights/byte = half the bytes); it is **not** automatically a compute win. The compute path still runs in whatever precision the tensor core supports (FP16 on Hopper for INT4). Keep the wins separate — even here.

---

## Python: Quantize, Simulate the Integer GEMM, and Compute the Savings

Three runnable snippets. First, per-tensor vs per-channel quantization and the error each incurs.

```python
import torch

def quantize_per_tensor_symmetric(x: torch.Tensor, num_bits: int = 8):
    """Symmetric per-tensor affine quant. Returns (q_int, scale)."""
    qmax = 2 ** (num_bits - 1) - 1            # 127 for int8
    amax = x.abs().max()                       # one scale for the whole tensor
    scale = amax / qmax                        # real units per integer step
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return q.to(torch.int8), scale

def quantize_per_channel_symmetric(x: torch.Tensor, num_bits: int = 8, axis: int = 0):
    """Symmetric per-output-channel quant (one scale per row). Returns (q_int, scale[axis])."""
    qmax = 2 ** (num_bits - 1) - 1
    dims = [d for d in range(x.dim()) if d != axis]
    amax = x.abs().amax(dim=dims, keepdim=True)        # per-channel max
    scale = amax / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return q.to(torch.int8), scale

def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale

torch.manual_seed(0)
# A weight matrix [out=512, in=1024] with ONE outlier channel to expose granularity effects.
W = torch.randn(512, 1024)
W[137] *= 40.0          # row 137 is a 40x-magnitude outlier channel

q_t, s_t = quantize_per_tensor_symmetric(W)
q_c, s_c = quantize_per_channel_symmetric(W, axis=0)

err_t = (dequantize(q_t, s_t) - W).abs().mean().item()
err_c = (dequantize(q_c, s_c) - W).abs().mean().item()
print(f"per-tensor  mean abs error: {err_t:.5f}")   # large: outlier stretches the single scale
print(f"per-channel mean abs error: {err_c:.5f}")   # ~40x smaller: outlier row isolated
print(f"per-channel is {err_t / err_c:.1f}x more accurate here")
```

Expected: per-channel is dramatically more accurate because the outlier row gets its own scale instead of poisoning all 512 rows — the hardware mechanism behind "use per-channel weight scales."

Next, simulate an INT8 matmul with **INT32 accumulation** and verify it matches the FP reference:

```python
import torch

def int8_gemm_simulated(Wf: torch.Tensor, Xf: torch.Tensor):
    """
    y = Wf @ Xf, executed the way an INT8 tensor core does it:
      INT8 multiply -> INT32 accumulate -> single float rescale.
    Wf: [N, K] float weights ; Xf: [K, M] float activations.
    """
    qmax = 127
    # symmetric per-tensor scales (kept simple; real kernels use per-channel/per-token)
    s_w = Wf.abs().max() / qmax
    s_x = Xf.abs().max() / qmax
    qW = torch.clamp(torch.round(Wf / s_w), -128, 127).to(torch.int8)
    qX = torch.clamp(torch.round(Xf / s_x), -128, 127).to(torch.int8)

    # The hardware reduction happens in INT32 (use int32 math to model it):
    acc_i32 = qW.to(torch.int32) @ qX.to(torch.int32)     # [N, M] int32 accumulator
    # Single float rescale per output element (amortized O(NM), not O(NMK)):
    y = acc_i32.to(torch.float32) * (s_w * s_x)
    return y, acc_i32

torch.manual_seed(0)
Wf = torch.randn(256, 1024)        # [N=256, K=1024]
Xf = torch.randn(1024, 64)         # [K=1024, M=64]
y_ref = Wf @ Xf                    # FP32 reference

y_int8, acc = int8_gemm_simulated(Wf, Xf)
rel_err = (y_int8 - y_ref).norm() / y_ref.norm()
print(f"INT32 accumulator dtype: {acc.dtype}, max |acc|: {acc.abs().max().item()}")
print(f"INT8-GEMM relative error vs FP32: {rel_err.item():.4f}")   # typically ~0.5-1%
# Note: max|acc| can exceed 2^15, which is WHY accumulation needs int32, not int16.
```

The point of the assertion `max|acc|` is hardware-real: with $K=1024$ and INT8 operands, the accumulator routinely exceeds the 16-bit range, which is the concrete reason tensor cores accumulate in **INT32**.

Finally, the memory & bandwidth savings for a 70B model's decode — the calculation you should be able to do in your head:

```python
def decode_throughput(n_params: float, bytes_per_param: float, hbm_bw_TBps: float):
    """Batch=1 decode: every weight streamed once per token (memory-bound)."""
    bytes_streamed = n_params * bytes_per_param
    t_token_s = bytes_streamed / (hbm_bw_TBps * 1e12)
    return bytes_streamed, t_token_s, 1.0 / t_token_s

N = 70e9                 # 70B params
HBM = 3.35               # H100 HBM3, TB/s
print(f"{'dtype':<8}{'bytes/param':>12}{'GB streamed':>14}{'ms/token':>12}{'tok/s':>10}{'speedup':>10}")
base_t = None
for name, bpp in [("BF16", 2.0), ("INT8", 1.0), ("INT4", 0.5)]:
    nbytes, t, tps = decode_throughput(N, bpp, HBM)
    base_t = base_t or t
    print(f"{name:<8}{bpp:>12.1f}{nbytes/1e9:>14.0f}{t*1e3:>12.1f}{tps:>10.0f}{base_t/t:>10.1f}x")
```

```text
dtype    bytes/param   GB streamed    ms/token     tok/s   speedup
BF16             2.0           140        41.8        24       1.0x
INT8             1.0            70        20.9        48       2.0x
INT4             0.5            35        10.4        96       4.0x
```

That table is the bandwidth win, made concrete: **decode throughput scales inversely with bytes/param** — entirely a consequence of the memory wall, with no tensor-core involvement.

---

## Why This Matters for Model Optimization

Quantization is the highest-leverage optimization you have, but only if you can *predict* what it will buy. The whole point of this chapter is to give you a mental model that answers, before you run a single benchmark: *will this quantization speed up my workload?*

<div class="diagram">
<div class="diagram-title">A Predictive Checklist for Your Quantization</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card green">
    <div class="card-title">1. Will it fit better? (Footprint)</div>
    <div class="card-desc">Always yes. bytes/param drops linearly with bits. Decide what fits in HBM — model + KV cache + activations.</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-title">2. Am I memory-bound? (Bandwidth)</div>
    <div class="card-desc">If decode/small-batch: yes → halving bits ≈ halving time, no datapath needed. Weight-only INT4 is your friend.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">3. Am I compute-bound + have the datapath?</div>
    <div class="card-desc">If prefill/training AND the GPU has the path (FP8 Hopper+, FP4 Blackwell): ~2× throughput. INT4 on Hopper: no.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">4. Is the dequant fused?</div>
    <div class="card-desc">If not (Marlin/CUTLASS), expect ~1×. Unfused dequant doubles traffic and erases the bandwidth win.</div>
  </div>
</div>
</div>

Concretely, the decisions this drives:

1. **Serving an LLM (decode-heavy)?** Reach for **weight-only INT4** with a fused kernel (Marlin/AWQ/GPTQ). You're memory-bound, so the ~3–4× bandwidth win is real even on hardware with no INT4 math, and footprint lets you raise batch and context. Quantize the **KV cache** too — at long context it's often the bigger win.
2. **Prefill- or compute-bound?** You need a **datapath** match: FP8 W8A8 on Hopper/Blackwell for ~2×, FP4 on Blackwell for more. INT4 buys you nothing for compute on Hopper.
3. **Training in low precision?** FP8 (E4M3 for forward, E5M2 for gradients) on Hopper+ gives ~2× compute; its floating-point range tolerates the dynamics better than INT8. INT4/FP4 training is still frontier.
4. **Seeing no speedup after quantizing?** Walk the checklist: wrong regime (compute-bound expecting a bandwidth win), missing datapath (INT4 on Hopper), or — most commonly — an **unfused dequant** doubling your HBM traffic.

The unifying idea, straight from [Chapter 11](./11_memory_wall_and_bandwidth.md): **bytes are the currency.** Quantization spends fewer of them. Whether that translates to speed depends entirely on whether bytes (bandwidth), space (footprint), or math (compute) was your binding constraint — and the next chapter shows that *sparsity*, the other great "do less work" lever, is governed by an even harsher version of the same hardware reality.

## Key Takeaways

- Quantization delivers **three independent wins**: footprint (always), bandwidth (when memory-bound), and compute throughput (only with a native datapath *and* when compute-bound). Conflating them is the #1 source of disappointing results.
- **LLM decode is memory-bound**, so its dominant win is **bandwidth**: halving bits ≈ halving decode time, even with no low-precision math units — INT4 weight-only gives ~3–4× decode speedup purely by streaming fewer bytes.
- **Compute speedups are hardware-gated**: FP8 ≈ 2× on Hopper+, FP4 ≈ 2× more on Blackwell, but INT4 *math* was dropped on Hopper — so INT4 there is bandwidth-only. Always check the datapath table for your GPU.
- The INT8 GEMM works by **INT8 multiply → INT32 accumulate → single float rescale**; the float cost is $O(MN)$, amortized over an $O(MNK)$ integer reduction, which is why it hits ~2× throughput.
- **Granularity is a dial**: per-tensor (cheap, outlier-prone) → per-channel weights (free, standard) → per-group (needed for INT4) → microscaling MX (the scale is part of the format, Ch 23).
- **Activations are the hard half**: outlier channels break naive W8A8, motivating per-token scales and SmoothQuant. Weight-only quant side-steps this entirely.
- **Memory savings are guaranteed; compute speedups are conditional.** And a quantized kernel is only as fast as its **fusion** — an unfused dequant doubles HBM traffic and erases the gain.

---

*Next: [Chapter 26 — Sparsity: The Hardware View](./26_sparsity_hardware_view.md), where we ask the same blunt question of the other great "do less work" lever — and find the hardware answer is far less forgiving.*

[← Back to Table of Contents](./README.md)
