---
title: "Chapter 26 — Sparsity: The Hardware View"
---

[← Back to Table of Contents](./README.md)

# Chapter 26 — Sparsity: The Hardware View

Sparsity is the most seductive optimization in machine learning and the most disappointing in practice. The pitch is irresistible: a trained network has tons of near-zero weights, so prune them, skip the multiply-by-zero, and you get free speed and free memory. The reality, from the hardware's point of view, is that **a GPU does not want to skip work — it wants to do dense, regular work at full tilt.** Multiplying by zero is *cheaper to do than to avoid*, because avoiding it means irregular memory access, idle lanes, and bookkeeping the hardware was never built for. The result is that 90% of your weights can be zero and your matmul runs at *exactly the same speed* as if they were all nonzero.

This chapter explains *why* sparsity is hard for silicon, and *to what extent* the various forms actually pay off. We'll see that the only sparsity that reliably wins on GPUs is **structured** sparsity — patterns regular enough that the hardware can exploit them with cheap, fixed decoders. The flagship example is NVIDIA's **2:4 fine-grained structured sparsity**, which gives up to 2× matmul throughput and explains the riddle from the README: *why 2:4 and not 3:7?* The answer is pure hardware. We build on the memory wall ([Chapter 11](./11_memory_wall_and_bandwidth.md)), the GPU's SIMT execution model ([Chapter 15](./15_gpus.md)), and systolic arrays ([Chapter 16](./16_tpus_and_systolic_arrays.md)) — and we'll compare sparsity head-to-head with quantization ([Chapter 25](./25_quantization_hardware_view.md)) as optimization levers.

> **The one-sentence version:** Hardware loves dense, regular tiles, so *unstructured* sparsity usually buys nothing on a GPU until extreme densities, while *structured* sparsity (2:4, block, channel pruning) trades flexibility for a fixed pattern the silicon can actually skip — turning "fewer nonzeros" into real, but bounded, speedups.

---

## The Promise vs the Reality

The theory says compute scales with the number of nonzero multiply-accumulates. Prune 75% of the weights and you should do 25% of the work. The hardware says otherwise, and to see why we have to look at how a GPU and a systolic array actually execute a matmul.

### Why Hardware Wants Dense, Regular Data

A modern matmul engine — whether a GPU's SIMT tensor cores ([Chapter 15](./15_gpus.md)) or a TPU's systolic array ([Chapter 16](./16_tpus_and_systolic_arrays.md)) — is a machine optimized for one thing: **streaming contiguous tiles of data through a fixed grid of multiply-accumulate (MAC) units at full clock.** Two architectural facts make irregular sparsity poison:

<div class="diagram">
<div class="diagram-title">Why Irregular Sparsity Wrecks a Matmul Engine</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">What the hardware wants</div>
    <ul>
      <li><strong>Coalesced loads</strong>: 32 threads read 32 contiguous addresses in one transaction</li>
      <li><strong>Full MAC-array occupancy</strong>: every PE has data every cycle</li>
      <li><strong>Static schedule</strong>: which element multiplies which is fixed at compile time</li>
      <li><strong>Dense tiles</strong> mapped 1:1 onto the systolic grid</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">What unstructured sparsity gives</div>
    <ul>
      <li><strong>Scattered loads</strong>: nonzeros at random indices → many small transactions</li>
      <li><strong>Idle lanes</strong>: a thread with a zero weight does nothing but can't be reassigned cheaply</li>
      <li><strong>Data-dependent control</strong>: which element is nonzero is known only at runtime</li>
      <li><strong>Index metadata</strong> that must be fetched and decoded alongside values</li>
    </ul>
  </div>
</div>
</div>

1. **SIMT executes in lockstep.** A GPU runs threads in groups of 32 (a *warp*) that share one instruction stream. If thread 5's weight is zero and thread 6's is not, you cannot make thread 5 "skip ahead" — the whole warp moves together. The zero lane simply does a multiply-by-zero. **The work isn't saved; the silicon just idles a lane.** This is the SIMT version of branch divergence.
2. **Systolic arrays are a fixed grid.** A TPU's $128\times128$ MAC array ([Chapter 16](./16_tpus_and_systolic_arrays.md)) pumps data through on a rigid clock. There is no mechanism to "not load" the zeros — the array is space-filling by design. A zero flows through and gets multiplied like everything else.
3. **Memory coalescing breaks.** Even if you *store* only the nonzeros (compressed), reading them back means gathering from scattered addresses. The GPU's memory system delivers peak bandwidth only for *contiguous* 32/128-byte transactions ([Chapter 11](./11_memory_wall_and_bandwidth.md)); a gather over random indices fragments those into many partial transactions, often *reducing* effective bandwidth below the dense case.

### The Honest Numbers on Unstructured Sparsity

So what does unstructured pruning actually do on a GPU?

| Sparsity (% zeros) | Theoretical compute | Realistic dense-GPU matmul speedup | Why |
|:------------------:|:-------------------:|:----------------------------------:|-----|
| 50% | 2× | ~1.0× | Zeros still flow through MAC units |
| 75% | 4× | ~1.0× | Same — no skip mechanism |
| 90% | 10× | ~1.0–1.2× | Maybe a tiny win via cuSPARSE at this density |
| 95% | 20× | ~1.5–2× (specialized SpMM) | Sparse kernels finally beat dense |
| 99%+ | 100× | meaningful | Scientific-computing territory |

The crossover where a sparse kernel beats a dense one on a GPU is typically **>95–99% sparsity** — far beyond what you can prune a neural network to without destroying accuracy (DNNs tolerate maybe 50–90% unstructured sparsity with retraining). In the useful range, **unstructured sparsity gives no GPU speedup.** It can still save *memory* if you store the weights compressed (values + indices), but even there the index overhead eats into the saving, and you must decompress to dense to actually compute.

> **The blunt truth:** on a GPU, multiplying by zero costs the same as multiplying by anything else. Unstructured pruning is a *statistics* win (fewer nonzeros on paper) that the hardware refuses to cash unless the pattern is regular or the density is extreme. This is the opposite of quantization, whose memory win is *unconditional*.

---

## Structured Sparsity: The Hardware-Friendly Compromise

If irregular sparsity is poison because it's *unpredictable*, the fix is to make it *predictable*. Constrain the pattern so the hardware knows, by construction, exactly where the zeros are — then it can build a cheap, fixed decoder to skip them. This is **structured sparsity**, and it is the only kind that reliably speeds up dense accelerators.

### 2:4 Fine-Grained Structured Sparsity

NVIDIA's answer, introduced with **Ampere (A100, 2020)** and present on Hopper and Blackwell, is **2:4 sparsity** (also "2:4 fine-grained structured sparsity" or "semi-structured"): in **every contiguous group of 4 weights, exactly 2 must be zero.** Not "about half" — *exactly* 2 of every 4, enforced as a hard constraint.

```text
Dense row (8 weights):     [ w0  w1  w2  w3 | w4  w5  w6  w7 ]
2:4 pruned (2 of each 4):  [ w0   0   0  w3 | 0  w5  w6   0 ]
                             └── group 0 ──┘  └── group 1 ──┘

Compressed storage (Sparse Tensor Core format):
  values:   [ w0  w3 | w5  w6 ]          ← 2 nonzeros per group of 4  (50% of bytes)
  indices:  [ 0   3  | 1   2  ]          ← which 2 positions, 2 bits each
            (2-bit index per kept value → 4 bits metadata per group of 4)
```

The hardware exploitation is elegant. The **Sparse Tensor Core** holds the compressed values and a tiny 2-bit-per-element index. When it multiplies the weight against the activations, the index tells a small multiplexer *which two* of the four activation elements to actually pair with the two stored weights. The other two products would have been zero, so they're simply never computed:

<div class="diagram">
<div class="diagram-title">How a Sparse Tensor Core Skips the Zeros</div>
<div class="flow">
  <div class="flow-node accent wide">Compressed weights: 2 values + 2-bit indices per group of 4</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Index drives a MUX: select the 2 matching activations from the group of 4</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Only 2 MACs per group of 4 (instead of 4) → half the math</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Up to 2× math throughput + ~50% weight memory & bandwidth</div>
</div>
</div>

The wins, quantified:
- **Up to 2× math throughput** on the matmul — the array does half the MACs, so it finishes in half the time (for the matmul portion).
- **~50% weight memory**: you store 2 values per 4, plus a small 2-bit index per value. Net storage ≈ 50% of dense + ~6% metadata overhead (the indices). On an FP16 weight, the index is 2 bits per kept 16-bit value — about 1/8 overhead on the kept values, so roughly 56% of dense, not 50%, but close.
- **~50% weight bandwidth**: half the bytes streamed from HBM — which, recall from [Chapter 25](./25_quantization_hardware_view.md), is the win that matters for memory-bound decode.

<div class="callout warn">
<span class="callout-title">Read the asterisk on every datasheet</span>
<p>Vendors usually quote tensor throughput <em>with</em> 2:4 sparsity, because it doubles the number. H100's famous "3,958 TFLOPS FP8" is the sparse figure; the dense rate is 1,979. B200's "18 PF FP4" is sparse; dense is 9. Unless your weights are actually pruned to 2:4 and the kernel uses the sparse path, divide the headline by two. This guide quotes dense numbers throughout, and the <a href="./atlas.html">Accelerator Atlas</a> does the same.</p>
</div>

2:4 is no longer NVIDIA-only. AMD's CDNA 3 and CDNA 4 matrix cores (MI300X, MI355X) support the same 2-of-4 pattern, and AWS Trainium2 and Trainium3 accept a family of N:M patterns up to 4 of every 16. The idea has become an industry convention; the 2× asterisk has too.

### Why 2:4 and Not 3:7? The Hardware Reason

This is the question the README poses, and the answer is entirely about the cost of the decoder. The constraint "exactly $N$ of every $M$" must be cheap to encode and cheap to *route in silicon*:

<div class="diagram">
<div class="diagram-title">Why 2:4 Is the Sweet Spot (decoder cost)</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Power-of-two group</div>
    <div class="card-desc">M=4 → 2 bits index each kept value, clean byte alignment. M=7 (3:7) needs awkward 3-bit indices, no clean alignment.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Cheap MUX</div>
    <div class="card-desc">"Pick 2 of 4" is a tiny fixed multiplexer next to each MAC. "Pick 3 of 7" is a bigger, slower selection network — eats the area you saved.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">50% is the useful ratio</div>
    <div class="card-desc">2:4 = exactly 2× — matches the area/throughput budget. Higher ratios (1:4 = 4×) hurt accuracy too much to be worth the harder decoder.</div>
  </div>
</div>
</div>

In short: **the hardware fixed the pattern so the zero-skipping logic could be a trivial, regular, area-cheap circuit.** A 2-of-4 selector is small enough to sit beside every MAC lane without bloating the tensor core. A 3-of-7 selector would be larger and irregular, and the extra area would erode the very throughput gain it was supposed to deliver. 2:4 is the point where the *speedup* (2×) is worth the *decoder cost* (negligible) at an *accuracy hit* the model can usually recover from with retraining. It's a co-design sweet spot, not a magic number.

### Try It: Prune a Weight Tile

The lab prunes a small random weight matrix under four patterns and shows what each keeps. The top grid is the dense matrix with surviving weights bright; the bottom strip is what a 2:4 kernel actually reads, half the values plus a 2-bit position for each. Compare 2:4 with unstructured 50%: the unstructured mask keeps slightly more of the weights' energy (it can keep three large values in one group of four), but its zeros fall anywhere, so the tensor core has nothing to exploit. Then try 1:4 to see how much more energy a 4× pattern throws away.

<div class="lab" data-lab="sparsity"><noscript><div class="lab-noscript">This lab needs JavaScript. It is also on the <a href="./labs.html">Interactive Labs</a> page.</div></noscript></div>

### Block Sparsity, N:M Generalizations, and Channel Pruning

2:4 is one point on a spectrum of structured sparsity, ordered roughly by how much regularity they impose (and thus how reliably the hardware can exploit them):

<div class="diagram">
<div class="diagram-title">The Structured-Sparsity Spectrum (coarser = more hardware-friendly)</div>
<div class="layer-stack">
  <div class="layer red">Unstructured — random zeros. Most flexible, least hardware-friendly. ~No GPU speedup.</div>
  <div class="layer orange">N:M fine-grained (2:4) — fixed ratio per small group. Sparse Tensor Core. ~2×, needs retraining.</div>
  <div class="layer yellow">Block sparsity — whole tiles (e.g. 32×32) zeroed. Dense matmul on surviving blocks → real speedup, coarser accuracy hit.</div>
  <div class="layer green">Channel / head / structured pruning — remove whole output channels, attention heads, or layers. Yields a smaller DENSE model. Most reliably fast.</div>
</div>
</div>

- **Block sparsity** zeros out entire rectangular tiles of the weight matrix. The surviving blocks are dense, so you run ordinary dense matmuls on just those — a genuine speedup on stock hardware (no special tensor core needed), at the cost of a coarser, harder-to-train pattern. Used in some long-sequence attention kernels (block-sparse attention).
- **N:M generalizations** (1:4, 4:8, 2:8…) trade more sparsity for more speedup but at steeper accuracy cost; 2:4 remains the production default because of the decoder economics above.
- **Channel / structured pruning** is the bluntest and, paradoxically, the *most reliably fast*. Remove an entire output channel, attention head, or layer, and you're left with a **smaller dense model** — fewer rows/columns, no special kernel, no metadata, full hardware efficiency on every remaining op. There's no "skip the zero" trickery because there are no zeros: the tensor is just smaller. The cost is that you're removing coarse units, so it's harder to prune deeply without accuracy loss, and it usually needs fine-tuning to recover.

> **The ordering principle:** the coarser and more regular the sparsity, the more reliably the hardware turns it into speed — but the harder it is to impose without hurting accuracy. Channel pruning gives a *dense* small model (always fast); 2:4 gives a fixed micro-pattern (fast on Sparse Tensor Cores); unstructured gives maximum flexibility (and minimum hardware payoff).

---

## To What Extent It Helps — the Honest Story

Time for the same blunt accounting we did for quantization. Sparsity's wins are real but smaller and more conditional than the headline numbers suggest.

### 2:4: ~2× on the Matmul, Less End-to-End

The "up to 2×" figure is the *dense-matmul throughput* ceiling for the sparse-eligible GEMMs. End-to-end model speedup is reliably **less**, for several reasons:

- **Not all of the model is the sparse GEMM.** Attention's softmax, LayerNorm, residual adds, activations, and the KV-cache reads are memory-bound and untouched by weight sparsity (Amdahl's law, [Chapter 8](./08_parallel_processing.md)). If the 2:4-able matmuls are 60% of runtime, a 2× on them is at best a 1.4× overall.
- **Overhead and minimum sizes.** The sparse path has setup cost and only beats dense above certain matrix sizes; small GEMMs may see no gain.
- **You must earn the pattern.** Achieving a *good* 2:4 model is not free pruning — it generally requires **retraining or distillation** to recover accuracy after enforcing the 2-of-4 constraint. NVIDIA's recommended recipe is train dense → prune to 2:4 → retrain. That's real engineering effort, unlike INT8 PTQ which is often a one-shot calibration. One-shot LLM pruners (SparseGPT, Wanda) can impose 2:4 on a trained model in hours without retraining, but they pay a noticeably larger accuracy loss than 50% unstructured pruning does, which is the price of the hardware-friendly pattern.

Realistically, 2:4 delivers something like **1.2–1.6× end-to-end** on a transformer where the linear layers dominate and the hardware (Ampere+) supports it — a worthwhile but not transformative win, and one that costs a retraining cycle.

### The Math of the Effective Speedup

Two formulas let you predict the real gain before you spend the retraining budget. First, **Amdahl's law** ([Chapter 8](./08_parallel_processing.md)) on the matmul fraction. If a fraction $f$ of runtime is the 2:4-eligible GEMMs and the sparse path speeds *those* up by $s$ (≈2× ceiling), the end-to-end speedup is:

$$\text{Speedup}_{\text{e2e}} = \frac{1}{(1 - f) + \dfrac{f}{s}}$$

| Matmul fraction $f$ | $s = 2.0$ (ideal 2:4) | $s = 1.7$ (realistic) |
|:-------------------:|:---------------------:|:---------------------:|
| 0.5 | 1.33× | 1.30× |
| 0.7 | 1.54× | 1.46× |
| 0.9 | 1.82× | 1.69× |

The lesson is stark: even a *perfect* 2× on the matmul gives only 1.33× end-to-end when matmuls are half the runtime. The untouched memory-bound ops — softmax, LayerNorm, KV-cache reads — set a ceiling that no sparse tensor core can break. This is the same Amdahl wall that limits every "speed up one kernel" optimization.

Second, the **storage overhead** of the 2:4 metadata. For a weight stored in $b$ bits, the compressed form keeps half the values plus a 2-bit index per kept value:

$$\frac{\text{sparse bytes}}{\text{dense bytes}} = \frac{0.5\,b + 0.5 \cdot 2}{b} = 0.5 + \frac{1}{b}$$

| Weight precision $b$ | Index overhead $1/b$ | Stored fraction of dense |
|:--------------------:|:--------------------:|:------------------------:|
| FP16 (16 bits) | 6.25% | 56.25% |
| INT8 (8 bits) | 12.5% | 62.5% |
| INT4 (4 bits) | 25% | 75% |

Notice the **metadata tax grows as the values shrink**: a 2-bit index is 6% overhead on a 16-bit value but a full 25% on a 4-bit value. This is why 2:4 + INT4 stacks less cleanly than 2:4 + FP16 — the fixed index cost becomes a larger share of an already-tiny value. The bandwidth win from 2:4 is real but always a bit less than the headline "50%."

### Unstructured Pruning: A Memory Trick, Rarely a Speed Trick

As established, unstructured sparsity on a GPU helps **memory** (if stored compressed) but rarely **speed**. The honest framing:

| Form | Memory win | GPU speed win | Cost to obtain |
|------|:----------:|:-------------:|----------------|
| Unstructured (50–90%) | Yes, if stored compressed (minus index overhead) | ~None below ~95% | Retraining/iterative pruning |
| 2:4 fine-grained | ~50% weight bytes | ~2× matmul, ~1.2–1.6× e2e (Ampere+) | Prune + retrain |
| Block sparse | Proportional to blocks removed | Real, dense matmul on survivors | Harder to train the pattern |
| Channel/head pruning | Yes — smaller dense model | Yes — fully dense efficiency | Fine-tune to recover |

There is one exception worth knowing, and it comes from the memory wall rather than the tensor core. In memory-bound decode, a kernel can *load* unstructured weights in compressed form and *expand* them into dense tiles in shared memory before the tensor cores see them. Flash-LLM (2023) called this "load as sparse, compute as dense": at the 70–90% sparsity levels pruned LLMs reach, it beat dense cuBLAS on the skinny matrix multiplies of decode, because the bottleneck was bytes, not MACs. The multiplies still run dense, zeros included; only the bandwidth win is cashed. That is the same lesson as weight-only INT4 in [Chapter 25](./25_quantization_hardware_view.md).

### 2:4 in the Memory-Bound Regime (the decode bandwidth angle)

There's a quieter win that mirrors the quantization story in [Chapter 25](./25_quantization_hardware_view.md). 2:4 stores ~56% of the dense weight bytes, and during **memory-bound LLM decode** ([Chapter 11](./11_memory_wall_and_bandwidth.md)) the clock is set by bytes streamed from HBM, not by tensor-core throughput. So the half-as-many weight bytes give a *bandwidth* win on decode — independent of whether the Sparse Tensor Core's 2× *math* throughput is even useful in that regime (it isn't, since decode is memory-bound and the math units are idle anyway).

In other words, 2:4 has the same two-faced character as quantization:

- **Compute-bound (prefill, training):** the ~2× *math* throughput of the Sparse Tensor Core is what helps.
- **Memory-bound (decode):** the ~56% *byte* footprint is what helps — fewer weight bytes per token, like a 1.8× quantization on the linears.

The metadata tax means it's ~1.8× of bandwidth rather than a clean 2×, but the principle from the last chapter holds verbatim: **fewer bytes speeds up memory-bound work regardless of the math datapath.** This is why 2:4 + weight quantization is attractive for serving — both shrink the bytes you stream.

### Sparsity + Quantization Combine

The two levers are largely orthogonal and **stack**: a weight can be both 2:4-sparse *and* INT8/INT4. NVIDIA's Sparse Tensor Cores support sparse INT8 and sparse FP8 (and sparse FP4 on Blackwell), so you can in principle get the ~2× quantization throughput *and* the ~2× sparsity throughput on the eligible GEMMs (up to ~4× on the matmul), plus the multiplied memory savings (50% from sparsity × 50% from INT8 = ~25% of dense FP16 bytes). The accuracy cost compounds too, and the practical end-to-end gain is bounded by the non-matmul parts as always — but the levers don't fight each other.

### Activation Sparsity and MoE: The Sparsity That Scales

There's a different flavor of sparsity that the hardware loves: **activation sparsity**, where it's the *activations* (not weights) that are mostly zero, in a *predictable* way.

- **ReLU** produces exact zeros, and historically there were attempts to skip those zero activations. But like unstructured weight sparsity, the zeros are data-dependent and irregular, so dense hardware rarely capitalizes — the same SIMT problem.
- **Mixture-of-Experts (MoE)** is the form of sparsity that *actually scales*, and it's structured at a coarse, hardware-friendly granularity. Instead of zeroing individual weights, MoE routes each token to a small subset of expert FFNs (say 2 of 64). The experts that aren't selected do **no work and load no weights** — you skip entire dense GEMMs, not individual elements. This is "conditional computation": the model has, say, 8× the parameters but activates only ~1/8 per token. Crucially, each selected expert is a *full dense matmul* the hardware runs at peak efficiency; the sparsity is in *which* matmuls run, not *within* a matmul. That coarse granularity is exactly why MoE delivers real, scalable savings where fine-grained sparsity stalls — and why frontier models lean on it. By 2025–26 most large open-weight models are MoE: DeepSeek-V3 activates 37B of its 671B parameters per token (8 of 256 routed experts plus one shared expert), Kimi K2 about 32B of 1T, and OpenAI's gpt-oss-120b about 5.1B of 117B. The hardware bill does not vanish; it moves. All experts must be resident, so capacity rather than FLOPs sets the GPU count, and routing tokens to experts on other GPUs turns into all-to-all traffic ([Chapter 28](./28_multi_gpu_and_distributed_training.md)). (See [Chapter 21 — ASICs](./21_asics.md) for how some accelerators specialize for MoE routing, and the per-token bandwidth implications in [Chapter 29](./29_memory_hierarchy_in_action.md).)

> **The pattern across all of these:** sparsity pays off in proportion to how *coarse and predictable* the skipped work is. Skip individual scattered elements → the hardware can't help. Skip a fixed micro-pattern (2:4) → cheap decoder, ~2×. Skip whole experts/channels/layers → run dense work on what remains, full efficiency. **Coarser structure = more reliable hardware payoff.**

---

## Quantization vs Sparsity as Optimization Levers

Having dissected both, here is the head-to-head an optimizer actually needs.

<div class="diagram">
<div class="diagram-title">Quantization vs Sparsity as Levers</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Quantization — broad, reliable</div>
    <ul>
      <li>Memory/footprint win is <strong>unconditional</strong></li>
      <li>Bandwidth win whenever memory-bound (decode)</li>
      <li>INT8/FP8 often near-lossless, one-shot PTQ</li>
      <li>Works on essentially all modern hardware</li>
      <li>Compute win gated only by datapath presence</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Sparsity — conditional, HW-gated</div>
    <ul>
      <li>Memory win only if stored <strong>compressed</strong></li>
      <li>Speed win needs <strong>structure</strong> (2:4, block, channel)</li>
      <li>2:4 needs Sparse Tensor Cores + <strong>retraining</strong></li>
      <li>Unstructured: ~no GPU speedup below ~95%</li>
      <li>Best reliable form = channel pruning (→ dense small model)</li>
    </ul>
  </div>
</div>
</div>

| Dimension | Quantization | Sparsity |
|-----------|--------------|----------|
| Memory win | Always (linear in bits) | Only if compressed; minus index overhead |
| Speed win — memory-bound | Yes (fewer bytes) | Only if compressed *and* coalesced |
| Speed win — compute-bound | Yes if datapath exists | Only structured (2:4 ~2×, channel = smaller dense) |
| Hardware needed | INT8 everywhere; FP8/FP4 newer | Sparse Tensor Cores (2:4); none for channel pruning |
| Effort to apply | PTQ often one-shot | Usually prune + retrain/distill |
| Accuracy risk | Low (INT8/FP8), rises at INT4 | Moderate; needs recovery training |
| Practitioner verdict | **Do it first** | **Do it if you have the HW + training budget** |

The takeaway: **quantization is the default, sparsity is the specialist.** Quantization gives you a near-guaranteed memory win and a frequently-guaranteed speed win with low effort. Sparsity gives you a *conditional* win that depends on having the right hardware (Sparse Tensor Cores), the right structure (2:4/block/channel), and the budget to retrain. When both apply, stack them — but if you only do one, quantize.

---

## Python: Build a 2:4 Mask, Compress It, and Show Why Unstructured Doesn't Help

First, construct a valid 2:4 mask (keep the 2 largest-magnitude of every 4) and compress to values + indices:

```python
import torch

def make_2to4_mask(W: torch.Tensor) -> torch.Tensor:
    """
    Build a 2:4 structured mask: in every contiguous group of 4 along the
    last dim, keep the 2 largest-|w| and zero the other 2.
    W: [..., K] with K divisible by 4.
    """
    *lead, K = W.shape
    assert K % 4 == 0, "last dim must be divisible by 4 for 2:4"
    g = W.reshape(*lead, K // 4, 4)                      # group into 4s
    # rank magnitudes within each group; keep top-2
    keep = g.abs().argsort(dim=-1, descending=True)[..., :2]   # indices of 2 kept
    mask = torch.zeros_like(g, dtype=torch.bool)
    mask.scatter_(-1, keep, True)
    return mask.reshape(*lead, K)

def compress_2to4(W: torch.Tensor):
    """Return (values, indices): 2 kept values per group + their 2-bit positions."""
    *lead, K = W.shape
    g = W.reshape(*lead, K // 4, 4)
    idx = g.abs().argsort(dim=-1, descending=True)[..., :2].sort(dim=-1).values  # [...,G,2]
    vals = torch.gather(g, -1, idx)                     # [...,G,2] kept values
    return vals, idx                                    # idx in {0,1,2,3}: 2 bits each

torch.manual_seed(0)
W = torch.randn(256, 1024)                              # [N=256, K=1024], FP16-ish
mask = make_2to4_mask(W)
W_sparse = W * mask

print(f"sparsity: {(W_sparse == 0).float().mean().item():.1%}")   # exactly 50%
# verify the hard constraint: every group of 4 has exactly 2 nonzeros
groups = (W_sparse != 0).reshape(256, 256, 4).sum(-1)
print(f"all groups have exactly 2 nonzeros: {(groups == 2).all().item()}")

vals, idx = compress_2to4(W)
# Storage accounting (FP16 values, 2-bit indices):
dense_bytes  = W.numel() * 2                            # FP16
value_bytes  = vals.numel() * 2                         # half the values
index_bits   = idx.numel() * 2                          # 2 bits per kept value
index_bytes  = index_bits / 8
sparse_bytes = value_bytes + index_bytes
print(f"dense:  {dense_bytes/1024:.1f} KB")
print(f"sparse: {sparse_bytes/1024:.1f} KB  ({sparse_bytes/dense_bytes:.1%} of dense)")
```

```text
sparsity: 50.0%
all groups have exactly 2 nonzeros: True
dense:  512.0 KB
sparse: 288.0 KB  (56.2% of dense)   # 50% values + 2-bit index overhead
```

Note the storage is ~56%, not 50% — the 2-bit indices are the metadata tax. The math throughput, on a Sparse Tensor Core, is still up to 2× because only 2 MACs run per group of 4.

Now the crucial demonstration: an **unstructured** mask does *not* speed up a dense matmul, because the zeros still flow through the MAC units:

```python
import torch, time

torch.manual_seed(0)
N, K, M = 4096, 4096, 4096
W = torch.randn(N, K)
X = torch.randn(K, M)

# Unstructured 75%-sparse weights: zero out random 75% of elements.
umask = (torch.rand_like(W) > 0.75)                     # ~25% kept
W_unstructured = W * umask
print(f"unstructured sparsity: {(W_unstructured == 0).float().mean().item():.1%}")

def timed_matmul(A, B, iters=20):
    if torch.cuda.is_available():
        A, B = A.cuda(), B.cuda(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        C = A @ B                                        # DENSE matmul either way
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

t_dense  = timed_matmul(W, X)
t_sparse = timed_matmul(W_unstructured, X)              # SAME dense kernel, zeros included
print(f"dense matmul:               {t_dense*1e3:.2f} ms")
print(f"75%-unstructured (as dense): {t_sparse*1e3:.2f} ms")
print(f"speedup from unstructured zeros: {t_dense/t_sparse:.2f}x  (≈ 1.0 — zeros aren't skipped)")

# Theoretical vs realistic 2:4 speedup on the matmul portion:
print("\n2:4 structured (Sparse Tensor Core, conceptual):")
print(f"  theoretical matmul speedup: 2.00x  (half the MACs)")
print(f"  realistic end-to-end:       ~1.2-1.6x  (Amdahl: non-matmul ops untouched)")
```

```text
unstructured sparsity: 75.0%
dense matmul:               <X> ms
75%-unstructured (as dense): <≈same> ms
speedup from unstructured zeros: ~1.00x  (zeros aren't skipped)
```

We ran an equivalent NumPy version at 2048×2048 on a 4-core cloud CPU: both multiplies took 36–43 ms, and the ratio wandered between 0.90× and 1.18× from run to run. That is timing noise, not a speedup. The result is the chapter in one experiment: **75% of the weights are zero, and the matmul runs at the same speed** — because `A @ B` is a dense kernel that multiplies the zeros anyway. To get a speedup you need either a true sparse kernel (only worth it >95%) or a *structured* pattern with hardware support (2:4 on a Sparse Tensor Core). The theoretical 2× of 2:4 is a matmul-only ceiling; Amdahl's law on the untouched memory-bound ops drags the real number down.

---

## Why This Matters for Model Optimization

Sparsity is where the gap between "FLOPs on paper" and "wall-clock on silicon" is widest. The practitioner who understands the hardware reasons above will reach for the right form of sparsity — and skip the forms that look great in a FLOP count and do nothing on a GPU.

<div class="diagram">
<div class="diagram-title">Choosing a Sparsity Strategy</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card orange">
    <div class="card-title">Want 2× on linear layers?</div>
    <div class="card-desc">Use 2:4 — but only if you have Sparse Tensor Cores (Ampere+) AND can afford prune→retrain. Expect ~1.2–1.6× e2e.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Want reliable speedup, any HW?</div>
    <div class="card-desc">Structured channel/head/layer pruning → a smaller DENSE model. No special kernel, full efficiency, fine-tune to recover.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Want sparsity that scales?</div>
    <div class="card-desc">MoE: route to a few experts, skip whole dense GEMMs. Coarse-grained → hardware-friendly. The sparsity frontier models actually use.</div>
  </div>
</div>
</div>

The decision rules, stated plainly:

1. **Don't expect unstructured pruning to speed up a GPU.** Below ~95% sparsity it won't. If your goal is *memory*, store compressed and accept the index overhead — but you'll decompress to dense to compute, so it's a footprint play, not a speed play.
2. **Reach for 2:4 only when the stars align:** Sparse-Tensor-Core hardware (Ampere/Hopper/Blackwell), linear layers that dominate runtime, and the training budget to prune-then-retrain. Then expect a real but bounded ~1.2–1.6× end-to-end, stackable with INT8/FP8.
3. **Prefer structured channel pruning when you want guaranteed, portable speed.** A smaller dense model is fast *everywhere*, with no metadata and no special kernel — it's the most reliable form, just harder to push deep.
4. **MoE is the sparsity that actually scales.** It skips entire experts (coarse, predictable), running dense matmuls on the chosen ones at peak efficiency — which is why conditional computation, not fine-grained pruning, is how the field grows parameter counts without proportionally growing per-token compute.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>A 2:4-pruned FP16 weight stores 56% of the dense bytes; a 2:4-pruned INT4 weight stores 75% of INT4's bytes. If your serving is memory-bound, which combination of sparsity and quantization gives the fewest bytes per token, and is the accuracy you give up for the sparsity worth more than the accuracy you would give up going one more bit lower instead?</p>
</div>

Step back and the unifying lesson — true for both this chapter and the last — is about **regularity**. Quantization wins broadly because fewer bits is a *uniform, regular* change the hardware handles natively. Sparsity wins *only* to the extent its pattern is regular enough for cheap silicon to exploit. The closer your "do less work" maps onto the dense, contiguous, statically-scheduled shapes the hardware was built for, the more of the theoretical saving you actually collect. That principle — *the hardware rewards regular work* — is exactly what made one architecture, built for regular parallel arithmetic, eat the entire field. That's the next chapter.

## Key Takeaways

- On a GPU, **multiplying by zero costs the same as any multiply** — SIMT lockstep and fixed systolic arrays have no cheap way to skip scattered zeros, so **unstructured sparsity gives ~no speedup below ~95%** density.
- **Unstructured sparsity is a memory trick, not a speed trick**: it saves bytes only if stored compressed (minus index overhead), and you decompress to dense to compute.
- **2:4 structured sparsity** (exactly 2 zeros per group of 4) is the hardware-friendly compromise: a tiny 2-bit-index MUX lets Sparse Tensor Cores (Ampere+) skip the zeros for **up to 2× matmul throughput and ~50% weight bytes**.
- **Why 2:4 and not 3:7:** a power-of-two group gives 2-bit indices and a trivial, area-cheap "pick 2 of 4" selector. A 3:7 selector would be larger and irregular, eroding the gain. It's a decoder-cost sweet spot.
- **End-to-end 2:4 speedup is ~1.2–1.6×**, not 2×, because non-matmul ops are untouched (Amdahl) and you must **prune-then-retrain** to recover accuracy.
- **Coarser structure = more reliable hardware payoff**: channel/head pruning yields a *smaller dense model* (fast everywhere), and **MoE** skips whole expert GEMMs — the structured sparsity that actually scales.
- **Headline TFLOPS usually include the 2:4 bonus**; halve them unless your model is actually 2:4-pruned.
- As levers: **quantization is broad and reliable; sparsity is conditional and hardware-gated.** Quantize first; add sparsity when you have the Sparse Tensor Cores and the training budget. They stack.

## Check Yourself

<details class="check">
<summary>A datasheet says 4,500 TFLOPS FP8. You run dense, unpruned weights. What number should you plan with if the vendor quoted sparse throughput, and how can you tell?</summary>
<p>2,250 TFLOPS. Look for "with sparsity" in the footnote, or check whether the FP8 figure is exactly twice what the vendor lists as dense elsewhere. Planning with the sparse number overestimates compute-bound speed by 2×.</p>
</details>

<details class="check">
<summary>Matmuls are 60% of a model's runtime. With a realistic 1.7× from 2:4 on those matmuls, what end-to-end speedup does Amdahl's law predict?</summary>
<p>1 / (0.4 + 0.6/1.7) = 1 / (0.4 + 0.353) ≈ 1.33×.</p>
</details>

<details class="check">
<summary>Why does a 2:4-sparse INT4 weight save proportionally less memory than a 2:4-sparse FP16 weight?</summary>
<p>The index costs a fixed 2 bits per kept value. Against a 16-bit value that is 12.5% of the kept bytes (56.25% of dense overall); against a 4-bit value it is 50% of the kept bytes (75% of dense overall).</p>
</details>

<details class="check">
<summary>Your 75%-unstructured-pruned model runs no faster in prefill but does run faster in batch-1 decode with a special kernel. Explain both observations.</summary>
<p>Prefill is compute-bound and the dense tensor cores still multiply every zero, so nothing changes. Decode is memory-bound; a load-as-sparse, compute-as-dense kernel reads roughly a quarter of the weight bytes (plus indices) and expands them on chip, so the bandwidth win shows up even though the math is dense.</p>
</details>

<details class="check">
<summary>Why is Mixture-of-Experts called sparsity, and why does the hardware like it when it dislikes unstructured zeros?</summary>
<p>Each token activates only a few experts, so most parameters do no work for that token. The skipped work is whole dense GEMMs, decided once per token, so every matmul that does run is dense and runs at full tensor-core efficiency. The costs move to memory capacity and all-to-all communication.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Accelerating Sparse Deep Neural Networks | Mishra et al. (NVIDIA), 2021 | [arxiv.org/abs/2104.08378](https://arxiv.org/abs/2104.08378) | 2:4 format, metadata, prune-retrain recipe |
| NVIDIA A100 Tensor Core GPU Architecture | NVIDIA, 2020 | [images.nvidia.com/aem-dam/en-zz/Solutions/dat…](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) | Sparse Tensor Core and the 2× sparse throughput |
| Sparse GPU Kernels for Deep Learning | Gale et al., 2020 | [arxiv.org/abs/2006.10901](https://arxiv.org/abs/2006.10901) | Where unstructured SpMM beats dense on GPUs |
| Sparsity in Deep Learning (survey) | Hoefler et al., 2021 | [arxiv.org/abs/2102.00554](https://arxiv.org/abs/2102.00554) | Taxonomy of structured and unstructured sparsity |
| SparseGPT | Frantar & Alistarh, 2023 | [arxiv.org/abs/2301.00774](https://arxiv.org/abs/2301.00774) | One-shot 50% and 2:4 pruning of LLMs |
| Wanda: A Simple and Effective Pruning Approach for LLMs | Sun et al., 2023 | [arxiv.org/abs/2306.11695](https://arxiv.org/abs/2306.11695) | Magnitude × activation pruning without retraining |
| Flash-LLM | Xia et al., 2023 | [arxiv.org/abs/2309.10285](https://arxiv.org/abs/2309.10285) | Load-as-sparse, compute-as-dense decode kernels |
| Switch Transformers | Fedus et al., 2022 | [arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961) | Mixture-of-Experts as conditional computation |
| DeepSeek-V3 Technical Report | DeepSeek-AI, 2024 | [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437) | 671B total / 37B active MoE configuration |

---

**Next:** [Chapter 27 — Why GPUs Won AI](./27_why_gpus_won_ai.md) — The thread running through these two chapters, that hardware rewards regular, parallel, dense arithmetic, explains how a graphics chip became the engine of modern machine learning.

[← Back to Table of Contents](./README.md)
