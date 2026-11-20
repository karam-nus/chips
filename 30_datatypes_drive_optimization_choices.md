---
title: "Chapter 30 — How Datatypes Drive Your Daily Optimization Choices"
---

[← Back to Table of Contents](./README.md)

# Chapter 30 — How Datatypes Drive Your Daily Optimization Choices

This is the chapter the rest of the guide was building toward. You now know what a transistor is ([Chapter 1](./01_what_is_a_chip.md)), how memory is a hierarchy of speed-vs-capacity tradeoffs ([Chapter 9](./09_memory_types.md)), why bandwidth — not FLOPs — usually bounds you ([Chapter 11](./11_memory_wall_and_bandwidth.md)), what FP8/INT4/NVFP4 actually are ([Chapter 23](./23_number_formats_deep_dive.md)), how a format reshapes the silicon and which generation supports which type ([Chapter 24](./24_datatypes_and_silicon.md)), how much quantization ([Chapter 25](./25_quantization_hardware_view.md)) and sparsity ([Chapter 26](./26_sparsity_hardware_view.md)) really buy, and how KV-cache, batching, and activations spend your memory budget ([Chapter 29](./29_memory_hierarchy_in_action.md)).

Here we tie all of it into a single **decision framework**: when you sit down on Monday morning to make a model fit, run faster, or cost less, *which* precision and *which* technique do you reach for — and why? The thesis is simple and, once you see it, hard to unsee: **every optimization you can perform is buying back exactly one of three scarce hardware resources — memory capacity, memory bandwidth, or compute throughput — and you pay for it in the only currency the model has, accuracy.** Name the resource you are short of, and the right tool is nearly always obvious.

> **The one-sentence version:** Pick the optimization that buys back the resource you are actually short of (capacity, bandwidth, or compute), spend the least accuracy to get it, and never pick a datatype your GPU can't accelerate.

---

## The Unifying Mental Model: Three Resources and One Currency

Strip away the jargon and a chip offers a model exactly three finite things, plus one thing the model itself owns:

<div class="diagram">
<div class="diagram-title">The Four Quantities Every Optimization Touches</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Memory capacity (GB)</div>
    <div class="card-desc">Can the weights + KV-cache + activations even fit? The "won't fit on one GPU" problem. Measured in GB of HBM.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Memory bandwidth (TB/s)</div>
    <div class="card-desc">How fast can bytes stream from HBM? Sets decode token rate. The "too slow at batch=1" problem (<a href="./11_memory_wall_and_bandwidth.html">Ch 11</a>).</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Compute throughput (FLOP/s)</div>
    <div class="card-desc">How fast can the tensor cores multiply? Sets training step time and prefill latency. The "compute-bound" problem.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Accuracy (the currency)</div>
    <div class="card-desc">What you spend to buy the above. Every lossy technique trades some model quality for capacity, bandwidth, or compute. The skill is spending the least.</div>
  </div>
</div>
</div>

Now map every technique in this guide to the resource it buys and the price it charges. **This single table is the chapter** — everything else is elaboration and worked examples.

| Technique | Buys back | How much | Accuracy risk | Hardware requirement |
|-----------|-----------|----------|---------------|----------------------|
| **BF16 instead of FP32** | compute + bandwidth + capacity | ~2× all three | ~none (training-stable range) | any modern tensor core (Ampere+) |
| **FP8 (E4M3/E5M2)** | compute + bandwidth | ~2× over BF16 | low–moderate; needs scaling | Hopper / Blackwell ([Ch 24](./24_datatypes_and_silicon.md)) |
| **INT8 weight+activation (W8A8)** | bandwidth + capacity (+ compute) | ~2× | low for weights, moderate for activations (outliers) | INT8 tensor cores (Turing+) |
| **INT4 / NVFP4 weight-only (W4A16)** | capacity + bandwidth | ~4× memory; ~up to 4× decode | moderate; needs calibration (GPTQ/AWQ) | fast INT4/FP4 path or dequant ([Ch 25](./25_quantization_hardware_view.md)) |
| **KV-cache quant (INT8/FP8)** | capacity (KV) + bandwidth | 2–4× smaller KV | low–moderate (activation outliers) | any; targets long context ([Ch 29](./29_memory_hierarchy_in_action.md)) |
| **2:4 structured sparsity** | compute + capacity | ~2× matmul; ~1.8× smaller weights (FP16) | needs sparse-aware retraining | Sparse Tensor Cores (Ampere+) ([Ch 26](./26_sparsity_hardware_view.md)) |
| **GQA / MQA** | capacity (KV) + bandwidth | $n_{heads}/n_{kv}$× smaller KV | small if trained in | none (architectural) |
| **Distillation / smaller model** | all three | proportional to size cut | task-dependent | none |
| **Tensor / pipeline / FSDP parallelism** | capacity (per-GPU) | ÷ number of GPUs | none (exact) | multi-GPU + interconnect ([Ch 28](./28_multi_gpu_and_distributed_training.md)) |
| **Gradient checkpointing** | capacity (activations, training) | $O(L) \to O(\sqrt{L})$ | none (exact) | ~+33% compute |
| **Operator fusion (FlashAttention, torch.compile)** | bandwidth | removes HBM round-trips | none (exact) | none |

> The two columns that matter most are **"buys back"** and **"accuracy risk."** Optimizations in the bottom group (parallelism, checkpointing, fusion, GQA) are **exact** — they cost zero accuracy and you should reach for them first. The lossy ones (lower precision, quantization, sparsity, distillation) are where you spend the accuracy currency, and they should be applied in proportion to how short you actually are.

---

## Are You Compute-Bound or Memory-Bound? Start at the Roofline

Before choosing a technique, classify your workload. The roofline ([Chapter 11](./11_memory_wall_and_bandwidth.md)) splits the world in two, and the split dictates *which resource a technique even helps with*:

<div class="diagram">
<div class="diagram-title">The Roofline Decides Which Techniques Matter</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Memory-bound (AI &lt; ridge ≈ 295)</div>
    <ul>
      <li>LLM <strong>decode</strong> at small batch, elementwise ops, GEMV</li>
      <li>Bottleneck = <strong>bytes streamed from HBM</strong></li>
      <li>Win by moving <em>fewer bytes</em>: quantize weights/KV, fuse, batch</li>
      <li>Extra dequant FLOPs are <strong>free</strong> (compute is idle)</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Compute-bound (AI &gt; ridge ≈ 295)</div>
    <ul>
      <li>Training, <strong>prefill</strong>, large-batch GEMMs</li>
      <li>Bottleneck = <strong>tensor-core throughput</strong></li>
      <li>Win by doing <em>cheaper math</em>: FP8/INT8 matmul, 2:4 sparsity</li>
      <li>Saving bytes alone barely helps (bandwidth isn't the limit)</li>
    </ul>
  </div>
</div>
</div>

This is why the same word "quantization" means different things in the two regimes. **Memory-bound:** INT4 weight-only gives ~4× speedup purely by reading 4× fewer bytes — the matmul stays in FP16 after dequant, and that's fine because the multipliers were idle. **Compute-bound:** you need a format the *tensor cores execute natively* (FP8, INT8, 2:4) so the math itself gets cheaper; merely storing weights smaller does nothing for a compute-bound GEMM.

---

## The Practitioner Decision Tree

Start with the first decision, which family of formats, and then walk the detailed branches below it. Each branch ends in a concrete precision/technique recommendation.

<!-- flowchart:dtype_picker -->
<div class="flowchart" id="fc-dtype_picker">
<div class="fc-title">Which number format? A first decision</div>
<svg viewBox="0 0 780 440" width="780" role="img" aria-label="Which number format? A first decision">
<path class="fc-edge" d="M390,40 L390,71"/><polygon class="fc-head" points="390,71 386.1,64 393.9,64"/>
<path class="fc-edge yes" d="M350,113 L350,138 L195,138 L195,163"/><polygon class="fc-head yes" points="195,163 191.2,156 198.8,156"/><text class="fc-elabel yes" x="300" y="140" text-anchor="end">training</text>
<path class="fc-edge no" d="M430,113 L430,138 L585,138 L585,163"/><polygon class="fc-head no" points="585,163 581.1,156 588.9,156"/><text class="fc-elabel no" x="480" y="140" text-anchor="start">inference</text>
<path class="fc-edge yes" d="M145,217 L145,246.5 L105,246.5 L105,276"/><polygon class="fc-head yes" points="105,276 101.2,269 108.8,269"/><text class="fc-elabel yes" x="128" y="252" text-anchor="end">yes</text>
<path class="fc-edge no" d="M245,217 L245,246.5 L290,246.5 L290,276"/><polygon class="fc-head no" points="290,276 286.1,269 293.9,269"/><text class="fc-elabel no" x="267" y="252" text-anchor="start">no</text>
<path class="fc-edge yes" d="M535,217 L535,246.5 L485,246.5 L485,276"/><polygon class="fc-head yes" points="485,276 481.1,269 488.9,269"/><text class="fc-elabel yes" x="508" y="252" text-anchor="end">yes</text>
<path class="fc-edge no" d="M635,217 L635,246.5 L680,246.5 L680,276"/><polygon class="fc-head no" points="680,276 676.1,269 683.9,269"/><text class="fc-elabel no" x="657" y="252" text-anchor="start">no: compute-bound</text>
<g class="fc-node start"><rect x="280" y="8" width="220" height="32" rx="16"/><text x="390" y="28" text-anchor="middle">Pick a number format</text></g>
<g class="fc-node decision"><polygon points="270,92 284,71 496,71 510,92 496,113 284,113"/><text x="390" y="96" text-anchor="middle">Training or inference?</text></g>
<g class="fc-node decision"><polygon points="45,190 59,163 331,163 345,190 331,217 59,217"/><text x="195" y="188" text-anchor="middle">FP8 tensor cores available?</text><text class="sub" x="195" y="201" text-anchor="middle">H100+, MI300+, Gaudi 3, Ironwood</text></g>
<g class="fc-node decision"><polygon points="435,190 449,163 721,163 735,190 721,217 449,217"/><text x="585" y="188" text-anchor="middle">Small-batch decode?</text><text class="sub" x="585" y="201" text-anchor="middle">memory-bound, latency-sensitive</text></g>
<a href="./23_number_formats_deep_dive.html"><g class="fc-node leaf k-green"><rect x="10" y="276" width="190" height="56" rx="6"/><text x="105" y="296" text-anchor="middle">BF16 master + FP8 GEMMs</text><text class="sub" x="105" y="309" text-anchor="middle">Transformer Engine or MXFP8;</text><text class="sub" x="105" y="321" text-anchor="middle">FP32 optimizer state</text></g></a>
<a href="./03_number_representation.html"><g class="fc-node leaf k-blue"><rect x="205" y="276" width="170" height="56" rx="6"/><text x="290" y="296" text-anchor="middle">BF16 mixed precision</text><text class="sub" x="290" y="309" text-anchor="middle">FP32 master weights;</text><text class="sub" x="290" y="321" text-anchor="middle">no loss scaling needed</text></g></a>
<a href="./25_quantization_hardware_view.html"><g class="fc-node leaf k-purple"><rect x="390" y="276" width="190" height="56" rx="6"/><text x="485" y="296" text-anchor="middle">Weight-only 4-bit + FP8 KV</text><text class="sub" x="485" y="309" text-anchor="middle">AWQ/GPTQ INT4 or NVFP4;</text><text class="sub" x="485" y="321" text-anchor="middle">W8A16 if 4-bit hurts</text></g></a>
<a href="./24_datatypes_and_silicon.html"><g class="fc-node leaf k-orange"><rect x="585" y="276" width="190" height="56" rx="6"/><text x="680" y="296" text-anchor="middle">W8A8 FP8, or W4A4 FP4</text><text class="sub" x="680" y="309" text-anchor="middle">FP4 only where tensor cores</text><text class="sub" x="680" y="321" text-anchor="middle">run it (Blackwell, MI355X, TPU 8)</text></g></a>
<a href="./26_sparsity_hardware_view.html"><g class="fc-node leaf k-teal"><rect x="90" y="384" width="600" height="32" rx="6"/><text x="390" y="404" text-anchor="middle">Then: 2:4 sparsity if accuracy survives fine-tuning; re-run the eval after every step</text></g></a>
</svg>
<div class="fc-caption">Defaults for transformer workloads as of late 2026. Every arrow ends in an experiment: confirm the choice on your own evaluation, not just perplexity.</div>
</div>
<!-- /flowchart:dtype_picker -->

<div class="diagram">
<div class="diagram-title">What Should I Do Monday Morning?</div>
<div class="flow">
  <div class="flow-node accent wide">START: Does the model + KV-cache + activations FIT in HBM? (run the Ch 29 budget)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node red wide">NO → capacity problem. In order of cheapness: GQA & KV-quant (inference) / checkpointing (training) → weight quant INT8→INT4 → tensor/pipeline/FSDP shard across GPUs (Ch 28)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">YES → Is it TRAINING or INFERENCE?</div>
</div>
</div>

<div class="diagram">
<div class="diagram-title">Branch A — Training</div>
<div class="flow">
  <div class="flow-node blue wide">TRAINING → default to BF16 compute + FP32 master weights (range &amp; stability)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">On Hopper/Blackwell &amp; compute-bound? → enable FP8 for the matmuls (≈2× over BF16), keep FP32 master weights &amp; loss scaling</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Activations don't fit? → gradient checkpointing → then CPU/NVMe offload → then shard (Ch 28)</div>
</div>
</div>

<div class="diagram">
<div class="diagram-title">Branch B — Inference</div>
<div class="flow">
  <div class="flow-node teal wide">INFERENCE → Prefill (compute-bound) or Decode (memory-bound)? Usually both — optimize each.</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">DECODE, low latency / single user (memory-bound): weight-only INT4/NVFP4 (≈4× fewer bytes) + INT8/FP8 KV-cache + GQA. Smallest model that meets quality.</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">THROUGHPUT serving (many users): batch hard (continuous batching, PagedAttention) until compute-bound, then W8A8 INT8 or FP8 so the now-saturated math is cheap.</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Have Sparse Tensor Cores + retraining budget + still compute-bound? → add 2:4 sparsity for ~2× on top.</div>
</div>
</div>

### When to Pick Each Format — the One-Liner Each

| Format | Pick it when… | Why |
|--------|---------------|-----|
| **FP32** | numerical reference / tiny models / debugging | full range + precision; ~no hardware ML acceleration |
| **BF16** | **training default** | same 8-bit exponent range as FP32 (no loss-scaling drama), half the bytes, native on every modern tensor core |
| **FP16** | legacy training / older GPUs | half bytes, but 5-bit exponent → narrow range → needs loss scaling; superseded by BF16 for training |
| **FP8 (E4M3/E5M2)** | Hopper+ and **compute-bound** (training matmuls, prefill) | ~2× compute & bandwidth over BF16; needs per-tensor scaling |
| **INT8 (W8A8)** | mature **inference**, throughput serving, edge | ~2× everything, widest hardware support; activations need care |
| **INT4 / NVFP4** | **memory-bound decode**, local/single-GPU, weight-only | ~4× memory & bandwidth; accept calibration (GPTQ/AWQ) and some quality work |
| **2:4 sparsity** | have **Sparse Tensor Cores** + a retraining budget, compute-bound | ~2× matmul + ~1.8× smaller weights; structure must be retrained in |

---

## Why Each Default Exists — the Reasoning You Should Internalize

The defaults the field has converged on are not arbitrary; each encodes one of the three-resource tradeoffs. Understanding *why* lets you break the default when your situation differs.

### Why training defaults to BF16 + FP32 master weights

Training needs **dynamic range**, not precision. Gradients span many orders of magnitude; weight updates can be tiny relative to weights. **BF16** keeps FP32's full 8-bit exponent (range ≈ $10^{\pm38}$) while halving storage and doubling tensor-core throughput — so it almost never underflows/overflows and needs no loss-scaling babysitting (unlike FP16, whose 5-bit exponent has range only ≈ $10^{\pm4.8}$). But BF16's 7-bit mantissa is too coarse for **accumulating** millions of small updates, so the optimizer keeps an **FP32 master copy** of the weights (and FP32 optimizer moments): compute in BF16 for speed, accumulate in FP32 for correctness. This is mixed precision, and it is the training default precisely because it spends almost no accuracy to buy ~2× compute and memory.

### Why inference loves weight-only INT4 for local/memory-bound use

Local LLM inference is **batch=1 decode → memory-bound** ([Chapter 29](./29_memory_hierarchy_in_action.md)). The token rate is `HBM_BW / model_bytes`, so the single most effective lever is **fewer weight bytes**. INT4 weight-only (W4A16) cuts weight bytes 4× → up to ~4× decode throughput *and* makes a 70B model fit where BF16 (140 GB) couldn't. Activations stay in FP16, so accuracy loss is contained to the weights (which quantize well with GPTQ/AWQ calibration). You don't need fast INT4 *math* — you're memory-bound, the dequant FLOPs are free.

### Why activations are harder to quantize than weights

Weights are a fixed, well-behaved distribution you can calibrate offline. **Activations are dynamic and have outliers** — a few channels in LLMs carry values 10–100× the median, and quantizing the whole tensor to that range crushes the rest into a handful of levels. This is why **weight-only** quantization (W4A16, W8A16) is easy and popular, while **activation** quantization (W8A8, and especially below 8 bits) needs tricks: per-channel/per-token scaling, outlier handling (SmoothQuant-style migration of scale from activations to weights), or simply keeping activations at higher precision. It is also why the **KV-cache** (which *is* activations) quantizes to INT8/FP8 but rarely cleanly to INT4. ([Chapter 25](./25_quantization_hardware_view.md) covers the mechanisms.)

### Why KV-cache quantization specifically targets long context

The KV-cache is negligible at short context and **dominates at long context** (about 104% of the weights for Llama 3 8B at 128k tokens; multiples of the weights at high batch — [Chapter 29](./29_memory_hierarchy_in_action.md)). So KV quantization is the right lever exactly and only when context/batch are large enough that the cache, not the weights, is your capacity-and-bandwidth problem. At 2k context it's not worth the accuracy risk; at 128k it's often the difference between fitting and not.

### Why FP8 needs scaling

FP8 has very few bits (E4M3 = 4 exponent, 3 mantissa). Its representable range is narrow, so tensors must be **scaled** into that range per-tensor (or per-block, as in MXFP/microscaling — [Chapter 23](./23_number_formats_deep_dive.md)) before casting, and unscaled after. Without scaling, values overflow to inf or underflow to zero and training diverges. Hopper's Transformer Engine automates this with running amax statistics. The lesson: FP8 is a ~2× compute win *if* the framework manages scaling; it is not a drop-in cast.

### Why you must not pick a format your GPU can't accelerate

A datatype is only fast if the **silicon has a datapath for it** ([Chapter 24](./24_datatypes_and_silicon.md)). A100 and older data-center GPUs have no FP8 math: FP8 there can only be a *storage* format that is upcast to FP16 before the multiply (Ada-generation cards such as the L40S and RTX 4090 do have FP8 tensor cores). INT4 *math* needs an INT4 path; on hardware without one, INT4 is a *storage* format that you dequantize to FP16 to compute (still great for memory-bound decode, useless for compute-bound). 2:4 sparsity needs **Sparse Tensor Cores** (Ampere+); without them the structured zeros buy storage but no compute. **Always check the support matrix before choosing a format** — the wrong choice can be a slowdown.

<div class="diagram">
<div class="diagram-title">Format × Hardware: Is There a Native Datapath?</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Ampere (A100, 2020)</div>
    <div class="card-desc">FP32, TF32, BF16, FP16, INT8, INT4, 2:4 sparse. No FP8.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Hopper (H100, 2022)</div>
    <div class="card-desc">Adds <strong>FP8</strong> (E4M3/E5M2) + Transformer Engine. BF16/FP16/INT8/2:4 as before.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Blackwell (B200, 2024)</div>
    <div class="card-desc">Adds <strong>FP4 / NVFP4</strong> microscaling + FP6; 2nd-gen FP8. Native low-precision throughput leaps.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Blackwell Ultra · Rubin (2025–26)</div>
    <div class="card-desc">FP4 rate ≈ 3× FP8 on B300; INT8 and FP64 cut back. Rubin is NVFP4-first. INT4 math is gone.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">AMD CDNA 3 → 4 (MI300 → MI355X)</div>
    <div class="card-desc">FP8 from MI300 (FNUZ variant), OCP FP8 + FP6 + FP4 on MI355X, with FP6 at the FP4 rate.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Google TPU (v5 → Ironwood → TPU 8)</div>
    <div class="card-desc">BF16 + INT8 through v6e; FP8 on Ironwood; native FP4 arrives with TPU 8.</div>
  </div>
</div>
</div>

---

## Honest Tradeoff Table: Speedup vs Accuracy Risk

Numbers below are *typical* on supported hardware for transformer LLMs; your mileage varies with model and task. "Speedup" is the regime where the technique actually helps.

| Technique | Resource bought | Typical speedup / saving | Accuracy risk | Hardware needed |
|-----------|-----------------|--------------------------|---------------|-----------------|
| FP32 → BF16 | compute, bw, capacity | ~2× compute, 2× memory | negligible | Ampere+ |
| BF16 → FP8 (training/prefill) | compute, bw | ~1.6–2× compute | low w/ scaling; watch loss curves | Hopper+ |
| FP16/BF16 → INT8 W8A8 (decode/serving) | bw, capacity, compute | ~1.5–2× | low (good calibration); activation outliers the risk | Turing+ |
| INT8 → INT4 weight-only (decode) | capacity, bw | up to ~2× more (4× vs FP16) | moderate; GPTQ/AWQ recover most | dequant path or FP4 |
| KV-cache FP16 → INT8 | capacity (KV), bw | 2× KV; enables larger batch/context | low–moderate (long-context quality) | any |
| Dense → 2:4 sparse | compute, capacity | ~2× matmul (compute-bound only) | needs sparse retraining; ~recoverable | Sparse Tensor Cores |
| Full model → distilled smaller | all three | proportional (e.g. 70B→13B ≈ 5×) | task-dependent, can be large | none |
| Single GPU → FSDP/TP shard | per-GPU capacity | fits N× bigger (adds comm cost) | none (exact) | NVLink/IB, N GPUs |

> Read the **accuracy-risk** column as the price tag. "Negligible/none" techniques (BF16, fusion, checkpointing, sharding, GQA) are nearly free — apply them by default. The lossy ones go on a budget: spend INT8 freely, INT4/FP8 with validation, 2:4 only when you have the retraining budget and the Sparse Tensor Cores to cash it in.

---

## Worked Scenarios: Reasoning End to End

Each scenario walks the decision tree and cites the chapter behind each step. These are the templates you adapt Monday morning.

### (a) Serve a 70B model on 1×80 GB GPU, low latency (single user)

- **Fit check ([Ch 29](./29_memory_hierarchy_in_action.md)):** BF16 weights = 140 GB > 80 GB. **Won't fit.** Capacity problem first.
- **Regime ([Ch 11](./11_memory_wall_and_bandwidth.md)):** single user → batch=1 decode → **memory-bound**.
- **Choice:** **INT4 weight-only** → 35 GB weights, fits with room for KV-cache. Decode ceiling jumps from infeasible to `3.35 TB/s ÷ 35 GB ≈ 96 tok/s` ([Ch 29](./29_memory_hierarchy_in_action.md)). Add **FP8 or INT8 KV-cache + GQA** so context room is generous. Activations stay FP16; accept GPTQ/AWQ calibration work ([Ch 25](./25_quantization_hardware_view.md)). On a Blackwell-class GPU the 4-bit format of choice is **NVFP4**: same bytes, native math, and its finer FP8 scales usually recover more accuracy than INT4.
- **Why not FP8/INT8?** They'd need 70 GB (INT8) — fits but leaves little KV room and only ~48 tok/s. INT4 is the latency and capacity win here because we're memory-bound, so the extra dequant is free.

### (b) Max throughput batch serving (many users, throughput per GPU-dollar)

- **Regime:** batch hard → AI ≈ batch → push past the ridge (~295) → **compute-bound** ([Ch 11](./11_memory_wall_and_bandwidth.md)).
- **Choice:** **continuous batching + PagedAttention** ([Ch 29](./29_memory_hierarchy_in_action.md)) to raise batch until HBM (KV-cache) fills; then **W8A8 INT8** (or **FP8** on Hopper+) so the now-saturated matmuls are cheap. KV-cache quant + GQA to fit *more* concurrent sequences (batch is throughput here).
- **Why not INT4 weight-only?** Once compute-bound, weight bytes aren't the bottleneck; you need cheaper *math* (INT8/FP8 native datapath), not just smaller storage.

### (c) On-device 7B on a phone NPU

- **Constraints:** tiny memory (a few GB), strict power budget, NPU often INT8/INT4-native, no FP8 ([Ch 17](./17_npus_and_edge_ai.md), [Ch 24](./24_datatypes_and_silicon.md)).
- **Choice:** **INT4 weight-only** (3.5 GB fits; matches mobile memory and the energy advantage of fewer-bit switching, $E \propto CV^2$ from [Ch 1](./01_what_is_a_chip.md)), INT8 activations where the NPU accelerates them. Possibly distill to a smaller model. Verify the NPU's actual supported formats — pick what its datapath accelerates, not what's theoretically smallest.

### (d) Train a model that won't fit

- **Fit check ([Ch 29](./29_memory_hierarchy_in_action.md)):** weights + FP32 master + optimizer moments + **activations** (which dominate at long sequence) exceed HBM.
- **Order of operations (cheapest accuracy-free lever first):** **gradient checkpointing** (~+33% compute, $O(\sqrt{L})$ activation memory) → **FP8 matmuls** if Hopper+ and compute-bound (2× compute, keep FP32 master) → **FSDP/ZeRO sharding** of params/grads/optimizer across GPUs ([Ch 28](./28_multi_gpu_and_distributed_training.md)) → **CPU/NVMe offload** as the last resort (capacity at steep bandwidth cost). Keep BF16 + FP32 master throughout for stability.

---

## Python: A Tiny Optimization Advisor

Give it the model size, your GPU's two numbers (capacity + bandwidth), and the regime; it names the **binding constraint** and suggests a precision/technique. It encodes the decision tree above. No dependencies.

```python
from dataclasses import dataclass

@dataclass
class GPU:
    name: str
    hbm_gb: float          # capacity
    hbm_bw_tbs: float      # bandwidth (TB/s)
    bf16_tflops: float     # dense BF16 tensor peak
    has_fp8: bool          # Hopper+ / MI300+
    has_fp4: bool          # Blackwell+ / MI355X+
    has_sparse: bool       # 2:4 Sparse Tensor Cores (Ampere+)

A100 = GPU("A100", 80, 2.04, 312, has_fp8=False, has_fp4=False, has_sparse=True)
H100 = GPU("H100", 80, 3.35, 989, has_fp8=True,  has_fp4=False, has_sparse=True)
B300 = GPU("B300", 288, 8.0, 2250, has_fp8=True, has_fp4=True,  has_sparse=True)

def ridge(g: GPU) -> float:
    """BF16 ridge point in FLOP/byte: this chip's peak over its own bandwidth."""
    return g.bf16_tflops * 1e12 / (g.hbm_bw_tbs * 1e12)

def advise(n_params: float, gpu: GPU, regime: str,
           batch: int = 1, kv_gb: float = 0.0):
    """
    regime ∈ {"train", "decode", "prefill", "serve"}
    Prints the binding constraint and a precision/technique recommendation.
    """
    print(f"\n{'='*60}\n{n_params/1e9:.0f}B params on {gpu.name} | regime={regime} | batch={batch}")

    # 1) Capacity check at a few precisions (weights + KV)
    fits = {}
    for label, bpw in [("BF16", 2.0), ("INT8", 1.0), ("INT4", 0.5)]:
        need = n_params * bpw / 1e9 + kv_gb
        fits[label] = need <= gpu.hbm_gb
        print(f"   weights {label:>4}: {need:6.1f} GB / {gpu.hbm_gb:.0f} GB  -> {'fits' if fits[label] else 'OOM'}")
    smallest_fit = next((k for k in ("BF16", "INT8", "INT4") if fits[k]), None)
    four_bit = "NVFP4/MXFP4" if gpu.has_fp4 else "INT4 (AWQ/GPTQ)"

    # 2) Regime-specific advice
    if regime == "train":
        states_gb = n_params * 16 / 1e9
        print(f"   CONSTRAINT: compute + training state: {states_gb:.0f} GB at 16 B/param"
              f" -> at least {-(-states_gb // gpu.hbm_gb):.0f} GPUs before activations (Ch 28)")
        rec = ["BF16 compute + FP32 master weights (default)"]
        if gpu.has_fp8: rec.append("FP8 matmuls (~2x compute) with per-block scaling")
        if gpu.has_fp4: rec.append("NVFP4 training only as a validated experiment (frontier)")
        rec.append("gradient checkpointing if activations OOM, then FSDP/ZeRO shard (Ch 28)")
        _print_rec(rec)

    elif regime in ("decode", "serve") and batch <= 8:
        ceiling = lambda bpw: gpu.hbm_bw_tbs * 1e12 / (n_params * bpw + kv_gb * 1e9)
        print(f"   CONSTRAINT: memory BANDWIDTH (AI≈{batch} << ridge≈{ridge(gpu):.0f})")
        print(f"   decode ceiling: BF16 {ceiling(2.0):.0f}  INT8 {ceiling(1.0):.0f}  INT4 {ceiling(0.5):.0f} tok/s")
        rec = []
        if smallest_fit != "BF16":
            rec.append("MUST quantize to fit: " + (smallest_fit or "shard across GPUs (Ch 28)"))
        rec.append(f"weight-only {four_bit} for ~4x fewer bytes (memory-bound: dequant is free)")
        rec.append(("FP8" if gpu.has_fp8 else "INT8") + " KV-cache + GQA for long context")
        _print_rec(rec)

    elif regime == "serve":  # large batch
        r = ridge(gpu)
        bound = "COMPUTE" if batch > r else "still bandwidth"
        print(f"   CONSTRAINT: {bound} (AI≈{batch} vs ridge≈{r:.0f})")
        rec = ["continuous batching + PagedAttention to raise batch (Ch 29)"]
        if batch > r:
            fmt = "W4A4 NVFP4" if gpu.has_fp4 else ("FP8 W8A8" if gpu.has_fp8 else "INT8 W8A8")
            rec.append(f"{fmt} so the saturated matmuls are cheap")
            if not fits["INT8"] and not gpu.has_fp4:
                rec.append("8-bit weights + this KV do not fit: TP=2 or a bigger-HBM GPU")
            if gpu.has_sparse: rec.append("2:4 sparsity for up to ~2x more (needs retraining)")
        _print_rec(rec)

    elif regime == "prefill":
        print(f"   CONSTRAINT: COMPUTE (big GEMM, AI > ridge≈{ridge(gpu):.0f})")
        rec = ["FlashAttention", "FP8 matmul" if gpu.has_fp8 else "BF16 or INT8 matmul",
               "this governs time-to-first-token"]
        _print_rec(rec)

def _print_rec(items):
    print("   RECOMMEND:")
    for x in items:
        print(f"     - {x}")

# --- Run the worked scenarios ---
advise(70e9, H100, "decode", batch=1, kv_gb=4.0)    # (a) 70B low-latency
advise(70e9, H100, "serve",  batch=512, kv_gb=20)   # (b) max throughput, Hopper
advise(70e9, B300, "serve",  batch=512, kv_gb=60)   # (b') same job on Blackwell Ultra
advise(7e9,  A100, "decode", batch=1)               # (c)-ish: no FP8 -> INT4
advise(70e9, H100, "train",  batch=8)               # (d) training that won't fit
```

**Sample output:**

```text
============================================================
70B params on H100 | regime=decode | batch=1
   weights BF16:  144.0 GB / 80 GB  -> OOM
   weights INT8:   74.0 GB / 80 GB  -> fits
   weights INT4:   39.0 GB / 80 GB  -> fits
   CONSTRAINT: memory BANDWIDTH (AI≈1 << ridge≈295)
   decode ceiling: BF16 23  INT8 45  INT4 86 tok/s
   RECOMMEND:
     - MUST quantize to fit: INT8
     - weight-only INT4 (AWQ/GPTQ) for ~4x fewer bytes (memory-bound: dequant is free)
     - FP8 KV-cache + GQA for long context

============================================================
70B params on H100 | regime=serve | batch=512
   weights BF16:  160.0 GB / 80 GB  -> OOM
   weights INT8:   90.0 GB / 80 GB  -> OOM
   weights INT4:   55.0 GB / 80 GB  -> fits
   CONSTRAINT: COMPUTE (AI≈512 vs ridge≈295)
   RECOMMEND:
     - continuous batching + PagedAttention to raise batch (Ch 29)
     - FP8 W8A8 so the saturated matmuls are cheap
     - 8-bit weights + this KV do not fit: TP=2 or a bigger-HBM GPU
     - 2:4 sparsity for up to ~2x more (needs retraining)

============================================================
70B params on B300 | regime=serve | batch=512
   weights BF16:  200.0 GB / 288 GB  -> fits
   weights INT8:  130.0 GB / 288 GB  -> fits
   weights INT4:   95.0 GB / 288 GB  -> fits
   CONSTRAINT: COMPUTE (AI≈512 vs ridge≈281)
   RECOMMEND:
     - continuous batching + PagedAttention to raise batch (Ch 29)
     - W4A4 NVFP4 so the saturated matmuls are cheap
     - 2:4 sparsity for up to ~2x more (needs retraining)

============================================================
7B params on A100 | regime=decode | batch=1
   weights BF16:   14.0 GB / 80 GB  -> fits
   weights INT8:    7.0 GB / 80 GB  -> fits
   weights INT4:    3.5 GB / 80 GB  -> fits
   CONSTRAINT: memory BANDWIDTH (AI≈1 << ridge≈153)
   decode ceiling: BF16 146  INT8 291  INT4 583 tok/s
   RECOMMEND:
     - weight-only INT4 (AWQ/GPTQ) for ~4x fewer bytes (memory-bound: dequant is free)
     - INT8 KV-cache + GQA for long context

============================================================
70B params on H100 | regime=train | batch=8
   weights BF16:  140.0 GB / 80 GB  -> OOM
   weights INT8:   70.0 GB / 80 GB  -> fits
   weights INT4:   35.0 GB / 80 GB  -> fits
   CONSTRAINT: compute + training state: 1120 GB at 16 B/param -> at least 14 GPUs before activations (Ch 28)
   RECOMMEND:
     - BF16 compute + FP32 master weights (default)
     - FP8 matmuls (~2x compute) with per-block scaling
     - gradient checkpointing if activations OOM, then FSDP/ZeRO shard (Ch 28)
```

Two lines in that output are worth a second look. The H100 serving case recommends FP8 and then warns that 8-bit weights plus 20 GB of KV do not fit in 80 GB, so the honest answer is two GPUs or a bigger one; the B300 case fits everything and moves straight to NVFP4. And the B300 ridge it prints (≈281) is the BF16 ridge: switching to FP4 math raises the ridge to roughly 1,900 FLOP/byte, while 4-bit weights raise the batch's intensity by 4× too, so the job stays compute-bound but only just. A format change moves both the operating point and the roof.

The advisor is deliberately small, but notice it does exactly what an experienced practitioner does: check capacity at several precisions, locate the workload on the roofline, name the binding resource, and recommend the technique that buys *that* resource — never a technique that helps a resource you aren't short of.

---

## Why This Matters for Model Optimization

<div class="diagram">
<div class="diagram-title">The Monday-Morning Checklist</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">1. Name the scarce resource</div>
    <div class="card-desc">Capacity (won't fit), bandwidth (slow decode), or compute (slow training/prefill)? Run the Ch 29 budget and the Ch 11 roofline. Everything follows from this.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">2. Apply the free levers first</div>
    <div class="card-desc">BF16, fusion, GQA, checkpointing, paging, sharding cost ~zero accuracy. Exhaust them before spending any quality.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">3. Spend accuracy in proportion</div>
    <div class="card-desc">INT8 freely; INT4/FP8 with validation; 2:4 only with a retraining budget. Buy back exactly the resource you're short of.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">4. Match format to silicon</div>
    <div class="card-desc">Check the Ch 24 support matrix. FP8 needs Hopper+/MI300+, FP4 needs Blackwell/MI355X/TPU 8, 2:4 needs Sparse Tensor Cores. The wrong format is a slowdown.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">5. Memory-bound ≠ compute-bound</div>
    <div class="card-desc">Decode wants fewer bytes (weight/KV quant); training/prefill/serving-at-scale want cheaper math (FP8/INT8/2:4). Same word, different tool.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">6. Measure, don't assume</div>
    <div class="card-desc">The framework here predicts order-of-magnitude. Always confirm with a profiler and a quality eval — the accuracy price is task-specific.</div>
  </div>
</div>
</div>

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Every "free" lever in this chapter is free in accuracy but not in something else: checkpointing costs compute, sharding costs communication, paging costs kernel complexity. Pick the last optimization you shipped. What did it actually cost, and in which of the three resources did you pay?</p>
</div>

The whole guide collapses to one habit: **identify which of the three resources binds you, then spend the least accuracy to buy it back, using a format your hardware actually accelerates.** Quantization, sparsity, batching, parallelism, checkpointing — they are not a grab-bag of tricks but a set of well-defined purchases against a fixed hardware budget. When you can say "I'm bandwidth-bound at batch=1, so I'll go INT4 weight-only because the dequant is free and my GPU has the path," you are doing exactly what the silicon was begging you to do all along.

---

## Key Takeaways

- **Every optimization buys back one of three resources** — memory capacity, memory bandwidth, or compute throughput — and you pay in accuracy. Name the scarce resource and the technique follows.
- **Classify first with the roofline.** Memory-bound (decode) wants *fewer bytes* (weight/KV quant, fusion, batching); compute-bound (training, prefill, large-batch serving) wants *cheaper math* (FP8/INT8/2:4). The same technique helps in only one regime.
- **The free levers cost ~zero accuracy** — BF16, GQA, operator fusion, gradient checkpointing, PagedAttention, sharding. Exhaust them before spending any model quality.
- **The defaults encode the tradeoffs:** BF16 + FP32 master for training (range + stable accumulation); weight-only INT4 for memory-bound local decode (fewer bytes = faster + fits); KV-quant for long context; FP8 needs scaling; activations are harder than weights because of outliers.
- **Never pick a format your GPU can't accelerate.** FP8 = Hopper+ or MI300+, FP4/NVFP4 = Blackwell, MI355X or TPU 8, 2:4 = Sparse Tensor Cores ([Ch 24](./24_datatypes_and_silicon.md)). The wrong format is emulation — a slowdown, not a speedup.
- **Walk the worked scenarios:** 70B single-GPU low-latency → INT4 weight-only + INT8 KV + GQA; max-throughput serving → batch hard then W8A8/FP8; on-device → INT4 matched to the NPU's datapath; training that won't fit → checkpointing → FP8 → shard → offload.
- This is the chapter you **return to**: identify the binding constraint, apply free levers, then spend accuracy in proportion, matched to your hardware — and verify with a profiler and an eval.

## Check Yourself

<details class="check">
<summary>A 30B model must serve one user at a time on a 48 GB GPU with 1 TB/s. Which format, and what is the token-rate ceiling?</summary>
<p>BF16 (60 GB) does not fit; INT8 (30 GB) fits but leaves ~15 GB for KV; 4-bit (≈16 GB with scales) fits comfortably. Batch-1 decode is memory-bound, so 4-bit wins: ≈ 1 TB/s ÷ 16 GB ≈ 60 tokens/s ceiling, versus ≈ 33 for INT8.</p>
</details>

<details class="check">
<summary>Why is BF16, not FP16, the training default?</summary>
<p>BF16 keeps FP32's 8-bit exponent, so gradients and activations do not overflow or underflow and no loss scaling is needed. FP16 has more mantissa precision but a 5-bit exponent, which is the wrong trade for training.</p>
</details>

<details class="check">
<summary>You switch a compute-bound prefill from BF16 to FP8 on an A100 and see no speedup. Why?</summary>
<p>A100 has no FP8 tensor cores. FP8 is upcast to FP16 before the multiply, so you saved memory but not math, and the work was compute-bound.</p>
</details>

<details class="check">
<summary>Which levers in this chapter cost no accuracy, and what does each cost instead?</summary>
<p>BF16 for FP32 (nothing, in practice), fusion (engineering), GQA if trained in (architecture change), checkpointing (~33% compute), sharding (communication), paging (kernel complexity).</p>
</details>

<details class="check">
<summary>On a B300, why does moving from FP8 to FP4 math raise the ridge point, and why might the job still be compute-bound?</summary>
<p>FP4 peak is ~3.3× the FP8 peak while bandwidth is unchanged, so FLOP/byte at the ridge rises. But 4-bit weights also halve the bytes per FLOP of the workload, raising its intensity; whether it stays above the ridge depends on batch size.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Mixed Precision Training | Micikevicius et al., 2018 | [arxiv.org/abs/1710.03740](https://arxiv.org/abs/1710.03740) | FP16 + FP32 master weights and loss scaling |
| A Study of BFLOAT16 for Deep Learning Training | Kalamkar et al., 2019 | [arxiv.org/abs/1905.12322](https://arxiv.org/abs/1905.12322) | Why BF16 trains without loss scaling |
| FP8 Formats for Deep Learning | Micikevicius et al., 2022 | [arxiv.org/abs/2209.05433](https://arxiv.org/abs/2209.05433) | E4M3/E5M2 and FP8 training |
| GPTQ | Frantar et al., 2023 | [arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323) | 4-bit weight-only quantization |
| AWQ | Lin et al., 2024 | [arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978) | Activation-aware 4-bit weights |
| SmoothQuant | Xiao et al., 2023 | [arxiv.org/abs/2211.10438](https://arxiv.org/abs/2211.10438) | W8A8 with activation outliers |
| Introducing NVFP4 for Efficient and Accurate Low-Precision Inference | NVIDIA, 2025 | [developer.nvidia.com/blog/introducing-nvfp4-f…](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) | NVFP4 block scaling and accuracy |
| OCP Microscaling Formats (MX) v1.0 | Open Compute Project, 2023 | [opencompute.org/documents/ocp-microscaling-fo…](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) | MXFP8/MXFP6/MXFP4 definitions |
| Roofline: An Insightful Visual Performance Model | Williams, Waterman & Patterson, 2009 | [doi.org/10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785) | Classifying memory- vs compute-bound |

---

**Next:** [Chapter 31 — Chip Design Flow](./31_chip_design_flow.md) — We leave the practitioner's seat and follow how the silicon you have been optimizing for is actually designed, from spec to RTL to tape-out.

[← Back to Table of Contents](./README.md)
