---
title: "Chapter 29 — Memory Hierarchy in Action: KV-Cache, Batching & Activations"
---

[← Back to Table of Contents](./README.md)

# Chapter 29 — Memory Hierarchy in Action: KV-Cache, Batching & Activations

[Chapter 9](./09_memory_types.md) gave you the memory hierarchy — registers, SRAM, HBM, NVMe — as a stack of speed-versus-capacity tradeoffs. [Chapter 11](./11_memory_wall_and_bandwidth.md) gave you the roofline: the single number, arithmetic intensity, that tells you whether a kernel waits on the multipliers or on the wires. This chapter is where those two ideas stop being theory and start being **the spreadsheet you keep open while you size a deployment.** Every "will it fit?", every "how many requests can I batch?", every "why did my throughput collapse at 8k context?" is the memory hierarchy talking to you directly.

We will work through the three places the hierarchy bites hardest in practice: the **KV-cache** (the runtime memory that grows with context and batch, and routinely *exceeds* the weights), **batching** (the lever that drags memory-bound decode back toward compute-bound), and **activation memory** in training (the hidden tensor mountain that dwarfs your parameters at long sequence length). For each, we derive the byte math from first principles, plug in real model and hardware numbers, and write Python you can run to size your own case.

Throughout, our reference hardware is the **NVIDIA H100 SXM**: 80 GB HBM3 at **~3.35 TB/s**, ~989 TFLOP/s dense BF16, ridge point ≈ 295 FLOP/byte (from [Chapter 11](./11_memory_wall_and_bandwidth.md)). Substitute your own GPU's two numbers — capacity (GB) and bandwidth (TB/s) — and every formula here still holds.

> **The one-sentence version:** LLM serving is an exercise in spending a fixed HBM budget — weights, KV-cache, and activations all compete for the same gigabytes and the same terabytes-per-second, and almost every serving decision you make is really a decision about which of those three to shrink.

---

## Recap: Why Decode Is Bandwidth-Bound (and the Token-Rate Ceiling)

Autoregressive **decode** generates one token per forward pass. To produce that one token, the GPU must read **every weight** of the model from HBM exactly once (at batch size 1). The arithmetic per token is tiny — a stack of matrix-*vector* products (GEMVs), arithmetic intensity ≈ 1 FLOP/byte — so the multipliers sit mostly idle and the **HBM bus is the bottleneck**. This is the memory wall in its purest form.

That gives a hard ceiling on token rate that depends only on two numbers:

$$\text{tokens/s} \;\lesssim\; \frac{\text{HBM bandwidth (bytes/s)}}{\text{bytes read per token}}$$

The denominator is "bytes read per token," which at batch=1 is essentially **the model size in bytes** (weights), plus the KV-cache bytes you must stream for attention. The weights dominate, so to first order the ceiling is `HBM_BW / model_bytes`.

<div class="diagram">
<div class="diagram-title">The Decode Token-Rate Ceiling (batch = 1)</div>
<div class="flow-h">
  <div class="flow-node accent">Model weights<br/><small>must stream all per token</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange">÷ HBM bandwidth<br/><small>3.35 TB/s on H100</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">= time/token<br/><small>→ tokens/s ceiling</small></div>
</div>
</div>

### Worked Example: A 70B Model at INT8

A 70-billion-parameter model quantized to **INT8** (1 byte/weight) occupies ~70 GB of weights. Per generated token, all 70 GB must traverse the HBM bus:

$$t_{\text{token}} \approx \frac{70 \times 10^{9}\ \text{bytes}}{3.35 \times 10^{12}\ \text{bytes/s}} \approx 20.9\ \text{ms} \quad\Rightarrow\quad \approx 48\ \text{tokens/s}$$

That **~48 tok/s** is a ceiling no amount of tensor-core horsepower can lift, because the tensor cores are not the constraint. The table below shows how the ceiling moves with precision and model size — read it as "the only two knobs that change decode speed at batch=1 are *fewer bytes per weight* and *fewer weights*."

| Model | Precision | Bytes/weight | Weight bytes | HBM read/token | tok/s ceiling (3.35 TB/s) |
|-------|:---------:|:------------:|:------------:|:--------------:|:-------------------------:|
| 7B | BF16 | 2 | 14 GB | 14 GB | ~239 |
| 7B | INT8 | 1 | 7 GB | 7 GB | ~479 |
| 7B | INT4 | 0.5 | 3.5 GB | 3.5 GB | ~957 |
| 70B | BF16 | 2 | 140 GB | *(won't fit on 80 GB)* | — |
| 70B | INT8 | 1 | 70 GB | 70 GB | ~48 |
| 70B | INT4 | 0.5 | 35 GB | 35 GB | ~96 |

> Note the immediate practical consequence: a 70B model in BF16 (**140 GB**) does not even *fit* in one 80 GB GPU, let alone run fast. Quantization here buys two things at once — it makes the model **fit** (capacity) and **stream faster** (bandwidth). This is why local/single-GPU LLM serving lives at INT4/INT8, a thread we pick up quantitatively in [Chapter 25](./25_quantization_hardware_view.md) and turn into a decision rule in [Chapter 30](./30_datatypes_drive_optimization_choices.md).

The ceiling is *optimistic* — it ignores the KV-cache reads, attention, kernel-launch overhead, and the fact that you never hit 100% of peak bandwidth (70–85% is typical). Real throughput is below it. But it tells you the *order of magnitude* before you write a line of serving code.

---

## The KV-Cache: The Memory That Grows With Your Conversation

### What It Is and Why It Exists

In attention, each new token must attend to **all previous tokens**. Naively, generating token $t$ would require recomputing the keys ($K$) and values ($V$) for tokens $1 \ldots t-1$ on every step — an $O(t^2)$ blow-up that would make long generations quadratically slow. The **KV-cache** is the fix: after a token's $K$ and $V$ vectors are computed once, they are **stored in HBM** and reused for every subsequent step. Decode then only computes $K, V$ for the *single* new token and reads the cached rest.

This trades **compute for memory** — the canonical hardware bargain. The cost is that the cache is *runtime* memory: it did not exist when you loaded the model, it grows with every token, and it grows with every concurrent request.

<div class="diagram">
<div class="diagram-title">KV-Cache: Store Once, Reuse Every Step</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Without KV-cache</div>
    <ul>
      <li>Token <em>t</em> recomputes K,V for all <em>1…t</em></li>
      <li>Per-token cost grows as <strong>O(t)</strong>, total <strong>O(T²)</strong></li>
      <li>No extra memory, but compute explodes</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">With KV-cache</div>
    <ul>
      <li>Compute K,V for the <strong>new token only</strong></li>
      <li>Read cached K,V for the past — per-token compute ~constant</li>
      <li>Extra HBM that grows linearly with sequence & batch</li>
    </ul>
  </div>
</div>
</div>

### The KV-Cache Size Formula

For a single sequence, the cache stores a $K$ and a $V$ tensor at every layer, for every cached position. The size in bytes is:

$$\text{KV bytes} = 2 \times L \times n_{kv} \times d_{head} \times \text{seq} \times \text{batch} \times \text{bytes}$$

where:

| Symbol | Meaning |
|--------|---------|
| $2$ | one $K$ tensor **and** one $V$ tensor |
| $L$ | number of transformer layers |
| $n_{kv}$ | number of **key/value** heads (= query heads for MHA; fewer for GQA/MQA) |
| $d_{head}$ | dimension per head (often $d_{model}/n_{heads}$) |
| $\text{seq}$ | number of cached positions (prompt + generated so far) |
| $\text{batch}$ | concurrent sequences |
| $\text{bytes}$ | bytes per element (2 for FP16/BF16, 1 for INT8/FP8) |

Note $n_{kv} \times d_{head}$ is the total KV width per token per layer. For a model with **multi-head attention (MHA)**, $n_{kv} = n_{heads}$ and this equals $d_{model}$. For **grouped-query attention (GQA)** it is much smaller — which is the whole point (below).

### Worked Example: Llama-3-style 8B at FP16

Take an 8B model with $L = 32$ layers, $d_{model} = 4096$, $n_{heads} = 32$, $d_{head} = 128$, and **GQA with $n_{kv} = 8$** key/value heads. Per token per layer the cache stores $2 \times n_{kv} \times d_{head} = 2 \times 8 \times 128 = 2048$ elements. Across 32 layers that is $32 \times 2048 = 65{,}536$ elements/token, or at FP16 (2 bytes) **128 KB per token**.

$$\text{KV bytes/token} = 2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\ \text{bytes} = 128\ \text{KB}$$

Now scale by context and batch:

| Context (seq) | Batch | KV-cache size | vs 16.1 GB weights (FP16) |
|:-------------:|:-----:|:-------------:|:-------------------------:|
| 2,048 | 1 | 0.27 GB | 1.7% |
| 8,192 | 1 | 1.07 GB | 6.7% |
| 32,768 | 1 | 4.29 GB | 27% |
| 128,000 | 1 | 16.78 GB | ~104% (≈ the weights!) |
| 8,192 | 32 | 34.4 GB | ~2× the weights |
| 8,192 | 64 | 68.7 GB | ~4× the weights |

> The headline: at **128k context, the KV-cache for one sequence nearly equals the entire model's weights.** At batch 64 and 8k context, the KV-cache is **four times the size of the weights.** This is why long context and high concurrency are "memory-hungry" — it is not vague; it is this formula. The KV-cache, not the weights, is usually what determines how many requests you can serve at once.

### The Same Model With *Multi-Head* Attention (No GQA)

Had this model used full MHA ($n_{kv} = 32$ instead of 8), every number above multiplies by **4×**. The 8,192-token, batch-32 case would need **~137 GB** of KV-cache — impossible on an 80 GB GPU. This single architectural choice (GQA) is the difference between "serves 64 concurrent users" and "OOMs at 16."

### Reducing the KV-Cache: GQA/MQA, Quantization, Paging, Prefix Reuse

Because the cache so often dominates, there is a whole toolbox aimed squarely at the four factors in the formula:

<div class="diagram">
<div class="diagram-title">Four Levers on the KV-Cache Formula</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Shrink n_kv → GQA / MQA</div>
    <div class="card-desc">Multi-Query (n_kv=1) and Grouped-Query (n_kv=4–8) share K/V across query heads. A 32→8 head reduction is a <strong>4× KV cut</strong> with negligible quality loss — now standard in Llama-3, Mistral, etc.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Shrink bytes → KV quantization</div>
    <div class="card-desc">Store K/V at INT8 or FP8 instead of FP16 → <strong>2× cut</strong>; some systems push to INT4 for <strong>4×</strong>. Targets long context specifically. See <a href="./25_quantization_hardware_view.html">Ch 25</a>.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Stop wasting it → PagedAttention</div>
    <div class="card-desc">Allocate the cache in fixed <strong>blocks (pages)</strong> like OS virtual memory (<a href="./10_memory_management.html">Ch 10</a>) instead of one contiguous slab per request. Kills fragmentation; lifts usable batch 2–4×.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Don't recompute → prefix caching</div>
    <div class="card-desc">Shared prompt prefixes (system prompts, few-shot examples) are cached once and reused across requests — saves prefill compute <em>and</em> KV memory for the shared span.</div>
  </div>
</div>
</div>

**GQA/MQA** (grouped-query and multi-query attention) directly attack $n_{kv}$. With $n_{heads}$ query heads but only $n_{kv}$ key/value heads, several query heads *share* one K/V head. MQA ($n_{kv}=1$) is the extreme; GQA ($n_{kv}=4$ or $8$) is the sweet spot that keeps quality. The KV-cache shrinks by exactly $n_{heads}/n_{kv}$.

DeepSeek's **Multi-head Latent Attention (MLA)**, used in DeepSeek-V2 and V3, goes further: instead of caching per-head K and V it caches one compressed 512-dimensional latent plus a 64-dimensional positional key per token per layer, and re-expands them inside the attention kernel. For DeepSeek-V3 that is 61 layers × 576 values × 2 bytes ≈ 69 KB per token, against about 320 KB for Llama 3 70B with GQA. The price is extra matmuls at decode time, a trade the roofline says is cheap because decode has compute to spare.

**KV-cache quantization** attacks the `bytes` factor. Keys and values are activations, so they are *harder* to quantize than weights (outliers — see [Chapter 25](./25_quantization_hardware_view.md)), but per-token or per-channel scaling makes INT8/FP8 KV practical, halving the cache. It is the right lever precisely when context is long (the cache, not the weights, is your capacity problem). FP8 KV is now a default option in vLLM, SGLang and TensorRT-LLM, and Blackwell-era stacks are moving to 4-bit KV with block scales.

**PagedAttention** (the idea behind vLLM) attacks *waste*. A naive serving system reserves a contiguous buffer sized for each request's *maximum* length, so a request that generates 100 tokens but was sized for 8,192 wastes 98% of its reservation — **internal fragmentation**. Paging the cache into small fixed blocks (e.g., 16 tokens) and mapping them through a block table — exactly the virtual-memory trick from [Chapter 10](./10_memory_management.md) — lets memory be allocated on demand, raising the number of concurrent sequences 2–4×.

**Prefix caching** attacks *redundancy*: many requests share a long system prompt or few-shot preamble. Compute its KV once, reuse it for every request that shares the prefix. Saves both the prefill FLOPs and the duplicate KV bytes.

---

## Batching: Trading Latency for Throughput by Raising Arithmetic Intensity

### Why Batching Works

We saw in [Chapter 11](./11_memory_wall_and_bandwidth.md) that decode arithmetic intensity at batch $B$ is approximately:

$$\text{AI}_{\text{decode}}(B) \approx B \quad \text{(FLOP/byte)}$$

because $B$ sequences all multiply against the **same** weight matrix that was read from HBM only once. You read the weights once and do $B$ times the arithmetic. Below the ridge point (~295 on H100), throughput scales nearly linearly with $B$; above it, you become compute-bound and further batching stops helping per-token speed.

<div class="diagram">
<div class="diagram-title">Batching Walks Decode Up the Roofline</div>
<div class="flow-h">
  <div class="flow-node accent">B=1<br/><small>AI≈1, memory-bound<br/>weights wasted</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange">B=32<br/><small>AI≈32, still memory-bound<br/>32× the work/byte</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">B≈295<br/><small>at the ridge<br/>balanced</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">B=1024<br/><small>compute-bound<br/>tensor cores saturated</small></div>
</div>
</div>

### The Throughput vs Latency Tradeoff

Batching is **free throughput but not free latency.** Two metrics pull in opposite directions:

| Metric | What it is | Effect of larger batch |
|--------|-----------|------------------------|
| **Per-token latency** (TPOT) | time to emit one token for a given request | rises modestly until ridge, then linearly |
| **Throughput** (total tok/s across all requests) | aggregate generation rate | rises ~linearly until ridge, then flat |
| **Time-to-first-token** (TTFT) | latency of the prefill | rises with queue depth / batch |

A single user wants low TPOT and TTFT → small batch. A serving fleet wants maximum aggregate tok/s per GPU-dollar → large batch. **The same hardware serves both, just at different points on the roofline.** This is the central tension every inference team negotiates, and [Chapter 30](./30_datatypes_drive_optimization_choices.md) turns it into explicit scenarios.

> A useful way to hold it: **batching converts idle HBM bandwidth into throughput.** At batch=1 you pay for 3.35 TB/s and use it to stream weights you immediately discard; batching makes each of those expensive byte-reads do $B$ tokens' worth of work.

### Continuous (In-Flight) Batching

Classic "static" batching waits to assemble $B$ requests, runs them lockstep until *all* finish, then starts the next batch — so one long generation stalls a whole batch of short ones, wrecking utilization. **Continuous batching** (a.k.a. in-flight or iteration-level batching) instead schedules at the **granularity of a single decode step**: finished sequences leave the batch and new ones join *every iteration*. Combined with PagedAttention, this keeps the GPU near its target batch size at all times and is why modern servers (vLLM, TensorRT-LLM, TGI) get multiples of the throughput of naive batching.

The constraint on how large the running batch can grow is **HBM capacity**, and the thing that fills it is — once more — the KV-cache. Continuous batching and PagedAttention exist to pack as many concurrent KV-caches as possible into the gigabytes left over after the weights.

---

## Activation Memory in Training: The Hidden Mountain

Inference is dominated by weights + KV-cache. **Training** adds a third, often largest, consumer: **activations** — the intermediate tensors saved during the forward pass so the backward pass can compute gradients. (We met the parameter/optimizer/gradient side of the training memory budget in [Chapter 28](./28_multi_gpu_and_distributed_training.md); this is the fourth pillar.)

### Why Activations Dominate at Long Sequence Length

For a transformer, the activations saved per layer scale with **batch × sequence × hidden size**, and the attention activations scale with **batch × heads × sequence²** before fusion. Crucially, activation memory grows with **sequence length** while parameter memory does **not**. So as context grows, activations come to dominate.

A rough per-layer activation estimate (forward, pre-checkpointing) is on the order of:

$$\text{act bytes/layer} \approx c \times B \times \text{seq} \times d_{model} \times \text{bytes}$$

with $c \approx 10\text{–}30$ depending on how many intermediate tensors the implementation stores (QKV, attention output, MLP intermediate, norms, etc.).

**Worked example.** A model with $d_{model}=4096$, $L=32$, training in BF16 at $B=8$, $\text{seq}=4096$, with $c \approx 16$:

$$\text{act/layer} \approx 16 \times 8 \times 4096 \times 4096 \times 2 \approx 4.3\ \text{GB} \quad\Rightarrow\quad \times 32\ \text{layers} \approx 137\ \text{GB}$$

That **137 GB of activations** dwarfs the ~16 GB of weights and would never fit. Double the sequence to 8,192 and it doubles to ~274 GB. This is why **long-context training is an activation-memory problem first and a compute problem second.**

### Activation / Gradient Checkpointing: Recompute Instead of Store

The fix is the same bargain as the KV-cache, run in reverse: **trade compute for memory** by *not* storing most activations. With **activation (gradient) checkpointing**, you save only a few activations per layer (the "checkpoints") and **recompute** the rest during the backward pass by re-running the forward of that segment.

<div class="diagram">
<div class="diagram-title">Checkpointing: Memory ↔ Compute Trade</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Store everything (default)</div>
    <ul>
      <li>Activation memory ~ <strong>O(L)</strong> — all layers kept</li>
      <li>Backward needs no recompute</li>
      <li>OOMs at long sequence / large batch</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Checkpoint (recompute)</div>
    <ul>
      <li>Activation memory ~ <strong>O(√L)</strong> or O(1) per segment</li>
      <li>~<strong>+33% compute</strong> (one extra forward) for big memory savings</li>
      <li>Recompute is cheap because the forward is often memory-bound anyway</li>
    </ul>
  </div>
</div>
</div>

The classic result (Chen et al., 2016): keeping a checkpoint every $\sqrt{L}$ layers reduces activation memory from $O(L)$ to roughly $O(\sqrt{L})$ at the cost of **one extra forward pass (~33% more compute)**. Transformer frameworks usually checkpoint at every layer boundary instead, keeping one tensor per layer rather than the 16–34 the forward pass would otherwise save; and *selective* recomputation (Korthikanti et al., 2022) recomputes only the cheap-to-redo attention internals, cutting activation memory about 5× for a few percent of extra compute. In PyTorch it is a one-line wrap with `torch.utils.checkpoint.checkpoint`. The recompute is comparatively cheap because — as [Chapter 11](./11_memory_wall_and_bandwidth.md) showed — much of the forward pass is memory-bound, so the "extra" FLOPs are partly hidden behind bandwidth you were paying for anyway.

### Offloading: Spilling to CPU and NVMe

When even checkpointing is not enough, you can **offload** tensors down the hierarchy from [Chapter 9](./09_memory_types.md): push optimizer states, gradients, or activations to **CPU DRAM** (~50–100 GB/s over PCIe) or even **NVMe** (~3–14 GB/s). This is what ZeRO-Offload / ZeRO-Infinity do ([Chapter 28](./28_multi_gpu_and_distributed_training.md)). It buys capacity at a steep bandwidth cost: moving a tensor to CPU and back over PCIe 5.0 (~64 GB/s) is ~50× slower than HBM, so offloading is a "fit it at all" tool, not a "make it fast" tool. Superchips change the ratio: Grace Hopper and Grace Blackwell connect the GPU to 480 GB of CPU LPDDR5X over NVLink-C2C at 900 GB/s, which makes optimizer and KV-cache offload far less painful than over PCIe. The art is overlapping the transfers with compute so the PCIe latency hides behind the GPU's work.

---

## Prefill vs Decode: Two Workloads, Two Optimizations

A single LLM request has **two phases with opposite roofline character**, and conflating them is a common and expensive mistake.

<!-- flowchart:request_life -->
<div class="flowchart" id="fc-request_life">
<div class="fc-title">The life of one LLM request, seen from memory</div>
<svg viewBox="0 0 800 330" width="800" role="img" aria-label="The life of one LLM request, seen from memory">
<path class="fc-edge" d="M180,50 L208,50"/><polygon class="fc-head" points="208,50 201,53.9 201,46.1"/>
<path class="fc-edge" d="M428,50 L520,50"/><polygon class="fc-head" points="520,50 513,53.9 513,46.1"/><text class="fc-elabel " x="474" y="41" text-anchor="middle">writes K,V</text>
<path class="fc-edge" d="M640,78 L640,140"/><polygon class="fc-head" points="640,140 636.1,133 643.9,133"/>
<path class="fc-edge" d="M640,196 L640,257"/><polygon class="fc-head" points="640,257 636.1,250 643.9,250"/>
<path class="fc-edge loop" d="M745,278 L790,278 L790,168 L770,168"/><polygon class="fc-head loop" points="770,168 777,164.2 777,171.8"/><text class="fc-elabel loop" x="784" y="226" text-anchor="end">no: next token</text>
<path class="fc-edge yes" d="M535,278 L465,278"/><polygon class="fc-head yes" points="465,278 472,274.1 472,281.9"/><text class="fc-elabel yes" x="525" y="270" text-anchor="end">yes</text>
<path class="fc-edge" d="M255,278 L200,278"/><polygon class="fc-head" points="200,278 207,274.1 207,281.9"/>
<path class="fc-edge loop" d="M430,168 L510,168"/><polygon class="fc-head loop" points="510,168 503,171.8 503,164.2"/>
<g class="fc-node start"><rect x="20" y="28" width="160" height="44" rx="16"/><text x="100" y="48" text-anchor="middle">Prompt arrives</text><text class="sub" x="100" y="61" text-anchor="middle">N tokens</text></g>
<g class="fc-node k-green"><rect x="208" y="22" width="220" height="56" rx="6"/><text x="318" y="42" text-anchor="middle">Prefill</text><text class="sub" x="318" y="55" text-anchor="middle">all N tokens, one batched GEMM pass</text><text class="sub" x="318" y="67" text-anchor="middle">compute-bound</text></g>
<a href="./09_memory_types.html"><g class="fc-node leaf k-purple"><rect x="520" y="22" width="240" height="56" rx="6"/><text x="640" y="42" text-anchor="middle">KV cache in HBM</text><text class="sub" x="640" y="55" text-anchor="middle">2 × layers × KV heads × head dim</text><text class="sub" x="640" y="67" text-anchor="middle">× bytes, per token</text></g></a>
<g class="fc-node k-orange"><rect x="510" y="140" width="260" height="56" rx="6"/><text x="640" y="160" text-anchor="middle">Decode step</text><text class="sub" x="640" y="173" text-anchor="middle">read every weight + this sequence&#x27;s KV</text><text class="sub" x="640" y="185" text-anchor="middle">memory-bound at small batch</text></g>
<g class="fc-node decision"><polygon points="535,278 549,257 731,257 745,278 731,299 549,299"/><text x="640" y="282" text-anchor="middle">End of sequence?</text></g>
<a href="./10_memory_management.html"><g class="fc-node k-cyan"><rect x="255" y="256" width="210" height="44" rx="6"/><text x="360" y="276" text-anchor="middle">Free the KV pages</text><text class="sub" x="360" y="289" text-anchor="middle">PagedAttention block table</text></g></a>
<g class="fc-node start"><rect x="20" y="262" width="180" height="32" rx="16"/><text x="110" y="282" text-anchor="middle">Response complete</text></g>
<a href="./11_memory_wall_and_bandwidth.html"><g class="fc-node k-teal"><rect x="170" y="140" width="260" height="56" rx="6"/><text x="300" y="160" text-anchor="middle">Continuous batching</text><text class="sub" x="300" y="173" text-anchor="middle">many sequences share each weight read,</text><text class="sub" x="300" y="185" text-anchor="middle">raising FLOPs per byte</text></g></a>
</svg>
<div class="fc-caption">Prefill is one large, compute-bound pass; decode is a long loop of small, memory-bound passes. Almost every serving optimisation targets the loop.</div>
</div>
<!-- /flowchart:request_life -->

<div class="diagram">
<div class="diagram-title">One Request, Two Regimes</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card green">
    <div class="card-title">Prefill (process the prompt)</div>
    <div class="card-desc">All prompt tokens at once → big <strong>GEMM</strong> (matrix×matrix), AI high → <strong>compute-bound</strong>. Builds the initial KV-cache. Determines time-to-first-token. Optimize FLOPs: FlashAttention, FP8/INT8 matmul, tensor-core utilization.</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-title">Decode (generate tokens)</div>
    <div class="card-desc">One token at a time → <strong>GEMV</strong> (matrix×vector), AI≈1 → <strong>memory-bound</strong>. Streams all weights + KV per token. Determines tokens/s. Optimize bytes: quantize weights & KV, batch hard, GQA.</div>
  </div>
</div>
</div>

| Aspect | Prefill | Decode |
|--------|---------|--------|
| Operation shape | GEMM ($\text{seq} \times d$) | GEMV ($1 \times d$) per step |
| Arithmetic intensity | high (compute-bound) | ≈1 (memory-bound) |
| Bottleneck | tensor-core FLOP/s | HBM bandwidth |
| Latency metric | time-to-first-token (TTFT) | time-per-output-token (TPOT) |
| Best levers | FlashAttention, low-precision matmul | weight + KV quantization, batching, GQA |

**Chunked prefill** is the modern reconciliation: a long prompt's prefill is broken into chunks and **interleaved with ongoing decode steps** of other requests in the same iteration. This stops a big prefill from blocking everyone's decode (a TTFT-vs-TPOT fairness fix) and keeps both the tensor cores (prefill) and the bandwidth (decode) busy simultaneously — using the *one* request type to fill the roof the other leaves empty.

**Disaggregated serving** takes the opposite approach: run prefill and decode on *different* GPUs (DistServe, Splitwise, NVIDIA Dynamo), shipping the finished KV cache from the prefill pool to the decode pool over NVLink or RDMA. Each pool is then sized and batched for its own roofline regime. By 2026 the split has reached the hardware roadmap: NVIDIA pairs Vera Rubin racks with Groq-derived LPX racks, whose on-chip SRAM is aimed squarely at the bandwidth-bound decode loop.

---

## Python: Sizing the Memory Hierarchy

The three calculators below are the ones you actually reach for: KV-cache sizing across (model, seq, batch, dtype), the decode token-rate ceiling, and the batch-vs-arithmetic-intensity curve. They have no dependencies beyond the standard library.

```python
from dataclasses import dataclass

# ---- Hardware (NVIDIA H100 SXM; swap for your GPU) ----
HBM_CAPACITY_GB = 80
HBM_BW_BYTES    = 3.35e12      # 3.35 TB/s
PEAK_FLOPS_BF16 = 989e12       # dense BF16 tensor-core FLOP/s
RIDGE = PEAK_FLOPS_BF16 / HBM_BW_BYTES   # ≈ 295 FLOP/byte

@dataclass
class Model:
    name: str
    n_params: float       # total parameters
    n_layers: int         # L
    d_model: int
    n_heads: int          # query heads
    n_kv_heads: int       # key/value heads (= n_heads for MHA, fewer for GQA/MQA)
    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

# A few reference models
LLAMA3_8B  = Model("Llama-3-8B",  8.03e9, 32, 4096, 32, 8)    # GQA 8 kv-heads
LLAMA3_70B = Model("Llama-3-70B", 70.6e9, 80, 8192, 64, 8)    # GQA 8 kv-heads
MISTRAL_7B = Model("Mistral-7B",  7.24e9, 32, 4096, 32, 8)


def kv_cache_bytes(m: Model, seq: int, batch: int, bytes_per_elem: int = 2) -> int:
    """KV-cache size = 2 * L * n_kv * d_head * seq * batch * bytes."""
    return 2 * m.n_layers * m.n_kv_heads * m.d_head * seq * batch * bytes_per_elem


def weight_bytes(m: Model, bytes_per_weight: float = 2.0) -> float:
    """Model weight footprint in bytes (2=BF16, 1=INT8, 0.5=INT4)."""
    return m.n_params * bytes_per_weight


def decode_tokens_per_sec_ceiling(m: Model, bytes_per_weight: float = 2.0,
                                  seq: int = 0, batch: int = 1,
                                  kv_bytes_per_elem: int = 2) -> float:
    """
    Upper bound on decode tok/s at batch=1: HBM_BW / bytes_read_per_token.
    bytes_read = all weights + (optionally) the KV-cache streamed for attention.
    """
    wb = weight_bytes(m, bytes_per_weight)
    kv = kv_cache_bytes(m, seq, batch, kv_bytes_per_elem) if seq else 0
    bytes_per_token = wb + kv
    return HBM_BW_BYTES / bytes_per_token * batch  # ×batch: each step emits `batch` tokens


def decode_arith_intensity(batch: int) -> float:
    """AI of decode ≈ batch (weights read once, reused across the batch)."""
    return float(batch)


def fits(m: Model, seq: int, batch: int,
         bytes_per_weight: float = 2.0, kv_bytes_per_elem: int = 2,
         capacity_gb: float = HBM_CAPACITY_GB) -> bool:
    total = weight_bytes(m, bytes_per_weight) + kv_cache_bytes(m, seq, batch, kv_bytes_per_elem)
    return total <= capacity_gb * 1e9


# ---------- 1) KV-cache across context & batch ----------
print(f"{'='*64}\nKV-cache for {LLAMA3_8B.name} (FP16 KV), weights={weight_bytes(LLAMA3_8B)/1e9:.1f} GB")
print(f"{'context':>9} {'batch':>6} {'KV (GB)':>10} {'% of weights':>14}")
for seq in (2048, 8192, 32768, 128000):
    for batch in (1, 32, 64):
        kv = kv_cache_bytes(LLAMA3_8B, seq, batch) / 1e9
        pct = 100 * kv / (weight_bytes(LLAMA3_8B) / 1e9)
        print(f"{seq:>9} {batch:>6} {kv:>10.2f} {pct:>13.0f}%")

# ---------- 2) Decode token-rate ceiling ----------
print(f"\n{'='*64}\nDecode tok/s ceiling (weights only, batch=1)")
print(f"{'model':>14} {'BF16':>8} {'INT8':>8} {'INT4':>8}")
for m in (MISTRAL_7B, LLAMA3_8B, LLAMA3_70B):
    bf16 = decode_tokens_per_sec_ceiling(m, 2.0)
    int8 = decode_tokens_per_sec_ceiling(m, 1.0)
    int4 = decode_tokens_per_sec_ceiling(m, 0.5)
    print(f"{m.name:>14} {bf16:>8.0f} {int8:>8.0f} {int4:>8.0f}")

# ---------- 3) Batch vs arithmetic intensity ----------
print(f"\n{'='*64}\nBatch vs arithmetic intensity (ridge ≈ {RIDGE:.0f} FLOP/byte)")
print(f"{'batch':>6} {'AI':>8} {'regime':>14}")
for b in (1, 8, 32, 128, 256, 295, 512, 1024):
    ai = decode_arith_intensity(b)
    regime = "memory-bound" if ai < RIDGE * 0.9 else ("compute-bound" if ai > RIDGE * 1.1 else "balanced")
    print(f"{b:>6} {ai:>8.0f} {regime:>14}")
```

**Sample output (abridged):**

```text
================================================================
KV-cache for Llama-3-8B (FP16 KV), weights=16.1 GB
  context  batch    KV (GB)   % of weights
     2048      1       0.27             2%
     8192      1       1.07             7%
     8192     32      34.36           214%
     8192     64      68.72           428%
   128000      1      16.78           104%
================================================================
Decode tok/s ceiling (weights only, batch=1)
         model     BF16     INT8     INT4
    Mistral-7B      231      463      925
    Llama-3-8B      209      417      834
   Llama-3-70B       24       47       95
================================================================
Batch vs arithmetic intensity (ridge ≈ 295 FLOP/byte)
 batch       AI         regime
     1        1   memory-bound
    32       32   memory-bound
   256      256   memory-bound
   295      295       balanced
   512      512  compute-bound
```

Read the three tables together and the whole chapter is in front of you: KV-cache can exceed the weights (table 1), so it caps your batch; decode speed is set by weight bytes alone at batch=1 (table 2), so precision is your speed knob; and you must batch into the hundreds to escape the memory-bound regime (table 3), which is exactly what the KV-cache capacity in table 1 is fighting you on.

---

## Putting It Together: The HBM Budget Equation

Everything in this chapter reduces to one accounting identity. The GPU's HBM must simultaneously hold:

$$\text{HBM used} = \underbrace{\text{weights}}_{\text{fixed}} + \underbrace{\text{KV-cache}}_{\propto\, \text{seq} \times \text{batch}} + \underbrace{\text{activations / workspace}}_{\text{transient}} \;\le\; \text{HBM capacity}$$

<div class="budget">
<div class="bg-legend"><span style="--c:var(--accent)">weights</span><span style="--c:#60a5fa">KV-cache room</span><span style="--c:#4b5563">workspace</span><span>bar length = HBM capacity</span></div>
<div class="bg-row"><div class="bg-name">H100 · 80 GB<small>Llama 3 70B, INT8 weights</small></div>
<div class="bg-track" style="--cap:27.8"><i class="w" style="--v:87.5">70 GB</i><i class="kv" style="--v:7.5"></i><i class="ws" style="--v:5"></i></div>
<div class="bg-note">6 GB of KV ≈ 18k tokens total (BF16 KV): two 8k chats</div></div>
<div class="bg-row"><div class="bg-name">H200 · 141 GB<small>same model, INT8 weights</small></div>
<div class="bg-track" style="--cap:49"><i class="w" style="--v:49.6">70 GB</i><i class="kv" style="--v:46.1">65 GB</i><i class="ws" style="--v:4.3"></i></div>
<div class="bg-note">≈ 198k tokens of BF16 KV, ≈ 397k in FP8</div></div>
<div class="bg-row"><div class="bg-name">B300 · 288 GB<small>same model, BF16 weights</small></div>
<div class="bg-track" style="--cap:100"><i class="w" style="--v:48.6">140 GB</i><i class="kv" style="--v:48.6">140 GB</i><i class="ws" style="--v:2.8"></i></div>
<div class="bg-note">≈ 427k tokens of BF16 KV with no quantization at all</div></div>
<div class="bg-row"><div class="bg-name">B300 · 288 GB<small>same model, NVFP4 weights</small></div>
<div class="bg-track" style="--cap:100"><i class="w" style="--v:13.7">39 GB</i><i class="kv" style="--v:83.5">241 GB</i><i class="ws" style="--v:2.8"></i></div>
<div class="bg-note">≈ 734k tokens BF16, ≈ 1.47M tokens FP8: batch is now bandwidth-limited, not capacity-limited</div></div>
</div>

Llama 3 70B stores 2 × 80 layers × 8 KV heads × 128 × 2 bytes = 320 KB of KV per token, so every gigabyte left after the weights buys about 3,000 tokens of context, shared across every request in the batch. The bars show why quantizing weights on a small-memory GPU is really about *making room for the KV cache*, and why the jump to 141–288 GB parts changed serving more than any FLOP increase did.

### Try It: Will It Fit?

The lab below runs the HBM budget equation for real models and GPUs. Set Llama 3 70B to BF16 on an H100 and it overflows; switch the weights to INT4 and it fits with room to spare; now raise the context to 128k and the batch to 8 and watch the KV cache take over. Switch the job to full fine-tuning to see the 16 bytes per parameter of [Chapter 28](./28_multi_gpu_and_distributed_training.md) arrive.

<div class="lab" data-lab="fit"><noscript><div class="lab-noscript">This lab needs JavaScript. It is also on the <a href="./labs.html">Interactive Labs</a> page.</div></noscript></div>

- Quantizing **weights** (INT8/INT4) frees capacity *and* raises the decode ceiling (bandwidth). Pick this for "won't fit" and "too slow at batch=1."
- Quantizing the **KV-cache** (INT8/FP8) and using **GQA** free the capacity that limits your batch. Pick these for "OOMs at high concurrency / long context."
- **PagedAttention + continuous batching** reclaim *wasted* capacity, letting you run closer to the true budget. Pick these always (they are free).
- **Checkpointing / offloading** trade compute or PCIe bandwidth for capacity in *training*. Pick these for "won't fit" during training.

Each lever buys back a specific scarce resource — capacity, bandwidth, or compute. Naming which one you are short of is the entire skill, and it is the subject of the next chapter.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>On the 288 GB part above, capacity stops being the limit and the batch is set by bandwidth instead: every decode step must read the weights <em>and</em> every active sequence's KV cache. At what total context does reading the KV cache cost more than reading 39 GB of FP4 weights, and what does that say about which lever matters next?</p>
</div>

---

## Why This Matters for Model Optimization

<div class="diagram">
<div class="diagram-title">The Memory Math Behind Every Serving Decision</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">"Will it fit?" is arithmetic</div>
    <div class="card-desc">weights + KV-cache + workspace ≤ capacity. The KV term (2·L·n_kv·d_head·seq·batch·bytes) is the one that surprises people — it can exceed the weights. Compute it before you deploy.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Decode speed = bytes ÷ bandwidth</div>
    <div class="card-desc">At batch=1, tok/s ≈ HBM_BW / model_bytes. The only knobs are fewer bytes/weight (quantize) and fewer weights (smaller/distilled model). Tensor-core peak is irrelevant here.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Batch to escape the memory wall</div>
    <div class="card-desc">AI ≈ batch. You must batch into the hundreds to reach the ridge. KV-cache capacity caps how far you can batch — which is why GQA/KV-quant/paging exist.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Prefill ≠ decode</div>
    <div class="card-desc">Prefill is compute-bound (TTFT, optimize FLOPs); decode is memory-bound (TPOT, optimize bytes). Chunked prefill interleaves them so each fills the roof the other leaves empty.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Training: activations dominate</div>
    <div class="card-desc">Activation memory grows with sequence length and dwarfs weights at long context. Checkpointing trades ~33% compute for O(√L) memory; offloading spills to CPU/NVMe.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Long context is a memory problem</div>
    <div class="card-desc">KV-cache (inference) and activations (training) both scale with sequence. Doubling context roughly doubles each. "Support 128k context" is first a capacity-and-bandwidth budget question.</div>
  </div>
</div>
</div>

The practical workflow this chapter hands you:

1. **Before deploying, run the HBM budget equation.** Sum weights + KV-cache (at your target seq and batch) + ~10% workspace. If it exceeds capacity, you have a *capacity* problem → quantize weights, quantize KV, GQA, or shard ([Chapter 28](./28_multi_gpu_and_distributed_training.md)).
2. **Compute the decode ceiling** (`HBM_BW / model_bytes`). If it is below your latency target, you have a *bandwidth* problem → lower precision or a smaller model. No amount of tuning beats this ceiling.
3. **Find your batch target on the roofline.** If throughput per GPU is too low, you are under-batching (memory-bound) → enable continuous batching + PagedAttention and raise the batch until either you hit the ridge or KV-cache fills HBM.
4. **For training, account for activations separately** and reach for checkpointing before offloading before sharding — cheapest resource first.

---

## Key Takeaways

- **Decode is memory-bandwidth-bound.** At batch=1 the token-rate ceiling is `HBM_BW / model_bytes`: a 70B INT8 model on an H100 (70 GB ÷ 3.35 TB/s) tops out near **~48 tok/s**, regardless of tensor-core peak.
- The **KV-cache** trades compute for memory and is sized by $2 \times L \times n_{kv} \times d_{head} \times \text{seq} \times \text{batch} \times \text{bytes}$. It grows with context and batch and **routinely exceeds the weights** (≈98% at 128k context; 4× the weights at batch 64 / 8k).
- **GQA/MQA** cut $n_{kv}$ (a 32→8 head reduction is a 4× KV cut), **MLA** caches a compressed latent instead, **KV quantization** cuts the byte width (2–4×), **PagedAttention** kills fragmentation, and **prefix caching** removes redundant prefill — all to fit more concurrent sequences.
- **Batching raises arithmetic intensity** (AI ≈ batch). It buys throughput by amortizing weight reads, at the cost of per-token latency — the throughput-vs-latency tradeoff. **Continuous batching** keeps the GPU full; KV-cache capacity caps how far you can go.
- **Prefill is compute-bound** (big GEMM, optimize FLOPs, governs TTFT); **decode is memory-bound** (GEMV, optimize bytes, governs TPOT). **Chunked prefill** interleaves them.
- In **training**, activation memory grows with sequence length and dwarfs weights at long context; **gradient checkpointing** trades ~33% extra compute for $O(\sqrt{L})$ memory, and **offloading** spills to CPU/NVMe to fit at all.
- Every serving decision is the **HBM budget equation**: weights + KV-cache + workspace ≤ capacity, streamed within the bandwidth ceiling. Name which of capacity / bandwidth / compute you are short of, and the right lever follows.

## Check Yourself

<details class="check">
<summary>Compute the KV cache per token for a model with 40 layers, 8 KV heads of dimension 128, in FP8. How many tokens fit in 20 GB?</summary>
<p>2 × 40 × 8 × 128 × 1 B = 81,920 B ≈ 80 KB per token. 20 × 10⁹ / 81,920 ≈ 244,000 tokens across all requests.</p>
</details>

<details class="check">
<summary>A 13B BF16 model decodes at batch 1 on a 4.8 TB/s H200. What is the token-rate ceiling, and what else reduces it at long context?</summary>
<p>26 GB per token → 4.8 TB/s / 26 GB ≈ 185 tokens/s. At long context the KV cache must be read every step as well, adding its bytes to the denominator.</p>
</details>

<details class="check">
<summary>Why is decode arithmetic intensity roughly equal to the batch size in BF16?</summary>
<p>Each weight (2 bytes) is read once per step and used for one multiply-add (2 FLOPs) per sequence in the batch, so FLOPs per byte ≈ 2B / 2 = B. KV-cache reads, which are not shared across sequences, pull it lower.</p>
</details>

<details class="check">
<summary>Your server OOMs at 64 concurrent 8k-token chats but GPU utilisation is low at 16. Name two levers and what each buys.</summary>
<p>FP8 KV cache (halves KV bytes, doubling the sequences that fit) and PagedAttention with continuous batching (removes fragmentation so the KV that is allocated is actually used). GQA or MLA help too but need a different model.</p>
</details>

<details class="check">
<summary>Why does disaggregating prefill and decode onto separate GPUs make sense from a roofline point of view?</summary>
<p>Prefill is compute-bound and decode is memory-bound. On separate pools each can be batched and sized for its own limit, and long prefills no longer stall decode steps, at the cost of moving the KV cache between pools.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Efficient Memory Management for LLM Serving with PagedAttention | Kwon et al., 2023 | [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180) | Paged KV cache and fragmentation numbers |
| GQA: Training Generalized Multi-Query Transformer Models | Ainslie et al., 2023 | [arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245) | Grouped-query attention |
| Fast Transformer Decoding: One Write-Head is All You Need | Shazeer, 2019 | [arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150) | Multi-query attention |
| DeepSeek-V2 | DeepSeek-AI, 2024 | [arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434) | Multi-head Latent Attention |
| Orca: A Distributed Serving System for Transformer-Based Generative Models | Yu et al., OSDI 2022 | [usenix.org/conference/osdi22/presentation/yu](https://www.usenix.org/conference/osdi22/presentation/yu) | Iteration-level (continuous) batching |
| SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills | Agrawal et al., 2023 | [arxiv.org/abs/2308.16369](https://arxiv.org/abs/2308.16369) | Chunked prefill |
| DistServe | Zhong et al., OSDI 2024 | [arxiv.org/abs/2401.09670](https://arxiv.org/abs/2401.09670) | Prefill/decode disaggregation |
| Training Deep Nets with Sublinear Memory Cost | Chen et al., 2016 | [arxiv.org/abs/1604.06174](https://arxiv.org/abs/1604.06174) | O(√L) activation checkpointing |
| Reducing Activation Recomputation in Large Transformer Models | Korthikanti et al., 2022 | [arxiv.org/abs/2205.05198](https://arxiv.org/abs/2205.05198) | Selective recomputation and sequence parallelism |
| ZeRO-Infinity | Rajbhandari et al., 2021 | [arxiv.org/abs/2104.07857](https://arxiv.org/abs/2104.07857) | Offloading to CPU and NVMe |

---

**Next:** [Chapter 30 — How Datatypes Drive Your Daily Optimization Choices](./30_datatypes_drive_optimization_choices.md) — These memory and bandwidth budgets become a single decision framework: given your model, GPU and regime, which datatype to use.

[← Back to Table of Contents](./README.md)
