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
    <div class="card-desc">Store K/V at INT8 or FP8 instead of FP16 → <strong>2× cut</strong>; some systems push to INT4 for <strong>4×</strong>. Targets long context specifically. See <a href="./25_quantization_hardware_view.md">Ch 25</a>.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Stop wasting it → PagedAttention</div>
    <div class="card-desc">Allocate the cache in fixed <strong>blocks (pages)</strong> like OS virtual memory (<a href="./10_memory_management.md">Ch 10</a>) instead of one contiguous slab per request. Kills fragmentation; lifts usable batch 2–4×.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Don't recompute → prefix caching</div>
    <div class="card-desc">Shared prompt prefixes (system prompts, few-shot examples) are cached once and reused across requests — saves prefill compute <em>and</em> KV memory for the shared span.</div>
  </div>
</div>
</div>

**GQA/MQA** ([grouped/multi-query attention]) directly attack $n_{kv}$. With $n_{heads}$ query heads but only $n_{kv}$ key/value heads, several query heads *share* one K/V head. MQA ($n_{kv}=1$) is the extreme; GQA ($n_{kv}=4$ or $8$) is the sweet spot that keeps quality. The KV-cache shrinks by exactly $n_{heads}/n_{kv}$.

**KV-cache quantization** attacks the `bytes` factor. Keys and values are activations, so they are *harder* to quantize than weights (outliers — see [Chapter 25](./25_quantization_hardware_view.md)), but per-token or per-channel scaling makes INT8/FP8 KV practical, halving the cache. It is the right lever precisely when context is long (the cache, not the weights, is your capacity problem).

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

The classic result: checkpointing every layer reduces activation memory from $O(L)$ to roughly $O(\sqrt{L})$ at the cost of **one extra forward pass (~33% more compute)**. In PyTorch it is a one-line wrap with `torch.utils.checkpoint.checkpoint`. The recompute is comparatively cheap because — as [Chapter 11](./11_memory_wall_and_bandwidth.md) showed — much of the forward pass is memory-bound, so the "extra" FLOPs are partly hidden behind bandwidth you were paying for anyway.

### Offloading: Spilling to CPU and NVMe

When even checkpointing is not enough, you can **offload** tensors down the hierarchy from [Chapter 9](./09_memory_types.md): push optimizer states, gradients, or activations to **CPU DRAM** (~50–100 GB/s over PCIe) or even **NVMe** (~3–14 GB/s). This is what ZeRO-Offload / ZeRO-Infinity do ([Chapter 28](./28_multi_gpu_and_distributed_training.md)). It buys capacity at a steep bandwidth cost: moving a tensor to CPU and back over PCIe 5.0 (~64 GB/s) is ~50× slower than HBM, so offloading is a "fit it at all" tool, not a "make it fast" tool. The art is overlapping the transfers with compute so the PCIe latency hides behind the GPU's work.

---

## Prefill vs Decode: Two Workloads, Two Optimizations

A single LLM request has **two phases with opposite roofline character**, and conflating them is a common and expensive mistake.

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

```text
┌──────────────────────── 80 GB HBM (H100) ────────────────────────┐
│ Weights (fixed)      │ KV-cache (grows with seq × batch) │ work │
│ e.g. 70B INT8 = 70GB │  ◄── this is what caps your batch ──►│ space│
└───────────────────────────────────────────────────────────────────┘
        capacity problem            capacity + bandwidth problem
```

- Quantizing **weights** (INT8/INT4) frees capacity *and* raises the decode ceiling (bandwidth). Pick this for "won't fit" and "too slow at batch=1."
- Quantizing the **KV-cache** (INT8/FP8) and using **GQA** free the capacity that limits your batch. Pick these for "OOMs at high concurrency / long context."
- **PagedAttention + continuous batching** reclaim *wasted* capacity, letting you run closer to the true budget. Pick these always (they are free).
- **Checkpointing / offloading** trade compute or PCIe bandwidth for capacity in *training*. Pick these for "won't fit" during training.

Each lever buys back a specific scarce resource — capacity, bandwidth, or compute. Naming which one you are short of is the entire skill, and it is the subject of the next chapter.

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
- **GQA/MQA** cut $n_{kv}$ (a 32→8 head reduction is a 4× KV cut), **KV quantization** cuts the byte width (2–4×), **PagedAttention** kills fragmentation, and **prefix caching** removes redundant prefill — all to fit more concurrent sequences.
- **Batching raises arithmetic intensity** (AI ≈ batch). It buys throughput by amortizing weight reads, at the cost of per-token latency — the throughput-vs-latency tradeoff. **Continuous batching** keeps the GPU full; KV-cache capacity caps how far you can go.
- **Prefill is compute-bound** (big GEMM, optimize FLOPs, governs TTFT); **decode is memory-bound** (GEMV, optimize bytes, governs TPOT). **Chunked prefill** interleaves them.
- In **training**, activation memory grows with sequence length and dwarfs weights at long context; **gradient checkpointing** trades ~33% extra compute for $O(\sqrt{L})$ memory, and **offloading** spills to CPU/NVMe to fit at all.
- Every serving decision is the **HBM budget equation**: weights + KV-cache + workspace ≤ capacity, streamed within the bandwidth ceiling. Name which of capacity / bandwidth / compute you are short of, and the right lever follows.

---

*Next: [Chapter 30 — How Datatypes Drive Your Daily Optimization Choices](./30_datatypes_drive_optimization_choices.md), where these memory and bandwidth budgets become a single decision framework: given your model, GPU, and regime, exactly which precision and technique to reach for.*

[← Back to Table of Contents](./README.md)
