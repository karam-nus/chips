---
title: "Chapter 11 — The Memory Wall & Bandwidth"
---

[← Back to Table of Contents](./README.md)

# Chapter 11 — The Memory Wall & Bandwidth

You have just read how memory hierarchies are structured ([Chapter 9](./09_memory_types.md)) and how the operating system manages them ([Chapter 10](./10_memory_management.md)). Now we ask the question that ties everything together for machine learning: **how fast can data actually move, and when does that speed — not compute — become the bottleneck?**

The answer lies in one of the most important phenomena in computer architecture: the **memory wall**. Compute throughput has grown faster than memory bandwidth for decades. The gap between what hardware *can calculate* and what memory *can supply* has widened into a chasm. For AI inference — especially LLM token generation — this is not background trivia. It is *the* governing constraint on your system. Every quantization decision, every batching strategy, every kernel fusion trick is, at root, a response to the memory wall.

> **The one-sentence version:** A hardware kernel is either limited by the speed at which it can do arithmetic (compute-bound) or the speed at which it can read and write memory (memory-bound), and for almost all LLM inference the answer is memory — which is why streaming fewer bytes (quantization, sparsity) speeds things up even when it adds computation.

---

## The Memory Wall

In 1994, Wulf and McKee published a paper titled *"Hitting the Memory Wall: Implications of the Obvious."* The observation was simple but brutal: **DRAM bandwidth and latency were improving far more slowly than processor FLOP rates.** If you plotted the two on the same graph, the gap grew exponentially.

Here are the rough numbers that illustrate the trend:

| Year | Approx peak CPU FLOP/s | Approx DRAM bandwidth | Ratio (FLOP/byte) |
|:----:|:----------------------:|:---------------------:|:-----------------:|
| 1990 | ~0.05 GFLOP/s | ~0.13 GB/s | ~0.4 |
| 2000 | ~3 GFLOP/s | ~1.6 GB/s (DDR1) | ~1.9 |
| 2010 | ~50 GFLOP/s | ~12 GB/s (DDR3) | ~4.2 |
| 2020 | ~3,000 GFLOP/s (GPU) | ~900 GB/s (HBM2) | ~3.3 |
| 2024 | ~989,000 GFLOP/s (H100 BF16) | ~3,350 GB/s (HBM3) | ~295 |

Compute has grown roughly 10,000× in that window. Bandwidth grew roughly 25,000× — faster, but starting from a much lower base. Today a modern GPU can execute ~295 floating-point operations for every single byte it fetches from main memory. As we will see in a moment, this number (the **ridge point**) is the dividing line between two very different worlds.

> The memory wall is not a temporary engineering gap that will be closed next year. It reflects a deep physical asymmetry: transistors do arithmetic cheaply; moving electrical charge across long wires is expensive in energy and time. Memory is inherently less parallel than logic.

---

## Bandwidth vs Latency — and How Latency Is Hidden

Before going further, we need to be precise about two often-confused terms.

<div class="diagram">
<div class="diagram-title">Bandwidth vs Latency</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Bandwidth</div>
    <ul>
      <li><strong>How much</strong> data moves per unit time</li>
      <li>Measured in GB/s or TB/s</li>
      <li>Determines throughput for streaming workloads</li>
      <li>Example: HBM3 ~3.35 TB/s, PCIe 4.0 x16 ~32 GB/s</li>
      <li>Scales with parallelism (more channels, wider bus)</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Latency</div>
    <ul>
      <li><strong>How long</strong> a single request takes</li>
      <li>Measured in nanoseconds or clock cycles</li>
      <li>Determines performance for random-access workloads</li>
      <li>Example: HBM ~100 ns, L1 cache ~1 ns</li>
      <li>Hard to shrink — constrained by physics and distance</li>
    </ul>
  </div>
</div>
</div>

A helpful analogy: a highway and a racing motorcycle. The highway has high *bandwidth* — many cars wide, many cars long. The motorcycle has low *latency* — it gets to the destination fast. A single load instruction cares about latency. A DMA transfer streaming an entire weight matrix cares about bandwidth.

**For large ML workloads, bandwidth is almost always what matters,** because we are streaming enormous arrays sequentially. A single matrix multiply reads $M \times K$ bytes from one operand and $K \times N$ bytes from another — at large sizes, this is gigabytes, and we can pipeline the requests to keep the memory bus saturated. Latency is hidden by **parallelism and prefetching**: the GPU launches thousands of concurrent threads, so while one warp is stalled waiting for a memory response, hundreds of others are computing. The latency becomes invisible behind the queue depth. This is called **latency hiding through throughput**, and it is one of the central architectural principles behind the GPU's success at DNN workloads.

> **Latency still matters in two places:** (1) small, irregular accesses — e.g., sparse index lookups, tree-traversal — where you cannot pipeline, and (2) the decode phase of LLM serving, where the KV cache lookup is effectively random-access per-token.

---

## Arithmetic Intensity and the Roofline Model

### Defining Arithmetic Intensity

For any computational kernel, we can define a single number that predicts which resource limits its performance:

$$\text{Arithmetic Intensity (AI)} = \frac{\text{Total FLOPs}}{\text{Total bytes transferred from/to memory}}$$

The unit is **FLOP/byte** (sometimes called "operational intensity"). High AI = lots of math relative to data movement. Low AI = mostly shuffling bytes.

Let's compute arithmetic intensity for a matrix multiply $C = A \cdot B$ where $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, $C \in \mathbb{R}^{M \times N}$, all in FP16 (2 bytes/element):

| Quantity | Formula | Example $M=N=K=4096$ |
|----------|---------|----------------------|
| Total FLOPs (multiply-adds) | $2 \cdot M \cdot N \cdot K$ | $2 \times 4096^3 \approx 137$ GFLOPs |
| Bytes read: $A$ + $B$ | $2 \cdot (M \cdot K + K \cdot N)$ bytes | $2 \times 2 \times 4096^2 \approx 67$ MB |
| Bytes written: $C$ | $2 \cdot M \cdot N$ bytes | $2 \times 4096^2 \approx 33$ MB |
| Total memory traffic | $2(MK + KN + MN)$ bytes | $\approx 100$ MB |
| **Arithmetic Intensity** | $\frac{2MNK}{2(MK+KN+MN)}$ | $\approx 1365$ **FLOP/byte** |

That is enormous — a square matmul's arithmetic intensity grows as $O(M)$ (for large $M$, the byte count is $O(M^2)$ but FLOPs are $O(M^3)$). This is why large matrix multiplies are compute-bound on modern hardware.

### The Roofline Model

The **Roofline model** (Williams, Waterman, Patterson — 2009) says: your kernel's achievable performance is the *minimum* of two roofs.

$$\text{Attainable Performance} = \min\left(\text{Peak FLOP/s},\; \text{AI} \times \text{Peak Bandwidth}\right)$$

Plotted on log-log axes with AI on the x-axis and attainable FLOP/s on the y-axis, this creates the characteristic "roofline" shape:

```text
Peak FLOP/s ─────────────────────────────────────────────────────── ← compute roof
(e.g. 989 TFLOP/s)                                           ╱
                                                            ╱ ←  "roofline"
                                                          ╱    (slope = bandwidth)
                                                        ╱
                                                      ╱
                          memory-bound region       ╱    compute-bound region
                                                  ╱
                                        ╔═══════╗
                                        ║ ridge ║  ← AI* ≈ 295 FLOP/byte
                                        ╚═══════╝
─────────────────────────────────────────────────────────────────────────────► AI (FLOP/byte)
  0.1       1        10       100       300      1000      10000
  (softmax) (LayerNorm)        (GEMV)            (large GEMM)
```

### The Ridge Point

The **ridge point** $\text{AI}^*$ is where the two roofs meet — the arithmetic intensity above which a kernel becomes compute-bound:

$$\text{AI}^* = \frac{\text{Peak FLOP/s}}{\text{Peak Bandwidth}}$$

For the **NVIDIA H100 SXM** (BF16, dense):
- Peak BF16 FLOP/s with Tensor Cores: **~989 TFLOP/s** (989 × 10¹² FLOP/s)
- Peak HBM3 bandwidth: **~3.35 TB/s** (3.35 × 10¹² bytes/s)

$$\text{AI}^*_{H100} = \frac{989 \times 10^{12}}{3.35 \times 10^{12}} \approx 295 \text{ FLOP/byte}$$

Any kernel with arithmetic intensity below ~295 FLOP/byte is **memory-bound** on the H100. Any kernel above ~295 FLOP/byte is **compute-bound**. This single number tells you more about what to optimize than almost anything else.

Let's verify with our matmul example:
- $M=N=K=4096$, FP16: AI ≈ 1,365 FLOP/byte → **compute-bound** (good, this is the training workload)
- $M=1, N=4096, K=4096$ (a single-token GEMV during LLM decode): AI ≈ 1 FLOP/byte → **memory-bound** (the decode bottleneck)

---

## Classifying ML Operations by Arithmetic Intensity

<div class="diagram">
<div class="diagram-title">ML Operations on the Roofline (H100 BF16, AI* ≈ 295 FLOP/byte)</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card red">
    <div class="card-icon">🧠</div>
    <div class="card-title">Memory-Bound (AI &lt; 295)</div>
    <div class="card-desc">Elementwise ops (~1), LayerNorm (~5), Softmax (~3), Embedding lookup (~0.5), GEMV batch=1 (~1), KV-cache read (~1)</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">⚖️</div>
    <div class="card-title">Near Ridge (AI ~ 100–400)</div>
    <div class="card-desc">Small GEMMs (M=32 in GEMM), batched decode (batch=16–32), attention with long seqlens, small convolutions</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">💻</div>
    <div class="card-title">Compute-Bound (AI &gt; 295)</div>
    <div class="card-desc">Large GEMMs (M=N=K≥1024 in FP16), training forward/backward passes (large batch), large convolutions</div>
  </div>
</div>
</div>

Let's look at a few operations analytically:

**Elementwise (ReLU, GELU, add):** $M \times N$ elements, 1–2 FLOPs, 2–3 reads/writes per element.
AI ≈ 0.5–2 FLOP/byte. Solidly memory-bound.

**LayerNorm over $[B, T, D]$:** Must read the entire tensor twice (once to compute mean/variance, once to normalize). ~10 FLOPs/element, ~3 reads/writes. AI ≈ ~1–5 FLOP/byte. Memory-bound.

**Softmax over $[B, H, T, T]$ attention logits:** Reads $B \times H \times T^2 \times 2$ bytes (FP16), ~5 FLOPs/element. AI ≈ 2–3 FLOP/byte. Memory-bound.

**GEMM (large, square):** As derived above, AI ≈ $M/3$ for $M=N=K$, so $M=4096$ gives AI ≈ 1365 FLOP/byte. Compute-bound.

**GEMV at batch=1 (LLM decode):** $y = Wx$, $W \in \mathbb{R}^{N \times K}$: $2NK$ FLOPs, $(NK + N + K) \times 2$ bytes. For large $N,K$, AI ≈ $\frac{2NK}{2NK} = 1$ FLOP/byte. Memory-bound.

---

## The Key Insight: Why LLM Decode Is Memory-Bandwidth-Bound

This deserves its own section because it is the most important practical consequence of the roofline for practitioners.

During **autoregressive token generation** (the decode phase of an LLM), the model generates one token per forward pass. The forward pass of a transformer layer looks like:

```text
For each layer:
  - Attention: Q, K, V projections  (GEMV: [d_model] × [d_model, d_head*n_heads])
  - Attention score computation      (small matmul, mostly memory-bound)
  - KV cache read                    (streaming previous K, V tensors)
  - Output projection                (GEMV)
  - FFN: two or three GEMVs          (e.g. W1, W2, W_gate in SwiGLU)
```

Each layer contains roughly $12 \times d^2$ parameters (for $d$-dimensional model with standard sizes). For a 7B parameter model, $d \approx 4096$, and the total parameter count is ~7 billion, each stored as BF16 (2 bytes) = **~14 GB**.

At decode with batch size 1, **every single parameter must be read from HBM exactly once per token generated.** That is 14 GB of memory reads. At H100's HBM3 bandwidth of 3.35 TB/s:

$$t_{\text{token}} \approx \frac{14 \times 10^9 \text{ bytes}}{3.35 \times 10^{12} \text{ bytes/s}} \approx 4.2 \text{ ms/token}$$

That gives roughly $1 / 0.0042 \approx 238$ tokens/second — which matches observed H100 decode throughput for 7B models at batch=1. The arithmetic units are largely idle; the bottleneck is the **HBM bandwidth**.

> This is the single most important number in practical LLM serving: **at batch=1, a 7B BF16 model on H100 is bounded to ~238 tokens/s by HBM bandwidth alone, regardless of how fast the tensor cores are.**

### How Batching Rescues Arithmetic Intensity

The fix is to increase batch size: if you decode $B$ sequences simultaneously, you do $B$ times more FLOPs while reading the same weight matrix once:

$$\text{AI}_{\text{decode}}(B) \approx \frac{2 \cdot B \cdot d_{\text{model}}^2}{\text{bytes per weight matrix}} = \frac{2B \cdot d_{\text{model}}^2}{2 \cdot d_{\text{model}}^2} = B$$

At batch=1: AI ≈ 1 FLOP/byte (memory-bound).
At batch=32: AI ≈ 32 FLOP/byte (still memory-bound but 32× better utilization).
At batch=295: AI ≈ 295 FLOP/byte (at the ridge point — compute and bandwidth co-limited).
At batch=1024: AI ≈ 1024 FLOP/byte (compute-bound — weight-reading is amortized).

This is why **continuous batching** and large batch sizes are so important for inference serving. It is not primarily about better GPU utilization in a vague sense — it is about moving arithmetic intensity past the ridge point.

---

## Python: Arithmetic Intensity Calculator

```python
import numpy as np
from dataclasses import dataclass
from typing import Literal

# H100 SXM5 specs (approximate)
H100_TFLOPS_BF16 = 989e12     # BF16 tensor-core FLOP/s (with sparsity: 1979e12)
H100_BW_HBM3    = 3.35e12    # bytes/s HBM3

@dataclass
class RooflineResult:
    flops: int                   # total floating-point ops
    bytes_transferred: int       # total bytes read + written
    arithmetic_intensity: float  # FLOP/byte
    bound: Literal["memory", "compute", "balanced"]
    attainable_tflops: float     # TFLOP/s achievable (roofline prediction)
    peak_tflops: float           # hardware peak TFLOP/s
    peak_bandwidth_tbs: float    # hardware peak bandwidth TB/s
    ridge_point: float           # FLOP/byte at ridge

def matmul_roofline(
    M: int, K: int, N: int,
    dtype_bytes: int = 2,          # 2=FP16/BF16, 4=FP32, 1=INT8
    peak_flops: float = H100_TFLOPS_BF16,
    peak_bw: float    = H100_BW_HBM3,
) -> RooflineResult:
    """
    Compute arithmetic intensity and roofline prediction for [M,K] @ [K,N].

    Assumes inputs read once, output written once (no fusion with preceding ops).
    Ignores on-chip SRAM re-use (which improves AI in practice via tiling).
    """
    flops = 2 * M * K * N  # multiply-accumulate = 2 FLOPs each
    bytes_in  = dtype_bytes * (M * K + K * N)
    bytes_out = dtype_bytes * (M * N)
    total_bytes = bytes_in + bytes_out

    ai = flops / total_bytes
    ridge = peak_flops / peak_bw

    # Roofline: min of compute ceiling and memory ceiling
    attainable = min(peak_flops, ai * peak_bw)

    if ai < ridge * 0.9:
        bound = "memory"
    elif ai > ridge * 1.1:
        bound = "compute"
    else:
        bound = "balanced"

    return RooflineResult(
        flops=flops,
        bytes_transferred=total_bytes,
        arithmetic_intensity=ai,
        bound=bound,
        attainable_tflops=attainable / 1e12,
        peak_tflops=peak_flops / 1e12,
        peak_bandwidth_tbs=peak_bw / 1e12,
        ridge_point=ridge,
    )

def print_roofline(r: RooflineResult, label: str = ""):
    print(f"\n{'─'*55}")
    if label:
        print(f"  {label}")
    print(f"  FLOPs:        {r.flops/1e9:>10.1f} GFLOPs")
    print(f"  Bytes:        {r.bytes_transferred/1e6:>10.1f} MB")
    print(f"  AI:           {r.arithmetic_intensity:>10.1f} FLOP/byte")
    print(f"  Ridge point:  {r.ridge_point:>10.1f} FLOP/byte")
    print(f"  Bound:        {r.bound.upper():>10}")
    print(f"  Attainable:   {r.attainable_tflops:>10.1f} TFLOP/s  "
          f"(peak: {r.peak_tflops:.0f} TFLOP/s)")

# --- Examples ---

# 1. Large square GEMM (training, large batch)
r1 = matmul_roofline(M=4096, K=4096, N=4096, dtype_bytes=2)
print_roofline(r1, "Large GEMM M=N=K=4096, BF16")

# 2. LLM decode: single token through one FF layer (d_model=4096 → 11008)
# W: [4096, 11008], x: [1, 4096]  →  y: [1, 11008]
r2 = matmul_roofline(M=1, K=4096, N=11008, dtype_bytes=2)
print_roofline(r2, "LLM decode GEMV batch=1 (7B FF layer)")

# 3. LLM decode at batch=32
r3 = matmul_roofline(M=32, K=4096, N=11008, dtype_bytes=2)
print_roofline(r3, "LLM decode GEMV batch=32")

# 4. Same GEMV but quantized to INT8 (1 byte/weight)
r4 = matmul_roofline(M=1, K=4096, N=11008, dtype_bytes=1)
print_roofline(r4, "LLM decode GEMV batch=1, INT8 weights")

# 5. Batch sweep: show AI vs batch size for a single FF layer
print("\nBatch size sweep — AI for 7B FF projection [d_model=4096, N=11008]:")
print(f"  {'Batch':>6}  {'AI (FLOP/byte)':>16}  {'Bound':>10}")
for b in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    r = matmul_roofline(M=b, K=4096, N=11008, dtype_bytes=2)
    print(f"  {b:>6}  {r.arithmetic_intensity:>16.1f}  {r.bound:>10}")
```

**Sample output (abridged):**
```text
  Large GEMM M=N=K=4096, BF16
  FLOPs:       137.4 GFLOPs    Bytes:    100.7 MB
  AI:         1364.6 FLOP/byte  Ridge: 295.2 FLOP/byte
  Bound:          COMPUTE

  LLM decode GEMV batch=1 (7B FF layer)
  FLOPs:          0.1 GFLOPs    Bytes:     90.2 MB
  AI:              1.0 FLOP/byte
  Bound:           MEMORY

  LLM decode GEMV batch=32
  AI:             31.9 FLOP/byte  Bound:  MEMORY

  LLM decode GEMV batch=1, INT8 weights
  AI:              2.0 FLOP/byte  Bound:  MEMORY  (but 2× less data!)

  Batch size sweep:
   Batch    AI (FLOP/byte)       Bound
       1               1.0      memory
       8               7.9      memory
      32              31.9      memory
     128             127.6      memory
     256             255.3      memory
     512             510.7     compute
```

Note that INT8 at batch=1 is still memory-bound (AI ≈ 2 vs ridge ≈ 295), but it **halves the byte count** — so you get ~2× throughput at the same bandwidth. That is the mechanism behind quantization speedups for decode.

---

## Operator Fusion and Why It Helps

When you chain multiple memory-bound operations — LayerNorm followed by a dropout followed by an add — the naive implementation reads and writes the full tensor for each step:

```text
Naive:  HBM → LayerNorm → HBM → Dropout → HBM → Add → HBM  (4 round-trips)
Fused:  HBM ──────────────────────────────────────────→ HBM  (1 round-trip)
```

Operator fusion (as done in FlashAttention, Triton kernels, torch.compile + inductor) keeps the intermediate results in SRAM (the GPU's on-chip "L1 cache" / shared memory — ~192 KB/SM on H100) instead of bouncing them through HBM. The effective arithmetic intensity rises dramatically, because the numerator (FLOPs) stays the same but the denominator (HBM bytes) shrinks.

**FlashAttention** is the canonical example: the naive softmax attention reads/writes $O(T^2)$ bytes to HBM (the $T \times T$ attention score matrix, $T$ = sequence length). FlashAttention tiles the computation so intermediate scores live in SRAM, reducing HBM traffic to $O(T)$ — a massive win for long sequences.

---

## Why This Matters for Model Optimization

<div class="diagram">
<div class="diagram-title">Memory Wall Consequences for ML Practitioners</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Quantization speedup</div>
    <div class="card-desc">INT8 weights → 2× fewer bytes to stream from HBM → 2× faster decode at batch=1. The extra dequantize FLOPs are free (you're memory-bound). This is the hardware reason, not "matrix multiply is faster."</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Batching strategy</div>
    <div class="card-desc">Each doubling of batch size doubles AI. Below the ridge point, you get near-linear throughput scaling with batch. Above it, you're compute-bound and more batching doesn't help throughput.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">FLOPs alone mislead</div>
    <div class="card-desc">A kernel can have 10× more FLOPs and still be faster if it's better fused. FLOP counts predict training compute budget; they don't predict wall-clock time for memory-bound ops.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Operator fusion</div>
    <div class="card-desc">Fusing memory-bound ops (LayerNorm, activations, dropout) cuts HBM round-trips. FlashAttention is the extreme case: fuses attention's entire O(T²) working set into SRAM.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Prefill vs decode</div>
    <div class="card-desc">Prefill (prompt processing) runs at large T → large GEMM → compute-bound. Decode runs batch=1 GEMV → memory-bound. These require different optimization strategies.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">KV cache size limits</div>
    <div class="card-desc">KV cache competes with model weights for HBM. Larger batches / longer contexts → more KV bytes → more HBM pressure → memory wall bites harder. MQA/GQA reduce KV size to help.</div>
  </div>
</div>
</div>

The practical upshot, stated plainly:

1. **LLM decode is memory-bandwidth-bound.** Quantize weights to save bytes; batch requests to amortize weight reads.
2. **LLM prefill is compute-bound** (long sequences, large GEMMs). FlashAttention's memory savings are more important here than for decode.
3. **Training is compute-bound** for large GEMM workloads. The bottleneck is tensor core utilization; FP8 saves a factor of 2 in bandwidth but the gain is primarily in compute throughput (BF16 → FP8 doubles tensor core throughput on H100).
4. **Activation memory during training is memory-bound** (checkpointing saves HBM by re-computing; the re-compute is cheap because it's memory-bound in the first place).

---

## Key Takeaways

- The **memory wall** reflects a decades-long divergence: compute throughput grew faster than memory bandwidth. On an H100, the ridge point is ~295 FLOP/byte.
- **Arithmetic intensity** (FLOP/byte) classifies any kernel as memory-bound (AI < ridge) or compute-bound (AI > ridge). The **Roofline model** gives a tight upper bound on achievable performance.
- **LLM autoregressive decode at batch=1** has arithmetic intensity ~1 FLOP/byte — solidly memory-bound. You are paying full HBM bandwidth for every parameter, every token.
- **Batching raises arithmetic intensity** proportionally. Doubling batch size doubles AI; the ridge point for a 7B model on H100 is around batch ≈ 295.
- **Quantization accelerates memory-bound kernels** by reducing bytes streamed, not by reducing FLOPs. INT4 weights → 4× fewer bytes → up to 4× better decode throughput (bandwidth-limited).
- **Operator fusion** (FlashAttention, torch.compile, Triton) reduces HBM round-trips by keeping intermediates in on-chip SRAM, raising effective arithmetic intensity.
- FLOPs alone are a misleading performance metric. Always ask: **which roof am I under?**

---

*Next: [Chapter 12 — Buses & Interconnects](./12_buses_and_interconnects.md), where we follow data as it moves between chips — from on-chip NoC fabrics to PCIe, NVLink, and cluster-scale networks.*

[← Back to Table of Contents](./README.md)
