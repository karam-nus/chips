---
title: "Chapter 16 — TPUs & Systolic Arrays"
---

[← Back to Table of Contents](./README.md)

# Chapter 16 — TPUs & Systolic Arrays

If you have ever wondered why Google can train Gemini on thousands of chips simultaneously without the network becoming the bottleneck, the answer is the **systolic array** — a 50-year-old idea that Google quietly turned into one of the most influential hardware architectures in AI history. The Tensor Processing Unit (TPU) is, at its heart, a very large, very careful implementation of that idea, wrapped in custom memory, interconnect, and a compiler that controls everything.

This chapter starts from first principles: why systolic arrays are so efficient for matrix multiplication, how data flows through a grid of MAC cells, and what that means for compute density. We then trace the full history of the Google TPU from the original inference-only TPUv1 in 2015 through the Trillium (v6) pods of 2024, paying careful attention to the numbers — TFLOP/s, memory bandwidth, interconnect bandwidth — that determine whether a model trains efficiently or wastes cycles.

> **The one-sentence version:** A systolic array is a 2-D grid of multiply-accumulate (MAC) units where weights sit still and activations/partial sums rhythmically pump through, so O(N²) multiply-accumulates happen with only O(N) memory reads — that extreme data-reuse ratio is why matrix math on a TPU is so efficient.

---

## Matrix Multiplication Is the Bottleneck

Before building the hardware, recall what neural network training and inference *are* in silicon terms. The dominant operation in a Transformer, CNN, or MLP is a **general matrix multiply (GEMM)**:

$$C = A \cdot B, \quad A \in \mathbb{R}^{M \times K},\; B \in \mathbb{R}^{K \times N},\; C \in \mathbb{R}^{M \times N}$$

The total compute is $2 \cdot M \cdot K \cdot N$ floating-point operations (one multiply + one add per MAC). For a typical LLM with hidden size 8192 and sequence length 2048, a single attention projection is roughly $2 \times 2048 \times 8192 \times 8192 \approx 275 \text{ GFLOP}$ — and this happens dozens of times per token, per layer, for every element in the batch.

The naive way to compute a GEMM is to stream matrix rows and columns from memory. The problem: at machine-learning scale, memory bandwidth becomes the bottleneck. As we established in [Chapter 11 — Memory Wall & Bandwidth](./11_memory_wall_and_bandwidth.md), bandwidth is precious and compute is cheap — the goal is to *maximize arithmetic intensity* (FLOPs per byte of memory traffic).

> **Arithmetic intensity of a naive GEMM:** Loading A costs $M \times K \times \text{bytes/element}$; loading B costs $K \times N \times \text{bytes/element}$; storing C costs $M \times N \times \text{bytes/element}$. For large square matrices ($M=K=N=n$) in BF16 (2 bytes), total I/O = $4n^2$ bytes, total FLOPs = $2n^3$, so arithmetic intensity = $n/2$. At $n = 1024$, that is 512 FLOP/byte — solidly in the compute-bound region. But the naive implementation doesn't achieve this; it re-reads tiles unnecessarily.

The systolic array is the hardware answer to maximizing data reuse.

---

## The Systolic Array: First Principles

The term "systolic" was coined by H.T. Kung and Charles Leiserson at CMU in 1978, by analogy to the heart: data "pulses" through a regular grid of processing elements in lockstep, the way blood pulses through arteries.

### Anatomy of a MAC Cell

Each cell in the array has:

- An **accumulator register** (holds the running partial sum for one output element of C)
- A **multiply unit**: computes $w \times a$ in one cycle
- Two **pass-through registers**: one for the activation flowing horizontally, one for the weight flowing vertically (or held stationary)
- Connections to four neighbors (left, right, up, down)

In the **weight-stationary** dataflow used by Google's Matrix Multiply Unit (MXU):

1. Weights are **pre-loaded** into the array — one weight per cell — and **held there** for the duration of the matmul.
2. Activations stream **left → right** along rows.
3. Partial sums propagate **top → bottom** along columns, accumulating as they go.

Each cycle: every cell multiplies its local (stationary) weight by the activation arriving from the left, adds that product to the partial sum arriving from above, and passes both activation and partial sum along. After $K$ cycles (the inner dimension), column $j$ has accumulated the full dot product of one row of A with column $j$ of B — i.e., one element of C.

### Step-by-Step ASCII Trace

Consider $A \in \mathbb{R}^{2 \times 2}$, $B \in \mathbb{R}^{2 \times 2}$, on a 2×2 array:

$$A = \begin{bmatrix} a_{00} & a_{01} \\ a_{10} & a_{11} \end{bmatrix}, \quad B = \begin{bmatrix} b_{00} & b_{01} \\ b_{10} & b_{11} \end{bmatrix}$$

Weights loaded into cells (weight-stationary):

```text
       col 0        col 1
row 0 [ b_00 ]    [ b_01 ]
row 1 [ b_10 ]    [ b_11 ]
```

Activations injected diagonally (staggered by 1 cycle so they arrive at the right cell at the right time):

```text
Cycle 0:   a_00 → row 0 left edge      (a_10 queued for cycle 1)
           partial sums = 0 everywhere

Cycle 0:
  Cell(0,0):  acc = 0 + a_00 * b_00  →  passes a_00 right; partial sum 0+a00*b00 down
  Cell(0,1):  acc = 0 + (nothing yet) * b_01
  Cell(1,0):  acc = 0 + (nothing yet) * b_10
  Cell(1,1):  acc = 0 + (nothing yet) * b_11

Cycle 1:
  a_00 reaches Cell(0,1); a_10 enters Cell(0,0); a_01 enters Cell(1,0) (next row, staggered)
  Cell(0,0):  acc += a_10 * b_00    → passes partial sum (a_00*b_00 + a_10*b_10... wait)

──────────────────────────────────────────────────────────────────────────────
Corrected view: input rows are staggered by 1 cycle to align operands properly
──────────────────────────────────────────────────────────────────────────────

End of K=2 cycles:
  Cell(0,0) accumulator = a_00*b_00 + a_01*b_10  = C[0,0]  ✓
  Cell(0,1) accumulator = a_00*b_01 + a_01*b_11  = C[0,1]  ✓
  Cell(1,0) accumulator = a_10*b_00 + a_11*b_10  = C[1,0]  ✓
  Cell(1,1) accumulator = a_10*b_01 + a_11*b_11  = C[1,1]  ✓
```

**Key insight:** during those $K$ cycles, *each weight was read from memory exactly once* (at pre-load time), and the activations were *broadcast along rows* — each activation value is read from the input buffer once and reused by every cell in its row. The memory traffic is $O(N)$ (edges of the array), while the compute is $O(N^2)$ per column of B loaded — for a large square array, the arithmetic intensity scales as $O(N)$, which is excellent.

### Why This Beats a GPU SIMT Core for Dense Matmul

A GPU SIMT thread doing a dot product has to re-fetch both operands from shared memory. A systolic array's cells *never go to memory* for weights or partial sums — they live in the cell's register/accumulator and are passed to the neighbor. The wiring *is* the data path; there is no cache miss to suffer. The MXU's data reuse is essentially perfect for weight-stationary GEMM.

<div class="diagram">
<div class="diagram-title">Systolic vs SIMT Dataflow</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Systolic Array (TPU MXU)</div>
    <ul>
      <li>Weights stationary in cells</li>
      <li>Activations pump horizontally</li>
      <li>Partial sums drain vertically</li>
      <li>Memory traffic: O(N) per O(N²) MACs</li>
      <li>No cache; wiring IS the datapath</li>
      <li>Deterministic latency, no hazards</li>
      <li>Poor at irregular / sparse work</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">GPU SIMT Tensor Core</div>
    <ul>
      <li>Operands loaded from shared memory each warp</li>
      <li>Thread groups do 16×16 wmma tiles</li>
      <li>Memory traffic: shared mem per tile</li>
      <li>Flexible: can handle varied ops</li>
      <li>Programmable; supports sparsity, varied dtypes</li>
      <li>Higher overhead for control, scheduling</li>
    </ul>
  </div>
</div>
</div>

---

## NumPy Simulation of a Systolic Array

The following simulation makes the step-by-step data flow concrete. It is intentionally simple — a software model, not cycle-accurate RTL — but it reveals *exactly* what happens each cycle.

```python
import numpy as np

def systolic_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Weight-stationary systolic-array simulation.
    A: (M, K)  activations
    B: (K, N)  weights (pre-loaded into the N×K virtual cells)
    Returns C: (M, N)
    
    Each "cell" at (col_j, row_k) holds weight B[k, j].
    Activations flow left-to-right; partial sums top-to-bottom.
    Input rows are staggered by 1 cycle to align operands.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dims must match: {K} vs {K2}"

    # Accumulator array — one per output cell
    acc = np.zeros((M, N), dtype=np.float64)

    # Each of the M rows of A is injected one cycle later than the previous
    # Total cycles needed: K + M - 1  (pipeline fill + drain)
    total_cycles = K + M - 1

    for cycle in range(total_cycles):
        # Which (row_of_A, k_step) pairs are active this cycle?
        for m in range(M):
            k = cycle - m           # activation A[m, k] arrives at the array at cycle = m + k
            if 0 <= k < K:
                # This row's k-th element is active: it multiplies every weight in column k
                for n in range(N):
                    acc[m, n] += A[m, k] * B[k, n]

    return acc

# --- Test ---
rng = np.random.default_rng(42)
M, K, N = 4, 8, 4
A = rng.standard_normal((M, K)).astype(np.float32)
B = rng.standard_normal((K, N)).astype(np.float32)

C_systolic = systolic_matmul(A.astype(np.float64), B.astype(np.float64))
C_ref = A.astype(np.float64) @ B.astype(np.float64)

print("Max error:", np.max(np.abs(C_systolic - C_ref)))   # should be ~0
# Max error: 0.0
```

On real hardware the "loop over cycle" is the clock, the "loop over n" is the N parallel MAC units in each row, and the `acc[m, n]` is a physical register inside cell (m, n). The weights in B are pre-loaded (via DMA from HBM) and stay resident while many tiles of A stream through — reuse across the M dimension as well.

---

## The bf16 Datatype: Google's Contribution

Before examining TPU generations, we need to understand **bfloat16 (bf16)**, because Google invented it specifically for TPU training.

IEEE-754 FP32 uses 1 sign bit, 8 exponent bits, 23 mantissa bits (32 bits total). **bf16 simply truncates the mantissa to 7 bits**, preserving all 8 exponent bits:

```text
FP32:  S [EEEEEEEE] [MMMMMMMMMMMMMMMMMMMMMMM]   32 bits
BF16:  S [EEEEEEEE] [MMMMMMM]                   16 bits
FP16:  S [EEEEE]    [MMMMMMMMMM]                16 bits
```

| Format | Bits | Exponent | Mantissa | Max value | Min normal |
|--------|------|----------|----------|-----------|------------|
| FP32 | 32 | 8 | 23 | ~3.4×10³⁸ | ~1.2×10⁻³⁸ |
| BF16 | 16 | 8 | 7 | ~3.4×10³⁸ | ~1.2×10⁻³⁸ |
| FP16 | 16 | 5 | 10 | ~6.5×10⁴ | ~6.1×10⁻⁵ |

The payoff: bf16 **covers the same dynamic range as FP32**, so gradient magnitudes and weight magnitudes that would overflow FP16 (and require loss scaling) fit naturally in bf16. You lose 16 bits of precision (mantissa 23→7), but it turns out that is acceptable for training because **gradient noise dominates numerical noise** in stochastic gradient descent. Converting FP32 ↔ bf16 is a truncation, not a rescaling — trivial in hardware.

> **Why not just use FP16?** FP16's exponent is only 5 bits, max value ≈ 65,504. Large gradient norms easily overflow. Loss scaling (artificially scaling the loss before the backward pass then re-scaling gradients) is the workaround — but it complicates the training loop and occasionally still produces NaNs. bf16 eliminates that problem. This is why TPUv2/v3 trained in bf16 while NVIDIA chips needed mixed-precision (FP16 with FP32 master weights) until Ampere added bf16 support in 2020.

```python
import struct

def fp32_to_bf16_bits(x: float) -> int:
    """Truncate FP32 to BF16 by dropping the low 16 mantissa bits."""
    bits = struct.pack('>f', x)                 # 4 bytes, big-endian
    return (bits[0] << 8) | bits[1]             # keep top 2 bytes

def bf16_to_fp32(bf16_bits: int) -> float:
    """Expand BF16 back to FP32 (zero-extend mantissa)."""
    fp32_bytes = bytes([bf16_bits >> 8, bf16_bits & 0xFF, 0, 0])
    return struct.unpack('>f', fp32_bytes)[0]

x = 3.14159
bf = fp32_to_bf16_bits(x)
print(f"FP32: {x:.6f}  →  BF16 approx: {bf16_to_fp32(bf):.6f}")
# FP32: 3.141590  →  BF16 approx: 3.140625  (error < 0.05%)
```

---

## Google TPU: History and Architecture

### TPUv1 (2015, deployed 2016) — Inference Only

Google deployed the TPUv1 internally to serve production inference workloads (AlphaGo ran on it). Design priorities: **high throughput INT8 inference** at low power, no training.

- **Matrix Multiply Unit (MXU):** 256 × 256 systolic array = 65,536 INT8 MAC units. Peak: 92 TOPS INT8 (tera-operations/second).
- **Weights FIFO:** 28 MB of on-chip weight memory — large enough to buffer layers and keep the MXU fed.
- **Accumulators:** 4 MB of 32-bit accumulator registers.
- **Unified buffer:** 24 MB SRAM for activations.
- **Memory:** 8 GB LPDDR3 at ≈34 GB/s — modest by modern standards; the philosophy was "keep weights on-chip."
- **Process:** 28 nm (older node), minimal area devoted to control.
- **Power:** ≈40 W TDP — under a typical PCIe card.

The TPUv1 had **no HBM, no floating-point training capability, and no programmable scalar unit** — it was purely a GEMM accelerator connected to a host CPU via PCIe. Despite this narrowness, it delivered ~30× better performance/watt than contemporary CPUs for neural-net inference in Google's production services.

### TPUv2 (2017) — Training + bf16 + HBM

TPUv2 introduced training capability, requiring gradient computation and higher precision.

- **MXU:** 128 × 128 systolic array (two chips on a board), bf16 input → FP32 accumulation.
- **HBM:** 8 GB HBM per chip at **600 GB/s** — critical for training's larger activation footprints.
- **Vector Unit (VPU):** 128-element SIMD for non-matmul ops (softmax, layer norm, activation functions).
- **Scalar Unit:** General-purpose control and scalar computation.
- **Inter-chip interconnect (ICI):** Each chip has **2D torus links** at **600 GB/s** bidirectional — allowing TPU Pods without CPU-mediated communication.
- **TPU Pod:** 64 TPUv2 chips in a 2D torus, ≈11.5 PFLOP/s peak (bf16).
- **Peak per chip:** 45 TFLOP/s bf16.

### TPUv3 (2018) — Liquid Cooling + More Memory

- **MXU:** Same architecture as v2, but faster clock and two MXUs per chip.
- **HBM:** 16 GB at **900 GB/s**.
- **Peak:** 420 TFLOP/s bf16 per chip (with two MXUs).
- **Pod:** 256 chips, ≈100 PFLOP/s — trained BERT in 76 minutes.
- **Cooling:** First liquid-cooled TPU (chip power ≈250 W, necessitating direct water cooling).

### TPUv4 (2021, publicly available 2023)

- **Architecture:** Redesigned MXU; FP32, bf16, INT8, complex number formats.
- **HBM2e:** 32 GB at **1.2 TB/s**.
- **Peak:** 275 TFLOP/s bf16 per chip.
- **ICI:** 3D torus topology — 6 high-speed optical interconnects per chip.
- **Pod:** 4096 chips in a 3D torus, >1 EFLOP/s (bf16).
- **Optical Reconfigurable Interconnect (OCS):** Software-defined topology — the pod's torus wiring can be reconfigured in minutes to accommodate different parallelism strategies.

### TPUv5e and TPUv5p (2023)

Google split the v5 family into two SKUs:

- **TPUv5e** ("e" = efficiency): Optimized for cost-effective inference and fine-tuning. 393 TFLOP/s bf16, 16 GB HBM, 1.6 Tb/s ICI. 256 chips per pod.
- **TPUv5p** ("p" = performance): Maximum training throughput. 459 TFLOP/s bf16, 95 GB HBM3, 4.8 Tb/s ICI per chip. 8960 chips per pod (≈4 EFLOP/s pod).

### Trillium — TPUv6 (2024)

- **Peak:** 4× the compute of TPUv5e per chip — ≈1570 TFLOP/s bf16 (estimated from Google's 4× claim).
- **HBM:** 192 GB HBM3e, ~4.7 TB/s bandwidth.
- **ICI bandwidth:** 1.8× vs v5e.
- **Power efficiency:** 67% better performance/watt vs v5e.
- **Pod scale:** Up to 100,000+ chips in a cluster via Titanium infrastructure.

---

## TPU Generations at a Glance

| Generation | Year | Peak (TFLOP/s bf16) | HBM | HBM BW | ICI BW/chip | Key dtype |
|:----------:|:----:|:-------------------:|:---:|:-------:|:-----------:|:---------:|
| TPUv1 | 2016 | 92 TOPS INT8 | — | 34 GB/s (LPDDR3) | — | INT8 |
| TPUv2 | 2017 | 45 | 8 GB HBM | 600 GB/s | 600 GB/s | bf16 |
| TPUv3 | 2018 | 420 | 16 GB HBM | 900 GB/s | 900 GB/s | bf16 |
| TPUv4 | 2021 | 275 | 32 GB HBM2e | 1,200 GB/s | ~1,600 GB/s | bf16/INT8 |
| TPUv5e | 2023 | 393 | 16 GB HBM | 1,600 GB/s | 1,600 Gb/s | bf16/INT8 |
| TPUv5p | 2023 | 459 | 95 GB HBM3 | 4,800 GB/s | 4,800 Gb/s | bf16/INT8 |
| Trillium (v6) | 2024 | ~1,570 | 192 GB HBM3e | ~4,700 GB/s | ~3,200 Gb/s | bf16/INT8 |

---

## The TPU Pod and Inter-Chip Interconnect (ICI)

A single TPU chip is powerful, but the real advantage of TPUs at Google's scale is the **Pod** — a tightly-coupled cluster where hundreds to thousands of chips communicate without touching the CPU's PCIe fabric.

<div class="diagram">
<div class="diagram-title">TPU Pod Topology</div>
<div class="flow">
  <div class="flow-node accent wide">TPU Chip — MXU + VPU + HBM</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">ICI: direct chip-to-chip links (6 directions in 3D torus)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">TPU Pod Slice (e.g. 4×4×4 = 64 chips)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Full Pod (256–8960 chips) — one logical supercomputer</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Multi-Pod Cluster (100k+ chips via Titanium fabric)</div>
</div>
</div>

The ICI is a **dedicated, high-bandwidth, low-latency** network on the same substrate as the chips. Key properties:

- **Bandwidth:** Up to 4.8 Tb/s bidirectional per chip (v5p) — comparable to NVLink on NVIDIA's flagship GPUs but integrated into the pod's topology without a switch.
- **Latency:** Sub-microsecond between adjacent chips (in-package links on v4 are optical).
- **Topology:** 2D torus (v2/v3), 3D torus (v4+). In a 3D torus with $d \times d \times d$ chips, the diameter is $3 \cdot d/2$ hops — a 16×16×16 pod has diameter 24 hops at sub-µs per hop.
- **All-reduce efficiency:** The torus topology allows ring-based all-reduce with near-linear bandwidth scaling, which is why data-parallel training across thousands of chips is efficient — unlike CPU-based parameter servers with PCIe bottlenecks.

> **For model optimization:** This ICI design is why "tensor parallelism" on TPU Pods is different from GPU clusters. On a TPU Pod you can split a single attention head across 8 chips with negligible ICI overhead; on a GPU cluster you pay NVLink latency and PCIe bandwidth asymmetry. The pod is the reason Google could train PaLM (540B parameters) across 6144 TPUv4 chips with 57.5% hardware utilization — among the highest ever reported for a supercomputer at that scale.

---

## The XLA Compiler and Static Shapes

TPUs are **compiled** devices, not interpreted ones. Programs run on a TPU must first be compiled by **XLA (Accelerated Linear Algebra)**, Google's optimizing compiler for tensor computations.

<div class="diagram">
<div class="diagram-title">XLA Compilation Model</div>
<div class="flow-h">
  <div class="flow-node accent">JAX/TF program<br/><small>high-level ops</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">HLO graph<br/><small>High-Level Ops IR</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">XLA optimization<br/><small>fusion, tiling, layout</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">LLO / LHLO<br/><small>low-level IR</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange">TPU binary<br/><small>statically scheduled</small></div>
</div>
</div>

**Why static shapes?** The MXU's systolic pipeline is *feed-forward* — there is no dynamic branching inside the array. XLA must tile the matmul into 128×128 (or 256×256) blocks at compile time, statically scheduling the prefetch of each tile from HBM into the unified buffer. If tensor shapes are not known at compile time, XLA cannot produce an optimal schedule. This is the trade-off versus GPU eager execution:

| | TPU / XLA | GPU / CUDA eager |
|--|-----------|-----------------|
| Compilation | Ahead-of-time, static | JIT or none |
| Shape flexibility | Requires static (or padded) shapes | Dynamic shapes OK |
| Kernel fusion | Aggressive, whole-graph | Op-by-op or manual |
| Overhead per op | Zero (compiled away) | CUDA launch overhead ~few µs |
| Best for | Large, regular matrix ops | Flexible, varied workloads |

```python
import jax
import jax.numpy as jnp

# JAX on TPU: shapes must be static at trace time
# @jax.jit triggers XLA compilation; result is a TPU binary
@jax.jit
def my_matmul(A, B):
    # XLA will tile this into 128×128 MXU blocks
    return jnp.dot(A, B)

# First call: compiles (~seconds). All subsequent calls: uses cached binary.
A = jax.random.normal(jax.random.PRNGKey(0), (4096, 8192), dtype=jnp.bfloat16)
B = jax.random.normal(jax.random.PRNGKey(1), (8192, 4096), dtype=jnp.bfloat16)

C = my_matmul(A, B)  # XLA compiles → MXU executes 4096×8192×4096 GEMM in bf16
# Compute: 2 * 4096 * 8192 * 4096 ≈ 274 GFLOP
# At 393 TFLOP/s (TPUv5e): ≈ 0.7 ms (ideally)
print(C.shape, C.dtype)  # (4096, 4096) bfloat16
```

> **Practical implication:** When you see `jax.jit` or `tf.function` with `experimental_compile=True`, you are submitting to XLA's pipeline. The "first step is slow" problem in JAX training loops is XLA compilation. The payoff is that every subsequent step runs the fully-optimized, statically-scheduled binary — with zero Python overhead.

---

## The Full TPU Chip Architecture

Beyond the MXU, a TPU chip contains several functional units that handle the ops a neural network needs beyond matmul:

<div class="diagram">
<div class="diagram-title">TPU Chip Architecture (v3/v4 style)</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">⚙️</div>
    <div class="card-title">MXU (Matrix Multiply Unit)</div>
    <div class="card-desc">128×128 or 256×256 systolic array; bf16 in → FP32 accum; ~2 TFLOP/s/MXU</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">📐</div>
    <div class="card-title">Vector Processing Unit (VPU)</div>
    <div class="card-desc">128-lane SIMD; element-wise ops: softmax, GELU, layer norm, ReLU, reductions</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🔢</div>
    <div class="card-title">Scalar Unit</div>
    <div class="card-desc">Control flow, loop counters, address arithmetic; feeds the MXU scheduler</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">💾</div>
    <div class="card-title">Unified Buffer (SRAM)</div>
    <div class="card-desc">8–16 MB scratchpad; holds activations and MXU output between layers</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">🧠</div>
    <div class="card-title">HBM (High-Bandwidth Memory)</div>
    <div class="card-desc">16–192 GB off-chip; weights + KV-cache + optimizer states</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-icon">🔗</div>
    <div class="card-title">ICI (Inter-Chip Interconnect)</div>
    <div class="card-desc">2D/3D torus links; all-reduce without CPU; hundreds to thousands of chips</div>
  </div>
</div>
</div>

The **Unified Buffer** is the scratchpad SRAM that XLA tiles into. A typical workflow for one Transformer layer:

1. DMA loads a tile of weight matrix from HBM → Unified Buffer (paying HBM bandwidth).
2. MXU consumes the tile × activation tile → partial sum in accumulator register.
3. After all K-dimension tiles, partial sums drain to Unified Buffer.
4. VPU applies layer norm + GELU in-place on the Unified Buffer (element-wise, cheap).
5. Next layer's weights arrive from HBM while VPU is running (overlapped DMA).

This double-buffering hides HBM latency behind MXU compute — critical to sustaining near-peak TFLOP/s.

---

## Systolic Arrays Everywhere

While this chapter focuses on TPUs, the systolic concept appears throughout modern AI hardware:

- **NVIDIA Tensor Cores** (Volta onward): Each tensor core performs a 4×4 (Volta) or 8×8 (Hopper) warp-level GEMM per clock. The tiles are small, but multiple tensor cores per SM operate simultaneously, and the data movement is managed by the register file — a partial-systolic design. See [Chapter 15 — GPUs](./15_gpus.md).
- **Apple Neural Engine** (ANE): Contains a systolic-style MAC array optimized for INT8 and FP16 inference. The M4 ANE delivers ~38 TOPS.
- **Qualcomm Hexagon HTP:** DSP-derived, includes a 4×4 or 8×8 MAC array for INT8/INT4 inference.
- **Renesas DRP-AI:** Dataflow-array (systolic-inspired) for embedded vision inference; see [Chapter 17](./17_npus_and_edge_ai.md).
- **Groq LPU:** An extreme-systolic design where the entire model is compiled into a single deterministic instruction stream flowing through a massive 2D array.

The systolic array's efficiency at dense GEMM is so high that almost every purpose-built AI accelerator includes one. The differences are in size, supported dtypes, surrounding memory hierarchy, and how the compiler fills it.

---

## Why This Matters for Model Optimization

Understanding TPU architecture converts several mysterious constraints into clear engineering decisions:

**1. Pad your shapes to multiples of 128 (or 256).** The MXU tiles in 128-element increments. A linear layer with output dimension 4097 wastes 127/128 of the last MXU tile. In practice: always choose hidden sizes divisible by 128. `nn.Linear(8192, 8192)` is ideal; `nn.Linear(8192, 8193)` wastes ≈1% of cycles.

**2. bf16 is the native dtype for TPU training.** Using FP32 halves your MXU utilization (each cell does one FP32 MAC vs two bf16 MACs). On TPUs, always train in bf16 unless there is a specific reason to use FP32 (e.g., loss scaling tests).

**3. Static shapes reduce recompilation.** Dynamic shapes (variable sequence length, variable batch size) trigger XLA recompilation. Bucketing your data (e.g., pad all sequences to the nearest power of 2) avoids recompile storms and keeps the XLA cache hot.

**4. The MXU roofline is different from a GPU.** A TPU's MXU has very high peak TFLOP/s but lower memory bandwidth relative to that peak compared to a GPU. The arithmetic-intensity threshold for "compute bound" on a TPUv5e is roughly 393 TFLOP/s ÷ 1,600 GB/s ≈ 246 FLOP/byte. Small batch sizes (e.g., batch=1 autoregressive decode) have low arithmetic intensity and will leave the MXU nearly idle. **Large batch training is where TPUs shine; single-stream inference is often bandwidth-limited.**

**5. Data parallelism across ICI is cheap.** Gradient all-reduces across the ICI are done in hardware (via dedicated ring-reduce logic) at full ICI bandwidth, without touching HBM. This means you can scale data-parallel training to thousands of chips with near-linear throughput scaling — something that is much harder on GPU clusters where all-reduce saturates the NVLink/Infiniband fabric.

---

## Key Takeaways

- A **systolic array** is a 2-D grid of MAC cells where weights are stationary and activations stream through; it achieves O(N²) compute with O(N) memory I/O, yielding extremely high arithmetic intensity for dense GEMM.
- **bf16** (8-bit exponent, 7-bit mantissa) was invented by Google to match FP32's dynamic range while halving memory and compute cost; it is now ubiquitous across all AI hardware.
- **TPUv1** (2016) was an INT8 inference accelerator; **TPUv2/v3** added bf16 training + HBM; **TPUv4** added 3D ICI; **Trillium (v6)** reaches ~1.5 TFLOP/s per chip with 192 GB HBM.
- The **MXU** is the TPU's systolic core; scalar and vector units handle control and non-GEMM operations; HBM feeds weights; the Unified Buffer (SRAM scratchpad) orchestrates data flow.
- **XLA compilation** requires static shapes but eliminates per-op overhead, fuses entire graphs, and generates fully-pipelined MXU schedules — the reason JAX's first step is slow but subsequent steps are fast.
- The **ICI torus** (2D/3D, sub-µs, Tb/s) allows gradient all-reduce without CPUs or switches, enabling near-linear scaling to thousands of chips in a TPU Pod.
- Systolic arrays appear inside **GPU tensor cores, Apple's ANE, Qualcomm Hexagon, and most NPUs** — the TPU made the idea famous, but the concept now underlies the whole AI-hardware industry.

---

*Next: [Chapter 17 — NPUs & Edge AI Accelerators](./17_npus_and_edge_ai.md), where we leave the data center and look at how AI inference is squeezed into a smartphone, a microcontroller, and an always-on sensor at milliwatt power budgets.*

[← Back to Table of Contents](./README.md)
