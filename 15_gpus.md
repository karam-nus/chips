---
title: "Chapter 15 — GPUs"
---

[← Back to Table of Contents](./README.md)

# Chapter 15 — GPUs

If the CPU is a sprinter — everything engineered around finishing one task as fast as possible — then the GPU is a conveyor belt: wide, relentless, and optimized to push enormous volumes of identical work through at maximum throughput. The GPU was designed to rasterize triangles in parallel for computer graphics; decades later, it turned out that training neural networks is structurally identical to that workload: enormous matrices, millions of multiply-accumulate operations, all independent of each other and all begging to be done simultaneously. The result is the most important piece of hardware in the history of AI.

This chapter explains mechanically *why* GPUs dominate ML. We will build the understanding bottom-up: from how threads are organized into warps and blocks, through the memory hierarchy from registers to HBM, to Tensor Cores and the roofline that governs when you are compute-bound versus memory-bound. By the end, you will be able to reason from first principles about occupancy, kernel efficiency, dtype impact, and what "3.35 TB/s of HBM3 bandwidth" actually means for your model.

> **The one-sentence version:** A GPU puts thousands of simple cores on one die, hides the latency of each with tens of thousands of concurrent threads, and adds dedicated matrix-multiply hardware (Tensor Cores) that does the bulk of neural network math — making it the dominant platform for deep learning training and inference.

---

## The GPU Design Philosophy: Throughput Over Latency

The contrast with [Chapter 14's CPU](./14_cpus.md) is the starting point:

<div class="diagram">
<div class="diagram-title">CPU vs GPU: How They Hide Latency</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">CPU: latency hiding via smart control</div>
    <ul>
      <li>Few (4–128) powerful OoO cores</li>
      <li>Large caches (MB per core)</li>
      <li>Branch prediction, speculation</li>
      <li>Deep pipeline, high clock (4–5 GHz)</li>
      <li>When a thread stalls: OoO finds other µops in <em>same</em> thread</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">GPU: latency hiding via massive multithreading</div>
    <ul>
      <li>Thousands of simple cores (CUDA cores)</li>
      <li>Tiny per-thread caches; large shared on-chip SRAM</li>
      <li>No branch prediction per thread; no OoO</li>
      <li>Shallower pipeline, lower clock (1–2 GHz)</li>
      <li>When a warp stalls: <em>switch to another ready warp instantly</em></li>
    </ul>
  </div>
</div>
</div>

The GPU's answer to latency is not to eliminate it — it is to **have so many threads in flight that there is always something ready to execute**. A memory access that stalls a warp for 500+ cycles is hidden if 400+ cycles worth of other warps are available to run in the meantime.

This design makes GPUs perfect for workloads that are:
1. **Data-parallel:** the same operation applied to millions of independent data elements.
2. **Arithmetic-intensive:** many FLOPs per byte fetched (high arithmetic intensity).
3. **Regular:** no complex branching or data-dependent control flow.

Matrix multiplication — the kernel operation of every neural network — satisfies all three properties perfectly. We will quantify this at the end of the chapter.

---

## The Architecture Hierarchy: From Die to Thread

A GPU is structured in a strict three-level hierarchy: **device → Streaming Multiprocessors (SMs) → CUDA cores**. Software maps to it as **grid → blocks → threads**.

<div class="diagram">
<div class="diagram-title">GPU Hierarchy: Hardware and Software Views</div>
<div class="flow-h">
  <div class="flow-node accent">GPU Device<br/><small>(e.g. H100)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">132 SMs<br/><small>Streaming Multiprocessors</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">4 warp schedulers/SM<br/><small>64 warps/SM resident</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">1 warp = 32 threads<br/><small>execute in SIMT lockstep</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange">1 CUDA core<br/><small>1 FP32 FMA per cycle</small></div>
</div>
</div>

### The Streaming Multiprocessor (SM)

The SM is the fundamental compute unit of a GPU — analogous to a CPU core, but far simpler and present in vast numbers. On the NVIDIA H100 SXM5 (Hopper architecture):

<div class="diagram">
<div class="diagram-title">H100 SM: Internal Components</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">⚙️</div>
    <div class="card-title">4 Warp Schedulers</div>
    <div class="card-desc">Each issues 1 instruction per cycle to one warp. Up to 4 warps can issue simultaneously from 64 resident warps.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🔢</div>
    <div class="card-title">128 FP32 CUDA Cores</div>
    <div class="card-desc">4 sets of 32 cores, one per warp scheduler partition. Each does one FP32 FMA (2 FLOPS) per cycle.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🧮</div>
    <div class="card-title">4 Tensor Core Units</div>
    <div class="card-desc">4th-gen Tensor Cores, each performing 512 dense FP16/BF16 FMAs per clock (1,024 for FP8). Together ~15× the throughput of the SM's FP32 CUDA cores.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">📦</div>
    <div class="card-title">256 KB Register File</div>
    <div class="card-desc">65,536 32-bit registers shared by up to 2,048 resident threads. A thread can use up to 255; at full occupancy each gets 32 (32 × 2,048 = 65,536).</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">💾</div>
    <div class="card-title">228 KB L1 / Shared Mem</div>
    <div class="card-desc">On-chip SRAM: software-managed "shared memory" (up to 228 KB) carved from the same array as the L1 data cache. Delivers ~128 bytes per clock per SM (~31 TB/s across the chip) at ~30-cycle latency.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-icon">🔀</div>
    <div class="card-title">Special Function Units</div>
    <div class="card-desc">16 SFUs for sin/cos/rcp/sqrt/log — slow (4+ cycles each) but present. Plus load/store units, INT32 ALUs.</div>
  </div>
</div>
</div>

**H100 SXM5** (132 of the full GH100 die's 144 SMs enabled), at the 1.83 GHz clock NVIDIA uses for its tensor figures:
- CUDA core FP32 throughput: 132 × 128 cores × 2 FLOPS × 1.83 GHz = **~62 TFLOPS FP32** (67 at the 1.98 GHz maximum boost)
- Tensor Core FP16/BF16 throughput: **~989 TFLOPS dense** (~1,979 with 2:4 sparsity)
- Tensor Core FP8 throughput: **~1,979 TFLOPS dense** (~3,958 with 2:4 sparsity)

The >16× gap between CUDA core and Tensor Core throughput is the reason that any ML kernel worth running must hit Tensor Cores.

---

<div class="die-wrap">
<div class="die-frame">
<div class="hbm-col"><div class="hbm">HBM3</div><div class="hbm">HBM3</div><div class="hbm">HBM3</div></div>
<div class="die" style="grid-template-columns:repeat(4,1fr);grid-template-areas:'g1 g2 g3 g4' 'la la lb lb' 'g5 g6 g7 g8' 'nv nv pc pc';max-width:560px">
<div class="blk compute" style="grid-area:g1">GPC<small>18 SMs</small></div><div class="blk compute" style="grid-area:g2">GPC<small>18 SMs</small></div><div class="blk compute" style="grid-area:g3">GPC<small>18 SMs</small></div><div class="blk compute" style="grid-area:g4">GPC<small>18 SMs</small></div>
<div class="blk cache" style="grid-area:la">L2 partition<small>30 MB</small></div><div class="blk cache" style="grid-area:lb">L2 partition<small>30 MB</small></div>
<div class="blk compute" style="grid-area:g5">GPC<small>18 SMs</small></div><div class="blk compute" style="grid-area:g6">GPC<small>18 SMs</small></div><div class="blk compute" style="grid-area:g7">GPC<small>18 SMs</small></div><div class="blk compute" style="grid-area:g8">GPC<small>18 SMs</small></div>
<div class="blk io" style="grid-area:nv">NVLink 4 · 18 links</div><div class="blk io" style="grid-area:pc">PCIe 5.0 · media · copy engines</div>
</div>
<div class="hbm-col"><div class="hbm">HBM3</div><div class="hbm">HBM3</div><div class="hbm">spare site</div></div>
</div>
<div class="die-legend"><span style="--k:rgba(232,145,106,.7)">SMs (4 Tensor Cores each)</span><span style="--k:rgba(59,130,246,.7)">L2</span><span style="--k:rgba(148,163,184,.7)">I/O</span><span style="--k:#5d4790">HBM stacks</span></div>
<div class="die-caption">GH100, stylised (not to scale): 8 GPCs × 9 TPCs × 2 SMs = 144 SMs and 60 MB of L2 on the full ~814 mm² die. The H100 SXM product enables 132 SMs, 50 MB of L2 and five of the six HBM sites; disabling spares is how a die this large still yields (<a href="./appendix_a_how_chips_are_made.html">Appendix A</a>).</div>
</div>

<div class="datasheet">
<div class="ds-head"><span class="ds-part">GH100</span><span class="ds-name">NVIDIA H100 SXM5 Tensor Core GPU</span><span class="ds-stamp ok">shipping since 2022</span><span class="ds-sub">TSMC 4N · 80 B transistors · 814 mm² · 700 W</span></div>
<div class="ds-grid">
<table><thead><tr><th colspan="2">Compute (dense)</th></tr></thead><tbody>
<tr><td>SMs / Tensor Cores</td><td class="v">132 / 528</td></tr>
<tr><td>FP64 (tensor)</td><td class="v">67 TFLOPS</td></tr>
<tr><td>TF32</td><td class="v">494 TFLOPS</td></tr>
<tr><td>BF16 / FP16</td><td class="v">989 TFLOPS</td></tr>
<tr><td>FP8 / INT8</td><td class="v">1,979 TFLOPS / TOPS</td></tr>
</tbody></table>
<table><thead><tr><th colspan="2">Memory and I/O</th></tr></thead><tbody>
<tr><td>HBM3</td><td class="v">80 GB · 3.35 TB/s</td></tr>
<tr><td>L2 cache</td><td class="v">50 MB</td></tr>
<tr><td>Shared memory / SM</td><td class="v">up to 228 KB</td></tr>
<tr><td>NVLink 4</td><td class="v">900 GB/s</td></tr>
<tr><td>Ridge point (BF16)</td><td class="v">~295 FLOP/B</td></tr>
</tbody></table>
</div>
<div class="ds-foot">Sparse (2:4) rates are 2× the dense figures. Source: <a href="https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper">NVIDIA H100 architecture whitepaper</a>; compare every generation in the <a href="./atlas.html">Accelerator Atlas</a>.</div>
</div>

---

## Warps and SIMT: How 32 Threads Become One

When the GPU executes code, it groups 32 consecutive threads into a **warp**. All 32 threads in a warp execute **the same instruction simultaneously**, but each operates on its own data — this is called **SIMT: Single Instruction, Multiple Threads**.

SIMT differs from CPU SIMD:
- **SIMD (CPU):** one instruction explicitly operates on a vector of data (e.g., `VMULPS ymm0, ymm1, ymm2` multiplies 8 float32 pairs).
- **SIMT (GPU):** 32 threads write *scalar* code (`float val = a[tid] * b[tid]`) but the GPU hardware executes them all on the same cycle with the same instruction, each on a different thread's `tid`.

From the programmer's view, each thread is independent; from the hardware's view, 32 threads are one warp executing in lockstep.

```python
# Conceptual: this Python loop...
results = [a[i] * b[i] for i in range(32)]

# ...becomes, in CUDA C (launched as 1 block of 32 threads):
# __global__ void mul_kernel(float* a, float* b, float* out) {
#     int tid = threadIdx.x;    // 0..31
#     out[tid] = a[tid] * b[tid];   // ALL 32 run this simultaneously
# }
# The 32 threads form 1 warp; the warp scheduler issues 1 mul instruction
# to 32 FP32 units simultaneously.
```

### Warp Scheduling: The Latency-Hiding Machine

Each SM holds up to **64 resident warps** (2048 threads / 32 threads per warp = 64 warps). Each warp is in one of several states:
- **Eligible:** all operands ready, can be issued.
- **Stalled:** waiting for a memory load, a long-latency operation, or a synchronization barrier.

The **warp scheduler** checks which warps are eligible each cycle and issues instructions from them. This is **zero-cost context switching**: switching between warps costs nothing because each warp's register state is permanently stored in the register file — there is no save/restore.

The key insight: if an SM has 64 warps resident and one warp stalls on a ~600-cycle HBM access, **63 other warps can keep the execution units busy** during those cycles. Latency is not eliminated; it is **pipelined away** by overlapping it with computation from other warps.

<div class="diagram">
<div class="diagram-title">Warp Scheduling: Latency Hidden by Multiplexing</div>
<div class="flow">
  <div class="flow-node accent wide">Cycle 1–5: Warp 0 executes arithmetic instructions</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node red wide">Cycle 6: Warp 0 issues a global memory load → STALL (300 cycle wait)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Cycles 7–306: Warp 1, 2, …, 63 execute their arithmetic — keeping SMs busy</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">Cycle 307: Warp 0's data arrives → Warp 0 becomes eligible again</div>
</div>
</div>

### Occupancy

**Occupancy** = (active warps on SM) / (maximum warps on SM). Higher occupancy gives the scheduler more warps to hide latency with. Maximum warps per SM = 64 (H100). If your kernel uses many registers per thread, fewer warps fit simultaneously.

$$\text{Warps per SM} = \min\!\left(\frac{\text{Register file size}}{\text{Registers per thread} \times 32}, 64\right) = \min\!\left(\frac{65536}{\text{regs\_per\_thread} \times 32}, 64\right)$$

For H100: 256 KB = 65536 × 4-byte registers. If a kernel uses 128 registers per thread: 65536 / (128 × 32) = 16 warps → **25% occupancy**. At 32 registers/thread: 64 warps → **100% occupancy**.

```python
# PyTorch profiler can measure SM occupancy indirectly.
# For direct inspection, use NVIDIA NSight Compute (ncu):
# ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
#     python your_script.py

# In PyTorch, you control register pressure via:
# 1. Kernel fusion (torch.compile / Triton) to avoid register spills
# 2. Reducing intermediate tensor sizes
# 3. Using half-precision (fewer bits stored in registers for the same count)
```

### Warp Divergence

SIMT requires all 32 threads to execute the same instruction. What happens when threads take different branches?

```python
# This innocent Python is bad news in a GPU kernel:
if x[tid] > 0:
    result = compute_positive(x[tid])   # some threads go here
else:
    result = compute_negative(x[tid])   # others go here
```

The GPU handles divergence by **serializing the branches**: it first executes the `if` path with threads where the condition is true (others masked off), then executes the `else` path with the remaining threads. The effective throughput is halved at minimum. Worse divergence (32 unique paths) serializes 32 times.

**Implication for ML:** this is why activation functions like ReLU (a branch on `x > 0`) are implemented as masked operations not branches in GPU kernels. It is also why attention masks and variable-length sequences require careful padding/packing to avoid divergence-heavy code.

---

## The GPU Memory Hierarchy: From Registers to HBM

A GPU has five distinct levels of memory, spanning orders of magnitude in size, speed, and scope:

<div class="diagram">
<div class="diagram-title">H100 GPU Memory Hierarchy (per SM → device)</div>
<div class="layer-stack">
  <div class="layer"><strong>Registers</strong> — 256 KB per SM (65,536 × 32-bit), operands every cycle, private per thread; the highest-bandwidth storage on the chip</div>
  <div class="layer"><strong>Shared Memory / L1 Cache</strong> — up to 228 KB per SM, ~30 cycles, shared within a thread block, ~128 B/clock per SM (~31 TB/s chip-wide)</div>
  <div class="layer"><strong>L2 Cache</strong> — 50 MB (two partitions), ~200–300 cycles, shared across all SMs, ~5–10 TB/s</div>
  <div class="layer"><strong>HBM3 (Global Memory)</strong> — 80 GB, ~500–800 cycles, five on-package stacks, <strong>3.35 TB/s</strong></div>
  <div class="layer"><strong>CPU DRAM (via PCIe / NVLink-C2C)</strong> — TBs, microseconds, PCIe 5.0 x16: ~64 GB/s per direction; Grace Hopper NVLink-C2C: 900 GB/s total</div>
</div>
</div>

| Memory Level | H100 Size | Latency | Bandwidth | Scope |
|--------------|-----------|---------|-----------|-------|
| Registers | 256 KB / SM (× 132 SMs = 33 MB) | operands each cycle | ~2.8 TB/s of operand reads per SM | Per thread |
| Shared mem / L1 | 228 KB / SM (software-managed) | ~30 cycles | ~128 B/clock/SM (~31 TB/s total) | Per block |
| L2 cache | 50 MB (whole GPU) | ~200–300 cycles | ~5–10 TB/s | All SMs |
| HBM3 global mem | 80 GB | ~500–800 cycles | **3.35 TB/s** | Whole device |
| Host DRAM (PCIe 5.0 x16) | system RAM | microseconds | ~64 GB/s per direction | Host ↔ device |
| Host DRAM (NVLink-C2C, GH200) | up to 480 GB LPDDR5X | ~1 µs | 900 GB/s total | Host ↔ device, coherent |

Latencies are approximate, from published microbenchmarks; they vary with clock, access pattern and generation.

### HBM: Why It Matters

**HBM (High Bandwidth Memory)** is the reason the GPU can sustain 3.35 TB/s. Unlike DDR5/LPDDR5 DRAM (which communicates over a relatively narrow bus, 64–512 bits wide), HBM uses a **very wide bus (1024 bits per stack × multiple stacks)** and is stacked physically next to the GPU die on the same package interposer. On an H100 SXM5:

- **Five active HBM3 stacks** (the package has six sites), each with a **1,024-bit bus**
- Total bus width: 5 × 1,024 = **5,120 bits**
- At ~5.2 Gbps per pin: 5,120 × 5.23 / 8 ≈ **3.35 TB/s**, the datasheet peak. (HBM3 is specified up to 6.4 Gbps; H100 runs it below that, and H200's six HBM3e stacks reach 4.8 TB/s.)

Compare to a high-end CPU: AMD EPYC Turin at 12 DDR5-6400 channels = ~614 GB/s, roughly **5× less bandwidth**. For memory-bound operations (see the roofline section), this gap is decisive.

We explore HBM in detail in [Chapter 9 — Memory Types](./09_memory_types.md) and the wider bandwidth story in [Chapter 11 — Memory Wall & Bandwidth](./11_memory_wall_and_bandwidth.md).

### Shared Memory: The Key Software Optimization

Shared memory is an **explicitly managed, programmer-controlled SRAM** on each SM. A thread block can use it as a scratchpad — writing data there that all threads in the block share, avoiding repeated slow global memory reads.

The canonical GPU optimization is **tiled matrix multiplication**: instead of each thread independently loading its row and column from global memory (slow, redundant), a block of threads cooperatively loads a tile of the matrix into shared memory, then all threads compute against the fast on-chip copy. This is exactly what cuBLAS and CUTLASS do internally, and what `torch.compile` / Triton generate for fused kernels.

```python
# Conceptual Triton kernel showing shared memory tiling:
# (Triton abstracts CUDA C but the principle is identical)

import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A, B, C,           # pointers to matrices in HBM
    M, N, K,           # matrix dimensions
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 64,
):
    # Each program instance handles one [BLOCK_M, BLOCK_N] output tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # Accumulators in registers (fastest memory):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K chunks — each chunk loaded
    # into shared memory (managed by Triton automatically):
    for k in range(0, K, BLOCK_K):
        a_tile = tl.load(A + ...)   # loads BLOCK_M × BLOCK_K from HBM → smem
        b_tile = tl.load(B + ...)   # loads BLOCK_K × BLOCK_N from HBM → smem
        acc += tl.dot(a_tile, b_tile)   # Tensor Core MMA in registers

    tl.store(C + ..., acc)         # write result tile back to HBM

# The key insight: each element of A and B is loaded from HBM once
# and reused BLOCK_N or BLOCK_M times from shared memory.
# Arithmetic intensity = (2 × M × N × K) / (M×K + K×N bytes) → ~BLOCK_M/2
```

### The Importance of Memory Coalescing

When a warp (32 threads) issues a global memory load, the hardware coalesces accesses into the fewest possible **128-byte transactions**. If thread `tid` loads `float32` at address `base + tid * 4` (stride-1, the 32 addresses are contiguous), all 32 loads are served in **one 128-byte transaction** — peak efficiency. If thread `tid` loads at `base + tid * 256` (stride-64, the addresses are spread out), each load is a separate transaction — **32× more memory traffic**, crippling bandwidth.

This is why row-major layout for matrix multiplications matters, why transposing tensors before a kernel is sometimes beneficial, and why careful attention to memory layout is part of GPU kernel optimization.

---

## Tensor Cores: The ML Workhorse

Tensor Cores are **dedicated fixed-function hardware units** for matrix-multiply-accumulate (MMA) operations. Introduced in Volta (2017), they have grown to provide the overwhelming majority of ML FLOPS on modern NVIDIA GPUs.

### What a Tensor Core Does

A single Tensor Core operation computes:

$$D = A \times B + C$$

where $A$, $B$, $C$, $D$ are small matrix fragments. The key is that **one Tensor Core instruction performs this entire small matmul in a few cycles** — hardware that is far more efficient than routing individual multiply-add operations through the CUDA core FP32 pipeline.

- **mma.sync** (since Volta, per warp): e.g. m16n8k16 for FP16/BF16, i.e. a 16×16 A fragment times a 16×8 B fragment into a 16×8 accumulator held in registers.
- **wgmma** (Hopper, per *warpgroup* of four warps): a 64×N×16 tile for FP16/BF16 (K = 32 for FP8 and INT8), N up to 256, issued asynchronously and reading B (and optionally A) straight from shared memory. Keeping the Tensor Cores fed from shared memory is what lets Hopper reach its peak.
- **tcgen05.mma** (Blackwell): issued by a single thread, accumulating into a dedicated 256 KB-per-SM **Tensor Memory**, with tiles that can span a pair of SMs. Each generation moves more of the matmul's data choreography into hardware.

The net result: **Tensor Core throughput is ~16× higher than FP32 CUDA core throughput** on H100. This is the number that matters for ML. Any kernel that cannot use Tensor Cores is leaving most of the GPU's performance on the table.

### Dtype Support by Generation

Tensor Core capabilities have expanded with each GPU generation:

| GPU Architecture | Launch | FP64 TC | TF32 | FP16 | BF16 | INT8 | INT4 | FP8 (E4M3/E5M2) | FP4 (NV) |
|---|---|---|---|---|---|---|---|---|---|
| Volta (V100) | 2017 | — | — | ✓ | — | — | — | — | — |
| Turing (T4, RTX 20xx) | 2018 | — | — | ✓ | — | ✓ | ✓ | — | — |
| Ampere (A100, A30) | 2020 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |
| Hopper (H100) | 2022 | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ (E4M3 + E5M2) | — |
| Blackwell (B200) | 2024 | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ (NVFP4, MXFP4, MXFP6) |
| Blackwell Ultra (B300) | 2025 | reduced | ✓ | ✓ | ✓ | reduced | — | ✓ | ✓ (1.5× B200's FP4 rate) |
| Rubin (2026) | 2026 | — | ✓ | ✓ | ✓ | — | — | ✓ | ✓ (NVFP4-centric) |

Hopper dropped the INT4 Tensor Core path that Turing and Ampere had; Blackwell replaced it with FP4, and Blackwell Ultra traded away most FP64 and INT8 throughput to spend the area on more FP4. (Rubin's per-format rates beyond FP4 had not been fully published at the time of writing.)

**TF32 (TensorFloat-32):** NVIDIA's hybrid format — 8 exponent bits, 10 mantissa bits, 19 bits total — that fits in a 32-bit storage location but uses only 19 significant bits for computation. Introduced in Ampere, TF32 allows `torch.matmul` on FP32 inputs to automatically use Tensor Cores with minimal accuracy loss. Enabled by default in PyTorch since 1.11.

```python
import torch

# TF32 is enabled by default for matmul and convolution on Ampere+:
print(torch.backends.cuda.matmul.allow_tf32)   # True (default on Ampere+)
print(torch.backends.cudnn.allow_tf32)          # True

# To disable (full FP32 precision, much slower):
torch.backends.cuda.matmul.allow_tf32 = False

# BF16 training — uses Tensor Cores natively on Ampere+:
model = model.to(torch.bfloat16).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Automatic Mixed Precision (AMP) — the standard for Volta/Ampere/Hopper:
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()   # needed for FP16 (not for BF16)

with autocast(dtype=torch.bfloat16):   # matmuls use BF16 Tensor Cores
    output = model(input_batch)
    loss = criterion(output, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**FP8 on H100:** Two FP8 formats are supported — `E4M3` (more precision, max 448: weights and activations in the forward pass) and `E5M2` (more range, max 57,344: gradients). FP8 doubles throughput versus FP16/BF16. The `transformer_engine` library wraps this for PyTorch.

```python
# FP8 inference via Transformer Engine (H100+):
import transformer_engine.pytorch as te

# Replace nn.Linear with te.Linear for automatic FP8 casting:
layer = te.Linear(4096, 4096, bias=False)
with te.fp8_autocast(enabled=True):
    output = layer(input_fp8)
# H100 Tensor Cores execute the GEMM in FP8 → ~2× vs BF16 throughput
```

### Tensor Core Throughput Numbers (H100 SXM5)

| dtype | Dense TFLOPS | Sparse TFLOPS (2:4) |
|-------|-------------|---------------------|
| FP64 | ~67 | — |
| TF32 | ~494 | ~989 |
| FP16 / BF16 | ~989 | ~1,979 |
| FP8 (E4M3/E5M2) | ~1,979 | ~3,958 |
| INT8 | ~1979 | ~3958 |

"Sparse" means the [2:4 structured sparsity](./26_sparsity_hardware_view.md) format: out of every 4 elements, exactly 2 are zero. H100 has a dedicated **Sparse Tensor Core** mode that skips the zero multiplications, doubling throughput for qualifying weight matrices.

---

## The CUDA Execution Model

### Grid → Block → Thread

CUDA's programming model organizes parallel work in a three-level hierarchy:

<div class="diagram">
<div class="diagram-title">CUDA Execution Hierarchy</div>
<div class="flow">
  <div class="flow-node accent wide">Grid — the entire kernel launch<br/><small>all thread blocks for this operation; can be 1D/2D/3D</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Thread Block — scheduled to one SM<br/><small>128–1024 threads; shares shared memory and can __syncthreads()</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">Warp — 32 consecutive threads in a block<br/><small>execute in SIMT lockstep; the unit of scheduling</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Thread — one scalar program instance<br/><small>has private registers; sees its own threadIdx/blockIdx</small></div>
</div>
</div>

A kernel launch: `my_kernel<<<grid_dim, block_dim>>>(args)` specifies the grid shape and block shape. The GPU's work distributor assigns thread blocks to SMs; multiple blocks may reside on one SM simultaneously (up to resource limits). The programmer cannot control which block goes to which SM — this is hardware-managed.

### A Minimal CUDA Kernel

```c
// CUDA C: element-wise ReLU on N float32 values
// -----------------------------------------------
// Shape: input/output are [N] float arrays in HBM

__global__ void relu_kernel(const float* __restrict__ x,
                             float* __restrict__ y,
                             int N) {
    // Each thread handles one element:
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        y[tid] = fmaxf(x[tid], 0.0f);   // ReLU: max(x, 0)
    }
}

// Launch: 1024 threads per block, ceil(N/1024) blocks
// For N = 1048576 (1M elements), this launches 1024 blocks
relu_kernel<<<1024, 1024>>>(d_x, d_y, N);

// Each thread block → scheduled to one SM
// Each SM gets multiple blocks → fills register file with warps
// Each warp of 32 threads → issues one fmaxf instruction simultaneously
// Effective SIMD width: 32 (one warp) × 4 bytes = 128 bytes processed together
```

### The PyTorch View

From PyTorch, you almost never write CUDA kernels directly. PyTorch's operators (`torch.relu`, `torch.matmul`, `torch.nn.Linear`) dispatch to cuDNN/cuBLAS/CUTLASS internally, which contain hand-tuned CUDA kernels. `torch.compile()` (backed by TorchInductor) and Triton allow you to write custom kernels at a higher level of abstraction.

```python
import torch

# All of these dispatch to GPU kernels internally:
x = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
y = torch.relu(x)            # → elementwise kernel (memory-bandwidth bound)
z = x @ x.T                  # → cuBLAS GEMM (Tensor Core, compute-bound)
w = torch.nn.functional.softmax(x, dim=-1)  # → fused softmax kernel

# torch.compile: fuses operations, selects optimal kernels, tiles for L2/shared mem:
model = torch.compile(model)   # Triton-based codegen on CUDA

# Profile GPU kernels:
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    with_flops=True
) as prof:
    output = model(x)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Kernel Launch Overhead

Each CUDA kernel invocation carries **~5–20 µs of overhead** (CPU-side driver validation, GPU work queue submission). For a model with many small operations, kernel launch overhead can dominate. This is why:
- **Operator fusion** (combining multiple elementwise ops into one kernel) is valuable.
- `torch.compile()` fuses operations aggressively (TorchScript's `torch.jit.script` is in maintenance mode).
- Large batch sizes amortize launch overhead across more work.
- **CUDA Graphs** (CUDA 10, 2018 onward, any recent GPU): capture a sequence of kernels as a graph once, then replay it with a single launch (essential for fixed-shape inference loops such as LLM decode).

```python
# CUDA Graphs: eliminate kernel launch overhead for repeated fixed-shape inference:
import torch

# Warmup run:
output = model(static_input)

# Capture graph:
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay (no CPU overhead, no kernel launch, just GPU execution):
static_input.copy_(new_input)   # update the input in-place
g.replay()                      # ~5–50µs vs 1–10ms for un-captured inference
result = static_output.clone()
```

---

## The GPU Roofline: When Are You Compute-Bound?

We introduced the roofline model in [Chapter 11 — Memory Wall & Bandwidth](./11_memory_wall_and_bandwidth.md). On a GPU:

$$\text{Performance (FLOP/s)} = \min\!\left(\text{Peak FLOPS},\ \text{Bandwidth} \times \text{Arithmetic Intensity}\right)$$

$$\text{Arithmetic Intensity (AI)} = \frac{\text{FLOPS per kernel invocation}}{\text{Bytes read from HBM}}$$

**H100 ridge point (BF16 Tensor Core):**

$$\text{AI}_{\text{ridge}} = \frac{989 \times 10^{12}\ \text{FLOPS}}{3.35 \times 10^{12}\ \text{bytes/s}} \approx 295\ \text{FLOP/byte}$$

Operations with AI > 295 FLOP/byte are **compute-bound**; below, **memory-bound**.

| Operation | Typical AI | Regime |
|-----------|-----------|--------|
| Large GEMM (e.g., 4096×4096×4096, FP16) | ~1,365 FLOP/byte | Compute-bound (Tensor Core) |
| Attention (QK^T, full sequence) | ~50–500 FLOP/byte (grows with seq len) | Mixed; Flash Attention → compute-bound |
| LayerNorm / RMSNorm | ~3–5 FLOP/byte | Memory-bound |
| Element-wise activations (ReLU, SiLU) | ~1–2 FLOP/byte | Memory-bound |
| LLM decode (batch=1, one token) | ~1–2 FLOP/byte | Memory-bound |
| LLM decode (batch=64, one token) | ~64 FLOP/byte (BF16 weights) | Still memory-bound; crosses the ridge near batch ≈ 300 |

**The decode vs prefill distinction for LLMs:** during prefill (processing a prompt of length L in parallel), the matmuls are [d_model × L] — large, high AI, compute-bound. During autoregressive decode (generating one token at a time with batch=1), each matmul is [d_model × 1] — equivalent to a matrix-vector product, AI ~= 1 FLOP/byte, entirely memory-bandwidth-bound. This is why:
1. Prefill is fast (Tensor Core saturated).
2. Decode is slow unless batched (bandwidth saturated at tiny batch, not Tensor Cores).
3. Continuous batching (batching many decode requests together) is the key serving optimization.

```python
# Roofline calculation for a GPU:
peak_tflops_bf16 = 989e12      # H100 SXM5 BF16 Tensor Core (dense)
hbm_bandwidth    = 3.35e12     # H100 HBM3 bandwidth (bytes/sec)

ridge_point = peak_tflops_bf16 / hbm_bandwidth   # ~295 FLOP/byte

# GEMM arithmetic intensity:
M, N, K = 4096, 4096, 4096
flops = 2 * M * N * K                            # ~134 GFLOPS
bytes_read = 2 * (M*K + K*N) * 2                # 2 matrices × BF16 (2B each)
ai_gemm = flops / bytes_read                     # ~2730 FLOP/byte

# Decode arithmetic intensity (batch=1, one linear layer):
d_model, d_ff = 4096, 16384
flops_decode = 2 * 1 * d_model * d_ff            # batch=1 matmul
bytes_decode  = d_model * d_ff * 2               # weight matrix in BF16
ai_decode     = flops_decode / bytes_decode      # = 2 FLOP/byte

print(f"Ridge point:    {ridge_point:.0f} FLOP/byte")
print(f"GEMM AI:        {ai_gemm:.0f} FLOP/byte  -> compute-bound")
print(f"Decode AI (b=1): {ai_decode:.1f} FLOP/byte -> memory-bound by {ridge_point/ai_decode:.0f}×")
```

---

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Blackwell Ultra gave up most of its FP64 and INT8 throughput to make room for more FP4. What does that bet say about which workloads NVIDIA expects to dominate, and what happens to customers whose models do not quantize well to 4 bits?</p>
</div>

## Consumer vs Datacenter GPUs

Not all GPUs are equal. The divide between consumer (gaming) and datacenter cards is wide:

| Feature | Consumer (RTX 5090, 2025) | Datacenter (H100 SXM) | Datacenter (B200) | Why It Matters |
|---------|---------------------|------------------------|------|----------------|
| Process | TSMC 4N | TSMC 4N | TSMC 4NP, two dies | Same node family |
| Memory | 32 GB GDDR7 | 80 GB HBM3 | 180–192 GB HBM3e | Model size limit |
| Mem BW | 1,792 GB/s | 3,350 GB/s | 8,000 GB/s | Decode speed |
| BF16 tensor TFLOPS (dense) | ~419 (FP16 accumulate) / ~209 (FP32) | ~989 | ~2,250 | Training and prefill |
| FP8 tensor TFLOPS (dense) | ~838 / ~419 | ~1,979 | ~4,500 | |
| GPU-to-GPU link | None (PCIe only) | NVLink 4, 900 GB/s | NVLink 5, 1.8 TB/s | Multi-GPU training |
| ECC on memory | Optional (GDDR7) | Always-on | Always-on | Long-run integrity |
| Power | 575 W | 700 W | 1,000 W | Cooling and density |
| Price (approx.) | ~$2,000 | ~$25,000–30,000 | ~$35,000–45,000 | Cost per FLOP vs cost per GB |
| Form factor | PCIe card | SXM module on HGX board | SXM / GB200 superchip | NVLink and power delivery |

The SXM form factor is a mezzanine module that plugs into an HGX baseboard, which carries the NVLink and NVSwitch wiring between GPUs and delivers 700–1,200 W per module. The connection to the host CPU is still PCIe (or NVLink-C2C on Grace superchips). Consumer PCIe GPUs have no NVLink since the RTX 40 series, so multi-GPU traffic goes over PCIe at ~32–64 GB/s per direction.

For training large models (7B+), the combination of ECC memory, NVLink, large VRAM, and sustained Tensor Core throughput makes datacenter GPUs a qualitatively different product, not just a faster consumer card.

---

## A Brief Generational Note: Volta → Ampere → Hopper → Blackwell

GPU generations follow a consistent pattern of expanding Tensor Core capabilities, HBM bandwidth, and SM count. We defer detailed generation-by-generation specs to [Chapter 34 — Generational Improvements](./34_generational_improvements.md), but the key milestones:

| Architecture | Year | Key ML Addition |
|---|---|---|
| Volta (V100) | 2017 | First Tensor Cores (FP16 only); NVLink 2 (300 GB/s); HBM2 900 GB/s |
| Turing (T4, RTX 20xx) | 2018 | INT8 + INT4 Tensor Cores; RT Cores for ray tracing; GDDR6 |
| Ampere (A100) | 2020 | BF16 + TF32 Tensor Cores; sparsity (2:4); HBM2e 2 TB/s; NVLink 3 |
| Hopper (H100) | 2022 | FP8 Tensor Cores; Transformer Engine; 4th-gen TC; HBM3 3.35 TB/s; NVLink 4 |
| Blackwell (B200) | 2024 | FP4 Tensor Cores (NVFP4); 5th-gen TC; HBM3e 8 TB/s; NVLink 5; GB200 NVL72 |
| Blackwell Ultra (B300) | 2025 | 1.5× FP4 throughput; 288 GB HBM3e; GB300 NVL72 |
| Rubin (VR200) | 2026 | HBM4 (~20+ TB/s); NVLink 6 (3.6 TB/s); Vera CPU; NVL72 racks; paired with Groq-derived LPX decode racks |

At BF16, each generation has roughly doubled to tripled tensor throughput while bandwidth grew 1.6–2.4×, so the BF16 ridge point stayed between ~140 and ~300 FLOP/byte. The real shift comes from new low-precision formats: each one doubles math per byte without making memory faster, so measured in each chip's fastest format the ridge has climbed from ~140 (V100) to ~1,900 (B300 FP4). See [Chapter 34](./34_generational_improvements.md) and the [Accelerator Atlas](./atlas.md).

---

## Why This Matters for Model Optimization

The GPU's architecture directly shapes every major optimization you apply to a neural network. Here is the mechanical connection:

<div class="diagram">
<div class="diagram-title">GPU Architecture → Your Optimization Decisions</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🔢</div>
    <div class="card-title">Use BF16 / FP8</div>
    <div class="card-desc">Tensor Cores only activate for supported dtypes. FP32 on Ampere+ falls back to TF32 automatically; explicit BF16 gets full Tensor Core throughput. FP8 doubles it again.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">📦</div>
    <div class="card-title">Large Batch Sizes</div>
    <div class="card-desc">At batch=1, matmuls are matrix-vector products (AI ≈ 1 FLOP/byte, memory-bound). At batch=128+, AI rises to compute-bound territory. Batching is the primary way to utilize Tensor Cores.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🔗</div>
    <div class="card-title">Kernel Fusion</div>
    <div class="card-desc">Memory-bound elementwise ops (LayerNorm, activation, dropout) become free when fused with adjacent Tensor Core kernels — they run from shared memory, not HBM. Flash Attention does this for attention.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">🗜️</div>
    <div class="card-title">Quantization (INT8/INT4)</div>
    <div class="card-desc">FP8 and INT8 Tensor Cores give 2× the BF16 throughput; FP4 on Blackwell gives 4×. More importantly, quantization shrinks the model → fits in VRAM, and cuts the HBM bytes per token → faster memory-bound decode.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">⬡</div>
    <div class="card-title">2:4 Structured Sparsity</div>
    <div class="card-desc">H100's Sparse Tensor Core skips zero multiplies in 2:4 weight matrices → 2× throughput for weights that satisfy the sparsity constraint. Covered in detail in ./26_sparsity_hardware_view.md.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-icon">💾</div>
    <div class="card-title">VRAM Budget</div>
    <div class="card-desc">80 GB HBM3 on H100 = model weights + activations + optimizer state. A 70B FP16 model is 140 GB → won't fit. Quantization to INT4 = 35 GB → fits. This is why quantization enables deployment, not just speed.</div>
  </div>
</div>
</div>

### Connecting to the Larger Story

The GPU's story does not end here. [Chapter 27 — Why GPUs Won AI](./27_why_gpus_won_ai.md) tells the *why-it-won* narrative: how the structural match between neural network math and GPU architecture explains the deep learning revolution, and why the alternatives (CPUs, FPGAs) fell behind. [Chapter 28 — Multi-GPU & Distributed Training](./28_multi_gpu_and_distributed_training.md) extends the picture to NVLink fabrics, tensor/pipeline/expert parallelism, and why even 80 GB of HBM is not enough for large model training.

---

## Key Takeaways

- A GPU hides memory latency through **massive multithreading**: thousands of warps reside on-chip simultaneously, and the warp scheduler switches to eligible warps the instant another stalls — zero context-switch cost.
- The **Streaming Multiprocessor (SM)** is the fundamental compute unit: H100 has 132 SMs, each with 128 CUDA cores, 4 Tensor Core units, 256 KB register file, and 228 KB shared memory / L1. All SMs share 50 MB L2 and 80 GB HBM3 at 3.35 TB/s.
- **Warps** (32 threads in SIMT lockstep) are the hardware's unit of scheduling; **warp divergence** (threads taking different branches) serializes execution and must be avoided in hot kernels.
- **Tensor Cores** are dedicated MMA hardware: each H100 Tensor Core does 512 dense BF16 FMAs per clock, giving ~15× the FP32 CUDA-core throughput. They support FP16, BF16, TF32, INT8 and FP8 (Hopper), and FP4/FP6 (Blackwell).
- The **GPU memory hierarchy** (registers → shared memory/L1 → L2 → HBM → host) spans ~30 to ~800 cycles of latency and three orders of magnitude of bandwidth, and decides which operations are compute-bound vs memory-bound. The H100's BF16 ridge point is ~295 FLOP/byte.
- **LLM decode is memory-bandwidth-bound at small batch sizes** (AI ~1–2 FLOP/byte); Tensor Cores sit idle. Larger batches or quantization (reducing bytes per token) are the levers.
- **Kernel fusion, BF16/FP8 dtypes, large batch sizes, and quantization** are not magic — they are direct consequences of the GPU's architecture: Tensor Core dtype requirements, the roofline ridge, the warp multithreading model, and the VRAM capacity limit.

## Check Yourself

<details class="check">
<summary>How many warps can an H100 SM keep resident if a kernel uses 96 registers per thread?</summary>
<p>65,536 / (96 × 32) = 21.3, rounded down to the allocation granularity: about 20 warps, i.e. ~31 % occupancy (other limits such as shared memory may cut it further).</p>
</details>

<details class="check">
<summary>Why does the 64×N×16 shape of wgmma matter for writing fast Hopper kernels?</summary>
<p>A warpgroup must hand the Tensor Cores a 64-row tile and stream B from shared memory asynchronously; kernels that use the older per-warp mma.sync or tiles that are not multiples of the instruction shape leave a large fraction of Hopper's tensor throughput unused.</p>
</details>

<details class="check">
<summary>A kernel achieves 2.9 TB/s and 40 TFLOPS on an H100. Which roof is it under and what would you try?</summary>
<p>Its intensity is 40 / 2.9 ≈ 14 FLOP/byte, far below the 295 ridge, and 2.9 TB/s is ~87 % of the 3.35 TB/s peak: it is memory-bound and near the limit. Only moving fewer bytes helps: fuse it with neighbours, use a smaller datatype, or restructure for reuse.</p>
</details>

<details class="check">
<summary>Why did H100 run its HBM3 below the 6.4 Gbps the standard allows?</summary>
<p>Power, signal integrity and supply: running thousands of interposer wires faster costs energy and margin, and the first HBM3 parts were binned for speed. NVIDIA chose 5 stacks at ~5.2 Gbps (3.35 TB/s); H200 later raised it with HBM3e.</p>
</details>

<details class="check">
<summary>What is the difference between occupancy and utilization?</summary>
<p>Occupancy is how many warps are resident relative to the maximum; utilization is how busy the execution units actually are. High occupancy helps hide latency but does not guarantee utilization; well-pipelined kernels (e.g. GEMMs using TMA and wgmma) often run at low occupancy and near-peak utilization.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| NVIDIA H100 Tensor Core GPU Architecture | NVIDIA, 2022 | [resources.nvidia.com/en-us-tensor-core/gtc22-…](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper) | SM layout, Tensor Core rates, HBM3 configuration |
| NVIDIA Blackwell Architecture Technical Brief | NVIDIA, 2024 | [resources.nvidia.com/en-us-blackwell-architec…](https://resources.nvidia.com/en-us-blackwell-architecture) | FP4/FP6, second-generation Transformer Engine, NV-HBI |
| CUDA C++ Programming Guide | NVIDIA | [docs.nvidia.com/cuda/cuda-c-programming-guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | Thread hierarchy, occupancy, CUDA Graphs |
| PTX ISA: warpgroup-level matrix instructions (wgmma) and tcgen05 | NVIDIA | [docs.nvidia.com/cuda/parallel-thread-execution](https://docs.nvidia.com/cuda/parallel-thread-execution/) | Tensor Core instruction shapes per generation |
| Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking | Z. Jia et al., 2018 | [arxiv.org/abs/1804.06826](https://arxiv.org/abs/1804.06826) | Measured cache and memory latencies (method used for later generations too) |
| Chips and Cheese: Nvidia's H100 | Chips and Cheese, 2023 | [chipsandcheese.com](https://chipsandcheese.com/) | Measured H100 cache bandwidth and latency |

---

**Next:** [Chapter 16 — TPUs & Systolic Arrays](./16_tpus_and_systolic_arrays.md) — We meet Google's purpose-built matrix-multiply chip and learn how a systolic array achieves even higher arithmetic intensity than a GPU by eliminating the memory hierarchy almost entirely for the innermost computation.

[← Back to Table of Contents](./README.md)
