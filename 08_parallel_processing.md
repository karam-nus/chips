---
title: "Chapter 8 — Parallel Processing: From ILP to SIMT"
---

[← Back to Table of Contents](./README.md)

# Chapter 8 — Parallel Processing: From ILP to SIMT

Every technique you use to make a model faster — larger batches, multi-GPU training, quantized matmuls, flash attention — ultimately rests on one hardware reality: **modern chips are parallel by design, not by accident.** [Chapter 1](./01_what_is_a_chip.md) ended with the death of Dennard scaling around 2005: transistors kept shrinking, but voltage could no longer follow them down, so clock frequencies stalled. The only remaining way to spend extra transistors productively was to make chips do *more things at once*. Parallelism is not an optional feature of modern hardware; it is the mandatory response to a fundamental physics limit.

This chapter maps the entire taxonomy of parallelism, from the finest grain (individual bits) to the coarsest (thousands of GPU threads), and introduces the mathematical laws — Amdahl and Gustafson — that govern how much you actually win when you parallelize. Along the way we will land on **SIMT**, the execution model that makes GPUs the dominant hardware for deep learning, and explain precisely why deep learning maps so well onto it.

> **The one-sentence version:** Because clock frequency stopped scaling, chips gain performance by doing many operations simultaneously — and SIMT, the GPU's "many threads, one instruction stream" model, happens to be a perfect match for the giant batched matrix multiplies that are neural network training and inference.

---

## Why Parallelism — the Physics Recap

Recall from [Chapter 1](./01_what_is_a_chip.md) the CMOS switching energy: $E_{\text{switch}} \approx \tfrac{1}{2}CV_{dd}^2$. When Dennard scaling held (1974–~2005), every new process node reduced $V_{dd}$, keeping power density constant while frequency climbed. Once voltage reduction hit a floor (leakage current dominates below ~0.7 V), power density instead *rose* with frequency (dynamic power $\propto f \cdot CV^2$). Chips would melt if you kept cranking the clock.

The inflection point is visible in any clock-frequency chart: single-core clock speeds plateau around **3–5 GHz** from ~2004 onward. The transistor budget has continued to grow per Moore's Law, but those extra transistors cannot be spent on faster single-core execution — they must be spent on **more parallel execution units**. This is the physical origin of every concept in this chapter.

<div class="diagram">
<div class="diagram-title">Parallelism Levels — Grain Size</div>
<div class="flow">
  <div class="flow-node accent wide">Bit-level parallelism — operate on N-bit words in one step</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Instruction-level parallelism (ILP) — overlap multiple instructions in one core</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">Data-level parallelism (DLP / SIMD) — same operation on multiple data elements</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Thread-level parallelism (TLP) — multiple independent threads, multiple cores</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">SIMT — thousands of lightweight threads sharing an instruction stream (GPU)</div>
</div>
</div>

---

## Level 1 — Bit-Level Parallelism

The simplest form of parallelism is implicit in hardware word width. An ALU built for **64-bit** words computes a 64-bit addition in one clock cycle — the same time a 32-bit ALU would use for 32-bit addition. When you shift from `float32` (32 bits) to `int8` (8 bits), the integer multiplier can potentially pack **4× more** 8-bit multiply-accumulate operations into the same data path using sub-word parallelism (e.g., x86's `VPDPBUSD` instruction, or a tensor core's INT8 mode).

This is why quantization gives throughput gains: not just from smaller memory footprint, but because the same silicon can perform *more operations per cycle* on narrower data. A 256-bit SIMD register holds **8 × fp32**, **16 × fp16**, or **32 × int8** values. The hardware is literally doing more arithmetic per cycle.

---

## Level 2 — Instruction-Level Parallelism (ILP)

### Pipelining

[Chapter 4](./04_anatomy_of_a_processor.md) introduced the fetch-decode-execute cycle. **Pipelining** overlaps those stages across successive instructions: while instruction $n$ is executing, instruction $n+1$ is decoding, and $n+2$ is being fetched. A 5-stage pipeline can — in steady state — complete **one instruction per cycle** even though each instruction takes 5 cycles end-to-end.

```text
Cycle:  1    2    3    4    5    6    7
Instr A: F    D    E    M    W
Instr B:      F    D    E    M    W
Instr C:           F    D    E    M    W
Instr D:                F    D    E    M    W
                                           ↑ 4 instructions done by cycle 8
```
(F=Fetch, D=Decode, E=Execute, M=Memory, W=Writeback)

The pipeline speedup is bounded by **hazards** — data dependencies (instruction $n+1$ needs $n$'s result), structural conflicts (two instructions need the same hardware unit), and control hazards (branches). Real CPUs handle these with stalls, forwarding/bypassing, and branch prediction.

### Superscalar (Multiple Issue)

Pipelining keeps the stages busy but still issues at most one instruction per cycle into the execute stage. A **superscalar** processor issues **multiple instructions per cycle** from a single thread, by replicating execution units — multiple ALUs, multiple FPUs, multiple load/store units — and dispatching to them simultaneously.

Modern high-end CPUs (e.g., AMD Zen 4, Apple M-series) are **4- to 8-way superscalar**: they can issue 4–8 instructions per cycle. The out-of-order engine scans ahead a **reorder buffer (ROB)** window of 300–600 instructions, finds ready-to-execute independent instructions, and dispatches them out of program order.

### Out-of-Order Execution (OoO)

**Out-of-order execution** decouples the program's *nominal* sequence from the *actual* execution sequence. If instruction 10 is stalled waiting for memory, instructions 11–20 that are independent proceed. The hardware maintains correctness by committing (retiring) results in program order via the reorder buffer.

OoO requires **register renaming** (physical registers far outnumber architectural ones, eliminating false dependencies) and a **reservation station** (instructions wait there until operands are ready). A modern CPU core spends roughly half its transistor budget on these OoO structures.

### Branch Prediction and Speculation

Control flow (if/else, loops) breaks pipelining. **Branch predictors** use history tables (2-bit saturating counters, TAGE predictors, perceptron predictors) to guess whether a branch is taken, achieving **>99% accuracy** on regular code. When the guess is wrong, the pipeline is flushed — a **15–25 cycle mispredict penalty** on a deep out-of-order core.

**Speculative execution** extends this: the CPU executes instructions *past* a not-yet-resolved branch, discarding results if it guessed wrong. This is the root of the Spectre/Meltdown vulnerability class, but it is also responsible for a large fraction of modern CPU performance.

### The Limits of ILP

The fundamental barrier to more ILP is the **dependency graph** of the program. Amdahl's Law (below) generalizes this: if 5% of the work is strictly sequential, no amount of parallelism in the remaining 95% can yield more than a 20× speedup. In practice:

- **ILP parallelism** extracted per core has been roughly flat for ~15 years. Modern OoO CPUs squeeze ~3–4 useful instructions per cycle from typical code.
- Going wider (8-way, 16-way superscalar) runs into diminishing returns and area/power costs that exceed the gains.
- The extra transistors are better spent on **more cores** (thread-level parallelism) or **SIMD units** (data-level parallelism).

---

## Level 3 — Data-Level Parallelism: SIMD and Vector

**SIMD** (Single Instruction, Multiple Data) is the hardware version of NumPy vectorization: **one instruction** is broadcast over a **register containing multiple data elements**.

### How SIMD Works

A **256-bit AVX2** register holds:
- 8 × `float32` (each 32 bits)
- 16 × `int16` (each 16 bits)
- 32 × `int8` (each 8 bits)

A single `VADDPS` (vector add single-precision) adds all 8 pairs simultaneously. The hardware contains 8 adders operating in lock-step — but they share the *decode and control* logic, so the overhead per element is far lower than issuing 8 separate instructions.

### SIMD in x86 / ARM

| Extension | Width | FP32 lanes | FP16 lanes | Notes |
|-----------|------:|----------:|----------:|-------|
| SSE2 | 128-bit | 4 | — | Pentium 4 era (2001) |
| AVX/AVX2 | 256-bit | 8 | — | Haswell (2013) |
| AVX-512 | 512-bit | 16 | 32 | Skylake-X (2017), FP16 via VNNI |
| ARM NEON | 128-bit | 4 | 8 | ARMv7+ / all modern ARM |
| ARM SVE2 | 128–2048-bit | variable | variable | Scalable vector extension |

### Python: SIMD Speedup is NumPy vs Python Loops

```python
import numpy as np
import time

N = 10_000_000
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)

# --- Python loop (no SIMD, no parallelism) ---
t0 = time.perf_counter()
c_slow = [a[i] + b[i] for i in range(N)]
t1 = time.perf_counter()
print(f"Python loop: {(t1-t0)*1e3:.1f} ms")    # ~2000 ms on a typical laptop

# --- NumPy (compiles to AVX/AVX-512 SIMD under the hood) ---
t0 = time.perf_counter()
c_fast = a + b                                   # one SIMD-vectorized call
t1 = time.perf_counter()
print(f"NumPy SIMD:  {(t1-t0)*1e3:.1f} ms")     # ~10 ms — ~200× faster
# Speedup ≈ (SIMD width) × (reduced Python overhead)
# With AVX-512: 16 FP32 elements per instruction
```

The lesson: "vectorize your operations" is not just a style tip — it is **switching from serial scalar execution to SIMD hardware parallelism**.

---

## Level 4 — Thread-Level Parallelism: Multicore and SMT

### Multiple Physical Cores

Where SIMD parallelizes a *single* instruction stream over data, **multicore** processors replicate the entire execution pipeline — multiple independent cores, each with its own registers, L1/L2 caches, OoO engine, and branch predictor — on one die.

A modern CPU die might contain **8–192 cores** (e.g., AMD EPYC Genoa: 96 cores @ 5 nm, 400 W TDP). Each core runs its own independent thread or process. True parallelism: two cores can physically execute different instructions at the same cycle.

### SMT / Hyperthreading

**Simultaneous Multithreading (SMT)**, branded by Intel as **Hyperthreading**, runs **two (or more) hardware threads** on a *single* physical core. The idea: one thread may be stalled on a cache miss or a long-latency instruction; while it waits, the OoO engine's spare execution units can execute instructions from the *other* thread.

Hardware cost: ~15–30% die area increase for ~15–40% throughput increase on workloads with independent threads. The two threads *share* the L1/L2 caches and execution units — so they can interfere with each other under high pressure.

For ML inference servers, SMT helps when request handling is I/O-bound; for sustained compute-bound workloads (e.g., PyTorch CPU inference), disabling HT can actually improve throughput by giving each physical thread fuller cache access.

---

## Level 5 — SIMT: The GPU Execution Model

### From SIMD to SIMT

SIMD and SIMT both execute one instruction over many data elements — but they differ in a crucial way:

<div class="diagram">
<div class="diagram-title">SIMD vs SIMT</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">SIMD (CPU)</div>
    <ul>
      <li>Fixed-width vector registers (e.g., 512-bit)</li>
      <li>Programmer/compiler must explicitly vectorize</li>
      <li>All lanes <em>must</em> do the same thing or you waste lanes</li>
      <li>Divergence = masking off lanes (wasted cycles)</li>
      <li>Control flow is the programmer's problem</li>
      <li>~8–32 lanes wide</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">SIMT (GPU)</div>
    <ul>
      <li>Thousands of scalar threads, written as if independent</li>
      <li>Hardware groups threads into <strong>warps/wavefronts</strong></li>
      <li>Threads in a warp execute the <em>same instruction</em> in lock-step</li>
      <li>Divergence: mask inactive threads — same cost model as SIMD</li>
      <li>Threads can have private registers and branch independently</li>
      <li>32 or 64 threads per warp</li>
    </ul>
  </div>
</div>
</div>

**SIMT** (Single Instruction, Multiple Threads) is NVIDIA's term (AMD uses "wavefront" of 64 threads; the concept is the same). You write a GPU **kernel** as if each thread runs independently — it has its own `threadIdx`, its own registers, its own control flow. The hardware then groups consecutive threads into **warps of 32** and executes them in SIMD fashion on the actual hardware lanes.

### Warps, Wavefronts, and the SM

An NVIDIA GPU has **Streaming Multiprocessors (SMs)** (AMD calls them **Compute Units, CUs**). Each SM contains:
- **32 CUDA cores** (FP32 ALUs) in older architectures, **128 in Ampere/Hopper**
- **Warp schedulers** (4 per SM in Hopper) — each schedules a 32-thread warp each cycle
- **Register file** — ~64 KB per SM; each thread gets a slice (e.g., 32–256 registers per thread)
- **Shared memory / L1 cache** — 128–256 KB per SM in Hopper

An H100 SXM5 has **132 SMs** × 128 FP32 CUDA cores = **16,896 FP32 CUDA cores**, clocked at ~1.98 GHz peak. That is ~67 TFLOP/s FP32 — but the tensor cores push **989 TFLOP/s** for FP16/BF16 with sparsity.

### Warp Divergence — the Key SIMT Pitfall

Because all threads in a warp execute the same instruction, a conditional branch creates **divergence**:

```python
# Conceptual CUDA-style pseudocode
# Warp = threads 0..31

if thread_id % 2 == 0:
    result = heavy_computation_A()   # threads 0,2,4,...30 active; 1,3,...31 masked
else:
    result = heavy_computation_B()   # threads 1,3,...31 active; 0,2,...30 masked
# Both branches execute serially — throughput is halved compared to no branch
```

In ML terms: activation functions with data-dependent branches (e.g., `if x > 0` in ReLU) are fine — every element in a batch sees the same branch direction most of the time. But **sparse operations** with data-dependent access patterns can hurt — this is why structured sparsity (2:4) is hardware-friendly while unstructured sparsity is not (see [Chapter 26](./26_sparsity_hardware_view.md)).

---

## Flynn's Taxonomy

A classic classification of computer architectures by the number of instruction and data streams:

<div class="diagram">
<div class="diagram-title">Flynn's Taxonomy (1966)</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-icon">➡️</div>
    <div class="card-title">SISD — Single Instruction, Single Data</div>
    <div class="card-desc">Classic scalar processor. One instruction, one data element per cycle. Uniprocessor CPUs pre-SIMD. Rare in modern hardware.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">⚡</div>
    <div class="card-title">SIMD — Single Instruction, Multiple Data</div>
    <div class="card-desc">One instruction, multiple data elements in lock-step. CPU vector extensions (AVX, NEON), GPU warps. The dominant model for numeric compute.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🔀</div>
    <div class="card-title">MISD — Multiple Instructions, Single Data</div>
    <div class="card-desc">Multiple instructions on the same data stream simultaneously. Fault-tolerant systems (space, avionics). Extremely rare; largely theoretical.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">🌐</div>
    <div class="card-title">MIMD — Multiple Instructions, Multiple Data</div>
    <div class="card-desc">Multiple processors, each with its own instruction stream and data. Multicore CPUs, multi-GPU clusters. The standard for distributed computing.</div>
  </div>
</div>
</div>

GPUs occupy an interesting position: they are **SIMT** (a variant of SIMD at the warp level) but **MIMD** across SMs — different SMs can execute entirely different kernels. A multi-GPU cluster is textbook MIMD.

---

## Amdahl's Law vs Gustafson's Law

The most important question in parallelism: **how much speedup do you actually get from adding more processors/cores/GPUs?**

### Amdahl's Law (1967) — Strong Scaling

**Strong scaling**: fix the problem size, add more processors.

Let $P$ be the fraction of the workload that can be parallelized (the "parallel fraction"), and $1-P$ the fraction that must run serially. With $N$ processors the speedup is:

$$S_{\text{Amdahl}}(N) = \frac{1}{(1-P) + \dfrac{P}{N}}$$

**As $N \to \infty$:** $S_{\infty} = \dfrac{1}{1-P}$. The serial fraction creates an absolute ceiling.

| Serial fraction $(1-P)$ | Max speedup (any $N$) | 8 GPUs | 64 GPUs | 512 GPUs |
|:---:|:---:|:---:|:---:|:---:|
| 50% | 2× | 1.78× | 1.97× | 2.00× |
| 10% | 10× | 4.71× | 8.77× | 9.83× |
| 1% | 100× | 7.75× | 39.0× | 84.3× |
| 0.1% | 1,000× | 7.94× | 60.6× | 338× |

> A workload with even **5% serial overhead** cannot exceed **20× speedup** no matter how many processors you throw at it.

**Worked example — data-parallel training with gradient synchronization.** Suppose AllReduce (the gradient averaging step across GPUs) takes 5% of total iteration time in a 2-GPU run. With $P = 0.95$, adding 64 GPUs yields $S = 1 / (0.05 + 0.95/64) = 1/0.0648 \approx 15.4×$ — not 64×. This is why communication ([Chapter 12](./12_buses_and_interconnects.md)) becomes the bottleneck before compute does.

```python
def amdahl(P: float, N: int) -> float:
    """Speedup from N processors given parallel fraction P."""
    return 1.0 / ((1 - P) + P / N)

# Print speedup table
import math
for serial_frac in [0.5, 0.1, 0.01, 0.001]:
    P = 1 - serial_frac
    row = [f"{amdahl(P, N):.2f}×" for N in [2, 4, 8, 16, 32, 64, 128]]
    ceiling = 1 / serial_frac
    print(f"Serial={serial_frac*100:.1f}%  ceiling={ceiling:.0f}×  {row}")
```

### Gustafson's Law (1988) — Weak Scaling

Amdahl assumes the problem size is fixed. Gustafson's insight: **with more processors, you typically solve a *bigger* problem in the same time**, not the same problem faster. This is **weak scaling**.

If $P$ is again the parallel fraction and you scale both the work and the processor count together:

$$S_{\text{Gustafson}}(N) = (1 - P) + N \cdot P$$

This grows **linearly** with $N$. The serial overhead is now amortized over a much larger total workload.

**Intuition:** A research team with 4 GPUs trains a 7B model in 2 days. A team with 256 GPUs doesn't train the *same* 7B model in $2/64 = 1.8$ hours — they train a **70B model** in 2 days. The parallel portion (forward/backward compute) scales with the model; the serial portion (checkpointing, logging, evaluation) doesn't. Gustafson's Law says the *effective speedup on the scaled problem* is nearly linear in the number of GPUs.

```python
def gustafson(P: float, N: int) -> float:
    """Scaled speedup from N processors given parallel fraction P."""
    return (1 - P) + N * P

# Comparison: Amdahl vs Gustafson at P=0.99 (1% serial)
P = 0.99
print(f"{'N':>6} {'Amdahl':>10} {'Gustafson':>12}")
for N in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    print(f"{N:>6} {amdahl(P, N):>9.1f}× {gustafson(P, N):>11.1f}×")
```

| $N$ | Amdahl ($P=0.99$) | Gustafson ($P=0.99$) |
|:---:|:---:|:---:|
| 8 | 7.5× | 7.9× |
| 64 | 39× | 63.4× |
| 512 | 84× | 507× |
| 4096 | 99× | 4059× |

**The strong- vs weak-scaling insight:** Amdahl governs "how fast can I do *this exact training run*?" Gustafson governs "how big a model/dataset can I train in a *given wall-clock time*?" Both matter. In practice, LLM research is dominated by Gustafson-style scaling — you use more GPUs to tackle larger problems, not (just) to speed up existing ones.

---

## Synchronization Basics

Parallel threads that share data need coordination mechanisms. Three essential primitives:

### Barriers

A **barrier** is a synchronization point: all threads in a group must arrive before any can proceed. In CUDA: `__syncthreads()` synchronizes all threads within a **thread block** (shared memory writes before this call are visible to all threads in the block after it). Barriers are cheap within a warp (implicit) but expensive across SMs (requires inter-SM coordination).

In PyTorch distributed training: `dist.barrier()` synchronizes all processes — every GPU waits until all have reached this point. Used before/after gradient averaging, checkpoint saving, etc. Barrier costs can dominate at scale.

### Atomic Operations

An **atomic** read-modify-write guarantees that no other thread's operation interleaves. Example: `atomicAdd(&counter, 1)` in CUDA increments a shared counter without a race condition. Atomics serialize conflicting accesses — use them for correctness, not performance. High atomic contention (many threads hitting the same address) creates bottlenecks.

### Race Conditions

A **race condition** occurs when the result of a computation depends on the non-deterministic timing of concurrent threads. Example:

```python
# Two threads both read x=0, both compute x+1=1, both write 1 → x=1 (wrong; expected 2)
# This is a read-modify-write race — needs an atomic or a lock
```

In ML: non-determinism in multi-GPU training (e.g., gradient accumulation with un-synchronized writes) can cause subtle correctness bugs. Libraries like PyTorch handle this, but understanding it explains why `torch.use_deterministic_algorithms(True)` is sometimes needed for reproducibility — and why it can be slower.

---

## Why This Matters for Model Optimization

<div class="diagram">
<div class="diagram-title">Parallelism Levels → ML Hardware Choices</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Bit-level</div>
    <div class="card-desc">INT8/INT4 quantization packs more ops per cycle. A 256-bit SIMD lane does 32× INT8 MACs vs 8× FP32. More throughput, not just less memory.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">ILP / Superscalar</div>
    <div class="card-desc">CPUs extract parallelism within a kernel. Flash Attention's fused kernel reduces memory-level hazards and makes better use of out-of-order dispatch.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">SIMD</div>
    <div class="card-desc">NumPy/PyTorch vectorization routes to AVX/NEON SIMD. Tensor layout (NCHW vs NHWC) affects which SIMD paths the kernel hits — NHWC is SIMD-friendlier for conv.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">SIMT (GPU)</div>
    <div class="card-desc">Deep learning is embarrassingly parallel: each output element of a matmul is independent. SIMT lets thousands of threads compute each independently — perfect fit.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Amdahl's Law</div>
    <div class="card-desc">Explains why adding GPUs to data-parallel training yields diminishing returns. The AllReduce (serial bottleneck) grows with the gradient tensor size vs bandwidth.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Gustafson's Law</div>
    <div class="card-desc">Explains why 1000-GPU clusters train trillion-parameter models in the same wall time as 8-GPU clusters train billion-parameter models. Weak scaling is the real win.</div>
  </div>
</div>
</div>

**Deep learning is "embarrassingly parallel" at the right granularity.** A single matrix multiply $C = AB$ where $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$ produces $M \times N$ output elements. Each output element $C_{ij} = \sum_k A_{ik} B_{kj}$ is *independent* of all others (given $A$ and $B$). There are no data dependencies between output elements. With $M=4096$, $N=4096$, that is **~16.7 million** completely independent dot products — exactly what a GPU's 16,896 CUDA cores (in H100) are built to compute simultaneously.

The serial bottleneck in multi-GPU training is **gradient synchronization** — AllReduce over the parameter vector. A 70B-parameter model in BF16 has $70 \times 10^9 \times 2 = 140$ GB of gradients to reduce per step. At NVLink's 900 GB/s bidirectional bandwidth on an 8-GPU DGX H100 node, the ring-AllReduce takes ~0.16 seconds per 200 GB (effective ~1.25 GB/s per GPU for 8-GPU ring) — easily 10–40% of step time for large models. This is the Amdahl serial fraction that [Chapter 12](./12_buses_and_interconnects.md) and [Chapter 28](./28_multi_gpu_and_distributed_training.md) address directly.

---

## Key Takeaways

- **Parallelism is hardware's answer to the end of Dennard scaling**: since ~2005, chips gain performance through more parallel units, not faster clocks.
- **ILP** (superscalar, OoO, speculation) extracts parallelism within a single thread — but tops out at ~3–4 useful instructions/cycle due to dependency graphs.
- **SIMD** performs one instruction on multiple data elements simultaneously; the CPU vector register width (128–512 bits) directly determines throughput for numeric workloads — the hardware basis for NumPy/PyTorch vectorization.
- **SIMT** (GPU) writes programs as independent threads; the hardware groups 32 into a **warp** and executes them in SIMD fashion. Warp divergence (conditional branches with different outcomes per thread) is the main pitfall.
- **Flynn's taxonomy** classifies architectures as SISD/SIMD/MISD/MIMD; GPUs are SIMT within a warp and MIMD across SMs; multi-GPU clusters are MIMD.
- **Amdahl's Law** ($S = 1/((1-P) + P/N)$) shows that even a small serial fraction caps speedup — explaining why communication overhead kills GPU-scaling efficiency.
- **Gustafson's Law** ($S = (1-P) + NP$) shows that scaling the problem with the processor count allows near-linear speedup — explaining how LLM research scales to trillion parameters.

---

*Next: [Chapter 9 — Memory Types & Technologies](./09_memory_types.md), where we tour the full memory hierarchy — registers through HBM — and learn why memory bandwidth, not FLOPs, is usually what you're actually waiting on.*

[← Back to Table of Contents](./README.md)
