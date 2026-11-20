---
title: "Chapter 14 — CPUs"
---

[← Back to Table of Contents](./README.md)

# Chapter 14 — CPUs

The CPU is the chip you have known your whole life — but its design philosophy is deeply specific, and understanding it explains *exactly* when a CPU is the right tool for an ML workload and when it is the wrong one. CPUs are built to finish a **single task as fast as possible**: minimize latency on a sequential stream of diverse, data-dependent instructions. Every baroque feature inside a modern CPU — out-of-order execution, superscalar issue, branch prediction, speculative execution, deep cache hierarchies — exists to serve that one goal.

GPUs and TPUs exist because that goal is *not* what neural network training needs most of the time. But CPUs are not irrelevant to ML: they gate your data pipelines, handle your embedding lookups, run small-batch inference in latency-sensitive serving stacks, and sit at the center of every distributed training job orchestrating accelerators. An ML engineer who understands CPUs makes better system decisions than one who does not.

> **The one-sentence version:** A CPU uses enormous amounts of silicon cleverness — out-of-order execution, branch prediction, deep caches, SIMD — to make a *sequential* program run as fast as possible on a handful of powerful cores; this is the opposite of the GPU's design, and that contrast explains almost everything about where each processor fits in the ML stack.

---

## The CPU Design Philosophy: Latency Over Throughput

Before drilling into mechanisms, set the frame. A CPU is **latency-optimized**:

<div class="diagram">
<div class="diagram-title">CPU vs GPU: Design Philosophy at a Glance</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">CPU</div>
    <ul>
      <li><strong>Goal:</strong> minimize time to finish one task</li>
      <li>~8–192 powerful cores (up to 288 small cores)</li>
      <li>Each core: complex OoO logic, large caches, branch predictor</li>
      <li>Clock: 3–5 GHz</li>
      <li>Optimized for <strong>branchy, irregular, data-dependent</strong> code</li>
      <li>Cache hierarchy hides memory latency per thread</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">GPU</div>
    <ul>
      <li><strong>Goal:</strong> maximize total work done per second</li>
      <li>~10,000+ simple CUDA cores</li>
      <li>Each SM: minimal OoO, tiny per-thread cache, many warps</li>
      <li>Clock: 1–2 GHz</li>
      <li>Optimized for <strong>regular, data-parallel, branch-free</strong> code</li>
      <li>Latency hidden by switching between thousands of threads</li>
    </ul>
  </div>
</div>
</div>

The CPU's powerful core is expensive in silicon area and power; it is justified because that complexity extracts near-maximum performance from irregular code that a simple core would crawl through.

The CPU's main adversaries are:
1. **Data hazards** — an instruction needs a result not yet computed.
2. **Control hazards** — a branch changes which instruction comes next.
3. **Memory latency** — DRAM is ~100 ns away; a 4 GHz clock tick is 0.25 ns.

The CPU's core microarchitecture is a system of interlocking solutions to all three.

---

## A Modern CPU Core in Depth

### Pipelining — the Baseline (Review)

We covered pipelining in [Chapter 4 — Anatomy of a Processor](./04_anatomy_of_a_processor.md). A quick refresher: a simple pipeline splits instruction execution into stages (Fetch → Decode → Execute → Writeback), with each stage handling a different instruction simultaneously. A 5-stage pipeline keeps 5 instructions "in flight" — but it gets a **hazard stall** whenever an instruction must wait for a predecessor. Modern CPUs add machinery *on top of* pipelining to eliminate most of those stalls.

### Out-of-Order (OoO) Execution

Consider this instruction sequence:

```text
A = load [addr1]       ; instruction 1 — may miss the cache (~40 cycles)
B = A + 1              ; instruction 2 — must wait for A
C = load [addr2]       ; instruction 3 — independent of A,B
D = C * 4              ; instruction 4 — depends on C
E = B + D              ; instruction 5 — depends on B and D
```

A simple in-order CPU stalls on instruction 2 until instruction 1's load returns — **40+ cycles wasted**. An **out-of-order (OoO)** core:

1. **Decodes** all instructions into micro-operations (µops).
2. **Renames registers** and places µops into a **Reservation Station** (RS) / **Scheduler**.
3. Each µop waits there until **all its inputs are ready** — and then it is **dispatched to an execution unit**, regardless of program order.
4. Instruction 3 and 4 execute *while* instruction 1 is still in the memory system.
5. When all µops complete, a **Reorder Buffer (ROB)** **retires** them in **original program order** (to preserve the correct visible state).

<div class="diagram">
<div class="diagram-title">Out-of-Order Core: The Key Structures</div>
<div class="flow">
  <div class="flow-node accent wide">Fetch + Decode (in-order)<br/><small>~4–6 instructions/cycle fetched from I-cache</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Register Rename (RAT — Register Alias Table)<br/><small>map arch. registers → physical registers; eliminate false dependencies</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">Reservation Stations / Scheduler<br/><small>hold µops; watch for operands; dispatch when ready</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Execution Units (out-of-order)<br/><small>ALU, FPU, AGU, load/store, SIMD — multiple per core</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Reorder Buffer (ROB)<br/><small>commit (retire) µops in program order; handle exceptions/faults</small></div>
</div>
</div>

**Register renaming** is the key insight that makes OoO safe. When a program writes register `R5` twice, the second write could look like it depends on the first (a "write-after-write" hazard). Renaming maps each architectural register name to a fresh **physical register** on every new definition — so the two writes go to different physical registers and can proceed independently.

A modern server core (e.g., AMD Zen 5, Intel Redwood Cove/Lion Cove) maintains:
- **ROB size:** ~320–512 µops (Zen 4: 320, Zen 5: 448, Intel Golden Cove: 512): the "window" of instructions the CPU can look ahead across to find independent work.
- **Reservation stations:** 128–256 entries.
- **Physical register file:** 280–400+ integer + 200+ FP physical registers.
- **Issue width:** 4–6 µops per cycle (superscalar, described next).

> **Implication for ML:** OoO execution is what allows a CPU to run Python (which is an interpreter with many branches and data dependencies) at tolerable speeds. Without OoO, a CPU would be 2–5× slower on irregular code. But OoO silicon area is also why a CPU core costs many times more transistors than a GPU "CUDA core."

### Superscalar Multiple-Issue

"4-wide superscalar" means the CPU issues **up to 4 µops per clock cycle** to its execution units. A modern Intel or AMD core has multiple independent **execution ports**, each capable of handling certain classes of µop:

| Ports (Intel Golden Cove) | Can execute |
|---|---|
| 0, 1, 5 | Integer ALU and vector ALU/FMA (512-bit operations pair ports 0+1, plus port 5 on server parts) |
| 6, 10 | Integer ALU (port 6 and port 0 also take branches) |
| 2, 3, 11 | Loads from the L1 data cache |
| 7, 8 | Store-address generation |
| 4, 9 | Store data |

The scheduler dispatches up to one µop per port per cycle — so if it can find 6 independent µops ready, it dispatches 6 simultaneously. This is the "width" of the processor.

**Peak throughput:** a 6-wide core running at 4 GHz can issue up to 6 × 4 × 10⁹ = **24 billion µops/second**. That is the upper bound; real code rarely reaches it due to dependencies and cache misses.

### Branch Prediction

Nearly all programs have branches (if/else, loops). The moment the CPU sees a branch instruction, it does not know which direction it goes until the condition is evaluated — but the condition might depend on a value still being computed. If the CPU simply stalls and waits, it wastes the entire pipeline depth (10–20+ cycles on a deep pipeline).

**Branch prediction** guesses the outcome *before* the condition is known, and the CPU **speculatively continues execution** down the predicted path. When the branch resolves:
- **Predicted correctly:** no stall; the speculative work is kept. Modern predictors are correct **~98–99%** of the time on typical code.
- **Mispredicted:** the CPU must **flush** all speculative µops from the ROB and RS, discard their results, and restart from the correct path. This costs ~15–20 cycles on a modern deep pipeline — a **branch misprediction penalty**.

**What makes a branch hard to predict?** Truly random outcomes (e.g., sorting random data with comparisons) — the classic example is why `std::sort` on random data is harder for branch predictors than sorting nearly-sorted data.

Modern predictors use sophisticated structures:
- **TAGE (Tagged Geometric-history length predictor):** maintains history tables keyed on global branch history of varying lengths; outstanding at capturing long-range patterns.
- **Indirect branch predictors:** handle jump-to-register instructions (virtual function calls, function pointers).
- **Loop predictors:** detect counted loops and predict the exit correctly.

The **branch predictor occupies a significant fraction of the core's power and area budget** — a reminder that instruction-level control logic is where CPUs put their transistors.

### Speculative Execution — and Spectre/Meltdown

Speculative execution is the broader policy: the CPU does not just predict branches — it may speculatively execute across **loads**, across **indirect calls**, across **virtually any potential hazard**, executing instructions whose effects it later discards if they turn out to be wrong.

**Spectre (2018) and Meltdown (2018)** revealed that speculative execution has a security side channel: even if speculative work is *logically* discarded, it can leave traces in the **cache state** (which cache lines were loaded). An attacker can time cache accesses to infer what data a speculative load touched — leaking across security boundaries. The consequences:
- Meltdown (speculative kernel/user privilege crossing): patched by OS-level fixes (KPTI — Kernel Page Table Isolation) that cost 5–30% of performance on I/O-heavy workloads.
- Spectre (branch predictor poisoning): harder to mitigate; requires serializing barriers in sensitive code, retpoline compiler mitigations, and microcode updates.

> For ML workloads, Spectre/Meltdown mitigations cost little on compute-heavy GPU work (GPU execution is unaffected), but **data-loading pipelines, system calls, and CPU-side pre-processing** see the overhead. This is why ML engineers running on VMs should keep OS and microcode updated, but can largely focus on data I/O patterns for the performance hit.

---

## The Cache Hierarchy: Racing Against DRAM

DRAM latency is ~80–120 ns load-to-use. A 4 GHz clock ticks every 0.25 ns, so a cache miss costs **300–500 cycles** of potential stall. The CPU's answer is a hierarchy of increasingly large, increasingly slow caches:

<div class="diagram">
<div class="diagram-title">CPU Cache Hierarchy (per modern server core)</div>
<div class="layer-stack">
  <div class="layer"><strong>Registers</strong> — ~16 int + 16 FP (x86-64), 1 cycle, ~1 KB, on-core</div>
  <div class="layer"><strong>L1 Instruction + Data Cache</strong> — 32–64 KB each, ~4–5 cycle latency, per-core, SRAM</div>
  <div class="layer"><strong>L2 Cache</strong> — 256 KB – 2 MB, ~12–14 cycles, per-core, SRAM</div>
  <div class="layer"><strong>L3 Cache (LLC)</strong> — 4–256 MB (shared across all cores), ~40–50 cycles, SRAM/eDRAM</div>
  <div class="layer"><strong>DRAM (DDR5 / LPDDR5X)</strong> — GBs to TBs, ~80–120 ns latency (~300–500 cycles), off-chip</div>
</div>
</div>

| Level | Typical size | Latency | Bandwidth | Shared? |
|-------|-------------|---------|-----------|---------|
| L1-D | 32–64 KB | 4–5 cycles | ~3 TB/s (on-chip) | Per-core |
| L1-I | 32–64 KB | 4–5 cycles | ~3 TB/s | Per-core |
| L2 | 256 KB – 2 MB | 12–14 cycles | ~1 TB/s | Per-core |
| L3 (LLC) | 4–256 MB | 40–50 cycles | ~200–400 GB/s | All cores |
| DRAM | 32 GB – 6 TB | 80–120 ns | 50–850 GB/s per socket | All cores + I/O |

**Cache line size:** 64 bytes on x86 (512 bits). When a cache miss occurs, the hardware fetches a full 64-byte line, not just the 4 or 8 bytes the instruction requested. This is why **spatial locality** (accessing adjacent memory locations) dramatically improves performance: subsequent accesses to the same line are free. Matrix operations in row-major order exploit this; column-major access to a row-major array defeats it.

**L3 is a weapon, not just a safety net.** AMD's EPYC 9684X (Genoa-X) stacks 64 MB of SRAM (3D V-Cache) on each of its 12 compute dies for **1,152 MB of L3** across 96 cores; Intel's Xeon 6980P has 504 MB. That is large enough to hold **meaningful parts of a model's working set**, e.g. embedding hot rows or a small model's weights, entirely on-chip.

### Prefetching

CPUs also have **hardware prefetchers** that detect stride patterns in memory access and issue loads speculatively before the data is requested. A sequential scan of an array triggers an aggressive stream prefetcher; random lookups cannot be prefetched. This is another reason why ML embedding lookups (random by key) stress CPUs more than sequential matrix operations.

---

## SIMD Units: AVX-512, ARM SVE, and AMX for ML

A single scalar core multiplying two `float32` values does one multiply per cycle. But the same transistors, organized as a **SIMD (Single Instruction, Multiple Data)** unit, can multiply 16 pairs simultaneously:

<div class="diagram">
<div class="diagram-title">Scalar vs SIMD: 16× float32 Multiplies</div>
<div class="flow-h">
  <div class="flow-node accent">Scalar FMUL<br/><small>1 × f32 per cycle</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">SIMD VMULPS (AVX-512)<br/><small>16 × f32 per cycle (512-bit reg / 32 bits)</small></div>
</div>
</div>

### x86 SIMD Generations

| Extension | Register width | float32 lanes | float64 lanes | int8 lanes | Introduced |
|-----------|---------------|--------------|--------------|-----------|------------|
| SSE2 | 128-bit | 4 | 2 | 16 | 2001 (P4) |
| AVX | 256-bit | 8 | 4 | — | 2011 (Sandy Bridge) |
| AVX2 | 256-bit | 8 | 4 | 32 | 2013 (Haswell) |
| AVX-512 (F) | 512-bit | 16 | 8 | 64 | 2017 (Skylake-SP) |
| AVX-512 VNNI | 512-bit | — | — | 64 (INT8 dots) | 2019 (Cascade Lake) |
| AVX-512 BF16 | 512-bit | — | — | 32 (BF16 dots) | 2021 (Cooper Lake) |
| AMX-INT8 / BF16 / FP16 | 8 tile registers × 1 KB | — | — | 1,024 INT8 MACs/cycle/core | 2023 (Sapphire Rapids); FP16 2024 (Granite Rapids) |

**AVX-512 VNNI** (Vector Neural Network Instructions) adds `VPDPBUSD`: compute 4 INT8 dot-products and accumulate into INT32 in a single instruction, across 16 lanes in parallel = **64 INT8 MACs per cycle per core**. This is what enables ~1–4 TOPS/socket for INT8 inference on modern Intel Xeon.

**AMX (Advanced Matrix Extensions)** goes further. It adds **2D tile registers** (8 tiles × 1 KB = 8 KB) and dedicated **TMUL hardware**: one tile instruction multiplies a 16×64 INT8 tile by a 64×16 tile (16,384 MACs) over 16 cycles, i.e. 1,024 INT8 MACs or 512 BF16 MACs per cycle per core — essentially a small systolic array bolted onto each core. A Sapphire Rapids core achieves **~1.4 TFLOPS BF16** with AMX, a 32× jump over scalar FP32.

```python
import torch

# AMX/AVX-512 is transparent via PyTorch's CPU backend.
# For INT8, use torch.ao.quantization; for BF16, cast directly:
x = torch.randn(1, 512, dtype=torch.float32)
model = torch.nn.Linear(512, 512)

# BF16 forward: uses AVX-512 BF16 / AMX-BF16 on Sapphire Rapids
model = model.to(torch.bfloat16)
x_bf16 = x.to(torch.bfloat16)
out = model(x_bf16)  # exploits AMX on supported CPUs

# Check if AMX is available (via CPUID):
import subprocess
result = subprocess.run(['grep', '-m1', 'amx_bf16', '/proc/cpuinfo'],
                        capture_output=True, text=True)
print("AMX available:", bool(result.stdout))
```

### ARM SVE (Scalable Vector Extension)

ARM's answer to AVX-512 is architecturally different: rather than fixing the register width, **SVE makes the vector length an implementation parameter**, from 128 to 2048 bits. Software compiled for SVE runs unmodified on any SVE implementation (ACLE vector-length-agnostic programming). AWS Graviton3 uses 256-bit SVE; Fujitsu A64FX uses 512-bit SVE. Apple M-series uses ARM NEON (128-bit fixed) plus custom AMX co-processor (not the same as Intel AMX — Apple's is wider and unlocked via Accelerate framework).

---

## Multicore and SMT (Simultaneous Multithreading)

Modern CPUs pack **multiple cores** on a single die, each a full OoO processor with its own L1 and L2 caches:

- **Intel Xeon 6:** up to 128 P-cores (Granite Rapids, 6900P series) or 288 E-cores (Sierra Forest; Clearwater Forest on Intel 18A in 2026).
- **AMD EPYC 9005 (Turin):** up to 128 Zen 5 or 192 Zen 5c cores; Zen 6 "Venice" follows on TSMC N2.
- **NVIDIA Grace:** 72 Arm Neoverse V2 cores beside each pair of Blackwell GPUs; Vera (Rubin generation) has 88 custom Arm cores.
- **Arm cloud CPUs:** AWS Graviton4 (96 cores), Google Axion, Microsoft Cobalt 100, AmpereOne (up to 192 cores).

<div class="die-wrap">
<div class="die-frame">
<div class="die" style="grid-template-columns:repeat(6,1fr);grid-template-areas:'a b c d e f' 'io io io io io io' 'g h i j k l';max-width:720px">
<div class="blk cpu" style="grid-area:a">CCD<small>16 × Zen 5c</small></div><div class="blk cpu" style="grid-area:b">CCD<small>16 × Zen 5c</small></div><div class="blk cpu" style="grid-area:c">CCD<small>16 × Zen 5c</small></div><div class="blk cpu" style="grid-area:d">CCD<small>16 × Zen 5c</small></div><div class="blk cpu" style="grid-area:e">CCD<small>16 × Zen 5c</small></div><div class="blk cpu" style="grid-area:f">CCD<small>16 × Zen 5c</small></div>
<div class="blk io" style="grid-area:io;min-height:64px">I/O die (TSMC N6)<small>12 DDR5 channels · 128 PCIe 5.0 / CXL 2.0 lanes · Infinity Fabric to every CCD</small></div>
<div class="blk cpu" style="grid-area:g">CCD<small>32 MB L3</small></div><div class="blk cpu" style="grid-area:h">CCD<small>32 MB L3</small></div><div class="blk cpu" style="grid-area:i">CCD<small>32 MB L3</small></div><div class="blk cpu" style="grid-area:j">CCD<small>32 MB L3</small></div><div class="blk cpu" style="grid-area:k">CCD<small>32 MB L3</small></div><div class="blk cpu" style="grid-area:l">CCD<small>32 MB L3</small></div>
</div>
</div>
<div class="die-caption">AMD EPYC 9965 (Turin, 2024), stylised: twelve 16-core compute dies (TSMC N3E) around one I/O die. Building 192 cores from small, high-yield chiplets is cheaper than one giant die; the cost is extra latency between dies (<a href="./appendix_e_packaging.html">Appendix E</a>).</div>
</div>

**SMT (Simultaneous Multithreading)** — called **Hyper-Threading** by Intel — runs two hardware threads per physical core by duplicating the register file and state, while sharing the execution units and caches. When one thread is stalled (e.g., waiting on a cache miss), the other thread's µops can be issued, improving **utilization** of the execution units. SMT typically adds ~10–30% throughput on mixed workloads for zero additional silicon in the execution units. It can hurt when both threads compete heavily for L1/L2 cache or reservation station entries.

```python
import os
import torch

# Physical cores vs logical threads (hyperthreading):
logical_cpus = os.cpu_count()               # e.g., 48 logical (24 physical × 2 SMT)
physical_cores = logical_cpus // 2          # rough estimate when HT is enabled

# PyTorch thread control — important for CPU inference performance:
torch.set_num_threads(physical_cores)       # avoid SMT contention on compute-heavy ops
torch.set_num_interop_threads(2)            # inter-op parallelism (pipeline stages)
print(f"PyTorch using {torch.get_num_threads()} threads")
```

### NUMA (Non-Uniform Memory Access)

A multi-socket server has **NUMA**: each CPU socket has its own directly attached DRAM bank, and accessing the *other* socket's memory crosses an inter-socket link (e.g., AMD Infinity Fabric, Intel UPI). Local DRAM: ~60 ns; remote DRAM: ~130 ns — a 2× penalty. For large ML workloads on CPU, **NUMA-aware allocation** (keeping data on the socket that processes it) matters.

```python
# Check NUMA topology with numactl (Linux):
# numactl --hardware     shows nodes, distances, memory
# numactl --localalloc   run process with local-node memory allocation

# In Python, via psutil:
import psutil
# memory per NUMA node not directly in psutil; use libnuma or hwloc bindings
```

---

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>A 2026 server CPU has 600–850 GB/s of memory bandwidth and terabytes of capacity; an H100 has 3.35 TB/s but only 80 GB. For a 400B-parameter MoE model where each token touches ~5 % of the weights, which machine decodes faster at batch 1, and what changes at batch 64?</p>
</div>

## Heterogeneous Cores: Big.LITTLE, P-core + E-core

Modern consumer CPUs blend **fast "performance" cores (P-cores)** with **efficient "efficiency" cores (E-cores)**, trading peak single-thread speed for better power/performance across mixed workloads:

<div class="diagram">
<div class="diagram-title">Heterogeneous Core Architecture</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-icon">🚀</div>
    <div class="card-title">P-cores (Performance)</div>
    <div class="card-desc">Full OoO, deep pipeline, large L2. High clock (5+ GHz). High power. Client P-cores run AVX2 (AVX-512 and AMX are on server Xeons). Example: Intel Lion Cove (Lunar Lake, Arrow Lake).</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🔋</div>
    <div class="card-title">E-cores (Efficiency)</div>
    <div class="card-desc">Narrower OoO, shorter pipeline, L2 shared by a 4-core cluster. Lower clock. A fraction of the area and power. Example: Intel Skymont (Lunar Lake, Arrow Lake).</div>
  </div>
</div>
</div>

**Apple Silicon** takes this furthest. An M4 Max (2024) has:
- 12 P-cores and 4 E-cores, with a matrix coprocessor that M4 exposes as Arm **SME**
- a 40-core GPU (M5 adds a "neural accelerator" to every GPU core)
- a 16-core Neural Engine at 38 TOPS INT8
- 546 GB/s of unified LPDDR5X memory shared by all of them

The scheduler routes latency-sensitive threads to P-cores and background threads (compile jobs, daemons) to E-cores transparently. For ML inference that stays on CPU, **pin PyTorch threads to P-cores** to avoid the scheduler migrating hot threads to E-cores mid-inference.

---

## Server CPUs for ML: Specifications and Comparison

| CPU | Cores | L3 / SLC | Vector and matrix units | Memory | Mem BW (peak) | TDP |
|-----|-------|----------|-------------------------|--------|---------------|-----|
| Intel Xeon 6980P (Granite Rapids, 2024) | 128 P | 504 MB | AVX-512 + AMX (INT8/BF16/FP16) | 12-ch DDR5-6400 or MRDIMM-8800 | 614 / 844 GB/s | 500 W |
| Intel Xeon 8592+ (Emerald Rapids, 2023) | 64 | 320 MB | AVX-512 + AMX | 8-ch DDR5-5600 | 358 GB/s | 350 W |
| AMD EPYC 9965 (Turin Dense, 2024) | 192 Zen 5c | 384 MB | AVX-512 (512-bit datapath) | 12-ch DDR5-6400 | 614 GB/s | 500 W |
| AMD EPYC 9684X (Genoa-X, 2023) | 96 Zen 4 | 1,152 MB (3D V-Cache) | AVX-512 (256-bit datapath) | 12-ch DDR5-4800 | 461 GB/s | 400 W |
| NVIDIA Grace (GH200 / GB200) | 72 Neoverse V2 | 114 MB | 4 × 128-bit SVE2 / NEON per core | LPDDR5X, on package | ~500 GB/s | ~250 W (CPU + memory) |
| Ampere Altra Max M128-30 | 128 Neoverse N1 | 16 MB SLC | 2 × 128-bit NEON per core | 8-ch DDR4-3200 | 205 GB/s | 250 W |
| Apple M4 Max | 12 P + 4 E | shared L2 + SLC | NEON + SME matrix unit | 512-bit LPDDR5X-8533 | 546 GB/s | ~70 W (est., whole chip) |

For scale: AMX performs 1,024 BF16 FLOPs (2,048 INT8 operations) per cycle per core, so a 128-core Xeon 6 at ~2 GHz peaks near **260 TFLOPS BF16** on paper, about a quarter of one H100, with 4–7× less memory bandwidth. Sustained application throughput is lower.

> **Memory bandwidth now decides CPU inference.** A Xeon 6 with MRDIMM-8800 reaches ~844 GB/s and an EPYC Turin ~614 GB/s per socket, both a large step up from the ~300–460 GB/s of the previous generation. For memory-bandwidth-bound work (large embedding lookups, CPU LLM decode), channels × transfer rate matters more than core count.

---

## Memory Channels and Bandwidth

CPU memory bandwidth is determined by: **channels × data rate × bus width**.

$$\text{Bandwidth} = N_\text{channels} \times \text{data rate (MT/s)} \times 8 \text{ bytes/transfer}$$

For AMD EPYC Genoa, 12 channels × DDR5-4800 × 8 B = 12 × 4800 × 10⁶ × 8 = **460.8 GB/s**.

DDR5 also supports **sub-channel mode**: each DIMM slot actually has two 32-bit sub-channels, effectively doubling the number of logical channels versus DDR4. This is why DDR5 can offer higher bandwidth even on the same physical channel count.

**Contrast with GPU:** an H100 SXM provides **3.35 TB/s** of HBM3 bandwidth and a B200 8 TB/s, 4–13× a top server CPU socket at ~600–850 GB/s. This is the root reason GPUs win memory-bandwidth-bound operations like LLM decode.

---

## Why This Matters for Model Optimization

### When CPUs Are the Right Choice

<div class="diagram">
<div class="diagram-title">CPU Use Cases in ML Stacks</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">📦</div>
    <div class="card-title">Data Loading & Preprocessing</div>
    <div class="card-desc">Decode JPEG/video, tokenize text, augment images, shuffle batches. Inherently sequential, branchy, uses OS syscalls. CPU wins unconditionally.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🔍</div>
    <div class="card-title">Embedding Lookups (Recommender)</div>
    <div class="card-desc">Sparse, random-access lookups into TB-scale embedding tables. CPU NUMA + huge L3 (V-Cache) competes with GPU for sparse access patterns.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">⚡</div>
    <div class="card-title">Low-Latency Single-Query Inference</div>
    <div class="card-desc">Batch size 1, small model. Each GPU kernel launch costs ~5–20 µs of CPU-side overhead plus framework dispatch; for a model with hundreds of tiny ops, a CPU running AMX matmuls in cache can win end-to-end.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">🦙</div>
    <div class="card-title">llama.cpp / CPU LLM Decode</div>
    <div class="card-desc">Memory-bandwidth-bound decode of quantized (Q4/Q8) weights; CPU with high memory BW (Grace, EPYC) can do ~10–50 tokens/sec for 7B models.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">🎛️</div>
    <div class="card-title">Accelerator Orchestration</div>
    <div class="card-desc">CPU runs the training loop, optimizer state updates (in mixed-precision), checkpoint logic, gradient compression. GPU offloading (ZeRO-Offload) uses CPU memory.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-icon">🏠</div>
    <div class="card-title">On-Device / Edge Inference</div>
    <div class="card-desc">Mobile SoCs: CPU handles background workloads, pre/post-processing while NPU runs the heavy model. Apple Neural Engine offloads from CPU AMX.</div>
  </div>
</div>
</div>

### The Roofline on a CPU: Memory Bandwidth Bounds LLM Decode

For LLM decode (autoregressive generation), each generated token requires loading the **entire model weight matrix** once — arithmetic intensity ≈ 2 FLOPs per 2-byte weight ≈ **1 FLOP/byte** in BF16. This is far below the ridge point on any CPU roofline:

```python
# CPU Roofline estimate for LLM decode (memory-bandwidth bound):
peak_flops = 14e12          # ~14 TFLOPS BF16 (Emerald Rapids 64c with AMX)
mem_bw     = 307e9          # ~307 GB/s (Emerald Rapids, 8-ch DDR5-8800)
ridge_point = peak_flops / mem_bw  # ~45.6 FLOP/byte

# LLM decode arithmetic intensity:
model_bytes = 7e9 * 2       # 7B params × 2 bytes (BF16)
flops_per_token = 2 * 7e9   # ~2 × param count for one forward pass
arith_intensity = flops_per_token / model_bytes  # 1.0 FLOP/byte

print(f"Ridge point: {ridge_point:.1f} FLOP/byte")
print(f"LLM decode intensity: {arith_intensity:.1f} FLOP/byte")
print(f"We are {ridge_point / arith_intensity:.0f}× below the ridge point")
# => memory-bandwidth-bound by ~45× on this CPU
# => throughput ≈ mem_bw × arith_intensity = 307 GB/s × 1 FLOP/byte = 307 GFLOPS
# => not limited by AMX compute at all; bandwidth is the only lever

tokens_per_second = mem_bw / model_bytes   # rough: 307e9 / 14e9 ≈ 21 tok/s
print(f"Estimated decode speed (7B BF16, CPU): ~{tokens_per_second:.0f} tokens/sec")
```

This is why **4-bit quantization (llama.cpp's Q4_K_M, ~4.8 bits per weight) speeds up CPU decode by roughly 3×** over FP16 and ~1.7× over Q8_0: tokens/s scales almost exactly with the inverse of bytes moved per token.

```python
# Thread/affinity tuning for CPU inference:
import torch
import os

# On a system with 96 cores (e.g. EPYC Genoa), avoid over-subscribing:
num_physical_cores = 96
torch.set_num_threads(num_physical_cores)
os.environ["OMP_NUM_THREADS"] = str(num_physical_cores)
os.environ["MKL_NUM_THREADS"] = str(num_physical_cores)
# Also: numactl --cpunodebind=0 --membind=0 python inference.py
# ensures all work stays on socket 0, avoiding NUMA penalty
```

### Practical CPU Inference with PyTorch

```python
import torch
import torch.nn as nn

# A transformer block benchmarked on CPU:
class FeedForward(nn.Module):
    def __init__(self, d_model=4096, d_ff=16384):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

model = FeedForward()

# INT8 static quantization (uses FBGEMM on x86 = AVX-512 VNNI):
model.eval()
model_int8 = torch.ao.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
# On Sapphire Rapids: INT8 FBGEMM path will use AVX-512 VNNI
# ~2-4× throughput vs FP32 for large matmuls

x = torch.randn(1, 4096)     # batch_size=1, d_model=4096
with torch.no_grad():
    out_fp32 = model(x)
    out_int8 = model_int8(x)

# torch.compile() for additional CPU speedup via inductor:
model_compiled = torch.compile(model, backend="inductor")
```

---

## Key Takeaways

- A CPU is **latency-optimized**: it uses **out-of-order execution, register renaming, superscalar issue, branch prediction, and deep caches** to finish irregular sequential code as fast as possible — at the cost of enormous silicon per core.
- **OoO execution** hides data hazards by finding independent µops in a window of ~512 instructions; the **ROB** ensures the visible state is always sequentially consistent.
- **Branch misprediction** costs 15–20 cycles; modern predictors (TAGE) achieve ~99% accuracy on regular code, but truly random branches still hurt.
- The **cache hierarchy** (L1 ~5 cycles, L2 ~14 cycles, L3 ~50 cycles, DRAM ~300 cycles) is the CPU's main weapon against the memory wall; **64-byte cache lines** reward spatial locality.
- **SIMD units** (AVX-512, ARM SVE) and **matrix extensions** (Intel AMX, Arm SME) give CPUs ~10–100× throughput on regular numerical code versus scalar; **AMX** puts a 128-core Xeon 6 near ~260 TFLOPS BF16 on paper, a quarter of an H100.
- Modern **server CPUs** (EPYC, Xeon, Grace, Altra) differ mainly in core count, memory channel count, cache size, and SIMD width — not in fundamental architecture.
- **LLM decode on CPU is memory-bandwidth-bound**, not compute-bound; quantization (INT4/INT8) directly multiplies throughput by reducing bytes transferred per token.
- CPUs are essential for **data pipelines, sparse embedding lookups, low-latency single-query inference, llama.cpp deployments, and accelerator orchestration** — and irreplaceable even in GPU-dominated stacks.

## Check Yourself

<details class="check">
<summary>What does register renaming eliminate, and what does it leave?</summary>
<p>It eliminates false dependencies (write-after-read and write-after-write on the same architectural register) by giving each new result its own physical register. True read-after-write dependencies remain: an instruction still has to wait for the value it actually consumes.</p>
</details>

<details class="check">
<summary>Peak memory bandwidth of 12 channels of DDR5-6400?</summary>
<p>12 × 6,400 × 10⁶ transfers/s × 8 bytes = 614.4 GB/s.</p>
</details>

<details class="check">
<summary>Roughly how many tokens/s can a 12-channel DDR5-6400 server decode for a 70B model in Q4_K_M (~42 GB)?</summary>
<p>Bound: 614 GB/s ÷ 42 GB ≈ 14.6 tokens/s at batch 1; real implementations reach perhaps 60–80 % of that, ~9–12 tokens/s.</p>
</details>

<details class="check">
<summary>Why do client Intel P-cores lack AVX-512 when server cores have it?</summary>
<p>Client chips pair P-cores with E-cores that do not implement AVX-512, and the OS must be able to move any thread to any core, so AVX-512 was fused off on hybrid parts from 2021. Intel's AVX10 plan exists to bring a common 512-bit ISA back to all cores.</p>
</details>

<details class="check">
<summary>When can a CPU beat a GPU for inference?</summary>
<p>When the model is small and batch size is 1 (GPU launch overhead dominates), when the model or embedding tables exceed GPU memory but fit in host DRAM, when data never needs to leave the CPU (pre- and post-processing), or when cost and availability matter more than peak speed.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Intel 64 and IA-32 Architectures Optimization Reference Manual | Intel | [intel.com/content/www/us/en/developer/article…](https://www.intel.com/content/www/us/en/developer/articles/technical/intel64-and-ia32-architectures-optimization.html) | Golden Cove ports, AMX throughput |
| Software Optimization Guide for AMD Family 1Ah (Zen 5) | AMD | [amd.com/en/search/documentation/hub.html](https://www.amd.com/en/search/documentation/hub.html) | Zen 5 core resources: ROB, schedulers, AVX-512 datapath |
| Chips and Cheese: microarchitecture deep dives | Chips and Cheese | [chipsandcheese.com](https://chipsandcheese.com/) | Measured latencies, cache sizes and core structures |
| A New Case for the TAGE Branch Predictor | A. Seznec, MICRO 2011 | [doi.org/10.1145/2155620.2155635](https://doi.org/10.1145/2155620.2155635) | The TAGE predictor family used in modern cores |
| llama.cpp | G. Gerganov et al. | [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | CPU LLM inference and the Q4_K_M format |
| Spectre Attacks: Exploiting Speculative Execution | P. Kocher et al., IEEE S&P 2019 | [doi.org/10.1109/SP.2019.00002](https://doi.org/10.1109/SP.2019.00002) | Speculative side channels and their mitigations |

---

**Next:** [Chapter 15 — GPUs](./15_gpus.md) — We meet the chip that actually dominates ML training: thousands of simple cores, Tensor Cores, HBM, and the warp model that hides latency through massive multithreading instead of clever control logic.

[← Back to Table of Contents](./README.md)
