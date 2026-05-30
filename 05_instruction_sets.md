---
title: "Chapter 5 — Instruction Sets (ISAs)"
---

[← Back to Table of Contents](./README.md)

# Chapter 5 — Instruction Sets (ISAs)

PyTorch's `torch.matmul` eventually becomes a stream of 1s and 0s that the CPU or GPU executes directly. Between your Python line and that binary stream are several translations — but the most fundamental boundary is the **Instruction Set Architecture (ISA)**: the contract between software and silicon that defines exactly which operations the processor promises to support, and in which binary encoding. Understanding this contract is what lets you reason about *why* AVX-512 gives a 2× to 8× speedup on CPU inference, *why* oneDNN chooses different code paths on different x86 chips, and *why* the GPU's "SIMT" execution model is the natural descendant of the CPU's "SIMD" extensions.

This chapter builds from first principles — what an instruction actually is — up through the RISC/CISC divide, the major ISAs you encounter in practice, and the vector/SIMD extensions that are the CPU-side ancestors of GPU tensor cores. It ends by connecting every concept directly to the integers and floats your models are made of.

> **The one-sentence version:** An ISA is the hardware/software contract — the list of operations, encodings, and rules a CPU promises to obey; the microarchitecture is the (replaceable) implementation that fulfills that promise.

---

## The ISA vs the Microarchitecture

These two terms are constantly confused. The distinction is critical.

<div class="diagram">
<div class="diagram-title">ISA vs Microarchitecture</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">ISA (the contract)</div>
    <ul>
      <li>The <em>specification</em>: which instructions exist, how they're encoded, what registers are visible, what memory model is guaranteed</li>
      <li>Defines the <strong>programmer-visible state</strong>: registers, flags, PC</li>
      <li>Stable for decades — x86-64 binaries from 2004 run on 2024 CPUs</li>
      <li>Examples: x86-64, ARMv9, RISC-V RV64GC</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Microarchitecture (the implementation)</div>
    <ul>
      <li>The <em>silicon design</em> that executes the ISA: pipeline stages, out-of-order engine, branch predictors, cache sizes</li>
      <li>Invisible to a correct program — only performance changes</li>
      <li>Changes every 1–3 years per generation</li>
      <li>Examples: Intel Raptor Cove (x86), ARM Cortex-X4 (ARMv9), SiFive P670 (RISC-V)</li>
    </ul>
  </div>
</div>
</div>

The payoff of this split: Intel can redesign every transistor in a new processor ("microarchitecture") without breaking any existing software, as long as it still faithfully executes x86-64 instructions ("ISA"). Conversely, AMD and Intel can both produce x86-64 chips — same ISA, completely different silicon.

> **For model optimization:** when you compile a model with `torch.compile` or export to ONNX and then AOT-compile for x86, you target an ISA (including which extensions are present). The runtime can then pick the best microarchitecture-specific implementation.

---

## Anatomy of an Instruction

An instruction is a small, fixed-format command that the processor can decode and execute in one (or a few) cycles. Every instruction has:

1. **Opcode** — which operation (e.g., `ADD`, `MUL`, `LOAD`, `BRANCH`)
2. **Operands** — what to operate on
3. **Encoding** — the binary layout that represents #1 and #2

### Instruction Encoding

Consider a simplified 32-bit fixed-length RISC encoding (like RISC-V's R-type):

```text
 31      25 24   20 19   15 14  12 11    7 6       0
┌──────────┬───────┬───────┬──────┬───────┬─────────┐
│  funct7  │  rs2  │  rs1  │funct3│  rd   │ opcode  │
│  7 bits  │ 5 bits│ 5 bits│3 bits│ 5 bits│  7 bits │
└──────────┴───────┴───────┴──────┴───────┴─────────┘

Example: ADD x3, x1, x2
  funct7=0000000  rs2=x2(00010)  rs1=x1(00001)  funct3=000  rd=x3(00011)  opcode=0110011
  Binary: 0000000_00010_00001_000_00011_0110011
```

x86-64, by contrast, uses **variable-length** encoding (1–15 bytes per instruction), with prefixes that modify width, segment, or lock behavior. This complexity is the defining feature of CISC — power at the cost of decoder complexity.

### Operand Types

| Operand type | Example (x86-64) | Description |
|---|---|---|
| Register | `rax`, `xmm0` | Value lives in a named CPU register (~ns latency) |
| Immediate | `42`, `0xFF` | Constant baked into the instruction encoding |
| Memory (direct) | `[0xDEAD]` | Load/store at a fixed address |
| Memory (register-indirect) | `[rsp + 8]` | Address computed as register + offset |
| Memory (SIB) | `[rax + rbx*4 + 16]` | Base + index×scale + displacement — for array indexing |

### Registers

Registers are the processor's fastest storage — typically **64-bit general-purpose** registers (x86-64 has 16: `rax`–`r15`; ARM64 has 31: `x0`–`x30`). Additionally:

- **SIMD/vector registers**: x86 `ymm0`–`ymm15` (256-bit), `zmm0`–`zmm31` (512-bit); ARM `v0`–`v31` (128-bit NEON, 2048-bit SVE max)
- **Status/flags register**: carries results of comparisons (ZF, CF, OF, SF)
- **Program counter (PC/RIP)**: address of the next instruction

Register access is ~1 cycle. An L1-cache miss is ~4 cycles. A DRAM access is ~200 cycles. The ISA exposes registers; the microarchitecture may rename them internally (register renaming in out-of-order cores) to far more physical registers than the ISA requires.

---

## RISC vs CISC: Philosophy and Hardware Consequences

This is one of the most consequential design debates in computer architecture.

<div class="diagram">
<div class="diagram-title">RISC vs CISC</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">CISC — Complex Instruction Set Computing</div>
    <ul>
      <li>Many, powerful instructions — each can do more work</li>
      <li>Variable-length encoding (x86-64: 1–15 bytes)</li>
      <li>Instructions can directly read/write memory (memory operands)</li>
      <li>Fewer instructions to write a program → smaller code size</li>
      <li>Harder to pipeline; x86 CPUs translate to micro-ops internally</li>
      <li><strong>Example: x86-64</strong></li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">RISC — Reduced Instruction Set Computing</div>
    <ul>
      <li>Fewer, simpler instructions — each does less</li>
      <li>Fixed-length encoding (ARM: 4 bytes; RISC-V: 4 bytes, 2 with C ext)</li>
      <li>Load-store architecture: only dedicated instructions touch memory</li>
      <li>More instructions to write a program, but each decodes in one cycle</li>
      <li>Simpler pipeline; easier to scale to many cores</li>
      <li><strong>Examples: ARM64 (AArch64), RISC-V RV64</strong></li>
    </ul>
  </div>
</div>
</div>

### The Modern Twist: x86 Translates Internally

Modern x86 CPUs are not really "CISC" in execution. Since the mid-1990s (Intel Pentium Pro, 1995), they translate x86 instructions into **micro-ops (µOPs)** — small, RISC-like internal operations — at decode time. The out-of-order engine then executes these micro-ops.

```text
x86-64 decode pipeline:

External ISA     Internal execution
─────────────    ──────────────────────────────────────
ADDQ [rsp+8], rax  →  µOP1: LOAD   tmp1 ← [rsp+8]
                   →  µOP2: ADD    rax  ← rax + tmp1
                   →  µOP3: (writeback is merged with ADD)
```

This means the "CISC vs RISC" distinction is today more of a **software ABI and encoding** story than an execution one. x86 chips execute RISC-like µOPs internally; the complexity is in the front-end decoder.

| Feature | x86-64 | ARM64 (AArch64) | RISC-V RV64GC |
|---|:---:|:---:|:---:|
| Encoding length | 1–15 bytes | 4 bytes (32-bit Thumb2: 2/4) | 4 bytes (2 w/ C ext) |
| Architecture family | CISC | RISC | RISC |
| Internal micro-ops | Yes (Intel/AMD) | No (executes natively) | No |
| Memory operands | Yes | No (load-store) | No (load-store) |
| General-purpose registers | 16 (64-bit) | 31 (64-bit) + SP | 32 (64-bit) |
| SIMD registers | 32×512-bit (ZMM) | 32×128-bit NEON | V ext: 32×VLEN-bit |
| Open-source ISA | No | No | **Yes** |
| Primary ecosystem | PC, server | Mobile, Apple silicon, server | Embedded, growing datacenter |

---

## The Big ISAs Today

### x86-64 (Intel / AMD)

The dominant server and desktop ISA — the machine your CUDA code usually runs the "host" side on. Originated with Intel's 8086 (1978) and extended to 64-bit by AMD in 2003 (hence "AMD64" or "x86-64").

Key facts for ML practitioners:
- Your CPU host code in PyTorch compiles to x86-64
- **AVX-512 and VNNI/AMX** (see SIMD section) give CPUs competitive INT8/BF16 throughput for inference
- Intel Core (Raptor Lake, Meteor Lake) and AMD Zen 4/5 are the current microarchitectures
- Server: Intel Sapphire Rapids/Granite Rapids, AMD EPYC Genoa/Turin

### ARM (ARMv8 / ARMv9 / AArch64)

ARM Holdings designs the ISA and licenses it; dozens of companies implement their own microarchitectures. The licensing model is why you see ARM in phones (Apple A-series, Qualcomm Snapdragon), servers (AWS Graviton, Ampere Altra), and embedded (Cortex-M).

Key facts for ML:
- **Apple M-series SoCs** (M1–M4) use ARMv8.5/v9 with Apple's own microarchitecture — and share memory between CPU and Neural Engine
- **AWS Graviton 3/4**: 64 Neoverse cores, 300 GB/s memory bandwidth — competitive for serving
- ARM's NEON (128-bit SIMD) and **SVE/SVE2** (Scalable Vector Extension — 128 to 2048-bit vectors, vector length determined at runtime) are the ML-critical extensions
- `bfloat16` native support added in ARMv8.6

### RISC-V (Open ISA)

RISC-V is an **open, royalty-free** ISA developed at UC Berkeley (2010). Anyone can implement it without paying licensing fees — a fundamental disruption to the ARM licensing model.

```text
RISC-V standard extensions (letters in ISA string RV64GC):
  I = Base integer (mandatory)
  M = Multiply/divide
  A = Atomic operations
  F = Single-precision float
  D = Double-precision float
  C = Compressed 16-bit instructions
  V = Vector extension (RISC-V V 1.0)
  B = Bit manipulation
  Zicsr = Control/status registers
```

Why it matters for ML:
- **Eliminates licensing fees** → lowers barriers for custom AI chip startups
- **Modular design** — add only the extensions you need (e.g., a custom ML extension `Xnn`)
- Already shipping: SiFive, Alibaba T-Head (C910), StarFive JH7110, ESWIN EIC7700X
- RISC-V V (vector) extension enables portable SIMD — same story as ARM SVE

---

## SIMD & Vector Extensions: The CPU Ancestor of GPU Parallelism

This is the section that matters most for ML practitioners working with CPU inference.

**SIMD** — **Single Instruction, Multiple Data** — means one instruction operates on a *vector* of values simultaneously. Instead of adding one pair of floats, add 8 or 16 at once. The hardware is literally 8 or 16 independent adders working in lockstep.

```text
Scalar (without SIMD):
  FADD xmm0, xmm1      ; add 1 × f32 = 32-bit lane

SSE (128-bit):
  ADDPS xmm0, xmm1     ; add 4 × f32 = 128-bit vector

AVX (256-bit):
  VADDPS ymm0, ymm1, ymm2   ; add 8 × f32 = 256-bit vector

AVX-512 (512-bit):
  VADDPS zmm0, zmm1, zmm2   ; add 16 × f32 = 512-bit vector
```

One instruction, 16 additions, executed in the same cycle (roughly). This is the CPU's answer to parallelism before you even leave a single core.

### x86 SIMD History

<div class="diagram">
<div class="diagram-title">x86 SIMD Extension Timeline</div>
<div class="flow-h">
  <div class="flow-node accent">MMX<br/><small>1996 · 64-bit</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">SSE–SSE4<br/><small>1999–2007 · 128-bit · FP32</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">AVX/AVX2<br/><small>2011–2013 · 256-bit · FP32/INT</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">AVX-512<br/><small>2017+ · 512-bit · masks</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange">VNNI / AMX<br/><small>2019+ · dot-product / tile matmul</small></div>
</div>
</div>

| Extension | Register width | FP32 lanes | INT8 lanes | Key ML relevance |
|---|:---:|:---:|:---:|---|
| SSE4.2 | 128-bit (XMM) | 4 | 16 | Legacy baseline |
| AVX / AVX2 | 256-bit (YMM) | 8 | 32 | Most x86 deployment target |
| AVX-512F | 512-bit (ZMM) | 16 | 64 | High-throughput FP32 inference |
| AVX-512 VNNI | 512-bit | — | 64 (w/ dot) | **INT8 dot-product for inference** |
| AMX-INT8 / AMX-BF16 | tile (1024-byte rows) | — | tile matmul | **Sapphire Rapids+ matrix units** |

### ARM SIMD: NEON and SVE

**NEON** (128-bit): standard SIMD for ARMv7/v8. All modern phones and Apple Silicon support NEON. 4× FP32 or 8× INT16 or 16× INT8 per instruction.

**SVE (Scalable Vector Extension)**: introduced in ARMv8.2, SVE registers are *not* a fixed width — the hardware declares a `VLEN` anywhere from 128 to 2048 bits. The same binary runs correctly on any SVE implementation; longer vectors give more lanes. AWS Graviton 3 uses 256-bit SVE; A64FX (Fujitsu, used in Fugaku supercomputer) uses 512-bit SVE.

```text
SVE vector length (VLEN) determines lanes:
  VLEN=256 → 8 × FP32 lanes   (like AVX2)
  VLEN=512 → 16 × FP32 lanes  (like AVX-512)
  VLEN=2048 → 64 × FP32 lanes (future hardware)
```

### The ML-Specific Extensions: VNNI and AMX

These are x86 ISA extensions added specifically because neural network inference is dominated by dot products on low-precision integers:

**AVX-512 VNNI** (Vector Neural Network Instructions, Intel Cascade Lake 2019+):

```text
VPDPBUSD zmm0, zmm1, zmm2
  ; Computes: zmm0 += zmm1[i] (UINT8) × zmm2[i] (INT8), 4 elements at a time
  ; 512-bit registers → 64 INT8 multiply-accumulates in ONE instruction
  ; This is exactly the quantized dot product in INT8 GEMM
```

**AMX** (Advanced Matrix Extensions, Intel Sapphire Rapids 2023+): full 2D tile registers (up to 16 rows × 64 bytes each, ~1024 bytes per tile). `TMUL` instructions multiply two tiles — essentially a tiny matrix unit built into the CPU.

```python
# Conceptual illustration: what VNNI does per clock in Python terms
import numpy as np

def vnni_dot_product(a_uint8: np.ndarray, b_int8: np.ndarray) -> np.ndarray:
    """
    Mimics AVX-512 VNNI: accumulate 4 INT8 pairs into one INT32 accumulator.
    Real VNNI processes 64 elements (512-bit / 8-bit) per instruction.
    
    a shape: (N,)  dtype=uint8   (activations, zero-point-shifted)
    b shape: (N,)  dtype=int8    (weights)
    returns: scalar int32 accumulation
    """
    assert a_uint8.dtype == np.uint8
    assert b_int8.dtype == np.int8
    # Widen to int32 before multiply to avoid overflow, then accumulate
    return np.sum(a_uint8.astype(np.int32) * b_int8.astype(np.int32))

# The key insight: 64 INT8 multiply-adds per AVX-512 cycle
# vs 16 FP32 multiply-adds per AVX-512 cycle
# → 4× more operations per cycle for INT8 vs FP32 on the same hardware
a = np.random.randint(0, 127, 64, dtype=np.uint8)
b = np.random.randint(-64, 63, 64, dtype=np.int8)
result = vnni_dot_product(a, b)
print(f"64 INT8 dot-product result: {result} (int32 accumulator)")
```

---

## Scalar vs SIMD: A Concrete Python Analogy

Here is the intuition for SIMD, expressed in NumPy terms (NumPy itself calls SIMD instructions under the hood):

```python
import numpy as np
import time

N = 1 << 22   # 4M elements

a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)

# --- "Scalar" loop (Python for-loop; no SIMD) ---
def scalar_add(a, b):
    out = np.empty_like(a)
    for i in range(len(a)):
        out[i] = a[i] + b[i]     # one float32 add per iteration
    return out

# --- SIMD (NumPy vectorized; maps to VADDPS on x86) ---
def simd_add(a, b):
    return a + b                  # one VADDPS zmm instruction = 16 float32 adds

t0 = time.perf_counter(); scalar_add(a, b); t1 = time.perf_counter()
t2 = time.perf_counter(); simd_add(a, b);   t3 = time.perf_counter()

print(f"Scalar loop:   {(t1-t0)*1000:.1f} ms")
print(f"NumPy SIMD:    {(t3-t2)*1000:.1f} ms")
print(f"Speedup:       ~{(t1-t0)/(t3-t2):.0f}×")
# Typical result: ~100–300× speedup
# The SIMD version also benefits from cache-friendly sequential memory access
```

The speedup is enormous because:
1. 16× more data processed per instruction (AVX-512 vs scalar)
2. The Python loop has interpreter overhead; NumPy eliminates it
3. SIMD also enables better memory access patterns

---

## The Same Computation: x86-64 vs ARM64

To see the ISA contrast concretely, here is a single float32 multiply-accumulate (the core of matmul) in both ISAs:

```asm
; === x86-64 (Intel syntax) ===
; Compute: result += a * b  (scalar f32)
; Assume: xmm0 = accumulator, xmm1 = a, xmm2 = b

vfmadd231ss xmm0, xmm1, xmm2    ; xmm0 += xmm1 * xmm2  (FMA: fused multiply-add)
; VFMADD231SS: 1 instruction, 1 f32 FMA, latency ~4 cycles, throughput 0.5 cycles
; (one new result every 0.5 cycles on Zen4 / Ice Lake with dual FMA units)

; AVX-512 version (16 floats at once):
vfmadd231ps zmm0, zmm1, zmm2     ; zmm0 += zmm1 * zmm2  (16 × f32 FMA)
; 16× the throughput of the scalar version in the same cycle count
```

```asm
; === ARM64 (AArch64) ===
; Compute: result += a * b  (scalar f32)
; Assume: s0 = accumulator, s1 = a, s2 = b

fmadd s0, s1, s2, s0             ; s0 = s1 * s2 + s0
; Clean load-store ISA: only FMADD touches registers
; No memory operands in arithmetic instructions (unlike CISC)

; NEON version (4 floats at once):
fmla v0.4s, v1.4s, v2.4s         ; v0 += v1 * v2  (4 × f32 FMA, 128-bit NEON)
; SVE version (VLEN/32 floats at once):
fmla z0.s, p0/m, z1.s, z2.s      ; z0 += z1 * z2  (predicated, all active lanes)
```

The computation is identical; the encoding and addressing philosophy differ. The RISC ARM version is simpler per instruction — but needs more instructions for complex memory addressing.

---

## VLIW: An Honorable Mention

**VLIW (Very Long Instruction Word)** is a third philosophy: pack *multiple independent operations* into one wide instruction word, letting the *compiler* (not hardware) figure out what can run in parallel.

```text
VLIW example (conceptual 128-bit wide instruction):
  [ADD R1, R2, R3] | [MUL R4, R5, R6] | [LOAD R7, [R8+4]] | [NOP]
  ^--- four slots, all execute in parallel in one cycle ---^
```

Hardware stays simple (no out-of-order engine needed); the compiler takes on the scheduling burden. Used in:
- **Intel Itanium** (IA-64): famously complex to compile for; failed in the market
- **TI C6000 DSP family**: successful in signal processing, video codecs
- **Older AI accelerators** (some vision DSPs): still appear in edge NPUs

VLIW is relevant to you because some DSPs and NPUs you'll target for edge inference (see [Chapter 18](./18_dsps.md)) use it, and their compilers expose this structure.

---

## Why This Matters for Model Optimization

<div class="diagram">
<div class="diagram-title">ISA Decisions → Your Optimization Choices</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">SIMD Lanes = CPU Throughput</div>
    <div class="card-desc">AVX-512 gives 16×FP32 or 64×INT8 per cycle. Libraries like oneDNN, OpenBLAS, and BLIS auto-select the widest available SIMD extension at runtime — which is why recompiling for a newer CPU often yields free speedups.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">VNNI / AMX = INT8 on CPU</div>
    <div class="card-desc">AVX-512 VNNI turns a modern CPU into a reasonable INT8 inference engine. Quantizing to INT8 (e.g., with Intel Neural Compressor or ONNX Runtime) exploits these instructions directly — 4× more MACs per cycle vs FP32.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">SIMD → SIMT: the GPU lineage</div>
    <div class="card-desc">GPU SIMT (Single Instruction, Multiple Threads) is conceptually SIMD scaled to thousands of lanes with thread-level addressing. Understanding SIMD first makes GPU warp execution (Ch 15) immediately intuitive.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">ISA Choice for Edge</div>
    <div class="card-desc">Deploying quantized models to ARM Cortex-A (NEON) vs Cortex-M (Helium/MVE)? The target ISA's SIMD width and INT8 support directly determines which quantization formats and kernels are worth using.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">RISC-V for Custom Silicon</div>
    <div class="card-desc">AI chip startups adding custom ML instructions extend RISC-V with custom extensions (Xnn). The open ISA eliminates licensing barriers — you'll see it in edge AI chips increasingly from 2025 onward.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Compile Targets</div>
    <div class="card-desc">torch.compile, XLA, and TVM all query the target ISA's features (via CPUID on x86, AT flags on ARM). Enabling AVX-512 or BF16 flags often unlocks faster kernel selection automatically.</div>
  </div>
</div>
</div>

### Checking Your ISA Extensions in Python

```python
import subprocess, platform

if platform.machine() in ("x86_64", "AMD64"):
    # Check which AVX extensions the current CPU supports
    try:
        import cpuinfo  # pip install py-cpuinfo
        info = cpuinfo.get_cpu_info()
        flags = info.get("flags", [])
        for ext in ["avx", "avx2", "avx512f", "avx512vnni", "amx_int8", "amx_bf16"]:
            status = "✓" if ext in flags else "✗"
            print(f"  {status} {ext}")
    except ImportError:
        # Fallback: read /proc/cpuinfo
        with open("/proc/cpuinfo") as f:
            cpuflags = next(l for l in f if l.startswith("flags"))
        for ext in ["avx", "avx2", "avx512f", "avx512_vnni"]:
            print(f"  {'✓' if ext in cpuflags else '✗'} {ext}")

# PyTorch also exposes this:
import torch
print(f"\ntorch.backends.cpu.get_cpu_capability(): {torch.backends.cpu.get_cpu_capability()}")
# Returns e.g. "AVX512" or "AVX2" — used internally to select kernel variants
```

Knowing your CPU's SIMD capability lets you reason about whether oneDNN will use AVX-512 VNNI kernels for INT8 and how much speedup to expect from quantization on that box.

---

## Key Takeaways

- An **ISA** is the hardware/software contract — the visible instruction set, encoding, and register file — while the **microarchitecture** is the implementation. The same ISA can be implemented by many microarchitectures (Intel vs AMD, all x86-64).
- **x86-64** is CISC in encoding but translates to internal RISC-like micro-ops for execution. **ARM64** and **RISC-V** are load-store RISC ISAs with fixed-width encodings — simpler decoders, better suited to many-core scaling.
- **SIMD extensions** (SSE/AVX/AVX-512, NEON/SVE) let one instruction process a *vector* of values — 4×, 8×, 16×, or more — per cycle. This is the CPU-level form of data parallelism, the ancestor of GPU SIMT.
- **VNNI** (AVX-512) adds dedicated INT8 dot-product instructions and is the reason INT8 CPU inference can be 4× faster than FP32 on modern x86 servers.
- **AMX** (Intel Sapphire Rapids+) brings tile-level matrix units into x86 CPUs — blurring the CPU/accelerator boundary for inference.
- **RISC-V**'s open ISA eliminates licensing fees, enabling a wave of custom AI chips with proprietary ML extensions.
- Every framework optimization (oneDNN, XLA, TVM, torch.compile) ultimately targets one of these ISAs and selects the widest available vector extension at runtime.

---

*Next: [Chapter 6 — Languages & Nomenclature of the Stack](./06_languages_and_nomenclature.md), where we trace the full journey from Python down to transistors — assembly, Verilog, CUDA, Triton, and every layer in between.*

[← Back to Table of Contents](./README.md)
