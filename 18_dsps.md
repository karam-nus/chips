---
title: "Chapter 18 — DSPs (Digital Signal Processors)"
---

[← Back to Table of Contents](./README.md)

# Chapter 18 — DSPs (Digital Signal Processors)

Before there were NPUs, before there were GPU tensor cores, before "AI accelerator" was a job title, there was the **digital signal processor** — a chip designed from the ground up to do one thing brilliantly: consume an endless stream of numbers, apply a repeating mathematical transform, and produce an output stream in real time, every cycle, without ever missing a sample. DSPs power the microphone in your earbuds, the radio in your phone, the radar in your car, the sonar in a submarine, and the wake-word detector that has been listening for "Hey [voice assistant]" since the moment you plugged in your device.

The DSP's design philosophy — single-cycle multiply-accumulate, deterministic latency, fixed-point arithmetic, SIMD with circular addressing — is the direct ancestor of every NPU discussed in the previous chapter. Understanding the DSP's architecture explains not only why it exists as a separate chip from the CPU, but also why "quantization" in neural networks is really just the AI industry rediscovering what signal processing engineers figured out in the 1970s.

> **The one-sentence version:** A DSP is a fixed-point SIMD processor with a single-cycle MAC, VLIW instruction issue, hardware-circular memory addressing, and saturating arithmetic — all tuned for real-time repetitive numerical transforms at milliwatt power with deterministic, zero-jitter latency.

---

## What Makes a DSP Different from a CPU?

A general-purpose CPU is an optimistic machine: it assumes workloads are varied, branches are unpredictable, memory access patterns are irregular, and the program counter jumps around. It spends enormous silicon area on **out-of-order execution, branch prediction, a large speculative reorder buffer, and a deep cache hierarchy** — all to handle unpredictability gracefully (see [Chapter 14 — CPUs](./14_cpus.md)).

A DSP starts from the opposite assumption: **the algorithm is known, the data arrives in a regular stream, and the same mathematical operation runs thousands to millions of times on consecutive samples.** The DSP therefore eliminates the machinery for handling unpredictability and replaces it with machinery for sustaining maximum throughput on regular arithmetic:

<div class="diagram">
<div class="diagram-title">CPU vs DSP Design Philosophy</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">CPU (e.g., Cortex-A78)</div>
    <ul>
      <li>Out-of-order, speculative execution</li>
      <li>Branch predictor (2K+ entry)</li>
      <li>Large L1/L2/L3 cache (1–32 MB)</li>
      <li>Unified address space, virtual memory</li>
      <li>FP32/FP64 floating point</li>
      <li>General SIMD (NEON, AVX)</li>
      <li>Unpredictable latency (cache miss = stall)</li>
      <li>Optimized for IPC on mixed workloads</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">DSP (e.g., TI C66x)</div>
    <ul>
      <li>In-order, statically scheduled (VLIW)</li>
      <li>No branch predictor (branches minimized)</li>
      <li>Scratchpad SRAM (not cache; no misses)</li>
      <li>Harvard architecture (separate I/D buses)</li>
      <li>Fixed-point Q-format arithmetic</li>
      <li>Specialized SIMD + circular addressing</li>
      <li>Deterministic, cycle-accurate latency</li>
      <li>Optimized for throughput on one algorithm</li>
    </ul>
  </div>
</div>
</div>

---

## Core DSP Architectural Hallmarks

### 1. The Hardware MAC: Single-Cycle Multiply-Accumulate

The innermost operation of nearly every signal processing algorithm is the **multiply-accumulate**:

$$y[n] = y[n-1] + x[k] \cdot h[n-k]$$

This appears in FIR filters, IIR filters, DFT/FFT butterfly stages, correlation, convolution, Viterbi decode, matrix multiply — everywhere. On a CPU without a dedicated MAC unit, this requires two instructions (MUL + ADD) and uses a general ALU that is designed for address computation and control flow, not sustained arithmetic throughput.

A DSP has a **dedicated MAC unit** that performs the full `acc += a * b` in a **single clock cycle**:

```text
        ┌────────┐    ┌────────┐
        │   a    │    │   b    │   ← two 16-bit (or 32-bit) operands
        └───┬────┘    └────┬───┘
            │             │
            └──────┬───────┘
                   │
           ┌───────▼───────┐
           │  MULTIPLIER   │      ← combinational logic; result in same cycle
           │  16×16 → 32   │
           └───────┬───────┘
                   │
           ┌───────▼───────┐
           │   ADDER/ACC   │      ← adds to accumulator register
           └───────┬───────┘
                   │
           ┌───────▼───────┐
           │  Accumulator  │      ← 40-bit extended accumulator to prevent overflow
           └───────────────┘
```

Key design detail: the accumulator is **wider than the operand word**. A 16-bit × 16-bit multiply produces a 32-bit result; accumulating N such products can overflow a 32-bit register after ~2¹⁶ = 65,536 additions. DSPs use a **40-bit** (or 64-bit) extended accumulator so that thousands of MACs can execute before the programmer needs to re-normalize. This is the hardware equivalent of FP32's larger dynamic range when used as an accumulator for INT8 multiplications — the same idea in [Chapter 16's TPU discussion](./16_tpus_and_systolic_arrays.md) where INT8 inputs produce FP32/INT32 accumulators.

### 2. VLIW: Very Long Instruction Word

Because the DSP's workload is regular and the compiler knows the schedule in advance, a DSP replaces the expensive **out-of-order execution engine** with a **Very Long Instruction Word (VLIW)** pipeline.

A VLIW instruction encodes multiple independent operations in parallel — one word, many execution units:

```text
┌──────────────────────────────────────────────────────────────────────┐
│  256-bit VLIW word (TI C66x example)                                 │
│                                                                      │
│  [ .L1 unit op ] [ .S1 unit op ] [ .M1 unit op ] [ .D1 unit op ]   │
│  [ .L2 unit op ] [ .S2 unit op ] [ .M2 unit op ] [ .D2 unit op ]   │
│                                                                      │
│  .Mx = multiplier (MAC);  .Lx = ALU;  .Sx = shift/branch;  .Dx = load/store │
└──────────────────────────────────────────────────────────────────────┘
```

The TI C66x (used in baseband modems, radar, etc.) can execute **8 MACs per cycle** by having two multiplier units (`.M1`, `.M2`) each doing 4× 16-bit MACs simultaneously (via SIMD), issuing 8 SIMD-packed MACs per 256-bit VLIW word. At 1.2 GHz: **9.6 GMAC/s** from a single C66x core.

**The trade-off:** VLIW requires the **compiler** to schedule operations into the correct slots — there is no out-of-order hardware to re-schedule at runtime. If the compiler can't find enough independent instructions to fill all slots, it inserts NOPs (no-operation). DSP compilers (and skilled DSP assembly programmers) spend considerable effort on **loop unrolling**, **software pipelining**, and **modulo scheduling** to sustain full utilization.

### 3. Harvard Architecture: Separate Instruction and Data Memories

A **Harvard architecture** (see [Chapter 4 — Anatomy of a Processor](./04_anatomy_of_a_processor.md)) uses **physically separate buses** for instructions and data, allowing the CPU to **fetch the next instruction and load data simultaneously** — one of the key throughput bottlenecks in von Neumann (shared-bus) architectures is eliminated.

DSPs extend this further with a **dual-data bus** (or quad-data bus in advanced DSPs): two or four data buses allowing the MAC to fetch both operands (`a` and `b`) in the same cycle from two different memory banks:

```text
Program Memory ──────► Instruction Fetch ──────► VLIW Decode
                              │
Data Memory Bank A ─────────► ┐
                               ├───► MAC (a × b + acc)
Data Memory Bank B ─────────► ┘
```

For a FIR filter, operand A comes from the circular input sample buffer, operand B from the coefficient array — both loaded in a single cycle, every cycle. This is what allows the MAC unit to be continuously fed without stalls.

### 4. Circular (Modulo) Buffer Addressing

DSP algorithms are inherently **sliding-window** computations. A FIR filter of length N maintains a buffer of the last N input samples; each new sample shifts the window by 1. Naively, this requires either:

- **Copying** the buffer on each sample (O(N) operations — terrible), or
- **Pointer arithmetic** with wrap-around modulo N.

DSPs implement **hardware circular addressing**: a dedicated address register auto-increments each cycle and **automatically wraps back to the start of the buffer** when it reaches the end — no branch, no modulo instruction, no comparison. The wrap-around logic is in the address generation unit (AGU), not the main pipeline.

```text
Sample buffer [x_0, x_1, x_2, ..., x_{N-1}] in Data Memory Bank A
                ^
                └── circular pointer (auto-wraps from x_{N-1} back to x_0)
                    increments every cycle for free
```

For a 64-tap FIR filter, 64 MAC operations execute in 64 cycles with zero overhead for pointer management — the AGU handles it transparently. On a CPU, the equivalent modulo operation would cost 1–3 extra cycles per sample.

### 5. Bit-Reversed Addressing for FFT

The Fast Fourier Transform (FFT) accesses memory in a **bit-reversed** pattern during the input permutation stage. For an $N$-point FFT, index $k$ maps to the integer whose binary representation is the bit-reversal of $k$'s binary representation.

A CPU handles this with a lookup table or explicit bit-reversal instructions. DSPs provide **hardware bit-reversed addressing**: the AGU has a mode where incrementing the address register applies bit-reversal automatically, making the FFT input-permutation stage a series of load/store operations with zero overhead.

```python
def bit_reverse(n: int, bits: int) -> int:
    """Compute bit-reversal of n using 'bits' total bits."""
    result = 0
    for _ in range(bits):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result

# FFT bit-reversal permutation — on a DSP this is free (hardware AGU)
# On a CPU this costs ~3–5 cycles per element
import numpy as np
N = 8
indices = [bit_reverse(k, 3) for k in range(N)]
print(f"FFT bit-reversed order for N=8: {indices}")
# [0, 4, 2, 6, 1, 5, 3, 7]
```

### 6. Saturating Fixed-Point Arithmetic

CPUs use **wrapping arithmetic** for integers: `127 + 1 = -128` in a signed 8-bit register. For signal processing this is catastrophic: an audio sample that slightly overflows will produce a large negative value — a loud click — instead of clipping to the maximum value.

DSPs implement **saturating arithmetic** as a hardware mode:

- **Saturating add:** if the result exceeds the maximum value for the bit width, it clamps to the max (or min for underflow). No wrap-around.
- **Saturating shift:** right-shift operations can saturate instead of producing garbage on overflow.

```python
import numpy as np

def saturating_add_q15(a: int, b: int) -> int:
    """
    Q15 saturating addition: values in [-32768, 32767].
    Models what a DSP's MAC unit does in Q15 fixed-point mode.
    """
    result = a + b
    if result > 32767:
        return 32767    # saturate high
    elif result < -32768:
        return -32768   # saturate low
    return result

# CPU (wrapping) vs DSP (saturating) — an audio example
a_cpu  = np.int16(30000)
b_cpu  = np.int16(10000)
print(f"CPU int16 wrap:       {np.int16(a_cpu + b_cpu)}")     # → -25536  ← wrong!
print(f"DSP saturate Q15:     {saturating_add_q15(30000, 10000)}")  # → 32767  ← correct (clipped)
```

See [Chapter 3 — Number Representation](./03_number_representation.md) for the Q-format fixed-point system.

### 7. Fixed-Point Q-Format Arithmetic

DSPs operate in **Q-format fixed-point** rather than floating point. A Q-format number represents a fractional value by choosing a fixed position for the binary point:

| Format | Integer bits | Fractional bits | Range | Resolution |
|:------:|:------------:|:---------------:|:-----:|:----------:|
| Q15 | 1 (sign) | 15 | [−1, 1) | 2⁻¹⁵ ≈ 3.05×10⁻⁵ |
| Q31 | 1 (sign) | 31 | [−1, 1) | 2⁻³¹ ≈ 4.65×10⁻¹⁰ |
| Q8.8 | 8 | 8 | [−128, 128) | 2⁻⁸ ≈ 3.9×10⁻³ |
| Q1.15 (== Q15) | 1 | 15 | [−1, 1) | 3.05×10⁻⁵ |

```python
def q15_multiply(a: int, b: int) -> int:
    """
    Q15 × Q15 multiplication.
    Both inputs in Q15 format (values represent a/32768 and b/32768).
    Product is in Q30 format; shift right by 15 to normalize to Q15.
    The 40-bit extended accumulator on a DSP prevents intermediate overflow.
    """
    product = a * b          # Q30 intermediate: -2^30 to ~2^30
    result = product >> 15   # normalize to Q15
    return saturating_add_q15(result, 0)   # clamp if needed

# Example: 0.5 × 0.75 = 0.375
a_q15 = int(0.5 * 32768)    # 16384
b_q15 = int(0.75 * 32768)   # 24576
r_q15 = q15_multiply(a_q15, b_q15)
print(f"Q15 result: {r_q15} → float: {r_q15 / 32768:.5f}")   # → 0.37500
```

**The connection to neural-network quantization:** INT8 quantization in a neural network is exactly Q-format fixed-point with a learned scale factor per layer or per channel. When you call `torch.quantize_per_tensor(x, scale=0.1, zero_point=0, dtype=torch.qint8)`, you are creating a Q-format representation. DSP engineers have been doing this since the 1970s — the AI hardware boom rediscovered it with per-channel and per-group granularity.

---

## FIR Filter: The DSP Hello World

A finite impulse response (FIR) filter is the canonical DSP algorithm — and an excellent proxy for understanding convolution in neural networks. Given input samples $x[n]$ and filter coefficients $h[k]$ ($k = 0, \ldots, N-1$):

$$y[n] = \sum_{k=0}^{N-1} h[k] \cdot x[n-k]$$

This is a **1-D convolution** — the same operation as a 1-D `nn.Conv1d` in PyTorch with `kernel_size=N`, `stride=1`, `padding=N-1`. The DSP's inner loop:

```python
import numpy as np

def fir_filter_python(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Direct-form FIR filter in Python (Q15 integer simulation).
    x: input samples, shape (L,), dtype int16 (Q15 values in [-32768, 32767])
    h: filter coefficients, shape (N,), dtype int16 (Q15)
    Returns y: filtered output, shape (L,), dtype int16
    
    On a DSP:
      - h[] is loaded into a circular buffer (hardware AGU wraps automatically)
      - each iteration: one MAC instruction (1 cycle on a 16-bit DSP)
      - total: N cycles per output sample (N = filter order)
    """
    N = len(h)
    L = len(x)
    y = np.zeros(L, dtype=np.int32)    # 32-bit accumulator (extended)
    
    for n in range(L):
        acc = 0    # 40-bit accumulator on hardware; int32/64 in software sim
        for k in range(N):
            idx = n - k
            sample = int(x[idx]) if idx >= 0 else 0   # zero-padding for causality
            coeff  = int(h[k])
            acc   += sample * coeff    # Q15 × Q15 → Q30 partial product
        # Normalize from Q30 back to Q15 by right-shifting 15 bits
        y[n] = np.clip(acc >> 15, -32768, 32767)
    
    return y.astype(np.int16)

# Generate a noisy signal: 440 Hz tone + noise at Q15 scale
fs = 8000      # 8 kHz sample rate (telephone audio)
t  = np.arange(512) / fs
signal = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
noise  = np.random.randint(-3000, 3000, 512, dtype=np.int16)
x_noisy = np.clip(signal.astype(np.int32) + noise.astype(np.int32), -32768, 32767).astype(np.int16)

# Simple 8-tap averaging (box) filter as coefficients
N_taps = 8
h = np.full(N_taps, int(32768 / N_taps), dtype=np.int16)   # Q15 normalized

y_filtered = fir_filter_python(x_noisy, h)
print(f"Input  std: {x_noisy.std():.1f}   Output std: {y_filtered.std():.1f}")
# Smoothing reduces std — high-frequency noise attenuated ✓
```

**Parallelism on a DSP:** With VLIW + SIMD (e.g., 4× 16-bit MAC per instruction), a DSP can compute **4 MACs per cycle** on vectorized input data. The TI C66x's two `.M` units each do `4×16b×16b` SIMD, giving **8 MACs/cycle** per core. For an 8-tap filter: 1 cycle if 8 MACs fit in one VLIW packet (with full SIMD utilization). For a 256-tap lowpass filter: 32 cycles per output sample at 8 MACs/cycle.

At 1.2 GHz and 8 MACs/cycle: **9.6 GMAC/s = 19.2 GOPS INT16** — comparable to a small NPU, achieved on a 0.5–1 W DSP die.

---

## Example Real-World DSPs

### Qualcomm Hexagon

The Qualcomm **Hexagon** processor is the most prominent DSP in consumer electronics — present in every Snapdragon SoC. It began as a pure DSP for the modem and audio subsystem, then expanded:

<div class="diagram">
<div class="diagram-title">Qualcomm Hexagon Evolution</div>
<div class="flow-h">
  <div class="flow-node accent">Hexagon V1–V3<br/><small>VLIW DSP for modem/baseband</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">Hexagon V5/V6<br/><small>+ HVX (128-byte SIMD) for computer vision</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">Hexagon V65+<br/><small>+ Hexagon NN: first neural-net ops</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">Hexagon HTP (v73/v75)<br/><small>Dedicated INT8/INT4 NPU tensor core</small></div>
</div>
</div>

The **Hexagon Vector eXtensions (HVX)** are 128-byte (1024-bit) SIMD registers — extremely wide for DSP standards. A single HVX instruction can perform **128 × 8-bit MACs** in one cycle (since each byte element contributes one MAC). At 1 GHz: **128 GOPS INT8** from the HVX unit alone.

This explains why many "NPU" benchmarks for Snapdragon include both the dedicated HTP NPU and the Hexagon HVX as shared contributors to AI throughput.

### Texas Instruments C6000 / C7x

TI's **C6000 family** (C62x, C64x+, C66x, C674x) is the standard bearer for traditional DSPs in industrial, automotive, and baseband applications:

| Core | Year | MACs/cycle | Word width | Notes |
|:----:|:----:|:----------:|:----------:|-------|
| C62x | 1997 | 2 (16-bit) | 32 | First C6000 |
| C64x+ | 2002 | 8 (16-bit) | 32 | 8 functional units |
| C66x | 2011 | 8 (16-bit) or 4 (32-bit) | 32/64 | Floating point added |
| C7x | 2018 | 512 (8-bit) | 512-bit vector | ANN/AI workloads |

The **C7x** is a dramatic evolution: its 512-bit vector unit performs **512 INT8 MACs per cycle** at 1 GHz = **512 GOPS INT8** — firmly in NPU territory. TI's MMA (Matrix Multiply Accelerator) extension adds an 8×8 INT8 systolic-style array to the C7x, making it a DSP-NPU hybrid. Modern TI AI-capable SoCs (TDA4VM for automotive ADAS) package C7x cores with dedicated deep-learning accelerators and MMA units.

### CEVA DSPs

**CEVA** is an IP licensing company (like ARM) that licenses DSP cores to SoC vendors. Their relevance to edge AI:

| Core | Target | Notes |
|:----:|:------:|-------|
| CEVA-BX2 | IoT/wearables | Ultra-low power audio/sensor DSP |
| CEVA-X4 | Mobile multimedia | LTE/5G baseband, audio |
| CEVA-NeuPro | AI/ML inference | INT8 systolic array, licensable NPU |

CEVA's **NeuPro** is a licensable NPU IP block derived from their DSP lineage — used by several embedded-vision SoC vendors. It illustrates the DSP→NPU design convergence.

### Cadence Tensilica HiFi DSP

The **Tensilica HiFi** (now Cadence) DSP family is the dominant audio DSP architecture. Found in:

- Amazon Echo (Alexa wake word)
- Google Nest (audio front-end, keyword spotting)
- Apple AirPods (ANC, transparency mode compute)
- Nearly all TWS (true wireless stereo) earbuds

| Core | MIPS | MACs/cycle | Power | Use case |
|:----:|:----:|:----------:|:-----:|---------|
| HiFi 1 | 400 | 2 (32-bit) | ~5 mW | Legacy audio |
| HiFi 4 | 600 | 8 (32-bit) | ~10 mW | High-quality audio, NN |
| HiFi 5 | 1000 | 8+8 (32-bit) | ~15 mW | Voice/NLP inference |

The **HiFi 4** added INT8 NN operations, making it capable of running small keyword-spotting models (CRNN, RNN-T stub) at the 10–15 mW power level required for always-on earbuds. The trade-off: HiFi is word-oriented (processes 32-bit audio samples), not image-oriented — it is slower at INT8 image convolution than an NPU MAC array but vastly lower power.

---

## DSP vs CPU vs NPU: A Comparison Table

| Trait | CPU (Cortex-A78) | DSP (TI C66x) | NPU (Apple ANE) |
|-------|:----------------:|:-------------:|:---------------:|
| Architecture | OOO superscalar | VLIW in-order | Systolic dataflow |
| Primary dtype | FP32/FP64 | INT16/INT32 (fixed-point) | INT8/INT4 |
| MAC throughput | ~0.1 GMAC/s/core | ~9.6 GMAC/s/core | ~19 TOPS (38 TOPS total) |
| Power | 1–3 W/core | 0.5–1 W/core | ~2 W total |
| TOPS/W | ~0.1 | ~10 | ~19 |
| Memory | Cache (unified) | Scratchpad SRAM (Harvard) | Scratchpad SRAM |
| Latency determinism | Low (cache miss risk) | High (scratchpad) | High (compiled schedule) |
| Best workload | General / irregular | Streaming signal processing | Dense NN inference (matmul) |
| Programmability | C/C++, any OS | C/DSP intrinsics, RTOS | Via Core ML / NNAPI only |
| Typical location | SoC application CPU | SoC modem/audio/sensor | SoC AI cluster |

---

## DSP Workloads Relevant to AI/ML

### Always-On Keyword Spotting

The most important DSP ML workload by deployment count: a tiny keyword-spotting model (< 100 KB INT8) runs **24/7** on the HiFi or Hexagon DSP while the application CPU sleeps:

```text
Pipeline:
  Microphone → ADC → DSP audio front-end (noise suppression, beamforming)
             → MFCC feature extraction (40 filter bank coefficients, 25ms windows)
             → Tiny RNN/CNN model (4 INT8 layers, ~50K parameters)
             → Probability of wake word
             → If p > threshold: wake application CPU
```

Power breakdown on a Cortex-A55 + HiFi 4 combo:
- CPU sleeping: ~1 mW
- HiFi 4 audio front-end: ~8 mW
- HiFi 4 NN inference (5 fps): ~2 mW
- Total: **~11 mW** continuous — achieves >24 h on a 500 mAh battery.

Running the same pipeline on the Cortex-A55 would cost ~500 mW — a 45× power penalty.

### Baseband / Modem

Every 4G/5G modem contains one or more DSP cores running channel coding (Turbo/LDPC decode), channel estimation, OFDM FFT/IFFT, MIMO beamforming — operations that are massive, structured, and real-time. The Qualcomm Hexagon V6x in the Snapdragon modem handles terabytes of numerical transforms per second while consuming ~500 mW.

### Radar Signal Processing (FMCW Radar)

Automotive radar (77 GHz FMCW) produces **range-Doppler cubes**: a 3-D tensor of shape `(chirps, receivers, range_bins)` that must be processed by 3-D FFT, CFAR detection, and clustering — all in real time at 10–20 frames per second. The TI AWR1843 radar SoC contains two C674x DSP cores specifically for this pipeline.

The 3-D FFT maps naturally to the DSP's efficient FFT support (bit-reversed addressing, butterfly SIMD) and the fixed-point representation (Q15/Q31 for the I/Q radar samples). **Modern radar AI** (CFAR replaced by a CNN for detection, or velocity estimation with a small network) runs on the same DSP via INT8 quantized inference — the DSP naturally handles both.

### Audio Codec and Noise Cancellation

Active noise cancellation (ANC) in earbuds runs a **feedback/feedforward FIR+IIR filter system** with coefficients updated thousands of times per second via adaptive filtering algorithms (LMS, RLS). This requires sustained, single-cycle MAC execution — the HiFi DSP is the standard choice for sub-20 mW ANC.

---

## The DSP ↔ NPU Convergence

The boundary between "DSP" and "NPU" has been eroding since ~2017. The trend:

<div class="diagram">
<div class="diagram-title">DSP → NPU Convergence Timeline</div>
<div class="flow">
  <div class="flow-node accent wide">1980s–2000s: Pure DSPs — FIR filters, FFTs, baseband modems</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">2010–2016: DSPs grow wider SIMD (HVX 1024-bit) — handle image resize, early CV</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">2017–2019: DSPs add INT8 NN ops (HiFi 4 NN, Hexagon NN) — run small CNNs</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">2020–2022: Dedicated MAC arrays added alongside DSP (Hexagon HTP, TI MMA) — hybrid</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">2023+: NPU = first-class block; DSP handles pre/post-processing; tightly coupled</div>
</div>
</div>

The modern approach: **DSP + NPU as co-processors**, not competitors. In a Snapdragon 8 Gen 3:

- The **Hexagon DSP** (HVX) handles: audio front-end, modem signal processing, sensor fusion, always-on wake word.
- The **Hexagon HTP** (NPU) handles: dense INT8/INT4 transformer inference, image classification, object detection.
- The **boundary** between what runs on HVX vs HTP is determined by the QNN compiler, based on arithmetic intensity and layer type.

This is identical to the observation in [Chapter 17 — NPUs & Edge AI](./17_npus_and_edge_ai.md): many "NPUs" are really enhanced DSPs, and many "DSPs" now run NN inference. The distinction is one of emphasis and origin, not a sharp architectural line.

---

## The Fixed-Point Heritage and Neural Network Quantization

The most important insight for an ML practitioner is that **neural-network quantization is not an AI-specific invention** — it is fixed-point signal processing, applied to weights and activations.

DSP engineers have used Q15, Q31, and custom Q-formats for 50 years to:
- Reduce memory by 2–4× compared to floating point.
- Achieve 4–16× higher MAC throughput with fixed-point hardware.
- Run deterministic real-time algorithms at low power.

The AI industry re-invented the same concepts under new names:

| DSP Term | Neural-Net Term | Same Idea? |
|:--------:|:---------------:|:----------:|
| Q-format fixed-point | Quantization | Yes |
| Scale factor per filter coefficient | Per-channel scale | Yes |
| Accumulator width extension (40-bit) | FP32 accumulator for INT8 MACs | Yes |
| Saturating arithmetic | Clamping to quant range | Yes |
| Coefficient quantization (Q15 weights) | Weight quantization (INT8) | Yes |
| Dynamic range scaling (gain normalization) | Zero-point shift | Yes |
| Block floating point (per-block exponent) | Block quantization (MXFP) | Yes (see Ch 23) |

Understanding this heritage explains *why* INT8 quantization preserves model accuracy as well as it does: the mathematical structure of neural networks (convolution, dot products, accumulated sums) maps almost perfectly onto the fixed-point framework that DSPs have executed reliably for decades.

```python
import numpy as np

# Demonstrating the equivalence of Q15 FIR and INT8 quantized Conv1d
# A 1-D convolutional layer with INT8 weights is mathematically a Q8 FIR filter

def conv1d_int8(x_q: np.ndarray, w_q: np.ndarray, x_scale: float, w_scale: float) -> np.ndarray:
    """
    INT8 quantized 1-D convolution (equivalent to a Q-format FIR filter).
    x_q: INT8 input, shape (L,)
    w_q: INT8 weights (filter coefficients), shape (K,)
    x_scale, w_scale: per-tensor scales
    Returns: FP32 output (dequantized)
    """
    L = len(x_q)
    K = len(w_q)
    output = np.zeros(L - K + 1, dtype=np.int32)  # INT32 accumulator (extended)
    
    for i in range(L - K + 1):
        acc = np.int32(0)
        for k in range(K):
            # INT8 × INT8 → INT32 accumulate: no overflow risk (max = 127*127*K ≤ 2^21 for K=1024)
            acc += np.int32(x_q[i + k]) * np.int32(w_q[k])
        output[i] = acc
    
    # Dequantize: multiply by both scales to get FP32 output
    return output.astype(np.float32) * x_scale * w_scale

# Test
x_fp32 = np.random.randn(64).astype(np.float32)
w_fp32 = np.random.randn(8).astype(np.float32)

x_scale = x_fp32.max() / 127.0
w_scale = w_fp32.max() / 127.0
x_q = (x_fp32 / x_scale).round().clip(-128, 127).astype(np.int8)
w_q = (w_fp32 / w_scale).round().clip(-128, 127).astype(np.int8)

y_int8 = conv1d_int8(x_q, w_q, x_scale, w_scale)
y_fp32 = np.convolve(x_fp32, w_fp32, mode='valid')

max_error = np.abs(y_int8 - y_fp32).max()
rel_error = max_error / np.abs(y_fp32).max()
print(f"Max absolute error: {max_error:.5f}")
print(f"Max relative error: {rel_error:.4%}")
# Typical: < 1% relative error — matches INT8 PTQ accuracy in neural nets
```

---

## Why This Matters for Model Optimization

**1. Always-on models must target DSP/MCU, not NPU.** For keyword spotting, anomaly detection, and sensor-triggered inference running 24/7 on battery, the DSP or MCU (Cortex-M + Ethos-U55) provides 10–100× better power efficiency than a full NPU. Use TFLite Micro or Arm Vela; quantize to INT8; keep models under 100 KB.

**2. Audio / speech front-ends are free.** The DSP's audio front-end (noise suppression, beamforming, MFCC) runs at near-zero marginal cost while the application CPU is sleeping. Model architectures that accept MFCC features (rather than raw waveforms) shift expensive pre-processing to the DSP, saving CPU and NPU cycles. This is why Whisper's mel-spectrogram front-end is efficient even on edge devices.

**3. Fixed-point quantization is not lossy mystery — it is 50-year-old engineering.** The error model for INT8 quantization is well understood: quantization noise has power $\sigma^2_q \approx \Delta^2 / 12$ where $\Delta = 2 \cdot x_{\max} / 255$ is the step size. This is exactly the quantization noise model from DSP textbooks. Understanding it helps you reason about *when* quantization error is acceptable (high-activation-variance layers, post-softmax) vs harmful (first/last layers, attention temperature).

**4. The accumulator width matters.** When you quantize a model to INT8 and run it on a DSP or NPU, the accumulator is INT32 (32 bits). This means you can accumulate up to $2^{32} / (127^2) \approx 266,000$ INT8 products before the accumulator overflows — far more than any practical layer depth. INT32 accumulators provide the safety margin that makes INT8 quantized inference reliable. Awareness of this lets you reason about whether intermediate rescaling (requantization) is necessary inside a layer.

**5. DSPs and NPUs will share your SoC.** On Snapdragon, Dimensity, and Renesas RZ/V devices, the DSP and NPU are on the same die and share the same LPDDR bus. A heavily-loaded DSP (audio processing) stealing bandwidth from LPDDR can reduce NPU effective throughput by 10–20%. Profile total SoC memory bandwidth, not just NPU TOPS, when benchmarking end-to-end inference pipelines.

---

## Key Takeaways

- A **DSP** is an in-order fixed-point processor with a single-cycle MAC, VLIW instruction issue, Harvard memory, circular/bit-reversed addressing, saturating arithmetic, and scratchpad SRAM — engineered for sustained throughput on repetitive signal processing at milliwatt power.
- The **FIR filter inner loop** is the archetypal DSP workload: N MACs per output sample, hardware circular addressing eliminates pointer overhead, and saturating accumulation prevents audio artifacts.
- **VLIW** encodes multiple execution units per instruction word — the compiler schedules, not the hardware — enabling multiple MACs per cycle with zero OOO hardware cost.
- **Q-format fixed-point** arithmetic is the direct predecessor of neural-network INT8 quantization; scale factors, extended accumulators, and saturating arithmetic are the same concepts under different names.
- Major DSPs: **Qualcomm Hexagon** (DSP-to-NPU evolution in Snapdragon), **TI C6000/C7x** (industrial/automotive/baseband), **Cadence HiFi** (always-on audio in earbuds/smart speakers), **CEVA** (licensable IP for IoT/mobile).
- The **DSP ↔ NPU boundary is dissolving**: modern SoCs integrate DSP + dedicated NPU MAC array as co-processors, with the compiler routing ops based on arithmetic intensity and operator type.
- For ML practitioners: DSPs power always-on inference at µW–mW scale; fixed-point heritage explains why INT8 quantization works; and understanding the DSP's accumulator model clarifies quantization error guarantees.

---

*Next: [Chapter 19 — ISPs](./19_isps.md), where we look at the Image Signal Processor — the dedicated pipeline that transforms raw sensor photons into the clean RGB frames your neural network actually sees.*

[← Back to Table of Contents](./README.md)
