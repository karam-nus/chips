---
title: "Chapter 17 — NPUs & Edge AI Accelerators"
---

[← Back to Table of Contents](./README.md)

# Chapter 17 — NPUs & Edge AI Accelerators

Every time your phone's camera decides it's looking at a face, or your earbuds transcribe speech, or a security camera distinguishes a person from a shrub, a tiny chip called a **Neural Processing Unit (NPU)** performs a real-time neural network inference — consuming only milliwatts, with no connection to a data center. The NPU is one of the fastest-growing categories in chip design: nearly every smartphone, laptop, automobile, and IoT device now integrates one. Yet for ML practitioners who work in the cloud, edge NPUs are often a black box with mystifying capability gaps.

This chapter tears that black box open. We start from the constraints that define edge AI — power, thermal, memory, latency — and derive why NPU architecture looks so different from a GPU or TPU. We then survey the major real-world NPUs (Apple, Qualcomm, Google, ARM, MediaTek, Intel, Renesas), examine the software frameworks that route work to them, and map everything back to your model-design decisions: why INT8 is the edge default, why small models matter, and why operator coverage determines whether your model runs on the NPU at all.

> **The one-sentence version:** An NPU is a low-power MAC array — often INT8 or INT4 — integrated into an SoC alongside the CPU and GPU, engineered from the start for quantized neural networks, where the primary goal is maximize TOPS/W rather than raw TOPS.

---

## The Edge Inference Constraint Set

A data-center accelerator operates at 700–1,400 W with liquid cooling and hundreds of gigabytes of HBM. An edge device is radically different:

<div class="diagram">
<div class="diagram-title">Edge vs Cloud Inference Constraints</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Data Center (GPU/TPU)</div>
    <ul>
      <li>Power budget: 700–1,400 W per accelerator</li>
      <li>Cooling: active air or liquid</li>
      <li>Memory: 80–432 GB HBM, 3–20 TB/s bandwidth</li>
      <li>Latency target: tens to hundreds of ms OK</li>
      <li>Models: FP32, FP16, bf16, FP8</li>
      <li>Reprogrammable, big SW stack (CUDA, JAX)</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Edge / Mobile / Embedded</div>
    <ul>
      <li>Power budget: 1–5 W (phone), 100 µW–500 mW (MCU/IoT)</li>
      <li>Cooling: passive only (no fan, no heatsink)</li>
      <li>Memory: 8–24 GB LPDDR5X on phones (~60–100 GB/s); laptops up to 128 GB at 120–550 GB/s</li>
      <li>Latency target: &lt;30 ms for real-time UX</li>
      <li>Models: INT8, INT4, sometimes INT2</li>
      <li>Fixed silicon, vendor SDK required</li>
    </ul>
  </div>
</div>
</div>

These constraints cascade into a specific set of hardware design choices:

1. **Power budget in mW → tiny MAC arrays (INT8, not FP16).** A FP16 multiplier uses roughly 4× the silicon area and ~3× the energy of an INT8 multiplier for the same bit-width result. At 1 W total SoC power, spending 500 mW on the NPU means ~500 GOPS INT8 is realistic; the same budget in FP16 would be ~125 GFLOP/s. Edge NPUs are quantization-first by design.

2. **No DRAM bandwidth → maximize on-chip SRAM.** LPDDR5 at 80 GB/s sounds fast, but an NPU doing 10 TOPS INT8 has an arithmetic intensity threshold of 10 TOPS ÷ 80 GB/s ≈ **125 INT8-ops/byte**. A MobileNetV3 layer has arithmetic intensity ~30–60; it is firmly memory-bound unless weights are cached on-chip. NPUs therefore spend a large fraction of their area on **SRAM buffers** (often 1–8 MB) to hold weights across many inference calls.

3. **Real-time latency → deterministic scheduling.** Unlike GPUs with their warp scheduler, most NPUs use a **compile-time static schedule**: the compiler tiles the network into operations and schedules each tile to run at a predetermined cycle, so there is zero runtime scheduling overhead and no cache-miss unpredictability.

4. **No reprogrammability → fixed operator support.** An NPU's instruction set covers a fixed set of operations: convolution, depthwise convolution, fully-connected, pooling, element-wise add, sigmoid, softmax. Any operation outside that set **falls back to the CPU or GPU**, often at 10–100× slower throughput. Operator coverage is the #1 practical gotcha in edge deployment.

---

## NPU Architecture Fundamentals

### The MAC Array Core

At the heart of every NPU is a 2-D array of **multiply-accumulate units**, conceptually similar to the systolic array in [Chapter 16 — TPUs & Systolic Arrays](./16_tpus_and_systolic_arrays.md) but much smaller:

- Typical sizes: 8×8 to 64×64 MAC cells (vs 256×256 in a TPU MXU).
- Each MAC cell does: `acc += weight * activation` per clock, where `weight` and `activation` are INT8 (or INT4), and `acc` is INT32.
- The array is organized as a **systolic or output-stationary dataflow**, depending on the vendor.

The key difference from a datacenter accelerator: the MAC array must sustain operation at <1 W, meaning fewer cells, lower clock frequency (often 500 MHz–1.5 GHz rather than 1.5–2.5 GHz for a GPU), and very tight power gating (cells that are not in use are clock-gated within microseconds).

### The SRAM Buffer Hierarchy

Because LPDDR access is both expensive (bandwidth-limited) and power-hungry, NPUs implement a **local SRAM hierarchy**:

```text
                 LPDDR5 (4–16 GB, 25–80 GB/s, 4–10 nJ/access)
                         │
              ┌──────────┴──────────┐
              │   NPU global SRAM   │  ← 1–8 MB, ~0.5 nJ/access
              │   (weight cache +   │
              │    activation buf)  │
              └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              │   MAC array local   │  ← 64–512 KB, ~0.05 nJ/access
              │   register file     │
              └─────────────────────┘
```

For a ResNet-50 with ~25 M parameters at INT8 (25 MB), the weights don't fit in 4 MB of SRAM; they are streamed layer by layer from LPDDR. But for a MobileNetV3-Small (2.5 M params = 2.5 MB INT8), the entire model fits in on-chip SRAM — allowing the NPU to run inference with **zero LPDDR accesses**, dropping power dramatically.

### Dataflow Types in NPUs

The dataflow strategy determines how weights, activations, and partial sums are scheduled through the MAC array:

| Dataflow | Reuse | Best for | Example |
|----------|-------|----------|---------|
| **Weight-stationary** | Weights stay in cells; activations pumped | Layers with large weight matrices | Google MXU, many ASICs |
| **Output-stationary** | Partial sums stay; weights + activations flow | Large output feature maps | Apple ANE, Arm Ethos |
| **Input-stationary** | Activations cached; weights rotate | Depthwise conv (low weight reuse) | Some mobile NPUs |
| **Row-stationary** | Maximizes overall reuse | Mixed workloads | Eyeriss academic |

Most production NPUs use **output-stationary** for image convolutions (many spatial positions, one output accumulator per position) and **weight-stationary** for dense fully-connected layers.

---

## Survey of Real-World Edge NPUs

### Apple Neural Engine (ANE)

Apple's Neural Engine debuted in the A11 Bionic (2017) for Face ID. By the A17 Pro (2023) and M4 (2024) it delivers **35–38 TOPS** (Apple does not publish its power; estimates put it at a few watts). Since 2025 Apple has also added **neural accelerators inside every GPU core** (A19 Pro, M5), so large on-device models increasingly run on the GPU while the ANE handles always-on and vision work.

| Chip | Year | ANE TOPS | Process | Notes |
|:----:|:----:|:--------:|:-------:|-------|
| A11 (iPhone X) | 2017 | 0.6 | TSMC 10nm | First ANE; Face ID only |
| A12 (iPhone XS) | 2018 | 5 | TSMC 7nm | CoreML accelerated |
| A15 (iPhone 13) | 2021 | 15.8 | TSMC 5nm | NLP + vision |
| A17 Pro (iPhone 15 Pro) | 2023 | 35 | TSMC N3B | On-device LLM; Apple's quoted TOPS here appear to count INT8, vs FP16 before |
| M4 (MacBook/iPad) | 2024 | 38 | TSMC N3E | |
| A20 Pro (iPhone 18 Pro) | 2026 | n/p | TSMC N2 (reported) | First 2 nm phone chip; AI work shared with GPU neural accelerators |

**Architecture:** Apple has not published ANE specifications. Patents and reverse engineering describe an INT8/FP16 MAC array with large dedicated on-chip SRAM and a compiler-scheduled dataflow, fed from the SoC's unified memory.

**Practical implication:** Core ML's model converter (coremltools) optimizes for the ANE at compile time. Operations that Core ML can express for the ANE run there; others fall back to the CPU/GPU. The key constraint is that **INT8-quantized models with standard layer types (Conv, Linear, LayerNorm, GELU) run on the ANE**; exotic custom operations typically do not.

### Qualcomm Hexagon (NPU / HTP)

Qualcomm's Snapdragon SoCs integrate the **Hexagon** block, which combines a DSP (see [Chapter 18 — DSPs](./18_dsps.md)), a vector extension (HVX), and since the Snapdragon 865 a dedicated **Hexagon NPU / Hexagon Tensor Processor (HTP)**.

| Snapdragon | Announced | AI milestone |
|:----------:|:---------:|-------|
| 865 | 2019 | Hexagon 698 with a tensor accelerator; "15 TOPS" for the whole AI Engine |
| 888 | 2020 | Fused scalar/vector/tensor Hexagon 780; 26 TOPS (AI Engine) |
| 8 Gen 2 | 2022 | INT4 support; Qualcomm stops quoting TOPS for phones |
| 8 Gen 3 | 2023 | On-device generative AI, models up to ~10B parameters in INT4 |
| 8 Elite / 8 Elite Gen 5 | 2024 / 2025 | Oryon CPU; NPU gains of ~40 % per generation (vendor) |
| X Elite / X2 Elite (PCs) | 2023 / 2025 | 45 TOPS / 80 TOPS INT8 NPU |

Qualcomm's "AI Engine" TOPS add up CPU, GPU and NPU, and newer phone chips are marketed by percentage gains instead, so cross-vendor phone TOPS comparisons are rarely like for like.

The Hexagon HTP is a **fixed-point SIMD array** (INT4, INT8, INT16) coupled with the DSP's VLIW scheduling. Software path: **Qualcomm AI Engine Direct SDK** → ONNX or TFLite models → compiled HTP binary. The key competitive advantage is **INT4 weight quantization**: Qualcomm was among the first to support 4-bit weights in production hardware, enabling Llama-2-7B inference on-device.

### Google Edge TPU

The **Edge TPU** (Coral) is a purpose-built ASIC from Google, available as a USB accelerator, PCIe M.2 card, or integrated in development boards. It implements a small version of the TPU systolic array concept.

- **Performance:** 4 TOPS INT8.
- **Power:** 2 W for 4 TOPS (2 TOPS/W, Google's figure).
- **Memory:** 8 MB on-chip SRAM — the *entire model must fit* in 8 MB (INT8 quantized), which is ~8 M parameters.
- **Compilation:** models must be fully INT8-quantized TensorFlow Lite files compiled with the **Edge TPU Compiler**; operations it cannot map run on the host CPU.
- **Limitation:** The 8 MB constraint is severe — only very small models (MobileNetV1 at 0.25 depth, EfficientNet-Lite0) fit entirely on-chip. Larger models are split: part runs on Edge TPU, part on CPU.

### ARM Ethos NPU

ARM licenses the **Ethos** NPU family to SoC vendors (MediaTek, NXP, Renesas, and others). Because ARM doesn't make chips, Ethos is an IP block that appears in hundreds of designs.

| Ethos Model | Peak (INT8) | Target |
|:-----------:|:----:|--------|
| Ethos-U55 | up to ~0.5 TOPS (32–256 MACs/cycle) | Cortex-M microcontrollers |
| Ethos-U65 | up to ~1 TOPS | Cortex-M and Cortex-A systems |
| Ethos-U85 (2024) | up to ~4 TOPS, with transformer operator support | Edge AI SoCs, Cortex-A and Cortex-M |

**Ethos-U55** sits next to a Cortex-M55 or M85 in the milliwatt "always-on" regime: keyword spotting, sensor anomaly detection, tiny vision models. Arm's earlier Ethos-N line for phones was discontinued; phone vendors build their own NPUs.

### MediaTek APU (AI Processing Unit)

MediaTek's Dimensity SoCs include the **APU (AI Processing Unit)**, used in many mid-range and flagship Android devices.

- Dimensity 9300 (2023, TSMC N4P): APU 790, built to run 7B-class LLMs in INT4 on-device; the Dimensity 9400 and 9500 followed in 2024–25.
- Architecture: clusters of INT4/INT8/FP16 MAC engines with a few MB of on-chip SRAM (MediaTek publishes few details).
- Software: **NeuroPilot SDK** → ONNX/TFLite.

### Intel NPU (Meteor Lake / Lunar Lake)

Intel added a dedicated NPU (formerly "VPU") to its laptop SoCs in Meteor Lake (2023). Microsoft's **Copilot+ PC** program (2024) set a 40 TOPS NPU floor for Windows AI features, which every PC vendor then designed to:

| Platform | NPU TOPS | CPU | GPU |
|:--------:|:--------:|:---:|:---:|
| Meteor Lake (Core Ultra) | 11.5 | Intel Core Ultra | Intel Arc |
| Lunar Lake (Core Ultra 200V) | 48 | Lion Cove / Skymont | Intel Arc (Xe2) |
| Panther Lake (Core Ultra series 3, Intel 18A, 2026) | ~50 | Cougar Cove / Darkmont | Intel Arc (Xe3) |
| AMD Ryzen AI 300 (XDNA 2, 2024) | 50 | Zen 5 / Zen 5c | Radeon 890M |
| Snapdragon X Elite / X2 Elite | 45 / 80 | Oryon | Adreno |

AMD's XDNA NPU descends from the AI Engine tiles of Xilinx's Versal FPGAs, a spatial array of VLIW vector processors.

The Intel NPU is implemented as a **multi-cluster VLIW DL accelerator** with INT8 and INT4 support. Software: **OpenVINO** and **ONNX Runtime** with the DirectML/NPU execution provider.

### Renesas DRP-AI

**Renesas** is a major embedded-systems semiconductor company (one of the two largest automotive MCU vendors, with Infineon) with a distinctive AI accelerator: the **DRP-AI (Dynamically Reconfigurable Processor + AI)**.

<div class="diagram">
<div class="diagram-title">Renesas DRP-AI Architecture</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🔄</div>
    <div class="card-title">DRP (Dynamically Reconfigurable Processor)</div>
    <div class="card-desc">Array of configurable cells reprogrammed per layer; excels at pre/post-processing (resize, normalize, NMS)</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">⚙️</div>
    <div class="card-title">AI MAC Array (DRP-AI)</div>
    <div class="card-desc">INT8 MAC array for Conv/Linear inference; the RZ/V2H's DRP-AI3 reaches 8 TOPS dense and up to 80 TOPS on pruned (sparse) models, ~10 TOPS/W by Renesas's figures</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🚗</div>
    <div class="card-title">Target Markets</div>
    <div class="card-desc">Industrial vision, automotive ADAS, robotics — not smartphones; emphasis on real-time, deterministic, low-power</div>
  </div>
</div>
</div>

Key Renesas DRP-AI products:

| SoC | Platform | DRP-AI class | CPU |
|:---:|:--------:|:-----------:|:---:|
| RZ/V2L | Embedded Linux | entry level (sub-1 TOPS class) | Cortex-A55 × 2 |
| RZ/V2M | Vision edge AI | ~1 TOPS class | Cortex-A53 × 2 |
| RZ/V2H | High-performance edge | DRP-AI3: 8 TOPS dense, up to 80 TOPS sparse | Cortex-A55 × 4 + Cortex-R8 × 2 + Cortex-M33 |

The DRP (non-AI part) is unusual: it can be reconfigured between frames to perform **image pre-processing** (Bayer demosaic, resize, color conversion) in hardware rather than burning CPU cycles — critical for vision pipelines where preprocessing and inference alternate 30–120 times per second.

Renesas also produces the widely-used **RA** and **RZ** MCU/MPU families:
- **RA series** (Cortex-M, no dedicated NPU): runs TensorFlow Lite Micro on the Cortex-M85 with Helium SIMD — appropriate for tiny "always-on" models (keyword spotting, anomaly detection).
- **RZ/V series** (Cortex-A + DRP-AI): real-time vision inference with the DRP-AI for industrial cameras and ADAS sensors.

**Software:** Renesas provides the **DRP-AI TVM** (a fork of Apache TVM) that compiles ONNX or PyTorch models to DRP-AI binaries. The DRP pre-processing pipeline is configured via the DRP library (C API).

---

## TOPS as a Metric — and Why TOPS/W Matters More

**TOPS (Tera-Operations Per Second)** is the headline number on every NPU data sheet. An operation is typically one INT8 multiply-accumulate (which has two arithmetic operations — a multiply and an add — so the metric is sometimes TOPS = 2× TMACS).

```python
# Computing TOPS for an NPU MAC array
mac_cells = 1024          # number of INT8 MAC units
clock_mhz = 1000          # MHz
ops_per_mac_per_cycle = 2 # one multiply + one add

tops = (mac_cells * clock_mhz * 1e6 * ops_per_mac_per_cycle) / 1e12
print(f"Peak TOPS: {tops:.3f}")  # → 2.048 TOPS
```

But TOPS alone is a misleading metric for three reasons:

1. **Utilization is rarely 100%.** Peak TOPS assumes the MAC array is always fed. Memory-bound layers (small batch, large weights) may achieve 10–30% utilization, meaning effective throughput is 1–6 TOPS even on a "20 TOPS" chip.

2. **INT4 TOPS ≠ INT8 TOPS.** Some vendors report INT4 TOPS (by packing two INT4 ops into one INT8 MAC lane). An "80 TOPS" INT4 chip may be "40 TOPS" INT8. Always check the datasheet's dtype qualifier.

3. **TOPS/W is what matters on-device.** At 1 W total system budget, a 10 TOPS chip at 10 TOPS/W is equivalent to a 40 TOPS chip at 10 TOPS/W — but they use different amounts of the total SoC power, leaving less for the CPU, modem, display, and radio. **TOPS/W (or equivalently, TOPS/mW for IoT) is the true edge efficiency metric.**

| NPU | Peak TOPS (INT8) | Power | TOPS/W | Power figure from |
|:---:|:-----------:|:-------------:|:------:|:----|
| Google Edge TPU | 4 | 2 W | 2 | Google datasheet |
| Renesas DRP-AI3 (RZ/V2H) | 8 dense / 80 sparse | — | ~10 | Renesas |
| Apple ANE (M4) | 38 | not published | — | — |
| Qualcomm Hexagon (X Elite) | 45 | not published | — | — |
| Intel NPU (Lunar Lake) | 48 | not published | — | — |
| AMD XDNA 2 (Ryzen AI 300) | 50 | not published | — | — |

Most vendors publish peak TOPS but not the NPU's own power, so TOPS/W comparisons between phone and laptop NPUs usually rest on estimates. When a vendor does publish efficiency, check whether it is measured at the dense INT8 rate or a sparse/INT4 "effective" rate: the RZ/V2H's 10× gap between its two TOPS figures is the example.

---

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Apple moved large-model work from the Neural Engine to neural accelerators inside the GPU, while PC vendors race to raise NPU TOPS. If decode is bound by memory bandwidth and prefill by compute, which unit should own each phase on a laptop, and does a separate NPU still make sense once the GPU has matrix units?</p>
</div>

## Quantization-First Design

Edge NPUs are not designed for FP32 — they are designed for **quantized inference**. The silicon implications:

### INT8: The Dominant Edge Dtype

An INT8 multiplier needs:
- **1-bit × 1-bit** = 1 AND gate. But **8-bit × 8-bit = 64 partial products**, requiring a tree of adders.
- Area: INT8 multiplier ≈ **1/9 the area** of an FP32 multiplier (8² vs 24² partial products, first order).
- Energy: INT8 multiply ≈ **1/18 the energy** of FP32 (0.2 pJ vs 3.7 pJ at 45 nm, Horowitz 2014).
- Bandwidth: INT8 weights use **1/4 the bandwidth** of FP32.

This is why a 1 W edge NPU can deliver 10+ TOPS INT8 while a 1 W CPU delivers only ~50 GFLOP/s FP32 — a **200× advantage** in raw throughput for the same power.

### INT4: The Emerging Edge Dtype

The Snapdragon 8 Gen 2 and MediaTek Dimensity 9300 support **INT4 weight quantization**: model weights stored as 4-bit integers (16 levels), dequantized to INT8 or FP16 on-the-fly for computation, or computed directly in INT4 MAC arrays. The bandwidth benefit: INT4 weights use **1/8 the bandwidth of FP32**, enabling larger models to fit in the same SRAM or to be streamed faster from LPDDR.

```python
import torch

# Simulating INT8 post-training quantization for edge deployment
def quantize_weight_int8(weight: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    """Symmetric per-tensor INT8 quantization."""
    w_max = weight.abs().max().item()
    scale = w_max / 127.0          # scale factor (float32)
    w_q = (weight / scale).round().clamp(-128, 127).to(torch.int8)
    zero_point = 0                 # symmetric → zero_point = 0
    return w_q, scale, zero_point

def dequantize_int8(w_q: torch.Tensor, scale: float) -> torch.Tensor:
    """Recover approximate FP32 values."""
    return w_q.float() * scale

# Example: a Linear layer weight
weight = torch.randn(256, 512)     # shape: [out, in]
w_q, scale, zp = quantize_weight_int8(weight)
w_dq = dequantize_int8(w_q, scale)

print(f"Weight dtype: {weight.dtype}, size: {weight.element_size() * weight.numel() / 1024:.1f} KB")
print(f"Quantized dtype: {w_q.dtype}, size: {w_q.element_size() * w_q.numel() / 1024:.1f} KB")
print(f"Max quantization error: {(weight - w_dq).abs().max().item():.5f}")
# Weight dtype: torch.float32, size: 512.0 KB
# Quantized dtype: torch.int8, size: 128.0 KB   ← 4× smaller
# Max quantization error: 0.00613
```

For INT4 weight-only quantization (AWQ, GPTQ-style):

```python
def quantize_weight_int4_grouped(weight: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Group-wise INT4 quantization (as used in AWQ/GPTQ for LLMs).
    weight: (out_features, in_features)
    group_size: number of input channels per quantization group
    Returns: (w_q as uint8 packed, scales as fp16)
    """
    out_f, in_f = weight.shape
    assert in_f % group_size == 0
    n_groups = in_f // group_size
    
    w_reshaped = weight.reshape(out_f, n_groups, group_size)
    w_max = w_reshaped.abs().max(dim=-1, keepdim=True).values
    scales = (w_max / 7.0).to(torch.float16)  # 4-bit signed → [-8, 7], use 7 for sym
    
    w_q = (w_reshaped / scales.float()).round().clamp(-8, 7).to(torch.int8)
    # In real hardware, two INT4 values are packed into one INT8 byte:
    # packed[i] = (w_q[2i] & 0xF) | ((w_q[2i+1] & 0xF) << 4)
    return w_q, scales

weight = torch.randn(4096, 4096)
w_q4, scales = quantize_weight_int4_grouped(weight, group_size=128)
print(f"FP32 size: {weight.numel() * 4 / 1e6:.1f} MB")
print(f"INT4 size: {weight.numel() * 0.5 / 1e6:.1f} MB")  # 2 values per byte
# FP32 size: 67.1 MB
# INT4 size:  8.4 MB  ← 8× compression
```

---

## The Software Stack: Who Routes to the NPU?

The software frameworks that sit between your PyTorch model and the NPU are a critical, often-painful layer. They implement **operator coverage** (what ops can run on the NPU), **model conversion** (translating PyTorch graphs to vendor-specific formats), and **fallback** (sending unsupported ops to CPU/GPU).

<div class="diagram">
<div class="diagram-title">Edge AI Software Stack</div>
<div class="flow">
  <div class="flow-node accent wide">PyTorch / TensorFlow model (FP32)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Quantization (PTQ / QAT): torch.ao, AIMET, bitsandbytes</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">Export: ONNX, TorchScript, TFLite, CoreML</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Vendor Compiler: Core ML Tools / QNN / OpenVINO / Vela / DRP-AI TVM</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Runtime: NNAPI, LiteRT, ExecuTorch, Core ML Runtime</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">NPU Hardware Execution (unsupported ops → CPU/GPU fallback)</div>
</div>
</div>

### Key Frameworks

**Core ML (Apple):** The only path to the Apple ANE. `coremltools.convert()` takes an ONNX or TorchScript model and converts to `.mlpackage` format. The Core ML compiler determines per-op whether to route to ANE, GPU, or CPU — this is **not user-controllable at the op level**. Models with standard quantized Conv/Linear/Attention ops route to the ANE; custom PyTorch ops or exotic activation functions often don't.

**LiteRT (Android):** **LiteRT** (formerly TensorFlow Lite) is the most common Android front end. Android's Neural Networks API (NNAPI) was **deprecated in Android 15** (2024); NPU access now goes through LiteRT delegates and vendor SDKs such as Qualcomm's QNN and MediaTek's NeuroPilot.

**ExecuTorch (PyTorch):** Meta's edge runtime, which targets Apple ANE (via Core ML backend), Qualcomm Hexagon (via QNN backend), and ARM CPUs. The workflow: `torch.export()` → `executorch.to_edge()` → vendor-specific backend compilation.

**OpenVINO (Intel):** Compiles ONNX/PyTorch models for Intel NPU, GPU, or CPU. Provides a unified API regardless of which execution unit runs the model.

**DRP-AI TVM (Renesas):** Apache TVM fork that targets the Renesas DRP-AI. Accepts ONNX or Relay IR, compiles to DRP-AI object files for deployment on RZ/V SoCs.

### Operator Coverage: The Practical Bottleneck

The most common edge-deployment failure mode is:

> "The NPU should be 20× faster than the CPU for this model, but profiling shows 80% of the time is spent on the CPU."

This happens because **unsupported operators fall back to CPU**. Common problem ops:

| Operator | Issue |
|----------|-------|
| Custom activation (Swish/SiLU in PyTorch) | May need decomposition to Sigmoid × input |
| Group normalization | Not on all NPU runtimes (Layer norm is more common) |
| Attention (SDPA / FlashAttention) | Often not a single NPU op; must be broken into BMM + Softmax |
| Dynamic shapes (variable seq len) | Static shapes required; must pad to fixed size |
| INT4 dequantize-then-matmul | Some NPUs do native INT4 matmul; others dequantize to INT8 first (slower) |
| Gather / scatter / non-contiguous indexing | Usually no NPU support; CPU only |

**The debugging workflow:**

```python
# ExecuTorch: checking which ops are delegated to the NPU vs CPU
import torch
from executorch.exir import to_edge
from executorch.backends.qualcomm.qnn_backend import QnnBackend

model = MyModel().eval()
example_inputs = (torch.randn(1, 3, 224, 224),)
exported = torch.export.export(model, example_inputs)

# Partition: ops supported by QNN go to NPU; rest stay on CPU
edge_program = to_edge(exported)
edge_program = edge_program.to_backend(QnnBackend())

# Print delegation map:
for node in edge_program.exported_program().graph.nodes:
    if node.op == "call_function":
        backend = getattr(node.meta, "delegation_tag", "CPU")
        print(f"  {node.name}: {backend}")
```

---

## On-Device LLM: The New Frontier

Smartphone NPUs are now fast enough to run **7B-parameter LLMs** at usable speeds:

| Model | Device | Performance | Quantization | Source |
|:-----:|:---:|:-----------:|:------------:|:----:|
| Apple on-device foundation model (~3B) | iPhone 15 Pro (A17 Pro) | ~30 tok/s generation, ~0.6 ms per prompt token | ~3.7 bits/weight (mixed 2/4-bit) | Apple, 2024 |
| Llama-class 7–8B | Snapdragon 8 Gen 3 / 8 Elite | ~10–20 tok/s in vendor demos | INT4 | Qualcomm demos |
| Phi-3-mini (3.8B) | Lunar Lake laptop (NPU/GPU) | ~15–20 tok/s reported | INT4 | Intel/Microsoft demos |

Treat vendor demos as upper bounds: real apps share memory bandwidth with the display, camera and OS.

At 20–30 tokens/s (typical reading speed ~4–5 words/s ≈ 5–7 tokens/s), an on-device 3B model is perceptibly fast for chat. The enabling technology is **INT4 weight-only quantization**: storing all 3B weights at 4 bits = 3B × 0.5 B = **1.5 GB** — fitting in the 4–8 GB of LPDDR5 available on a phone.

The key bottleneck for LLM inference on edge NPUs is not compute — it is **weight bandwidth**. Autoregressive decoding processes one token at a time (batch size = 1), so arithmetic intensity is:

$$\text{AI} = \frac{2 \cdot d_{\text{model}}^2}{\text{(bytes per weight)} \times d_{\text{model}}^2} = \frac{2}{\text{bytes/weight}}$$

For INT4 (0.5 B/weight): AI = 2 / 0.5 = **4 FLOP/byte**: extremely memory-bound. The bound on decode speed is therefore memory bandwidth ÷ model bytes: a ~3B model at ~3.7 bits per weight is ~1.4 GB, so a phone's ~60 GB/s of LPDDR5X allows at most ~40 tokens/s, which is why Apple's measured ~30 tokens/s is close to the ceiling. The NPU's TOPS barely matter for decode; they matter for prefill (processing the prompt), which is compute-bound.

---

## Why This Matters for Model Optimization

**1. Quantize to INT8 (minimum) before targeting any edge NPU.** FP32 and FP16 models will either be refused by the NPU or automatically quantized at runtime with poor calibration. Use post-training quantization (PTQ) with a representative calibration dataset for best accuracy. For 7B LLMs, use INT4 group-wise quantization (AWQ or GPTQ).

**2. Verify operator coverage early.** Before choosing a target architecture, export your model to the vendor's format and check the delegation report. Every op that falls back to CPU is an order-of-magnitude slowdown. Prefer standard PyTorch ops (Conv2d, Linear, LayerNorm, GELU — but confirm GELU is supported as a fused op, not three separate ops).

**3. Use static shapes.** Dynamic sequence lengths, variable batch sizes, and conditional control flow all break NPU compilation. Pad inputs to fixed shapes. Use bucketing for sequence lengths (e.g., 128, 256, 512, 1024) and compile a separate model for each bucket.

**4. Match model size to NPU memory.** An 8 MB on-chip SRAM (e.g., Google Edge TPU) limits you to ~8 M INT8 parameters if you want zero DRAM access. A Snapdragon with 1 MB NPU SRAM for weights requires LPDDR streaming, making it bandwidth-bound. Understand the on-chip vs off-chip boundary for your target.

**5. TOPS/W, not TOPS, drives the design tradeoff.** If your edge device has a 500 mW total SoC budget, a 10 TOPS @ 10 TOPS/W NPU and a 20 TOPS @ 5 TOPS/W NPU are not equivalent — the latter drains the battery faster for the same throughput. For always-on workloads (keyword spotting, wakeword detection), even 0.1 TOPS @ 500 TOPS/W (MCU territory) beats everything else.

---

## Key Takeaways

- An **NPU** is an INT8/INT4 MAC array integrated into an SoC, engineered for quantized neural-network inference at milliwatt-to-watt power levels; the primary figure of merit is **TOPS/W**, not raw TOPS.
- Edge inference is constrained by power budget, thermal, LPDDR bandwidth, and real-time latency; these constraints force INT8/INT4 quantization, large on-chip SRAM, static scheduling, and fixed operator sets.
- Real NPUs include Apple's ANE (35–38 TOPS) plus GPU neural accelerators from 2025, Qualcomm Hexagon (45–80 TOPS on PCs), Intel and AMD laptop NPUs (48–50 TOPS, set by Microsoft's 40 TOPS Copilot+ floor), Google Edge TPU (4 TOPS), Arm Ethos-U (≤ 4 TOPS) and Renesas DRP-AI (8 TOPS dense).
- **Operator coverage** — which ops the vendor compiler can map to NPU hardware — is the dominant practical constraint; unsupported ops fall back to CPU at 10–100× slowdown.
- The software stack (Core ML, LiteRT, ExecuTorch, OpenVINO, QNN, DRP-AI TVM) determines the model-to-silicon path; each framework has different supported op sets and quantization requirements, and Android's NNAPI is now deprecated.
- On-device LLMs (3–8B parameters, INT4) are now viable on 2024 flagship SoCs at 10–20 tokens/s; the bottleneck is LPDDR5 weight-streaming bandwidth, not NPU compute TOPS.
- **Model architecture decisions** — hidden size divisible by the NPU's SIMD width, standard activations, no dynamic shapes, small enough to fit key weights in on-chip SRAM — determine whether you get 1× or 20× speedup over CPU on the same device.

## Check Yourself

<details class="check">
<summary>A phone has 64 GB/s of memory bandwidth. What is the decode ceiling for a 3B model at 4 bits per weight (plus 10 % overhead), and what would doubling NPU TOPS change?</summary>
<p>3 × 10⁹ × 0.5 B × 1.1 ≈ 1.65 GB per token; 64 / 1.65 ≈ 39 tokens/s. Doubling TOPS changes almost nothing for decode (it is memory-bound) but speeds up prefill of long prompts.</p>
</details>

<details class="check">
<summary>Why do edge NPUs prefer output-stationary dataflows for convolutions?</summary>
<p>A convolution produces many output positions, each accumulating many products; keeping each partial sum in a local accumulator avoids writing and re-reading it from SRAM, the most frequent and expensive data movement in that layer.</p>
</details>

<details class="check">
<summary>The NPU profile shows 80 % of time on the CPU. What is the usual cause and the first fix?</summary>
<p>Unsupported operators falling back to the CPU, often with costly layout conversions at each boundary. Check the delegation report, then rewrite or decompose the offending ops (e.g. SiLU as sigmoid × x, fixed shapes, supported attention patterns) so the whole graph stays on the NPU.</p>
</details>

<details class="check">
<summary>Why are vendor TOPS numbers hard to compare?</summary>
<p>They mix precisions (INT4 vs INT8), sparse vs dense rates, and sometimes sum CPU, GPU and NPU together; power is usually unpublished. Compare at the same precision and density, and prefer measured throughput on your model.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Apple Intelligence Foundation Language Models | Apple, 2024 | [arxiv.org/abs/2407.21075](https://arxiv.org/abs/2407.21075) | On-device model size, quantization and ~30 tok/s on iPhone 15 Pro |
| Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for CNNs | Y.-H. Chen, J. Emer, V. Sze, ISCA 2016 | [doi.org/10.1109/ISCA.2016.40](https://doi.org/10.1109/ISCA.2016.40) | Weight-, output- and row-stationary dataflows |
| Coral Edge TPU performance benchmarks | Google Coral | [coral.ai/docs/edgetpu/benchmarks](https://coral.ai/docs/edgetpu/benchmarks/) | 4 TOPS at 2 W; INT8 TFLite-only model requirements |
| Arm Ethos-U85 NPU | Arm | [arm.com/products/silicon-ip-cpu/ethos/ethos-u85](https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u85) | Edge NPU IP range and transformer support |
| Neural Networks API deprecation (Android 15) | Android Developers | [developer.android.com/ndk/guides/neuralnetwor…](https://developer.android.com/ndk/guides/neuralnetworks/migration-guide) | NNAPI deprecated; migrate to LiteRT delegates |
| Renesas RZ/V2H with DRP-AI3 | Renesas | [renesas.com/en/products/rz-v2h](https://www.renesas.com/en/products/rz-v2h) | 8 TOPS dense / 80 TOPS sparse, ~10 TOPS/W |
| ExecuTorch documentation | PyTorch | [pytorch.org/executorch](https://pytorch.org/executorch/) | PyTorch's edge runtime and its Core ML / QNN back ends |

---

**Next:** [Chapter 18 — DSPs](./18_dsps.md) — We look at the digital signal processor — the predecessor to the NPU and the compute engine behind always-on audio, baseband, radar, and the sensor pipeline that feeds most edge AI systems.

[← Back to Table of Contents](./README.md)
