---
title: "Chapter 7 — Types of Chips: A Taxonomy"
---

[← Back to Table of Contents](./README.md)

# Chapter 7 — Types of Chips: A Taxonomy

You've read papers that benchmark on "A100s," blogs that mention "edge NPUs," datasheets for "Xilinx FPGAs," and job posts asking for experience with "custom ASICs." These are all chips, but they are strikingly different objects — different programming models, different performance profiles, different roles in an ML system. Before the deep-dive chapters in Part III, you need a **map**.

This chapter is that map. We survey the full spectrum from general-purpose CPU to fixed-function ASIC, explain the fundamental tradeoff (flexibility vs efficiency), define each chip type in one solid paragraph, and then collect everything into a master comparison table. By the end, you will know *why* you train on GPUs, serve inference on NPUs or ASICs, and occasionally prototype on FPGAs — not as received wisdom, but as a necessary consequence of physics and economics.

> **The one-sentence version:** Every chip type is a different answer to the question "how should we spend our transistor budget given the flexibility we need and the efficiency we demand?" — and ML is systematically pushing the answer toward specialization.

---

## The Fundamental Tradeoff: Flexibility vs Efficiency

Every design decision in chip architecture can be traced to one axis:

<div class="diagram">
<div class="diagram-title">The Flexibility–Efficiency Tradeoff</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Flexible (General-Purpose)</div>
    <ul>
      <li>Can run any algorithm — sort, render, train, simulate</li>
      <li>Programmable via software; logic can change at runtime</li>
      <li>Hardware must handle all cases → overhead for common cases</li>
      <li>Lower efficiency per operation (TOPS/W, TOPS/mm²)</li>
      <li>Supports all dtypes and operators</li>
      <li><strong>Examples: CPU, GPU</strong></li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Efficient (Specialized)</div>
    <ul>
      <li>Optimized for one class of workload — often one specific operator</li>
      <li>Fixed function or tightly constrained programmability</li>
      <li>Every transistor serves the target workload → very high utilization</li>
      <li>High efficiency (TOPS/W), low latency, low power</li>
      <li>Limited dtypes (often INT8 or FP8 only)</li>
      <li><strong>Examples: NPU, ASIC, ISP</strong></li>
    </ul>
  </div>
</div>
</div>

The spectrum is not binary — it is a continuum:

```text
More flexible ◄────────────────────────────────────────► More efficient

CPU   →   GPU   →   DSP/ISP   →   NPU/TPU   →   FPGA   →   ASIC
(any workload)                  (ML-focused)  (reconfigurable)  (one job)
```

> **Key insight for ML:** Research and training require flexibility (trying new architectures, new ops, debugging). Production inference on high volume, known workloads rewards specialization. This is exactly why you see **GPUs dominating training** and **NPUs/ASICs dominating inference** as ML matures.

---

## The Chip Types, Defined

### CPU — Central Processing Unit

The **CPU** is the general-purpose brain of a computer system. It is optimized for **low-latency, sequential, control-heavy workloads** — tasks where each step depends on the result of the previous one. Modern CPUs are multi-core out-of-order superscalar processors: each core can execute instructions out of order, reorder memory accesses, predict branches, and run multiple instructions simultaneously per cycle. A high-end server CPU (AMD EPYC Genoa, Intel Sapphire Rapids) might have 96–128 cores, each with AVX-512 and AMX extensions for SIMD and matrix computation.

CPUs are irreplaceable for the **host side** of any compute job: launching GPU kernels, managing data pipelines, running PyTorch's autograd graph, and handling Python interpreter overhead. They remain competitive for **small-model CPU-only inference** (especially with INT8 quantization on VNNI/AMX) and for **irregular graph neural networks** where data-dependent control flow defeats GPU parallelism.

Covered in depth: [Chapter 14 — CPUs](./14_cpus.md)

### GPU — Graphics Processing Unit

The **GPU** began as a rasterization accelerator but is now the **primary training accelerator for deep learning**. Its architecture is built around **massive data parallelism**: thousands of small cores (CUDA cores, shader processors) organized into Streaming Multiprocessors (SMs), executing the same instruction on many data elements simultaneously (SIMT — Single Instruction, Multiple Threads).

Modern ML GPUs (NVIDIA H100, AMD MI300X) add **Tensor Cores** — dedicated 4×4 or 8×8 matrix-multiply hardware that executes FP16/BF16/FP8/INT8 GEMMs at dramatically higher throughput than CUDA cores alone. An H100 SXM5 delivers ~3.9 PFLOPS of BF16 dense throughput (tensor core), backed by 80 GB HBM3 at ~3.35 TB/s memory bandwidth. GPUs are programmable in CUDA, HIP, or Triton, and support all major dtypes from FP64 down to FP4.

Covered in depth: [Chapter 15 — GPUs](./15_gpus.md)

### TPU — Tensor Processing Unit (Google)

Google's **TPU** is a family of ASICs designed specifically for large-scale neural network training and inference. The architecture centers on a **systolic array** — a 2D grid of multiply-accumulate units where partial sums flow from cell to cell like a "wave," minimizing the need to load/store intermediate values. The TPU v1 (2016) was INT8-only inference; by TPU v4/v5 (Trillium), the chips support BF16 training and FP8, with large HBM capacity and high inter-chip bandwidth (ICI, their NVLink equivalent).

TPUs are tightly integrated with Google's software stack (JAX/XLA) and only available through Google Cloud. Their efficiency advantage over GPUs comes from the systolic array's near-100% utilization for matmul-dominated workloads and from co-design with XLA's compiler. LLMs like Gemini are primarily trained on TPU pods.

Covered in depth: [Chapter 16 — TPUs & Systolic Arrays](./16_tpus_and_systolic_arrays.md)

### NPU — Neural Processing Unit

An **NPU** (also called Neural Engine, AI Accelerator, or ML Accelerator) is a chip or chip block specifically designed for neural network **inference** at the edge or in mobile devices. The defining characteristics: small die area, low power (milliwatts to a few watts), fixed or semi-fixed dataflow (often a MAC array or systolic array), and support for quantized (INT8, INT4) operations.

Examples:
- **Apple Neural Engine (ANE)**: 16-core neural engine in M-series and A-series SoCs; ~38 TOPS on M4; optimized for CoreML FP16/INT8
- **Qualcomm Hexagon NPU**: in Snapdragon SoCs; used for on-device LLM inference (Phi-3, Llama)
- **Renesas DRP-AI**: dynamically reconfigurable processor with a tiny power envelope
- **Samsung Exynos NPU**: in Galaxy SoCs
- **Google Edge TPU**: a tiny 4 TOPS INT8 inference chip (Coral)

NPUs typically run INT8 or INT4 quantized models, support a fixed operator set (Conv2D, LSTM, Attention in modern versions), and are programmed through vendor SDKs (CoreML, SNPE, OpenVX) rather than CUDA/Triton.

Covered in depth: [Chapter 17 — NPUs & Edge AI](./17_npus_and_edge_ai.md)

### DSP — Digital Signal Processor

A **DSP** is a processor optimized for **real-time signal processing**: audio, radio, radar, sensor fusion. Architecturally, DSPs lean on **VLIW** (Very Long Instruction Word) execution (the compiler packs multiple operations per cycle), fixed-point arithmetic, dedicated **MAC** (multiply-accumulate) units, and memory architectures suited for streaming data.

DSPs appear in ML contexts as:
- **Pre-processing** for AI pipelines: decimation, filtering, FFTs before a neural network
- **Qualcomm Hexagon DSP**: also used for running ML workloads on Android (alongside the NPU)
- **TI C6000 family**: used in radar/lidar pre-processing for autonomous driving
- **Audio NPU/DSP hybrids**: Alexa, Siri wake-word detection on tiny power budgets

The boundary between a "DSP" and an "NPU" is blurring — modern mobile DSPs often include dedicated neural network MACs.

Covered in depth: [Chapter 18 — DSPs](./18_dsps.md)

### ISP — Image Signal Processor

An **ISP** is a fixed-function pipeline chip (or block within an SoC) that processes raw image sensor data into usable frames in real time: demosaicing (Bayer pattern → RGB), noise reduction, white balance, HDR tone mapping, lens distortion correction, compression (JPEG/HEIC). Every phone, webcam, and machine-vision camera has one.

ISPs are increasingly relevant to ML because:
1. They sit **upstream** of any neural network in a vision pipeline — ISP output quality directly affects model accuracy
2. Modern ISPs are adding neural-network-based processing (learned noise reduction, semantic segmentation for bokeh)
3. Some ISP vendors run lightweight inference models on the ISP DSP core itself

The ISP's workload is extremely regular (fixed image dimensions, fixed pipeline order), so it is highly amenable to fixed-function design with minimal programmability.

Covered in depth: [Chapter 19 — ISPs](./19_isps.md)

### FPGA — Field-Programmable Gate Array

An **FPGA** is a chip whose logic is **reconfigurable after manufacturing** — an array of programmable logic blocks (LUTs, flip-flops), configurable routing, and hard blocks (DSP slices, BRAMs, PCIe, high-speed serial I/O). You "program" an FPGA by loading a **bitstream** that configures what each logic block does and how they connect.

FPGAs occupy a special position: they offer **near-ASIC efficiency** (no general-purpose overhead) while retaining **post-silicon flexibility** (redesign = reload bitstream, no new fab run). The cost: very high NRE in design time and tools, and lower clock frequencies (~250–700 MHz) vs an equivalent ASIC (1–3+ GHz), with higher power than a custom ASIC at the same logic density.

FPGAs in ML:
- **Prototyping** new neural chips before tape-out (an expensive and irreversible step)
- **Low-latency inference** for HFT (microsecond, not millisecond)
- **Custom hardware for new sparsity patterns** (academic research on structured sparsity)
- **Microsoft Project Brainwave / Azure FPGA**: served Bing search ranking, ResNet inference on FPGA clusters
- Major vendors: Xilinx (now AMD), Intel (Altera division), Microchip/Lattice for edge

Covered in depth: [Chapter 20 — FPGAs](./20_fpgas.md)

### ASIC — Application-Specific Integrated Circuit

An **ASIC** is a chip designed from scratch for one specific application. Once fabricated, its logic is permanent. Every transistor is placed to maximize efficiency for that one workload. The result: **10×–100× better energy efficiency and performance per dollar vs an FPGA**, and often **2×–5× vs a GPU** for the same workload.

The catch: **NRE (Non-Recurring Engineering) costs** for an ASIC are enormous — $10M–$100M+ for design, verification, and a single tape-out at a leading-edge node (3nm TSMC). You need extremely high volume or extreme performance requirements to justify it.

AI ASICs in production:
- **Google TPU v1–v5** (inference → training, internal Google)
- **AWS Trainium / Inferentia** (AWS-custom for training and inference)
- **Cerebras CS-3** (single giant wafer-scale chip, 4T transistors)
- **Groq LPU** (deterministic low-latency inference ASIC)
- **Tesla Dojo** (distributed training ASIC for video)
- **Graphcore IPU** (intelligence processing unit, graph-centric architecture)

Covered in depth: [Chapter 21 — ASICs](./21_asics.md)

### SoC — System-on-Chip

An **SoC** integrates multiple chip types onto **one die** (or a small multi-die package): CPU cores, GPU, NPU, ISP, DSP, memory controller, I/O (PCIe, USB, display), codec hardware, and often SRAM. The integration eliminates the power and latency penalties of chip-to-chip communication — the CPU and NPU share the same memory fabric and can pass tensor data without going off-chip.

SoCs dominate mobile and embedded compute:
- **Apple M4**: ARM CPU + GPU + Neural Engine + media engine + memory controller, unified 120 GB/s memory bandwidth across all engines
- **Qualcomm Snapdragon X Elite**: ARM CPU + Adreno GPU + Hexagon NPU, up to 45 TOPS
- **NVIDIA Orin**: 12-core ARM Cortex-A78 + Ampere GPU + NVDLA (deep learning accelerator) + PVA (signal processing) — up to 275 TOPS

The SoC model is increasingly adopted in AI servers too: **HBM + compute die** as one package (NVIDIA GB200, AMD MI300X), or **CPU + GPU + NPU** in one tile (Intel Meteor Lake).

Covered in depth: [Chapter 22 — SoCs](./22_socs.md)

### MCU — Microcontroller Unit

An **MCU** is the smallest member of the compute family: a complete computer (CPU core + SRAM + flash + I/O peripherals) on one chip, designed for **embedded control** at milliwatts or microwatts. MCUs run bare-metal or RTOS firmware, typically with 4–256 KB of SRAM, no DRAM, and clock frequencies of 10–500 MHz.

MCUs are becoming ML-relevant as **TinyML**: running tiny CNNs or keyword-spotting models on microcontrollers for always-on inference (wake-word detection, anomaly detection on sensors). The 2022 Cortex-M55 added Arm Helium (MVE) — an 8-wide INT8 SIMD engine specifically for neural network MAC operations. MCU ML frameworks: TensorFlow Lite Micro, Edge Impulse, TVM Micro.

### APU — Accelerated Processing Unit (mention)

**APU** usually refers to AMD's integrated CPU+GPU die (Ryzen "APU" parts: unified LPDDR5 memory shared between CPU and iGPU, 4–12 RDNA GPU compute units). Also sometimes used loosely for any AI-accelerated processor. Apple's M-series chips are sometimes called APUs in informal usage. Not a separate category — an SoC where CPU and GPU are on the same die with shared memory.

---

## The Master Comparison Table

| Chip Type | Primary Purpose | Parallelism Style | Flexibility | Typical Dtypes | Perf/Watt | Programmability | Where Used | Example Parts |
|-----------|----------------|-------------------|-------------|---------------|-----------|-----------------|-----------|---------------|
| **CPU** | General-purpose, control | SIMD, out-of-order ILP, multicore (~128 cores) | Very high | FP64/32/16/BF16, INT32/8 | Low–Medium | C/C++/Python, any language | Host control, CPU inference, preprocessing | AMD EPYC 9754, Intel Xeon SPR, Apple M4 Pro |
| **GPU** | Throughput: GEMM, rendering, simulation | SIMT: thousands of threads, tensor cores | High | FP32/16/BF16/FP8/INT8/INT4 | Medium–High | CUDA, HIP, Triton, OpenCL | DL training, inference, HPC | NVIDIA H100, AMD MI300X, NVIDIA RTX 4090 |
| **TPU** | DL training & inference (matmul) | Systolic array, tensor-level | Low–Medium | BF16, FP8, INT8 | Very High | JAX/XLA (only) | Google Cloud, Google internal | Google TPU v4, TPU v5e (Trillium) |
| **NPU** | Neural network inference at edge | Fixed MAC/systolic array | Low | INT8, INT4, FP16 | Extremely High | Vendor SDK (CoreML, SNPE) | Mobile, edge devices, laptops | Apple ANE, Qualcomm Hexagon NPU, Google Edge TPU |
| **DSP** | Real-time signal processing | VLIW, vector MAC | Medium | Fixed-point, INT16/8 | High | C + vendor intrinsics | Audio, radar, wireless, preprocessing | TI C6000, Qualcomm Hexagon DSP, Tensilica HiFi |
| **ISP** | Camera/image pipeline | Fixed pipeline, streaming | Very Low | RAW (10–14 bit), YUV | Extremely High | Limited (vendor API) | Phones, cameras, automotive | Apple ISP (A/M series), Sony ISP, OmniVision |
| **FPGA** | Reconfigurable logic, prototyping | LUTs, DSP slices, custom | Very High (reconfigurable) | Anything you implement | Medium | Verilog/VHDL/HLS | Prototyping, low-latency inference, networks | Xilinx/AMD Alveo, Intel Stratix 10, Lattice ECP5 |
| **ASIC** | One specific workload | Custom (systolic, dataflow, etc.) | None (fixed) | Custom (INT8, FP8, INT4) | Highest | Fixed; often SDK | High-volume inference, Google/AWS/Meta AI | Google TPU, AWS Trainium, Groq LPU |
| **SoC** | Integrated heterogeneous system | CPU + GPU + NPU + DSP on one die | High (via sub-engines) | All (per sub-engine) | Very High | Per-engine APIs | Mobile, embedded, laptops, edge servers | Apple M4, Qualcomm Snapdragon X, NVIDIA Orin |
| **MCU** | Embedded control, TinyML | SIMD (Helium/MVE), scalar | Medium | INT8, INT16, FP32 (tiny) | Ultra High | C, bare-metal, RTOS | IoT, sensors, wake-word, anomaly detection | STM32H7, Nordic nRF9160, ARM Cortex-M85 |

---

## Key Axes Explained

### Control-Dominated vs Data-Parallel

A workload is **control-dominated** if execution depends heavily on data values — branches, irregular memory access, dynamic dispatch. A compiler is control-dominated. **Data-parallel** workloads apply the same operation to independent data elements — matrix multiply, convolution, rendering. CPUs excel at control; GPUs/TPUs excel at data-parallel.

$$\text{GPU speedup} \approx \frac{T_{\text{CPU}}}{T_{\text{GPU}}} \propto \frac{\text{parallelism available}}{\text{synchronization overhead} + \text{launch overhead}}$$

For a 4096×4096 matmul, parallelism >> overhead → GPU wins by 100×. For a lookup table with data-dependent branching → CPU wins.

### Latency-Oriented vs Throughput-Oriented

<div class="diagram">
<div class="diagram-title">Latency vs Throughput Orientation</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Latency-Oriented (CPU)</div>
    <ul>
      <li>Complete <em>one task</em> as fast as possible</li>
      <li>Large caches, out-of-order execution, branch prediction</li>
      <li>Single-thread performance: 3–5 GHz, ~5 ns/instruction</li>
      <li>Ideal for: interactive requests, control logic, small batches</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Throughput-Oriented (GPU/TPU)</div>
    <ul>
      <li>Complete as many tasks per second as possible</li>
      <li>Many simple cores; hide latency with many in-flight operations</li>
      <li>Single-thread: ~1.5–2 GHz, ~10–20 ns/instruction</li>
      <li>Ideal for: batched inference, training, large GEMM</li>
    </ul>
  </div>
</div>
</div>

> **For ML serving:** a single-token autoregressive decode step is latency-sensitive (user waiting). A large batch of independent requests is throughput-oriented. This tension drives KV-cache sizing, batching strategies, and the choice between CPU+NPU vs GPU deployment (see [Chapter 29](./29_memory_hierarchy_in_action.md)).

### Fixed-Function vs Programmable

```text
Fixed-function  ←────────────────────────────────────────→  Fully programmable
  ISP             ASIC         NPU          DSP          FPGA   GPU    CPU
  (one pipeline)  (one algo)   (NN ops)     (C + VLIW)   (HDL)  (CUDA) (anything)
```

"Programmable" has multiple levels:
- **Algorithm-programmable** (GPU, CPU): you can run any algorithm
- **Kernel-programmable** (NPU via SDK): you load pre-compiled operator kernels, no new operators
- **Netlist-programmable** (FPGA): you configure the gate-level logic itself
- **Fixed** (ASIC, ISP): logic is permanent; parameters may be adjustable

### The Role of SoC Integration

Modern ML systems increasingly work as **heterogeneous SoCs** where the key advantage is **zero-copy memory sharing** between engines. On Apple M4:

```python
# On Apple Silicon: CPU and Neural Engine share the same LPDDR5X
# tensor stays in place; no copy between "CPU RAM" and "GPU VRAM"
import torch
import coremltools as ct

# Model runs partly on CPU (Python), partly on ANE (CoreML),
# partly on GPU (Metal) — all sharing 120 GB/s unified memory
model = ct.models.MLModel("model.mlpackage")
result = model.predict({"input": data})  # ANE executes, no PCIe copy
```

Contrast with a discrete GPU setup: every tensor transfer between CPU RAM and GPU VRAM costs 16–128 GB/s PCIe bandwidth (vs on-chip memory fabric at TB/s speeds). SoC integration eliminates this bottleneck for edge/laptop deployments.

---

## A Visual Taxonomy

<div class="diagram">
<div class="diagram-title">Chip Taxonomy: Flexibility vs Efficiency</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🖥️</div>
    <div class="card-title">CPU</div>
    <div class="card-desc">Max flexibility. Low perf/watt for ML. Essential for host control. VNNI/AMX for INT8 inference.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🎮</div>
    <div class="card-title">GPU</div>
    <div class="card-desc">ML training workhorse. Thousands of SIMT threads + Tensor Cores. CUDA ecosystem. Programmable.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🧮</div>
    <div class="card-title">TPU / ASIC</div>
    <div class="card-desc">Highest efficiency for matmul-dominated inference/training. Systolic arrays. Limited flexibility.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">📱</div>
    <div class="card-title">NPU / DSP</div>
    <div class="card-desc">Edge inference: milliwatts, INT8/INT4. Vendor SDK. Apple ANE, Hexagon, Renesas DRP-AI.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">🔌</div>
    <div class="card-title">FPGA</div>
    <div class="card-desc">Post-silicon reconfigurable. Prototyping + low-latency inference. HDL / HLS programming.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-icon">📷</div>
    <div class="card-title">SoC / ISP</div>
    <div class="card-desc">Integration: CPU + GPU + NPU + ISP on one die. Zero-copy memory sharing. Apple M, Snapdragon.</div>
  </div>
</div>
</div>

---

## Numerical Perspective: Efficiency Numbers

To make the flexibility/efficiency tradeoff concrete, here are representative numbers (order of magnitude; exact figures depend on workload, precision, utilization, and generation):

| Chip | INT8 TOPS | TDP (W) | INT8 TOPS/W | INT8 TOPS/mm² die |
|------|:---:|:---:|:---:|:---:|
| Intel Xeon SPR (cpu + AMX) | ~80 | 350 | ~0.23 | low |
| NVIDIA H100 SXM5 (GPU) | ~3,958 | 700 | ~5.7 | medium |
| Google TPU v5p (ASIC) | ~459 (BF16) | 197 | ~2.3 | high |
| Qualcomm Snapdragon X Elite NPU | ~45 | ~5 (NPU only) | ~9 | very high |
| Apple M4 Neural Engine | ~38 | ~8 (total SoC) | ~4.75 | very high |
| AWS Inferentia2 (ASIC) | ~380 | 110 | ~3.5 | high |
| Xilinx/AMD Alveo U280 (FPGA) | ~170* | 225 | ~0.75 | low |

*FPGA number is for a dense INT8 inference implementation; reconfigurability overhead visible.

```python
# Quick efficiency comparison in Python
chips = {
    "Xeon SPR (CPU+AMX)": {"tops_int8": 80,    "tdp_w": 350},
    "H100 SXM5 (GPU)":    {"tops_int8": 3958,   "tdp_w": 700},
    "Snapdragon X NPU":   {"tops_int8": 45,     "tdp_w": 5},
    "Apple M4 ANE":       {"tops_int8": 38,     "tdp_w": 8},
    "AWS Inferentia2":    {"tops_int8": 380,    "tdp_w": 110},
}

print(f"{'Chip':<25} {'INT8 TOPS':>10} {'TDP (W)':>8} {'TOPS/W':>8}")
print("-" * 55)
for name, v in sorted(chips.items(), key=lambda x: -x[1]["tops_int8"] / x[1]["tdp_w"]):
    tpw = v["tops_int8"] / v["tdp_w"]
    print(f"{name:<25} {v['tops_int8']:>10} {v['tdp_w']:>8} {tpw:>8.2f}")
```

The trend is clear: **specialized chips (NPU, ASIC) achieve 3–10× better TOPS/W** than general-purpose GPUs, and the gap widens at lower precision (INT4 vs INT8).

---

## Why This Matters for Model Optimization

The chip taxonomy maps directly to where your model lives in its lifecycle:

<div class="diagram">
<div class="diagram-title">ML Lifecycle → Chip Type</div>
<div class="flow">
  <div class="flow-node accent wide">Research / Experimentation: GPU (flexibility — new architectures, new ops, fast iteration)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Large-Scale Training: GPU cluster / TPU pod (throughput — matmul-dominated)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">Cloud Inference (high volume): GPU or ASIC (Inferentia, Trainium, Groq) — efficiency matters at scale</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Edge / On-Device Inference: NPU / DSP in SoC (power budget: 1–10W total)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Ultra-Low-Power IoT: MCU with TinyML (microwatts, INT8 only)</div>
</div>
</div>

**Why inference moves toward specialized silicon:**
- Once a model is deployed, its architecture is fixed. The flexibility of a GPU is *wasted* on a fixed compute graph.
- Efficiency directly translates to cost-per-query (inference at scale) and battery life (edge).
- INT8 quantization (which is lossless or near-lossless for most inference tasks) maps perfectly onto NPU/ASIC MAC arrays.
- Custom silicon can eliminate memory overhead that GPUs carry for programmability (e.g., the CUDA warp scheduler, register file, L1/L2 caches for arbitrary access patterns).

**Why research stays on GPUs:**
- New architectures (Mamba, Mixture-of-Experts, diffusion, flow matching) require new operators, new memory patterns, irregular compute graphs.
- GPU + CUDA is the universal research platform — every library, every optimization, every paper benchmark uses it.
- The time from "idea" to "running on GPU" is hours; from "idea" to "running on custom ASIC" is years and tens of millions of dollars.

This is the **hardware-software co-design loop** at the industry scale: researchers discover what works on GPUs, and the insights flow into the next generation of specialized silicon.

---

## Key Takeaways

- Every chip type is a point on the flexibility/efficiency spectrum: **CPU → GPU → DSP/NPU → FPGA → ASIC**, with efficiency rising (TOPS/W) and flexibility falling.
- **CPUs** handle control, host code, and irregular workloads; **GPUs** dominate training and flexible inference; **TPUs/ASICs** provide maximum efficiency for fixed, matmul-heavy workloads; **NPUs** bring inference to mobile/edge at milliwatt budgets.
- **FPGAs** are reconfigurable post-silicon — near-ASIC efficiency with the flexibility to reload logic, making them ideal for prototyping and latency-sensitive fixed workloads.
- **SoCs** integrate multiple chip types on one die, enabling zero-copy memory sharing between CPU, GPU, NPU, and ISP — the dominant architecture for mobile and edge AI.
- The key axes are: **control-dominated vs data-parallel**, **latency-oriented vs throughput-oriented**, and **fixed-function vs programmable**.
- The ML lifecycle maps naturally onto the spectrum: research → GPU (flexible); production inference → NPU/ASIC (efficient); edge → SoC NPU; IoT → MCU.
- Specialized silicon (NPU, ASIC) achieves 3–10× better TOPS/W than GPU for the same INT8 inference workload — explaining the industry's rapid investment in custom AI silicon.

---

*Next: [Chapter 8 — Parallel Processing](./08_parallel_processing.md), where we go deep on ILP, superscalar execution, SIMD, SIMT, multicore, Flynn's taxonomy, and Amdahl's Law — the concepts that explain why all these chip types are built around parallelism.*

[← Back to Table of Contents](./README.md)
