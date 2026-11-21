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

CPU   →   GPU   →   FPGA   →   DSP   →   NPU/TPU   →   ASIC   →   ISP
(any workload)    (reconfigurable)       (ML-focused)      (one job)  (one pipeline)
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

Modern ML GPUs (NVIDIA H100, AMD MI300X) add **Tensor Cores** — dedicated small-matrix multiply engines that executes FP16/BF16/FP8/INT8 GEMMs at dramatically higher throughput than CUDA cores alone. An H100 SXM5 delivers ~0.99 PFLOPS of dense BF16 tensor throughput (1.98 PFLOPS in FP8), backed by 80 GB HBM3 at ~3.35 TB/s memory bandwidth. GPUs are programmable in CUDA, HIP, or Triton, and support all major dtypes from FP64 down to FP4.

Covered in depth: [Chapter 15 — GPUs](./15_gpus.md)

### TPU — Tensor Processing Unit (Google)

Google's **TPU** is a family of ASICs designed specifically for large-scale neural network training and inference. The architecture centers on a **systolic array** — a 2D grid of multiply-accumulate units where partial sums flow from cell to cell like a "wave," minimizing the need to load/store intermediate values. The TPU v1 (2016) was INT8-only inference; by TPU v6e (Trillium) and v7 (Ironwood), the chips support BF16 training and FP8, with large HBM capacity and high inter-chip bandwidth (ICI, their NVLink equivalent).

TPUs are tightly integrated with Google's software stack (JAX/XLA) and are rented mainly through Google Cloud, though large customers now sign dedicated capacity deals (Anthropic's covers gigawatts of TPUs from 2026–27). Their efficiency advantage over GPUs comes from the systolic array's near-100% utilization for matmul-dominated workloads and from co-design with XLA's compiler. LLMs like Gemini are primarily trained on TPU pods.

Covered in depth: [Chapter 16 — TPUs & Systolic Arrays](./16_tpus_and_systolic_arrays.md)

### NPU — Neural Processing Unit

An **NPU** (also called Neural Engine, AI Accelerator, or ML Accelerator) is a chip or chip block specifically designed for neural network **inference** at the edge or in mobile devices. The defining characteristics: small die area, low power (milliwatts to a few watts), fixed or semi-fixed dataflow (often a MAC array or systolic array), and support for quantized (INT8, INT4) operations.

Examples:
- **Apple Neural Engine (ANE)**: 16-core neural engine in M-series and A-series SoCs; ~38 TOPS on M4; optimized for CoreML FP16/INT8
- **Qualcomm Hexagon NPU**: in Snapdragon SoCs; used for on-device LLM inference (Phi-3, Llama)
- **Renesas DRP-AI**: dynamically reconfigurable processor with a tiny power envelope
- **Samsung Exynos NPU**: in Galaxy SoCs
- **Google Edge TPU**: a tiny 4 TOPS INT8 inference chip (Coral)

NPUs typically run INT8 or INT4 quantized models, support a fixed operator set (Conv2D, LSTM, Attention in modern versions), and are programmed through vendor toolchains (Core ML, Qualcomm AI Runtime/QNN, LiteRT, ONNX Runtime execution providers) rather than CUDA/Triton.

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

FPGAs occupy a special position: they are far more efficient than a CPU for a fixed pipeline (though typically ~10× less than an equivalent ASIC) while retaining **post-silicon flexibility** (redesign = reload bitstream, no new fab run). The cost: very high NRE in design time and tools, and lower clock frequencies (~250–700 MHz) vs an equivalent ASIC (1–3+ GHz), with higher power than a custom ASIC at the same logic density.

FPGAs in ML:
- **Prototyping** new neural chips before tape-out (an expensive and irreversible step)
- **Low-latency inference** for HFT (microsecond, not millisecond)
- **Custom hardware for new sparsity patterns** (academic research on structured sparsity)
- **Microsoft Project Brainwave / Azure FPGA**: served Bing search ranking, ResNet inference on FPGA clusters
- Major vendors: AMD (Xilinx), Altera (spun out of Intel in 2025, now majority-owned by Silver Lake), Lattice and Microchip for edge

Covered in depth: [Chapter 20 — FPGAs](./20_fpgas.md)

### ASIC — Application-Specific Integrated Circuit

An **ASIC** is a chip designed from scratch for one specific application. Once fabricated, its logic is permanent. Every transistor is placed to maximize efficiency for that one workload. The result: **10×–100× better energy efficiency and performance per dollar vs an FPGA**, and often **2×–5× vs a GPU** for the same workload.

The catch: **NRE (Non-Recurring Engineering) costs** for an ASIC are enormous — a leading-edge (3 nm-class) chip project commonly costs hundreds of millions of dollars in design, verification, IP and software, and the mask set alone runs to tens of millions. You need extremely high volume or extreme performance requirements to justify it.

AI ASICs in production:
- **Google TPU v1–v7 and TPU 8t/8i** (inference → training → split training/inference lines)
- **AWS Trainium / Inferentia** (AWS-custom for training and inference)
- **Cerebras CS-3** (single giant wafer-scale chip, 4T transistors)
- **Groq LPU** (deterministic low-latency inference ASIC)
- **Meta MTIA**, **Microsoft Maia**, **OpenAI's Broadcom-built accelerator** (hyperscaler in-house chips)
- **Tesla Dojo** (training ASIC for video; wound down in 2025) and **Graphcore IPU** (bought by SoftBank in 2024): reminders that most AI-chip efforts do not survive

Covered in depth: [Chapter 21 — ASICs](./21_asics.md)

### SoC — System-on-Chip

An **SoC** integrates multiple chip types onto **one die** (or a small multi-die package): CPU cores, GPU, NPU, ISP, DSP, memory controller, I/O (PCIe, USB, display), codec hardware, and often SRAM. The integration eliminates the power and latency penalties of chip-to-chip communication — the CPU and NPU share the same memory fabric and can pass tensor data without going off-chip.

SoCs dominate mobile and embedded compute:
- **Apple M4**: ARM CPU + GPU + Neural Engine + media engine + memory controller, unified 120 GB/s memory bandwidth across all engines
- **Qualcomm Snapdragon X Elite**: ARM CPU + Adreno GPU + Hexagon NPU, up to 45 TOPS
- **NVIDIA Orin**: 12-core ARM Cortex-A78 + Ampere GPU + NVDLA (deep learning accelerator) + PVA (signal processing) — up to 275 TOPS

The SoC model is increasingly adopted in AI servers too: **HBM + compute die** as one package (NVIDIA GB200, AMD MI300X), or **CPU + GPU** sharing one HBM pool (AMD MI300A), and CPU + GPU + NPU tiles in one client package (Intel Core Ultra).

Covered in depth: [Chapter 22 — SoCs](./22_socs.md)

### MCU — Microcontroller Unit

An **MCU** is the smallest member of the compute family: a complete computer (CPU core + SRAM + flash + I/O peripherals) on one chip, designed for **embedded control** at milliwatts or microwatts. MCUs run bare-metal or RTOS firmware, typically with 4–256 KB of SRAM, no DRAM, and clock frequencies of 10–500 MHz.

MCUs are becoming ML-relevant as **TinyML**: running tiny CNNs or keyword-spotting models on microcontrollers for always-on inference (wake-word detection, anomaly detection on sensors). Arm's Cortex-M55 (announced 2020) introduced Helium (MVE), a 128-bit vector extension that executes 8 INT8 MACs per cycle on that core, and Arm's Ethos-U microNPUs sit beside such cores for heavier models. MCU ML frameworks: TensorFlow Lite Micro, Edge Impulse, TVM Micro.

### APU — Accelerated Processing Unit (mention)

**APU** usually refers to AMD's integrated CPU+GPU die (Ryzen "APU" parts: unified LPDDR5 memory shared between CPU and iGPU, 4–12 RDNA GPU compute units). Also sometimes used loosely for any AI-accelerated processor. Apple's M-series chips are sometimes called APUs in informal usage. Not a separate category — an SoC where CPU and GPU are on the same die with shared memory.

---

## The Master Comparison Table

| Chip Type | Primary Purpose | Parallelism Style | Flexibility | Typical Dtypes | Perf/Watt | Programmability | Where Used | Example Parts |
|-----------|----------------|-------------------|-------------|---------------|-----------|-----------------|-----------|---------------|
| **CPU** | General-purpose, control | SIMD, out-of-order ILP, multicore (up to 192–288 cores per socket) | Very high | FP64/32/16/BF16, INT32/8 | Low–Medium | C/C++/Python, any language | Host control, CPU inference, preprocessing | AMD EPYC 9754, Intel Xeon 6, Apple M4 Pro |
| **GPU** | Throughput: GEMM, rendering, simulation | SIMT: thousands of threads, tensor cores | High | FP32/16/BF16/FP8/INT8/INT4 | Medium–High | CUDA, HIP, Triton, OpenCL | DL training, inference, HPC | NVIDIA H100, AMD MI300X, NVIDIA RTX 4090 |
| **TPU** | DL training & inference (matmul) | Systolic array, tensor-level | Low–Medium | BF16, FP8, INT8 | Very High | JAX/XLA, PyTorch/XLA | Google Cloud, Google internal | Google TPU v5p, v6e (Trillium), v7 (Ironwood) |
| **NPU** | Neural network inference at edge | Fixed MAC/systolic array | Low | INT8, INT4, FP16 | Extremely High | Vendor toolchain (Core ML, QNN, LiteRT) | Mobile, edge devices, laptops | Apple ANE, Qualcomm Hexagon NPU, Google Edge TPU |
| **DSP** | Real-time signal processing | VLIW, vector MAC | Medium | Fixed-point, INT16/8 | High | C + vendor intrinsics | Audio, radar, wireless, preprocessing | TI C6000, Qualcomm Hexagon DSP, Tensilica HiFi |
| **ISP** | Camera/image pipeline | Fixed pipeline, streaming | Very Low | RAW (10–14 bit), YUV | Extremely High | Limited (vendor API) | Phones, cameras, automotive | Apple ISP (A/M series), Sony ISP, OmniVision |
| **FPGA** | Reconfigurable logic, prototyping | LUTs, DSP slices, custom | Very High (reconfigurable) | Anything you implement | Medium | Verilog/VHDL/HLS | Prototyping, low-latency inference, networks | Xilinx/AMD Alveo, Altera Agilex, Lattice ECP5 |
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
      <li>Single thread: 4–6 GHz and several instructions retired per cycle</li>
      <li>Ideal for: interactive requests, control logic, small batches</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Throughput-Oriented (GPU/TPU)</div>
    <ul>
      <li>Complete as many tasks per second as possible</li>
      <li>Many simple cores; hide latency with many in-flight operations</li>
      <li>Single thread: ~1.5–2 GHz and slow on its own; speed comes from thousands of threads</li>
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

## Choosing a Chip: A Decision Path

The table above is the map; the flowchart below is a route through it. It asks the questions an architect asks first, in the order that eliminates options fastest.

<!-- flowchart:chip_picker -->
<div class="flowchart" id="fc-chip_picker">
<div class="fc-title">Which kind of chip should run this workload?</div>
<svg viewBox="0 0 780 580" width="780" role="img" aria-label="Which kind of chip should run this workload?">
<path class="fc-edge" d="M230,42 L230,73"/><polygon class="fc-head" points="230,73 226.2,66 233.8,66"/>
<path class="fc-edge no" d="M230,127 L230,159"/><polygon class="fc-head no" points="230,159 226.2,152 233.8,152"/><text class="fc-elabel no" x="236" y="141" text-anchor="start">no</text>
<path class="fc-edge no" d="M230,213 L230,245"/><polygon class="fc-head no" points="230,245 226.2,238 233.8,238"/><text class="fc-elabel no" x="236" y="227" text-anchor="start">no</text>
<path class="fc-edge no" d="M230,299 L230,331"/><polygon class="fc-head no" points="230,331 226.2,324 233.8,324"/><text class="fc-elabel no" x="236" y="313" text-anchor="start">no</text>
<path class="fc-edge no" d="M230,385 L230,417"/><polygon class="fc-head no" points="230,417 226.2,410 233.8,410"/><text class="fc-elabel no" x="236" y="399" text-anchor="start">no</text>
<path class="fc-edge no" d="M230,471 L230,512"/><polygon class="fc-head no" points="230,512 226.2,505 233.8,505"/><text class="fc-elabel no" x="236" y="485" text-anchor="start">no</text>
<path class="fc-edge yes" d="M380,100 L485,100"/><polygon class="fc-head yes" points="485,100 478,103.8 478,96.2"/><text class="fc-elabel yes" x="390" y="94" text-anchor="start">yes</text>
<path class="fc-edge yes" d="M380,186 L485,186"/><polygon class="fc-head yes" points="485,186 478,189.8 478,182.2"/><text class="fc-elabel yes" x="390" y="180" text-anchor="start">yes</text>
<path class="fc-edge yes" d="M380,272 L485,272"/><polygon class="fc-head yes" points="485,272 478,275.9 478,268.1"/><text class="fc-elabel yes" x="390" y="266" text-anchor="start">yes</text>
<path class="fc-edge yes" d="M380,358 L485,358"/><polygon class="fc-head yes" points="485,358 478,361.9 478,354.1"/><text class="fc-elabel yes" x="390" y="352" text-anchor="start">yes</text>
<path class="fc-edge yes" d="M380,444 L485,444"/><polygon class="fc-head yes" points="485,444 478,447.9 478,440.1"/><text class="fc-elabel yes" x="390" y="438" text-anchor="start">yes</text>
<g class="fc-node start"><rect x="120" y="10" width="220" height="32" rx="16"/><text x="230" y="30" text-anchor="middle">A workload to run</text></g>
<g class="fc-node decision"><polygon points="80,100 94,73 366,73 380,100 366,127 94,127"/><text x="230" y="98" text-anchor="middle">Branchy, serial or OS-level code?</text><text class="sub" x="230" y="111" text-anchor="middle">control flow dominates, little data parallelism</text></g>
<g class="fc-node decision"><polygon points="80,186 94,159 366,159 380,186 366,213 94,213"/><text x="230" y="184" text-anchor="middle">A fixed signal or image pipeline?</text><text class="sub" x="230" y="197" text-anchor="middle">filters, FFTs, demosaic, codecs</text></g>
<g class="fc-node decision"><polygon points="80,272 94,245 366,245 380,272 366,299 94,299"/><text x="230" y="270" text-anchor="middle">Battery or sub-10 W power budget?</text><text class="sub" x="230" y="283" text-anchor="middle">phone, laptop, camera, wearable</text></g>
<g class="fc-node decision"><polygon points="80,358 94,331 366,331 380,358 366,385 94,385"/><text x="230" y="356" text-anchor="middle">Microsecond latency or custom I/O,</text><text class="sub" x="230" y="369" text-anchor="middle">at low volume?</text></g>
<g class="fc-node decision"><polygon points="80,444 94,417 366,417 380,444 366,471 94,471"/><text x="230" y="442" text-anchor="middle">Millions of units of a stable model</text><text class="sub" x="230" y="455" text-anchor="middle">family, and a silicon team?</text></g>
<a href="./14_cpus.html"><g class="fc-node leaf k-orange"><rect x="485" y="78" width="250" height="44" rx="6"/><text x="610" y="98" text-anchor="middle">CPU</text><text class="sub" x="610" y="111" text-anchor="middle">out-of-order cores, big caches</text></g></a>
<a href="./18_dsps.html"><g class="fc-node leaf k-teal"><rect x="485" y="164" width="250" height="44" rx="6"/><text x="610" y="184" text-anchor="middle">DSP / ISP</text><text class="sub" x="610" y="197" text-anchor="middle">fixed-point MACs, VLIW, hardwired stages</text></g></a>
<a href="./17_npus_and_edge_ai.html"><g class="fc-node leaf k-green"><rect x="485" y="250" width="250" height="44" rx="6"/><text x="610" y="270" text-anchor="middle">NPU inside an SoC</text><text class="sub" x="610" y="283" text-anchor="middle">INT8/INT4 MAC arrays, tiny power</text></g></a>
<a href="./20_fpgas.html"><g class="fc-node leaf k-yellow"><rect x="485" y="336" width="250" height="44" rx="6"/><text x="610" y="356" text-anchor="middle">FPGA</text><text class="sub" x="610" y="369" text-anchor="middle">reconfigurable fabric, deterministic</text></g></a>
<a href="./21_asics.html"><g class="fc-node leaf k-pink"><rect x="485" y="422" width="250" height="44" rx="6"/><text x="610" y="442" text-anchor="middle">ASIC / TPU-class accelerator</text><text class="sub" x="610" y="455" text-anchor="middle">best perf/W, huge NRE, slow to change</text></g></a>
<a href="./15_gpus.html"><g class="fc-node leaf k-blue"><rect x="80" y="512" width="300" height="44" rx="6"/><text x="230" y="532" text-anchor="middle">GPU</text><text class="sub" x="230" y="545" text-anchor="middle">the programmable default for dense math</text></g></a>
</svg>
<div class="fc-caption">A first-cut decision path. Real systems mix several answers: a phone SoC has a CPU, GPU, NPU, DSP and ISP on one die. Click a box to jump to its chapter.</div>
</div>
<!-- /flowchart:chip_picker -->

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>The path ends at "GPU" whenever none of the specialised answers applies. Which question would you have to answer "yes" to before a startup building a custom inference ASIC beats simply buying next year's GPU, and how long does that answer need to stay true?</p>
</div>

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

| Chip | Dense INT8 TOPS | Power (W) | INT8 TOPS/W | Source of power figure |
|------|:---:|:---:|:---:|---|
| Intel Xeon 6 / SPR, 56 cores with AMX | ~200 (peak) | 350 | ~0.6 | socket TDP |
| NVIDIA H100 SXM | 1,979 | 700 | ~2.8 | board TDP |
| AMD MI300X | 2,615 | 750 | ~3.5 | board TDP |
| NVIDIA B200 (HGX) | 4,500 | 1,000 | ~4.5 | board TDP |
| Qualcomm Snapdragon X Elite NPU | 45 | ~5 (est.) | ~9 | estimate; NPU power not published |
| Apple M4 Neural Engine | 38 | ~5 (est.) | ~8 | estimate; ANE power not published |

Peak numbers at the precision each chip is best at; real workloads reach a fraction of them. Google and AWS do not publish per-chip power for TPUs or Trainium/Inferentia, so they are left out rather than guessed. (The [Accelerator Atlas](./atlas.md) lists their compute and bandwidth.)

```python
# Quick efficiency comparison in Python (dense INT8 peak, published or estimated power)
chips = {
    "Xeon SPR (CPU+AMX)": {"tops_int8": 200,  "tdp_w": 350},
    "H100 SXM (GPU)":     {"tops_int8": 1979, "tdp_w": 700},
    "MI300X (GPU)":       {"tops_int8": 2615, "tdp_w": 750},
    "B200 HGX (GPU)":     {"tops_int8": 4500, "tdp_w": 1000},
    "Snapdragon X NPU":   {"tops_int8": 45,   "tdp_w": 5},    # estimate
    "Apple M4 ANE":       {"tops_int8": 38,   "tdp_w": 5},    # estimate
}

print(f"{'Chip':<22} {'INT8 TOPS':>10} {'W':>6} {'TOPS/W':>7}")
for name, v in sorted(chips.items(), key=lambda x: -x[1]["tops_int8"] / x[1]["tdp_w"]):
    print(f"{name:<22} {v['tops_int8']:>10} {v['tdp_w']:>6} {v['tops_int8'] / v['tdp_w']:>7.2f}")
```

The trend: **edge NPUs reach roughly 2–3× the TOPS/W of datacenter GPUs** at a thousandth of the absolute power, and each GPU generation narrows the gap (H100 → B200 is +60 % TOPS/W). In the datacenter, GPUs and ASICs built on the same process node land within about a factor of two of each other; the node and the memory system usually matter more than the chip's category.

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
- INT8 (and increasingly FP8/FP4) quantization, near-lossless for many inference tasks when calibrated, maps directly onto NPU/ASIC MAC arrays.
- Custom silicon can eliminate memory overhead that GPUs carry for programmability (e.g., the CUDA warp scheduler, register file, L1/L2 caches for arbitrary access patterns).

**Why research stays on GPUs:**
- New architectures (Mamba, Mixture-of-Experts, diffusion, flow matching) require new operators, new memory patterns, irregular compute graphs.
- GPU + CUDA is the universal research platform — every library, every optimization, every paper benchmark uses it.
- The time from "idea" to "running on GPU" is hours; from "idea" to "running on custom ASIC" is years and tens of millions of dollars.

This is the **hardware-software co-design loop** at the industry scale: researchers discover what works on GPUs, and the insights flow into the next generation of specialized silicon.

---

## Key Takeaways

- Every chip type is a point on the flexibility/efficiency spectrum: **CPU → GPU → FPGA → DSP/NPU → ASIC**, with efficiency (TOPS/W) rising and flexibility falling.
- **CPUs** handle control, host code, and irregular workloads; **GPUs** dominate training and flexible inference; **TPUs/ASICs** provide maximum efficiency for fixed, matmul-heavy workloads; **NPUs** bring inference to mobile/edge at milliwatt budgets.
- **FPGAs** are reconfigurable post-silicon: much more efficient than CPUs on a fixed pipeline, roughly an order of magnitude behind an ASIC, and ideal for prototyping and latency-critical fixed workloads.
- **SoCs** integrate multiple chip types on one die, enabling zero-copy memory sharing between CPU, GPU, NPU, and ISP — the dominant architecture for mobile and edge AI.
- The key axes are: **control-dominated vs data-parallel**, **latency-oriented vs throughput-oriented**, and **fixed-function vs programmable**.
- The ML lifecycle maps naturally onto the spectrum: research → GPU (flexible); production inference → NPU/ASIC (efficient); edge → SoC NPU; IoT → MCU.
- Edge NPUs reach roughly 2–3× the TOPS/W of datacenter GPUs; in the datacenter, process node and memory system often matter more than whether the chip is called a GPU or an ASIC.

## Check Yourself

<details class="check">
<summary>Why do research teams stay on GPUs even when an ASIC would be more efficient for today's model?</summary>
<p>Research changes the model weekly: new operators, new memory patterns, new datatypes. A GPU runs whatever you can write in PyTorch or Triton within hours; an ASIC freezes assumptions years in advance, so its efficiency only pays off for a stable, high-volume workload.</p>
</details>

<details class="check">
<summary>What does "programmable" mean for an FPGA versus a GPU?</summary>
<p>A GPU is algorithm-programmable: you change the instructions it runs. An FPGA is netlist-programmable: you change the circuit itself by loading a new bitstream, which allows custom datapaths (any bit width, any pipeline) at the cost of lower clocks and long compile times.</p>
</details>

<details class="check">
<summary>The efficiency table puts H100 at ~2.8 INT8 TOPS/W and B200 at ~4.5. What changed, given both are on 4 nm-class nodes?</summary>
<p>Mostly architecture and packaging: B200 joins two reticle-sized dies with a 10 TB/s die-to-die link, doubles the tensor-core rate per SM generation, and runs at a higher power envelope where the fixed overheads (I/O, HBM PHYs, control) are a smaller share of the total.</p>
</details>

<details class="check">
<summary>Why is zero-copy memory sharing an advantage for an SoC, and what is its cost?</summary>
<p>CPU, GPU and NPU read the same physical memory, so handing a tensor between engines needs no PCIe copy. The cost is that all engines share one memory bus (100–800 GB/s on laptop SoCs), far below the HBM bandwidth of a discrete accelerator.</p>
</details>

<details class="check">
<summary>Name two AI-chip efforts that were wound down or sold, and one lesson they share.</summary>
<p>Tesla's Dojo training program (wound down in 2025) and Graphcore (sold to SoftBank in 2024). Both built strong silicon but had to compete with a moving target, NVIDIA's yearly cadence and CUDA ecosystem, which is the hardest part of the custom-chip business case.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Efficient Processing of Deep Neural Networks: A Tutorial and Survey | V. Sze, Y.-H. Chen, T.-J. Yang, J. Emer, Proc. IEEE 2017 | [arxiv.org/abs/1703.09039](https://arxiv.org/abs/1703.09039) | Taxonomy of DNN accelerators and their dataflows |
| A Domain-Specific Architecture for Deep Neural Networks | N. Jouppi, C. Young, N. Patil, D. Patterson, CACM 2018 | [doi.org/10.1145/3154484](https://doi.org/10.1145/3154484) | Why specialised silicon beats general-purpose processors on DNNs |
| NVIDIA H100 Tensor Core GPU Architecture | NVIDIA, 2022 | [resources.nvidia.com/en-us-tensor-core/gtc22-…](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper) | H100 dense and sparse tensor throughput |
| AMD Instinct MI300X data sheet | AMD, 2023 | [amd.com/en/products/accelerators/instinct/mi3…](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) | MI300X throughput and board power |
| Intel Advanced Matrix Extensions overview | Intel | [intel.com/content/www/us/en/products/docs/acc…](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html) | AMX operations per cycle per core |
| A Configurable Cloud-Scale DNN Processor for Real-Time AI (Brainwave) | J. Fowers et al., ISCA 2018 | [doi.org/10.1109/ISCA.2018.00012](https://doi.org/10.1109/ISCA.2018.00012) | FPGAs for low-latency datacenter inference |

---

**Next:** [Chapter 8 — Parallel Processing](./08_parallel_processing.md) — We go deep on ILP, superscalar execution, SIMD, SIMT, multicore, Flynn's taxonomy, and Amdahl's Law — the concepts that explain why all these chip types are built around parallelism.

[← Back to Table of Contents](./README.md)
