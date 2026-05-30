---
title: "Chapter 22 — SoCs (Systems on Chip)"
---

[← Back to Table of Contents](./README.md)

# Chapter 22 — SoCs (Systems on Chip)

When you run a local LLM on an M3 MacBook, the CPU preprocesses the prompt, the GPU and Neural Engine split the matrix multiplications, the memory controller streams 70 billion quantized weights from a shared LPDDR5X pool, and the display controller renders the token stream — all without a single byte crossing a PCIe slot or a board-level bus. The CPU and the NPU share the same physical DRAM array, addressing it through a shared memory controller on the same die. There is no copy. This is the **System on Chip** (SoC): every functional block a system needs, integrated onto one die (or one package), communicating over a high-speed internal fabric that no external chip-to-chip link can match in latency, bandwidth, or power efficiency.

The SoC is both the enabling technology of mobile computing — there is no other way to pack a full computer into a 5 mm-thick phone at 5 W — and the direction that datacenter hardware is moving as the co-packaging revolution turns a GPU rack into a SoC-like integrated system. Understanding SoCs is understanding why Apple's M-series can run 70B quantized LLMs when a discrete GPU with the same FLOPS cannot, why a Snapdragon phone routes inference across three different processor types simultaneously, and why NVIDIA's Grace Hopper is not just "a GPU with a CPU bolted on" but a fundamentally different memory architecture.

> **The one-sentence version:** An SoC integrates CPU cores, GPU, NPU/DSP, ISP, memory controllers, I/O blocks, and the interconnect fabric that ties them all together onto one die or package, making it possible to build a full computing system with far lower latency, power, cost, and physical size than a discrete multi-chip system — and enabling **unified/shared memory** that is the key to running large ML models on constrained hardware.

---

## Why Integrate? The SoC Value Proposition

A traditional PC or server separates every function onto a different chip: CPU on one die, GPU on another, connected by PCIe across a motherboard. This separation has real costs:

<div class="diagram">
<div class="diagram-title">Discrete Multi-Chip vs SoC: What Integration Buys</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Discrete Multi-Chip (PC/Server)</div>
    <ul>
      <li>CPU ↔ GPU via PCIe Gen5: ~128 GB/s (bidirectional)</li>
      <li>Every CPU→GPU tensor copy: 5–20 μs DMA overhead</li>
      <li>CPU RAM separate from GPU VRAM — copies mandatory</li>
      <li>Each chip has its own voltage regulators, PCB traces, decaps</li>
      <li>Total system: 3–5 chips, 5–10 W idle</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">SoC (e.g. Apple M3, Snapdragon 8 Gen 3)</div>
    <ul>
      <li>CPU ↔ GPU via on-die fabric: 400–800 GB/s</li>
      <li>Zero-copy: CPU, GPU, NPU share the same physical DRAM</li>
      <li>One shared memory controller, one voltage domain</li>
      <li>On-chip caches (L2, L3, system-level cache) shared coherently</li>
      <li>Total system: 1 chip, 3–5 W idle</li>
    </ul>
  </div>
</div>
</div>

The four integration benefits are **latency** (on-chip fabric vs PCIe), **bandwidth** (wider internal buses vs limited pins), **power** (no repeaters, no long traces, shared voltage delivery), and **size** (one package vs multiple).

The mobile industry drove SoC integration hardest — a smartphone has a 4–6 W power budget for the whole computer, which is only achievable with tight integration. The first iPhone (2007) used Samsung's SoC (ARM + GPU on one die). By 2020, Apple's A14 integrated a 6-core CPU, 4-core GPU, 16-core Neural Engine, ISP, video encoder/decoder, and memory controller into 11.8B transistors on 5 nm — a level of integration unthinkable with discrete chips.

---

## Anatomy of a Modern SoC

A phone-class SoC typically contains these blocks, all interconnected:

<div class="diagram">
<div class="diagram-title">Phone SoC Block Diagram (Snapdragon / Apple A-series style)</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🧠</div>
    <div class="card-title">CPU Cluster</div>
    <div class="card-desc">Big cores (3–4 Cortex-X / A-series performance) + Little cores (4× Cortex-A55 / efficiency). Heterogeneous; DVFS per cluster.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🎮</div>
    <div class="card-title">GPU</div>
    <div class="card-desc">Mobile GPU (Adreno, Mali, Apple GPU). 6–12 shader cores. Accesses unified DRAM; no separate VRAM. Runs general + ML shaders.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">⚡</div>
    <div class="card-title">NPU / Neural Engine</div>
    <div class="card-desc">Apple ANE (16–38 TOPS), Qualcomm Hexagon NPU (87 TOPS on 8 Gen 3), MediaTek APU. Fixed-function ML ASIC. See Ch 17.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">📷</div>
    <div class="card-title">ISP</div>
    <div class="card-desc">Camera pipeline (Ch 19). Bayer→RGB, denoise, HDR. Feeds NPU/GPU with NV12 frames. Qualcomm Spectra, Apple ISP.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">🎵</div>
    <div class="card-title">DSP / Audio</div>
    <div class="card-desc">Hexagon DSP (Qualcomm), Sensory DSP. Low-power always-on processing for audio, sensor fusion, wake-word detection. See Ch 18.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-icon">🗃️</div>
    <div class="card-title">Memory Controller</div>
    <div class="card-desc">LPDDR5/5X controller. Unified — all on-chip masters (CPU, GPU, NPU, ISP) share one physical DRAM pool via IOMMU/fabric.</div>
  </div>
  <div class="diagram-card teal">
    <div class="card-icon">📡</div>
    <div class="card-title">Modem / RF</div>
    <div class="card-desc">5G NR modem (integrated on die or separate companion die). Qualcomm X70 / Apple C1. Largest non-compute block on phone SoC.</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-icon">🎞️</div>
    <div class="card-title">Video Encoder/Decoder</div>
    <div class="card-desc">H.264/H.265/AV1 HW codec. 8K60 encode/decode. Fixed-function; frees GPU from codec work.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🔌</div>
    <div class="card-title">I/O Controllers</div>
    <div class="card-desc">USB 3.2 / Thunderbolt, Wi-Fi 7, Bluetooth, MIPI CSI/DSI, PCIe, UFS storage controller. Dozens of I/O masters on the fabric.</div>
  </div>
</div>
</div>

| Block | Area share (approx.) | Power at full load |
|---|---|---|
| CPU cluster | 15–25% | 2–5 W |
| GPU | 20–35% | 3–8 W |
| NPU | 5–10% | 1–3 W |
| ISP | 5–8% | 0.5–1 W |
| Modem (if integrated) | 25–35% | 3–5 W |
| Memory controller + PHY | 5–8% | 0.5–1 W |
| I/O / other | 5–10% | 0.5–1 W |

---

## The Shared On-Chip Interconnect and Unified Memory

The most important architectural feature of an SoC for ML is **unified shared memory**. In a discrete GPU system:

```
CPU RAM (64 GB DDR5)  ←→  PCIe Gen5 (128 GB/s)  ←→  GPU VRAM (80 GB HBM3)
```

To run inference, you copy weights from CPU RAM to GPU VRAM. For a 70B BF16 model (140 GB), this takes over 1 second just for the copy — before the first FLOP runs. And if the model doesn't fit in 80 GB VRAM, it can't run at all on a single GPU.

In an SoC with unified memory:

```
CPU / GPU / NPU                              All access same physical DRAM pool
     ↓↑         ↖↗ on-chip fabric (400–800 GB/s)
Memory Controller ←→ LPDDR5X / DRAM (shared)
```

**There is no copy.** The CPU writes a tensor to address 0xBEEF0000 and the GPU reads from 0xBEEF0000 — same physical bits, no DMA transfer, no PCIe round-trip. The only cost is the bandwidth to DRAM itself.

This has a profound implication for LLMs: a quantized 70B INT4 model weighs ~35 GB. On a Mac with 64 GB unified memory, the entire model fits and inference starts immediately — no copy, no VRAM limit. The CPU pre-fills (prefill is matmul-heavy → GPU + ANE) and the GPU decodes. **The memory is the model's only home, and every processor can access it.**

### Memory Bandwidth: The New Bottleneck

With unified memory, the shared LPDDR bandwidth becomes the binding constraint. Transformer decode (one token at a time, batch=1) is **memory-bandwidth-bound**: you read all the weights once per token. At INT4, a 7B model has $7 \times 10^9 \times 0.5 = 3.5$ GB of weights. At 128 GB/s (Apple M3 LPDDR5X), reading all weights takes $3.5 / 128 = 27$ ms → **37 tokens/second max** (roofline ceiling, ignoring compute and KV-cache). See [Chapter 11](./11_memory_wall_and_bandwidth.md) for the roofline derivation.

---

## Apple M-Series: The Canonical Unified-Memory SoC for ML

Apple's M-series chips are the highest-profile example of unified-memory SoCs for ML inference. Each generation increases memory bandwidth faster than it increases compute TOPS, reflecting an understanding that bandwidth is the binding constraint for LLM decode.

| Chip | Year | CPU | GPU cores | ANE TOPS | Memory BW | Max RAM | LPDDR gen |
|---|---|---|---|---|---|---|---|
| M1 | 2020 | 8-core | 8 | 11 TOPS | 68 GB/s | 16 GB | LPDDR4X |
| M1 Max | 2021 | 10-core | 32 | 11 TOPS | 400 GB/s | 64 GB | LPDDR5 |
| M1 Ultra | 2022 | 20-core (2×) | 64 (2×) | 22 TOPS (2×) | 800 GB/s (2×) | 128 GB | LPDDR5 |
| M2 | 2022 | 8-core | 10 | 15.8 TOPS | 100 GB/s | 24 GB | LPDDR5 |
| M2 Max | 2023 | 12-core | 38 | 15.8 TOPS | 400 GB/s | 96 GB | LPDDR5 |
| M3 | 2023 | 8-core | 10 | 18 TOPS | 100 GB/s | 24 GB | LPDDR5X |
| M3 Max | 2023 | 16-core | 40 | 18 TOPS | 400 GB/s | 128 GB | LPDDR5X |
| M4 | 2024 | 10-core | 10 | 38 TOPS | 120 GB/s | 32 GB | LPDDR5X |
| M4 Max | 2024 | 16-core | 40 | 38 TOPS | 546 GB/s | 128 GB | LPDDR5X |

> **The Ultra trick:** Apple's M1 Ultra (and M2/M3/M4 Ultra variants) are **two identical Max dies** connected by Apple's **UltraFusion** die-to-die interconnect (2.5 TB/s aggregate inter-die bandwidth). The OS and applications see it as one chip with doubled everything. This is an early form of **chiplet packaging** (see below and [Appendix E](./appendix_e_packaging.md)) implemented by Apple before chiplets became industry-standard.

### Practical LLM Bandwidth-Roofline on Apple M-Series

```python
import math

def llm_decode_roofline(
    model_params: int,   # total parameters
    bits_per_param: int, # quantization (4 for INT4, 8 for INT8, 16 for BF16)
    bandwidth_GBs: float,# memory bandwidth in GB/s
    flops_per_token: int = None  # if None, estimated as 2*params per token
) -> dict:
    """
    Compute the roofline ceiling on tokens/second for autoregressive LLM decode.
    Decode is memory-bandwidth-bound at batch=1: must read all weights once per token.
    """
    bytes_per_param = bits_per_param / 8.0
    model_size_GB = model_params * bytes_per_param / 1e9
    
    # Time to read all weights = model_size / bandwidth
    seconds_per_token = model_size_GB / bandwidth_GBs
    tokens_per_second = 1.0 / seconds_per_token
    
    return {
        "model_size_GB": round(model_size_GB, 1),
        "seconds_per_token": round(seconds_per_token * 1000, 2),   # ms
        "roofline_tokens_per_sec": round(tokens_per_second, 1),
    }

configs = [
    ("Llama-3 8B INT4,  M3 (100 GB/s)",  8e9,   4, 100),
    ("Llama-3 8B BF16,  M3 (100 GB/s)",  8e9,  16, 100),
    ("Llama-3 70B INT4, M3 Max (400 GB/s)", 70e9, 4, 400),
    ("Llama-3 70B INT4, M4 Max (546 GB/s)", 70e9, 4, 546),
    ("Llama-3 70B BF16, M2 Ultra (800 GB/s)", 70e9, 16, 800),
]

for name, params, bits, bw in configs:
    r = llm_decode_roofline(int(params), bits, bw)
    print(f"{name}")
    print(f"  Model: {r['model_size_GB']:.0f} GB, {r['seconds_per_token']} ms/token, "
          f"ceiling: {r['roofline_tokens_per_sec']} tok/s\n")
```

```
Llama-3 8B INT4,  M3 (100 GB/s)
  Model: 4 GB, 0.04 ms/token, ceiling: 25000.0 tok/s   ← compute-bound at batch=1

Llama-3 8B BF16,  M3 (100 GB/s)
  Model: 16 GB, 0.16 ms/token, ceiling: 625.0 tok/s    ← ~50 tok/s in practice (KV cache overhead)

Llama-3 70B INT4, M3 Max (400 GB/s)
  Model: 35 GB, 0.09 ms/token, ceiling: 114.3 tok/s    ← achievable; llama.cpp ~40–60 tok/s

Llama-3 70B INT4, M4 Max (546 GB/s)
  Model: 35 GB, 0.06 ms/token, ceiling: 156.0 tok/s    ← fits in 128 GB RAM

Llama-3 70B BF16, M2 Ultra (800 GB/s)
  Model: 140 GB, 0.17 ms/token, ceiling: 5.7 tok/s     ← fits in 192 GB RAM; BW-limited
```

The roofline shows why INT4 quantization is so impactful on unified-memory hardware: it quadruples the effective tokens/second ceiling by quartering the bytes read per token.

---

## Mobile SoCs: Routing ML Layers Across CPU / GPU / NPU

A flagship phone SoC runs ML inference across all three compute blocks simultaneously, orchestrated by the driver stack:

<div class="diagram">
<div class="diagram-title">Mobile SoC ML Layer Routing (Snapdragon 8 Gen 3 / Apple A17)</div>
<div class="flow-h">
  <div class="flow-node accent">Input Frame<br/><small>(NV12 from ISP)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">ISP<br/><small>preprocess, resize</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">NPU<br/><small>conv layers, INT8</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">GPU<br/><small>normalization, activation</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange">CPU<br/><small>post-process, NMS</small></div>
</div>
</div>

The Qualcomm **AI Stack** (QNN SDK) and Apple **Core ML** both implement a **graph partitioner** that splits a neural network's op-graph across NPU, GPU, and CPU at compile time. Each op is assigned to the processor that handles it most efficiently:

| Operation type | Best processor | Reason |
|---|---|---|
| Conv2D (large, INT8) | NPU | Tensor core / MAC array; 87 TOPS peak |
| Attention (variable seq) | GPU | Flexible SIMT for irregular shapes |
| LayerNorm, Softmax | GPU or CPU | Element-wise; NPU may not support natively |
| Non-Maximum Suppression | CPU | Sequential, branch-heavy |
| Always-on keyword detect | DSP | Ultra-low power; runs when screen off |
| Camera preprocessing | ISP | Zero CPU overhead; zero-copy to NPU |

The partitioner minimizes **cross-processor data copies** (each copy costs bandwidth and latency) while keeping each processor's utilization high. The result is that a 50 ms model inference on a phone may spend 30 ms on the NPU, 10 ms on the GPU, 5 ms on the CPU, and 5 ms in transfers — all pipelined across multiple input frames.

---

## The Chiplet Revolution: When Monolithic Dies Hit Limits

A monolithic SoC packs everything on one die. As models grow larger and requirements diverge (a phone needs a modem; a server doesn't), the monolithic approach runs into two hard limits:

**Reticle limit:** A single photolithography exposure covers at most ~858 mm² (the **reticle limit** set by the stepper lens size). An H100 die is 814 mm² — nearly at the limit. You cannot make a larger monolithic die.

**Yield limit:** A 500 mm² die at TSMC 5 nm with a 0.1 defect/cm² defect density has:

$$\text{Yield} \approx e^{-D \cdot A} = e^{-0.1 \times 5} \approx 60\%$$

A 1000 mm² die would yield ~37%. You throw away more than half your wafer. The economics collapse.

**Chiplets** solve this by splitting the design into smaller dies (**chiplets**) that each have high yield, then packaging them together so software sees one system:

<div class="diagram">
<div class="diagram-title">Monolithic Die vs Chiplet Architecture</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Monolithic (traditional SoC)</div>
    <ul>
      <li>One die: CPU + GPU + IO on same silicon + process node</li>
      <li>All blocks must use the same manufacturing process</li>
      <li>Die size limited by reticle (~858 mm²) and yield</li>
      <li>One defect anywhere kills the whole chip</li>
      <li>Example: Apple M1 (119 mm², high yield at 5 nm)</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Chiplet (disaggregated)</div>
    <ul>
      <li>Compute die on latest node (5 nm / 3 nm)</li>
      <li>IO/analog die on mature node (28 nm — cheaper, better yield)</li>
      <li>HBM stacks on interposer for high bandwidth</li>
      <li>Each chiplet independently tested before assembly</li>
      <li>Examples: AMD EPYC Genoa, Intel Ponte Vecchio, Nvidia H100 (die + HBM)</li>
    </ul>
  </div>
</div>
</div>

### UCIe: The Chiplet Interconnect Standard

The key enabler for chiplets is the **die-to-die interface**. Historically each vendor used proprietary links (Apple UltraFusion, AMD Infinity Fabric, Intel EMIB). In 2022, the industry coalesced around **UCIe (Universal Chiplet Interconnect Express)** — an open standard for die-to-die connectivity:

- **Standard package (2D)**: up to 16 GB/s/mm bandwidth density
- **Advanced package (2.5D interposer)**: up to 1 TB/s/mm bandwidth density
- **Protocol layer**: runs PCIe or CXL over the physical link for software compatibility

UCIe means a CPU chiplet from one vendor can be paired with an NPU chiplet from another vendor on a shared interposer — a "Lego of silicon." For AI hardware, this enables mixing-and-matching: compute chiplets optimized for matmul, memory chiplets optimized for bandwidth, I/O chiplets optimized for connectivity, all on one package that the OS sees as a single SoC. See [Appendix E](./appendix_e_packaging.md) for packaging details.

---

## Datacenter SoCs and the Superchip Trend

The SoC idea — integrate everything, share memory, eliminate inter-chip bottlenecks — is now reaching into datacenter hardware.

### NVIDIA Grace Hopper / GB200

The **GH200 Grace Hopper Superchip** combines:
- **Hopper H100 GPU** (80 GB HBM3, 3.35 TB/s)
- **Grace ARM CPU** (72 Neoverse V2 cores, 480 GB LPDDR5X at 512 GB/s)
- Connected by **NVLink-C2C** (die-to-die link at 900 GB/s bidirectional; coherent cache access)

The NVLink-C2C link is **7× wider than PCIe Gen5** and is **cache-coherent** — the CPU and GPU share the same coherency domain, so the GPU can directly access CPU LPDDR (and vice versa) without copies. This is the datacenter equivalent of a phone SoC's unified memory, applied to a 700 W system.

**GB200 (Blackwell)** extends this further: two B200 GPU dies + one Grace CPU on one board, with 1.8 TB of total memory visible as one pool.

### AMD MI300A APU

AMD's **MI300A** (Accelerated Processing Unit, 2023) is explicitly a datacenter SoC:
- **3 CPU chiplets** (Zen 4, 24 cores, on 5 nm)
- **6 GPU compute chiplets** (CDNA3, on 5 nm)
- **4 HBM3 stacks** (128 GB, 5.3 TB/s)
- All connected on a **silicon interposer** (2.5D packaging)
- Unified memory: CPU and GPU share the same HBM3 pool — no copies, no PCIe

The MI300A delivers **~383 TFLOPS** FP16 (GPU compute only) at 750 W, with the CPU and GPU colocated and sharing 5.3 TB/s of HBM3 bandwidth. For LLM inference, the unified memory eliminates the CPU→GPU weight-copy latency that plagues discrete GPU systems.

### The Trend

<div class="diagram">
<div class="diagram-title">The Datacenter Superchip Trend (2020–2025)</div>
<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">2020</div>
    <div class="timeline-desc"><strong>A100</strong> — GPU only, 80 GB HBM2e, 2 TB/s. CPU and GPU discrete, PCIe or NVLink between boards.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2022</div>
    <div class="timeline-desc"><strong>GH100 / GH200</strong> — Grace Hopper: ARM CPU + H100 GPU on one package, NVLink-C2C at 900 GB/s. First datacenter CPU+GPU SoC.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2023</div>
    <div class="timeline-desc"><strong>MI300A</strong> — AMD: CPU chiplets + GPU chiplets + HBM3 on interposer. First fully unified HBM-backed CPU+GPU datacenter SoC.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2024</div>
    <div class="timeline-desc"><strong>GB200 / NVL72</strong> — NVIDIA: 2×B200 GPU + 1×Grace CPU per "superchip"; 72 such superchips in one NVL72 rack with 1.8 TB pooled memory.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2025+</div>
    <div class="timeline-desc"><strong>Co-packaged optics / CXL pooling</strong> — Memory and compute increasingly disaggregated across optical links; see <a href="./35_where_the_industry_is_headed.md">Ch 35</a> and <a href="./appendix_c_photonics.md">Appendix C</a>.</div>
  </div>
</div>
</div>

---

## SoC Interconnect: The On-Chip Fabric

All these blocks communicate over an **on-chip network (NoC)** — a hierarchy of crossbars and buses implementing the **AXI** or **CHI** protocol (from ARM's AMBA specification). See [Chapter 12](./12_buses_and_interconnects.md) for interconnect fundamentals.

In a phone SoC (e.g., Snapdragon 8 Gen 3):
- **System Cache (SLC)**: 8–16 MB shared LLC between CPU, GPU, NPU, modem — reduces DRAM traffic by catching hot data
- **NoC bandwidth**: aggregate fabric bandwidth is typically 200–400 GB/s; DRAM bandwidth is the bottleneck (68–78 GB/s for LPDDR5X on phone)
- **IOMMU**: each master (GPU, NPU, DMA) sees a virtual address space; the IOMMU translates to physical DRAM addresses, enabling safe shared memory without OS copies

The IOMMU is what makes **zero-copy tensor passing** work safely: when Core ML hands a frame from the ISP to the ANE, it passes a virtual memory handle — no byte is copied, the ANE's IOMMU maps the same physical DRAM. This is the architectural basis for the ISP ↔ NPU zero-copy path described in [Chapter 19](./19_isps.md).

---

## Case Study: Running a 70B LLM on Apple M-Series

Let's put the SoC concepts together with a concrete example. You run `llama.cpp` with a Q4_K_M quantized Llama-3 70B model on an M2 Ultra (192 GB unified memory, 800 GB/s LPDDR5).

**Step 1 — Load:** The model file (≈40 GB) is `mmap`'d — mapped into virtual address space without a copy. When a page is first accessed, it's loaded from storage. CPU, GPU, ANE all address the same physical RAM.

**Step 2 — Prefill:** The prompt tokens (e.g., 512 tokens) are processed. Each layer requires one forward pass through attention + MLP. Prefill is **compute-bound** (many tokens, large batch dimension). `llama.cpp` runs this primarily on the GPU (Metal shaders), which has higher FLOPS than the ANE for variable-shape attention.

**Step 3 — Decode:** One token per step. Each step reads:
- All attention weight matrices: $4 \times d_{model}^2 \times 4 \times \text{layers} \approx 4 \times 8192^2 \times 4 \times 80 \approx$ 86 GB of weights (at INT4: ~43 GB)
- KV-cache for the current context (512+ tokens × 80 layers × 2 × 8192 × 2 bytes ≈ ~5 GB at FP16)

At 800 GB/s, reading 43 GB of weights takes **54 ms** → ceiling of **~18 tokens/second**. In practice, llama.cpp achieves 8–12 tokens/second (overhead from KV-cache, scalar ops, memory allocation).

**Why this matters:** On a discrete RTX 4090 (24 GB VRAM), the 70B INT4 model (40 GB) does **not fit** — you get an OOM error or must shard to CPU RAM and suffer PCIe bottleneck (~20 GB/s). The M2 Ultra runs it natively at 8–12 tok/s with no drama. The unified memory architecture is the enabler.

---

## Why This Matters for Model Optimization

<div class="diagram">
<div class="diagram-title">SoC Implications for ML Practitioners</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Unified Memory Changes Model Size Limits</div>
    <div class="card-desc">On discrete GPU: model must fit in VRAM (8–80 GB). On Apple Silicon or MI300A: model can use all system RAM (up to 192 GB or 128 GB HBM). Quantization determines which models fit (Ch 25).</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Bandwidth is the Decode Bottleneck</div>
    <div class="card-desc">With unified memory and small batch decode, tok/s ≈ bandwidth / model_bytes. Quantization to INT4 gives ~4× speedup. Doubling RAM bandwidth (M3→M4 Max) gives ~1.4× speedup.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Know Which Processor Runs Your Op</div>
    <div class="card-desc">Core ML / QNN automatically assigns ops to CPU/GPU/NPU. Custom ops or unsupported dtypes fall back to CPU (slow). Always profile with Xcode Instruments / Snapdragon Profiler to see which ops fall off the NPU.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Shared Memory Budget</div>
    <div class="card-desc">GPU, NPU, and ISP share the same DRAM pool. Running camera + LLM simultaneously compresses effective bandwidth. Schedule inference outside peak ISP load or use the system cache to buffer.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Chiplets Incoming for Edge</div>
    <div class="card-desc">UCIe-based chiplet SoCs will let edge devices mix compute chiplets. Future phones may have a swappable NPU chiplet. Design your deployment stack to abstract the compute substrate.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Datacenter Superchips Unify Memory</div>
    <div class="card-desc">GH200 / MI300A bring unified memory to datacenter. Large models that required tensor-parallel multi-GPU can run single-node. Check if your serving stack is updated to exploit Grace Hopper / MI300A's coherent CPU+GPU addressing.</div>
  </div>
</div>
</div>

---

## Key Takeaways

- An **SoC** integrates CPU, GPU, NPU, DSP, ISP, memory controller, video codec, and I/O into one die or package, connected by an on-chip **NoC fabric** — eliminating the PCIe/board-level bottlenecks of discrete systems.
- The defining feature for ML is **unified/shared memory**: all processors address the same physical DRAM through IOMMUs, enabling zero-copy tensor passing between ISP → NPU → GPU → CPU.
- **Unified memory removes the VRAM limit**: a 70B INT4 model that cannot fit on a 24 GB GPU VRAM runs natively on a Mac with 128 GB unified memory, at bandwith-limited speeds (~8–18 tok/s depending on generation).
- **Apple M-series memory bandwidth** — not TOPS — is the binding constraint for LLM decode; INT4 quantization is ~4× more effective than compute optimization for improving tokens/second on these chips.
- **Chiplets** solve monolithic die limits (reticle ~858 mm², yield degradation): decompose the SoC into smaller dies on different process nodes, assembled on an interposer. **UCIe** is the open standard for chiplet-to-chiplet interconnects.
- The **datacenter superchip trend** (Grace Hopper, GB200, MI300A) brings SoC-style unified memory to server hardware — CPU and GPU share the same HBM pool with cache-coherent links, removing the copy bottleneck for large-model inference.
- Mobile SoC ML runtimes (**Core ML**, **QNN**) automatically partition neural networks across NPU/GPU/CPU; profiling which ops fall back to CPU is essential for achieving peak mobile inference performance.

---

*Next: [Chapter 23 — Number Formats Deep Dive](./23_number_formats_deep_dive.md), where we examine exactly how FP32, BF16, FP8, INT8, FP4, and microscaling formats are represented in bits — and how those bit layouts reshape the silicon that computes with them.*

[← Back to Table of Contents](./README.md)
