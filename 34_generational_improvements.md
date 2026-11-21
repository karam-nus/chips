---
title: "Chapter 34 — Generational Improvements: Reading Hardware History"
---

[← Back to Table of Contents](./README.md)

# Chapter 34 — Generational Improvements: Reading Hardware History

Every GPU generation announcement comes with a table of numbers: TFLOP/s, GB/s, GB, TDP, process node. Most practitioners read these tables superficially — "it's 2× faster" — and miss the more interesting story underneath: **which levers actually moved, and what that tells you about the next generation and about your workload.** A 3× TFLOP/s increase might come from more dies (doesn't help memory-bound code), from lower precision support (only helps if your loss curve tolerates it), or from doubled HBM bandwidth (helps nearly everything). These are not equivalent.

This chapter gives you the tools to read hardware history like an analyst rather than a consumer. We will walk through the complete generational tables for NVIDIA datacenter GPUs, Google TPUs, AMD Instinct, Apple Silicon, and HBM memory generations — then derive the underlying improvement rates, decompose one real generational step, and write code that visualizes the trends. The repeated finding: **compute throughput has outpaced memory bandwidth by roughly 3–4× over the past decade**, which is the hardware-level reason the memory wall ([Chapter 11](./11_memory_wall_and_bandwidth.md)) keeps getting worse.

> **The one-sentence version:** Each hardware generation combines multiple improvement levers — process shrink, architectural changes, new precision support, more memory, and faster interconnects — and you can only benefit from levers that align with your workload's actual bottleneck.

---

## The Six Levers of Generational Improvement

Modern chip generations improve along six distinct axes. Understanding which axes moved tells you which workloads actually benefit.

<div class="diagram">
<div class="diagram-title">The Six Levers of Generational Improvement</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">1. Process Node Shrink</div>
    <div class="card-desc">More transistors/mm², lower energy/switch. Historically 2× transistors per node, now ~1.5–1.7× at advanced nodes. Enables everything else — but it's slower now.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">2. Architecture — IPC/Tensor Cores</div>
    <div class="card-desc">Better utilization, more tensor cores per SM, new matmul dataflow, deeper pipelines. Independent of node. Volta→Ampere tensor core redesign gave 2× MMA throughput on same node-class.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">3. New Datatypes / Precision</div>
    <div class="card-desc">FP16→BF16→FP8→FP4. Each halving of bitwidth roughly doubles ops/cycle for matmul (if hardware supports it). Cheapest FLOP/s gains — but require model tolerance (Ch 23–24).</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">4. Memory Capacity & Bandwidth</div>
    <div class="card-desc">HBM generations (HBM2→HBM3→HBM4) grow both capacity and per-stack bandwidth. Scaling up HBM stacks per chip increases total BW. This lever is hardest to scale quickly.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">5. Interconnect (NVLink / CXL)</div>
    <div class="card-desc">GPU-to-GPU bandwidth for model parallelism. NVLink roughly doubles each generation (300 → 600 → 900 → 1,800 → 3,600 GB/s per GPU, V100 to Rubin). Enables larger distributed workloads efficiently (Ch 12).</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">6. Packaging & Multi-Die</div>
    <div class="card-desc">Chiplets, CoWoS, 3D stacking ([Appendix E](./appendix_e_packaging.html)). Blackwell B200 is two dies in one package. Scales area beyond reticle limits. Expensive in packaging cost and power.</div>
  </div>
</div>
</div>

### The End of Dennard: Where Gains Now Come From

Before ~2005, the dominant lever was Dennard scaling: shrink the node → get more transistors *and* lower power density → raise clock frequency. That lever is gone ([Chapter 1](./01_what_is_a_chip.md)). Modern generational gains come overwhelmingly from:

1. **Specialization** — tensor cores, sparse tensor cores, FP8/FP4 units dedicated to specific matrix operations
2. **Memory and packaging** — HBM generations, CoWoS interposers, 3D stacking
3. **Lower precision** — each new dtype halves bits and doubles ops (the "cheapest FLOPs" lever)
4. **Die count scaling** — multi-chip modules exceeding reticle limits

Process node still matters, but it contributes a fraction of total performance gain compared to the Dennard era. A useful decomposition: the H100 → B200 dense FP8 improvement is 4,500 / 1,979 ≈ 2.3×. That comes from:
- ~1.1× more streaming multiprocessors, spread across two dies instead of one
- ~2× tensor throughput per SM (fifth-generation Tensor Cores)
- ~1.0× from clocks (both are TSMC 4N-class; there was no real node shrink)
- Combined: 1.1 × 2 × 1.0 ≈ 2.2×. Count the new FP4 format and the headline becomes 9,000 / 1,979 ≈ 4.5×, which is the number most marketing uses

---

## NVIDIA Datacenter GPU Family History

The most important table for an ML practitioner. All figures are for SXM/HBM variants (highest-performance class). Peak TFLOP/s shown at the stated dtype.

| GPU | Generation | Year | Process | Transistors | FP32 TFLOP/s | FP16/BF16 TFLOP/s | FP8 TFLOP/s | FP4 TFLOP/s | HBM GB | HBM BW TB/s | NVLink (per GPU) | TDP (W) |
|-----|-----------|------|---------|-------------|:------------:|:------------:|:-----------:|:-----------:|:------:|:-----------:|:-----------------:|:-------:|
| P100 SXM2 | Pascal | 2016 | 16nm (TSMC) | 15.3 B | 10.6 | 21.2 (no TC) | — | — | 16 | 0.72 | NVLink 1: 160 GB/s | 300 |
| V100 SXM2 | Volta | 2017 | 12nm (TSMC) | 21.1 B | 15.7 | 125 (TC) | — | — | 32 | 0.90 | NVLink 2: 300 GB/s | 300 |
| A100 SXM4 | Ampere | 2020 | 7nm (TSMC) | 54.2 B | 19.5 | 312 (TC) | — | — | 80 | 2.0 | NVLink 3: 600 GB/s | 400 |
| H100 SXM5 | Hopper | 2022 | TSMC 4N | 80 B | 67 | 989 (TC) | 1,979 | — | 80 | 3.35 | NVLink 4: 900 GB/s | 700 |
| H200 SXM5 | Hopper | 2024 | TSMC 4N | 80 B | 67 | 989 (TC) | 1,979 | — | 141 | 4.8 | NVLink 4: 900 GB/s | 700 |
| B200 SXM | Blackwell | 2024 | TSMC 4NP | 208 B (2 dies) | ~80 | 2,250 (TC) | 4,500 | 9,000 | 192 | 8.0 | NVLink 5: 1,800 GB/s | 1,000 |
| B300 SXM | Blackwell Ultra | 2025 | TSMC 4NP | 208 B (2 dies) | ~75 | ~2,250 | ~4,500 | ~15,000 | 288 | 8.0 | NVLink 5: 1,800 GB/s | 1,100–1,400 |
| Rubin | Rubin | 2026 | TSMC 3 nm class | 336 B (2 dies) | n/p | n/p | n/p | ~35,000 (training) | 288 HBM4 | ~20 | NVLink 6: 3,600 GB/s | n/p |

All tensor figures are **dense** (no 2:4 sparsity bonus, which doubles the headline numbers NVIDIA usually quotes). **TC** = Tensor Core; **—** = not supported; **n/p** = not yet published. Rubin's figure is NVIDIA's NVFP4 training rate (50 PF for inference); its HBM4 bandwidth was announced at 22 TB/s and is reported nearer 20 TB/s in shipping parts. The same data, for every vendor, drives the [Accelerator Atlas](./atlas.html).

Key observations from this table:

- **FP16 Tensor Core TFLOP/s** grew from 21 (P100 non-TC) → 125 → 312 → 989 → 2,250 over 8 years: roughly **106× in 8 years, or doubling every ~14 months**. Counting the new low-precision formats (FP8, then FP4) the headline grew far faster: B300's FP4 rate is ~700× P100's FP16. But FP32 non-tensor grew only ~8.5× over the same period.
- **Memory bandwidth**: 0.72 → 8.0 TB/s ≈ 11× in 8 years — **roughly doubling every 2.3 years**. Far slower than compute. HBM4 on Rubin is the first jump large enough (~2.5×) to narrow the gap for a generation.
- **The bandwidth gap** (compute growth ÷ bandwidth growth): 106 / 11 ≈ **10× widening of the compute-to-bandwidth ratio at the same precision**, and ~40× if you let the precision fall from FP16 to FP4. This is the memory wall in numbers.
- **H100 → H200**: same die, more and faster HBM (HBM3e vs HBM3). This is a pure **memory lever** generation — TFLOP/s unchanged, bandwidth +43%, capacity +76%.
- **V100 Volta** introduced Tensor Cores — a pure **architectural/precision lever** that multiplied FP16 throughput by ~6× (21 → 125 TFLOP/s) on 12 nm, a refinement of Pascal's 16 nm rather than a new node.

### Reading the Spec Sheet: What Actually Helps Your Workload

```python
# Roofline perspective: will a generation upgrade help your workload?
# Arithmetic Intensity (AI) = FLOPs per byte of memory traffic

def roofline_gain(
    old_tflops: float, new_tflops: float,
    old_bw_tbs: float, new_bw_tbs: float,
    workload_ai: float  # FLOPs per byte
) -> dict:
    """
    Given arithmetic intensity of your workload,
    compute whether you're compute-bound or memory-bound
    on old and new hardware, and predict effective speedup.
    """
    # Ridge point = TFLOPS / BW (in FLOPs/byte)
    old_ridge = (old_tflops * 1e12) / (old_bw_tbs * 1e12)
    new_ridge = (new_tflops * 1e12) / (new_bw_tbs * 1e12)

    old_roofline = min(old_tflops * 1e12,
                       workload_ai * old_bw_tbs * 1e12)
    new_roofline = min(new_tflops * 1e12,
                       workload_ai * new_bw_tbs * 1e12)

    speedup = new_roofline / old_roofline
    old_bound = "compute" if workload_ai > old_ridge else "memory"
    new_bound = "compute" if workload_ai > new_ridge else "memory"

    return {
        "old_ridge_pt": f"{old_ridge:.0f} FLOPs/byte",
        "new_ridge_pt": f"{new_ridge:.0f} FLOPs/byte",
        "old_bound": old_bound,
        "new_bound": new_bound,
        "predicted_speedup": f"{speedup:.2f}x",
    }

# Example 1: Dense training matmul — compute bound (AI ~300 FLOPs/byte)
print("A100 → H100, training matmul (AI=300):")
print(roofline_gain(312, 989, 2.0, 3.35, workload_ai=300))
# → compute-bound on both (just); 3.17× speedup

# Example 2: LLM token generation — memory bound (AI ~5 FLOPs/byte)
print("\nA100 → H100, LLM decode (AI=5):")
print(roofline_gain(312, 989, 2.0, 3.35, workload_ai=5))
# → memory-bound on both; 1.68× speedup (from BW, not TFLOPS)

# Example 3: LLM decode on H100 → H200 (same TFLOPS, more BW)
print("\nH100 → H200, LLM decode (AI=5):")
print(roofline_gain(989, 989, 3.35, 4.8, workload_ai=5))
# → memory-bound on both; 1.43× speedup — exactly what H200 is designed for
```

The output from the third example explains the entire H200 product rationale: for memory-bandwidth-bound inference (token generation in LLMs), the H200 with HBM3e delivers meaningful gains while the TFLOP/s are unchanged — because bandwidth, not compute, was the bottleneck.

---

## Google TPU Family History

Google's TPU ([Chapter 16](./16_tpus_and_systolic_arrays.md)) follows a different philosophy: a **systolic array** custom ASIC optimized for matrix multiply, not a general-purpose programmable GPU. This specialization delivers exceptional utilization rates on transformer workloads at the cost of flexibility.

| TPU | Year | Process | Matrix Units | Peak per chip | HBM / SRAM | Memory BW | Interconnect (ICI) | Notes |
|-----|------|---------|-------------|:-----------------------:|:----------:|:---------:|:------------------:|-------|
| v1 | 2015 | 28nm | 256×256 systolic | 92 TOPS (INT8) | 8 GB DDR3 | 34 GB/s | PCIe only | Inference-only; internal deployment |
| v2 | 2017 | 16nm | 128×128 MXUs | 45 TFLOP/s (BF16) | 16 GB HBM | 0.70 TB/s | ICI 2D torus | First training TPU; ICI interconnect introduced |
| v3 | 2018 | 16nm | 128×128 MXUs | 123 (BF16) | 32 GB HBM | 0.90 TB/s | ICI 2D torus | Liquid cooling |
| v4 | 2021 | 7nm (TSMC) | 128×128 MXUs | 275 (BF16) | 32 GB HBM | 1.2 TB/s | ICI 3D torus | Optical circuit switches for pod-scale |
| v5e | 2023 | n/p | 128×128 MXUs | 197 (BF16) | 16 GB HBM | 0.82 TB/s | ICI | Cost-optimized |
| v5p | 2023 | n/p | 128×128 MXUs | 459 (BF16) | 95 GB HBM2e | 2.76 TB/s | ICI 3D torus | High-end; Gemini training |
| v6e (Trillium) | 2024 | n/p | 256×256 MXUs | 918 (BF16) | 32 GB HBM | 1.64 TB/s | ICI | 4.7× peak compute of v5e |
| v7 (Ironwood) | 2025 | n/p | 256×256 MXUs | 4,614 (FP8); ~2,300 BF16 | 192 GB HBM3e | 7.37 TB/s | ICI, 9,216-chip pods | First TPU with native FP8 |
| TPU 8t | 2026 (announced) | n/p | — | 12,600 (FP4) | 216 GB HBM3e | ~6.5 TB/s | Virgo network | Training half of a split generation |
| TPU 8i | 2026 (announced) | n/p | — | 10,100 (FP4) | 288 GB HBM3e + 384 MB SRAM | ~8.6 TB/s | Boardfly | Inference half; large on-chip SRAM |

Key observations:
- BF16 TFLOP/s growth from v2 (45) to Ironwood (~2,300): **~50× in 8 years**, before counting FP8 and FP4. Slightly slower than NVIDIA's TC TFLOP/s growth on the same timeline, but TPUs run at higher utilization due to software co-design.
- The **ICI (Inter-Chip Interconnect)** is Google's answer to NVLink: proprietary, ultra-high-bandwidth optical/electrical fabric for TPU pod-scale (thousands of chips). This is why JAX on TPU achieves near-linear scaling at scales NVIDIA only matches with NVLink-connected DGX pods.
- v5e is a **cost-optimized** design (reduced memory, lower BW) — evidence that Google segments its hardware for training-at-scale (v5p) vs inference and experimentation (v5e), and in 2026 it made the split explicit with separate TPU 8t and TPU 8i chips; an approach every hyperscaler now copies.

---

## AMD Instinct Family History

AMD's GPU compute line has tracked TSMC process nodes closely, with the MI300X representing a fundamental packaging architecture change.

| GPU | Year | Process | GPU Dies | FP64 TFLOP/s (matrix) | FP16/BF16 TFLOP/s | FP8 TFLOP/s | FP4 TFLOP/s | HBM GB | HBM BW TB/s | TDP (W) |
|-----|------|---------|----------|:------------:|:------------:|:-----------:|:-----------:|:------:|:-----------:|:-------:|
| MI100 | 2020 | 7nm (TSMC) | 1 (CDNA1) | 11.5 | 184.6 | — | — | 32 HBM2 | 1.2 | 300 |
| MI250 | 2021 | 6nm (TSMC) | 2 (CDNA2) | 90.5 | 362 | — | — | 128 HBM2e | 3.2 | 560 |
| MI250X | 2021 | 6nm (TSMC) | 2 (CDNA2) | 95.7 | 383 | — | — | 128 HBM2e | 3.2 | 560 |
| MI300A | 2023 | 5nm + 6nm | 6 GPU XCDs + 3 CPU CCDs | 122.6 | 981 | 1,961 | — | 128 HBM3 | 5.3 | 550 |
| MI300X | 2023 | 5nm + 6nm | 8 XCDs | 163.4 | 1,307 | 2,615 | — | 192 HBM3 | 5.3 | 750 |
| MI325X | 2024 | 5nm + 6nm | 8 XCDs | 163.4 | 1,307 | 2,615 | — | 256 HBM3e | 6.0 | 1,000 |
| MI355X | 2025 | N3P + N6 | 8 XCDs | ~79 | ~2,500 | ~5,000 | ~10,000 (FP6 same) | 288 HBM3e | 8.0 | 1,400 |
| MI455X | 2026 (announced) | 2 nm class | n/p | n/p | n/p | ~20,000 | ~40,000 | 432 HBM4 | ~19.6 | n/p |

Key observations:
- **MI300X architecture**: 8 accelerator compute dies (XCDs) stacked on 4 I/O dies (IODs), assembled on a single package with TSVs. The memory-die separation allows each HBM stack to connect to a dedicated MID, which then connects to all GPU dies over the internal fabric. This is a **packaging architecture** innovation, not just a process improvement.
- The jump from MI250X (128 GB) to MI300X (192 GB) to MI325X (256 GB) is a **pure HBM lever**: more HBM stacks per package, or HBM3e stacks with greater per-stack capacity (32 GB per stack in MI325X vs 24 GB).
- **ROCm software gap**: despite competitive hardware specs, MI300X adoption has been constrained by ROCm maturity. ROCm 6.x (2024) significantly closed the gap for PyTorch workloads; vLLM MI300X support shipped in 2024. The hardware is there; the software ecosystem is catching up on a 2–3 year lag.

---

## Apple Silicon Family History

Apple Silicon is the only vertically integrated consumer/professional ML platform — custom ISA cores, custom GPU, custom Neural Engine, all sharing a single unified memory pool. The platform targets different use cases than NVIDIA (on-device ML, power-constrained inference), but the bandwidth-per-watt figures are competitive with discrete GPUs.

| Chip | Year | Process | CPU Cores | GPU Cores | Neural Engine | Unified Mem (max) | Memory BW GB/s | Notes |
|------|------|---------|-----------|-----------|--------------|:-----------------:|:-----------:|-------|
| M1 | 2020 | 5nm (TSMC N5) | 8 (4P+4E) | 8 | 16-core, 11 TOPS | 16 GB | 68 | MacBook Air |
| M1 Ultra | 2022 | N5 | 20 (2×M1 Max) | 64 | 32-core, 22 TOPS | 128 GB | 800 | 2× M1 Max joined via UltraFusion |
| M2 | 2022 | N5P | 8 (4P+4E) | 10 | 16-core, 15.8 TOPS | 24 GB | 100 | |
| M2 Ultra | 2023 | N5P | 24 (2×M2 Max) | 76 | 32-core, 31.6 TOPS | 192 GB | 800 | Mac Pro / Mac Studio |
| M3 | 2023 | N3B | 8 (4P+4E) | 10 | 16-core, 18 TOPS | 24 GB | 100 | Hardware ray tracing |
| M4 | 2024 | N3E | 10 (4P+6E) | 10 | 16-core, 38 TOPS | 32 GB | 120 | iPad Pro first |
| M4 Max | 2024 | N3E | 16 (12P+4E) | 40 | 16-core, 38 TOPS | 128 GB | 546 | No M4 Ultra was released |
| M3 Ultra | 2025 | N3B | 32 (24P+8E) | 80 | 32-core | 512 GB | ~819 | Mac Studio; UltraFusion |
| M5 | 2025 | N3P | 10 (4P+6E) | 10, each with a neural accelerator | 16-core | 32 GB | ~153 | >4× peak GPU AI compute vs M4 |

Key observations:
- **Unified Memory Architecture (UMA)**: CPU, GPU, and Neural Engine share one LPDDR5/LPDDR5X pool with a single physical address space. No `cudaMemcpy` needed; tensors reside at one location. This eliminates PCIe bandwidth as a bottleneck, giving the GPU ~68–800 GB/s depending on model (far above PCIe-attached discrete GPUs' effective ~64 GB/s on PCIe 4.0 ×16).
- **UltraFusion** (M1 Ultra/M2 Ultra/M3 Ultra): two Max-class dies bonded with a die-to-die interconnect at 2.5 TB/s, appearing as a single chip to software. Bandwidth scales linearly (2× dies = 2× bandwidth). This is Apple's answer to chiplets.
- **Neural Engine (ANE)**: dedicated matmul hardware for INT8/INT16; throughput has grown from 11 TOPS (M1) to 38 TOPS (M4). With M5, Apple also put matrix "neural accelerators" inside every GPU core, a sign that the GPU, not the ANE, is where large-model inference on a Mac now runs. The ANE runs CoreML workloads; for PyTorch, the GPU backend (MPS) is used instead, which delivers better flexibility.
- **ML inference sweet spot**: M2 Ultra (192 GB) and M3 Ultra (up to 512 GB) can hold Llama-3 70B at FP16 in memory and serve tokens entirely from unified memory bandwidth (~800 GB/s), achieving throughputs competitive with H100 for batch-1 inference — and doing so at ~215W vs H100's 700W.

---

## HBM Generation History

HBM (High Bandwidth Memory) is the stacked DRAM technology that enables AI GPU memory bandwidth. Each generation increases both per-stack bandwidth and per-stack capacity through wider bus widths, higher data rates, and taller stacks.

| Generation | Year | Channels per Stack | Bus Width (bits) | Data Rate (Gbps/pin) | BW per Stack | Capacity per Stack | Notes |
|-----------|------|-------------------|:----------------:|:-------------------:|:------------:|:------------------:|-------|
| HBM1 | 2015 | 8 | 1,024 | 1.0 | ~128 GB/s | 1 GB | AMD Fiji (R9 Fury, 4 stacks = 4 GB); first HBM deployment |
| HBM2 | 2016 | 8 | 1,024 | 2.0 | ~256 GB/s | 8 GB | NVIDIA P100; standard AI GPU HBM for 4 years |
| HBM2e | 2019 | 8 | 1,024 | 3.6 | ~461 GB/s | 8–16 GB | NVIDIA A100 (80 GB uses 5× 16 GB stacks) |
| HBM3 | 2022 | 16 | 1,024 | 6.4 | ~819 GB/s | 16–24 GB | NVIDIA H100; doubled channel count vs HBM2e |
| HBM3e | 2024 | 16 | 1,024 | 8–9.6 | ~1.0–1.2 TB/s | 24–36 GB | H200 (6 × 24 GB stacks, 141 GB usable), B200 (8 × 24 GB), B300 and MI355X (8 × 36 GB) |
| HBM4 | 2026 | 32 | 2,048 | 8 (JEDEC); ~10–11 shipping | ~2.0–2.8 TB/s | 36–48 GB | JEDEC JESD270-4 (Apr 2025); volume from early 2026 for Rubin and MI455X |
| HBM4E | 2027 | 32 | 2,048 | 12+ (target) | ~3+ TB/s | 48–64 GB | Samples late 2026 |

Key observations:
- Per-stack bandwidth: 128 (HBM1) → ~1,200 GB/s (HBM3e) ≈ **9.4× in 9 years**
- HBM3 doubled bandwidth partly through channel count doubling (8→16), not just speed
- **HBM4** (volume 2026) doubles the bus width per stack to 2,048 bits and the channel count to 32 — the most dramatic structural change since HBM1. At shipping data rates it delivers ~2–2.8 TB/s per stack, and its base die can now be made on a logic process, so GPU vendors can put custom logic under the DRAM
- TSMC's **CoWoS** (Chip-on-Wafer-on-Substrate) interposer packaging is required to place HBM dies adjacent to GPU dies; CoWoS capacity has been a real supply constraint for H100/B200 production ([Appendix E](./appendix_e_packaging.md))

---

## Visualizing the Trends: Compute vs Bandwidth

The following Python snippet tabulates and plots the growth of peak FP16 TFLOP/s and memory bandwidth over time for NVIDIA datacenter GPUs, computes approximate doubling times, and highlights the widening compute-to-bandwidth ratio.

```python
import math

# NVIDIA datacenter GPU trends (SXM/HBM variants; DENSE tensor-core FP16/BF16)
gpus = [
    # (name,         year, fp16_tflops, hbm_bw_tbs)
    ("P100 SXM2",   2016,    21.2,      0.72),
    ("V100 SXM2",   2017,   125.0,      0.90),
    ("A100 SXM4",   2020,   312.0,      2.00),
    ("H100 SXM5",   2022,   989.0,      3.35),
    ("H200 SXM5",   2024,   989.0,      4.80),
    ("B200 SXM",    2024,  2250.0,      8.00),
]

print(f"{'GPU':<16} {'Year':>4} {'FP16 TC TFLOP/s':>16} {'HBM BW TB/s':>12} {'Compute/BW ratio':>17}")
print("-" * 72)
for name, year, tflops, bw in gpus:
    ratio = (tflops * 1e12) / (bw * 1e12)  # FLOPs per byte
    print(f"{name:<16} {year:>4} {tflops:>16.0f} {bw:>12.2f} {ratio:>14.0f}  FLOPs/B")

# Doubling times, P100 (2016) -> B200 (2024)
baseline_tflops, baseline_bw = 21.2, 0.72
final_tflops, final_bw = 2250.0, 8.00
years = 2024 - 2016

tflops_doublings = math.log2(final_tflops / baseline_tflops)
bw_doublings     = math.log2(final_bw     / baseline_bw)

print(f"\nFP16 TC TFLOP/s growth: {final_tflops/baseline_tflops:.0f}×  "
      f"({tflops_doublings:.1f} doublings in {years} years = "
      f"doubling every ~{years*12/tflops_doublings:.0f} months)")
print(f"HBM bandwidth growth:  {final_bw/baseline_bw:.1f}×  "
      f"({bw_doublings:.1f} doublings in {years} years = "
      f"doubling every ~{years*12/bw_doublings:.0f} months)")
print(f"\nCompute/BW ratio widened: {(final_tflops/final_bw) / (baseline_tflops/baseline_bw):.1f}×  (same-precision FP16)")
print(f"Counting the new formats (B200 FP4 = 9,000 TFLOP/s): {(9000/final_bw) / (baseline_tflops/baseline_bw):.0f}×")
```

Output:

```text
GPU              Year  FP16 TC TFLOP/s  HBM BW TB/s  Compute/BW ratio
------------------------------------------------------------------------
P100 SXM2        2016               21         0.72             29  FLOPs/B
V100 SXM2        2017              125         0.90            139  FLOPs/B
A100 SXM4        2020              312         2.00            156  FLOPs/B
H100 SXM5        2022              989         3.35            295  FLOPs/B
H200 SXM5        2024              989         4.80            206  FLOPs/B
B200 SXM         2024             2250         8.00            281  FLOPs/B

FP16 TC TFLOP/s growth: 106×  (6.7 doublings in 8 years = doubling every ~14 months)
HBM bandwidth growth:  11.1×  (3.5 doublings in 8 years = doubling every ~28 months)

Compute/BW ratio widened: 9.6×  (same-precision FP16)
Counting the new formats (B200 FP4 = 9,000 TFLOP/s): 38×
```

> The compute-to-bandwidth ridge point on the H100 is ~295 FLOPs/byte. Any workload with arithmetic intensity below this number is **memory-bandwidth-bound**, not compute-bound. For LLM token generation (autoregressive decode), the arithmetic intensity of a single-token step is approximately $2 \cdot P / M$ where $P$ is parameter count and $M$ is memory bytes accessed — typically **< 10 FLOPs/byte**, far below the ridge. This is why more TFLOP/s does not linearly help LLM inference.

---

## Cross-Vendor Memory Bandwidth Comparison

For memory-bound workloads (which includes nearly all LLM inference), memory bandwidth is the primary determinant of throughput. Here is a cross-vendor comparison as of late 2026 (dense BF16 ridge points; FP8 and FP4 ridges are 2× and 4–7× higher):

| Chip | Memory Type | Total BW TB/s | Memory GB | Ridge (dense BF16) |
|------|------------|:-------------:|:---------:|:-------------------------------:|
| NVIDIA H100 SXM5 | HBM3 | 3.35 | 80 | ~295 FLOPs/B |
| NVIDIA H200 SXM5 | HBM3e | 4.8 | 141 | ~206 FLOPs/B |
| NVIDIA B200 SXM | HBM3e | 8.0 | 192 | ~281 FLOPs/B |
| NVIDIA B300 SXM | HBM3e | 8.0 | 288 | ~281 FLOPs/B (FP4: ~1,875) |
| AMD MI300X | HBM3 | 5.3 | 192 | ~247 FLOPs/B |
| AMD MI355X | HBM3e | 8.0 | 288 | ~313 FLOPs/B |
| Google TPU v5p | HBM2e | 2.76 | 95 | ~166 FLOPs/B |
| Google Trillium (v6e) | HBM | 1.64 | 32 | ~560 FLOPs/B |
| Google Ironwood (v7) | HBM3e | 7.37 | 192 | ~313 FLOPs/B (FP8: ~626) |
| Apple M2 Ultra (GPU) | LPDDR5 unified | 0.80 | 192 | ~34 FLOPs/B (27 TFLOP/s FP32/FP16) |

> The Apple M-series ridge point (~34 FLOPs/B on the M2 Ultra GPU) is an order of magnitude lower than a data-center GPU's, but batch-1 decode (1–4 FLOPs/B) is still well below it, so a quantized Llama model is **memory-bound on Apple Silicon too**: tokens/s tracks memory bandwidth and bytes per weight, exactly as on an H100. Where the low ridge shows up is **prefill**: long prompts become compute-bound quickly, which is why time-to-first-token on a Mac is slow even when generation speed is good.

---

## Decomposing a Generational Step: H100 → B200

To make the improvement analysis concrete, here is a decomposition of the H100 → B200 FP8 TFLOP/s gain:

$$\text{B200 FP8 (dense)} \approx 4{,}500 \text{ TFLOP/s}, \quad \text{H100 FP8 (dense)} \approx 1{,}979 \text{ TFLOP/s}$$

$$\text{Total gain} = \frac{4{,}500}{1{,}979} \approx 2.3\times \qquad \left(\text{or } \frac{9{,}000\ \text{FP4}}{1{,}979\ \text{FP8}} \approx 4.5\times \text{ counting the new format}\right)$$

Decomposed:
- **Die count**: B200 = 2 reticle-sized dies, 208 B transistors in total, vs one 80 B-transistor H100 die. The enabled SM count rises only ~1.1×; much of the extra area went to larger tensor cores, cache and the 10 TB/s die-to-die link.
- **Micro-architecture**: fifth-generation Tensor Cores deliver ~2× the FP8 throughput per SM per clock, and add FP6/FP4 with hardware block scaling.
- **Clock / node**: TSMC 4N → 4NP is a refinement, not a shrink; clocks are similar (~1.0×).
- **Combined**: 1.1 × 2.0 × 1.0 ≈ **2.2×** at the same precision, matching the 2.3× in the datasheets; the FP4 datapath doubles it again.

This decomposition reveals that the B200's gain is primarily **architectural and die-count driven**, not process-node driven. We are now in an era where process scaling contributes a minority of generational performance improvement — a structural shift from the Dennard era.

---

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Three of the last five NVIDIA steps (H100 → H200, B200 → B300, Blackwell → Rubin) were driven mainly by memory rather than compute. If the next five follow that pattern, which of your workloads gets faster for free, and which ones only get faster if you change precision?</p>
</div>

## Why This Matters for Model Optimization

Understanding the levers tells you which generational upgrades actually benefit your workload:

**Training on large models (compute-bound):** You benefit from TFLOP/s increases, dtype support (FP8 training support in H100 → B200 is critical), and NVLink bandwidth for tensor parallelism. Jumping from A100 to H100 gave ~3× at peak (~2–3× measured) for training, from tensor-core improvements plus FP8.

**Inference decode (memory-bandwidth-bound):** You benefit from bandwidth and memory capacity, not TFLOP/s. The H100 → H200 upgrade makes sense for inference (more HBM3e bandwidth and capacity) while delivering identical training throughput. The MI300X's 192 GB is a capacity argument: fewer GPUs needed per model = fewer servers = lower inference cost.

**On-device / edge (power-constrained):** Apple M-series and Qualcomm AI chips lead here. The unified memory architecture means the "GPU BW" numbers in the table are fully available without PCIe overhead, enabling practical LLM inference on laptop-class devices.

**The precision lever is the most reliable:** Every time a new dtype is supported (INT8, FP8, FP4), you can capture it without new hardware. Going from BF16 to FP8 on H100 gives ~2× compute throughput on the same chip — but only if your model can tolerate it ([Chapter 24](./24_datatypes_and_silicon.md), [Chapter 25](./25_quantization_hardware_view.md), [Chapter 30](./30_datatypes_drive_optimization_choices.md)).

**"FLOPs went 5× but my decode didn't" — explained:** If your workload's arithmetic intensity is below the ridge point (likely for all autoregressive generation), TFLOP/s improvements deliver sub-linear speedups. The gain is entirely bounded by bandwidth growth, which has run at about half the rate of compute growth (doubling every ~28 months versus ~14). This is the memory wall in practice, not theory.

---

## Key Takeaways

- The six levers of generational improvement are: **process node**, **architecture/tensor cores**, **lower precision support**, **memory capacity & bandwidth**, **interconnect**, and **multi-die packaging** — and they affect workloads very differently.
- **Dense FP16 tensor throughput has doubled roughly every 14 months** for NVIDIA GPUs since 2016; **memory bandwidth every ~28 months**, compounding to a ~10× wider compute-to-bandwidth ratio over 8 years at the same precision (~40× counting FP4).
- **H100 → H200** is a pure memory lever: same TFLOP/s, more and faster HBM — designed specifically for memory-bandwidth-bound inference.
- **AMD MI300X → MI355X → MI455X** is a packaging-and-capacity lever: 192 → 288 → 432 GB per package, keeping AMD at or ahead of NVIDIA on HBM per GPU.
- **Apple M-series** has a ridge point an order of magnitude below data-center GPUs and huge unified memory (up to 512 GB on M3 Ultra). Decode is still memory-bound there; prefill turns compute-bound early, which is why long prompts are slow on a Mac.
- **Google TPU** trades hardware flexibility for systolic array utilization; the ICI optical interconnect enables near-linear scaling at pod scale in ways NVLink + InfiniBand matches only with significant engineering overhead.
- For your workload: first measure arithmetic intensity, compare it to the target hardware's ridge point, then choose the generation/vendor whose improved bottleneck resource (compute vs bandwidth) aligns with your actual constraint.

## Check Yourself

<details class="check">
<summary>A vendor's new GPU has 2× the dense FP8 FLOP/s and 1.3× the bandwidth of the old one. Your decode runs at arithmetic intensity 8. What speedup should you expect?</summary>
<p>Decode at AI 8 is far below either ridge, so it is memory-bound on both: expect ~1.3×, not 2×.</p>
</details>

<details class="check">
<summary>Why is H100 → H200 called a pure memory-lever generation?</summary>
<p>Same die and the same 989 TFLOP/s BF16, but HBM3e raised bandwidth from 3.35 to 4.8 TB/s (+43%) and capacity from 80 to 141 GB (+76%).</p>
</details>

<details class="check">
<summary>Compute the BF16 ridge point of Ironwood and explain why it is similar to B200's.</summary>
<p>~2,307 TFLOP/s ÷ 7.37 TB/s ≈ 313 FLOP/byte, versus 2,250 ÷ 8.0 ≈ 281 for B200. Both vendors scaled compute and HBM bandwidth by similar factors, so their balance points converge.</p>
</details>

<details class="check">
<summary>A spec sheet says B200 does "18 PFLOPS FP4". What is the dense FP4 number, and the dense BF16 number?</summary>
<p>18 PF FP4 is the 2:4-sparse figure; dense FP4 is 9 PF. Dense BF16 is a quarter of that, 2.25 PF.</p>
</details>

<details class="check">
<summary>HBM4 doubles the interface width to 2,048 bits. Why doesn't per-stack bandwidth double over the fastest HBM3e?</summary>
<p>The JEDEC baseline pin rate dropped to 8 Gb/s (from up to 9.6 Gb/s), trading speed for width and signal integrity; shipping parts at ~10–11 Gb/s land around 2.5–2.8 TB/s per stack.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| NVIDIA H100 / Blackwell architecture whitepapers | NVIDIA, 2022 / 2024 | [resources.nvidia.com/en-us-tensor-core](https://resources.nvidia.com/en-us-tensor-core) | Dense tensor throughput by generation |
| NVIDIA GB300 NVL72 | NVIDIA, 2025 | [nvidia.com/en-us/data-center/gb300-nvl72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) | Blackwell Ultra FP4 rate and 288 GB HBM3e |
| AMD Instinct MI355X | AMD, 2025 | [amd.com/en/products/accelerators/instinct/mi3…](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html) | CDNA 4 rates, 288 GB, 8 TB/s |
| Cloud TPU documentation (v5e, v5p, v6e) | Google Cloud | [cloud.google.com/tpu/docs/system-architecture…](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm) | Per-chip TPU specifications |
| Ironwood: the first Google TPU for the age of inference | Google, 2025 | [blog.google/products/google-cloud/ironwood-tp…](https://blog.google/products/google-cloud/ironwood-tpu-age-of-inference/) | Ironwood FP8, HBM and pod size |
| JEDEC HBM4 standard (JESD270-4) | JEDEC, Apr 2025 | [jedec.org/news/pressreleases/jedec-and-indust…](https://www.jedec.org/news/pressreleases/jedec-and-industry-leaders-collaborate-release-jesd270-4-hbm4-standard) | 2,048-bit interface, 32 channels |
| Apple M3 Ultra / M5 announcements | Apple, 2025 | [apple.com/newsroom](https://www.apple.com/newsroom/) | Unified memory capacity and bandwidth |
| AI and Memory Wall | Gholami et al., 2024 | [arxiv.org/abs/2403.14123](https://arxiv.org/abs/2403.14123) | Compute vs bandwidth growth rates |

---

**Next:** [Chapter 35 — Where the Industry Is Headed](./35_where_the_industry_is_headed.md) — We look beyond current roadmaps to the structural shifts (chiplets, CXL memory, co-packaged optics, in-memory compute and the end of easy scaling) that will reshape ML hardware over the next decade.

[← Back to Table of Contents](./README.md)
