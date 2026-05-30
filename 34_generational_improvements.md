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
    <div class="card-desc">GPU-to-GPU bandwidth for model parallelism. NVLink gen-over-gen doubling (300→600→900→3600 GB/s aggregate). Enables larger distributed workloads efficiently (Ch 12).</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">6. Packaging & Multi-Die</div>
    <div class="card-desc">Chiplets, CoWoS, 3D stacking (Ch Appendix E). Blackwell B200 is two dies in one package. Scales area beyond reticle limits. Expensive in packaging cost and power.</div>
  </div>
</div>
</div>

### The End of Dennard: Where Gains Now Come From

Before ~2005, the dominant lever was Dennard scaling: shrink the node → get more transistors *and* lower power density → raise clock frequency. That lever is gone ([Chapter 1](./01_what_is_a_chip.md)). Modern generational gains come overwhelmingly from:

1. **Specialization** — tensor cores, sparse tensor cores, FP8/FP4 units dedicated to specific matrix operations
2. **Memory and packaging** — HBM generations, CoWoS interposers, 3D stacking
3. **Lower precision** — each new dtype halves bits and doubles ops (the "cheapest FLOPs" lever)
4. **Die count scaling** — multi-chip modules exceeding reticle limits

Process node still matters, but it contributes a fraction of total performance gain compared to the Dennard era. A useful decomposition: the H100 → B200 peak FP8 TFLOP/s improvement is ~2×. That comes from:
- ~1.3× from the dual-die architecture (more SMs)
- ~1.3× from FP8 enhancements (FP8 with sparsity in B200)
- ~1.1× from process/frequency improvements
- Combined: ~1.3 × 1.3 × 1.1 ≈ 1.86×, explaining the reported ~2× figure

---

## NVIDIA Datacenter GPU Family History

The most important table for an ML practitioner. All figures are for SXM/HBM variants (highest-performance class). Peak TFLOP/s shown at the stated dtype.

| GPU | Generation | Year | Process | Transistors | FP32 TFLOP/s | FP16 TFLOP/s | BF16 TFLOP/s | FP8 TFLOP/s | HBM GB | HBM BW TB/s | NVLink BW (total) | TDP (W) |
|-----|-----------|------|---------|-------------|:------------:|:------------:|:------------:|:-----------:|:------:|:-----------:|:-----------------:|:-------:|
| P100 SXM2 | Pascal | 2016 | 16nm (TSMC) | 15.3 B | 10.6 | 21.2 | — | — | 16 | 0.72 | NVLink 1: 160 GB/s | 300 |
| V100 SXM2 | Volta | 2017 | 12nm (TSMC) | 21.1 B | 15.7 | 125 (TC) | — | — | 32 | 0.90 | NVLink 2: 300 GB/s | 300 |
| A100 SXM4 | Ampere | 2020 | 7nm (TSMC) | 54.2 B | 19.5 | 312 (TC) | 312 (TC) | — | 80 | 2.0 | NVLink 3: 600 GB/s | 400 |
| H100 SXM5 | Hopper | 2022 | 4nm (TSMC N4) | 80 B | 67 | 989 (TC) | 989 (TC) | 1,979 (TC) | 80 | 3.35 | NVLink 4: 900 GB/s | 700 |
| H200 SXM5 | Hopper | 2024 | 4nm (TSMC N4) | 80 B | 67 | 989 (TC) | 989 (TC) | 1,979 (TC) | 141 | 4.8 | NVLink 4: 900 GB/s | 700 |
| B200 SXM | Blackwell | 2024 | 4nm (TSMC 4NP) | 208 B (2-die) | ~90 | ~4,500 (TC) | ~4,500 (TC) | ~9,000 (TC) | 192 | 8.0 | NVLink 5: 1,800 GB/s | 1,000 |

**TC** = Tensor Core (matrix-multiply units); **—** = not supported / not meaningful at that generation.

Key observations from this table:

- **FP16 Tensor Core TFLOP/s** grew from 21 (P100 non-TC) → 125 → 312 → 989 → ~4,500 over 8 years: roughly **215× in 8 years, or doubling every ~13 months**. But FP32 non-tensor grew only ~8.5× over the same period.
- **Memory bandwidth**: 0.72 → 8.0 TB/s ≈ 11× in 8 years — **roughly doubling every 2.5 years**. Far slower than compute.
- **The bandwidth gap** (compute growth ÷ bandwidth growth): 215 / 11 ≈ **20× widening of the compute-to-bandwidth ratio**. This is the memory wall in numbers.
- **H100 → H200**: same die, more and faster HBM (HBM3e vs HBM3). This is a pure **memory lever** generation — TFLOP/s unchanged, bandwidth +43%, capacity +76%.
- **V100 Volta** introduced Tensor Cores — a pure **architectural/precision lever** that multiplied FP16 throughput by 8× with no node change from Pascal's generation class.

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
# → compute-bound on both; ~3.2× speedup

# Example 2: LLM token generation — memory bound (AI ~5 FLOPs/byte)
print("\nA100 → H100, LLM decode (AI=5):")
print(roofline_gain(312, 989, 2.0, 3.35, workload_ai=5))
# → memory-bound on both; ~1.7× speedup (from BW, not TFLOPS)

# Example 3: LLM decode on H100 → H200 (same TFLOPS, more BW)
print("\nH100 → H200, LLM decode (AI=5):")
print(roofline_gain(989, 989, 3.35, 4.8, workload_ai=5))
# → memory-bound on both; ~1.4× speedup — exactly what H200 is designed for
```

The output from the third example explains the entire H200 product rationale: for memory-bandwidth-bound inference (token generation in LLMs), the H200 with HBM3e delivers meaningful gains while the TFLOP/s are unchanged — because bandwidth, not compute, was the bottleneck.

---

## Google TPU Family History

Google's TPU ([Chapter 16](./16_tpus_and_systolic_arrays.md)) follows a different philosophy: a **systolic array** custom ASIC optimized for matrix multiply, not a general-purpose programmable GPU. This specialization delivers exceptional utilization rates on transformer workloads at the cost of flexibility.

| TPU | Year | Process | Matrix Units | BF16 TFLOP/s (per chip) | HBM / SRAM | Memory BW | Interconnect (ICI) | Notes |
|-----|------|---------|-------------|:-----------------------:|:----------:|:---------:|:------------------:|-------|
| v1 | 2015 | 28nm | 256×256 systolic | ~92 (INT8) | 8 GB LPDDR3 | 34 GB/s | PCIe only | Inference-only; internal deployment |
| v2 | 2017 | 16nm | 128×128 × 2 MXU | 45 (BF16) | 16 GB HBM | 0.70 TB/s | ICI 2D torus | First training TPU; ICI interconnect introduced |
| v3 | 2018 | 16nm | 128×128 × 2 MXU | 123 (BF16) | 32 GB HBM | 0.90 TB/s | ICI 2D torus | Liquid cooling; 2× v2 |
| v4 | 2021 | 7nm (TSMC) | 128×128 × 4 MXU | 275 (BF16) | 32 GB HBM | 1.2 TB/s | ICI 3D torus | Optical interconnect for pod-scale |
| v5e | 2023 | 7nm | 128×128 × 4 | 197 (BF16) | 16 GB HBM | 0.82 TB/s | ICI | Cost-optimized; 2× perf/$ vs v4 |
| v5p | 2023 | 7nm | 128×128 × 4 | 459 (BF16) | 95 GB HBM | 2.76 TB/s | ICI 3D torus | High-end; Gemini training |
| v6 (Trillium) | 2024 | 4nm (TSMC) | — | ~918 (BF16) | 32 GB HBM | 4.7 TB/s | ICI | 4.7× perf vs v5e; reported June 2024 |

Key observations:
- BF16 TFLOP/s growth from v2 (45) to Trillium (~918): **20× in 7 years**. Slightly slower than NVIDIA's TC TFLOP/s growth on the same timeline, but TPUs run at higher utilization due to software co-design.
- The **ICI (Inter-Chip Interconnect)** is Google's answer to NVLink: proprietary, ultra-high-bandwidth optical/electrical fabric for TPU pod-scale (thousands of chips). This is why JAX on TPU achieves near-linear scaling at scales NVIDIA only matches with NVLink-connected DGX pods.
- v5e is a **cost-optimized** design (reduced memory, lower BW) — evidence that Google segments its hardware for training-at-scale (v5p) vs inference and experimentation (v5e), an approach every hyperscaler now copies.

---

## AMD Instinct Family History

AMD's GPU compute line has tracked TSMC process nodes closely, with the MI300X representing a fundamental packaging architecture change.

| GPU | Year | Process | GPU Dies | FP64 TFLOP/s | FP16 TFLOP/s | BF16 TFLOP/s | FP8 TFLOP/s | HBM GB | HBM BW TB/s | TDP (W) |
|-----|------|---------|----------|:------------:|:------------:|:------------:|:-----------:|:------:|:-----------:|:-------:|
| MI100 | 2020 | 7nm (TSMC) | 1 (CDNA1) | 11.5 | 184 | — | — | 32 | 1.2 | 300 |
| MI250 | 2021 | 6nm (TSMC) | 2 (CDNA2) | 45.3 | 362 | 362 | — | 128 | 3.2 | 560 |
| MI250X | 2021 | 6nm (TSMC) | 2 (CDNA2) | 47.9 | 383 | 383 | — | 128 | 3.2 | 560 |
| MI300A | 2023 | 5nm (TSMC) | 3 GPU + 3 CPU | 122 | 977 | 977 | — | 128 HBM3 | 5.3 | 550 |
| MI300X | 2023 | 5nm (TSMC) | 8 GPU dies | 164 | 1,307 | 1,307 | 2,614 | 192 HBM3 | 5.3 | 750 |
| MI325X | 2024 | 5nm (TSMC) | 8 GPU dies | 164 | 1,307 | 1,307 | 2,614 | 256 HBM3e | 6.0 | 750 |
| MI350X | 2025 (est.) | 3nm (TSMC) | — | — | ~2,600 | ~2,600 | ~5,200 | 288 HBM3e | ~1,000 | est. |

Key observations:
- **MI300X architecture**: 8 GPU compute dies (GCDs) + 4 memory I/O dies (MIDs) assembled on a single package with TSVs. The memory-die separation allows each HBM stack to connect to a dedicated MID, which then connects to all GPU dies over the internal fabric. This is a **packaging architecture** innovation, not just a process improvement.
- The jump from MI250X (128 GB) to MI300X (192 GB) to MI325X (256 GB) is a **pure HBM lever**: more HBM stacks per package, or HBM3e stacks with greater per-stack capacity (24 GB per stack vs 16 GB).
- **ROCm software gap**: despite competitive hardware specs, MI300X adoption has been constrained by ROCm maturity. ROCm 6.x (2024) significantly closed the gap for PyTorch workloads; vLLM MI300X support shipped in 2024. The hardware is there; the software ecosystem is catching up on a 2–3 year lag.

---

## Apple Silicon Family History

Apple Silicon is the only vertically integrated consumer/professional ML platform — custom ISA cores, custom GPU, custom Neural Engine, all sharing a single unified memory pool. The platform targets different use cases than NVIDIA (on-device ML, power-constrained inference), but the bandwidth-per-watt figures are competitive with discrete GPUs.

| Chip | Year | Process | CPU Cores | GPU Cores | Neural Engine | Unified Mem (max) | CPU BW GB/s | GPU BW GB/s | TDP (W, chip) | Notes |
|------|------|---------|-----------|-----------|--------------|:-----------------:|:-----------:|:-----------:|:-------------:|-------|
| M1 | 2020 | 5nm (TSMC) | 8 (4P+4E) | 8 | 16-core, 11 TOPS | 16 GB | 68 | 68 | ~15–20 | Laptop; MacBook Air |
| M1 Ultra | 2022 | 5nm (TSMC) | 20 (2×M1 Max) | 64 | 32-core, 22 TOPS | 128 GB | 800 | 800 | ~215 | 2× M1 Max joined via UltraFusion |
| M2 | 2022 | 5nm+ (TSMC) | 8 (4P+4E) | 10 | 16-core, 15.8 TOPS | 24 GB | 100 | 100 | ~20 | |
| M2 Ultra | 2023 | 5nm+ (TSMC) | 24 (2×M2 Max) | 76 | 32-core, 31.6 TOPS | 192 GB | 800 | 800 | ~215 | Mac Pro / Mac Studio |
| M3 | 2023 | 3nm (TSMC N3B) | 8 (4P+4E) | 10 | 18-core, ~18 TOPS | 24 GB | 100 | 100 | ~22 | Hardware ray tracing |
| M3 Ultra | 2025 | 3nm | 32 (2×M3 Max) | 80 | 32-core | 192 GB | 800 | 800 | ~215 | Mac Studio/Pro |
| M4 | 2024 | 3nm 2nd gen | 10 (4P+6E) | 10 | 38-core, ~38 TOPS | 32 GB | 120 | 120 | ~18 | iPad Pro; improved ANE |
| M4 Ultra | 2025 | 3nm 2nd gen | 32 (2×M4 Max) | 80 | 64-core, ~76 TOPS | 192 GB | 800 | 800 | ~215 | |

Key observations:
- **Unified Memory Architecture (UMA)**: CPU, GPU, and Neural Engine share one LPDDR5/LPDDR5X pool with a single physical address space. No `cudaMemcpy` needed; tensors reside at one location. This eliminates PCIe bandwidth as a bottleneck, giving the GPU ~68–800 GB/s depending on model (far above PCIe-attached discrete GPUs' effective ~64 GB/s on PCIe 4.0 ×16).
- **UltraFusion** (M1 Ultra/M2 Ultra/M3 Ultra): two Max-class dies bonded with a die-to-die interconnect at 2.5 TB/s, appearing as a single chip to software. Bandwidth scales linearly (2× dies = 2× bandwidth). This is Apple's answer to chiplets.
- **Neural Engine (ANE)**: dedicated matmul hardware for INT8/INT16; throughput has grown from 11 TOPS (M1) to 76 TOPS (M4 Ultra). The ANE runs CoreML workloads; for PyTorch, the GPU backend (MPS) is used instead, which delivers better flexibility.
- **ML inference sweet spot**: M2/M3/M4 Ultra with 192 GB unified memory can hold Llama-3 70B at FP16 in-memory and serve tokens entirely from unified memory bandwidth (~800 GB/s), achieving throughputs competitive with H100 for batch-1 inference — and doing so at ~215W vs H100's 700W.

---

## HBM Generation History

HBM (High Bandwidth Memory) is the stacked DRAM technology that enables AI GPU memory bandwidth. Each generation increases both per-stack bandwidth and per-stack capacity through wider bus widths, higher data rates, and taller stacks.

| Generation | Year | Channels per Stack | Bus Width (bits) | Data Rate (Gbps/pin) | BW per Stack | Capacity per Stack | Notes |
|-----------|------|-------------------|:----------------:|:-------------------:|:------------:|:------------------:|-------|
| HBM1 | 2015 | 8 | 1,024 | 1.0 | ~128 GB/s | 4–8 GB | AMD Fiji (R9 Fury); first HBM deployment |
| HBM2 | 2016 | 8 | 1,024 | 2.0 | ~256 GB/s | 8 GB | NVIDIA P100; standard AI GPU HBM for 4 years |
| HBM2e | 2019 | 8 | 1,024 | 3.6 | ~461 GB/s | 8–16 GB | NVIDIA A100 (80 GB uses 5× 16 GB stacks) |
| HBM3 | 2022 | 16 | 1,024 | 6.4 | ~819 GB/s | 16–24 GB | NVIDIA H100; doubled channel count vs HBM2e |
| HBM3e | 2024 | 16 | 1,024 | 9.6 | ~1.2 TB/s | 24–36 GB | NVIDIA H200 (141 GB, 8 stacks × ~17.6 GB), B200; higher speed HBM3 spec |
| HBM4 | 2025+ | 16+ | 2,048 | 12.0+ | ~3.2 TB/s | 32–64 GB | Samsung/SK Hynix roadmap; 2× bus width vs HBM3 |

Key observations:
- Per-stack bandwidth: 128 (HBM1) → ~1,200 GB/s (HBM3e) ≈ **9.4× in 9 years**
- HBM3 doubled bandwidth partly through channel count doubling (8→16), not just speed
- **HBM4** (2025+) doubles the bus width per stack to 2,048 bits — the most dramatic structural change since HBM1. Combined with higher data rates, it targets ~3.2 TB/s per stack
- TSMC's **CoWoS** (Chip-on-Wafer-on-Substrate) interposer packaging is required to place HBM dies adjacent to GPU dies; CoWoS capacity has been a real supply constraint for H100/B200 production ([Appendix E](./appendix_e_packaging.md))

---

## Visualizing the Trends: Compute vs Bandwidth

The following Python snippet tabulates and plots the growth of peak FP16 TFLOP/s and memory bandwidth over time for NVIDIA datacenter GPUs, computes approximate doubling times, and highlights the widening compute-to-bandwidth ratio.

```python
import math

# NVIDIA Datacenter GPU Trends (SXM/HBM variants, tensor core FP16 where available)
gpus = [
    # (name,         year, fp16_tflops, hbm_bw_tbs)
    ("P100 SXM2",   2016,    21.2,      0.72),
    ("V100 SXM2",   2017,   125.0,      0.90),
    ("A100 SXM4",   2020,   312.0,      2.00),
    ("H100 SXM5",   2022,   989.0,      3.35),
    ("H200 SXM5",   2024,   989.0,      4.80),
    ("B200 SXM",    2024,  4500.0,      8.00),
]

print(f"{'GPU':<16} {'Year':>4} {'FP16 TC TFLOP/s':>16} {'HBM BW TB/s':>12} {'Compute/BW ratio':>17}")
print("-" * 72)
for name, year, tflops, bw in gpus:
    ratio = (tflops * 1e12) / (bw * 1e12)  # FLOPs per byte
    print(f"{name:<16} {year:>4} {tflops:>16.0f} {bw:>12.2f} {ratio:>14.0f}  FLOPs/B")

# Compute doubling times (P100 baseline to B200)
baseline_tflops, baseline_bw = 21.2, 0.72
final_tflops, final_bw = 4500.0, 8.00
years = 2024 - 2016  # 8 years

tflops_doublings = math.log2(final_tflops / baseline_tflops)
bw_doublings     = math.log2(final_bw     / baseline_bw)

tflops_months_per_doubling = (years * 12) / tflops_doublings
bw_months_per_doubling     = (years * 12) / bw_doublings

print(f"\nFP16 TC TFLOP/s growth: {final_tflops/baseline_tflops:.0f}×  "
      f"({tflops_doublings:.1f} doublings in {years} years = "
      f"doubling every ~{tflops_months_per_doubling:.0f} months)")
print(f"HBM bandwidth growth:  {final_bw/baseline_bw:.1f}×  "
      f"({bw_doublings:.1f} doublings in {years} years = "
      f"doubling every ~{bw_months_per_doubling:.0f} months)")
print(f"\nCompute/BW ratio widened: {(final_tflops/final_bw) / (baseline_tflops/baseline_bw):.1f}×")
print("→ Memory wall is ~18–20× worse than in 2016 for peak TFLOP/s workloads")
```

Expected output (approximate):

```text
GPU               Year  FP16 TC TFLOP/s  HBM BW TB/s  Compute/BW ratio
------------------------------------------------------------------------
P100 SXM2         2016               21         0.72              29  FLOPs/B
V100 SXM2         2017              125         0.90             139  FLOPs/B
A100 SXM4         2020              312         2.00             156  FLOPs/B
H100 SXM5         2022              989         3.35             295  FLOPs/B
H200 SXM5         2024              989         4.80             206  FLOPs/B
B200 SXM          2024             4500         8.00             563  FLOPs/B

FP16 TC TFLOP/s growth: 212×  (7.7 doublings in 8 years = doubling every ~12 months)
HBM bandwidth growth:  11.1×  (3.5 doublings in 8 years = doubling every ~27 months)

Compute/BW ratio widened: 19.4×
→ Memory wall is ~18–20× worse than in 2016 for peak TFLOP/s workloads
```

> The compute-to-bandwidth ridge point on the H100 is ~295 FLOPs/byte. Any workload with arithmetic intensity below this number is **memory-bandwidth-bound**, not compute-bound. For LLM token generation (autoregressive decode), the arithmetic intensity of a single-token step is approximately $2 \cdot P / M$ where $P$ is parameter count and $M$ is memory bytes accessed — typically **< 10 FLOPs/byte**, far below the ridge. This is why more TFLOP/s does not linearly help LLM inference.

---

## Cross-Vendor Memory Bandwidth Comparison

For memory-bound workloads (which includes nearly all LLM inference), memory bandwidth is the primary determinant of throughput. Here is a 2024 cross-vendor comparison:

| Chip | Memory Type | Total BW TB/s | Memory GB | Compute/BW Ridge (FP16 BF16 TC) |
|------|------------|:-------------:|:---------:|:-------------------------------:|
| NVIDIA B200 SXM | HBM3e | 8.0 | 192 | ~563 FLOPs/B |
| NVIDIA H100 SXM5 | HBM3 | 3.35 | 80 | ~295 FLOPs/B |
| NVIDIA H200 SXM5 | HBM3e | 4.8 | 141 | ~206 FLOPs/B |
| AMD MI300X | HBM3 | 5.3 | 192 | ~247 FLOPs/B |
| AMD MI325X | HBM3e | 6.0 | 256 | ~218 FLOPs/B |
| Google TPU v5p | HBM2e | 2.76 | 95 | ~166 FLOPs/B |
| Google TPU Trillium | HBM | ~4.7 | 32 | ~195 FLOPs/B |
| Apple M2 Ultra | LPDDR5 unified | 0.80 | 192 | ~9 FLOPs/B |
| Apple M4 Ultra | LPDDR5x unified | 0.80 | 192 | ~8 FLOPs/B |

> The Apple M-series ridge point (~8–9 FLOPs/B) is below even LLM decode intensity in typical quantized regimes — meaning a quantized Llama model is **compute-bound** on Apple Silicon rather than memory-bound. This is why Apple M-series inference shows relatively linear scaling with quantization, while H100 decode performance is dominated by bandwidth not compute.

---

## Decomposing a Generational Step: H100 → B200

To make the improvement analysis concrete, here is a decomposition of the H100 → B200 FP8 TFLOP/s gain:

$$\text{B200 FP8 TC} \approx 9{,}000 \text{ TFLOP/s}, \quad \text{H100 FP8 TC} \approx 1{,}979 \text{ TFLOP/s}$$

$$\text{Total gain} = \frac{9{,}000}{1{,}979} \approx 4.5\times$$

Decomposed:
- **Die count**: B200 = 2 dies × ~114 B transistors each. H100 = 1 die × 80 B. SM count: B200 ~208 SMs (2-die) vs H100 132 SMs → ~1.57× from die count
- **Micro-architecture**: B200 FP8 tensor core achieves higher ops/cycle through improved dataflow and second-generation FP8 support: ~1.5× architectural improvement on same-era compute
- **Clock / node**: 4NP → 4NP, same node generation; modest clock improvements ~1.05×
- **Combined**: 1.57 × 1.5 × 1.05 ≈ **2.5–4.5×** (range reflects uncertainty in SM-level MMA throughput specs)

This decomposition reveals that the B200's gain is primarily **architectural and die-count driven**, not process-node driven. We are now in an era where process scaling contributes a minority of generational performance improvement — a structural shift from the Dennard era.

---

## Why This Matters for Model Optimization

Understanding the levers tells you which generational upgrades actually benefit your workload:

**Training on large models (compute-bound):** You benefit from TFLOP/s increases, dtype support (FP8 training support in H100 → B200 is critical), and NVLink bandwidth for tensor parallelism. Jumping from A100 to H100 gave ~3× for training due to tensor core improvements.

**Inference decode (memory-bandwidth-bound):** You benefit from bandwidth and memory capacity, not TFLOP/s. The H100 → H200 upgrade makes sense for inference (more HBM3e bandwidth and capacity) while delivering identical training throughput. The MI300X's 192 GB is a capacity argument: fewer GPUs needed per model = fewer servers = lower inference cost.

**On-device / edge (power-constrained):** Apple M-series and Qualcomm AI chips lead here. The unified memory architecture means the "GPU BW" numbers in the table are fully available without PCIe overhead, enabling practical LLM inference on laptop-class devices.

**The precision lever is the most reliable:** Every time a new dtype is supported (INT8, FP8, FP4), you can capture it without new hardware. Going from BF16 to FP8 on H100 gives ~2× compute throughput on the same chip — but only if your model can tolerate it ([Chapter 24](./24_datatypes_and_silicon.md), [Chapter 25](./25_quantization_hardware_view.md), [Chapter 30](./30_datatypes_drive_optimization_choices.md)).

**"FLOPs went 5× but my decode didn't" — explained:** If your workload's arithmetic intensity is below the ridge point (likely for all autoregressive generation), TFLOP/s improvements deliver sub-linear speedups. The gain is entirely bounded by bandwidth growth, which is ~3× slower than compute growth. This is the memory wall in practice, not theory.

---

## Key Takeaways

- The six levers of generational improvement are: **process node**, **architecture/tensor cores**, **lower precision support**, **memory capacity & bandwidth**, **interconnect**, and **multi-die packaging** — and they affect workloads very differently.
- **Compute throughput (FP16 TC) has doubled roughly every 12 months** for NVIDIA GPUs since Volta; **memory bandwidth doubles every ~27 months** — a ~2× divergence rate, compounding to a ~20× wider compute-to-bandwidth ratio over 8 years.
- **H100 → H200** is a pure memory lever: same TFLOP/s, more and faster HBM — designed specifically for memory-bandwidth-bound inference.
- **AMD MI300X/MI325X** is a packaging-lever product: more HBM stacks per package than any competing GPU, targeting inference serving of 70B+ models that would require 2+ H100s.
- **Apple M-series** achieves uniquely low compute/bandwidth ratios via unified memory, making quantized LLM inference compute-bound (rather than memory-bound) and delivering competitive tokens/second per watt.
- **Google TPU** trades hardware flexibility for systolic array utilization; the ICI optical interconnect enables near-linear scaling at pod scale in ways NVLink + InfiniBand matches only with significant engineering overhead.
- For your workload: first measure arithmetic intensity, compare it to the target hardware's ridge point, then choose the generation/vendor whose improved bottleneck resource (compute vs bandwidth) aligns with your actual constraint.

---

*Next: [Chapter 35 — Where the Industry Is Headed](./35_where_the_industry_is_headed.md), where we look beyond current roadmaps to the structural shifts — chiplets, CXL memory disaggregation, co-packaged optics, in-memory compute, and the end of easy scaling — that will reshape ML hardware over the next decade.*

[← Back to Table of Contents](./README.md)
