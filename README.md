---
title: "Home"
permalink: /
---

# 🔬 Chips — The Complete Guide

> **From Sand to Supercomputers**: a first-principles guide to chip design and architecture, written for the **AI/ML practitioner who optimizes models**. By the end you will understand *the hardware reasons behind every quantization, sparsity, batching, and parallelism decision you make.*

## Who This Is For

You train, fine-tune, quantize, prune, or serve neural networks. You are fluent in Python and PyTorch, comfortable with tensors, dtypes, and shapes — but the **silicon underneath** is a black box. Why is `bf16` better than `fp16` for training? Why does INT8 give a ~2× speedup but INT4 sometimes doesn't? Why does a 70B model refuse to fit on one 80 GB GPU? Why is memory bandwidth — not FLOPs — usually what you're actually waiting on? Why did 2:4 sparsity become a thing, and why not 3:7?

This guide answers all of that **from first principles** — starting at the transistor and building all the way up to multi-GPU training clusters and the post-Moore future. It is **self-contained**: every concept is explained here, from the ground up.

<div class="diagram">
<div class="diagram-title">What This Guide Covers</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🧱</div>
    <div class="card-title">Foundations</div>
    <div class="card-desc">Transistors, logic, number systems, processors, ISAs, the vocabulary of the stack</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🧠</div>
    <div class="card-title">Memory & Movement</div>
    <div class="card-desc">SRAM/DRAM/HBM, caches, the memory wall, roofline, buses & interconnects</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">⚙️</div>
    <div class="card-title">The Chips</div>
    <div class="card-desc">CPU, GPU, TPU, NPU, DSP, ISP, FPGA, ASIC, SoC — what each is and why</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">🔢</div>
    <div class="card-title">Datatypes</div>
    <div class="card-desc">FP32→FP4, bit layouts, microscaling, and how formats reshape silicon</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">🗜️</div>
    <div class="card-title">Optimization</div>
    <div class="card-desc">Quantization & sparsity from the hardware's point of view — and their limits</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-icon">🚀</div>
    <div class="card-title">Hardware → Usage</div>
    <div class="card-desc">Why GPUs won AI, multi-GPU training, KV-cache, and the industry's future</div>
  </div>
</div>
</div>

## 📋 Table of Contents

| # | Chapter | What You'll Learn |
|---|---------|-------------------|
| **Part I — Foundations: What Is a Chip?** | | |
| 1 | [What Is a Chip?](./01_what_is_a_chip.md) | Sand → silicon → transistors; the MOSFET, doping, Moore's Law, Dennard scaling |
| 2 | [Transistors to Logic](./02_transistors_to_logic.md) | Boolean algebra, gates, combinational vs sequential logic, flip-flops, the ALU |
| 3 | [Number Representation](./03_number_representation.md) | Binary, two's complement, fixed vs floating point, IEEE-754 — with bit-inspection code |
| 4 | [Anatomy of a Processor](./04_anatomy_of_a_processor.md) | von Neumann vs Harvard, fetch-decode-execute, datapath, registers, the clock, pipelining |
| 5 | [Instruction Sets](./05_instruction_sets.md) | ISA vs microarchitecture, RISC vs CISC, x86/ARM/RISC-V, SIMD & vector extensions |
| 6 | [Languages & Nomenclature](./06_languages_and_nomenclature.md) | Bare metal, firmware, microcode, assembly, Verilog/VHDL/RTL, HLS, CUDA, ROCm, Triton |
| 7 | [Types of Chips (Taxonomy)](./07_types_of_chips_taxonomy.md) | The full map: CPU/GPU/TPU/NPU/DSP/ISP/FPGA/ASIC/SoC/MCU in one comparison |
| **Part II — Parallelism, Memory & Movement** | | |
| 8 | [Parallel Processing](./08_parallel_processing.md) | ILP, superscalar, SIMD/SIMT, multicore, Flynn's taxonomy, Amdahl vs Gustafson |
| 9 | [Memory Types](./09_memory_types.md) | Registers, SRAM, DRAM, DDR/GDDR/HBM, NAND; the hierarchy, latency & bandwidth numbers |
| 10 | [Memory Management](./10_memory_management.md) | Virtual memory, paging, MMU/TLB, cache coherence (MESI), NUMA, consistency |
| 11 | [Memory Wall & Bandwidth](./11_memory_wall_and_bandwidth.md) | The roofline model, arithmetic intensity — *why bandwidth, not FLOPs, bounds ML* |
| 12 | [Buses & Interconnects](./12_buses_and_interconnects.md) | On-chip (NoC, AXI) vs off-chip (PCIe, CXL, NVLink, UALink); topologies & bandwidth |
| 13 | [Reliability & Fault Tolerance](./13_reliability_and_fault_tolerance.md) | ECC, RAS, soft errors, silent data corruption, redundancy — and long training runs |
| **Part III — The Chips Themselves** | | |
| 14 | [CPUs](./14_cpus.md) | Out-of-order superscalar cores, branch prediction, caches; when/why CPUs for ML |
| 15 | [GPUs](./15_gpus.md) | SIMT, SMs, warps, tensor cores, memory hierarchy — *why GPUs won ML* |
| 16 | [TPUs & Systolic Arrays](./16_tpus_and_systolic_arrays.md) | Systolic arrays, matrix-multiply units, dataflow; Google TPU v1→Trillium |
| 17 | [NPUs & Edge AI](./17_npus_and_edge_ai.md) | Mobile/edge NPUs, Apple Neural Engine, Qualcomm Hexagon, Renesas DRP-AI |
| 18 | [DSPs](./18_dsps.md) | MAC units, VLIW, signal-processing pipelines; where DSPs live |
| 19 | [ISPs](./19_isps.md) | Image signal processors, the camera pipeline, their place in SoCs |
| 20 | [FPGAs](./20_fpgas.md) | LUTs, reconfigurable fabric, when/why FPGAs, AI on FPGA |
| 21 | [ASICs](./21_asics.md) | The FPGA→ASIC spectrum, NRE/full-custom, the AI-ASIC boom |
| 22 | [SoCs](./22_socs.md) | Integration, chiplets, heterogeneous compute, Apple M-series, mobile SoCs |
| **Part IV — Datatypes (the keystone)** | | |
| 23 | [Number Formats Deep Dive](./23_number_formats_deep_dive.md) | FP32→FP4, INT8→INT1, bit layouts, microscaling (MXFP, NVFP4) — with code |
| 24 | [Datatypes & Silicon](./24_datatypes_and_silicon.md) | How a format reshapes the multiplier, tensor core, and per-generation HW support |
| **Part V — Optimization (HW ↔ model co-design)** | | |
| 25 | [Quantization (Hardware View)](./25_quantization_hardware_view.md) | How low precision saves area/power/bandwidth — *to what extent, and where it breaks* |
| 26 | [Sparsity (Hardware View)](./26_sparsity_hardware_view.md) | Structured 2:4 sparsity, sparse tensor cores — *to what extent it actually helps* |
| **Part VI — From Hardware to Usage** | | |
| 27 | [Why GPUs Won AI](./27_why_gpus_won_ai.md) | How GPU parallelism maps to neural-net math; matmul as the workhorse |
| 28 | [Multi-GPU & Distributed Training](./28_multi_gpu_and_distributed_training.md) | Data/tensor/pipeline/expert parallelism, FSDP, ZeRO, 3D — *why 70B won't fit on one GPU* |
| 29 | [Memory Hierarchy in Action](./29_memory_hierarchy_in_action.md) | How HW memory limits drive KV-cache, batching, activation memory, offloading |
| 30 | [Datatypes Drive Optimization](./30_datatypes_drive_optimization_choices.md) | A practitioner's decision framework: when INT8, when FP8, when 2:4 sparsity |
| **Part VII — How Chips Are Designed, Verified & Tested** | | |
| 31 | [Chip Design Flow](./31_chip_design_flow.md) | Spec → RTL → synthesis → place & route → timing → DRC/LVS → tapeout; the EDA stack |
| 32 | [Verification & Validation](./32_chip_verification_and_validation.md) | Simulation, UVM, assertions, formal, coverage, DFT, emulation, post-silicon — with examples |
| **Part VIII — The Industry: Players, Evolution & Future** | | |
| 33 | [Design-Paradigm Wars](./33_design_paradigm_wars.md) | NVIDIA vs AMD vs Intel vs ARM — design philosophies and how the lead changed hands |
| 34 | [Generational Improvements](./34_generational_improvements.md) | Process nodes, bandwidth growth, and tabulated family histories year over year |
| 35 | [Where the Industry Is Headed](./35_where_the_industry_is_headed.md) | Chiplets, 3D stacking, CXL pooling, co-packaged optics, analog/in-memory, the post-Moore era |
| **Appendices** | | |
| A | [How Chips Are Made](./appendix_a_how_chips_are_made.md) | Sand → wafer → fab → EUV lithography → etch/doping → packaging → test; yield & binning |
| B | [Foundry Ecosystem](./appendix_b_foundry_ecosystem.md) | TSMC/Samsung/Intel, the fabless model, EDA tools, IP cores, the supply chain |
| C | [Photonics](./appendix_c_photonics.md) | Silicon photonics, co-packaged optics, optical interconnect & compute |
| D | [Power & Thermals](./appendix_d_power_and_thermals.md) | Power delivery, DVFS, cooling, performance-per-watt, the energy cost of AI |
| E | [Packaging & Integration](./appendix_e_packaging.md) | 2.5D/3D, CoWoS, interposers, HBM stacking, chiplets/UCIe |
| F | [Glossary](./appendix_f_glossary.md) | Every acronym and term in one place |
| G | [Resources](./appendix_g_resources.md) | Books, courses, channels, blogs, papers, podcasts for going deeper |
| H | [The Life of a Running Computer](./appendix_h_how_a_computer_runs.md) | What happens at power-on, how interrupts & the scheduler work, what "idle" (HLT/C-states) really is, and who keeps the screen lit |

## 🗺️ Learning Path

<div class="diagram">
<div class="diagram-title">Recommended Learning Path</div>
<div class="flow">
  <div class="flow-node accent wide">🧱 Ch 1–7: Foundations — transistors to ISAs to the chip taxonomy</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">🧠 Ch 8–13: Parallelism, memory, and how data moves</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">⚙️ Ch 14–22: The chips themselves, one architecture at a time</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">🔢 Ch 23–24: Datatypes — the keystone of optimization</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">🗜️ Ch 25–26: Quantization & sparsity from the silicon's view</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">🚀 Ch 27–30: Hardware → usage (why GPUs won, multi-GPU, KV-cache, your choices)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node teal wide">🏭 Ch 31–35 + Appendices: Design, verification, the giants, the future, fabrication</div>
</div>
</div>

## ⚡ Quick Start Paths

### Path A: "I optimize models and want to know *why* my choices work" (6 chapters)

1. [11 — Memory Wall & Bandwidth](./11_memory_wall_and_bandwidth.md) — the roofline; why you're memory-bound
2. [23 — Number Formats Deep Dive](./23_number_formats_deep_dive.md) — what FP8/INT4 really are
3. [25 — Quantization (Hardware View)](./25_quantization_hardware_view.md) — how much it actually buys you
4. [26 — Sparsity (Hardware View)](./26_sparsity_hardware_view.md) — when 2:4 helps
5. [29 — Memory Hierarchy in Action](./29_memory_hierarchy_in_action.md) — KV-cache & batching
6. [30 — Datatypes Drive Optimization](./30_datatypes_drive_optimization_choices.md) — the decision framework

### Path B: "I want to understand the hardware end-to-end" (full guide)

Read Chapters 1 → 35 in order, then the appendices. Each builds on the previous.

### Path C: "I'm scaling training to many GPUs" (5 chapters)

1. [09 — Memory Types](./09_memory_types.md) — HBM and the memory hierarchy
2. [12 — Buses & Interconnects](./12_buses_and_interconnects.md) — NVLink, PCIe, the fabric
3. [15 — GPUs](./15_gpus.md) — what's inside the accelerator
4. [27 — Why GPUs Won AI](./27_why_gpus_won_ai.md) — the hardware/math fit
5. [28 — Multi-GPU & Distributed Training](./28_multi_gpu_and_distributed_training.md) — DP/TP/PP/FSDP/ZeRO

### Path D: "I'm curious how chips are actually built" (4 reads)

1. [31 — Chip Design Flow](./31_chip_design_flow.md) — RTL to GDSII
2. [32 — Verification & Validation](./32_chip_verification_and_validation.md) — how correctness is proven
3. [Appendix A — How Chips Are Made](./appendix_a_how_chips_are_made.md) — the physical fab
4. [Appendix B — Foundry Ecosystem](./appendix_b_foundry_ecosystem.md) — the industry behind it

## 📚 Prerequisites

You'll get the most out of this guide if you're comfortable with:

- **Python & PyTorch** — tensors, dtypes, shapes, basic `nn.Module` usage
- **Basic ML** — what training, inference, and a matrix multiply are
- **A little binary** — bits and bytes (Chapter 3 reteaches everything else)
- **Curiosity about hardware** — no electrical-engineering background required

Everything else — semiconductor physics, logic design, computer architecture, HDLs — is taught from scratch.

## 🏗️ How This Guide Is Built

Every chapter follows a **What → Why → Which → When → Where** rhythm, and shares a consistent toolkit:

<div class="diagram">
<div class="diagram-title">Chapter Toolkit</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">First Principles</div>
    <div class="card-desc">Concepts built up from the transistor, never assumed</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Tables & Diagrams</div>
    <div class="card-desc">Comparison tables and CSS diagrams for every key idea</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Math Where It Helps</div>
    <div class="card-desc">Roofline, bandwidth, area/power — derived, not hand-waved</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Python/PyTorch Code</div>
    <div class="card-desc">Runnable snippets; low-level languages where they're the topic</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Dtypes & Shapes</div>
    <div class="card-desc">Bit layouts, byte sizes, tensor shapes, bandwidth numbers, always annotated</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">ML Linkage</div>
    <div class="card-desc">Every hardware idea tied back to a real model-optimization decision</div>
  </div>
</div>
</div>

## 📝 Changelog

| Date | Changes |
|------|---------|
| May 2026 | Initial release — 35 chapters + 7 appendices |

---

*Last updated: May 2026 · Built with Jekyll, deployed on GitHub Pages.*
