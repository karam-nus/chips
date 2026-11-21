---
title: "Home"
permalink: /
---

# 🔬 Chips — The Complete Guide

> **From Sand to Supercomputers**: a first-principles guide to chip design and architecture, written for the **AI/ML practitioner who optimizes models**. By the end you will understand *the hardware reasons behind every quantization, sparsity, batching, and parallelism decision you make.*

<div class="stats">
  <div class="stat"><div class="stat-num"><em>35</em></div><div class="stat-label">chapters, transistor to rack</div></div>
  <div class="stat"><div class="stat-num"><em>8</em></div><div class="stat-label">appendices: fab, foundry, optics, power, packaging</div></div>
  <div class="stat"><div class="stat-num"><em>8</em></div><div class="stat-label">interactive labs, tested in Node</div></div>
  <div class="stat"><div class="stat-num"><em>26</em></div><div class="stat-label">accelerators in the atlas, dense rates, sourced</div></div>
  <div class="stat"><div class="stat-num"><em>198</em></div><div class="stat-label">Check Yourself questions with worked answers</div></div>
</div>

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
| 16 | [TPUs & Systolic Arrays](./16_tpus_and_systolic_arrays.md) | Systolic arrays, matrix-multiply units, dataflow; Google TPU v1 → Ironwood → TPU 8t/8i |
| 17 | [NPUs & Edge AI](./17_npus_and_edge_ai.md) | Mobile/edge NPUs, Apple Neural Engine, Qualcomm Hexagon, Renesas DRP-AI |
| 18 | [DSPs](./18_dsps.md) | MAC units, VLIW, signal-processing pipelines; where DSPs live |
| 19 | [ISPs](./19_isps.md) | Image signal processors, the camera pipeline, their place in SoCs |
| 20 | [FPGAs](./20_fpgas.md) | LUTs, reconfigurable fabric, when/why FPGAs, AI on FPGA |
| 21 | [ASICs](./21_asics.md) | The FPGA→ASIC spectrum, NRE/full-custom, the AI-ASIC boom (Trainium, Maia, MTIA, Groq, Cerebras) |
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
| 35 | [Where the Industry Is Headed](./35_where_the_industry_is_headed.md) | HBM4, chiplets, CXL tiers, co-packaged optics, in-memory compute, power as the binding constraint |
| **Interactive** | | |
| ⚗ | [Interactive Labs](./labs.md) | Roofline, bit explorer, will-it-fit, systolic array, wafer yield, 2:4 sparsity, quantization error |
| 🗺 | [Accelerator Atlas](./atlas.md) | 26 accelerators on one bandwidth-vs-compute chart, sortable, with sources |
| **Appendices** | | |
| A | [How Chips Are Made](./appendix_a_how_chips_are_made.md) | Sand → wafer → fab → EUV lithography → etch/doping → packaging → test; yield & binning |
| B | [Foundry Ecosystem](./appendix_b_foundry_ecosystem.md) | TSMC/Samsung/Intel, the fabless model, EDA tools, IP cores, the supply chain |
| C | [Photonics](./appendix_c_photonics.md) | Silicon photonics, co-packaged optics, optical interconnect & compute |
| D | [Power & Thermals](./appendix_d_power_and_thermals.md) | Power delivery, DVFS, cooling, performance-per-watt, the energy cost of AI |
| E | [Packaging & Integration](./appendix_e_packaging.md) | 2.5D/3D, CoWoS, interposers, HBM stacking, chiplets/UCIe |
| F | [Glossary](./appendix_f_glossary.md) | Every acronym and term in one place |
| G | [Resources](./appendix_g_resources.md) | Books, courses, channels, blogs, papers and tools for going deeper |
| H | [The Life of a Running Computer](./appendix_h_how_a_computer_runs.md) | What happens at power-on, how interrupts & the scheduler work, what "idle" (HLT/C-states) really is, and who keeps the screen lit |

## 🗺️ Learning Path

Read the board left to right, then follow the trace down and back. Every chip is a link to the first chapter of that part.

<div class="pcb"><svg viewBox="0 0 900 372" role="img" aria-label="Learning path drawn as a circuit board: eight parts of the guide connected in reading order, plus appendices and interactive labs">
<text class="silk big" x="34" y="34">CHIPS GUIDE · LEARNING PATH</text><text class="silk dim" x="34" y="50">REV 2026.11 · FOLLOW THE TRACE · CLICK A CHIP</text>
<text class="silk dim" x="866" y="34" text-anchor="end">8 PARTS · 35 CH · 8 APPX</text>
<path class="trace" d="M212,112 L244,112"/>
<path class="trace" d="M424,112 L456,112"/>
<path class="trace" d="M636,112 L668,112"/>
<path class="trace" d="M848,112 L876,112 L876,242 L848,242"/>
<path class="trace" d="M668,242 L636,242"/>
<path class="trace" d="M456,242 L424,242"/>
<path class="trace" d="M244,242 L212,242"/>
<path class="trace thin" d="M122,274 L122,320"/>
<path class="trace thin" d="M758,274 L758,320"/>
<a href="./appendix_a_how_chips_are_made.html"><g><rect class="ic-body" x="7" y="320" width="250" height="30" rx="3"/>
<circle class="pad" cx="25" cy="343" r="3.2"/>
<circle class="pad" cx="56" cy="343" r="3.2"/>
<circle class="pad" cx="86" cy="343" r="3.2"/>
<circle class="pad" cx="117" cy="343" r="3.2"/>
<circle class="pad" cx="147" cy="343" r="3.2"/>
<circle class="pad" cx="178" cy="343" r="3.2"/>
<circle class="pad" cx="208" cy="343" r="3.2"/>
<circle class="pad" cx="239" cy="343" r="3.2"/>
<text class="ic-label" x="132" y="333" text-anchor="middle">APPENDIX A–H</text>
<text class="silk dim" x="7" y="315">J1 · fab, foundry, optics, power, packaging…</text></g></a>
<a href="./labs.html"><g><rect class="ic-body" x="623" y="320" width="250" height="30" rx="3"/>
<circle class="pad" cx="641" cy="343" r="3.2"/>
<circle class="pad" cx="672" cy="343" r="3.2"/>
<circle class="pad" cx="702" cy="343" r="3.2"/>
<circle class="pad" cx="733" cy="343" r="3.2"/>
<circle class="pad" cx="763" cy="343" r="3.2"/>
<circle class="pad" cx="794" cy="343" r="3.2"/>
<circle class="pad" cx="824" cy="343" r="3.2"/>
<circle class="pad" cx="855" cy="343" r="3.2"/>
<text class="ic-label" x="748" y="333" text-anchor="middle">LABS + ATLAS</text>
<text class="silk dim" x="623" y="315">J2 · 8 interactive labs, 26 accelerators</text></g></a>
<circle class="via" cx="876" cy="112" r="5"/>
<circle class="via" cx="876" cy="242" r="5"/>
<a href="./01_what_is_a_chip.html"><g>
<rect class="ic-body" x="32" y="80" width="180" height="64" rx="4"/>
<rect class="ic-pin" x="43.5" y="74" width="5" height="6"/><rect class="ic-pin" x="43.5" y="144" width="5" height="6"/>
<rect class="ic-pin" x="65.2" y="74" width="5" height="6"/><rect class="ic-pin" x="65.2" y="144" width="5" height="6"/>
<rect class="ic-pin" x="86.9" y="74" width="5" height="6"/><rect class="ic-pin" x="86.9" y="144" width="5" height="6"/>
<rect class="ic-pin" x="108.6" y="74" width="5" height="6"/><rect class="ic-pin" x="108.6" y="144" width="5" height="6"/>
<rect class="ic-pin" x="130.4" y="74" width="5" height="6"/><rect class="ic-pin" x="130.4" y="144" width="5" height="6"/>
<rect class="ic-pin" x="152.1" y="74" width="5" height="6"/><rect class="ic-pin" x="152.1" y="144" width="5" height="6"/>
<rect class="ic-pin" x="173.8" y="74" width="5" height="6"/><rect class="ic-pin" x="173.8" y="144" width="5" height="6"/>
<rect class="ic-pin" x="195.5" y="74" width="5" height="6"/><rect class="ic-pin" x="195.5" y="144" width="5" height="6"/>
<circle class="pad" cx="42" cy="90" r="2.6"/>
<text class="ic-label" x="122" y="109" text-anchor="middle">PART I · FOUNDATIONS</text>
<text class="ic-sub" x="122" y="126" text-anchor="middle">Ch 1–7 · transistor → ISA</text>
<text class="silk" x="32" y="70">U1</text>
</g></a>
<a href="./08_parallel_processing.html"><g>
<rect class="ic-body" x="244" y="80" width="180" height="64" rx="4"/>
<rect class="ic-pin" x="255.5" y="74" width="5" height="6"/><rect class="ic-pin" x="255.5" y="144" width="5" height="6"/>
<rect class="ic-pin" x="277.2" y="74" width="5" height="6"/><rect class="ic-pin" x="277.2" y="144" width="5" height="6"/>
<rect class="ic-pin" x="298.9" y="74" width="5" height="6"/><rect class="ic-pin" x="298.9" y="144" width="5" height="6"/>
<rect class="ic-pin" x="320.6" y="74" width="5" height="6"/><rect class="ic-pin" x="320.6" y="144" width="5" height="6"/>
<rect class="ic-pin" x="342.4" y="74" width="5" height="6"/><rect class="ic-pin" x="342.4" y="144" width="5" height="6"/>
<rect class="ic-pin" x="364.1" y="74" width="5" height="6"/><rect class="ic-pin" x="364.1" y="144" width="5" height="6"/>
<rect class="ic-pin" x="385.8" y="74" width="5" height="6"/><rect class="ic-pin" x="385.8" y="144" width="5" height="6"/>
<rect class="ic-pin" x="407.5" y="74" width="5" height="6"/><rect class="ic-pin" x="407.5" y="144" width="5" height="6"/>
<circle class="pad" cx="254" cy="90" r="2.6"/>
<text class="ic-label" x="334" y="109" text-anchor="middle">PART II · MEMORY</text>
<text class="ic-sub" x="334" y="126" text-anchor="middle">Ch 8–13 · memory, roofline</text>
<text class="silk" x="244" y="70">U2</text>
</g></a>
<a href="./14_cpus.html"><g>
<rect class="ic-body" x="456" y="80" width="180" height="64" rx="4"/>
<rect class="ic-pin" x="467.5" y="74" width="5" height="6"/><rect class="ic-pin" x="467.5" y="144" width="5" height="6"/>
<rect class="ic-pin" x="489.2" y="74" width="5" height="6"/><rect class="ic-pin" x="489.2" y="144" width="5" height="6"/>
<rect class="ic-pin" x="510.9" y="74" width="5" height="6"/><rect class="ic-pin" x="510.9" y="144" width="5" height="6"/>
<rect class="ic-pin" x="532.6" y="74" width="5" height="6"/><rect class="ic-pin" x="532.6" y="144" width="5" height="6"/>
<rect class="ic-pin" x="554.4" y="74" width="5" height="6"/><rect class="ic-pin" x="554.4" y="144" width="5" height="6"/>
<rect class="ic-pin" x="576.1" y="74" width="5" height="6"/><rect class="ic-pin" x="576.1" y="144" width="5" height="6"/>
<rect class="ic-pin" x="597.8" y="74" width="5" height="6"/><rect class="ic-pin" x="597.8" y="144" width="5" height="6"/>
<rect class="ic-pin" x="619.5" y="74" width="5" height="6"/><rect class="ic-pin" x="619.5" y="144" width="5" height="6"/>
<circle class="pad" cx="466" cy="90" r="2.6"/>
<text class="ic-label" x="546" y="109" text-anchor="middle">PART III · THE CHIPS</text>
<text class="ic-sub" x="546" y="126" text-anchor="middle">Ch 14–22 · CPU, GPU, TPU, NPU…</text>
<text class="silk" x="456" y="70">U3</text>
</g></a>
<a href="./23_number_formats_deep_dive.html"><g>
<rect class="ic-body" x="668" y="80" width="180" height="64" rx="4"/>
<rect class="ic-pin" x="679.5" y="74" width="5" height="6"/><rect class="ic-pin" x="679.5" y="144" width="5" height="6"/>
<rect class="ic-pin" x="701.2" y="74" width="5" height="6"/><rect class="ic-pin" x="701.2" y="144" width="5" height="6"/>
<rect class="ic-pin" x="722.9" y="74" width="5" height="6"/><rect class="ic-pin" x="722.9" y="144" width="5" height="6"/>
<rect class="ic-pin" x="744.6" y="74" width="5" height="6"/><rect class="ic-pin" x="744.6" y="144" width="5" height="6"/>
<rect class="ic-pin" x="766.4" y="74" width="5" height="6"/><rect class="ic-pin" x="766.4" y="144" width="5" height="6"/>
<rect class="ic-pin" x="788.1" y="74" width="5" height="6"/><rect class="ic-pin" x="788.1" y="144" width="5" height="6"/>
<rect class="ic-pin" x="809.8" y="74" width="5" height="6"/><rect class="ic-pin" x="809.8" y="144" width="5" height="6"/>
<rect class="ic-pin" x="831.5" y="74" width="5" height="6"/><rect class="ic-pin" x="831.5" y="144" width="5" height="6"/>
<circle class="pad" cx="678" cy="90" r="2.6"/>
<text class="ic-label" x="758" y="109" text-anchor="middle">PART IV · DATATYPES</text>
<text class="ic-sub" x="758" y="126" text-anchor="middle">Ch 23–24 · FP32 → FP4</text>
<text class="silk" x="668" y="70">U4</text>
</g></a>
<a href="./25_quantization_hardware_view.html"><g>
<rect class="ic-body" x="668" y="210" width="180" height="64" rx="4"/>
<rect class="ic-pin" x="679.5" y="204" width="5" height="6"/><rect class="ic-pin" x="679.5" y="274" width="5" height="6"/>
<rect class="ic-pin" x="701.2" y="204" width="5" height="6"/><rect class="ic-pin" x="701.2" y="274" width="5" height="6"/>
<rect class="ic-pin" x="722.9" y="204" width="5" height="6"/><rect class="ic-pin" x="722.9" y="274" width="5" height="6"/>
<rect class="ic-pin" x="744.6" y="204" width="5" height="6"/><rect class="ic-pin" x="744.6" y="274" width="5" height="6"/>
<rect class="ic-pin" x="766.4" y="204" width="5" height="6"/><rect class="ic-pin" x="766.4" y="274" width="5" height="6"/>
<rect class="ic-pin" x="788.1" y="204" width="5" height="6"/><rect class="ic-pin" x="788.1" y="274" width="5" height="6"/>
<rect class="ic-pin" x="809.8" y="204" width="5" height="6"/><rect class="ic-pin" x="809.8" y="274" width="5" height="6"/>
<rect class="ic-pin" x="831.5" y="204" width="5" height="6"/><rect class="ic-pin" x="831.5" y="274" width="5" height="6"/>
<circle class="pad" cx="678" cy="220" r="2.6"/>
<text class="ic-label" x="758" y="239" text-anchor="middle">PART V · OPTIMIZATION</text>
<text class="ic-sub" x="758" y="256" text-anchor="middle">Ch 25–26 · quantize, sparsify</text>
<text class="silk" x="668" y="200">U5</text>
</g></a>
<a href="./27_why_gpus_won_ai.html"><g>
<rect class="ic-body" x="456" y="210" width="180" height="64" rx="4"/>
<rect class="ic-pin" x="467.5" y="204" width="5" height="6"/><rect class="ic-pin" x="467.5" y="274" width="5" height="6"/>
<rect class="ic-pin" x="489.2" y="204" width="5" height="6"/><rect class="ic-pin" x="489.2" y="274" width="5" height="6"/>
<rect class="ic-pin" x="510.9" y="204" width="5" height="6"/><rect class="ic-pin" x="510.9" y="274" width="5" height="6"/>
<rect class="ic-pin" x="532.6" y="204" width="5" height="6"/><rect class="ic-pin" x="532.6" y="274" width="5" height="6"/>
<rect class="ic-pin" x="554.4" y="204" width="5" height="6"/><rect class="ic-pin" x="554.4" y="274" width="5" height="6"/>
<rect class="ic-pin" x="576.1" y="204" width="5" height="6"/><rect class="ic-pin" x="576.1" y="274" width="5" height="6"/>
<rect class="ic-pin" x="597.8" y="204" width="5" height="6"/><rect class="ic-pin" x="597.8" y="274" width="5" height="6"/>
<rect class="ic-pin" x="619.5" y="204" width="5" height="6"/><rect class="ic-pin" x="619.5" y="274" width="5" height="6"/>
<circle class="pad" cx="466" cy="220" r="2.6"/>
<text class="ic-label" x="546" y="239" text-anchor="middle">PART VI · USAGE</text>
<text class="ic-sub" x="546" y="256" text-anchor="middle">Ch 27–30 · GPUs, clusters, KV</text>
<text class="silk" x="456" y="200">U6</text>
</g></a>
<a href="./31_chip_design_flow.html"><g>
<rect class="ic-body" x="244" y="210" width="180" height="64" rx="4"/>
<rect class="ic-pin" x="255.5" y="204" width="5" height="6"/><rect class="ic-pin" x="255.5" y="274" width="5" height="6"/>
<rect class="ic-pin" x="277.2" y="204" width="5" height="6"/><rect class="ic-pin" x="277.2" y="274" width="5" height="6"/>
<rect class="ic-pin" x="298.9" y="204" width="5" height="6"/><rect class="ic-pin" x="298.9" y="274" width="5" height="6"/>
<rect class="ic-pin" x="320.6" y="204" width="5" height="6"/><rect class="ic-pin" x="320.6" y="274" width="5" height="6"/>
<rect class="ic-pin" x="342.4" y="204" width="5" height="6"/><rect class="ic-pin" x="342.4" y="274" width="5" height="6"/>
<rect class="ic-pin" x="364.1" y="204" width="5" height="6"/><rect class="ic-pin" x="364.1" y="274" width="5" height="6"/>
<rect class="ic-pin" x="385.8" y="204" width="5" height="6"/><rect class="ic-pin" x="385.8" y="274" width="5" height="6"/>
<rect class="ic-pin" x="407.5" y="204" width="5" height="6"/><rect class="ic-pin" x="407.5" y="274" width="5" height="6"/>
<circle class="pad" cx="254" cy="220" r="2.6"/>
<text class="ic-label" x="334" y="239" text-anchor="middle">PART VII · DESIGN</text>
<text class="ic-sub" x="334" y="256" text-anchor="middle">Ch 31–32 · RTL → GDSII, verify</text>
<text class="silk" x="244" y="200">U7</text>
</g></a>
<a href="./33_design_paradigm_wars.html"><g>
<rect class="ic-body" x="32" y="210" width="180" height="64" rx="4"/>
<rect class="ic-pin" x="43.5" y="204" width="5" height="6"/><rect class="ic-pin" x="43.5" y="274" width="5" height="6"/>
<rect class="ic-pin" x="65.2" y="204" width="5" height="6"/><rect class="ic-pin" x="65.2" y="274" width="5" height="6"/>
<rect class="ic-pin" x="86.9" y="204" width="5" height="6"/><rect class="ic-pin" x="86.9" y="274" width="5" height="6"/>
<rect class="ic-pin" x="108.6" y="204" width="5" height="6"/><rect class="ic-pin" x="108.6" y="274" width="5" height="6"/>
<rect class="ic-pin" x="130.4" y="204" width="5" height="6"/><rect class="ic-pin" x="130.4" y="274" width="5" height="6"/>
<rect class="ic-pin" x="152.1" y="204" width="5" height="6"/><rect class="ic-pin" x="152.1" y="274" width="5" height="6"/>
<rect class="ic-pin" x="173.8" y="204" width="5" height="6"/><rect class="ic-pin" x="173.8" y="274" width="5" height="6"/>
<rect class="ic-pin" x="195.5" y="204" width="5" height="6"/><rect class="ic-pin" x="195.5" y="274" width="5" height="6"/>
<circle class="pad" cx="42" cy="220" r="2.6"/>
<text class="ic-label" x="122" y="239" text-anchor="middle">PART VIII · INDUSTRY</text>
<text class="ic-sub" x="122" y="256" text-anchor="middle">Ch 33–35 · players, future</text>
<text class="silk" x="32" y="200">U8</text>
</g></a>
</svg></div>

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

### Path D: "I learn by poking at things" (labs first)

1. [Roofline lab](./labs.md#1-roofline) — move a workload until it crosses the ridge, then read [Chapter 11](./11_memory_wall_and_bandwidth.md)
2. [Bit explorer](./labs.md#2-bit-explorer) — flip bits in FP8 and FP4, then read [Chapter 23](./23_number_formats_deep_dive.md)
3. [Will it fit?](./labs.md#3-will-it-fit) — size a 70B model's KV cache, then read [Chapter 29](./29_memory_hierarchy_in_action.md)
4. [Accelerator Atlas](./atlas.md) — compare every chip in the guide, then read [Chapter 34](./34_generational_improvements.md)

### Path E: "I'm curious how chips are actually built" (4 reads)

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
    <div class="card-title">Visuals That Carry Weight</div>
    <div class="card-desc">Datasheets, die floorplans, cross-sections, receipts, spec bars and clickable SVG flowcharts</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Math Where It Helps</div>
    <div class="card-desc">Roofline, bandwidth, yield, power — derived, not hand-waved</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Code That Runs</div>
    <div class="card-desc">Python snippets whose printed outputs were re-run and checked; low-level languages where they're the topic</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Labs</div>
    <div class="card-desc">Eight in-page instruments built on pure functions with a Node test suite</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Check Yourself + References</div>
    <div class="card-desc">Every chapter ends with worked questions and a sourced reference table</div>
  </div>
</div>
</div>

### Keeping it honest

- **Dense numbers only.** Throughput figures exclude the 2:4 sparsity bonus vendors often fold into headlines; every chip in the [atlas](./atlas.md) links to its primary source.
- **One data file.** The atlas, the roofline and fit labs, and the memory-wall figure all read [`assets/data/accelerators.json`](https://github.com/karam-nus/chips/blob/main/assets/data/accelerators.json).
- **Generated, checked artefacts.** `python3 tools/build_flowcharts.py --check`, `python3 tools/make_figures.py --check` and `node tools/test_labs.js` keep the flowcharts, figures and labs in sync with their sources. Authoring conventions are in [PROMPT.md](https://github.com/karam-nus/chips/blob/main/PROMPT.md).

## 📝 Changelog

| Date | Changes |
|------|---------|
| Nov 2026 | Full refresh: every chapter fact-checked against 2025–26 hardware (Blackwell Ultra, Rubin, MI355X/MI455X, Ironwood/TPU 8, Trainium3, HBM4); dense-vs-sparse figures corrected throughout; code outputs re-run; 8 interactive labs, the Accelerator Atlas, 8 SVG flowcharts, generated figures; Check Yourself and References in every chapter; new theme with Part navigation, reading progress and chapter search |
| May 2026 | Initial release — 35 chapters + 8 appendices |

---

## 🧭 More Guides in This Family

Same theme, same first-principles approach: [Quantization](https://github.com/karam-nus/quantization) · [Model Optimization](https://github.com/karam-nus/model-optimization) · [Vectorization](https://github.com/karam-nus/vectorization) · [Inference](https://github.com/karam-nus/inference) · [Training](https://github.com/karam-nus/training) · [Networking](https://github.com/karam-nus/networking) · [NVIDIA](https://github.com/karam-nus/nvidia) · [Programming](https://github.com/karam-nus/programming) — all collected at [karam-nus.github.io](https://karam-nus.github.io/).

*Last updated: November 2026 · Built with Jekyll, deployed on GitHub Pages.*
