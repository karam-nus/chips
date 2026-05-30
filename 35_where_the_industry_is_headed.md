---
title: "Chapter 35 — Where the Industry Is Headed"
---

[← Back to Table of Contents](./README.md)

# Chapter 35 — Where the Industry Is Headed

The previous two chapters told the story of who the players are and how performance has grown. This chapter asks the harder question: **what happens next, and what does it mean for how you build, optimize, and deploy models?** The answer requires confronting a structural constraint that runs through every trend in this guide: we are running out of the free performance gains that defined the first five decades of computing, and the responses to that shortage — chiplets, 3D stacking, new memory fabrics, co-packaged optics, analog compute, and architectural specialization — will reshape the hardware landscape more fundamentally than any individual GPU generation.

The unifying insight of this chapter is one sentence that appears in multiple forms throughout the guide: **the binding constraint is data movement, not arithmetic.** Moving bits through a chip, through a package, across a board, between nodes, or off and onto memory costs energy orders of magnitude greater than performing an arithmetic operation on those bits. Every frontier technology described here is, at root, an attempt to either eliminate data movement, shorten it, or make it cheaper per bit.

> **The one-sentence version:** Post-Moore hardware progress comes from smarter integration (chiplets, 3D stacking, CXL disaggregation), cheaper data movement (co-packaged optics, in-memory compute), and specialization — not from transistors getting faster, and the ML practitioner who understands this will make better infrastructure and model-design choices.

---

## The End of Easy Scaling

### What "Easy Scaling" Was

From 1965 to ~2005, the semiconductor industry operated under a remarkable combination of Moore's Law and Dennard scaling: every two years, you got more transistors at lower energy per operation, and clock frequencies climbed automatically. A programmer who did nothing — wrote no new code, made no architectural changes — woke up to a faster computer every two years for free. This is "easy scaling."

Two constraints killed it:
1. **Dennard scaling collapsed** (~2005): voltage can't drop further without leakage dominance; power density rises; clocks stalled at 3–5 GHz ([Chapter 1](./01_what_is_a_chip.md)).
2. **Moore's Law is slowing** (~2018–present): transistor density still grows, but at 2nm and below, lithography costs ($200M+ per EUV mask set) and quantum tunneling effects constrain density improvements. Node names are now marketing labels; the real density gain per node generation is ~1.1–1.3× rather than the historical 1.7–2×.

<div class="diagram">
<div class="diagram-title">From Easy Scaling to Hard-Won Gains</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Easy Scaling Era (pre-2005)</div>
    <ul>
      <li>Shrink node → more transistors + lower V<sub>dd</sub></li>
      <li>Clock frequency rises automatically</li>
      <li>Power density stays flat (Dennard)</li>
      <li>Application perf improves ~2× every 2 years without software changes</li>
      <li>Design principle: "the next node will fix it"</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Hard-Won Era (post-2005, accelerating post-2018)</div>
    <ul>
      <li>Node shrinks give ~1.2–1.5× density, not 2×</li>
      <li>No free frequency gains; TDP rises faster than perf</li>
      <li>Gains require architectural specialization (tensor cores)</li>
      <li>Packaging and integration replace process as key differentiator</li>
      <li>Design principle: "architect around data movement costs"</li>
    </ul>
  </div>
</div>
</div>

The industry's response to the end of easy scaling is a portfolio of approaches — no single silver bullet — and they operate on different timescales, readiness levels, and workload fit.

---

## Chiplets and Heterogeneous Integration

### The Chiplet Model

A **chiplet** is a small die designed to be integrated with other dies into a single package, rather than combined with all functionality on one monolithic die ([Chapter 22](./22_socs.md)). AMD pioneered the approach in CPUs (Zen 2, 2019); NVIDIA moved to a 2-die package with Blackwell (2024); Intel's Ponte Vecchio and Meteor Lake are chiplet-based.

The structural advantages are well-established:
- **Yield**: smaller dies have fewer defects; doubling transistors on one die reduces yield by more than half
- **Disaggregated process optimization**: compute dies on N3, I/O on N6, analog on N16 — use the expensive leading-edge node only where it matters
- **Reuse**: the same CCD (compute chiplet) appears in 6-core desktop, 16-core laptop, and 128-core server EPYC

The missing piece for heterogeneous chiplet integration has been a **standard die-to-die interface**. AMD's Infinity Fabric and NVIDIA's NVLink-C2C are proprietary. **UCIe** (Universal Chiplet Interconnect Express), announced 2022 with backing from Intel, AMD, ARM, Qualcomm, Samsung, TSMC, and others, defines a standard physical and protocol layer for die-to-die communication:

| UCIe Standard | Bandwidth Density | Distance | Notes |
|--------------|:----------------:|:--------:|-------|
| UCIe 1.0 (Advanced packaging) | 28 GB/s/mm | <2 mm | Die-to-die on interposer / bridge |
| UCIe 1.0 (Standard packaging) | 2 GB/s/mm | <25 mm | PCB-level, lower density |
| UCIe 2.0 (est. 2025+) | ~112 GB/s/mm | <5 mm | Next-generation; targets HBM-adjacent bandwidth |

UCIe standardization enables a **chiplet marketplace**: a company designs a memory controller chiplet, a PCIe chiplet, or an AI accelerator chiplet, and it can be combined with any compatible compute die. This is happening now — Marvell, Broadcom, and AMD have announced UCIe-compatible chiplets for custom AI accelerator programs.

---

## Advanced Packaging: 2.5D, 3D, and CoWoS

### Why Packaging Now Matters as Much as Process

For the first 40 years of the IC era, **packaging** was an afterthought — a mechanical enclosure to protect the die and fan out its connections to a PCB. That has changed fundamentally. Advanced packaging is now a primary performance differentiator, because it is the only way to get HBM memory physically adjacent to GPU dies with the short, wide interconnects that deliver terabytes/second of bandwidth.

See [Appendix E — Packaging & Integration](./appendix_e_packaging.md) for deep detail. The key approaches:

<div class="diagram">
<div class="diagram-title">Advanced Packaging Approaches — 2026 Landscape</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">CoWoS (Chip-on-Wafer-on-Substrate)</div>
    <div class="card-desc">TSMC's 2.5D interposer: GPU die + HBM stacks side-by-side on silicon interposer. Enables 5–10 TB/s HBM bandwidth. Used in H100, A100, MI300X. CoWoS-S, CoWoS-R expanding area.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Hybrid Bonding / Cu-Cu Direct Bond</div>
    <div class="card-desc">Bond two dies face-to-face with copper-to-copper direct connections at <10 µm pitch. 100–1,000× higher connection density than flip-chip bumps. Enables 3D die stacking with near-on-chip bandwidth.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">3D Stacking (3D-IC)</div>
    <div class="card-desc">Dies stacked vertically with through-silicon vias (TSVs) or hybrid bonds. Puts memory directly atop the compute die. Eliminates long horizontal wires. Used in HBM stacks; now extending to logic-on-logic.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">FOVEROS / Embedded Multi-Die</div>
    <div class="card-desc">Intel's 3D stacking for logic dies. Tiles face-up on a base die connected via hybrid bonding. Enables disaggregated CPU tiles with shared power/IO base. Used in Meteor Lake, Lunar Lake.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">SoIC (System on Integrated Chips)</div>
    <div class="card-desc">TSMC's 3D stacking using hybrid bond. SoIC-XS targets < 9 µm bonding pitch. Logic stacked on logic at near-BEOL interconnect density. Prototype GPUs demoed with stacked SRAM on compute die.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Wafer-Level Fan-Out (FOWLP)</div>
    <div class="card-desc">Dies embedded in reconstituted wafer, connections routed through packaging substrate. Used for mobile SoCs (TSMC InFO). High volume, lower cost, moderate bandwidth density.</div>
  </div>
</div>
</div>

The near-term trajectory (2025–2028): CoWoS packages will grow from ~1,200 mm² (H100) to **>2,500 mm²** (multi-reticle interposers) to accommodate more HBM stacks and larger multi-die compute complexes. TSMC's "System on Wafer" (SoW) concepts push this further toward wafer-scale integration for the highest-end systems.

---

## HBM4 and Memory Beyond

HBM3e (deployed in H200, MI325X, B200) is approaching the bandwidth-density ceiling of the current physical interface. **HBM4**, expected in volume production 2025–2026, represents the most significant HBM architectural change since HBM2:

| | HBM3e | HBM4 |
|--|-------|------|
| Bus width per stack | 1,024 bits | **2,048 bits** |
| Data rate | 9.6 Gbps/pin | 12–16 Gbps/pin |
| BW per stack | ~1.2 TB/s | **~3.2–4.0 TB/s** |
| Stacks per GPU (target) | 8 | 8–12 |
| Total BW target | ~8–10 TB/s | **~30–40 TB/s** |
| Capacity per stack | 24–36 GB | 32–64 GB |

HBM4's doubled bus width requires changes to both the DRAM stack and the logic die interface — it is a co-designed specification, not just a DRAM improvement. NVIDIA's Rubin (2026) and AMD's MI400 are expected to launch with HBM4.

Further out: **compute-in-memory DRAM** (Samsung HBM-PIM, SK Hynix AiM) places simple processing elements inside the HBM stack, allowing matrix-vector operations at the memory's bandwidth rather than requiring data to traverse the HBM interface to the GPU's ALUs. Early HBM-PIM products showed 2.5× bandwidth-efficiency gains for transformer attention on certain workloads. Full in-memory compute is discussed in a later section.

---

## CXL Memory Pooling and Disaggregation

**CXL** (Compute Express Link) is a PCIe 5.0-based coherent interconnect protocol that enables **memory disaggregation**: attaching large pools of DRAM (or other memory) to CPUs and GPUs over a standard fabric, with cache-coherent access ([Chapter 12](./12_buses_and_interconnects.md)).

The core idea is powerful: instead of each GPU having its own fixed HBM allocation, a CXL fabric allows **pooled memory** shared across multiple accelerators on demand. A training job that needs 512 GB of model state can draw from a 2 TB CXL memory pool; an inference job that needs only 48 GB relinquishes the rest. This is the **memory disaggregation** thesis.

<div class="diagram">
<div class="diagram-title">CXL Memory Disaggregation Architecture</div>
<div class="flow">
  <div class="flow-node accent wide">GPU 0 (80 GB HBM) ←CXL 3.0→ GPU 1 (80 GB HBM) ←CXL 3.0→ GPU N</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">CXL Switch / Fabric (CXL 3.0: up to 68 GB/s per port)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">CXL Memory Expander — 512 GB–2 TB DDR5/LPDDR5 modules (Samsung CMM-H, SK Hynix eMPM)</div>
</div>
</div>

**CXL protocol versions:**
- **CXL 1.0/1.1**: type-2 coherent access (GPU to host memory); PCIe 5.0 physical
- **CXL 2.0**: memory pooling — multiple hosts to one memory pool; switch support
- **CXL 3.0**: fabric-level multi-host memory sharing; back-to-back topology; PCIe 6.0 physical

Practical CXL memory pools (Samsung CMM-H with HBM, SK Hynix eMPM) entered production in 2024 for hyperscaler preview. The bandwidth of CXL (~68 GB/s at PCIe 5.0 ×16) is roughly 50–100× lower than HBM — so CXL memory is not a replacement for HBM's bandwidth but a **capacity expansion** for data that doesn't need peak bandwidth (model weights that are loaded once per inference request, KV caches for long-context, activation buffers).

For ML inference: CXL enables **economically serving 405B+ parameter models** on clusters that previously required all-HBM nodes, by placing model weights in CXL memory and KV caches in HBM. This changes the cost curve for long-context inference significantly.

---

## Co-Packaged Optics and Silicon Photonics

Electrical interconnects become exponentially more expensive per bit as distance increases and bandwidth scales. The energy cost to move a bit across a PCB trace at 56 Gbps is ~5–10 pJ/bit; across a 1m copper cable at 100 Gbps, ~15–25 pJ/bit; over a 100m fiber with an optical transceiver, ~2–5 pJ/bit. At long distances, **photons are already cheaper than electrons**.

The challenge is the last millimeter: electrical I/O remains necessary at the chip boundary. **Co-packaged optics (CPO)** eliminates the separate pluggable transceiver module and integrates the optical engine — modulator, detector, fiber coupler — directly into the chip package:

<div class="diagram">
<div class="diagram-title">Traditional vs Co-Packaged Optics</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Traditional Pluggable Module</div>
    <ul>
      <li>PCB → electrical trace (~10 cm) → cage → QSFP-DD pluggable → fiber</li>
      <li>~2–3 pJ/bit total; transceiver is ~15–20 W per 400G port</li>
      <li>SerDes runs at 56–112 Gbps per lane; reach limited</li>
      <li>Bandwidth density limited by front-panel port count</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Co-Packaged Optics (CPO)</div>
    <ul>
      <li>Silicon photonic die integrated in same package as switch/GPU ASIC</li>
      <li>Electrical connection between dies is < 5 mm (intra-package)</li>
      <li>~0.5–1 pJ/bit total; 3–5× power reduction per bit</li>
      <li>Bandwidth density improves ~5× vs pluggable</li>
    </ul>
  </div>
</div>
</div>

See [Appendix C — Photonics](./appendix_c_photonics.md) for silicon photonics fundamentals. CPO deployment timeline:
- **Switch ASICs**: Broadcom Tomahawk 5, Cisco Silicon One already deploying CPO in hyperscaler fabrics (2024–2025)
- **GPU interconnect**: NVIDIA Spectrum-X networking with optical-native design; longer timeline for GPU-to-GPU optical NVLink
- **Compute tiles**: fully optical compute (arithmetic in the optical domain) is 10–15+ years away from practicality at scale

For ML practitioners: CPO primarily matters for **multi-rack and inter-datacenter training clusters**. The bandwidth wall between nodes (typically 200–400 Gb/s InfiniBand per GPU vs 900+ GB/s NVLink within a node) is the binding constraint for distributed training at scale. CPO can increase inter-node bandwidth by 5–10× while reducing power — which would dramatically change the economics of pipeline parallelism vs tensor parallelism ([Chapter 28](./28_multi_gpu_and_distributed_training.md)).

---

## In-Memory and Analog Compute

### The von Neumann Bottleneck

Every compute device described in this guide is a **von Neumann machine**: compute logic and memory are physically separate, connected by a bus. The energy cost of reading a weight from SRAM and delivering it to an ALU is roughly 200–1,000× the energy of performing the multiply-accumulate on that weight. The compute is nearly free; the movement is the cost.

**Processing-in-Memory (PIM)** or **compute-in-memory (CIM)** places simple arithmetic units directly inside or adjacent to the memory array, eliminating the bulk of data movement:

- **Near-memory compute (NMC)**: logic die stacked on or beside DRAM, sharing the DRAM interface. Samsung HBM-PIM (2021) placed simple SIMD cores inside each HBM stack, demonstrating ~1.4× bandwidth efficiency improvement for GEMV operations. SK Hynix AiM (Accelerator-in-Memory) followed similar architecture.
- **In-SRAM compute**: performing analog dot-product operations using the SRAM bitcell's discharge current. Multiple startups (Mythic, Gyrfalcon, Hailo) have deployed 8-bit MAC operations in analog SRAM. The challenge: analog is inherently noisy, making precision beyond ~6 bits difficult; writes wear out SRAM faster than reads; and area efficiency depends heavily on workload regularity.
- **Resistive memory (ReRAM/PCM/MRAM)**: memristive devices can store multi-bit weights in their resistance state and perform analog MAC by Kirchhoff's current law. A weight matrix can be "programmed" into the conductance of a crossbar array and a dot product performed in one physical step. Analog precision (~4–6 bits effective), endurance, and write speed remain engineering challenges, but companies like Mythic (analog flash), Weebit Nano (ReRAM), and various academic groups have demonstrated inference-capable prototypes.

<div class="diagram">
<div class="diagram-title">Compute-in-Memory Spectrum</div>
<div class="flow">
  <div class="flow-node accent wide">Digital SRAM CIM — bitcell row/column modification for 1-bit operations; area inefficient but fully digital precision</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Analog SRAM CIM — partial-swing discharge MAC; ~4–8 bit precision; no additional memory area</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">HBM-PIM / DRAM-PIM — digital SIMD inside DRAM stack; full-precision digital; bandwidth-limited by internal array organization</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">ReRAM/PCM Crossbar — conductance-based analog MAC; matrix multiplication in O(1) time per crossbar; currently < 100 TOPS at inference precision</div>
</div>
</div>

**Practicality timeline for ML practitioners**: Digital PIM (HBM-PIM style) is commercially available now and useful for bandwidth-limited inference. Analog CIM at useful precision and scale is a 5–10 year horizon for mainstream deployment, with 2–3 products at the edge AI scale available today (Mythic's 25 TOPS analog inference chip).

---

## Wafer-Scale Computing

**Cerebras Systems** built the largest chip ever manufactured: the **Wafer Scale Engine 3 (WSE-3)**, shipped 2024. It is a single die spanning an entire 300mm silicon wafer — approximately 46,225 mm² (vs 815 mm² for H100). Key specs:

| | H100 SXM5 | Cerebras WSE-3 |
|--|:---------:|:--------------:|
| Die area | ~815 mm² | ~46,225 mm² |
| Transistors | ~80 B | ~4 trillion |
| SRAM on-chip | ~50 MB | ~900 GB |
| Core count | 132 SMs | 900,000 cores |
| Memory BW (on-chip SRAM) | ~33 TB/s (L2) | ~21 PB/s (all SRAM) |
| Peak FP16 TFLOP/s | 989 | ~125,000 (theoretical) |

The key insight is **on-chip SRAM bandwidth**: with ~900 GB of SRAM directly on-die, a Cerebras CS-3 can hold the weights of an entire 70B-parameter model (at appropriate precision) in SRAM — which delivers ~21 PB/s of bandwidth, ~2,600× greater than H100 HBM bandwidth. For models where the entire activation and weight set fits in SRAM, the memory wall essentially disappears.

The tradeoffs: extreme NRE cost (a wafer is ~$20,000+ to manufacture, most of it wasted on defects without the redundancy/yield tricks Cerebras employs), limited supply, and a software stack that requires rewriting models in Cerebras's custom toolchain. For training large language models, Cerebras reports significant speed advantages per system, but total cost of ownership versus a NVIDIA cluster depends strongly on workload.

---

## Neuromorphic and Spiking Neural Networks

**Neuromorphic chips** (Intel Loihi 2, IBM TrueNorth, SpiNNaker2) implement **spiking neural network** (SNN) computation: rather than multiplying weight matrices by dense activation vectors, neurons communicate via sparse binary spikes, and energy is consumed only when a neuron fires. The theoretical energy efficiency for sparse, event-driven workloads is extraordinary — Intel claims Loihi 2 can perform certain sparse workloads at **100–1,000× lower energy** than equivalent GPU computation.

The fundamental challenge: **training SNNs with backpropagation is hard.** The spike function is non-differentiable; surrogate gradient methods exist but loss convergence is slower and accuracy on complex benchmarks (ImageNet-scale) lags behind standard DNNs by a significant margin as of 2024. Neuromorphic hardware is currently best suited for:
- Ultra-low-power sensory processing (always-on keyword detection, vibration monitoring)
- Spiking models of scientific simulation (neuroscience)
- Research exploration of sparse-temporal computation

A direct path from modern transformer LLMs to neuromorphic hardware does not exist today. Neuromorphic is a long-horizon bet on whether the ML community finds model architectures that exploit event-driven sparsity — possible but not certain.

---

## Optical Computing

True **optical computing** — performing matrix multiplication in the optical domain using interference patterns and photodetectors — promises:
- Computation at the speed of light (literally)
- Matrix-vector products in $O(1)$ time per optical pass
- Near-zero compute energy (photons don't dissipate energy in transit)

Companies pursuing this include Lightmatter (Passage photonic interconnect + Envise computing platform), Luminous Computing (acquired), and academic groups at MIT, Stanford, and Oxford.

The practical engineering challenges are severe:
- **Precision**: optical analog systems are currently limited to ~4–8 bits effective precision due to noise and imperfect fabrication
- **Reprogrammability**: reloading weights (reprogramming the optical circuit) is slower than digital SRAM writes
- **Scale**: demonstrated systems are dozens to thousands of operations; useful scale for LLM inference requires billions
- **Integration**: converting between digital and optical domains (ADC/DAC at high resolution) consumes power that partially offsets compute savings

**Realistic timeline**: Optical interconnects (already deployed in co-packaged optics) are a 2025–2028 mainstream technology. Optical matrix compute for AI workloads is 10–20 years from commercial practicality at useful precision and scale. Lightmatter's Passage photonic interconnect chip (2024) is a real product for chip-to-chip optical communication — this is the near-term foothold.

---

## Quantum Computing: Tempered Expectations

Quantum computing receives outsized coverage in general technology media and occasionally in AI circles. The correct framing for ML practitioners:

**What quantum computing is**: a computational model using quantum mechanical superposition and entanglement to solve certain problem classes (integer factorization, quantum simulation, optimization on certain graph structures) with exponentially fewer operations than classical computers.

**What quantum computing is not**: a replacement for GPU matrix multiplication. The transformer attention mechanism, convolution, and linear algebra operations that constitute modern DL workloads do not have known quantum speedups that scale usefully with model size. Grover's algorithm provides a $\sqrt{N}$ speedup for unstructured search — which doesn't map onto neural network training.

**Current state (2024)**: Google's Willow chip (2024) demonstrated 105 qubits with below-threshold error correction — a genuine milestone in fault-tolerant quantum computing. IBM's 1,000+ qubit systems exist. But "quantum advantage" for practically relevant (non-synthetic benchmark) problems remains undemonstrated for machine learning.

**Honest timeline**: quantum advantage for ML training or inference optimization: **unlikely in the 5-year horizon**; speculative in the 10-year horizon. The energy efficiency of quantum hardware for optimization problems may become relevant for hyperparameter search or certain combinatorial ML subproblems in the 2035+ horizon.

---

## RISC-V and Open Hardware Momentum

RISC-V's trajectory (introduced in [Chapter 33](./33_design_paradigm_wars.md)) is accelerating:
- **China** has invested heavily; Alibaba's T-Head XuanTie RISC-V cores are in production silicon
- **India** has a national RISC-V SoC program (VEGA series)
- **RISC-V International** membership exceeds 4,000 organizations (2024)
- **SiFive P870** (2024) targets Neoverse N2 class performance
- **Ventana Veyron V2** (2024) claims competitive server-class throughput

For AI specifically: **RISC-V vector extension (RVV)** enables SIMD/matrix operations in a standard ISA, allowing custom AI accelerators to use RISC-V as the host/control processor (avoiding ARM royalties) while the accelerator matrix engine handles compute. This architecture is increasingly common in edge AI chips from Chinese vendors.

The RISC-V threat to the incumbent ISA landscape is real but gradual. The missing piece remains the software ecosystem: GCC/LLVM RISC-V support is excellent, but the equivalent of CUDA or CoreML-quality acceleration stacks on RISC-V are 3–5 years from maturity for mainstream ML workloads.

---

## Energy and Sustainability as the Binding Constraint

The largest datacenters in the world are now constrained by **power delivery**, not by silicon. A GB200 NVL72 rack consumes **~120 kW**. A 100,000-GPU cluster (not unusual for frontier training) consumes **~1.2 GW** — comparable to a small nuclear power plant.

Key figures (2024–2025):
- Training GPT-4-class models: estimated 50–100 MWh per training run
- Training Llama-3 405B: estimated ~30 MWh
- Global AI compute energy (2024): estimated 100–150 TWh/year, growing ~2× per year
- Google, Microsoft, Amazon all announcing new nuclear (SMR) power agreements to meet AI datacenter demand

This is not an abstract sustainability concern — it is a **binding engineering constraint** that will shape chip design:

1. **Power walls at the rack**: liquid cooling, rear-door heat exchangers, immersion cooling are becoming standard, not optional ([Appendix D — Power & Thermals](./appendix_d_power_and_thermals.md))
2. **Performance-per-watt competition**: the metric that determines data-center profitability is not TFLOP/s but **TFLOP/s per watt** (or tokens/joule for inference). Apple M-series and ARM Neoverse V2 win this metric for many workloads.
3. **Memory bottleneck amplified**: moving data consumes energy; the convergence of bandwidth costs and power budgets doubly penalizes memory-intensive operations.
4. **Low-precision dividend**: FP8 and FP4 reduce switching energy ($E \approx \tfrac{1}{2}CV_{dd}^2$, fewer capacitance switching) in addition to reducing bandwidth — the precision lever is simultaneously a compute, memory, and energy optimization.

---

## Industry Roadmap: 2025–2030

<div class="diagram">
<div class="diagram-title">Chip Industry Technology Roadmap (2025–2030)</div>
<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">2025</div>
    <div class="timeline-title">HBM4 First Silicon + Gate-All-Around at Scale</div>
    <div class="timeline-desc">Samsung/SK Hynix HBM4 samples; TSMC N2 GAA in volume (Apple A18); Intel 18A RibbonFET; UCIe 1.0 chiplet designs shipping; NVIDIA Rubin architecture disclosed</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2026</div>
    <div class="timeline-title">NVIDIA Rubin + AMD MI400 + CXL 3.0 Deployment</div>
    <div class="timeline-desc">Rubin on TSMC N2/A16 with HBM4; MI400 AMD chiplet accelerator; CXL 3.0 memory pools in hyperscaler deployments; co-packaged optics in switch ASICs reaching volume</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2027</div>
    <div class="timeline-title">3D Logic Stacking + SoIC in Production AI Chips</div>
    <div class="timeline-desc">TSMC SoIC-XS SRAM-on-GPU stacking in products; hybrid bonded L2 cache density increases 3–4×; GPT-scale training clusters operating below 1 GW; RISC-V competitive server class</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2028</div>
    <div class="timeline-title">HBM5 + CXL Memory Pooling Mainstream</div>
    <div class="timeline-desc">~8–12 TB/s per chip from HBM5; CXL 3.1 memory disaggregation at hyperscaler scale; wafer-scale chiplet integration via SoW; in-memory DRAM compute (HBM-PIM v3) shipping</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2029–2030</div>
    <div class="timeline-title">Post-GAA Nodes + Potential 1nm Class</div>
    <div class="timeline-desc">TSMC A14 / Intel 14A (beyond GAA); potential new transistor paradigms (CFET, 2D channel materials); co-packaged optics standard in GPU fabrics; optical compute prototypes at useful scale</div>
  </div>
</div>
</div>

---

## The "Hardware Lottery" and Specialization Risk

A point worth explicit attention: every technology described in this chapter is optimized — explicitly or implicitly — for **transformer-based neural networks**. Tensor cores are matmul engines; HBM bandwidth targets the attention and FFN access patterns of transformers; NVLink bandwidth is sized for the all-reduce patterns of data-parallel transformer training.

This creates a **hardware lottery** risk: silicon bets placed today (3–5 year design cycles) will ship into a 2027–2029 world that may have meaningfully different dominant model architectures. If the research community converges on:
- **State-space models** (Mamba, RWKV) instead of attention — the memory access pattern shifts from quadratic attention to sequential state updates
- **Mixture-of-experts** at scale — the bottleneck is routing and expert-to-expert communication, not bulk matmul
- **Sparse activation** architectures — memory bandwidth for weight loading, not matmul throughput, becomes the critical path

...then hardware designed around dense transformer matmul efficiency may partially miss the target. The companies best positioned for architecture-agnostic hardware are those that maintain **programmability** (NVIDIA's CUDA flexibility) or that scale a general-purpose axis (memory bandwidth, which benefits nearly any architecture).

> **The winning strategy over uncertain architecture horizons is to improve the axes that help regardless of model type: memory bandwidth, memory capacity, and interconnect bandwidth. These are universal constraints across all neural network inference and training patterns — only the degree of sensitivity changes.**

---

## Specialization Trend: Custom Silicon at Scale

The most important structural trend in the AI hardware industry outside of NVIDIA's dominance is **hyperscaler custom silicon**:

| Company | Chip | Purpose | Generation | Notes |
|---------|------|---------|-----------|-------|
| Google | TPU v5p / Trillium | Training + Inference | v5p 2023, Trillium 2024 | Used for all Gemini model training and inference |
| Amazon (AWS) | Trainium 2 / Inferentia 3 | Training / Inference | 2024 | NeuronSDK; growing PyTorch support |
| Microsoft | Azure Maia 100 | Training | 2023 | OpenAI models in Azure; co-designed with NVIDIA for Copilot+ |
| Meta | MTIA (Meta Training and Inference Accelerator) | Inference | v2 2024 | Recommendation models, ad ranking; growing LLM inference use |
| Apple | Neural Engine (ANE) | On-device inference | M4 Ultra 76 TOPS | CoreML; supports INT8/INT16; transformer optimizations in M4 |

The hyperscalers' motivation is straightforward: at 100,000+ GPU scale, a 10% efficiency improvement from hardware-software co-design saves hundreds of millions of dollars per year. NVIDIA captures most of this economic surplus today; custom silicon lets hyperscalers recapture it. Each of these chips runs a **proprietary software compiler** (Google XLA, AWS NeuronSDK, etc.) — further fragmenting the ML hardware ecosystem.

---

## Why This Matters for Model Optimization (3–5 Year View)

Translating the above into actionable guidance:

**Bandwidth and memory stay king.** The memory wall ([Chapter 11](./11_memory_wall_and_bandwidth.md)) is not going away; it is getting worse for peak-TFLOP/s hardware and only partially addressed by HBM4/CXL. Architectures that minimize memory footprint (weight tying, quantization, parameter sharing) and inference patterns that batch efficiently will have disproportionate hardware advantage.

**Precision will keep dropping.** FP8 is mainstream by 2025; FP4 (NVIDIA's NVFP4 in Blackwell's 4th-gen tensor cores) is viable for inference by 2025; MX (microscaling) formats ([Chapter 23](./23_number_formats_deep_dive.md)) allow per-block quantization at FP6/FP4 with minimal accuracy loss. Model training at FP8 end-to-end is an active research frontier. The practitioners who invest in quantization-aware training and mixed-precision workflows now will have the most hardware leverage in 2026.

**Software portability becomes a competitive advantage.** As the hardware landscape fragments — NVIDIA, AMD, Google TPU, AWS Trainium, Apple, custom accelerators — the ability to run on multiple substrates becomes economically valuable. Triton kernels, torch.compile, OpenXLA, and model quantization formats (GGUF, ONNX, MLIR) are the portability layer. Investing in framework-agnostic model design reduces your cost and risk.

**Co-design deepens.** The line between model architecture and hardware will continue to blur. The GPU's 2:4 structured sparsity format ([Chapter 26](./26_sparsity_hardware_view.md)) required model training that accommodated it; FP8 training required modified loss scaling and gradient handling. As HBM4, CXL memory pooling, and in-memory compute mature, the models and systems that take advantage will be co-designed from the start — not retrofitted.

**Energy per inference, not peak TFLOP/s, is the business metric that matters.** A model deployed at 100M users × 50 requests/day × 1,000 tokens ≈ $5 \times 10^{12}$ tokens/day. At H100 pricing ($3.00/GPU-hour, ~5,000 tokens/second), that requires ~27,000 GPU-hours/day. A 2× improvement in tokens/watt reduces electricity costs and hardware footprint proportionately. This is why FP8/FP4, speculative decoding, attention sparsity, and KV cache compression ([Chapter 29](./29_memory_hierarchy_in_action.md)) all directly impact business case.

<div class="diagram">
<div class="diagram-title">The Recurring Theme: Data Movement Is the Bottleneck</div>
<div class="layer-stack">
  <div class="layer accent">Model optimization (quantization, sparsity, architecture) → smaller activations, fewer bytes moved</div>
  <div class="layer blue">Advanced packaging (CoWoS, 3D, UCIe) → shorter, denser die-to-die paths</div>
  <div class="layer green">HBM4 / in-memory compute → higher bandwidth and/or moving compute to the data</div>
  <div class="layer purple">CXL memory pooling → right-size memory without bottlenecking bandwidth</div>
  <div class="layer orange">Co-packaged optics → cheaper bits over longer distances</div>
  <div class="layer teal">All address the same root constraint: moving data costs more than computing on it</div>
</div>
</div>

---

## Key Takeaways

- **The post-Moore era** shifts generational gains from process scaling to specialization, packaging, memory architecture, and interconnects — the levers that address data movement rather than raw switching speed.
- **Chiplets and UCIe** disaggregate SoC design, enabling mixed-node packaging and a future chiplet marketplace; AMD proved the concept in CPUs, and the entire industry is following.
- **Advanced packaging** (CoWoS, 3D hybrid bonding, SoIC) has become a primary performance differentiator — HBM on an interposer is the reason AI GPUs have TB/s of bandwidth at all.
- **CXL memory pooling** enables disaggregated, right-sized memory for AI inference infrastructure, reducing cost for 70B+ model serving; commercially available at small scale in 2024.
- **Co-packaged optics** will reduce inter-node communication energy 3–5× by 2027–2028, partially relieving the distributed training bandwidth tax; already deployed in hyperscaler switch fabrics.
- **In-memory and analog compute** are real but constrained to precision-tolerant edge inference today; digital PIM (HBM-PIM) is commercially available for bandwidth-limited workloads.
- **Wafer-scale (Cerebras)** and **neuromorphic** chips occupy specialist niches; optical compute and quantum are 10–20 year horizons for ML workloads.
- **Energy is now the binding constraint** at datacenter scale; perf/watt dominates perf/dollar; quantization and efficiency improvements compound into hundreds of millions of dollars in savings at hyperscaler deployment scale.

---

*Next: [Appendix A — How Chips Are Made](./appendix_a_how_chips_are_made.md), a step-by-step walk through the physical fabrication process — from sand and wafer to EUV lithography, etch, doping, packaging, and yield — the manufacturing story that underpins everything in this guide.*

[← Back to Table of Contents](./README.md)
