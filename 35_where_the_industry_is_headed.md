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
2. **Moore's Law is slowing** (~2018–present): transistor density still grows, but at 2nm and below, cost (an EUV scanner is ~$200M, a High-NA scanner ~$380M, a leading-edge fab $20B+) and physics (tunneling, contact resistance, variability) constrain each step. Node names are now marketing labels; logic density still improves ~1.15–1.6× per node, but SRAM and analog barely shrink, and cost per transistor has stopped falling.

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

| UCIe version | Data rate | Bandwidth density | Reach | Notes |
|--------------|:---------:|:----------------:|:--------:|-------|
| UCIe 1.0/1.1, advanced package | up to 32 GT/s | up to ~1.3 TB/s per mm of die edge | ~2 mm | Silicon interposer or bridge (CoWoS, EMIB) |
| UCIe 1.0/1.1, standard package | up to 32 GT/s | up to ~224 GB/s per mm of die edge | ~25 mm | Organic substrate, lower cost |
| UCIe 2.0 (2024) | — | adds UCIe-3D: bandwidth per mm² of face, hybrid bonding at ≤10 µm pitch | vertical | 3D stacking; manageability and test |
| UCIe 3.0 (2025) | 48 and 64 GT/s | ~2× UCIe 1.x | as above | Higher-speed 2D links |

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
    <div class="card-desc">TSMC's 2.5D interposer: GPU die + HBM stacks side-by-side on silicon interposer. Enables 3–8 TB/s of HBM bandwidth per package today, ~20 TB/s with HBM4. Used in A100, H100, MI300X; CoWoS-L (silicon bridges in an organic interposer) carries the two-die B200/B300 and Rubin.</div>
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
    <div class="card-desc">Intel's 3D stacking for logic dies. Tiles face-up on a base die connected via hybrid bonding. Enables disaggregated CPU tiles with shared power/IO base. Used in Meteor Lake, Lunar Lake and Panther Lake; EMIB bridges connect tiles side by side.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">SoIC (System on Integrated Chips)</div>
    <div class="card-desc">TSMC's 3D stacking using hybrid bond. SoIC-XS targets < 9 µm bonding pitch. Logic stacked on logic at near-BEOL interconnect density. AMD's MI300 stacks compute dies on I/O dies with it; AMD's X3D CPUs stack cache the same way.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Wafer-Level Fan-Out (FOWLP)</div>
    <div class="card-desc">Dies embedded in reconstituted wafer, connections routed through packaging substrate. Used for mobile SoCs (TSMC InFO). High volume, lower cost, moderate bandwidth density.</div>
  </div>
</div>
</div>

The near-term trajectory (2025–2028): CoWoS interposers have grown from ~2,500 mm² (H100, about three reticles) to ~3.3 reticles for Blackwell, and TSMC is planning 5.5- and 9-reticle versions to accommodate more HBM stacks and larger multi-die compute complexes. TSMC's "System on Wafer" (SoW) concepts push this further toward wafer-scale integration for the highest-end systems.

---

## HBM4 and Memory Beyond

HBM3e (deployed in H200, MI325X, B200) is approaching the bandwidth-density ceiling of the current physical interface. **HBM4**, in volume production since early 2026, is the most significant HBM architectural change since HBM2:

| | HBM3e | HBM4 |
|--|-------|------|
| Bus width per stack | 1,024 bits | **2,048 bits** |
| Channels per stack | 16 | **32** |
| Data rate | 8–9.6 Gbps/pin | 8 Gbps (JEDEC); ~10–11 shipping |
| BW per stack | ~1.0–1.2 TB/s | **~2.0–2.8 TB/s** |
| Stacks per GPU | 8 | 8 (Rubin), 12 (MI455X) |
| Total BW | ~8 TB/s | **~20 TB/s** |
| Capacity per stack | 24–36 GB | 36 GB (12-high) → 48 GB (16-high) |

HBM4's doubled bus width requires changes to both the DRAM stack and the logic die interface — it is a co-designed specification, not just a DRAM improvement. NVIDIA's Rubin and AMD's MI450/MI455X are the first accelerators built on it (2026). For the first time the base die can be made on a logic process (TSMC N12/N3-class), so GPU vendors can place custom logic under the DRAM stack; HBM4E follows in 2027.

Further out: **compute-in-memory DRAM** (Samsung HBM-PIM, SK Hynix AiM) places simple processing elements inside the HBM stack, allowing matrix-vector operations at the memory's bandwidth rather than requiring data to traverse the HBM interface to the GPU's ALUs. Samsung reported roughly 2× performance and large energy savings for memory-bound kernels in its HBM-PIM trials; none has yet reached a mainstream GPU product. Full in-memory compute is discussed in a later section.

---

## CXL Memory Pooling and Disaggregation

**CXL** (Compute Express Link) is a coherent interconnect that runs over the PCIe physical layer (PCIe 5 for CXL 1.1/2.0, PCIe 6 for CXL 3.x, PCIe 7 for CXL 4.0 from late 2025) that enables **memory disaggregation**: attaching large pools of DRAM (or other memory) to CPUs and GPUs over a standard fabric, with cache-coherent access ([Chapter 12](./12_buses_and_interconnects.md)).

The core idea is powerful: instead of each GPU having its own fixed HBM allocation, a CXL fabric allows **pooled memory** shared across multiple accelerators on demand. A training job that needs 512 GB of model state can draw from a 2 TB CXL memory pool; an inference job that needs only 48 GB relinquishes the rest. This is the **memory disaggregation** thesis.

<div class="diagram">
<div class="diagram-title">CXL Memory Disaggregation Architecture</div>
<div class="flow">
  <div class="flow-node accent wide">Host CPUs (and CXL-capable accelerators) — each with local DDR5 / HBM</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">CXL switch / fabric (x16 port: ~64 GB/s per direction on PCIe 5, ~128 GB/s on PCIe 6)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">CXL memory expanders and pooled appliances — 128 GB–2 TB of DDR5 per device (Samsung CMM-D, SK hynix CMM-DDR5, Micron CZ120)</div>
</div>
</div>

**CXL protocol versions:**
- **CXL 1.0/1.1**: defines Type 1 (cache-coherent accelerators), Type 2 (accelerators with their own memory) and Type 3 (memory expanders); PCIe 5.0 physical
- **CXL 2.0**: memory pooling — multiple hosts to one memory pool; switch support
- **CXL 3.x**: fabric-level multi-host memory sharing, multi-level switching; PCIe 6.0 physical
- **CXL 4.0** (Nov 2025): PCIe 7.0 signalling (128 GT/s) and bundled ports for more bandwidth per device

CXL Type 3 memory expanders from Samsung, SK hynix and Micron have shipped since 2024, and Intel Xeon 6 and AMD EPYC 9005 servers support them. The bandwidth of a CXL link (~64 GB/s per direction at PCIe 5.0 ×16) is roughly 100× lower than a modern GPU's HBM, so CXL memory is not a replacement for HBM bandwidth but a **capacity tier** for data that is touched rarely: idle sessions' KV caches, prefix caches, embedding tables, checkpoints and optimizer state being offloaded.

For ML inference the practical win is **KV-cache tiering**: keep active sequences' KV in HBM, park paused conversations and shared prefixes in host or CXL memory, and page them back in a few milliseconds instead of recomputing the prefill. Model weights do *not* belong there: decode reads every weight every step, and at CXL bandwidth a 405B model would generate a token every few seconds. NVIDIA's own answer to the same need is a different link: Grace's NVLink-C2C reaches CPU memory at 900 GB/s.

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
- **Switch ASICs (shipping)**: Broadcom's Bailly (51.2 Tb/s Tomahawk 5 CPO) is in volume and its 102.4 Tb/s Tomahawk 6 CPO is sampling; NVIDIA's Quantum-X Photonics InfiniBand and Spectrum-X Photonics Ethernet switches (built with TSMC's COUPE optical engines) ship from 2026, with NVIDIA claiming ~3.5× better power efficiency than pluggable optics
- **Chip I/O**: optical chiplets (Ayar Labs TeraPHY) and photonic interposers (Lightmatter Passage) bring fibre to the package edge of compute dies; still pre-volume
- **GPU-to-GPU scale-up**: NVLink stays copper inside the rack through Rubin Ultra; optics first replaces the scale-out network
- **Compute tiles**: fully optical compute (arithmetic in the optical domain) is 10–15+ years away from practicality at scale

For ML practitioners: CPO primarily matters for **multi-rack and inter-datacenter training clusters**. The bandwidth wall between nodes (typically 400–800 Gb/s, i.e. 50–100 GB/s, of InfiniBand or Ethernet per GPU vs 900–1,800 GB/s of NVLink within a node) is the binding constraint for distributed training at scale. CPO can increase inter-node bandwidth by 5–10× while reducing power — which would dramatically change the economics of pipeline parallelism vs tensor parallelism ([Chapter 28](./28_multi_gpu_and_distributed_training.md)).

---

## In-Memory and Analog Compute

### The von Neumann Bottleneck

Every compute device described in this guide is a **von Neumann machine**: compute logic and memory are physically separate, connected by a bus. The energy cost of reading a weight from SRAM and delivering it to an ALU is roughly 200–1,000× the energy of performing the multiply-accumulate on that weight. The compute is nearly free; the movement is the cost.

**Processing-in-Memory (PIM)** or **compute-in-memory (CIM)** places simple arithmetic units directly inside or adjacent to the memory array, eliminating the bulk of data movement:

- **Near-memory compute (NMC)**: logic die stacked on or beside DRAM, sharing the DRAM interface. Samsung HBM-PIM (2021) placed simple SIMD cores inside each HBM stack, demonstrating ~1.4× bandwidth efficiency improvement for GEMV operations. SK Hynix AiM (Accelerator-in-Memory) followed similar architecture.
- **In-memory compute**: performing dot products inside the memory array. Mythic computes in analog inside flash cells; EnCharge AI uses charge on capacitors next to SRAM; d-Matrix's Corsair uses *digital* in-SRAM compute for data-center inference; IBM Research demonstrated a 14 nm analog phase-change chip with 35 million weights (Nature, 2023). The challenge: analog is inherently noisy, making precision beyond ~6 bits difficult; writes wear out SRAM faster than reads; and area efficiency depends heavily on workload regularity.
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
| SRAM on-chip | ~50 MB L2 (+ ~30 MB SMEM/L1) | ~44 GB |
| Core count | 132 SMs | 900,000 cores |
| Memory BW | 3.35 TB/s HBM; ~30 TB/s aggregate SMEM/L1 | ~21 PB/s (all SRAM) |
| Peak FP16 TFLOP/s | 989 (dense) | ~125,000 (vendor peak, with sparsity) |

The key insight is **on-chip SRAM bandwidth**: 44 GB of SRAM spread across the wafer delivers ~21 PB/s, about 6,000× an H100's HBM bandwidth. 44 GB holds ~20B parameters at FP16, so Cerebras serves larger models by pipelining layers across several wafers (and, for training, streams weights in from external MemoryX servers). For whatever fits in SRAM, the memory wall essentially disappears, which is why Cerebras and Groq lead single-stream token rates. Cerebras went public in May 2026.

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

Companies have pursued this, but the most prominent, Lightmatter, has shifted its emphasis from optical compute (Envise) to optical *interconnect* (Passage), which says a lot about where the near-term value is. Academic groups at MIT, Stanford and Oxford continue the compute work.

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

**Current state (2026)**: Google's Willow chip (Dec 2024) demonstrated 105 qubits with below-threshold error correction, and in 2025 Google reported a verifiable speed-up on a physics simulation task ("Quantum Echoes"). IBM, with 1,000+ qubit systems already built, targets a fault-tolerant machine (Starling) around 2029. But "quantum advantage" for practically relevant (non-synthetic benchmark) problems remains undemonstrated for machine learning.

**Honest timeline**: quantum advantage for ML training or inference optimization: **unlikely in the 5-year horizon**; speculative in the 10-year horizon. The energy efficiency of quantum hardware for optimization problems may become relevant for hyperparameter search or certain combinatorial ML subproblems in the 2035+ horizon.

---

## RISC-V and Open Hardware Momentum

RISC-V's trajectory (introduced in [Chapter 33](./33_design_paradigm_wars.md)) is accelerating:
- **China** has invested heavily; Alibaba's T-Head XuanTie RISC-V cores are in production silicon
- **India** has a national RISC-V SoC program (VEGA series)
- **RISC-V International** membership exceeds 4,000 organizations, and the RVA23 profile (ratified Oct 2024) gives application processors a stable target including the vector extension
- **Tenstorrent** builds AI accelerators around RISC-V cores and licenses its Ascalon CPU IP
- **SiFive P870** (2024) targets Neoverse N2 class performance
- **Ventana Veyron V2** (2024) claims competitive server-class throughput

For AI specifically: **RISC-V vector extension (RVV)** enables SIMD/matrix operations in a standard ISA, allowing custom AI accelerators to use RISC-V as the host/control processor (avoiding ARM royalties) while the accelerator matrix engine handles compute. This architecture is increasingly common in edge AI chips from Chinese vendors.

The RISC-V threat to the incumbent ISA landscape is real but gradual. The missing piece remains the software ecosystem: GCC/LLVM RISC-V support is excellent, but the equivalent of CUDA or CoreML-quality acceleration stacks on RISC-V are 3–5 years from maturity for mainstream ML workloads.

---

## Energy and Sustainability as the Binding Constraint

The largest datacenters in the world are now constrained by **power delivery**, not by silicon. A GB200 NVL72 rack consumes **~120–130 kW**, about 1.7 kW per GPU once CPUs, switches and fans are counted. A 100,000-GPU Blackwell cluster therefore draws roughly **150–200 MW** with cooling, and the gigawatt campuses announced in 2025 (OpenAI's Stargate sites, Meta's Hyperion) are built for several hundred thousand GPUs each — comparable to a large nuclear power plant.

Key figures:
- Training Llama 3.1 405B took 30.8 million H100 GPU-hours (Meta's model card): about **21 GWh** at the GPU's 700 W rating alone, more with hosts and cooling
- GPT-4-class training runs are estimated in the **tens of GWh**
- All data centres used about **415 TWh** of electricity in 2024 (~1.5% of the world's), and the IEA projects ~945 TWh by 2030 with AI the main driver
- Google, Microsoft, Amazon all announcing new nuclear (SMR) power agreements to meet AI datacenter demand

This is not an abstract sustainability concern — it is a **binding engineering constraint** that will shape chip design:

1. **Power walls at the rack**: liquid cooling, rear-door heat exchangers, immersion cooling are becoming standard, not optional ([Appendix D — Power & Thermals](./appendix_d_power_and_thermals.md))
2. **Performance-per-watt competition**: the metric that determines data-center profitability is not TFLOP/s but **TFLOP/s per watt** (or tokens/joule for inference). For inference that becomes tokens per joule, where the newest accelerators running FP4 lead and each generation's gain comes as much from datatypes as from silicon.
3. **Memory bottleneck amplified**: moving data consumes energy; the convergence of bandwidth costs and power budgets doubly penalizes memory-intensive operations.
4. **Low-precision dividend**: FP8 and FP4 reduce switching energy ($E \approx \tfrac{1}{2}CV_{dd}^2$, fewer capacitance switching) in addition to reducing bandwidth — the precision lever is simultaneously a compute, memory, and energy optimization.

---

## Industry Roadmap: 2025–2030

<div class="diagram">
<div class="diagram-title">Chip Industry Technology Roadmap (2025–2030)</div>
<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">2025 (done)</div>
    <div class="timeline-title">Blackwell Ultra, N2 in Volume, 18A Ramps</div>
    <div class="timeline-desc">GB300 NVL72 ships; TSMC N2 (first GAA) enters volume in Q4; Intel 18A with backside power ramps; HBM4 sampling; MI355X ships; NVFP4 pretraining demonstrated</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2026 (now)</div>
    <div class="timeline-title">HBM4 Year: Rubin, Helios, TPU 8</div>
    <div class="timeline-desc">Vera Rubin NVL72 with HBM4 and NVLink 6; AMD MI450-series Helios racks with UALink; Google TPU 8t/8i; Trainium3 at scale; CPO switches ship; first N2 phones</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2027</div>
    <div class="timeline-title">Rubin Ultra, HBM4E, Backside Power at TSMC</div>
    <div class="timeline-desc">Rubin Ultra (4 GPU dies per package) in Kyber racks; HBM4E; TSMC A16 with Super Power Rail; Trainium4 with NVLink Fusion</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2028</div>
    <div class="timeline-title">Feynman, A14, High-NA at Intel</div>
    <div class="timeline-desc">NVIDIA Feynman generation; TSMC A14; Intel 14A as the first High-NA EUV node; PCIe 8.0 finalized; CXL memory tiers routine in AI servers</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2029–2030</div>
    <div class="timeline-title">Beyond Nanosheets</div>
    <div class="timeline-desc">TSMC A13/A12; complementary FET (CFET) and 2D-channel research toward production; optics moving into GPU scale-up fabrics; optical compute still in the lab</div>
  </div>
</div>
</div>

---

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Every bet in this chapter attacks the cost of moving data. Pick the one you think arrives in your own workload first (HBM4, CPO, CXL tiers, SRAM-first decode chips or in-memory compute), and name the single number that would tell you it has arrived.</p>
</div>

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
| Google | TPU v7 Ironwood, TPU 8t / 8i | Training + inference (split in gen 8) | 2025 / 2026 | Gemini training and serving; sold to outside labs (Anthropic: up to 1M TPUs) |
| Amazon (AWS) | Trainium2 / Trainium3 | Training and inference | 2024 / Dec 2025 | Neuron SDK; Trainium4 to support NVLink Fusion |
| Microsoft | Maia 100 → Maia 200 | Inference | 2023 / Jan 2026 | Maia 200: TSMC 3 nm, ~10 PF FP4, 216 GB HBM3e |
| Meta | MTIA v2 and successors | Ranking, recommendation, growing LLM inference | 2024 onward | 256 MB SRAM + LPDDR5 on v2 |
| OpenAI | Custom accelerator with Broadcom | Inference | first racks 2H 2026 | 10 GW deployment plan |
| Apple | Neural Engine + GPU neural accelerators | On-device inference | M4 (38 TOPS ANE), M5 (2025) | Core ML; MLX; M5 puts matrix units in every GPU core |

The hyperscalers' motivation is straightforward: at 100,000+ GPU scale, a 10% efficiency improvement from hardware-software co-design saves hundreds of millions of dollars per year. NVIDIA captures most of this economic surplus today; custom silicon lets hyperscalers recapture it. Each of these chips runs a **proprietary software compiler** (Google XLA, AWS NeuronSDK, etc.) — further fragmenting the ML hardware ecosystem.

---

## Why This Matters for Model Optimization (3–5 Year View)

Translating the above into actionable guidance:

**Bandwidth and memory stay king.** The memory wall ([Chapter 11](./11_memory_wall_and_bandwidth.md)) is not going away; it is getting worse for peak-TFLOP/s hardware and only partially addressed by HBM4/CXL. Architectures that minimize memory footprint (weight tying, quantization, parameter sharing) and inference patterns that batch efficiently will have disproportionate hardware advantage.

**Precision will keep dropping.** FP8 is mainstream by 2025; FP4 (NVFP4 in Blackwell's fifth-generation tensor cores, MXFP4 on AMD and TPU 8) is in production inference by 2025–26, and 4-bit pretraining has been demonstrated; MX (microscaling) formats ([Chapter 23](./23_number_formats_deep_dive.md)) allow per-block quantization at FP6/FP4 with minimal accuracy loss. FP8 training is in production (DeepSeek-V3 was trained in FP8). The practitioners who invest in quantization-aware training and mixed-precision workflows now will have the most hardware leverage in 2026.

**Software portability becomes a competitive advantage.** As the hardware landscape fragments — NVIDIA, AMD, Google TPU, AWS Trainium, Apple, custom accelerators — the ability to run on multiple substrates becomes economically valuable. Triton kernels, torch.compile, OpenXLA, and model quantization formats (GGUF, ONNX, MLIR) are the portability layer. Investing in framework-agnostic model design reduces your cost and risk.

**Co-design deepens.** The line between model architecture and hardware will continue to blur. The GPU's 2:4 structured sparsity format ([Chapter 26](./26_sparsity_hardware_view.md)) required model training that accommodated it; FP8 training required modified loss scaling and gradient handling. As HBM4, CXL memory pooling, and in-memory compute mature, the models and systems that take advantage will be co-designed from the start — not retrofitted.

**Energy per inference, not peak TFLOP/s, is the business metric that matters.** A model deployed at 100M users × 50 requests/day × 1,000 tokens ≈ $5 \times 10^{12}$ tokens/day. At ~5,000 tokens/second per H100, that is about 280,000 GPU-hours a day — roughly 11,600 GPUs running around the clock, ~$830,000 a day at $3 per GPU-hour. A 2× improvement in tokens/watt reduces electricity costs and hardware footprint proportionately. This is why FP8/FP4, speculative decoding, attention sparsity, and KV cache compression ([Chapter 29](./29_memory_hierarchy_in_action.md)) all directly impact business case.

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
- **CXL memory** adds a cheap, slow capacity tier (KV-cache parking, embeddings, offload), not a substitute for HBM bandwidth; expanders have shipped since 2024 and CXL 4.0 moves to PCIe 7.
- **Co-packaged optics** is shipping in switches in 2026 (Broadcom, NVIDIA Quantum-X/Spectrum-X Photonics) with ~3.5× better link power efficiency; scale-up NVLink stays copper for now.
- **In-memory and analog compute** are real but constrained to precision-tolerant edge inference today; digital PIM (HBM-PIM) is commercially available for bandwidth-limited workloads.
- **Wafer-scale (Cerebras)** and **neuromorphic** chips occupy specialist niches; optical compute and quantum are 10–20 year horizons for ML workloads.
- **Energy is now the binding constraint** at datacenter scale; perf/watt dominates perf/dollar; quantization and efficiency improvements compound into hundreds of millions of dollars in savings at hyperscaler deployment scale.

## Check Yourself

<details class="check">
<summary>A 100,000-GPU cluster of NVL72 racks: estimate its power draw, and the annual energy at full load.</summary>
<p>~1.7 kW per GPU all-in → ~170 MW, ~200 MW with cooling overhead (PUE ~1.2). × 8,760 h ≈ 1.75 TWh per year.</p>
</details>

<details class="check">
<summary>Why is CXL memory a poor place for model weights during decode, but a good place for paused KV caches?</summary>
<p>Decode reads every weight every step, so weights need the full HBM bandwidth (~8 TB/s vs ~64 GB/s for CXL). A paused conversation's KV cache is read once when it resumes, so a few milliseconds of transfer beats recomputing the prefill.</p>
</details>

<details class="check">
<summary>How much SRAM does a Cerebras WSE-3 have, and roughly how many FP16 parameters fit in it?</summary>
<p>~44 GB, which holds ~22B parameters at 2 bytes each (less once activations and KV are counted). Larger models are pipelined across several wafers.</p>
</details>

<details class="check">
<summary>HBM4 doubles the interface width. Why does a Rubin GPU get ~2.5× the bandwidth of B200 rather than 2×?</summary>
<p>Width doubles (1,024 → 2,048 bits per stack) and the shipping pin rate (~10–11 Gb/s) is slightly above HBM3e's ~8 Gb/s in B200, so per-stack bandwidth rises ~2.5× with the same eight stacks.</p>
</details>

<details class="check">
<summary>Llama 3.1 405B used 30.8 million H100 GPU-hours. Estimate its training energy and explain why the estimate is a lower bound.</summary>
<p>30.8 × 10⁶ h × 0.7 kW ≈ 21.6 GWh. It counts only GPU board power at TDP; CPUs, networking, storage and cooling (PUE) add more, while GPUs rarely sit at full TDP, which pulls the other way.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| UCIe specifications | UCIe Consortium | [uciexpress.org/specifications](https://www.uciexpress.org/specifications) | Die-to-die standard; 2.0 (3D) and 3.0 (64 GT/s) |
| JEDEC HBM4 standard (JESD270-4) | JEDEC, Apr 2025 | [jedec.org/news/pressreleases/jedec-and-indust…](https://www.jedec.org/news/pressreleases/jedec-and-industry-leaders-collaborate-release-jesd270-4-hbm4-standard) | HBM4 interface and channels |
| CXL specifications | CXL Consortium | [computeexpresslink.org](https://computeexpresslink.org/) | CXL 1.1 through 4.0 |
| NVIDIA Spectrum-X and Quantum-X silicon photonics switches | NVIDIA, Mar 2025 | [nvidianews.nvidia.com/news/nvidia-spectrum-x-…](https://nvidianews.nvidia.com/news/nvidia-spectrum-x-co-packaged-optics-networking-switches-ai-factories) | CPO switches and power claims |
| Energy and AI | International Energy Agency, 2025 | [iea.org/reports/energy-and-ai](https://www.iea.org/reports/energy-and-ai) | Data-centre electricity 2024 and 2030 |
| Llama 3.1 model card | Meta, 2024 | [github.com/meta-llama/llama-models/blob/main/…](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) | Training GPU-hours for 405B |
| Cerebras WSE-3 | Cerebras, 2024 | [cerebras.ai/chip](https://www.cerebras.ai/chip) | 44 GB SRAM, 21 PB/s, 900k cores |
| An analog-AI chip for energy-efficient speech recognition and transcription | Ambrogio et al., Nature 2023 | [doi.org/10.1038/s41586-023-06337-5](https://doi.org/10.1038/s41586-023-06337-5) | IBM phase-change in-memory compute |
| The Hardware Lottery | Hooker, 2020 | [arxiv.org/abs/2009.06489](https://arxiv.org/abs/2009.06489) | Specialization risk |
| Quantum error correction below the surface code threshold | Google Quantum AI, Nature 2024 | [doi.org/10.1038/s41586-024-08449-y](https://doi.org/10.1038/s41586-024-08449-y) | Willow below-threshold result |

---

**Next:** [Appendix A — How Chips Are Made](./appendix_a_how_chips_are_made.md) — A step-by-step walk through fabrication, from sand to EUV lithography, etch, doping, packaging and yield: the manufacturing story under every chip in this guide.

[← Back to Table of Contents](./README.md)
