---
title: "Appendix E — Packaging & Advanced Integration"
---

[← Back to Table of Contents](./README.md)

# Appendix E — Packaging & Advanced Integration

For the first four decades of the integrated circuit era, packaging was an afterthought — just the plastic or ceramic shell you needed to connect a chip to a PCB. That changed around 2010. As transistor scaling slowed and Moore's Law began to deliver diminishing returns (see [Chapter 35](./35_where_the_industry_is_headed.md)), the industry discovered that how you *assemble* multiple chips together could provide the bandwidth, density, and die-size benefits that the fab alone could no longer offer cheaply. Today, advanced packaging is a primary competitive differentiator. It is why an H100 has 80 GB of HBM sitting microns away from the compute die, and why NVIDIA's Blackwell and AMD's MI300X are "chips" made of multiple dies acting as one.

---

## Why Packaging Became a Primary Lever

Three forces converged to elevate packaging:

<div class="diagram">
<div class="diagram-title">Why Advanced Packaging Became Necessary</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">The Reticle Limit</div>
    <div class="card-desc">EUV scanners expose ~858 mm² per shot. A single die cannot be larger. H100 is 814 mm² — already at the edge. To go bigger, you need multiple dies.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Yield Economics</div>
    <div class="card-desc">At 0.1 defects/cm², a 1,000 mm² die would yield ~37%, while a 500 mm² die yields ~61%. Chiplets restore economics. (See Appendix A.)</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Bandwidth Wall</div>
    <div class="card-desc">A GPU across a PCIe 5 slot reaches host DRAM at ~64 GB/s each way. HBM stacks beside the die on the same interposer give 3.35 TB/s (H100, five stacks) to 8 TB/s (B200, eight). The only way to get there is proximity.</div>
  </div>
</div>
</div>

> **The packaging hierarchy** has expanded: die → package → board → rack is now sometimes die → chiplet → 2.5D interposer → package → board. Each level is a design space with its own interconnect density, cost, and power tradeoffs.

---

## Traditional Packaging

In a conventional **2D wire-bond or flip-chip** package, a single die is mounted on a substrate (an organic laminate similar to a mini-PCB), and connections are made either by wire bonds (tiny gold wires from die pads to substrate pads) or by **C4 bumps** (solder balls under the die — "flip chip"). The substrate fans the ~100s of connections out to the BGA (ball grid array) on the bottom, which solders to the board.

```text
Traditional 2D Package (flip-chip BGA):

   ┌────────────────────────────────┐
   │      Single Die                │  ← Active silicon
   │  (compute, cache, I/O on one)  │
   └───────────┬────────────────────┘
               │ C4 bumps (50–150 µm pitch)
   ┌───────────┴────────────────────┐
   │      Organic Substrate         │  ← routing + power delivery
   └───────────┬────────────────────┘
               │ BGA balls (0.5–1 mm pitch)
              PCB
```

**Limitations:**
- Die size limited by reticle (~858 mm²).
- Memory must live in separate DRAM packages, connected via PCB traces — limited bandwidth (DDR5 ~38–51 GB/s per 64-bit channel; the widest GDDR7 graphics card, RTX 5090, reaches ~1.8 TB/s from 16 chips on a 512-bit bus).
- Off-package interconnects are lossy and power-hungry.

---

## 2.5D Integration: The Silicon Interposer

**2.5D packaging** places multiple dies side-by-side on a large piece of silicon called an **interposer** — a passive (no active transistors) routing layer with extremely dense wiring.

### The Interposer Concept

A silicon interposer is manufactured using semiconductor fab processes (but without transistors) and thus achieves **wire pitch of 0.4–1 µm**, roughly 5–20× finer than the 2–10 µm lines of an organic substrate. This density enables thousands of die-to-die connections over short distances, enabling:

- **Very high bandwidth** between adjacent dies (multi-TB/s)
- **Low power per bit** (short wires, small capacitance)
- **Different process nodes on the same package** (logic on latest node, memory on mature node)

<div class="diagram">
<div class="diagram-title">2.5D CoWoS Cross-Section</div>
</div>

```text
2.5D CoWoS (Chip-on-Wafer-on-Substrate) cross-section:

 ┌───────────────┐    ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
 │  Compute Die  │    │ HBM  │ │ HBM  │ │ HBM  │ │ HBM  │
 │  (GPU/ASIC)   │    │ stk 1│ │ stk 2│ │ stk 3│ │ stk 4│
 │  ~800 mm²     │    │      │ │      │ │      │ │      │
 └───────┬───────┘    └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
         │               │        │         │         │
         └───────────────┴────────┴─────────┴─────────┘
                      Silicon Interposer
              (passive, fine-pitch wiring, ~1200 mm²)
              micro-bumps: ~45 µm pitch, ~10,000+ connections
                         │
              ┌──────────┴──────────┐
              │   Organic Substrate  │   ← power delivery, BGA
              └──────────┬──────────┘
                         │ BGA
                        PCB
```

### CoWoS: TSMC's 2.5D Platform

**CoWoS (Chip-on-Wafer-on-Substrate)** is TSMC's branded 2.5D technology:

- The dies (GPU compute die + HBM stacks) are placed on the interposer at **wafer level** before dicing — hence "on-wafer."
- Micro-bumps at ~45 µm pitch connect die to interposer (vs ~130 µm for standard flip-chip).
- The assembled interposer is then mounted on an organic substrate.

**This is how every H100, H200, A100, and AMD MI300-series GPU is built.** The NVIDIA H100 SXM uses CoWoS-S (with an interposer); it places the GH100 die and six HBM3 sites (five active) on one interposer, delivering **3.35 TB/s** of HBM bandwidth — roughly 7× a 12-channel DDR5 server socket. Blackwell moved to **CoWoS-L**: small silicon bridges (local silicon interconnects) embedded in a larger organic interposer, which is cheaper to make at the ~3.3-reticle size two GPU dies and eight HBM stacks need.

| Technology | Wiring / bond pitch | What it connects | Examples |
|---|---|---|---|
| Standard organic substrate | 2–10 µm lines; ~130 µm C4 bumps | Die to board | Consumer GPUs, CPUs |
| CoWoS-S (full silicon interposer) | 0.4–1 µm lines; ~40–55 µm micro-bumps | GPU + HBM side by side | A100, H100, H200, MI250, MI300 |
| CoWoS-R (RDL organic interposer) | ~2 µm lines | Cheaper 2.5D, lower density | Networking, some ASICs |
| CoWoS-L (organic + silicon bridges) | bridges at silicon density | Multi-reticle 2.5D | B200, B300, Rubin |
| SoIC (3D hybrid bonding) | ~6–9 µm Cu-Cu, shrinking | Die stacked on die | AMD MI300 (compute dies on I/O dies), X3D CPUs |

---

## HBM: The 3D Memory Stack

**High Bandwidth Memory (HBM)** is itself a 3D integration story and deserves its own treatment here alongside the interposer. (For the full HBM specification, see [Chapter 9](./09_memory_types.md).)

An HBM "stack" consists of:
- A **base logic die** (with ECC, refresh, PHY)
- 8–16 **DRAM dies** stacked on top (12-high is common for HBM3e; 16-high arrives with HBM4)
- Connected by **Through-Silicon Vias (TSVs)** — vertical conductors etched through each die

```text
HBM3 Stack (8-die example):

  ┌──────────────────┐  ← DRAM die 8
  ├──────────────────┤  ← DRAM die 7
  ├──────────────────┤     ...
  ├──────────────────┤  ← DRAM die 2
  ├──────────────────┤  ← DRAM die 1
  ├──────────────────┤  ← Base logic die (ECC, refresh, PHY)
  └────────┬─────────┘
           │ micro-bumps → interposer
```

Each TSV is ~5–10 µm diameter — tiny enough to pack thousands of them per die, giving HBM its wide **1,024-bit** memory interface per stack. Compare to GDDR7: a 32-bit bus per device, 16 devices → 512-bit total (RTX 5090), across much longer PCB traces.

| Memory | Bus width | Bandwidth | Connection |
|---|---|---|---|
| GDDR7 (RTX 5090) | 32-bit per chip, 512-bit total | ~1.8 TB/s for 16 chips | PCB traces |
| HBM2e | 1,024-bit per stack | ~461 GB/s per stack | Interposer micro-bumps |
| HBM3 | 1,024-bit | ~819 GB/s per stack (at 6.4 Gb/s) | Interposer micro-bumps |
| HBM3e | 1,024-bit | ~1.0–1.2 TB/s per stack | Interposer micro-bumps |
| HBM4 | 2,048-bit | ~2.0–2.8 TB/s per stack | Interposer; logic-process base die |

Latency is not HBM's advantage: from the GPU core, HBM and GDDR both take on the order of hundreds of nanoseconds. Its advantage is width.

Five HBM3 stacks (as in H100 SXM5) at H100's ~5.2 Gb/s pin rate → **3.35 TB/s** (4.1 TB/s at the full HBM3 rate). This is what allows the roofline analysis in [Chapter 11](./11_memory_wall_and_bandwidth.md) to shift so dramatically for GPU clusters versus CPU servers.

---

## 3D Stacking: Logic-on-Logic

True **3D integration** stacks *active* logic dies on top of each other. The key enabling technologies:

### Through-Silicon Vias (TSVs)

A **TSV** is a vertical electrical connection that passes completely through a silicon die. Manufacturing requires:
1. Deep reactive-ion etching (DRIE) of holes ~5–10 µm in diameter, 50–100 µm deep.
2. Insulation liner deposition (SiO₂).
3. Fill with copper or tungsten.
4. Die grinding ("back-grind") to expose TSV tops.

TSVs are how HBM stacks its DRAM layers, and how some 3D logic products connect two compute dies.

### Hybrid Bonding

**Hybrid bonding** is the latest advance in die-to-die interconnect. Unlike micro-bumps (solder or copper pillars), hybrid bonding directly bonds:
- **Dielectric to dielectric** (SiO₂ surfaces fused together)
- **Copper to copper** (Cu pads at 1–10 µm pitch, bonded at wafer level)

The result is a **virtually bumpless** connection at sub-10 µm pitch — 5–10× denser than micro-bumps. TSMC's **SoIC (System on Integrated Chips)** platform uses hybrid bonding.

### AMD 3D V-Cache: Logic-on-Logic Today

**AMD 3D V-Cache** (used in Ryzen 7 5800X3D, EPYC Genoa-X) stacks an additional 64 MB SRAM cache die directly on top of the CPU compute die using hybrid bonding. The two dies are connected with 9 µm pitch Cu-Cu bonds, achieving ~2 TB/s of bandwidth between the core and the 3D-stacked SRAM — nearly 10× more than a 2D approach. From Zen 5 (Ryzen 9000X3D, 2024) AMD flipped the stack and put the cache die *under* the compute die, so the hot cores sit next to the heat spreader.

```text
AMD 3D V-Cache (cross-section concept):

  ┌──────────────────────────────┐
  │     64 MB SRAM Cache Die     │  ← stacked on top
  └────────────┬─────────────────┘
               │ hybrid bond pads (~9 µm pitch, ~200K connections)
  ┌────────────┴─────────────────┐
  │   CPU Compute Die (CCD)      │  ← bottom die (compute + L1/L2)
  └──────────────────────────────┘
```

---

## Chiplets: Disaggregating the Monolithic Die

A **chiplet** is a smaller die designed to be integrated with other dies in a multi-chip package. Instead of one giant monolithic die, you break the design into functional blocks, each fabricated separately and assembled together.

### The Yield and Cost Argument

The relationship between die area and yield is strongly nonlinear (see [Appendix A](./appendix_a_how_chips_are_made.md)):

$$Y_{\text{Poisson}} = e^{-D_0 A} \qquad Y_{\text{neg-bin}} \approx \left(1 + \frac{D_0 \cdot A}{n}\right)^{-n}$$

At a defect density $D_0 = 0.1\,\text{defects/cm}^2$ (and clustering $n = 2$ for the second model):

| Die area | Poisson yield | Negative binomial, n = 2 |
|---|---|---|
| 100 mm² | ~90% | ~91% |
| 200 mm² | ~82% | ~83% |
| 400 mm² | ~67% | ~69% |
| 800 mm² | ~45% | ~51% |
| 1,600 mm² (impossible monolithic) | ~20% | ~31% |

**Chiplet strategy:** Instead of one 800 mm² die at ~45% yield, use four 200 mm² dies at ~82% yield each. Cost per good die drops dramatically.

<div class="diagram">
<div class="diagram-title">Chiplet Packaging Layout (conceptual multi-die package)</div>
</div>

```text
Chiplet-Based Package (top-down view):

  ┌─────────────────────────────────────────────────────────┐
  │                    Package Substrate                      │
  │                                                           │
  │  ┌────────────┐  ┌────────────┐  ┌──────┐ ┌──────┐      │
  │  │  Compute   │  │  Compute   │  │ I/O  │ │ I/O  │      │
  │  │  Chiplet   │  │  Chiplet   │  │ Die  │ │ Die  │      │
  │  │  (N5 node) │  │  (N5 node) │  │(N7)  │ │(N7)  │      │
  │  └────────────┘  └────────────┘  └──────┘ └──────┘      │
  │                                                           │
  │  ┌────────────┐  ┌────────────┐  ┌──────┐ ┌──────┐      │
  │  │  Compute   │  │  Compute   │  │ I/O  │ │ I/O  │      │
  │  │  Chiplet   │  │  Chiplet   │  │ Die  │ │ Die  │      │
  │  └────────────┘  └────────────┘  └──────┘ └──────┘      │
  │                                                           │
  └─────────────────────────────────────────────────────────┘
    Different nodes, same package — "heterogeneous integration"
```

### Real-World Chiplet Implementations

**AMD Zen architecture (EPYC / Ryzen):** AMD pioneered chiplets at scale in 2017. An EPYC processor uses:
- Multiple **CCDs (Core Chiplets)** — compute dies on latest node (e.g., TSMC N5)
- One or two **IODs (I/O Dies)** — memory controllers, PCIe, Infinity Fabric — on a mature cheaper node (N6/N7)

This split lets AMD put high-performance CPU cores on expensive N5 silicon while handling I/O on cheaper N6, reducing overall cost significantly.

**NVIDIA Blackwell (B200):** The B200 is a single "GPU" but consists of two connected dies:
- Two reticle-sized Blackwell dies, linked by NV-HBI (NVIDIA High-Bandwidth Interface) at 10 TB/s.
- Together they appear as a single 208B-transistor, 1,000W GPU.
- Both dies and eight HBM3e stacks sit on one CoWoS-L interposer; the die-to-die link runs through it, and software sees one GPU.

**AMD MI300X:** Arguably the most aggressive chiplet integration for AI:
- 8 accelerator compute dies (XCDs, CDNA 3) on TSMC N5
- hybrid-bonded (SoIC) on top of 4 I/O dies on N6, which hold the Infinity Cache and memory controllers
- 8 HBM3 stacks beside them on a CoWoS-S interposer
- MI300A swaps two XCDs for three Zen 4 CPU dies (CCDs), making an APU with 128 GB of shared HBM
- Total for MI300X: **192 GB HBM3** at 5.3 TB/s, 153B transistors across 12 dies

---

## UCIe: Standardizing Die-to-Die Interconnect

The chiplet ecosystem had a problem: die-to-die interconnects from AMD, Intel, and NVIDIA were all proprietary. **UCIe (Universal Chiplet Interconnect Express)** is an industry standard (announced 2022, backed by Intel, AMD, TSMC, ARM, Qualcomm, Samsung and others) that defines:

- **Physical layer:** bump pitch (from ~100–130 µm on organic substrates to 25–55 µm on advanced packages, and ≤10 µm for UCIe-3D), pad definition, packaging options
- **Die-to-die adapter:** protocol-agnostic streaming and packet layers
- **Protocol mapping:** PCIe 5/6 and CXL 2/3 can run over UCIe

| UCIe option | Bump pitch | Bandwidth density (UCIe 1.x, 32 GT/s) | Target |
|---|---|---|---|
| Standard package (organic) | ~100–130 µm | up to ~224 GB/s per mm of die edge | Low cost |
| Advanced package (bridge or interposer) | ~25–55 µm | up to ~1.3 TB/s per mm of die edge | HPC chiplets |
| UCIe-3D (2.0, hybrid bonding) | ≤10 µm | measured per mm² of die face | 3D stacks |

UCIe 3.0 (2025) raises the 2D data rate to 48 and 64 GT/s.

UCIe makes it possible to buy a compute chiplet from one vendor, an HBM controller chiplet from another, and an I/O die from a third — and assemble them. This is the "chiplet marketplace" vision; it is still early but is actively shaping [Chapter 35](./35_where_the_industry_is_headed.md)'s post-Moore roadmap.

---

## Packaging Technologies: Full Comparison

| Technology | Integration | Connection pitch | Examples | ML relevance |
|---|---|---|---|---|
| Wire-bond (2D) | Single die on substrate | ~50–100 µm pads | Low-end MCUs, sensors | Legacy |
| Flip-chip (2D) | Single die + C4 bumps | ~130–150 µm | CPUs, mainstream GPUs | Off-package DRAM only |
| EMIB (Intel) | 2.5D silicon bridge in organic substrate | ~36–55 µm micro-bumps | Ponte Vecchio, Sapphire Rapids, Xeon 6 | Modular multi-die |
| Foveros (Intel) | 3D logic on base die | ~36 µm bumps → ~9 µm (Foveros Direct) | Meteor/Lunar/Panther Lake | Client SoCs |
| CoWoS-S / -L (TSMC 2.5D) | Dies + HBM on interposer | ~40–55 µm micro-bumps | H100, MI300 (S); B200, B300, Rubin (L) | HBM enablement |
| SoIC (TSMC 3D) | Die on die, hybrid bonded | ~6–9 µm | MI300 XCD-on-IOD | Dense 3D compute |
| HBM DRAM stack (3D) | DRAM on DRAM via TSV | ~55 µm micro-bumps to interposer | Every AI accelerator | Training and inference bandwidth |
| AMD 3D V-Cache | SRAM on logic | ~9 µm hybrid bond | Ryzen X3D, EPYC Genoa-X | Cache-heavy workloads |

---

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>HBM bandwidth grows with the number of stacks you can fit around the die, which is set by interposer size, which is set by how many reticles TSMC can stitch. If interposers stop growing, where does the next 2× of bandwidth come from: faster pins, wider stacks, or memory stacked directly on the compute die?</p>
</div>

## ML Implications: Why Packaging Is Your Problem Too

As an ML practitioner optimizing models, packaging decisions upstream determine what you're working with:

<div class="diagram">
<div class="diagram-title">How Packaging Decisions Reach Your Model</div>
<div class="flow">
  <div class="flow-node accent wide">2.5D CoWoS places HBM 1 mm from compute die<br/><small>→ 3.35–5.3 TB/s bandwidth becomes available</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">High bandwidth enables large KV-caches without stalls<br/><small>→ longer context windows become feasible on one GPU (Ch 29)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">Chiplets let GPU scale past reticle limit (208B transistors in B200)<br/><small>→ more tensor cores, more SRAM capacity, bigger batches</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Multiple GPU dies in one package require intra-chip NVLink<br/><small>→ transparent to software but sets tensor/pipeline parallelism topology</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">192–288 GB of HBM per package (MI300X, B300, MI355X) changes the "fits on one GPU" math<br/><small>→ a 70B FP16 model (140 GB) fits on one device instead of two H100s</small></div>
</div>
</div>

Concretely:
- The reason you can do long-context inference without spilling to system memory is HBM bandwidth — which only exists because of 2.5D packaging.
- The reason B200 has 208B transistors is its two-die design — a single die that size would exceed the reticle, and yield would collapse long before that.
- The reason MI300X can hold a 70B model in one device (192 GB) is aggressive HBM stacking on an interposer.

None of these are software decisions — but they set the envelope within which every software decision you make (batching, quantization, parallelism strategy) plays out.

---

## Key Takeaways

- **Packaging became a primary performance lever** as Moore's Law slowed. Getting dies physically close enables bandwidth that no off-package approach can match.
- **2.5D (CoWoS-S, CoWoS-L)** places compute dies + HBM stacks on an interposer at ~40–55 µm micro-bump pitch, enabling TB/s-class memory bandwidth. Every modern AI accelerator uses it; CoWoS capacity was the AI supply bottleneck of 2023–25.
- **HBM is 3D integration** — DRAM dies stacked via TSVs on a base die, giving a 1,024-bit-wide interface per stack. Five HBM3 stacks give H100 its 3.35 TB/s; eight HBM4 stacks give Rubin ~20 TB/s.
- **Chiplets** break a large monolithic die into smaller, higher-yield pieces assembled in-package. AMD's EPYC and NVIDIA Blackwell are flagship examples. Yield math makes this decisive at large die areas.
- **Reticle limit (~858 mm²)** is the hard ceiling on a single die. To go bigger — more compute, more HBM — you must go multi-die.
- **UCIe** is the emerging open standard for die-to-die interconnect, enabling a future chiplet ecosystem where compute, memory, and I/O dies from different vendors snap together.
- For ML practitioners: packaging choices upstream set your memory capacity, bandwidth, and compute ceiling. Understanding them explains why different hardware generations have the memory sizes, bandwidths, and multi-device topologies they do.

## Check Yourself

<details class="check">
<summary>Why does a silicon interposer allow more die-to-die bandwidth than an organic substrate?</summary>
<p>It is patterned with fab processes, so wires are ~5–20× finer and micro-bumps far denser; thousands of short, low-capacitance connections fit along each die edge.</p>
</details>

<details class="check">
<summary>H100's HBM3 runs at ~5.2 Gb/s per pin on five 1,024-bit stacks. Show that this gives ~3.35 TB/s.</summary>
<p>5 × 1,024 bits × 5.2 Gb/s = 26.6 Tb/s ≈ 3.33 TB/s.</p>
</details>

<details class="check">
<summary>What is the difference between CoWoS-S and CoWoS-L, and why did Blackwell need the latter?</summary>
<p>CoWoS-S uses one monolithic silicon interposer; CoWoS-L embeds small silicon bridges in an organic interposer. Two reticle-sized dies plus eight HBM stacks need an interposer ~3.3 reticles in size, which is cheaper and higher-yielding as CoWoS-L.</p>
</details>

<details class="check">
<summary>How is MI300X built, die by die?</summary>
<p>Eight CDNA 3 compute dies (N5) hybrid-bonded onto four I/O dies (N6), with eight HBM3 stacks beside them on a CoWoS-S interposer: twelve logic dies plus memory in one package.</p>
</details>

<details class="check">
<summary>At 0.1 defects/cm², compare the Poisson yield of one 800 mm² die with four 200 mm² chiplets.</summary>
<p>e^(−0.8) ≈ 45% versus e^(−0.2) ≈ 82% per chiplet. Because bad chiplets are discarded individually, far more of each wafer ends up in working products.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| TSMC 3DFabric (CoWoS, InFO, SoIC) | TSMC | [3dfabric.tsmc.com](https://3dfabric.tsmc.com/) | Packaging platform definitions |
| AMD Instinct MI300 series architecture | AMD, 2023 | [amd.com/en/products/accelerators/instinct/mi3…](https://www.amd.com/en/products/accelerators/instinct/mi300.html) | XCD/IOD stacking and HBM3 |
| NVIDIA Blackwell architecture | NVIDIA, 2024 | [nvidia.com/en-us/data-center/technologies/bla…](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) | Two-die GPU with 10 TB/s NV-HBI |
| JEDEC HBM4 standard (JESD270-4) | JEDEC, Apr 2025 | [jedec.org/news/pressreleases/jedec-and-indust…](https://www.jedec.org/news/pressreleases/jedec-and-industry-leaders-collaborate-release-jesd270-4-hbm4-standard) | 2,048-bit HBM4 |
| UCIe specifications | UCIe Consortium | [uciexpress.org/specifications](https://www.uciexpress.org/specifications) | Die-to-die standard |
| Intel EMIB and Foveros | Intel | [intel.com/content/www/us/en/foundry/packaging…](https://www.intel.com/content/www/us/en/foundry/packaging.html) | 2.5D bridges and 3D stacking |
| AMD 3D V-Cache (ISSCC 2022) | Wuu et al., ISSCC 2022 | [doi.org/10.1109/ISSCC42614.2022.9731565](https://doi.org/10.1109/ISSCC42614.2022.9731565) | Hybrid-bonded SRAM on logic |

---

**Next:** [Appendix F — Glossary](./appendix_f_glossary.md) — Every acronym and term from the guide, defined in one place.

[← Back to Table of Contents](./README.md)
