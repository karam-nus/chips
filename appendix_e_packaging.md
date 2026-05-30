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
    <div class="card-desc">Defect density means a 1,000 mm² die might yield 30%, while two 500 mm² dies yield 60–70% each. Chiplets restore economics. (See Appendix A.)</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Bandwidth Wall</div>
    <div class="card-desc">A GPU across a PCIe slot gets ~128 GB/s to DRAM. An HBM stack co-packaged on the same substrate gives 3.35 TB/s. The only way to get there is proximity.</div>
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
- Memory must live on a separate DRAM package, connected via PCB traces — limited bandwidth (DDR5 ~50–100 GB/s per channel, typical GPU with 8 channels = ~800 GB/s).
- Off-package interconnects are lossy and power-hungry.

---

## 2.5D Integration: The Silicon Interposer

**2.5D packaging** places multiple dies side-by-side on a large piece of silicon called an **interposer** — a passive (no active transistors) routing layer with extremely dense wiring.

### The Interposer Concept

A silicon interposer is manufactured using semiconductor fab processes (but without transistors) and thus achieves **wire pitch of 0.4–1 µm** — 50–100× finer than an organic substrate's 2–10 µm pitch. This density enables thousands of die-to-die connections over short distances, enabling:

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

**This is how every H100, H200, A100, and AMD MI300-series GPU is built.** The NVIDIA H100 SXM uses CoWoS-S (with an interposer); it places the GH100 die + 5 HBM3 stacks on one interposer, delivering **3.35 TB/s** of HBM bandwidth — 20× more than a comparable DDR5 arrangement.

| Technology | Die-to-die pitch | Bandwidth | TSMC name | Examples |
|---|---|---|---|---|
| Standard organic substrate | 2–10 µm wire | ~100 GB/s off-pkg | N/A | Consumer GPU, CPU |
| CoWoS-S (silicon interposer) | 0.4–1 µm | 1–4 TB/s | CoWoS | H100, A100, MI250 |
| CoWoS-R (RDL interposer) | ~2 µm | 0.5–2 TB/s | CoWoS | More cost-efficient |
| SoIC (3D) | ~9 µm (Cu-Cu) | >10 TB/s | SoIC | N/A yet at GPU scale |

---

## HBM: The 3D Memory Stack

**High Bandwidth Memory (HBM)** is itself a 3D integration story and deserves its own treatment here alongside the interposer. (For the full HBM specification, see [Chapter 9](./09_memory_types.md).)

An HBM "stack" consists of:
- A **base logic die** (with ECC, refresh, PHY)
- 4–12 **DRAM dies** stacked on top
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

Each TSV is ~5–10 µm diameter — tiny enough to pack thousands of them per die, giving HBM its wide **1,024-bit** memory interface per stack. Compare to GDDR6X: 32-bit bus per device, 16 devices → 512-bit total, across much longer PCB traces.

| Memory | Bus width | Per-stack BW (HBM3) | Connection | Latency |
|---|---|---|---|---|
| GDDR6X | 32-bit per chip (512-bit total) | ~1.2 TB/s for 16 chips | PCB traces | ~100 ns |
| HBM2e | 1,024-bit | 461 GB/s | Interposer micro-bumps | ~60 ns |
| HBM3 | 1,024-bit | 819 GB/s | Interposer micro-bumps | ~50 ns |
| HBM3e | 1,024-bit | 1,230 GB/s | Interposer micro-bumps | ~50 ns |

Five HBM3 stacks (as in H100 SXM5) → **4.1 TB/s**. This is what allows the roofline analysis in [Chapter 11](./11_memory_wall_and_bandwidth.md) to shift so dramatically for GPU clusters versus CPU servers.

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

**AMD 3D V-Cache** (used in Ryzen 7 5800X3D, EPYC Genoa-X) stacks an additional 64 MB SRAM cache die directly on top of the CPU compute die using hybrid bonding. The two dies are connected with 9 µm pitch Cu-Cu bonds, achieving ~2 TB/s of bandwidth between the core and the 3D-stacked SRAM — nearly 10× more than a 2D approach.

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

$$Y \approx \left(1 + \frac{D_0 \cdot A}{n}\right)^{-n}$$

At a defect density $D_0 = 0.1\,\text{defects/cm}^2$ and $n = 2$:

| Die area | Approx yield |
|---|---|
| 100 mm² | ~91% |
| 400 mm² | ~67% |
| 800 mm² | ~44% |
| 1,600 mm² (impossible monolithic) | ~20% hypothetical |

**Chiplet strategy:** Instead of one 800 mm² die at 44% yield, use four 200 mm² dies at ~82% yield each. Cost per good die drops dramatically.

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
- Two **B100 dies** (each ~814 mm²), linked by NVLink-Chip (NVL-C) at 10 TB/s aggregate between the two halves.
- Together they appear as a single 208B-transistor, 1,000W GPU.
- This is not CoWoS multi-die but a chip-to-chip NVLink fabric at the package level.

**AMD MI300X:** Arguably the most aggressive chiplet integration for AI:
- 3 GPU compute dies (CDNA 3 architecture)
- 4 CDNA 2 CPU dies (for MI300A variant)
- 8 HBM3 stacks
- All on a 2.5D silicon interposer (CoWoS-equivalent)
- Total: **192 GB HBM3** at 5.3 TB/s

---

## UCIe: Standardizing Die-to-Die Interconnect

The chiplet ecosystem had a problem: die-to-die interconnects from AMD, Intel, and NVIDIA were all proprietary. **UCIe (Universal Chiplet Interconnect Express)** is an industry standard (announced 2022, backed by Intel, AMD, TSMC, ARM, Qualcomm, Samsung and others) that defines:

- **Physical layer:** bump pitch (25 µm → 2 µm), pad definition, packaging options
- **Die-to-die adapter:** protocol-agnostic streaming and packet layers
- **Protocol mapping:** PCIe 5/6 and CXL 2/3 can run over UCIe

| UCIe tier | Bump pitch | Bandwidth density | Target |
|---|---|---|---|
| Standard (organic pkg) | 25–55 µm | ~2 Gb/s/mm | Low cost |
| Advanced (silicon bridge) | 10 µm | ~28 Gb/s/mm | HPC chiplets |
| Advanced+ (interposer) | 2–4 µm | >100 Gb/s/mm | Future AI |

UCIe makes it possible to buy a compute chiplet from one vendor, an HBM controller chiplet from another, and an I/O die from a third — and assemble them. This is the "chiplet marketplace" vision; it is still early but is actively shaping [Chapter 35](./35_where_the_industry_is_headed.md)'s post-Moore roadmap.

---

## Packaging Technologies: Full Comparison

| Technology | Integration | Die-to-die pitch | BW/mm | Examples | ML relevance |
|---|---|---|---|---|---|
| Wire-bond (2D) | Single die on substrate | N/A | N/A | Low-end MCUs, sensors | Legacy |
| Flip-chip (2D) | Single die + C4 bumps | N/A (off-pkg DRAM) | Low | CPUs, mainstream GPUs | Limited |
| EMIB (Intel) | 2.5D silicon bridge | ~55 µm | ~4 Gb/s/mm | Xe GPU, Ponte Vecchio | Modular AI |
| CoWoS-S (TSMC 2.5D) | Multi-die on Si interposer | ~45 µm (micro-bump) | ~6 Tb/s total | H100, A100, MI250 | HBM enablement |
| TSMC SoIC (3D) | Stacked logic + memory | ~9 µm hybrid bond | High | N5+N5 stacked future | Future ML |
| HBM DRAM stack (3D) | DRAM on DRAM via TSV | ~55 µm (μbump to pkg) | 1–5 TB/s per stack | H100, MI300, TPU | Training bandwidth |
| AMD 3D V-Cache | Logic on logic | ~9 µm hybrid bond | ~2 TB/s cache | EPYC Genoa-X, Ryzen X3D | Cache-heavy inference |

---

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
  <div class="flow-node orange wide">MI300X: 192 GB HBM on-package changes the "fits on one GPU" math<br/><small>→ 70B FP16 model fits in 140 GB — one MI300X vs 2× H100 for memory capacity</small></div>
</div>
</div>

Concretely:
- The reason you can do long-context inference without spilling to system memory is HBM bandwidth — which only exists because of 2.5D packaging.
- The reason B200 has 208B transistors is two-die chiplet design — otherwise it would be yield-limited.
- The reason MI300X can hold a 70B model in one device (192 GB) is aggressive HBM stacking on an interposer.

None of these are software decisions — but they set the envelope within which every software decision you make (batching, quantization, parallelism strategy) plays out.

---

## Key Takeaways

- **Packaging became a primary performance lever** as Moore's Law slowed. Getting dies physically close enables bandwidth that no off-package approach can match.
- **2.5D (CoWoS)** places compute die + HBM stacks on a silicon interposer at ~45 µm micro-bump pitch, enabling TB/s-class memory bandwidth. Every modern AI accelerator (H100, A100, MI300X) uses this.
- **HBM is 3D integration** — DRAM dies stacked via TSVs on a base die, giving a 1,024-bit-wide interface per stack. Five stacks at HBM3 delivers ~4 TB/s.
- **Chiplets** break a large monolithic die into smaller, higher-yield pieces assembled in-package. AMD's EPYC and NVIDIA Blackwell are flagship examples. Yield math makes this decisive at large die areas.
- **Reticle limit (~858 mm²)** is the hard ceiling on a single die. To go bigger — more compute, more HBM — you must go multi-die.
- **UCIe** is the emerging open standard for die-to-die interconnect, enabling a future chiplet ecosystem where compute, memory, and I/O dies from different vendors snap together.
- For ML practitioners: packaging choices upstream set your memory capacity, bandwidth, and compute ceiling. Understanding them explains why different hardware generations have the memory sizes, bandwidths, and multi-device topologies they do.

---

*Next: [Appendix F — Glossary](./appendix_f_glossary.md), where every acronym and term from the guide is defined in one place.*

[← Back to Table of Contents](./README.md)
