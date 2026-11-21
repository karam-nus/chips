---
title: "Appendix A — How Chips Are Made"
---

[← Back to Table of Contents](./README.md)

# Appendix A — How Chips Are Made

Chapter 1 established what a chip *is* — billions of transistors etched into silicon, wired together by metal. This appendix asks the harder question: *how do you actually make one?* The answer involves growing crystals the size of a person, polishing surfaces flat to atomic precision, printing patterns smaller than a virus with light that is nearly X-ray, and repeating those patterns dozens of times to build a 3D stack that is simultaneously the most complex and most mass-produced object humanity has ever made.

The fabrication journey spans roughly three months, costs hundreds of millions of dollars for an advanced design, and requires a supply chain so specialized that a single company — ASML — holds a monopoly on the most critical tool. Understanding it explains why leading-edge GPUs are perpetually supply-constrained, why a die can't simply be made bigger to add more compute, and why every generation of AI accelerator comes with a price tag that would have seemed absurd a decade ago.

> **The one-sentence version:** A chip is made by growing an ultra-pure silicon crystal, slicing it into wafers, then repeatedly coating the surface with light-sensitive material, projecting a circuit pattern onto it with deep-UV or EUV light, etching away what the light revealed, implanting atoms to change conductivity, and depositing new material — layer by layer, perhaps a hundred times — until transistors and metal wires fill a stack half a millimeter thick but patterned to within a few atoms.

The whole journey on one page; each box is a section below.

<!-- flowchart:fab_loop -->
<div class="flowchart" id="fc-fab_loop">
<div class="fc-title">Sand to packaged chip: the manufacturing loop</div>
<svg viewBox="0 0 800 380" width="800" role="img" aria-label="Sand to packaged chip: the manufacturing loop">
<path class="fc-edge" d="M175,32 L200,32"/><polygon class="fc-head" points="200,32 193,35.9 193,28.1"/>
<path class="fc-edge" d="M370,32 L390,32"/><polygon class="fc-head" points="390,32 383,35.9 383,28.1"/>
<path class="fc-edge" d="M560,32 L580,32"/><polygon class="fc-head" points="580,32 573,35.9 573,28.1"/>
<path class="fc-edge" d="M680,54 L680,84 L120,84 L120,118"/><polygon class="fc-head" points="120,118 116.2,111 123.8,111"/>
<path class="fc-edge" d="M205,134 L225,134"/><polygon class="fc-head" points="225,134 218,137.8 218,130.2"/>
<path class="fc-edge" d="M395,134 L410,134"/><polygon class="fc-head" points="410,134 403,137.8 403,130.2"/>
<path class="fc-edge" d="M590,134 L605,134"/><polygon class="fc-head" points="605,134 598,137.8 598,130.2"/>
<path class="fc-edge" d="M690,150 L690,210"/><polygon class="fc-head" points="690,210 686.1,203 693.9,203"/>
<path class="fc-edge" d="M605,226 L590,226"/><polygon class="fc-head" points="590,226 597,222.2 597,229.8"/>
<path class="fc-edge" d="M410,226 L395,226"/><polygon class="fc-head" points="395,226 402,222.2 402,229.8"/>
<path class="fc-edge" d="M225,226 L205,226"/><polygon class="fc-head" points="205,226 212,222.2 212,229.8"/>
<path class="fc-edge loop" d="M120,210 L120,150"/><polygon class="fc-head loop" points="120,150 123.8,157 116.2,157"/><text class="fc-elabel loop" x="128" y="186" text-anchor="start">next layer</text>
<path class="fc-edge" d="M120,242 L120,320"/><polygon class="fc-head" points="120,320 116.2,313 123.8,313"/><text class="fc-elabel " x="128" y="286" text-anchor="start">all layers done</text>
<path class="fc-edge" d="M215,336 L255,336"/><polygon class="fc-head" points="255,336 248,339.9 248,332.1"/>
<path class="fc-edge" d="M405,336 L425,336"/><polygon class="fc-head" points="425,336 418,339.9 418,332.1"/>
<path class="fc-edge" d="M595,336 L610,336"/><polygon class="fc-head" points="610,336 603,339.9 603,332.1"/>
<g class="fc-node start"><rect x="15" y="16" width="160" height="32" rx="16"/><text x="95" y="36" text-anchor="middle">Quartz sand</text></g>
<g class="fc-node"><rect x="200" y="10" width="170" height="44" rx="6"/><text x="285" y="30" text-anchor="middle">Polysilicon</text><text class="sub" x="285" y="43" text-anchor="middle">99.9999999 % pure</text></g>
<g class="fc-node"><rect x="390" y="10" width="170" height="44" rx="6"/><text x="475" y="30" text-anchor="middle">Single-crystal ingot</text><text class="sub" x="475" y="43" text-anchor="middle">Czochralski pull</text></g>
<g class="fc-node"><rect x="580" y="10" width="200" height="44" rx="6"/><text x="680" y="30" text-anchor="middle">Wafers</text><text class="sub" x="680" y="43" text-anchor="middle">slice, polish to &lt; 1 nm roughness</text></g>
<g class="fc-node k-cyan"><rect x="35" y="118" width="170" height="32" rx="6"/><text x="120" y="138" text-anchor="middle">Deposit film</text></g>
<g class="fc-node k-cyan"><rect x="225" y="118" width="170" height="32" rx="6"/><text x="310" y="138" text-anchor="middle">Coat photoresist</text></g>
<g class="fc-node k-purple"><rect x="410" y="112" width="180" height="44" rx="6"/><text x="500" y="132" text-anchor="middle">Expose (DUV / EUV)</text><text class="sub" x="500" y="145" text-anchor="middle">through the mask (reticle)</text></g>
<g class="fc-node k-cyan"><rect x="605" y="118" width="170" height="32" rx="6"/><text x="690" y="138" text-anchor="middle">Develop</text></g>
<g class="fc-node k-cyan"><rect x="605" y="210" width="170" height="32" rx="6"/><text x="690" y="230" text-anchor="middle">Etch</text></g>
<g class="fc-node k-cyan"><rect x="410" y="210" width="180" height="32" rx="6"/><text x="500" y="230" text-anchor="middle">Implant / strip resist</text></g>
<g class="fc-node k-cyan"><rect x="225" y="210" width="170" height="32" rx="6"/><text x="310" y="230" text-anchor="middle">CMP &amp; clean</text></g>
<g class="fc-node k-yellow"><rect x="35" y="210" width="170" height="32" rx="6"/><text x="120" y="230" text-anchor="middle">Inspect &amp; measure</text></g>
<g class="fc-node k-orange"><rect x="25" y="320" width="190" height="32" rx="6"/><text x="120" y="340" text-anchor="middle">Wafer sort (probe test)</text></g>
<g class="fc-node k-orange"><rect x="255" y="320" width="150" height="32" rx="6"/><text x="330" y="340" text-anchor="middle">Dice</text></g>
<a href="./appendix_e_packaging.html"><g class="fc-node k-orange"><rect x="425" y="314" width="170" height="44" rx="6"/><text x="510" y="334" text-anchor="middle">Package</text><text class="sub" x="510" y="347" text-anchor="middle">flip-chip, CoWoS, HBM</text></g></a>
<g class="fc-node leaf k-green"><rect x="610" y="320" width="180" height="32" rx="6"/><text x="700" y="340" text-anchor="middle">Final test &amp; binning</text></g>
<text class="fc-elabel" x="405" y="184" text-anchor="middle">repeated for every mask layer</text>
</svg>
<div class="fc-caption">The ring in the middle runs once per mask layer, dozens of times for a leading-edge logic wafer, which is why a wafer spends roughly three months in the fab.</div>
</div>
<!-- /flowchart:fab_loop -->

---

## From Sand to Wafer: The Starting Material

### Silicon Feedstock

Silicon is element 14. In nature it is never found pure — it bonds greedily with oxygen to form silicon dioxide (SiO₂), the main constituent of beach sand. The first step is breaking those bonds.

**Metallurgical-grade silicon** (MG-Si, ~98 % pure) is made by reducing quartzite (high-purity SiO₂) with carbon in an electric-arc furnace at ~1900 °C:

$$\text{SiO}_2 + 2\,\text{C} \;\longrightarrow\; \text{Si} + 2\,\text{CO}$$

That 98 % purity is nowhere near good enough. A single misplaced dopant atom per trillion silicon atoms can ruin a transistor. The chip industry needs **electronic-grade silicon** at 9N to 11N purity — that is, 99.999999999 % pure — one of the purest substances on Earth.

The route to 11N purity is the **Siemens process**: MG-Si is converted to trichlorosilane gas (SiHCl₃), fractionally distilled to remove impurities, then decomposed back to ultra-pure silicon on heated rods at ~1100 °C. The result is a polycrystalline silicon rod called **polysilicon** (or poly-Si), which is the feedstock for crystal growth.

### Czochralski Crystal Growth

Polysilicon is pure, but it is *polycrystalline* — a random mass of tiny crystal grains. Transistors need *single-crystal* silicon: one continuous lattice with no grain boundaries to scatter charge carriers or trap impurities.

The **Czochralski (CZ) process** (invented by Jan Czochralski in 1916; adapted for silicon in the 1950s) solves this elegantly:

<div class="diagram">
<div class="diagram-title">Czochralski Crystal Growth</div>
<div class="flow">
  <div class="flow-node accent wide">Melt polysilicon in a pure quartz crucible at ~1415 °C (just above Si melting point)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Dip a small single-crystal seed into the melt surface — crystal lattice acts as a template</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">Slowly pull seed upward (~1 mm/min) while rotating; silicon solidifies onto the seed, extending its lattice</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Control pull rate and temperature to maintain a target diameter (300 mm for HVM)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Result: a cylindrical single-crystal silicon ingot, ~2 m tall, ~300 mm in diameter, ~250–300 kg</div>
</div>
</div>

A small amount of dopant (phosphorus for n-type, boron for p-type substrate) is added to the melt to establish the base conductivity of the wafer. The resulting ingot is a **single crystal**: every atom sits in its correct lattice position across the entire cylinder.

### Wafer Slicing, Grinding, and Polishing

The cylindrical ingot is trimmed, then sliced into wafers using a **diamond wire saw** or inner-diameter saw, producing discs roughly **775 µm thick** (a fraction under a millimeter). Each cut introduces damage that must be repaired:

1. **Lapping** — both surfaces ground flat with abrasive slurry, removing saw damage. Thickness variation reduced to ~2 µm.
2. **Chemical etching** — removes residual mechanical damage and contamination.
3. **CMP (Chemical-Mechanical Planarization)** — the final polish. An abrasive slurry (colloidal silica, pH ~10) removes material mechanically while the chemistry softens surface peaks. Wafers emerge mirror-smooth to within ~0.1 nm RMS roughness — flat enough that a 1 mm height variation across a 300 mm wafer would be like a millimeter-tall bump across a continent spanning many hundreds of kilometers.

| Parameter | Value |
|-----------|-------|
| Wafer diameter | 300 mm (HVM); 200 mm (mature/specialty) |
| Wafer thickness | ~775 µm |
| Final surface roughness | < 0.1 nm RMS |
| Wafers per ingot | ~1,000–2,000 (a ~1.7 m body at ~1 mm per wafer including saw kerf) |
| Wafer cost (300 mm, polished) | ~$100–$200 per blank wafer |

---

## The Fabrication Environment: The Cleanroom

The fab is built around the **cleanroom** — an environment of astonishing cleanliness. A single dust particle (diameter ~0.5 µm) landing on a wafer during patterning can destroy a die. A modern leading-edge cleanroom achieves **Class 1**: fewer than 1 particle ≥ 0.1 µm per cubic foot of air. For context, a hospital operating theater is roughly Class 1000 (1000 particles per cubic foot). Normal air is ~1,000,000 particles per cubic foot.

Maintaining this requires:
- Massive HEPA and ULPA filter systems recirculating air constantly (the air turns over every 30–60 seconds)
- Positive air pressure to prevent outside air infiltration
- Wafer-processing robots operating in sealed FOUP (Front-Opening Unified Pod) cassettes — wafers barely see open air
- Workers wearing full-body "bunny suits" to shed zero contamination

Temperature (±0.01 °C), humidity (±0.5 %), and vibration are also tightly controlled: a photolithography tool requires a vibration-isolated table because nanometer-scale exposures cannot tolerate any floor vibration.

---

## The Patterning Loop: Building Layers One at a Time

Fabrication is an **iterative process**. A single "layer" of the chip requires executing a cycle of steps: deposit material, coat with photoresist, expose the pattern, develop the resist, etch or implant, then strip the resist and clean. Repeat ~60–100 times for a modern chip. The transistors themselves are built in the **front-end-of-line (FEOL)** steps; the metal wiring is added in **back-end-of-line (BEOL)** steps.

<div class="diagram">
<div class="diagram-title">The Patterning Loop (one layer)</div>
<div class="flow">
  <div class="flow-node accent wide">1. Deposition — grow or deposit the target material (oxide, metal, polysilicon, etc.)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">2. Coat — spin photoresist onto wafer surface (~30–100 nm thick)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">3. Expose — project circuit pattern through a mask/reticle with UV or EUV light</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">4. Develop — rinse with developer solution; exposed (or unexposed) resist dissolves away</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">5. Etch or implant — remove material or implant dopant ions where resist is absent</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">6. Strip &amp; clean — remove resist, clean wafer; inspect &amp; metrology</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node teal wide">7. CMP (if needed) — planarize the surface before the next layer</div>
</div>
</div>

Let's walk through each step in detail.

---

## Photolithography: Printing at the Nanoscale

### The Physics of Printing With Light

Photolithography works like a camera in reverse: instead of light reflecting *off* a scene onto a sensor, light shines *through* a **mask** (reticle) and projects a shrunken image of the circuit pattern *onto* the wafer.

The minimum feature size a lens can resolve is bounded by the **Rayleigh criterion**:

$$\text{Critical Dimension (CD)} = k_1 \cdot \frac{\lambda}{\text{NA}}$$

where:
- $\lambda$ = wavelength of the light
- NA = numerical aperture of the objective lens
- $k_1$ = a process-dependent constant (~0.25–0.4 in production)

**To print smaller features, you need shorter wavelengths or larger NA.** This simple equation has driven the entire history of semiconductor lithography.

### The Wavelength Journey: From 436 nm to 13.5 nm

| Era | Wavelength | Light Source | Approx. Minimum Feature |
|-----|-----------|-------------|------------------------|
| g-line (1980s) | 436 nm | Mercury lamp | ~500 nm |
| i-line (1990s) | 365 nm | Mercury lamp | ~250 nm |
| KrF (late 1990s–2000s) | 248 nm | Krypton fluoride excimer laser | ~130 nm |
| ArF dry (2000s) | 193 nm | Argon fluoride excimer laser | ~65 nm |
| ArF immersion (2007–present) | 193 nm + water | ArF + water between lens and wafer | ~38 nm (single), ~10 nm (multi-patterning) |
| EUV (2019–present) | 13.5 nm | Laser-produced plasma | ~12–15 nm (single), ~5–8 nm (multi) |

### 193 nm Immersion and Multi-Patterning

The transition from dry to **immersion lithography** in 2007 was a key inflection point. Water (refractive index ~1.44) fills the gap between the final lens element and the wafer surface, effectively shortening the wavelength in the medium by 1/n: the 193 nm light behaves like 134 nm light. NA values above 1.0 (impossible in air) become achievable; leading tools reach NA ~1.35.

But even immersion with 193 nm cannot print the 5–10 nm features needed for modern nodes by itself. The solution is **multi-patterning**: print the same layer in two, three, or four separate exposures, each offset, so that features interleave to achieve a pitch that no single shot could resolve.

- **LELE (Litho-Etch-Litho-Etch)**: two separate litho steps with two masks to pattern one layer
- **SAQP (Self-Aligned Quadruple Patterning)**: uses spacer layers grown around an initial pattern to define 4× the pitch of the original, without additional masks

Multi-patterning works, but at a steep cost: more process steps, more masks, more cycle time, tighter overlay requirements, and higher error accumulation. A single layer that once took one exposure now takes 4–8 process steps.

### EUV: The $200 Million Machine That Changed Everything

**Extreme Ultraviolet (EUV) lithography** uses light at **13.5 nm** wavelength — nearly 15× shorter than ArF. At this wavelength, almost every material absorbs light (air itself is opaque to EUV), which is what makes EUV so difficult:

<div class="diagram">
<div class="diagram-title">Why EUV Is an Engineering Miracle</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">The EUV Source</div>
    <div class="card-desc">Tiny tin droplets (30 µm diameter) fall at 50,000 droplets/second; a 20 kW CO₂ laser hits each droplet twice (pre-pulse to flatten it, main pulse to vaporize it) creating a plasma at ~200,000 K that emits EUV. Conversion efficiency: ~5 %. Source power at resist: ~250 W.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">All-Reflective Optics</div>
    <div class="card-desc">You cannot use refractive lenses for EUV — glass absorbs it. All optical elements are mirrors with 40–50 alternating Mo/Si layers (each pair ~6.8 nm thick) acting as Bragg reflectors. Each mirror reflects ~70 % of EUV; a 6-mirror system passes ~7 % of source light to the wafer.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Vacuum Environment</div>
    <div class="card-desc">The entire optical path must be in high vacuum (~10⁻⁶ mbar). The tool is roughly the size of a school bus — ~150 tonnes. A single tool costs ~$200M; the NA = 0.55 "High-NA" EXE:5000 series costs about $380M. Intel plans to use High-NA first, on its 14A node (2027–28); TSMC has said it does not need it through 2029.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Mask Complexity</div>
    <div class="card-desc">EUV masks are reflective (not transmissive), requiring special handling and pellicle development (a thin membrane to protect from particles). A single EUV mask costs ~$300K–$500K.</div>
  </div>
</div>
</div>

**The ASML monopoly**: ASML in Veldhoven, Netherlands, is the only company in the world that manufactures EUV scanners. This was not inevitable — it emerged from decades of sustained investment and the consolidation of critical subsystems (laser source from Trumpf/Cymer, optics from Zeiss). Today there is no substitute. A country's access to leading-edge fabs is gated entirely by access to ASML's EUV machines. (See [Appendix B](./appendix_b_foundry_ecosystem.md) and [Chapter 35](./35_where_the_industry_is_headed.md) for the geopolitical implications.)

---

## Etch

After exposure and development, the photoresist leaves a patterned template. **Etching** transfers that pattern into the underlying material.

**Wet etching** uses liquid chemicals (e.g., HF to etch SiO₂). It is isotropic — it removes material equally in all directions — so it undercuts below the resist mask, which is acceptable for coarse features but unusable at sub-100 nm scales.

**Dry etching (plasma etching / RIE)** is the modern standard. An RF-powered plasma generates reactive ions (e.g., fluorine-based for silicon etching) and accelerates them vertically toward the wafer:

```text
     Plasma (F*, CF₃+, etc.)
     ↓  ↓  ↓  ↓  ↓  ↓  ↓
     ■■■■■■■■■■■■■■■■■■   ← photoresist (masks target material)
     ───────────────────
          │           │
     ○○○○○○○○○○○○○○○○○○   ← material to etch (e.g., SiO₂)
     ═══════════════════   ← underlying silicon (etch stop)
```

The result is **anisotropic** etching: deep, vertical sidewalls with minimal undercutting. Modern **Atomic Layer Etching (ALE)** removes material one monolayer at a time with atomic precision.

---

## Ion Implantation: Doping on Demand

As introduced in [Chapter 1](./01_what_is_a_chip.md), transistors require precisely doped regions: n-type wells, p-type wells, source/drain regions, halo doping. Rather than relying on diffusion (the old method, impossible to control at nanometer scales), modern fabs use **ion implantation**:

1. An ion source ionizes the dopant gas (AsH₃ for arsenic, BF₃ for boron, PH₃ for phosphorus)
2. Ions are accelerated to a controlled energy (typically 1 keV–1 MeV)
3. A mass-spectrometer magnet selects only the desired ion species
4. Ions bombard the wafer surface, embedding at a depth determined by energy
5. A post-implant **anneal** (rapid thermal process, ~1000 °C for ~1 second) repairs crystal damage and activates dopants by moving them to substitutional lattice sites

The dose is controlled to within ~0.5 % and lateral implant profile is adjusted by tilt angle. This is how every transistor's source, drain, and well regions are defined with near-atomic precision.

---

## Deposition: Building Material Up

The complementary operation to etching is **deposition** — growing or placing new material on the wafer surface. Three major techniques are used at different points in the flow:

| Technique | Abbreviation | How It Works | Typical Use |
|-----------|-------------|-------------|-------------|
| Chemical Vapor Deposition | CVD | Reactive gases decompose/react on hot wafer surface | Gate oxides, polysilicon, nitrides, inter-layer dielectrics |
| Physical Vapor Deposition | PVD (sputtering) | Target material bombarded with ions; atoms sputter onto wafer | Metal films (tungsten plugs, copper barriers, aluminum) |
| Atomic Layer Deposition | ALD | Alternating self-limiting surface reactions; one monolayer per cycle | Ultra-thin high-k gate dielectrics (~0.5–2 nm), barrier layers |

**ALD** is critical at advanced nodes because high-k dielectrics (hafnium oxide, HfO₂, replacing SiO₂ for the gate insulator since ~45 nm) must be deposited to sub-nm precision. A single ALD cycle deposits ~0.1 nm; building a 1.5 nm gate oxide takes ~15 cycles, each requiring precise gas pulsing and purging.

---

## CMP: Planarization Between Layers

After each deposition-etch cycle, the wafer surface is no longer flat — it has topology from the patterned features below. **Chemical-Mechanical Planarization (CMP)** grinds the surface flat again.

A rotating polishing pad, abrasive slurry, and downforce combine mechanical abrasion with chemical softening (the chemistry weakens surface bonds so the abrasive can remove them). The wafer is pressed against the pad face-down with ~30–50 kPa pressure. Material removal rates are 100–300 nm/min; endpoint is detected by monitoring the motor current (a change signals reaching a different material layer).

Without CMP between layers, topology from lower layers would make it impossible to focus a lithography tool accurately on higher layers — the depth of focus of an EUV scanner is only ~60–100 nm.

---

## Transistor Structures: The FEOL Evolution

The patterning loop is used to build transistors in the **Front-End-of-Line (FEOL)** — the first 20–40 layers that create the actual devices.

### Planar MOSFET (pre-2011)

The original device described in Chapter 1: source, drain, and channel all in the silicon surface plane; gate sits on top separated by a thin oxide. Gate length shrank from micrometers down to ~22 nm. Below ~20 nm, leakage became unmanageable because the gate's electric field could no longer fully control the thin channel.

### FinFET (2011–present, ~22 nm–5 nm)

Intel (2011, 22 nm), then TSMC and Samsung, replaced the flat channel with a **vertical fin** of silicon rising above the substrate. The gate wraps around three sides of the fin, giving much better electrostatic control.

```text
         Gate (wraps 3 sides)
         │ ─────────────── │
         │ ▐  n+       n+▌ │
         │ ▐   FIN (p) ▌  │
         │ ▐           ▌  │
         └───────────────┘
              │ buried oxide │
              └──────────────┘
```

FinFETs reduced leakage by ~5× vs planar at the same gate length, and enabled continued scaling to 5 nm. The fin width at 5 nm is ~5–7 nm — just 20–30 atomic diameters.

### GAA Nanosheet (2022–present, ≤ 3 nm)

As fins became too narrow to maintain adequate drive current, the industry transitioned to **Gate-All-Around (GAA) nanosheets** (Samsung Foundry at 3 nm in 2022; Intel 18A "RibbonFET" and TSMC N2 both entering volume in late 2025). Instead of one fin, multiple horizontal silicon nanosheets (~5 nm thick, ~20–30 nm wide) are stacked vertically; the gate wraps entirely around each sheet.

```text
      Gate electrode
      ┌─────────────────┐
      │ ┌─────────────┐ │  ← nanosheet 3 (~5 nm thick)
      │ └─────────────┘ │
      │ ┌─────────────┐ │  ← nanosheet 2
      │ └─────────────┘ │
      │ ┌─────────────┐ │  ← nanosheet 1
      │ └─────────────┘ │
      └─────────────────┘
      source ─────── drain
```

Gate-all-around provides full 360° gate control, dramatically reducing short-channel effects. Nanosheet dimensions can be tuned to trade off drive current vs. power — a degree of freedom unavailable with FinFET.

**CFET (Complementary FET)** is the next evolution on the roadmap: NMOS and PMOS nanosheets stacked vertically on top of each other, roughly halving the footprint of a CMOS cell. TSMC, Intel and imec have shown working CFET devices; volume production is not expected before the early 2030s.

<div class="diagram">
<div class="diagram-title">Transistor Structure Evolution</div>
<div class="flow-h">
  <div class="flow-node accent">Planar<br/><small>≥ 22 nm<br/>Gate on top</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">FinFET<br/><small>22–5 nm<br/>Gate 3 sides</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">GAA Nanosheet<br/><small>3–2 nm<br/>Gate 4 sides</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">CFET<br/><small>≤ 2 nm<br/>N+P stacked</small></div>
</div>
</div>

---

## Back-End-of-Line (BEOL): The Metal Wiring

After transistors are built, they must be wired together. The **Back-End-of-Line (BEOL)** adds 10–20 layers of metal interconnect above the transistors. At advanced nodes these layers are made of copper (lower resistivity than aluminum) deposited by **damascene** plating:

1. Deposit inter-layer dielectric (ILD)
2. Pattern and etch trenches and vias in the dielectric
3. Deposit thin barrier metal (TaN/Ta) to prevent copper diffusion
4. Electroplate copper to fill trenches
5. CMP to remove excess copper and planarize

Lower metal layers (M0–M4) use the finest pitches (~20–30 nm metal pitch at N3/N2-class nodes, so lines ~10–15 nm wide) for local transistor-to-transistor wiring; upper layers (M8–M15) use coarser pitch (~100–500 nm) for power/clock distribution and long-range signals. The top-most layers are often aluminum for external connections (bump pads).

**Backside power delivery** is the biggest BEOL change in decades. Until now, power and signals shared the same metal stack above the transistors, so power rails stole routing space and long, thin power paths caused voltage (IR) drop. Intel's 18A (**PowerVia**, 2025) and TSMC's A16 (**Super Power Rail**, expected 2027) move the power network to the *back* of the wafer: the wafer is flipped, thinned to a few hundred nanometres above the transistors, and a second wiring stack for power is built there. Signal routing gets the whole front side, and power reaches the transistors through short, fat wires.

---

## Wafer Test, Dicing, and Packaging

### Wafer-Level Test (Probe)

After all layers are complete, each die on the wafer is electrically tested by touching fine probes to its pads. This **wafer probe** test identifies dies that are clearly defective (no power, gross shorts, basic functional failures). Dies that fail are inked or flagged in a wafer map. This avoids wasting packaging cost on known-bad dies.

### Dicing

The wafer is mounted on a tape frame and cut into individual dies using:
- **Diamond blade dicing**: a spinning blade, ~200 µm kerf width
- **Laser dicing**: cleaner edges, smaller kerf (~10 µm), preferred at advanced nodes where die-edge cracking is a reliability risk

A 300 mm wafer holds roughly **63** candidate dies of H100 size (814 mm²), using the standard estimate πr²/A − πd/√(2A) to subtract the partial dies at the edge.

### Packaging

The individual die is placed into a package — the structure you actually plug into a board. Packaging (detailed in [Appendix E](./appendix_e_packaging.md)) accomplishes:
- **Mechanical protection** of the fragile die
- **Electrical interconnect** from die microbumps to board-level pads/BGA balls
- **Thermal spreading** — paths for heat to escape to a heatsink

For AI accelerators, advanced packaging is critical: NVIDIA's A100 and H100 use **CoWoS-S (Chip-on-Wafer-on-Substrate)** to place HBM memory and the GPU die side-by-side on a silicon interposer (3.35 TB/s on H100); the two-die B200 moved to CoWoS-L, which embeds small silicon bridges in a larger organic interposer (8 TB/s) (link: [Chapter 9](./09_memory_types.md) and [Chapter 35](./35_where_the_industry_is_headed.md)).

### Final Test and Burn-In

After packaging, chips go through rigorous **final test**: full functional validation at multiple voltages and temperatures. High-performance parts are tested at maximum speed; the distribution of passing frequencies determines which product bin the chip becomes.

**Burn-in** (hours at elevated temperature and voltage) screens for early-life failures ("infant mortality" per the bathtub reliability curve — see [Chapter 13](./13_reliability_and_fault_tolerance.md)).

---

## Yield: The Economics of the Die

### Why Yield Is Everything

A fab makes money only from working chips. **Yield** — the fraction of dies on a wafer that pass test — is the master variable of fab economics. A new process node at the start of production might yield 20–30 % on a complex chip; a mature process might yield 80–90 %+ on the same design. The entire cost structure of a $10,000 GPU depends on this number.

### The Defect Density Model

During fabrication, random defects (particles, impurities, local process variations) are deposited across the wafer. If a defect lands within a die's active area, that die fails. Assuming a random Poisson distribution of defects with density $D$ (defects/cm²):

**Simple Poisson / exponential model:**

$$Y \approx e^{-D \cdot A}$$

where $A$ is die area in cm². This is the simplest closed-form estimate.

A more empirically accurate model is the **negative-binomial (Murphy) model**, which accounts for defect clustering:

$$Y = \left(1 + \frac{D \cdot A}{\alpha}\right)^{-\alpha}$$

where $\alpha$ is a clustering parameter (~3–5 in practice). As $\alpha \to \infty$, this converges to the Poisson model; real defects cluster, so the Murphy model generally predicts higher yield (defects in clusters hit the same die rather than distributing uniformly).

### Yield vs Die Area: The Reticle-Limit Problem

The Poisson relationship has a devastating implication: **larger dies yield much worse**.

```python
import numpy as np

def yield_poisson(D: float, A: float) -> float:
    """Simple Poisson yield model. D in defects/cm², A in cm²."""
    return np.exp(-D * A)

def yield_negbin(D: float, A: float, alpha: float = 4.0) -> float:
    """Negative-binomial (Stapper) model; alpha = defect clustering."""
    return (1 + D * A / alpha) ** (-alpha)

def yield_murphy(D: float, A: float) -> float:
    """Murphy's model: defect density varies (triangular) across wafers."""
    x = D * A
    return ((1 - np.exp(-x)) / x) ** 2

# Representative values for a mature leading-edge process
D = 0.10  # defects/cm² (mature N3 process, roughly 0.08–0.15)
areas = np.linspace(0.01, 8.0, 500)  # cm², 1 mm² to 800 mm²

Y_poisson = yield_poisson(D, areas)
Y_negbin  = yield_negbin(D, areas)

# Reference points
for label, A in [("RTX 4090 (AD102, 609 mm²)", 6.09),
                 ("H100 (GH100, 814 mm²)",    8.14),
                 ("Apple M2 (251 mm²)",        2.51)]:
    print(f"{label:28s} A={A:.2f} cm²  Poisson {yield_poisson(D, A):.1%}"
          f"  neg-binomial {yield_negbin(D, A):.1%}  Murphy {yield_murphy(D, A):.1%}")
```

Running this with $D = 0.10$ defects/cm² and $\alpha = 4$ yields approximately:

| Die | Area | Poisson | Negative binomial (α = 4) | Murphy |
|-----|------|:------------:|:------------:|:------------:|
| Apple M2 | 251 mm² (2.51 cm²) | 77.8 % | 78.4 % | 78.2 % |
| RTX 4090 AD102 | 609 mm² (6.09 cm²) | 54.4 % | 56.7 % | 56.1 % |
| H100 GH100 | 814 mm² (8.14 cm²) | 44.3 % | 47.7 % | 46.8 % |

At this defect density the three models agree within a few points; clustering only starts to matter when $D \cdot A$ is large (big dies on an immature process), where the Poisson model is too pessimistic. The die-size penalty itself is model-independent: tripling the area from M2 to H100 cuts yield from ~78% to ~45%.

### Try It: Defects on a Wafer

The lab scatters random defects on a 300 mm wafer and scraps every die that catches one. Start with an 814 mm² die at 0.1 defects/cm², then switch to ~200 mm² chiplets at the same defect density: the fraction of good silicon jumps, which is the whole economic argument for chiplets. Then raise the defect density to an early-ramp 0.3/cm² and watch the big die collapse.

<div class="lab" data-lab="wafer"><noscript><div class="lab-noscript">This lab needs JavaScript. It is also on the <a href="./labs.html">Interactive Labs</a> page.</div></noscript></div>

> These are illustrative calculations. Real yield is also affected by systematic defects, edge exclusion, and test overkill — but the qualitative trend is real and consequential.

### The Reticle Limit

A **reticle** (mask) in a step-and-scan lithography tool can only expose a field of approximately **26 × 33 mm** = 858 mm². A single die cannot exceed this limit — there is no way to expose a larger area in one shot without stitching (impractical for standard chips). The **reticle limit** is therefore the hard ceiling on single-die area, and thus on single-die transistor count.

This is the primary driver behind **chiplets** (explored in [Chapter 35](./35_where_the_industry_is_headed.md)): instead of one huge die that yields poorly, split the design into multiple smaller dies ("chiplets") that yield well individually, then connect them in a package.

### Redundancy and Binning

Two techniques reclaim value from partially-working dies:

**Redundancy**: Build more functional units than you need. GPU designers include extra SMs (streaming multiprocessors); if a few are defective at test, they are laser-fused or e-fused off. The full GH100 die has 144 SMs; no shipping H100 enables all of them. H100 SXM5 ships with 132 and H100 PCIe with 114, so a die with several defective SMs is still a sellable product. That is why one die becomes several SKUs, and why effective yield on huge GPUs is far better than the simple models above suggest.

**Binning**: Dies are sorted by the maximum frequency/voltage at which they pass test. Fast dies become the flagship product; slower dies become the lower-tier SKU. A "binning cascade" squeezes revenue from nearly every die on the wafer.

<div class="diagram">
<div class="diagram-title">From Wafer to Binned Products</div>
<div class="flow-h">
  <div class="flow-node accent">Full die<br/><small>All units pass</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">Fused die<br/><small>1–2 units off</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">Binned lower<br/><small>Lower freq/TDP</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node red">Scrap<br/><small>Too many defects</small></div>
</div>
</div>

---

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>A wafer costs the same whether it holds 63 big dies or 300 small ones. If an H100-size die yields ~45% and a quarter-size chiplet ~80%, how much does the chiplet design save in silicon per working GPU, and what new costs (packaging, die-to-die links, power) does it have to pay back?</p>
</div>

## Why This Matters for Model Optimization

The fabrication reality directly sets the parameters of the hardware you train on:

**Supply constraints**: An H100-size die gives ~63 candidates per 300 mm wafer and, with spare SMs, perhaps 40–55 sellable parts. Front-end wafers were rarely the bottleneck in 2023–25; **CoWoS packaging capacity** was, because every HBM-equipped accelerator needs an interposer several reticles in size, and TSMC has been roughly doubling that capacity each year to keep up. This is not only a logistics problem; it is a physics and economics problem. New fabs take 3–5 years and $20B+ to build (see [Appendix B](./appendix_b_foundry_ecosystem.md)).

**Die area caps compute**: The reticle limit (~858 mm²) means a single-die GPU cannot simply grow to have more SM cores. Every compute scaling decision after ~2020 — chiplets (AMD), multi-die packaging (NVIDIA B200 = two reticle-sized Blackwell dies joined by a 10 TB/s link), 3D stacking (HBM) — is a direct consequence of the fab geometry explored in this appendix.

**Process node determines efficiency**: Moving from 7 nm to 5 nm to 3 nm to 2 nm has cut power at equal speed by roughly 25–35 % per node (TSMC's own figures) (from density gains and lower $V_{dd}$). That same $E = \frac{1}{2}CV_{dd}^2$ from [Chapter 1](./01_what_is_a_chip.md) applies here: smaller transistors have lower capacitance. For a 175B-parameter model, training throughput is not just a software problem — it is a function of what process node your accelerator was built at.

**Reliability and redundancy**: ECC memory and the ECC bits in HBM exist precisely because a real wafer has real defects, and [Chapter 13](./13_reliability_and_fault_tolerance.md) shows how they map to training stability.

---

## Key Takeaways

- Silicon starts as SiO₂ (sand); it becomes electronic-grade 11N-purity polysilicon, then a single-crystal 300 mm ingot via Czochralski growth, then a polished wafer that is the substrate for all fabrication.
- The core fabrication cycle — deposit, coat, expose (lithography), develop, etch or implant, strip, planarize — repeats 60–100 times per chip to build transistors in the FEOL and metal wiring in the BEOL.
- Photolithography resolution is bounded by $\text{CD} = k_1 \lambda / \text{NA}$; the industry has pushed from 436 nm g-line to 13.5 nm EUV, and ASML holds a monopoly on EUV scanners — a geopolitical chokepoint.
- Transistor structures evolved from planar MOSFET → FinFET → GAA nanosheet → (coming) CFET to maintain gate control as dimensions shrank below 20 nm.
- Yield follows approximately $Y \approx e^{-DA}$ (or its clustered variants, which differ by a few points at mature defect densities); because yield degrades strongly with die area, the reticle-limited ceiling (~858 mm²) is a hard physical constraint that forces chiplet architectures.
- Redundancy (extra units fused off) and binning (speed sorting) extract commercial value from every die on the wafer, giving rise to product families from a single silicon mask set.
- For ML practitioners: supply scarcity, die-area limits, and per-node efficiency gains are not marketing or supply-chain stories — they are direct consequences of the physics and economics described in this appendix.

## Check Yourself

<details class="check">
<summary>Compute the Poisson yield of a 600 mm² die at D₀ = 0.2 defects/cm². How does it compare with four 150 mm² chiplets?</summary>
<p>Y = e^(−0.2 × 6) = e^(−1.2) ≈ 30%. Each 1.5 cm² chiplet yields e^(−0.3) ≈ 74%; since bad chiplets are discarded individually, ~74% of chiplet silicon is usable versus ~30% for the monolithic die.</p>
</details>

<details class="check">
<summary>Using CD = k₁ λ / NA, what half-pitch can a 0.33-NA EUV scanner print at k₁ = 0.3, and what does High-NA (0.55) change?</summary>
<p>0.3 × 13.5 / 0.33 ≈ 12.3 nm; at NA 0.55 it is ≈ 7.4 nm, enough to replace some EUV double-patterning with single exposures.</p>
</details>

<details class="check">
<summary>Why does a leading-edge wafer spend about three months in the fab?</summary>
<p>Dozens of mask layers, each needing deposition, lithography, etch or implant, cleaning, CMP and inspection, add up to roughly 1,000 process steps with queueing between tools.</p>
</details>

<details class="check">
<summary>What problem does backside power delivery solve, and which nodes introduce it?</summary>
<p>It moves the power network under the transistors, freeing front-side metal for signals and cutting IR drop. Intel 18A (PowerVia, 2025) and TSMC A16 (Super Power Rail, expected 2027).</p>
</details>

<details class="check">
<summary>Why are there 144 SMs on the GH100 die but at most 132 enabled on any H100?</summary>
<p>Spare SMs let dies with a few defects still ship. Fusing off defective units turns a yield problem into a binning decision, which is how a reticle-sized die is economical at all.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| ASML EUV lithography systems | ASML | [asml.com/en/products/euv-lithography-systems](https://www.asml.com/en/products/euv-lithography-systems) | NXE and EXE (High-NA) scanners |
| Yield models for integrated circuits | Stapper, IBM J. Res. Dev. 1983 | [doi.org/10.1147/rd.274.0453](https://doi.org/10.1147/rd.274.0453) | Negative-binomial yield and clustering |
| Cost-Size Optima of Monolithic Integrated Circuits | Murphy, Proc. IEEE 1964 | [doi.org/10.1109/PROC.1964.3442](https://doi.org/10.1109/PROC.1964.3442) | Murphy yield model |
| Intel 18A (RibbonFET and PowerVia) | Intel | [intel.com/content/www/us/en/foundry/process/1…](https://www.intel.com/content/www/us/en/foundry/process/18a.html) | GAA plus backside power |
| TSMC A16 and N2 technology | TSMC | [tsmc.com/english/dedicatedFoundry/technology/…](https://www.tsmc.com/english/dedicatedFoundry/technology/logic/l_A16) | Super Power Rail; nanosheet node |
| NVIDIA H100 Tensor Core GPU Architecture | NVIDIA, 2022 | [resources.nvidia.com/en-us-tensor-core](https://resources.nvidia.com/en-us-tensor-core) | 144-SM GH100, 132 enabled on SXM5 |

---

**Next:** [Appendix B — The Foundry & Fabless Ecosystem](./appendix_b_foundry_ecosystem.md) — The industry structure behind who builds, equips and licenses the chips that run your models.

[← Back to Table of Contents](./README.md)
