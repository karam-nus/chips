---
title: "Appendix B — The Foundry & Fabless Ecosystem"
---

[← Back to Table of Contents](./README.md)

# Appendix B — The Foundry & Fabless Ecosystem

[Appendix A](./appendix_a_how_chips_are_made.md) described the physical alchemy of chip fabrication — Czochralski crystals, EUV scanners, patterning loops. But *who* does all that, for whom, with what tools, and at what cost? The answer is a tightly coupled global ecosystem that has no parallel in any other industry: a handful of companies at each layer of the stack, each holding near-monopoly leverage, operating on decade-long investment cycles, and increasingly entangled in national-security geopolitics.

For an ML practitioner, this matters directly: the price you pay for an H100, whether you can get one at all, and whether you are legally permitted to export it to another country are all determined not by NVIDIA's pricing department but by the ecosystem described in this appendix. The structural bottlenecks of the semiconductor supply chain are the structural bottlenecks of your training infrastructure.

> **The one-sentence version:** The chip industry is split into fabless designers (who draw the circuits), foundries (who manufacture them), EDA companies (who provide the software toolchain), IP licensors (who provide reusable sub-circuits), and equipment makers (who build the machines that fabs run) — with single companies holding irreplaceable positions at nearly every layer.

---

## The Core Split: Fabless, Foundry, and IDM

For most of the industry's first three decades, chip companies were **Integrated Device Manufacturers (IDMs)**: they designed and manufactured their own chips. Intel and Texas Instruments are the canonical IDMs. This required enormous capital expenditure — a leading-edge fab costs $20B+ — which made sense only for companies at sufficient scale.

The structural innovation that opened the modern era was the **fabless/foundry split**:

<div class="diagram">
<div class="diagram-title">Business Model Spectrum</div>
<div class="flow-h">
  <div class="flow-node accent">Fabless<br/><small>Designs only<br/>No fabs</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">Fab-lite<br/><small>Some fabs<br/>Outsources leading edge</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">IDM<br/><small>Designs + manufactures<br/>Internal fabs</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">Pure-play Foundry<br/><small>Manufactures only<br/>No own products</small></div>
</div>
</div>

**Morris Chang** founded TSMC in 1987 on the insight that dedicated, neutral foundries could serve many fabless customers simultaneously, and that foundry-scale utilization would amortize fab capital more efficiently than each company building its own. The model was spectacularly vindicated: today the fabless model dominates design of the most advanced chips (NVIDIA, AMD, Apple, Qualcomm, Broadcom, MediaTek are all fabless), while pure-play foundries capture the manufacturing.

The design/fab separation is explored further in [Chapter 33](./33_design_paradigm_wars.md).

---

## The Foundries

### TSMC: The Indispensable Manufacturer

Taiwan Semiconductor Manufacturing Company is, by any measure, the most important technology company most people have never heard of. It is the **dominant pure-play foundry** for leading-edge logic, with ~60 % of total foundry revenue and >90 % share of the most advanced nodes (≤ 5 nm).

| TSMC Node | Marketing Name | ~Transistor Density | Volume Production | Key Customers |
|-----------|---------------|:-------------------:|:-----------------:|---------------|
| N7 | 7 nm | ~91 MTr/mm² | 2018–present | NVIDIA Ampere, AMD Zen 2/3 |
| N5 | 5 nm | ~171 MTr/mm² | 2020–present | Apple A14/M1, NVIDIA H100 variants |
| N3 | 3 nm | ~292 MTr/mm² | 2022–present | Apple A17/M3, NVIDIA Blackwell |
| N2 | 2 nm | ~400+ MTr/mm² | 2025–present | Apple A19, AMD next-gen |
| A16 | 1.6 nm | ~500+ MTr/mm² (est.) | 2026 (planned) | — |

TSMC's key structural advantages:
- **Process yield leadership**: years of accumulated know-how give TSMC the best defect densities (lowest $D$ in the yield model from [Appendix A](./appendix_a_how_chips_are_made.md)).
- **Neutral intermediary**: TSMC does not compete with customers by selling its own chips. Samsung Foundry, by contrast, competes with Qualcomm in mobile processors while also manufacturing for Qualcomm — a structural tension.
- **Scale**: TSMC's Fab 18 (N3) in Tainan alone required ~$35B investment. Total annual capex exceeds $30–40B/year.
- **Geographic concentration**: ~90 % of leading-edge capacity sits in Taiwan — a geopolitical risk discussed below.

TSMC operates CoWoS advanced packaging for AI chips (CoWoS-S for H100-class; CoWoS-L for Blackwell B200), which is itself a bottleneck: CoWoS capacity has limited H100 and B200 supply independently of wafer production (see [Appendix E](./appendix_e_packaging.md)).

### Samsung Foundry

Samsung is the world's second leading-edge foundry and, uniquely, also the world's largest memory chip maker and a major consumer electronics company. Samsung's foundry process naming is different from TSMC's:

| Samsung Node | Year | Comparable TSMC | Notes |
|-------------|------|-----------------|-------|
| 7LPP | 2018 | N7 | First EUV in production |
| 5LPE | 2020 | N5 | Qualcomm Snapdragon 888 |
| 4nm | 2021 | N5P | Snapdragon 8 Gen 1 |
| 3GAE | 2022 | N3 | First GAA nanosheet in production |
| 2nm (SF2) | 2025 | N2 | — |

Samsung's challenge has been yield: on several advanced nodes Samsung lagged TSMC in defect density, leading key customers (Apple, AMD, NVIDIA) to shift volume entirely to TSMC. Samsung's internal chip divisions (Exynos, HBM memory) guarantee baseline utilization even in lean periods.

### Intel Foundry Services (IFS)

Intel is an IDM that has historically manufactured only for itself. **Intel Foundry Services** (IFS, rebranded from Intel Foundry in 2024) is Intel's effort to become a contract foundry:

- Intel 18A (targeted 2025): uses **RibbonFET** (Intel's GAA architecture) and **PowerVia** (backside power delivery), promising density competitive with TSMC N2.
- Major challenges: Intel has missed multiple node transition deadlines (10nm was years late, 7nm was cancelled and reformulated), denting customer confidence.
- Strategic importance: as the only leading-edge fab located in the United States with scale aspirations, IFS is a linchpin of U.S. industrial policy under the CHIPS and Science Act.
- Key customers (announced/committed): Amazon AWS, Qualcomm, Microsoft (Azure tiles), U.S. government programs.

### GlobalFoundries, SMIC, and Trailing-Edge Foundries

**GlobalFoundries** (GF) spun out of AMD in 2009 and abandoned the leading-edge race in 2018, stopping at 12/14 nm. It focuses on:
- Specialized nodes: RF CMOS (for wireless modems), silicon photonics (for transceivers), FD-SOI (for IoT/automotive low-power), aerospace-grade
- Mature nodes (28/40/65/130 nm) that are critically important for automotive, industrial, and defense applications

**SMIC (Semiconductor Manufacturing International Corporation)** is China's most advanced foundry, reaching approximately N+1 (roughly equivalent to 7 nm) in 2023 despite U.S. export controls on EUV tools. SMIC cannot advance to true 5 nm without EUV access.

**PSMC, UMC (Taiwan), Tower (Israel/US), DB HiTek (Korea)** and others serve the mature-node market (~180 nm to 28 nm), which, despite lacking glamour, is enormous: automotive ECUs, microcontrollers, power regulators, and sensors all run on mature nodes. The 2020–2022 semiconductor shortage demonstrated how supply constraints at mature nodes (auto-grade 28/40 nm) can halt production lines.

---

## The Fabless Companies: Designing Without Fabs

The fabless model enables companies to focus entirely on design, which now requires tens of billions of dollars in R&D but zero spent on building fabs. The top fabless companies are, by revenue, among the most profitable companies in the world.

<div class="diagram">
<div class="diagram-title">Major Fabless Companies by Segment</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">AI Accelerators</div>
    <div class="card-desc"><strong>NVIDIA</strong> (H100/B200/Blackwell), AMD (MI300), Google (TPU — internal ASIC), Amazon (Trainium/Inferentia), Microsoft (Maia)</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">CPUs & SoCs</div>
    <div class="card-desc"><strong>Apple</strong> (M-series, A-series), Qualcomm (Snapdragon), AMD (Ryzen/EPYC), ARM (designs, licenses to all others)</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Networking & Storage</div>
    <div class="card-desc"><strong>Broadcom</strong> (Tomahawk/Jericho switches, custom AI ASICs), Marvell (DPUs, SerDes), Micron/SK Hynix (HBM — technically IDMs)</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Mobile</div>
    <div class="card-desc">Qualcomm (Snapdragon), MediaTek (Dimensity), Apple (A-series). All fabless, all TSMC-dependent.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Hyperscaler Custom Silicon</div>
    <div class="card-desc">Google TPU, Amazon Trainium/Inferentia, Microsoft Maia, Meta MTIA — all fabless design teams, all on TSMC</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">AI Startups</div>
    <div class="card-desc">Cerebras (WSE, wafer-scale), Groq (LPU), Tenstorrent, SambaNova, d-Matrix — all fabless, various foundries</div>
  </div>
</div>
</div>

A key implication: **nearly all AI training and inference hardware flows through TSMC**. NVIDIA, AMD, Google, Amazon, Microsoft, Meta — their silicon supply chains converge at TSMC's Tainan fabs. This single-point dependency is why a TSMC disruption would cripple global AI infrastructure.

---

## EDA: The Software Without Which Chips Are Impossible

**Electronic Design Automation (EDA)** refers to the software tools used to design, simulate, verify, and implement chips. Without EDA, a chip with billions of transistors is simply impossible to design — a human could not route even 1 % of the interconnects manually.

The EDA market is effectively a **triopoly** (plus a few specialists):

| Company | Core Strengths | Key Products |
|---------|---------------|-------------|
| **Synopsys** | Synthesis, formal verification, DFT, IP | Design Compiler, VCS, Formality, IC Compiler, PrimePower, ARC/DesignWare IP |
| **Cadence** | Place-and-route, simulation, analog | Innovus, Virtuoso, Spectre, Xcelium, Tempus, Quantus |
| **Siemens EDA** (ex-Mentor) | DRC/LVS, emulation, PCB | Calibre (the industry-standard DRC/LVS), Veloce emulation, Xpedition PCB |
| **Ansys** | Thermal/signal-integrity simulation | RedHawk, Totem, SIwave |

Every advanced chip — every GPU, every TPU, every SoC — was designed using tools from this list. Synopsys and Cadence each generate ~$5–6B in annual revenue (FY2024); their software is licensed to every fabless company and IDM on Earth.

The entire [Chapter 31](./31_chip_design_flow.md) (Chip Design Flow from RTL to GDSII) is the story of using these tools. The EDA companies also maintain **Process Design Kits (PDKs)** in partnership with foundries — the files that encode TSMC's N3 or Samsung's SF3E rules, device models, and design constraints that every chip designer must use.

### Why EDA Matters for Geopolitics

EDA tools are U.S.-headquartered (Synopsys: Sunnyvale, Cadence: San Jose, Siemens EDA: Fremont/Munich). U.S. export controls (BIS Entity List, EAR) on EDA software restrict Chinese chip companies' access to leading-edge EDA flows — an upstream chokepoint that complements the ASML/EUV hardware restriction.

---

## IP Cores: The Building Blocks You License

Even with EDA tools and a foundry process, a chip design team does not implement every functional block from scratch. Complex, heavily-verified blocks — CPU cores, USB/PCIe controllers, display interfaces, security enclaves, memory PHYs — are licensed as **IP cores** (Intellectual Property).

**ARM Holdings** is the dominant IP licensor for processor cores:
- Designs CPU/GPU/NPU architectures (Cortex-A, Cortex-M, Mali GPU, Ethos NPU)
- Licenses the ISA (Instruction Set Architecture) to allow customers to design their own implementations (Apple's M-series are custom ARM ISA implementations)
- Does not fab anything; collects royalties on every shipped chip (~25B ARM-based chips shipped per year)
- RISC-V is the open alternative (no royalties), adopted by SiFive, Western Digital (storage), NVIDIA (embedded cores in GPUs), and Chinese companies seeking to escape ARM licensing constraints

**Synopsys DesignWare** provides thousands of verified IP blocks: PCIe Gen5/6 controllers, DDR5 PHY, USB4, Ethernet MAC/PHY, AES/RSA cryptographic cores, and more. Broadcom, NVIDIA, and Apple all license these blocks rather than implementing them internally.

**Imagination Technologies** licenses PowerVR GPU IP; **Cadence** and **Rambus** license memory interface IP. The economics: a complex verified IP block costs $1–10M to license, versus $50–200M+ to develop internally with comparable quality and schedule certainty.

---

## Equipment Makers: Who Builds the Machines That Build Chips

Fabs cannot operate without specialized semiconductor capital equipment. This layer has some of the most concentrated supply in the entire industry.

| Equipment Category | Leading Suppliers | Tool Examples |
|--------------------|------------------|---------------|
| Lithography | **ASML** (monopoly on EUV); Nikon, Canon (legacy DUV) | EUV NXE:3600D, TWINSCAN ArF immersion |
| Etch | **Lam Research**, Applied Materials, Tokyo Electron (TEL) | Kiyo, Centris, Versys |
| Deposition (CVD/ALD/PVD) | **Applied Materials (AMAT)**, Lam, TEL, ASM International | Centura, Vantage ALD, Polygon |
| CMP | **Applied Materials**, Ebara | Reflexion CMP, F-REX |
| Ion Implant | **Applied Materials**, Axcelis | VIISta HE, Purion |
| Metrology/Inspection | **KLA Corporation**, Applied Materials | Surfscan, eDR-7xxx, Archer overlay |
| Wafer Cleaning | **SCREEN Semiconductor** (Tokyo), LAM | SSL (Single-wafer spin clean) |

**ASML** deserves special attention (as introduced in [Appendix A](./appendix_a_how_chips_are_made.md)). It is literally the only company on Earth that makes EUV scanners. This positions ASML as the ultimate upstream bottleneck: no EUV scanner → no leading-edge node → no H100/B200/TPU. ASML's Veldhoven campus is one of the most security-sensitive industrial sites in the world. The Dutch government (at U.S. pressure) has blocked ASML from shipping EUV tools to China since 2019.

**KLA**, **Applied Materials**, and **Lam Research** collectively dominate process control, deposition, and etch — three of the other major equipment categories. All three are U.S.-headquartered, putting them under U.S. export-control jurisdiction.

---

## Materials and Substrates

Beyond equipment, advanced fabrication depends on hyper-specialized materials:

| Material | Use | Key Suppliers |
|----------|-----|---------------|
| Ultra-pure silicon wafers | Substrate | Shin-Etsu (Japan), SUMCO (Japan) — ~60 % global share combined |
| High-k dielectrics (HfO₂) | Gate insulator | Various precursor chemical companies |
| Low-k dielectrics | BEOL ILD | DuPont, Merck KGaA |
| Photoresists (EUV CAR) | EUV lithography | JSR, TOK, Shin-Etsu Chemical (Japan — ~90 % global share) |
| Process gases (NF₃, WF₆, SiH₄) | Etch, CVD | Air Products, Linde, SK Materials |
| Copper/cobalt/ruthenium | Interconnect metals | Various specialty chemical suppliers |
| Advanced packaging substrates | ABF substrates | **Ibiden, Shinko, AT&S** (extreme concentration in Japan/Austria) |

The **ABF (Ajinomoto Build-up Film) substrate** for advanced package interposers — used in every BGA-style GPU package — is dominated by Ibiden and Shinko Electric Industries of Japan. A shortage here in 2021 contributed to GPU supply constraints.

---

## The Supply Chain and Geopolitics

### Taiwan Concentration Risk

The semiconductor supply chain has optimized for efficiency over resilience for 30 years, producing extreme geographic concentration:

<div class="diagram">
<div class="diagram-title">Geographic Dependencies in the Leading-Edge Stack</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Taiwan</div>
    <div class="card-desc">~90 % of sub-5nm wafer capacity (TSMC). Disruption of Taiwan — by conflict, earthquake, or political instability — would halt AI accelerator production globally within months. There is no substitute with comparable scale or yield at leading edge.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Netherlands</div>
    <div class="card-desc">100 % of EUV scanner supply (ASML). ASML's Veldhoven campus is irreplaceable; no other company has successfully produced an EUV scanner. Even ASML's own supply chain stretches to Zeiss (optics, Germany) and Trumpf/Cymer (laser, Germany/USA).</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Japan</div>
    <div class="card-desc">~60 % of silicon wafer supply (Shin-Etsu, SUMCO), ~90 % of EUV photoresist supply (JSR, TOK, Shin-Etsu Chemical), dominant share of ABF packaging substrates (Ibiden, Shinko), major share of specialty process gases and chemicals.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">USA</div>
    <div class="card-desc">EDA software (Synopsys, Cadence, Siemens EDA HQ), major equipment (Applied Materials, Lam Research, KLA, Axcelis), IP licensing (ARM HQ is UK, but key revenue is IP-routed globally). Critical design talent.</div>
  </div>
</div>
</div>

### Export Controls and the AI Chip Chokepoint

Beginning in October 2022, the U.S. Bureau of Industry and Security (BIS) implemented sweeping export controls targeting China's ability to acquire and produce advanced AI chips:

- **Initial (October 2022)**: Export license required for chips with performance ≥ a certain compute-density threshold (effectively targeting H100/A100-class and above to Chinese end-users).
- **BIS October 2023 update**: Closed loopholes; added A800/H800 (NVIDIA's China-market workarounds) to the list; expanded to additional countries.
- **Ongoing updates (2024–2025)**: Further refinements, including controls on chip architecture features that could be used to exceed the threshold.

ASML was restricted from shipping EUV tools to China by the Dutch government in January 2019 (and DUV immersion tools from 2023). This makes it impossible for SMIC or any Chinese foundry to advance to true leading-edge nodes without a domestic EUV breakthrough.

The technical enforcement mechanism is clever: AI chips are measured by **Total Processing Performance (TPP)** and **Performance Density** thresholds, with an interconnect bandwidth provision. A chip that exceeds either threshold — measured in FLOPS/second for certain precisions — requires a license that is generally denied for Chinese military or advanced AI applications.

For ML practitioners: this directly determines which GPU models you can legally ship to, sell services from, or operate in different jurisdictions. A model trained on H100s may legally run inference only on chips approved for the target market.

### CHIPS and Science Act (2022)

The U.S. CHIPS and Science Act allocated $52.7B for semiconductor R&D and manufacturing incentives, with $39B in direct manufacturing grants:
- TSMC Arizona: $6.6B grant for two (later three) fabs in Phoenix, targeting N3/N2 production (operational 2025–2028)
- Intel Ohio and Arizona: ~$8.5B grant for IFS fabs
- Samsung Texas: $6.4B grant
- Micron New York: $6.1B grant for DRAM manufacturing
- GlobalFoundries New York/Vermont: $1.5B

The economics are stark: building a new leading-edge fab in the U.S. costs 40–50 % more than building in Taiwan or Korea, due to labor costs, regulatory compliance, and lack of an established supply chain ecosystem. The grants partially offset this premium but do not eliminate it.

---

## Cost Structure: What It Takes to Make a Chip

To make this concrete, here is an approximate cost breakdown for a leading-edge chip:

<div class="diagram">
<div class="diagram-title">Who Gets Paid to Make a Chip</div>
<div class="flow">
  <div class="flow-node accent wide">EDA Tools License: $5–50M/year per design team (Synopsys + Cadence + Siemens stacks)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">IP Core Licenses: $5–100M+ per chip (CPU core, PCIe PHY, memory interface, security, etc.)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">Mask Set (one tapeout): $5–30M (N3 mask set ~$15–20M; EUV masks ~$300–500K each × 80–120 layers)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Foundry Wafers: $16,000–$25,000/wafer (N3); $6,000–$10,000 (N7). H100-class: ~35–50 wafers per reticle per lot</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Advanced Packaging (CoWoS): +$600–$1,000 per unit above conventional packaging</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">Test: $50–300 per unit (final test, burn-in) for complex parts</div>
</div>
</div>

| Cost Item | Approximate Range | Notes |
|-----------|:----------------:|-------|
| Leading-edge fab (greenfield) | $20–40B | Does not include land, utilities, ecosystem buildout |
| Wafer cost (TSMC N3) | ~$20,000–$25,000/wafer | N5 was ~$16,000; N7 ~$10,000 |
| NRE (design + verification) | $100M–$1B+ | Complex GPU or SoC; includes EDA licenses, IP, masks, engineering |
| Die cost (H100-class, ~814 mm²) | $3,000–$5,000/die | At ~50–60 % yield, ~60–80 good dies per wafer; includes wafer + test + packaging prorated |
| Finished H100 SXM5 (MSRP) | ~$30,000–$40,000 (2023–2024) | Includes NVIDIA design margin, memory (HBM3), CoWoS, board, NRE amortization |

The gap between die cost (~$3–5K) and selling price (~$30–40K) reflects NVIDIA's design value, brand, software ecosystem (CUDA), NRE amortization, and supply scarcity. It also reflects the extraordinary difficulty of getting a chip working, fully verified, and in software-complete volume production — which is ultimately what [Chapter 31](./31_chip_design_flow.md) and [Chapter 32](./32_chip_verification_and_validation.md) are about.

---

## ASICs: When You Build Your Own

As explored in [Chapter 21](./21_asics.md), large hyperscalers have found that building custom ASICs for their specific workloads at sufficient volume justifies the NRE:

| Company | ASIC | Node | Purpose |
|---------|------|------|---------|
| Google | TPU v5p | TSMC N4P | Training & inference |
| Amazon | Trainium 2 | TSMC N3 | Training |
| Amazon | Inferentia 2 | TSMC N4 | Inference |
| Microsoft | Maia 100 | TSMC N5 | Training |
| Meta | MTIA v2 | TSMC N4P | Inference |
| Apple | M4 series | TSMC N3 | On-device AI + compute |

All of these are fabless designs manufactured at TSMC. They are customers of Synopsys/Cadence for EDA, customers of ARM or internal CPU teams for core IP, and customers of TSMC for wafers — reinforcing the same three structural concentration points.

---

## Why This Matters for Model Optimization

The foundry/fabless ecosystem defines three dimensions of your hardware environment:

**Price and availability**: The $30,000–40,000 price of an H100 is not arbitrary — it reflects the cost structure above, TSMC's yield, NVIDIA's NRE amortization, and supply scarcity amplified by CoWoS packaging constraints. When you amortize that cost over a training run, it directly determines whether a technique like quantization or sparsity is worth the engineering effort.

**Export controls and cluster geography**: The chips you can legally use in a given country, and which you can export or operate cross-border, is determined by BIS regulations that use the technical parameters described here. Multi-datacenter ML deployments must account for which accelerator SKUs are permitted at each location.

**Lead times and supply**: A greenfield TSMC fab takes 3–5 years from announcement to volume production. TSMC's Arizona Fab 21 was announced in 2020 and reached volume production in 2025. This 3–5 year lead time means the hardware available for your training cluster in 2027 is already determined by investment decisions being made today. Understanding the supply chain explains why hardware scarcity is a recurring, structural feature of AI infrastructure — not a temporary shortage.

For the future evolution of this ecosystem — chiplets, 3D integration, and co-packaged optics — see [Chapter 35](./35_where_the_industry_is_headed.md).

---

## Key Takeaways

- The chip industry splits into **fabless designers** (NVIDIA, AMD, Apple, Qualcomm), **pure-play foundries** (TSMC, Samsung, Intel Foundry), **IDMs** (Intel, Samsung memory), **EDA vendors** (Synopsys, Cadence, Siemens EDA), **IP licensors** (ARM, Synopsys DesignWare), and **equipment makers** (ASML, Applied Materials, Lam, KLA) — with extreme concentration at every layer.
- **TSMC** controls >90 % of leading-edge (≤ 5 nm) capacity; **ASML** has a 100 % monopoly on EUV scanners; **Japan** supplies ~60 % of wafers and ~90 % of EUV photoresist — three single points of failure in one supply chain.
- **EDA software** (Synopsys + Cadence in particular) is a prerequisite for any advanced chip design and is subject to U.S. export controls alongside physical equipment.
- **NRE costs** ($100M–$1B+) mean only large-volume products justify custom chip design; IP licensing makes complexity affordable by eliminating reinvention of verified blocks.
- **Export controls** (BIS October 2022, October 2023, and ongoing) have created a bifurcated global AI hardware market, restricting H100/B200-class chips to non-Chinese deployments and blocking EUV tool exports to China.
- **The CHIPS Act** ($39B in manufacturing grants) is attempting to geographically diversify leading-edge production, but structural cost advantages mean Taiwan and Korea will retain dominance for at least 5–10 years.
- For ML practitioners: the price, availability, and legality of the accelerators you train on are all direct outputs of the ecosystem described here — not market accidents.

---

*Next: [Appendix C — Photonics & Optical Interconnect](./appendix_c_photonics.md), where we examine how light is beginning to replace electrons for data movement inside datacenters — and eventually, perhaps, inside chips themselves.*

[← Back to Table of Contents](./README.md)
