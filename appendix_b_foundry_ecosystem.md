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

Taiwan Semiconductor Manufacturing Company is, by any measure, the most important technology company most people have never heard of. It is the **dominant pure-play foundry** for leading-edge logic, with roughly 70 % of total foundry revenue in 2025 and >90 % share of the most advanced nodes (≤ 5 nm).

| TSMC Node | Marketing Name | ~Transistor Density | Volume Production | Key Customers |
|-----------|---------------|:-------------------:|:-----------------:|---------------|
| N7 | 7 nm | ~91 MTr/mm² | 2018–present | NVIDIA A100, AMD Zen 2/3, MI100 |
| N5 / N4 (incl. NVIDIA 4N, 4NP) | 5 nm class | ~138–173 MTr/mm² | 2020–present | Apple A14/M1–M2, NVIDIA H100 and B200/B300, AMD MI300 compute dies |
| N3 (N3B, N3E, N3P) | 3 nm | ~1.6× N5 logic | 2022–present | Apple A17–A19 and M3–M5, AMD MI355X, Google and AWS accelerators, NVIDIA Rubin |
| N2 | 2 nm (first GAA) | ~1.15× N3E | volume since Q4 2025 | Apple A20 (2026), AMD Zen 6 "Venice" |
| A16 | 1.6 nm + backside power | ~1.1× N2 | expected 2027 | HPC/AI first |

Density figures for N3 and below are TSMC's relative claims; transistor counts per mm² depend heavily on the mix of logic, SRAM and analog, and SRAM has barely shrunk since N5.

TSMC's key structural advantages:
- **Process yield leadership**: years of accumulated know-how give TSMC the best defect densities (lowest $D$ in the yield model from [Appendix A](./appendix_a_how_chips_are_made.md)).
- **Neutral intermediary**: TSMC does not compete with customers by selling its own chips. Samsung Foundry, by contrast, competes with Qualcomm in mobile processors while also manufacturing for Qualcomm — a structural tension.
- **Scale**: a single leading-edge "GigaFab" phase costs $20B+, and TSMC's annual capex passed $40B in 2025.
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
| 2nm (SF2) | 2025 | N2 | Exynos 2600; Tesla's next AI chip (AI6) contracted for Samsung's Texas fab |

Samsung's challenge has been yield: on several advanced nodes Samsung lagged TSMC in defect density, leading key customers (Apple, AMD, NVIDIA) to shift volume entirely to TSMC. Samsung's internal chip divisions (Exynos, HBM memory) guarantee baseline utilization even in lean periods.

### Intel Foundry Services (IFS)

Intel is an IDM that has historically manufactured only for itself. **Intel Foundry** (called Intel Foundry Services until a 2024 rebrand) is Intel's effort to become a contract foundry:

- Intel 18A (in production since late 2025, Panther Lake first): **RibbonFET** (Intel's GAA transistor) plus **PowerVia** (backside power delivery), the first node anywhere with both. Yields were still ramping into 2026, and Intel has said 14A (2027–28, its first High-NA EUV node) depends on winning outside customers.
- Major challenges: Intel has missed multiple node transition deadlines (10nm was years late, 7nm was cancelled and reformulated), denting customer confidence.
- Strategic importance: as the only leading-edge fab located in the United States with scale aspirations, IFS is a linchpin of U.S. industrial policy under the CHIPS and Science Act.
- Key customers (announced): Microsoft and AWS for 18A-based chips, U.S. defense programs (RAMP-C, secure enclave); external volume is still small. Since August 2025 the U.S. government holds roughly 10% of Intel, and NVIDIA invested $5B in September 2025.

### GlobalFoundries, SMIC, and Trailing-Edge Foundries

**GlobalFoundries** (GF) spun out of AMD in 2009 and abandoned the leading-edge race in 2018, stopping at 12/14 nm. It focuses on:
- Specialized nodes: RF CMOS (for wireless modems), silicon photonics (for transceivers), FD-SOI (for IoT/automotive low-power), aerospace-grade
- Mature nodes (28/40/65/130 nm) that are critically important for automotive, industrial, and defense applications

**SMIC (Semiconductor Manufacturing International Corporation)** is China's most advanced foundry, which shipped an N+2 (7 nm-class) process in Huawei's Kirin 9000S in 2023 and an N+3 (5 nm-class) process in 2024–25, both made with DUV multi-patterning at lower yield and higher cost than EUV. It also builds Huawei's Ascend AI accelerators. Without EUV, each further step gets disproportionately harder.

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

The EDA market is effectively a **triopoly**, and since Synopsys bought Ansys in July 2025, simulation physics sits inside it too:

| Company | Core Strengths | Key Products |
|---------|---------------|-------------|
| **Synopsys** | Synthesis, formal verification, DFT, IP | Design Compiler, VCS, Formality, IC Compiler, PrimePower, ARC/DesignWare IP |
| **Cadence** | Place-and-route, simulation, analog | Innovus, Virtuoso, Spectre, Xcelium, Tempus, Quantus |
| **Siemens EDA** (ex-Mentor) | DRC/LVS, emulation, PCB | Calibre (the industry-standard DRC/LVS), Veloce emulation, Xpedition PCB |
| **Ansys** (part of Synopsys since 2025) | Power integrity, thermal and signal-integrity simulation | RedHawk-SC, Totem, SIwave |

Every advanced chip — every GPU, every TPU, every SoC — was designed using tools from this list. Synopsys and Cadence generated about $6B and $4.6B of revenue in 2024; their software is licensed to every fabless company and IDM on Earth.

The entire [Chapter 31](./31_chip_design_flow.md) (Chip Design Flow from RTL to GDSII) is the story of using these tools. The EDA companies also maintain **Process Design Kits (PDKs)** in partnership with foundries — the files that encode TSMC's N3 or Samsung's SF3E rules, device models, and design constraints that every chip designer must use.

### Why EDA Matters for Geopolitics

EDA tools are U.S.-headquartered (Synopsys: Sunnyvale, Cadence: San Jose, Siemens EDA: Fremont/Munich). U.S. export controls (BIS Entity List, EAR) on EDA software restrict Chinese chip companies' access to leading-edge EDA flows — an upstream chokepoint that complements the ASML/EUV hardware restriction. In May 2025 the U.S. briefly required licences for all EDA sales to China, then lifted the rule in July as part of trade talks, a reminder that this lever can be pulled at short notice.

---

## IP Cores: The Building Blocks You License

Even with EDA tools and a foundry process, a chip design team does not implement every functional block from scratch. Complex, heavily-verified blocks — CPU cores, USB/PCIe controllers, display interfaces, security enclaves, memory PHYs — are licensed as **IP cores** (Intellectual Property).

**ARM Holdings** is the dominant IP licensor for processor cores:
- Designs CPU/GPU/NPU architectures (Cortex-A, Cortex-M, Mali GPU, Ethos NPU)
- Licenses the ISA (Instruction Set Architecture) to allow customers to design their own implementations (Apple's M-series are custom ARM ISA implementations)
- Does not fab anything; collects royalties on every shipped chip (roughly 25–30 billion Arm-based chips shipped per year)
- RISC-V is the open alternative (no royalties), adopted by SiFive, Western Digital (storage), NVIDIA (embedded cores in GPUs), and Chinese companies seeking to escape ARM licensing constraints

**Synopsys DesignWare** provides thousands of verified IP blocks: PCIe Gen5/6 controllers, DDR5 PHY, USB4, Ethernet MAC/PHY, AES/RSA cryptographic cores, and more. Broadcom, NVIDIA, and Apple all license these blocks rather than implementing them internally.

**Imagination Technologies** licenses PowerVR GPU IP; **Cadence** and **Rambus** license memory interface IP. The economics: a complex verified IP block costs $1–10M to license, versus $50–200M+ to develop internally with comparable quality and schedule certainty.

---

## Equipment Makers: Who Builds the Machines That Build Chips

Fabs cannot operate without specialized semiconductor capital equipment. This layer has some of the most concentrated supply in the entire industry.

| Equipment Category | Leading Suppliers | Tool Examples |
|--------------------|------------------|---------------|
| Lithography | **ASML** (monopoly on EUV); Nikon, Canon (DUV) | EUV NXE:3800E, High-NA EXE:5000/5200, TWINSCAN ArF immersion |
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

The **ABF (Ajinomoto Build-up Film) substrate** — the organic package substrate under every BGA-style GPU package, including the ones that carry CoWoS interposers — is dominated by Ibiden and Shinko Electric Industries of Japan. A shortage here in 2021 contributed to GPU supply constraints.

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
- **December 2024**: HBM itself and more chipmaking tools added to the controls.
- **2025**: the "AI Diffusion" rule (January) would have capped exports to most countries by tier; it was rescinded in May before taking effect. NVIDIA's China-specific H20 required a licence from April, then was allowed in August in exchange for a share of revenue to the U.S. government, while Beijing discouraged its purchase; in December the administration moved to license H200-class sales to approved Chinese customers on similar terms. Expect the rules to keep moving; check the current text before planning a deployment.

ASML was restricted from shipping EUV tools to China by the Dutch government in January 2019 (and DUV immersion tools from 2023). This makes it impossible for SMIC or any Chinese foundry to advance to true leading-edge nodes without a domestic EUV breakthrough.

The technical enforcement mechanism is clever: AI chips are measured by **Total Processing Performance (TPP)** and **Performance Density** thresholds, with an interconnect bandwidth provision. A chip that exceeds either threshold — measured in FLOPS/second for certain precisions — requires a license that is generally denied for Chinese military or advanced AI applications.

For ML practitioners: this directly determines which GPU models you can legally ship to, sell services from, or operate in different jurisdictions. A model trained on H100s may legally run inference only on chips approved for the target market.

### CHIPS and Science Act (2022)

The U.S. CHIPS and Science Act allocated $52.7B for semiconductor R&D and manufacturing incentives, with $39B in direct manufacturing grants:
- TSMC Arizona: $6.6B grant; TSMC's planned Arizona investment grew to $165B (six fabs, packaging and R&D). The first fab (N4) entered volume production in late 2024; N3 and N2 fabs follow
- Intel Ohio and Arizona: ~$7.9B grant (finalized late 2024), converted in August 2025 into a roughly 10% government equity stake
- Samsung Texas: ~$4.7B grant (finalized late 2024, reduced from $6.4B)
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
  <div class="flow-node blue wide">Mask Set (one tapeout): tens of millions of dollars at N3 (estimates vary); ~80–100 masks, about two dozen of them EUV</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Foundry Wafers: ~$20,000 (N3), ~$30,000 (N2, reported), ~$10,000 (N7). An H100-size die gives ~63 candidates per wafer</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">HBM + Advanced Packaging (CoWoS): the largest per-unit costs for an AI GPU — HBM alone is estimated at well over $1,000 per H100-class part</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">Test: $50–300 per unit (final test, burn-in) for complex parts</div>
</div>
</div>

| Cost Item | Approximate Range | Notes |
|-----------|:----------------:|-------|
| Leading-edge fab (greenfield) | $20–40B | Does not include land, utilities, ecosystem buildout |
| Wafer cost (TSMC N3) | ~$20,000/wafer | N2 reported ~$30,000; N5 ~$16,000; N7 ~$10,000 |
| NRE (design + verification) | $100M–$1B+ | Complex GPU or SoC; includes EDA licenses, IP, masks, engineering |
| Manufacturing cost (H100 SXM5) | ~$3,000–3,500 per unit (analyst estimates) | ~63 candidate dies/wafer, perhaps 30–45 good: a few hundred dollars of logic wafer per GPU. HBM, CoWoS, test and board are the bigger items |
| Finished H100 SXM5 (MSRP) | ~$30,000–$40,000 (2023–2024) | Includes NVIDIA design margin, memory (HBM3), CoWoS, board, NRE amortization |

The gap between manufacturing cost (~$3K) and selling price (~$30–40K) reflects NVIDIA's design value, brand, software ecosystem (CUDA), NRE amortization, and supply scarcity. It also reflects the extraordinary difficulty of getting a chip working, fully verified, and in software-complete volume production — which is ultimately what [Chapter 31](./31_chip_design_flow.md) and [Chapter 32](./32_chip_verification_and_validation.md) are about.

---

## ASICs: When You Build Your Own

As explored in [Chapter 21](./21_asics.md), large hyperscalers have found that building custom ASICs for their specific workloads at sufficient volume justifies the NRE:

| Company | ASIC | Node | Purpose |
|---------|------|------|---------|
| Google (with Broadcom) | TPU v7 Ironwood; TPU 8t / 8i | not disclosed | Training & inference (split in gen 8) |
| Amazon | Trainium2 / Trainium3 | TSMC 5 nm class / 3 nm | Training and inference |
| Microsoft | Maia 200 | TSMC 3 nm | Inference |
| Meta | MTIA v2 | TSMC 5 nm | Ranking/recommendation inference |
| OpenAI (with Broadcom) | Custom accelerator | not disclosed | Inference, first racks 2H 2026 |
| Apple | M5 / A19 | TSMC N3P | On-device AI + compute |

All of these are fabless designs manufactured at TSMC. They are customers of Synopsys/Cadence for EDA, customers of ARM or internal CPU teams for core IP, and customers of TSMC for wafers — reinforcing the same three structural concentration points.

---

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Count the single points of failure under one GPU: one EUV supplier, one leading-edge foundry, a handful of HBM makers, one advanced-packaging line, two EDA vendors. Which one, if it stalled for six months, would hurt your own roadmap most, and is there any substitute at all?</p>
</div>

## Why This Matters for Model Optimization

The foundry/fabless ecosystem defines three dimensions of your hardware environment:

**Price and availability**: The $30,000–40,000 price of an H100 is not arbitrary — it reflects the cost structure above, TSMC's yield, NVIDIA's NRE amortization, and supply scarcity amplified by CoWoS packaging constraints. When you amortize that cost over a training run, it directly determines whether a technique like quantization or sparsity is worth the engineering effort.

**Export controls and cluster geography**: The chips you can legally use in a given country, and which you can export or operate cross-border, is determined by BIS regulations that use the technical parameters described here. Multi-datacenter ML deployments must account for which accelerator SKUs are permitted at each location.

**Lead times and supply**: A greenfield TSMC fab takes 3–5 years from announcement to volume production. TSMC's Arizona Fab 21 was announced in 2020 and reached volume production in late 2024. This 3–5 year lead time means the hardware available for your training cluster in 2027 is already determined by investment decisions being made today. Understanding the supply chain explains why hardware scarcity is a recurring, structural feature of AI infrastructure — not a temporary shortage.

For the future evolution of this ecosystem — chiplets, 3D integration, and co-packaged optics — see [Chapter 35](./35_where_the_industry_is_headed.md).

---

## Key Takeaways

- The chip industry splits into **fabless designers** (NVIDIA, AMD, Apple, Qualcomm), **pure-play foundries** (TSMC, Samsung, Intel Foundry), **IDMs** (Intel, Samsung memory), **EDA vendors** (Synopsys, Cadence, Siemens EDA), **IP licensors** (ARM, Synopsys DesignWare), and **equipment makers** (ASML, Applied Materials, Lam, KLA) — with extreme concentration at every layer.
- **TSMC** controls >90 % of leading-edge (≤ 5 nm) capacity; **ASML** has a 100 % monopoly on EUV scanners; **Japan** supplies ~60 % of wafers and ~90 % of EUV photoresist — three single points of failure in one supply chain.
- **EDA software** (Synopsys + Cadence in particular) is a prerequisite for any advanced chip design and is subject to U.S. export controls alongside physical equipment.
- **NRE costs** ($100M–$1B+) mean only large-volume products justify custom chip design; IP licensing makes complexity affordable by eliminating reinvention of verified blocks.
- **Export controls** (BIS October 2022, October 2023, December 2024, and repeated 2025 changes) have created a bifurcated global AI hardware market, restricting H100/B200-class chips to non-Chinese deployments and blocking EUV tool exports to China.
- **The CHIPS Act** ($39B in manufacturing grants) is attempting to geographically diversify leading-edge production, but structural cost advantages mean Taiwan and Korea will retain dominance for at least 5–10 years.
- For ML practitioners: the price, availability, and legality of the accelerators you train on are all direct outputs of the ecosystem described here — not market accidents.

## Check Yourself

<details class="check">
<summary>Why can SMIC still make 5 nm-class chips without EUV, and what does it cost them?</summary>
<p>DUV immersion with multi-patterning (several exposures and etches per critical layer) can print the features, but each extra pass adds cost, cycle time and overlay error, so yields are lower and each wafer is far more expensive.</p>
</details>

<details class="check">
<summary>Roughly how much of an H100's selling price is logic-die wafer cost?</summary>
<p>A ~$20K wafer gives ~63 candidates and perhaps 30–45 good dies, so a few hundred dollars per GPU: about 1–2% of a $30K+ price. HBM, packaging, design, software and margin are the rest.</p>
</details>

<details class="check">
<summary>Which three companies or countries form single points of failure in the leading-edge supply chain?</summary>
<p>ASML (EUV scanners, Netherlands), TSMC (leading-edge logic and CoWoS, Taiwan), and Japan's materials suppliers (wafers, EUV photoresist, ABF substrates).</p>
</details>

<details class="check">
<summary>What changed about Intel's ownership and foundry strategy in 2025?</summary>
<p>The U.S. government converted CHIPS grants into a ~10% stake, NVIDIA invested $5B, and Intel said 14A depends on external customers, while 18A entered production for its own products.</p>
</details>

<details class="check">
<summary>Why does TSMC's neutrality matter to fabless customers?</summary>
<p>TSMC sells no chips of its own, so customers like Apple, NVIDIA and AMD are not handing designs and roadmaps to a competitor, unlike Samsung Foundry, whose parent also sells phones and SoCs.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| TSMC Annual Report and technology pages | TSMC | [tsmc.com/english/dedicatedFoundry/technology/…](https://www.tsmc.com/english/dedicatedFoundry/technology/logic) | Node roadmap and customers |
| Semiconductor Industry Association: State of the Industry | SIA, 2025 | [semiconductors.org](https://www.semiconductors.org/) | Market structure and supply chain |
| Emerging Resilience in the Semiconductor Supply Chain | SIA / BCG, 2024 | [semiconductors.org/emerging-resilience-in-the…](https://www.semiconductors.org/emerging-resilience-in-the-semiconductor-supply-chain/) | Geographic concentration |
| BIS advanced computing export controls | U.S. Department of Commerce | [bis.gov](https://www.bis.gov/) | 2022–2025 rules |
| CHIPS for America awards | NIST | [nist.gov/chips](https://www.nist.gov/chips) | Grant amounts and recipients |
| Synopsys completes acquisition of Ansys | Synopsys, July 2025 | [news.synopsys.com/2025-07-17-Synopsys-Complet…](https://news.synopsys.com/2025-07-17-Synopsys-Completes-Acquisition-of-Ansys) | EDA consolidation |
| NVIDIA and Intel collaboration | NVIDIA, Sept 2025 | [nvidianews.nvidia.com/news/nvidia-and-intel-t…](https://nvidianews.nvidia.com/news/nvidia-and-intel-to-develop-ai-infrastructure-and-personal-computing-products) | $5B investment |
| Chip War | Chris Miller, 2022 | [simonandschuster.com/books/Chip-War/Chris-Mil…](https://www.simonandschuster.com/books/Chip-War/Chris-Miller/9781982172008) | History of the foundry model and geopolitics |

---

**Next:** [Appendix C — Photonics & Optical Interconnect](./appendix_c_photonics.md) — We examine how light is beginning to replace electrons for data movement inside data centres, and eventually perhaps inside packages.

[← Back to Table of Contents](./README.md)
