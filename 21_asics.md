---
title: "Chapter 21 — ASICs (Application-Specific Integrated Circuits)"
---

[← Back to Table of Contents](./README.md)

# Chapter 21 — ASICs (Application-Specific Integrated Circuits)

In 2013, Google's data centers were running neural-network inference at a scale that would, within three years, require **doubling the number of data centers** to keep up — if they continued using CPUs. Instead of buying more hardware, they built their own. The result was the **TPU v1**, a chip designed for exactly one job: multiplying matrices. It ran at 700 MHz, consumed 40 W, and delivered 92 TOPS at INT8 — roughly 30× the performance-per-watt of the CPUs it replaced for inference. It was an ASIC: a chip whose function is fixed in silicon forever, optimized for a single purpose to an extent no general-purpose processor can match.

The ASIC — **Application-Specific Integrated Circuit** — is the endpoint of the hardware efficiency journey. Every FPGA emulates an ASIC. Every GPU ships with fixed tensor-core arrays that are, for matrix multiplication, themselves moving toward ASIC-like specialization. When volume is high enough and the workload stable enough, an ASIC is nearly always the right answer. The entire AI infrastructure boom of 2016–2026 is, at its heart, a wave of AI ASICs either displacing or supplementing GPUs wherever inference at scale is the workload.

> **The one-sentence version:** An ASIC is a chip designed and manufactured for one specific function, with every transistor, wire, and clock domain optimized for that function — giving it a 10–100× efficiency advantage over a general-purpose chip doing the same job, at the cost of complete inflexibility and massive upfront engineering expense.

---

## The FPGA → ASIC Spectrum

"ASIC" is not a binary label; it is one end of a spectrum of specificity. Understanding the spectrum prevents the common mistake of treating "ASIC vs FPGA" as the only choice:

<div class="diagram">
<div class="diagram-title">The Hardware Specificity Spectrum</div>
<div class="flow-h">
  <div class="flow-node accent">FPGA<br/><small>reconfigurable fabric; any function; zero NRE</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">Structured ASIC<br/><small>pre-placed cells; partial NRE; semi-custom</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">Standard-Cell ASIC<br/><small>synthesized from HDL; full NRE; scalable</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">Full-Custom ASIC<br/><small>every transistor hand-placed; maximum efficiency; rare</small></div>
</div>
</div>

| Category | NRE Cost | Per-unit cost | Dev time | Efficiency vs FPGA |
|---|---|---|---|---|
| FPGA | $0 | $500–$30K | Days–months | 1× baseline |
| Structured ASIC | $1–5M | $10–$100 | 3–6 months | 3–5× |
| Standard-cell ASIC (28 nm) | $3–10M | $2–$50 | 9–18 months | 5–20× |
| Standard-cell ASIC (7 nm) | $30–80M | $5–$200 | 18–30 months | 10–40× |
| Standard-cell ASIC (3 nm) | $200–500M | $10–$500 | 24–36 months | 15–60× |
| Full-custom ASIC | $100M+ | Low | 3–5 years | 20–100× |

The "NRE cost" (**Non-Recurring Engineering** cost) at leading nodes is dominated by **mask set costs** (EUV multi-patterning masks at 3 nm run $15–30M per full mask set) plus design, verification, and validation. The transition from FPGA to 7 nm ASIC is only economically justified when the per-unit savings multiplied by the volume exceeds the NRE — a break-even that takes millions of units at commodity prices.

### Standard-Cell ASIC: The Industry Standard

The vast majority of AI ASICs (TPUs, Trainium, Groq, etc.) are **standard-cell ASICs**: the designer writes RTL (Verilog/SystemVerilog), runs it through synthesis (mapping to a library of standard cells — pre-characterized NAND, NOR, DFF, MUX cells), then runs automated place-and-route. The tools optimize for power, performance, and area (PPA). See [Chapter 31](./31_chip_design_flow.md) for the full design flow.

---

## The NRE Cost and Volume Break-Even

The fundamental ASIC economics argument:

**FPGA**: pay $F$ per unit, $0$ NRE. Total cost at volume $V$: $C_{\text{FPGA}} = F \cdot V$

**ASIC**: pay $A$ per unit ($A \ll F$), $N$ NRE. Total cost at volume $V$: $C_{\text{ASIC}} = N + A \cdot V$

Break-even volume $V^*$ is where costs are equal:

$$V^* = \frac{N}{F - A}$$

For example, a 7 nm ASIC replacing an Alveo U250 FPGA:
- $F = \$10{,}000$ (FPGA unit cost)
- $A = \$200$ (ASIC unit cost at volume)
- $N = \$50{,}000{,}000$ (7 nm NRE)

$$V^* = \frac{50{,}000{,}000}{10{,}000 - 200} = \frac{50M}{9{,}800} \approx \mathbf{5{,}100\ \text{units}}$$

At 5,100 units, the ASIC breaks even. Google deploys **hundreds of thousands** of TPUs — the ASIC economics are overwhelming. A startup deploying 500 units never makes this work.

Plotting the cost curves:

```python
import numpy as np

V = np.logspace(2, 6, 1000)   # 100 to 1M units

# FPGA (Alveo U250-class)
F = 10_000   # $/unit
C_fpga = F * V

# 7nm ASIC
A = 200      # $/unit at volume
N = 50e6     # NRE
C_asic_7nm = N + A * V

# 28nm ASIC (cheaper NRE, worse perf/W but still much better than FPGA)
A28 = 50     # $/unit
N28 = 5e6    # NRE
C_asic_28nm = N28 + A28 * V

# Break-even points
V_star_7nm  = N   / (F - A)    # ~5,100 units
V_star_28nm = N28 / (F - A28)  # ~506 units

print(f"7nm ASIC break-even:  {V_star_7nm:,.0f} units")
print(f"28nm ASIC break-even: {V_star_28nm:,.0f} units")
# 7nm ASIC break-even:    5,102 units
# 28nm ASIC break-even:     504 units
```

The take-away: even a **28 nm ASIC breaks even at ~500 units** when replacing a high-end FPGA. For hyperscalers deploying at 100,000+ unit scale, the 7 nm ASIC is the obvious choice every time.

---

## Why ASICs Win on Efficiency

An ASIC's efficiency advantage comes from removing everything the workload doesn't need:

<div class="diagram">
<div class="diagram-title">Where Silicon Area Goes: GPU vs ASIC (rough, illustrative shares)</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">GPU (e.g. H100)</div>
    <ul>
      <li>Fetch / decode / scheduler units (~15% die)</li>
      <li>Register files, warp state (~10% die)</li>
      <li>L1/L2 cache (flexible, large) (~20% die)</li>
      <li>Tensor cores (~25% die)</li>
      <li>FP64 / FP32 / special-function units (~15% die)</li>
      <li>PCIe / NVLink IO (~15% die)</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">ASIC (e.g. TPU v4)</div>
    <ul>
      <li>Systolic array (matrix multiply) (~60% die)</li>
      <li>SRAM buffers (weight stationary) (~25% die)</li>
      <li>DMA controllers, HBM interface (~10% die)</li>
      <li>Simple scalar CPU (control) (~5% die)</li>
      <li><em>No</em> general caches, <em>no</em> warp schedulers</li>
    </ul>
  </div>
</div>
</div>

The GPU must be a **general-purpose parallel processor** — it needs warp schedulers, branch support, register files for arbitrary kernel state, L1/L2 caches for irregular access patterns, FP64 for scientific computing, and wide I/O for diverse workloads. An AI ASIC has none of that. It is a matmul engine with just enough control logic to feed it.

This is why a TPU v4 (275 TFLOP/s bf16 on a 7 nm die) and an H100 (989 TFLOP/s on 4 nm) land within a small factor of each other in performance per watt despite very different designs: the TPU spends almost no area on generality, the GPU spends its area on a newer node and a bigger die. The TPU's decisive advantage is cost: Google pays for wafers, not for NVIDIA's margins.

---

## The AI-ASIC Boom: A Tour of the Landscape

The 2016–2026 period has produced more AI-specific ASICs than any prior decade of specialized silicon. Here is the landscape:

### Google TPU (Tensor Processing Unit)

The canonical AI ASIC. Deployed internally from 2015, disclosed publicly in 2016, and described in detail in 2017.

| Generation | Year | Precision | Peak per chip | Key architecture |
|---|---|---|---|---|
| TPU v1 | 2015 | INT8 only | 92 TOPS | 256×256 systolic array; inference only; 75 W TDP |
| TPU v2 | 2017 | BF16 + FP32 | ~45 TFLOP/s | Training capable; HBM; 2 TensorCores |
| TPU v3 | 2018 | BF16 + FP32 | ~123 TFLOP/s | Liquid-cooled; 32 GiB HBM |
| TPU v4 | 2021 | BF16 + INT8 | 275 TFLOP/s | 4,096-chip pods; optical circuit switching |
| TPU v5e / v5p | 2023 | BF16 + INT8 | 197 / 459 TFLOP/s | Cost-optimised vs largest pods (8,960 chips) |
| Trillium (v6e) | 2024 | BF16 + INT8 | 918 TFLOP/s | 256×256 MXUs; +67 % perf/W vs v5e |
| Ironwood (v7) | 2025 | FP8 + BF16 | 4,614 TFLOP/s FP8 | 192 GB HBM3e; 9,216-chip pods |
| TPU 8t / 8i | 2026 (announced) | FP4 + FP8 + BF16 | 12.6 / 10.1 PFLOP/s FP4 | Separate training and inference chips |

Google does not publish per-chip power for recent TPUs. Per-chip numbers matter: Google's older "180" and "420 TFLOPS" figures for v2/v3 describe four-chip boards.

The systolic array architecture is covered in depth in [Chapter 16](./16_tpus_and_systolic_arrays.md). The key ASIC insight: **the TPU's systolic array has no programmable routing, no speculation, no caches** — data flows deterministically through a grid of MAC units, and that regularity is why the same area delivers far more MACs than a GPU.

### AWS Trainium and Inferentia

Amazon's **Inferentia** (2019) and **Inferentia2** (2023) are inference-focused ASICs for AWS cloud:

- **Inferentia2**: two **NeuronCore-v2** per chip, each with a 128×128 tensor (systolic) engine plus vector, scalar and general-purpose SIMD engines and tens of MB of on-chip SRAM; 190 TFLOP/s BF16/FP16 (380 TOPS INT8) per chip, 32 GB HBM at 820 GB/s
- Programmed via the **AWS Neuron SDK** (compiles PyTorch/JAX models through XLA-style graphs)

**Trainium** (2021), **Trainium2** (2024) and **Trainium3** (2025) extend the design to training:
- **Trainium2**: 667 TFLOP/s BF16 / 1.3 PFLOP/s FP8 dense, 96 GB HBM at 2.9 TB/s, about 4× Trainium1; 64-chip Trn2 UltraServers and the Anthropic-dedicated "Project Rainier" cluster
- **Trainium3** (TSMC 3 nm): 2.52 PFLOP/s FP8 dense, 144 GB HBM3e at 4.9 TB/s, 16:4 structured sparsity, 144-chip UltraServers
- **Trainium4** (announced): roughly 6× Trainium3's FP4 throughput and support for **NVLink Fusion**, so Trainium racks can share NVIDIA's scale-up fabric

### Groq LPU (Language Processing Unit)

Groq's architecture is the most unusual in the landscape. Rather than use HBM, the **Groq Language Processing Unit (LPU)** keeps all weights in **on-chip SRAM**:

- **GroqChip (TSP, Tensor Streaming Processor, 14 nm)**: 230 MB of SRAM with ~80 TB/s of on-die bandwidth, no HBM, no DRAM; ~188 TFLOP/s FP16 and 750 TOPS INT8 at 900 MHz
- **No caches, no dynamic scheduling**: the compiler schedules every data movement cycle by cycle, so latency is fully deterministic
- **Many chips per model**: 230 MB holds ~230 M INT8 weights, so a 70B model needs ~300 chips at INT8 (Groq's early Llama-2 70B deployment used 576 chips across eight racks). Throughput per user is excellent; capital cost per model is high.

> **Why so fast?** Decode is memory-bound, and Groq trades capacity for bandwidth: ~80 TB/s of SRAM per chip, multiplied across hundreds of chips that each hold a slice of the model, so every weight is read from SRAM instead of HBM.

In December 2025 **NVIDIA** signed a ~$20B non-exclusive licensing deal for Groq's inference technology and hired its founder and much of its team; GroqCloud continues independently. NVIDIA's first product from it, the **Groq 3 LPX** rack (256 LPUs, 128 GB of SRAM), is positioned as a decode co-processor beside Vera Rubin NVL72 racks. The SRAM-only idea has gone mainstream.

### Cerebras Wafer-Scale Engine (WSE)

Cerebras takes the opposite approach to chiplets: **put an entire cluster's worth of compute on one wafer**:

- **WSE-3**: a single silicon wafer (46,225 mm²; H100 GPU die is ~814 mm²), 900,000 cores, 44 GB on-wafer SRAM, 21 PB/s on-wafer memory bandwidth
- The wafer is **not diced**: the reticle-sized dies are connected across the scribe lines by on-wafer wiring, so all 900,000 cores sit on one fabric with no PCIe, no NVLink and no off-chip hops between them
- On-wafer SRAM bandwidth: 21 PB/s, roughly 800× the combined HBM bandwidth of eight H100s (~27 TB/s). Capacity is the catch: 44 GB, so larger models stream weights from external MemoryX servers or span several wafers

Yield management is Cerebras's core manufacturing innovation: a single wafer will have defective dies; the WSE architecture **routes around defects** at power-on, using redundant cores. This is only feasible because TSMC's 5 nm node has high yield, and because Cerebras has per-die redundancy built in.

Cerebras went public on Nasdaq in May 2026 (one of the year's largest IPOs) and in August 2026 announced the **CS-4**, a rack with three higher-clocked WSE-3 wafers; a new WSE-4 is expected with the CS-5 in 2027.

**What it's good at:** extremely large models and sparse/irregular access patterns that benefit from massive on-chip memory bandwidth. Not ideal for models that fit in HBM; very good for models that require cross-layer communication (e.g., mixture-of-experts with all experts on one chip).

### Groq, SambaNova, Tenstorrent

| Company | Architecture | Key differentiator |
|---|---|---|
| Groq | TSP (SRAM-only, static schedule) | Lowest-latency decode; technology licensed to NVIDIA (2025) |
| SambaNova | RDU (Reconfigurable Dataflow Unit) | Dataflow-programmable; SN40L adds a three-tier memory (SRAM, HBM, DDR) |
| Tenstorrent | Tensix cores (mesh NoC, RISC-V control) | Open software stack; Wormhole and Blackhole chips |
| Etched (Sohu) | Transformer-specific (claimed) | Bets on transformer permanence; performance figures unverified |
| Meta MTIA | In-house inference ASIC | MTIA v2 (2024): 354 TOPS INT8 dense, 256 MB SRAM, 90 W; ranking and recommendation |
| Microsoft Maia | In-house Azure accelerator | Maia 100 (2023); **Maia 200** (Jan 2026): TSMC 3 nm, ~10 PFLOP/s FP4, 216 GB HBM3e at 7 TB/s, 272 MB SRAM, ~750 W |
| OpenAI (with Broadcom) | Custom inference accelerator | 10 GW deployment plan announced 2025, first racks from 2H 2026 |

### Etched / Sohu: "Hardwiring the Transformer"

Etched's **Sohu** chip represents the most extreme specialization bet: **hardwire the transformer architecture** — attention, MLP, residual connections — at the transistor level. There is no instruction set; the "program" is the physical circuit. Every mm² of silicon performs exactly one transformer operation, with optimal wire lengths and no fetch/decode overhead.

Etched has claimed very high transformer throughput for an 8-chip Sohu server; those figures had not been independently verified at the time of writing. The risk is plain: if attention is replaced (state space models, linear attention, etc.), a hardwired chip loses its advantage. It is the ultimate bet that the transformer architecture is here to stay.

---

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>NVIDIA licensed Groq's SRAM-centric design, AWS is adopting NVLink Fusion for Trainium4, and Microsoft and Google publish benchmark comparisons against each other's chips. Is the market converging on "GPU plus specialised decode silicon" in every data centre, and if so, what remains defensible about building your own?</p>
</div>

## The Hyperscaler Argument: Why Build Your Own?

Why do Google, AWS, Meta, Microsoft, and others build ASICs instead of buying NVIDIA GPUs?

<div class="diagram">
<div class="diagram-title">Why Hyperscalers Build AI ASICs</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Cost at Scale</div>
    <div class="card-desc">At 100K+ units, the gap between an ASIC's manufacturing cost (hundreds to a few thousand dollars, plus HBM and packaging) and a merchant GPU's price (tens of thousands) is worth billions per deployment cycle.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Supply Chain Control</div>
    <div class="card-desc">NVIDIA GPU shortages in 2023–2024 caused training delays. A company with its own ASIC controls its own wafer allocation with TSMC directly.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Workload Co-Design</div>
    <div class="card-desc">An ASIC can be designed around the company's exact model architecture — BF16 arithmetic tailored to their softmax implementation, memory layout matched to their batching strategy.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Power Efficiency</div>
    <div class="card-desc">Data center power is the binding constraint for AI at scale. A 2× better TOPS/W ASIC directly halves the cooling and electricity bill — billions of dollars per year at Google's scale.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Differentiation</div>
    <div class="card-desc">A proprietary training chip means lower model training cost → faster iteration → better models → competitive moat. The chip IS the competitive advantage.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Custom Precision</div>
    <div class="card-desc">Hyperscalers can deploy BF16, FP8, or INT4 dataflows precisely matched to their models — not whatever NVIDIA's roadmap provides. Google used BF16 in TPU v2 (2017) before any GPU natively supported it.</div>
  </div>
</div>
</div>

---

## The ASIC Risk: Inflexibility vs Fast-Moving Models

The ASIC's strength — being optimized for one thing — is also its existential risk in a field that changes as fast as deep learning.

**The rigidity problem:** An ASIC designed in 2020 for transformer inference is fabulous at transformer inference. It is useless at:
- State space models (Mamba, RWKV) with their recurrent scan operations
- Sparse MoE attention with dynamic expert routing
- Flash-attention variants that require custom on-chip SRAM management
- New number formats (FP6, MXFP4) introduced after tape-out

Google has mitigated this by making TPU systolic arrays **programmable at the VLIW instruction level** (each TPU core has a simple instruction set), so new dataflows can be expressed without new hardware. Groq's fully static scheduling makes algorithm changes harder but not impossible — the compiler handles new model architectures as long as they decompose into matrix multiplications.

**Why training favors flexibility:** Research training workloads change monthly. A researcher testing a new attention mechanism needs hardware that runs their modified kernel today, not in 24 months after a new ASIC is taped out. This is why **GPU clusters dominate research training** and ASICs dominate **production inference** — the workload is stable in production, variable in research.

---

## ASIC Examples at a Glance

| ASIC | Maker | Purpose | Peak (dense) | Memory | Key feature |
|---|---|---|---|---|---|
| TPU v5p | Google | Training | 459 TFLOP/s BF16 | 95 GB HBM2e | 8,960-chip pods |
| Ironwood (TPU v7) | Google | Inference + training | 4,614 TFLOP/s FP8 | 192 GB HBM3e | 9,216-chip pods |
| Trainium2 | AWS | Training | 667 TFLOP/s BF16 | 96 GB HBM | NeuronLink; Project Rainier |
| Trainium3 | AWS | Training + inference | 2.52 PFLOP/s FP8 | 144 GB HBM3e | 3 nm; 144-chip UltraServers |
| Inferentia2 | AWS | Inference | 190 TFLOP/s BF16 / 380 TOPS INT8 | 32 GB HBM, 820 GB/s | Low cost per token |
| Groq TSP | Groq | Low-latency inference | 188 TFLOP/s FP16 / 750 TOPS INT8 | 230 MB SRAM | Deterministic, SRAM-only |
| WSE-3 | Cerebras | Training + inference | 125 PFLOP/s (vendor, sparse FP16) | 44 GB on-wafer SRAM | One wafer; ~23 kW CS-3 system |
| MTIA v2 | Meta | Inference | 354 TOPS INT8 | 256 MB SRAM + 128 GB LPDDR5 | 90 W |
| Maia 200 | Microsoft | Inference | ~10 PFLOP/s FP4 | 216 GB HBM3e, 7 TB/s | 3 nm; ~750 W |

Throughput in each chip's headline format; see the [Accelerator Atlas](./atlas.md) for comparable dense figures.

---

## Why This Matters for Model Optimization

<div class="diagram">
<div class="diagram-title">ASIC Implications for ML Practitioners</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Inference Costs Move to ASIC</div>
    <div class="card-desc">As workloads stabilize, inference migrates to ASICs. If you're serving at scale, your cost/token depends on the ASIC your cloud provider deploys — understand its supported precisions and batch sizes.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Supported Formats Are Fixed</div>
    <div class="card-desc">An ASIC supports exactly the number formats baked into it. Inferentia2 runs BF16/FP16/FP8 and INT8, Trainium3 adds MXFP8/MXFP4-style formats, older TPUs stop at BF16/INT8. Check before deploying: you may need to re-quantize.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Memory Architecture Drives Batching</div>
    <div class="card-desc">TPU's scratchpad SRAM vs GPU's HBM cache changes optimal batch sizes. TPU benefits from large, static batches (systolic array fill); Groq (SRAM-only) excels at batch=1 decode.</div>
  </div>
</div>
</div>

---

## Key Takeaways

- An **ASIC** is fixed-function silicon optimized entirely for one application, giving **10–100× better efficiency** than a general-purpose chip by eliminating every transistor not needed for the task.
- The spectrum runs **FPGA → structured ASIC → standard-cell ASIC → full-custom ASIC**; standard-cell ASICs are the industry standard, synthesized from RTL using commercial EDA tools.
- **NRE cost** at 7 nm is $30–80M for a mask set + design; this only makes sense when deploying **thousands to millions of units**, where per-unit savings overwhelm the fixed cost.
- The **volume break-even** formula: $V^* = N / (F - A)$ — at $50M NRE and $9,800 unit savings, break-even is ~5,100 units; for hyperscalers at 100K+ units, ASIC economics are decisive.
- The **AI-ASIC boom** includes Google TPU (systolic array, now split into training and inference chips), AWS Trainium/Inferentia (NeuronCore), Microsoft Maia, Meta MTIA, Groq LPU (SRAM-only; licensed to NVIDIA), Cerebras WSE (wafer-scale), and extreme bets like Etched's transformer-specific design.
- **Hyperscalers build ASICs** for cost control, supply chain independence, workload co-design, and power efficiency — each 2× TOPS/W improvement directly halves the data center electricity bill.
- The **key risk** is inflexibility: a 2020 transformer ASIC cannot run a 2024 state-space model efficiently. This is why research/training stays on GPUs — only production inference workloads are stable enough to justify silicon.

## Check Yourself

<details class="check">
<summary>An ASIC costs $120M in NRE and $800 per unit; the GPU it replaces costs $25,000. What is the break-even volume?</summary>
<p>V* = 120,000,000 / (25,000 − 800) ≈ 4,960 units. Hyperscalers deploying hundreds of thousands of accelerators clear that easily; a startup selling a few hundred does not.</p>
</details>

<details class="check">
<summary>How many Groq chips would hold an 8B model in INT8, and why might that still be attractive?</summary>
<p>8 GB ÷ 230 MB ≈ 35 chips. Each token reads every weight from on-chip SRAM at ~80 TB/s per chip with no HBM stalls, giving very high single-stream token rates, at the price of many chips per model.</p>
</details>

<details class="check">
<summary>Why is the Cerebras wafer not cut into dies like a normal wafer?</summary>
<p>Because the point is to keep all cores on one fabric: Cerebras stitches wiring across the scribe lines and routes around defective cores with spare rows, avoiding the bandwidth and energy cost of going off-chip.</p>
</details>

<details class="check">
<summary>What is the main technical risk of a hardwired transformer ASIC?</summary>
<p>Model architectures change faster than chips: new attention variants, state-space layers, MoE routing or new number formats may not map onto fixed datapaths, stranding the hardware before it has paid back its NRE.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| In-Datacenter Performance Analysis of a Tensor Processing Unit | N. Jouppi et al., ISCA 2017 | [arxiv.org/abs/1704.04760](https://arxiv.org/abs/1704.04760) | TPUv1 design and ASIC efficiency argument |
| Think Fast: A Tensor Streaming Processor (TSP) for Accelerating Deep Learning Workloads | D. Abts et al., ISCA 2020 | [doi.org/10.1109/ISCA45697.2020.00023](https://doi.org/10.1109/ISCA45697.2020.00023) | Groq TSP: 230 MB SRAM, static scheduling |
| AWS Neuron documentation: Inferentia2 and Trainium architecture | AWS | [awsdocs-neuron.readthedocs-hosted.com/en/late…](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/index.html) | NeuronCore engines, HBM, throughput |
| Announcing Amazon EC2 Trn3 UltraServers | AWS, Dec 2025 | [aws.amazon.com/about-aws/whats-new/2025/12/am…](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-ec2-trn3-ultraservers) | Trainium3 specs; Trainium4 with NVLink Fusion |
| Maia 200: The AI accelerator built for inference | Microsoft, Jan 2026 | [blogs.microsoft.com/blog/2026/01/26/maia-200-…](https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/) | Maia 200 specs |
| Our next-generation Meta Training and Inference Accelerator (MTIA v2) | Meta, 2024 | [ai.meta.com/blog/next-generation-meta-trainin…](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/) | MTIA v2 throughput, SRAM, power |
| Nvidia confirms $20 billion Groq deal | Tom's Hardware, 2025 | [tomshardware.com/tech-industry/semiconductors…](https://www.tomshardware.com/tech-industry/semiconductors/nvidia-confirms-20-billion-groq-deal-to-bolster-ai-inference-dominance) | NVIDIA–Groq licensing deal and Groq 3 LPX |

---

**Next:** [Chapter 22 — SoCs](./22_socs.md) — We see how CPU, GPU, NPU, ISP, and all the chips we've studied are integrated onto a single die — and why that integration is what makes mobile AI and Apple's M-series possible.

[← Back to Table of Contents](./README.md)
