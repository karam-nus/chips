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
<div class="diagram-title">Where Silicon Area Goes: GPU vs ASIC (Matrix Multiply)</div>
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

This is why the TPU v4 delivers **275 TOPS** at ~170 W (1.6 TOPS/W at BF16) while an H100 SXM delivers ~990 TOPS at 700 W (1.4 TOPS/W) — competitive, but the TPU achieves similar perf/W in far less die area, at lower cost per TOPS when deployed at scale.

---

## The AI-ASIC Boom: A Tour of the Landscape

The 2016–2026 period has produced more AI-specific ASICs than any prior decade of specialized silicon. Here is the landscape:

### Google TPU (Tensor Processing Unit)

The canonical AI ASIC. Introduced internally in 2015–2016, publicly disclosed 2017.

| Generation | Year | Precision | Peak TOPS | TDP | Key Architecture |
|---|---|---|---|---|---|
| TPU v1 | 2016 | INT8 only | 92 | 40 W | 256×256 systolic array; inference only |
| TPU v2 | 2017 | BF16 + FP32 | 180 | 280 W | Training capable; HBM; 2 cores |
| TPU v3 | 2018 | BF16 + FP32 | 420 | 450 W | Liquid-cooled; 2 cores; 16 GB HBM |
| TPU v4 | 2021 | BF16 + INT8 | 275 (BF16) | ~170 W | 4096-chip pods; ICI optical mesh |
| TPU v5e | 2023 | BF16 + INT8 | 197 (BF16) | ~170 W | High efficiency; volume inference |
| TPU v5p | 2023 | BF16 + INT8 | 459 (BF16) | ~500 W | Largest pods; training focus |
| Trillium (v6e) | 2024 | BF16 + INT8 | ~4× v5e | ~100 W | 4× compute/W improvement |

The systolic array architecture is covered in depth in [Chapter 16](./16_tpus_and_systolic_arrays.md). The key ASIC insight: **the TPU's systolic array has no programmable routing, no speculation, no caches** — data flows deterministically through a grid of MAC units, and that regularity is why the same area delivers far more MACs than a GPU.

### AWS Trainium and Inferentia

Amazon's **Inferentia** (2019) and **Inferentia2** (2023) are inference-focused ASICs for AWS cloud:

- **Inferentia2**: 190 TOPS INT8, 384 GB/s HBM, 32 GB capacity, ~300 W, dual-core
- Each core has two **NeuronCore-v2**: 2× 128×128 matrix multipliers + 16 GB SRAM (fast scratchpad, not cache)
- Programmed via **AWS Neuron SDK** (compiles PyTorch/JAX models to Neuron IR)
- Cost: ~$0.08/TOPS-hour vs ~$0.15/TOPS-hour for equivalent GPU capacity

**Trainium** (2021) and **Trainium2** (2024) extend the design to training:
- Trainium2: BF16/FP8, 20× better performance than Trainium on large LLMs, built for 100K-chip "ultra-cluster" deployments

### Groq LPU (Language Processing Unit)

Groq's architecture is the most unusual in the landscape. Rather than use HBM, the **Groq LanguageModel Processing Unit (LPU)** packs all weights into **on-chip SRAM**:

- **GroqChip (TSP — Tensor Streaming Processor)**: 80 MB SRAM (no HBM, no DRAM), 750 MHz, ~188 TOPS BF16, ~75 W
- **No caches, no dynamic scheduling**: the compiler statically schedules every data movement cycle-by-cycle at compile time
- Result: **deterministic, extremely low-latency inference** — single-chip latency for a token of a 7B INT8 model is ~0.1 ms; 8-chip system is ~0.05 ms/token

The trade-off: **model size is limited by on-chip SRAM capacity**. An 80 MB chip can fit ~80M INT8 parameters. For a 70B parameter model you need ~875 GroqChips networked together, each holding a shard. In practice, Groq deploys multi-chip clusters for large models.

> **Why ultra-low latency?** HBM access latency is ~50–100 ns; SRAM is ~1–5 ns. By eliminating HBM, Groq removes the memory-access stall that limits every weight-fetching operation in transformer decode — the workload is compute-bound rather than memory-bound. At decode with batch=1, this matters enormously. See [Chapter 11](./11_memory_wall_and_bandwidth.md) for the roofline analysis.

### Cerebras Wafer-Scale Engine (WSE)

Cerebras takes the opposite approach to chiplets: **put an entire cluster's worth of compute on one wafer**:

- **WSE-3**: a single silicon wafer (46,225 mm²; H100 GPU die is ~814 mm²), 900,000 cores, 44 GB on-wafer SRAM, 21 PB/s on-wafer memory bandwidth
- The wafer is diced into "good die" sections and interconnected with **on-silicon fabric** — no PCIe, no NVLink, no off-chip hops between cores
- On-wafer bandwidth: 21,000 GB/s — ~27× more than 8 × H100 connected by NVLink

Yield management is Cerebras's core manufacturing innovation: a single wafer will have defective dies; the WSE architecture **routes around defects** at power-on, using redundant cores. This is only feasible because TSMC's 5 nm node has high yield, and because Cerebras has per-die redundancy built in.

**What it's good at:** extremely large models and sparse/irregular access patterns that benefit from massive on-chip memory bandwidth. Not ideal for models that fit in HBM; very good for models that require cross-layer communication (e.g., mixture-of-experts with all experts on one chip).

### Groq, SambaNova, Tenstorrent

| Company | Architecture | Key differentiator |
|---|---|---|
| Groq | TSP (SRAM-only, static schedule) | Lowest latency decode; fully deterministic |
| SambaNova | RDU (Reconfigurable Dataflow Unit) | Dataflow-programmable; dynamic sparsity |
| Tenstorrent | Tensix cores (mesh NoC) | Open ISA; Wormhole/Blackhole chips; developer-friendly |
| Etched/Sohu | Transformer-hardwired (no programmability) | Every transistor is a transformer op; 0 flexibility |
| Meta MTIA | Training/inference custom ASIC | In-house; powers Meta's recommendation & LLM systems |
| Microsoft Maia 100 | Azure-deployed LLM training ASIC | Paired with Cobalt CPU (ARM); Azure OpenAI backend |

### Etched / Sohu: "Hardwiring the Transformer"

Etched's **Sohu** chip represents the most extreme specialization bet: **hardwire the transformer architecture** — attention, MLP, residual connections — at the transistor level. There is no instruction set; the "program" is the physical circuit. Every mm² of silicon performs exactly one transformer operation, with optimal wire lengths and no fetch/decode overhead.

Claimed peak: ~144 PFLOPS for transformer operations at 700 W. The risk: if attention is replaced (state space models, linear attention, etc.), the chip is worthless. This is the ultimate bet that the transformer architecture is here to stay.

---

## The Hyperscaler Argument: Why Build Your Own?

Why do Google, AWS, Meta, Microsoft, and others build ASICs instead of buying NVIDIA GPUs?

<div class="diagram">
<div class="diagram-title">Why Hyperscalers Build AI ASICs</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Cost at Scale</div>
    <div class="card-desc">At 100K+ units, ASIC unit cost ($200–500) vs H100 ($30K) means billions in savings per deployment cycle. Google has millions of TPU chips deployed.</div>
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

| ASIC | Maker | Purpose | Precision | TOPS | TDP | Key Feature |
|---|---|---|---|---|---|---|
| TPU v4 | Google | Training + Inference | BF16 | 275 | ~170 W | Systolic array; optical ICI mesh |
| TPU v5p | Google | Large-scale Training | BF16/INT8 | 459 | ~500 W | Largest TPU pod (8960 chips) |
| Trainium2 | AWS | Training | BF16/FP8 | ~340 | ~500 W | 96 GB HBM3; NeuronLink mesh |
| Inferentia2 | AWS | Inference | INT8/BF16 | 190 | ~300 W | 384 GB/s; 32 GB HBM; low cost |
| Groq TSP | Groq | Low-latency Inference | INT8/BF16 | 188 | 75 W | SRAM-only; deterministic latency |
| WSE-3 | Cerebras | Large-model Training | FP16/BF16 | 125 PFLOPS | 23 kW | One wafer; 44 GB on-chip SRAM |
| MTIA v2 | Meta | Inference | INT8/BF16 | ~40 | ~25 W | Recommendation + LLM serving |
| Maia 100 | Microsoft | Training | BF16/FP8 | n/a | ~500 W | Azure OpenAI backend |
| Sohu | Etched | Transformer Inference | BF16 | 144 PFLOPS | ~700 W | Hardwired transformer; no ISA |

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
    <div class="card-desc">An ASIC supports exactly the number formats baked into it. AWS Inferentia2 supports BF16 and INT8. If your model needs FP8 or INT4, check before deploying — you may need re-quantization.</div>
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
- The **AI-ASIC boom** includes Google TPU (systolic array), AWS Trainium/Inferentia (NeuronCore), Groq LPU (SRAM-only deterministic latency), Cerebras WSE (wafer-scale), and extreme bets like Etched Sohu (hardwired transformer).
- **Hyperscalers build ASICs** for cost control, supply chain independence, workload co-design, and power efficiency — each 2× TOPS/W improvement directly halves the data center electricity bill.
- The **key risk** is inflexibility: a 2020 transformer ASIC cannot run a 2024 state-space model efficiently. This is why research/training stays on GPUs — only production inference workloads are stable enough to justify silicon.

---

*Next: [Chapter 22 — SoCs](./22_socs.md), where we see how CPU, GPU, NPU, ISP, and all the chips we've studied are integrated onto a single die — and why that integration is what makes mobile AI and Apple's M-series possible.*

[← Back to Table of Contents](./README.md)
