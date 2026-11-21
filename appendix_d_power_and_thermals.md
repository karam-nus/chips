---
title: "Appendix D — Power, Thermals & Energy"
---

[← Back to Table of Contents](./README.md)

# Appendix D — Power, Thermals & Energy

Power is not a footnote to chip design — it *is* chip design. Every architectural decision in this guide, from transistor sizing to tensor-core precision to HBM placement, is ultimately constrained by watts. This appendix connects the physics of a switching transistor (introduced in [Chapter 1](./01_what_is_a_chip.md)) all the way to the kilowatt-hour cost of training a large language model and the grid-level sustainability challenge that now shapes the AI industry.

---

## Where the Power Goes: Dynamic vs Static

### Dynamic Power

The dominant form of power in a running digital chip is **dynamic power** — power consumed every time a transistor switches. Recall from [Chapter 1](./01_what_is_a_chip.md) the energy to switch one CMOS gate:

$$E_{\text{switch}} \approx \tfrac{1}{2}\, C V_{dd}^{2}$$

Extend this to a whole chip switching at frequency $f$, with $\alpha$ the **activity factor** (fraction of gates switching each cycle), and $C$ the total switched capacitance:

$$\boxed{P_{\text{dynamic}} = \alpha \, C \, V_{dd}^{2} \, f}$$

This single equation is the master formula behind nearly every power-saving technique in the industry:

| Lever | Reduction in $P$ | Practical example |
|---|---|---|
| Lower $V_{dd}$ (voltage scaling) | Quadratic — halve $V$ → ¼ power | DVFS at light workload |
| Lower $f$ (clock scaling) | Linear | Reduce GPU boost clock under thermal limit |
| Lower $\alpha$ (fewer switches) | Linear | Clock-gating idle units; quantization (fewer bits to charge) |
| Lower $C$ (smaller transistors) | Linear | Smaller process node; INT8 adder < FP32 adder area |

> **Why quantization saves power:** A 32-bit adder has roughly 4× the switching capacitance of an 8-bit adder. Moving from FP32 → INT8 reduces $C$ and $\alpha$ together, giving roughly a 4–16× reduction in arithmetic energy — before any memory-bandwidth savings. See [Chapter 25](./25_quantization_hardware_view.md).

### Static (Leakage) Power

Even a transistor doing nothing consumes **leakage power**: current that seeps through the gate oxide and from drain to source when the transistor is "off." Leakage was negligible at 250 nm but has grown dramatically with each shrink:

- At 5 nm, **subthreshold leakage** and **gate-oxide leakage** together can be 20–40% of total chip power under load, and most of it at idle.
- This is why [Dennard scaling ended](./01_what_is_a_chip.md) (~2005): shrinking the transistor no longer let you lower $V_{dd}$ freely, because the on/off current ratio degraded and leakage exploded.
- An idle GB200 NVL72 rack still draws on the order of 10–20 kW, much of it leakage and always-on clocks, fans and pumps.

$$P_{\text{total}} = P_{\text{dynamic}} + P_{\text{static}}$$

<div class="diagram">
<div class="diagram-title">Power Budget of a Modern GPU (approximate, H100 SXM at load)</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Dynamic Power (~75%)</div>
    <ul>
      <li>Compute units (tensor cores, SFUs): ~55%</li>
      <li>Memory system (SRAM reads/writes, HBM I/O): ~15%</li>
      <li>Interconnect (NVLink PHYs, PCIe): ~5%</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Static / Overhead (~25%)</div>
    <ul>
      <li>Leakage across 80 B transistors: ~15%</li>
      <li>Clock-tree power (global distribution): ~7%</li>
      <li>I/O & power-delivery circuits: ~3%</li>
    </ul>
  </div>
</div>
</div>

---

## TDP vs Actual Power vs Peak

Three numbers you will encounter constantly:

| Metric | What it measures | Typical H100 SXM value |
|---|---|---|
| **TDP** (Thermal Design Power) | Maximum *sustained* power the cooling system must handle; vendors design thermals to this | 700 W (SXM5) |
| **Peak instantaneous power** | Very short bursts (ms-scale) above TDP; managed by on-package capacitors and power delivery | Can briefly reach 1.1–1.3× TDP |
| **Typical workload power** | Actual average during a real training job; often 80–95% of TDP for dense GEMM | ~600–680 W |
| **Idle power** | Leakage + clocks running at minimal frequency | ~50–80 W |

> **TDP is not a hard cap.** It is the value the cooling system is *rated* for. If cooling is insufficient, the chip will **thermally throttle** — reducing $f$ and $V_{dd}$ until power dissipation falls within what can be shed. See [Chapter 13](./13_reliability_and_fault_tolerance.md) on throttling as a reliability mechanism.

---

## Power Delivery: Getting 1,000 Amperes Onto a Die

Delivering power to a modern GPU is a formidable engineering challenge. H100 at 700 W and 0.8 V supply requires:

$$I = \frac{P}{V} = \frac{700\,\text{W}}{0.8\,\text{V}} \approx 875\,\text{A}$$

B200 at 1,000 W and ~0.85 V requires over **1,100 A** — flowing into a package roughly 100 cm².

### Voltage Regulator Modules (VRMs)

- **VRMs** sit on the PCB (or increasingly in the package) and step the board-level 12–48 V supply down to the 0.7–1.0 V the die needs.
- Modern GPU cards use 8–12 phase VRMs; server boards may use 20+ phases.
- **48 V bus architectures** (Google TPU pods, OCP Open Rack v3, NVIDIA's NVL72 racks) reduce the current delivered across the board (halving V doubles I, so using 48 V instead of 12 V cuts bus current by 4×), reducing $I^2R$ losses in board traces.

### IR Drop: The Voltage Distribution Problem

When 875 A flows through the on-die power grid (metal layers M1–M10 acting as a power network), the resistance of those metal wires causes a voltage drop $\Delta V = I \cdot R$. A 2% IR drop at 0.8 V is only 16 mV — but that can push the local $V_{dd}$ below the minimum needed for reliable timing, causing **setup-time violations** and incorrect computation.

<div class="diagram">
<div class="diagram-title">Power Delivery Path (board → die)</div>
<div class="flow-h">
  <div class="flow-node accent">48 V / 12 V<br/><small>datacenter bus</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">VRM<br/><small>board-level regulator</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">Package power<br/><small>bumps & C4 balls</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">On-die power<br/><small>metal grid, decaps</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange">Transistor<br/><small>0.7–1.0 V, local</small></div>
</div>
</div>

**On-die voltage regulation (ODVR / FIVR)** — placing small voltage regulators directly on the chip die — reduces the last few mm of IR drop. Intel pioneered Fully Integrated Voltage Regulators (FIVR, Haswell, 2013) and the industry is moving toward die-embedded regulators to support fine-grained per-cluster DVFS.

The rack is changing too. At ~130 kW per NVL72 rack, its 48–54 V busbars carry ~2,500 A; NVIDIA's 2027 "Kyber" racks for Rubin Ultra aim at ~600 kW and beyond, and NVIDIA has proposed **800 V DC** distribution from the facility to the rack to cut copper and conversion losses.

---

## DVFS: Dynamic Voltage and Frequency Scaling

**DVFS** is the primary runtime power-management tool. The insight is that $P \propto V^2 f$, and since $f_{\max} \approx k(V - V_t)^\alpha / L$ (transistor switching speed depends on overdrive voltage), you can:

- Lower $V_{dd}$ → lowers both $f_{\max}$ and $P$ quadratically (power wins more than speed loses at low utilization).
- Raise $V_{dd}$ → allows higher $f$ (boost clocks) at higher power.

**NVIDIA GPU DVFS** cycles through hundreds of operating points per second:

```python
# Conceptual: energy per token vs DVFS point
import numpy as np

def token_energy_joules(
    freq_mhz: float,
    voltage_v: float,
    tokens_per_sec: float,
    capacitance_nf: float = 500.0,    # typical GPU effective C in nanofarads
    alpha: float = 0.6,               # activity factor
) -> float:
    """Compute energy per token given a DVFS operating point."""
    power_w = alpha * (capacitance_nf * 1e-9) * (voltage_v ** 2) * (freq_mhz * 1e6)
    energy_per_token_j = power_w / tokens_per_sec
    return energy_per_token_j

# High-performance point
e_high = token_energy_joules(freq_mhz=1980, voltage_v=1.0, tokens_per_sec=2200)

# Power-save point
e_save = token_energy_joules(freq_mhz=1350, voltage_v=0.85, tokens_per_sec=1600)

print(f"High-perf: {e_high*1000:.3f} mJ/token")
print(f"Power-save: {e_save*1000:.3f} mJ/token")
print(f"Energy ratio: {e_high/e_save:.2f}x  |  Throughput ratio: {2200/1600:.2f}x")
# High-perf: 270.000 mJ/token
# Power-save: 182.883 mJ/token
# Energy ratio: 1.48x  |  Throughput ratio: 1.38x
# => 38% more tokens/s costs 48% more energy per token — not always worth it
```

### Clock Gating and Power Gating

- **Clock gating**: Disable the clock to an idle block (tensor core cluster, cache bank). Zero $\alpha$ → zero dynamic power for that block. Standard in every modern chip; can save 20–40% of chip-level dynamic power during sparse workloads.
- **Power gating**: Disconnect supply voltage entirely from an idle block. Eliminates leakage too. Slower to wake up (microseconds) but critical for aggressive idle power on mobile. On GPUs, entire streaming multiprocessor (SM) clusters are power-gated when context is evicted.

---

## Performance per Watt: The True Figure of Merit

When comparing accelerators, raw FLOPS is a misleading metric. The useful quantity is **performance per watt** — equivalently, **energy per operation**:

$$\eta = \frac{\text{TOPS}}{W} \quad \text{or} \quad \varepsilon = \frac{\text{pJ}}{\text{MAC}}$$

> See [Chapter 7](./07_types_of_chips_taxonomy.md) for the full chip taxonomy and how each class positions on this axis.

| Chip | Year | Dense BF16 TFLOPS | Lowest-precision dense peak | TDP (W) | BF16 TFLOPS/W | Low-precision TOPS/W |
|---|---|---|---|---|---|---|
| NVIDIA A100 SXM4 | 2020 | 312 | 624 (INT8) | 400 | 0.78 | 1.6 |
| NVIDIA H100 SXM5 | 2022 | 989 | 1,979 (FP8) | 700 | 1.41 | 2.8 |
| NVIDIA B200 | 2024 | 2,250 | 9,000 (FP4) | 1,000 | 2.25 | 9.0 |
| NVIDIA B300 | 2025 | ~2,250 | ~15,000 (FP4) | 1,100–1,400 | ~1.6–2.0 | ~11–14 |
| AMD MI355X | 2025 | 2,500 | 10,000 (FP4) | 1,400 | 1.8 | 7.1 |
| Google TPU v4 | 2021 | 275 | 275 (INT8) | ~170–190 | ~1.5 | ~1.5 |
| Qualcomm Cloud AI 100 | 2020 | — | 400 (INT8) | 75 | — | 5.3 |

The BF16 column has improved only ~3× in five years; the low-precision column ~7–9×. Most of the efficiency gain since 2022 came from new number formats, not from the transistors.

**Quantization's role:** On tensor-core hardware, moving from FP16/BF16 to INT8 or FP8 roughly doubles peak throughput at nearly constant TDP, nearly doubling TOPS/W; FP4 doubles it again on chips that support it. This is why inference-optimized chips lean heavily on 8-bit and below — [Chapter 30](./30_datatypes_drive_optimization_choices.md).

---

## Thermals: Heat Flux and Why It Matters

Power dissipated in a chip turns entirely into heat. The fundamental problem is **heat flux** — watts per unit area:

$$\phi = \frac{P}{A} \quad (\text{W/cm}^2)$$

An H100 die is about 814 mm² = 8.14 cm². At 700 W:

$$\phi_{\text{H100}} = \frac{700}{8.14} \approx 86\,\text{W/cm}^2$$

For comparison, the fuel rods in a pressurized-water reactor run at roughly 50–100 W/cm² of surface heat flux, and the *surface* of the Sun radiates ~6,300 W/cm². A GPU is doing serious heat-flux engineering.

<div class="diagram">
<div class="diagram-title">Cooling Hierarchy — From Heatsink to Immersion</div>
<div class="flow">
  <div class="flow-node accent wide">Passive heatsink (fins)<br/><small>up to ~5 W/cm² — adequate for CPUs at 65 W TDP, not AI accelerators</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Forced-air cooling (fans + heatsink)<br/><small>up to ~15–25 W/cm² — workstation GPUs, PCIe server cards up to ~350 W</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">Direct liquid cooling (DLC) — cold plate on die<br/><small>up to ~150 W/cm² and beyond — standard for B200/B300-class racks; GB200/GB300 NVL72 is liquid-cooled by design</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Single-phase immersion (mineral oil)<br/><small>entire server submerged; excellent for density; 30–45°C operating fluid</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Two-phase immersion (dielectric boiling)<br/><small>fluid boils at chip, vapor condenses on coils; highest heat-flux capacity</small></div>
</div>
</div>

### The Thermal Resistance Chain

Heat flows from the die through a series of thermal resistances to the coolant. The **junction temperature** $T_j$ (temperature at the transistors) must stay below ~105°C for reliability:

$$T_j = T_{\text{coolant}} + P \cdot (\theta_{jc} + \theta_{ca})$$

where $\theta_{jc}$ (junction-to-case) is typically **~0.05–0.1 °C/W** for a large GPU die (managed by the TIM — thermal interface material, usually indium or liquid metal solder), and $\theta_{ca}$ (case-to-ambient) depends entirely on the cooling solution.

### Why H100/B200-Class Parts Require Liquid Cooling

- H100 SXM5 at 700 W: with a $\theta_{jc}$ of ~0.07 °C/W, the die sits ~50°C above its lid before any heatsink.
- Air cooling still works at this power, but only with very large heatsinks, high airflow and cool inlet air: the DGX H100 is an 8U air-cooled box for eight 700 W GPUs.
- At 1,000 W+ per GPU and 72 GPUs per rack, air runs out: a GB200/GB300 NVL72 rack dissipates ~120–130 kW, several times what an air-cooled rack can handle, so it is built around cold plates and a coolant distribution unit.
- Liquid also helps reliability and clocks: at 25–35°C coolant, $T_j$ stays far enough below the throttle point that the GPU can hold its boost clock under sustained load.

### Thermal Throttling

When $T_j$ approaches the maximum, the hardware **throttles**: the firmware gradually reduces boost clocks and voltage to keep temperature in bounds. A training job that observes unexpectedly low GPU utilization in `nvidia-smi` is often cooling-limited, not compute-limited. See [Chapter 13](./13_reliability_and_fault_tolerance.md) for reliability implications.

---

## The Energy Cost of AI

### Training Energy

A single large training run is an enormous energy event. Rough order-of-magnitude estimates for a 100B-parameter dense Transformer trained on ~1T tokens (Chinchilla-optimal scale):

$$E_{\text{train}} \approx N_{\text{GPU}} \times t_{\text{hours}} \times P_{\text{GPU}} \times \text{PUE} / 1000 \quad (\text{kWh})$$

```python
def training_energy_kwh(
    n_gpu: int,
    training_hours: float,
    gpu_tdp_w: float = 700,          # H100 SXM5
    pue: float = 1.2,                # typical modern DC; 1.0 = perfect; Google ~1.08
    utilization: float = 0.90,       # fraction of TDP, average
) -> dict:
    """Estimate training run energy in kWh and CO2e."""
    gpu_energy_kwh = n_gpu * training_hours * (gpu_tdp_w * utilization) / 1000
    total_dc_kwh = gpu_energy_kwh * pue          # includes cooling, networking
    co2e_kg = total_dc_kwh * 0.386               # US average grid: 0.386 kg CO2e/kWh (2024)
    return {
        "gpu_energy_kwh": round(gpu_energy_kwh),
        "total_dc_kwh": round(total_dc_kwh),
        "co2e_metric_tons": round(co2e_kg / 1000, 1),
    }

# GPT-4-class run (estimated): ~25k H100s for ~90 days
result = training_energy_kwh(n_gpu=25_000, training_hours=90 * 24)
print(result)
# {'gpu_energy_kwh': 34020000, 'total_dc_kwh': 40824000, 'co2e_metric_tons': 15758.1}
# ~34 GWh GPU energy, ~41 GWh at the meter; ~15,800 metric tons CO2e
```

For context: 34 GWh is roughly **the annual electricity consumption of 3,200 average US homes.** A single training run.

**Published estimates** (these involve significant uncertainty; reported figures vary):
- GPT-3 training (2020): ~1,290 MWh (Patterson et al., 2021)
- PaLM (2022, 540B params): estimated ~3,400 MWh
- Llama 3.1 405B (2024): 30.84M H100 GPU-hours (Meta's model card) → ~21,600 MWh at 700 W, before cooling and hosts  
- Frontier GPT-4-class runs: credibly in the range of 30,000–100,000 MWh

### Inference Energy per Token

Inference is lower power per run but enormously higher in aggregate (a popular model serves billions of requests). Energy per token depends on:

- Model size and batch size (affects GPU utilization and memory bandwidth)
- Hardware generation (newer = fewer mJ/token)
- Quantization (INT8/FP8 roughly halves energy vs FP16)

Batch size dominates everything. At batch 1 the GPUs burn near-full power streaming weights for a single token; at large batch the same weight read is shared by hundreds of sequences. Order-of-magnitude estimates (GPU power only; batch 1 from the bandwidth roofline at ~400 W per GPU, large batch from MLPerf-style throughput):

| Model | Hardware | Batch | Approx energy per output token |
|---|---|---|---|
| 7B (FP16) | 1× H100 | 1 | ~1–2 J |
| 7B (FP16) | 1× H100 | large (100+) | ~0.02–0.05 J |
| 70B (FP16/FP8) | 8× H100 | 1 | ~10–20 J |
| 70B (FP8) | 8× H100 | large (MLPerf offline: ~24k tokens/s) | ~0.2–0.3 J |

At $0.10/kWh, 0.25 J/token is about **$0.007 of electricity per million tokens**; at batch 1 (15 J/token) it is ~$0.40 per million. API prices of roughly $0.10–$15 per million tokens are therefore almost entirely hardware amortization and margin, not energy, which is why batching and utilization matter more to cost than the power bill.

### PUE: Datacenter Efficiency

**Power Usage Effectiveness (PUE)** = Total facility power / IT equipment power. A PUE of 1.0 is theoretical perfection; 1.2 means 20% overhead from cooling, lighting, UPS:

| PUE | What it means |
|---|---|
| 1.0 | Theoretical — all power reaches compute |
| 1.05–1.10 | Hyperscaler best-in-class (Google, Meta) |
| 1.15–1.25 | Modern AI-focused datacenters with DLC |
| 1.4–1.6 | Older air-cooled facilities |

At cluster scale, improving PUE from 1.5 → 1.2 on a 50 MW datacenter saves **15 MW** of overhead power — the equivalent of eliminating an entire server hall.

### Grid and Sustainability: The Binding Constraint

AI compute has become a **grid-scale energy consumer.** Since 2024:
- Hyperscalers and AI labs announced multi-gigawatt campuses (OpenAI's Stargate sites, Meta's Hyperion, Microsoft's Fairwater), and data centres used ~415 TWh worldwide in 2024 (IEA), projected to roughly double by 2030.
- A 1 GW AI campus runs at the power draw of a mid-size city (~750,000 homes).
- This has driven direct investment in nuclear (Microsoft's deal to restart Three Mile Island Unit 1, Google's Kairos and Amazon's X-energy small-reactor agreements, Meta's Clinton plant deal) and in on-site gas turbines, as well as renewable PPAs.

> **The implication for ML practitioners:** Energy is increasingly the binding economic and physical constraint — not raw FLOPS. This is why performance/watt (not TOPS) is what accelerator vendors now compete on, why efficient architectures (MoE, speculative decoding, quantization) are not academic niceties but economic necessities, and why hardware/software co-design is accelerating.

---

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Energy per token falls ~50× from batch 1 to a large batch on the same GPU. If you run a latency-sensitive product, how much of your fleet's power is spent keeping batch sizes small, and what would a 2× larger batch cost you in tail latency?</p>
</div>

## Python: Training-Run Energy Estimator

```python
import math

def full_training_estimator(
    model_params: float,            # billions of parameters
    tokens: float,                  # billions of training tokens
    flops_per_token_per_param: float = 6.0,  # ~6 FLOPs/token/param for dense Transformers
    hardware: str = "h100",
) -> dict:
    """
    Estimate training compute (PetaFLOPs), GPU-hours, energy (kWh), and CO2e.
    Uses Chinchilla FLOPs estimate: FLOPs ≈ 6 * N * D
    """
    configs = {
        "h100":  {"peak_tflops_bf16": 989,  "utilization": 0.45, "tdp_w": 700},
        "a100":  {"peak_tflops_bf16": 312,  "utilization": 0.40, "tdp_w": 400},
        "tpu_v4":{"peak_tflops_bf16": 275,  "utilization": 0.55, "tdp_w": 170},
    }
    cfg = configs[hardware]

    total_flops = flops_per_token_per_param * (model_params * 1e9) * (tokens * 1e9)
    effective_tflops = cfg["peak_tflops_bf16"] * 1e12 * cfg["utilization"]
    gpu_seconds = total_flops / effective_tflops
    gpu_hours = gpu_seconds / 3600

    energy_kwh = gpu_hours * cfg["tdp_w"] / 1000
    dc_kwh = energy_kwh * 1.2          # PUE 1.2
    co2e_tons = dc_kwh * 0.386 / 1000

    return {
        "total_petaflops": round(total_flops / 1e15),
        "gpu_hours": round(gpu_hours),
        f"{hardware}_count_for_30d": round(gpu_hours / (30 * 24)),
        "gpu_energy_kwh": round(energy_kwh),
        "datacenter_kwh": round(dc_kwh),
        "co2e_metric_tons": round(co2e_tons, 1),
    }

# Llama-2 70B scale: 70B params, 2T tokens
result = full_training_estimator(model_params=70, tokens=2000)
for k, v in result.items():
    print(f"  {k}: {v:,}")
# total_petaflops: 840,000,000     gpu_hours: 524,286     h100_count_for_30d: 728
# gpu_energy_kwh: 367,000          datacenter_kwh: 440,400    co2e_metric_tons: 170.0
# (Meta reported 1.72M A100-hours for Llama 2 70B; at A100 speed this estimator gives ~1.87M)
```

---

## Key Takeaways

- **Dynamic power** $P = \alpha C V^2 f$ is the master equation: every power-saving technique (DVFS, quantization, clock-gating) is a lever on one of its four terms.
- **Leakage grew with each node shrink** — it ended Dennard scaling (~2005) and remains 15–40% of total GPU power, motivating power-gating of idle units.
- **TDP vs actual power vs peak:** TDP is the cooling-system design target. Real workloads run 80–95% of TDP. Brief peaks above TDP are absorbed by capacitors. Insufficient cooling causes thermal throttling.
- **Power delivery is engineering at scale:** 1,000+ A into a GPU die at <1 V requires 48 V bus distribution, multi-phase VRMs, and fine-grained on-die voltage regulation to manage IR drop.
- **Performance per watt (TOPS/W)** — not raw TOPS — is the accelerator metric that matters for economics. Same-precision efficiency has grown ~3× in five years; new low-precision formats delivered most of the rest.
- **AI training is a grid-scale energy event.** GPT-4-class runs consume tens of GWh each; the inference tail is far larger in aggregate. PUE, grid mix, and energy efficiency are now first-order concerns for every AI team.
- Liquid cooling is the default for B200/B300-class racks: 700 W GPUs can still be air-cooled, but 1 kW+ GPUs at 72 per rack cannot. Understanding this shapes datacenter, rack, and cluster design decisions.

## Check Yourself

<details class="check">
<summary>A chip runs at 1.0 V and 2 GHz. If DVFS drops it to 0.8 V and 1.5 GHz, by how much does dynamic power fall?</summary>
<p>P ∝ V²f: (0.8² × 1.5) / (1.0² × 2.0) = 0.96 / 2.0 = 0.48, so about 52% lower, for 25% less clock.</p>
</details>

<details class="check">
<summary>How much current does a 1,000 W GPU draw at 0.8 V, and why do racks move to 48 V (and proposals to 800 V DC)?</summary>
<p>1,250 A at the die. Delivering the same power at higher voltage cuts current proportionally and resistive loss (I²R) by the square, so less copper and fewer conversion losses.</p>
</details>

<details class="check">
<summary>Estimate GPU-only training energy for 10,000 H100s running 60 days at 80% of 700 W.</summary>
<p>10,000 × 1,440 h × 0.56 kW ≈ 8.1 GWh; ~9.7 GWh at PUE 1.2.</p>
</details>

<details class="check">
<summary>Why is energy per token ~50× higher at batch 1 than at large batch on the same GPU?</summary>
<p>Decode is memory-bound: the GPU spends near-full power streaming all the weights for each step. At batch 1 that cost buys one token; at batch 100+ the same read serves every sequence in the batch.</p>
</details>

<details class="check">
<summary>Why can an H100 be air-cooled while a GB200 NVL72 rack cannot?</summary>
<p>At 700 W a large heatsink and high airflow keep the junction in range (DGX H100 does it in 8U). A 72-GPU rack at ~120–130 kW has far more heat per rack than air can carry away, so cold plates and liquid are required.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Carbon Emissions and Large Neural Network Training | Patterson et al., 2021 | [arxiv.org/abs/2104.10350](https://arxiv.org/abs/2104.10350) | GPT-3 and other training-energy estimates |
| Llama 3.1 model card | Meta, 2024 | [github.com/meta-llama/llama-models/blob/main/…](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) | GPU-hours and power for 405B |
| MLPerf Inference results | MLCommons | [mlcommons.org/benchmarks/inference-datacenter](https://mlcommons.org/benchmarks/inference-datacenter/) | Llama 2 70B throughput on 8× H100 |
| Energy and AI | International Energy Agency, 2025 | [iea.org/reports/energy-and-ai](https://www.iea.org/reports/energy-and-ai) | Data-centre electricity in 2024 and 2030 |
| NVIDIA 800 V HVDC architecture | NVIDIA, 2025 | [developer.nvidia.com/blog/nvidia-800-v-hvdc-a…](https://developer.nvidia.com/blog/nvidia-800-v-hvdc-architecture-will-power-the-next-generation-of-ai-factories/) | Rack power distribution for 1 MW racks |
| Power Usage Effectiveness | Google Data Centers | [google.com/about/datacenters/efficiency](https://www.google.com/about/datacenters/efficiency/) | Fleet PUE ~1.1 |
| FIVR — Fully Integrated Voltage Regulators on 4th Generation Intel Core SoCs | Burton et al., APEC 2014 | [doi.org/10.1109/APEC.2014.6803344](https://doi.org/10.1109/APEC.2014.6803344) | On-die voltage regulation |

---

**Next:** [Appendix E — Packaging & Advanced Integration](./appendix_e_packaging.md) — We look at how chiplets, 2.5D interposers and 3D stacking let GPUs keep growing even as individual dies hit the reticle limit.

[← Back to Table of Contents](./README.md)
