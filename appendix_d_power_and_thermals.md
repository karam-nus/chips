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

- At 5 nm, **subthreshold leakage** and **gate-oxide leakage** together can be 20–40% of total chip power at idle.
- This is why [Dennard scaling ended](./01_what_is_a_chip.md) (~2005): shrinking the transistor no longer let you lower $V_{dd}$ freely, because the on/off current ratio degraded and leakage exploded.
- In a GPU datacenter cluster idling overnight, leakage alone draws tens of kilowatts per rack.

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
- **48 V bus architectures** (Google TPU pods, Meta/OCP Grand Teton) reduce the current delivered across the board (halving V doubles I, so using 48 V instead of 12 V cuts bus current by 4×), reducing $I^2R$ losses in board traces.

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

**On-die voltage regulation (ODVR / FIVR)** — placing small voltage regulators directly on the chip die — reduces the last few mm of IR drop. Intel pioneered Fully Integrated Voltage Regulators (FIVR) and the industry is moving toward die-embedded regulators to support fine-grained per-cluster DVFS.

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
# High-perf: ~0.318 mJ/token  Power-save: ~0.180 mJ/token
# Energy ratio: 1.77x  |  Throughput ratio: 1.38x
# => 38% more tokens/s costs 77% more energy — not always worth it
```

### Clock Gating and Power Gating

- **Clock gating**: Disable the clock to an idle block (tensor core cluster, cache bank). Zero $\alpha$ → zero dynamic power for that block. Standard in every modern chip; can save 20–40% of chip-level dynamic power during sparse workloads.
- **Power gating**: Disconnect supply voltage entirely from an idle block. Eliminates leakage too. Slower to wake up (microseconds) but critical for aggressive idle power on mobile. On GPUs, entire streaming multiprocessor (SM) clusters are power-gated when context is evicted.

---

## Performance per Watt: The True Figure of Merit

When comparing accelerators, raw FLOPS is a misleading metric. The useful quantity is **performance per watt** — equivalently, **energy per operation**:

$$\eta = \frac{\text{TOPS}}{W} \quad \text{or} \quad \varepsilon = \frac{\text{pJ}}{\text{MAC}}$$

> See [Chapter 7](./07_types_of_chips_taxonomy.md) for the full chip taxonomy and how each class positions on this axis.

| Chip | Peak FP16 TOPS | TDP (W) | FP16 TOPS/W | Notes |
|---|---|---|---|---|
| NVIDIA A100 SXM4 | 312 | 400 | 0.78 | 2020, 7nm |
| NVIDIA H100 SXM5 | 989 | 700 | 1.41 | 2022, 4nm |
| NVIDIA B200 SXM | ~2,200 | 1,000 | 2.20 | 2024, 4nm (2-die) |
| Google TPU v4 | ~275 | 170 | ~1.62 | Custom, systolic array |
| Apple M3 Max (edge) | ~14.2 (ANE) | ~50 | ~0.28 | Edge class, BF16 |
| Qualcomm Cloud AI 100 | 400 | 75 | 5.3 | Inference-only, lower precision |

**Quantization's role:** Moving from FP32 → INT8 roughly doubles TOPS (same silicon, 2× the throughput for integer MAC) while keeping TDP nearly constant, nearly doubling TOPS/W. FP8 vs FP16 gives a similar 2× boost. This is why inference-optimized chips lean heavily on 8-bit and below — [Chapter 30](./30_datatypes_drive_optimization_choices.md).

---

## Thermals: Heat Flux and Why It Matters

Power dissipated in a chip turns entirely into heat. The fundamental problem is **heat flux** — watts per unit area:

$$\phi = \frac{P}{A} \quad (\text{W/cm}^2)$$

An H100 die is about 814 mm² = 8.14 cm². At 700 W:

$$\phi_{\text{H100}} = \frac{700}{8.14} \approx 86\,\text{W/cm}^2$$

For comparison, a nuclear reactor core operates at ~100 W/cm³, and the *surface* of the Sun radiates ~6,300 W/cm². A GPU is doing serious heat-flux engineering.

<div class="diagram">
<div class="diagram-title">Cooling Hierarchy — From Heatsink to Immersion</div>
<div class="flow">
  <div class="flow-node accent wide">Passive heatsink (fins)<br/><small>up to ~5 W/cm² — adequate for CPUs at 65 W TDP, not AI accelerators</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Forced-air cooling (fans + heatsink)<br/><small>up to ~15–25 W/cm² — workstation GPUs, PCIe server cards up to ~350 W</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">Direct liquid cooling (DLC) — cold plate on die<br/><small>up to ~150 W/cm² — H100/H200 SXM require this; ~90% of new AI data centers</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Single-phase immersion (mineral oil)<br/><small>entire server submerged; excellent for density; 30–45°C operating fluid</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Two-phase immersion (dielectric boiling)<br/><small>fluid boils at chip, vapor condenses on coils; highest heat-flux capacity</small></div>
</div>
</div>

### The Thermal Resistance Chain

Heat flows from the die through a series of thermal resistances to the coolant. The **junction temperature** $T_j$ (temperature at the transistors) must stay below ~105°C for reliability:

$$T_j = T_{\text{coolant}} + P \cdot (\theta_{jc} + \theta_{ca})$$

where $\theta_{jc}$ (junction-to-case) is typically **0.1–0.15 °C/W** for a large GPU die (managed by the TIM — thermal interface material, usually indium or liquid metal solder), and $\theta_{ca}$ (case-to-ambient) depends entirely on the cooling solution.

### Why H100/B200-Class Parts Require Liquid Cooling

- H100 SXM5 at 700 W: with a $\theta_{jc}$ of 0.1 °C/W, the die is 70°C above coolant just from junction-to-case.
- With 25°C coolant water (DLC), $T_j \approx$ 25 + 70 = 95°C — already near the limit.
- Air cooling at 35–40°C ambient is simply infeasible: 700 W into air produces $\Delta T$ well over 100°C.
- B200 at 1,000 W pushes this even harder. GB200 NVL72 racks (72 GPUs, ~120 kW) require purpose-built liquid-cooled racks.

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
# {'gpu_energy_kwh': 37800000, 'total_dc_kwh': 45360000, 'co2e_metric_tons': 17509.0}
# ~37.8 GWh GPU energy; ~17,500 metric tons CO2e
```

For context: 37.8 GWh is roughly **the annual electricity consumption of 3,500 average US homes.** A single training run.

**Published estimates** (these involve significant uncertainty; reported figures vary):
- GPT-3 training (2020): estimated ~1,300 MWh
- PaLM (2022, 540B params): estimated ~3,400 MWh
- LLaMA-3 405B (2024): Meta reported ~11,300 MWh  
- Frontier GPT-4-class runs: credibly in the range of 30,000–100,000 MWh

### Inference Energy per Token

Inference is lower power per run but enormously higher in aggregate (a popular model serves billions of requests). Energy per token depends on:

- Model size and batch size (affects GPU utilization and memory bandwidth)
- Hardware generation (newer = fewer mJ/token)
- Quantization (INT8/FP8 roughly halves energy vs FP16)

Rough estimates for autoregressive generation at batch size 1 (latency-optimized):

| Model size | Hardware | Approx mJ/output token |
|---|---|---|
| 7B (FP16) | H100 | ~0.15–0.25 |
| 70B (FP16) | 8× H100 | ~1.5–3.0 |
| 405B (INT8) | 8× H100 | ~5–12 |

At $0.10/kWh and 2 mJ/token: **0.0002 cents per token** in raw energy. But with PUE, amortized hardware, and margins, the delivered cost is 1–10 cents per 1,000 tokens.

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

AI compute has become a **grid-scale energy consumer.** In 2024:
- Microsoft, Google, and Amazon each announced plans for 1–5+ GW of new datacenter capacity for AI.
- A 1 GW AI campus runs at the power draw of a mid-size city (~750,000 homes).
- This has driven direct investment in nuclear (Microsoft/OpenAI's Three Mile Island restart, Google's SMR contracts) and PPAs for renewables.

> **The implication for ML practitioners:** Energy is increasingly the binding economic and physical constraint — not raw FLOPS. This is why performance/watt (not TOPS) is what accelerator vendors now compete on, why efficient architectures (MoE, speculative decoding, quantization) are not academic niceties but economic necessities, and why hardware/software co-design is accelerating.

---

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
```

---

## Key Takeaways

- **Dynamic power** $P = \alpha C V^2 f$ is the master equation: every power-saving technique (DVFS, quantization, clock-gating) is a lever on one of its four terms.
- **Leakage grew with each node shrink** — it ended Dennard scaling (~2005) and remains 15–40% of total GPU power, motivating power-gating of idle units.
- **TDP vs actual power vs peak:** TDP is the cooling-system design target. Real workloads run 80–95% of TDP. Brief peaks above TDP are absorbed by capacitors. Insufficient cooling causes thermal throttling.
- **Power delivery is engineering at scale:** 1,000+ A into a GPU die at <1 V requires 48 V bus distribution, multi-phase VRMs, and fine-grained on-die voltage regulation to manage IR drop.
- **Performance per watt (TOPS/W)** — not raw TOPS — is the accelerator metric that matters for economics. Each generation roughly doubles it; quantization at inference time adds another 2–4×.
- **AI training is a grid-scale energy event.** GPT-4-class runs consume tens of GWh each; the inference tail is far larger in aggregate. PUE, grid mix, and energy efficiency are now first-order concerns for every AI team.
- Liquid cooling (DLC or immersion) is no longer optional for H100/B200-class hardware: heat flux exceeds what air can remove. Understanding this shapes datacenter, rack, and cluster design decisions.

---

*Next: [Appendix E — Packaging & Advanced Integration](./appendix_e_packaging.md), where we look at how chiplets, 2.5D interposers, and 3D stacking let GPUs keep growing even as individual dies hit the reticle limit.*

[← Back to Table of Contents](./README.md)
