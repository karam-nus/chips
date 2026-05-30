---
title: "Chapter 1 — What Is a Chip?"
---

[← Back to Table of Contents](./README.md)

# Chapter 1 — What Is a Chip?

Before we can explain why `bf16` beats `fp16` for training, or why a 70-billion-parameter model won't fit on one GPU, we have to answer a more basic question: **what *is* a chip?** Not metaphorically — physically. What is the thing that PyTorch's `tensor.cuda()` ultimately runs on?

This chapter starts at the very bottom: a grain of sand. We'll build up through the transistor — the single most-manufactured object in human history — to the integrated circuit, and arrive at the two exponential curves (Moore's Law and Dennard scaling) whose rise and fall explain *everything* that follows in this guide, including why parallel hardware exists at all.

> **The one-sentence version:** A chip is billions of microscopic electrical switches (transistors) etched into a sliver of silicon, wired together so that the *patterns* in which they switch on and off perform computation.

---

## What a Chip Physically Is

A **chip** (or **integrated circuit**, IC, or **die**) is a small, flat piece of **semiconductor** — almost always **silicon** — onto which an enormous number of tiny electronic components, primarily **transistors**, have been fabricated and interconnected.

The vocabulary stack, from raw material to the thing in your server:

<div class="diagram">
<div class="diagram-title">From Sand to a Running Kernel</div>
<div class="flow-h">
  <div class="flow-node accent">Silicon<br/><small>(refined sand)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">Wafer<br/><small>300 mm disc</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">Die<br/><small>one chip</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">Package<br/><small>die + pins</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange">Board / GPU<br/><small>what you buy</small></div>
</div>
</div>

- **Silicon** is element 14 — the second-most-abundant element in Earth's crust, the main ingredient of sand (silicon dioxide, SiO₂). It is a **semiconductor**: its ability to conduct electricity sits *between* a conductor (copper) and an insulator (glass), and — crucially — that conductivity can be **controlled**. That controllability is the whole game.
- A **wafer** is a polished disc of ultra-pure crystalline silicon, today usually **300 mm** (≈12 inches) in diameter, sliced from a single grown crystal. Hundreds of chips are fabricated on one wafer simultaneously (see [Appendix A — How Chips Are Made](./appendix_a_how_chips_are_made.md)).
- A **die** is one individual chip cut from the wafer.
- A **package** encases the die, protects it, spreads its heat, and fans its microscopic connections out to pins/balls you can solder to a board.

When this guide says "a chip," we almost always mean **the die**: the active silicon where computation happens.

## Why Silicon? The Semiconductor

To understand the transistor, you need one idea: **you can change how well silicon conducts electricity by injecting impurities into it.** This is called **doping**.

Pure silicon has 4 outer (valence) electrons, each locked into a bond with a neighbor — a tidy crystal with few free charge carriers, so it barely conducts. Doping breaks that tidiness on purpose:

<div class="diagram">
<div class="diagram-title">Doping: Engineering Charge Carriers</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">n-type (negative)</div>
    <ul>
      <li>Dope with <strong>phosphorus/arsenic</strong> (5 valence electrons)</li>
      <li>One <strong>extra electron</strong> per dopant atom — free to move</li>
      <li>Majority carriers: <strong>electrons</strong> (−)</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">p-type (positive)</div>
    <ul>
      <li>Dope with <strong>boron</strong> (3 valence electrons)</li>
      <li>One <strong>missing electron</strong> — a "hole" — that moves</li>
      <li>Majority carriers: <strong>holes</strong> (+)</li>
    </ul>
  </div>
</div>
</div>

Put n-type and p-type silicon next to each other and you get a **junction** — the basis of diodes. Arrange three doped regions cleverly and you get the device that runs the modern world: the **transistor**.

## The Transistor: A Voltage-Controlled Switch

The transistor used in essentially every digital chip is the **MOSFET** (Metal-Oxide-Semiconductor Field-Effect Transistor). Conceptually, it is dead simple: **an electrical switch with no moving parts, flipped by a voltage.**

An **n-channel MOSFET (NMOS)** has three terminals plus a body:

- **Source** and **Drain** — two n-type regions in a p-type substrate (where current flows in/out).
- **Gate** — a conductor sitting just above the channel between source and drain, separated by an ultra-thin insulating layer of oxide.

```text
            GATE (control voltage)
              │
        ┌─────┴─────┐
        │   oxide   │   ← ultra-thin insulator (a few atoms thick!)
   ╔════╪═══════════╪════╗
   ║ n+ │  channel  │ n+ ║   ← apply +V to gate → channel turns conductive
   ║(S) │ (p-type)  │(D) ║
   ╚════╧═══════════╧════╝
        p-type substrate (body)
```

**How it switches:**

- **Gate voltage low (≈0 V):** the p-type channel separates source and drain. No current flows. The switch is **OFF** → logical **0**.
- **Gate voltage high (≈Vdd):** the gate's electric field pulls electrons into the channel, forming a conductive bridge ("inversion layer"). Current flows from drain to source. The switch is **ON** → logical **1**.

That is the atom of digital computing: **a voltage on the gate decides whether current flows.** No mechanical parts, switchable billions of times per second, and shrinkable to a few nanometers. A **PMOS** transistor is the mirror image (p-type source/drain in an n-type body) and conducts when the gate is *low*.

> **Why "field-effect"?** The gate never touches the channel — it's insulated by oxide. It controls the channel purely through its *electric field*. This is why the gate draws almost no steady-state current, which is what makes MOSFETs so energy-efficient to hold in a state.

### CMOS: Why Transistors Come in Pairs

Modern logic is **CMOS** (Complementary MOS): every gate uses NMOS and PMOS transistors *in complementary pairs*. The payoff is enormous: in a static CMOS gate, there is **(almost) no path from power to ground except during a switch**. Power is consumed mainly when bits *change*, not when they sit still. This single property is why a phone can hold billions of transistors and not melt.

The energy to switch a CMOS gate once is approximately:

$$E_{\text{switch}} \approx \tfrac{1}{2}\, C V_{dd}^{2}$$

where $C$ is the load capacitance and $V_{dd}$ is the supply voltage. **Remember this formula** — its quadratic dependence on voltage is the lever behind nearly every power-saving technique in this book, from low-voltage inference accelerators to [DVFS](./appendix_d_power_and_thermals.md).

## From One Transistor to Tens of Billions

A single transistor is a switch. Wire a handful together and you get a **logic gate** (Chapter 2). Wire millions of gates together and you get an **adder**, a **register**, a **core**. Wire billions together and you get a GPU.

The leap that made this economical was the **integrated circuit** (Jack Kilby & Robert Noyce, 1958–59): fabricate all the transistors *and* their wiring together on one piece of silicon, instead of soldering discrete parts. A modern chip is a layered 3D structure — transistors on the bottom, then **10–20 metal interconnect layers** stacked above to wire everything up.

<div class="diagram">
<div class="diagram-title">A Chip Is a Layered Structure (bottom → top)</div>
<div class="layer-stack">
  <div class="layer">Metal layers M1–M15+ — the "wiring," signal & power routing</div>
  <div class="layer">Contacts / vias — vertical connections between layers</div>
  <div class="layer">Transistor layer — billions of MOSFETs (the "devices")</div>
  <div class="layer">Silicon substrate — the crystalline wafer base</div>
</div>
</div>

How many transistors are we talking about? The growth is staggering — and the table below is your first lesson in *reading hardware history* (a skill we'll use constantly):

| Chip | Year | Transistors | Process node | Notes |
|------|:----:|:-----------:|:------------:|-------|
| Intel 4004 | 1971 | 2,300 | 10,000 nm | First commercial microprocessor |
| Intel 80486 | 1989 | 1.2 M | 1,000 nm | Early PC era |
| Intel Pentium 4 | 2000 | 42 M | 180 nm | The GHz race |
| NVIDIA G80 (first CUDA GPU) | 2006 | 681 M | 90 nm | GPGPU is born |
| Apple A14 (iPhone SoC) | 2020 | 11.8 B | 5 nm | A phone chip |
| NVIDIA H100 (Hopper GPU) | 2022 | 80 B | 4 nm (TSMC N4) | The AI workhorse |
| NVIDIA B200 (Blackwell) | 2024 | 208 B (2 dies) | 4 nm (TSMC 4NP) | Two dies, one package |

> Note the units: from **thousands** to **hundreds of billions** in ~50 years. That is roughly a **10⁸×** increase — the steepest sustained engineering trend in history. It has a name.

## The Two Curves That Explain Everything: Moore & Dennard

### Moore's Law — *how many* transistors

In 1965, Gordon Moore observed that the number of transistors economically placed on a chip was **doubling roughly every ~2 years**. This was never a law of physics — it was an *economic and engineering* trend, a self-fulfilling industry roadmap. It held for ~5 decades and is what filled the table above.

$$N_{\text{transistors}}(t) \approx N_0 \cdot 2^{\,t / 2}\quad (t \text{ in years})$$

More transistors per chip means you can spend them on: more cores, bigger caches, wider vector units, dedicated matrix-multiply hardware (tensor cores), and so on. **Every accelerator in Part III is, fundamentally, a different answer to the question: "now that we have billions of transistors, what should we build with them?"**

### Dennard Scaling — *how fast and how power-hungry*

Moore's Law's quieter, more important partner was **Dennard scaling** (1974): as transistors shrink, you can also lower their voltage, so **power density stays constant** while both *speed goes up* and *energy per operation goes down*. For ~30 years this let clock frequencies climb from MHz to GHz "for free."

<div class="diagram">
<div class="diagram-title">The End of the Free Lunch (~2005)</div>
<div class="flow-h">
  <div class="flow-node accent">Transistors keep shrinking<br/><small>(Moore's Law continues)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node red">But voltage stops dropping<br/><small>leakage current explodes</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange">Power density rises<br/><small>the "Power Wall"</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">Can't raise clock freq<br/><small>so… go parallel</small></div>
</div>
</div>

**Around 2005, Dennard scaling broke.** Transistors kept shrinking, but voltage couldn't keep dropping without leakage currents wasting too much power. Clock speeds stalled at a few GHz, where they remain today. The industry's response defines the modern era:

> If you can't make one core faster, use the extra transistors to build **more cores** — and design hardware that does **many operations in parallel**. This is the hinge on which the rest of this guide turns: it's why CPUs went multi-core, why GPUs (thousands of simple cores) became dominant, and ultimately **why deep learning — an embarrassingly parallel pile of matrix multiplies — found a perfect hardware match.** (We follow this thread explicitly in [Chapter 8](./08_parallel_processing.md) and [Chapter 27](./27_why_gpus_won_ai.md).)

### What a "Process Node" (5nm, 3nm) Actually Means

You'll see chips described by a **node**: "TSMC N5," "Intel 4," "3nm." Historically the number was a real physical feature size (the gate length). **Today it is essentially a marketing label** for a generation of manufacturing technology — "3nm" transistors are not literally 3 nm wide. What still holds: a smaller node generally means **more transistors per mm², lower energy per switch, and higher cost per wafer**. We dissect this in [Chapter 34](./34_generational_improvements.md) and [Appendix A](./appendix_a_how_chips_are_made.md).

## A Transistor as a Switch — in Code

To make the abstraction concrete, here is an NMOS/PMOS pair modeled as the Python switches they logically are, composed into a CMOS inverter (the simplest gate — output is the *opposite* of the input):

```python
def nmos(gate: int, source_when_on: int = 0) -> int | None:
    """N-channel MOSFET: conducts (passes source) when gate is HIGH (1)."""
    return source_when_on if gate == 1 else None   # None = high-impedance (open)

def pmos(gate: int, source_when_on: int = 1) -> int | None:
    """P-channel MOSFET: conducts when gate is LOW (0)."""
    return source_when_on if gate == 0 else None

def cmos_inverter(a: int) -> int:
    """
    PMOS pulls output HIGH (to Vdd=1) when input a=0.
    NMOS pulls output LOW  (to GND=0) when input a=1.
    Exactly one transistor conducts -> almost no power wasted.
    """
    pull_up   = pmos(gate=a, source_when_on=1)   # connects out to Vdd when a=0
    pull_down = nmos(gate=a, source_when_on=0)   # connects out to GND when a=1
    out = pull_up if pull_up is not None else pull_down
    return out

assert [cmos_inverter(a) for a in (0, 1)] == [1, 0]   # NOT gate ✓
```

This is the whole idea in miniature: **logic is what happens when you wire switches together.** Chapter 2 builds every gate, then an adder, from exactly this primitive. Note the data here is a single **bit** (`int` 0/1) — the indivisible unit of information a transistor stores. Eight bits make a **byte**; the floating-point numbers your model is made of are just structured bundles of these bits (Chapter 3 and Chapter 23).

## Why This Matters for Model Optimization

It may feel a long way from a MOSFET to a quantized LLM. It isn't. Three facts established here drive decisions you'll make weekly:

<div class="diagram">
<div class="diagram-title">First Principles → Your Daily Choices</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Transistors are finite</div>
    <div class="card-desc">A chip has a fixed transistor & area budget. Spending them on FP32 units vs many INT8 units is a real trade-off — and it's why low precision gets more throughput (Ch 24–25).</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Switching costs energy</div>
    <div class="card-desc">E ≈ ½CV². Fewer/smaller bits switched = less energy. This is the physical reason quantization saves power, not just memory (Ch 25).</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">The free lunch ended</div>
    <div class="card-desc">Dennard's end forced parallelism. Your model runs fast only because it maps onto thousands of parallel units — which is why it must be expressed as big matmuls (Ch 27).</div>
  </div>
</div>
</div>

## Key Takeaways

- A **chip** is billions of **transistors** (voltage-controlled switches) fabricated in **silicon** and wired together by metal layers. Computation is the *pattern* of switching.
- The **MOSFET** switches based on a gate voltage; **CMOS** pairs make it so power is spent mostly when bits *change* ($E \approx \tfrac{1}{2}CV_{dd}^2$).
- **Moore's Law** (more transistors) gave us the budget for ever-bigger compute; **Dennard scaling** (free speed/efficiency) gave us frequency — until it **ended around 2005**.
- Dennard's end forced the pivot to **parallelism**, which is the root cause of GPUs, accelerators, and the whole hardware story behind modern AI.
- Every optimization in this guide (quantization, sparsity, parallelism) is ultimately about spending a **finite transistor, energy, and bandwidth budget** wisely.

---

*Next: [Chapter 2 — Transistors to Logic](./02_transistors_to_logic.md), where we turn these switches into gates, adders, and the arithmetic units that actually multiply your matrices.*

[← Back to Table of Contents](./README.md)
