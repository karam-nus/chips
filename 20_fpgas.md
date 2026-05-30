---
title: "Chapter 20 — FPGAs (Field-Programmable Gate Arrays)"
---

[← Back to Table of Contents](./README.md)

# Chapter 20 — FPGAs (Field-Programmable Gate Arrays)

Every chip in this guide so far has been **fixed at manufacture** — the silicon is etched, and its function never changes. The transistors that implement your GPU's tensor cores are always tensor cores. An ASIC is always the ASIC it was designed to be. But there is a class of chip that breaks this rule: after it is manufactured, you can reach in and **completely rewire its internal logic** using a configuration file. That chip is the **FPGA**.

FPGAs occupy a unique position in the hardware landscape. They are slower than ASICs at any given task, more expensive per unit than a GPU, and harder to program than either. And yet they are irreplaceable — in telecom base stations, in trading firms that need sub-microsecond latency, in aerospace hardware that must be updated on the fly, in data centers running Microsoft's Bing search index, and in chip design itself, where every ASIC is emulated on FPGAs before it is committed to silicon. For the ML practitioner, FPGAs matter most in three scenarios: **ultra-low-latency inference**, **custom quantized dataflows** that no GPU/ASIC supports yet, and **hardware emulation** of new accelerator designs before tape-out.

> **The one-sentence version:** An FPGA is a chip full of configurable logic blocks and programmable interconnect wires whose function is determined by a bitstream you load at runtime — making it infinitely (within its resource limits) reconfigurable, at the cost of being slower and less efficient than a purpose-built chip for any particular task.

---

## What Problem Does an FPGA Solve?

The tension FPGAs resolve is between **flexibility** and **efficiency**. Consider the extremes:

<div class="diagram">
<div class="diagram-title">The Hardware Flexibility–Efficiency Spectrum</div>
<div class="flow-h">
  <div class="flow-node accent">CPU<br/><small>fully flexible, 100–500 GFLOPS, ~200 W</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">GPU<br/><small>data-parallel flex, ~1000 TOPS INT8, ~700 W</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">FPGA<br/><small>reconfigurable, ~10–100 TOPS INT8, 25–300 W</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">ASIC<br/><small>fixed-function, ~100–1000 TOPS INT8, 10–300 W</small></div>
</div>
</div>

An FPGA wins when you need **a custom dataflow** that no standard processor implements efficiently AND you need it **without the NRE cost and lead time** of an ASIC (which can be $10–50M+ and 12–24 months; see [Chapter 21](./21_asics.md)). The FPGA is deployed in hours; the bitstream is updated weekly if needed.

---

## FPGA Building Blocks

An FPGA chip is built from a **regular 2D array** of primitive resources connected by a programmable routing fabric. Understanding those resources is the key to understanding FPGAs.

### Look-Up Tables (LUTs)

The **LUT** is the atom of an FPGA. An $n$-input LUT is literally a **tiny SRAM** with $2^n$ entries. You supply $n$ input bits as an address; the LUT outputs the stored bit at that address.

A 4-input LUT (LUT4) has $2^4 = 16$ SRAM cells. By programming those 16 cells you can implement **any** Boolean function of 4 variables:

```text
LUT4 configured as a 2-input AND gate (ignoring inputs C, D):
Truth table stored in LUT4 SRAM:
  A=0,B=0 → 0   address 0b0000 → 0
  A=0,B=1 → 0   address 0b0010 → 0
  A=1,B=0 → 0   address 0b0100 → 0
  A=1,B=1 → 1   address 0b0110 → 1
  (all entries with C=1 or D=1 → 0 as well)
```

Modern FPGAs use **LUT6** (6-input, $2^6 = 64$ SRAM cells), which can implement any 6-variable Boolean function — or be split into two LUT5s for simpler functions. The programming cost for a LUT6: **64 bits of configuration SRAM**.

```python
class LUT6:
    """Simulate a 6-input LUT — the atom of FPGA logic."""
    def __init__(self, truth_table: int):
        # truth_table is a 64-bit integer; bit i is the output when inputs == i
        assert 0 <= truth_table < (1 << 64)
        self.table = truth_table
    
    def __call__(self, a: int, b: int, c: int, d: int, e: int, f: int) -> int:
        idx = (f<<5) | (e<<4) | (d<<3) | (c<<2) | (b<<1) | a
        return (self.table >> idx) & 1

# Configure as a 3-input majority function: out=1 if ≥2 of {a,b,c} are 1
MAJ3_TABLE = 0
for i in range(64):
    a, b, c = i&1, (i>>1)&1, (i>>2)&1
    if a + b + c >= 2:
        MAJ3_TABLE |= (1 << i)

maj3 = LUT6(MAJ3_TABLE)
assert maj3(1, 1, 0, 0, 0, 0) == 1   # majority of 1,1,0 → 1
assert maj3(1, 0, 0, 0, 0, 0) == 0   # majority of 1,0,0 → 0
```

> **The insight:** because a LUT is just SRAM, implementing an AND gate costs exactly the same silicon resources as implementing a CRC-32 bit, or an AES S-box entry lookup. The LUT is **function-agnostic**. This is the root of FPGA flexibility.

### Flip-Flops (Registers)

Each LUT is paired with one or more **D flip-flops** (DFFs) — a 1-bit register clocked on a rising edge. Together, LUT + DFF is the core of a **logic slice**: the LUT computes combinational logic; the DFF optionally registers the output for pipelined designs.

Without registers, long combinational chains would violate timing (too slow). Registering at each stage allows the FPGA to be **deeply pipelined** — a key technique for achieving high throughput at lower clock frequencies (e.g., 500 MHz on FPGA vs 3 GHz on a CPU, but with 256 pipeline stages in flight simultaneously).

### Configurable Logic Blocks (CLBs) / Logic Slices

LUTs and flip-flops are grouped into **CLBs** (Xilinx/AMD terminology) or **Logic Elements** (Intel/Altera). A Xilinx UltraScale+ CLB contains:
- **8 LUT6s** (each splittable into 2 LUT5s → up to 16 LUT5-equivalent functions)
- **16 flip-flops**
- **Carry-chain logic** for fast arithmetic (ripple carry for adders, comparators)
- **MUX4/MUX8** built from the LUTs for wide multiplexing

A large FPGA (e.g., Xilinx Alveo U280) has **1.3 million LUTs** and **2.6 million flip-flops** on a single chip.

### Programmable Routing / Interconnect

LUTs and flip-flops are useless unless you can wire them together. The **routing fabric** is a hierarchical mesh of wires and **programmable switch matrices** — SRAM-controlled pass transistors that connect or disconnect wire segments. The routing configuration consumes 70–80% of the total FPGA configuration bits.

Routing is the FPGA's perennial bottleneck: moving a signal from one side of a large FPGA to the other can add 3–5 ns of delay (wire + switch resistance/capacitance), limiting the maximum clock frequency — typically **300–700 MHz** for typical designs, versus 3–5 GHz for a CPU.

### DSP Slices: Hardened Multiply-Accumulate

Implementing a 27×18-bit multiplier in LUTs requires ~200 LUTs and has poor timing. So FPGAs include **DSP slices** — hard, optimized multiplier + accumulator circuits that implement:

$$P = A \times B + C \quad (A: 27\text{-bit},\ B: 18\text{-bit},\ P: 48\text{-bit\ acc})$$

at **up to 891 MHz** on Xilinx UltraScale+ (faster than LUT-based implementations). A single DSP slice does $2 \times 891 \times 10^6 = 1.78$ GOPS at 27-bit precision per cycle.

| FPGA | DSP Slices | Peak INT8 TOPS (using DSP) | Peak BF16 TOPS |
|---|---|---|---|
| Xilinx Alveo U250 | 12,288 | ~10 TOPS | ~4 TOPS |
| Xilinx Alveo U280 | 9,024 | ~7 TOPS | ~3 TOPS |
| Intel Stratix 10 NX | 8,192 | ~15 TOPS | ~5 TOPS |
| AMD Versal Premium | 2,880 DSP58 + 400 AI Engines | ~100 TOPS | ~58 TOPS |
| AMD Versal AI Core | 1,968 DSP58 + 128 AI Engines | ~58 TOPS | ~34 TOPS |

> **AI Engines** (AMD Versal): 1 GHz VLIW+SIMD cores, separate from LUTs, dedicated to floating-point and INT16/INT8 vector computation. A sign that FPGA vendors are hybridizing with ASIC-like structures to close the performance gap for ML.

### Block RAM (BRAM)

Large lookup tables, weight buffers, and FIFO queues can't be built from LUTs efficiently. FPGAs include **block RAMs (BRAMs)** — hard SRAM macros of 18 Kb or 36 Kb each, clocked at FPGA frequency:

- Xilinx Alveo U280: **2,016 BRAMs × 36 Kb = 72 Mb** of BRAM = **9 MB**
- Bandwidth: 2 ports × 512 bits × 700 MHz = **716 GB/s aggregate** (across all BRAMs simultaneously)

This per-BRAM bandwidth is huge compared to DDR (77 GB/s for LPDDR5). ML inference designs pack weight tensors into BRAM to avoid DRAM bandwidth bottlenecks — at the cost of fitting only small models (9 MB is ~9M INT8 params; a ViT-S fits; GPT-2 117M doesn't).

### UltraRAM (URAM)

AMD UltraScale+ devices add **URAMs** — 288 Kb (36 KB) blocks with 72-bit ports, optimized for large deep-memory (storage, not bandwidth). An Alveo U280 has 960 URAMs = 34.5 MB of URAM, useful for buffering activation tensors.

### Hardened IP Blocks

FPGAs also embed **fixed-function IP** that would be too costly in LUTs:
- **PCIe Gen4/5 controllers** (hard transceiver + controller logic)
- **CMAC / Ethernet 100G/400G** transceivers
- **HBM2 controllers** (Alveo U280 has 2 × HBM2 stacks for 460 GB/s)
- **DDR4/LPDDR4 PHYs**
- **ARM Cortex-A cores** (in SoC-FPGAs like Xilinx Zynq, Intel Cyclone)

The HBM2 on a U280 is particularly relevant: 460 GB/s bandwidth at < 10W for the HBM, enabling FPGA inference designs that are **bandwidth-limited** rather than compute-limited — the ideal operating point for transformer decode.

---

## SoC-FPGAs and AI Engines

Modern FPGAs blur into **SoC-FPGAs** — chips that combine an FPGA fabric with ARM cores and hardened DSP clusters:

<div class="diagram">
<div class="diagram-title">Evolution of the FPGA Chip</div>
<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">1985</div>
    <div class="timeline-desc"><strong>Pure fabric FPGA</strong> — Xilinx XC2064: 64 CLBs, programmable I/O only</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2001</div>
    <div class="timeline-desc"><strong>Hard multipliers</strong> — Virtex-II: 18×18 multipliers embedded; ML arithmetic feasible</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2010</div>
    <div class="timeline-desc"><strong>SoC-FPGA</strong> — Xilinx Zynq-7000: dual ARM Cortex-A9 + FPGA fabric on one die</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2016</div>
    <div class="timeline-desc"><strong>HBM FPGA</strong> — Xilinx Alveo with HBM2; 460 GB/s; AI inference at scale</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2019</div>
    <div class="timeline-desc"><strong>Versal ACAP</strong> — AMD/Xilinx: FPGA + ARM + AI Engines (VLIW-SIMD at 1 GHz); hybrid accelerator</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2023</div>
    <div class="timeline-desc"><strong>Versal Premium / AI Edge</strong> — up to 400 AI Engines; ~58 TOPS INT8; power ~50–150 W</div>
  </div>
</div>
</div>

---

## How You Program an FPGA

FPGA programming is a radically different mental model from CPU/GPU programming. You are not writing instructions — you are **describing hardware**.

### The Toolchain: HDL → Bitstream

```text
You write:         RTL (Verilog / VHDL / SystemVerilog)
        ↓
Synthesis:         Map RTL to FPGA primitives (LUTs, DSPs, BRAMs)
                   Tool: Vivado (AMD), Quartus (Intel), Synplify
        ↓
Place & Route:     Assign primitives to physical CLB/DSP/BRAM locations
                   Route wires between them through switch matrices
                   Tool: Vivado (AMD), Quartus (Intel)
        ↓
Timing Analysis:   Verify all paths meet setup/hold at target frequency
        ↓
Bitstream:         Binary file (~100 MB–1 GB) encoding all SRAM cells
        ↓
Device load:       JTAG or SPI flash → configures FPGA in seconds
```

A tiny example — a 2:1 multiplexer in Verilog:

```verilog
// A 1-bit 2:1 MUX: out = sel ? b : a
// This will be implemented by one LUT in the FPGA
module mux2 (
    input  wire sel,
    input  wire a,
    input  wire b,
    output wire out
);
    assign out = sel ? b : a;
endmodule
```

For a full ML accelerator, the Verilog would describe: input FIFO → weight fetch FSM → systolic array of DSP slices → output DMA controller — thousands of lines, but still **hardware description**, not imperative code. See [Chapter 6](./06_languages_and_nomenclature.md) for a deeper dive into HDL and RTL, and [Chapter 31](./31_chip_design_flow.md) for the full design flow.

### High-Level Synthesis (HLS)

Writing RTL by hand for complex ML kernels is slow. **HLS** (High-Level Synthesis) tools compile C/C++ (or Python-subset) into RTL:

- **Xilinx/AMD Vitis HLS**: C++ with `#pragma HLS PIPELINE`, `#pragma HLS UNROLL` → Verilog
- **Intel HLS Compiler**: C++ with `[[intel::kernel_args_restrict]]` annotations
- **Bambu / LegUp**: open-source HLS research tools
- **Keras2Chip / hls4ml**: Keras model → HLS → FPGA bitstream (CERN's hls4ml is widely used in physics experiments and edge ML)

```cpp
// HLS C++ sketch: pipelined dot product on FPGA
// After synthesis, DSP slices implement the MACs; BRAM holds weights
#include <ap_fixed.h>   // Xilinx arbitrary-precision fixed-point

typedef ap_fixed<8,2> data_t;    // 8-bit fixed, 2 integer bits
typedef ap_fixed<24,6> accum_t;  // 24-bit accumulator

accum_t dot_product(data_t a[128], data_t w[128]) {
    #pragma HLS PIPELINE II=1       // one output per clock cycle
    #pragma HLS ARRAY_PARTITION variable=a complete
    #pragma HLS ARRAY_PARTITION variable=w complete
    accum_t acc = 0;
    for (int i = 0; i < 128; i++) {
        acc += a[i] * w[i];
    }
    return acc;
}
// After synthesis: 128 DSP slices run in parallel, fully pipelined.
// At 500 MHz: 128 × 500M × 2 = 128 GOPS from this one function.
```

---

## FPGA vs CPU vs GPU vs ASIC Tradeoffs

| Property | CPU | GPU | FPGA | ASIC |
|---|---|---|---|---|
| Flexibility | Fully flexible | Moderate (fixed ISA) | Reconfigurable | Fixed forever |
| Reconfigure time | N/A (s/w) | N/A (s/w) | Seconds (bitstream load) | Never |
| Clock frequency | 3–5 GHz | 2–3 GHz | 300–750 MHz | 1–3 GHz |
| INT8 TOPS per chip | 0.5–5 | 500–1000 | 7–100 | 50–2000 |
| Power (peak) | 15–400 W | 300–1000 W | 25–300 W | 10–400 W |
| Latency determinism | Poor (OS, branch) | Poor (scheduling) | **Excellent** (cycle-exact) | Excellent |
| NRE cost | $0 | $0 | $0 | $10M–$500M |
| Unit cost | $100–$10K | $2K–$30K | $500–$30K | $0.10–$1K |
| Development time | Days | Days | Weeks–months | 12–24 months |
| Ideal batch size | Any | Large (>16) | **1** (low latency) | Any |

The **latency determinism** row is the key FPGA differentiator for ML. A GPU has stochastic scheduling jitter; an FPGA pipeline has **cycle-exact latency** — once the pipeline is filled, every inference takes *exactly* N clock cycles, guaranteed. For algorithmic trading, autonomous vehicle control, or network packet processing (where microsecond jitter causes failures), this is not a nice-to-have but a hard requirement.

---

## AI on FPGAs

### Microsoft Project Brainwave / Azure FPGA

Starting in 2017, Microsoft deployed Intel Stratix 10 FPGAs in Azure data centers, running a **hardwired NPU** implemented in FPGA fabric. Brainwave's FPGA NPU processes requests at **single-digit millisecond latency** for real-time Bing search ranking — a latency target GPUs cannot reliably hit due to scheduling jitter. The design uses **INT8 and bfloat16 weights**, custom variable-precision arithmetic units built from DSP slices, and on-chip DRAM to avoid PCIe round-trips.

> **The key insight:** for small-batch or batch=1 inference where GPU occupancy is low and scheduling overhead matters, an FPGA pipeline can **outperform a GPU on latency** while using less power.

### Custom Quantized Dataflows

FPGAs can implement **arbitrary bit-width arithmetic** that no CPU or GPU supports natively:

- **INT4** (4-bit weights × 8-bit activations): pack two INT4 weights into one DSP B-input (8-bit), compute two MACs with one DSP slice. 2× throughput vs INT8.
- **Ternary / binary weights** (-1, 0, +1 or {-1, +1}): replace multiplications with additions/subtractions, implementable in LUTs with near-zero DSP cost.
- **Block floating point** (a la MXFP; see [Chapter 23](./23_number_formats_deep_dive.md)): implement custom scale + mantissa arithmetic matched to your model's dynamic range.

None of these are available on a stock GPU (which has INT8 tensor cores but not INT4 native in most generations, and no ternary path). An FPGA lets you tailor the multiplier width to exactly the model's quantization scheme.

```python
# Demonstrate ternary weight inference — what an FPGA would compute in LUTs
import numpy as np

def ternary_matmul(A: np.ndarray, W_ternary: np.ndarray) -> np.ndarray:
    """
    A:         (batch, in_features)   float32
    W_ternary: (out_features, in_features) values in {-1, 0, +1}
    
    On FPGA: W is stored as 2-bit values; mul is replace/discard/negate — no DSP needed.
    """
    assert np.all(np.isin(W_ternary, [-1, 0, 1])), "Weights must be ternary"
    return A @ W_ternary.T   # CPU fallback; FPGA does this with adder tree

# Model with ternary weights: 95% of the multiplier area eliminated
W = np.random.choice([-1, 0, 1], size=(256, 512), p=[0.1, 0.8, 0.1])
A = np.random.randn(1, 512).astype(np.float32)
out = ternary_matmul(A, W)   # shape: (1, 256)
```

### FPGA for ASIC Prototyping and Hardware Emulation

Before committing $20M+ to an ASIC tape-out, chip designers **emulate their RTL on an FPGA farm**. A large SoC might require 10–50 FPGA chips networked together to emulate at 1–10 MHz (much slower than the target 1 GHz ASIC, but fast enough to run software stacks and find bugs). Companies like **Cadence** (Palladium), **Synopsys** (ZeBu), and **Mentor** sell multi-million-dollar FPGA emulation platforms. Every major AI chip (Google TPU, NVIDIA Hopper, Apple M-series) was prototyped on FPGAs before tape-out (see [Chapter 31](./31_chip_design_flow.md) and [Chapter 32](./32_chip_verification_and_validation.md)).

---

## The FPGA Ecosystem

| Vendor | Key FPGA Families | Notes |
|---|---|---|
| AMD (acquired Xilinx 2022) | Artix, Kintex, Virtex, Versal, Alveo | Market leader; Vivado toolchain |
| Intel (acquired Altera 2015) | Cyclone, Arria, Stratix | Quartus toolchain; used in Azure Brainwave |
| Lattice | ECP5, Crosslink, Nexus | Low-power, edge; used in automotive and IoT |
| Microchip (acquired Microsemi) | SmartFusion, PolarFire | Radiation-tolerant; space, defense |
| Achronix | Speedster7t | HBM-enabled; ML-focused |

AMD/Intel have ~90% of the high-performance FPGA market.

---

## Why This Matters for Model Optimization

<div class="diagram">
<div class="diagram-title">When to Consider an FPGA for ML</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Ultra-Low Latency Inference</div>
    <div class="card-desc">Batch=1, &lt;1 ms requirements. FPGAs give cycle-exact deterministic pipelines that GPUs cannot match due to scheduling jitter. Typical: trading, real-time control, network function.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Custom Bit-Width Arithmetic</div>
    <div class="card-desc">Your model uses INT4, ternary, or block-float formats that GPU tensor cores don't natively support. FPGAs implement arbitrary-width MACs in DSP slices or LUTs.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Pre-ASIC Prototyping</div>
    <div class="card-desc">Validate your NPU or accelerator RTL at functional speed before committing to tape-out. Every AI chip starts here. Saves millions in re-spin costs.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Low Volume / Short Lifetime</div>
    <div class="card-desc">If you need &lt;100K units and/or expect the algorithm to change, FPGA's zero NRE beats the ASIC economics every time (see Ch 21 cost crossover analysis).</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Bandwidth-Bound Models</div>
    <div class="card-desc">FPGAs with HBM2 (460 GB/s) can match a GPU on bandwidth-bound workloads (small-batch transformer decode) at lower power and lower latency variance.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Edge / Power-Constrained</div>
    <div class="card-desc">Mid-range FPGAs (Kintex, Arria) run at 10–25 W and can implement INT8 inference pipelines without the overhead of a full GPU stack (drivers, CUDA, etc.).</div>
  </div>
</div>
</div>

---

## Key Takeaways

- An FPGA is a chip of **LUTs** (tiny SRAM-based function tables), **flip-flops**, **DSP slices** (hardened MACs), **BRAM/URAM**, and **programmable interconnect** whose function is determined by a loaded **bitstream** — rewritable in seconds.
- A **LUT6** implements any Boolean function of 6 inputs by storing a 64-bit truth table in SRAM; the configuration SRAM is what the bitstream programs.
- **DSP slices** provide hardened 27×18-bit multiply-accumulate at up to 891 MHz, delivering 1.78 GOPS each; a large FPGA has ~10K DSP slices.
- FPGAs operate at **300–750 MHz** (much lower than CPUs), but compensate with deep pipelining and extreme parallelism — every DSP, LUT, and BRAM works simultaneously.
- The FPGA's killer advantage is **cycle-exact deterministic latency** — every inference takes exactly the same number of clock cycles, making it ideal for latency-critical applications that GPUs fail on due to scheduling jitter.
- **AI on FPGAs** includes Microsoft Brainwave (Azure low-latency inference), hls4ml (physics and edge ML), and custom quantized (INT4, ternary) dataflows no GPU supports natively.
- FPGAs are the essential tool for **ASIC prototyping** — every major AI chip (TPU, Hopper, M-series) was emulated on FPGA farms before tape-out.

---

*Next: [Chapter 21 — ASICs](./21_asics.md), where we go all the way to the other extreme: purpose-built silicon with no flexibility at all — and 10–100× better efficiency for it.*

[← Back to Table of Contents](./README.md)
