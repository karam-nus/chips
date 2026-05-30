---
title: "Chapter 31 — The Chip Design Flow (RTL → GDSII)"
---

[← Back to Table of Contents](./README.md)

# Chapter 31 — The Chip Design Flow (RTL → GDSII)

The AI accelerator on your training cluster didn't spring fully-formed from a chip vendor's imagination. It began as a document — a specification written by architects. It passed through months of RTL coding, years of verification, and multiple rounds of synthesis and timing closure before a single electron of manufacturing current was passed through silicon. The finished product — a GDSII layout file — is a precisely annotated description of billions of polygons, each representing a doped region, a metal wire, or a via, that a fab will translate into physical silicon using photolithography. The journey from that spec document to that GDSII file is the **chip design flow**, and understanding it explains nearly everything puzzling about the AI hardware industry: why a new accelerator takes 2–3 years, why changing an architecture mid-cycle is catastrophic, why FPGAs (Chapter 20) exist as a distinct product category, and why the NRE for a leading-edge ASIC (Chapter 21) runs into the tens or hundreds of millions of dollars.

This chapter walks through every stage of that flow, names the tools, explains the concepts, and identifies the decision points where an ML-oriented architect makes choices that will determine whether the resulting chip can execute the matrix multiplications your models need at the power and latency you require.

> **The one-sentence version:** Chip design is a pipeline that transforms an English-language specification into a geometrically precise manufacturable layout through a sequence of increasingly concrete representations — from architecture to RTL to gate netlist to placed-and-routed polygons — each step adding detail and removing degrees of freedom, culminating in a GDSII file handed to a fab.

---

## The Big Picture: Representations and Transformations

The design flow is best understood as a series of **representation changes**, each computable from the one before it:

<div class="diagram">
<div class="diagram-title">The Chip Design Flow — Top to Bottom</div>
<div class="flow">
  <div class="flow-node accent wide">Specification & Architecture<br/><small>what does the chip do? what are the PPA targets?</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Microarchitecture Design<br/><small>pipeline stages, dataflow, memory hierarchy, block diagram</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">RTL Coding (Verilog / SystemVerilog)<br/><small>register-transfer level — hardware behavior described in HDL</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Functional Verification (Chapter 32)<br/><small>simulation, formal, emulation — prove RTL is correct</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Logic Synthesis<br/><small>RTL → gate-level netlist using a standard-cell library</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">DFT Insertion<br/><small>scan chains, BIST, JTAG — making the chip testable post-fab</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node teal wide">Physical Design (PnR): Floorplan → Place → CTS → Route<br/><small>cells placed in 2D space; wires routed between them</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Timing Closure (Static Timing Analysis)<br/><small>setup/hold slack verified across all corners and modes</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">Physical Verification (DRC / LVS)<br/><small>layout is manufacturable; matches the schematic</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">Tapeout → Mask Generation → Fab → Test<br/><small>GDSII to foundry; masks made; wafers processed; dice shipped</small></div>
</div>
</div>

These stages are not purely sequential — verification runs in parallel with every other stage, and timing concerns intrude as early as microarchitecture. But logically, each stage produces the input the next requires.

---

## Stage 1 — Specification and Architecture

### What Gets Decided Here

The chip begins as a set of requirements, usually captured in a **micro-architecture specification document** (often called a "micro-arch spec" or just "spec"):

- **Workload targets**: inference-only or training? which models? what batch sizes?
- **Performance targets**: TOPS (tera-operations per second), latency, throughput.
- **Power envelope**: total chip TDP (e.g., 300 W), per-domain power budgets.
- **Area target**: die area in mm² (which determines yield and cost).
- **Interface requirements**: HBM, PCIe, NVLink, host CPU interface.
- **Process node**: 5 nm, 3 nm, 2 nm — the foundry and node choice.
- **IP cores to license**: SerDes PHYs, PCIe/NVLink controllers, memory controllers.

The **PPA triangle** — **Power, Performance, Area** — is the master trade-off. Every subsequent decision is a negotiation among these three:

<div class="diagram">
<div class="diagram-title">PPA — The Designer's Triangle</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Performance</div>
    <div class="card-desc">Clock frequency (GHz), throughput (TOPS/TFLOPS), latency (ns). Higher performance costs more area (wider datapaths) and power (faster switching, more activity).</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Power</div>
    <div class="card-desc">Total Watts (TDP), power density (W/mm²), energy per operation (pJ/op). Bounded by cooling and power delivery; quadratic in supply voltage (E ≈ ½CV²).</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Area</div>
    <div class="card-desc">Die area in mm². Directly sets unit cost and yield. More area = more transistors = more function or more registers/buffers, but costs more per die.</div>
  </div>
</div>
</div>

The **process node** sets the playing field. At TSMC N3 (2023), a standard NAND2 cell is roughly 0.01 µm²; at N7 (2018) it was roughly 0.04 µm². The 4× density gain between N7 and N3 means you can either fit 4× more logic in the same area or shrink the die by 4× at the same logic count.

### Architecture Choices That Constrain Everything

Key architectural decisions made here ripple through every later stage:

| Decision | Downstream impact |
|---|---|
| Dataflow: systolic array vs vector vs spatial | Pipeline depth, register file size, memory bandwidth requirement |
| On-chip SRAM capacity | A large fraction of die area; power-hungry at high utilization |
| Numerical formats: FP32 vs BF16 vs INT8/FP8 | Multiplier area; 8-bit multiplier ≈ 4× smaller than 16-bit |
| Clock domains | Complexity of CDC (clock-domain crossing) logic; verification effort |
| Memory interface: HBM2e vs HBM3 vs LPDDR5 | Bandwidth ceiling; PHY area; package constraints |
| Number of compute tiles | Floorplan feasibility; NoC complexity |

---

## Stage 2 — RTL: Describing Hardware Behavior

**RTL (Register-Transfer Level)** is the level of abstraction where hardware is described as **registers** (flip-flops) holding state and **combinational logic** transforming that state each clock cycle. It is the primary language of chip design.

The dominant HDLs are **Verilog** and **SystemVerilog** (a superset of Verilog with richer type system and verification constructs). VHDL is older and more common in defense/aerospace. See [Chapter 6](./06_languages_and_nomenclature.md) for the full HDL context.

### What RTL Looks Like: A Pipelined Multiplier-Accumulator

Here is a small but real RTL example: a single-cycle MAC (multiply-accumulate) unit, the atom of any neural-network accelerator. It takes two INT8 operands, multiplies them (producing an INT16 product), and accumulates into a 32-bit sum — the fundamental operation of a matrix multiply.

```verilog
// mac_unit.v  -- 8-bit multiply-accumulate, pipelined (2 stages)
// Stage 1: register inputs, compute product
// Stage 2: accumulate into running sum
// Reset: synchronous active-high

module mac_unit (
    input  wire        clk,
    input  wire        rst_n,      // active-low async reset
    input  wire        valid_in,
    input  wire signed [7:0]  a,   // INT8 input
    input  wire signed [7:0]  b,   // INT8 weight
    input  wire               accum_clear,
    output reg  signed [31:0] accum_out,
    output reg                valid_out
);

    // Stage 1 pipeline registers
    reg signed [15:0] product_s1;
    reg               valid_s1;
    reg               clear_s1;

    // Stage 1: multiply
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            product_s1 <= 16'sd0;
            valid_s1   <= 1'b0;
            clear_s1   <= 1'b0;
        end else begin
            product_s1 <= a * b;    // synthesizes to a 8x8 signed multiplier
            valid_s1   <= valid_in;
            clear_s1   <= accum_clear;
        end
    end

    // Stage 2: accumulate
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accum_out <= 32'sd0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_s1;
            if (clear_s1)
                accum_out <= {{16{product_s1[15]}}, product_s1};  // sign-extend
            else if (valid_s1)
                accum_out <= accum_out + {{16{product_s1[15]}}, product_s1};
        end
    end

endmodule
```

Notice several important RTL properties:
- **Everything is synchronous with the clock** — state only changes at `posedge clk`. This is the register-transfer model.
- **`always @(posedge clk)` blocks** describe flip-flops; **combinational logic** (`a * b`) is evaluated between edges.
- The **pipeline registers** (`product_s1`, `valid_s1`) add latency (2 cycles) but allow higher clock frequency because each combinational stage is shorter.
- **The `*` operator** on signed 8-bit values synthesizes to a real 8×8 bit multiplier in silicon — a few hundred gates for INT8, thousands for FP16.

### RTL Design Hierarchy

Real chips are organized as hierarchies of modules:

```text
chip_top
├── compute_cluster [×8]
│   ├── vector_unit [×16]
│   │   ├── mac_unit [×128]
│   │   └── accumulator_tree
│   ├── register_file (256 × 32b)
│   └── local_sram (256 KB)
├── memory_subsystem
│   ├── hbm_controller [×4]
│   ├── noc_router [×16]
│   └── l2_cache (32 MB)
├── host_interface
│   ├── pcie_endpoint
│   └── dma_engine
└── clock_reset_controller
```

A modern AI accelerator may have **1–10 million lines of RTL** spread across hundreds of files. Managing that complexity — naming conventions, interfaces, design-rule enforcement — is itself a major engineering discipline.

---

## Stage 3 — Logic Synthesis

**Synthesis** translates RTL (Verilog) into a **gate-level netlist**: a structural description of interconnected logic cells from a **standard-cell library**.

### What Is a Standard-Cell Library?

A **standard cell** is a pre-characterized logic primitive — a NAND2, NOR3, DFF, MUX2, XNOR2, full adder, and hundreds of variants — laid out at a fixed height to fit on a design grid. Each cell in the library is characterized at multiple **corners** (process/voltage/temperature combinations) to capture worst-case timing:

| Cell | Function | Drive strength | Area (N3 ≈) | Delay (typ) |
|---|---|---|---|---|
| `NAND2_X1` | 2-input NAND | 1× | ~0.012 µm² | ~10 ps |
| `NAND2_X4` | 2-input NAND | 4× | ~0.040 µm² | ~8 ps |
| `DFF_X1` | D flip-flop | 1× | ~0.025 µm² | clk→q ~20 ps |
| `MUX2_X2` | 2:1 mux | 2× | ~0.030 µm² | ~12 ps |
| `FA_X1` | Full adder | 1× | ~0.050 µm² | ~18 ps |

The library is called a **PDK (Process Design Kit)** or **standard-cell library** and is provided by the foundry (e.g., TSMC N5 standard cells). Using a standard-cell library is why chip design is feasible — you do not hand-design billions of gates; the synthesizer maps your RTL to cells from the library automatically.

### What the Synthesis Tool Does

The synthesis tool (Synopsys Design Compiler, Cadence Genus, or open-source Yosys) performs three passes:

1. **Elaboration**: parse Verilog, build an internal AST, resolve hierarchies and parameters.
2. **Technology mapping**: map RTL constructs (adders, muxes, registers, comparators) to optimal combinations of standard cells. A 32-bit adder synthesizes to a carry-lookahead tree of full-adder cells; a 16-bit barrel shifter synthesizes to MUX trees.
3. **Optimization**: iteratively swap cells for equivalents with better timing or area; remove redundant logic; re-time registers across logic to balance path delays.

The **synthesis constraints** file (written in Tcl) tells the tool your targets:

```tcl
# synthesis_constraints.tcl  -- typical for a 1 GHz target
set_units -time ns -resistance kOhm -capacitance fF

# Define the clock
create_clock -period 1.0 -name clk [get_ports clk]
set_clock_uncertainty -setup 0.05 [get_clocks clk]  ; # 50 ps setup uncertainty
set_clock_uncertainty -hold  0.02 [get_clocks clk]  ; # 20 ps hold uncertainty

# Input/output timing budgets
set_input_delay  -clock clk -max 0.2 [all_inputs]
set_output_delay -clock clk -max 0.3 [all_outputs]

# Area/power objectives
set_max_area 0              ; # minimize area after timing is met
set_max_dynamic_power 2.5   ; # 2.5 W power budget for this block

# Dont-touch: preserve these manually optimized cells
set_dont_touch [get_cells pcie_phy/*]
```

**Output of synthesis**: a gate-level netlist (Verilog, but now structural — just wires and cell instances) plus a timing report showing the **critical path** and **timing slack**.

### Slack, Setup, and Hold

The synthesized netlist must meet **timing constraints** at the target clock frequency. The key concepts:

- **Setup time** ($t_{su}$): the data must arrive at a flip-flop's input *before* the clock edge by at least $t_{su}$.
- **Hold time** ($t_h$): the data must remain valid for at least $t_h$ *after* the clock edge.
- **Slack** = (allowed time) − (actual path delay). **Positive slack = pass; negative slack = violation = the chip will malfunction.**

$$\text{Setup slack} = T_{\text{clk}} - (t_{clq} + t_{\text{logic}} + t_{\text{interconnect}}) - t_{su}$$

Where:
- $T_{\text{clk}}$ = clock period (e.g., 1.0 ns for 1 GHz)
- $t_{clq}$ = flip-flop clock-to-Q delay
- $t_{\text{logic}}$ = combinational logic delay (sum of all gates on path)
- $t_{\text{interconnect}}$ = wire delay (estimated at synthesis; accurate post-routing)
- $t_{su}$ = setup time of the capturing flip-flop

The **critical path** is the path with the least (most negative) slack. Achieving timing closure at a given frequency means eliminating all negative-slack paths — by substituting faster cells, adding pipeline registers, restructuring logic, or reducing fanout.

---

## Stage 4 — DFT: Design for Test

Before physical design begins, **DFT (Design for Test)** structures are inserted into the netlist to make the manufactured chip testable:

<div class="diagram">
<div class="diagram-title">DFT Structures Overview</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card blue">
    <div class="card-title">Scan Chains</div>
    <div class="card-desc">All flip-flops are connected into a shift register. In test mode, you shift in a test pattern, capture the response, shift out the result, and compare against expected output. Enables testing billions of logic nodes.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">BIST (Built-In Self-Test)</div>
    <div class="card-desc">Logic embedded on-chip that generates patterns and checks responses autonomously. MBIST tests SRAM arrays; LBIST tests random logic. Essential for memories too large to test via scan.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">JTAG / TAP Controller</div>
    <div class="card-desc">IEEE 1149.1 boundary-scan interface: a 4-wire serial interface to access scan chains, memory BIST, debug registers, and on-chip logic analyzers from a test probe.</div>
  </div>
</div>
</div>

DFT is detailed in [Chapter 32](./32_chip_verification_and_validation.md). Here the key point is that DFT insertion adds 5–15% area overhead and must be planned from the architectural stage; retrofitting DFT late is expensive.

---

## Stage 5 — Physical Design (Place and Route)

Physical design translates the gate-level netlist into a **geometrically placed layout**: every cell has a physical (X, Y) coordinate, and every net (wire) has a routed path through metal layers. This is the most computationally intensive stage.

### Floorplanning

**Floorplanning** is the highest-level physical decision: where do the major blocks (compute clusters, memory arrays, I/O pads, clock networks, power rails) live on the die?

A good floorplan:
- Places memories close to the logic they feed (minimizing wire length → lower latency and power)
- Reserves space for the clock distribution network and power grid
- Satisfies pin-assignment constraints for off-chip connections

A typical AI accelerator floorplan might look like:

```text
┌─────────────────────────────────────────────────────────────┐
│    HBM PHY [×4]          ─── wide memory bus ───             │
├────────────┬──────────────────────────────┬─────────────────┤
│  PCIe / NVLink PHY       NoC Crossbar       Host Interface   │
├────────────┴──────────────────────────────┴─────────────────┤
│                                                               │
│   Compute Tile 0      Compute Tile 1      Compute Tile 2     │
│   (SRAM + MACs)       (SRAM + MACs)       (SRAM + MACs)      │
│                                                               │
│   Compute Tile 3      Compute Tile 4      Compute Tile 5     │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│         Shared L3 Cache / SRAM Bank (large area)             │
├──────────────────────────────────────────────────────────────┤
│  Clock Distribution   Power Management   Debug / DFT Logic   │
└──────────────────────────────────────────────────────────────┘
       Die dimensions: ~500–800 mm² for a large AI ASIC
```

### Placement

**Placement** assigns each standard cell a specific (X, Y) coordinate within the floorplan. The placer optimizes for:
- **Timing**: cells on critical paths placed close together to minimize wire delay
- **Congestion**: avoid routing bottlenecks (too many wires in too small a region)
- **Power**: cells with high switching activity placed near power grid straps

Modern placers (Cadence Innovus, Synopsys ICC2) run **global placement** (approximate positions, minimize wire-length estimates) followed by **legalization** (snap cells to grid, resolve overlaps) followed by **detailed placement** (local swaps to fix timing).

### Clock-Tree Synthesis (CTS)

The clock signal must arrive at every flip-flop at **nearly the same time** — if the clock is 1 ns (1 GHz) and arrives at one flip-flop 200 ps later than another, paths between them will violate hold timing. **CTS** builds an H-tree or mesh of buffers and inverters that distributes the clock with controlled **skew** (< 50–100 ps target) and **insertion delay**.

A billion-transistor chip may need **100,000+ clock-tree buffer cells** to distribute clock uniformly. Clock power alone can be 20–30% of total chip dynamic power.

### Routing

**Routing** connects every pair of pins with actual metal wires, observing:
- **Design rules**: minimum wire width, minimum spacing, via rules — enforced by the foundry's PDK.
- **Layer assignment**: thin lower metals for local connections; thick upper metals for global signals and power.
- **Timing**: wires on critical paths are widened and kept short; buffer insertion for long wires.

At 3 nm, a chip may have **15+ metal layers**; the router is managing **hundreds of millions of wire segments** simultaneously. Routing typically takes 12–48 hours of compute time on large server farms for a leading-edge design.

**Parasitic extraction** follows routing: resistance ($R$) and capacitance ($C$) of every wire are extracted from the actual geometry. Wire delay is now:

$$t_{\text{wire}} = 0.38 \cdot R \cdot C \quad \text{(Elmore delay for an RC wire)}$$

These parasitics are back-annotated into the timing analysis for **sign-off accuracy**.

---

## Stage 6 — Timing Closure (Static Timing Analysis)

After routing and parasitic extraction, **STA (Static Timing Analysis)** checks every possible timing path in the chip:

- **Across all corners**: worst-case slow process (slow transistors, low voltage, hot temperature) for setup; best-case fast process (fast transistors, high voltage, cold temperature) for hold.
- **Across all modes**: normal operation, test mode, power-saving mode, each with different clock frequencies.
- **Across all paths**: for a 1 billion gate design, there are trillions of timing paths — STA uses graph algorithms to check all simultaneously without simulation.

| PVT Corner | Purpose | Typical condition |
|---|---|---|
| SS/0.72V/125°C (worst slow) | Setup check | Slowest transistors, lowest voltage, hottest |
| FF/0.88V/−40°C (best fast) | Hold check | Fastest transistors, highest voltage, coldest |
| TT/0.80V/25°C (typical) | Nominal power/performance | Process center |

**Timing closure** is the iterative process of fixing all negative-slack paths. Techniques include:
- **Cell upsizing**: swap a cell for a stronger-drive version (bigger transistors, lower resistance, but more area and power).
- **Logic restructuring**: rebalance deep vs shallow logic trees.
- **Register retiming**: legally move pipeline registers to shorten combinational paths.
- **Clock skew optimization**: intentionally skew the clock to "borrow time" — giving more time to a slow path, taking it from an adjacent fast one.

Timing closure on a large AI accelerator can take **months of iterative engineering** — this is one of the biggest schedule risks in a chip project.

---

## Stage 7 — Physical Verification (DRC and LVS)

Before tapeout, two final checks are mandatory:

### DRC — Design Rule Checking

**DRC** verifies the layout conforms to the foundry's manufacturing rules: minimum wire widths, minimum spacings, via enclosure rules, antenna rules (charge build-up during etch), density rules (metal coverage must be within a range or dummy fill is added). A single DRC run on a leading-edge chip can take 24–48 hours and find **thousands of violations** that must all be resolved to zero before tapeout.

### LVS — Layout vs. Schematic

**LVS** extracts the netlist from the physical layout (which transistors are where, which wires connect what) and compares it to the schematic netlist (the gate-level netlist from synthesis). They must match exactly — any discrepancy means the layout does not implement what was designed.

Together, DRC and LVS are the **signoff checks** that a foundry requires before accepting the design for manufacturing.

---

## Stage 8 — Tapeout, Mask Generation, and Fabrication

When DRC, LVS, and all timing checks pass, the design is **taped out**: the GDSII file is sent to the foundry. The foundry:

1. **Runs their own checks** (OPC — optical proximity correction, adding sub-wavelength corrections to mask shapes to compensate for diffraction).
2. **Generates photolithographic masks** — one per layer, per patterning step. At N3, a full mask set is ~80–100 masks and costs $15–30M.
3. **Manufactures wafers** through ~1,000 process steps over 8–14 weeks (see [Appendix A](./appendix_a_how_chips_are_made.md)).
4. **Performs wafer-level electrical test** (probe testing).
5. **Dices wafers** and packages good dies.
6. **Ships to the designer** for post-silicon validation (Chapter 32).

The total elapsed time from tapeout to packaged silicon in the designer's hands is typically **3–5 months** for a leading-edge foundry.

---

## The EDA Toolchain

The design flow is enabled by **EDA (Electronic Design Automation)** software — a $14B/year industry that is an effective duopoly:

<div class="diagram">
<div class="diagram-title">EDA Toolchain — Key Players</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Synopsys</div>
    <div class="card-desc">Design Compiler (synthesis), PrimeTime (STA), ICC2 (P&R), Formality (formal equiv.), VCS (simulation), VC Formal, Synopsys DSO.ai (ML-driven optimization). ~$6B revenue.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Cadence</div>
    <div class="card-desc">Genus (synthesis), Innovus (P&R), Tempus (STA), Xcelium (simulation), JasperGold (formal). ~$4B revenue. Strong in analog/mixed-signal.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Siemens EDA (formerly Mentor)</div>
    <div class="card-desc">Calibre (DRC/LVS — essentially industry monopoly for physical verification), ModelSim (simulation), Tessent (DFT). Acquired by Siemens 2017.</div>
  </div>
</div>
</div>

**Open-source tools** have emerged for smaller projects and academic use:
- **Yosys**: open-source synthesis for FPGAs and small ASICs; supports Verilog/SystemVerilog. Used in the Skywater open-source PDK flow.
- **OpenROAD**: open-source RTL-to-GDSII flow integrating synthesis, floorplan, placement, CTS, routing, and STA. Backed by DARPA's IDEA program.
- **OpenLane**: automated flow wrapper for OpenROAD targeting Skywater 130nm and GlobalFoundries 180nm PDKs. Excellent for learning.
- **SymbiYosys**: open-source formal verification frontend.

Open tools currently cannot compete with commercial EDA at leading nodes — the optimizations required for 3–5 nm at billion-gate scale need years of proprietary development. But they enable learning and smaller designs without six-figure tool licenses.

---

## Standard Cells, PDKs, and IP Cores

### What You Build vs. What You License

A modern chip team does not design everything from scratch. A typical AI accelerator leverages:

| Component | Built in-house? | Notes |
|---|---|---|
| Compute (MACs, vector units, schedulers) | Yes | The differentiating IP |
| SRAM macros | Mostly licensed | Foundry provides SRAM compilers |
| PCIe controller | Licensed (IP core) | Synopsys/Cadence/Rambus provide hard IPs |
| HBM memory controller + PHY | Licensed | Complex analog + digital; years to develop |
| SerDes PHY (NVLink, Ethernet) | Licensed | Heavily analog; node-specific |
| Standard cells | Provided by foundry PDK | TSMC N3 standard cell library |
| CPU subsystem (for firmware) | Licensed (ARM cores) | Common in SoCs; see Chapter 22 |
| Compression / video codecs | Licensed | Often cheaper than in-house |

**IP core licensing** is a major cost in a chip project. A high-speed PCIe 5.0 or NVLink PHY from Synopsys or Cadence costs **$5–20M per project** and takes months to integrate and verify. This is one reason NRE for AI ASICs starts high even at older process nodes.

### Process Design Kit (PDK)

The **PDK** is the foundry's gift to the designer: it contains every characterization table, SPICE model, design rule document, standard cell library, memory compiler, and I/O cell library needed to design for that process. At leading nodes, the PDK is massive (terabytes) and kept under strict NDA. A chip designed for TSMC N5 cannot be fabricated at Samsung S4 without a nearly complete redesign — the PDK is entirely foundry-specific.

---

## Why This Takes 2–3 Years (and Costs $100M+)

The design flow above describes a sequence of stages, each with substantial iteration. Here is a realistic schedule for a leading-edge AI accelerator:

<div class="diagram">
<div class="diagram-title">AI Accelerator Design Timeline (Leading-Edge Node)</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Months 1–6: Architecture & Microarch</div>
    <div class="card-desc">Workload analysis, microarch spec, power/area estimates, PDK selection, team ramp-up, IP vendor selection and contracts.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Months 6–18: RTL & Verification</div>
    <div class="card-desc">RTL coding (hundreds of engineers), testbench development, UVM environments, simulation, formal verification, emulation bringup. Verification is 60–70% of the engineering work.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Months 12–20: Synthesis & Physical Design</div>
    <div class="card-desc">Synthesis, DFT insertion, floorplanning, placement, CTS, routing. Runs in parallel with late-stage verification. Multiple timing closure iterations.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Months 18–24: Signoff & Tapeout</div>
    <div class="card-desc">Timing signoff across all corners, DRC/LVS clean, IP integration final checks, foundry review, tapeout. One missed week here ripples into months.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Months 24–28: Fab + First Silicon</div>
    <div class="card-desc">Wafer fabrication (~12 weeks), packaging, first silicon in hand. Post-silicon bring-up begins (Chapter 32).</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Months 28–36: Validation & Production Ramp</div>
    <div class="card-desc">Post-silicon characterization, software stack development, reliability testing, production yield qualification, customer samples.</div>
  </div>
</div>
</div>

The **cost breakdown** for a 5 nm AI ASIC ($80–150M NRE total):

| Cost category | Approximate share |
|---|---|
| Engineering headcount (design + verification) | 40–50% |
| EDA tool licenses | 15–25% |
| IP core licensing | 15–20% |
| Mask set cost | 10–15% |
| MPW / shuttle / early fab runs | 3–5% |
| Post-silicon lab equipment & validation | 5–10% |

---

## A Synthesis Trace: From Verilog to Cells

To make the synthesis stage concrete, here is what Yosys produces for a simple module (edited for readability):

**Input RTL** (`adder4.v`):
```verilog
module adder4(
    input  [3:0] a, b,
    input        cin,
    output [3:0] sum,
    output       cout
);
    assign {cout, sum} = a + b + cin;
endmodule
```

**After Yosys synthesis** (mapped to a hypothetical cell library — actual output is structural Verilog):
```text
Synthesis report for adder4:
  Number of wires:                 14
  Number of cells:                 17
    FA (full adder):                4    <-- one per bit of the 4-bit sum
    BUF_X1:                         5    <-- output drivers
    AND2_X1:                        3    <-- carry generation assist
    XOR2_X1:                        5    <-- sum bits

  Estimated area:   17 cells × ~0.025 µm²/cell = ~0.43 µm²  (N5 library)
  Critical path:    4× full-adder carry chain ≈ 4 × 18ps = 72 ps
                    Max frequency: 1/72ps ≈ 13.8 GHz (1 stage, no register)
  With register:    Max frequency depends on clock-to-Q + setup + uncertainty
```

A real 32-bit adder at the same node would use a carry-lookahead or carry-select architecture to reduce the critical path from $32 \times 18\,\text{ps}$ to roughly $\log_2(32) \times 18\,\text{ps}$ — a 5× speedup from better topology, not smaller transistors.

---

## Where FPGAs Fit

[Chapter 20](./20_fpgas.md) covers FPGAs in detail; here is how they relate to this design flow:

FPGAs skip the synthesis-through-fab portion. You still write RTL, run synthesis (targeting LUTs and DSP blocks instead of standard cells), run place-and-route on the fixed FPGA fabric, and do timing closure — but you do it in **hours to days**, not months, and you can reprogram the chip tomorrow. The FPGA itself was already fabricated.

This is why the standard ASIC development methodology uses FPGAs as **pre-silicon emulators**: run the RTL on FPGA clusters at 10–100 MHz to find bugs and run software stacks before first silicon (Chapter 32). Google, Apple, NVIDIA, and AWS all maintain large FPGA farms (Palladium-class emulators or HAPS boards) for exactly this purpose.

The tradeoff: your FPGA-mapped accelerator runs at 200 MHz instead of 1.6 GHz, consumes 3× more power per operation, and has 5–20× more area per gate. The ASIC investment only pays off above a certain volume and when the workload is stable — the precise break-even analysis from [Chapter 21](./21_asics.md) applies.

---

## Why This Matters for Model Optimization / ML Engineers

Understanding the design flow reframes several puzzling observations about AI hardware:

**Why you can't get a new accelerator to support your new model format next month.** The chip on your server was taped out 18–24 months ago. Adding FP8 E4M3 support requires RTL changes → re-synthesis → re-routing → re-signoff → new tapeout → 3 more months of fab → post-silicon validation. The minimum cycle is 12–18 months even for a change that takes a software engineer a week. This is why [Chapter 30](./30_datatypes_drive_optimization_choices.md) emphasizes choosing the format the hardware already supports.

**Why hardware can't chase monthly model changes.** LLM architectures changed substantially in 2022, 2023, 2024, and 2025. Any AI ASIC designed in 2022 was specced in 2020–2021 and reflects architectural assumptions of that era. This is the fundamental argument for FPGAs and software-programmable cores (compute units where the instruction set can be extended) versus fixed-function hardwired logic.

**Why verification is where most of the money goes.** Chapter 32 details this, but even from this chapter's timeline: 60–70% of design engineering effort is verification. A chip that passes RTL simulation but has a logic bug costs $20–50M in respin NRE and 6+ months of schedule. The 1994 Intel Pentium FDIV bug — an incorrect lookup table in the floating-point divider that survived simulation — cost Intel $475M in replacements. Verification is not bureaucracy; it is existential risk management.

**Why AI ASICs are a concentrated bet.** A company that commits $150M NRE to a chip for Transformer inference in 2023 is betting that Transformer-like architectures remain dominant for the 3-year period it takes to get to revenue. If the dominant paradigm shifts to something with different dataflow requirements — different tile shapes, different memory access patterns, different numerical formats — the chip may be suboptimal or obsolete before it ships. This is the tension at the heart of the [Chapter 21](./21_asics.md) ASIC economics story.

**Why PPA numbers in marketing materials require context.** A vendor quoting "2,000 TOPS at INT8" without stating the process node, die area, power envelope, and arithmetic utilization tells you almost nothing. The design flow shows you that those 2,000 TOPS came from a specific set of PPA trade-offs; a chip at 2× the die area, 2× the power, and a process node 2 generations newer might deliver that number trivially — or heroically, depending on which of those were spent.

---

## Key Takeaways

- The chip design flow transforms a specification through RTL → gate netlist → placed-routed layout → GDSII, each stage adding geometric concreteness and removing flexibility. **Late-stage changes propagate upstream and cost months.**
- **PPA (Power, Performance, Area)** is the master trade-off. Every architectural, microarchitectural, synthesis, and physical-design decision is a negotiation among these three.
- **Synthesis** maps RTL to standard cells from a PDK-provided library; the synthesizer optimizes for timing, area, and power within constraints expressed in Tcl.
- **Physical design** (floorplan → place → CTS → route) translates a gate netlist to physical coordinates. Parasitic extraction after routing enables accurate **STA (Static Timing Analysis)** at the actual $R$/$C$ values.
- **Timing closure** — eliminating all negative-slack paths across all PVT corners and modes — is one of the largest schedule risks and an iterative, months-long effort.
- **EDA is a duopoly** (Synopsys + Cadence) with Siemens dominating physical verification. Open tools (Yosys, OpenROAD) are viable for learning and small designs.
- **AI accelerator timelines of 2–3 years** are not organizational failure — they reflect the physics and economics of the design flow. Understanding the stages explains every constraint the ML practitioner encounters when trying to exploit new hardware.

---

*Next: [Chapter 32 — Chip Verification & Validation](./32_chip_verification_and_validation.md), the deep dive into how correctness is proved before and after silicon: simulation, UVM, assertions, formal verification, emulation, DFT, and post-silicon bring-up — with concrete code examples.*

[← Back to Table of Contents](./README.md)
