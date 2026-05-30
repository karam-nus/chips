---
title: "Chapter 4 — Anatomy of a Processor"
---

[← Back to Table of Contents](./README.md)

# Chapter 4 — Anatomy of a Processor

You now have all the raw ingredients: transistors that switch, gates that compute logic, adders and multipliers that do arithmetic ([Chapter 2](./02_transistors_to_logic.md)), and number representations that map bit patterns to mathematical values ([Chapter 3](./03_number_representation.md)). This chapter assembles those pieces into something that can *run a program* — a **processor**, or CPU core.

Understanding the processor's anatomy — the program counter, register file, ALU, control unit, and memory interface — unlocks everything that follows. The **fetch–decode–execute** cycle is the heartbeat of every computation ever performed on a chip. **Pipelining** is why a single core can sustain high throughput even when individual operations take multiple cycles. And grasping why a scalar CPU does only one MAC every few cycles is the essential counterpoint that explains why GPUs, TPUs, and tensor cores exist at all.

> **The one-sentence version:** A processor is a state machine that repeatedly reads an instruction from memory, decodes it, performs the specified operation on registers using the ALU, and writes the result back — and pipelining allows multiple instructions to be in flight at once so the hardware is never idle.

---

## The Stored-Program Computer

Before the stored-program model (circa 1945, independently by von Neumann, Eckert, Mauchly, and others), programs were *wired in* — you changed a computation by rewiring the machine. The insight was simple and transformative: **store the program in the same memory as the data**, encoded as numbers. Then the processor just needs to be told where in memory to start reading.

This gave rise to the **fetch–decode–execute** loop — a program is nothing more than a list of numbered instructions, and the CPU executes them by reading each one, figuring out what it means, and carrying it out. The **program counter (PC)** holds the address of the next instruction to fetch.

---

## Von Neumann vs Harvard Architecture

<div class="diagram">
<div class="diagram-title">Von Neumann vs Harvard</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Von Neumann</div>
    <ul>
      <li><strong>One memory</strong> for instructions and data</li>
      <li>One shared bus between CPU and memory</li>
      <li>Simpler; standard for general-purpose CPUs</li>
      <li>Bottleneck: can't fetch instruction and read data simultaneously (<em>von Neumann bottleneck</em>)</li>
      <li>Examples: x86 PC, most server CPUs</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Harvard</div>
    <ul>
      <li><strong>Separate memories</strong> for instructions and data</li>
      <li>Separate buses — both can be accessed simultaneously</li>
      <li>Higher throughput; common in DSPs, microcontrollers</li>
      <li>More complex; fixed split between instruction and data RAM</li>
      <li>Examples: PIC microcontrollers, some DSP cores</li>
    </ul>
  </div>
</div>
</div>

Modern "von Neumann" processors use a **modified Harvard** architecture in practice: physically, instruction and data caches are separate (Harvard-style) so both can be read in parallel; logically, they map to the same address space (von Neumann-style). The L1 instruction cache (L1-I) and L1 data cache (L1-D) are separate; L2 and beyond unify them. You'll see this when we cover caches in [Chapter 9](./09_memory_types.md) and [Chapter 10](./10_memory_management.md).

---

## The Components of a Processor

<div class="diagram">
<div class="diagram-title">Processor Block Diagram</div>
<div class="flow">
  <div class="flow-node accent wide">Memory (DRAM / Cache) — holds instructions and data</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Fetch — Program Counter (PC) → Instruction Memory → Instruction Register (IR)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">Decode — Control Unit decodes opcode → control signals</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Execute — Register File + ALU + (Load/Store unit)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Write-back — result written to register file or memory</div>
</div>
</div>

### Program Counter (PC)

The **program counter** is a single register — 64 bits on a 64-bit CPU — that holds the **address** of the next instruction to fetch. After each instruction is fetched, the PC is incremented by the instruction size (4 bytes for fixed-width 32-bit ISAs like RISC-V; variable for x86). Branch and jump instructions change the PC to an arbitrary target.

### Register File

The **register file** is a small, ultra-fast array of registers on-chip. A typical 64-bit CPU (RISC-V, ARM, x86-64) exposes 16–32 general-purpose registers, each 64 bits wide. Key properties:

| Architecture | GPRs | Width | Total size |
|-------------|:----:|:-----:|----------:|
| x86-64 | 16 (RAX–R15) | 64 bit | 128 B |
| ARM AArch64 | 31 + SP + ZR | 64 bit | 256 B |
| RISC-V (RV64I) | 32 (x0–x31) | 64 bit | 256 B |

Register access latency is **< 1 clock cycle** (effectively zero from the pipeline's perspective — the register file is read in the same stage as decode). Compare this to L1 cache (~4 cycles), L2 (~12 cycles), DRAM (~100–300 cycles). Registers are the fastest memory in the system, by far.

### ALU

The **Arithmetic Logic Unit** from [Chapter 2](./02_transistors_to_logic.md) — add, subtract, AND, OR, XOR, shift, compare. In a pipelined CPU the ALU is the **execute stage**: it takes two register values (or one register and an immediate) and produces a result. On out-of-order cores there may be multiple ALUs in parallel (Chapter 14 covers this), but here we discuss the simple scalar case.

### Control Unit

The **control unit** decodes the instruction (the raw bit pattern in the instruction register) and generates **control signals** — a bundle of bits that tell each component what to do: which ALU operation to perform, whether to read or write memory, which register to write the result to, whether to update the PC (branch), etc. In a hardwired control unit, this is a combinational logic circuit; in older microcode-based designs (x86 internally), a small ROM look-up translates complex instructions into sequences of micro-operations.

### Memory Interface

The processor communicates with memory through a **load/store unit** that generates memory addresses and handles reads/writes. In modern CPUs, "memory" first checks the cache hierarchy (L1 → L2 → L3 → DRAM) before going off-chip. We cover this hierarchy exhaustively in [Chapter 9](./09_memory_types.md).

---

## The Fetch–Decode–Execute Cycle

This is the operational heartbeat of every processor ever built. Let's walk through it with a concrete tiny instruction: `ADD R3, R1, R2` — add register R1 and R2, write result to R3.

### Step 1: Fetch

```text
PC = 0x0000_1000    ← contains address of next instruction
Instruction Memory[0x1000] = 0x00208133  ← raw 32-bit RISC-V encoding
IR ← 0x00208133     ← loaded into Instruction Register
PC ← PC + 4         ← advance to next instruction (0x1004)
```

The PC value is presented to the instruction cache; the 4-byte instruction is read and stored in the instruction register (IR). Simultaneously, the PC is incremented so the next fetch is already in progress (this overlap is the basis of pipelining).

### Step 2: Decode

The control unit inspects the bit fields of the IR:

```text
Instruction: 0x00208133  =  0000 0000 0010  00001  000  00011  0110011
                             │ funct7      │ rs2  │rs1 │funct3│  rd  │ opcode
                             └─ 0         └─ 2   └─1  └─ADD  └─ 3   └─R-type
```

Decoded: opcode = R-type integer arithmetic, operation = ADD, source registers = R1 and R2, destination register = R3.

Control signals generated:
- ALUop = ADD
- RegRead: read R1 and R2 from register file
- RegWrite: write result to R3
- MemRead = 0, MemWrite = 0 (no memory access)

### Step 3: Execute

The values of R1 and R2 are presented to the ALU, which performs addition (using the full-adder chain from Chapter 2) and produces a 64-bit result. Any condition flags (zero, negative, overflow) are also set.

### Step 4: Memory Access (if needed)

For an ADD instruction, this step is a no-op. For a `LOAD` (`lw rd, offset(rs1)`), the ALU computes the effective address `rs1 + offset`, and the data cache is read. For a `STORE`, the data cache is written.

### Step 5: Write-Back

The result from the ALU (or from memory for a load) is written to the destination register R3 in the register file.

### A Tiny Python CPU

```python
class TinyCPU:
    """
    A minimal von Neumann CPU: 8 × 32-bit registers, simple instruction set.
    Instruction format (32-bit): [opcode:4][dest:4][src1:4][src2:4][imm:16]
    Opcodes: 0=NOP, 1=ADD, 2=SUB, 3=LOAD_IMM, 4=HALT
    """
    def __init__(self):
        self.regs  = [0] * 8       # R0–R7 (32-bit, ignoring overflow)
        self.pc    = 0
        self.mem   = {}            # address → 32-bit instruction/data
        self.halted = False

    def load_program(self, instructions: list[int], start: int = 0):
        for i, instr in enumerate(instructions):
            self.mem[start + i * 4] = instr

    def make_instr(self, op: int, dest: int = 0, src1: int = 0,
                   src2: int = 0, imm: int = 0) -> int:
        return ((op & 0xF) << 28) | ((dest & 0xF) << 24) | \
               ((src1 & 0xF) << 20) | ((src2 & 0xF) << 16) | (imm & 0xFFFF)

    def step(self):
        """One fetch-decode-execute cycle."""
        # FETCH
        instr = self.mem.get(self.pc, 0)
        self.pc += 4
        # DECODE
        op   = (instr >> 28) & 0xF
        dest = (instr >> 24) & 0xF
        src1 = (instr >> 20) & 0xF
        src2 = (instr >> 16) & 0xF
        imm  = instr & 0xFFFF
        # EXECUTE + WRITE-BACK
        if   op == 0:  pass                                    # NOP
        elif op == 1:  self.regs[dest] = (self.regs[src1] + self.regs[src2]) & 0xFFFFFFFF  # ADD
        elif op == 2:  self.regs[dest] = (self.regs[src1] - self.regs[src2]) & 0xFFFFFFFF  # SUB
        elif op == 3:  self.regs[dest] = imm                  # LOAD_IMM
        elif op == 4:  self.halted = True                      # HALT
        return op

    def run(self, max_cycles: int = 100) -> list[dict]:
        trace = []
        for cycle in range(max_cycles):
            if self.halted: break
            op = self.step()
            trace.append({'cycle': cycle, 'pc': self.pc - 4,
                          'regs': self.regs[:], 'op': op})
        return trace

# Program: R1=10, R2=3, R3=R1+R2, R4=R3-R2, HALT
cpu = TinyCPU()
program = [
    cpu.make_instr(op=3, dest=1, imm=10),         # R1 = 10
    cpu.make_instr(op=3, dest=2, imm=3),          # R2 = 3
    cpu.make_instr(op=1, dest=3, src1=1, src2=2), # R3 = R1 + R2 = 13
    cpu.make_instr(op=2, dest=4, src1=3, src2=2), # R4 = R3 - R2 = 10
    cpu.make_instr(op=4),                          # HALT
]
cpu.load_program(program)
trace = cpu.run()
for t in trace:
    ops = ['NOP','ADD','SUB','LOAD_IMM','HALT']
    print(f"Cycle {t['cycle']} PC={t['pc']:04X} op={ops[t['op']]:<10s} "
          f"R1={t['regs'][1]} R2={t['regs'][2]} R3={t['regs'][3]} R4={t['regs'][4]}")
```

Output:
```
Cycle 0 PC=0000 op=LOAD_IMM  R1=10 R2=0  R3=0  R4=0
Cycle 1 PC=0004 op=LOAD_IMM  R1=10 R2=3  R3=0  R4=0
Cycle 2 PC=0008 op=ADD        R1=10 R2=3  R3=13 R4=0
Cycle 3 PC=000C op=SUB        R1=10 R2=3  R3=13 R4=10
Cycle 4 PC=0010 op=HALT       R1=10 R2=3  R3=13 R4=10
```

> **But real computers don't stop when they `HALT`.** This `TinyCPU` ends because its program does — but the machine in front of you never runs out of program. When no task is ready, the operating system runs an *idle loop* that halts the core until the next **interrupt** wakes it. How power-on, interrupts, the scheduler, the idle `HLT`, and even keeping the screen lit all fit together is the whole story of [Appendix H — The Life of a Running Computer](./appendix_h_how_a_computer_runs.md).

---

## Datapath vs Control

The processor is cleanly divided into two subsystems:

- **Datapath:** the wires and functional units through which data flows — the register file, ALU, multiplexers, adders, shifters. This is *where the computation happens*.
- **Control:** the logic that decides *what* the datapath does each cycle — the instruction decoder, finite-state machine, control signal generators. This is *what directs the computation*.

A clean datapath/control split is the architecture principle behind most processor and accelerator design. In hardware description languages (Verilog/VHDL, Chapter 6), the datapath is written as assignments and arithmetic; the control is written as `always` blocks and state machines.

---

## The Clock and Performance Equation

The fundamental performance equation for a single-threaded scalar processor:

$$\text{Execution time} = \frac{\text{Instructions} \times \text{CPI}}{\text{Clock frequency}}$$

where:
- **Instructions** = total instruction count for the program
- **CPI** = Cycles Per Instruction (average)
- **Clock frequency** = cycles per second (Hz)

Equivalently, **IPC = 1/CPI** (Instructions Per Cycle). A simple in-order scalar CPU executes at most 1 instruction per cycle, so CPI ≥ 1 in the ideal case. Cache misses, branch mispredictions, and data hazards increase CPI above 1.

| CPU type | Ideal IPC | Typical real IPC (ML workload) |
|----------|:---------:|:------------------------------:|
| Simple in-order scalar | 1 | 0.5–0.9 |
| In-order 2-wide | 2 | 0.8–1.6 |
| Out-of-order 4-wide (modern laptop CPU) | 4 | 2–3.5 |
| Out-of-order 8-wide (server CPU) | 8 | 3–6 |

For a matmul computation, a scalar CPU running at 3 GHz with IPC = 2 performs:

$$3 \times 10^9 \times 2 = 6 \times 10^9 \text{ ops/s} = 6 \text{ GFLOPS}$$

An H100 GPU peaks at ~67,000 GFLOPS (INT8). The ~10,000× gap is not from a faster clock (GPUs typically run at 1–2 GHz) — it is from **parallelism**: tens of thousands of ALUs operating simultaneously. This gap is precisely what motivates Chapter 8 (parallel processing) and Chapter 15 (GPUs).

---

## Pipelining: Throughput Without Latency

A scalar non-pipelined CPU executes one instruction entirely before starting the next. Every stage (fetch, decode, execute, memory, write-back) must complete — then the next instruction starts. If each stage takes 1 ns and there are 5 stages, one instruction takes **5 ns**.

**Pipelining** overlaps the stages of successive instructions, like an assembly line. While instruction $k$ is in the execute stage, instruction $k+1$ is being decoded, and instruction $k+2$ is being fetched.

### The Laundry Analogy

Imagine doing 4 loads of laundry, each requiring: Wash (30 min) → Dry (40 min) → Fold (20 min).

- **Non-pipelined:** each load finishes completely before the next starts. Total = 4 × 90 min = 360 min.
- **Pipelined:** as soon as load 1 moves to the dryer, load 2 goes into the washer. Total ≈ 90 + 3 × 40 = 210 min — nearly 2× faster for 4 loads, approaching 3× for many loads.

### The Classic 5-Stage RISC Pipeline

The canonical RISC pipeline (inspired by the MIPS R2000, 1985 — still relevant as the template used in RISC-V textbooks):

```text
Cycle:  1    2    3    4    5    6    7    8    9
Instr1: IF   ID   EX   MEM  WB
Instr2:      IF   ID   EX   MEM  WB
Instr3:           IF   ID   EX   MEM  WB
Instr4:                IF   ID   EX   MEM  WB
Instr5:                     IF   ID   EX   MEM  WB
```

| Stage | Name | What happens |
|-------|------|--------------|
| IF | Instruction Fetch | Read instruction from I-cache using PC; update PC |
| ID | Instruction Decode | Decode opcode; read source registers from register file |
| EX | Execute | ALU computes result; branch target computed |
| MEM | Memory Access | Load/store to D-cache (if needed); otherwise pass-through |
| WB | Write-Back | Write result to destination register |

In the ideal pipelined case, the processor starts a new instruction every cycle (throughput = 1 instruction/cycle), even though each instruction's latency is still 5 cycles. **Throughput ≠ latency** — this distinction is crucial for ML inference.

### Pipeline Throughput and Latency

```python
def pipeline_throughput_example():
    """
    Demonstrate the throughput vs latency distinction.
    Non-pipelined vs 5-stage pipelined execution time for N instructions.
    """
    N = 20          # instructions
    stages = 5      # pipeline depth
    cycle_ns = 1.0  # 1 GHz for easy math

    # Non-pipelined: each instruction takes `stages` cycles
    non_pipelined_cycles = N * stages
    # Pipelined: fill pipeline (stages-1 cycles), then 1 new result per cycle
    pipelined_cycles = stages + (N - 1)   # or equivalently: N + stages - 1

    throughput_non = N / (non_pipelined_cycles * cycle_ns)   # instructions/ns = GIPS
    throughput_pip = N / (pipelined_cycles * cycle_ns)

    print(f"Non-pipelined: {non_pipelined_cycles} cycles, {non_pipelined_cycles*cycle_ns:.0f} ns")
    print(f"Pipelined:     {pipelined_cycles} cycles, {pipelined_cycles*cycle_ns:.0f} ns")
    print(f"Speedup:       {non_pipelined_cycles / pipelined_cycles:.2f}×")
    print(f"Throughput non-pipelined: {throughput_non:.3f} GIPS")
    print(f"Throughput pipelined:     {throughput_pip:.3f} GIPS")

pipeline_throughput_example()
```

Output:
```
Non-pipelined: 100 cycles, 100.0 ns
Pipelined:     24 cycles, 24.0 ns
Speedup:       4.17×
Throughput non-pipelined: 0.200 GIPS
Throughput pipelined:     0.833 GIPS
```

For large N, speedup approaches the number of pipeline stages (5×), bounded by the longest stage (the critical path sets the clock period).

---

## Pipeline Hazards

A **hazard** is a situation where the next instruction cannot execute in the expected cycle. Three types:

<div class="diagram">
<div class="diagram-title">Pipeline Hazard Types</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Structural Hazard</div>
    <div class="card-desc">Two instructions need the same hardware resource in the same cycle. Classic example: a single-ported memory can't be read for both instruction fetch and data load simultaneously. Avoided by separate I-cache and D-cache (modified Harvard).</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Data Hazard (RAW)</div>
    <div class="card-desc">Read-After-Write: instruction 2 reads a register that instruction 1 writes, but instruction 1 hasn't reached WB yet. Most common hazard in arithmetic-heavy code. Solved by forwarding (bypassing) or stalls (bubbles).</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Control Hazard</div>
    <div class="card-desc">A branch changes the PC, but the pipeline has already fetched the next sequential instruction(s). Solved by branch prediction + speculative execution; misprediction flushes the wrongly-fetched instructions (penalty: ~15–25 cycles on modern CPUs).</div>
  </div>
</div>
</div>

### Forwarding (Bypassing) — Resolving Data Hazards Without Stalls

```text
Cycle:  1    2    3    4    5
ADD R1: IF   ID   EX   MEM  WB
SUB R2: IF   ID   EX  ← needs R1; forward from ADD's EX output
```

Instead of waiting for R1 to be written back to the register file (WB in cycle 5), the EX output of the ADD is forwarded directly to the input of the next instruction's EX stage. This requires a **bypass network** (MUX at the ALU inputs), but eliminates stalls for most RAW hazards.

Without forwarding, a RAW hazard requires **stalls** (pipeline bubbles — NOP cycles inserted by the hardware), increasing CPI.

### Branch Misprediction Cost

Modern out-of-order CPUs (Chapter 14) use **branch predictors** with >95–99% accuracy on typical code. Still, a misprediction costs:

$$\text{misprediction penalty} = \text{pipeline depth to branch resolution} \approx 15–25 \text{ cycles on modern x86}$$

For a loop over 1 million iterations with 99.9% prediction accuracy: ~1,000 mispredictions × 20 cycles = 20,000 wasted cycles. For a tight kernel, this is negligible. For code with many unpredictable branches (like decision trees, or highly data-dependent if-else), it dominates — which is part of why neural networks replaced decision trees in ML: a matrix multiply has almost no branches.

---

## Toward Superscalar: One Instruction Is Never Enough

A 5-stage pipeline reaches at most 1 IPC. To do better, a **superscalar** processor fetches, decodes, and issues *multiple* instructions per cycle. A 4-wide superscalar has four sets of functional units and can retire up to 4 instructions per cycle. Combined with **out-of-order execution** (dynamically reordering instructions to execute as soon as their inputs are ready), modern server CPUs achieve 3–6 effective IPC.

We cover superscalar, out-of-order, and vector (SIMD) extensions in depth in [Chapter 8 — Parallel Processing](./08_parallel_processing.md) and [Chapter 14 — CPUs](./14_cpus.md). For now, the key observation:

> Even an 8-wide out-of-order superscalar CPU at 3 GHz peaks at ~24 GFLOPS for scalar FP32 operations. The arithmetic intensity of a large matmul demands **teraflops** — four to five orders of magnitude more. No amount of pipelining a scalar processor closes that gap; you need a fundamentally different architecture. That is Chapter 15.

---

## Putting It Together: A Pipeline Throughput Calculator

```python
def pipeline_performance(
    instructions: int,
    clock_ghz: float,
    pipeline_depth: int = 5,
    stalls_per_100: float = 10.0,   # stall cycles per 100 instructions
    mispredictions_per_100: float = 1.0,
    mispredict_penalty: int = 20,
) -> dict:
    """
    Estimate execution time for a program on a simple pipelined scalar CPU.
    stalls_per_100: data hazard stalls (forwarding can't resolve all)
    """
    # Ideal pipelined: 1 instruction per cycle after fill
    ideal_cycles = pipeline_depth + instructions - 1

    # Add stall cycles from data hazards
    stall_cycles = int(instructions * stalls_per_100 / 100)

    # Add branch misprediction cycles
    mispredict_cycles = int(instructions * mispredictions_per_100 / 100) * mispredict_penalty

    total_cycles = ideal_cycles + stall_cycles + mispredict_cycles
    cpi = total_cycles / instructions
    exec_time_ns = total_cycles / (clock_ghz * 1e9) * 1e9   # in ns
    gflops = instructions / (exec_time_ns * 1e-9) / 1e9

    return {
        'total_cycles':      total_cycles,
        'ideal_cycles':      ideal_cycles,
        'stall_cycles':      stall_cycles,
        'mispredict_cycles': mispredict_cycles,
        'CPI':               round(cpi, 3),
        'exec_time_ns':      round(exec_time_ns, 1),
        'effective_GOPS':    round(gflops, 3),
    }

# A 10,000-instruction floating-point kernel
result = pipeline_performance(
    instructions=10_000,
    clock_ghz=3.0,
    pipeline_depth=5,
    stalls_per_100=8,          # ~8 stalls per 100 instructions
    mispredictions_per_100=2,  # 2 mispredicted branches per 100
    mispredict_penalty=20,
)
print("Pipeline Performance Estimate:")
for k, v in result.items():
    print(f"  {k:<22s}: {v}")
```

---

## Why This Matters for Model Optimization

<div class="diagram">
<div class="diagram-title">Processor Architecture → ML Practitioner Decisions</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Scalar CPU = Poor Matmul</div>
    <div class="card-desc">One ALU, one MAC per cycle: a CPU core at 3 GHz, IPC=2 does ~6 GFLOPS. A 7B-parameter model inference requires ~14 TFLOPS at FP16. This 2,300× gap is why you never run large LLM inference on CPU cores — you need a GPU or accelerator.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Throughput ≠ Latency</div>
    <div class="card-desc">A pipelined GPU issues one new tensor-core result per clock even though each operation has multi-cycle latency. Batch size = 1 inference sees latency; large-batch training sees throughput. This is why bigger batches have higher GPU utilization — they fill the pipeline.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Branch-Free Code is Fast Code</div>
    <div class="card-desc">Attention masks, ReLU, top-k sampling — any data-dependent conditional adds branch misprediction risk. Hardware-friendly activations (GELU, SiLU) are sometimes preferred partly because they are computed arithmetically (no branching).</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Register Pressure = Kernel Efficiency</div>
    <div class="card-desc">CUDA/Triton kernels compete for a fixed number of registers per SM. Using too many registers reduces occupancy (fewer threads in flight), reducing throughput. The register file is the very first memory level — spilling to L1 is expensive.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">CPI and the Memory Wall</div>
    <div class="card-desc">A cache miss stalls the pipeline for 100–300 cycles (DRAM latency). A GPU with 1,000 threads can hide this latency by switching to another warp. A scalar CPU cannot. This is the fundamental reason why memory-bound ML kernels must be batched well.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">fetch–decode–execute in GPU Context</div>
    <div class="card-desc">A GPU SM still runs fetch–decode–execute, but on 32-thread warps simultaneously (SIMT). The "instruction" is applied to 32 FP16 or INT8 values at once. Tensor cores accelerate further by making the "instruction" a 16×16 matrix multiply.</div>
  </div>
</div>
</div>

---

## Key Takeaways

- A **stored-program processor** holds instructions in memory and executes them via the **fetch–decode–execute** cycle, driven by the **program counter**.
- **Von Neumann** uses one memory for code and data; **Harvard** separates them. Modern CPUs use modified Harvard (split caches, unified address space).
- Core components: **PC, register file, ALU, control unit, memory interface** — datapath carries data; control directs it.
- Performance = Instructions × CPI / Clock. Scalar CPUs peak at ~1 IPC; real-world CPI > 1 due to hazards.
- **Pipelining** overlaps instruction execution (like an assembly line), approaching 1 instruction/cycle throughput at the cost of startup latency and hazard handling.
- **Data hazards** (RAW) are resolved by **forwarding**; **control hazards** (branches) are resolved by **prediction** — mispredictions cost 15–25 cycles.
- A scalar CPU at ~6 GFLOPS is ~2,000–10,000× slower than a GPU for matmul — the gap that motivates all parallel hardware in the rest of this guide.

---

*Next: [Chapter 5 — Instruction Sets](./05_instruction_sets.md), where we look at what the ISA layer exposes to software — RISC vs CISC, x86/ARM/RISC-V, and the SIMD vector extensions that give CPUs their highest ML throughput.*

[← Back to Table of Contents](./README.md)
