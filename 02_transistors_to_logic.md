---
title: "Chapter 2 — Transistors to Logic"
---

[← Back to Table of Contents](./README.md)

# Chapter 2 — Transistors to Logic

[Chapter 1](./01_what_is_a_chip.md) established that a chip is billions of voltage-controlled switches. This chapter answers the next question: **how does flipping switches perform mathematics?** You'll see how to build every logic gate from CMOS transistor pairs, how gates compose into the adder — the fundamental arithmetic primitive — and how adders assemble into the Arithmetic Logic Unit (ALU) that lives at the heart of every processor. By the end you'll know exactly what happens, at the transistor level, when your GPU executes a multiply-accumulate.

That multiply-accumulate — the MAC — is not an accident of hardware design. It is the *atom* of every matrix multiplication, every convolution, every attention score. A modern tensor core is nothing more exotic than thousands of MACs pipelined tightly together. Understanding the MAC from first principles explains why hardware accelerators are shaped the way they are, and why changing numeric precision from FP32 to INT8 yields the speedups it does.

> **The one-sentence version:** Logic gates, built from CMOS pairs, compose into adders; adders compose into multipliers and ALUs; and a Multiply-Accumulate unit (MAC) built from those pieces is the single instruction that underlies every neural-network computation on every chip ever built for AI.

---

## Boolean Algebra: The Language of Switches

A switch is either **on** (1, True, HIGH, Vdd) or **off** (0, False, LOW, GND). That binary choice maps perfectly onto **Boolean algebra**, developed by George Boole in 1854 — decades before the transistor existed.

Three primitive operations are sufficient to express every Boolean function:

| Operation | Symbol | Spoken | Rule |
|-----------|--------|--------|------|
| AND | $A \cdot B$ or `A & B` | "A and B" | Output 1 only if *both* inputs are 1 |
| OR | $A + B$ or `A \| B` | "A or B" | Output 1 if *either* input is 1 |
| NOT | $\overline{A}$ or `~A` | "not A" | Output is the opposite of input |

**De Morgan's laws** are the most useful algebraic identity you'll use when reading hardware datasheets and HDL:

$$\overline{A \cdot B} = \overline{A} + \overline{B} \qquad \text{(NAND = not-AND)} $$

$$\overline{A + B} = \overline{A} \cdot \overline{B} \qquad \text{(NOR = not-OR)} $$

These tell you that **NAND and NOR are the complements of AND and OR**, and they have a deeper significance: both are *functionally complete*, meaning you can build any Boolean function from NAND gates (or NOR gates) alone.

## Building Gates from CMOS Transistor Pairs

Recall from Chapter 1: an **NMOS** transistor conducts when its gate is HIGH; a **PMOS** transistor conducts when its gate is LOW. CMOS design places them in complementary pairs — a **pull-up network** (PMOS, connects output to Vdd=1) and a **pull-down network** (NMOS, connects output to GND=0). Exactly one network conducts for any given input.

### The Inverter (NOT Gate): 2 Transistors

```text
         Vdd (1)
          |
       ┌──┴──┐
  A ───┤ PMOS├─── out        A=0 → PMOS on  → out=1
       └──┬──┘
          │
       ┌──┴──┐
  A ───┤ NMOS├─── out        A=1 → NMOS on  → out=0
       └──┬──┘
         GND (0)
```

One PMOS + one NMOS = one inverter. At every moment, exactly one transistor is on — no current bleeds from Vdd to GND, so static power is near zero.

### NAND Gate: 4 Transistors

NAND is the natural primitive of CMOS because the PMOS pull-up network places transistors in **parallel** (either input can pull high), while the NMOS pull-down network places them in **series** (both inputs must be high to pull low). Parallel PMOS = few transistors; series NMOS = few transistors. This is the minimum transistor count for a two-input gate.

```text
        Vdd
       / \
   PMOS  PMOS        (A in parallel)
  (A)    (B)
       | |
       out
       |
   NMOS (A)          (A in series)
       |
   NMOS (B)
       |
      GND
```

| A | B | NAND (out) |
|:-:|:-:|:----------:|
| 0 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**NAND is universal:** you can build NOT, AND, OR, NOR, and any other gate from NAND gates alone. In practice, the standard-cell libraries EDA tools build from contain hundreds of cells (NAND, NOR, AND-OR-INVERT, multiplexers, flip-flops, each in several drive strengths), but inverting gates like NAND dominate because they need the fewest transistors for a given speed.

### NOR Gate: 4 Transistors

NOR inverts the topology: PMOS in **series** (both must be low to pull high), NMOS in **parallel** (either high pulls low). NOR is also universal.

### Why NAND is the Preferred Primitive

<div class="diagram">
<div class="diagram-title">NAND vs NOR: Transistor Topology</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">NAND (preferred)</div>
    <ul>
      <li>PMOS in <strong>parallel</strong> → stronger pull-up</li>
      <li>NMOS in <strong>series</strong> → accepted penalty</li>
      <li>PMOS is slower; parallel PMOS compensates</li>
      <li>Overall faster, lower area</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">NOR</div>
    <ul>
      <li>PMOS in <strong>series</strong> → weak pull-up</li>
      <li>NMOS in <strong>parallel</strong></li>
      <li>Series PMOS is slow — NOR is inherently slower</li>
      <li>Used less often in combinational logic</li>
    </ul>
  </div>
</div>
</div>

> PMOS transistors are inherently slower than NMOS (hole mobility ≈ 450 cm²/V·s vs electron mobility ≈ 1400 cm²/V·s in silicon). Placing the slow PMOS transistors in parallel (as in NAND) minimises the penalty. Series PMOS in NOR incurs the penalty multiplicatively.

### Full Gate Library from NAND

```python
def nand(a: int, b: int) -> int:
    return 1 - (a & b)   # NAND = NOT (A AND B)

def inv(a: int) -> int:
    return nand(a, a)    # NOT via NAND: NAND(a,a) = ¬(a·a) = ¬a

def and_gate(a: int, b: int) -> int:
    return inv(nand(a, b))   # AND = NOT(NAND)

def or_gate(a: int, b: int) -> int:
    return nand(inv(a), inv(b))   # De Morgan: ¬(¬a · ¬b) = a + b

def xor_gate(a: int, b: int) -> int:
    n = nand(a, b)
    return nand(nand(a, n), nand(b, n))  # XOR from 4 NANDs

# Verify truth tables
for a in (0,1):
    for b in (0,1):
        print(f"a={a} b={b}  AND={and_gate(a,b)}  OR={or_gate(a,b)}  XOR={xor_gate(a,b)}")
```

XOR is particularly important — you'll see it again immediately, because XOR implements 1-bit addition.

---

## Combinational Logic: Gates That Compute

**Combinational logic** has no memory: its output is a pure function of its current inputs. This includes all arithmetic and encoding circuits.

### Multiplexer (MUX): Selecting Between Values

A **2:1 MUX** routes one of two inputs to the output based on a select line $S$:

$$\text{out} = (A \cdot \overline{S}) + (B \cdot S)$$

A 32:1 MUX selects 1 of 32 values — this is how processors read from a **register file** (32 or 64 registers, and the instruction encodes which one you want). MUXes are everywhere.

### Decoder: Address to One-Hot

A 3-to-8 decoder takes a 3-bit address (0–7) and drives exactly one of 8 output lines high. This is how memory arrays, register files, and cache banks are addressed.

### The Half-Adder: Adding Two Bits

A **half-adder** takes two single bits and produces a **Sum** (the LSB of the result) and a **Carry** (the overflow into the next bit position):

$$\text{Sum} = A \oplus B \qquad \text{Carry} = A \cdot B$$

```text
 A ─┬─[XOR]─ Sum
    │   /
 B ─┴─[AND]─ Carry
```

```python
def half_adder(a: int, b: int) -> tuple[int, int]:
    """Returns (sum, carry)."""
    return xor_gate(a, b), and_gate(a, b)

assert half_adder(0, 0) == (0, 0)
assert half_adder(0, 1) == (1, 0)
assert half_adder(1, 1) == (0, 1)  # 1+1 = 10 in binary: sum=0, carry=1
```

### The Full-Adder: Adding Three Bits (the key building block)

A half-adder cannot chain: it ignores the carry-in from the previous column. A **full-adder** adds three bits — $A$, $B$, and $C_{in}$ — producing a sum and $C_{out}$:

$$\text{Sum} = A \oplus B \oplus C_{in} \qquad C_{out} = (A \cdot B) + (B \cdot C_{in}) + (A \cdot C_{in})$$

| $A$ | $B$ | $C_{in}$ | Sum | $C_{out}$ |
|:---:|:---:|:--------:|:---:|:---------:|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 1 | 0 |
| 0 | 1 | 0 | 1 | 0 |
| 0 | 1 | 1 | 0 | 1 |
| 1 | 0 | 0 | 1 | 0 |
| 1 | 0 | 1 | 0 | 1 |
| 1 | 1 | 0 | 0 | 1 |
| 1 | 1 | 1 | 1 | 1 |

A full-adder can be built from two half-adders plus an OR gate:

```python
def full_adder(a: int, b: int, cin: int) -> tuple[int, int]:
    """Returns (sum, carry_out)."""
    s1, c1 = half_adder(a, b)
    s2, c2 = half_adder(s1, cin)
    cout = or_gate(c1, c2)
    return s2, cout
```

### Ripple-Carry Adder: Chaining Full-Adders

To add two **N-bit** integers, chain N full-adders with the $C_{out}$ of each feeding the $C_{in}$ of the next:

```text
Bit 0:  FA(A[0], B[0], 0)   → Sum[0], C[1]
Bit 1:  FA(A[1], B[1], C[1]) → Sum[1], C[2]
  ...
Bit N-1: FA(A[N-1], B[N-1], C[N-1]) → Sum[N-1], C_overflow
```

This is a **ripple-carry adder**. Simple, but the carry *ripples* through — bit N-1 can't finish until bit 0 is done. Propagation delay scales as $O(N)$, which matters for wide data paths.

```python
def ripple_carry_adder(a_bits: list[int], b_bits: list[int]) -> list[int]:
    """
    Add two N-bit numbers represented as lists of bits (LSB first).
    Returns (N+1)-bit result (LSB first), where index N is the overflow carry.
    """
    assert len(a_bits) == len(b_bits)
    result, carry = [], 0
    for a, b in zip(a_bits, b_bits):
        s, carry = full_adder(a, b, carry)
        result.append(s)
    result.append(carry)   # overflow bit
    return result

def int_to_bits(n: int, width: int) -> list[int]:
    return [(n >> i) & 1 for i in range(width)]   # LSB first

def bits_to_int(bits: list[int]) -> int:
    return sum(b << i for i, b in enumerate(bits))

# Test: 13 + 11 = 24
a, b = int_to_bits(13, 8), int_to_bits(11, 8)
result = ripple_carry_adder(a, b)
assert bits_to_int(result) == 24
print(f"13 + 11 = {bits_to_int(result)}")   # 24 ✓
```

### Carry-Lookahead Adder (CLA): Breaking the O(N) Chain

The ripple-carry bottleneck inspired **carry-lookahead**: pre-compute whether each bit position will *generate* ($G_i = A_i \cdot B_i$) or *propagate* ($P_i = A_i \oplus B_i$) a carry, then compute all carries in parallel using 2-level logic. This reduces delay from $O(N)$ to $O(\log N)$, at the cost of more transistors.

Modern chips use 64-bit integer adders with carry-lookahead or parallel-prefix topologies (Kogge-Stone, Brent-Kung, Han-Carlson) that resolve all 64 carries in about log₂64 = 6 prefix levels. The whole adder settles in roughly **100 ps** at a 5 nm-class node, which is why a 5 GHz CPU core can feed one add's result into the next in a single 200 ps cycle.

---

## The ALU: Composition of Adders and Logic

The **Arithmetic Logic Unit (ALU)** is the circuit that performs arithmetic (add, subtract, compare) and logic (AND, OR, XOR, NOT, shift) operations. It is the computational core of every processor.

<div class="diagram">
<div class="diagram-title">ALU Block Diagram</div>
<div class="flow-h">
  <div class="flow-node accent">A (operand)<br/><small>N bits</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">ALU<br/><small>add/sub/and/or/xor/shift</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">Result<br/><small>N bits</small></div>
</div>
</div>

A simplified ALU uses a **MUX** to select which operation's output to pass through:

```python
def alu(a: int, b: int, op: str, width: int = 8) -> int:
    """
    Minimal ALU: add, sub, and, or, xor.
    Operates on unsigned integers; 'sub' uses two's complement: a - b = a + (~b + 1).
    """
    mask = (1 << width) - 1
    if   op == 'add': return (a + b) & mask
    elif op == 'sub': return (a - b) & mask
    elif op == 'and': return (a & b) & mask
    elif op == 'or' : return (a | b) & mask
    elif op == 'xor': return (a ^ b) & mask
    else: raise ValueError(f"Unknown op: {op}")

# Subtraction via addition: 20 - 7 = 20 + (-7) = 20 + (two's complement of 7)
print(alu(20, 7, 'sub', width=8))   # 13 ✓
```

The key insight: **subtraction is addition with the two's complement**. The adder circuit does not change; only the input is modified. Chapter 3 explains why two's complement makes this work so elegantly.

---

## Sequential Logic: Adding Memory

Combinational circuits are stateless — outputs change instantly with inputs. But computation requires **state**: registers, counters, program counters, memory. We need a circuit that *remembers* its last value. This is **sequential logic**.

### The SR Latch: Two Cross-Coupled NOR Gates

The simplest memory element is the **SR (Set-Reset) latch**, built from two NOR gates with outputs cross-coupled as inputs:

```text
  S ──┬──[NOR]── Q
      └── ↑  └───┐
           │      │
  R ──┬──[NOR]── Q̄
      └──────────┘
```

- **S=1, R=0:** Q is forced to 1 (Set).
- **S=0, R=1:** Q is forced to 0 (Reset).
- **S=0, R=0:** Q holds its previous value. This is the "memory" state.
- **S=1, R=1:** Invalid (undefined) — forbidden state.

The latch *remembers* because each NOR gate's output feeds back to the other's input — once set to 1, the circuit self-reinforces.

### The D Flip-Flop: Capturing a Value on a Clock Edge

The SR latch is **level-sensitive**: the output can change whenever the inputs are active. Digital systems need something cleaner — a circuit that captures its input **exactly once, on a clock edge**, and holds that value until the next edge. This is the **D flip-flop** (DFF).

```text
       ___     ___
CLK  _|   |___|   |___    (rising edge at ↑)

                 ↑── Q captures D here
D    ──────X─────────X──
Q    ──────────X──────── (holds until next ↑)
```

A DFF is built from two SR latches in a master-slave configuration, gated by the clock and its inverse. It is the fundamental unit of **state storage** in digital systems.

### Registers: Banks of Flip-Flops

A **register** is simply N D flip-flops clocked together, storing N bits as one unit. An 8-bit register stores one byte; a 64-bit register stores one 64-bit integer or float64.

A **register file** is an array of M registers with addressing logic (decoders + MUXes) to read/write any register by index. A 32-register 64-bit register file (typical in an in-order RISC CPU, see [Chapter 5](./05_instruction_sets.md)) stores 32 × 64 = 2,048 bits = 256 bytes total.

---

## The Clock and Edge-Triggering

<div class="diagram">
<div class="diagram-title">The Clock: Heartbeat of a Digital System</div>
<div class="flow-h">
  <div class="flow-node accent">Clock generator<br/><small>crystal oscillator</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">Clock distribution<br/><small>H-tree buffer network</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">All flip-flops<br/><small>capture simultaneously</small></div>
</div>
</div>

The **clock** is a square wave toggling between 0 and 1 at a fixed frequency — 3 GHz means 3×10⁹ cycles per second, i.e., one cycle every **~333 ps (0.333 ns)**.

At each rising edge, every flip-flop in the chip captures its input and updates its output **simultaneously** (within clock skew tolerances). Between edges, combinational logic computes new values. This synchronous model is what makes chip design tractable: all timing is relative to a single reference event.

### Propagation Delay and the Critical Path

Every gate takes a finite time to produce its correct output after inputs change — its **propagation delay** ($t_{pd}$). Order-of-magnitude values at a 5 nm-class node (they vary with load, supply voltage, temperature and cell library):

| Gate type | Propagation delay |
|-----------|-------------------|
| Inverter driving four copies of itself (the "FO4" yardstick) | ~5–10 ps |
| 2-input NAND/NOR | ~8–15 ps |
| Full-adder (1 bit) | ~25–40 ps |
| 64-bit parallel-prefix adder | ~80–150 ps |
| 64-bit integer multiplier | ~400–600 ps, so CPUs pipeline it over 3–4 cycles |

Designers often count delay in **FO4s**: a high-frequency CPU pipeline stage is roughly 15–20 FO4 deep, which is why clock rates top out around 5–6 GHz.

The **critical path** is the longest chain of gate delays between any two flip-flops. This path determines the minimum clock period — you cannot run the clock faster than the critical path allows, or the flip-flop captures the wrong (still-settling) value.

$$f_{\text{max}} = \frac{1}{t_{\text{critical path}} + t_{\text{setup}} + t_{\text{clock skew}}}$$

A 3 GHz chip has a clock period of 333 ps. If the critical path is 280 ps and setup + skew consume 40 ps, there is 13 ps of slack. This is why **timing closure** — ensuring every critical path is short enough — is one of the central challenges in chip design ([Chapter 31](./31_chip_design_flow.md)).

> Lowering the supply voltage makes every gate slower, so the critical path gets longer and the clock must come down with it. In exchange, switching energy falls with $V^2$. This is the basis of DVFS (Dynamic Voltage-Frequency Scaling) — a technique used by every modern inference accelerator to hit the right power/performance point.

---

## From Adder to MAC: The Workhorse of Neural Networks

So far we have addition. Neural network inference requires **Multiply-Accumulate**: $\text{acc} \mathrel{+}= A \times B$. Two more circuits complete the picture.

### The Multiplier: Partial Products

Multiplying two N-bit numbers $A \times B$ is structurally similar to long multiplication by hand: compute N **partial products** (each is $A$ AND-ed with one bit of $B$, shifted left by its bit position), then **add all partial products together**.

For 8-bit × 8-bit:

```python
def multiply(a: int, b: int, width: int = 8) -> int:
    """Unsigned N×N-bit multiply via partial products (educational, not hardware-optimal)."""
    mask = (1 << width) - 1
    a, b = a & mask, b & mask
    result = 0
    for i in range(width):
        if (b >> i) & 1:
            result += a << i   # add partial product
    return result & ((1 << (2 * width)) - 1)  # keep 2N-bit result

assert multiply(13, 11) == 143   # 13 × 11 = 143 ✓
```

Hardware multipliers are optimized with **Booth encoding** (halving the number of partial products), **Wallace tree reduction** (summing partial products with carry-save adders in a tree rather than ripple chain), and a final fast adder. A straightforward 8-bit × 8-bit array multiplier needs 64 AND gates for the partial products and about 56 full-adders to sum them, roughly **2,000–3,000 transistors**, and settles in well under 100 ps at a modern node.

### The MAC Unit: Multiply Then Accumulate

```python
def mac(accumulator: int, a: int, b: int, width: int = 8) -> int:
    """accumulator += a * b (integer, overflow not checked)."""
    return accumulator + multiply(a, b, width)

# A dot-product: sum(a[i]*b[i]) — the inner loop of every matmul
def dot_product(vec_a: list[int], vec_b: list[int]) -> int:
    acc = 0
    for a, b in zip(vec_a, vec_b):
        acc = mac(acc, a, b)
    return acc

a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
print(dot_product(a, b))   # 1×5 + 2×6 + 3×7 + 4×8 = 70 ✓
```

<div class="diagram">
<div class="diagram-title">Building a MAC from First Principles</div>
<div class="flow">
  <div class="flow-node accent wide">CMOS transistors (NAND/NOR gates)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">XOR + AND → Half-adder → Full-adder → Ripple-carry adder</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">Partial products + Wallace tree → Multiplier (N×N bits)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Multiplier + Accumulator register → MAC unit</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Array of MACs, clocked in parallel → Tensor Core / Matrix Unit</div>
</div>
</div>

The first Tensor Core (Volta, 2017) performed a 4×4×4 matrix multiply-accumulate, 64 FMAs, per clock. Each generation widened it: an H100 SXM has **528 Tensor Cores** (4 in each of 132 Streaming Multiprocessors), and each one performs **512 dense FP16/BF16 FMAs per clock**. At the 1.83 GHz clock NVIDIA uses for its throughput figures:

$$528 \text{ TCs} \times 512 \text{ FMA/TC/cycle} \times 2 \text{ FLOP/FMA} \times 1.83 \text{ GHz} \approx 989 \text{ TFLOPS (dense BF16)}$$

That matches NVIDIA's datasheet number, and the FP8 rate (1,979 TFLOPS) is exactly twice it, because an 8-bit multiplier is a quarter of the area of a 16-bit one (see the bars below).

This headline number is just thousands of the full-adder circuits you just built, running at GHz speeds.

---

## A Complete Python Simulation

Here is everything composed end-to-end: from individual NAND gates through an 8-bit adder, multiplier, and MAC:

```python
import time

# ── Gate primitives (NAND-universal) ──────────────────────────────────────────
def nand2(a, b): return 1 - (a & b)
def inv(a):      return nand2(a, a)
def and2(a, b):  return inv(nand2(a, b))
def or2(a, b):   return nand2(inv(a), inv(b))
def xor2(a, b):
    n = nand2(a, b)
    return nand2(nand2(a, n), nand2(b, n))

# ── Adder ─────────────────────────────────────────────────────────────────────
def half_adder(a, b):      return xor2(a, b), and2(a, b)
def full_adder(a, b, cin):
    s1, c1 = half_adder(a, b)
    s2, c2 = half_adder(s1, cin)
    return s2, or2(c1, c2)

def add_n(x: int, y: int, width: int = 8) -> int:
    carry, result = 0, 0
    for i in range(width):
        s, carry = full_adder((x >> i) & 1, (y >> i) & 1, carry)
        result |= (s << i)
    return result  # ignores final carry for simplicity

# ── Multiplier (partial products) ────────────────────────────────────────────
def mul_n(a: int, b: int, width: int = 8) -> int:
    acc = 0
    for i in range(width):
        if (b >> i) & 1:
            acc = add_n(acc, a << i, width=2 * width)
    return acc

# ── MAC unit ─────────────────────────────────────────────────────────────────
def mac_unit(acc: int, a: int, b: int) -> int:
    return add_n(acc, mul_n(a, b), width=32)

# ── Vector dot-product (inner loop of matmul) ─────────────────────────────────
def dot(va: list[int], vb: list[int]) -> int:
    acc = 0
    for a, b in zip(va, vb):
        acc = mac_unit(acc, a, b)
    return acc

# Test
import random
random.seed(42)
n = 16
va = [random.randint(0, 15) for _ in range(n)]   # 4-bit values
vb = [random.randint(0, 15) for _ in range(n)]
expected = sum(a * b for a, b in zip(va, vb))
got = dot(va, vb)
print(f"dot product: expected={expected}, gate-level={got}, match={expected==got}")
```

---

## Why This Matters for Model Optimization

Every matrix multiplication and convolution in your model ultimately decomposes into dot products, and every dot product decomposes into MACs. This single insight drives a large fraction of hardware design choices:

<div class="diagram">
<div class="diagram-title">MAC Arithmetic → Hardware Decisions</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Precision = Transistor Count</div>
    <div class="card-desc">A 16-bit multiplier needs ~4× more transistors than an 8-bit one. Halving precision roughly quadruples the MACs you can fit per mm². On H100 this shows up as INT8/FP8 at 2× the BF16 tensor rate and 4× TF32.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Critical Path = Clock Limit</div>
    <div class="card-desc">A wider multiplier has a longer critical path. FP32 multiplication (23-bit mantissa multiply) has a longer critical path than INT8 — another reason lower precision enables higher throughput at the same clock.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Accumulator Width Matters</div>
    <div class="card-desc">Even when inputs are INT8, the accumulator must be wider (INT32) to avoid overflow across a long dot product. An INT16 accumulator would overflow after a handful of products, so INT8 tensor cores accumulate in INT32, and FP8 ones in FP32 (or slightly narrower internally).</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Tensor Core = MAC Array</div>
    <div class="card-desc">An NVIDIA tensor core is a wired-up matrix of MACs with matched input staging. Its "shape" (4×4, 8×8, 16×16 tiles) directly dictates the tile sizes you should use in CUDA kernels and Flash Attention variants.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Fused Multiply-Add (FMA)</div>
    <div class="card-desc">Modern FPUs compute a×b+c with one rounding step (not two). cuBLAS GEMMs and convolutions are built from FMAs, which is one reason fused kernels are usually slightly more accurate than separate multiply-then-add.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Sparsity Skips MACs</div>
    <div class="card-desc">If a weight is zero, its MAC contributes nothing. 2:4 structured sparsity (Ch 26) physically skips 50% of the multipliers in the tensor core, halving compute. The hardware "knows" zeros are zero — it doesn't just compute a zero product.</div>
  </div>
</div>
</div>

<div class="specbars">
<div class="sb-title">First-order multiplier area ∝ (significand bits)²</div>
<div class="sb-row"><span class="sb-name">FP32 (24-bit significand)</span><span class="sb-track"><i class="k5" style="--v:1"></i></span><span class="sb-val">576 · 9.0×</span></div>
<div class="sb-row"><span class="sb-name">FP16 / TF32 (11)</span><span class="sb-track"><i class="k2" style="--v:0.21"></i></span><span class="sb-val">121 · 1.9×</span></div>
<div class="sb-row"><span class="sb-name">BF16 (8)</span><span class="sb-track"><i style="--v:0.111"></i></span><span class="sb-val">64 · 1.0×</span></div>
<div class="sb-row"><span class="sb-name">INT8 (8)</span><span class="sb-track"><i style="--v:0.111"></i></span><span class="sb-val">64 · 1.0×</span></div>
<div class="sb-row"><span class="sb-name">FP8 E4M3 (4)</span><span class="sb-track"><i class="k4" style="--v:0.028"></i></span><span class="sb-val">16 · 0.25×</span></div>
<div class="sb-row"><span class="sb-name">FP4 E2M1 (2)</span><span class="sb-track"><i class="k4" style="--v:0.007"></i></span><span class="sb-val">4 · 0.06×</span></div>
<div class="sb-note">An N×N-bit array multiplier has N² partial-product bits, so area and energy grow with the square of the significand width (implicit leading 1 included), relative to INT8. Exponent adders, alignment shifters and accumulators add a smaller, roughly linear cost; <a href="./24_datatypes_and_silicon.html">Chapter 24</a> follows this through to real tensor cores.</div>
</div>

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>BF16 and INT8 need multipliers of about the same size, yet INT8 tensor throughput is usually quoted at twice BF16's. Where does the other half of the BF16 cost go? (Hint: think about what has to happen to the exponents before two floating-point products can be added.)</p>
</div>

---

## Key Takeaways

- **Boolean algebra** (AND/OR/NOT, De Morgan) is the language of transistor networks; CMOS NAND gates are the natural, minimal, universal primitive.
- A **full-adder** from XOR and AND gates chains into a **ripple-carry adder**; carry-lookahead cuts delay from O(N) to O(log N).
- The **ALU** composes adders and logic units, selected by a MUX; subtraction is just two's-complement addition.
- **Flip-flops** (D-type, edge-triggered) are the storage cells; registers are arrays of flip-flops. The clock synchronises all state updates.
- The **critical path** through combinational logic sets the maximum clock frequency ($f_{max} = 1/(t_{critical} + t_{setup} + t_{skew})$); a 64-bit adder settles in ~100 ps at a 5 nm-class node.
- A **Multiply-Accumulate (MAC) unit** — multiplier + accumulator — is the atom of every matmul and convolution.
- A tensor core is thousands of MACs; INT8 fits more MACs per mm² than FP32, directly explaining low-precision speedups.

## Check Yourself

<details class="check">
<summary>Why do CMOS designers prefer NAND to NOR as a building block?</summary>
<p>In NAND the slow PMOS transistors sit in parallel and the fast NMOS in series; NOR is the reverse. Holes are about 3× less mobile than electrons in silicon, so stacking PMOS in series (NOR) makes the pull-up much slower or forces much wider transistors.</p>
</details>

<details class="check">
<summary>A 3 GHz design has a 360 ps critical path. Name three ways to close timing.</summary>
<p>The period is 333 ps, so the path is ~30 ps too long. Options: split the logic across two pipeline stages (or retime registers into it); swap in faster cells or a faster adder topology; raise the supply voltage (faster gates, more power); or accept a lower clock, about 2.7 GHz.</p>
</details>

<details class="check">
<summary>Why does an INT8 dot product of length 4096 need a 32-bit accumulator?</summary>
<p>Each product can be as large as 127 × 127 = 16,129, and 4096 of them can sum to about 6.6 × 10⁷, which needs 27 bits. INT16 tops out at 32,767 and would overflow after three worst-case products.</p>
</details>

<details class="check">
<summary>Using the first-order N² rule, how much smaller is an FP8 E4M3 multiplier than a BF16 one?</summary>
<p>BF16 carries an 8-bit significand (7 stored + the implicit 1), E4M3 a 4-bit one: (8/4)² = 4× smaller. That is why FP8 tensor throughput is 2× BF16 at roughly the same area (the saving is shared with wider accumulation and conversion logic).</p>
</details>

<details class="check">
<summary>What does a carry-lookahead or prefix adder trade to beat a ripple-carry adder?</summary>
<p>Area and wiring. It computes generate and propagate signals for every bit and combines them in a log₂N-deep tree, so delay drops from O(N) to O(log N) at the cost of more gates and longer wires.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| CMOS VLSI Design: A Circuits and Systems Perspective, 4th ed. | N. Weste, D. Harris, 2010 | [pearson.com/en-us/subject-catalog/p/cmos-vlsi…](https://www.pearson.com/en-us/subject-catalog/p/cmos-vlsi-design-a-circuits-and-systems-perspective/P200000003426) | Gate topologies, FO4 delay, adder and multiplier design |
| Digital Design and Computer Architecture, RISC-V Edition | S. Harris, D. Harris, 2021 | [elsevier.com/books/digital-design-and-compute…](https://www.elsevier.com/books/digital-design-and-computer-architecture-risc-v-edition/harris/978-0-12-820064-3) | Boolean algebra through flip-flops, at the level used here |
| A Parallel Algorithm for the Efficient Solution of a General Class of Recurrence Equations | P. Kogge, H. Stone, IEEE TC 1973 | [doi.org/10.1109/TC.1973.5009159](https://doi.org/10.1109/TC.1973.5009159) | The parallel-prefix adder |
| A Suggestion for a Fast Multiplier | C. S. Wallace, IEEE TEC 1964 | [doi.org/10.1109/PGEC.1964.263830](https://doi.org/10.1109/PGEC.1964.263830) | Carry-save reduction trees for partial products |
| A Signed Binary Multiplication Technique | A. D. Booth, 1951 | [doi.org/10.1093/qjmam/4.2.236](https://doi.org/10.1093/qjmam/4.2.236) | Booth recoding: halving the partial products |
| NVIDIA H100 Tensor Core GPU Architecture | NVIDIA, 2022 | [resources.nvidia.com/en-us-tensor-core/gtc22-…](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper) | Tensor Core count, per-clock FMA rate, 989 TFLOPS dense BF16 |

---

**Next:** [Chapter 3 — Number Representation](./03_number_representation.md) — We examine the bit patterns that flow through these adders and MACs — from unsigned integers to two's complement, fixed-point, and IEEE-754 floating point.

[← Back to Table of Contents](./README.md)
