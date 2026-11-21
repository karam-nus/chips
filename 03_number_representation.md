---
title: "Chapter 3 — Number Representation"
---

[← Back to Table of Contents](./README.md)

# Chapter 3 — Number Representation

The gates and adders from [Chapter 2](./02_transistors_to_logic.md) operate on bits. But what *are* those bits representing? When PyTorch stores a weight tensor as `torch.float32`, exactly which pattern of 32 bits is written into memory, and why? When you quantize to `int8`, what do those 8 bits mean, and what is lost? This chapter builds number representation from first principles: unsigned binary, signed integers, fixed-point, and finally floating point — all the way through to the IEEE-754 standard that governs every `float32` computation you have ever run.

This chapter is the **foundation for Chapter 23** ([Number Formats Deep Dive](./23_number_formats_deep_dive.md)), which extends these ideas to FP16, BF16, FP8, FP4, and microscaling formats. Every analysis there assumes you understand the material here. We will write Python code that reaches inside NumPy arrays and reads the raw bits — this is not abstract; it is the exact binary representation sitting in GPU memory while your model runs.

> **The one-sentence version:** Every number stored in a chip is a fixed-width pattern of bits; the *interpretation scheme* — unsigned, two's complement, fixed-point, or floating-point — determines the mapping between bit pattern and mathematical value, and that choice governs precision, dynamic range, overflow behaviour, and ultimately your model's accuracy and speed.

---

## Bits, Bytes, and Words

The **bit** is the atom: 0 or 1. Eight bits form one **byte** (8 b). Beyond that, widths are called **words** and vary by context:

| Name | Bits | Bytes | Common use |
|------|:----:|:-----:|------------|
| bit | 1 | — | A single binary digit |
| nibble | 4 | 0.5 | One hex digit; sometimes used in INT4 inference |
| byte | 8 | 1 | Char, INT8, the universal addressable unit |
| halfword / short | 16 | 2 | INT16, FP16, BF16 |
| word | 32 | 4 | INT32, FP32, the default "float" in C |
| doubleword / long | 64 | 8 | INT64, FP64 (double precision) |
| quadword | 128 | 16 | One SSE/NEON SIMD register (e.g. 4 × FP32 lanes) |

The width of a data type directly determines **memory consumption** and **bandwidth pressure**:

$$\text{model memory (bytes)} = \text{parameters} \times \text{bytes per parameter}$$

A 7-billion-parameter model in FP32 costs $7 \times 10^9 \times 4 = 28\,\text{GB}$; in INT8 it is 7 GB; in INT4 it is 3.5 GB. The bit width is not an abstraction — it is the first thing that determines whether a model fits on your GPU.

---

## Unsigned Binary

**Unsigned binary** encodes non-negative integers directly: each bit position $i$ contributes $\text{bit}_i \times 2^i$.

$$V = \sum_{i=0}^{N-1} b_i \cdot 2^i$$

```python
import numpy as np

def uint_to_bin(n: int, width: int = 8) -> str:
    """Show the binary representation of an unsigned integer."""
    return format(n, f'0{width}b')

print(uint_to_bin(0,   8))   # 00000000
print(uint_to_bin(1,   8))   # 00000001
print(uint_to_bin(127, 8))   # 01111111
print(uint_to_bin(255, 8))   # 11111111  ← maximum for uint8
```

For an N-bit unsigned integer: range is $[0,\; 2^N - 1]$. A `uint8` spans 0–255; a `uint16` spans 0–65535.

**Hex** (base 16) is a shorthand: each hex digit maps to exactly 4 bits. `0xFF` = `0b11111111` = 255. Reading hex is a core hardware skill — register dumps, memory addresses, and bit masks are all expressed in hex.

| Hex | Binary | Decimal |
|:---:|:------:|:-------:|
| 0 | 0000 | 0 |
| 7 | 0111 | 7 |
| 8 | 1000 | 8 |
| F | 1111 | 15 |
| FF | 11111111 | 255 |
| 1A3B | 0001 1010 0011 1011 | 6715 |

---

## Signed Integers: Two's Complement

### Sign-Magnitude: The Obvious Approach That Fails

The naive way to represent negative numbers: use bit N-1 as a sign bit, the rest as magnitude.

| Bit pattern | Signed (sign-mag) |
|:-----------:|:-----------------:|
| `0000 0001` | +1 |
| `1000 0001` | −1 |
| `0000 0000` | +0 |
| `1000 0000` | −0 ← **two zeros!** |

Sign-magnitude has two representations of zero and requires special-case hardware for addition of mixed-sign values. It was used in some early machines but is now essentially obsolete for integers.

### Two's Complement: One Zero, Free Addition

**Two's complement** is the universal signed integer encoding used by every modern processor, every GPU, every quantization scheme. Its two key properties:

1. **One zero**: `00…0` is the only zero.
2. **Addition just works**: signed addition uses the *same* adder circuit as unsigned addition, with no special cases.

**Encoding rule:** for a positive number, same as unsigned. For a negative number $-x$ ($x > 0$): flip all bits of $x$, then add 1.

$$-x = \sim x + 1 \quad \text{(bitwise NOT, then +1)}$$

```python
def twos_complement(x: int, width: int = 8) -> int:
    """Encode signed integer x in two's complement with `width` bits."""
    mask = (1 << width) - 1
    if x >= 0:
        return x & mask
    else:
        return ((~(-x)) + 1) & mask   # flip bits of magnitude, add 1

def from_twos_complement(bits: int, width: int = 8) -> int:
    """Decode a two's complement bit pattern to a Python signed int."""
    if bits >= (1 << (width - 1)):   # MSB is 1 → negative
        return bits - (1 << width)
    return bits

# Encoding
for v in [0, 1, 127, -1, -127, -128]:
    enc = twos_complement(v, 8)
    dec = from_twos_complement(enc, 8)
    print(f"{v:5d} → 0b{enc:08b} (0x{enc:02X}) → decoded back: {dec}")
```

Output:
```
    0 → 0b00000000 (0x00) → decoded back: 0
    1 → 0b00000001 (0x01) → decoded back: 1
  127 → 0b01111111 (0x7F) → decoded back: 127
   -1 → 0b11111111 (0xFF) → decoded back: -1
 -127 → 0b10000001 (0x81) → decoded back: -127
 -128 → 0b10000000 (0x80) → decoded back: -128
```

For an N-bit two's complement integer: range is $[-2^{N-1},\; 2^{N-1} - 1]$.

| Type | Width | Min | Max |
|------|:-----:|----:|----:|
| INT4 | 4 | −8 | 7 |
| INT8 | 8 | −128 | 127 |
| INT16 | 16 | −32,768 | 32,767 |
| INT32 | 32 | −2,147,483,648 | 2,147,483,647 |

**Why does addition "just work"?** Because two's complement wraps modulo $2^N$. Adding −1 (= `0xFF` for 8-bit) to +1 (= `0x01`):

```
  0000 0001  (+1)
+ 1111 1111  (-1 in two's complement)
= 0000 0000  (0, carry discarded)  ✓
```

The adder from Chapter 2 does not need to know the numbers are signed. This is the elegant payoff.

### Overflow and Wraparound

```python
# NumPy INT8 overflow wraps around (two's complement)
a = np.array([127], dtype=np.int8)
b = np.array([1],   dtype=np.int8)
print(a + b)   # [-128]  ← 127 + 1 overflows to -128

# Unsigned overflow
a8 = np.array([255], dtype=np.uint8)
print(a8 + np.uint8(1))   # [0]  ← 255 + 1 wraps to 0
```

Overflow is silent in hardware (no exception is raised unless the ISA has overflow flags). This matters for quantization: if your INT8 accumulator overflows, you get wrong answers — which is why quantized neural network frameworks clip values to the representable range and use wider accumulators (INT32) for partial sums.

---

## Fixed-Point Representation

**Fixed-point** numbers are integers with an *implicit* binary point at a fixed position. If you have an N-bit integer but agree that the binary point is M positions from the right, the value is:

$$V = \text{integer\_value} \times 2^{-M}$$

The **Q-format** notation (common in DSPs and embedded systems) writes this as **QI.F** or just **QM** where I is the number of integer bits (including sign) and F is the fractional bits:

| Format | Bits | Range | Resolution |
|--------|:----:|------:|----------:|
| Q1.15 (Q15) | 16 | [−1, +1 − 2⁻¹⁵] | $2^{-15} \approx 3.05 \times 10^{-5}$ |
| Q8.8 | 16 | [−128, +128 − 2⁻⁸] | $2^{-8} = 0.00391$ |
| Q1.7 (INT8 interpreted as a fraction) | 8 | [−1, +1 − 2⁻⁷] | $2^{-7} = 0.0078$ |

```python
def fixed_point_encode(value: float, fractional_bits: int, total_bits: int = 16) -> int:
    """Encode a float as Q-format fixed-point integer."""
    scale = 1 << fractional_bits                 # 2^F
    max_val = (1 << (total_bits - 1)) - 1        # e.g. 32767 for Q1.15
    min_val = -(1 << (total_bits - 1))           # e.g. -32768
    int_val = round(value * scale)
    int_val = max(min_val, min(max_val, int_val))  # clip to range
    return int_val & ((1 << total_bits) - 1)     # mask to N bits

def fixed_point_decode(bits: int, fractional_bits: int, total_bits: int = 16) -> float:
    """Decode Q-format fixed-point integer to float."""
    signed_val = from_twos_complement(bits, total_bits)
    return signed_val / (1 << fractional_bits)

# Q1.15: represent -0.5 and 0.9999694 (max representable near 1.0)
enc = fixed_point_encode(-0.5, fractional_bits=15)
print(f"-0.5  → 0x{enc:04X} → {fixed_point_decode(enc, 15):.7f}")

enc2 = fixed_point_encode(0.99995, fractional_bits=15)
print(f"~1.0  → 0x{enc2:04X} → {fixed_point_decode(enc2, 15):.7f}")
```

**Why fixed-point matters for ML:** Fixed-point is the basis of INT8/INT4 quantization. When you quantize a weight tensor with scale $s$ and zero-point $z$:

$$x_{\text{quant}} = \text{clip}\!\left(\text{round}\!\left(\frac{x}{s}\right) + z,\; -128,\; 127\right)$$

you are encoding a float as a scaled integer — exactly fixed-point arithmetic. The hardware does integer MACs, and the scaling is "applied" at the output. Fixed-point does not require special hardware; the same integer adder and multiplier from Chapter 2 handle it. The only requirement is that scale factors are tracked by software or baked into separate hardware paths.

---

## Floating Point from First Principles

Fixed-point is efficient but inflexible: all values use the same resolution. A weight tensor might span values from $10^{-6}$ to $10^{2}$ — fixed-point would need enormous precision everywhere to handle that dynamic range. **Floating-point** solves this by representing numbers in scientific notation, base 2:

$$V = (-1)^s \times 1.m \times 2^{e - \text{bias}}$$

Three fields in the bit pattern:

<div class="diagram">
<div class="diagram-title">Floating-Point Bit Layout (Generic)</div>
<div class="flow-h">
  <div class="flow-node accent">S<br/><small>sign<br/>1 bit</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">Exponent<br/><small>e bits<br/>biased</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">Mantissa (fraction)<br/><small>m bits<br/>implicit leading 1</small></div>
</div>
</div>

- **Sign (S):** 1 bit. 0 = positive, 1 = negative.
- **Exponent (E):** stored as an *unsigned* integer, but interpreted as $e = E - \text{bias}$ where $\text{bias} = 2^{k-1} - 1$ (for $k$-bit exponent). This allows exponents to represent both very small and very large numbers, with the midpoint of the exponent range corresponding to $e = 0$.
- **Mantissa / Fraction (M):** the fractional part of the significand. The **implicit leading 1** means the significand is $1.M$ (we never store the leading 1 bit, because for normalised numbers it is always 1 — saving one bit of precision for free). This is called the **hidden bit**.

### IEEE-754 FP32 (Single Precision)

The dominant format for ML training for decades.

```text
 31  30          23  22                         0
  S  EEEEEEEE        MMMMMMMMMMMMMMMMMMMMMMM
  ↑  ← 8 bits →     ←────── 23 bits ──────────→
```

| Field | Bits | Value formula |
|-------|:----:|---------------|
| Sign | 1 | $(-1)^S$ |
| Exponent | 8 | $e = E - 127$ (bias = 127) |
| Mantissa | 23 | $1.M$ (hidden bit; $M$ is the fractional part) |

**Value:** $(-1)^S \times 1.M \times 2^{E - 127}$ (for normalised numbers)

**Range and precision:**

| Property | FP32 | FP64 |
|----------|:----:|:----:|
| Sign bits | 1 | 1 |
| Exponent bits | 8 | 11 |
| Mantissa bits | 23 | 52 |
| Exponent bias | 127 | 1023 |
| Max normal | ≈ 3.4 × 10³⁸ | ≈ 1.8 × 10³⁰⁸ |
| Min normal | ≈ 1.2 × 10⁻³⁸ | ≈ 2.2 × 10⁻³⁰⁸ |
| Decimal precision | ~7 sig. digits | ~15 sig. digits |
| Size | 4 bytes / 32 bits | 8 bytes / 64 bits |

### Bit Inspection in Python

```python
import struct
import numpy as np

def inspect_float32(f: float) -> dict:
    """Decompose an IEEE-754 float32 into its three fields."""
    # Pack as 4 bytes, unpack as uint32
    bits = struct.unpack('>I', struct.pack('>f', f))[0]
    sign    = (bits >> 31) & 0x1
    exp_raw = (bits >> 23) & 0xFF
    mant    = bits & 0x7FFFFF
    # Decode
    exp_val = exp_raw - 127
    # Reconstruct significand
    if exp_raw == 0:      # subnormal or zero
        significand = mant / (1 << 23)   # no hidden bit
        exp_val = -126
    else:
        significand = 1 + mant / (1 << 23)  # hidden bit = 1
    value = ((-1) ** sign) * significand * (2 ** exp_val)
    return {
        'bits':         f'{bits:032b}',
        'sign':         sign,
        'exp_raw':      exp_raw,
        'exp_unbiased': exp_val,
        'mantissa_raw': f'{mant:023b}',
        'significand':  significand,
        'reconstructed': value
    }

for val in [1.0, -1.0, 0.5, 0.1, 3.14159, 1e-38, float('inf'), float('nan')]:
    d = inspect_float32(val)
    print(f"{str(val):10s}  bits={d['bits']}  "
          f"S={d['sign']} E={d['exp_raw']:3d}({d['exp_unbiased']:+4d}) "
          f"M=0b{d['mantissa_raw'][:8]}...  ≈{d['reconstructed']:.6g}")
```

Sample output (exact bits — not approximate):
```
1.0         bits=00111111100000000000000000000000  S=0 E=127(  0) M=0b00000000...  ≈1
-1.0        bits=10111111100000000000000000000000  S=1 E=127(  0) M=0b00000000...  ≈-1
0.5         bits=00111111000000000000000000000000  S=0 E=126( -1) M=0b00000000...  ≈0.5
0.1         bits=00111101110011001100110011001101  S=0 E=123( -4) M=0b11001100...  ≈0.1
```

Notice `0.1` has a repeating binary fraction — **0.1 is not exactly representable in binary floating point.** The error in one stored value is tiny (half a unit in the last place); what hurts in practice is rounding error *accumulating* over millions of operations, as the examples below show.

### Using NumPy for Bit Inspection

```python
def float32_bits(f: float) -> np.uint32:
    """Return the raw 32-bit integer interpretation of a float32."""
    return np.frombuffer(np.float32(f).tobytes(), dtype=np.uint32)[0]

def bits_to_float32(b: int) -> float:
    """Convert raw 32-bit integer back to float32."""
    return float(np.frombuffer(np.uint32(b).tobytes(), dtype=np.float32)[0])

# The smallest difference representable near 1.0 = machine epsilon
one        = float32_bits(1.0)       # 0x3F800000
one_plus   = float32_bits(1.0 + np.finfo(np.float32).eps)
print(f"1.0 = 0x{one:08X}")
print(f"1.0+eps = 0x{one_plus:08X}")
print(f"difference in ulps: {one_plus - one}")  # 1 ULP

# Absorption: adding a small number to a large one
a = np.float32(1e8)               # spacing between FP32 values near 1e8 is 8.0
b = np.float32(1.0)
print(np.spacing(a))              # 8.0
print(a + b)                      # 1e+08 — the 1.0 is below half a spacing and vanishes
print(a + b - a)                  # 0.0
# (At 1e7 the spacing is exactly 1.0, so 1e7 + 1 is still representable.)
```

---

## Biasing, Subnormals, and Special Values

### The Bias Convention

The exponent is stored as an unsigned integer $E$ but means $e = E - \text{bias}$. For FP32: bias = 127.

- $E = 0$: reserved for zero / subnormals.
- $E = 1$ through $E = 254$: normalised numbers. $e$ ranges from −126 to +127.
- $E = 255$: reserved for $\pm\infty$ and NaN.

This allows the stored exponent to be compared as an unsigned integer, making floating-point comparison nearly as fast as integer comparison — an important hardware optimization.

### Subnormal Numbers: Graceful Underflow

When $E = 0$ and mantissa $\neq 0$, the number is **subnormal** (also called denormalized). The hidden bit is now 0 (not 1), giving:

$$V_{\text{subnormal}} = (-1)^S \times 0.M \times 2^{-126}$$

Subnormals fill the gap between zero and the smallest normalised number, providing **gradual underflow** — errors shrink gracefully toward zero instead of suddenly jumping to zero. Their cost depends on the hardware: many x86 CPUs handle them with a microcode assist that can take on the order of 100 cycles per operation, while NVIDIA GPUs process FP32 and FP16 subnormals in the FMA pipeline at full speed.

> Compilers and kernels can **flush subnormals to zero** (FTZ), e.g. `nvcc --use_fast_math` or the x86 FTZ/DAZ flags. On a GPU that buys little speed but removes the bottom of the range, so values below the smallest normal (6.1 × 10⁻⁵ in FP16) become exactly zero. For FP16 gradients that is a real risk; loss scaling exists to keep them above it.

### Special Values

| Bit pattern | Value | Behaviour |
|-------------|-------|-----------|
| S=x, E=0, M=0 | $\pm 0$ | Two zeros; identical for arithmetic |
| S=x, E=0, M≠0 | $\pm\text{subnormal}$ | Gradual underflow |
| S=x, E=255, M=0 | $\pm\infty$ | Overflow; propagates through operations |
| S=x, E=255, M≠0 | NaN | "Not a Number"; $0/0$, $\infty - \infty$, etc. |

```python
print(np.float32('inf') * np.float32(0))   # nan
print(np.float32('inf') + np.float32(1))   # inf
print(np.float32(1) / np.float32(0))       # inf (with RuntimeWarning)
print(np.isnan(np.float32('nan')))          # True

# Subnormal territory
tiny_normal = np.float32(1.175494e-38)      # smallest normal FP32
print(tiny_normal / 2)                      # subnormal — still non-zero
print(tiny_normal / 2 == 0)                 # False (gradual underflow)
```

NaN propagation in training — a weight update producing `nan` then multiplying everything — is a common failure mode for FP16 mixed-precision training when loss scaling is misconfigured.

---

## Rounding Modes

When a result cannot be represented exactly, the hardware must **round**. IEEE-754 specifies five modes:

| Mode | Rule | ML Use |
|------|------|--------|
| Round to nearest, ties to even (RNE) | Default; round to closest, break ties toward even mantissa | FP32, FP64 default |
| Round toward +∞ | Always round up | Interval arithmetic |
| Round toward −∞ | Always round down | Interval arithmetic |
| Round toward zero (truncate) | Chop off extra bits | Sometimes faster |
| Round to nearest, ties away from zero | Common in school arithmetic | Rare in hardware |

A sixth scheme matters in ML even though IEEE-754 does not define it: **stochastic rounding**, which rounds up with probability equal to the fractional distance to the next value. It is unbiased *in expectation*, so many tiny updates that RNE would round away still move a low-precision weight on average. Graphcore IPUs and AWS Trainium support it in hardware, and several FP8/FP4 training recipes use it for gradients.

**Round to nearest ties to even (banker's rounding)** is the default for IEEE-754. "Ties to even" means when a result is exactly halfway between two representable values, round to whichever has an even LSB in the mantissa. This eliminates statistical bias in repeated rounding — important for long accumulations in ML training.

---

## Precision vs Dynamic Range: The Central Trade-Off

This is the most important design axis for ML practitioners:

<div class="diagram">
<div class="diagram-title">Precision vs Dynamic Range</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Precision (more mantissa bits)</div>
    <ul>
      <li>Finer resolution between adjacent values</li>
      <li>Less rounding error per operation</li>
      <li>Needed for gradient updates (small deltas)</li>
      <li>More bits → more memory, less throughput</li>
      <li>FP32: 23 mantissa bits → 7 decimal digits</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Dynamic range (more exponent bits)</div>
    <ul>
      <li>Larger ratio of max to min representable value</li>
      <li>Handles large weight disparities, loss spikes</li>
      <li>Needed for stable training of large models</li>
      <li>More exponent bits → wider range but same total bits</li>
      <li>BF16: 8 exponent bits → FP32-equivalent range</li>
    </ul>
  </div>
</div>
</div>

This tension is the root of why **BF16 (Brain Float 16)** was invented. Drawn to scale, the three formats split their bits like this:

<div class="bitfield"><div class="bf-seg sign" style="flex:1">S<small>1</small></div><div class="bf-seg exponent" style="flex:8">exponent<small>8 bits</small></div><div class="bf-seg mantissa" style="flex:23">mantissa<small>23 bits</small></div></div>
<div class="bitfield-label">FP32 · bias 127 · ~7 decimal digits</div>
<div class="bitfield"><div class="bf-seg sign" style="flex:1">S<small>1</small></div><div class="bf-seg exponent" style="flex:5">exponent<small>5 bits</small></div><div class="bf-seg mantissa" style="flex:10">mantissa<small>10 bits</small></div></div>
<div class="bitfield-label">FP16 · bias 15 · max 65,504</div>
<div class="bitfield"><div class="bf-seg sign" style="flex:1">S<small>1</small></div><div class="bf-seg exponent" style="flex:8">exponent<small>8 bits</small></div><div class="bf-seg mantissa" style="flex:7">mantissa<small>7 bits</small></div></div>
<div class="bitfield-label">BF16 · bias 127 · FP32's range with 8 bits of significand</div>


| Format | Sign | Exponent | Mantissa | Total | Range | Precision |
|--------|:----:|:--------:|:--------:|:-----:|-------|-----------|
| FP32 | 1 | 8 | 23 | 32 | ≈ 10⁻³⁸ to 10³⁸ | ~7 decimal digits |
| FP16 | 1 | 5 | 10 | 16 | ≈ 6 × 10⁻⁸ to 6.5 × 10⁴ | ~3 decimal digits |
| BF16 | 1 | 8 | 7 | 16 | same as FP32 | ~2 decimal digits |

FP16 has more precision (10 mantissa bits vs BF16's 7) but far less range (5 vs 8 exponent bits). In training, **range matters more** than precision: gradients span many orders of magnitude, and FP16's narrow range causes overflow (producing +inf) or requires aggressive loss scaling. BF16 retains FP32's exponent, avoiding overflow with simpler tooling. Chapter 23 extends this to FP8 (two variants: E4M3 and E5M2), FP4, INT8, INT4, and microscaling formats.

---

## Demonstrating Catastrophic Cancellation

```python
import numpy as np

# Catastrophic cancellation: subtracting two nearly-equal numbers
a = np.float32(1000000.1)
b = np.float32(1000000.0)
result_f32 = a - b
print(f"FP32: {a} - {b} = {result_f32}")   # 0.125 — should be 0.1!

# In FP64, there's enough precision
a64 = np.float64(1000000.1)
b64 = np.float64(1000000.0)
result_f64 = a64 - b64
print(f"FP64: {a64} - {b64} = {result_f64:.10f}")  # 0.0999999...

# Accumulation error: add 0.1 one million times, four ways
data = np.full(1_000_000, 0.1, dtype=np.float32)

def kahan_sum_f32(values):
    """Compensated summation with every operation in FP32."""
    total, comp = np.float32(0.0), np.float32(0.0)
    for v in values:
        y = v - comp
        t = total + y
        comp = (t - total) - y      # the low-order bits that were just lost
        total = t
    return total

print(f"{data.sum(dtype=np.float64):.4f}")                 # 100000.0015  exact sum of the FP32 inputs
print(f"{np.cumsum(data, dtype=np.float32)[-1]:.4f}")      # 100958.3438  one running FP32 accumulator
print(f"{data.sum(dtype=np.float32):.4f}")                 # 100000.0078  NumPy's pairwise summation
print(f"{kahan_sum_f32(data):.4f}")                        # 100000.0000  Kahan, still all FP32
```

(Outputs from NumPy 2.5 on x86-64.) A single running FP32 accumulator is off by almost 1 %: once the total passes 65,536 the spacing between FP32 values is 1/128, so each added 0.1 is rounded by a relatively large amount, always in the same direction. Summing in a tree (pairwise) or carrying the lost bits (Kahan) fixes it without a wider type. GPU reductions and GEMM kernels use the tree idea for the same reason.

This is why PyTorch's `torch.sum` and `torch.nn.LayerNorm` use FP32 accumulation internally even when the input tensor is FP16 or BF16 — the accumulator precision matters.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>A training run accumulates gradients over 64 micro-batches in BF16 before the optimizer step. Using the accumulation example above, what would you expect to happen to small gradients late in the accumulation, and which of the three fixes (wider accumulator, pairwise order, compensated sum) would a framework most likely choose?</p>
</div>

Try any value in any format below. Click bits to flip them and watch the decoded value change; `0.1` in BF16 versus FP16 shows the precision trade in one glance.

<div class="lab" data-lab="bits"><noscript><div class="lab-noscript">The bit explorer needs JavaScript.</div></noscript></div>

---

## Two's Complement and Floating Point Together

```python
import struct, numpy as np

# Inspect integer and float side by side
def show_int8(n: int):
    u = n & 0xFF
    print(f"INT8 {n:5d}: 0b{u:08b}  (0x{u:02X})")

def show_float32(f: float):
    bits = struct.unpack('>I', struct.pack('>f', f))[0]
    print(f"FP32 {f:10.4f}: 0b{bits:032b}  (0x{bits:08X})")

print("=== INT8 examples ===")
for v in [0, 1, 127, -1, -128]:
    show_int8(v)

print("\n=== FP32 examples ===")
for v in [0.0, 1.0, -1.0, 0.1, 1e38, float('inf')]:
    show_float32(v)
```

---

## Why This Matters for Model Optimization

<div class="diagram">
<div class="diagram-title">Number Representation → ML Decisions</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Bit Width = Memory Budget</div>
    <div class="card-desc">FP32 weights cost 4× more bytes than INT8. Halving the bits halves the bytes moved, so a memory-bound kernel can run up to 2× faster: the primary driver of quantization speedups (Ch 25).</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Range vs Precision Trade-off</div>
    <div class="card-desc">BF16 keeps FP32's exponent → stable training without loss scaling. FP16's narrow range → requires loss scaling + overflow detection. FP8-E4M3 favours precision; FP8-E5M2 favours range (Ch 23).</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Accumulation Precision</div>
    <div class="card-desc">Even INT8 kernels accumulate in INT32 to prevent overflow. torch.sum in FP16 secretly uses FP32 accumulators. Choosing a too-narrow accumulator format directly causes accuracy loss in quantized models.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Two's Complement Symmetry</div>
    <div class="card-desc">INT8 has 256 levels, −128 to 127. Symmetric quantizers usually restrict themselves to [−127, 127] so that ±max map to the same magnitude and negating a value can never overflow; zero is exact either way.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Subnormals = Lost Range</div>
    <div class="card-desc">FP16 gradients below 6.1 × 10⁻⁵ go subnormal and, under flush-to-zero, vanish. GPUs pay no speed penalty for subnormals, CPUs often do. Loss scaling (or BF16's wider exponent) keeps gradients in the normal range.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Fixed-Point = Quantization</div>
    <div class="card-desc">INT8 quantization is fixed-point with a per-tensor or per-channel scale. The hardware does integer MACs; scale factors are applied in software or by dedicated dequantization units (Ch 25).</div>
  </div>
</div>
</div>

---

## Key Takeaways

- **Unsigned binary** maps bit patterns to non-negative integers via $V = \sum b_i \cdot 2^i$; N bits span $[0, 2^N - 1]$.
- **Two's complement** is the universal signed integer format: one zero, addition uses the same hardware, $-x = \sim\! x + 1$; N bits span $[-2^{N-1}, 2^{N-1}-1]$.
- **Fixed-point** (Q-format) is an integer with an implicit binary point; quantization is fixed-point arithmetic with a learned scale factor.
- **IEEE-754 floating point** encodes $(-1)^S \times 1.M \times 2^{E-\text{bias}}$; the bias allows unsigned comparison; the hidden bit gives one free bit of mantissa.
- **FP32** has 8 exponent bits and 23 mantissa bits; **BF16** trades mantissa for FP32-equivalent range; **FP16** trades range for more mantissa — range matters more for training.
- **Subnormals** provide gradual underflow (slow on many CPUs, full speed on NVIDIA GPUs); **special values** (±∞, NaN) propagate through arithmetic.
- **How you accumulate** matters as much as the format: a single running FP32 sum of 10⁶ values can be off by ~1 %, while pairwise or Kahan summation in the same format is exact to ~10⁻⁷.
- The **precision vs dynamic range** trade-off governs every ML dtype decision: Chapter 23 extends these foundations to FP8, FP4, INT4, and microscaling formats.

## Check Yourself

<details class="check">
<summary>What is the INT8 bit pattern of −5, and how do you get it?</summary>
<p>Take +5 = 0000 0101, invert every bit to 1111 1010, add 1: <code>1111 1011</code> (0xFB). Adding it to +5 in an ordinary adder gives 1 0000 0000; the carry out is discarded and the result is 0.</p>
</details>

<details class="check">
<summary>Why does BF16 rarely need loss scaling while FP16 does?</summary>
<p>BF16 keeps FP32's 8-bit exponent, so it covers the same ~10⁻³⁸ to 10³⁸ range and small gradients stay representable. FP16 has 5 exponent bits: normal values stop at 6.1 × 10⁻⁵ and the maximum is 65,504, so gradients underflow (or activations overflow) unless the loss is scaled up before the backward pass.</p>
</details>

<details class="check">
<summary>In FP32, why does 1e8 + 1 equal 1e8 while 1e7 + 1 does not?</summary>
<p>FP32 has a 24-bit significand, so between 2²⁶ and 2²⁷ (which contains 1e8) adjacent values are 2³ = 8 apart; adding 1 is less than half a spacing and rounds back to 1e8. Between 2²³ and 2²⁴ (which contains 1e7) the spacing is exactly 1, so 1e7 + 1 is representable.</p>
</details>

<details class="check">
<summary>A kernel sums a million BF16 values with one BF16 accumulator. What goes wrong and what is the usual fix?</summary>
<p>Once the running total is large, each new small value is smaller than half the spacing of BF16 (only 8 significant bits) and is rounded away or heavily rounded, so the total stalls. Kernels accumulate in FP32 and reduce in a tree (pairwise); both are standard in cuBLAS and PyTorch reductions.</p>
</details>

<details class="check">
<summary>Why is round-to-nearest-even preferred over round-half-up for long computations?</summary>
<p>Round-half-up always resolves ties in the same direction, adding a small positive bias that accumulates over many operations. Ties-to-even sends half of the ties up and half down, so the expected rounding error is zero.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| IEEE Standard for Floating-Point Arithmetic (IEEE 754-2019) | IEEE, 2019 | [doi.org/10.1109/IEEESTD.2019.8766229](https://doi.org/10.1109/IEEESTD.2019.8766229) | Formats, special values, the five rounding modes |
| What Every Computer Scientist Should Know About Floating-Point Arithmetic | D. Goldberg, ACM Computing Surveys 1991 | [doi.org/10.1145/103162.103163](https://doi.org/10.1145/103162.103163) | ULPs, cancellation, rounding error analysis |
| Further Remarks on Reducing Truncation Errors | W. Kahan, CACM 1965 | [doi.org/10.1145/363707.363723](https://doi.org/10.1145/363707.363723) | Compensated (Kahan) summation |
| Accuracy and Stability of Numerical Algorithms, 2nd ed. | N. Higham, SIAM 2002 | [doi.org/10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027) | Error bounds for recursive vs pairwise summation |
| A Study of BFLOAT16 for Deep Learning Training | D. Kalamkar et al., 2019 | [arxiv.org/abs/1905.12322](https://arxiv.org/abs/1905.12322) | Why an FP32-range 16-bit format trains without loss scaling |
| CUDA C++ Programming Guide: floating-point standard and intrinsics | NVIDIA | [docs.nvidia.com/cuda/cuda-c-programming-guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | Denormal handling and flush-to-zero on GPUs |

---

**Next:** [Chapter 4 — Anatomy of a Processor](./04_anatomy_of_a_processor.md) — We wire these number representations and adders together into a full CPU — fetch, decode, execute, and the pipeline.

[← Back to Table of Contents](./README.md)
