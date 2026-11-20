---
title: "Chapter 13 — Reliability, Fault Tolerance & Robustness"
---

[← Back to Table of Contents](./README.md)

# Chapter 13 — Reliability, Fault Tolerance & Robustness

A modern GPU training cluster contains tens of thousands of chips, each containing billions of transistors, running at temperatures above 80°C for months at a time. The universe is constantly trying to corrupt that computation — cosmic rays from space, alpha particles from packaging materials, electromigration from constant current flow, thermal stress from repeated heating and cooling. In a cluster of 10,000 H100s, you should expect hardware faults not as exceptional events but as a routine operational reality.

For ML practitioners, the implications are concrete: a training run that takes 30 days on 1,000 GPUs can be destroyed by a single bit flip in the wrong place at the wrong time — unless the system is designed to detect, correct, and recover from faults. This chapter explains the physics of hardware failure, the engineering mechanisms for tolerating it, and the specific decisions that matter when you're designing or running large-scale training.

> **The one-sentence version:** Hardware faults are inevitable at scale; the difference between a successful 30-day training run and a corrupted waste of compute is whether your stack has the right combination of ECC memory, frequent checkpointing, anomaly detection, and redundancy — and understanding why these work requires knowing how bits actually get flipped.

---

## Why Chips Fail — A Taxonomy of Faults

<div class="diagram">
<div class="diagram-title">Fault Classification</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Transient (Soft) Errors</div>
    <ul>
      <li>Single event: occurs once, does not recur at same location</li>
      <li>Cause: cosmic rays, alpha particles, thermal noise</li>
      <li>Effect: bit flip in memory or flip-flop</li>
      <li>Hardware unchanged after the event</li>
      <li>Detectable/correctable with ECC</li>
      <li>Rate: lab soft-error rates for modern DRAM are ~1 FIT per megabit or less; SRAM is higher per bit</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Permanent (Hard) Errors</div>
    <ul>
      <li>Wearout: location is permanently degraded</li>
      <li>Cause: electromigration, oxide breakdown, NBTI, HCI</li>
      <li>Effect: stuck bit, open circuit, short</li>
      <li>Hardware is damaged; fault recurs</li>
      <li>Requires redundancy, remapping, or replacement</li>
      <li>Rate: increases exponentially with temperature</li>
    </ul>
  </div>
</div>
</div>

### Soft Errors: The Physics of Bit Flips

**Single Event Upsets (SEUs)** — colloquially "bit flips" — occur when a charged particle deposits energy in a CMOS transistor, momentarily creating electron-hole pairs that flip the stored state of a memory cell or flip-flop.

**Cosmic rays** are high-energy particles (mostly protons) from supernovae and other astrophysical sources. They collide with the upper atmosphere and shower the ground with secondary neutrons; at sea level the reference flux of neutrons energetic enough to upset a cell is about **13 per cm² per hour** (JEDEC JESD89, New York City), several times higher at airliner altitude, but data centers are not immune — at altitude (Denver, Mexico City) the flux can be 3–10× higher. When a high-energy neutron collides with a silicon nucleus, it creates a spray of secondary particles that can deposit enough charge (~100 fC / 100 femtocoulombs) in a memory cell to flip it.

**Alpha particles** come from trace radioactive contamination in chip packaging materials — even "clean" packaging has parts-per-billion-level uranium and thorium isotopes that emit alpha particles with enough range (~25 μm in silicon) to reach nearby DRAM cells.

**FIT (Failure in Time):** the standard unit for error rates. 1 FIT = 1 failure per 10⁹ device-hours (equivalently, a mean time to failure of ~114,000 years for one device). A 32 GB DIMM holds 2.56 × 10¹¹ bits = 256,000 Mbit; at a lab soft-error rate of ~1 FIT/Mbit:

$$\text{FIT}_{\text{DIMM}} = \frac{32 \times 10^9 \times 8}{10^6} \times 1 \approx 256{,}000 \text{ FIT}$$

$$\text{Error rate} = \frac{256{,}000}{10^9} \text{ per hour} = 2.56 \times 10^{-4} \text{ per hour} \approx 1 \text{ error per } 3{,}900 \text{ hours}$$

So an unprotected 32 GB DIMM would see a soft error every ~5 months, and a server with 2 TB of RAM (64 such DIMMs) one every ~2.5 days. Field studies tell a harsher story than the lab: Google's measurements across its fleet (Schroeder et al., 2009) found about a third of machines and 8 % of DIMMs seeing correctable errors each year, at effective rates of 25,000–75,000 FIT/Mbit, dominated by *hard* and intermittent faults (bad cells, failing chips) rather than particle strikes. Either way, ECC is not optional in any serious compute environment.

### Hard Errors: Wearout Mechanisms

**Electromigration:** when large currents flow through thin metal wires (which, at 3 nm process nodes, are only a few atoms wide), momentum transfer from electrons gradually displaces metal atoms. Over years of operation, this can create voids (open circuits) or hillocks (short circuits). Accelerated dramatically by high temperature — the rate roughly doubles for every 10°C increase (Arrhenius relationship).

**Negative Bias Temperature Instability (NBTI):** a wearout mechanism in PMOS transistors under sustained negative gate voltage. Gradually shifts threshold voltage, slowing transistors and eventually causing timing failures. Worst at high temperature and high activity.

**Hot Carrier Injection (HCI):** energetic ("hot") carriers trapped in gate oxide, shifting threshold voltage over time. More severe in thin-oxide technologies.

All three mechanisms accelerate with temperature, which is why **keeping chips cool is a reliability requirement, not just a performance optimization.**

---

## ECC Memory: How Error Correction Works

### The Core Idea: Redundant Bits

**ECC (Error-Correcting Code) memory** adds extra bits to every stored word, computed from the data bits using a carefully designed parity code. On read, the stored data bits are used to recompute the expected parity, and discrepancies pinpoint (and, for SECDED, correct) errors.

### Hamming Codes: SECDED

**SECDED (Single Error Correct, Double Error Detect)** is the dominant ECC scheme for DRAM. It uses Hamming codes extended with an overall parity bit.

For a $k$-bit data word, you need $r$ parity bits where $2^r \geq k + r + 1$. For $k=8$ (one byte): $r=4$ (check bits) → 12-bit codeword. For $k=64$ bits (standard DRAM word): $r=7$ check bits → 71 bits, rounded to 72 bits (adding 1 overall parity bit for double-error detection) → **64-bit data + 8 check bits = 72 bits**.

This is why ECC DDR4 DIMMs have **72-bit wide buses** instead of 64-bit. DDR5 ECC RDIMMs split each module into two 40-bit sub-channels (32 data + 8 check bits each), 80 bits in total, which also lets them correct more than one bad bit per chip.

**How parity bits are positioned (Hamming construction):**

Place parity bits at positions that are powers of 2 (1, 2, 4, 8, …). Each parity bit covers all positions whose index has that power-of-2 bit set. On a single-bit error, the parity checks that fail form a binary syndrome pointing directly to the corrupted bit position.

```text
Bit positions (1-indexed):
Pos:  1  2  3  4  5  6  7  8  9 10 11 12
Type: P1 P2 D1 P4 D2 D3 D4 P8 D5 D6 D7 D8

P1 covers: positions 1,3,5,7,9,11  (bit 0 of index = 1)
P2 covers: positions 2,3,6,7,10,11 (bit 1 of index = 1)
P4 covers: positions 4,5,6,7,12    (bit 2 of index = 1)
P8 covers: positions 8,9,10,11,12  (bit 3 of index = 1)

If bit at position 7 (= 0111 binary) is flipped:
  P1 fails (position 7 has bit 0 set) → contribute 1 to syndrome
  P2 fails (position 7 has bit 1 set) → contribute 2 to syndrome
  P4 fails (position 7 has bit 2 set) → contribute 4 to syndrome
  P8 passes (position 7 has bit 3 clear)
  Syndrome = 1+2+4 = 7 → flip bit 7 to correct
```

**SECDED adds one more overall parity bit** (XOR of all bits including parity). If exactly one error occurred: the syndrome is nonzero AND the overall parity disagrees → single-bit error → correctable. If two errors: syndrome is nonzero BUT overall parity agrees → double error detected (not correctable — signal UECC/uncorrectable error).

### ECC Variants in Modern Systems

| ECC Type | Description | Where Used |
|:--------:|:-----------:|:----------:|
| SECDED (72-bit) | Corrects 1-bit, detects 2-bit errors per 64-bit word | DDR4/5 ECC DIMMs, CPU servers |
| Chipkill / x4 SDDC | Full DRAM chip failure correction; corrects all bits from one ×4 DRAM | High-end servers (IBM, HPE) |
| On-die ECC | ECC inside the DRAM die itself, invisible to the memory controller; mops up weak cells before data leaves the chip | All DDR5 (mandatory), LPDDR5, HBM3 |
| HBM ECC | Check bits carried beside the data on each HBM channel, plus on-die ECC inside HBM3/HBM3e stacks | A100, H100, B200, MI300X and later |
| SRAM ECC | Dedicated ECC for on-chip caches and register files | L2/L3 caches, GPU shared memory |

**HBM ECC** is particularly important for GPUs. NVIDIA datacenter GPUs since Volta keep ECC on by default for HBM, L2, L1/shared memory and the register file, with no capacity cost for HBM (the check bits travel in dedicated sideband lanes), and newer generations combine it with on-die ECC in the HBM stack and row remapping for cells that keep failing.

---

## RAS: Reliability, Availability, Serviceability

**RAS** is the umbrella term for hardware features designed to keep systems running correctly:

<div class="diagram">
<div class="diagram-title">RAS Features</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🛡️</div>
    <div class="card-title">Reliability</div>
    <div class="card-desc">ECC memory, lockstep cores, parity on buses and caches, redundant power supplies, end-to-end data protection. Prevent incorrect computation.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">⏱️</div>
    <div class="card-title">Availability</div>
    <div class="card-desc">Hot-swap components, RAID, spare rows in DRAM (post-package repair), live migration, graceful degradation. Keep the system running despite faults.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🔧</div>
    <div class="card-title">Serviceability</div>
    <div class="card-desc">Detailed error logging (MCELOG, MCA), remote management (BMC/IPMI), in-field diagnostics, BIST (built-in self-test). Identify and fix faults quickly.</div>
  </div>
</div>
</div>

Modern NVIDIA GPUs expose RAS counters via `nvidia-smi` and NVML:

```bash
nvidia-smi --query-gpu=ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv
# corrected = single-bit errors, corrected transparently
# uncorrected = double-bit errors (or uncorrectable) — system should page-retire
```

A GPU with rapidly rising corrected ECC errors is typically an early warning sign of a memory cell degrading toward failure. NVIDIA's driver pages out memory rows with repeated CECC errors (post-package repair / DRAM row remapping) and retires pages with UECC errors, removing them from the allocatable pool.

---

## Silent Data Corruption: The Insidious Threat

**Silent Data Corruption (SDC)** — also called **silent errors** — is the most dangerous failure mode: the hardware produces wrong answers without any error signal. No ECC alarm, no crash, no NaN — just numerically wrong output.

SDC has become a recognized operational problem at hyperscale. Google and Meta have both published research on finding "mercurial" compute cores: CPUs or GPUs that silently compute wrong results on a tiny fraction of operations. The rates are tiny — perhaps 1 error in $10^{14}$ operations on an affected unit — but at scale:

$$10^{14} \text{ ops} \div (10^{15} \text{ FLOP/s}) = 0.1 \text{ seconds of GPU time}$$

A 10,000-GPU cluster performs on the order of $10^{23}$ FLOPs per day, so even a handful of mercurial units among them can inject corrupted values many times a day, and nothing in the hardware will say so.

**Mechanisms for SDC:**

1. **Timing margin violations**: manufacturing variation means some transistors are slightly slow. At nominal voltage and clock, most are fine. But certain input patterns create critical timing paths that fail only on specific data values — the gate switches, but just barely, and the output is captured in the wrong state.

2. **Voltage droop**: sudden large power draws (e.g., activating all tensor cores simultaneously) cause momentary voltage drops. If the droop is severe enough, fast logic fails transiently.

3. **Crosstalk**: closely packed wires at advanced process nodes can capacitively couple, causing a neighboring wire's transition to induce a glitch.

4. **Aging effects** (NBTI, HCI) that push timing margins to failure without triggering hard errors.

**Hyperscaler response:** Google's 2021 paper ("Cores that don't count") described a systematic screening process: run compute-bound workloads that stress specific instruction patterns, compare outputs across multiple units, flag units that disagree. Meta (Dixit et al., 2021) described finding ~1 in 2,000 servers with mercurial DRAM that silently corrupts specific address patterns. The practical recommendation: **burn-in and validation testing** before deploying new hardware into training clusters, and **gradient/loss spike detection** during runs.

---

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>ECC protects bits at rest and on the wire, but a mercurial multiplier corrupts results <em>while computing</em>, where no code covers them. If you had to protect a matrix multiply against that, would you recompute, checksum (A·B·1 = A·(B·1) is cheap to verify), or vote across replicas, and what does each cost at 10,000 GPUs?</p>
</div>

## Redundancy Techniques

### Triple Modular Redundancy (TMR) / Voting

The most conceptually direct fault tolerance: replicate a computation three times, take a majority vote. One module can fail arbitrarily (flip a bit, crash, return garbage) and the voted output remains correct. Used in safety-critical systems (avionics, spacecraft, automotive ASIL-D).

$$\text{TMR output} = \text{majority}(A, B, C)$$

**Cost: 3× hardware, 3× power.** Not used in general ML accelerators, but lockstep execution (two identical pipelines in parallel, output compared) is used in automotive chips (NVIDIA DRIVE Orin with lockstep Cortex-R52s for ASIL-D).

### Spare Rows and Columns in DRAM

DRAM arrays are manufactured with extra rows and columns beyond the nominal capacity. Post-fabrication testing identifies defective rows (stuck-at faults from manufacturing variations) and blows laser fuses to remap the defective row address to a spare. This is how modern DRAM achieves high yield even at high transistor counts — the chip contains e.g. 104 rows where the spec requires 100, and up to 4 defective rows can be remapped. The same mechanism applies post-field-deployment: **post-package repair (PPR)** allows the memory controller to request a row remap for a row exhibiting excessive soft errors, without replacing the DIMM.

### Redundant Compute Cores for Yield (GPU)

GPU dies ship with more shader multiprocessors (SMs) than the product specification requires. An RTX 4090 enables 128 of the 144 SMs on NVIDIA's AD102 die, and an H100 SXM enables 132 of GH100's 144. If up to 16 SMs are defective, the die still passes as a fully functional product — defective SMs are disabled via fusing. Dies with 128–143 functional SMs sell as 4090; dies with 100–127 functional SMs sell as lower-tier 4080/4070 variants. This **binning** dramatically improves yield and economics. See [Appendix A — How Chips Are Made](./appendix_a_how_chips_are_made.md) for the full yield/binning discussion.

---

## Thermal Throttling and Reliability

Semiconductor reliability is a strong function of temperature. The **Arrhenius model** governs most wearout mechanisms:

$$\text{Lifetime} \propto e^{E_a / (k_B T)}$$

where $E_a$ is the activation energy (typically 0.5–1.0 eV for electromigration), $k_B$ is Boltzmann's constant, and $T$ is absolute temperature. Every 10°C rise roughly halves the mean time to failure for most wearout mechanisms.

Modern GPUs implement **thermal throttling** (also called **dynamic thermal management, DTM**): when the junction temperature exceeds a threshold (typically in the 80s °C for NVIDIA GPUs; datacenter parts report a slowdown temperature through NVML), the driver reduces the GPU clock and/or power limit to bring temperature down. This is automatic and transparent — but it silently reduces compute throughput.

```text
H100 SXM5 temperature-performance profile (approximate):
  < 80°C : full boost clock (~1.98 GHz), full TDP (700 W)
  80–87°C: clock starts stepping back, approaching TDP throttle
  > 87°C : thermal throttle, frequency reduced to maintain < 83°C junction
  > 90°C : aggressive throttle; may trigger hardware protection
  > 95°C : emergency power cutoff (GPU power rails cut to protect die)
```

**For training runs:** a thermal throttle reduces throughput without causing a crash. You may not notice it unless you monitor `nvidia-smi dmon` or `nvml`. On air-cooled servers with poor airflow, GPU frequency can be 10–20% below rated spec throughout a run. The fix is proper cooling infrastructure — direct liquid cooling (DLC) is increasingly standard for 700 W GPUs.

---

## Checkpointing as Fault Tolerance

No hardware mechanism alone can protect a 30-day training run against all failure modes: uncorrectable ECC errors, kernel panics, NaN poisoning from silent corruption, network failures, or simple power outages. **Checkpointing** is the software-level solution: periodically save the full training state to persistent storage, and resume from the last checkpoint on failure.

**What to checkpoint:**
- Model weights (`model.state_dict()`)
- Optimizer state (Adam's `m` and `v`: two FP32 values per parameter, 4× the size of BF16 weights) and the FP32 master weights
- LR scheduler state
- RNG states (for reproducibility of dropout)
- Step number, epoch, dataloader position

**Checkpointing cost:** for a 70B model, the FP32 master weights (280 GB) plus Adam's m and v (560 GB) come to ~840 GB per checkpoint. Written by one machine at ~7 GB/s that would take two minutes; in practice every data-parallel rank writes its own shard in parallel, and asynchronous checkpointing copies state to host memory in seconds and drains it to storage in the background. At a typical checkpoint frequency of every 100–500 training steps (~5–25 minutes), the overhead is 0.5–3%. Distributed checkpointing (each GPU writes its own shard in parallel) reduces this proportionally with GPU count.

**Mean time between failures (MTBF) considerations:** at 1,000 GPUs, if each GPU has an MTBF of 20,000 hours (~2.3 years), the cluster's MTBF is:

$$\text{MTBF}_{\text{cluster}} = \frac{20,000}{1,000} = 20 \text{ hours}$$

A 30-day run should expect approximately $30 \times 24 / 20 = 36$ individual GPU failures. Without checkpointing, each failure restarts from scratch. With hourly checkpoints, each failure costs at most 1 hour of compute. **This is not optional at scale.**

Real numbers are worse than this textbook estimate. Meta's Llama 3 405B pre-training run on 16,384 H100s saw **466 job interruptions in 54 days**, 419 of them unexpected (about one every three hours). Roughly 78 % were hardware: faulty GPUs (30 % of unexpected interruptions) and HBM3 failures (17 %) led the list, followed by software bugs, network links and host machines. Despite that, automated detection and restart kept effective training time above 90 %.

---

## Python: Hamming SECDED Encode/Decode + FIT Rate Estimation

```python
import numpy as np
from typing import Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Hamming SECDED on an 8-bit (1-byte) data word
# We use an [12,8] SECDED code: 8 data bits + 3 Hamming parity bits + 1 overall parity
# Extended to a 12-bit codeword for clarity.
# ─────────────────────────────────────────────────────────────────────────────

def hamming_encode_8bit(data: int) -> int:
    """
    Encode an 8-bit data word into a 12-bit SECDED Hamming codeword.
    
    Bit positions 1-indexed. Parity bits at positions 1,2,4,8 (powers of 2).
    Data bits d1..d8 placed at positions 3,5,6,7,9,10,11,12.
    Position 0 (msb of our 12-bit word, pos index 12) is overall parity.
    
    Returns: 12-bit integer (bit 11 is pos 12, bit 0 is pos 1)
    """
    assert 0 <= data < 256, "Data must be 8-bit (0-255)"
    
    # Extract data bits (d1..d8 from LSB to MSB of 'data')
    d = [(data >> i) & 1 for i in range(8)]  # d[0]..d[7]
    
    # Place data bits in codeword (0-indexed array, position = index+1)
    # Positions 1,2,4,8 reserved for parity; data goes to 3,5,6,7,9,10,11,12
    c = [0] * 12  # c[0] = position 1, c[11] = position 12
    data_positions = [2, 4, 5, 6, 8, 9, 10, 11]  # 0-indexed = positions 3,5,6,7,9,10,11,12
    for i, pos in enumerate(data_positions):
        c[pos] = d[i]
    
    # Compute Hamming parity bits (XOR of all positions covered)
    # p1 (pos 1 = c[0]): covers positions 1,3,5,7,9,11 → indices 0,2,4,6,8,10
    c[0] = c[0] ^ c[2] ^ c[4] ^ c[6] ^ c[8] ^ c[10]
    # p2 (pos 2 = c[1]): covers positions 2,3,6,7,10,11 → indices 1,2,5,6,9,10
    c[1] = c[1] ^ c[2] ^ c[5] ^ c[6] ^ c[9] ^ c[10]
    # p4 (pos 4 = c[3]): covers positions 4,5,6,7,12 → indices 3,4,5,6,11
    c[3] = c[3] ^ c[4] ^ c[5] ^ c[6] ^ c[11]
    # p8 (pos 8 = c[7]): covers positions 8,9,10,11,12 → indices 7,8,9,10,11
    c[7] = c[7] ^ c[8] ^ c[9] ^ c[10] ^ c[11]
    
    # Overall parity (SECDED extension): XOR of all bits → stored at position 0
    # We'll prepend it as bit 12 of the codeword (index 11 already used; use a 13th slot)
    # For simplicity, encode as a 13-bit word: bit 12 = overall parity
    overall_parity = 0
    for bit in c:
        overall_parity ^= bit
    
    # Pack into integer: bits [12..0], bit 12 = overall parity
    codeword = 0
    for i, bit in enumerate(c):
        codeword |= (bit << i)
    codeword |= (overall_parity << 12)
    
    return codeword  # 13-bit SECDED codeword

def hamming_decode_8bit(codeword: int) -> Tuple[int, str]:
    """
    Decode a 13-bit SECDED Hamming codeword.
    Returns (corrected_data, status) where status is 'ok', 'corrected', or 'UECC'.
    """
    # Unpack
    overall_parity_stored = (codeword >> 12) & 1
    c = [(codeword >> i) & 1 for i in range(12)]
    
    # Recompute syndrome
    s1 = c[0] ^ c[2] ^ c[4] ^ c[6] ^ c[8] ^ c[10]
    s2 = c[1] ^ c[2] ^ c[5] ^ c[6] ^ c[9] ^ c[10]
    s4 = c[3] ^ c[4] ^ c[5] ^ c[6] ^ c[11]
    s8 = c[7] ^ c[8] ^ c[9] ^ c[10] ^ c[11]
    syndrome = s1 | (s2 << 1) | (s4 << 2) | (s8 << 3)
    
    # Recompute overall parity
    overall_parity_computed = 0
    for bit in c:
        overall_parity_computed ^= bit
    overall_parity_computed ^= overall_parity_stored  # include stored parity bit
    
    status = "ok"
    if syndrome == 0 and overall_parity_computed == 0:
        status = "ok"
    elif syndrome != 0 and overall_parity_computed == 1:
        # Single-bit error: syndrome points to error position (1-indexed)
        error_pos = syndrome  # 1-indexed bit position in the 12-bit Hamming word
        if 1 <= error_pos <= 12:
            c[error_pos - 1] ^= 1  # correct it
        status = "corrected"
    else:
        # Double (or higher) error detected: overall parity agrees but syndrome nonzero
        status = "UECC"
    
    # Extract data bits from corrected codeword
    data_positions = [2, 4, 5, 6, 8, 9, 10, 11]
    data = 0
    for i, pos in enumerate(data_positions):
        data |= (c[pos] << i)
    
    return data, status

# --- Demonstrate encode/decode/correction ---
original = 0b10110101  # 181
cw = hamming_encode_8bit(original)
print(f"Original data:  {original:08b} = {original}")
print(f"SECDED codeword: {cw:013b} (13 bits)")

# No error
decoded, status = hamming_decode_8bit(cw)
print(f"\nNo error:   decoded={decoded:08b}={decoded}, status={status}")

# Inject single-bit error (flip bit 5 of codeword)
cw_1bit = cw ^ (1 << 5)
decoded, status = hamming_decode_8bit(cw_1bit)
print(f"1-bit error: decoded={decoded:08b}={decoded}, status={status}")
assert decoded == original, "Single-bit correction failed!"

# Inject double-bit error (flip bits 5 and 7)
cw_2bit = cw ^ (1 << 5) ^ (1 << 7)
decoded, status = hamming_decode_8bit(cw_2bit)
print(f"2-bit error: decoded={decoded:08b}={decoded}, status={status}")
# Should be UECC — double error detected but not corrected


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Expected bit flips over a large training run given a FIT rate
# ─────────────────────────────────────────────────────────────────────────────

def estimate_bit_flips(
    num_gpus: int,
    hbm_per_gpu_gb: float,
    run_duration_hours: float,
    fit_per_mbit: float = 1.0,      # typical unprotected HBM FIT rate
    ecc_coverage: bool = True,
    secded_uecc_ratio: float = 1e-6, # fraction of errors that are double-bit (uncorrectable)
) -> dict:
    """
    Estimate expected correctable and uncorrectable ECC events during a training run.
    
    FIT rate model: errors arrive as a Poisson process at the given FIT/Mbit rate.
    """
    # Total memory in megabits
    total_bits = num_gpus * hbm_per_gpu_gb * 1e9 * 8  # bits
    total_mbits = total_bits / 1e6
    
    # FIT = failures per 1e9 device-hours
    # Expected errors = total_mbits * fit_per_mbit * hours / 1e9
    total_fit = total_mbits * fit_per_mbit
    expected_errors = total_fit * run_duration_hours / 1e9
    
    if ecc_coverage:
        expected_cecc = expected_errors * (1 - secded_uecc_ratio)
        expected_uecc = expected_errors * secded_uecc_ratio
    else:
        expected_cecc = 0.0
        expected_uecc = expected_errors  # all errors are uncorrected
    
    # Probability of zero uncorrectable errors (Poisson)
    p_zero_uecc = float(np.exp(-expected_uecc))
    
    return {
        "num_gpus": num_gpus,
        "hbm_tb": num_gpus * hbm_per_gpu_gb / 1e3,
        "total_mbits": total_mbits / 1e6,   # Tbits for display
        "run_hours": run_duration_hours,
        "expected_total_errors": expected_errors,
        "expected_cecc": expected_cecc,
        "expected_uecc": expected_uecc,
        "p_zero_uecc_percent": p_zero_uecc * 100,
        "ecc_coverage": ecc_coverage,
    }

def print_fit_estimate(r: dict, label: str = ""):
    print(f"\n{'─'*60}")
    if label: print(f"  {label}")
    print(f"  GPUs:             {r['num_gpus']} × {r['hbm_tb']*1e3/r['num_gpus']:.0f} GB HBM")
    print(f"  Total HBM:        {r['hbm_tb']:.1f} TB  ({r['total_mbits']:.1f} Tbits)")
    print(f"  Run duration:     {r['run_hours']:.0f} hours  ({r['run_hours']/24:.0f} days)")
    print(f"  ECC enabled:      {r['ecc_coverage']}")
    print(f"  Expected CECC:    {r['expected_cecc']:.1f} (corrected silently)")
    print(f"  Expected UECC:    {r['expected_uecc']:.4f} (uncorrectable → crash/corruption)")
    print(f"  P(zero UECC):     {r['p_zero_uecc_percent']:.2f}%")

# Scenario 1: Small run — 8 GPUs, 80 GB each, 1 week
r1 = estimate_bit_flips(num_gpus=8, hbm_per_gpu_gb=80, run_duration_hours=168)
print_fit_estimate(r1, "8× H100 (80 GB), 1 week, ECC enabled")

# Scenario 2: Large training — 1024 GPUs, 80 GB each, 30 days
r2 = estimate_bit_flips(num_gpus=1024, hbm_per_gpu_gb=80, run_duration_hours=720)
print_fit_estimate(r2, "1024× H100, 30-day run, ECC enabled")

# Scenario 3: Same without ECC (for comparison)
r3 = estimate_bit_flips(num_gpus=1024, hbm_per_gpu_gb=80,
                         run_duration_hours=720, ecc_coverage=False)
print_fit_estimate(r3, "1024× H100, 30-day run, NO ECC")
```

**Output** (Python 3.12, abridged):

```text
  8× H100 (80 GB), 1 week, ECC enabled
  Total HBM:    0.6 TB    Expected CECC: 0.9    P(zero UECC): 100.00%

  1024× H100, 30-day run, ECC enabled
  Total HBM:   81.9 TB    Expected CECC: 471.9  P(zero UECC): 99.95%
  Expected UECC: 0.0005  ← rare but real

  1024× H100, 30-day run, NO ECC
  Expected UECC: 471.9  ← catastrophic
  P(zero UECC): ~0%
```

At the lab rate of 1 FIT/Mbit: without ECC, ~470 silent corruptions over 30 days across 1,024 GPUs; with ECC, ~470 corrected transparently and ~0.0005 uncorrectable (one UECC event per ~2,000 such runs). Field error rates are higher and dominated by failing parts, which ECC also catches and reports. **ECC is non-negotiable for serious training.**

---

## Why This Matters for Model Optimization

<div class="diagram">
<div class="diagram-title">Reliability Failures in Long Training Runs</div>
<div class="flow">
  <div class="flow-node accent wide">Month-long run on 1,000+ GPUs</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node red wide">Guaranteed: multiple GPU failures, soft errors, thermal events</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Without mitigations: wasted compute, corrupted weights, lost run</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">With mitigations: automatic recovery, bounded compute loss</div>
</div>
</div>

**Checkpointing frequency:** The right checkpoint interval balances overhead vs exposure. For a 1,000-GPU cluster with MTBF ~20 hours/cluster, checkpoint every 0.5–1 hour to limit expected wasted compute per failure to 25–50 GPU-hours.

**Loss/gradient spike detection:** A single silent data corruption in the gradient accumulation buffer can inject a massive gradient spike, destabilizing training for many subsequent steps. Monitoring `loss` for sudden spikes (>3–5× normal) and having automatic rollback to the previous checkpoint is standard practice in large runs (used at OpenAI, Google, Anthropic).

**Quantization and bit-flip sensitivity:** Lower-precision formats have smaller dynamic ranges. An INT4 weight has only 16 levels; a single bit flip in the most significant bit changes the value by 8 quantization steps — potentially a large perturbation. FP8 exponent bits are similarly sensitive (flipping the MSB of the exponent changes value by 128×). This means quantized models may be *more sensitive* to residual soft errors that ECC misses (e.g., in L1/shared memory without ECC on older GPUs). When deploying quantized models in high-reliability settings, prefer hardware with full ECC coverage including L1/SRAM (H100 has this; some older GPUs do not).

**Detection beats hoping for robustness:** no model architecture reliably tolerates corrupted weights or gradients. Practical defenses are monitoring (loss and gradient-norm spikes, NaN/Inf checks), checksums on checkpoints, and periodically re-running a known batch on each GPU to catch mercurial hardware.

**Thermal management:** GPU frequency throttling under poor cooling is a silent performance bug. Monitoring junction temperature and ensuring adequate cooling (liquid cooling for 700 W+ GPUs) prevents both reduced throughput and accelerated wearout.

---

## Key Takeaways

- **Soft errors (SEUs)** from cosmic-ray neutrons and alpha particles flip bits at roughly 1 FIT/Mbit (1 per 10⁹ hours per megabit) in modern DRAM; field error rates including failing parts are far higher. At 82 TB of HBM across 1,024 GPUs, the lab rate alone gives ~470 bit flips per 30-day run, all corrected by ECC.
- **SECDED Hamming codes** add 12.5 % overhead (8 check bits per 64 data bits) to correct any single-bit error and detect any double-bit error; GPUs and DDR5 layer stronger codes and on-die ECC on top.
- **Hard errors** (electromigration, NBTI) cause permanent wearout accelerated by high temperature. Thermal throttling protects the chip at the cost of throughput; proper cooling prevents both.
- **Silent data corruption** from mercurial compute units is a real hyperscaler problem: Google and Meta report on the order of one affected machine per thousand. Screen hardware before and during critical training runs.
- **Checkpointing** is the primary software fault tolerance mechanism. Llama 3 405B saw an unexpected interruption every ~3 hours on 16K GPUs; checkpoint often, asynchronously, and automate restart.
- **Quantized/low-precision models** (INT4, FP8) may be more sensitive to the residual soft errors that slip through ECC coverage gaps (uncovered SRAM, single-lane bus errors). Prefer full-coverage ECC (H100 covers L1/SRAM) in production.
- **Loss spike monitoring** is the practical defense against silent corruptions that ECC does not catch — a corrupted gradient tensor produces a visible anomaly in the training curve.

## Check Yourself

<details class="check">
<summary>Convert 256,000 FIT into errors per day.</summary>
<p>256,000 / 10⁹ per hour = 2.56 × 10⁻⁴ per hour, × 24 ≈ 0.006 per day: about one error every 160 days.</p>
</details>

<details class="check">
<summary>Why does SECDED need 8 check bits for 64 data bits?</summary>
<p>A Hamming code needs r bits with 2<sup>r</sup> ≥ k + r + 1 to point at any single erroneous position among k + r bits; for k = 64 that gives r = 7. One more overall-parity bit distinguishes single errors (correctable) from double errors (detectable only): 7 + 1 = 8.</p>
</details>

<details class="check">
<summary>A GPU's corrected-ECC counter climbs steadily over a week. What does that suggest and what should you do?</summary>
<p>Most likely a degrading cell or chip (a hard or intermittent fault), not cosmic rays, which are rare and random. Drain the GPU, let the driver remap the rows (or run diagnostics), and replace it if the count keeps rising: rising correctable errors often precede uncorrectable ones.</p>
</details>

<details class="check">
<summary>Why can't ECC catch silent data corruption from a faulty multiplier?</summary>
<p>ECC protects stored and transmitted bits; it checks that what is read equals what was written. A mercurial ALU writes a wrong result with correct check bits, so every later read looks perfectly valid.</p>
</details>

<details class="check">
<summary>With a cluster MTBF of 3 hours and a 1-minute checkpoint cost, roughly how often should you checkpoint?</summary>
<p>Young's approximation gives the optimal interval √(2 × cost × MTBF) = √(2 × 1 min × 180 min) ≈ 19 min. Checkpointing every 15–20 minutes balances checkpoint overhead against expected lost work.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| DRAM Errors in the Wild: A Large-Scale Field Study | B. Schroeder, E. Pinheiro, W.-D. Weber, SIGMETRICS 2009 | [doi.org/10.1145/1555349.1555372](https://doi.org/10.1145/1555349.1555372) | Field DRAM error rates (25,000–75,000 FIT/Mbit), hard faults dominate |
| Cores that don't count | P. Hochschild et al. (Google), HotOS 2021 | [doi.org/10.1145/3458336.3465297](https://doi.org/10.1145/3458336.3465297) | Mercurial cores and silent data corruption at scale |
| Silent Data Corruptions at Scale | H. D. Dixit et al. (Meta), 2021 | [arxiv.org/abs/2102.11245](https://arxiv.org/abs/2102.11245) | Detecting SDC in a large fleet |
| The Llama 3 Herd of Models (Sec. 3.3.4, reliability) | Llama Team, Meta, 2024 | [arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783) | 466 interruptions in 54 days on 16K H100s, failure breakdown |
| JESD89 Measurement and Reporting of Alpha Particle and Terrestrial Cosmic Ray-Induced Soft Errors | JEDEC | [jedec.org/standards-documents/docs/jesd-89a](https://www.jedec.org/standards-documents/docs/jesd-89a) | Reference neutron flux and SER test methods |
| A First Order Approximation to the Optimum Checkpoint Interval | J. W. Young, CACM 1974 | [doi.org/10.1145/361147.361115](https://doi.org/10.1145/361147.361115) | Optimal checkpoint interval √(2·C·MTBF) |

---

**Next:** [Chapter 14 — CPUs](./14_cpus.md) — We examine out-of-order superscalar cores, deep pipelines, and cache hierarchies — and understand when and why CPUs remain relevant for ML workloads.

[← Back to Table of Contents](./README.md)
