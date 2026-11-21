---
title: "Chapter 9 — Memory Types & Technologies"
---

[← Back to Table of Contents](./README.md)

# Chapter 9 — Memory Types & Technologies

Ask most ML practitioners what limits their model, and they'll say "I need more VRAM." That intuition is right, but it understates a deeper truth: it's not just the *capacity* that constrains you, it's the **speed** at which data moves in and out of that memory. The H100's 80 GB of HBM3 has roughly **3.35 TB/s** of bandwidth. Its FP16 tensor cores can theoretically consume **1,979 TFLOP/s** of compute. That ratio — ~0.59 FLOP per byte — tells you almost everything about why inference of large models is memory-bound, not compute-bound.

Understanding *why* different memory technologies exist at different speeds, densities, and price points requires going one level deeper: into the physics and circuit-level tradeoffs that make SRAM fast, DRAM dense, HBM exotic, and NAND non-volatile. Every decision you make about model size, quantization precision, KV-cache length, and batch size is ultimately a negotiation with these tradeoffs.

> **The one-sentence version:** Memory is a hierarchy of speed-vs-density tradeoffs, and in modern AI accelerators the real bottleneck is almost always HBM bandwidth — which is why you should care more about bytes-per-second than about bytes-total.

---

## The Memory Hierarchy

No single memory technology is fast enough, big enough, and cheap enough to satisfy all needs simultaneously. Instead, hardware builds a **hierarchy**: small, fast, expensive memory close to the processor; large, slow, cheap memory far away. Data migrates up and down as needed — a strategy that works because programs have **locality** (they reuse recently-touched data).

<div class="diagram">
<div class="diagram-title">The Memory Hierarchy Pyramid</div>
<div class="layer-stack">
  <div class="layer">Registers — ~KB, &lt;1 ns, on-core, ~TB/s effective throughput</div>
  <div class="layer">L1 Cache — 32–128 KB, ~1–4 ns, on-core, ~3–6 TB/s</div>
  <div class="layer">L2 Cache — 256 KB–4 MB, ~4–12 ns, on-core or shared, ~1–2 TB/s</div>
  <div class="layer">L3 Cache — 4–512 MB, ~20–50 ns, shared across cores, ~200–500 GB/s</div>
  <div class="layer">Main memory (DDR5 / LPDDR5X / GDDR7 / HBM3e / HBM4) — 4 GB–6 TB, ~80–120 ns, ~50–20,000 GB/s</div>
  <div class="layer">NVMe SSD (NAND Flash) — 1–30+ TB, ~100 µs, ~7 GB/s</div>
  <div class="layer">HDD / Object Storage — PB scale, ms–seconds, ~0.2 GB/s</div>
</div>
</div>

### The Hierarchy at a Glance

| Level | Typical Size | Latency (approx.) | Bandwidth | Relative cost/GB | Volatile? |
|-------|:---:|:---:|:---:|:---:|:---:|
| Registers | 1–4 KB | < 1 ns / 1 cycle | ~TB/s (per core) | N/A (on-core) | Yes |
| L1 Cache | 32–128 KB | 1–4 ns / 4–12 cycles | 3–6 TB/s | N/A (on-core) | Yes |
| L2 Cache | 256 KB–16 MB | 5–15 ns / 15–50 cycles | 1–2 TB/s | N/A (on-die) | Yes |
| L3 Cache | 8–512 MB | 20–50 ns / 60–150 cycles | 200–500 GB/s | N/A (on-die) | Yes |
| DRAM (DDR5) | 8 GB–6 TB per server | 80–120 ns load-to-use | 50–70 GB/s per channel | 1× (reference) | Yes |
| HBM3e / HBM4 (accelerator memory) | 80–432 GB per package | ~100–150 ns | 3,350–20,000 GB/s per package | several × DDR5 | Yes |
| NVMe SSD | 500 GB–32 TB | 50–200 µs | 3–14 GB/s | ~$0.08–0.15/GB | **No** |
| HDD | 1–30 TB | 5–20 ms | 0.1–0.3 GB/s | ~$0.02–0.03/GB | **No** |

> Numbers are representative of 2025–2026 consumer and datacenter hardware; always consult datasheets for exact specs. DRAM prices swung sharply in 2025–26 as AI demand absorbed capacity (HBM wafers crowd out commodity DRAM), so treat cost columns as relative, not absolute.

---

## SRAM — The Fast But Expensive Memory

**Static RAM (SRAM)** is the technology behind CPU/GPU caches and register files. Every cache line, every weight in an on-chip scratchpad (e.g., GPU shared memory) is stored in SRAM cells.

### The 6T SRAM Cell

Each SRAM bit is stored by **6 transistors** in a cross-coupled inverter configuration:

```text
         Vdd
          │
     ┌──[P1]──┐     ┌──[P2]──┐
     │        │     │        │
  BL─┤[N3]   ├─Q  /Q─┤[N4]   ├─/BL
     │        │     │        │
     │  [N1]──┘     └──[N2]  │
     │    │              │   │
     └────┘              └───┘
         GND
```

Two cross-coupled CMOS inverters (4 transistors) store the bit in a **stable state** — one side high, one side low. Two **access transistors** (N3, N4) connect the storage nodes to the bit lines (BL, /BL) during read/write, controlled by the word line (not shown). The state is maintained *as long as power is applied* — no refresh needed, hence "static."

### Why SRAM is Fast, Big, and Expensive

- **Fast:** No refresh required; sense amplifiers detect tiny voltage differentials in ~1 ns; direct random access.
- **Large cell area:** 6 transistors per bit vs 1 transistor + 1 capacitor for DRAM. A high-density SRAM bitcell at TSMC N5 is about **0.021 µm²**, and it barely shrank at N3E (still ~0.021 µm²); N2's nanosheets bring it to roughly 0.0175 µm². A current DRAM cell is ~0.001–0.002 µm², so SRAM is 10–20× larger per bit. **SRAM no longer scales like logic**, which is why designers now move large caches onto separate, older-node dies (AMD 3D V-Cache) instead of growing them on the leading-edge die.
- **No refresh:** Access latency is symmetric (no waiting for row activation).

At the chip level, an H100 GPU has roughly **115 MB of on-chip SRAM**: 33 MB of registers and 33 MB of L1/shared memory spread over 132 SMs, plus a 50 MB L2. By contrast, the 80 GB HBM is off-chip. The on-chip SRAM runs at ~TB/s bandwidth; the HBM runs at ~3.35 TB/s but with 100 ns latency.

---

## DRAM — Dense, Affordable, and Surprisingly Weird

**Dynamic RAM (DRAM)** is the technology behind system RAM (DDR), GPU VRAM (GDDR, HBM), and any large memory pool. It sacrifices speed for density by using just **1 transistor + 1 capacitor (1T1C)** per bit.

### The 1T1C DRAM Cell

```text
         Word Line (WL)
              │
         ┌───┤
BL ──────┤ N ├──── Storage Node
         └───┘     │
                   ═   ← capacitor C (stores charge = 1, or no charge = 0)
                   │
                  GND
```

The **capacitor** stores the bit as charge (charged ≈ 1, discharged ≈ 0). The transistor gates access to the capacitor. This is dramatically more area-efficient than SRAM's 6T cell.

### Why DRAM Needs Refreshing

The capacitor **leaks** charge over time (junction leakage through the transistor, dielectric leakage through the capacitor). Within **~64 ms** (the DRAM spec), the charge decays enough to lose the bit. Every DRAM bank must be **refreshed** — read and rewritten — at least every 64 ms (the `tREFI` interval, typically 7.8 µs per row). This means:

- A few percent of DRAM bandwidth goes to refresh, more for the largest (32 Gb) DDR5 dies; DDR5's same-bank refresh lets other banks keep working meanwhile
- All accesses to a row being refreshed must wait (the **refresh stall**)
- Higher temperature → faster leakage → more frequent refresh → higher overhead

### DRAM Timing Parameters

DRAM access is not simple. A **row** must first be opened (**activated**), the data read out to a **row buffer** (sense amplifier array), then addressed by column:

| Timing | Meaning | Typical DDR5 value |
|--------|---------|:-:|
| `tRCD` | Row → Column Delay (RAS to CAS) | ~14–18 ns |
| `tCL` | CAS Latency (column → data out) | ~14–18 ns |
| `tRP` | Row Precharge (close row) | ~14–18 ns |
| `tRAS` | Row Active time (min) | ~32 ns |
| Row miss at the DRAM | `tRP + tRCD + tCL` ≈ | **~45–55 ns** |
| Load-to-use seen by a CPU core | + controller, fabric, queues | **~80–120 ns** |

This ~100 ns random access latency is why DRAM-bound workloads with poor locality are slow: every cache miss pays it. The DRAM cell itself has barely sped up in twenty years; what grew is bandwidth, by fetching wider bursts from more banks in parallel.

---

## The DRAM Family: DDR5, LPDDR5, GDDR6/6X, HBM

The DRAM cell itself is universal, but the **interface and packaging** determines bandwidth, power, and cost:

### DDR5 — Server and Desktop Main Memory

**DDR5** (Double Data Rate 5, JEDEC JESD79-5) is the current mainstream server/desktop DRAM standard.

- **Transfer rate:** 4,800–6,400 MT/s on mainstream DIMMs; MRDIMMs (multiplexed-rank) reach 8,800 MT/s on Intel Xeon 6
- **Bus width:** 32 or 64 bits per channel (with ECC: 72 bits)
- **Bandwidth:** 1 channel × 64-bit bus × 6,400 MT/s = ~51.2 GB/s per channel; dual-channel ≈ 102 GB/s
- **Capacity:** Modules from 16 GB to 256 GB (server RDIMM)
- **Voltage:** 1.1 V (down from DDR4's 1.2 V)
- **Where used:** CPU host memory, "system RAM"

For ML, host memory speed matters for data loading, CPU-side preprocessing, and gradient checkpointing/offloading with [ZeRO-Infinity](./28_multi_gpu_and_distributed_training.md).

### LPDDR5 / LPDDR5X — Mobile and Edge

**Low-Power DDR5** sacrifices some bandwidth for dramatically lower power — critical for phones and edge AI hardware.

- **Transfer rate:** LPDDR5X now ships at 8,533–10,700 MT/s; Apple M4 Max runs LPDDR5X-8533
- **Bus width:** 32 or 64 bits (multiple channels in parallel on SoC)
- **Bandwidth:** Apple M4 Max: 512-bit bus × 8,533 MT/s ÷ 8 ≈ **546 GB/s**; the base M5 reaches 153 GB/s
- **Voltage:** 1.05 V (LPDDR5X: 0.9 V)
- **Where used:** Smartphone SoCs (Snapdragon, Dimensity), Apple M-series, edge NPUs, Renesas automotive SoCs

Apple's M3 Ultra (2025 Mac Studio) runs 819 GB/s to up to 512 GB of unified memory, enough to hold a 4-bit 400B-parameter model, and competitive in bandwidth with older datacenter GPUs. The "unified" part means CPU, GPU and Neural Engine share the same pool: no PCIe copy overhead.

### GDDR6 / GDDR6X — Consumer GPU VRAM

**Graphics DDR** is optimized for high-bandwidth GPU memory, sacrificing capacity and power efficiency for raw throughput.

- **GDDR6 transfer rate:** up to 16–20 Gbps per pin; **GDDR7** (2025, RTX 50 series) uses three-level **PAM3** signaling at 28–32 Gbps, heading to 36+
- **GDDR6X** (NVIDIA): uses **PAM4 signaling** (4 voltage levels vs 2 in NRZ) to reach 21–23 Gbps per pin
- **Bus width:** 256 or 384 bits (multiple GDDR chips in parallel)
- **Bandwidth:** RTX 4090: 384-bit × 21 Gbps = ~1,008 GB/s
- **Capacity:** 8–32 GB on consumer cards (RTX 5090: 32 GB on a 512-bit bus, 1,792 GB/s); up to 96 GB on workstation boards
- **Voltage:** ~1.35 V (GDDR6), ~1.36 V (GDDR6X)
- **Where used:** Consumer and workstation GPUs (RTX 40/50, RTX PRO, Radeon RX 9000), and inference cards that trade bandwidth for cost

GDDR6/6X has high bandwidth but is **planar** — chips sit beside the GPU die on the PCB. This limits how close they are and caps the bus at roughly 384–512 bits: every extra 32 bits is another chip and another set of fast traces on the board.

### HBM2e / HBM3 / HBM3e — High Bandwidth Memory

**HBM (High Bandwidth Memory)** is the most expensive and highest-bandwidth DRAM technology, used in datacenter GPUs and AI accelerators. Instead of flat chips on a PCB, HBM **stacks DRAM dies vertically** using **Through-Silicon Vias (TSVs)** and places the stack on an **interposer** right next to the compute die.

<div class="diagram">
<div class="diagram-title">HBM Stack Architecture</div>
<div class="flow">
  <div class="flow-node accent wide">DRAM Die 8 (top) — active memory array</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">DRAM Die 7</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">... (stacked dies) ...</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">DRAM Die 1 (bottom)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Logic Base Die — I/O, refresh control, power mgmt</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">Silicon Interposer (2.5D) — connects HBM stacks to GPU die</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">GPU / Compute Die — connected via 1,024-bit (per stack) interface</div>
</div>
</div>

The key advantage: the **memory bus width is enormous** — 1,024 bits *per HBM stack* vs 256–384 bits for GDDR. This allows extreme bandwidth without requiring extreme per-pin signaling rates. See [Appendix E — Packaging & Integration](./appendix_e_packaging.md) for TSV and interposer details.

### HBM Version Comparison

Sliced open, an HBM package looks like this (not to scale; an HBM3e stack is ~0.7 mm tall and each DRAM die is thinned to ~30–50 µm):

<div class="xsec">
<div class="xs-row"><div class="xs dram">DRAM die 12</div><div class="xs gap" style="flex:2.2"></div><div class="xs dram">DRAM die 12</div></div>
<div class="xs-row"><div class="xs tsv">⋮ TSVs through every die</div><div class="xs gap" style="flex:2.2"></div><div class="xs tsv">⋮</div></div>
<div class="xs-row"><div class="xs dram">DRAM die 1</div><div class="xs gap" style="flex:2.2"></div><div class="xs dram">DRAM die 1</div></div>
<div class="xs-row"><div class="xs die">HBM base (logic) die<small>PHY, test, repair</small></div><div class="xs die" style="flex:2.2">GPU / compute die<small>HBM controllers + PHYs on its edges</small></div><div class="xs die">HBM base die</div></div>
<div class="xs-row"><div class="xs bump"></div><div class="xs bump" style="flex:2.2"></div><div class="xs bump"></div></div>
<div class="xs-row"><div class="xs interposer" style="flex:4.2">Silicon interposer (CoWoS-S) or RDL + bridges (CoWoS-L): ~1,000–2,000 wires per stack, a few mm long</div></div>
<div class="xs-row"><div class="xs bump" style="flex:4.2"></div></div>
<div class="xs-row"><div class="xs substrate" style="flex:4.2">Organic package substrate<small>fans the signals out to ~0.4–1 mm ball pitch</small></div></div>
<div class="xs-row"><div class="xs board" style="flex:4.2">Board / OAM module</div></div>
</div>

| Version | Per-Stack BW | Capacity/stack | Stacked dies | Where used |
|---------|:---:|:---:|:---:|---|
| HBM2 | 256–307 GB/s | 4–8 GB | 4–8 | V100, A100 40GB, AMD Vega |
| HBM2e | 410–460 GB/s | 8–16 GB | 8 | A100 80GB, MI250X, TPU v5p, Gaudi 3 |
| HBM3 | up to 819 GB/s | 16–24 GB | 8–12 | H100, MI300X |
| HBM3e | ~1.2 TB/s | 24–36 GB | 8–12 | H200, B200/B300, MI325X/MI355X, Ironwood, Trainium3 |
| **HBM4** | ~2–2.8 TB/s (2,048-bit interface) | 24–64 GB | 12–16 | Rubin, MI455X (2026) |

HBM4 (JEDEC JESD270-4, April 2025) doubles the interface to 2,048 bits per stack, so it gains bandwidth mostly by going wider rather than faster, and allows the base die to be built on a logic process (TSMC N12/N3) so accelerator vendors can put custom logic under the DRAM. Samsung and Micron began volume shipments in early 2026.

### Bandwidth and Capacity: Full System Examples

| GPU/Accelerator | Memory Type | Stacks | Total BW | Total Capacity |
|----------------|------------|:---:|:---:|:---:|
| NVIDIA A100 SXM4 | HBM2e | 5 | 2,000 GB/s | 80 GB |
| NVIDIA H100 SXM5 | HBM3 | 5 | 3,350 GB/s | 80 GB |
| NVIDIA H200 SXM | HBM3e | 6 | 4,800 GB/s | 141 GB |
| NVIDIA B200 | HBM3e | 8 | 8,000 GB/s | 180–192 GB |
| AMD MI355X | HBM3e | 8 | 8,000 GB/s | 288 GB |
| NVIDIA Rubin (2026) | HBM4 | 8 | ~20,000–22,000 GB/s | 288 GB |
| AMD MI455X (2026) | HBM4 | 12 | ~19,600 GB/s | 432 GB |
| AMD MI300X | HBM3 | 8 | 5,300 GB/s | 192 GB |
| RTX 5090 | GDDR7 | — | 1,792 GB/s | 32 GB |
| Apple M3 Ultra | LPDDR5 | — | ~819 GB/s | up to 512 GB |

---

## DRAM Technology Comparison Summary

<div class="diagram">
<div class="diagram-title">Memory Technology Positioning</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">High Bandwidth (HBM3e / GDDR6X)</div>
    <ul>
      <li>1,000–22,000 GB/s per package</li>
      <li>Small capacity (16–192 GB)</li>
      <li>Highest cost per GB, and supply-constrained</li>
      <li>Close to compute die (short traces → low latency)</li>
      <li>Used in datacenter GPUs, AI accelerators</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">High Capacity (DDR5 / LPDDR5)</div>
    <ul>
      <li>50–300 GB/s bandwidth</li>
      <li>Large capacity (8 GB–6 TB in servers)</li>
      <li>Low cost ($3–5/GB for DDR5)</li>
      <li>Far from compute (motherboard traces → higher latency)</li>
      <li>Used in CPU host memory, unified memory SoCs</li>
    </ul>
  </div>
</div>
</div>

---

## NAND Flash & SSDs

**NAND flash** is non-volatile: data persists without power. It stores charge on a **floating gate** (or a charge trap layer in modern 3D NAND), which changes the transistor's threshold voltage. Reading is non-destructive; writing requires an erase cycle (slow, ~1–10 ms).

### NAND Cell Types and Tradeoffs

| Type | Bits/cell | Read Latency | Write Endurance | Use Case |
|------|:---:|:---:|:---:|---|
| SLC (Single-Level Cell) | 1 | ~20 µs | ~100,000 cycles | Enterprise, WAL logs |
| MLC (Multi-Level Cell) | 2 | ~60–100 µs | ~3,000–10,000 cycles | Consumer SSDs |
| TLC (Triple-Level Cell) | 3 | ~100–200 µs | ~1,000–3,000 cycles | Most consumer SSDs |
| QLC (Quad-Level Cell) | 4 | ~200–400 µs | ~100–1,000 cycles | Cold storage, bulk data |

Modern **3D NAND** (200–330+ layers in 2025–26 parts) stacks NAND cells vertically to increase density and improve endurance vs planar NAND. An **NVMe SSD** reaches ~7 GB/s sequential read over PCIe 4.0 ×4 (Samsung 990 Pro) and ~14 GB/s over PCIe 5.0 ×4 (Samsung 9100 Pro) — fast for storage, but 500× slower than HBM3.

For ML: SSDs matter for **dataset streaming** (loading training data faster than it is consumed by the GPU) and **model offloading** (ZeRO-Infinity, DeepSpeed with NVMe offload). NVMe SSDs at 7 GB/s can feed a GPU preprocessing pipeline but cannot substitute for VRAM.

---

## Volatile vs Non-Volatile: The Key Distinction

| Property | SRAM | DRAM | NAND Flash | HDD |
|----------|:---:|:---:|:---:|:---:|
| Retains data on power-off | No | No | **Yes** | **Yes** |
| Random read latency | ~1 ns | ~70 ns | ~100 µs | ~10 ms |
| Sequential bandwidth | ~TB/s | ~50–5300 GB/s | ~7 GB/s | ~0.2 GB/s |
| Bit density (relative) | 1× | ~6× | ~30× | ~100× |
| Endurance | Unlimited | Unlimited | Limited (cycles) | Limited (mechanical) |

**Volatile** memory (SRAM, DRAM) loses all data on power loss — which is why model checkpointing to SSD/disk matters for long training runs.

---

## Python: Computing Model Memory Footprints

The most immediate consequence of memory types for ML practitioners: **does your model fit in VRAM?**

```python
import math

# --- Model weight memory footprint by dtype ---
param_counts = {
    "7B":  7_000_000_000,
    "13B": 13_000_000_000,
    "70B": 70_000_000_000,
    "405B": 405_000_000_000,
}

dtype_bytes = {
    "fp32":  4,   # 32-bit IEEE-754 float
    "bf16":  2,   # bfloat16 (16 bits: 1 sign, 8 exp, 7 mantissa)
    "fp16":  2,   # IEEE half-precision (1 sign, 5 exp, 10 mantissa)
    "int8":  1,   # 8-bit integer (AWQ, GPTQ, bitsandbytes)
    "int4":  0.5, # 4-bit integer — 2 values per byte (NF4, GGUF Q4)
    "fp4":   0.5, # 4-bit float (NVFP4, MXFP4 emerging)
}

print(f"{'Model':<8} {'fp32':>8} {'bf16':>8} {'int8':>8} {'int4':>8}  (GB)")
print("-" * 50)
for name, params in param_counts.items():
    row = []
    for fmt in ["fp32", "bf16", "int8", "int4"]:
        gb = params * dtype_bytes[fmt] / 1e9
        row.append(f"{gb:>7.1f}")
    print(f"{name:<8} {'  '.join(row)}")
```

Sample output:
```
Model       fp32    bf16    int8    int4  (GB)
--------------------------------------------------
7B         28.0    14.0     7.0     3.5
13B        52.0    26.0    13.0     6.5
70B       280.0   140.0    70.0    35.0
405B     1620.0   810.0   405.0   202.5
```

> A 70B model in BF16 needs **140 GB** of VRAM — that's two H100-80GB GPUs just for weights. With KV-cache for a 32K context, full fine-tuning (Adam in mixed precision needs ~16 bytes per parameter in total, 8× the BF16 weights), and activations, the real requirement grows substantially.

```python
# --- HBM bandwidth and capacity constraints for inference ---
hbm_configs = {
    "H100 SXM5 (80GB)":  {"capacity_gb": 80,  "bandwidth_tbs": 3.35},
    "H200 SXM (141GB)":  {"capacity_gb": 141, "bandwidth_tbs": 4.80},
    "MI300X (192GB)":    {"capacity_gb": 192, "bandwidth_tbs": 5.30},
    "A100 SXM4 (80GB)":  {"capacity_gb": 80,  "bandwidth_tbs": 2.00},
}

model_bf16_gb = {"7B": 14, "13B": 26, "70B": 140, "405B": 810}

print("\nModel fits on GPU? (weights only, BF16):")
print(f"{'GPU':<28} {'Cap(GB)':>8} {'BW(TB/s)':>10} {'7B':>5} {'13B':>6} {'70B':>6} {'405B':>7}")
print("-" * 75)
for gpu, cfg in hbm_configs.items():
    fits = {m: "✓" if gb <= cfg["capacity_gb"] else "✗"
            for m, gb in model_bf16_gb.items()}
    print(f"{gpu:<28} {cfg['capacity_gb']:>8} {cfg['bandwidth_tbs']:>10.2f} "
          f"{fits['7B']:>5} {fits['13B']:>6} {fits['70B']:>6} {fits['405B']:>7}")
```

### Bandwidth Arithmetic: Why Decode is Memory-Bound

During autoregressive inference (token-by-token generation), each forward pass touches *all* model weights once but produces only a few output tokens. The weight matrix for a single linear layer with shape $[d_{\text{model}}, d_{\text{ff}}]$ in BF16:

$$\text{Bytes loaded} = d_{\text{model}} \times d_{\text{ff}} \times 2 \text{ bytes}$$

For LLaMA-2 70B at batch size 1, *every* weight is read once per generated token: $70 \times 10^9 \times 2$ bytes $= 140$ GB per token in BF16. The model does not fit on one 80 GB H100, so assume it is split across four with tensor parallelism, whose HBM bandwidths add:

$$\text{Time per token} \ge \frac{140 \text{ GB}}{4 \times 3.35 \text{ TB/s}} \approx 10.4 \text{ ms} \;\Rightarrow\; \le 96 \text{ tokens/s}$$

Measured batch-1 decode for 70B models on 4–8 H100s sits at roughly 40–70 % of that ceiling once communication, KV-cache reads and kernel launch overheads are added, confirming that **decode is bandwidth-bound, not compute-bound**: the GPUs perform only 140 GFLOP per token, about 0.1 % of their peak. Storing the weights in FP8 or INT8 halves the bytes and nearly doubles tokens/s; 4-bit roughly doubles it again. That is a purely bandwidth-driven win.

---

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>From HBM3 to HBM4 the per-pin speed barely changes; the bandwidth comes from doubling the wires per stack. What does that imply about where the next 2× must come from, and why are vendors now putting custom logic (and even compute) into the HBM base die?</p>
</div>

<div class="lab" data-lab="fit"><noscript><div class="lab-noscript">The will-it-fit lab needs JavaScript.</div></noscript></div>

## Why This Matters for Model Optimization

<div class="diagram">
<div class="diagram-title">Memory Technology → ML Optimization Decisions</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">VRAM Capacity Caps Model Size</div>
    <div class="card-desc">A 70B BF16 model needs 140 GB VRAM (weights only). This forces multi-GPU sharding (tensor/pipeline parallelism) or quantization to INT8/INT4 to fit on fewer GPUs.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">HBM Bandwidth Caps Decode Speed</div>
    <div class="card-desc">Batch-1 generation reads every weight per token: 140 GB for a 70B BF16 model. Each 3.35 TB/s H100 contributes ~24 tokens/s of ceiling, so four of them top out near 96 tokens/s. Bandwidth, not FLOP/s, sets the limit.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Quantization is a Bandwidth Optimization</div>
    <div class="card-desc">INT8 halves bytes loaded per weight → ~2× decode throughput. INT4 halves again. The FLOP savings are secondary — the bandwidth savings are primary for memory-bound decode.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">KV Cache Competes with Weights</div>
    <div class="card-desc">KV cache = 2 × layers × KV heads × head_dim × seq_len × dtype_bytes. LLaMA-2 70B (GQA, 8 KV heads) needs 320 KB per token in BF16, ~10.7 GB per 32K-token sequence; a batch of 16 such requests needs more memory than the weights.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">HBM Pays for Itself in Throughput</div>
    <div class="card-desc">HBM costs several times DDR5 per GB but delivers 50–100× the bandwidth per package (8 TB/s on a B200 vs ~0.5 TB/s for a 12-channel DDR5 socket). For memory-bound inference that bandwidth turns directly into tokens per second.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">NVMe Offloading (ZeRO-Infinity)</div>
    <div class="card-desc">DeepSpeed's ZeRO-Infinity offloads optimizer states (and even weights) to NVMe. At 7 GB/s, this only works if the compute time >> weight-reload time — i.e., at very large batch sizes.</div>
  </div>
</div>
</div>

---

## Key Takeaways

- The **memory hierarchy** spans registers (~TB/s, KB) through DRAM (~GB/s–TB/s, GB) to SSD (~GB/s, TB): each level trades off speed, capacity, and cost.
- **SRAM** (6T cell) is fast (~1 ns), but large and expensive — used for on-chip caches, register files, and GPU shared memory.
- **DRAM** (1T1C cell) is dense and cheap but requires **periodic refresh** (charge leakage every ~64 ms) and has ~70 ns random latency.
- **HBM** achieves extreme bandwidth (~1.2 TB/s per HBM3e stack, ~2+ TB/s per HBM4 stack; 8–20+ TB/s per package) by stacking DRAM dies with TSVs and placing them millimetres from the compute die on an interposer, with a 1,024-bit interface per stack (2,048-bit for HBM4).
- **GDDR6X/GDDR7** achieve high bandwidth via fast per-pin signaling (PAM4 at ~21–23 Gbps; PAM3 at 28–32+ Gbps), used in consumer and workstation GPUs: cheaper than HBM, but with far less bandwidth per package.
- **LPDDR5** is a power-optimized DRAM for mobile/edge SoCs; Apple's unified memory architecture uses it to blur the CPU/GPU memory boundary.
- For AI inference, **VRAM capacity** caps model size and KV-cache length; **HBM bandwidth** caps tokens-per-second in memory-bound autoregressive decode.
- **Quantization** (INT8, INT4) primarily wins by **reducing bytes loaded per weight**, directly improving memory-bandwidth-limited throughput — capacity savings are a bonus, not the mechanism.

## Check Yourself

<details class="check">
<summary>Why is SRAM 10–20× larger per bit than DRAM, and why has that gap stopped closing?</summary>
<p>An SRAM bit is six transistors; a DRAM bit is one transistor and a capacitor stacked above it. Logic transistors kept shrinking, but SRAM's bitcell barely changed from N5 to N3E because its tightly matched transistors and wires stopped scaling, so caches now consume a growing share of leading-edge die area.</p>
</details>

<details class="check">
<summary>A 70B model in BF16 runs batch-1 decode on four H100s. What is the bandwidth ceiling on tokens/s, and what single change would raise it most?</summary>
<p>140 GB per token ÷ (4 × 3.35 TB/s) ≈ 10.4 ms, so ≤ ~96 tokens/s. Quantizing the weights to 4 bits cuts bytes per token ~4× (to ~35–40 GB including scales), raising the ceiling ~3.5–4×; more GPUs also help but add communication.</p>
</details>

<details class="check">
<summary>How big is the KV cache for one 32K-token sequence on LLaMA-2 70B in BF16?</summary>
<p>2 (K and V) × 80 layers × 8 KV heads × 128 dims × 2 bytes = 327,680 bytes ≈ 320 KB per token; × 32,768 tokens ≈ 10.7 GB.</p>
</details>

<details class="check">
<summary>What does HBM4 change compared with HBM3e, and why?</summary>
<p>It doubles the interface to 2,048 bits per stack instead of pushing per-pin speeds much higher, because faster signaling across thousands of short interposer wires costs too much power. It also lets the base die be built on a logic process, so vendors can add custom logic beneath the DRAM.</p>
</details>

<details class="check">
<summary>Why do consumer GPUs use GDDR instead of HBM?</summary>
<p>GDDR chips sit on an ordinary PCB beside the GPU and need no silicon interposer or advanced packaging, so they are far cheaper and not capacity-constrained by CoWoS. The price is a narrow bus (256–512 bits) and therefore much lower bandwidth and capacity per package.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| JEDEC JESD270-4 High Bandwidth Memory (HBM4) | JEDEC, 2025 | [jedec.org/standards-documents/docs/jesd270-4](https://www.jedec.org/standards-documents/docs/jesd270-4) | 2,048-bit interface, stack heights, base-die options |
| JEDEC JESD79-5 DDR5 SDRAM | JEDEC | [jedec.org/standards-documents/docs/jesd79-5](https://www.jedec.org/standards-documents/docs/jesd79-5) | DDR5 timing parameters and sub-channel architecture |
| Memory Systems: Cache, DRAM, Disk | B. Jacob, S. Ng, D. Wang, 2007 | [elsevier.com/books/memory-systems/jacob/978-0…](https://www.elsevier.com/books/memory-systems/jacob/978-0-12-379751-3) | DRAM organisation, timing and refresh |
| NVIDIA H100 Tensor Core GPU Architecture | NVIDIA, 2022 | [resources.nvidia.com/en-us-tensor-core/gtc22-…](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper) | SM register file and L1/shared capacity, 50 MB L2, HBM3 configuration |
| Efficient Memory Management for Large Language Model Serving with PagedAttention | W. Kwon et al., SOSP 2023 | [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180) | KV-cache sizes and why they dominate serving memory |
| Samsung and Micron begin shipping HBM4 | The Register, Feb 2026 | [theregister.com/2026/02/13/samsung_and_micron…](https://www.theregister.com/2026/02/13/samsung_and_micron_start_shipping/) | HBM4 production timing |

---

**Next:** [Chapter 10 — Memory Management](./10_memory_management.md) — We explore how the processor manages this hierarchy — virtual memory, TLBs, cache coherence, and NUMA — and why it matters for tensor layout, multi-GPU locality, and training performance.

[← Back to Table of Contents](./README.md)
