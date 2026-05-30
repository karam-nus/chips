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
  <div class="layer">Main DRAM (DDR5 / LPDDR5 / HBM) — 4–1000+ GB, ~60–100 ns, ~50–3350 GB/s</div>
  <div class="layer">NVMe SSD (NAND Flash) — 1–30+ TB, ~100 µs, ~7 GB/s</div>
  <div class="layer">HDD / Object Storage — PB scale, ms–seconds, ~0.2 GB/s</div>
</div>
</div>

### The Hierarchy at a Glance

| Level | Typical Size | Latency (approx.) | Bandwidth | Cost/GB (2025 approx.) | Volatile? |
|-------|:---:|:---:|:---:|:---:|:---:|
| Registers | 1–4 KB | < 1 ns / 1 cycle | ~TB/s (per core) | N/A (on-core) | Yes |
| L1 Cache | 32–128 KB | 1–4 ns / 4–12 cycles | 3–6 TB/s | N/A (on-core) | Yes |
| L2 Cache | 256 KB–16 MB | 5–15 ns / 15–50 cycles | 1–2 TB/s | N/A (on-die) | Yes |
| L3 Cache | 8–512 MB | 20–50 ns / 60–150 cycles | 200–500 GB/s | N/A (on-die) | Yes |
| DRAM (DDR5) | 8–512 GB | 60–100 ns / 200–300 cycles | 50–90 GB/s | ~$3–5/GB | Yes |
| HBM3 (GPU VRAM) | 16–192 GB | ~100 ns | 1,200–3,350 GB/s | ~$30–80/GB | Yes |
| NVMe SSD | 500 GB–32 TB | 50–200 µs | 3–14 GB/s | ~$0.08–0.15/GB | **No** |
| HDD | 1–30 TB | 5–20 ms | 0.1–0.3 GB/s | ~$0.02–0.03/GB | **No** |

> Numbers are representative of 2024–2025 consumer/datacenter hardware. Always consult datasheets for exact specs.

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
- **Large cell area:** 6 transistors per bit vs 1 transistor + 1 capacitor for DRAM. An SRAM cell in a 5 nm process is roughly **0.021–0.030 µm²** per bit; DRAM is ~0.002–0.006 µm² per bit — SRAM is ~5–10× larger.
- **No refresh:** Access latency is symmetric (no waiting for row activation).

At the chip level, an H100 GPU has roughly **50 MB of on-chip SRAM** distributed across 132 SMs (shared memory + L1 cache + register file). By contrast, the 80 GB HBM is off-chip. The on-chip SRAM runs at ~TB/s bandwidth; the HBM runs at ~3.35 TB/s but with 100 ns latency.

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

- ~1–3% of total DRAM bandwidth is consumed by refresh operations
- All accesses to a row being refreshed must wait (the **refresh stall**)
- Higher temperature → faster leakage → more frequent refresh → higher overhead

### DRAM Timing Parameters

DRAM access is not simple. A **row** must first be opened (**activated**), the data read out to a **row buffer** (sense amplifier array), then addressed by column:

| Timing | Meaning | Typical DDR5 value |
|--------|---------|:-:|
| `tRCD` | Row → Column Delay (RAS to CAS) | 16–40 ns |
| `tCL` | CAS Latency (column → data out) | 14–32 ns |
| `tRP` | Row Precharge (close row) | 16–40 ns |
| `tRAS` | Row Active time (min) | 32–70 ns |
| Full random access | `tRCD + tCL + tRP` ≈ | **60–100 ns** |

This ~70 ns random access latency is why DRAM-bound workloads with poor locality are slow — every cache miss pays this price.

---

## The DRAM Family: DDR5, LPDDR5, GDDR6/6X, HBM

The DRAM cell itself is universal, but the **interface and packaging** determines bandwidth, power, and cost:

### DDR5 — Server and Desktop Main Memory

**DDR5** (Double Data Rate 5, JEDEC JESD79-5) is the current mainstream server/desktop DRAM standard.

- **Transfer rate:** 4,800–7,200 MT/s per channel (MT = megatransfers)
- **Bus width:** 32 or 64 bits per channel (with ECC: 72 bits)
- **Bandwidth:** 1 channel × 64-bit bus × 6,400 MT/s = ~51.2 GB/s per channel; dual-channel ≈ 102 GB/s
- **Capacity:** Modules from 16 GB to 256 GB (server RDIMM)
- **Voltage:** 1.1 V (down from DDR4's 1.2 V)
- **Where used:** CPU host memory, "system RAM"

For ML, host memory speed matters for data loading, CPU-side preprocessing, and gradient checkpointing/offloading with [ZeRO-Infinity](./28_multi_gpu_and_distributed_training.md).

### LPDDR5 / LPDDR5X — Mobile and Edge

**Low-Power DDR5** sacrifices some bandwidth for dramatically lower power — critical for phones and edge AI hardware.

- **Transfer rate:** LPDDR5X up to 8,533 MT/s; Apple LPDDR5 on M3 ≈ 6,400 MT/s per channel
- **Bus width:** 32 or 64 bits (multiple channels in parallel on SoC)
- **Bandwidth:** Apple M3 Max: 6 channels × 32-bit × 6,400 MT/s ≈ ~300 GB/s
- **Voltage:** 1.05 V (LPDDR5X: 0.9 V)
- **Where used:** Smartphone SoCs (Snapdragon, Dimensity), Apple M-series, edge NPUs, Renesas automotive SoCs

Apple's M3 Ultra unified memory runs at 800 GB/s, making it competitive with older GPU VRAM for certain workloads. The "unified" part means CPU and GPU share the same LPDDR5 pool — no PCIe copy overhead.

### GDDR6 / GDDR6X — Consumer GPU VRAM

**Graphics DDR** is optimized for high-bandwidth GPU memory, sacrificing capacity and power efficiency for raw throughput.

- **GDDR6 transfer rate:** up to 16–18 Gbps per pin
- **GDDR6X** (NVIDIA): uses **PAM4 signaling** (4 voltage levels vs 2 in NRZ) to reach 21–23 Gbps per pin
- **Bus width:** 256 or 384 bits (multiple GDDR chips in parallel)
- **Bandwidth:** RTX 4090: 384-bit × 21 Gbps = ~1,008 GB/s
- **Capacity:** 8–24 GB per GPU
- **Voltage:** ~1.35 V (GDDR6), ~1.36 V (GDDR6X)
- **Where used:** Consumer/prosumer NVIDIA/AMD GPUs (RTX 3000/4000/5000, RX 7000 series)

GDDR6/6X has high bandwidth but is **planar** — chips sit beside the GPU die on the PCB. This limits how close they are and imposes a practical bus width ceiling of ~384 bits.

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

| Version | Per-Stack BW | Capacity/stack | Stacked dies | Where used |
|---------|:---:|:---:|:---:|---|
| HBM2 | 256 GB/s | 4–8 GB | 4–8 | AMD Vega, A100 (early) |
| HBM2e | 460 GB/s | 8–16 GB | 8 | NVIDIA A100, AMD MI250X |
| HBM3 | 819 GB/s | 16–24 GB | 12 | NVIDIA H100, AMD MI300X |
| HBM3e | 1,200 GB/s | 24–36 GB | 12–16 | NVIDIA H200, AMD MI325X |

### Bandwidth and Capacity: Full System Examples

| GPU/Accelerator | Memory Type | Stacks | Total BW | Total Capacity |
|----------------|------------|:---:|:---:|:---:|
| NVIDIA A100 SXM4 | HBM2e | 5 | 2,000 GB/s | 80 GB |
| NVIDIA H100 SXM5 | HBM3 | 5 | 3,350 GB/s | 80 GB |
| NVIDIA H200 SXM | HBM3e | 5 | 4,800 GB/s | 141 GB |
| AMD MI300X | HBM3 | 8 | 5,300 GB/s | 192 GB |
| RTX 4090 | GDDR6X | — | 1,008 GB/s | 24 GB |
| Apple M3 Ultra | LPDDR5 | — | ~800 GB/s | 192 GB |

---

## DRAM Technology Comparison Summary

<div class="diagram">
<div class="diagram-title">Memory Technology Positioning</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">High Bandwidth (HBM3e / GDDR6X)</div>
    <ul>
      <li>1,200–5,300 GB/s bandwidth</li>
      <li>Small capacity (16–192 GB)</li>
      <li>Very high cost ($30–80/GB)</li>
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

Modern **3D NAND** (64–232 layers) stacks NAND cells vertically to increase density and improve endurance vs planar NAND. A **NVMe SSD** (e.g., Samsung 990 Pro) achieves ~7.5 GB/s sequential read over PCIe 5.0 × 4 — fast for storage, but 500× slower than HBM3.

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

> A 70B model in BF16 needs **140 GB** of VRAM — that's two H100-80GB GPUs just for weights. With KV-cache for a 32K context, optimizer states for fine-tuning (8× extra for Adam), and activations, the real requirement grows substantially.

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

For LLaMA-2 70B: $d_{\text{model}} = 8192$, $d_{\text{ff}} = 28672$. One FFN layer = $2 \times 8192 \times 28672 \approx 0.47$ GB per layer. With 80 layers ≈ **37 GB** of weight reads per token. At HBM3's 3.35 TB/s:

$$\text{Time per token} = \frac{37 \text{ GB}}{3,350 \text{ GB/s}} \approx 11 \text{ ms} \approx 90 \text{ tokens/s}$$

This is within striking distance of observed throughput (~40–80 tokens/s on H100 for 70B), confirming that **decode is bandwidth-bound, not compute-bound**. Quantization to INT8 halves the byte count → ~doubles tokens/s — a purely bandwidth-driven win.

---

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
    <div class="card-desc">Autoregressive generation loads all weights per token. At 3.35 TB/s (H100), a 70B BF16 model yields ~90 tokens/s ceiling — bandwidth, not FLOP/s, sets the limit.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Quantization is a Bandwidth Optimization</div>
    <div class="card-desc">INT8 halves bytes loaded per weight → ~2× decode throughput. INT4 halves again. The FLOP savings are secondary — the bandwidth savings are primary for memory-bound decode.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">KV Cache Competes with Weights</div>
    <div class="card-desc">KV cache = 2 × layers × heads × head_dim × seq_len × dtype_bytes. At 32K context, 70B LLaMA-2 in FP16 needs ~20 GB KV cache — competing with the 140 GB weights for VRAM.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">HBM Pays for Itself in Throughput</div>
    <div class="card-desc">HBM3 at $30–80/GB is 10–20× more expensive than DDR5. That premium buys 30–60× more bandwidth. For batch inference, this bandwidth translates directly into more requests/second.</div>
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
- **HBM** achieves extreme bandwidth (1,200–5,300 GB/s) by stacking DRAM dies vertically with TSVs and placing them millimeters from the compute die on an interposer, using a 1,024-bit-wide interface per stack.
- **GDDR6/6X** achieves high bandwidth via fast per-pin signaling (~21 Gbps/pin, PAM4), used in consumer GPUs; lower cost than HBM but lower peak bandwidth at scale.
- **LPDDR5** is a power-optimized DRAM for mobile/edge SoCs; Apple's unified memory architecture uses it to blur the CPU/GPU memory boundary.
- For AI inference, **VRAM capacity** caps model size and KV-cache length; **HBM bandwidth** caps tokens-per-second in memory-bound autoregressive decode.
- **Quantization** (INT8, INT4) primarily wins by **reducing bytes loaded per weight**, directly improving memory-bandwidth-limited throughput — capacity savings are a bonus, not the mechanism.

---

*Next: [Chapter 10 — Memory Management](./10_memory_management.md), where we explore how the processor manages this hierarchy — virtual memory, TLBs, cache coherence, and NUMA — and why it matters for tensor layout, multi-GPU locality, and training performance.*

[← Back to Table of Contents](./README.md)
