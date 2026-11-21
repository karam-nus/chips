---
title: "Chapter 12 — Buses & Interconnects"
---

[← Back to Table of Contents](./README.md)

# Chapter 12 — Buses & Interconnects

Computation happens in logic gates. Results live in registers and caches. Model weights live in HBM. Activations are passed between GPUs. Training gradients are all-reduced across a cluster of thousands of nodes. All of this requires data to *move* — and how fast it can move, over what distances, at what latency and bandwidth, shapes every architectural decision from within a single GPU to across a data center.

This chapter traces data movement at every scale: the microscopic wires inside a chip that link functional units, the copper traces on a board that connect GPU to CPU, the optical cables linking racks in a training cluster. Each scale has its own technology, its own bandwidth regime, and its own implications for how you partition a model across hardware.

> **The one-sentence version:** Every interconnect in the stack trades off bandwidth, latency, distance, cost, and protocol complexity — and the ones you choose determine which parallelism strategies are viable, because tensor parallelism needs microsecond, TB/s-class links while pipeline parallelism can tolerate GB/s inter-node links.

---

## On-Chip Interconnects: Wiring a Die Together

### The Bus — Simple but Limited

The oldest on-chip communication primitive is the **bus**: a shared set of wires that multiple functional units connect to. A bus is simple and area-efficient. Its pathology is equally simple: only one sender can use the bus at a time. As chips grew from tens to thousands of functional units, bus contention became crippling.

A **shared bus** model: $N$ masters, 1 shared wire. At any cycle, only 1 transfer occurs. **Effective bandwidth per master = total bus bandwidth / N.** With N=1000, this is catastrophic.

### Crossbars: Full Non-Blocking Connectivity

A **crossbar switch** solves contention by providing a dedicated wire for every (source, destination) pair. With $N$ masters and $M$ slaves, a crossbar needs $N \times M$ switch points. Bandwidth scales linearly with port count, but area scales as $O(N^2)$. Feasible for small N (e.g., connecting 4–8 memory controllers), impractical for N=thousands.

Modern GPUs use crossbars internally to connect their L2 cache partitions to memory controllers. On the NVIDIA H100 there are 80 L2 cache slices connected via a high-bandwidth crossbar to 5 HBM3 stacks through 10 memory controllers — aggregate L2 bandwidth is ~15 TB/s, well above the 3.35 TB/s that leaves the chip.

### Network-on-Chip (NoC)

For larger chips (hundreds to thousands of functional blocks), the dominant approach is the **Network-on-Chip (NoC)**: a packet-switched network embedded in the chip's metal layers. Data is broken into flits (flow control digits), routed hop-by-hop through routers embedded in the chip, and reassembled at the destination.

<div class="diagram">
<div class="diagram-title">Common NoC Topologies</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Ring</div>
    <div class="card-desc">Each node connects to two neighbors. Simple wiring. Used in many CPU L3 caches (Intel). Latency scales O(N/2). Low bandwidth for large N.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Mesh / Torus</div>
    <div class="card-desc">2D grid. Each node has 4 neighbors (mesh) or wrap-around (torus). Diameter = O(√N). Used in many TPUs and large SoCs. Good bandwidth scalability.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Crossbar / Fat-Tree</div>
    <div class="card-desc">Full or hierarchical crossbar. Non-blocking. High bandwidth, high area. Used for high-performance memory interconnects and multi-chiplet fabrics.</div>
  </div>
</div>
</div>

**AMBA AXI** (Advanced eXtensible Interface) is the dominant on-chip bus *protocol* standard for SoC design (ARM-defined, universally licensed). AXI defines separate read and write channels, burst transactions, out-of-order responses, and QoS signaling. It is the "language" that blocks inside a chip — CPU core, DMA engine, GPU shader, ISP — use to talk to each other and to the memory subsystem. A typical SoC has dozens of AXI masters connected through an AXI interconnect fabric (often a hierarchical crossbar or NoC).

---

## Off-Chip Interconnects: Getting Data to the Board and Beyond

### PCIe — The Universal Connector

**PCIe (Peripheral Component Interconnect Express)** is the dominant off-chip high-speed serial interface. A PCIe link consists of one or more **lanes**, each a pair of differential signal lines (TX + RX). Data rates have roughly doubled each generation:

| PCIe Gen | Per-lane rate | x16 bandwidth (each direction) | Encoding | Spec year |
|:---------:|:-------------:|:-----------------:|:-----------------:|:----:|
| Gen 3.0 | 8 GT/s | ~16 GB/s each dir | 128b/130b | 2010 |
| Gen 4.0 | 16 GT/s | ~32 GB/s each dir | 128b/130b | 2017 |
| Gen 5.0 | 32 GT/s | ~64 GB/s each dir | 128b/130b | 2019 |
| Gen 6.0 | 64 GT/s | ~128 GB/s each dir | PAM4 + FLIT | 2022 |
| Gen 7.0 | 128 GT/s | ~256 GB/s | PAM4 + FLIT | 2025 |
| Gen 8.0 (draft) | 256 GT/s | ~512 GB/s | under definition | 2028 (target) |

Servers shipping in 2025–26 use PCIe 5.0 (H100, MI300X) or PCIe 6.0 (B300, Rubin); PCIe 7.0 hardware is expected from about 2027.

A GPU in a server is connected to the CPU host via PCIe — typically x16 (16 lanes). For a PCIe 4.0 x16 connection: ~32 GB/s in each direction. This is the bandwidth budget for copying model weights from CPU RAM to GPU memory, or for exchanging activations between a CPU host and GPU accelerator.

> **Why PCIe matters for ML:** When you call `tensor.cuda()`, data moves over PCIe. A 7B model in BF16 = 14 GB. At PCIe 4.0 x16 speed (32 GB/s), that takes ~0.44 seconds — noticeable at model load time. During inference, if data must return to CPU (e.g., CPU-GPU split for memory offloading like llama.cpp), you pay this bandwidth tax per layer.

### CXL — Cache-Coherent Memory Expansion

**CXL (Compute Express Link)** is a newer standard built on the PCIe physical layer (same wires, same connectors in CXL 1.1/2.0) but adds **cache-coherent protocols** on top. Where PCIe is a simple point-to-point, non-coherent DMA protocol, CXL defines three sub-protocols:

- **CXL.io** — standard PCIe-like I/O (device enumeration, DMA)
- **CXL.cache** — lets the device (e.g., an accelerator) coherently cache host memory
- **CXL.mem** — lets the host access device memory as if it were part of the host's own memory map

The transformative use case for ML is **CXL memory expansion**: attach hundreds of gigabytes of additional DRAM via CXL, accessible at latency ~200–300 ns (versus ~100 ns for local DRAM, versus ~1,000+ ns for PCIe DMA). This enables "CPU + CXL memory pool" to host large models that don't fit in GPU HBM, without the overhead of explicit DMA transfers. See [Chapter 35 — Where the Industry Is Headed](./35_where_the_industry_is_headed.md) for how CXL is reshaping data-center memory architectures.

Where CXL stands in late 2026: CXL 2.0 memory expanders (Samsung CMM-D, Micron CZ120 and others) work with Intel Xeon 6 and AMD EPYC Turin hosts; CXL 3.x adds switched fabrics and shared memory pools over PCIe 6.0; and **CXL 4.0** (November 2025) doubles the link rate to 128 GT/s on PCIe 7.0 and adds bundled ports for multi-rack pooling.

### NVLink and NVSwitch — GPU-to-GPU at Full Speed

PCIe is general-purpose. For GPU-to-GPU communication, NVIDIA built **NVLink**: a proprietary, high-bandwidth, low-latency interconnect designed specifically to connect GPUs (and GPUs to CPUs).

| NVLink Gen | Per-link bandwidth (bidirectional) | Links per GPU | Total GPU-GPU BW | Introduced |
|:----------:|:----------------------------------:|:-------------:|:----------------:|:----------:|
| NVLink 1 | 40 GB/s | 4 | 160 GB/s | P100 (2016) |
| NVLink 2 | 50 GB/s | 6 | 300 GB/s | V100 (2017) |
| NVLink 3 | 50 GB/s | 12 | 600 GB/s | A100 (2020) |
| NVLink 4 | 50 GB/s | 18 | ~900 GB/s | H100 (2022) |
| NVLink 5 | 100 GB/s | 18 | ~1,800 GB/s | B200 (2024) |
| NVLink 6 | 100 GB/s | 36 | ~3,600 GB/s | Rubin (2026) |

Compare NVLink 4 (~900 GB/s, both directions) with PCIe 5.0 x16 (~128 GB/s, both directions): **NVLink is ~7× faster** between GPUs, ~14× against PCIe 4.0. This is not an incremental improvement — it changes what is feasible.

**NVSwitch** scales NVLink beyond point-to-point. An NVSwitch chip is a crossbar that interconnects multiple NVLink ports. In the DGX H100:
- 8 H100 GPUs, each with 18 NVLink 4 ports
- 4 NVSwitch 3 chips, each with 64 NVLink 4 ports
- Any GPU can reach any other GPU at full ~900 GB/s bidirectional bandwidth
- Aggregate GPU bandwidth inside the node: 8 × 900 GB/s = **7.2 TB/s**

This is the **NVLink domain** (or island): a group of GPUs connected by NVLink/NVSwitch with near-uniform, extremely high bandwidth. An HGX/DGX H100 or B200 node is one 8-GPU domain. Since 2024 NVIDIA also builds **rack-scale** domains: a **GB200/GB300 NVL72** rack joins 72 GPUs through nine NVSwitch trays (18 switch chips) at 1.8 TB/s per GPU (~130 TB/s in total), and the **Vera Rubin NVL72** (2026) doubles the per-GPU link to 3.6 TB/s (~260 TB/s per rack). Beyond the rack, traffic still crosses PCIe to a NIC and the external network.

### AMD Infinity Fabric

AMD's equivalent to NVLink is **Infinity Fabric**, which serves as both the on-chip interconnect (connecting CPU cores to each other and to I/O dies in AMD's multi-chiplet CPUs) and the chip-to-chip interconnect for GPU-to-GPU and CPU-GPU communication (xGMI/XGMI protocol over Infinity Fabric links).

| Infinity Fabric | Bandwidth | Notes |
|:---------------:|:---------:|-------|
| On-chip (CPU die-to-die) | ~500+ GB/s | Epyc multi-die, connects CCDs |
| MI300X GPU HBM internal | ~5.2 TB/s | HBM3 to GPU compute |
| GPU-to-GPU (xGMI, MI300X) | ~896 GB/s | 8-GPU NPS config |

AMD's **MI300A** (2023, the APU in the El Capitan supercomputer) goes further: CPU cores and GPU compute chiplets share one 128 GB HBM3 pool through Infinity Fabric, a different architectural answer to the memory wall. (Its sibling MI300X is GPU-only.) For rack scale, AMD's **Helios** (2026) connects 72 MI455X GPUs with UALink-protocol traffic.

### UALink and NVLink Fusion — The Emerging Layer

As of 2024–2025, two new initiatives aim to standardize or extend GPU-to-GPU connectivity:

**UALink (Ultra Accelerator Link)**: an open consortium standard (founded by AMD, Intel, Google, Meta, Microsoft, Cisco, HPE and others in 2024; AWS joined later) for scale-up accelerator links. UALink 1.0 (April 2025) runs at 200 GT/s per lane over Ethernet-class SerDes, not the PCIe PHY, with a lightweight memory-semantic protocol, and scales to 1,024 accelerators per pod. First switch silicon is due in late 2026; AMD's MI400 series is the first major accelerator to adopt it. Ethernet itself is also being tuned for scale-up (Broadcom's Tomahawk Ultra and the OCP ESUN effort).

**NVLink Fusion** (announced May 2025): NVIDIA licenses NVLink chiplets and IP so that **non-NVIDIA chips** (third-party CPUs and custom accelerators) can join an NVLink fabric beside NVIDIA GPUs. Partners include MediaTek, Marvell, Alchip, Astera Labs, Fujitsu and Qualcomm, and AWS has said Trainium4 will support it. It is NVIDIA's answer to UALink: openness on NVIDIA's terms.

---

## Scale-Out Networking: InfiniBand vs Ethernet/RoCE

Within an NVLink island (one node), GPU-to-GPU bandwidth is ~1 TB/s. Between nodes in a cluster, you rely on the external network. Two technologies dominate:

**InfiniBand (IB):**
- Purpose-built for HPC and AI clusters
- Ultra-low latency: ~1–2 μs end-to-end
- High bandwidth: HDR = 200 Gb/s, NDR = 400 Gb/s, XDR = 800 Gb/s per port (Quantum-X800, 2025)
- Native RDMA (Remote Direct Memory Access) — GPU can DMA directly to/from remote GPU memory without CPU involvement
- Expensive; requires Mellanox/NVIDIA switch fabric (NVIDIA acquired Mellanox in 2020)
- Used in: NVIDIA DGX SuperPOD, most large AI training clusters

**RoCE (RDMA over Converged Ethernet):**
- RDMA semantics (same low CPU overhead) but runs over standard Ethernet switching
- Requires DCB (Data Center Bridging) for losslessness, or sophisticated ECN/PFC
- Latency: ~2–5 μs (higher than IB), improving with 400G/800G Ethernet
- Lower cost; reuses Ethernet infrastructure
- Used in: Meta's training clusters (alongside IB), Microsoft Azure, xAI's Colossus (NVIDIA Spectrum-X Ethernet); the **Ultra Ethernet Consortium** published its 1.0 specification in 2025 to make Ethernet a first-class AI transport

| Metric | InfiniBand NDR | 400G RoCE |
|--------|:--------------:|:---------:|
| Per-port bandwidth | 400 Gb/s (~50 GB/s) | 400 Gb/s (~50 GB/s) |
| Latency | ~1–2 μs | ~2–5 μs |
| Cost | High (proprietary) | Lower (commodity) |
| RDMA native | Yes | Yes (with DCB) |
| GPU Direct support | Full | Full (NCCL) |

At 400 Gb/s (50 GB/s per direction) per port, and typically one 400G NIC per GPU, inter-node bandwidth is ~9× less than NVLink 4's 450 GB/s per direction per GPU (18× less than NVLink 5). This asymmetry — fast within a node, slower between nodes — defines the "scale-up vs scale-out" distinction and shapes how models are parallelized.

---

## Topology Deep Dive

How you wire together N nodes determines the aggregate bandwidth, the latency under load, and the cost.

<div class="diagram">
<div class="diagram-title">Cluster Network Topologies</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Fat-Tree</div>
    <div class="card-desc">Hierarchical: edge → aggregation → core switches. Full bisection bandwidth if over-subscribed 1:1. Standard for InfiniBand clusters (NVIDIA DGX SuperPOD). Expensive but predictable.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Torus (3D / 6D)</div>
    <div class="card-desc">Each node connects to 2k neighbors (k dimensions). Low-diameter, good bisection bandwidth. Google TPU pods use a 3D torus. Routing is simpler in regular grids (all-reduce maps well).</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Dragonfly</div>
    <div class="card-desc">Groups of nodes fully connected internally; sparse all-to-all between groups. Very high bandwidth at lower cost than fat-tree. Used in Frontier and Aurora (HPC).</div>
  </div>
</div>
</div>

```text
Fat-Tree topology (2-level, 4 nodes per leaf):

      [Core Switch 1]  [Core Switch 2]
           |    \      /    |
           |     \    /     |
      [Agg-1]  [Agg-2]  [Agg-3]  [Agg-4]
       / \         / \
     L1  L2      L3  L4          ← leaf switches
    /|   |\     /|   |\
  G0 G1 G2 G3  G4 G5 G6 G7      ← GPU nodes

Any Gi can reach any Gj through ≤ 4 hops; full bisection at core level.
```

---

## SerDes and Signaling Basics

All high-speed links (PCIe, NVLink, InfiniBand, CXL) use **SerDes (Serializer/Deserializer)** circuits. The raw compute data inside a chip flows on many parallel wide buses (e.g., 256-bit AXI). Sending 256 parallel wires off-chip over centimeters or meters is impractical (signal integrity, crosstalk, pin count). Instead:

1. **Serialize**: convert 256-bit parallel to a single high-frequency serial stream (e.g., 32 GT/s × 8b = ~256 Gb/s on one wire pair)
2. **Encode**: add DC balance, clock embedding, error detection (8b/10b, 128b/130b, or PAM4 for PCIe 6)
3. **Equalize**: compensate for frequency-dependent signal degradation over the physical medium (copper, fiber)
4. **Deserialize**: at the far end, recover clock and data, convert back to parallel

**PAM4 (Pulse Amplitude Modulation, 4 levels)** is used in PCIe Gen 6 and 800G Ethernet: each symbol encodes 2 bits (four voltage levels: −1, −1/3, +1/3, +1) instead of NRZ's 1 bit. This doubles data rate without doubling symbol frequency, at the cost of tighter noise margins and more complex analog circuitry. The next step, ~224 Gb/s per lane PAM4, underlies NVLink 5/6, UALink and 1.6 Tb/s Ethernet.

---

## Interconnect Comparison Table

| Interconnect | Per-link BW (bidirectional) | Typical Use | Scope | Protocol | Latency |
|:------------:|:---------------------------:|:-----------:|:-----:|:--------:|:-------:|
| AXI on-chip | 100s GB/s to TB/s | Block-to-block in SoC | Intra-die | ARM AMBA | <1 ns |
| PCIe 4.0 x16 | 32 GB/s | GPU↔CPU, NIC, storage | Intra-node | PCIe 4.0 | ~500 ns |
| PCIe 5.0 x16 | 64 GB/s | GPU↔CPU (modern servers) | Intra-node | PCIe 5.0 | ~500 ns |
| PCIe 6.0 x16 | 128 GB/s | Next-gen servers | Intra-node | PCIe 6.0 | ~500 ns |
| PCIe 7.0 x16 | 512 GB/s | Servers from ~2027 | Intra-node | PCIe 7.0 | ~500 ns |
| CXL 2.0 | 32–64 GB/s | Memory expansion, coherent | Intra-node | CXL/PCIe | ~200–300 ns |
| NVLink 4 (H100) | ~900 GB/s (all 18 links) | GPU↔GPU direct | Intra-node | Proprietary | ~1–2 μs |
| NVLink 5 (B200) | ~1,800 GB/s (all 18 links) | GPU↔GPU direct | Intra-node | Proprietary | ~1–2 μs |
| NVLink 6 (Rubin) | ~3,600 GB/s | GPU↔GPU across an NVL72 rack | Rack | Proprietary | ~1–2 μs |
| AMD xGMI (MI300X) | ~896 GB/s | GPU↔GPU, GPU↔CPU | Intra-node | Infinity Fabric | ~1–2 μs |
| UALink 1.0 | ~100 GB/s per x4 port per direction | Open accelerator scale-up | Rack | Open std | ~1 μs (target) |
| InfiniBand NDR | ~50 GB/s/port | Cluster scale-out | Inter-node | IB/RDMA | ~1–2 μs |
| 400G RoCE | ~50 GB/s/port | Cluster scale-out | Inter-node | RDMA/ETH | ~2–5 μs |
| 800G Ethernet | ~100 GB/s/port | Next-gen clusters | Inter-node | RDMA/ETH | ~2–5 μs |
| InfiniBand XDR / 800G Ethernet | ~100 GB/s per port per direction | Current clusters | Inter-node | IB or RDMA/ETH | ~1–5 μs |

---

## Python: All-Reduce Communication Time

All-reduce is the fundamental collective operation for data-parallel training: each GPU has a gradient tensor, and all GPUs end up with the sum. With the **ring all-reduce** algorithm, the total data each GPU sends/receives is $2 \times (N-1)/N \times \text{data\_size} \approx 2 \times \text{data\_size}$ for large $N$.

```python
from dataclasses import dataclass

@dataclass
class AllReduceResult:
    model_params: int
    dtype_bytes: int
    num_gpus: int
    link_bw_gbs: float        # GB/s per GPU (bidirectional, effective)
    data_gb: float            # gradient tensor size in GB
    algo_bw_gb: float         # bytes each GPU sends/receives
    time_ms: float            # estimated time (ms)
    bottleneck: str

def ring_allreduce_time(
    model_params: int,
    num_gpus: int,
    link_bw_gbs: float,       # GB/s effective per GPU on the ring
    dtype_bytes: int = 4,     # 4 = FP32 gradients, 2 = BF16/FP16
    overlap_factor: float = 0.9,  # fraction of comm. that overlaps compute
) -> AllReduceResult:
    """
    Estimate ring all-reduce time for a given model + topology.

    Ring all-reduce sends 2*(N-1)/N * data_size bytes per GPU.
    Time = data_sent / link_bandwidth (if not overlapped).

    Args:
        model_params: number of model parameters
        num_gpus: data-parallel degree
        link_bw_gbs: per-GPU effective ring bandwidth (GB/s)
        dtype_bytes: bytes per gradient element
        overlap_factor: how much of comm. overlaps with backward pass (0–1)
    """
    data_bytes = model_params * dtype_bytes
    data_gb = data_bytes / 1e9

    # Ring all-reduce: each GPU sends 2*(N-1)/N * data_size
    # For large N this converges to 2 * data_size
    ring_factor = 2 * (num_gpus - 1) / num_gpus
    algo_bytes = ring_factor * data_bytes
    algo_gb = algo_bytes / 1e9

    time_s = algo_bytes / (link_bw_gbs * 1e9)
    # Overlap: only the exposed (non-overlapping) fraction adds to critical path
    exposed_time_s = time_s * (1 - overlap_factor)
    time_ms = exposed_time_s * 1e3

    if link_bw_gbs >= 100:
        bottleneck = "compute (link fast enough)"
    elif link_bw_gbs >= 10:
        bottleneck = "mixed (link is a significant fraction of step time)"
    else:
        bottleneck = "communication (link likely dominates)"

    return AllReduceResult(
        model_params=model_params,
        dtype_bytes=dtype_bytes,
        num_gpus=num_gpus,
        link_bw_gbs=link_bw_gbs,
        data_gb=data_gb,
        algo_bw_gb=algo_gb,
        time_ms=time_ms,
        bottleneck=bottleneck,
    )

def print_allreduce(r: AllReduceResult, label: str = ""):
    print(f"\n{'─'*60}")
    if label: print(f"  {label}")
    print(f"  Parameters:     {r.model_params/1e9:.1f}B  ({r.dtype_bytes}B/param = {r.data_gb:.1f} GB)")
    print(f"  GPUs:           {r.num_gpus}")
    print(f"  Link BW:        {r.link_bw_gbs:.0f} GB/s per GPU")
    print(f"  Data sent/GPU:  {r.algo_bw_gb:.1f} GB")
    print(f"  Exposed time:   {r.time_ms:.1f} ms  (90% overlap assumed)")
    print(f"  Bottleneck:     {r.bottleneck}")

# Scenario 1: 7B model, 8 GPUs, NVLink 4 intra-node (~450 GB/s effective per GPU on ring)
r1 = ring_allreduce_time(7e9, num_gpus=8, link_bw_gbs=450, dtype_bytes=2)
print_allreduce(r1, "7B BF16 grads, 8 GPUs, NVLink 4 intra-node (450 GB/s)")

# Scenario 2: 70B model, 64 GPUs (8 nodes x 8 GPUs), inter-node InfiniBand NDR (~50 GB/s)
r2 = ring_allreduce_time(70e9, num_gpus=64, link_bw_gbs=50, dtype_bytes=2)
print_allreduce(r2, "70B BF16 grads, 64 GPUs, InfiniBand NDR (50 GB/s)")

# Scenario 3: 7B model, 8 GPUs, PCIe only (~16 GB/s effective — old server)
r3 = ring_allreduce_time(7e9, num_gpus=8, link_bw_gbs=16, dtype_bytes=2)
print_allreduce(r3, "7B BF16 grads, 8 GPUs, PCIe 4.0 (16 GB/s) — no NVLink")

# Scenario 4: 70B model, 64 GPUs, 800G RoCE (~100 GB/s)
r4 = ring_allreduce_time(70e9, num_gpus=64, link_bw_gbs=100, dtype_bytes=2)
print_allreduce(r4, "70B BF16 grads, 64 GPUs, 800G RoCE (100 GB/s)")
```

**Output** (Python 3.12, abridged):

```text
  7B BF16 grads, 8 GPUs, NVLink 4 intra-node (450 GB/s)
  Data sent/GPU:  24.5 GB
  Exposed time:   5.4 ms  (90% overlap assumed)

  70B BF16 grads, 64 GPUs, InfiniBand NDR (50 GB/s)
  Data sent/GPU:  275.6 GB
  Exposed time:   551.2 ms  (90% overlap assumed)

  7B BF16 grads, 8 GPUs, PCIe 4.0 (16 GB/s) — no NVLink
  Data sent/GPU:  24.5 GB
  Exposed time:   153.1 ms  (90% overlap assumed)

  70B BF16 grads, 64 GPUs, 800G RoCE (100 GB/s)
  Exposed time:   275.6 ms  (90% overlap assumed)
```

The NVLink vs PCIe comparison is stark: the same all-reduce is ~28× faster over NVLink, the ratio of the two link bandwidths. That gap, more than raw FLOPs, is why multi-GPU training gravitated to NVLink-connected systems.

---

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>A 72-GPU NVLink rack turns what used to be "scale-out" traffic into "scale-up" traffic. Which parallelism dimension benefits most when the fast domain grows from 8 to 72 GPUs: tensor, expert or pipeline? (Hint: which one sends the most bytes per step, and to how many peers?)</p>
</div>

## Scale-Up vs Scale-Out

The interconnect landscape splits into two regimes with very different characteristics:

<div class="diagram">
<div class="diagram-title">Scale-Up vs Scale-Out</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Scale-Up (intra-node)</div>
    <ul>
      <li>Technology: NVLink / Infinity Fabric / UALink</li>
      <li>Bandwidth: 0.9–3.6 TB/s per GPU</li>
      <li>Latency: 1–2 μs</li>
      <li>Supports: tensor parallelism, expert parallelism</li>
      <li>Limit: 8 GPUs (HGX node) to 72 (NVL72 rack); larger domains announced for Rubin Ultra</li>
      <li>Cost: premium, tightly coupled hardware</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Scale-Out (inter-node)</div>
    <ul>
      <li>Technology: InfiniBand / RoCE / Ethernet</li>
      <li>Bandwidth: 50–200 GB/s per port</li>
      <li>Latency: 1–10 μs</li>
      <li>Supports: data parallelism, pipeline parallelism</li>
      <li>Limit: thousands of nodes (fat-tree, dragonfly)</li>
      <li>Cost: commodity switches, but traffic engineering needed</li>
    </ul>
  </div>
</div>
</div>

This asymmetry directly drives how models are split across hardware. **Tensor parallelism** (splitting a single matrix multiply across multiple GPUs) requires two all-reduces of the activations in every transformer layer, dozens per forward pass. At NVLink 4 speeds this cost is negligible; at PCIe or inter-node InfiniBand it would dominate. Therefore:

- **Tensor parallelism (TP)**: must stay within an NVLink island (typically 8 GPUs)
- **Pipeline parallelism (PP)**: sends activations (not gradients) between stages; much smaller data → can tolerate inter-node links
- **Data parallelism (DP)**: gradient all-reduce; can use inter-node IB/RoCE with sufficient overlap
- **Expert parallelism (EP)**: all-to-all routing of tokens to experts; needs high bisection bandwidth within node

This is why Meta trained Llama 3 405B on 16,384 H100s with TP = 8 inside each NVLink node, PP = 16 across nodes and DP = 128 across replicas (plus context parallelism for its long-context phase): a "4D parallelism" configuration that matches the bandwidth hierarchy of the hardware. Rack-scale NVL72 domains let the TP or EP group grow beyond 8 GPUs. See [Chapter 28 — Multi-GPU & Distributed Training](./28_multi_gpu_and_distributed_training.md).

---

## Why This Matters for Model Optimization

<div class="diagram">
<div class="diagram-title">Interconnect → Parallelism Strategy</div>
<div class="flow">
  <div class="flow-node accent wide">How fast is your GPU-GPU link?</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">NVLink domain: tensor parallelism is viable → large TP degree (8–16)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">Cross-node: pipeline or data parallelism → minimize inter-node all-reduces</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Cluster fabric: fat-tree → all-to-all for EP; torus → ring all-reduce for DP</div>
</div>
</div>

Three rules of thumb:

1. **NVLink gates tensor parallelism.** Without NVLink-class bandwidth, TP > 1 costs more than it saves. If you're serving a 70B model on 8×A10 (PCIe-only servers), data parallelism with pipeline parallelism is likely better than tensor parallelism.

2. **Inter-node bandwidth gates gradient overlap.** If your step time is 500 ms and your exposed all-reduce is 550 ms (as in the 70B / InfiniBand example above), you have a communication bottleneck. Solutions: larger batch (fewer steps per second), gradient compression (FP8 or PowerSGD), or overlapping with backward using bucketized all-reduce (torch.DDP default behavior).

3. **PCIe is a hard ceiling for offloaded inference.** If weights stream from host RAM on every forward pass, a 7B BF16 model (14 GB) can run at most 32 / 14 ≈ 2.3 forward passes per second over PCIe 4.0 x16 (≈ 4.6 over PCIe 5.0), orders of magnitude below HBM-resident decode. Offloading only works when each streamed byte is reused across a large batch. (DeepSpeed ZeRO-Infinity's NVMe tier is limited further by SSD bandwidth.)

---

## Key Takeaways

- On-chip interconnects (AXI/NoC/crossbar) operate at TB/s and nanosecond latency; off-chip links step down dramatically in each dimension.
- **PCIe** is the universal GPU↔CPU link (~32–256 GB/s per direction from Gen 4 to Gen 7); critical for data loading and offloading but a bottleneck for GPU-GPU traffic.
- **NVLink/NVSwitch** provides 0.9–3.6 TB/s per GPU, ~7× PCIe 5.0, across 8-GPU nodes or 72-GPU racks, enabling tensor and expert parallelism that would be impractical otherwise.
- **CXL** adds cache-coherent memory expansion over PCIe PHY; key for cost-effective memory pooling in large-scale inference deployments.
- **UALink** (open, Ethernet-class SerDes) and **NVLink Fusion** (NVIDIA-licensed) are the two routes to NVLink-class scale-up for non-NVIDIA silicon; first UALink switches arrive in late 2026.
- **InfiniBand and RoCE** provide ~50–100 GB/s inter-node bandwidth for cluster scale-out; latency is 1–5 μs versus the ~1 μs of NVLink.
- The **scale-up / scale-out asymmetry** (TB/s within a node, tens of GB/s between nodes) is the hardware reason for 3D parallelism: TP within NVLink islands, PP/DP across nodes.

## Check Yourself

<details class="check">
<summary>How long does copying 14 GB of BF16 weights from host to GPU take over PCIe 5.0 x16, and over a GB200's NVLink-C2C?</summary>
<p>PCIe 5.0 x16 carries ~64 GB/s per direction: ~0.22 s. NVLink-C2C carries ~450 GB/s per direction (900 GB/s total): ~31 ms, about 7× faster.</p>
</details>

<details class="check">
<summary>Why does tensor parallelism stay inside an NVLink domain while data parallelism can cross the cluster network?</summary>
<p>Tensor parallelism all-reduces activations twice per layer, dozens of times per forward pass, on the critical path. Data parallelism all-reduces gradients once per step and can overlap them with the backward pass, so it tolerates the 9–18× lower bandwidth between nodes.</p>
</details>

<details class="check">
<summary>What does CXL add on top of PCIe, and why does it matter for inference servers?</summary>
<p>CXL reuses the PCIe physical layer but adds coherent protocols: CXL.mem lets the host use device-attached DRAM as ordinary memory, CXL.cache lets devices cache host memory coherently. Servers can add or pool hundreds of GB of DRAM (at ~200–300 ns) for KV caches, embeddings or offloaded weights without explicit DMA.</p>
</details>

<details class="check">
<summary>What is the difference between UALink and NVLink Fusion?</summary>
<p>UALink is an open standard anyone can implement, built on Ethernet-class SerDes with its own lightweight protocol. NVLink Fusion licenses NVIDIA's proprietary NVLink so third-party CPUs and accelerators can join NVIDIA's fabric. Both target rack-scale, memory-semantic accelerator interconnects.</p>
</details>

<details class="check">
<summary>Why does PAM4 double the data rate without doubling the symbol rate, and what does it cost?</summary>
<p>Each symbol carries one of four voltage levels, i.e. two bits instead of NRZ's one. The levels are only a third as far apart, so noise margins shrink; links need stronger equalisation and forward error correction (hence PCIe 6.0's FLIT mode).</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| PCI Express Base Specification (Rev. 6.0, 7.0) | PCI-SIG, 2022 / 2025 | [pcisig.com/specifications](https://pcisig.com/specifications) | Per-lane rates, PAM4 and FLIT mode |
| Compute Express Link Specification 4.0 | CXL Consortium, 2025 | [computeexpresslink.org](https://computeexpresslink.org/) | CXL.io/.cache/.mem, switching and CXL 4.0 rates |
| UALink 200G 1.0 Specification | UALink Consortium, 2025 | [ualinkconsortium.org](https://ualinkconsortium.org/) | Open scale-up interconnect: 200 GT/s lanes, 1,024-accelerator pods |
| NVIDIA NVLink and NVLink Switch | NVIDIA | [nvidia.com/en-us/data-center/nvlink](https://www.nvidia.com/en-us/data-center/nvlink/) | NVLink generations, NVL72 domains, NVLink Fusion |
| The Llama 3 Herd of Models | Llama Team, Meta, 2024 | [arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783) | 16K-GPU 4D parallelism layout (TP 8, PP 16, DP 128) |
| Ultra Ethernet Specification 1.0 | Ultra Ethernet Consortium, 2025 | [ultraethernet.org](https://ultraethernet.org/) | Ethernet transport designed for AI and HPC |

---

**Next:** [Chapter 13 — Reliability, Fault Tolerance & Robustness](./13_reliability_and_fault_tolerance.md) — We look at what happens when hardware fails — and why month-long training runs demand ECC, checkpointing, and silent-corruption screening.

[← Back to Table of Contents](./README.md)
