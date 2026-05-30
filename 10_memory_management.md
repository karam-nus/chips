---
title: "Chapter 10 — Memory Management"
---

[← Back to Table of Contents](./README.md)

# Chapter 10 — Memory Management

[Chapter 9](./09_memory_types.md) catalogued the hardware: SRAM, DRAM, HBM, the hierarchy of speed and capacity. But having fast memory is only half the problem. The other half is *managing it* — making sure the right data is in the right level at the right time, at the right address, visible to the right processor. If you've ever wondered why `tensor.contiguous()` matters, why `pin_memory=True` in a DataLoader speeds things up, why `NCHW` and `NHWC` have different performance, or why NUMA topology affects multi-GPU training even though you haven't written a single line of C — this chapter has the answers.

Memory management is the software/hardware interface that makes the memory hierarchy invisible to programs while ensuring it performs well. It operates at several layers simultaneously: **virtual memory** (translating addresses, protecting processes from each other), **caching** (transparently migrating hot data toward the CPU), and **coherence** (ensuring multiple cores and GPUs see a consistent view of shared state). Each layer has failure modes that appear as mysterious slowdowns in ML workloads.

> **The one-sentence version:** Memory management is the full hardware+OS stack that translates program addresses to physical locations, caches hot data automatically, and maintains consistency across cores — and its failure modes (TLB misses, cache thrash, coherence traffic, NUMA penalties) are real performance hazards in large-model training and inference.

---

## Why We Need Memory Management

Consider a server with 4 GPUs, 96 CPU cores, 1.5 TB of DDR5, and a 70B-parameter model. Several problems arise immediately without memory management:

1. **Addresses would conflict.** With multiple processes and the OS running simultaneously, two programs might try to use the same physical address. A bug in one program would corrupt another.
2. **Programs cannot always fit in physical memory.** A model larger than VRAM must be partitioned or swapped.
3. **Without caching, every memory access would pay full DRAM latency (~70 ns)** — L1 cache at 1–4 ns is a 17–70× speedup.
4. **Multiple cores write the same data.** Without coherence, Core 0 might read a stale value after Core 1 updated it.
5. **Memory on different NUMA nodes has different latency.** Ignoring this causes 10–40% throughput losses in multi-socket servers.

Each of these problems has a hardware/OS solution. We cover them in order.

---

## Virtual Memory

### The Problem: Physical Address Chaos

Early computers ran one program at a time; the program controlled all physical memory. Modern systems run hundreds of processes simultaneously. Each process believes it owns the entire address space — in 64-bit systems, a 48-bit (256 TB) virtual address space.

**Virtual memory** is the abstraction that makes this possible. Each process has its own **virtual address space**: addresses 0 through $2^{48}-1$. The hardware + OS **transparently maps** these virtual addresses to **physical addresses** (the actual DRAM locations) at runtime.

<div class="diagram">
<div class="diagram-title">Virtual → Physical Address Translation</div>
<div class="flow-h">
  <div class="flow-node accent">Program uses<br/><small>virtual address 0x7fff_1234</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">TLB lookup<br/><small>(fast path: ~1 cycle)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">Page table walk<br/><small>(slow path: ~100–300 cycles)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">Physical address<br/><small>0x3a80_0234</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange">DRAM / cache<br/><small>actual data fetch</small></div>
</div>
</div>

### Paging and Page Tables

The virtual address space is divided into fixed-size **pages** (typically **4 KB**, but also 2 MB or 1 GB — "huge pages"). Physical memory is divided into **page frames** of the same size. A **page table** (a data structure in DRAM, maintained by the OS) maps each virtual page number to a physical frame number.

For a 48-bit virtual address with 4 KB pages, the layout is:

```text
Virtual Address (48-bit):
┌─────────────────────────────────────────────────────┐
│  [47:39]  │  [38:30]  │  [29:21]  │  [20:12]  │  [11:0]  │
│  PGD idx  │  PUD idx  │  PMD idx  │  PT idx   │  offset  │
│  (9 bits) │  (9 bits) │  (9 bits) │  (9 bits) │ (12 bits)│
└─────────────────────────────────────────────────────┘
     L4 table    L3 table    L2 table    L1 table    offset in page
```

This is a **4-level page table** (x86-64 uses exactly this). The page walk traverses 4 levels of tables stored in physical memory, each at a different address — potentially 4 DRAM accesses, each ~70 ns, totaling ~280 ns just to translate one address. This would be catastrophic without caching.

### The MMU and TLB

The **Memory Management Unit (MMU)** is the hardware component inside the CPU/GPU that performs address translation. It contains:

- **TLB (Translation Lookaside Buffer):** A small, fully-associative cache of recent virtual→physical translations. On a TLB hit (the translation is cached), translation costs **1–3 cycles**. On a **TLB miss**, the MMU performs the full page-table walk — 4 memory accesses, ~100–300 cycles.

| TLB Level | Entries | Latency on hit | Coverage (4K pages) | Coverage (2MB pages) |
|-----------|:---:|:---:|:---:|:---:|
| L1 iTLB (instructions) | 64–256 | 1 cycle | 256 KB–1 MB | 128 MB–512 MB |
| L1 dTLB (data) | 32–64 | 1 cycle | 128–256 KB | 64 MB–128 MB |
| L2 STLB (unified) | 512–4096 | 7–12 cycles | 2 MB–16 MB | 1 GB–8 GB |

A modern CPU's L2 STLB with 4096 entries covers only **16 MB** of 4 KB pages. A 70B model in BF16 is 140 GB — accessing it with 4 KB pages would cause **millions of TLB misses** per second if not addressed (pun intended).

### Huge Pages to the Rescue

**Huge pages** (2 MB on x86; 1 GB with 1GB pages) map much larger regions with a single TLB entry. Instead of 140 GB / 4 KB = **36 million TLB entries**, 140 GB / 2 MB = **70,000 entries** — fitting comfortably in the L2 STLB.

PyTorch uses `mmap` with `MAP_HUGETLB` for large weight tensors when huge pages are available. Linux **Transparent Huge Pages (THP)** attempts to do this automatically. For large model inference servers, **explicitly enabling 1GB huge pages** for VRAM-sized allocations can reduce TLB pressure and improve throughput by 5–15%.

```bash
# Enable 1GB huge pages at boot (Linux)
# /etc/default/grub: add "hugepagesz=1G hugepages=128" to GRUB_CMDLINE_LINUX
# After reboot:
grep HugePages /proc/meminfo
# HugePages_Total:     128
# HugePages_Free:       64
```

---

## Caches in Depth

### Cache Lines

The CPU does not load a single byte from DRAM on a cache miss — it loads a **cache line**: a contiguous block of **64 bytes** (8 × `float64`, 16 × `float32`, 32 × `fp16/bf16`, 64 × `int8`). This exploits **spatial locality**: if you access address $A$, you'll likely soon access $A+4$, $A+8$, etc. Loading a full line amortizes the DRAM access cost over 64 bytes.

Consequence for ML: accessing a tensor row-by-row (column-major for a row-major layout) loads 64 bytes, uses 4–8 of them, then discards the rest — catastrophically wasteful. Accessing in **the tensor's storage order** (row-major for the default C-contiguous PyTorch layout) uses all 64 bytes before eviction.

### Associativity

A **cache** must decide where to store each cache line. Three main policies:

<div class="diagram">
<div class="diagram-title">Cache Associativity: Three Designs</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">1️⃣</div>
    <div class="card-title">Direct-Mapped</div>
    <div class="card-desc">Each memory block maps to exactly one cache slot (set). Simple, fast. High conflict-miss rate: two blocks at addresses 0 and N (N = cache size) evict each other repeatedly.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🔀</div>
    <div class="card-title">N-way Set-Associative</div>
    <div class="card-desc">Cache divided into sets of N ways. A block maps to one set but any way within it. Reduces conflict misses dramatically. Modern L1/L2/L3: 4–16 way. Most common design.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🌐</div>
    <div class="card-title">Fully Associative</div>
    <div class="card-desc">A block can go anywhere. Zero conflict misses. Expensive: requires comparing all tags simultaneously. Used for small, critical structures like TLBs and victim caches.</div>
  </div>
</div>
</div>

A modern L3 cache might be **16-way set associative** with 512 MB: 512 MB / 64 B line / 16 ways = **512K sets**, each holding 16 cache lines. The index field of an address selects one of the 512K sets; the tag field identifies which specific line is stored in each way.

### Replacement Policies

When all ways in a set are full and a new line must be loaded, the cache evicts one. Common policies:

- **LRU (Least Recently Used):** Evict the line accessed longest ago. Near-optimal for temporal locality. Expensive to implement exactly for large caches — approximated with pseudo-LRU trees.
- **PLRU (Pseudo-LRU):** Uses a binary tree to approximate LRU. Hardware-efficient; used in most L2/L3 caches.
- **RRIP (Re-Reference Interval Prediction):** Smarter; predicts how soon each line will be reused. Used in Intel's L3 caches.

For ML: if your model weights are larger than the L3 cache (almost always true), the cache behaves as a **streaming buffer** — each line is used once and evicted. LRU/PLRU makes no difference. What matters is **spatial locality** (cache-line fill efficiency) and **prefetching**.

### Write Policies

- **Write-through:** Every store writes to both the cache and DRAM immediately. Cache always matches DRAM; consistent, but every write incurs a DRAM write.
- **Write-back:** Stores update only the cache; the line is marked **dirty**. DRAM is updated only when the dirty line is evicted. Lower DRAM write traffic, but the OS/other devices may see stale data.
- **Write-allocate:** On a write miss, load the line into cache first, then write. Pairs with write-back; standard for CPUs.

L1/L2/L3 caches on modern CPUs are write-back with write-allocate. GPU L1/L2 are typically write-back as well.

### The Three C's of Cache Misses

Every cache miss falls into one of three categories:

<div class="diagram">
<div class="diagram-title">The Three C's of Cache Misses</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">1️⃣</div>
    <div class="card-title">Compulsory (Cold) Misses</div>
    <div class="card-desc">First access to any address — data was never in cache. Unavoidable; happens once per cache line. Prefetching hides these. Dominant during model weight loading.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">2️⃣</div>
    <div class="card-title">Capacity Misses</div>
    <div class="card-desc">The working set is larger than the cache — lines get evicted before reuse even with perfect placement. Solved only by a bigger cache. Dominant when weight matrix >> L3.</div>
  </div>
  <div class="diagram-card red">
    <div class="card-icon">3️⃣</div>
    <div class="card-title">Conflict Misses</div>
    <div class="card-desc">Two heavily-used lines compete for the same cache set — one evicts the other even though the cache has overall space. Solved by higher associativity. Rarer with modern 16-way caches.</div>
  </div>
</div>
</div>

### Spatial and Temporal Locality

- **Temporal locality:** A recently accessed datum will likely be accessed again soon. Caches exploit this by keeping recently-used lines.
- **Spatial locality:** If address $A$ is accessed, $A+4$, $A+8$, … are likely to be accessed soon. Cache lines (64 bytes) exploit this by loading a block.

```python
import numpy as np
import time

N = 4096  # 4096×4096 float32 matrix = 64 MB (>> typical L3 cache)
A = np.random.rand(N, N).astype(np.float32)  # row-major (C order)
B = np.random.rand(N, N).astype(np.float32)

# --- Cache-FRIENDLY: row-major sequential scan (spatial locality) ---
t0 = time.perf_counter()
total = 0.0
for i in range(N):
    total += A[i, :].sum()   # access full row: sequential in memory
t1 = time.perf_counter()
print(f"Row scan (friendly): {(t1-t0)*1e3:.1f} ms")

# --- Cache-UNFRIENDLY: column-major scan (stride = 4096 × 4 bytes = 16 KB) ---
t0 = time.perf_counter()
total = 0.0
for j in range(N):
    total += A[:, j].sum()   # access full column: stride 16 KB — one element per cache line
t1 = time.perf_counter()
print(f"Col scan (unfriendly): {(t1-t0)*1e3:.1f} ms")
# Expected: column scan is 3–10× slower depending on CPU cache size
```

### Cache-Friendly Matmul: Tiling / Blocking

Naïve matmul $C = AB$ with $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$:

```python
# SLOW: naive triple loop — accesses B column-by-column (poor locality for row-major B)
def matmul_naive(A, B):
    M, K = A.shape
    K2, N = B.shape
    C = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]   # B[k,j]: stride-N access
    return C
```

**Blocked (tiled) matmul** partitions the matrices into tiles that fit in L2/L3 cache, dramatically improving reuse:

```python
def matmul_blocked(A, B, block=64):
    """
    Block size 64: tile = 64×64 float32 = 16 KB — fits in L2 cache (256 KB).
    Each tile of A is reused N/block times; each tile of B is reused M/block times.
    """
    M, K = A.shape
    K2, N = B.shape
    C = np.zeros((M, N), dtype=A.dtype)
    for i0 in range(0, M, block):
        for j0 in range(0, N, block):
            for k0 in range(0, K, block):
                # These three slices are all in cache simultaneously
                Atile = A[i0:i0+block, k0:k0+block]   # block × block = 16 KB
                Btile = B[k0:k0+block, j0:j0+block]   # block × block = 16 KB
                C[i0:i0+block, j0:j0+block] += Atile @ Btile
    return C
```

This is conceptually what **cuBLAS, OneDNN, and Flash Attention** do — tile computations to maximize cache reuse and minimize DRAM traffic. On GPU, the "cache" is **shared memory** (32–64 KB per SM), and tiling is done explicitly in the CUDA kernel.

---

## Cache Coherence

### The Problem

A modern server might have **96 CPU cores**, each with its own L1 and L2 cache, all sharing the same physical DRAM. Suppose:

1. Core 0 loads address $X$ into its L1 cache (value = 5).
2. Core 1 loads address $X$ into its L1 cache (value = 5).
3. Core 0 writes $X = 7$ — updates its L1 (dirty line).
4. Core 1 reads $X$ — gets 5 from its own L1 (stale!).

Without coherence, parallel programs are **undefined**. Hardware must enforce: **if Core 1 reads $X$ after Core 0 wrote $X$, it gets the new value.**

### The MESI Protocol

**MESI** (Modified–Exclusive–Shared–Invalid) is the most widely used cache coherence protocol. Each cache line has a state:

<div class="diagram">
<div class="diagram-title">MESI Cache Line States</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-icon">✏️</div>
    <div class="card-title">M — Modified</div>
    <div class="card-desc">Line is in this cache only; dirty (differs from DRAM). DRAM copy is stale. Must write-back before another cache can read.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🔒</div>
    <div class="card-title">E — Exclusive</div>
    <div class="card-desc">Line is in this cache only; clean (matches DRAM). Can be promoted to M on a write without notifying others. Silent transition.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">📖</div>
    <div class="card-title">S — Shared</div>
    <div class="card-desc">Line is in multiple caches; clean. Any core can read. A write must first invalidate all other copies (RFO: Read For Ownership).</div>
  </div>
  <div class="diagram-card red">
    <div class="card-icon">🚫</div>
    <div class="card-title">I — Invalid</div>
    <div class="card-desc">Line is not valid in this cache. A read triggers a cache miss, initiating a fetch from DRAM or a modified cache line in another core.</div>
  </div>
</div>
</div>

### Snooping vs Directory Coherence

**Snooping:** Every cache monitors (snoops) a shared bus or interconnect. When a core writes, it broadcasts an **invalidate** message; every cache holding that line transitions to Invalid. Simple; used in small (<16 core) systems.

**Directory coherence:** A centralized (or distributed) **directory** tracks which caches hold each physical line. On a write, the directory sends invalidates only to the caches that have the line — much more scalable. Used in large multi-socket servers (EPYC, Xeon) with hundreds of cores.

### False Sharing

A pernicious coherence problem: two threads on different cores each modify *different* fields that happen to occupy the **same 64-byte cache line**. Every write from one core triggers an invalidate to the other, forcing a cache miss on the next access — despite no logical data sharing.

```python
import numpy as np

# Array where adjacent elements are in the same cache line (64 bytes = 16 float32s)
# Thread 0 writes index 0; Thread 1 writes index 1 → same cache line → false sharing
data = np.zeros(1024, dtype=np.float32)

# Mitigation: pad each "thread-private" element to a full cache line (64 bytes = 16 floats)
# data_padded[thread_id * 16] avoids false sharing between adjacent threads
```

In PyTorch: gradient accumulation buffers, counters, and locks can suffer from false sharing. Libraries like `torch.distributed` and `deepspeed` are carefully designed to avoid this.

---

## Memory Consistency Models

**Coherence** says "every core eventually sees a write to address $X$." **Consistency** says "in what order do writes to *different* addresses become visible?"

- **Sequential Consistency (SC):** The ideal model. All operations appear to happen in some global total order. Easy to reason about; expensive to implement.
- **Total Store Order (TSO):** x86's model. Stores are buffered (store buffer) before becoming globally visible; loads from the same core bypass the store buffer. Relaxes SC while remaining easy to reason about.
- **Weak/Release Consistency:** ARM, POWER, RISC-V. Stores and loads can be reordered aggressively by hardware for performance; memory barriers (`DMB`/`DSB` on ARM, `FENCE` on RISC-V) enforce ordering at specific points.

For ML: PyTorch's `torch.distributed` operations, CUDA's `__threadfence()` and `__threadfence_system()`, and Python's `threading.Lock()` all insert the appropriate barriers. You generally don't need to manage this manually — but understanding it explains why GPU kernels need `__syncthreads()` after shared memory writes, and why naïve `atomicAdd` accesses are slower on relaxed-consistency architectures.

---

## NUMA — Non-Uniform Memory Access

### The Problem with Multi-Socket Systems

A server with two CPU sockets (e.g., dual-socket AMD EPYC with 96 cores per socket) has two NUMA nodes. Each NUMA node has its own local DRAM controller and memory banks, connected by an inter-socket interconnect (AMD's Infinity Fabric, Intel's UPI).

| NUMA scenario | Latency | Bandwidth |
|---|:---:|:---:|
| Local DRAM access (same socket) | ~60–80 ns | ~280 GB/s (8-channel DDR5) |
| Remote DRAM access (other socket) | ~120–180 ns | ~140 GB/s (limited by inter-socket link) |
| PCIe device (GPU) DMA to host DRAM | ~200–400 ns | ~64 GB/s (PCIe 5.0 × 16) |

**Remote NUMA access is 1.5–3× slower** than local. If CPU processes that handle a specific GPU are scheduled on the non-local NUMA node, every data transfer from host to GPU pays the inter-socket overhead.

### NUMA in Multi-GPU Systems

In a DGX-H100 with 8 GPUs and 2 CPU sockets:

```text
NUMA Node 0 (Socket 0, 48 cores, 768 GB DDR5):
  ├── GPU 0, 1, 2, 3 (via PCIe 5.0 / NVLink)
  └── NVLink switch 0

NUMA Node 1 (Socket 1, 48 cores, 768 GB DDR5):
  ├── GPU 4, 5, 6, 7 (via PCIe 5.0 / NVLink)
  └── NVLink switch 1
```

For data-parallel training: the **DataLoader workers** pinning memory and copying to GPU should be **NUMA-local** to their GPU. Using `os.sched_setaffinity()` or `numactl --cpunodebind=0 --membind=0` pins processes to NUMA node 0 for GPU 0–3, avoiding remote DRAM.

```python
# PyTorch DataLoader: pin_memory=True allocates host tensors in pinned (page-locked)
# NUMA-local memory for fast DMA to the GPU.
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,
    pin_memory=True,          # allocates in pinned memory → DMA-friendly
    pin_memory_device="cuda:0",  # pins to NUMA node local to GPU 0 (PyTorch ≥2.1)
)
```

**Pinned memory** (page-locked, non-pageable) lets the GPU DMA engine copy directly from host RAM to GPU VRAM without a CPU intermediate copy — critical for avoiding PCIe transfer overhead when streaming training data.

---

## Tensor Layout, Contiguity, and Memory Management

### Why `tensor.contiguous()` Exists

PyTorch tensors have a **storage** (a flat 1D array in physical memory) and **metadata** (shape, strides, offset) that describe how the logical N-D tensor maps onto the 1D storage.

```python
import torch

x = torch.randn(3, 4, 5)  # shape [3,4,5], strides (20, 5, 1) — contiguous, row-major
y = x.transpose(0, 2)      # shape [5,4,3], strides (1, 5, 20) — same storage, different view
print(y.is_contiguous())   # False — stride order doesn't match dim order

z = y.contiguous()         # Allocates new storage, copies data in new stride order
print(z.is_contiguous())   # True
```

A non-contiguous tensor accesses memory with **non-unit strides** — skipping over memory locations. For the transposed `y` with stride `(1, 5, 20)`, iterating over the last dimension accesses elements 20 apart in memory — one element per cache line (at float32: 20 × 4 = 80 bytes > 64 byte line). Calling `.contiguous()` creates a new allocation with sequential access, making it cache-friendly.

### NCHW vs NHWC: Cache Implications

Convolutional layers operate on 4D tensors with shape `[batch, channels, height, width]`. Two common layouts:

<div class="diagram">
<div class="diagram-title">NCHW vs NHWC Memory Layout</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">NCHW (channels-first)</div>
    <ul>
      <li>Storage: all of channel 0, then channel 1, …</li>
      <li>Spatial neighbors are adjacent in memory</li>
      <li>Good for per-channel operations (BN stats)</li>
      <li>Default in PyTorch; friendly for SIMD on width dimension</li>
      <li>Poor locality when combining channels per pixel</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">NHWC (channels-last)</div>
    <ul>
      <li>Storage: all channels for pixel (0,0), then (0,1), …</li>
      <li>All channels for a pixel are adjacent</li>
      <li>Better for convolutions (accumulate over C at each spatial point)</li>
      <li>Preferred by TensorFlow/ONNX, GPU conv libraries (cuDNN)</li>
      <li>Often 10–30% faster on GPU for conv due to HBM access pattern</li>
    </ul>
  </div>
</div>
</div>

```python
import torch
import time

# Create NCHW and NHWC tensors
N, C, H, W = 32, 256, 64, 64
x_nchw = torch.randn(N, C, H, W, device='cpu')
x_nhwc = x_nchw.to(memory_format=torch.channels_last)  # PyTorch channels_last

# On GPU, cuDNN will auto-convert if it detects channels_last input
# Pass memory_format=torch.channels_last to model layers for best GPU conv perf:
conv = torch.nn.Conv2d(256, 256, 3, padding=1).to(memory_format=torch.channels_last)
```

---

## Why This Matters for Model Optimization

<div class="diagram">
<div class="diagram-title">Memory Management → ML Optimization Decisions</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Tensor Layout & Contiguity</div>
    <div class="card-desc">`tensor.contiguous()`, NCHW vs NHWC, and stride order directly determine cache-line utilization. Channels-last layout in PyTorch gives 10–30% faster GPU convolutions by improving HBM access patterns.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Tiling / Blocking in Kernels</div>
    <div class="card-desc">Flash Attention, cuBLAS, and Triton kernels all tile matmuls to fit tiles in GPU shared memory (SRAM). This minimizes HBM round-trips — the same insight as CPU cache blocking.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">TLB Pressure & Huge Pages</div>
    <div class="card-desc">Large model weights (140 GB for 70B BF16) with 4KB pages → 36M TLB entries. Enabling 2MB or 1GB huge pages reduces this to thousands of entries, eliminating TLB miss overhead.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Pinned Memory for Transfers</div>
    <div class="card-desc">`pin_memory=True` in DataLoader allocates page-locked memory, enabling zero-copy DMA from host to GPU. Skip it and every H2D transfer incurs an extra CPU memcpy — limiting throughput by up to 2×.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">NUMA-Aware Multi-GPU</div>
    <div class="card-desc">In multi-socket servers, CPU worker threads and GPU devices should be NUMA-local. Wrong NUMA affinity can increase host-to-GPU transfer latency by 2× and reduce DataLoader throughput significantly.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Cache Coherence & Multi-GPU</div>
    <div class="card-desc">Shared CPU-accessible memory buffers (e.g., NCCL staging areas, gradient buckets) undergo MESI protocol transitions as multiple CPU cores update them — motivating contiguous gradient buffers per process.</div>
  </div>
</div>
</div>

### The Full Picture: One Forward Pass

A single transformer forward pass on a multi-GPU node involves all the memory management concepts in this chapter:

```text
1. DataLoader workers (pinned memory, NUMA-local) → H2D DMA (virtual→physical, TLB walk)
2. Embedding lookup → L2/L3 cache misses (compulsory), DRAM/HBM access
3. Attention matmul → tiled kernel, tiles in GPU shared memory (SRAM), HBM for K/V/Q
4. FFN matmul → cache-friendly NHWC layout; weight tensor too large for L3 (capacity misses)
5. Gradient AllReduce → inter-GPU via NVLink; CPU gradient buffers under MESI coherence
6. Optimizer step → reads/writes parameter + optimizer state tensors; NUMA-aware pinning
7. Checkpoint write → DMA to pinned host memory → async copy to NVMe SSD
```

Each step is a negotiation with the memory hierarchy. Understanding these layers is why ML systems engineers spend so much time on memory layout, pinning, affinity, and tiling.

---

## Key Takeaways

- **Virtual memory** gives each process its own address space; the **MMU** translates virtual → physical addresses using page tables. The **TLB** caches recent translations (~1 cycle hit vs ~100–300 cycle miss on a page-table walk).
- **Huge pages** (2 MB, 1 GB) dramatically reduce TLB pressure for large allocations — important for model weights and KV caches exceeding tens of gigabytes.
- **Cache lines** are 64 bytes; accessing tensors in non-contiguous or column-major order wastes most of each cache line and slows performance 3–10× — the hardware basis for `tensor.contiguous()` and NHWC layout.
- **The three C's** — compulsory, capacity, and conflict misses — classify why a cache miss occurs; capacity misses dominate for model weights larger than L3.
- **Tiling/blocking** is the universal technique for fitting working sets into fast SRAM (CPU L2 or GPU shared memory) to maximize reuse and minimize DRAM traffic — the core idea behind cuBLAS, Triton, and Flash Attention.
- **MESI** cache coherence ensures multiple cores see a consistent memory view via Modified/Exclusive/Shared/Invalid states; false sharing (two threads modifying different fields in the same cache line) can silently destroy parallel performance.
- **NUMA** topology means that remote-socket DRAM accesses are 1.5–3× slower; GPU DataLoaders and DMA transfers should be **NUMA-local** to their target GPU. `pin_memory=True` enables zero-copy DMA and is almost always a free win.

---

*Next: [Chapter 11 — Memory Wall & Bandwidth](./11_memory_wall_and_bandwidth.md), where we quantify exactly how badly memory limits compute using the roofline model and arithmetic intensity — the tools that explain why almost every large-model inference workload is memory-bound.*

[← Back to Table of Contents](./README.md)
