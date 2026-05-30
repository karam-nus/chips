---
title: "Chapter 28 — Multi-GPU & Distributed Training"
---

[← Back to Table of Contents](./README.md)

# Chapter 28 — Multi-GPU & Distributed Training

[Chapter 27](./27_why_gpus_won_ai.md) ended on a promise: the same matmul-everywhere structure that fits one GPU also *shards* cleanly across many. This chapter cashes that in. It is the dedicated tour of why a single GPU is not enough for modern models, and of every way the field has invented to split a training job across tens, hundreds, or tens of thousands of GPUs.

The starting point is not a strategy — it's a constraint. An H100 has 80 GB of HBM ([Ch 9](./09_memory_types.md), [Ch 15](./15_gpus.md)). A 70-billion-parameter model, *just to hold the numbers needed to train it*, needs well over a terabyte. The arithmetic is unforgiving and it forces the entire field of distributed training: **you must shard, and the only question is how.** Each answer — data, tensor, pipeline, sequence, expert parallelism — splits the work along a different axis, communicates a different thing, uses a different collective operation, and demands a different interconnect ([Ch 12](./12_buses_and_interconnects.md)). A real cluster combines several at once, mapping each onto the layer of the network whose bandwidth it needs.

> **The one-sentence version:** A modern model won't fit in one GPU's memory or finish in one GPU's lifetime, so you split it — across the batch (data parallel), across each matmul (tensor parallel), across layers (pipeline parallel), across the sequence, or across experts — and the interconnect bandwidth between GPUs decides which splits are even viable.

---

## Why a Single GPU Isn't Enough: The Memory Math

The temptation is to think about model size as "parameter count × 2 bytes." That's the *inference* weight footprint, and it's already too big for many models — but **training** needs far more. To train with the standard mixed-precision Adam recipe, every parameter drags along several companions.

### The 16–20 Bytes-per-Parameter Rule

Consider one scalar weight under mixed-precision training with Adam ([Ch 23](./23_number_formats_deep_dive.md) covers the dtypes). Here is everything the optimizer must keep resident in HBM for that one weight:

| Per-parameter state | dtype | Bytes | Why it exists |
|---------------------|:-----:|:-----:|---------------|
| Parameter (compute copy) | fp16/bf16 | 2 | used in forward/backward matmuls |
| Gradient | fp16/bf16 | 2 | accumulated during backward |
| Master parameter (fp32) | fp32 | 4 | high-precision copy for the update |
| Adam first moment $m$ | fp32 | 4 | running mean of gradients |
| Adam second moment $v$ | fp32 | 4 | running variance of gradients |
| **Total (states only)** | | **16** | per parameter, before activations |

That's the famous **~16 bytes/param** for mixed-precision Adam — and it's a *floor*. Add temporary buffers, gradient accumulation copies, or fp32 gradients and it creeps to **18–20 bytes/param**. (Plain SGD with momentum is lighter, ~12 bytes/param; full fp32 Adam is heavier; but the bf16+Adam recipe above is the modern default.) On top of all this come **activations** — the intermediate tensors saved during the forward pass for the backward pass — which scale with batch size and sequence length, not parameter count, and can rival or exceed the state memory.

> **The one-line proof you must shard:** a 70B model needs $70 \times 10^9 \times 16 \approx 1.12$ **TB** just for optimizer states — versus **80 GB** on an H100. You are off by **14×** before a single activation. There is no dtype trick that closes a 14× gap on one GPU; you *must* spread the state across many GPUs.

### The Table That Settles It

<div class="diagram">
<div class="diagram-title">Training Memory vs One 80 GB GPU (mixed-precision Adam, ~16 B/param)</div>
<div class="flow-h">
  <div class="flow-node green">7B<br/><small>~112 GB → 2+ GPUs</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange">70B<br/><small>~1.12 TB → 16+ GPUs</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node red">405B<br/><small>~6.5 TB → 90+ GPUs</small></div>
</div>
</div>

| Model | Params | Inference (bf16, 2 B/p) | Training states (~16 B/p) | + Activations (rough) | Min 80 GB GPUs (states only) |
|-------|:------:|:-----------------------:|:-------------------------:|:---------------------:|:----------------------------:|
| 7B | 7×10⁹ | 14 GB | **112 GB** | +tens of GB | 2 |
| 70B | 70×10⁹ | 140 GB | **1,120 GB (1.12 TB)** | +hundreds of GB | 16 |
| 405B | 405×10⁹ | 810 GB | **6,480 GB (6.48 TB)** | +hundreds of GB | 81 (→ 90+ with activations) |

Even the 7B model — which *infers* comfortably on one GPU — needs at least two GPUs to train with full Adam state and a real batch. The 70B and 405B models are not even close. This single table is why the rest of the chapter exists.

There's a second reason beyond memory: **time**. Even if a model fit, a large pretraining run is $10^{23}$–$10^{25}$ FLOPs; at one H100's ~$10^{15}$ FLOP/s that's centuries. You need thousands of GPUs working in parallel, which is the Gustafson-style weak scaling of [Ch 8](./08_parallel_processing.md). Memory forces *some* sharding; throughput forces *lots* of it.

```python
def training_memory_gb(params: int, bytes_per_param: float = 16,
                       act_gb: float = 0.0) -> float:
    """Resident HBM for mixed-precision Adam training (states + activations)."""
    return params * bytes_per_param / 1e9 + act_gb

for name, p in [("7B", 7e9), ("70B", 70e9), ("405B", 405e9)]:
    gb = training_memory_gb(int(p))
    print(f"{name:>5}: {gb:8.0f} GB states  ->  {gb/80:5.1f}× one 80 GB GPU")
# 7B:    112 GB ->  1.4× ;  70B: 1120 GB -> 14.0× ;  405B: 6480 GB -> 81.0×
```

---

## The Parallelism Dimensions

There are a handful of orthogonal axes along which you can split a training job. Each cuts the work differently, communicates a different quantity, and leans on a different collective ([collectives section](#collective-communication-the-glue)). We'll take them one at a time, then combine them.

<div class="diagram">
<div class="diagram-title">The Axes of Parallelism (what gets split)</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Data (DP / DDP)</div>
    <div class="card-desc">Split the <strong>batch</strong>. Replicate the model. All-reduce gradients.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">ZeRO / FSDP</div>
    <div class="card-desc">Data parallel, but <strong>shard the states</strong> across GPUs. Gather params just-in-time.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Tensor (TP)</div>
    <div class="card-desc">Split <strong>each matmul</strong> across GPUs. All-reduce per layer. Needs NVLink.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Pipeline (PP)</div>
    <div class="card-desc">Split <strong>layers</strong> into stages. Pass activations. Mind the bubble.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Sequence / Context</div>
    <div class="card-desc">Split the <strong>sequence</strong> dim for long context. Communicate K/V or partial attention.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Expert (EP)</div>
    <div class="card-desc">Split <strong>MoE experts</strong> across GPUs. All-to-all routes tokens.</div>
  </div>
</div>
</div>

### Data Parallelism (DP / DDP)

The simplest and oldest: **replicate the full model on every GPU, split the batch, and average the gradients.** Each GPU runs forward and backward on its own slice of the batch, producing its own gradients; an **all-reduce** sums (then averages) them so every replica applies an identical update and stays in sync.

```text
Data Parallelism (4 GPUs) — full model replicated, batch split:

  Global batch [B=256]  -->  split into 4 micro-batches of 64
  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
  │  GPU 0     │ │  GPU 1     │ │  GPU 2     │ │  GPU 3     │
  │ full model │ │ full model │ │ full model │ │ full model │
  │ batch 0:64 │ │ batch 64:.. │ │ ..        │ │ ..192:256 │
  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
        └────── ALL-REDUCE gradients (sum/avg) across all 4 ───────┘
        every GPU now has identical averaged grads → identical update
```

- **What's communicated:** the full gradient vector, every step. Comms volume ∝ **model size** (independent of batch). For a 7B model in bf16 that's 14 GB to reduce per step.
- **Collective:** `all-reduce`.
- **Memory:** *no savings* — every GPU holds the full ~16 B/param of state. DP solves *throughput*, not the memory wall. (That's what ZeRO/FSDP fix next.)
- **When:** model fits on one GPU and you want to scale the batch / speed up. The default first move.

PyTorch's `DistributedDataParallel` (DDP) overlaps the all-reduce with the backward pass: as soon as a layer's gradients are ready, it kicks off their reduction (bucketed) while later layers are still computing backward. With a fast link this hides almost all comms (see the ring-all-reduce timing in [Ch 12](./12_buses_and_interconnects.md)).

### ZeRO / FSDP: Shard the State, Stay Data-Parallel

DP wastes memory: holding $N$ identical copies of 16 B/param across $N$ GPUs is enormously redundant. **ZeRO (Zero Redundancy Optimizer)** and PyTorch's **FSDP (Fully Sharded Data Parallel)** keep the data-parallel batch split but **shard the redundant state across the $N$ GPUs**, reconstructing pieces only when needed.

<div class="diagram">
<div class="diagram-title">ZeRO Stages: Progressively Shard the 16 B/param</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">What each stage shards (across N GPUs)</div>
    <ul>
      <li><strong>ZeRO-1</strong>: optimizer states (m, v, fp32 master) → ~8 B/p sharded</li>
      <li><strong>ZeRO-2</strong>: + gradients → +2 B/p sharded</li>
      <li><strong>ZeRO-3 / FSDP</strong>: + parameters → all 16 B/p sharded</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Per-GPU memory (70B, N=64)</div>
    <ul>
      <li>Plain DP: 1120 GB each — impossible</li>
      <li>ZeRO-1: ~280 GB each — still too big</li>
      <li>ZeRO-2: ~250 GB each</li>
      <li>ZeRO-3/FSDP: ~17.5 GB each — <strong>fits!</strong></li>
    </ul>
  </div>
</div>
</div>

The mechanism for ZeRO-3 / FSDP: parameters live sharded; right before a layer's forward, the GPUs **all-gather** the shards to reconstruct that layer's full weights, run the matmul, then **discard** the gathered copy. Backward does the same, and gradients are **reduce-scattered** so each GPU ends up owning (and updating) only its shard. The memory drops by up to $N\times$; the price is extra communication.

```text
FSDP / ZeRO-3 forward for one layer:

  params sharded: GPU0 has W[0:k], GPU1 has W[k:2k], ...
        │  ALL-GATHER  → every GPU briefly holds full W
        ▼
  compute  Y = X · W   (full layer)
        │  free the gathered W  (memory reclaimed)
        ▼
  backward → REDUCE-SCATTER grads → each GPU keeps only its grad shard
```

- **What's communicated:** all-gather of params (forward + backward) + reduce-scatter of grads. Total volume is roughly **1.5×** a plain DP all-reduce — a modest comms increase for an up-to-$N\times$ memory saving.
- **Collectives:** `all-gather` (params), `reduce-scatter` (grads).
- **The tradeoff:** ZeRO-1 is nearly free (comms = DP) and saves the biggest chunk (optimizer states are 8 of the 16 bytes). ZeRO-3/FSDP saves the most memory but adds the param all-gathers. You climb stages until the model fits.

```python
# FSDP wrap sketch (PyTorch). Wrapping per-block lets FSDP gather/free
# one transformer block at a time, so peak memory ≈ one block's params.
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

model = build_transformer().cuda()                   # 70B params, sharded by FSDP
policy = functools.partial(transformer_auto_wrap_policy,
                           transformer_layer_cls={TransformerBlock})

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,   # = ZeRO-3 (shard p, g, opt)
    auto_wrap_policy=policy,                          # gather/free one block at a time
    device_id=torch.cuda.current_device(),
)
# FULL_SHARD = ZeRO-3 ; SHARD_GRAD_OP = ZeRO-2 ; NO_SHARD = plain DDP
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)  # states are sharded too
```

### Tensor Parallelism (TP): Split the Matmul Itself

DP/FSDP keep each matmul whole on one GPU. **Tensor parallelism splits the individual matmul across GPUs.** Since a layer is $Y = XW$, and a matrix can be partitioned by columns or rows, you can put different slices of $W$ on different GPUs and combine the partial results.

```text
Column-parallel then row-parallel (the Megatron MLP pattern, TP=2):

  W1 split by COLUMNS:   GPU0 = W1[:, :half]   GPU1 = W1[:, half:]
     each computes  H_i = GELU(X · W1_i)     (no comms — X is replicated)

  W2 split by ROWS:      GPU0 = W2[:half, :]   GPU1 = W2[half:, :]
     each computes  Y_i = H_i · W2_i          (partial sums)
        │  ALL-REDUCE  Y = Y0 + Y1   ← one all-reduce per layer block
        ▼
  identical Y on both GPUs, ready for the next block
```

The clever part of the Megatron scheme: pairing a **column-split** first matmul with a **row-split** second matmul means the only synchronization is **one all-reduce per block** (and one in the backward pass). Attention is split the analogous way — heads are distributed across GPUs.

- **What's communicated:** activations (the $Y$ partials), **twice per transformer block** (once forward, once backward), *every layer, every step*. This is frequent and latency-sensitive.
- **Collective:** `all-reduce`, in the critical path of every layer.
- **The interconnect demand:** because the all-reduce happens at *every* layer (every few hundred microseconds), TP only makes sense over **NVLink/NVSwitch** ([Ch 12](./12_buses_and_interconnects.md)) — ~900 GB/s within a node. Over PCIe or inter-node InfiniBand the per-layer all-reduce would dominate step time. **This is why TP stays within an NVLink island (typically ≤8 GPUs).**

### Pipeline Parallelism (PP): Split the Layers Into Stages

If TP splits *within* a layer, **pipeline parallelism splits *across* layers**: GPU 0 holds layers 1–8, GPU 1 holds 9–16, and so on. Activations flow forward stage to stage; gradients flow backward. The communication is small (just the activations crossing each stage boundary, not gradients or weights) — which is why **PP tolerates slower inter-node links**.

The catch is the **pipeline bubble**: while stage 0 processes the first micro-batch, stages 1–3 sit idle waiting for data; at the end, stage 0 is idle while the tail drains. With $P$ stages and $M$ micro-batches, the naive (GPipe) bubble fraction is:

$$\text{bubble fraction} = \frac{P-1}{M + P - 1}.$$

```text
GPipe schedule, P=4 stages, M=4 micro-batches (F=forward, B=backward):
time →
 GPU0: F1 F2 F3 F4 . . . . B4 B3 B2 B1
 GPU1: .  F1 F2 F3 F4 . . B4 B3 B2 B1 .
 GPU2: .  .  F1 F2 F3 F4 B4 B3 B2 B1 . .
 GPU3: .  .  .  F1 F2 F3 F4 B4 B3 B2 B1
       └ fill ┘            └ drain ┘     ← idle GPUs = the "bubble"

Fix: more micro-batches (M≫P) shrinks the bubble: (P-1)/(M+P-1).
Better schedule: 1F1B (interleaved) keeps memory bounded and reduces idle time.
```

- **What's communicated:** activations at stage boundaries (forward) and their gradients (backward) — **small** relative to weights/gradients.
- **Collective:** point-to-point `send`/`recv` (not a global collective).
- **Mitigations:** raise $M$ (more micro-batches) so the bubble fraction shrinks; use the **1F1B** (one-forward-one-backward) interleaved schedule to bound activation memory and cut idle time versus GPipe.
- **When:** to span **across nodes**, because the small, infrequent activation transfers survive the ~50 GB/s inter-node links.

### Sequence / Context Parallelism and Expert Parallelism (EP)

Two more axes handle workloads the three classic ones don't cover well.

**Sequence / Context Parallelism** splits the **sequence dimension** $T$ across GPUs — essential for long context (100K+ tokens), where the activation and attention memory grows with $T$ (and attention compute with $T^2$). Each GPU owns a chunk of the sequence; attention then needs each query chunk to see *all* key/value chunks, so K/V (or partial attention results) are exchanged between GPUs (e.g., the **Ring Attention** pattern passes K/V around a ring). It complements TP: TP shrinks per-layer weight memory, sequence parallelism shrinks per-token activation memory.

**Expert Parallelism (EP)** is for **Mixture-of-Experts (MoE)** models, where each token is routed to only a few of many expert MLPs. Put different experts on different GPUs; a router sends each token to its chosen expert's GPU, the expert computes, and results come back. The token routing is an **all-to-all** collective (every GPU sends a different subset of tokens to every other GPU), which needs high **bisection bandwidth** — so EP, like TP, prefers NVLink-class fabrics.

```text
Expert Parallelism (MoE), 4 experts on 4 GPUs:

  tokens (each routed to top-1 expert):
    GPU0: [t→E2, t→E0, t→E3, ...]   ── ALL-TO-ALL ──►  regroup by expert
    GPU1: [t→E1, t→E1, t→E2, ...]                      GPU0 gets all E0 tokens
    ...                                                 GPU1 gets all E1 tokens, ...
  each GPU runs ONLY its expert MLP on its gathered tokens
        │  ALL-TO-ALL back  → tokens return to their original GPU/position
        ▼
  continue with next layer
```

### 3D / n-D Parallelism: Combine Them on the Topology

Real frontier runs use **all of these at once**, and — crucially — map each axis onto the layer of the network whose bandwidth it needs ([Ch 12](./12_buses_and_interconnects.md) scale-up vs scale-out). The standard recipe:

<div class="diagram">
<div class="diagram-title">3D Parallelism Mapped to the Network Hierarchy</div>
<div class="flow">
  <div class="flow-node blue wide">Tensor Parallel (TP) — inside one node, over NVLink (~900 GB/s). Per-layer all-reduce. Degree ≤8.</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Pipeline Parallel (PP) — across nodes, over InfiniBand (~50 GB/s). Small activation send/recv tolerates the slower link.</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">Data Parallel (DP / FSDP) — outermost, across pipeline replicas. Gradient all-reduce overlaps with backward.</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">(+ Expert Parallel for MoE; + Sequence Parallel for long context) — slotted in where bisection bandwidth allows.</div>
</div>
</div>

The arithmetic of degrees multiplies: total GPUs $= \text{TP} \times \text{PP} \times \text{DP}$. A 405B training run might use **TP=8** (one NVLink node), **PP=16** (16 nodes deep), **DP=16** (16 replicas) $= 2048$ GPUs. The mapping is deliberate: the chatty per-layer all-reduce rides NVLink, the infrequent pipeline activations cross the slower inter-node fabric, and the gradient all-reduce sits outermost where it can be overlapped with the whole backward pass. **Parallelism strategy is, fundamentally, a function of the interconnect topology.**

---

## Collective Communication: The Glue

Every strategy above is named by a **collective communication primitive** — an operation where a group of GPUs cooperatively move/combine data. These are implemented by **NCCL** (NVIDIA Collective Communications Library), the GPU-aware MPI of deep learning, which picks ring or tree algorithms and routes over NVLink/InfiniBand automatically.

<div class="diagram">
<div class="diagram-title">The Five Collectives You Actually Use</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">All-Reduce</div>
    <div class="card-desc">Every GPU contributes a tensor; all end with the sum. The DP gradient sync and TP layer sync. = reduce-scatter + all-gather.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">All-Gather</div>
    <div class="card-desc">Each GPU holds a shard; all end with the full concatenation. FSDP param reconstruction.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Reduce-Scatter</div>
    <div class="card-desc">Sum across GPUs, but each keeps only its slice of the result. FSDP gradient sharding.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">All-to-All</div>
    <div class="card-desc">Each GPU sends a distinct chunk to every other GPU. MoE token routing (EP).</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Broadcast</div>
    <div class="card-desc">One GPU's tensor copied to all. Initial weight/seed distribution at startup.</div>
  </div>
</div>
</div>

### The Ring-All-Reduce Bandwidth Formula

The all-reduce is the workhorse, so its cost model matters. The **ring all-reduce** arranges $N$ GPUs in a logical ring and proceeds in two phases — a reduce-scatter then an all-gather — each moving $(N-1)/N$ of the data around the ring. The total bytes each GPU sends (and receives) is:

$$\text{bytes per GPU} \approx 2 \cdot \frac{N-1}{N} \cdot \text{size} \xrightarrow{\text{large } N} 2 \cdot \text{size}$$

so the time is approximately

$$T_{\text{all-reduce}} \approx \frac{2(N-1)}{N} \cdot \frac{\text{size}}{\text{BW}} \approx \frac{2 \cdot \text{size}}{\text{BW}}.$$

The beautiful property: the per-GPU traffic is **independent of $N$** for large $N$ — ring all-reduce is bandwidth-optimal. (A **tree** all-reduce instead has *latency* $O(\log N)$, better for very small tensors where latency, not bandwidth, dominates; NCCL switches between them.) This is the same formula derived and benchmarked in [Ch 12](./12_buses_and_interconnects.md), where it showed the same 7B all-reduce running ~300× faster on NVLink than on PCIe.

> **Why the interconnect gates scaling.** The all-reduce time is $2\cdot\text{size}/\text{BW}$ — and `BW` is the GPU-to-GPU link bandwidth, *not* the HBM bandwidth. A 7B model's 14 GB bf16 gradient all-reduce is ~0.06 s over NVLink's ~450 GB/s effective, but ~0.9 s over PCIe's ~16 GB/s. Multiply by thousands of steps and the link, not the GPU, sets your training throughput. The entire point of **NVSwitch** (intra-node) and **InfiniBand** (inter-node) is to make `BW` large enough that comms hides behind compute.

### Overlapping Comms with Compute

The reason all this is even tractable is **overlap**: communication is launched on a separate CUDA stream and runs *while* the GPUs are still computing. DDP fires each layer's gradient all-reduce the moment that layer's backward finishes, so the reduction of early layers overlaps the backward of later ones. FSDP prefetches the *next* layer's param all-gather while the current layer computes. As long as the exposed (non-overlapped) comms time is less than the compute time, scaling stays efficient; when the link is too slow, comms becomes the exposed Amdahl serial fraction ([Ch 8](./08_parallel_processing.md)) that caps your speedup.

---

## Python: A Minimal DDP Sketch and a Memory Estimator

A bare-metal `torch.distributed` data-parallel step, stripped to the collective so the mechanics are visible:

```python
import torch, torch.distributed as dist

def ddp_step(model, batch, optimizer):
    """One manual data-parallel step: local backward, then all-reduce grads."""
    loss = model(batch).loss
    loss.backward()                                  # local grads on each GPU

    world = dist.get_world_size()
    for p in model.parameters():                     # (real DDP buckets these)
        if p.grad is not None:
            # SUM grads across all ranks, then average -> identical update
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world

    optimizer.step()                                 # every rank applies same update
    optimizer.zero_grad(set_to_none=True)

# In practice you just wrap the model and DDP handles bucketed, overlapped
# all-reduce on a side stream for you:
#   from torch.nn.parallel import DistributedDataParallel as DDP
#   model = DDP(model, device_ids=[local_rank])
```

And a memory estimator that turns a model + parallelism config into per-GPU GB — the calculation you should run *before* launching a job:

```python
from dataclasses import dataclass

@dataclass
class MemPlan:
    per_gpu_gb: float
    fits_80gb: bool
    note: str

def estimate_per_gpu_gb(
    params: int,
    dp: int = 1, tp: int = 1, pp: int = 1,   # parallelism degrees
    zero_stage: int = 0,                     # 0=DDP, 1/2/3 = ZeRO/FSDP
    bytes_per_param: float = 16,             # mixed-precision Adam floor
    activation_gb_per_gpu: float = 20.0,     # rough, depends on batch/seq
) -> MemPlan:
    """Per-GPU resident HBM for a given model + 3D + ZeRO config."""
    # TP and PP physically partition params/states across their groups:
    state_params = params / (tp * pp)
    state_bytes = state_params * bytes_per_param

    # ZeRO/FSDP additionally shards across the DP group:
    if zero_stage == 1:    # shard optimizer states (8 of 16 B/p) across DP
        state_bytes -= state_params * 8 * (1 - 1 / dp)
    elif zero_stage == 2:  # + gradients (2 B/p)
        state_bytes -= state_params * 10 * (1 - 1 / dp)
    elif zero_stage == 3:  # + params (2 B/p) -> all 16 B/p sharded across DP
        state_bytes = state_bytes / dp

    gb = state_bytes / 1e9 + activation_gb_per_gpu
    return MemPlan(gb, gb <= 80,
                   f"dp={dp} tp={tp} pp={pp} zero={zero_stage}")

# 70B on 64 GPUs, three plans:
print(estimate_per_gpu_gb(70_000_000_000, dp=64, zero_stage=0))   # plain DDP: ~1140 GB -> NO
print(estimate_per_gpu_gb(70_000_000_000, dp=64, zero_stage=3))   # FSDP:     ~37 GB  -> fits
print(estimate_per_gpu_gb(70_000_000_000, dp=8, tp=8, zero_stage=1))  # TP=8 + ZeRO-1
```

---

## How to Choose a Strategy

The decision flows directly from three numbers: model size, GPU memory, and interconnect bandwidth. There is no single right answer — every "it won't fit" has several solutions with different comms costs.

<div class="diagram">
<div class="diagram-title">Choosing a Parallelism Strategy</div>
<div class="flow">
  <div class="flow-node green wide">Fits on 1 GPU (with batch)? → plain DDP. Scale the batch, all-reduce grads. Done.</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">States too big but params fit? → FSDP / ZeRO. Climb stages 1→2→3 until it fits. Comms ≈ 1.5× DDP.</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">A single layer's matmul is too big / too slow? → add TP within the NVLink node (degree ≤8).</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Model is too deep for one node even with TP? → add PP across nodes; raise micro-batches to shrink the bubble.</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Frontier scale? → 3D: TP(NVLink) × PP(InfiniBand) × DP(outermost). + EP for MoE, + SP for long context.</div>
</div>
</div>

| Strategy | Splits | Communicates | Collective | Memory saved | Needs |
|----------|--------|--------------|:----------:|:------------:|-------|
| DDP | batch | full gradients / step | all-reduce | none | any link (overlap helps) |
| ZeRO-1/2 | batch + states | grads (+ states) | reduce-scatter, all-gather | states / DP | DP-class link |
| ZeRO-3 / FSDP | batch + all state | params + grads | all-gather, reduce-scatter | up to N× | DP-class link |
| TP | each matmul | activations / layer | all-reduce | params / TP | **NVLink** (intra-node) |
| PP | layers | activations / stage | send/recv | params / PP | tolerates inter-node |
| EP | experts | token routing | all-to-all | experts / EP | high bisection (NVLink) |
| SP | sequence | K/V or partial attn | send/recv, all-gather | activations / SP | NVLink preferred |

Two rules of thumb that fall out of the whole chapter:

1. **Memory chooses the *minimum* sharding; bandwidth chooses *how* you shard.** First make it fit (FSDP/TP/PP), then arrange the axes so the chattiest collective (TP's per-layer all-reduce) rides the fastest link (NVLink) and the infrequent one (DP gradient all-reduce) sits where it can overlap.
2. **"It won't fit" is never a dead end.** FSDP-shard the state, TP-split the matmul, PP-split the depth, or offload to CPU/NVMe ([Ch 12](./12_buses_and_interconnects.md), [Ch 29](./29_memory_hierarchy_in_action.md)) — each trades memory for a different, quantifiable comms cost.

---

## Why This Matters for Model Optimization

<div class="diagram">
<div class="diagram-title">Distributed Training → Your Decisions</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Run the memory math first</div>
    <div class="card-desc">~16 B/param × params + activations vs (GPUs × 80 GB). This single calc dictates the minimum world size and whether FSDP/TP/PP is mandatory.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">FSDP is the default scaler</div>
    <div class="card-desc">For most "won't fit" cases, ZeRO-3/FSDP is the lowest-friction fix: data-parallel ergonomics, near-N× memory saving, ~1.5× comms.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Keep TP inside the node</div>
    <div class="card-desc">TP's per-layer all-reduce only pays off over NVLink. Setting TP across nodes is a classic, costly misconfiguration.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Pipeline = fight the bubble</div>
    <div class="card-desc">PP scales depth across slow links but idles GPUs. More micro-batches and 1F1B scheduling recover the lost utilization.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Comms must overlap compute</div>
    <div class="card-desc">If exposed all-reduce time ≳ compute time, you're bandwidth-bound. Bigger batches, bf16/fp8 gradients, and bucketing restore overlap.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Lower precision helps twice</div>
    <div class="card-desc">bf16/fp8 shrink both the state footprint (fits sooner) and the gradient all-reduce volume (comms cheaper). Ties straight back to Ch 23–25.</div>
  </div>
</div>
</div>

---

## Key Takeaways

- **Memory forces sharding.** Mixed-precision Adam costs **~16 bytes/param** (fp16 param 2 + fp16 grad 2 + fp32 master 4 + Adam $m,v$ 4+4), plus activations. A **70B model needs ~1.12 TB** of state versus 80 GB/GPU — a 14× gap no dtype trick closes. You *must* shard.
- **Data parallelism (DDP)** replicates the model, splits the batch, and **all-reduces gradients** (comms ∝ model size); it scales throughput but saves no memory.
- **ZeRO / FSDP** keep the data-parallel batch split but **shard optimizer states (ZeRO-1), gradients (ZeRO-2), and parameters (ZeRO-3/FSDP)** across GPUs — up to $N\times$ memory savings for ~1.5× the comms, via just-in-time `all-gather` and `reduce-scatter`.
- **Tensor parallelism (TP)** splits each matmul (column/row partition) and **all-reduces within every layer** — so it must stay inside an **NVLink island (≤8 GPUs)**. **Pipeline parallelism (PP)** splits layers into stages, passes small activations (tolerates inter-node links), but pays a **bubble** of $(P-1)/(M+P-1)$, mitigated by more micro-batches and 1F1B.
- **Sequence/context parallelism** splits the sequence for long context; **expert parallelism (EP)** routes MoE tokens to expert GPUs via **all-to-all**. Frontier runs combine everything into **3D/n-D parallelism**: TP on NVLink, PP across nodes, DP outermost.
- **Collectives (all-reduce, all-gather, reduce-scatter, all-to-all, broadcast)** are implemented by **NCCL**. Ring all-reduce costs $\approx 2(N-1)/N \cdot \text{size}/\text{BW}$ — bandwidth-optimal and $N$-independent — so **interconnect bandwidth, not HBM, gates scaling**, and comms must **overlap** compute.
- **Choose the strategy from the numbers:** model size + GPU memory pick the minimum sharding; interconnect bandwidth picks the arrangement (chatty collectives on fast links). "It won't fit" always has solutions — each with a different, quantifiable comms cost.

---

*Next: [Chapter 29 — Memory Hierarchy in Action](./29_memory_hierarchy_in_action.md), where we drop from the cluster back down to a single GPU's HBM and show how the memory hierarchy drives the KV-cache, batching, activation checkpointing, and offloading decisions you make at inference and fine-tuning time.*

[← Back to Table of Contents](./README.md)
