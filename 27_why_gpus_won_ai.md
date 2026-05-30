---
title: "Chapter 27 — Why GPUs Won AI"
---

[← Back to Table of Contents](./README.md)

# Chapter 27 — Why GPUs Won AI

Every chapter so far has built a piece of the machine: transistors ([Ch 1](./01_what_is_a_chip.md)), the parallelism pivot forced by the end of Dennard scaling ([Ch 8](./08_parallel_processing.md)), HBM ([Ch 9](./09_memory_types.md)), the roofline ([Ch 11](./11_memory_wall_and_bandwidth.md)), NVLink ([Ch 12](./12_buses_and_interconnects.md)), and the GPU itself ([Ch 15](./15_gpus.md)). This chapter is the **payoff** — where all of that hardware meets the math you actually run. It answers the question every ML practitioner half-knows but rarely sees derived: *why does my entire stack assume a GPU?*

The short answer is not "GPUs are fast." It's that **the dominant operation in a neural network — dense matrix multiplication — is the exact operation a GPU was already built to do millions of times per frame for computer graphics.** This was not a plan. Graphics hardware and deep learning evolved independently for two decades and then collided in 2012, when a convolutional network trained on two gaming GPUs won an image-recognition contest by a margin that ended the debate. The collision was structural, not lucky: both workloads are piles of independent multiply-accumulates over big arrays, and both are starved for memory bandwidth in exactly the same way.

> **The one-sentence version:** GPUs won AI because neural networks are, mechanically, sequences of large dense matrix multiplies — the same embarrassingly parallel multiply-accumulate workload GPUs were built to do for graphics — and CUDA made that hardware programmable just in time for deep learning to need it.

---

## The Accident That Wasn't: Graphics Math Is Neural-Net Math

A GPU was designed to answer one question billions of times per second: *what color is this pixel?* Rendering a 3D scene means transforming millions of vertices (matrix-vector products), then shading millions of pixels — each computation independent of its neighbors, each a small dot product or multiply-accumulate, all running the same program on different data. This is the textbook definition of an **embarrassingly parallel** workload, and it's why the GPU evolved into thousands of simple arithmetic units fed by very wide memory (the throughput philosophy of [Ch 15](./15_gpus.md)).

Now look at a neural network. A fully-connected layer transforms a vector by a weight matrix — a matrix-vector product. A batch of inputs makes it a matrix-matrix product. A convolution is a sliding dot product. Attention is a cascade of matrix multiplies. **The arithmetic of "transform millions of independent things by a shared set of weights" is identical** to "transform millions of independent vertices/pixels by a shared transform." The GPU didn't adapt to AI; AI turned out to be shaped like graphics.

<div class="diagram">
<div class="diagram-title">Two Workloads, One Shape</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Graphics (what GPUs were built for)</div>
    <ul>
      <li>Millions of independent vertices/pixels</li>
      <li>Each: small dot product / matrix-vector op</li>
      <li>Same shader program, different data (SIMT)</li>
      <li>Streams huge vertex/texture arrays from memory</li>
      <li>Latency irrelevant; throughput is everything</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Deep learning (what showed up later)</div>
    <ul>
      <li>Millions of independent activations</li>
      <li>Each: a row of a matrix multiply (MAC chain)</li>
      <li>Same layer math, different data (SIMT)</li>
      <li>Streams huge weight/activation tensors from HBM</li>
      <li>Latency hidden by batching; throughput is everything</li>
    </ul>
  </div>
</div>
</div>

### Two Dates That Made It Happen: CUDA (2007) and AlexNet (2012)

For decades GPUs were locked behind graphics APIs (OpenGL, DirectX): to do general math you had to disguise it as a shading operation. That changed when NVIDIA shipped **CUDA** (Compute Unified Device Architecture, first public toolkit 2007 on the G80) — a C/C++ dialect that let you write arbitrary parallel programs ("kernels") for the GPU without pretending they were graphics. CUDA is the reason a GPU became a general matrix-multiply engine you could target from research code (we cover the whole stack — CUDA, PTX, cuBLAS, Triton — in [Ch 6](./06_languages_and_nomenclature.md)).

The match became undeniable in 2012. **AlexNet**, a deep convolutional network, won the ImageNet classification challenge with a top-5 error of ~15.3% versus ~26% for the next-best (non-deep) entry. Crucially, it was trained on **two NVIDIA GTX 580 GPUs** (3 GB GDDR5 each) — the network was even split across the two cards because it didn't fit on one, a foreshadowing of [Chapter 28](./28_multi_gpu_and_distributed_training.md). The lesson the field absorbed overnight: deep nets work *if* you can afford the compute, and GPUs make the compute affordable. Every subsequent leap — ResNets, Transformers, GPT, diffusion models — rode the same hardware.

<div class="diagram">
<div class="diagram-title">The Collision (independent histories, then one moment)</div>
<div class="flow">
  <div class="flow-node accent wide">GPUs evolve for graphics: thousands of parallel shader ALUs, wide memory (1999–2006)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">CUDA (2007): the GPU becomes a programmable general matmul engine, not just a graphics pipe</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">AlexNet (2012): a deep CNN on 2× GTX 580 crushes ImageNet — the field pivots to GPUs</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Tensor Cores (2017+): dedicated matmul silicon makes the fit explicit in hardware (Ch 15/24)</div>
</div>
</div>

---

## The Mechanics: Every Layer Is a Matrix Multiply

The claim "neural nets are matrix multiplies" is so often repeated it's become a slogan. Let's make it mechanical and exact, because the *shapes and byte counts* are what connect to the hardware.

### A Linear Layer Is Literally a GEMM

A linear (fully-connected) layer computes $Y = XW$ (we'll fold in bias separately). For a Transformer processing a batch of $B$ sequences of length $T$ with hidden size $d_{\text{in}}$, projecting to $d_{\text{out}}$:

| Tensor | Shape | dtype | Bytes (BF16) |
|--------|-------|:-----:|:------------:|
| Input $X$ | $[B\cdot T,\ d_{\text{in}}]$ | bf16 | $2 \cdot B T \cdot d_{\text{in}}$ |
| Weight $W$ | $[d_{\text{in}},\ d_{\text{out}}]$ | bf16 | $2 \cdot d_{\text{in}} \cdot d_{\text{out}}$ |
| Output $Y$ | $[B\cdot T,\ d_{\text{out}}]$ | bf16 | $2 \cdot B T \cdot d_{\text{out}}$ |

This is exactly a **GEMM** (GEneral Matrix Multiply) with $M = B\cdot T$, $K = d_{\text{in}}$, $N = d_{\text{out}}$. The "batch" and "sequence" dimensions collapse into a single tall $M$ dimension — which is *why* batching and longer sequences raise arithmetic intensity (more rows reuse the same weight matrix; see the roofline below and [Ch 11](./11_memory_wall_and_bandwidth.md)).

The FLOP count of a GEMM is the single most important number in this guide:

$$\text{FLOPs}_{\text{GEMM}} = 2\,M N K$$

The factor of 2 is one multiply + one add per inner-product term; there are $MN$ output elements each requiring a length-$K$ dot product, hence $M N K$ multiply-adds $= 2MNK$ FLOPs. A Llama-style 7B model at $d=4096$, $d_{\text{ff}}=11008$, processing 2048 tokens, runs roughly $2 \times 2048 \times 4096 \times 11008 \approx 1.8 \times 10^{11}$ FLOPs **for one MLP up-projection of one layer** — and there are dozens of layers, each with several such matmuls.

### Attention Is Batched Matmuls

Self-attention looks exotic but is four matmuls and a softmax. With $H$ heads, head dim $d_h$, sequence length $T$:

```text
Per head h (shapes shown for one sequence in the batch):
  Q  = X · W_Q        [T, d_h]   ← GEMM
  K  = X · W_K        [T, d_h]   ← GEMM
  V  = X · W_V        [T, d_h]   ← GEMM
  S  = Q · Kᵀ         [T, T]     ← batched GEMM   (FLOPs = 2·T·T·d_h)
  P  = softmax(S)     [T, T]     ← memory-bound elementwise (NOT a matmul)
  O  = P · V          [T, d_h]   ← batched GEMM   (FLOPs = 2·T·T·d_h)
Then concat heads and project:  Out = concat(O) · W_O   [T, d_model]  ← GEMM
```

The $QK^\top$ and $PV$ steps are **batched matmuls**: one independent GEMM per (batch, head) pair, which the GPU runs as a single batched kernel across thousands of warps. The softmax in the middle is the awkward leftover we'll return to — it's bandwidth-bound, not a matmul, and it's exactly what Flash Attention exists to fuse away.

### Convolution Is a Matmul in Disguise (im2col / implicit GEMM)

A 2D convolution slides a $C_{\text{in}}\times k \times k$ filter over a feature map. It *looks* like a stencil, but it's lowered to a GEMM by **im2col**: every receptive-field patch is flattened into a column, turning the input into a big matrix, after which the convolution is one matrix multiply against the flattened filters.

```text
im2col lowering of a conv layer:

  Input  [C_in, H, W]  --im2col-->  Patches [C_in·k·k, H_out·W_out]
  Filters [C_out, C_in·k·k]                              │
                                                          ▼
  Output [C_out, H_out·W_out]  =  Filters  ·  Patches        ← one GEMM
                                  M=C_out  K=C_in·k·k  N=H_out·W_out
```

Modern libraries (cuDNN) usually skip the explicit im2col buffer and use **implicit GEMM** — they index the input on-the-fly so the data movement matches a GEMM without materializing the (memory-hungry) patch matrix. Either way: **a convolution becomes a matrix multiply that runs on Tensor Cores.**

### The Punchline: >90% of FLOPs Are GEMM

Add it up across a real model and the matmuls dominate overwhelmingly:

| Operation class | Examples | Share of FLOPs (typical Transformer) | Hardware |
|-----------------|----------|:------------------------------------:|----------|
| Dense GEMM | QKV/out proj, MLP up/down, LM head | **~95–99%** | Tensor Cores |
| Batched GEMM | $QK^\top$, $PV$ in attention | (included above) | Tensor Cores |
| Elementwise / reductions | softmax, LayerNorm, GELU, residual add | ~1–5% | CUDA cores (memory-bound) |

> The entire reason **Tensor Cores** exist ([Ch 15](./15_gpus.md), [Ch 24](./24_datatypes_and_silicon.md)) is this 95%+. If you build dedicated silicon that does only one thing — a small matmul fragment per clock — and that one thing is 95% of the workload, you win. On an H100 the Tensor Cores deliver ~989 BF16 TFLOPS versus ~62 FP32 TFLOPS from the general CUDA cores: a ~16× gap. A kernel that doesn't hit Tensor Cores leaves the GPU 90%+ idle.

---

## Why GPUs Specifically — Not CPUs

If a matmul is just multiply-adds, why can't a CPU do it? It can — just slowly. The difference is architectural philosophy, recapped from [Ch 14](./14_cpus.md) and [Ch 15](./15_gpus.md): the CPU is a **latency** machine (finish one thread's work as fast as possible, with big caches, out-of-order execution, branch prediction), while the GPU is a **throughput** machine (keep tens of thousands of threads in flight and hide every stall behind another warp).

A matmul wants throughput. It has no branches, no data-dependent control flow, and effectively unlimited independent work (every output element is independent). The CPU's expensive latency machinery — branch predictors, reorder buffers — buys nothing here; the GPU's thousands of MAC units and TB/s of HBM bandwidth buy everything.

<div class="diagram">
<div class="diagram-title">Computing a 4096×4096×4096 BF16 GEMM: CPU vs GPU</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">CPU (e.g. EPYC Genoa)</div>
    <ul>
      <li>~64–96 OoO cores, AVX-512 SIMD (16 fp32 MACs/core/cycle)</li>
      <li>~3–5 TFLOPS fp32 peak across all cores</li>
      <li>~461 GB/s DDR5 memory bandwidth</li>
      <li>Spends transistors on caches/prediction, not MACs</li>
      <li>Matmul: ~50+ ms — bandwidth + ALU starved</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">GPU (H100 SXM5)</div>
    <ul>
      <li>132 SMs × 4 Tensor Cores, thousands of MAC lanes</li>
      <li>~989 BF16 TFLOPS via Tensor Cores</li>
      <li>3.35 TB/s HBM3 bandwidth (~7× the CPU)</li>
      <li>Spends transistors on MACs; hides latency via 64 warps/SM</li>
      <li>Matmul (~137 GFLOP): ~0.15 ms — Tensor-Core-bound</li>
    </ul>
  </div>
</div>
</div>

Three GPU properties make the difference, each tracing back to an earlier chapter:

1. **Thousands of MAC units.** The GPU spends its transistor budget ([Ch 1](./01_what_is_a_chip.md)) on arithmetic, not control. Tensor Cores pack the multiply-accumulate density a matmul craves.
2. **HBM bandwidth.** A matmul streams gigabytes of weights/activations. The H100's 3.35 TB/s HBM3 ([Ch 9](./09_memory_types.md)) feeds the MACs; a CPU's ~461 GB/s DDR5 starves them.
3. **Latency hidden by oversubscription.** With up to 64 warps resident per SM, when one warp stalls on an HBM load (~300–400 cycles), 63 others run ([Ch 15](./15_gpus.md)). The CPU instead spends silicon predicting and reordering to hide latency for *one* thread.

### The Roofline Says the Same Thing

The roofline model ([Ch 11](./11_memory_wall_and_bandwidth.md)) makes "matmul loves the GPU" quantitative. The H100's BF16 ridge point is

$$\text{AI}^*_{H100} = \frac{989 \times 10^{12}\ \text{FLOP/s}}{3.35 \times 10^{12}\ \text{B/s}} \approx 295\ \text{FLOP/byte}.$$

A large square GEMM has arithmetic intensity $\approx M/3$ (FLOPs grow as $M^3$, bytes as $M^2$), so for $M=N=K=4096$ the AI is ~1365 FLOP/byte — *far* above the ridge. The matmul sits firmly in the compute-bound regime where the GPU's peak throughput is the limit, exactly what you want. The non-matmul ops (softmax, LayerNorm) sit far to the left, memory-bound — which is precisely why they're the "awkward leftovers" and why fusion exists.

---

## The Software Moat: It's Not Just Hardware

If GPUs were *only* fast silicon, competitors would have caught up. The deeper reason GPUs (specifically NVIDIA's) won is the **software ecosystem** built on top — a moat that takes years to cross. The matmul has to be *driven* by something, and over two decades NVIDIA built every layer:

<div class="diagram">
<div class="diagram-title">The CUDA Stack: Why It's a Moat (top = your code)</div>
<div class="layer-stack">
  <div class="layer"><strong>Frameworks</strong> — PyTorch, JAX, TensorFlow: where you actually write models</div>
  <div class="layer"><strong>Kernel libraries</strong> — cuDNN (conv/attention), cuBLAS/cuBLASLt (GEMM), CUTLASS, NCCL (collectives)</div>
  <div class="layer"><strong>Compilers / DSLs</strong> — torch.compile→Triton, XLA; PTX virtual ISA for forward compatibility</div>
  <div class="layer"><strong>CUDA runtime + driver</strong> — kernel launch, memory management, streams, CUDA Graphs</div>
  <div class="layer"><strong>Hardware</strong> — SMs, Tensor Cores, HBM, NVLink</div>
</div>
</div>

The crucial libraries are **cuBLAS** (hand-tuned GEMM that actually extracts the 989 TFLOPS — naive Tensor Core code gets a fraction) and **cuDNN** (the implicit-GEMM convolutions and fused attention). When you write `x @ w` in PyTorch, you ride a dispatch path through ATen → cuBLAS → a PTX kernel using Tensor Core `mma` instructions ([Ch 6](./06_languages_and_nomenclature.md) traces this exact path). Decades of tuning are baked into that one call.

This is why "we built a faster matmul chip" is necessary but nowhere near sufficient. AMD's ROCm/HIP and various AI ASICs are competitive on raw FLOPS but spent years rebuilding the kernel libraries, framework integration, and debugging tools that CUDA accreted. The moat is the reason your stack assumes `device='cuda'` by default — we revisit this competitive dynamic in [Chapter 33 — Design-Paradigm Wars](./33_design_paradigm_wars.md).

---

## The Flywheel

Put hardware and software together and you get a self-reinforcing loop that explains the last decade:

<div class="diagram">
<div class="diagram-title">The GPU/AI Flywheel</div>
<div class="flow-h">
  <div class="flow-node accent">Bigger models<br/><small>more matmul FLOPs</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">More GPU demand<br/><small>revenue + roadmap</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">Better GPUs<br/><small>more TC FLOPS, more HBM</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">Bigger models again<br/><small>scaling laws reward it</small></div>
</div>
</div>

Each GPU generation roughly **doubles** effective Tensor Core throughput *and* HBM bandwidth ([Ch 15](./15_gpus.md), [Ch 34](./34_generational_improvements.md)), keeping the ridge point near 200–400 FLOP/byte so that large matmuls stay compute-bound generation after generation. That predictable doubling let researchers bet on scaling laws, which created demand for the next doubling. The flywheel is why a 2012 contest entry on two 3 GB cards became 2024 clusters of tens of thousands of 80–192 GB GPUs.

---

## Python: Express a Net as Explicit Matmuls, and Count the FLOPs

To cement the mechanics, here's a tiny MLP block and a single attention head written as **explicit matmuls**, with a FLOP counter using the $2MNK$ rule.

```python
import torch

def gemm_flops(M: int, N: int, K: int) -> int:
    """FLOPs for an M×K · K×N matrix multiply: 2*M*N*K (mul + add per MAC)."""
    return 2 * M * N * K

# ---- A Transformer MLP block as two explicit GEMMs ----------------------
B, T, d_model, d_ff = 8, 2048, 4096, 11008      # Llama-7B-ish dimensions
M = B * T                                        # tall dim: batch*seq folds into M

x  = torch.randn(M, d_model, dtype=torch.bfloat16, device='cuda')   # [16384, 4096]
W1 = torch.randn(d_model, d_ff, dtype=torch.bfloat16, device='cuda') # up-proj
W2 = torch.randn(d_ff, d_model, dtype=torch.bfloat16, device='cuda') # down-proj

h = torch.nn.functional.silu(x @ W1)             # GEMM  M×d_model · d_model×d_ff
y = h @ W2                                        # GEMM  M×d_ff   · d_ff×d_model
# The SiLU between them is the memory-bound "leftover" — not a matmul.

mlp_flops = gemm_flops(M, d_ff, d_model) + gemm_flops(M, d_model, d_ff)
print(f"MLP block: {mlp_flops/1e9:.1f} GFLOP   (two GEMMs, M={M})")  # ~2956 GFLOP

# ---- One attention head as batched matmuls ------------------------------
d_h = 128
q = torch.randn(B, T, d_h, dtype=torch.bfloat16, device='cuda')
k = torch.randn(B, T, d_h, dtype=torch.bfloat16, device='cuda')
v = torch.randn(B, T, d_h, dtype=torch.bfloat16, device='cuda')

s = q @ k.transpose(-2, -1)                       # [B, T, T]  batched GEMM
p = torch.softmax(s, dim=-1)                      # memory-bound, NOT a matmul
o = p @ v                                          # [B, T, d_h] batched GEMM

attn_flops = B * (gemm_flops(T, T, d_h) + gemm_flops(T, d_h, T))
print(f"Attn (1 head): {attn_flops/1e9:.1f} GFLOP  (two batched GEMMs)")

# ---- Sanity: matmul is the workload ------------------------------------
# Everything that matters for throughput is the GEMM term (2*M*N*K).
# The softmax/SiLU FLOPs are tiny by comparison but memory-bound -> Ch 11.
```

The takeaway in code: the lines that cost FLOPs are `@` (matmul); the `silu`/`softmax` lines cost almost no FLOPs but, being memory-bound, can still cost real *time* if not fused — which is the whole subject of the memory chapters ([Ch 11](./11_memory_wall_and_bandwidth.md), [Ch 29](./29_memory_hierarchy_in_action.md)).

---

## Why This Matters for Model Optimization

Understanding *why* GPUs won reframes nearly every architecture and optimization decision you make:

<div class="diagram">
<div class="diagram-title">"GPUs Won" → Your Daily Choices</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Architect as big matmuls</div>
    <div class="card-desc">Wide MLPs, big attention, large batch — designs that are mostly GEMM map onto Tensor Cores and run near peak. "GPU-friendly" means "matmul-heavy."</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Non-matmul ops are the cost</div>
    <div class="card-desc">Softmax, norms, activations are bandwidth-bound leftovers. Fusing them (Flash Attention, torch.compile) is where real speedups hide (Ch 29).</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">dtype = Tensor Core eligibility</div>
    <div class="card-desc">BF16/FP8/INT8 aren't just smaller — they unlock the dedicated matmul silicon (Ch 24). FP32 falls back to slow CUDA cores or TF32.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Batch to feed the beast</div>
    <div class="card-desc">A GPU is a throughput machine; batch=1 decode wastes it. Batching raises arithmetic intensity past the ridge point (Ch 11/29).</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">The stack is CUDA-shaped</div>
    <div class="card-desc">Your defaults (device='cuda', cuBLAS GEMM, NCCL) reflect the software moat. Portability to other accelerators means rebuilding it (Ch 33).</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">One GPU isn't enough</div>
    <div class="card-desc">The same matmul-everywhere structure that fits one GPU also shards cleanly across many — which is the next chapter (Ch 28).</div>
  </div>
</div>
</div>

---

## Key Takeaways

- GPUs won AI because **neural networks are dominated by dense matrix multiply** — the same embarrassingly parallel multiply-accumulate workload GPUs were already built to do for graphics. The fit was structural, not lucky.
- **CUDA (2007)** turned the graphics chip into a programmable general matmul engine; **AlexNet (2012)**, trained on 2× GTX 580, proved deep nets + GPUs win and triggered the field's pivot.
- Every layer reduces to a **GEMM**: linear layers are $Y=XW$ with $M=B\cdot T$; attention is batched $QK^\top$ and $PV$; convolution is im2col/implicit GEMM. A GEMM costs $\mathbf{2MNK}$ FLOPs, and **>90% of model FLOPs are GEMM** → Tensor Cores.
- GPUs beat CPUs not by being "faster" but by **philosophy**: thousands of MAC units, TB/s HBM, and latency hidden by warp oversubscription — exactly what a branchless, infinitely-parallel matmul wants. The roofline puts large GEMMs at AI ≈ $M/3 \gg 295$, firmly compute-bound.
- The **software moat** (cuBLAS, cuDNN, NCCL, framework integration) is why raw FLOPS aren't enough to displace NVIDIA — your stack defaults to `cuda` for a reason (Ch 33).
- The **flywheel** (bigger models → more demand → better GPUs → bigger models), powered by per-generation doubling of TC FLOPS and HBM bandwidth, explains the scale-up of the last decade.
- For you: **architect models as big matmuls**, pick Tensor-Core dtypes, batch to raise arithmetic intensity, and treat non-matmul ops as bandwidth-bound leftovers to fuse away.

---

*Next: [Chapter 28 — Multi-GPU & Distributed Training](./28_multi_gpu_and_distributed_training.md), where one GPU stops being enough: we do the memory math that proves a 70B model can't fit on 80 GB, then tour every way to split the work — data, tensor, pipeline, expert, and n-D parallelism — and the collective communication that glues them together.*

[← Back to Table of Contents](./README.md)
