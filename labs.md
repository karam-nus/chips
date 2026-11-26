---
title: "Interactive Labs"
---

[← Back to Table of Contents](./README.md)

# Interactive Labs

> *"What I cannot create, I do not understand."* — Richard Feynman

## Overview

Nine small instruments, one per load-bearing idea in this guide. Each one is a plain JavaScript widget with no dependencies, built around a pure function that is checked against hand-derived values by `node tools/test_labs.js`. Every lab also appears inside the chapter it belongs to; this page collects them so you can play with all of them in one place.

<div class="diagram">
<div class="diagram-grid cols-4">
  <div class="diagram-card accent"><div class="card-title">Roofline</div><div class="card-desc">Compute- or memory-bound? <a href="./11_memory_wall_and_bandwidth.html">Ch 11</a></div></div>
  <div class="diagram-card purple"><div class="card-title">Bit explorer</div><div class="card-desc">FP32 → FP4, bit by bit. <a href="./23_number_formats_deep_dive.html">Ch 23</a></div></div>
  <div class="diagram-card cyan"><div class="card-title">Will it fit?</div><div class="card-desc">Weights, KV cache, optimizer state. <a href="./29_memory_hierarchy_in_action.html">Ch 29</a></div></div>
  <div class="diagram-card orange"><div class="card-title">Systolic array</div><div class="card-desc">A matmul, one cycle at a time. <a href="./16_tpus_and_systolic_arrays.html">Ch 16</a></div></div>
  <div class="diagram-card green"><div class="card-title">Wafer yield</div><div class="card-desc">Why big dies cost so much. <a href="./appendix_a_how_chips_are_made.html">App A</a></div></div>
  <div class="diagram-card pink"><div class="card-title">2:4 sparsity</div><div class="card-desc">What the sparse tensor core stores. <a href="./26_sparsity_hardware_view.html">Ch 26</a></div></div>
  <div class="diagram-card yellow"><div class="card-title">Quantization</div><div class="card-desc">Bits, groups, outliers, MXFP4 vs NVFP4. <a href="./25_quantization_hardware_view.html">Ch 25</a></div></div>
  <div class="diagram-card cyan"><div class="card-title">Cache simulator</div><div class="card-desc">Lines, sets, ways, three kinds of miss. <a href="./appendix_i_memory_and_caches.html">App I</a></div></div>
  <div class="diagram-card blue"><div class="card-title">Atlas</div><div class="card-desc">26 accelerators, sortable. <a href="./atlas.html">Atlas</a></div></div>
</div>
</div>

---

## 1. Roofline

Pick a chip and a precision, then a workload. The operating point lands on the memory slope or the compute ceiling. The three things to try: batch-1 decode on any chip (memory-bound everywhere), the same decode at batch 256 (it crosses the ridge on most chips), and decode attention, which does not move at all with batch size.

<div class="lab" data-lab="roofline"><noscript><div class="lab-noscript">This lab needs JavaScript.</div></noscript></div>

## 2. Bit explorer

Type a number and see the bit pattern each format stores, the value it decodes back to, and the error. Click any bit to flip it. Good starting points: `0.1` in BF16 vs FP16, `65504` and `65520` in FP16, `448` and `500` in FP8 E4M3, and `2.5` in FP4 (it rounds to 2 because ties go to the even code).

<div class="lab" data-lab="bits"><noscript><div class="lab-noscript">This lab needs JavaScript.</div></noscript></div>

## 3. Will it fit?

The memory arithmetic behind every "out of memory" error: weights, KV cache, gradients, optimizer state, activations. Try Llama 3.1 70B on one H100 (it does not fit in BF16), then switch to INT4 weights; then raise the context to 128k at batch 8 and watch the KV cache overtake the weights.

<div class="lab" data-lab="fit"><noscript><div class="lab-noscript">This lab needs JavaScript.</div></noscript></div>

## 4. Systolic array

A 4×4 output-stationary array computing C = A·B. Step through it and notice the diagonal wavefront: on cycle *t*, processing element (*i*, *j*) consumes element *k* = *t* − *i* − *j*. The array needs 3N − 2 cycles for N cycles of useful work, which is why systolic machines want large, steady batches.

<div class="lab" data-lab="systolic"><noscript><div class="lab-noscript">This lab needs JavaScript.</div></noscript></div>

## 5. Wafer yield

A 300 mm wafer, a die size and a defect density. Defects land at random; any die with a defect is scrapped. Compare an 814 mm² die (H100-sized) with a 200 mm² chiplet at the same defect density and you have the economic case for chiplets in one picture.

<div class="lab" data-lab="wafer"><noscript><div class="lab-noscript">This lab needs JavaScript.</div></noscript></div>

## 6. 2:4 structured sparsity

Magnitude pruning under different patterns. Only 2:4 maps to NVIDIA's Sparse Tensor Cores, which read the compressed values plus a 2-bit index per kept value. Unstructured pruning keeps slightly more of the weights' energy but gives the hardware nothing to skip.

<div class="lab" data-lab="sparsity"><noscript><div class="lab-noscript">This lab needs JavaScript.</div></noscript></div>

## 7. Quantization error

Signal-to-quantization-noise for a synthetic tensor with optional outlier channels. Per-tensor scales collapse when outliers are present; per-group scales recover most of the loss for a fraction of a bit. The microscaling formats make the same trade in hardware: MXFP4 uses a power-of-two scale per 32 values, NVFP4 an FP8 scale per 16.

<div class="lab" data-lab="quant"><noscript><div class="lab-noscript">This lab needs JavaScript.</div></noscript></div>

## 8. Cache simulator

A small set-associative cache with LRU replacement. Walk an array, walk matrix columns or read at random, and watch every miss get classified as compulsory, capacity or conflict. The interesting case: columns of a matrix whose rows are exactly 4 KB apart, in a direct-mapped cache, then the same with one line of padding per row.

<div class="lab" data-lab="cache"><noscript><div class="lab-noscript">This lab needs JavaScript.</div></noscript></div>

## 9. Accelerator atlas

The full data set lives on its own page: [Accelerator Atlas](./atlas.md).

---

**Next:** [Accelerator Atlas](./atlas.md): every chip in this guide on one bandwidth-versus-compute chart.
