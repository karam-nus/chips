---
title: "Accelerator Atlas"
---

[← Back to Table of Contents](./README.md)

# Accelerator Atlas

> *"In God we trust. All others must bring data."* — attributed to W. Edwards Deming

## Overview

Every accelerator this guide discusses, on one chart and in one table. The x-axis is memory bandwidth, the y-axis is dense tensor throughput, and the dashed diagonals are lines of constant **ridge point**: the arithmetic intensity (FLOP per byte) an operation needs before it stops waiting on memory ([Chapter 11](./11_memory_wall_and_bandwidth.md)). Read the chart from bottom-left to top-right and the history of the last decade is visible: compute has climbed faster than bandwidth, so each new generation lands on a higher ridge-point line than the one before it.

<div class="lab" data-lab="atlas"><noscript><div class="lab-noscript">The atlas needs JavaScript. The raw data is in <a href="https://github.com/karam-nus/chips/blob/main/assets/data/accelerators.json">assets/data/accelerators.json</a>.</div></noscript></div>

<div class="callout note">
<span class="callout-title">How to read the numbers</span>
<p>All throughput figures are <strong>dense</strong>: no 2:4 or 16:4 structured-sparsity bonus, which vendors often fold into headline numbers. "Lowest supported precision" plots FP4 where the chip has it, else FP8, else BF16, so a newer chip's dot also reflects its new datatype. Status <em>ramping</em> means production shipments have begun but volume is still building; <em>announced</em> figures are vendor projections and are marked as such in each chip's notes. Hover a dot or a table row for caveats; each chip name links to its primary source.</p>
</div>

## What the chart says

| Observation | Evidence on the chart | Why it matters for you |
|---|---|---|
| **The ridge keeps rising** | V100's ridge is ~140 FLOP/B (FP16); H100's is ~590 (FP8); B300's FP4 ridge is ~1900 | Low-batch inference gets *relatively* slower on every new chip unless you cut bytes ([Ch 25](./25_quantization_hardware_view.md)) |
| **Bandwidth comes in steps, not slopes** | HBM2 → HBM2e → HBM3 → HBM3e → HBM4 are visible as jumps along x | Memory generation, not the GPU's name, sets your decode speed ([Ch 9](./09_memory_types.md)) |
| **Capacity diverged by vendor** | AMD leads on GB per package (192 → 256 → 288 → 432 GB) | Fewer GPUs per model replica means less tensor-parallel communication ([Ch 28](./28_multi_gpu_and_distributed_training.md)) |
| **SRAM-only designs are off the scale** | Tick "include SRAM-only" and Cerebras and Groq jump two orders of magnitude right | On-chip bandwidth is enormous, capacity is tiny; models are spread over many chips ([Ch 21](./21_asics.md)) |
| **Datatype is a hardware feature** | FP4 appears only from Blackwell, MI355X, TPU 8, and later | A format the silicon cannot multiply natively saves memory but not math ([Ch 24](./24_datatypes_and_silicon.md)) |

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>If the ridge point doubles every generation but your model's arithmetic intensity is fixed by its batch size and precision, which of those two numbers will you change first when you upgrade, and what does each cost you?</p>
</div>

## Maintaining the data

The atlas, the roofline lab, the "will it fit?" lab and the generated figures all read one file: [`assets/data/accelerators.json`](https://github.com/karam-nus/chips/blob/main/assets/data/accelerators.json). To add a chip, append one object with its primary source in `source` and any caveat in `notes`; nothing else needs to change.

---

**Next:** [Appendix A — How Chips Are Made](./appendix_a_how_chips_are_made.md): from sand to a packaged die, with the yield lab.
