---
title: "Appendix G — Resources & Further Reading"
---

[← Back to Table of Contents](./README.md)

# Appendix G — Resources & Further Reading

This is a curated reading list for going deeper on every major topic in the guide. Each item is annotated with one line explaining why it earns its place here. Items are listed by name; URLs are included only when they are well-known canonical links unlikely to rot.

---

## Books

### Computer Architecture & Systems

**Hennessy & Patterson — *Computer Architecture: A Quantitative Approach* (Morgan Kaufmann)**
The definitive graduate-level textbook for everything from ISAs and pipelining to memory hierarchies and GPUs. Dense and comprehensive; use it as a reference alongside this guide. The 6th edition (2017) added a chapter on domain-specific architectures, with the TPU as its case study.

**Patterson & Hennessy — *Computer Organization and Design* (ARM or RISC-V edition, Morgan Kaufmann)**
The undergraduate companion to the above; gentler pacing with worked exercises. The RISC-V edition (5th/6th) is the best starting point if you are new to computer architecture.

**Harris & Harris — *Digital Design and Computer Architecture* (Morgan Kaufmann)**
Combines digital logic, Verilog/VHDL, and microarchitecture in one volume. Excellent for understanding RTL design before moving to synthesis and the full design flow.

**Patterson & Waterman — *The RISC-V Reader: An Open Architecture Atlas***
A short, readable primer on RISC-V; useful for understanding the ISA layer without the noise of x86's legacy complexity.

### GPU and Parallel Programming

**Kirk & Hwu — *Programming Massively Parallel Processors* (PMPP, 4th ed., Morgan Kaufmann)**
The standard CUDA programming textbook — memory hierarchy, kernel optimization, warp divergence, tensor cores. Essential reading for anyone writing GPU kernels.

**Sanders & Kandrot — *CUDA by Example* (Addison-Wesley)**
An older, gentler CUDA introduction. Good starting point if PMPP feels too dense initially.

### Hardware for Machine Learning

**Sze, Chen, Yang & Emer — *Efficient Processing of Deep Neural Networks* (Morgan & Claypool, 2020)**
The textbook on accelerator design for neural networks: dataflows, sparsity, reduced precision and how to evaluate an accelerator honestly. The best companion to Parts IV–VI of this guide.

**Jacob Austin et al. (Google DeepMind) — *How to Scale Your Model* (2025, free online)**
A modern, quantitative treatment of rooflines, TPU and GPU sharding, and LLM training and inference at scale. Reads like a sequel to Chapters 11, 28 and 29.

**Hugging Face — *The Ultra-Scale Playbook* (2025, free online)**
Training LLMs on GPU clusters, with measured results for data, tensor, pipeline, context and expert parallelism. The practical counterpart to Chapter 28.

**Chris Miller — *Chip War* (Scribner, 2022)**
The history and geopolitics of the semiconductor industry, from the transistor to export controls. Background for Chapter 33 and Appendix B.

### Electronics & Low-Level Foundations

**Horowitz & Hill — *The Art of Electronics* (3rd ed., Cambridge University Press)**
The canonical reference for analog and mixed-signal electronics. Most ML practitioners will not need it cover-to-cover, but the digital-electronics chapters and the treatment of noise are genuinely useful context.

**Weste & Harris — *CMOS VLSI Design* (4th ed., Addison-Wesley)**
Goes below the RTL abstraction into transistor-level circuit design, layout, and CMOS logic families. Relevant for Appendix A topics (fab, cells) and for understanding why FinFET changed everything.

---

## Online Courses & Lectures

**MIT 6.004 — Computation Structures**
MIT OpenCourseWare. Builds digital logic, assembly, and processor microarchitecture from first principles. One of the cleanest pedagogical treatments of the material in Chapters 2–4 of this guide. Free on OCW.

**MIT 6.5930/6.5931 — Hardware Architecture for Deep Learning (Sze and Emer; formerly 6.812/6.825)**
More advanced; focuses on the hardware/ML co-design topics of Parts III–VI of this guide. Lecture slides are available from MIT and are excellent.

**CMU 18-447 — Introduction to Computer Architecture**
A rigorous university course covering pipelining, out-of-order execution, caches, and memory systems. Lecture notes from Carnegie Mellon are freely accessible.

**Berkeley CS152 — Computer Architecture and Engineering**
The UC Berkeley graduate course on computer architecture. Strong on memory hierarchy and parallelism.

**Stanford CS149 — Parallel Computing**
Covers SIMD, SIMT, GPU programming model, parallel patterns, and memory consistency. Directly relevant to Chapters 8, 15, and 27. Course materials available at cs149.stanford.edu.

**NVIDIA Deep Learning Institute (DLI)**
NVIDIA's official training platform. The courses on CUDA, Nsight profiling, and Transformer optimization are practically oriented and well-maintained. Many are free or low-cost.

**GPU MODE (lectures and community)**
A community lecture series and Discord on writing fast GPU kernels: CUDA, Triton, CUTLASS, profiling and quantization kernels, often taught by the people who wrote the libraries.

**fast.ai — Practical Deep Learning for Coders**
Not a hardware course, but provides the ML context that anchors everything in Parts V–VI of this guide. Jeremy Howard's explanations of quantization and model optimization are excellent.

---

## YouTube Channels

**Asianometry**
The best single source for semiconductor industry history, geopolitics, and fab technology. Covers EUV lithography, TSMC's rise, CoWoS packaging, and the competitive landscape with excellent research depth.

**Branch Education**
Exceptional animated explainers of how electronics work physically — from transistors to SSDs to GPU architecture. The GPU deep-dive video is highly recommended as a companion to Chapter 15.

**Computerphile**
University of Nottingham academics explain CS concepts clearly. Good for Number Representation, caches, and computer architecture concepts at an accessible level.

**TechTechPotato (Ian Cutress)**
Deep technical analysis of chip announcements, die shots, and architecture changes. Ian Cutress (formerly of AnandTech) provides detailed professional-level breakdowns of each GPU and CPU generation.

**Chips and Cheese (YouTube)**
Companion video content to the Chips and Cheese blog; focuses on microarchitecture benchmarking and analysis.

---

## Blogs & Technical Sites

**SemiAnalysis (Dylan Patel)**
The highest-quality paid/free newsletter on semiconductor industry dynamics, AI hardware economics, and supply chain analysis. Essential for understanding the $/TOPS, packaging capacity, and datacenter build-out topics in Chapter 35.

**Chips and Cheese**
Deep microarchitecture analysis of CPU and GPU designs — cache structures, execution ports, memory subsystems. Highly technical and rigorous. Strong on comparing AMD, Intel, and ARM cores.

**WikiChip**
A detailed wiki of processor microarchitectures, process nodes, and die shots with annotated diagrams. Indispensable reference for specs and floorplan analysis.

**Real World Tech (David Kanter)**
Long-form microarchitecture analysis by one of the field's best independent analysts. The Broadwell, Zen, and Apple M1 analyses are landmark pieces.

**AnandTech Archives (Anand Lal Shimpi, Ian Cutress)**
Though the site stopped publishing in August 2024, its archive remains the definitive historical record of CPU, GPU, and SoC launches with deep architectural analysis. Archive accessible via web.

**TSMC Technology Symposium papers and IEDM proceedings**
Primary source material on process node advances, FinFET→GAA transitions, CoWoS generations, and 3D IC roadmaps. Publicly available and surprisingly readable for engineering professionals.

**NVIDIA Developer Blog and GTC session recordings**
First-party technical content on GPU architecture, CUDA optimization, Transformer Engine, and new hardware features. GTC talks from the H100 and B200 launches are excellent.

**Modal GPU Glossary**
A well-linked, plain-language glossary of GPU hardware and CUDA terms, from SMs and warps to tensor memory. A good companion to Chapter 15.

**MLCommons MLPerf results**
Audited training and inference benchmark results across vendors. The most reliable public source for real (not peak) throughput on current hardware.

**ARM Architecture Reference Manual**
The definitive specification for ARM64 (AArch64) ISA. Free to download from ARM's developer site. Useful for Chapter 5 topics.

---

## Landmark Papers

**"In-Datacenter Performance Analysis of a Tensor Processing Unit" — Jouppi et al., ISCA 2017**
Google's original TPU paper. Introduces the systolic array architecture, roofline analysis for inference, and the argument for domain-specific hardware. Directly relevant to Chapter 16. Published in ACM Digital Library; widely reproduced.

**"Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures" — Williams, Waterman, Patterson, 2009**
The paper that introduced the roofline model. Short and readable; foundational for Chapter 11 and any bandwidth-bound kernel analysis. Published in *Communications of the ACM*.

**"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" — Dao et al., NeurIPS 2022**
Demonstrates how understanding the GPU memory hierarchy (HBM vs SRAM) directly enables a 2–4× faster and memory-efficient attention implementation. The ideal example of hardware-aware algorithm design (relevant to Ch 11, 15, 29).

**"Training Compute-Optimal Large Language Models" (Chinchilla) — Hoffmann et al., DeepMind, 2022**
The scaling-laws paper that derived the optimal model size / token count ratio given a compute budget. Directly relevant to the training energy math in Appendix D.

**"The Hardware Lottery" — Sara Hooker, Communications of the ACM, 2021**
An essay arguing that research ideas succeed partly because they happen to match the hardware available at the time — not purely on merit. Thought-provoking context for the whole guide.

**"ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet) — Krizhevsky, Sutskever, Hinton, NeurIPS 2012**
The paper that started the GPU-for-deep-learning era by winning ImageNet by a large margin using GPU-trained CNNs. Historical context for Chapter 27.

**"Megatron-LM" — Shoeybi et al., 2019, and "ZeRO" — Rajbhandari et al., 2020**
The two papers behind tensor parallelism and sharded data parallelism; together they define the vocabulary of Chapter 28.

**"Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM) — Kwon et al., SOSP 2023**
Virtual memory for the KV cache, and the reason modern serving engines can batch so many requests (Ch 29).

**"AI and Memory Wall" — Gholami et al., IEEE Micro 2024**
Quantifies how compute has outgrown memory bandwidth across two decades of hardware; the data behind Chapter 11.

**DeepSeek-V3 Technical Report (2024) and "The Llama 3 Herd of Models" (Meta, 2024)**
Two unusually detailed accounts of frontier training runs: FP8 training, parallelism layouts, failure rates and hardware co-design (Ch 13, 25, 28).

**"Attention Is All You Need" — Vaswani et al., NeurIPS 2017**
The Transformer paper. Understanding the architecture — what the GEMM operations are, why attention is memory-bandwidth-heavy, what the KV-cache is — is prerequisite context for Chapters 28–30.

---

## Standards Bodies & Specifications

**OCP (Open Compute Project) — MX Number Format Specification**
The MX (microscaling) format standard (MXFP8, MXFP6, MXFP4, MXINT8) defining block-scaled low-precision formats for AI inference. Freely available at opencompute.org.

**JEDEC — HBM and DDR Standards**
JEDEC publishes the HBM standards (JESD235 for HBM/HBM2, JESD238 for HBM3, JESD270-4 for HBM4) and DDR5 (JESD79-5). Freely downloadable (registration required) at jedec.org. Authoritative source for the bandwidth and latency numbers in Chapter 9.

**PCI-SIG — PCIe Specification**
The authoritative PCIe specification (4.0 through 7.0; 8.0 in development). The base spec and CEM spec are available at pcisig.com for member download. Chapter 12 bandwidth numbers come from here.

**UCIe Consortium — UCIe Specification 1.0 / 2.0 / 3.0**
The die-to-die interconnect standard described in Appendix E. Available at uciexpress.org. Defines bump pitch, electrical specs, and protocol layers.

**CXL Consortium — CXL Specification**
The Compute Express Link standard enabling cache-coherent memory expansion (Ch 12). Available at computeexpresslink.org.

---

## Tools to Explore

**Godbolt / Compiler Explorer (godbolt.org)**
Interactive web tool to see how C/C++/Rust/Fortran compiles to assembly for x86, ARM, RISC-V, and more. Essential for understanding ISA-level decisions (Ch 5, 6). Free.

**Verilator**
Open-source Verilog/SystemVerilog simulator that compiles RTL to fast C++ models. The standard open-source tool for learning RTL simulation (Ch 32). Free, actively maintained.

**Yosys**
Open-source RTL synthesis tool. Combined with Verilator, lets you run the synthesis step of the design flow on real Verilog (Ch 31). Free, widely used in academia.

**OpenROAD / OpenROAD-flow-scripts**
Open-source place-and-route and physical design flow. Supports synthesis → floorplan → PnR → timing → GDSII on a standard-cell library. The closest open tool to what commercial EDA suites do (Ch 31). Free at theopenroadproject.org.

**NVIDIA Nsight Systems / Nsight Compute**
NVIDIA's official profiling tools. Nsight Systems gives timeline traces; Nsight Compute gives per-kernel memory/compute analysis (roofline, occupancy, stall reasons). Free with CUDA toolkit. Essential for any GPU optimization work (links to Ch 11, 15, 29).

**PyTorch Profiler (torch.profiler)**
Built-in PyTorch tool for tracing kernel execution, memory allocation, and communication timing in training/inference. Free; outputs Chrome trace format viewable in TensorBoard or Perfetto. Directly relevant to the optimization workflow of Parts V–VI.

**Triton**
OpenAI's Python-embedded language for writing GPU kernels; what `torch.compile` generates for many fused operations. The fastest way to experiment with the memory-hierarchy ideas of Chapters 11 and 29.

**CACTI (HP Labs)**
A cache/memory modeling tool for estimating SRAM area, power, and access time at different process nodes. Useful for understanding the cache design tradeoffs in Ch 9 and Ch 14. Free.

---

## A Final Note

You have now read — or are equipped to read — every layer of the stack, from the quantum mechanics of a p-n junction to the grid-scale energy challenge of a frontier training run. The hardware landscape changes fast: new process nodes every 2–3 years, new GPU generations annually, new packaging technologies in continuous development. But first principles do not change. The transistor, the memory hierarchy, the bandwidth-compute tradeoff, and the economics of yield are the permanent lenses through which every new announcement should be interpreted.

The resources above will take you deeper in any direction. The field rewards people who understand both the silicon and the software — and there are very few of them.

You now have the full picture — from sand to supercomputers. Build something with it.

---

**Next:** [Appendix H — The Life of a Running Computer](./appendix_h_how_a_computer_runs.md) — From the power button to the idle loop: what the hardware is doing while "nothing" happens.

[← Back to Table of Contents](./README.md)
