---
title: "Appendix F — Glossary"
---

[← Back to Table of Contents](./README.md)

# Appendix F — Glossary

Every acronym, concept, and term that appears across the guide, defined in one place. Entries are organized alphabetically, 1–2 sentences each. Chapter cross-links point you to the deeper treatment.

---

## #

| Term | Definition |
|---|---|
| **2:4 sparsity** | A structured sparsity pattern where exactly 2 of every 4 weights are nonzero. NVIDIA Tensor Cores support hardware-accelerated 2:4 sparse matrix multiplication at near-dense throughput (Ch 26). |
| **3D stacking** | Placing multiple active dies directly on top of each other, connected by TSVs or hybrid bonds, to achieve much higher interconnect density and bandwidth than side-by-side 2D integration (Appendix E). |
| **2.5D packaging** | Mounting multiple dies side-by-side on a passive silicon interposer, which provides fine-pitch wiring between dies without fully stacking them. CoWoS is TSMC's 2.5D technology (Appendix E). |

---

## A

| Term | Definition |
|---|---|
| **ALU** | Arithmetic Logic Unit. The combinational block inside a CPU/GPU core that executes integer arithmetic and bitwise logic operations (Ch 2, Ch 4). |
| **ASIC** | Application-Specific Integrated Circuit. A chip designed for one task (e.g., a TPU, a Bitcoin miner, a radar processor). Maximum efficiency for that task; no flexibility (Ch 21). |
| **AVX / AVX-512** | Advanced Vector Extensions. Intel's SIMD ISA extension sets; AVX operates on 256-bit registers (8× FP32), AVX-512 on 512-bit (16× FP32). The main vector path on x86 CPUs (Ch 5, Ch 8). |
| **Activity factor ($\alpha$)** | In the dynamic power formula $P = \alpha C V^2 f$, the fraction of transistors switching per clock cycle. Clock-gating idle units drives $\alpha$ toward zero (Appendix D). |

---

## B

| Term | Definition |
|---|---|
| **BF16** | Brain Float 16. A 16-bit floating-point format with the same 8-bit exponent as FP32 but only 7 mantissa bits. Preserves FP32's dynamic range; default training dtype on TPUs and NVIDIA Ampere+ (Ch 23). |
| **BIST** | Built-In Self-Test. Logic embedded on a die that can test memories or logic blocks autonomously, without external test equipment (Ch 32). |
| **Bandwidth** | The rate at which data can be moved, measured in GB/s or TB/s. Memory bandwidth — not FLOPS — is usually the binding constraint for AI inference (Ch 11). |
| **Bump pitch** | The center-to-center spacing between solder bumps or copper pads on a die or interposer. Finer pitch enables more connections and higher bandwidth density (Appendix E). |

---

## C

| Term | Definition |
|---|---|
| **CISC** | Complex Instruction Set Computer. An ISA philosophy (e.g., x86) where individual instructions can perform multi-step operations (memory load + ALU + store). Historically meant to simplify compilers; microarchitecture translates to RISC-like micro-ops internally (Ch 5). |
| **CMOS** | Complementary Metal-Oxide-Semiconductor. The standard digital logic family using paired NMOS+PMOS transistors. Key property: almost no static power path — power is consumed mainly during switching (Ch 1). |
| **CoWoS** | Chip-on-Wafer-on-Substrate. TSMC's 2.5D packaging technology placing compute die + HBM stacks on a silicon interposer before mounting on an organic substrate. Used in H100, A100, MI250 (Appendix E). |
| **CUDA** | Compute Unified Device Architecture. NVIDIA's parallel computing platform and programming model, allowing GPU kernels to be written in C/C++/Fortran-like syntax (Ch 6, Ch 15). |
| **CXL** | Compute Express Link. An open interconnect standard over PCIe 5/6 physical layer enabling cache-coherent memory expansion and CPU-to-accelerator communication (Ch 12). |
| **Cache coherence** | The guarantee that all processors in a multicore system see a consistent view of shared memory. Implemented via protocols such as MESI (Ch 10). |
| **Chiplet** | A smaller die designed to be assembled with others in a single package, replacing one large monolithic die. Improves yield and enables mixing of process nodes (Ch 22, Appendix E). |
| **Clock gating** | Disabling the clock signal to an idle block to eliminate its dynamic power consumption without fully powering it down (Appendix D). |

---

## D

| Term | Definition |
|---|---|
| **DDR** | Double Data Rate. A DRAM interface standard that transfers data on both the rising and falling clock edge. DDR5 bandwidth ~50–51.2 GB/s per channel (Ch 9). |
| **Dennard scaling** | The observation (R. Dennard, 1974) that as transistors shrink, their power density stays constant — allowing both more speed and lower energy per operation. It broke around 2005 when leakage made voltage scaling infeasible, forcing the pivot to parallelism (Ch 1). |
| **DFT** | Design for Testability. Techniques added to a design — scan chains, BIST, JTAG interfaces — to make it easier to test after manufacture (Ch 32). |
| **DRAM** | Dynamic Random-Access Memory. Main memory technology using one capacitor + one transistor per cell. Requires periodic refresh; dense and cheap but slower than SRAM. Includes DDR, GDDR, HBM (Ch 9). |
| **DVFS** | Dynamic Voltage and Frequency Scaling. A runtime technique that adjusts supply voltage and clock frequency together to trade performance for power based on workload demand (Appendix D). |
| **Dynamic power** | Power consumed when transistors switch: $P = \alpha C V^2 f$. The dominant power component in active chips (Ch 1, Appendix D). |
| **Dataflow** | In the context of systolic arrays and AI accelerators, the pattern by which data (weights, activations) moves through the compute array — output-stationary, weight-stationary, or input-stationary (Ch 16). |

---

## E

| Term | Definition |
|---|---|
| **ECC** | Error-Correcting Code. Memory encoding that detects and corrects bit errors (typically single-bit correct / double-bit detect for SECDED). Essential for long AI training runs (Ch 13). |
| **EUV** | Extreme Ultraviolet lithography. Uses 13.5 nm wavelength light (vs 193 nm for deep-UV) to print sub-7 nm features. Required for N5 and below nodes; made by ASML exclusively (Appendix A). |

---

## F

| Term | Definition |
|---|---|
| **FinFET** | Fin Field-Effect Transistor. A 3D transistor geometry where the channel is a thin vertical "fin" of silicon, giving the gate control on three sides. Replaced planar MOSFETs at 22 nm (Intel) and 16 nm (TSMC/Samsung). Succeeded by GAA at 3 nm (Ch 34, Appendix A). |
| **FLOP** | Floating-Point Operation (usually a multiply-add = 2 FLOPs). Used to measure computational throughput: a100 = 312 TFLOPS (FP16), H100 = 989 TFLOPS (FP16 dense) (Ch 11). |
| **FP8** | 8-bit floating-point. Two common formats: E4M3 (wider range) and E5M2 (more exponent bits). Supported natively on NVIDIA H100 Transformer Engine; enables ~2× throughput over FP16 (Ch 23). |
| **FP4 / NVFP4** | 4-bit floating-point. NVIDIA Blackwell introduces hardware FP4 tensor cores (NVFP4: E2M1 with shared scale). Extreme density at the cost of reduced precision (Ch 23). |
| **FPGA** | Field-Programmable Gate Array. A chip with reconfigurable logic (LUTs, flip-flops, DSP blocks, BRAMs). Can be reprogrammed after manufacture to implement any logic function (Ch 20). |
| **FSDP** | Fully Sharded Data Parallel. A PyTorch distributed training strategy (equivalent to ZeRO-3) that shards optimizer states, gradients, and parameters across all GPUs (Ch 28). |

---

## G

| Term | Definition |
|---|---|
| **GAA** | Gate-All-Around. The next-generation transistor geometry after FinFET, where the gate wraps entirely around a nanosheet or nanowire channel. TSMC N2 and Samsung SF3 use GAA. Improves electrostatic control, reducing leakage (Ch 34, Appendix A). |
| **GDDR** | Graphics DDR. High-bandwidth DRAM on the PCB of a GPU, connected via wide parallel bus. GDDR6X provides ~1 TB/s for a mid-range GPU; lower latency than HBM but lower total bandwidth (Ch 9). |
| **GEMM** | General Matrix Multiply. The $C = A \times B$ operation that underlies the vast majority of neural-network computation. Tensor cores and systolic arrays are hardware purpose-built for GEMM (Ch 16, Ch 27). |

---

## H

| Term | Definition |
|---|---|
| **HBM** | High Bandwidth Memory. A 3D-stacked DRAM technology using TSVs, mounted on a silicon interposer adjacent to the compute die via CoWoS. HBM3 delivers ~819 GB/s per stack; H100 has 5 stacks = ~3.35 TB/s (Ch 9, Appendix E). |
| **HLS** | High-Level Synthesis. Tools (Vitis HLS, Catapult) that compile C/C++ or Python into RTL for FPGA or ASIC implementation, bypassing manual Verilog/VHDL (Ch 6, Ch 20). |
| **Hybrid bonding** | A die-to-die bonding technique that directly fuses dielectric surfaces and copper pads (no solder bumps) at ~1–10 µm pitch. Used in AMD 3D V-Cache and TSMC SoIC (Appendix E). |

---

## I

| Term | Definition |
|---|---|
| **ILP** | Instruction-Level Parallelism. The degree to which instructions in a program can be executed simultaneously (out-of-order execution, superscalar dispatch, speculative execution) (Ch 8). |
| **INT8** | 8-bit signed integer. The most common quantization target for inference: ~4× less memory than FP32, ~2× over FP16. NVIDIA tensor cores support INT8 GEMM natively at 2× FP16 throughput (Ch 25). |
| **ISA** | Instruction Set Architecture. The contract between hardware and software defining the available instructions, registers, addressing modes, and calling conventions. Examples: x86-64, ARM64, RISC-V (Ch 5). |
| **Interposer** | A passive silicon (or organic) substrate placed between dies in a 2.5D package, providing fine-pitch routing between dies and to the package substrate (Appendix E). |

---

## K

| Term | Definition |
|---|---|
| **KV-cache** | Key-Value cache. In autoregressive Transformer inference, the stored K and V tensors from previously generated tokens, avoiding recomputation. Size grows as $O(\text{layers} \times \text{heads} \times \text{seq\_len} \times \text{d\_head})$ — often the memory bottleneck (Ch 29). |

---

## L

| Term | Definition |
|---|---|
| **Leakage power** | Static power consumed by transistors that are nominally "off" due to subthreshold and gate-oxide leakage currents. Grew dramatically at nodes below 65 nm and helped end Dennard scaling (Ch 1, Appendix D). |
| **LUT** | Look-Up Table. The basic reconfigurable compute element in an FPGA — typically a 4–6 input truth table stored in SRAM. Any combinational logic function of $n$ inputs can be implemented with a $2^n$-entry LUT (Ch 20). |

---

## M

| Term | Definition |
|---|---|
| **MAC** | Multiply-Accumulate. The fundamental operation of neural networks: $\text{acc} \mathrel{+}= a \times b$. One MAC = 2 FLOPs. Tensor cores and systolic arrays perform hundreds of MACs per clock (Ch 2, Ch 16). |
| **MMU** | Memory Management Unit. Hardware that translates virtual addresses to physical addresses using page tables, provides memory protection, and manages the TLB (Ch 10). |
| **MoE** | Mixture of Experts. A neural-network architecture that routes each token through only a sparse subset of "expert" sub-networks, decoupling model capacity from per-token compute (Ch 28). |
| **Moore's Law** | The empirical trend (Gordon Moore, 1965) that transistor count per chip roughly doubles every ~2 years. An economic and engineering roadmap, not a physical law; still loosely holds but with diminishing returns (Ch 1). |
| **MXU** | Matrix Multiply Unit. Google's name for the systolic array inside a TPU that performs large matrix multiplications in a wave of multiply-accumulate operations (Ch 16). |
| **MXFP / MX formats** | Microscaling floating-point. OCP MX specification (MXFP8, MXFP6, MXFP4) adds a per-block shared exponent scale factor to low-precision formats, reducing quantization error (Ch 23). |

---

## N

| Term | Definition |
|---|---|
| **NCCL** | NVIDIA Collective Communications Library. Provides optimized collective operations (AllReduce, AllGather, ReduceScatter) over NVLink and network fabrics for multi-GPU training (Ch 28). |
| **NoC** | Network-on-Chip. An on-chip communication infrastructure (routers, switches, links) that replaces ad-hoc bus wiring for complex SoCs with many IP blocks (Ch 12). |
| **NPU** | Neural Processing Unit. A dedicated hardware block (in mobile SoCs, edge devices) optimized for inference workloads — INT8/FP16 GEMM at low power. Examples: Apple Neural Engine, Qualcomm Hexagon (Ch 17). |
| **NVLink** | NVIDIA's proprietary high-speed GPU-to-GPU interconnect. NVLink 4 (H100) provides 900 GB/s bidirectional per GPU; used in NVSwitch-based DGX/HGX nodes (Ch 12). |
| **NUMA** | Non-Uniform Memory Access. A multiprocessor memory architecture where access latency/bandwidth depends on which processor the memory is attached to (Ch 10). |

---

## O

| Term | Definition |
|---|---|
| **OoO** | Out-of-Order execution. A superscalar CPU technique that reorders instruction execution to hide memory latency, using a reorder buffer (ROB) to commit results in program order (Ch 14). |

---

## P

| Term | Definition |
|---|---|
| **PCIe** | Peripheral Component Interconnect Express. The standard CPU-to-accelerator interconnect. PCIe 5.0 x16 = 128 GB/s bidirectional; PCIe 6.0 doubles that. Often the bandwidth bottleneck between CPU host and GPU (Ch 12). |
| **PnR** | Place and Route. The EDA step that assigns synthesized logic cells to physical locations on the die and connects them with metal wiring, meeting timing and DRC constraints (Ch 31). |
| **PPA** | Power, Performance, Area. The three-way trade-off optimized in chip design. Every architectural decision (wider datapath, larger cache, more pipeline stages) shifts this triangle (Ch 31). |
| **PUE** | Power Usage Effectiveness. Ratio of total datacenter facility power to IT equipment power. A PUE of 1.2 means 20% overhead from cooling, networking, UPS (Appendix D). |
| **Power gating** | Cutting supply voltage to an idle block entirely, eliminating both dynamic and leakage power. Slower to restore than clock gating but saves more power at deep idle (Appendix D). |
| **Process node** | A named generation of semiconductor manufacturing technology (e.g., TSMC N5 "5 nm"). The number was once a physical feature size; today it is a marketing label for a tech generation characterized by transistor density, power, and speed (Ch 1, Ch 34). |

---

## R

| Term | Definition |
|---|---|
| **RAS** | Reliability, Availability, Serviceability. A set of features (ECC, redundant power, hot-swap, health monitoring) designed to keep a system running correctly and recoverable (Ch 13). |
| **RISC** | Reduced Instruction Set Computer. An ISA philosophy (e.g., ARM, RISC-V) favoring simple, fixed-width instructions, large register files, and load-store memory access. Easier to pipeline and more power-efficient (Ch 5). |
| **RoCE** | RDMA over Converged Ethernet. A networking protocol enabling remote direct memory access (RDMA) over standard Ethernet, used in large GPU cluster fabrics (Ch 12). |
| **Roofline model** | An analytical performance model plotting compute throughput vs. arithmetic intensity (FLOPs/byte). Shows whether a kernel is memory-bandwidth-bound or compute-bound. Essential for quantization analysis (Ch 11). |
| **RTL** | Register-Transfer Level. An abstraction of digital hardware as registers and combinational logic between them, described in Verilog or VHDL. The primary design representation before synthesis (Ch 6, Ch 31). |
| **Reticle limit** | The maximum area of a single die (~858 mm²) set by the exposure field of an EUV lithography scanner. Dies larger than this require multi-die / chiplet approaches (Appendix E). |

---

## S

| Term | Definition |
|---|---|
| **SerDes** | Serializer/Deserializer. High-speed analog I/O circuits that serialize parallel data into a high-speed serial stream for off-chip transmission (e.g., PCIe, NVLink, ethernet) and deserialize it on the other end (Ch 12). |
| **SIMD** | Single Instruction, Multiple Data. A parallelism model where one instruction operates on a vector of data elements simultaneously. AVX, ARM NEON, and SVE are CPU SIMD; GPU SIMT is the GPU analogue (Ch 8). |
| **SIMT** | Single Instruction, Multiple Threads. NVIDIA's GPU execution model where groups of 32 threads (a warp) execute the same instruction in lockstep, each on different data (Ch 15). |
| **SM** | Streaming Multiprocessor. The fundamental compute unit of an NVIDIA GPU. An H100 has 132 SMs, each with tensor cores, CUDA cores, shared memory, and warp schedulers (Ch 15). |
| **SoC** | System-on-Chip. An IC that integrates CPU, GPU/NPU, memory controllers, I/O, and other subsystems on one die. Apple M-series and Qualcomm Snapdragon are SoCs (Ch 22). |
| **SRAM** | Static Random-Access Memory. Fast, low-latency memory (6T cell) used for caches and register files on-chip. ~100× faster than DRAM but ~30× more expensive per bit and ~30× less dense (Ch 9). |
| **SVA** | SystemVerilog Assertions. A formal specification language embedded in SystemVerilog for expressing design properties that verification tools check exhaustively or in simulation (Ch 32). |
| **Systolic array** | A 2D array of processing elements (PEs) where data flows rhythmically between neighbors like a heartbeat. Each PE performs a MAC. The Google TPU's MXU is a 128×128 or 256×256 systolic array (Ch 16). |
| **Subthreshold leakage** | Current that flows between source and drain of a MOSFET even when the gate voltage is below the threshold voltage. Grew with each process node shrink and is a major source of idle power (Ch 1, Appendix D). |
| **Scan chain** | A DFT technique that connects all flip-flops in a chip into a shift register for testing, allowing any state to be loaded and observed serially (Ch 32). |

---

## T

| Term | Definition |
|---|---|
| **TDP** | Thermal Design Power. The maximum sustained power (in watts) the cooling solution must be rated to handle for a chip. Not a hard cap; exceeding it triggers thermal throttling (Appendix D). |
| **TLB** | Translation Lookaside Buffer. A hardware cache of recent virtual-to-physical address translations, avoiding full page-table walks on every memory access (Ch 10). |
| **TOPS** | Tera-Operations Per Second. A measure of integer/fixed-point throughput, commonly used for NPUs and inference accelerators (e.g., 100 TOPS INT8). Compare TFLOPS for floating-point (Ch 17). |
| **TPU** | Tensor Processing Unit. Google's ASIC for neural network inference and training, built around a large systolic array (MXU) for matrix multiplication (Ch 16). |
| **TSV** | Through-Silicon Via. A vertical electrical connection etched through a silicon die, enabling 3D stacking. TSVs connect DRAM layers in HBM and logic dies in SoIC (Appendix E). |
| **TDP** | See above. |
| **Tensor core** | Specialized compute units on NVIDIA GPUs (Volta+) that perform mixed-precision matrix-multiply-accumulate on small tiles (e.g., 16×16×16 FP16×FP16→FP32) in a single clock cycle (Ch 15, Ch 24). |
| **Thermal throttling** | Automatic reduction of CPU/GPU clock frequency and voltage when junction temperature approaches the maximum, protecting the chip at the cost of performance (Ch 13, Appendix D). |
| **Two's complement** | The near-universal signed integer encoding where the most significant bit has weight $-2^{n-1}$. Subtraction is addition of the two's complement; there is no negative zero (Ch 3). |

---

## U

| Term | Definition |
|---|---|
| **UCIe** | Universal Chiplet Interconnect Express. An open industry standard (2022) for die-to-die interconnects, defining physical bump geometry, electrical specs, and protocol mapping (PCIe/CXL). Enables multi-vendor chiplet integration (Ch 35, Appendix E). |
| **UVM** | Universal Verification Methodology. A standardized SystemVerilog framework for building reusable, layered testbenches with constrained-random stimulus generation and coverage collection (Ch 32). |

---

## V

| Term | Definition |
|---|---|
| **VLIW** | Very Long Instruction Word. A CPU/DSP architecture where the compiler explicitly packs multiple independent operations into one wide instruction word. Places parallelism burden on the compiler rather than hardware (Ch 5, Ch 18). |
| **VRM** | Voltage Regulator Module. A switching power supply on the PCB (or in-package) that steps board-level voltage (12–48 V) down to the die's operating voltage (0.7–1.0 V). Modern GPU cards use multi-phase VRMs (Appendix D). |
| **von Neumann architecture** | The classical stored-program computer model (Burks, Goldstine, von Neumann, 1945): a single shared memory holds both instructions and data, accessed by a CPU over a bus. The von Neumann bottleneck is the bandwidth limit of that single bus (Ch 4). |

---

## W

| Term | Definition |
|---|---|
| **Warp** | In NVIDIA GPU terminology, a group of 32 threads that execute in lockstep (SIMT). The warp is the scheduling unit; all 32 threads share one instruction fetch/decode (Ch 15). |

---

## Y

| Term | Definition |
|---|---|
| **Yield** | The fraction of dies on a wafer that pass all tests and are functional. Yield falls with larger die area, higher defect density, and tighter design rules. The central economic variable in chip manufacturing (Appendix A, Appendix E). |

---

## Z

| Term | Definition |
|---|---|
| **ZeRO** | Zero Redundancy Optimizer (Microsoft DeepSpeed). Partitions optimizer states, gradients, and parameters across data-parallel ranks to reduce per-GPU memory. ZeRO-3 = FSDP (Ch 28). |

---

*Next: [Appendix G — Resources & Further Reading](./appendix_g_resources.md), a curated guide to books, courses, papers, and tools for going deeper.*

[← Back to Table of Contents](./README.md)
