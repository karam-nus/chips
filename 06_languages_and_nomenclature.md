---
title: "Chapter 6 — Languages & Nomenclature of the Stack"
---

[← Back to Table of Contents](./README.md)

# Chapter 6 — Languages & Nomenclature of the Stack

When a colleague says "I wrote a custom kernel," do they mean a Linux kernel module or a CUDA function? When a hardware engineer says "the RTL passed timing," what does "RTL" mean? When you read a conference paper claiming "we implemented the attention operator in Triton," what layer of the stack is that? The hardware/software world is awash in overloaded terminology — words like *kernel*, *microcode*, *driver*, *firmware*, and *assembler* each mean something precise, but that precision is rarely explained.

This chapter is the demystification chapter. We build the full vocabulary of the stack from silicon up to Python, show real (short) authentic code at every layer, and explain exactly what happens to your PyTorch line as it descends — from Python bytecode through the framework, the compiler, the driver, the ISA, the microarchitecture, and finally to switching transistors.

> **The one-sentence version:** Every layer of the hardware/software stack is a contract that hides complexity below it; knowing which layer you're at tells you what you can control, how portable your work is, and how much it will cost.

---

## The Full Stack, Visualized

<div class="diagram">
<div class="diagram-title">The Complete Software/Hardware Stack</div>
<div class="layer-stack">
  <div class="layer">Python / PyTorch / JAX / user model  ← you are here</div>
  <div class="layer">ML Framework internals (autograd, dispatcher, graph capture)</div>
  <div class="layer">Compiler / kernel libraries (torch.compile / XLA / TVM / cuBLAS / oneDNN)</div>
  <div class="layer">CUDA / HIP / Triton / OpenCL / SYCL  — accelerator programming layer</div>
  <div class="layer">GPU/CPU Driver + Runtime (libcuda.so, ROCm runtime, Metal)</div>
  <div class="layer">OS Kernel (Linux, macOS XNU) — resource management, memory, scheduling</div>
  <div class="layer">ISA / Machine Code — binary the CPU/GPU actually executes (Chapter 5)</div>
  <div class="layer">Microarchitecture — out-of-order execution, caches, pipelines (Chapter 4)</div>
  <div class="layer">RTL (Register-Transfer Level) — the synthesizable hardware description</div>
  <div class="layer">Standard cells / logic gates — from the foundry's library</div>
  <div class="layer">Transistors — MOSFETs in silicon (Chapter 1)</div>
</div>
</div>

We will walk this stack from the bottom up, then re-examine the GPU-specific column, then close with the definitive glossary table.

---

## Part 1: From Silicon Up — the Software Side

### Bare Metal

**Bare metal** means code that runs **with no operating system** — the CPU executes your code directly after power-on, with nothing between your program and the hardware. Boot loaders, embedded firmware, and the first-stage of an OS boot process all run bare metal.

Bare-metal code must:
- Set up the stack pointer manually
- Initialize DRAM (call the memory controller)
- Configure interrupt vectors
- Not call `malloc`, `printf`, or any OS service — they don't exist

```c
/* bare_metal_startup.c — runs before any OS on an ARM Cortex-M MCU */
/* The linker places _start at the reset vector (address 0x00000004)  */

extern int main(void);
extern unsigned int _stack_top;        /* from linker script */

void _start(void) {
    /* Zero the BSS segment (uninitialized global variables) */
    extern unsigned int _bss_start, _bss_end;
    for (unsigned int *p = &_bss_start; p < &_bss_end; p++) *p = 0;

    /* Copy initialized data from flash to RAM */
    extern unsigned int _data_lma, _data_start, _data_end;
    unsigned int *src = &_data_lma, *dst = &_data_start;
    while (dst < &_data_end) *dst++ = *src++;

    main();         /* call user code */
    while (1) {}    /* halt if main returns */
}
```

There is no `stdlib`, no heap allocator, no scheduler — just your code and the registers. Many edge AI chips initially run their inference runtime bare metal for the lowest latency and power.

### Firmware and BIOS/UEFI

**Firmware** is software permanently stored in non-volatile memory (flash ROM, EEPROM) on a hardware device. It is the first software a processor executes. Examples:

- The **BIOS/UEFI** on a PC motherboard: initializes CPU, DRAM, PCIe, then loads the OS
- A **GPU's firmware**: the microcontroller that initializes the GPU, negotiates PCIe, responds to driver commands
- An **SSD controller firmware**: manages NAND wear-leveling, error correction, the NVMe protocol

**BIOS** (Basic Input/Output System) was the legacy 16-bit standard (1970s–2010s). **UEFI** (Unified Extensible Firmware Interface) replaced it: full 64-bit execution, a rich pre-OS environment, secure boot (cryptographic verification of the OS loader), and a standard API.

When your GPU hangs and the driver reports "firmware mismatch" or "GSP firmware error," you're seeing the CPU-side driver failing to communicate with the GPU's on-chip firmware.

### Microcode

**Microcode** is the internal, writable instruction table that maps **complex ISA instructions → simpler micro-operations** inside a CISC processor. It exists one level *below* assembly but one level *above* the logic gates — in a ROM (or SRAM with updates) inside the CPU itself.

```text
CPU internal layers (Intel x86-64):

Assembly/machine code (ISA level):
  FDIV  ST(0), ST(1)     ; floating-point divide

                ↓  hardware decode

Microcode ROM lookup:
  µop0: FMUL_APPROX  →  initial reciprocal estimate
  µop1: FNMA_CORR    →  Newton-Raphson refinement iteration 1
  µop2: FNMA_CORR    →  Newton-Raphson refinement iteration 2
  µop3: FMUL_FINAL   →  multiply numerator by refined reciprocal

                ↓  these µops execute in the out-of-order engine
```

Microcode is security-critical: CPU vulnerabilities like **Spectre/Meltdown (2018)** and **REPTAR (2023)** were partially patched by microcode updates loaded by the OS at boot (`/lib/firmware/intel-ucode/` on Linux). Microcode updates can *slow* a chip (adding checks) or fix silent data corruption bugs — which is why training cluster operators must stay current.

### Machine Code vs Assembly

**Machine code** is the binary the CPU directly executes: a sequence of bytes, each instruction encoded according to the ISA. It's not human-readable.

```text
Machine code bytes (x86-64, hexadecimal):
  48 01 D0              ; ADD rax, rdx
  C5 FC 58 05 00 00 00  ; VADDPS ymm0, ymm0, [rip+offset]  (AVX2)
  66 0F EF C0           ; PXOR xmm0, xmm0  (SSE2 zero a register)
```

**Assembly language** is a human-readable, 1-to-1 symbolic representation of machine code. Each assembly mnemonic corresponds to exactly one machine-code encoding. An **assembler** converts assembly text → machine code bytes.

```asm
; === x86-64 assembly (Intel syntax, NASM) ===
; Function: dot_product_f32(float* a, float* b, int n) → float
; Uses SSE2: 4-wide float32 SIMD
section .text
global dot_product_f32

dot_product_f32:
    ; rdi=a, rsi=b, edx=n
    xorps   xmm0, xmm0          ; accumulator = 0.0
    xor     ecx, ecx             ; i = 0
.loop:
    cmp     ecx, edx
    jge     .done
    movups  xmm1, [rdi + rcx*4]  ; load 4 floats from a[i..i+3]
    movups  xmm2, [rsi + rcx*4]  ; load 4 floats from b[i..i+3]
    mulps   xmm1, xmm2           ; element-wise multiply
    addps   xmm0, xmm1           ; accumulate
    add     ecx, 4               ; i += 4
    jmp     .loop
.done:
    ; horizontal add: sum xmm0[0]+[1]+[2]+[3]
    haddps  xmm0, xmm0
    haddps  xmm0, xmm0
    ret                          ; return value in xmm0 (System V ABI)
```

```asm
; === ARM64 assembly (AArch64, GNU syntax) ===
; Same function: dot_product_f32(float* a, float* b, int n) → float
; Uses NEON: 4-wide float32 SIMD
.global dot_product_f32
dot_product_f32:
    movi    v0.4s, #0          // accumulator = {0,0,0,0}
    cbz     w2, .done          // if n==0, return 0
    mov     w3, #0             // i = 0
.loop:
    cmp     w3, w2
    b.ge    .done
    ld1     {v1.4s}, [x0], #16  // load 4 floats from a, advance pointer
    ld1     {v2.4s}, [x1], #16  // load 4 floats from b
    fmla    v0.4s, v1.4s, v2.4s // v0 += v1 * v2  (fused multiply-accumulate)
    add     w3, w3, #4
    b       .loop
.done:
    faddp   v0.4s, v0.4s, v0.4s // pairwise add → {s0+s1, s2+s3, ?, ?}
    faddp   s0, v0.2s            // final reduction → scalar in s0
    ret
```

Notice the differences: ARM has no memory operands in arithmetic instructions (load-store ISA), uses post-increment addressing (`[x0], #16`), and `fmla` is a fused multiply-accumulate — one instruction. x86 needs separate `mulps` + `addps` (unless using FMA extensions).

### The Toolchain: Compiler → Assembler → Linker → Loader

Most software is not written in assembly. It's written in C, C++, or higher — and a **toolchain** translates it down to machine code.

<div class="diagram">
<div class="diagram-title">The Compilation Toolchain</div>
<div class="flow-h">
  <div class="flow-node accent">Source (.c/.cpp)<br/><small>human-written</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">Compiler (cc1/clang)<br/><small>→ assembly (.s)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">Assembler (as)<br/><small>→ object (.o)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">Linker (ld)<br/><small>→ ELF binary</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange">Loader (OS)<br/><small>maps to memory, runs</small></div>
</div>
</div>

1. **Compiler** (`clang`, `gcc`, `nvcc`): translates source code → assembly (or directly to object code). Performs optimization passes: constant folding, dead-code elimination, loop unrolling, auto-vectorization (inserting SIMD instructions). With `-O3 -march=native`, GCC/Clang will emit AVX-512 or NEON instructions automatically.

2. **Assembler** (`as`, `nasm`): converts assembly text → binary object file (`.o`), a chunk of machine code with symbol tables but unresolved references.

3. **Linker** (`ld`, `lld`): combines multiple `.o` files + library archives (`.a`, `.so`) → a single executable or shared library. Resolves symbol references (function calls between files), assigns final addresses, creates the **ELF** (Executable and Linkable Format) binary on Linux.

4. **ELF binary** (on Linux) or **Mach-O** (macOS) or **PE** (Windows): a structured file containing:
   - `.text` section: machine code
   - `.data` / `.bss`: initialized / zero-initialized globals
   - `.rodata`: read-only constants (string literals, float tables)
   - Symbol table, relocation info, dynamic linking metadata

5. **Loader** (OS): when you run the binary, the OS kernel's loader maps the ELF sections into virtual memory, resolves dynamic library symbols (`libcuda.so`, `libtorch.so`), and jumps to `_start`.

```bash
# Inspect the compilation pipeline yourself:
# 1. C → preprocessed C
gcc -E hello.c -o hello.i
# 2. Preprocessed C → assembly
gcc -S -O2 -march=native hello.c -o hello.s
# 3. Assembly → object
gcc -c hello.s -o hello.o
# 4. Object → ELF binary
gcc hello.o -o hello
# 5. Inspect ELF sections
readelf -S hello | grep -E "\.text|\.data|\.bss"
# 6. Disassemble (machine code → assembly)
objdump -d hello | head -40
```

### LLVM IR: The Portable Middle Layer

**LLVM** is a compiler infrastructure (originally from UIUC, 2003). Its key idea is a **typed, portable intermediate representation (IR)** that sits between the language-specific front-end and the machine-specific back-end.

```llvm
; === LLVM IR ===
; C source: int add(int a, int b) { return a + b; }
; Compiled with: clang -S -emit-llvm -O1 add.c -o add.ll

define i32 @add(i32 %a, i32 %b) {
entry:
  %result = add i32 %a, %b    ; typed SSA: i32 = 32-bit signed integer
  ret i32 %result
}

; More interesting: vectorized loop kernel
; C source: void scale(float* x, float s, int n) { for(int i=0;i<n;i++) x[i]*=s; }
define void @scale(float* %x, float %s, i32 %n) {
entry:
  %vscale = insertelement <4 x float> undef, float %s, i32 0
  %vsplat = shufflevector <4 x float> %vscale, <4 x float> undef,
                           <4 x i32> zeroinitializer    ; broadcast scalar to vector
  ; ... loop body would use vector multiply ...
  ret void
}
```

LLVM IR is **strongly typed** and in **SSA form** (Static Single Assignment — each variable assigned exactly once, uses Φ-functions at merge points). Optimizations (inlining, loop vectorization, common subexpression elimination) run at the IR level, independent of source language or target ISA.

**Why it matters for ML:** `torch.compile` (TorchInductor) generates Triton or C++ code → LLVM IR → machine code. XLA uses a similar design. TVM compiles to LLVM IR. If you ever profile and see strange performance behavior in compiled models, understanding the LLVM pipeline helps you know where to look.

### C and C++: The Systems Language Layer

**C** is the language that sits closest to the hardware that still has a compiler managing registers and calling conventions. It was designed by Dennis Ritchie at Bell Labs (1972) to write UNIX. Nearly all OS kernels, drivers, firmware, and runtime libraries are written in C or C++.

```c
/* matmul_naive.c — a simple matrix multiply in C */
/* Shapes: A[M][K], B[K][N], C[M][N]  dtype: float (IEEE-754 f32) */
#include <stddef.h>

void matmul_f32(const float* restrict A,  /* [M*K], row-major */
                const float* restrict B,  /* [K*N] */
                float*       restrict C,  /* [M*N] output, pre-zeroed */
                size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; i++)
        for (size_t k = 0; k < K; k++) {
            float a_ik = A[i*K + k];     /* load once, reuse across inner loop */
            for (size_t j = 0; j < N; j++)
                C[i*N + j] += a_ik * B[k*N + j];   /* 1 FMA per iteration */
        }
}
/* With -O3 -march=native -ffast-math, GCC will auto-vectorize the j-loop to AVX-512 */
/* oneDNN / OpenBLAS replace this with hand-written assembly kernels for 10–100× more */
```

**C++** adds object-orientation, templates, RAII, and the standard library. PyTorch's ATen and Torch libraries are almost entirely C++. CUDA kernels can be written in a C-like dialect with CUDA extensions.

---

## Part 2: The Hardware Description Side

### RTL — Register-Transfer Level

**RTL (Register-Transfer Level)** is the abstraction layer where hardware is described as **registers** (flip-flops storing state) connected by **combinational logic** (gates performing operations). RTL is what hardware engineers write; it describes *what* the hardware does (its logical behavior) without specifying *how* it is physically laid out.

RTL can be **simulated** (to verify behavior) and **synthesized** (to produce a gate-level netlist that a foundry can eventually turn into transistors). It is the "source code" of a chip.

### Verilog and SystemVerilog

**Verilog** (1984, now IEEE 1364) is the most widely used HDL (Hardware Description Language) for design and simulation. **SystemVerilog** (IEEE 1800) is the modern superset adding object-oriented constructs, interfaces, assertions, and better support for verification testbenches.

```verilog
// === Verilog ===
// A parameterized 8-bit accumulator with synchronous reset
// Useful primitive: think of it as the hardware version of a running sum

module accumulator #(
    parameter WIDTH = 8          // configurable bit-width
) (
    input  wire             clk,   // clock: all state changes on rising edge
    input  wire             rst_n, // active-low synchronous reset
    input  wire             en,    // enable: only accumulate when high
    input  wire [WIDTH-1:0] din,   // input data (WIDTH bits)
    output reg  [WIDTH-1:0] dout   // accumulated output (registered)
);
    always @(posedge clk) begin
        if (!rst_n)
            dout <= {WIDTH{1'b0}};   // reset to all zeros
        else if (en)
            dout <= dout + din;      // accumulate: dout += din
        // else: hold current value
    end
endmodule

// Testbench snippet (SystemVerilog)
module tb_accumulator;
    logic        clk = 0, rst_n, en;
    logic [7:0]  din, dout;

    accumulator #(.WIDTH(8)) dut (.*);   // instantiate DUT

    always #5 clk = ~clk;               // 100 MHz clock (period=10ns)

    initial begin
        rst_n = 0; en = 0; din = 8'h00;
        @(posedge clk); rst_n = 1;
        en = 1; din = 8'h03;            // add 3 each cycle
        repeat(4) @(posedge clk);       // 4 cycles → dout should be 12
        $display("dout=%0d (expect 12)", dout);
        $finish;
    end
endmodule
```

The `always @(posedge clk)` block is the heart of RTL: it describes logic that executes on every rising clock edge — this maps directly to flip-flops (the D-type latch from Chapter 2) in silicon.

### VHDL

**VHDL** (VHSIC Hardware Description Language, IEEE 1076) is the other major HDL, dominant in Europe and in aerospace/defense. It is more verbose and strongly typed than Verilog.

```vhdl
-- === VHDL ===
-- Same 8-bit accumulator in VHDL
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity accumulator is
    generic (WIDTH : integer := 8);
    port (
        clk   : in  std_logic;
        rst_n : in  std_logic;
        en    : in  std_logic;
        din   : in  std_logic_vector(WIDTH-1 downto 0);
        dout  : out std_logic_vector(WIDTH-1 downto 0)
    );
end entity accumulator;

architecture rtl of accumulator is
    signal acc : unsigned(WIDTH-1 downto 0) := (others => '0');
begin
    process(clk)
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                acc <= (others => '0');
            elsif en = '1' then
                acc <= acc + unsigned(din);   -- accumulate
            end if;
        end if;
    end process;
    dout <= std_logic_vector(acc);
end architecture rtl;
```

Both Verilog and VHDL describe the same hardware; choice is often team preference or tooling. Modern verification uses **UVM** (Universal Verification Methodology, SystemVerilog-based) for complex chip-level testbenches (see [Chapter 32](./32_chip_verification_and_validation.md)).

### HLS — High-Level Synthesis

**HLS (High-Level Synthesis)** compiles **C/C++/SystemC → RTL automatically**. Instead of writing Verilog by hand, an engineer writes a C function, annotates it with pragma directives (pipeline, unroll, array partition), and the tool synthesizes hardware.

```c
/* === C + HLS pragmas (Vitis HLS / Xilinx style) ===
   Synthesizes to a pipelined dot-product hardware unit      */
#include <hls_stream.h>
#include <ap_fixed.h>

// Fixed-point type: 16-bit total, 8 fractional bits (ap_fixed<total, integer>)
typedef ap_fixed<16, 8> fixed16_t;

void dot_product_hls(
    fixed16_t a[64],   // input array a, 64 elements
    fixed16_t b[64],   // input array b, 64 elements
    fixed16_t* result  // output: scalar dot product
) {
#pragma HLS PIPELINE II=1     // initiation interval=1: one new input set per cycle
#pragma HLS ARRAY_PARTITION variable=a complete  // partition array → parallel access
#pragma HLS ARRAY_PARTITION variable=b complete

    fixed16_t acc = 0;
LOOP: for (int i = 0; i < 64; i++) {
#pragma HLS UNROLL             // unroll fully → 64 parallel multipliers in hardware
        acc += a[i] * b[i];
    }
    *result = acc;
}
// HLS tools can synthesize this into a pipelined MAC array with
// throughput of 1 result per clock cycle — hardware from C code.
```

HLS bridges the gap between algorithm (C) and silicon. Tools: **Xilinx/AMD Vitis HLS**, **Catapult HLS** (Siemens), **Intel HLS Compiler**. Used extensively for FPGA acceleration and increasingly for ASIC design. The synthesis flow from RTL to a gate-level netlist is covered in [Chapter 31](./31_chip_design_flow.md).

---

## Part 3: The Accelerator Side

This is the layer you interact with when writing custom GPU code.

### The "Kernel" Disambiguation

> **WARNING: "kernel" is one of the most overloaded words in this field. It means at least three different things:**

<div class="diagram">
<div class="diagram-title">Three Meanings of "Kernel"</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">OS Kernel</div>
    <div class="card-desc">The core of an operating system (Linux kernel, macOS XNU). Manages hardware resources: CPU scheduling, memory, device drivers, syscalls. Lives in kernel space, privileged.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Compute Kernel (CUDA/GPU)</div>
    <div class="card-desc">A function that runs on many GPU threads in parallel. Written in CUDA C++, HIP, or Triton. Launched from host code with <<<grid, block>>> syntax. This is what people mean by "writing a kernel."</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Mathematical Kernel</div>
    <div class="card-desc">A function in math/ML (e.g., an RBF kernel for SVMs, an attention kernel). Unrelated to hardware. Context tells you which meaning applies.</div>
  </div>
</div>
</div>

From here, "kernel" means **compute kernel** unless explicitly noted otherwise.

### CUDA

**CUDA** (Compute Unified Device Architecture, NVIDIA 2006) is NVIDIA's parallel programming platform. It extends C++ with annotations and intrinsics for GPU programming.

Key concepts:
- **Host**: the CPU and its memory (RAM). Host code runs normally.
- **Device**: the GPU and its memory (VRAM/HBM). Device kernels run on thousands of GPU threads.
- **Kernel function**: annotated `__global__`, launched on the device, executed by many threads
- **Thread hierarchy**: threads → warps (32 threads, execute in lockstep) → thread blocks (≤1024 threads) → grid (all blocks for one kernel launch)
- **Memory spaces**: global (HBM, slow, ~1–2 TB/s), shared (SRAM per SM, fast, ~20 TB/s), registers (fastest, private per thread)

```cuda
// === CUDA C++ ===
// Vector addition kernel: C[i] = A[i] + B[i]
// Each thread handles one element.

#include <cuda_runtime.h>
#include <stdio.h>

// __global__: runs on device (GPU), called from host (CPU)
__global__ void vec_add_kernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float*       __restrict__ C,
                               int N) {
    // Each thread computes its global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];   // one float add per thread
    }
    // This kernel launches N threads total → N parallel adds
}

int main() {
    const int N = 1 << 20;          // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate on device (GPU global memory / HBM)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);   // allocates N*4 bytes = 4 MB on GPU
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // ... (fill d_A, d_B from host via cudaMemcpy) ...

    // Launch: 256 threads per block, ceil(N/256) blocks
    int threads = 256, blocks = (N + threads - 1) / threads;
    vec_add_kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    // Synchronize: wait for all GPU threads to finish
    cudaDeviceSynchronize();

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
```

Real production kernels are more complex: they use **shared memory** tiling to exploit the GPU's memory hierarchy, write **coalesced memory accesses** to maximize HBM bandwidth (see [Chapter 15](./15_gpus.md)), and use **tensor core intrinsics** (`wmma` or PTX `mma` instructions) for matrix math.

### PTX and SASS: NVIDIA's Two ISAs

NVIDIA has two levels of "assembly" for its GPUs:

**PTX** (Parallel Thread eXecution): a **virtual ISA** — a stable, architecture-independent assembly-like language that NVIDIA guarantees forward compatibility for. A CUDA kernel compiled to PTX will run on future NVIDIA GPUs.

```ptx
// === PTX (Parallel Thread eXecution) ===
// Fragment of vec_add_kernel at PTX level
// (generated by nvcc with -ptx flag; human-readable)

.version 7.5
.target sm_80          // logical target: Ampere
.address_size 64

.visible .entry vec_add_kernel(
    .param .u64 param_A,
    .param .u64 param_B,
    .param .u64 param_C,
    .param .u32 param_N
) {
    .reg .f32   %f<4>;          // declare float registers
    .reg .u32   %r<8>;          // declare int registers
    .reg .u64   %rd<8>;
    .reg .pred  %p<2>;          // predicate (boolean) registers

    ld.param.u64  %rd1, [param_A];   // load pointer A from parameter space
    ld.param.u32  %r1,  [param_N];   // load N
    // ... (compute idx, bounds check, load A[idx] and B[idx]) ...
    ld.global.f32 %f1, [%rd2];       // load from global memory (HBM)
    ld.global.f32 %f2, [%rd3];
    add.f32       %f3, %f1, %f2;     // floating-point add
    st.global.f32 [%rd4], %f3;       // store result to global memory
    ret;
}
```

**SASS** (Streaming ASSembler): the actual binary machine ISA of the specific GPU architecture (SM80 for Ampere, SM90 for Hopper). SASS is not publicly documented by NVIDIA and changes between GPU generations. The driver's JIT compiler translates PTX → SASS at kernel load time.

```text
PTX vs SASS relationship:
  Source .cu  → nvcc compile → PTX (.ptx)  → driver JIT → SASS (runs on GPU)
                                             ↑
                         This happens at kernel load time on the target GPU
                         PTX is forward-compatible; SASS is generation-specific
```

**Why this matters for ML:** if you ship a model or inference engine with PTX, it will work on future NVIDIA GPUs (driver compiles PTX → new SASS). If you ship SASS for SM80, it will NOT work on SM90 (Hopper) without recompiling. `torch.jit` and TensorRT packages often embed PTX for this reason.

### ROCm and HIP (AMD)

AMD's answer to CUDA is **ROCm** (Radeon Open Compute) with the **HIP** (Heterogeneous-computing Interface for Portability) programming model. HIP code is syntactically almost identical to CUDA — many CUDA programs can be automatically converted with the `hipify` tool.

```cpp
// === HIP (AMD ROCm) — nearly identical to CUDA ===
#include <hip/hip_runtime.h>

__global__ void vec_add_hip(const float* A, const float* B, float* C, int N) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

// Launch: hipLaunchKernelGGL(func, grid, block, shared_mem, stream, args...)
hipLaunchKernelGGL(vec_add_hip, dim3(blocks), dim3(256), 0, 0, d_A, d_B, d_C, N);
```

AMD's assembly equivalent to PTX is **HSAIL** (legacy) / **GCN ISA** / **RDNA/CDNA ISA**. The compute-focused GPUs (MI300X) use the **CDNA** architecture. PyTorch supports ROCm officially; many libraries have HIP ports.

### OpenCL and SYCL/oneAPI

**OpenCL** (Open Computing Language, Khronos Group): the open standard for heterogeneous computing — CPU, GPU, FPGA, DSP. Vendor-neutral but more verbose than CUDA. Used in Apple Metal's ancestor, and still used for some FPGA and edge targets.

**SYCL** (Single-source Heterogeneous Programming for C++): the modern C++17-based open standard, championed by Intel as part of their **oneAPI** initiative. Compiles the same source for CPU (AVX), GPU (Intel Xe / Arc / Gaudi), and FPGA.

```cpp
// === SYCL/oneAPI ===
#include <sycl/sycl.hpp>
using namespace sycl;

queue q;  // selects default accelerator (GPU if available)
float *d_C = malloc_device<float>(N, q);

q.parallel_for(range<1>(N), [=](id<1> i) {
    d_C[i] = d_A[i] + d_B[i];   // kernel body — runs on device
}).wait();
```

### Triton

**Triton** (OpenAI, 2021) is a Python-like language and compiler for writing **GPU kernels** at a higher level than CUDA, targeting a tile/block-level abstraction rather than individual threads. It compiles via LLVM/MLIR to PTX (NVIDIA) or GCN (AMD).

```python
# === Triton ===
# Vector addition kernel — same operation, much less boilerplate than CUDA
import triton
import triton.language as tl
import torch

@triton.jit
def vec_add_triton(
    A_ptr,  B_ptr,  C_ptr,   # device pointers (torch.Tensor data_ptr())
    N,
    BLOCK_SIZE: tl.constexpr  # compile-time constant: tile size
):
    # Each Triton "program" (roughly: a thread block) handles one tile
    pid = tl.program_id(axis=0)                      # which tile am I?
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # element indices
    mask = offsets < N                               # guard against out-of-bounds
    
    a = tl.load(A_ptr + offsets, mask=mask)          # load tile from HBM
    b = tl.load(B_ptr + offsets, mask=mask)
    c = a + b                                        # element-wise add (all lanes)
    tl.store(C_ptr + offsets, c, mask=mask)          # store result

# Python launch wrapper
def vec_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and a.shape == b.shape
    c = torch.empty_like(a)
    N = a.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)   # number of Triton programs
    vec_add_triton[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)
    return c
```

Triton is the layer where most ML practitioners who "write custom kernels" actually work. Flash Attention 2 is written in Triton. xFormers and many custom attention operators use Triton. It generates PTX internally — you get performance close to hand-written CUDA with much less C++ boilerplate.

### Intrinsics

**Intrinsics** are C/C++ function calls that map 1-to-1 to specific assembly instructions — they let you use hardware features (SIMD, atomic ops, cryptography) without writing assembly.

```c
// === C intrinsics (x86 AVX-512 intrinsics, header: immintrin.h) ===
#include <immintrin.h>

void scale_avx512(float* x, float scalar, int n) {
    __m512 vs = _mm512_set1_ps(scalar);   // broadcast scalar to 16-wide vector
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);          // load 16 floats (unaligned)
        __m512 vy = _mm512_mul_ps(vx, vs);            // 16 × f32 multiply
        _mm512_storeu_ps(x + i, vy);                  // store 16 floats
    }
    // scalar tail for remaining elements
    for (; i < n; i++) x[i] *= scalar;
}
```

CUDA also has intrinsics: `__float2half_rn()`, `__hfma()`, `__ldg()` (texture cache loads), `__shfl_sync()` (warp shuffle — communicate between threads without shared memory).

### Drivers and Runtime

The **GPU driver** is a kernel-mode (ring 0) component of the OS that owns the GPU hardware — it manages VRAM allocation, context switching between processes, firmware communication, and translating high-level API calls into hardware register writes.

The **GPU runtime** (e.g., `libcuda.so`, `librocm_smi.so`) is the user-space library your process links against. `cudaMalloc`, `cudaMemcpy`, `cudaLaunchKernel` are runtime calls.

```text
Call stack when you call cuda kernel:
  PyTorch Python  →  ATen C++  →  cudaLaunchKernel (runtime API)
                                        ↓
                               libcuda.so (user-space driver)
                                        ↓
                              nvidia.ko (kernel-mode driver)
                                        ↓
                       GPU hardware register write → kernel executes
```

When you see errors like `CUDA error: device-side assert triggered` or `NCCL error: unhandled system error`, they originate at or below the runtime layer.

---

## Tracing a PyTorch Line Through the Stack

Let's make this concrete. What happens when you execute `y = x @ w` on a CUDA tensor?

```python
import torch

x = torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda")   # [512, 4096] bf16
w = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")  # [4096, 4096] bf16

y = x @ w   # <-- trace this line through the stack
# y shape: [512, 4096], dtype: bfloat16
```

<div class="diagram">
<div class="diagram-title">Tracing x @ w Through the Full Stack</div>
<div class="flow">
  <div class="flow-node accent wide">Python: dispatches via __matmul__ → torch._C._VariableFunctions.mm</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">ATen dispatcher (C++): selects "mm" kernel for CUDA + bfloat16</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">cuBLAS / cuBLASLt library: GEMM kernel selection (BLAS Level 3)</div>
  <div class="flow-arrow accent"></div>
  <invoke name="flow-node purple wide">CUDA PTX kernel: uses Tensor Core wmma / mma instructions (H100: BF16 Tensor Cores)</invoke>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Driver JIT-compiles PTX → SASS for your specific GPU (e.g., SM90 for H100)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">GPU hardware: Tensor Cores execute 16×16×16 BF16 matrix tiles per cycle</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node teal wide">Transistors: billions of MOSFETs switching on/off to perform the multiply-accumulates</div>
</div>
</div>

When you use `torch.compile`:

```python
compiled_fn = torch.compile(lambda x, w: x @ w, backend="inductor")
y = compiled_fn(x, w)
```

TorchInductor generates a **Triton kernel** (or falls back to cuBLAS for large GEMMs), compiles Triton → PTX via LLVM → driver JIT → SASS. The path is longer but the result can be faster because the compiler can fuse operations (e.g., matmul + activation + layer norm in one kernel launch, avoiding round-trips to HBM).

---

## The Master Glossary Table

| Term | One-line definition | Layer in the stack |
|------|---------------------|-------------------|
| **Bare metal** | Code running with no OS | Below OS |
| **Firmware** | Software stored in non-volatile memory on a device | Below OS |
| **BIOS/UEFI** | First firmware to run on a PC; initializes hardware, loads OS | Below OS |
| **Microcode** | Internal ROM inside a CISC CPU mapping complex instructions → µops | Inside CPU |
| **Machine code** | Raw binary bytes the CPU executes directly | ISA level |
| **Assembly** | Human-readable 1-to-1 symbolic representation of machine code | ISA level |
| **Assembler** | Tool that converts assembly text → machine code | Toolchain |
| **Compiler** | Translates high-level language (C/C++) → assembly or machine code | Toolchain |
| **Linker** | Combines object files + libraries → a single executable | Toolchain |
| **Loader** | OS component that maps an executable into memory and runs it | OS |
| **ELF / PE / Mach-O** | Binary executable file formats (Linux / Windows / macOS) | Toolchain |
| **LLVM IR** | Portable typed SSA intermediate representation used by Clang, Triton, etc. | Compiler IR |
| **RTL** | Register-Transfer Level: hardware described as registers + combinational logic | Hardware design |
| **Verilog / SystemVerilog** | HDLs for designing and verifying digital hardware | Hardware design |
| **VHDL** | Alternative HDL (strongly typed); common in Europe and defense | Hardware design |
| **HLS** | High-Level Synthesis: compiles C/C++ → RTL hardware automatically | Hardware design |
| **Netlist** | Gate-level representation after synthesis, before place-and-route | Physical design |
| **GDSII** | The final geometric file sent to the fab to make the chip | Physical design |
| **CUDA** | NVIDIA's parallel programming platform (C++ extensions + driver) | Accelerator |
| **HIP** | AMD's CUDA-compatible GPU programming API (part of ROCm) | Accelerator |
| **OpenCL** | Open, vendor-neutral heterogeneous compute standard | Accelerator |
| **SYCL / oneAPI** | Modern C++17-based open heterogeneous programming (Intel) | Accelerator |
| **Triton** | Python-like language for writing GPU kernels; compiles to PTX/GCN | Accelerator |
| **PTX** | NVIDIA's virtual GPU assembly ISA; forward-compatible across generations | GPU ISA |
| **SASS** | NVIDIA's actual binary GPU ISA for a specific SM generation | GPU ISA |
| **Intrinsic** | C/C++ function that maps 1-to-1 to a specific assembly instruction | Toolchain |
| **Kernel (compute)** | A GPU function launched to run on many parallel threads | Accelerator |
| **Kernel (OS)** | The core of an operating system managing hardware resources | OS |
| **Driver** | OS kernel module that owns and manages a hardware device | OS |
| **Runtime** | User-space library providing API to use a device (libcuda.so) | User space |
| **ISA** | Instruction Set Architecture: the hardware/software contract (Chapter 5) | Architecture |
| **Microarchitecture** | The implementation of an ISA in silicon | Architecture |

---

## Why This Matters for Model Optimization

<div class="diagram">
<div class="diagram-title">Choosing Your Layer of Intervention</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Python / torch.compile</div>
    <div class="card-desc">90% of optimization work lives here. Operator fusion, graph rewriting, dtype casting. No assembly required.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Triton Kernels</div>
    <div class="card-desc">Custom attention variants, quantized GEMM, fused softmax. You control memory access patterns and tiling — the most impactful lever after algorithm choice.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">CUDA C++ Kernels</div>
    <div class="card-desc">Maximum control: shared memory layout, warp-level primitives, tensor core intrinsics (wmma/mma). Reserved for ops where Triton's abstractions limit performance.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">PTX vs SASS</div>
    <div class="card-desc">Ship PTX for forward-compatibility. Profile with Nsight Compute to see SASS instruction mix — if you're seeing unexpected register spills or memory traffic, the SASS view explains why.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">RTL / HLS for ASICs</div>
    <div class="card-desc">If you're building a custom inference chip, this is your layer. HLS lets ML-aware engineers specify compute patterns (e.g., systolic array tiling) and synthesize them to hardware.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Firmware / Driver</div>
    <div class="card-desc">Keep drivers updated — microcode patches fix bugs that silently corrupt training (rare but real). GPU firmware mismatches cause hangs in multi-GPU setups.</div>
  </div>
</div>
</div>

The key principle: **each layer is a contract that lets the layer above ignore everything below**. Triton programmers don't write PTX. PTX programmers don't write SASS. This is why the stack exists — but understanding the layer below yours tells you what is truly possible and what the limits are.

---

## Key Takeaways

- **Bare metal, firmware, microcode** are the layers below the OS, closest to hardware. Microcode lets CISC CPUs translate complex instructions to simple µops internally — and is patchable for security fixes.
- **Assembly** is the human-readable 1-to-1 representation of machine code; the toolchain (compiler → assembler → linker → loader) converts source code to a running binary. **LLVM IR** is the portable middle layer many modern compilers share.
- **RTL (Verilog / VHDL)** is the hardware description layer — the "source code" of a chip. **HLS** compiles C to RTL, lowering the bar for custom hardware.
- **CUDA** programs the GPU from C++ with a host/device split and a thread hierarchy (thread → warp → block → grid). **PTX** is the forward-compatible virtual GPU ISA; **SASS** is the architecture-specific binary.
- **Triton** is the practical entry point for ML engineers writing custom GPU kernels — Python-like, compiles to PTX, used for Flash Attention and other fused operators.
- **"Kernel"** means three different things: OS kernel, compute kernel (GPU function), or mathematical kernel. Context is everything.
- Every `x @ w` in PyTorch traverses the entire stack — Python → ATen → cuBLAS/Triton → PTX → SASS → Tensor Cores → transistors. Knowing the layers tells you where to intervene when performance is wrong.

---

*Next: [Chapter 7 — Types of Chips: A Taxonomy](./07_types_of_chips_taxonomy.md), the map of all chip types — CPU, GPU, TPU, NPU, DSP, FPGA, ASIC — and the flexibility/efficiency tradeoff that explains why you train on GPUs and deploy on NPUs.*

[← Back to Table of Contents](./README.md)
