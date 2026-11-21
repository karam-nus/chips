---
title: "Appendix H — The Life of a Running Computer"
---

[← Back to Table of Contents](./README.md)

# Appendix H — The Life of a Running Computer

Here is a question almost no architecture text answers directly: **what actually happens when you press the power button, and what is the CPU doing right now while you read this — when "nothing" is running?** It feels like the processor must need a constant stream of instructions to "stay on," as if someone has to keep feeding it work or it stops. And what keeps the screen lit if the CPU is doing nothing?

This appendix answers that end to end. It ties together the fetch–decode–execute loop ([Chapter 4](./04_anatomy_of_a_processor.md)), firmware and the OS ([Chapter 6](./06_languages_and_nomenclature.md)), interrupts ([Chapter 13](./13_reliability_and_fault_tolerance.md) touched them), and power/idle states ([Appendix D](./appendix_d_power_and_thermals.md)) into one story: **how a computer comes alive, stays responsive, and sleeps between your keystrokes.** The punchline reshapes how you think about your training loop too — because a GPU job is, at bottom, a host CPU doing exactly this dance with the accelerator.

> **The one-sentence version:** A computer is *not* a loop you must keep spinning — it is an **interrupt-driven** machine that spends most of its life **halted and asleep**, woken thousands of times a second by hardware **interrupts** (a timer, your keyboard, a network packet, a "GPU done" signal) to do a tiny burst of work and then go back to sleep.

---

## The Mental Model to Unlearn

Two opposite intuitions are both wrong:

<div class="diagram">
<div class="diagram-title">Two Wrong Models vs Reality</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card red">
    <div class="card-title">❌ "It stops when done"</div>
    <div class="card-desc">Like the TinyCPU in Ch 4 that hits HALT and ends. Real systems never run out of program — there is always an OS to return to.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">❌ "It frantically loops"</div>
    <div class="card-desc">You might picture the CPU spinning in a busy loop checking everything. That would waste 100% of the power for nothing.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">✅ "It sleeps & is woken"</div>
    <div class="card-desc">The CPU halts (clock stopped, low power) and is woken by interrupts only when there is something to do. Idle ≈ asleep-but-instant-wake.</div>
  </div>
</div>
</div>

The **program counter** (Chapter 4) does always point at *some* next instruction — that part of your intuition is correct. But "having a next instruction" is not the same as "executing continuously." When there is genuinely nothing to do, the CPU executes one special instruction — `HLT` — that says *"stop fetching and freeze my clock until an interrupt arrives."* It then draws a trickle of power and does literally nothing until the world pokes it. We'll build up to exactly how that works.

---

## From Power Button to Desktop: The Boot Sequence

The whole life of the machine fits in one picture: a short relay that runs once, then a loop that runs until power-off.

<!-- flowchart:boot_loop -->
<div class="flowchart" id="fc-boot_loop">
<div class="fc-title">Boot once, then sleep and wake for the rest of its life</div>
<svg viewBox="0 0 800 420" width="800" role="img" aria-label="Boot once, then sleep and wake for the rest of its life">
<path class="fc-edge" d="M159,36 L173,36"/><polygon class="fc-head" points="173,36 166,39.9 166,32.1"/>
<path class="fc-edge" d="M319,36 L333,36"/><polygon class="fc-head" points="333,36 326,39.9 326,32.1"/>
<path class="fc-edge" d="M479,36 L493,36"/><polygon class="fc-head" points="493,36 486,39.9 486,32.1"/>
<path class="fc-edge" d="M639,36 L653,36"/><polygon class="fc-head" points="653,36 646,39.9 646,32.1"/>
<path class="fc-edge" d="M726,64 L726,160 L550,160"/><polygon class="fc-head" points="550,160 557,156.2 557,163.8"/>
<path class="fc-edge yes" d="M330,181 L330,212.5 L150,212.5 L150,244"/><polygon class="fc-head yes" points="150,244 146.2,237 153.8,237"/><text class="fc-elabel yes" x="322" y="214" text-anchor="end">yes</text>
<path class="fc-edge no" d="M470,181 L470,212.5 L650,212.5 L650,244"/><polygon class="fc-head no" points="650,244 646.1,237 653.9,237"/><text class="fc-elabel no" x="478" y="214" text-anchor="start">no</text>
<path class="fc-edge loop" d="M35,272 L20,272 L20,160 L250,160"/><polygon class="fc-head loop" points="250,160 243,163.8 243,156.2"/><text class="fc-elabel loop" x="26" y="150" text-anchor="start">blocks / slice ends</text>
<path class="fc-edge" d="M650,300 L650,372 L550,372"/><polygon class="fc-head" points="550,372 557,368.1 557,375.9"/><text class="fc-elabel " x="660" y="340" text-anchor="start">device raises IRQ</text>
<path class="fc-edge" d="M250,372 L150,372 L150,300"/><polygon class="fc-head" points="150,300 153.8,307 146.2,307"/><text class="fc-elabel " x="140" y="365" text-anchor="end">ISR marks a thread ready</text>
<g class="fc-node start"><rect x="13" y="14" width="146" height="44" rx="16"/><text x="86" y="34" text-anchor="middle">Reset</text><text class="sub" x="86" y="47" text-anchor="middle">PC = reset vector</text></g>
<g class="fc-node k-cyan"><rect x="173" y="8" width="146" height="56" rx="6"/><text x="246" y="28" text-anchor="middle">Firmware</text><text class="sub" x="246" y="41" text-anchor="middle">UEFI / BIOS: POST,</text><text class="sub" x="246" y="53" text-anchor="middle">trains DRAM</text></g>
<g class="fc-node k-cyan"><rect x="333" y="8" width="146" height="56" rx="6"/><text x="406" y="28" text-anchor="middle">Bootloader</text><text class="sub" x="406" y="41" text-anchor="middle">loads the kernel</text><text class="sub" x="406" y="53" text-anchor="middle">into RAM</text></g>
<g class="fc-node k-blue"><rect x="493" y="8" width="146" height="56" rx="6"/><text x="566" y="28" text-anchor="middle">Kernel init</text><text class="sub" x="566" y="41" text-anchor="middle">page tables, IDT,</text><text class="sub" x="566" y="53" text-anchor="middle">other cores online</text></g>
<g class="fc-node k-blue"><rect x="653" y="8" width="146" height="56" rx="6"/><text x="726" y="28" text-anchor="middle">init (PID 1)</text><text class="sub" x="726" y="41" text-anchor="middle">services, login,</text><text class="sub" x="726" y="53" text-anchor="middle">display server</text></g>
<g class="fc-node decision"><polygon points="250,160 264,139 536,139 550,160 536,181 264,181"/><text x="400" y="164" text-anchor="middle">Scheduler: any thread ready?</text></g>
<g class="fc-node k-green"><rect x="35" y="244" width="230" height="56" rx="6"/><text x="150" y="264" text-anchor="middle">Run a thread</text><text class="sub" x="150" y="277" text-anchor="middle">for a time slice, until it</text><text class="sub" x="150" y="289" text-anchor="middle">blocks or is preempted</text></g>
<g class="fc-node k-purple"><rect x="535" y="244" width="230" height="56" rx="6"/><text x="650" y="264" text-anchor="middle">Idle thread</text><text class="sub" x="650" y="277" text-anchor="middle">HLT / WFI, then deeper</text><text class="sub" x="650" y="289" text-anchor="middle">C-states while nothing happens</text></g>
<g class="fc-node k-orange"><rect x="250" y="350" width="300" height="44" rx="6"/><text x="400" y="370" text-anchor="middle">Interrupt</text><text class="sub" x="400" y="383" text-anchor="middle">timer tick, keypress, packet, &quot;GPU kernel done&quot;</text></g>
</svg>
<div class="fc-caption">The top row runs once per power-on. Everything after that is the bottom loop: the scheduler runs ready threads, and when none are ready the core halts until an interrupt wakes it.</div>
</div>
<!-- /flowchart:boot_loop -->

When power stabilizes, the CPU is electrically **reset**: its registers take defined values and the program counter is forced to a hard-wired address called the **reset vector** (on x86, physical `0xFFFFFFF0`, right at the top of the address space; on ARM, an address defined by the SoC). Whatever code lives there runs first. That code is **firmware**, baked into a flash chip on the motherboard.

<div class="diagram">
<div class="diagram-title">Power-On → Login (each box is code the CPU fetch-decode-executes)</div>
<div class="flow-h">
  <div class="flow-node accent">Reset<br/><small>PC → reset vector</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">Firmware<br/><small>BIOS / UEFI · POST</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">Bootloader<br/><small>GRUB / systemd-boot</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">Kernel<br/><small>Linux / NT / XNU</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange">init (PID 1)<br/><small>systemd / launchd</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan">Login / Desktop</div>
</div>
</div>

| Stage | Who runs | What it does |
|-------|----------|--------------|
| **Reset** | CPU hardware | Forces PC to the reset vector; caches/MMU off, single core awake |
| **Firmware (BIOS/UEFI)** | Flash ROM code | **POST** (power-on self-test), trains and initializes **DRAM**, enumerates devices (PCIe, USB, storage), then loads the next stage |
| **Bootloader** | e.g. GRUB, systemd-boot, Windows Boot Manager | Finds the OS kernel on disk, loads it into RAM, hands over control |
| **Kernel init** | OS kernel | Sets up **page tables** (Ch 10), the **interrupt vector table**, drivers, brings the **other CPU cores online**, starts the scheduler |
| **init / PID 1** | systemd (Linux), launchd (macOS), smss (Windows) | Starts background **services/daemons**, the display server, the login screen |

Three things to notice. First, **booting is a relay race of programs**, each loading and jumping to the next — all of it ordinary fetch–decode–execute. Second, **DRAM doesn't work until firmware trains it**, so the very first code runs from ROM/SRAM. Third, by the end, a special process — the **scheduler** — is in charge of deciding what every core does from then on. The machine is now "on," and it will stay on the same way forever after: by being interrupted.

---

## Interrupts: How the Outside World Breaks In

An **interrupt** is a hardware signal that says *"stop what you're doing and run this handler."* It is the single most important idea in this appendix, and the reason a CPU does not need to poll the world in a loop.

The mechanism:

<div class="diagram">
<div class="diagram-title">Interrupt Flow</div>
<div class="flow">
  <div class="flow-node accent wide">Device asserts an IRQ (e.g. keyboard, timer, NIC, GPU "kernel done")</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">CPU finishes the current instruction, then saves PC + registers</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">Looks up the handler address in the Interrupt Vector Table (IDT on x86)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Runs the ISR (Interrupt Service Routine) — tiny, fast</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Restores saved state and returns (iret) — or calls the scheduler</div>
</div>
</div>

A dedicated chip — the **interrupt controller** (APIC on x86, GIC on ARM) — collects IRQ lines from dozens of devices and prioritizes them. Interrupts come in three flavors:

- **Hardware interrupts** — asynchronous, from devices: the **timer**, keyboard, mouse, network card, disk/NVMe, USB, and — critically for us — the **GPU signalling that a kernel finished**.
- **Exceptions / faults** — synchronous, from the CPU itself: a page fault (Ch 10), divide-by-zero, illegal instruction.
- **Software interrupts / system calls** — a program *deliberately* traps into the kernel (`syscall`/`svc`) to ask for I/O, memory, etc.

The most important one is the **timer interrupt** — the OS's heartbeat. The kernel programs a hardware timer to fire periodically (historically 100 Hz, often 250–1000 Hz on Linux via `CONFIG_HZ`; modern kernels can run **tickless**). Every tick is a chance for the OS to regain control and decide who runs next.

> **Interrupt-driven beats polling.** Polling = "ask every device, in a loop, forever" → 100% CPU, 100% power, even when idle. Interrupt-driven = "sleep until a device raises its hand." The exception: very high-throughput paths (NVMe, DPDK networking, spinlocks held briefly) sometimes *poll on purpose* because an interrupt's ~1 µs entry cost is too slow when events arrive millions of times per second — the same throughput-vs-latency trade you saw in [Chapter 11](./11_memory_wall_and_bandwidth.md).

---

## The Scheduler: Who Gets the Core

After boot, every core runs under the **scheduler**. The kernel keeps a **run queue** of threads that are *ready* to execute. On each timer tick (or when a thread blocks), the scheduler picks one and runs it for a **time slice** (a few milliseconds), then preempts it and picks another — **preemptive multitasking**. The illusion of "everything running at once" on 8 cores is really the scheduler rapidly rotating dozens of threads through those cores.

Switching from one thread to another is a **context switch**: save the current thread's registers and PC into its kernel data structure (the PCB / `task_struct`), load the next thread's, and resume. It costs ~1–5 µs (plus cache/TLB cold-start afterward — Ch 10).

So when you think "nothing is running," reality is: a few dozen background **services/daemons** exist, but almost all of them are **blocked**, sleeping on an event (a timer, a socket, a keypress). They are *not* in the run queue. Which raises the question the whole appendix is building toward: if the run queue is empty, what does the core do?

---

## What "Idle" Really Means: the Idle Loop, `HLT` & C-states

When the scheduler finds **no ready thread**, it dispatches a special built-in thread: the **idle thread** (one per core). Its entire job is to put the core to sleep. Conceptually the kernel's idle path looks like this:

```c
/* Simplified Linux-style per-core idle loop (cpu_idle_loop) */
while (1) {
    while (!need_resched()) {       /* nothing ready to run?           */
        /* pick the deepest safe C-state, then halt the core:          */
        asm volatile("hlt");        /* x86: stop clock until interrupt */
        /* ARM equivalent: asm volatile("wfi");  // Wait For Interrupt  */
    }
    schedule();                     /* an interrupt woke us & queued work */
}
```

The `HLT` instruction (x86) / `WFI` (ARM) **stops the core's clock**. The core consumes almost no dynamic power (recall $P_{\text{dyn}} \approx \alpha C V^2 f$ from [Appendix D](./appendix_d_power_and_thermals.md) — set the switching activity to zero and you stop burning energy) and simply waits. The **next interrupt** — even just the timer tick — wakes it: the clock restarts, the ISR runs, and if that made a thread ready, the scheduler dispatches it; otherwise it halts again.

Modern CPUs go far deeper than a simple halt, via **C-states** (CPU idle power states) chosen by the OS governor based on how long it expects to stay idle:

| C-state | Name | What's turned off | Wake latency | Relative power |
|:-------:|------|-------------------|:------------:|:--------------:|
| **C0** | Active | nothing — executing | — | 100% |
| **C1** | Halt | core clock gated (`HLT`) | ~1–2 µs | ~30–50% |
| **C3** | Sleep | L1/L2 flushed, PLLs off | ~10–60 µs | ~10–20% |
| **C6** | Deep power down | core voltage ≈ 0, state saved to SRAM | ~50–200 µs | ~1–5% |

The exact latencies vary by processor; on Linux, `/sys/devices/system/cpu/cpu0/cpuidle/state*/latency` lists the ones your kernel uses.

Deeper states save more power but take longer to wake — a latency/efficiency trade-off the kernel manages continuously. And **tickless (NO_HZ) kernels** go one step further: if a core will be idle for a while, the kernel *stops the periodic timer interrupt itself*, so a deeply idle laptop core can sleep for tens of milliseconds uninterrupted — which is why your laptop battery lasts hours while "doing nothing."

> **So: when no process is running, the CPU is not executing instructions in a loop — it is halted in a C-state, clock stopped, sipping milliwatts, waiting for the next interrupt.** That is the precise answer to "what happens when it's idle."

---

## Who Keeps the Screen Lit While the CPU Sleeps?

If the CPU is asleep most of the time, how does the monitor keep showing a stable image at 60 Hz? Answer: **the CPU isn't involved frame-to-frame at all.**

A region of memory holds the **framebuffer** — one pixel value per screen pixel (e.g. 3840×2160 × 4 bytes ≈ 33 MB for a 4K display). A dedicated hardware block — the **display controller** (a.k.a. display engine / CRTC, part of the GPU or SoC) — autonomously reads that framebuffer over **DMA** and streams it pixel-by-pixel to the panel through its timing generator, **60–144 times per second**, with *zero CPU instructions per frame*.

<div class="diagram">
<div class="diagram-title">Display Scanout — All Hardware, No CPU Loop</div>
<div class="flow-h">
  <div class="flow-node green">Framebuffer in RAM<br/><small>~33 MB @ 4K</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">Display controller<br/><small>DMA + timing gen</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">Panel @ 60–144 Hz<br/><small>continuous scanout</small></div>
</div>
</div>

The CPU/GPU only do work when the image **changes** — to render a new frame and write it into the (back) framebuffer. On a static screen (this page, sitting still), the CPU is overwhelmingly in a deep C-state while the display engine quietly refreshes the unchanged pixels. A few related tricks:

- **Double buffering + VSync:** the GPU draws into a back buffer and swaps it to the front at the vertical-blank (**vblank**) boundary, so you never see a half-drawn frame (tearing).
- **Panel Self-Refresh (PSR):** on laptops, when the image is static, the panel refreshes itself from its own local memory so even the *display engine* and memory can idle — more battery saved.

This is the second half of the answer to your question: **the computer "stays on and displays" because dedicated hardware, not the CPU, drives the screen.**

---

## Putting It Together: One Keystroke, End to End

Watch the whole machine breathe through a single event:

<div class="diagram">
<div class="diagram-title">A Day in the Life: You Press a Key</div>
<div class="flow">
  <div class="flow-node cyan wide">① Idle: every core in C6 (clocks stopped). Display engine scans out the unchanged framebuffer at 60 Hz.</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">② You press a key → keyboard controller raises an IRQ.</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">③ A core wakes (C6 → C0 in tens to hundreds of µs), runs the keyboard ISR, queues a "key" event, marks the focused app's thread ready.</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">④ Scheduler dispatches the app thread; it computes the new character and writes pixels into the back framebuffer.</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">⑤ At the next vblank the display engine scans out the updated framebuffer — you see the letter appear.</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">⑥ Run queue empties again → the core executes HLT and drops back to sleep. Total active time: well under a millisecond.</div>
</div>
</div>

A "busy" desktop is just this loop happening thousands of times a second — timer ticks, mouse moves, network packets, disk completions — each a brief wake-work-sleep cycle. The CPU is *reactive*, not *continuous*.

---

## Why This Matters for Model Optimization

This exact pattern governs how your training and inference jobs actually run — because **a GPU workload is a host CPU performing this wake/sleep dance with the accelerator.**

<div class="diagram">
<div class="diagram-title">The Host ↔ Device Loop Is the Same Story</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">The host loop launches, then sleeps</div>
    <div class="card-desc">Your Python loop calls the driver to enqueue kernels on a CUDA stream, then often waits in <code>torch.cuda.synchronize()</code>. By default the CUDA driver may <em>spin</em> on a completion flag for the lowest latency, or yield the core; with blocking-sync it parks the thread until the GPU raises a <strong>completion interrupt</strong>. The CPU isn't "running the model"; it's dispatching and waiting.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">A slow host loop starves the GPU</div>
    <div class="card-desc">If the CPU can't enqueue the next kernel before the current one finishes, the GPU goes idle between kernels → low "GPU utilization." This is why <strong>kernel-launch overhead</strong>, Python overhead, <strong>CUDA Graphs</strong>, kernel fusion, and async <code>DataLoader</code> prefetch matter so much.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">DataLoaders ride the scheduler</div>
    <div class="card-desc">Multi-worker data loading uses OS processes/threads, interrupts (disk/network I/O completions), and the scheduler to overlap CPU data prep with GPU compute — the same interrupt-driven concurrency, applied to feeding the accelerator.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Idle hardware still costs money</div>
    <div class="card-desc">An idle GPU between inference requests clock-gates but still draws baseline power. Underutilized accelerators waste $/token — the economic reason serving systems batch aggressively (Ch 29) to keep the device in C0, not idling.</div>
  </div>
</div>
</div>

When you profile a training step and see the GPU sitting at 60% utilization, you are watching this appendix in miniature: the device finishing work and *idling* while the host loop scrambles to feed it the next kernel. Understanding the wake/dispatch/sleep cycle is what lets you tell a *compute* bottleneck from a *host-overhead* bottleneck.

---

## Key Takeaways

- A computer is **interrupt-driven, not loop-driven**: it spends most of its life **halted** and is woken by hardware interrupts to do brief bursts of work.
- **Boot** is a relay of programs — reset vector → firmware (BIOS/UEFI) → bootloader → kernel → init — each ordinary fetch–decode–execute, ending with the **scheduler** in charge.
- **Interrupts** (timer, devices, the GPU's "done" signal) and the **interrupt vector table** are how the CPU reacts to the world without polling; the **timer interrupt** is the OS's heartbeat.
- The **scheduler** rotates ready threads across cores via **context switches**; "idle" background services are blocked, not spinning.
- **Idle = `HLT`/`WFI` + C-states**: the core stops its clock and parks in a low-power state (C1→C6) until an interrupt; **tickless** kernels let it sleep even longer. *This* is what happens when "nothing" is running.
- The **screen stays lit by dedicated hardware** — the display controller DMA-streams the framebuffer to the panel at the refresh rate with no per-frame CPU work.
- Your **GPU jobs are the same dance**: the host enqueues kernels and idles awaiting a completion interrupt — so host-loop overhead, CUDA Graphs, and batching directly determine whether the accelerator stays busy.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>A GPU server spends most of its life waiting: the CPU for the GPU, the GPU for the next batch, both for the network. Pick one wait in your own pipeline and ask whether it is a sleep (cheap, slow to wake) or a spin (instant, burns a core). Which would you rather have there?</p>
</div>

## Check Yourself

<details class="check">
<summary>A core in C6 takes ~100 µs to wake. Why might a latency-critical server disable deep C-states, and what does it cost?</summary>
<p>Every interrupt that lands on a sleeping core pays the wake-up latency before the handler runs, which shows up as tail latency. Limiting cores to C1 keeps wake-ups at ~1 µs but raises idle power.</p>
</details>

<details class="check">
<summary>Why does a 4K display keep showing an image while every CPU core is halted?</summary>
<p>The display controller DMA-reads the ~33 MB framebuffer and drives the panel on its own timing; the CPU and GPU only write a new frame when the image changes. Panel self-refresh can idle even that.</p>
</details>

<details class="check">
<summary>What is the reset vector, and why can't the first firmware code use DRAM?</summary>
<p>The fixed address the program counter is forced to at reset (0xFFFFFFF0 on x86), mapped to flash. DRAM has not been trained or initialized yet, so early firmware runs from flash and cache-as-RAM until it brings memory up.</p>
</details>

<details class="check">
<summary>Your training loop shows the GPU 60% busy, with gaps between kernels. Name two host-side causes and a fix for each.</summary>
<p>Python and launch overhead per kernel (fix: CUDA Graphs or torch.compile to batch launches) and data loading on the critical path (fix: more DataLoader workers, pinned memory and prefetching).</p>
</details>

<details class="check">
<summary>Why is interrupt-driven I/O usually better than polling, and when is polling better?</summary>
<p>Interrupts let idle cores sleep and wake only when needed. At very high event rates (100G networking, NVMe at millions of IOPS) the per-interrupt overhead dominates, so drivers switch to polling (NAPI, io_uring polling, DPDK).</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Operating Systems: Three Easy Pieces | Arpaci-Dusseau & Arpaci-Dusseau | [pages.cs.wisc.edu/~remzi/OSTEP](https://pages.cs.wisc.edu/~remzi/OSTEP/) | Scheduling, interrupts, context switches |
| UEFI Specification | UEFI Forum | [uefi.org/specifications](https://uefi.org/specifications) | Firmware boot phases |
| Linux CPU idle (cpuidle) documentation | Linux kernel | [docs.kernel.org/admin-guide/pm/cpuidle.html](https://docs.kernel.org/admin-guide/pm/cpuidle.html) | C-states, governors, tickless idle |
| Intel 64 and IA-32 Architectures Software Developer's Manual | Intel | [intel.com/content/www/us/en/developer/article…](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html) | Reset vector, HLT, interrupt descriptor table |
| CUDA Runtime API: cudaSetDeviceFlags | NVIDIA | [docs.nvidia.com/cuda/cuda-runtime-api/group__…](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html) | Spin, yield and blocking-sync scheduling |

---

*This appendix is the runtime counterpart to [Chapter 4 — Anatomy of a Processor](./04_anatomy_of_a_processor.md): Chapter 4 shows how one instruction executes; this shows how billions of them get orchestrated into a living, responsive machine.*

[← Back to Table of Contents](./README.md)
