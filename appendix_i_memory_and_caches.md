---
title: "Appendix I — Caches & the Memory Zoo"
---

[← Back to Table of Contents](./README.md)

# Appendix I — Caches & the Memory Zoo

You keep hearing the words: **L1, L2, L3**, register file, shared memory, SRAM, DRAM, HBM, LPDDR, NAND, NOR, eFuse, MRAM, RRAM. They are not a random pile of acronyms. Every one of them is an answer to the same question — *where should this bit live so it is close enough, big enough and cheap enough?* — asked at a different place in the machine.

This appendix is the field guide. It draws the whole map first, then zooms in: the physics of a stored bit, how a cache decides what to keep, what memories sit inside real CPUs, GPUs, TPUs and phones, the small memory chips hiding all over a motherboard, and the new memory technologies trying to break in. [Chapter 9](./09_memory_types.md) covers SRAM, DRAM and HBM in depth; [Chapter 10](./10_memory_management.md) covers virtual memory and coherence; [Chapter 11](./11_memory_wall_and_bandwidth.md) explains why all of this decides your model's speed.

> **The one-sentence version:** Fast memory is small and expensive per bit, big memory is slow and cheap, so every chip stacks several kinds — registers, then SRAM caches (L1, L2, L3), then DRAM or HBM, then flash — and spends a great deal of silicon guessing which bytes you will need next.

<div class="stats">
  <div class="stat"><div class="stat-num">~<em>1</em> ns</div><div class="stat-label">an L1 cache hit (4–5 clock cycles)</div></div>
  <div class="stat"><div class="stat-num">~<em>100</em> ns</div><div class="stat-label">a trip to DRAM: ~300 cycles of waiting</div></div>
  <div class="stat"><div class="stat-num">~<em>80</em> µs</div><div class="stat-label">a random read from an NVMe SSD</div></div>
  <div class="stat"><div class="stat-num"><em>6</em> vs <em>1</em></div><div class="stat-label">transistors per bit: SRAM vs DRAM</div></div>
  <div class="stat"><div class="stat-num"><em>64</em> B</div><div class="stat-label">a cache line: the unit every level moves</div></div>
</div>

---

## The Map: Every Memory, From the Core Outward

Start at the arithmetic units and walk outward. Each step is roughly **10× bigger and 3–10× slower** than the one before it, and each is built from a different physical kind of memory.

<div class="diagram">
<div class="diagram-title">The memory hierarchy of a modern computer (fastest at the top)</div>
<div class="layer-stack">
  <div class="layer accent"><strong>Registers</strong> — flip-flops / register-file SRAM inside each core · bytes to KBs · ~0.3 ns · read every cycle</div>
  <div class="layer green"><strong>L1 cache</strong> (split into instruction L1i and data L1d) — SRAM, private to one core · 32–192 KB · ~1 ns</div>
  <div class="layer teal"><strong>L2 cache</strong> — SRAM, per core or per small cluster · 1–16 MB · ~3–5 ns</div>
  <div class="layer cyan"><strong>L3 / last-level cache (LLC) / system-level cache (SLC)</strong> — SRAM, shared by all cores (and on phones, the GPU and NPU too) · 8–256 MB · ~10–20 ns</div>
  <div class="layer blue"><strong>Main memory</strong> — DRAM: DDR5 DIMMs, LPDDR5X on the package, or HBM stacks beside a GPU · GBs to TBs · ~80–120 ns</div>
  <div class="layer purple"><strong>Memory expansion</strong> — DRAM behind a CXL link, or a neighbour's memory over the network · TBs · ~200 ns to µs</div>
  <div class="layer orange"><strong>Storage</strong> — NAND flash SSDs (and spinning disks for cold data) · TBs to PBs · ~10–100 µs (SSD), ~5 ms (HDD)</div>
</div>
</div>

The numbers on that stack span **eight orders of magnitude** in latency. A log-scale bar is the only honest way to draw them:

<div class="specbars">
<div class="sb-title">How long one random read takes (log scale, 0.1 ns → 10 ms)</div>
<div class="sb-row"><span class="sb-name">Register</span><span class="sb-track"><i style="--v:0.06"></i></span><span class="sb-val">~0.3 ns</span></div>
<div class="sb-row"><span class="sb-name">L1 cache</span><span class="sb-track"><i style="--v:0.125"></i></span><span class="sb-val">~1 ns</span></div>
<div class="sb-row"><span class="sb-name">L2 cache</span><span class="sb-track"><i style="--v:0.2"></i></span><span class="sb-val">~4 ns</span></div>
<div class="sb-row"><span class="sb-name">L3 cache</span><span class="sb-track"><i style="--v:0.27"></i></span><span class="sb-val">~15 ns</span></div>
<div class="sb-row"><span class="sb-name">DRAM (DDR5)</span><span class="sb-track"><i class="k2" style="--v:0.375"></i></span><span class="sb-val">~100 ns</span></div>
<div class="sb-row"><span class="sb-name">CXL-attached DRAM</span><span class="sb-track"><i class="k2" style="--v:0.425"></i></span><span class="sb-val">~200–300 ns</span></div>
<div class="sb-row"><span class="sb-name">NVMe SSD (NAND)</span><span class="sb-track"><i class="k4" style="--v:0.74"></i></span><span class="sb-val">~80 µs</span></div>
<div class="sb-row"><span class="sb-name">Hard disk</span><span class="sb-track"><i class="k5" style="--v:0.96"></i></span><span class="sb-val">~5 ms</span></div>
<div class="sb-note">Typical values for a 2025-era server CPU; exact figures vary by product and load. Bandwidth tells a similar story in reverse: an L1 can deliver several TB/s to one core, while a DRAM channel delivers ~50 GB/s.</div>
</div>

Numbers this far apart are hard to feel. Scale them so that an L1 hit takes one second:

<div class="receipts">
<div class="receipt">
<div class="r-head">LATENCY RECEIPT</div>
<div class="r-sub">one read · scaled so 1 ns = 1 second</div>
<div class="r-rule"></div>
<div class="r-line"><span>L1 hit</span><span>1 s</span></div>
<div class="r-line"><span>L2 hit</span><span>4 s</span></div>
<div class="r-line"><span>L3 hit</span><span>15 s</span></div>
<div class="r-line"><span>DRAM</span><span>1 min 40 s</span></div>
<div class="r-rule"></div>
<div class="r-line hl"><span>NVMe SSD</span><span>~22 hours</span></div>
<div class="r-line hl"><span>Hard disk</span><span>~2 months</span></div>
<div class="r-rule"></div>
<div class="r-total"><span>DRAM / L1</span><span>~100×</span></div>
<div class="r-foot">why every chip spends area on caches</div>
<div class="r-barcode"></div>
</div>
</div>

### Why not just build one big, fast memory?

Because three things fight each other, and no single technology wins all three:

| Property | Wants… | Why it can't have everything |
|---|---|---|
| **Speed** | small arrays, short wires, many transistors per bit | Wire delay grows with distance, so a big array is a slow array, whatever it is made of |
| **Capacity** | the fewest, smallest devices per bit | A DRAM bit is one transistor and a capacitor; an SRAM bit is six transistors — ~10–20× the area |
| **Cost and persistence** | cheap bits that survive power-off | Flash stores charge for years, but writing it takes microseconds and wears it out |

So designers do the next best thing: build a small amount of very fast memory, a large amount of cheap memory, and arrange things so that **most accesses hit the small fast one**. That trick only works because programs have *locality*, which is where caches come in.

---

## Three Ways to Store a Bit

Almost every memory on a chip is one of three physical ideas: **a latch** (SRAM, registers), **a charged capacitor** (DRAM, HBM) or **trapped charge in a transistor** (flash). Newer memories store a bit as a *resistance* instead.

<figure class="figure"><svg viewBox="0 0 900 300" role="img" aria-label="Three memory cells side by side: an SRAM latch of six transistors, a DRAM cell of one transistor and one capacitor, and a flash cell whose transistor traps electrons in a charge-storage layer" style="max-width:100%;height:auto;font-family:Inter,system-ui,sans-serif"><rect width="900" height="300" fill="#141414"/><g fill="none" stroke="#5a5a5a" stroke-width="1"><rect x="14" y="14" width="280" height="272" rx="8"/><rect x="310" y="14" width="280" height="272" rx="8"/><rect x="606" y="14" width="280" height="272" rx="8"/></g><text x="154" y="40" text-anchor="middle" fill="#e8916a" font-size="15" font-weight="700">SRAM · 6 transistors</text><text x="450" y="40" text-anchor="middle" fill="#a78bfa" font-size="15" font-weight="700">DRAM · 1 transistor + 1 capacitor</text><text x="746" y="40" text-anchor="middle" fill="#22d3ee" font-size="15" font-weight="700">Flash · 1 charge-trap transistor</text><line x1="34" y1="64" x2="274" y2="64" stroke="#fbbf24" stroke-width="2"/><text x="40" y="58" fill="#fbbf24" font-size="10">word line</text><line x1="50" y1="64" x2="50" y2="196" stroke="#60a5fa" stroke-width="2"/><line x1="258" y1="64" x2="258" y2="196" stroke="#60a5fa" stroke-width="2"/><text x="50" y="210" fill="#60a5fa" font-size="10" text-anchor="middle">BL</text><text x="258" y="210" fill="#60a5fa" font-size="10" text-anchor="middle">BL̄</text><g stroke="#e5e5e5" stroke-width="1.6" fill="none"><path d="M118 108 L148 120 L118 132 Z"/><circle cx="152" cy="120" r="4"/><path d="M190 158 L160 170 L190 182 Z"/><circle cx="156" cy="170" r="4"/><path d="M98 120 L118 120"/><path d="M156 120 L210 120 L210 170 L190 170"/><path d="M152 170 L98 170 L98 120"/><path d="M50 120 L62 120"/><path d="M86 120 L98 120"/><path d="M210 170 L222 170"/><path d="M246 170 L258 170"/></g><rect x="62" y="112" width="24" height="16" rx="3" fill="#2a2a2a" stroke="#fbbf24"/><rect x="222" y="162" width="24" height="16" rx="3" fill="#2a2a2a" stroke="#fbbf24"/><line x1="74" y1="64" x2="74" y2="112" stroke="#fbbf24" stroke-width="1.2" stroke-dasharray="3 3"/><line x1="234" y1="64" x2="234" y2="162" stroke="#fbbf24" stroke-width="1.2" stroke-dasharray="3 3"/><circle cx="98" cy="120" r="2.5" fill="#e8916a"/><circle cx="210" cy="170" r="2.5" fill="#e8916a"/><text x="104" y="160" fill="#e8916a" font-size="9.5">Q</text><text x="198" y="112" fill="#e8916a" font-size="9.5">Q̄</text><text x="154" y="236" text-anchor="middle" fill="#a3a3a3" font-size="11">two inverters hold each other's value</text><text x="154" y="252" text-anchor="middle" fill="#a3a3a3" font-size="11">(4 transistors) + 2 access transistors</text><text x="154" y="278" text-anchor="middle" fill="#e5e5e5" font-size="11.5">fast · no refresh · big · volatile</text><line x1="330" y1="64" x2="570" y2="64" stroke="#fbbf24" stroke-width="2"/><text x="336" y="58" fill="#fbbf24" font-size="10">word line</text><line x1="370" y1="64" x2="370" y2="196" stroke="#60a5fa" stroke-width="2"/><text x="370" y="210" fill="#60a5fa" font-size="10" text-anchor="middle">bit line</text><rect x="416" y="112" width="34" height="22" rx="3" fill="#2a2a2a" stroke="#fbbf24"/><text x="433" y="127" text-anchor="middle" fill="#e5e5e5" font-size="10">T</text><line x1="433" y1="64" x2="433" y2="112" stroke="#fbbf24" stroke-width="1.2" stroke-dasharray="3 3"/><line x1="370" y1="123" x2="416" y2="123" stroke="#e5e5e5" stroke-width="1.6"/><path d="M450 123 L500 123 L500 150" stroke="#e5e5e5" stroke-width="1.6" fill="none"/><line x1="476" y1="150" x2="524" y2="150" stroke="#a78bfa" stroke-width="3"/><line x1="476" y1="162" x2="524" y2="162" stroke="#a78bfa" stroke-width="3"/><line x1="500" y1="162" x2="500" y2="190" stroke="#e5e5e5" stroke-width="1.6"/><line x1="486" y1="190" x2="514" y2="190" stroke="#e5e5e5" stroke-width="1.6"/><g fill="#a78bfa"><circle cx="488" cy="144" r="2.6"/><circle cx="500" cy="144" r="2.6"/><circle cx="512" cy="144" r="2.6"/></g><text x="530" y="160" fill="#a78bfa" font-size="10">C ≈ 10 fF</text><text x="450" y="226" text-anchor="middle" fill="#a3a3a3" font-size="11">the bit is charge on a tiny capacitor;</text><text x="450" y="242" text-anchor="middle" fill="#a3a3a3" font-size="11">it leaks, so every cell is refreshed every ~32–64 ms</text><text x="450" y="278" text-anchor="middle" fill="#e5e5e5" font-size="11.5">dense · cheap · needs refresh · volatile</text><rect x="660" y="190" width="172" height="20" fill="#2a3550" stroke="#5a5a5a"/><text x="746" y="204" text-anchor="middle" fill="#a3a3a3" font-size="10">channel (silicon)</text><rect x="680" y="160" width="132" height="26" fill="#0f3b44" stroke="#22d3ee"/><g fill="#22d3ee"><circle cx="700" cy="173" r="3"/><circle cx="716" cy="173" r="3"/><circle cx="732" cy="173" r="3"/><circle cx="748" cy="173" r="3"/><circle cx="764" cy="173" r="3"/><circle cx="780" cy="173" r="3"/><circle cx="796" cy="173" r="3"/></g><text x="818" y="166" fill="#22d3ee" font-size="10">trapped</text><text x="818" y="179" fill="#22d3ee" font-size="10">electrons</text><rect x="680" y="120" width="132" height="34" fill="#2a2a2a" stroke="#fbbf24"/><text x="746" y="141" text-anchor="middle" fill="#fbbf24" font-size="10.5">control gate (word line)</text><text x="746" y="96" text-anchor="middle" fill="#a3a3a3" font-size="11">electrons pushed through an insulator</text><text x="746" y="112" text-anchor="middle" fill="#a3a3a3" font-size="11">shift the transistor's threshold</text><text x="746" y="236" text-anchor="middle" fill="#a3a3a3" font-size="11">3D NAND stacks 200–300+ layers of these;</text><text x="746" y="252" text-anchor="middle" fill="#a3a3a3" font-size="11">TLC/QLC store 3–4 bits per cell</text><text x="746" y="278" text-anchor="middle" fill="#e5e5e5" font-size="11.5">dense · keeps data unpowered · slow writes · wears out</text></svg><figcaption>The three workhorse cells, simplified. Registers and caches use the SRAM latch; main memory and HBM use the DRAM cell; SSDs, phone storage and firmware chips use flash.</figcaption></figure>

The full menu of memory technologies you will meet on or around a chip:

| Technology | How a bit is stored | Keeps data when off? | Speed (read) | Density | Where you find it |
|---|---|:---:|---|---|---|
| **Flip-flops / latches** | Cross-coupled gates, clocked | No | Fastest (same cycle) | Lowest | Pipeline registers, architectural registers, state machines |
| **Register-file SRAM** | Multi-ported SRAM | No | Sub-ns | Low | CPU physical register files; GPU register files (256 KB per H100 SM) |
| **SRAM** | 6-transistor latch | No | ~1–20 ns (depends on array size) | ~0.02 µm² per bit at 3 nm | L1/L2/L3 caches, GPU shared memory, TLBs, scratchpads, Groq and Cerebras main memory |
| **eDRAM** | Capacitor built in a logic process | No | ~Few ns | ~3× SRAM | Older IBM POWER/z L3/L4 caches, some game consoles; rare today |
| **DRAM** (DDR, LPDDR, GDDR) | 1 transistor + 1 capacitor | No (refreshed every ~64 ms) | ~50–100 ns | ~0.001–0.002 µm² per bit | Main memory: DIMMs, phone/laptop LPDDR, graphics GDDR |
| **HBM** | DRAM dies stacked with TSVs | No | Similar to DRAM; huge width | Same cell as DRAM | Beside AI accelerators on an interposer ([Appendix E](./appendix_e_packaging.md)) |
| **NAND flash** | Charge trapped in a 3D-stacked transistor | Yes (years) | ~20–100 µs; writes slower | Highest of all (TLC/QLC, 200–300+ layers) | SSDs, phone storage (UFS), USB sticks |
| **NOR flash** | Charge-trap / floating-gate transistor, random-access | Yes | ~50–100 ns read; very slow writes | Low | Firmware chips (UEFI/BIOS, GPU VBIOS), microcontroller program memory |
| **Mask ROM** | Wired in at manufacture | Yes | Fast | High | Boot ROM inside every SoC: the first code after reset |
| **eFuse / OTP** | A link burned open, once | Yes (forever) | Read at boot | Tiny (bits to KBs) | Chip serial numbers, security keys, which spare cores are disabled |
| **MRAM** (STT/SOT) | Magnetic orientation of a nanoscale junction | Yes | ~10 ns | Medium | Embedded non-volatile memory in microcontrollers at 22–28 nm, replacing embedded flash |
| **RRAM / ReRAM** | Resistance of a filament formed in an oxide | Yes | ~10–100 ns | Medium–high | Embedded NVM in some MCUs; research on analog compute-in-memory |
| **PCM** | Crystalline vs amorphous glass (resistance) | Yes | ~100s ns | High | Intel Optane / 3D XPoint (discontinued 2022); research on analog AI |
| **FeRAM / FeFET** | Polarization of a ferroelectric | Yes | ~10s ns | Medium | Low-power MCUs, smart cards; ferroelectric DRAM and FeFET are in research |

---

## Caches: How a Small Memory Pretends to Be a Big One

A **cache** is a small, fast memory that holds copies of recently used data from a bigger, slower one, and it is managed *by hardware*, invisibly to your program. Caches work because real programs have **locality**:

- **Temporal locality** — data you used recently, you will probably use again soon (a loop counter, a weight tile reused across a batch).
- **Spatial locality** — data *near* what you just used will probably be used next (the next element of an array).

So caches never move single bytes. They move **cache lines**: 64-byte aligned blocks on essentially every modern CPU (128 bytes on some Apple and GPU caches, which track 32-byte sectors). Read one float and its 15 neighbours come along for free; walk an array in order and 15 of every 16 four-byte reads are hits.

### Where can a line go? Sets, ways and tags

A cache is a table of **sets**; each set has a few **ways** (slots). The address itself says which set a line must go in:

<div class="bitfield"><div class="bf-seg payload" style="flex:36">tag<small>36 bits — which line is this?</small></div><div class="bf-seg exponent" style="flex:6">set index<small>6 bits → 64 sets</small></div><div class="bf-seg mantissa" style="flex:6">offset<small>6 bits → 64 B</small></div></div>
<div class="bitfield-label">A 48-bit address in a 32 KB, 8-way cache with 64-byte lines: 32 KB ÷ 64 B ÷ 8 ways = 64 sets</div>

On every access the hardware uses the **index** bits to pick one set, compares the **tag** against all the ways in that set *in parallel*, and uses the **offset** to pick bytes within the line. The number of ways is the design dial:

<div class="diagram">
<div class="diagram-title">Associativity: how many places one line may live</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card red">
    <div class="card-title">Direct-mapped (1 way)</div>
    <div class="card-desc">Each line has exactly one possible slot. Fastest and cheapest lookup, but two hot lines that share an index keep evicting each other (<em>conflict misses</em>).</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-title">Set-associative (4–16 ways)</div>
    <div class="card-desc">Each line may go in any way of its set. The standard compromise: L1 caches are typically 8–12-way, L2 and L3 8–20-way.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Fully associative</div>
    <div class="card-desc">Any line, any slot: no conflict misses, but every lookup compares every tag. Used only for small structures such as TLBs and victim buffers.</div>
  </div>
</div>
</div>

Misses come in three kinds, the "three Cs": **compulsory** (the first touch of a line — unavoidable without prefetching), **capacity** (the working set is bigger than the cache) and **conflict** (the cache had room, but this line's set did not). The lab below lets you produce each one on purpose.

### Try It: Simulate a Cache

Start with the defaults: a 16 KB array walked twice through a 32 KB cache. The first pass misses once per 64-byte line (yellow), the second pass hits everything. Now raise the working set to 64 KB: the second pass misses again (blue, capacity). Finally choose *walk matrix columns*, a 256 KB working set and a direct-mapped cache: rows that are 4 KB apart all land in the same few sets and evict each other (red, conflict) even though the cache is mostly empty. Tick *pad matrix rows* and the conflicts disappear. That padding trick is exactly what GPU and CPU kernel writers do with shared-memory and array layouts.

<div class="lab" data-lab="cache"><noscript><div class="lab-noscript">This lab needs JavaScript. It is also on the <a href="./labs.html">Interactive Labs</a> page.</div></noscript></div>

### The policies every cache has to choose

| Decision | Options | What real chips do |
|---|---|---|
| **Which line to evict** | LRU, pseudo-LRU, random, RRIP | Approximations of LRU; L3s use smarter "re-reference" predictors so one big scan doesn't flush everything |
| **When to write to memory** | Write-through (immediately) or write-back (when evicted) | Write-back almost everywhere: a dirty bit marks lines that must be written out |
| **Inclusive or exclusive** | Does L3 hold a copy of everything in L1/L2? | Mixed: Intel moved from inclusive to non-inclusive L3s; AMD's L3 is a victim cache of the L2s |
| **Fetch ahead?** | Prefetchers that spot sequential or strided streams | Several per core; they turn many compulsory misses into hits |
| **Keep cores consistent** | Coherence protocols (MESI, MOESI) and snoop filters | Every multicore CPU; GPUs keep L1s loosely coherent and rely on L2 ([Chapter 10](./10_memory_management.md)) |

### "Caches" that don't hold your data

The same idea — keep a small table of recent answers close to where they're needed — appears all over a processor:

<div class="diagram">
<div class="diagram-grid cols-3">
  <div class="diagram-card cyan"><div class="card-title">TLB</div><div class="card-desc">Translation lookaside buffer: caches virtual→physical address translations so every load doesn't walk the page tables. Usually two levels, a few thousand entries.</div></div>
  <div class="diagram-card purple"><div class="card-title">µop cache</div><div class="card-desc">Holds already-decoded x86 instructions, so hot loops skip the expensive decoders.</div></div>
  <div class="diagram-card orange"><div class="card-title">Branch predictor tables</div><div class="card-desc">Caches of past branch outcomes and targets (BTB). Several KB of SRAM that decide whether the pipeline guesses right.</div></div>
  <div class="diagram-card green"><div class="card-title">Store buffer</div><div class="card-desc">Holds writes until they can enter the L1, so a store never makes the core wait.</div></div>
  <div class="diagram-card blue"><div class="card-title">Constant / texture caches</div><div class="card-desc">Small read-only GPU caches for data every thread reads, such as kernel parameters.</div></div>
  <div class="diagram-card pink"><div class="card-title">DRAM row buffer</div><div class="card-desc">Each DRAM bank keeps the last opened row (~1–8 KB) in sense amplifiers: a cache inside the memory chip itself.</div></div>
</div>
</div>

### Where one load goes

Put it together and a single load instruction walks down the hierarchy until something answers:

<!-- flowchart:memory_lookup -->
<div class="flowchart" id="fc-memory_lookup">
<div class="fc-title">Where does one load instruction find its data? (a modern CPU core)</div>
<svg viewBox="0 0 780 580" width="780" role="img" aria-label="Where does one load instruction find its data? (a modern CPU core)">
<path class="fc-edge" d="M230,48 L230,72"/><polygon class="fc-head" points="230,72 226.2,65 233.8,65"/>
<path class="fc-edge" d="M230,128 L230,159"/><polygon class="fc-head" points="230,159 226.2,152 233.8,152"/>
<path class="fc-edge" d="M230,213 L230,245"/><polygon class="fc-head" points="230,245 226.2,238 233.8,238"/><text class="fc-elabel " x="236" y="227" text-anchor="start">miss</text>
<path class="fc-edge" d="M230,299 L230,331"/><polygon class="fc-head" points="230,331 226.2,324 233.8,324"/><text class="fc-elabel " x="236" y="313" text-anchor="start">miss</text>
<path class="fc-edge" d="M230,385 L230,417"/><polygon class="fc-head" points="230,417 226.2,410 233.8,410"/><text class="fc-elabel " x="236" y="399" text-anchor="start">miss</text>
<path class="fc-edge no" d="M230,471 L230,508"/><polygon class="fc-head no" points="230,508 226.2,501 233.8,501"/><text class="fc-elabel no" x="236" y="485" text-anchor="start">no</text>
<path class="fc-edge" d="M380,186 L485,186"/><polygon class="fc-head" points="485,186 478,189.8 478,182.2"/><text class="fc-elabel " x="390" y="180" text-anchor="start">hit</text>
<path class="fc-edge" d="M380,272 L485,272"/><polygon class="fc-head" points="485,272 478,275.9 478,268.1"/><text class="fc-elabel " x="390" y="266" text-anchor="start">hit</text>
<path class="fc-edge" d="M380,358 L485,358"/><polygon class="fc-head" points="485,358 478,361.9 478,354.1"/><text class="fc-elabel " x="390" y="352" text-anchor="start">hit</text>
<path class="fc-edge" d="M380,444 L485,444"/><polygon class="fc-head" points="485,444 478,447.9 478,440.1"/><text class="fc-elabel " x="390" y="438" text-anchor="start">hit</text>
<g class="fc-node start"><rect x="120" y="4" width="220" height="44" rx="16"/><text x="230" y="24" text-anchor="middle">Core executes a load</text><text class="sub" x="230" y="37" text-anchor="middle">virtual address</text></g>
<a href="./10_memory_management.html"><g class="fc-node k-cyan"><rect x="80" y="72" width="300" height="56" rx="6"/><text x="230" y="92" text-anchor="middle">Translate address</text><text class="sub" x="230" y="105" text-anchor="middle">TLB hit ≈ free; miss → page-table walk</text><text class="sub" x="230" y="117" text-anchor="middle">(itself cached in L1/L2)</text></g></a>
<g class="fc-node decision"><polygon points="80,186 94,159 366,159 380,186 366,213 94,213"/><text x="230" y="184" text-anchor="middle">In L1 data cache?</text><text class="sub" x="230" y="197" text-anchor="middle">32–64 KB per core, SRAM</text></g>
<g class="fc-node decision"><polygon points="80,272 94,245 366,245 380,272 366,299 94,299"/><text x="230" y="270" text-anchor="middle">In L2 cache?</text><text class="sub" x="230" y="283" text-anchor="middle">1–3 MB per core, SRAM</text></g>
<g class="fc-node decision"><polygon points="80,358 94,331 366,331 380,358 366,385 94,385"/><text x="230" y="356" text-anchor="middle">In L3 / last-level cache?</text><text class="sub" x="230" y="369" text-anchor="middle">tens–hundreds of MB, shared</text></g>
<g class="fc-node decision"><polygon points="80,444 94,417 366,417 380,444 366,471 94,471"/><text x="230" y="442" text-anchor="middle">In DRAM?</text><text class="sub" x="230" y="455" text-anchor="middle">GBs–TBs, DDR5 / LPDDR5X / HBM</text></g>
<g class="fc-node leaf k-green"><rect x="485" y="164" width="250" height="44" rx="6"/><text x="610" y="184" text-anchor="middle">L1 hit · ~1 ns</text><text class="sub" x="610" y="197" text-anchor="middle">4–5 cycles</text></g>
<g class="fc-node leaf k-teal"><rect x="485" y="250" width="250" height="44" rx="6"/><text x="610" y="270" text-anchor="middle">L2 hit · ~3–5 ns</text><text class="sub" x="610" y="283" text-anchor="middle">~12–16 cycles</text></g>
<g class="fc-node leaf k-blue"><rect x="485" y="336" width="250" height="44" rx="6"/><text x="610" y="356" text-anchor="middle">L3 hit · ~10–20 ns</text><text class="sub" x="610" y="369" text-anchor="middle">~40–70 cycles</text></g>
<a href="./09_memory_types.html"><g class="fc-node leaf k-purple"><rect x="485" y="422" width="250" height="44" rx="6"/><text x="610" y="442" text-anchor="middle">DRAM · ~80–120 ns</text><text class="sub" x="610" y="455" text-anchor="middle">~300 cycles; line also filled into L3/L2/L1</text></g></a>
<a href="./appendix_h_how_a_computer_runs.html"><g class="fc-node leaf k-red"><rect x="80" y="508" width="300" height="44" rx="6"/><text x="230" y="528" text-anchor="middle">Page fault → OS reads from SSD</text><text class="sub" x="230" y="541" text-anchor="middle">~10–100 µs: a million cycles</text></g></a>
</svg>
<div class="fc-caption">Each "no" costs roughly 3–10× more time than the level above it. A GPU follows the same path with registers, L1/shared memory, L2 and HBM, but every step takes several times longer, which is why GPUs hide latency with thousands of threads instead.</div>
</div>
<!-- /flowchart:memory_lookup -->

---

## Inside Real Chips: Who Has Which Memory

### A CPU core and its package

<div class="die-wrap">
<div class="die-frame">
<div class="die" style="grid-template-columns:repeat(4,1fr);grid-template-areas:'c0 c1 c2 c3' 'l3 l3 l3 l3' 'c4 c5 c6 c7' 'mc mc io io';max-width:640px">
<div class="blk cpu" style="grid-area:c0">Core 0<small>regs · L1i 32K · L1d 48K · L2 1M</small></div><div class="blk cpu" style="grid-area:c1">Core 1<small>regs · L1 · L2 1M</small></div><div class="blk cpu" style="grid-area:c2">Core 2<small>regs · L1 · L2 1M</small></div><div class="blk cpu" style="grid-area:c3">Core 3<small>regs · L1 · L2 1M</small></div>
<div class="blk cache" style="grid-area:l3;min-height:52px">Shared L3 · 32 MB SRAM<small>every core can hit any slice; X3D parts stack another 64 MB of SRAM on top</small></div>
<div class="blk cpu" style="grid-area:c4">Core 4<small>regs · L1 · L2 1M</small></div><div class="blk cpu" style="grid-area:c5">Core 5<small>regs · L1 · L2 1M</small></div><div class="blk cpu" style="grid-area:c6">Core 6<small>regs · L1 · L2 1M</small></div><div class="blk cpu" style="grid-area:c7">Core 7<small>regs · L1 · L2 1M</small></div>
<div class="blk mem" style="grid-area:mc">DDR5 memory controllers<small>(on the I/O die) → DIMMs</small></div><div class="blk io" style="grid-area:io">PCIe / CXL<small>→ SSDs, GPUs, CXL memory</small></div>
</div>
</div>
<div class="die-legend"><span style="--k:rgba(249,115,22,.7)">cores with private registers, L1, L2</span><span style="--k:rgba(59,130,246,.7)">shared L3</span><span style="--k:rgba(168,85,247,.7)">path to DRAM</span><span style="--k:rgba(148,163,184,.7)">I/O</span></div>
<div class="die-caption">An AMD Zen 5 compute die (CCD), stylised: eight cores, each with 32 KB L1i, 48 KB L1d and 1 MB L2, sharing a 32 MB L3. The memory controllers live on a separate I/O die (<a href="./14_cpus.html">Chapter 14</a>).</div>
</div>

### A GPU: more registers than cache

<div class="die-wrap">
<div class="die-frame">
<div class="hbm-col"><div class="hbm">HBM3</div><div class="hbm">HBM3</div><div class="hbm">HBM3</div></div>
<div class="die" style="grid-template-columns:repeat(4,1fr);grid-template-areas:'s1 s2 s3 s4' 'l2 l2 l2 l2' 's5 s6 s7 s8';max-width:560px">
<div class="blk compute" style="grid-area:s1">SM<small>256 KB regs · 256 KB L1/shared</small></div><div class="blk compute" style="grid-area:s2">SM<small>256 KB regs · 256 KB L1/shared</small></div><div class="blk compute" style="grid-area:s3">SM<small>× 132 in all</small></div><div class="blk compute" style="grid-area:s4">SM<small>…</small></div>
<div class="blk cache" style="grid-area:l2;min-height:52px">L2 cache · 50 MB SRAM<small>shared by all SMs; every HBM byte passes through it</small></div>
<div class="blk compute" style="grid-area:s5">SM</div><div class="blk compute" style="grid-area:s6">SM</div><div class="blk compute" style="grid-area:s7">SM</div><div class="blk compute" style="grid-area:s8">SM</div>
</div>
<div class="hbm-col"><div class="hbm">HBM3</div><div class="hbm">HBM3</div><div class="hbm">80 GB total</div></div>
</div>
<div class="die-legend"><span style="--k:rgba(232,145,106,.7)">SMs: register file + L1/shared memory</span><span style="--k:rgba(59,130,246,.7)">L2</span><span style="--k:#5d4790">HBM stacks (DRAM)</span></div>
<div class="die-caption">NVIDIA H100, stylised: ~33 MB of registers and ~33 MB of L1/shared memory spread over 132 SMs, a 50 MB L2, and 80 GB of HBM3 on the package. A GPU holds more data in registers than in its L1s, the opposite of a CPU.</div>
</div>

The striking thing about a GPU is how much of its fast memory is **software-managed**. A CPU's caches decide for you; a GPU kernel explicitly copies tiles into **shared memory** (a programmer-controlled slice of the L1 SRAM), and Blackwell adds a further 256 KB per SM of **Tensor Memory** that holds matrix-multiply accumulators right next to the tensor cores. FlashAttention, CUTLASS and every fast GPU kernel are, at heart, careful plans for which tile sits in which of these memories when.

### The memory map across chip types

| Chip | Registers | Private / near cache | Shared on-chip | Main memory | Notes |
|---|---|---|---|---|---|
| **Server CPU** (AMD Zen 5) | ~200+ physical int + vector regs per core | L1i 32 KB, L1d 48 KB, L2 1 MB per core | L3 32 MB per 8-core die (+64 MB stacked on X3D) | DDR5, 12 channels per socket, up to TBs | Caches managed entirely by hardware |
| **Laptop SoC** (Apple M4) | per core | L1i 192 KB, L1d 128 KB per performance core | 16 MB L2 per P-core cluster; system-level cache shared with GPU and NPU | LPDDR5X on the package, unified for CPU/GPU/NPU | No separate GPU memory: one pool ([Chapter 22](./22_socs.md)) |
| **Phone SoC** | per core | L1 + L2 per core or cluster | L3 + system-level cache (SLC) for CPU, GPU, NPU, ISP | LPDDR5X stacked on top of the SoC (package-on-package) | The SLC lets the camera, NPU and GPU share data without touching DRAM |
| **GPU** (H100) | 256 KB per SM (~33 MB total) | 256 KB L1/shared per SM | 50 MB L2 | 80 GB HBM3 at 3.35 TB/s | Shared memory is software-managed |
| **GPU** (B200/B300) | 256 KB per SM | L1/shared + 256 KB Tensor Memory per SM | Large L2 split across two dies | 192–288 GB HBM3e at 8 TB/s | Tensor Memory feeds 5th-gen tensor cores |
| **AMD MI300X** | per CU | L1 + LDS per CU; 4 MB L2 per compute die | 256 MB Infinity Cache on the I/O dies | 192 GB HBM3 | The big shared cache sits under the compute dies |
| **Google TPU** | vector registers | Software-managed vector memory (VMEM) next to the MXUs | v4 added a 128 MB common memory (CMEM) | 16–288 GB HBM per chip | No hardware caches for the matrix path: the compiler (XLA) schedules every move |
| **Groq LPU** | — | — | 230 MB SRAM *is* the memory | none | Compiler-scheduled; many chips per model |
| **Cerebras WSE-3** | per core | 48 KB SRAM next to each of 900,000 cores | 44 GB SRAM across the wafer | external MemoryX for training | 21 PB/s of on-chip bandwidth |
| **Microcontroller** | a few | sometimes a tiny cache | 64 KB–2 MB SRAM | NOR flash or MRAM for program code | Code often runs directly from flash |

Two design philosophies run through this table. **Hardware-managed caches** (CPUs, most of a GPU's L1 and L2) make programs fast without the programmer's help. **Software-managed scratchpads** (GPU shared memory and Tensor Memory, TPU VMEM, Groq, Cerebras) waste no energy on tags and guesswork, but only pay off when a compiler or kernel author plans every byte. AI accelerators lean hard toward the second, because neural-network access patterns are known in advance.

---

## The Memory Chips You Never Think About

Look at a server or laptop board and you will find a dozen *other* memories, each doing a small job:

<div class="die-wrap">
<div class="die-frame">
<div class="die" style="grid-template-columns:repeat(4,1fr);grid-template-areas:'cpu cpu dimm dimm' 'cpu cpu spi tpm' 'gpu gpu ssd ssd' 'gpu gpu bmc nic';max-width:680px">
<div class="blk cpu" style="grid-area:cpu;min-height:110px">CPU package<small>registers · L1/L2/L3 SRAM · mask-ROM boot code · eFuses (ID, keys, disabled cores)</small></div>
<div class="blk mem" style="grid-area:dimm">DDR5 DIMMs<small>DRAM chips + a small SPD EEPROM with the module's timings</small></div>
<div class="blk ctrl" style="grid-area:spi">SPI NOR flash<small>16–64 MB: UEFI firmware + settings</small></div>
<div class="blk ctrl" style="grid-area:tpm">TPM<small>secure non-volatile store for keys</small></div>
<div class="blk gpu" style="grid-area:gpu;min-height:110px">GPU board<small>HBM or GDDR · on-die SRAM · its own flash with the VBIOS</small></div>
<div class="blk media" style="grid-area:ssd">NVMe SSD<small>controller + DRAM cache (~1 GB per TB) + NAND dies; part of the NAND acts as a fast SLC write cache</small></div>
<div class="blk io" style="grid-area:bmc">BMC<small>tiny management computer with its own DRAM + flash</small></div>
<div class="blk io" style="grid-area:nic">NIC / DPU<small>firmware flash + packet SRAM (switches add TCAM)</small></div>
</div>
</div>
<div class="die-caption">A server board, as a memory map (not to scale). Every "chip" on it carries some memory of its own; most of them are firmware stores that the boot sequence reads before your operating system exists (<a href="./appendix_h_how_a_computer_runs.html">Appendix H</a>).</div>
</div>

A few of these deserve a sentence each:

- **Boot ROM and eFuses.** The first instructions a processor runs come from a small mask ROM on the die itself. eFuses, burned once at the factory, record the chip's identity, security keys and which spare cores or cache slices were disabled during [binning](./appendix_a_how_chips_are_made.md).
- **SPI NOR flash.** A small, slow, non-volatile chip holding the UEFI/BIOS firmware; GPUs, NICs, SSD controllers and baseboard management controllers each have their own.
- **SPD EEPROM.** A few hundred bytes on every memory module describing its size and timings, read by the firmware to train the memory interface.
- **Inside an SSD.** An SSD is a computer: a controller with its own CPU cores, a DRAM cache holding the map from logical blocks to NAND pages (budget SSDs borrow host memory instead, "HMB"), and NAND dies, part of which runs in fast single-bit-per-cell mode as a write cache.
- **TCAM.** Network switches use ternary content-addressable memory, which searches every entry at once by content instead of by address, to look up routes in a single cycle.

---

## Emerging and Future Memories

<div class="datasheet">
<div class="ds-head"><span class="ds-part">WATCHLIST</span><span class="ds-name">Memory technologies to know in 2026</span><span class="ds-stamp soon">moving fast</span><span class="ds-sub">what each one could replace, and how far along it is</span></div>
<div class="ds-grid">
<table><thead><tr><th colspan="2">In products</th></tr></thead><tbody>
<tr><td>HBM4 (2,048-bit stacks)</td><td class="v">Rubin, MI455X · 2026</td></tr>
<tr><td>3D-stacked SRAM cache</td><td class="v">AMD X3D, MI300</td></tr>
<tr><td>Embedded MRAM / RRAM</td><td class="v">MCUs at 22–28 nm</td></tr>
<tr><td>CXL memory expanders</td><td class="v">servers since 2024</td></tr>
<tr><td>LPDDR5X for data-centre CPUs</td><td class="v">NVIDIA Grace / Vera</td></tr>
</tbody></table>
<table><thead><tr><th colspan="2">Coming or in research</th></tr></thead><tbody>
<tr><td>HBM4E</td><td class="v">samples late 2026</td></tr>
<tr><td>High-bandwidth flash (HBF)</td><td class="v">proposed 2025</td></tr>
<tr><td>3D DRAM (vertical cells)</td><td class="v">research / pilot</td></tr>
<tr><td>FeRAM / FeFET memory</td><td class="v">research</td></tr>
<tr><td>Analog compute-in-memory (RRAM, PCM)</td><td class="v">research chips</td></tr>
</tbody></table>
</div>
<div class="ds-foot">See <a href="./35_where_the_industry_is_headed.html">Chapter 35</a> for HBM4, CXL and in-memory compute in context.</div>
</div>

- **SRAM has stopped shrinking.** A high-density SRAM bit was ~0.021 µm² at TSMC N5 and barely smaller at N3E; logic kept shrinking. That is why caches are moving onto *separate, stacked dies* (AMD's 3D V-Cache) and why every new accelerator agonises over how much SRAM to spend area on.
- **MRAM and RRAM** are winning the embedded job that flash can no longer do at advanced nodes: holding a microcontroller's program and data without power. Both are also the leading candidates for **analog compute-in-memory**, where a matrix of resistances performs a matrix-vector multiply in place ([Chapter 35](./35_where_the_industry_is_headed.md)).
- **Persistent memory** had its big test with Intel Optane (phase-change based), which was fast and durable but discontinued in 2022 for lack of a market between DRAM and flash. CXL-attached DRAM and fast SSDs now fill that gap.
- **High-bandwidth flash** — NAND stacked and wired like HBM — has been proposed (SanDisk and SK hynix announced work on a standard in 2025) as a way to hold huge model weights for read-mostly inference near an accelerator.

---

## Why This Matters for Model Optimization

<div class="diagram">
<div class="diagram-title">The hierarchy in your training and serving jobs</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent"><div class="card-title">Kernels are cache plans</div><div class="card-desc">FlashAttention's whole speed-up is keeping attention tiles in shared memory/SRAM instead of writing the T×T score matrix to HBM. Fused kernels win for the same reason.</div></div>
  <div class="diagram-card green"><div class="card-title">Decode lives below L2</div><div class="card-desc">A 70B model's weights (35–140 GB) never fit in a 50–126 MB L2, so batch-1 decode streams every weight from HBM each token. That is the memory wall of <a href="./11_memory_wall_and_bandwidth.html">Chapter 11</a>.</div></div>
  <div class="diagram-card blue"><div class="card-title">SRAM-first chips are a bet</div><div class="card-desc">Groq and Cerebras keep weights in SRAM to get enormous bandwidth, at the cost of needing many chips per model — the same trade-off as L1 vs DRAM, at rack scale.</div></div>
  <div class="diagram-card purple"><div class="card-title">Layout decides hit rates</div><div class="card-desc">Contiguous tensors, tile sizes that fit shared memory, and padding to dodge bank and set conflicts are cache-design ideas applied by hand.</div></div>
  <div class="diagram-card orange"><div class="card-title">Storage feeds the GPU</div><div class="card-desc">Datasets stream from NVMe through the OS page cache (DRAM) into pinned host memory and then HBM; checkpoints flow back the other way. A slow tier anywhere stalls the whole pipeline.</div></div>
  <div class="diagram-card cyan"><div class="card-title">KV caches are tiered too</div><div class="card-desc">Serving systems keep active KV blocks in HBM and park idle ones in host DRAM or CXL memory: a software cache hierarchy (<a href="./29_memory_hierarchy_in_action.html">Chapter 29</a>).</div></div>
</div>
</div>

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>A GPU spends more silicon on registers than on L1, a CPU the reverse, and Groq spends almost everything on SRAM with no DRAM at all. If you were designing a chip only for batch-1 LLM decode, where on this spectrum would you put your transistors, and what would you give up?</p>
</div>

## Key Takeaways

- Memory is a **hierarchy** because speed, capacity and cost per bit cannot all be had at once: registers → L1 → L2 → L3 → DRAM/HBM → CXL memory → flash, each level ~10× bigger and several times slower.
- **SRAM** (a 6-transistor latch) makes registers and caches; **DRAM** (one transistor and a capacitor) makes main memory, GDDR and HBM; **flash** (trapped charge) makes SSDs and firmware chips. MRAM, RRAM, PCM and FeRAM store a bit as resistance or polarisation and are filling niches.
- **Caches** exploit locality: they move 64-byte lines, place them by set and tag, and miss for three reasons — compulsory, capacity and conflict.
- The cache idea is everywhere: TLBs, µop caches, branch predictors, DRAM row buffers, SSD DRAM caches, KV-cache tiers.
- CPUs rely on hardware-managed caches; GPUs and AI accelerators rely more and more on **software-managed SRAM** (shared memory, Tensor Memory, VMEM, all-SRAM designs) planned by kernels and compilers.
- Dozens of small memories — boot ROM, eFuses, SPI NOR, SPD EEPROM, TPM, SSD DRAM — exist outside the main hierarchy, mostly to boot and identify the machine.

## Check Yourself

<details class="check">
<summary>A 64 KB, 4-way cache uses 64-byte lines. How many sets does it have, and how many address bits are used for the set index and the offset?</summary>
<p>64 KB ÷ 64 B = 1,024 lines; ÷ 4 ways = 256 sets. The index needs log₂ 256 = 8 bits and the offset log₂ 64 = 6 bits; the remaining high bits are the tag.</p>
</details>

<details class="check">
<summary>You sum a 1 GB float32 array once, in order. Roughly what fraction of loads hit in L1, and what kind are the misses?</summary>
<p>Each 64-byte line holds 16 floats, so 15 of 16 loads hit (~94%) and 1 in 16 is a compulsory miss — and in practice the hardware prefetcher fetches lines ahead, hiding most of those too.</p>
</details>

<details class="check">
<summary>Why can walking down a column of a matrix with 4,096-byte rows be slow even when the cache has plenty of free space?</summary>
<p>Addresses 4 KB apart share the same low-order index bits, so they all map to the same few sets and evict each other: conflict misses. Padding each row by one cache line spreads them across sets.</p>
</details>

<details class="check">
<summary>Why does DRAM need refreshing while SRAM does not, and what does that cost?</summary>
<p>A DRAM bit is charge on a tiny capacitor that leaks away within tens of milliseconds, so every row must be read and rewritten periodically (typically within 32–64 ms); that steals bandwidth and power. An SRAM latch actively holds its value as long as it is powered.</p>
</details>

<details class="check">
<summary>What is the difference between a GPU's L1 cache and its shared memory, if they are the same SRAM?</summary>
<p>The L1 part is managed by hardware (it decides what to keep, using tags); the shared-memory part is an address range the kernel explicitly loads and stores, with no tags or guessing. The split between them is configurable per kernel.</p>
</details>

<details class="check">
<summary>Name three memories on a server that hold data before the operating system starts.</summary>
<p>The CPU's boot ROM and eFuses, the SPI NOR flash with the UEFI firmware, the SPD EEPROMs on the DIMMs, and the firmware flash on the GPU, NIC, SSD and BMC.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Computer Architecture: A Quantitative Approach (6th ed.) | Hennessy & Patterson, 2017 | [elsevier.com/books/computer-architecture/hennes…](https://www.elsevier.com/books/computer-architecture/hennessy/978-0-12-811905-1) | Cache organisation, the three Cs, memory hierarchy design |
| What Every Programmer Should Know About Memory | Drepper, 2007 | [akkadia.org/drepper/cpumemory.pdf](https://www.akkadia.org/drepper/cpumemory.pdf) | Cache lines, associativity, prefetching, DRAM internals |
| NVIDIA H100 Tensor Core GPU Architecture | NVIDIA, 2022 | [resources.nvidia.com/en-us-tensor-core](https://resources.nvidia.com/en-us-tensor-core) | Register file, L1/shared memory and L2 sizes |
| CUDA C++ Programming Guide | NVIDIA | [docs.nvidia.com/cuda/cuda-c-programming-guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | Shared memory, memory spaces, Tensor Memory on Blackwell |
| AMD Zen 5 microarchitecture | Chips and Cheese, 2024 | [chipsandcheese.com](https://chipsandcheese.com/) | Zen 5 cache sizes and latencies |
| TPU v4: An Optically Reconfigurable Supercomputer | Jouppi et al., ISCA 2023 | [arxiv.org/abs/2304.01433](https://arxiv.org/abs/2304.01433) | TPU memory organisation (VMEM, CMEM, HBM) |
| Computing's Energy Problem (and what we can do about it) | Horowitz, ISSCC 2014 | [doi.org/10.1109/ISSCC.2014.6757323](https://doi.org/10.1109/ISSCC.2014.6757323) | Energy of SRAM vs DRAM accesses |
| FlashAttention | Dao et al., 2022 | [arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135) | Tiling attention into on-chip SRAM |
| In-memory computing with resistive switching devices | Ielmini & Wong, Nature Electronics 2018 | [doi.org/10.1038/s41928-018-0092-2](https://doi.org/10.1038/s41928-018-0092-2) | RRAM, PCM and MRAM; in-memory computing |

---

**Next:** [Back to the Table of Contents](./README.md) — You have reached the end of the appendices; the [Interactive Labs](./labs.md) and [Accelerator Atlas](./atlas.md) are good places to keep exploring.

[← Back to Table of Contents](./README.md)
