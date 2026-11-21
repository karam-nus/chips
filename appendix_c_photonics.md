---
title: "Appendix C — Photonics & Optical Interconnect"
---

[← Back to Table of Contents](./README.md)

# Appendix C — Photonics & Optical Interconnect

Every major bottleneck in large-scale AI training eventually traces back to moving data: between GPU registers and SRAM, between SRAM and HBM, between GPUs on a node, between nodes across a switch fabric, between racks across a datacenter. As explored in [Chapter 12](./12_buses_and_interconnects.md), interconnect bandwidth — not raw compute FLOPS — increasingly determines training throughput in large clusters. Electrical signaling has served magnificently for five decades, but physics is closing in: at the bandwidths required to feed clusters of thousands of GPUs, copper traces on a PCB consume too much power, cannot reach far enough, and cannot scale further without prohibitive cost.

Light offers an escape. Photons travel without resistive loss, can multiplex many channels onto a single fiber via wavelength-division multiplexing (WDM), and in the right architecture consume far less energy per bit than equivalent electrical signaling. The transition from pluggable optical modules to **co-packaged optics** — integrating the optics next to the compute die — is already underway and will reshape how AI clusters are built in the late 2020s.

Beyond interconnect, a more speculative idea is **optical computing**: performing the matrix multiplications of neural networks in the analog optical domain, using interferometers rather than transistors. This idea has genuine physical appeal but honest technical limitations that are worth understanding clearly.

> **The one-sentence version:** Light is replacing electrons for long-distance data movement in datacenters (done today), is being integrated directly into chip packages to eliminate electrical SerDes bottlenecks (co-packaged optics, commercially arriving now), and is being explored as a compute substrate for linear algebra — though analog optical compute faces fundamental challenges that keep it at research stage.

---

## Why Electrical Interconnect Is Running Out of Road

### The Energy Wall of Copper

Sending a bit from one chip to another over an electrical channel (a copper trace, PCIe lane, or backplane) requires:
1. **Driving the wire capacitance**: energy $E = \frac{1}{2}CV^2$ per bit, where $C$ is proportional to wire length
2. **Overcoming skin effect and dielectric loss** at high frequencies, which increases with frequency squared
3. **SerDes (serializer-deserializer) circuit overhead** for encoding, equalizing, and recovering the signal

Typical energy costs for electrical signaling:

| Channel | Energy per Bit | Practical Reach |
|---------|:--------------:|:---------------:|
| On-chip (Cu, within die) | ~0.05–0.2 pJ/bit | mm scale |
| Chip-to-chip (package, HBM microbumps) | ~0.5–1 pJ/bit | ~10–20 mm |
| PCIe Gen5 (off-package) | ~5–10 pJ/bit | ~30–50 cm |
| Long-reach 112G/224G SerDes over backplane or copper cable | ~5–10 pJ/bit (more with retimers) | ~1–2 m |
| NVLink (H100/B200 class, copper in the rack) | a few pJ/bit (estimate; not published) | within a rack |
| 800G pluggable optical module | ~18–20 pJ/bit (15–16 W) | 100 m–2 km |

As link rates climb (100G → 400G → 800G → 1.6T per lane), driving increasingly capacitive traces at increasingly high frequencies costs more energy. A 51.2 Tb/s switch ASIC (like NVIDIA Spectrum-X or Broadcom Tomahawk 5) needs ~35–50 TB/s of total I/O bandwidth; delivering that electrically over long reach consumes a substantial fraction of the chip's total power budget in SerDes alone.

The **reach** problem is equally limiting: at 112 Gbaud (the current leading edge for electrical SerDes), reliable reaches over copper are ~2–5 cm on a PCIe connector and ~30–50 cm on a differential trace before equalization becomes impractical.

### What That Means for Multi-GPU Clusters

A cluster of 10,000–100,000+ GPUs (NVIDIA's Eos had 10,752 H100s; Meta built two 24,576-GPU H100 clusters in 2024; xAI's Colossus started at 100,000 H100s the same year) requires every GPU to be able to exchange gradients and activations with many others. The interconnect from GPU to top-of-rack switch, to spine switch, and across to another GPU's rack spans 1–100 meters. Electrical signaling over that distance, at 800G+ per port, requires optics. The question is *where* the electrical-to-optical conversion happens — and that is the co-packaged optics story.

For deep background on multi-GPU topologies and parallelism strategies, see [Chapter 12](./12_buses_and_interconnects.md) and [Chapter 28](./28_multi_gpu_and_distributed_training.md).

---

## Silicon Photonics Basics

**Silicon photonics** is the technology of building optical devices from silicon (and closely related materials), using the same CMOS-compatible fabrication processes used for electronic chips. This matters enormously because it means optical components can be manufactured at the same fabs that make CPUs and GPUs — at volume, at low cost, and integrated alongside electronic circuits.

### The Four Building Blocks

An optical link needs four elements: a light source, a modulator, a waveguide, and a detector.

<div class="diagram">
<div class="diagram-title">Silicon Photonic Link</div>
<div class="flow-h">
  <div class="flow-node accent">Laser<br/><small>Light source<br/>(off-chip, III-V)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">Modulator<br/><small>Encodes data<br/>onto light</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">Waveguide<br/><small>Guides light<br/>in Si or SiN</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">Photodetector<br/><small>Converts light<br/>back to current</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange">TIA + CDR<br/><small>Transimpedance amp<br/>clock recovery</small></div>
</div>
</div>

**Waveguides** are the "wires" of photonics. A silicon waveguide is a narrow strip of Si (~450 × 220 nm cross-section, surrounded by SiO₂ cladding) that confines light via total internal reflection (Si refractive index ~3.5 vs SiO₂ ~1.45 — a large contrast). Propagation loss is ~2–3 dB/cm for Si waveguides; **silicon nitride (SiN)** waveguides offer ~0.1–0.5 dB/cm and better performance at shorter wavelengths.

**Modulators** convert an electrical data signal into a modulated optical signal by briefly changing the silicon's refractive index. The dominant silicon photonic modulator is the **ring resonator modulator**: a small ring (~10 µm diameter) placed adjacent to the waveguide. When the ring is on resonance, light couples from the waveguide into the ring and is absorbed; an applied reverse bias voltage shifts the resonance via the plasma dispersion effect, toggling the transmission. Ring modulators achieve 40–60+ Gbaud with <1 V drive swing and ~1–5 fJ/bit switching energy at small signal swing.

**Photodetectors** in silicon photonics use **germanium (Ge)** integrated onto silicon. Germanium absorbs light at telecom wavelengths (~1310 nm and ~1550 nm) where silicon is transparent. A Ge photodiode ~20 × 10 µm can achieve >40 GHz bandwidth; responsivity is ~1 A/W at 1310 nm. Germanium integration is well-established in silicon photonics processes.

**Lasers: the off-chip problem**. Silicon has an **indirect bandgap** — it is not an efficient light emitter. The carrier recombination that generates photons in direct-bandgap semiconductors (GaAs, InP, GaN) largely produces heat in silicon. As a result, silicon photonics lacks an efficient on-chip laser source. Current solutions:

- **External laser + coupling**: InP or GaAs DFB laser in a separate package, light coupled via edge coupler or grating coupler into the Si waveguide (~1–3 dB coupling loss)
- **Heterogeneous integration**: III-V gain material bonded to silicon wafer; processed together. Intel Integrated Photonics uses this approach for their optical transceivers.
- **Wafer bonding**: direct oxide bonding of InP films to Si, then processing the laser structure aligned to Si waveguides (IIIV-Lab, various academic groups)
- **Epitaxial growth** of GeSn or GaAs on Si: active research, not yet in production for lasers

The absence of a manufacturable on-chip silicon laser is the primary fabrication challenge for fully integrated silicon photonics. It is being addressed incrementally but is not solved for high-volume production as of 2026; CPO products use external laser modules that can be replaced from the front panel.

---

## Optical Interconnect: The Pluggable-to-Co-Packaged Transition

### Pluggable Optics (Today's Standard)

Today, optical connectivity in datacenters uses **pluggable optical transceivers** — modules (QSFP-DD, OSFP, 800G forms) that plug into a switch ASIC's front-panel ports. The module contains the laser, modulators, photodetectors, and all optics; it connects to the ASIC via a high-speed electrical interface (typically CAUI-4 or 112 Gbaud PAM4).

A 400G QSFP-DD module consumes ~8–12 W and costs $500–2000 (2024). An 800G OSFP consumes ~15–20 W. For a 51.2 Tb/s switch with 64 × 800G ports, the 64 optical modules alone consume ~1 kW and occupy the entire front panel.

The fundamental limitation: **the electrical reach from ASIC to module cage is ~5–10 cm**, requiring significant SerDes equalization power. As port rates rise from 800G to 1.6T and beyond, this SerDes-to-module path becomes the dominant power draw of the switch system.

### Co-Packaged Optics (CPO)

**Co-packaged optics** integrates the optical engines directly into the chip package, adjacent to (or on top of) the switch/compute ASIC. By eliminating the long electrical trace to the module cage, CPO reduces the SerDes power draw dramatically:

<div class="diagram">
<div class="diagram-title">Pluggable vs Co-Packaged Optics</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Pluggable (Traditional)</div>
    <ul>
      <li>ASIC → PCB trace (~8 cm) → cage → module → fiber</li>
      <li>SerDes power: ~5–8 pJ/bit</li>
      <li>Module power: ~15–20 W per 800G port</li>
      <li>Total I/O power for 51.2T switch: ~1–2 kW just for optics</li>
      <li>Advantages: field-replaceable, proven, interoperable</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Co-Packaged Optics (CPO)</div>
    <ul>
      <li>ASIC → ~2 mm package trace → optical engine → fiber</li>
      <li>SerDes power: ~1–2 pJ/bit (short reach)</li>
      <li>Optical engine: ~3–5 W per 800G port</li>
      <li>Total I/O power: ~35–50 % lower vs pluggable</li>
      <li>Advantages: energy, bandwidth density, signal integrity</li>
    </ul>
  </div>
</div>
</div>

Key numbers for CPO at 800G (approximate, 2024–2025):

| Metric | Pluggable optics | Co-packaged optics |
|--------|:-------------------:|:--------------:|
| Energy per bit (SerDes + optics) | ~18–20 pJ/bit | ~5–7 pJ/bit |
| Power per 1.6 Tb/s port (NVIDIA's figures) | ~30 W | ~9 W |
| Electrical path ASIC → optics | 10–20 cm of PCB plus connector | a few mm inside the package |
| Serviceability | Field-replaceable module | Optics fixed; external lasers replaceable |

**Who is building CPO:**

- **Broadcom** ships Bailly, a 51.2 Tb/s Tomahawk 5 switch with co-packaged optics, in volume, and its 102.4 Tb/s Tomahawk 6 "Davisson" CPO switch is sampling.
- **NVIDIA** announced Quantum-X Photonics (InfiniBand, 144 × 800G ports) and Spectrum-X Photonics (Ethernet, up to 102.4 Tb/s) switches in March 2025, built with TSMC's COUPE optical engines; InfiniBand parts ship from early 2026 and Ethernet in the second half. NVIDIA claims 3.5× better power efficiency than pluggables.
- **Intel** demonstrated CPO switches and optical compute interconnect chiplets, but sold its pluggable silicon-photonics transceiver business to Jabil in 2023.
- **Ayar Labs** builds **TeraPHY**, an optical I/O chiplet (with a UCIe interface since 2025) and an external multi-wavelength laser (SuperNova), designed to sit in the package beside a compute or networking die.
- **Lightmatter** builds **Passage**, a photonic interposer and interconnect (M1000, 2025) that stacks compute dies on an optical substrate.
- **Marvell**, **Ranovus** and others target 1.6T+ optical engines.

### Optical I/O for Compute Dies

The natural extension of CPO for switches is **optical I/O for GPU/accelerator packages**: rather than using NVLink copper cables or PCIe electrical lanes between GPUs, use optical waveguides. This would allow:
- Longer distances without signal degradation (meters instead of centimeters)
- Higher aggregate bandwidth per package edge
- Reduced power for inter-GPU collective operations (all-reduce, all-gather — the bottleneck in tensor-parallel training, [Chapter 28](./28_multi_gpu_and_distributed_training.md))

Ayar Labs' TeraPHY is specifically targeted at this use case: a companion optical chiplet that plugs into a GPU or TPU package and provides optical I/O. As of 2026, this is in customer sampling/pre-production rather than volume deployment. The technical challenge is that GPU-to-GPU bandwidth requirements (NVLink 4.0 at 900 GB/s total between two H100s) demand a very large number of optical lanes, each with its own modulator, detector, and laser tap — integrating this reliably at scale is non-trivial.

---

## Wavelength Division Multiplexing (WDM)

One of optical interconnect's major advantages over electrical is **WDM**: multiple independent data channels can simultaneously occupy the same waveguide or fiber, each at a different wavelength, completely non-interfering. This is physically impossible for electrical signals on a wire.

**CWDM (Coarse WDM)**: channels spaced ~20 nm apart in the O-band (~1310 nm center) or C-band (~1550 nm center). A 4-channel CWDM multiplexer at 1271/1291/1311/1331 nm is standard in 400G-LR4 transceivers.

**DWDM (Dense WDM)**: channels spaced ~0.8 nm (100 GHz) or 0.4 nm (50 GHz). Used in long-haul telecom; 80–160 channels on one fiber, each at 400G = 64 Tb/s on one strand.

For datacenter CPO, using 4–8 WDM channels per waveguide multiplies bandwidth without adding physical lanes:

$$\text{Total bandwidth} = N_{\text{channels}} \times B_{\text{per channel}} \times N_{\text{waveguides}}$$

At 8 channels × 100 Gbps × 16 waveguides = 12.8 Tbps bidirectional — achievable in a small package footprint.

---

## Optical / Photonic Computing: The Promise and the Honest Limits

### The Appeal: Linear Algebra With Light

A **Mach-Zehnder Interferometer (MZI)** is a Y-splitter that divides light, phase-shifts one branch, then recombines the two. The output intensity depends on the phase difference:

$$I_{\text{out}} = I_{\text{in}} \cos^2\!\left(\frac{\Delta\phi}{2}\right)$$

By cascading MZIs in a **mesh** (a 2D grid of N×N interferometers), you can implement any unitary matrix multiplication $\mathbf{y} = U\mathbf{x}$ in the optical domain — the light propagates through the mesh and performs the computation at the speed of light, with no switching energy proportional to data (passive propagation costs essentially nothing in a photonic waveguide). This is the basis for **photonic matrix accelerators**.

The energy appeal: a passive MZI mesh performs a complex multiply-accumulate using only the phase of light — no switching transistors, no charge movement. The theoretical lower bound is simply the photon energy, roughly $10^{-19}$ J/operation at 1 µW input power — orders of magnitude below the ~0.1–1 pJ/MAC of a digital tensor core.

**Lightmatter** (Boston) was the most prominent company commercializing this approach with its **Envise** photonic compute chip, a photonic matrix unit in a hybrid photonic-electronic package. Tellingly, its commercial emphasis has since shifted to **Passage**, photonic *interconnect*, which is where the near-term value in AI hardware is.

Academic groups (MIT, Stanford, Caltech, Oxford and Münster among them) have demonstrated photonic neural-network inference on small models.

### The Honest Limits

Optical compute faces several fundamental challenges that must be understood clearly:

**1. Nonlinearity is hard.** Neural networks are not purely linear. Activation functions (ReLU, GELU, SiLU) and normalization operations are *nonlinear*. Light in a linear waveguide medium does not interact with itself — two photons in a linear waveguide pass through each other unchanged. To implement nonlinearity, you must convert back to electronics, apply the nonlinear function electrically, then convert back to optics. These **O/E/O conversions** (optical→electrical→optical) consume energy and add latency, substantially eroding the theoretical efficiency advantage.

**2. The ADC/DAC wall.** Inputs to and outputs from an optical compute element must be converted between analog optical amplitude and digital values. A high-speed, high-resolution ADC (e.g., 8-bit at 25 Gbaud) consumes ~100–500 fJ/conversion. At matrix sizes relevant for LLM inference (e.g., 4096 × 4096 weight matrices), the number of I/O conversions dominates total energy if done naively. This is the **ADC/DAC wall** of analog compute — it is not unique to photonics, but photonics faces it acutely.

**3. Precision and noise.** Optical power is analog; maintaining 8-bit precision requires a signal-to-noise ratio of ~50 dB, which requires careful noise management. Phase stability in an MZI mesh depends on temperature, vibration, and fabrication imperfections. Large MZI meshes (N > 32) suffer from accumulated phase errors that require active calibration (itself consuming power).

**4. Programming speed.** Phase shifters in most silicon photonics processes use either **thermo-optic** (heating a waveguide changes its refractive index — ~10–100 µs response, low power) or **electro-optic (plasma dispersion)** (nanosecond response but larger loss). Reprogramming a 512×512 MZI matrix between inference calls takes milliseconds with thermal phase shifters — far too slow for weight updates during training, and borderline for batched inference depending on batch size.

**5. Memory bandwidth still exists.** Even an optical matrix multiply must load weights from DRAM or SRAM. The memory wall ([Chapter 11](./11_memory_wall_and_bandwidth.md)) applies whether the compute is photonic or electronic. For LLM inference (memory-bandwidth-bound, not compute-bound), photonic compute does not address the bottleneck.

<div class="diagram">
<div class="diagram-title">Where Optical Compute Helps (and Doesn't)</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Genuine Advantages</div>
    <ul>
      <li>Passive matrix-vector multiply: near-zero active energy</li>
      <li>Speed-of-light propagation through mesh: low latency</li>
      <li>High bandwidth density per waveguide (WDM)</li>
      <li>Potential for ultra-low-power inference on small fixed models</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Real Limitations</div>
    <ul>
      <li>Nonlinearity requires O/E/O: expensive in practice</li>
      <li>ADC/DAC overhead dominates at large matrix sizes</li>
      <li>Precision limited by analog SNR and thermal drift</li>
      <li>Programming speed (thermal phase shifters: µs–ms)</li>
      <li>Training: gradient computation is fundamentally electronic</li>
      <li>Memory bandwidth bottleneck unchanged</li>
    </ul>
  </div>
</div>
</div>

### What Is Actually Real Today

Being precise about the state of the technology in late 2026:

**Datacenter networking (optical interconnect)**: Mature, deployed at scale. 400G/800G pluggable optics are standard in every hyperscale datacenter. CPO switches are in volume from Broadcom and shipping from NVIDIA in 2026. This is real and consequential for AI clusters.

**Optical I/O for compute packages (Ayar Labs, etc.)**: Pre-production / early deployment for specific use cases (government, research clusters). Volume deployment on AI accelerators has not happened yet; NVLink stays copper inside the rack through at least Rubin Ultra (2027).

**Photonic compute chips (Lightmatter Envise, etc.)**: Research and early commercial stage. Lightmatter has demonstrated working photonic compute and has a commercial product strategy, but comparison benchmarks to H100-class ASICs at equivalent power have not been shown at scale. The technology is credible in principle, but the engineering challenges (nonlinearity, ADC/DAC, precision) mean it is not competitive with leading digital ASICs for training large models today.

**Optical compute for training**: Not feasible for the foreseeable future. Training requires high-precision weight updates, billions of nonlinear operations, and rapid weight reprogramming — all antithetical to today's photonic architectures.

**Optical compute for fixed-weight small-model inference**: The most plausible near-term niche — small, fixed networks (e.g., optical LIDAR processing, sensor fusion, scientific ML) where weights are rarely changed and the energy advantage of passive linear computation outweighs the ADC overhead.

---

## Silicon Photonics in AI Hardware: The Near-Term Reality

The place where photonics *has* a clear, near-term impact on ML infrastructure is in the **network fabric** connecting GPUs:

**Reducing all-reduce cost**: Collective communication (all-reduce for data-parallel gradient sync; all-to-all for expert-parallel MoE routing) is the primary communication bottleneck in large distributed training. Moving to CPO-based switches reduces energy per bit for this traffic by ~50–70 % and increases bandwidth density, enabling either larger clusters at the same power, or faster collective operations at the same cost. This directly maps to faster gradient sync and thus either larger batch sizes or higher throughput.

**Scale-out vs. scale-up**: Today's large clusters (e.g., 10,000+ H100s) require spine-leaf switch fabrics with ~3 hops of latency across the datacenter. High-bandwidth CPO switches with lower serialization latency reduce the penalty of these hops, reducing the gap between scale-up (NVLink, ~1.5 µs latency) and scale-out (InfiniBand/Ethernet, ~5–10 µs). For pipeline-parallel training schedules where micro-batch bubble size depends on hop latency, this matters.

```python
# Illustrative: energy savings from CPO in a training cluster all-reduce
# Not benchmarked against real hardware; order-of-magnitude illustration only

def allreduce_energy(
    n_gpus: int,
    param_bytes: int,
    hops: int = 3,
    pJ_per_bit_pluggable: float = 20.0,   # ~20 pJ/bit for 800G pluggable at switch
    pJ_per_bit_cpo: float = 7.0,           # ~5-7 pJ/bit for CPO equivalent
) -> dict:
    """Estimate energy consumed in all-reduce communication for one gradient step."""
    bits = param_bytes * 8
    # Ring all-reduce transfers 2*(N-1)/N * total_bits per GPU; approximate as 2x bits
    total_bits_switched = bits * 2 * hops * n_gpus

    energy_pluggable_J = total_bits_switched * pJ_per_bit_pluggable * 1e-12
    energy_cpo_J       = total_bits_switched * pJ_per_bit_cpo * 1e-12

    return {
        "pluggable_kJ": energy_pluggable_J / 1e3,
        "cpo_kJ":       energy_cpo_J / 1e3,
        "savings_pct":  (1 - pJ_per_bit_cpo / pJ_per_bit_pluggable) * 100,
    }

# 1024 GPUs, Llama-3-70B gradient (~140 GB of BF16 grads), 3 switch hops
result = allreduce_energy(
    n_gpus=1024,
    param_bytes=140 * 1024**3,
    hops=3,
)
print(f"Pluggable: {result['pluggable_kJ']:.1f} kJ per gradient step")
print(f"CPO:       {result['cpo_kJ']:.1f} kJ per gradient step")
print(f"Energy savings: {result['savings_pct']:.0f}%")
# Pluggable: 147.8 kJ per gradient step
# CPO:       51.7 kJ per gradient step
# Energy savings: 65%
# (Illustrative only — real figures depend on topology, link efficiency, RDMA stack)
```

> Put the result next to the compute: 1,024 GPUs at 700 W running a 10-second step use about 7,200 kJ, so the all-reduce's optics account for roughly 2% of the step's energy, and CPO saves about 1.3%. The bigger wins from CPO are power per port (which caps how many links a switch can carry), reliability and density. That is the honest version of the photonics case: optics matter because they set how big and how well-connected a cluster can be within a power budget, not because the bits themselves are expensive.

---

## Looking Further: Integrated Photonics and the Post-2030 Horizon

Several longer-range photonics directions are worth tracking:

**3D photonic integration**: stacking photonic dies on top of electronic dies (similar to HBM stacking on GPU — see [Chapter 35](./35_where_the_industry_is_headed.md)), enabling very short optical paths between compute and interconnect. MIT Lincoln Laboratory and DARPA programs have demonstrated this.

**Photonic-electronic neural networks**: hybrid architectures where the linear (matrix multiply) portion uses photonics and the nonlinear portion uses electronics, communicating via very short O/E/O conversions. The energy advantage is real only if the O/E/O overhead is smaller than the energy saved on the linear portion — this requires very high-throughput optical compute units and very efficient ADC/DAC, neither of which is mature.

**All-optical networks**: datacenter networks built entirely on optical switching (without electrical switch ASICs), using optical circuit switches (OCS) or optical packet switches. Google's Jupiter data-centre network and its TPU pods (since TPU v4) use MEMS-based optical circuit switches ("Apollo") to reconfigure topology; all-optical packet switching at high speed remains research-level.

**Mid-board optics and on-board optics**: intermediate stages between fully pluggable and fully co-packaged. Mid-board optics (MBO) moves the optical engine onto the PCB close to the ASIC, reducing electrical reach without the full packaging complexity of CPO. This is commercially available today from Inphi/Marvell, Intel, and others.

---

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>NVLink has stayed copper while the scale-out network goes optical. What would have to be true about laser reliability, cost per port and latency before you would trust optics for the per-layer all-reduce of tensor parallelism?</p>
</div>

## Why This Matters for Model Optimization

Interconnect is the scaling bottleneck you cannot optimize away with quantization or sparsity:

**All-reduce is a hard dependency**: In data-parallel training, the all-reduce operation must complete before the next forward pass begins (in synchronous training). No amount of compute efficiency reduces this latency — it is determined by bandwidth and topology. Optical interconnect addresses this directly.

**Tensor and expert parallelism**: tensor parallelism puts an all-reduce of activations in every layer's critical path, and MoE expert parallelism adds all-to-all traffic. Both stay inside the copper NVLink domain today; optics would let that domain grow beyond a rack.

**Pipeline parallelism bubble**: Pipeline bubbles (idle time waiting for micro-batches to move between pipeline stages) scale with inter-stage latency. CPO's lower serialization latency shrinks these bubbles slightly.

**Cluster power budgets**: A datacenter running 10,000 H100s at 700 W each = 7 MW of GPU power. Switches and optics typically add on the order of 5–15 %, a share that grows with cluster size and link speed. CPO's efficiency improvement translates directly to either more GPUs per facility or reduced cooling costs — both of which affect the economics of your training run.

For the broader trajectory of interconnect, packaging, and how they shape next-generation cluster design, see [Chapter 12](./12_buses_and_interconnects.md), [Chapter 35](./35_where_the_industry_is_headed.md), and [Appendix E](./appendix_e_packaging.md).

---

## Key Takeaways

- Electrical signaling consumes ~5–40 pJ/bit at datacenter distances and hits a reach wall at current bandwidths; optical links cost ~18–20 pJ/bit as pluggable modules and ~5–7 pJ/bit co-packaged, with reach no copper can match, which makes them the solution for rack-to-rack and, eventually, larger GPU-to-GPU domains.
- Silicon photonics builds waveguides, modulators (ring resonators), and photodetectors (Ge on Si) in CMOS-compatible processes; **lasers must be off-chip (III-V material)** because silicon's indirect bandgap prevents efficient light emission — this is the primary integration challenge.
- **Co-packaged optics (CPO)** eliminates the ~8 cm PCB trace from ASIC to pluggable module, reducing SerDes power by ~50–70 % and increasing bandwidth density by ~3×; Broadcom (in volume) and NVIDIA (shipping 2026) sell CPO switches; optical I/O on compute packages (Ayar Labs, Lightmatter) is still pre-volume.
- **WDM** (wavelength division multiplexing) multiplies the effective bandwidth of a single waveguide by carrying N independent channels at N wavelengths simultaneously — physically impossible for electrical signaling.
- **Photonic compute** (MZI meshes for matrix multiply) has genuine theoretical energy appeal but faces fundamental engineering barriers: nonlinear activations require O/E/O conversion, ADC/DAC overhead dominates at scale, analog precision is limited, and thermal phase shifters are too slow for rapid weight updates.
- The honest picture in late 2026: optical interconnect is mature and deployed; CPO is commercially arriving; optical compute for training is not feasible; optical compute for small fixed-weight inference is a narrow but plausible niche.
- For ML practitioners: the primary photonics impact on your training workloads is in the switch fabric — CPO-based switches reduce all-reduce energy and increase inter-GPU bandwidth, enabling either larger clusters at the same power budget or faster collective operations at the same cost.

## Check Yourself

<details class="check">
<summary>An 800G pluggable module draws ~16 W. What is its energy per bit, and what does CPO at ~9 W per 1.6T port imply?</summary>
<p>16 W ÷ 800 Gb/s = 20 pJ/bit. 9 W ÷ 1.6 Tb/s ≈ 5.6 pJ/bit, about 3.5× better, matching NVIDIA's claim.</p>
</details>

<details class="check">
<summary>Why do silicon-photonic links use an external laser?</summary>
<p>Silicon has an indirect bandgap and emits light very inefficiently, so the light source is a III-V (InP, GaAs) laser, either bonded onto the chip or, in today's CPO products, in a replaceable module.</p>
</details>

<details class="check">
<summary>How many channels does a WDM link with 8 wavelengths × 16 waveguides × 100 Gb/s carry, and what is the total?</summary>
<p>128 channels, 12.8 Tb/s.</p>
</details>

<details class="check">
<summary>Why is photonic matrix multiplication not used for training?</summary>
<p>Training needs high precision, nonlinear functions, frequent weight updates and large memory traffic. Photonic meshes give ~4–8 effective bits, need O/E/O conversion for nonlinearities and ADC/DACs that dominate the energy, and reprogram slowly.</p>
</details>

<details class="check">
<summary>In the chapter's example, the all-reduce optics used ~148 kJ per step on 1,024 GPUs. What fraction of the step's energy is that if the step takes 10 s at 700 W per GPU?</summary>
<p>GPU energy ≈ 1,024 × 700 W × 10 s ≈ 7.2 MJ, so ~2%. The network matters more for cluster size and bandwidth than for energy.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Silicon photonics (review) | Shekhar et al., Nature Communications 2024 | [doi.org/10.1038/s41467-024-44750-0](https://doi.org/10.1038/s41467-024-44750-0) | Building blocks: waveguides, modulators, Ge detectors |
| Deep learning with coherent nanophotonic circuits | Shen et al., Nature Photonics 2017 | [doi.org/10.1038/nphoton.2017.93](https://doi.org/10.1038/nphoton.2017.93) | MZI-mesh optical matrix multiply |
| NVIDIA Spectrum-X Photonics and Quantum-X Photonics | NVIDIA, Mar 2025 | [nvidianews.nvidia.com/news/nvidia-spectrum-x-…](https://nvidianews.nvidia.com/news/nvidia-spectrum-x-co-packaged-optics-networking-switches-ai-factories) | CPO switches, 3.5× efficiency claim |
| Broadcom co-packaged optics | Broadcom | [broadcom.com/info/optics/cpo](https://www.broadcom.com/info/optics/cpo) | Bailly and Tomahawk 6 CPO |
| Ayar Labs TeraPHY | Ayar Labs | [ayarlabs.com](https://ayarlabs.com/) | Optical I/O chiplets |
| Lightmatter Passage | Lightmatter | [lightmatter.co](https://lightmatter.co/) | Photonic interposer and interconnect |
| Jupiter Evolving: Transforming Google's Datacenter Network via Optical Circuit Switches | Poutievski et al., SIGCOMM 2022 | [doi.org/10.1145/3544216.3544265](https://doi.org/10.1145/3544216.3544265) | Optical circuit switching at scale |
| TPU v4: An Optically Reconfigurable Supercomputer | Jouppi et al., ISCA 2023 | [arxiv.org/abs/2304.01433](https://arxiv.org/abs/2304.01433) | OCS in TPU pods |

---

**Next:** [Appendix D — Power & Thermals](./appendix_d_power_and_thermals.md) — We trace where the watts go, from the transistor switch to the data-centre cooling plant, and why power is now the binding constraint.

[← Back to Table of Contents](./README.md)
