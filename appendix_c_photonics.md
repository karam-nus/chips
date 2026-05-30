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
| 100G Ethernet / backplane copper | ~20–40 pJ/bit | ~1–2 m |
| NVLink 4.0 (NVIDIA H100) | ~3–5 pJ/bit (integrated) | module-to-module |

As link rates climb (100G → 400G → 800G → 1.6T per lane), driving increasingly capacitive traces at increasingly high frequencies costs more energy. A 51.2 Tb/s switch ASIC (like NVIDIA Spectrum-X or Broadcom Tomahawk 5) needs ~35–50 TB/s of total I/O bandwidth; delivering that electrically over long reach consumes a substantial fraction of the chip's total power budget in SerDes alone.

The **reach** problem is equally limiting: at 112 Gbaud (the current leading edge for electrical SerDes), reliable reaches over copper are ~2–5 cm on a PCIe connector and ~30–50 cm on a differential trace before equalization becomes impractical.

### What That Means for Multi-GPU Clusters

A cluster of 10,000 H100s (roughly what Meta's AI Research SuperCluster and NVIDIA's Eos system look like) requires every GPU to be able to exchange gradients and activations with many others. The interconnect from GPU to top-of-rack switch, to spine switch, and across to another GPU's rack spans 1–100 meters. Electrical signaling over that distance, at 800G+ per port, requires optics. The question is *where* the electrical-to-optical conversion happens — and that is the co-packaged optics story.

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

The absence of a manufacturable on-chip silicon laser is the primary fabrication challenge for fully integrated silicon photonics. It is being addressed incrementally but is not solved for high-volume production as of 2025.

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

| Metric | Pluggable OSFP 800G | CPO 800G Engine |
|--------|:-------------------:|:--------------:|
| Energy per bit (SerDes + optics) | ~18–25 pJ/bit | ~6–10 pJ/bit |
| Bandwidth density (Tb/s per mm² of package edge) | ~5–8 Tb/s/mm | ~15–25 Tb/s/mm |
| Latency (electrical to optical) | ~100–200 ps | ~20–50 ps |
| Module replaceability | Field-replaceable | Not field-replaceable (fixed) |

**Who is building CPO:**

- **Intel** demonstrated CPO switch integration at OFC 2021 and has since productized optical engines; Intel's 12.8T CPO switch uses Intel Silicon Photonics engines bonded to the switch die.
- **Broadcom** announced CPO for Tomahawk 5+ class switches (51.2T and beyond), with CPO optical engines from partners.
- **NVIDIA** Spectrum-4 and future Spectrum-X switches are designed with CPO paths; NVIDIA acquired Mellanox (networking) and has invested in photonics integration.
- **Ayar Labs** is the leading startup in this space: their **TeraPHY** optical chiplet provides ~2 Tb/s bidirectional bandwidth per chiplet using a monolithically integrated silicon photonics process, designed to sit in the package alongside a compute or networking die. Ayar has partnership agreements with Intel (using Intel's co-investment in their technology) and has shown CPO demos with DARPA and hyperscaler customers.
- **Marvell** (COLORZ II, LightPath) and **Ranovus** are also in the CPO space, targeting 1.6T+ switches.

### Optical I/O for Compute Dies

The natural extension of CPO for switches is **optical I/O for GPU/accelerator packages**: rather than using NVLink copper cables or PCIe electrical lanes between GPUs, use optical waveguides. This would allow:
- Longer distances without signal degradation (meters instead of centimeters)
- Higher aggregate bandwidth per package edge
- Reduced power for inter-GPU collective operations (all-reduce, all-gather — the bottleneck in tensor-parallel training, [Chapter 28](./28_multi_gpu_and_distributed_training.md))

Ayar Labs' TeraPHY is specifically targeted at this use case: a companion optical chiplet that plugs into a GPU or TPU package and provides optical I/O. As of 2025, this is in customer sampling/pre-production rather than volume deployment. The technical challenge is that GPU-to-GPU bandwidth requirements (NVLink 4.0 at 900 GB/s total between two H100s) demand a very large number of optical lanes, each with its own modulator, detector, and laser tap — integrating this reliably at scale is non-trivial.

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

**Lightmatter** (Cambridge, MA) is the most prominent company commercializing this approach. Their **Passage** silicon photonics chip is a 4×4 Tb/s photonic interconnect fabric, but their core technology thesis is the **Envise** photonic compute chip: a photonic matrix unit embedded in a hybrid photonic-electronic chip.

**Luminous Computing**, **QuiX Quantum**, and various academic groups (MIT, Stanford, Caltech) have demonstrated photonic neural network inference on small models.

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

Being precise about the state of the technology in 2025:

**Datacenter networking (optical interconnect)**: Mature, deployed at scale. 400G/800G pluggable optics are standard in every hyperscale datacenter. CPO is transitioning from sampling to volume production (Broadcom, Intel, Marvell targeting 2025–2027 volume for 51.2T+ switches). This is real and consequential for AI clusters.

**Optical I/O for compute packages (Ayar Labs, etc.)**: Pre-production / early deployment for specific use cases (government, research clusters). Volume commercial deployment for AI accelerators is 2–3 years out as of 2025.

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
    pJ_per_bit_cpo: float = 7.0,           # ~7 pJ/bit for CPO equivalent
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

# 1024 GPUs, Llama-3-70B gradient (~140 GB FP32 grads), 3 switch hops
result = allreduce_energy(
    n_gpus=1024,
    param_bytes=140 * 1024**3,
    hops=3,
)
print(f"Pluggable: {result['pluggable_kJ']:.1f} kJ per gradient step")
print(f"CPO:       {result['cpo_kJ']:.1f} kJ per gradient step")
print(f"Energy savings: {result['savings_pct']:.0f}%")
# Pluggable: ~9,400 kJ per gradient step
# CPO:       ~3,300 kJ per gradient step
# Energy savings: ~65%
# (Illustrative only — real figures depend on topology, link efficiency, RDMA stack)
```

> This illustrative calculation shows the scale of energy at stake: all-reduce communication at hyperscale consumes meaningful fractions of total cluster energy. CPO's ~50–70 % energy reduction in the switching fabric is consequential — not just for cost, but for the power-delivery and cooling infrastructure of a datacenter.

---

## Looking Further: Integrated Photonics and the Post-2030 Horizon

Several longer-range photonics directions are worth tracking:

**3D photonic integration**: stacking photonic dies on top of electronic dies (similar to HBM stacking on GPU — see [Chapter 35](./35_where_the_industry_is_headed.md)), enabling very short optical paths between compute and interconnect. MIT Lincoln Laboratory and DARPA programs have demonstrated this.

**Photonic-electronic neural networks**: hybrid architectures where the linear (matrix multiply) portion uses photonics and the nonlinear portion uses electronics, communicating via very short O/E/O conversions. The energy advantage is real only if the O/E/O overhead is smaller than the energy saved on the linear portion — this requires very high-throughput optical compute units and very efficient ADC/DAC, neither of which is mature.

**All-optical networks**: datacenter networks built entirely on optical switching (without electrical switch ASICs), using optical circuit switches (OCS) or optical packet switches. Google's Jupiter network has used some optical circuit switching; all-optical packet switching at high speed remains research-level.

**Mid-board optics and on-board optics**: intermediate stages between fully pluggable and fully co-packaged. Mid-board optics (MBO) moves the optical engine onto the PCB close to the ASIC, reducing electrical reach without the full packaging complexity of CPO. This is commercially available today from Inphi/Marvell, Intel, and others.

---

## Why This Matters for Model Optimization

Interconnect is the scaling bottleneck you cannot optimize away with quantization or sparsity:

**All-reduce is a hard dependency**: In data-parallel training, the all-reduce operation must complete before the next forward pass begins (in synchronous training). No amount of compute efficiency reduces this latency — it is determined by bandwidth and topology. Optical interconnect addresses this directly.

**Tensor parallelism activations**: In tensor-parallel Transformers (splitting attention heads across GPUs), all-to-all communication of activations between layers is in the critical path. Higher optical switch bandwidth density allows larger tensor-parallel groups with acceptable communication overhead.

**Pipeline parallelism bubble**: Pipeline bubbles (idle time waiting for micro-batches to move between pipeline stages) scale with inter-stage latency. CPO's lower serialization latency shrinks these bubbles slightly.

**Cluster power budgets**: A datacenter running 10,000 H100s at 700 W each = 7 MW of compute power. The networking infrastructure adds another 15–25 %. CPO's efficiency improvement translates directly to either more GPUs per facility or reduced cooling costs — both of which affect the economics of your training run.

For the broader trajectory of interconnect, packaging, and how they shape next-generation cluster design, see [Chapter 12](./12_buses_and_interconnects.md), [Chapter 35](./35_where_the_industry_is_headed.md), and [Appendix E](./appendix_e_packaging.md).

---

## Key Takeaways

- Electrical signaling consumes ~5–40 pJ/bit at datacenter distances and hits a reach wall at current bandwidths; optical interconnect at ~1–3 pJ/bit over unlimited-distance fiber is the solution for GPU-to-GPU and rack-to-rack communication.
- Silicon photonics builds waveguides, modulators (ring resonators), and photodetectors (Ge on Si) in CMOS-compatible processes; **lasers must be off-chip (III-V material)** because silicon's indirect bandgap prevents efficient light emission — this is the primary integration challenge.
- **Co-packaged optics (CPO)** eliminates the ~8 cm PCB trace from ASIC to pluggable module, reducing SerDes power by ~50–70 % and increasing bandwidth density by ~3×; Broadcom, Intel, NVIDIA, and Ayar Labs are in volume production or late pre-production for CPO switches as of 2025.
- **WDM** (wavelength division multiplexing) multiplies the effective bandwidth of a single waveguide by carrying N independent channels at N wavelengths simultaneously — physically impossible for electrical signaling.
- **Photonic compute** (MZI meshes for matrix multiply) has genuine theoretical energy appeal but faces fundamental engineering barriers: nonlinear activations require O/E/O conversion, ADC/DAC overhead dominates at scale, analog precision is limited, and thermal phase shifters are too slow for rapid weight updates.
- The honest picture in 2025: optical interconnect is mature and deployed; CPO is commercially arriving; optical compute for training is not feasible; optical compute for small fixed-weight inference is a narrow but plausible niche.
- For ML practitioners: the primary photonics impact on your training workloads is in the switch fabric — CPO-based switches reduce all-reduce energy and increase inter-GPU bandwidth, enabling either larger clusters at the same power budget or faster collective operations at the same cost.

---

*Next: [Appendix D — Power & Thermals](./appendix_d_power_and_thermals.md), where we trace where the watts go — from the transistor switch to the datacenter cooling infrastructure — and understand why power is the binding constraint on every AI accelerator.*

[← Back to Table of Contents](./README.md)
