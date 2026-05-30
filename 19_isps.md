---
title: "Chapter 19 — ISPs (Image Signal Processors)"
---

[← Back to Table of Contents](./README.md)

# Chapter 19 — ISPs (Image Signal Processors)

Every image your phone's camera produces started as a stream of mostly-green raw numbers — chaotic, noisy, and colorless. Within 30–60 milliseconds those raw numbers became a vivid, sharp, color-correct photograph. Something did that work at real time, consuming under 200 mW, in a block about as wide as your thumbnail. That block is the **Image Signal Processor** (ISP).

The ISP is the first piece of hardware your vision model's input ever touches. Long before a tensor reaches an NPU, the ISP has already decided what signal-to-noise ratio you get, what colors look like, how fine textures are rendered, and how much dynamic range is preserved. If you deploy vision AI on a camera-equipped device — and most edge AI is exactly that — understanding the ISP is not academic. It shapes your model's inputs, determines what preprocessing you can offload to silicon, and increasingly blurs the line between signal processing and neural inference.

> **The one-sentence version:** An ISP is a dedicated fixed-function (and increasingly programmable) pipeline that converts raw camera sensor data into a processed image or video frame at real-time throughput, handling tasks — demosaicing, denoising, color correction, tone mapping — that would be far too expensive to run on a CPU or GPU within the power budget of a phone.

---

## The Raw Sensor: What the Camera Actually Gives You

A camera image sensor is a 2D array of **photodiodes** — tiny capacitors that accumulate charge proportional to how many photons hit them during the exposure. Each photodiode measures one thing: **light intensity**. Not color. A color image sensor gets color by placing a **Color Filter Array (CFA)** over the photodiodes, most commonly the **Bayer filter**, invented by Kodak engineer Bryce Bayer in 1976.

The Bayer pattern tiles 2×2 blocks, each with one Red, two Green, and one Blue filter:

```text
R  G  R  G  R  G
G  B  G  B  G  B
R  G  R  G  R  G
G  B  G  B  G  B
```

Why 2 greens per block? Human vision is most sensitive to green light; doubling green pixels reduces luminance noise where it hurts most.

The raw output — called a **Bayer RAW image** — is typically stored as:

| Property | Typical value |
|---|---|
| Pixel depth | 10–14 bits per pixel (e.g., RAW12, RAW14) |
| Resolution | 12–200 MP (phone); 50 MP typical for flagship |
| Throughput at 4K@60 | ~2.1 Gpixels/s of raw data |
| Bandwidth at 12 bpp, 4K@60 | ≈3.2 GB/s from sensor to ISP |
| Color information per pixel | Only 1 of R, G, B — the other two are *missing* |

The raw file literally looks dark and greenish if you display it directly — the missing color channels and non-linear sensor response make it perceptually wrong. The ISP's job is to fix all of that, in order, at wire speed.

---

## The Camera Pipeline: Stage by Stage

The ISP is a **pipeline of fixed-function stages**, each chained to the next. Pixels flow in from the sensor and processed pixels flow out to a display buffer or video encoder. The full pipeline for a modern ISP looks like this:

<div class="diagram">
<div class="diagram-title">The ISP Camera Pipeline (Bayer RAW → Processed Image)</div>
<div class="flow">
  <div class="flow-node accent wide">1. Bayer RAW Input<br/><small>10–14 bpp · e.g. 12.6 GB/s at 200 MP@5fps</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">2. Black Level Correction<br/><small>subtract dark current offset per channel</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">3. Lens Shading Correction<br/><small>vignetting rolloff compensated per pixel</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">4. Demosaicing<br/><small>interpolate missing R/G/B per pixel → full 3-channel</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">5. White Balance<br/><small>per-channel gain multiply → neutral gray</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">6. Noise Reduction (NR)<br/><small>spatial + temporal filtering; most power-hungry stage</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node teal wide">7. Color Correction Matrix (CCM)<br/><small>3×3 matrix mul per pixel: sensor color space → sRGB</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">8. Tone Mapping / Gamma<br/><small>HDR → SDR; gamma LUT; local tone map for HDR display</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">9. Sharpening / Edge Enhancement<br/><small>unsharp mask or edge-aware filter</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">10. Encode / Output<br/><small>H.265/AV1 encoder; JPEG; NV12 YUV to NPU/GPU</small></div>
</div>
</div>

Let's walk through each stage precisely.

### Stage 1 — Bayer RAW Input

The sensor streams pixels over a high-speed MIPI **CSI-2** (Camera Serial Interface) bus, typically at 2–4 Gbps per lane, with 4–8 lanes. At 200 MP and 5 fps that is $200 \times 10^6 \times 14\ \text{bits} \times 5 = 14\ \text{Gbps}$ — requiring 4 lanes at 4 Gbps each. The ISP's front-end receives and buffers this stream into on-chip SRAM (line buffers, usually 4–16 rows deep), feeding the pipeline without full-frame storage.

### Stage 2 — Black Level Correction (BLC)

Even with no light, photodiodes accumulate a small **dark current** that varies with temperature and per-channel. BLC subtracts a calibrated constant (the "black level," e.g., 64 DN on a 10-bit sensor) from every pixel:

$$p'_{ch} = \text{clamp}(p_{ch} - \text{BL}_{ch},\ 0,\ 2^{\text{bpp}}-1)$$

This is a trivial scalar subtract — one of the cheapest stages.

### Stage 3 — Lens Shading Correction (LSC)

Lenses are brighter at center than edges (**vignetting**). LSC compensates by multiplying each pixel by a 2D gain map stored in ROM (one map per color channel, often at 64×64 or 128×128 resolution, bilinearly interpolated). Output is still Bayer, full bit-depth.

### Stage 4 — Demosaicing

This is the most algorithmically interesting stage. The Bayer pattern gives only one color per pixel; demosaicing **reconstructs all three**. The simplest approach: **bilinear interpolation** — average the nearest neighbors of the same color.

```python
import numpy as np

def bilinear_demosaic(bayer: np.ndarray) -> np.ndarray:
    """
    bayer: (H, W) uint16 Bayer RGGB array
    returns: (H, W, 3) float32 RGB
    
    This is a sketch — real ISPs use adaptive methods
    (AHD, LMMSE, or learned) to avoid color fringing at edges.
    """
    H, W = bayer.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    # Extract channels from RGGB Bayer layout
    R  = bayer[0::2, 0::2].astype(np.float32)   # even rows, even cols
    Gr = bayer[0::2, 1::2].astype(np.float32)   # even rows, odd cols
    Gb = bayer[1::2, 0::2].astype(np.float32)   # odd rows, even cols
    B  = bayer[1::2, 1::2].astype(np.float32)   # odd rows, odd cols

    # Bilinear upsample each channel to full resolution
    from scipy.ndimage import zoom
    rgb[:, :, 0] = zoom(R,  2, order=1)  # Red
    rgb[:, :, 1] = zoom((Gr + Gb) / 2, 2, order=1)  # Green (average Gr,Gb)
    rgb[:, :, 2] = zoom(B,  2, order=1)  # Blue
    return rgb
```

> **Why bilinear is not enough.** Bilinear demosaicing causes **color fringing** (zipper artifacts) at high-frequency edges. Real ISPs use **adaptive homogeneity-directed (AHD)** or **LMMSE** demosaicing, which detect edges and interpolate along them. Modern ISPs increasingly use **learned demosaicing** (a small CNN), run on the ISP's programmable SIMD core or on the NPU.

### Stage 5 — White Balance (AWB)

The color of "white" light changes dramatically — from 2700 K tungsten (~2× more red than blue) to 7000 K overcast sky (~2× more blue than red). White balance corrects by **multiplying each channel by a per-channel gain**:

$$[R', G', B'] = [g_R \cdot R,\ g_G \cdot G,\ g_B \cdot B]$$

where $g_R, g_G, g_B$ are chosen so that a gray reference patch comes out equal in all three channels. The AWB *algorithm* (finding the right gains from the scene) runs on a small microcontroller embedded in the ISP; the *application* of the gains is a trivial per-pixel multiply. This is essentially a diagonal 3×3 color matrix with zeros off-diagonal.

### Stage 6 — Noise Reduction (NR)

The most computationally demanding stage. Photon shot noise (Poisson) and read noise degrade every image; at low light they dominate. ISP NR uses:

- **Spatial NR:** a bilateral filter or guided filter — a weighted average of nearby pixels, with weights that decay with both spatial distance *and* color difference, so edges are not blurred. Kernel size is typically 5×5 to 9×9.
- **Temporal NR (TNR):** align the current frame to the previous frame (motion estimation), then blend them. This halves noise variance per frame blended ($\sigma^2 \to \sigma^2/n$), at the cost of requiring a full-resolution motion-compensated reference frame in DDR (≈12–24 MB for 12MP at 16 bpp).

**Power consumption note:** NR on a modern phone ISP (e.g., Qualcomm Spectra, Apple) accounts for ~40–60% of the ISP's power budget — typically 50–150 mW for the NR block alone at 4K@30.

### Stage 7 — Color Correction Matrix (CCM)

The sensor's spectral sensitivity is not sRGB — different sensors have different dye responses. The CCM is a **3×3 matrix multiply** applied per pixel that maps from sensor RGB to standard sRGB (or Display P3 on modern phones):

$$\begin{pmatrix} R_{sRGB} \\ G_{sRGB} \\ B_{sRGB} \end{pmatrix} = M_{3\times3} \begin{pmatrix} R_{sensor} \\ G_{sensor} \\ B_{sensor} \end{pmatrix}$$

For a 12 MP image at 4K@60, this is $12 \times 10^6 \times 60 \times 9 = 6.48 \times 10^9$ multiply-adds per second — exactly why a dedicated multiply-accumulate datapath handles it in hardware.

### Stage 8 — Tone Mapping

Sensors can capture 12–14 stops of dynamic range; a standard display shows 8 stops. **Tone mapping** compresses the range while preserving perceptual detail. Simple global tone mapping applies a **gamma curve** (lookup table): $V_{out} = V_{in}^{1/2.2}$. Local tone mapping (for HDR output) uses spatially-varying compression — computationally heavier, typically done with a Laplacian pyramid decompose/recompose loop.

For HDR video output (HDR10, Dolby Vision), the ISP applies **OETF/EOTF** transforms (PQ or HLG curves) instead of gamma, and metadata is written to the bitstream.

### Stage 9 — Sharpening / Edge Enhancement

A final **unsharp mask** (subtract a blurred copy, add back a scaled version of the difference) restores apparent sharpness lost to optical diffraction and the anti-aliasing filter:

$$O = I + \lambda (I - \text{Gaussian}(I, \sigma))$$

Typical $\lambda = 0.5$–$1.5$, $\sigma = 1.0$–$2.0$. The ISP's implementation is a separable convolution — two 1D passes — keeping it cheap.

### Stage 10 — Output / Encode

The processed image exits the ISP in one of several formats:

| Output path | Format | Typical consumer |
|---|---|---|
| Video record | H.265/AV1 bitstream | Storage / streaming |
| Preview / viewfinder | NV12 YUV 4:2:0 | Display compositor |
| ML inference | NV12 YUV or RGB888 | NPU / GPU vision model |
| Still capture | JPEG / HEIF | Gallery |
| RAW passthrough | RAW10/RAW12 | Pro camera apps, ML training data |

The **NV12** format is the ISP↔NPU handshake in virtually every mobile SoC: Y plane (full resolution) followed by interleaved UV plane (half resolution). One 4K NV12 frame = $3840 \times 2160 \times 1.5 = 12.4$ MB.

---

## Why a Dedicated Block? Throughput and Power

Why can't a CPU or GPU do this? Consider the throughput requirements:

| Target | Raw pixels/second | Notes |
|---|---|---|
| 4K@30 video | 248 Mpix/s | Continuous, real-time |
| 4K@60 video | 497 Mpix/s | Continuous |
| 4K@120 (gaming phone) | 994 Mpix/s | Continuous |
| 200 MP still, 5 fps burst | 1,000 Mpix/s | Burst mode |
| 1B pix/s "dual camera" | 1,000 Mpix/s | Front + rear simultaneously |

A CPU core at 3 GHz executing one instruction per clock would need to process **one pixel every 3 ns** to reach 333 Mpix/s — but each pixel requires ~50 operations across all stages. That's 16,700 MIPS, or roughly 20 A-class cores working flat-out on nothing else. Power: catastrophic. Latency: 100–200 ms instead of the sub-10 ms the pipeline achieves.

The ISP achieves this by being **deeply pipelined and spatially parallel**:
- Every stage runs simultaneously on different rows of pixels
- Each stage is a **fixed-function datapath** — no instruction fetch, no branch prediction overhead
- The pipeline is clocked at 500–800 MHz but is as wide as 32–128 pixels per clock
- Total ISP power: 100–300 mW at 4K@30 on a 5 nm SoC

A GPU doing the same tasks would need 5–20× more power for the same throughput.

---

## Fixed-Function vs Programmable ISPs

Traditional ISPs are purely fixed-function — the algorithm is baked into gates and cannot change post-manufacture. This is maximally efficient but inflexible.

Modern ISPs add a **programmable SIMD core** (sometimes called an "ISP processor" or "scalar + vector" unit) that lets vendors update algorithms via firmware:

<div class="diagram">
<div class="diagram-title">Fixed-Function vs Programmable ISP Architecture</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Fixed-Function ISP</div>
    <ul>
      <li>Every stage is hardwired logic</li>
      <li>Maximum throughput, minimum power</li>
      <li>Algorithm frozen at tape-out</li>
      <li>Example: simple sensor ISPs, surveillance cameras</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Programmable ISP</div>
    <ul>
      <li>Fixed-function "fast path" for known stages</li>
      <li>SIMD co-processor for custom / updated algorithms</li>
      <li>Firmware-updatable (e.g., new denoise, new demosaic)</li>
      <li>Examples: Qualcomm Spectra, Apple ISP, Sony ISP, Google Tensor ISP</li>
    </ul>
  </div>
</div>
</div>

Qualcomm's **Spectra ISP** (Snapdragon 8 Gen series) uses a VLIW-SIMD core called the "Hexagon Vector eXtensions" front-end that can run custom convolution kernels in the ISP pipeline — letting Qualcomm ship firmware updates that improve NR without hardware changes. The Spectra 680 (Snapdragon 8 Gen 2) supports up to **3.2 Gpix/s** across three cameras simultaneously.

Apple's ISP (M3/A17) is tightly coupled to the Neural Engine and can pass intermediate pipeline outputs directly to the ANE without going to DRAM — **zero-copy** from camera to neural inference, a critical power saving.

---

## Where the ISP Lives in an SoC

<div class="diagram">
<div class="diagram-title">ISP in a Mobile SoC (Simplified)</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">📷</div>
    <div class="card-title">Camera Sensor(s)</div>
    <div class="card-desc">MIPI CSI-2 bus at 2–4 Gbps/lane; up to 4 lanes per camera</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🖼️</div>
    <div class="card-title">ISP Block</div>
    <div class="card-desc">BLC → demosaic → WB → NR → CCM → tone map → sharpen. 100–300 mW at 4K@30</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🧠</div>
    <div class="card-title">NPU / GPU</div>
    <div class="card-desc">Receives NV12/RGB from ISP via DMA or direct fabric; runs vision models</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">🎞️</div>
    <div class="card-title">Video Encoder</div>
    <div class="card-desc">H.265/AV1 HW encoder; operates in parallel with ISP on same frame</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">🗃️</div>
    <div class="card-title">DDR / LPDDR5</div>
    <div class="card-desc">Shared memory for frame buffers; ISP is one of the highest-BW consumers in the SoC</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-icon">⚙️</div>
    <div class="card-title">On-chip Fabric</div>
    <div class="card-desc">AXI/NoC interconnect; ISP, NPU, encoder DMA controllers share bandwidth</div>
  </div>
</div>
</div>

The ISP is a **DMA master**: it pulls raw frames from a MIPI input buffer and pushes processed frames to an output buffer in DDR, all without CPU involvement. Its memory bandwidth at 4K@30 is roughly:

$$\text{ISP BW} = 2 \times 3840 \times 2160 \times 1.5 \times 30 = 1.12\ \text{GB/s (read + write)}$$

At 4K@60 with temporal NR (requiring a reference frame): ~3.5–4 GB/s — a significant fraction of LPDDR5's 77 GB/s total.

---

## HDR and Multi-Frame Processing

Single-exposure images clip highlights (bright areas saturate to white) or lose shadows (dark areas drown in noise). **HDR** techniques blend multiple exposures.

### Exposure Bracketing and Stacking

The ISP can capture 3–5 frames at different exposures within a single "shutter press" (burst HDR) or use sensor-level **dual-gain** (two exposures interleaved per frame in the same MIPI packet). The ISP then:

1. **Aligns** frames (optical flow, typically 4×4 block matching)
2. **Merges** using a spatially-varying weight map (more weight from the exposure where local pixels are not blown/clipped)
3. **Tone-maps** the merged HDR image to the display's range

NVIDIA's **Multi-Frame NR** and Apple's **Smart HDR** (combining up to 9 frames) are both implemented as ISP + Neural Engine pipelines, with the ISP handling motion compensation and the NPU doing learned merge/tone-map.

### Computational Photography and the ISP↔NPU Handoff

"Computational photography" refers to **algorithms that treat photography as a signal-processing/ML problem** rather than a pure optics problem. Google's **Night Sight** (Pixel phones), Apple's **Photonic Engine**, and Qualcomm's **AI-ISP** all route ISP intermediate outputs through an NPU:

<div class="diagram">
<div class="diagram-title">ISP ↔ NPU Cooperation in Computational Photography</div>
<div class="flow-h">
  <div class="flow-node accent">Bayer RAW frames<br/><small>(burst of 8–15)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">ISP fast path<br/><small>BLC, WB, align</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">NPU<br/><small>learned merge, denoise, HDR tone map</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">ISP post path<br/><small>CCM, sharpen, encode</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange">Output<br/><small>JPEG / HEIF</small></div>
</div>
</div>

Google's Tensor G3 ISP ("Tensor ISP") has a dedicated "**Semantic Segmentation ISP**" feature — the ISP can request a per-pixel class label from a 17-class semantic segmentation model running on the NPU in real time, and use those labels to apply different NR/sharpening parameters per region (e.g., extra NR on sky, less on faces to preserve skin texture). This is the ISP treating NPU output as a control signal.

---

## RAW vs Processed Input for Vision Models

One of the most consequential choices for on-device vision is: **what format should frames be in when they reach the model?**

| Input to Model | Pros | Cons |
|---|---|---|
| ISP-processed RGB | Standard training data; ImageNet-pretrained models work | ISP choices (tone map, NR, sharpening) baked in; no control |
| ISP-processed YUV (NV12) | Common GPU/NPU-native format; saves a color-space convert | Slightly different color representation; need to handle Y/UV planes |
| RAW Bayer | Maximum information; no irreversible compression | Models must be trained on RAW; much larger (10–14 bpp vs 8 bpp); no pretrained ecosystem |
| RAW-RGB (after demosaic only) | More controllable than full ISP; pretrained-compatible | Still need custom training pipeline |

For most deployments, **ISP-processed NV12 → model** is the right answer: it leverages the ISP's efficient NR and color processing, and the standard pretrained model zoo. The key exception is when you **control the training data collection** — then training directly on RAW or partial-ISP output can recover detail the ISP would otherwise discard.

```python
import torch
import torchvision.transforms.functional as TF

def load_nv12_as_rgb(nv12_buffer: bytes, H: int = 2160, W: int = 3840) -> torch.Tensor:
    """
    Convert an NV12 buffer (from ISP) to RGB tensor for inference.
    NV12: Y plane (H×W) followed by interleaved UV plane (H/2 × W).
    Output: (3, H, W) float32 in [0, 1]
    """
    import numpy as np
    nv12 = np.frombuffer(nv12_buffer, dtype=np.uint8)
    Y  = nv12[:H * W].reshape(H, W).astype(np.float32)
    UV = nv12[H * W:].reshape(H // 2, W)
    U  = UV[:, 0::2].repeat(2, axis=0).repeat(2, axis=1).astype(np.float32)
    V  = UV[:, 1::2].repeat(2, axis=0).repeat(2, axis=1).astype(np.float32)

    # BT.601 YUV → RGB (limited range)
    Y  = (Y  - 16.0) / 219.0
    U  = (U  - 128.0) / 224.0
    V  = (V  - 128.0) / 224.0
    R = Y + 1.402 * V
    G = Y - 0.344136 * U - 0.714136 * V
    B = Y + 1.772 * U
    rgb = np.stack([R, G, B], axis=0).clip(0, 1)      # (3, H, W)
    return torch.from_numpy(rgb)                        # shape: (3, 2160, 3840) float32
```

---

## Learned ISP: The Convergence of Signal Processing and AI

The boundary between classical ISP and neural network is dissolving. Several ISP stages are now commonly replaced or augmented by small learned models:

| Stage | Classical approach | Learned replacement |
|---|---|---|
| Demosaicing | AHD, LMMSE | Small CNN (e.g., 3×3 conv, ~5 layers, 0.5M params) |
| Noise reduction | Bilateral, BM3D | DnCNN, FFDNet, camera-specific learned NR |
| HDR merge | Exposure bracketing + Laplacian pyramid | U-Net or attention-based merge |
| Face enhancement | Guided filter, frequency separation | GAN-based enhancement (ArcFace-conditioned) |
| Tone mapping | Global gamma + local | Lightweight Transformer or CNN |

**Constraint:** these learned ISP stages run on an NPU at real-time video rates. At 4K@30 = 30 frames/second, you have **33 ms per frame** budget. A typical learned NR model with ~1M parameters at INT8 costs ~10M MACs/pixel × 8M pixels = 80G MACs/frame. At 10 TOPS (INT8 NPU), that is 8 ms — feasible, but it consumes most of a mid-tier NPU's budget.

The trend for 2024–2026: **ISPs expose their internal buffers as NPU input/output interfaces**, letting vendors insert learned stages at any point in the pipeline — not just as a post-processing step, but inline between classical stages.

---

## Why This Matters for Model Optimization

If you build or deploy vision models on hardware with cameras, the ISP is your silent collaborator:

<div class="diagram">
<div class="diagram-title">ISP Decisions That Affect Your Model</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Preprocessing Offload</div>
    <div class="card-desc">Resize, normalize, color-space convert can be done by the ISP or GPU, freeing NPU cycles. Know your SoC's ISP output format (NV12 vs RGB) to avoid a redundant copy.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Domain Gap</div>
    <div class="card-desc">ISP tone mapping, sharpening, and NR settings affect how your model perceives a scene. A model trained on ISP-A images may degrade on ISP-B. Always validate on the target ISP's output.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-title">Bandwidth Budget</div>
    <div class="card-desc">ISP and NPU share DDR bandwidth. On budget SoCs (&lt;30 GB/s), the ISP alone can consume 30–50% of available bandwidth. Schedule ISP and NPU inference to avoid contention.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Latency Path</div>
    <div class="card-desc">ISP adds ~10–30 ms before the first frame is ready. For ultra-low-latency applications (AR overlays, robotics), consider partial-pipeline passthrough or learn to fuse ISP + NPU latency.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Learned Demosaic/Denoise</div>
    <div class="card-desc">If you control the device firmware, replacing classical NR with a model-aware NR (trained jointly with your downstream vision model) can recover 1–2 dB SNR and measurably improve task accuracy at no extra NR power.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">RAW ML Training Data</div>
    <div class="card-desc">Collecting ISP RAW output alongside processed frames builds a training dataset that captures real-world sensor noise. This is how large phone companies train their learned ISP components.</div>
  </div>
</div>
</div>

---

## Key Takeaways

- An **ISP** is a dedicated pipeline chip (or SoC block) that converts **Bayer RAW** sensor data into a processed image through ~10 sequential stages: black level → demosaic → white balance → denoise → CCM → tone map → sharpen → encode.
- The ISP processes **hundreds of millions to over a billion pixels per second** at 100–300 mW — a CPU doing the same would use 10–20× more power and still be too slow for real-time 4K.
- **Demosaicing** interpolates the missing two color channels per pixel; **NR** (bilateral or temporal) is the most computationally expensive stage, typically using bilateral filtering and motion-compensated frame blending.
- Modern ISPs are **programmable** (SIMD co-processor + firmware), enabling algorithm updates post-manufacture and learned stages running on an embedded NPU or the SoC's main NPU.
- The **ISP↔NPU interface** is the key architectural seam for vision AI: most ISPs output **NV12 YUV** frames that the NPU consumes directly; zero-copy paths (Apple, Qualcomm) avoid DDR round-trips.
- **Computational photography** (Night Sight, Smart HDR, Portrait mode) is ISP + NPU co-execution — the ISP handles motion alignment and fixed-function stages; the NPU runs learned merge, enhancement, and semantic segmentation.
- For ML practitioners: know your ISP's output format and pipeline settings — they determine your model's input distribution and your bandwidth budget on shared-memory SoCs.

---

*Next: [Chapter 20 — FPGAs](./20_fpgas.md), where we move to the other end of the flexibility spectrum: reconfigurable logic that you can rewire in software.*

[← Back to Table of Contents](./README.md)
