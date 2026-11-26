#!/usr/bin/env python3
"""Render the guide's flowcharts to inline SVG and inject them into the chapters.

Each chart is data (nodes with centre coordinates, edges between ports) so the
pictures stay editable. Output goes between markers in the Markdown source:

    <!-- flowchart:NAME -->
    ...generated, do not edit...
    <!-- /flowchart:NAME -->

Styling comes from the `.flowchart` / `.fc-*` rules in assets/css/style.css, so
the SVG carries classes, not colours. Links use .html because jekyll-relative-links
does not rewrite hrefs inside raw HTML.

Run:  python3 tools/build_flowcharts.py          (rewrites files in place)
      python3 tools/build_flowcharts.py --check  (exit 1 if anything is stale)
"""
import html
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- shapes
def N(id, x, y, lines, kind="proc", k=None, sub=(), w=None, h=None, link=None):
    lines = [lines] if isinstance(lines, str) else list(lines)
    sub = [sub] if isinstance(sub, str) else list(sub)
    if w is None:
        w = 300 if kind == "dec" else 190
    if h is None:
        h = 18 + 14 * len(lines) + 12 * len(sub) + (10 if kind == "dec" else 0)
    return dict(id=id, x=x, y=y, lines=lines, kind=kind, k=k, sub=sub, w=w, h=h, link=link)


def E(a, b, pa="b", pb="t", label=None, kind=None, via=(), lpos=None):
    return dict(a=a, b=b, pa=pa, pb=pb, label=label, kind=kind, via=list(via), lpos=lpos)


def port(n, p):
    """'b', 't', 'l', 'r', optionally with an offset along that side, e.g. 'b-60'."""
    x, y, w, h = n["x"], n["y"], n["w"], n["h"]
    side, off = p[0], float(p[1:]) if len(p) > 1 else 0.0
    if side in "bt":
        return (x + off, y + h / 2 if side == "b" else y - h / 2)
    return (x - w / 2 if side == "l" else x + w / 2, y + off)


def route(n1, n2, e):
    p0, p1 = port(n1, e["pa"]), port(n2, e["pb"])
    if e["via"]:
        return [p0, *e["via"], p1]
    (x0, y0), (x1, y1) = p0, p1
    if abs(x0 - x1) < 0.5 or abs(y0 - y1) < 0.5:
        return [p0, p1]
    if e["pa"][0] in "bt" and e["pb"][0] in "bt":
        ym = (y0 + y1) / 2
        return [p0, (x0, ym), (x1, ym), p1]
    if e["pa"][0] in "lr" and e["pb"][0] in "lr":
        xm = (x0 + x1) / 2
        return [p0, (xm, y0), (xm, y1), p1]
    if e["pa"][0] in "lr":                    # side then down/up
        return [p0, (x1, y0), p1]
    return [p0, (x0, y1), p1]                 # down/up then side


def f(v):
    return f"{v:.1f}".rstrip("0").rstrip(".")


def svg_node(n):
    x, y, w, h = n["x"], n["y"], n["w"], n["h"]
    cls = "fc-node"
    if n["kind"] == "dec":
        cls += " decision"
    elif n["kind"] == "start":
        cls += " start"
    elif n["kind"] == "leaf":
        cls += " leaf"
    if n["k"]:
        cls += " k-" + n["k"]
    out = [f'<g class="{cls}">']
    if n["kind"] == "dec":
        c = 14
        pts = [(x - w / 2, y), (x - w / 2 + c, y - h / 2), (x + w / 2 - c, y - h / 2), (x + w / 2, y), (x + w / 2 - c, y + h / 2), (x - w / 2 + c, y + h / 2)]
        out.append('<polygon points="' + " ".join(f"{f(a)},{f(b)}" for a, b in pts) + '"/>')
    else:
        r = h / 2 if n["kind"] == "start" else 6
        out.append(f'<rect x="{f(x - w / 2)}" y="{f(y - h / 2)}" width="{f(w)}" height="{f(h)}" rx="{f(min(r, 16))}"/>')
    total = 14 * len(n["lines"]) + 12 * len(n["sub"])
    ty = y - total / 2 + 11
    for line in n["lines"]:
        out.append(f'<text x="{f(x)}" y="{f(ty)}" text-anchor="middle">{html.escape(line)}</text>')
        ty += 14
    for line in n["sub"]:
        out.append(f'<text class="sub" x="{f(x)}" y="{f(ty - 1)}" text-anchor="middle">{html.escape(line)}</text>')
        ty += 12
    out.append("</g>")
    s = "".join(out)
    if n["link"]:
        s = f'<a href="{n["link"]}">{s}</a>'
    return s


def svg_edge(n1, n2, e):
    pts = route(n1, n2, e)
    k = e["kind"] or ("yes" if e["label"] and e["label"].lower().startswith("yes") else "no" if e["label"] and e["label"].lower().startswith("no") else "")
    cls = f"fc-edge {k}".strip()
    d = "M" + " L".join(f"{f(a)},{f(b)}" for a, b in pts)
    out = [f'<path class="{cls}" d="{d}"/>']
    (xa, ya), (xb, yb) = pts[-2], pts[-1]
    L = max(1e-6, ((xb - xa) ** 2 + (yb - ya) ** 2) ** 0.5)
    ux, uy = (xb - xa) / L, (yb - ya) / L
    s = 7
    p1 = (xb - ux * s - uy * s * 0.55, yb - uy * s + ux * s * 0.55)
    p2 = (xb - ux * s + uy * s * 0.55, yb - uy * s - ux * s * 0.55)
    head = f"fc-head {k}".strip()
    out.append(f'<polygon class="{head}" points="{f(xb)},{f(yb)} {f(p1[0])},{f(p1[1])} {f(p2[0])},{f(p2[1])}"/>')
    if e["label"]:
        if e["lpos"]:
            lx, ly, anchor = e["lpos"]
        else:
            (x0, y0), (x1, y1) = pts[0], pts[1]
            if abs(y1 - y0) > abs(x1 - x0):
                lx, ly, anchor = x0 + 6, y0 + 14, "start"
            else:
                lx, ly, anchor = x0 + (10 if x1 > x0 else -10), y0 - 6, "start" if x1 > x0 else "end"
        out.append(f'<text class="fc-elabel {k}" x="{f(lx)}" y="{f(ly)}" text-anchor="{anchor}">{html.escape(e["label"])}</text>')
    return "".join(out)


def render(name, chart):
    nodes = {n["id"]: n for n in chart["nodes"]}
    parts = [f'<svg viewBox="0 0 {chart["w"]} {chart["h"]}" width="{chart["w"]}" role="img" aria-label="{html.escape(chart["title"])}">']
    for e in chart["edges"]:
        parts.append(svg_edge(nodes[e["a"]], nodes[e["b"]], e))
    for n in chart["nodes"]:
        parts.append(svg_node(n))
    for (x, y, text, anchor) in chart.get("notes", []):
        parts.append(f'<text class="fc-elabel" x="{x}" y="{y}" text-anchor="{anchor}">{html.escape(text)}</text>')
    parts.append("</svg>")
    body = "\n".join(parts)
    cap = f'<div class="fc-caption">{chart["caption"]}</div>' if chart.get("caption") else ""
    return (f'<div class="flowchart" id="fc-{name}">\n<div class="fc-title">{html.escape(chart["title"])}</div>\n'
            f'{body}\n{cap}\n</div>').replace("\n\n", "\n")


# --------------------------------------------------------------------------- charts
CHARTS = {}

# 07 — which chip?  (spine: each decision either answers "yes, this chip" or falls through)
_y = [26, 100, 186, 272, 358, 444, 534]
CHARTS["chip_picker"] = dict(
    file="07_types_of_chips_taxonomy.md", w=780, h=580,
    title="Which kind of chip should run this workload?",
    caption="A first-cut decision path. Real systems mix several answers: a phone SoC has a CPU, GPU, NPU, DSP and ISP on one die. Click a box to jump to its chapter.",
    nodes=[
        N("s", 230, _y[0], "A workload to run", kind="start", w=220),
        N("d1", 230, _y[1], ["Branchy, serial or OS-level code?"], kind="dec", sub="control flow dominates, little data parallelism"),
        N("d2", 230, _y[2], ["A fixed signal or image pipeline?"], kind="dec", sub="filters, FFTs, demosaic, codecs"),
        N("d3", 230, _y[3], ["Battery or sub-10 W power budget?"], kind="dec", sub="phone, laptop, camera, wearable"),
        N("d4", 230, _y[4], ["Microsecond latency or custom I/O,"], kind="dec", sub="at low volume?"),
        N("d5", 230, _y[5], ["Millions of units of a stable model"], kind="dec", sub="family, and a silicon team?"),
        N("cpu", 610, _y[1], "CPU", kind="leaf", k="orange", sub="out-of-order cores, big caches", w=250, link="./14_cpus.html"),
        N("dsp", 610, _y[2], "DSP / ISP", kind="leaf", k="teal", sub="fixed-point MACs, VLIW, hardwired stages", w=250, link="./18_dsps.html"),
        N("npu", 610, _y[3], "NPU inside an SoC", kind="leaf", k="green", sub="INT8/INT4 MAC arrays, tiny power", w=250, link="./17_npus_and_edge_ai.html"),
        N("fpga", 610, _y[4], "FPGA", kind="leaf", k="yellow", sub="reconfigurable fabric, deterministic", w=250, link="./20_fpgas.html"),
        N("asic", 610, _y[5], "ASIC / TPU-class accelerator", kind="leaf", k="pink", sub="best perf/W, huge NRE, slow to change", w=250, link="./21_asics.html"),
        N("gpu", 230, _y[6], "GPU", kind="leaf", k="blue", sub="the programmable default for dense math", w=300, link="./15_gpus.html"),
    ],
    edges=[E("s", "d1")] + [E(f"d{i}", f"d{i+1}", label="no") for i in range(1, 5)] + [E("d5", "gpu", label="no")] +
          [E(f"d{i}", t, "r", "l", label="yes") for i, t in zip(range(1, 6), ["cpu", "dsp", "npu", "fpga", "asic"])],
)

# 11 — why is it slow?
CHARTS["bottleneck"] = dict(
    file="11_memory_wall_and_bandwidth.md", w=780, h=470,
    title="Why is my kernel slow? A roofline diagnosis",
    caption="Measure first (Nsight Systems for gaps, Nsight Compute or the PyTorch profiler for achieved FLOP/s and bytes/s), then walk down. The 70 % and 60 % thresholds are rules of thumb, not laws.",
    nodes=[
        N("s", 330, 24, "Model step or kernel is slow", kind="start", w=250),
        N("d1", 330, 92, ["Is the accelerator idle between kernels?"], kind="dec", w=330, sub="gaps in the profiler timeline"),
        N("ov", 640, 92, "Overhead-bound", kind="leaf", k="red", sub=["CUDA Graphs, torch.compile,", "fuse tiny ops, bigger batches"], w=230),
        N("d2", 330, 188, ["Arithmetic intensity below the ridge?"], kind="dec", w=330, sub="FLOPs ÷ bytes moved vs peak ÷ bandwidth"),
        N("d3", 175, 290, ["Achieved bandwidth ≥ ~70 % of peak?"], kind="dec", w=300),
        N("d4", 560, 290, ["Achieved FLOP/s ≥ ~60 % of peak?"], kind="dec", w=300),
        N("mem", 95, 408, "Memory-bound", kind="leaf", k="purple", sub=["cut bytes: fuse, quantize,", "batch, keep data in SRAM"], w=180, link="./29_memory_hierarchy_in_action.html"),
        N("lat", 285, 408, "Latency-bound", kind="leaf", k="yellow", sub=["more blocks in flight, coalesce,", "async copies (TMA, cp.async)"], w=190, link="./15_gpus.html"),
        N("tc", 475, 408, "Under-using the math units", kind="leaf", k="orange", sub=["tile sizes, dims ÷ 16, pipelining,", "a dtype the tensor cores run"], w=180, link="./24_datatypes_and_silicon.html"),
        N("cmp", 670, 408, "Compute-bound", kind="leaf", k="green", sub=["lower precision, 2:4 sparsity,", "more chips"], w=190, link="./25_quantization_hardware_view.html"),
    ],
    edges=[E("s", "d1"), E("d1", "ov", "r", "l", label="yes"), E("d1", "d2", label="no"),
           E("d2", "d3", "b-60", "t", label="yes", lpos=(250, 236, "end")), E("d2", "d4", "b+60", "t", label="no", lpos=(410, 236, "start")),
           E("d3", "mem", "b-40", "t", label="yes", lpos=(118, 360, "end")), E("d3", "lat", "b+40", "t", label="no", lpos=(262, 360, "start")),
           E("d4", "tc", "b-40", "t", label="no", lpos=(498, 360, "end")), E("d4", "cmp", "b+40", "t", label="yes", lpos=(647, 360, "start"))],
)

# 28 — which parallelism?
_p = [24, 96, 184, 272, 360, 452]
CHARTS["parallelism"] = dict(
    file="28_multi_gpu_and_distributed_training.md", w=800, h=500,
    title="Which parallelism does this training run need?",
    caption="Start at the top and add techniques only when the previous answer is no. Large runs end at the bottom box: tensor parallel inside an NVLink domain, pipeline across nodes, data/FSDP across replicas.",
    nodes=[
        N("s", 240, _p[0], "Train a model on N GPUs", kind="start", w=230),
        N("d1", 240, _p[1], ["Does a full replica fit on one GPU?"], kind="dec", w=360, sub="≈16 B/param (BF16 + FP32 Adam) + activations"),
        N("d2", 240, _p[2], ["Does it fit once states are sharded?"], kind="dec", w=360, sub="16 B/param ÷ N + activations"),
        N("d3", 240, _p[3], ["Is one layer (or its activations) too big,"], kind="dec", w=360, sub="or is step time too long?"),
        N("d4", 240, _p[4], ["Mixture-of-experts model?"], kind="dec", w=360),
        N("dp", 620, _p[1], "Data parallel (DDP)", kind="leaf", k="green", sub="replicate, all-reduce gradients", w=260),
        N("fsdp", 620, _p[2], "FSDP / ZeRO-3", kind="leaf", k="cyan", sub="shard params, grads, optimizer", w=260),
        N("tp", 620, _p[3], "Tensor + sequence parallel", kind="leaf", k="purple", sub="split each matmul inside one NVLink domain", w=260),
        N("ep", 620, _p[4], "Expert parallel", kind="leaf", k="pink", sub="experts on different GPUs, all-to-all", w=260),
        N("mix", 400, _p[5], "Combine: TP inside a node × pipeline across nodes × DP/FSDP across replicas", kind="leaf", k="orange", sub="+ context parallel for long sequences (\"3D/4D parallelism\")", w=620),
    ],
    edges=[E("s", "d1"), E("d1", "d2", label="no"), E("d2", "d3", label="no"), E("d3", "d4", label="no"),
           E("d4", "mix", "b", "t", label="no", via=[(240, 412), (400, 412)]),
           E("d1", "dp", "r", "l", label="yes"), E("d2", "fsdp", "r", "l", label="yes"), E("d3", "tp", "r", "l", label="yes"), E("d4", "ep", "r", "l", label="yes")],
)

# 29 — life of an LLM request
CHARTS["request_life"] = dict(
    file="29_memory_hierarchy_in_action.md", w=800, h=330,
    title="The life of one LLM request, seen from memory",
    caption="Prefill is one large, compute-bound pass; decode is a long loop of small, memory-bound passes. Almost every serving optimisation targets the loop.",
    nodes=[
        N("p", 100, 50, "Prompt arrives", kind="start", sub="N tokens", w=160),
        N("pf", 318, 50, "Prefill", k="green", sub=["all N tokens, one batched GEMM pass", "compute-bound"], w=220),
        N("kv", 640, 50, "KV cache in HBM", kind="leaf", k="purple", sub=["2 × layers × KV heads × head dim", "× bytes, per token"], w=240, link="./09_memory_types.html"),
        N("dec", 640, 168, "Decode step", k="orange", sub=["read every weight + this sequence's KV", "memory-bound at small batch"], w=260),
        N("eos", 640, 278, ["End of sequence?"], kind="dec", w=210),
        N("free", 360, 278, "Free the KV pages", k="cyan", sub="PagedAttention block table", w=210, link="./10_memory_management.html"),
        N("done", 110, 278, "Response complete", kind="start", w=180),
        N("bat", 300, 168, "Continuous batching", k="teal", sub=["many sequences share each weight read,", "raising FLOPs per byte"], w=260, link="./11_memory_wall_and_bandwidth.html"),
    ],
    edges=[E("p", "pf", "r", "l"), E("pf", "kv", "r", "l", label="writes K,V", lpos=(474, 41, "middle")), E("kv", "dec", "b", "t"),
           E("dec", "eos", "b", "t"), E("eos", "dec", "r", "r", label="no: next token", kind="loop", via=[(790, 278), (790, 168)], lpos=(784, 226, "end")),
           E("eos", "free", "l", "r", label="yes", lpos=(525, 270, "end")), E("free", "done", "l", "r"),
           E("bat", "dec", "r", "l", kind="loop")],
)

# 30 — which datatype?
CHARTS["dtype_picker"] = dict(
    file="30_datatypes_drive_optimization_choices.md", w=780, h=440,
    title="Which number format? A first decision",
    caption="Defaults for transformer workloads as of late 2026. Every arrow ends in an experiment: confirm the choice on your own evaluation, not just perplexity.",
    nodes=[
        N("s", 390, 24, "Pick a number format", kind="start", w=220),
        N("d1", 390, 92, ["Training or inference?"], kind="dec", w=240),
        N("d2", 195, 190, ["FP8 tensor cores available?"], kind="dec", w=300, sub="H100+, MI300+, Gaudi 3, Ironwood"),
        N("d3", 585, 190, ["Small-batch decode?"], kind="dec", w=300, sub="memory-bound, latency-sensitive"),
        N("t8", 105, 304, "BF16 master + FP8 GEMMs", kind="leaf", k="green", sub=["Transformer Engine or MXFP8;", "FP32 optimizer state"], w=190, link="./23_number_formats_deep_dive.html"),
        N("t16", 290, 304, "BF16 mixed precision", kind="leaf", k="blue", sub=["FP32 master weights;", "no loss scaling needed"], w=170, link="./03_number_representation.html"),
        N("i4", 485, 304, "Weight-only 4-bit + FP8 KV", kind="leaf", k="purple", sub=["AWQ/GPTQ INT4 or NVFP4;", "W8A16 if 4-bit hurts"], w=190, link="./25_quantization_hardware_view.html"),
        N("i8", 680, 304, "W8A8 FP8, or W4A4 FP4", kind="leaf", k="orange", sub=["FP4 only where tensor cores", "run it (Blackwell, MI355X, TPU 8)"], w=190, link="./24_datatypes_and_silicon.html"),
        N("sp", 390, 400, "Then: 2:4 sparsity if accuracy survives fine-tuning; re-run the eval after every step", kind="leaf", k="teal", w=600, link="./26_sparsity_hardware_view.html"),
    ],
    edges=[E("s", "d1"), E("d1", "d2", "b-40", "t", label="training", kind="yes", lpos=(300, 140, "end")), E("d1", "d3", "b+40", "t", label="inference", kind="no", lpos=(480, 140, "start")),
           E("d2", "t8", "b-50", "t", label="yes", lpos=(128, 252, "end")), E("d2", "t16", "b+50", "t", label="no", lpos=(267, 252, "start")),
           E("d3", "i4", "b-50", "t", label="yes", lpos=(508, 252, "end")), E("d3", "i8", "b+50", "t", label="no: compute-bound", lpos=(657, 252, "start"))],
)

# 31 — RTL to GDSII, with the loops that make it slow
CHARTS["design_flow"] = dict(
    file="31_chip_design_flow.md", w=780, h=420,
    title="From specification to silicon, and the loops in between",
    caption="Dashed copper edges are iterations. Most of a project's calendar is spent in the loops, not the straight path: verification typically consumes more engineering effort than design.",
    nodes=[
        N("spec", 130, 40, "Spec & architecture", kind="start", sub="performance model, PPA targets", w=200),
        N("rtl", 130, 122, "RTL design", k="blue", sub="SystemVerilog / Verilog", w=200),
        N("ver", 130, 204, "Functional verification", k="purple", sub="simulation, UVM, formal, emulation", w=200, link="./32_chip_verification_and_validation.html"),
        N("syn", 130, 286, "Logic synthesis", k="cyan", sub="RTL → gate netlist", w=200),
        N("fp", 390, 286, "Floorplan & power grid", k="green", w=200),
        N("pl", 390, 204, "Placement", k="green", w=200),
        N("cts", 390, 122, "Clock-tree synthesis", k="green", w=200),
        N("rt", 390, 40, "Routing", k="green", w=200),
        N("sta", 650, 40, "Static timing analysis", k="orange", sub="every path, every corner", w=200),
        N("pv", 650, 122, "Physical verification", k="orange", sub="DRC / LVS / antenna", w=200),
        N("sig", 650, 204, "Signoff", k="orange", sub="IR drop, EM, power, ECO", w=200),
        N("tape", 650, 286, "Tapeout (GDSII → masks)", kind="leaf", k="pink", w=200, link="./appendix_a_how_chips_are_made.html"),
        N("post", 650, 372, "Post-silicon bring-up", k="red", sub="validation on real parts", w=200),
    ],
    edges=[E("spec", "rtl"), E("rtl", "ver"), E("ver", "syn"), E("syn", "fp", "r", "l"), E("fp", "pl", "t", "b"), E("pl", "cts", "t", "b"), E("cts", "rt", "t", "b"),
           E("rt", "sta", "r", "l"), E("sta", "pv"), E("pv", "sig"), E("sig", "tape"), E("tape", "post"),
           E("ver", "rtl", "l", "l", label="bug", kind="loop", via=[(14, 204), (14, 122)], lpos=(20, 166, "start")),
           E("sta", "pl", "l", "r", label="timing fails", kind="loop", via=[(520, 48), (520, 204)], lpos=(526, 160, "start")),
           E("post", "rtl", "l", "l", label="silicon bug: metal ECO or respin", kind="loop", via=[(4, 372), (4, 130)], lpos=(300, 366, "middle"))],
)

# Appendix A — the fab loop
CHARTS["fab_loop"] = dict(
    file="appendix_a_how_chips_are_made.md", w=800, h=380,
    title="Sand to packaged chip: the manufacturing loop",
    caption="The ring in the middle runs once per mask layer, dozens of times for a leading-edge logic wafer, which is why a wafer spends roughly three months in the fab.",
    nodes=[
        N("sand", 95, 32, "Quartz sand", kind="start", w=160),
        N("poly", 285, 32, "Polysilicon", sub="99.9999999 % pure", w=170),
        N("ingot", 475, 32, "Single-crystal ingot", sub="Czochralski pull", w=170),
        N("wafer", 680, 32, "Wafers", sub="slice, polish to < 1 nm roughness", w=200),
        N("dep", 120, 134, "Deposit film", k="cyan", w=170),
        N("coat", 310, 134, "Coat photoresist", k="cyan", w=170),
        N("exp", 500, 134, "Expose (DUV / EUV)", k="purple", sub="through the mask (reticle)", w=180),
        N("dev", 690, 134, "Develop", k="cyan", w=170),
        N("etch", 690, 226, "Etch", k="cyan", w=170),
        N("imp", 500, 226, "Implant / strip resist", k="cyan", w=180),
        N("cmp", 310, 226, "CMP & clean", k="cyan", w=170),
        N("ins", 120, 226, "Inspect & measure", k="yellow", w=170),
        N("probe", 120, 336, "Wafer sort (probe test)", k="orange", w=190),
        N("dice", 330, 336, "Dice", k="orange", w=150),
        N("pkg", 510, 336, "Package", k="orange", sub="flip-chip, CoWoS, HBM", w=170, link="./appendix_e_packaging.html"),
        N("bin", 700, 336, "Final test & binning", kind="leaf", k="green", w=180),
    ],
    edges=[E("sand", "poly", "r", "l"), E("poly", "ingot", "r", "l"), E("ingot", "wafer", "r", "l"),
           E("wafer", "dep", "b", "t", via=[(680, 84), (120, 84)]),
           E("dep", "coat", "r", "l"), E("coat", "exp", "r", "l"), E("exp", "dev", "r", "l"), E("dev", "etch"),
           E("etch", "imp", "l", "r"), E("imp", "cmp", "l", "r"), E("cmp", "ins", "l", "r"),
           E("ins", "dep", "t", "b", label="next layer", kind="loop", lpos=(128, 186, "start")),
           E("ins", "probe", label="all layers done", lpos=(128, 286, "start")),
           E("probe", "dice", "r", "l"), E("dice", "pkg", "r", "l"), E("pkg", "bin", "r", "l")],
    notes=[(405, 184, "repeated for every mask layer", "middle")],
)


# Appendix H — boot once, then sleep and wake forever
CHARTS["boot_loop"] = dict(
    file="appendix_h_how_a_computer_runs.md", w=800, h=420,
    title="Boot once, then sleep and wake for the rest of its life",
    caption="The top row runs once per power-on. Everything after that is the bottom loop: the scheduler runs ready threads, and when none are ready the core halts until an interrupt wakes it.",
    nodes=[
        N("reset", 86, 36, "Reset", kind="start", sub="PC = reset vector", w=146),
        N("fw", 246, 36, "Firmware", k="cyan", sub=["UEFI / BIOS: POST,", "trains DRAM"], w=146),
        N("boot", 406, 36, "Bootloader", k="cyan", sub=["loads the kernel", "into RAM"], w=146),
        N("kern", 566, 36, "Kernel init", k="blue", sub=["page tables, IDT,", "other cores online"], w=146),
        N("init", 726, 36, "init (PID 1)", k="blue", sub=["services, login,", "display server"], w=146),
        N("sched", 400, 160, ["Scheduler: any thread ready?"], kind="dec", w=300),
        N("run", 150, 272, "Run a thread", k="green", sub=["for a time slice, until it", "blocks or is preempted"], w=230),
        N("idle", 650, 272, "Idle thread", k="purple", sub=["HLT / WFI, then deeper", "C-states while nothing happens"], w=230),
        N("irq", 400, 372, "Interrupt", k="orange", sub="timer tick, keypress, packet, \"GPU kernel done\"", w=300),
    ],
    edges=[E("reset", "fw", "r", "l"), E("fw", "boot", "r", "l"), E("boot", "kern", "r", "l"), E("kern", "init", "r", "l"),
           E("init", "sched", "b", "r", via=[(726, 160)]),
           E("sched", "run", "b-70", "t", label="yes", kind="yes", lpos=(322, 214, "end")),
           E("sched", "idle", "b+70", "t", label="no", kind="no", lpos=(478, 214, "start")),
           E("run", "sched", "l", "l", label="blocks / slice ends", kind="loop", via=[(20, 272), (20, 160)], lpos=(26, 150, "start")),
           E("idle", "irq", "b", "r", label="device raises IRQ", lpos=(660, 340, "start"), via=[(650, 372)]),
           E("irq", "run", "l", "b", label="ISR marks a thread ready", lpos=(140, 365, "end"), via=[(150, 372)])],
)


# Appendix I — where does one load go?
_m = [26, 100, 186, 272, 358, 444, 530]
CHARTS["memory_lookup"] = dict(
    file="appendix_i_memory_and_caches.md", w=780, h=580,
    title="Where does one load instruction find its data? (a modern CPU core)",
    caption="Each \"no\" costs roughly 3–10× more time than the level above it. A GPU follows the same path with registers, L1/shared memory, L2 and HBM, but every step takes several times longer, which is why GPUs hide latency with thousands of threads instead.",
    nodes=[
        N("s", 230, _m[0], "Core executes a load", kind="start", sub="virtual address", w=220),
        N("tlb", 230, _m[1], "Translate address", k="cyan", sub=["TLB hit ≈ free; miss → page-table walk", "(itself cached in L1/L2)"], w=300, link="./10_memory_management.html"),
        N("d1", 230, _m[2], ["In L1 data cache?"], kind="dec", sub="32–64 KB per core, SRAM"),
        N("d2", 230, _m[3], ["In L2 cache?"], kind="dec", sub="1–3 MB per core, SRAM"),
        N("d3", 230, _m[4], ["In L3 / last-level cache?"], kind="dec", sub="tens–hundreds of MB, shared"),
        N("d4", 230, _m[5], ["In DRAM?"], kind="dec", sub="GBs–TBs, DDR5 / LPDDR5X / HBM"),
        N("l1", 610, _m[2], "L1 hit · ~1 ns", kind="leaf", k="green", sub="4–5 cycles", w=250),
        N("l2", 610, _m[3], "L2 hit · ~3–5 ns", kind="leaf", k="teal", sub="~12–16 cycles", w=250),
        N("l3", 610, _m[4], "L3 hit · ~10–20 ns", kind="leaf", k="blue", sub="~40–70 cycles", w=250),
        N("dram", 610, _m[5], "DRAM · ~80–120 ns", kind="leaf", k="purple", sub="~300 cycles; line also filled into L3/L2/L1", w=250, link="./09_memory_types.html"),
        N("ssd", 230, _m[6], "Page fault → OS reads from SSD", kind="leaf", k="red", sub="~10–100 µs: a million cycles", w=300, link="./appendix_h_how_a_computer_runs.html"),
    ],
    edges=[E("s", "tlb"), E("tlb", "d1")] + [E(f"d{i}", f"d{i+1}", label="miss") for i in range(1, 4)] + [E("d4", "ssd", label="no")] +
          [E(f"d{i}", t, "r", "l", label="hit") for i, t in zip(range(1, 5), ["l1", "l2", "l3", "dram"])],
)


# --------------------------------------------------------------------------- inject
def inject(check=False):
    stale = []
    for name, chart in CHARTS.items():
        path = ROOT / chart["file"]
        text = path.read_text()
        start, end = f"<!-- flowchart:{name} -->", f"<!-- /flowchart:{name} -->"
        if start not in text:
            print(f"  ! {chart['file']}: no marker for {name}; skipped")
            continue
        block = f"{start}\n{render(name, chart)}\n{end}"
        new = re.sub(re.escape(start) + r".*?" + re.escape(end), lambda m: block, text, flags=re.S)
        if new != text:
            stale.append(chart["file"])
            if not check:
                path.write_text(new)
        print(f"  {'stale' if new != text else 'ok   '} {name:14s} → {chart['file']}")
    if check and stale:
        sys.exit(1)


if __name__ == "__main__":
    inject(check="--check" in sys.argv)
