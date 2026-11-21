#!/usr/bin/env python3
"""Generate the guide's static SVG figures (no plotting library needed).

    python3 tools/make_figures.py          # writes assets/images/*.svg
    python3 tools/make_figures.py --check  # exit 1 if a figure is out of date

Figures:
  transistor_counts.svg  Chapter 1  — transistors per chip, 1971–2026, log scale
  memory_wall.svg        Chapter 11 — NVIDIA compute vs bandwidth, indexed to V100,
                                      read from assets/data/accelerators.json

The SVGs are drawn for the guide's dark theme (they sit on the figure's dark
surface) and use the same copper accent as the rest of the site.
"""
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "images"

FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"
MONO = "JetBrains Mono, SF Mono, Consolas, monospace"
TEXT, MUTED, GRID, AXIS = "#e5e5e5", "#a3a3a3", "#2a2a2a", "#525252"
COPPER, BLUE, AMBER, GREY = "#e8916a", "#60a5fa", "#fbbf24", "#737373"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(w, h, body, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-labelledby="t d" font-family="{FONT}">\n'
            f'<title id="t">{esc(title)}</title>\n<desc id="d">{esc(desc)}</desc>\n'
            f'<rect width="{w}" height="{h}" fill="#141414"/>\n{body}</svg>\n')


def text(x, y, s, size=11, fill=MUTED, anchor="start", weight=400, family=FONT):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-family="{family}">{esc(s)}</text>\n')


def line(x1, y1, x2, y2, stroke=GRID, width=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{d}/>\n'


def fmt_count(v):
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if v >= div:
            x = v / div
            return f"{x:.0f}{unit}" if x >= 10 or x == int(x) else f"{x:.1f}{unit}"
    return f"{v:.0f}"


# --------------------------------------------------------------------------- figure 1
CHIPS = [  # (year, transistors, name, kind, label?)
    (1971, 2.3e3, "Intel 4004", "cpu", True),
    (1974, 4.5e3, "Intel 8080", "cpu", False),
    (1978, 2.9e4, "Intel 8086", "cpu", True),
    (1982, 1.34e5, "Intel 80286", "cpu", False),
    (1985, 2.75e5, "Intel 80386", "cpu", False),
    (1989, 1.2e6, "Intel 80486", "cpu", True),
    (1993, 3.1e6, "Pentium", "cpu", False),
    (1997, 7.5e6, "Pentium II", "cpu", False),
    (2000, 4.2e7, "Pentium 4", "cpu", True),
    (2006, 2.91e8, "Core 2 Duo", "cpu", False),
    (2006, 6.81e8, "G80 (first CUDA GPU)", "gpu", True),
    (2010, 3.0e9, "Fermi GF100", "gpu", False),
    (2012, 7.1e9, "Kepler GK110", "gpu", False),
    (2016, 1.53e10, "P100", "gpu", False),
    (2017, 2.11e10, "V100", "gpu", False),
    (2020, 1.18e10, "Apple A14", "cpu", True),
    (2020, 5.42e10, "A100", "gpu", False),
    (2022, 8.0e10, "H100", "gpu", True),
    (2023, 1.53e11, "MI300X", "gpu", False),
    (2024, 2.08e11, "B200 (2 dies)", "gpu", True),
    (2026, 3.36e11, "Rubin (2 dies)", "gpu", True),
]
WAFER = (2024, 4.0e12, "Cerebras WSE-3 (whole wafer)")
LABEL_POS = {  # (dx, dy, anchor) for labels that would collide with neighbours or the edge
    "Intel 4004": (8, 14, "start"),
    "Apple A14": (8, 14, "start"),
    "H100": (-9, -8, "end"),
    "B200 (2 dies)": (-9, -6, "end"),
    "Rubin (2 dies)": (-9, -12, "end"),
}


def transistor_counts():
    W, H = 760, 440
    L, R, T, B = 70, 24, 52, 52
    x0, x1, y0, y1 = 1968, 2028, 3, 13          # years; log10 range
    px = lambda yr: L + (yr - x0) / (x1 - x0) * (W - L - R)
    py = lambda v: T + (y1 - math.log10(v)) / (y1 - y0) * (H - T - B)
    b = []
    b.append(text(L, 24, "Transistors per chip, 1971–2026", 15, TEXT, weight=700))
    b.append(text(L, 41, "log scale · dashed line = doubling every two years from the 4004", 11, MUTED))
    for e in range(y0, y1 + 1):
        y = py(10 ** e)
        b.append(line(L, y, W - R, y))
        b.append(text(L - 8, y + 4, fmt_count(10 ** e), 10.5, MUTED, "end", family=MONO))
    for yr in range(1970, 2030, 10):
        x = px(yr)
        b.append(line(x, T, x, H - B))
        b.append(text(x, H - B + 18, str(yr), 10.5, MUTED, "middle", family=MONO))
    b.append(line(L, H - B, W - R, H - B, AXIS))
    # Moore line: 2,300 doubling every 2 years
    pts = [(yr, 2300 * 2 ** ((yr - 1971) / 2)) for yr in range(1971, 2028)]
    pts = [(x, v) for x, v in pts if v <= 10 ** y1]
    d = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(v):.1f}" for i, (x, v) in enumerate(pts))
    b.append(f'<path d="{d}" fill="none" stroke="{GREY}" stroke-width="1.5" stroke-dasharray="5 4"/>\n')
    for yr, v, name, kind, lab in CHIPS:
        c = BLUE if kind == "cpu" else COPPER
        x, y = px(yr), py(v)
        b.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{c}" stroke="#141414" stroke-width="1"/>\n')
        if lab:
            dx, dy, anchor = LABEL_POS.get(name, (8, -6, "start"))
            b.append(text(x + dx, y + dy, f"{name} · {fmt_count(v)}", 10, TEXT, anchor))
    x, y = px(WAFER[0]), py(WAFER[1])
    b.append(f'<rect x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" fill="none" stroke="{AMBER}" stroke-width="1.6"/>\n')
    b.append(text(x - 9, y + 4, f"{WAFER[2]} · 4T", 10, AMBER, "end"))
    # legend
    lx, ly = L + 12, T + 14
    for i, (c, s) in enumerate([(BLUE, "CPUs and phone SoCs"), (COPPER, "GPUs and AI accelerators")]):
        b.append(f'<circle cx="{lx + 4}" cy="{ly + i * 17 - 4}" r="4.2" fill="{c}"/>\n')
        b.append(text(lx + 14, ly + i * 17, s, 10.5, TEXT))
    return svg(W, H, "".join(b), "Transistors per chip, 1971 to 2026",
               "Log-scale scatter of transistor counts from the Intel 4004 (2,300) to NVIDIA Rubin (336 billion), "
               "with a reference line doubling every two years. The points track the line for five decades.")


# --------------------------------------------------------------------------- figure 2
def memory_wall():
    data = {c["id"]: c for c in json.loads((ROOT / "assets/data/accelerators.json").read_text())["chips"]}
    ids = ["v100", "a100", "h100", "h200", "b200", "b300", "rubin"]
    rows = [data[i] for i in ids]
    base_bf16, base_bw = rows[0]["bf16_tf"], rows[0]["mem_tbps"]
    best = lambda c: c["fp4_tf"] or c["fp8_tf"] or c["bf16_tf"]
    series = [
        ("Tensor FLOP/s at the lowest supported precision", COPPER, [best(c) / base_bf16 for c in rows]),
        ("Tensor FLOP/s at FP16/BF16 (same precision)", AMBER, [c["bf16_tf"] / base_bf16 if c["bf16_tf"] else None for c in rows]),
        ("HBM bandwidth", BLUE, [c["mem_tbps"] / base_bw for c in rows]),
    ]
    W, H = 760, 440
    L, R, T, B = 64, 150, 52, 60
    y0, y1 = 0, 3   # log10 of index: 1x .. 1000x
    n = len(rows)
    px = lambda i: L + 30 + i * (W - L - R - 60) / (n - 1)
    py = lambda v: T + (y1 - math.log10(v)) / (y1 - y0) * (H - T - B)
    b = []
    b.append(text(L, 24, "The memory wall: NVIDIA compute vs bandwidth, V100 = 1×", 15, TEXT, weight=700))
    b.append(text(L, 41, "log scale · dense rates · data from assets/data/accelerators.json", 11, MUTED))
    for v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        y = py(v)
        b.append(line(L, y, W - R, y, GRID if v not in (1, 10, 100, 1000) else "#333"))
        b.append(text(L - 8, y + 4, f"{v}×", 10.5, MUTED, "end", family=MONO))
    for i, c in enumerate(rows):
        x = px(i)
        b.append(line(x, T, x, H - B))
        nm = c["name"].replace(" SXM2", "").replace(" SXM 80GB", "").replace(" SXM", "").replace(" (Blackwell Ultra)", "").replace(" (HGX)", "").replace(" (VR200)", "")
        b.append(text(x, H - B + 18, nm, 10.5, TEXT, "middle", 600))
        b.append(text(x, H - B + 32, str(c["year"]), 10, MUTED, "middle", family=MONO))
    b.append(line(L, H - B, W - R, H - B, AXIS))
    for name, col, vals in series:
        pts = [(px(i), py(v)) for i, v in enumerate(vals) if v]
        d = " ".join(f"{'M' if k == 0 else 'L'}{x:.1f},{y:.1f}" for k, (x, y) in enumerate(pts))
        b.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.4" stroke-linejoin="round"/>\n')
        for x, y in pts:
            b.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{col}" stroke="#141414" stroke-width="1"/>\n')
        last_i = max(i for i, v in enumerate(vals) if v)
        lx, ly = px(last_i) + 10, py(vals[last_i]) + 4
        b.append(text(lx, ly, f"{vals[last_i]:.0f}×", 12, col, weight=700, family=MONO))
    # legend under the title area, right column
    lx, ly = W - R + 18, T + 120
    for i, (name, col, _) in enumerate(series):
        words, lines_, cur = name.split(), [], ""
        for w_ in words:
            if len(cur) + len(w_) > 17:
                lines_.append(cur); cur = w_
            else:
                cur = (cur + " " + w_).strip()
        lines_.append(cur)
        yb = ly + i * 62
        b.append(f'<rect x="{lx}" y="{yb - 9}" width="12" height="3" fill="{col}"/>\n')
        for k, s in enumerate(lines_):
            b.append(text(lx + 18, yb - 4 + k * 13, s, 10, TEXT))
    gap_last = series[0][2][-1] / series[2][2][-1]
    b.append(text(lx, ly + 3 * 62 + 4, f"gap in 2026: ~{gap_last:.0f}×", 11, COPPER, weight=700))
    return svg(W, H, "".join(b), "The memory wall, V100 to Rubin",
               "Three lines indexed to V100 on a log scale: lowest-precision tensor throughput rises about 280 times, "
               "same-precision BF16 throughput about 18 times by B300, and HBM bandwidth about 24 times by Rubin.")


FIGURES = {"transistor_counts.svg": transistor_counts, "memory_wall.svg": memory_wall}


def main(check=False):
    OUT.mkdir(parents=True, exist_ok=True)
    stale = []
    for name, fn in FIGURES.items():
        path, new = OUT / name, fn()
        old = path.read_text() if path.exists() else None
        if old != new:
            stale.append(name)
            if not check:
                path.write_text(new)
        print(f"  {'stale' if old != new else 'ok   '} {name}")
    if check and stale:
        sys.exit(1)


if __name__ == "__main__":
    main("--check" in sys.argv)
