#!/usr/bin/env python3
"""Generate SVG benchmark charts from Google Benchmark JSON output.

Usage:
    ./build/im2col_bench --benchmark_format=json --benchmark_out=docs/results.json
    ./build/gemm_bench   --benchmark_format=json --benchmark_out=docs/results_gemm.json
    python3 bench/gen_charts.py          # uses docs/results.json + docs/results_gemm.json

Produces SVG files in docs/:
    docs/gemm_comparison.svg   -- naive / cache-friendly / intrinsics GEMM (via conv)
    docs/conv_comparison.svg   -- conv_naive vs conv_im2col + intrinsics
    docs/speedup.svg           -- speedup line chart
    docs/gemm_size.svg         -- standalone GEMM time vs matrix size
"""

import json, sys, os, math

# ── style ──────────────────────────────────────────────────────────────────────
LABEL_COLOR = "#c9d1d9"
DIM_COLOR   = "#6e7681"
GRID_COLOR  = "#21262d"
AXIS_COLOR  = "#8b949e"

COLORS = {
    "BM_ConvNaive":                   "#58a6ff",
    "BM_ConvIm2colNaiveGemm":         "#bc8cff",
    "BM_ConvIm2colCacheFriendlyGemm": "#3fb950",
    "BM_ConvIm2colIntrinsicsGemm":    "#f78166",
}
LABELS = {
    "BM_ConvNaive":                   "conv_naive",
    "BM_ConvIm2colNaiveGemm":         "im2col + naive GEMM",
    "BM_ConvIm2colCacheFriendlyGemm": "im2col + cache-friendly",
    "BM_ConvIm2colIntrinsicsGemm":    "im2col + intrinsics",
}


# ── JSON parsing ───────────────────────────────────────────────────────────────
def parse_results(path):
    with open(path) as f:
        raw = json.load(f)

    # Two-pass: first collect all entries, prefer median aggregates when present.
    # If --benchmark_repetitions was used, JSON contains both individual runs and
    # aggregates (mean/median/stddev). We prefer _median; fall back to raw run.
    out      = {}   # {bench_name: {K: ms}}
    has_agg  = any(bm.get("aggregate_name") == "median"
                   for bm in raw["benchmarks"])

    for bm in raw["benchmarks"]:
        agg = bm.get("aggregate_name", "")

        if has_agg:
            if agg != "median":   # skip mean, stddev, individual runs
                continue
        else:
            if bm.get("run_type") == "aggregate":
                continue

        # strip aggregate suffix from name ("BM_Foo/1/2_median" -> "BM_Foo/1/2")
        name  = bm["name"].removesuffix(f"_{agg}") if agg else bm["name"]
        parts = name.split("/")
        bname = parts[0]
        if len(parts) < 7:
            continue
        k_h, k_w = int(parts[5]), int(parts[6])
        if k_h != k_w:
            continue
        K = k_h

        t    = bm["real_time"]
        unit = bm.get("time_unit", "ns")
        t_ms = {"ns": t / 1e6, "us": t / 1e3, "ms": t, "s": t * 1e3}[unit]

        out.setdefault(bname, {})[K] = t_ms

    mode = "median" if has_agg else "single run"
    print(f"  mode: {mode}")
    return out


# ── SVG primitives ─────────────────────────────────────────────────────────────
def svg_open(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n')

def svg_close():
    return '</svg>\n'

def svgtext(x, y, s, anchor="middle", size=12, color=LABEL_COLOR,
            bold=False, italic=False):
    style = f"font-family:sans-serif;font-size:{size}px;"
    if bold:   style += "font-weight:700;"
    if italic: style += "font-style:italic;"
    return (f'  <text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'style="{style}" fill="{color}">{s}</text>\n')

def svgline(x1, y1, x2, y2, color=AXIS_COLOR, width=1.5, dash=""):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <line x1="{x1:.1f}" y1="{y1:.1f}" '
            f'x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}"{da}/>\n')

def svgrect(x, y, w, h, color, opacity=0.85, rx=3):
    return (f'  <rect x="{x:.1f}" y="{y:.1f}" '
            f'width="{w:.1f}" height="{h:.1f}" '
            f'fill="{color}" fill-opacity="{opacity}" rx="{rx}"/>\n')

def svgcircle(x, y, r, color):
    return f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}"/>\n'

def svgpolyline(points, color, width=2.5):
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (f'  <polyline points="{pts}" fill="none" '
            f'stroke="{color}" stroke-width="{width}" '
            f'stroke-linejoin="round" stroke-linecap="round"/>\n')

def svgrottext(x, y, s, angle, size=11, color=DIM_COLOR):
    style = f"font-family:sans-serif;font-size:{size}px;"
    return (f'  <text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" '
            f'transform="rotate({angle},{x:.1f},{y:.1f})" '
            f'style="{style}" fill="{color}">{s}</text>\n')


# ── nice Y-axis scale ──────────────────────────────────────────────────────────
def nice_scale(v_max, n_ticks=5):
    raw = v_max / n_ticks
    mag = 10 ** math.floor(math.log10(raw))
    for step in (1, 2, 2.5, 5, 10):
        s = step * mag
        if s * n_ticks >= v_max:
            return s * n_ticks, s
    return mag * 10 * n_ticks, mag * 10


# ── grouped bar chart ──────────────────────────────────────────────────────────
def make_bar_chart(title, data, series_keys, k_values, out_path):
    W, H            = 800, 480
    ML, MR, MT, MB  = 78, 28, 58, 96

    cw = W - ML - MR
    ch = H - MT - MB

    series   = [(k, COLORS[k], LABELS[k]) for k in series_keys if k in data]
    k_values = [K for K in k_values
                if any(K in data[bk] for bk, *_ in series)]

    if not series or not k_values:
        print(f"  no data for {out_path}, skipping")
        return

    max_t         = max(data[bk][K]
                        for bk, _, _ in series
                        for K in k_values if K in data[bk])
    y_max, y_step = nice_scale(max_t * 1.15)

    n_groups = len(k_values)
    n_bars   = len(series)
    group_w  = cw / n_groups
    bar_gap  = 5
    bar_w    = (group_w - bar_gap * (n_bars + 1)) / n_bars

    def bar_x(gi, bi):
        return ML + gi * group_w + bar_gap * (bi + 1) + bar_w * bi

    def val_y(v):
        return MT + ch - ch * v / y_max

    L = [svg_open(W, H)]
    L.append(svgtext(W / 2, 30, title, size=14, bold=True))

    # Y grid + ticks
    n_ticks = round(y_max / y_step)
    for i in range(n_ticks + 1):
        v = i * y_step
        y = val_y(v)
        dash = "4,4" if i > 0 else ""
        col  = GRID_COLOR if i > 0 else AXIS_COLOR
        L.append(svgline(ML, y, ML + cw, y, color=col, dash=dash))
        L.append(svgtext(ML - 8, y + 4, f"{v:.1f}", anchor="end",
                         size=10, color=DIM_COLOR))

    # Y axis label (rotated)
    L.append(svgrottext(16, MT + ch / 2, "time, ms", -90, size=11))

    # X axis
    L.append(svgline(ML, MT + ch, ML + cw, MT + ch))

    # bars
    for gi, K in enumerate(k_values):
        for bi, (bk, color, _) in enumerate(series):
            if K not in data[bk]:
                continue
            t  = data[bk][K]
            bx = bar_x(gi, bi)
            by = val_y(t)
            bh = MT + ch - by
            L.append(svgrect(bx, by, bar_w, bh, color))
            if bh > 16:
                L.append(svgtext(bx + bar_w / 2, by - 4,
                                 f"{t:.1f}", size=9, color=DIM_COLOR))

        gx = ML + gi * group_w + group_w / 2
        L.append(svgtext(gx, MT + ch + 18, f"K = {K}", size=12))

    # legend (two rows if needed)
    leg_x, leg_y = ML, MT + ch + 46
    for idx, (bk, color, label) in enumerate(series):
        row = idx // 2
        col = idx % 2
        lx  = leg_x + col * (cw // 2)
        ly  = leg_y + row * 22
        L.append(svgrect(lx, ly - 10, 13, 13, color, rx=2))
        L.append(svgtext(lx + 18, ly, label, anchor="start",
                         size=11, color=LABEL_COLOR))

    L.append(svg_close())
    with open(out_path, "w") as f:
        f.writelines(L)
    print(f"Written {out_path}")


# ── speedup line chart ─────────────────────────────────────────────────────────
def make_speedup_chart(data, k_values, out_path):
    ref  = "BM_ConvNaive"
    best = "BM_ConvIm2colIntrinsicsGemm"

    if ref not in data or best not in data:
        print(f"  missing data for {out_path}, skipping")
        return

    speedup = {K: data[ref][K] / data[best][K]
               for K in k_values
               if K in data[ref] and K in data[best]}

    if not speedup:
        return

    W, H            = 640, 420
    ML, MR, MT, MB  = 78, 40, 58, 72

    cw = W - ML - MR
    ch = H - MT - MB

    ks            = sorted(speedup)
    max_s         = max(speedup.values())
    y_max, y_step = nice_scale(max_s * 1.2)
    color         = "#f78166"

    def px(K):
        i = ks.index(K)
        return ML + (i * cw / (len(ks) - 1)) if len(ks) > 1 else ML + cw / 2

    def py(s):
        return MT + ch - ch * s / y_max

    L = [svg_open(W, H)]
    L.append(svgtext(W / 2, 30,
                     "Speedup: conv_naive / conv_im2col + intrinsics",
                     size=14, bold=True))

    # Y grid + ticks
    n_ticks = round(y_max / y_step)
    for i in range(n_ticks + 1):
        v = i * y_step
        y = py(v)
        dash = "4,4" if i > 0 else ""
        col  = GRID_COLOR if i > 0 else AXIS_COLOR
        L.append(svgline(ML, y, ML + cw, y, color=col, dash=dash))
        L.append(svgtext(ML - 8, y + 4, f"{v:.1f}×",
                         anchor="end", size=10, color=DIM_COLOR))

    # 1× reference
    if y_max >= 1.0:
        y1 = py(1.0)
        L.append(svgline(ML, y1, ML + cw, y1,
                         color="#58a6ff", width=1, dash="6,3"))
        L.append(svgtext(ML + cw + 6, y1 + 4, "1×",
                         anchor="start", size=10, color="#58a6ff"))

    # Y label
    L.append(svgrottext(16, MT + ch / 2, "speedup", -90, size=11))

    # X axis
    L.append(svgline(ML, MT + ch, ML + cw, MT + ch))

    # line
    L.append(svgpolyline([(px(K), py(speedup[K])) for K in ks], color))

    # dots + labels
    for K in ks:
        x, y = px(K), py(speedup[K])
        L.append(svgcircle(x, y, 5, color))
        L.append(svgtext(x, y - 13, f"{speedup[K]:.1f}×", size=10))
        L.append(svgtext(x, MT + ch + 18, f"K = {K}", size=12))

    L.append(svg_close())
    with open(out_path, "w") as f:
        f.writelines(L)
    print(f"Written {out_path}")


GEMM_COLORS = {
    "BM_GemmNaive":         "#58a6ff",
    "BM_GemmCacheFriendly": "#3fb950",
    "BM_GemmIntrinsics":    "#f78166",
}
GEMM_LABELS = {
    "BM_GemmNaive":         "naive",
    "BM_GemmCacheFriendly": "cache-friendly",
    "BM_GemmIntrinsics":    "intrinsics",
}


# ── GEMM JSON parsing ──────────────────────────────────────────────────────────
def parse_gemm_results(path):
    with open(path) as f:
        raw = json.load(f)

    out     = {}
    has_agg = any(bm.get("aggregate_name") == "median"
                  for bm in raw["benchmarks"])

    for bm in raw["benchmarks"]:
        agg = bm.get("aggregate_name", "")
        if has_agg:
            if agg != "median":
                continue
        else:
            if bm.get("run_type") == "aggregate":
                continue

        name  = bm["name"].removesuffix(f"_{agg}") if agg else bm["name"]
        parts = name.split("/")
        if len(parts) < 2:
            continue
        bname = parts[0]
        N     = int(parts[1])

        t    = bm["real_time"]
        unit = bm.get("time_unit", "ns")
        t_ms = {"ns": t / 1e6, "us": t / 1e3, "ms": t, "s": t * 1e3}[unit]

        out.setdefault(bname, {})[N] = t_ms

    mode = "median" if has_agg else "single run"
    print(f"  mode: {mode}")
    return out


# ── GEMM line chart ────────────────────────────────────────────────────────────
def make_gemm_line_chart(data, out_path):
    series_keys = ["BM_GemmNaive", "BM_GemmCacheFriendly", "BM_GemmIntrinsics"]
    series = [(k, GEMM_COLORS[k], GEMM_LABELS[k]) for k in series_keys if k in data]

    if not series:
        print(f"  no GEMM data for {out_path}, skipping")
        return

    all_sizes = sorted({N for bk, _, _ in series for N in data[bk]})
    if not all_sizes:
        return

    W, H            = 700, 460
    ML, MR, MT, MB  = 78, 40, 58, 80
    cw = W - ML - MR
    ch = H - MT - MB

    max_t         = max(data[bk][N]
                        for bk, _, _ in series
                        for N in all_sizes if N in data[bk])
    y_max, y_step = nice_scale(max_t * 1.15)

    def px(N):
        i = all_sizes.index(N)
        return ML + (i * cw / (len(all_sizes) - 1)) if len(all_sizes) > 1 else ML + cw / 2

    def py(t):
        return MT + ch - ch * t / y_max

    L = [svg_open(W, H)]
    L.append(svgtext(W / 2, 30, "GEMM: time vs matrix size (N×N×N)", size=14, bold=True))

    n_ticks = round(y_max / y_step)
    for i in range(n_ticks + 1):
        v   = i * y_step
        y   = py(v)
        dash = "4,4" if i > 0 else ""
        col  = GRID_COLOR if i > 0 else AXIS_COLOR
        L.append(svgline(ML, y, ML + cw, y, color=col, dash=dash))
        L.append(svgtext(ML - 8, y + 4, f"{v:.1f}", anchor="end", size=10, color=DIM_COLOR))

    L.append(svgrottext(16, MT + ch / 2, "time, ms", -90, size=11))
    L.append(svgline(ML, MT + ch, ML + cw, MT + ch))

    for bk, color, _ in series:
        pts = [(px(N), py(data[bk][N])) for N in all_sizes if N in data[bk]]
        if pts:
            L.append(svgpolyline(pts, color))
            for x, y in pts:
                L.append(svgcircle(x, y, 4, color))

    for N in all_sizes:
        L.append(svgtext(px(N), MT + ch + 18, str(N), size=12))
    L.append(svgtext(W / 2, MT + ch + 40, "matrix size N", size=11, color=DIM_COLOR))

    leg_x, leg_y = ML, MT + ch + 56
    for idx, (bk, color, label) in enumerate(series):
        lx = leg_x + idx * (cw // len(series))
        L.append(svgrect(lx, leg_y - 10, 13, 13, color, rx=2))
        L.append(svgtext(lx + 18, leg_y, label, anchor="start", size=11, color=LABEL_COLOR))

    L.append(svg_close())
    with open(out_path, "w") as f:
        f.writelines(L)
    print(f"Written {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")

def main():
    json_path      = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DOCS_DIR, "results.json")
    gemm_json_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DOCS_DIR, "results_gemm.json")
    out_dir        = DOCS_DIR
    os.makedirs(out_dir, exist_ok=True)

    print(f"Parsing {json_path} ...")
    data     = parse_results(json_path)
    k_values = [3, 5, 7, 9, 11]

    make_bar_chart(
        "GEMM backend comparison (im2col path)",
        data,
        ["BM_ConvIm2colNaiveGemm",
         "BM_ConvIm2colCacheFriendlyGemm",
         "BM_ConvIm2colIntrinsicsGemm"],
        k_values,
        os.path.join(out_dir, "gemm_comparison.svg"),
    )

    make_bar_chart(
        "conv_naive  vs  conv_im2col + intrinsics GEMM",
        data,
        ["BM_ConvNaive",
         "BM_ConvIm2colIntrinsicsGemm"],
        k_values,
        os.path.join(out_dir, "conv_comparison.svg"),
    )

    make_speedup_chart(
        data,
        k_values,
        os.path.join(out_dir, "speedup.svg"),
    )

    if os.path.exists(gemm_json_path):
        print(f"Parsing {gemm_json_path} ...")
        gemm_data = parse_gemm_results(gemm_json_path)
        make_gemm_line_chart(gemm_data, os.path.join(out_dir, "gemm_size.svg"))
    else:
        print(f"  {gemm_json_path} not found, skipping gemm_size.svg")


if __name__ == "__main__":
    main()
