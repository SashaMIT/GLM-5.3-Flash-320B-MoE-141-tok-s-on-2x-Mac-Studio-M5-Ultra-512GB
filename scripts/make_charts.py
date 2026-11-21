#!/usr/bin/env python3
"""Draw the benchmark figures as plain SVG (no dependencies), in a classic
textbook / journal style: white page, boxed black axes, grey fills and hatch
patterns instead of colour, a Key box, and a numbered caption under each figure.

Every number comes from charts/data.json, which copies the round 7 report's
tables exactly (MEASURED, 4-bit lossy). Run:  python3 scripts/make_charts.py
PNGs are the SVGs rendered at 2x in a headless browser.
"""
import json, os, html

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = json.load(open(os.path.join(ROOT, "charts", "data.json")))
OUT = os.path.join(ROOT, "charts")

SANS = "Arial, Helvetica, 'Liberation Sans', sans-serif"
SERIF = "'Times New Roman', Times, 'Liberation Serif', serif"
INK = "#000000"
W, H = 1100, 700

PATTERNS = """<defs>
<pattern id="p_dark" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#6b6b6b"/></pattern>
<pattern id="p_hlines" width="8" height="6" patternUnits="userSpaceOnUse"><rect width="8" height="6" fill="#fff"/><line x1="0" y1="3" x2="8" y2="3" stroke="#000" stroke-width="1"/></pattern>
<pattern id="p_cross" width="10" height="10" patternUnits="userSpaceOnUse"><rect width="10" height="10" fill="#fff"/><path d="M0,0 L10,10 M10,0 L0,10" stroke="#000" stroke-width="1"/></pattern>
</defs>"""


def esc(s):
    return html.escape(str(s), quote=True)


def t(x, y, s, size=15, anchor="start", weight=400, family=SANS, style="normal", fill=INK, rotate=None):
    tr = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" font-style="{style}" '
            f'font-family="{family}" fill="{fill}" text-anchor="{anchor}"{tr}>{esc(s)}</text>')


def page(title, caption_lines, body, label, h=H):
    cap = "".join(t(W / 2, h - 58 + 22 * i, ln, 16 if i == 0 else 14.5, "middle", 400, SERIF,
                    "normal" if i == 0 else "italic") for i, ln in enumerate(caption_lines))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" height="{h}" role="img" aria-label="{esc(label)}">'
            + PATTERNS + f'<rect width="{W}" height="{h}" fill="#ffffff"/>'
            + t(W / 2, 40, title, 21, "middle", 700) + body + cap + "</svg>")


def axes(left, right, top, bottom, vmax, step, ylabel, grid=False, minor=True):
    out = [f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="none" stroke="{INK}" stroke-width="1.2"/>']
    sy = lambda v: bottom - (bottom - top) * v / vmax
    for v in range(0, vmax + 1, step):
        y = sy(v)
        if grid and 0 < v < vmax:
            out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#000" stroke-width="0.6" stroke-dasharray="3 3"/>')
        out.append(f'<line x1="{left-7}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="{INK}" stroke-width="1.2"/>')
        out.append(t(left - 12, y + 5, v, 15, "end"))
    if minor:
        for v in range(step // 2, vmax, step):
            y = sy(v)
            out.append(f'<line x1="{left-4}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="{INK}" stroke-width="1"/>')
    out.append(t(left - 62, (top + bottom) / 2, ylabel, 17, "middle", 700, rotate=-90))
    return "".join(out), sy


def bar_key(cx, y, items):
    w = 20 + sum(34 + 8.2 * len(lbl) for lbl, _ in items)
    out = [f'<rect x="{cx - w/2:.1f}" y="{y}" width="{w:.1f}" height="34" fill="#fff" stroke="{INK}" stroke-width="1"/>']
    x = cx - w / 2 + 14
    for lbl, fill in items:
        out.append(f'<rect x="{x:.1f}" y="{y+10}" width="16" height="14" fill="{fill}" stroke="{INK}" stroke-width="1"/>')
        out.append(t(x + 22, y + 23, lbl, 15))
        x += 34 + 8.2 * len(lbl)
    return "".join(out)


def line_key(x, y, items, title="Key"):
    w, h = 210, 26 + 28 * len(items) + 14
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#fff" stroke="{INK}" stroke-width="1"/>',
           t(x + w / 2, y + 22, title, 16, "middle", 400, SERIF, "italic")]
    yy = y + 50
    for lbl, mk, dash in items:
        out.append(f'<line x1="{x+14}" y1="{yy-5}" x2="{x+64}" y2="{yy-5}" stroke="{INK}" stroke-width="1.6" stroke-dasharray="{dash}"/>')
        out.append(mk(x + 39, yy - 5))
        out.append(t(x + 74, yy, lbl, 15, family=SERIF))
        yy += 28
    return "".join(out)


def bar(x, y0, w, h, fill):
    return f'<rect x="{x:.1f}" y="{y0-h:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{INK}" stroke-width="1"/>'


def chart_glance():
    d = DATA["glance"]; cards = d["cards"]
    cols = [("Job", 290), ("Engine", 190), ("Set-up", 150), ("Result", 150), ("Note", 250)]
    total_w = sum(c[1] for c in cols)
    left = (W - total_w) / 2; top = 120; rh = 50; nrows = len(cards) + 1
    out = [t(W / 2, top - 24, "Table 1", 19, "middle", 400, SERIF),
           f'<rect x="{left}" y="{top}" width="{total_w}" height="{rh*nrows}" fill="#fff" stroke="{INK}" stroke-width="1.4"/>',
           f'<line x1="{left}" y1="{top+rh}" x2="{left+total_w}" y2="{top+rh}" stroke="{INK}" stroke-width="1.4"/>']
    x = left
    for name, w in cols:
        out.append(t(x + w / 2, top + 32, name, 17, "middle", 700, SERIF))
        x += w
        if x < left + total_w - 1:
            out.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top+rh*nrows}" stroke="{INK}" stroke-width="1"/>')
    for i, c in enumerate(cards):
        y = top + rh * (i + 1)
        if i:
            out.append(f'<line x1="{left}" y1="{y}" x2="{left+total_w}" y2="{y}" stroke="{INK}" stroke-width="0.8"/>')
        vals = [c["job"], c["who"], c["how"], f'{c["value"]} {c["unit"]}', c["note"].replace(" · ", ", ")]
        x = left
        for j, (name, w) in enumerate(cols):
            mid = j in (2, 3)
            out.append(t(x + w / 2 if mid else x + 12, y + 31, vals[j], 13.5 if j == 4 else 15.5,
                         "middle" if mid else "start", 700 if j == 3 else 400, SERIF))
            x += w
    return page("The Right Engine for Each Job",
                ["Table 1. Same TensorFold model pack, same two Mac Studios (M5 Ultra, 512 GB), same day.",
                 "Single user: median of 144 requests. Many users: median of 3 repeats. All MEASURED, 4-bit, lossy."],
                "".join(out), d["title"], h=560)


def chart_single_user():
    d = DATA["single_user"]; rows = d["rows"]
    groups = [("GLM-Tandem", "2 Macs", rows[0]["e2e"], rows[1]["e2e"]),
              ("TensorFold 1.2.10", "1 Mac", rows[2]["e2e"], rows[3]["e2e"]),
              ("mlx-vlm 0.8.0", "1 Mac (no drafting)", None, rows[4]["e2e"])]
    left, right, top, bottom = 140, 1040, 120, 520
    ax, sy = axes(left, right, top, bottom, 140, 20, "Speed (tokens per second)")
    out = [ax]
    slot = (right - left) / len(groups); bw = 74
    for i, (eng, macs, on, off) in enumerate(groups):
        cx = left + slot * (i + 0.5)
        pairs = [(on, "url(#p_dark)"), (off, "url(#p_hlines)")] if on is not None else [(off, "url(#p_hlines)")]
        for j, (v, fill) in enumerate(pairs):
            x = (cx - bw - 4 + j * (bw + 8)) if len(pairs) == 2 else cx - bw / 2
            out.append(bar(x, bottom, bw, bottom - sy(v), fill))
            out.append(t(x + bw / 2, sy(v) - 8, f"{v:.1f}", 15, "middle", 700))
        out.append(t(cx, bottom + 26, eng, 16, "middle"))
        out.append(t(cx, bottom + 46, macs, 15, "middle", fill="#333"))
    out.append(t((left + right) / 2, bottom + 80, "Engine", 17, "middle", 700))
    out.append(bar_key((left + right) / 2, 66, [("Drafting on", "url(#p_dark)"), ("Drafting off", "url(#p_hlines)")]))
    return page("Single-User Speed by Engine",
                ["Figure 1. Median end-to-end speed, 144 timed requests per set-up over 3 fresh restarts (24 prompts, T = 0).",
                 "E2E tok/s = n_out / (t_last − t0). Two Macs ÷ best one Mac = 1.47 [1.43, 1.49], 95% bootstrap CI."],
                "".join(out), d["title"])


def marker(kind):
    if kind == "circle":
        return lambda x, y: f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#fff" stroke="{INK}" stroke-width="1.5"/>'
    if kind == "square":
        return lambda x, y: f'<rect x="{x-5:.1f}" y="{y-5:.1f}" width="10" height="10" fill="{INK}"/>'
    return lambda x, y: f'<path d="M{x:.1f},{y-6:.1f} L{x+6:.1f},{y+5:.1f} L{x-6:.1f},{y+5:.1f} Z" fill="#fff" stroke="{INK}" stroke-width="1.5"/>'


def chart_many_users():
    d = DATA["many_users"]
    left, right, top, bottom = 140, 1040, 80, 520
    xmax = 64
    ax, sy = axes(left, right, top, bottom, 350, 50, "Total speed (tokens per second)", grid=True)
    sx = lambda u: left + (right - left) * u / xmax
    out = [ax]
    for u in [0, 8, 16, 24, 32, 40, 48, 56, 64]:
        x = sx(u)
        if 0 < u < xmax:
            out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#000" stroke-width="0.6" stroke-dasharray="3 3"/>')
        out.append(f'<line x1="{x:.1f}" y1="{bottom}" x2="{x:.1f}" y2="{bottom+7}" stroke="{INK}" stroke-width="1.2"/>')
        out.append(t(x, bottom + 26, u, 15, "middle"))
    out.append(t((left + right) / 2, bottom + 58, "Number of users at the same time (short prompts)", 17, "middle", 700))
    style = {"tandem": ("circle", "7 3 2 3", "GLM-Tandem pair"), "tensorfold": ("square", "", "TensorFold, one per Mac"),
             "mlxvlm": ("triangle", "", "mlx-vlm, one per Mac")}
    bye = {s["engine"]: s for s in d["series"]}
    keyitems = []
    for e in ["tandem", "tensorfold", "mlxvlm"]:
        s = bye[e]; mk, dash, nm = style[e]
        pts = [(sx(u), sy(v)) for u, v in s["points"]]
        out.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{INK}" stroke-width="1.6" stroke-dasharray="{dash}"/>')
        out += [marker(mk)(x, y) for x, y in pts]
        keyitems.append((nm, marker(mk), dash))
    out.append(line_key(right - 250, bottom - 150, keyitems))
    return page("Total Speed as More People Share Two Mac Studios",
                ["Figure 2. Total output speed for all users together; each point is the median of 3 repeats.",
                 "TensorFold (one per Mac) leads at 8, 16 and 32 users (level with GLM-Tandem at 2); two mlx-vlm servers reach the highest total at 48–64."],
                "".join(out), d["title"])


def chart_fairness():
    d = DATA["fairness"]; rows = d["rows"]
    names = {"tensorfold": ("TensorFold 1.2.10", "1 Mac"), "tandem": ("GLM-Tandem", "2 Macs"), "mlxvlm": ("mlx-vlm 0.8.0", "1 Mac")}
    left, right, top, bottom = 140, 1040, 120, 520
    ax, sy = axes(left, right, top, bottom, 40, 5, "Wait for first token (seconds)", minor=False)
    out = [ax]
    slot = (right - left) / len(rows); bw = 74
    for i, r in enumerate(rows):
        cx = left + slot * (i + 0.5)
        for j, (v, fill) in enumerate([(r["p50"], "url(#p_dark)"), (r["max"], "url(#p_cross)")]):
            x = cx - bw - 4 + j * (bw + 8)
            out.append(bar(x, bottom, bw, bottom - sy(v), fill))
            out.append(t(x + bw / 2, sy(v) - 8, f"{v:.1f}", 15, "middle", 700))
        n1, n2 = names[r["engine"]]
        out.append(t(cx, bottom + 26, n1, 16, "middle"))
        out.append(t(cx, bottom + 46, n2 + (" *" if r.get("note") else ""), 15, "middle", fill="#333"))
    out.append(t((left + right) / 2, bottom + 80, "Engine", 17, "middle", 700))
    out.append(bar_key((left + right) / 2, 66, [("Median short request", "url(#p_dark)"), ("Slowest short request", "url(#p_cross)")]))
    return page("Waiting Time for Short Requests Beside a 32k-Token Document",
                ["Figure 4. One 32k prompt plus 8 short prompts staggered 0.25 s apart; fresh server, no cache. TTFT = t_first − t0. Lower is better.",
                 "* mlx-vlm reached the 80% memory limit during this test."],
                "".join(out), d["title"])


def chart_gsm8k():
    d = DATA["gsm8k"]; rows = d["rows"]
    left, right, top, bottom = 140, 1040, 80, 490
    ax, sy = axes(left, right, top, bottom, 140, 20, "Speed (tokens per second)")
    out = [ax]
    slot = (right - left) / len(rows); bw = 150
    fills = ["url(#p_hlines)", "url(#p_dark)"]
    for i, r in enumerate(rows):
        cx = left + slot * (i + 0.5)
        out.append(bar(cx - bw / 2, bottom, bw, bottom - sy(r["e2e"]), fills[i]))
        out.append(t(cx, sy(r["e2e"]) - 10, f'{r["e2e"]:.1f}', 16, "middle", 700))
        out.append(t(cx, bottom + 26, r["name"], 16, "middle"))
        out.append(t(cx, bottom + 48, f'Accuracy {r["vote"]} by majority vote; {r["samples"]} single samples', 14.5, "middle", 400, SERIF, "italic"))
    out.append(t((left + right) / 2, bottom + 84, "GLM-Tandem on 2 Macs, GSM8K (first 100 test questions)", 17, "middle", 700))
    return page("Maths Accuracy and Speed With and Without Drafting",
                ["Figure 3. Five samples per question at temperature 0.6; median end-to-end speed.",
                 "Paired exact McNemar test on majority votes: p = 1.0000 (n = 100). Majority-vote accuracy is 100/100 in both."],
                "".join(out), d["title"])


def chart_speed():
    d = DATA["speed"]; rows = d["rows"]
    panels = [("(a) Prompt reading (prefill)", "prefill", 2500, 500, 130, 520),
              ("(b) Output (decode)", "decode", 160, 20, 680, 1060)]
    out = []
    top, bottom = 110, 500
    for title, key, vmax, step, left, right in panels:
        out.append(f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="none" stroke="{INK}" stroke-width="1.2"/>')
        sy = lambda v, vmax=vmax: bottom - (bottom - top) * v / vmax
        for v in range(0, vmax + 1, step):
            y = sy(v)
            out.append(f'<line x1="{left-7}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="{INK}" stroke-width="1.2"/>')
            out.append(t(left - 12, y + 5, f"{v:,}", 14, "end"))
        out.append(t((left + right) / 2, top - 14, title, 17, "middle", 700))
        out.append(t(left - 72, (top + bottom) / 2, "tokens per second", 15, "middle", 700, rotate=-90))
        slot = (right - left) / len(rows); bw = 70
        for i, r in enumerate(rows):
            cx = left + slot * (i + 0.5); v = r[key]
            if v is None:
                out.append(t(cx, bottom - 20, "not measured", 13, "middle", 400, SERIF, "italic"))
            else:
                out.append(bar(cx - bw / 2, bottom, bw, bottom - sy(v), "url(#p_dark)" if key == "decode" else "url(#p_hlines)"))
                out.append(t(cx, sy(v) - 8, f"{v:,.1f}", 14.5, "middle", 700))
            out.append(t(cx, bottom + 24, r["engine"], 14.5, "middle"))
            out.append(t(cx, bottom + 43, r["setup"], 13.5, "middle", fill="#333"))
    return page("Prompt Reading and Output Speed, GLM-5.3-Flash (320B MoE, 4-bit)",
                ["Figure 0. (a) Prefill on one long prompt (GLM-Tandem 53,990 tokens, TensorFold 31,995 tokens), fresh server, no cache.",
                 "(b) Median output (decode) speed, single user, 144 requests per set-up (Round 7). The two panels use separate scales."],
                "".join(out), d["title"], h=640)


for name, fn in [("00-speed", chart_speed), ("0-at-a-glance", chart_glance), ("1-single-user-speed", chart_single_user),
                 ("2-many-users", chart_many_users), ("3-long-prompt-fairness", chart_fairness),
                 ("4-gsm8k-accuracy-speed", chart_gsm8k)]:
    with open(os.path.join(OUT, name + ".svg"), "w") as f:
        f.write(fn())
    print("wrote charts/" + name + ".svg")
