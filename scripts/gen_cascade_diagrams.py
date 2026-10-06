"""Generate the /traffic cascade-demo architecture diagrams into docs/diagrams/.

Three self-contained HTML + inline-SVG pages, drawn to the diagram-design skill's
rules (github.com/cathrynlavery/diagram-design): default skin, slide-16x9 preset
(viewBox 1280x720, presentation type ramp), 4px grid, orthogonal connectors.

    python3 scripts/gen_cascade_diagrams.py [output_dir]

Text widths and the 14-character arrow-label limit are asserted while
building, so an over-long label fails here instead of overflowing its box.
"""
import math
import os
import sys
from html import escape

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "docs", "diagrams")

# --- default skin tokens (style-guide.md) ---
PAPER, INK, MUTED, SOFT = "#f5f5f5", "#2d3142", "#4f5d75", "#7a8399"
ACCENT, LINK = "#eb6c36", "#2e5aa8"
KIND = {  # node type -> (fill, stroke, tag stroke, tag text)
    "focal": ("rgba(235,108,54,0.08)", ACCENT, "rgba(235,108,54,0.50)", ACCENT),
    "backend": ("#ffffff", INK, "rgba(45,49,66,0.40)", INK),
    "store": ("rgba(45,49,66,0.05)", MUTED, "rgba(79,93,117,0.50)", MUTED),
    "input": ("rgba(79,93,117,0.10)", SOFT, "rgba(122,131,153,0.40)", SOFT),
}
# 24x24 stroked icons: Tabler Icons (MIT), as shipped in the diagram-design skill's
# primitive-icons.md (database, users, desktop, terminal); camera, movie and cpu are
# Tabler icons in the same style, which the skill's set doesn't include.
ICONS = {
    "camera": ["M5 7h1a2 2 0 0 0 2 -2a1 1 0 0 1 1 -1h6a1 1 0 0 1 1 1a2 2 0 0 0 2 2h1a2 2 0 0 1 2 2v9a2 2 0 0 1 -2 2h-14a2 2 0 0 1 -2 -2v-9a2 2 0 0 1 2 -2",
               "M9 13a3 3 0 1 0 6 0a3 3 0 0 0 -6 0"],
    "movie": ["M4 6a2 2 0 0 1 2 -2h12a2 2 0 0 1 2 2v12a2 2 0 0 1 -2 2h-12a2 2 0 0 1 -2 -2z", "M8 4v16", "M16 4v16",
              "M4 8h4", "M4 16h4", "M4 12h16", "M16 8h4", "M16 16h4"],
    "cpu": ["M5 6a1 1 0 0 1 1 -1h12a1 1 0 0 1 1 1v12a1 1 0 0 1 -1 1h-12a1 1 0 0 1 -1 -1z", "M9 9h6v6h-6z",
            "M3 10h2", "M3 14h2", "M10 3v2", "M14 3v2", "M21 10h-2", "M21 14h-2", "M14 21v-2", "M10 21v-2"],
    "crop": ["M8 5v10a1 1 0 0 0 1 1h10", "M5 8h10a1 1 0 0 1 1 1v10"],
    "database": ["M4 6a8 3 0 1 0 16 0a8 3 0 1 0 -16 0", "M4 6v6a8 3 0 0 0 16 0v-6", "M4 12v6a8 3 0 0 0 16 0v-6"],
    "users": ["M5 7a4 4 0 1 0 8 0a4 4 0 1 0 -8 0", "M3 21v-2a4 4 0 0 1 4 -4h4a4 4 0 0 1 4 4v2", "M16 3.13a4 4 0 0 1 0 7.75", "M21 21v-2a4 4 0 0 0 -3 -3.85"],
    "desktop": ["M3 5a1 1 0 0 1 1 -1h16a1 1 0 0 1 1 1v10a1 1 0 0 1 -1 1h-16a1 1 0 0 1 -1 -1v-10", "M7 20h10", "M9 16v4", "M15 16v4"],
    "terminal": ["M5 7l5 5l-5 5", "M12 19l7 0"],
}


def icon(name, x, y, size, color):
    paths = "".join(f'<path d="{d}"/>' for d in ICONS[name])
    return (f'<g transform="translate({x},{y}) scale({size / 24:g})" fill="none" stroke="{color}" stroke-width="1.5" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths}</g>')


# presentation ramp (output-spec.md)
NAME, SUB, ALABEL, TAG, LEGEND = 16, 12, 12, 8, 11
SANS, MONO = "'Geist', sans-serif", "'Geist Mono', monospace"


def r4(v):
    return int(math.ceil(v / 4.0) * 4)


def w_sans(s, px):
    return len(s) * 0.60 * px


def w_mono(s, px, track=0.0):
    return len(s) * (0.62 + track) * px


class SVG:
    def __init__(self, slug, title, desc, h=720):
        self.slug, self.title, self.desc, self.h = slug, title, desc, h
        self.zones, self.arrows, self.labels, self.nodes, self.extra = [], [], [], [], []

    # ---------- primitives ----------
    def zone(self, x, y, w, h, label):
        lw = r4(w_mono(label, TAG, 0.14) + 12)
        cx = x + w // 2
        self.zones.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="rgba(45,49,66,0.02)" stroke="rgba(45,49,66,0.10)" stroke-width="0.8"/>'
            f'<rect x="{cx - lw // 2}" y="{y + 4}" width="{lw}" height="12" rx="2" fill="{PAPER}"/>'
            f'<text x="{cx}" y="{y + 13}" fill="rgba(45,49,66,0.45)" font-size="{TAG}" font-family="{MONO}" text-anchor="middle" letter-spacing="0.14em">{escape(label)}</text>'
        )

    def arrow(self, d, kind="muted"):
        stroke, marker, extra = {
            "muted": (MUTED, "arrow", 'stroke-width="1.2"'),
            "link": (LINK, "arrow-link", 'stroke-width="1.2"'),
            "accent": (ACCENT, "arrow-accent", 'stroke-width="1.4"'),
            "dashed": (MUTED, "arrow", 'stroke-width="1" stroke-dasharray="4,3"'),
            "dashed-link": (LINK, "arrow-link", 'stroke-width="1" stroke-dasharray="4,3"'),
            "async": (MUTED, "arrow-open", 'stroke-width="1" stroke-dasharray="4,3"'),
            "plain": (MUTED, None, 'stroke-width="1.2"'),
        }[kind]
        m = f' marker-end="url(#{marker})"' if marker else ""
        self.arrows.append(f'<path d="{d}" fill="none" stroke="{stroke}" {extra}{m}/>')

    def label(self, cx, top, text, color=SOFT, anchor="middle"):
        """Arrow label: opaque mask 16 tall starting at `top`, text 12px mono."""
        assert len(text) <= 14, ("arrow label over 14 chars", text)
        w = r4(w_mono(text, ALABEL, 0.06) + 12)
        x = cx - w // 2 if anchor == "middle" else cx
        tx = cx if anchor == "middle" else cx + w // 2
        self.labels.append(
            f'<rect x="{x}" y="{top}" width="{w}" height="16" rx="2" fill="{PAPER}"/>'
            f'<text x="{tx}" y="{top + 12}" fill="{color}" font-size="{ALABEL}" font-family="{MONO}" text-anchor="middle" letter-spacing="0.06em">{escape(text)}</text>'
        )
        return x, x + w

    def node(self, x, y, w, h, kind, tag, name, subs=(), num=None, header=False):
        fill, stroke, tstroke, ttext = KIND[kind]
        assert w_sans(name, NAME) <= w - 16, (name, w_sans(name, NAME), w)
        for s in subs:
            assert w_mono(s, SUB) <= w - 16, (s, w_mono(s, SUB), w)
        tw = r4(w_mono(tag, TAG, 0.08) + 12)
        cx = x + w // 2
        p = [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{PAPER}"/>',
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1"/>',
            f'<rect x="{x + 8}" y="{y + 8}" width="{tw}" height="12" rx="2" fill="transparent" stroke="{tstroke}" stroke-width="0.8"/>',
            f'<text x="{x + 8 + tw / 2:g}" y="{y + 17}" fill="{ttext}" font-size="{TAG}" font-family="{MONO}" text-anchor="middle" letter-spacing="0.08em">{escape(tag)}</text>',
        ]
        if num:
            op = "0.10" if kind == "focal" else "0.06"
            col = f"rgba(235,108,54,{op})" if kind == "focal" else f"rgba(45,49,66,{op})"
            p.append(f'<text x="{x + w - 8}" y="{y + 32}" fill="{col}" font-size="24" font-weight="600" font-family="{MONO}" text-anchor="end">{num}</text>')
        # name + sublabels, block centred in the space under the tag (or at the top for a header node)
        block = NAME + 6 + (len(subs) - 1) * 18 + SUB if subs else NAME
        top = y + 24 if header else y + 20 + (h - 20 - block) / 2
        p.append(f'<text x="{cx}" y="{top + NAME - 3:g}" fill="{INK}" font-size="{NAME}" font-weight="600" font-family="{SANS}" text-anchor="middle">{escape(name)}</text>')
        for i, s in enumerate(subs):
            p.append(f'<text x="{cx}" y="{top + NAME + 6 + i * 18 + SUB - 2:g}" fill="{MUTED}" font-size="{SUB}" font-family="{MONO}" text-anchor="middle">{escape(s)}</text>')
        self.nodes.append("".join(p))

    def inode(self, x, y, w, h, kind, icon_name, name, subs=(), step=None):
        """Node with a left gutter (step badge + icon) and left-aligned text."""
        fill, stroke, _, _ = KIND[kind]
        tx = x + 64
        assert w_sans(name, NAME) <= w - 64 - 12, (name, w_sans(name, NAME), w)
        for s_ in subs:
            assert w_mono(s_, SUB) <= w - 64 - 12, (s_, w_mono(s_, SUB), w)
        p = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{PAPER}"/>',
             f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1"/>']
        if step:
            assert h >= 96, ("stepped node needs h>=96 for badge + icon", name)
            p.append(f'<rect x="{x + 16}" y="{y + 16}" width="28" height="28" rx="4" fill="{INK}"/>'
                     f'<text x="{x + 30}" y="{y + 35}" fill="{PAPER}" font-size="{SUB}" font-weight="600" font-family="{MONO}" text-anchor="middle">{step}</text>')
            p.append(icon(icon_name, x + 14, y + 52, 32, stroke if kind == "focal" else MUTED))
        else:
            p.append(icon(icon_name, x + 14, y + (h - 32) // 2, 32, stroke if kind == "focal" else MUTED))
        p.append(f'<text x="{tx}" y="{y + 34}" fill="{INK}" font-size="{NAME}" font-weight="600" font-family="{SANS}">{escape(name)}</text>')
        for i, s_ in enumerate(subs):
            p.append(f'<text x="{tx}" y="{y + 56 + i * 18}" fill="{MUTED}" font-size="{SUB}" font-family="{MONO}">{escape(s_)}</text>')
        self.nodes.append("".join(p))

    def raw(self, s):
        self.extra.append(s)

    def legend(self, items, y=652):
        out = [f'<line x1="40" y1="{y - 8}" x2="1240" y2="{y - 8}" stroke="rgba(45,49,66,0.10)" stroke-width="0.8"/>',
               f'<text x="40" y="{y + 8}" fill="{MUTED}" font-size="8" font-family="{MONO}" letter-spacing="0.18em">LEGEND</text>']
        x = 40
        iy = y + 24
        for kind, text in items:
            if kind in KIND:
                f, s, _, _ = KIND[kind]
                out.append(f'<rect x="{x}" y="{iy}" width="16" height="12" rx="2" fill="{f}" stroke="{s}" stroke-width="1"/>')
            else:
                st = {"muted": (MUTED, "arrow", "1.2", ""), "link": (LINK, "arrow-link", "1.2", ""),
                      "accent": (ACCENT, "arrow-accent", "1.4", ""), "dashed": (MUTED, "arrow", "1", ' stroke-dasharray="4,3"'),
                      "async": (MUTED, "arrow-open", "1", ' stroke-dasharray="4,3"')}[kind]
                out.append(f'<line x1="{x}" y1="{iy + 6}" x2="{x + 24}" y2="{iy + 6}" stroke="{st[0]}" stroke-width="{st[2]}"{st[3]} marker-end="url(#{st[1]})"/>')
            out.append(f'<text x="{x + 32}" y="{iy + 10}" fill="{MUTED}" font-size="{LEGEND}" font-family="{SANS}">{escape(text)}</text>')
            x += 32 + r4(w_sans(text, LEGEND)) + 36
        assert x - 36 <= 1240, ("legend too wide", x)
        self.extra.append("".join(out))

    def render(self):
        defs = (
            f'<marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{MUTED}"/></marker>'
            f'<marker id="arrow-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{ACCENT}"/></marker>'
            f'<marker id="arrow-link" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{LINK}"/></marker>'
            f'<marker id="arrow-open" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polyline points="0 0, 8 3, 0 6" fill="none" stroke="{MUTED}" stroke-width="1.2"/></marker>'
        )
        body = "\n".join(self.zones + self.arrows + self.labels + self.nodes + self.extra)
        return (
            f'<svg viewBox="0 0 1280 {self.h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{self.slug}-title {self.slug}-desc">\n'
            f'<title id="{self.slug}-title">{escape(self.title)}</title>\n'
            f'<desc id="{self.slug}-desc">{escape(self.desc)}</desc>\n'
            f"<defs>{defs}</defs>\n"
            f'<rect width="100%" height="100%" fill="{PAPER}"/>\n{body}\n</svg>'
        )


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --color-paper: #f5f5f5; --color-ink: #2d3142; --color-muted: #4f5d75; --color-accent: #eb6c36;
      --font-sans: 'Geist', system-ui, sans-serif;
      --font-serif: 'Instrument Serif', serif;
      --font-mono: 'Geist Mono', ui-monospace, monospace;
    }}
    body {{ font-family: var(--font-sans); background: var(--color-paper); color: var(--color-ink);
           min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 2.5rem 2rem; }}
    .frame {{ max-width: 1280px; width: 100%; min-width: 0; }}
    .diagram-container {{ width: 100%; overflow-x: auto; }}
    .eyebrow {{ font-family: var(--font-mono); font-size: 0.66rem; font-weight: 500; letter-spacing: 0.18em;
               text-transform: uppercase; color: var(--color-muted); margin-bottom: 0.5rem; }}
    h1 {{ font-family: var(--font-serif); font-size: clamp(1.75rem, 2.6vw + 0.75rem, 2.5rem); font-weight: 400;
          letter-spacing: -0.02em; line-height: 1.15; color: var(--color-ink); margin-bottom: 0.4rem; }}
    .subtitle {{ color: var(--color-muted); font-size: 0.95rem; margin-bottom: 1.25rem; }}
    .subtitle code {{ font-family: var(--font-mono); font-size: 0.85em; }}
    .nav {{ margin-top: 1rem; font-family: var(--font-mono); font-size: 0.7rem; letter-spacing: 0.08em; color: var(--color-muted); }}
    .nav a {{ color: #2e5aa8; text-decoration: none; margin-right: 1.25rem; }}
    svg {{ width: 100%; min-width: 1280px; display: block; }}
    @media print {{
      .diagram-container {{ overflow-x: visible; }}
      svg {{ min-width: 0; }}
    }}
  </style>
</head>
<body>
  <div class="frame">
    <p class="eyebrow">{eyebrow}</p>
    <h1>{h1}</h1>
    <p class="subtitle">{subtitle}</p>
    <div class="diagram-container">
{svg}
    </div>
    <p class="nav"><a href="cascade-how-it-works.html">How it works</a><a href="cascade-overview.html">1 · System overview</a><a href="cascade-pipeline.html">2 · Per-frame pipeline</a><a href="cascade-describe.html">3 · Click-to-Describe</a></p>
  </div>
</body>
</html>
"""


def write(fname, svg, eyebrow, h1, subtitle):
    with open(f"{OUT}/{fname}", "w") as f:
        f.write(PAGE.format(title=escape(svg.title), eyebrow=escape(eyebrow), h1=escape(h1), subtitle=subtitle, svg=svg.render()))


# =====================================================================
# 1. System overview (Architecture)
# =====================================================================
d = SVG("cascade-overview", "Object detection to VLM cascade on the AFE-A503",
        "Architecture of the on-device traffic demo: a looped video is decoded, vehicles are detected by YOLO26n on the "
        "Hexagon NPU, tracked inside an entry zone and streamed to the browser, and a clicked vehicle's crop is described "
        "by Qwen3-VL-4B running in GenieX on the same NPU.")
d.zone(40, 96, 872, 328, "WEBUI SERVER · PYTHON · :8090")
d.zone(40, 464, 592, 160, "GENIEX · LOCAL VLM RUNTIME")
d.zone(960, 96, 280, 528, "BROWSER · FIREFOX")

# arrows (before nodes)
d.arrow("M 248,184 H 320")                                   # video -> yolo
d.arrow("M 560,184 H 632")                                   # yolo -> tracker
d.arrow("M 888,168 H 992", "link")                           # tracker -> browser (WebRTC)
d.arrow("M 888,200 H 992", "link")                           # tracker -> browser (WS detections)
d.arrow("M 760,232 V 328")                                   # tracker -> describe (best crop)
d.arrow("M 440,232 V 360 Q 440,368 432,368 H 264", "dashed")  # yolo -> monitor (NPU busy time)
d.arrow("M 992,352 H 888", "link")                           # browser -> describe (request)
d.arrow("M 888,384 H 992", "dashed-link")                    # describe -> browser (caption)
# monitor -> browser: hop over the describe->proxy vertical at x=736
d.arrow("M 164,408 V 440 Q 164,448 172,448 H 728 a 8,8 0 0,1 16,0 H 992", "link")
d.arrow("M 736,408 V 536 Q 736,544 728,544 H 560", "link")   # describe -> proxy (HTTP)
d.arrow("M 320,544 H 264", "link")                           # proxy -> qwen

# labels (masks clear of nodes; 6px+ gap from strokes)
d.label(284, 160, "FRAMES")
d.label(596, 160, "BOXES")
d.label(940, 146, "WEBRTC", LINK)
d.label(940, 206, "DETECTIONS", LINK)
d.label(768, 272, "BEST CROP", SOFT, anchor="left")
d.label(352, 346, "NPU TIME", SOFT)
d.label(940, 330, "DESCRIBE", LINK)
d.label(940, 390, "CAPTION", LINK)
d.label(452, 426, "STATS · 4 HZ", LINK)
d.label(648, 520, "HTTP /V1/CHAT", LINK)

# nodes
d.node(64, 144, 184, 80, "store", "FILE", "Demo video", ["looped · PyAV"], "01")
d.node(320, 144, 240, 88, "focal", "NPU", "YOLO26n detector", ["LiteRT + QNN HTP", "640×640 · ~15 ms"], "02")
d.node(632, 144, 256, 88, "backend", "CPU", "Tracker + entry zone", ["IoU match", "best crop per vehicle"], "03")
d.node(64, 328, 200, 80, "backend", "MON", "System monitor", ["CPU GPU NPU · °C"])
d.node(632, 328, 256, 80, "backend", "API", "Describe service", ["crops · lock · cooldown"], "04")
d.node(320, 504, 240, 80, "backend", "PROXY", "GenieX proxy", [":18182 · strips tag"])
d.node(64, 504, 200, 80, "focal", "NPU", "Qwen3-VL-4B", ["W4A16 · :18181"], "05")
d.node(992, 144, 216, 456, "input", "UI", "/traffic page", ["HDMI · kiosk"], header=True)
# browser node contents: panels as inner rows
rows = [("Live video", "boxes + zone"), ("Detected vehicles", "click → Describe"),
        ("System stats", "CPU · RAM · NPU"), ("Settings", "video · models")]
for i, (a, b) in enumerate(rows):
    y = 232 + i * 88
    d.raw(f'<rect x="1008" y="{y}" width="184" height="72" rx="4" fill="#ffffff" stroke="rgba(45,49,66,0.14)" stroke-width="0.8"/>'
          f'<text x="1100" y="{y + 30}" fill="{INK}" font-size="14" font-weight="600" font-family="{SANS}" text-anchor="middle">{escape(a)}</text>'
          f'<text x="1100" y="{y + 52}" fill="{MUTED}" font-size="11" font-family="{MONO}" text-anchor="middle">{escape(b)}</text>')
d.legend([("focal", "AI model on the NPU"), ("backend", "Service"), ("store", "File"), ("input", "User UI"),
          ("muted", "In-process"), ("link", "WebRTC · WS · HTTP"), ("dashed", "Return / passive")])
write("cascade-overview.html", d, "Architecture · Advantech AFE-A503 · Qualcomm QCS9075 (IQ9)",
      "Object detection → VLM cascade, on one board",
      "YOLO26n finds the vehicles; Qwen3-VL-4B describes the one you click. Both run on the Hexagon NPU — nothing leaves the device.")

# =====================================================================
# 2. Per-frame pipeline (Architecture)
# =====================================================================
d = SVG("cascade-pipeline", "Per-frame detection pipeline",
        "Each video frame is decoded on the CPU, letterboxed to 640x640, run through YOLO26n on the Hexagon NPU, filtered "
        "by confidence, NMS, vehicle class and entry zone, drawn and streamed to the page, while a tracker logs each "
        "vehicle once to the panel and keeps its best crop for Describe.")
d.zone(424, 144, 192, 144, "HEXAGON NPU")
d.zone(1024, 144, 192, 368, "BROWSER")
top = [(40, "store", "CPU", "Decode frame", ["PyAV · CPU", "latest frame only"], "01"),
       (240, "backend", "CPU", "Letterbox", ["640×640 pad", "RGB float 0–1"], "02"),
       (440, "focal", "NPU", "YOLO26n", ["LiteRT + QNN", "~15 ms / frame"], "03"),
       (640, "backend", "CPU", "Filter + zone", ["conf≥0.45 · NMS", "vehicles in zone"], "04"),
       (840, "backend", "CPU", "Draw boxes", ["OpenCV", "+ zone outline"], "05"),
       (1040, "input", "UI", "Live video", ["WebRTC · aiortc", "<video> on page"], "06")]
for i in range(5):
    x2 = top[i + 1][0]
    d.arrow(f"M {x2 - 40},224 H {x2}", "link" if i == 4 else "muted")
d.arrow("M 720,272 V 400")                     # filter -> tracker
d.arrow("M 640,448 H 600")                     # tracker -> store
d.arrow("M 800,448 H 1040", "link")            # tracker -> panel
d.label(728, 320, "DETECTIONS", SOFT, anchor="left")
d.label(920, 424, "NEW VEHICLE", LINK)
for x, kind, tag, name, subs, num in top:
    d.node(x, 176, 160, 96, kind, tag, name, subs, num)
d.node(440, 400, 160, 96, "store", "MEM", "Crop store", ["200 best crops", "→ Describe"])
d.node(640, 400, 160, 96, "backend", "CPU", "Tracker", ["IoU 0.3 · 3 hits", "log after peak"], "07")
d.node(1040, 400, 160, 96, "input", "UI", "Vehicles panel", ["160 px thumbs", "click → Describe"])
d.raw(f'<text x="40" y="560" fill="{MUTED}" font-size="14" font-style="italic" font-family="\'Instrument Serif\', serif">'
      f'<tspan x="40" dy="0">Only the ~15 ms detection step runs on the NPU; decode, drawing and WebRTC encode</tspan>'
      f'<tspan x="40" dy="20">stay on the CPU. One detection worker, so frames never queue up behind the NPU.</tspan></text>')
d.legend([("focal", "NPU step"), ("backend", "CPU step"), ("store", "Source / store"), ("input", "On the page"),
          ("muted", "Per frame"), ("link", "To the browser")])
write("cascade-pipeline.html", d, "Detail · Stage 1 · Detection",
      "Every frame: decode → NPU → boxes + vehicle log",
      "From <code>DetectionVideoTrack.recv()</code>: what happens to each 1080p frame between the video file and the page.")

# =====================================================================
# 3. Click-to-Describe (Sequence)
# =====================================================================
d = SVG("cascade-describe", "Click-to-Describe sequence",
        "Sequence of a Describe click: the browser asks the WebUI server over WebSocket, the server checks its cooldown and "
        "lock, sends the vehicle's best crop through the GenieX proxy to Qwen3-VL-4B on the NPU, and returns the caption, "
        "or after a 25 second timeout returns an error and starts a 40 second cooldown.")
L = {"B": 160, "S": 460, "P": 760, "G": 1040}
for k, cx in L.items():
    d.raw(f'<line x1="{cx}" y1="152" x2="{cx}" y2="628" stroke="rgba(45,49,66,0.20)" stroke-width="1" stroke-dasharray="3,3"/>')
# alt fragment frame (zone layer: drawn before messages)
d.zones.append(
    '<rect x="136" y="452" width="928" height="176" rx="4" fill="rgba(45,49,66,0.02)" stroke="rgba(45,49,66,0.22)" stroke-width="1"/>'
    f'<rect x="136" y="452" width="40" height="16" rx="2" fill="{PAPER}" stroke="rgba(45,49,66,0.22)" stroke-width="1"/>'
    f'<text x="156" y="464" fill="{MUTED}" font-size="8" font-family="{MONO}" text-anchor="middle" letter-spacing="0.12em">ALT</text>'
    f'<text x="188" y="464" fill="{MUTED}" font-size="11" font-family="{MONO}" letter-spacing="0.04em">[reply within 25 s]</text>'
    '<line x1="144" y1="560" x2="1056" y2="560" stroke="rgba(45,49,66,0.20)" stroke-width="1" stroke-dasharray="4,3"/>'
    f'<text x="148" y="578" fill="{MUTED}" font-size="11" font-family="{MONO}" letter-spacing="0.04em">[timeout or error]</text>'
)
# activation bars
for cx, y1, y2 in [(L["S"], 192, 616), (L["P"], 296, 520), (L["G"], 376, 496)]:
    d.raw(f'<rect x="{cx - 4}" y="{y1}" width="8" height="{y2 - y1}" fill="rgba(45,49,66,0.06)" stroke="{MUTED}" stroke-width="0.8"/>')


def msg(a, b, y, text, kind="link", color=None):
    x1, x2 = L[a], L[b]
    s = 4 if x2 > x1 else -4
    d.arrow(f"M {x1 + s},{y} H {x2 - s}", kind)
    d.label((x1 + x2) // 2, y - 22, text, color or (LINK if "link" in kind else SOFT))


def self_msg(a, y, text):
    cx = L[a] + 4
    d.arrow(f"M {cx},{y} H {cx + 28} Q {cx + 36},{y} {cx + 36},{y + 8} V {y + 16} Q {cx + 36},{y + 24} {cx + 28},{y + 24} H {cx + 4}")
    d.label(cx + 48, y + 4, text, SOFT, anchor="left")


msg("B", "S", 208, "DESCRIBE {ID}")
self_msg("S", 224, "COOLDOWN·LOCK")
msg("S", "B", 280, "PENDING", "async", SOFT)
msg("S", "P", 312, "POST /V1/CHAT")
self_msg("P", 328, "STRIP :W4A16")
msg("P", "G", 392, "CROP + PROMPT")
self_msg("G", 408, "QWEN3-VL · NPU")
msg("G", "P", 496, "COMPLETION", "dashed", SOFT)
msg("P", "S", 520, "CAPTION", "dashed", SOFT)
msg("S", "B", 544, "CAPTION DONE", "accent", ACCENT)
msg("S", "B", 616, "RETRY IN 40 S", "dashed", SOFT)
d.node(40, 88, 240, 64, "input", "UI", "Browser", ["/traffic page"])
d.node(340, 88, 240, 64, "backend", "API", "WebUI server", ["Describe service"])
d.node(640, 88, 240, 64, "backend", "PROXY", "GenieX proxy", [":18182"])
d.node(920, 88, 240, 64, "focal", "NPU", "Qwen3-VL-4B", ["GenieX :18181 · ~3–10 s"])
d.legend([("link", "Request"), ("dashed", "Return"), ("async", "Async notify"), ("accent", "Caption shown"),
          ("focal", "Runs on the NPU")])
write("cascade-describe.html", d, "Detail · Stage 2 · Vision-language",
      "Click a vehicle → Qwen3-VL describes it",
      "One caption at a time (lock), 25 s client timeout, 40 s cooldown after a failure so abandoned GenieX requests can't pile up.")
# =====================================================================
# 0. How it works (Architecture, simplified - the one to present)
# =====================================================================
d = SVG("cascade-how-it-works", "How the detection to VLM cascade works",
        "A camera or video feeds YOLO26n on the Hexagon NPU, which detects vehicles; a tracker crops the best view of "
        "each one; Qwen3-VL-4B, also on the NPU, turns each crop into a structured record of colour, type, brand and "
        "number plate, which is stored in a database and shown to operators on a dashboard.")
d.zone(40, 96, 208, 528, "INPUT")
d.zone(288, 96, 600, 528, "ON-DEVICE · ADVANTECH AFE-A503 · QUALCOMM IQ-9075")
d.zone(920, 96, 320, 528, "RESULTS")
# arrows
d.arrow("M 232,172 H 304")                                            # camera -> yolo
d.arrow("M 232,316 H 260 Q 268,316 268,308 V 212 Q 268,204 276,204 H 304")  # video -> yolo
d.arrow("M 576,184 H 616")                                            # yolo -> tracker
d.arrow("M 744,240 V 304")                                            # tracker -> vlm (crops)
d.arrow("M 616,424 H 448 Q 440,424 440,432 V 496")                    # vlm -> json
d.arrow("M 576,552 H 936", "link")                                    # json -> database
d.arrow("M 1080,504 V 408", "link")                                   # database -> dashboard
d.arrow("M 1080,312 V 224", "link")                                   # dashboard -> operators
d.label(752, 264, "VEHICLE CROPS", SOFT, anchor="left")
d.label(528, 400, "JSON", SOFT)
d.label(744, 528, "STORE RECORD", LINK)
d.label(1088, 452, "QUERY", LINK, anchor="left")
d.label(1088, 260, "LIVE + ALERTS", LINK, anchor="left")
# nodes
d.inode(56, 128, 176, 96, "input", "camera", "Camera", ["RTSP · USB", "field camera"], step=1)
d.inode(56, 272, 176, 88, "input", "movie", "Video file", ["recorded", "MP4 · looped"])
d.inode(304, 128, 272, 112, "focal", "cpu", "YOLO26n detection", ["Ultralytics · 640×640", "Hexagon NPU (QNN HTP)", "~15 ms per frame"], step=2)
d.inode(616, 128, 256, 112, "backend", "crop", "Track + crop", ["IoU tracker", "entry zone filter", "best crop / vehicle"], step=3)
d.inode(616, 304, 256, 152, "focal", "cpu", "Qwen3-VL-4B", ["Instruct · W4A16", "GenieX · Hexagon NPU", "~3–10 s per vehicle"], step=4)
# prompt chip inside the VLM node
d.raw(f'<rect x="628" y="412" width="232" height="32" rx="4" fill="#ffffff" stroke="rgba(235,108,54,0.35)" stroke-width="0.8"/>'
      + icon("terminal", 636, 420, 16, ACCENT)
      + f'<text x="660" y="432" fill="{INK}" font-size="11" font-family="{MONO}">color · type · brand · plate</text>')
# JSON record card (values from a real run on the parking clip)
lines = ['{', '  "vehicle_type": "sedan",', '  "color": "black",', '  "brand": "Subaru",', '  "number_plate": "KR 2492K"', '}']
d.raw(f'<rect x="304" y="496" width="272" height="112" rx="6" fill="{PAPER}"/>'
      f'<rect x="304" y="496" width="272" height="112" rx="6" fill="#ffffff" stroke="rgba(45,49,66,0.30)" stroke-width="1"/>')
for i, ln in enumerate(lines):
    assert w_mono(ln, SUB) <= 272 - 24, ln
    if ":" in ln:
        k, v = ln.split(":", 1)
        body = f'<tspan fill="{INK}">{escape(k)}:</tspan><tspan fill="{LINK}">{escape(v)}</tspan>'
    else:
        body = f'<tspan fill="{SOFT}">{escape(ln)}</tspan>'
    d.raw(f'<text x="320" y="{520 + i * 16}" font-size="{SUB}" font-family="{MONO}" xml:space="preserve">{body}</text>')
d.inode(936, 504, 288, 96, "store", "database", "Database", ["plate · time · camera", "+ vehicle crop"], step=5)
d.inode(936, 312, 288, 96, "backend", "desktop", "Dashboard", ["live view · alerts", "search by plate"])
d.inode(936, 128, 288, 96, "input", "users", "Operators", ["security desk", "facility team"])
d.raw(f'<text x="304" y="296" fill="{MUTED}" font-size="14" font-style="italic" font-family="\'Instrument Serif\', serif">'
      f'<tspan x="304" dy="0">Both AI models run on the board\'s</tspan>'
      f'<tspan x="304" dy="20">Hexagon NPU — video and plates</tspan>'
      f'<tspan x="304" dy="20">never leave the device.</tspan></text>')
d.legend([("focal", "AI model on the NPU"), ("backend", "Processing / app"), ("input", "Source / people"),
          ("store", "Data"), ("muted", "On-device flow"), ("link", "Results out")])
write("cascade-how-it-works.html", d, "How it works · Advantech AFE-A503 · Qualcomm IQ-9075",
      "From camera to searchable vehicle records — on one device",
      "YOLO26n spots and tracks each vehicle; Qwen3-VL-4B reads its colour, type, brand and plate from the best crop; "
      "the record goes to a database and the operators' dashboard.")
print("written")
