"""Draw the system architecture figure for the IET Software manuscript.

SCI-journal style: muted palette, hairline boxes, Times typography,
orthogonal connectors, single-column canvas (3.5 in wide) so the figure
renders at 1:1 when included with width=\\linewidth.

The layout mirrors the actual serving path in app/api.py:
    client -> FastAPI -> preprocessing -> model router -> one of four
    backends -> JSON response carrying label, confidence and latency.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures" / "architecture.png"
BENCH = ROOT / "experiments" / "inference_bench.json"

bench = json.loads(BENCH.read_text(encoding="utf-8"))
res = {r["model"]: r for r in bench["results"]}

# ---- typography: Times New Roman if installed, else a clean sans fallback ----
_installed = {f.name for f in fm.fontManager.ttflist}
FONT = "Times New Roman" if "Times New Roman" in _installed else "DejaVu Sans"
plt.rcParams.update({"font.family": FONT, "mathtext.fontset": "stix"})

# ---- muted palette ----
INK = "#4A4A4A"                                   # connectors
SVC_FC, SVC_EC, SVC_TC = "#EEF3F9", "#6C8FB5", "#1E3A56"   # ingress / response
RTR_FC, RTR_EC, RTR_TC = "#F4F4F4", "#8A8A8A", "#3A3A3A"   # router
BE_FC, BE_EC, BE_TC = "#EDF4EE", "#5F8A6B", "#274D36"      # backends
BE_METRIC = "#4E6E58"

W, H = 3.50, 3.24
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, fc, ec, r=0.05):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
        linewidth=0.8, edgecolor=ec, facecolor=fc, zorder=3))


def txt(x, y, s, size, color, weight="normal", style="normal"):
    ax.text(x, y, s, ha="center", va="center", color=color, fontsize=size,
            weight=weight, style=style, zorder=5)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
        linewidth=0.8, color=INK, zorder=2, shrinkA=0, shrinkB=0))


def seg(x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color=INK, linewidth=0.8, zorder=2,
            solid_capstyle="round")


# ---------------- row 1: ingress + service ----------------
TY, TH, TW = 2.58, 0.56, 1.04
TXS = [0.04, 1.23, 2.42]
top_boxes = [
    ("Client", "POST /predict", "POST /predict/batch"),
    ("FastAPI service", "ModelManager", "artifact cache"),
    ("Preprocessing", "clean + normalise", "text"),
]
for x, (l1, l2, l3) in zip(TXS, top_boxes):
    box(x, TY, TW, TH, SVC_FC, SVC_EC)
    txt(x + TW / 2, TY + 0.40, l1, 7.2, SVC_TC, weight="bold")
    txt(x + TW / 2, TY + 0.24, l2, 6.4, SVC_TC)
    txt(x + TW / 2, TY + 0.09, l3, 6.4, SVC_TC)
arrow(TXS[0] + TW, TY + TH / 2, TXS[1], TY + TH / 2)
arrow(TXS[1] + TW, TY + TH / 2, TXS[2], TY + TH / 2)

# ---------------- router ----------------
RW, RH, RY = 1.50, 0.62, 1.66
RX = 2.50 - RW / 2
box(RX, RY, RW, RH, RTR_FC, RTR_EC)
txt(2.50, RY + 0.40, "Model router", 7.2, RTR_TC, weight="bold")
txt(2.50, RY + 0.18, "model = {bert, tfidf_*, vader}", 6.4, RTR_TC)
# preprocessing bottom -> router top
arrow(TXS[2] + TW / 2, TY, TXS[2] + TW / 2, RY + RH)

# ---------------- backend row ----------------
BY, BH, BW = 0.72, 0.64, 0.81
BXS = [0.025, 0.905, 1.785, 2.665]
DISPLAY = ["vader", "tfidf_logreg", "tfidf_svc", "distilbert"]
NAMES = {
    "vader": ("VADER", "lexicon baseline"),
    "tfidf_logreg": ("TF-IDF +", "LogReg"),
    "tfidf_svc": ("TF-IDF +", "LinearSVC"),
    "distilbert": ("DistilBERT", "fine-tuned"),
}
# router -> backend bus
BUS_Y = RY - 0.16
seg(2.50, RY, 2.50, BUS_Y)
seg(BXS[0] + BW / 2, BUS_Y, BXS[-1] + BW / 2, BUS_Y)
for i, key in enumerate(DISPLAY):
    x = BXS[i]
    cx = x + BW / 2
    r = res[key]
    box(x, BY, BW, BH, BE_FC, BE_EC)
    l1, l2 = NAMES[key]
    txt(cx, BY + 0.50, l1, 7.2, BE_TC, weight="bold")
    txt(cx, BY + 0.375, l2, 6.4, BE_TC, style="italic")
    seg(x + 0.12, BY + 0.27, x + BW - 0.12, BY + 0.27)
    txt(cx, BY + 0.155, f"{r['latency_median_ms']:.2f} ms median", 6.4, BE_METRIC)
    txt(cx, BY + 0.045, f"{r['rss_footprint_mb']:.0f} MB resident", 6.4, BE_METRIC)
    arrow(cx, BUS_Y, cx, BY + BH)

# ---------------- collector rail + response ----------------
RAIL_Y = 0.50
for x in BXS:
    seg(x + BW / 2, BY, x + BW / 2, RAIL_Y)
seg(BXS[0] + BW / 2, RAIL_Y, BXS[-1] + BW / 2, RAIL_Y)
JW, JH, JY = 1.70, 0.42, 0.02
JX = 1.75 - JW / 2
box(JX, JY, JW, JH, SVC_FC, SVC_EC)
txt(1.75, JY + 0.29, "JSON response", 7.2, SVC_TC, weight="bold")
txt(1.75, JY + 0.11, "label + confidence + latency_ms", 6.4, SVC_TC)
arrow(1.75, RAIL_Y, 1.75, JY + JH)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300, facecolor="white")
plt.close(fig)

from PIL import Image  # noqa: E402
im = Image.open(OUT)
im.save(OUT, dpi=(300, 300))
print(f"font: {FONT}")
for key in DISPLAY:
    r = res[key]
    print(f"  {key}: median={r['latency_median_ms']} ms, rss={r['rss_footprint_mb']} MB")
print(f"written: {OUT} ({OUT.stat().st_size:,} bytes)")
print("size:", Image.open(OUT).size, "dpi:", Image.open(OUT).info.get("dpi"))
