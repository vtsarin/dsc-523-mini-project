"""Generate the methodology flowchart.

Design constraints derived from three prior failure modes:

  (F1) Boxes positioned at negative x -> text clipped on the left.
  (F2) Boxes packed too close -> arrows invisible between them.
  (F3) Boxes too small for the text -> text overflows and is clipped
       at the box edges.

The layout below explicitly chooses

  * x bounds that include generous padding on both sides (F1),
  * inter-box gaps of >= 2.0 coordinate units (F2), and
  * box widths and text lengths sized together so every label fits
    inside its box at the chosen font size (F3).
"""
from pathlib import Path

import matplotlib.patches as mp
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

# --- Geometry ------------------------------------------------------------
# We deliberately pick a source figure size (inches) close to the final rendered
# size in LaTeX (~5.7 in at 0.95\textwidth). That keeps the scale factor ~0.6
# so fonts, arrows and box outlines stay crisp in the printed PDF.
FIG_W, FIG_H = 10.5, 4.4        # figure size in inches
XMAX, YMAX   = 34, 9            # plot coordinate system
BOX_W, BOX_H = 2.5, 0.80        # half dimensions -> full 5.0 x 1.6
SPACING      = 7.0              # top-row centre-to-centre distance
                                # => inter-box gap = 7.0 - 5.0 = 2.0

TOP_Y, BOT_Y = 6.6, 1.9         # row heights (vertical arrow is 3.1 long)

# Leftmost top centre is placed one half-width + padding from the left edge.
PAD = 0.5
x0 = PAD + BOX_W                            # 3.0
TOP_XS = [x0 + i * SPACING for i in range(5)]
# Bottom row uses the same horizontal extent; 4 boxes across the same range.
BOT_XS = list(reversed(
    [x0 + i * (SPACING * 4 / 3) for i in range(4)]
))

TOP_STAGES = [
    ("Data Acquisition\nKaggle train.csv\n(n = 891)",         "#3498db"),
    ("Profiling\nmissingness + types",                        "#9b59b6"),
    ("Cleaning & Imputation\ngroup-median Age,\nmode Embarked","#e67e22"),
    ("Feature Engineering\nTitle, FamilySize,\nDeck, bins",   "#1abc9c"),
    ("Encoding\nSex binary +\none-hot categoricals",          "#f39c12"),
]
BOT_STAGES = [
    ("80 / 20 Stratified Split\ntrain = 712,  test = 179",    "#16a085"),
    ("10-Fold Stratified CV\nmodel selection on train",       "#2980b9"),
    ("Classifiers\nNB  |  DT  |  LR  |  RF",                  "#8e44ad"),
    ("Evaluation\nAcc / P / R / F1 / AUC\n+ McNemar test",    "#c0392b"),
]

# --- Drawing -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, XMAX)
ax.set_ylim(0, YMAX)
ax.axis("off")


def add_box(x, y, text, color):
    box = mp.FancyBboxPatch(
        (x - BOX_W, y - BOX_H),
        2 * BOX_W, 2 * BOX_H,
        boxstyle="round,pad=0.06,rounding_size=0.22",
        linewidth=1.7, edgecolor="#1a1a1a",
        facecolor=color, alpha=0.92,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=11, weight="bold", color="white",
            linespacing=1.25)


def arrow(x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>,head_length=0.6,head_width=0.4",
            lw=3.0, color="#1f2d3d", mutation_scale=22,
            shrinkA=0, shrinkB=0,
        ),
    )


# Top row (left -> right)
for x, (text, color) in zip(TOP_XS, TOP_STAGES):
    add_box(x, TOP_Y, text, color)
for i in range(len(TOP_XS) - 1):
    arrow(TOP_XS[i] + BOX_W, TOP_Y, TOP_XS[i + 1] - BOX_W, TOP_Y)

# Descent arrow (right-hand side): from bottom of last top box to top of first
# bottom box (which sits at the same x).
arrow(TOP_XS[-1], TOP_Y - BOX_H, BOT_XS[0], BOT_Y + BOX_H)

# Bottom row (right -> left)
for x, (text, color) in zip(BOT_XS, BOT_STAGES):
    add_box(x, BOT_Y, text, color)
for i in range(len(BOT_XS) - 1):
    arrow(BOT_XS[i] - BOX_W, BOT_Y, BOT_XS[i + 1] + BOX_W, BOT_Y)

fig.tight_layout(pad=0.2)
fig.savefig(FIG / "methodology_flowchart.png", dpi=300, bbox_inches="tight")
print("Saved methodology_flowchart.png")
print("Top centres:", [round(x, 2) for x in TOP_XS])
print("Bottom centres:", [round(x, 2) for x in BOT_XS])
print(f"Top inter-box gap: {SPACING - 2*BOX_W:.1f}")
print(f"Bottom inter-box gap: {SPACING*4/3 - 2*BOX_W:.1f}")
