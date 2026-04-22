"""Generate a clean methodology flowchart PNG for the LaTeX paper."""
from pathlib import Path

import matplotlib.patches as mp
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

stages = [
    (1.0, 8.4, "Data Acquisition\n(Kaggle train.csv, 891 rows)", "#3498db"),
    (1.0, 6.4, "Profiling\n(missing, types, distributions)", "#9b59b6"),
    (1.0, 4.4, "Cleaning & Imputation\n(group-median Age, mode Embarked)", "#e67e22"),
    (1.0, 2.4, "Feature Engineering\n(Title, FamilySize, IsAlone, Deck, bins)", "#1abc9c"),
    (5.2, 2.4, "Encoding\n(Sex binary, one-hot categoricals)", "#f39c12"),
    (5.2, 4.4, "80/20 Stratified Split\n(Train n=712, Test n=179)", "#16a085"),
    (5.2, 6.4, "10-Fold Stratified CV\n(model selection on train)", "#2980b9"),
    (5.2, 8.4, "Classifiers\nNB, DT, LR, RF", "#8e44ad"),
    (8.9, 5.4, "Evaluation\nAcc / P / R / F1 / AUC\nMcNemar significance", "#c0392b"),
]

for x, y, text, color in stages:
    box = mp.FancyBboxPatch(
        (x - 1.4, y - 0.7), 2.8, 1.4,
        boxstyle="round,pad=0.08",
        linewidth=1.6, edgecolor="black",
        facecolor=color, alpha=0.82,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=9.5, weight="bold", color="white")


def arrow(x1, y1, x2, y2):
    ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#2c3e50"))


# Left column top-to-bottom
arrow(1.0, 7.7, 1.0, 7.1)
arrow(1.0, 5.7, 1.0, 5.1)
arrow(1.0, 3.7, 1.0, 3.1)
# Left to right
arrow(2.4, 2.4, 3.8, 2.4)
# Right column bottom-to-top
arrow(5.2, 3.1, 5.2, 3.7)
arrow(5.2, 5.1, 5.2, 5.7)
arrow(5.2, 7.1, 5.2, 7.7)
# Right to evaluation
arrow(6.6, 8.4, 7.5, 6.0)
arrow(6.6, 6.4, 7.5, 5.4)

ax.set_title("KDD Process Applied to Titanic Survival Prediction",
             fontsize=14, weight="bold", pad=16)

fig.tight_layout()
fig.savefig(FIG / "methodology_flowchart.png", dpi=300, bbox_inches="tight")
print("Saved methodology_flowchart.png")
