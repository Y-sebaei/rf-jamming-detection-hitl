"""
Figures for the revised paper (IC-FTAI 2026, paper 577280).
Values are the ones printed in _revision_out.txt by ReviewerRevision.py.
Writes PNG and PDF versions of each figure into figures/.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "pdf.fonttype": 42,
})

# Figures land in the repository. The paper build directory this script
# originally also wrote to lives outside the repo and is not assumed here.
# The figure names already carry the figures/ prefix.
OUT_DIRS = ["."]


def save(fig, name):
    """Write a vector PDF (what LaTeX prefers) and a flat RGB PNG fallback."""
    from PIL import Image
    for d in OUT_DIRS:
        if not os.path.isdir(d):
            continue
        pdf_path = os.path.join(d, name.replace(".png", ".pdf"))
        png_path = os.path.join(d, name)
        fig.savefig(pdf_path)
        fig.savefig(png_path)
        im = Image.open(png_path)
        if im.mode in ("RGBA", "LA", "P"):
            flat = Image.new("RGB", im.size, (255, 255, 255))
            flat.paste(im, mask=im.convert("RGBA").split()[-1])
            flat.save(png_path)
    plt.close(fig)
    print("wrote", name, "and PDF")


# ---------------------------------------------------------------- budget curve
budgets = [100, 200, 400, 800]
unc_m = [77.61, 82.25, 81.81, 83.10]
unc_s = [5.09, 2.96, 4.63, 4.70]
rnd_m = [85.69, 89.95, 93.16, 94.83]
rnd_s = [3.58, 3.39, 2.60, 2.38]
baseline = 53.55

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.errorbar(budgets, rnd_m, yerr=rnd_s, marker="o", capsize=4, linewidth=2,
            color="#0F766E", label="Random selection")
ax.errorbar(budgets, unc_m, yerr=unc_s, marker="s", capsize=4, linewidth=2,
            color="#B91C1C", label="Uncertainty band")
ax.axhline(baseline, linestyle=":", color="0.35", linewidth=2,
           label="No adaptation (53.55%)")
ax.set_xscale("log")
ax.set_xticks(budgets)
ax.set_xticklabels([str(b) for b in budgets])
ax.minorticks_off()
ax.set_xlabel("Labeling budget (windows)")
ax.set_ylabel("Accuracy on target test set (%)")
ax.set_ylim(45, 100)
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
save(fig, "figures/budget_uncertainty_vs_random.png")

# ------------------------------------------------------------ reliability plot
# (mean predicted probability, observed jamming frequency, n) per 0.1 bin
cross = [(0.062, 0.041, 74), (0.143, 0.022, 181), (0.255, 0.098, 82),
         (0.354, 0.070, 43), (0.459, 0.089, 169), (0.563, 0.273, 242),
         (0.642, 0.191, 456), (0.758, 0.280, 707), (0.846, 0.131, 1146),
         (0.985, 0.700, 5091)]
within = [(0.022, 0.000, 2228), (0.142, 0.000, 129), (0.246, 0.000, 41),
          (0.345, 0.000, 22), (0.445, 0.522, 23), (0.559, 0.565, 23),
          (0.653, 0.800, 50), (0.758, 0.964, 84), (0.862, 0.988, 171),
          (0.981, 0.999, 2144)]

fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.plot([0, 1], [0, 1], linestyle="--", color="0.4", linewidth=1.5,
        label="Perfect calibration")
cx = [c[0] for c in cross]
cy = [c[1] for c in cross]
wx = [w[0] for w in within]
wy = [w[1] for w in within]
ax.plot(wx, wy, marker="o", linewidth=2, color="#0F766E",
        label="Within location (ECE 0.035)")
ax.plot(cx, cy, marker="s", linewidth=2, color="#B91C1C",
        label="Cross location (ECE 0.366)")
ax.axvspan(0.20, 0.80, color="0.85", zorder=0)
ax.text(0.5, 0.055, "uncertainty band", ha="center", fontsize=11, color="0.3")
ax.set_xlabel("Mean predicted probability of jamming")
ax.set_ylabel("Observed fraction of jamming")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
save(fig, "figures/calibration_reliability.png")
