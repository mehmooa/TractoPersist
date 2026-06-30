"""
TractoPersist - Figure 5: Clean Pipeline Diagram
Professional methodology flowchart
White background, clean design

Run: python step_fig5_pipeline.py
Output: F:\ADNI_Features\figures_paper\fig5_pipeline.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, FancyArrowPatch)
import warnings
warnings.filterwarnings('ignore')

fig_path = r"F:\ADNI_Features\figures_paper"
os.makedirs(fig_path, exist_ok=True)

print("=" * 60)
print("  Figure 5: Pipeline Diagram")
print("=" * 60)

fig, ax = plt.subplots(
    figsize=(18, 5),
    facecolor='white')
ax.set_facecolor('white')
ax.set_xlim(0, 18)
ax.set_ylim(0, 5)
ax.axis('off')

# ── PIPELINE STEPS ────────────────────────────────────────────
steps = [
    ("DTI\nAcquisition",
     "ADNI-3\n54 gradients\nb=1000 s/mm²",
     "#1A237E", "①"),
    ("Preprocessing",
     "MP-PCA denoise\nGibbs correction\nBrain mask",
     "#283593", "②"),
    ("CSD\nTractography",
     "iFOD2 algorithm\n100K streamlines\nProbabilistic",
     "#1565C0", "③"),
    ("SIFT2\nFiltering",
     "Streamline weights\nFibre density\nBiologically valid",
     "#0277BD", "④"),
    ("Connectome\n(48 ROI)",
     "Harvard-Oxford\nAtlas parcellation\n48×48 matrix",
     "#00695C", "⑤"),
    ("FA/MD/RD\nExtraction",
     "Per-ROI metrics\n144 features\nMicrostructure",
     "#1B5E20", "⑥"),
    ("Persistent\nHomology",
     "Vietoris-Rips\nH0 + H1 features\n13 descriptors",
     "#4A148C", "⑦"),
    ("Hybrid\nGAT+GT GNN",
     "Topology attention\n2.5M parameters\nClassification",
     "#880E4F", "⑧"),
]

bw  = 1.85
bh  = 2.8
yb  = 0.95
xs  = 0.30
gap = 2.22

for i, (title, sub,
         col, num) in enumerate(steps):
    x = xs + i * gap

    # Shadow
    shadow = FancyBboxPatch(
        (x + 0.06, yb - 0.06),
        bw, bh,
        boxstyle="round,pad=0.1",
        facecolor='#E8E8E8',
        linewidth=0,
        zorder=1)
    ax.add_patch(shadow)

    # Main box
    rect = FancyBboxPatch(
        (x, yb), bw, bh,
        boxstyle="round,pad=0.1",
        facecolor='white',
        edgecolor=col,
        linewidth=2.5,
        zorder=2)
    ax.add_patch(rect)

    # Colour top accent
    top = FancyBboxPatch(
        (x, yb + bh * 0.62),
        bw, bh * 0.38,
        boxstyle="round,pad=0.05",
        facecolor=col,
        linewidth=0,
        zorder=3)
    ax.add_patch(top)

    # Step number circle
    circ = plt.Circle(
        (x + 0.28, yb + bh - 0.28),
        0.22,
        color='white',
        zorder=5)
    ax.add_patch(circ)
    ax.text(
        x + 0.28, yb + bh - 0.28,
        num,
        ha='center', va='center',
        fontsize=10,
        fontweight='bold',
        color=col, zorder=6)

    # Title (white on colour)
    ax.text(
        x + bw/2,
        yb + bh * 0.80,
        title,
        ha='center', va='center',
        fontsize=10,
        fontweight='bold',
        color='white',
        linespacing=1.3,
        zorder=4)

    # Divider
    ax.plot(
        [x + 0.2, x + bw - 0.2],
        [yb + bh * 0.61,
         yb + bh * 0.61],
        '-', color='#DDDDDD',
        lw=1.2, zorder=3)

    # Subtitle (dark on white)
    ax.text(
        x + bw/2,
        yb + bh * 0.28,
        sub,
        ha='center', va='center',
        fontsize=7.8,
        color='#333333',
        linespacing=1.4,
        zorder=4)

    # Arrow
    if i < len(steps) - 1:
        ax.annotate(
            '',
            xy=(x + bw + 0.31,
                yb + bh/2),
            xytext=(x + bw + 0.05,
                    yb + bh/2),
            arrowprops=dict(
                arrowstyle='-|>',
                color='#444444',
                lw=2.0,
                mutation_scale=18),
            zorder=6)

# ── INPUT / OUTPUT LABELS ─────────────────────────────────────
ax.text(
    0.30 + bw/2,
    yb - 0.35,
    'Input:\nADNI-3 dMRI\n(n=214)',
    ha='center', fontsize=8,
    color='#1A237E',
    fontweight='bold',
    style='italic')

ax.text(
    xs + 7 * gap + bw/2,
    yb - 0.35,
    'Output:\nAD Classification\n& Topology Analysis',
    ha='center', fontsize=8,
    color='#880E4F',
    fontweight='bold',
    style='italic')

ax.set_title(
    "TractoPersist: End-to-End"
    " Processing Pipeline",
    fontsize=14,
    fontweight='bold',
    color='#1A237E',
    pad=10)

save_path = os.path.join(
    fig_path, 'fig5_pipeline.png')
plt.savefig(
    save_path,
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none')
plt.close()
print(f"  Saved: {save_path}")
print("  Done!")
