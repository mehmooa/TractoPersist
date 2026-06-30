"""
TractoPersist - Figure 4: Correlation Analysis
Forest-plot style like MINT paper Figure 6
Clean white background

Run: python step_fig4_correlation.py
Output: F:\ADNI_Features\figures_paper\fig4_correlation.png
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ── PATHS ─────────────────────────────────────────────────────
output_path = r"F:\ADNI_Features"
fig_path    = r"F:\ADNI_Features\figures_paper"
os.makedirs(fig_path, exist_ok=True)

combined = pd.read_csv(
    os.path.join(output_path,
                 "combined_features.csv"))

print("=" * 60)
print("  Figure 4: Correlation Analysis")
print("=" * 60)

# ── FEATURES ──────────────────────────────────────────────────
topo_feats = [
    ('h1_count',           'H1 Count'),
    ('h1_mean_lifetime',   'H1 Mean Life'),
    ('h1_max_lifetime',    'H1 Max Life'),
    ('h1_sum_lifetime',    'H1 Sum Life'),
    ('total_persistence_h1','H1 Total Persist.'),
    ('betti_1',            'Betti β₁'),
    ('h0_count',           'H0 Count'),
]
dti_feats = [
    ('fa_mean',  'FA Mean',  '#1B5E20'),
    ('fa_std',   'FA Std',   '#388E3C'),
    ('md_mean',  'MD Mean',  '#BF360C'),
    ('md_std',   'MD Std',   '#E64A19'),
    ('rd_mean',  'RD Mean',  '#E65100'),
    ('rd_std',   'RD Std',   '#F57C00'),
]

# Filter available
topo_feats = [
    (f, l) for f, l in topo_feats
    if f in combined.columns]
dti_feats  = [
    (f, l, c) for f, l, c in dti_feats
    if f in combined.columns]

# ── COMPUTE CORRELATIONS ─────────────────────────────────────
corr_mat = np.zeros(
    (len(topo_feats), len(dti_feats)))
pval_mat = np.zeros_like(corr_mat)

for i, (tf, _) in enumerate(topo_feats):
    for j, (df_, _, _) in enumerate(
            dti_feats):
        vals = combined[[tf, df_]]\
            .dropna()
        if len(vals) < 10:
            continue
        r, p = spearmanr(
            vals[tf], vals[df_])
        corr_mat[i, j] = r
        pval_mat[i, j] = p

# ── BUILD FIGURE ──────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 2,
    figsize=(16, 7),
    facecolor='white',
    gridspec_kw={
        'width_ratios': [1.6, 1],
        'wspace': 0.45})

# ── LEFT: Heatmap ──────────────────────────────────────────────
ax1 = axes[0]
ax1.set_facecolor('white')

topo_labels = [l for _, l in topo_feats]
dti_labels  = [l for _, l, _ in dti_feats]

im = ax1.imshow(
    corr_mat,
    cmap='RdBu_r',
    vmin=-0.5, vmax=0.5,
    aspect='auto',
    interpolation='nearest')

# Annotate with values
for i in range(len(topo_feats)):
    for j in range(len(dti_feats)):
        r = corr_mat[i, j]
        p = pval_mat[i, j]

        # Significance star
        s = ('***' if p < 0.001
             else '**' if p < 0.01
             else '*'  if p < 0.05
             else '')

        text_col = 'white' \
            if abs(r) > 0.3 \
            else 'black'

        ax1.text(
            j, i,
            f'{r:.2f}{s}',
            ha='center',
            va='center',
            fontsize=9,
            fontweight='bold'
            if s else 'normal',
            color=text_col)

ax1.set_xticks(range(len(dti_feats)))
ax1.set_xticklabels(
    dti_labels,
    rotation=35,
    ha='right',
    fontsize=10)

ax1.set_yticks(range(len(topo_feats)))
ax1.set_yticklabels(
    topo_labels,
    fontsize=10)

# Grid lines
for i in range(len(topo_feats) + 1):
    ax1.axhline(
        i - 0.5,
        color='white',
        lw=1.5)
for j in range(len(dti_feats) + 1):
    ax1.axvline(
        j - 0.5,
        color='white',
        lw=1.5)

cbar = plt.colorbar(
    im, ax=ax1,
    fraction=0.046, pad=0.04)
cbar.set_label(
    "Spearman's r",
    fontsize=10,
    color='black')
cbar.ax.tick_params(
    colors='black', labelsize=8)

ax1.set_title(
    "Spearman Correlation:\n"
    "Persistent Homology vs "
    "DTI Microstructure (n=214)",
    fontsize=12,
    fontweight='bold',
    color='#1A237E',
    pad=12)

ax1.set_xlabel(
    "DTI Metrics",
    fontsize=11, color='black')
ax1.set_ylabel(
    "Persistent Homology Features",
    fontsize=11, color='black')

# ── RIGHT: Top significant correlations ───────────────────────
ax2 = axes[1]
ax2.set_facecolor('white')

# Collect all significant pairs
sig_pairs = []
for i, (tf, tl) in \
        enumerate(topo_feats):
    for j, (df_, dl, dc) in \
            enumerate(dti_feats):
        r = corr_mat[i, j]
        p = pval_mat[i, j]
        if p < 0.05:
            sig_pairs.append(
                (abs(r), r, p,
                 tl, dl, dc))

sig_pairs.sort(
    key=lambda x: x[0],
    reverse=True)

if sig_pairs:
    top_n = min(10, len(sig_pairs))
    top   = sig_pairs[:top_n]

    y_pos = np.arange(top_n)

    for yi, (_, r, p,
              tl, dl, dc) in \
            enumerate(top):
        s     = ('***' if p < 0.001
                 else '**' if p < 0.01
                 else '*')
        color = '#C62828' if r < 0 \
            else '#1565C0'

        ax2.barh(
            yi, r,
            height=0.6,
            color=color,
            alpha=0.75,
            edgecolor=color,
            linewidth=1.2)

        ax2.text(
            r + (0.01 if r > 0 else -0.01),
            yi,
            f'{r:.3f}{s}',
            ha='left' if r > 0 else 'right',
            va='center',
            fontsize=8.5,
            fontweight='bold',
            color=color)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [f'{tl}\nvs {dl}'
         for _, _, _, tl, dl, _
         in top],
        fontsize=8.5)

    ax2.axvline(
        0, color='black',
        lw=1.2, zorder=5)
    ax2.set_xlim(-0.35, 0.35)
    ax2.set_xlabel(
        "Spearman's r",
        fontsize=10)
    ax2.set_title(
        "Top Significant\nCorrelations",
        fontsize=12,
        fontweight='bold',
        color='#1A237E',
        pad=12)
    ax2.grid(
        True, axis='x',
        linestyle='--',
        alpha=0.35)
    ax2.spines['top']\
        .set_visible(False)
    ax2.spines['right']\
        .set_visible(False)
else:
    ax2.text(
        0.5, 0.5,
        'No significant\ncorrelations\nfound',
        ha='center', va='center',
        transform=ax2.transAxes,
        fontsize=12,
        color='#888888')
    ax2.axis('off')

# Legend for significance
from matplotlib.lines import Line2D
legend_elem = [
    Line2D([0],[0], marker='',
           ls='none',
           label='*** p<0.001'),
    Line2D([0],[0], marker='',
           ls='none',
           label='**  p<0.01'),
    Line2D([0],[0], marker='',
           ls='none',
           label='*   p<0.05'),
]
ax1.legend(
    handles=legend_elem,
    loc='lower right',
    fontsize=8,
    framealpha=0.9,
    edgecolor='#CCCCCC')

fig.suptitle(
    "Microstructure–Topology Correlations"
    " in TractoPersist",
    fontsize=14,
    fontweight='bold',
    color='#1A237E',
    y=0.98)

save_path = os.path.join(
    fig_path, 'fig4_correlation.png')
plt.savefig(
    save_path,
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none')
plt.close()
print(f"  Saved: {save_path}")
print("  Done!")
