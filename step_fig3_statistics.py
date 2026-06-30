"""
TractoPersist - Figure 3: Clean Statistical Plots
Professional violin + box plots like Nature papers
White background, minimal design

Run: python step_fig3_statistics.py
Output: F:\ADNI_Features\figures_paper\fig3_statistics.png
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu
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
print("  Figure 3: Statistical Violin Plots")
print("=" * 60)

# ── COLOURS ───────────────────────────────────────────────────
GRP_COL = {
    'CN':  '#2E7D32',
    'MCI': '#E65100',
    'AD':  '#C62828',
}
GRP_LIGHT = {
    'CN':  '#A5D6A7',
    'MCI': '#FFCC80',
    'AD':  '#EF9A9A',
}

# ── SIGNIFICANCE STARS ────────────────────────────────────────
def stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else:          return 'ns'

def sig_colour(p):
    if p < 0.001: return '#C62828'
    elif p < 0.01: return '#AD1457'
    elif p < 0.05: return '#1565C0'
    else:          return '#888888'

# ── FEATURES TO PLOT ──────────────────────────────────────────
features = [
    ('h1_count',
     'H1 Topological Loops',
     'Count', '#7B1FA2'),
    ('h1_mean_lifetime',
     'H1 Mean Lifetime',
     'Persistence', '#1565C0'),
    ('h1_max_lifetime',
     'H1 Max Lifetime',
     'Persistence', '#0277BD'),
    ('md_mean',
     'Mean Diffusivity (MD)',
     '×10⁻³ mm²/s', '#BF360C'),
    ('rd_mean',
     'Radial Diffusivity (RD)',
     '×10⁻³ mm²/s', '#E65100'),
    ('fa_mean',
     'Fractional Anisotropy (FA)',
     'FA value', '#1B5E20'),
]
features = [
    (f, t, u, c)
    for f, t, u, c in features
    if f in combined.columns]

grp_order = ['CN', 'MCI', 'AD']

# ── BUILD FIGURE ──────────────────────────────────────────────
fig = plt.figure(
    figsize=(16, 11),
    facecolor='white')

gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    hspace=0.45,
    wspace=0.38,
    left=0.07, right=0.96,
    top=0.90, bottom=0.10)

np.random.seed(42)

for idx, (feat, title,
           unit, col) in \
        enumerate(features[:6]):

    ax = fig.add_subplot(
        gs[idx // 3, idx % 3])
    ax.set_facecolor('white')

    data_g = [
        combined[
            combined['group'] == g][feat]
        .dropna().values
        for g in grp_order]

    x_pos = [1, 2, 3]

    # ── Violin ────────────────────────────
    vp = ax.violinplot(
        data_g,
        positions=x_pos,
        showmedians=False,
        showextrema=False,
        widths=0.6)

    for pc, grp in zip(
            vp['bodies'], grp_order):
        pc.set_facecolor(GRP_LIGHT[grp])
        pc.set_edgecolor(GRP_COL[grp])
        pc.set_alpha(0.7)
        pc.set_linewidth(1.5)

    # ── Box plot overlay ──────────────────
    bp = ax.boxplot(
        data_g,
        positions=x_pos,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(
            color='black',
            linewidth=2.5,
            zorder=5),
        whiskerprops=dict(
            color='#333333',
            linewidth=1.5),
        capprops=dict(
            color='#333333',
            linewidth=1.5),
        boxprops=dict(
            linewidth=1.5))

    for patch, grp in zip(
            bp['boxes'], grp_order):
        patch.set_facecolor('white')
        patch.set_edgecolor(GRP_COL[grp])
        patch.set_linewidth(2)
        patch.set_alpha(0.95)

    # ── Jitter points ─────────────────────
    for xi, (g, d) in enumerate(
            zip(grp_order, data_g)):
        jitter = np.random.normal(
            0, 0.06, len(d))
        ax.scatter(
            xi + 1 + jitter, d,
            alpha=0.25,
            s=10,
            color=GRP_COL[g],
            zorder=2,
            edgecolors='none',
            linewidths=0)

    # ── Statistics ────────────────────────
    y_all = np.concatenate(
        [d for d in data_g
         if len(d) > 0])
    y_max   = np.percentile(y_all, 97)
    y_min   = np.percentile(y_all, 3)
    y_range = y_max - y_min

    _, p_ad_cn = mannwhitneyu(
        data_g[2], data_g[0],
        alternative='two-sided')
    _, p_mci_cn = mannwhitneyu(
        data_g[1], data_g[0],
        alternative='two-sided')

    # MCI vs CN bracket
    y1 = y_max + y_range * 0.10
    ax.plot([1, 2], [y1, y1],
            '-', color='#333333',
            lw=1.2, zorder=4)
    ax.plot([1, 1],
            [y_max + y_range*0.03, y1],
            '-', color='#333333',
            lw=1.2, zorder=4)
    ax.plot([2, 2],
            [y_max + y_range*0.03, y1],
            '-', color='#333333',
            lw=1.2, zorder=4)
    ax.text(1.5, y1 + y_range*0.01,
            stars(p_mci_cn),
            ha='center', fontsize=12,
            color=sig_colour(p_mci_cn),
            fontweight='bold',
            zorder=5)

    # AD vs CN bracket
    y2 = y_max + y_range * 0.22
    ax.plot([1, 3], [y2, y2],
            '-', color='#333333',
            lw=1.2, zorder=4)
    ax.plot([1, 1],
            [y1 + y_range*0.01, y2],
            '-', color='#333333',
            lw=1.2, zorder=4)
    ax.plot([3, 3],
            [y_max + y_range*0.03, y2],
            '-', color='#333333',
            lw=1.2, zorder=4)
    ax.text(2, y2 + y_range*0.01,
            stars(p_ad_cn),
            ha='center', fontsize=12,
            color=sig_colour(p_ad_cn),
            fontweight='bold',
            zorder=5)

    # ── Axes styling ──────────────────────
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        grp_order,
        fontsize=11,
        fontweight='bold')
    for tick, grp in zip(
            ax.get_xticklabels(),
            grp_order):
        tick.set_color(GRP_COL[grp])

    ax.set_ylabel(
        unit, fontsize=10,
        color='#333333')
    ax.set_title(
        title,
        fontsize=11,
        fontweight='bold',
        color='#1A237E',
        pad=6)

    ax.set_ylim(
        y_min - y_range * 0.08,
        y2     + y_range * 0.08)

    ax.grid(
        True, axis='y',
        linestyle='--',
        alpha=0.35,
        color='#AAAAAA',
        zorder=0)
    ax.set_axisbelow(True)

    # p-value box
    ax.text(
        0.97, 0.97,
        f'p = {p_ad_cn:.4f} '
        f'{stars(p_ad_cn)}',
        ha='right', va='top',
        transform=ax.transAxes,
        fontsize=8,
        color=sig_colour(p_ad_cn),
        fontweight='bold',
        bbox=dict(
            facecolor='white',
            alpha=0.9,
            edgecolor=col,
            linewidth=1.2,
            boxstyle='round,pad=0.3'))

    # Group means below x-axis
    for xi, (g, d) in enumerate(
            zip(grp_order, data_g)):
        ax.text(
            xi + 1,
            y_min - y_range * 0.06,
            f'μ={d.mean():.3f}',
            ha='center', fontsize=7.5,
            color=GRP_COL[g],
            fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(
        '#CCCCCC')
    ax.spines['bottom'].set_color(
        '#CCCCCC')

# ── LEGEND ────────────────────────────────────────────────────
legend_elements = [
    Patch(facecolor=GRP_LIGHT[g],
          edgecolor=GRP_COL[g],
          linewidth=2,
          label=f'{g} (n='
                f'{sum(combined["group"]==g)})')
    for g in grp_order]

fig.legend(
    handles=legend_elements,
    loc='lower center',
    ncol=3,
    fontsize=11,
    framealpha=0.95,
    edgecolor='#CCCCCC',
    bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    "Topological and Microstructural"
    " Features Across the AD Spectrum\n"
    "Mann-Whitney U test  |  "
    "*** p<0.001   ** p<0.01"
    "   * p<0.05   ns not significant",
    fontsize=13,
    fontweight='bold',
    color='#1A237E',
    y=0.975)

save_path = os.path.join(
    fig_path, 'fig3_statistics.png')
plt.savefig(
    save_path,
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none')
plt.close()
print(f"  Saved: {save_path}")
print("  Done!")
