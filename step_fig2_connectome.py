"""
TractoPersist - Figure 2: Improved Connectome
Better glass brain + cleaner matrices
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nilearn import plotting
import warnings
warnings.filterwarnings('ignore')

# ── PATHS ──────────────────────────────────────────────────────
base_path   = r"F:\ADNI_NIfTI"
output_path = r"F:\ADNI_Features"
fig_path    = r"F:\ADNI_Features\figures_paper"
os.makedirs(fig_path, exist_ok=True)

combined = pd.read_csv(
    os.path.join(output_path,
                 "combined_features.csv"))

print("=" * 60)
print("  Figure 2: Connectome (Improved)")
print("=" * 60)

groups_info = {
    'CN':  (combined[
                combined['group']=='CN'],
            os.path.join(base_path,'CN1')),
    'MCI': (combined[
                combined['group']=='MCI'],
            os.path.join(base_path,'MCI_v2')),
    'AD':  (combined[
                combined['group']=='AD'],
            os.path.join(base_path,'AD_v2')),
}

group_colours = {
    'CN':  '#2E7D32',
    'MCI': '#E65100',
    'AD':  '#C62828',
}

cmaps_conn = {
    'CN':  'Greens',
    'MCI': 'Oranges',
    'AD':  'Reds',
}

# ── LOAD MEAN CONNECTOME ───────────────────────────────────────
def load_mean_connectome(gdf, gpath):
    matrices = []
    for _, row in gdf.iterrows():
        subj   = row['subject']
        conn_f = os.path.join(
            gpath, subj,
            f"{subj}_connectome.csv")
        if not os.path.exists(conn_f):
            continue
        try:
            mat = np.loadtxt(
                conn_f, delimiter=',')
            if mat.shape == (48, 48):
                matrices.append(mat)
        except Exception:
            pass
    if not matrices:
        return np.zeros((48, 48))
    mean = np.mean(matrices, axis=0)
    print(f"  Loaded {len(matrices)} "
          f"matrices, mean="
          f"{mean.mean():.2f}")
    return mean

# HO atlas approximate MNI coords (48 ROIs)
ho_coords = np.array([
    [-24,27,51],[26,28,50],
    [-31,36,22],[32,37,22],
    [-31,16,30],[31,16,30],
    [-24,-3,66],[25,-4,65],
    [-31,47,4],[32,46,4],
    [-46,26,5],[47,26,4],
    [-7,-12,66],[7,-11,66],
    [-12,-32,77],[13,-31,76],
    [-29,-40,65],[29,-38,63],
    [-46,-52,38],[46,-50,36],
    [-54,-53,22],[52,-50,23],
    [-30,-61,48],[31,-59,47],
    [-16,-65,50],[16,-63,49],
    [-20,-95,8],[20,-94,7],
    [-54,-19,11],[55,-18,11],
    [-56,-28,7],[56,-26,6],
    [-54,-55,-7],[55,-54,-7],
    [-28,-38,-13],[28,-37,-12],
    [-27,6,-22],[27,6,-22],
    [-50,14,-22],[50,13,-21],
    [-25,-15,-22],[25,-14,-22],
    [-8,-76,-36],[8,-75,-35],
    [-34,-79,-17],[34,-78,-16],
    [-3,-57,27],[3,-56,26],
    [-8,2,4],[8,2,4],
])[:48]

grp_order = ['CN', 'MCI', 'AD']

# ── BUILD FIGURE ───────────────────────────────────────────────
# Layout: top = glass brain (larger)
#         bottom = matrix
fig = plt.figure(
    figsize=(18, 12),
    facecolor='white')

gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    hspace=0.30,
    wspace=0.25,
    left=0.04,
    right=0.98,
    top=0.90,
    bottom=0.05)

print("\nLoading connectomes...")
mean_mats = {}
for grp in grp_order:
    gdf, gpath = groups_info[grp]
    mean_mats[grp] = \
        load_mean_connectome(
            gdf, gpath)

for ci, grp in enumerate(grp_order):
    mean_mat = mean_mats[grp]

    # ── Glass brain ──────────────────────
    ax_top = fig.add_subplot(gs[0, ci])

    # Threshold top 20% connections
    thresh_val = np.percentile(
        mean_mat[mean_mat > 0], 80) \
        if mean_mat.max() > 0 else 1

    mat_t = mean_mat.copy()
    mat_t[mat_t < thresh_val] = 0

    # Normalise for display
    if mat_t.max() > 0:
        mat_norm = mat_t / mat_t.max()
    else:
        mat_norm = mat_t

    try:
        disp = plotting.plot_connectome(
            mat_norm,
            ho_coords,
            node_size=25,
            node_color=group_colours[grp],
            edge_threshold="80%",
            edge_cmap=cmaps_conn[grp],
            edge_vmin=0.1,
            edge_vmax=1.0,
            display_mode='lyrz',
            colorbar=False,
            title='',
            axes=ax_top,
            figure=fig,
            annotate=False,
            black_bg=False,
            alpha=0.6)
    except Exception as e:
        print(f"  Glass brain error "
              f"{grp}: {e}")
        ax_top.text(
            0.5, 0.5,
            f'{grp} Connectome',
            ha='center', va='center',
            transform=ax_top.transAxes,
            fontsize=14,
            color=group_colours[grp],
            fontweight='bold')
        ax_top.set_facecolor('white')

    ax_top.set_title(
        f'{grp} '
        f'(n={len(mean_mats[grp])})\n'
        f'Top 20% connections shown',
        fontsize=12,
        fontweight='bold',
        color=group_colours[grp],
        pad=8)

    # ── Connectome matrix ─────────────────
    ax_bot = fig.add_subplot(gs[1, ci])

    # Log transform, zero diagonal
    log_mat = np.log1p(mean_mat)
    np.fill_diagonal(log_mat, 0)

    im = ax_bot.imshow(
        log_mat,
        cmap=cmaps_conn[grp],
        aspect='equal',
        interpolation='nearest',
        vmin=0,
        vmax=log_mat.max())

    # Module lines
    for m in range(1, 6):
        pos = m * 8 - 0.5
        ax_bot.axhline(
            pos, color='white',
            lw=1.2, alpha=0.7,
            zorder=3)
        ax_bot.axvline(
            pos, color='white',
            lw=1.2, alpha=0.7,
            zorder=3)

    # Stats
    off = mean_mat[
        ~np.eye(48, dtype=bool)]
    mean_c = off[off > 0].mean() \
        if off[off > 0].size > 0 else 0
    n_nonzero = np.sum(
        mean_mat > 0) // 2

    ax_bot.text(
        0.02, 0.97,
        f'Mean: {mean_c:.1f}\n'
        f'Connections: {n_nonzero}',
        ha='left', va='top',
        transform=ax_bot.transAxes,
        fontsize=8.5,
        fontweight='bold',
        color='white',
        bbox=dict(
            facecolor=group_colours[grp],
            alpha=0.85,
            boxstyle='round,pad=0.3'))

    cbar = plt.colorbar(
        im, ax=ax_bot,
        fraction=0.046, pad=0.04)
    cbar.set_label(
        'log(SIFT2 weight + 1)',
        fontsize=8.5, color='black')
    cbar.ax.tick_params(
        labelsize=7.5, colors='black')

    ax_bot.set_xlabel(
        'Brain Region (ROI)',
        fontsize=10)
    ax_bot.set_ylabel(
        'Brain Region (ROI)',
        fontsize=10)

    for sp in ax_bot.spines.values():
        sp.set_edgecolor(
            group_colours[grp])
        sp.set_linewidth(2.5)

fig.suptitle(
    "SIFT2-Filtered Structural Connectomes\n"
    "Glass Brain (top) and Mean "
    "Connectivity Matrix (bottom)\n"
    "48-ROI Harvard-Oxford Atlas",
    fontsize=13,
    fontweight='bold',
    color='#1A237E',
    y=0.97)

save_path = os.path.join(
    fig_path, 'fig2_connectome.png')
plt.savefig(
    save_path,
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none')
plt.close()
print(f"\n  Saved: {save_path}")
print("  Done!")
