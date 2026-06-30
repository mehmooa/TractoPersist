"""
TractoPersist - Figure 1: Fixed Brain Maps
Fixes MD/RD black image issue
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
print("  Figure 1: Real Brain Maps (Fixed)")
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

group_labels = {
    'CN':  'Cognitively Normal\n(CN, n=100)',
    'MCI': 'Mild Cognitive Impairment\n(MCI, n=89)',
    'AD':  "Alzheimer's Disease\n(AD, n=25)",
}

def get_median_subject(df):
    median_fa = df['fa_mean'].median()
    idx = (df['fa_mean'] -
           median_fa).abs().argmin()
    return df.iloc[idx]['subject']

def load_best_slice(nii_path, metric):
    """
    Load the best axial slice from NIfTI
    Handles different value ranges properly
    """
    img  = nib.load(nii_path)
    data = img.get_fdata()
    data = np.nan_to_num(data, 0)
    data[data < 0] = 0

    # Find slice with most brain content
    # rather than just middle slice
    n_slices = data.shape[2]
    best_slice = n_slices // 2

    # Search around middle for best slice
    best_content = 0
    for z in range(
            n_slices//3,
            2*n_slices//3):
        sl = data[:, :, z]
        # Count non-zero voxels
        if metric == 'FA':
            content = np.sum(sl > 0.05)
        else:
            # MD/RD are in mm2/s range
            # 0.0001 to 0.005
            content = np.sum(sl > 0.0001)
        if content > best_content:
            best_content = content
            best_slice = z

    sl = data[:, :, best_slice]
    sl = np.rot90(sl)
    return sl

# ── METRIC DEFINITIONS ─────────────────────────────────────────
# Key fix: MD and RD values are ~0.001
# not 0-1 like FA!
metric_info = [
    ('FA',
     '_FA.nii.gz',
     'hot',
     0.05,   # vmin - ignore background
     0.80,   # vmax
     'Fractional Anisotropy (FA)',
     0.05),  # mask threshold
    ('MD',
     '_MD.nii.gz',
     'Blues',
     0.0003, # vmin for MD
     0.0030, # vmax for MD (mm2/s)
     'Mean Diffusivity (×10⁻³ mm²/s)',
     0.0001),# mask threshold for MD
    ('RD',
     '_RD.nii.gz',
     'Reds',
     0.0001, # vmin for RD
     0.0025, # vmax for RD
     'Radial Diffusivity (×10⁻³ mm²/s)',
     0.0001),# mask threshold for RD
]

# ── BUILD FIGURE ───────────────────────────────────────────────
fig = plt.figure(
    figsize=(14, 10),
    facecolor='white')

# 3 rows (metrics) x 3 cols (groups)
# plus space for row labels
gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    hspace=0.12,
    wspace=0.08,
    left=0.10,
    right=0.94,
    top=0.88,
    bottom=0.06)

grp_order = ['CN', 'MCI', 'AD']

# Column headers
for ci, grp in enumerate(grp_order):
    fig.text(
        0.10 + ci * 0.285,
        0.92,
        group_labels[grp],
        ha='left',
        va='center',
        fontsize=12,
        fontweight='bold',
        color=group_colours[grp])

# Row labels
row_y = [0.78, 0.50, 0.22]
for ri, (metric, _, _, _, _, _, _) \
        in enumerate(metric_info):
    fig.text(
        0.02,
        row_y[ri],
        metric,
        ha='center',
        va='center',
        fontsize=13,
        fontweight='bold',
        color='#1A237E',
        rotation=90)

for ri, (metric, suffix,
          cmap, vmin, vmax,
          mlabel,
          mask_thresh) in \
        enumerate(metric_info):

    for ci, grp in enumerate(grp_order):
        gdf, gpath = groups_info[grp]
        subj = get_median_subject(gdf)
        subj_dir = os.path.join(
            gpath, subj)
        nii_file = os.path.join(
            subj_dir,
            f"{subj}{suffix}")

        ax = fig.add_subplot(gs[ri, ci])
        ax.set_facecolor('black')

        try:
            sl = load_best_slice(
                nii_file, metric)

            # Proper masking per metric
            sl_m = np.ma.masked_where(
                sl <= mask_thresh, sl)

            # Check if we have data
            valid = sl[sl > mask_thresh]
            if len(valid) == 0:
                raise ValueError(
                    "No valid data found")

            print(f"  {grp} {metric}: "
                  f"min={valid.min():.6f} "
                  f"max={valid.max():.6f} "
                  f"mean={valid.mean():.6f}")

            im = ax.imshow(
                sl_m,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect='equal',
                interpolation='bilinear',
                origin='lower')

            # Mean value
            mean_v = valid.mean()
            # Format based on metric
            if metric == 'FA':
                fmt = f'μ = {mean_v:.4f}'
            else:
                # Show in ×10⁻³
                fmt = f'μ = {mean_v*1000:.4f}×10⁻³'

            ax.text(
                0.97, 0.04,
                fmt,
                ha='right', va='bottom',
                transform=ax.transAxes,
                fontsize=8,
                color='white',
                fontweight='bold',
                bbox=dict(
                    facecolor='black',
                    alpha=0.55,
                    pad=2,
                    boxstyle='round'))

            # Colorbar on right column
            if ci == 2:
                cbar = plt.colorbar(
                    im, ax=ax,
                    fraction=0.046,
                    pad=0.03,
                    aspect=20)
                cbar.ax.tick_params(
                    colors='black',
                    labelsize=7)
                cbar.set_label(
                    mlabel,
                    color='black',
                    fontsize=7.5,
                    rotation=270,
                    labelpad=14)

        except FileNotFoundError:
            ax.text(
                0.5, 0.5,
                f'File not found\n{subj}',
                ha='center', va='center',
                transform=ax.transAxes,
                color='white', fontsize=8)
        except Exception as e:
            print(f"  ERROR {grp} "
                  f"{metric}: {e}")
            ax.text(
                0.5, 0.5,
                f'Error loading\n{metric}',
                ha='center', va='center',
                transform=ax.transAxes,
                color='yellow', fontsize=9)

        ax.set_xticks([])
        ax.set_yticks([])

        for sp in ax.spines.values():
            sp.set_edgecolor(
                group_colours[grp])
            sp.set_linewidth(2.5)

# Footer
fig.text(
    0.50, 0.015,
    'MD: p = 0.0004 *** (AD vs CN)  |  '
    'RD: p = 0.0005 *** (AD vs CN)  |  '
    'FA: p = 0.560 ns (global level)',
    ha='center', fontsize=9,
    color='#1A237E', style='italic')

fig.suptitle(
    "DTI Microstructure Maps — "
    "Representative Subjects "
    "(Median FA Subject per Group, "
    "Axial Slice)",
    fontsize=13,
    fontweight='bold',
    color='#1A237E',
    y=0.97)

save_path = os.path.join(
    fig_path, 'fig1_brain_maps.png')
plt.savefig(
    save_path,
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none')
plt.close()
print(f"\n  Saved: {save_path}")
print("  Done!")
