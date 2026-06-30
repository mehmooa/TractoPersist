"""
TractoPersist — Step 3: Combine Features
Merges connectome graph features, persistent homology
features, and ROI-level FA/MD/RD microstructure metrics
into a single subject-level feature table.

Inputs:
  F:\ADNI_Features\connectome_features.csv   (from step1)
  F:\ADNI_Features\homology_features.csv     (from step2)
  F:\ADNI_NIfTI\{AD_v2,MCI_v2,CN1}\<subject>\<subject>_{FA,MD,RD}.nii.gz
  F:\ADNI_NIfTI\{AD_v2,MCI_v2,CN1}\<subject>\<subject>_atlas.nii.gz
      (48-ROI Harvard-Oxford atlas resampled to subject space —
       produced by extract_metrics.sh / register_atlas.sh)

Output: F:\ADNI_Features\combined_features.csv

Run: python step3_combine_features.py
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import warnings
warnings.filterwarnings('ignore')

# ── PATHS ──────────────────────────────────────────────────
base_path   = r"F:\ADNI_NIfTI"
output_path = r"F:\ADNI_Features"
os.makedirs(output_path, exist_ok=True)

groups = {
    'AD':  os.path.join(base_path, 'AD_v2'),
    'MCI': os.path.join(base_path, 'MCI_v2'),
    'CN':  os.path.join(base_path, 'CN1'),
}

N_ROI = 48

print("=" * 60)
print("  Step 3: Combine Features")
print("=" * 60)

# ── LOAD STEP 1 + STEP 2 OUTPUTS ───────────────────────────
conn_csv = os.path.join(
    output_path, 'connectome_features.csv')
homo_csv = os.path.join(
    output_path, 'homology_features.csv')

if not os.path.isfile(conn_csv):
    raise SystemExit(
        f"Missing {conn_csv} — run step1 first.")
if not os.path.isfile(homo_csv):
    raise SystemExit(
        f"Missing {homo_csv} — run step2 first.")

conn_df = pd.read_csv(conn_csv)
homo_df = pd.read_csv(homo_csv)

print(f"  Connectome features: {conn_df.shape}")
print(f"  Homology features:   {homo_df.shape}")


# ── ROI-LEVEL FA/MD/RD EXTRACTION ──────────────────────────
def extract_roi_metrics(subject_dir, subject):
    """
    Load FA, MD, RD maps and the 48-ROI atlas
    (already resampled to native diffusion space by
    extract_metrics.sh) and compute per-ROI mean values
    plus whole-brain summary stats.
    Returns a flat dict, or None if files are missing.
    """
    fa_f = os.path.join(
        subject_dir, f"{subject}_FA.nii.gz")
    md_f = os.path.join(
        subject_dir, f"{subject}_MD.nii.gz")
    rd_f = os.path.join(
        subject_dir, f"{subject}_RD.nii.gz")
    atlas_f = os.path.join(
        subject_dir, f"{subject}_atlas.nii.gz")

    if not all(os.path.isfile(f) for f in
               [fa_f, md_f, rd_f]):
        return None

    fa = nib.load(fa_f).get_fdata()
    md = nib.load(md_f).get_fdata()
    rd = nib.load(rd_f).get_fdata()

    feats = {}

    if os.path.isfile(atlas_f):
        atlas = nib.load(atlas_f).get_fdata()
        for roi in range(1, N_ROI + 1):
            roi_mask = atlas == roi
            i = roi - 1
            if roi_mask.sum() > 0:
                feats[f'roi_fa_{i}'] = \
                    fa[roi_mask].mean()
                feats[f'roi_md_{i}'] = \
                    md[roi_mask].mean()
                feats[f'roi_rd_{i}'] = \
                    rd[roi_mask].mean()
            else:
                feats[f'roi_fa_{i}'] = np.nan
                feats[f'roi_md_{i}'] = np.nan
                feats[f'roi_rd_{i}'] = np.nan
    else:
        # Atlas not available — skip per-ROI breakdown
        print(f"    (no atlas for {subject}, "
              f"global metrics only)")

    # Whole-brain summary (excludes background, FA<0.05)
    brain_mask = fa > 0.05
    feats['fa_mean'] = fa[brain_mask].mean()
    feats['fa_std']  = fa[brain_mask].std()
    feats['md_mean'] = md[brain_mask].mean()
    feats['md_std']  = md[brain_mask].std()
    feats['rd_mean'] = rd[brain_mask].mean()
    feats['rd_std']  = rd[brain_mask].std()

    return feats


dti_rows = []
for group, gpath in groups.items():
    if not os.path.isdir(gpath):
        continue
    print(f"\n  Group: {group}")
    for subject in sorted(os.listdir(gpath)):
        sdir = os.path.join(gpath, subject)
        feats = extract_roi_metrics(sdir, subject)
        if feats is None:
            continue
        feats['subject'] = subject
        dti_rows.append(feats)
        print(f"    OK: {subject}")

dti_df = pd.DataFrame(dti_rows)
print(f"\n  DTI microstructure features: {dti_df.shape}")

# ── MERGE ALL THREE SOURCES ────────────────────────────────
merged = conn_df.merge(
    homo_df.drop(columns=['group']),
    on='subject', how='inner')
merged = merged.merge(
    dti_df, on='subject', how='inner')

print(f"\n  Connectome features: {conn_df.shape}")
print(f"  Homology features:   {homo_df.shape}")
print(f"  Combined features:   {merged.shape}")
print(f"  Subjects:            {len(merged)}")
print(f"  Total features:      {merged.shape[1]}")

print("\n  Subjects per group:")
print(merged['group'].value_counts())

out_csv = os.path.join(
    output_path, 'combined_features.csv')
merged.to_csv(out_csv, index=False)
print(f"\n  Saved: {out_csv}")

# Quick sanity check on key clinical features
print("\n  Key features by group (sanity check):")
for feat in ['fa_mean', 'md_mean', 'rd_mean',
             'h1_count', 'h1_mean_lifetime']:
    if feat in merged.columns:
        print(f"\n  {feat}:")
        print(merged.groupby('group')[feat]
              .agg(['mean', 'std']).round(4))
