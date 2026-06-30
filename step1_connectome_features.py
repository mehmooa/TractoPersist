"""
TractoPersist — Step 1: Connectome Feature Extraction
Loads per-subject SIFT2-weighted connectome matrices
(produced by pipeline.sh) and computes graph-theoretic
node/global features for every subject.

Input : F:\ADNI_NIfTI\{AD_v2,MCI_v2,CN1}\<subject>\<subject>_connectome.csv
Output: F:\ADNI_Features\connectome_features.csv

Run: python step1_connectome_features.py
"""

import os
import numpy as np
import pandas as pd
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

N_ROI = 48  # Harvard-Oxford cortical atlas

print("=" * 60)
print("  Step 1: Connectome Feature Extraction")
print("=" * 60)


def graph_features(mat):
    """
    Compute node-level and global graph features
    from a symmetric weighted connectivity matrix.
    Returns a flat dict of features.
    """
    n = mat.shape[0]
    feats = {}

    # Node strength (weighted degree)
    strength = mat.sum(axis=1)
    for i in range(n):
        feats[f'roi_str_{i}'] = strength[i]

    # Node degree (binary, threshold > 0)
    degree = (mat > 0).sum(axis=1)
    for i in range(n):
        feats[f'roi_deg_{i}'] = degree[i]

    # Flattened upper-triangular connectome
    # (used downstream by persistent homology step)
    iu = np.triu_indices(n, k=1)
    conn_flat = mat[iu]
    for i, v in enumerate(conn_flat):
        feats[f'conn_{i}'] = v

    # Global summary metrics
    off_diag = mat[~np.eye(n, dtype=bool)]
    feats['global_strength']    = strength.mean()
    feats['global_density']     = (mat > 0).sum() / (n * (n - 1))
    feats['global_efficiency']  = np.mean(
        1.0 / (mat[mat > 0])) if (mat > 0).any() else 0.0
    feats['mean_edge_weight']   = off_diag[off_diag > 0].mean() \
        if (off_diag > 0).any() else 0.0
    feats['max_edge_weight']    = off_diag.max()
    feats['n_nonzero_edges']    = int((mat > 0).sum() / 2)

    return feats


rows = []

for group, gpath in groups.items():
    if not os.path.isdir(gpath):
        print(f"  SKIP missing group dir: {gpath}")
        continue

    print(f"\n  Group: {group}")
    subjects = sorted(os.listdir(gpath))

    for subject in subjects:
        sdir = os.path.join(gpath, subject)
        conn_file = os.path.join(
            sdir, f"{subject}_connectome.csv")

        if not os.path.isfile(conn_file):
            continue

        try:
            mat = np.loadtxt(conn_file, delimiter=',')
        except Exception as e:
            print(f"    SKIP {subject}: {e}")
            continue

        if mat.shape != (N_ROI, N_ROI):
            print(f"    SKIP {subject}: "
                  f"unexpected shape {mat.shape}")
            continue

        feats = graph_features(mat)
        feats['subject'] = subject
        feats['group']   = group
        rows.append(feats)
        print(f"    OK: {subject}")

df = pd.DataFrame(rows)

# Reorder: subject, group first
cols = ['subject', 'group'] + \
       [c for c in df.columns
        if c not in ('subject', 'group')]
df = df[cols]

out_csv = os.path.join(
    output_path, 'connectome_features.csv')
df.to_csv(out_csv, index=False)

print("\n" + "=" * 60)
print(f"  Subjects processed: {len(df)}")
print(f"  Features per subject: {len(df.columns)-2}")
print(f"  Saved: {out_csv}")
print("=" * 60)
