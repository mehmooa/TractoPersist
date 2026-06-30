"""
TractoPersist — Step 2: Persistent Homology Feature Extraction
Computes H0/H1 topological features from each subject's
SIFT2-weighted connectome matrix using a Vietoris-Rips
filtration (Ripser).

Input : F:\ADNI_NIfTI\{AD_v2,MCI_v2,CN1}\<subject>\<subject>_connectome.csv
Output: F:\ADNI_Features\homology_features.csv

Requires: pip install ripser persim --break-system-packages

Run: python step2_persistent_homology.py
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from ripser import ripser
except ImportError:
    raise SystemExit(
        "Missing dependency. Install with:\n"
        "  pip install ripser persim --break-system-packages")

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
print("  Step 2: Persistent Homology Feature Extraction")
print("=" * 60)


def persistence_features(mat):
    """
    Run Vietoris-Rips persistent homology on a
    SIFT2-weighted connectome matrix and derive
    13 scalar H0/H1 descriptors.
    """
    # Convert connectivity -> distance
    # (high connectivity = low topological distance)
    mat_norm = mat / (mat.max() + 1e-10)
    dist = 1.0 - mat_norm
    np.fill_diagonal(dist, 0.0)
    # Ensure symmetry (numerical safety)
    dist = (dist + dist.T) / 2.0

    result = ripser(
        dist, distance_matrix=True, maxdim=1)
    dgms = result['dgms']
    h0 = dgms[0]
    h1 = dgms[1]

    # H0: drop the single infinite-lifetime component
    h0_finite = h0[np.isfinite(h0[:, 1])]
    h0_life = h0_finite[:, 1] - h0_finite[:, 0]

    h1_life = h1[:, 1] - h1[:, 0]
    h1_life = h1_life[np.isfinite(h1_life)]

    def entropy(life):
        if len(life) == 0 or life.sum() == 0:
            return 0.0
        p = life / life.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    feats = {
        # H0 (connected components)
        'h0_count':              len(h0_finite),
        'h0_mean_lifetime':      h0_life.mean()
            if len(h0_life) else 0.0,
        'h0_max_lifetime':       h0_life.max()
            if len(h0_life) else 0.0,
        'h0_sum_lifetime':       h0_life.sum()
            if len(h0_life) else 0.0,
        'h0_entropy':            entropy(h0_life),
        'betti_0':               len(h0_finite) + 1,
        'total_persistence_h0':  h0_life.sum()
            if len(h0_life) else 0.0,

        # H1 (topological loops)
        'h1_count':              len(h1_life),
        'h1_mean_lifetime':      h1_life.mean()
            if len(h1_life) else 0.0,
        'h1_max_lifetime':       h1_life.max()
            if len(h1_life) else 0.0,
        'h1_sum_lifetime':       h1_life.sum()
            if len(h1_life) else 0.0,
        'h1_entropy':            entropy(h1_life),
        'betti_1':               len(h1_life),
        'total_persistence_h1':  h1_life.sum()
            if len(h1_life) else 0.0,
    }
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

        try:
            feats = persistence_features(mat)
        except Exception as e:
            print(f"    ERROR {subject}: {e}")
            continue

        feats['subject'] = subject
        feats['group']   = group
        rows.append(feats)
        print(f"    OK: {subject}  "
              f"(H0={feats['h0_count']}, "
              f"H1={feats['h1_count']})")

df = pd.DataFrame(rows)
cols = ['subject', 'group'] + \
       [c for c in df.columns
        if c not in ('subject', 'group')]
df = df[cols]

out_csv = os.path.join(
    output_path, 'homology_features.csv')
df.to_csv(out_csv, index=False)

print("\n" + "=" * 60)
print(f"  Subjects processed: {len(df)}")
print(f"  Features per subject: {len(df.columns)-2}")
print(f"  Saved: {out_csv}")
print("=" * 60)
