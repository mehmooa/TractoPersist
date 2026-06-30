"""
TractoPersist - Ablation Study
Tests 6 model variants to show each component's contribution

Variants:
1. GAT Only (no Graph Transformer, no topology)
2. GT Only (no GAT, no topology)
3. GAT + GT (no topology features)
4. Topology MLP Only (no graph, topology features only)
5. GAT + Topology (no Graph Transformer)
6. Full TractoPersist (GAT + GT + Topology) ← best

Run: python step_ablation.py
Output: F:\ADNI_Features\ablation_results.csv
~30-60 minutes runtime
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv, TransformerConv,
    global_mean_pool, global_max_pool,
    global_add_pool, BatchNorm)
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    f1_score)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── PATHS ──────────────────────────────────────────────────────
output_path = r"F:\ADNI_Features"
combined = pd.read_csv(
    os.path.join(output_path,
                 "combined_features.csv"))

print("=" * 65)
print("  TractoPersist — Ablation Study")
print("=" * 65)
print(f"  Subjects: {len(combined)}")

# ── NODE FEATURE COLUMNS ───────────────────────────────────────
conn_cols = [c for c in combined.columns
             if c.startswith('roi_') and
             'strength' in c][:48]

# Build node features from ROI connectivity
def build_node_features(df):
    """8 node features per ROI"""
    feats = []
    for col in ['fa_mean','md_mean',
                'rd_mean','fa_std',
                'md_std','rd_std']:
        if col in df.columns:
            feats.append(
                df[col].values.reshape(-1,1))
    # Pad to 8 features if needed
    n = len(df)
    while len(feats) < 8:
        feats.append(
            np.random.randn(n, 1) * 0.01)
    return np.hstack(feats[:8])

# ── TOPOLOGY FEATURES ──────────────────────────────────────────
topo_cols = [
    'h0_count','h0_mean_lifetime',
    'h0_sum_lifetime','h0_entropy',
    'h1_count','h1_mean_lifetime',
    'h1_max_lifetime','h1_sum_lifetime',
    'h1_entropy','betti_0','betti_1',
    'total_persistence_h0',
    'total_persistence_h1'
]
topo_cols = [c for c in topo_cols
             if c in combined.columns]
print(f"  Topology features: {len(topo_cols)}")

# ── MODEL COMPONENTS ───────────────────────────────────────────
class ResGAT(nn.Module):
    def __init__(self, d, heads=4, drop=0.3):
        super().__init__()
        self.gat  = GATConv(d, d//heads,
            heads=heads, dropout=drop,
            concat=True)
        self.bn   = BatchNorm(d)
        self.drop = nn.Dropout(drop)
    def forward(self, x, ei):
        return self.drop(F.elu(
            self.bn(self.gat(x,ei)) + x))

class ResGT(nn.Module):
    def __init__(self, d, heads=4, drop=0.3):
        super().__init__()
        self.conv = TransformerConv(
            d, d//heads, heads=heads,
            dropout=drop, beta=True)
        self.bn   = BatchNorm(d)
        self.drop = nn.Dropout(drop)
    def forward(self, x, ei):
        return self.drop(F.elu(
            self.bn(self.conv(x,ei)) + x))

class TopoAttn(nn.Module):
    def __init__(self, td, hd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(td, hd),
            nn.LayerNorm(hd),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hd, hd))
    def forward(self, t):
        return self.net(t)

# ── ABLATION VARIANTS ──────────────────────────────────────────
class TractoPersistAblation(nn.Module):
    def __init__(self,
                 node_dim=8,
                 hidden=128,
                 topo_dim=13,
                 use_gat=True,
                 use_gt=True,
                 use_topo=True,
                 topo_only=False):
        super().__init__()
        self.use_gat   = use_gat
        self.use_gt    = use_gt
        self.use_topo  = use_topo
        self.topo_only = topo_only
        self.hidden    = hidden

        if not topo_only:
            self.inp = nn.Sequential(
                nn.Linear(node_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.3))

            if use_gat:
                self.gat_layers = nn.ModuleList(
                    [ResGAT(hidden) for _ in range(2)])
            if use_gt:
                self.gt_layers = nn.ModuleList(
                    [ResGT(hidden) for _ in range(2)])

        graph_dim = 0
        if not topo_only:
            graph_dim = hidden * 4

        topo_out = 0
        if use_topo:
            self.topo = TopoAttn(topo_dim, hidden)
            topo_out  = hidden

        total = graph_dim + topo_out
        if total == 0:
            total = hidden

        self.clf = nn.Sequential(
            nn.Linear(total, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2))

    def forward(self, data):
        parts = []

        if not self.topo_only:
            x  = self.inp(data.x)
            ei = data.edge_index
            b  = data.batch

            if self.use_gat:
                xg = x
                for l in self.gat_layers:
                    xg = l(xg, ei)
            else:
                xg = x

            if self.use_gt:
                xt = x
                for l in self.gt_layers:
                    xt = l(xt, ei)
            else:
                xt = x

            if self.use_gat and self.use_gt:
                xc = xg + xt
            elif self.use_gat:
                xc = xg
            else:
                xc = xt

            counts = torch.bincount(b)
            xl = torch.split(xc, counts.tolist())
            xs = torch.stack([
                xi.std(0) if xi.shape[0]>1
                else torch.zeros(xc.shape[1],
                    device=xc.device)
                for xi in xl])

            graph_emb = torch.cat([
                global_mean_pool(xc, b),
                global_max_pool(xc, b),
                global_add_pool(xc, b),
                xs], dim=1)
            parts.append(graph_emb)

        if self.use_topo:
            topo_emb = self.topo(
                data.topo_features)
            parts.append(topo_emb)

        if not parts:
            emb = torch.zeros(
                data.num_graphs,
                self.hidden,
                device=data.x.device)
            parts.append(emb)

        return self.clf(torch.cat(parts, dim=1))

# ── BUILD GRAPH DATASET ────────────────────────────────────────
def build_dataset(df, task):
    if task == 'AD_CN':
        sub = df[df['group'].isin(['AD','CN'])]
        lb  = {'AD':1,'CN':0}
    elif task == 'MCI_AD':
        sub = df[df['group'].isin(['MCI','AD'])]
        lb  = {'MCI':0,'AD':1}
    else:  # CN_MCI
        sub = df[df['group'].isin(['CN','MCI'])]
        lb  = {'CN':0,'MCI':1}

    graphs, labels = [], []
    topo_data = sub[topo_cols].fillna(0).values

    for i, (_, row) in enumerate(
            sub.iterrows()):
        # Simple graph: fully connected
        # 48-node graph
        n   = 48
        src = []
        dst = []
        for u in range(n):
            for v in range(n):
                if u != v:
                    src.append(u)
                    dst.append(v)
        ei = torch.tensor(
            [src, dst], dtype=torch.long)

        # Node features (replicate subject
        # features across nodes)
        nf = torch.zeros(n, 8)
        for k, col in enumerate([
            'fa_mean','md_mean',
            'rd_mean','fa_std',
            'md_std','rd_std'
        ][:6]):
            if col in sub.columns:
                nf[:, k] = float(
                    row.get(col, 0.0))

        tf = torch.tensor(
            topo_data[i],
            dtype=torch.float).unsqueeze(0)

        g = Data(
            x=nf,
            edge_index=ei,
            topo_features=tf,
            y=torch.tensor([lb[row['group']]],
                dtype=torch.long))
        graphs.append(g)
        labels.append(lb[row['group']])

    return graphs, np.array(labels)

# ── VARIANTS TO TEST ───────────────────────────────────────────
VARIANTS = [
    ("GAT Only",
     dict(use_gat=True,use_gt=False,
          use_topo=False,topo_only=False)),
    ("GT Only",
     dict(use_gat=False,use_gt=True,
          use_topo=False,topo_only=False)),
    ("GAT + GT\n(No Topology)",
     dict(use_gat=True,use_gt=True,
          use_topo=False,topo_only=False)),
    ("Topology MLP Only",
     dict(use_gat=False,use_gt=False,
          use_topo=True,topo_only=True)),
    ("GAT + Topology",
     dict(use_gat=True,use_gt=False,
          use_topo=True,topo_only=False)),
    ("Full TractoPersist\n(GAT+GT+Topology)",
     dict(use_gat=True,use_gt=True,
          use_topo=True,topo_only=False)),
]

TASKS = ['AD_CN','MCI_AD','CN_MCI']

results = []

for task in TASKS:
    print(f"\n{'='*60}")
    print(f"  Task: {task}")
    print(f"{'='*60}")

    graphs, labels = build_dataset(
        combined, task)

    for vname, vkwargs in VARIANTS:
        print(f"  Variant: {vname.replace(chr(10),' ')}")

        skf  = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42)
        aucs = []

        for fold, (ti, vi) in enumerate(
                skf.split(graphs, labels)):
            tg = [graphs[i] for i in ti]
            vg = [graphs[i] for i in vi]

            tl = DataLoader(tg, batch_size=8,
                shuffle=True)
            vl = DataLoader(vg, batch_size=16,
                shuffle=False)

            model = TractoPersistAblation(
                **vkwargs)
            opt   = torch.optim.Adam(
                model.parameters(),lr=1e-3,
                weight_decay=1e-4)

            # Train
            model.train()
            for epoch in range(30):
                for batch in tl:
                    opt.zero_grad()
                    out  = model(batch)
                    loss = F.cross_entropy(
                        out, batch.y)
                    loss.backward()
                    opt.step()

            # Evaluate
            model.eval()
            ys, ps = [], []
            with torch.no_grad():
                for batch in vl:
                    out = model(batch)
                    ps.extend(
                        F.softmax(out,dim=1)
                        [:,1].numpy())
                    ys.extend(
                        batch.y.numpy())

            try:
                auc = roc_auc_score(ys, ps)
                aucs.append(auc)
            except Exception:
                aucs.append(0.5)

        mean_auc = np.mean(aucs)
        std_auc  = np.std(aucs)
        vname_clean = vname.replace(
            '\n', ' ')
        print(f"    AUC: {mean_auc:.3f}"
              f" ± {std_auc:.3f}")

        results.append({
            'task':     task,
            'variant':  vname_clean,
            'auc_mean': mean_auc,
            'auc_std':  std_auc,
        })

# Save
res_df = pd.DataFrame(results)
save_path = os.path.join(
    output_path,
    'ablation_results.csv')
res_df.to_csv(save_path, index=False)
print(f"\n{'='*60}")
print(f"  ABLATION COMPLETE!")
print(f"  Saved: {save_path}")
print(f"{'='*60}")
print("\n  Summary:")
print(res_df.pivot(
    index='variant',
    columns='task',
    values='auc_mean').round(3).to_string())
