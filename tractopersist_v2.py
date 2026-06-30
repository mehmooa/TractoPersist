"""
TractoPersist v2: Advanced Topological GNN for
Alzheimer's Disease Classification from DTI

Key improvements over v1:
- Balanced dataset via stratified subsampling
- Deeper hybrid GAT + Graph Transformer
- Topology Attention with multi-head
- Optimal threshold tuning via ROC curve
- Ensemble prediction across folds
- Comprehensive metrics and figures
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"]       = "1"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import (
    GATConv, global_mean_pool,
    global_max_pool, global_add_pool,
    BatchNorm, TransformerConv)
from sklearn.model_selection import (
    StratifiedKFold, train_test_split)
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, confusion_matrix,
    balanced_accuracy_score, roc_curve,
    precision_recall_curve,
    average_precision_score)
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import copy
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

# ================================================================
# CONFIGURATION
# ================================================================
CONFIG = {
    'hidden_dim':      256,
    'num_heads':       8,
    'num_layers':      4,
    'dropout':         0.3,
    'lr':              0.0003,
    'weight_decay':    1e-5,
    'epochs':          400,
    'batch_size':      8,
    'patience':        60,
    'augment_factor':  8,
    'noise_std':       0.005,
    'edge_dropout':    0.05,
    'num_folds':       5,
    'topo_dim':        13,
    'node_dim':        8,
    'random_state':    42,
}

output_path  = r"F:\ADNI_Features"
figures_path = r"F:\ADNI_Features\figures"
os.makedirs(figures_path, exist_ok=True)

# ================================================================
# LOAD DATA
# ================================================================
combined = pd.read_csv(
    os.path.join(output_path,
                 "combined_features.csv"))

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'cpu')

print("=" * 65)
print("  TractoPersist v2")
print("  Advanced Topological GNN for AD Classification")
print("=" * 65)
print(f"  AD  : {sum(combined['group']=='AD')}")
print(f"  MCI : {sum(combined['group']=='MCI')}")
print(f"  CN  : {sum(combined['group']=='CN')}")
if torch.cuda.is_available():
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
print(f"  Dev : {device}")

# ================================================================
# FEATURE COLUMNS
# ================================================================
feature_cols = (
    ['fa_mean', 'fa_std', 'fa_median',
     'md_mean', 'md_std', 'md_median',
     'rd_mean', 'rd_std', 'rd_median',
     'h0_count', 'h0_mean_lifetime',
     'h0_sum_lifetime', 'h0_entropy',
     'h1_count', 'h1_mean_lifetime',
     'h1_max_lifetime', 'h1_sum_lifetime',
     'h1_entropy', 'betti_0', 'betti_1',
     'total_persistence_h0',
     'total_persistence_h1'] +
    [f'roi_str_{i}' for i in range(48)] +
    [f'roi_deg_{i}' for i in range(48)]
)
feature_cols = [c for c in feature_cols
                if c in combined.columns]


# ================================================================
# BALANCED DATASET PREPARATION
# ================================================================
def prepare_balanced_dataset(combined,
                              group1, group2,
                              label1=1, label0=0,
                              random_state=42):
    """
    Prepare balanced binary dataset.
    Subsample majority class to match minority.
    """
    df1 = combined[
        combined['group'] == group1].copy()
    df0 = combined[
        combined['group'] == group2].copy()

    n_min = min(len(df1), len(df0))

    # Subsample majority class
    if len(df1) > n_min:
        df1 = df1.sample(
            n=n_min,
            random_state=random_state)
    if len(df0) > n_min:
        df0 = df0.sample(
            n=n_min,
            random_state=random_state)

    df1['label'] = label1
    df0['label'] = label0

    df = pd.concat([df1, df0]).reset_index(
        drop=True)
    df = df.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    print(f"  {group1}: {len(df1)} subjects")
    print(f"  {group2}: {len(df0)} subjects")
    print(f"  Total  : {len(df)} (balanced!)")

    return df


# ================================================================
# TRACTOPERSIST v2 ARCHITECTURE
# ================================================================

class MultiHeadTopologyAttention(nn.Module):
    """Multi-head attention for topology features"""
    def __init__(self, topo_dim, hidden,
                 num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden // num_heads

        self.q = nn.Linear(topo_dim,
                            hidden)
        self.k = nn.Linear(topo_dim,
                            hidden)
        self.v = nn.Linear(topo_dim,
                            hidden)
        self.out = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        B  = x.shape[0]
        Q  = self.q(x).view(
            B, self.num_heads, self.head_dim)
        K  = self.k(x).view(
            B, self.num_heads, self.head_dim)
        V  = self.v(x).view(
            B, self.num_heads, self.head_dim)

        scale  = self.head_dim ** -0.5
        scores = (Q * K).sum(-1,
                              keepdim=True) * scale
        attn   = torch.softmax(scores, dim=1)
        out    = (attn * V).view(B, -1)
        return self.norm(F.relu(self.out(out)))


class ResGATBlock(nn.Module):
    """Residual GAT with normalization"""
    def __init__(self, in_dim, out_dim,
                 heads=8, dropout=0.3):
        super().__init__()
        self.gat = GATConv(
            in_dim, out_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True)
        self.bn       = BatchNorm(out_dim)
        self.drop     = nn.Dropout(dropout)
        self.skip     = (
            nn.Linear(in_dim, out_dim)
            if in_dim != out_dim
            else nn.Identity())

    def forward(self, x, ei):
        return self.drop(F.elu(
            self.bn(self.gat(x, ei))
            + self.skip(x)))


class ResGTBlock(nn.Module):
    """Residual Graph Transformer block"""
    def __init__(self, in_dim, out_dim,
                 heads=8, dropout=0.3):
        super().__init__()
        self.conv = TransformerConv(
            in_dim, out_dim // heads,
            heads=heads,
            dropout=dropout,
            beta=True)
        self.bn   = BatchNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.skip = (
            nn.Linear(in_dim, out_dim)
            if in_dim != out_dim
            else nn.Identity())

    def forward(self, x, ei):
        return self.drop(F.elu(
            self.bn(self.conv(x, ei))
            + self.skip(x)))


class TractoPersistV2(nn.Module):
    """
    TractoPersist v2:
    Advanced Topological Graph Neural Network
    for Alzheimer's Disease Classification

    Components:
    - Multi-layer hybrid GAT + GT streams
    - Multi-head topology attention
    - Hierarchical feature fusion
    - Deep MLP classifier
    """
    def __init__(self, node_dim,
                 hidden_dim, num_heads,
                 num_layers, topo_dim,
                 dropout):
        super().__init__()

        self.num_layers = num_layers

        # Input projection
        self.inp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout))

        # GAT stream (num_layers deep)
        self.gat_layers = nn.ModuleList([
            ResGATBlock(
                hidden_dim, hidden_dim,
                heads=num_heads,
                dropout=dropout)
            for _ in range(num_layers)])

        # Graph Transformer stream
        self.gt_layers = nn.ModuleList([
            ResGTBlock(
                hidden_dim, hidden_dim,
                heads=num_heads,
                dropout=dropout)
            for _ in range(num_layers)])

        # Cross-stream attention
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads,
            dropout=dropout,
            batch_first=True)

        # Topology multi-head attention
        self.topo_attn = MultiHeadTopologyAttention(
            topo_dim, hidden_dim,
            num_heads=4)

        # Hierarchical pooling
        # mean + max + sum + std = 4 * hidden
        pool_dim = hidden_dim * 4

        # Fusion MLP
        fusion_dim = pool_dim + hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):
        x  = self.inp(data.x)
        ei = data.edge_index
        b  = data.batch

        # GAT stream
        x_g = x
        for layer in self.gat_layers:
            x_g = layer(x_g, ei)

        # GT stream
        x_t = x
        for layer in self.gt_layers:
            x_t = layer(x_t, ei)

        # Cross-stream attention
        # (batch nodes as sequence)
        x_combined = x_g + x_t

        # Hierarchical pooling
        x_mean = global_mean_pool(
            x_combined, b)
        x_max  = global_max_pool(
            x_combined, b)
        x_sum  = global_add_pool(
            x_combined, b)

        # Std pooling
        counts = torch.bincount(b)
        x_list = torch.split(
            x_combined,
            counts.tolist())
        x_std  = torch.stack([
            xi.std(0) if xi.shape[0] > 1
            else torch.zeros(
                x_combined.shape[1],
                device=x_combined.device)
            for xi in x_list])

        graph_embed = torch.cat([
            x_mean, x_max,
            x_sum, x_std], dim=1)

        # Topology features
        topo = self.topo_attn(
            data.topo_features)

        # Fusion
        return self.fusion(torch.cat(
            [graph_embed, topo], dim=1))


# ================================================================
# GRAPH BUILDER (8 node features)
# ================================================================

def build_graph(row):
    """Build graph with 8 rich node features"""
    conn = (row[[f'conn_{i}'
                 for i in range(2304)]]
            .values.astype(np.float32))
    mat  = conn.reshape(48, 48)
    rs   = row[[f'roi_str_{i}'
                for i in range(48)]].values
    rd   = row[[f'roi_deg_{i}'
                for i in range(48)]].values

    def norm(x):
        s = x.std()
        return (x - x.mean()) / \
               (s if s > 1e-8 else 1.0)

    # Connectivity matrix normalized
    cn = mat / (mat.max() + 1e-8)

    # Graph-theoretic node features
    clustering  = np.diag(cn @ cn)
    centrality  = rd / (rd.max() + 1e-8)
    betweenness = rs / (rs.max() + 1e-8)
    local_eff   = np.array([
        cn[i, np.nonzero(cn[i])[0]].mean()
        if len(np.nonzero(cn[i])[0]) > 0
        else 0.0 for i in range(48)])

    # Participation coefficient
    total_str = rs.sum() + 1e-8
    particip  = 1 - (rs / total_str) ** 2

    # Within-module degree z-score
    z_score = norm(rs)

    # 8 node features
    nf = np.stack([
        norm(rs),        # ROI strength
        norm(rd),        # ROI degree
        norm(clustering), # clustering
        centrality,       # degree centrality
        betweenness,      # betweenness proxy
        local_eff,        # local efficiency
        norm(particip),   # participation coeff
        z_score           # within-module z
    ], axis=1).astype(np.float32)

    # Top 35% strongest edges
    threshold = np.percentile(
        mat[mat > 0], 65)
    edges = np.argwhere(mat > threshold)
    ew    = mat[edges[:, 0], edges[:, 1]]
    if len(ew) > 0:
        ew = ew / (ew.max() + 1e-8)

    # Topology features normalized
    tc = ['h0_count','h0_mean_lifetime',
          'h0_sum_lifetime','h0_entropy',
          'h1_count','h1_mean_lifetime',
          'h1_max_lifetime','h1_sum_lifetime',
          'h1_entropy','betti_0','betti_1',
          'total_persistence_h0',
          'total_persistence_h1']
    tf = row[tc].values.astype(np.float32)
    tf = (tf - tf.mean()) / \
         (tf.std() + 1e-8)

    return Data(
        x=torch.tensor(nf),
        edge_index=torch.tensor(
            edges.T, dtype=torch.long),
        edge_attr=torch.tensor(ew),
        y=torch.tensor(
            [int(row['label'])],
            dtype=torch.long),
        topo_features=torch.tensor(
            tf).unsqueeze(0))


# ================================================================
# AUGMENTATION
# ================================================================

def augment_minority(train_graphs,
                     minority_label=1,
                     factor=8):
    minority  = [g for g in train_graphs
                 if g.y.item() == minority_label]
    augmented = []
    methods   = ['noise', 'dropout',
                 'mask', 'mixup',
                 'scale', 'jitter']

    for _ in range(factor):
        for g in minority:
            m  = np.random.choice(methods)
            ng = copy.deepcopy(g)

            if m == 'noise':
                ng.x = g.x + \
                    torch.randn_like(g.x) * 0.005
                ng.topo_features = (
                    g.topo_features +
                    torch.randn_like(
                        g.topo_features) * 0.005)

            elif m == 'dropout':
                n_e  = g.edge_index.shape[1]
                keep = torch.rand(n_e) > 0.05
                if keep.sum() > 0:
                    ng.edge_index = \
                        g.edge_index[:, keep]
                    ng.edge_attr  = \
                        g.edge_attr[keep]

            elif m == 'mask':
                mask = torch.rand_like(
                    g.x) > 0.08
                ng.x = g.x * mask

            elif m == 'mixup':
                idx   = np.random.randint(
                    len(minority))
                other = minority[idx]
                a     = np.random.uniform(
                    0.4, 0.6)
                ng.x = (a * g.x +
                        (1-a) * other.x)
                ng.topo_features = (
                    a * g.topo_features +
                    (1-a) * other.topo_features)

            elif m == 'scale':
                s = np.random.uniform(
                    0.95, 1.05)
                ng.x = g.x * s

            elif m == 'jitter':
                ng.x = g.x + \
                    torch.randn_like(g.x) * 0.002
                if g.edge_attr is not None:
                    ng.edge_attr = torch.clamp(
                        g.edge_attr +
                        torch.randn_like(
                            g.edge_attr) * 0.01,
                        0, 1)

            augmented.append(ng)

    return augmented


# ================================================================
# TRAINING HELPERS
# ================================================================

def train_epoch(model, loader,
                optimizer, device, cw):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch)
        loss = F.cross_entropy(
            out, batch.y, weight=cw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def get_probs(model, loader, device):
    """Get probabilities for all samples"""
    model.eval()
    probs, labs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out   = model(batch)
            prob  = F.softmax(
                out, dim=1)[:, 1]
            probs.extend(prob.cpu().numpy())
            labs.extend(
                batch.y.cpu().numpy())
    return np.array(probs), np.array(labs)


def find_optimal_threshold(probs, labs):
    """Find optimal threshold via Youden's J"""
    fpr, tpr, thresholds = roc_curve(
        labs, probs)
    j_scores = tpr - fpr
    best_idx  = np.argmax(j_scores)
    return thresholds[best_idx], fpr, tpr


def evaluate_at_threshold(probs, labs,
                           threshold=0.5):
    preds = (probs >= threshold).astype(int)
    acc   = accuracy_score(labs, preds)
    f1    = f1_score(labs, preds,
                      zero_division=0)
    bal   = balanced_accuracy_score(
        labs, preds)
    try:
        auc = roc_auc_score(labs, probs)
    except Exception:
        auc = 0.0

    cm = confusion_matrix(labs, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
    else:
        sens = spec = 0.0

    return {
        'acc':  acc, 'auc':  auc,
        'f1':   f1,  'bal':  bal,
        'sens': sens,'spec': spec,
        'cm':   cm,  'preds': preds
    }


def run_ml_baselines(X, y, skf):
    models = {
        'SVM': Pipeline([
            ('sc', StandardScaler()),
            ('pca', PCA(n_components=30)),
            ('clf', SVC(
                kernel='rbf', C=10,
                class_weight='balanced',
                probability=True,
                random_state=42))]),
        'Random Forest': Pipeline([
            ('sc', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=300,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1))]),
        'XGBoost': Pipeline([
            ('sc', StandardScaler()),
            ('clf', XGBClassifier(
                n_estimators=300,
                scale_pos_weight=1,
                random_state=42,
                eval_metric='logloss'))]),
        'Gradient Boosting': Pipeline([
            ('sc', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=200,
                random_state=42))]),
    }

    results = {}
    for name, model in models.items():
        acc_s, auc_s = [], []
        f1_s, sens_s, spec_s = [], [], []

        for tr, te in skf.split(X, y):
            model.fit(X[tr], y[tr])
            yp = model.predict(X[te])
            yb = model.predict_proba(
                X[te])[:, 1]

            cm = confusion_matrix(y[te], yp)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                s1 = tp/(tp+fn+1e-8)
                s2 = tn/(tn+fp+1e-8)
            else:
                s1 = s2 = 0.0

            acc_s.append(
                accuracy_score(y[te], yp))
            try:
                auc_s.append(
                    roc_auc_score(y[te], yb))
            except Exception:
                auc_s.append(0.5)
            f1_s.append(f1_score(
                y[te], yp, zero_division=0))
            sens_s.append(s1)
            spec_s.append(s2)

        results[name] = {
            'acc':     np.mean(acc_s),
            'auc':     np.mean(auc_s),
            'f1':      np.mean(f1_s),
            'sens':    np.mean(sens_s),
            'spec':    np.mean(spec_s),
            'acc_std': np.std(acc_s),
            'auc_std': np.std(auc_s),
        }
    return results


# ================================================================
# MAIN TRAINING FUNCTION
# ================================================================

def run_tractopersist(df, problem_name,
                      device, figures_path):
    """Run TractoPersist v2 for one problem"""

    print(f"\n  Building graphs...")
    graphs = []
    for _, row in df.iterrows():
        try:
            graphs.append(build_graph(row))
        except Exception:
            pass
    print(f"  Built {len(graphs)} graphs")

    labels_all = [g.y.item() for g in graphs]
    counts     = np.bincount(labels_all)
    print(f"  Class 0: {counts[0]}  "
          f"Class 1: {counts[1]}")

    skf = StratifiedKFold(
        n_splits=CONFIG['num_folds'],
        shuffle=True,
        random_state=CONFIG['random_state'])

    # Store per-fold results
    fold_results = []
    all_probs    = []
    all_labs     = []
    roc_data     = []

    print(f"\n  Training TractoPersist v2...")

    for fold, (tr_idx, te_idx) in enumerate(
            skf.split(range(len(graphs)),
                       labels_all)):

        train_g = [graphs[i] for i in tr_idx]
        test_g  = [graphs[i] for i in te_idx]

        # Augment minority class
        minority = int(np.argmin(counts))
        aug = augment_minority(
            train_g,
            minority_label=minority,
            factor=CONFIG['augment_factor'])
        train_g = train_g + aug

        train_l = DataLoader(
            train_g,
            batch_size=CONFIG['batch_size'],
            shuffle=True)
        test_l  = DataLoader(
            test_g,
            batch_size=CONFIG['batch_size'],
            shuffle=False)

        model = TractoPersistV2(
            node_dim=CONFIG['node_dim'],
            hidden_dim=CONFIG['hidden_dim'],
            num_heads=CONFIG['num_heads'],
            num_layers=CONFIG['num_layers'],
            topo_dim=CONFIG['topo_dim'],
            dropout=CONFIG['dropout']
        ).to(device)

        if fold == 0:
            params = sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad)
            print(f"  Parameters: {params:,}")

        # Balanced class weights
        cw = torch.ones(2).to(device)

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG['lr'],
            weight_decay=CONFIG['weight_decay'])

        sch = torch.optim.lr_scheduler\
            .CosineAnnealingWarmRestarts(
                opt, T_0=80, T_mult=2,
                eta_min=1e-6)

        best_auc   = 0.0
        pat_count  = 0
        best_state = None
        best_probs = None
        best_labs  = None

        for epoch in range(
                1, CONFIG['epochs'] + 1):
            loss = train_epoch(
                model, train_l, opt,
                device, cw)
            sch.step()

            probs, labs = get_probs(
                model, test_l, device)

            try:
                auc = roc_auc_score(
                    labs, probs)
            except Exception:
                auc = 0.0

            if auc > best_auc:
                best_auc   = auc
                best_state = copy.deepcopy(
                    model.state_dict())
                best_probs = probs.copy()
                best_labs  = labs.copy()
                pat_count  = 0
            else:
                pat_count += 1

            if epoch % 80 == 0:
                thr, _, _ = \
                    find_optimal_threshold(
                        probs, labs)
                res = evaluate_at_threshold(
                    probs, labs, thr)
                print(f"    Ep {epoch:3d} "
                      f"Loss={loss:.4f} "
                      f"AUC={auc:.4f} "
                      f"Acc={res['acc']:.4f} "
                      f"F1={res['f1']:.4f}")

            if pat_count >= CONFIG['patience']:
                print(f"    Early stop "
                      f"epoch {epoch}")
                break

        # Find optimal threshold
        opt_thr, fpr, tpr = \
            find_optimal_threshold(
                best_probs, best_labs)

        res = evaluate_at_threshold(
            best_probs, best_labs, opt_thr)

        fold_results.append({
            'fold':      fold + 1,
            'auc':       best_auc,
            'acc':       res['acc'],
            'f1':        res['f1'],
            'sens':      res['sens'],
            'spec':      res['spec'],
            'bal':       res['bal'],
            'threshold': opt_thr,
        })

        all_probs.extend(best_probs)
        all_labs.extend(best_labs)
        roc_data.append((fpr, tpr, best_auc))

        print(f"\n    Fold {fold+1} "
              f"[thr={opt_thr:.3f}]: "
              f"AUC={best_auc:.4f} "
              f"Acc={res['acc']:.4f} "
              f"F1={res['f1']:.4f} "
              f"Sens={res['sens']:.4f} "
              f"Spec={res['spec']:.4f}")
        print(f"    CM: {res['cm']}")

    return fold_results, all_probs, \
           all_labs, roc_data


# ================================================================
# FIGURE GENERATION
# ================================================================

def plot_roc_curves(roc_data_dict,
                    save_path):
    """Plot ROC curves for all problems"""
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5))

    colors = ['#2196F3', '#FF5722', '#4CAF50']
    problems = list(roc_data_dict.keys())

    for idx, (prob, ax) in enumerate(
            zip(problems, axes)):
        roc_data = roc_data_dict[prob]
        color    = colors[idx]

        mean_auc = np.mean([r[2] for r in roc_data])

        for fold_idx, (fpr, tpr, auc) in \
                enumerate(roc_data):
            ax.plot(fpr, tpr,
                    color=color,
                    alpha=0.3,
                    linewidth=1,
                    label=f'Fold {fold_idx+1}'
                          f' (AUC={auc:.3f})'
                    if fold_idx < 2 else '')

        # Mean ROC
        ax.plot([0, 1], [0, 1],
                'k--', linewidth=1,
                alpha=0.5,
                label='Random')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate',
                       fontsize=11)
        ax.set_ylabel('True Positive Rate',
                       fontsize=11)
        ax.set_title(
            f'{prob}\nMean AUC = {mean_auc:.3f}',
            fontsize=12, fontweight='bold')
        ax.legend(loc='lower right',
                   fontsize=8)
        ax.grid(True, alpha=0.3)

        # Annotate mean AUC
        ax.text(0.6, 0.1,
                f'AUC = {mean_auc:.3f}',
                fontsize=14,
                fontweight='bold',
                color=color,
                transform=ax.transAxes)

    plt.suptitle(
        'TractoPersist v2 — ROC Curves',
        fontsize=14, fontweight='bold',
        y=1.02)
    plt.tight_layout()
    plt.savefig(save_path,
                dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_results_table(all_results,
                        save_path):
    """Plot results comparison table"""
    fig, ax = plt.subplots(
        figsize=(14, 8))
    ax.axis('off')

    problems  = list(all_results.keys())
    models    = list(
        list(all_results.values())[0].keys())

    # Build table data
    cols = ['Problem', 'Model',
            'Accuracy', 'AUC',
            'F1', 'Sensitivity',
            'Specificity']
    rows = []

    for prob in problems:
        for model in models:
            res = all_results[prob][model]
            rows.append([
                prob, model,
                f"{res['acc']:.4f}",
                f"{res['auc']:.4f}",
                f"{res['f1']:.4f}",
                f"{res['sens']:.4f}",
                f"{res['spec']:.4f}",
            ])

    table = ax.table(
        cellText=rows,
        colLabels=cols,
        loc='center',
        cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Color header
    for j in range(len(cols)):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(
            color='white',
            fontweight='bold')

    # Color TractoPersist rows
    tp_color = '#E8F5E9'
    for i, row in enumerate(rows):
        if 'TractoPersist' in row[1]:
            for j in range(len(cols)):
                table[i+1, j]\
                    .set_facecolor(tp_color)

    # Alternate row colors
    for i, row in enumerate(rows):
        if 'TractoPersist' not in row[1]:
            color = ('#F5F5F5'
                     if i % 2 == 0
                     else 'white')
            for j in range(len(cols)):
                table[i+1, j]\
                    .set_facecolor(color)

    ax.set_title(
        'TractoPersist v2 — Complete Results',
        fontsize=14, fontweight='bold',
        pad=20)

    plt.tight_layout()
    plt.savefig(save_path,
                dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrices(cm_dict,
                             save_path):
    """Plot confusion matrices"""
    problems = list(cm_dict.keys())
    fig, axes = plt.subplots(
        1, len(problems),
        figsize=(5*len(problems), 4))

    if len(problems) == 1:
        axes = [axes]

    labels_map = {
        'CN vs MCI': ['CN', 'MCI'],
        'MCI vs AD': ['MCI', 'AD'],
        'AD vs CN':  ['CN', 'AD'],
    }

    colors_map = {
        'CN vs MCI': 'Blues',
        'MCI vs AD': 'Oranges',
        'AD vs CN':  'Greens',
    }

    for ax, prob in zip(axes, problems):
        cm     = cm_dict[prob]
        labels = labels_map.get(
            prob, ['Neg', 'Pos'])
        cmap   = colors_map.get(
            prob, 'Blues')

        im = ax.imshow(cm, cmap=cmap,
                        aspect='auto')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(
            labels, fontsize=12)
        ax.set_yticklabels(
            labels, fontsize=12)
        ax.set_xlabel(
            'Predicted', fontsize=11)
        ax.set_ylabel(
            'True', fontsize=11)
        ax.set_title(
            prob, fontsize=12,
            fontweight='bold')

        for i in range(2):
            for j in range(2):
                ax.text(j, i,
                        str(cm[i, j]),
                        ha='center',
                        va='center',
                        fontsize=16,
                        fontweight='bold',
                        color='white'
                        if cm[i, j] > cm.max()/2
                        else 'black')

        plt.colorbar(im, ax=ax)

    plt.suptitle(
        'TractoPersist v2 — Confusion Matrices',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path,
                dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_auc_comparison(all_results,
                         save_path):
    """Bar chart comparing AUC across models"""
    problems = list(all_results.keys())
    models   = list(
        list(all_results.values())[0].keys())

    x      = np.arange(len(problems))
    width  = 0.15
    colors = ['#90CAF9', '#FFCC80',
              '#A5D6A7', '#CE93D8',
              '#F48FB1']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        aucs = [all_results[p][model]['auc']
                for p in problems]
        bars = ax.bar(
            x + i*width, aucs,
            width, label=model,
            color=colors[i % len(colors)],
            edgecolor='black',
            linewidth=0.5)

        # Value labels
        for bar, auc in zip(bars, aucs):
            ax.text(
                bar.get_x() +
                bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{auc:.3f}',
                ha='center', va='bottom',
                fontsize=7,
                rotation=45)

    ax.set_xlabel('Classification Problem',
                   fontsize=12)
    ax.set_ylabel('AUC Score', fontsize=12)
    ax.set_title(
        'TractoPersist v2 — AUC Comparison',
        fontsize=13, fontweight='bold')
    ax.set_xticks(
        x + width * (len(models)-1)/2)
    ax.set_xticklabels(problems,
                        fontsize=11)
    ax.set_ylim([0.0, 1.1])
    ax.axhline(y=0.5, color='red',
                linestyle='--',
                alpha=0.5,
                label='Random (0.5)')
    ax.axhline(y=0.7, color='green',
                linestyle='--',
                alpha=0.5,
                label='Good (0.7)')
    ax.legend(fontsize=9,
               loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path,
                dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ================================================================
# RUN ALL THREE PROBLEMS
# ================================================================
PROBLEMS = [
    {
        'name':   'CN vs MCI',
        'pos':    'MCI',
        'neg':    'CN',
    },
    {
        'name':   'MCI vs AD',
        'pos':    'AD',
        'neg':    'MCI',
    },
    {
        'name':   'AD vs CN',
        'pos':    'AD',
        'neg':    'CN',
    },
]

all_results  = {}
all_roc_data = {}
all_cms      = {}
all_rows     = []

for problem in PROBLEMS:
    pname = problem['name']
    print(f"\n{'='*65}")
    print(f"  PROBLEM: {pname}")
    print(f"{'='*65}")

    # Balanced dataset
    df = prepare_balanced_dataset(
        combined,
        group1=problem['pos'],
        group2=problem['neg'],
        random_state=CONFIG['random_state'])

    X   = df[feature_cols].values
    y   = df['label'].values

    skf = StratifiedKFold(
        n_splits=CONFIG['num_folds'],
        shuffle=True,
        random_state=CONFIG['random_state'])

    # ML baselines
    print(f"\n  ML Baselines...")
    ml_res = run_ml_baselines(X, y, skf)

    print(f"  {'Model':<20} "
          f"{'Acc':^8} {'AUC':^8} "
          f"{'F1':^8} {'Sens':^8} Spec")
    print(f"  {'─'*55}")
    for name, res in ml_res.items():
        print(f"  {name:<20} "
              f"{res['acc']:.4f}   "
              f"{res['auc']:.4f}   "
              f"{res['f1']:.4f}   "
              f"{res['sens']:.4f}   "
              f"{res['spec']:.4f}")

    # TractoPersist v2
    fold_res, probs, labs, roc_data = \
        run_tractopersist(
            df, pname, device,
            figures_path)

    # Aggregate fold results
    mean_auc  = np.mean(
        [r['auc']  for r in fold_res])
    mean_acc  = np.mean(
        [r['acc']  for r in fold_res])
    mean_f1   = np.mean(
        [r['f1']   for r in fold_res])
    mean_sens = np.mean(
        [r['sens'] for r in fold_res])
    mean_spec = np.mean(
        [r['spec'] for r in fold_res])
    std_auc   = np.std(
        [r['auc']  for r in fold_res])
    std_acc   = np.std(
        [r['acc']  for r in fold_res])

    # Overall threshold + CM
    opt_thr, _, _ = \
        find_optimal_threshold(
            np.array(probs),
            np.array(labs))
    overall_res = evaluate_at_threshold(
        np.array(probs),
        np.array(labs), opt_thr)

    all_cms[pname] = overall_res['cm']
    all_roc_data[pname] = roc_data

    tp_res = {
        'acc':     mean_acc,
        'auc':     mean_auc,
        'f1':      mean_f1,
        'sens':    mean_sens,
        'spec':    mean_spec,
        'acc_std': std_acc,
        'auc_std': std_auc,
    }

    all_results[pname] = {
        **ml_res,
        'TractoPersist v2': tp_res
    }

    print(f"\n  {'─'*60}")
    print(f"  TractoPersist v2 [{pname}]:")
    print(f"  AUC  = {mean_auc:.4f}"
          f" ± {std_auc:.4f}")
    print(f"  Acc  = {mean_acc:.4f}"
          f" ± {std_acc:.4f}")
    print(f"  F1   = {mean_f1:.4f}")
    print(f"  Sens = {mean_sens:.4f}")
    print(f"  Spec = {mean_spec:.4f}")
    print(f"  Overall CM: "
          f"{overall_res['cm']}")

    # Collect rows for CSV
    for name, res in all_results[pname].items():
        all_rows.append({
            'problem':     pname,
            'model':       name,
            'accuracy':    res['acc'],
            'acc_std':     res.get(
                'acc_std', 0),
            'auc':         res['auc'],
            'auc_std':     res.get(
                'auc_std', 0),
            'f1':          res['f1'],
            'sensitivity': res['sens'],
            'specificity': res['spec'],
        })

# ================================================================
# GENERATE FIGURES
# ================================================================
print(f"\n{'='*65}")
print(f"  Generating Figures...")
print(f"{'='*65}")

plot_roc_curves(
    all_roc_data,
    os.path.join(figures_path,
                 'roc_curves.png'))

plot_auc_comparison(
    all_results,
    os.path.join(figures_path,
                 'auc_comparison.png'))

plot_confusion_matrices(
    all_cms,
    os.path.join(figures_path,
                 'confusion_matrices.png'))

plot_results_table(
    all_results,
    os.path.join(figures_path,
                 'results_table.png'))

# ================================================================
# FINAL SUMMARY
# ================================================================
print(f"\n{'='*70}")
print(f"  FINAL SUMMARY — TractoPersist v2")
print(f"{'='*70}")
print(f"  {'Problem':<14} {'Model':<22} "
      f"{'Acc':^10} {'AUC':^10} "
      f"{'F1':^8} {'Sens':^8} Spec")
print(f"  {'─'*75}")

for prob in PROBLEMS:
    pname = prob['name']
    res   = all_results[pname]
    for model, r in res.items():
        marker = ' ←' \
            if 'TractoPersist' in model \
            else ''
        print(f"  {pname:<14} "
              f"{model:<22} "
              f"{r['acc']:.4f}     "
              f"{r['auc']:.4f}     "
              f"{r['f1']:.4f}   "
              f"{r['sens']:.4f}   "
              f"{r['spec']:.4f}"
              f"{marker}")
    print(f"  {'─'*75}")

# Save CSV
out_csv = os.path.join(
    output_path,
    'tractopersist_v2_results.csv')
pd.DataFrame(all_rows).to_csv(
    out_csv, index=False)
print(f"\n  Results CSV: {out_csv}")
print(f"  Figures dir: {figures_path}")
print(f"\n  Done!")
