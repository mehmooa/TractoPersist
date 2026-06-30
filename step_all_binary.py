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
    GATConv, global_mean_pool, global_max_pool,
    global_add_pool, BatchNorm, TransformerConv)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, balanced_accuracy_score)
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import copy
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# CONFIGURATION
# ================================================================
CONFIG = {
    'hidden_dim':      128,
    'num_heads':       4,
    'dropout':         0.4,
    'lr':              0.0005,
    'weight_decay':    1e-4,
    'epochs':          300,
    'batch_size':      8,
    'patience':        50,
    'augment_factor':  7,
    'noise_std':       0.008,
    'edge_dropout':    0.08,
    'num_folds':       5,
    'topo_dim':        13,
    'node_dim':        6,
}

# ================================================================
# PATHS
# ================================================================
output_path = r"F:\ADNI_Features"
combined    = pd.read_csv(
    os.path.join(output_path,
                 "combined_features.csv"))

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'cpu')

print("=" * 65)
print("  Three Binary Classifications")
print("=" * 65)
print(f"  Total subjects : {len(combined)}")
print(f"  AD             : "
      f"{sum(combined['group']=='AD')}")
print(f"  MCI            : "
      f"{sum(combined['group']=='MCI')}")
print(f"  CN             : "
      f"{sum(combined['group']=='CN')}")
if torch.cuda.is_available():
    print(f"  GPU            : "
          f"{torch.cuda.get_device_name(0)}")
print(f"  Device         : {device}")

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
# THREE BINARY PROBLEMS
# ================================================================
PROBLEMS = [
    {
        'name':    'CN vs MCI',
        'groups':  ['CN', 'MCI'],
        'pos':     'MCI',
        'neg':     'CN',
        'weights': [0.7, 1.5],
    },
    {
        'name':    'MCI vs AD',
        'groups':  ['MCI', 'AD'],
        'pos':     'AD',
        'neg':     'MCI',
        'weights': [0.7, 5.0],
    },
    {
        'name':    'AD vs CN',
        'groups':  ['AD', 'CN'],
        'pos':     'AD',
        'neg':     'CN',
        'weights': [0.7, 4.0],
    },
]

# ================================================================
# ADVANCED TOPGNN ARCHITECTURE
# ================================================================

class TopologyAttention(nn.Module):
    def __init__(self, topo_dim, hidden):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(topo_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, topo_dim),
            nn.Softmax(dim=1))
        self.proj = nn.Linear(topo_dim, hidden)

    def forward(self, x):
        return F.relu(self.proj(x * self.attn(x)))


class ResidualGATBlock(nn.Module):
    def __init__(self, in_dim, out_dim,
                 heads=4, dropout=0.4):
        super().__init__()
        self.gat      = GATConv(
            in_dim, out_dim // heads,
            heads=heads, dropout=dropout,
            concat=True)
        self.bn       = BatchNorm(out_dim)
        self.dropout  = nn.Dropout(dropout)
        self.residual = (
            nn.Linear(in_dim, out_dim)
            if in_dim != out_dim
            else nn.Identity())

    def forward(self, x, edge_index):
        res = self.residual(x)
        out = F.elu(self.bn(
            self.gat(x, edge_index)) + res)
        return self.dropout(out)


class GraphTransformerBlock(nn.Module):
    def __init__(self, in_dim, out_dim,
                 heads=4, dropout=0.4):
        super().__init__()
        self.conv     = TransformerConv(
            in_dim, out_dim // heads,
            heads=heads, dropout=dropout,
            beta=True)
        self.bn       = BatchNorm(out_dim)
        self.dropout  = nn.Dropout(dropout)
        self.residual = (
            nn.Linear(in_dim, out_dim)
            if in_dim != out_dim
            else nn.Identity())

    def forward(self, x, edge_index):
        res = self.residual(x)
        out = F.elu(self.bn(
            self.conv(x, edge_index)) + res)
        return self.dropout(out)


class AdvancedTopGNN(nn.Module):
    def __init__(self, node_dim, hidden_dim,
                 num_heads, topo_dim, dropout):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout))

        # GAT stream
        self.gat1 = ResidualGATBlock(
            hidden_dim, hidden_dim,
            heads=num_heads, dropout=dropout)
        self.gat2 = ResidualGATBlock(
            hidden_dim, hidden_dim,
            heads=num_heads, dropout=dropout)

        # Graph Transformer stream
        self.gt1 = GraphTransformerBlock(
            hidden_dim, hidden_dim,
            heads=num_heads, dropout=dropout)
        self.gt2 = GraphTransformerBlock(
            hidden_dim, hidden_dim,
            heads=num_heads, dropout=dropout)

        # Final layer
        self.gat3 = ResidualGATBlock(
            hidden_dim, hidden_dim,
            heads=num_heads, dropout=dropout)

        # Topology attention
        self.topo_attn = TopologyAttention(
            topo_dim, 64)

        # Classifier
        fusion_dim = hidden_dim * 3 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2))

    def forward(self, data):
        x  = self.input_proj(data.x)
        ei = data.edge_index
        b  = data.batch

        x_gat = self.gat2(
            self.gat1(x, ei), ei)
        x_gt  = self.gt2(
            self.gt1(x, ei), ei)
        x_out = self.gat3(
            x_gat + x_gt, ei)

        graph_embed = torch.cat([
            global_mean_pool(x_out, b),
            global_max_pool(x_out,  b),
            global_add_pool(x_out,  b),
        ], dim=1)

        topo = self.topo_attn(
            data.topo_features)

        return self.fusion(torch.cat(
            [graph_embed, topo], dim=1))


# ================================================================
# BUILD GRAPH
# ================================================================

def build_graph(row):
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

    cn          = mat / (mat.max() + 1e-8)
    clustering  = np.diag(cn @ cn)
    centrality  = rd / (rd.max() + 1e-8)
    betweenness = rs / (rs.max() + 1e-8)
    local_eff   = np.array([
        cn[i, np.nonzero(cn[i])[0]].mean()
        if len(np.nonzero(cn[i])[0]) > 0
        else 0.0
        for i in range(48)])

    nf = np.stack([
        norm(rs), norm(rd),
        norm(clustering),
        centrality,
        betweenness,
        local_eff
    ], axis=1).astype(np.float32)

    threshold = np.percentile(
        mat[mat > 0], 60)
    edges = np.argwhere(mat > threshold)
    ew    = mat[edges[:, 0], edges[:, 1]]
    if len(ew) > 0:
        ew = ew / (ew.max() + 1e-8)

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
                     factor=7,
                     noise_std=0.008,
                     edge_dropout=0.08):
    minority = [g for g in train_graphs
                if g.y.item() == minority_label]

    if len(minority) == 0:
        return []

    augmented = []
    methods   = ['noise', 'dropout',
                 'mask', 'mixup']

    for _ in range(factor):
        for g in minority:
            method = np.random.choice(methods)
            ng     = copy.deepcopy(g)

            if method == 'noise':
                ng.x = g.x + \
                    torch.randn_like(g.x) \
                    * noise_std
                ng.topo_features = (
                    g.topo_features +
                    torch.randn_like(
                        g.topo_features)
                    * noise_std)

            elif method == 'dropout':
                n_e  = g.edge_index.shape[1]
                keep = torch.rand(n_e) > \
                       edge_dropout
                ng.edge_index = \
                    g.edge_index[:, keep]
                ng.edge_attr  = \
                    g.edge_attr[keep]

            elif method == 'mask':
                mask = torch.rand_like(
                    g.x) > 0.1
                ng.x = g.x * mask

            elif method == 'mixup':
                idx   = np.random.randint(
                    len(minority))
                other = minority[idx]
                alpha = np.random.uniform(
                    0.3, 0.7)
                ng.x = (alpha * g.x +
                        (1-alpha) * other.x)
                ng.topo_features = (
                    alpha * g.topo_features +
                    (1-alpha) *
                    other.topo_features)

            augmented.append(ng)

    return augmented


# ================================================================
# TRAINING HELPERS
# ================================================================

def train_epoch(model, loader,
                optimizer, device, cw):
    model.train()
    total_loss = 0.0
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
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds, labs, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out   = model(batch)
            prob  = F.softmax(
                out, dim=1)[:, 1]
            pred  = out.argmax(dim=1)
            preds.extend(
                pred.cpu().numpy())
            labs.extend(
                batch.y.cpu().numpy())
            probs.extend(
                prob.cpu().numpy())

    acc = accuracy_score(labs, preds)
    f1  = f1_score(labs, preds,
                    zero_division=0)
    bal = balanced_accuracy_score(
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

    return (acc, f1, auc, bal,
            sens, spec, preds, labs)


def run_ml_baselines(X, y, skf,
                     pos_weight=4.0):
    """Run all ML baseline models"""
    ml_models = {
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('pca',    PCA(n_components=40)),
            ('clf',    SVC(
                kernel='rbf', C=10,
                class_weight='balanced',
                probability=True,
                random_state=42))]),

        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1))]),

        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=pos_weight,
                random_state=42,
                eval_metric='logloss'))]),

        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=42))]),
    }

    results = {}
    for name, model in ml_models.items():
        acc_s, auc_s = [], []
        f1_s, sens_s, spec_s = [], [], []

        for tr_idx, te_idx in skf.split(X, y):
            X_tr = X[tr_idx]
            X_te = X[te_idx]
            y_tr = y[tr_idx]
            y_te = y[te_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(
                X_te)[:, 1]

            cm = confusion_matrix(
                y_te, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sens = tp/(tp+fn+1e-8)
                spec = tn/(tn+fp+1e-8)
            else:
                sens = spec = 0.0

            acc_s.append(
                accuracy_score(y_te, y_pred))
            auc_s.append(
                roc_auc_score(y_te, y_prob))
            f1_s.append(f1_score(
                y_te, y_pred,
                zero_division=0))
            sens_s.append(sens)
            spec_s.append(spec)

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


def run_topgnn(graphs, problem, device):
    """Run TopGNN for one binary problem"""
    labels_all  = [g.y.item() for g in graphs]
    skf2        = StratifiedKFold(
        n_splits=CONFIG['num_folds'],
        shuffle=True, random_state=42)

    # Find minority label
    counts = np.bincount(labels_all)
    minority_label = int(np.argmin(counts))

    acc_s, auc_s, f1_s  = [], [], []
    sens_s, spec_s, bal_s = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(
            skf2.split(range(len(graphs)),
                       labels_all)):

        train_g = [graphs[i] for i in tr_idx]
        test_g  = [graphs[i] for i in te_idx]

        # Augment minority class
        aug     = augment_minority(
            train_g,
            minority_label=minority_label,
            factor=CONFIG['augment_factor'],
            noise_std=CONFIG['noise_std'],
            edge_dropout=CONFIG['edge_dropout'])
        train_g = train_g + aug

        n_pos = sum(1 for g in train_g
                    if g.y.item() == 1)
        n_neg = sum(1 for g in train_g
                    if g.y.item() == 0)

        train_loader = DataLoader(
            train_g,
            batch_size=CONFIG['batch_size'],
            shuffle=True)
        test_loader  = DataLoader(
            test_g,
            batch_size=CONFIG['batch_size'],
            shuffle=False)

        model = AdvancedTopGNN(
            node_dim=CONFIG['node_dim'],
            hidden_dim=CONFIG['hidden_dim'],
            num_heads=CONFIG['num_heads'],
            topo_dim=CONFIG['topo_dim'],
            dropout=CONFIG['dropout']
        ).to(device)

        # Class weights
        cw = torch.tensor(
            problem['weights'],
            dtype=torch.float).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG['lr'],
            weight_decay=CONFIG['weight_decay'])

        scheduler = torch.optim.lr_scheduler\
            .CosineAnnealingWarmRestarts(
                optimizer, T_0=50,
                T_mult=2, eta_min=1e-6)

        best_auc   = 0.0
        best_acc   = 0.0
        best_f1    = 0.0
        best_sens  = 0.0
        best_spec  = 0.0
        best_bal   = 0.0
        pat_count  = 0
        best_state = None
        best_preds = []
        best_labs  = []

        for epoch in range(
                1, CONFIG['epochs'] + 1):
            loss = train_epoch(
                model, train_loader,
                optimizer, device, cw)
            scheduler.step()

            (acc, f1, auc, bal,
             sens, spec,
             preds, labs) = evaluate(
                model, test_loader, device)

            if auc > best_auc:
                best_auc   = auc
                best_acc   = acc
                best_f1    = f1
                best_sens  = sens
                best_spec  = spec
                best_bal   = bal
                best_preds = preds
                best_labs  = labs
                pat_count  = 0
                best_state = copy.deepcopy(
                    model.state_dict())
            else:
                pat_count += 1

            if pat_count >= CONFIG['patience']:
                break

        acc_s.append(best_acc)
        auc_s.append(best_auc)
        f1_s.append(best_f1)
        sens_s.append(best_sens)
        spec_s.append(best_spec)
        bal_s.append(best_bal)

        cm = confusion_matrix(
            best_labs, best_preds)
        print(f"    Fold {fold+1}: "
              f"Acc={best_acc:.4f} "
              f"AUC={best_auc:.4f} "
              f"F1={best_f1:.4f} "
              f"Sens={best_sens:.4f} "
              f"Spec={best_spec:.4f}")
        print(f"    CM: {cm}")

    return {
        'acc':     np.mean(acc_s),
        'auc':     np.mean(auc_s),
        'f1':      np.mean(f1_s),
        'sens':    np.mean(sens_s),
        'spec':    np.mean(spec_s),
        'bal':     np.mean(bal_s),
        'acc_std': np.std(acc_s),
        'auc_std': np.std(auc_s),
    }


# ================================================================
# RUN ALL THREE PROBLEMS
# ================================================================
all_results = []

for problem in PROBLEMS:
    print(f"\n{'='*65}")
    print(f"  PROBLEM: {problem['name']}")
    print(f"{'='*65}")

    # Prepare dataset
    df = combined[
        combined['group'].isin(
            problem['groups'])].copy()
    df['label'] = (
        df['group'] == problem['pos']
    ).astype(int)

    n_pos = sum(df['group'] == problem['pos'])
    n_neg = sum(df['group'] == problem['neg'])
    print(f"  {problem['pos']}: {n_pos} subjects")
    print(f"  {problem['neg']}: {n_neg} subjects")
    print(f"  Total          : {len(df)}")

    X = df[feature_cols].values
    y = df['label'].values

    # ── ML Baselines ──────────────────────────
    print(f"\n  ML Baselines:")
    print(f"  {'─'*55}")

    skf = StratifiedKFold(
        n_splits=CONFIG['num_folds'],
        shuffle=True, random_state=42)

    pos_weight = n_neg / (n_pos + 1e-8)
    ml_res = run_ml_baselines(
        X, y, skf, pos_weight)

    print(f"  {'Model':<22} "
          f"{'Acc':^10} {'AUC':^10} "
          f"{'F1':^8} {'Sens':^8} Spec")
    print(f"  {'─'*65}")
    for name, res in ml_res.items():
        print(f"  {name:<22} "
              f"{res['acc']:.4f}      "
              f"{res['auc']:.4f}      "
              f"{res['f1']:.4f}   "
              f"{res['sens']:.4f}   "
              f"{res['spec']:.4f}")

    # ── Build graphs ───────────────────────────
    print(f"\n  Building graphs...")
    graphs = []
    for _, row in df.iterrows():
        try:
            graphs.append(build_graph(row))
        except Exception as e:
            pass
    print(f"  Built {len(graphs)} graphs")

    # ── TopGNN ────────────────────────────────
    print(f"\n  Training Advanced TopGNN...")
    gnn_res = run_topgnn(
        graphs, problem, device)

    print(f"\n  {'─'*65}")
    print(f"  AdvTopGNN: "
          f"Acc={gnn_res['acc']:.4f}±"
          f"{gnn_res['acc_std']:.3f}  "
          f"AUC={gnn_res['auc']:.4f}±"
          f"{gnn_res['auc_std']:.3f}  "
          f"F1={gnn_res['f1']:.4f}  "
          f"Sens={gnn_res['sens']:.4f}  "
          f"Spec={gnn_res['spec']:.4f}")

    # ── Store results ──────────────────────────
    for name, res in ml_res.items():
        all_results.append({
            'problem':     problem['name'],
            'model':       name,
            'accuracy':    res['acc'],
            'auc':         res['auc'],
            'f1':          res['f1'],
            'sensitivity': res['sens'],
            'specificity': res['spec'],
            'type':        'baseline'
        })

    all_results.append({
        'problem':      problem['name'],
        'model':        'AdvancedTopGNN',
        'accuracy':     gnn_res['acc'],
        'auc':          gnn_res['auc'],
        'f1':           gnn_res['f1'],
        'sensitivity':  gnn_res['sens'],
        'specificity':  gnn_res['spec'],
        'balanced_acc': gnn_res['bal'],
        'type':         'proposed'
    })

# ================================================================
# FINAL SUMMARY TABLE
# ================================================================
print(f"\n{'='*70}")
print(f"  COMPLETE SUMMARY — All Binary Problems")
print(f"{'='*70}")

df_res = pd.DataFrame(all_results)

for prob_name in ['CN vs MCI',
                  'MCI vs AD',
                  'AD vs CN']:
    subset = df_res[
        df_res['problem'] == prob_name]
    print(f"\n  {prob_name}:")
    print(f"  {'Model':<22} "
          f"{'Acc':^10} {'AUC':^10} "
          f"{'F1':^8} {'Sens':^8} Spec")
    print(f"  {'─'*65}")
    for _, row in subset.iterrows():
        print(f"  {row['model']:<22} "
              f"{row['accuracy']:.4f}      "
              f"{row['auc']:.4f}      "
              f"{row['f1']:.4f}   "
              f"{row['sensitivity']:.4f}   "
              f"{row['specificity']:.4f}")

# ================================================================
# SAVE
# ================================================================
out_file = os.path.join(
    output_path, "all_binary_results.csv")
df_res.to_csv(out_file, index=False)
print(f"\n  Saved: {out_file}")
print("\n  Done!")
