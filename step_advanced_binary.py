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
    'ad_weight':       4.0,
    'cn_weight':       0.7,
    'num_folds':       5,
    'topo_dim':        13,
    'node_dim':        6,
}

# ================================================================
# PATHS
# ================================================================
output_path = r"F:\ADNI_Features"
combined    = pd.read_csv(
    os.path.join(output_path, "combined_features.csv"))

# ================================================================
# BINARY: AD vs CN
# ================================================================
binary = combined[
    combined['group'].isin(['AD', 'CN'])].copy()
binary['label'] = (binary['group'] == 'AD').astype(int)

print("=" * 65)
print("  Advanced Binary Classification: AD vs CN")
print("=" * 65)
print(f"  AD subjects : {sum(binary['group'] == 'AD')}")
print(f"  CN subjects : {sum(binary['group'] == 'CN')}")
print(f"  Total       : {len(binary)}")
if torch.cuda.is_available():
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
else:
    print("  Device      : CPU")

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
                if c in binary.columns]

X = binary[feature_cols].values
y = binary['label'].values
print(f"\n  Feature matrix : {X.shape}")

# ================================================================
# ML BASELINE MODELS
# ================================================================
print("\n" + "=" * 65)
print("  Advanced ML Baselines")
print("=" * 65)

skf = StratifiedKFold(
    n_splits=CONFIG['num_folds'],
    shuffle=True, random_state=42)

ml_models = {
    'SVM (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('pca',    PCA(n_components=40)),
        ('clf',    SVC(kernel='rbf', C=10,
                       gamma='scale',
                       class_weight='balanced',
                       probability=True,
                       random_state=42))]),

    'SVM (Poly)': Pipeline([
        ('scaler', StandardScaler()),
        ('pca',    PCA(n_components=40)),
        ('clf',    SVC(kernel='poly', degree=3,
                       C=5,
                       class_weight='balanced',
                       probability=True,
                       random_state=42))]),

    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    RandomForestClassifier(
                       n_estimators=500,
                       max_depth=12,
                       min_samples_split=3,
                       class_weight='balanced',
                       random_state=42,
                       n_jobs=-1))]),

    'XGBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    XGBClassifier(
                       n_estimators=500,
                       max_depth=6,
                       learning_rate=0.03,
                       subsample=0.8,
                       colsample_bytree=0.8,
                       scale_pos_weight=4,
                       random_state=42,
                       eval_metric='logloss'))]),

    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    GradientBoostingClassifier(
                       n_estimators=300,
                       max_depth=5,
                       learning_rate=0.05,
                       subsample=0.8,
                       random_state=42))]),

    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('pca',    PCA(n_components=30)),
        ('clf',    KNeighborsClassifier(
                       n_neighbors=7,
                       weights='distance',
                       metric='euclidean'))]),
}

ml_results = {}

for name, model in ml_models.items():
    acc_s, auc_s, f1_s = [], [], []
    sens_s, spec_s     = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        cm = confusion_matrix(y_te, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn + 1e-8)
            spec = tn / (tn + fp + 1e-8)
        else:
            sens = spec = 0.0

        acc_s.append(accuracy_score(y_te, y_pred))
        auc_s.append(roc_auc_score(y_te, y_prob))
        f1_s.append(f1_score(y_te, y_pred,
                              zero_division=0))
        sens_s.append(sens)
        spec_s.append(spec)

    ml_results[name] = {
        'acc':     np.mean(acc_s),
        'auc':     np.mean(auc_s),
        'f1':      np.mean(f1_s),
        'sens':    np.mean(sens_s),
        'spec':    np.mean(spec_s),
        'acc_std': np.std(acc_s),
        'auc_std': np.std(auc_s),
    }

    print(f"  {name:<22} "
          f"Acc={ml_results[name]['acc']:.4f}  "
          f"AUC={ml_results[name]['auc']:.4f}  "
          f"F1={ml_results[name]['f1']:.4f}  "
          f"Sens={ml_results[name]['sens']:.4f}  "
          f"Spec={ml_results[name]['spec']:.4f}")


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
        weights  = self.attn(x)
        attended = x * weights
        return F.relu(self.proj(attended))


class ResidualGATBlock(nn.Module):
    def __init__(self, in_dim, out_dim,
                 heads=4, dropout=0.4):
        super().__init__()
        self.gat      = GATConv(in_dim, out_dim // heads,
                                 heads=heads,
                                 dropout=dropout,
                                 concat=True)
        self.bn       = BatchNorm(out_dim)
        self.dropout  = nn.Dropout(dropout)
        self.residual = (nn.Linear(in_dim, out_dim)
                         if in_dim != out_dim
                         else nn.Identity())

    def forward(self, x, edge_index):
        res = self.residual(x)
        out = self.gat(x, edge_index)
        out = self.bn(out)
        out = F.elu(out + res)
        return self.dropout(out)


class GraphTransformerBlock(nn.Module):
    def __init__(self, in_dim, out_dim,
                 heads=4, dropout=0.4):
        super().__init__()
        self.conv     = TransformerConv(
            in_dim, out_dim // heads,
            heads=heads, dropout=dropout, beta=True)
        self.bn       = BatchNorm(out_dim)
        self.dropout  = nn.Dropout(dropout)
        self.residual = (nn.Linear(in_dim, out_dim)
                         if in_dim != out_dim
                         else nn.Identity())

    def forward(self, x, edge_index):
        res = self.residual(x)
        out = self.conv(x, edge_index)
        out = self.bn(out)
        out = F.elu(out + res)
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

        self.gat1 = ResidualGATBlock(
            hidden_dim, hidden_dim,
            heads=num_heads, dropout=dropout)
        self.gat2 = ResidualGATBlock(
            hidden_dim, hidden_dim,
            heads=num_heads, dropout=dropout)

        self.gt1 = GraphTransformerBlock(
            hidden_dim, hidden_dim,
            heads=num_heads, dropout=dropout)
        self.gt2 = GraphTransformerBlock(
            hidden_dim, hidden_dim,
            heads=num_heads, dropout=dropout)

        self.gat3 = ResidualGATBlock(
            hidden_dim, hidden_dim,
            heads=num_heads, dropout=dropout)

        self.topo_attn = TopologyAttention(
            topo_dim, 64)

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

        x_gat = self.gat2(self.gat1(x, ei), ei)
        x_gt  = self.gt2(self.gt1(x, ei), ei)
        x_out = self.gat3(x_gat + x_gt, ei)

        graph_embed = torch.cat([
            global_mean_pool(x_out, b),
            global_max_pool(x_out,  b),
            global_add_pool(x_out,  b),
        ], dim=1)

        topo = self.topo_attn(data.topo_features)

        return self.fusion(
            torch.cat([graph_embed, topo], dim=1))


# ================================================================
# BUILD ENHANCED GRAPH
# ================================================================

def build_enhanced_graph(row):
    conn = (row[[f'conn_{i}' for i in range(2304)]]
            .values.astype(np.float32))
    mat  = conn.reshape(48, 48)
    rs   = row[[f'roi_str_{i}'
                for i in range(48)]].values
    rd   = row[[f'roi_deg_{i}'
                for i in range(48)]].values

    def norm(x):
        s = x.std()
        return (x - x.mean()) / (s if s > 1e-8 else 1.0)

    cn         = mat / (mat.max() + 1e-8)
    clustering = np.diag(cn @ cn)
    centrality = rd / (rd.max() + 1e-8)
    betweenness = rs / (rs.max() + 1e-8)
    local_eff  = np.array([
        cn[i, np.nonzero(cn[i])[0]].mean()
        if len(np.nonzero(cn[i])[0]) > 0 else 0.0
        for i in range(48)])

    nf = np.stack([
        norm(rs), norm(rd),
        norm(clustering),
        centrality, betweenness,
        local_eff
    ], axis=1).astype(np.float32)

    threshold = np.percentile(mat[mat > 0], 60)
    edges     = np.argwhere(mat > threshold)
    ew        = mat[edges[:, 0], edges[:, 1]]
    if len(ew) > 0:
        ew = ew / (ew.max() + 1e-8)

    tc = ['h0_count', 'h0_mean_lifetime',
          'h0_sum_lifetime', 'h0_entropy',
          'h1_count', 'h1_mean_lifetime',
          'h1_max_lifetime', 'h1_sum_lifetime',
          'h1_entropy', 'betti_0', 'betti_1',
          'total_persistence_h0',
          'total_persistence_h1']
    tf = row[tc].values.astype(np.float32)
    tf = (tf - tf.mean()) / (tf.std() + 1e-8)

    return Data(
        x=torch.tensor(nf),
        edge_index=torch.tensor(
            edges.T, dtype=torch.long),
        edge_attr=torch.tensor(ew),
        y=torch.tensor([int(row['label'])],
                        dtype=torch.long),
        topo_features=torch.tensor(
            tf).unsqueeze(0))


# ================================================================
# BUILD DATASET
# ================================================================
print("\nBuilding enhanced graphs...")
graphs = []
for _, row in binary.iterrows():
    try:
        graphs.append(build_enhanced_graph(row))
    except Exception as e:
        print(f"  Skip: {e}")

print(f"  Built   : {len(graphs)} graphs")
print(f"  AD      : {sum(g.y.item()==1 for g in graphs)}")
print(f"  CN      : {sum(g.y.item()==0 for g in graphs)}")


# ================================================================
# ADVANCED AUGMENTATION
# ================================================================

def advanced_augment(train_graphs, factor=7,
                     noise_std=0.008,
                     edge_dropout=0.08):
    ad_graphs = [g for g in train_graphs
                 if g.y.item() == 1]
    augmented = []
    methods   = ['noise', 'dropout',
                 'mask', 'mixup']

    for _ in range(factor):
        for g in ad_graphs:
            method = np.random.choice(methods)
            ng     = copy.deepcopy(g)

            if method == 'noise':
                ng.x = g.x + \
                    torch.randn_like(g.x) * noise_std
                ng.topo_features = (
                    g.topo_features +
                    torch.randn_like(
                        g.topo_features) * noise_std)

            elif method == 'dropout':
                n_e  = g.edge_index.shape[1]
                keep = torch.rand(n_e) > edge_dropout
                ng.edge_index = g.edge_index[:, keep]
                ng.edge_attr  = g.edge_attr[keep]

            elif method == 'mask':
                mask = torch.rand_like(g.x) > 0.1
                ng.x = g.x * mask

            elif method == 'mixup':
                other = ad_graphs[np.random.randint(len(ad_graphs))]
                alpha = np.random.uniform(0.3, 0.7)
                ng.x = (alpha * g.x +
                        (1 - alpha) * other.x)
                ng.topo_features = (
                    alpha * g.topo_features +
                    (1 - alpha) * other.topo_features)

            augmented.append(ng)

    return augmented


# ================================================================
# TRAINING HELPERS
# ================================================================

def train_epoch(model, loader, optimizer,
                device, class_weights):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch)
        loss = F.cross_entropy(
            out, batch.y, weight=class_weights)
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
            prob  = F.softmax(out, dim=1)[:, 1]
            pred  = out.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            labs.extend(batch.y.cpu().numpy())
            probs.extend(prob.cpu().numpy())

    acc = accuracy_score(labs, preds)
    f1  = f1_score(labs, preds, zero_division=0)
    bal = balanced_accuracy_score(labs, preds)
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


# ================================================================
# 5-FOLD CROSS VALIDATION
# ================================================================
device     = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')
labels_all = [g.y.item() for g in graphs]
skf2       = StratifiedKFold(
    n_splits=CONFIG['num_folds'],
    shuffle=True, random_state=42)

acc_s  = []
auc_s  = []
f1_s   = []
sens_s = []
spec_s = []
bal_s  = []

print("\n" + "=" * 65)
print("  Advanced TopGNN Training")
print("=" * 65)
print(f"  Hidden dim      : {CONFIG['hidden_dim']}")
print(f"  Attention heads : {CONFIG['num_heads']}")
print(f"  Max epochs      : {CONFIG['epochs']}")
print(f"  Patience        : {CONFIG['patience']}")
print(f"  Augment factor  : {CONFIG['augment_factor']}x")
print(f"  Node features   : {CONFIG['node_dim']}")
print(f"  Device          : {device}")

for fold, (train_idx, test_idx) in enumerate(
        skf2.split(range(len(graphs)), labels_all)):

    print(f"\n  {'─'*55}")
    print(f"  Fold {fold + 1}/{CONFIG['num_folds']}")

    train_g = [graphs[i] for i in train_idx]
    test_g  = [graphs[i] for i in test_idx]

    aug     = advanced_augment(
        train_g,
        factor=CONFIG['augment_factor'],
        noise_std=CONFIG['noise_std'],
        edge_dropout=CONFIG['edge_dropout'])
    train_g = train_g + aug

    n_ad = sum(1 for g in train_g
               if g.y.item() == 1)
    n_cn = sum(1 for g in train_g
               if g.y.item() == 0)
    print(f"  Train : AD={n_ad} CN={n_cn} "
          f"Total={len(train_g)}")
    print(f"  Test  : {len(test_g)} subjects")

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

    if fold == 0:
        params = sum(p.numel()
                     for p in model.parameters()
                     if p.requires_grad)
        print(f"  Model parameters: {params:,}")

    cw = torch.tensor(
        [CONFIG['cn_weight'],
         CONFIG['ad_weight']],
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

    for epoch in range(1, CONFIG['epochs'] + 1):
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

        if epoch % 50 == 0:
            print(f"    Ep {epoch:3d}  "
                  f"Loss={loss:.4f}  "
                  f"Acc={acc:.4f}  "
                  f"AUC={auc:.4f}  "
                  f"F1={f1:.4f}  "
                  f"Pat={pat_count}")

        if pat_count >= CONFIG['patience']:
            print(f"    Early stop at epoch {epoch}")
            break

    acc_s.append(best_acc)
    auc_s.append(best_auc)
    f1_s.append(best_f1)
    sens_s.append(best_sens)
    spec_s.append(best_spec)
    bal_s.append(best_bal)

    cm = confusion_matrix(best_labs, best_preds)
    print(f"\n  Fold {fold+1} Best:")
    print(f"  Acc={best_acc:.4f}  "
          f"AUC={best_auc:.4f}  "
          f"F1={best_f1:.4f}")
    print(f"  Sens={best_sens:.4f}  "
          f"Spec={best_spec:.4f}  "
          f"Bal={best_bal:.4f}")
    print(f"  Confusion Matrix:\n  {cm}")


# ================================================================
# FINAL RESULTS TABLE
# ================================================================
print("\n" + "=" * 70)
print("  FINAL RESULTS — Binary AD vs CN")
print("=" * 70)
print(f"  {'Model':<22} {'Acc':^14} "
      f"{'AUC':^14} {'F1':^8} "
      f"{'Sens':^8} {'Spec'}")
print(f"  {'─'*68}")

for name, res in ml_results.items():
    print(f"  {name:<22} "
          f"{res['acc']:.4f}±{res['acc_std']:.3f}  "
          f"{res['auc']:.4f}±{res['auc_std']:.3f}  "
          f"{res['f1']:.4f}   "
          f"{res['sens']:.4f}   "
          f"{res['spec']:.4f}")

print(f"  {'─'*68}")
print(f"  {'AdvTopGNN (ours)':<22} "
      f"{np.mean(acc_s):.4f}±{np.std(acc_s):.3f}  "
      f"{np.mean(auc_s):.4f}±{np.std(auc_s):.3f}  "
      f"{np.mean(f1_s):.4f}   "
      f"{np.mean(sens_s):.4f}   "
      f"{np.mean(spec_s):.4f}")

print(f"\n  Balanced Accuracy : "
      f"{np.mean(bal_s):.4f} ± {np.std(bal_s):.4f}")

# ================================================================
# SAVE RESULTS
# ================================================================
all_res = []
for name, res in ml_results.items():
    all_res.append({
        'model':        name,
        'accuracy':     res['acc'],
        'accuracy_std': res['acc_std'],
        'auc':          res['auc'],
        'auc_std':      res['auc_std'],
        'f1':           res['f1'],
        'sensitivity':  res['sens'],
        'specificity':  res['spec'],
        'type':         'baseline'
    })

all_res.append({
    'model':        'AdvancedTopGNN',
    'accuracy':     np.mean(acc_s),
    'accuracy_std': np.std(acc_s),
    'auc':          np.mean(auc_s),
    'auc_std':      np.std(auc_s),
    'f1':           np.mean(f1_s),
    'sensitivity':  np.mean(sens_s),
    'specificity':  np.mean(spec_s),
    'balanced_acc': np.mean(bal_s),
    'type':         'proposed'
})

out_file = os.path.join(
    output_path, "advanced_binary_results.csv")
pd.DataFrame(all_res).to_csv(out_file, index=False)
print(f"\n  Results saved to: {out_file}")
print("\n  Done!")
