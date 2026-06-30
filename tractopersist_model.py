"""
TractoPersist — Core Model Architecture
Hybrid GAT + Graph Transformer with Topology Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv, TransformerConv,
    global_mean_pool, global_max_pool,
    global_add_pool, BatchNorm)


class MultiHeadTopologyAttention(nn.Module):
    """Multi-head attention for persistent homology features"""
    def __init__(self, topo_dim=13,
                 hidden=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden // num_heads
        self.q  = nn.Linear(topo_dim, hidden)
        self.k  = nn.Linear(topo_dim, hidden)
        self.v  = nn.Linear(topo_dim, hidden)
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
        scores = (Q * K).sum(-1,
            keepdim=True) * (self.head_dim**-0.5)
        attn = torch.softmax(scores, dim=1)
        out  = (attn * V).view(B, -1)
        return self.norm(
            F.relu(self.out(out)))


class ResGATBlock(nn.Module):
    """Residual Graph Attention block"""
    def __init__(self, in_dim, out_dim,
                 heads=8, dropout=0.3):
        super().__init__()
        self.gat  = GATConv(
            in_dim, out_dim // heads,
            heads=heads, dropout=dropout,
            concat=True)
        self.bn   = BatchNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.skip = (nn.Linear(in_dim, out_dim)
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
            heads=heads, dropout=dropout,
            beta=True)
        self.bn   = BatchNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.skip = (nn.Linear(in_dim, out_dim)
                     if in_dim != out_dim
                     else nn.Identity())

    def forward(self, x, ei):
        return self.drop(F.elu(
            self.bn(self.conv(x, ei))
            + self.skip(x)))


class TractoPersist(nn.Module):
    """
    TractoPersist: Topological Graph Neural Network
    for Alzheimer's Disease Classification

    Architecture:
    - Parallel GAT + Graph Transformer streams
    - Multi-head topology attention (H0+H1 features)
    - Hierarchical pooling: mean+max+sum+std
    - Deep MLP classifier

    Args:
        node_dim:   number of node features (default: 8)
        hidden_dim: hidden dimension (default: 256)
        num_heads:  attention heads (default: 8)
        num_layers: GNN layers per stream (default: 4)
        topo_dim:   topology feature dim (default: 13)
        dropout:    dropout rate (default: 0.3)
        num_classes: output classes (default: 2)
    """
    def __init__(self,
                 node_dim=8,
                 hidden_dim=256,
                 num_heads=8,
                 num_layers=4,
                 topo_dim=13,
                 dropout=0.3,
                 num_classes=2):
        super().__init__()

        self.num_layers = num_layers

        # Input projection
        self.inp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout))

        # GAT stream
        self.gat_layers = nn.ModuleList([
            ResGATBlock(hidden_dim, hidden_dim,
                        heads=num_heads,
                        dropout=dropout)
            for _ in range(num_layers)])

        # Graph Transformer stream
        self.gt_layers = nn.ModuleList([
            ResGTBlock(hidden_dim, hidden_dim,
                       heads=num_heads,
                       dropout=dropout)
            for _ in range(num_layers)])

        # Topology attention
        self.topo_attn = MultiHeadTopologyAttention(
            topo_dim=topo_dim,
            hidden=hidden_dim,
            num_heads=4)

        # Fusion MLP
        # mean+max+sum+std = 4*hidden + hidden (topo)
        fusion_dim = hidden_dim * 4 + hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_classes))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight)
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

        # Combine streams
        x_c = x_g + x_t

        # Hierarchical pooling
        counts = torch.bincount(b)
        x_list = torch.split(
            x_c, counts.tolist())
        x_std  = torch.stack([
            xi.std(0) if xi.shape[0] > 1
            else torch.zeros(
                x_c.shape[1],
                device=x_c.device)
            for xi in x_list])

        graph_embed = torch.cat([
            global_mean_pool(x_c, b),
            global_max_pool(x_c,  b),
            global_add_pool(x_c,  b),
            x_std,
        ], dim=1)

        # Topology attention
        topo = self.topo_attn(
            data.topo_features)

        return self.fusion(
            torch.cat([graph_embed, topo],
                      dim=1))


def count_parameters(model):
    return sum(p.numel()
               for p in model.parameters()
               if p.requires_grad)


if __name__ == "__main__":
    model = TractoPersist(
        node_dim=8,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        topo_dim=13,
        dropout=0.3,
        num_classes=2)

    print(f"TractoPersist")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Architecture:")
    print(f"  GAT stream: 4 layers × 8 heads")
    print(f"  GT stream:  4 layers × 8 heads")
    print(f"  Topology:   multi-head attention (4 heads)")
    print(f"  Pooling:    mean+max+sum+std")
    print(f"  Classifier: 512→256→128→64→2")
