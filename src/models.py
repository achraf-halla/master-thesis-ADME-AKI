import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GATConv, JumpingKnowledge, Set2Set
from torch_geometric.data import Data

class ResidualGINEGATBlock(nn.Module):
    def __init__(self, dim, drop):
        super().__init__()
        self.gine = GINEConv(
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)),
            edge_dim=dim, train_eps=True
        )
        self.norm1 = nn.LayerNorm(dim)

        self.gat = GATConv(dim, dim // 4, heads=4, concat=True)
        self.norm2 = nn.LayerNorm(dim)

        self.drop = drop

    def forward(self, x, edge_index, edge_attr):
        h = self.gine(x, edge_index, edge_attr)
        h = self.norm1(F.relu(h))
        h = F.dropout(h, p=self.drop, training=self.training) + x

        g = self.gat(h, edge_index)
        g = self.norm2(F.relu(g))
        return F.dropout(g, p=self.drop, training=self.training) + h

class GNNEncoder(nn.Module):
    def __init__(self, atom_dim, bond_dim, hidden_dim=128, num_layers=3, drop=0.1):
        super().__init__()
        self.atom_emb = nn.Linear(atom_dim, hidden_dim)
        self.bond_emb = nn.Linear(bond_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResidualGINEGATBlock(hidden_dim, drop) for _ in range(num_layers)
        ])
        self.jk = JumpingKnowledge(mode='cat')
        self.pool = Set2Set(hidden_dim * num_layers, processing_steps=3)

        self.out_dim = hidden_dim * num_layers * 2

    def forward(self, data: Data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.atom_emb(x))
        ea = F.relu(self.bond_emb(ea))

        xs = []
        for blk in self.blocks:
            x = blk(x, ei, ea)
            xs.append(x)

        x_jk = self.jk(xs)
        return self.pool(x_jk, batch)

class ADMEModel(nn.Module):
    def __init__(
        self,
        descriptor_dim,
        atom_dim=6,
        bond_dim=4,
        graph_hidden=128,
        desc_hidden=128,
        num_gnn_layers=4,
        fused_dim=256,
        drop=0.1,
        use_film=True
    ):
        super().__init__()
        self.use_film = use_film

        self.graph_encoder = GNNEncoder(atom_dim, bond_dim, graph_hidden, num_gnn_layers, drop)
        self.desc_encoder = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, desc_hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(desc_hidden, desc_hidden), nn.ReLU(), nn.Dropout(drop)
        )

        if use_film:
            self.film = nn.Linear(desc_hidden, self.graph_encoder.out_dim * 2)

        self.gate = nn.Sequential(
            nn.Linear(self.graph_encoder.out_dim + desc_hidden, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid()
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.graph_encoder.out_dim + desc_hidden, fused_dim),
            nn.ReLU(), nn.Dropout(drop),
            nn.Linear(fused_dim, fused_dim), nn.ReLU(), nn.Dropout(drop)
        )

        self.adapters = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self.fused_dim = fused_dim

        self.task_losses = {}

    def add_head(self, task_id, output_size=1):
        self.adapters[task_id] = nn.Sequential(
            nn.Linear(self.fused_dim, self.fused_dim // 2),
            nn.ReLU(),
            nn.Linear(self.fused_dim // 2, self.fused_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.heads[task_id] = nn.Linear(self.fused_dim, output_size)

    def forward(self, graph_batch, desc, task_id):
        g = self.graph_encoder(graph_batch)
        d = self.desc_encoder(desc)

        if self.use_film:
            gamma, beta = self.film(d).chunk(2, dim=1)
            g = gamma * g + beta

        fused_in = torch.cat([g, d], dim=1)
        gate_val = self.gate(fused_in)
        z = self.fusion_proj(fused_in) * gate_val

        h = self.adapters[task_id](z)
        return self.heads[task_id](h)

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if not name.startswith(('adapters', 'heads')):
                param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = True

