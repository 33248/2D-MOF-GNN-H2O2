import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class CGConv(MessagePassing):

    def __init__(self, node_dim: int, edge_dim: int):
        super(CGConv, self).__init__(aggr='add', node_dim=0)
        self.hidden_dim = node_dim
        self.edge_dim = edge_dim


        self.edge_network = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + edge_dim, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, self.hidden_dim)
        )


        self.node_network = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 128),
            nn.Softplus(),
            nn.Linear(128, self.hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr):


        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):


        z = torch.cat([x_i, x_j, edge_attr], dim=-1)

        return self.edge_network(z)

    def update(self, aggr_out, x):


        z = torch.cat([x, aggr_out], dim=-1)

        return self.node_network(z) + x


class TripletAngleBlock(nn.Module):


    def __init__(self, node_dim: int, angle_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim + angle_dim, 128),
            nn.Softplus(),
            nn.Linear(128, node_dim)
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x: torch.Tensor, triplet_index: torch.Tensor, angle_attr: torch.Tensor) -> torch.Tensor:

        if triplet_index is None or angle_attr is None:
            return x
        if triplet_index.numel() == 0:
            return x
        i = triplet_index[0].to(x.device)
        j = triplet_index[1].to(x.device)
        k = triplet_index[2].to(x.device)
        angle_attr = angle_attr.to(x.device)

        msg_in = torch.cat([x[j], x[k], angle_attr], dim=-1)
        msg = self.mlp(msg_in)

        out = torch.zeros_like(x)
        out.index_add_(0, i, msg)
        return self.norm(x + out)


class DirectAdsorptionEnergyPredictor(nn.Module):


    def __init__(
        self,
        node_dim: int = 64, edge_dim: int = 32, n_conv: int = 3,
        atom_fea_len: int = 92, edge_fea_len: int = 1,
        use_triplet_angle: bool = False, angle_dim: int = 32,
    ):
        super(DirectAdsorptionEnergyPredictor, self).__init__()

        self.node_embedding = nn.Linear(atom_fea_len, node_dim)
        self.edge_embedding = nn.Linear(edge_fea_len, edge_dim)

        self.conv_layers = nn.ModuleList([CGConv(node_dim, edge_dim) for _ in range(n_conv)])

        self.use_triplet_angle = bool(use_triplet_angle)
        self.angle_dim = int(angle_dim)
        if self.use_triplet_angle:
            self.triplet_layers = nn.ModuleList([TripletAngleBlock(node_dim, self.angle_dim) for _ in range(n_conv)])
        else:
            self.triplet_layers = None

        self.site_attention = nn.MultiheadAttention(embed_dim=node_dim, num_heads=4, batch_first=True)
        self.pool = nn.Sequential(nn.Linear(node_dim, 128), nn.Softplus(), nn.Linear(128, 64))


        self.predictor = nn.Sequential(
            nn.Linear(64, 128), nn.Softplus(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Softplus(), nn.Linear(64, 1)
        )

    def forward(self, data):
        x = self.node_embedding(data.x)
        edge_attr = self.edge_embedding(data.edge_attr)

        for li, conv in enumerate(self.conv_layers):
            x = conv(x, data.edge_index, edge_attr)
            if self.use_triplet_angle and hasattr(data, 'triplet_index') and hasattr(data, 'angle_attr'):
                x = self.triplet_layers[li](x, data.triplet_index, data.angle_attr)

        batch_size = data.batch.max().item() + 1
        attended_features = []
        for i in range(batch_size):
            mask = (data.batch == i)
            graph_x = x[mask].unsqueeze(0)
            attn_out, _ = self.site_attention(graph_x, graph_x, graph_x)
            attended_features.append(torch.mean(attn_out.squeeze(0), dim=0))

        structural_features = self.pool(torch.stack(attended_features))


        return self.predictor(structural_features)
