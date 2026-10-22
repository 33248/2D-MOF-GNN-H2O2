import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Tuple, Optional


if 'ElectronicStructurePredictor' not in globals():
    class ElectronicStructurePredictor(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise RuntimeError('ElectronicStructurePredictor is missing from this file. Please restore its definition.')

if 'AdsorptionEnergyPredictor' not in globals():
    class AdsorptionEnergyPredictor(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise RuntimeError('AdsorptionEnergyPredictor is missing from this file. Please restore its definition.')


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


class GlobalFeatureExtractor(nn.Module):


    def __init__(self):
        super(GlobalFeatureExtractor, self).__init__()

    def forward(self, data):


        batch_size = data.batch.max().item() + 1
        global_features = []

        for i in range(batch_size):
            mask = (data.batch == i)
            graph_data = self._extract_single_graph_features(data, mask)
            global_features.append(graph_data)

        return torch.stack(global_features)

    def _extract_single_graph_features(self, data, mask):

        if mask is None:
            raise ValueError('mask is None')
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)
        mask = mask.bool()
        device = data.x.device if hasattr(data, 'x') and torch.is_tensor(data.x) else None
        def _to_tensor(x):
            if torch.is_tensor(x):
                return x.to(device) if device is not None else x
            if isinstance(x, np.ndarray):
                t = torch.from_numpy(x)
            else:
                t = torch.as_tensor(x)
            return t.to(device) if device is not None else t

        if hasattr(data, 'coordination_numbers'):
            data.coordination_numbers = _to_tensor(data.coordination_numbers)
        if hasattr(data, 'electronegativity'):
            data.electronegativity = _to_tensor(data.electronegativity)
        if hasattr(data, 'cell'):
            data.cell = _to_tensor(data.cell)

        features = []


        if hasattr(data, 'cell'):
            cell = data.cell
            if torch.is_tensor(cell):
                if cell.dim() == 2:
                    cell = cell
                elif cell.dim() == 3:
                    cell = cell[mask][0] if cell.size(0) == mask.numel() else cell[0]
                else:
                    cell = cell.view(3, 3)
            else:
                cell = torch.eye(3, device=device) if device is not None else torch.eye(3)
        else:
            cell = torch.eye(3, device=device) if device is not None else torch.eye(3)


        a_vec, b_vec = cell[0, :2], cell[1, :2]
        a_length = torch.norm(a_vec).clamp(min=1e-8)
        b_length = torch.norm(b_vec).clamp(min=1e-8)


        cos_angle = torch.dot(a_vec, b_vec) / (a_length * b_length)
        ab_angle = torch.acos(cos_angle.clamp(-1.0, 1.0))
        area = a_length * b_length * torch.sin(ab_angle).clamp(min=1e-8)


        aspect_ratio = torch.max(a_length, b_length) / torch.min(a_length, b_length)


        features.extend([area, a_length, b_length, ab_angle, aspect_ratio])


        n_atoms = mask.sum().float()
        atomic_density_2d = n_atoms / area
        features.append(atomic_density_2d)


        coord_numbers = data.coordination_numbers[mask]
        coord_mean = torch.mean(coord_numbers.float())
        coord_std = torch.std(coord_numbers.float())
        coord_max = torch.max(coord_numbers.float())
        features.extend([coord_mean, coord_std, coord_max])


        electronegativity = data.electronegativity[mask]
        en_mean = torch.mean(electronegativity)
        en_std = torch.std(electronegativity)
        en_range = torch.max(electronegativity) - torch.min(electronegativity)
        features.extend([en_mean, en_std, en_range])


        pos_2d = data.pos[mask][:, :2]


        centroid_2d = torch.mean(pos_2d, dim=0)
        centroid_deviation_2d = torch.norm(centroid_2d)
        features.append(centroid_deviation_2d)


        distances_2d = torch.cdist(pos_2d, pos_2d)
        distances_2d = distances_2d[distances_2d > 0]

        if len(distances_2d) > 0:
            rdf_mean_2d = torch.mean(distances_2d)
            rdf_std_2d = torch.std(distances_2d)

            rdf_max_2d = torch.max(distances_2d)
            features.extend([rdf_mean_2d, rdf_std_2d, rdf_max_2d])
        else:
            features.extend([torch.tensor(0.0, device=device),
                             torch.tensor(0.0, device=device),
                             torch.tensor(0.0, device=device)])


        return torch.stack(features)


class ElectronicStructurePredictor(nn.Module):


    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 32,
        global_dim: int = 16,
        n_conv: int = 4,
        atom_fea_len: int = 92,
        edge_fea_len: int = 1,
        use_triplet_angle: bool = False,
        angle_dim: int = 32,
    ):
        super(ElectronicStructurePredictor, self).__init__()

        self.node_embedding = nn.Linear(atom_fea_len, node_dim)
        self.edge_embedding = nn.Linear(edge_fea_len, edge_dim)

        self.conv_layers = nn.ModuleList([
            CGConv(node_dim, edge_dim) for _ in range(n_conv)
        ])

        self.use_triplet_angle = bool(use_triplet_angle)
        self.angle_dim = int(angle_dim)
        if self.use_triplet_angle:
            self.triplet_layers = nn.ModuleList([TripletAngleBlock(node_dim, self.angle_dim) for _ in range(n_conv)])
        else:
            self.triplet_layers = None

        self.global_extractor = GlobalFeatureExtractor()


        self.pool = nn.Sequential(
            nn.Linear(node_dim, 128),
            nn.Softplus(),
            nn.Linear(128, 64)
        )


        self.fusion_macro = nn.Sequential(
            nn.Linear(64 + 64 + global_dim, 128),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Dropout(0.2),
        )


        self.fusion_micro = nn.Sequential(
            nn.Linear(64 + global_dim, 128),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Dropout(0.2),
        )


        self.work_function_predictor = nn.Linear(128, 1)
        self.dband_predictor = nn.Linear(128, 1)

    def forward(self, data):
        x = self.node_embedding(data.x)
        edge_attr = self.edge_embedding(data.edge_attr)

        for li, conv in enumerate(self.conv_layers):
            x = conv(x, data.edge_index, edge_attr)
            if self.use_triplet_angle and hasattr(data, 'triplet_index') and hasattr(data, 'angle_attr'):
                x = self.triplet_layers[li](x, data.triplet_index, data.angle_attr)

        batch_size = data.batch.max().item() + 1

        pooled_macro = []
        pooled_metal = []


        is_metal = data.atomic_numbers >= 21

        for i in range(batch_size):
            mask = (data.batch == i)
            graph_x = x[mask]


            pooled_macro.append(torch.mean(graph_x, dim=0))


            metal_mask = is_metal[mask]
            if metal_mask.sum() > 0:
                pooled_metal.append(torch.mean(graph_x[metal_mask], dim=0))
            else:

                pooled_metal.append(torch.mean(graph_x, dim=0))

        local_macro = self.pool(torch.stack(pooled_macro))
        local_metal = self.pool(torch.stack(pooled_metal))

        global_features = self.global_extractor(data)


        repr_macro = self.fusion_macro(torch.cat([local_macro, local_metal, global_features], dim=-1))


        repr_micro = self.fusion_micro(torch.cat([local_metal, global_features], dim=-1))

        electronic_features = {
            'work_function': self.work_function_predictor(repr_macro),
            'd_band_center': self.dband_predictor(repr_micro)
        }


        electronic_repr = repr_macro + repr_micro

        return electronic_features, electronic_repr


class AdsorptionEnergyPredictor(nn.Module):


    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 32,
        electronic_dim: int = 128,
        n_conv: int = 3,
        atom_fea_len: int = 92,
        edge_fea_len: int = 1,
        use_triplet_angle: bool = False,
        angle_dim: int = 32,
    ):
        super(AdsorptionEnergyPredictor, self).__init__()

        self.node_embedding = nn.Linear(atom_fea_len, node_dim)
        self.edge_embedding = nn.Linear(edge_fea_len, edge_dim)


        self.conv_layers = nn.ModuleList([
            CGConv(node_dim, edge_dim) for _ in range(n_conv)
        ])

        self.use_triplet_angle = bool(use_triplet_angle)
        self.angle_dim = int(angle_dim)
        if self.use_triplet_angle:
            self.triplet_layers = nn.ModuleList([TripletAngleBlock(node_dim, self.angle_dim) for _ in range(n_conv)])
        else:
            self.triplet_layers = None


        self.site_attention = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=4,
            batch_first=True
        )


        self.pool = nn.Sequential(
            nn.Linear(node_dim, 128),
            nn.Softplus(),
            nn.Linear(128, 64)
        )


        self.electronic_proj = nn.Sequential(
            nn.Linear(electronic_dim + 2, 128),
            nn.Softplus(),
            nn.Linear(128, 64)
        )


        self.predictor = nn.Sequential(
            nn.Linear(64 + 64, 256),
            nn.Softplus(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Linear(64, 1)
        )

    def forward(self, data, electronic_features, electronic_repr):


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

        structural_features = torch.stack(attended_features)
        structural_features = self.pool(structural_features)


        electronic_concat = torch.cat([
            electronic_repr,
            electronic_features['work_function'].view(-1, 1),
            electronic_features['d_band_center'].view(-1, 1)
        ], dim=-1)
        electronic_processed = self.electronic_proj(electronic_concat)


        combined = torch.cat([structural_features, electronic_processed], dim=-1)
        adsorption_energy_pred = self.predictor(combined)

        return adsorption_energy_pred


class HierarchicalCGCNN(nn.Module):


    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 32,
        global_dim: int = 16,
        electronic_dim: int = 128,
        atom_fea_len: int = 92,
        edge_fea_len: int = 1,
        use_triplet_angle: bool = False,
        angle_dim: int = 32,
    ):
        super(HierarchicalCGCNN, self).__init__()

        self.electronic_predictor = ElectronicStructurePredictor(
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=global_dim,
            atom_fea_len=atom_fea_len,
            edge_fea_len=edge_fea_len,
            use_triplet_angle=use_triplet_angle,
            angle_dim=angle_dim,
        )

        self.adsorption_energy_predictor = AdsorptionEnergyPredictor(
            node_dim=node_dim,
            edge_dim=edge_dim,
            electronic_dim=electronic_dim,
            atom_fea_len=atom_fea_len,
            edge_fea_len=edge_fea_len,
            use_triplet_angle=use_triplet_angle,
            angle_dim=angle_dim,
        )

    def forward(self, data, return_electronic: bool = False):


        electronic_features, electronic_repr = self.electronic_predictor(data)


        adsorption_energy = self.adsorption_energy_predictor(
            data, electronic_features, electronic_repr
        )

        if return_electronic:
            return adsorption_energy, electronic_features
        return adsorption_energy


class HierarchicalLoss(nn.Module):


    def __init__(
        self,
        lambda_electronic: float = 0.3,
        lambda_adsorption_energy: float = 1.0,
        lambda_constraint: float = 0.1,
        norm_stats: dict | None = None,
    ):
        super(HierarchicalLoss, self).__init__()
        self.lambda_electronic = lambda_electronic
        self.lambda_adsorption_energy = lambda_adsorption_energy
        self.lambda_constraint = lambda_constraint


        self.norm_stats = norm_stats or {}

    def _maybe_norm(self, key: str, tensor: torch.Tensor) -> torch.Tensor:

        if key not in self.norm_stats:
            return tensor
        stats = self.norm_stats[key]
        mean = float(stats.get("mean", 0.0))
        std = float(stats.get("std", 1.0))
        if std <= 0:
            std = 1.0

        mean_t = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
        std_t = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
        return (tensor - mean_t) / (std_t + 1e-12)

    def forward(self, predictions, targets):


        loss = 0.0
        loss_dict = {}


        if (
            self.lambda_adsorption_energy > 0
            and isinstance(predictions, dict)
            and isinstance(targets, dict)
            and 'adsorption_energy' in predictions
            and 'adsorption_energy' in targets
            and targets['adsorption_energy'] is not None
        ):
            pred_adsorption = predictions['adsorption_energy'].view(-1)
            tgt_adsorption = targets['adsorption_energy'].view(-1)

            pred_adsorption_n = self._maybe_norm('adsorption_energy', pred_adsorption)
            tgt_adsorption_n = self._maybe_norm('adsorption_energy', tgt_adsorption)
            adsorption_loss = F.mse_loss(pred_adsorption_n, tgt_adsorption_n)
            loss += self.lambda_adsorption_energy * adsorption_loss
            loss_dict['adsorption_energy'] = adsorption_loss.item()


        electronic_loss = 0.0
        n_electronic = 0
        for key in ['work_function', 'd_band_center']:
            if key in targets and targets[key] is not None:


                pred = predictions['electronic_features'][key]
                tgt = targets[key]
                if pred is not None:
                    pred = pred.view(-1)
                if tgt is not None:
                    tgt = tgt.view(-1)

                pred_n = self._maybe_norm(key, pred)
                tgt_n = self._maybe_norm(key, tgt)
                e_loss = F.mse_loss(pred_n, tgt_n)
                electronic_loss += e_loss
                n_electronic += 1
                loss_dict[f'electronic_{key}'] = e_loss.item()

        if n_electronic > 0:
            electronic_loss /= n_electronic
            loss += self.lambda_electronic * electronic_loss
            loss_dict['electronic_total'] = electronic_loss.item()


        constraint_loss = self._compute_constraint_loss(predictions)
        if self.lambda_constraint > 0:
            loss += self.lambda_constraint * constraint_loss
        loss_dict['constraint'] = constraint_loss.item() if torch.is_tensor(constraint_loss) else float(constraint_loss)

        loss_dict['total'] = loss.item()

        return loss, loss_dict

    def _compute_constraint_loss(self, predictions):


        if not isinstance(predictions, dict) or 'electronic_features' not in predictions:
            return torch.tensor(0.0)

        ef = predictions['electronic_features']
        if 'work_function' not in ef:
            return torch.tensor(0.0)


        wf = ef['work_function']
        return torch.mean(F.relu(-wf))


if __name__ == "__main__":
    print("=" * 60)
    print("Hierarchical CGCNN for 2D MOF Catalytic Activity Prediction")
    print("=" * 60)


    model = HierarchicalCGCNN(
        node_dim=64,
        edge_dim=32,
        global_dim=16,
        electronic_dim=128
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")


    n_atoms = 50
    n_edges = 200


    data = Data(
        x=torch.randn(n_atoms, 92),
        edge_index=torch.randint(0, n_atoms, (2, n_edges)),
        edge_attr=torch.randn(n_edges, 1),
        pos=torch.randn(n_atoms, 3),
        batch=torch.zeros(n_atoms, dtype=torch.long),
        atomic_numbers=torch.randint(1, 93, (n_atoms,)),
        coordination_numbers=torch.randint(2, 7, (n_atoms,)),
        electronegativity=torch.rand(n_atoms) * 3 + 0.5,
        cell=torch.eye(3).unsqueeze(0).expand(n_atoms, 3, 3)
    )


    print("\nRunning forward pass...")
    with torch.no_grad():
        adsorption_energy, electronic_features = model(data, return_electronic=True)

    print(f"\nPredictions:")
    print(f"  Adsorption energy: {adsorption_energy.item():.4f} eV")
    print(f"\nElectronic properties:")
    print(f"  Work function: {electronic_features['work_function'].item():.4f} eV")
    print(f"  d-band center: {electronic_features['d_band_center'].item():.4f} eV")
