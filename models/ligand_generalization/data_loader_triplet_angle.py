


import torch
from torch_geometric.data import Data, Dataset
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.cif import CifParser
from typing import List, Dict, Optional, Tuple
import os
import json
import warnings
from pathlib import Path


def _load_zval_map(root: Optional[str], zval_map_path: Optional[str]):


    if zval_map_path is None and root is not None:
        zval_map_path = str(Path(root) / "zval_map.json")

    if zval_map_path is None:
        return None, None, None

    if not os.path.isfile(zval_map_path):
        raise FileNotFoundError(
            f"Cannot find zval_map.json: {zval_map_path}\n"
            "To use valence-electron features, place zval_map.json in the data directory or specify its path on the command line."
        )

    with open(zval_map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    zval_map = {int(k): float(v) for k, v in raw.items()}
    if len(zval_map) == 0:
        raise ValueError(f"zval_map.json is empty: {zval_map_path}")

    zval_max = max(zval_map.values())
    if zval_max <= 0:
        zval_max = 1.0

    return zval_map, float(zval_max), zval_map_path


class AtomCustomJSONInitializer:


    def __init__(self, atom_init_json: str):
        if not os.path.exists(atom_init_json):
            raise FileNotFoundError(
                f"Cannot find atom_init.json: {atom_init_json}\n"
                "Copy data/sample-regression/atom_init.json from the CGCNN repository into the data directory, "
                "or specify the path using --atom-init in train.py."
            )

        with open(atom_init_json, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self._embedding = {}
        feat_len = None
        for k, v in raw.items():
            z = int(k)
            vec = torch.tensor(v, dtype=torch.float)
            if feat_len is None:
                feat_len = int(vec.numel())
            elif int(vec.numel()) != feat_len:
                raise ValueError(
                    f"Inconsistent atom_init.json vector dimensions across elements: {feat_len} vs {int(vec.numel())} (Z={z})"
                )
            self._embedding[z] = vec

        if feat_len is None:
            raise ValueError(f"atom_init.json is empty: {atom_init_json}")

        self.feat_len = feat_len

    def get_atom_fea(self, atomic_number: int) -> torch.Tensor:
        if atomic_number not in self._embedding:
            raise KeyError(
                f"atom_init.json is missing Z={atomic_number} vector.\n"
                "If the dataset contains elements with Z > 92 or other unsupported elements, extend atom_init.json."
            )
        return self._embedding[atomic_number]


ELECTRONEGATIVITY = {
    1: 2.20, 2: 0.00, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 10: 0.00,
    11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 18: 0.00,
    19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83, 27: 1.88,
    28: 1.91, 29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55, 35: 2.96, 36: 3.00,
    37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33, 41: 1.6, 42: 2.16, 43: 1.9, 44: 2.2, 45: 2.28,
    46: 2.20, 47: 1.93, 48: 1.69, 49: 1.78, 50: 1.96, 51: 2.05, 52: 2.1, 53: 2.66, 54: 2.6,
    55: 0.79, 56: 0.89, 57: 1.10, 58: 1.12, 59: 1.13, 60: 1.14, 61: 1.13, 62: 1.17, 63: 1.2,
    64: 1.2, 65: 1.1, 66: 1.22, 67: 1.23, 68: 1.24, 69: 1.25, 70: 1.1, 71: 1.27, 72: 1.3,
    73: 1.5, 74: 2.36, 75: 1.9, 76: 2.2, 77: 2.20, 78: 2.28, 79: 2.54, 80: 2.00, 81: 1.62,
    82: 2.33, 83: 2.02, 84: 2.0, 85: 2.2, 86: 0.00, 87: 0.7, 88: 0.9, 89: 1.1, 90: 1.3,
    91: 1.5, 92: 1.38
}


def _load_ionization_map(root: Optional[str], ionization_map_path: Optional[str]):


    if ionization_map_path is None and root is not None:
        ionization_map_path = str(Path(root) / "ionization_energy_map.json")

    if ionization_map_path is None:
        return None, None, None

    if not os.path.isfile(ionization_map_path):
        raise FileNotFoundError(f"Ionization energy map not found: {ionization_map_path}")

    with open(ionization_map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    ie_map = {str(k): float(v) for k, v in raw.items()}
    if len(ie_map) == 0:
        raise ValueError(f"Ionization energy map is empty: {ionization_map_path}")

    vals = np.array(list(ie_map.values()), dtype=np.float32)
    vmin, vmax = float(vals.min()), float(vals.max())
    return ie_map, vmin, vmax


class MOFDataPreprocessor:


    def __init__(
        self,
        root: Optional[str] = None,
        cutoff: float = 8.0,
        max_neighbors: int = 12,
        atom_init_path: Optional[str] = None,
        use_rbf: bool = True,
        rbf_bins: int = 64,
        rbf_cutoff: Optional[float] = None,
        rbf_gamma: Optional[float] = None,

        use_zval: bool = False,
        zval_map_path: Optional[str] = None,
        zval_normalize: str = "max",

        use_ionization_energy: bool = False,
        ionization_map_path: Optional[str] = None,
        ionization_normalize: str = "max",

        use_coordination_number: bool = True,
        coordination_normalize: str = "fixed",
        coordination_divisor: float = 12.0,
        use_surface_mask_feature: bool = True,
        use_z_position_feature: bool = True,

        use_cn: Optional[bool] = None,
        use_surface: Optional[bool] = None,
        use_znorm: Optional[bool] = None,

        use_triplet_angle: bool = False,
        angle_rbf_bins: int = 32,
        angle_gamma: float = 10.0,
        max_triplets_per_center: int = 64,
        ):


        self.cutoff = float(cutoff)
        self.max_neighbors = int(max_neighbors)
        self.cnn = CrystalNN()


        if atom_init_path is None and root is not None:
            atom_init_path = str(Path(root) / "atom_init.json")
        if atom_init_path is None:
            raise ValueError(
                "No atom_init.json path was provided. Place atom_init.json in the dataset directory, "
                "or specify it with --atom-init in train.py."
            )
        self.atom_init_path = atom_init_path
        self.atom_initializer = AtomCustomJSONInitializer(atom_init_path)


        self.use_zval = bool(use_zval)
        self.zval_normalize = str(zval_normalize).lower()
        self.zval_map, self.zval_max, self.zval_map_path = _load_zval_map(root, zval_map_path) if self.use_zval else (None, None, None)
        if self.use_zval and self.zval_normalize not in {"max", "none"}:
            raise ValueError(f"Unsupported zval_normalize: {zval_normalize}. Use 'max' or 'none'.")


        self.use_ionization_energy = bool(use_ionization_energy)
        self.ionization_normalize = str(ionization_normalize).lower()
        self.ionization_map, self.ie_min, self.ie_max = (
            _load_ionization_map(root, ionization_map_path) if self.use_ionization_energy else (None, None, None)
        )
        if self.use_ionization_energy and self.ionization_normalize not in {"max", "none"}:
            raise ValueError(
                f"Unsupported ionization_normalize: {ionization_normalize}. Use 'max' or 'none'."
            )


        self.use_coordination_number = bool(use_coordination_number)
        self.coordination_normalize = str(coordination_normalize).lower()
        self.coordination_divisor = float(coordination_divisor)

        self.use_surface_mask_feature = bool(use_surface_mask_feature)
        self.use_z_position_feature = bool(use_z_position_feature)


        if use_cn is not None:
            self.use_coordination_number = bool(use_cn)
        if use_surface is not None:
            self.use_surface_mask_feature = bool(use_surface)
        if use_znorm is not None:
            self.use_z_position_feature = bool(use_znorm)

        if self.coordination_normalize not in {"fixed", "none"}:
            raise ValueError(
                f"Unsupported coordination_normalize: {coordination_normalize}. Use 'fixed' or 'none'."
            )
        if self.coordination_normalize == "fixed" and self.coordination_divisor <= 0:
            self.coordination_divisor = 1.0


        self.use_triplet_angle = bool(use_triplet_angle)
        self.angle_rbf_bins = int(angle_rbf_bins)
        self.angle_gamma = float(angle_gamma)
        self.max_triplets_per_center = int(max_triplets_per_center)
        if self.angle_rbf_bins <= 0:
            self.angle_rbf_bins = 1
        if self.max_triplets_per_center <= 0:
            self.max_triplets_per_center = 1


        base_len = int(self.atom_initializer.feat_len)
        self.atom_fea_len = (
            base_len
            + (1 if self.use_zval else 0)
            + (1 if self.use_ionization_energy else 0)
            + (1 if self.use_coordination_number else 0)
            + (1 if self.use_surface_mask_feature else 0)
            + (1 if self.use_z_position_feature else 0)
        )


        self.use_rbf = bool(use_rbf)
        self.rbf_bins = int(rbf_bins)
        self.rbf_cutoff = float(rbf_cutoff) if rbf_cutoff is not None else float(self.cutoff)
        if self.rbf_bins <= 0:
            raise ValueError("rbf_bins must be > 0")
        self.rbf_centers = torch.linspace(0.0, self.rbf_cutoff, self.rbf_bins)
        if self.rbf_bins == 1:
            default_gamma = 1.0
        else:
            delta = float(self.rbf_centers[1] - self.rbf_centers[0])
            default_gamma = 1.0 / max(delta * delta, 1e-8)
        self.rbf_gamma = float(rbf_gamma) if rbf_gamma is not None else float(default_gamma)
        self.edge_fea_len = int(self.rbf_bins) if self.use_rbf else 1

        if not self.use_rbf:
            warnings.warn(
                "use_rbf=False: edge features contain only scalar distances, which are generally less expressive than RBF features.",
                stacklevel=2,
            )

    def cif_to_graph(self, cif_path: str, properties: Optional[Dict] = None) -> Data:


        parser = CifParser(cif_path)
        structure = parser.get_structures()[0]


        atomic_numbers = [site.specie.Z for site in structure]
        positions = structure.cart_coords


        edge_index, edge_attr, distances = self._build_edges(structure)


        coordination_numbers = self._compute_coordination_numbers(structure)


        electronegativity = torch.tensor([
            ELECTRONEGATIVITY.get(z, 0.0) for z in atomic_numbers
        ], dtype=torch.float)


        atom_features = self._encode_atoms(atomic_numbers)


        cell = torch.tensor(structure.lattice.matrix, dtype=torch.float)


        surface_mask = self._identify_surface_atoms(positions)


        extra_node_feats = []
        n_atoms = len(atomic_numbers)

        if self.use_coordination_number:
            cn = torch.as_tensor(coordination_numbers, dtype=torch.float)
            if self.coordination_normalize == "fixed":
                cn = cn / float(self.coordination_divisor)
            extra_node_feats.append(cn.view(n_atoms, 1))

        if self.use_surface_mask_feature:
            sf = surface_mask.to(torch.float).view(n_atoms, 1)
            extra_node_feats.append(sf)

        if self.use_z_position_feature:
            z = torch.tensor(positions[:, 2], dtype=torch.float)
            z_min = float(z.min().item())
            z_max = float(z.max().item())
            denom = (z_max - z_min) if (z_max - z_min) > 1e-8 else 1.0
            z_norm = ((z - z_min) / denom).view(n_atoms, 1)
            extra_node_feats.append(z_norm)

        if len(extra_node_feats) > 0:
            atom_features = torch.cat([atom_features] + extra_node_feats, dim=1)


        data = Data(
            x=atom_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=torch.tensor(positions, dtype=torch.float),
            atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
            coordination_numbers=(cn if self.use_coordination_number else None),
            electronegativity=(torch.as_tensor(electronegativity, dtype=torch.float) if electronegativity is not None else None),
            cell=cell.unsqueeze(0).expand(len(atomic_numbers), 3, 3),
            surface_mask=(surface_mask if isinstance(surface_mask, torch.Tensor) else torch.as_tensor(surface_mask, dtype=torch.float)),
            distances=distances
        )


        if self.use_triplet_angle:


            import inspect
            num_atoms = int(atom_features.size(0))
            edge_vecs = data.pos[edge_index[1]] - data.pos[edge_index[0]]

            try:
                n_params = len(inspect.signature(self._build_triplets).parameters)
            except Exception:
                n_params = 0

            if n_params == 4:

                triplet_index, angle_attr = self._build_triplets(edge_index, edge_vecs, num_atoms)
            else:

                triplet_index, angle_attr = self._build_triplets(edge_index, data.pos, distances)

            data.triplet_index = triplet_index
            data.angle_attr = angle_attr


        data.sample_id = os.path.splitext(os.path.basename(cif_path))[0]


        def _safe_float_label(props, key):
            if props is None:
                return float("nan")
            if key not in props or props[key] is None:
                return float("nan")
            try:
                v = float(props[key])
                return v if np.isfinite(v) else float("nan")
            except Exception:
                return float("nan")


        target_value = _safe_float_label(properties, "adsorption_energy")
        if not np.isfinite(target_value):
            target_value = _safe_float_label(properties, "overpotential")
        data.y = torch.tensor([target_value], dtype=torch.float)
        data.adsorption_energy = torch.tensor([_safe_float_label(properties, "adsorption_energy")], dtype=torch.float)
        data.overpotential = torch.tensor([_safe_float_label(properties, "overpotential")], dtype=torch.float)


        data.work_function = torch.tensor([_safe_float_label(properties, "work_function")], dtype=torch.float)
        data.d_band_center = torch.tensor([_safe_float_label(properties, "d_band_center")], dtype=torch.float)

        return data

    def _distance_to_rbf(self, distances: torch.Tensor) -> torch.Tensor:


        d = distances.view(-1, 1)
        c = self.rbf_centers.view(1, -1).to(d.device)
        return torch.exp(-self.rbf_gamma * (d - c) ** 2)


def _angle_to_rbf(self, angles: torch.Tensor) -> torch.Tensor:


    a = angles.view(-1, 1)
    centers = torch.linspace(0.0, float(np.pi), self.angle_rbf_bins, device=a.device).view(1, -1)
    return torch.exp(-self.angle_gamma * (a - centers) ** 2)

def _build_triplets(self, edge_index: torch.Tensor, pos: torch.Tensor, distances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:


    src = edge_index[0]
    dst = edge_index[1]
    device = pos.device

    triplets = []
    angles = []


    n_nodes = int(pos.size(0))
    for i in range(n_nodes):
        mask = (src == i)
        idx = torch.nonzero(mask, as_tuple=False).view(-1)
        if idx.numel() < 2:
            continue

        d_i = distances[idx]
        order = torch.argsort(d_i)
        idx = idx[order]
        neigh = dst[idx].tolist()


        count = 0
        for a_i in range(len(neigh)):
            for b_i in range(a_i + 1, len(neigh)):
                j = int(neigh[a_i])
                k = int(neigh[b_i])

                vj = pos[j] - pos[i]
                vk = pos[k] - pos[i]
                denom = (vj.norm() * vk.norm()).clamp(min=1e-8)
                cosang = torch.dot(vj, vk) / denom
                cosang = cosang.clamp(-1.0, 1.0)
                ang = torch.acos(cosang)
                triplets.append((i, j, k))
                angles.append(ang)
                count += 1
                if count >= self.max_triplets_per_center:
                    break
            if count >= self.max_triplets_per_center:
                break

    if len(triplets) == 0:
        triplet_index = torch.empty((3, 0), dtype=torch.long, device=device)
        angle_attr = torch.empty((0, self.angle_rbf_bins), dtype=torch.float, device=device)
        return triplet_index, angle_attr

        triplet_index = torch.tensor(triplets, dtype=torch.long, device=device).t().contiguous()
    angles_t = torch.stack(angles).to(device=device, dtype=torch.float)
    angle_attr = self._angle_to_rbf(angles_t)
    return triplet_index, angle_attr


    def _build_edges(self, structure: Structure) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        edge_index_list = []
        edge_attr_list = []
        distance_list = []


        for i, site_i in enumerate(structure):
            try:
                neighbors = structure.get_neighbors(site_i, self.cutoff)


                if len(neighbors) > self.max_neighbors:

                    neighbors = sorted(neighbors, key=lambda x: x.nn_distance)[:self.max_neighbors]

                for neighbor_info in neighbors:


                    try:
                        neighbor_site = neighbor_info.site
                        distance = neighbor_info.nn_distance
                    except AttributeError:

                        if hasattr(neighbor_info, '__getitem__'):
                            neighbor_site = neighbor_info[0]
                            distance = neighbor_info[1]
                        else:
                            continue

                    j = structure.index(neighbor_site)
                    edge_index_list.append([i, j])
                    edge_attr_list.append([distance])
                    distance_list.append(distance)
            except Exception as e:

                continue

        if len(edge_index_list) == 0:

            n_atoms = len(structure)
            edge_index_list = [[i, i] for i in range(n_atoms)]
            edge_attr_list = [[0.0] for _ in range(n_atoms)]
            distance_list = [0.0 for _ in range(n_atoms)]

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
        distances = torch.tensor(distance_list, dtype=torch.float)

        if self.use_rbf:
            edge_attr = self._distance_to_rbf(distances)
        else:
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

        return edge_index, edge_attr, distances

    def _compute_coordination_numbers(self, structure: Structure) -> torch.Tensor:

        coord_numbers = []
        for i, site in enumerate(structure):
            try:

                nn_info = self.cnn.get_nn_info(structure, i)
                coord_numbers.append(len(nn_info))
            except:

                neighbors = structure.get_neighbors(site, 3.5)
                coord_numbers.append(len(neighbors))

        return torch.tensor(coord_numbers, dtype=torch.long)

    def _get_zval_feature(self, atomic_number: int) -> float:
        if not self.use_zval:
            raise RuntimeError("use_zval is False but _get_zval_feature was called")
        if self.zval_map is None:
            raise RuntimeError("use_zval=True but zval_map is not loaded")
        if atomic_number not in self.zval_map:
            raise KeyError(
                f"zval_map.json is missing Z={atomic_number} ZVAL.\n"
                "Complete zval_map.json, preferably using the ZVAL values from the actual POTCAR files."
            )
        zval = float(self.zval_map[atomic_number])
        if self.zval_normalize == "max":
            zval = zval / float(self.zval_max)
        return float(zval)

    def _get_ionization_feature(self, atomic_number: int) -> float:
        if not self.use_ionization_energy:
            raise RuntimeError("use_ionization_energy is False but _get_ionization_feature was called")
        if self.ionization_map is None:
            raise RuntimeError("use_ionization_energy=True but ionization_map is not loaded")
        key = str(int(atomic_number))
        if key not in self.ionization_map:
            raise KeyError(f"Ionization energy missing for atomic number {key} in ionization map")
        val = float(self.ionization_map[key])
        if self.ionization_normalize == "none":
            return val

        if self.ie_max is None or self.ie_max == 0:
            return val
        return val / float(self.ie_max)

    def _encode_atoms(self, atomic_numbers: List[int]) -> torch.Tensor:

        feats = []
        for z in atomic_numbers:
            z_int = int(z)
            fea = self.atom_initializer.get_atom_fea(z_int)
            if self.use_zval:
                zval = self._get_zval_feature(z_int)
                fea = torch.cat([fea, torch.tensor([zval], dtype=torch.float)], dim=0)
            if self.use_ionization_energy:
                ie = self._get_ionization_feature(z_int)
                fea = torch.cat([fea, torch.tensor([ie], dtype=torch.float)], dim=0)
            feats.append(fea)
        return torch.stack(feats, dim=0)

    def _identify_surface_atoms(self, positions: np.ndarray,
                                threshold: float = 2.0) -> torch.Tensor:


        z_coords = positions[:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()


        surface_mask = (z_coords <= z_min + threshold) | (z_coords >= z_max - threshold)
        return torch.tensor(surface_mask, dtype=torch.bool)


class MOFDataset(Dataset):


    def __init__(
        self,
        root: str,
        cif_files: List[str],
        properties_dict: Optional[Dict[str, Dict]] = None,
        transform=None,
        pre_transform=None,

        cutoff: float = 8.0,
        max_neighbors: int = 12,
        atom_init_path: Optional[str] = None,
        use_rbf: bool = True,
        rbf_bins: int = 64,
        rbf_cutoff: Optional[float] = None,
        rbf_gamma: Optional[float] = None,

        use_zval: bool = False,
        zval_map_path: Optional[str] = None,
        zval_normalize: str = "max",

        use_ionization_energy: bool = False,
        ionization_map_path: Optional[str] = None,
        ionization_normalize: str = "max",

        use_coordination_number: bool = True,
        coordination_normalize: str = "fixed",
        coordination_divisor: float = 12.0,
        use_surface_mask_feature: bool = True,
        use_z_position_feature: bool = True,

        use_cn: Optional[bool] = None,
        use_surface: Optional[bool] = None,
        use_znorm: Optional[bool] = None,

        use_triplet_angle: bool = False,
        angle_rbf_bins: int = 32,
        angle_gamma: float = 10.0,
        max_triplets_per_center: int = 64,
        ):


        self.cif_files = cif_files
        self.properties_dict = properties_dict or {}
        self.preprocessor = MOFDataPreprocessor(
            root=root,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            atom_init_path=atom_init_path,
            use_rbf=use_rbf,
            rbf_bins=rbf_bins,
            rbf_cutoff=rbf_cutoff,
            rbf_gamma=rbf_gamma,
            use_zval=use_zval,
            zval_map_path=zval_map_path,
            zval_normalize=zval_normalize,
            use_ionization_energy=use_ionization_energy,
            ionization_map_path=ionization_map_path,
            ionization_normalize=ionization_normalize,
            use_coordination_number=use_coordination_number,
            coordination_normalize=coordination_normalize,
            coordination_divisor=coordination_divisor,
            use_surface_mask_feature=use_surface_mask_feature,
            use_z_position_feature=use_z_position_feature,
            use_cn=use_cn,
            use_surface=use_surface,
            use_znorm=use_znorm,
        )


        self.atom_fea_len = int(self.preprocessor.atom_fea_len)
        self.edge_fea_len = int(self.preprocessor.edge_fea_len)

        super(MOFDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.cif_files

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.cif_files))]

    def download(self):
        pass

    def process(self):
        for idx, cif_file in enumerate(self.cif_files):

            cif_name = os.path.basename(cif_file)
            properties = self.properties_dict.get(cif_name, None)


            data = self.preprocessor.cif_to_graph(cif_file, properties)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))

    def len(self):
        return len(self.cif_files)

    def get(self, idx):

        data = torch.load(
            os.path.join(self.processed_dir, f'data_{idx}.pt'),
            weights_only=False
        )
        return data


def create_synthetic_dataset(n_samples: int = 100,
                            save_dir: str = "./data/synthetic_mof") -> MOFDataset:


    os.makedirs(save_dir, exist_ok=True)


    atom_init_path = os.path.join(save_dir, "atom_init.json")
    if not os.path.exists(atom_init_path):
        ident = {}
        for z in range(1, 93):
            vec = [0.0] * 92
            vec[z - 1] = 1.0
            ident[str(z)] = vec
        with open(atom_init_path, "w", encoding="utf-8") as f:
            json.dump(ident, f)
    os.makedirs(os.path.join(save_dir, "raw"), exist_ok=True)

    cif_files = []
    properties_dict = {}

    for i in range(n_samples):

        a = np.random.uniform(10, 15)
        b = np.random.uniform(10, 15)
        c = np.random.uniform(20, 30)


        n_atoms = np.random.randint(20, 50)
        species = np.random.choice(['C', 'N', 'O', 'H', 'Zn', 'Cu'], n_atoms)


        coords = []
        for _ in range(n_atoms):
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)
            z = np.random.normal(0.5, 0.05)
            coords.append([x, y, z])

        structure = Structure(
            lattice=[[a, 0, 0], [0, b, 0], [0, 0, c]],
            species=species,
            coords=coords
        )


        cif_path = os.path.join(save_dir, "raw", f"mof_{i}.cif")
        structure.to(filename=cif_path)
        cif_files.append(cif_path)


        properties_dict[f"mof_{i}.cif"] = {
            'overpotential': np.random.uniform(0.1, 1.5),
            'work_function': np.random.uniform(4.0, 6.0),
            'd_band_center': np.random.uniform(-3.0, -1.0)
        }


    dataset = MOFDataset(
        root=save_dir,
        cif_files=cif_files,
        properties_dict=properties_dict
    )

    return dataset


class MOFAugmentation:


    @staticmethod
    def rotate_in_plane(data: Data, angle: float) -> Data:

        theta = np.radians(angle)
        rotation_matrix = torch.tensor([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ], dtype=torch.float)

        data.pos = torch.matmul(data.pos, rotation_matrix.t())
        return data

    @staticmethod
    def add_noise(data: Data, noise_level: float = 0.05) -> Data:

        noise = torch.randn_like(data.pos) * noise_level
        data.pos = data.pos + noise
        return data


if __name__ == "__main__":
    print("=" * 60)
    print("2D MOF data loading and preprocessing module")
    print("=" * 60)


    print("\nCreating a synthetic dataset...")
    dataset = create_synthetic_dataset(n_samples=10)

    print(f"Dataset size: {len(dataset)}")


    sample = dataset[0]
    print(f"\nFirst sample:")
    print(f"  Number of atoms: {sample.x.shape[0]}")
    print(f"  Number of edges: {sample.edge_index.shape[1]}")
    print(f"  Overpotential: {sample.y.item():.4f} V")
    print(f"  Work function: {sample.work_function.item():.4f} eV")
    print(f"  Number of surface atoms: {sample.surface_mask.sum().item()}")


    print("\nData augmentation example:")
    augmented = MOFAugmentation.rotate_in_plane(sample, angle=45)
    print(f"  Position change after rotation: {torch.norm(sample.pos - augmented.pos).item():.4f}")

    print("\n" + "=" * 60)
    print("Data preprocessing completed!")
    print("=" * 60)


def _fallback_build_edges(self, structure: Structure):

    edge_index_list = []
    edge_attr_list = []
    distance_list = []

    for i, site_i in enumerate(structure):
        neighbors = structure.get_neighbors(site_i, self.cutoff)
        if len(neighbors) > self.max_neighbors:
            neighbors = sorted(neighbors, key=lambda x: x.nn_distance)[: self.max_neighbors]
        for nb in neighbors:
            j = int(nb.index)
            d = float(nb.nn_distance)

            try:
                vec = structure[j].coords - site_i.coords
            except Exception:
                vec = np.array([0.0, 0.0, 0.0], dtype=float)

            edge_index_list.append([i, j])
            distance_list.append(d)

            if getattr(self, "use_rbf", True):

                if hasattr(self, "_distance_to_rbf"):
                    edge_attr_list.append(self._distance_to_rbf(torch.tensor([d], dtype=torch.float)).squeeze(0).cpu().numpy())
                else:
                    edge_attr_list.append([d])
            else:
                edge_attr_list.append([d])

    if len(edge_index_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, getattr(self, "rbf_bins", 1) if getattr(self, "use_rbf", True) else 1), dtype=torch.float)
        distances = torch.empty((0,), dtype=torch.float)
        return edge_index, edge_attr, distances

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    distances = torch.tensor(distance_list, dtype=torch.float)
    return edge_index, edge_attr, distances


def _fallback_encode_atoms(self, atomic_numbers):


    feats = []
    for z in atomic_numbers:
        z_int = int(z)
        if hasattr(self, "atom_initializer") and self.atom_initializer is not None and hasattr(self.atom_initializer, "get_atom_fea"):
            fea = self.atom_initializer.get_atom_fea(z_int)
            if not torch.is_tensor(fea):
                fea = torch.tensor(fea, dtype=torch.float)
        else:
            fea = torch.tensor([float(z_int)], dtype=torch.float)

        if getattr(self, "use_zval", False):
            try:
                zval = self._get_zval_feature(z_int) if hasattr(self, "_get_zval_feature") else 0.0
            except Exception:
                zval = 0.0
            fea = torch.cat([fea, torch.tensor([float(zval)], dtype=torch.float)], dim=0)

        if getattr(self, "use_ionization_energy", False):
            try:
                ie = self._get_ionization_feature(z_int) if hasattr(self, "_get_ionization_feature") else 0.0
            except Exception:
                ie = 0.0
            fea = torch.cat([fea, torch.tensor([float(ie)], dtype=torch.float)], dim=0)

        feats.append(fea)

    if len(feats) == 0:
        return torch.empty((0, 0), dtype=torch.float)

    return torch.stack(feats, dim=0)

def _fallback_compute_coordination_numbers(self, structure: Structure):

    try:
        cnn = CrystalNN()
        cn = np.zeros(len(structure), dtype=float)
        for i in range(len(structure)):
            try:
                cn[i] = float(len(cnn.get_nn_info(structure, i)))
            except Exception:
                cn[i] = 0.0
        return cn
    except Exception:
        cn = np.zeros(len(structure), dtype=float)
        for i, site_i in enumerate(structure):
            try:
                cn[i] = float(len(structure.get_neighbors(site_i, self.cutoff)))
            except Exception:
                cn[i] = 0.0
        return cn


def _fallback_identify_surface_atoms(self, positions):


    try:
        if isinstance(positions, torch.Tensor):
            pos = positions.detach().cpu().numpy()
        else:
            pos = np.asarray(positions)
        if pos.ndim != 2 or pos.shape[1] < 3:

            n = int(pos.shape[0]) if pos.ndim == 2 else 0
            return torch.zeros(n, dtype=torch.bool)
        z = pos[:, 2].astype(float)
        z_min = float(np.min(z))
        z_max = float(np.max(z))
        thr = float(getattr(self, "surface_threshold", 2.0))
        mask = (z <= z_min + thr) | (z >= z_max - thr)
        return torch.as_tensor(mask, dtype=torch.bool)
    except Exception:
        try:
            n = int(positions.size(0)) if isinstance(positions, torch.Tensor) else int(len(positions))
        except Exception:
            n = 0
        return torch.zeros(n, dtype=torch.bool)


try:

    if not hasattr(MOFDataPreprocessor, "_build_edges"):
        MOFDataPreprocessor._build_edges = _fallback_build_edges
    if not hasattr(MOFDataPreprocessor, "build_edges"):
        MOFDataPreprocessor.build_edges = MOFDataPreprocessor._build_edges


    if not hasattr(MOFDataPreprocessor, "_encode_atoms"):
        MOFDataPreprocessor._encode_atoms = _fallback_encode_atoms
    if not hasattr(MOFDataPreprocessor, "encode_atoms"):
        MOFDataPreprocessor.encode_atoms = MOFDataPreprocessor._encode_atoms


    if not hasattr(MOFDataPreprocessor, "_identify_surface_atoms"):
        MOFDataPreprocessor._identify_surface_atoms = _fallback_identify_surface_atoms
    if not hasattr(MOFDataPreprocessor, "identify_surface_atoms"):
        MOFDataPreprocessor.identify_surface_atoms = MOFDataPreprocessor._identify_surface_atoms


    if not hasattr(MOFDataPreprocessor, "_compute_coordination_numbers"):
        MOFDataPreprocessor._compute_coordination_numbers = _fallback_compute_coordination_numbers
    if not hasattr(MOFDataPreprocessor, "compute_coordination_numbers"):
        MOFDataPreprocessor.compute_coordination_numbers = MOFDataPreprocessor._compute_coordination_numbers
except Exception:
    pass
