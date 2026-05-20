from torch_geometric.data import Data, Dataset
from ase.neighborlist import neighbor_list
import torch
import numpy as np
import os
from tqdm import tqdm

from .materials_db_reader import MaterialsProjectDatabase


def get_ase_gmtnet_neighbor_list(atoms, cutoff: float, max_neighbors: int = 12):
    current_cutoff = cutoff
    while True:
        src, dst, displacement = neighbor_list("ijD", atoms, current_cutoff)
        counts = np.bincount(src, minlength=len(atoms))
        if len(atoms) == 0 or counts.min() >= max_neighbors:
            break
        cell_lengths = atoms.cell.lengths()
        next_cutoff = max(cell_lengths) if current_cutoff < max(cell_lengths) else 2 * current_cutoff
        current_cutoff = next_cutoff

    src_nodes = []
    dst_nodes = []
    edge_vecs = []
    for atom_idx in range(len(atoms)):
        edge_ids = np.where(src == atom_idx)[0]
        if len(edge_ids) == 0:
            continue
        distances = np.linalg.norm(displacement[edge_ids], axis=1)
        order = np.argsort(distances)
        edge_ids = edge_ids[order]
        distances = distances[order]
        max_dist = distances[min(max_neighbors, len(distances)) - 1]
        selected = edge_ids[distances <= max_dist]
        src_nodes.extend(src[selected].tolist())
        dst_nodes.extend(dst[selected].tolist())
        edge_vecs.extend(displacement[selected].tolist())

    if len(src_nodes) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_vec = torch.empty((0, 3), dtype=torch.float32)
    else:
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_vec = torch.tensor(np.array(edge_vecs), dtype=torch.float32).reshape(-1, 3)
    return edge_index, edge_vec


class MPDataset(Dataset):
    # Only structure information is used in this dataset.
    def __init__(
        self,
        db_path: str,
        cutoff: float,
        num_perturb: int = 1,
        graph_mode: str = "high_order",
        max_neighbors: int = 12,
    ):
        self.db_path = db_path
        self.cutoff = cutoff
        self.graph_mode = graph_mode
        self.max_neighbors = max_neighbors
        # Deprecated parameter, do no use
        self.num_perturb = num_perturb
        # Cache edge_index and edge_vec to avoid repeated neighbor list calculation
        self.neighbor_list_path = os.path.join(
            os.path.dirname(self.db_path),
            "neighbor_list",
        )
        cache_name = f"mp_neighbor_list_{self.graph_mode}_cutoff_{self.cutoff:.2f}"
        if self.graph_mode == "gmtnet":
            cache_name += f"_max_neighbors_{self.max_neighbors}"
        self.neighbor_list_filename = os.path.join(
            self.neighbor_list_path,
            f"{cache_name}.pt",
        )
        os.makedirs(self.neighbor_list_path, exist_ok=True)
        if not os.path.exists(self.neighbor_list_filename):
            self.cached_neighbor_data = self._build_cache()
        else:
            self.cached_neighbor_data = torch.load(self.neighbor_list_filename)
            if not self._validate_cache():
                print("Invalid MP neighbor-list cache. Regenerating...")
                self.cached_neighbor_data = self._build_cache()
        
        # Filter non-empty structures
        self.valid_indices = []
        for i in range(len(self.cached_neighbor_data)):
            if self.cached_neighbor_data[i] is not None:
                self.valid_indices.append(i)

    def _validate_cache(self):
        with MaterialsProjectDatabase(self.db_path) as db:
            if len(self.cached_neighbor_data) != db.get_row_count():
                return False
            for i, cached in enumerate(self.cached_neighbor_data):
                atoms = db.get_atoms_by_id(i)
                if atoms is None:
                    if cached is not None:
                        return False
                    continue
                if cached is None or len(cached) != 2:
                    return False
                edge_index, edge_vec = cached
                if not isinstance(edge_index, torch.Tensor) or not isinstance(edge_vec, torch.Tensor):
                    return False
                if edge_index.dim() != 2 or edge_index.shape[0] != 2:
                    return False
                if edge_vec.dim() != 2 or edge_vec.shape[0] != edge_index.shape[1] or edge_vec.shape[1] != 3:
                    return False
                if edge_index.numel() > 0 and edge_index.max().item() >= len(atoms):
                    return False
        return True

    def _build_cache(self):
        cached_neighbor_data = []
        print("Calculating neighbor list for MP dataset...")
        with MaterialsProjectDatabase(self.db_path) as db:
            for i in tqdm(range(db.get_row_count())):
                atoms = db.get_atoms_by_id(i)
                if atoms is None:
                    cached_neighbor_data.append(None)
                    continue
                if self.graph_mode == "gmtnet":
                    edge_index, edge_vec = get_ase_gmtnet_neighbor_list(
                        atoms,
                        self.cutoff,
                        self.max_neighbors,
                    )
                else:
                    src, dst, displacement = neighbor_list("ijD", atoms, self.cutoff)
                    edge_index = torch.tensor(
                        np.array([src, dst]),
                        dtype=torch.long,
                    )
                    edge_vec = torch.tensor(
                        displacement,
                        dtype=torch.float32,
                    ).reshape(-1, 3)
                cached_neighbor_data.append((edge_index, edge_vec))
        torch.save(cached_neighbor_data, self.neighbor_list_filename)
        return cached_neighbor_data

    def __len__(self) -> int:
        # Only non-empty structures can be used
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Data:
        # convert to valid index
        idx = self.valid_indices[idx]
        with MaterialsProjectDatabase(self.db_path) as db:
            # atoms: ase.Atoms
            atoms = db.get_atoms_by_id(idx)
            if atoms is None:
                raise ValueError(f"No atoms data found for index {idx}")
            # atom_type: (num_nodes,)
            atom_type = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
            num_nodes = int(atom_type.shape[0])
            # atom_coords: (num_nodes, 3)
            atom_coords = torch.tensor(atoms.get_positions(), dtype=torch.float32)
            # edge_index and edge_vec from cache (already PBC corrected)
            edge_index, edge_vec = self.cached_neighbor_data[idx]

            # Perturbate the coordinates
            unstable_atom_coords = atom_coords + torch.randn_like(atom_coords)
            src, dst = edge_index
            delta = unstable_atom_coords - atom_coords
            unstable_edge_vec = edge_vec + delta[dst] - delta[src]
            if edge_index.numel() > 0:
                assert edge_index.max().item() < atom_type.shape[0]
            assert edge_vec.shape[0] == edge_index.shape[1]
            assert unstable_edge_vec.shape[0] == edge_index.shape[1]
            # no batch dimension because num_atoms varies between structures
            force = (unstable_atom_coords - atom_coords)

            return Data(
                atom_type=atom_type,
                edge_index=edge_index,
                edge_vec=edge_vec,
                atom_coords=atom_coords,
                unstable_atom_coords=unstable_atom_coords,
                unstable_edge_vec=unstable_edge_vec,
                force=force,
                num_nodes=num_nodes,
            )
