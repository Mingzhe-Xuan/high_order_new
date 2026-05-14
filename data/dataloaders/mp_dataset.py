from torch_geometric.data import Data, Dataset
from ase.neighborlist import neighbor_list
import torch
import numpy as np
import os
from tqdm import tqdm

from .materials_db_reader import MaterialsProjectDatabase


class MPDataset(Dataset):
    # Only structure information is used in this dataset.
    def __init__(self, db_path: str, cutoff: float, num_perturb: int = 1):
        self.db_path = db_path
        self.cutoff = cutoff
        # Deprecated parameter, do no use
        self.num_perturb = num_perturb
        # Cache edge_index and edge_vec to avoid repeated neighbor list calculation
        self.neighbor_list_path = os.path.join(
            os.path.dirname(self.db_path),
            "neighbor_list",
        )
        self.neighbor_list_filename = os.path.join(
            self.neighbor_list_path,
            f"mp_neighbor_list_cutoff_{self.cutoff:.2f}.pt",
        )
        os.makedirs(self.neighbor_list_path, exist_ok=True)
        if not os.path.exists(self.neighbor_list_filename):
            self.cached_neighbor_data = []
            print("Calculating neighbor list for MP dataset...")
            with MaterialsProjectDatabase(self.db_path) as db:
                for i in tqdm(range(db.get_row_count())):
                    atoms = db.get_atoms_by_id(i)
                    if atoms is None:
                        self.cached_neighbor_data.append(None)
                        continue
                    # Use 'ijd' format to get source, destination, and displacement vectors
                    # The displacement vectors already account for periodic boundary conditions
                    neighbor_result = neighbor_list("ijD", atoms, self.cutoff)
                    src, dst, displacement = neighbor_result
                    edge_index = torch.tensor(
                        np.array([src, dst]),
                        dtype=torch.long,
                    )
                    edge_vec = torch.tensor(
                        displacement,
                        dtype=torch.float32,
                    ).reshape(-1, 3)
                    self.cached_neighbor_data.append((edge_index, edge_vec))
            torch.save(
                self.cached_neighbor_data,
                self.neighbor_list_filename,
            )
        else:
            self.cached_neighbor_data = torch.load(self.neighbor_list_filename)
        
        # Filter non-empty structures
        self.valid_indices = []
        for i in range(len(self.cached_neighbor_data)):
            if self.cached_neighbor_data[i] is not None:
                self.valid_indices.append(i)

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
            num_nodes = torch.tensor(atom_type.shape[0], dtype=torch.long)
            # atom_coords: (num_nodes, 3)
            atom_coords = torch.tensor(atoms.get_positions(), dtype=torch.float32)
            # edge_index and edge_vec from cache (already PBC corrected)
            edge_index, edge_vec = self.cached_neighbor_data[idx]

            # Perturbate the coordinates
            unstable_atom_coords = atom_coords + torch.randn_like(atom_coords)
            src, dst = edge_index
            unstable_edge_vec = unstable_atom_coords[dst] - unstable_atom_coords[src]
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
