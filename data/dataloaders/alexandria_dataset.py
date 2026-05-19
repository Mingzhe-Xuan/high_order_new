from torch_geometric.data import Data, Dataset
from ase.neighborlist import neighbor_list
import torch
import numpy as np
import os
from tqdm import tqdm

from .alexandria_db_reader import AlexandriaDatabase
from .tensor_dataset import get_gmtnet_neighbor_list, get_neighbor_list


class AlexandriaDataset(Dataset):
    # Only structure information is used in this dataset.
    def __init__(
        self,
        db_path: str,
        cutoff: float,
        graph_mode: str = "high_order",
        max_neighbors: int = 12,
    ):
        self.db_path = db_path
        self.cutoff = cutoff
        self.graph_mode = graph_mode
        self.max_neighbors = max_neighbors
        # Cache edge_index and edge_vec to avoid repeated neighbor list calculation
        self.neighbor_list_path = os.path.join(
            os.path.dirname(self.db_path),
            "neighbor_list",
        )
        cache_name = f"alexandria_neighbor_list_{self.graph_mode}_cutoff_{self.cutoff:.2f}"
        if self.graph_mode == "gmtnet":
            cache_name += f"_max_neighbors_{self.max_neighbors}"
        self.neighbor_list_filename = os.path.join(
            self.neighbor_list_path,
            f"{cache_name}.pt",
        )
        os.makedirs(self.neighbor_list_path, exist_ok=True)
        if not os.path.exists(self.neighbor_list_filename):
            self.cached_neighbor_data = []
            print("Calculating neighbor list for Alexandria dataset...")
            with AlexandriaDatabase(self.db_path) as db:
                for i in tqdm(range(db.get_row_count())):
                    structure = db.get_structure_by_id(i)
                    if structure is None:
                        self.cached_neighbor_data.append(None)
                        continue
                    atom_coords = torch.tensor(structure.cart_coords, dtype=torch.float32)
                    lattice_mat = structure.lattice.matrix
                    if self.graph_mode == "gmtnet":
                        edge_index, edge_vec = get_gmtnet_neighbor_list(
                            structure,
                            self.cutoff,
                            self.max_neighbors,
                        )
                    else:
                        edge_index, edge_vec = get_neighbor_list(
                            atom_coords,
                            self.cutoff,
                            lattice_mat,
                        )
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
        with AlexandriaDatabase(self.db_path) as db:
            structure = db.get_structure_by_id(idx)
            if structure is None:
                raise ValueError(f"No structure data found for index {idx}")
            # atom_type: (num_nodes,)
            atom_type = torch.tensor(structure.atomic_numbers, dtype=torch.long)
            num_nodes = torch.tensor(atom_type.shape[0], dtype=torch.long)
            # atom_coords: (num_nodes,3)
            atom_coords = torch.tensor(structure.cart_coords, dtype=torch.float32)
            # lattice_mat: (3, 3)
            lattice_mat = structure.lattice.matrix
            
            # Get cached neighbor data
            edge_index, edge_vec = self.cached_neighbor_data[idx]

            # Perturbate the coordinates
            unstable_atom_coords = atom_coords + torch.randn_like(atom_coords)
            src, dst = edge_index
            delta = unstable_atom_coords - atom_coords
            unstable_edge_vec = edge_vec + delta[dst] - delta[src]
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
