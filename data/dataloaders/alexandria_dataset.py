from torch_geometric.data import Data, Dataset
from ase.neighborlist import neighbor_list
import torch
import numpy as np
import os
from tqdm import tqdm

from .alexandria_db_reader import AlexandriaDatabase


class AlexandriaDataset(Dataset):
    # Only structure information is used in this dataset.
    def __init__(self, db_path: str, cutoff: float):
        self.db_path = db_path
        self.cutoff = cutoff
        # Cache edge_index and edge_vec to avoid repeated neighbor list calculation
        self.neighbor_list_path = os.path.join(
            os.path.dirname(self.db_path),
            "neighbor_list",
        )
        self.neighbor_list_filename = os.path.join(
            self.neighbor_list_path,
            f"alexandria_neighbor_list_cutoff_{self.cutoff:.2f}.pt",
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
                    # Get neighbor list with images for PBC correction
                    neighbor_list = structure.get_neighbor_list(self.cutoff)
                    # Format: (center_indices, site_indices, images, distances)
                    edge_index = torch.tensor(
                        np.array(neighbor_list[:2]),
                        dtype=torch.long,
                    )
                    images = torch.tensor(
                        neighbor_list[2],
                        dtype=torch.float32,
                    )
                    self.cached_neighbor_data.append((edge_index, images))
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
            edge_index, images = self.cached_neighbor_data[idx]
            src, dst = edge_index
            edge_vec = atom_coords[dst] - atom_coords[src]
            
            # # Calculate edge vectors considering periodic boundary conditions
            # src, dst = edge_index
            # pos_src = atom_coords[src]
            # pos_dst = atom_coords[dst]
            
            # # Apply periodic boundary correction using images
            # # images shape: (num_edges, 3), lattice_mat shape: (3, 3)
            # edge_vec = (pos_dst + images @ torch.tensor(lattice_mat, dtype=torch.float32)) - pos_src

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
