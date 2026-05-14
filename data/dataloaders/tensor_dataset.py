import pickle as pkl
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
import os
from tqdm import tqdm


def get_neighbor_list(
    atom_coords: torch.Tensor, cutoff: float, lattice_mat: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate neighbor list and edge vectors considering periodic boundary conditions.
    
    Returns:
        edge_index: (2, num_edges) - source and destination node indices
        edge_vec: (num_edges, 3) - edge vectors considering PBC
    """
    num_nodes = atom_coords.shape[0]
    cell = torch.tensor(lattice_mat, dtype=atom_coords.dtype, device=atom_coords.device)

    shifts = torch.tensor(
        [[dx, dy, dz] for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1]],
        dtype=atom_coords.dtype,
        device=atom_coords.device,
    )

    offsets = shifts @ cell

    pos_i = atom_coords.unsqueeze(1).unsqueeze(2)
    pos_j = atom_coords.unsqueeze(0).unsqueeze(2)
    offsets_expanded = offsets.unsqueeze(0).unsqueeze(0)

    pos_j_pbc = pos_j + offsets_expanded

    diff = pos_j_pbc - pos_i
    dists = torch.norm(diff, dim=-1)

    min_dists, min_shift_idx = torch.min(dists, dim=2)

    mask = (min_dists <= cutoff) & (min_dists > 0)

    src, dst = torch.where(mask)

    if len(src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=atom_coords.device)
        edge_vec = torch.empty((0, 3), dtype=atom_coords.dtype, device=atom_coords.device)
    else:
        edge_index = torch.stack([src, dst], dim=0)
        
        # Calculate edge vectors considering periodic boundary conditions
        # Get the optimal shift for each edge
        edge_shifts = offsets[min_shift_idx[mask]]
        
        # Calculate periodic boundary condition corrected edge vectors
        pos_src = atom_coords[src]
        pos_dst = atom_coords[dst]
        edge_vec = (pos_dst + edge_shifts) - pos_src

    return edge_index, edge_vec


class TensorDataset(Dataset):
    def __init__(self, path: str, property_name: str, l_max: int, cutoff: float):
        self.property_name = property_name
        self.l_max = l_max
        self.cutoff = cutoff
        with open(path, "rb") as f:
            self.data = pkl.load(f)

        if len(self.data) == 0:
            raise ValueError("Dataset is empty")
        if self.property_name not in self.data[0]:
            raise ValueError(f"Data must contain '{self.property_name}' key")

        self.neighbor_list_path = os.path.join(
            os.path.dirname(path),
            "neighbor_list",
        )
        self.neighbor_list_filename = os.path.join(
            self.neighbor_list_path,
            f"{self.property_name}_neighbor_list_cutoff_{self.cutoff:.2f}.pt",
        )
        os.makedirs(self.neighbor_list_path, exist_ok=True)
        if not os.path.exists(self.neighbor_list_filename):
            # Cache both edge_index and edge_vec with PBC correction
            print(f"Calculating neighbor list for {self.property_name} dataset...")
            self.cached_neighbor_data = []
            for i in tqdm(range(len(self.data))):
                structure = self.data[i]["structure"]
                atom_coords = torch.tensor(structure.cart_coords, dtype=torch.float32)
                lattice_mat = structure.lattice.matrix
                
                # Calculate edge_index and edge_vec considering PBC
                edge_index, edge_vec = get_neighbor_list(atom_coords, self.cutoff, lattice_mat)
                
                self.cached_neighbor_data.append((edge_index, edge_vec))
            torch.save(self.cached_neighbor_data, self.neighbor_list_filename)
            print(f"Saved neighbor list to {self.neighbor_list_filename}")
        else:
            try:
                self.cached_neighbor_data = torch.load(self.neighbor_list_filename)
                print(f"Loaded neighbor list from {self.neighbor_list_filename}")
            except Exception as e:
                print(f"Error loading cache file: {e}")
                print(f"Deleting corrupted cache and recalculating...")
                os.remove(self.neighbor_list_filename)
                print(f"Calculating neighbor list for {self.property_name} dataset...")
                self.cached_neighbor_data = []
                for i in tqdm(range(len(self.data))):
                    structure = self.data[i]["structure"]
                    atom_coords = torch.tensor(structure.cart_coords, dtype=torch.float32)
                    lattice_mat = structure.lattice.matrix
                    
                    # Calculate edge_index and edge_vec considering PBC
                    edge_index, edge_vec = get_neighbor_list(atom_coords, self.cutoff, lattice_mat)
                    
                    self.cached_neighbor_data.append((edge_index, edge_vec))
                torch.save(self.cached_neighbor_data, self.neighbor_list_filename)
                print(f"Saved neighbor list to {self.neighbor_list_filename}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Data:
        assert "structure" in self.data[idx], "Data must contain 'structure' key"
        assert (
            self.data[idx]["structure"] is not None
        ), "Data must contain 'structure' key"
        assert (
            self.property_name in self.data[idx]
        ), f"Data must contain '{self.property_name}' key"
        assert (
            self.data[idx][self.property_name] is not None
        ), f"Data must contain '{self.property_name}' key"

        structure = self.data[idx]["structure"]
        # atom_type: (num_nodes,)
        atom_type = torch.tensor(structure.atomic_numbers, dtype=torch.long)
        num_nodes = torch.tensor(atom_type.shape[0], dtype=torch.long)
        # atom_coords: (num_nodes,3)
        atom_coords = torch.tensor(structure.cart_coords, dtype=torch.float32)
        
        # Get cached neighbor data (edge_index and edge_vec with PBC correction)
        edge_index, edge_vec = self.cached_neighbor_data[idx]
        
        # tensor_property
        # if "piezoelectric" in self.property_name:
        #     tensor_property = torch.tensor(
        #         np.array(self.data[idx][self.property_name]), dtype=torch.float32
        #     ).unsqueeze(0) * 10**3
        # # convert unit from C/m^2 to pC/N, otherwise the property is numerically tiny 
        # # and cannot be trained effectively
        # else:
        if True:
            tensor_property = torch.tensor(
                np.array(self.data[idx][self.property_name]), dtype=torch.float32
            ).unsqueeze(0)  # Add batch dimension
            
        # lattice_mat: (3, 3)
        lattice_mat = torch.tensor(structure.lattice.matrix, dtype=torch.float32).unsqueeze(0)
        
        return Data(
            atom_type=atom_type,
            atom_coords=atom_coords,
            edge_index=edge_index,
            edge_vec=edge_vec,
            lattice_mat=lattice_mat,
            tensor_property=tensor_property,
            num_nodes=num_nodes,
        )
