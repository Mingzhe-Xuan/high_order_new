import pickle as pkl
from typing import Any, Dict, List
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
import os
from tqdm import tqdm

from .tensor_dataset import get_gmtnet_neighbor_list, get_neighbor_list


class ScalarDataset(Dataset):
    def __init__(
        self,
        path: str,
        property_name: str,
        cutoff: float,
        graph_mode: str = "high_order",
        max_neighbors: int = 12,
    ):
        self.property_name = property_name
        self.cutoff = cutoff
        self.graph_mode = graph_mode
        self.max_neighbors = max_neighbors
        with open(path, "rb") as f:
            self.data: List[Dict[str, Any]] = pkl.load(f)

        if len(self.data) == 0:
            raise ValueError("Dataset is empty")
        if "structure" not in self.data[0]:
            raise ValueError("Data must contain 'structure' key")
        if self.property_name not in self.data[0]:
            raise ValueError(f"Data must contain '{self.property_name}' key")

        self.neighbor_list_path = os.path.join(
            os.path.dirname(path),
            "neighbor_list",
        )
        cache_name = f"{self.property_name}_neighbor_list_{self.graph_mode}_cutoff_{self.cutoff:.2f}"
        if self.graph_mode == "gmtnet":
            cache_name += f"_max_neighbors_{self.max_neighbors}"
        self.neighbor_list_filename = os.path.join(
            self.neighbor_list_path,
            f"{cache_name}.pt",
        )
        os.makedirs(self.neighbor_list_path, exist_ok=True)
        
        if not os.path.exists(self.neighbor_list_filename):
            print(f"Calculating neighbor list for {self.property_name} dataset...")
            self.cached_neighbor_data = []
            for i in tqdm(range(len(self.data))):
                structure = self.data[i]["structure"]
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
            
            # Save with error handling
            try:
                torch.save(self.cached_neighbor_data, self.neighbor_list_filename)
                print(f"Successfully saved cache to {self.neighbor_list_filename}")
            except Exception as e:
                print(f"Error saving cache: {e}")
                # Remove corrupted cache file
                if os.path.exists(self.neighbor_list_filename):
                    os.remove(self.neighbor_list_filename)
                raise
        else:
            # Load with error handling
            try:
                self.cached_neighbor_data = torch.load(self.neighbor_list_filename)
                print(f"Successfully loaded cache from {self.neighbor_list_filename}")
                
                # Validate loaded data
                if len(self.cached_neighbor_data) != len(self.data):
                    print(f"Warning: Cache length mismatch. Expected {len(self.data)}, got {len(self.cached_neighbor_data)}")
                    print("Regenerating cache...")
                    self._regenerate_cache()
                else:
                    # Validate first entry
                    if len(self.cached_neighbor_data) > 0:
                        edge_index, edge_vec = self.cached_neighbor_data[0]
                        if not isinstance(edge_index, torch.Tensor) or not isinstance(edge_vec, torch.Tensor):
                            print("Warning: Invalid cache format. Regenerating...")
                            self._regenerate_cache()
            except Exception as e:
                print(f"Error loading cache: {e}")
                print("Regenerating cache...")
                self._regenerate_cache()

    def _regenerate_cache(self):
        """Regenerate the cache file"""
        print(f"Regenerating cache for {self.property_name} dataset...")
        self.cached_neighbor_data = []
        for i in tqdm(range(len(self.data))):
            structure = self.data[i]["structure"]
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
        
        # Save the regenerated cache
        torch.save(self.cached_neighbor_data, self.neighbor_list_filename)
        print(f"Successfully regenerated cache to {self.neighbor_list_filename}")

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
        num_nodes = int(atom_type.shape[0])
        # atom_coords: (num_nodes,3)
        atom_coords = torch.tensor(structure.cart_coords, dtype=torch.float32)
        # lattice_mat: (3, 3)
        lattice_mat = structure.lattice.matrix
        
        # Get cached neighbor data
        edge_index, edge_vec = self.cached_neighbor_data[idx]
        
        # scalar_property: float
        scalar_property = torch.tensor(
            self.data[idx][self.property_name], dtype=torch.float32
        )
        
        # lattice_mat: (3, 3)
        lattice_mat_tensor = torch.tensor(lattice_mat, dtype=torch.float32).unsqueeze(0)

        return Data(
            atom_type=atom_type,
            edge_index=edge_index,
            edge_vec=edge_vec,
            scalar_property=scalar_property,
            atom_coords=atom_coords,
            lattice_mat=lattice_mat_tensor,
            num_nodes=num_nodes,
        )
