import pickle as pkl
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
import os
from tqdm import tqdm
from e3nn import o3
from e3nn.io import CartesianTensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from collections import defaultdict


SYMMETRY_CONFIGS = {
    "dielectric": {
        "irreps": "1x0e + 1x0o + 1x1e + 1x1o + 1x2e + 1x2o + 1x3e + 1x3o",
        "tensor_formula": "ij",
        "mask_size": 32,
        "large_from": 8,
        "mask_indices": [0, 2, 3, 4, 8, 9, 10, 11, 12],
    },
    "piezoelectric": {
        "irreps": "2x0e + 2x0o + 2x1e + 2x1o + 2x2e + 2x2o + 2x3e + 2x3o",
        "tensor_formula": "ijk=ikj",
        "mask_size": 64,
        "large_from": 16,
        "mask_indices": [10, 11, 12, 13, 14, 15, 26, 27, 28, 29, 30, 50, 51, 52, 53, 54, 55, 56],
    },
    "elastic": {
        "irreps": "2x0e + 2x0o + 2x1e + 2x1o + 2x2e + 2x2o + 2x3e + 2x3o + 1x4e",
        "tensor_formula": "ijkl=ijlk=jikl=klij",
        "mask_size": 73,
        "large_from": 16,
        "mask_indices": [0, 1, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 64, 65, 66, 67, 68, 69, 70, 71, 72],
    },
}


def _target_name(property_name: str) -> str:
    if "dielectric" in property_name:
        return "dielectric"
    if "piezoelectric" in property_name:
        return "piezoelectric"
    if "elastic" in property_name:
        return "elastic"
    raise NotImplementedError(f"property_name {property_name} not supported")


def _rm_duplicates(vectors: np.ndarray) -> np.ndarray:
    seen = []
    for vector in vectors.reshape(-1, 9):
        if not any(np.allclose(vector, item, atol=1e-5) for item in seen):
            seen.append(vector.copy())
    return np.array(seen).reshape(-1, 3, 3)


def _find_almost_equal_entries(matrix: torch.Tensor, include_opposites: bool) -> torch.Tensor:
    matrix = matrix.reshape(-1)
    same = torch.abs(matrix.unsqueeze(0) - matrix.unsqueeze(1)) < (
        0.0001 * torch.abs(matrix.unsqueeze(0) + matrix.unsqueeze(1)) / 2
    )
    if not include_opposites:
        return same
    opposite = torch.abs(matrix.unsqueeze(0) + matrix.unsqueeze(1)) < (
        0.0001 * torch.abs(matrix.unsqueeze(0) - matrix.unsqueeze(1)) / 2
    )
    return torch.stack([same, opposite])


def _contract_piezo_tensor(tensor: np.ndarray) -> np.ndarray:
    contracted = np.zeros((3, 6))
    mapping = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
    for i in range(3):
        for j, (k, l) in enumerate(mapping):
            contracted[i, j] = tensor[i, k, l]
    return contracted


def _contract_elastic_tensor(tensor: np.ndarray) -> np.ndarray:
    contracted = np.zeros((6, 6))
    mapping = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
    for i, (m, n) in enumerate(mapping):
        for j, (k, l) in enumerate(mapping):
            contracted[i, j] = tensor[m, n, k, l]
    return contracted


def _compute_symmetry_metadata(structure, property_name: str):
    target = _target_name(property_name)
    config = SYMMETRY_CONFIGS[target]
    analyzer = SpacegroupAnalyzer(structure, symprec=1e-5)
    rotations = analyzer.get_symmetry_dataset()["rotations"]
    rotations = _rm_duplicates(np.array(rotations))
    lattice = structure.lattice.matrix.T
    lattice_inv = np.linalg.inv(lattice)
    cart_rotations = np.matmul(lattice, np.matmul(rotations, lattice_inv))

    irreps_output = o3.Irreps(config["irreps"])
    converter = CartesianTensor(config["tensor_formula"])
    d_matrices = irreps_output.D_from_matrix(torch.tensor(cart_rotations, dtype=torch.float32))
    feature_mask = d_matrices.sum(dim=0)

    marker = torch.arange(config["mask_size"], dtype=torch.float32) + 10.0
    marker[config["large_from"]:] *= 100
    mask_total = torch.matmul(feature_mask, marker)[config["mask_indices"]]
    ideal_matrix = converter.to_cartesian(mask_total)
    if target == "piezoelectric":
        ideal_matrix = torch.tensor(_contract_piezo_tensor(ideal_matrix.detach().cpu().numpy()), dtype=torch.float32)
    elif target == "elastic":
        ideal_matrix = torch.tensor(_contract_elastic_tensor(ideal_matrix.detach().cpu().numpy()), dtype=torch.float32)

    feature_mask = feature_mask / d_matrices.shape[0]
    feature_mask *= (feature_mask > 1e-5).float()
    equality = _find_almost_equal_entries(ideal_matrix, include_opposites=target != "dielectric")
    return feature_mask.float(), equality.bool()


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


def _canonize_edge(src_id, dst_id, src_image, dst_image):
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))
    return src_id, dst_id, src_image, dst_image


def get_gmtnet_neighbor_list(structure, cutoff: float, max_neighbors: int = 12):
    neighbor_lists = structure.get_all_neighbors(r=cutoff)
    min_neighbors = min(len(neighbors) for neighbors in neighbor_lists)
    if min_neighbors < max_neighbors:
        lattice = structure.lattice
        next_cutoff = max(lattice.a, lattice.b, lattice.c) if cutoff < max(lattice.a, lattice.b, lattice.c) else 2 * cutoff
        return get_gmtnet_neighbor_list(structure, next_cutoff, max_neighbors)

    edges = defaultdict(set)
    for site_idx, neighbor_list in enumerate(neighbor_lists):
        neighbor_list = sorted(neighbor_list, key=lambda item: item.nn_distance)
        distances = np.array([neighbor.nn_distance for neighbor in neighbor_list])
        ids = np.array([neighbor.index for neighbor in neighbor_list])
        images = np.array([neighbor.image for neighbor in neighbor_list])
        max_dist = distances[max_neighbors - 1]
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        for dst, image in zip(ids, images):
            dst = int(dst)
            src_id, dst_id, _, dst_image = _canonize_edge(site_idx, dst, (0, 0, 0), tuple(image))
            edges[(src_id, dst_id)].add(dst_image)

    src_nodes = []
    dst_nodes = []
    edge_vecs = []
    for (src_id, dst_id), images in edges.items():
        for dst_image in images:
            dst_coord = structure.frac_coords[dst_id] + dst_image
            edge_vec = structure.lattice.get_cartesian_coords(dst_coord - structure.frac_coords[src_id])
            src_nodes.extend([src_id, dst_id])
            dst_nodes.extend([dst_id, src_id])
            edge_vecs.extend([edge_vec, -edge_vec])

    if len(src_nodes) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_vec = torch.empty((0, 3), dtype=torch.float32)
    else:
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_vec = torch.tensor(np.array(edge_vecs), dtype=torch.float32)
    return edge_index, edge_vec


class TensorDataset(Dataset):
    def __init__(self, path: str, property_name: str, l_max: int, cutoff: float, graph_mode: str = "high_order", max_neighbors: int = 12):
        self.property_name = property_name
        self.l_max = l_max
        self.cutoff = cutoff
        self.graph_mode = graph_mode
        self.max_neighbors = max_neighbors
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
        cache_name = f"{self.property_name}_neighbor_list_{self.graph_mode}_cutoff_{self.cutoff:.2f}"
        if self.graph_mode == "gmtnet":
            cache_name += f"_max_neighbors_{self.max_neighbors}"
        self.neighbor_list_filename = os.path.join(
            self.neighbor_list_path,
            f"{cache_name}.pt",
        )
        self.symmetry_metadata_filename = os.path.join(
            self.neighbor_list_path,
            f"{self.property_name}_symmetry_metadata.pt",
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
                
                if self.graph_mode == "gmtnet":
                    edge_index, edge_vec = get_gmtnet_neighbor_list(structure, self.cutoff, self.max_neighbors)
                else:
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
                    
                    if self.graph_mode == "gmtnet":
                        edge_index, edge_vec = get_gmtnet_neighbor_list(
                            structure,
                            self.cutoff,
                            self.max_neighbors,
                        )
                    else:
                        edge_index, edge_vec = get_neighbor_list(atom_coords, self.cutoff, lattice_mat)
                    
                    self.cached_neighbor_data.append((edge_index, edge_vec))
                torch.save(self.cached_neighbor_data, self.neighbor_list_filename)
                print(f"Saved neighbor list to {self.neighbor_list_filename}")

        self.cached_symmetry_data = self._load_or_create_symmetry_metadata()

    def _load_or_create_symmetry_metadata(self):
        has_metadata = all(
            "feature_mask" in item and item["feature_mask"] is not None
            and "matrix_equal" in item and item["matrix_equal"] is not None
            for item in self.data
        )
        if has_metadata:
            return None
        if os.path.exists(self.symmetry_metadata_filename):
            try:
                symmetry_data = torch.load(self.symmetry_metadata_filename)
                if len(symmetry_data) == len(self.data):
                    print(f"Loaded symmetry metadata from {self.symmetry_metadata_filename}")
                    return symmetry_data
                print("Symmetry metadata cache length mismatch. Recalculating...")
            except Exception as e:
                print(f"Error loading symmetry metadata cache: {e}. Recalculating...")

        print(f"Calculating symmetry metadata for {self.property_name} dataset...")
        symmetry_data = []
        for item in tqdm(self.data):
            symmetry_data.append(_compute_symmetry_metadata(item["structure"], self.property_name))
        torch.save(symmetry_data, self.symmetry_metadata_filename)
        print(f"Saved symmetry metadata to {self.symmetry_metadata_filename}")
        return symmetry_data

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
        
        data_kwargs = {
            "atom_type": atom_type,
            "atom_coords": atom_coords,
            "edge_index": edge_index,
            "edge_vec": edge_vec,
            "lattice_mat": lattice_mat,
            "tensor_property": tensor_property,
            "num_nodes": num_nodes,
        }
        if "feature_mask" in self.data[idx] and self.data[idx]["feature_mask"] is not None:
            data_kwargs["feat_mask"] = torch.tensor(
                np.array(self.data[idx]["feature_mask"]), dtype=torch.float32
            ).unsqueeze(0)
        elif self.cached_symmetry_data is not None:
            data_kwargs["feat_mask"] = self.cached_symmetry_data[idx][0].unsqueeze(0)
        if "matrix_equal" in self.data[idx] and self.data[idx]["matrix_equal"] is not None:
            data_kwargs["equality"] = torch.tensor(
                np.array(self.data[idx]["matrix_equal"]), dtype=torch.bool
            ).unsqueeze(0)
        elif self.cached_symmetry_data is not None:
            data_kwargs["equality"] = self.cached_symmetry_data[idx][1].unsqueeze(0)

        return Data(**data_kwargs)
