import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.o3 import Irreps
from typing import List, Union
from torch_scatter import scatter

from .utils import add_irreps_tensor
from .middle_mlp import MiddleMLP

# Deprecated
# class Model(nn.Module):
# def __init__(
#     self,
#     l_max: int,
#     symmetry: str,
#     max_atom_type: int,
#     cutoff: float,
#     emb_func: str,
#     num_invariant_layers: int,
#     invariant_layer: Union[str, List[str]],
#     equivariant_layer: Union[str, List[str]],
#     scalar_dim: int,
#     irreps_list: Union[
#         List[str], List[Irreps]
#     ],  # the list of irreps with lifting l in equi layers
#     tp_method: str = "fully_connected",
#     # pool_method: str = "final",
#     hidden_irreps_list: Union[List[str], None] = None,  # if using equiformer update
#     embed_atom: nn.Module = None,
#     embed_dist: nn.Module = None,
#     invariant_layers: nn.ModuleList = None,
#     equivariant_layers: nn.ModuleList = None,
#     readout_layer: nn.Module = None,
# ):
#     super(Model, self).__init__()
#     # Embed atom and distance
#     if embed_atom is None:
#         self.embed_atom = EmbedAtom(
#             embed_dim=scalar_dim, max_atom_type=max_atom_type
#         )
#     else:
#         self.embed_atom = embed_atom
#     if embed_dist is None:
#         self.embed_dist = EmbedDist(
#             embed_func=emb_func, embed_dim=scalar_dim, cutoff=cutoff
#         )
#     else:
#         self.embed_dist = embed_dist
#     if hidden_irreps_list is None:
#         hidden_irreps_list = [None for _ in irreps_list]
#     if irreps_list[0] != "0e":
#         Warning("The first irreps in irreps_list should be 0e, we add it for you.")
#         irreps_list = [f"{scalar_dim}x0e"] + irreps_list
#     if invariant_layers is None:
#         if isinstance(invariant_layer, str):
#             # Invariant layer(s)
#             self.invariant_layers = nn.ModuleList(
#                 [
#                     InvariantLayer(
#                         update_method=invariant_layer, scalar_dim=scalar_dim
#                     )
#                     for _ in range(num_invariant_layers)
#                 ]
#             )
#         else:
#             # Invariant layer(s)
#             self.invariant_layers = nn.ModuleList(
#                 [
#                     InvariantLayer(update_method=layer, scalar_dim=scalar_dim)
#                     for layer in invariant_layer
#                 ]
#             )
#     else:
#         self.invariant_layers = invariant_layers
#     # Equivariant layers
#     self.irreps_list = [Irreps(ir) for ir in irreps_list]
#     self.equivariant_layers = nn.ModuleList()
#     if equivariant_layers is None:
#         if isinstance(equivariant_layer, str):
#             for i in range(len(irreps_list) - 1):
#                 self.equivariant_layers.append(
#                     EquivariantLayer(
#                         update_method=equivariant_layer,
#                         irreps_in=irreps_list[i],
#                         irreps_out=irreps_list[i + 1],
#                         irreps_hidden=hidden_irreps_list[i],
#                     )
#                 )
#         else:
#             for i in range(len(irreps_list) - 1):
#                 self.equivariant_layers.append(
#                     EquivariantLayer(
#                         update_method=equivariant_layer[i],
#                         irreps_in=irreps_list[i],
#                         irreps_out=irreps_list[i + 1],
#                         irreps_hidden=hidden_irreps_list[i],
#                     )
#                 )
#     else:
#         self.equivariant_layers = equivariant_layers
#     # Readout layer
#     if readout_layer is None:
#         self.readout_layer = ReadoutLayer(l_max=l_max, symmetry=symmetry)
#     else:
#         self.readout_layer = readout_layer

# def forward(
#     self,
#     property_name: str,
#     atom_type: torch.Tensor,
#     edge_vec: torch.Tensor,
#     edge_index: torch.Tensor,
#     batch_index: torch.Tensor,
# ) -> torch.Tensor:
#     # atom_feature: (num_nodes, scalar_dim)
#     atom_feature = self.embed_atom(atom_type)
#     # edge_feature: (num_edges, scalar_dim)
#     dist = torch.norm(edge_vec, dim=1)
#     edge_feature = self.embed_dist(dist)
#     # edge_index: (2, num_edges)
#     src, dst = edge_index
#     # atom_feature: (num_nodes, scalar_dim)
#     for invariant_layer in self.invariant_layers:
#         atom_feature, edge_feature = invariant_layer(
#             src_feature=atom_feature[src],
#             dst_feature=atom_feature[dst],
#             edge_feature=edge_feature,
#         )
#     # atom_feature: (num_nodes, irreps_out.dim)
#     for equivariant_layer in self.equivariant_layers:
#         atom_feature = equivariant_layer(
#             atom_feature=atom_feature,
#             edge_vector=edge_vec,
#             edge_index=edge_index,
#         )
#     # global_feature: (num_graphs, irreps_out.dim)
#     # Global feature cannot involve per-layer feature
#     # because irreps is changing in equivariant layer
#     global_feature = scatter(
#         atom_feature,
#         batch_index,
#         dim=0,
#         reduce="mean",
#     )
#     # property_out: (num_graphs, 3, 3, ...)
#     # or voigt (num_graphs, 3, 6) / (num_graphs, 6, 6)
#     property_out = self.readout_layer(
#         global_feature, self.irreps_list[-1], property_name=property_name
#     )
#     return property_out


class Model(nn.Module):
    def __init__(
        self,
        embedding_layer: nn.Module = None,
        invariant_layers: nn.ModuleList = None,
        middle_mlp: nn.Module = None,
        equivariant_layers: nn.ModuleList = None,
        final_mlp: nn.Module = None,
        readout_layer: nn.Module = None,
        self_train: bool = False,
        final_pooling: bool = True,
        irreps_list: Union[
            List[str], List[Irreps], None
        ] = None,  # Note that the input irreps list contains only the irreps in equivariant layers
    ):
        super(Model, self).__init__()
        assert embedding_layer is not None, "embedding_layer should be provided"
        assert invariant_layers is not None, "invariant_layers should be provided"
        assert middle_mlp is not None, "middle_mlp should be provided"
        assert equivariant_layers is not None, "equivariant_layers should be provided"
        assert final_mlp is not None, "final_mlp should be provided"
        if not self_train:
            assert readout_layer is not None, "readout_layer should be provided"
        if not final_pooling:
            assert irreps_list is not None, "irreps_list should be provided"
            self.irreps_list = [Irreps(ir) for ir in irreps_list]
            assert self.irreps_list[0].lmax == 0, "lmax of first irreps should be 0"
        else:
            self.irreps_list = []

        self.embedding_layer = embedding_layer
        self.invariant_layers = invariant_layers
        self.atom_middle_mlp = middle_mlp
        self.edge_middle_mlp = middle_mlp
        self.equivariant_layers = equivariant_layers
        self.final_mlp = final_mlp
        self.readout_layer = readout_layer
        self.self_train = self_train
        self.final_pooling = final_pooling

    def forward(self, atom_type, edge_vec, edge_index, batch_index):
        # atom_feature: (num_nodes, scalar_dim)
        # edge_feature: (num_edges, scalar_dim)
        edge_dist = torch.norm(edge_vec, dim=1)
        atom_feature, edge_feature = self.embedding_layer(atom_type, edge_dist)

        for invariant_layer in self.invariant_layers:
            atom_feature, edge_feature = invariant_layer(
                atom_feature, edge_feature, edge_index
            )

        atom_feature = self.atom_middle_mlp(atom_feature)
        edge_feature = self.edge_middle_mlp(edge_feature)

        atom_feature_list = [atom_feature]
        edge_feature_list = [edge_feature]

        for equivariant_layer in self.equivariant_layers:
            atom_feature, edge_feature = equivariant_layer(
                atom_feature, edge_vec, edge_index, edge_feature
            )
            atom_feature_list.append(atom_feature)
            edge_feature_list.append(edge_feature)

        atom_feature = self.final_mlp(atom_feature)
        edge_feature = self.final_mlp(edge_feature)
        edge_feature_list.append(edge_feature)
        atom_feature_list.append(atom_feature)

        if not self.self_train:
            if not self.final_pooling:
                self.irreps_list.append(self.final_mlp.irreps_out)
                atom_feature = add_irreps_tensor(
                    self.irreps_list, atom_feature_list
                ) / len(atom_feature_list)
                global_feature = scatter(atom_feature, batch_index, dim=0, reduce="mean")
            # global_feature: (num_graphs, irreps_out.dim)
            else:
                global_feature = scatter(atom_feature, batch_index, dim=0, reduce="mean")
            property_out = self.readout_layer(global_feature)
            return property_out
        else:
            # force_out: (num_nodes, 3)
            force_out = self.readout_layer(atom_feature)
            return force_out

class InvariantOnlyModel(nn.Module):
    def __init__(
        self, 
        embedding_layer: nn.Module = None, 
        invariant_layers: nn.ModuleList = None, 
        readout_layer: nn.Module = None
        ):
        super().__init__()
        assert embedding_layer and invariant_layers, "embedding_layer, invariant_layers should be provided"
        self.embedding_layer = embedding_layer
        self.invariant_layers = invariant_layers
        scalar_dim = invariant_layers[-1].scalar_dim
        if readout_layer is not None:
            self.readout_layer = readout_layer
        else:
            self.readout_layer = nn.Sequential(
                nn.Linear(scalar_dim, 4 * scalar_dim),
                nn.SiLU(),
                nn.Linear(4 * scalar_dim, 1),
            )

    def forward(self, atom_type, edge_vec, edge_index, batch_index):
        # atom_feature: (num_nodes, scalar_dim)
        # edge_feature: (num_edges, scalar_dim)
        edge_dist = torch.norm(edge_vec, dim=1)
        atom_feature, edge_feature = self.embedding_layer(atom_type, edge_dist)
        for invariant_layer in self.invariant_layers:
            atom_feature, edge_feature = invariant_layer(
                atom_feature, edge_feature, edge_index
            )
        
        property_feature = self.readout_layer(atom_feature)
        
        assert len(property_feature) == len(batch_index), "property_feature and batch_index should have the same length"
        property_out = scatter(property_feature, batch_index, dim=0, reduce="mean")
        
        return property_out
