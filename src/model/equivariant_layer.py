import torch
import torch.nn as nn
from torch_scatter import scatter
import torch.nn.functional as F
from typing import Union, Optional

from e3nn.o3 import Irreps, Linear, spherical_harmonics
from e3nn.nn import Gate

try:
    from .tensor_product import get_tp
    from .utils import add_irreps_tensor
    from .utils.add_irreps_tensor import selective_residual_add
    from .final_mlp import FinalMLP
    from .layer_norm import SeperableLayerNorm
except ImportError:
    from tensor_product import get_tp
    from utils import add_irreps_tensor
    from utils.add_irreps_tensor import selective_residual_add
    from final_mlp import FinalMLP
    from layer_norm import SeperableLayerNorm


class BaseEquivariantLayer(nn.Module):
    """Base class containing shared parameters and common functionality."""

    def __init__(
        self,
        irreps_in: Union[str, Irreps],
        irreps_out: Union[str, Irreps],
        irreps_vec: Optional[Union[str, Irreps]] = None,
        irreps_hidden: Optional[Union[str, Irreps, None]] = None,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.irreps_vec = Irreps(irreps_vec) if irreps_vec is not None else self.irreps_in
        default_irreps_hidden = self.irreps_in + self.irreps_out
        self.irreps_hidden = Irreps(irreps_hidden) if irreps_hidden is not None else default_irreps_hidden

    def compute_spherical_harmonics(self, edge_vector: torch.Tensor, irreps_vec: Irreps = None):
        """Compute spherical harmonics for edge vectors."""
        if irreps_vec is not None:
            return spherical_harmonics(irreps_vec, edge_vector, normalize=True)
        return spherical_harmonics(self.irreps_vec, edge_vector, normalize=True)

    def aggregate_messages(
        self,
        messages: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Aggregate messages to nodes."""
        new_atom_feature = torch.zeros(
            num_nodes, self.irreps_out.dim, device=device, dtype=dtype
        )
        new_atom_feature = new_atom_feature + scatter(
            messages, dst, dim=0, dim_size=num_nodes, reduce="sum"
        )
        return new_atom_feature


class EquiformerLayer(BaseEquivariantLayer):
    """Implementation of the equiformer update mechanism."""

    def __init__(
        self,
        irreps_in: Union[str, Irreps],
        irreps_out: Union[str, Irreps],
        irreps_vec: Optional[Union[str, Irreps]] = None,
        irreps_hidden: Optional[Union[str, Irreps, None]] = None,
        tp_method: str = "fully_connected",
        residual: bool = True,
    ):
        super().__init__(irreps_in, irreps_out, irreps_vec, irreps_hidden, residual)
        
        self.tp_method = tp_method
        
        scalar_irreps = [(mul, (l, p)) for mul, (l, p) in self.irreps_hidden if l == 0]
        self.irreps_scalar = Irreps(scalar_irreps)
        non_scalar_irreps = [
            (mul, (l, p)) for mul, (l, p) in self.irreps_hidden if l > 0
        ]
        self.irreps_non_scalar = Irreps(non_scalar_irreps)

        self.lin_src = Linear(self.irreps_in, self.irreps_in)
        self.lin_dst = Linear(self.irreps_in, self.irreps_in)

        if len(self.irreps_scalar) == 0:
            raise ValueError(
                "equiformer requires irreps_hidden to contain at least one scalar (l=0) irreps"
            )
        if len(self.irreps_non_scalar) == 0:
            raise ValueError(
                "equiformer requires irreps_hidden to contain at least one non-scalar (l>0) irreps"
            )

        num_scalar_irreps = self.irreps_scalar.num_irreps
        num_non_scalar_irreps = self.irreps_non_scalar.num_irreps

        if num_scalar_irreps != num_non_scalar_irreps:
            if num_scalar_irreps > num_non_scalar_irreps:
                new_scalar_irreps = []
                remaining = num_non_scalar_irreps
                for mul, (l, p) in self.irreps_scalar:
                    if remaining <= 0:
                        break
                    take = min(mul, remaining)
                    new_scalar_irreps.append((take, (l, p)))
                    remaining -= take
                self.irreps_scalar = Irreps(new_scalar_irreps)
            else:
                new_scalar_irreps = []
                remaining = num_non_scalar_irreps
                for mul, (l, p) in self.irreps_scalar:
                    while remaining > 0:
                        new_scalar_irreps.append((mul, (l, p)))
                        remaining -= mul
                if remaining < 0:
                    last_mul, last_irrep = new_scalar_irreps[-1]
                    new_scalar_irreps[-1] = (last_mul + remaining, last_irrep)
                self.irreps_scalar = Irreps(new_scalar_irreps)

            self.irreps_hidden = self.irreps_scalar + self.irreps_non_scalar

        self.i_scalar = self.irreps_scalar.dim
        self.tp1 = get_tp(
            tp_method, self.irreps_in, self.irreps_vec, self.irreps_hidden
        )
        self.tp2 = get_tp(
            tp_method, self.irreps_hidden, self.irreps_vec, self.irreps_out
        )
        self.lin_scalar = Linear(self.irreps_scalar, self.irreps_out)
        self.gate = Gate(
            irreps_scalars="",
            act_scalars=[],
            irreps_gates=self.irreps_scalar,
            act_gates=[F.sigmoid],
            irreps_gated=self.irreps_non_scalar,
        )
        self.lin_hidden = Linear(self.irreps_hidden, self.irreps_hidden)

    def equiformer_update(
        self,
        atom_feature: torch.Tensor,
        edge_vector: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        # edge_vector: (num_edges, 3)
        # edge_sh: (num_edges, irreps_vec.dim)
        edge_sh = self.compute_spherical_harmonics(edge_vector)
        # edge_index: (2, num_edges)
        # src, dst: (num_edges, )
        src, dst = edge_index
        # src_feature: (num_edges, irreps_in.dim)
        src_feature = atom_feature[src]
        # dst_feature: (num_edges, irreps_in.dim)
        dst_feature = atom_feature[dst]

        # src_feature: (num_edges, irreps_in.dim)
        src_feature = self.lin_src(src_feature)
        # dst_feature: (num_edges, irreps_in.dim)
        dst_feature = self.lin_dst(dst_feature)
        # hidden_feature: (num_edges, irreps_hidden.dim)
        # Here tp will fail if irreps_in, irreps_vec and irreps_hidden are not matching,
        # e.g. they do not satisfy the triangular inequality.
        if self.tp_method == "so2":
            hidden_feature = self.tp1(src_feature, edge_vector.clone())
        else:
            hidden_feature = self.tp1(src_feature, edge_sh)
        hidden_feature = self.lin_hidden(hidden_feature)

        hidden_scalar_activated = F.softmax(
            self.lin_scalar(F.leaky_relu(hidden_feature[..., : self.i_scalar])), dim=-1
        )
        hidden_non_scalar_gated = self.gate(hidden_feature)

        if self.tp_method == "so2":
            hidden_non_scalar_after_tp = self.tp2(hidden_feature, edge_vector.clone())
        else:
            hidden_non_scalar_after_tp = self.tp2(hidden_feature, edge_sh)

        # hidden_out: (num_edges, irreps_out.dim)
        hidden_out = hidden_scalar_activated * hidden_non_scalar_after_tp
        # atom_feature: (num_nodes, irreps_out.dim)
        num_nodes = atom_feature.size(0)
        aggregated_message = self.aggregate_messages(
                hidden_out, dst, num_nodes, atom_feature.device, atom_feature.dtype
            )
        if self.residual:
            # atom_feature = add_irreps_tensor(
            #     [self.irreps_in, self.irreps_out], [atom_feature, aggregated_message]
            # )
            atom_feature = selective_residual_add(
                self.irreps_in, self.irreps_out, atom_feature, aggregated_message
            )
        else:
            atom_feature = aggregated_message
        return atom_feature, None

    def forward(
        self,
        atom_feature: torch.Tensor,
        edge_vector: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        return self.equiformer_update(atom_feature, edge_vector, edge_index)


class TpconvLayer(BaseEquivariantLayer):
    """Implementation of the tpconv update mechanism."""

    def __init__(
        self,
        irreps_in: Union[str, Irreps],
        irreps_out: Union[str, Irreps],
        irreps_vec: Optional[Union[str, Irreps]] = None,
        irreps_hidden: Optional[Union[str, Irreps, None]] = None,
        tp_method: str = "fully_connected",
        residual: bool = True,
    ):
        super().__init__(irreps_in, irreps_out, irreps_vec, irreps_hidden, residual)
        
        self.tp_method = tp_method
        self.tp = get_tp(tp_method, self.irreps_in, self.irreps_vec, self.irreps_out)

    def tpconv_update(
        self,
        atom_feature: torch.Tensor,
        edge_vector: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        # edge_vector: (num_edges, 3)
        # edge_dist: (num_edges, )
        edge_dist = torch.norm(edge_vector, dim=-1)
        # edge_sh: (num_edges, irreps_vec.dim)
        edge_sh = self.compute_spherical_harmonics(edge_vector)
        # edge_index: (2, num_edges)
        # src, dst: (num_edges, )
        src, dst = edge_index
        # src_feature: (num_edges, irreps_in.dim)
        src_feature = atom_feature[src]
        # dst_feature: (num_edges, irreps_in.dim)
        dst_feature = atom_feature[dst]
        # message: (num_edges, irreps_out.dim)
        if self.tp_method == "so2":
            edge_vector_so2 = edge_vector.detach().requires_grad_(
                edge_vector.requires_grad
            )
            message = self.tp(src_feature, edge_vector_so2)
        else:
            message = self.tp(src_feature, edge_sh)
        # atom_feature: (num_nodes, irreps_out.dim)
        num_nodes = atom_feature.size(0)
        aggregated_message = self.aggregate_messages(
            message, dst, num_nodes, atom_feature.device, atom_feature.dtype
        )
        if self.residual:
            # atom_feature = add_irreps_tensor(
            #     [self.irreps_in, self.irreps_out], [atom_feature, aggregated_message]
            # )
            atom_feature = selective_residual_add(
                self.irreps_in, self.irreps_out, atom_feature, aggregated_message
            )
        else:
            atom_feature = aggregated_message
        return atom_feature, torch.zeros_like(atom_feature)

    def forward(
        self,
        atom_feature: torch.Tensor,
        edge_vector: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feature: Optional[torch.Tensor] = None,
    ):
        return self.tpconv_update(atom_feature, edge_vector, edge_index)


class TpconvWithEdgeLayer(BaseEquivariantLayer):
    """Implementation of the tpconv_with_edge update mechanism."""

    def __init__(
        self,
        irreps_in: Union[str, Irreps],
        irreps_out: Union[str, Irreps],
        irreps_vec: Optional[Union[str, Irreps]] = None,
        irreps_hidden: Optional[Union[str, Irreps, None]] = None,
        tp_method: str = "fully_connected",
        residual: bool = True,
    ):
        super().__init__(irreps_in, irreps_out, irreps_vec, irreps_hidden, residual)
        
        self.tp_method = tp_method
        if self.irreps_in.lmax == 0:
            self.tp = get_tp("fully_connected", self.irreps_in, self.irreps_out, self.irreps_out)
        else:
            self.tp = get_tp(tp_method, self.irreps_in, self.irreps_out, self.irreps_out, latent_dim=self.irreps_in.dim)
        if self.irreps_in.lmax == 0:
            weight_dim = self.tp.weight_numel
        elif tp_method == "so2":
            weight_dim = self.irreps_in.dim
        else:
            weight_dim = self.tp.weight_numel
        self.weight_linear = nn.Sequential(
            nn.Linear(self.irreps_in.dim, 4 * self.irreps_in.dim),
            nn.ReLU(),
            nn.Linear(4 * self.irreps_in.dim, weight_dim),
        )
        # self.weight_linear = nn.Sequential(
        #     nn.Linear(self.irreps_in.dim, self.irreps_in.dim),
        #     nn.Softplus(),
        #     nn.Linear(self.irreps_in.dim, weight_dim),
        # )
        # self.weight_linear = nn.Linear(self.irreps_in.dim, weight_dim)
        self.norm = SeperableLayerNorm(self.irreps_out)
        self.final_mlp = FinalMLP(self.irreps_out, self.irreps_out, self.irreps_hidden, num_hidden_layers=1)

    def tpconv_with_edge_update(
        self,
        atom_feature: torch.Tensor,
        edge_vector: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feature: torch.Tensor,
    ) -> torch.Tensor:
        # edge_feature: (num_edges, irreps_in.dim)
        # weight: (num_edges, weight_num)
        weight = self.weight_linear(edge_feature)
        # edge_index: (2, num_edges)
        # src, dst: (num_edges, )
        src, dst = edge_index
        # src_feature: (num_edges, irreps_in.dim)
        src_feature = atom_feature[src]
        # dst_feature: (num_edges, irreps_in.dim)
        dst_feature = atom_feature[dst]
        # message: (num_edges, irreps_out.dim)
        # if self.irreps_in.lmax == 0:
        #     edge_sh = self.compute_spherical_harmonics(edge_vector, self.irreps_out)
        #     message = self.tp(src_feature, edge_sh, weight)
        # else:
        #     message = self.tp(src_feature, edge_feature)
        edge_sh = self.compute_spherical_harmonics(edge_vector, self.irreps_out)
        if self.tp_method == "so2" and self.irreps_in.lmax != 0:
            edge_vector_so2 = edge_vector.clone()
            message = self.tp(src_feature, edge_vector_so2, latents=weight)
        else:
            message = self.tp(src_feature, edge_sh, weight)
        # atom_feature: (num_nodes, irreps_out.dim)
        num_nodes = atom_feature.size(0)
        aggregated_message = self.aggregate_messages(
            message, dst, num_nodes, atom_feature.device, atom_feature.dtype
        )
        # # NORM OR NOT? NORM!
        aggregated_message = self.norm(aggregated_message)
        if self.residual:
            # atom_feature = add_irreps_tensor(
            #     [self.irreps_in, self.irreps_out], [atom_feature, aggregated_message]
            # )
            # edge_feature = add_irreps_tensor(
            #     [self.irreps_in, self.irreps_out], [edge_feature, message]
            # )
            atom_feature = selective_residual_add(
                self.irreps_in, self.irreps_out, atom_feature, aggregated_message
            )
            edge_feature = selective_residual_add(
                self.irreps_in, self.irreps_out, edge_feature, message
            )
        else:
            atom_feature = self.final_mlp(aggregated_message)
            edge_feature = self.final_mlp(message)
        return atom_feature, edge_feature

    def forward(
        self,
        atom_feature: torch.Tensor,
        edge_vector: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feature: torch.Tensor,
    ):
        return self.tpconv_with_edge_update(
            atom_feature, edge_vector, edge_index, edge_feature
        )


class EquivariantLayer(nn.Module):
    """
    Original EquivariantLayer class maintained for backward compatibility.
    Creates the appropriate specialized layer based on the update_method parameter.
    """

    def __init__(
        self,
        update_method: str,
        irreps_in: Union[str, Irreps],
        irreps_out: Union[str, Irreps],
        irreps_vec: Optional[Union[str, Irreps]] = None,
        irreps_hidden: Optional[Union[str, Irreps, None]] = None,
        tp_method: str = "fully_connected",
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.update_method = update_method
        assert update_method == "tpconv_with_edge", "Only tpconv_with_edge is supported for now"
        self.layer = TpconvWithEdgeLayer(
            irreps_in, irreps_out, irreps_vec, irreps_hidden, tp_method, residual
        )

    def forward(
        self,
        atom_feature: torch.Tensor,
        edge_vector: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feature: Optional[torch.Tensor] = None,
    ):
        if isinstance(self.layer, TpconvWithEdgeLayer):
            assert (
                edge_feature is not None
            ), "edge_feature cannot be None if update_method is tpconv_with_edge"
            return self.layer(atom_feature, edge_vector, edge_index, edge_feature)
        else:
            return self.layer(atom_feature, edge_vector, edge_index)
