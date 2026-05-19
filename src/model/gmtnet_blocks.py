import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.io import CartesianTensor
from torch import Tensor, nn
from torch.autograd import grad
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_scatter import scatter


class RBFExpansion(nn.Module):
    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer("centers", torch.linspace(self.vmin, self.vmax, self.bins))
        if lengthscale is None:
            self.lengthscale = torch.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale
        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale**2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (distance.unsqueeze(1) - self.centers) ** 2)


class ComformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels)
        self.lin_concate = nn.Linear(heads * out_channels, out_channels)
        self.lin_msg_update = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.softplus = nn.Softplus()
        self.bn_att = nn.BatchNorm1d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.key_update = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights=None,
    ):
        h, c = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x = (x, x)

        query = self.lin_query(x[1]).view(-1, h, c)
        key = self.lin_key(x[0]).view(-1, h, c)
        value = self.lin_value(x[0]).view(-1, h, c)
        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr,
            size=None,
        )
        out = out.view(-1, self.heads * self.out_channels)
        out = self.lin_concate(out)
        return self.softplus(x[1] + out)

    def message(
        self,
        query_i: Tensor,
        key_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        value_i: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        key_j = self.key_update(torch.cat((key_i, key_j, edge_attr), dim=-1))
        alpha = (query_i * key_j) / math.sqrt(self.out_channels)
        out = self.lin_msg_update(torch.cat((value_i, value_j, edge_attr), dim=-1))
        out = out * self.sigmoid(
            self.bn_att(alpha.view(-1, self.out_channels)).view(
                -1, self.heads, self.out_channels
            )
        )
        return out


class TensorProductConvLayer(nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True):
        super().__init__()
        self.residual = residual
        self.tp = o3.FullyConnectedTensorProduct(
            in_irreps, sh_irreps, out_irreps, shared_weights=False
        )
        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, n_edge_features),
            nn.Softplus(),
            nn.Linear(n_edge_features, self.tp.weight_numel),
        )

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce="mean"):
        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))
        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)
        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded
        return out


class ComformerConvEqui(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        edge_dim: Optional[int] = None,
        ns: int = 16,
        nv: int = 2,
        residual: bool = True,
        target_irreps: str = "1x0e + 1x0o + 1x1e + 1x1o + 1x2e + 1x2o + 1x3e + 1x3o",
    ):
        super().__init__()
        irrep_seq = [
            f"{ns}x0e",
            f"{ns}x0e + {nv}x1o + {nv}x2e",
            f"{ns}x0e + {nv}x1o + {nv}x1e + {nv}x2e + {nv}x2o",
            target_irreps,
        ]
        self.node_linear = nn.Linear(in_channels, ns)
        self.sh = "1x0e + 1x1o + 1x2e"
        self.nlayer_1 = TensorProductConvLayer(
            irrep_seq[0], self.sh, irrep_seq[1], edge_dim, residual=residual
        )
        self.nlayer_2 = TensorProductConvLayer(
            irrep_seq[1], self.sh, irrep_seq[2], edge_dim, residual=False
        )
        self.nlayer_3 = TensorProductConvLayer(
            irrep_seq[2], self.sh, irrep_seq[3], edge_dim, residual=False
        )

    def forward(self, node_feature, edge_vec, edge_index, edge_feature):
        edge_irr = o3.spherical_harmonics(
            self.sh, edge_vec, normalize=True, normalization="component"
        )
        node_feature = self.node_linear(node_feature)
        node_feature = self.nlayer_1(node_feature, edge_index, edge_feature, edge_irr)
        node_feature = self.nlayer_2(node_feature, edge_index, edge_feature, edge_irr)
        return self.nlayer_3(node_feature, edge_index, edge_feature, edge_irr)


class GMTNetForceTensorProductHead(nn.Module):
    def __init__(
        self,
        node_irreps,
        edge_dim: int,
        sh_irreps: str = "1x0e + 1x1o + 1x2e",
        reduce: str = "mean",
    ):
        super().__init__()
        self.sh = sh_irreps
        self.reduce = reduce
        self.tp = o3.FullyConnectedTensorProduct(
            node_irreps,
            self.sh,
            "1x1o",
            shared_weights=False,
        )
        self.fc = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.Softplus(),
            nn.Linear(edge_dim, self.tp.weight_numel),
        )

    def forward(self, node_features, edge_vec, edge_index, edge_features):
        edge_src, edge_dst = edge_index
        edge_sh = o3.spherical_harmonics(
            self.sh,
            edge_vec,
            normalize=True,
            normalization="component",
        )
        messages = self.tp(node_features[edge_dst], edge_sh, self.fc(edge_features))
        return scatter(
            messages,
            edge_src,
            dim=0,
            dim_size=node_features.shape[0],
            reduce=self.reduce,
        )


class GMTNetScalarTensorProductHead(nn.Module):
    def __init__(
        self,
        node_irreps,
        edge_dim: int,
        scalar_channels: int = 16,
        sh_irreps: str = "1x0e + 1x1o + 1x2e",
        reduce: str = "mean",
    ):
        super().__init__()
        self.sh = sh_irreps
        self.reduce = reduce
        scalar_irreps = f"{scalar_channels}x0e"
        self.tp = o3.FullyConnectedTensorProduct(
            node_irreps,
            self.sh,
            scalar_irreps,
            shared_weights=False,
        )
        self.fc = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.Softplus(),
            nn.Linear(edge_dim, self.tp.weight_numel),
        )
        self.mlp = nn.Sequential(
            nn.Linear(scalar_channels, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, 1),
        )

    def forward(self, node_features, edge_vec, edge_index, edge_features, batch_index):
        edge_src, edge_dst = edge_index
        edge_sh = o3.spherical_harmonics(
            self.sh,
            edge_vec,
            normalize=True,
            normalization="component",
        )
        messages = self.tp(node_features[edge_dst], edge_sh, self.fc(edge_features))
        node_scalars = scatter(
            messages,
            edge_src,
            dim=0,
            dim_size=node_features.shape[0],
            reduce=self.reduce,
        )
        graph_scalars = scatter(node_scalars, batch_index, dim=0, reduce="mean")
        return self.mlp(graph_scalars).squeeze(-1)


class GradientBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.sh = "1x1o"
        self.tp = o3.FullyConnectedTensorProduct(
            "1x0e + 1x0o + 1x1e + 1x1o + 1x2e + 1x2o + 1x3e + 1x3o",
            self.sh,
            "1x1o",
            internal_weights=False,
        )
        self.constant_w = nn.Parameter(torch.ones(self.tp.weight_numel), requires_grad=False)

    def forward(self, node_feature):
        with torch.enable_grad():
            bs = node_feature.shape[0]
            outer_e = torch.ones(bs, 3, device=node_feature.device, requires_grad=True)
            e_feature = o3.spherical_harmonics(self.sh, outer_e, normalize=False)
            d_feature = self.tp(node_feature, e_feature, self.constant_w.to(node_feature.device))
            dielectric = []
            for i in range(3):
                grad_outputs = torch.zeros(bs, 3, device=node_feature.device)
                grad_outputs[:, i] = 1.0
                dielectric.append(
                    grad(
                        d_feature,
                        outer_e,
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                )
            return torch.stack(dielectric).transpose(0, 1)


class PiezoBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.stress = "1x0e + 1x2e"
        self.converter = CartesianTensor("ij=ji")
        self.tp = o3.FullyConnectedTensorProduct(
            "2x0e + 2x0o + 2x1e + 2x1o + 2x2e + 2x2o + 2x3e + 2x3o",
            self.stress,
            "1x1o",
            internal_weights=False,
        )
        self.constant_w = nn.Parameter(torch.ones(self.tp.weight_numel))
        self.idx = [0, 4, 8, 1, 5, 6]

    def forward(self, node_feature):
        with torch.enable_grad():
            bs = node_feature.shape[0]
            outer_s = torch.ones(bs, 3, 3, device=node_feature.device, requires_grad=True)
            stress = self.converter.from_cartesian(outer_s)
            d_feature = self.tp(node_feature, stress, self.constant_w.to(node_feature.device))
            piezo = []
            for i in range(3):
                grad_outputs = torch.zeros(bs, 3, device=node_feature.device)
                grad_outputs[:, i] = 1.0
                piezo.append(
                    grad(
                        d_feature,
                        outer_s,
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        retain_graph=True,
                    )[0].reshape(bs, 9)[:, [0, 4, 8, 1, 5, 6]]
                )
            return torch.stack(piezo).transpose(0, 1)


class ElasticBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.strain = "1x0e + 1x2e"
        self.converter = CartesianTensor("ij=ji")
        self.tp = o3.FullyConnectedTensorProduct(
            "2x0e + 2x0o + 2x1e + 2x1o + 2x2e + 2x2o + 2x3e + 2x3o + 1x4e",
            self.strain,
            "1x0e + 1x2e",
            internal_weights=False,
        )
        self.constant_w = nn.Parameter(torch.ones(self.tp.weight_numel))
        self.idx = [0, 4, 8, 1, 5, 6]

    def forward(self, node_feature):
        with torch.enable_grad():
            bs = node_feature.shape[0]
            outer_strain = torch.ones(
                bs, 3, 3, device=node_feature.device, requires_grad=True
            )
            strain = self.converter.from_cartesian(outer_strain)
            stress = self.tp(node_feature, strain, self.constant_w)
            final_stress = self.converter.to_cartesian(stress).view(bs, -1)
            grad_outputs = torch.zeros(bs, 9, device=node_feature.device)
            elastic = []
            for i in range(6):
                grad_outputs.zero_()
                grad_outputs[:, self.idx[i]] = 1.0
                grad_elastic = grad(
                    final_stress,
                    outer_strain,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                elastic.append(grad_elastic.reshape(bs, -1)[:, self.idx].reshape(bs, -1))
            return torch.stack(elastic).transpose(0, 1)
