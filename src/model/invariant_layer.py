import torch
import torch.nn as nn
import math
from torch_scatter import scatter
import torch.nn.functional as F


class BiasGATLayer(nn.Module):
    """Implementation of the bias GAT update mechanism."""
    def __init__(self, scalar_dim: int):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.q_lin = nn.Linear(scalar_dim, scalar_dim)
        self.k_lin = nn.Linear(scalar_dim, scalar_dim)
        self.v_lin = nn.Linear(scalar_dim, scalar_dim)
        self.e_lin = nn.Linear(scalar_dim, scalar_dim)
        self.alpha_norm = nn.LayerNorm(scalar_dim)
        self.message_norm = nn.LayerNorm(scalar_dim)
        self.act = nn.Softplus()
        self.k_mlp = nn.Sequential(
            nn.Linear(scalar_dim * 3, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        self.v_mlp = nn.Sequential(
            nn.Linear(scalar_dim * 3, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        self.gate = nn.Sigmoid()

    def bias_gat_attn(
        self,
        src_feature: torch.Tensor,  # (num_edges, scalar_dim)
        dst_feature: torch.Tensor,  # (num_edges, scalar_dim)
        edge_feature: torch.Tensor,  # (num_edges, scalar_dim)
    ) -> torch.Tensor:
        # q, k, v: (num_edges, scalar_dim)
        q = self.q_lin(dst_feature)
        k = self.k_lin(src_feature)
        v = self.v_lin(src_feature)
        # edge_feature: (num_edges, scalar_dim)
        edge_feature_after_lin = self.e_lin(edge_feature)
        # attn: (num_edges, scalar_dim)
        attn = (
            torch.softmax(q * k, dim=-1)
            / torch.sqrt(torch.tensor(self.scalar_dim, dtype=q.dtype))
            + edge_feature_after_lin
        )
        # message: (num_edges, scalar_dim)
        message = attn * v
        return message, attn

    def bias_gat_update(
        self,
        atom_feature: torch.Tensor,
        edge_feature: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        src_feature = atom_feature[src]
        dst_feature = atom_feature[dst]
        message, attn = self.bias_gat_attn(src_feature, dst_feature, edge_feature)
        num_nodes = atom_feature.size(0)
        atom_feature = atom_feature + scatter(
            message, dst, dim=0, dim_size=num_nodes, reduce="sum"
        )
        edge_feature = edge_feature + attn
        return atom_feature, edge_feature

    def forward(
        self,
        atom_feature: torch.Tensor,
        edge_feature: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        return self.bias_gat_update(atom_feature, edge_feature, edge_index)


# Revised from comformer, batchnorm -> layernorm to adapt variable num_atoms
class ComformerLayer(nn.Module):
    """Implementation of the ComFormer update mechanism."""
    def __init__(self, scalar_dim: int, heads: int = 8):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.heads = heads
        self.head_dim = scalar_dim // heads if heads > 1 else scalar_dim
        
        self.q_lin = nn.Linear(scalar_dim, scalar_dim)
        self.k_lin = nn.Linear(scalar_dim, scalar_dim)
        self.v_lin = nn.Linear(scalar_dim, scalar_dim)
        self.e_lin = nn.Linear(scalar_dim, scalar_dim)
        self.alpha_norm = nn.LayerNorm(scalar_dim)
        self.message_norm = nn.LayerNorm(scalar_dim)
        self.act = nn.Softplus()
        self.k_mlp = nn.Sequential(
            nn.Linear(scalar_dim * 3, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        self.v_mlp = nn.Sequential(
            nn.Linear(scalar_dim * 3, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        self.gate = nn.Sigmoid()
        
        self.key_update_mh = nn.Sequential(
            nn.Linear(self.head_dim * 3, self.head_dim),
            nn.SiLU(),
            nn.Linear(self.head_dim, self.head_dim),
        )
        self.msg_update_mh = nn.Sequential(
            nn.Linear(self.head_dim * 3, self.head_dim),
            nn.SiLU(),
            nn.Linear(self.head_dim, self.head_dim),
        )
        self.bn_att = nn.BatchNorm1d(self.head_dim)
        self.bn_msg = nn.BatchNorm1d(scalar_dim)
        self.sigmoid = nn.Sigmoid()

    def comformer_node_attn(
        self,
        src_feature: torch.Tensor,  # (num_edges, scalar_dim)
        dst_feature: torch.Tensor,  # (num_edges, scalar_dim)
        edge_feature: torch.Tensor,  # (num_edges, scalar_dim)
    ) -> torch.Tensor:
        # _q: (num_edges, scalar_dim)
        _q = self.q_lin(dst_feature)
        # _k: (num_edges, scalar_dim * 3)
        _k = torch.stack(
            [
                self.k_lin(src_feature),
                self.k_lin(dst_feature),
                self.e_lin(edge_feature),
            ],
            dim=-1,
        ).view(-1, self.scalar_dim * 3)
        # _v: (num_edges, scalar_dim * 3)
        _v = torch.stack(
            [
                self.v_lin(src_feature),
                self.v_lin(dst_feature),
                self.e_lin(edge_feature),
            ],
            dim=-1,
        ).view(-1, self.scalar_dim * 3)
        # q, k, v: (num_edges, scalar_dim)
        q = _q
        k = self.k_mlp(_k)
        v = self.v_mlp(_v)
        # alpha: (num_edges, scalar_dim)
        scale = torch.tensor(self.scalar_dim, dtype=q.dtype)
        alpha = self.alpha_norm(q * k / torch.sqrt(scale))
        # message: (num_edges, scalar_dim)
        message = self.message_norm(self.gate(alpha) * v)
        return message

    def comformer_node_attn_multi_head(
        self,
        src_feature: torch.Tensor,  # (num_edges, scalar_dim)
        dst_feature: torch.Tensor,  # (num_edges, scalar_dim)
        edge_feature: torch.Tensor,  # (num_edges, scalar_dim)
    ) -> torch.Tensor:
        num_edges = src_feature.size(0)
        H, D = self.heads, self.head_dim
        
        q = self.q_lin(dst_feature).view(num_edges, H, D)
        k = self.k_lin(src_feature).view(num_edges, H, D)
        k_dst = self.k_lin(dst_feature).view(num_edges, H, D)
        k_edge = self.e_lin(edge_feature).view(num_edges, H, D)
        k_combined = torch.cat([k, k_dst, k_edge], dim=-1)
        k_transformed = self.key_update_mh(k_combined)
        
        v = self.v_lin(src_feature).view(num_edges, H, D)
        v_dst = self.v_lin(dst_feature).view(num_edges, H, D)
        v_edge = self.e_lin(edge_feature).view(num_edges, H, D)
        v_combined = torch.cat([v, v_dst, v_edge], dim=-1)
        v_transformed = self.msg_update_mh(v_combined)
        
        alpha = (q * k_transformed) / math.sqrt(D)
        alpha = self.bn_att(alpha.view(-1, D)).view(num_edges, H, D)
        
        message = v_transformed * self.sigmoid(alpha)
        message = message.view(num_edges, H * D)
        message = self.bn_msg(message)
        
        return message

    def comformer_update(
        self,
        atom_feature: torch.Tensor,  # (num_nodes, scalar_dim)
        edge_feature: torch.Tensor,  # (num_edges, scalar_dim)
        edge_index: torch.Tensor,  # (2, num_edges)
    ) -> torch.Tensor:
        src, dst = edge_index
        # src_feature: (num_edges, scalar_dim)
        src_feature = atom_feature[src]
        # dst_feature: (num_edges, scalar_dim)
        dst_feature = atom_feature[dst]
        # message: (num_edges, scalar_dim)
        message = self.comformer_node_attn(
            src_feature, dst_feature, edge_feature
        )
        # atom_feature: (num_nodes, scalar_dim)
        num_nodes = atom_feature.size(0)
        atom_feature = atom_feature + scatter(
            message, dst, dim=0, dim_size=num_nodes, reduce="sum"
        )
        # edge_feature: (num_edges, scalar_dim)
        edge_feature = F.softplus(edge_feature + message)
        return atom_feature, edge_feature

    def comformer_update_multi_head(
        self,
        atom_feature: torch.Tensor,  # (num_nodes, scalar_dim)
        edge_feature: torch.Tensor,  # (num_edges, scalar_dim)
        edge_index: torch.Tensor,  # (2, num_edges)
    ) -> torch.Tensor:
        src, dst = edge_index
        src_feature = atom_feature[src]
        dst_feature = atom_feature[dst]
        message = self.comformer_node_attn_multi_head(
            src_feature, dst_feature, edge_feature
        )
        num_nodes = atom_feature.size(0)
        atom_feature = atom_feature + scatter(
            message, dst, dim=0, dim_size=num_nodes, reduce="sum"
        )
        edge_feature = F.softplus(edge_feature + message)
        return atom_feature, edge_feature

    def forward(
        self,
        atom_feature: torch.Tensor,  # (num_nodes, scalar_dim)
        edge_feature: torch.Tensor,  # (num_edges, scalar_dim)
        edge_index: torch.Tensor,  # (2, num_edges)
    ):
        if self.heads > 1:
            return self.comformer_update_multi_head(atom_feature, edge_feature, edge_index)
        return self.comformer_update(atom_feature, edge_feature, edge_index)

# class ComformerConv(MessagePassing):
#     _alpha: OptTensor

#     def __init__(
#         self,
#         in_channels: Union[int, Tuple[int, int]],
#         out_channels: int,
#         heads: int = 1,
#         concat: bool = True,
#         beta: bool = False,
#         dropout: float = 0.0,
#         edge_dim: Optional[int] = None,
#         bias: bool = True,
#         root_weight: bool = True,
#         **kwargs,
#     ):
#         kwargs.setdefault('aggr', 'add')
#         super(ComformerConv, self).__init__(node_dim=0, **kwargs)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.heads = heads
#         self.beta = beta and root_weight
#         self.root_weight = root_weight
#         self.concat = concat
#         self.dropout = dropout
#         self.edge_dim = edge_dim
#         self._alpha = None

#         if isinstance(in_channels, int):
#             in_channels = (in_channels, in_channels)

#         self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
#         self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
#         self.lin_value = nn.Linear(in_channels[0], heads * out_channels)
#         self.lin_edge = nn.Linear(edge_dim, heads * out_channels)
#         self.lin_concate = nn.Linear(heads * out_channels, out_channels)
        
#         self.lin_msg_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
#                                         nn.SiLU(),
#                                         nn.Linear(out_channels, out_channels))
#         self.softplus = nn.Softplus()
#         self.silu = nn.SiLU()
#         self.key_update = nn.Sequential(nn.Linear(out_channels * 3, out_channels),
#                                         nn.SiLU(),
#                                         nn.Linear(out_channels, out_channels))
#         self.bn = nn.BatchNorm1d(out_channels)
#         self.bn_att = nn.BatchNorm1d(out_channels)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
#                 edge_attr: OptTensor = None, return_attention_weights=None):

#         H, C = self.heads, self.out_channels
#         if isinstance(x, Tensor):
#             x: PairTensor = (x, x)
        
#         query = self.lin_query(x[1]).view(-1, H, C)
#         key = self.lin_key(x[0]).view(-1, H, C)
#         value = self.lin_value(x[0]).view(-1, H, C)

#         out = self.propagate(edge_index, query=query, key=key, value=value,
#                              edge_attr=edge_attr, size=None)
        
#         out = out.view(-1, self.heads * self.out_channels)
#         out = self.lin_concate(out)
        
#         return self.softplus(x[1] + out)

#     def message(self, query_i: Tensor, key_i: Tensor, key_j: Tensor, value_j: Tensor, value_i: Tensor,
#                 edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
#                 size_i: Optional[int]) -> Tensor:

#         edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
#         key_j = self.key_update(torch.cat((key_i, key_j, edge_attr), dim=-1))
#         alpha = (query_i * key_j) / math.sqrt(self.out_channels)
#         out = self.lin_msg_update(torch.cat((value_i, value_j, edge_attr), dim=-1))
#         out = out * self.sigmoid(self.bn_att(alpha.view(-1, self.out_channels)).view(-1, self.heads, self.out_channels))
#         return out

class InvariantLayer(nn.Module):
    """
    InvariantLayer class that creates either a BiasGATLayer or ComformerLayer
    based on the update_method parameter.
    """
    def __init__(self, update_method: str, scalar_dim: int):
        super().__init__()
        self.scalar_dim = scalar_dim
        if update_method == 'bias_gat':
            self.layer = BiasGATLayer(scalar_dim)
        elif update_method == 'comformer':
            self.layer = ComformerLayer(scalar_dim)
        else:
            raise NotImplementedError(f'Not implemented yet: {update_method}')

    def forward(
        self,
        atom_feature: torch.Tensor,  # (num_nodes, scalar_dim)
        edge_feature: torch.Tensor,  # (num_edges, scalar_dim)
        edge_index: torch.Tensor,  # (2, num_edges)
    ):
        return self.layer(atom_feature, edge_feature, edge_index)
