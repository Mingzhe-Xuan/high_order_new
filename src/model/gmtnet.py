from argparse import Namespace

import torch
from torch import nn
from torch_scatter import scatter

from .gmtnet_blocks import (
    ComformerConv,
    ComformerConvEqui,
    ElasticBlock,
    GMTNetForceTensorProductHead,
    GMTNetScalarTensorProductHead,
    GradientBlock,
    PiezoBlock,
    RBFExpansion,
)


def equality_adjustment(equality, batch):
    if equality.dim() == 4:
        equality = equality[:, 0]
    b, l1, l2 = batch.size()
    batch = batch.reshape(b, l1 * l2)
    for i in range(b):
        mask = equality[i]
        for j in range(l1 * l2):
            for k in range(j + 1, l1 * l2):
                if mask[j, k]:
                    batch[i, j] = batch[i, k] = (batch[i, j] + batch[i, k]) / 2
    return batch.reshape(b, l1, l2)


def _target_name(target):
    if target in {"dielectric", "dielectric_ionic"}:
        return "dielectric"
    if target in {"piezoelectric", "piezoelectric_C_m2", "piezoelectric_e_Angst"}:
        return "piezoelectric"
    if target in {"elastic", "elastic_sym_kbar", "elastic_total_kbar"}:
        return "elastic"
    raise NotImplementedError(f"{target} property not implemented")


def _target_irreps(target):
    target = _target_name(target)
    if target == "dielectric":
        return "1x0e + 1x0o + 1x1e + 1x1o + 1x2e + 1x2o + 1x3e + 1x3o"
    if target == "piezoelectric":
        return "2x0e + 2x0o + 2x1e + 2x1o + 2x2e + 2x2o + 2x3e + 2x3o"
    return "2x0e + 2x0o + 2x1e + 2x1o + 2x2e + 2x2o + 2x3e + 2x3o + 1x4e"


DEFAULT_GMTNET_BACKBONE_IRREPS = "2x0e + 2x0o + 2x1e + 2x1o + 2x2e + 2x2o + 2x3e + 2x3o"


class HighOrderGMTNet(nn.Module):
    def __init__(
        self,
        args: Namespace | None = None,
        target: str | None = None,
        task_mode: str = "tensor",
    ):
        super().__init__()
        if args is None:
            args = Namespace()
        self.task_mode = task_mode
        if self.task_mode not in {"tensor", "scalar", "force"}:
            raise ValueError(f"task_mode {self.task_mode} is not implemented")
        self.target = target or getattr(args, "target", None)
        embsize = getattr(args, "gmtnet_embed_dim", 128)
        atom_feature_dim = getattr(args, "gmtnet_atom_feature_dim", 118)
        num_attention_layers = getattr(args, "gmtnet_num_attention_layers", 2)
        target_irreps = _target_irreps(self.target) if self.task_mode == "tensor" else getattr(
            args,
            "gmtnet_backbone_irreps",
            DEFAULT_GMTNET_BACKBONE_IRREPS,
        )
        sh_irreps = getattr(args, "gmtnet_sh_irreps", "1x0e + 1x1o + 1x2e")
        self.mask = getattr(args, "use_mask", False)
        self.atom_embedding = nn.Linear(atom_feature_dim, embsize)
        self.rbf = nn.Sequential(
            RBFExpansion(vmin=-4.0, vmax=0.0, bins=512),
            nn.Linear(512, embsize),
            nn.Softplus(),
        )
        self.att_layers = nn.ModuleList(
            [
                ComformerConv(
                    in_channels=embsize,
                    out_channels=embsize,
                    heads=1,
                    edge_dim=embsize,
                )
                for _ in range(num_attention_layers)
            ]
        )
        self.equi_update = ComformerConvEqui(
            in_channels=embsize,
            edge_dim=embsize,
            target_irreps=target_irreps,
        )
        if self.task_mode == "force":
            self.output_block = GMTNetForceTensorProductHead(
                node_irreps=target_irreps,
                edge_dim=embsize,
                sh_irreps=sh_irreps,
                reduce=getattr(args, "gmtnet_force_reduce", "mean"),
            )
        elif self.task_mode == "scalar":
            self.output_block = GMTNetScalarTensorProductHead(
                node_irreps=target_irreps,
                edge_dim=embsize,
                scalar_channels=getattr(args, "gmtnet_scalar_channels", 16),
                sh_irreps=sh_irreps,
                reduce=getattr(args, "gmtnet_scalar_reduce", "mean"),
            )
        else:
            target_name = _target_name(self.target)
            if target_name == "dielectric":
                self.output_block = GradientBlock()
            elif target_name == "piezoelectric":
                self.output_block = PiezoBlock()
            else:
                self.output_block = ElasticBlock()

    def forward(
        self,
        atom_type,
        edge_vec,
        edge_index,
        batch_index,
        feat_mask=None,
        equality=None,
    ) -> torch.Tensor:
        node_features, edge_features, crystal_features = self._forward_backbone(
            atom_type,
            edge_vec,
            edge_index,
            batch_index,
        )
        if self.task_mode == "force":
            return self.output_block(node_features, edge_vec, edge_index, edge_features)
        if self.task_mode == "scalar":
            return self.output_block(
                node_features,
                edge_vec,
                edge_index,
                edge_features,
                batch_index,
            )
        if self.mask and feat_mask is not None:
            crystal_features = torch.bmm(feat_mask, crystal_features.unsqueeze(-1)).squeeze(-1)
        outputs = self.output_block(crystal_features)
        if equality is not None and outputs.dim() == 3:
            outputs = equality_adjustment(equality, outputs)
        return outputs

    def _forward_backbone(self, atom_type, edge_vec, edge_index, batch_index):
        node_features = self.atom_embedding(self._atom_features(atom_type))
        edge_dist = torch.norm(edge_vec, dim=1).clamp_min(1e-8)
        edge_features = self.rbf(-0.75 / edge_dist)
        for att_layer in self.att_layers:
            node_features = att_layer(node_features, edge_index, edge_features)
        node_features = self.equi_update(node_features, edge_vec, edge_index, edge_features)
        crystal_features = scatter(node_features, batch_index, dim=0, reduce="mean")
        return node_features, edge_features, crystal_features

    def _atom_features(self, atom_type):
        if atom_type.dim() == 2:
            return atom_type.float()
        num_classes = self.atom_embedding.in_features
        zero_based = atom_type.clamp_min(1) - 1
        if zero_based.numel() > 0 and zero_based.max().item() >= num_classes:
            raise ValueError(
                f"atom_type contains atomic number {atom_type.max().item()}, "
                f"but gmtnet_atom_feature_dim={num_classes}"
            )
        return torch.nn.functional.one_hot(zero_based, num_classes=num_classes).float()
