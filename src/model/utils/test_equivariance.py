import os
import sys
from argparse import Namespace
from pathlib import Path

import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from e3nn.o3 import Irreps, rand_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import (  # noqa: E402
    EmbeddingLayer,
    EquivariantLayer,
    FinalMLP,
    InvariantLayer,
    MiddleMLP,
    Model,
    ReadoutLayer,
)
from src.model.gmtnet import HighOrderGMTNet  # noqa: E402


def _rotate_cartesian_tensor(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1:
        return tensor
    if tensor.dim() == 2:
        return tensor @ rotation
    if tensor.dim() == 3:
        return rotation.T.unsqueeze(0) @ tensor @ rotation
    raise NotImplementedError(f"No rotation rule for tensor shape {tuple(tensor.shape)}")


def _assert_close(name: str, expected: torch.Tensor, actual: torch.Tensor, atol: float = 1e-4, rtol: float = 1e-4):
    if not torch.allclose(expected, actual, atol=atol, rtol=rtol):
        max_error = (expected - actual).abs().max().item()
        raise AssertionError(f"{name} failed, max abs error={max_error:.6e}")


def _toy_graph(device: torch.device):
    atom_type = torch.tensor([6, 8, 14, 3, 12], dtype=torch.long, device=device)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 0, 2, 4],
            [1, 2, 3, 4, 0, 2, 0, 1],
        ],
        dtype=torch.long,
        device=device,
    )
    edge_vec = torch.tensor(
        [
            [1.2, 0.2, -0.1],
            [-0.4, 1.1, 0.3],
            [0.3, -0.5, 1.4],
            [1.0, 0.7, -0.6],
            [-1.2, 0.4, 0.8],
            [0.6, -1.0, 0.2],
            [-0.7, 0.3, -1.1],
            [0.5, 0.9, 0.4],
        ],
        dtype=torch.float32,
        device=device,
    )
    batch_index = torch.zeros(atom_type.shape[0], dtype=torch.long, device=device)
    return atom_type, edge_vec, edge_index, batch_index


def _build_high_order_model(device: torch.device):
    scalar_dim = 4
    vec_dim = 2
    num_equi_layers = 1
    irreps_list = [f"{scalar_dim}x0e"]
    for l in range(1, num_equi_layers + 1):
        irreps = f"{scalar_dim}x0e"
        for current_l in range(l + 1):
            parity = "e" if current_l % 2 == 0 else "o"
            irreps += f"+{vec_dim}x{current_l}{parity}"
        irreps_list.append(irreps)

    irreps_vec = "0e"
    for l in range(1, num_equi_layers + 1):
        parity = "e" if l % 2 == 0 else "o"
        irreps_vec += f"+{l}{parity}"

    final_irreps_hidden = "8x0e+4x1o+4x2e"
    final_irreps_out = "4x0e+2x1o+2x2e"

    model = Model(
        embedding_layer=EmbeddingLayer(
            dist_emb_func="gaussian",
            embed_dim=8,
            max_atom_type=118,
            cutoff=5.0,
        ),
        invariant_layers=nn.ModuleList(),
        middle_mlp=MiddleMLP(
            scalar_dim_in=8,
            scalar_dim_hidden=16,
            scalar_dim_out=scalar_dim,
            num_hidden_layers=1,
        ),
        equivariant_layers=nn.ModuleList(
            [
                EquivariantLayer(
                    update_method="tpconv_with_edge",
                    irreps_in=irreps_list[i],
                    irreps_out=irreps_list[i + 1],
                    irreps_vec=irreps_vec,
                    tp_method="so2",
                    residual=False,
                )
                for i in range(num_equi_layers)
            ]
        ),
        final_mlp=FinalMLP(
            irreps_in=Irreps(irreps_list[-1]),
            irreps_hidden=final_irreps_hidden,
            irreps_out=final_irreps_out,
            num_hidden_layers=1,
        ),
        readout_layer=ReadoutLayer(
            l_max=1,
            symmetry=None,
            irreps_out=final_irreps_out,
        ),
        self_train=True,
    )
    return model.to(device).eval()


def _build_gmtnet(target: str, task_mode: str, device: torch.device):
    model = HighOrderGMTNet(
        target=target,
        task_mode=task_mode,
        args=Namespace(
            gmtnet_embed_dim=16,
            gmtnet_atom_feature_dim=118,
            gmtnet_num_attention_layers=1,
            gmtnet_scalar_channels=4,
            gmtnet_force_reduce="mean",
            gmtnet_scalar_reduce="mean",
            use_mask=False,
        ),
    )
    return model.to(device).eval()


def test_equivariant_layer_current_interface():
    torch.manual_seed(0)
    device = torch.device("cpu")
    irreps_in = Irreps("4x0e")
    irreps_out = Irreps("4x0e+2x1o+2x2e")
    irreps_vec = Irreps("0e+1o+2e")
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], dtype=torch.long)
    edge_vec = torch.randn(edge_index.shape[1], 3)
    atom_feature = torch.randn(4, irreps_in.dim)
    edge_feature = torch.randn(edge_index.shape[1], irreps_in.dim)

    rotation = rand_matrix()
    layer = EquivariantLayer(
        "tpconv_with_edge",
        irreps_in,
        irreps_out,
        irreps_vec,
        tp_method="so2",
        residual=False,
    ).eval()

    atom_out, edge_out = layer(atom_feature, edge_vec, edge_index, edge_feature)
    atom_out_rot, edge_out_rot = layer(
        atom_feature @ irreps_in.D_from_matrix(rotation),
        edge_vec @ rotation,
        edge_index,
        edge_feature @ irreps_in.D_from_matrix(rotation),
    )

    _assert_close("EquivariantLayer atom output", atom_out @ irreps_out.D_from_matrix(rotation), atom_out_rot)
    _assert_close("EquivariantLayer edge output", edge_out @ irreps_out.D_from_matrix(rotation), edge_out_rot)


def test_readout_layer_current_interface():
    torch.manual_seed(1)
    irreps_in = Irreps("4x0e+2x1o+2x2e")
    layer = ReadoutLayer(l_max=2, symmetry="ij=ji", irreps_out=irreps_in).eval()
    feature = torch.randn(3, irreps_in.dim)
    rotation = rand_matrix()

    output = layer(feature)
    rotated_output = layer(feature @ irreps_in.D_from_matrix(rotation))

    _assert_close("ReadoutLayer", _rotate_cartesian_tensor(output, rotation), rotated_output)


def test_high_order_model_force_equivariance():
    torch.manual_seed(2)
    device = torch.device("cpu")
    model = _build_high_order_model(device)
    atom_type, edge_vec, edge_index, batch_index = _toy_graph(device)
    rotation = rand_matrix().to(device)

    output = model(atom_type, edge_vec, edge_index, batch_index)
    rotated_output = model(atom_type, edge_vec @ rotation, edge_index, batch_index)

    _assert_close("HighOrder force model", output @ rotation, rotated_output, atol=5e-4, rtol=5e-4)


def test_gmtnet_force_equivariance():
    torch.manual_seed(3)
    device = torch.device("cpu")
    model = _build_gmtnet(target=None, task_mode="force", device=device)
    atom_type, edge_vec, edge_index, batch_index = _toy_graph(device)
    rotation = rand_matrix().to(device)

    output = model(atom_type, edge_vec, edge_index, batch_index)
    rotated_output = model(atom_type, edge_vec @ rotation, edge_index, batch_index)

    _assert_close("GMTNet force model", output @ rotation, rotated_output)


def test_gmtnet_scalar_invariance():
    torch.manual_seed(4)
    device = torch.device("cpu")
    model = _build_gmtnet(target="formation_energy", task_mode="scalar", device=device)
    atom_type, edge_vec, edge_index, batch_index = _toy_graph(device)
    rotation = rand_matrix().to(device)

    output = model(atom_type, edge_vec, edge_index, batch_index)
    rotated_output = model(atom_type, edge_vec @ rotation, edge_index, batch_index)

    _assert_close("GMTNet scalar model", output, rotated_output, atol=5e-4, rtol=5e-4)


def test_gmtnet_dielectric_tensor_equivariance():
    torch.manual_seed(5)
    device = torch.device("cpu")
    model = _build_gmtnet(target="dielectric", task_mode="tensor", device=device)
    atom_type, edge_vec, edge_index, batch_index = _toy_graph(device)
    rotation = rand_matrix().to(device)

    output = model(atom_type, edge_vec, edge_index, batch_index)
    rotated_output = model(atom_type, edge_vec @ rotation, edge_index, batch_index)

    _assert_close("GMTNet dielectric tensor model", _rotate_cartesian_tensor(output, rotation), rotated_output, atol=5e-4, rtol=5e-4)


def run_all_tests():
    test_equivariant_layer_current_interface()
    test_readout_layer_current_interface()
    test_high_order_model_force_equivariance()
    test_gmtnet_force_equivariance()
    test_gmtnet_scalar_invariance()
    test_gmtnet_dielectric_tensor_equivariance()


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    run_all_tests()
    print("Equivariance tests passed for high_order and GMTNet models.")
