import torch
from e3nn.o3 import Irreps, rand_matrix
from src.model.equivariant_layer import EquivariantLayer
from src.model.readout_layer import ReadoutLayer

def test_equivariant_layer():
    irreps_in = Irreps("0e + 2x1o")
    irreps_out = Irreps("0e + 3x1o + 5x2e")
    irreps_vec = Irreps("0e + 1o + 2e")
    num_nodes = 10
    src = torch.tensor([0, 1, 1, 2, 3])
    dst = torch.tensor([1, 2, 3, 4, 0])
    edge_index = torch.stack([src, dst])
    batch_index = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    assert num_nodes == len(batch_index)
    num_edges = edge_index.shape[1]

    edge_vec = torch.randn(num_edges, 3)
    atom_feature = torch.randn(num_nodes, irreps_in.dim)
    edge_feature = torch.randn(num_edges, irreps_in.dim)

    rot_mat = rand_matrix()
    D_in = irreps_in.D_from_matrix(rot_mat)
    D_out = irreps_out.D_from_matrix(rot_mat)

    layer = EquivariantLayer("tpconv_with_edge", irreps_in, irreps_out, irreps_vec)

    atom_out, edge_out = layer(atom_feature, edge_vec, edge_index, edge_feature)
    atom_out_after_rot, edge_out_after_rot = layer(
        atom_feature @ D_in, edge_vec @ rot_mat, edge_index, edge_feature @ D_in
    )

    assert torch.allclose(atom_out @ D_out, atom_out_after_rot, atol=1e-4)
    assert torch.allclose(edge_out @ D_out, edge_out_after_rot, atol=1e-4)

def test_readout_layer():
    irreps_in = Irreps("0e + 2x1o + 3x2e")
    num_nodes = 10
    formula = "ij=ji"
    l_max = 2
    global_feature = torch.randn(num_nodes, irreps_in.dim)
    
    rot_mat = rand_matrix()
    D_in = irreps_in.D_from_matrix(rot_mat)

    layer = ReadoutLayer(l_max, symmetry=formula, irreps_out=irreps_in)
    property_out = layer(global_feature)
    property_out_after_rot = layer(global_feature @ D_in)

    assert torch.allclose(rot_mat.T @ property_out @ rot_mat, property_out_after_rot, atol=1e-4)


if __name__ == "__main__":
    test_equivariant_layer()
    test_readout_layer()
    print("Equivariance test passed!")