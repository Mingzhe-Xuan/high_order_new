import torch
import torch.nn as nn
import torch.nn.functional as F
# from e3nn.o3 import Irreps, Linear
from torch.nn import Linear


class MiddleMLP(nn.Module):
    def __init__(
        self,
        scalar_dim_in: int,
        scalar_dim_hidden: int,
        scalar_dim_out: int,
        num_hidden_layers: int,
        act: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.middle_mlp = nn.ModuleList()
        self.act = act
        assert num_hidden_layers > 0, "num_hidden_layers must be greater than 0"

        self.middle_mlp.append(Linear(scalar_dim_in, scalar_dim_hidden))
        for _ in range(num_hidden_layers - 1):
            self.middle_mlp.append(Linear(scalar_dim_hidden, scalar_dim_hidden))
        self.middle_mlp.append(Linear(scalar_dim_hidden, scalar_dim_out))

    def forward(self, scalar_feature: torch.Tensor) -> torch.Tensor:
        for layer in self.middle_mlp:
            scalar_feature = self.act(layer(scalar_feature))
        scalar_feature = F.layer_norm(scalar_feature, scalar_feature.shape[1:])
        return scalar_feature
