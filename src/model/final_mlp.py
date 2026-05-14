import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

from e3nn.o3 import Irreps, Linear
from e3nn.nn import Gate, BatchNorm

try:
    from .layer_norm import SeperableLayerNorm
except ImportError:
    from layer_norm import SeperableLayerNorm


class FinalMLP(nn.Module):
    def __init__(
        self,
        irreps_in: Union[Irreps, str],
        irreps_out: Union[Irreps, str],
        irreps_hidden: Union[Irreps, str],
        num_hidden_layers: int,
    ):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.irreps_hidden = Irreps(irreps_hidden)
        self.num_hidden_layers = num_hidden_layers
        assert num_hidden_layers > 0, "num_hidden_layers must be greater than 0"

        irreps_hidden_scalar = Irreps(
            [(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0]
        )
        irreps_hidden_gated = Irreps(
            [(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0]
        )
        irreps_hidden_gates = Irreps([(mul, "0e") for mul, _ in irreps_hidden_gated])

        irreps_out_scalar = Irreps(
            [(mul, ir) for mul, ir in self.irreps_out if ir.l == 0]
        )
        irreps_out_gated = Irreps(
            [(mul, ir) for mul, ir in self.irreps_out if ir.l > 0]
        )
        irreps_out_gates = Irreps([(mul, "0e") for mul, _ in irreps_out_gated])

        self.gate_hidden = Gate(
            irreps_scalars=irreps_hidden_scalar,
            act_scalars=[F.tanh for _ in irreps_hidden_scalar],
            irreps_gates=irreps_hidden_gates,
            act_gates=[F.sigmoid for _ in irreps_hidden_gates],
            irreps_gated=irreps_hidden_gated,
        )
        self.gate_out = Gate(
            irreps_scalars=irreps_out_scalar,
            act_scalars=[F.tanh for _ in irreps_out_scalar],
            irreps_gates=irreps_out_gates,
            act_gates=[F.sigmoid for _ in irreps_out_gates],
            irreps_gated=irreps_out_gated,
        )

        self.mlp = nn.ModuleList()
        self.mlp.append(
            nn.Sequential(
                Linear(self.irreps_in, self.gate_hidden.irreps_in),
                self.gate_hidden,
                # SeperableLayerNorm(self.gate_hidden.irreps_out),
            )
        )
        for _ in range(num_hidden_layers):
            self.mlp.append(
                nn.Sequential(
                    Linear(self.gate_hidden.irreps_out, self.gate_hidden.irreps_in),
                    self.gate_hidden,
                    # SeperableLayerNorm(self.gate_hidden.irreps_out),
                )
            )
        self.mlp.append(
            nn.Sequential(
                Linear(self.gate_hidden.irreps_out, self.gate_out.irreps_in),
                self.gate_out,
                SeperableLayerNorm(self.gate_out.irreps_out),
                Linear(self.gate_out.irreps_out, self.irreps_out),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.mlp:
            x = layer(x)

        return x
