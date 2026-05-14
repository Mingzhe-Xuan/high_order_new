from math import e
import torch
import torch.nn as nn
from e3nn.io import CartesianTensor
from e3nn.o3 import Irreps, Linear
from typing import Union
import numpy as np

try:
    from .utils import full2voigt, voigt2full, _l_max
except ImportError:
    from utils import full2voigt, voigt2full, _l_max

class ReadoutViaGradLayer(nn.Module):
    def __init__(self, l_max: int, symmetry: str = None):
        super().__init__()
        self.l_max = l_max
        self.symmetry = symmetry
    
    def forward(self, atom_feature: torch.Tensor) -> torch.Tensor:
        pass

class ReadoutLayer(nn.Module):
    def __init__(self, l_max: int, symmetry: str = None, irreps_out: Irreps = None):
        super().__init__()
        self.l_max = l_max
        self.symmetry = symmetry
        self.irreps_out = Irreps(irreps_out)

        # 根据l_max自动选择合适的CartesianTensor formula
        if l_max > 0:
            if symmetry is None:
                if l_max == 1:
                    formula = "i"  # 1阶张量（向量）
                elif l_max == 2:
                    formula = "ij"  # 2阶张量（3x3矩阵）
                elif l_max == 3:
                    formula = "ijk"  # 3阶张量
                elif l_max == 4:
                    formula = "ijkl"  # 4阶张量
                else:
                    raise ValueError(
                        f"Unsupported l_max: {l_max}. Supported values are 1, 2, 3, 4."
                    )
            else:
                formula = symmetry

            self.cartesian_tensor = CartesianTensor(formula)
        else:
            # CartesianTensor cannot handle l=0 with an empty formula
            self.cartesian_tensor = Irreps("0e")

        self.linear_out = Linear(irreps_out, self.cartesian_tensor)
        

    # def _l_3_tensor_to_voigt(self, d_ijk):
    #     """
    #     将 3x3x3 压电张量转换为 3x6 Voigt 记号形式

    #     Args:
    #         d_ijk: 形状为 (batch_size, 3, 3, 3) 的压电张量

    #     Returns:
    #         d_voigt: 形状为 (batch_size, 3, 6) 的 Voigt 记号矩阵
    #     """
    #     batch_size = d_ijk.shape[0]
    #     d_voigt = np.zeros((batch_size, 3, 6))

    #     # Voigt 记号映射
    #     voigt_map = {
    #         0: (0, 0),  # 1 -> (0,0)
    #         1: (1, 1),  # 2 -> (1,1)
    #         2: (2, 2),  # 3 -> (2,2)
    #         3: (1, 2),  # 4 -> (1,2)
    #         4: (0, 2),  # 5 -> (0,2)
    #         5: (0, 1),  # 6 -> (0,1)
    #     }

    #     for b in range(batch_size):
    #         for i in range(3):
    #             for m, (j, k) in voigt_map.items():
    #                 d_voigt[b, i, m] = d_ijk[b, i, j, k]

    #     return d_voigt

    # def _l_4_tensor_to_voigt(self, C_ijkl):
    #     """
    #     将 3x3x3x3 弹性张量转换为 6x6 Voigt 记号形式

    #     Args:
    #         C_ijkl: 形状为 (batch_size, 3, 3, 3, 3) 的弹性张量

    #     Returns:
    #         C_voigt: 形状为 (batch_size, 6, 6) 的 Voigt 记号矩阵
    #     """
    #     batch_size = C_ijkl.shape[0]
    #     C_voigt = np.zeros((batch_size, 6, 6))

    #     # Voigt 记号映射
    #     voigt_map = {
    #         0: (0, 0),  # 1 -> (0,0)
    #         1: (1, 1),  # 2 -> (1,1)
    #         2: (2, 2),  # 3 -> (2,2)
    #         3: (1, 2),  # 4 -> (1,2)
    #         4: (0, 2),  # 5 -> (0,2)
    #         5: (0, 1),  # 6 -> (0,1)
    #     }

    #     # 因子：正应变为1，剪切应变为2
    #     factor = np.array([1, 1, 1, 2, 2, 2])

    #     for b in range(batch_size):
    #         for m in range(6):
    #             for n in range(6):
    #                 i, j = voigt_map[m]
    #                 k, l = voigt_map[n]
    #                 C_voigt[b, m, n] = C_ijkl[b, i, j, k, l] * factor[m] * factor[n]

    #     return C_voigt

    def forward(
        self,
        global_feature: torch.Tensor,
        # irreps_out: Union[str, Irreps],
        # property_name: Union[str, None] = None,
        # linear_adaption: bool = False, -- Deprecated --
    ) -> torch.Tensor:
        # if isinstance(self.irreps_out, str):
        #     self.irreps_out = Irreps(self.irreps_out)

        if self.l_max > _l_max:
            raise ValueError(
                f"Unsupported l_max: {self.l_max}. Max supported l_max is {_l_max}."
            )

        # else:
        #     # if not linear_adaption:
        #     #     # 获取CartesianTensor的l值列表
        #     #     cart_ls = self.cartesian_tensor.ls
        #     #     # 将irreps_out按照l值分组
        #     #     l_to_features = {}
        #     #     start_idx = 0

        #     #     for mul, (l, p) in irreps_out:
        #     #         dim = (2 * l + 1) * mul

        #     #         if l in cart_ls:
        #     #             if start_idx + dim <= global_feature.shape[1]:
        #     #                 feature_slice = global_feature[
        #     #                     :, start_idx : start_idx + dim
        #     #                 ]
        #     #             else:
        #     #                 feature_slice = global_feature[:, start_idx:]
        #     #                 padding = torch.zeros(
        #     #                     global_feature.shape[0],
        #     #                     dim - feature_slice.shape[1],
        #     #                     device=global_feature.device,
        #     #                     dtype=global_feature.dtype,
        #     #                 )
        #     #                 feature_slice = torch.cat([feature_slice, padding], dim=1)

        #     #             if mul > 1:
        #     #                 feature_slice = feature_slice.view(
        #     #                     global_feature.shape[0], mul, 2 * l + 1
        #     #                 )
        #     #                 feature_slice = feature_slice.mean(dim=1)

        #     #             if l not in l_to_features:
        #     #                 l_to_features[l] = []
        #     #             l_to_features[l].append(feature_slice)

        #     #         start_idx += dim

        #     #     # 为CartesianTensor的每个l值收集特征
        #     #     pooled_features = []
        #     #     for l in cart_ls:
        #     #         if l in l_to_features and l_to_features[l]:
        #     #             l_features = torch.stack(l_to_features[l], dim=0).mean(dim=0)
        #     #             pooled_features.append(l_features)
        #     #         else:
        #     #             pooled_features.append(
        #     #                 torch.zeros(
        #     #                     global_feature.shape[0],
        #     #                     2 * l + 1,
        #     #                     device=global_feature.device,
        #     #                     dtype=global_feature.dtype,
        #     #                 )
        #     #             )

        #     #     global_feature = torch.cat(pooled_features, dim=1)
        #     # else:
        # Deprecated: parameters of linear layer in this version will be different
        # each time call the forward method, which is not what we want.
        # More importantly, this will destroy the equivariance of the model
        # if we use the model multiple times.
        #         linear_out = Linear(irreps_out, self.cartesian_tensor)
        #         global_feature = linear_out(global_feature)

        # Create Linear in this way will cause no gradient tracking
        # # 创建或获取 Linear 层
        # irreps_key = str(irreps_out)
        
        # if irreps_key not in self._linear_layers:
        #     self._linear_layers[irreps_key] = Linear(irreps_out, self.cartesian_tensor)
        #     self._linear_layers[irreps_key].to(global_feature.device)
        # else:
        #     linear_layer = self._linear_layers[irreps_key]
        #     if next(linear_layer.parameters()).device != global_feature.device:
        #         self._linear_layers[irreps_key].to(global_feature.device)
        
        # linear_out = self._linear_layers[irreps_key]


        global_feature = self.linear_out(global_feature)

        if self.l_max == 0:
            cart_property = global_feature
        else:
            cart_property = self.cartesian_tensor.to_cartesian(global_feature)
        # if "dielectric" in property_name:
        #     return cart_property
        # elif "piezoelectric" in property_name:
        #     return torch.from_numpy(
        #         self._l_3_tensor_to_voigt(cart_property.cpu().numpy())
        #     )
        # elif "elastic" in property_name:
        #     return torch.from_numpy(
        #         self._l_4_tensor_to_voigt(cart_property.cpu().numpy())
        #     )
        # else:
        #     raise NotImplementedError("property_name not supported")
        return full2voigt(self.l_max, cart_property)
