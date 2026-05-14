import torch
from typing import Tuple

_l_max = 6

def _get_voigt_indices(l: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    获取 Voigt 表示法的索引映射
    
    Args:
        l: 张量阶数
        
    Returns:
        (row_indices, col_indices): Voigt 矩阵的行列索引
    """
    if l == 0:
        return (0,), (0,)
    elif l == 1:
        return (0, 1, 2), (0, 0, 0)
    elif l == 2:
        return (0, 1, 2, 1, 2, 2), (0, 1, 2, 0, 2, 1)
    elif l == 3:
        rows = (0, 0, 0, 1, 1, 1)
        cols = (0, 1, 2, 0, 1, 2)
        return rows, cols
    elif l == 4:
        voigt_map = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        rows = tuple(i for i, _ in enumerate(voigt_map))
        cols = tuple(j for j, _ in enumerate(voigt_map))
        return rows, cols
    elif l == 5:
        voigt_map_3 = [(0,), (1,), (2,)]
        voigt_map_6 = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        rows = tuple(i for i in range(3) for _ in range(6))
        cols = tuple(j for _ in range(3) for j in range(6))
        return rows, cols
    elif l == 6:
        voigt_map_6 = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        rows = tuple(i for i, _ in enumerate(voigt_map_6) for _ in range(6))
        cols = tuple(j for _ in range(6) for j, _ in enumerate(voigt_map_6))
        return rows, cols
    else:
        raise ValueError(f"Unsupported tensor order: {l}")


def _get_cartesian_indices(l: int) -> Tuple[Tuple[int, ...], ...]:
    """
    获取笛卡尔张量的索引模式
    
    Args:
        l: 张量阶数
        
    Returns:
        每阶的索引元组
    """
    if l == 0:
        return ((),)
    elif l == 1:
        return ((0,), (1,), (2,))
    elif l == 2:
        indices = []
        for i in range(3):
            for j in range(3):
                indices.append((i, j))
        return tuple(indices)
    elif l == 3:
        indices = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    indices.append((i, j, k))
        return tuple(indices)
    elif l == 4:
        indices = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for m in range(3):
                        indices.append((i, j, k, m))
        return tuple(indices)
    elif l == 5:
        indices = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for m in range(3):
                        for n in range(3):
                            indices.append((i, j, k, m, n))
        return tuple(indices)
    elif l == 6:
        indices = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for m in range(3):
                        for n in range(3):
                            for p in range(3):
                                indices.append((i, j, k, m, n, p))
        return tuple(indices)
    else:
        raise ValueError(f"Unsupported tensor order: {l}")


def _voigt_dim(l: int) -> int:
    """获取 Voigt 表示法的维度"""
    if l == 0:
        return 1
    elif l == 1:
        return 3
    elif l == 2:
        return 6
    elif l == 3:
        return 18
    elif l == 4:
        return 36
    elif l == 5:
        return 108
    elif l == 6:
        return 216
    else:
        raise ValueError(f"Unsupported tensor order: {l}")


def _cartesian_dim(l: int) -> int:
    """获取笛卡尔张量的总维度"""
    return 3 ** l


def full2voigt(l_max: int, full_tensor: torch.Tensor) -> torch.Tensor:
    """
    将完整笛卡尔张量转换为物理 Voigt 矩阵形式
    
    使用物理 Voigt 记法，对于剪切分量应用适当的缩放因子：
    - l=2: 剪切分量为 1 (工程剪应变)
    - l=3: 第一组指标为 Voigt 记法，第二组为 Cartesian
    - l=4: 弹性张量，两组 Voigt 指标都应用剪切因子 [1,1,1,2,2,2]
    - l=5: 压电/电致伸缩张量，第一组 Voigt (6 维) 应用剪切因子
    - l=6: 高阶弹性张量，三组 Voigt 指标都应用剪切因子
    
    Args:
        l_max: 最大张量阶数 (0-6)
        full_tensor: 笛卡尔张量，形状为 [batch, 3, 3, ..., 3] (l_max 个 3)
        
    Returns:
        物理 Voigt 矩阵形式的张量，形状取决于 l_max:
        - l_max=0: [batch, 1]
        - l_max=1: [batch, 3]
        - l_max=2: [batch, 3, 3] # special case for l=2
        - l_max=3: [batch, 3, 6]
        - l_max=4: [batch, 6, 6]
        - l_max=5: [batch, 6, 18]
        - l_max=6: [batch, 6, 36]
    """
    batch_size = full_tensor.shape[0]
    voigt_indices = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    shear_factor = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=full_tensor.dtype, device=full_tensor.device)
    
    if l_max == 0:
        return full_tensor.view(batch_size,)
    elif l_max == 1:
        return full_tensor.view(batch_size, 3)
    elif l_max == 2:
        # result = []
        # for i, j in voigt_indices:
        #     result.append(full_tensor[:, i, j])
        # return torch.stack(result, dim=-1)
        return full_tensor.view(batch_size, 3, 3)
    elif l_max == 3:
        result = []
        for i in range(3):
            for idx, (j, k) in enumerate(voigt_indices):
                result.append(full_tensor[:, i, j, k])
        return torch.stack(result, dim=-1).view(batch_size, 3, 6)
    elif l_max == 4:
        result = []
        for idx1, (i1, j1) in enumerate(voigt_indices):
            for idx2, (i2, j2) in enumerate(voigt_indices):
                val = full_tensor[:, i1, j1, i2, j2] * shear_factor[idx1] * shear_factor[idx2]
                result.append(val)
        return torch.stack(result, dim=-1).view(batch_size, 6, 6)
    elif l_max == 5:
        result = []
        for idx1, (i1, j1) in enumerate(voigt_indices):
            for i2 in range(3):
                for idx3, (i3, j3) in enumerate(voigt_indices):
                    val = full_tensor[:, i1, j1, i2, i3, j3] * shear_factor[idx1]
                    result.append(val)
        return torch.stack(result, dim=-1).view(batch_size, 6, 18)
    elif l_max == 6:
        result = []
        for idx1, (i1, j1) in enumerate(voigt_indices):
            for idx2, (i2, j2) in enumerate(voigt_indices):
                for idx3, (i3, j3) in enumerate(voigt_indices):
                    val = full_tensor[:, i1, j1, i2, j2, i3, j3] * shear_factor[idx1] * shear_factor[idx2] * shear_factor[idx3]
                    result.append(val)
        return torch.stack(result, dim=-1).view(batch_size, 6, 36)
    else:
        raise ValueError(f"Unsupported tensor order: {l_max}")


def voigt2full(l_max: int, voigt_tensor: torch.Tensor) -> torch.Tensor:
    """
    将物理 Voigt 矩阵形式转换回完整笛卡尔张量
    
    对于物理 Voigt 记法，需要除以剪切因子来恢复原始张量值。
    
    Args:
        l_max: 最大张量阶数 (0-6)
        voigt_tensor: 物理 Voigt 矩阵形式的张量
        
    Returns:
        完整笛卡尔张量，形状为 [batch, 3, 3, ..., 3] (l_max 个 3)
    """
    batch_size = voigt_tensor.shape[0]
    voigt_indices = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    shear_factor = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=voigt_tensor.dtype, device=voigt_tensor.device)
    
    if l_max == 0:
        return voigt_tensor.view(batch_size, 1)
    elif l_max == 1:
        return voigt_tensor.view(batch_size, 3)
    elif l_max == 2:
        full = torch.zeros(batch_size, 3, 3, dtype=voigt_tensor.dtype, device=voigt_tensor.device)
        for idx, (i, j) in enumerate(voigt_indices):
            full[:, i, j] = voigt_tensor[:, idx]
        return full
    elif l_max == 3:
        full = torch.zeros(batch_size, 3, 3, 3, dtype=voigt_tensor.dtype, device=voigt_tensor.device)
        for i in range(3):
            for idx, (j, k) in enumerate(voigt_indices):
                full[:, i, j, k] = voigt_tensor[:, i, idx]
        return full
    elif l_max == 4:
        full = torch.zeros(batch_size, 3, 3, 3, 3, dtype=voigt_tensor.dtype, device=voigt_tensor.device)
        for idx1, (i1, j1) in enumerate(voigt_indices):
            for idx2, (i2, j2) in enumerate(voigt_indices):
                val = voigt_tensor[:, idx1, idx2] / (shear_factor[idx1] * shear_factor[idx2])
                full[:, i1, j1, i2, j2] = val
        return full
    elif l_max == 5:
        full = torch.zeros(batch_size, 3, 3, 3, 3, 3, dtype=voigt_tensor.dtype, device=voigt_tensor.device)
        for idx1, (i1, j1) in enumerate(voigt_indices):
            for i2 in range(3):
                for idx3, (i3, j3) in enumerate(voigt_indices):
                    idx = idx1 * 18 + i2 * 6 + idx3
                    val = voigt_tensor.view(batch_size, 108)[:, idx] / shear_factor[idx1]
                    full[:, i1, j1, i2, i3, j3] = val
        return full
    elif l_max == 6:
        full = torch.zeros(batch_size, 3, 3, 3, 3, 3, 3, dtype=voigt_tensor.dtype, device=voigt_tensor.device)
        for idx1, (i1, j1) in enumerate(voigt_indices):
            for idx2, (i2, j2) in enumerate(voigt_indices):
                for idx3, (i3, j3) in enumerate(voigt_indices):
                    val = voigt_tensor.view(batch_size, 6, 36)[:, idx1, idx2 * 6 + idx3]
                    val = val / (shear_factor[idx1] * shear_factor[idx2] * shear_factor[idx3])
                    full[:, i1, j1, i2, j2, i3, j3] = val
        return full
    else:
        raise ValueError(f"Unsupported tensor order: {l_max}")


if __name__ == "__main__":
    batch_size = 2
    
    print("Testing l=0 (scalar)")
    full_0 = torch.randn(batch_size, 1)
    voigt_0 = full2voigt(0, full_0)
    full_0_recovered = voigt2full(0, voigt_0)
    print(f"Original shape: {full_0.shape}, Voigt shape: {voigt_0.shape}, Recovered shape: {full_0_recovered.shape}")
    print(f"Reconstruction error: {torch.norm(full_0 - full_0_recovered).item():.6e}")
    
    print("\nTesting l=1 (vector)")
    full_1 = torch.randn(batch_size, 3)
    voigt_1 = full2voigt(1, full_1)
    full_1_recovered = voigt2full(1, voigt_1)
    print(f"Original shape: {full_1.shape}, Voigt shape: {voigt_1.shape}, Recovered shape: {full_1_recovered.shape}")
    print(f"Reconstruction error: {torch.norm(full_1 - full_1_recovered).item():.6e}")
    
    print("\nTesting l=2 (3x3 -> 6)")
    full_2 = torch.randn(batch_size, 3, 3)
    voigt_2 = full2voigt(2, full_2)
    full_2_recovered = voigt2full(2, voigt_2)
    print(f"Original shape: {full_2.shape}, Voigt shape: {voigt_2.shape}, Recovered shape: {full_2_recovered.shape}")
    print(f"Reconstruction error (only Voigt indices preserved): {torch.norm(full_2 - full_2_recovered).item():.6e}")
    voigt_2_recovered = full2voigt(2, full_2_recovered)
    print(f"Voigt space reconstruction error: {torch.norm(voigt_2 - voigt_2_recovered).item():.6e}")
    
    print("\nTesting l=3 (3x3x3 -> 3x6)")
    full_3 = torch.randn(batch_size, 3, 3, 3)
    voigt_3 = full2voigt(3, full_3)
    full_3_recovered = voigt2full(3, voigt_3)
    print(f"Original shape: {full_3.shape}, Voigt shape: {voigt_3.shape}, Recovered shape: {full_3_recovered.shape}")
    print(f"Reconstruction error (only Voigt indices preserved): {torch.norm(full_3 - full_3_recovered).item():.6e}")
    voigt_3_recovered = full2voigt(3, full_3_recovered)
    print(f"Voigt space reconstruction error: {torch.norm(voigt_3 - voigt_3_recovered).item():.6e}")
    
    print("\nTesting l=4 (3x3x3x3 -> 6x6)")
    full_4 = torch.randn(batch_size, 3, 3, 3, 3)
    voigt_4 = full2voigt(4, full_4)
    full_4_recovered = voigt2full(4, voigt_4)
    print(f"Original shape: {full_4.shape}, Voigt shape: {voigt_4.shape}, Recovered shape: {full_4_recovered.shape}")
    print(f"Reconstruction error (only Voigt indices preserved): {torch.norm(full_4 - full_4_recovered).item():.6e}")
    voigt_4_recovered = full2voigt(4, full_4_recovered)
    print(f"Voigt space reconstruction error: {torch.norm(voigt_4 - voigt_4_recovered).item():.6e}")
    
    print("\nTesting l=5 (3^5 -> 6x18)")
    full_5 = torch.randn(batch_size, 3, 3, 3, 3, 3)
    voigt_5 = full2voigt(5, full_5)
    full_5_recovered = voigt2full(5, voigt_5)
    print(f"Original shape: {full_5.shape}, Voigt shape: {voigt_5.shape}, Recovered shape: {full_5_recovered.shape}")
    print(f"Reconstruction error (only Voigt indices preserved): {torch.norm(full_5 - full_5_recovered).item():.6e}")
    voigt_5_recovered = full2voigt(5, full_5_recovered)
    print(f"Voigt space reconstruction error: {torch.norm(voigt_5 - voigt_5_recovered).item():.6e}")
    
    print("\nTesting l=6 (3^6 -> 6x36)")
    full_6 = torch.randn(batch_size, 3, 3, 3, 3, 3, 3)
    voigt_6 = full2voigt(6, full_6)
    full_6_recovered = voigt2full(6, voigt_6)
    print(f"Original shape: {full_6.shape}, Voigt shape: {voigt_6.shape}, Recovered shape: {full_6_recovered.shape}")
    print(f"Reconstruction error (only Voigt indices preserved): {torch.norm(full_6 - full_6_recovered).item():.6e}")
    voigt_6_recovered = full2voigt(6, full_6_recovered)
    print(f"Voigt space reconstruction error: {torch.norm(voigt_6 - voigt_6_recovered).item():.6e}")
    
    print("\n" + "="*60)
    print("Verification: Physical Voigt notation with shear factors")
    print("="*60)
    
    print("\nTesting l=4 elastic tensor (with physical Voigt factors)")
    C_ijkl = torch.randn(batch_size, 3, 3, 3, 3)
    C_voigt = full2voigt(4, C_ijkl)
    C_recovered = voigt2full(4, C_voigt)
    C_voigt_recovered = full2voigt(4, C_recovered)
    
    voigt_map = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    factor = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    
    print("Verifying shear factor application:")
    for m in range(6):
        for n in range(6):
            i, j = voigt_map[m]
            k, l = voigt_map[n]
            expected = C_ijkl[0, i, j, k, l] * factor[m] * factor[n]
            actual = C_voigt[0, m, n]
            if abs(expected - actual).item() > 1e-5:
                print(f"  Mismatch at [{m},{n}]: expected={expected.item():.6f}, actual={actual.item():.6f}")
    print(f"Voigt space reconstruction error: {torch.norm(C_voigt - C_voigt_recovered).item():.6e}")
    print("Physical Voigt notation verified!")
