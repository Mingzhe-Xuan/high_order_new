import torch
from e3nn.o3 import Irreps

def get_irreps_from_ns_nv_lmax(ns: int, nv: int, l_max: int) -> Irreps:
    irreps = f"{ns}x0e"
    if l_max == 0:
        return Irreps(irreps)
    for l in range(1, l_max + 1):
        p = "e" if l % 2 == 0 else "o"
        irreps += f"+{nv}x{l}{p}"
    return Irreps(irreps)