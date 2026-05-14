from .dataloaders.mp_loader import get_mp_dataloader
from .dataloaders.alexandria_dataloader import get_alexandria_dataloader
from .dataloaders.scalar_loader import (
    get_scalar_dataloader,
    get_scalar_dataloaders_split,
    # get_jarvis_formation_energy_dataloader,
    # get_jarvis_formation_energy_dataloaders_split,
    # get_jarvis_opt_band_gap_dataloader,
    # get_jarvis_opt_band_gap_dataloaders_split,
    # get_jarvis_total_energy_dataloader,
    # get_jarvis_total_energy_dataloaders_split,
    # get_jarvis_ehull_dataloader,
    # get_jarvis_ehull_dataloaders_split,
    # get_jarvis_mbj_bandgap_dataloader,
    # get_jarvis_mbj_bandgap_dataloaders_split,
    # get_jarvis_bandgap_dataloader,
    # get_jarvis_bandgap_dataloaders_split,
    # get_jarvis_e_form_dataloader,
    # get_jarvis_e_form_dataloaders_split,
    # get_jarvis_bulk_modulus_dataloader,
    # get_jarvis_bulk_modulus_dataloaders_split,
    # get_jarvis_shear_modulus_dataloader,
    # get_jarvis_shear_modulus_dataloaders_split,
)
from .dataloaders.tensor_loader import (
    get_tensor_dataloader,
    get_tensor_dataloaders_split,
    # get_dielectric_dataloader,
    # get_dielectric_dataloaders_split,
    # get_dielectric_ionic_dataloader,
    # get_dielectric_ionic_dataloaders_split,
    # get_piezoelectric_C_m2_dataloader,
    # get_piezoelectric_C_m2_dataloaders_split,
    # get_piezoelectric_e_Angst_dataloader,
    # get_piezoelectric_e_Angst_dataloaders_split,
    # get_elastic_sym_kbar_dataloader,
    # get_elastic_sym_kbar_dataloaders_split,
    # get_elastic_total_kbar_dataloader,
    # get_elastic_total_kbar_dataloaders_split,
)

import json
from pathlib import Path
CURRENT_DIR = Path(__file__).parent

def load_name_path_dict():
    """从 data/dataloader/name_path.json 加载数据"""
    # 构建 JSON 文件路径
    json_path = CURRENT_DIR / "dataloaders" / "name_path.json"
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

name_path_dict = load_name_path_dict()
assert isinstance(name_path_dict, dict), "name_path_dict must be a dict"

def load_readout_configs():
    """从 data/dataloader/lmax_symmetry.json 加载数据"""
    # 构建 JSON 文件路径
    json_path = CURRENT_DIR / "dataloaders" / "lmax_symmetry.json"
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

readout_configs = load_readout_configs()
assert isinstance(readout_configs, dict), "readout_configs must be a dict"

def load_scalar_properties():
    """从 data/scalar_properties.json 加载数据"""
    # 构建 JSON 文件路径
    json_path = CURRENT_DIR / "scalar_properties.json"
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

scalar_properties = load_scalar_properties()
assert isinstance(scalar_properties, list), "scalar_properties must be a list"

def load_tensor_properties():
    """从 data/tensor_properties.json 加载数据"""
    # 构建 JSON 文件路径
    json_path = CURRENT_DIR / "tensor_properties.json"
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

tensor_properties = load_tensor_properties()
assert isinstance(tensor_properties, list), "tensor_properties must be a list"
