from torch_geometric.loader import DataLoader
import torch

from .mp_dataset import MPDataset
# from .name_path import PathNameTuple
from . import name_path_dict

def get_mp_dataloader(
    cutoff: float,
    batch_size: int,
    pin_memory: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
    worker_init_fn = None,
) -> DataLoader:
    db_path = name_path_dict["mp"]
    dataset = MPDataset(db_path, cutoff)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        pin_memory=pin_memory, 
        num_workers=num_workers, 
        shuffle=shuffle,
        worker_init_fn=worker_init_fn,
    )