from torch_geometric.loader import DataLoader
import torch

from .alexandria_dataset import AlexandriaDataset
# from .name_path import PathNameTuple
from . import name_path_dict

def get_alexandria_dataloader(
    cutoff: float,
    batch_size: int,
    pin_memory: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
    worker_init_fn = None,
) -> DataLoader:
    db_path = name_path_dict["alexandria"]
    dataset = AlexandriaDataset(db_path, cutoff)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        pin_memory=pin_memory, 
        num_workers=num_workers, 
        shuffle=shuffle,
        worker_init_fn=worker_init_fn,
    )