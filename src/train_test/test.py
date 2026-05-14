import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from . import scalar_test, tensor_test


def test(
    scalar_models: dict[str, nn.Module],
    tensor_models: dict[str, nn.Module],
    scalar_dataloaders: dict[str, DataLoader],
    tensor_dataloaders: dict[str, DataLoader],
    pic_dir: str,
    metric_dir: str,
    scalar_train_history: dict = None,
    tensor_train_history: dict = None,
):
    """
    Test scalar and tensor property prediction models.
    
    Args:
        scalar_models: Dictionary of trained scalar models
        tensor_models: Dictionary of trained tensor models
        scalar_dataloaders: Dictionary of scalar dataloaders
        tensor_dataloaders: Dictionary of tensor dataloaders
        pic_dir: Directory to save plots
        metric_dir: Directory to save metrics
        scalar_train_history: Training history for scalar models
        tensor_train_history: Training history for tensor models
        
    Returns:
        tuple: (scalar_results, tensor_results)
    """
    scalar_results = None
    tensor_results = None
    
    if scalar_models is not None:
        scalar_results = scalar_test(
            scalar_models=scalar_models,
            scalar_dataloaders=scalar_dataloaders,
            pic_dir=pic_dir,
            metric_dir=metric_dir,
            train_history=scalar_train_history,
        )
    if tensor_models is not None:
        tensor_results = tensor_test(
            tensor_models=tensor_models,
            tensor_dataloaders=tensor_dataloaders,
            pic_dir=pic_dir,
            metric_dir=metric_dir,
            train_history=tensor_train_history,
        )
        
    return scalar_results, tensor_results
