import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch

from src.train_test.utils.visualization import get_session_timestamp


def get_checkpoint_dir(checkpoint_base_dir: str, create_if_not_exists: bool = True) -> str:
    """
    Get the checkpoint directory with timestamp subfolder.
    Uses the same session timestamp as visualization for consistency.
    
    Args:
        checkpoint_base_dir: Base directory for checkpoints (e.g., "checkpoints")
        create_if_not_exists: Whether to create the directory if it doesn't exist
        
    Returns:
        Full path to the timestamped checkpoint directory
    """
    timestamp = get_session_timestamp()
    checkpoint_dir = os.path.join(checkpoint_base_dir, timestamp)
    if create_if_not_exists:
        os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def save_params_json(params: Dict[str, Any], checkpoint_base_dir: str) -> str:
    """
    Save parameters to a JSON file in the checkpoint directory.
    
    Args:
        params: Dictionary of parameters to save
        checkpoint_base_dir: Base directory for checkpoints
        
    Returns:
        Path to the saved JSON file
    """
    checkpoint_dir = get_checkpoint_dir(checkpoint_base_dir)
    params_path = os.path.join(checkpoint_dir, "params.json")
    
    serializable_params = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            serializable_params[key] = value
        elif isinstance(value, (list, tuple)):
            serializable_params[key] = list(value)
        elif isinstance(value, dict):
            serializable_params[key] = dict(value)
        else:
            serializable_params[key] = str(value)
    
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_params, f, indent=4, ensure_ascii=False)
    
    return params_path


def generate_checkpoint_filename(
    property_name: str,
    num_epochs: int,
    epoch: Optional[int] = None,
    is_best: bool = False,
) -> str:
    """
    Generate a checkpoint filename with property name and num_epochs.
    
    Args:
        property_name: Name of the property being trained
        num_epochs: Total number of training epochs
        epoch: Current epoch number (for intermediate checkpoints)
        is_best: Whether this is the best model checkpoint
        
    Returns:
        Generated checkpoint filename
    """
    if is_best:
        return f"{property_name}_epochs{num_epochs}_best.pth"
    elif epoch is not None:
        return f"{property_name}_epochs{num_epochs}_epoch{epoch+1}.pth"
    else:
        return f"{property_name}_epochs{num_epochs}_final.pth"


def save_checkpoint(
    checkpoint_data: Dict[str, Any],
    checkpoint_base_dir: str,
    property_name: str,
    num_epochs: int,
    epoch: Optional[int] = None,
    is_best: bool = False,
) -> str:
    """
    Save a checkpoint file with proper naming convention.
    
    Args:
        checkpoint_data: Dictionary containing model state and training info
        checkpoint_base_dir: Base directory for checkpoints
        property_name: Name of the property being trained
        num_epochs: Total number of training epochs
        epoch: Current epoch number (for intermediate checkpoints)
        is_best: Whether this is the best model checkpoint
        
    Returns:
        Path to the saved checkpoint file
    """
    checkpoint_dir = get_checkpoint_dir(checkpoint_base_dir)
    property_dir = os.path.join(checkpoint_dir, property_name)
    os.makedirs(property_dir, exist_ok=True)
    
    filename = generate_checkpoint_filename(
        property_name=property_name,
        num_epochs=num_epochs,
        epoch=epoch,
        is_best=is_best,
    )
    
    checkpoint_path = os.path.join(property_dir, filename)
    torch.save(checkpoint_data, checkpoint_path)
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint to
        
    Returns:
        Dictionary containing the checkpoint data
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.load(checkpoint_path, map_location=device)
