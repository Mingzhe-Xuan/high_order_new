import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from .tensor_dataset import TensorDataset


def get_tensor_dataloader(
    path, property_name, cutoff, batch_size, pin_memory=True, num_workers=0, shuffle=True, graph_mode="high_order", max_neighbors=12
):
    if "dielectric" in property_name:
        l_max = 2
    elif "piezoelectric" in property_name:
        l_max = 3
    elif "elastic" in property_name:
        l_max = 4
    else:
        raise NotImplementedError("property_name not supported")

    dataset = TensorDataset(path, property_name, l_max, cutoff, graph_mode=graph_mode, max_neighbors=max_neighbors)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    return dataloader


def get_tensor_dataloaders_split(
    path: str,
    property_name: str,
    cutoff: float,
    train_val_test: tuple[float, float, float],
    train_batch_size: int,
    seed: int = 42,
    pin_memory: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
    val_batch_size: int = None,
    test_batch_size: int = None,
    worker_init_fn = None,
    graph_mode: str = "high_order",
    max_neighbors: int = 12,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test data loaders from a tensor dataset with specified ratios.
    
    Args:
        path: Path to the pickle file containing the dataset
        property_name: Name of the property to predict
        cutoff: Cutoff distance for neighbor list construction
        train_val_test: Tuple of ratios for (train, val, test) splits, should sum to 1.0
        train_batch_size: Batch size for training data loader
        seed: Random seed for reproducible splitting
        pin_memory: Whether to pin memory in data loader
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
        val_batch_size: Batch size for validation data loader (defaults to train_batch_size if None)
        test_batch_size: Batch size for test data loader (defaults to train_batch_size if None)
        worker_init_fn: Worker initialization function for reproducibility with multiple workers
    
    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
    """
    if val_batch_size is None:
        val_batch_size = train_batch_size
    if test_batch_size is None:
        test_batch_size = train_batch_size
    
    # Set seed for reproducible splitting
    torch.manual_seed(seed)
    
    # Determine l_max based on property name
    if "dielectric" in property_name:
        l_max = 2
    elif "piezoelectric" in property_name:
        l_max = 3
    elif "elastic" in property_name:
        l_max = 4
    else:
        raise NotImplementedError("property_name not supported")
    
    # Create the full dataset
    dataset = TensorDataset(path, property_name, l_max, cutoff, graph_mode=graph_mode, max_neighbors=max_neighbors)
    
    # Validate the ratios
    total_ratio = sum(train_val_test)
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Train, validation, and test ratios must sum to 1.0, got {total_ratio}")
    
    if len(train_val_test) != 3:
        raise ValueError(f"Expected 3 values for train_val_test, got {len(train_val_test)}")
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_val_test[0] * total_size)
    val_size = int(train_val_test[1] * total_size)
    test_size = total_size - train_size - val_size  # Remaining samples go to test
    
    # Adjust for rounding errors
    if train_size + val_size + test_size != total_size:
        diff = total_size - (train_size + val_size + test_size)
        test_size += diff
    
    print(f"Dataset split - Total: {total_size}, Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)  # For reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed) if num_workers == 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,  # Usually don't shuffle validation data
        pin_memory=pin_memory,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,  # Usually don't shuffle test data
        pin_memory=pin_memory,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )
    
    return train_loader, val_loader, test_loader

def get_dielectric_dataloader(
    path: str,
    cutoff: float,
    batch_size: int,
    pin_memory: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    return get_tensor_dataloader(
        path, "dielectric", cutoff, batch_size, pin_memory, num_workers, shuffle
    )

def get_dielectric_ionic_dataloader(
    path: str,
    cutoff: float,
    batch_size: int,
    pin_memory: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    return get_tensor_dataloader(
        path, "dielectric_ionic", cutoff, batch_size, pin_memory, num_workers, shuffle
    )

def get_piezoelectric_C_m2_dataloader(
    path: str,
    cutoff: float,
    batch_size: int,
    pin_memory: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    return get_tensor_dataloader(
        path, "piezoelectric_C_m2", cutoff, batch_size, pin_memory, num_workers, shuffle
    )

def get_piezoelectric_e_Angst_dataloader(
    path: str,
    cutoff: float,
    batch_size: int,
    pin_memory: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    return get_tensor_dataloader(
        path, "piezoelectric_e_Angst", cutoff, batch_size, pin_memory, num_workers, shuffle
    )

def get_elastic_sym_kbar_dataloader(
    path: str,
    cutoff: float,
    batch_size: int,
    pin_memory: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    return get_tensor_dataloader(
        path, "elastic_sym_kbar", cutoff, batch_size, pin_memory, num_workers, shuffle
    )

def get_elastic_total_kbar_dataloader(
    path: str,
    cutoff: float,
    batch_size: int,
    pin_memory: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    return get_tensor_dataloader(
        path, "elastic_total_kbar", cutoff, batch_size, pin_memory, num_workers, shuffle
    )


def get_dielectric_dataloaders_split(
    path: str,
    cutoff: float,
    train_val_test: tuple[float, float, float],
    train_batch_size: int,
    seed: int = 42,
    pin_memory: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
    val_batch_size: int = None,
    test_batch_size: int = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test data loaders for dielectric dataset with specified ratios.
    
    Args:
        path: Path to the pickle file containing the dataset
        cutoff: Cutoff distance for neighbor list construction
        train_val_test: Tuple of ratios for (train, val, test) splits, should sum to 1.0
        train_batch_size: Batch size for training data loader
        seed: Random seed for reproducible splitting
        pin_memory: Whether to pin memory in data loader
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
        val_batch_size: Batch size for validation data loader (defaults to train_batch_size if None)
        test_batch_size: Batch size for test data loader (defaults to train_batch_size if None)
    
    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
    """
    return get_tensor_dataloaders_split(
        path,
        "dielectric",
        cutoff,
        train_val_test,
        train_batch_size,
        seed,
        pin_memory,
        num_workers,
        shuffle,
        val_batch_size,
        test_batch_size,
    )


def get_dielectric_ionic_dataloaders_split(
    path: str,
    cutoff: float,
    train_val_test: tuple[float, float, float],
    train_batch_size: int,
    seed: int = 42,
    pin_memory: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
    val_batch_size: int = None,
    test_batch_size: int = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test data loaders for dielectric ionic dataset with specified ratios.
    
    Args:
        path: Path to the pickle file containing the dataset
        cutoff: Cutoff distance for neighbor list construction
        train_val_test: Tuple of ratios for (train, val, test) splits, should sum to 1.0
        train_batch_size: Batch size for training data loader
        seed: Random seed for reproducible splitting
        pin_memory: Whether to pin memory in data loader
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
        val_batch_size: Batch size for validation data loader (defaults to train_batch_size if None)
        test_batch_size: Batch size for test data loader (defaults to train_batch_size if None)
    
    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
    """
    return get_tensor_dataloaders_split(
        path,
        "dielectric_ionic",
        cutoff,
        train_val_test,
        train_batch_size,
        seed,
        pin_memory,
        num_workers,
        shuffle,
        val_batch_size,
        test_batch_size,
    )


def get_piezoelectric_C_m2_dataloaders_split(
    path: str,
    cutoff: float,
    train_val_test: tuple[float, float, float],
    train_batch_size: int,
    seed: int = 42,
    pin_memory: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
    val_batch_size: int = None,
    test_batch_size: int = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test data loaders for piezoelectric C_m2 dataset with specified ratios.
    
    Args:
        path: Path to the pickle file containing the dataset
        cutoff: Cutoff distance for neighbor list construction
        train_val_test: Tuple of ratios for (train, val, test) splits, should sum to 1.0
        train_batch_size: Batch size for training data loader
        seed: Random seed for reproducible splitting
        pin_memory: Whether to pin memory in data loader
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
        val_batch_size: Batch size for validation data loader (defaults to train_batch_size if None)
        test_batch_size: Batch size for test data loader (defaults to train_batch_size if None)
    
    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
    """
    return get_tensor_dataloaders_split(
        path,
        "piezoelectric_C_m2",
        cutoff,
        train_val_test,
        train_batch_size,
        seed,
        pin_memory,
        num_workers,
        shuffle,
        val_batch_size,
        test_batch_size,
    )


def get_piezoelectric_e_Angst_dataloaders_split(
    path: str,
    cutoff: float,
    train_val_test: tuple[float, float, float],
    train_batch_size: int,
    seed: int = 42,
    pin_memory: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
    val_batch_size: int = None,
    test_batch_size: int = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test data loaders for piezoelectric e_Angst dataset with specified ratios.
    
    Args:
        path: Path to the pickle file containing the dataset
        cutoff: Cutoff distance for neighbor list construction
        train_val_test: Tuple of ratios for (train, val, test) splits, should sum to 1.0
        train_batch_size: Batch size for training data loader
        seed: Random seed for reproducible splitting
        pin_memory: Whether to pin memory in data loader
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
        val_batch_size: Batch size for validation data loader (defaults to train_batch_size if None)
        test_batch_size: Batch size for test data loader (defaults to train_batch_size if None)
    
    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
    """
    return get_tensor_dataloaders_split(
        path,
        "piezoelectric_e_Angst",
        cutoff,
        train_val_test,
        train_batch_size,
        seed,
        pin_memory,
        num_workers,
        shuffle,
        val_batch_size,
        test_batch_size,
    )


def get_elastic_sym_kbar_dataloaders_split(
    path: str,
    cutoff: float,
    train_val_test: tuple[float, float, float],
    train_batch_size: int,
    seed: int = 42,
    pin_memory: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
    val_batch_size: int = None,
    test_batch_size: int = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test data loaders for elastic sym kbar dataset with specified ratios.
    
    Args:
        path: Path to the pickle file containing the dataset
        cutoff: Cutoff distance for neighbor list construction
        train_val_test: Tuple of ratios for (train, val, test) splits, should sum to 1.0
        train_batch_size: Batch size for training data loader
        seed: Random seed for reproducible splitting
        pin_memory: Whether to pin memory in data loader
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
        val_batch_size: Batch size for validation data loader (defaults to train_batch_size if None)
        test_batch_size: Batch size for test data loader (defaults to train_batch_size if None)
    
    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
    """
    return get_tensor_dataloaders_split(
        path,
        "elastic_sym_kbar",
        cutoff,
        train_val_test,
        train_batch_size,
        seed,
        pin_memory,
        num_workers,
        shuffle,
        val_batch_size,
        test_batch_size,
    )


def get_elastic_total_kbar_dataloaders_split(
    path: str,
    cutoff: float,
    train_val_test: tuple[float, float, float],
    train_batch_size: int,
    seed: int = 42,
    pin_memory: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
    val_batch_size: int = None,
    test_batch_size: int = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test data loaders for elastic total kbar dataset with specified ratios.
    
    Args:
        path: Path to the pickle file containing the dataset
        cutoff: Cutoff distance for neighbor list construction
        train_val_test: Tuple of ratios for (train, val, test) splits, should sum to 1.0
        train_batch_size: Batch size for training data loader
        seed: Random seed for reproducible splitting
        pin_memory: Whether to pin memory in data loader
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
        val_batch_size: Batch size for validation data loader (defaults to train_batch_size if None)
        test_batch_size: Batch size for test data loader (defaults to train_batch_size if None)
    
    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
    """
    return get_tensor_dataloaders_split(
        path,
        "elastic_total_kbar",
        cutoff,
        train_val_test,
        train_batch_size,
        seed,
        pin_memory,
        num_workers,
        shuffle,
        val_batch_size,
        test_batch_size,
    )