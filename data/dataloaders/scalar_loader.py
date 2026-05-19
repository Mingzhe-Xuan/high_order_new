import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from .scalar_dataset import ScalarDataset
# from .name_path import PathNameTuple

def get_scalar_dataloader(
    path: str,
    property_name: str,
    cutoff: float,
    batch_size: int,
    pin_memory: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
    graph_mode: str = "high_order",
    max_neighbors: int = 12,
) -> DataLoader:
    dataset = ScalarDataset(path, property_name, cutoff, graph_mode, max_neighbors)
    return DataLoader(
        dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle
    )


def get_scalar_dataloaders_split(
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
    Creates train, validation, and test data loaders from a scalar dataset with specified ratios.
    
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
    
    # Create the full dataset
    dataset = ScalarDataset(path, property_name, cutoff, graph_mode, max_neighbors)
    
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


# def get_jarvis_formation_energy_dataloader(
#     cutoff: float,
#     batch_size: int,
#     pin_memory: bool = True,
#     num_workers: int = 0,
#     shuffle: bool = True,
# ) -> DataLoader:
#     formation_energy_path = PathNameTuple.formation_energy[0]
#     property_name = PathNameTuple.formation_energy[1]
#     dataset = ScalarDataset(formation_energy_path, property_name, cutoff)
#     return DataLoader(
#         dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle
#     )


# def get_jarvis_opt_band_gap_dataloader(
#     cutoff: float,
#     batch_size: int,
#     pin_memory: bool = True,
#     num_workers: int = 0,
#     shuffle: bool = True,
# ) -> DataLoader:
#     opt_bandgap_path = PathNameTuple.opt_bandgap[0]
#     property_name = PathNameTuple.opt_bandgap[1]
#     dataset = ScalarDataset(opt_bandgap_path, property_name, cutoff)
#     return DataLoader(
#         dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle
#     )

# def get_jarvis_total_energy_dataloader(
#     cutoff: float,
#     batch_size: int,
#     pin_memory: bool = True,
#     num_workers: int = 0,
#     shuffle: bool = True,
# ) -> DataLoader:
#     total_energy_path = PathNameTuple.total_energy[0]
#     property_name = PathNameTuple.total_energy[1]
#     dataset = ScalarDataset(total_energy_path, property_name, cutoff)
#     return DataLoader(
#         dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle
#     )

# def get_jarvis_ehull_dataloader(
#     cutoff: float,
#     batch_size: int,
#     pin_memory: bool = True,
#     num_workers: int = 0,
#     shuffle: bool = True,
# ) -> DataLoader:
#     ehull_path = PathNameTuple.ehull[0]
#     property_name = PathNameTuple.ehull[1]
#     dataset = ScalarDataset(ehull_path, property_name, cutoff)
#     return DataLoader(
#         dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle
#     )

# def get_jarvis_mbj_bandgap_dataloader(
#     cutoff: float,
#     batch_size: int,
#     pin_memory: bool = True,
#     num_workers: int = 0,
#     shuffle: bool = True,
# ) -> DataLoader:
#     mbj_bandgap_path = PathNameTuple.mbj_bandgap[0]
#     property_name = PathNameTuple.mbj_bandgap[1]
#     dataset = ScalarDataset(mbj_bandgap_path, property_name, cutoff)
#     return DataLoader(
#         dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle
#     )

# def get_jarvis_bandgap_dataloader(
#     cutoff: float,
#     batch_size: int,
#     pin_memory: bool = True,
#     num_workers: int = 0,
#     shuffle: bool = True,
# ) -> DataLoader:
#     bandgap_path = PathNameTuple.bandgap[0]
#     property_name = PathNameTuple.bandgap[1]
#     dataset = ScalarDataset(bandgap_path, property_name, cutoff)
#     return DataLoader(
#         dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle
#     )

# def get_jarvis_e_form_dataloader(
#     cutoff: float,
#     batch_size: int,
#     pin_memory: bool = True,
#     num_workers: int = 0,
#     shuffle: bool = True,
# ) -> DataLoader:
#     e_form_path = PathNameTuple.e_form[0]
#     property_name = PathNameTuple.e_form[1]
#     dataset = ScalarDataset(e_form_path, property_name, cutoff)
#     return DataLoader(
#         dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle
#     )

# def get_jarvis_bulk_modulus_dataloader(
#     cutoff: float,
#     batch_size: int,
#     pin_memory: bool = True,
#     num_workers: int = 0,
#     shuffle: bool = True,
# ) -> DataLoader:
#     bulk_modulus_path = PathNameTuple.bulk_modulus[0]
#     property_name = PathNameTuple.bulk_modulus[1]
#     dataset = ScalarDataset(bulk_modulus_path, property_name, cutoff)
#     return DataLoader(
#         dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle
#     )

# def get_jarvis_shear_modulus_dataloader(
#     cutoff: float,
#     batch_size: int,
#     pin_memory: bool = True,
#     num_workers: int = 0,
#     shuffle: bool = True,
# ) -> DataLoader:
#     shear_modulus_path = PathNameTuple.shear_modulus[0]
#     property_name = PathNameTuple.shear_modulus[1]
#     dataset = ScalarDataset(shear_modulus_path, property_name, cutoff)
#     return DataLoader(
#         dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle
#     )


# def get_jarvis_formation_energy_dataloaders_split(
#     cutoff: float,
#     train_val_test: tuple[float, float, float],
#     train_batch_size: int,
#     seed: int = 42,
#     pin_memory: bool = True,
#     num_workers: int = 0,
#     shuffle: bool = True,
#     val_batch_size: int = None,
#     test_batch_size: int = None,
# ) -> tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Creates train, validation, and test data loaders for Jarvis formation energy dataset with specified ratios.
    
#     Args:
#         cutoff: Cutoff distance for neighbor list construction
#         train_val_test: Tuple of ratios for (train, val, test) splits, should sum to 1.0
#         train_batch_size: Batch size for training data loader
#         seed: Random seed for reproducible splitting
#         pin_memory: Whether to pin memory in data loader
#         num_workers: Number of worker processes for data loading
#         shuffle: Whether to shuffle the data
#         val_batch_size: Batch size for validation data loader (defaults to train_batch_size if None)
#         test_batch_size: Batch size for test data loader (defaults to train_batch_size if None)
    
#     Returns:
#         tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
#     """
#     formation_energy_path = PathNameTuple.formation_energy[0]
#     property_name = PathNameTuple.formation_energy[1]
#     return get_scalar_dataloaders_split(
#         formation_energy_path,
#         property_name,
#         cutoff,
#         train_val_test,
#         train_batch_size,
#         seed,
#         pin_memory,
#         num_workers,
#         shuffle,
#         val_batch_size,
#         test_batch_size,
#     )


# def get_jarvis_opt_band_gap_dataloaders_split(
#     cutoff: float,
#     train_val_test: tuple[float, float, float],
#     train_batch_size: int,
#     seed: int = 42,
#     pin_memory: bool = True,
#     num_workers: int = 0,
#     shuffle: bool = True,
#     val_batch_size: int = None,
#     test_batch_size: int = None,
# ) -> tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Creates train, validation, and test data loaders for Jarvis opt band gap dataset with specified ratios.
    
#     Args:
#         cutoff: Cutoff distance for neighbor list construction
#         train_val_test: Tuple of ratios for (train, val, test) splits, should sum to 1.0
#         train_batch_size: Batch size for training data loader
#         seed: Random seed for reproducible splitting
#         pin_memory: Whether to pin memory in data loader
#         num_workers: Number of worker processes for data loading
#         shuffle: Whether to shuffle the data
#         val_batch_size: Batch size for validation data loader (defaults to train_batch_size if None)
#         test_batch_size: Batch size for test data loader (defaults to train_batch_size if None)
    
#     Returns:
#         tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
#     """
#     opt_bandgap_path = PathNameTuple.opt_bandgap[0]
#     property_name = PathNameTuple.opt_bandgap[1]
#     return get_scalar_dataloaders_split(
#         opt_bandgap_path,
#         property_name,
#         cutoff,
#         train_val_test,
#         train_batch_size,
#         seed,
#         pin_memory,
#         num_workers,
#         shuffle,
#         val_batch_size,
#         test_batch_size,
#     )


# def get_jarvis_total_energy_dataloaders_split(
#     cutoff: float,
#     train_val_test: tuple[float, float, float],
#     train_batch_size: int,
#     seed: int = 42,
#     pin_memory: bool = True,
#     num_workers: int = 0,
#     shuffle: bool = True,
#     val_batch_size: int = None,
#     test_batch_size: int = None,
# ) -> tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Creates train, validation, and test data loaders for Jarvis total energy dataset with specified ratios.
    
#     Args:
#         cutoff: Cutoff distance for neighbor list construction
#         train_val_test: Tuple of ratios for (train, val, test) splits, should sum to 1.0
#         train_batch_size: Batch size for training data loader
#         seed: Random seed for reproducible splitting
#         pin_memory: Whether to pin memory in data loader
#         num_workers: Number of worker processes for data loading
#         shuffle: Whether to shuffle the data
#         val_batch_size: Batch size for validation data loader (defaults to train_batch_size if None)
#         test_batch_size: Batch size for test data loader (defaults to train_batch_size if None)
    
#     Returns:
#         tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
#     """
#     total_energy_path = PathNameTuple.total_energy[0]
#     property_name = PathNameTuple.total_energy[1]
#     return get_scalar_dataloaders_split(
#         total_energy_path,
#         property_name,
#         cutoff,
#         train_val_test,
#         train_batch_size,
#         seed,
#         pin_memory,
#         num_workers,
#         shuffle,
#         val_batch_size,
#         test_batch_size,
#     )


# def get_jarvis_ehull_dataloaders_split(
#     cutoff: float,
#     train_val_test: tuple[float, float, float],
#     train_batch_size: int,
#     seed: int = 42,
#     pin_memory: bool = True,
#     num_workers: int = 0,
#     shuffle: bool = True,
#     val_batch_size: int = None,
#     test_batch_size: int = None,
# ) -> tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Creates train, validation, and test data loaders for Jarvis ehull dataset with specified ratios.
    
#     Args:
#         cutoff: Cutoff distance for neighbor list construction
#         train_val_test: Tuple of ratios for (train, val, test) splits, should sum to 1.0
#         train_batch_size: Batch size for training data loader
#         seed: Random seed for reproducible splitting
#         pin_memory: Whether to pin memory in data loader
#         num_workers: Number of worker processes for data loading
#         shuffle: Whether to shuffle the data
#         val_batch_size: Batch size for validation data loader (defaults to train_batch_size if None)
#         test_batch_size: Batch size for test data loader (defaults to train_batch_size if None)
    
#     Returns:
#         tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
#     """
#     ehull_path = PathNameTuple.ehull[0]
#     property_name = PathNameTuple.ehull[1]
#     return get_scalar_dataloaders_split(
#         ehull_path,
#         property_name,
#         cutoff,
#         train_val_test,
#         train_batch_size,
#         seed,
#         pin_memory,
#         num_workers,
#         shuffle,
#         val_batch_size,
#         test_batch_size,
#     )


# def get_jarvis_mbj_bandgap_dataloaders_split(
#     cutoff: float,
#     train_val_test: tuple[float, float, float],
#     train_batch_size: int,
#     seed: int = 42,
#     pin_memory: bool = True,
#     num_workers: int = 0,
#     shuffle: bool = True,
#     val_batch_size: int = None,
#     test_batch_size: int = None,
# ) -> tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Creates train, validation, and test data loaders for Jarvis MBJ bandgap dataset with specified ratios.
    
#     Args:
#         cutoff: Cutoff distance for neighbor list construction
#         train_val_test: Tuple of ratios for (train, val, test) splits, should sum to 1.0
#         train_batch_size: Batch size for training data loader
#         seed: Random seed for reproducible splitting
#         pin_memory: Whether to pin memory in data loader
#         num_workers: Number of worker processes for data loading
#         shuffle: Whether to shuffle the data
#         val_batch_size: Batch size for validation data loader (defaults to train_batch_size if None)
#         test_batch_size: Batch size for test data loader (defaults to train_batch_size if None)
    
#     Returns:
#         tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
#     """
#     mbj_bandgap_path = PathNameTuple.mbj_bandgap[0]
#     property_name = PathNameTuple.mbj_bandgap[1]
#     return get_scalar_dataloaders_split(
#         mbj_bandgap_path,
#         property_name,
#         cutoff,
#         train_val_test,
#         train_batch_size,
#         seed,
#         pin_memory,
#         num_workers,
#         shuffle,
#         val_batch_size,
#         test_batch_size,
#     )


# def get_jarvis_bandgap_dataloaders_split(
#     cutoff: float,
#     train_val_test: tuple[float, float, float],
#     train_batch_size: int,
#     seed: int = 42,
#     pin_memory: bool = True,
#     num_workers: int = 0,
#     shuffle: bool = True,
#     val_batch_size: int = None,
#     test_batch_size: int = None,
# ) -> tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Creates train, validation, and test data loaders for Jarvis bandgap dataset with specified ratios.
    
#     Args:
#         cutoff: Cutoff distance for neighbor list construction
#         train_val_test: Tuple of ratios for (train, val, test) splits, should sum to 1.0
#         train_batch_size: Batch size for training data loader
#         seed: Random seed for reproducible splitting
#         pin_memory: Whether to pin memory in data loader
#         num_workers: Number of worker processes for data loading
#         shuffle: Whether to shuffle the data
#         val_batch_size: Batch size for validation data loader (defaults to train_batch_size if None)
#         test_batch_size: Batch size for test data loader (defaults to train_batch_size if None)
    
#     Returns:
#         tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
#     """
#     bandgap_path = PathNameTuple.bandgap[0]
#     property_name = PathNameTuple.bandgap[1]
#     return get_scalar_dataloaders_split(
#         bandgap_path,
#         property_name,
#         cutoff,
#         train_val_test,
#         train_batch_size,
#         seed,
#         pin_memory,
#         num_workers,
#         shuffle,
#         val_batch_size,
#         test_batch_size,
#     )


# def get_jarvis_e_form_dataloaders_split(
#     cutoff: float,
#     train_val_test: tuple[float, float, float],
#     train_batch_size: int,
#     seed: int = 42,
#     pin_memory: bool = True,
#     num_workers: int = 0,
#     shuffle: bool = True,
#     val_batch_size: int = None,
#     test_batch_size: int = None,
# ) -> tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Creates train, validation, and test data loaders for Jarvis e_form dataset with specified ratios.
    
#     Args:
#         cutoff: Cutoff distance for neighbor list construction
#         train_val_test: Tuple of ratios for (train, val, test) splits, should sum to 1.0
#         train_batch_size: Batch size for training data loader
#         seed: Random seed for reproducible splitting
#         pin_memory: Whether to pin memory in data loader
#         num_workers: Number of worker processes for data loading
#         shuffle: Whether to shuffle the data
#         val_batch_size: Batch size for validation data loader (defaults to train_batch_size if None)
#         test_batch_size: Batch size for test data loader (defaults to train_batch_size if None)
    
#     Returns:
#         tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
#     """
#     e_form_path = PathNameTuple.e_form[0]
#     property_name = PathNameTuple.e_form[1]
#     return get_scalar_dataloaders_split(
#         e_form_path,
#         property_name,
#         cutoff,
#         train_val_test,
#         train_batch_size,
#         seed,
#         pin_memory,
#         num_workers,
#         shuffle,
#         val_batch_size,
#         test_batch_size,
#     )


# def get_jarvis_bulk_modulus_dataloaders_split(
#     cutoff: float,
#     train_val_test: tuple[float, float, float],
#     train_batch_size: int,
#     seed: int = 42,
#     pin_memory: bool = True,
#     num_workers: int = 0,
#     shuffle: bool = True,
#     val_batch_size: int = None,
#     test_batch_size: int = None,
# ) -> tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Creates train, validation, and test data loaders for Jarvis bulk modulus dataset with specified ratios.
    
#     Args:
#         cutoff: Cutoff distance for neighbor list construction
#         train_val_test: Tuple of ratios for (train, val, test) splits, should sum to 1.0
#         train_batch_size: Batch size for training data loader
#         seed: Random seed for reproducible splitting
#         pin_memory: Whether to pin memory in data loader
#         num_workers: Number of worker processes for data loading
#         shuffle: Whether to shuffle the data
#         val_batch_size: Batch size for validation data loader (defaults to train_batch_size if None)
#         test_batch_size: Batch size for test data loader (defaults to train_batch_size if None)
    
#     Returns:
#         tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
#     """
#     bulk_modulus_path = PathNameTuple.bulk_modulus[0]
#     property_name = PathNameTuple.bulk_modulus[1]
#     return get_scalar_dataloaders_split(
#         bulk_modulus_path,
#         property_name,
#         cutoff,
#         train_val_test,
#         train_batch_size,
#         seed,
#         pin_memory,
#         num_workers,
#         shuffle,
#         val_batch_size,
#         test_batch_size,
#     )


# def get_jarvis_shear_modulus_dataloaders_split(
#     cutoff: float,
#     train_val_test: tuple[float, float, float],
#     train_batch_size: int,
#     seed: int = 42,
#     pin_memory: bool = True,
#     num_workers: int = 0,
#     shuffle: bool = True,
#     val_batch_size: int = None,
#     test_batch_size: int = None,
# ) -> tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Creates train, validation, and test data loaders for Jarvis shear modulus dataset with specified ratios.
    
#     Args:
#         cutoff: Cutoff distance for neighbor list construction
#         train_val_test: Tuple of ratios for (train, val, test) splits, should sum to 1.0
#         train_batch_size: Batch size for training data loader
#         seed: Random seed for reproducible splitting
#         pin_memory: Whether to pin memory in data loader
#         num_workers: Number of worker processes for data loading
#         shuffle: Whether to shuffle the data
#         val_batch_size: Batch size for validation data loader (defaults to train_batch_size if None)
#         test_batch_size: Batch size for test data loader (defaults to train_batch_size if None)
    
#     Returns:
#         tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
#     """
#     shear_modulus_path = PathNameTuple.shear_modulus[0]
#     property_name = PathNameTuple.shear_modulus[1]
#     return get_scalar_dataloaders_split(
#         shear_modulus_path,
#         property_name,
#         cutoff,
#         train_val_test,
#         train_batch_size,
#         seed,
#         pin_memory,
#         num_workers,
#         shuffle,
#         val_batch_size,
#         test_batch_size,
#     )
