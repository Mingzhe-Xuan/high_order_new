import warnings
import torch
import torch.nn as nn
import json
import os
import numpy as np
import random
from typing import Union
from e3nn.o3 import Irreps
from torch_geometric.loader import DataLoader
import argparse


def seed_everything(seed: int):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed} for full reproducibility")

def worker_init_fn(worker_id: int, seed: int):
    """
    Initialize worker process with specific seed for reproducibility.
    
    Args:
        worker_id: Worker ID
        seed: Base seed value
    """
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)
        torch.cuda.manual_seed_all(worker_seed)


from data import (
    get_mp_dataloader,
    get_alexandria_dataloader,
    get_scalar_dataloaders_split,
    get_tensor_dataloaders_split,
    scalar_properties,
    tensor_properties,
    name_path_dict,
    readout_configs,
)
from src.train_test import (
    train,
    test,
)
from src.train_test.utils.save_metrics import save_results_to_markdown
from src.train_test.utils.checkpoint import save_params_json


def _create_scalar_dataloaders(
    name_path_dict,
    scalar_properties,
    cutoff,
    train_val_test,
    seed,
    train_batch_size,
    val_batch_size,
    test_batch_size,
    pin_memory,
    num_workers,
    worker_init_fn=None,
    graph_mode="high_order",
    max_neighbors=12,
):
    """Create dataloaders for scalar properties."""
    dataloaders = {}
    for prop in scalar_properties:
        trainset, valset, testset = get_scalar_dataloaders_split(
            path=name_path_dict[prop],
            property_name=prop,
            cutoff=cutoff,
            train_val_test=train_val_test,
            seed=seed,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=True,
            worker_init_fn=worker_init_fn,
            graph_mode=graph_mode,
            max_neighbors=max_neighbors,
        )
        dataloaders[f"{prop}_trainset"] = trainset
        dataloaders[f"{prop}_valset"] = valset
        dataloaders[f"{prop}_testset"] = testset
    return dataloaders


def _create_tensor_dataloaders(
    name_path_dict,
    tensor_properties,
    cutoff,
    train_val_test,
    seed,
    train_batch_size,
    val_batch_size,
    test_batch_size,
    pin_memory,
    num_workers,
    graph_mode,
    max_neighbors,
    worker_init_fn=None,
):
    """Create dataloaders for tensor properties."""
    dataloaders = {}
    for prop in tensor_properties:
        trainset, valset, testset = get_tensor_dataloaders_split(
            path=name_path_dict[prop],
            property_name=prop,
            cutoff=cutoff,
            train_val_test=train_val_test,
            seed=seed,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=True,
            worker_init_fn=worker_init_fn,
            graph_mode=graph_mode,
            max_neighbors=max_neighbors,
        )
        dataloaders[f"{prop}_trainset"] = trainset
        dataloaders[f"{prop}_valset"] = valset
        dataloaders[f"{prop}_testset"] = testset
    return dataloaders


def _get_params_to_save(
    *,
    model_type,
    cutoff,
    train_batch_size,
    val_batch_size,
    test_batch_size,
    pin_memory,
    num_workers,
    seed,
    train_val_test,
    dist_emb_func,
    embed_dim,
    max_atom_type,
    inv_update_method,
    num_inv_layers,
    middle_scalar_hidden_dim,
    num_middle_hidden_layers,
    equi_update_method,
    num_equi_layers,
    tp_method,
    scalar_dim,
    vec_dim,
    num_final_hidden_layers,
    final_scalar_hidden_dim,
    final_vec_hidden_dim,
    final_scalar_out_dim,
    final_vec_out_dim,
    need_self_train,
    need_scalar_train,
    need_tensor_train,
    freeze,
    final_pooling,
    self_num_epochs,
    scalar_num_epochs,
    tensor_num_epochs,
    lr,
    weight_decay,
    clip_grad_norm,
    save_interval,
    optimizer,
    scheduler,
    warmup_periods,
    self_loss_func,
    scalar_loss_func,
    tensor_loss_func,
    self_train_limit,
    scalar_train_limit,
    tensor_train_limit,
    start_epoch,
    gmtnet_embed_dim,
    gmtnet_num_attention_layers,
    graph_mode,
    max_neighbors,
    use_tensorboard,
):
    if model_type == "gmtnet":
        return {
            "model": "gmtnet",
            "epochs": tensor_num_epochs,
            "batch_size": train_batch_size,
            "val_batch_size": val_batch_size,
            "test_batch_size": test_batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "loss": tensor_loss_func,
            "split_seed": seed,
            "train_ratio": train_val_test[0],
            "val_ratio": train_val_test[1],
            "test_ratio": train_val_test[2],
            "cutoff": cutoff,
            "load_model": False,
            "reduce_cell": False,
            "use_mask": True,
            "gmtnet_embed_dim": gmtnet_embed_dim,
            "gmtnet_num_attention_layers": gmtnet_num_attention_layers,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "clip_grad_norm": clip_grad_norm,
            "save_interval": save_interval,
            "start_epoch": start_epoch,
            "tensor_train_limit": tensor_train_limit,
            "pin_memory": pin_memory,
            "num_workers": num_workers,
            "graph_mode": graph_mode,
            "max_neighbors": max_neighbors,
            "use_tensorboard": use_tensorboard,
        }

    return {
        "cutoff": cutoff,
        "train_batch_size": train_batch_size,
        "val_batch_size": val_batch_size,
        "test_batch_size": test_batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
        "seed": seed,
        "train_val_test": train_val_test,
        "dist_emb_func": dist_emb_func,
        "embed_dim": embed_dim,
        "max_atom_type": max_atom_type,
        "inv_update_method": inv_update_method,
        "num_inv_layers": num_inv_layers,
        "middle_scalar_hidden_dim": middle_scalar_hidden_dim,
        "num_middle_hidden_layers": num_middle_hidden_layers,
        "equi_update_method": equi_update_method,
        "num_equi_layers": num_equi_layers,
        "tp_method": tp_method,
        "scalar_dim": scalar_dim,
        "vec_dim": vec_dim,
        "num_final_hidden_layers": num_final_hidden_layers,
        "final_scalar_hidden_dim": final_scalar_hidden_dim,
        "final_vec_hidden_dim": final_vec_hidden_dim,
        "final_scalar_out_dim": final_scalar_out_dim,
        "final_vec_out_dim": final_vec_out_dim,
        "need_self_train": need_self_train,
        "need_scalar_train": need_scalar_train,
        "need_tensor_train": need_tensor_train,
        "freeze": freeze,
        "final_pooling": final_pooling,
        "self_num_epochs": self_num_epochs,
        "scalar_num_epochs": scalar_num_epochs,
        "tensor_num_epochs": tensor_num_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "clip_grad_norm": clip_grad_norm,
        "save_interval": save_interval,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "warmup_periods": warmup_periods,
        "self_loss_func": self_loss_func,
        "scalar_loss_func": scalar_loss_func,
        "tensor_loss_func": tensor_loss_func,
        "self_train_limit": self_train_limit,
        "scalar_train_limit": scalar_train_limit,
        "tensor_train_limit": tensor_train_limit,
        "start_epoch": start_epoch,
        "model_type": model_type,
        "gmtnet_embed_dim": gmtnet_embed_dim,
        "gmtnet_num_attention_layers": gmtnet_num_attention_layers,
        "graph_mode": graph_mode,
        "max_neighbors": max_neighbors,
        "use_tensorboard": use_tensorboard,
    }

def main(
    cutoff: float = 5.0,
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    test_batch_size: int = 32,
    pin_memory: bool = True,
    num_workers: int = 4,
    seed: int = 42,
    train_val_test: tuple[float, float, float] = (0.8, 0.1, 0.1),
    dist_emb_func: str = "gaussian",
    embed_dim: int = 64,
    max_atom_type: int = 118,
    inv_update_method: str = "comformer",
    num_inv_layers: int = 3,
    middle_scalar_hidden_dim: int = 128,
    num_middle_hidden_layers: int = 1,
    equi_update_method: str = "tpconv_with_edge",
    num_equi_layers: int = 4,
    tp_method: str = "so2",
    scalar_dim: int = 16,
    vec_dim: int = 8,
    num_final_hidden_layers: int = 1,
    final_scalar_hidden_dim: int = 64,
    final_vec_hidden_dim: int = 16,
    final_scalar_out_dim: int = 16,
    final_vec_out_dim: int = 8,
    need_self_train: bool = True,
    need_scalar_train: bool = True,
    need_tensor_train: bool = True,
    final_pooling: bool = True,
    self_num_epochs: int = 100,
    scalar_num_epochs: int = 400,
    tensor_num_epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    clip_grad_norm: float = 1.0,
    save_interval: int = 5,
    optimizer: str = "adamw",
    scheduler: str = "cosine_annealing",
    warmup_periods: int = 10,
    self_loss_func: str = "huber",
    scalar_loss_func: str = "huber",
    tensor_loss_func: str = "huber",
    self_train_limit: int = None,
    scalar_train_limit: int = None,
    tensor_train_limit: int = None,
    checkpoint_dir: str = "checkpoints",
    pic_dir: str = "pics",
    metric_dir: str = "metrics",
    start_epoch: int = 0,
    resume_self_train: str = None,
    resume_scalar_train: str = None,
    resume_tensor_train: str = None,
    freeze: bool = False,
    share_middle_mlp: bool = True,
    scalar_invariant_only: bool = False,
    only_use_embedding: bool = False,
    model_type: str = "high_order",
    gmtnet_embed_dim: int = 128,
    gmtnet_num_attention_layers: int = 2,
    graph_mode: str = "high_order",
    max_neighbors: int = 12,
    use_tensorboard: bool = True,
):
    print("Start running...")
    
    # Note: Random seed is already set in main.py (entry point)
    # to ensure reproducibility from the very beginning of the program.
    
    params_to_save = _get_params_to_save(
        model_type=model_type,
        cutoff=cutoff,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        test_batch_size=test_batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        seed=seed,
        train_val_test=train_val_test,
        dist_emb_func=dist_emb_func,
        embed_dim=embed_dim,
        max_atom_type=max_atom_type,
        inv_update_method=inv_update_method,
        num_inv_layers=num_inv_layers,
        middle_scalar_hidden_dim=middle_scalar_hidden_dim,
        num_middle_hidden_layers=num_middle_hidden_layers,
        equi_update_method=equi_update_method,
        num_equi_layers=num_equi_layers,
        tp_method=tp_method,
        scalar_dim=scalar_dim,
        vec_dim=vec_dim,
        num_final_hidden_layers=num_final_hidden_layers,
        final_scalar_hidden_dim=final_scalar_hidden_dim,
        final_vec_hidden_dim=final_vec_hidden_dim,
        final_scalar_out_dim=final_scalar_out_dim,
        final_vec_out_dim=final_vec_out_dim,
        need_self_train=need_self_train,
        need_scalar_train=need_scalar_train,
        need_tensor_train=need_tensor_train,
        freeze=freeze,
        final_pooling=final_pooling,
        self_num_epochs=self_num_epochs,
        scalar_num_epochs=scalar_num_epochs,
        tensor_num_epochs=tensor_num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm,
        save_interval=save_interval,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_periods=warmup_periods,
        self_loss_func=self_loss_func,
        scalar_loss_func=scalar_loss_func,
        tensor_loss_func=tensor_loss_func,
        self_train_limit=self_train_limit,
        scalar_train_limit=scalar_train_limit,
        tensor_train_limit=tensor_train_limit,
        start_epoch=start_epoch,
        gmtnet_embed_dim=gmtnet_embed_dim,
        gmtnet_num_attention_layers=gmtnet_num_attention_layers,
        graph_mode=graph_mode,
        max_neighbors=max_neighbors,
        use_tensorboard=use_tensorboard,
    )
    params_path = save_params_json(params_to_save, checkpoint_dir)
    print(f"Saved parameters to {params_path}")

    print("Start loading data...")
    if need_self_train:
        USE_MP = True
        USE_ALEX = not USE_MP
        assert (USE_MP and not USE_ALEX) or (not USE_MP and USE_ALEX), "Only one of MP and Alexandria can be used for self-training."
        if USE_MP:
            self_trainset = get_mp_dataloader(
                cutoff=cutoff,
                batch_size=train_batch_size,
                pin_memory=pin_memory,
                num_workers=num_workers,
                shuffle=True,
                worker_init_fn=lambda worker_id: worker_init_fn(worker_id, seed) if num_workers > 0 else None,
                graph_mode=graph_mode,
                max_neighbors=max_neighbors,
            )
            batch_save_interval = None
        if USE_ALEX:
            self_trainset = get_alexandria_dataloader(
                cutoff=cutoff,
                batch_size=train_batch_size,
                pin_memory=pin_memory,
                num_workers=num_workers,
                shuffle=True,
                worker_init_fn=lambda worker_id: worker_init_fn(worker_id, seed) if num_workers > 0 else None,
                graph_mode=graph_mode,
                max_neighbors=max_neighbors,
            )
            batch_save_interval = 10000
    else:
        self_trainset = None
        batch_save_interval = None
    if need_scalar_train:
        scalar_dataloaders = _create_scalar_dataloaders(
            name_path_dict,
            scalar_properties,
            cutoff,
            train_val_test,
            seed,
            train_batch_size,
            val_batch_size,
            test_batch_size,
            pin_memory,
            num_workers,
            worker_init_fn=lambda worker_id: worker_init_fn(worker_id, seed) if num_workers > 0 else None,
            graph_mode=graph_mode,
            max_neighbors=max_neighbors,
        )
    else:
        scalar_dataloaders = None
    if need_tensor_train:
        tensor_dataloaders = _create_tensor_dataloaders(
            name_path_dict,
            tensor_properties,
            cutoff,
            train_val_test,
            seed,
            train_batch_size,
            val_batch_size,
            test_batch_size,
            pin_memory,
            num_workers,
            graph_mode,
            max_neighbors,
            worker_init_fn=lambda worker_id: worker_init_fn(worker_id, seed) if num_workers > 0 else None,
        )
    else:
        tensor_dataloaders = None

    print("Start training...")
    (
        scalar_models,
        tensor_models,
        embedding_layer,
        invariant_layers,
        equivariant_layers,
        scalar_train_history,
        tensor_train_history,
    ) = train(
        dist_emb_func=dist_emb_func,
        embed_dim=embed_dim,
        max_atom_type=max_atom_type,
        cutoff=cutoff,
        inv_update_method=inv_update_method,
        num_inv_layers=num_inv_layers,
        middle_scalar_hidden_dim=middle_scalar_hidden_dim,
        num_middle_hidden_layers=num_middle_hidden_layers,
        equi_update_method=equi_update_method,
        num_equi_layers=num_equi_layers,
        tp_method=tp_method,
        scalar_dim=scalar_dim,
        vec_dim=vec_dim,
        num_final_hidden_layers=num_final_hidden_layers,
        final_scalar_hidden_dim=final_scalar_hidden_dim,
        final_vec_hidden_dim=final_vec_hidden_dim,
        final_scalar_out_dim=final_scalar_out_dim,
        final_vec_out_dim=final_vec_out_dim,
        need_self_train=need_self_train,
        need_scalar_train=need_scalar_train,
        need_tensor_train=need_tensor_train,
        self_trainset=self_trainset,
        scalar_dataloaders=scalar_dataloaders,
        tensor_dataloaders=tensor_dataloaders,
        final_pooling=final_pooling,
        self_num_epochs=self_num_epochs,
        scalar_num_epochs=scalar_num_epochs,
        tensor_num_epochs=tensor_num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm,
        save_interval=save_interval,
        batch_save_interval=batch_save_interval,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_periods=warmup_periods,
        self_loss_func=self_loss_func,
        scalar_loss_func=scalar_loss_func,
        tensor_loss_func=tensor_loss_func,
        self_limit=self_train_limit,
        scalar_limit=scalar_train_limit,
        tensor_limit=tensor_train_limit,
        checkpoint_dir=checkpoint_dir,
        pic_dir=pic_dir,
        start_epoch=start_epoch,
        resume_self_train=resume_self_train,
        resume_scalar_train=resume_scalar_train,
        resume_tensor_train=resume_tensor_train,
        freeze=freeze,
        share_middle_mlp=share_middle_mlp,
        scalar_invariant_only=scalar_invariant_only,
        only_use_embedding=only_use_embedding,
        model_type=model_type,
        gmtnet_embed_dim=gmtnet_embed_dim,
        gmtnet_num_attention_layers=gmtnet_num_attention_layers,
        use_tensorboard=use_tensorboard,
    )

    print("Start testing...")
    scalar_results, tensor_results = test(
        scalar_models=scalar_models,
        tensor_models=tensor_models,
        scalar_dataloaders=scalar_dataloaders,
        tensor_dataloaders=tensor_dataloaders,
        pic_dir=pic_dir,
        metric_dir=metric_dir,
        scalar_train_history=scalar_train_history,
        tensor_train_history=tensor_train_history,
    )
    save_results_to_markdown(
        params=locals(),
        scalar_results=scalar_results,
        tensor_results=tensor_results,
        metric_dir=metric_dir,
    )

    print("Done.")
