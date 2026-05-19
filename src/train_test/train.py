import argparse
import torch
import torch.nn as nn
import json
import os
from typing import Union
from e3nn.o3 import Irreps
from torch_geometric.loader import DataLoader

from src.model.utils import get_irreps_from_ns_nv_lmax

from .self_train import self_train
from .scalar_train import scalar_train
from .tensor_train import tensor_train
from .utils import analyze_model_components, save_num_params_markdown
from .utils.checkpoint import get_checkpoint_dir
from .utils.freeze_parameters import freeze_parameters
from data import scalar_properties, tensor_properties, readout_configs
from src.model import (
    EmbeddingLayer,
    InvariantLayer,
    MiddleMLP,
    EquivariantLayer,
    FinalMLP,
    ReadoutLayer,
    Model,
)
from src.model.gmtnet import HighOrderGMTNet


def _create_shared_components(
    dist_emb_func,
    embed_dim,
    max_atom_type,
    cutoff,
    inv_update_method,
    num_inv_layers,
    num_equi_layers,
    equi_update_method,
    tp_method,
    scalar_dim,
    vec_dim,
    num_final_hidden_layers,
    final_scalar_hidden_dim,
    final_vec_hidden_dim,
    final_scalar_out_dim,
    final_vec_out_dim,
):
    """Create shared model components."""
    # Shared embedding layer
    embedding_layer = EmbeddingLayer(
        dist_emb_func=dist_emb_func,
        embed_dim=embed_dim,
        max_atom_type=max_atom_type,
        cutoff=cutoff,
    )

    # Shared invariant layers
    invariant_layers = nn.ModuleList(
        [
            InvariantLayer(update_method=inv_update_method, scalar_dim=embed_dim)
            for _ in range(num_inv_layers)
        ]
    )

    # Build irreps list
    irreps_list = [f"{scalar_dim}x0e"]
    l_max = num_equi_layers

    #####################################################################
    # SO(3) PROBLEM
    for l in range(1, l_max + 1):
        irreps = f"{scalar_dim}x0e"
        for _l in range(l + 1):
            _p = "e" if _l % 2 == 0 else "o"
            irreps += f"+{vec_dim}x{_l}{_p}"
        irreps_list.append(irreps)
    print(irreps_list)

    # irreps = ""
    #####################################################################

    # create irreps_vec
    irreps_vec = "0e"
    for l in range(1, l_max + 1):
        p = "e" if l % 2 == 0 else "o"
        irreps_vec += f"+{l}{p}"

    # Shared equivariant layers
    equivariant_layers = nn.ModuleList(
        [
            EquivariantLayer(
                update_method=equi_update_method,
                irreps_in=irreps_list[i],
                irreps_out=irreps_list[i + 1],
                irreps_vec=irreps_vec,
                tp_method=tp_method,
                residual=False,
            )
            for i in range(num_equi_layers)
        ]
    )

    final_irreps_hidden = get_irreps_from_ns_nv_lmax(
        ns=final_scalar_hidden_dim, nv=final_vec_hidden_dim, l_max=l_max
    )
    final_irreps_out = get_irreps_from_ns_nv_lmax(
        ns=final_scalar_out_dim, nv=final_vec_out_dim, l_max=l_max
    )

    return (
        embedding_layer,
        invariant_layers,
        equivariant_layers,
        irreps_list,
        final_irreps_hidden,
        final_irreps_out,
        l_max,
    )


def _create_scalar_models(
    scalar_properties,
    embedding_layer,
    invariant_layers,
    equivariant_layers,
    irreps_list,
    final_irreps_hidden,
    final_irreps_out,
    final_pooling,
    embed_dim,
    scalar_dim,
    middle_scalar_hidden_dim,
    num_middle_hidden_layers,
    num_final_hidden_layers,
):
    """Create models for scalar properties."""
    models = {}

    for prop in scalar_properties:
        # Create model components for each property
        middle_mlp = MiddleMLP(
            scalar_dim_in=embed_dim,
            scalar_dim_hidden=middle_scalar_hidden_dim,
            scalar_dim_out=scalar_dim,
            num_hidden_layers=num_middle_hidden_layers,
        )

        final_mlp = FinalMLP(
            irreps_in=Irreps(irreps_list[-1]),
            irreps_hidden=final_irreps_hidden,
            irreps_out=final_irreps_out,
            num_hidden_layers=num_final_hidden_layers,
        )

        readout_layer = ReadoutLayer(
            l_max=0,
            symmetry=None,
            irreps_out=final_irreps_out,
        )

        model = Model(
            embedding_layer=embedding_layer,
            invariant_layers=invariant_layers,
            middle_mlp=middle_mlp,
            equivariant_layers=equivariant_layers,
            final_mlp=final_mlp,
            readout_layer=readout_layer,
            self_train=False,
            final_pooling=final_pooling,
            irreps_list=irreps_list,
        )

        models[prop] = model

    return models


def _create_tensor_models(
    tensor_properties,
    embedding_layer,
    invariant_layers,
    equivariant_layers,
    irreps_list,
    final_irreps_hidden,
    final_irreps_out,
    final_pooling,
    scalar_dim,
    vec_dim,
    middle_scalar_hidden_dim,
    num_middle_hidden_layers,
    num_final_hidden_layers,
):
    """Create models for tensor properties."""
    models = {}

    # Define readout layer configurations for different tensor properties
    # readout_configs = {
    #     "dielectric": {"l_max": 2, "symmetry": "ij=ji"},
    #     "dielectric_ionic": {"l_max": 2, "symmetry": "ij=ji"},
    #     "elastic_sym_kbar": {"l_max": 4, "symmetry": "ijkl=jikl=ijlk=klij"},
    #     "elastic_total_kbar": {"l_max": 4, "symmetry": "ijkl=jikl=ijlk=klij"},
    #     "piezoelectric_C_m2": {"l_max": 2, "symmetry": "i,jk=kj"},
    #     "piezoelectric_e_Angst": {"l_max": 2, "symmetry": "i,jk=kj"},
    # }

    embed_dim = embedding_layer.embed_dim
    for prop in tensor_properties:
        # Create model components for each property
        middle_mlp = MiddleMLP(
            scalar_dim_in=embed_dim,
            scalar_dim_hidden=middle_scalar_hidden_dim,
            scalar_dim_out=scalar_dim,
            num_hidden_layers=num_middle_hidden_layers,
        )

        final_mlp = FinalMLP(
            irreps_in=Irreps(irreps_list[-1]),
            irreps_hidden=final_irreps_hidden,
            irreps_out=final_irreps_out,
            num_hidden_layers=num_final_hidden_layers,
        )

        readout_config = readout_configs.get(prop, {"l_max": 2, "symmetry": None})
        readout_layer = ReadoutLayer(
            l_max=readout_config["l_max"],
            symmetry=readout_config["symmetry"],
            irreps_out=final_irreps_out,
        )

        model = Model(
            embedding_layer=embedding_layer,
            invariant_layers=invariant_layers,
            middle_mlp=middle_mlp,
            equivariant_layers=equivariant_layers,
            final_mlp=final_mlp,
            readout_layer=readout_layer,
            self_train=False,
            final_pooling=final_pooling,
            irreps_list=irreps_list,
        )

        models[prop] = model

    return models


def _create_gmtnet_tensor_models(tensor_properties, **kwargs):
    models = {}
    for prop in tensor_properties:
        models[prop] = HighOrderGMTNet(
            target=prop,
            args=argparse.Namespace(**kwargs),
            task_mode="tensor",
        )
    return models


def _create_gmtnet_scalar_models(scalar_properties, **kwargs):
    models = {}
    for prop in scalar_properties:
        models[prop] = HighOrderGMTNet(
            target=prop,
            args=argparse.Namespace(**kwargs),
            task_mode="scalar",
        )
    return models


def _create_gmtnet_force_model(**kwargs):
    return HighOrderGMTNet(
        args=argparse.Namespace(**kwargs),
        task_mode="force",
    )


def train(
    # # random seed
    # seed: int = 42,
    # model
    # embedding layer
    dist_emb_func: str = "gaussian",
    embed_dim: int = 64,
    max_atom_type: int = 118,
    cutoff: float = 5.0,
    # invariant layers
    inv_update_method: str = "comformer",
    # inv_dim: int = 64,
    num_inv_layers: int = 3,
    # middle_mlp
    middle_scalar_hidden_dim: int = 128,  # hidden dim should be 2 times of the input by convention
    num_middle_hidden_layers: int = 1,
    # equivariant layers
    equi_update_method: str = "tpconv_with_edge",
    num_equi_layers: int = 4,
    tp_method: str = "so2",
    scalar_dim: int = 16,
    vec_dim: int = 8,
    # irreps_list: Union[list[str], list[Irreps]] = ["128x0e", ""],
    # final mlp
    num_final_hidden_layers: int = 1,
    final_scalar_hidden_dim: int = 64,
    final_vec_hidden_dim: int = 16,
    final_scalar_out_dim: int = 16,
    final_vec_out_dim: int = 8,
    # train
    need_self_train: bool = True,
    need_scalar_train: bool = True,
    need_tensor_train: bool = True,
    self_trainset: DataLoader = None,
    scalar_dataloaders: dict[str, DataLoader] = None,
    tensor_dataloaders: dict[str, DataLoader] = None,
    final_pooling: bool = True,
    # train_val_test: tuple[float, float, float] = (0.8, 0.1, 0.1),
    # batch_size: int = 32,
    # num_workers: int = 0,
    # pin_memory: bool = True,
    self_num_epochs: int = 100,
    scalar_num_epochs: int = 400,
    tensor_num_epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    clip_grad_norm: float = 1.0,
    save_interval: int = 5,
    batch_save_interval: int = None,
    optimizer: str = "adamw",
    scheduler: str = "cosine_annealing",
    warmup_periods: int = 10,
    self_loss_func: str = "huber",
    scalar_loss_func: str = "huber",
    tensor_loss_func: str = "huber",
    self_limit: int = None,
    scalar_limit: int = None,
    tensor_limit: int = None,
    checkpoint_dir: str = "checkpoints",
    pic_dir: str = "pics",
    start_epoch: int = 0,
    resume_self_train: str = None,
    resume_scalar_train: str = None,
    resume_tensor_train: str = None,
    freeze: bool = False,
    use_amp: bool = False,
    share_middle_mlp: bool = True,
    scalar_invariant_only: bool = True,
    only_use_embedding: bool = False,
    model_type: str = "high_order",
    gmtnet_embed_dim: int = 128,
    gmtnet_num_attention_layers: int = 2,
    use_tensorboard: bool = True,
):
    """
    1. self train: emb, inv, eqv, deco, readout - scalar train: emb, inv, deco - tensor train: emb, inv, eqv, deco, readout
    2. different readout layers, different final mlps, shared embedding, invariant and equivariant layers
    3. use comformer as invariant layer, tpconv_with_edge by default
    4. different scalar properties: train on different scalar properties one batch each time with different models, and so on.
    """
    # scalar_properties = [
    #     "formation_energy",
    #     "opt_bandgap",
    #     "total_energy",
    #     "ehull",
    #     "mbj_bandgap",
    #     "bandgap",
    #     "e_form",
    #     "bulk_modulus",
    #     "shear_modulus",
    # ]
    # tensor_properties = [
    #     "dielectric",
    #     "dielectric_ionic",
    #     "elastic_sym_kbar",
    #     "elastic_total_kbar",
    #     "piezoelectric_C_m2",
    #     "piezoelectric_e_Angst",
    # ]
    assert (
        self_trainset is not None or not need_self_train
    ), "self_trainset is required when need_self_train is True"
    assert (
        scalar_dataloaders is not None or not need_scalar_train
    ), "scalar_dataloaders is required when need_scalar_train is True"
    assert (
        tensor_dataloaders is not None or not need_tensor_train
    ), "tensor_dataloaders is required when need_tensor_train is True"
    ################################ This part should be in main.py ##################################
    # # datasets
    # # 1. self trainset
    # # 2. scalar trainset, valset, testset
    # # 3. tensor trainset, valset, testset
    # with open("data/dataloaders/name_path.json") as f:
    #     name_path_dict = json.load(f)

    # if need_self_train:
    #     self_trainset = get_mp_dataloader(
    #         cutoff=cutoff,
    #         batch_size=batch_size,
    #         pin_memory=pin_memory,
    #         num_workers=num_workers,
    #         shuffle=True,
    #     )

    # # Create scalar dataloaders
    # if need_scalar_train:
    #     scalar_dataloaders = _create_scalar_dataloaders(
    #         name_path_dict,
    #         scalar_properties,
    #         cutoff,
    #         train_val_test,
    #         seed,
    #         batch_size,
    #         pin_memory,
    #         num_workers,
    #     )

    # # Create tensor dataloaders
    # if need_tensor_train:
    #     tensor_dataloaders = _create_tensor_dataloaders(
    #         name_path_dict,
    #         tensor_properties,
    #         cutoff,
    #         train_val_test,
    #         seed,
    #         batch_size,
    #         pin_memory,
    #         num_workers,
    #     )
    ##################################################################################################

    assert model_type in {"high_order", "gmtnet"}, f"model_type {model_type} is not implemented"

    # Create shared model components
    (
        embedding_layer,
        invariant_layers,
        equivariant_layers,
        irreps_list,
        final_irreps_hidden,
        final_irreps_out,
        l_max,
    ) = _create_shared_components(
        dist_emb_func,
        embed_dim,
        max_atom_type,
        cutoff,
        inv_update_method,
        num_inv_layers,
        num_equi_layers,
        equi_update_method,
        tp_method,
        scalar_dim,
        vec_dim,
        num_final_hidden_layers,
        final_scalar_hidden_dim,
        final_vec_hidden_dim,
        final_scalar_out_dim,
        final_vec_out_dim,
    )

    # Analyze and save model parameters before training
    print("Analyzing model parameters...")
    params_dict = {
        "dist_emb_func": dist_emb_func,
        "embed_dim": embed_dim,
        "max_atom_type": max_atom_type,
        "cutoff": cutoff,
        "inv_update_method": inv_update_method,
        "num_inv_layers": num_inv_layers,
        "num_equi_layers": num_equi_layers,
        "equi_update_method": equi_update_method,
        "tp_method": tp_method,
        "scalar_dim": scalar_dim,
        "vec_dim": vec_dim,
        "num_final_hidden_layers": num_final_hidden_layers,
        "final_scalar_hidden_dim": final_scalar_hidden_dim,
        "final_vec_hidden_dim": final_vec_hidden_dim,
        "final_scalar_out_dim": final_scalar_out_dim,
        "final_vec_out_dim": final_vec_out_dim,
        "self_num_epochs": self_num_epochs,
        "scalar_num_epochs": scalar_num_epochs,
        "tensor_num_epochs": tensor_num_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "model_type": model_type,
        "use_tensorboard": use_tensorboard,
    }
    
    # Create a temporary middle_mlp and final_mlp for analysis
    temp_middle_mlp = MiddleMLP(
        scalar_dim_in=embed_dim,
        scalar_dim_hidden=middle_scalar_hidden_dim,
        scalar_dim_out=scalar_dim,
        num_hidden_layers=num_middle_hidden_layers,
    )
    temp_final_mlp = FinalMLP(
        irreps_in=Irreps(irreps_list[-1]),
        irreps_hidden=final_irreps_hidden,
        irreps_out=final_irreps_out,
        num_hidden_layers=num_final_hidden_layers,
    )
    
    # Move model components to GPU before analysis
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_layer = embedding_layer.to(device)
    invariant_layers = invariant_layers.to(device)
    equivariant_layers = equivariant_layers.to(device)
    temp_middle_mlp = temp_middle_mlp.to(device)
    temp_final_mlp = temp_final_mlp.to(device)
    
    # Analyze model components
    model_summaries = analyze_model_components(
        embedding_layer=embedding_layer,
        invariant_layers=invariant_layers,
        middle_mlp=temp_middle_mlp,
        equivariant_layers=equivariant_layers,
        final_mlp=temp_final_mlp,
        readout_layer=None,
    )
    
    # Save parameter summary to markdown file
    checkpoint_dir_with_timestamp = get_checkpoint_dir(checkpoint_dir)
    save_num_params_markdown(
        model_summaries=model_summaries,
        output_path=checkpoint_dir_with_timestamp,
        params_dict=params_dict,
    )
    print(f"Model parameters summary saved to {checkpoint_dir_with_timestamp}")
    
    # # Clean up temporary components
    # del temp_middle_mlp, temp_final_mlp
    middle_mlp = temp_middle_mlp
    final_mlp = temp_final_mlp

    # Self train
    scalar_train_history = {}
    tensor_train_history = {}

    if need_self_train:
        if model_type == "gmtnet":
            gmtnet_force_model = _create_gmtnet_force_model(
                gmtnet_embed_dim=gmtnet_embed_dim,
                gmtnet_num_attention_layers=gmtnet_num_attention_layers,
            )
            self_trained_model = self_train(
                embedding_layer=None,
                invariant_layers=None,
                middle_mlp=None,
                equivariant_layers=None,
                final_mlp=None,
                readout_layer=None,
                dataloader=self_trainset,
                num_epochs=self_num_epochs,
                checkpoint_dir=checkpoint_dir,
                pic_dir=pic_dir,
                start_epoch=start_epoch,
                resume_from=resume_self_train,
                save_interval=save_interval,
                batch_save_interval=batch_save_interval,
                clip_grad_norm=clip_grad_norm,
                loss_func=self_loss_func,
                learning_rate=lr,
                weight_decay=weight_decay,
                optimizer=optimizer,
                scheduler=scheduler,
                warmup_periods=warmup_periods,
                limit=self_limit,
                use_amp=use_amp,
                model_instance=gmtnet_force_model,
            )
        else:
            self_middle_mlp = MiddleMLP(
                scalar_dim_in=embed_dim,
                scalar_dim_hidden=middle_scalar_hidden_dim,
                scalar_dim_out=scalar_dim,
                num_hidden_layers=num_middle_hidden_layers,
            )
            self_final_mlp = FinalMLP(
                irreps_in=Irreps(irreps_list[-1]),
                irreps_hidden=final_irreps_hidden,
                irreps_out=final_irreps_out,
                num_hidden_layers=num_final_hidden_layers,
            )
            self_readout_layer = ReadoutLayer(
                l_max=1,
                symmetry=None,
                irreps_out=final_irreps_out,
            )

            self_trained_model = self_train(
                embedding_layer=embedding_layer,
                invariant_layers=invariant_layers,
                middle_mlp=self_middle_mlp,
                equivariant_layers=equivariant_layers,
                final_mlp=self_final_mlp,
                readout_layer=self_readout_layer,
                dataloader=self_trainset,
                num_epochs=self_num_epochs,
                checkpoint_dir=checkpoint_dir,
                pic_dir=pic_dir,
                start_epoch=start_epoch,
                resume_from=resume_self_train,
                save_interval=save_interval,
                batch_save_interval=batch_save_interval,
                clip_grad_norm=clip_grad_norm,
                loss_func=self_loss_func,
                learning_rate=lr,
                weight_decay=weight_decay,
                optimizer=optimizer,
                scheduler=scheduler,
                warmup_periods=warmup_periods,
                limit=self_limit,
                use_amp=use_amp,
            )

            embedding_layer = self_trained_model.embedding_layer
            if not only_use_embedding:
                invariant_layers = self_trained_model.invariant_layers
                middle_mlp = self_trained_model.middle_mlp
                equivariant_layers = self_trained_model.equivariant_layers

            if freeze:
                embedding_layer = freeze_parameters(embedding_layer)
                invariant_layers = freeze_parameters(invariant_layers)
                middle_mlp = freeze_parameters(middle_mlp)
                equivariant_layers = freeze_parameters(equivariant_layers)

    if need_scalar_train:
        if model_type == "gmtnet":
            scalar_models = _create_gmtnet_scalar_models(
                scalar_properties,
                gmtnet_embed_dim=gmtnet_embed_dim,
                gmtnet_num_attention_layers=gmtnet_num_attention_layers,
            )
        else:
            scalar_models = _create_scalar_models(
                scalar_properties,
                embedding_layer,
                invariant_layers,
                equivariant_layers,
                irreps_list,
                final_irreps_hidden,
                final_irreps_out,
                final_pooling,
                embed_dim,
                scalar_dim,
                middle_scalar_hidden_dim,
                num_middle_hidden_layers,
                num_final_hidden_layers,
            )

        for prop in scalar_properties:
            if model_type == "gmtnet":
                trained_model, history = scalar_train(
                    property_name=prop,
                    embedding_layer=None,
                    invariant_layers=None,
                    middle_mlp=None,
                    equivariant_layers=None,
                    final_mlp=None,
                    readout_layer=None,
                    scalar_trainset=scalar_dataloaders[f"{prop}_trainset"],
                    scalar_valset=scalar_dataloaders[f"{prop}_valset"],
                    num_epochs=scalar_num_epochs,
                    checkpoint_dir=checkpoint_dir,
                    pic_dir=pic_dir,
                    start_epoch=start_epoch,
                    resume_from=resume_scalar_train,
                    save_interval=save_interval,
                    clip_grad_norm=clip_grad_norm,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_func=scalar_loss_func,
                    limit=scalar_limit,
                    use_amp=use_amp,
                    model_instance=scalar_models[prop],
                )
            elif share_middle_mlp:
                trained_model, history = scalar_train(
                    property_name=prop,
                    embedding_layer=embedding_layer,
                    invariant_layers=invariant_layers,
                    middle_mlp=middle_mlp,
                    equivariant_layers=equivariant_layers,
                    final_mlp=scalar_models[prop].final_mlp,
                    readout_layer=scalar_models[prop].readout_layer,
                    scalar_trainset=scalar_dataloaders[f"{prop}_trainset"],
                    scalar_valset=scalar_dataloaders[f"{prop}_valset"],
                    num_epochs=scalar_num_epochs,
                    checkpoint_dir=checkpoint_dir,
                    pic_dir=pic_dir,
                    start_epoch=start_epoch,
                    resume_from=resume_scalar_train,
                    save_interval=save_interval,
                    clip_grad_norm=clip_grad_norm,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scalar_invariant_only=scalar_invariant_only,
                    loss_func=scalar_loss_func,
                    limit=scalar_limit,
                    use_amp=use_amp,
                )
            else:
                trained_model, history = scalar_train(
                    property_name=prop,
                    embedding_layer=embedding_layer,
                    invariant_layers=invariant_layers,
                    middle_mlp=scalar_models[prop].middle_mlp,
                    equivariant_layers=equivariant_layers,
                    final_mlp=scalar_models[prop].final_mlp,
                    readout_layer=scalar_models[prop].readout_layer,
                    scalar_trainset=scalar_dataloaders[f"{prop}_trainset"],
                    scalar_valset=scalar_dataloaders[f"{prop}_valset"],
                    num_epochs=scalar_num_epochs,
                    checkpoint_dir=checkpoint_dir,
                    pic_dir=pic_dir,
                    start_epoch=start_epoch,
                    resume_from=resume_scalar_train,
                    save_interval=save_interval,
                    clip_grad_norm=clip_grad_norm,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_func=scalar_loss_func,
                    limit=scalar_limit,
                    use_amp=use_amp,
                )
            scalar_models[prop] = trained_model
            scalar_train_history[prop] = history

            # Each scalar property has its own model
            # embedding_layer = trained_model.embedding_layer
            # invariant_layers = trained_model.invariant_layers
            # equivariant_layers = trained_model.equivariant_layers
    else:
        scalar_models = None

    if need_tensor_train:
        if model_type == "gmtnet":
            tensor_models = _create_gmtnet_tensor_models(
                tensor_properties,
                use_mask=True,
                gmtnet_embed_dim=gmtnet_embed_dim,
                gmtnet_num_attention_layers=gmtnet_num_attention_layers,
            )
        else:
            tensor_models = _create_tensor_models(
                tensor_properties,
                embedding_layer,
                invariant_layers,
                equivariant_layers,
                irreps_list,
                final_irreps_hidden,
                final_irreps_out,
                final_pooling,
                scalar_dim,
                vec_dim,
                middle_scalar_hidden_dim,
                num_middle_hidden_layers,
                num_final_hidden_layers,
            )

        for prop in tensor_properties:
            if model_type == "gmtnet":
                trained_model, history = tensor_train(
                    property_name=prop,
                    embedding_layer=None,
                    invariant_layers=None,
                    middle_mlp=None,
                    equivariant_layers=None,
                    final_mlp=None,
                    readout_layer=None,
                    tensor_trainset=tensor_dataloaders[f"{prop}_trainset"],
                    tensor_valset=tensor_dataloaders[f"{prop}_valset"],
                    num_epochs=tensor_num_epochs,
                    checkpoint_dir=checkpoint_dir,
                    pic_dir=pic_dir,
                    start_epoch=start_epoch,
                    resume_from=resume_tensor_train,
                    save_interval=save_interval,
                    clip_grad_norm=clip_grad_norm,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_func=tensor_loss_func,
                    limit=tensor_limit,
                    use_amp=use_amp,
                    model_instance=tensor_models[prop],
                    use_tensorboard=use_tensorboard,
                )
            elif share_middle_mlp:
                trained_model, history = tensor_train(
                    property_name=prop,
                    embedding_layer=embedding_layer,
                    invariant_layers=invariant_layers,
                    middle_mlp=middle_mlp,
                    equivariant_layers=equivariant_layers,
                    final_mlp=tensor_models[prop].final_mlp,
                    readout_layer=tensor_models[prop].readout_layer,
                    tensor_trainset=tensor_dataloaders[f"{prop}_trainset"],
                    tensor_valset=tensor_dataloaders[f"{prop}_valset"],
                    num_epochs=tensor_num_epochs,
                    checkpoint_dir=checkpoint_dir,
                    pic_dir=pic_dir,
                    start_epoch=start_epoch,
                    resume_from=resume_tensor_train,
                    save_interval=save_interval,
                    clip_grad_norm=clip_grad_norm,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_func=tensor_loss_func,
                    limit=tensor_limit,
                    use_amp=use_amp,
                )
            else:
                trained_model, history = tensor_train(
                    property_name=prop,
                    embedding_layer=embedding_layer,
                    invariant_layers=invariant_layers,
                    middle_mlp=tensor_models[prop].middle_mlp,
                    equivariant_layers=equivariant_layers,
                    final_mlp=tensor_models[prop].final_mlp,
                    readout_layer=tensor_models[prop].readout_layer,
                    tensor_trainset=tensor_dataloaders[f"{prop}_trainset"],
                    tensor_valset=tensor_dataloaders[f"{prop}_valset"],
                    num_epochs=tensor_num_epochs,
                    checkpoint_dir=checkpoint_dir,
                    pic_dir=pic_dir,
                    start_epoch=start_epoch,
                    resume_from=resume_tensor_train,
                    save_interval=save_interval,
                    clip_grad_norm=clip_grad_norm,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_func=tensor_loss_func,
                    limit=tensor_limit,
                    use_amp=use_amp,
                )
            tensor_models[prop] = trained_model
            tensor_train_history[prop] = history

            # Each tensor property has its own model
            # equi_shared = True
            # if equi_shared:
            #     embedding_layer = trained_model.embedding_layer
            #     invariant_layers = trained_model.invariant_layers
            #     equivariant_layers = trained_model.equivariant_layers
    else:
        tensor_models = None

    return (
        scalar_models,
        tensor_models,
        embedding_layer,
        invariant_layers,
        equivariant_layers,
        scalar_train_history,
        tensor_train_history,
    )
