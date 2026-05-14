import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from pathlib import Path

from src.model import Model, InvariantOnlyModel
from data import get_scalar_dataloader
from src.train_test.utils.visualization import (
    get_visualization_dir,
    plot_train_val_metrics,
)
from src.train_test.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


def validate_model(model, val_loader, device, loss_fn):
    """
    Validate the model on the validation dataset.
    
    Returns:
        tuple: (avg_val_loss, avg_val_mae)
    """
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            atom_type = batch.atom_type.to(device)
            edge_index = batch.edge_index.to(device)
            edge_vec = batch.edge_vec.to(device)
            batch_index = batch.batch.to(device)
            scalar_property = batch.scalar_property.to(device)

            pred_scalar_property = model(atom_type, edge_vec, edge_index, batch_index)
            mae = (pred_scalar_property - scalar_property).abs().mean()
            loss = loss_fn(pred_scalar_property, scalar_property)
            
            val_loss += loss.item()
            val_mae += mae.item()
            num_batches += 1

    avg_val_loss = val_loss / num_batches
    avg_val_mae = val_mae / num_batches
    
    model.train()
    return avg_val_loss, avg_val_mae


def scalar_train(
    property_name: str,
    embedding_layer,
    invariant_layers,
    middle_mlp,
    equivariant_layers,
    final_mlp,
    readout_layer,
    scalar_trainset,
    scalar_valset,
    num_epochs: int,
    checkpoint_dir: str = "checkpoints",
    pic_dir: str = "pics",
    start_epoch: int = 0,
    resume_from: str = None,
    save_interval: int = 5,
    clip_grad_norm: float = 1.0,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    optimizer: str = "adamw",
    scheduler: str = "cosine_annealing",
    loss_func: str = "huber",
    limit: int = None,
    use_amp: bool = True,
    scalar_invariant_only: bool = True,
):
    """
    Train a scalar property prediction model with validation.

    Args:
        property_name: Name of the property to predict
        embedding_layer: Embedding layer of the model
        invariant_layers: Invariant layers of the model
        middle_mlp: Middle MLP layers
        equivariant_layers: Equivariant layers of the model
        final_mlp: Final MLP layers
        readout_layer: Readout layer
        scalar_trainset: Training dataset
        scalar_valset: Validation dataset
        num_epochs: Number of training epochs
        checkpoint_dir: Directory to save checkpoints (a subfolder with timestamp will be created)
        pic_dir: Directory to save plots (a subfolder with timestamp will be created)
        start_epoch: Starting epoch (for resuming)
        resume_from: Path to resume from checkpoint
        save_interval: Interval to save checkpoints
        clip_grad_norm: Gradient clipping norm
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        optimizer: Type of optimizer ('adamw', 'adam', 'sgd')
        scheduler: Type of scheduler ('cosine_annealing', 'step')
        loss_func: Type of loss function ('huber', 'mse', 'l1')
        limit: Limit number of epochs (optional)
        use_amp: Whether to use automatic mixed precision

    Returns:
        tuple: (model, training_history)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tensorboard_dir = os.path.join(checkpoint_dir, "tensorboard", property_name)
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    
    if scalar_invariant_only:
        model = InvariantOnlyModel(
            embedding_layer,
            invariant_layers,
            # readout_layer,
        )
    else:
        model = Model(
            embedding_layer,
            invariant_layers,
            middle_mlp,
            equivariant_layers,
            final_mlp,
            readout_layer,
        )
    model = model.to(device)

    best_loss = float("inf")
    train_losses = []
    train_mae = []

    val_losses = []
    val_mae_scores = []

    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    opt = None
    sched = None

    if optimizer == "adamw":
        opt = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "sgd":
        opt = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"optimizer {optimizer} is not implemented")

    # if scheduler == "cosine_annealing":
    #     sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    # elif scheduler == "cosine_warm_restarts":
    #     sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=num_epochs // 2, T_mult=2, eta_min=1e-6)
    if scheduler == "cosine_annealing" or scheduler == "cosine_warm_restarts":
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    elif scheduler == "step":
        sched = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    else:
        raise NotImplementedError(f"scheduler {scheduler} is not implemented")

    if resume_from and os.path.exists(resume_from):
        checkpoint = load_checkpoint(resume_from, device)
        
        required_keys = ["model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "epoch", "best_loss"]
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(f"Checkpoint is missing required keys: {missing_keys}")
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        sched.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        train_losses = checkpoint["train_losses"]
        train_mae = checkpoint["train_mae"]
        val_losses = checkpoint.get("val_losses", [])
        val_mae_scores = checkpoint.get("val_mae_scores", [])
        
        checkpoint_use_amp = checkpoint.get("use_amp", True)
        if checkpoint_use_amp and "scaler_state_dict" in checkpoint:
            if use_amp and device.type == "cuda" and scaler is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            elif not use_amp:
                print("Warning: Checkpoint was saved with AMP, but use_amp=False. Scaler state will not be loaded.")
        elif not checkpoint_use_amp and use_amp:
            print("Warning: Checkpoint was saved without AMP, but use_amp=True. Starting with fresh scaler.")
        
        print(f"Resumed from checkpoint: {resume_from}, epoch {start_epoch}")

    batches = scalar_trainset

    if loss_func == "huber":
        loss_fn = nn.HuberLoss()
    elif loss_func == "mse":
        loss_fn = nn.MSELoss()
    elif loss_func == "l1":
        loss_fn = nn.L1Loss()
    else:
        raise NotImplementedError(f"loss_func {loss_func} is not implemented")

    if limit is None:
        limit = num_epochs
    
    val_mae = 0.0
    if val_mae_scores:
        val_mae = val_mae_scores[-1]
    
    model.train()
    for epoch in range(start_epoch, min(num_epochs, start_epoch + limit)):
        epoch_loss = 0.0
        epoch_mae = 0.0
        num_batches = 0

        pbar = tqdm(batches, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            atom_type = batch.atom_type.to(device)
            edge_index = batch.edge_index.to(device)
            edge_vec = batch.edge_vec.to(device)
            batch_index = batch.batch.to(device)
            scalar_property = batch.scalar_property.to(device)
            num_atoms = torch.bincount(batch_index).to(device)

            opt.zero_grad()
            
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    pred_scalar_property = model(atom_type, edge_vec, edge_index, batch_index)
                    assert pred_scalar_property.shape == scalar_property.shape, f"pred_scalar_property shape: {pred_scalar_property.shape}, scalar_property shape: {scalar_property.shape}"
                    assert num_atoms.shape == scalar_property.shape, f"num_atoms shape: {num_atoms.shape}, scalar_property shape: {scalar_property.shape}"
                    loss = loss_fn(pred_scalar_property, scalar_property)
                
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch}, batch {num_batches}")
                    print(f"Skipping this batch to prevent parameter update")
                    scaler.update()
                    continue
                
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                scaler.step(opt)
                scaler.update()
            else:
                pred_scalar_property = model(atom_type, edge_vec, edge_index, batch_index)
                loss = loss_fn(pred_scalar_property, scalar_property)
                
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch}, batch {num_batches}")
                    print(f"Skipping this batch to prevent parameter update")
                    continue
                
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                opt.step()
            
            # energy in dataset has already calculated per-atom-ly
            # if property_name in ["formation_energy", "total_energy", "e_form"]:
            #     mae = (pred_scalar_property - scalar_property).abs().div(num_atoms).mean()
            # else:
            if True:
                mae = (pred_scalar_property - scalar_property).abs().mean()

            epoch_loss += loss.item()
            epoch_mae += mae.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.6f}", "mae": f"{mae.item():.6f}"})

        avg_loss = epoch_loss / num_batches
        avg_mae = epoch_mae / num_batches
        train_losses.append(avg_loss)
        train_mae.append(avg_mae)
        
        val_loss, val_mae = validate_model(model, scalar_valset, device, loss_fn)
        val_losses.append(val_loss)
        val_mae_scores.append(val_mae)
        
        sched.step()

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("MAE/train", avg_mae, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("MAE/val", val_mae, epoch)
        writer.add_scalar("Learning_Rate", sched.get_last_lr()[0], epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.6f}, Train MAE: {avg_mae:.6f}, Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")

        checkpoint_data = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "best_loss": best_loss,
            "train_losses": train_losses,
            "train_mae": train_mae,
            "val_losses": val_losses,
            "val_mae_scores": val_mae_scores,
        }
        if use_amp and device.type == "cuda" and scaler is not None:
            checkpoint_data["scaler_state_dict"] = scaler.state_dict()
        checkpoint_data["use_amp"] = use_amp

        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = save_checkpoint(
                checkpoint_data=checkpoint_data,
                checkpoint_base_dir=checkpoint_dir,
                property_name=property_name,
                num_epochs=num_epochs,
                is_best=True,
            )
            print(f"Saved best model with val loss: {best_loss:.6f} to {checkpoint_path}")

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = save_checkpoint(
                checkpoint_data=checkpoint_data,
                checkpoint_base_dir=checkpoint_dir,
                property_name=property_name,
                num_epochs=num_epochs,
                epoch=epoch,
            )
            print(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")

    print(f"Training completed. Best val loss: {best_loss:.6f}, Final val MAE: {val_mae:.6f}")
    
    writer.close()
    
    vis_dir = get_visualization_dir(pic_dir)
    property_vis_dir = os.path.join(vis_dir, property_name)
    
    plot_train_val_metrics(
        train_values=train_losses,
        val_values=val_losses,
        save_dir=property_vis_dir,
        property_name=property_name,
        metric_name="loss",
        train_color="blue",
        val_color="orange",
    )
    
    plot_train_val_metrics(
        train_values=train_mae,
        val_values=val_mae_scores,
        save_dir=property_vis_dir,
        property_name=property_name,
        metric_name="mae",
        train_color="red",
        val_color="green",
    )
    
    training_history = {
        "train_losses": train_losses,
        "train_mae": train_mae,
        "val_losses": val_losses,
        "val_mae_scores": val_mae_scores,
    }
    
    return model, training_history
