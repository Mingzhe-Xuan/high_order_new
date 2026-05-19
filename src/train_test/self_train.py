import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from pathlib import Path

from src.model import Model
from data import get_mp_dataloader, get_alexandria_dataloader
from src.train_test.utils.visualization import get_visualization_dir, plot_train_val_metrics
from src.train_test.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)

def has_nan_inf(model):
    for param in model.parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            return True
    return False


def self_train(
    embedding_layer,
    invariant_layers,
    middle_mlp,
    equivariant_layers,
    final_mlp,
    readout_layer,
    dataloader,
    num_epochs: int,
    checkpoint_dir: str = "checkpoints",
    pic_dir: str = "pics",
    start_epoch: int = 0,
    resume_from: str = None,
    save_interval: int = 5,
    batch_save_interval: int = 50000,
    clip_grad_norm: float = 1.0,
    loss_func: str = "huber",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    optimizer: str = "adamw",
    scheduler: str = "cosine_warm_restarts",
    warmup_periods: int = 10,
    limit: int = None,
    use_amp: bool = True,
    model_instance=None,
):
    """
    Self-supervised training for the model using force prediction.
    
    Args:
        embedding_layer: Embedding layer of the model
        invariant_layers: Invariant layers of the model
        middle_mlp: Middle MLP layers
        equivariant_layers: Equivariant layers of the model
        final_mlp: Final MLP layers
        readout_layer: Readout layer
        dataloader: Training dataloader
        num_epochs: Number of training epochs
        checkpoint_dir: Directory to save checkpoints (a subfolder with timestamp will be created)
        pic_dir: Directory to save plots (a subfolder with timestamp will be created)
        start_epoch: Starting epoch (for resuming)
        resume_from: Path to resume from checkpoint
        save_interval: Interval to save checkpoints
        clip_grad_norm: Gradient clipping norm
        loss_func: Type of loss function ('huber', 'mse', 'l1')
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        optimizer: Type of optimizer ('adamw', 'adam', 'sgd')
        scheduler: Type of scheduler ('cosine_annealing', 'step', 'cosine_warm_restarts')
        warmup_periods: Number of periods for cosine warm restarts (default: 50)
        limit: Limit number of epochs (optional)
        use_amp: Whether to use automatic mixed precision

    Returns:
        model: The trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tensorboard_dir = os.path.join(checkpoint_dir, "tensorboard", "self_train")
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    
    if model_instance is None:
        model = Model(
            embedding_layer,
            invariant_layers,
            middle_mlp,
            equivariant_layers,
            final_mlp,
            readout_layer,
            self_train=True,
        )
    else:
        model = model_instance
    model = model.to(device)

    best_loss = float("inf")
    train_losses = []
    train_mae = []
    train_mse = []
    train_mean_fnorm_percent_error = []

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

    if scheduler == "cosine_annealing":
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    elif scheduler == "step":
        sched = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    elif scheduler == "cosine_warm_restarts":
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=warmup_periods, T_mult=2)
    else:
        raise NotImplementedError(f"scheduler {scheduler} is not implemented")

    if resume_from and os.path.exists(resume_from):
        checkpoint = load_checkpoint(resume_from, device)
        
        required_keys = ["model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "epoch", "best_loss"]
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(f"Checkpoint is missing required keys: {missing_keys}")
        
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        sched.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        train_losses = checkpoint["train_losses"]
        train_mae = checkpoint.get("train_mae", [])
        train_mse = checkpoint.get("train_mse", [])
        train_mean_fnorm_percent_error = checkpoint.get("train_mean_fnorm_percent_error", [])
        
        checkpoint_use_amp = checkpoint.get("use_amp", True)
        if checkpoint_use_amp and "scaler_state_dict" in checkpoint:
            if use_amp and device.type == "cuda" and scaler is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            elif not use_amp:
                print("Warning: Checkpoint was saved with AMP, but use_amp=False. Scaler state will not be loaded.")
        elif not checkpoint_use_amp and use_amp:
            print("Warning: Checkpoint was saved without AMP, but use_amp=True. Starting with fresh scaler.")
        
        print(f"Resumed from checkpoint: {resume_from}, epoch {start_epoch}")

    batches = dataloader

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
    model.train()
    for epoch in range(start_epoch, min(num_epochs, start_epoch + limit)):
        epoch_loss = 0.0
        epoch_mae_sum = 0.0
        epoch_mse_sum = 0.0
        epoch_fnorm_err_sum = 0.0
        num_batches = 0

        pbar = tqdm(batches, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            # # For debugging, skip the first 2610 batches
            # # if num_batches < 2610:
            # if num_batches < 4016:
            #     num_batches += 1
            #     continue
            atom_type = batch.atom_type.to(device)
            edge_index = batch.edge_index.to(device)
            edge_vec = batch.edge_vec.to(device)
            unstable_edge_vec = batch.unstable_edge_vec.to(device)
            batch_index = batch.batch.to(device)
            force = batch.force.to(device)
            # assert atom_type
            # assert force
            # assert edge_index
            # print("="*50)
            # print(atom_type)
            # print(edge_index)
            # print(force)
            # print("="*50)

            opt.zero_grad()
            
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    pred_force = model(atom_type, unstable_edge_vec, edge_index, batch_index)
                    # assert pred_force is not None
                    loss = loss_fn(pred_force, force)
                    pointwise_mae = (pred_force - force).view(-1).abs().mean()
                    mse = (pred_force - force).view(-1).pow(2).mean()
                    fnorm_error = torch.norm(pred_force - force, dim=-1)
                    fnorm = torch.norm(force, dim=-1)
                    mean_fnorm_percent_error = (fnorm_error / (fnorm + 1e-8)).mean()

                # if has_nan_inf(model):
                #     print("Nan or inf detected in model parameters!")
                #     print(model.state_dict())
                #     exit(1)
                
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch+1}, batch {num_batches}")
                    # num_batches += 1
                    print(f"Skipping this batch to prevent parameter update")
                    scaler.update()
                    continue
                
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                scaler.step(opt)
                scaler.update()
            else:
                pred_force = model(atom_type, unstable_edge_vec, edge_index, batch_index)
                # print("pred_force:", pred_force)
                # assert pred_force.all()
                loss = loss_fn(pred_force, force)
                pointwise_mae = (pred_force - force).view(-1).abs().mean()
                mse = (pred_force - force).view(-1).pow(2).mean()
                fnorm_error = abs(
                    torch.norm(pred_force, dim=-1) - torch.norm(force, dim=-1)
                )
                fnorm = torch.norm(force, dim=-1)
                mean_fnorm_percent_error = (fnorm_error / (fnorm + 1e-8)).mean()

                if has_nan_inf(model):
                    print("Nan or inf detected in model parameters!")
                    print(model.state_dict())
                    raise ValueError("NaN or inf detected in model parameters!")
                
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch+1}, batch {num_batches}")
                    # num_batches += 1
                    print(f"Skipping this batch to prevent parameter update")
                    continue
                
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                opt.step()

            epoch_loss += loss.item()
            epoch_mae_sum += pointwise_mae.item()
            epoch_mse_sum += mse.item()
            epoch_fnorm_err_sum += mean_fnorm_percent_error.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "mae": f"{pointwise_mae.item():.6f}",
                "mse": f"{mse.item():.6f}",
                "fnorm_err%": f"{mean_fnorm_percent_error.item():.6f}"
            })

            if batch_save_interval and num_batches % batch_save_interval == 0:
                avg_loss = epoch_loss / num_batches
                best_loss = min(best_loss, avg_loss)
                avg_mae = epoch_mae_sum / num_batches
                avg_mse = epoch_mse_sum / num_batches
                avg_fnorm_err = epoch_fnorm_err_sum / num_batches
                train_losses.append(avg_loss)
                train_mae.append(avg_mae)
                train_mse.append(avg_mse)
                train_mean_fnorm_percent_error.append(avg_fnorm_err)
                checkpoint_data = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "loss": avg_loss,
                    "best_loss": best_loss,
                    "train_losses": train_losses,
                    "train_mae": train_mae,
                    "train_mse": train_mse,
                    "train_mean_fnorm_percent_error": train_mean_fnorm_percent_error,
                }
                save_checkpoint(
                    checkpoint_data=checkpoint_data,
                    checkpoint_base_dir=checkpoint_dir,
                    property_name="self_train",
                    num_epochs=num_epochs,
                    is_best=False,
                )

        avg_loss = epoch_loss / num_batches
        avg_mae = epoch_mae_sum / num_batches
        avg_mse = epoch_mse_sum / num_batches
        avg_fnorm_err = epoch_fnorm_err_sum / num_batches
        
        train_losses.append(avg_loss)
        train_mae.append(avg_mae)
        train_mse.append(avg_mse)
        train_mean_fnorm_percent_error.append(avg_fnorm_err)
        
        sched.step()

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("MAE/train", avg_mae, epoch)
        writer.add_scalar("MSE/train", avg_mse, epoch)
        writer.add_scalar("FNorm_Percent_Error/train", avg_fnorm_err, epoch)
        writer.add_scalar("Learning_Rate", sched.get_last_lr()[0], epoch)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.6f}, "
            f"Train MAE: {avg_mae:.6f}, Train MSE: {avg_mse:.6f}, "
            f"Train Mean FNORM % Error: {avg_fnorm_err:.6f}"
        )

        checkpoint_data = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "loss": avg_loss,
            "best_loss": best_loss,
            "train_losses": train_losses,
            "train_mae": train_mae,
            "train_mse": train_mse,
            "train_mean_fnorm_percent_error": train_mean_fnorm_percent_error,
        }
        if use_amp and device.type == "cuda" and scaler is not None:
            checkpoint_data["scaler_state_dict"] = scaler.state_dict()
        checkpoint_data["use_amp"] = use_amp

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = save_checkpoint(
                checkpoint_data=checkpoint_data,
                checkpoint_base_dir=checkpoint_dir,
                property_name="self_train",
                num_epochs=num_epochs,
                is_best=True,
            )
            print(f"Saved best model with loss: {best_loss:.6f} to {checkpoint_path}")

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = save_checkpoint(
                checkpoint_data=checkpoint_data,
                checkpoint_base_dir=checkpoint_dir,
                property_name="self_train",
                num_epochs=num_epochs,
                epoch=epoch,
            )
            print(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")

    print(f"Training completed. Best loss: {best_loss:.6f}")
    
    writer.close()
    
    vis_dir = get_visualization_dir(pic_dir)
    self_train_vis_dir = os.path.join(vis_dir, "self_train")
    
    plot_train_val_metrics(
        train_values=train_losses,
        val_values=[],
        save_dir=self_train_vis_dir,
        property_name="self_train",
        metric_name="loss",
        train_color="blue",
        val_color="orange",
        title="Self-Training Loss Over Epochs",
        filename="self_train_loss.png",
    )
    
    plot_train_val_metrics(
        train_values=train_mae,
        val_values=[],
        save_dir=self_train_vis_dir,
        property_name="self_train",
        metric_name="mae",
        train_color="red",
        val_color="green",
        title="Self-Training MAE Over Epochs",
        filename="self_train_mae.png",
    )
    
    plot_train_val_metrics(
        train_values=train_mse,
        val_values=[],
        save_dir=self_train_vis_dir,
        property_name="self_train",
        metric_name="mse",
        train_color="purple",
        val_color="brown",
        title="Self-Training MSE Over Epochs",
        filename="self_train_mse.png",
    )
    
    plot_train_val_metrics(
        train_values=train_mean_fnorm_percent_error,
        val_values=[],
        save_dir=self_train_vis_dir,
        property_name="self_train",
        metric_name="mean_fnorm_percent_error",
        train_color="olive",
        val_color="teal",
        title="Self-Training Mean FNORM % Error Over Epochs",
        filename="self_train_fnorm_error.png",
    )
    
    return model
