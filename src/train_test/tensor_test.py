import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import numpy as np
from torch_geometric.loader import DataLoader

from data import tensor_properties
from src.train_test.utils.visualization import (
    get_visualization_dir,
    plot_all_train_val_test_metrics,
    plot_prediction_scatter,
    plot_fnorm_scatter,
)


def calculate_tensor_metrics(y_true, y_pred):
    """
    Calculate various tensor-specific metrics for model evaluation.

    Args:
        y_true: Ground truth tensor values with length=num_strctures
        y_pred: Predicted tensor values with length=num_strctures

    Returns:
        dict: Dictionary containing various metrics
    """
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true)
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred)
    property_dim = tuple(range(1, y_true.dim()))
    num_structures = y_true.shape[0]

    # scalar
    mae = F.l1_loss(y_pred, y_true)
    # scalar
    mse = F.mse_loss(y_pred, y_true)
    # scalar
    rmse = torch.sqrt(mse)
    # scalar
    # mean() calculates mean over all dimensions
    pointwise_mae = (y_pred - y_true).reshape(-1).abs().mean()

    # (num_strctures,)
    # fnorm_error = torch.abs(
    #     torch.norm(y_pred, dim=property_dim) - torch.norm(y_true, dim=property_dim)
    # )
    fnorm_error = torch.norm(y_pred - y_true, dim=property_dim)
    # scalar
    mean_fnorm_error = fnorm_error.mean()
    # (num_strctures,)
    fnorm = torch.norm(y_true, dim=property_dim)
    # scalar
    mean_fnorm_percent_error = (fnorm_error / (fnorm + 1e-8)).mean() * 100

    # # Nonsense
    # batchwise_rsse = (y_pred - y_true).view(-1).pow(2).sum().sqrt()
    # batchwise_sum_fnorm = y_true.view(-1).pow(2).sum().sqrt()
    # batchwise_percent_error = batchwise_rsse / (batchwise_sum_fnorm + 1e-8)
    # (num_strctures,)
    sse = ((y_pred - y_true) ** 2).sum(dim=property_dim)
    # # (num_strctures,)
    # -----------------------------------------
    # percent_error = sse / (fnorm ** 2 + 1e-8)
    percent_error = (fnorm_error / (fnorm + 1e-8))
    # -----------------------------------------
    # High Quality Prediction Rate (EwT)
    # scalar
    EwT_25 = (percent_error < 0.25).sum() / num_structures * 100
    # scalar
    EwT_10 = (percent_error < 0.1).sum() / num_structures * 100
    # scalar
    EwT_5 = (percent_error < 0.05).sum() / num_structures * 100

    # Mean Absolute Percentage Error (MAPE)
    epsilon = 1e-8
    mape = torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    # # Nonsense
    # ss_res = torch.sum((y_true - y_pred) ** 2)
    # ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    # r2_score = 1 - (ss_res / (ss_tot + 1e-8))

    return {
        "mae": mae.item(),
        "mse": mse.item(),
        "rmse": rmse.item(),
        "pointwise_mae": pointwise_mae.item(),
        "mean_fnorm_error": mean_fnorm_error.item(),
        "mean_fnorm_percent_error": mean_fnorm_percent_error.item(),
        "mape": mape.item(),
        "EwT_25": EwT_25.item(),
        "EwT_10": EwT_10.item(),
        "EwT_5": EwT_5.item(),
    }


def tensor_test(
    tensor_models: dict[str, nn.Module],
    tensor_dataloaders: dict[str, DataLoader],
    pic_dir: str,
    metric_dir: str = "metrics",
    train_history: dict = None,
):
    """
    Test tensor property prediction models.

    Args:
        tensor_models: Dictionary of trained models keyed by property name
        tensor_dataloaders: Dictionary of dataloaders for each property
        pic_dir: Directory to save plots (a subfolder with timestamp will be created)
        metric_dir: Directory to save metrics
        train_history: Dictionary containing training history for each property
                      with keys like 'train_loss', 'val_loss', etc.

    Returns:
        dict: Results for each property including metrics and predictions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for key in tensor_models:
        tensor_models[key] = tensor_models[key].to(device)
        tensor_models[key].eval()

    results = {}

    vis_dir = get_visualization_dir(pic_dir)

    with torch.no_grad():
        for prop in tensor_properties:
            testset = tensor_dataloaders[f"{prop}_testset"]

            all_true_values = []
            all_pred_values = []
            total_loss = 0.0
            num_batches = 0

            for batch in testset:
                atom_type = batch.atom_type.to(device)
                edge_index = batch.edge_index.to(device)
                edge_vec = batch.edge_vec.to(device)
                batch_index = batch.batch.to(device)
                tensor_property = batch.tensor_property.to(device)

                feat_mask = batch.feat_mask.to(device) if hasattr(batch, "feat_mask") else None
                equality = batch.equality.to(device) if hasattr(batch, "equality") else None

                pred_tensor_property = tensor_models[prop](
                    atom_type, edge_vec, edge_index, batch_index, feat_mask, equality
                )

                # # For debugging
                # print(f"pred_tensor_property: {pred_tensor_property}")
                # print(f"tensor_property: {tensor_property}")
                # print(f"pred_tensor_property shape: {pred_tensor_property.shape}")
                # print(f"tensor_property shape: {tensor_property.shape}")
                # exit()

                all_true_values.append(tensor_property.cpu())
                all_pred_values.append(pred_tensor_property.cpu())

                batch_loss = F.mse_loss(pred_tensor_property, tensor_property)
                total_loss += batch_loss.item()
                num_batches += 1

            all_true_values = torch.cat(all_true_values, dim=0)
            all_pred_values = torch.cat(all_pred_values, dim=0)
            
            # print("="*100)
            # print(all_true_values.shape)
            # print("="*100)
            # assert (torch.tensor(all_pred_values) == torch.zeros_like(all_pred_values)).all()

            metrics = calculate_tensor_metrics(all_true_values, all_pred_values)
            avg_loss = total_loss / num_batches

            print(f"\nTensor Test Results for {prop} (without mask):")
            print(f"  Average Loss: {avg_loss:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  Pointwise MAE: {metrics['pointwise_mae']:.6f}")
            print(f"  Mean FNORM Error: {metrics['mean_fnorm_error']:.6f}")
            print(f"  Mean FNORM % Error: {metrics['mean_fnorm_percent_error']:.6f}%")
            print(f"  MAPE: {metrics['mape']:.6f}%")
            print(f"  EwT_25: {metrics['EwT_25']:.2f}%")
            print(f"  EwT_10: {metrics['EwT_10']:.2f}%")
            print(f"  EwT_5: {metrics['EwT_5']:.2f}%")

            # For experiment: mask out the zero values
            zero_mask = torch.ones_like(all_true_values)
            zero_mask[all_true_values == 0] = 0
            all_pred_values = all_pred_values * zero_mask

            metrics = calculate_tensor_metrics(all_true_values, all_pred_values)
            avg_loss = total_loss / num_batches

            print(f"\nTensor Test Results for {prop} (with mask):")
            print(f"  Average Loss: {avg_loss:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  Pointwise MAE: {metrics['pointwise_mae']:.6f}")
            print(f"  Mean FNORM Error: {metrics['mean_fnorm_error']:.6f}")
            print(f"  Mean FNORM % Error: {metrics['mean_fnorm_percent_error']:.6f}%")
            print(f"  MAPE: {metrics['mape']:.6f}%")
            print(f"  EwT_25: {metrics['EwT_25']:.2f}%")
            print(f"  EwT_10: {metrics['EwT_10']:.2f}%")
            print(f"  EwT_5: {metrics['EwT_5']:.2f}%")

            results[prop] = {
                "avg_loss": avg_loss,
                "metrics": metrics,
                "true_values": all_true_values,
                "predicted_values": all_pred_values,
            }

            property_vis_dir = os.path.join(vis_dir, prop)

            plot_prediction_scatter(
                y_true=all_true_values,
                y_pred=all_pred_values,
                save_dir=property_vis_dir,
                property_name=prop,
                title=f"{prop} - Test: Prediction vs True Values",
                filename=f"{prop}_test_prediction_scatter.png",
                color_by_component=True,
            )

            plot_fnorm_scatter(
                y_true=all_true_values,
                y_pred=all_pred_values,
                save_dir=property_vis_dir,
                property_name=prop,
                title=f"{prop} - Test: Frobenius Norm: Prediction vs True Values",
                filename=f"{prop}_test_fnorm_scatter.png",
            )

            if train_history is not None and prop in train_history:
                prop_history = train_history[prop]

                train_metrics = {
                    "train_loss": prop_history.get("train_losses", []),
                    "train_mae": prop_history.get("train_mae", []),
                    "train_pointwise_mae": prop_history.get("train_pointwise_mae", []),
                    "train_mse": prop_history.get("train_mse", []),
                    "train_mean_fnorm_percent_error": prop_history.get(
                        "train_mean_fnorm_percent_error", []
                    ),
                }

                val_metrics = {
                    "val_loss": prop_history.get("val_losses", []),
                    "val_mae": prop_history.get("val_mae_scores", []),
                    "val_pointwise_mae": prop_history.get("val_pointwise_mae", []),
                    "val_mse": prop_history.get("val_mse_scores", []),
                    "val_mean_fnorm_percent_error": prop_history.get(
                        "val_mean_fnorm_percent_error", []
                    ),
                }

                test_metrics = {
                    "loss": avg_loss,
                    "mae": metrics["mae"],
                    "pointwise_mae": metrics["pointwise_mae"],
                    "mse": metrics["mse"],
                    "mean_fnorm_percent_error": metrics["mean_fnorm_percent_error"],
                }

                plot_all_train_val_test_metrics(
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    test_metrics=test_metrics,
                    save_dir=property_vis_dir,
                    property_name=prop,
                )

    return results
