import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import numpy as np
from torch_geometric.loader import DataLoader
from typing import Union

from data import scalar_properties
from src.train_test.utils.visualization import (
    get_visualization_dir,
    plot_train_val_test_metrics,
    plot_prediction_scatter,
)


def calculate_metrics(y_true, y_pred, num_atoms, per_atom=False):
    """
    Calculate various metrics for model evaluation.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        dict: Dictionary containing various metrics
    """
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true)
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred)
    
    # mae = F.l1_loss(y_pred, y_true)
    # mse = F.mse_loss(y_pred, y_true)
    # rmse = torch.sqrt(mse)
    if per_atom:
        mae = torch.mean(torch.abs(y_pred - y_true) / num_atoms)
        mse = torch.mean((y_pred - y_true) ** 2 / num_atoms)
        rmse = torch.sqrt(mse)
    else:
        mae = F.l1_loss(y_pred, y_true)
        mse = F.mse_loss(y_pred, y_true)
        rmse = torch.sqrt(mse)
    
    epsilon = 1e-8
    mape = torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2_score = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'mae': mae.item(),
        'mse': mse.item(),
        'rmse': rmse.item(),
        'mape': mape.item(),
        'r2_score': r2_score.item()
    }


def scalar_test(
    scalar_models: dict[str, nn.Module],
    scalar_dataloaders: dict[str, DataLoader],
    pic_dir: str,
    metric_dir: str = "metrics",
    train_history: Union[dict, None] = None,
):
    """
    Test scalar property prediction models.
    
    Args:
        scalar_models: Dictionary of trained models keyed by property name
        scalar_dataloaders: Dictionary of dataloaders for each property
        pic_dir: Directory to save plots (a subfolder with timestamp will be created)
        metric_dir: Directory to save metrics
        train_history: Dictionary containing training history for each property
                      with keys like 'train_losses', 'val_losses', etc.
                      
    Returns:
        dict: Results for each property including metrics and predictions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for key in scalar_models:
        scalar_models[key] = scalar_models[key].to(device)
        scalar_models[key].eval()
    
    results = {}
    
    vis_dir = get_visualization_dir(pic_dir)

    with torch.no_grad():
        for prop in scalar_properties:
            testset = scalar_dataloaders[f"{prop}_testset"]
            
            all_true_values = []
            all_pred_values = []
            all_num_atoms = []
            total_loss = 0.0
            num_batches = 0

            for batch in testset:
                atom_type = batch.atom_type.to(device)
                edge_index = batch.edge_index.to(device)
                edge_vec = batch.edge_vec.to(device)
                batch_index = batch.batch.to(device)
                scalar_property = batch.scalar_property.to(device)
                num_atoms = torch.bincount(batch_index).to(device)
                
                pred_scalar_property = scalar_models[prop](atom_type, edge_vec, edge_index, batch_index)
                
                all_true_values.append(scalar_property.cpu())
                all_pred_values.append(pred_scalar_property.cpu())
                all_num_atoms.append(num_atoms.cpu())
                
                batch_loss = F.mse_loss(pred_scalar_property, scalar_property)
                total_loss += batch_loss.item()
                num_batches += 1
            
            all_true_values = torch.cat(all_true_values, dim=0)
            all_pred_values = torch.cat(all_pred_values, dim=0)
            all_num_atoms = torch.cat(all_num_atoms, dim=0)
            
            # energy in dataset has already calculated per-atom-ly
            # per_atom = True if prop in ["formation_energy", "total_energy", "e_form"]  else False
            per_atom = False
            
            metrics = calculate_metrics(all_true_values, all_pred_values, all_num_atoms, per_atom)
            avg_loss = total_loss / num_batches
            
            print(f"\nScalar Test Results for {prop}:")
            print(f"  Average Loss: {avg_loss:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  MAPE: {metrics['mape']:.6f}%")
            print(f"  R² Score: {metrics['r2_score']:.6f}")
            
            results[prop] = {
                'avg_loss': avg_loss,
                'metrics': metrics,
                'true_values': all_true_values,
                'predicted_values': all_pred_values
            }
            
            property_vis_dir = os.path.join(vis_dir, prop)
            
            plot_prediction_scatter(
                y_true=all_true_values,
                y_pred=all_pred_values,
                save_dir=property_vis_dir,
                property_name=prop,
                title=f"{prop} - Test: Prediction vs True Values",
                filename=f"{prop}_test_prediction_scatter.png",
            )
            
            if train_history is not None and prop in train_history:
                prop_history = train_history[prop]
                
                train_losses = prop_history.get("train_losses", [])
                train_mae = prop_history.get("train_mae", [])
                val_losses = prop_history.get("val_losses", [])
                val_mae_scores = prop_history.get("val_mae_scores", [])
                
                test_loss = avg_loss
                test_mae = metrics['mae']
                
                plot_train_val_test_metrics(
                    train_values=train_losses,
                    val_values=val_losses,
                    test_value=test_loss,
                    save_dir=property_vis_dir,
                    property_name=prop,
                    metric_name="loss",
                )
                
                plot_train_val_test_metrics(
                    train_values=train_mae,
                    val_values=val_mae_scores,
                    test_value=test_mae,
                    save_dir=property_vis_dir,
                    property_name=prop,
                    metric_name="mae",
                )

    return results
