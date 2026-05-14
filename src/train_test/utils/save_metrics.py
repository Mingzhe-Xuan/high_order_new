import datetime
import os

def save_results_to_markdown(
    params: dict,
    scalar_results: dict,
    tensor_results: dict,
    metric_dir: str,
):
    """
    Save parameter information, scalar_results, and tensor_results to a markdown file.
    """
    # Create a timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_results_{timestamp}.md"
    os.makedirs(metric_dir, exist_ok=True)
    output_file = os.path.join(metric_dir, filename)
    
    with open(output_file, "w", encoding="utf-8") as f:
        # Write title and timestamp
        f.write("# Experiment Results Report\n\n")
        f.write(
            f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        # Write Parameters Section
        f.write("## Parameters\n\n")
        f.write("| Parameter | Value |\n")
        f.write("| --- | --- |\n")

        # Define parameters to include in the report
        param_keys = [
            "cutoff",
            "batch_size",
            "pin_memory",
            "num_workers",
            "seed",
            "train_val_test",
            "dist_emb_func",
            "embed_dim",
            "max_atom_type",
            "inv_update_method",
            "num_inv_layers",
            "middle_scalar_hidden_dim",
            "num_middle_hidden_layers",
            "equi_update_method",
            "num_equi_layers",
            "tp_method",
            "scalar_dim",
            "vec_dim",
            "num_final_hidden_layers",
            "final_scalar_hidden_dim",
            "final_vec_hidden_dim",
            "final_scalar_out_dim",
            "final_vec_out_dim",
            "need_self_train",
            "need_scalar_train",
            "need_tensor_train",
            "final_pooling",
            "num_epochs",
            "lr",
            "weight_decay",
            "clip_grad_norm",
            "save_interval",
            "optimizer",
            "scheduler",
            "self_loss_func",
            "scalar_loss_func",
            "tensor_loss_func",
            "self_train_limit",
            "scalar_train_limit",
            "tensor_train_limit",
            "checkpoint_dir",
            "pic_dir",
            "metric_dir",
            "start_epoch",
        ]

        for key in param_keys:
            if key in params:
                value = params[key]
                # Format complex objects nicely
                if isinstance(value, (list, tuple)):
                    value = str(list(value))
                elif isinstance(value, (dict, set)):
                    value = (
                        str(dict(value)) if isinstance(value, dict) else str(set(value))
                    )
                f.write(f"| {key} | {value} |\n")

        # Write Scalar Results Section
        f.write("\n## Scalar Model Results\n\n")
        if scalar_results:
            for prop, result in scalar_results.items():
                f.write(f"### Property: {prop}\n\n")
                f.write(f"- Average Loss: {result.get('avg_loss', 'N/A'):.6f}\n")

                metrics = result.get("metrics", {})
                if metrics:
                    f.write("- Metrics:\n")
                    for metric_name, metric_value in metrics.items():
                        f.write(f"  - {metric_name}: {metric_value:.6f}\n")

                f.write("\n")
        else:
            f.write("No scalar results available.\n\n")

        # Write Tensor Results Section
        f.write("## Tensor Model Results\n\n")
        if tensor_results:
            for prop, result in tensor_results.items():
                f.write(f"### Property: {prop}\n\n")
                f.write(f"- Average Loss: {result.get('avg_loss', 'N/A'):.6f}\n")

                metrics = result.get("metrics", {})
                if metrics:
                    f.write("- Metrics:\n")
                    for metric_name, metric_value in metrics.items():
                        f.write(f"  - {metric_name}: {metric_value:.6f}\n")

                f.write("\n")
        else:
            f.write("No tensor results available.\n\n")

        print(f"Results saved to {output_file}")
