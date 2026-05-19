# Experiment Results Report

Generated on: 2026-05-19 23:51:00

## Parameters

| Parameter | Value |
| --- | --- |
| cutoff | 4.0 |
| pin_memory | True |
| num_workers | 4 |
| seed | 32 |
| train_val_test | [0.8, 0.1, 0.1] |
| dist_emb_func | gaussian |
| embed_dim | 64 |
| max_atom_type | 118 |
| inv_update_method | comformer |
| num_inv_layers | 3 |
| middle_scalar_hidden_dim | 128 |
| num_middle_hidden_layers | 1 |
| equi_update_method | tpconv_with_edge |
| num_equi_layers | 4 |
| tp_method | so2 |
| scalar_dim | 16 |
| vec_dim | 8 |
| num_final_hidden_layers | 1 |
| final_scalar_hidden_dim | 64 |
| final_vec_hidden_dim | 16 |
| final_scalar_out_dim | 16 |
| final_vec_out_dim | 8 |
| need_self_train | False |
| need_scalar_train | False |
| need_tensor_train | True |
| final_pooling | True |
| lr | 0.001 |
| weight_decay | 1e-05 |
| clip_grad_norm | 1.0 |
| save_interval | 5 |
| optimizer | adamw |
| scheduler | cosine_annealing |
| self_loss_func | huber |
| scalar_loss_func | huber |
| tensor_loss_func | huber |
| self_train_limit | None |
| scalar_train_limit | None |
| tensor_train_limit | None |
| checkpoint_dir | checkpoints |
| pic_dir | pics |
| metric_dir | metrics |
| start_epoch | 0 |

## Scalar Model Results

No scalar results available.

## Tensor Model Results

### Property: dielectric

- Average Loss: 18.343648
- Metrics:
  - mae: 0.792570
  - mse: 18.343639
  - rmse: 4.282948
  - pointwise_mae: 0.792570
  - mean_fnorm_error: 4.404497
  - mean_fnorm_percent_error: 20.968929
  - mape: 5349.732422
  - EwT_25: 81.991524
  - EwT_10: 46.398308
  - EwT_5: 22.457626

