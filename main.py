import argparse
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

from src.main import main, seed_everything, worker_init_fn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run high_order training and evaluation.")

    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-val-test", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--dist-emb-func", default="gaussian")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--max-atom-type", type=int, default=118)
    parser.add_argument("--inv-update-method", default="comformer")
    parser.add_argument("--num-inv-layers", type=int, default=3)
    parser.add_argument("--middle-scalar-hidden-dim", type=int, default=128)
    parser.add_argument("--num-middle-hidden-layers", type=int, default=1)
    parser.add_argument("--equi-update-method", default="tpconv_with_edge")
    parser.add_argument("--num-equi-layers", type=int, default=4)
    parser.add_argument("--tp-method", default="so2")
    parser.add_argument("--scalar-dim", type=int, default=16)
    parser.add_argument("--vec-dim", type=int, default=8)
    parser.add_argument("--num-final-hidden-layers", type=int, default=1)
    parser.add_argument("--final-scalar-hidden-dim", type=int, default=64)
    parser.add_argument("--final-vec-hidden-dim", type=int, default=16)
    parser.add_argument("--final-scalar-out-dim", type=int, default=16)
    parser.add_argument("--final-vec-out-dim", type=int, default=8)
    parser.add_argument("--need-self-train", "--self-train", dest="need_self_train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--need-scalar-train", "--scalar-train", dest="need_scalar_train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--need-tensor-train", "--tensor-train", dest="need_tensor_train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--final-pooling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-num-epochs", type=int, default=350)
    parser.add_argument("--scalar-num-epochs", type=int, default=400)
    parser.add_argument("--tensor-num-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--scheduler", default="cosine_warm_restarts")
    parser.add_argument("--warmup-periods", type=int, default=10)
    parser.add_argument("--self-loss-func", default="huber")
    parser.add_argument("--scalar-loss-func", default="huber")
    parser.add_argument("--tensor-loss-func", default="huber")
    parser.add_argument("--self-train-limit", type=int, default=None)
    parser.add_argument("--scalar-train-limit", type=int, default=None)
    parser.add_argument("--tensor-train-limit", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--pic-dir", default="pics")
    parser.add_argument("--metric-dir", default="metrics")
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--resume-self-train", default=None)
    parser.add_argument("--resume-scalar-train", default=None)
    parser.add_argument("--resume-tensor-train", default=None)
    parser.add_argument("--freeze", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--share-middle-mlp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scalar-invariant-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--only-use-embedding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model-type", "--model_type", "--model", dest="model_type", default="high_order")
    parser.add_argument("--graph-mode", choices=("high_order", "gmtnet"), default=None)
    parser.add_argument("--max-neighbors", type=int, default=12)
    parser.add_argument("--gmtnet-embed-dim", type=int, default=128)
    parser.add_argument("--gmtnet-num-attention-layers", type=int, default=2)
    parser.add_argument("--use-tensorboard", action=argparse.BooleanOptionalAction, default=True)

    return parser


def _apply_model_defaults(kwargs: dict, parser: argparse.ArgumentParser) -> None:
    if kwargs["model_type"] != "gmtnet":
        return

    defaults = {
        "train_batch_size": 64,
        "val_batch_size": 64,
        "test_batch_size": 1,
        "seed": 32,
        "cutoff": 4.0,
        "tensor_num_epochs": 200,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "tensor_loss_func": "huber",
        "graph_mode": "gmtnet",
        "max_neighbors": 12,
        "gmtnet_embed_dim": 128,
        "gmtnet_num_attention_layers": 2,
    }
    parser_defaults = {action.dest: action.default for action in parser._actions}
    for key, value in defaults.items():
        if key == "graph_mode":
            if kwargs[key] is None:
                kwargs[key] = value
        elif kwargs[key] == parser_defaults[key]:
            kwargs[key] = value


def cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs["train_val_test"] = tuple(kwargs["train_val_test"])
    _apply_model_defaults(kwargs, parser)
    if kwargs["graph_mode"] is None:
        kwargs["graph_mode"] = "high_order"
    seed_everything(kwargs["seed"])
    main(**kwargs)


if __name__ == "__main__":
    cli()


__all__ = ["main", "seed_everything", "worker_init_fn", "build_parser", "cli"]
