import argparse
import os

import numpy as np
import torch

from o2o.config import DSRConfig
from o2o.datasets import load_npz_dataset
from o2o.models.dsr import DSR, DSRMetadata, DSRNet, train_dsr
from o2o.utils.envs import make_env, get_space_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline_path", type=str, required=True)
    parser.add_argument("--env_id", type=str, required=True)
    parser.add_argument("--dsr_out", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--num_negatives", type=int, default=1)
    parser.add_argument("--action_noise", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay (0 to disable)")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping norm (0 to disable)")
    # validation & calibration
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument(
        "--neg_modes",
        type=str,
        nargs="*",
        default=["uniform"],
        help="Negative sampling modes: uniform shuffle jitter",
    )
    parser.add_argument(
        "--neg_weights",
        type=float,
        nargs="*",
        default=None,
        help="Weights for negative modes (same length as neg_modes)",
    )
    parser.add_argument("--jitter_std", type=float, default=0.1, help="Std for jitter negatives (fraction of action range)")
    parser.add_argument("--calibrate_temperature", action="store_true", help="Calibrate post-hoc temperature on val split")
    # logging
    parser.add_argument("--wandb", action="store_true", help="Log training to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="o2o")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None)
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    offline = load_npz_dataset(args.offline_path, args.max_samples)
    env = make_env(args.env_id)
    spec = get_space_spec(env)

    wandb_run = None
    wandb_mod = None
    if args.wandb:
        try:
            import wandb as _wandb

            wandb_mod = _wandb
            wandb_run = wandb_mod.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                group=args.wandb_group,
                tags=args.wandb_tags,
                mode=args.wandb_mode,
                config=vars(args),
            )
        except Exception as e:
            print(f"[wandb] init failed, continuing without wandb: {e}")
            wandb_run = None

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    def log_cb(metrics: dict, step: int):
        if wandb_run and wandb_mod:
            # Drop None values to avoid noisy charts
            metrics_clean = {k: v for k, v in metrics.items() if v is not None}
            wandb_mod.log(metrics_clean, step=step)

    dsr = train_dsr(
        states=offline["states"],
        actions=offline["actions"],
        discrete=spec.discrete,
        action_dim=spec.action_dim,
        action_low=spec.action_low,
        action_high=spec.action_high,
        hidden=tuple(args.hidden),
        activation=args.activation,
        num_negatives=args.num_negatives,
        lr=3e-4,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
        action_noise=args.action_noise,
        temperature=args.temperature,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seed=args.seed,
        val_split=args.val_split,
        early_stop_patience=args.early_stop_patience,
        neg_modes=tuple(args.neg_modes),
        neg_weights=tuple(args.neg_weights) if args.neg_weights is not None else None,
        jitter_std=args.jitter_std,
        calibrate_temperature=bool(args.calibrate_temperature),
        log_callback=log_cb if wandb_run else None,
    )

    os.makedirs(os.path.dirname(args.dsr_out), exist_ok=True)
    torch.save(dsr.state_dict(), args.dsr_out)
    print(f"Saved DSR to {args.dsr_out}")
    if wandb_run and wandb_mod:
        try:
            art = wandb_mod.Artifact(f"dsr-{args.env_id}".replace("/", "-"), type="model")
            art.add_file(args.dsr_out)
            wandb_run.log_artifact(art)
        except Exception as e:
            print(f"[wandb] artifact log failed: {e}")
        wandb_run.finish()


if __name__ == "__main__":
    main()
