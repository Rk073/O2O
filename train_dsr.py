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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--num_negatives", type=int, default=1)
    parser.add_argument("--action_noise", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    offline = load_npz_dataset(args.offline_path, args.max_samples)
    env = make_env(args.env_id)
    spec = get_space_spec(env)

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
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
    )

    os.makedirs(os.path.dirname(args.dsr_out), exist_ok=True)
    torch.save(dsr.state_dict(), args.dsr_out)
    print(f"Saved DSR to {args.dsr_out}")


if __name__ == "__main__":
    main()

