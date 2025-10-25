import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from o2o.datasets import load_npz_dataset
from o2o.utils.envs import make_env, get_space_spec
from o2o.models.ppo import ActorDiscrete, ActorGaussian


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline_path", type=str, required=True)
    parser.add_argument("--env_id", type=str, required=True)
    parser.add_argument("--out_actor", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy_coeff", type=float, default=0.01, help="Entropy regularization for discrete")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_cosine", action="store_true", help="Use cosine LR schedule")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    data = load_npz_dataset(args.offline_path)
    env = make_env(args.env_id)
    spec = get_space_spec(env)

    if spec.discrete:
        actor = ActorDiscrete(spec.state_dim, spec.action_dim, hidden=tuple(args.hidden), activation=args.activation)
    else:
        low = np.array(spec.action_low_vec, dtype=np.float32)
        high = np.array(spec.action_high_vec, dtype=np.float32)
        actor = ActorGaussian(
            spec.state_dim,
            spec.action_dim,
            action_low=low,
            action_high=high,
            hidden=tuple(args.hidden),
            activation=args.activation,
        )
    actor.to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)

    states = torch.as_tensor(data["states"], dtype=torch.float32, device=device)
    actions = data["actions"]
    if spec.discrete:
        actions = torch.as_tensor(actions, dtype=torch.long, device=device)
    else:
        actions = torch.as_tensor(actions, dtype=torch.float32, device=device)

    N = states.shape[0]
    steps_per_epoch = int(np.ceil(N / args.batch_size))
    # Optional LR scheduler
    scheduler = None
    if args.use_cosine:
        # T_max in steps (epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, args.epochs)
        )

    for epoch in range(args.epochs):
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        for it in range(steps_per_epoch):
            idx = perm[it * args.batch_size : (it + 1) * args.batch_size]
            s = states[idx]
            a = actions[idx]
            dist = actor(s)
            if spec.discrete:
                # maximize log-likelihood with entropy regularization
                logp = dist.log_prob(a)
                ent = dist.entropy().mean()
                loss = -(logp.mean() + args.entropy_coeff * ent)
            else:
                # maximize log likelihood under squashed Gaussian
                loss = -dist.log_prob(a).mean()
            opt.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
            opt.step()
            total_loss += loss.item() * s.shape[0]
        if scheduler is not None:
            scheduler.step()
        print(f"[BC] epoch {epoch} loss {total_loss / N:.4f}")

    dirn = os.path.dirname(args.out_actor)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    torch.save(actor.state_dict(), args.out_actor)
    print(f"Saved BC actor to {args.out_actor}")


if __name__ == "__main__":
    main()
