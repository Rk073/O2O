import argparse
import os
from collections import deque

import numpy as np
import torch

from o2o.config import PPOConfig
from o2o.models.dsr import DSR
from o2o.models.ppo import PPOAgent, ActorDiscrete, ActorGaussian, Critic
from o2o.utils.envs import make_env, get_space_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, required=True)
    parser.add_argument("--dsr_path", type=str, required=True)
    parser.add_argument("--total_steps", type=int, default=200000)
    parser.add_argument("--steps_per_epoch", type=int, default=4096)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--train_iters", type=int, default=10)
    parser.add_argument("--hidden", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    # pessimism and bonus
    parser.add_argument("--pessimism_beta", type=float, default=1.0)
    parser.add_argument("--pessimism_gamma", type=float, default=1.0)
    parser.add_argument("--bonus_eta", type=float, default=0.1)
    parser.add_argument("--bonus_center", type=float, default=0.7)
    parser.add_argument("--bonus_sigma", type=float, default=0.15)
    parser.add_argument("--bonus_type", type=str, default="boundary", choices=["boundary", "entropy", "ucb"])  # new
    parser.add_argument("--use_bonus", action="store_true")
    # baselines & init
    parser.add_argument("--no_pessimism", action="store_true")
    parser.add_argument("--no_bonus", action="store_true")
    parser.add_argument("--init_actor", type=str, default=None, help="Path to actor weights (BC pretrain)")
    # early stopping and logging
    parser.add_argument("--early_stop_avg_return", type=float, default=None)
    parser.add_argument("--early_stop_window", type=int, default=10)
    parser.add_argument("--log_csv", type=str, default=None, help="logs/run.csv if set")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    env = make_env(args.env_id, seed=args.seed)
    spec = get_space_spec(env)
    dsr = DSR.load(args.dsr_path, device=device)

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
    critic = Critic(spec.state_dim, hidden=tuple(args.hidden), activation=args.activation)

    pess_beta = 0.0 if args.no_pessimism else args.pessimism_beta
    use_bonus = False if args.no_bonus else bool(args.use_bonus)

    agent = PPOAgent(
        actor=actor,
        critic=critic,
        dsr=dsr,
        discrete=spec.discrete,
        action_dim=spec.action_dim,
        device=device,
        lr_actor=3e-4,
        lr_critic=3e-4,
        clip_ratio=0.2,
        target_kl=0.01,
        vf_coeff=0.5,
        ent_coeff=0.0,
        gamma=0.99,
        gae_lambda=0.95,
        pessimism_beta=pess_beta,
        pessimism_gamma=args.pessimism_gamma,
        bonus_eta=args.bonus_eta,
        bonus_center=args.bonus_center,
        bonus_sigma=args.bonus_sigma,
        use_bonus=use_bonus,
        bonus_type=args.bonus_type,
    )

    # optional init from BC
    if args.init_actor:
        sd = torch.load(args.init_actor, map_location=device)
        agent.actor.load_state_dict(sd)

    # reset supports both gym and gymnasium
    reset_out = env.reset(seed=args.seed)
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    ep_ret, ep_len = 0.0, 0
    ep_returns = deque(maxlen=10)

    # running stats for reward scaling
    rew_mean, rew_var, rew_cnt = 0.0, 1.0, 1e-6

    def update_running_stats(x: float):
        nonlocal rew_mean, rew_var, rew_cnt
        rew_cnt += 1.0
        delta = x - rew_mean
        rew_mean += delta / rew_cnt
        rew_var += delta * (x - rew_mean)

    # logger
    logger = None
    if args.log_csv:
        from o2o.utils.logger import CSVLogger
        import time, os
        log_path = args.log_csv
        if os.path.isdir(log_path) or log_path.endswith(os.sep):
            ts = int(time.time())
            os.makedirs(log_path, exist_ok=True)
            log_path = os.path.join(log_path, f"run_{ts}.csv")
        logger = CSVLogger(
            log_path,
            [
                "steps",
                "avg_ep_ret",
                "support_mean",
                "support_p10",
                "support_p90",
                "pess_reg",
                "v_mse",
                "loss_actor",
                "bonus_used",
                "rew_std",
            ],
        )

    total_steps = args.total_steps
    steps_per_epoch = args.steps_per_epoch
    minibatch_size = args.minibatch_size
    train_iters = args.train_iters

    buf = {k: [] for k in ["obs", "act", "rew", "val", "logp", "done", "support", "support_std"]}
    t = 0
    while t < total_steps:
        # collect rollout
        for _ in range(steps_per_epoch):
            a, logp, ent = agent.select_action(obs)
            step_out = env.step(a)
            if len(step_out) == 5:
                next_obs, r, done, truncated, _ = step_out
            else:
                # gym API fallback
                next_obs, r, done, info = step_out
                truncated = info.get("TimeLimit.truncated", False) if isinstance(info, dict) else False

            # compute support (and std via jitter for continuous)
            if not spec.discrete:
                a_vec = np.array(a, dtype=np.float32)
                low = np.array(spec.action_low_vec, dtype=np.float32)
                high = np.array(spec.action_high_vec, dtype=np.float32)
                K = 5
                eps = 0.05 * (high - low)
                a_jit = np.clip(
                    a_vec[None, ...]
                    + np.random.uniform(-eps, eps, size=(K, a_vec.shape[0])).astype(np.float32),
                    low,
                    high,
                )
                s_rep = np.repeat(obs[None, ...], K, axis=0)
                with torch.no_grad():
                    sup_batch = (
                        agent.dsr.support(
                            torch.as_tensor(s_rep, dtype=torch.float32, device=agent.device),
                            torch.as_tensor(a_jit, dtype=torch.float32, device=agent.device),
                        )
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                sup_mean = float(np.mean(sup_batch))
                sup_std = float(np.std(sup_batch))
            else:
                sup_mean = agent.compute_support(obs, np.array([a]))
                sup_std = 0.0

            # intrinsic bonus based on type
            bonus_arr = agent.compute_bonus(np.array([sup_mean], dtype=np.float32), np.array([sup_std], dtype=np.float32))
            bonus = float(max(0.0, float(bonus_arr[0])))

            # scale bonus by running reward std
            rew_std = float(np.sqrt(rew_var / max(1.0, rew_cnt - 1.0)))
            bonus = float(np.clip(bonus, 0.0, 0.5 * max(1e-6, rew_std)))
            r_total = float(r + bonus)

            v = agent.evaluate_value(obs)
            buf["obs"].append(obs)
            buf["act"].append(a)
            buf["rew"].append(r_total)
            buf["val"].append(v)
            buf["logp"].append(logp)
            buf["done"].append(float(done or truncated))
            buf["support"].append(sup_mean)
            buf["support_std"].append(sup_std)

            # update running stats on extrinsic reward only
            update_running_stats(float(r))

            ep_ret += r
            ep_len += 1
            t += 1

            obs = next_obs
            if done or truncated:
                ep_returns.append(ep_ret)
                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                ep_ret, ep_len = 0.0, 0

            if t >= total_steps:
                break

        # compute GAE advantages
        last_val = agent.evaluate_value(obs)
        rewards = np.array(buf["rew"], dtype=np.float32)
        values = np.array(buf["val"], dtype=np.float32)
        dones = np.array(buf["done"], dtype=np.float32)
        values_plus = np.concatenate([values, np.array([last_val], dtype=np.float32)])
        adv, ret = agent.compute_advantages(rewards, values_plus, dones, agent.gamma, agent.gae_lambda)

        # pack buffer
        batch = {
            "obs": np.array(buf["obs"], dtype=np.float32),
            "act": np.array(buf["act"], dtype=np.int64 if spec.discrete else np.float32),
            "adv": adv.astype(np.float32),
            "ret": ret.astype(np.float32),
            "logp": np.array(buf["logp"], dtype=np.float32),
            "support": np.array(buf["support"], dtype=np.float32),
            "support_std": np.array(buf["support_std"], dtype=np.float32),
        }
        metrics = agent.update(batch, minibatch_size=minibatch_size, train_iters=train_iters)

        # clear buffer
        buf = {k: [] for k in buf}

        avg_ret = np.mean(ep_returns) if len(ep_returns) else 0.0
        sup = batch["support"]
        sup_mean = float(np.mean(sup)) if sup.size else 0.0
        p10 = float(np.percentile(sup, 10)) if sup.size else 0.0
        p90 = float(np.percentile(sup, 90)) if sup.size else 0.0
        print(
            f"Steps {t}/{total_steps}  AvgEpRet {avg_ret:.2f}  DSR[mean/p10/p90] {sup_mean:.2f}/{p10:.2f}/{p90:.2f}  "
            f"pess_reg {metrics['pess_reg']:.4f}  bonus_scale {0.5 * float(np.sqrt(rew_var / max(1.0, rew_cnt - 1.0))):.3f}"
        )

        if logger:
            logger.log(
                {
                    "steps": t,
                    "avg_ep_ret": avg_ret,
                    "support_mean": sup_mean,
                    "support_p10": p10,
                    "support_p90": p90,
                    "pess_reg": metrics.get("pess_reg", 0.0),
                    "v_mse": metrics.get("v_mse", 0.0),
                    "loss_actor": metrics.get("loss_actor", 0.0),
                    "bonus_used": bonus,
                    "rew_std": float(np.sqrt(rew_var / max(1.0, rew_cnt - 1.0))),
                }
            )

        # early stopping
        if args.early_stop_avg_return is not None and len(ep_returns) >= args.early_stop_window:
            if float(np.mean(ep_returns)) >= args.early_stop_avg_return:
                print(
                    f"Early stop reached: avg({args.early_stop_window}) >= {args.early_stop_avg_return}. Steps {t}."
                )
                break

    env.close()
    if logger:
        logger.close()


if __name__ == "__main__":
    main()
