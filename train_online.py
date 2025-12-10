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
    # Observation normalization (PPO only)
    parser.add_argument("--obs_norm", action="store_true")
    parser.add_argument("--obs_norm_clip", type=float, default=10.0)
    # SW-PPO pessimism knobs
    parser.add_argument("--pess_alpha0", type=float, default=0.0, help="Initial pessimism coefficient")
    parser.add_argument("--pess_alpha_final", type=float, default=0.0, help="Final pessimism coefficient")
    parser.add_argument("--pess_anneal_steps", type=float, default=100000, help="Anneal steps for pessimism")
    parser.add_argument("--pess_gamma", type=float, default=1.0, help="Exponent for inverse-support in pessimism")
    parser.add_argument("--adv_gate_tau", type=float, default=0.5, help="Support threshold for actor gating")
    parser.add_argument("--adv_gate_k", type=float, default=0.0, help="Steepness of actor gate (0 disables)")
    # bonus
    parser.add_argument("--bonus_eta", type=float, default=0.1)
    parser.add_argument("--bonus_center", type=float, default=0.7)
    parser.add_argument("--bonus_sigma", type=float, default=0.15)
    parser.add_argument("--bonus_type", type=str, default="boundary", choices=["boundary", "entropy", "ucb"])  # new
    # adaptive knobs
    parser.add_argument("--adaptive_bonus", action="store_true", help="Adapt bonus with performance")
    parser.add_argument("--bonus_alpha", type=float, default=1.0, help="Scale for adaptive bonus")
    parser.add_argument("--target_return", type=float, default=200.0, help="Target return for scaling")
    # baselines & init
    parser.add_argument("--no_pessimism", action="store_true")
    parser.add_argument("--no_bonus", action="store_true")
    parser.add_argument("--init_actor", type=str, default=None, help="Path to actor weights (BC pretrain)")
    # PPO stability + trust-region
    parser.add_argument("--ent_coeff", type=float, default=0.0)
    parser.add_argument("--target_kl", type=float, default=0.01)
    parser.add_argument("--vf_clip", type=float, default=0.0, help="Value clip (0 disables)")
    parser.add_argument("--actor_grad_clip", type=float, default=0.0)
    parser.add_argument("--critic_grad_clip", type=float, default=0.0)
    parser.add_argument("--support_adaptive_clip", action="store_true")
    # BC anchor schedule
    parser.add_argument("--kl_bc_coef", type=float, default=0.0, help="Fixed BC anchor (if no schedule)")
    parser.add_argument("--kl_bc_pow", type=float, default=1.0)
    parser.add_argument("--kl_bc_coef0", type=float, default=0.0)
    parser.add_argument("--kl_bc_coef_final", type=float, default=0.0)
    parser.add_argument("--kl_bc_anneal_steps", type=float, default=0.0)
    # early stopping and logging
    parser.add_argument("--early_stop_avg_return", type=float, default=None)
    parser.add_argument("--early_stop_window", type=int, default=10)
    parser.add_argument("--log_csv", type=str, default=None, help="logs/run.csv if set")
    parser.add_argument("--wandb", action="store_true", help="Log training to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="o2o")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None)
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
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
                config={
                    "env_id": args.env_id,
                    "dsr_path": args.dsr_path,
                    "total_steps": args.total_steps,
                    "steps_per_epoch": args.steps_per_epoch,
                    "minibatch_size": args.minibatch_size,
                    "train_iters": args.train_iters,
                    "hidden": args.hidden,
                    "activation": args.activation,
                    "obs_norm": args.obs_norm,
                    "pess_alpha0": args.pess_alpha0,
                    "pess_alpha_final": args.pess_alpha_final,
                    "pess_anneal_steps": args.pess_anneal_steps,
                    "pess_gamma": args.pess_gamma,
                    "adv_gate_tau": args.adv_gate_tau,
                    "adv_gate_k": args.adv_gate_k,
                    "bonus_eta": args.bonus_eta,
                    "bonus_center": args.bonus_center,
                    "bonus_sigma": args.bonus_sigma,
                    "bonus_type": args.bonus_type,
                    "adaptive_bonus": args.adaptive_bonus,
                    "target_return": args.target_return,
                    "no_pessimism": args.no_pessimism,
                    "no_bonus": args.no_bonus,
                    "init_actor": args.init_actor,
                    "ent_coeff": args.ent_coeff,
                    "target_kl": args.target_kl,
                    "vf_clip": args.vf_clip,
                    "actor_grad_clip": args.actor_grad_clip,
                    "critic_grad_clip": args.critic_grad_clip,
                    "support_adaptive_clip": args.support_adaptive_clip,
                    "kl_bc_coef": args.kl_bc_coef,
                    "kl_bc_pow": args.kl_bc_pow,
                    "kl_bc_coef0": args.kl_bc_coef0,
                    "kl_bc_coef_final": args.kl_bc_coef_final,
                    "kl_bc_anneal_steps": args.kl_bc_anneal_steps,
                    "seed": args.seed,
                },
            )
        except Exception as e:
            print(f"[wandb] init failed, continuing without wandb: {e}")
            wandb_run = None

    env = make_env(args.env_id, seed=args.seed)
    spec = get_space_spec(env)
    dsr = DSR.load(args.dsr_path, device=device)

    # Determine actor hidden sizes; if init_actor is provided and shapes differ,
    # infer hidden sizes from checkpoint to avoid size mismatch.
    hidden_actor = tuple(args.hidden)
    if args.init_actor:
        try:
            sd_probe = torch.load(args.init_actor, map_location="cpu")
            if isinstance(sd_probe, dict):
                w_items = []
                for k, v in sd_probe.items():
                    if k.startswith("net.") and k.endswith(".weight") and hasattr(v, "shape"):
                        parts = k.split(".")
                        if len(parts) >= 3 and parts[1].isdigit():
                            w_items.append((int(parts[1]), tuple(v.shape)))
                if w_items:
                    w_items.sort(key=lambda x: x[0])
                    # hidden sizes are out_features of all but last layer
                    hidden_from_sd = tuple(int(s[0]) for _, s in w_items[:-1])
                    if hidden_from_sd and hidden_from_sd != hidden_actor:
                        hidden_actor = hidden_from_sd
                        print(f"[train_online] Overriding actor hidden sizes to {hidden_actor} to match init_actor")
        except Exception as e:
            print(f"[train_online] Warning: could not inspect init_actor: {e}")

    if spec.discrete:
        actor = ActorDiscrete(spec.state_dim, spec.action_dim, hidden=hidden_actor, activation=args.activation)
    else:
        low = np.array(spec.action_low_vec, dtype=np.float32)
        high = np.array(spec.action_high_vec, dtype=np.float32)
        actor = ActorGaussian(
            spec.state_dim,
            spec.action_dim,
            action_low=low,
            action_high=high,
            hidden=hidden_actor,
            activation=args.activation,
        )
    # Allow critic to use the user-provided hidden sizes independently
    critic = Critic(spec.state_dim, hidden=tuple(args.hidden), activation=args.activation)

    # Configure pessimism knobs (disable if --no_pessimism)
    if args.no_pessimism:
        pess_alpha0 = 0.0
        pess_alpha_final = 0.0
        adv_gate_k = 0.0  # disables gating
    else:
        pess_alpha0 = args.pess_alpha0
        pess_alpha_final = args.pess_alpha_final
        adv_gate_k = args.adv_gate_k
    use_bonus = not args.no_bonus

    # Set up a frozen reference actor (BC) for KL anchor if requested
    ref_actor = None
    if args.init_actor:
        if spec.discrete:
            ref_actor = ActorDiscrete(spec.state_dim, spec.action_dim, hidden=hidden_actor, activation=args.activation)
        else:
            ref_actor = ActorGaussian(
                spec.state_dim,
                spec.action_dim,
                action_low=np.array(spec.action_low_vec, dtype=np.float32),
                action_high=np.array(spec.action_high_vec, dtype=np.float32),
                hidden=hidden_actor,
                activation=args.activation,
            )
        sd_ref = torch.load(args.init_actor, map_location=device)
        ref_actor.load_state_dict(sd_ref)

    agent = PPOAgent(
        actor=actor,
        critic=critic,
        dsr=dsr,
        ref_actor=ref_actor,
        discrete=spec.discrete,
        action_dim=spec.action_dim,
        device=device,
        lr_actor=3e-4,
        lr_critic=3e-4,
        clip_ratio=0.2,
        target_kl=args.target_kl,
        vf_coeff=0.5,
        ent_coeff=args.ent_coeff,
        gamma=0.99,
        gae_lambda=0.95,
        pess_alpha0=pess_alpha0,
        pess_alpha_final=pess_alpha_final,
        pess_anneal_steps=args.pess_anneal_steps,
        pess_gamma=args.pess_gamma,
        adv_gate_tau=args.adv_gate_tau,
        adv_gate_k=adv_gate_k,
        bonus_eta=args.bonus_eta,
        bonus_center=args.bonus_center,
        bonus_sigma=args.bonus_sigma,
        use_bonus=use_bonus,
        bonus_type=args.bonus_type,
        kl_bc_coef=args.kl_bc_coef,
        kl_bc_pow=args.kl_bc_pow,
        kl_bc_coef0=args.kl_bc_coef0,
        kl_bc_coef_final=args.kl_bc_coef_final,
        kl_bc_anneal_steps=args.kl_bc_anneal_steps,
        actor_grad_clip=(args.actor_grad_clip if args.actor_grad_clip > 0 else None),
        critic_grad_clip=(args.critic_grad_clip if args.critic_grad_clip > 0 else None),
        vf_clip=(args.vf_clip if args.vf_clip > 0 else None),
        support_adaptive_clip=bool(args.support_adaptive_clip),
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
                "alpha_pess",
                "w_actor_mean",
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

    # Observation normalization for PPO (DSR uses raw obs)
    class RunningNorm:
        def __init__(self, shape: int, clip: float = 10.0):
            self.mean = np.zeros((shape,), dtype=np.float32)
            self.M2 = np.ones((shape,), dtype=np.float32)
            self.count = 1e-4
            self.clip = float(clip)
        def update(self, x: np.ndarray):
            x = np.asarray(x, dtype=np.float32)
            delta = x - self.mean
            self.count += 1.0
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.M2 += delta * delta2
        def std(self):
            return np.sqrt(self.M2 / max(1.0, self.count - 1.0))
        def normalize(self, x: np.ndarray):
            x = np.asarray(x, dtype=np.float32)
            s = self.std()
            x_n = (x - self.mean) / (s + 1e-8)
            return np.clip(x_n, -self.clip, self.clip)

    obs_rn = RunningNorm(spec.state_dim, clip=args.obs_norm_clip) if args.obs_norm else None
    if obs_rn is not None:
        obs_rn.update(obs)
        obs_p = obs_rn.normalize(obs)
    else:
        obs_p = obs

    bonus = 0.0
    buf = {k: [] for k in ["obs", "act", "rew", "val", "logp", "done", "support", "support_std"]}
    t = 0
    while t < total_steps:
        # collect rollout
        for _ in range(steps_per_epoch):
            # if observation is invalid, reset episode to avoid NaNs propagating
            if not np.all(np.isfinite(obs)):
                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                ep_ret, ep_len = 0.0, 0
                continue

            # normalize obs for PPO
            if obs_rn is not None:
                obs_rn.update(obs)
                obs_p = obs_rn.normalize(obs)
            else:
                obs_p = obs

            a, logp, ent = agent.select_action(obs_p)
            # guard invalid actions and clip to bounds for Box spaces
            if not spec.discrete:
                low = np.array(spec.action_low_vec, dtype=np.float32)
                high = np.array(spec.action_high_vec, dtype=np.float32)
                if isinstance(a, np.ndarray):
                    if not np.all(np.isfinite(a)):
                        a = np.zeros_like(a, dtype=np.float32)
                    a = np.clip(a, low, high)
                else:
                    if not np.isfinite(a):
                        a = 0.0
                    a = float(np.clip(a, low, high))
            else:
                if not np.isfinite(a):
                    a = int(0)
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
            bonus_arr = agent.compute_bonus(
                np.array([sup_mean], dtype=np.float32), np.array([sup_std], dtype=np.float32)
            )
            bonus = float(max(0.0, float(bonus_arr[0])))

            # scale bonus by running reward std
            rew_std = float(np.sqrt(rew_var / max(1.0, rew_cnt - 1.0)))
            # optional adaptive scaling by performance
            if args.adaptive_bonus and len(ep_returns) > 0:
                avg_return = float(np.mean(ep_returns))
                perf_scale = max(0.1, 1.0 - avg_return / max(1e-6, args.target_return))
                bonus *= (args.bonus_alpha * perf_scale)
            bonus = float(np.clip(bonus, 0.0, 0.5 * max(1e-6, rew_std)))
            r_total = float(r + bonus)

            v = agent.evaluate_value(obs_p)
            buf["obs"].append(obs_p)
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

            # If next_obs is invalid, treat as episode end and reset
            if not np.all(np.isfinite(next_obs)):
                ep_returns.append(ep_ret)
                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                ep_ret, ep_len = 0.0, 0
            else:
                obs = next_obs
            if done or truncated:
                ep_returns.append(ep_ret)
                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                ep_ret, ep_len = 0.0, 0

            if t >= total_steps:
                break

        # compute GAE advantages
        # bootstrap value from normalized obs
        if obs_rn is not None:
            obs_p = obs_rn.normalize(obs)
        else:
            obs_p = obs
        last_val = agent.evaluate_value(obs_p)
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
            "v_old": np.array(buf["val"], dtype=np.float32),
        }
        # filter out any rows with non-finite values to avoid NaNs in updates
        mask = np.isfinite(batch["obs"]).all(axis=1)
        mask &= np.isfinite(batch["ret"]) & np.isfinite(batch["adv"]) & np.isfinite(batch["logp"]) & np.isfinite(batch["support"])
        if not spec.discrete:
            mask &= np.isfinite(batch["act"]).all(axis=-1)
        else:
            mask &= np.isfinite(batch["act"]).astype(bool)
        if mask.sum() < len(mask):
            for k in list(batch.keys()):
                batch[k] = batch[k][mask]
            print(f"[train_online] Dropped {int((~mask).sum())} invalid samples before update")
        if batch["obs"].size == 0:
            print("[train_online] Warning: no valid samples this epoch; skipping update")
            buf = {k: [] for k in buf}
            continue
        metrics = agent.update(
            batch,
            minibatch_size=minibatch_size,
            train_iters=train_iters,
            current_total_steps=t,
        )

        # clear buffer
        buf = {k: [] for k in buf}

        avg_ret = np.mean(ep_returns) if len(ep_returns) else 0.0
        sup = batch["support"]
        sup_mean = float(np.mean(sup)) if sup.size else 0.0
        p10 = float(np.percentile(sup, 10)) if sup.size else 0.0
        p90 = float(np.percentile(sup, 90)) if sup.size else 0.0
        print(
            f"Steps {t}/{total_steps}  AvgEpRet {avg_ret:.2f}  DSR[mean/p10/p90] {sup_mean:.2f}/{p10:.2f}/{p90:.2f}  "
            f"alpha {metrics.get('alpha_pess', 0.0):.3f}  w_mean {metrics.get('w_actor_mean', 1.0):.3f}  "
            f"bonus_scale {0.5 * float(np.sqrt(rew_var / max(1.0, rew_cnt - 1.0))):.3f}"
        )
        if wandb_run:
            wandb_mod.log(
                {
                    "steps": t,
                    "avg_ep_ret": avg_ret,
                    "support/mean": sup_mean,
                    "support/p10": p10,
                    "support/p90": p90,
                    "alpha_pess": metrics.get("alpha_pess", 0.0),
                    "w_actor_mean": metrics.get("w_actor_mean", 0.0),
                    "v_mse": metrics.get("v_mse", 0.0),
                    "loss_actor": metrics.get("loss_actor", 0.0),
                    "bonus_used": bonus,
                    "rew_std": float(np.sqrt(rew_var / max(1.0, rew_cnt - 1.0))),
                    "approx_kl": metrics.get("approx_kl", 0.0),
                    "bc_alpha": metrics.get("bc_alpha", 0.0),
                },
                step=t,
            )

        if logger:
            logger.log(
                {
                    "steps": t,
                    "avg_ep_ret": avg_ret,
                    "support_mean": sup_mean,
                    "support_p10": p10,
                    "support_p90": p90,
                    "alpha_pess": metrics.get("alpha_pess", 0.0),
                    "w_actor_mean": metrics.get("w_actor_mean", 0.0),
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
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
