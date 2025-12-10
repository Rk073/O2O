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
    parser.add_argument("--bonus_type", type=str, default="boundary", choices=["boundary", "entropy", "ucb"])
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
    parser.add_argument("--actor_freeze_steps", type=int, default=2048, help="Steps to train critic only before actor updates")
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
                project=args.wandb_project, name=args.wandb_run_name, group=args.wandb_group,
                tags=args.wandb_tags, mode=args.wandb_mode, config=vars(args)
            )
        except Exception as e:
            print(f"[wandb] init failed: {e}")

    env = make_env(args.env_id, seed=args.seed)
    spec = get_space_spec(env)
    dsr = DSR.load(args.dsr_path, device=device)

    # Determine actor hidden sizes
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
                    hidden_from_sd = tuple(int(s[0]) for _, s in w_items[:-1])
                    if hidden_from_sd and hidden_from_sd != hidden_actor:
                        hidden_actor = hidden_from_sd
                        print(f"[train_online] Overriding actor hidden sizes to {hidden_actor}")
        except Exception as e:
            print(f"[train_online] Warning: could not inspect init_actor: {e}")

    if spec.discrete:
        actor = ActorDiscrete(spec.state_dim, spec.action_dim, hidden=hidden_actor, activation=args.activation)
    else:
        low = np.array(spec.action_low_vec, dtype=np.float32)
        high = np.array(spec.action_high_vec, dtype=np.float32)
        actor = ActorGaussian(spec.state_dim, spec.action_dim, action_low=low, action_high=high, hidden=hidden_actor, activation=args.activation)
    
    critic = Critic(spec.state_dim, hidden=tuple(args.hidden), activation=args.activation)

    # Pessimism knobs
    if args.no_pessimism:
        pess_alpha0, pess_alpha_final = 0.0, 0.0
        adv_gate_k = 0.0
    else:
        pess_alpha0, pess_alpha_final = args.pess_alpha0, args.pess_alpha_final
        adv_gate_k = args.adv_gate_k
    use_bonus = not args.no_bonus

    # Reference actor (BC)
    ref_actor = None
    if args.init_actor:
        if spec.discrete:
            ref_actor = ActorDiscrete(spec.state_dim, spec.action_dim, hidden=hidden_actor, activation=args.activation)
        else:
            ref_actor = ActorGaussian(spec.state_dim, spec.action_dim, action_low=np.array(spec.action_low_vec, dtype=np.float32), action_high=np.array(spec.action_high_vec, dtype=np.float32), hidden=hidden_actor, activation=args.activation)
        sd_ref = torch.load(args.init_actor, map_location=device)
        ref_actor.load_state_dict(sd_ref)

    agent = PPOAgent(
        actor=actor, critic=critic, dsr=dsr, ref_actor=ref_actor, discrete=spec.discrete, action_dim=spec.action_dim, device=device,
        lr_actor=3e-4, lr_critic=3e-4, clip_ratio=0.2, target_kl=args.target_kl, vf_coeff=0.5, ent_coeff=args.ent_coeff,
        gamma=0.99, gae_lambda=0.95,
        pess_alpha0=pess_alpha0, pess_alpha_final=pess_alpha_final, pess_anneal_steps=args.pess_anneal_steps, pess_gamma=args.pess_gamma,
        adv_gate_tau=args.adv_gate_tau, adv_gate_k=adv_gate_k,
        bonus_eta=args.bonus_eta, bonus_center=args.bonus_center, bonus_sigma=args.bonus_sigma, use_bonus=use_bonus, bonus_type=args.bonus_type,
        kl_bc_coef=args.kl_bc_coef, kl_bc_pow=args.kl_bc_pow, kl_bc_coef0=args.kl_bc_coef0, kl_bc_coef_final=args.kl_bc_coef_final, kl_bc_anneal_steps=args.kl_bc_anneal_steps,
        actor_grad_clip=args.actor_grad_clip, critic_grad_clip=args.critic_grad_clip, vf_clip=args.vf_clip, support_adaptive_clip=bool(args.support_adaptive_clip),
    )

    if args.init_actor:
        agent.actor.load_state_dict(torch.load(args.init_actor))
        if not spec.discrete and hasattr(agent.actor, "log_std"):
            print("Resetting Actor LogStd for Online Exploration...")
            with torch.no_grad():
                # Set log_std to -0.5 (approx 0.6 std dev) to allow exploration
                agent.actor.log_std.fill_(-0.5)

    # Observation Normalization for PPO only
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
    
    def get_obs_p(raw_obs):
        if obs_rn is not None:
            obs_rn.update(raw_obs)
            return obs_rn.normalize(raw_obs)
        return raw_obs

    # Auto-Calibration of Bonus Center (heuristic)
    if args.bonus_center < 0:
        print("[train_online] Auto-tuning bonus center...")
        obs_cal = env.reset(seed=args.seed)[0]
        obs_cal_batch = []
        act_cal_batch = []
        for _ in range(256):
            obs_p_cal = get_obs_p(obs_cal)
            a_cal, _, _ = agent.select_action(obs_p_cal)
            obs_cal_batch.append(obs_cal)
            act_cal_batch.append(a_cal)
            step_res = env.step(a_cal)
            if len(step_res) == 5: obs_cal, _, d, t, _ = step_res
            else: obs_cal, _, d, _ = step_res
            if d or t: obs_cal = env.reset()[0]
            
        sup_cal_mean = agent.compute_support(np.stack(obs_cal_batch), np.stack(act_cal_batch))
        agent.bonus_center = float(sup_cal_mean * 0.9)
        print(f"[train_online] Bonus center set to {agent.bonus_center:.3f} (based on init policy support {sup_cal_mean:.3f})")

    obs = env.reset(seed=args.seed)[0]
    obs_p = get_obs_p(obs)
    ep_ret, ep_len = 0.0, 0
    ep_returns = deque(maxlen=10)
    rew_mean, rew_var, rew_cnt = 0.0, 1.0, 1e-6

    def update_running_stats(x: float):
        nonlocal rew_mean, rew_var, rew_cnt
        rew_cnt += 1.0
        delta = x - rew_mean
        rew_mean += delta / rew_cnt
        rew_var += delta * (x - rew_mean)

    logger = None
    if args.log_csv:
        from o2o.utils.logger import CSVLogger
        import time, os
        log_path = args.log_csv
        if os.path.isdir(log_path) or log_path.endswith(os.sep):
            ts = int(time.time())
            os.makedirs(log_path, exist_ok=True)
            log_path = os.path.join(log_path, f"run_{ts}.csv")
        logger = CSVLogger(log_path, ["steps", "avg_ep_ret", "support_mean", "support_p10", "support_p90", "alpha_pess", "w_actor_mean", "v_mse", "loss_actor", "bonus_used", "rew_std", "pess_penalty"])

    t = 0
    buf = {k: [] for k in ["obs", "act", "rew", "val", "logp", "done", "support", "support_std"]}
    
    while t < args.total_steps:
        for _ in range(args.steps_per_epoch):
            if not np.all(np.isfinite(obs)):
                obs = env.reset()[0]
                obs_p = get_obs_p(obs)
                ep_ret, ep_len = 0.0, 0
                continue

            a, logp, ent = agent.select_action(obs_p)
            
            if not spec.discrete:
                low = np.array(spec.action_low_vec, dtype=np.float32)
                high = np.array(spec.action_high_vec, dtype=np.float32)
                if isinstance(a, np.ndarray):
                    if not np.all(np.isfinite(a)): a = np.zeros_like(a, dtype=np.float32)
                    a = np.clip(a, low, high)
                else:
                    if not np.isfinite(a): a = 0.0
                    a = float(np.clip(a, low, high))
            else:
                if not np.isfinite(a): a = int(0)

            step_res = env.step(a)
            if len(step_res) == 5:
                next_obs, r, done, truncated, _ = step_res
            else:
                next_obs, r, done, info = step_res
                truncated = info.get("TimeLimit.truncated", False) if isinstance(info, dict) else False

            # DSR Support Calculation (Use RAW obs)
            if not spec.discrete:
                a_vec = np.array(a, dtype=np.float32)
                low, high = np.array(spec.action_low_vec, dtype=np.float32), np.array(spec.action_high_vec, dtype=np.float32)
                eps = 0.05 * (high - low)
                a_jit = np.clip(a_vec[None, ...] + np.random.uniform(-eps, eps, size=(5, a_vec.shape[0])).astype(np.float32), low, high)
                s_rep = np.repeat(obs[None, ...], 5, axis=0)
                with torch.no_grad():
                    sup_batch = agent.dsr.support(s_rep, a_jit).detach().cpu().numpy().reshape(-1)
                sup_mean = float(np.mean(sup_batch))
                sup_std = float(np.std(sup_batch))
            else:
                sup_mean = agent.compute_support(obs, np.array([a]))
                sup_std = 0.0

            # Exploration Bonus
            bonus_arr = agent.compute_bonus(np.array([sup_mean], dtype=np.float32), np.array([sup_std], dtype=np.float32))
            bonus = float(max(0.0, float(bonus_arr[0])))
            
            rew_std_val = float(np.sqrt(rew_var / max(1.0, rew_cnt - 1.0)))
            if args.adaptive_bonus and len(ep_returns) > 0:
                avg_return = float(np.mean(ep_returns))
                perf_scale = max(0.1, 1.0 - avg_return / max(1e-6, args.target_return))
                bonus *= (args.bonus_alpha * perf_scale)
            bonus = float(np.clip(bonus, 0.0, 0.5 * max(1e-6, rew_std_val)))

            # Pessimism Penalty (Reward Shaping)
            pess_alpha = agent.get_pessimism_alpha(t)
            inv_sup = max(0.0, 1.0 - sup_mean)
            if agent.pess_gamma != 1.0:
                inv_sup = inv_sup ** agent.pess_gamma
            pess_penalty = pess_alpha * inv_sup
            
            # PROFESSOR FIX: Modify return directly, do not hack value target
            
            # Scale rewards down by 100x or 10x for HalfCheetah to keep values small (~1.0)
            reward_scale = 0.05 
            r_total = float((r + bonus - pess_penalty) * reward_scale)

            v = agent.evaluate_value(obs_p)
            buf["obs"].append(obs_p)
            buf["act"].append(a)
            buf["rew"].append(r_total)
            buf["val"].append(v)
            buf["logp"].append(logp)
            # Only mark true environment termination; truncate should still bootstrap
            buf["done"].append(float(done))
            buf["support"].append(sup_mean)
            buf["support_std"].append(sup_std)

            update_running_stats(float(r))
            ep_ret += r
            ep_len += 1
            t += 1

            if not np.all(np.isfinite(next_obs)):
                ep_returns.append(ep_ret)
                obs = env.reset()[0]
                obs_p = get_obs_p(obs)
                ep_ret, ep_len = 0.0, 0
            else:
                obs = next_obs
                obs_p = get_obs_p(obs)

            if done or truncated:
                # Prevent value bleed across episodes: treat truncation as terminal for advantages
                if truncated and len(buf["done"]) > 0:
                    buf["done"][-1] = 1.0
                ep_returns.append(ep_ret)
                obs = env.reset()[0]
                obs_p = get_obs_p(obs)
                ep_ret, ep_len = 0.0, 0

            if t >= args.total_steps:
                break

        last_val = agent.evaluate_value(obs_p)
        rewards = np.array(buf["rew"], dtype=np.float32)
        values = np.array(buf["val"], dtype=np.float32)
        dones = np.array(buf["done"], dtype=np.float32)
        values_plus = np.concatenate([values, np.array([last_val], dtype=np.float32)])
        adv, ret = agent.compute_advantages(rewards, values_plus, dones, agent.gamma, agent.gae_lambda)

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
        
        mask = np.isfinite(batch["obs"]).all(axis=1) & np.isfinite(batch["ret"]) & np.isfinite(batch["adv"])
        if not spec.discrete: mask &= np.isfinite(batch["act"]).all(axis=-1)
        else: mask &= np.isfinite(batch["act"])
        
        if mask.sum() < len(mask):
            for k in list(batch.keys()): batch[k] = batch[k][mask]
            
        if batch["obs"].size == 0:
            buf = {k: [] for k in buf}
            continue

        train_actor = t >= args.actor_freeze_steps
        metrics = agent.update(
            batch,
            minibatch_size=args.minibatch_size,
            train_iters=args.train_iters,
            current_total_steps=t,
            train_actor=train_actor,
        )
        buf = {k: [] for k in buf}

        avg_ret = np.mean(ep_returns) if len(ep_returns) else 0.0
        sup = batch["support"]
        sup_mean = float(np.mean(sup)) if sup.size else 0.0
        print(f"Steps {t}/{args.total_steps} AvgRet {avg_ret:.1f} Sup {sup_mean:.2f} Alpha {metrics.get('alpha_pess',0.0):.3f} Bonus {bonus:.3f} Penalty {pess_penalty:.3f}")
        
        if logger:
            logger.log({
                "steps": t, "avg_ep_ret": avg_ret, "support_mean": sup_mean,
                "support_p10": float(np.percentile(sup, 10)) if sup.size else 0.0,
                "support_p90": float(np.percentile(sup, 90)) if sup.size else 0.0,
                "alpha_pess": metrics.get("alpha_pess", 0.0),
                "w_actor_mean": metrics.get("w_actor_mean", 0.0),
                "v_mse": metrics.get("v_mse", 0.0),
                "loss_actor": metrics.get("loss_actor", 0.0),
                "bonus_used": bonus,
                "rew_std": rew_std_val,
                "pess_penalty": pess_penalty,
            })
        
        if wandb_run:
            wandb_mod.log({
                "steps": t, "avg_ep_ret": avg_ret, "support/mean": sup_mean,
                "pess_penalty": pess_penalty, "alpha_pess": metrics.get("alpha_pess", 0.0)
            }, step=t)

    env.close()
    if logger: logger.close()
    if wandb_run: wandb_run.finish()

if __name__ == "__main__":
    main()
