from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from o2o.models.dsr import DSR
from o2o.utils.support_bonus import boundary_bonus, entropy_bonus, ucb_bonus


def mlp(sizes, activation="tanh", out_act=None):
    layers = []
    act = nn.Tanh if activation == "tanh" else nn.ReLU
    for j in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[j], sizes[j + 1])]
        if j < len(sizes) - 2:
            layers += [act()]
        elif out_act:
            layers += [out_act()]
    return nn.Sequential(*layers)


class ActorDiscrete(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden=(64, 64), activation="tanh"):
        super().__init__()
        self.net = mlp([state_dim, *hidden, action_dim], activation=activation)

    def forward(self, s: torch.Tensor):
        logits = self.net(s)
        return torch.distributions.Categorical(logits=logits)


class ActorGaussian(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, action_low: np.ndarray, action_high: np.ndarray, hidden=(64, 64), activation="tanh"):
        super().__init__()
        self.net = mlp([state_dim, *hidden, action_dim], activation=activation)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)
        self.register_buffer("act_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("act_high", torch.as_tensor(action_high, dtype=torch.float32))
        self.register_buffer("act_scale", (self.act_high - self.act_low) / 2.0)
        self.register_buffer("act_bias", (self.act_high + self.act_low) / 2.0)

    def forward(self, s: torch.Tensor):
        mu = self.net(s)
        std = torch.exp(self.log_std.clamp(min=-5.0, max=2.0))
        base = torch.distributions.Independent(torch.distributions.Normal(mu, std), 1)
        return TanhTransformedDist(base, self.act_scale, self.act_bias)


class TanhTransformedDist(torch.distributions.Distribution):
    arg_constraints = {}
    def __init__(self, base_dist, scale, bias):
        self.base_dist = base_dist
        self.scale = scale
        self.bias = bias
        super().__init__(base_dist.batch_shape, base_dist.event_shape, validate_args=False)

    @property
    def mean(self):
        return torch.tanh(self.base_dist.mean) * self.scale + self.bias

    def sample(self):
        z = self.base_dist.rsample()
        a = torch.tanh(z)
        return a * self.scale + self.bias

    def rsample(self):
        return self.sample()

    def log_prob(self, value):
        a = (value - self.bias) / (self.scale + 1e-8)
        a = a.clamp(-0.999999, 0.999999)
        z = 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh
        log_prob = self.base_dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-8).sum(-1)
        return log_prob

    def entropy(self):
        return self.base_dist.base_dist.entropy().sum(-1)


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden=(64, 64), activation="tanh"):
        super().__init__()
        self.net = mlp([state_dim, *hidden, 1], activation=activation)

    def forward(self, s: torch.Tensor):
        return self.net(s).squeeze(-1)


@dataclass
class PPOAgent:
    actor: nn.Module
    critic: nn.Module
    dsr: DSR
    ref_actor: nn.Module | None
    discrete: bool
    action_dim: int
    device: str
    lr_actor: float
    lr_critic: float
    clip_ratio: float
    target_kl: float
    vf_coeff: float
    ent_coeff: float
    gamma: float
    gae_lambda: float
    pess_alpha0: float
    pess_alpha_final: float
    pess_anneal_steps: float
    pess_gamma: float
    adv_gate_tau: float
    adv_gate_k: float
    bonus_eta: float
    bonus_center: float
    bonus_sigma: float
    use_bonus: bool
    bonus_type: str = "boundary"
    kl_bc_coef: float = 0.0
    kl_bc_pow: float = 1.0
    kl_bc_coef0: float = 0.0
    kl_bc_coef_final: float = 0.0
    kl_bc_anneal_steps: float = 0.0
    actor_grad_clip: float | None = None
    critic_grad_clip: float | None = None
    vf_clip: float | None = None
    support_adaptive_clip: bool = False
    adv_gate_tau0: float | None = None
    adv_gate_tau_final: float | None = None
    adv_gate_tau_anneal_steps: float | None = None
    adv_gate_uncert_kappa: float = 0.0

    def __post_init__(self):
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        if self.ref_actor is not None:
            self.ref_actor.to(self.device)
            for p in self.ref_actor.parameters():
                p.requires_grad = False
            self.ref_actor.eval()

    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.actor(s)
        a = dist.sample()
        logp = dist.log_prob(a)
        if self.discrete:
            a_np = a.item()
        else:
            a_np = a.detach().cpu().numpy()[0]
        return a_np, logp.item(), dist.entropy().mean().item()

    def compute_advantages(self, rewards, values, dones, gamma, lam):
        adv = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            adv[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        ret = adv + values[:-1]
        return adv, ret

    def get_pessimism_alpha(self, current_total_steps: int) -> float:
        """Calculate current pessimism coefficient."""
        frac = max(0.0, 1.0 - (float(current_total_steps) / max(1.0, float(self.pess_anneal_steps))))
        alpha = self.pess_alpha_final + (self.pess_alpha0 - self.pess_alpha_final) * frac
        return alpha

    def update(self, buf: dict, minibatch_size: int, train_iters: int, current_total_steps: int):
        obs = torch.as_tensor(buf["obs"], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(buf["act"], dtype=torch.float32 if not self.discrete else torch.long, device=self.device)
        adv = torch.as_tensor(buf["adv"], dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(buf["ret"], dtype=torch.float32, device=self.device)
        logp_old = torch.as_tensor(buf["logp"], dtype=torch.float32, device=self.device)
        support = torch.as_tensor(buf["support"], dtype=torch.float32, device=self.device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Gate weights calculation
        with torch.no_grad():
            tau = self.adv_gate_tau
            if self.adv_gate_tau0 is not None and self.adv_gate_tau_final is not None and self.adv_gate_tau_anneal_steps is not None and self.adv_gate_tau_anneal_steps > 0:
                frac_tau = max(0.0, 1.0 - (float(current_total_steps) / float(self.adv_gate_tau_anneal_steps)))
                tau = self.adv_gate_tau_final + (self.adv_gate_tau0 - self.adv_gate_tau_final) * frac_tau
            if self.adv_gate_k is not None and self.adv_gate_k > 0:
                w = torch.sigmoid(self.adv_gate_k * (support - tau))
            else:
                w = torch.ones_like(support)

        n = obs.shape[0]
        idxs = np.arange(n)
        avg_v_mse = 0.0
        avg_loss_actor = 0.0
        count = 0
        approx_kl_mean = 0.0
        kl_count = 0
        early_stop = False
        bc_alpha_value = 0.0
        
        for _ in range(train_iters):
            np.random.shuffle(idxs)
            for start in range(0, n, minibatch_size):
                mb = idxs[start : start + minibatch_size]
                
                # Actor Update
                dist = self.actor(obs[mb])
                logp = dist.log_prob(act[mb])
                ratio = torch.exp(logp - logp_old[mb])
                
                # Asymmetric Advantage Gating:
                # Gate positive advantages (don't overfit to OOD luck)
                # Pass negative advantages fully (learn from OOD mistakes)
                adv_mb = adv[mb]
                adv_gated = torch.where(adv_mb > 0, adv_mb * w[mb], adv_mb)
                
                # Normalize after gating for stability
                if adv_gated.std() > 1e-8:
                    adv_gated = (adv_gated - adv_gated.mean()) / (adv_gated.std() + 1e-8)
                
                if self.support_adaptive_clip:
                    coef = 0.5 + 0.5 * w[mb]
                    clip_low = 1 - self.clip_ratio * coef
                    clip_high = 1 + self.clip_ratio * coef
                    clip_adv = torch.clamp(ratio, clip_low, clip_high) * adv_gated
                else:
                    clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_gated
                
                ppo_obj = torch.min(ratio * adv_gated, clip_adv)
                loss_actor = -(ppo_obj).mean()

                # BC Anchor
                if self.kl_bc_coef is not None and self.kl_bc_coef > 0 and self.ref_actor is not None:
                    if self.kl_bc_anneal_steps is not None and self.kl_bc_anneal_steps > 0:
                        frac_bc = max(0.0, 1.0 - (float(current_total_steps) / max(1.0, float(self.kl_bc_anneal_steps))))
                        bc_alpha = self.kl_bc_coef_final + (self.kl_bc_coef0 - self.kl_bc_coef_final) * frac_bc
                    else:
                        bc_alpha = self.kl_bc_coef
                    
                    with torch.no_grad():
                        w_bc = torch.clamp(1.0 - support[mb], 0.0, 1.0)
                        if self.kl_bc_pow is not None and self.kl_bc_pow != 1.0:
                            w_bc = torch.pow(w_bc, self.kl_bc_pow)
                    
                    dist_ref = self.ref_actor(obs[mb])
                    logp_ref = dist_ref.log_prob(act[mb])
                    bc_reg = -(w_bc * logp_ref).mean()
                    loss_actor = loss_actor + bc_alpha * bc_reg
                    bc_alpha_value = float(bc_alpha)
                
                ent = dist.entropy().mean()
                
                self.opt_actor.zero_grad()
                (loss_actor - self.ent_coeff * ent).backward()
                if self.actor_grad_clip is not None and self.actor_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
                self.opt_actor.step()

                # Critic Update
                v = self.critic(obs[mb])
                value_target = ret[mb] # Rewards already shaped in buffer
                
                v_old_mb = None
                if 'v_old' in buf:
                    v_old_mb = torch.as_tensor(buf['v_old'][mb], dtype=torch.float32, device=self.device)
                
                if self.vf_clip is not None and self.vf_clip > 0 and v_old_mb is not None:
                    v_clipped = v_old_mb + (v - v_old_mb).clamp(-self.vf_clip, self.vf_clip)
                    mse_unclipped = (v - value_target) ** 2
                    mse_clipped = (v_clipped - value_target) ** 2
                    mse = torch.max(mse_unclipped, mse_clipped).mean()
                else:
                    mse = F.mse_loss(v, value_target)
                loss_critic = self.vf_coeff * mse

                self.opt_critic.zero_grad()
                loss_critic.backward()
                if self.critic_grad_clip is not None and self.critic_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
                self.opt_critic.step()

                # KL Check
                with torch.no_grad():
                    kl = (logp_old[mb] - logp).mean().abs().item()
                    approx_kl_mean += kl * len(mb)
                    kl_count += len(mb)
                    if self.target_kl is not None and kl > 1.5 * self.target_kl:
                        early_stop = True
                        break

                bs = len(mb)
                avg_v_mse += mse.detach().item() * bs
                avg_loss_actor += loss_actor.detach().item() * bs
                count += bs
            if early_stop:
                break

        if count > 0:
            avg_v_mse /= count
            avg_loss_actor /= count
        if kl_count > 0:
            approx_kl_mean /= kl_count
        return {
            "v_mse": avg_v_mse,
            "loss_actor": avg_loss_actor,
            "alpha_pess": self.get_pessimism_alpha(current_total_steps),
            "w_actor_mean": float(w.mean().item()),
            "approx_kl": float(approx_kl_mean),
            "early_stop": bool(early_stop),
            "bc_alpha": float(bc_alpha_value),
        }

    def evaluate_value(self, obs: np.ndarray) -> float:
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            v = self.critic(s).item()
        return v

    def compute_support(self, obs: np.ndarray, act: np.ndarray) -> float:
        # DSR expects raw or offline-normalized inputs (handled internally in DSR.support)
        # obs should be RAW observations here
        with torch.no_grad():
            sup = self.dsr.support(
                torch.as_tensor(obs[None, ...], dtype=torch.float32, device=self.device),
                torch.as_tensor(
                    act[None, ...],
                    dtype=torch.float32 if not self.discrete else torch.long,
                    device=self.device,
                ),
            ).item()
        return sup

    def compute_entropy_bonus(self, support: np.ndarray) -> np.ndarray:
        sup_t = torch.as_tensor(support, dtype=torch.float32, device=self.device)
        return entropy_bonus(sup_t, eta=self.bonus_eta).detach().cpu().numpy()

    def compute_ucb_bonus(self, support_mean: np.ndarray, support_std: np.ndarray) -> np.ndarray:
        m = torch.as_tensor(support_mean, dtype=torch.float32, device=self.device)
        s = torch.as_tensor(support_std, dtype=torch.float32, device=self.device)
        return ucb_bonus(m, s, eta=self.bonus_eta, kappa=1.0).detach().cpu().numpy()

    def compute_boundary_bonus(self, support: np.ndarray) -> np.ndarray:
        sup_t = torch.as_tensor(support, dtype=torch.float32, device=self.device)
        b = boundary_bonus(sup_t, center=self.bonus_center, sigma=self.bonus_sigma, eta=self.bonus_eta)
        return b.detach().cpu().numpy()

    def compute_bonus(self, support_mean: np.ndarray, support_std: np.ndarray | None = None) -> np.ndarray:
        if not self.use_bonus:
            return np.zeros_like(support_mean)
        if self.bonus_type == "ucb":
            if support_std is None:
                support_std = np.zeros_like(support_mean)
            return self.compute_ucb_bonus(support_mean, support_std)
        elif self.bonus_type == "entropy":
            return self.compute_entropy_bonus(support_mean)
        else:
            return self.compute_boundary_bonus(support_mean)