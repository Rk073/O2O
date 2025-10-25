from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from o2o.models.dsr import DSR
from o2o.utils.support_bonus import boundary_bonus


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
        std = torch.exp(self.log_std)
        base = torch.distributions.Independent(torch.distributions.Normal(mu, std), 1)
        return TanhTransformedDist(base, self.act_scale, self.act_bias)


class TanhTransformedDist(torch.distributions.Distribution):
    def __init__(self, base_dist, scale, bias):
        self.base_dist = base_dist
        self.scale = scale
        self.bias = bias
        super().__init__(base_dist.batch_shape, base_dist.event_shape)

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
        # inverse tanh-scaling
        a = (value - self.bias) / (self.scale + 1e-8)
        a = a.clamp(-0.999999, 0.999999)
        z = 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh
        log_prob = self.base_dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-8).sum(-1)
        return log_prob

    def entropy(self):
        # approx entropy via unsquashed entropy (common shortcut)
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
    pessimism_beta: float
    pessimism_gamma: float
    bonus_eta: float
    bonus_center: float
    bonus_sigma: float
    use_bonus: bool

    def __post_init__(self):
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

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

    def update(self, buf: dict, minibatch_size: int, train_iters: int):
        obs = torch.as_tensor(buf["obs"], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(buf["act"], dtype=torch.float32 if not self.discrete else torch.long, device=self.device)
        adv = torch.as_tensor(buf["adv"], dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(buf["ret"], dtype=torch.float32, device=self.device)
        logp_old = torch.as_tensor(buf["logp"], dtype=torch.float32, device=self.device)
        support = torch.as_tensor(buf["support"], dtype=torch.float32, device=self.device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        idxs = np.arange(n)
        avg_pess_reg = 0.0
        avg_v_mse = 0.0
        avg_loss_actor = 0.0
        count = 0
        for _ in range(train_iters):
            np.random.shuffle(idxs)
            for start in range(0, n, minibatch_size):
                mb = idxs[start : start + minibatch_size]
                # Actor loss
                dist = self.actor(obs[mb])
                logp = dist.log_prob(act[mb])
                ratio = torch.exp(logp - logp_old[mb])
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv[mb]
                loss_actor = -(torch.min(ratio * adv[mb], clip_adv)).mean()
                ent = dist.entropy().mean()

                self.opt_actor.zero_grad()
                (loss_actor - self.ent_coeff * ent).backward()
                self.opt_actor.step()

                # Critic loss with pessimism regularizer
                v = self.critic(obs[mb])
                mse = F.mse_loss(v, ret[mb])
                pess_w = (1.0 - support[mb]).pow(self.pessimism_gamma)
                pess_reg = (pess_w * (v.detach() if False else v).pow(2)).mean()
                loss_critic = self.vf_coeff * mse + self.pessimism_beta * pess_reg

                self.opt_critic.zero_grad()
                loss_critic.backward()
                self.opt_critic.step()

                # accum metrics
                bs = len(mb)
                avg_pess_reg += pess_reg.detach().item() * bs
                avg_v_mse += mse.detach().item() * bs
                avg_loss_actor += loss_actor.detach().item() * bs
                count += bs

        if count > 0:
            avg_pess_reg /= count
            avg_v_mse /= count
            avg_loss_actor /= count
        return {
            "pess_reg": avg_pess_reg,
            "v_mse": avg_v_mse,
            "loss_actor": avg_loss_actor,
        }

    def evaluate_value(self, obs: np.ndarray) -> float:
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            v = self.critic(s).item()
        return v

    def compute_support(self, obs: np.ndarray, act: np.ndarray) -> float:
        with torch.no_grad():
            sup = self.dsr.support(obs[None, ...], act[None, ...]).item()
        return sup

    def compute_bonus(self, support: np.ndarray) -> np.ndarray:
        if not self.use_bonus:
            return np.zeros_like(support)
        sup_t = torch.as_tensor(support, dtype=torch.float32, device=self.device)
        b = boundary_bonus(sup_t, center=self.bonus_center, sigma=self.bonus_sigma, eta=self.bonus_eta)
        return b.detach().cpu().numpy()
