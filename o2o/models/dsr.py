from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, activation="relu", out_act=None):
    layers = []
    act = nn.ReLU if activation == "relu" else nn.Tanh
    for j in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[j], sizes[j + 1])]
        if j < len(sizes) - 2:
            layers += [act()]
        elif out_act:
            layers += [out_act()]
    return nn.Sequential(*layers)


class DSRNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden=(256, 256), activation="relu"):
        super().__init__()
        self.net = mlp([state_dim + action_dim, *hidden, 1], activation=activation)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)  # logits


@dataclass
class DSRMetadata:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray | None
    action_std: np.ndarray | None
    state_dim: int
    action_dim: int
    discrete: bool
    action_low: float | None
    action_high: float | None
    temperature: float


class DSR:
    def __init__(self, net: DSRNet, meta: DSRMetadata, device: str = "cpu"):
        self.net = net.to(device)
        self.meta = meta
        self.device = device

    @torch.no_grad()
    def support(self, states: np.ndarray | torch.Tensor, actions: np.ndarray | torch.Tensor) -> torch.Tensor:
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        s = (s - torch.as_tensor(self.meta.state_mean, device=self.device)) / (
            torch.as_tensor(self.meta.state_std, device=self.device) + 1e-6
        )
        if not self.meta.discrete:
            a = (a - torch.as_tensor(self.meta.action_mean, device=self.device)) / (
                torch.as_tensor(self.meta.action_std, device=self.device) + 1e-6
            )
        logits = self.net(s, self._one_hot_if_needed(a)) / max(self.meta.temperature, 1e-6)
        return torch.sigmoid(logits)

    def _one_hot_if_needed(self, a: torch.Tensor) -> torch.Tensor:
        if self.meta.discrete:
            a_idx = a.long().view(-1)
            oh = torch.zeros((a_idx.shape[0], self.meta.action_dim), device=a.device)
            oh[torch.arange(a_idx.shape[0]), a_idx.clamp(0, self.meta.action_dim - 1)] = 1.0
            return oh
        return a

    def state_dict(self) -> Dict:
        return {"model": self.net.state_dict(), "meta": self.meta.__dict__}

    @staticmethod
    def load(path: str, device: str = "cpu") -> "DSR":
        # weights_only=False to allow loading metadata dict with numpy arrays (trusted local file)
        chk = torch.load(path, map_location=device, weights_only=False)
        meta = DSRMetadata(**chk["meta"])
        net = DSRNet(meta.state_dim, meta.action_dim)
        net.load_state_dict(chk["model"])
        return DSR(net, meta, device=device)


def train_dsr(
    states: np.ndarray,
    actions: np.ndarray,
    discrete: bool,
    action_dim: int,
    action_low: float | None,
    action_high: float | None,
    hidden=(256, 256),
    activation="relu",
    num_negatives: int = 1,
    lr: float = 3e-4,
    batch_size: int = 1024,
    epochs: int = 50,
    device: str = "cpu",
    action_noise: float = 0.0,
    temperature: float = 1.0,
    log_every: int = 100,
    weight_decay: float = 0.0,
    grad_clip: float | None = None,
) -> DSR:
    rng = np.random.default_rng()
    n = states.shape[0]
    state_dim = states.shape[1]
    action_raw = actions

    # normalization
    s_mean = states.mean(axis=0)
    s_std = states.std(axis=0) + 1e-6
    states_n = (states - s_mean) / s_std

    if discrete:
        a_mean = None
        a_std = None
        a_train = one_hot(actions.astype(np.int64), action_dim)
    else:
        a_mean = actions.mean(axis=0)
        a_std = actions.std(axis=0) + 1e-6
        a_train = (actions - a_mean) / a_std

    net = DSRNet(state_dim, action_dim, hidden=hidden, activation=activation).to(device)
    if weight_decay and weight_decay > 0.0:
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(net.parameters(), lr=lr)

    # training loop
    steps_per_epoch = int(np.ceil(n / batch_size))
    for epoch in range(epochs):
        perm = rng.permutation(n)
        for it in range(steps_per_epoch):
            idx = perm[it * batch_size : (it + 1) * batch_size]
            s_pos = torch.as_tensor(states_n[idx], dtype=torch.float32, device=device)
            a_pos = torch.as_tensor(a_train[idx], dtype=torch.float32, device=device)

            # generate negatives (same s, random a from broad prior)
            if discrete:
                neg_idx = np.random.randint(0, action_dim, size=(len(idx), num_negatives))
                a_neg_np = np.eye(action_dim, dtype=np.float32)[neg_idx]
            else:
                low = action_low if action_low is not None else -1.0
                high = action_high if action_high is not None else 1.0
                a_neg_np = np.random.rand(len(idx), num_negatives, a_train.shape[1]).astype(np.float32)
                a_neg_np = a_neg_np * (high - low) + low
                if action_noise > 0:
                    a_neg_np = a_neg_np + action_noise * np.random.randn(*a_neg_np.shape).astype(np.float32)
                # normalize with dataset stats
                a_neg_np = (a_neg_np - a_mean) / a_std

            s_neg = s_pos.repeat_interleave(num_negatives, dim=0)
            a_neg = torch.as_tensor(a_neg_np.reshape(len(idx) * num_negatives, -1), dtype=torch.float32, device=device)

            # logits
            logits_pos = net(s_pos, a_pos) / max(temperature, 1e-6)
            logits_neg = net(s_neg, a_neg) / max(temperature, 1e-6)

            # BCE with logits: positives->1, negatives->0
            loss_pos = F.binary_cross_entropy_with_logits(logits_pos, torch.ones_like(logits_pos))
            loss_neg = F.binary_cross_entropy_with_logits(logits_neg, torch.zeros_like(logits_neg))
            loss = loss_pos + loss_neg

            opt.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            opt.step()

            global_it = epoch * steps_per_epoch + it
            if (global_it + 1) % log_every == 0:
                with torch.no_grad():
                    p_pos = torch.sigmoid(logits_pos).mean().item()
                    p_neg = torch.sigmoid(logits_neg).mean().item()
                print(f"[DSR] epoch {epoch} it {it} loss {loss.item():.4f} p_pos {p_pos:.3f} p_neg {p_neg:.3f}")

    meta = DSRMetadata(
        state_mean=s_mean,
        state_std=s_std,
        action_mean=a_mean,
        action_std=a_std,
        state_dim=state_dim,
        action_dim=action_dim,
        discrete=bool(discrete),
        action_low=action_low,
        action_high=action_high,
        temperature=temperature,
    )
    return DSR(net, meta, device=device)


def one_hot(a_idx: np.ndarray, n: int) -> np.ndarray:
    a_idx = a_idx.astype(np.int64).reshape(-1)
    out = np.zeros((a_idx.shape[0], n), dtype=np.float32)
    out[np.arange(a_idx.shape[0]), np.clip(a_idx, 0, n - 1)] = 1.0
    return out
