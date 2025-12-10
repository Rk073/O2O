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
    calib_temperature: float | None = 1.0


class DSR:
    def __init__(self, net: DSRNet, meta: DSRMetadata, device: str = "cpu"):
        self.net = net.to(device)
        self.meta = meta
        self.device = device

    @torch.no_grad()
    def support(self, states: np.ndarray | torch.Tensor, actions: np.ndarray | torch.Tensor) -> torch.Tensor:
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        
        # Normalize inputs using offline statistics
        s = (s - torch.as_tensor(self.meta.state_mean, device=self.device)) / (
            torch.as_tensor(self.meta.state_std, device=self.device) + 1e-6
        )
        if not self.meta.discrete:
            a = (a - torch.as_tensor(self.meta.action_mean, device=self.device)) / (
                torch.as_tensor(self.meta.action_std, device=self.device) + 1e-6
            )
            
        temp = float(self.meta.temperature) if self.meta.temperature is not None else 1.0
        ctemp = float(self.meta.calib_temperature) if hasattr(self.meta, 'calib_temperature') and self.meta.calib_temperature is not None else 1.0
        logits = self.net(s, self._one_hot_if_needed(a)) / max(temp * ctemp, 1e-6)
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
        chk = torch.load(path, map_location=device, weights_only=False)
        meta = DSRMetadata(**chk["meta"])
        hidden = None
        try:
            w_keys = [(k, v) for k, v in chk["model"].items() if k.startswith("net.") and k.endswith(".weight")]
            def layer_idx(k):
                parts = k.split(".")
                return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            w_keys.sort(key=lambda kv: layer_idx(kv[0]))
            outs = [int(w.shape[0]) for _, w in w_keys]
            if len(outs) >= 2:
                hidden = tuple(outs[:-1])
        except Exception:
            hidden = None
        if hidden is None or len(hidden) == 0:
            hidden = (256, 256)
        net = DSRNet(meta.state_dim, meta.action_dim, hidden=hidden)
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
    seed: int | None = None,
    val_split: float = 0.0,
    early_stop_patience: int = 0,
    neg_modes: tuple[str, ...] = ("hard", "jitter"),
    neg_weights: tuple[float, ...] | None = None,
    jitter_std: float = 0.1,
    calibrate_temperature: bool = False,
    log_callback=None,
) -> DSR:
    rng = np.random.default_rng(seed)
    try:
        import torch
        if seed is not None:
            torch.manual_seed(int(seed))
    except Exception:
        pass
    n = states.shape[0]
    state_dim = states.shape[1]
    
    # Normalization (Train internal DSR on normalized data)
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

    # split train/val
    idx_all = np.arange(n)
    val_n = int(n * max(0.0, min(0.5, val_split)))
    if val_n > 0:
        rng.shuffle(idx_all)
        idx_val = idx_all[:val_n]
        idx_train = idx_all[val_n:]
    else:
        idx_train = idx_all
        idx_val = None

    # REVISED Negative Sampler: Prioritizes Hard/Jitter over Uniform
    def gen_negatives(s_pos_np: np.ndarray, a_pos_np: np.ndarray, k: int) -> np.ndarray:
        mlist = list(neg_modes) if neg_modes else ["hard", "jitter"]
        default_w = [1.0] * len(mlist)
        w = np.array(neg_weights if neg_weights is not None else default_w, dtype=np.float32)
        w = w / (w.sum() + 1e-8)
        counts = np.maximum(1, np.round(w * k).astype(int))
        
        # adjust to sum exactly k
        diff = k - counts.sum()
        if diff != 0:
            counts[np.argmax(counts)] += diff
            
        outs = []
        B = s_pos_np.shape[0]
        
        for mode, c in zip(mlist, counts):
            if c <= 0: continue
            
            if mode == "hard":
                # Hard Negatives: Sample real actions from other parts of the dataset
                rand_idxs = rng.integers(0, len(a_train), size=(B, c))
                a_hard = a_train[rand_idxs]
                outs.append(a_hard)
                
            elif not discrete and mode == "jitter":
                # Jitter Negatives: Local perturbation of the positive action
                # a_pos_np is normalized, so we jitter in normalized space
                noise = rng.standard_normal(size=(B, c, a_train.shape[1])).astype(np.float32) * jitter_std
                a_j = a_pos_np[:, None, :] + noise
                # Clip to reasonable normalized bounds (approx 4 std devs)
                a_j = np.clip(a_j, -4.0, 4.0)
                outs.append(a_j)
                
            elif not discrete and mode == "uniform":
                # Uniform Negatives: Global coverage
                low_n = -3.0 # Approx 3 std devs
                high_n = 3.0
                a_u = rng.uniform(low_n, high_n, size=(B, c, a_train.shape[1])).astype(np.float32)
                outs.append(a_u)
                
            elif discrete:
                # Discrete fallbacks
                neg_idx = np.random.randint(0, action_dim, size=(B, c))
                a_neg = np.eye(action_dim, dtype=np.float32)[neg_idx]
                outs.append(a_neg)

        if not outs: 
            return np.zeros((B, k, a_train.shape[1] if not discrete else action_dim), dtype=np.float32)
            
        return np.concatenate(outs, axis=1)

    # training loop
    steps_per_epoch = int(np.ceil(len(idx_train) / batch_size))
    best_state = None
    best_val = float("inf")
    patience = early_stop_patience
    last_train_loss = None
    
    for epoch in range(epochs):
        perm = rng.permutation(len(idx_train))
        for it in range(steps_per_epoch):
            idx_loc = perm[it * batch_size : (it + 1) * batch_size]
            if idx_loc.size == 0: continue
            idx = idx_train[idx_loc]
            if len(idx) == 0: continue
            
            s_pos_np = states_n[idx]
            a_pos_np = a_train[idx]
            
            s_pos = torch.as_tensor(s_pos_np, dtype=torch.float32, device=device)
            a_pos = torch.as_tensor(a_pos_np, dtype=torch.float32, device=device)

            # Generate Negatives
            a_neg_np = gen_negatives(s_pos_np, a_pos_np, num_negatives)
            s_neg = s_pos.repeat_interleave(num_negatives, dim=0)
            a_neg = torch.as_tensor(a_neg_np.reshape(len(idx) * num_negatives, -1), dtype=torch.float32, device=device)

            # Logits
            logits_pos = net(s_pos, a_pos) / max(temperature, 1e-6)
            logits_neg = net(s_neg, a_neg) / max(temperature, 1e-6)

            # Loss (Positives -> 1, Negatives -> 0)
            loss_pos = F.binary_cross_entropy_with_logits(logits_pos, torch.ones_like(logits_pos))
            loss_neg = F.binary_cross_entropy_with_logits(logits_neg, torch.zeros_like(logits_neg))
            loss = loss_pos + loss_neg

            opt.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            opt.step()
            last_train_loss = float(loss.item())

            global_it = epoch * steps_per_epoch + it
            if (global_it + 1) % log_every == 0:
                with torch.no_grad():
                    p_pos = torch.sigmoid(logits_pos).mean().item()
                    p_neg = torch.sigmoid(logits_neg).mean().item()
                print(f"[DSR] epoch {epoch} it {it} loss {loss.item():.4f} p_pos {p_pos:.3f} p_neg {p_neg:.3f}")

        # Validation
        val_l = None
        if idx_val is not None:
            with torch.no_grad():
                val_bs = min(4096, len(idx_val))
                sel = rng.choice(idx_val, size=val_bs, replace=False)
                s_val = torch.as_tensor(states_n[sel], dtype=torch.float32, device=device)
                a_val_np = a_train[sel]
                a_val = torch.as_tensor(a_val_np, dtype=torch.float32, device=device)
                
                a_val_neg_np = gen_negatives(states_n[sel], a_val_np, num_negatives)
                s_val_neg = s_val.repeat_interleave(num_negatives, dim=0)
                a_val_neg = torch.as_tensor(a_val_neg_np.reshape(val_bs * num_negatives, -1), dtype=torch.float32, device=device)
                
                lp = net(s_val, a_val) / max(temperature, 1e-6)
                ln = net(s_val_neg, a_val_neg) / max(temperature, 1e-6)
                val_loss = F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp)) + F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))
                val_l = float(val_loss.item())
                
            if val_l < best_val - 1e-6:
                best_val = val_l
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                patience = early_stop_patience
            else:
                patience -= 1
                
            print(f"[DSR] epoch {epoch} val_loss {val_l:.4f} best {best_val:.4f} patience {patience}")
            if early_stop_patience and patience < 0:
                print("[DSR] Early stopping due to no val improvement.")
                break

        if log_callback is not None:
            try:
                log_callback({"epoch": int(epoch), "train_loss": float(last_train_loss), "val_loss": float(val_l)}, step=epoch)
            except Exception: pass

    meta = DSRMetadata(
        state_mean=s_mean, state_std=s_std, action_mean=a_mean, action_std=a_std,
        state_dim=state_dim, action_dim=action_dim, discrete=bool(discrete),
        action_low=action_low, action_high=action_high, temperature=temperature, calib_temperature=1.0,
    )
    dsr = DSR(net, meta, device=device)
    if best_state is not None:
        dsr.net.load_state_dict(best_state)

    # Post-hoc calibration
    if calibrate_temperature and idx_val is not None and len(idx_val) > 0:
        with torch.no_grad():
            val_bs = min(10000, len(idx_val))
            sel = rng.choice(idx_val, size=val_bs, replace=False)
            s_val = torch.as_tensor(states_n[sel], dtype=torch.float32, device=device)
            a_val_np = a_train[sel]
            a_val = torch.as_tensor(a_val_np, dtype=torch.float32, device=device)
            a_val_neg_np = gen_negatives(states_n[sel], a_val_np, num_negatives)
            s_val_neg = s_val.repeat_interleave(num_negatives, dim=0)
            a_val_neg = torch.as_tensor(a_val_neg_np.reshape(val_bs * num_negatives, -1), dtype=torch.float32, device=device)
            
            lp = dsr.net(s_val, a_val)
            ln = dsr.net(s_val_neg, a_val_neg)
            y = torch.cat([torch.ones_like(lp), torch.zeros_like(ln)], dim=0)
            logits = torch.cat([lp, ln], dim=0)
            
            cands = torch.logspace(np.log10(0.5), np.log10(3.0), steps=21, device=device)
            best_c, best_bce = 1.0, float("inf")
            for c in cands:
                bce = F.binary_cross_entropy_with_logits(logits / (temperature * c), y).item()
                if bce < best_bce:
                    best_bce = bce
                    best_c = float(c.item())
            dsr.meta.calib_temperature = best_c
            print(f"[DSR] Calibrated temperature factor: {best_c:.3f} (val BCE {best_bce:.4f})")

    return dsr


def one_hot(a_idx: np.ndarray, n: int) -> np.ndarray:
    a_idx = a_idx.astype(np.int64).reshape(-1)
    out = np.zeros((a_idx.shape[0], n), dtype=np.float32)
    out[np.arange(a_idx.shape[0]), np.clip(a_idx, 0, n - 1)] = 1.0
    return out