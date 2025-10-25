from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SpaceSpec:
    discrete: bool
    state_dim: int
    action_dim: int
    action_low: float | None = None
    action_high: float | None = None
    action_low_vec: list[float] | None = None
    action_high_vec: list[float] | None = None


def make_env(env_id: str, seed: int = 0):
    try:
        import gymnasium as gym
    except Exception:
        import gym
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env


def get_space_spec(env) -> SpaceSpec:
    obs_space = env.observation_space
    act_space = env.action_space

    # flatten for typical Box/Categorical
    if hasattr(act_space, "n"):
        discrete = True
        action_dim = act_space.n
        action_low = None
        action_high = None
        action_low_vec = None
        action_high_vec = None
    else:
        discrete = False
        action_dim = int(act_space.shape[0])
        action_low = float(act_space.low.min())
        action_high = float(act_space.high.max())
        action_low_vec = act_space.low.astype(float).tolist()
        action_high_vec = act_space.high.astype(float).tolist()

    if hasattr(obs_space, "shape") and obs_space.shape is not None:
        state_dim = int(obs_space.shape[0])
    else:
        raise ValueError("Unsupported observation space; expected flat Box")

    return SpaceSpec(
        discrete=discrete,
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        action_low_vec=action_low_vec,
        action_high_vec=action_high_vec,
    )


def sample_uniform_actions(act_space, n: int):
    import numpy as np

    if hasattr(act_space, "n"):
        ints = np.random.randint(0, act_space.n, size=(n,))
        return ints.astype("int64")
    else:
        low = act_space.low
        high = act_space.high
        return np.random.rand(n, *act_space.shape).astype("float32") * (high - low) + low


def to_numpy(x):
    import numpy as np
    import torch

    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()
