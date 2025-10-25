import argparse
import os
from typing import Dict, Tuple

import numpy as np


def load_npz_dataset(path: str, max_samples: int | None = None) -> Dict[str, np.ndarray]:
    data = np.load(path)
    states = data["states"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    if max_samples is not None and max_samples < len(states):
        idx = np.random.permutation(len(states))[:max_samples]
        states = states[idx]
        actions = actions[idx]
    return {"states": states, "actions": actions}


def save_npz_dataset(path: str, states: np.ndarray, actions: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, states=states, actions=actions)


def collect_random_dataset(env_id: str, episodes: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import gymnasium as gym
    except Exception:
        import gym

    env = gym.make(env_id)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    states = []
    actions = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            if hasattr(env.action_space, "n"):
                a = rng.integers(env.action_space.n)
            else:
                low = env.action_space.low
                high = env.action_space.high
                a = rng.random(size=env.action_space.shape, dtype=np.float32) * (high - low) + low
            states.append(np.asarray(obs, dtype=np.float32))
            actions.append(np.asarray(a, dtype=np.float32))
            obs, r, done, truncated, _ = env.step(a)

    env.close()
    return np.stack(states), np.stack(actions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    s, a = collect_random_dataset(args.env_id, args.episodes, args.seed)
    save_npz_dataset(args.out, s, a)
    print(f"Saved dataset: {args.out}  states={s.shape} actions={a.shape}")

