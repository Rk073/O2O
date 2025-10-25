from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DSRConfig:
    hidden_sizes: tuple = (256, 256)
    activation: str = "relu"
    lr: float = 3e-4
    batch_size: int = 1024
    epochs: int = 50
    device: str = "cuda"
    num_negatives: int = 1
    action_noise: float = 0.0  # for continuous actions when generating negatives
    log_every: int = 100
    temperature: float = 1.0  # optional calibration factor on logits


@dataclass
class PPOConfig:
    hidden_sizes: tuple = (256, 256)
    activation: str = "tanh"
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    train_iters: int = 80
    target_kl: float = 0.01
    steps_per_epoch: int = 4096
    max_ep_len: int = 1000
    minibatch_size: int = 256
    device: str = "cuda"
    vf_coeff: float = 0.5
    ent_coeff: float = 0.0

    # DSR integration
    pessimism_beta: float = 1.0
    pessimism_gamma: float = 1.0
    bonus_eta: float = 0.1
    bonus_center: float = 0.7  # center of boundary-focused bonus in support space
    bonus_sigma: float = 0.15
    use_bonus: bool = True


@dataclass
class TrainDSRArgs:
    offline_path: str
    env_id: str
    dsr_out: str
    seed: int = 0
    max_samples: Optional[int] = None
    config: DSRConfig = field(default_factory=DSRConfig)


@dataclass
class TrainOnlineArgs:
    env_id: str
    dsr_path: str
    total_steps: int = 200000
    seed: int = 0
    config: PPOConfig = field(default_factory=PPOConfig)
