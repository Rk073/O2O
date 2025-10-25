import torch


def boundary_bonus(
    support: torch.Tensor, center: float = 0.7, sigma: float = 0.15, eta: float = 0.1
) -> torch.Tensor:
    x = support.clamp(0.0, 1.0)
    bonus = torch.exp(-0.5 * ((x - center) / (sigma + 1e-6)) ** 2)
    return eta * bonus


def entropy_bonus(support: torch.Tensor, eta: float = 0.1) -> torch.Tensor:
    x = support.clamp(1e-8, 1 - 1e-8)
    H = -(x * torch.log(x) + (1 - x) * torch.log(1 - x))
    return eta * H


def ucb_bonus(
    mean_support: torch.Tensor,
    std_support: torch.Tensor,
    eta: float = 0.1,
    kappa: float = 1.0,
) -> torch.Tensor:
    m = mean_support.clamp(0.0, 1.0)
    s = std_support.clamp_min(0.0)
    # gate: avoid far-OOD (<0.2) and already-known (>0.9) regions
    band_mask = (m >= 0.2) & (m <= 0.9)
    m_clip = m.clamp(1e-8, 1 - 1e-8)
    entropy_component = 0.5 * (
        -(m_clip * torch.log(m_clip) + (1 - m_clip) * torch.log(1 - m_clip))
    )
    return eta * (entropy_component + kappa * s) * band_mask.float()
