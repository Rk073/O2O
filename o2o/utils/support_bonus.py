import torch


def boundary_bonus(support: torch.Tensor, center: float = 0.7, sigma: float = 0.15, eta: float = 0.1) -> torch.Tensor:
    # Gaussian bump centered slightly below 1.0 support; rewards just-beyond boundary
    # support in [0,1]
    x = support.clamp(0.0, 1.0)
    bonus = torch.exp(-0.5 * ((x - center) / (sigma + 1e-6)) ** 2)
    return eta * bonus

