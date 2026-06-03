from __future__ import annotations

import torch
from torch import nn


def centered_ifft_image(vis: torch.Tensor) -> torch.Tensor:
    z = torch.complex(vis[:, 0], vis[:, 1])
    z = torch.fft.ifftshift(z, dim=(-2, -1))
    image = torch.fft.ifft2(z, norm="ortho")
    image = torch.fft.fftshift(image, dim=(-2, -1)).real[:, None]
    flat = image.flatten(1)
    lo = flat.min(dim=1).values[:, None, None, None]
    hi = flat.max(dim=1).values[:, None, None, None]
    return (image - lo) / (hi - lo).clamp_min(1e-6)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class UVDCStage(nn.Module):
    def __init__(self, hidden_channels: int = 64, blocks: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(5, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(inplace=True),
        ]
        layers.extend(ResidualBlock(hidden_channels) for _ in range(blocks))
        layers.append(nn.Conv2d(hidden_channels, 2, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, current: torch.Tensor, measured: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        update = self.net(torch.cat([current, measured, mask], dim=1))
        predicted = current + update
        return predicted * (1.0 - mask) + measured * mask


class UVDCNet(nn.Module):
    """Unrolled visibility reconstruction with hard UV data consistency."""

    def __init__(self, stages: int = 5, hidden_channels: int = 64, blocks_per_stage: int = 4) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            UVDCStage(hidden_channels=hidden_channels, blocks=blocks_per_stage) for _ in range(stages)
        )

    def forward(self, measured: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        current = measured
        for stage in self.stages:
            current = stage(current, measured, mask)
        return current

