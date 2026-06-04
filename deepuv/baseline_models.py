from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def fourier_features(x: torch.Tensor, bands: int = 8) -> torch.Tensor:
    features = [x]
    for freq in range(1, bands + 1):
        angle = 2.0 * math.pi * float(freq) * x
        features.extend([torch.sin(angle), torch.cos(angle)])
    return torch.cat(features, dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 32, depth: int = 3) -> None:
        super().__init__()
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        channels = in_channels
        skips: list[int] = []
        for level in range(depth):
            out = base_channels * (2**level)
            self.down_blocks.append(ConvBlock(channels, out))
            self.pools.append(nn.MaxPool2d(2))
            skips.append(out)
            channels = out
        self.bottleneck = ConvBlock(channels, channels * 2)
        channels *= 2
        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for skip_channels in reversed(skips):
            self.up_convs.append(nn.ConvTranspose2d(channels, skip_channels, 2, stride=2))
            self.up_blocks.append(ConvBlock(skip_channels * 2, skip_channels))
            channels = skip_channels
        self.out = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block, pool in zip(self.down_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, block, skip in zip(self.up_convs, self.up_blocks, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.out(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class EDSRImage(nn.Module):
    def __init__(self, in_channels: int = 1, channels: int = 64, blocks: int = 8) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.body = nn.Sequential(*(ResBlock(channels) for _ in range(blocks)), nn.Conv2d(channels, channels, 3, padding=1))
        self.tail = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head = self.head(x)
        return torch.sigmoid(self.tail(head + self.body(head)))


class R2D2ImageSeries(nn.Module):
    def __init__(self, steps: int = 5, channels: int = 32) -> None:
        super().__init__()
        self.steps = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(2, channels, 3, padding=1),
                nn.SiLU(inplace=True),
                ResBlock(channels),
                ResBlock(channels),
                nn.Conv2d(channels, 1, 3, padding=1),
            )
            for _ in range(steps)
        )

    def forward(self, dirty: torch.Tensor) -> torch.Tensor:
        current = dirty
        for step in self.steps:
            residual = dirty - current
            current = torch.clamp(current + step(torch.cat([current, residual], dim=1)), 0.0, 1.0)
        return current


class PolarNeuralField(nn.Module):
    """Compact PolarRec-style transformer-conditioned neural field."""

    def __init__(
        self,
        fourier_bands: int = 8,
        token_dim: int = 128,
        context_dim: int = 128,
        transformer_layers: int = 2,
        heads: int = 4,
        mlp_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.fourier_bands = fourier_bands
        pe_dim = 2 * (1 + 2 * fourier_bands)
        self.token_in = nn.Linear(pe_dim + 2, token_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.context = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, context_dim), nn.GELU())
        self.query = nn.Sequential(
            nn.Linear(pe_dim + context_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 2),
        )

    def forward(self, sparse_uv: torch.Tensor, sparse_vis: torch.Tensor, dense_uv: torch.Tensor) -> torch.Tensor:
        sparse_pe = fourier_features(sparse_uv, self.fourier_bands)
        dense_pe = fourier_features(dense_uv, self.fourier_bands)
        tokens = self.token_in(torch.cat([sparse_pe, sparse_vis], dim=-1))
        encoded = self.encoder(tokens)
        context = self.context(encoded.mean(dim=1))
        context = context[:, None].expand(-1, dense_pe.shape[1], -1)
        return self.query(torch.cat([dense_pe, context], dim=-1))

