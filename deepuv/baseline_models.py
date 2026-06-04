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


class FiLMLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, context_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.gamma = nn.Linear(context_dim, out_dim)
        self.beta = nn.Linear(context_dim, out_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.gamma(context) + self.beta(context)


class PolarRecPaperNet(nn.Module):
    """PolarRec-like transformer-conditioned neural field adapted to the fixed HDF5 split.

    The structure follows the imported PolarRec code path: positional encoding
    on sparse UV samples, polar-angle token sorting and group pooling, a
    transformer encoder, and a FiLM-conditioned MLP queried at dense UV points.
    """

    def __init__(
        self,
        fourier_bands: int = 64,
        token_dim: int = 512,
        transformer_layers: int = 4,
        heads: int = 16,
        context_dim: int = 1024,
        output_tokens: int = 8,
        mlp_hidden: int = 256,
        group_size: int = 16,
    ) -> None:
        super().__init__()
        self.fourier_bands = fourier_bands
        self.output_tokens = output_tokens
        self.group_size = group_size
        pe_dim = 2 * (1 + 2 * fourier_bands)
        self.pe_dim = pe_dim
        self.value_embedding = nn.Linear(2, token_dim - pe_dim)
        self.before_pool = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(token_dim // 2, token_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.context_heads = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, context_dim)) for _ in range(output_tokens)
        )
        self.mlp_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        in_dim = pe_dim
        for _ in range(output_tokens - 1):
            self.mlp_layers.append(FiLMLinear(in_dim, mlp_hidden, context_dim))
            self.activations.append(nn.ReLU(inplace=True))
            in_dim = mlp_hidden
        self.final = FiLMLinear(in_dim, 2, context_dim)

    def encode_context(self, sparse_uv: torch.Tensor, sparse_vis: torch.Tensor) -> torch.Tensor:
        sparse_pe = fourier_features(sparse_uv, self.fourier_bands)
        value = self.value_embedding(sparse_vis)
        tokens = torch.cat([sparse_pe, value], dim=-1)
        angles = torch.atan2(sparse_uv[..., 1], sparse_uv[..., 0])
        order = torch.argsort(angles, dim=1)
        gather_idx = order[..., None].expand(-1, -1, tokens.shape[-1])
        tokens = torch.gather(tokens, dim=1, index=gather_idx)
        tokens = self.before_pool(tokens)

        bsz, n_tokens, channels = tokens.shape
        pooled_tokens = max(self.output_tokens, n_tokens // self.group_size)
        if n_tokens % pooled_tokens:
            pad = pooled_tokens - (n_tokens % pooled_tokens)
            tokens = torch.cat([tokens, tokens[:, -1:].expand(-1, pad, -1)], dim=1)
            n_tokens = tokens.shape[1]
        tokens = tokens.reshape(bsz, pooled_tokens, n_tokens // pooled_tokens, channels).mean(dim=2)
        encoded = self.encoder(tokens)
        if encoded.shape[1] < self.output_tokens:
            pad = self.output_tokens - encoded.shape[1]
            encoded = torch.cat([encoded, encoded[:, -1:].expand(-1, pad, -1)], dim=1)
        contexts = [head(encoded[:, i]) for i, head in enumerate(self.context_heads)]
        return torch.stack(contexts, dim=1)

    def forward(self, sparse_uv: torch.Tensor, sparse_vis: torch.Tensor, dense_uv: torch.Tensor) -> torch.Tensor:
        contexts = self.encode_context(sparse_uv, sparse_vis)
        x = fourier_features(dense_uv, self.fourier_bands)
        for i, (layer, activation) in enumerate(zip(self.mlp_layers, self.activations)):
            context = contexts[:, i, :][:, None]
            x = activation(layer(x, context))
        return self.final(x, contexts[:, -1, :][:, None])
