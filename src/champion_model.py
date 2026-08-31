"""Portfolio-safe inference model definitions from the EdgeSynapse AI Track 1 search.

Official model contract:
    input:  (batch, 250, 62)
    output: (batch, 2)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU(alpha=1.0)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


def _pool1d(kind: str, kernel: int) -> nn.Module:
    if kind == "max":
        return nn.MaxPool1d(kernel_size=kernel)
    if kind == "avg":
        return nn.AvgPool1d(kernel_size=kernel)
    raise ValueError(f"Unknown pool kind: {kind}")


class InputNormalizer(nn.Module):
    def __init__(
        self,
        n_channels: int = 62,
        mode: str = "global",
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        clip: float = 8.0,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.clip = float(clip)
        if mean is None:
            mean = torch.zeros(n_channels, dtype=torch.float32)
        if std is None:
            std = torch.ones(n_channels, dtype=torch.float32)
        self.register_buffer("mean", mean.float().reshape(1, n_channels, 1))
        self.register_buffer("std", std.float().clamp_min(1e-6).reshape(1, n_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        if self.mode == "none":
            return x
        if self.mode == "global":
            x = (x - self.mean) / self.std
        elif self.mode == "demean_global":
            x = x - x.mean(dim=-1, keepdim=True)
            x = x / self.std
        elif self.mode == "trial_channel":
            x = x - x.mean(dim=-1, keepdim=True)
            x = x / x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")
        return x.clamp(-self.clip, self.clip)


class RegularizedEEGCNN(nn.Module):
    """Architecture recovery target for the archived 71.76% checkpoint."""

    def __init__(
        self,
        activation: str = "relu",
        pool1_kind: str = "max",
        pool1_kernel: int = 2,
        pool2_kind: str = "max",
        pool2_kernel: int = 2,
        aggregate: str = "avg",
    ) -> None:
        super().__init__()
        if aggregate == "avg":
            agg: nn.Module = nn.AdaptiveAvgPool1d(1)
        elif aggregate == "max":
            agg = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError(f"Unknown aggregate: {aggregate}")

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(62, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            _activation(activation),
            _pool1d(pool1_kind, pool1_kernel),
            nn.Dropout(0.20),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            _activation(activation),
            _pool1d(pool2_kind, pool2_kernel),
            nn.Dropout(0.30),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            _activation(activation),
            agg,
            nn.Dropout(0.35),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.40),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        return self.classifier(self.feature_extractor(x))


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 62,
        n_times: int = 250,
        f1: int = 16,
        depth_multiplier: int = 2,
        f2: int = 32,
        temporal_kernel: int = 51,
        separable_kernel: int = 15,
        dropout: float = 0.35,
        norm_mode: str = "global",
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.norm = InputNormalizer(n_channels, norm_mode, mean, std)
        self.temporal = nn.Conv2d(
            1, f1, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(f1)
        self.spatial = nn.Conv2d(
            f1,
            f1 * depth_multiplier,
            (n_channels, 1),
            groups=f1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(f1 * depth_multiplier)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.depthwise = nn.Conv2d(
            f1 * depth_multiplier,
            f1 * depth_multiplier,
            (1, separable_kernel),
            padding=(0, separable_kernel // 2),
            groups=f1 * depth_multiplier,
            bias=False,
        )
        self.pointwise = nn.Conv2d(f1 * depth_multiplier, f2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        final_t = (n_times // 4) // 8
        self.classifier = nn.Linear(f2 * final_t, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).unsqueeze(1)
        x = self.bn1(self.temporal(x))
        x = self.spatial(x)
        x = F.elu(self.bn2(x))
        x = self.drop1(self.pool1(x))
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = F.elu(self.bn3(x))
        x = self.drop2(self.pool2(x))
        return self.classifier(x.flatten(1))

    @torch.no_grad()
    def constrain(self) -> None:
        w = self.spatial.weight
        flat = w.flatten(1)
        norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(1.0 / norms, max=1.0)
        w.mul_(scale.view(-1, 1, 1, 1))


class ShallowConvNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 62,
        n_times: int = 250,
        n_filters: int = 40,
        temporal_kernel: int = 25,
        pool_kernel: int = 25,
        pool_stride: int = 5,
        dropout: float = 0.50,
        norm_mode: str = "global",
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.norm = InputNormalizer(n_channels, norm_mode, mean, std)
        self.temporal = nn.Conv2d(
            1, n_filters, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False
        )
        self.spatial = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters, momentum=0.1)
        self.pool = nn.AvgPool2d((1, pool_kernel), stride=(1, pool_stride))
        self.drop = nn.Dropout(dropout)
        final_t = (n_times - pool_kernel) // pool_stride + 1
        self.classifier = nn.Linear(n_filters * final_t, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).unsqueeze(1)
        x = self.temporal(x)
        x = self.bn(self.spatial(x))
        x = x.square()
        x = self.pool(x)
        x = torch.log(x.clamp_min(1e-6))
        x = self.drop(x)
        return self.classifier(x.flatten(1))

    @torch.no_grad()
    def constrain(self) -> None:
        w = self.spatial.weight
        flat = w.flatten(1)
        norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(2.0 / norms, max=1.0)
        w.mul_(scale.view(-1, 1, 1, 1))


class FilterBankNet(nn.Module):
    """Fixed FIR filter bank + learned spatial filters + log-variance classifier."""

    def __init__(
        self,
        n_channels: int = 62,
        n_times: int = 250,
        n_bands: int = 7,
        spatial_filters: int = 6,
        fir_taps: int = 31,
        window: int = 50,
        stride: int = 25,
        dropout: float = 0.40,
        norm_mode: str = "global",
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        fir_coeffs: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.spatial_filters = spatial_filters
        self.fir_taps = fir_taps
        self.norm = InputNormalizer(n_channels, norm_mode, mean, std)
        if fir_coeffs is None:
            fir_coeffs = torch.zeros(n_bands, fir_taps, dtype=torch.float32)
        self.register_buffer("fir", fir_coeffs.float())
        self.spatial = nn.Parameter(torch.empty(n_bands, spatial_filters, n_channels))
        nn.init.orthogonal_(self.spatial.view(n_bands * spatial_filters, n_channels))
        n_windows = (n_times - window) // stride + 1
        n_features = n_bands * spatial_filters * n_windows
        self.feature_bn = nn.BatchNorm1d(n_features)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(n_features, 2)
        self.window = window
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        filtered = []
        for b in range(self.n_bands):
            kernel = self.fir[b].view(1, 1, self.fir_taps).repeat(self.n_channels, 1, 1)
            filtered.append(
                F.conv1d(x, kernel, padding=self.fir_taps // 2, groups=self.n_channels)
            )
        x = torch.stack(filtered, dim=1)
        x = torch.einsum("bkct,kfc->bkft", x, self.spatial)
        x = x.reshape(x.shape[0], self.n_bands * self.spatial_filters, x.shape[-1])
        x = F.avg_pool1d(x.square(), kernel_size=self.window, stride=self.stride)
        x = torch.log(x.clamp_min(1e-6)).flatten(1)
        x = self.drop(self.feature_bn(x))
        return self.classifier(x)

    @torch.no_grad()
    def constrain(self) -> None:
        norms = self.spatial.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.spatial.mul_(torch.clamp(2.0 / norms, max=1.0))


class FBCSPClassifier(nn.Module):
    """Filter-bank common spatial patterns with a linear classifier."""

    def __init__(
        self,
        n_channels: int = 62,
        n_bands: int = 7,
        n_spatial: int = 6,
        fir_taps: int = 31,
        scale_channels: bool = False,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.n_spatial = n_spatial
        self.fir_taps = fir_taps
        self.scale_channels = bool(scale_channels)
        self.register_buffer("fir", torch.zeros(n_bands, fir_taps))
        self.register_buffer("csp", torch.zeros(n_bands, n_spatial, n_channels))
        self.register_buffer("channel_std", torch.ones(n_channels))
        self.register_buffer("feature_mean", torch.zeros(n_bands * n_spatial))
        self.register_buffer("feature_std", torch.ones(n_bands * n_spatial))
        self.classifier = nn.Linear(n_bands * n_spatial, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = x - x.mean(dim=-1, keepdim=True)
        if self.scale_channels:
            x = x / self.channel_std.view(1, -1, 1).clamp_min(1e-6)
        features = []
        for b in range(self.n_bands):
            kernel = self.fir[b].view(1, 1, self.fir_taps).repeat(self.n_channels, 1, 1)
            band = F.conv1d(x, kernel, padding=self.fir_taps // 2, groups=self.n_channels)
            projected = torch.einsum("bct,fc->bft", band, self.csp[b])
            variance = projected.var(dim=-1, unbiased=False).clamp_min(1e-8)
            variance = variance / variance.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            features.append(torch.log(variance))
        features = torch.cat(features, dim=1)
        features = (features - self.feature_mean) / self.feature_std.clamp_min(1e-6)
        return self.classifier(features)


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = dilation * 4
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size=9,
            padding=pad,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.pw(y)
        y = self.drop(F.gelu(self.bn(y)))
        return x + y


class MultiScaleTCN(nn.Module):
    def __init__(
        self,
        n_channels: int = 62,
        branch_filters: int = 32,
        kernels: Sequence[int] = (7, 15, 31),
        dropout: float = 0.30,
        norm_mode: str = "global",
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.norm = InputNormalizer(n_channels, norm_mode, mean, std)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(n_channels, branch_filters, k, padding=k // 2, bias=False)
                for k in kernels
            ]
        )
        width = branch_filters * len(kernels)
        self.bn = nn.BatchNorm1d(width)
        self.blocks = nn.Sequential(
            DepthwiseResidualBlock(width, 1, dropout),
            DepthwiseResidualBlock(width, 2, dropout),
            DepthwiseResidualBlock(width, 4, dropout),
        )
        self.attention = nn.Conv1d(width, 1, kernel_size=1)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(width, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = F.gelu(self.bn(x))
        x = self.blocks(x)
        weights = torch.softmax(self.attention(x), dim=-1)
        x = (x * weights).sum(dim=-1)
        return self.classifier(self.drop(x))


def build_model(kind: str, config: Dict[str, Any]) -> nn.Module:
    if kind == "regularized_cnn":
        return RegularizedEEGCNN(**config)
    if kind == "eegnet":
        return EEGNet(**config)
    if kind == "shallow":
        return ShallowConvNet(**config)
    if kind == "filterbank":
        return FilterBankNet(**config)
    if kind == "multiscale_tcn":
        return MultiScaleTCN(**config)
    if kind == "fbcsp":
        return FBCSPClassifier(**config)
    raise ValueError(f"Unknown model kind: {kind}")


def shift_time(x: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return x
    out = torch.zeros_like(x)
    if shift > 0:
        out[:, shift:, :] = x[:, :-shift, :]
    else:
        s = -shift
        out[:, :-s, :] = x[:, s:, :]
    return out


class ChampionEnsemble(nn.Module):
    def __init__(self, checkpoint: Dict[str, Any]) -> None:
        super().__init__()
        self.models = nn.ModuleList()
        self.tta_shifts: List[List[int]] = []
        coefficients = []
        for member in checkpoint["members"]:
            model = build_model(member["kind"], member["config"])
            model.load_state_dict(member["state_dict"], strict=True)
            model.eval()
            self.models.append(model)
            self.tta_shifts.append([int(s) for s in member.get("tta_shifts", [0])])
            coefficients.append(float(member["coefficient"]))
        coeff = torch.tensor(coefficients, dtype=torch.float32)
        coeff = coeff / coeff.sum().clamp_min(1e-8)
        self.register_buffer("coefficients", coeff)
        threshold = float(checkpoint.get("margin_threshold", 0.0))
        self.register_buffer(
            "class_bias",
            torch.tensor([threshold / 2.0, -threshold / 2.0], dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = None
        for idx, model in enumerate(self.models):
            logits = None
            shifts = self.tta_shifts[idx]
            for shift in shifts:
                current = model(shift_time(x, shift))
                logits = current if logits is None else logits + current
            logits = logits / float(len(shifts))
            weighted = self.coefficients[idx] * logits
            total = weighted if total is None else total + weighted
        return total + self.class_bias


def load_champion(
    path: str,
    map_location: str | torch.device = "cpu",
) -> ChampionEnsemble:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    return ChampionEnsemble(checkpoint)
