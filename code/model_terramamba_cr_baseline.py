"""TerraMamba-CR: frozen TerraMind with a trainable Mamba-Conv decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from model_fpmamba_cr import (
    ConvNormAct,
    DownsampleStage,
    ResidualConvBlock,
    SpatialMambaBlock,
    SpectralMambaRefiner,
)


@dataclass(frozen=True)
class TerraMambaConfig:
    base_channels: int = 48
    foundation_dim: int = 768
    mamba_backend: str = "mamba_ssm"
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_fast_path: bool = False
    use_spectral_refiner: bool = True
    spectral_dim: int = 32
    spectral_pool_size: int = 4


class LocalMultimodalEncoder(nn.Module):
    """Preserve high-resolution S2 detail and local SAR structures."""

    S1_MEAN = (-12.599, -20.293)
    S1_STD = (5.195, 5.890)

    def __init__(self, channels: tuple[int, int, int, int, int]):
        super().__init__()
        c0, c1, c2, c3, c4 = channels
        self.optical_stem = nn.Sequential(
            ConvNormAct(13, c0),
            ResidualConvBlock(c0),
        )
        sar_channels = max(16, c0 // 2)
        self.sar_stem = nn.Sequential(
            ConvNormAct(2, sar_channels),
            ResidualConvBlock(sar_channels),
        )
        self.initial_fusion = nn.Sequential(
            ConvNormAct(c0 + sar_channels, c0, kernel_size=1),
            ResidualConvBlock(c0),
        )
        self.down1 = DownsampleStage(c0, c1, depth=2)
        self.down2 = DownsampleStage(c1, c2, depth=2)
        self.down3 = DownsampleStage(c2, c3, depth=2)
        self.down4 = DownsampleStage(c3, c4, depth=2)

        self.register_buffer(
            "sar_mean", torch.tensor(self.S1_MEAN).view(1, 2, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "sar_std", torch.tensor(self.S1_STD).view(1, 2, 1, 1),
            persistent=False,
        )

    def forward(
        self, cloudy_optical: torch.Tensor, sar_db: torch.Tensor
    ) -> list[torch.Tensor]:
        optical = self.optical_stem(cloudy_optical)
        normalized_sar = (sar_db - self.sar_mean) / self.sar_std
        sar = self.sar_stem(normalized_sar)
        f0 = self.initial_fusion(torch.cat([optical, sar], dim=1))
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        f4 = self.down4(f3)
        return [f0, f1, f2, f3, f4]


class HybridUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        config: TerraMambaConfig,
        use_mamba: bool,
    ):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvNormAct(in_channels + skip_channels, out_channels),
            ResidualConvBlock(out_channels),
        )
        if use_mamba:
            self.refine = SpatialMambaBlock(
                out_channels,
                config.mamba_backend,
                config.mamba_d_state,
                config.mamba_d_conv,
                config.mamba_expand,
                config.mamba_fast_path,
            )
        else:
            self.refine = ResidualConvBlock(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.refine(self.fuse(torch.cat([x, skip], dim=1)))


class TerraMambaCR(nn.Module):
    """Cloud removal network with a frozen TerraMind encoder in its forward path.

    Inputs:
        cloudy_optical: [B, 13, H, W], reflectance scaled to [0, 1]
        sar_db: [B, 2, H, W], Sentinel-1 VV/VH in dB
    Output:
        dictionary containing prediction [B, 13, H, W]
    """

    def __init__(
        self,
        foundation_encoder: nn.Module,
        config: TerraMambaConfig | None = None,
    ):
        super().__init__()
        self.config = config or TerraMambaConfig()
        cfg = self.config
        c0 = cfg.base_channels
        channels = (c0, c0 * 2, c0 * 4, c0 * 6, c0 * 8)
        self.foundation_encoder = foundation_encoder
        for parameter in self.foundation_encoder.parameters():
            parameter.requires_grad = False

        self.local_encoder = LocalMultimodalEncoder(channels)
        self.foundation_projection = nn.Sequential(
            ConvNormAct(cfg.foundation_dim, channels[4], kernel_size=1),
            ResidualConvBlock(channels[4]),
        )
        self.bottleneck_fusion = nn.Sequential(
            ConvNormAct(channels[4] * 2, channels[4], kernel_size=1),
            ResidualConvBlock(channels[4]),
            SpatialMambaBlock(
                channels[4],
                cfg.mamba_backend,
                cfg.mamba_d_state,
                cfg.mamba_d_conv,
                cfg.mamba_expand,
                cfg.mamba_fast_path,
            ),
        )

        self.up3 = HybridUpBlock(channels[4], channels[3], channels[3], cfg, True)
        self.up2 = HybridUpBlock(channels[3], channels[2], channels[2], cfg, True)
        self.up1 = HybridUpBlock(channels[2], channels[1], channels[1], cfg, False)
        self.up0 = HybridUpBlock(channels[1], channels[0], channels[0], cfg, False)
        self.reconstruction_head = nn.Sequential(
            ResidualConvBlock(channels[0]),
            nn.Conv2d(channels[0], 13, kernel_size=1),
        )
        final_layer = self.reconstruction_head[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

        if cfg.use_spectral_refiner:
            self.spectral_refiner = SpectralMambaRefiner(
                backend=cfg.mamba_backend,
                dim=cfg.spectral_dim,
                pool_size=cfg.spectral_pool_size,
                use_fast_path=cfg.mamba_fast_path,
            )
        else:
            self.spectral_refiner = nn.Identity()

    def train(self, mode: bool = True):
        super().train(mode)
        # Ensure the frozen foundation branch remains deterministic even when
        # the decoder is in training mode.
        self.foundation_encoder.eval()
        return self

    def forward(
        self, cloudy_optical: torch.Tensor, sar_db: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if cloudy_optical.ndim != 4 or cloudy_optical.shape[1] != 13:
            raise ValueError("cloudy_optical应为[B,13,H,W]")
        if sar_db.ndim != 4 or sar_db.shape[1] != 2:
            raise ValueError("sar_db应为[B,2,H,W]")
        if cloudy_optical.shape[-2:] != sar_db.shape[-2:]:
            raise ValueError("S2与SAR空间尺寸必须一致")

        with torch.no_grad():
            foundation = self.foundation_encoder(cloudy_optical, sar_db)
        local = self.local_encoder(cloudy_optical, sar_db)
        foundation = self.foundation_projection(foundation.to(local[-1].dtype))
        if foundation.shape[-2:] != local[-1].shape[-2:]:
            foundation = F.interpolate(
                foundation,
                size=local[-1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = self.bottleneck_fusion(torch.cat([local[-1], foundation], dim=1))
        x = self.up3(x, local[3])
        x = self.up2(x, local[2])
        x = self.up1(x, local[1])
        x = self.up0(x, local[0])
        residual = self.reconstruction_head(x)
        preliminary = torch.clamp(cloudy_optical + residual, 0.0, 1.0)
        prediction = torch.clamp(self.spectral_refiner(preliminary), 0.0, 1.0)
        return {
            "prediction": prediction,
            "preliminary": preliminary,
            "residual": residual,
            "foundation_feature": foundation,
        }

    def parameter_report(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        foundation = sum(
            parameter.numel()
            for parameter in self.foundation_encoder.parameters()
        )
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_foundation_parameters": foundation,
        }

