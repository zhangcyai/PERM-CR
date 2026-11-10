"""PRISMamba-CR: prior-conditioned reliability-aware TerraMamba.

This is a drop-in evolution of TerraMambaCR. It keeps the frozen TerraMind
encoder and the spatial/spectral Mamba decoder, but replaces premature S2-SAR
fusion with modality-specific encoders, a TerraMind prior pyramid and
prior-conditioned evidence routing. No cloud mask is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from model_fpmamba_cr import (
    ConvNormAct,
    DownsampleStage,
    LayerNorm2d,
    ResidualConvBlock,
    SpatialMambaBlock,
    SpectralMambaRefiner,
)


@dataclass(frozen=True)
class TerraMambaConfig:
    base_channels: int = 48
    sar_channel_ratio: float = 0.5
    foundation_dim: int = 768
    mamba_backend: str = "mamba_ssm"
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_fast_path: bool = False
    use_spectral_refiner: bool = True
    spectral_dim: int = 32
    spectral_pool_size: int = 4
    use_foundation_prior: bool = True
    use_evidence_router: bool = True
    use_spatial_mamba: bool = True
    router_temperature: float = 1.0
    clear_residual_floor: float = 0.05
    fusion_mamba_scales: tuple[int, ...] = (2, 3, 4)
    decoder_mamba_scales: tuple[int, ...] = (2, 3)


def _validate_config(config: TerraMambaConfig) -> None:
    if config.base_channels <= 0:
        raise ValueError("base_channels必须大于0")
    if not 0.0 < config.sar_channel_ratio <= 1.0:
        raise ValueError("sar_channel_ratio必须位于(0,1]")
    if config.router_temperature <= 0:
        raise ValueError("router_temperature必须大于0")
    if not 0.0 <= config.clear_residual_floor <= 1.0:
        raise ValueError("clear_residual_floor必须位于[0,1]")
    for scale in (*config.fusion_mamba_scales, *config.decoder_mamba_scales):
        if scale not in {0, 1, 2, 3, 4}:
            raise ValueError(f"Mamba尺度索引必须位于0至4，实际为{scale}")


class ModalitySpecificEncoder(nn.Module):
    """Preserve optical radiometry and SAR geometry before evidence fusion."""

    S1_MEAN = (-12.599, -20.293)
    S1_STD = (5.195, 5.890)

    def __init__(
        self,
        optical_channels: tuple[int, int, int, int, int],
        sar_channels: tuple[int, int, int, int, int],
    ):
        super().__init__()
        o0, o1, o2, o3, o4 = optical_channels
        s0, s1, s2, s3, s4 = sar_channels
        self.optical_stem = nn.Sequential(
            ConvNormAct(13, o0), ResidualConvBlock(o0)
        )
        self.optical_down = nn.ModuleList(
            [
                DownsampleStage(o0, o1, depth=2),
                DownsampleStage(o1, o2, depth=2),
                DownsampleStage(o2, o3, depth=2),
                DownsampleStage(o3, o4, depth=2),
            ]
        )
        self.sar_stem = nn.Sequential(
            ConvNormAct(2, s0), ResidualConvBlock(s0)
        )
        self.sar_down = nn.ModuleList(
            [
                DownsampleStage(s0, s1, depth=1),
                DownsampleStage(s1, s2, depth=1),
                DownsampleStage(s2, s3, depth=1),
                DownsampleStage(s3, s4, depth=1),
            ]
        )
        self.register_buffer(
            "sar_mean",
            torch.tensor(self.S1_MEAN).view(1, 2, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "sar_std",
            torch.tensor(self.S1_STD).view(1, 2, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _encode(stem, stages, image: torch.Tensor) -> list[torch.Tensor]:
        features = [stem(image)]
        for stage in stages:
            features.append(stage(features[-1]))
        return features

    def forward(
        self, cloudy_optical: torch.Tensor, sar_db: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        optical = self._encode(
            self.optical_stem, self.optical_down, cloudy_optical
        )
        normalized_sar = (sar_db - self.sar_mean) / self.sar_std
        sar = self._encode(self.sar_stem, self.sar_down, normalized_sar)
        return optical, sar


class FoundationPriorPyramid(nn.Module):
    """Convert a coarse TerraMind representation into a semantic pyramid."""

    def __init__(
        self,
        foundation_dim: int,
        channels: tuple[int, int, int, int, int],
    ):
        super().__init__()
        c0, c1, c2, c3, c4 = channels
        self.top = nn.Sequential(
            LayerNorm2d(foundation_dim),
            ConvNormAct(foundation_dim, c4, kernel_size=1),
            ResidualConvBlock(c4),
        )
        self.laterals = nn.ModuleList(
            [
                ConvNormAct(c4, c3),
                ConvNormAct(c3, c2),
                ConvNormAct(c2, c1),
                ConvNormAct(c1, c0),
            ]
        )

    def forward(
        self,
        foundation: torch.Tensor,
        reference: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        p4 = self.top(foundation)
        if p4.shape[-2:] != reference[4].shape[-2:]:
            p4 = F.interpolate(
                p4,
                size=reference[4].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        descending = [p4]
        current = p4
        for output_scale, lateral in zip((3, 2, 1, 0), self.laterals):
            current = F.interpolate(
                current,
                size=reference[output_scale].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            current = lateral(current)
            descending.append(current)
        p4, p3, p2, p1, p0 = descending
        return [p0, p1, p2, p3, p4]


class PriorConditionedEvidenceRouter(nn.Module):
    """Route optical, SAR and foundation evidence with per-pixel weights.

    The router explicitly uses prior-observation discrepancies. Its softmax
    weights are relative evidence support, not an external cloud mask.
    """

    def __init__(
        self,
        optical_channels: int,
        sar_channels: int,
        use_router: bool,
        temperature: float,
        use_mamba: bool,
        config: TerraMambaConfig,
    ):
        super().__init__()
        self.use_router = bool(use_router)
        self.temperature = float(temperature)
        self.optical_adapter = nn.Sequential(
            LayerNorm2d(optical_channels),
            ConvNormAct(optical_channels, optical_channels, kernel_size=1),
        )
        self.sar_adapter = nn.Sequential(
            LayerNorm2d(sar_channels),
            ConvNormAct(sar_channels, optical_channels, kernel_size=1),
        )
        self.prior_adapter = nn.Sequential(
            LayerNorm2d(optical_channels),
            ConvNormAct(optical_channels, optical_channels, kernel_size=1),
        )
        if self.use_router:
            hidden = max(16, optical_channels // 2)
            self.router = nn.Sequential(
                ConvNormAct(optical_channels * 5, hidden, kernel_size=1),
                ConvNormAct(hidden, hidden, groups=hidden),
                nn.Conv2d(hidden, 3, kernel_size=1),
            )
            nn.init.zeros_(self.router[-1].weight)
            with torch.no_grad():
                # Conservative optical-dominant initialization.
                self.router[-1].bias.copy_(torch.tensor([1.5, -0.5, 0.0]))
        else:
            self.router = None

        self.fusion = nn.Sequential(
            ConvNormAct(optical_channels * 3, optical_channels, kernel_size=1),
            ResidualConvBlock(optical_channels),
        )
        self.refine = (
            SpatialMambaBlock(
                optical_channels,
                config.mamba_backend,
                config.mamba_d_state,
                config.mamba_d_conv,
                config.mamba_expand,
                config.mamba_fast_path,
            )
            if use_mamba
            else ResidualConvBlock(optical_channels)
        )
        self.residual_scale = nn.Parameter(
            torch.full((1, optical_channels, 1, 1), 1e-2)
        )

    def forward(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
        prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        optical = self.optical_adapter(optical)
        sar = self.sar_adapter(sar)
        prior = self.prior_adapter(prior)
        if self.router is None:
            weights = optical.new_tensor([1.0 / 3.0] * 3).view(1, 3, 1, 1)
            weights = weights.expand(optical.shape[0], -1, *optical.shape[-2:])
        else:
            evidence = torch.cat(
                [
                    optical,
                    sar,
                    prior,
                    (optical - prior).abs(),
                    (sar - prior).abs(),
                ],
                dim=1,
            )
            weights = torch.softmax(
                self.router(evidence) / self.temperature, dim=1
            )
        routed = self.fusion(
            torch.cat(
                [
                    weights[:, 0:1] * optical,
                    weights[:, 1:2] * sar,
                    weights[:, 2:3] * prior,
                ],
                dim=1,
            )
        )
        refined = self.refine(routed)
        return optical + self.residual_scale * refined, weights


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
        self.refine = (
            SpatialMambaBlock(
                out_channels,
                config.mamba_backend,
                config.mamba_d_state,
                config.mamba_d_conv,
                config.mamba_expand,
                config.mamba_fast_path,
            )
            if use_mamba
            else ResidualConvBlock(out_channels)
        )

    def forward(self, feature: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        feature = F.interpolate(
            feature,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.refine(self.fuse(torch.cat([feature, skip], dim=1)))


class TerraMambaCR(nn.Module):
    """Prior-conditioned reliability-aware SAR-assisted cloud removal."""

    def __init__(
        self,
        foundation_encoder: nn.Module,
        config: TerraMambaConfig | None = None,
    ):
        super().__init__()
        self.config = config or TerraMambaConfig()
        _validate_config(self.config)
        cfg = self.config
        c = cfg.base_channels
        optical_channels = (c, c * 2, c * 4, c * 6, c * 8)
        sar_channels = tuple(
            max(16, int(round(channel * cfg.sar_channel_ratio)))
            for channel in optical_channels
        )
        self.optical_channels = optical_channels
        self.sar_channels = sar_channels
        self.foundation_encoder = foundation_encoder
        for parameter in self.foundation_encoder.parameters():
            parameter.requires_grad = False

        self.modality_encoder = ModalitySpecificEncoder(
            optical_channels, sar_channels
        )
        self.prior_pyramid = (
            FoundationPriorPyramid(cfg.foundation_dim, optical_channels)
            if cfg.use_foundation_prior
            else None
        )
        self.evidence_fusion = nn.ModuleList(
            [
                PriorConditionedEvidenceRouter(
                    optical_channels[index],
                    sar_channels[index],
                    cfg.use_evidence_router,
                    cfg.router_temperature,
                    cfg.use_spatial_mamba
                    and index in cfg.fusion_mamba_scales,
                    cfg,
                )
                for index in range(5)
            ]
        )
        self.up3 = HybridUpBlock(
            optical_channels[4], optical_channels[3], optical_channels[3],
            cfg, cfg.use_spatial_mamba and 3 in cfg.decoder_mamba_scales,
        )
        self.up2 = HybridUpBlock(
            optical_channels[3], optical_channels[2], optical_channels[2],
            cfg, cfg.use_spatial_mamba and 2 in cfg.decoder_mamba_scales,
        )
        self.up1 = HybridUpBlock(
            optical_channels[2], optical_channels[1], optical_channels[1],
            cfg, cfg.use_spatial_mamba and 1 in cfg.decoder_mamba_scales,
        )
        self.up0 = HybridUpBlock(
            optical_channels[1], optical_channels[0], optical_channels[0],
            cfg, cfg.use_spatial_mamba and 0 in cfg.decoder_mamba_scales,
        )
        self.reconstruction_head = nn.Sequential(
            ResidualConvBlock(optical_channels[0]),
            nn.Conv2d(optical_channels[0], 13, kernel_size=1),
        )
        nn.init.zeros_(self.reconstruction_head[-1].weight)
        nn.init.zeros_(self.reconstruction_head[-1].bias)
        self.spectral_refiner = (
            SpectralMambaRefiner(
                backend=cfg.mamba_backend,
                dim=cfg.spectral_dim,
                pool_size=cfg.spectral_pool_size,
                use_fast_path=cfg.mamba_fast_path,
            )
            if cfg.use_spectral_refiner
            else None
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.foundation_encoder.eval()
        return self

    @staticmethod
    def _unwrap_foundation(output) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, dict):
            for key in ("feature", "factual", "foundation_feature", "out"):
                if key in output and torch.is_tensor(output[key]):
                    return output[key]
            tensors = [value for value in output.values() if torch.is_tensor(value)]
            if len(tensors) == 1:
                return tensors[0]
        if isinstance(output, (list, tuple)) and output:
            return TerraMambaCR._unwrap_foundation(output[-1])
        raise TypeError(f"无法解析TerraMind输出类型：{type(output)!r}")

    def _make_prior_pyramid(
        self,
        cloudy_optical: torch.Tensor,
        sar_db: torch.Tensor,
        optical: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if self.prior_pyramid is None:
            priors = [torch.zeros_like(feature) for feature in optical]
            return priors, priors[-1]
        with torch.no_grad():
            foundation = self.foundation_encoder(cloudy_optical, sar_db)
        foundation = self._unwrap_foundation(foundation).to(optical[-1].dtype)
        priors = self.prior_pyramid(foundation, optical)
        return priors, priors[-1]

    def forward(
        self, cloudy_optical: torch.Tensor, sar_db: torch.Tensor
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if cloudy_optical.ndim != 4 or cloudy_optical.shape[1] != 13:
            raise ValueError("cloudy_optical应为[B,13,H,W]")
        if sar_db.ndim != 4 or sar_db.shape[1] != 2:
            raise ValueError("sar_db应为[B,2,H,W]")
        if cloudy_optical.shape[-2:] != sar_db.shape[-2:]:
            raise ValueError("S2与SAR空间尺寸必须一致")
        if cloudy_optical.shape[-2] % 16 or cloudy_optical.shape[-1] % 16:
            raise ValueError("输入高度和宽度必须是16的整数倍")

        optical, sar = self.modality_encoder(cloudy_optical, sar_db)
        priors, coarse_foundation = self._make_prior_pyramid(
            cloudy_optical, sar_db, optical
        )
        fused: list[torch.Tensor] = []
        evidence_weights: list[torch.Tensor] = []
        for fusion, optical_feature, sar_feature, prior_feature in zip(
            self.evidence_fusion, optical, sar, priors
        ):
            feature, weights = fusion(
                optical_feature, sar_feature, prior_feature
            )
            fused.append(feature)
            evidence_weights.append(weights)

        feature = self.up3(fused[4], fused[3])
        feature = self.up2(feature, fused[2])
        feature = self.up1(feature, fused[1])
        feature = self.up0(feature, fused[0])
        residual = self.reconstruction_head(feature)

        optical_confidence = evidence_weights[0][:, 0:1]
        reconstruction_confidence = self.config.clear_residual_floor + (
            1.0 - self.config.clear_residual_floor
        ) * (1.0 - optical_confidence)
        preliminary = cloudy_optical + reconstruction_confidence * residual
        prediction = (
            self.spectral_refiner(preliminary, reconstruction_confidence)
            if self.spectral_refiner is not None else preliminary
        )
        # Do not clamp during training. Clamp detached validation/export data.
        return {
            "prediction": prediction,
            "preliminary": preliminary,
            "residual": residual,
            "foundation_feature": coarse_foundation,
            "foundation_pyramid": priors,
            "evidence_weights": evidence_weights,
            "optical_confidence": optical_confidence,
            "sar_confidence": evidence_weights[0][:, 1:2],
            "prior_confidence": evidence_weights[0][:, 2:3],
            "reconstruction_confidence": reconstruction_confidence,
        }

    def parameter_report(self) -> dict[str, int | str]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters()
            if parameter.requires_grad
        )
        foundation = sum(
            parameter.numel() for parameter in self.foundation_encoder.parameters()
        )
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_foundation_parameters": foundation,
            "mamba_backend": self.config.mamba_backend,
            "method": "PRISMamba-CR",
        }


if __name__ == "__main__":
    class DummyFoundation(nn.Module):
        def forward(self, optical, sar):
            feature = F.avg_pool2d(
                torch.cat([optical, sar], dim=1).mean(dim=1, keepdim=True),
                kernel_size=16, stride=16,
            )
            return feature.repeat(1, 64, 1, 1)

    model = TerraMambaCR(
        DummyFoundation(),
        TerraMambaConfig(
            base_channels=8,
            sar_channel_ratio=1.0,
            foundation_dim=64,
            mamba_backend="portable",
            spectral_dim=8,
        ),
    )
    cloudy = torch.rand(1, 13, 64, 64)
    sar = torch.randn(1, 2, 64, 64) * 5.0 - 15.0
    output = model(cloudy, sar)
    output["prediction"].mean().backward()
    print("prediction:", tuple(output["prediction"].shape))
    print("evidence:", [tuple(item.shape) for item in output["evidence_weights"]])
    print(model.parameter_report())
