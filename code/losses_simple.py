"""Radiometric, spectral and multi-scale structural losses for PRISMamba-CR."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


# Zero-based indices of Sentinel-2 B2, B3, B4 and B8. These four bands carry
# native 10 m spatial information in the 13-band SEN12MS-CR arrangement.
STRUCTURE_BANDS = (1, 2, 3, 7)


def spectral_angle_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Mean spectral angle in radians over pixels with a valid spectrum."""
    prediction = prediction.float()
    target = target.float()
    numerator = (prediction * target).sum(dim=1)
    denominator = prediction.norm(dim=1) * target.norm(dim=1)
    valid = denominator > epsilon
    cosine = numerator / denominator.clamp_min(epsilon)
    angle = torch.acos(cosine.clamp(-1.0 + 1e-5, 1.0 - 1e-5))
    weight = valid.to(angle.dtype)
    return (angle * weight).sum() / weight.sum().clamp_min(1.0)


def _gaussian_kernel(
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinate = torch.arange(window_size, device=device, dtype=dtype)
    coordinate = coordinate - (window_size - 1) / 2.0
    kernel_1d = torch.exp(-(coordinate.square()) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d)


def ssim_index(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Differentiable mean SSIM over a batch of multi-channel images.

    Computation is forced to FP32 because local variances are numerically
    sensitive under BF16 autocast on DCU. No hard output clamp is used, so
    predictions outside the physical range still receive useful gradients.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"SSIM输入形状不一致：{tuple(prediction.shape)}和{tuple(target.shape)}"
        )
    if prediction.ndim != 4:
        raise ValueError("SSIM输入应为[B,C,H,W]")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("SSIM窗口尺寸必须为正奇数")
    if sigma <= 0 or data_range <= 0:
        raise ValueError("sigma和data_range必须大于0")

    prediction = prediction.float()
    target = target.float()
    channels = prediction.shape[1]
    kernel = _gaussian_kernel(
        window_size,
        sigma,
        prediction.device,
        prediction.dtype,
    ).view(1, 1, window_size, window_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    padding = window_size // 2
    if min(prediction.shape[-2:]) <= padding:
        raise ValueError(
            f"SSIM输入尺寸{tuple(prediction.shape[-2:])}小于窗口要求"
        )

    def local_filter(image: torch.Tensor) -> torch.Tensor:
        image = F.pad(image, (padding, padding, padding, padding), mode="reflect")
        return F.conv2d(image, kernel, groups=channels)

    mean_prediction = local_filter(prediction)
    mean_target = local_filter(target)
    mean_prediction_sq = mean_prediction.square()
    mean_target_sq = mean_target.square()
    mean_cross = mean_prediction * mean_target

    variance_prediction = local_filter(prediction.square()) - mean_prediction_sq
    variance_target = local_filter(target.square()) - mean_target_sq
    covariance = local_filter(prediction * target) - mean_cross

    # Small negative values can be introduced by floating-point cancellation.
    variance_prediction = variance_prediction.clamp_min(0.0)
    variance_target = variance_target.clamp_min(0.0)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mean_cross + c1) * (2.0 * covariance + c2)
    denominator = (
        mean_prediction_sq
        + mean_target_sq
        + c1
    ) * (variance_prediction + variance_target + c2)
    return (numerator / denominator.clamp_min(epsilon)).mean()


def multi_scale_ssim_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scales: int = 3,
    scale_weights: Sequence[float] | None = None,
    bands: Sequence[int] = STRUCTURE_BANDS,
    window_size: int = 11,
    sigma: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weighted multi-scale SSIM loss and its similarity index.

    This is a stable multi-resolution SSIM objective rather than the original
    multiplicative MS-SSIM formulation. It remains well-defined for small
    training patches and is easy to ablate scale by scale.
    """
    if scales <= 0:
        raise ValueError("ssim_scales必须大于0")
    if not bands:
        raise ValueError("SSIM至少需要一个结构波段")
    if min(bands) < 0 or max(bands) >= prediction.shape[1]:
        raise ValueError(f"SSIM波段索引越界：{tuple(bands)}")
    if scale_weights is None:
        # Fine scales receive larger weights because the purpose is to restore
        # roads, rivers and parcel boundaries rather than only global contrast.
        default = (0.5, 0.3, 0.2)
        if scales <= len(default):
            scale_weights = default[:scales]
        else:
            scale_weights = tuple(1.0 / scales for _ in range(scales))
    if len(scale_weights) != scales:
        raise ValueError("scale_weights长度必须等于ssim_scales")
    if any(weight < 0 for weight in scale_weights):
        raise ValueError("SSIM尺度权重不能小于0")
    weight_sum = float(sum(scale_weights))
    if not math.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("SSIM尺度权重之和必须大于0")
    weights = [float(weight) / weight_sum for weight in scale_weights]

    pred = prediction[:, tuple(bands)].float()
    truth = target[:, tuple(bands)].float()
    similarities = []
    for scale in range(scales):
        similarities.append(
            ssim_index(
                pred,
                truth,
                window_size=window_size,
                sigma=sigma,
            )
        )
        if scale + 1 < scales:
            if min(pred.shape[-2:]) < 2:
                raise ValueError("输入尺寸不足以继续计算多尺度SSIM")
            pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
            truth = F.avg_pool2d(truth, kernel_size=2, stride=2)
    similarity = torch.stack(
        [weight * value for weight, value in zip(weights, similarities)]
    ).sum()
    # SSIM can theoretically lie in [-1,1]. Dividing by two keeps the loss
    # approximately in [0,1] and makes its weight easier to interpret.
    loss = 0.5 * (1.0 - similarity)
    return loss, similarity


class L1SAMSSIMLoss(nn.Module):
    """Joint radiometric, spectral-angle and structural objective."""

    def __init__(
        self,
        sam_weight: float = 0.10,
        ssim_weight: float = 0.10,
        ssim_scales: int = 3,
        ssim_window_size: int = 11,
        ssim_sigma: float = 1.5,
        structure_bands: Sequence[int] = STRUCTURE_BANDS,
    ):
        super().__init__()
        if sam_weight < 0 or ssim_weight < 0:
            raise ValueError("sam_weight和ssim_weight不能小于0")
        self.sam_weight = float(sam_weight)
        self.ssim_weight = float(ssim_weight)
        self.ssim_scales = int(ssim_scales)
        self.ssim_window_size = int(ssim_window_size)
        self.ssim_sigma = float(ssim_sigma)
        self.structure_bands = tuple(int(index) for index in structure_bands)

    def forward(
        self,
        outputs: dict[str, torch.Tensor] | torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = outputs["prediction"] if isinstance(outputs, dict) else outputs
        l1 = F.l1_loss(prediction, target)
        # Outer training autocast must not reduce the precision of local SSIM
        # statistics or arccos in the spectral-angle term.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            sam = spectral_angle_loss(prediction, target)
            ssim_loss, ssim = multi_scale_ssim_loss(
                prediction,
                target,
                scales=self.ssim_scales,
                bands=self.structure_bands,
                window_size=self.ssim_window_size,
                sigma=self.ssim_sigma,
            )
        weighted_sam = self.sam_weight * sam
        weighted_ssim = self.ssim_weight * ssim_loss
        total = l1 + weighted_sam + weighted_ssim
        return total, {
            "total": total.detach(),
            "l1": l1.detach(),
            "sam_rad": sam.detach(),
            "weighted_sam": weighted_sam.detach(),
            "ssim_loss": ssim_loss.detach(),
            "weighted_ssim": weighted_ssim.detach(),
            "train_ssim": ssim.detach(),
        }


# Compatibility for scripts that still import L1SAMLoss. New training uses
# L1SAMSSIMLoss explicitly.
class L1SAMLoss(L1SAMSSIMLoss):
    def __init__(self, sam_weight: float = 0.10):
        super().__init__(sam_weight=sam_weight, ssim_weight=0.0)

    def forward(
        self,
        outputs: dict[str, torch.Tensor] | torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = outputs["prediction"] if isinstance(outputs, dict) else outputs
        l1 = F.l1_loss(prediction, target)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            sam = spectral_angle_loss(prediction, target)
        weighted_sam = self.sam_weight * sam
        total = l1 + weighted_sam
        return total, {
            "total": total.detach(),
            "l1": l1.detach(),
            "sam_rad": sam.detach(),
            "weighted_sam": weighted_sam.detach(),
        }
