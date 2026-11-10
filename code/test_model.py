"""CPU smoke test that does not require TerraTorch or TerraMind weights."""

import torch
from torch import nn
from torch.nn import functional as F

from losses_simple import L1SAMSSIMLoss
from model_terramamba_cr import TerraMambaCR, TerraMambaConfig
from distributed_utils import DistributedEvalSampler


class DummyFrozenFoundation(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.channels = channels
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, cloudy: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        source = torch.cat([cloudy, sar], dim=1).mean(dim=1, keepdim=True)
        source = F.avg_pool2d(source, kernel_size=16, stride=16)
        return self.scale * source.repeat(1, self.channels, 1, 1)


def main():
    sample_ids = list(range(23))
    partitions = [
        list(DistributedEvalSampler(sample_ids, num_replicas=7, rank=rank))
        for rank in range(7)
    ]
    flattened = [index for partition in partitions for index in partition]
    assert sorted(flattened) == sample_ids
    assert len(flattened) == len(set(flattened))

    foundation = DummyFrozenFoundation()
    model = TerraMambaCR(
        foundation,
        TerraMambaConfig(
            base_channels=8,
            sar_channel_ratio=1.0,
            foundation_dim=64,
            mamba_backend="portable",
            use_spectral_refiner=True,
            spectral_dim=8,
        ),
    )
    cloud = torch.rand(1, 13, 64, 64)
    sar = torch.randn(1, 2, 64, 64) * 5.0 - 15.0
    target = torch.rand(1, 13, 64, 64)
    outputs = model(cloud, sar)
    loss, components = L1SAMSSIMLoss(
        sam_weight=0.1,
        ssim_weight=0.1,
        ssim_scales=3,
    )(outputs, target)
    loss.backward()

    assert outputs["prediction"].shape == target.shape
    assert torch.isfinite(outputs["prediction"]).all()
    expected_sizes = (64, 32, 16, 8, 4)
    for size, evidence in zip(expected_sizes, outputs["evidence_weights"]):
        assert evidence.shape == (1, 3, size, size)
        assert torch.allclose(
            evidence.sum(dim=1),
            torch.ones_like(evidence[:, 0]),
            atol=1e-5,
        )
    assert outputs["reconstruction_confidence"].shape == (1, 1, 64, 64)
    assert set(("l1", "sam_rad", "ssim_loss", "train_ssim")) <= set(
        components
    )
    assert foundation.scale.grad is None
    trainable_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert trainable_gradients
    print("prediction:", tuple(outputs["prediction"].shape))
    print("foundation feature:", tuple(outputs["foundation_feature"].shape))
    print("losses:", {k: float(v) for k, v in components.items()})
    print("parameters:", model.parameter_report())
    print("distributed validation partitions:", partitions)
    print("Frozen TerraMind + Mamba-Conv forward/backward: OK")


if __name__ == "__main__":
    main()
