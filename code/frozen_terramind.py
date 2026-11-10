"""Frozen TerraMind multimodal encoder used in the inference graph.

Unlike teacher-student distillation, this encoder consumes only inputs that are
available at inference time: cloudy Sentinel-2 L1C and co-registered
Sentinel-1 GRD.  Its weights are permanently frozen, while the reconstruction
decoder is optimized for cloud removal.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class FrozenTerraMindEncoder(nn.Module):
    S2L1C_MEAN = (
        2357.089,
        2137.385,
        2018.788,
        2082.986,
        2295.651,
        2854.537,
        3122.849,
        3040.560,
        3306.481,
        1473.847,
        506.070,
        2472.825,
        1838.929,
    )
    S2L1C_STD = (
        1624.683,
        1675.806,
        1557.708,
        1833.702,
        1823.738,
        1733.977,
        1732.131,
        1679.732,
        1727.260,
        1024.687,
        442.165,
        1331.411,
        1160.419,
    )
    S1GRD_MEAN = (-12.599, -20.293)
    S1GRD_STD = (5.195, 5.890)

    def __init__(
        self,
        model_name: str = "terramind_v1_base",
        checkpoint_path: str | Path | None = None,
        expected_dim: int = 768,
    ):
        super().__init__()
        # DTK/ROCm builds used by this project do not provide the CUDA
        # flash-attention shared library expected by some SDPA dispatch paths.
        # The mathematically equivalent SDPA implementation is stable on DCU.
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
                torch.backends.cuda.enable_cudnn_sdp(False)

        try:
            from terratorch import BACKBONE_REGISTRY
        except ImportError as error:
            raise ImportError(
                "FrozenTerraMindEncoder需要terratorch及其TerraMind注册接口"
            ) from error

        options = {
            "pretrained": checkpoint_path is None,
            "modalities": ["S2L1C", "S1GRD"],
            "merge_method": "mean",
        }
        if checkpoint_path is not None:
            checkpoint = Path(checkpoint_path).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"TerraMind权重不存在：{checkpoint}")
            options["ckpt_path"] = str(checkpoint)

        self.backbone = BACKBONE_REGISTRY.build(model_name, **options)
        self.model_name = model_name
        self.expected_dim = int(expected_dim)

        self.register_buffer(
            "s2_mean", torch.tensor(self.S2L1C_MEAN).view(1, 13, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "s2_std", torch.tensor(self.S2L1C_STD).view(1, 13, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "s1_mean", torch.tensor(self.S1GRD_MEAN).view(1, 2, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "s1_std", torch.tensor(self.S1GRD_STD).view(1, 2, 1, 1),
            persistent=False,
        )

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        # model.train() must never reactivate stochastic depth or dropout in the
        # frozen foundation encoder.
        super().train(False)
        self.backbone.eval()
        return self

    @staticmethod
    def _unwrap_tokens(output) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, dict):
            tensors = [value for value in output.values() if torch.is_tensor(value)]
            if not tensors:
                raise TypeError("TerraMind字典输出中没有张量")
            if len(tensors) == 1:
                return tensors[0]
            if all(tensor.shape == tensors[0].shape for tensor in tensors):
                return torch.stack(tensors, dim=0).mean(dim=0)
            raise ValueError("TerraMind多模态输出形状不一致")
        if isinstance(output, (tuple, list)) and output:
            return FrozenTerraMindEncoder._unwrap_tokens(output[-1])
        raise TypeError(f"不支持的TerraMind输出类型：{type(output)!r}")

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 4:
            return tokens
        if tokens.ndim != 3:
            raise ValueError(
                f"TerraMind输出应为[B,N,C]或[B,C,H,W]，实际为{tuple(tokens.shape)}"
            )
        batch, token_count, channels = tokens.shape
        grid = round(token_count**0.5)
        if grid * grid != token_count:
            grid = round((token_count - 1) ** 0.5)
            if grid * grid == token_count - 1:
                tokens = tokens[:, 1:, :]
            else:
                raise ValueError(f"无法将{token_count}个Token恢复为方形特征图")
        return tokens.transpose(1, 2).reshape(batch, channels, grid, grid)

    @torch.no_grad()
    def forward(
        self,
        cloudy_optical: torch.Tensor,
        sar_db: torch.Tensor,
    ) -> torch.Tensor:
        if cloudy_optical.ndim != 4 or cloudy_optical.shape[1] != 13:
            raise ValueError("TerraMind的S2L1C输入应为[B,13,H,W]")
        if sar_db.ndim != 4 or sar_db.shape[1] != 2:
            raise ValueError("TerraMind的S1GRD输入应为[B,2,H,W]")

        s2_dn = cloudy_optical.float() * 10000.0
        s2 = (s2_dn - self.s2_mean) / self.s2_std
        s1 = (sar_db.float() - self.s1_mean) / self.s1_std
        output = self.backbone({"S2L1C": s2, "S1GRD": s1})
        feature = self._tokens_to_map(self._unwrap_tokens(output)).float()
        if feature.shape[1] != self.expected_dim:
            raise ValueError(
                f"TerraMind输出维度为{feature.shape[1]}，期望{self.expected_dim}"
            )
        return feature

