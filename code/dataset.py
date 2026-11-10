"""Dataset for paired Sentinel-2 cloudy / Sentinel-1 / clear Sentinel-2 data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re

import numpy as np
import torch
from osgeo import gdal
from torch.utils.data import Dataset


gdal.UseExceptions()


def read_tif(
    path: str | Path,
    expected_channels: int,
    value_scale: float = 1.0,
    value_offset: float = 0.0,
) -> np.ndarray:
    """Read a TIFF as a contiguous float32 [C,H,W] array.

    Returned values equal ``raw * value_scale + value_offset``.  Optical data
    therefore normally uses ``value_scale=1/10000``.  Standard SEN12MS-CR SAR
    files already store dB values and should use the defaults.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"影像不存在：{path}")
    dataset = None
    try:
        dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
        image = dataset.ReadAsArray()
    except RuntimeError as error:
        raise RuntimeError(f"GDAL无法读取：{path}") from error
    finally:
        dataset = None
    if image is None:
        raise RuntimeError(f"GDAL未返回数据：{path}")
    if image.ndim == 2:
        image = image[None]
    if image.ndim != 3 or image.shape[0] != expected_channels:
        raise ValueError(
            f"{path}应为[{expected_channels},H,W]，实际为{image.shape}"
        )
    image = image.astype(np.float32, copy=False)
    image = image * float(value_scale) + float(value_offset)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(image)


@dataclass(frozen=True)
class DatasetLayout:
    cloud_dir: str = "s2_cloudy"
    sar_dir: str = "s1"
    target_dir: str = "s2"
    mask_dir: str | None = None
    cloud_template: str = "{roi}_s2_cloudy_{patch}.tif"
    sar_template: str = "{roi}_s1_SAR_{patch}.tif"
    target_template: str = "{roi}_s2_Clear_{patch}.tif"
    mask_template: str = "{roi}_s2_CD_{patch}.tif"


class S1S2CloudRemovalDataset(Dataset):
    """Read paired S1/S2 samples from a text list.

    Preferred format: ``folder cloud_image_stem``.  The exact spelling of the
    cloudy filename is retained, so both ``s2_cloudy`` and ``s2_Cloudy`` work.
    The previous three-column ``folder ROI_id patch_id`` format remains
    supported for compatibility.
    """

    CLOUD_STEM_PATTERN = re.compile(
        r"^(?P<roi>ROIs_\d+)_s2_cloudy_(?P<patch>p\d+)$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        root_path: str | Path,
        list_path: str | Path,
        layout: DatasetLayout | None = None,
        expected_size: tuple[int, int] = (256, 256),
        optical_scale: float = 10000.0,
        sar_value_scale: float = 1.0,
        sar_value_offset: float = 0.0,
        augment: bool = False,
        mask_threshold: float = 0.04,
        mask_softness: float = 0.02,
        mask_type: str = "binary",
        mask_scale: float = 1.0,
        list_format: str = "cloud_name",
        clip_optical: bool = True,
        return_mask: bool = True,
    ):
        super().__init__()
        self.root = Path(root_path).expanduser().resolve()
        list_path = Path(list_path).expanduser()
        if not list_path.is_absolute():
            list_path = self.root / list_path
        self.list_path = list_path.resolve()
        self.layout = layout or DatasetLayout()
        self.expected_size = tuple(expected_size)
        self.optical_value_scale = 1.0 / float(optical_scale)
        self.sar_value_scale = float(sar_value_scale)
        self.sar_value_offset = float(sar_value_offset)
        self.augment = augment
        self.mask_threshold = float(mask_threshold)
        self.mask_softness = float(mask_softness)
        self.mask_type = mask_type
        self.mask_scale = float(mask_scale)
        self.list_format = list_format
        self.clip_optical = clip_optical
        self.return_mask = bool(return_mask)

        if self.list_format not in {"cloud_name", "legacy", "auto"}:
            raise ValueError(
                "list_format必须为'cloud_name'、'legacy'或'auto'"
            )
        if self.mask_type not in {"binary", "confidence"}:
            raise ValueError("mask_type必须为'binary'或'confidence'")
        if self.mask_scale <= 0:
            raise ValueError("mask_scale必须大于0")

        if not self.list_path.is_file():
            raise FileNotFoundError(f"数据清单不存在：{self.list_path}")
        self.samples: list[tuple[str, str, str, str]] = []
        with self.list_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if self.list_format == "cloud_name" and len(fields) != 2:
                    raise ValueError(
                        f"{self.list_path}第{line_number}行应为两列："
                        f"文件夹 有云图像主文件名；实际为：{line}"
                    )
                if self.list_format == "legacy" and len(fields) != 3:
                    raise ValueError(
                        f"{self.list_path}第{line_number}行应为旧三列："
                        f"文件夹 ROI编号 patch编号；实际为：{line}"
                    )
                if len(fields) == 2:
                    folder, cloud_name = fields
                    cloud_stem = Path(cloud_name).stem
                    match = self.CLOUD_STEM_PATTERN.fullmatch(cloud_stem)
                    if match is None:
                        raise ValueError(
                            f"{self.list_path}第{line_number}行的有云图像名称"
                            f"不符合ROIs_xx_s2_cloudy_pxxxx格式：{cloud_name}"
                        )
                    roi = match.group("roi")
                    patch = match.group("patch")
                    self.samples.append((folder, cloud_name, roi, patch))
                elif len(fields) == 3:
                    folder, roi, patch = fields
                    values = {"folder": folder, "roi": roi, "patch": patch}
                    cloud_name = self.layout.cloud_template.format(**values)
                    self.samples.append((folder, cloud_name, roi, patch))
                else:
                    raise ValueError(
                        f"{self.list_path}第{line_number}行应包含2列"
                        f"（推荐）或3列（兼容旧格式）：{line}"
                    )
        if not self.samples:
            raise RuntimeError(f"数据清单为空：{self.list_path}")

    @staticmethod
    def _resolve_cloud_path(
        cloud_root: Path, cloud_name: str
    ) -> Path:
        """Resolve a listed cloud image while preserving stem capitalization."""
        listed = Path(cloud_name)
        if listed.suffix:
            return cloud_root / listed.name
        for suffix in (".tif", ".tiff", ".TIF", ".TIFF"):
            candidate = cloud_root / f"{cloud_name}{suffix}"
            if candidate.is_file():
                return candidate
        # Return the conventional path so read_tif emits an informative error.
        return cloud_root / f"{cloud_name}.tif"

    def _sample_paths(
        self, folder: str, cloud_name: str, roi: str, patch: str
    ) -> dict[str, Path | None]:
        values = {"folder": folder, "roi": roi, "patch": patch}
        sample_root = self.root / folder
        layout = self.layout
        mask_path = None
        if layout.mask_dir:
            mask_path = (
                sample_root
                / layout.mask_dir
                / layout.mask_template.format(**values)
            )
        return {
            "cloud": self._resolve_cloud_path(
                sample_root / layout.cloud_dir, cloud_name
            ),
            "sar": sample_root
            / layout.sar_dir
            / layout.sar_template.format(**values),
            "target": sample_root
            / layout.target_dir
            / layout.target_template.format(**values),
            "mask": mask_path,
        }

    def _pseudo_mask(
        self, cloud: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        # B2/B3/B4/B8 carry reliable 10 m evidence of cloud and shadow changes.
        difference = torch.abs(cloud[[1, 2, 3, 7]] - target[[1, 2, 3, 7]])
        difference = difference.mean(dim=0, keepdim=True)
        softness = max(self.mask_softness, 1e-6)
        return torch.sigmoid((difference - self.mask_threshold) / softness)

    @staticmethod
    def _d4_augment(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        rotation = random.randint(0, 3)
        flip = random.random() < 0.5
        output = []
        for tensor in tensors:
            tensor = torch.rot90(tensor, rotation, dims=(-2, -1))
            if flip:
                tensor = torch.flip(tensor, dims=(-1,))
            output.append(tensor.contiguous())
        return output

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        folder, cloud_name, roi, patch = self.samples[index]
        paths = self._sample_paths(folder, cloud_name, roi, patch)
        cloud = torch.from_numpy(
            read_tif(paths["cloud"], 13, self.optical_value_scale)
        )
        target = torch.from_numpy(
            read_tif(paths["target"], 13, self.optical_value_scale)
        )
        sar = torch.from_numpy(
            read_tif(
                paths["sar"],
                2,
                self.sar_value_scale,
                self.sar_value_offset,
            )
        )
        if self.clip_optical:
            cloud = cloud.clamp(0.0, 1.0)
            target = target.clamp(0.0, 1.0)

        expected = self.expected_size
        for name, tensor in (("cloud", cloud), ("sar", sar), ("target", target)):
            if tuple(tensor.shape[-2:]) != expected:
                raise ValueError(
                    f"{folder}/{roi}/{patch}的{name}尺寸应为{expected}，"
                    f"实际为{tuple(tensor.shape[-2:])}"
                )

        mask = None
        if self.return_mask:
            mask_path = paths["mask"]
            if mask_path is not None:
                mask = torch.from_numpy(read_tif(mask_path, 1))
                mask = (mask / self.mask_scale).clamp(0.0, 1.0)
                if self.mask_type == "binary":
                    mask = (mask >= 0.5).to(torch.float32)
            else:
                mask = self._pseudo_mask(cloud, target)

        if self.augment:
            tensors = [cloud, sar, target]
            if mask is not None:
                tensors.append(mask)
            augmented = self._d4_augment(tensors)
            cloud, sar, target = augmented[:3]
            if mask is not None:
                mask = augmented[3]
        name = f"{folder}_{Path(cloud_name).stem}".replace("/", "_")
        sample = {
            "cloud": cloud,
            "sar": sar,
            "target": target,
            "name": name,
            "cloud_path": str(paths["cloud"]),
        }
        if mask is not None:
            sample["mask"] = mask
        return sample

    def __len__(self) -> int:
        return len(self.samples)
