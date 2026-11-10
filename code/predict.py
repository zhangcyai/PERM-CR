"""Run PRISMamba-CR inference and save georeferenced 13-band GeoTIFFs."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from osgeo import gdal
from torch.utils.data import DataLoader

from dataset import DatasetLayout, S1S2CloudRemovalDataset
from frozen_terramind import FrozenTerraMindEncoder
from model_terramamba_cr import TerraMambaCR, TerraMambaConfig


def load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # This fallback is appropriate only for a checkpoint created by this
        # project (or another source the user explicitly trusts).
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_geotiff(path: Path, source_path: str, image: np.ndarray):
    source = gdal.Open(source_path, gdal.GA_ReadOnly)
    if source is None:
        raise FileNotFoundError(f"无法读取源GeoTIFF：{source_path}")
    height, width = image.shape[-2:]
    output = gdal.GetDriverByName("GTiff").Create(
        str(path),
        width,
        height,
        image.shape[0],
        gdal.GDT_UInt16,
        options=["COMPRESS=DEFLATE", "TILED=YES"],
    )
    output.SetGeoTransform(source.GetGeoTransform())
    output.SetProjection(source.GetProjection())
    digital_number = np.clip(image * 10000.0, 0, 10000).round().astype(np.uint16)
    for band_index in range(digital_number.shape[0]):
        output.GetRasterBand(band_index + 1).WriteArray(digital_number[band_index])
    output.FlushCache()
    output = None
    source = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", default="../../data")
    parser.add_argument("--test-list", default="data_txt/test.txt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--foundation-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp-dtype", choices=("bf16", "none"), default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("未检测到DCU")
    device = torch.device("cuda", 0)
    state = load_checkpoint(args.checkpoint)
    saved_args = state.get("arguments", {})
    config = TerraMambaConfig(
        base_channels=int(saved_args.get("base_channels", 48)),
        sar_channel_ratio=float(saved_args.get("sar_channel_ratio", 0.5)),
        foundation_dim=int(saved_args.get("foundation_dim", 768)),
        mamba_backend=str(saved_args.get("mamba_backend", "mamba_ssm")),
        mamba_d_state=int(saved_args.get("mamba_d_state", 16)),
        mamba_d_conv=int(saved_args.get("mamba_d_conv", 4)),
        mamba_expand=int(saved_args.get("mamba_expand", 2)),
        mamba_fast_path=bool(saved_args.get("mamba_fast_path", False)),
        use_spectral_refiner=bool(saved_args.get("spectral_refiner", True)),
        spectral_dim=int(saved_args.get("spectral_dim", 32)),
        spectral_pool_size=int(saved_args.get("spectral_pool_size", 4)),
        use_foundation_prior=bool(saved_args.get("foundation_prior", True)),
        use_evidence_router=bool(saved_args.get("evidence_router", True)),
        use_spatial_mamba=bool(saved_args.get("spatial_mamba", True)),
        router_temperature=float(saved_args.get("router_temperature", 1.0)),
        clear_residual_floor=float(
            saved_args.get("clear_residual_floor", 0.05)
        ),
        fusion_mamba_scales=tuple(
            int(value)
            for value in saved_args.get("fusion_mamba_scales", [2, 3, 4])
        ),
        decoder_mamba_scales=tuple(
            int(value)
            for value in saved_args.get("decoder_mamba_scales", [2, 3])
        ),
    )
    foundation = FrozenTerraMindEncoder(
        model_name=str(saved_args.get("foundation_name", "terramind_v1_base")),
        checkpoint_path=args.foundation_checkpoint,
        expected_dim=config.foundation_dim,
    )
    model = TerraMambaCR(foundation, config).to(device)
    incompatible = model.load_state_dict(state["model_state_dict"], strict=False)
    illegal_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith("foundation_encoder.")
    ]
    if illegal_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"模型权重不匹配：missing={illegal_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()

    layout = DatasetLayout(
        cloud_dir=str(saved_args.get("cloud_dir", "s2_cloudy")),
        sar_dir=str(saved_args.get("sar_dir", "s1")),
        target_dir=str(saved_args.get("target_dir", "s2")),
        cloud_template=str(
            saved_args.get("cloud_template", "{roi}_s2_cloudy_{patch}.tif")
        ),
        sar_template=str(
            saved_args.get("sar_template", "{roi}_s1_SAR_{patch}.tif")
        ),
        target_template=str(
            saved_args.get("target_template", "{roi}_s2_Clear_{patch}.tif")
        ),
    )
    image_size = int(saved_args.get("image_size", 256))
    dataset = S1S2CloudRemovalDataset(
        args.root_path,
        args.test_list,
        layout=layout,
        expected_size=(image_size, image_size),
        optical_scale=float(saved_args.get("optical_scale", 10000.0)),
        sar_value_scale=float(saved_args.get("sar_value_scale", 1.0)),
        sar_value_offset=float(saved_args.get("sar_value_offset", 0.0)),
        augment=False,
        list_format=str(saved_args.get("list_format", "cloud_name")),
        return_mask=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    with torch.no_grad():
        for batch in loader:
            cloud = batch["cloud"].to(device, torch.float32, non_blocking=True)
            sar = batch["sar"].to(device, torch.float32, non_blocking=True)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if args.amp_dtype == "bf16"
                else torch.autocast(device_type="cuda", enabled=False)
            )
            with context:
                prediction = model(cloud, sar)["prediction"]
            prediction = prediction.float().clamp(0.0, 1.0).cpu().numpy()
            for index in range(prediction.shape[0]):
                output_path = output_dir / f"{batch['name'][index]}_clear.tif"
                save_geotiff(
                    output_path,
                    batch["cloud_path"][index],
                    prediction[index],
                )
                completed += 1
                print(f"saved {completed}/{len(dataset)}: {output_path}", flush=True)


if __name__ == "__main__":
    main()
