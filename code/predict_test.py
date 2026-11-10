"""PRISMamba-CR: distributed test metrics, GeoTIFF, RGB and evidence figures.

Copy beside train.py in the supplied prismamba_cr_complete project. Each rank
owns an independent model and disjoint sample indices; no DDP forward calls.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from distributed_utils import DistributedEvalSampler
from engine_simple import (
    EVALUATION_METRIC_NAMES, _correlation, _edge_mae, _normalized_difference,
    _spectral_angle_deg, autocast_context, save_visualization,
)
from predict import load_checkpoint


def _ssim_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Exact requested local-window SSIM, including zero-padded boundaries.
    channels = pred.shape[1]
    coords = torch.arange(11, device=pred.device, dtype=pred.dtype) - 5
    gaussian = torch.exp(-coords.square() / (2 * 1.5 ** 2))
    gaussian /= gaussian.sum()
    window = (gaussian[:, None] * gaussian[None, :]).view(1, 1, 11, 11)
    window = window.expand(channels, 1, 11, 11)
    def filt(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, window, padding=5, groups=channels)
    mx, my = filt(pred), filt(target)
    vx, vy = filt(pred.square()) - mx.square(), filt(target.square()) - my.square()
    covariance = filt(pred * target) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mx * my + c1) * (2 * covariance + c2) /
             ((mx.square() + my.square() + c1) * (vx + vy + c2)))
    return score.mean(dim=(1, 2, 3))


def resolve_checkpoint(history_root: Path, run_time: str,
                       explicit: str | None = None) -> tuple[Path, Path]:
    if not run_time or run_time in {".", ".."} or any(c in run_time for c in "/\\"):
        raise ValueError("--run-time必须为训练目录名或其时间前缀，不能含路径分隔符")
    if explicit:
        checkpoint = Path(explicit).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"检查点不存在：{checkpoint}")
        return checkpoint, checkpoint.parent.parent
    root = history_root.expanduser().resolve()
    exact = root / run_time
    if exact.is_dir():
        candidates = [exact]
    else:
        candidates = sorted(p for p in root.iterdir()
                            if p.is_dir() and p.name.startswith(run_time + "-"))
    if not candidates:
        raise FileNotFoundError(f"{root}中未找到训练目录：{run_time}")
    if len(candidates) != 1:
        raise ValueError("训练时间前缀匹配多个目录，请输入完整名称：\n" +
                         "\n".join(str(p) for p in candidates))
    checkpoint = candidates[0] / "model" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"最佳模型不存在：{checkpoint}")
    return checkpoint, candidates[0]


def foundation_path(options, state, saved) -> Path:
    if options.foundation_checkpoint:
        path = Path(options.foundation_checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"指定的TerraMind权重不存在：{path}")
        return path
    candidates = [state.get("foundation_checkpoint"),
                  saved.get("foundation_checkpoint"),
                  os.getenv("FOUNDATION_CHECKPOINT"),
                  "/home/zhangchengyao/source_packages/terramind_weights/TerraMind_v1_base.pt"]
    for value in candidates:
        if value and Path(value).expanduser().is_file():
            return Path(value).expanduser().resolve()
    raise FileNotFoundError("未找到冻结TerraMind权重，请传入 --foundation-checkpoint 路径")


def build_saved_model(options, checkpoint, device):
    # Import GPU-dependent packages only after setup has selected the device.
    from frozen_terramind import FrozenTerraMindEncoder
    from model_terramamba_cr import TerraMambaCR, TerraMambaConfig
    state = load_checkpoint(checkpoint)
    if "model_state_dict" not in state or not isinstance(state.get("arguments"), dict):
        raise ValueError("检查点必须包含训练脚本保存的model_state_dict和arguments")
    saved = dict(state["arguments"])
    foundation = foundation_path(options, state, saved)
    weights = state["model_state_dict"]
    if weights and all(key.startswith("module.") for key in weights):
        weights = {key[7:]: value for key, value in weights.items()}
    metadata = {"epoch": state.get("epoch"), "training_metrics": state.get("metrics", {}),
                "foundation_checkpoint": str(foundation)}
    del state  # Release optimizer state before loading the large foundation model.
    aliases = {"use_spectral_refiner": "spectral_refiner",
               "use_foundation_prior": "foundation_prior",
               "use_evidence_router": "evidence_router",
               "use_spatial_mamba": "spatial_mamba"}
    config_values = {}
    for key in TerraMambaConfig.__dataclass_fields__:
        source = aliases.get(key, key)
        if source in saved:
            config_values[key] = saved[source]
    for key in ("fusion_mamba_scales", "decoder_mamba_scales"):
        if key in config_values:
            config_values[key] = tuple(config_values[key])
    config = TerraMambaConfig(**config_values)
    encoder = FrozenTerraMindEncoder(
        model_name=saved.get("foundation_name", "terramind_v1_base"),
        checkpoint_path=foundation, expected_dim=config.foundation_dim,
    )
    model = TerraMambaCR(encoder, config)
    incompatible = model.load_state_dict(weights, strict=False)
    illegal = [key for key in incompatible.missing_keys
               if not key.startswith("foundation_encoder.")]
    if illegal or incompatible.unexpected_keys:
        raise RuntimeError(f"模型权重不匹配：missing={illegal}, unexpected={incompatible.unexpected_keys}")
    del weights
    return model.to(device).eval(), saved, metadata


class IndexedTestDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.names = []
        for folder, cloud_name, _, _ in dataset.samples:
            clean_folder = folder.replace("/", "_").replace("\\", "_")
            self.names.append(f"{Path(cloud_name).stem}_{clean_folder}")
        if len(self.names) != len(set(self.names)):
            raise ValueError("测试清单中存在重复输出名称，无法确保每张图像独立保存")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["test_index"] = index
        sample["output_name"] = self.names[index]
        sample["cloud_fraction"] = self.dataset.samples[index][0]
        return sample


def make_test_dataset(options, saved):
    from dataset import DatasetLayout, S1S2CloudRemovalDataset
    layout = DatasetLayout(**{
        key: saved[key] for key in ("cloud_dir", "sar_dir", "target_dir", "cloud_template",
                                    "sar_template", "target_template") if key in saved
    })
    size = int(saved.get("image_size", 256))
    dataset = S1S2CloudRemovalDataset(
        options.root_path or saved.get("root_path", "../../data"), options.test_list,
        layout=layout, expected_size=(size, size),
        optical_scale=float(saved.get("optical_scale", 10000.0)),
        sar_value_scale=float(saved.get("sar_value_scale", 1.0)),
        sar_value_offset=float(saved.get("sar_value_offset", 0.0)),
        list_format=saved.get("list_format", "cloud_name"), augment=False, return_mask=False,
    )
    return IndexedTestDataset(dataset)


def percentile_rgb(prediction, low=2.0, high=98.0):
    rgb = np.moveaxis(prediction[[3, 2, 1]], 0, -1).astype(np.float32)
    limits = np.percentile(rgb, [low, high], axis=(0, 1))
    stretched = (rgb - limits[0]) / np.maximum(limits[1] - limits[0], 1e-6)
    return np.round(np.clip(stretched, 0, 1) * 255).astype(np.uint8)


def save_prediction_tif(path, source_path, image, dtype="float32", optical_scale=10000.0):
    from osgeo import gdal
    source = gdal.Open(str(source_path), gdal.GA_ReadOnly)
    if source is None:
        raise FileNotFoundError(f"无法打开源影像：{source_path}")
    height, width = image.shape[-2:]
    temporary = path.with_name(path.stem + ".tmp.tif")
    output = None
    try:
        output = gdal.GetDriverByName("GTiff").Create(
            str(temporary), width, height, image.shape[0],
            gdal.GDT_Float32 if dtype == "float32" else gdal.GDT_UInt16,
            options=["COMPRESS=DEFLATE", "TILED=YES"],
        )
        if output is None:
            raise RuntimeError(f"无法创建GeoTIFF：{temporary}")
        output.SetGeoTransform(source.GetGeoTransform())
        output.SetProjection(source.GetProjection())
        data = image.astype(np.float32) if dtype == "float32" else np.round(
            np.clip(image, 0, 1) * optical_scale).astype(np.uint16)
        for index in range(image.shape[0]):
            band = output.GetRasterBand(index + 1)
            band.WriteArray(data[index])
            if hasattr(band, "SetDescription"):
                band.SetDescription(("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
                                     "B8A", "B9", "B10", "B11", "B12")[index])
            if hasattr(band, "SetScale"):
                band.SetScale(1.0 if dtype == "float32" else 1.0 / optical_scale)
                band.SetOffset(0.0)
        output.FlushCache()
        band = None
        output = None
        os.replace(temporary, path)
    finally:
        output = None
        source = None


def image_metrics(prediction, target, ssim, evidence_mean, reconstruction_mean):
    from skimage.metrics import structural_similarity
    pred, truth = prediction.astype(np.float64), target.astype(np.float64)
    mse = float(np.square(pred - truth).mean())
    return {
        "mae": float(np.abs(pred - truth).mean()), "mse": mse,
        "psnr": -10.0 * math.log10(mse) if mse > 0 else float("inf"), "ssim": float(ssim),
        "cc": _correlation(pred, truth), "sam": _spectral_angle_deg(pred, truth),
        "ndvi_mae": float(np.abs(_normalized_difference(pred, 7, 3) -
                                  _normalized_difference(truth, 7, 3)).mean()),
        "ndwi_mae": float(np.abs(_normalized_difference(pred, 2, 7) -
                                  _normalized_difference(truth, 2, 7)).mean()),
        "edge_mae": _edge_mae(pred, truth),
        "structure_ssim": float(structural_similarity(
            truth[[1, 2, 3, 7]], pred[[1, 2, 3, 7]], data_range=1, channel_axis=0)),
        "optical_confidence": float(evidence_mean[0]),
        "sar_confidence": float(evidence_mean[1]),
        "prior_confidence": float(evidence_mean[2]),
        "reconstruction_confidence": float(reconstruction_mean),
    }


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value == math.inf else "-Infinity" if value == -math.inf else None
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def merge_reports(output, dataset_size, world_size, metrics, metadata):
    records = []
    for rank in range(world_size):
        with (output / f".rank_{rank:04d}.csv").open(encoding="utf-8", newline="") as stream:
            records.extend(csv.DictReader(stream))
    records.sort(key=lambda row: int(row["test_index"]))
    if [int(row["test_index"]) for row in records] != list(range(dataset_size)):
        raise RuntimeError("测试分片存在重复或缺失样本，拒绝输出最终指标")
    fields = ["test_index", "file_name", "cloud_fraction", *EVALUATION_METRIC_NAMES]
    with (output / "per_image_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader();writer.writerows(records)
    groups = {}
    for row in records:
        groups.setdefault(row["cloud_fraction"], []).append(row)
    by_cloud = {}
    for cloud, rows in sorted(groups.items()):
        by_cloud[cloud] = {"samples": len(rows), **{
            name: sum(float(row[name]) for row in rows) / len(rows) for name in EVALUATION_METRIC_NAMES
        }}
    with (output / "metrics_by_cloud.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["cloud_fraction", "samples", *EVALUATION_METRIC_NAMES])
        writer.writeheader()
        for cloud, values in by_cloud.items():writer.writerow({"cloud_fraction": cloud, **values})
    report = {**metadata, "status": "completed", "samples": dataset_size,
              "metrics": metrics, "by_cloud_fraction": by_cloud,
              "aggregation": "arithmetic mean of per-image metrics; all 13 bands",
              "ssim": "Gaussian 11x11 sigma=1.5, zero padding=5, C1=.01^2 C2=.03^2; float32",
              "structure_ssim": "original validation skimage SSIM on B2/B3/B4/B8",
              "sam_unit": "degrees"}
    write_json(output / "test_metrics.json", report)
    lines = [f"Test samples: {dataset_size}", f"Checkpoint: {metadata['checkpoint']}",
             "SSIM: requested 13-band Gaussian window implementation", "SAM unit: degrees"]
    lines.extend(f"{name}: {metrics[name]:.8f}" for name in EVALUATION_METRIC_NAMES)
    (output / "test_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "run_status.json", {"status": "completed", "samples": dataset_size})
    print("\n".join(lines), flush=True)
    print(f"全部结果已保存：{output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--history-root", default="./history")
    parser.add_argument("--checkpoint", default=None, help="Optional exact checkpoint path")
    parser.add_argument("--root-path", default=os.getenv("DATA_ROOT"))
    parser.add_argument("--test-list", default="data_txt/test.txt")
    parser.add_argument("--foundation-checkpoint", default=None)
    parser.add_argument("--result-root", default="./result")
    parser.add_argument("--batch-size", type=int, default=64, help="Per DCU")
    parser.add_argument("--num-workers", type=int, default=2, help="Per process")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "none"), default=None)
    parser.add_argument("--tif-dtype", choices=("float32", "uint16"), default="float32")
    parser.add_argument("--rgb-percentiles", nargs=2, type=float, default=[2., 98.])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--distributed-timeout-minutes", type=int, default=120)
    parser.add_argument("--log-interval", type=int, default=25)
    options = parser.parse_args()
    if options.batch_size < 1 or options.num_workers < 0 or options.log_interval < 1:
        parser.error("batch-size/log-interval须为正数，num-workers不能为负数")
    if not 0 <= options.rgb_percentiles[0] < options.rgb_percentiles[1] <= 100:
        parser.error("RGB百分位须满足0<=low<high<=100")
    if options.distributed_timeout_minutes < 1:
        parser.error("分布式超时必须为正数")
    return options


def main():
    options = parse_args()
    rank, world_size = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    distributed = world_size > 1
    device = torch.device("cuda", local_rank) if options.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise RuntimeError("可见DCU不足，请检查VISIBLE_DCUS和设备掩膜")
        torch.cuda.set_device(local_rank);torch.cuda.init()
    elif distributed and options.backend != "gloo":
        raise ValueError("CPU分布式检查需要 --backend gloo")
    if distributed:
        dist.init_process_group(options.backend, timeout=timedelta(minutes=options.distributed_timeout_minutes))
    lock = None
    lock_acquired = False
    output = None
    started = time.monotonic()
    try:
        checkpoint, training_dir = resolve_checkpoint(Path(options.history_root), options.run_time, options.checkpoint)
        output = Path(options.result_root).expanduser().resolve() / options.run_time
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            lock = (output / ".predict.lock").open("a")
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_acquired = True
            except BlockingIOError:raise RuntimeError(f"该结果目录已有推理任务：{output}")
            for folder in ("test_tif", "test_rgb", "test_duibi"):(output / folder).mkdir(exist_ok=True)
            write_json(output / "run_status.json", {"status": "running", "checkpoint": str(checkpoint)})
        if distributed:dist.barrier()
        # Serialized model construction avoids simultaneous large CPU loads.
        model = saved = model_metadata = None
        for owner in range(world_size):
            if rank == owner:
                print(f"rank {rank}/{world_size}: 加载 {checkpoint}", flush=True)
                model, saved, model_metadata = build_saved_model(options, checkpoint, device)
            if distributed:dist.barrier()
        amp_dtype = options.amp_dtype or (saved.get("amp_dtype", "bf16") if device.type == "cuda" else "none")
        if device.type == "cpu" and amp_dtype != "none":raise ValueError("CPU检查须使用 --amp-dtype none")
        dataset = make_test_dataset(options, saved)
        sampler = DistributedEvalSampler(dataset, world_size, rank)
        loader_options = {"multiprocessing_context": "spawn"} if options.num_workers else {}
        loader = DataLoader(dataset, batch_size=options.batch_size, sampler=sampler,
                            num_workers=options.num_workers, pin_memory=device.type == "cuda", **loader_options)
        if options.tif_dtype == "uint16" and not 0 < float(saved.get("optical_scale", 10000)) <= 65535:
            raise ValueError("UInt16导出要求optical_scale位于(0,65535]")
        metadata = {"requested_run_time": options.run_time, "training_directory": str(training_dir),
                    "checkpoint": str(checkpoint), **model_metadata,
                    "world_size": world_size, "amp_dtype": amp_dtype, "tif_dtype": options.tif_dtype,
                    "rgb_percentiles": options.rgb_percentiles, "training_arguments": saved,
                    "test_list": str(dataset.dataset.list_path), "data_root": str(dataset.dataset.root),
                    "started_at": datetime.now().isoformat()}
        if rank == 0:write_json(output / "inference_config.json", metadata)
        totals = {name: 0.0 for name in EVALUATION_METRIC_NAMES};count = 0
        fields = ["test_index", "file_name", "cloud_fraction", *EVALUATION_METRIC_NAMES]
        with (output / f".rank_{rank:04d}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader()
            with torch.inference_mode():
                for batch in loader:
                    cloud = batch["cloud"].to(device, torch.float32, non_blocking=True)
                    sar = batch["sar"].to(device, torch.float32, non_blocking=True)
                    target = batch["target"].to(device, torch.float32, non_blocking=True)
                    with autocast_context(device, amp_dtype):outputs = model(cloud, sar)
                    raw_prediction = outputs["prediction"].float()
                    if not torch.isfinite(raw_prediction).all():raise FloatingPointError("预测包含NaN或Inf")
                    prediction = raw_prediction.clamp(0, 1)
                    # Both operands are float32; no AMP in SSIM evaluation.
                    with torch.autocast(device_type=device.type, enabled=False):
                        ssim = _ssim_per_image(prediction, target).cpu().numpy()
                    weights = outputs.get("evidence_weights")
                    confidence = outputs.get("reconstruction_confidence")
                    if weights is None or confidence is None:
                        raise RuntimeError("当前模型未返回证据权重或重建置信度，无法保存请求的热力图")
                    evidence = weights[0].float()
                    confidence = confidence.float()
                    if not torch.isfinite(evidence).all() or not torch.isfinite(confidence).all():
                        raise FloatingPointError("热力图包含NaN或Inf")
                    evidence_means = evidence.mean(dim=(-2, -1)).cpu().numpy()
                    confidence_means = confidence.mean(dim=(-3, -2, -1)).cpu().numpy()
                    evidence_maps = F.interpolate(evidence, size=prediction.shape[-2:], mode="bilinear",
                                                  align_corners=False).cpu().numpy()
                    confidence_maps = F.interpolate(confidence, size=prediction.shape[-2:], mode="bilinear",
                                                    align_corners=False).cpu().numpy()
                    pred_np, target_np = prediction.cpu().numpy(), target.cpu().numpy()
                    cloud_np, sar_np = batch["cloud"].numpy(), batch["sar"].numpy()
                    for i in range(pred_np.shape[0]):
                        name = batch["output_name"][i]
                        scores = image_metrics(pred_np[i], target_np[i], ssim[i],
                                               evidence_means[i], confidence_means[i])
                        save_prediction_tif(output / "test_tif" / f"{name}_pred.tif",
                                            batch["cloud_path"][i], pred_np[i], options.tif_dtype,
                                            float(saved.get("optical_scale", 10000)))
                        Image.fromarray(percentile_rgb(pred_np[i], *options.rgb_percentiles)).save(
                            output / "test_rgb" / f"{name}_pred_rgb.png")
                        save_visualization(output / "test_duibi" / f"{name}_duibi.png",
                                           cloud_np[i], sar_np[i], pred_np[i], target_np[i],
                                           evidence_maps[i], confidence_maps[i])
                        writer.writerow({"test_index": int(batch["test_index"][i]), "file_name": name,
                                         "cloud_fraction": batch["cloud_fraction"][i], **scores})
                        for key in totals:totals[key] += scores[key]
                        count += 1
                        if count == 1 or count % options.log_interval == 0 or count == len(sampler):
                            print(f"rank {rank}: {count}/{len(sampler)}，{name}", flush=True)
                    stream.flush()
                    # Release the previous batch before the next model forward.
                    del outputs, weights, confidence, evidence, raw_prediction, prediction
                    del cloud, sar, target
        packed = torch.tensor([totals[key] for key in EVALUATION_METRIC_NAMES] + [count],
                              dtype=torch.float64, device=device)
        if distributed:dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        samples = int(packed[-1].item())
        if samples != len(dataset):raise RuntimeError(f"全局样本数{samples}与测试清单{len(dataset)}不符")
        metrics = {key: float(packed[i].item() / samples) for i, key in enumerate(EVALUATION_METRIC_NAMES)}
        if rank == 0:
            metadata["elapsed_seconds_including_exports"] = time.monotonic() - started
            merge_reports(output, samples, world_size, metrics, metadata)
        if distributed:dist.barrier()
    except Exception as error:
        if rank == 0 and output is not None and lock_acquired:
            write_json(output / "run_status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        if lock is not None:lock.close()
        if distributed and dist.is_initialized():dist.destroy_process_group()


if __name__ == "__main__":
    main()
