"""Training, validation and visualization without cloud-mask supervision."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


EVALUATION_METRIC_NAMES = (
    "mae",
    "mse",
    "psnr",
    "ssim",
    "cc",
    "sam",
    "ndvi_mae",
    "ndwi_mae",
    "edge_mae",
    "structure_ssim",
    "optical_confidence",
    "sar_confidence",
    "prior_confidence",
    "reconstruction_confidence",
)


def autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def show_progress(prefix: str, index: int, total: int, interval: int = 25):
    total = max(1, total)
    completed = index + 1
    percent = int(100 * completed / total)
    previous = int(100 * index / total)
    if index == 0 or percent // interval > previous // interval or completed == total:
        print(f"{prefix}: {percent:3d}% ({completed}/{total})", flush=True)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "cloud": batch["cloud"].to(device, torch.float32, non_blocking=True),
        "sar": batch["sar"].to(device, torch.float32, non_blocking=True),
        "target": batch["target"].to(device, torch.float32, non_blocking=True),
        "name": batch["name"],
    }


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device: torch.device,
    scaler,
    amp_dtype: str,
    accumulation_steps: int,
    gradient_clip: float,
    show_log: bool,
) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    component_sums: dict[str, float] = {}
    sample_count = 0
    remainder = len(data_loader) % accumulation_steps
    final_group_start = len(data_loader) - (remainder or accumulation_steps)

    for batch_index, raw_batch in enumerate(data_loader):
        batch = move_batch(raw_batch, device)
        current_accumulation = (
            remainder
            if remainder and batch_index >= final_group_start
            else accumulation_steps
        )
        should_step = (
            (batch_index + 1) % accumulation_steps == 0
            or batch_index + 1 == len(data_loader)
        )
        sync_context = nullcontext()
        if not should_step and hasattr(model, "no_sync"):
            sync_context = model.no_sync()

        with sync_context:
            with autocast_context(device, amp_dtype):
                outputs = model(batch["cloud"], batch["sar"])
                loss, components = criterion(outputs, batch["target"])
                backward_loss = loss / current_accumulation
            if scaler is not None and scaler.is_enabled():
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()

        if should_step:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = batch["target"].shape[0]
        sample_count += batch_size
        for name, value in components.items():
            component_sums[name] = component_sums.get(name, 0.0) + (
                float(value.item()) * batch_size
            )
        if show_log:
            show_progress("train progress", batch_index, len(data_loader))

    if sample_count == 0:
        raise RuntimeError("训练DataLoader没有产生batch")
    return component_sums, sample_count


def _spectral_angle_deg(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    numerator = np.sum(prediction * target, axis=0)
    denominator = np.linalg.norm(prediction, axis=0) * np.linalg.norm(
        target, axis=0
    )
    valid = denominator > 1e-12
    if not np.any(valid):
        return 0.0
    cosine = np.clip(numerator[valid] / denominator[valid], -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)).mean())


def _correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    if np.std(prediction) < 1e-12 or np.std(target) < 1e-12:
        return float(np.allclose(prediction, target))
    return float(np.corrcoef(prediction, target)[0, 1])


def _normalized_difference(image: np.ndarray, a: int, b: int) -> np.ndarray:
    return (image[a] - image[b]) / (image[a] + image[b] + 1e-6)


def _edge_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    indices = [1, 2, 3, 7]
    pred = prediction[indices]
    truth = target[indices]
    horizontal = np.abs(
        np.diff(pred, axis=-1) - np.diff(truth, axis=-1)
    ).mean()
    vertical = np.abs(
        np.diff(pred, axis=-2) - np.diff(truth, axis=-2)
    ).mean()
    return float(0.5 * (horizontal + vertical))


def _rgb_fixed(image: np.ndarray, upper: float = 0.30) -> np.ndarray:
    # Fixed reflectance stretch makes prediction/target brightness comparable.
    rgb = np.moveaxis(image[[3, 2, 1]], 0, -1)
    return np.clip(rgb / upper, 0.0, 1.0)


def _sar_stretch(sar: np.ndarray) -> np.ndarray:
    vv = sar[0].astype(np.float32)
    return np.clip((vv + 25.0) / 25.0, 0.0, 1.0)


def save_visualization(
    path: str | Path,
    cloud: np.ndarray,
    sar: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    evidence: np.ndarray | None = None,
    reconstruction_confidence: np.ndarray | None = None,
):
    if evidence is None or reconstruction_confidence is None:
        figure, axes = plt.subplots(1, 4, figsize=(17, 4.5))
        axes = np.asarray(axes).reshape(1, 4)
    else:
        figure, axes = plt.subplots(2, 4, figsize=(17, 9.0))
    axes[0, 0].imshow(_rgb_fixed(cloud))
    axes[0, 0].set_title("Cloudy S2")
    axes[0, 1].imshow(_sar_stretch(sar), cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("SAR VV")
    axes[0, 2].imshow(_rgb_fixed(prediction))
    axes[0, 2].set_title("Prediction")
    axes[0, 3].imshow(_rgb_fixed(target))
    axes[0, 3].set_title("Target")
    if evidence is not None and reconstruction_confidence is not None:
        titles = ("Optical evidence", "SAR evidence", "Foundation evidence")
        for index, title in enumerate(titles):
            axes[1, index].imshow(
                evidence[index], cmap="magma", vmin=0.0, vmax=1.0
            )
            axes[1, index].set_title(title)
        axes[1, 3].imshow(
            reconstruction_confidence.squeeze(),
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        axes[1, 3].set_title("Reconstruction confidence")
    for axis in axes.reshape(-1):
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


@torch.inference_mode()
def evaluate(
    model,
    data_loader,
    device: torch.device,
    amp_dtype: str,
    figure_root: str | Path,
    epoch: int,
    max_visualizations: int = 3,
    distributed: bool = False,
    rank: int = 0,
) -> dict[str, float]:
    model.eval()
    if distributed and not dist.is_initialized():
        raise RuntimeError("distributed=True但分布式进程组尚未初始化")
    sums = {name: 0.0 for name in EVALUATION_METRIC_NAMES}
    sample_count = 0
    saved = 0
    save_figures = rank == 0 and max_visualizations > 0
    figure_dir = Path(figure_root) / f"epoch_{epoch + 1:04d}"
    if save_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)

    for batch_index, raw_batch in enumerate(data_loader):
        batch = move_batch(raw_batch, device)
        with autocast_context(device, amp_dtype):
            outputs = model(batch["cloud"], batch["sar"])
        prediction = outputs["prediction"].float().clamp(0, 1).cpu().numpy()
        target = batch["target"].cpu().numpy()
        need_batch_visuals = save_figures and saved < max_visualizations
        cloud = batch["cloud"].cpu().numpy() if need_batch_visuals else None
        sar = batch["sar"].cpu().numpy() if need_batch_visuals else None

        evidence_tensor = outputs.get("evidence_weights")
        evidence_means = None
        evidence_visuals = None
        if evidence_tensor is not None:
            evidence_tensor = evidence_tensor[0].float()
            evidence_means = (
                evidence_tensor.mean(dim=(-2, -1)).cpu().numpy()
            )
        if evidence_tensor is not None and need_batch_visuals:
            evidence_visuals = torch.nn.functional.interpolate(
                evidence_tensor,
                size=prediction.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).cpu().numpy()

        reconstruction_tensor = outputs.get("reconstruction_confidence")
        reconstruction_means = None
        reconstruction_visuals = None
        if reconstruction_tensor is not None:
            reconstruction_tensor = reconstruction_tensor.float()
            reconstruction_means = (
                reconstruction_tensor.mean(dim=(-3, -2, -1)).cpu().numpy()
            )
        if reconstruction_tensor is not None and need_batch_visuals:
            reconstruction_visuals = torch.nn.functional.interpolate(
                reconstruction_tensor,
                size=prediction.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).cpu().numpy()

        for index in range(prediction.shape[0]):
            pred = prediction[index].astype(np.float64)
            truth = target[index].astype(np.float64)
            absolute = np.abs(pred - truth)
            sums["mae"] += float(absolute.mean())
            mse = float(np.square(pred - truth).mean())
            sums["mse"] += mse
            sums["psnr"] += float(
                peak_signal_noise_ratio(truth, pred, data_range=1)
            )
            sums["ssim"] += float(
                structural_similarity(truth, pred, data_range=1, channel_axis=0)
            )
            sums["structure_ssim"] += float(
                structural_similarity(
                    truth[[1, 2, 3, 7]],
                    pred[[1, 2, 3, 7]],
                    data_range=1,
                    channel_axis=0,
                )
            )
            sums["cc"] += _correlation(pred, truth)
            sums["sam"] += _spectral_angle_deg(pred, truth)
            sums["ndvi_mae"] += float(
                np.abs(
                    _normalized_difference(pred, 7, 3)
                    - _normalized_difference(truth, 7, 3)
                ).mean()
            )
            sums["ndwi_mae"] += float(
                np.abs(
                    _normalized_difference(pred, 2, 7)
                    - _normalized_difference(truth, 2, 7)
                ).mean()
            )
            sums["edge_mae"] += _edge_mae(pred, truth)
            if evidence_means is not None:
                sums["optical_confidence"] += float(evidence_means[index, 0])
                sums["sar_confidence"] += float(evidence_means[index, 1])
                sums["prior_confidence"] += float(evidence_means[index, 2])
            if reconstruction_means is not None:
                sums["reconstruction_confidence"] += float(
                    reconstruction_means[index]
                )
            if save_figures and saved < max_visualizations:
                save_visualization(
                    figure_dir / f"sample_{saved:02d}.png",
                    cloud[index],
                    sar[index],
                    pred,
                    truth,
                    (
                        None
                        if evidence_visuals is None
                        else evidence_visuals[index]
                    ),
                    None
                    if reconstruction_visuals is None
                    else reconstruction_visuals[index],
                )
                saved += 1
            sample_count += 1
        if rank == 0:
            show_progress(
                "validation rank0 progress",
                batch_index,
                len(data_loader),
            )

    packed = torch.tensor(
        [sums[name] for name in EVALUATION_METRIC_NAMES]
        + [float(sample_count)],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    global_count = packed[-1].item()
    if global_count <= 0:
        raise RuntimeError("全局验证DataLoader没有产生样本")
    metrics = {
        name: packed[index].item() / global_count
        for index, name in enumerate(EVALUATION_METRIC_NAMES)
    }
    if rank == 0:
        print(
            {
                "epoch": epoch + 1,
                "samples": int(global_count),
                **metrics,
            },
            flush=True,
        )
    return metrics
