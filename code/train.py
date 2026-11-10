"""Single-node multi-DCU training for PRISMamba-CR."""

from __future__ import annotations

import argparse
import csv
from datetime import timedelta
import math
import os
import pickle
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from dataset import DatasetLayout, S1S2CloudRemovalDataset
from distributed_utils import DistributedEvalSampler
from engine_simple import (
    EVALUATION_METRIC_NAMES,
    evaluate,
    train_one_epoch,
)
from frozen_terramind import FrozenTerraMindEncoder
from losses_simple import L1SAMSSIMLoss
from model_terramamba_cr import TerraMambaCR, TerraMambaConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prior-conditioned frozen TerraMind + spatial-spectral Mamba "
            "SAR-assisted cloud removal"
        )
    )
    parser.add_argument("--root-path", default="../../data")
    parser.add_argument("--train-list", default="data_txt/train.txt")
    parser.add_argument("--val-list", default="data_txt/val.txt")
    parser.add_argument(
        "--list-format",
        choices=("cloud_name", "legacy", "auto"),
        default="cloud_name",
        help=(
            "cloud_name读取'文件夹 有云图像主文件名'两列格式；"
            "legacy读取旧三列格式；auto自动兼容"
        ),
    )
    parser.add_argument("--cloud-dir", default="s2_cloudy")
    parser.add_argument("--sar-dir", default="s1")
    parser.add_argument("--target-dir", default="s2")
    parser.add_argument(
        "--cloud-template", default="{roi}_s2_cloudy_{patch}.tif"
    )
    parser.add_argument(
        "--sar-template", default="{roi}_s1_SAR_{patch}.tif"
    )
    parser.add_argument(
        "--target-template", default="{roi}_s2_Clear_{patch}.tif"
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--optical-scale", type=float, default=10000.0)
    parser.add_argument(
        "--sar-value-scale",
        type=float,
        default=1.0,
        help="读入后SAR=raw*scale+offset；默认假定TIFF已经是dB",
    )
    parser.add_argument("--sar-value-offset", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--val-num-workers",
        type=int,
        default=2,
        help="每个验证进程的读取worker数；7卡验证建议1至2",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=1,
        help="每隔多少个epoch验证一次；首轮和最后一轮始终验证",
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=4,
        help="每次验证仅由rank 0保存的最大样本图数量",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--amp-dtype", choices=("bf16", "fp16", "none"), default="bf16"
    )

    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--sar-channel-ratio", type=float, default=0.5)
    parser.add_argument("--foundation-dim", type=int, default=768)
    parser.add_argument("--foundation-name", default="terramind_v1_base")
    parser.add_argument("--foundation-checkpoint", required=True)
    parser.add_argument(
        "--mamba-backend",
        choices=("mamba_ssm", "portable"),
        default="mamba_ssm",
    )
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument(
        "--mamba-fast-path",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="DCU先用--no-mamba-fast-path；确认融合核可用后再开启",
    )
    parser.add_argument("--spectral-dim", type=int, default=32)
    parser.add_argument("--spectral-pool-size", type=int, default=4)
    parser.add_argument(
        "--spectral-refiner",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--foundation-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="消融时使用--no-foundation-prior",
    )
    parser.add_argument(
        "--evidence-router",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="消融时使用--no-evidence-router改为三类证据等权融合",
    )
    parser.add_argument(
        "--spatial-mamba",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="消融时使用--no-spatial-mamba改为卷积残差块",
    )
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--clear-residual-floor", type=float, default=0.05)
    parser.add_argument(
        "--fusion-mamba-scales",
        nargs="+",
        type=int,
        default=[2, 3, 4],
        help="0至4分别表示256、128、64、32、16尺度",
    )
    parser.add_argument(
        "--decoder-mamba-scales",
        nargs="+",
        type=int,
        default=[2, 3],
    )
    parser.add_argument("--sam-weight", type=float, default=0.10)
    parser.add_argument("--ssim-weight", type=float, default=0.10)
    parser.add_argument("--ssim-scales", type=int, default=3)
    parser.add_argument("--ssim-window-size", type=int, default=11)
    parser.add_argument("--ssim-sigma", type=float, default=1.5)

    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--start-epoch", type=int, default=None,
        help="续训的首轮显示编号（从1计数）；100表示本次第一轮日志为100。默认按检查点继续",
    )
    parser.add_argument(
        "--load-optimizer", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-root", default="history")
    parser.add_argument("--run-name", default="PRISMambaCR")
    parser.add_argument(
        "--backend", default=os.environ.get("DIST_BACKEND", "nccl")
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
    )
    return parser.parse_args()


def setup_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    distributed = world_size > 1
    local_rank = args.local_rank
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun已启动，但PyTorch没有发现DCU")
        visible = torch.cuda.device_count()
        if local_rank >= visible:
            raise RuntimeError(f"LOCAL_RANK={local_rank}，仅发现{visible}张DCU")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=args.backend,
            init_method="env://",
            timeout=timedelta(hours=3),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    if torch.cuda.is_available():
        device_index = local_rank if distributed else 0
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
    else:
        device = torch.device("cpu")
    return distributed, rank, world_size, local_rank, device


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def create_grad_scaler(amp_dtype: str):
    enabled = amp_dtype == "fp16" and torch.cuda.is_available()
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_layout(args) -> DatasetLayout:
    return DatasetLayout(
        cloud_dir=args.cloud_dir,
        sar_dir=args.sar_dir,
        target_dir=args.target_dir,
        cloud_template=args.cloud_template,
        sar_template=args.sar_template,
        target_template=args.target_template,
    )


def make_dataset(args, list_path: str, augment: bool):
    return S1S2CloudRemovalDataset(
        args.root_path,
        list_path,
        layout=make_layout(args),
        expected_size=(args.image_size, args.image_size),
        optical_scale=args.optical_scale,
        sar_value_scale=args.sar_value_scale,
        sar_value_offset=args.sar_value_offset,
        augment=augment,
        list_format=args.list_format,
        return_mask=False,
    )


def make_model(args) -> TerraMambaCR:
    foundation = FrozenTerraMindEncoder(
        model_name=args.foundation_name,
        checkpoint_path=args.foundation_checkpoint,
        expected_dim=args.foundation_dim,
    )
    config = TerraMambaConfig(
        base_channels=args.base_channels,
        sar_channel_ratio=args.sar_channel_ratio,
        foundation_dim=args.foundation_dim,
        mamba_backend=args.mamba_backend,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        mamba_fast_path=args.mamba_fast_path,
        use_spectral_refiner=args.spectral_refiner,
        spectral_dim=args.spectral_dim,
        spectral_pool_size=args.spectral_pool_size,
        use_foundation_prior=args.foundation_prior,
        use_evidence_router=args.evidence_router,
        use_spatial_mamba=args.spatial_mamba,
        router_temperature=args.router_temperature,
        clear_residual_floor=args.clear_residual_floor,
        fusion_mamba_scales=tuple(args.fusion_mamba_scales),
        decoder_mamba_scales=tuple(args.decoder_mamba_scales),
    )
    return TerraMambaCR(foundation, config)


def build_model_serialized(args, distributed, rank, world_size, device):
    """Load the 1.5 GB foundation checkpoint one rank at a time."""
    if not distributed:
        return make_model(args).to(device)

    model = None
    for owner in range(world_size):
        if rank == owner:
            print(f"rank {rank}开始加载冻结TerraMind", flush=True)
            model = make_model(args).to(device)
        dist.barrier()
    if model is None:
        raise RuntimeError(f"rank {rank}未成功构建模型")
    return model


def reduce_components(component_sums, sample_count, device, distributed):
    names = sorted(component_sums)
    values = [component_sums[name] for name in names] + [float(sample_count)]
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    count = tensor[-1].item()
    if count <= 0:
        raise RuntimeError("全局训练样本数为0")
    return {name: tensor[index].item() / count for index, name in enumerate(names)}


def cosine_lambda(epoch, epochs, warmup_epochs, min_ratio):
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return max(min_ratio, float(epoch + 1) / warmup_epochs)
    denominator = max(1, epochs - warmup_epochs)
    progress = min(1.0, max(0.0, (epoch - warmup_epochs) / denominator))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    metrics,
    args,
    training_state,
):
    # TerraMind is immutable and is loaded from its original checkpoint.  Do
    # not duplicate ~1.5 GB of frozen weights in every training checkpoint.
    decoder_state = {
        name: tensor
        for name, tensor in unwrap_model(model).state_dict().items()
        if not name.startswith("foundation_encoder.")
    }
    state = {
        "epoch": epoch,
        "model_state_dict": decoder_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": {name: float(value) for name, value in metrics.items()},
        "training_state": training_state,
        "arguments": vars(args),
        "foundation_checkpoint": str(Path(args.foundation_checkpoint).resolve()),
    }
    if scaler is not None and scaler.is_enabled():
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)


def load_checkpoint(path, model):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Only use this fallback for checkpoints produced by this project.
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    incompatible = model.load_state_dict(state["model_state_dict"], strict=False)
    illegal_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith("foundation_encoder.")
    ]
    if illegal_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "断点与当前模型结构不一致："
            f"missing={illegal_missing}, unexpected={incompatible.unexpected_keys}"
        )
    return state


def append_metrics(path: Path, row: dict[str, float | int]):
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def align_scheduler_epoch(optimizer, scheduler, epoch_index):
    """Position LambdaLR before an explicitly numbered training epoch."""
    rates = [
        base_lr * function(epoch_index)
        for base_lr, function in zip(scheduler.base_lrs, scheduler.lr_lambdas)
    ]
    state = scheduler.state_dict()
    state["last_epoch"] = epoch_index
    state["_step_count"] = epoch_index + 1
    state["_last_lr"] = rates
    scheduler.load_state_dict(state)
    for group, rate in zip(optimizer.param_groups, rates):
        group["lr"] = rate


def main():
    args = parse_args()
    if args.start_epoch is not None:
        if not args.resume:
            raise ValueError("--start-epoch必须与--resume一起使用")
        if not 1 <= args.start_epoch <= args.epochs:
            raise ValueError("start-epoch须在1至epochs之间（显示轮次，从1计数）")
    if args.val_interval <= 0:
        raise ValueError("val-interval必须大于0")
    if args.batch_size <= 0 or args.val_batch_size <= 0:
        raise ValueError("batch-size和val-batch-size必须大于0")
    if args.num_workers < 0 or args.val_num_workers < 0:
        raise ValueError("num-workers和val-num-workers不能小于0")
    if args.max_visualizations < 0:
        raise ValueError("max-visualizations不能小于0")
    distributed, rank, world_size, local_rank, device = setup_distributed(args)
    is_main = rank == 0
    seed_everything(args.seed + rank)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    writer = None

    try:
        timestamp = torch.tensor(
            [int(time.time()) if is_main else 0], dtype=torch.int64, device=device
        )
        if distributed:
            dist.broadcast(timestamp, src=0)
        time_label = time.strftime(
            "%Y-%m-%d-%H-%M-%S", time.localtime(timestamp.item())
        )
        run_root = Path(args.output_root) / f"{args.run_name}_{time_label}"
        model_dir = run_root / "model"
        figure_dir = run_root / "validation_figures"
        metrics_path = run_root / "metrics.csv"
        if is_main:
            model_dir.mkdir(parents=True, exist_ok=False)
            figure_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(run_root / "tensorboard")
        if distributed:
            dist.barrier()

        train_dataset = make_dataset(args, args.train_list, augment=True)
        validation_dataset = make_dataset(
            args,
            args.val_list,
            augment=False,
        )
        train_sampler = None
        validation_sampler = None
        if distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )
            validation_sampler = DistributedEvalSampler(
                validation_dataset,
                num_replicas=world_size,
                rank=rank,
            )
        generator = torch.Generator().manual_seed(args.seed + rank)
        train_loader_options = {
            "num_workers": args.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": seed_worker,
            "persistent_workers": args.num_workers > 0,
        }
        if args.num_workers > 0:
            train_loader_options["prefetch_factor"] = 2
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            drop_last=True,
            generator=generator,
            **train_loader_options,
        )
        validation_loader_options = {
            "num_workers": args.val_num_workers,
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": seed_worker,
            "persistent_workers": args.val_num_workers > 0,
        }
        if args.val_num_workers > 0:
            validation_loader_options["prefetch_factor"] = 2
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.val_batch_size,
            sampler=validation_sampler,
            shuffle=False,
            drop_last=False,
            **validation_loader_options,
        )

        model = build_model_serialized(
            args, distributed, rank, world_size, device
        )
        resume_state = None
        if args.resume:
            resume_state = load_checkpoint(args.resume, model)
        if distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        raw_model = unwrap_model(model)

        optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in raw_model.parameters()
                if parameter.requires_grad
            ],
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
        min_ratio = args.min_learning_rate / args.learning_rate
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: cosine_lambda(
                epoch,
                args.epochs,
                args.warmup_epochs,
                min_ratio,
            ),
        )
        scaler = create_grad_scaler(args.amp_dtype)
        criterion = L1SAMSSIMLoss(
            sam_weight=args.sam_weight,
            ssim_weight=args.ssim_weight,
            ssim_scales=args.ssim_scales,
            ssim_window_size=args.ssim_window_size,
            ssim_sigma=args.ssim_sigma,
        ).to(device)

        start_epoch = 0
        best_epoch = -1
        best_psnr = float("-inf")
        best_ssim = float("-inf")
        last_validation_epoch = -1
        last_validation_metrics = {
            name: float("nan") for name in EVALUATION_METRIC_NAMES
        }
        if resume_state is not None:
            start_epoch = int(resume_state["epoch"]) + 1
            args.resume_source_epoch = int(resume_state["epoch"]) + 1
            previous = resume_state.get("metrics", {})
            restored_training_state = resume_state.get("training_state", {})
            best_epoch = int(
                restored_training_state.get(
                    "best_epoch",
                    resume_state["epoch"] if "psnr" in previous else -1,
                )
            )
            best_psnr = float(
                restored_training_state.get(
                    "best_psnr",
                    previous.get("psnr", float("-inf")),
                )
            )
            best_ssim = float(
                restored_training_state.get(
                    "best_ssim",
                    previous.get("ssim", float("-inf")),
                )
            )
            last_validation_epoch = int(
                restored_training_state.get(
                    "last_validation_epoch",
                    resume_state["epoch"] if "psnr" in previous else -1,
                )
            )
            restored_metrics = restored_training_state.get(
                "last_validation_metrics",
                previous,
            )
            last_validation_metrics = {
                name: float(restored_metrics.get(name, float("nan")))
                for name in EVALUATION_METRIC_NAMES
            }
            if args.load_optimizer:
                optimizer.load_state_dict(resume_state["optimizer_state_dict"])
                scheduler.load_state_dict(resume_state["scheduler_state_dict"])
                if scaler.is_enabled() and "scaler_state_dict" in resume_state:
                    scaler.load_state_dict(resume_state["scaler_state_dict"])

        if args.start_epoch is not None:
            requested_index = args.start_epoch - 1
            jumped = requested_index != start_epoch
            start_epoch = requested_index
            align_scheduler_epoch(optimizer, scheduler, start_epoch)
            if jumped:
                # A numbering jump must not exhaust old early-stop patience.
                # Establish best metrics from actual validation in this new run.
                best_epoch = -1
                best_psnr = float("-inf")
                best_ssim = float("-inf")
                last_validation_epoch = -1
                last_validation_metrics = {
                    name: float("nan") for name in EVALUATION_METRIC_NAMES
                }
            if is_main:
                print({
                    "resume_checkpoint": args.resume,
                    "checkpoint_completed_epoch": args.resume_source_epoch,
                    "first_training_epoch": args.start_epoch,
                    "learning_rate_aligned": optimizer.param_groups[0]["lr"],
                    "best_and_early_stop_reset": jumped,
                }, flush=True)

        if is_main:
            print(
                {
                    "project_path": os.getcwd(),
                    "train_samples": len(train_dataset),
                    "validation_samples": len(validation_dataset),
                    "distributed_validation": distributed,
                    "validation_batches_on_rank0": len(validation_loader),
                    "validation_batch_per_dcu": args.val_batch_size,
                    "validation_workers_per_dcu": args.val_num_workers,
                    "validation_interval": args.val_interval,
                    "world_size": world_size,
                    "micro_batch_per_dcu": args.batch_size,
                    "accumulation_steps": args.accumulation_steps,
                    "effective_global_batch": args.batch_size
                    * world_size
                    * args.accumulation_steps,
                    "device": str(device),
                    "backend": dist.get_backend() if distributed else "none",
                    "amp_dtype": args.amp_dtype,
                    "foundation": args.foundation_name,
                    "foundation_checkpoint": str(
                        Path(args.foundation_checkpoint).resolve()
                    ),
                    "loss": (
                        f"L1 + {args.sam_weight}*SAM + "
                        f"{args.ssim_weight}*MS-SSIM-loss"
                    ),
                    "structure_bands": "S2 B2/B3/B4/B8",
                    "ssim_scales": args.ssim_scales,
                    **raw_model.parameter_report(),
                },
                flush=True,
            )

        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            epoch_learning_rate = optimizer.param_groups[0]["lr"]
            epoch_start = time.time()
            local_sums, local_count = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                scaler,
                args.amp_dtype,
                args.accumulation_steps,
                args.gradient_clip,
                is_main,
            )
            train_metrics = reduce_components(
                local_sums, local_count, device, distributed
            )
            train_seconds = time.time() - epoch_start
            # Advance before checkpointing so a resumed run starts with exactly
            # the learning rate of the next epoch.
            scheduler.step()
            stop_training = False
            should_validate = (
                epoch == start_epoch
                or (epoch + 1) % args.val_interval == 0
                or epoch + 1 == args.epochs
            )
            validation_seconds = 0.0
            validation_metrics = {
                name: float("nan") for name in EVALUATION_METRIC_NAMES
            }
            if should_validate:
                validation_start = time.time()
                validation_metrics = evaluate(
                    raw_model,
                    validation_loader,
                    device,
                    args.amp_dtype,
                    figure_dir,
                    epoch,
                    max_visualizations=(
                        args.max_visualizations if is_main else 0
                    ),
                    distributed=distributed,
                    rank=rank,
                )
                validation_seconds = time.time() - validation_start

            if is_main:
                improved = False
                if should_validate:
                    last_validation_epoch = epoch
                    last_validation_metrics = {
                        name: float(validation_metrics[name])
                        for name in EVALUATION_METRIC_NAMES
                    }
                    psnr = validation_metrics["psnr"]
                    ssim = validation_metrics["ssim"]
                    improved = psnr > best_psnr or (
                        np.isclose(psnr, best_psnr) and ssim > best_ssim
                    )
                    if improved:
                        best_epoch = epoch
                        best_psnr = psnr
                        best_ssim = ssim

                training_state = {
                    "best_epoch": int(best_epoch),
                    "best_psnr": float(best_psnr),
                    "best_ssim": float(best_ssim),
                    "last_validation_epoch": int(last_validation_epoch),
                    "last_validation_metrics": {
                        name: float(value)
                        for name, value in last_validation_metrics.items()
                    },
                }
                row = {
                    "epoch": epoch + 1,
                    "learning_rate": epoch_learning_rate,
                    "validation_performed": int(should_validate),
                    **{
                        f"train_{name}": value
                        for name, value in train_metrics.items()
                    },
                    **validation_metrics,
                    "train_seconds": train_seconds,
                    "validation_seconds": validation_seconds,
                }
                append_metrics(metrics_path, row)
                for name, value in row.items():
                    if name != "epoch" and math.isfinite(float(value)):
                        writer.add_scalar(name, value, epoch + 1)

                # Always keep an exact continuation point.  best.pt is for
                # evaluation; last.pt is preferred after an interrupted run.
                save_checkpoint(
                    model_dir / "last.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    last_validation_metrics,
                    args,
                    training_state,
                )
                if improved:
                    save_checkpoint(
                        model_dir / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        validation_metrics,
                        args,
                        training_state,
                    )
                    print(
                        f"更新最佳模型：epoch={epoch + 1}，"
                        f"PSNR={validation_metrics['psnr']:.5f}，"
                        f"SSIM={validation_metrics['ssim']:.5f}",
                        flush=True,
                    )
                if (
                    should_validate
                    and best_epoch >= 0
                    and epoch - best_epoch >= args.early_stop_patience
                ):
                    print(
                        f"连续{args.early_stop_patience}个epoch未提升，停止训练；"
                        f"最佳epoch={best_epoch + 1}",
                        flush=True,
                    )
                    stop_training = True
                print(
                    {
                        "epoch": epoch + 1,
                        "train": train_metrics,
                        "validation": (
                            validation_metrics if should_validate else "skipped"
                        ),
                        "train_minutes": train_seconds / 60.0,
                        "validation_minutes": (
                            validation_seconds / 60.0
                            if should_validate
                            else 0.0
                        ),
                    },
                    flush=True,
                )

            stop_tensor = torch.tensor(
                [int(stop_training)], dtype=torch.int32, device=device
            )
            if distributed:
                dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item():
                break
    finally:
        if writer is not None:
            writer.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
