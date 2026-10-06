import argparse
import math
import multiprocessing
import os
import socket
import tempfile
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
import torch._dynamo
import torch.distributed as dist
import torch.multiprocessing as mp
from ray.train import Checkpoint
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from yacs.config import CfgNode as CN

from src.config.default import get_cfg, merge_cfg
from src.utils import (
    find_config_path,
    get_data_loaders,
    import_class_from_path,
    load_model,
)


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not distributed_is_initialized() or dist.get_rank() == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DDP/torch.compile wrappers down to the underlying nn.Module."""
    while True:
        if isinstance(model, DDP):
            model = model.module
            continue
        original_model = getattr(model, "_orig_mod", None)
        if isinstance(original_model, nn.Module):
            model = original_model
            continue
        break
    return model


def model_state_dict(model: nn.Module) -> Dict[str, Any]:
    """Return a checkpoint that is loadable without DDP/torch.compile wrappers."""
    return unwrap_model(model).state_dict()


def save_training_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    mixed_precision_scaler: Optional[torch.amp.grad_scaler.GradScaler],
    best_val_metrics: Dict[str, float],
    lowest_val_loss: float,
    running_train_losses: List[float],
    running_val_losses: List[float],
) -> None:
    """Save everything needed to resume training exactly where it left off.

    Written to a temp file first and atomically renamed, so a job killed mid-save
    (e.g. by a cluster preemption) can never leave a truncated checkpoint that a
    subsequent resume would fail to load.
    """
    tmp_path = path + ".tmp"
    torch.save(
        {
            "epoch": epoch,
            "model": model_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "scaler": mixed_precision_scaler.state_dict() if mixed_precision_scaler is not None else None,
            "best_val_metrics": best_val_metrics,
            "lowest_val_loss": lowest_val_loss,
            "running_train_losses": running_train_losses,
            "running_val_losses": running_val_losses,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def load_training_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    mixed_precision_scaler: Optional[torch.amp.grad_scaler.GradScaler],
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if lr_scheduler is not None and checkpoint.get("lr_scheduler") is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    if mixed_precision_scaler is not None and checkpoint.get("scaler") is not None:
        mixed_precision_scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def synchronize_epoch_statistics(
    total_loss: float,
    num_cases: int,
    metric_sums: Dict[str, float],
    metric_counts: Dict[str, int],
    device: torch.device,
) -> Tuple[float, int, Dict[str, float], Dict[str, int]]:
    """Sum epoch statistics over all DDP processes."""
    if not distributed_is_initialized():
        return total_loss, num_cases, metric_sums, metric_counts

    metric_names = list(metric_sums)
    values = [total_loss, float(num_cases)]
    for name in metric_names:
        values.extend([metric_sums[name], float(metric_counts[name])])

    statistics = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(statistics, op=dist.ReduceOp.SUM)

    total_loss = float(statistics[0].item())
    num_cases = int(statistics[1].item())
    for index, name in enumerate(metric_names):
        metric_sums[name] = float(statistics[2 + index * 2].item())
        metric_counts[name] = int(statistics[3 + index * 2].item())
    return total_loss, num_cases, metric_sums, metric_counts


_AMP_DTYPES: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "none": torch.float32,
}


def resolve_amp_dtype(name: Any, device: torch.device) -> torch.dtype:
    """Map the TRAIN.AMP_DTYPE config string onto a torch dtype.

    'float32' disables autocast entirely. BF16 is only a real option on Ampere
    and newer; on a V100 it exists but is emulated, so warn instead of silently
    running an order of magnitude slower.
    """
    key = str(name).lower()
    if key not in _AMP_DTYPES:
        raise ValueError(
            f"Unsupported TRAIN.AMP_DTYPE {name!r}; expected one of {sorted(set(_AMP_DTYPES))}."
        )
    dtype = _AMP_DTYPES[key]
    if dtype is torch.bfloat16 and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
        print(
            f"WARNING: TRAIN.AMP_DTYPE 'bfloat16' on {torch.cuda.get_device_name(device)} "
            "(pre-Ampere) has no hardware support and will be emulated."
        )
    return dtype


def optimized_parameters(optimizer: torch.optim.Optimizer) -> List[torch.Tensor]:
    return [parameter for group in optimizer.param_groups for parameter in group["params"]]


def parameter_groups_without_normalized_decay(
    model: nn.Module, weight_decay: float
) -> List[Dict[str, Any]]:
    """Split parameters into a decayed and an undecayed group.

    A normalisation layer divides out the scale of whatever produced its input,
    so the forward pass is invariant to the norm of the convolution weights
    feeding it, and to the norm's own affine weight. Weight decay on those
    tensors therefore shrinks them with no opposing force from the loss. As they
    shrink the variance reaching the norm falls below its eps, at which point the
    norm's backward gain saturates at 1 / sqrt(eps) and compounds over depth
    until the gradient overflows. Excluding them is the standard remedy.
    """
    normalization = (nn.modules.batchnorm._NormBase, nn.GroupNorm, nn.LayerNorm)
    exempt: set = set()
    for module in model.modules():
        if isinstance(module, normalization):
            for parameter in module.parameters(recurse=False):
                exempt.add(id(parameter))
        if isinstance(module, nn.Sequential):
            children = list(module)
            for current, following in zip(children, children[1:]):
                if isinstance(current, nn.modules.conv._ConvNd) and isinstance(following, normalization):
                    if current.weight is not None:
                        exempt.add(id(current.weight))

    decayed, undecayed = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (undecayed if id(parameter) in exempt else decayed).append(parameter)
    groups = []
    if decayed:
        groups.append({"params": decayed, "weight_decay": weight_decay})
    if undecayed:
        groups.append({"params": undecayed, "weight_decay": 0.0})
    return groups


def clip_or_measure_gradients(
    optimizer: torch.optim.Optimizer, grad_clip: Optional[float]
) -> torch.Tensor:
    """Return the total gradient norm, clipping to `grad_clip` when one is set.

    The norm doubles as the non-finite check: NaN or inf in any gradient
    propagates into it, so one reduction covers every parameter. A GradScaler
    performs its own check, but BF16 and FP32 runs have no scaler at all, and
    without this a NaN gradient would reach the optimizer unopposed.

    The accumulation is FP64 on purpose. An L2 norm squares its inputs, so
    gradients past about 1e19 overflow an FP32 accumulator while every element is
    still perfectly finite. That would report healthy gradients as non-finite,
    and clip_grad_norm_ would compound it by scaling with max_norm / inf == 0,
    silently zeroing them. Clipping is applied here rather than delegated for the
    same reason.
    """
    parameters = optimized_parameters(optimizer)
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return torch.zeros((), device=parameters[0].device if parameters else None)

    per_tensor = torch.stack(
        [torch.linalg.vector_norm(gradient, dtype=torch.float64) for gradient in gradients]
    )
    total_norm = torch.linalg.vector_norm(per_tensor)

    if grad_clip and bool(torch.isfinite(total_norm)):
        clip_coefficient = float(grad_clip) / (float(total_norm) + 1e-6)
        if clip_coefficient < 1.0:
            torch._foreach_mul_(gradients, clip_coefficient)
    return total_norm.to(torch.float32)


def all_ranks_agree(flag: bool, device: torch.device) -> bool:
    """Reduce a per-rank boolean to a value every rank agrees on.

    Every rank must take the same branch around the optimizer step, otherwise
    the replicas silently drift apart (and a rank that skips its backward pass
    deadlocks the DDP gradient all-reduce).
    """
    if not distributed_is_initialized():
        return flag
    flag_tensor = torch.tensor([1.0 if flag else 0.0], dtype=torch.float32, device=device)
    dist.all_reduce(flag_tensor, op=dist.ReduceOp.MIN)
    return bool(flag_tensor.item() > 0)


def run_epoch(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: torch.utils.data.DataLoader[Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    mixed_precision_scaler: Optional[torch.amp.grad_scaler.GradScaler] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    eval: bool = False,
    epoch: Optional[int] = None,
    max_epochs: Optional[int] = None,
    device: str | torch.device = "cpu",
    return_epoch_data: bool = True,
    metric_instances: Optional[List[nn.Module]] = None,
    disable_tqdm: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    grad_clip: Optional[float] = None,
    min_grad_scale: Optional[float] = 1.0,
) -> Tuple[float, int, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Dict[str, float]]:
    if not eval and optimizer is None:
        raise ValueError("Optimizer must be provided for training (eval=False).")

    if len(dataloader) == 0:
        raise ValueError("Dataloader must not be empty.")

    if not isinstance(device, torch.device):
        device = torch.device(device)

    if eval:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    num_cases = 0
    predictions: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    progress_bar = tqdm(dataloader) if not disable_tqdm else dataloader

    metrics = metric_instances or []
    metric_sums: Dict[str, float] = {metric.__name__: 0.0 for metric in metrics}  # type: ignore
    metric_counts: Dict[str, int] = {metric.__name__: 0 for metric in metrics}  # type: ignore
    autocast_enabled = amp_dtype != torch.float32
    scaler_enabled = mixed_precision_scaler is not None and mixed_precision_scaler.is_enabled()
    num_skipped_batches = 0
    num_skipped_gradients = 0

    for predictors, target in progress_bar:
        with torch.set_grad_enabled(not eval):
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            predictors = predictors.to(device)
            target = target.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                output = model(predictors)

                if return_epoch_data:
                    predictions.append(output.detach().cpu())
                    targets.append(target.detach().cpu())

                # Keep the memory-intensive model forward under AMP, but
                # calculate Dice + Focal loss explicitly in FP32.
                with torch.autocast(device_type=device.type, enabled=False):
                    loss = criterion(output.float(), target.float())

                with torch.no_grad():
                    for metric in metrics:
                        metric_value = metric(output, target)
                        metric_value = metric_value.detach().float()
                        metric_sums[metric.__name__] += float(metric_value.sum().item())  # type: ignore
                        metric_counts[metric.__name__] += metric_value.numel()  # type: ignore

            # Everything below deliberately runs OUTSIDE the autocast context.
            # Autocast re-casts every matmul it sees, so keeping the optimizer
            # step inside it silently pulled Muon's Newton-Schulz iteration back
            # into FP16 regardless of the dtype adammuon.py had chosen.
            loss_value = float(loss.detach().item())
            batch_is_finite = all_ranks_agree(math.isfinite(loss_value), device)

            if not batch_is_finite:
                # One non-finite batch must neither reach the optimizer nor
                # poison the running epoch average for every later epoch.
                num_skipped_batches += 1
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
            elif eval:
                pass
            elif scaler_enabled:
                mixed_precision_scaler.scale(loss).backward()  # type: ignore
                if grad_clip:
                    mixed_precision_scaler.unscale_(optimizer)  # type: ignore
                    clip_or_measure_gradients(optimizer, grad_clip)  # type: ignore
                scale_before = mixed_precision_scaler.get_scale()  # type: ignore
                # GradScaler runs its own inf check and skips the step itself; a
                # backed-off scale afterwards is how that skip becomes visible.
                mixed_precision_scaler.step(optimizer)  # type: ignore
                mixed_precision_scaler.update()  # type: ignore
                if mixed_precision_scaler.get_scale() < scale_before:  # type: ignore
                    num_skipped_gradients += 1
                if min_grad_scale and mixed_precision_scaler.get_scale() < min_grad_scale:  # type: ignore
                    # A scale that decays to 0 makes unscale_() multiply by
                    # 1/0 == inf. Zero gradients then become 0 * inf == NaN
                    # *without* tripping the inf check (it inspects the value
                    # before the multiply), so the step runs and every single
                    # parameter turns into NaN at once.
                    mixed_precision_scaler.update(new_scale=float(min_grad_scale))  # type: ignore
            else:
                loss.backward()
                # No scaler here (BF16/FP32), so nothing else checks the
                # gradients. A finite loss can still backward into NaN, and a
                # single NaN gradient turns every parameter NaN in one step.
                total_norm = clip_or_measure_gradients(optimizer, grad_clip)
                if all_ranks_agree(bool(torch.isfinite(total_norm)), device):
                    optimizer.step()  # type: ignore
                else:
                    num_skipped_gradients += 1
                    optimizer.zero_grad(set_to_none=True)  # type: ignore

            if lr_scheduler and not eval:
                lr_scheduler.step()

            if batch_is_finite:
                batch_size = predictors.shape[0]
                total_loss += loss_value * batch_size
                num_cases += batch_size

            if epoch is not None and not disable_tqdm and num_cases > 0:
                max_epochs_str = f"/{max_epochs}" if max_epochs else ""
                description = f"Epoch {epoch + 1}{max_epochs_str}, Loss: {total_loss / num_cases:.4f}"
                if num_skipped_batches or num_skipped_gradients:
                    description += f", skipped: {num_skipped_batches}/{num_skipped_gradients}"
                progress_bar.set_description(description)  # type: ignore

    if (num_skipped_batches or num_skipped_gradients) and is_main_process():
        split = "val" if eval else "train"
        scale_str = f", grad scale: {mixed_precision_scaler.get_scale()}" if scaler_enabled else ""
        print(
            f"WARNING: {split} epoch skipped {num_skipped_batches} batches with a non-finite loss "
            f"and {num_skipped_gradients} steps with non-finite gradients{scale_str}."
        )

    total_loss, num_cases, metric_sums, metric_counts = synchronize_epoch_statistics(
        total_loss, num_cases, metric_sums, metric_counts, device
    )
    calculated_metrics = {
        name: metric_sums[name] / metric_counts[name]
        for name in metric_sums
        if metric_counts[name] > 0
    }

    return total_loss, num_cases, predictors, predictions, targets, calculated_metrics


def train(
    model: nn.Module,
    criterion: nn.Module,
    train_dataloader: torch.utils.data.DataLoader[Any],
    val_dataloader: torch.utils.data.DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    mixed_precision_scaler: Optional[torch.amp.grad_scaler.GradScaler],
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    use_ray: bool,
    epochs: int,
    metrics: Dict[Any, Any],
    device: str | torch.device = "cpu",
    checkpoint_dir: Optional[str] = "sandbox",
    amp_dtype: torch.dtype = torch.float16,
    grad_clip: Optional[float] = None,
    min_grad_scale: Optional[float] = 1.0,
    resume: bool = False,
) -> None:
    if not isinstance(device, torch.device):
        device = torch.device(device)

    main_process = is_main_process()
    # print(f"Training model:\n {model}")
    if main_process:
        print(f"Using device: {device} ({dist.get_world_size() if distributed_is_initialized() else 1} process(es))")
        print(f"Trainable model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        scaler_state = "on" if (mixed_precision_scaler and mixed_precision_scaler.is_enabled()) else "off"
        print(
            f"AMP dtype: {amp_dtype}, grad scaler: {scaler_state}, "
            f"grad clip: {grad_clip or 'off'}, min grad scale: {min_grad_scale or 'off'}"
        )
    model.to(device)

    # Resolved on every rank (not just main_process) so that a resume can load
    # the same checkpoint file everywhere; only main_process writes to it.
    resolved_checkpoint_dir = None
    if not use_ray and checkpoint_dir:
        resolved_checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir))
        if main_process:
            os.makedirs(resolved_checkpoint_dir, exist_ok=True)
            print(f"Saving checkpoints to: {resolved_checkpoint_dir}")
    if distributed_is_initialized():
        dist.barrier()

    running_train_losses = []
    running_val_losses = []
    num_epochs = epochs

    metrics_per_split = {}
    val_split_prefix = "val"
    train_split_prefix = "train"

    for split_name in [train_split_prefix, val_split_prefix]:
        curr_metrics = []
        for metric in metrics:
            if metric.get("fit", False):
                metric_class = import_class_from_path(metric["class_path"])(split_name=split_name, **metric["KWARGS"])
                metric_class.fit(train_dataloader.dataset.data)  # type: ignore
            else:
                metric_class = import_class_from_path(metric["class_path"])(split_name=split_name, **metric["KWARGS"])
            curr_metrics.append(metric_class)
        metrics_per_split[split_name] = curr_metrics

    best_val_metrics = {metric.__name__: -np.inf for metric in metrics_per_split[val_split_prefix]}
    lowest_val_loss = np.inf
    start_epoch = 0

    if resume and resolved_checkpoint_dir:
        latest_checkpoint_path = os.path.join(resolved_checkpoint_dir, "checkpoint_last.pt")
        if os.path.exists(latest_checkpoint_path):
            checkpoint = load_training_checkpoint(
                latest_checkpoint_path, model, optimizer, lr_scheduler, mixed_precision_scaler, device
            )
            start_epoch = int(checkpoint["epoch"]) + 1
            running_train_losses = list(checkpoint.get("running_train_losses", []))
            running_val_losses = list(checkpoint.get("running_val_losses", []))
            best_val_metrics.update(checkpoint.get("best_val_metrics", {}))
            lowest_val_loss = float(checkpoint.get("lowest_val_loss", lowest_val_loss))
            if main_process:
                print(f"Resuming training from {latest_checkpoint_path}, starting at epoch {start_epoch + 1}.")
        elif main_process:
            print(f"TRAIN.KWARGS.resume is set but no checkpoint found at {latest_checkpoint_path}; starting fresh.")

    if start_epoch >= num_epochs:
        if main_process:
            print(f"Checkpoint is already at epoch {start_epoch}/{num_epochs}; nothing to train.")
        return

    if distributed_is_initialized():
        dist.barrier()

    epoch_bar = (
        tqdm(range(start_epoch, num_epochs), desc="Training epochs")
        if use_ray
        else range(start_epoch, num_epochs)
    )

    for epoch in epoch_bar:
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)  # type: ignore

        train_loss, train_cases, _, _, _, train_metrics = run_epoch(
            model,
            criterion,
            train_dataloader,
            optimizer,
            mixed_precision_scaler=mixed_precision_scaler,
            lr_scheduler=lr_scheduler,
            eval=False,
            epoch=epoch,
            max_epochs=num_epochs,
            device=device,
            return_epoch_data=False,
            metric_instances=metrics_per_split[train_split_prefix],
            disable_tqdm=use_ray or not main_process,
            amp_dtype=amp_dtype,
            grad_clip=grad_clip,
            min_grad_scale=min_grad_scale,
        )
        if train_cases == 0:
            raise RuntimeError(
                f"Every training batch of epoch {epoch + 1} produced a non-finite loss; "
                "training cannot continue. Lower TRAIN.OPTIMIZER.KWARGS.lr or switch "
                "TRAIN.AMP_DTYPE to 'bfloat16'/'float32'."
            )
        running_train_losses.append(train_loss / train_cases)

        val_loss, val_cases, _, _, _, val_metrics = run_epoch(
            model,
            criterion,
            val_dataloader,
            None,
            mixed_precision_scaler=mixed_precision_scaler,
            eval=True,
            epoch=epoch,
            max_epochs=num_epochs,
            device=device,
            return_epoch_data=False,
            metric_instances=metrics_per_split[val_split_prefix],
            disable_tqdm=use_ray or not main_process,
            amp_dtype=amp_dtype,
            grad_clip=grad_clip,
            min_grad_scale=min_grad_scale,
        )
        if val_cases == 0:
            raise RuntimeError(
                f"Every validation batch of epoch {epoch + 1} produced a non-finite loss; "
                "the model has most likely diverged."
            )

        running_val_losses.append(val_loss / val_cases)

        calculated_metrics = train_metrics | val_metrics
        curr_val_loss = val_loss / val_cases
        is_new_best_metric = curr_val_loss < lowest_val_loss
        lowest_val_loss = min(curr_val_loss, lowest_val_loss)

        for name, value in val_metrics.items():
            if value < best_val_metrics[name]:
                best_val_metrics[name] = value
                is_new_best_metric = True

        results = {"train_loss": train_loss / train_cases, "val_loss": curr_val_loss} | calculated_metrics

        if not use_ray:
            if main_process:
                print(f"Epoch {epoch + 1}/{num_epochs}: {results}")
                if resolved_checkpoint_dir:
                    checkpoint_path = os.path.join(resolved_checkpoint_dir, f"weights_{epoch + 1}.pt")
                    torch.save(model_state_dict(model), checkpoint_path)
                    save_training_checkpoint(
                        os.path.join(resolved_checkpoint_dir, "checkpoint_last.pt"),
                        epoch,
                        model,
                        optimizer,
                        lr_scheduler,
                        mixed_precision_scaler,
                        best_val_metrics,
                        lowest_val_loss,
                        running_train_losses,
                        running_val_losses,
                    )
        elif is_new_best_metric:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                torch.save((model_state_dict(model), optimizer.state_dict()), path)
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                ray.train.report(
                    results,
                    checkpoint=checkpoint,
                )
        else:
            ray.train.report(results)

    if main_process:
        print(f"Train losses: {running_train_losses}")
        print(f"Val losses: {running_val_losses}")
        print(f"Trainable model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


def load_and_train(ray_config: CN, config: CN) -> None:
    if ray_config:
        config = merge_cfg(config, CN(ray_config))
        with open(os.path.join(ray.train.get_context().get_trial_dir(), "config.yml"), "w") as f:
            f.write(config.dump())

    distributed = distributed_is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    configured_device = torch.device(config.TRAIN.KWARGS.device)
    if distributed:
        if configured_device.type != "cuda":
            raise ValueError("DDP training requires TRAIN.KWARGS.device to be 'cuda'.")
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    else:
        device = configured_device

    model = load_model(config)
    model.to(device)
    if config.TRAIN.COMPILE:
        model = torch.compile(model)  # type: ignore
    if distributed:
        model = DDP(model, device_ids=[device.index], output_device=device.index)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_kwargs = dict(config.TRAIN.OPTIMIZER.KWARGS)
    if bool(getattr(config.TRAIN, "DECOUPLE_NORMALIZED_WEIGHT_DECAY", False)):
        optimizer_input: Any = parameter_groups_without_normalized_decay(
            model, float(optimizer_kwargs.get("weight_decay", 0.0))
        )
        if rank == 0:
            decayed = sum(p.numel() for p in optimizer_input[0]["params"])
            undecayed = sum(p.numel() for group in optimizer_input[1:] for p in group["params"])
            print(f"Weight decay applied to {decayed} parameters, withheld from {undecayed} "
                  "normalised ones (norm affines and the convolutions they follow).")
    else:
        optimizer_input = trainable_parameters
    optimizer = import_class_from_path(config.TRAIN.OPTIMIZER.class_path)(
        optimizer_input, **optimizer_kwargs
    )

    # Default to FP16 so that configs written before TRAIN.AMP_DTYPE existed keep
    # their original behaviour.
    amp_dtype = resolve_amp_dtype(getattr(config.TRAIN, "AMP_DTYPE", "float16"), device)
    grad_clip = float(getattr(config.TRAIN, "GRAD_CLIP", 0.0)) or None
    min_grad_scale = float(getattr(config.TRAIN, "MIN_GRAD_SCALE", 1.0)) or None

    mixed_precision_scaler = None
    if hasattr(config.TRAIN, "MIXED_PRECISION_SCALER"):
        if amp_dtype is torch.float16:
            scaler_kwargs = dict(config.TRAIN.MIXED_PRECISION_SCALER.KWARGS)
            if config.TRAIN.MIXED_PRECISION_SCALER.class_path == "torch.amp.GradScaler":
                scaler_kwargs.setdefault("device", device.type)
            mixed_precision_scaler = import_class_from_path(
                config.TRAIN.MIXED_PRECISION_SCALER.class_path
            )(**scaler_kwargs)
        elif rank == 0:
            # Loss scaling only exists to rescue FP16's narrow exponent range;
            # BF16 and FP32 do not need it and a scaler is a liability there.
            print(f"Ignoring TRAIN.MIXED_PRECISION_SCALER: it is not used with AMP dtype {amp_dtype}.")

    lr_scheduler = (
        import_class_from_path(config.TRAIN.LR_SCHEDULER.class_path)(optimizer, **config.TRAIN.LR_SCHEDULER.KWARGS)
        if hasattr(config.TRAIN, "LR_SCHEDULER")
        else None
    )
    criterion_kwargs = dict(config.TRAIN.CRITERION.KWARGS) if hasattr(config.TRAIN, "CRITERION") else {}
    for key, value in criterion_kwargs.items():
        if isinstance(value, list):
            criterion_kwargs[key] = torch.tensor(value, dtype=torch.float32, device=device)
    criterion = import_class_from_path(config.TRAIN.CRITERION.class_path)(**criterion_kwargs)
    criterion.to(device)

    train_dataloader, val_dataloader, _ = get_data_loaders(
        config.DATASET,
        config.DATALOADER.KWARGS,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    train_kwargs = dict(config.TRAIN.KWARGS)
    train_kwargs["device"] = device
    train(
        model,
        criterion,
        train_dataloader,
        val_dataloader,
        optimizer,
        mixed_precision_scaler=mixed_precision_scaler,
        lr_scheduler=lr_scheduler,
        use_ray=bool(ray_config),
        amp_dtype=amp_dtype,
        grad_clip=grad_clip,
        min_grad_scale=min_grad_scale,
        **train_kwargs,
    )
    if distributed:
        dist.barrier()
    return None


def load_hyperparameter_search(config: CN) -> Any | Dict[str, Any]:
    # TODO: Add support for nested and conditional spaces (https://docs.ray.io/en/latest/tune/faq.html#nested-spaces).
    if "SAMPLE_TYPE" in config:
        return import_class_from_path(config["SAMPLE_TYPE"])(**config["KWARGS"])

    parsed_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            parsed_config[key] = load_hyperparameter_search(value)
    return parsed_config


def distributed_config_value(config: CN, key: str, default: Any) -> Any:
    if not hasattr(config.TRAIN, "DISTRIBUTED"):
        return default
    return getattr(config.TRAIN.DISTRIBUTED, key, default)


def requested_world_size(config: CN) -> int:
    if not bool(distributed_config_value(config, "ENABLED", False)):
        return 1
    if torch.device(config.TRAIN.KWARGS.device).type != "cuda":
        raise ValueError("TRAIN.DISTRIBUTED.ENABLED requires TRAIN.KWARGS.device: 'cuda'.")

    available_gpus = torch.cuda.device_count()
    requested_gpus = int(distributed_config_value(config, "NUM_GPUS", -1))
    if requested_gpus == -1:
        requested_gpus = available_gpus
    if requested_gpus < 1:
        raise RuntimeError("Distributed training is enabled, but no CUDA GPUs are visible.")
    if requested_gpus > available_gpus:
        raise RuntimeError(
            f"Requested {requested_gpus} GPUs, but only {available_gpus} are visible. "
            "Adjust TRAIN.DISTRIBUTED.NUM_GPUS or CUDA_VISIBLE_DEVICES."
        )
    return requested_gpus


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run_distributed_process(config: CN) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} CUDA GPUs are visible."
        )

    torch.cuda.set_device(local_rank)
    backend = str(distributed_config_value(config, "BACKEND", "nccl"))
    dist.init_process_group(backend=backend, init_method="env://")
    try:
        torch.backends.cudnn.benchmark = config.TRAIN.CUDNN_BENCHMARK
        load_and_train(None, config)
    finally:
        dist.destroy_process_group()


def spawn_distributed_worker(
    local_rank: int,
    world_size: int,
    config: CN,
    master_addr: str,
    master_port: int,
) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    run_distributed_process(config)


def main(config: CN) -> Optional[ExperimentAnalysis]:
    torch.backends.cudnn.benchmark = config.TRAIN.CUDNN_BENCHMARK
    external_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    search_space = config.HYPERPARAMETER_SEARCH.SEARCH_SPACE

    if not search_space:
        if external_world_size > 1:
            run_distributed_process(config)
            return None

        world_size = requested_world_size(config)
        if world_size == 1:
            return load_and_train(None, config)  # type: ignore

        master_addr = str(distributed_config_value(config, "MASTER_ADDR", "127.0.0.1"))
        configured_port = int(distributed_config_value(config, "MASTER_PORT", 0))
        master_port = configured_port if configured_port > 0 else find_free_port()
        print(f"Launching DDP training on {world_size} GPUs at {master_addr}:{master_port}")
        mp.spawn(
            spawn_distributed_worker,
            args=(world_size, config, master_addr, master_port),
            nprocs=world_size,
            join=True,
        )
        return None

    if external_world_size > 1 or requested_world_size(config) > 1:
        raise ValueError("DDP and Ray hyperparameter search cannot be enabled at the same time.")

    ray_config = load_hyperparameter_search(search_space)
    scheduler = (
        import_class_from_path(config.HYPERPARAMETER_SEARCH.SCHEDULER.class_path)(
            **config.HYPERPARAMETER_SEARCH.SCHEDULER.KWARGS
        )
        if "SCHEDULER" in config.HYPERPARAMETER_SEARCH
        else None
    )
    stopper = (
        import_class_from_path(config.TRAIN.STOPPER.class_path)(**config.TRAIN.STOPPER.KWARGS)
        if "STOPPER" in config.TRAIN
        else None
    )

    # Set seed for Ray Tune's random search. If you remove this line, you will
    # get different configurations each time you run the script.
    np.random.seed(42)

    result: ExperimentAnalysis = ray.tune.run(
        partial(load_and_train, config=config),
        resources_per_trial={"cpu": 16, "gpu": 1},
        config=ray_config,
        num_samples=1,
        scheduler=scheduler,
        stop=stopper,
    )
    return result


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # CUDA does not support "fork", which is default on linux.
    parser = argparse.ArgumentParser(description="Train a model with the specified configuration.")
    parser.add_argument(
        "--config",
        type=str,
        default="unet.yml",
        help="Config file name or path (searched in . and src/config/). Default: unet.yml",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from TRAIN.KWARGS.checkpoint_dir/checkpoint_last.pt instead of training from scratch.",
    )
    args = parser.parse_args()

    config_path = find_config_path(args.config)
    cfg = get_cfg(config_path)
    if args.resume:
        cfg.TRAIN.KWARGS.resume = True

    main(cfg)
