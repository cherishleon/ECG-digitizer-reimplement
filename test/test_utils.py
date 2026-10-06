import torch
from torch.utils.data import DistributedSampler

from src.config.default import get_cfg
from src.model.unet import UNet
from src.optimizer.adammuon import newton_schulz_compute_dtype, zeropower_via_newtonschulz5
from src.utils import CosineToConstantLR, get_data_loaders


def test_cosine_to_constant_lr() -> None:
    T_MAX = 10
    LR = 0.1
    ETA_MIN_DIVISOR = 10.0
    eta_min = LR / ETA_MIN_DIVISOR
    eps = 1e-7

    model = torch.nn.Sequential(torch.nn.Linear(10, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineToConstantLR(optimizer, eta_min_divisor=ETA_MIN_DIVISOR, T_max=T_MAX)

    prev_lr = scheduler.get_lr()
    for lr in prev_lr:
        assert abs(lr - LR) < eps

    for _ in range(T_MAX - 1):
        scheduler.step()
        for curr_lr, curr_prev_lr in zip(scheduler.get_lr(), prev_lr):
            assert eta_min < curr_lr < curr_prev_lr

    for _ in range(10):
        scheduler.step()

        for lr in scheduler.get_lr():
            assert abs(lr - eta_min) < eps


def test_distributed_data_loaders_use_distributed_samplers() -> None:
    cfg = get_cfg("./test/test_data/config/unet.yml")
    train_loader, val_loader, test_loader = get_data_loaders(
        cfg.DATASET,
        cfg.DATALOADER.KWARGS,
        distributed=True,
        rank=0,
        world_size=2,
    )

    for loader in (train_loader, val_loader, test_loader):
        assert isinstance(loader.sampler, DistributedSampler)
        assert loader.sampler.rank == 0
        assert loader.sampler.num_replicas == 2

    assert train_loader.sampler.shuffle is False
    assert val_loader.sampler.shuffle is False
    assert test_loader.sampler.shuffle is False


def test_newton_schulz_compute_dtype_by_device(monkeypatch) -> None:
    assert newton_schulz_compute_dtype(torch.device("cpu")) == torch.float32

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (7, 0))
    assert newton_schulz_compute_dtype(torch.device("cuda")) == torch.float32

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 0))
    assert newton_schulz_compute_dtype(torch.device("cuda")) == torch.bfloat16


def test_newton_schulz_cpu_output_is_finite_fp32() -> None:
    output = zeropower_via_newtonschulz5(torch.randn(4, 3))
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_unet_all_trainable_parameters_receive_gradients() -> None:
    model = UNet(num_in_channels=3, num_out_channels=4, depth=1, dims=[2, 4, 8])
    output = model(torch.randn(1, 3, 32, 32))
    output.mean().backward()

    missing_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing_gradients == []

    frozen_parameters = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    assert frozen_parameters == {
        "encoder_downscaling.2.weight",
        "decoder_skips.0.0.weight",
        "decoder_skips.1.0.weight",
    }
