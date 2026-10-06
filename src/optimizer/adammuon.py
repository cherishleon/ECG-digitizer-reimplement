from typing import Any, Callable, Optional, overload

import torch
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer


def newton_schulz_compute_dtype(device: torch.device) -> torch.dtype:
    """Use native BF16 on Ampere+ and FP32 on older CUDA GPUs such as V100."""
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float32


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    compute_dtype = newton_schulz_compute_dtype(G.device)
    # Normalize and iterate in FP32 on V100 to avoid FP16 numerical instability.
    X = G.float()
    X /= X.norm() + eps
    X = X.to(compute_dtype)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(Optimizer):
    def __init__(self, params: Any, lr: float, momentum: float, weight_decay: float):
        # weight_decay lives in the defaults so that a caller passing explicit
        # parameter groups can override it per group.
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.weight_decay = weight_decay

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group.get("weight_decay", self.weight_decay)
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum)

                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)  # whiten the update

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)  # apply weight decay

                p.data.add_(update, alpha=-lr)  # take a step
        return loss


class AdamMuon(Optimizer):
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-3,
        muon_momentum: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-5,
    ):
        # Accept either a flat parameter list or explicit groups. Groups let the
        # caller set weight_decay per group, which matters because a following
        # normalisation layer cancels the scale of the weights before it: decay
        # then shrinks them with nothing pushing back, and the gradient gain
        # through that norm grows without bound.
        if isinstance(params, (list, tuple)) and params and isinstance(params[0], dict):
            incoming = [dict(group) for group in params]
        else:
            incoming = [{"params": list(params)}]

        muon_groups: list[dict[str, Any]] = []
        adam_groups: list[dict[str, Any]] = []
        for group in incoming:
            group_weight_decay = group.get("weight_decay", weight_decay)
            muon_params = [p for p in group["params"] if p.ndim > 1]
            adam_params = [p for p in group["params"] if p.ndim <= 1]
            if muon_params:
                muon_groups.append(
                    {"params": muon_params, "lr": lr, "momentum": muon_momentum,
                     "weight_decay": group_weight_decay}
                )
            if adam_params:
                adam_groups.append(
                    {"params": adam_params, "lr": lr, "betas": betas, "eps": eps,
                     "weight_decay": group_weight_decay}
                )

        self.muon = Muon(muon_groups, lr=lr, momentum=muon_momentum, weight_decay=weight_decay) if muon_groups else None
        self.adam = AdamW(adam_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay) if adam_groups else None

        # Combine param groups for compatibility with PyTorch schedulers
        param_groups = []
        if self.muon:
            for g in self.muon.param_groups:
                g["optimizer"] = "muon"
                param_groups.append(g)
        if self.adam:
            for g in self.adam.param_groups:
                g["optimizer"] = "adam"
                param_groups.append(g)

        defaults = dict(lr=lr, momentum=muon_momentum, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()

        if self.muon:
            self.muon.step()
        if self.adam:
            self.adam.step()

        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self.muon:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adam:
            self.adam.zero_grad(set_to_none=set_to_none)
