"""Two-rank, two-step CPU/Gloo smoke test for the published UNet DDP path.

Usage: python -m torch.distributed.run --standalone --nproc-per-node=2 scripts/smoke_ddp.py
"""
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.unet import UNet
from src.optimizer.adammuon import AdamMuon
from src.loss.loss import DiceFocalLoss

torch.set_num_threads(1)
dist.init_process_group("gloo")
try:
    rank = int(os.environ["RANK"])
    torch.manual_seed(1234)
    model = DistributedDataParallel(UNet(3, 4, 1, [2, 4, 8], norm_eps=1e-2))
    optimizer = AdamMuon([p for p in model.parameters() if p.requires_grad], lr=0.0037)
    criterion = DiceFocalLoss()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(42 + rank + step * 2)
        output = model(torch.rand(2, 3, 32, 32))
        loss = criterion(output.float(), torch.rand(2, 3, 32, 32))
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
        optimizer.step()
        flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        expected = flat.clone()
        dist.broadcast(expected, src=0)
        assert torch.equal(flat, expected), "Ranks diverged after optimizer update."
    if rank == 0:
        print("Two-rank Gloo UNet + FP32 Dice/Focal + AdamMuon, two optimizer steps: OK")
finally:
    dist.destroy_process_group()
