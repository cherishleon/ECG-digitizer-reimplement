"""Build path-sanitized provenance reports from local reproduction artifacts.

Run with explicit input paths; never copy clinical inputs or whole training logs.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from collections import Counter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(name: str, value: object) -> None:
    (REPO / "reports" / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def dataset_reports(root: Path) -> None:
    report = json.loads((root / "generation_report.json").read_text())
    report["config"]["output_dir"] = "data/npj_ecg_dataset_new"
    report["config"]["chinese_font"] = Path(report["config"]["chinese_font"]).name
    if report["config"].get("unrectified_overlay_files"):
        raise ValueError("Private overlays need separate review before publication.")
    if report.get("failures"):
        raise ValueError("Failure entries need separate path sanitization.")
    actual_counts = {}
    mask_counts = {}
    for split in ("train", "val", "test"):
        names = {entry.name for entry in os.scandir(root / split) if entry.is_file()}
        images = {name for name in names if name.endswith(".png") and not name.endswith("_mask.png")}
        masks = {name for name in names if name.endswith("_mask.png")}
        assert {name[:-4] + "_mask.png" for name in images} == masks, f"Image/mask mismatch in {split}."
        actual_counts[split] = len(images)
        mask_counts[split] = len(masks)
    report["actual_image_counts_at_release_audit"] = actual_counts
    write_json("generation_report.json", report)

    counts: Counter = Counter()
    groups: dict[tuple[str, str], set[str]] = {}
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        # Audit locally; publish only aggregate counts, never individual rows.
        for row in reader:
            counts[row["split"]] += 1
            stem = re.sub(r"__blank_\d+$", "", row["source_sample_id"])
            stem = re.sub(r"_(?:hr|lr)$", "", stem, flags=re.I)
            groups.setdefault((row["dataset"], stem), set()).add(row["split"])
    assert dict(counts) == {k: v for k, v in report["split_counts"].items() if v}
    overlaps = [key for key, splits in groups.items() if len(splits) > 1]
    assert not overlaps, "Source recording overlaps across splits."
    write_json(
        "dataset_audit.json",
        {
            "manifest_rows": sum(counts.values()),
            "source_record_groups": len(groups),
            "split_counts": dict(counts),
            "source_group_overlap_count": len(overlaps),
            "patient_level_isolation_verified": False,
            "images_match_generation_report": actual_counts == report["split_counts"],
            "mask_counts": mask_counts,
            "every_image_has_matching_mask": True,
            "manifest_published": False,
        },
    )


def training_report(log: Path) -> None:
    rows = []
    skipped_losses = skipped_gradients = 0
    for line in log.read_text().splitlines():
        skipped = re.search(
            r"train epoch skipped (\d+) batches.*?and (\d+) steps", line
        )
        if skipped:
            skipped_losses, skipped_gradients = map(int, skipped.groups())
        match = re.match(r"Epoch (\d+)/(\d+): (\{.*\})", line)
        if match:
            result = ast.literal_eval(match.group(3))
            row = {
                "epoch": int(match.group(1)),
                "planned_epochs": int(match.group(2)),
                "train_loss": result["train_loss"],
                "val_loss": result["val_loss"],
                "skipped_loss_batches": skipped_losses,
                "skipped_gradient_steps": skipped_gradients,
            }
            assert all(math.isfinite(row[key]) for key in ("train_loss", "val_loss"))
            rows.append(row)
            skipped_losses = skipped_gradients = 0
    if not rows:
        raise ValueError("No epoch results in log.")
    with (REPO / "reports" / "training_newdata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    best = min(rows, key=lambda row: row["val_loss"])
    write_json(
        "training_summary.json",
        {
            "source_log_label": "newdata_training_run",
            "source_log_sha256": sha256(log),
            "logged_epochs": len(rows),
            "best_epoch": best,
            "last_epoch": rows[-1],
            "hardware_from_log": "2 x NVIDIA A100-SXM4-80GB",
            "precision_from_log": "bfloat16, scaler off, grad_clip=1.0",
            "independent_test_result": None,
            "historical_config_dump_available": False,
        },
    )


def checkpoint_reports(local_dir: Path) -> None:
    import torch
    from src.config.default import get_cfg
    from src.model.unet import UNet

    torch.set_num_threads(2)
    published = []
    for path in sorted((REPO / "checkpoints").glob("*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert isinstance(state, dict) and all(
            isinstance(tensor, torch.Tensor) for tensor in state.values()
        ), "Expected a tensor-only model state_dict."
        finite = all(bool(torch.isfinite(tensor).all()) for tensor in state.values())
        assert finite, f"Non-finite checkpoint: {path.name}"
        # Upstream compiled models have this prefix; inference uses the same
        # normalization. Keep the checkpoint bytes unchanged in the release.
        stripped_prefix = any("_orig_mod." in key for key in state)
        normalized_state = {key.replace("_orig_mod.", ""): value for key, value in state.items()}
        assert len(normalized_state) == len(state), "Checkpoint key collision after prefix removal."
        if path.name.startswith("lead_name"):
            kwargs = dict(get_cfg(str(REPO / "src/config/lead_name_unet.yml")).MODEL.KWARGS)
        else:
            kwargs = dict(num_in_channels=3, num_out_channels=4, depth=2,
                          dims=[32, 64, 128, 256, 320, 320, 320, 320],
                          norm_eps=1e-2 if "newdata" in path.name else 1e-5)
        model = UNet(**kwargs)
        model.load_state_dict(normalized_state, strict=True)
        smoke_finite = None
        if "newdata" in path.name:
            model.eval()
            with torch.no_grad():
                output = model(torch.rand(1, 3, 256, 256))
            smoke_finite = bool(torch.isfinite(output).all())
            assert smoke_finite
        published.append(
            {"file": str(path.relative_to(REPO)), "bytes": path.stat().st_size,
             "sha256": sha256(path), "tensor_count": len(state),
             "all_tensors_finite": finite, "strict_load": True,
             "compiled_prefix_removed_for_validation": stripped_prefix,
             "synthetic_forward_finite": smoke_finite,
             "model_kwargs": kwargs, "optimizer_state_included": False}
        )
    write_json("checkpoints.json", published)
    write_json(
        "local_checkpoint_inventory.json",
        [{"filename": path.name, "bytes": path.stat().st_size,
          "included_in_release": path.name in {p["file"].split("/")[-1] for p in published}}
         for path in sorted(local_dir.glob("*.pt"))],
    )


def provenance(source: Path, upstream: Path) -> None:
    def revision(root: Path) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()

    files = []
    for path in sorted(source.rglob("*")):
        if path.suffix not in (".py", ".yml") or ".git" in path.parts:
            continue
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(source)
        if relative.parts[0] not in ("src", "test") and len(relative.parts) != 1:
            continue
        # Do not publish filenames from local-only inputs or private templates.
        # Renamed public configurations are inventoried separately below.
        if not (REPO / relative).is_file():
            continue
        baseline = upstream / relative
        files.append(
            {"file": str(relative), "source_sha256": sha256(path),
             "upstream_sha256": sha256(baseline) if baseline.exists() else None,
             "differs_from_upstream_snapshot": not baseline.exists()
             or sha256(path) != sha256(baseline)}
        )
    write_json(
        "code_provenance.json",
        {
            "source_local_commit_published": False,
            "upstream_audit_commit": revision(upstream),
            "upstream_url": "https://github.com/Ahus-AIM/Open-ECG-Digitizer",
            "note": "Current upstream snapshot is an audit baseline, not a proven original training commit.",
            "current_source_files": files,
            "file_selection": "Only source files with matching published paths; renamed public configurations are listed separately.",
            "published_configurations": [
                {"file": str(path.relative_to(REPO)), "sha256": sha256(path)}
                for path in sorted((REPO / "src/config").glob("*.yml"))
            ],
            "original_launchers": [
                {"file": name, "sha256": sha256(source.parent / name)}
                for name in ("run_a100_newdata.sh", "run_a100_v2.sh")
            ],
        },
    )
    packages = {}
    for name in ("torch", "torchvision", "numpy", "opencv-python", "pillow", "yacs", "ray", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    write_json(
        "environment_snapshot.json",
        {"python": sys.version.split()[0], "packages": packages,
         "purpose": "Release audit environment; historical training environment was not locked."},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--upstream-project", type=Path, required=True)
    args = parser.parse_args()
    (REPO / "reports").mkdir(exist_ok=True)
    dataset_reports(args.dataset_root)
    training_report(args.training_log)
    checkpoint_reports(args.checkpoint_dir)
    provenance(args.source_project, args.upstream_project)
    print("Dataset, training, checkpoint and source provenance reports: OK")


if __name__ == "__main__":
    main()
