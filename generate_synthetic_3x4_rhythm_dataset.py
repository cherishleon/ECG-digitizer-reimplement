#!/usr/bin/env python3
"""Render PTB-XL and Chapman waveforms as six-row ECG pages and masks.

The layout is three 3x4 rows followed by configurable 10-second rhythm strips
(V1, II and V5 by default). Each JPG and its sparse six-channel mask are made
in the same operation, so no later JPG/XML mask-generation pass is needed.

Defaults:
  PTB-XL  -> 3508x2480
  Chapman -> 3320x2552


Barcode labels are kept by default so the model learns to recover the hidden
waveform. ``--barcode-mask-policy drop`` is only for an explicit ablation.

它会直接完成：
- PTB-XL 波形 → 3508×2480 JPG。
- Chapman 波形 → 3320×2552 JPG。
- 同步生成六通道 (6,H,W) 稀疏 mask。
- 布局为上面 3×4，下面 V1 / II / V5。
- PTB-XL 随机生成粉色、红色、纯白、灰色点阵背景。
- 默认额外增加约 10% 的空白心电图，不替换原始波形。
- 支持摩尔纹、眩光、空白遮挡块。
- 默认 35% 概率在最右侧生成约 0.2 秒宽的条码，覆盖前五行。
- 条码和局部空白遮挡下的 mask 默认保留，用于训练模型恢复被遮挡波形。
- 输出 manifest.csv 和 generation_report.json。
- 支持断点续跑，已有 JPG+mask 默认跳过。
完整生成：
cd ECG-digitizer


Full run
python generate_synthetic_3x4_rhythm_dataset.py \
  --output-dir data/legacy_synthetic \
  --workers 24
如果有私有的未整流 JPG，可以加入低透明度叠加：
python ecg_guangzhou/generate_synthetic_3x4_rhythm_dataset.py \
  --output-dir data/legacy_synthetic \
  --unrectified-overlay-dir /path/to/private_raw_jpg \
  --unrectified-overlay-probability 0.15 \
  --workers 4
Smoke test::   建议先小规模检查：
python /generate_synthetic_3x4_rhythm_dataset.py \
  --output-dir data/legacy_smoke \
  --limit-ptb 10 \
  --limit-chapman 10
不需要先运行原来的 generate_3x4_rhythm_masks.py。原脚本只用于真实的 JPG+XML → mask；这个新脚本在画图时已经同步得到精确 mask。
训练时要使用相同的基线：
python imple/ecg_digitization_2/train_3x4_rhythm_masks.py \
  --csv data/legacy_synthetic/manifest.csv \
  --data-root data/legacy_synthetic \
  --mask-dir data/legacy_synthetic/masks \
  --output-dir outputs/legacy_checkpoints \
  --baselines 0.339,0.448,0.555,0.661,0.769,0.876 \
  --fusion-type shared_conv2d
已用 PTB-XL、Chapman、额外空白页及双进程完成测试：两种目标尺寸正确，所有 mask 均为六通道，空白页六个 mask 全为空，条码遮挡区域的监督仍然保留。全量约 3.3 万张高分辨率图片，建议先确认剩余磁盘空间。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

if __package__:
    from . import ecg_page_rendering as page_renderer
else:
    import ecg_page_rendering as page_renderer


LEAD_NAMES = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)
TOP_LAYOUT = (
    ("I", "aVR", "V1", "V4"),
    ("II", "aVL", "V2", "V5"),
    ("III", "aVF", "V3", "V6"),
)
DEFAULT_RHYTHM_LEADS = ("V1", "II", "V5")
# Legacy values used only with --layout-profile custom.
DEFAULT_BASELINES = (0.190, 0.320, 0.450, 0.580, 0.710, 0.840)


@dataclass(frozen=True)
class SourceRecord:
    dataset: str
    sample_id: str
    signal_path: str
    force_blank: bool = False


@dataclass(frozen=True)
class RenderConfig:
    output_dir: str
    ptb_width: int
    ptb_height: int
    chapman_width: int
    chapman_height: int
    baselines: tuple[float, ...]
    rhythm_leads: tuple[str, ...]
    x_start: float
    x_end: float
    seconds: float
    chapman_divisor: float
    signal_clip_mv: float
    jpeg_quality: int
    blank_page_probability: float
    blank_patch_probability: float
    moire_probability: float
    glare_probability: float
    barcode_probability: float
    barcode_seconds: float
    barcode_rows: int
    barcode_mask_policy: str
    unrectified_overlay_probability: float
    unrectified_overlay_files: tuple[str, ...]
    layout_profile: str
    seed: int
    overwrite: bool


def parse_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[xX,]\s*(\d+)\s*", value)
    if not match:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT")
    width, height = int(match.group(1)), int(match.group(2))
    if width < 256 or height < 256:
        raise argparse.ArgumentTypeError("image dimensions must be at least 256")
    return width, height


def parse_floats(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(x.strip()) for x in value.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def canonical_lead(name: str) -> str:
    upper = name.strip().upper()
    return {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}.get(upper, upper)


def stable_seed(seed: int, record: SourceRecord) -> int:
    digest = hashlib.sha1(
        f"{seed}:{record.dataset}:{record.sample_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def stable_fold(sample_id: str, folds: int = 5) -> int:
    digest = hashlib.sha1(sample_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % folds


def discover_records(args: argparse.Namespace) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    if args.datasets in ("both", "ptb-xl"):
        paths = sorted(args.ptb_root.rglob("*.hea"))
        if args.limit_ptb is not None:
            paths = paths[:args.limit_ptb]
        records += [
            SourceRecord("ptb_xl", f"ptbxl_{p.stem}", str(p)) for p in paths
        ]
    if args.datasets in ("both", "chapman"):
        paths = sorted(args.chapman_root.glob("*.csv"))
        if args.limit_chapman is not None:
            paths = paths[:args.limit_chapman]
        records += [
            SourceRecord("chapman", f"chapman_{p.stem}", str(p)) for p in paths
        ]
    # Blank pages are extra hard negatives; never replace waveform records.
    signal_records = list(records)
    blank_count = int(round(len(signal_records) * args.blank_page_probability))
    if blank_count:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(
            len(signal_records), size=blank_count,
            replace=blank_count > len(signal_records),
        )
        for ordinal, index in enumerate(indices):
            source = signal_records[int(index)]
            records.append(SourceRecord(
                source.dataset,
                f"{source.sample_id}__blank_{ordinal:05d}",
                source.signal_path,
                True,
            ))
    return records


def _parse_gain(token: str) -> tuple[float, float | None]:
    match = re.match(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\(([+-]?\d+)\))?", token
    )
    if not match:
        raise ValueError(f"cannot parse WFDB gain token {token!r}")
    gain = float(match.group(1))
    baseline = float(match.group(2)) if match.group(2) is not None else None
    if not math.isfinite(gain) or gain == 0:
        raise ValueError(f"invalid WFDB gain {gain}")
    return gain, baseline


def load_ptb_record(header_path: Path) -> tuple[dict[str, np.ndarray], float]:
    lines = [
        line.strip()
        for line in header_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if line.strip()
    ]
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError("malformed WFDB first header line")
    channels, sample_rate, samples = int(first[1]), float(first[2]), int(first[3])
    if channels != 12:
        raise ValueError(f"expected 12 PTB-XL channels, got {channels}")
    if len(lines) < channels + 1:
        raise ValueError("WFDB header has too few signal specification lines")

    specs: list[tuple[str, float, float]] = []
    data_names: set[str] = set()
    for line in lines[1:channels + 1]:
        tokens = line.split()
        if len(tokens) < 9 or tokens[1] != "16":
            raise ValueError("only interleaved WFDB format 16 is supported")
        gain, baseline = _parse_gain(tokens[2])
        baseline = float(tokens[4]) if baseline is None else baseline
        specs.append((canonical_lead(tokens[-1]), gain, baseline))
        data_names.add(tokens[0])
    if len(data_names) != 1:
        raise ValueError("multi-file WFDB records are not supported")

    raw = np.fromfile(header_path.with_name(next(iter(data_names))), dtype="<i2")
    expected = samples * channels
    if raw.size != expected:
        raise ValueError(f"WFDB data has {raw.size} values; expected {expected}")
    raw = raw.reshape(samples, channels)
    signals: dict[str, np.ndarray] = {}
    for channel, (lead, gain, baseline) in enumerate(specs):
        digital = raw[:, channel].astype(np.float32)
        missing = digital == -32768
        if missing.any():
            valid = ~missing
            if not valid.any():
                raise ValueError(f"lead {lead} consists only of missing values")
            digital[missing] = float(np.median(digital[valid]))
        signals[lead] = (digital - baseline) / gain
    return signals, sample_rate


def load_chapman_record(
    csv_path: Path, divisor: float
) -> tuple[dict[str, np.ndarray], float]:
    values = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    if values.shape == (12, 5000):
        values = values.T
    if values.ndim != 2 or values.shape[1] != 12:
        raise ValueError(f"Chapman array must be Nx12, got {values.shape}")
    for channel in range(12):
        column = values[:, channel]
        finite = np.isfinite(column)
        if not finite.any():
            raise ValueError(f"lead {LEAD_NAMES[channel]} has no finite values")
        column[~finite] = float(np.median(column[finite]))
    values /= divisor
    return {lead: values[:, i] for i, lead in enumerate(LEAD_NAMES)}, 500.0


def centered(values: np.ndarray, clip_mv: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("signal has no finite samples")
    median = float(np.median(values[finite]))
    clean = np.nan_to_num(values, nan=median, posinf=median, neginf=median) - median
    return np.clip(clean, -clip_mv, clip_mv)


def resample(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) == count:
        return values.astype(np.float32, copy=False)
    old_x = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, count, dtype=np.float32)
    return np.interp(new_x, old_x, values).astype(np.float32)


def make_row_signals(
    signals: dict[str, np.ndarray],
    sample_rate: float,
    seconds: float,
    rhythm_leads: tuple[str, ...],
    clip_mv: float,
) -> tuple[list[np.ndarray], list[str]]:
    required = {lead for row in TOP_LAYOUT for lead in row} | set(rhythm_leads)
    missing = sorted(required - set(signals))
    if missing:
        raise ValueError(f"missing leads: {', '.join(missing)}")
    count = int(round(sample_rate * seconds))
    if min(len(signals[lead]) for lead in required) < count:
        raise ValueError(f"one or more leads are shorter than {seconds:g} seconds")
    edges = np.rint(np.linspace(0, count, 5)).astype(int)
    rows: list[np.ndarray] = []
    names: list[str] = []
    for layout_row in TOP_LAYOUT:
        parts = [
            centered(signals[lead][edges[i]:edges[i + 1]], clip_mv)
            for i, lead in enumerate(layout_row)
        ]
        rows.append(np.concatenate(parts))
        names.append("|".join(layout_row))
    for lead in rhythm_leads:
        rows.append(centered(signals[lead][:count], clip_mv))
        names.append(lead)
    return rows, names


def choose_paper_style(dataset: str, rng: np.random.Generator) -> str:
    if dataset == "ptb_xl":
        probabilities = (0.40, 0.25, 0.25, 0.10)
    else:
        probabilities = (0.50, 0.25, 0.15, 0.10)
    return str(rng.choice(("pink", "red", "white", "dotted"), p=probabilities))


def draw_paper(
    height: int, width: int, pixels_per_mm: float, style: str
) -> np.ndarray:
    if style == "pink":
        # OpenCV colors are BGR, not RGB.
        base, minor, major = (253, 253, 255), (220, 214, 255), (160, 143, 244)
    elif style == "red":
        base, minor, major = (249, 249, 255), (201, 198, 252), (112, 105, 227)
    elif style == "white":
        return np.full((height, width, 3), 255, dtype=np.uint8)
    else:
        canvas = np.full((height, width, 3), 252, dtype=np.uint8)
        step = max(3, int(round(pixels_per_mm)))
        for y in range(0, height, step):
            xs = np.arange(0, width, step, dtype=np.int32)
            canvas[y, xs] = (175, 175, 175)
        return canvas

    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = base
    minor_step = max(3, int(round(pixels_per_mm)))
    major_step = max(minor_step, int(round(5 * pixels_per_mm)))
    for x in range(0, width, minor_step):
        cv2.line(canvas, (x, 0), (x, height - 1), minor, 1, cv2.LINE_AA)
    for y in range(0, height, minor_step):
        cv2.line(canvas, (0, y), (width - 1, y), minor, 1, cv2.LINE_AA)
    for x in range(0, width, major_step):
        cv2.line(canvas, (x, 0), (x, height - 1), major, 1, cv2.LINE_AA)
    for y in range(0, height, major_step):
        cv2.line(canvas, (0, y), (width - 1, y), major, 1, cv2.LINE_AA)
    return canvas


def draw_calibration(
    image: np.ndarray, x0: int, baseline: float,
    pixels_per_mm: float, thickness: int, first_trace_y: int,
) -> None:
    """Draw the reference pulse and join it to the first waveform sample.

    The relative x positions follow ecg_data/sigxml2img.py: -8.5, -7.5,
    -7.5, -2.5, -2.5, 0 and 0 mm. The 2.5-mm zero-voltage tail
    after the pulse is intentionally retained.
    """
    pulse_height = int(round(10 * pixels_per_mm))
    y = int(round(baseline))
    left = max(2, x0 - int(round(8.5 * pixels_per_mm)))
    pulse_start = x0 - int(round(7.5 * pixels_per_mm))
    pulse_end = x0 - int(round(2.5 * pixels_per_mm))
    points = np.array([
        [left, y], [pulse_start, y], [pulse_start, y - pulse_height],
        [pulse_end, y - pulse_height], [pulse_end, y],
        [x0, y], [x0, first_trace_y],
    ], dtype=np.int32)
    cv2.polylines(image, [points], False, (28, 28, 28), thickness, cv2.LINE_AA)


def sparse_fractional_trace(
    signal: np.ndarray,
    width: int,
    height: int,
    x0: int,
    x1: int,
    baseline: float,
    pixels_per_mv: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.arange(x0, x1, dtype=np.int32)
    values = resample(signal, len(xs))
    continuous_y = np.clip(
        baseline - values * pixels_per_mv, 0.0, height - 1.0001
    )
    lower = np.floor(continuous_y).astype(np.int32)
    fraction = (continuous_y - lower).astype(np.float32)
    upper = np.minimum(lower + 1, height - 1)
    use_upper = (fraction > 1e-6) & (upper != lower)
    mask_x = np.concatenate((xs, xs[use_upper]))
    mask_y = np.concatenate((lower, upper[use_upper]))
    mask_v = np.concatenate((1.0 - fraction, fraction[use_upper])).astype(np.float32)
    image_points = np.column_stack((xs, np.rint(continuous_y).astype(np.int32)))
    return mask_x, mask_y, mask_v, image_points


def save_sparse_mask(
    path: Path,
    shape: tuple[int, int, int],
    coordinates: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    payload: dict[str, np.ndarray] = {"shape": np.asarray(shape, dtype=np.int32)}
    for channel, (xs, ys, values) in enumerate(coordinates):
        payload[f"ch{channel}_x"] = xs.astype(np.int32, copy=False)
        payload[f"ch{channel}_y"] = ys.astype(np.int32, copy=False)
        payload[f"ch{channel}_v"] = values.astype(np.float32, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def add_moire(image: np.ndarray, rng: np.random.Generator) -> None:
    height, width = image.shape[:2]
    x = np.arange(width, dtype=np.float32)[None, :]
    y = np.arange(height, dtype=np.float32)[:, None]
    pattern = np.sin(x * rng.uniform(0.025, 0.08) + y * rng.uniform(0.01, 0.04))
    delta = pattern * rng.uniform(3.0, 10.0)
    cv2.add(
        image, np.repeat(delta[:, :, None], 3, axis=2), dst=image,
        dtype=cv2.CV_8U,
    )


def add_glare(image: np.ndarray, rng: np.random.Generator) -> None:
    height, width = image.shape[:2]
    cx, cy = rng.integers(0, width), rng.integers(0, height)
    radius = max(height, width) * rng.uniform(0.10, 0.40)
    x = np.arange(width, dtype=np.float32)[None, :]
    y = np.arange(height, dtype=np.float32)[:, None]
    flare = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * radius**2))
    delta = (flare * rng.uniform(25.0, 85.0)).astype(np.float32)
    cv2.add(
        image, np.repeat(delta[:, :, None], 3, axis=2), dst=image,
        dtype=cv2.CV_8U,
    )


def add_blank_patch(image: np.ndarray, rng: np.random.Generator) -> list[int]:
    height, width = image.shape[:2]
    patch_w = int(rng.uniform(0.04, 0.20) * width)
    patch_h = int(rng.uniform(0.05, 0.22) * height)
    x0 = int(rng.integers(0, max(1, width - patch_w)))
    y0 = int(rng.integers(0, max(1, height - patch_h)))
    image[y0:y0 + patch_h, x0:x0 + patch_w] = int(rng.integers(245, 256))
    return [x0, y0, x0 + patch_w, y0 + patch_h]


def add_unrectified_overlay(
    image: np.ndarray, overlay_path: str, rng: np.random.Generator
) -> None:
    overlay = cv2.imread(overlay_path, cv2.IMREAD_COLOR)
    if overlay is None:
        raise ValueError(f"cannot read overlay image {overlay_path}")
    height, width = image.shape[:2]
    overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_AREA)
    matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), float(rng.uniform(-4.0, 4.0)), 1.0
    )
    overlay = cv2.warpAffine(
        overlay, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    alpha = float(rng.uniform(0.035, 0.12))
    cv2.addWeighted(image, 1.0 - alpha, overlay, alpha, 0.0, dst=image)


def add_right_barcode(
    image: np.ndarray,
    rng: np.random.Generator,
    x0: int,
    x1: int,
    baselines: Sequence[float],
    seconds: float,
    barcode_seconds: float,
    barcode_rows: int,
) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    barcode_width = max(12, int(round((x1 - x0) * barcode_seconds / seconds)))
    right = min(width - 2, x1 + max(0, int(round(0.006 * width))))
    left = max(x0, right - barcode_width)
    covered_rows = min(barcode_rows, len(baselines))
    row_gap = float(np.median(np.diff(baselines)))
    top = max(0, int(round(baselines[0] - 0.60 * row_gap)))
    bottom = min(
        height, int(round(baselines[covered_rows - 1] + 0.60 * row_gap))
    )
    cv2.rectangle(
        image, (left - 3, top - 3), (right + 3, bottom + 3),
        (255, 255, 255), -1,
    )
    cursor = left
    while cursor < right:
        bar = int(rng.integers(1, max(2, barcode_width // 13 + 1)))
        gap = int(rng.integers(1, max(2, barcode_width // 18 + 1)))
        if rng.random() < 0.72:
            cv2.rectangle(
                image, (cursor, top), (min(cursor + bar, right), bottom),
                (12, 12, 12), -1,
            )
        cursor += bar + gap
    cv2.rectangle(
        image, (left - 2, top - 2), (right + 2, bottom + 2),
        (60, 60, 60), 1,
    )
    return left, top, right, bottom


def empty_coordinates(
    rows: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [
        (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )
        for _ in range(rows)
    ]


def render_record(record: SourceRecord, config: RenderConfig) -> dict[str, object]:
    rng = np.random.default_rng(stable_seed(config.seed, record))
    if record.dataset == "ptb_xl":
        width, height = config.ptb_width, config.ptb_height
    else:
        width, height = config.chapman_width, config.chapman_height
    root = Path(config.output_dir)
    image_path = root / "images" / record.dataset / f"{record.sample_id}.jpg"
    mask_path = root / "masks" / record.dataset / f"{record.sample_id}.mask-coo.npz"
    if image_path.is_file() and mask_path.is_file() and not config.overwrite:
        return {
            "status": "existing", "sample_id": record.sample_id,
            "dataset": record.dataset,
            "image_path": str(image_path.relative_to(root)),
            "mask_path": str(mask_path.relative_to(root)),
            "fold": stable_fold(record.sample_id), "width": width, "height": height,
            "is_blank": "", "paper_style": "", "barcode": "",
            "barcode_mask_policy": config.barcode_mask_policy,
        }

    if record.force_blank:
        rows = [np.empty(0, dtype=np.float32) for _ in config.baselines]
        row_names = ["|".join(row) for row in TOP_LAYOUT] + list(config.rhythm_leads)
    else:
        if record.dataset == "ptb_xl":
            signals, sample_rate = load_ptb_record(Path(record.signal_path))
        else:
            signals, sample_rate = load_chapman_record(
                Path(record.signal_path), config.chapman_divisor
            )
        if abs(sample_rate - 500.0) > 1e-6:
            raise ValueError(f"expected 500 Hz, got {sample_rate}")
        rows, row_names = make_row_signals(
            signals, sample_rate, config.seconds, config.rhythm_leads,
            config.signal_clip_mv,
        )
    if len(rows) != len(config.baselines):
        raise ValueError(
            f"{len(rows)} rows require {len(rows)} baselines; "
            f"got {len(config.baselines)}"
        )

    style = choose_paper_style(record.dataset, rng)
    is_blank = record.force_blank
    if config.layout_profile == "private_strict":
        profile = page_renderer.profile_for_size(width, height)
    else:
        profile = page_renderer.custom_profile(
            width, height, config.x_start, config.x_end, config.baselines
        )
    rendered = page_renderer.render_ecg_page(
        rows,
        row_names,
        profile,
        style,
        is_blank=is_blank,
        amplitude_scale=(
            1.0 if is_blank else float(rng.uniform(0.90, 1.10))
        ),
    )
    image = rendered.image
    coordinates = rendered.coordinates
    x0, x1 = rendered.x0, rendered.x1
    baselines = rendered.baselines

    augmentations: list[str] = []
    if (
        config.unrectified_overlay_files
        and rng.random() < config.unrectified_overlay_probability
    ):
        add_unrectified_overlay(
            image, str(rng.choice(config.unrectified_overlay_files)), rng
        )
        augmentations.append("unrectified_overlay")
    if rng.random() < config.moire_probability:
        add_moire(image, rng)
        augmentations.append("moire")
    if rng.random() < config.glare_probability:
        add_glare(image, rng)
        augmentations.append("glare")
    blank_patch = ""
    if not is_blank and rng.random() < config.blank_patch_probability:
        blank_patch = json.dumps(add_blank_patch(image, rng), separators=(",", ":"))
        augmentations.append("blank_patch")

    barcode_box: tuple[int, int, int, int] | None = None
    if (
        not is_blank
        and config.barcode_rows > 0
        and rng.random() < config.barcode_probability
    ):
        barcode_box = add_right_barcode(
            image, rng, x0, x1, baselines, config.seconds,
            config.barcode_seconds, config.barcode_rows,
        )
        augmentations.append("right_barcode")
        if config.barcode_mask_policy == "drop":
            barcode_left = barcode_box[0]
            for channel in range(min(config.barcode_rows, len(coordinates))):
                xs, ys, values = coordinates[channel]
                keep = xs < barcode_left
                coordinates[channel] = xs[keep], ys[keep], values[keep]

    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
    ):
        raise OSError(f"failed to write {image_path}")
    save_sparse_mask(mask_path, (len(rows), height, width), coordinates)
    return {
        "status": "generated", "sample_id": record.sample_id,
        "dataset": record.dataset,
        "image_path": str(image_path.relative_to(root)),
        "mask_path": str(mask_path.relative_to(root)),
        "fold": stable_fold(record.sample_id), "width": width, "height": height,
        "is_blank": int(is_blank), "paper_style": style,
        "barcode": "" if barcode_box is None else json.dumps(
            barcode_box, separators=(",", ":")
        ),
        "barcode_mask_policy": config.barcode_mask_policy,
        "blank_patch": blank_patch, "augmentations": ";".join(augmentations),
        "row_names": ";".join(row_names),
        "profile": rendered.profile.name,
        "baselines": ";".join(
            f"{value:.6g}" for value in rendered.profile.baselines
        ),
        "x_start": rendered.profile.x_start,
        "x_end": rendered.profile.x_end,
    }


def worker(payload: tuple[SourceRecord, RenderConfig]) -> dict[str, object]:
    record, config = payload
    try:
        return render_record(record, config)
    except Exception as exc:
        return {
            "status": "failed", "sample_id": record.sample_id,
            "dataset": record.dataset, "signal_path": record.signal_path,
            "error": f"{type(exc).__name__}: {exc}",
        }


def iter_results(
    records: list[SourceRecord], config: RenderConfig, workers: int
) -> Iterable[dict[str, object]]:
    payloads = ((record, config) for record in records)
    if workers == 1:
        for payload in payloads:
            yield worker(payload)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(worker, payloads, chunksize=1)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    preferred = [
        "sample_id", "dataset", "image_path", "mask_path", "fold", "width",
        "height", "is_blank", "paper_style", "barcode", "barcode_mask_policy",
        "blank_patch", "augmentations", "row_names", "profile", "baselines", "x_start", "x_end",
    ]
    extra = {key for row in rows for key in row} - set(preferred) - {"status"}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=preferred + sorted(extra), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def find_overlay_files(directory: Path | None) -> tuple[str, ...]:
    if directory is None:
        return ()
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return tuple(
        str(path) for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in suffixes
    )


def build_parser() -> argparse.ArgumentParser:
    data_root = Path("data/sources")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ptb-root", type=Path, default=data_root / "ptb_xl")
    parser.add_argument(
        "--chapman-root", type=Path, default=data_root / "chapman_ECG"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets", choices=("both", "ptb-xl", "chapman"), default="both"
    )
    parser.add_argument(
        "--ptb-size", type=parse_size, default=(3508, 2480), metavar="WIDTHxHEIGHT"
    )
    parser.add_argument(
        "--chapman-size", type=parse_size, default=(3320, 2552),
        metavar="WIDTHxHEIGHT",
    )
    parser.add_argument("--rhythm-leads", nargs="+", default=DEFAULT_RHYTHM_LEADS)
    parser.add_argument(
        "--baselines", type=parse_floats, default=DEFAULT_BASELINES,
        help="custom profile only: normalized y positions, one per output row",
    )
    parser.add_argument("--x-start", type=float, default=0.070)
    parser.add_argument("--x-end", type=float, default=0.972)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument(
        "--chapman-divisor", type=float, default=1000.0,
        help="convert Chapman microvolt-like values to mV",
    )
    parser.add_argument("--signal-clip-mv", type=float, default=5.0)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument(
        "--blank-page-ratio", "--blank-page-probability",
        dest="blank_page_probability", type=float, default=0.01,
        help="additional empty pages divided by signal pages (default: 0.01)",
    )
    parser.add_argument("--blank-patch-probability", type=float, default=0.20)
    parser.add_argument("--moire-probability", type=float, default=0.25)
    parser.add_argument("--glare-probability", type=float, default=0.20)
    parser.add_argument("--barcode-probability", type=float, default=0.35)
    parser.add_argument("--barcode-seconds", type=float, default=0.20)
    parser.add_argument(
        "--barcode-rows", type=int, default=5,
        help="number of top rows vertically covered by the right barcode",
    )
    parser.add_argument(
        "--barcode-mask-policy", choices=("keep", "drop"), default="keep"
    )
    parser.add_argument("--unrectified-overlay-dir", type=Path)
    parser.add_argument(
        "--unrectified-overlay-probability", type=float, default=0.15
    )
    parser.add_argument(
        "--layout-profile", choices=("private_strict", "custom"),
        default="private_strict",
        help="private strict profiles by target size, or legacy CLI geometry",
    )
    parser.add_argument("--limit-ptb", type=int)
    parser.add_argument("--limit-chapman", type=int)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="each high-resolution worker needs substantial RAM",
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.rhythm_leads = tuple(canonical_lead(x) for x in args.rhythm_leads)
    expected_rows = len(TOP_LAYOUT) + len(args.rhythm_leads)
    if len(args.baselines) != expected_rows:
        parser.error(f"need {expected_rows} baselines for {expected_rows} output rows")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if not 0 <= args.barcode_rows <= expected_rows:
        parser.error(f"--barcode-rows must be in [0, {expected_rows}]")
    if args.barcode_seconds <= 0 or args.barcode_seconds >= args.seconds:
        parser.error("--barcode-seconds must be positive and shorter than --seconds")
    probability_names = (
        "blank_page_probability", "blank_patch_probability", "moire_probability",
        "glare_probability", "barcode_probability",
        "unrectified_overlay_probability",
    )
    for name in probability_names:
        value = getattr(args, name)
        if not 0 <= value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")

    records = discover_records(args)
    if not records:
        parser.error("no waveform records found")
    overlay_files = find_overlay_files(args.unrectified_overlay_dir)
    if args.unrectified_overlay_dir and not overlay_files:
        parser.error(f"no images found below {args.unrectified_overlay_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = RenderConfig(
        output_dir=str(args.output_dir.resolve()),
        ptb_width=args.ptb_size[0], ptb_height=args.ptb_size[1],
        chapman_width=args.chapman_size[0], chapman_height=args.chapman_size[1],
        baselines=tuple(args.baselines), rhythm_leads=args.rhythm_leads,
        x_start=args.x_start, x_end=args.x_end, seconds=args.seconds,
        chapman_divisor=args.chapman_divisor,
        signal_clip_mv=args.signal_clip_mv, jpeg_quality=args.jpeg_quality,
        blank_page_probability=args.blank_page_probability,
        blank_patch_probability=args.blank_patch_probability,
        moire_probability=args.moire_probability,
        glare_probability=args.glare_probability,
        barcode_probability=args.barcode_probability,
        barcode_seconds=args.barcode_seconds, barcode_rows=args.barcode_rows,
        barcode_mask_policy=args.barcode_mask_policy,
        unrectified_overlay_probability=args.unrectified_overlay_probability,
        unrectified_overlay_files=overlay_files,
        layout_profile=args.layout_profile,
        seed=args.seed, overwrite=args.overwrite,
    )

    print(f"Discovered {len(records)} records; output={config.output_dir}")
    print(
        f"PTB-XL={config.ptb_width}x{config.ptb_height}, "
        f"Chapman={config.chapman_width}x{config.chapman_height}, "
        f"rows={expected_rows}"
    )
    successful: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for index, result in enumerate(
        iter_results(records, config, args.workers), start=1
    ):
        if result["status"] == "failed":
            failures.append(result)
            print(
                f"ERROR {result['sample_id']}: {result['error']}", file=sys.stderr
            )
        else:
            successful.append(result)
        if index == 1 or index % 100 == 0 or index == len(records):
            print(
                f"[{index}/{len(records)}] usable={len(successful)} "
                f"failed={len(failures)}"
            )

    manifest_path = args.output_dir / "manifest.csv"
    write_manifest(manifest_path, successful)
    report = {
        "requested": len(records),
        "usable": len(successful),
        "failed": len(failures),
        "by_dataset": {
            name: sum(row.get("dataset") == name for row in successful)
            for name in ("ptb_xl", "chapman")
        },
        "config": asdict(config) | {
            "unrectified_overlay_files": len(overlay_files)
        },
        "failures": failures,
    }
    (args.output_dir / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Manifest: {manifest_path}")
    print(f"Finished: {len(successful)} usable, {len(failures)} failed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
