#!/usr/bin/env python3
"""Generate paired ECG page PNGs and Open-ECG-compatible RGB masks.

Output pairs are split/xxx.png and split/xxx_mask.png. Mask channels are
R=paper grid, G=text/control/nuisance, B=physiological trace. All-zero pixels
are background. No img_T0 intermediate is needed.

Every PTB-XL and Chapman record is rendered in all three layouts; each page
draws one canvas size uniformly from that layout's pool, so the sizes end up
roughly balanced:

  3x4_plus_3  3x4 segments + V1/II/V5 rhythm  3508x2480 / 3320x2552
  6x2_plus_1  6x2 segments + II rhythm        3508x2480 / 1667x1261
  3x4_plus_1  3x4 segments + II rhythm        3508x2480 / 1667x1261

Where two leads meet inside a printed row the pen either runs straight through
(``continuous``), lifts for a couple of millimetres (``gap``), or prints a 1 mV
separator pulse (``pulse``), drawn per page at ``--transition-weights``
(default 0.20/0.40/0.40). The separator pulse is a control mark, so it is
supervised in the green channel rather than the blue trace channel.

Splits are assigned per source recording, so all layouts, sizes and separation
styles of one record land in the same split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__:
    from . import ecg_page_rendering as page_renderer
    from . import generate_synthetic_3x4_rhythm_dataset as legacy
else:
    import ecg_page_rendering as page_renderer
    import generate_synthetic_3x4_rhythm_dataset as legacy


LAYOUT_4X3 = "4x3_plus_3"
LAYOUT_6X2 = "6x2_plus_1"
LAYOUT_4X3_1 = "4x3_plus_1"
LAYOUTS = (LAYOUT_4X3, LAYOUT_6X2, LAYOUT_4X3_1)
FOUR_BY_THREE_ROWS = (
    ("I", "aVR", "V1", "V4"),
    ("II", "aVL", "V2", "V5"),
    ("III", "aVF", "V3", "V6"),
)
SIX_BY_TWO_ROWS = (
    ("I", "V1"), ("II", "V2"), ("III", "V3"),
    ("aVR", "V4"), ("aVL", "V5"), ("aVF", "V6"),
)
# Geometry shared by every page of a layout; only the canvas size varies.
# ``baseline_cells`` is empty for LAYOUT_4X3 because that layout keeps the
# private-scan calibrated baselines from ``page_renderer.profile_for_size``.
LAYOUT_TABLE = {
    LAYOUT_4X3: {
        "segment_rows": FOUR_BY_THREE_ROWS,
        "rhythm_leads": ("V1", "II", "V5"),
        "grid_rows": 39,
        "grid_columns": 56,
        "baseline_cells": (),
        "footer_text": "4 x 2.5s + 3 rhythm leads",
    },
    LAYOUT_6X2: {
        "segment_rows": SIX_BY_TWO_ROWS,
        "rhythm_leads": ("II",),
        "grid_rows": 30,
        "grid_columns": 54,
        "baseline_cells": (3.5, 7.5, 11.5, 15.5, 19.5, 23.5, 27.5),
        "footer_text": "6 x 5.0s + 1 rhythm lead",
    },
    LAYOUT_4X3_1: {
        "segment_rows": FOUR_BY_THREE_ROWS,
        "rhythm_leads": ("II",),
        "grid_rows": 36,
        "grid_columns": 56,
        "baseline_cells": (6.0, 14.0, 22.0, 30.0),
        "footer_text": "4 x 2.5s + 1 rhythm lead",
    },
}
DEFAULT_SIZES = {
    LAYOUT_4X3: ((3508, 2480), (3320, 2552)),
    LAYOUT_6X2: ((3508, 2480), (1667, 1261)),
    LAYOUT_4X3_1: ((3508, 2480), (1667, 1261)),
}
# Lead-separation styles, in the order --transition-weights expects them.
TRANSITIONS = page_renderer.TRANSITIONS
DEFAULT_TRANSITION_WEIGHTS = (0.20, 0.40, 0.40)
DEFAULT_CHINESE_FONTS = (
    "/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)
DEFAULT_LATIN_FONT = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"


@dataclass(frozen=True)
class LayoutSpec:
    name: str
    segment_rows: tuple[tuple[str, ...], ...]
    rhythm_leads: tuple[str, ...]
    grid_rows: int
    grid_columns: int
    baseline_cells: tuple[float, ...]
    footer_text: str
    size: tuple[int, int]


@dataclass(frozen=True)
class GenerationConfig:
    output_dir: str
    # ((layout name, ((width, height), ...)), ...); one size is drawn per job.
    layout_sizes: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]
    transition_weights: tuple[float, float, float]
    seconds: float
    chapman_divisor: float
    signal_clip_mv: float
    png_compression: int
    amplitude_min: float
    amplitude_max: float
    chinese_text_probability: float
    chinese_font: str
    perspective_probability: float
    perspective_distortion: float
    barcode_probability: float
    blank_patch_probability: float
    moire_probability: float
    glare_probability: float
    unrectified_overlay_probability: float
    unrectified_overlay_files: tuple[str, ...]
    seed: int
    overwrite: bool


@dataclass(frozen=True)
class Job:
    record: legacy.SourceRecord
    layout: str
    variant: int
    split: str
    size: tuple[int, int]
    transition: str


def stable_u64(text: str) -> int:
    return int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], "little")


def source_group(record: legacy.SourceRecord) -> str:
    stem = re.sub(r"_(?:hr|lr)$", "", Path(record.signal_path).stem, flags=re.I)
    return f"{record.dataset}:{re.sub(r'__blank_\d+$', '', stem)}"


def choose_split(
    record: legacy.SourceRecord, ratios: tuple[float, float, float], seed: int
) -> str:
    value = stable_u64(f"split:{seed}:{source_group(record)}") / float(2**64)
    if value < ratios[0]:
        return "train"
    if value < ratios[0] + ratios[1]:
        return "val"
    return "test"


def job_seed(job: Job, seed: int) -> int:
    key = (
        f"render:{seed}:{job.record.dataset}:{job.record.sample_id}:"
        f"{job.layout}:{job.variant}"
    )
    return stable_u64(key)


def choose_size(
    record: legacy.SourceRecord,
    layout: str,
    variant: int,
    sizes: Sequence[tuple[int, int]],
    seed: int,
) -> tuple[int, int]:
    """Draw one canvas size uniformly; the hash keeps runs reproducible."""
    key = f"size:{seed}:{record.dataset}:{record.sample_id}:{layout}:{variant}"
    return tuple(sizes[stable_u64(key) % len(sizes)])


def choose_transition(
    record: legacy.SourceRecord,
    layout: str,
    variant: int,
    weights: Sequence[float],
    seed: int,
) -> str:
    """Draw a lead-separation style; the hash keeps runs reproducible."""
    key = f"transition:{seed}:{record.dataset}:{record.sample_id}:{layout}:{variant}"
    value = stable_u64(key) / float(2**64) * float(sum(weights))
    cumulative = 0.0
    for name, weight in zip(TRANSITIONS, weights):
        cumulative += weight
        if value < cumulative:
            return name
    return TRANSITIONS[-1]


def parse_transition_weights(value: str) -> tuple[float, float, float]:
    try:
        parts = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if len(parts) != len(TRANSITIONS) or any(item < 0 for item in parts):
        raise argparse.ArgumentTypeError(
            f"expected {len(TRANSITIONS)} non-negative weights for "
            f"{','.join(TRANSITIONS)}"
        )
    if sum(parts) <= 0:
        raise argparse.ArgumentTypeError("transition weights must sum above zero")
    return parts


def parse_ratios(value: str) -> tuple[float, float, float]:
    try:
        parts = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if len(parts) != 3 or any(item < 0 for item in parts) or sum(parts) <= 0:
        raise argparse.ArgumentTypeError(
            "ratios must be three non-negative values with a positive sum"
        )
    total = sum(parts)
    return (parts[0] / total, parts[1] / total, parts[2] / total)


def probability(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("probability must be in [0, 1]")
    return number


def resolve_chinese_font(requested: Path | None) -> str:
    if requested is not None:
        if not requested.is_file():
            raise FileNotFoundError(f"Chinese font not found: {requested}")
        return str(requested)
    for candidate in DEFAULT_CHINESE_FONTS:
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(
        "No CJK font found; pass --chinese-font or "
        "--chinese-text-probability 0"
    )


def discover_records(args: argparse.Namespace) -> list[legacy.SourceRecord]:
    records: list[legacy.SourceRecord] = []
    if args.datasets in ("both", "ptb-xl"):
        preferred = {
            "high": args.ptb_root / "records500",
            "low": args.ptb_root / "records100",
        }
        if args.ptb_resolution == "all":
            paths = sorted(args.ptb_root.rglob("*.hea"))
        else:
            root = preferred[args.ptb_resolution]
            paths = sorted(root.rglob("*.hea")) if root.is_dir() else []
            if not paths:
                paths = sorted(args.ptb_root.rglob("*.hea"))
        paths = paths if args.limit_ptb is None else paths[:args.limit_ptb]
        records.extend(
            legacy.SourceRecord("ptb_xl", f"ptbxl_{path.stem}", str(path))
            for path in paths
        )
    if args.datasets in ("both", "chapman"):
        paths = sorted(args.chapman_root.glob("*.csv"))
        paths = paths if args.limit_chapman is None else paths[:args.limit_chapman]
        records.extend(
            legacy.SourceRecord("chapman", f"chapman_{path.stem}", str(path))
            for path in paths
        )
    if args.blank_page_ratio > 0 and records:
        count = int(round(len(records) * args.blank_page_ratio))
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(records), size=count, replace=count > len(records))
        for ordinal, index in enumerate(indices):
            source = records[int(index)]
            records.append(legacy.SourceRecord(
                source.dataset,
                f"{source.sample_id}__blank_{ordinal:05d}",
                source.signal_path,
                True,
            ))
    return records


def sizes_for(config: GenerationConfig, layout: str) -> tuple[tuple[int, int], ...]:
    for name, sizes in config.layout_sizes:
        if name == layout:
            return sizes
    raise ValueError(f"no sizes configured for layout {layout!r}")


def layout_spec(name: str, size: tuple[int, int]) -> LayoutSpec:
    try:
        entry = LAYOUT_TABLE[name]
    except KeyError as exc:
        raise ValueError(f"unknown layout {name!r}") from exc
    return LayoutSpec(name=name, size=tuple(size), **entry)


def square_grid_profile(spec: LayoutSpec) -> page_renderer.PageProfile:
    width, height = spec.size
    if not spec.baseline_cells:
        profile = page_renderer.profile_for_size(width, height)
        return page_renderer.PageProfile(**{
            **asdict(profile),
            "grid_rows": spec.grid_rows,
            "grid_columns": spec.grid_columns,
        })

    cell = min(0.96 * width / spec.grid_columns, 0.88 * height / spec.grid_rows)
    grid_width, grid_height = cell * spec.grid_columns, cell * spec.grid_rows
    left = (width - grid_width) / 2.0
    top = max(2.0, 0.20 * (height - grid_height))
    # 50 major cells span the 250 mm covered by 10 s at 25 mm/s.
    x_start, x_end = left + 2.0 * cell, left + 52.0 * cell
    baseline_cells = spec.baseline_cells
    return page_renderer.PageProfile(
        name=f"open_ecg_{spec.name}_{width}x{height}",
        width=width,
        height=height,
        x_start=x_start / width,
        x_end=x_end / width,
        baselines=tuple((top + value * cell) / height for value in baseline_cells),
        grid_left=left / width,
        grid_right=(left + grid_width) / width,
        grid_top=top / height,
        grid_bottom=(top + grid_height) / height,
        grid_rows=spec.grid_rows,
        grid_columns=spec.grid_columns,
        calibration_mm=(-8.5, -7.5, -7.5, -2.5, -2.5, 0.0, 0.0),
    )


def make_rows(
    signals: dict[str, np.ndarray],
    sample_rate: float,
    spec: LayoutSpec,
    seconds: float,
    clip_mv: float,
) -> tuple[list[list[np.ndarray]], list[str]]:
    """Return each printed row as its separate lead segments.

    Keeping the parts apart is what lets the renderer break the pen or print a
    separator pulse at a column boundary instead of running straight through.
    """
    required = {lead for row in spec.segment_rows for lead in row}
    required.update(spec.rhythm_leads)
    missing = sorted(required - set(signals))
    if missing:
        raise ValueError(f"missing leads: {', '.join(missing)}")
    count = int(round(sample_rate * seconds))
    if min(len(signals[lead]) for lead in required) < count:
        raise ValueError(f"one or more leads are shorter than {seconds:g} seconds")
    rows: list[list[np.ndarray]] = []
    names: list[str] = []
    for display_row in spec.segment_rows:
        edges = np.rint(np.linspace(0, count, len(display_row) + 1)).astype(int)
        rows.append([
            legacy.centered(signals[lead][edges[i]:edges[i + 1]], clip_mv)
            for i, lead in enumerate(display_row)
        ])
        names.append("|".join(display_row))
    for lead in spec.rhythm_leads:
        rows.append([legacy.centered(signals[lead][:count], clip_mv)])
        names.append(lead)
    return rows, names


def random_chinese_lines(rng: np.random.Generator) -> list[str]:
    surnames = ("张", "李", "王", "赵", "陈", "刘", "杨", "黄")
    departments = ("心内科", "急诊科", "门诊", "体检中心", "老年医学科")
    findings = ("窦性心律", "心电图大致正常", "ST-T改变", "请结合临床")
    year, month, day = (
        int(rng.integers(2020, 2027)),
        int(rng.integers(1, 13)),
        int(rng.integers(1, 29)),
    )
    hour, minute = int(rng.integers(24)), int(rng.integers(60))
    return [
        str(rng.choice(("心电图报告", "十二导联心电图", "静息心电图检查报告"))),
        (
            f"姓名：{rng.choice(surnames)}某某  性别："
            f"{rng.choice(('男', '女'))}  年龄：{int(rng.integers(18, 91))}岁"
        ),
        (
            f"科室：{rng.choice(departments)}  检查号："
            f"{int(rng.integers(100000, 999999))}"
        ),
        f"检查时间：{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}",
        f"诊断提示：{rng.choice(findings)}    未确认",
    ]


def draw_mixed_text(
    image_draw: ImageDraw.ImageDraw,
    mask_draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    cjk_font: ImageFont.FreeTypeFont,
    latin_font: ImageFont.FreeTypeFont,
    image_value: int,
    target_value: int,
) -> None:
    """Draw ASCII with DejaVu and CJK with the supplied Chinese font."""
    cursor, y = float(position[0]), position[1]
    for character in text:
        font = latin_font if character.isascii() else cjk_font
        image_draw.text(
            (cursor, y), character, font=font,
            fill=(image_value, image_value, image_value),
        )
        mask_draw.text((cursor, y), character, font=font, fill=target_value)
        cursor += image_draw.textlength(character, font=font)


def draw_chinese_report_text(
    image: np.ndarray,
    text_mask: np.ndarray,
    profile: page_renderer.PageProfile,
    font_path: str,
    rng: np.random.Generator,
) -> list[str]:
    height, width = image.shape[:2]
    font_size = max(11, int(round(width * rng.uniform(0.0055, 0.0075))))
    title_font = ImageFont.truetype(font_path, int(round(font_size * 1.18)))
    body_font = ImageFont.truetype(font_path, font_size)
    latin_title_font = ImageFont.truetype(
        DEFAULT_LATIN_FONT, int(round(font_size * 1.18))
    )
    latin_body_font = ImageFont.truetype(DEFAULT_LATIN_FONT, font_size)
    lines = random_chinese_lines(rng)
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    target = Image.fromarray(text_mask)
    image_draw, mask_draw = ImageDraw.Draw(rgb), ImageDraw.Draw(target)
    dark, target_value = int(rng.integers(18, 80)), int(rng.integers(190, 256))
    grid_top = int(round(profile.grid_top * height))
    gap = max(font_size + 3, int(round(font_size * 1.28)))
    x0, y0 = (
        int(round(profile.grid_left * width)) + max(4, font_size // 2),
        max(3, grid_top + max(2, font_size // 4)),
    )
    placements = (
        (lines[0], int(0.67 * width), y0, title_font, latin_title_font),
        (lines[1], x0, y0, body_font, latin_body_font),
        (lines[2], x0, y0 + gap, body_font, latin_body_font),
        (lines[3], int(0.40 * width), y0 + gap, body_font, latin_body_font),
        (lines[4], int(0.67 * width), y0 + gap, body_font, latin_body_font),
    )
    for text, x, y, cjk_font, latin_font in placements:
        draw_mixed_text(
            image_draw, mask_draw, (x, y), text, cjk_font, latin_font,
            dark, target_value,
        )
    image[:] = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    text_mask[:] = np.asarray(target)
    return lines


def add_barcode(
    image: np.ndarray,
    layers: Sequence[np.ndarray],
    profile: page_renderer.PageProfile,
    rng: np.random.Generator,
) -> list[int]:
    height, width = image.shape[:2]
    x1 = int(round(profile.x_end * width))
    cell = (profile.grid_right - profile.grid_left) * width / profile.grid_columns
    barcode_width = max(12, int(round(rng.uniform(0.9, 1.8) * cell)))
    right = min(width - 3, x1 + int(round(1.4 * cell)))
    left = max(1, right - barcode_width)
    top = max(1, int(round(profile.baselines[0] * height - 0.55 * cell)))
    bottom_row = min(5, len(profile.baselines)) - 1
    bottom = min(
        height - 2,
        int(round(profile.baselines[bottom_row] * height + 0.55 * cell)),
    )
    image[top - 2:bottom + 3, left - 2:right + 3] = 255
    for layer in layers:
        layer[top - 2:bottom + 3, left - 2:right + 3] = 0
    cursor = left
    while cursor < right:
        bar = int(rng.integers(1, max(2, barcode_width // 11 + 1)))
        gap = int(rng.integers(1, max(2, barcode_width // 17 + 1)))
        if rng.random() < 0.78:
            cv2.rectangle(
                image, (cursor, top), (min(cursor + bar, right), bottom),
                (12, 12, 12), -1,
            )
            cv2.rectangle(
                layers[1], (cursor, top), (min(cursor + bar, right), bottom),
                255, -1,
            )
        cursor += bar + gap
    return [left, top, right, bottom]


def add_blank_patch(
    image: np.ndarray, layers: Sequence[np.ndarray], rng: np.random.Generator
) -> list[int]:
    height, width = image.shape[:2]
    patch_w, patch_h = (
        int(rng.uniform(0.04, 0.18) * width),
        int(rng.uniform(0.04, 0.18) * height),
    )
    x0 = int(rng.integers(0, max(1, width - patch_w)))
    y0 = int(rng.integers(0, max(1, height - patch_h)))
    image[y0:y0 + patch_h, x0:x0 + patch_w] = int(rng.integers(246, 256))
    for layer in layers:
        layer[y0:y0 + patch_h, x0:x0 + patch_w] = 0
    return [x0, y0, x0 + patch_w, y0 + patch_h]


def add_unrectified_overlay(
    image: np.ndarray,
    text_mask: np.ndarray,
    overlay_path: str,
    rng: np.random.Generator,
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
    darkness = 255 - cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    nuisance = np.rint(darkness.astype(np.float32) * min(1.0, 4.0 * alpha))
    np.maximum(text_mask, nuisance.astype(np.uint8), out=text_mask)


def perspective_warp(
    image: np.ndarray,
    layers: Sequence[np.ndarray],
    rng: np.random.Generator,
    distortion: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    source = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )
    jitter = np.column_stack((
        rng.uniform(-distortion * width, distortion * width, 4),
        rng.uniform(-distortion * height, distortion * height, 4),
    )).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(source, source + jitter)
    warped = cv2.warpPerspective(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(238, 238, 238),
    )
    for layer in layers:
        layer[:] = cv2.warpPerspective(
            layer, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    return warped


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample"


def render_job(payload: tuple[Job, GenerationConfig]) -> dict[str, object]:
    job, config = payload
    record, spec = job.record, layout_spec(job.layout, job.size)
    profile = square_grid_profile(spec)
    rng = np.random.default_rng(job_seed(job, config.seed))
    stem = safe_stem(
        f"{record.sample_id}__{job.layout}__{spec.size[0]}x{spec.size[1]}"
        f"__{job.transition}__v{job.variant:03d}"
    )
    split_dir = Path(config.output_dir) / job.split
    image_path, mask_path = (
        split_dir / f"{stem}.png",
        split_dir / f"{stem}_mask.png",
    )
    base = {
        "sample_id": stem,
        "source_sample_id": record.sample_id,
        "dataset": record.dataset,
        "split": job.split,
        "layout": job.layout,
        "variant": job.variant,
        "segment_transition": job.transition,
        "image_path": str(image_path.relative_to(config.output_dir)),
        "mask_path": str(mask_path.relative_to(config.output_dir)),
        "source_path": record.signal_path,
        "width": profile.width,
        "height": profile.height,
        "grid_rows": spec.grid_rows,
        "grid_columns": spec.grid_columns,
    }
    if not config.overwrite and image_path.is_file() and mask_path.is_file():
        return {**base, "status": "skipped"}
    try:
        if record.force_blank:
            rows = [[np.zeros(2, dtype=np.float32)] for _ in profile.baselines]
            row_names = [
                *("|".join(row) for row in spec.segment_rows),
                *spec.rhythm_leads,
            ]
        else:
            if record.dataset == "ptb_xl":
                signals, sample_rate = legacy.load_ptb_record(Path(record.signal_path))
            elif record.dataset == "chapman":
                signals, sample_rate = legacy.load_chapman_record(
                    Path(record.signal_path), config.chapman_divisor
                )
            else:
                raise ValueError(f"unknown dataset {record.dataset!r}")
            rows, row_names = make_rows(
                signals, sample_rate, spec, config.seconds, config.signal_clip_mv
            )
        style = legacy.choose_paper_style(record.dataset, rng)
        amplitude = float(rng.uniform(config.amplitude_min, config.amplitude_max))
        pixels_per_mm = (
            (profile.x_end - profile.x_start) * profile.width / 250.0
        )
        # One break width per page keeps a printout internally consistent.
        break_mm = (
            float(rng.uniform(1.2, 3.0)) if job.transition == "gap"
            else float(rng.uniform(3.6, 5.2))
        )
        rendered = page_renderer.render_ecg_page(
            rows, row_names, profile, style,
            transition=job.transition,
            break_px=max(2, int(round(break_mm * pixels_per_mm))),
            is_blank=record.force_blank, amplitude_scale=amplitude,
            footer_layout_text=spec.footer_text,
        )
        image = rendered.image
        layers = [rendered.grid_mask, rendered.text_mask, rendered.signal_mask]
        augmentations: list[str] = []
        chinese_lines: list[str] = []
        barcode: list[int] = []
        patch: list[int] = []
        if rng.random() < config.chinese_text_probability:
            chinese_lines = draw_chinese_report_text(
                image, layers[1], profile, config.chinese_font, rng
            )
            augmentations.append("chinese_report_text")
        if rng.random() < config.barcode_probability:
            barcode = add_barcode(image, layers, profile, rng)
            augmentations.append("barcode")
        if rng.random() < config.blank_patch_probability:
            patch = add_blank_patch(image, layers, rng)
            augmentations.append("blank_patch")
        if (
            config.unrectified_overlay_files
            and rng.random() < config.unrectified_overlay_probability
        ):
            overlay_path = str(rng.choice(config.unrectified_overlay_files))
            add_unrectified_overlay(image, layers[1], overlay_path, rng)
            augmentations.append("unrectified_overlay")
        if rng.random() < config.perspective_probability:
            image = perspective_warp(
                image, layers, rng, config.perspective_distortion
            )
            augmentations.append("perspective")
        if rng.random() < config.moire_probability:
            legacy.add_moire(image, rng)
            augmentations.append("moire")
        if rng.random() < config.glare_probability:
            legacy.add_glare(image, rng)
            augmentations.append("glare")

        mask_rgb = page_renderer.compose_rgb_mask(*layers)
        split_dir.mkdir(parents=True, exist_ok=True)
        options = [cv2.IMWRITE_PNG_COMPRESSION, config.png_compression]
        if not cv2.imwrite(str(image_path), image, options):
            raise OSError(f"failed to write {image_path}")
        if not cv2.imwrite(
            str(mask_path), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR), options
        ):
            raise OSError(f"failed to write {mask_path}")
        pixels = [
            int(np.count_nonzero(mask_rgb[:, :, channel])) for channel in range(3)
        ]
        return {
            **base,
            "status": "written",
            "profile": profile.name,
            "paper_style": style,
            "is_blank": int(record.force_blank),
            "row_names": "|".join(row_names),
            "amplitude_scale": amplitude,
            "augmentations": "|".join(augmentations),
            "chinese_lines": "|".join(chinese_lines),
            "barcode_box": json.dumps(barcode),
            "blank_patch_box": json.dumps(patch),
            "mask_grid_pixels": pixels[0],
            "mask_text_pixels": pixels[1],
            "mask_signal_pixels": pixels[2],
        }
    except Exception as exc:
        return {
            **base, "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def iter_results(
    jobs: list[Job], config: GenerationConfig, workers: int
) -> Iterable[dict[str, object]]:
    payloads = ((job, config) for job in jobs)
    if workers == 1:
        for payload in payloads:
            yield render_job(payload)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            yield from executor.map(render_job, payloads, chunksize=1)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    preferred = [
        "sample_id", "source_sample_id", "dataset", "split", "layout",
        "variant", "segment_transition", "image_path", "mask_path",
        "source_path", "width", "height",
        "grid_rows", "grid_columns", "profile", "paper_style", "is_blank",
        "row_names", "amplitude_scale", "augmentations", "chinese_lines",
        "barcode_box", "blank_patch_box", "mask_grid_pixels",
        "mask_text_pixels", "mask_signal_pixels", "error",
    ]
    extra = sorted({key for row in rows for key in row} - set(preferred) - {"status"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=preferred + extra, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    root = Path("data/sources")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ptb-root", type=Path, default=root / "ptb_xl")
    parser.add_argument("--chapman-root", type=Path, default=root / "chapman_ECG")
    parser.add_argument(
        "--datasets", choices=("both", "ptb-xl", "chapman"), default="both"
    )
    parser.add_argument(
        "--ptb-resolution", choices=("high", "low", "all"), default="high"
    )
    parser.add_argument("--limit-ptb", type=int)
    parser.add_argument("--limit-chapman", type=int)
    parser.add_argument(
        "--layouts", nargs="+", choices=LAYOUTS, default=list(LAYOUTS)
    )
    parser.add_argument(
        "--sizes-4x3", nargs="+", type=legacy.parse_size,
        default=list(DEFAULT_SIZES[LAYOUT_4X3]), metavar="WIDTHxHEIGHT",
        help="canvas pool for 4x3_plus_3; one is drawn uniformly per page",
    )
    parser.add_argument(
        "--sizes-6x2", nargs="+", type=legacy.parse_size,
        default=list(DEFAULT_SIZES[LAYOUT_6X2]), metavar="WIDTHxHEIGHT",
        help="canvas pool for 6x2_plus_1; one is drawn uniformly per page",
    )
    parser.add_argument(
        "--sizes-4x3-1", nargs="+", type=legacy.parse_size,
        default=list(DEFAULT_SIZES[LAYOUT_4X3_1]), metavar="WIDTHxHEIGHT",
        help="canvas pool for 4x3_plus_1; one is drawn uniformly per page",
    )
    parser.add_argument(
        "--transition-weights", type=parse_transition_weights,
        default=DEFAULT_TRANSITION_WEIGHTS, metavar="CONTINUOUS,GAP,PULSE",
        help="relative frequency of the three lead-separation styles",
    )
    parser.add_argument("--variants-per-record", type=int, default=1)
    parser.add_argument(
        "--split-ratios", type=parse_ratios, default=(0.80, 0.10, 0.10),
        metavar="TRAIN,VAL,TEST",
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--chapman-divisor", type=float, default=1000.0)
    parser.add_argument("--signal-clip-mv", type=float, default=5.0)
    parser.add_argument("--amplitude-min", type=float, default=0.85)
    parser.add_argument("--amplitude-max", type=float, default=1.15)
    parser.add_argument("--png-compression", type=int, choices=range(10), default=3)
    parser.add_argument(
        "--chinese-text-probability", type=probability, default=0.80
    )
    parser.add_argument("--chinese-font", type=Path)
    parser.add_argument(
        "--perspective-probability", type=probability, default=0.35
    )
    parser.add_argument("--perspective-distortion", type=float, default=0.025)
    parser.add_argument("--barcode-probability", type=probability, default=0.25)
    parser.add_argument(
        "--blank-patch-probability", type=probability, default=0.10
    )
    parser.add_argument("--moire-probability", type=probability, default=0.20)
    parser.add_argument("--glare-probability", type=probability, default=0.15)
    parser.add_argument("--blank-page-ratio", type=float, default=0.0)
    parser.add_argument("--unrectified-overlay-dir", type=Path)
    parser.add_argument(
        "--unrectified-overlay-probability", type=probability, default=0.0
    )
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.variants_per_record < 1 or args.workers < 1:
        raise ValueError("variants and workers must be at least 1")
    if args.seconds <= 0 or not 0 <= args.blank_page_ratio:
        raise ValueError("seconds must be positive and blank ratio non-negative")
    if not 0 <= args.perspective_distortion <= 0.12:
        raise ValueError("--perspective-distortion must be in [0, 0.12]")
    if not 0 < args.amplitude_min <= args.amplitude_max:
        raise ValueError("invalid amplitude range")
    for layout, sizes in layout_size_pools(args):
        if not sizes:
            raise ValueError(f"layout {layout} needs at least one size")


def layout_size_pools(
    args: argparse.Namespace,
) -> tuple[tuple[str, tuple[tuple[int, int], ...]], ...]:
    pools = {
        LAYOUT_4X3: args.sizes_4x3,
        LAYOUT_6X2: args.sizes_6x2,
        LAYOUT_4X3_1: args.sizes_4x3_1,
    }
    return tuple(
        (layout, tuple(tuple(size) for size in pools[layout]))
        for layout in args.layouts
    )


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    chinese_font = (
        resolve_chinese_font(args.chinese_font)
        if args.chinese_text_probability > 0 else ""
    )
    overlay_files = legacy.find_overlay_files(args.unrectified_overlay_dir)
    if args.unrectified_overlay_probability > 0 and not overlay_files:
        raise FileNotFoundError(
            "overlay probability is positive but no overlay files were found"
        )
    config = GenerationConfig(
        output_dir=str(args.output_dir),
        layout_sizes=layout_size_pools(args),
        transition_weights=tuple(args.transition_weights),
        seconds=args.seconds,
        chapman_divisor=args.chapman_divisor,
        signal_clip_mv=args.signal_clip_mv,
        png_compression=args.png_compression,
        amplitude_min=args.amplitude_min,
        amplitude_max=args.amplitude_max,
        chinese_text_probability=args.chinese_text_probability,
        chinese_font=chinese_font,
        perspective_probability=args.perspective_probability,
        perspective_distortion=args.perspective_distortion,
        barcode_probability=args.barcode_probability,
        blank_patch_probability=args.blank_patch_probability,
        moire_probability=args.moire_probability,
        glare_probability=args.glare_probability,
        unrectified_overlay_probability=args.unrectified_overlay_probability,
        unrectified_overlay_files=overlay_files,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    records = discover_records(args)
    if not records:
        raise FileNotFoundError("No source ECG records found")
    for split in ("train", "val", "test"):
        (args.output_dir / split).mkdir(parents=True, exist_ok=True)
    jobs = [
        Job(
            record, layout, variant,
            choose_split(record, args.split_ratios, args.seed),
            choose_size(
                record, layout, variant, sizes_for(config, layout), args.seed
            ),
            choose_transition(
                record, layout, variant, config.transition_weights, args.seed
            ),
        )
        for record in records
        for layout in args.layouts
        for variant in range(args.variants_per_record)
    ]
    successful: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for index, result in enumerate(iter_results(jobs, config, args.workers), 1):
        if result["status"] == "failed":
            failures.append(result)
            print(
                f"[{index}/{len(jobs)}] FAILED {result['sample_id']}: "
                f"{result['error']}", flush=True,
            )
        else:
            successful.append(result)
            print(
                f"[{index}/{len(jobs)}] {result['status']} "
                f"{result['split']}/{result['sample_id']}", flush=True,
            )
    write_manifest(args.output_dir / "manifest.csv", successful)
    report = {
        "records": len(records),
        "jobs": len(jobs),
        "successful": len(successful),
        "written": sum(row["status"] == "written" for row in successful),
        "skipped": sum(row["status"] == "skipped" for row in successful),
        "failed": len(failures),
        "split_counts": {
            split: sum(row["split"] == split for row in successful)
            for split in ("train", "val", "test")
        },
        "layout_counts": {
            layout: sum(row["layout"] == layout for row in successful)
            for layout in args.layouts
        },
        "dataset_counts": {
            name: sum(row["dataset"] == name for row in successful)
            for name in ("ptb_xl", "chapman")
        },
        "transition_counts": {
            name: sum(row["segment_transition"] == name for row in successful)
            for name in TRANSITIONS
        },
        "size_counts": {
            f"{layout}@{width}x{height}": sum(
                row["layout"] == layout
                and row["width"] == width
                and row["height"] == height
                for row in successful
            )
            for layout, sizes in config.layout_sizes
            for width, height in sizes
        },
        "mask_contract": {
            "format": "RGB PNG soft target",
            "red": "paper grid",
            "green": "text, calibration, barcode and nuisance marks",
            "blue": "physiological ECG trace",
            "implicit": "background = 1 - channel_sum/255",
        },
        "config": asdict(config),
        "failures": failures,
    }
    with (args.output_dir / "generation_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(
        f"Done: {len(successful)} successful, {len(failures)} failed; "
        f"output={args.output_dir}", flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
