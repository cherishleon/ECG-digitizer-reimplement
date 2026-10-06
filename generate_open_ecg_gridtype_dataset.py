#!/usr/bin/env python3
"""Generate ECG page PNGs with many grid types plus perspective-distorted pairs.

This builds on ``generate_open_ecg_segmentation_dataset`` (imported as ``base``,
never modified) and adds the two axes of variation that ecg-image-kit
(alphanumericslab/ecg-image-kit) exposes through ``--random_grid_color``,
``--random_grid_present`` and its distortion pass:

1. Multiple grid types. Paper colour, grid line colour, line style
   (solid/dotted/dashed), line width, and whether the minor or major lines are
   printed at all are drawn per page from ``GRID_STYLES``, with optional
   per-channel colour jitter so no two pages share exactly one palette. Ink and
   label colours vary with the style. Grid *geometry* is never randomised: the
   5 mm cell stays a 5 mm cell so 25 mm/s and 10 mm/mV remain exact.

2. Lead separation styles. Where two leads meet inside a printed row the pen
   either runs straight through (``continuous``), lifts for a couple of
   millimetres (``gap``), or prints a 1 mV separator pulse (``pulse``), drawn
   at ``--transition-weights`` (default 0.20/0.40/0.40). The separator pulse is
   a control mark, so it is supervised in the green channel rather than the
   blue trace channel.

3. Perspective distortions. Every base page is written undistorted; with
   ``--extra-perspective-probability`` (default 0.20) an *additional* pair
   ``<stem>__persp.png`` / ``<stem>__persp_mask.png`` is written from the same
   render with rotation + perspective warp applied to the image and to all three
   mask layers under one homography, so image and mask stay pixel-aligned.

Output pairs are split/xxx.png and split/xxx_mask.png. Mask channels are
R=paper grid, G=text/control/nuisance, B=physiological trace; all-zero pixels
are background. Splits are assigned per source recording, so every layout,
size, grid type and distorted twin of one record lands in the same split.

Example:
    python generate_open_ecg_gridtype_dataset.py \
        --output-dir /path/to/out --workers 24
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

if __package__:
    from . import ecg_page_rendering as page_renderer
    from . import generate_open_ecg_segmentation_dataset as base
    from . import generate_synthetic_3x4_rhythm_dataset as legacy
else:
    import ecg_page_rendering as page_renderer
    import generate_open_ecg_segmentation_dataset as base
    import generate_synthetic_3x4_rhythm_dataset as legacy


BGR = tuple[int, int, int]


@dataclass(frozen=True)
class GridStyle:
    """One printable paper/grid appearance. All colours are OpenCV BGR."""

    name: str
    paper: BGR
    minor: BGR
    major: BGR
    ink: BGR = (24, 24, 24)
    label: BGR = (35, 35, 35)
    footer: BGR = (85, 85, 85)
    kind: str = "solid"  # solid | dotted | dashed
    minor_width: float = 1.0
    major_width: float = 1.0
    show_minor: bool = True
    show_major: bool = True
    minor_value: int = 72  # soft target value written into the grid mask
    major_value: int = 144


# Palettes follow real thermal/laser ECG printouts and the colour families
# ecg-image-kit samples from (red/orange/brown standard paper, plus the
# grid-free and dot-matrix variants seen on scanned reports).
GRID_STYLES: tuple[GridStyle, ...] = (
    GridStyle("pink_fine", (253, 253, 255), (220, 214, 255), (160, 143, 244),
              footer=(150, 88, 170)),
    GridStyle("red_fine", (249, 249, 255), (201, 198, 252), (112, 105, 227),
              footer=(150, 88, 170)),
    GridStyle("red_bold", (247, 247, 255), (196, 192, 250), (90, 82, 222),
              major_width=2.0, major_value=168, footer=(150, 88, 170)),
    GridStyle("orange", (250, 252, 255), (190, 214, 250), (96, 150, 240),
              footer=(96, 120, 170)),
    GridStyle("brown", (248, 250, 253), (196, 206, 224), (110, 132, 168),
              ink=(30, 30, 36), footer=(100, 110, 130)),
    GridStyle("blue", (255, 252, 250), (250, 222, 204), (222, 150, 100),
              ink=(80, 40, 24), label=(80, 44, 30), footer=(180, 120, 80)),
    GridStyle("green", (250, 255, 250), (214, 240, 214), (120, 190, 120),
              ink=(28, 40, 28), footer=(90, 140, 90)),
    GridStyle("grey_white", (255, 255, 255), (226, 226, 226), (176, 176, 176)),
    GridStyle("grey_faint", (252, 252, 252), (240, 240, 240), (214, 214, 214),
              minor_value=48, major_value=104),
    GridStyle("dotted_grey", (253, 253, 253), (182, 182, 182), (105, 105, 105),
              kind="dotted", major_width=2.0, minor_value=96, major_value=192),
    GridStyle("dotted_red", (252, 252, 255), (196, 190, 246), (120, 110, 226),
              kind="dotted", major_width=2.0, minor_value=96, major_value=192,
              footer=(150, 88, 170)),
    GridStyle("dashed_grey", (254, 254, 254), (206, 206, 206), (150, 150, 150),
              kind="dashed", minor_value=88, major_value=176),
    GridStyle("major_only_red", (250, 250, 255), (201, 198, 252),
              (112, 105, 227), show_minor=False, major_value=160,
              footer=(150, 88, 170)),
    GridStyle("major_only_grey", (253, 253, 253), (226, 226, 226),
              (150, 150, 150), show_minor=False, major_value=160),
    GridStyle("no_grid", (255, 255, 255), (255, 255, 255), (255, 255, 255),
              show_minor=False, show_major=False),
)
GRID_STYLE_BY_NAME = {style.name: style for style in GRID_STYLES}
DEFAULT_GRID_STYLES = tuple(style.name for style in GRID_STYLES)

# How neighbouring leads in one printed row are separated. Real recorders
# either run the pen straight through ("continuous"), lift it briefly
# ("gap"), or print a 1 mV separator pulse between the columns ("pulse").
TRANSITIONS = page_renderer.TRANSITIONS
DEFAULT_TRANSITION_WEIGHTS = base.DEFAULT_TRANSITION_WEIGHTS


@dataclass(frozen=True)
class GenerationConfig:
    output_dir: str
    layout_sizes: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]
    grid_styles: tuple[str, ...]
    grid_color_jitter: int
    transition_weights: tuple[float, float, float]
    seconds: float
    chapman_divisor: float
    signal_clip_mv: float
    png_compression: int
    amplitude_min: float
    amplitude_max: float
    chinese_text_probability: float
    chinese_font: str
    extra_perspective_probability: float
    perspective_probability: float
    perspective_distortion: float
    perspective_rotation: float
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
    grid_style: str
    transition: str


@dataclass
class RenderedPage:
    image: np.ndarray
    grid_mask: np.ndarray
    text_mask: np.ndarray
    signal_mask: np.ndarray


def choose_grid_style(
    record: legacy.SourceRecord,
    layout: str,
    variant: int,
    styles: Sequence[str],
    seed: int,
) -> str:
    """Draw one grid type uniformly; the hash keeps runs reproducible."""
    key = f"grid:{seed}:{record.dataset}:{record.sample_id}:{layout}:{variant}"
    return styles[base.stable_u64(key) % len(styles)]


def jitter_color(color: BGR, amount: int, rng: np.random.Generator) -> BGR:
    if amount <= 0:
        return color
    shifted = np.asarray(color, dtype=np.int32) + rng.integers(
        -amount, amount + 1, size=3
    )
    return tuple(int(value) for value in np.clip(shifted, 0, 255))


def jitter_style(
    style: GridStyle, amount: int, rng: np.random.Generator
) -> GridStyle:
    """Perturb colours and line widths so each page prints slightly differently."""
    replacements = {
        "paper": jitter_color(style.paper, max(0, amount // 3), rng),
        "minor": jitter_color(style.minor, amount, rng),
        "major": jitter_color(style.major, amount, rng),
        "ink": jitter_color(style.ink, max(0, amount // 2), rng),
        "label": jitter_color(style.label, max(0, amount // 2), rng),
        "minor_width": style.minor_width * float(rng.uniform(0.9, 1.3)),
        "major_width": style.major_width * float(rng.uniform(0.9, 1.4)),
    }
    return GridStyle(**{**asdict(style), **replacements})


def line_segments(
    start: int, stop: int, kind: str, spacing: float
) -> list[tuple[int, int]]:
    """Split one grid line into the pieces a solid/dotted/dashed print leaves."""
    if kind == "solid":
        return [(start, stop)]
    if kind == "dotted":
        # Same dot pitch as the legacy dotted paper: three dots per 1 mm cell.
        step = max(3, int(round(spacing / 3.0)))
        return [(value, value) for value in range(start, stop, step)]
    if kind == "dashed":
        dash = max(3, int(round(1.4 * spacing)))
        gap = max(2, int(round(0.8 * spacing)))
        segments, cursor = [], start
        while cursor < stop:
            segments.append((cursor, min(cursor + dash, stop)))
            cursor += dash + gap
        return segments
    raise ValueError(f"unknown grid line kind {kind!r}")


def stroke_grid_lines(
    canvas: np.ndarray,
    mask: np.ndarray,
    coordinates: np.ndarray,
    extent: tuple[int, int],
    vertical: bool,
    color: BGR,
    value: int,
    thickness: int,
    kind: str,
    spacing: float,
) -> None:
    """Draw one family of grid lines into the page and its soft target alike."""
    segments = line_segments(extent[0], extent[1] - 1, kind, spacing)
    for position in coordinates:
        position = int(position)
        for first, last in segments:
            if vertical:
                p1, p2 = (position, first), (position, last)
            else:
                p1, p2 = (first, position), (last, position)
            cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)
            cv2.line(mask, p1, p2, value, thickness, cv2.LINE_AA)


def draw_paper(
    profile: page_renderer.PageProfile, style: GridStyle
) -> tuple[np.ndarray, np.ndarray]:
    """Render the paper for ``style`` and the exactly aligned soft grid target."""
    height, width = profile.height, profile.width
    left, right, top, bottom = page_renderer._pixel_bounds(profile)
    minor_x, major_x, minor_y, major_y = page_renderer.grid_line_positions(profile)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[top:bottom, left:right] = style.paper
    mask = np.zeros((height, width), dtype=np.uint8)
    spacing = float(np.median(np.diff(minor_x)))
    scale = width / 3508.0
    families = (
        (style.show_minor, minor_x, minor_y, style.minor, style.minor_value,
         style.minor_width),
        (style.show_major, major_x, major_y, style.major, style.major_value,
         style.major_width),
    )
    for enabled, xs, ys, color, value, line_width in families:
        if not enabled:
            continue
        thickness = max(1, int(round(line_width * max(1.0, scale))))
        stroke_grid_lines(
            canvas, mask, xs, (top, bottom), True, color, value, thickness,
            style.kind, spacing,
        )
        stroke_grid_lines(
            canvas, mask, ys, (left, right), False, color, value, thickness,
            style.kind, spacing,
        )
    return canvas, mask


def put_text(
    image: np.ndarray,
    text_mask: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: BGR,
) -> None:
    for target, fill in ((image, color), (text_mask, 255)):
        cv2.putText(
            target, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, fill, 1,
            cv2.LINE_AA,
        )


def draw_calibration(
    image: np.ndarray,
    text_mask: np.ndarray,
    profile: page_renderer.PageProfile,
    x0: int,
    baseline: float,
    pixels_per_mm: float,
    thickness: int,
    first_trace_y: int,
    color: BGR,
) -> None:
    y = int(round(baseline))
    pulse_height = int(round(10 * pixels_per_mm))
    xs = [x0 + int(round(value * pixels_per_mm)) for value in profile.calibration_mm]
    ys = (y, y, y - pulse_height, y - pulse_height, y, y, first_trace_y)
    points = np.column_stack((xs, ys)).astype(np.int32)
    cv2.polylines(image, [points], False, color, thickness, cv2.LINE_AA)
    cv2.polylines(text_mask, [points], False, 255, thickness, cv2.LINE_AA)


def draw_footer(
    image: np.ndarray,
    text_mask: np.ndarray,
    profile: page_renderer.PageProfile,
    color: BGR,
    layout_text: str,
) -> None:
    left, right, _, bottom = page_renderer._pixel_bounds(profile)
    span = right - left
    scale = max(0.35, profile.width / 6600.0)
    footer_height = max(9, int(round(24 * scale)))
    space_below = profile.height - bottom
    if space_below > 2 * footer_height:
        y = min(profile.height - 8, bottom + space_below // 2)
    else:
        y = min(profile.height - 8, bottom - 7)
    labels = (
        (0.005, "150 Hz"),
        (0.090, "25.0 mm/s    10.0 mm/mV"),
        (0.315, "marquette electronics, inc."),
        (0.610, layout_text),
        (0.840, "MAC55 009C"),
    )
    for fraction, text in labels:
        put_text(
            image, text_mask, text,
            (left + int(round(fraction * span)), y), scale, color,
        )


def render_page(
    rows: Sequence[Sequence[np.ndarray]],
    row_names: Sequence[str],
    profile: page_renderer.PageProfile,
    style: GridStyle,
    *,
    is_blank: bool,
    amplitude_scale: float,
    footer_text: str,
    transition: str,
    rng: np.random.Generator,
) -> RenderedPage:
    """Same geometry as page_renderer.render_ecg_page, with a pluggable style."""
    if len(rows) != len(profile.baselines):
        raise ValueError(
            f"profile has {len(profile.baselines)} baselines for {len(rows)} rows"
        )
    width, height = profile.width, profile.height
    x0 = int(round(profile.x_start * width))
    x1 = int(round(profile.x_end * width))
    baselines = [fraction * height for fraction in profile.baselines]
    pixels_per_mm = (x1 - x0) / 250.0
    pixels_per_mv = 10.0 * pixels_per_mm
    image, grid_mask = draw_paper(profile, style)
    text_mask = np.zeros((height, width), dtype=np.uint8)
    signal_mask = np.zeros((height, width), dtype=np.uint8)
    draw_footer(image, text_mask, profile, style.footer, footer_text)
    thickness = max(1, int(round(width / 1750)))

    if not is_blank:
        pulse_height = int(round(10 * pixels_per_mm))
        # One break width per page keeps a printout internally consistent.
        break_mm = (
            float(rng.uniform(1.2, 3.0)) if transition == "gap"
            else float(rng.uniform(3.6, 5.2))
        )
        break_px = max(2, int(round(break_mm * pixels_per_mm)))
        for row, baseline in zip(rows, baselines):
            parts = [
                part * amplitude_scale for part in page_renderer.row_parts(row)
            ]
            pieces = page_renderer.row_trace_pieces(
                parts, width, height, x0, x1, baseline, pixels_per_mv,
                transition, break_px,
            )
            for *_, points in pieces:
                cv2.polylines(
                    image, [points], False, style.ink, thickness, cv2.LINE_AA
                )
                cv2.polylines(
                    signal_mask, [points], False, 255, thickness, cv2.LINE_AA
                )
            if transition == "pulse":
                for index in range(len(pieces) - 1):
                    entry, exit_points = pieces[index][3], pieces[index + 1][3]
                    page_renderer.draw_separator_pulse(
                        image, text_mask, int(entry[-1, 0]),
                        int(exit_points[0, 0]), baseline, pixels_per_mm,
                        pulse_height, thickness, style.ink,
                        int(entry[-1, 1]), int(exit_points[0, 1]),
                    )
            draw_calibration(
                image, text_mask, profile, x0, baseline, pixels_per_mm,
                thickness, int(pieces[0][3][0, 1]), style.ink,
            )

        label_scale = max(0.45, width / 5000.0)
        row_gap = float(np.median(np.diff(baselines)))
        for channel, name in enumerate(row_names):
            leads = name.split("|")
            if len(leads) > 1:
                for column, lead in enumerate(leads):
                    label_x = x0 + int((x1 - x0) * column / len(leads)) + 3
                    label_y = int(round(baselines[channel] - 0.18 * row_gap))
                    put_text(
                        image, text_mask, lead, (label_x, max(15, label_y)),
                        label_scale, style.label,
                    )
            else:
                put_text(
                    image, text_mask, name,
                    (x0 + 3, int(baselines[channel] - 8)), label_scale,
                    style.label,
                )
    return RenderedPage(image, grid_mask, text_mask, signal_mask)


def perspective_distort(
    image: np.ndarray,
    layers: Sequence[np.ndarray],
    rng: np.random.Generator,
    distortion: float,
    max_rotation: float,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, float]]:
    """Rotate and perspective-warp page and masks under one shared homography."""
    height, width = image.shape[:2]
    angle = float(rng.uniform(-max_rotation, max_rotation))
    zoom = float(rng.uniform(0.93, 1.0))
    rotation = np.vstack((
        cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, zoom),
        np.array([0.0, 0.0, 1.0]),
    )).astype(np.float32)
    source = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )
    jitter = np.column_stack((
        rng.uniform(-distortion * width, distortion * width, 4),
        rng.uniform(-distortion * height, distortion * height, 4),
    )).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(source, source + jitter) @ rotation
    border = int(rng.integers(214, 246))
    warped = cv2.warpPerspective(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(border, border, border),
    )
    warped_layers = [
        cv2.warpPerspective(
            layer, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        for layer in layers
    ]
    return warped, warped_layers, {"rotation_deg": angle, "zoom": zoom}


def write_pair(
    image: np.ndarray,
    layers: Sequence[np.ndarray],
    image_path: Path,
    mask_path: Path,
    compression: int,
) -> list[int]:
    mask_rgb = page_renderer.compose_rgb_mask(*layers)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    options = [cv2.IMWRITE_PNG_COMPRESSION, compression]
    if not cv2.imwrite(str(image_path), image, options):
        raise OSError(f"failed to write {image_path}")
    if not cv2.imwrite(
        str(mask_path), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR), options
    ):
        raise OSError(f"failed to write {mask_path}")
    return [int(np.count_nonzero(mask_rgb[:, :, channel])) for channel in range(3)]


def render_job(payload: tuple[Job, GenerationConfig]) -> list[dict[str, object]]:
    job, config = payload
    record = job.record
    spec = base.layout_spec(job.layout, job.size)
    profile = base.square_grid_profile(spec)
    rng = np.random.default_rng(base.job_seed(job, config.seed))
    stem = base.safe_stem(
        f"{record.sample_id}__{job.layout}__{job.size[0]}x{job.size[1]}"
        f"__{job.grid_style}__{job.transition}__v{job.variant:03d}"
    )
    split_dir = Path(config.output_dir) / job.split
    common = {
        "source_sample_id": record.sample_id,
        "dataset": record.dataset,
        "split": job.split,
        "layout": job.layout,
        "variant": job.variant,
        "grid_style": job.grid_style,
        "segment_transition": job.transition,
        "source_path": record.signal_path,
        "width": profile.width,
        "height": profile.height,
        "grid_rows": spec.grid_rows,
        "grid_columns": spec.grid_columns,
    }
    outputs = [(stem, "none")]
    make_twin = rng.random() < config.extra_perspective_probability
    if make_twin:
        outputs.append((f"{stem}__persp", "perspective"))
    paths = {
        name: (split_dir / f"{name}.png", split_dir / f"{name}_mask.png")
        for name, _ in outputs
    }
    if not config.overwrite and all(
        image_path.is_file() and mask_path.is_file()
        for image_path, mask_path in paths.values()
    ):
        return [
            {
                **common, "status": "skipped", "sample_id": name,
                "parent_sample_id": stem, "distortion": distortion,
                "image_path": str(paths[name][0].relative_to(config.output_dir)),
                "mask_path": str(paths[name][1].relative_to(config.output_dir)),
            }
            for name, distortion in outputs
        ]

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
            rows, row_names = base.make_rows(
                signals, sample_rate, spec, config.seconds, config.signal_clip_mv
            )
        style = jitter_style(
            GRID_STYLE_BY_NAME[job.grid_style], config.grid_color_jitter, rng
        )
        amplitude = float(rng.uniform(config.amplitude_min, config.amplitude_max))
        rendered = render_page(
            rows, row_names, profile, style,
            is_blank=record.force_blank, amplitude_scale=amplitude,
            footer_text=spec.footer_text, transition=job.transition, rng=rng,
        )
        image = rendered.image
        layers = [rendered.grid_mask, rendered.text_mask, rendered.signal_mask]
        augmentations: list[str] = []
        chinese_lines: list[str] = []
        barcode: list[int] = []
        patch: list[int] = []
        if rng.random() < config.chinese_text_probability:
            chinese_lines = base.draw_chinese_report_text(
                image, layers[1], profile, config.chinese_font, rng
            )
            augmentations.append("chinese_report_text")
        if rng.random() < config.barcode_probability:
            barcode = base.add_barcode(image, layers, profile, rng)
            augmentations.append("barcode")
        if rng.random() < config.blank_patch_probability:
            patch = base.add_blank_patch(image, layers, rng)
            augmentations.append("blank_patch")
        if (
            config.unrectified_overlay_files
            and rng.random() < config.unrectified_overlay_probability
        ):
            overlay_path = str(rng.choice(config.unrectified_overlay_files))
            base.add_unrectified_overlay(image, layers[1], overlay_path, rng)
            augmentations.append("unrectified_overlay")
        if rng.random() < config.perspective_probability:
            image, layers, _ = perspective_distort(
                image, layers, rng, config.perspective_distortion,
                config.perspective_rotation,
            )
            augmentations.append("perspective")
        if rng.random() < config.moire_probability:
            legacy.add_moire(image, rng)
            augmentations.append("moire")
        if rng.random() < config.glare_probability:
            legacy.add_glare(image, rng)
            augmentations.append("glare")

        results: list[dict[str, object]] = []
        for name, distortion in outputs:
            if distortion == "perspective":
                page, page_layers, warp = perspective_distort(
                    image.copy(), [layer.copy() for layer in layers], rng,
                    config.perspective_distortion, config.perspective_rotation,
                )
                extra = {
                    "augmentations": "|".join([*augmentations, "extra_perspective"]),
                    "rotation_deg": round(warp["rotation_deg"], 4),
                    "zoom": round(warp["zoom"], 4),
                }
            else:
                page, page_layers = image, layers
                extra = {"augmentations": "|".join(augmentations)}
            image_path, mask_path = paths[name]
            pixels = write_pair(
                page, page_layers, image_path, mask_path, config.png_compression
            )
            results.append({
                **common,
                "status": "written",
                "sample_id": name,
                "parent_sample_id": stem,
                "distortion": distortion,
                "image_path": str(image_path.relative_to(config.output_dir)),
                "mask_path": str(mask_path.relative_to(config.output_dir)),
                "profile": profile.name,
                "is_blank": int(record.force_blank),
                "row_names": "|".join(row_names),
                "amplitude_scale": amplitude,
                "chinese_lines": "|".join(chinese_lines),
                "barcode_box": json.dumps(barcode),
                "blank_patch_box": json.dumps(patch),
                "mask_grid_pixels": pixels[0],
                "mask_text_pixels": pixels[1],
                "mask_signal_pixels": pixels[2],
                **extra,
            })
        return results
    except Exception as exc:
        return [{
            **common, "status": "failed", "sample_id": stem,
            "parent_sample_id": stem, "distortion": "none",
            "image_path": str(paths[stem][0].relative_to(config.output_dir)),
            "mask_path": str(paths[stem][1].relative_to(config.output_dir)),
            "error": f"{type(exc).__name__}: {exc}",
        }]


def iter_results(
    jobs: list[Job], config: GenerationConfig, workers: int
) -> Iterable[list[dict[str, object]]]:
    payloads = ((job, config) for job in jobs)
    if workers == 1:
        for payload in payloads:
            yield render_job(payload)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            yield from executor.map(render_job, payloads, chunksize=1)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    preferred = [
        "sample_id", "parent_sample_id", "source_sample_id", "dataset", "split",
        "layout", "variant", "grid_style", "segment_transition", "distortion",
        "image_path",
        "mask_path", "source_path", "width", "height", "grid_rows",
        "grid_columns", "profile", "is_blank", "row_names", "amplitude_scale",
        "augmentations", "rotation_deg", "zoom", "chinese_lines", "barcode_box",
        "blank_patch_box", "mask_grid_pixels", "mask_text_pixels",
        "mask_signal_pixels", "error",
    ]
    extra = sorted({key for row in rows for key in row} - set(preferred) - {"status"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=preferred + extra, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def grid_style_name(value: str) -> str:
    if value not in GRID_STYLE_BY_NAME:
        raise argparse.ArgumentTypeError(
            f"unknown grid style {value!r}; choose from "
            f"{', '.join(DEFAULT_GRID_STYLES)}"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    root = Path("data/sources")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
        "--layouts", nargs="+", choices=base.LAYOUTS, default=list(base.LAYOUTS)
    )
    parser.add_argument(
        "--sizes-4x3", nargs="+", type=legacy.parse_size,
        default=list(base.DEFAULT_SIZES[base.LAYOUT_4X3]), metavar="WIDTHxHEIGHT",
    )
    parser.add_argument(
        "--sizes-6x2", nargs="+", type=legacy.parse_size,
        default=list(base.DEFAULT_SIZES[base.LAYOUT_6X2]), metavar="WIDTHxHEIGHT",
    )
    parser.add_argument(
        "--sizes-4x3-1", nargs="+", type=legacy.parse_size,
        default=list(base.DEFAULT_SIZES[base.LAYOUT_4X3_1]), metavar="WIDTHxHEIGHT",
    )
    parser.add_argument(
        "--grid-styles", nargs="+", type=grid_style_name,
        default=list(DEFAULT_GRID_STYLES),
        help="grid types to sample from; one is drawn uniformly per page",
    )
    parser.add_argument(
        "--grid-color-jitter", type=int, default=10,
        help="per-channel colour jitter applied to the chosen grid type",
    )
    parser.add_argument(
        "--transition-weights", type=base.parse_transition_weights,
        default=DEFAULT_TRANSITION_WEIGHTS, metavar="CONTINUOUS,GAP,PULSE",
        help="relative frequency of the three lead-separation styles",
    )
    parser.add_argument("--variants-per-record", type=int, default=1)
    parser.add_argument(
        "--split-ratios", type=base.parse_ratios, default=(0.85, 0.15, 0.00),
        metavar="TRAIN,VAL,TEST",
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--chapman-divisor", type=float, default=1000.0)
    parser.add_argument("--signal-clip-mv", type=float, default=5.0)
    parser.add_argument("--amplitude-min", type=float, default=0.85)
    parser.add_argument("--amplitude-max", type=float, default=1.15)
    parser.add_argument("--png-compression", type=int, choices=range(10), default=3)
    parser.add_argument(
        "--chinese-text-probability", type=base.probability, default=0.80
    )
    parser.add_argument("--chinese-font", type=Path)
    parser.add_argument(
        "--extra-perspective-probability", type=base.probability, default=0.20,
        help="probability of writing an extra perspective-distorted twin page",
    )
    parser.add_argument(
        "--perspective-probability", type=base.probability, default=0.0,
        help="probability of distorting the base page itself (off by default)",
    )
    parser.add_argument("--perspective-distortion", type=float, default=0.030)
    parser.add_argument("--perspective-rotation", type=float, default=4.0)
    parser.add_argument("--barcode-probability", type=base.probability, default=0.25)
    parser.add_argument(
        "--blank-patch-probability", type=base.probability, default=0.10
    )
    parser.add_argument("--moire-probability", type=base.probability, default=0.20)
    parser.add_argument("--glare-probability", type=base.probability, default=0.15)
    parser.add_argument("--blank-page-ratio", type=float, default=0.0)
    parser.add_argument("--unrectified-overlay-dir", type=Path)
    parser.add_argument(
        "--unrectified-overlay-probability", type=base.probability, default=0.0
    )
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def layout_size_pools(
    args: argparse.Namespace,
) -> tuple[tuple[str, tuple[tuple[int, int], ...]], ...]:
    pools = {
        base.LAYOUT_4X3: args.sizes_4x3,
        base.LAYOUT_6X2: args.sizes_6x2,
        base.LAYOUT_4X3_1: args.sizes_4x3_1,
    }
    return tuple(
        (layout, tuple(tuple(size) for size in pools[layout]))
        for layout in args.layouts
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.variants_per_record < 1 or args.workers < 1:
        raise ValueError("variants and workers must be at least 1")
    if args.seconds <= 0 or args.blank_page_ratio < 0:
        raise ValueError("seconds must be positive and blank ratio non-negative")
    if not 0 <= args.perspective_distortion <= 0.12:
        raise ValueError("--perspective-distortion must be in [0, 0.12]")
    if not 0 <= args.perspective_rotation <= 20:
        raise ValueError("--perspective-rotation must be in [0, 20] degrees")
    if not 0 < args.amplitude_min <= args.amplitude_max:
        raise ValueError("invalid amplitude range")
    if not args.grid_styles:
        raise ValueError("at least one grid style is required")
    if args.grid_color_jitter < 0:
        raise ValueError("--grid-color-jitter must be non-negative")
    for layout, sizes in layout_size_pools(args):
        if not sizes:
            raise ValueError(f"layout {layout} needs at least one size")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    chinese_font = (
        base.resolve_chinese_font(args.chinese_font)
        if args.chinese_text_probability > 0 else ""
    )
    overlay_files = legacy.find_overlay_files(args.unrectified_overlay_dir)
    if args.unrectified_overlay_probability > 0 and not overlay_files:
        raise FileNotFoundError(
            "overlay probability is positive but no overlay files were found"
        )
    layout_sizes = layout_size_pools(args)
    config = GenerationConfig(
        output_dir=str(args.output_dir),
        layout_sizes=layout_sizes,
        grid_styles=tuple(args.grid_styles),
        grid_color_jitter=args.grid_color_jitter,
        transition_weights=tuple(args.transition_weights),
        seconds=args.seconds,
        chapman_divisor=args.chapman_divisor,
        signal_clip_mv=args.signal_clip_mv,
        png_compression=args.png_compression,
        amplitude_min=args.amplitude_min,
        amplitude_max=args.amplitude_max,
        chinese_text_probability=args.chinese_text_probability,
        chinese_font=chinese_font,
        extra_perspective_probability=args.extra_perspective_probability,
        perspective_probability=args.perspective_probability,
        perspective_distortion=args.perspective_distortion,
        perspective_rotation=args.perspective_rotation,
        barcode_probability=args.barcode_probability,
        blank_patch_probability=args.blank_patch_probability,
        moire_probability=args.moire_probability,
        glare_probability=args.glare_probability,
        unrectified_overlay_probability=args.unrectified_overlay_probability,
        unrectified_overlay_files=overlay_files,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    records = base.discover_records(args)
    if not records:
        raise FileNotFoundError("No source ECG records found")
    for split in ("train", "val", "test"):
        (args.output_dir / split).mkdir(parents=True, exist_ok=True)
    jobs = [
        Job(
            record, layout, variant,
            base.choose_split(record, args.split_ratios, args.seed),
            base.choose_size(
                record, layout, variant,
                base.sizes_for(config, layout), args.seed,
            ),
            choose_grid_style(
                record, layout, variant, config.grid_styles, args.seed
            ),
            base.choose_transition(
                record, layout, variant, config.transition_weights, args.seed
            ),
        )
        for record in records
        for layout in args.layouts
        for variant in range(args.variants_per_record)
    ]
    successful: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for index, results in enumerate(iter_results(jobs, config, args.workers), 1):
        for result in results:
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
        "pages": len(successful),
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
        "size_counts": {
            f"{layout}@{width}x{height}": sum(
                row["layout"] == layout
                and row["width"] == width
                and row["height"] == height
                for row in successful
            )
            for layout, sizes in layout_sizes
            for width, height in sizes
        },
        "grid_style_counts": {
            name: sum(row["grid_style"] == name for row in successful)
            for name in config.grid_styles
        },
        "transition_counts": {
            name: sum(row["segment_transition"] == name for row in successful)
            for name in TRANSITIONS
        },
        "distortion_counts": {
            name: sum(row["distortion"] == name for row in successful)
            for name in ("none", "perspective")
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
        f"Done: {len(successful)} pages, {len(failures)} failed; "
        f"output={args.output_dir}", flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
