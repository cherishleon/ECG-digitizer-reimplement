"""Private-scan-calibrated ECG page rendering shared by generation/comparison.

The two profiles below were fitted from paired JPG/XML records in
``ecg_data/dataset_inner/strict``.  The 3508 profile uses original scans that
are 3507 pixels wide and normalises the one-pixel scanner difference.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class PageProfile:
    name: str
    width: int
    height: int
    x_start: float
    x_end: float
    baselines: tuple[float, ...]
    grid_left: float
    grid_right: float
    grid_top: float
    grid_bottom: float
    grid_rows: int
    grid_columns: int
    calibration_mm: tuple[float, ...]


@dataclass
class RenderedPage:
    image: np.ndarray
    # Open-ECG-Digitizer compatible soft target.  RGB channels are
    # grid/paper, text/control marks, and physiological signal respectively.
    mask_rgb: np.ndarray
    grid_mask: np.ndarray
    text_mask: np.ndarray
    signal_mask: np.ndarray
    coordinates: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    x0: int
    x1: int
    baselines: list[float]
    pixels_per_mm: float
    pixels_per_mv: float
    profile: PageProfile


# Median geometry from three paired private records per size cluster.
STRICT_3508 = PageProfile(
    name="strict_3508x2480",
    width=3508,
    height=2480,
    x_start=432 / 3508,
    x_end=3408 / 3508,
    baselines=(0.3343, 0.4440, 0.5540, 0.6617, 0.7714, 0.8790),
    # Preserve the paired-scan right edge and cover the full 56-column paper.
    grid_left=0.038,
    grid_right=0.988,
    grid_top=0.036,
    grid_bottom=0.974,
    grid_rows=39,
    grid_columns=56,
    # This scanner/template places the pulse closer to the trace.
    calibration_mm=(-6.5, -5.5, -5.5, -0.9, -0.9, 0.0, 0.0),
)

STRICT_3320 = PageProfile(
    name="strict_3320x2552",
    width=3320,
    height=2552,
    x_start=268 / 3320,
    x_end=3246 / 3320,
    baselines=(0.3444, 0.4510, 0.5560, 0.6611, 0.7672, 0.8723),
    grid_left=0.000,
    grid_right=0.999,
    grid_top=0.052,
    grid_bottom=0.969,
    grid_rows=39,
    grid_columns=56,
    calibration_mm=(-9.0, -8.0, -8.0, -3.0, -3.0, 0.0, 0.0),
)


def profile_for_size(width: int, height: int) -> PageProfile:
    """Select the closest calibrated private layout for a target canvas."""
    profiles = (STRICT_3508, STRICT_3320)
    profile = min(
        profiles,
        key=lambda item: abs(width - item.width) + abs(height - item.height),
    )
    # Preserve normalized geometry when a scanner differs by a few pixels.
    return replace(profile, width=width, height=height)


def custom_profile(
    width: int,
    height: int,
    x_start: float,
    x_end: float,
    baselines: Sequence[float],
) -> PageProfile:
    """Build the old full-page-grid geometry for explicit CLI overrides."""
    return PageProfile(
        name="custom",
        width=width,
        height=height,
        x_start=x_start,
        x_end=x_end,
        baselines=tuple(baselines),
        grid_left=0.0,
        grid_right=1.0,
        grid_top=0.0,
        grid_bottom=1.0,
        grid_rows=39,
        grid_columns=56,
        calibration_mm=(-8.5, -7.5, -7.5, -2.5, -2.5, 0.0, 0.0),
    )


def _pixel_bounds(profile: PageProfile) -> tuple[int, int, int, int]:
    left = int(round(profile.grid_left * profile.width))
    right = int(round(profile.grid_right * profile.width))
    top = int(round(profile.grid_top * profile.height))
    bottom = int(round(profile.grid_bottom * profile.height))
    return (
        max(0, min(profile.width - 1, left)),
        max(1, min(profile.width, right)),
        max(0, min(profile.height - 1, top)),
        max(1, min(profile.height, bottom)),
    )


def _grid_line_positions(
    start: int,
    stop: int,
    cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return minor/major coordinates for exactly ``cells`` major cells."""
    if cells <= 0 or stop <= start:
        raise ValueError(f"invalid grid extent: start={start} stop={stop} cells={cells}")
    minor = np.rint(
        np.linspace(start, stop - 1, cells * 5 + 1, dtype=np.float64)
    ).astype(np.int32)
    if np.any(np.diff(minor) <= 0):
        raise ValueError("grid extent is too small for five subdivisions per cell")
    return minor, minor[::5]


def grid_line_positions(profile: PageProfile) -> tuple[np.ndarray, ...]:
    """Expose exact grid coordinates for rendering checks and comparisons."""
    left, right, top, bottom = _pixel_bounds(profile)
    minor_x, major_x = _grid_line_positions(left, right, profile.grid_columns)
    minor_y, major_y = _grid_line_positions(top, bottom, profile.grid_rows)
    return minor_x, major_x, minor_y, major_y


def _draw_dotted_grid(
    canvas: np.ndarray,
    bounds: tuple[int, int, int, int],
    minor_x: np.ndarray,
    major_x: np.ndarray,
    minor_y: np.ndarray,
    major_y: np.ndarray,
) -> None:
    left, right, top, bottom = bounds
    minor_spacing = int(round(float(np.median(np.diff(minor_x)))))
    dot_gap = max(3, minor_spacing // 3)
    minor = np.array((182, 182, 182), dtype=np.uint8)
    major = np.array((105, 105, 105), dtype=np.uint8)
    for x in minor_x:
        canvas[top:bottom:dot_gap, x] = minor
    for y in minor_y:
        canvas[y, left:right:dot_gap] = minor
    for x in major_x:
        canvas[top:bottom:dot_gap, x:min(x + 2, right)] = major
    for y in major_y:
        canvas[y:min(y + 2, bottom), left:right:dot_gap] = major


def draw_paper(
    profile: PageProfile,
    pixels_per_mm: float,
    style: str,
) -> np.ndarray:
    """Draw colored, white, or dotted ECG paper; every style has a grid."""
    height, width = profile.height, profile.width
    bounds = _pixel_bounds(profile)
    left, right, top, bottom = bounds
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    minor_x, major_x, minor_y, major_y = grid_line_positions(profile)

    if style == "pink":
        base, minor, major = (253, 253, 255), (220, 214, 255), (160, 143, 244)
    elif style == "red":
        base, minor, major = (249, 249, 255), (201, 198, 252), (112, 105, 227)
    elif style == "white":
        base, minor, major = (255, 255, 255), (226, 226, 226), (176, 176, 176)
    elif style == "dotted":
        canvas[top:bottom, left:right] = 253
        _draw_dotted_grid(
            canvas, bounds, minor_x, major_x, minor_y, major_y
        )
        return canvas
    else:
        raise ValueError(f"unknown paper style {style!r}")

    canvas[top:bottom, left:right] = base
    for x in minor_x:
        cv2.line(canvas, (x, top), (x, bottom - 1), minor, 1, cv2.LINE_AA)
    for y in minor_y:
        cv2.line(canvas, (left, y), (right - 1, y), minor, 1, cv2.LINE_AA)
    for x in major_x:
        cv2.line(canvas, (x, top), (x, bottom - 1), major, 1, cv2.LINE_AA)
    for y in major_y:
        cv2.line(canvas, (left, y), (right - 1, y), major, 1, cv2.LINE_AA)
    return canvas


def _draw_grid_mask(profile: PageProfile, style: str) -> np.ndarray:
    """Draw a soft single-channel mask using exactly the paper grid geometry."""
    height, width = profile.height, profile.width
    left, right, top, bottom = _pixel_bounds(profile)
    minor_x, major_x, minor_y, major_y = grid_line_positions(profile)
    mask = np.zeros((height, width), dtype=np.uint8)
    minor_value = 72 if style != "dotted" else 96
    major_value = 144 if style != "dotted" else 192

    if style == "dotted":
        minor_spacing = int(round(float(np.median(np.diff(minor_x)))))
        dot_gap = max(3, minor_spacing // 3)
        for x in minor_x:
            mask[top:bottom:dot_gap, x] = minor_value
        for y in minor_y:
            mask[y, left:right:dot_gap] = minor_value
        for x in major_x:
            mask[top:bottom:dot_gap, x:min(x + 2, right)] = major_value
        for y in major_y:
            mask[y:min(y + 2, bottom), left:right:dot_gap] = major_value
        return mask

    for x in minor_x:
        cv2.line(mask, (x, top), (x, bottom - 1), minor_value, 1, cv2.LINE_AA)
    for y in minor_y:
        cv2.line(mask, (left, y), (right - 1, y), minor_value, 1, cv2.LINE_AA)
    for x in major_x:
        cv2.line(mask, (x, top), (x, bottom - 1), major_value, 1, cv2.LINE_AA)
    for y in major_y:
        cv2.line(mask, (left, y), (right - 1, y), major_value, 1, cv2.LINE_AA)
    return mask


def draw_paper_with_grid_mask(
    profile: PageProfile,
    pixels_per_mm: float,
    style: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the rendered paper and its aligned soft grid target."""
    return draw_paper(profile, pixels_per_mm, style), _draw_grid_mask(profile, style)


def _put_cv_text(
    image: np.ndarray,
    text_mask: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        text_mask,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        255,
        thickness,
        cv2.LINE_AA,
    )


def draw_footer(
    image: np.ndarray,
    profile: PageProfile,
    style: str,
    text_mask: np.ndarray | None = None,
    layout_text: str = "4 x 2.5s + 3 rhythm leads",
) -> None:
    """Add the small device-style footer seen below the last rhythm strip."""
    left, right, _, bottom = _pixel_bounds(profile)
    span = right - left
    scale = max(0.35, profile.width / 6600.0)
    footer_height = max(9, int(round(24 * scale)))
    space_below = profile.height - bottom
    if space_below > 2 * footer_height:
        y = min(profile.height - 8, bottom + space_below // 2)
    else:
        y = min(profile.height - 8, bottom - 7)
    color = (150, 88, 170) if style in {"pink", "red"} else (85, 85, 85)
    labels = (
        (0.005, "150 Hz"),
        (0.090, "25.0 mm/s    10.0 mm/mV"),
        (0.315, "marquette electronics, inc."),
        (0.610, layout_text),
        (0.840, "MAC55 009C"),
    )
    if text_mask is None:
        text_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for fraction, text in labels:
        _put_cv_text(
            image,
            text_mask,
            text,
            (left + int(round(fraction * span)), y),
            scale,
            color,
            1,
        )


def resample(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == count:
        return values
    old_x = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, count, dtype=np.float32)
    return np.interp(new_x, old_x, values).astype(np.float32)


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


# How neighbouring leads inside one printed row are separated. Real recorders
# either run the pen straight through, lift it briefly, or print a 1 mV
# separator pulse between the columns.
TRANSITIONS = ("continuous", "gap", "pulse")


def row_parts(row: Sequence[np.ndarray] | np.ndarray) -> list[np.ndarray]:
    """Accept a row either as one concatenated signal or as per-lead parts."""
    if isinstance(row, np.ndarray):
        return [row] if row.ndim == 1 else [np.asarray(part) for part in row]
    if isinstance(row, (list, tuple)):
        return [np.asarray(part, dtype=np.float32) for part in row]
    return [np.asarray(row, dtype=np.float32)]


def draw_separator_pulse(
    image: np.ndarray,
    text_mask: np.ndarray,
    left: int,
    right: int,
    baseline: float,
    pixels_per_mm: float,
    pulse_height: int,
    thickness: int,
    color: tuple[int, int, int] = (28, 28, 28),
    entry_y: int | None = None,
    exit_y: int | None = None,
) -> None:
    """Print the 1 mV separator between two lead columns.

    ``entry_y``/``exit_y`` are the last and first trace pixels either side, so
    the pen visibly drops to the baseline and climbs back out again. The pulse
    is a control mark, so it is supervised in the text/control channel and
    never in the physiological trace channel.
    """
    y = int(round(baseline))
    lead_in = max(1, int(round(0.5 * pixels_per_mm)))
    bar = max(2, int(round(1.8 * pixels_per_mm)))
    start = min(left + lead_in, max(left, right - bar - 1))
    stop = min(start + bar, right)
    points = [
        (left, y if entry_y is None else entry_y),
        (left, y), (start, y), (start, y - pulse_height),
        (stop, y - pulse_height), (stop, y), (right, y),
    ]
    if exit_y is not None:
        points.append((right, exit_y))
    polyline = np.array(points, dtype=np.int32)
    cv2.polylines(image, [polyline], False, color, thickness, cv2.LINE_AA)
    cv2.polylines(text_mask, [polyline], False, 255, thickness, cv2.LINE_AA)


def row_trace_pieces(
    parts: Sequence[np.ndarray],
    width: int,
    height: int,
    x0: int,
    x1: int,
    baseline: float,
    pixels_per_mv: float,
    transition: str,
    break_px: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Split one printed row into the polylines the pen actually leaves.

    For ``gap`` and ``pulse`` each lead is resampled across its own full column
    so the 25 mm/s time base stays exact, and only the tail pixels the break
    covers are dropped -- a pen lift hides samples, it does not compress them.
    """
    if transition not in TRANSITIONS:
        raise ValueError(f"unknown transition {transition!r}")
    if transition == "continuous" or len(parts) == 1:
        return [sparse_fractional_trace(
            np.concatenate([np.asarray(part) for part in parts]),
            width, height, x0, x1, baseline, pixels_per_mv,
        )]
    edges = np.rint(np.linspace(x0, x1, len(parts) + 1)).astype(int)
    pieces = []
    for index, part in enumerate(parts):
        xs, ys, values, points = sparse_fractional_trace(
            np.asarray(part), width, height, int(edges[index]),
            int(edges[index + 1]), baseline, pixels_per_mv,
        )
        if index < len(parts) - 1:
            stop = max(int(edges[index]) + 2, int(edges[index + 1]) - break_px)
            keep = xs < stop
            xs, ys, values = xs[keep], ys[keep], values[keep]
            points = points[points[:, 0] < stop]
        if len(points) >= 2:
            pieces.append((xs, ys, values, points))
    return pieces


def draw_calibration(
    image: np.ndarray,
    profile: PageProfile,
    x0: int,
    baseline: float,
    pixels_per_mm: float,
    thickness: int,
    first_trace_y: int,
    text_mask: np.ndarray | None = None,
) -> None:
    y = int(round(baseline))
    pulse_height = int(round(10 * pixels_per_mm))
    xs = [x0 + int(round(value * pixels_per_mm)) for value in profile.calibration_mm]
    ys = (y, y, y - pulse_height, y - pulse_height, y, y, first_trace_y)
    points = np.column_stack((xs, ys)).astype(np.int32)
    cv2.polylines(image, [points], False, (28, 28, 28), thickness, cv2.LINE_AA)
    if text_mask is not None:
        cv2.polylines(text_mask, [points], False, 255, thickness, cv2.LINE_AA)


def compose_rgb_mask(
    grid_mask: np.ndarray,
    text_mask: np.ndarray,
    signal_mask: np.ndarray,
) -> np.ndarray:
    """Compose exclusive soft RGB targets with signal > text > grid priority."""
    signal = signal_mask.astype(np.float32) / 255.0
    text = text_mask.astype(np.float32) / 255.0
    grid = grid_mask.astype(np.float32) / 255.0
    text *= 1.0 - signal
    # Allocate grid only from the probability mass left after the higher
    # priority classes. This keeps R+G+B <= 255 even after perspective
    # interpolation creates partially overlapping soft edges.
    grid *= np.clip(1.0 - signal - text, 0.0, 1.0)
    return np.rint(np.stack((grid, text, signal), axis=2) * 255.0).astype(np.uint8)


def _empty_coordinates(
    count: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [
        (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )
        for _ in range(count)
    ]


def render_ecg_page(
    rows: Sequence[np.ndarray],
    row_names: Sequence[str],
    profile: PageProfile,
    style: str,
    *,
    is_blank: bool = False,
    amplitude_scale: float = 1.0,
    footer_layout_text: str = "4 x 2.5s + 3 rhythm leads",
    transition: str = "continuous",
    break_px: int = 0,
) -> RenderedPage:
    """Render rows and exact sparse targets with private-scan geometry.

    ``rows`` may hold one concatenated signal per printed row, or a sequence of
    per-lead parts. Parts are required for the ``gap`` and ``pulse``
    transitions, which need to know where one lead column ends.
    """
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
    image, grid_mask = draw_paper_with_grid_mask(profile, pixels_per_mm, style)
    text_mask = np.zeros((height, width), dtype=np.uint8)
    signal_mask = np.zeros((height, width), dtype=np.uint8)
    draw_footer(image, profile, style, text_mask, footer_layout_text)
    coordinates = _empty_coordinates(len(rows))
    thickness = max(1, int(round(width / 1750)))

    if not is_blank:
        coordinates = []
        pulse_height = int(round(10 * pixels_per_mm))
        for row, baseline in zip(rows, baselines):
            parts = [part * amplitude_scale for part in row_parts(row)]
            pieces = row_trace_pieces(
                parts, width, height, x0, x1, baseline, pixels_per_mv,
                transition, break_px,
            )
            coordinates.append(tuple(
                np.concatenate([piece[channel] for piece in pieces])
                for channel in range(3)
            ))
            for *_, points in pieces:
                cv2.polylines(
                    image, [points], False, (24, 24, 24), thickness, cv2.LINE_AA
                )
                cv2.polylines(
                    signal_mask, [points], False, 255, thickness, cv2.LINE_AA
                )
            if transition == "pulse":
                for index in range(len(pieces) - 1):
                    entry, exit_points = pieces[index][3], pieces[index + 1][3]
                    draw_separator_pulse(
                        image, text_mask, int(entry[-1, 0]),
                        int(exit_points[0, 0]), baseline, pixels_per_mm,
                        pulse_height, thickness, (28, 28, 28),
                        int(entry[-1, 1]), int(exit_points[0, 1]),
                    )
            draw_calibration(
                image,
                profile,
                x0,
                baseline,
                pixels_per_mm,
                thickness,
                int(pieces[0][3][0, 1]),
                text_mask,
            )

        label_scale = max(0.45, width / 5000.0)
        row_gap = float(np.median(np.diff(baselines)))
        for channel, name in enumerate(row_names):
            leads = name.split("|")
            if len(leads) > 1:
                for column, lead in enumerate(leads):
                    label_x = x0 + int((x1 - x0) * column / len(leads)) + 3
                    label_y = int(round(baselines[channel] - 0.18 * row_gap))
                    _put_cv_text(
                        image,
                        text_mask,
                        lead,
                        (label_x, max(15, label_y)),
                        label_scale,
                        (35, 35, 35),
                        1,
                    )
            else:
                _put_cv_text(
                    image,
                    text_mask,
                    name,
                    (x0 + 3, int(baselines[channel] - 8)),
                    label_scale,
                    (35, 35, 35),
                    1,
                )

    mask_rgb = compose_rgb_mask(grid_mask, text_mask, signal_mask)
    return RenderedPage(
        image=image,
        mask_rgb=mask_rgb,
        grid_mask=grid_mask,
        text_mask=text_mask,
        signal_mask=signal_mask,
        coordinates=coordinates,
        x0=x0,
        x1=x1,
        baselines=baselines,
        pixels_per_mm=pixels_per_mm,
        pixels_per_mv=pixels_per_mv,
        profile=profile,
    )
