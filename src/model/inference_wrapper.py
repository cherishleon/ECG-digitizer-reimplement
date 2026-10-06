import time
from contextlib import contextmanager
from typing import Any, Generator

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.nn import Module
from yacs.config import CfgNode as CN

from src.utils import import_class_from_path


@contextmanager
def timed_section(name: str, times_dict: dict[str, float]) -> Generator[None, None, None]:
    """Context manager for timing code blocks.

    Args:
        name: Name of the section.
        times_dict: Dictionary to store timing.
    """
    start = time.time()
    yield
    times_dict[name] = time.time() - start


class InferenceWrapper(Module):
    def __init__(
        self,
        config: CN,
        device: str,
        resample_size: None | tuple[int, ...] = None,
        grid_class: int = 0,
        text_background_class: int = 1,
        signal_class: int = 2,
        background_class: int = 3,
        rotate_on_resample: bool = False,
        enable_timing: bool = False,
        minimum_image_size: int = 512,
        apply_dewarping: bool = True,
        grid_content_threshold: float = 0.30,
        signal_content_threshold: float = 0.30,
        min_active_pixels: int = 3,
        max_overshoot_fraction: float = 0.50,
        max_gap_fraction: float = 0.02,
        content_padding_fraction: float = 0.02,
        remove_calibration_pulse: bool = True,
        calibration_search_mm: float = 20.0,
        calibration_min_run_mm: float = 6.0,
        calibration_min_rows: int = 3,
        calibration_margin_mm: float = 0.5,
    ) -> None:
        """Inference wrapper for ECG pipeline.

        Args:
            config: Configuration node.
            device: Torch device string.
            resample_size: Optional resample target size.
            grid_class: Grid class index.
            text_background_class: Text and background class index.
            signal_class: Signal class index.
            background_class: Background class index.
            rotate_on_resample: Whether to rotate on resample.
            enable_timing: Whether to print timings.
            minimum_image_size: Minimum allowed image size.
            apply_dewarping: Whether to apply dewarping (perspective correction is still performed regardless).
            grid_content_threshold: Fraction of the typical (75th percentile) grid density a row/column must reach
                to count as printed paper. Sets the anchor extent of the post-warp crop.
            signal_content_threshold: Probability level at which a pixel counts as signal. Applied per pixel, not
                to the row/column sum: a row-sum criterion is dominated by long flat baselines and would rate a
                narrow tall R peak far below any threshold that still excludes background.
            min_active_pixels: Pixels above that level needed for a row/column to count as carrying signal. Keeps
                isolated segmentation noise from extending the crop.
            max_overshoot_fraction: Hard cap on how far, as a fraction of the grid extent, the crop may follow
                signal beyond the grid.
            max_gap_fraction: Longest run of signal-free rows/columns, as a fraction of the grid extent, that the
                crop will step over while following signal outward. This is what separates a tall R peak (joined
                to the trace inside the grid) from header text (separated from it by white space).
            content_padding_fraction: Margin kept around the final crop, as a fraction of its extent.
            remove_calibration_pulse: Whether to blank the 1 mV calibration pulse printed at the left of each
                row before extracting signals.
            calibration_search_mm: How far in from the left of the signal content to look for the pulse.
            calibration_min_run_mm: Vertical millimetres of unbroken ink that count as one stroke of the pulse.
            calibration_min_rows: Strokes a column must stack before it is accepted as the pulse's falling edge.
                The pulse is printed once per row, so its edge stacks one stroke per row; a QRS spike, however
                tall, stacks one or two.
            calibration_margin_mm: Extra margin blanked to the right of the detected falling edge.
        """
        super().__init__()
        self.config = config
        self.device = device
        self.resample_size = resample_size
        self.grid_class = grid_class
        self.text_background_class = text_background_class
        self.signal_class = signal_class
        self.background_class = background_class
        self.rotate_on_resample = rotate_on_resample
        self._timing_enabled = enable_timing
        self.minimum_image_size = minimum_image_size
        self.apply_dewarping = apply_dewarping
        self.grid_content_threshold = grid_content_threshold
        self.signal_content_threshold = signal_content_threshold
        self.min_active_pixels = min_active_pixels
        self.max_overshoot_fraction = max_overshoot_fraction
        self.max_gap_fraction = max_gap_fraction
        self.content_padding_fraction = content_padding_fraction
        self.remove_calibration_pulse = remove_calibration_pulse
        self.calibration_search_mm = calibration_search_mm
        self.calibration_min_run_mm = calibration_min_run_mm
        self.calibration_min_rows = calibration_min_rows
        self.calibration_margin_mm = calibration_margin_mm

        self.signal_extractor = self._load_signal_extractor()
        self.perspective_detector: Any = self._load_perspective_detector()
        self.segmentation_model: Any = self._load_segmentation_model().to(self.device)
        self.cropper: Any = self._load_cropper()
        self.pixel_size_finder: Any = self._load_pixel_size_finder()
        self.dewarper: Any = self._load_dewarper()
        self.identifier = self._load_layout_identifier()
        self.times: dict[str, float] = {}

    @torch.no_grad()
    def forward(
        self, image: Tensor, layout_should_include_substring: None | str
    ) -> dict[str, Tensor | str | float | None | dict[str, Any]]:
        """Performs full inference on an input image.

        Args:
            image: Input image tensor.
            layout_should_include_substring: Optional substring to filter layout names.

        Returns:
            Dictionary with processed outputs and intermediate results.
        """
        self._check_image_dimensions(image)
        image = self.min_max_normalize(image)
        image = image.to(self.device)

        self.times = {}
        image = self._resample_image(image)

        signal_prob, grid_prob, text_prob = self._get_feature_maps(image)

        with timed_section("Perspective detection", self.times):
            alignment_params = self.perspective_detector(grid_prob)

        with timed_section("Cropping", self.times):
            source_points = self.cropper(signal_prob, alignment_params, grid_prob)

        aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob = self._align_feature_maps(
            image, signal_prob, grid_prob, text_prob, source_points
        )

        with timed_section("Pixel size search", self.times):
            mm_per_pixel_x, mm_per_pixel_y = self.pixel_size_finder(aligned_grid_prob)
            avg_pixel_per_mm = (1 / mm_per_pixel_x + 1 / mm_per_pixel_y) / 2

        with timed_section("Calibration pulse removal", self.times):
            aligned_signal_prob, calibration_pulse_x = self._blank_calibration_pulse(
                aligned_signal_prob, 1 / mm_per_pixel_x, 1 / mm_per_pixel_y
            )

        with timed_section("Dewarping", self.times):
            if self.apply_dewarping:
                self.dewarper.fit(aligned_grid_prob.squeeze(), avg_pixel_per_mm)
                aligned_signal_prob = self.dewarper.transform(aligned_signal_prob.squeeze())

        with timed_section("Signal extraction", self.times):
            signals = self.signal_extractor(aligned_signal_prob.squeeze())

        self._print_profiling_results()

        layout = self.identifier(
            signals,
            aligned_text_prob,
            avg_pixel_per_mm,
            layout_should_include_substring=layout_should_include_substring,
        )
        # `flip` is absent whenever the identifier bailed out before matching (too few lead names
        # detected). Reading it with [] used to raise, and the handler then reported a fabricated
        # cost of 1.0 -- right at the threshold most triage uses -- for what is a total failure.
        layout_str = str(layout.get("layout", "Unknown layout"))
        layout_is_flipped = str(layout.get("flip", False))
        layout_cost = layout.get("cost", float("nan"))

        return {
            "layout_name": layout_str,
            "input_image": image.cpu(),
            "aligned": {
                "image": aligned_image.cpu(),
                "signal_prob": aligned_signal_prob.cpu(),
                "grid_prob": aligned_grid_prob.cpu(),
                "text_prob": aligned_text_prob.cpu(),
            },
            "signal": {
                "raw_lines": signals.cpu(),
                "canonical_lines": layout.get("canonical_lines", None),
                "lines": layout.get("lines", None),
                "layout_matching_cost": layout_cost,
                "layout_is_flipped": layout_is_flipped,
            },
            "pixel_spacing_mm": {
                "x": mm_per_pixel_x,
                "y": mm_per_pixel_y,
                "average_pixel_per_mm": avg_pixel_per_mm,
            },
            # Raw quality signals. `layout_matching_cost` folds these into one number via a scaling
            # factor, so it cannot be read back apart from them; triage wants them separately.
            "n_detected_lead_names": layout.get("n_detected", None),
            "rows_in_layout": layout.get("rows_in_layout", None),
            "calibration_pulse_x": calibration_pulse_x,
            "source_points": source_points.cpu(),
        }

    def _align_feature_maps(
        self,
        image: Tensor,
        signal_prob: Tensor,
        grid_prob: Tensor,
        text_prob: Tensor,
        source_points: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Aligns image and feature maps using perspective cropping.

        Returns:
            Aligned image, signal, grid, and text tensors.
        """
        with timed_section("Feature map resampling", self.times):
            aligned_signal_prob = self.cropper.apply_perspective(signal_prob, source_points, fill_value=0)
            aligned_image = self.cropper.apply_perspective(image, source_points, fill_value=0)
            aligned_grid_prob = self.cropper.apply_perspective(grid_prob, source_points, fill_value=0)
            aligned_text_prob = self.cropper.apply_perspective(text_prob, source_points, fill_value=0)
            if self.rotate_on_resample:
                aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob = self._rotate_on_resample(
                    aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob
                )
            aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob = self._crop_y(
                aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob
            )

            return aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob

    def _crop_y(
        self, image: Tensor, signal_prob: Tensor, grid_prob: Tensor, text_prob: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Crops tensors in y and x to the printed content.

        The extent is anchored on the grid, which is the only reliable geometric boundary on the page, and then
        walked outward for as long as signal keeps appearing -- tall R peaks are routinely drawn above the top
        grid line and used to be cut here. The walk stops at the first run of blank rows longer than
        `max_gap_fraction`, which is what keeps detached header/footer text out of the crop even when it leaks
        into the signal class. Cropping in x as well removes the zero-fill introduced by the perspective warp,
        which would otherwise skew the grid-line count assumed by the pixel size search.

        Returns:
            Cropped image, signal, grid, and text tensors.
        """
        y1, y2 = self._content_bounds(signal_prob, grid_prob, axis=-2)
        x1, x2 = self._content_bounds(signal_prob, grid_prob, axis=-1)

        slices = (slice(None), slice(None), slice(y1, y2 + 1), slice(x1, x2 + 1))
        return image[slices], signal_prob[slices], grid_prob[slices], text_prob[slices]

    def _content_bounds(self, signal_prob: Tensor, grid_prob: Tensor, axis: int) -> tuple[int, int]:
        """Finds the first and last index along `axis` that carries printed content.

        Args:
            signal_prob: Signal probability map, shape (1, 1, H, W).
            grid_prob: Grid probability map, shape (1, 1, H, W).
            axis: -2 for row bounds, -1 for column bounds.

        Returns:
            Inclusive (first, last) index along the axis.
        """
        size = int(signal_prob.shape[axis])
        reduce_dim = -1 if axis == -2 else -2

        def extent(tensor: Tensor, relative_threshold: float, use_quantile_reference: bool) -> tuple[int, int] | None:
            profile = tensor.squeeze()
            while profile.dim() > 2:
                profile = profile.sum(0)
            profile = profile.sum(dim=reduce_dim).float()
            profile = profile - profile.min()
            # A quantile reference ignores individual over-dense rows (a thick border line, a dense text
            # block); the peak is the right reference when the content is sparse by nature.
            reference = torch.quantile(profile, 0.75) if use_quantile_reference else profile.max()
            if not bool(torch.isfinite(reference)) or reference <= 0:
                return None
            above = (profile >= relative_threshold * reference).nonzero(as_tuple=True)[0]
            if above.numel() == 0:
                return None
            return int(above[0].item()), int(above[-1].item())

        grid_extent = extent(grid_prob, self.grid_content_threshold, use_quantile_reference=True)
        signal_active = self._active_mask(signal_prob, reduce_dim)

        if grid_extent is None:
            if signal_active is None:
                return 0, size - 1
            present = [i for i, flag in enumerate(signal_active) if flag]
            low, high = present[0], present[-1]
        else:
            low, high = grid_extent
            if signal_active is not None:
                span = high - low + 1
                limit = int(self.max_overshoot_fraction * span)
                max_gap = int(self.max_gap_fraction * span)
                low = self._follow_signal(signal_active, low, -1, max_gap, limit)
                high = self._follow_signal(signal_active, high, 1, max_gap, limit)

        pad = int(self.content_padding_fraction * (high - low + 1))
        return max(0, low - pad), min(size - 1, high + pad)

    def _active_mask(self, signal_prob: Tensor, reduce_dim: int) -> list[bool] | None:
        """Per-index flags for where the signal map carries anything, along the axis kept by `reduce_dim`.

        Counts pixels above `signal_content_threshold` rather than summing probabilities, so a six-pixel-wide R
        peak registers as strongly as a full-width baseline.
        """
        probabilities = signal_prob.squeeze()
        while probabilities.dim() > 2:
            probabilities = probabilities.amax(0)
        if probabilities.dim() != 2:
            return None
        counts = (probabilities >= self.signal_content_threshold).sum(dim=reduce_dim)
        if int(counts.max().item()) < self.min_active_pixels:
            return None
        return (counts >= self.min_active_pixels).tolist()

    def _follow_signal(self, active: list[bool], start: int, direction: int, max_gap: int, limit: int) -> int:
        """Walks outward from `start` for as long as signal keeps appearing.

        Runs of up to `max_gap` inactive indices are stepped over, so a peak that briefly thins out is still
        followed; a longer gap ends the walk, which is what keeps detached header/footer text out of the crop.

        Returns:
            The furthest active index reached, or `start` if nothing lies beyond it.
        """
        furthest = start
        gap = 0
        index = start + direction
        while 0 <= index < len(active) and abs(index - start) <= limit:
            if active[index]:
                furthest = index
                gap = 0
            else:
                gap += 1
                if gap > max_gap:
                    break
            index += direction
        return furthest

    def _blank_calibration_pulse(
        self, signal_prob: Tensor, pixel_per_mm_x: float, pixel_per_mm_y: float
    ) -> tuple[Tensor, int | None]:
        """Blanks the 1 mV calibration pulse printed at the left of every row.

        The pulse is not ECG, but it is drawn in the signal ink and sits left of the first trace, so it reaches
        the extractor as part of the same connected component: it corrupts the centroid in its own columns, can
        surface as an extra row, drags the left edge used to slice the page into columns, and biases the
        per-row mean removal.

        Detection keys on the pulse's falling edge, and specifically on how many rows it stacks: the pulse is
        printed once per row, so its edge is a single column carrying one tall unbroken stroke per row, while a
        QRS spike -- which can be just as tall in ink terms -- carries one or two. Counting strokes rather than
        ink is what separates the two; on the sample pages the falling edge stacks 3-5 strokes and the tallest
        QRS column nearby stacks at most 3. Nothing but the pulse lies left of that edge, so the whole strip up
        to it is blanked.

        Returns:
            The map with the pulse blanked, and the column blanked up to (None if no pulse was found).
        """
        if not self.remove_calibration_pulse:
            return signal_prob, None
        # PixelSizeFinder reports its estimates as 0-dim tensors.
        pixel_per_mm_x = float(pixel_per_mm_x)
        pixel_per_mm_y = float(pixel_per_mm_y)
        if not (pixel_per_mm_x > 0 and pixel_per_mm_y > 0):
            return signal_prob, None

        probabilities = signal_prob.squeeze()
        while probabilities.dim() > 2:
            probabilities = probabilities.amax(0)
        if probabilities.dim() != 2:
            return signal_prob, None

        ink = (probabilities >= self.signal_content_threshold).float()
        occupied = (ink.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
        if occupied.numel() == 0:
            return signal_prob, None

        minimum_run = int(round(self.calibration_min_run_mm * pixel_per_mm_y))
        window = int(self.calibration_search_mm * pixel_per_mm_x)
        if minimum_run < 1 or window < 1 or minimum_run > ink.shape[0]:
            return signal_prob, None

        # A sliding window down each column is fully inked wherever a stroke of at least `minimum_run`
        # covers it; the number of strokes in the column is then the number of times that becomes true.
        kernel = torch.ones(1, 1, minimum_run, 1, device=ink.device, dtype=ink.dtype)
        covered = F.conv2d(ink[None, None], kernel)[0, 0] >= minimum_run
        starts = covered.clone()
        starts[1:] &= ~covered[:-1]
        strokes = starts.sum(dim=0)

        left = int(occupied[0].item())
        searched = strokes[left : left + window]
        if searched.numel() == 0:
            return signal_prob, None
        most_strokes = int(searched.max().item())
        if most_strokes < self.calibration_min_rows:
            return signal_prob, None

        # Leftmost column reaching the maximum: a falling edge two pixels wide would otherwise let the
        # cut drift right, and the margin below already covers the edge's own width.
        edge_column = left + int((searched == most_strokes).nonzero(as_tuple=True)[0][0].item())
        edge = edge_column + int(round(self.calibration_margin_mm * pixel_per_mm_x))
        edge = min(edge, int(probabilities.shape[1]) - 1)

        blanked = signal_prob.clone()
        blanked[..., : edge + 1] = 0
        return blanked, edge

    def _print_profiling_results(self) -> None:
        """Prints the timings for each timed section."""
        if not self._timing_enabled:
            return
        print(" Timing results:")
        max_length = max(len(section) for section in self.times.keys())
        for section, duration in self.times.items():
            print(f"    {section:<{max_length+2}}{duration:.2f} s")
        total_time = sum(self.times.values())
        print(f"Total time: {total_time:.2f} s")

    def _rotate_on_resample(
        self,
        aligned_image: Tensor,
        aligned_signal_prob: Tensor,
        aligned_grid_prob: Tensor,
        aligned_text_prob: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Rotates all tensors if height > width.

        Returns:
            Rotated tensors in same order.
        """
        if aligned_image.shape[2] > aligned_image.shape[3]:
            aligned_image = torch.rot90(aligned_image, k=3, dims=(2, 3))
            aligned_signal_prob = torch.rot90(aligned_signal_prob, k=3, dims=(2, 3))
            aligned_grid_prob = torch.rot90(aligned_grid_prob, k=3, dims=(2, 3))
            aligned_text_prob = torch.rot90(aligned_text_prob, k=3, dims=(2, 3))
        return aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob

    def _resample_image(self, image: Tensor) -> Tensor:
        with timed_section("Initial resampling", self.times):
            if self.resample_size is None:
                return image

            height, width = image.shape[2], image.shape[3]
            min_dim = min(height, width)
            max_dim = max(height, width)

            if min_dim < self.minimum_image_size:
                scale: float = self.minimum_image_size / min_dim
                new_size: tuple[int, int] = (int(height * scale), int(width * scale))
                interpolated: Tensor = F.interpolate(image, size=new_size, mode="bilinear", align_corners=False)
                return interpolated

            if isinstance(self.resample_size, int):
                if max_dim > self.resample_size:
                    scale = self.resample_size / max_dim
                    new_size = (int(height * scale), int(width * scale))
                    return F.interpolate(image, size=new_size, mode="bilinear", align_corners=False, antialias=True)
                return image

            if isinstance(self.resample_size, tuple):
                interpolated = F.interpolate(
                    image, size=self.resample_size, mode="bilinear", align_corners=False, antialias=True
                )
                return interpolated

            raise ValueError(f"Invalid resample_size: {self.resample_size}. Expected int or tuple of (height, width).")

    def process_sparse_prob(self, signal_prob: Tensor) -> Tensor:
        signal_prob = signal_prob - signal_prob.mean() * 1
        signal_prob = torch.clamp(signal_prob, min=0)
        signal_prob = signal_prob / (signal_prob.max() + 1e-9)
        return signal_prob

    def _get_feature_maps(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        with timed_section("Segmentation", self.times):
            logits = self.segmentation_model(image)
            prob = torch.softmax(logits, dim=1)

            signal_prob = prob[:, [self.signal_class], :, :]
            grid_prob = prob[:, [self.grid_class], :, :]
            text_prob = prob[:, [self.text_background_class], :, :]

            signal_prob = self.process_sparse_prob(signal_prob)
            grid_prob = self.process_sparse_prob(grid_prob)
            text_prob = self.process_sparse_prob(text_prob)

            return signal_prob, grid_prob, text_prob

    def min_max_normalize(self, image: Tensor) -> Tensor:
        return (image - image.min()) / (image.max() - image.min())

    def _load_signal_extractor(self) -> Any:
        signal_extractor_class = import_class_from_path(self.config.SIGNAL_EXTRACTOR.class_path)
        extractor: Any = signal_extractor_class(**self.config.SIGNAL_EXTRACTOR.KWARGS)
        return extractor

    def _load_perspective_detector(self) -> Any:
        perspective_detector_class = import_class_from_path(self.config.PERSPECTIVE_DETECTOR.class_path)
        perspective_detector: Any = perspective_detector_class(**self.config.PERSPECTIVE_DETECTOR.KWARGS)
        return perspective_detector

    def _load_segmentation_model(self) -> Any:
        segmentation_model_class = import_class_from_path(self.config.SEGMENTATION_MODEL.class_path)
        segmentation_model: Any = segmentation_model_class(**self.config.SEGMENTATION_MODEL.KWARGS)
        self._load_segmentation_model_weights(segmentation_model)
        return segmentation_model.eval()

    def _load_cropper(self) -> Any:
        cropper_class = import_class_from_path(self.config.CROPPER.class_path)
        cropper: Any = cropper_class(**self.config.CROPPER.KWARGS)
        return cropper

    def _load_pixel_size_finder(self) -> Any:
        pixel_size_finder_class = import_class_from_path(self.config.PIXEL_SIZE_FINDER.class_path)
        pixel_size_finder: Any = pixel_size_finder_class(**self.config.PIXEL_SIZE_FINDER.KWARGS)
        return pixel_size_finder

    def _load_dewarper(self) -> Any:
        dewarper_class = import_class_from_path(self.config.DEWARPER.class_path)
        dewarper: Any = dewarper_class(**self.config.DEWARPER.KWARGS)
        return dewarper

    def _load_layout_identifier(self) -> Any:
        layouts = yaml.safe_load(open(self.config.LAYOUT_IDENTIFIER.config_path, "r"))
        unet_cfg = yaml.safe_load(open(self.config.LAYOUT_IDENTIFIER.unet_config_path, "r"))
        unet_class = import_class_from_path(unet_cfg["MODEL"]["class_path"])
        unet: torch.nn.Module = unet_class(**unet_cfg["MODEL"]["KWARGS"])
        checkpoint = torch.load(self.config.LAYOUT_IDENTIFIER.unet_weight_path, map_location=self.device)
        checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        unet.load_state_dict(checkpoint)
        unet.eval()

        identifier_class = import_class_from_path(self.config.LAYOUT_IDENTIFIER.class_path)
        identifier: Any = identifier_class(
            layouts=layouts,
            unet=unet,
            **self.config.LAYOUT_IDENTIFIER.KWARGS,
        )
        return identifier

    def _load_segmentation_model_weights(self, segmentation_model: torch.nn.Module) -> None:
        """Loads weights for segmentation model.

        Args:
            segmentation_model: The model to load weights into.
        """
        checkpoint = torch.load(self.config.SEGMENTATION_MODEL.weight_path, weights_only=True, map_location=self.device)
        if isinstance(checkpoint, tuple):
            checkpoint = checkpoint[0]
        checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        segmentation_model.load_state_dict(checkpoint)

    def _check_image_dimensions(self, image: Tensor) -> None:
        """Checks input image dimensions.

        Args:
            image: Image tensor.

        Raises:
            NotImplementedError: If batch or channel dims are incorrect.
        """
        if image.dim() != 4:
            raise NotImplementedError(f"Expected 4 dimensions, got tensor with {image.dim()} dimensions")
        if image.shape[0] != 1:
            raise NotImplementedError(f"Batch processing not supported, got tensor with shape {image.shape}")
        if image.shape[1] != 3:
            raise NotImplementedError(f"Expected 3 channels, got tensor with shape {image.shape}")
