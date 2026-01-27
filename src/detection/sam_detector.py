"""SAM (Segment Anything Model) detector for automatic object segmentation."""

import gc
import logging
import time
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

from src.detection.models import BoundingBox, Detection, DetectionResult

logger = logging.getLogger(__name__)

# Maximum image dimension to prevent OOM errors
MAX_IMAGE_DIMENSION = 1024


class SAMDetector:
    """
    SAM (Segment Anything Model) detector for automatic mask generation.

    SAM excels at finding object boundaries without text prompts. It generates
    masks for all objects in an image, which can then be converted to bounding
    boxes. This is useful for discovering plant regions that text-based models
    might miss.
    """

    def __init__(
        self,
        model_name: str = "facebook/sam-vit-base",
        device: str | None = None,
    ):
        """
        Initialize the SAM detector.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on ('cuda', 'mps', 'cpu')
        """
        self.device = device or self._select_device()
        self.model_name = model_name

        # Lazy loading
        self._processor: SamProcessor | None = None
        self._model: SamModel | None = None

    def _select_device(self) -> str:
        """Auto-select the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def processor(self) -> SamProcessor:
        """Lazy-load the processor."""
        if self._processor is None:
            logger.info(f"Loading SAM processor: {self.model_name}")
            self._processor = SamProcessor.from_pretrained(self.model_name)
        return self._processor

    @property
    def model(self) -> SamModel:
        """Lazy-load the model."""
        if self._model is None:
            logger.info(f"Loading SAM model: {self.model_name}")
            self._model = SamModel.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            self._model.eval()
            logger.info(f"SAM model loaded on {self.device}")
        return self._model

    def load_image(
        self,
        source: Union[str, Path, Image.Image],
        max_dimension: int = MAX_IMAGE_DIMENSION,
    ) -> Image.Image:
        """Load an image from path or return if already a PIL Image, resizing if needed."""
        if isinstance(source, Image.Image):
            img = source.convert("RGB")
        else:
            img = Image.open(source).convert("RGB")

        # Resize if too large to prevent OOM
        width, height = img.size
        if width > max_dimension or height > max_dimension:
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return img

    def detect(
        self,
        target_image: Union[str, Path, Image.Image],
        min_mask_area: float = 0.005,
        max_mask_area: float = 0.25,
        points_per_side: int = 16,
    ) -> DetectionResult:
        """
        Detect objects using automatic mask generation.

        SAM generates masks for all objects it finds. We convert these masks
        to bounding boxes and filter by size.

        Args:
            target_image: The image to segment
            min_mask_area: Minimum mask area as fraction of image (filters tiny noise)
            max_mask_area: Maximum mask area as fraction of image (filters huge regions)
            points_per_side: Grid density for automatic point sampling (lower = faster)

        Returns:
            DetectionResult with detected objects (labeled as "object")
        """
        start_time = time.perf_counter()

        # Load target image
        target = self.load_image(target_image)
        target_width, target_height = target.size

        # Generate a grid of input points for automatic segmentation
        # SAM uses these points as prompts to generate masks
        input_points = self._generate_grid_points(
            target_width, target_height, points_per_side
        )

        # Process in batches to avoid OOM
        all_detections: list[Detection] = []
        batch_size = 4  # Process 4 points at a time

        for i in range(0, len(input_points), batch_size):
            batch_points = input_points[i : i + batch_size]

            # Each point needs to be in format [[[x, y]]] for SAM
            for point in batch_points:
                inputs = self.processor(
                    target,
                    input_points=[[[point]]],
                    return_tensors="pt",
                )
                # Convert float64 to float32 for MPS compatibility (MPS doesn't support float64)
                inputs = {
                    k: v.float().to(self.device) if v.dtype == torch.float64 else v.to(self.device)
                    for k, v in inputs.items()
                }

                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Get masks and scores
                masks = self.processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu(),
                )

                scores = outputs.iou_scores.cpu()

                # Process each mask
                for mask_idx in range(masks[0].shape[0]):
                    mask = masks[0][mask_idx].numpy()
                    score = float(scores[0][mask_idx].max())

                    # Skip low-confidence masks
                    if score < 0.5:
                        continue

                    # Find bounding box from mask
                    bbox = self._mask_to_bbox(mask, target_width, target_height)
                    if bbox is None:
                        continue

                    # Filter by size
                    area = bbox.width * bbox.height
                    if area < min_mask_area or area > max_mask_area:
                        continue

                    detection = Detection(
                        label="plant region",
                        confidence=score,
                        box=bbox,
                    )
                    all_detections.append(detection)

                # Clean up batch
                del inputs, outputs, masks, scores

        # Clean up
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # Deduplicate overlapping boxes (SAM often finds overlapping regions)
        result = DetectionResult(
            detections=all_detections,
            image_width=target_width,
            image_height=target_height,
            inference_time_ms=0,  # Will be set after dedup
        )

        # Aggressive deduplication since SAM finds many overlapping masks
        result = result.deduplicate(min_distance=0.15)

        # Sort by confidence
        result.detections.sort(key=lambda d: d.confidence, reverse=True)

        inference_time = (time.perf_counter() - start_time) * 1000

        return DetectionResult(
            detections=result.detections,
            image_width=target_width,
            image_height=target_height,
            inference_time_ms=inference_time,
        )

    def _generate_grid_points(
        self, width: int, height: int, points_per_side: int
    ) -> list[tuple[int, int]]:
        """Generate a grid of points for automatic mask generation."""
        points = []
        x_step = width / (points_per_side + 1)
        y_step = height / (points_per_side + 1)

        for i in range(1, points_per_side + 1):
            for j in range(1, points_per_side + 1):
                x = int(i * x_step)
                y = int(j * y_step)
                points.append((x, y))

        return points

    def _mask_to_bbox(
        self, mask, img_width: int, img_height: int
    ) -> BoundingBox | None:
        """Convert a binary mask to a normalized bounding box."""
        # Handle different mask shapes
        if len(mask.shape) == 3:
            mask = mask[0]  # Take first channel

        # Find nonzero pixels
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        y_min, y_max = row_indices[0], row_indices[-1]
        x_min, x_max = col_indices[0], col_indices[-1]

        # Normalize to 0-1 range
        return BoundingBox(
            x_min=max(0.0, min(1.0, x_min / img_width)),
            y_min=max(0.0, min(1.0, y_min / img_height)),
            x_max=max(0.0, min(1.0, x_max / img_width)),
            y_max=max(0.0, min(1.0, y_max / img_height)),
        )

    def warmup(self) -> None:
        """Warm up the model with a dummy inference."""
        dummy = Image.new("RGB", (224, 224), color="green")
        self.detect(
            target_image=dummy,
            points_per_side=4,
        )
