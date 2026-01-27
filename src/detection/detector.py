"""OWLv2-based weed detector using few-shot learning."""

import gc
import time
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from src.config import settings
from src.detection.models import BoundingBox, Detection, DetectionResult

# Maximum image dimension to prevent OOM errors on MPS/GPU
MAX_IMAGE_DIMENSION = 1024
MAX_REFERENCE_DIMENSION = 384


class WeedDetector:
    """
    Few-shot weed detector using OWLv2.

    OWLv2 performs "image-guided detection" - you provide reference images
    of what you want to find, and it locates similar objects in target images.
    No training required; just good reference images.
    """

    def __init__(
        self,
        model_name: str = settings.model_name,
        device: str | None = None,
    ):
        """
        Initialize the detector.

        Args:
            model_name: HuggingFace model identifier for OWLv2
            device: Device to run inference on ('cuda', 'mps', 'cpu')
        """
        self.device = device or self._select_device()
        self.model_name = model_name

        # Lazy loading - model loaded on first use
        self._processor: Owlv2Processor | None = None
        self._model: Owlv2ForObjectDetection | None = None

    def _select_device(self) -> str:
        """Auto-select the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def processor(self) -> Owlv2Processor:
        """Lazy-load the processor."""
        if self._processor is None:
            self._processor = Owlv2Processor.from_pretrained(self.model_name)
        return self._processor

    @property
    def model(self) -> Owlv2ForObjectDetection:
        """Lazy-load the model."""
        if self._model is None:
            self._model = Owlv2ForObjectDetection.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            self._model.eval()
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
        reference_images: dict[str, list[Union[str, Path, Image.Image]]],
        confidence_threshold: float = settings.default_confidence_threshold,
    ) -> DetectionResult:
        """
        Detect weeds in target image using reference images.

        Args:
            target_image: The image to search for weeds in
            reference_images: Dict mapping weed labels to lists of reference images
                Example: {"dandelion": [img1, img2, ...], "clover": [img1, ...]}
            confidence_threshold: Minimum confidence for detections

        Returns:
            DetectionResult with all detected weeds above threshold
        """
        start_time = time.perf_counter()

        # Load target image
        target = self.load_image(target_image)
        target_width, target_height = target.size

        all_detections: list[Detection] = []

        # Process each weed type separately
        # OWLv2 image-guided detection works best with focused queries
        for label, refs in reference_images.items():
            if not refs:
                continue

            # Load reference images for this weed type (use smaller size for refs)
            # Limit to max 5 reference images to reduce memory usage
            refs_to_use = refs[:5]
            query_images = [
                self.load_image(ref, max_dimension=MAX_REFERENCE_DIMENSION)
                for ref in refs_to_use
            ]

            # Process inputs for the model
            inputs = self.processor(
                images=target,
                query_images=query_images,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.model.image_guided_detection(**inputs)

            # Post-process results
            # target_sizes needs to match batch dimension (one per query image)
            num_queries = len(query_images)
            target_sizes = torch.tensor(
                [[target_height, target_width]] * num_queries, device=self.device
            )

            results = self.processor.post_process_image_guided_detection(
                outputs=outputs,
                threshold=confidence_threshold,
                nms_threshold=0.2,  # Aggressive NMS to reduce overlapping boxes
                target_sizes=target_sizes,
            )

            # Extract detections from results
            # Results is a list with one entry per query image - merge all
            if results:
                for result in results:
                    boxes = result["boxes"]
                    scores = result["scores"]

                    for box, score in zip(boxes, scores):
                        # Boxes are in pixel coords, convert to normalized and clamp to 0-1
                        x_min, y_min, x_max, y_max = box.tolist()
                        detection = Detection(
                            label=label,
                            confidence=float(score),
                            box=BoundingBox(
                                x_min=max(0.0, min(1.0, x_min / target_width)),
                                y_min=max(0.0, min(1.0, y_min / target_height)),
                                x_max=max(0.0, min(1.0, x_max / target_width)),
                                y_max=max(0.0, min(1.0, y_max / target_height)),
                            ),
                        )
                        all_detections.append(detection)

            # Clean up memory after each weed type
            del inputs, outputs, results
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        # Sort by confidence descending
        all_detections.sort(key=lambda d: d.confidence, reverse=True)

        inference_time = (time.perf_counter() - start_time) * 1000

        result = DetectionResult(
            detections=all_detections,
            image_width=target_width,
            image_height=target_height,
            inference_time_ms=inference_time,
        )

        # Filter out tiny boxes (noise) - min 0.2% of image area
        result = result.filter_by_size(min_size=0.002, max_size=1.0)

        # Aggressively deduplicate nearby boxes
        # min_distance of 0.2 = boxes within 20% of image size are merged
        result = result.deduplicate(min_distance=0.2)

        return result

    def detect_by_text(
        self,
        target_image: Union[str, Path, Image.Image],
        text_queries: list[str],
        confidence_threshold: float = settings.default_confidence_threshold,
    ) -> DetectionResult:
        """
        Detect objects using text descriptions instead of reference images.

        This is more precise than image-guided detection because it won't
        match background patterns - only actual objects matching the text.

        Args:
            target_image: The image to search in
            text_queries: List of text descriptions like ["dandelion flower", "yellow dandelion"]
            confidence_threshold: Minimum confidence for detections

        Returns:
            DetectionResult with detected objects
        """
        start_time = time.perf_counter()

        # Load target image
        target = self.load_image(target_image)
        target_width, target_height = target.size

        # Process inputs for text-guided detection
        inputs = self.processor(
            text=text_queries,
            images=target,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        target_sizes = torch.tensor([[target_height, target_width]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=confidence_threshold,
            target_sizes=target_sizes,
        )[0]  # Single image, get first result

        all_detections: list[Detection] = []
        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]

        for box, score, label_idx in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box.tolist()
            # Map label index back to text query
            label = text_queries[label_idx] if label_idx < len(text_queries) else "unknown"

            detection = Detection(
                label=label,
                confidence=float(score),
                box=BoundingBox(
                    x_min=max(0.0, min(1.0, x_min / target_width)),
                    y_min=max(0.0, min(1.0, y_min / target_height)),
                    x_max=max(0.0, min(1.0, x_max / target_width)),
                    y_max=max(0.0, min(1.0, y_max / target_height)),
                ),
            )
            all_detections.append(detection)

        # Clean up
        del inputs, outputs, results
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # Sort by confidence
        all_detections.sort(key=lambda d: d.confidence, reverse=True)

        inference_time = (time.perf_counter() - start_time) * 1000

        result = DetectionResult(
            detections=all_detections,
            image_width=target_width,
            image_height=target_height,
            inference_time_ms=inference_time,
        )

        # Deduplicate nearby boxes
        result = result.deduplicate(min_distance=0.1)

        return result

    def warmup(self) -> None:
        """
        Warm up the model with a dummy inference.

        Call this at startup to avoid cold-start latency on first request.
        """
        # Create a small dummy image
        dummy = Image.new("RGB", (224, 224), color="green")
        dummy_ref = Image.new("RGB", (224, 224), color="yellow")

        # Run inference to load model weights into memory
        self.detect(
            target_image=dummy,
            reference_images={"warmup": [dummy_ref]},
            confidence_threshold=0.99,  # High threshold = no results
        )
