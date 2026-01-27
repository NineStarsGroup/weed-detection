"""Detection API routes."""

import base64
import logging
import uuid
from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

# Register HEIC/HEIF format support with PIL
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    logger_temp = logging.getLogger(__name__)
    logger_temp.info("HEIC/HEIF image support enabled")
except ImportError:
    pass  # pillow-heif not installed, HEIC won't be supported

from src.api.dependencies import get_detector, get_reference_manager
from src.api.schemas import DetectResponse, DetectWithImageResponse
from src.config import DetectionMode, DETECTION_MODES
from src.detection import WeedDetector
from src.references import ReferenceImageManager
from src.visualization import annotate_image, create_comparison_image, image_to_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/detect", tags=["detection"])

# In-memory store for recent detection images (simple cache)
_image_cache: dict[str, dict] = {}

# Lazy-loaded detectors (separate from OWLv2)
_grounding_dino_detector = None
_grounding_dino_local_swint_detector = None
_grounding_dino_local_swinb_detector = None
_grounding_dino_1_5_edge_detector = None
_grounding_dino_1_5_pro_detector = None
_dynamic_dino_detector = None
_sam_detector = None
_rf_detr_detector = None


def _get_grounding_dino_detector(use_tensorrt: bool = False):
    """Get or create the Grounding DINO (Original) detector (lazy loading)."""
    global _grounding_dino_detector
    if _grounding_dino_detector is None:
        from src.detection.grounding_dino import GroundingDINODetector
        _grounding_dino_detector = GroundingDINODetector()
    return _grounding_dino_detector


def _get_grounding_dino_local_swint_detector():
    """Get or create the Grounding DINO local Swin-T detector (lazy loading)."""
    global _grounding_dino_local_swint_detector
    if _grounding_dino_local_swint_detector is None:
        from src.detection.grounding_dino_local import GroundingDINOLocalDetector, SWINT_WEIGHTS, DEFAULT_WEIGHTS_DIR
        weights_path = DEFAULT_WEIGHTS_DIR / SWINT_WEIGHTS
        _grounding_dino_local_swint_detector = GroundingDINOLocalDetector(weights_path=weights_path)
    return _grounding_dino_local_swint_detector


def _get_grounding_dino_local_swinb_detector():
    """Get or create the Grounding DINO local Swin-B detector (lazy loading)."""
    global _grounding_dino_local_swinb_detector
    if _grounding_dino_local_swinb_detector is None:
        from src.detection.grounding_dino_local import GroundingDINOLocalDetector, SWINB_WEIGHTS, DEFAULT_WEIGHTS_DIR
        weights_path = DEFAULT_WEIGHTS_DIR / SWINB_WEIGHTS
        _grounding_dino_local_swinb_detector = GroundingDINOLocalDetector(weights_path=weights_path)
    return _grounding_dino_local_swinb_detector


def _get_grounding_dino_1_5_edge_detector(use_tensorrt: bool = False):
    """Get or create the DINO 1.5 Edge detector (lazy loading)."""
    global _grounding_dino_1_5_edge_detector
    if _grounding_dino_1_5_edge_detector is None:
        from src.detection.grounding_dino_1_5_edge import GroundingDINO15EdgeDetector
        _grounding_dino_1_5_edge_detector = GroundingDINO15EdgeDetector(
            use_tensorrt=use_tensorrt
        )
    elif use_tensorrt and not _grounding_dino_1_5_edge_detector.use_tensorrt:
        # Re-enable TensorRT if requested
        _grounding_dino_1_5_edge_detector.use_tensorrt = True
    return _grounding_dino_1_5_edge_detector


def _get_grounding_dino_1_5_pro_detector(use_tensorrt: bool = False):
    """Get or create the DINO 1.5 Pro detector (lazy loading)."""
    global _grounding_dino_1_5_pro_detector
    if _grounding_dino_1_5_pro_detector is None:
        from src.detection.grounding_dino_1_5_pro import GroundingDINO15ProDetector
        _grounding_dino_1_5_pro_detector = GroundingDINO15ProDetector(
            use_tensorrt=use_tensorrt
        )
    elif use_tensorrt and not _grounding_dino_1_5_pro_detector.use_tensorrt:
        _grounding_dino_1_5_pro_detector.use_tensorrt = True
    return _grounding_dino_1_5_pro_detector


def _get_dynamic_dino_detector(use_tensorrt: bool = False):
    """Get or create the Dynamic-DINO detector (lazy loading)."""
    global _dynamic_dino_detector
    if _dynamic_dino_detector is None:
        from src.detection.dynamic_dino import DynamicDINODetector
        _dynamic_dino_detector = DynamicDINODetector(use_tensorrt=use_tensorrt)
    elif use_tensorrt and not _dynamic_dino_detector.use_tensorrt:
        _dynamic_dino_detector.use_tensorrt = True
    return _dynamic_dino_detector


def _get_sam_detector():
    """Get or create the SAM detector (lazy loading)."""
    global _sam_detector
    if _sam_detector is None:
        from src.detection.sam_detector import SAMDetector
        _sam_detector = SAMDetector()
    return _sam_detector


def _get_rf_detr_detector():
    """Get or create the RF-DETR detector (lazy loading)."""
    global _rf_detr_detector
    if _rf_detr_detector is None:
        from pathlib import Path
        from src.detection.rf_detr import RFDETRDetector
        # Look for fine-tuned weights in the weights directory
        weights_path = Path("weights/rf_detr_weed_weights.pt")
        _rf_detr_detector = RFDETRDetector(
            weights_path=weights_path if weights_path.exists() else None,
            model_size="medium",
        )
    return _rf_detr_detector


def _build_text_queries(weed_types: list[str], for_dino: bool = False) -> list[str]:
    """
    Build text queries for text-guided detection models.

    Args:
        weed_types: List of weed types to detect
        for_dino: If True, use fewer but more targeted prompts for DINO
                  to reduce overlapping detections
    """
    text_queries = []
    for weed_type in weed_types:
        if weed_type.lower() == "dandelion":
            if for_dino:
                # DINO: Focus on leaves only - detect plants even without flowers
                text_queries.extend([
                    "dandelion plant with jagged leaves",    # Leaf rosettes (primary)
                    "rosette of serrated green leaves",      # Leaf pattern description
                    "dandelion leaves growing from center",  # Radial leaf pattern
                ])
            else:
                # OWLv2: simpler prompts work better
                text_queries.extend([
                    "dandelion",
                    "yellow dandelion flower",
                    "dandelion puffball",
                    "dandelion leaves",
                ])

        elif weed_type.lower() == "clover":
            if for_dino:
                # Focus on the distinctive three-leaf pattern
                text_queries.extend([
                    "clover with three round leaves",        # Trifoliate pattern (primary)
                    "three leaflets in clover pattern",      # Leaf structure
                    "clover leaf with three rounded lobes",  # Individual leaf detail
                ])
            else:
                text_queries.extend([
                    "clover",
                    "three leaf clover",
                    "clover leaves",
                    "trifoliate clover",
                ])

        elif weed_type.lower() == "crabgrass":
            if for_dino:
                # Focus on the distinctive spreading grass pattern
                text_queries.extend([
                    "crabgrass with spreading stems",    # Leaf/stem pattern
                    "low growing grass weed",            # Growth habit
                ])
            else:
                text_queries.extend([
                    "crabgrass",
                    "crabgrass weed",
                ])

        elif weed_type.lower() == "poa_annua":
            if for_dino:
                # Poa annua (annual bluegrass) - taller clumpy grass that stands above turf
                text_queries.extend([
                    "tall grass clump standing above lawn",     # Height difference (primary)
                    "raised grass tuft in mowed lawn",          # Clumpy growth habit
                    "grass clump taller than surrounding turf", # Relative height
                ])
            else:
                text_queries.extend([
                    "poa annua",
                    "annual bluegrass",
                    "tall grass clump",
                ])

        elif weed_type.lower() == "silverleaf_nightshade":
            if for_dino:
                # Silverleaf nightshade - focus on distinctive silvery/fuzzy leaf texture
                text_queries.extend([
                    "plant with silver-gray fuzzy leaves",           # Leaf texture (primary)
                    "nightshade with wavy-edged silvery leaves",     # Leaf shape + color
                    "broadleaf weed with dusty silver foliage",      # Overall leaf appearance
                ])
            else:
                text_queries.extend([
                    "silverleaf nightshade",
                    "nightshade plant",
                    "silvery weed leaves",
                ])

        elif weed_type.lower() == "field_bindweed":
            if for_dino:
                # Field bindweed - focus on distinctive arrow-shaped leaves and vine habit
                text_queries.extend([
                    "vine with arrow-shaped leaves",                 # Leaf shape (primary)
                    "creeping vine with arrowhead leaves",           # Growth habit + leaf shape
                    "bindweed leaves shaped like arrowheads",        # Specific leaf morphology
                ])
            else:
                text_queries.extend([
                    "field bindweed",
                    "bindweed vine",
                    "morning glory weed",
                ])

        elif weed_type.lower() == "broom_snakeweed":
            if for_dino:
                # Broom snakeweed - focus on distinctive thin stems and woody base
                text_queries.extend([
                    "woody shrub with thin needle-like leaves",      # Leaf/stem structure (primary)
                    "snakeweed with fine thread-like green stems",   # Stem appearance
                    "broom-like plant with many thin branches",      # Growth habit
                ])
            else:
                text_queries.extend([
                    "broom snakeweed",
                    "snakeweed shrub",
                    "yellow flowering weed",
                ])

        elif weed_type.lower() == "palmers_amaranth":
            if for_dino:
                # Palmer's amaranth - focus on distinctive large oval leaves and tall growth
                text_queries.extend([
                    "tall weed with large oval leaves",              # Leaf shape + height (primary)
                    "pigweed with smooth diamond-shaped leaves",     # Leaf morphology
                    "amaranth plant with broad green leaves on long stems",  # Leaf arrangement
                ])
            else:
                text_queries.extend([
                    "palmer amaranth",
                    "pigweed",
                    "amaranth weed",
                ])

        elif weed_type.lower() == "russian_thistle":
            if for_dino:
                # Russian thistle (tumbleweed) - spiny round plant
                text_queries.extend([
                    "russian thistle with spiny branches",
                    "tumbleweed plant with thin needle-like leaves",
                    "round bushy weed with spine-tipped stems",
                ])
            else:
                text_queries.extend([
                    "russian thistle",
                    "tumbleweed",
                    "prickly weed",
                ])

        else:
            text_queries.append(weed_type)
            if not for_dino:
                text_queries.append(f"{weed_type} plant")
    return text_queries


@router.post("", response_model=DetectResponse)
async def detect_weeds(
    image: Annotated[UploadFile, File(description="Image to analyze for weeds")],
    confidence_threshold: Annotated[
        float, Form(description="Minimum confidence (0-1)")
    ] = 0.25,
    adaptive_threshold: Annotated[
        bool, Form(description="Use adaptive thresholding based on detection density")
    ] = True,
    weed_types: Annotated[
        str | None, Form(description="Comma-separated weed types to detect")
    ] = None,
    detector: WeedDetector = Depends(get_detector),
    ref_manager: ReferenceImageManager = Depends(get_reference_manager),
) -> DetectResponse:
    """
    Detect weeds in an uploaded image (JSON response only).

    Upload an image and get back bounding boxes for detected weeds.
    Uses few-shot learning with stored reference images.
    """
    # Validate image file
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Load the uploaded image
        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        try:
            pil_image = Image.open(BytesIO(contents)).convert("RGB")
        except Exception as img_err:
            logger.error(f"Failed to open image: {img_err}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {img_err}")

        # Get reference images for requested weed types
        if weed_types:
            requested_types = [t.strip() for t in weed_types.split(",")]
        else:
            requested_types = ref_manager.list_weed_types()

        if not requested_types:
            return DetectResponse(
                success=False,
                error="No reference images available. Upload reference images first.",
            )

        # Build reference images dict
        reference_images = {}
        for weed_type in requested_types:
            refs = ref_manager.get_reference_images(weed_type)
            if refs:
                reference_images[weed_type] = refs

        if not reference_images:
            return DetectResponse(
                success=False,
                error=f"No reference images found for types: {requested_types}",
            )

        # Run detection with low initial threshold (filtering happens after)
        initial_threshold = 0.01 if adaptive_threshold else confidence_threshold
        result = detector.detect(
            target_image=pil_image,
            reference_images=reference_images,
            confidence_threshold=initial_threshold,
        )

        # Apply filtering
        if adaptive_threshold:
            result = result.filter_adaptive(base_threshold=confidence_threshold)
        else:
            result = result.filter_by_confidence(confidence_threshold)

        logger.info(
            f"Detection complete: {result.count} weeds found in {result.inference_time_ms:.1f}ms"
        )

        return DetectResponse(success=True, result=result)

    except Exception as e:
        logger.exception("Detection failed")
        return DetectResponse(success=False, error=str(e))


@router.post("/visualize", response_model=DetectWithImageResponse)
async def detect_weeds_with_visualization(
    image: Annotated[UploadFile, File(description="Image to analyze for weeds")],
    confidence_threshold: Annotated[
        float, Form(description="Minimum confidence (0-1)")
    ] = 0.25,
    detection_mode: Annotated[
        str, Form(description="Detection mode: text_owlv2, grounding_dino, grounding_dino_1_5_pro, grounding_dino_1_5_edge, dynamic_dino, sam_auto, image_owlv2")
    ] = "text_owlv2",
    group_overlapping: Annotated[
        str, Form(description="Merge overlapping detections into unified regions")
    ] = "false",
    weed_types: Annotated[
        str | None, Form(description="Comma-separated weed types to detect")
    ] = None,
    use_tensorrt: Annotated[
        str, Form(description="Enable TensorRT acceleration (requires NVIDIA GPU)")
    ] = "false",
    detector: WeedDetector = Depends(get_detector),
    ref_manager: ReferenceImageManager = Depends(get_reference_manager),
) -> DetectWithImageResponse:
    """
    Detect weeds and return annotated image with bounding boxes.

    Returns JSON with detection results plus base64-encoded images:
    - original_image: The uploaded image
    - annotated_image: Image with bounding boxes, labels, and confidence scores
    - comparison_image: Side-by-side before/after view
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        logger.info(f"Received image: {image.filename}, size={len(contents)} bytes, content_type={image.content_type}")

        try:
            pil_image = Image.open(BytesIO(contents)).convert("RGB")
        except Exception as img_err:
            logger.error(f"Failed to open image: {img_err}, first 20 bytes: {contents[:20]}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {img_err}")

        # Get weed types - use provided list, reference manager, or defaults for text modes
        mode = DetectionMode(detection_mode) if detection_mode in [m.value for m in DetectionMode] else DetectionMode.TEXT_OWLV2

        if weed_types:
            requested_types = [t.strip() for t in weed_types.split(",")]
        elif mode in (
            DetectionMode.GROUNDING_DINO,
            DetectionMode.GROUNDING_DINO_LOCAL_SWINT,
            DetectionMode.GROUNDING_DINO_LOCAL_SWINB,
            DetectionMode.GROUNDING_DINO_1_5_EDGE,
            DetectionMode.GROUNDING_DINO_1_5_PRO,
            DetectionMode.DYNAMIC_DINO,
            DetectionMode.TEXT_OWLV2,
        ):
            # Text-based modes can use default weed types without reference images
            requested_types = ["dandelion", "clover", "crabgrass"]
        elif mode == DetectionMode.RF_DETR:
            # RF-DETR uses its trained classes - default to main weeds
            requested_types = ["dandelion", "clover", "crabgrass", "dock", "milk_thistle"]
        else:
            requested_types = ref_manager.list_weed_types()

        if not requested_types:
            return DetectWithImageResponse(
                success=False,
                error="No weed types configured. Upload reference images first.",
            )

        # Parse TensorRT flag
        enable_tensorrt = use_tensorrt.lower() == "true"

        # Run detection based on selected mode
        if mode == DetectionMode.TEXT_OWLV2:
            # OWLv2 TEXT-BASED detection
            text_queries = _build_text_queries(requested_types, for_dino=False)
            logger.info(f"Running OWLv2 TEXT detection with threshold={confidence_threshold}")
            result = detector.detect_by_text(
                target_image=pil_image,
                text_queries=text_queries,
                confidence_threshold=confidence_threshold,
            )
        elif mode == DetectionMode.GROUNDING_DINO:
            # Grounding DINO Original detection - uses detailed visual prompts
            text_queries = _build_text_queries(requested_types, for_dino=True)
            dino_detector = _get_grounding_dino_detector()
            logger.info(f"Running Grounding DINO (Original) detection with threshold={confidence_threshold}, queries={text_queries}")
            result = dino_detector.detect(
                target_image=pil_image,
                text_queries=text_queries,
                confidence_threshold=confidence_threshold,
            )
        elif mode == DetectionMode.GROUNDING_DINO_LOCAL_SWINT:
            # Grounding DINO with local Swin-T weights (faster)
            text_queries = _build_text_queries(requested_types, for_dino=True)
            dino_detector = _get_grounding_dino_local_swint_detector()
            logger.info(f"Running Grounding DINO Swin-T (Local) detection with threshold={confidence_threshold}")
            result = dino_detector.detect(
                target_image=pil_image,
                text_queries=text_queries,
                confidence_threshold=confidence_threshold,
            )
        elif mode == DetectionMode.GROUNDING_DINO_LOCAL_SWINB:
            # Grounding DINO with local Swin-B weights (more accurate)
            text_queries = _build_text_queries(requested_types, for_dino=True)
            dino_detector = _get_grounding_dino_local_swinb_detector()
            logger.info(f"Running Grounding DINO Swin-B (Local) detection with threshold={confidence_threshold}")
            result = dino_detector.detect(
                target_image=pil_image,
                text_queries=text_queries,
                confidence_threshold=confidence_threshold,
            )
        elif mode == DetectionMode.GROUNDING_DINO_1_5_EDGE:
            # DINO 1.5 Edge - optimized for speed (~75 FPS with TensorRT)
            text_queries = _build_text_queries(requested_types, for_dino=True)
            dino_detector = _get_grounding_dino_1_5_edge_detector(use_tensorrt=enable_tensorrt)
            trt_status = " (TensorRT)" if enable_tensorrt else ""
            logger.info(f"Running DINO 1.5 Edge{trt_status} detection with threshold={confidence_threshold}")
            result = dino_detector.detect(
                target_image=pil_image,
                text_queries=text_queries,
                confidence_threshold=confidence_threshold,
            )
        elif mode == DetectionMode.GROUNDING_DINO_1_5_PRO:
            # DINO 1.5 Pro - highest accuracy (54.3 AP)
            text_queries = _build_text_queries(requested_types, for_dino=True)
            dino_detector = _get_grounding_dino_1_5_pro_detector(use_tensorrt=enable_tensorrt)
            trt_status = " (TensorRT)" if enable_tensorrt else ""
            logger.info(f"Running DINO 1.5 Pro{trt_status} detection with threshold={confidence_threshold}")
            result = dino_detector.detect(
                target_image=pil_image,
                text_queries=text_queries,
                confidence_threshold=confidence_threshold,
            )
        elif mode == DetectionMode.DYNAMIC_DINO:
            # Dynamic-DINO - MoE architecture, balanced speed/accuracy
            text_queries = _build_text_queries(requested_types, for_dino=True)
            dino_detector = _get_dynamic_dino_detector(use_tensorrt=enable_tensorrt)
            trt_status = " (TensorRT)" if enable_tensorrt else ""
            logger.info(f"Running Dynamic-DINO{trt_status} detection with threshold={confidence_threshold}")
            result = dino_detector.detect(
                target_image=pil_image,
                text_queries=text_queries,
                confidence_threshold=confidence_threshold,
            )
        elif mode == DetectionMode.SAM_AUTO:
            # SAM automatic segmentation - finds all objects
            sam_detector = _get_sam_detector()
            logger.info(f"Running SAM auto-segmentation")
            result = sam_detector.detect(
                target_image=pil_image,
                min_mask_area=0.005,
                max_mask_area=0.25,
                points_per_side=8,  # Reduced from 16 for faster inference (~4x speedup)
            )
        elif mode == DetectionMode.RF_DETR:
            # RF-DETR fine-tuned detection - closed-vocabulary, trained on weed classes
            rf_detr_detector = _get_rf_detr_detector()
            logger.info(f"Running RF-DETR detection with threshold={confidence_threshold}")
            result = rf_detr_detector.detect(
                target_image=pil_image,
                confidence_threshold=confidence_threshold,
                weed_types=requested_types,  # Filter to requested types (post-inference)
            )
        else:
            # IMAGE-BASED detection - uses reference images
            reference_images = {}
            for weed_type in requested_types:
                refs = ref_manager.get_reference_images(weed_type)
                if refs:
                    reference_images[weed_type] = refs

            if not reference_images:
                return DetectWithImageResponse(
                    success=False,
                    error=f"No reference images found for types: {requested_types}",
                )

            logger.info(f"Running OWLv2 IMAGE detection with threshold={confidence_threshold}")
            result = detector.detect(
                target_image=pil_image,
                reference_images=reference_images,
                confidence_threshold=confidence_threshold,
            )

        # Filter only extreme outliers (>50% of image is almost certainly wrong)
        # Using a high threshold to avoid filtering legitimate large weed patches
        result = result.filter_by_size(min_size=0.0005, max_size=0.5)

        # Apply grouping if requested (merges overlapping flower + leaf detections)
        should_group = group_overlapping.lower() == "true"
        if should_group:
            logger.info(f"Grouping overlapping detections (before: {result.count})")
            result = result.cluster_overlapping()
            logger.info(f"After grouping: {result.count} detections")

        # No cap on detections - show all found weeds

        # Create annotated image
        annotated = annotate_image(pil_image, result)

        # Create comparison image
        comparison = create_comparison_image(pil_image, annotated)

        # Convert images to base64
        original_b64 = base64.b64encode(image_to_bytes(pil_image)).decode("utf-8")
        annotated_b64 = base64.b64encode(image_to_bytes(annotated)).decode("utf-8")
        comparison_b64 = base64.b64encode(image_to_bytes(comparison)).decode("utf-8")

        # Also cache for direct image endpoints
        cache_id = str(uuid.uuid4())[:8]
        _image_cache[cache_id] = {
            "original": pil_image,
            "annotated": annotated,
            "comparison": comparison,
        }
        # Keep cache small (last 10 detections)
        if len(_image_cache) > 10:
            oldest_key = next(iter(_image_cache))
            del _image_cache[oldest_key]

        logger.info(
            f"Detection with visualization: {result.count} weeds found in {result.inference_time_ms:.1f}ms"
        )

        return DetectWithImageResponse(
            success=True,
            result=result,
            original_image=original_b64,
            annotated_image=annotated_b64,
            comparison_image=comparison_b64,
            image_id=cache_id,
        )

    except Exception as e:
        logger.exception("Detection with visualization failed")
        return DetectWithImageResponse(success=False, error=str(e))


@router.get("/image/{image_id}/{image_type}")
async def get_detection_image(
    image_id: str,
    image_type: str,
) -> Response:
    """
    Get a detection image directly (for embedding in HTML).

    Args:
        image_id: ID returned from /detect/visualize
        image_type: One of 'original', 'annotated', or 'comparison'

    Returns:
        JPEG image
    """
    if image_id not in _image_cache:
        raise HTTPException(status_code=404, detail="Image not found (may have expired)")

    if image_type not in ("original", "annotated", "comparison"):
        raise HTTPException(
            status_code=400,
            detail="image_type must be 'original', 'annotated', or 'comparison'",
        )

    image = _image_cache[image_id][image_type]
    image_bytes = image_to_bytes(image)

    return Response(content=image_bytes, media_type="image/jpeg")
