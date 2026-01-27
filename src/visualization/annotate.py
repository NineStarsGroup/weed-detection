"""Image annotation utilities for visualizing detection results."""

from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

from src.detection.models import DetectionResult


# Color palette for different weed types (RGB)
# Keys match shortened labels from grounding_dino._normalize_label()
LABEL_COLORS = {
    "dande": (255, 215, 0),           # Gold - dandelion
    "dandelion": (255, 215, 0),       # Gold (fallback for full name)
    "clover": (50, 205, 50),          # Lime green
    "crab": (255, 99, 71),            # Tomato red - crabgrass
    "crabgrass": (255, 99, 71),       # Tomato red (fallback)
    "thistle": (186, 85, 211),        # Medium orchid
    "plant": (70, 130, 180),          # Steel blue - plantain
    "plantain": (70, 130, 180),       # Steel blue (fallback)
    "chick": (255, 182, 193),         # Light pink - chickweed
    "chickweed": (255, 182, 193),     # Light pink (fallback)
    "poa": (135, 206, 235),           # Sky blue - poa annua
    "poa_annua": (135, 206, 235),     # Sky blue (fallback)
    "silver": (147, 112, 219),        # Medium purple - silverleaf nightshade
    "silverleaf": (147, 112, 219),    # Medium purple (fallback)
    "silverleaf_nightshade": (147, 112, 219),  # Medium purple (fallback)
    "bindwd": (255, 105, 180),        # Hot pink - field bindweed
    "bindweed": (255, 105, 180),      # Hot pink (fallback)
    "field_bindweed": (255, 105, 180),  # Hot pink (fallback)
    "snake": (218, 165, 32),          # Goldenrod - broom snakeweed
    "snakeweed": (218, 165, 32),      # Goldenrod (fallback)
    "broom_snakeweed": (218, 165, 32),  # Goldenrod (fallback)
    "amarth": (220, 20, 60),          # Crimson - palmer's amaranth
    "amaranth": (220, 20, 60),        # Crimson (fallback)
    "palmers_amaranth": (220, 20, 60),  # Crimson (fallback)
    "rthistle": (139, 69, 19),        # Saddle brown - russian thistle
    "russian_thistle": (139, 69, 19), # Saddle brown (fallback)
    "tumbleweed": (139, 69, 19),      # Saddle brown (fallback)
    "weed": (255, 165, 0),            # Orange - generic weed
    "default": (255, 0, 0),           # Red fallback
}


def get_color_for_label(label: str) -> tuple[int, int, int]:
    """Get a consistent color for a weed type label."""
    return LABEL_COLORS.get(label.lower(), LABEL_COLORS["default"])


def annotate_image(
    image: Image.Image,
    result: DetectionResult,
    box_width: int = 3,
    font_size: int = 16,
    show_confidence: bool = True,
) -> Image.Image:
    """
    Draw detection bounding boxes and labels on an image.

    Args:
        image: PIL Image to annotate
        result: DetectionResult containing detections to draw
        box_width: Width of bounding box lines
        font_size: Size of label text
        show_confidence: Whether to show confidence percentage

    Returns:
        New PIL Image with annotations drawn
    """
    # Work on a copy to preserve original
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    # Try to load a nice font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    img_width, img_height = image.size

    for detection in result.detections:
        # Convert normalized coordinates to pixels
        box = detection.box
        x1 = int(box.x_min * img_width)
        y1 = int(box.y_min * img_height)
        x2 = int(box.x_max * img_width)
        y2 = int(box.y_max * img_height)

        # Get color for this weed type
        color = get_color_for_label(detection.label)

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

        # Build label text
        if show_confidence:
            label_text = f"{detection.label} {detection.confidence:.0%}"
        else:
            label_text = detection.label

        # Calculate label background size
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Position label above box, or inside if at top edge
        label_x = x1
        label_y = y1 - text_height - 4
        if label_y < 0:
            label_y = y1 + 4

        # Draw label background
        padding = 2
        draw.rectangle(
            [
                label_x - padding,
                label_y - padding,
                label_x + text_width + padding,
                label_y + text_height + padding,
            ],
            fill=color,
        )

        # Draw label text (black for contrast)
        draw.text((label_x, label_y), label_text, fill=(0, 0, 0), font=font)

    return annotated


def create_comparison_image(
    original: Image.Image,
    annotated: Image.Image,
    gap: int = 20,
) -> Image.Image:
    """
    Create a side-by-side comparison of original and annotated images.

    Args:
        original: Original image (before detection)
        annotated: Annotated image (with bounding boxes)
        gap: Pixels between the two images

    Returns:
        New PIL Image with both images side by side
    """
    # Ensure same size
    width, height = original.size

    # Create new image to hold both
    comparison = Image.new("RGB", (width * 2 + gap, height), color=(40, 40, 40))

    # Paste original on left
    comparison.paste(original, (0, 0))

    # Paste annotated on right
    comparison.paste(annotated, (width + gap, 0))

    # Add labels
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()

    draw.text((10, 10), "Original", fill=(255, 255, 255), font=font)
    draw.text((width + gap + 10, 10), "Detected", fill=(255, 255, 255), font=font)

    return comparison


def image_to_bytes(image: Image.Image, format: str = "JPEG", quality: int = 90) -> bytes:
    """Convert PIL Image to bytes for HTTP response."""
    buffer = BytesIO()
    image.save(buffer, format=format, quality=quality)
    buffer.seek(0)
    return buffer.getvalue()
