#!/usr/bin/env python3
"""
Quick test script to verify the detection pipeline works.

This creates synthetic test images to verify the model loads and runs.
For real testing, use actual dandelion/grass images.

Usage:
    python scripts/test_detection.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw

from src.detection import WeedDetector


def create_test_images():
    """Create simple synthetic test images."""
    # Create a "grass" background with a yellow circle (fake dandelion)
    target = Image.new("RGB", (640, 480), color=(34, 139, 34))  # Forest green
    draw = ImageDraw.Draw(target)
    # Draw a yellow circle in the center
    draw.ellipse([250, 180, 390, 320], fill=(255, 215, 0))  # Gold/yellow

    # Create a reference image - just a yellow circle on green
    reference = Image.new("RGB", (224, 224), color=(34, 139, 34))
    ref_draw = ImageDraw.Draw(reference)
    ref_draw.ellipse([50, 50, 174, 174], fill=(255, 215, 0))

    return target, reference


def main():
    print("=" * 60)
    print("Weed Detection Test")
    print("=" * 60)

    # Create detector
    print("\n1. Initializing detector...")
    detector = WeedDetector()
    print(f"   Device: {detector.device}")

    # Create test images
    print("\n2. Creating test images...")
    target, reference = create_test_images()
    print(f"   Target: {target.size}")
    print(f"   Reference: {reference.size}")

    # Run detection
    print("\n3. Running detection (this will download the model on first run)...")
    print("   This may take 1-2 minutes on first run...")

    result = detector.detect(
        target_image=target,
        reference_images={"test_weed": [reference]},
        confidence_threshold=0.05,  # Low threshold for synthetic images
    )

    # Show results
    print("\n4. Results:")
    print(f"   Inference time: {result.inference_time_ms:.1f}ms")
    print(f"   Detections found: {result.count}")

    for i, det in enumerate(result.detections[:5]):  # Show up to 5
        print(f"\n   Detection {i + 1}:")
        print(f"     Label: {det.label}")
        print(f"     Confidence: {det.confidence:.3f}")
        print(f"     Box: ({det.box.x_min:.2f}, {det.box.y_min:.2f}) -> ({det.box.x_max:.2f}, {det.box.y_max:.2f})")

    print("\n" + "=" * 60)
    if result.count > 0:
        print("SUCCESS: Detection pipeline is working!")
    else:
        print("NOTE: No detections found (normal for synthetic images)")
        print("      Try with real dandelion photos for actual detection.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
