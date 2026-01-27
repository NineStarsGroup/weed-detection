# Fine-Tuning Guide - Complete Weed Detection

## Version 1.0 (January 21, 2026)

---

## The Problem

Current zero-shot models (DINO, OWLv2) detect **leaves** but not **stems**. For practical weed removal, we need to detect the complete weed including:
- Leaf rosette
- Stem
- Root attachment point (where weed meets ground)

---

## Why Fine-Tuning is Needed

Zero-shot models are trained on internet images where "dandelion" typically means the distinctive parts (flowers, leaves). Stems are rarely annotated because they're:
- Thin and hard to see
- Green-on-green (low contrast against grass)
- Not what humans typically label

**Solution:** Fine-tune the model on a custom dataset with bounding boxes around **complete weeds**.

---

## Fine-Tuning Options

### Option A: Multi-Class Dataset (All Weeds at Once) ✅ Recommended

Train **one model** to detect all weed types simultaneously:

```
image_001.jpg:
  - bbox: [x1, y1, x2, y2], label: "dandelion_complete"
  - bbox: [x3, y3, x4, y4], label: "clover_complete"

image_002.jpg:
  - bbox: [x1, y1, x2, y2], label: "crabgrass_complete"
```

**Pros:**
- Single model handles everything
- Model learns shared features (green stems, ground attachment patterns)
- More efficient inference

**Cons:**
- Need diverse images with multiple weed types
- Requires more total annotations

---

### Option B: Single-Class "Complete Weed" Model

Train to detect just **"weed"** as one class (regardless of type):

```
image_001.jpg:
  - bbox: [x1, y1, x2, y2], label: "weed"
  - bbox: [x3, y3, x4, y4], label: "weed"
```

Then use existing DINO prompts to **classify** what type it is.

**Pros:**
- Simpler labeling (just draw boxes, don't worry about species)
- Faster to create dataset
- Can combine with existing detection for classification

**Cons:**
- Loses species-specific detection in one pass

---

## Dataset Requirements

| Approach | Images Needed | Labels per Image | Training Time |
|----------|---------------|------------------|---------------|
| Multi-class (all weeds) | 200-500 | 1-5 boxes with species | ~2-4 hours (GPU) |
| Single-class (just "weed") | 100-200 | 1-5 boxes, all same label | ~1-2 hours (GPU) |

---

## Labeling Strategy: Mixed vs Single Species

A common question: Do I need 100-500 images of the **same weed species**, or can I mix different species?

### Option 1: Mixed Species, Single Class ✅ Recommended to Start

Label 100-500 images containing **any weed species**, all with the same label: `complete_weed`

```
image_001.jpg: dandelion with stem → "complete_weed"
image_002.jpg: clover with stem → "complete_weed"
image_003.jpg: crabgrass with stem → "complete_weed"
```

**Pros:**
- Fastest to label (don't need to identify species)
- Model learns the general concept of "complete plant with stem/root point"
- Works across all weed types
- Can use existing DINO prompts for species classification after detection

**Cons:**
- Single output class (just "weed", not species-specific)

**Workflow with Single Class:**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Fine-tuned DINO │────▶│ Detects complete│────▶│ Existing DINO   │
│ (mixed species) │     │ weed boundaries │     │ classifies type │
└─────────────────┘     └─────────────────┘     └─────────────────┘
   Trained on                Returns box           "dandelion" vs
   "complete_weed"           with stem             "clover" etc.
```

---

### Option 2: Mixed Species, Multi-Class Labels

Label 100-500 images with **species-specific labels**:

```
image_001.jpg: dandelion with stem → "dandelion_complete"
image_002.jpg: clover with stem → "clover_complete"
image_003.jpg: crabgrass with stem → "crabgrass_complete"
```

**Pros:**
- Single model detects AND classifies in one pass
- More information per inference
- No need for secondary classification step

**Cons:**
- Need ~50-100 images **per species** (more total labeling)
- Must correctly identify each weed species while labeling
- Requires botanical knowledge or reference guide

---

### Option 3: Single Species Focus

Label 100-500 images of **only one weed type** (e.g., just dandelions):

```
image_001.jpg: dandelion with stem → "dandelion_complete"
image_002.jpg: dandelion with stem → "dandelion_complete"
...all dandelions...
```

**Pros:**
- Easiest if you have lots of one species
- Model becomes expert at that one weed
- Good for high-value target weeds

**Cons:**
- Need to repeat training for each weed type
- Multiple models or training runs
- Won't generalize to other species

---

### Minimum Images Per Species (If Multi-Class)

If you want species-specific detection in a single model:

| Species | Minimum Images | Recommended |
|---------|----------------|-------------|
| Dandelion | 50 | 100+ |
| Clover | 50 | 100+ |
| Crabgrass | 50 | 100+ |
| Poa Annua | 50 | 100+ |
| **Total** | **200** | **400+** |

---

### Recommendation

**Start with Option 1 (Mixed Species, Single Class)** because:

| Reason | Benefit |
|--------|---------|
| Faster labeling | Just draw boxes, don't identify species |
| Model learns "stem/root" concept | Transfers across all species |
| Works with existing setup | DINO prompts still classify species |
| Can upgrade later | Add species labels to same images if needed |

Once the single-class model works well, you can optionally:
1. Go back and add species labels to your existing annotations
2. Re-train as multi-class
3. Or keep single-class and use two-stage detection (detect → classify)

---

## Model Options

| Model | Accuracy | Speed | Text Grounding | Training Complexity |
|-------|----------|-------|----------------|---------------------|
| **Grounding DINO (fine-tuned)** | High | ~8 FPS | ✅ Yes | Medium |
| **DINO 1.5 Pro (fine-tuned)** | Highest | ~5 FPS | ✅ Yes | Medium |
| **YOLOv8-L/X** | High | ~30-60 FPS | ❌ No | Easy |
| **RT-DETR** | High | ~30 FPS | ❌ No | Easy |

### Recommendation: Fine-Tune Grounding DINO

Since the app already has DINO infrastructure, fine-tuning it makes the most sense:

**Why DINO over YOLO:**
- Keeps text grounding - can still use prompts like "dandelion with complete stem"
- Integrates directly into existing app
- Can combine fine-tuned knowledge with zero-shot capabilities

---

## Full Training Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        FINE-TUNING PIPELINE                               │
└──────────────────────────────────────────────────────────────────────────┘

1. COLLECT IMAGES          2. LABEL                 3. PREPARE DATASET
   ───────────────            ─────                    ───────────────
   - Lawn photos              - Label Studio           - COCO format
   - Various angles           - Draw boxes around      - Train/val split
   - Various lighting           complete weeds         - Augmentations
   - 200-500 images           - Include stems/base

4. TRAIN                   5. EVALUATE              6. DEPLOY
   ─────                      ────────                 ──────
   - GPU cluster              - mAP metrics            - Export weights
   - 10-50 epochs             - Visual inspection      - Add to app
   - ~2-8 hours               - Test on new images     - A/B test vs base
```

---

## Step 1: Collect Images

### Image Requirements
- **Quantity:** 200-500 images minimum
- **Resolution:** 1024x1024 or higher recommended
- **Content:** Lawn photos with visible weeds
- **Diversity:**
  - Various angles (top-down, 45°, side)
  - Various lighting (sun, shade, overcast)
  - Various growth stages (seedling, mature, flowering)
  - Various lawn conditions (well-maintained, patchy, dense)

### Image Sources
- Photos from your own lawn
- Public datasets (iNaturalist, Pl@ntNet)
- Synthetic data augmentation

---

## Step 2: Label Images

### Labeling Tools

| Tool | Type | Cost | Ease of Use | Export Formats |
|------|------|------|-------------|----------------|
| **Label Studio** | Self-hosted | Free | Medium | COCO, YOLO, Pascal VOC |
| **Roboflow** | Cloud | Free tier | Easy | COCO, YOLO, TFRecord |
| **CVAT** | Self-hosted | Free | Complex | COCO, YOLO, Pascal VOC |

### Labeling Guidelines

1. **Draw tight bounding boxes** around the complete weed
2. **Include the full plant:**
   - All visible leaves
   - Stem (if visible)
   - Base/root attachment point
3. **Extend box to ground level** even if stem is occluded by grass
4. **Label partially visible weeds** if >50% visible
5. **Skip very small/distant weeds** that are just a few pixels

### Example Annotation

```
Good bounding box:          Bad bounding box:
┌──────────────┐            ┌──────────┐
│   leaves     │            │  leaves  │
│              │            └──────────┘
│   stem       │            (missing stem/base)
│     ↓        │
└──────●───────┘
    root point
```

---

## Step 3: Prepare Dataset

### COCO Format

Standard format for object detection training:

```json
{
  "info": {
    "description": "Weed Detection Dataset",
    "version": "1.0",
    "year": 2026
  },
  "images": [
    {
      "id": 1,
      "file_name": "lawn_001.jpg",
      "width": 1024,
      "height": 768
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 200, 150, 300],
      "area": 45000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "dandelion_complete"},
    {"id": 2, "name": "clover_complete"},
    {"id": 3, "name": "crabgrass_complete"}
  ]
}
```

### Train/Validation Split

- **Training set:** 80% of images
- **Validation set:** 20% of images
- Use stratified split to ensure each weed type is represented in both sets

### Data Augmentation

Apply augmentations to increase effective dataset size:

| Augmentation | Purpose |
|--------------|---------|
| Horizontal flip | More orientations |
| Random rotation (±15°) | Camera angle variance |
| Brightness/contrast | Lighting variance |
| Random crop | Scale variance |
| Blur | Focus variance |

---

## Step 4: Train the Model

### Fine-Tuning Grounding DINO

```python
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

# Load pre-trained DINO
model_name = "IDEA-Research/grounding-dino-tiny"
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# Load your COCO dataset
dataset = load_dataset("json", data_files={
    "train": "data/train/annotations.json",
    "validation": "data/val/annotations.json"
})

# Training arguments
training_args = TrainingArguments(
    output_dir="./weed-dino-finetuned",
    num_train_epochs=20,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,  # Mixed precision for faster training
    dataloader_num_workers=4,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)
trainer.train()

# Save fine-tuned model
model.save_pretrained("./weed-dino-finetuned")
processor.save_pretrained("./weed-dino-finetuned")
```

### Training Parameters

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| Learning rate | 1e-5 to 5e-5 | Lower for fine-tuning |
| Batch size | 4-8 | Depends on GPU memory |
| Epochs | 10-50 | Monitor validation loss |
| Weight decay | 0.01 | Prevents overfitting |
| Warmup steps | 500 | Gradual learning rate increase |

### Hardware Requirements

| GPU | VRAM | Batch Size | Training Time (500 images) |
|-----|------|------------|---------------------------|
| RTX 3080 | 10GB | 4 | ~4 hours |
| RTX 4090 | 24GB | 8 | ~2 hours |
| A100 | 40GB | 16 | ~1 hour |
| H100 | 80GB | 32 | ~30 min |

---

## Step 5: Evaluate the Model

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **mAP@0.5** | Mean Average Precision at IoU 0.5 | >0.7 |
| **mAP@0.5:0.95** | mAP averaged over IoU thresholds | >0.5 |
| **Precision** | True positives / All detections | >0.8 |
| **Recall** | True positives / All ground truth | >0.7 |

### Visual Inspection

Always visually inspect results on held-out test images:

1. Are complete weeds being detected?
2. Is the bounding box extending to the ground/root point?
3. Are stems being included?
4. Any systematic errors (missing small weeds, false positives)?

---

## Step 6: Deploy the Model

### Integration with Existing App

Create a new detector class that loads fine-tuned weights:

```python
# src/detection/grounding_dino_finetuned.py

class GroundingDINOFinetunedDetector:
    def __init__(self, model_path: str = "./weed-dino-finetuned"):
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path)
        self.model.to(device)

    def detect(self, image, text_queries, confidence_threshold=0.3):
        # Same interface as existing detectors
        ...
```

### Add to Detection Modes

Update `src/config.py`:

```python
class DetectionMode(Enum):
    # ... existing modes ...
    GROUNDING_DINO_FINETUNED = "grounding_dino_finetuned"
```

### A/B Testing

Compare fine-tuned model against base model:

| Metric | Base DINO | Fine-tuned DINO |
|--------|-----------|-----------------|
| Leaf detection | ✅ | ✅ |
| Stem detection | ❌ | ✅ |
| Root point | ❌ | ✅ |
| Speed | ~8 FPS | ~8 FPS |

---

## Advanced: DINO → YOLOv8 Knowledge Distillation ✅ Recommended

This approach combines the best of both worlds: DINO's accuracy with YOLO's speed.

### The Concept

Use fine-tuned DINO as a "teacher" to automatically label thousands of images, then train YOLOv8 on that large dataset.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ 1. FINE-TUNE    │───▶│ 2. AUTO-LABEL   │───▶│ 3. TRAIN YOLO   │
│ DINO on 100-500 │    │ 1000s of images │    │ on large dataset│
│ hand-labeled    │    │ using DINO      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
   Manual labels          DINO generates        Fast inference
   (slow, accurate)       pseudo-labels         (30-60 FPS)
```

### Why This Works

| Step | Effort | Output |
|------|--------|--------|
| Hand-label 100-500 images | High (manual) | Small, high-quality dataset |
| Fine-tune DINO | Medium (GPU time) | Accurate "complete weed" detector |
| Run DINO on 5,000+ unlabeled images | Low (automated) | Large pseudo-labeled dataset |
| Train YOLOv8 on pseudo-labels | Medium (GPU time) | **Fast production model** |

**Result:** YOLO's speed (30-60 FPS) with DINO's accuracy, using only 100-500 manually labeled images.

### Comparison: Teacher vs Student

| Aspect | DINO (Teacher) | YOLOv8 (Student) |
|--------|----------------|------------------|
| Speed | ~8 FPS | ~30-60 FPS |
| Text grounding | ✅ Yes | ❌ No |
| Edge deployment | ❌ Heavy (~350MB) | ✅ Lightweight (~25MB) |
| Mobile friendly | ❌ No | ✅ Yes |
| Real-time video | ❌ No | ✅ Yes |
| Training data needed | 100-500 manual | 5,000+ auto-labeled |

### Step-by-Step Pipeline

#### Phase 1: Fine-Tune DINO (Teacher Model)

Follow Steps 1-5 above to create your fine-tuned DINO model that detects complete weeds.

#### Phase 2: Auto-Label with DINO

```python
# scripts/auto_label_with_dino.py

import os
import json
from PIL import Image
from pathlib import Path
from src.detection.grounding_dino_finetuned import GroundingDINOFinetunedDetector

def auto_label_images(
    unlabeled_dir: str,
    output_dir: str,
    confidence_threshold: float = 0.4,
):
    """
    Use fine-tuned DINO to generate pseudo-labels for unlabeled images.

    Args:
        unlabeled_dir: Directory containing unlabeled lawn images
        output_dir: Directory to save YOLO-format labels
        confidence_threshold: Minimum confidence for pseudo-labels
    """
    # Load fine-tuned DINO
    detector = GroundingDINOFinetunedDetector("./models/weed-dino-finetuned")

    # Prepare output directories
    images_out = Path(output_dir) / "images"
    labels_out = Path(output_dir) / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Process each image
    image_files = list(Path(unlabeled_dir).glob("*.jpg")) + \
                  list(Path(unlabeled_dir).glob("*.png"))

    for img_path in image_files:
        print(f"Processing {img_path.name}...")

        # Load image
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # Run DINO detection
        result = detector.detect(
            image,
            text_queries=["complete weed plant"],
            confidence_threshold=confidence_threshold,
        )

        # Skip images with no detections
        if result.count == 0:
            continue

        # Convert to YOLO format (class_id, x_center, y_center, width, height)
        # All values normalized to 0-1
        yolo_labels = []
        for detection in result.detections:
            box = detection.box
            x_center = (box.x_min + box.x_max) / 2
            y_center = (box.y_min + box.y_max) / 2
            box_width = box.x_max - box.x_min
            box_height = box.y_max - box.y_min

            # Map label to class ID
            class_id = get_class_id(detection.label)

            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

        # Save image (copy to output)
        image.save(images_out / img_path.name)

        # Save YOLO label file
        label_file = labels_out / f"{img_path.stem}.txt"
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_labels))

    print(f"Auto-labeled {len(list(labels_out.glob('*.txt')))} images")


def get_class_id(label: str) -> int:
    """Map weed label to YOLO class ID."""
    class_map = {
        "dandelion": 0,
        "dande": 0,
        "dandelion_complete": 0,
        "clover": 1,
        "clover_complete": 1,
        "crabgrass": 2,
        "crab": 2,
        "crabgrass_complete": 2,
        "poa": 3,
        "poa_annua": 3,
        "poa_complete": 3,
        "weed": 0,  # Default class for single-class model
        "complete_weed": 0,
    }
    return class_map.get(label.lower(), 0)


if __name__ == "__main__":
    auto_label_images(
        unlabeled_dir="data/unlabeled_lawns",
        output_dir="data/yolo_dataset",
        confidence_threshold=0.4,
    )
```

#### Phase 3: Train YOLOv8 on Pseudo-Labels

```python
# scripts/train_yolo_from_pseudolabels.py

from ultralytics import YOLO

def train_yolo_student():
    """Train YOLOv8 on pseudo-labeled dataset from DINO."""

    # Load pre-trained YOLOv8 (choose size based on deployment target)
    # yolov8n.pt = nano (fastest, mobile)
    # yolov8s.pt = small (fast, edge)
    # yolov8m.pt = medium (balanced)
    # yolov8l.pt = large (accurate)
    # yolov8x.pt = extra-large (most accurate)

    model = YOLO("yolov8m.pt")  # Medium - good balance

    # Train on pseudo-labeled dataset
    results = model.train(
        data="data/yolo_dataset/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,  # GPU
        workers=8,
        patience=20,  # Early stopping
        save=True,
        plots=True,

        # Augmentation (important for pseudo-labels)
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,

        # Learning rate
        lr0=0.01,
        lrf=0.01,

        # Regularization
        weight_decay=0.0005,
        dropout=0.1,
    )

    # Export for deployment
    model.export(format="onnx")  # For general deployment
    model.export(format="coreml")  # For iOS
    model.export(format="tflite")  # For Android

    return results


if __name__ == "__main__":
    train_yolo_student()
```

#### Dataset YAML for YOLOv8

```yaml
# data/yolo_dataset/dataset.yaml

path: /path/to/data/yolo_dataset
train: images/train
val: images/val

# Class names (must match pseudo-labeling)
names:
  0: dandelion_complete
  1: clover_complete
  2: crabgrass_complete
  3: poa_complete

# Or for single-class:
# names:
#   0: complete_weed
```

### Quality Control for Pseudo-Labels

Pseudo-labels aren't perfect. Use these strategies to improve quality:

| Strategy | Description |
|----------|-------------|
| **High threshold** | Use 0.4-0.5 confidence threshold (higher = fewer but cleaner labels) |
| **Manual review** | Spot-check 5-10% of pseudo-labels for errors |
| **Confidence weighting** | Weight training loss by pseudo-label confidence |
| **Self-training** | Iteratively re-train DINO on corrected pseudo-labels |
| **Ensemble** | Use multiple DINO runs, keep consistent detections |

### Expected Results

| Metric | Fine-tuned DINO | YOLOv8 Student |
|--------|-----------------|----------------|
| mAP@0.5 | ~0.75 | ~0.70-0.73 |
| Speed (GPU) | ~8 FPS | ~60 FPS |
| Speed (CPU) | ~1 FPS | ~15 FPS |
| Model size | ~350MB | ~25MB |
| Mobile ready | ❌ | ✅ |

The student typically achieves 90-95% of the teacher's accuracy at 5-10x the speed.

### Deployment Options for YOLOv8

| Target | Export Format | Command |
|--------|---------------|---------|
| Server (GPU) | PyTorch | `model.export(format="torchscript")` |
| Server (CPU) | ONNX | `model.export(format="onnx")` |
| iOS | CoreML | `model.export(format="coreml")` |
| Android | TFLite | `model.export(format="tflite")` |
| Edge (Jetson) | TensorRT | `model.export(format="engine")` |
| Web | ONNX + onnxruntime-web | `model.export(format="onnx")` |

---

## Alternative: Direct YOLOv8 Fine-Tuning

If text grounding isn't needed, YOLOv8 is simpler:

```python
from ultralytics import YOLO

# Load pre-trained YOLOv8
model = YOLO("yolov8l.pt")  # Large model

# Train on your dataset
model.train(
    data="weed_dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,  # GPU
)

# Export for deployment
model.export(format="onnx")
```

### YOLOv8 Dataset Format

```yaml
# weed_dataset.yaml
path: /path/to/dataset
train: images/train
val: images/val

names:
  0: dandelion_complete
  1: clover_complete
  2: crabgrass_complete
```

---

## Quick Reference: Commands

### Label Studio Setup
```bash
pip install label-studio
label-studio start
# Open http://localhost:8080
```

### Training (DINO)
```bash
python scripts/train_dino.py \
    --data_dir data/weed_dataset \
    --output_dir models/weed-dino-finetuned \
    --epochs 20 \
    --batch_size 4
```

### Training (YOLOv8)
```bash
yolo detect train \
    data=weed_dataset.yaml \
    model=yolov8l.pt \
    epochs=50 \
    imgsz=640
```

### Evaluation
```bash
python scripts/evaluate.py \
    --model models/weed-dino-finetuned \
    --test_data data/test
```

---

## Next Steps

### Option A: DINO-Only (keeps text grounding)
1. [ ] Decide: Single-class or multi-class labeling
2. [ ] Set up Label Studio
3. [ ] Collect/organize training images
4. [ ] Label 100-200 images (start small, iterate)
5. [ ] Run initial training
6. [ ] Evaluate results
7. [ ] Label more images if needed
8. [ ] Final training and deployment

### Option B: DINO → YOLOv8 Pipeline (recommended for production)
1. [ ] Decide: Single-class or multi-class labeling
2. [ ] Set up Label Studio
3. [ ] Collect/organize 100-500 images for manual labeling
4. [ ] Collect 5,000+ unlabeled lawn images
5. [ ] Label 100-500 images manually
6. [ ] Fine-tune DINO on manual labels
7. [ ] Evaluate DINO (must be accurate before proceeding)
8. [ ] Run auto-labeling script on unlabeled images
9. [ ] Spot-check 5-10% of pseudo-labels
10. [ ] Train YOLOv8 on pseudo-labeled dataset
11. [ ] Evaluate YOLOv8 vs DINO
12. [ ] Export YOLOv8 for target platform (mobile, edge, web)
13. [ ] Deploy fast YOLOv8 model to production

---

## Resources

- **Grounding DINO Paper:** https://arxiv.org/abs/2303.05499
- **HuggingFace Fine-Tuning Guide:** https://huggingface.co/docs/transformers/training
- **Label Studio Docs:** https://labelstud.io/guide/
- **YOLOv8 Docs:** https://docs.ultralytics.com/
- **COCO Format Spec:** https://cocodataset.org/#format-data
