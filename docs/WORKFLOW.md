# Weed Detection - Workflow Process

## Version 2.0 (January 21, 2026)

---

## End-to-End Detection Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER WORKFLOW                                    │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │ 1. SETUP     │────▶│ 2. TRAIN     │────▶│ 3. DETECT    │
    │              │     │ (Upload Refs) │     │              │
    └──────────────┘     └──────────────┘     └──────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
    Run server &         Upload 5-10          Upload lawn photo,
    open Web UI          reference images     see bounding boxes
    localhost:8000       via References tab   via Detect tab
```

---

## Phase 1: Initial Setup

### Developer Setup
```bash
# 1. Clone and enter project
cd "ai preception"

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Copy environment config
cp .env.example .env

# 5. (Optional) Edit .env for GPU support
# DEVICE=mps   # For Apple Silicon
# DEVICE=cuda  # For NVIDIA GPU

# 6. Run the server
python main.py
```

### First Run Expectations
- First request downloads ~2GB model (one-time)
- Model caches to `~/.cache/huggingface/`
- Subsequent startups are fast

---

## Phase 2: Training (Reference Image Upload)

### Collecting Good Reference Images

```
┌─────────────────────────────────────────────────────────────┐
│                  REFERENCE IMAGE CHECKLIST                   │
├─────────────────────────────────────────────────────────────┤
│ □ Tightly cropped around the weed (minimize background)     │
│ □ Multiple angles: top-down, side view, 45°                 │
│ □ Various lighting: direct sun, shade, overcast             │
│ □ Various growth stages: seedling, mature, flowering        │
│ □ At least 5 images per weed type (10 is better)           │
│ □ Resolution: 300-1000px (auto-resized to 384px for model)  │
└─────────────────────────────────────────────────────────────┘
```

### Upload Process (Web UI - Recommended)

1. Open http://localhost:8000
2. Click "Manage References" tab
3. Enter weed type (e.g., "dandelion")
4. Select image file
5. Click "Upload Reference"
6. Repeat for 5-10 images per weed type

### Upload Process (API)

```bash
# Upload a single reference image
curl -X POST "http://localhost:8000/references/upload" \
  -F "image=@path/to/dandelion_1.jpg" \
  -F "weed_type=dandelion"

# Repeat for all reference images
# Recommended: 5-10 images per weed type
```

### Reference Image Organization

```
data/references/
├── dandelion/
│   ├── a1b2c3d4_e5f6.jpg
│   ├── b2c3d4e5_f6g7.jpg
│   └── ... (5-10 images)
├── clover/
│   └── ...
└── crabgrass/
    └── ...
```

---

## Phase 3: Detection

### Web UI Detection (Recommended)

1. Open http://localhost:8000
2. Stay on "Detect Weeds" tab
3. Drag & drop or click to upload a lawn photo
4. Adjust confidence threshold if needed (default 0.25)
5. Click "Detect Weeds"
6. View results:
   - **Original Image** (left) - your uploaded photo
   - **Detection Result** (right) - same photo with bounding boxes
   - **Side-by-Side Comparison** - both images together
   - **Detected Weeds list** - each detection with confidence bar

### Detection Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Upload      │────▶│ Process     │────▶│ Return      │
│ Lawn Photo  │     │ Detection   │     │ Results     │
└─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │
      │                   │                   │
      ▼                   ▼                   ▼
   JPEG/PNG         OWLv2 inference     JSON + annotated
   auto-resize      + adaptive filter   images with boxes
   to 1024px max    + memory cleanup
```

### API Request (JSON only)

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "image=@lawn_photo.jpg" \
  -F "confidence_threshold=0.25" \
  -F "adaptive_threshold=true" \
  -F "weed_types=dandelion,clover"  # Optional: detect specific types
```

### API Request (with visualization)

```bash
curl -X POST "http://localhost:8000/detect/visualize" \
  -F "image=@lawn_photo.jpg" \
  -F "confidence_threshold=0.25" \
  -F "adaptive_threshold=true"
```

Returns JSON including:
- `original_image` - base64 JPEG of original
- `annotated_image` - base64 JPEG with bounding boxes drawn
- `comparison_image` - side-by-side view

### Response Format

```json
{
  "success": true,
  "result": {
    "detections": [
      {
        "label": "dandelion",
        "confidence": 0.87,
        "box": {
          "x_min": 0.23,
          "y_min": 0.45,
          "x_max": 0.31,
          "y_max": 0.58
        }
      }
    ],
    "image_width": 1280,
    "image_height": 960,
    "inference_time_ms": 1245.5
  }
}
```

### Converting Box Coordinates to Pixels

```javascript
// Client-side conversion
const pixelBox = {
  x: detection.box.x_min * imageWidth,
  y: detection.box.y_min * imageHeight,
  width: (detection.box.x_max - detection.box.x_min) * imageWidth,
  height: (detection.box.y_max - detection.box.y_min) * imageHeight
};
```

---

## Internal Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DETECTION PIPELINE (Internal)                         │
└─────────────────────────────────────────────────────────────────────────┘

    POST /detect
         │
         ▼
    ┌─────────────────┐
    │ Validate Image  │ ─── Reject if not image/*
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Load Reference  │ ─── From data/references/{weed_type}/
    │ Images          │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ OWLv2 Inference │ ─── For each weed type:
    │ (per weed type) │     - Process with query images
    └────────┬────────┘     - Get boxes + scores
             │
             ▼
    ┌─────────────────┐
    │ Adaptive Filter │ ─── If enabled:
    │                 │     - Calculate density
    └────────┬────────┘     - Adjust threshold
             │              - Filter detections
             ▼
    ┌─────────────────┐
    │ Return Results  │ ─── JSON response
    └─────────────────┘
```

---

## Detection Mode Selection

The system supports **nine detection modes** (v2.0). Choose based on your use case:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DETECTION MODE DECISION TREE (v2.0)                   │
└─────────────────────────────────────────────────────────────────────────┘

                    What are you trying to detect?
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
      Flowers/          Leaves/           All plant
      Seed heads        Rosettes          boundaries
            │                 │                 │
            ▼                 ▼                 ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ OWLv2 Text    │ │ Grounding     │ │ SAM Auto      │
    │ Threshold:    │ │ DINO          │ │ No text       │
    │ 0.5-0.7       │ │ Threshold:    │ │ prompts       │
    └───────────────┘ │ 0.2-0.3       │ └───────────────┘
                      └───────┬───────┘
                              │
                              ▼
                    Which DINO variant? (v2.0)
                              │
            ┌─────────┬───────┼───────┬─────────┐
            │         │       │       │         │
            ▼         ▼       ▼       ▼         ▼
       Original    1.5 Edge  1.5 Pro  Dynamic  Local
       (default)   (fast)   (accurate)(balanced)(weights)
            │         │       │       │         │
            ▼         ▼       ▼       ▼         ▼
       ~8 FPS     ~30 FPS   ~5 FPS  ~25 FPS   Swin-T/B
       1024px      640px    1024px   800px    .pth files
```

### Mode Comparison (v2.0)

| Mode | Best For | Threshold | Speed | Notes |
|------|----------|-----------|-------|-------|
| **OWLv2 Text** | Flowers, seed heads | 0.5-0.7 | ~2-3 sec | |
| **DINO (Original)** | Leaves, fine details | 0.2-0.3 | ~1-2 sec | Default |
| **DINO 1.5 Edge** | Real-time detection | 0.2-0.3 | ~30 FPS | 640px images |
| **DINO 1.5 Pro** | Maximum accuracy | 0.2-0.3 | ~5 FPS | Uses grounding-dino-base |
| **Dynamic-DINO** | Balanced speed/accuracy | 0.2-0.3 | ~25 FPS | 800px images |
| **DINO Swin-T (Local)** | Local weights, faster | 0.2-0.3 | ~8 FPS | Requires .pth |
| **DINO Swin-B (Local)** | Local weights, accurate | 0.2-0.3 | ~5 FPS | Requires .pth |
| **SAM Auto** | All plant boundaries | N/A | ~3-5 sec | |
| **OWLv2 Image** | Clean reference matching | 0.3-0.5 | ~2-3 sec | |

### DINO Variant Selection Guide (v2.0)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DINO VARIANT SELECTION (v2.0)                         │
└─────────────────────────────────────────────────────────────────────────┘

                    What's your priority?
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
        Speed            Accuracy           Balance
            │                 │                 │
            ▼                 ▼                 ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ 1.5 Edge      │ │ 1.5 Pro or    │ │ Dynamic-DINO  │
    │ ~30-75 FPS    │ │ Local Swin-B  │ │ or Original   │
    │ (w/ TensorRT) │ │ ~5 FPS        │ │ ~8-25 FPS     │
    └───────────────┘ └───────────────┘ └───────────────┘
                              │
                              ▼
                    Have local .pth weights?
                              │
                    ┌─────────┴─────────┐
                    │ Yes               │ No
                    ▼                   ▼
            ┌───────────────┐   Use HuggingFace
            │ Local Swin-T  │   models (DINO 1.5
            │ or Swin-B     │   Pro/Edge/Dynamic)
            └───────────────┘
```

---

## Threshold Tuning Guide

| Scenario | Recommended Threshold | Adaptive |
|----------|----------------------|----------|
| First test | 0.25 | On |
| Missing obvious weeds | 0.15 | On |
| Too many false positives | 0.35 | On |
| Consistent lighting | 0.25 | Off |
| Variable lighting (sun/shade) | 0.20 | On |
| Low quality reference images | 0.15 | On |

### Model-Specific Thresholds

| Mode | Start | Lower if missing | Raise if noisy |
|------|-------|------------------|----------------|
| Grounding DINO | 0.2 | 0.15 | 0.3 |
| OWLv2 Text | 0.5 | 0.3 | 0.7 |
| OWLv2 Image | 0.3 | 0.2 | 0.5 |

---

## Group Overlapping Feature

Enable "Group overlapping" when you see **multiple boxes on the same plant**.

**Use case:** DINO often detects flower, leaves, and full plant separately. Grouping merges these into one unified bounding box.

```
Before grouping:              After grouping:
┌──────┐                      ┌─────────────┐
│flower│                      │             │
├──────┴───┐                  │   unified   │
│  leaves  │       ──▶        │   region    │
├──────────┴───┐              │             │
│ full plant   │              └─────────────┘
└──────────────┘              (1 detection)
(3 detections)
```

---

## Troubleshooting

### No Detections
1. Check reference images exist: `GET /references/weed-types`
2. Lower threshold to 0.1 to see if anything detected
3. Verify reference images are properly cropped
4. Try different/more reference images

### Too Many False Positives
1. Raise threshold to 0.35-0.4
2. Improve reference image quality (tighter crops)
3. Add more diverse reference images

### Slow Inference
1. Check device: `GET /health` shows current device
2. Enable GPU: set `DEVICE=cuda` or `DEVICE=mps` in .env
3. Resize input images to max 1280px before upload
4. Reduce number of weed types per request

### Out of Memory
1. Images are now auto-resized (max 1024px target, 384px refs) - should be rare
2. If still occurs: reduce number of weed types per request
3. Restart server to clear GPU memory
4. For MPS: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` disables memory limit (risky)

---

## v2.0 Local Weights Setup

### Downloading Local Weights

The "Local Weights" options require downloading the original GroundingDINO weights:

```bash
# Create weights directory
mkdir -p weights

# Download Swin-T (faster, ~340MB)
curl -L -o weights/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Download Swin-B (more accurate, ~690MB)
curl -L -o weights/groundingdino_swinb_cogcoor.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

### Verifying Weights

```bash
# Check weights are present
ls -la weights/
# Should show:
# groundingdino_swint_ogc.pth (~340MB)
# groundingdino_swinb_cogcoor.pth (~690MB)
```

**Note:** If you don't have the weights file, selecting "DINO Swin-T (Local)" or "DINO Swin-B (Local)" will show an error with download instructions.

---

## v2.0 TensorRT Acceleration

### When TensorRT Option Appears

The "TensorRT Acceleration" checkbox is shown for:
- DINO 1.5 Edge
- DINO 1.5 Pro
- Dynamic-DINO

### TensorRT Requirements

- NVIDIA GPU with CUDA support
- TensorRT installed (`pip install tensorrt`)
- First inference triggers model conversion (cached)

### Performance with TensorRT

| Scenario | Without TensorRT | With TensorRT | Improvement |
|----------|-----------------|---------------|-------------|
| Desktop (RTX 4090) | ~8 FPS | ~75 FPS | 9x faster |
| Desktop (RTX 3080) | ~6 FPS | ~25 FPS | 4x faster |
| Edge (Jetson Orin NX) | ~3 FPS | ~15 FPS | 5x faster |
| Cloud (A100) | ~8 FPS | ~75 FPS | 9x faster |

---

## v2.0 API Parameters

```bash
# v2.0 API call with new options
curl -X POST "http://localhost:8000/detect/visualize" \
  -F "image=@lawn_photo.jpg" \
  -F "confidence_threshold=0.2" \
  -F "detection_mode=grounding_dino_1_5_edge" \  # DINO variant
  -F "use_tensorrt=true"                          # TensorRT acceleration
```

Available detection_mode values (v2.0):
- `text_owlv2` - OWLv2 Text Detection
- `grounding_dino` - DINO Tiny (Original)
- `grounding_dino_1_5_edge` - DINO Tiny (640px, fast)
- `grounding_dino_1_5_pro` - DINO Base (1024px, accurate)
- `dynamic_dino` - DINO Tiny (800px, balanced)
- `grounding_dino_local_swint` - Local Swin-T weights
- `grounding_dino_local_swinb` - Local Swin-B weights
- `sam_auto` - SAM Auto-Segment
- `image_owlv2` - OWLv2 Image Detection
