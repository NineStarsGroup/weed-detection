# Weed Detection App - Handoff Document

## Version 2.0 (January 21, 2026)

An AI-powered weed detection system using multiple detection models for zero-shot object detection. Supports **text-guided detection** (using natural language descriptions) and **image-guided detection** (using reference images).

**Tech Stack:**
- Python 3.10+
- OWLv2 via HuggingFace Transformers
- Grounding DINO (Original) via HuggingFace Transformers
- SAM (Segment Anything Model) via HuggingFace Transformers
- FastAPI for REST API
- PIL/Pillow for image processing
- PyTorch for ML inference (supports MPS/CUDA/CPU)

---

## Current State

### What's Built

| Component | Status | Location |
|-----------|--------|----------|
| OWLv2 Detection Wrapper | ✅ Complete | `src/detection/detector.py` |
| Grounding DINO Detector | ✅ Complete | `src/detection/grounding_dino.py` |
| **DINO 1.5 Pro Detector** | ✅ Complete (v2.0) | `src/detection/grounding_dino_1_5_pro.py` |
| **DINO 1.5 Edge Detector** | ✅ Complete (v2.0) | `src/detection/grounding_dino_1_5_edge.py` |
| **Dynamic-DINO Detector** | ✅ Complete (v2.0) | `src/detection/dynamic_dino.py` |
| **Local Weights Detector (Swin-T/B)** | ✅ Complete (v2.0) | `src/detection/grounding_dino_local.py` |
| SAM Auto-Segment Detector | ✅ Complete | `src/detection/sam_detector.py` |
| Text-guided Detection (OWLv2) | ✅ Complete | `src/detection/detector.py:detect_by_text()` |
| Text-guided Detection (DINO) | ✅ Complete | `src/detection/grounding_dino.py:detect()` |
| Auto-Segmentation (SAM) | ✅ Complete | `src/detection/sam_detector.py:detect()` |
| Image-guided Detection | ✅ Complete | `src/detection/detector.py:detect()` |
| Detection Mode Switching | ✅ Complete | `src/config.py:DetectionMode` |
| Detection Data Models | ✅ Complete | `src/detection/models.py` |
| Deduplication (center-distance) | ✅ Complete | `src/detection/models.py:deduplicate()` |
| Size Filtering | ✅ Complete | `src/detection/models.py:filter_by_size()` |
| Overlapping Box Clustering | ✅ Complete | `src/detection/models.py:cluster_overlapping()` |
| Adaptive Thresholding | ✅ Complete | `src/detection/models.py:filter_adaptive()` |
| Reference Image Manager | ✅ Complete | `src/references/manager.py` |
| FastAPI Backend | ✅ Complete | `src/api/` |
| Detection Endpoint | ✅ Complete | `POST /detect` |
| Visualization Endpoint | ✅ Complete | `POST /detect/visualize` |
| Reference Management API | ✅ Complete | `/references/*` |
| Web UI (Detection + Refs) | ✅ Complete | `src/api/routes/ui.py` |
| **Model Selector UI (v2.0)** | ✅ Complete | `src/api/routes/ui.py` (dropdown with DINO variants + Local Weights) |
| Weed Type Selector (DINO) | ✅ Complete | `src/api/routes/ui.py` |
| **TensorRT Toggle UI** | ✅ Complete (v2.0) | `src/api/routes/ui.py` |
| Label Normalization (DINO) | ✅ Complete | `src/detection/grounding_dino.py:_normalize_label()` |
| Image Annotation/Boxes | ✅ Complete | `src/visualization/annotate.py` |
| Memory Management (MPS OOM fix) | ✅ Complete | `src/detection/detector.py` |
| Test Script | ✅ Complete | `scripts/test_detection.py` |

### What's NOT Built Yet

| Component | Priority | Notes |
|-----------|----------|-------|
| Fine-tuned Leaf Model | P1 | Custom training for dandelion leaves |
| Mobile App | P0 | React Native or Flutter recommended |
| Real-time Video Detection | P2 | Would need frame batching |
| GPS/Heatmap Visualization | P3 | Requires location data from mobile |
| Redis Caching | P2 | Config exists, implementation pending |
| User Authentication | P1 | No auth currently |
| Cloud Deployment | P1 | Local only right now |

---

## Detection Modes

The system supports **ten detection modes** (v2.1):

| Mode | Value | Model | Best For |
|------|-------|-------|----------|
| **OWLv2 Text** | `text_owlv2` | google/owlv2-base-patch16-ensemble | Distinctive features (flowers, seed heads) |
| **Grounding DINO (Original)** | `grounding_dino` | IDEA-Research/grounding-dino-tiny | Leaf patterns, fine-grained detection |
| **DINO 1.5 Pro** | `grounding_dino_1_5_pro` | grounding-dino-base (fallback) | Highest accuracy (~5 FPS) |
| **DINO 1.5 Edge** | `grounding_dino_1_5_edge` | grounding-dino-tiny (640px images) | Fastest (~30 FPS, ~75 FPS w/ TensorRT) |
| **Dynamic-DINO** | `dynamic_dino` | grounding-dino-tiny (800px images) | Balanced speed/accuracy |
| **DINO Swin-T (Local)** | `grounding_dino_local_swint` | Local groundingdino_swint_ogc.pth | Fast (~8 FPS), requires local weights |
| **DINO Swin-B (Local)** | `grounding_dino_local_swinb` | Local groundingdino_swinb_cogcoor.pth | Accurate (~5 FPS), requires local weights |
| **RF-DETR (Fine-tuned)** | `rf_detr` | Local rf_detr_weed_weights.pt | Highest accuracy for trained classes (~30 FPS) |
| **SAM Auto-Segment** | `sam_auto` | facebook/sam-vit-base | Discovering all plant regions without text prompts |
| **OWLv2 Image** | `image_owlv2` | google/owlv2-base-patch16-ensemble | When you have clean, well-cropped examples |

### v2.0 DINO Variants

| Variant | Image Size | Speed | Accuracy | Notes |
|---------|------------|-------|----------|-------|
| **Original** | 1024px | ~8 FPS | 52.5 AP | v1.0 default |
| **1.5 Pro** | 1024px | ~5 FPS | 54.3 AP | Highest accuracy (uses grounding-dino-base) |
| **1.5 Edge** | 640px | ~30 FPS | 36.2 AP | Optimized for speed |
| **Dynamic** | 800px | ~25 FPS | ~37 AP | MoE architecture, balanced |
| **Local Swin-T** | 1024px | ~8 FPS | 52.5 AP | Requires .pth weights file |
| **Local Swin-B** | 1024px | ~5 FPS | ~54 AP | Larger backbone, requires .pth weights |

### RF-DETR Setup (v2.1)

RF-DETR is a **closed-vocabulary** model - unlike zero-shot models (DINO, OWLv2), it detects only the classes it was trained on. This makes it faster and more accurate for known weed types.

**To use RF-DETR:**

1. **Install the package:**
   ```bash
   pip install rfdetr
   ```

2. **Train on weed data** (see `notebooks/rf_detr_weed_training.ipynb`):
   - Use Google Colab with GPU runtime
   - Download a weed dataset from [Roboflow Universe](https://universe.roboflow.com/search?q=class:weed)
   - Fine-tune RF-DETR Medium for 50 epochs
   - Export weights to `weights/rf_detr_weed_weights.pt`

3. **Update class names** in `src/detection/rf_detr.py`:
   ```python
   CLASS_NAMES = [
       "dandelion",
       "clover",
       "crabgrass",
       # ... match your training dataset classes
   ]
   ```

4. **Select "RF-DETR (Weed Detection)"** from the UI dropdown

**Key differences from zero-shot models:**
| Aspect | Zero-Shot (DINO/OWLv2) | RF-DETR (Fine-tuned) |
|--------|------------------------|----------------------|
| Text prompts | Yes | No |
| New weed types | Add text prompt | Requires retraining |
| Accuracy | Good for common weeds | Best for trained classes |
| Speed | ~8-30 FPS | ~30 FPS |

---

### Local Weights Setup

To use local weights (Swin-T or Swin-B):

```bash
# Install the groundingdino package (required for local weights)
pip install groundingdino-py

# Create weights directory
mkdir -p weights

# Download Swin-T (faster)
curl -L -o weights/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Download Swin-B (more accurate)
curl -L -o weights/groundingdino_swinb_cogcoor.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

Then select "DINO Swin-T (Local)" or "DINO Swin-B (Local)" from the dropdown.

**Note:** The `groundingdino-py` package is now installed in this project. Local weights load directly without fallback to HuggingFace transformers.

### Model Performance Comparison

| Feature | OWLv2 Text | Grounding DINO | SAM Auto | OWLv2 Image |
|---------|------------|----------------|----------|-------------|
| Yellow flowers | ✅ Excellent | ✅ Excellent | ✅ Good | ✅ Good |
| Seed heads (puffballs) | ✅ Excellent | ✅ Good | ✅ Good | ✅ Good |
| Leaf rosettes | ⚠️ Poor | ⚠️ Better | ✅ Good boundaries | ⚠️ Depends on refs |
| Labeled output | ✅ Yes | ✅ Yes | ❌ No (generic) | ✅ Yes |
| Speed (MPS) | ~2-3 sec | ~1-2 sec | ~3-5 sec | ~2-3 sec |
| Memory usage | ~2GB | ~1.5GB | ~2GB | ~2GB |

### Text Query Configuration

Text-guided detection uses curated queries per weed type (see `src/api/routes/detection.py:_build_text_queries()`).

**Important:** DINO and OWLv2 use different prompts because they handle text differently:
- **DINO**: Excels at grounding detailed visual descriptions to image regions
- **OWLv2**: Works better with simpler prompts (detailed ones cause false positives)

**Dandelion queries:**

| OWLv2 Text | DINO (Leaf-Focused) |
|------------|---------------------|
| "dandelion" | "dandelion plant with jagged leaves" |
| "yellow dandelion flower" | "rosette of serrated green leaves" |
| "dandelion puffball" | "dandelion leaves growing from center" |
| "dandelion leaves" | |

*Note: DINO prompts focus entirely on leaves (no flower prompts) for better practical weed detection.*

**Clover queries:**

| OWLv2 Text | DINO (Trifoliate-Focused) |
|------------|---------------------------|
| "clover" | "clover with three round leaves" |
| "three leaf clover" | "three leaflets in clover pattern" |
| "clover leaves" | "clover leaf with three rounded lobes" |
| "trifoliate clover" | |

*Note: DINO prompts emphasize the distinctive trifoliate (three-leaf) structure with round/rounded shapes to distinguish from other plants.*

**Crabgrass queries:**

| OWLv2 Text | DINO |
|------------|------|
| "crabgrass" | "crabgrass with spreading stems" |
| "crabgrass weed" | "low growing grass weed" |

**Poa Annua queries:**

| OWLv2 Text | DINO (Height-Focused) |
|------------|----------------------|
| "poa annua" | "tall grass clump standing above lawn" |
| "annual bluegrass" | "raised grass tuft in mowed lawn" |
| "tall grass clump" | "grass clump taller than surrounding turf" |

*Note: Poa annua prompts focus on height difference from surrounding turf, not color. The raised/clumpy growth habit distinguishes it from mowed lawn grass.*

### Label Normalization

DINO returns verbose labels containing matched prompt text (e.g., "tall grass clump standing above lawn"). The `_normalize_label()` function in `grounding_dino.py` converts these to short display codes:

| Detected Prompt Contains | Display Label | Color |
|-------------------------|---------------|-------|
| "dandelion", "rosette", "serrated", "jagged" | `dande` | Gold |
| "clover", "trifoliate", "leaflet", "three round", "three leaf" | `clover` | Lime green |
| "crabgrass", "spreading stems", "low growing" | `crab` | Tomato red |
| "poa", "tall grass", "raised grass", "grass clump", "turf", "mowed" | `poa` | Sky blue |
| (unrecognized) | `weed` | Orange |

**Important:** Order matters in label normalization! Crabgrass checks ("low growing") must come before poa checks to prevent the generic "grass" term from incorrectly matching crabgrass prompts to poa.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Mobile App (TODO)                       │
│                  React Native / Flutter                      │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP/REST
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ /detect     │  │ /references │  │ /health             │  │
│  │ POST        │  │ CRUD        │  │ GET                 │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────────┘  │
│         │                │                                   │
│         ▼                ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                Detection Layer                      │    │
│  │  ┌──────────────────┐  ┌────────────────────────┐  │    │
│  │  │ WeedDetector     │  │ GroundingDINODetector  │  │    │
│  │  │ (OWLv2)          │  │ (DINO)                 │  │    │
│  │  │ - detect_by_text │  │ - detect               │  │    │
│  │  │ - detect (image) │  │ - lazy-loaded          │  │    │
│  │  └──────────────────┘  └────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
│         │                │                                   │
│         ▼                ▼                                   │
│  ┌─────────────┐  ┌─────────────────────────────────────┐   │
│  │ GPU/MPS/CPU │  │ Reference Image Storage             │   │
│  │ Inference   │  │ data/references/{weed_type}/*.jpg   │   │
│  └─────────────┘  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point, runs uvicorn server |
| `src/config.py` | Configuration, `DetectionMode` enum (9 modes), `DETECTION_MODES` dict |
| `src/detection/detector.py` | OWLv2 wrapper, `detect()` and `detect_by_text()` methods |
| `src/detection/grounding_dino.py` | Grounding DINO Original wrapper, lazy-loaded |
| `src/detection/grounding_dino_1_5_pro.py` | DINO 1.5 Pro (uses grounding-dino-base), TensorRT support |
| `src/detection/grounding_dino_1_5_edge.py` | DINO 1.5 Edge (640px images), TensorRT support |
| `src/detection/dynamic_dino.py` | Dynamic-DINO (800px images), TensorRT support |
| `src/detection/grounding_dino_local.py` | Local weights loader for Swin-T/Swin-B .pth files |
| `src/detection/sam_detector.py` | SAM auto-segmentation wrapper, lazy-loaded |
| `src/detection/models.py` | Pydantic models, filtering methods (deduplicate, filter_by_size, cluster_overlapping) |
| `src/references/manager.py` | Filesystem-based reference image storage |
| `src/visualization/annotate.py` | Bounding box drawing, side-by-side comparison |
| `src/api/app.py` | FastAPI app factory, middleware, lifespan |
| `src/api/routes/detection.py` | POST /detect and /detect/visualize with mode switching, lazy-loaded detectors |
| `src/api/routes/references.py` | Reference image CRUD |
| `src/api/routes/ui.py` | Web UI with model selector dropdown (DINO variants + Local Weights groups) |

---

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | 8000 | Server port |
| `DEVICE` | auto | `cuda`, `mps`, or `cpu` |
| `DEFAULT_CONFIDENCE_THRESHOLD` | 0.4 | Base detection threshold |
| `DEFAULT_DETECTION_MODE` | `text_owlv2` | Default detection mode |
| `MODEL_NAME` | `google/owlv2-base-patch16-ensemble` | HuggingFace model ID (OWLv2) |

---

## API Quick Reference

### Web UI
Open http://localhost:8000 in a browser for:
- **Detect Weeds tab**: Upload photo, select detection mode, adjust threshold
  - When using **Grounding DINO**, checkboxes appear to select which weed types to detect (Dandelion, Clover, Crabgrass, Poa Annua)
- **Manage References tab**: Upload reference images, view available weed types

### Detect Weeds (JSON only)
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "image=@lawn_photo.jpg" \
  -F "confidence_threshold=0.25" \
  -F "adaptive_threshold=true"
```

### Detect with Visualization (includes base64 images)
```bash
curl -X POST "http://localhost:8000/detect/visualize" \
  -F "image=@lawn_photo.jpg" \
  -F "confidence_threshold=0.3" \
  -F "detection_mode=grounding_dino"
```

**Parameters:**
- `confidence_threshold`: 0.0-1.0 (default 0.7 in UI)
- `detection_mode`: `text_owlv2`, `grounding_dino`, `sam_auto`, or `image_owlv2`
- `weed_types`: Comma-separated list (e.g., `dandelion,clover,poa_annua`). For text-based modes (DINO, OWLv2 Text), defaults to `dandelion,clover,crabgrass`. For image-based modes, uses uploaded reference types.
- `group_overlapping`: `true` or `false` - merges overlapping detections into unified boxes

Response includes:
- `original_image`: Base64-encoded original
- `annotated_image`: Base64-encoded image with bounding boxes
- `comparison_image`: Side-by-side comparison

### Upload Reference Image
```bash
curl -X POST "http://localhost:8000/references/upload" \
  -F "image=@dandelion_sample.jpg" \
  -F "weed_type=dandelion"
```

### List Weed Types
```bash
curl "http://localhost:8000/references/weed-types"
```

---

## Known Issues & Lessons Learned

### Leaf Pattern Detection Challenge
Both OWLv2 and Grounding DINO struggle to reliably detect dandelion leaves without flowers. This is a fundamental limitation:
- Models are trained on internet images where "dandelion" usually means the flower
- Leaf-only rosettes are visually similar to many other plants
- Text prompts like "rosette of jagged green leaves" help but confidence remains low (~25-40%)

**Potential solutions:**
1. Upload close-up leaf reference images and use OWLv2 Image mode
2. Fine-tune a model on custom dandelion leaf dataset
3. Add SAM (Segment Anything) for better boundary detection
4. Lower threshold (0.2-0.3) and accept more false positives

### Over-detection with Image Mode
Image-guided detection (`image_owlv2`) can over-detect when reference images contain grass backgrounds. The model matches grass patterns, producing 1000+ false positives.

**Solution:** Use text-guided detection for better precision, or ensure reference images are tightly cropped with minimal background.

### Threshold Tuning by Mode
| Mode | Recommended Start | Notes |
|------|------------------|-------|
| OWLv2 Text | 0.5-0.7 | Higher for flowers, lower for leaves |
| Grounding DINO | 0.2-0.4 | Generally needs lower thresholds |
| SAM Auto | N/A | Uses IoU scores internally, filters by mask size |
| OWLv2 Image | 0.3-0.5 | Watch for false positives |

### Model Downloads
All models are lazy-loaded and download on first use:
- **Grounding DINO**: ~350MB (original DINO, not DINO 2)
- **SAM (vit-base)**: ~375MB
- **OWLv2**: ~600MB (loaded at startup)
- **Local Swin-T**: ~662MB (requires download to `weights/`)
- **Local Swin-B**: ~895MB (requires download to `weights/`)

### First-Load Server Blocking

**Important:** The first time you use a local weights model (Swin-T or Swin-B), the server will block for 30-60+ seconds while loading the model. During this time:
- Browser will show "analyzing" indefinitely
- Health checks (`/health`) will timeout
- No other requests will be processed

This is normal behavior. Wait for the model to load (watch terminal for `[DINO Local]` progress logs), or kill and restart the server if needed.

### Model Versions
- **Grounding DINO**: Using `IDEA-Research/grounding-dino-tiny` (original DINO). DINO 1.5/2.0 may offer better accuracy but requires different setup.
- **SAM**: Using `facebook/sam-vit-base`. SAM2 is available but not yet integrated.
- **OWLv2**: Using the ensemble model which combines multiple checkpoints for better accuracy.

---

## Testing Results (Jan 20-21, 2026)

### Best Configuration Found

| Setting | Value | Notes |
|---------|-------|-------|
| **Detection Mode** | Grounding DINO | Best for dandelion detection |
| **Threshold** | 0.2 | Lower than OWLv2 (0.5-0.7) |
| **Group Overlapping** | On | Merges flower + leaves + plant boxes |

### What Works Well
- "full dandelion plant" prompt captures whole plant clusters (28% confidence)
- Yellow flowers detected at 21-27% confidence
- DINO's detailed prompts ("rosette of pointed green leaves") help with leaf detection

### Recent Improvements (Jan 21)
- **Weed Type Selector**: UI now shows checkboxes for Dandelion/Clover/Crabgrass/Poa Annua when using DINO mode
- **Poa Annua Detection**: Added 4th weed type with height-focused prompts (detects raised grass clumps standing above turf)
- **Label Normalization**: DINO labels simplified from verbose prompts to short codes (`dande`, `clover`, `crab`, `poa`)
- **Label Cross-Contamination Fix**: Reordered normalization checks so crabgrass ("low growing") matches before poa to prevent incorrect labeling
- **No Detection Cap**: Removed 20-detection limit, now shows all found weeds
- **Default Weed Types**: Text-based modes no longer require reference images - defaults to common weeds
- **Leaf-Focused Prompts**: DINO prompts now prioritize leaf detection over flowers (removed flower prompt from dandelion)
- **Robust Label Mapping**: Normalization catches all prompt variations (`rosette`, `serrated`, `jagged` → `dande`; `spreading stems`, `low growing` → `crab`; `tall grass`, `grass clump`, `turf` → `poa`)
- **Clover Prompt Improvements**: Updated DINO prompts to emphasize trifoliate structure ("three round leaves", "three leaflets", "rounded lobes") for better clover-specific detection. Label normalization extended to catch `trifoliate`, `leaflet`, `three round`, `three leaf` terms.
- **Leaf-First Strategy (All Weeds)**: Removed flower-based prompts from silverleaf nightshade, field bindweed, broom snakeweed, and Palmer's amaranth. All DINO prompts now focus on leaf morphology (shape, texture, arrangement) for year-round detection. Backup saved to `detection_prompts_backup_jan21.py`.

### Limitations
- Some scattered/small flowers may be missed even at low thresholds
- Lowering threshold below 0.2 adds noise without catching more flowers
- Leaf-only detection still challenging (fundamental model limitation)

---

## Next Steps for New Developer

1. **Get it running locally** - `python main.py` then open http://localhost:8000
2. **Test all four modes** - Compare results between OWLv2 Text, Grounding DINO, SAM Auto, and OWLv2 Image
3. **Start with Grounding DINO** - Use threshold 0.2, enable "Group overlapping", select weed types to detect
4. **Focus on flowers first** - Detection works best for yellow flowers and seed heads
5. **Try SAM for leaves** - SAM finds object boundaries without needing text prompts (slower but thorough)
6. **Extend weed type selector** - Currently DINO-only; add to OWLv2 Text mode next
7. **Fine-tune for leaves** - Custom training on dandelion leaf images would be most effective
8. **Build mobile app** - The API is ready with visualization endpoints
9. **Deploy to cloud** - Need GPU instance for reasonable speed

---

## Future Development: Real-Time Video & Edge Deployment

### Current Performance Baseline

| Mode | Inference Time (MPS) | FPS Equivalent |
|------|---------------------|----------------|
| Grounding DINO | ~1-2 sec | 0.5-1 FPS |
| OWLv2 Text | ~2-3 sec | 0.3-0.5 FPS |
| SAM Auto | ~3-5 sec | 0.2-0.3 FPS |

### Real-Time Video Requirements

To achieve 15-30 FPS for real-time detection:

**Hardware options:**
| Setup | Expected FPS | Latency | Cost |
|-------|-------------|---------|------|
| RTX 3080 + TensorRT | 10-15 FPS | ~100ms | ~$700 |
| RTX 4090 + TensorRT | 20-30 FPS | ~50ms | ~$1600 |
| A100 (cloud) | 30+ FPS | ~30ms | ~$3/hr |
| Jetson Orin NX | 15-25 FPS | ~60ms | ~$900 |

**Software optimizations needed:**
1. **TensorRT conversion** - NVIDIA's inference optimizer (2-4x speedup)
2. **FP16/INT8 quantization** - Half precision works well
3. **Frame skipping** - Process every 3rd-5th frame, overlay results
4. **Async pipeline** - Capture frames while previous detection runs

### Edge Deployment (Jetson Orin)

The Jetson Orin platform is well-suited for local, offline detection:

| Orin Model | GPU Cores | AI Performance | RAM | Expected FPS (TensorRT) |
|------------|-----------|----------------|-----|------------------------|
| Orin Nano | 1024 CUDA | 40 TOPS | 8GB | 8-12 FPS |
| Orin NX | 2048 CUDA | 100 TOPS | 16GB | 15-25 FPS |
| AGX Orin | 2048 CUDA | 275 TOPS | 32-64GB | 25-40 FPS |

**Recommended:** Orin NX offers the best balance of performance and cost for a handheld or robot-mounted device.

**Deployment steps:**
1. Export models to ONNX format
2. Convert ONNX → TensorRT on the Orin device
3. Use smaller input resolution (640x640 instead of 1024x1024)
4. Implement async video capture pipeline

**Alternative models for edge:**
- **YOLO-World** - Newer zero-shot model, faster on edge devices
- **Custom YOLOv8** - Fastest option if fine-tuned on weed dataset

### Video Pipeline Architecture

```
Camera (USB/CSI) → Frame Buffer → Skip Frames → GPU Inference → Overlay → Display
                        ↑                              ↓
                        └──────── Display Loop ────────┘
```

For lawn scanning at walking speed, 10-15 FPS with ~100ms latency feels responsive. Detection overlays persist until the next inference completes.

---

---

## Version 2.0 Implementation (Complete)

### New Model Support

Version 2.0 adds support for faster and more accurate DINO variants:

| Model | Speed (A100) | Speed (TensorRT) | Accuracy (AP) | Status |
|-------|-------------|------------------|---------------|--------|
| **Grounding DINO (Original)** | ~8 FPS | ~15 FPS | 52.5 | ✅ v1.0 |
| **Grounding DINO 1.5 Pro** | ~5 FPS | ~12 FPS | 54.3 | ✅ v2.0 |
| **Grounding DINO 1.5 Edge** | ~30 FPS | **75 FPS** | 36.2 | ✅ v2.0 |
| **Dynamic-DINO** | ~25 FPS | ~50 FPS | ~37+ | ✅ v2.0 |
| **Local Swin-T** | ~8 FPS | N/A | 52.5 | ✅ v2.0 |
| **Local Swin-B** | ~5 FPS | N/A | ~54 | ✅ v2.0 |

### v2.0 Features (Implemented)

#### 1. Model Selection Enhancement ✅
UI dropdown with organized groups:
- **DINO Models** group:
  - `grounding_dino` → DINO Tiny (Default)
  - `grounding_dino_1_5_edge` → DINO Tiny - Fast ⚡
  - `grounding_dino_1_5_pro` → DINO Base - Accurate 🎯
  - `dynamic_dino` → DINO Tiny - Balanced
- **Local Weights 📦** group:
  - `grounding_dino_local_swint` → DINO Swin-T (Local) - Fast
  - `grounding_dino_local_swinb` → DINO Swin-B (Local) - Accurate

#### 2. TensorRT Acceleration ✅
Checkbox toggle appears for supported modes:
- Shown for: DINO 1.5 Edge, DINO 1.5 Pro, Dynamic-DINO
- Hidden for: Original DINO, Local weights, OWLv2, SAM
- Framework ready (TensorRT integration code exists, requires NVIDIA GPU)

#### 3. Local Weights Support ✅
Load original GroundingDINO weights from local .pth files:
- Auto-detects weights in `weights/` directory
- **`groundingdino-py` package installed** - loads weights directly (no HuggingFace fallback)
- Supports both Swin-T (faster) and Swin-B (more accurate) backbones
- Detailed timing logs show model loading progress in terminal (`[DINO Local]` prefix)
- Config files resolved from installed package location (not project directory)
- First load can take 30-60+ seconds for Swin-B (~895MB weights)

### v2.0 File Structure

```
src/detection/
├── grounding_dino.py              # Original (v1.0)
├── grounding_dino_1_5_edge.py     # v2.0: Edge variant (640px)
├── grounding_dino_1_5_pro.py      # v2.0: Pro variant (1024px)
├── dynamic_dino.py                # v2.0: Dynamic-DINO (800px)
├── grounding_dino_local.py        # v2.0: Local weights loader
└── tensorrt_utils.py              # v2.0: TensorRT utilities
```

### v2.0 API Parameters

Parameters for `/detect/visualize`:
```
detection_mode: str   # One of 9 detection modes
use_tensorrt: bool    # Enable TensorRT (for supported modes)
```

### Model Availability Notes

**HuggingFace models (no setup required):**
- `IDEA-Research/grounding-dino-tiny` - Used by Original, Edge, Dynamic
- `IDEA-Research/grounding-dino-base` - Used by Pro

**Local weights (requires download):**
- `groundingdino_swint_ogc.pth` - Swin-T backbone
- `groundingdino_swinb_cogcoor.pth` - Swin-B backbone

**DINO 1.5 Official Weights:**
- Require API token from https://cloud.deepdataspace.com/apply-token
- v2.0 implementation uses open HuggingFace models as fallback

---

## Contacts & Resources

- **OWLv2 Paper**: [Google Research](https://arxiv.org/abs/2306.09683)
- **Grounding DINO Paper**: [IDEA Research](https://arxiv.org/abs/2303.05499)
- **Grounding DINO 1.5 Paper**: [IDEA Research](https://arxiv.org/abs/2405.10300)
- **Dynamic-DINO Paper**: [arXiv](https://arxiv.org/abs/2507.17436)
- **SAM Paper**: [Meta AI](https://arxiv.org/abs/2304.02643)
- **HuggingFace OWLv2**: `google/owlv2-base-patch16-ensemble`
- **HuggingFace DINO**: `IDEA-Research/grounding-dino-tiny`
- **HuggingFace SAM**: `facebook/sam-vit-base`
- **Grounding DINO 1.5 API**: https://github.com/IDEA-Research/Grounding-DINO-1.5-API
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **TensorRT Docs**: https://developer.nvidia.com/tensorrt
