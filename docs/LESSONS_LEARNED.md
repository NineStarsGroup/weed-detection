# Lessons Learned - Weed Detection Project

## Version 2.0 (January 21, 2026)

---

## What Works Well

### 1. OWLv2 for Few-Shot Detection
**Why it works:** No training required. Upload reference images → detect immediately.

- Users can add new weed types without ML expertise
- Works with as few as 3-5 reference images per weed type
- The `google/owlv2-base-patch16-ensemble` model balances speed and accuracy well

**Key insight:** Image-guided detection (showing the model what to find) is more intuitive than text-based queries for visual tasks like weed identification.

---

### 2. Adaptive Thresholding
**Why it works:** Fixed thresholds fail across varying conditions.

- Bright sunlight → many false positives → adaptive raises threshold
- Shadows/overcast → missed detections → adaptive lowers threshold
- Log-scale adjustment prevents over-correction

**The math that works:**
```python
adjustment = log2(density_ratio) * 0.05
effective_threshold = base_threshold + adjustment
```

---

### 3. Lazy Model Loading
**Why it works:** 2GB model download shouldn't block app startup.

- Model loads on first detection request
- Warmup can run in background after startup
- Users see responsive UI immediately

---

### 4. Normalized Bounding Boxes
**Why it works:** Client-agnostic coordinates.

- Boxes stored as 0-1 floats, not pixels
- Works regardless of display resolution
- Easy to convert: `pixel_x = normalized_x * image_width`

---

## What Doesn't Work / Pitfalls

### 1. Very Low Thresholds (< 0.1)
**Problem:** Floods results with false positives.

- Grass texture triggers many low-confidence detections
- 0.25 is a better starting point than 0.1
- Adaptive mode helps, but garbage-in still means garbage-out

**Recommendation:** Start conservative (0.25-0.3), lower if missing obvious weeds.

---

### 2. Poor Reference Images
**Problem:** Garbage reference images = garbage detections.

Common mistakes:
- Reference images with too much background (not cropped to weed)
- All reference images from same angle/lighting
- Blurry or low-resolution reference images
- Reference images of wrong growth stage

**Good reference images:**
- Tightly cropped around the weed
- Various angles (top-down, side, 45°)
- Various lighting (sun, shade, overcast)
- Various growth stages (seedling, mature, flowering)
- 5-10 images per weed type is the sweet spot

---

### 3. Processing Each Weed Type Separately
**Problem:** Scales linearly with weed types.

Current approach loops through each weed type:
```python
for label, refs in reference_images.items():
    # Run inference for this type
```

- 5 weed types = 5x inference time
- Could batch, but OWLv2's image-guided mode expects focused queries

**Workaround:** Limit to 3-4 most common weed types per scan, or accept longer inference times.

---

### 4. CPU Inference is Slow
**Problem:** 5-15 seconds per image on CPU.

- MPS (Apple Silicon) helps: ~2-3 seconds
- CUDA (NVIDIA GPU): ~0.3-0.5 seconds
- CPU: ~5-15 seconds depending on image size

**Recommendation:** Cloud deployment needs GPU instance. For mobile, consider:
- Smaller model variant
- On-device Core ML / TensorFlow Lite conversion
- Accept latency for cloud processing

---

### 5. Large Images Cause MPS Out-of-Memory
**Problem:** Large images (especially 4K+) cause GPU memory exhaustion.

- 4K images (4000x3000) can OOM on 8GB GPU
- MPS (Apple Silicon) is particularly sensitive - reported: "MPS allocated: 13.08 GiB, trying to allocate 9.85 GiB"
- Inference time scales with image area

**Solution implemented (Jan 2026):**
- Target images auto-resize to max 1024px dimension in `detector.py`
- Reference images auto-resize to max 384px dimension
- Memory cleanup with `torch.mps.empty_cache()` after each weed type processed
- Limit to 5 reference images per weed type during inference
- `gc.collect()` called between weed type iterations

```python
# Key constants in detector.py
MAX_IMAGE_DIMENSION = 1024
MAX_REFERENCE_DIMENSION = 384
```

---

## Technical Debt / Known Issues

| Issue | Impact | Fix Effort | Status |
|-------|--------|------------|--------|
| ~~No input validation on image size~~ | ~~High memory usage~~ | ~~Low~~ | ✅ Fixed (auto-resize) |
| Math import inside method | Minor perf hit | Trivial | Open |
| No rate limiting | DoS risk | Medium | Open |
| No authentication | Security risk | Medium | Open |
| Synchronous inference blocks | Poor under load | High (needs async/queue) | Open |

---

## Recommendations for Future Development

### Short Term
1. ~~Add image size validation/resizing on upload~~ ✅ Done
2. Move `import math` to module level
3. Add basic rate limiting

### Medium Term
1. Implement Redis caching for repeat detections
2. Add user authentication
3. Build mobile app MVP (API + visualization ready)

### Long Term
1. Convert model to Core ML for on-device iOS inference
2. Add real-time video mode with frame batching
3. GPS integration for lawn mapping

---

## Session Notes (Jan 20, 2026)

### What Was Built This Session
1. **Visualization endpoint** (`POST /detect/visualize`) - returns original, annotated, and comparison images as base64
2. **Image annotation** (`src/visualization/annotate.py`) - draws colored bounding boxes with labels and confidence %
3. **Web UI with tabs** - "Detect Weeds" shows side-by-side original/annotated, "Manage References" for uploading
4. **MPS memory fix** - auto-resizing images, memory cleanup between iterations, ref image limits

### Why MPS OOM Happened
- OWLv2 processes both target image AND reference images through the model
- Large target images (4K) + multiple reference images = massive memory allocation
- MPS doesn't handle memory fragmentation as well as CUDA
- Fix: aggressive resizing (1024px target, 384px refs) + memory cleanup between weed types

---

## Session Notes (Jan 20, 2026 - Part 2)

### Grounding DINO Testing Results

**Key Finding:** Grounding DINO with detailed visual prompts outperformed OWLv2 for dandelion detection.

**Best Results Achieved:**
- **Threshold:** 0.2 (DINO needs lower thresholds than OWLv2)
- **Prompt strategy:** Detailed visual descriptions work better for DINO
  - "dandelion leaves with jagged edges"
  - "rosette of pointed green leaves"
  - "full dandelion plant" (captures whole plant clusters)

**Detection Performance:**
- Yellow flowers detected at 21-27% confidence
- "full dandelion plant" prompt captured large clusters at 28%
- Successfully detected most visible weeds in test images
- Some scattered/small flowers may still be missed

### Model-Specific Prompt Strategies

| Model | Prompt Style | Example |
|-------|-------------|---------|
| **Grounding DINO** | Detailed visual descriptions | "dandelion plant with toothed leaves radiating from center" |
| **OWLv2** | Simple, concise prompts | "dandelion leaves" |

**Why the difference?**
- DINO excels at grounding detailed text to image regions
- OWLv2 with detailed prompts causes false positives (matches grass textures)

### Group Overlapping Feature

**When to use:** Multiple overlapping detections for same weed (flower + leaves + stem)

**How it works:**
1. Uses union-find style clustering
2. Merges boxes with IoU > 0.3 that share same label
3. Creates single bounding box encompassing all overlapping detections

**Best practice:** Enable grouping when you see fragmented detections of single plants

### What Didn't Help

- **Lowering threshold below 0.2:** Didn't catch missed flowers, just added noise
- **Very low thresholds (< 0.15):** Grass texture triggers false positives

### Recommended Starting Points

| Detection Mode | Threshold | Use Case |
|---------------|-----------|----------|
| Grounding DINO | 0.2-0.3 | Best for leaf detection, detailed features |
| OWLv2 Text | 0.5-0.7 | Good for flowers, seed heads |
| SAM Auto | N/A | Discovering all plant boundaries |
| OWLv2 Image | 0.3-0.5 | When you have clean reference images |

---

## Session Notes (Jan 21, 2026)

### Clover Detection Prompt Improvements

**Problem:** Generic "clover" prompts weren't specific enough to distinguish clover from other plants.

**Solution:** Updated prompts to emphasize the distinctive trifoliate (three-leaf) structure:

| Before | After |
|--------|-------|
| "three leaf clover plant" | "clover with three round leaves" |
| "clover patch with round leaves" | "three leaflets in clover pattern" |
| "white clover flower" | "clover leaf with three rounded lobes" |

**Key Insight:** Describing the physical leaf structure (trifoliate, round/rounded, leaflets) rather than just naming the plant improves DINO's ability to ground the text to the correct visual features.

**Label Normalization Extended:**
- Added `trifoliate`, `leaflet`, `three round`, `three leaf` → all map to `clover` label
- Ensures verbose DINO outputs like "three leaflets in clover pattern" display as clean `clover` label

### Prompt Design Patterns for DINO

| Weed Type | Effective Prompt Pattern |
|-----------|-------------------------|
| Dandelion | Focus on leaf shape: "jagged", "serrated", "rosette" |
| Clover | Focus on leaf structure: "trifoliate", "three round leaves", "leaflets" |
| Crabgrass | Focus on growth habit: "spreading stems", "low growing" |
| Poa Annua | Focus on height difference: "tall grass clump", "standing above lawn" |

**General Principle:** DINO prompts work best when they describe distinctive visual features rather than just naming the plant.

---

## Session Notes (Jan 21, 2026 - Part 2)

### Leaf-First Detection Strategy

**Key Learning:** Flower-based DINO prompts don't help as much as expected. The primary focus should be on **leaves** for practical weed detection.

**Why Leaves > Flowers:**
1. **Year-round detection**: Leaves persist; flowers are seasonal/ephemeral
2. **Earlier intervention**: Detect and treat weeds *before* they flower and spread seeds
3. **More consistent targets**: Leaf morphology is stable; flowers vary by growth stage
4. **Practical lawn care**: Most homeowners want to catch weeds early, not when flowering

### Changes Made

**Backup created:** `src/api/routes/detection_prompts_backup_jan21.py`

**Weed types updated to leaf-focused prompts:**

| Weed Type | Before (Flower-Based) | After (Leaf-Based) |
|-----------|----------------------|-------------------|
| Silverleaf Nightshade | "star-shaped purple flowers" | "wavy-edged silvery leaves", "dusty silver foliage" |
| Field Bindweed | "trumpet-shaped flowers", "white or pink flowers" | "arrow-shaped leaves", "arrowhead leaves" |
| Broom Snakeweed | "yellow flower clusters" | "needle-like leaves", "thread-like green stems" |
| Palmer's Amaranth | "tall seed spike", "reddish flower spike" | "large oval leaves", "diamond-shaped leaves" |

**Label normalization updated** to catch new leaf-focused terms:
- Silverleaf: Added `fuzzy leaves`, `wavy-edged`, `dusty silver`
- Bindweed: Added `arrowhead`, kept `arrow-shaped`
- Snakeweed: Added `thread-like`, `broom-like`
- Amaranth: Added `oval leaves`, `diamond-shaped`, `broad leaves`

### Rollback Instructions

If leaf-focused prompts perform worse than expected:

```bash
# Copy backup function back to detection.py
cp src/api/routes/detection_prompts_backup_jan21.py /tmp/backup_ref.py
# Then manually replace _build_text_queries() in detection.py with the backup version
```

### Design Principle for Future Weed Types

When adding new weed types, prioritize prompts in this order:
1. **Leaf shape/structure** (jagged, oval, arrow-shaped, trifoliate)
2. **Leaf texture/color** (fuzzy, silvery, waxy, glossy)
3. **Growth habit** (rosette, spreading, climbing, clumping)
4. **Stem characteristics** (thin, woody, creeping)
5. **Flowers** (only as last resort or for species where flower is the main identifier)

---

## Version 1.0 Summary

### What v1.0 Delivers

| Feature | Status | Notes |
|---------|--------|-------|
| OWLv2 text-guided detection | ✅ Complete | Best for flowers/seed heads |
| OWLv2 image-guided detection | ✅ Complete | Requires reference images |
| Grounding DINO (Original) | ✅ Complete | Best for leaf detection |
| SAM auto-segmentation | ✅ Complete | Finds all plant boundaries |
| Web UI with model selection | ✅ Complete | localhost:8000 |
| Weed type checkboxes (DINO) | ✅ Complete | Dandelion, Clover, Crabgrass, Poa Annua |
| Leaf-first prompt strategy | ✅ Complete | All DINO prompts focus on leaves |
| Label normalization | ✅ Complete | Short display codes |
| Memory management (MPS/CUDA) | ✅ Complete | Auto-resize, cleanup |
| Adaptive thresholding | ✅ Complete | Adjusts to conditions |

### v1.0 Performance Baseline

| Configuration | Speed | Hardware |
|--------------|-------|----------|
| Grounding DINO | ~1-2 sec | Apple M1/M2 (MPS) |
| Grounding DINO | ~0.3-0.5 sec | NVIDIA GPU (CUDA) |
| Grounding DINO | ~8 FPS | A100 |
| OWLv2 | ~2-3 sec | Apple M1/M2 (MPS) |
| SAM Auto | ~3-5 sec | Apple M1/M2 (MPS) |

### Key Learnings from v1.0 Development

1. **Leaf-first detection is critical** - Flowers are ephemeral; leaves persist year-round
2. **DINO needs detailed prompts** - "rosette of serrated leaves" beats "dandelion"
3. **OWLv2 needs simple prompts** - Detailed prompts cause false positives
4. **Lower thresholds for DINO** - 0.2-0.3 vs 0.5-0.7 for OWLv2
5. **Label normalization is essential** - DINO returns verbose labels that need mapping
6. **Order matters in normalization** - Crabgrass before Poa to avoid cross-contamination

---

## Version 2.0 Planning

### Why v2.0?

v1.0 delivers functional weed detection, but:
- **Speed is limited** - Original DINO runs at ~8 FPS max
- **No TensorRT support** - Missing 2-4x speedup on NVIDIA GPUs
- **Single DINO variant** - No access to faster/more accurate versions

### v2.0 Goals

| Goal | Target | Current (v1.0) |
|------|--------|----------------|
| Real-time detection | 30+ FPS | ~8 FPS |
| Edge deployment | 15+ FPS on Jetson | ~3 FPS |
| Model options | 4 DINO variants | 1 variant |
| Hardware optimization | Auto TensorRT | None |

### v2.0 Model Research

**Grounding DINO 1.5 Pro:**
- Highest accuracy (54.3 AP vs 52.5 AP)
- Slightly slower than original
- API-based (cloud dependency)
- Best for: Maximum accuracy when speed isn't critical

**Grounding DINO 1.5 Edge:**
- Optimized for speed (75 FPS with TensorRT)
- Lower accuracy (36.2 AP)
- Open weights available
- Best for: Real-time video, edge devices

**Dynamic-DINO (2025):**
- Mixture of Experts architecture
- Outperforms 1.5 Edge with less training data
- Open-source
- Best for: Balance of speed and accuracy

### v2.0 Technical Requirements

**TensorRT Integration:**
```python
# Detection flow with TensorRT
def detect_with_tensorrt(image, model_path):
    # 1. Check if TensorRT engine exists
    engine_path = f"{TENSORRT_CACHE_DIR}/{model_name}.engine"

    if not os.path.exists(engine_path):
        # 2. Convert model (one-time)
        convert_to_tensorrt(model_path, engine_path)

    # 3. Load engine and run inference
    engine = load_tensorrt_engine(engine_path)
    return engine.infer(image)
```

**Hardware Detection:**
```python
def get_acceleration_options():
    options = {"pytorch": True}  # Always available

    if torch.cuda.is_available():
        options["cuda"] = True

        # Check for TensorRT
        try:
            import tensorrt
            options["tensorrt"] = True
            options["tensorrt_version"] = tensorrt.__version__
        except ImportError:
            options["tensorrt"] = False

    return options
```

### v2.0 Implementation Checklist (Complete)

- [x] Add DINO 1.5 Edge detector class (`grounding_dino_1_5_edge.py`)
- [x] Add DINO 1.5 Pro detector class (`grounding_dino_1_5_pro.py`)
- [x] Add Dynamic-DINO detector class (`dynamic_dino.py`)
- [x] Add Local Weights loader (`grounding_dino_local.py`)
- [x] Implement TensorRT toggle in UI (framework ready)
- [x] Add hardware detection utility (`src/config.py:can_use_tensorrt()`)
- [x] Update UI with variant selection (dropdown with optgroups)
- [x] Update UI with TensorRT toggle (checkbox for supported modes)
- [x] Update API parameters (`detection_mode`, `use_tensorrt`)
- [x] Update documentation (HANDOFF.md, WORKFLOW.md, LESSONS_LEARNED.md)
- [ ] Performance benchmarking (manual testing completed)
- [ ] TensorRT engine caching (framework ready, needs NVIDIA GPU testing)

---

## Session Notes (Jan 21, 2026 - v2.0 Implementation)

### v2.0 Features Implemented

**1. New DINO Detector Variants:**
- `grounding_dino_1_5_pro.py` - Uses grounding-dino-base, 1024px images
- `grounding_dino_1_5_edge.py` - Uses grounding-dino-tiny, 640px images (faster)
- `dynamic_dino.py` - Uses grounding-dino-tiny, 800px images (balanced)

**2. Local Weights Support:**
- `grounding_dino_local.py` - Loads original .pth weights from `weights/` directory
- Supports both Swin-T (faster) and Swin-B (accurate) backbones
- Auto-detects which backbone based on filename
- **`groundingdino-py` package now installed** - loads weights directly without fallback
- Detailed timing logs added (`[DINO Local]` prefix) to monitor loading progress

**3. UI Enhancements:**
- Detection Mode dropdown organized into groups:
  - "DINO Models" - HuggingFace-based variants
  - "Local Weights 📦" - Local .pth file variants
- TensorRT checkbox appears for 1.5 Edge, 1.5 Pro, and Dynamic-DINO
- Descriptions update based on selected mode

### Key Learnings from v2.0

**1. HuggingFace Model Availability:**
- Only `grounding-dino-tiny` and `grounding-dino-base` are publicly available
- DINO 1.5 Pro/Edge require API token from DeepDataSpace
- Dynamic-DINO weights not yet on HuggingFace (research model)

**2. Fallback Strategy Works Well:**
- DINO 1.5 variants fall back to available HuggingFace models
- Different image sizes provide speed/accuracy tradeoff
- Edge (640px) → faster, Pro (1024px) → more accurate

**3. Local Weights Provide Original Models:**
- `groundingdino_swint_ogc.pth` - Swin-T backbone (~662MB)
- `groundingdino_swinb_cogcoor.pth` - Swin-B backbone (~895MB)
- Download from GitHub releases (IDEA-Research/GroundingDINO)

**4. Lazy Loading is Essential:**
- All DINO variants are lazy-loaded in `detection.py`
- Prevents loading multiple models at startup
- Each model loads only when first used

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| "Model not found on HuggingFace" | Detector classes fall back to available models |
| "Weights not found" error | Download .pth files to `weights/` directory |
| UI not updating after code changes | Kill uvicorn, clear `__pycache__`, restart without `--reload` |
| TensorRT checkbox not showing | Only appears for Edge, Pro, Dynamic modes |
| Local weights loading slowly | Install `groundingdino-py` package (now installed) |
| Can't see loading progress | Timing logs print to terminal with `[DINO Local]` prefix |
| Config file not found (local weights) | Fixed: configs now resolve from `groundingdino` package install location |
| Server hangs on first Swin-B load | Normal behavior - Swin-B (~895MB) takes 30-60+ seconds on first load |
| Browser shows "analyzing" indefinitely | Model loading blocks server; wait or kill (`kill -9 PID`) and restart |
| Server not responding to health check | Server likely blocked on model load; check terminal for `[DINO Local]` logs |

### v2.0 Architecture Decisions

**Why separate detector classes?**
- Each variant has different image size requirements
- TensorRT support varies by model
- Cleaner separation of concerns
- Easier to add new variants in future

**Why local weights loader?**
- Original weights provide exact pretrained performance
- No dependency on HuggingFace model availability
- Supports official Swin-T and Swin-B backbones
- `groundingdino-py` package installed - uses native loading (faster, no download)

**Why dropdown groups in UI?**
- Clear separation between HuggingFace and local models
- Users understand which options require weights download
- Easier to add new variants without cluttering dropdown

### v2.0 Backwards Compatibility

- All v1.0 API parameters still work
- Default detection mode unchanged (`text_owlv2`)
- New `detection_mode` values are additive
- `use_tensorrt` parameter is optional (defaults to false)

---

## Session Notes (Jan 21, 2026 - Local Weights Config Fix)

### Problem: Config File Not Found Error

When selecting "DINO Swin-B (Local)", users encountered:
```
Error: file "/Users/mark/.../ai preception/groundingdino/config/GroundingDINO_SwinB.py" does not exist
```

### Root Cause Analysis

Two issues combined:
1. **Wrong config filename**: The `groundingdino-py` package uses `GroundingDINO_SwinB_cfg.py` (with `_cfg` suffix), not `GroundingDINO_SwinB.py`
2. **Relative path resolution**: The original code used relative paths like `"groundingdino/config/..."` which resolved from the project directory, not the installed package location

### Solution Implemented

Changed `grounding_dino_local.py` to resolve config paths from the installed package:

```python
# Before (broken):
if "swinb" in str(self.weights_path).lower():
    config_file = "groundingdino/config/GroundingDINO_SwinB.py"

# After (fixed):
import groundingdino
package_dir = Path(groundingdino.__file__).parent
if "swinb" in str(self.weights_path).lower():
    config_file = str(package_dir / "config" / "GroundingDINO_SwinB_cfg.py")
else:
    config_file = str(package_dir / "config" / "GroundingDINO_SwinT_OGC.py")
```

### Key Learning: Package Config Locations

When using installed Python packages, config files must be resolved from the package installation location, not relative paths from the working directory.

**Pattern to find package location:**
```python
import some_package
package_dir = Path(some_package.__file__).parent
config_path = package_dir / "config" / "some_config.py"
```

### Server Deadlock Issue

After the fix, another issue emerged: the server appeared to hang indefinitely when loading the Swin-B model.

**Symptoms:**
- Browser shows "analyzing" forever
- `curl http://localhost:8000/health` times out
- Server process is running but not responding

**Cause:** First-time model loading is synchronous and blocks the entire FastAPI server. The Swin-B model (~895MB) can take 30-60+ seconds to load, during which no requests are processed.

**Solution:** Wait for model to load, or kill the server process (`kill -9 PID`) and restart. Subsequent requests are fast once the model is cached.

**Future improvement:** Consider async model loading or a loading endpoint that returns progress.

---

## Session Notes (Jan 27, 2026 - RF-DETR Integration)

### Why RF-DETR?

Added RF-DETR as a new detection model option to address limitations of zero-shot models:

| Issue with Zero-Shot | RF-DETR Solution |
|---------------------|------------------|
| Leaf detection is unreliable | Fine-tuned on actual weed images including leaves |
| Text prompts require tuning | No prompts - model learns directly from examples |
| Generic models not optimized for weeds | Specialized training on weed dataset |

### Key Architectural Difference

**Zero-shot models (DINO, OWLv2):**
- Accept text queries at inference time
- Can detect any object you describe
- Lower accuracy for specific domains

**RF-DETR (closed-vocabulary):**
- Classes fixed at training time
- Returns class IDs that map to trained classes
- Higher accuracy for known classes
- Cannot detect new classes without retraining

### Integration Pattern

RF-DETR follows the existing detector pattern but with key differences:

```python
# Zero-shot: text queries at inference
result = dino_detector.detect(image, text_queries=["dandelion plant"])

# Closed-vocabulary: no text queries, filter by class after
result = rf_detr_detector.detect(image, weed_types=["dandelion", "clover"])
```

The `weed_types` parameter filters results post-inference (the model always runs on all trained classes).

### Files Added/Modified

| File | Change |
|------|--------|
| `src/detection/rf_detr.py` | New detector class |
| `src/config.py` | Added `RF_DETR` detection mode |
| `src/api/routes/detection.py` | Lazy-loader and route handling |
| `src/api/routes/ui.py` | "Fine-Tuned Models" dropdown group |
| `notebooks/rf_detr_weed_training.ipynb` | Colab training notebook |

### Training Requirements

- **Dataset**: COCO-format weed dataset (Roboflow Universe has several)
- **Hardware**: GPU required (T4 minimum, A100 recommended)
- **Time**: ~2-3 hours for 50 epochs on 1000 images with T4
- **Output**: `rf_detr_weed_weights.pt` → copy to `weights/` folder

### When to Use RF-DETR vs Zero-Shot

| Use RF-DETR when... | Use DINO/OWLv2 when... |
|--------------------|------------------------|
| You have training data | No training data available |
| Detecting known weed types | Exploring new weed types |
| Need highest accuracy | Need flexibility |
| Production deployment | Rapid prototyping |

### Next Steps

1. Train RF-DETR on weed dataset using Colab notebook
2. Copy weights to `weights/rf_detr_weed_weights.pt`
3. Update `CLASS_NAMES` in `rf_detr.py` to match dataset
4. Compare accuracy vs DINO on test images
