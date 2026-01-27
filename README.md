# Weed Detection API

AI-powered weed detection using OWLv2 few-shot learning. Upload reference images of weeds, then detect them in lawn photos - no training required.

## Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env

# Run
python main.py
```

API runs at http://localhost:8000 (docs at /docs).

## Usage

1. **Upload reference images** (5-10 per weed type):
```bash
curl -X POST "http://localhost:8000/references/upload" \
  -F "image=@dandelion.jpg" \
  -F "weed_type=dandelion"
```

2. **Detect weeds**:
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "image=@lawn_photo.jpg"
```

## Configuration

Edit `.env` for GPU support:
```
DEVICE=mps   # Apple Silicon
DEVICE=cuda  # NVIDIA GPU
```

See `docs/` for detailed documentation.
