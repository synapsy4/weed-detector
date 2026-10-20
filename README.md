# Weed Detector

YOLO-based crop vs. weed detection and classification service.

## Overview

The **Weed Detector** is a computer vision service that detects and classifies plants in images as either **crop** or **weed**. It uses a fine-tuned YOLOv8n model trained on the [Kaggle Crop and Weed Detection Dataset](https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes).

The project provides:

- **Data pipeline**: download, preprocess, and split raw images/labels into YOLO-format train/val/test sets
- **Training**: fine-tune YOLOv8n on the crop/weed dataset with resume support
- **Evaluation**: run validation on the test split with metrics and visualizations
- **Prediction service**: a FastAPI HTTP API that accepts image uploads and returns bounding boxes with class labels and confidence scores
- **Observability**: structured logging throughout, Prometheus metrics exposed at `/metrics`

## Project Structure

```
weed-detector/
├── README.md                    # This file
├── docs/                          # Development plan and architecture
├── pyproject.toml               # Project metadata + dependencies (uv)
├── uv.lock                      # Locked dependency tree
├── .python-version              # Python 3.12
├── .gitignore                   # Git ignore rules
├── configs/
│   ├── config.yaml              # Hyperparameters, paths, train settings
│   └── yolo.yaml                # YOLO data configuration (auto-generated)
├── data/
│   ├── raw/                     # Downloaded Kaggle dataset (gitignored)
│   └── processed/               # Train/val/test splits in YOLO format (gitignored)
├── models/
│   ├── pretrained/              # yolov8n.pt (gitignored)
│   └── trained/                 # Fine-tuned model weights (gitignored)
├── notebooks/                   # Jupyter notebooks for exploration
├── scripts/
│   └── run.py                   # CLI entry point (train/eval/predict)
└── src/
    └── weed_detector/           # Python package
        ├── __init__.py
        ├── config.py            # YAML config loader
        ├── api/
        │   ├── __init__.py
        │   ├── main.py          # FastAPI app factory + lifespan
        │   ├── schemas.py       # Pydantic request/response models
        │   ├── service.py       # PredictionService (model + validation)
        │   └── routes/
        │       ├── __init__.py
        │       └── predict.py   # POST /predict, GET /health, GET /metrics
        ├── data/
        │   ├── __init__.py
        │   ├── download.py      # Kaggle dataset download
        │   └── preprocess.py    # Train/val/test splitting
        ├── inference/
        │   ├── __init__.py
        │   └── predict.py       # Single-image prediction wrapper
        ├── model/
        │   ├── __init__.py
        │   └── detector.py      # YOLO model loading
        ├── training/
        │   ├── __init__.py
        │   ├── train.py         # Training entry point
        │   └── evaluate.py      # Evaluation entry point
        ├── utils/
        │   ├── __init__.py
        │   ├── logging.py       # Structured logging setup
        │   ├── labels.py        # Class index <-> name mapping
        │   ├── paths.py         # Path resolution helpers
        │   └── yolo.py          # YOLO YAML writer, train kwargs, device selection
        └── visualization/
            ├── __init__.py
            └── plots.py         # Bounding-box visualization
```


## Prerequisites

- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- CPU is sufficient for inference; GPU optional (CUDA detected automatically if available)

## Setup

```bash
# Clone the repository
git clone https://github.com/synapsy4/weed-detector.git
cd weed-detector

# Install dependencies (creates .venv)
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

## Code Quality & Local CI

All checks (lint, format, type check, tests with coverage) run
locally with a **single command**:

```bash
uv run pre-commit run --all-files
```

This mirrors the GitHub Actions pipeline exactly (see `.github/workflows/ci.yml`).

To run the checks automatically on every `git commit`, install the Git hook once:

```bash
uv run pre-commit install
```

## Usage

### 1. Command-Line Interface (CLI)

The CLI supports three modes: `train`, `eval`, and `predict`.

```bash
python scripts/run.py --mode train --config config.yaml --config-root configs
python scripts/run.py --mode eval   --config config.yaml --config-root configs
python scripts/run.py --mode predict --config config.yaml --config-root configs --source path/to/image.jpg
```

| Flag | Description |
|------|-------------|
| `--mode` | `train`, `eval`, or `predict` (required) |
| `--config` | Config filename (default: `config.yaml`) |
| `--config-root` | Config directory (default: `configs`) |
| `--name` | Override experiment/model name |
| `--source` | Image path for `predict` mode (required) |

### 2. Prediction API Service

Start the FastAPI server:

```bash
uv run uvicorn weed_detector.api.main:app --host 0.0.0.0 --port 8000
```

The service loads the best trained weights (`models/trained/modelv2/train/weights/best.pt`) on startup. If no trained weights exist, it falls back to the pretrained `yolov8n.pt`.

#### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /health` | Health check | Returns `{"status": "ok", "model_loaded": true}` |
| `POST /predict` | Predict | Upload an image, get bounding boxes with classes and confidences |
| `GET /metrics` | Metrics | Prometheus-formatted metrics |
| `GET /docs` | API docs | Interactive Swagger UI |

#### Example: Call the prediction endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/image.jpg" \
  -F "conf_threshold=0.25"
```

#### Example response

```json
{
  "predictions": [
    {
      "bbox": {"x1": 122.1, "y1": 64.9, "x2": 512.0, "y2": 402.8},
      "class_id": 1,
      "class_name": "weed",
      "confidence": 0.535
    },
    {
      "bbox": {"x1": 136.4, "y1": 62.4, "x2": 374.2, "y2": 405.4},
      "class_id": 1,
      "class_name": "weed",
      "confidence": 0.254
    }
  ],
  "image_size": [512, 512],
  "model_name": "modelv2"
}
```

#### Example: Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true}
```
## Configuration

Edit `configs/config.yaml` to change hyperparameters:

```yaml
seed: 99
data:
  test_split: 0.15
  val_split: 0.15
model:
  model_name: modelv2
  pretrained_weights: yolov8n.pt
  imgsz: 512
train:
  epochs: 2
  batch_size: 16
  resume_blocking: false
paths:
  raw_dir: data/raw
  raw_data_dir: data/raw/agri_data/data
  processed_data_dir: data/processed
  yolo_yaml_path: configs/yolo.yaml
  models_pretrained_dir: models/pretrained
  models_trained_dir: models/trained
```

## Logging

All modules use structured logging via `weed_detector.utils.logging`. Log level defaults to `INFO`.

Example log output:

```
2026-10-20 14:42:42 | INFO     | weed_detector.api.main:lifespan:23 | Starting weed detector API service
2026-10-20 14:42:42 | INFO     | weed_detector.api.service:_load_model:51 | Loading model weights from '.../best.pt'
2026-10-20 14:42:42 | INFO     | weed_detector.api.service:_load_model:53 | Model loaded successfully
```

## Dataset

The model is trained on the [Crop and Weed Detection Dataset](https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes) from Kaggle, which contains JPEG images and YOLO-format `.txt` label files. Classes:

| Class ID | Name |
|----------|------|
| 0 | crop |
| 1 | weed |

Images are 512×512 pixels.

## Current State

- ✅ Phase 0: Codebase finetuning (logging, `__init__.py` exports, bug fixes) — **completed**
- ✅ Phase 1: FastAPI prediction service — **completed** (`POST /predict`, `GET /health`, `GET /metrics`)
- ⬜ Phase 2: Docker containerization
- ⬜ Phase 3: nginx security layer
- ⬜ Phase 4: Experiment tracking (MLflow) + data versioning (DVC on DagsHub)
- ⬜ Phase 5: Monitoring (Prometheus + Grafana)
- ⬜ Phase 6: Vanilla HTML/JS UI
- ⬜ Phase 7: Full-stack integration & deployment prep


