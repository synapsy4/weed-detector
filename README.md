# Weed Detector

YOLO-based crop-vs-weed detection service for training, evaluating, and serving a plant-detection model.

## Overview

This project trains a YOLOv8-based model to detect weeds and crops in plant imagery and exposes the trained model through a FastAPI service. It is designed for local experimentation and reproducible training runs, with dataset versioning and experiment tracking backed by DVC and MLflow through DagsHub.

The repository combines:

- a data processing pipeline for downloading and structuring the raw dataset
- YOLO training and evaluation workflows
- a prediction API that serves the production model from the MLflow model registry
- DVC-based versioning for data and model artifacts
- MLflow-based tracking for experiment metadata, parameters, and metrics
- an MLflow model registry with gated promotion (`production` / `fallback` aliases)
- Docker-based runtime setup for local deployment

## Project structure

```text
weed-detector/
├── README.md
├── pyproject.toml
├── uv.lock
├── docker-compose.yml
├── .env.example
├── .dvc/
├── dvc.yaml
├── configs/
│   ├── config.yaml
│   └── yolo.yaml
├── data/
│   ├── raw/
│   └── processed/
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.nginx
│   └── nginx/
├── docs/
│   ├── API.md
│   └── DEVELOPMENT_PLAN.md
├── models/
│   ├── pretrained/
│   └── trained/
├── notebooks/
├── scripts/
│   └── run.py
├── src/
│   └── weed_detector/
│       ├── api/
│       ├── config.py
│       ├── data/
│       ├── inference/
│       ├── model/
│       ├── training/
│       ├── utils/
│       └── visualization/
├── tests/
└── .gitignore
```

## Requirements

- Python 3.12
- [uv](https://github.com/astral-sh/uv)
- Git
- A DagsHub account and repository for the MLflow/DVC integration
- Kaggle access for downloading the dataset used by the training pipeline

## Local setup

1. Clone the repository

```bash
git clone https://github.com/synapsy4/weed-detector.git
cd weed-detector
```

2. Create the project environment and install dependencies

```bash
uv sync
source .venv/bin/activate
```

3. Create a local environment file from the template

```bash
cp .env.example .env
```

4. Fill in the values in `.env` with your project-specific settings:

```text
DAGSHUB_REPO_URL=https://dagshub.com/<owner>/<repo>
DAGSHUB_TOKEN=<your_dagshub_token>
MLFLOW_TRACKING_URI=https://dagshub.com/<owner>/<repo>.mlflow
MLFLOW_TRACKING_TOKEN=<your_dagshub_token>
MLFLOW_EXPERIMENT_NAME=weed-detector
DVC_REMOTE_URL=https://dagshub.com/<owner>/<repo>.dvc
```

The values above are required for the DVC remote and the MLflow tracking backend used by the project.

## DVC and MLflow setup

This repository is configured to store data and model artifacts in DVC and to send training metadata to MLflow on DagsHub.

### One-time DVC remote configuration

```bash
uv run dvc remote default origin
```

If the remote is not yet registered, configure it once with:

```bash
uv run dvc remote add origin "$DVC_REMOTE_URL"
```

### Training workflow with tracking enabled

```bash
uv run dvc status
uv run python scripts/run.py download
uv run python scripts/run.py preprocess
uv run python scripts/run.py train
uv run python scripts/run.py eval
uv run python scripts/run.py promote
uv run dvc push
```

The `scripts/run.py` CLI is the project entry point for the pipeline stages. DVC calls these same commands to keep the data lifecycle reproducible and consistent (`dvc repro` runs download, preprocess, train, evaluate and promote).

### Continuing training

If `model.model_name` already has trained weights and `train.resume_blocking` is `false`, training continues from the model's `last.pt`. Ultralytics strips the optimizer and epoch state from finished runs, so this is a **warm start**: the weights are reused, but the optimizer and the epoch counter start fresh and `train.epochs` additional epochs are run. The previous `results.csv` is archived as `results_<timestamp>.csv`, and the new MLflow run is tagged `continued_from=<previous run id>`.

### Model registry and promotion

After `train` and `eval`, `promote` decides whether the trained model should serve predictions:

1. The model's **weed recall** on the test split must reach `registry.min_threshold`.
2. The current `production` model is re-evaluated on the *current* test set, and the candidate must beat it by at least `registry.min_improvement`.
3. If both checks pass, the candidate is registered in the MLflow model registry and gets the alias `production`; the previous production version gets the alias `fallback`. The first model above the threshold becomes `production` directly.

Rejected candidates are not registered; the decision is recorded as tags on their MLflow run. To undo a promotion, swap the two aliases:

```bash
uv run python scripts/run.py rollback
```

Registered versions carry tags such as `imgsz`, the test metrics, the DVC data hash (`data_version`) and the git commit.

### Local files vs. registry

| Where | What | Role |
|-------|------|------|
| `models/pretrained/` | Base weights (e.g. `yolov8n.pt`) | Starting point for new training runs |
| `models/trained/<name>/` | `last.pt`, `best.pt`, plots, `metrics.json`, `mlflow_run_id.txt` | Local training workspace and DVC pipeline outputs; `last.pt` is needed for warm starts |
| MLflow model registry | Promoted model versions (`production`, `fallback`) | **Source of truth for serving**; the API never reads `models/` |

## CLI usage

The canonical CLI interface is subcommand based:

```bash
uv run python scripts/run.py download
uv run python scripts/run.py preprocess
uv run python scripts/run.py train
uv run python scripts/run.py eval
uv run python scripts/run.py promote
uv run python scripts/run.py rollback
uv run python scripts/run.py predict --source path/to/image.jpg
```

`predict` is a local development helper and uses the local `best.pt` of the configured model, not the registry.

Common options:

- `--config`: config filename, default `config.yaml`
- `--config-root`: config directory, default `configs`
- `--name`: override the experiment or model name
- `--source`: file or folder path used by `predict`
- `--set KEY=VALUE`: override a config value for this run, e.g. `--set train.epochs=50 --set model.model_name=modelv3` (repeatable; the key must exist in the config)

## Configuration

The main project config is in `configs/config.yaml`. It contains hyperparameters, dataset settings, model names, and path definitions.

```yaml
seed: 99
data:
  test_split: 0.15
  val_split: 0.15
model:
  model_name: modelv0
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
registry:
  model_name: weed-detector   # name in the MLflow model registry
  metric: weed_recall         # promotion metric (test split)
  min_threshold: 0.3          # absolute minimum
  min_improvement: 0.01       # required gain over the production model
```

## Prediction API

The project includes a FastAPI service for inference. It loads its model from the MLflow model registry: the version with the alias `production`, or, if that cannot be loaded, the one with the alias `fallback`. Downloaded versions are cached on disk, so a restart while the registry is unreachable still serves the last-known-good model. There is deliberately **no pretrained fallback**: without a registered model the API starts anyway, but `/ready` and `/predict` answer `503` until a model can be loaded. It keeps retrying in the background and picks up newly promoted versions automatically.

Start the API locally (registry access needs `DAGSHUB_USER_TOKEN` or `DAGSHUB_TOKEN` and `MLFLOW_TRACKING_URI` in `.env`):

```bash
uv run uvicorn weed_detector.api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`: liveness, always 200; reports whether a model is loaded and which version
- `GET /ready`: readiness, 200 only if a model is loaded, otherwise 503 with the reason
- `POST /predict`: predictions; the response includes `model_name` and `model_version`
- `GET /metrics`: Prometheus metrics, including `weed_detector_model_loaded` and `weed_detector_model_info`
- `GET /docs`: interactive API docs
- `POST /admin/reload`: re-check the registry immediately; requires the `X-Admin-Token` header and is disabled unless `ADMIN_TOKEN` is set (not exposed through nginx)

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/image.jpg" \
  -F "conf_threshold=0.25"
```

Runtime settings (environment variables):

| Variable | Default | Purpose |
|----------|---------|---------|
| `DAGSHUB_USER_TOKEN` / `DAGSHUB_TOKEN` | - | Token for reading the registry |
| `MODEL_CACHE_DIR` | `~/.cache/weed-detector/models` (`/var/cache/weed-detector/models` in Docker) | Cache of downloaded model versions |
| `MODEL_REFRESH_SECONDS` | `300` | Registry check interval while `production` is served |
| `MODEL_RETRY_SECONDS` | `30` | Check interval while no model or only `fallback` is served |
| `ADMIN_TOKEN` | empty (disabled) | Enables `/admin/reload` |

## Docker and local deployment

The project includes Docker and nginx configuration for a local deployment stack:

```bash
docker compose up --build
```

This starts the API and the reverse proxy. The API container reads its registry credentials from `.env` at runtime (never baked into the image) and keeps downloaded models in the `model-cache` volume. Its healthcheck uses `/ready`. The default local routes are:

- `http://localhost:8080/health`
- `https://localhost:8443/health`

## Dataset

The training dataset is the Kaggle crop-and-weed detection dataset, with annotations in YOLO text format. The class mapping is:

- `0`: crop
- `1`: weed

## Code quality

Run the project linting and test checks with:

```bash
uv run pre-commit run --all-files
```

If you want the hooks to run automatically on each commit:

```bash
uv run pre-commit install
```

## Notes

- Training outputs and large assets are intentionally kept out of Git and handled through DVC; promoted models additionally live in the MLflow model registry.
- MLflow logs training parameters, metrics (including the test-split metrics from `eval`), and artifacts to the configured DagsHub experiment.
- The repo is structured so that the training script stays focused on training orchestration, while DVC and MLflow integration is handled in dedicated utility modules.


