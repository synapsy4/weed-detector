# Weed Detector

YOLO-based crop-vs-weed detection service for training, evaluating, and serving a plant-detection model.

## Overview

This project trains a YOLOv8-based model to detect weeds and crops in plant imagery and exposes the trained model through a FastAPI service. It is designed for local experimentation and reproducible training runs, with dataset versioning and experiment tracking backed by DVC and MLflow through DagsHub.

The repository combines:

- a data processing pipeline for downloading and structuring the raw dataset
- YOLO training and evaluation workflows
- a prediction API for inference on uploaded images
- DVC-based versioning for data and model artifacts
- MLflow-based tracking for experiment metadata, parameters, and metrics
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
uv run dvc push
```

The `scripts/run.py` CLI is the project entry point for the pipeline stages. DVC calls these same commands to keep the data lifecycle reproducible and consistent.

## CLI usage

The canonical CLI interface is subcommand based:

```bash
uv run python scripts/run.py download
uv run python scripts/run.py preprocess
uv run python scripts/run.py train
uv run python scripts/run.py eval
uv run python scripts/run.py predict --source path/to/image.jpg
```

Common options:

- `--config`: config filename, default `config.yaml`
- `--config-root`: config directory, default `configs`
- `--name`: override the experiment or model name
- `--source`: file or folder path used by `predict`

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
```

## Prediction API

The project includes a FastAPI service for inference.

Start the API locally:

```bash
uv run uvicorn weed_detector.api.main:app --host 0.0.0.0 --port 8000
```

Endpoints include:

- `GET /health`
- `POST /predict`
- `GET /metrics`
- `GET /docs`

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/image.jpg" \
  -F "conf_threshold=0.25"
```

## Docker and local deployment

The project includes Docker and nginx configuration for a local deployment stack:

```bash
docker compose up --build
```

This starts the API and the reverse proxy. The default local routes are:

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

- Training outputs and large assets are intentionally kept out of Git and handled through DVC.
- MLflow logs training parameters, metrics, and artifacts to the configured DagsHub experiment.
- The repo is structured so that the training script stays focused on training orchestration, while DVC and MLflow integration is handled in dedicated utility modules.


