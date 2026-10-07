# DeepFashion AI

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/Transformers-4.30%2B-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/docs/transformers)
[![Ollama](https://img.shields.io/badge/Ollama-optional-000000?logo=ollama&logoColor=white)](https://ollama.com/)

Intelligent clothing cataloguing from images using Fashion-CLIP, general-purpose CLIP models, an optional supervised CNN baseline, and a local LLaVA deployment through Ollama.

![Application interface](./ui.png)

## Problem and solution

Manual fashion cataloguing is slow and produces inconsistent category, attribute, and colour metadata. DeepFashion AI exposes a web interface and a JSON API that receive a clothing image and return a structured catalogue entry:

- the most likely clothing category and its top-k alternatives;
- up to ten detected attributes;
- the five highest-scoring colours;
- model, device, TTA, threshold, and timestamp metadata.

The default pipeline is zero-shot: it uses a pretrained Fashion-CLIP model and compares image embeddings with prompt ensembles built from the DeepFashion taxonomy. The project also includes a ResNet50/EfficientNet category classifier trained on DeepFashion and an optional local LLaVA integration.

## Features and technical architecture

### Main features

- Drag-and-drop web UI with image preview, confidence display, top-k categories, attributes, colours, and raw JSON.
- REST-style multipart endpoint at `POST /predict` and equivalent `POST /api/predict`.
- Runtime model selection without changing application code.
- Test-Time Augmentation (TTA) for CLIP inference: original crop, horizontal flip, and colour jitter; it can be disabled per request.
- Lazy model loading and automatic CPU/CUDA selection.
- Structured prediction files saved next to uploaded images under `uploads/`.
- DeepFashion acquisition, image validation, duplicate detection, and train/validation/test splitting utilities.
- Optional supervised benchmark with ResNet50 or EfficientNet, including Top-1, Top-5, per-class accuracy, reports, and plots.

### Architecture

```text
Browser / API client
          |
          v
Flask app (app.py)
          |
          +--> FashionPredictor
          |      CLIP/Fashion-CLIP + prompt banks + optional TTA
          |
          +--> CNNPredictor
          |      ResNet50/EfficientNet checkpoint
          |
          +--> Ollama predictor
                 Local LLaVA HTTP API
          |
          v
Structured JSON response + uploads/<timestamp>_<filename>.json
```

| Component | Responsibility |
| --- | --- |
| [`app.py`](./app.py) | Flask application, upload validation, model routing, JSON API, and saved results |
| [`src/predictor.py`](./src/predictor.py) | CLIP/Fashion-CLIP inference, prompt ensembles, DeepFashion taxonomy, TTA, and JSON schema |
| [`src/cnn_predictor.py`](./src/cnn_predictor.py) | Inference from a trained ResNet50/EfficientNet checkpoint; category only |
| [`src/cnn_trainer.py`](./src/cnn_trainer.py) | Transfer-learning model construction and training |
| [`src/cnn_dataset.py`](./src/cnn_dataset.py) | DeepFashion category dataset and image transforms |
| [`src/ollama_predictor.py`](./src/ollama_predictor.py) | Calls the local `deepfashion-llava` Ollama model and normalises its response |
| [`src/dataset_manager.py`](./src/dataset_manager.py) | Image validation, simple duplicate detection, indexing, and dataset splits |
| [`scripts/setup_dataset.py`](./scripts/setup_dataset.py) | Downloads the configured DeepFashion files and extracts the image archive |
| [`scripts/train_cnn.py`](./scripts/train_cnn.py) | CNN training entry point |
| [`scripts/eval_cnn.py`](./scripts/eval_cnn.py) | Test-set evaluation and benchmark artefact generation |
| [`templates/`](./templates/), [`static/`](./static/) | Web page and frontend assets |

### Supported inference models

| Identifier | Behaviour |
| --- | --- |
| `patrickjohncyh/fashion-clip` | Default Fashion-CLIP model, recommended for clothing |
| `openai/clip-vit-base-patch32` | OpenAI CLIP ViT-B/32 |
| `openai/clip-vit-base-patch16` | OpenAI CLIP ViT-B/16 |
| `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | LAION CLIP ViT-B/32 |
| `openai/clip-vit-large-patch14` | OpenAI CLIP ViT-L/14; higher resource usage |
| `cnn/resnet50` | Trained ResNet50; category prediction only |
| `ollama/llava` | Local `deepfashion-llava` model through Ollama |

## Installation and usage

### Requirements

- Python 3.10 or newer;
- enough disk space for the DeepFashion images and Hugging Face model cache;
- an NVIDIA GPU is optional; PyTorch automatically uses CUDA when available;
- Ollama is optional and only required for `ollama/llava`.

The commands below use a Windows PowerShell workflow. Run them from the repository root.

### 1. Create the environment and install dependencies

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Prepare DeepFashion

The CLIP pipeline requires the category and attribute metadata under `deepfashion/Anno_coarse/`. The included downloader fetches the configured DeepFashion files, extracts the image archive, and creates the colour taxonomy used by the predictor:

```powershell
python scripts\setup_dataset.py
```

The expected minimum layout is:

```text
deepfashion/
├── Anno_coarse/
│   ├── list_attr_cloth.txt
│   ├── list_attr_colors.txt
│   ├── list_attr_img.txt
│   ├── list_category_cloth.txt
│   └── list_category_img.txt
├── Eval/
│   └── list_eval_partition.txt
└── img/
```

To validate the downloaded images and create deterministic dataset split files:

```powershell
python src\dataset_manager.py
```

### 3. Start the web application

```powershell
python app.py
```

Open <http://localhost:5000>. The first CLIP request downloads the selected model from Hugging Face and builds the text banks, so the first prediction can take longer than subsequent requests. Uploaded images and their JSON results are written to `uploads\`.

Accepted image formats are PNG, JPG, JPEG, GIF, and WEBP, with a maximum request size of 16 MB.

### 4. Use the API

PowerShell example:

```powershell
curl.exe -X POST http://localhost:5000/api/predict `
  -F "image=@notebooks\testing\model.jpg" `
  -F "model=patrickjohncyh/fashion-clip"
```

Optional form fields:

- `model`: one of the identifiers in the supported-models table;
- `no_tta=true`: disables TTA for CLIP-based inference.

The response contains `category`, `attributes`, `colors`, `metadata`, `upload_url`, and `model_used`. The CNN response keeps the same schema but leaves `attributes` and `colors` empty because that model predicts categories only.

### 5. Train and evaluate the CNN baseline (optional)

Train either supported architecture:

```powershell
python scripts\train_cnn.py --model resnet50 --epochs 20 --batch 64 --lr 1e-3
python scripts\train_cnn.py --model efficientnet --epochs 20 --batch 64 --lr 1e-3
```

The best checkpoint is saved under `models\` (`resnet50_best.pth` or `efficientnet_best.pth`). Evaluate it on the test split:

```powershell
python scripts\eval_cnn.py `
  --checkpoint models\resnet50_best.pth `
  --model resnet50
```

To use a different CNN checkpoint in the web app, set `CNN_CHECKPOINT`; to change its architecture, set `CNN_MODEL`:

```powershell
$env:CNN_CHECKPOINT = "models\resnet50_best.pth"
$env:CNN_MODEL = "resnet50"
python app.py
```

### 6. Enable local LLaVA through Ollama (optional)

Install Ollama, ensure its service is running, and create the model defined by [`Modelfile`](./Modelfile):

```powershell
ollama create deepfashion-llava -f Modelfile
```

Select `ollama/llava` in the UI or send that value as the `model` form field. The integration calls `http://localhost:11434/api/generate` and does not require a cloud API key.

## Academic authorship

This repository was developed collaboratively for the Artificial Intelligence Project course of the Master's Degree in Artificial Intelligence at the University of Minho (UMinho). It combines the implementation, experiments, benchmarking notebooks, and project documentation produced by the team.
