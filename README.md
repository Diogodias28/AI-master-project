# DeepFashion AI — Catalogação Inteligente

Interface Web para upload e catalogação automática de peças de roupa via CLIP / Fashion-CLIP, com suporte total para Ollama (LLaVA) já configurado.

## Project Tree

```
./
├── app.py
├── Backlog_PIA(1).pdf
├── requirements.txt
├── src
│   ├── dataset_manager.py
│   ├── ollama_predictor.py
│   └── predictor.py
├── static
│   ├── css
│   │   └── style.css
│   └── js
│       └── main.js
├── templates
│   └── index.html
└── ui.png
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python3.12 app.py
```

A aplicação fica disponível em `http://localhost:5000`.

## TODO
Adcionar SYSTEM.md para criar o deepfashion-llava
