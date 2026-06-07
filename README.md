# DeepFashion AI — Catalogação Inteligente

Interface Web para upload e catalogação automática de peças de roupa via CLIP / Fashion-CLIP e Ollama (LLaVA).

![UI](./ui.png)

## Project Tree

```
.
├── app.py
├── Backlog_PIA(1).pdf
├── deepfashion
│   ├── Anno_coarse
│   │   ├── list_attr_cloth.txt
│   │   ├── list_attr_colors.txt
│   │   ├── list_attr_img.txt
│   │   ├── list_bbox.txt
│   │   ├── list_category_cloth.txt
│   │   └── list_category_img.txt
│   ├── Anno_fine
│   │   ├── list_attr_cloth.txt
│   │   ├── list_attr_img.txt
│   │   ├── list_category_cloth.txt
│   │   ├── test_attr.txt
│   │   ├── test_bbox.txt
│   │   ├── test_cate.txt
│   │   ├── test_landmarks.txt
│   │   ├── test.txt
│   │   ├── train_attr.txt
│   │   ├── train_bbox.txt
│   │   ├── train_cate.txt
│   │   ├── train_landmarks.txt
│   │   ├── train.txt
│   │   ├── val_attr.txt
│   │   ├── val_bbox.txt
│   │   ├── val_cate.txt
│   │   ├── val_landmarks.txt
│   │   └── val.txt
│   ├── Eval
│   |    └── list_eval_partition.txt
│   └── img
├── Modelfile
├── README.md
├── requirements.txt
├── scripts
│   └── setup_dataset.py
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

## Estrutura

| Ficheiro / Diretório | Descrição |
|----------------------|-----------|
| `app.py` | Aplicação Flask (UI + API REST) |
| `src/predictor.py` | Pipeline CLIP/Fashion-CLIP com TTA e output JSON |
| `src/ollama_predictor.py` | Wrapper para LLaVA via Ollama (schema compatível) |
| `src/dataset_manager.py` | Validação, deduplicação e split do dataset DeepFashion |
| `scripts/setup_dataset.py` | Download automático do dataset DeepFashion (gdown) |
| `Modelfile` | Definição do modelo Ollama customizado (base: `llava`) |
| `deepfashion/` | Dataset DeepFashion (anotações coarse + fine + eval + imagens) |
| `static/`, `templates/` | Assets da interface web |

## Setup

### 1. Dependências Python

```bash
pip install -r requirements.txt
```

### 2. Dataset DeepFashion

```bash
python scripts/setup_dataset.py
```

Ou manualmente: colocar o dataset em `deepfashion/` com a estrutura esperada (`Anno_coarse/`, `Anno_fine/`, `Eval/`, `img/`).

### 3. Ollama (opcional, para LLaVA local)

1. Instalar [Ollama](https://ollama.com/)
2. Criar o modelo customizado:

```bash
ollama create deepfashion-llava -f Modelfile
```

## Run

```bash
python3 app.py
```

A aplicação fica disponível em `http://localhost:5000`.

## Modelos Suportados

A interface permite alternar dinamicamente entre os seguintes modelos:

| Modelo | Descrição |
|--------|-----------|
| `patrickjohncyh/fashion-clip` | **Padrão** — Fashion-CLIP otimizado para vestuário |
| `openai/clip-vit-base-patch32` | CLIP ViT-B/32 (OpenAI) |
| `openai/clip-vit-base-patch16` | CLIP ViT-B/16 (OpenAI) |
| `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | CLIP ViT-B/32 (LAION-2B) |
| `openai/clip-vit-large-patch14` | CLIP ViT-L/14 (OpenAI — mais pesado) |
| `ollama/llava` | LLaVA finetuned (deepfashion-llava) via Ollama |

## API Endpoints

- `POST /predict` — Upload de imagem + predição (multipart/form-data)
- `POST /api/predict` — Endpoint API puro (mesma funcionalidade)
- `GET /uploads/<filename>` — Servir imagens carregadas

Parâmetros opcionais no `POST`:
- `model` — nome do modelo a usar (ver tabela acima)
- `no_tta` — `"true"` para desativar Data Augmentation

## TODO
- Arranjar alternativas para o ollama (AI gateway Vercel / Ollama Cloud?)
