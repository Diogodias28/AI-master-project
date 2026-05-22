"""
ollama_predictor.py — Wrapper para LLaVA via Ollama
======================================================
Envia imagens para o modelo deepfashion-llava local (Ollama)
e normaliza o output para o mesmo schema JSON do FashionPredictor.
"""
import base64
import json
from datetime import datetime
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepfashion-llava"


def encode_image(img_path: str | Path) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def predict_with_ollama(img_path: str | Path, timeout: int = 120) -> dict:
    """
    Envia imagem para Ollama deepfashion-llava e devolve JSON normalizado.
    """
    b64 = encode_image(img_path)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": "Analyze this clothing image and output only JSON.",
        "images": [b64],
        "stream": False,
        "format": "json",
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    raw = json.loads(data.get("response", "{}"))

    # ── Normalização para schema do CLIP ──
    now = datetime.now().isoformat()

    # Cores: o llava pode usar "color" em vez de "name"
    colors = []
    for c in raw.get("colors", [])[:5]:
        colors.append({
            "name": c.get("name") or c.get("color", "unknown"),
            "confidence": float(c.get("confidence", 0.5))
        })

    # Atributos: garantir top-10
    attrs = []
    for a in raw.get("attributes", [])[:10]:
        attrs.append({
            "name": a.get("name", ""),
            "confidence": float(a.get("confidence", 0.8)),
            "type": str(a.get("type", "3"))
        })

    # Categoria
    cat = raw.get("category", {})
    top_k = raw.get("top_k", [])
    if not top_k and cat:
        top_k = [{"rank": 1, "label": cat.get("label", ""), "confidence": 0.85}]

    result = {
        "image": str(img_path),
        "category": {
            "label": cat.get("label", "Unknown"),
            "confidence": float(cat.get("confidence", 0.85)),
            "top_k": top_k
        },
        "attributes": attrs,
        "colors": colors,
        "metadata": {
            "model": "ollama/llava",
            "device": "ollama local",
            "tta_used": False,
            "attr_threshold": 0.22,
            "timestamp": now
        }
    }
    return result
