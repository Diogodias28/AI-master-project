"""
app.py — Tarefa 2.4: Interface Web para Upload e Visualização
===============================================================
Flask app com drag-and-drop, preview da imagem e exibição dos metadados
previstos pelo pipeline CLIP. Suporta troca dinâmica de modelo.
"""
import json
import os
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Adiciona src/ ao path para importar o predictor
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from predictor import FashionPredictor
from ollama_predictor import predict_with_ollama

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}

# ── Catálogo de modelos disponíveis ──
AVAILABLE_MODELS = {
    "patrickjohncyh/fashion-clip": "Fashion-CLIP (recomendado)",
    "openai/clip-vit-base-patch32": "CLIP ViT-B/32 (OpenAI)",
    "openai/clip-vit-base-patch16": "CLIP ViT-B/16 (OpenAI)",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": "CLIP ViT-B/32 (LAION-2B)",
    "openai/clip-vit-large-patch14": "CLIP ViT-L/14 (OpenAI — pesado)",
    "ollama/llava": "LLaVA via Ollama (multimodal local)",
}

DEFAULT_MODEL = "patrickjohncyh/fashion-clip"

# ── Inicialização do modelo CLIP (pesado, carrega uma vez) ──
print(f"[Flask] A inicializar FashionPredictor com {DEFAULT_MODEL} …")
predictor = None
current_model = None


def get_predictor(model_name: str = DEFAULT_MODEL):
    global predictor, current_model
    if predictor is None or current_model != model_name:
        print(f"[Flask] A carregar modelo {model_name} …")
        predictor = FashionPredictor(data_root=".", model_name=model_name, device=None, use_tta=True)
        current_model = model_name
    return predictor


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html", models=AVAILABLE_MODELS, default_model=DEFAULT_MODEL)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Nome de ficheiro vazio"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Formato não suportado. Use PNG, JPG, JPEG, GIF ou WEBP."}), 400

    # Guarda upload
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename}"
    upload_path = Path(app.config["UPLOAD_FOLDER"]) / filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    file.save(upload_path)

    # Verifica se o utilizador pediu sem TTA
    no_tta = request.form.get("no_tta", "false").lower() == "true"

    # Modelo escolhido
    model_name = request.form.get("model", DEFAULT_MODEL)
    if model_name not in AVAILABLE_MODELS:
        model_name = DEFAULT_MODEL

    # Executa predição
    try:
        if model_name == "ollama/llava":
            pred = predict_with_ollama(upload_path)
        else:
            pred = get_predictor(model_name).predict(upload_path, use_tta=not no_tta)

        pred["upload_url"] = f"/uploads/{filename}"
        pred["model_used"] = model_name

        # Grava também JSON no lado do servidor
        json_path = upload_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(pred, f, indent=2, ensure_ascii=False)
        return jsonify(pred)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Endpoint API puro (aceita multipart)."""
    return predict()


if __name__ == "__main__":
    # Garante que a pasta uploads existe
    Path("uploads").mkdir(exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
