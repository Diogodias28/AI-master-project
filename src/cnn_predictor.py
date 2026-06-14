"""
cnn_predictor.py — Inferência com ResNet50/EfficientNet treinado em DeepFashion
==============================================================================
Wrapper que carrega o modelo CNN treinado e devolve metadados num JSON
idêntico ao schema do FashionPredictor (CLIP), mas com campos simplificados:
  • category (label, confidence, top_k)
  • metadata (model, device, timestamp)

Nota: Atributos e cores não são detetados pela CNN (apenas categoria).
"""
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from cnn_dataset import CAT_NAMES, NUM_CLASSES
from cnn_trainer import build_resnet50, build_efficientnet

warnings.filterwarnings("ignore")

# ── Transform de inferência ──
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class CNNPredictor:
    """
    Predictor CNN com output normalizado para o mesmo schema do CLIP.
    """
    def __init__(self, checkpoint_path: str | Path,
                 model_name: str = "resnet50",
                 device: str | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[CNNPredictor] A carregar {model_name} de {checkpoint_path} …")

        # Reconstrói arquitetura
        if model_name == "resnet50":
            self.model = build_resnet50(pretrained=False)
        elif model_name == "efficientnet":
            self.model = build_efficientnet(pretrained=False)
        else:
            raise ValueError(f"Modelo não suportado: {model_name}")

        # Carrega pesos
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(self.device).eval()

        print(f"[CNNPredictor] Pronto. {NUM_CLASSES} categorias em {self.device}.")

    @torch.no_grad()
    def predict(self, img_path: str | Path, top_k: int = 5) -> dict[str, Any]:
        """
        Executa inferência numa imagem e devolve dicionário JSON-ready.
        """
        image = Image.open(img_path).convert("RGB")
        tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()

        # Top-k
        topk_scores, topk_idxs = torch.topk(probs, min(top_k, NUM_CLASSES))
        top_k_list = [
            {"rank": r + 1, "label": CAT_NAMES[idx], "confidence": round(score.item(), 4)}
            for r, (score, idx) in enumerate(zip(topk_scores, topk_idxs))
        ]

        top_category = CAT_NAMES[topk_idxs[0]]
        top_confidence = probs[topk_idxs[0]].item()

        result = {
            "image": str(img_path),
            "category": {
                "label": top_category,
                "confidence": round(top_confidence, 4),
                "top_k": top_k_list
            },
            "attributes": [],
            "colors": [],
            "metadata": {
                "model": f"cnn/{self.model_name}",
                "device": self.device,
                "tta_used": False,
                "attr_threshold": 0.22,
                "timestamp": datetime.now().isoformat()
            }
        }
        return result

    def predict_and_save(self, img_path: str | Path, output_path: str | Path | None = None) -> dict:
        result = self.predict(img_path)
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[CNNPredictor] Resultado gravado em {out}")
        return result


# ── Teste local ──
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="models/resnet50_best.pth")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--output", type=str, default="prediction_cnn.json")
    args = parser.parse_args()

    pred = CNNPredictor(checkpoint_path=args.checkpoint, model_name=args.model)
    res = pred.predict_and_save(args.image, args.output)
    print(json.dumps(res, indent=2, ensure_ascii=False))
