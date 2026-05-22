"""
predictor.py — Tarefas 2.2 & 2.3
Pipeline CLIP com Test-Time Augmentation (TTA) e output JSON estruturado.
"""
import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings("ignore")

# ── Templates de prompt ensemble ──
CAT_TEMPLATES = [
    "a photo of a {}",
    "a fashion photo of a {}",
    "a person wearing a {}",
    "a clothing product photo of a {}",
    "a studio photo of a {}",
]

ATTR_TEMPLATES = {
    "1": [  # padrão
        "clothing with {} pattern",
        "a {} print fabric",
        "a garment with {} design",
    ],
    "2": [  # textura / acabamento
        "a {} fabric garment",
        "clothing with {} texture",
        "a {} finish clothing item",
    ],
    "3": [  # silhueta
        "a {} style clothing",
        "a {} cut garment",
        "a {} shaped piece of clothing",
    ],
    "6": [  # cor
        "{} colored clothing",
        "a garment in {} color",
        "clothing that is {}",
    ],
}

USEFUL_ATTR_TYPES = {"1", "2", "3", "6"}


# ── Data Augmentation para TTA ──
def get_tta_transforms():
    """Devolve lista de transforms para TTA."""
    base = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])

    flip = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])

    jitter = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])

    return [base, flip, jitter]


class FashionPredictor:
    """
    Encapsula todo o pipeline CLIP para previsão de moda DeepFashion.
    """
    def __init__(self, data_root: str | Path = ".",
                 model_name: str = "patrickjohncyh/fashion-clip",
                 device: str | None = None,
                 attr_threshold: float = 0.22,
                 use_tta: bool = True):
        self.data_root = Path(data_root)
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.attr_threshold = attr_threshold
        self.use_tta = use_tta

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        print(f"[Predictor] A carregar modelo {model_name} em {self.device}…")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Carrega metadados
        self.idx_to_cat, self.cat_type = self._load_categories()
        self.idx_to_attr, self.attr_type = self._load_attributes()
        self.idx_to_color, self.color_type = self._load_colors()

        # Pré-computa text banks
        print("[Predictor] A construir text banks…")
        self.cat_idxs, self.cat_bank = self._build_category_text_bank()
        self.attr_idxs, self.attr_bank = self._build_attr_text_bank()
        self.color_idxs, self.color_bank = self._build_color_text_bank()

        print(f"[Predictor] Pronto. {len(self.idx_to_cat)} categorias, {len(self.idx_to_attr)} atributos, {len(self.idx_to_color)} cores.")

    # ── Leitura de metadados ──
    def _load_categories(self):
        cat_file = self.data_root / "Anno_coarse" / "list_category_cloth.txt"
        idx_to_name, idx_to_type = {}, {}
        with open(cat_file) as f:
            f.readline(); f.readline()
            for i, line in enumerate(f, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                idx_to_type[i] = parts[-1]
                idx_to_name[i] = " ".join(parts[:-1])
        return idx_to_name, idx_to_type

    def _load_attributes(self):
        attr_file = self.data_root / "Anno_coarse" / "list_attr_cloth.txt"
        idx_to_name, idx_to_type = {}, {}
        with open(attr_file) as f:
            f.readline(); f.readline()
            for i, line in enumerate(f, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                atype = parts[-1]
                if atype in USEFUL_ATTR_TYPES:
                    idx_to_name[i] = " ".join(parts[:-1])
                    idx_to_type[i] = atype
        return idx_to_name, idx_to_type

    def _load_colors(self):
        """Lê list_attr_colors.txt e devolve dicionário de cores."""
        color_file = self.data_root / "Anno_coarse" / "list_attr_colors.txt"
        idx_to_name, idx_to_type = {}, {}
        if not color_file.exists():
            print(f"[Predictor] Aviso: {color_file} não encontrado — cores desativadas.")
            return idx_to_name, idx_to_type
        with open(color_file) as f:
            f.readline(); f.readline()
            for i, line in enumerate(f, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                idx_to_name[i] = " ".join(parts[:-1])
                idx_to_type[i] = parts[-1]
        return idx_to_name, idx_to_type

    # ── Text banks ──
    @torch.no_grad()
    def _encode_texts(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_text_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output if hasattr(feats, "pooler_output") and feats.pooler_output is not None else feats.last_hidden_state[:, 0, :]
        return F.normalize(feats, dim=-1)

    def _build_category_text_bank(self):
        ordered = sorted(self.idx_to_cat.keys())
        embs = []
        for idx in ordered:
            name = self.idx_to_cat[idx]
            prompts = [t.format(name.lower()) for t in CAT_TEMPLATES]
            emb = self._encode_texts(prompts)
            avg = F.normalize(emb.mean(dim=0, keepdim=True), dim=-1)
            embs.append(avg)
        return ordered, torch.cat(embs, dim=0)

    def _build_attr_text_bank(self):
        ordered = sorted(self.idx_to_attr.keys())
        embs = []
        for idx in ordered:
            name = self.idx_to_attr[idx]
            atype = self.attr_type[idx]
            templates = ATTR_TEMPLATES.get(atype, CAT_TEMPLATES)
            prompts = [t.format(name.lower()) for t in templates]
            emb = self._encode_texts(prompts)
            avg = F.normalize(emb.mean(dim=0, keepdim=True), dim=-1)
            embs.append(avg)
        return ordered, torch.cat(embs, dim=0)

    def _build_color_text_bank(self):
        ordered = sorted(self.idx_to_color.keys())
        embs = []
        for idx in ordered:
            name = self.idx_to_color[idx]
            templates = ATTR_TEMPLATES.get("6", CAT_TEMPLATES)
            prompts = [t.format(name.lower()) for t in templates]
            emb = self._encode_texts(prompts)
            avg = F.normalize(emb.mean(dim=0, keepdim=True), dim=-1)
            embs.append(avg)
        return ordered, torch.cat(embs, dim=0)

    # ── Inferência ──
    @torch.no_grad()
    def _encode_image(self, img_path: str | Path, force_tta: bool | None = None) -> torch.Tensor:
        """Codifica uma imagem com TTA opcional."""
        img_path = str(img_path)
        image = Image.open(img_path).convert("RGB")
        tta = self.use_tta if force_tta is None else force_tta

        if not tta:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self.model.get_image_features(**inputs)
            if not isinstance(feats, torch.Tensor):
                feats = feats.pooler_output if hasattr(feats, "pooler_output") and feats.pooler_output is not None else feats.last_hidden_state[:, 0, :]
            return F.normalize(feats, dim=-1)

        # TTA: aplica vários augmentations e faz média dos embeddings
        tta_transforms = get_tta_transforms()
        embs = []
        for tform in tta_transforms:
            aug = tform(image)
            # processor espera batch de imagens PIL, mas aqui já é tensor
            # usamos o modelo diretamente
            pixel_values = aug.unsqueeze(0).to(self.device)
            feats = self.model.get_image_features(pixel_values=pixel_values)
            if not isinstance(feats, torch.Tensor):
                feats = feats.pooler_output if hasattr(feats, "pooler_output") and feats.pooler_output is not None else feats.last_hidden_state[:, 0, :]
            embs.append(F.normalize(feats, dim=-1))
        avg_emb = F.normalize(torch.cat(embs, dim=0).mean(dim=0, keepdim=True), dim=-1)
        return avg_emb

    # ── Predição ──
    def predict(self, img_path: str | Path, top_k: int = 5, use_tta: bool | None = None) -> dict[str, Any]:
        """
        Executa o pipeline completo numa imagem e devolve dicionário JSON-ready.
        use_tta=None usa o default do construtor; use_tta=True/False força.
        """
        tta = self.use_tta if use_tta is None else use_tta
        img_emb = self._encode_image(img_path, force_tta=tta)

        # Categoria
        sims_cat = (img_emb @ self.cat_bank.T).squeeze(0)
        cat_scores = {self.idx_to_cat[self.cat_idxs[i]]: float(sims_cat[i].cpu())
                      for i in range(len(self.cat_idxs))}
        sorted_cats = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)
        top_category = sorted_cats[0][0]
        top_category_score = sorted_cats[0][1]

        # Atributos
        sims_attr = (img_emb @ self.attr_bank.T).squeeze(0)
        attr_results = []
        for i, idx in enumerate(self.attr_idxs):
            score = float(sims_attr[i].cpu())
            if score >= self.attr_threshold:
                attr_results.append({
                    "name": self.idx_to_attr[idx],
                    "confidence": round(score, 4),
                    "type": self.attr_type[idx]
                })
        attr_results.sort(key=lambda x: x["confidence"], reverse=True)
        attr_results = attr_results[:10]  # limita a 10 atributos

        # Top-k categorias detalhadas
        top_k_cats = [
            {"rank": r + 1, "label": label, "confidence": round(score, 4)}
            for r, (label, score) in enumerate(sorted_cats[:top_k])
        ]

        # Cores (top-5)
        color_results = []
        if self.color_idxs:
            sims_color = (img_emb @ self.color_bank.T).squeeze(0)
            for i, idx in enumerate(self.color_idxs):
                score = float(sims_color[i].cpu())
                color_results.append({
                    "name": self.idx_to_color[idx],
                    "confidence": round(score, 4)
                })
            color_results.sort(key=lambda x: x["confidence"], reverse=True)
            color_results = color_results[:5]

        result = {
            "image": str(img_path),
            "category": {
                "label": top_category,
                "confidence": round(top_category_score, 4),
                "top_k": top_k_cats
            },
            "attributes": attr_results,
            "colors": color_results,
            "metadata": {
                "model": self.model_name,
                "device": self.device,
                "tta_used": self.use_tta,
                "attr_threshold": self.attr_threshold,
                "timestamp": datetime.now().isoformat()
            }
        }
        return result

    def predict_and_save(self, img_path: str | Path, output_path: str | Path | None = None, use_tta: bool | None = None) -> dict:
        result = self.predict(img_path, use_tta=use_tta)
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[Predictor] Resultado gravado em {out}")
        return result


# ── Teste local ──
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="prediction.json")
    parser.add_argument("--no_tta", action="store_true")
    args = parser.parse_args()

    pred = FashionPredictor(use_tta=not args.no_tta)
    res = pred.predict_and_save(args.image, args.output)
    print(json.dumps(res, indent=2, ensure_ascii=False))
