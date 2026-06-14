"""
cnn_dataset.py — Dataset PyTorch para DeepFashion (classificação de categoria)
================================================================================
Carrega o dataset DeepFashion com anotações coarse (50 categorias) e o split
oficial (list_eval_partition.txt) para train/val/test.
"""
import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ── DeepFashion Category Map (1-indexed → 0-indexed) ──
CAT_NAMES = [
    "Anorak", "Blazer", "Blouse", "Bomber", "Button-Down", "Cardigan", "Flannel",
    "Halter", "Henley", "Hoodie", "Jacket", "Jersey", "Parka", "Peacoat", "Poncho",
    "Sweater", "Tank", "Tee", "Top", "Turtleneck", "Capris", "Chinos", "Culottes",
    "Cutoffs", "Gauchos", "Jeans", "Jeggings", "Jodhpurs", "Joggers", "Leggings",
    "Sarong", "Shorts", "Skirt", "Sweatpants", "Sweatshorts", "Trunks", "Caftan",
    "Cape", "Coat", "Coverup", "Dress", "Jumpsuit", "Kaftan", "Kimono", "Nightdress",
    "Onesie", "Robe", "Romper", "Shirtdress", "Sundress",
]

NAME_TO_IDX = {name: i for i, name in enumerate(CAT_NAMES)}
NUM_CLASSES = 50


# ── Transforms ──
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── Dataset ──
class DeepFashionCategoryDataset(Dataset):
    """
    Dataset DeepFashion para classificação de categoria.
    Lê anotações de list_category_img.txt e list_eval_partition.txt.
    """
    def __init__(self, data_root: str | Path, split: str = "train",
                 transform=None, cache_annotations: bool = True):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform or get_val_transforms()
        self.samples = []

        cache_path = self.data_root / f"cnn_{split}_cache.json"
        if cache_annotations and cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                self.samples = json.load(f)
            print(f"[Dataset] Cache carregado: {len(self.samples)} samples ({split})")
        else:
            self._load_annotations()
            if cache_annotations:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(self.samples, f)
                print(f"[Dataset] Cache gravado: {cache_path}")

    def _load_annotations(self):
        cat_file = self.data_root / "Anno_coarse" / "list_category_img.txt"
        split_file = self.data_root / "Eval" / "list_eval_partition.txt"

        # --- 1) Mapear imagem -> categoria ---
        img_to_cat = {}
        with open(cat_file, "r", encoding="utf-8") as f:
            f.readline()  # skip total count
            f.readline()  # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(None, 1)
                if len(parts) != 2:
                    continue
                img_path, cat_label = parts
                img_path = img_path.strip()
                cat_label = int(cat_label.strip())
                # DeepFashion labels are 1-indexed
                cat_idx = cat_label - 1
                if 0 <= cat_idx < NUM_CLASSES:
                    img_to_cat[img_path] = cat_idx

        # --- 2) Filtrar pelo split ---
        with open(split_file, "r", encoding="utf-8") as f:
            f.readline()  # skip total count
            f.readline()  # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(None, 1)
                if len(parts) != 2:
                    continue
                img_path, status = parts
                img_path = img_path.strip()
                status = status.strip()
                if status == self.split and img_path in img_to_cat:
                    self.samples.append({
                        "path": img_path,
                        "label": img_to_cat[img_path],
                    })

        print(f"[Dataset] {self.split}: {len(self.samples)} amostras carregadas")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = self.data_root / sample["path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return image, label


if __name__ == "__main__":
    ds = DeepFashionCategoryDataset("deepfashion", split="train")
    print(f"Train samples: {len(ds)}")
    img, lbl = ds[0]
    print(f"Image shape: {img.shape}, Label: {lbl} ({CAT_NAMES[lbl]})")

    ds_val = DeepFashionCategoryDataset("deepfashion", split="val")
    print(f"Val samples: {len(ds_val)}")

    ds_test = DeepFashionCategoryDataset("deepfashion", split="test")
    print(f"Test samples: {len(ds_test)}")
