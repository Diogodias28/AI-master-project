"""
dataset_manager.py — Tarefa 2.1: Aquisição, Limpeza e Organização do Dataset
================================================================================
Responsável por validar imagens do DeepFashion, remover duplicadas simples
(e.g. por nome/size) e exportar um mapeamento limpo JSON para uso do pipeline.
"""
import json
import hashlib
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def validate_images(data_root: Path, min_size: int = 224, output_json: str = "dataset_index.json") -> dict:
    """
    Percorre img/ e valida cada imagem:
      - consegue abrir com PIL
      - tem largura/altura >= min_size
      - formato RGB (converte se necessário)
    
    Devolve um dicionário com estatísticas e lista de imagens válidas,
    e grava o índice em output_json.
    """
    img_dir = data_root / "img"
    valid_images = []
    invalid_count = 0
    duplicate_count = 0
    seen_hashes = set()

    all_images = sorted(img_dir.rglob("*.jpg"))
    print(f"[Dataset] Encontradas {len(all_images)} imagens em {img_dir}")

    for img_path in tqdm(all_images, desc="Validando"):
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                w, h = im.size
                if w < min_size or h < min_size:
                    invalid_count += 1
                    continue

                # hash simples do conteúdo (primeiros 4KB) para detetar duplicados exatos
                with open(img_path, "rb") as f:
                    file_hash = hashlib.md5(f.read(4096)).hexdigest()
                if file_hash in seen_hashes:
                    duplicate_count += 1
                    continue
                seen_hashes.add(file_hash)

                valid_images.append({
                    "path": str(img_path.relative_to(data_root)),
                    "width": w,
                    "height": h,
                    "folder": img_path.parent.name
                })
        except Exception as e:
            invalid_count += 1
            continue

    dataset_index = {
        "total_found": len(all_images),
        "valid": len(valid_images),
        "invalid": invalid_count,
        "duplicates_removed": duplicate_count,
        "images": valid_images
    }

    out_path = data_root / output_json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset_index, f, indent=2, ensure_ascii=False)

    print(f"[Dataset] Índice gravado em {out_path}")
    print(f"          Válidas: {len(valid_images)} | Inválidas: {invalid_count} | Duplicados: {duplicate_count}")
    return dataset_index


def split_dataset(data_root: Path, index_json: str = "dataset_index.json", train_ratio=0.7, val_ratio=0.15, seed: int = 42):
    """
    Divide o índice em train / val / test e grava os ficheiros split_*.json.
    """
    import random
    index_path = data_root / index_json
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split_name, split_imgs in splits.items():
        out_path = data_root / f"split_{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_imgs, f, indent=2, ensure_ascii=False)
        print(f"[Dataset] {split_name}: {len(split_imgs)} imagens → {out_path}")

    return splits


if __name__ == "__main__":
    data_root = Path("deepfashion")
    index = validate_images(data_root)
    split_dataset(data_root)
