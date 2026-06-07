import os
import zipfile
import gdown

BASE_DIR = "deepfashion"

DOWNLOADS = [
    {
        "url"    : "https://drive.google.com/uc?id=0B7EVK8r0v71pa2EyNEJ0dE9zbU0&export=download&resourcekey=0-CPiKS-AiE8IDonk54WJ5_w&confirm=t",
        "name"   : "img.zip"
    },
    # Eval
    {
        "url"    : "https://drive.google.com/file/d/0B7EVK8r0v71pdS1FMlNreEwtc1E",
        "name"   : "Eval/list_eval_partition.txt"
    },
    # Anno_coarse
    {
        "url"    : "https://drive.google.com/file/d/0B7EVK8r0v71pbU5semhCZ1Jwdkk",
        "name"   : "Anno_coarse/list_bbox.txt"
    },
    {
        "url"    : "https://drive.google.com/file/d/0B7EVK8r0v71pYnBKQVBOaHR1WWs",
        "name"   : "Anno_coarse/list_landmarks.txt"
    },
    {
        "url"    : "https://drive.google.com/file/d/0B7EVK8r0v71pWnFiNlNGTVloLUk",
        "name"   : "Anno_coarse/list_attr_cloth.txt"
    },
    {
        "url"    : "https://drive.google.com/file/d/0B7EVK8r0v71pWXE4QWotX2hxQ1U",
        "name"   : "Anno_coarse/list_attr_img.txt"
    },
    {
        "url"    : "https://drive.google.com/file/d/0B7EVK8r0v71pTGNoWkhZeVpzbFk",
        "name"   : "Anno_coarse/list_category_cloth.txt"
    },
    {
        "url"    : "https://drive.google.com/file/d/0B7EVK8r0v71pT0RDNENnV2NiWTQ",
        "name"   : "Anno_coarse/list_category_img.txt"
    },
]

def create_color_file(base_dir="deepfashion"):
    content = """18
color_name  color_type
black       6
white       6
red         6
blue        6
green       6
yellow      6
beige       6
navy        6
grey        6
brown       6
pink        6
purple      6
orange      6
burgundy    6
teal        6
olive       6
ivory       6
coral       6
"""
    path = os.path.join(base_dir, "Anno_coarse", "list_attr_colors.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"Created: {path}")




def download_item(item: dict, base_dir: str) -> None:
    url     = item["url"]
    name    = item["name"]

    dest_path = os.path.join(base_dir, name)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"{'='*60}")

    gdown.download(url=url, output=dest_path, quiet=False)

    if name == "img.zip":
        print(f"\nExtracting {name} ...")
        extract_dir = os.path.join(base_dir, "img")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(dest_path, "r") as zf:
            zf.extractall(extract_dir)
        print(f"Extracted to : {extract_dir}")
        os.remove(dest_path)
        print(f"Removed zip  : {dest_path}")


def main():
    print("DeepFashion Dataset Downloader")
    print("=" * 60)
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Output directory: {os.path.abspath(BASE_DIR)}")

    failed = []
    for item in DOWNLOADS:
        try:
            download_item(item, BASE_DIR)
            print(f"✓ Done: {item['name']}")
        except Exception as e:
            print(f"✗ Failed: {item['name']} — {e}")
            failed.append(item["name"])

    create_color_file()
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total   = len(DOWNLOADS)
    success = total - len(failed)
    print(f"  Succeeded : {success}/{total}")
    if failed:
        print(f"  Failed    : {', '.join(failed)}")
    else:
        print("  All files downloaded successfully!")
    print(f"  Location  : {os.path.abspath(BASE_DIR)}/")


if __name__ == "__main__":
    main()