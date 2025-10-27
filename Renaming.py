import os
from pathlib import Path

def rename_images_by_folder(root_dir):
    """
    Recursively renames images in each subfolder as <foldername>_<index>.<ext>.
    Example:
      images_to_embed/
        airplane/
          img1.jpg -> airplane_001.jpg
        bird/
          photo.png -> bird_001.png
    """
    root = Path(root_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    for folder in root.rglob("*"):
        if folder.is_dir():
            images = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
            if not images:
                continue

            print(f"📂 Renaming in folder: {folder}")
            for i, img_path in enumerate(images, start=1):
                new_name = f"{folder.name}_{i:03d}{img_path.suffix.lower()}"
                new_path = img_path.with_name(new_name)
                # avoid overwriting if already named correctly
                if new_path != img_path:
                    img_path.rename(new_path)
            print(f"✅ Renamed {len(images)} images in '{folder.name}'")

# Example usage:
if __name__ == "__main__":
    rename_images_by_folder( r"D:\Dataset\images_to_embedd")
