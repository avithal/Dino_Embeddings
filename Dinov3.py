"""
Requirements (recommended):
  pip install --upgrade git+https://github.com/huggingface/transformers.git
  pip install torch torchvision
  pip install pillow scikit-learn matplotlib tqdm
(If transformers stable release supports DINOv3 in your environment, `pip install transformers` is fine.)
References: Hugging Face DINOv3 docs and facebookresearch dinov3 repo.
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel  # transformers >= support for dinov3
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------
# Settings
# -------------------------
# Replace with a DINOv3 model from Hugging Face that you have access to.
#MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
MODEL_NAME = "facebook/dinov2-base"


# folder containing images you want to embed
IMAGE_FOLDER = r"D:\Dataset\images_to_embedd"  # put your images here

BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
RANDOM_SEED = 42

# -------------------------
# Helper dataset
# -------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, folder: str, image_processor):
        self.folder = Path(folder)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        #self.paths = [p for p in sorted(self.folder.iterdir()) if p.suffix.lower() in exts]


        # supported image extensions
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        # recursively find all image paths under root
        self.paths = [p for p in self.folder.rglob("*") if p.suffix.lower() in exts]

        self.processor = image_processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        # Let the processor handle resizing / normalization etc.
        processed = self.processor(images=img, return_tensors="pt")
        # processed is dict e.g. {"pixel_values": tensor([1, C, H, W])}
        return {"pixel_values": processed["pixel_values"].squeeze(0), "path": str(p)}

# -------------------------
# Embedding extraction
# -------------------------
def collate_batch(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
    paths = [item["path"] for item in batch]
    return {"pixel_values": pixel_values, "paths": paths}

@torch.no_grad()
def compute_embeddings(model, processor, dataloader, device):
    model.eval()
    embeddings = []
    img_paths = []
    for batch in tqdm(dataloader, desc="Embedding batches"):
        pixel_values = batch["pixel_values"].to(device)  # shape (B, C, H, W) or (B, num_channels, H, W)
        # For DINOv3 via Hugging Face AutoModel, the forward returns last_hidden_state (B, seq_len, dim).
        outputs = model(pixel_values=pixel_values, output_hidden_states=False, return_dict=True)
        # Many vision models (ViT-style) put a CLS token at position 0, or you can mean-pool patch tokens.
        # We'll try to use a global pooled output if available, otherwise mean pool:
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output  # shape (B, hidden_dim)
        else:
            # last_hidden_state: (B, seq_len, hidden_dim)
            last = outputs.last_hidden_state  # tensor
            # exclude any special tokens if present; usually CLS is at index 0 — taking mean is robust:
            emb = last.mean(dim=1)
        emb = emb.cpu().numpy()
        embeddings.append(emb)
        img_paths.extend(batch["paths"])
    embeddings = np.vstack(embeddings)  # (N, D)
    return embeddings, img_paths

# -------------------------
# TSNE visualization
# -------------------------
def plot_tsne(embeddings, paths, title="DINOv2 embeddings (t-SNE)", figsize=(10, 8)):
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, n_iter=TSNE_N_ITER, random_state=RANDOM_SEED)
    emb2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=figsize)
    plt.scatter(emb2d[:, 0], emb2d[:, 1], s=35)
    # annotate a subset to avoid clutter; annotate every nth point
    n = max(1, len(paths) // 50)
    for i, p in enumerate(paths):
        if i % n == 0:
            label = Path(p).name
            plt.annotate(label, (emb2d[i, 0], emb2d[i, 1]), fontsize=8, alpha=0.8)
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.show()

# -------------------------
# Main
# -------------------------
def main():
    # load processor and model
    print("Loading model and image processor:", MODEL_NAME)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    #print("✅ Loaded successfully!")

    # dataset + dataloader
    ds = ImageFolderDataset(IMAGE_FOLDER, processor)
    if len(ds) == 0:
        raise RuntimeError(f"No images found in {IMAGE_FOLDER}. Put .jpg/.png files there.")
    dl = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_batch)

    # compute embeddings
    embeddings, paths = compute_embeddings(model, processor, dl, DEVICE)
    print("Embeddings shape:", embeddings.shape)

    # t-SNE and plot
    plot_tsne(embeddings, paths)

if __name__ == "__main__":
    main()
