#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, io, json, argparse
from typing import List, Dict, Any, Tuple, Optional

import torch
import pandas as pd
from PIL import Image
import numpy as np
import requests

from transformers import AutoImageProcessor, AutoModel

def l2_normalize(t: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return t / (t.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))


def get_image_embeddings(model, processor, images: List[Image.Image], device: torch.device) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        if hasattr(model, "get_image_features"):
            feats = model.get_image_features(**inputs)
        else:
            out = model(**inputs)
            feats = out.last_hidden_state.mean(dim=1)
    return l2_normalize(feats, dim=-1)


def combine_embeddings(embs: torch.Tensor) -> torch.Tensor:
    if embs.ndim == 1:
        embs = embs.unsqueeze(0)
    return l2_normalize(embs.mean(dim=0), dim=0)


def ensure_outdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def safe_folder(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-+" else "_" for c in s)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Path to JSONL with App ID and Image links.")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for parquet files.")
    ap.add_argument("--limit", type=int, default=5, help="Max number of apps to process.")
    # Offline model usage
    ap.add_argument("--model_dir", type=str, default=None, help="Local snapshot dir of the model (offline).")
    ap.add_argument("--model_id", type=str, default="google/siglip-base-patch16-224", help="HF ID if online.")
    ap.add_argument("--local_images_root", type=str, default=None,
                    help="If set, read images from this root: <root>/<App ID>/*.jpg")
    return ap.parse_args()


def load_model(processor_id: str,
               model_id: str,
               local: bool,
               device: torch.device,
               model_dir: Optional[str] = None):
    """
    If local is True, load both processor and model from model_dir.
    Otherwise, load from the HF Hub using the given IDs.
    """
    if local:
        if not model_dir or not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Local model_dir not found or not a directory: {model_dir}")
        processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=True)
        model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    else:
        processor = AutoImageProcessor.from_pretrained(processor_id, use_fast=True)
        model = AutoModel.from_pretrained(model_id)

    model.to(device).eval()
    return processor, model


def list_local_images(root: str, app_id: str) -> List[str]:
    d = os.path.join(root, safe_folder(app_id))
    if not os.path.isdir(d):
        return []
    files = sorted(
        os.path.join(d, fn) for fn in os.listdir(d)
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
    )
    return files


def fetch_image_http(url: str, timeout: int = 20) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def open_local_image(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def main():
    args = parse_args()
    ensure_outdir(args.outdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    offline = bool(args.model_dir)

    # Preflight checks
    if offline and (not args.model_dir or not os.path.isdir(args.model_dir)):
        raise FileNotFoundError(f"--model_dir is required and must exist in offline mode: {args.model_dir}")
    if args.local_images_root and not os.path.isdir(args.local_images_root):
        raise FileNotFoundError(f"--local_images_root not found: {args.local_images_root}")
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"--input not found: {args.input}")

    proc_id = args.model_dir or args.model_id
    model_id = args.model_dir or args.model_id
    processor, model = load_model(proc_id, model_id, local=offline, device=device, model_dir=args.model_dir)

    # Read up to N apps
    apps: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if args.limit and len(apps) >= args.limit:
                break
            if not line.strip():
                continue
            o = json.loads(line)
            app_id = o.get("App ID")
            links = o.get("Image links") or []
            if app_id and isinstance(links, list) and links:
                apps.append({"App ID": app_id, "Image links": links})

    image_rows, app_rows, broken_rows = [], [], []

    for app in apps:
        app_id = app["App ID"]
        used_images: List[Image.Image] = []
        used_sources: List[str] = []

        if args.local_images_root:
            # Offline: use pre-fetched local images
            paths = list_local_images(args.local_images_root, app_id)
            for p in paths:
                img = open_local_image(p)
                if img is not None:
                    used_images.append(img)
                    used_sources.append(p)
                else:
                    broken_rows.append({"app_id": app_id, "url": p, "reason": "Cannot open local image"})
        else:
            # Online: fetch from URLs
            for url in app["Image links"]:
                img = fetch_image_http(url)
                if img is not None:
                    used_images.append(img)
                    used_sources.append(url)
                else:
                    broken_rows.append({"app_id": app_id, "url": url, "reason": "HTTP fetch/open failed"})

        if not used_images:
            app_rows.append({"app_id": app_id, "num_images_used": 0, "dim": None, "embedding": None})
            continue

        # Batch embeddings
        bs, all_embs = 8, []
        for i in range(0, len(used_images), bs):
            embs = get_image_embeddings(model, processor, used_images[i:i+bs], device)
            all_embs.append(embs.cpu())
        embs = torch.cat(all_embs, dim=0)  # [N, D]
        dim = int(embs.shape[1])

        # Per image
        for idx, (src, vec) in enumerate(zip(used_sources, embs)):
            image_rows.append({
                "app_id": app_id,
                "image_index": idx,
                "source": src,
                "dim": dim,
                "embedding": json.dumps(vec.tolist())
            })

        # Combined per app
        combined = combine_embeddings(embs.to(device)).cpu().numpy().tolist()
        app_rows.append({
            "app_id": app_id,
            "num_images_used": len(used_images),
            "dim": dim,
            "embedding": json.dumps(combined)
        })

    # Write Parquet outputs
    image_df = pd.DataFrame(image_rows)
    app_df = pd.DataFrame(app_rows)
    broken_df = pd.DataFrame(broken_rows)

    img_path = os.path.join(args.outdir, "image_embeddings.parquet")
    app_path = os.path.join(args.outdir, "app_embeddings.parquet")
    bro_path = os.path.join(args.outdir, "broken_links.parquet")

    (image_df if not image_df.empty else pd.DataFrame(columns=["app_id", "image_index", "source", "dim", "embedding"])).to_parquet(img_path, index=False)
    (app_df if not app_df.empty else pd.DataFrame(columns=["app_id", "num_images_used", "dim", "embedding"])).to_parquet(app_path, index=False)
    (broken_df if not broken_df.empty else pd.DataFrame(columns=["app_id", "url", "reason"])).to_parquet(bro_path, index=False)

    print("Saved:", img_path)
    print("Saved:", app_path)
    print("Saved:", bro_path)


if __name__ == "__main__":
    main()
