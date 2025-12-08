#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pooled Embedding Generator (Text + Image)

This script loads:
    1. A parquet file containing text embeddings
    2. A parquet file containing image embeddings
It groups embeddings by app_id, normalizes them, and produces a pooled
(mean) embedding per app. The result is written to an output parquet file.

All functionality remains identical to the original script. The only change
is that input/output file paths are now provided as command-line arguments
instead of being hardcoded.

Usage example:

    python pooled_embedding_generator.py \
        --text_parquet /path/to/text_embeddings.parquet \
        --image_parquet /path/to/image_embeddings.parquet \
        --output_parquet /path/to/output_pooled.parquet
"""

import os, json
import argparse
from typing import Dict, List, Any
import numpy as np
import pandas as pd


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def load_emb_list_from_parquet(path: str, id_col: str) -> Dict[str, List[np.ndarray]]:
    df = pd.read_parquet(path)
    buckets: Dict[str, List[np.ndarray]] = {}
    for app_id, emb_json in zip(df[id_col].astype(str).values, df["embedding"].values):
        if pd.isna(emb_json):
            continue
        vec = np.asarray(json.loads(emb_json), dtype=np.float32)
        buckets.setdefault(app_id, []).append(vec)
    return buckets


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pooled text+image embeddings.")
    parser.add_argument("--text_parquet", required=True, help="Path to text embeddings parquet file.")
    parser.add_argument("--image_parquet", required=True, help="Path to image embeddings parquet file.")
    parser.add_argument("--output_parquet", required=True, help="Where to write pooled output parquet.")
    return parser.parse_args()


def main():
    args = parse_args()

    text_by_app = load_emb_list_from_parquet(args.text_parquet, "app_id")
    img_by_app  = load_emb_list_from_parquet(args.image_parquet, "app_id")

    all_app_ids = sorted(set(text_by_app.keys()) | set(img_by_app.keys()))
    out_rows: List[Dict[str, Any]] = []

    for app_id in all_app_ids:
        text_list = text_by_app.get(app_id, [])
        img_list  = img_by_app.get(app_id, [])
        all_vecs = text_list + img_list
        if not all_vecs:
            continue
        M = np.stack(all_vecs, axis=0).astype(np.float32)
        M = l2_normalize(M, axis=1)
        pooled = l2_normalize(M.mean(axis=0))
        out_rows.append({
            "app_id": app_id,
            "n_text": len(text_list),
            "n_images": len(img_list),
            "dim": int(pooled.shape[0]),
            "embedding": json.dumps(pooled.tolist())
        })

    out_df = pd.DataFrame(out_rows, columns=["app_id","n_text","n_images","dim","embedding"])
    os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)
    out_df.to_parquet(args.output_parquet, index=False)
    print(f"[done] wrote {len(out_df)} pooled embeddings → {args.output_parquet}")


if __name__ == "__main__":
    main()
