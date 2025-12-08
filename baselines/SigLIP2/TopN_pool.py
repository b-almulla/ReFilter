#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute multimodal similarity rankings between target apps.

This script:
  - Loads text and image embeddings from two parquet files
  - Groups embeddings by app_id (multiple embeddings per modality allowed)
  - L2-normalizes vectors
  - For each target app:
        * Computes a top-k similarity score against all other apps
        * Optionally streams all element-to-element similarities to CSV
        * Writes a ranking CSV of (app_id, similarity, rank)

"""

import os, csv, json
import argparse
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# ----------------------------------------------------
# Argument parsing
# ----------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute multimodal top-k similarity rankings.")

    parser.add_argument(
        "--text_parquet",
        required=True,
        help="Path to TEXT embedding parquet file (columns: app_id, embedding)."
    )
    parser.add_argument(
        "--image_parquet",
        required=True,
        help="Path to IMAGE embedding parquet file (columns: app_id, embedding)."
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Directory where the output CSVs will be written."
    )
    parser.add_argument(
        "--targets",
        required=True,
        help="Comma-separated list of target appIds (e.g., com.a,com.b,com.c)."
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Top-k element similarities to aggregate (default: 3)."
    )
    parser.add_argument(
        "--write_all_pairs",
        action="store_true",
        help="If set, write all element-level similarities to CSV."
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=1024,
        help="Block size for streaming similarity computation."
    )
    return parser.parse_args()

# ----------------------------------------------------
# Helper functions
# ----------------------------------------------------

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-+" else "_" for c in s)

def _read_parquet_minimal(path: str, columns: List[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return pd.read_parquet(path, columns=columns, engine="pyarrow")

def load_embeddings(text_parquet: str, image_parquet: str) -> Dict[str, Dict[str, List[np.ndarray]]]:
    by_app: Dict[str, Dict[str, List[np.ndarray]]] = {}

    def _load(path: str, modality: str):
        df = _read_parquet_minimal(path, ["app_id", "embedding"])
        for app_id, emb_json in zip(df["app_id"].astype(str).values, df["embedding"].values):
            if pd.isna(emb_json):
                continue
            vec = np.asarray(json.loads(emb_json), dtype=np.float32)
            if vec.ndim != 1:
                continue
            by_app.setdefault(app_id, {"text": [], "image": []})[modality].append(vec)

    _load(text_parquet, "text")
    _load(image_parquet, "image")

    for app_id in by_app:
        for mod in ("text", "image"):
            if by_app[app_id][mod]:
                arr = np.stack(by_app[app_id][mod], axis=0).astype(np.float32)
                by_app[app_id][mod] = [v for v in l2_normalize(arr, axis=1)]

    return by_app

def compute_app_similarity_topk(target: np.ndarray, seed: np.ndarray, topk: int, block_size: int) -> float:
    if target.size == 0 or seed.size == 0:
        return 0.0

    top_scores: List[float] = []

    T, S = target.shape[0], seed.shape[0]
    total_pairs = T * S

    if total_pairs <= 2000000:
        sims = target @ seed.T
        flat = sims.reshape(-1)
        if flat.size <= topk:
            return float(np.mean(flat))
        return float(np.mean(np.partition(flat, -topk)[-topk:]))

    for i0 in range(0, T, block_size):
        i1 = min(i0 + block_size, T)
        sims_block = target[i0:i1] @ seed.T
        flat = sims_block.reshape(-1)
        if flat.size <= topk:
            top_scores.extend(flat.tolist())
        else:
            top_scores.extend(np.partition(flat, -topk)[-topk:].tolist())

    if not top_scores:
        return 0.0

    top_scores = np.array(top_scores, dtype=np.float32)
    if top_scores.size <= topk:
        return float(np.mean(top_scores))
    return float(np.mean(np.partition(top_scores, -topk)[-topk:]))

def write_all_pairs_stream(all_pairs_csv: str, T: np.ndarray, S: np.ndarray, tgt: str, seed: str, block_size: int):
    os.makedirs(os.path.dirname(all_pairs_csv), exist_ok=True)
    file_exists = os.path.exists(all_pairs_csv)

    with open(all_pairs_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists or os.path.getsize(all_pairs_csv) == 0:
            w.writerow(["target_app","seed_app","target_elem_idx","seed_elem_idx","similarity"])

        Tn, Sn = T.shape[0], S.shape[0]
        for i0 in range(0, Tn, block_size):
            i1 = min(i0 + block_size, Tn)
            sims = T[i0:i1] @ S.T
            bi = i1 - i0
            for ii in range(bi):
                t_idx = i0 + ii
                row = sims[ii]
                for j in range(Sn):
                    w.writerow([tgt, seed, t_idx, j, f"{row[j]:.7f}"])

# ----------------------------------------------------
# Main
# ----------------------------------------------------

def main():
    args = parse_args()

    target_app_ids = [s.strip() for s in args.targets.split(",") if s.strip()]

    by_app = load_embeddings(args.text_parquet, args.image_parquet)
    seed_app_ids = sorted(by_app.keys())
    os.makedirs(args.output_root, exist_ok=True)

    for tgt in target_app_ids:
        if tgt not in by_app:
            print(f"[warn] target not found: {tgt}")
            continue

        tgt_vecs = by_app[tgt]["text"] + by_app[tgt]["image"]
        if not tgt_vecs:
            print(f"[warn] no embeddings for target: {tgt}")
            continue

        T = np.stack(tgt_vecs, axis=0).astype(np.float32)

        safe_tgt = safe_name(tgt)
        all_pairs_csv = os.path.join(args.output_root, f"{safe_tgt}_all_pairs.csv")
        ranking_csv   = os.path.join(args.output_root, f"{safe_tgt}_ranking.csv")

        if args.write_all_pairs and os.path.exists(all_pairs_csv):
            os.remove(all_pairs_csv)

        scores: List[Tuple[str, float]] = []

        for seed in seed_app_ids:
            if seed == tgt:
                continue

            seed_vecs = by_app[seed]["text"] + by_app[seed]["image"]
            if not seed_vecs:
                scores.append((seed, 0.0))
                continue

            S = np.stack(seed_vecs, axis=0).astype(np.float32)
            score = compute_app_similarity_topk(T, S, args.topk, args.block_size)
            scores.append((seed, score))

            if args.write_all_pairs:
                write_all_pairs_stream(all_pairs_csv, T, S, tgt, seed, args.block_size)

        scores.sort(key=lambda x: x[1], reverse=True)
        rows = [{"app_id": app, "similarity": sim, "rank": i+1}
                for i, (app, sim) in enumerate(scores)]

        pd.DataFrame(rows).to_csv(ranking_csv, index=False)

        print(f"[done] target={tgt}, ranking saved → {ranking_csv}")
        if args.write_all_pairs:
            print(f"[done] all pairs saved → {all_pairs_csv}")


if __name__ == "__main__":
    main()
