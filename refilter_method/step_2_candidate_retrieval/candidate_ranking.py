"""
Rank candidate lists for target apps using embedding similarity.

Given:
  - A Parquet file with app embeddings.
  - A list of target app IDs.

This script:
  1. Loads all embeddings from the Parquet file.
  2. L2 normalizes the embedding vectors.
  3. For each target app ID:
       - Computes cosine similarity between the target and all apps.
       - Ranks all apps by similarity (descending).
       - Writes a CSV file <target_app_id>.csv with columns:
           appId, similarity, rank

Typical usage:

  python build_candidate_lists_from_embeddings.py \
      --embeddings_parquet data/linq_embeddings_high_downloads.parquet \
      --output_dir data/candidates_linq \
      --targets_file config/eval_targets.txt

Requirements:
  - Python packages: numpy, pandas, pyarrow
  - The Parquet file must contain:
        - an ID column: "appId" or "app_id"
        - a column "embedding" with list-like embeddings
  - The targets file (if used) must contain one appId per line.
"""

import os
import ast
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------- Helpers ----------

def parse_emb(x) -> np.ndarray:
    """
    Convert an embedding value into a 1D float32 numpy array.

    Handles:
      - Python lists
      - Stringified lists (e.g., "[0.1, 0.2, ...]")
      - Other array-like values
    """
    if isinstance(x, list):
        arr = np.array(x, dtype=np.float32)
    elif isinstance(x, str):
        arr = np.array(ast.literal_eval(x), dtype=np.float32)
    else:
        arr = np.asarray(x, dtype=np.float32)
    return arr


def load_and_normalize(parquet_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load embeddings from a Parquet file and L2 normalize them.

    Expects:
      - ID column: "appId" or "app_id"
      - Embedding column: "embedding"

    Returns:
      - ids: list of app IDs as strings
      - X: numpy array of shape (N, D), row-normalized
    """
    df = pd.read_parquet(parquet_path, engine="pyarrow")

    # Find the ID column
    if "appId" in df.columns:
        id_col = "appId"
    elif "app_id" in df.columns:
        id_col = "app_id"
    else:
        raise ValueError(
            f"Could not find appId or app_id in columns: {df.columns.tolist()}"
        )

    if "embedding" not in df.columns:
        raise ValueError(f"'embedding' column missing in {parquet_path}")

    ids = df[id_col].astype(str).tolist()
    vecs = [parse_emb(x) for x in df["embedding"]]
    X = np.vstack(vecs).astype(np.float32)

    # L2 normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    X = X / norms
    return ids, X


def rank_for_target(target_id: str, ids: List[str], X: np.ndarray) -> pd.DataFrame:
    """
    For a single target appId, compute similarity to all apps.

    Returns a DataFrame with:
      - appId
      - similarity
      - rank (1 = most similar)

    If the target is not found, returns None.
    """
    try:
        idx = ids.index(target_id)
    except ValueError:
        return None

    q = X[idx]         # already normalized
    sims = X @ q       # cosine similarity via dot product

    out = pd.DataFrame({
        "appId": ids,
        "similarity": sims.astype(np.float32),
    }).sort_values("similarity", ascending=False, kind="mergesort")
    out["rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def load_targets_from_args(args: argparse.Namespace) -> List[str]:
    """
    Load target app IDs from either:
      - --target_ids (comma separated), or
      - --targets_file (one per line).

    Raises ValueError if neither is provided or if the file is empty.
    """
    # Highest priority: explicit comma separated list
    if args.target_ids:
        ids = [s.strip() for s in args.target_ids.split(",") if s.strip()]
        if not ids:
            raise ValueError("No valid target_ids found in --target_ids.")
        return ids

    # Fallback: file with one ID per line
    if args.targets_file:
        if not os.path.isfile(args.targets_file):
            raise FileNotFoundError(f"targets_file not found: {args.targets_file}")
        ids: List[str] = []
        with open(args.targets_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                ids.append(line)
        if not ids:
            raise ValueError("No valid target appIds found in targets_file.")
        return ids

    raise ValueError("You must provide either --targets_file or --target_ids.")


# ---------- Main ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build candidate lists from embeddings using cosine similarity."
    )
    parser.add_argument(
        "--embeddings_parquet",
        type=str,
        required=True,
        help="Path to Parquet file with embeddings (columns: appId/app_id, embedding).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where per target CSV files will be written.",
    )
    parser.add_argument(
        "--targets_file",
        type=str,
        default=None,
        help="Path to a text file with one target appId per line.",
    )
    parser.add_argument(
        "--target_ids",
        type=str,
        default=None,
        help="Comma separated list of target appIds (overrides --targets_file).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load embeddings
    print(f"Loading embeddings from: {args.embeddings_parquet}")
    ids, X = load_and_normalize(args.embeddings_parquet)
    id_set = set(ids)

    # Load target IDs
    targets = load_targets_from_args(args)
    print(f"Number of target apps: {len(targets)}")

    # Check which targets are missing
    missing = [t for t in targets if t not in id_set]
    if missing:
        print(f"[WARN] {len(missing)} target appIds not found in embeddings file.")
        print(f"First few missing: {missing[:3]}")

    # Process each target
    for app_id in targets:
        df_rank = rank_for_target(app_id, ids, X)
        if df_rank is None:
            # Write a small note so the user notices the missing ID
            note_path = os.path.join(args.output_dir, f"{app_id}__NOT_FOUND.txt")
            with open(note_path, "w", encoding="utf-8") as f:
                f.write(f"{app_id} not found in embeddings dataset.\n")
            print(f"[WARN] {app_id} not found, wrote {note_path}")
            continue

        out_path = os.path.join(args.output_dir, f"{app_id}.csv")
        df_rank.to_csv(out_path, index=False)
        print(f"[OK] wrote {out_path} (rows={len(df_rank)})")


if __name__ == "__main__":
    main()

