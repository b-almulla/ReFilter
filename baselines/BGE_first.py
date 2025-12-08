# === embed_apps_bge_first.py ===
import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
from typing import Any, Dict, Optional, List

def split_text_with_overlap(text, chunk_size=2000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += max(1, chunk_size - chunk_overlap)
    return chunks

def robust_json_stream(path: str):
    """
    Accepts both JSONL (one object per line) and multi-line JSON objects.
    Accumulates lines until json.loads succeeds.
    """
    buf: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line and not buf:
                continue
            buf.append(line)
            try:
                yield json.loads("\n".join(buf))
                buf = []
            except json.JSONDecodeError:
                continue
    # tail
    if buf:
        try:
            yield json.loads("\n".join(buf))
        except json.JSONDecodeError:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON/JSONL file")
    parser.add_argument("--output_base", type=str, required=True, help="Base path for output files (no extension)")
    parser.add_argument("--chunk_size", type=int, default=2000, help="Character length of each chunk")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Character overlap between chunks")
    parser.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5", help="SentenceTransformer model name")
    args = parser.parse_args()

    # Ensure parent dir exists
    os.makedirs(os.path.dirname(args.output_base), exist_ok=True)

    model = SentenceTransformer(args.model)

    jsonl_path = args.output_base + ".jsonl"
    parquet_path = args.output_base + ".parquet"

    app_ids = []
    genre_ids = []
    num_chunks_list = []
    embedding_list = []

    seen = 0
    kept = 0
    skipped_no_text = 0
    skipped_no_id = 0
    errors = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_out:
        for doc in tqdm(robust_json_stream(args.input), desc="Processing apps"):
            seen += 1
            try:
                app_id = doc.get("App ID") or doc.get("appId")
                genre_id = doc.get("Category") or doc.get("genreId")
                desc = doc.get("Description") or doc.get("description", "")

                if not app_id:
                    skipped_no_id += 1
                    continue

                if not (isinstance(desc, str) and desc.strip()):
                    skipped_no_text += 1
                    continue

                chunks = split_text_with_overlap(desc, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
                first_chunk = chunks[0]

                embedding = model.encode(first_chunk, device=device)
                # Ensure list[float] for parquet/json
                emb_list = embedding.tolist()

                # JSONL line
                json_line = {
                    "appId": app_id,
                    "genreId": genre_id,
                    "num_chunks": len(chunks),
                    "embedding": emb_list
                }
                jsonl_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")

                # Collect for Parquet
                app_ids.append(app_id)
                genre_ids.append(genre_id)
                num_chunks_list.append(len(chunks))
                embedding_list.append(emb_list)

                kept += 1
            except Exception as e:
                errors += 1
                # Print minimal info (avoid massive logs)
                print(f"[ERROR] app_id={doc.get('App ID') or doc.get('appId')}: {e}")

    # Save Parquet
    df = pd.DataFrame({
        "appId": app_ids,
        "genreId": genre_ids,
        "num_chunks": num_chunks_list,
        "embedding": embedding_list
    })
    df.to_parquet(parquet_path, engine="pyarrow", compression="zstd", index=False)

    print(f"✅ Saved {kept} apps to:")
    print(f"    ➤ JSONL:   {jsonl_path}")
    print(f"    ➤ Parquet: {parquet_path}")
    print("---- Summary ----")
    print(f"Seen: {seen} | Kept: {kept} | No-ID: {skipped_no_id} | No-text: {skipped_no_text} | Errors: {errors}")
