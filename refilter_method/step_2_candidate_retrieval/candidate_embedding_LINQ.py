"""
ReFilter App Embedding Script (Linq-Embed-Mistral)

This script generates dense vector embeddings for app descriptions
using the Linq-AI-Research/Linq-Embed-Mistral model. These embeddings
are used in the Candidate Retrieval stage of ReFilter to identify
semantically related apps before LLM-based filtering.

The script performs the following steps:

1. Reads app records from a JSON/JSONL file using a robust streaming
   loader that tolerates multi-line descriptions.

2. Extracts the cleaned description text for embedding
   (typically produced by the preprocessing pipeline).

3. Tokenizes each description and computes embeddings using:
   - Linq-Embed-Mistral
   - Last-token pooling
   - L2 normalization for stable cosine similarity comparison

4. Writes embeddings, along with app IDs and app names, into a
   compressed Parquet file using a reproducible schema:
   {
       "app_id": string,
       "app_name": string,
       "embedding": list<float32>
   }

5. Produces a `manifest.json` file containing run metadata such as:
   - model name
   - pooling strategy
   - embedding count
   - skipped records
   - output file paths

Usage Example:
    python embed_apps_linq.py \
        --input final_preprocessed.jsonl \
        --output_dir embeddings_linq \
        --outfile linq_embeddings.parquet \
        --use_gpu \
        --batch_write 2000

Arguments:
    --input        Path to preprocessed JSONL input.
    --output_dir   Directory for Parquet + manifest outputs.
    --outfile      Parquet filename.
    --max_len      Maximum token length for truncation.
    --use_gpu      Enable CUDA if available.
    --limit        Embed only the first N records (0 = no limit).

This script is designed for scalability (tens of thousands of apps),
reproducibility, and compatibility with downstream similarity search
and evaluation components of ReFilter.
"""


# === embed_apps_linq.py (LINQ + last-token pooling; clean Parquet) ===
import os
import json
import argparse
from typing import Dict, Any, Optional, Iterator

import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MODEL_NAME = "Linq-AI-Research/Linq-Embed-Mistral"


# ---------------------------- Utilities ----------------------------

def l2_normalize(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=dim)

def get_first(d: Dict[str, Any], *keys, default: Optional[str] = None) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

#This reader won’t die on multi-line descriptions.
def robust_json_stream(path: str) -> Iterator[Dict[str, Any]]:
    """
    Accepts:
      - JSONL (one object per line), and
      - multi-line JSON objects (e.g., descriptions with newlines).
    Accumulates lines until json.loads succeeds.
    """
    buf = []
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
    if buf:
        try:
            yield json.loads("\n".join(buf))
        except json.JSONDecodeError:
            pass

def get_text_for_embedding(rec: Dict[str, Any]) -> str:
    desc = get_first(rec, "Description", "description", default="") or ""
#    name = get_first(rec, "App Name", "appName", default="") or ""
#    text = desc.strip() if desc.strip() else name.strip()
    text = desc.strip()
    return " ".join(text.split())

#tokens: first token represents itself, the other tokens represent themselveself and the one behind them. So we take the last one 
def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Return representation of the last *non-pad* token for each sequence.
    Works regardless of left/right padding as long as attention_mask marks real tokens.
    last_hidden_states: (B, T, H)
    attention_mask:     (B, T) with 1 for real tokens, 0 for pads
    """
    # Index of the last non-pad token: sum(mask) - 1  (zero-based)
    idx = attention_mask.sum(dim=1) - 1
    b = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
    return last_hidden_states[b, idx]  # (B, H)

def embed_single_text(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_len: int,
) -> np.ndarray:
    """
    Returns a 1-D float32 numpy array:
    - tokenize with truncation
    - last-token pooling
    - L2 normalize
    - SQUEEZE to 1-D
    """
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        emb = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
        emb = l2_normalize(emb, dim=1)  # (B, H) -> unit length
    vec = emb.squeeze(0).detach().cpu().numpy().astype(np.float32)  # (H,)
    return vec


# ---------------------------- Main ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to JSON/JSONL file with app records.")
    parser.add_argument("--output_dir", type=str, default="output_linq",
                        help="Directory to write outputs.")
    parser.add_argument("--outfile", type=str, default="linq_embeddings.parquet",
                        help="Parquet file name.")
    parser.add_argument("--batch_write", type=int, default=2000,
                        help="Write to Parquet every N apps.")
    parser.add_argument("--max_len", type=int, default=4096,
                        help="Max token length.")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU if available.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Embed only first N records (0 = no limit).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.outfile)

    # Device
    device = torch.device(
        "cuda" if (args.use_gpu and torch.cuda.is_available())
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # (Optional) ensure right-padding for nicer mental model; works either way
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "right"

    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
    model.eval()
    torch.set_grad_enabled(False)
    print(f"[INFO] Loaded {MODEL_NAME} on device: {device}")

    # Prepare Parquet writer with explicit schema (list<float32>)
    schema = pa.schema([
        pa.field("app_id", pa.string()),
        pa.field("app_name", pa.string()),
        pa.field("embedding", pa.list_(pa.float32())),
    ])
    writer = pq.ParquetWriter(out_path, schema=schema, compression="zstd")

    # Buffers
    buf_app_id, buf_app_name, buf_emb = [], [], []
    seen = good = no_id = empty_text = 0

    pbar = tqdm(robust_json_stream(args.input), desc="Embedding apps")
    for rec in pbar:
        seen += 1
        app_id = get_first(rec, "App ID", "AppID", "app_id", "appId")
        if not app_id:
            no_id += 1
            continue

        text = get_text_for_embedding(rec)
        if not text:
            empty_text += 1
            continue

        app_name = get_first(rec, "App Name", "appName", default="") or ""
        vec = embed_single_text(text, tokenizer, model, device, args.max_len)

        # --- CRITICAL: store as a Python list[float] of shape (H,) ---
        emb_list = vec.tolist()  # 1-D, float32
        buf_app_id.append(str(app_id))
        buf_app_name.append(str(app_name))
        buf_emb.append(emb_list)
        good += 1

        # Periodic flush
        if good % args.batch_write == 0:
            table = pa.Table.from_pydict({
                "app_id":   pa.array(buf_app_id,   type=pa.string()),
                "app_name": pa.array(buf_app_name, type=pa.string()),
                "embedding": pa.array(buf_emb,     type=pa.list_(pa.float32())),
            }, schema=schema)
            writer.write_table(table)
            buf_app_id, buf_app_name, buf_emb = [], [], []
            pbar.set_postfix({"written": good})

        if args.limit and good >= args.limit:
            break

    # Flush remainder
    if buf_app_id:
        table = pa.Table.from_pydict({
            "app_id":   pa.array(buf_app_id,   type=pa.string()),
            "app_name": pa.array(buf_app_name, type=pa.string()),
            "embedding": pa.array(buf_emb,     type=pa.list_(pa.float32())),
        }, schema=schema)
        writer.write_table(table)

    writer.close()

    # Manifest
    manifest = {
        "model_name": MODEL_NAME,
        "pooling": "last_token",
        "normalized": True,
        "max_len": args.max_len,
        "parquet_path": out_path,
        "total_seen": seen,
        "total_embedded": good,
        "skipped_no_id": no_id,
        "skipped_empty_text": empty_text,
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("==== SUMMARY ====")
    for k, v in manifest.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
