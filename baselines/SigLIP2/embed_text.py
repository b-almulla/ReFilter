#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, io, json, time, argparse
from typing import List, Dict, Any, Optional, Tuple

import torch
import pandas as pd
from transformers import AutoProcessor, AutoModel

def l2_normalize(t: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return t / (t.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))

def parse_args():
    ap = argparse.ArgumentParser("Embed app descriptions with SigLIP2")
    ap.add_argument("--input", type=str, required=True,
                    help="Path to JSONL containing 'App ID' and 'Description'")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory")
    ap.add_argument("--model_id", type=str, default="fancyfeast/so400m-long",
                    help="HF model ID or local dir with model snapshot")
    ap.add_argument("--local_files_only", action="store_true",
                    help="Set if model is already cached locally (offline runs)")
    ap.add_argument("--device", type=str, default="cuda",
                    help="cuda or cpu")
    ap.add_argument("--max_tokens", type=int, default=256,
                    help="Text encoder max tokens (256 for so400m-long).")
    ap.add_argument("--overlap", type=int, default=32,
                    help="Token overlap between chunks.")
    ap.add_argument("--batch_size", type=int, default=128,
                    help="Batch size for chunks -> embeddings.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after producing this many CHUNK embeddings (0 = no limit).")
    ap.add_argument("--flush_every_apps", type=int, default=2000,
                    help="Overwrite output files after processing this many apps.")
    return ap.parse_args()

def ensure_outdir(p: str): os.makedirs(p, exist_ok=True)

def atomic_write_parquet(df: pd.DataFrame, path: str):
    tmp = path + ".part"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)

def atomic_write_csv(df: pd.DataFrame, path: str):
    tmp = path + ".part"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def chunk_tokens(input_ids: List[int], max_len: int, overlap: int) -> List[List[int]]:
    if max_len <= 0: return [input_ids]
    if overlap < 0: overlap = 0
    chunks = []
    i = 0
    step = max_len - overlap if max_len > overlap else max_len
    while i < len(input_ids):
        chunk = input_ids[i:i+max_len]
        if not chunk: break
        chunks.append(chunk)
        if i + max_len >= len(input_ids): break
        i += step
    return chunks

def main():
    args = parse_args()
    ensure_outdir(args.outdir)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # Load model + processor
    processor = AutoProcessor.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(args.model_id, local_files_only=args.local_files_only).to(device).eval()

    # Outputs
    parquet_path = os.path.join(args.outdir, "text_embeddings.parquet")
    counts_csv   = os.path.join(args.outdir, "text_counts.csv")
    summary_txt  = os.path.join(args.outdir, "text_run_summary.txt")

    # Buffers (written every flush_every_apps)
    rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    fail_apps: List[Tuple[str, str]] = []  # (app_id, reason)

    # Progress stats
    t0 = time.time()
    total_apps = 0
    success_apps = 0
    total_chunks = 0

    # Read streaming
    def write_outputs(overwrite: bool = True):
        # Convert buffers to DataFrames and merge with any existing outputs (overwrite full files)
        # Parquet (embeddings)
        if rows:
            new_df = pd.DataFrame(rows)
        else:
            new_df = pd.DataFrame(columns=["app_id","chunk_idx","n_tokens","embedding"])

        if overwrite and os.path.exists(parquet_path):
            try:
                old_df = pd.read_parquet(parquet_path)
                merged = pd.concat([old_df, new_df], ignore_index=True)
                # Deduplicate on (app_id, chunk_idx)
                merged.drop_duplicates(subset=["app_id","chunk_idx"], inplace=True)
                atomic_write_parquet(merged, parquet_path)
            except Exception:
                # Fallback: write only new rows
                atomic_write_parquet(new_df, parquet_path)
        else:
            atomic_write_parquet(new_df, parquet_path)

        # CSV (counts)
        if counts:
            new_counts = pd.DataFrame([{"app_id": k, "n_chunks": v} for k, v in counts.items()])
        else:
            new_counts = pd.DataFrame(columns=["app_id","n_chunks"])

        if overwrite and os.path.exists(counts_csv):
            try:
                old_counts = pd.read_csv(counts_csv)
                merged_c = pd.concat([old_counts, new_counts], ignore_index=True)
                merged_c = merged_c.groupby("app_id", as_index=False)["n_chunks"].max()
                atomic_write_csv(merged_c, counts_csv)
            except Exception:
                atomic_write_csv(new_counts, counts_csv)
        else:
            atomic_write_csv(new_counts, counts_csv)

        # Summary (TXT) — append a snapshot line
        with open(summary_txt, "a", encoding="utf-8") as f:
            f.write(f"[snapshot] apps_done={total_apps} success_apps={success_apps} "
                    f"chunks={total_chunks} fails={len(fail_apps)} elapsed_sec={time.time()-t0:.1f}\n")

    # Process
    emitted = 0
    apps_since_flush = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            app_id = obj.get("App ID")
            desc = obj.get("Description") or ""

            if not app_id:
                continue

            total_apps += 1
            apps_since_flush += 1

            try:
                # Tokenize without truncation to get full ids
                toks = processor(text=[desc], padding=False, truncation=False, return_tensors=None)
                # processor returns dict with "input_ids" as list[list[int]]
                input_ids = toks["input_ids"][0]

                # Make chunks
                chunks = chunk_tokens(input_ids, max_len=args.max_tokens, overlap=args.overlap)

                # Now encode in batches
                chunk_idx = 0
                batch: List[List[int]] = []
                batch_lens: List[int] = []

                counts_for_app = 0

                def flush_batch(batch_ids: List[List[int]], batch_lens: List[int], app_id: str, start_idx: int):
                    nonlocal emitted, total_chunks, rows
                    if not batch_ids: return 0
                    # Convert token id batches back to text strings using tokenizer.decode
                    texts = [processor.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch_ids]
                    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_tokens)
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        feats = model.get_text_features(**inputs)  # [B, D]
                        feats = l2_normalize(feats, dim=-1).cpu()

                    # record rows
                    for j, (vec, n_tok) in enumerate(zip(feats, batch_lens)):
                        rows.append({
                            "app_id": app_id,
                            "chunk_idx": start_idx + j,
                            "n_tokens": n_tok,
                            "embedding": json.dumps(vec.tolist())
                        })
                        emitted += 1
                        total_chunks += 1
                        if args.limit and emitted >= args.limit:
                            return j + 1  # number consumed from this batch
                    return len(batch_ids)

                for ch in chunks:
                    batch.append(ch)
                    batch_lens.append(len(ch))
                    if len(batch) >= args.batch_size:
                        consumed = flush_batch(batch, batch_lens, app_id, chunk_idx)
                        chunk_idx += consumed
                        counts_for_app += consumed
                        batch, batch_lens = [], []
                        if args.limit and emitted >= args.limit: break

                if batch and (not args.limit or emitted < args.limit):
                    consumed = flush_batch(batch, batch_lens, app_id, chunk_idx)
                    chunk_idx += consumed
                    counts_for_app += consumed
                    batch, batch_lens = [], []

                counts[app_id] = counts_for_app
                if counts_for_app > 0:
                    success_apps += 1

            except Exception as e:
                fail_apps.append((app_id, f"{type(e).__name__}: {e}"))

            # Periodic overwrite write
            if args.flush_every_apps > 0 and apps_since_flush >= args.flush_every_apps:
                write_outputs(overwrite=True)
                rows.clear()
                counts.clear()
                apps_since_flush = 0

            if args.limit and emitted >= args.limit:
                break

    # Final write
    write_outputs(overwrite=True)

    # Final summary
    elapsed = time.time() - t0
    # avg per app: only for apps that produced at least one chunk
    avg_per_app = elapsed / max(1, success_apps)
    with open(summary_txt, "a", encoding="utf-8") as f:
        f.write("\n=== FINAL SUMMARY ===\n")
        f.write(f"total_apps_seen: {total_apps}\n")
        f.write(f"success_apps: {success_apps}\n")
        f.write(f"failed_apps: {len(fail_apps)}\n")
        f.write(f"total_text_embeddings: {total_chunks}\n")
        f.write(f"elapsed_seconds: {elapsed:.2f}\n")
        f.write(f"avg_seconds_per_embedded_app: {avg_per_app:.4f}\n")
        if fail_apps:
            f.write("failed_list:\n")
            for aid, reason in fail_apps:
                f.write(f"  - {aid}: {reason}\n")

if __name__ == "__main__":
    main()
