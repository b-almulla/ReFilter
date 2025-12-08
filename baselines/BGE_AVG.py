import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch


# === Custom splitter ===
def split_text_with_overlap(text, chunk_size=2000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input .jsonl file")
    parser.add_argument("--output", type=str, required=True, help="Path to output .csv file")
    parser.add_argument("--chunk_size", type=int, default=2000, help="Character length of each chunk")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Character overlap between chunks")
    parser.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5", help="SentenceTransformer model name")

    args = parser.parse_args()

    model = SentenceTransformer(args.model)
    data = []

    with open(args.input, "r") as f:
        for line in tqdm(f, desc="Processing apps"):
            try:
                doc = json.loads(line.strip())
                app_id = doc.get("App ID") or doc.get("appId")
                genre_id = doc.get("Category") or doc.get("genreId")
                desc = doc.get("Description") or doc.get("description", "")

                if not desc.strip():
                    continue

                chunks = split_text_with_overlap(desc, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
                embeddings = model.encode(chunks, device="cuda" if torch.cuda.is_available() else "cpu")
                avg_embedding = np.mean(embeddings, axis=0)

                data.append({
                    "appId": app_id,
                    "genreId": genre_id,
                    "num_chunks": len(chunks),
                    "embedding": avg_embedding.tolist()
                })
            except Exception as e:
                print(f"[ERROR] {doc.get('App ID', 'UNKNOWN')}: {e}")

    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    print(f"✅ Saved {len(df)} apps to: {args.output}")
