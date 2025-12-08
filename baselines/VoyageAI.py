#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voyage Multimodal v3 — parallel pipeline (JSONL-only during run; Parquet at end)

Features
- Streams an input JSONL of apps and sends multiple Voyage requests concurrently.
- Resumes safely by skipping app_ids already present in an output JSONL.
- Writes progress + ETA to a TXT log (no console prints during the run).
- Optionally builds a single Parquet file at the end from the JSONL.

Inputs (provided via command-line arguments)
- --input_jsonl       : Path to the input apps JSONL file.
- --output_jsonl      : Path to the output embeddings JSONL file.
- --output_parquet    : Path to the final Parquet file (used if BUILD_PARQUET_AT_END is True).
- --progress_log_txt  : Path to the TXT log file for progress and ETA.

Requirements
- pip install voyageai pillow requests pandas pyarrow
- tqdm is only needed if you add prescan progress bars yourself.

Environment
- export VOYAGE_API_KEY='...'
"""

import os
import io
import json
import time
import math
import random
import requests
import threading
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from datetime import datetime

from PIL import Image
import voyageai
import pandas as pd

# ===================== Config =====================
# NOTE: these path variables are set at runtime from CLI args in main()
INPUT_JSONL: str = ""
OUTPUT_JSONL: str = ""
OUTPUT_PARQUET: str = ""
PROGRESS_LOG_TXT: str = ""

MODEL_NAME             = "voyage-multimodal-3"
MAX_APPS               = None         # None = all
MAX_SCREENSHOTS        = 50           # cap screenshots per app
BATCH_SIZE             = 4            # apps per Voyage request
CONCURRENT_REQUESTS    = 6            # number of parallel Voyage requests
THREADS_FOR_IMAGES     = 4            # per-batch image download threads
IMG_DOWNLOAD_TIMEOUT   = 20           # seconds per image
RETRY_MAX              = 5            # retries per Voyage call
RETRY_BASE_DELAY       = 2.0          # exponential backoff base
LOG_EVERY_SECONDS      = 90           # how often to append a progress line
BUILD_PARQUET_AT_END   = True         # convert JSONL -> Parquet when done
PARQUET_COMPRESSION    = "snappy"     # 'snappy' | 'zstd' | 'gzip' etc.
# ==================================================

# --------------- Globals / Locks ------------------
write_lock = threading.Lock()   # serialize writes to OUTPUT_JSONL + PROGRESS_LOG_TXT
stats_lock = threading.Lock()   # serialize stats updates

# rolling stats
stats = {
    "start_time": None,
    "last_log_time": 0.0,
    "processed": 0,
    "skipped": 0,
    "failed_batches": 0,
    "apps_in_input": 0,
    "total_to_do": 0,           # apps_in_input - already_done
    "total_seen": 0,            # how many apps scanned from input
    "sum_app_seconds": 0.0,     # total time / app (for average)
}

# ---------------- Arg parser ----------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Voyage Multimodal v3 parallel embedding pipeline (JSONL → JSONL [+ Parquet])."
    )
    parser.add_argument(
        "--input_jsonl",
        required=True,
        help="Path to input JSONL with app metadata (must include 'App ID', 'Description', 'Image links').",
    )
    parser.add_argument(
        "--output_jsonl",
        required=True,
        help="Path to output JSONL file where embeddings will be appended.",
    )
    parser.add_argument(
        "--output_parquet",
        required=True,
        help="Path to the final Parquet file (used if BUILD_PARQUET_AT_END is True).",
    )
    parser.add_argument(
        "--progress_log_txt",
        required=True,
        help="Path to a TXT log file for progress, ETA, and warnings.",
    )
    return parser.parse_args()

# ---------------- Utilities -----------------------

def append_log(line: str):
    """Append a single line to the progress log (timestamped)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{ts}] {line}\n"
    with write_lock:
        with open(PROGRESS_LOG_TXT, "a", encoding="utf-8") as f:
            f.write(msg)

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def prescan_count(path: str, max_apps: Optional[int]) -> int:
    """Fast pass to count total apps (for ETA). Respects MAX_APPS."""
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            total += 1
            if max_apps and total >= max_apps:
                break
    return total

def load_done_ids_from_jsonl(path: str) -> set:
    """Collect app_ids already written (resume)."""
    done = set()
    p = Path(path)
    if not p.exists():
        return done
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                aid = obj.get("app_id")
                if aid:
                    done.add(aid)
            except Exception:
                continue
    return done

def pick_screenshots(app: Dict[str, Any], max_n: int) -> List[str]:
    screenshots = app.get("Image links", []) or []
    if not isinstance(screenshots, list):
        return []
    return screenshots[:max_n]

def download_image(url: str, timeout: int = IMG_DOWNLOAD_TIMEOUT) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        return img
    except Exception:
        return None

def build_inputs_for_batch(apps: List[Dict[str, Any]]) -> List[List[object]]:
    """
    Each app becomes: [description, img1, img2, ...]
    """
    to_fetch: List[Tuple[int, str]] = []  # (index_in_batch, url)
    descs: List[str] = []
    per_app_urls: List[List[str]] = []

    for app in apps:
        desc = app.get("Description") or ""
        descs.append(desc)
        urls = pick_screenshots(app, MAX_SCREENSHOTS)
        per_app_urls.append(urls)
        idx = len(descs) - 1
        for u in urls:
            to_fetch.append((idx, u))

    results: Dict[Tuple[int, str], Optional[Image.Image]] = {}
    if to_fetch:
        max_workers = min(THREADS_FOR_IMAGES, max(1, len(to_fetch)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut2key = {ex.submit(download_image, u): (i, u) for i, u in to_fetch}
            for fut in as_completed(fut2key):
                i, url = fut2key[fut]
                img = None
                try:
                    img = fut.result()
                except Exception:
                    img = None
                results[(i, url)] = img

    inputs: List[List[object]] = []
    for i, desc in enumerate(descs):
        parts: List[object] = [desc]
        for url in per_app_urls[i]:
            img = results.get((i, url))
            if img is not None:
                parts.append(img)
        inputs.append(parts)
    return inputs

def call_voyage_multimodal(api_key: str, inputs: List[List[object]]) -> List[List[float]]:
    """
    Thread-safe approach: new client per call. Retries with backoff + jitter.
    """
    attempt = 0
    while True:
        try:
            client = voyageai.Client(api_key=api_key)
            resp = client.multimodal_embed(
                inputs=inputs,
                model=MODEL_NAME,
                truncation=True,
                input_type="document",
            )
            return resp.embeddings
        except Exception as e:
            attempt += 1
            if attempt > RETRY_MAX:
                raise
            sleep_s = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            sleep_s *= (0.75 + 0.5 * random.random())  # jitter
            append_log(f"WARN Voyage call failed (attempt {attempt}/{RETRY_MAX}): {e}. Retry in {sleep_s:.1f}s")
            time.sleep(sleep_s)

def write_jsonl_rows(path: str, rows: List[Dict[str, Any]]):
    with write_lock:
        with open(path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

def update_stats_and_maybe_log(now: float, force: bool = False):
    with stats_lock:
        if stats["start_time"] is None:
            stats["start_time"] = now

        elapsed = now - stats["start_time"]
        processed = stats["processed"]
        skipped = stats["skipped"]
        failed = stats["failed_batches"]
        total_to_do = stats["total_to_do"]

        avg_s = (stats["sum_app_seconds"] / processed) if processed else float("nan")
        remaining = max(total_to_do - processed, 0)
        eta_s = (avg_s * remaining) if (processed > 0 and math.isfinite(avg_s)) else float("nan")

        should_log = force or (now - stats["last_log_time"] >= LOG_EVERY_SECONDS)
        if should_log:
            stats["last_log_time"] = now
            human_avg = f"{avg_s:.2f}s/app" if math.isfinite(avg_s) else "n/a"
            human_eta = (
                f"{int(eta_s // 3600)}h {int((eta_s % 3600) // 60)}m {int(eta_s % 60)}s"
                if math.isfinite(eta_s) else "n/a"
            )
            append_log(
                f"PROGRESS processed={processed} skipped={skipped} failed_batches={failed} "
                f"avg={human_avg} elapsed={int(elapsed // 60)}m{int(elapsed % 60)}s "
                f"remaining={remaining} ETA={human_eta}"
            )

# --------------- Worker Task ----------------------

def process_batch(api_key: str, batch_objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build inputs, call Voyage, return rows to write.
    Also returns elapsed per-app timing for stats.
    """
    t0 = time.time()
    inputs = build_inputs_for_batch(batch_objs)
    embeddings = call_voyage_multimodal(api_key, inputs)
    t1 = time.time()

    per_app_time = (t1 - t0) / max(1, len(batch_objs))

    rows = []
    for app_obj, emb in zip(batch_objs, embeddings):
        rows.append({
            "app_id": app_obj.get("App ID"),
            "embedding": emb,
            "meta": {
                "model": MODEL_NAME,
                "num_images_used": min(MAX_SCREENSHOTS, len(app_obj.get("Image links", []) or [])),
                "desc_len_chars": len((app_obj.get("Description") or "")),
            }
        })
    return rows, per_app_time

# ------------------- Main -------------------------

def main():
    global INPUT_JSONL, OUTPUT_JSONL, OUTPUT_PARQUET, PROGRESS_LOG_TXT

    args = parse_args()
    INPUT_JSONL = args.input_jsonl
    OUTPUT_JSONL = args.output_jsonl
    OUTPUT_PARQUET = args.output_parquet
    PROGRESS_LOG_TXT = args.progress_log_txt

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY is not set. export VOYAGE_API_KEY='your_key'")

    Path(OUTPUT_JSONL).parent.mkdir(parents=True, exist_ok=True)
    Path(PROGRESS_LOG_TXT).parent.mkdir(parents=True, exist_ok=True)

    total_in_input = prescan_count(INPUT_JSONL, MAX_APPS)
    done_ids = load_done_ids_from_jsonl(OUTPUT_JSONL)

    total_to_do = total_in_input - len(done_ids)
    if total_to_do < 0:
        total_to_do = 0

    with stats_lock:
        stats["apps_in_input"] = total_in_input
        stats["total_to_do"] = total_to_do
        stats["start_time"] = time.time()
        stats["last_log_time"] = 0.0

    append_log(
        f"START total_in_input={total_in_input} already_done={len(done_ids)} total_to_do={total_to_do} "
        f"batch_size={BATCH_SIZE} concurrent_requests={CONCURRENT_REQUESTS} max_screenshots={MAX_SCREENSHOTS}"
    )

    executor = ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS)
    futures: List[Future] = []
    batch_buffer: List[Dict[str, Any]] = []

    scanned = 0
    try:
        with open(INPUT_JSONL, "r", encoding="utf-8") as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                aid = obj.get("App ID")
                if not aid:
                    continue

                scanned += 1
                if MAX_APPS and scanned > MAX_APPS:
                    break

                if aid in done_ids:
                    with stats_lock:
                        stats["skipped"] += 1
                    update_stats_and_maybe_log(time.time())
                    continue

                batch_buffer.append(obj)
                if len(batch_buffer) >= BATCH_SIZE:
                    batch = batch_buffer
                    batch_buffer = []
                    fut = executor.submit(process_batch, api_key, batch)
                    futures.append(fut)

                    done_list = [ft for ft in futures if ft.done()]
                    for ft in done_list:
                        futures.remove(ft)
                        try:
                            rows, per_app_time = ft.result()
                        except Exception as e:
                            append_log(f"WARN batch failed permanently: {e}")
                            with stats_lock:
                                stats["failed_batches"] += 1
                            continue

                        write_jsonl_rows(OUTPUT_JSONL, rows)
                        with stats_lock:
                            stats["processed"] += len(rows)
                            stats["sum_app_seconds"] += per_app_time * len(rows)
                            for r in rows:
                                done_ids.add(r["app_id"])
                        update_stats_and_maybe_log(time.time())

        if batch_buffer:
            fut = executor.submit(process_batch, api_key, batch_buffer)
            futures.append(fut)

        for ft in as_completed(futures):
            try:
                rows, per_app_time = ft.result()
            except Exception as e:
                append_log(f"WARN batch failed permanently: {e}")
                with stats_lock:
                    stats["failed_batches"] += 1
                continue

            write_jsonl_rows(OUTPUT_JSONL, rows)
            with stats_lock:
                stats["processed"] += len(rows)
                stats["sum_app_seconds"] += per_app_time * len(rows)
                for r in rows:
                    done_ids.add(r["app_id"])
            update_stats_and_maybe_log(time.time())

    except KeyboardInterrupt:
        append_log("INTERRUPTED by user, safely stopping...")

    finally:
        executor.shutdown(wait=False)

    update_stats_and_maybe_log(time.time(), force=True)
    append_log(
        f"DONE processed={stats['processed']} skipped={stats['skipped']} "
        f"failed_batches={stats['failed_batches']} -> {OUTPUT_JSONL}"
    )

    if BUILD_PARQUET_AT_END:
        t0 = time.time()
        append_log("PARQUET build started...")
        rows = []
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                rows.append(obj)
        df = pd.DataFrame(rows)
        if len(df):
            df.to_parquet(OUTPUT_PARQUET, index=False, compression=PARQUET_COMPRESSION)
            append_log(
                f"PARQUET built -> {OUTPUT_PARQUET} rows={len(df)} time={time.time()-t0:.1f}s"
            )
        else:
            append_log("PARQUET skipped: JSONL contained 0 rows")

if __name__ == "__main__":
    main()
