#!/usr/bin/env python3
"""
ReFilter Extended Preprocessing

This script applies additional preprocessing to the baseline-cleaned
app descriptions described in Wei et al. (ASE 2024), including:

- HTML unescape and tag removal
- Lowercasing of text
- Whitespace normalization
- Filtering apps by minimum description length

Usage:
    python postprocess_baseline1.py \
        --input baseline_preprocessed.jsonl \
        --out_long final_long.jsonl \
        --out_short final_short.jsonl \
        --threshold 200
We exclude apps with short descriptions from embedding
"""

import json
import re
import argparse
from html import unescape

HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"[\t ]+")
MULTINEWLINE_RE = re.compile(r"\n{2,}")


def strip_html(text: str) -> str:
    text = unescape(text)
    return HTML_TAG_RE.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "")
    text = MULTISPACE_RE.sub(" ", text)
    text = MULTINEWLINE_RE.sub("\n", text)
    return text.strip()


def clean_desc(desc):
    if not isinstance(desc, str) or not desc.strip():
        return ""
    desc = strip_html(desc)
    desc = desc.lower()
    desc = normalize_whitespace(desc)
    return desc


def detect_desc_key(app):
    if "Description" in app and isinstance(app["Description"], str):
        return "Description"
    if "description" in app and isinstance(app["description"], str):
        return "description"
    return None


def process_file(input_path, out_long, out_short, threshold):
    total = kept_long = kept_short = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(out_long, "w", encoding="utf-8") as fh_long, \
         open(out_short, "w", encoding="utf-8") as fh_short:

        for line in infile:
            if not line.strip():
                continue

            total += 1
            try:
                app = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = detect_desc_key(app)
            cleaned = clean_desc(app.get(key, ""))

            app["Description"] = cleaned

            if len(cleaned) >= threshold:
                fh_long.write(json.dumps(app, ensure_ascii=False) + "\n")
                kept_long += 1
            else:
                fh_short.write(json.dumps(app, ensure_ascii=False) + "\n")
                kept_short += 1

    print("=== Post-processing Complete ===")
    print(f"Input file: {input_path}")
    print(f"Total apps processed: {total}")
    print(f"Apps >= {threshold} chars: {kept_long} → {out_long}")
    print(f"Apps <  {threshold} chars: {kept_short} → {out_short}")


def main():
    parser = argparse.ArgumentParser(description="ReFilter extended preprocessing")
    parser.add_argument("--input", required=True, help="Path to baseline-preprocessed JSONL file")
    parser.add_argument("--out_long", required=True, help="Output file for apps with long descriptions")
    parser.add_argument("--out_short", required=True, help="Output file for apps with short descriptions")
    parser.add_argument("--threshold", type=int, default=200, help="Minimum description length")

    args = parser.parse_args()

    process_file(args.input, args.out_long, args.out_short, args.threshold)


if __name__ == "__main__":
    main()
