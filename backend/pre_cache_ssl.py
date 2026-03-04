#!/usr/bin/env python3
"""
Pre-cache SSL (wav2vec2 / HuBERT) features for all audio in the training dataset.

Run once before training so IELTSDataset reads from disk cache instead of
running the SSL encoder live. Typical runtime: ~3-5s per sample on GPU,
~10-15s on CPU.

Usage:
    python backend/pre_cache_ssl.py
    python backend/pre_cache_ssl.py --data data/ielts_samples.filtered.jsonl
    python backend/pre_cache_ssl.py --model facebook/hubert-base-ls960

Environment overrides (same as train_model.py):
    DATA_PATH, SSL_MODEL_NAME, SSL_CACHE_DIR
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backend.features import extract_ssl_features_from_path
except ImportError:
    from features import extract_ssl_features_from_path


def main():
    parser = argparse.ArgumentParser(description="Pre-cache SSL features for training dataset")
    parser.add_argument("--data", default=os.getenv("DATA_PATH", "data/ielts_samples.jsonl"),
                        help="Path to JSONL dataset")
    parser.add_argument("--model", default=os.getenv("SSL_MODEL_NAME", "facebook/wav2vec2-base"),
                        help="HuggingFace model name")
    parser.add_argument("--cache-dir", default=os.getenv("SSL_CACHE_DIR", "data/ssl_features"),
                        help="Directory for cached .pt files")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if cache file exists")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: Dataset not found: {args.data}")
        sys.exit(1)

    os.makedirs(args.cache_dir, exist_ok=True)

    with open(args.data, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    print(f"Dataset: {args.data} ({len(rows)} samples)")
    print(f"SSL model: {args.model}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Force re-extract: {args.force}")
    print()

    cached = 0
    skipped = 0
    failed = 0
    missing_audio = 0
    t_start = time.time()

    for i, row in enumerate(rows):
        audio_path = row.get("audio_path", "")
        feature_id = os.path.basename(audio_path).split(".")[0]
        cache_path = os.path.join(args.cache_dir, f"{feature_id}_ssl.pt")

        if not args.force and os.path.exists(cache_path):
            skipped += 1
            continue

        if not os.path.exists(audio_path):
            missing_audio += 1
            continue

        t0 = time.time()
        feat = extract_ssl_features_from_path(audio_path, sr=16000, model_name=args.model)

        if feat is not None:
            torch.save(feat.squeeze(0).half(), cache_path)
            cached += 1
            dt = time.time() - t0
            if (cached) % 10 == 0 or cached == 1:
                print(f"  [{i+1}/{len(rows)}] Cached {feature_id} ({dt:.1f}s)")
        else:
            failed += 1
            print(f"  [{i+1}/{len(rows)}] FAILED {feature_id}")

    elapsed = time.time() - t_start
    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"  Cached:        {cached}")
    print(f"  Already cached: {skipped}")
    print(f"  Missing audio: {missing_audio}")
    print(f"  Failed:        {failed}")
    print(f"  Total:         {len(rows)}")


if __name__ == "__main__":
    main()
