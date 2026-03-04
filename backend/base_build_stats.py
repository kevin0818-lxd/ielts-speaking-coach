import argparse
import json
import math
import os
import re
from collections import Counter
from typing import Iterable, List, Optional


def _iter_text_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".txt", ".md")):
                yield os.path.join(dirpath, fn)


_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _neg_logprob_stats(tokens: List[str], bigram_counts: Counter) -> Optional[float]:
    if len(tokens) < 3:
        return None
    total_bigrams = sum(bigram_counts.values())
    v = max(1, len(bigram_counts))
    denom = float(total_bigrams + v)
    s = 0.0
    n = 0
    for i in range(len(tokens) - 1):
        k = f"{tokens[i]} {tokens[i + 1]}"
        c = bigram_counts.get(k, 0)
        p = (c + 1.0) / denom
        s += -math.log(p)
        n += 1
    if n == 0:
        return None
    return s / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--top_bigrams", type=int, default=200000)
    ap.add_argument("--marker_file", default="")
    args = ap.parse_args()

    bigram_counts: Counter = Counter()
    neg_logprobs: List[float] = []

    for p in _iter_text_files(args.input_dir):
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        tokens = _tokenize(text)
        if len(tokens) < 3:
            continue
        for i in range(len(tokens) - 1):
            bigram_counts[f"{tokens[i]} {tokens[i + 1]}"] += 1

    if args.top_bigrams and args.top_bigrams > 0:
        bigram_counts = Counter(dict(bigram_counts.most_common(args.top_bigrams)))

    for p in _iter_text_files(args.input_dir):
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        tokens = _tokenize(text)
        v = _neg_logprob_stats(tokens, bigram_counts)
        if v is not None:
            neg_logprobs.append(v)

    avg = float(sum(neg_logprobs) / len(neg_logprobs)) if neg_logprobs else None
    std = None
    if neg_logprobs and len(neg_logprobs) > 1:
        m = avg
        var = sum((x - m) ** 2 for x in neg_logprobs) / (len(neg_logprobs) - 1)
        std = float(math.sqrt(var))

    markers: List[str] = []
    if args.marker_file:
        with open(args.marker_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    markers.append(s.lower())

    out = {
        "name": "BASE (custom build)",
        "version": 1,
        "total_bigrams": int(sum(bigram_counts.values())),
        "avg_bigram_neg_logprob": avg,
        "std_bigram_neg_logprob": std,
        "marker_phrases": markers,
        "bigram_counts": dict(bigram_counts),
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

