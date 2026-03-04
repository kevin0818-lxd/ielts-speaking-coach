import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an == 0.0 or bn == 0.0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def _iter_lemma_tokens(doc) -> Iterable[str]:
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        t = token.lemma_.lower().strip()
        if not t:
            continue
        if any(c.isalpha() for c in t):
            yield t


@dataclass(frozen=True)
class BaseFitResult:
    fit: float
    boost: float
    matched_marker_count: int
    top_markers: List[str]
    debug: Dict[str, Any]


class BaseCorpusCalibrator:
    def __init__(self, stats_path: Optional[str] = None):
        self.stats_path = stats_path or os.environ.get("BASE_STATS_PATH")
        self._stats: Optional[Dict[str, Any]] = None
        self._bigram_counts: Dict[str, int] = {}
        self._total_bigrams: int = 0
        self._avg_neg_logprob: Optional[float] = None
        self._std_neg_logprob: Optional[float] = None
        self._markers: List[str] = []
        self._centroid: Optional[np.ndarray] = None

        if self.stats_path and os.path.exists(self.stats_path):
            self._load(self.stats_path)

    def is_available(self) -> bool:
        return self._stats is not None

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        bigrams = stats.get("bigram_counts", {})
        if isinstance(bigrams, dict):
            self._bigram_counts = {str(k): int(v) for k, v in bigrams.items() if int(v) > 0}
        self._total_bigrams = int(stats.get("total_bigrams", 0))

        self._avg_neg_logprob = stats.get("avg_bigram_neg_logprob")
        self._std_neg_logprob = stats.get("std_bigram_neg_logprob")
        if self._avg_neg_logprob is not None:
            self._avg_neg_logprob = float(self._avg_neg_logprob)
        if self._std_neg_logprob is not None:
            self._std_neg_logprob = float(self._std_neg_logprob)

        markers = stats.get("marker_phrases", [])
        if isinstance(markers, list):
            self._markers = [str(m).lower() for m in markers if str(m).strip()]

        centroid = stats.get("embedding_centroid")
        if isinstance(centroid, list) and centroid:
            try:
                self._centroid = np.array([float(x) for x in centroid], dtype=np.float32)
            except Exception:
                self._centroid = None

        self._stats = stats

    def score_doc(self, doc) -> Optional[BaseFitResult]:
        if not self.is_available():
            return None

        tokens = list(_iter_lemma_tokens(doc))
        if len(tokens) < 3:
            return BaseFitResult(
                fit=0.5,
                boost=0.0,
                matched_marker_count=0,
                top_markers=[],
                debug={"reason": "too_short"},
            )

        ngram_fit, ngram_debug = self._ngram_fit(tokens)
        embed_fit, embed_debug = self._embedding_fit(doc)

        fit_parts: List[Tuple[float, float]] = []
        debug: Dict[str, Any] = {"ngram": ngram_debug, "embedding": embed_debug}

        if ngram_fit is not None:
            fit_parts.append((ngram_fit, 0.65))
        if embed_fit is not None:
            fit_parts.append((embed_fit, 0.35))

        if not fit_parts:
            overall_fit = 0.5
        else:
            wsum = sum(w for _, w in fit_parts)
            overall_fit = sum(v * w for v, w in fit_parts) / max(1e-9, wsum)

        matched_marker_count, top_markers = self._marker_hits(doc.text)
        boost = self._fit_to_boost(overall_fit, matched_marker_count, len(tokens))

        debug.update(
            {
                "overall_fit": float(overall_fit),
                "marker_count": int(matched_marker_count),
                "token_count": int(len(tokens)),
            }
        )

        return BaseFitResult(
            fit=float(overall_fit),
            boost=float(boost),
            matched_marker_count=int(matched_marker_count),
            top_markers=top_markers,
            debug=debug,
        )

    def _ngram_fit(self, tokens: List[str]) -> Tuple[Optional[float], Dict[str, Any]]:
        if not self._bigram_counts or self._total_bigrams <= 0:
            return None, {"available": False}

        v = max(1, len(self._bigram_counts))
        denom = float(self._total_bigrams + v)
        neg_logprob_sum = 0.0
        count = 0

        for i in range(len(tokens) - 1):
            k = f"{tokens[i]} {tokens[i + 1]}"
            c = self._bigram_counts.get(k, 0)
            p = (c + 1.0) / denom
            neg_logprob_sum += -math.log(p)
            count += 1

        if count == 0:
            return None, {"available": True, "reason": "no_bigrams"}

        avg_neg_logprob = neg_logprob_sum / count

        if self._avg_neg_logprob is not None:
            std = self._std_neg_logprob if self._std_neg_logprob and self._std_neg_logprob > 1e-9 else 1.0
            z = (self._avg_neg_logprob - avg_neg_logprob) / std
            fit = _sigmoid(z)
            return float(fit), {
                "available": True,
                "avg_neg_logprob": float(avg_neg_logprob),
                "z": float(z),
                "method": "zscore_sigmoid",
            }

        fit = max(0.0, min(1.0, 1.0 - (avg_neg_logprob / 12.0)))
        return float(fit), {"available": True, "avg_neg_logprob": float(avg_neg_logprob), "method": "heuristic"}

    def _embedding_fit(self, doc) -> Tuple[Optional[float], Dict[str, Any]]:
        if self._centroid is None:
            return None, {"available": False}
        if not getattr(doc, "vector_norm", 0.0):
            return None, {"available": True, "reason": "no_vector"}
        v = np.array(doc.vector, dtype=np.float32)
        sim = _cosine(v, self._centroid)
        fit = max(0.0, min(1.0, (sim + 1.0) / 2.0))
        return float(fit), {"available": True, "cosine": float(sim)}

    def _marker_hits(self, text: str) -> Tuple[int, List[str]]:
        if not self._markers:
            return 0, []
        t = text.lower()
        hits = [m for m in self._markers if m in t]
        hits.sort(key=len, reverse=True)
        return len(hits), hits[:3]

    def _fit_to_boost(self, fit: float, marker_count: int, token_count: int) -> float:
        if token_count < 30:
            return 0.0
        marker_bonus = 0.0
        if marker_count >= 2:
            marker_bonus = 0.05
        elif marker_count == 1:
            marker_bonus = 0.02

        if fit >= 0.82:
            return min(0.5, 0.45 + marker_bonus)
        if fit >= 0.70:
            return min(0.35, 0.25 + marker_bonus)
        if fit >= 0.60:
            return min(0.25, 0.15 + marker_bonus)
        return 0.0


_base_calibrator_singleton: Optional[BaseCorpusCalibrator] = None


def get_base_calibrator() -> BaseCorpusCalibrator:
    global _base_calibrator_singleton
    if _base_calibrator_singleton is None:
        _base_calibrator_singleton = BaseCorpusCalibrator()
    return _base_calibrator_singleton

