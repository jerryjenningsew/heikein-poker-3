from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Sequence

import numpy as np

TEMPORAL_FEATURE_NAMES: List[str] = [
    "t_aggr_autocorr1",
    "t_passive_autocorr1",
    "t_actioncount_autocorr1",
    "t_aggr_cv",
    "t_actioncount_cv",
    "t_street_cv",
    "t_signature_repeat_rate",
    "t_signature_distinct_ratio",
    "t_signature_top2_share",
    "t_signature_entropy",
    "t_betsize_top3_share",
    "t_betsize_distinct_ratio",
    "t_betsize_entropy",
    "t_signature_run_max_share",
    "t_aggr_run_max_share",
]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    return math.sqrt(max(0.0, _mean([(v - m) ** 2 for v in values])))


def _entropy(items: Sequence[Any]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    total = float(sum(counts.values()))
    if total <= 0 or len(counts) <= 1:
        return 0.0
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log(p + 1e-12)
    return ent / math.log(len(counts))


def _max_run_share(items: Sequence[Any]) -> float:
    if not items:
        return 0.0
    longest = 1
    current = 1
    for previous, value in zip(items, items[1:]):
        if previous == value:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest / len(items)


def _autocorr1(series: Sequence[float]) -> float:
    if len(series) < 3:
        return 0.0
    m = _mean(series)
    numerator = sum((series[i] - m) * (series[i + 1] - m) for i in range(len(series) - 1))
    denominator = sum((v - m) ** 2 for v in series)
    return numerator / denominator if denominator else 0.0


def _chunk_vector(hands: Sequence[Dict[str, Any]]) -> List[float]:
    aggr_seq: List[float] = []
    passive_seq: List[float] = []
    count_seq: List[float] = []
    street_seq: List[float] = []
    signatures: List[tuple] = []
    amounts: List[float] = []

    for hand in hands:
        actions = hand.get("actions") or []
        types = [str((a or {}).get("action_type") or "").lower() for a in actions]
        counts = Counter(types)
        meaningful = max(
            counts.get("call", 0) + counts.get("check", 0) + counts.get("bet", 0)
            + counts.get("raise", 0) + counts.get("fold", 0),
            1,
        )
        aggr_seq.append((counts.get("bet", 0) + counts.get("raise", 0)) / meaningful)
        passive_seq.append((counts.get("call", 0) + counts.get("check", 0)) / meaningful)
        count_seq.append(float(len(actions)))
        street_seq.append(float(len({str((a or {}).get("street") or "") for a in actions})))
        signatures.append(tuple(types))
        for action in actions:
            try:
                value = float((action or {}).get("normalized_amount_bb") or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                amounts.append(round(value, 2))

    n = max(len(hands), 1)
    sig_counts = Counter(signatures)
    if amounts:
        amount_counts = Counter(amounts)
        betsize_top3 = sum(c for _, c in amount_counts.most_common(3)) / len(amounts)
        betsize_distinct = len(amount_counts) / len(amounts)
        betsize_entropy = _entropy(amounts)
    else:
        betsize_top3 = betsize_distinct = betsize_entropy = 0.0

    return [
        _autocorr1(aggr_seq),
        _autocorr1(passive_seq),
        _autocorr1(count_seq),
        _std(aggr_seq) / (_mean(aggr_seq) + 1e-6),
        _std(count_seq) / (_mean(count_seq) + 1e-6),
        _std(street_seq) / (_mean(street_seq) + 1e-6),
        sum(c for c in sig_counts.values() if c > 1) / n,
        len(sig_counts) / n,
        sum(c for _, c in sig_counts.most_common(2)) / n,
        _entropy(signatures),
        betsize_top3,
        betsize_distinct,
        betsize_entropy,
        _max_run_share(signatures),
        _max_run_share([round(v, 1) for v in aggr_seq]),
    ]


def temporal_matrix(chunks: Sequence[Sequence[Dict[str, Any]]]) -> np.ndarray:
    if not chunks:
        return np.zeros((0, len(TEMPORAL_FEATURE_NAMES)), dtype=float)
    rows = [_chunk_vector(list(chunk or [])) for chunk in chunks]
    return np.asarray(rows, dtype=float)
