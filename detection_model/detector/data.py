from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from detection_model.corefeat.schema import clean_hand

DEFAULT_BENCHMARK = (
    Path(__file__).resolve().parents[1] / "public_miner_benchmark.json.gz"
)


def _clean_chunk(hands: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [clean_hand(hand) for hand in hands if isinstance(hand, dict)]


def load_corpus(path: str | Path | None = None) -> List[Dict[str, Any]]:
    source = Path(path) if path else DEFAULT_BENCHMARK
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    records: List[Dict[str, Any]] = []
    for entry in payload.get("labeled_chunks", []):
        hands = _clean_chunk(entry.get("hands") or [])
        if not hands:
            continue
        records.append(
            {
                "hands": hands,
                "label": int(bool(entry.get("is_bot"))),
                "split": str(entry.get("split") or "train"),
                "chunk_id": str(entry.get("chunk_id") or ""),
            }
        )
    return records


def split_corpus(
    records: Sequence[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train = [r for r in records if r["split"] != "validation"]
    validation = [r for r in records if r["split"] == "validation"]
    return train, validation


def resample(
    train: Sequence[Dict[str, Any]],
    *,
    per_label: int,
    seed: int,
    size_range: Tuple[int, int] = (60, 120),
) -> List[Dict[str, Any]]:
    """Expand the labeled set by drawing fresh chunks from per-label hand pools.

    Each synthesized chunk keeps a single behavioural label but a new hand
    composition and size, which broadens the chunk-level statistics the model
    sees without inventing any out-of-distribution hands.
    """
    rng = random.Random(seed)
    pools: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}
    for record in train:
        pools[record["label"]].extend(record["hands"])
    out: List[Dict[str, Any]] = [dict(record) for record in train]
    for label in (0, 1):
        pool = pools[label]
        if not pool:
            continue
        for _ in range(per_label):
            size = rng.randint(size_range[0], size_range[1])
            hands = [rng.choice(pool) for _ in range(size)]
            out.append({"hands": hands, "label": label, "split": "synth"})
    rng.shuffle(out)
    return out


def chunk_lists(records: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    return [record["hands"] for record in records]


def labels(records: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([record["label"] for record in records], dtype=int)
