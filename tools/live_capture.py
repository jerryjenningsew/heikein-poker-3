"""Opt-in capture of live validator payloads to a local JSONL file.

Enable with P44_CAPTURE=1. Every scored batch is appended as one JSON
line to P44_CAPTURE_DIR (default: local live_capture/). Chunks are
content-deduplicated in-process. Capture is best-effort and never
affects serving.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

_SEEN: set[str] = set()
_MAX_SEEN = 50_000


def _capture_path() -> Path:
    base = os.getenv("P44_CAPTURE_DIR", "")
    root = Path(base) if base else Path(__file__).resolve().parents[1] / "live_capture"
    root.mkdir(parents=True, exist_ok=True)
    return root / "live_capture.jsonl"


def _chunk_key(chunk: Sequence[Dict[str, Any]]) -> str:
    raw = json.dumps(chunk, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def capture_batch(
    chunks: List[List[Dict[str, Any]]],
    raw_scores: Sequence[float] | None = None,
) -> int:
    """Append unseen chunks of one scored batch; returns how many were written."""
    if not chunks:
        return 0
    path = _capture_path()
    written = 0
    scores = list(raw_scores) if raw_scores is not None else [None] * len(chunks)
    with path.open("a", encoding="utf-8") as handle:
        for chunk, score in zip(chunks, scores):
            key = _chunk_key(chunk)
            if key in _SEEN:
                continue
            if len(_SEEN) >= _MAX_SEEN:
                _SEEN.clear()
            _SEEN.add(key)
            record = {
                "t": round(time.time(), 3),
                "n_hands": len(chunk),
                "score": None if score is None else float(score),
                "chunk": chunk,
            }
            handle.write(json.dumps(record, default=str) + "\n")
            written += 1
    return written
