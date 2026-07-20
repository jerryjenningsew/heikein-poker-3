from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np

from detection_model.corefeat.schema import clean_hand
from detection_model.corefeat.calibration import apply_mapper
from detection_model.chunkfeat.features import matrix_for_chunks
from detection_model.chunkfeat.mapping import chunk_tie_key, exact_rank_map

from .boundary import adaptive_rank_map
from .model import raw_scores


class Detector:
    def __init__(self, artifact: Dict[str, Any]) -> None:
        self.artifact = artifact
        self.model = artifact["model"]
        self.calibrator = artifact.get("calibrator")
        self.mapper = dict(artifact.get("mapper") or {"cut": 0.5, "scale": 0.1})
        self.batch_top_fraction = float(artifact.get("batch_top_fraction", 0.10))
        self.boundary = str(artifact.get("boundary", "exact"))
        self.use_temporal = bool(artifact.get("use_temporal", False))
        if not 0.0 < self.batch_top_fraction < 1.0:
            raise ValueError("batch_top_fraction must be between zero and one")

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: str | None = None) -> "Detector":
        artifact_path = Path(path).expanduser()
        if expected_sha256:
            expected = str(expected_sha256).strip().lower()
            digest = hashlib.sha256()
            with artifact_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            actual = digest.hexdigest()
            if len(expected) != 64 or not hmac.compare_digest(actual, expected):
                raise ValueError(
                    "artifact SHA-256 mismatch before deserialization: "
                    f"expected={expected} actual={actual}"
                )
        return cls(joblib.load(artifact_path))

    @staticmethod
    def _clean(chunks: Sequence[Sequence[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        return [
            [clean_hand(hand) for hand in (chunk or []) if isinstance(hand, dict)]
            for chunk in chunks
        ]

    def _matrix(self, clean: List[List[Dict[str, Any]]]) -> np.ndarray:
        matrix = matrix_for_chunks(clean)
        if self.use_temporal:
            from .features_ext import temporal_matrix

            matrix = np.hstack([matrix, temporal_matrix(clean)])
        return matrix

    def _raw(self, matrix: np.ndarray) -> np.ndarray:
        values = np.clip(raw_scores(self.model, matrix), 1e-6, 1.0 - 1e-6)
        if self.calibrator is not None:
            values = np.clip(self.calibrator.predict(values), 1e-6, 1.0 - 1e-6)
        return values

    def raw_scores(self, chunks: Sequence[Sequence[Dict[str, Any]]]) -> np.ndarray:
        if not chunks:
            return np.zeros(0, dtype=float)
        return self._raw(self._matrix(self._clean(chunks)))

    def predict_chunks(
        self,
        chunks: List[List[Dict[str, Any]]],
        *,
        batch_map: bool = True,
    ) -> List[float]:
        if not chunks:
            return []
        clean = self._clean(chunks)
        raw = self._raw(self._matrix(clean))
        if batch_map and len(clean) >= 8:
            keys = [chunk_tie_key(chunk) for chunk in clean]
            if self.boundary == "adaptive":
                scores = adaptive_rank_map(
                    raw, tie_keys=keys, default_fraction=self.batch_top_fraction
                )
            else:
                scores = exact_rank_map(raw, self.batch_top_fraction, tie_keys=keys)
        else:
            scores = apply_mapper(raw, self.mapper)
        scores = np.nan_to_num(scores, nan=0.5, posinf=0.99, neginf=0.01)
        return [round(float(np.clip(value, 0.01, 0.99)), 8) for value in scores]

    def predict_chunk(self, chunk: List[Dict[str, Any]]) -> float:
        return self.predict_chunks([chunk], batch_map=False)[0]
