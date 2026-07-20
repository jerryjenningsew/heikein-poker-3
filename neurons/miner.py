from __future__ import annotations

import hashlib
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Tuple

import bittensor as bt
from dotenv import load_dotenv

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

from detection_model.detector import model as model_spec
from detection_model.detector.inference import Detector

load_dotenv()

MODEL_REPO_PATH = "detection_model"


def _sha256_file(path: str | Path) -> str:
    path = Path(path).expanduser()
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _existing_paths(paths: list[str | Path]) -> list[Path]:
    resolved = [Path(value).expanduser() for value in paths]
    missing = [path for path in resolved if not path.exists() or not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing implementation files: " + ", ".join(str(p) for p in missing)
        )
    return resolved


def _git(args: list[str], repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(repo_root),
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _normalize_repo_url(url: str) -> str:
    url = url.strip()
    if not url:
        return ""
    if url.startswith("git@"):
        host, _, path = url[len("git@"):].partition(":")
        url = f"https://{host}/{path}"
    elif url.startswith("ssh://git@"):
        url = "https://" + url[len("ssh://git@"):]
    if url.endswith(".git"):
        url = url[: -len(".git")]
    return url


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("Detection miner started")

        repo_root = Path(__file__).resolve().parents[1]
        model_repo_root = Path(MODEL_REPO_PATH).expanduser()

        self.model_path = os.getenv(
            "P44_MODEL_PATH",
            f"detection_model/artifacts/{model_spec.MODEL_NAME}.joblib",
        )
        self.prediction_threshold = float(os.getenv("P44_PREDICTION_THRESHOLD", "0.5"))
        self.require_model = os.getenv("P44_REQUIRE_MODEL", "0").strip().lower() in {
            "1", "true", "yes", "on",
        }
        self.top_k = int(os.getenv("P44_TOP_K", "0"))
        self.top_k_frac = float(os.getenv("P44_TOP_K_FRAC", "0"))
        try:
            self.latency_warn_seconds = float(os.getenv("P44_LATENCY_WARN_SECONDS", "15"))
        except (TypeError, ValueError):
            self.latency_warn_seconds = 15.0

        self.detector = None
        self.model_manifest = self._build_model_manifest(repo_root, model_repo_root)
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup()
        self._load_trained_model()
        bt.logging.info(f"Axon created: {self.axon}")

    def _build_model_manifest(self, repo_root: Path, model_repo_root: Path) -> dict:
        model_artifact_path = Path(self.model_path).expanduser()
        implementation_paths = [
            Path(__file__).resolve(),
            model_repo_root / "detector" / "inference.py",
            model_repo_root / "detector" / "model.py",
            model_repo_root / "detector" / "train.py",
            model_repo_root / "detector" / "data.py",
            model_repo_root / "detector" / "boundary.py",
            model_repo_root / "chunkfeat" / "features.py",
            model_repo_root / "chunkfeat" / "mapping.py",
            model_repo_root / "corefeat" / "schema.py",
            model_repo_root / "corefeat" / "features.py",
            model_repo_root / "corefeat" / "calibration.py",
            repo_root / "poker44" / "validator" / "payload_view.py",
            repo_root / "poker44" / "score" / "scoring.py",
        ]
        extra = model_repo_root / "detector" / "features_ext.py"
        if extra.exists():
            implementation_paths.append(extra)
        implementation_files = _existing_paths(implementation_paths)
        artifact_sha256 = os.getenv(
            "POKER44_MODEL_ARTIFACT_SHA256", _sha256_file(model_artifact_path)
        )

        repo_url = (
            os.getenv("P44_MANIFEST_REPO_URL")
            or os.getenv("POKER44_MODEL_REPO_URL")
            or _normalize_repo_url(_git(["remote", "get-url", "origin"], repo_root))
        )
        repo_commit = (
            os.getenv("P44_MANIFEST_REPO_COMMIT")
            or os.getenv("POKER44_MODEL_REPO_COMMIT")
            or _git(["rev-parse", "HEAD"], repo_root)
        )

        return build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=implementation_files,
            defaults={
                "open_source": True,
                "model_name": model_spec.MODEL_NAME,
                "model_version": model_spec.MODEL_VERSION,
                "framework": model_spec.FRAMEWORK,
                "license": os.getenv("P44_MANIFEST_LICENSE", "MIT"),
                "repo_url": repo_url,
                "repo_commit": repo_commit,
                "artifact_url": os.getenv("P44_MANIFEST_ARTIFACT_URL", ""),
                "artifact_sha256": artifact_sha256,
                "model_card_url": os.getenv("P44_MANIFEST_MODEL_CARD_URL", ""),
                "training_data_statement": (
                    "Trained only on the public benchmark corpus projected to the "
                    "public miner-visible schema, expanded by per-label chunk "
                    "resampling. No validator-only evaluation data, hidden labels, "
                    "or private payloads were used."
                ),
                "training_data_sources": [
                    "public poker bot-detection benchmark chunks",
                    "public miner-visible projections of those payloads",
                ],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data, "
                    "live eval batches, hidden labels, or any private validator data."
                ),
                "data_attestation": (
                    "All training sources are public benchmark data. The published "
                    "commit identifies the serving and training source; the deployed "
                    "artifact is pinned by SHA-256."
                ),
                "inference_mode": "local",
                "notes": (
                    "Deterministic local inference using the pinned artifact and the "
                    "implementation files at the published repository commit."
                ),
            },
        )

    def _load_trained_model(self) -> None:
        model_path = Path(self.model_path).expanduser()
        if not model_path.exists():
            message = f"Model artifact not found: {model_path}"
            if self.require_model:
                raise FileNotFoundError(message)
            bt.logging.error(f"{message}. Using heuristic fallback.")
            return
        try:
            expected = os.getenv("POKER44_MODEL_ARTIFACT_SHA256", "").strip().lower()
            if expected:
                if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                    raise ValueError("POKER44_MODEL_ARTIFACT_SHA256 must be 64 hex characters")
                actual = _sha256_file(model_path).lower()
                if actual != expected:
                    raise ValueError(
                        f"artifact SHA-256 mismatch: expected={expected} actual={actual}"
                    )
            self.detector = Detector.load(model_path)
            env_threshold = os.getenv("P44_PREDICTION_THRESHOLD")
            if env_threshold is not None:
                self.prediction_threshold = float(env_threshold)
            bt.logging.info(
                f"detector loaded | features={self.detector.artifact.get('feature_dim')} "
                f"training={self.detector.artifact.get('training_count')} "
                f"boundary={self.detector.boundary}"
            )
        except Exception as exc:
            if self.require_model:
                raise RuntimeError(f"Required model failed to load: {exc}") from exc
            bt.logging.error(f"Failed to load model: {exc}. Using heuristic fallback.")
            self.detector = None

    def _log_manifest_startup(self) -> None:
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"digest={self.manifest_digest} "
            f"open_source={self.model_manifest.get('open_source')}"
        )

    def _predict_chunks(self, chunks: list[list[dict]]) -> list[float]:
        if self.detector is None:
            return [self.score_chunk(chunk) for chunk in chunks]
        return self.detector.predict_chunks(chunks)

    def _finalize_score(self, score: float) -> float:
        return round(max(0.0, min(1.0, float(score))), 6)

    def _resolve_top_k(self, n: int) -> int:
        if self.top_k and self.top_k > 0:
            return min(self.top_k, n)
        if self.top_k_frac and self.top_k_frac > 0.0:
            return max(1, min(n, int(round(self.top_k_frac * n))))
        return 0

    def _apply_top_k(self, scores: list[float]) -> list[float]:
        n = len(scores)
        k = self._resolve_top_k(n)
        if k <= 0 or k >= n:
            return scores
        order = sorted(range(n), key=lambda i: (scores[i], i), reverse=True)
        bot_idx = set(order[:k])
        bot_vals = [scores[i] for i in order[:k]]
        hum_vals = [scores[i] for i in order[k:]]
        b_lo, b_hi = min(bot_vals), max(bot_vals)
        h_lo, h_hi = min(hum_vals), max(hum_vals)
        out: list[float] = []
        for i, s in enumerate(scores):
            if i in bot_idx:
                frac = 0.0 if b_hi == b_lo else (s - b_lo) / (b_hi - b_lo)
                out.append(round(0.5 + 1e-6 + frac * (0.5 - 1e-6), 6))
            else:
                frac = 0.0 if h_hi == h_lo else (s - h_lo) / (h_hi - h_lo)
                out.append(round(frac * (0.5 - 1e-6), 6))
        return out

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        bt.logging.info(f"Received synapse from validator hotkey: {synapse.dendrite.hotkey}")
        if not chunks:
            synapse.risk_scores = []
            synapse.predictions = []
            synapse.model_manifest = dict(self.model_manifest)
            return synapse

        request_started = time.perf_counter()
        try:
            if self.detector is None:
                bt.logging.warning("Model not loaded. Using heuristic fallback.")
                raw_scores = [self.score_chunk(chunk) for chunk in chunks]
            else:
                raw_scores = self._predict_chunks(chunks)
            if len(raw_scores) != len(chunks):
                raise ValueError(f"score count mismatch: {len(raw_scores)} != {len(chunks)}")

            if os.getenv("P44_CAPTURE", "0") == "1":
                try:
                    from tools.live_capture import capture_batch

                    capture_batch(chunks, raw_scores)
                except Exception as capture_error:
                    bt.logging.warning(f"live capture skipped: {capture_error}")

            scores = [self._finalize_score(s) for s in raw_scores]
            scores = self._apply_top_k(scores)
            synapse.risk_scores = scores
            synapse.predictions = [s >= 0.5 for s in scores]
            synapse.model_manifest = dict(self.model_manifest)

            elapsed = time.perf_counter() - request_started
            bt.logging.info(
                f"Scored {len(chunks)} chunks | latency={elapsed:.3f}s | "
                f"bots={sum(s >= 0.5 for s in scores)}"
            )
            if self.latency_warn_seconds > 0 and elapsed >= self.latency_warn_seconds:
                bt.logging.warning(f"latency {elapsed:.3f}s exceeded warn budget")
            return synapse
        except Exception as exc:
            elapsed = time.perf_counter() - request_started
            bt.logging.error(f"Inference failed after {elapsed:.3f}s: {exc}")
            fallback = [0.49 for _ in chunks]
            synapse.risk_scores = fallback
            synapse.predictions = [False for _ in fallback]
            synapse.model_manifest = dict(self.model_manifest)
            return synapse

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _score_hand(cls, hand: dict) -> float:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}
        action_counts = Counter(action.get("action_type") for action in actions)
        meaningful = max(1, sum(action_counts.get(k, 0) for k in ("call", "check", "bet", "raise", "fold")))
        call_ratio = action_counts.get("call", 0) / meaningful
        check_ratio = action_counts.get("check", 0) / meaningful
        fold_ratio = action_counts.get("fold", 0) / meaningful
        raise_ratio = action_counts.get("raise", 0) / meaningful
        street_depth = len(streets) / 3.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0
        player_count_signal = (6 - min(len(players), 6)) / 4.0 if players else 0.0
        score = 0.0
        score += 0.32 * street_depth
        score += 0.22 * showdown_flag
        score += 0.18 * cls._clamp01(call_ratio / 0.35)
        score += 0.12 * cls._clamp01(check_ratio / 0.30)
        score += 0.08 * cls._clamp01(player_count_signal)
        score -= 0.18 * cls._clamp01(fold_ratio / 0.55)
        score -= 0.10 * cls._clamp01(raise_ratio / 0.20)
        return cls._clamp01(score)

    @classmethod
    def score_chunk(cls, chunk: list[dict]) -> float:
        if not chunk:
            return 0.5
        hand_scores = [cls._score_hand(hand) for hand in chunk]
        return round(cls._clamp01(sum(hand_scores) / len(hand_scores)), 6)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
