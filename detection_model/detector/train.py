from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from detection_model.corefeat.calibration import apply_mapper, fit_fixed_mapper
from detection_model.chunkfeat.features import (
    FEATURE_IMPLEMENTATION_SHA256,
    FEATURE_NAMES,
    matrix_for_chunks,
)
from detection_model.chunkfeat.mapping import exact_rank_map
from poker44.score.scoring import reward

from . import data as data_mod
from . import model as model_mod
from .boundary import adaptive_rank_map

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "detection_model" / "artifacts" / f"{model_mod.MODEL_NAME}.joblib"


def _matrix(clean: List[List[dict]]) -> np.ndarray:
    matrix = matrix_for_chunks(clean)
    if model_mod.USE_TEMPORAL:
        from .features_ext import temporal_matrix

        matrix = np.hstack([matrix, temporal_matrix(clean)])
    return matrix


def _feature_names() -> List[str]:
    names = list(FEATURE_NAMES)
    if model_mod.USE_TEMPORAL:
        from .features_ext import TEMPORAL_FEATURE_NAMES

        names = names + list(TEMPORAL_FEATURE_NAMES)
    return names


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _map_eval(raw: np.ndarray, boundary: str, fraction: float) -> np.ndarray:
    if raw.size < 8:
        return raw
    if boundary == "adaptive":
        return adaptive_rank_map(raw, default_fraction=fraction)
    return exact_rank_map(raw, fraction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=model_mod.SEED)
    parser.add_argument("--per-label", type=int, default=model_mod.PER_LABEL)
    args = parser.parse_args()

    corpus = data_mod.load_corpus(args.benchmark)
    train_records, val_records = data_mod.split_corpus(corpus)
    aug = data_mod.resample(train_records, per_label=args.per_label, seed=args.seed)
    print(f"train={len(train_records)} val={len(val_records)} augmented={len(aug)}")

    x_all = _matrix(data_mod.chunk_lists(aug))
    y_all = data_mod.labels(aug)

    x_fit, x_cal, y_fit, y_cal = train_test_split(
        x_all, y_all, test_size=0.18, random_state=args.seed, stratify=y_all
    )
    estimator = model_mod.build_and_fit(x_fit, y_fit, seed=args.seed)

    cal_raw = np.clip(model_mod.raw_scores(estimator, x_cal), 1e-6, 1 - 1e-6)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(cal_raw, y_cal)
    cal_adj = np.clip(calibrator.predict(cal_raw), 1e-6, 1 - 1e-6)
    mapper = fit_fixed_mapper(cal_adj, y_cal, target_human_fpr=0.05)

    val_chunks = data_mod.chunk_lists(val_records)
    val_y = data_mod.labels(val_records)
    if len(val_chunks) >= 1:
        v_raw = np.clip(model_mod.raw_scores(estimator, _matrix(val_chunks)), 1e-6, 1 - 1e-6)
        v_adj = np.clip(calibrator.predict(v_raw), 1e-6, 1 - 1e-6)
        if len(val_chunks) >= 8:
            v_scores = _map_eval(v_adj, model_mod.BOUNDARY, model_mod.BATCH_TOP_FRACTION)
        else:
            v_scores = apply_mapper(v_adj, mapper)
        rew, det = reward(np.asarray(v_scores), val_y)
        print(
            f"validation reward={rew:.4f} ap={det['ap_score']:.4f} "
            f"recall={det['bot_recall']:.4f} threshold_sanity={det['threshold_sanity_quality']:.3f}"
        )

    artifact = {
        "format": "detector-1",
        "model_name": model_mod.MODEL_NAME,
        "model_version": model_mod.MODEL_VERSION,
        "framework": model_mod.FRAMEWORK,
        "feature_names": _feature_names(),
        "feature_dim": int(x_all.shape[1]),
        "feature_implementation_sha256": FEATURE_IMPLEMENTATION_SHA256,
        "training_count": int(x_fit.shape[0]),
        "model": estimator,
        "calibrator": calibrator,
        "mapper": mapper,
        "batch_top_fraction": float(model_mod.BATCH_TOP_FRACTION),
        "boundary": model_mod.BOUNDARY,
        "use_temporal": bool(model_mod.USE_TEMPORAL),
        "seed": int(args.seed),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)
    print(f"saved {output}")
    print(f"artifact_sha256={_sha256(output)}")


if __name__ == "__main__":
    main()
