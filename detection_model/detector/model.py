from __future__ import annotations

import numpy as np
<<<<<<< HEAD
from xgboost import XGBClassifier

MODEL_NAME = "pdx-gbr"
MODEL_VERSION = "1.0.0"
FRAMEWORK = "gradient-boosted-trees"
BOUNDARY = "exact"
BATCH_TOP_FRACTION = 0.1
USE_TEMPORAL = False
SEED = 1301
PER_LABEL = 260


def build_and_fit(x: np.ndarray, y: np.ndarray, *, seed: int = SEED):
    positive = max(int(np.sum(y == 1)), 1)
    negative = max(int(np.sum(y == 0)), 1)
    estimator = XGBClassifier(
        n_estimators=600,
        learning_rate=0.028,
        max_depth=6,
        min_child_weight=3.0,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=2.5,
        reg_alpha=0.5,
        gamma=0.5,
        scale_pos_weight=negative / positive,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )
    estimator.fit(x, y)
    return estimator


def raw_scores(estimator, x: np.ndarray) -> np.ndarray:
    return np.asarray(estimator.predict_proba(x)[:, 1], dtype=float)
=======
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "pdx-nt"
MODEL_VERSION = "1.0.0"
FRAMEWORK = "neural-temporal-ensemble"
BOUNDARY = "exact"
BATCH_TOP_FRACTION = 0.16
USE_TEMPORAL = True
SEED = 4523
PER_LABEL = 260

_WEIGHTS = (0.55, 0.45)


def build_and_fit(x: np.ndarray, y: np.ndarray, *, seed: int = SEED):
    hist = HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.04, max_depth=7, min_samples_leaf=3,
        l2_regularization=1.0, random_state=seed,
    ).fit(x, y)
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(160, 64), activation="relu", alpha=4e-4,
            batch_size=64, learning_rate_init=1e-3, max_iter=400,
            early_stopping=True, n_iter_no_change=20, random_state=seed + 7,
        ),
    ).fit(x, y)
    return {"hist": hist, "mlp": mlp}


def raw_scores(model, x: np.ndarray) -> np.ndarray:
    hist = model["hist"].predict_proba(x)[:, 1]
    mlp = model["mlp"].predict_proba(x)[:, 1]
    return np.asarray(_WEIGHTS[0] * hist + _WEIGHTS[1] * mlp, dtype=float)
>>>>>>> 19a2a21 (first)
