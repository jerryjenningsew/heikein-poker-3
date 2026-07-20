from __future__ import annotations

from typing import Sequence

import numpy as np

from detection_model.chunkfeat.mapping import exact_rank_map


def largest_gap_fraction(
    scores: Sequence[float] | np.ndarray,
    *,
    lo: float = 0.05,
    hi: float = 0.55,
    min_gap: float = 0.0,
) -> float:
    """Estimate the positive fraction from the widest separation in the scores.

    The batch is read as an ascending-sorted score vector; the high group begins
    at the widest interior gap located in the plausible positive-rate window. The
    returned fraction is the size of that high group relative to the batch, so the
    operating boundary can be placed inside the natural separation rather than at
    a fixed preset rate.
    """
    values = np.sort(np.asarray(scores, dtype=float))
    n = int(values.size)
    if n < 4:
        return 0.10
    lo_idx = max(1, int(round(n * (1.0 - hi))))
    hi_idx = min(n - 1, int(round(n * (1.0 - lo))))
    best_gap = -1.0
    best_cut = hi_idx
    for i in range(lo_idx, hi_idx + 1):
        gap = values[i] - values[i - 1]
        if gap > best_gap:
            best_gap = gap
            best_cut = i
    if best_gap < min_gap:
        return 0.0
    return float(n - best_cut) / float(n)


def adaptive_rank_map(
    scores: Sequence[float] | np.ndarray,
    *,
    tie_keys: Sequence[str] | None = None,
    default_fraction: float = 0.10,
    min_gap: float = 0.02,
) -> np.ndarray:
    """Rank-preserving boundary whose positive budget adapts to each batch.

    Falls back to ``default_fraction`` when no credible separation is present, so
    a single-population batch is never forced to manufacture a crossing.
    """
    values = np.asarray(scores, dtype=float)
    if values.size < 8:
        return exact_rank_map(values, default_fraction, tie_keys=tie_keys)
    fraction = largest_gap_fraction(values, min_gap=min_gap)
    if fraction <= 0.0:
        fraction = default_fraction
    fraction = float(min(0.55, max(0.02, fraction)))
    return exact_rank_map(values, fraction, tie_keys=tie_keys)
