"""Small vectorised 2D vector helpers. All functions operate on (N, 2) arrays."""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def norms(v: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean length, shape (N,)."""
    return np.linalg.norm(v, axis=-1)


def safe_normalize(v: np.ndarray) -> np.ndarray:
    """Unit vectors; zero rows stay zero instead of producing NaN."""
    n = norms(v)[..., None]
    return np.where(n > EPS, v / np.maximum(n, EPS), 0.0)


def clamp_norm(v: np.ndarray, max_len: float) -> np.ndarray:
    """Scale down rows longer than max_len, leave shorter rows unchanged."""
    n = norms(v)[..., None]
    scale = np.where(n > max_len, max_len / np.maximum(n, EPS), 1.0)
    return v * scale
