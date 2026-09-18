"""Features for AI CUP 2026 v4 — reuse v3 dense extractors; add set-level helpers."""
from __future__ import annotations

import sys
from pathlib import Path

_V3 = Path(__file__).resolve().parents[1] / "baseline_v3"
if str(_V3) not in sys.path:
    sys.path.insert(0, str(_V3))

from features_v3 import *  # noqa: F401,F403
from features_v3 import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    DENSE_DIM,
    FEATURE_NAMES,
    RANK_ORDER,
    RANK_TO_IDX,
    data_dir,
    dense_matrix,
    ensure_dir,
    load_or_cache_features_v3,
    rank_distance_score,
)

__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DENSE_DIM",
    "FEATURE_NAMES",
    "RANK_ORDER",
    "RANK_TO_IDX",
    "data_dir",
    "dense_matrix",
    "ensure_dir",
    "load_or_cache_features_v3",
    "rank_distance_score",
]
