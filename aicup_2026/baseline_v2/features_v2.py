"""Richer SGF / CSV features for AI CUP 2026 Go v2 models.

Extends baseline/features.py patterns: streamable chunked CSV reads,
richer per-game vectors (opening n-grams, regions, move-deltas, pass rate).
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Reuse baseline helpers when available
_BASELINE = Path(__file__).resolve().parents[1] / "baseline"
if str(_BASELINE) not in sys.path:
    sys.path.insert(0, str(_BASELINE))

from features import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    RANK_ORDER,
    RANK_TO_IDX,
    cosine_sim,
    data_dir,
    ensure_dir,
    jaccard_from_counters,
    length_hist,
    list_rank_csvs,
    rank_distance_score,
    top5_exp_decay_score,
)

__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "RANK_ORDER",
    "RANK_TO_IDX",
    "cosine_sim",
    "data_dir",
    "ensure_dir",
    "jaccard_from_counters",
    "length_hist",
    "list_rank_csvs",
    "rank_distance_score",
    "top5_exp_decay_score",
    "parse_moves",
    "coord_to_xy",
    "region_id",
    "rich_game_features",
    "features_from_chunk_v2",
    "iter_feature_chunks_v2",
    "load_sampled_features_v2",
    "DENSE_DIM",
    "FEATURE_NAMES",
]

_MOVE_RE = re.compile(r";([BW])\[([a-s]{0,2})\]", re.IGNORECASE)
_CHARS = "abcdefghijklmnopqrs"
_COORD_TO_I = {c: i for i, c in enumerate(_CHARS)}

# Dense vector layout (fixed size for mean/std aggregation)
# scalar block + length hist(7) + region hist(9) + delta hist(8)
# + early 3x3 heat(9) + first-region onehot(9) + opening-region trigrams hashed(24)
_N_LENGTH = 7
_N_REGION = 9
_N_DELTA = 8
_N_EARLY = 9
_N_FIRST = 9
_N_OPEN_HASH = 24
_N_SCALAR = 8  # n_moves, pass_rate, color_b, mean_manh, std_manh, corner_r, side_r, center_r

DENSE_DIM = (
    _N_SCALAR
    + _N_LENGTH
    + _N_REGION
    + _N_DELTA
    + _N_EARLY
    + _N_FIRST
    + _N_OPEN_HASH
)

FEATURE_NAMES: List[str] = (
    [
        "n_moves",
        "pass_rate",
        "color_b",
        "mean_manh",
        "std_manh",
        "corner_rate",
        "side_rate",
        "center_rate",
    ]
    + [f"len_bin_{i}" for i in range(_N_LENGTH)]
    + [f"region_{i}" for i in range(_N_REGION)]
    + [f"delta_{i}" for i in range(_N_DELTA)]
    + [f"early_{i}" for i in range(_N_EARLY)]
    + [f"first_r_{i}" for i in range(_N_FIRST)]
    + [f"open_h_{i}" for i in range(_N_OPEN_HASH)]
)


def parse_moves(sgf: str) -> List[Tuple[str, str]]:
    if not isinstance(sgf, str) or not sgf:
        return []
    return [(m.group(1).upper(), m.group(2).lower()) for m in _MOVE_RE.finditer(sgf)]


def coord_to_xy(coord: str) -> Optional[Tuple[int, int]]:
    if not coord or len(coord) != 2:
        return None
    x = _COORD_TO_I.get(coord[0])
    y = _COORD_TO_I.get(coord[1])
    if x is None or y is None:
        return None
    return x, y


def region_id(x: int, y: int) -> int:
    """0–3 corners, 4–7 sides, 8 center. Corner radius 6 on 19×19."""
    near_l, near_r = x <= 5, x >= 13
    near_t, near_b = y <= 5, y >= 13
    if near_l and near_t:
        return 0
    if near_r and near_t:
        return 1
    if near_l and near_b:
        return 2
    if near_r and near_b:
        return 3
    if near_t:
        return 4
    if near_b:
        return 5
    if near_l:
        return 6
    if near_r:
        return 7
    return 8


def _coarse3(x: int, y: int) -> int:
    """Map 19×19 → 3×3 cell index 0..8."""
    cx = min(2, x // 7)
    cy = min(2, y // 7)
    return cy * 3 + cx


def _hash_token(s: str, n: int) -> int:
    # Stable lightweight hash (avoid hashlib overhead in inner loop)
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h % n


def rich_game_features(
    sgf: str,
    color: str = "?",
    opening_n: int = 12,
) -> Tuple[np.ndarray, str, str, str, List[str]]:
    """Return (dense_vec, opening12, opening8, first_coord, opening_tokens)."""
    moves = parse_moves(sgf)
    n = len(moves)
    n_pass = sum(1 for _, c in moves if not c)
    color_b = 1.0 if str(color).upper()[:1] == "B" else 0.0

    length_bin = _N_LENGTH - 1
    edges = (50, 100, 150, 200, 250, 300)
    for i, e in enumerate(edges):
        if n < e:
            length_bin = i
            break

    region_h = np.zeros(_N_REGION, dtype=np.float64)
    delta_h = np.zeros(_N_DELTA, dtype=np.float64)
    early_h = np.zeros(_N_EARLY, dtype=np.float64)
    first_h = np.zeros(_N_FIRST, dtype=np.float64)
    open_h = np.zeros(_N_OPEN_HASH, dtype=np.float64)

    manhs: List[float] = []
    prev_xy: Optional[Tuple[int, int]] = None
    corner = side = center = 0
    placed = 0

    opening_moves = moves[:opening_n]
    opening12 = "".join(f"{c}{coord}" for c, coord in opening_moves)
    opening8 = "".join(f"{c}{coord}" for c, coord in moves[:8])
    first_coord = moves[0][1] if moves else ""
    tokens: List[str] = []

    # Opening tokens: single moves + bigrams (style fingerprint)
    for i, (col, coord) in enumerate(opening_moves):
        tok = f"{col}{coord}" if coord else f"{col}pass"
        tokens.append(tok)
        if i > 0:
            prev = opening_moves[i - 1]
            prev_tok = f"{prev[0]}{prev[1]}" if prev[1] else f"{prev[0]}pass"
            tokens.append(f"{prev_tok}>{tok}")

    # Hash opening region trigrams into bag
    open_regions: List[int] = []
    for _, coord in opening_moves:
        xy = coord_to_xy(coord)
        if xy is None:
            continue
        open_regions.append(region_id(*xy))
    for i in range(len(open_regions)):
        open_h[_hash_token(f"r{open_regions[i]}", _N_OPEN_HASH)] += 1.0
        if i + 1 < len(open_regions):
            open_h[
                _hash_token(f"r{open_regions[i]}-{open_regions[i+1]}", _N_OPEN_HASH)
            ] += 1.0
        if i + 2 < len(open_regions):
            open_h[
                _hash_token(
                    f"r{open_regions[i]}-{open_regions[i+1]}-{open_regions[i+2]}",
                    _N_OPEN_HASH,
                )
            ] += 1.0
    if open_h.sum() > 0:
        open_h /= open_h.sum()

    for i, (_, coord) in enumerate(moves):
        xy = coord_to_xy(coord)
        if xy is None:
            continue
        placed += 1
        rid = region_id(*xy)
        region_h[rid] += 1.0
        if rid <= 3:
            corner += 1
        elif rid <= 7:
            side += 1
        else:
            center += 1
        if i < 20:
            early_h[_coarse3(*xy)] += 1.0
        if i == 0:
            first_h[rid] = 1.0
        if prev_xy is not None:
            manh = abs(xy[0] - prev_xy[0]) + abs(xy[1] - prev_xy[1])
            manhs.append(float(manh))
            # bins: 0-1,2-3,4-5,6-7,8-10,11-14,15-20,21+
            if manh <= 1:
                bi = 0
            elif manh <= 3:
                bi = 1
            elif manh <= 5:
                bi = 2
            elif manh <= 7:
                bi = 3
            elif manh <= 10:
                bi = 4
            elif manh <= 14:
                bi = 5
            elif manh <= 20:
                bi = 6
            else:
                bi = 7
            delta_h[bi] += 1.0
        prev_xy = xy

    if region_h.sum() > 0:
        region_h /= region_h.sum()
    if delta_h.sum() > 0:
        delta_h /= delta_h.sum()
    if early_h.sum() > 0:
        early_h /= early_h.sum()

    denom = max(placed, 1)
    mean_manh = float(np.mean(manhs)) if manhs else 0.0
    std_manh = float(np.std(manhs)) if manhs else 0.0
    pass_rate = n_pass / max(n, 1)
    corner_r = corner / denom
    side_r = side / denom
    center_r = center / denom

    len_h = np.zeros(_N_LENGTH, dtype=np.float64)
    len_h[length_bin] = 1.0

    dense = np.concatenate(
        [
            np.array(
                [
                    float(n),
                    pass_rate,
                    color_b,
                    mean_manh,
                    std_manh,
                    corner_r,
                    side_r,
                    center_r,
                ],
                dtype=np.float64,
            ),
            len_h,
            region_h,
            delta_h,
            early_h,
            first_h,
            open_h,
        ]
    )
    assert dense.shape[0] == DENSE_DIM
    return dense, opening12, opening8, first_coord, tokens


def features_from_chunk_v2(chunk: pd.DataFrame, opening_n: int = 12) -> pd.DataFrame:
    dens_list = []
    op12, op8, firsts = [], [], []
    tok_list = []
    colors = []
    for sgf, col in zip(chunk["sgf_content"].tolist(), chunk["color"].tolist()):
        d, o12, o8, fc, toks = rich_game_features(sgf, color=col, opening_n=opening_n)
        dens_list.append(d)
        op12.append(o12)
        op8.append(o8)
        firsts.append(fc)
        tok_list.append(" ".join(toks))
        colors.append(str(col).upper()[:1] if col is not None else "?")

    dens = np.vstack(dens_list) if dens_list else np.zeros((0, DENSE_DIM))
    out = {
        "player_id": chunk["player_id"].values,
        "game_id": chunk["game_id"].values,
        "rank": chunk["rank"].values,
        "color": colors,
        "n_moves": dens[:, 0] if len(dens) else [],
        "opening": op12,  # longer fingerprint for Jaccard
        "opening8": op8,
        "first_coord": firsts,
        "opening_tokens": tok_list,
        "length_bin": dens[:, _N_SCALAR : _N_SCALAR + _N_LENGTH].argmax(axis=1)
        if len(dens)
        else [],
    }
    df = pd.DataFrame(out)
    # attach dense columns
    for i, name in enumerate(FEATURE_NAMES):
        df[name] = dens[:, i] if len(dens) else []
    return df


def iter_feature_chunks_v2(
    csv_path: Path,
    chunksize: int = 3000,
    max_rows: Optional[int] = None,
    opening_n: int = 12,
) -> Iterator[pd.DataFrame]:
    cols = ["player_id", "game_id", "rank", "color", "sgf_content"]
    seen = 0
    for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=chunksize):
        if max_rows is not None:
            remain = max_rows - seen
            if remain <= 0:
                break
            if len(chunk) > remain:
                chunk = chunk.iloc[:remain]
        yield features_from_chunk_v2(chunk, opening_n=opening_n)
        seen += len(chunk)
        if max_rows is not None and seen >= max_rows:
            break


def load_sampled_features_v2(
    task_dir: Path,
    max_games_per_rank: int = 6000,
    chunksize: int = 2500,
    opening_n: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Stream + subsample richer features per rank."""
    rng = np.random.default_rng(seed)
    parts: List[pd.DataFrame] = []
    for csv_path in list_rank_csvs(task_dir):
        collected: List[pd.DataFrame] = []
        n_got = 0
        # Reservoir-ish: take first chunks with random subsample to hit target faster
        for feat_df in iter_feature_chunks_v2(
            csv_path, chunksize=chunksize, opening_n=opening_n
        ):
            need = max_games_per_rank - n_got
            if need <= 0:
                break
            if len(feat_df) > need:
                idx = rng.choice(len(feat_df), size=need, replace=False)
                feat_df = feat_df.iloc[np.sort(idx)]
            collected.append(feat_df)
            n_got += len(feat_df)
            if n_got >= max_games_per_rank:
                break
        if collected:
            parts.append(pd.concat(collected, ignore_index=True))
            print(f"  sampled {n_got} from {csv_path.name}")
    if not parts:
        raise FileNotFoundError(f"No train_*.csv under {task_dir}")
    return pd.concat(parts, ignore_index=True)


def dense_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_NAMES].to_numpy(dtype=np.float64)
