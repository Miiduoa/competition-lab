"""SGF / CSV feature helpers for AI CUP 2026 Go baselines.

Designed for streaming: never load all ~1M SGFs into RAM at once.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Ordered ranks: weak → strong (official dataset order)
RANK_ORDER: List[str] = [
    "D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D",
]
RANK_TO_IDX: Dict[str, int] = {r: i for i, r in enumerate(RANK_ORDER)}

_MOVE_RE = re.compile(r";([BW])\[([a-s]{0,2})\]", re.IGNORECASE)

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"


def data_dir(task: str = "task1", root: Optional[Path] = None) -> Path:
    root = Path(root) if root else DEFAULT_DATA_ROOT
    return root / task


def list_rank_csvs(task_dir: Path) -> List[Path]:
    paths = []
    for r in RANK_ORDER:
        p = task_dir / f"train_{r}.csv"
        if p.exists():
            paths.append(p)
    return paths


def parse_moves(sgf: str) -> List[Tuple[str, str]]:
    """Return list of (color, coord) from SGF. Coord '' = pass."""
    if not isinstance(sgf, str) or not sgf:
        return []
    return [(m.group(1).upper(), m.group(2).lower()) for m in _MOVE_RE.finditer(sgf)]


def move_count(sgf: str) -> int:
    if not isinstance(sgf, str) or not sgf:
        return 0
    return len(_MOVE_RE.findall(sgf))


def opening_fingerprint(sgf: str, n: int = 8) -> str:
    """First N stone placements as compact string, e.g. 'BqdWdpBpqWdd…'."""
    moves = parse_moves(sgf)[:n]
    return "".join(f"{c}{coord}" for c, coord in moves)


def first_move_coord(sgf: str) -> str:
    moves = parse_moves(sgf)
    return moves[0][1] if moves else ""


def color_of_row(color) -> str:
    return str(color).upper()[:1] if color is not None else "?"


def game_length_bin(n_moves: int, edges: Sequence[int] = (50, 100, 150, 200, 250, 300)) -> int:
    for i, e in enumerate(edges):
        if n_moves < e:
            return i
    return len(edges)


def _sgf_stats(sgf: str, opening_n: int = 8) -> Tuple[int, str, str]:
    moves = parse_moves(sgf)
    n = len(moves)
    opening = "".join(f"{c}{coord}" for c, coord in moves[:opening_n])
    first = moves[0][1] if moves else ""
    return n, opening, first


def features_from_chunk(chunk: pd.DataFrame, opening_n: int = 8) -> pd.DataFrame:
    """Vectorized-ish feature extraction for one CSV chunk."""
    stats = [_sgf_stats(s, opening_n) for s in chunk["sgf_content"].tolist()]
    n_moves = [t[0] for t in stats]
    openings = [t[1] for t in stats]
    firsts = [t[2] for t in stats]
    return pd.DataFrame(
        {
            "player_id": chunk["player_id"].values,
            "game_id": chunk["game_id"].values,
            "rank": chunk["rank"].values,
            "color": [color_of_row(c) for c in chunk["color"].tolist()],
            "n_moves": n_moves,
            "length_bin": [game_length_bin(n) for n in n_moves],
            "opening": openings,
            "first_coord": firsts,
        }
    )


def iter_feature_chunks(
    csv_path: Path,
    chunksize: int = 5000,
    max_rows: Optional[int] = None,
    opening_n: int = 8,
    usecols: Optional[Sequence[str]] = None,
) -> Iterator[pd.DataFrame]:
    """Stream a rank CSV and yield DataFrames of lightweight features."""
    cols = list(usecols) if usecols else ["player_id", "game_id", "rank", "color", "sgf_content"]
    seen = 0
    for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=chunksize):
        if max_rows is not None:
            remain = max_rows - seen
            if remain <= 0:
                break
            if len(chunk) > remain:
                chunk = chunk.iloc[:remain]
        yield features_from_chunk(chunk, opening_n=opening_n)
        seen += len(chunk)
        if max_rows is not None and seen >= max_rows:
            break


def load_sampled_features(
    task_dir: Path,
    max_games_per_rank: int = 4000,
    chunksize: int = 4000,
    opening_n: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Load up to max_games_per_rank rows per rank file (streamed + subsampled)."""
    rng = np.random.default_rng(seed)
    parts: List[pd.DataFrame] = []
    for csv_path in list_rank_csvs(task_dir):
        collected: List[pd.DataFrame] = []
        n_got = 0
        for feat_df in iter_feature_chunks(
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
        raise FileNotFoundError(f"No train_*.csv found under {task_dir}")
    return pd.concat(parts, ignore_index=True)


def length_hist(counts: Iterable[int], n_bins: int = 7) -> np.ndarray:
    h = np.zeros(n_bins, dtype=np.float64)
    for b in counts:
        b = int(b)
        if 0 <= b < n_bins:
            h[b] += 1.0
    s = h.sum()
    if s > 0:
        h /= s
    return h


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def jaccard_from_counters(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return inter / union if union else 0.0


def top5_exp_decay_score(true_id: str, ranked_ids: Sequence[str]) -> float:
    """Official-style Top-5 exponential decay: e^{-(k-1)} for rank k=1..5."""
    for k, pid in enumerate(ranked_ids[:5], start=1):
        if pid == true_id:
            return float(np.exp(-(k - 1)))
    return 0.0


def rank_distance_score(true_rank: str, pred_rank: str) -> float:
    """Exact=1, ±1 level=1/e, else 0."""
    if true_rank not in RANK_TO_IDX or pred_rank not in RANK_TO_IDX:
        return 0.0
    d = abs(RANK_TO_IDX[true_rank] - RANK_TO_IDX[pred_rank])
    if d == 0:
        return 1.0
    if d == 1:
        return float(1.0 / np.e)
    return 0.0


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
