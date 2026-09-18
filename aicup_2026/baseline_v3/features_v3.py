"""Richer SGF / CSV features for AI CUP 2026 Go v3 models.

Extends v2: joseki/pattern hashes, midgame region n-grams, color-conditioned
region stats, cheap capture / contact proxies, larger opening tokens.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

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
    "rich_game_features_v3",
    "features_from_chunk_v3",
    "iter_feature_chunks_v3",
    "load_sampled_features_v3",
    "load_or_cache_features_v3",
    "dense_matrix",
    "DENSE_DIM",
    "FEATURE_NAMES",
]

_MOVE_RE = re.compile(r";([BW])\[([a-s]{0,2})\]", re.IGNORECASE)
_CHARS = "abcdefghijklmnopqrs"
_COORD_TO_I = {c: i for i, c in enumerate(_CHARS)}
_NEIGH = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Dense layout
_N_LENGTH = 7
_N_REGION = 9
_N_DELTA = 8
_N_EARLY = 9
_N_FIRST = 9
_N_OPEN_HASH = 48  # was 24
_N_MID_HASH = 32
_N_JOSEKI_HASH = 32
_N_OWN_REGION = 9
_N_OPP_REGION = 9
_N_CAPTURE = 8  # capture/contact proxies
_N_SCALAR = 12  # extended scalars

DENSE_DIM = (
    _N_SCALAR
    + _N_LENGTH
    + _N_REGION
    + _N_DELTA
    + _N_EARLY
    + _N_FIRST
    + _N_OPEN_HASH
    + _N_MID_HASH
    + _N_JOSEKI_HASH
    + _N_OWN_REGION
    + _N_OPP_REGION
    + _N_CAPTURE
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
        "early_corner",
        "contact_rate",
        "capture_rate",
        "self_atari_proxy",
    ]
    + [f"len_bin_{i}" for i in range(_N_LENGTH)]
    + [f"region_{i}" for i in range(_N_REGION)]
    + [f"delta_{i}" for i in range(_N_DELTA)]
    + [f"early_{i}" for i in range(_N_EARLY)]
    + [f"first_r_{i}" for i in range(_N_FIRST)]
    + [f"open_h_{i}" for i in range(_N_OPEN_HASH)]
    + [f"mid_h_{i}" for i in range(_N_MID_HASH)]
    + [f"joseki_h_{i}" for i in range(_N_JOSEKI_HASH)]
    + [f"own_r_{i}" for i in range(_N_OWN_REGION)]
    + [f"opp_r_{i}" for i in range(_N_OPP_REGION)]
    + [f"cap_{i}" for i in range(_N_CAPTURE)]
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
    cx = min(2, x // 7)
    cy = min(2, y // 7)
    return cy * 3 + cx


def _hash_token(s: str, n: int) -> int:
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h % n


def _corner_mirror_key(x: int, y: int) -> Tuple[int, int]:
    """Map stone into top-left-ish for joseki hash (fold board into 10x10)."""
    xx = x if x <= 9 else 18 - x
    yy = y if y <= 9 else 18 - y
    return xx, yy


def _capture_contact_stats(
    moves: List[Tuple[str, str]], max_moves: int = 80
) -> Tuple[float, float, float, np.ndarray]:
    """Return contact_rate, capture_rate, self_atari_proxy, cap_hist(8)."""
    board = np.zeros((19, 19), dtype=np.int8)
    contact = 0
    placed = 0
    captures = 0
    self_atari = 0
    cap_h = np.zeros(8, dtype=np.float64)
    # Use simple adjacent-empty liberty proxy without full UF for speed;
    # count captures via flood-fill when a play fills last liberty of neighbor group.

    def liberties(x: int, y: int, col: int, visited_group: set) -> set:
        stack = [(x, y)]
        libs = set()
        group = set()
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in group:
                continue
            group.add((cx, cy))
            for dx, dy in _NEIGH:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < 19 and 0 <= ny < 19):
                    continue
                v = board[ny, nx]
                if v == 0:
                    libs.add((nx, ny))
                elif v == col and (nx, ny) not in group:
                    stack.append((nx, ny))
        visited_group |= group
        return libs

    def remove_group(group: set) -> None:
        for gx, gy in group:
            board[gy, gx] = 0

    for i, (col, coord) in enumerate(moves[:max_moves]):
        xy = coord_to_xy(coord)
        if xy is None:
            continue
        x, y = xy
        if board[y, x] != 0:
            continue
        c = 1 if col == "B" else 2
        opp = 3 - c
        # contact?
        touched = False
        for dx, dy in _NEIGH:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 19 and 0 <= ny < 19 and board[ny, nx] == opp:
                touched = True
                break
        if touched:
            contact += 1
        board[y, x] = c
        placed += 1
        # capture opponent groups
        cap_this = 0
        for dx, dy in _NEIGH:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < 19 and 0 <= ny < 19):
                continue
            if board[ny, nx] != opp:
                continue
            group: set = set()
            libs = liberties(nx, ny, opp, group)
            if not libs:
                cap_this += len(group)
                remove_group(group)
        captures += cap_this
        if cap_this == 0:
            bi = 0
        elif cap_this == 1:
            bi = 1
        elif cap_this == 2:
            bi = 2
        elif cap_this <= 4:
            bi = 3
        elif cap_this <= 8:
            bi = 4
        elif cap_this <= 15:
            bi = 5
        elif cap_this <= 30:
            bi = 6
        else:
            bi = 7
        cap_h[bi] += 1.0
        # self-atari proxy: own group has 1 liberty after place
        group_s: set = set()
        own_libs = liberties(x, y, c, group_s)
        if len(own_libs) <= 1:
            self_atari += 1

    denom = max(placed, 1)
    if cap_h.sum() > 0:
        cap_h /= cap_h.sum()
    return contact / denom, captures / denom, self_atari / denom, cap_h


def rich_game_features_v3(
    sgf: str,
    color: str = "?",
    opening_n: int = 16,
) -> Tuple[np.ndarray, str, str, str, str, List[str], List[str]]:
    """Return dense, opening16, opening8, opening12, first, tokens, pattern_tokens."""
    moves = parse_moves(sgf)
    n = len(moves)
    n_pass = sum(1 for _, c in moves if not c)
    player_color = str(color).upper()[:1]
    color_b = 1.0 if player_color == "B" else 0.0

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
    mid_h = np.zeros(_N_MID_HASH, dtype=np.float64)
    joseki_h = np.zeros(_N_JOSEKI_HASH, dtype=np.float64)
    own_r = np.zeros(_N_OWN_REGION, dtype=np.float64)
    opp_r = np.zeros(_N_OPP_REGION, dtype=np.float64)

    manhs: List[float] = []
    prev_xy: Optional[Tuple[int, int]] = None
    corner = side = center = 0
    early_corner = 0
    placed = 0

    opening_moves = moves[:opening_n]
    opening16 = "".join(f"{c}{coord}" for c, coord in opening_moves)
    opening12 = "".join(f"{c}{coord}" for c, coord in moves[:12])
    opening8 = "".join(f"{c}{coord}" for c, coord in moves[:8])
    first_coord = moves[0][1] if moves else ""
    tokens: List[str] = []
    pattern_tokens: List[str] = []

    # Opening tokens + bigrams + color-tagged
    for i, (col, coord) in enumerate(opening_moves):
        tok = f"{col}{coord}" if coord else f"{col}pass"
        tokens.append(tok)
        tokens.append(f"P{player_color}:{tok}")
        if i > 0:
            prev = opening_moves[i - 1]
            prev_tok = f"{prev[0]}{prev[1]}" if prev[1] else f"{prev[0]}pass"
            tokens.append(f"{prev_tok}>{tok}")
        # joseki folded coords for first 12
        if i < 12:
            xy = coord_to_xy(coord)
            if xy is not None:
                mx, my = _corner_mirror_key(*xy)
                jtok = f"J{col}{mx:x}{my:x}"
                pattern_tokens.append(jtok)
                joseki_h[_hash_token(jtok, _N_JOSEKI_HASH)] += 1.0
                if i > 0:
                    pxy = coord_to_xy(opening_moves[i - 1][1])
                    if pxy is not None:
                        pmx, pmy = _corner_mirror_key(*pxy)
                        jbi = f"J{opening_moves[i-1][0]}{pmx:x}{pmy:x}>{col}{mx:x}{my:x}"
                        pattern_tokens.append(jbi)
                        joseki_h[_hash_token(jbi, _N_JOSEKI_HASH)] += 1.0

    if joseki_h.sum() > 0:
        joseki_h /= joseki_h.sum()

    # Region n-grams for opening
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

    # Midgame region n-grams (moves 16..60)
    mid_regions: List[int] = []
    for _, coord in moves[16:60]:
        xy = coord_to_xy(coord)
        if xy is None:
            continue
        mid_regions.append(region_id(*xy))
    for i in range(len(mid_regions)):
        mid_h[_hash_token(f"m{mid_regions[i]}", _N_MID_HASH)] += 1.0
        if i + 1 < len(mid_regions):
            mid_h[
                _hash_token(f"m{mid_regions[i]}-{mid_regions[i+1]}", _N_MID_HASH)
            ] += 1.0
        if i + 2 < len(mid_regions):
            mid_h[
                _hash_token(
                    f"m{mid_regions[i]}-{mid_regions[i+1]}-{mid_regions[i+2]}",
                    _N_MID_HASH,
                )
            ] += 1.0
        pattern_tokens.append(f"M{mid_regions[i]}")
        if i + 1 < len(mid_regions):
            pattern_tokens.append(f"M{mid_regions[i]}-{mid_regions[i+1]}")
    if mid_h.sum() > 0:
        mid_h /= mid_h.sum()

    for i, (col, coord) in enumerate(moves):
        xy = coord_to_xy(coord)
        if xy is None:
            continue
        placed += 1
        rid = region_id(*xy)
        region_h[rid] += 1.0
        if rid <= 3:
            corner += 1
            if i < 20:
                early_corner += 1
        elif rid <= 7:
            side += 1
        else:
            center += 1
        if i < 24:
            early_h[_coarse3(*xy)] += 1.0
        if i == 0:
            first_h[rid] = 1.0
        # color-conditioned: player's own color vs opponent
        if player_color in ("B", "W"):
            if col == player_color:
                own_r[rid] += 1.0
            else:
                opp_r[rid] += 1.0
        if prev_xy is not None:
            manh = abs(xy[0] - prev_xy[0]) + abs(xy[1] - prev_xy[1])
            manhs.append(float(manh))
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
    if own_r.sum() > 0:
        own_r /= own_r.sum()
    if opp_r.sum() > 0:
        opp_r /= opp_r.sum()

    contact_rate, capture_rate, self_atari_proxy, cap_h = _capture_contact_stats(moves)

    denom = max(placed, 1)
    mean_manh = float(np.mean(manhs)) if manhs else 0.0
    std_manh = float(np.std(manhs)) if manhs else 0.0
    pass_rate = n_pass / max(n, 1)
    corner_r = corner / denom
    side_r = side / denom
    center_r = center / denom
    early_corner_r = early_corner / max(min(20, placed), 1)

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
                    early_corner_r,
                    contact_rate,
                    capture_rate,
                    self_atari_proxy,
                ],
                dtype=np.float64,
            ),
            len_h,
            region_h,
            delta_h,
            early_h,
            first_h,
            open_h,
            mid_h,
            joseki_h,
            own_r,
            opp_r,
            cap_h,
        ]
    )
    assert dense.shape[0] == DENSE_DIM, (dense.shape[0], DENSE_DIM)
    return dense, opening16, opening8, opening12, first_coord, tokens, pattern_tokens


def features_from_chunk_v3(chunk: pd.DataFrame, opening_n: int = 16) -> pd.DataFrame:
    dens_list = []
    op16, op8, op12, firsts = [], [], [], []
    tok_list, pat_list = [], []
    colors = []
    for sgf, col in zip(chunk["sgf_content"].tolist(), chunk["color"].tolist()):
        d, o16, o8, o12, fc, toks, pats = rich_game_features_v3(
            sgf, color=col, opening_n=opening_n
        )
        dens_list.append(d)
        op16.append(o16)
        op8.append(o8)
        op12.append(o12)
        firsts.append(fc)
        tok_list.append(" ".join(toks))
        pat_list.append(" ".join(pats))
        colors.append(str(col).upper()[:1] if col is not None else "?")

    dens = np.vstack(dens_list) if dens_list else np.zeros((0, DENSE_DIM))
    meta = pd.DataFrame(
        {
            "player_id": chunk["player_id"].values,
            "game_id": chunk["game_id"].values,
            "rank": chunk["rank"].values,
            "color": colors,
            "opening": op16,
            "opening8": op8,
            "opening12": op12,
            "first_coord": firsts,
            "opening_tokens": tok_list,
            "pattern_tokens": pat_list,
            "length_bin": dens[:, _N_SCALAR : _N_SCALAR + _N_LENGTH].argmax(axis=1)
            if len(dens)
            else np.array([], dtype=int),
        }
    )
    dens_df = pd.DataFrame(dens, columns=FEATURE_NAMES)
    return pd.concat([meta, dens_df], axis=1)


def iter_feature_chunks_v3(
    csv_path: Path,
    chunksize: int = 2000,
    max_rows: Optional[int] = None,
    opening_n: int = 16,
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
        yield features_from_chunk_v3(chunk, opening_n=opening_n)
        seen += len(chunk)
        if max_rows is not None and seen >= max_rows:
            break


def load_sampled_features_v3(
    task_dir: Path,
    max_games_per_rank: int = 15000,
    chunksize: int = 1500,
    opening_n: int = 16,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts: List[pd.DataFrame] = []
    for csv_path in list_rank_csvs(task_dir):
        collected: List[pd.DataFrame] = []
        n_got = 0
        for feat_df in iter_feature_chunks_v3(
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


def load_or_cache_features_v3(
    task_dir: Path,
    cache_path: Path,
    max_games_per_rank: int = 15000,
    seed: int = 42,
    force: bool = False,
) -> pd.DataFrame:
    """Load features from parquet cache if compatible, else compute and save."""
    if cache_path.exists() and not force:
        meta_path = cache_path.with_suffix(".meta.json")
        ok = False
        if meta_path.exists():
            import json

            meta = json.loads(meta_path.read_text())
            ok = (
                meta.get("max_games_per_rank") == max_games_per_rank
                and meta.get("seed") == seed
                and meta.get("dense_dim") == DENSE_DIM
            )
        if ok:
            print(f"  [cache] loading {cache_path}")
            return pd.read_parquet(cache_path)
    df = load_sampled_features_v3(
        task_dir, max_games_per_rank=max_games_per_rank, seed=seed
    )
    ensure_dir(cache_path.parent)
    try:
        df.to_parquet(cache_path, index=False)
        import json

        cache_path.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "max_games_per_rank": max_games_per_rank,
                    "seed": seed,
                    "dense_dim": DENSE_DIM,
                    "n_rows": int(len(df)),
                }
            )
        )
        print(f"  [cache] wrote {cache_path}")
    except Exception as e:
        print(f"  [cache] skip write ({e})")
    return df
