"""Locate competition tables and optional extra passages."""

from __future__ import annotations

from pathlib import Path

from .extra import load_and_clean_extra
from .pipeline import find_test_path, find_train_path, load_table, normalize_frame


def load_competition(
    data_dir: str | Path | None = None,
    extra_roots: list[str | Path] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir) if data_dir else None
    train_extra = []
    test_extra = []
    if data_dir is not None:
        train_extra.extend(
            [
                data_dir / "train.csv",
                data_dir / "train.parquet",
            ]
        )
        test_extra.extend(
            [
                data_dir / "test.csv",
                data_dir / "test.parquet",
            ]
        )
    train = normalize_frame(load_table(find_train_path(train_extra)))
    test = normalize_frame(load_table(find_test_path(test_extra)))
    roots = list(extra_roots or [])
    if data_dir is not None:
        roots.append(data_dir / "external")
    roots.append("/kaggle/input")
    extra = load_and_clean_extra(roots, train, test_ids=test["id"].astype(str), max_rows=16_000)
    return train, test, extra
