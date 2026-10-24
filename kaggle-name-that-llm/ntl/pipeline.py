"""Multi-view n-gram + stylometry authorship classifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .constants import (
    FAMILIES,
    FAMILY_OF,
    KAGGLE_TEST_CANDIDATES,
    KAGGLE_TRAIN_CANDIDATES,
    LABELS,
    MARK_PATTERNS,
)
from .features import (
    extract_dense,
    mark_hits,
    to_word_shape,
)

WORD_TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"


def balanced_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def softmax_scores(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64) / max(temperature, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


def rebalance_probs(probs: np.ndarray, target: np.ndarray | None = None, n_iter: int = 8) -> np.ndarray:
    q = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, None)
    k = q.shape[1]
    if target is None:
        target = np.full(k, 1.0 / k)
    target = np.asarray(target, dtype=np.float64)
    target = target / target.sum()
    for _ in range(n_iter):
        q = q / q.sum(axis=1, keepdims=True)
        col = q.mean(axis=0)
        q = q * (target / np.clip(col, 1e-12, None))
    return q / q.sum(axis=1, keepdims=True)


def _first_existing(paths: Iterable[str | Path]) -> Path | None:
    for p in paths:
        path = Path(p)
        if path.exists():
            return path
    return None


def find_train_path(extra: Sequence[str | Path] | None = None) -> Path:
    candidates = list(extra or [])
    candidates.extend(KAGGLE_TRAIN_CANDIDATES)
    found = _first_existing(candidates)
    if found is None:
        raise FileNotFoundError("train.csv not found")
    return found


def find_test_path(extra: Sequence[str | Path] | None = None) -> Path:
    candidates = list(extra or [])
    candidates.extend(KAGGLE_TEST_CANDIDATES)
    found = _first_existing(candidates)
    if found is None:
        raise FileNotFoundError("test.csv not found")
    return found


def load_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame()
    if "id" in cols:
        out["id"] = df[cols["id"]].astype(str)
    else:
        out["id"] = [f"row{i:06d}" for i in range(len(df))]
    if "text" not in cols:
        raise ValueError(f"missing text column in {list(df.columns)}")
    out["text"] = df[cols["text"]].fillna("").astype(str)
    if "model" in cols:
        out["model"] = df[cols["model"]].astype(str)
    if "prompt_id" in cols:
        out["prompt_id"] = df[cols["prompt_id"]].astype(str)
    else:
        out["prompt_id"] = out["id"]
    return out


def load_extra_frames(roots: Sequence[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    skip_names = {"train.csv", "test.csv", "sample_submission.parquet", "sample_submission.csv"}
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        paths = [root_path] if root_path.is_file() else list(root_path.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet"}:
                continue
            if path.name in skip_names:
                continue
            try:
                if path.stat().st_size > 80_000_000:
                    continue
            except OSError:
                continue
            try:
                df = normalize_frame(load_table(path))
            except Exception:
                continue
            if "model" not in df.columns:
                continue
            df = df[df["model"].isin(LABELS) & df["text"].str.len().gt(40)]
            if df.empty:
                continue
            frames.append(df[["id", "prompt_id", "text", "model"]])
    if not frames:
        return pd.DataFrame(columns=["id", "prompt_id", "text", "model"])
    extra = pd.concat(frames, ignore_index=True)
    extra = extra.drop_duplicates(subset=["text"])
    return extra.reset_index(drop=True)


def make_vectorizers() -> list[tuple[str, TfidfVectorizer, Any]]:
    # Character unigrams keep em dashes, quotes and other single-mark fingerprints.
    return [
        (
            "char",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(1, 5),
                min_df=2,
                max_features=300_000,
                sublinear_tf=True,
                lowercase=False,
            ),
            None,
        ),
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_features=200_000,
                sublinear_tf=True,
                lowercase=False,
                token_pattern=WORD_TOKEN_PATTERN,
            ),
            None,
        ),
        (
            "shape",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(2, 4),
                min_df=3,
                max_features=20_000,
                sublinear_tf=True,
                lowercase=False,
                token_pattern=r"\S+",
            ),
            to_word_shape,
        ),
    ]


def _transform_view(vec: TfidfVectorizer, texts: Sequence[str], preprocessor, fit: bool) -> csr_matrix:
    series = pd.Series(list(texts), dtype=str)
    if preprocessor is not None:
        series = series.map(preprocessor)
    if fit:
        return vec.fit_transform(series)
    return vec.transform(series)


def fit_mark_illegal(texts: Sequence[str], labels: Sequence[str]) -> dict[str, set[str]]:
    illegal: dict[str, set[str]] = {}
    frame = pd.DataFrame({"text": list(texts), "model": list(labels)})
    for name, pat in MARK_PATTERNS.items():
        rates = frame.groupby("model")["text"].apply(lambda s: s.str.contains(pat, regex=True).mean())
        illegal[name] = set(rates[rates <= 1e-12].index.tolist())
    return illegal


def apply_mark_penalties(
    probs: np.ndarray,
    texts: Sequence[str],
    classes: Sequence[str],
    illegal: dict[str, set[str]],
    factor: float = 0.08,
) -> np.ndarray:
    out = np.asarray(probs, dtype=np.float64).copy()
    class_index = {c: i for i, c in enumerate(classes)}
    for i, text in enumerate(texts):
        hits = mark_hits(text)
        for mark, present in hits.items():
            if not present:
                continue
            for lab in illegal.get(mark, ()):
                j = class_index.get(lab)
                if j is not None:
                    out[i, j] *= factor
        s = out[i].sum()
        if s > 0:
            out[i] /= s
    return out


def _logit() -> LogisticRegression:
    return LogisticRegression(
        C=10.0,
        max_iter=2500,
        class_weight="balanced",
        solver="lbfgs",
    )


def _map_proba(clf, x, all_names: Sequence[str]) -> np.ndarray:
    raw = clf.predict_proba(x)
    out = np.zeros((x.shape[0], len(all_names)), dtype=np.float64)
    idx = {str(c): i for i, c in enumerate(all_names)}
    for col, name in enumerate(clf.classes_):
        j = idx.get(str(name))
        if j is not None:
            out[:, j] = raw[:, col]
    s = out.sum(axis=1, keepdims=True)
    return np.divide(out, s, out=np.full_like(out, 1.0 / max(len(all_names), 1)), where=s > 0)


@dataclass
class NTLClassifier:
    extra_weight: float = 0.55
    blend_global: float = 1.0
    blend_hier: float = 0.0
    blend_style: float = 0.0
    rebalance: bool = False
    mark_factor: float = 0.25
    use_dense: bool = False
    classes_: np.ndarray = field(default_factory=lambda: np.array(LABELS))
    family_classes_: np.ndarray = field(default_factory=lambda: np.array(FAMILIES))

    def __post_init__(self) -> None:
        self.vectorizers_ = make_vectorizers()
        self.scaler_ = StandardScaler()
        self.global_clf_ = _logit()
        self.family_clf_ = _logit()
        self.within_: dict[str, LogisticRegression] = {}
        self.style_clf_ = None
        self.use_style_head_ = False
        self.illegal_marks_: dict[str, set[str]] = {}
        self.n_sparse_ = 0

    def _fit_sparse(self, texts: Sequence[str]) -> csr_matrix:
        blocks = []
        for _, vec, prep in self.vectorizers_:
            blocks.append(_transform_view(vec, texts, prep, fit=True))
        x = hstack(blocks).tocsr()
        self.n_sparse_ = x.shape[1]
        return x

    def _transform_sparse(self, texts: Sequence[str]) -> csr_matrix:
        blocks = []
        for _, vec, prep in self.vectorizers_:
            blocks.append(_transform_view(vec, texts, prep, fit=False))
        return hstack(blocks).tocsr()

    def _style_estimator(self, n_classes: int):
        try:
            import lightgbm as lgb

            return lgb.LGBMClassifier(
                n_estimators=240,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.8,
                min_child_samples=12,
                objective="multiclass",
                class_weight="balanced",
                verbosity=-1,
                n_jobs=-1,
            )
        except Exception:
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.08,
                max_iter=180,
                class_weight="balanced",
            )

    def fit(
        self,
        texts: Sequence[str],
        y: Sequence[str],
        sample_weight: np.ndarray | None = None,
    ) -> "NTLClassifier":
        texts = [str(t) for t in texts]
        y = np.asarray(y)
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=np.float64)
        present = set(y)
        self.classes_ = np.array([c for c in LABELS if c in present] + [c for c in present if c not in LABELS])
        fam = np.array([FAMILY_OF.get(v, "other") for v in y])

        x_sparse = self._fit_sparse(texts)
        if self.use_dense:
            x_dense = extract_dense(texts)
            x_dense_s = self.scaler_.fit_transform(x_dense)
            x_all = hstack([x_sparse, csr_matrix(x_dense_s)]).tocsr()
        else:
            x_all = x_sparse
            self.scaler_ = None

        self.global_clf_.fit(x_all, y, sample_weight=sample_weight)
        self.within_.clear()
        if self.blend_hier > 0:
            self.family_clf_.fit(x_all, fam, sample_weight=sample_weight)
            for family in FAMILIES:
                idx = np.where(fam == family)[0]
                labs = np.unique(y[idx])
                if len(labs) <= 1 or len(idx) < 8:
                    continue
                clf = _logit()
                clf.fit(x_all[idx], y[idx], sample_weight=sample_weight[idx])
                self.within_[family] = clf

        self.style_clf_ = None
        if self.blend_style > 0:
            self.style_clf_ = self._style_estimator(len(self.classes_))
            try:
                self.style_clf_.fit(x_dense_s, y, sample_weight=sample_weight)
            except TypeError:
                self.style_clf_.fit(x_dense_s, y)

        self.illegal_marks_ = fit_mark_illegal(texts, y)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        texts = [str(t) for t in texts]
        x_sparse = self._transform_sparse(texts)
        if self.use_dense and self.scaler_ is not None:
            x_dense = extract_dense(texts)
            x_dense_s = self.scaler_.transform(x_dense)
            x_all = hstack([x_sparse, csr_matrix(x_dense_s)]).tocsr()
        else:
            x_all = x_sparse
            x_dense_s = None

        class_names = [str(c) for c in self.classes_]
        p_global = _map_proba(self.global_clf_, x_all, class_names)
        p_hier = p_global
        if self.blend_hier > 0:
            p_fam = _map_proba(self.family_clf_, x_all, list(FAMILIES))
            p_hier = np.zeros_like(p_global)
            remaining: dict[str, list[str]] = {fam: [] for fam in FAMILIES}
            for lab in class_names:
                fam = FAMILY_OF.get(lab)
                if fam in remaining:
                    remaining[fam].append(lab)
            for family, labs in remaining.items():
                if not labs:
                    continue
                cols = [class_names.index(lab) for lab in labs]
                fam_col = list(FAMILIES).index(family)
                fam_p = p_fam[:, fam_col]
                if family in self.within_ and len(labs) > 1:
                    within_p = _map_proba(self.within_[family], x_all, labs)
                else:
                    within_p = np.full((len(texts), len(labs)), 1.0 / max(len(labs), 1))
                p_hier[:, cols] = within_p * fam_p[:, None]
            p_hier = p_hier / np.clip(p_hier.sum(axis=1, keepdims=True), 1e-12, None)

        if self.style_clf_ is not None and x_dense_s is not None and hasattr(self.style_clf_, "predict_proba"):
            p_style = _map_proba(self.style_clf_, x_dense_s, class_names)
        else:
            p_style = p_global

        w_g, w_h, w_s = self.blend_global, self.blend_hier, self.blend_style
        wsum = w_g + w_h + w_s
        probs = (w_g * p_global + w_h * p_hier + w_s * p_style) / wsum
        probs = apply_mark_penalties(probs, texts, class_names, self.illegal_marks_, self.mark_factor)
        if self.rebalance:
            probs = rebalance_probs(probs)
        return probs

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        probs = self.predict_proba(texts)
        idx = probs.argmax(axis=1)
        return np.asarray(self.classes_)[idx]


def grouped_cv(
    train: pd.DataFrame,
    extra: pd.DataFrame | None = None,
    n_splits: int = 5,
    extra_weight: float = 0.55,
    seed: int = 42,
) -> dict[str, Any]:
    y = train["model"].astype(str).to_numpy()
    groups = train["prompt_id"].astype(str).to_numpy()
    texts = train["text"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.empty(len(train), dtype=object)
    oof_prob = np.zeros((len(train), len(LABELS)), dtype=np.float64)
    fold_scores: list[float] = []
    extra = extra if extra is not None and len(extra) else pd.DataFrame()

    for fold, (trn, val) in enumerate(splitter.split(train, y, groups=groups), start=1):
        tr_texts = list(texts[trn])
        tr_y = list(y[trn])
        weights = [1.0] * len(tr_texts)
        if len(extra):
            tr_texts.extend(extra["text"].astype(str).tolist())
            tr_y.extend(extra["model"].astype(str).tolist())
            weights.extend([extra_weight] * len(extra))
        model = NTLClassifier(extra_weight=extra_weight)
        model.fit(tr_texts, tr_y, sample_weight=np.asarray(weights, dtype=np.float64))
        val_texts = list(texts[val])
        proba = model.predict_proba(val_texts)
        pred = np.asarray(model.classes_)[proba.argmax(axis=1)]
        oof_pred[val] = pred
        col = {str(c): i for i, c in enumerate(model.classes_)}
        aligned = np.zeros((len(val), len(LABELS)))
        for j, lab in enumerate(LABELS):
            if lab in col:
                aligned[:, j] = proba[:, col[lab]]
        oof_prob[val] = aligned
        score = balanced_accuracy(y[val], pred)
        fold_scores.append(score)
        print(f"fold {fold}: balanced_acc={score:.4f}", flush=True)

    overall = balanced_accuracy(y, oof_pred)
    per_class = {}
    for lab in LABELS:
        mask = y == lab
        per_class[lab] = float((oof_pred[mask] == lab).mean()) if mask.any() else 0.0
    return {
        "overall": overall,
        "fold_scores": fold_scores,
        "per_class": per_class,
        "oof_pred": oof_pred,
        "oof_prob": oof_prob,
        "n_train": int(len(train)),
        "n_extra": int(len(extra)),
    }


def write_submission(ids: Sequence[str], models: Sequence[str], path: str | Path) -> Path:
    path = Path(path)
    sub = pd.DataFrame({"id": list(ids), "model": list(models)})
    if sub["id"].duplicated().any():
        raise ValueError("duplicate ids in submission")
    if sub.isnull().any().any():
        raise ValueError("nulls in submission")
    bad = set(sub["model"]) - set(LABELS)
    if bad:
        raise ValueError(f"invalid labels: {sorted(bad)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(path, index=False)
    return path


def make_synthetic(n_prompts: int = 8, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    topics = [
        "the heat death of the universe",
        "Banach-Tarski paradox",
        "why inner ears sense rotation",
        "the illusion of explanatory depth",
        "neutron star crusts",
        "language death and star names",
        "data science communities",
        "stress and motivated reasoning",
        "Jevons paradox",
        "Stern-Gerlach experiment",
        "allopatric speciation",
        "type-token curves in small corpora",
    ]
    rows = []
    for p in range(n_prompts):
        topic = topics[p % len(topics)]
        pid = f"p{p:03d}"
        for model in LABELS:
            body = _synthetic_passage(model, topic, rng)
            rows.append(
                {
                    "id": f"{pid}-{model}",
                    "prompt_id": pid,
                    "text": body,
                    "model": model,
                }
            )
    train = pd.DataFrame(rows)
    test = train.sample(n=min(12, len(train)), random_state=seed).copy()
    test = test[["id", "text"]].reset_index(drop=True)
    return train, test


def _synthetic_passage(model: str, topic: str, rng: np.random.Generator) -> str:
    family = FAMILY_OF[model]
    n_words = int(rng.integers(45, 90))
    base = (
        f"A careful account of {topic} begins with the parts that actually do work, "
        f"not with a slogan. The mechanism is local, the consequences are global, "
        f"and the usual shortcuts hide the joints where the argument can break."
    )
    extra = []
    if family == "anthropic":
        extra.append("The claim is modest — the supporting structure is not.")
        extra.append("What remains is a straight path from evidence to consequence.")
    elif family == "google":
        extra.append("The tapestry of causes is more intricate than a single lever.")
        extra.append("In this landscape the key constraint is geometric, not verbal.")
    elif family == "openai":
        extra.append("That is, the same pattern shows up in small cases and in large ones.")
        extra.append("For instance, the counting argument and the geometric argument agree.")
    elif family == "qwen":
        extra.append("In other words, the bound is tight, and the examples are not decorative.")
    elif family == "zai":
        extra.append("The remaining uncertainty is practical — not conceptual.")
    else:
        extra.append("Hence the useful move is to keep the invariant in view.")

    if model.startswith("gpt-"):
        extra.append("Numbers, names, and commas do real work here, 1922, 5, and 18 alike.")
        text = base + " " + " ".join(extra)
        text = text.replace("'", "’")
        if "gpt-5.5" in model:
            text = text.replace(" — ", ", and ")
    elif model.startswith("gemini-"):
        text = (base + " " + " ".join(extra)).replace("—", ",").replace("’", "'")
        text = text.replace("n't", "not")
    elif model.startswith("claude-"):
        text = base + " " + " ".join(extra)
        text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    else:
        text = base + " " + " ".join(extra)

    words = text.split()
    if len(words) < n_words:
        words = words + (["the", "same", "structure", "repeats"] * ((n_words - len(words)) // 4 + 1))
    text = " ".join(words[: max(n_words, 40)])
    if not text.endswith("."):
        text += "."
    return text
