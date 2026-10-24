"""Qwen LoRA classifier for Name That LLM (Kaggle GPU, internet off)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from .constants import LABELS
from .pipeline import apply_mark_penalties, fit_mark_illegal, rebalance_probs

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except Exception:  # pragma: no cover - imported on Kaggle GPU images
    LoraConfig = TaskType = get_peft_model = prepare_model_for_kbit_training = None


QWEN_HINTS = ("qwen3", "qwen2", "qwen")
MAX_LEN = 384


def find_base_model(roots: Sequence[str | Path] | None = None) -> Path:
    search = [Path(p) for p in (roots or [])]
    search.extend(
        [
            Path("/kaggle/input"),
            Path("models"),
        ]
    )
    ranked: list[tuple[int, Path]] = []
    seen = set()
    for root in search:
        if not root.exists():
            continue
        for cfg in root.rglob("config.json"):
            if "adapter" in str(cfg).lower():
                continue
            try:
                data = json.loads(cfg.read_text())
            except Exception:
                continue
            model_type = str(data.get("model_type", "")).lower()
            arch = " ".join(map(str, data.get("architectures", []))).lower()
            blob = model_type + " " + arch + " " + str(cfg).lower()
            if not any(h in blob for h in QWEN_HINTS):
                continue
            parent = cfg.parent.resolve()
            if parent in seen:
                continue
            seen.add(parent)
            hidden = int(data.get("hidden_size") or data.get("n_embd") or 0)
            ranked.append((hidden, parent))
    if not ranked:
        raise FileNotFoundError(
            "No Qwen base model found. Attach qwen-lm/qwen-3 Transformers 8b-base "
            "(Kaggle Models) with internet off."
        )
    ranked.sort(reverse=True)
    return ranked[0][1]


class PassageDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int] | None, tokenizer, max_len: int = MAX_LEN):
        self.texts = [str(t) for t in texts]
        self.labels = None if labels is None else [int(x) for x in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _label_ids(y: Sequence[str]) -> list[int]:
    index = {lab: i for i, lab in enumerate(LABELS)}
    return [index[str(v)] for v in y]


def _prepare_tokenizer(model_path: Path):
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def _bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


class _PooledHead(torch.nn.Module):
    def __init__(self, backbone, hidden: int, n_labels: int):
        super().__init__()
        self.backbone = backbone
        self.score = torch.nn.Linear(hidden, n_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        logits = self.score(pooled.float())
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return type("Out", (), {"loss": loss, "logits": logits})


def build_model(model_path: Path, n_labels: int = 18, use_4bit: bool = True):
    tok = _prepare_tokenizer(model_path)
    max_mem = {0: "13GiB", 1: "13GiB", "cpu": "24GiB"} if torch.cuda.device_count() > 1 else None
    load_kw = dict(torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
    if max_mem is not None:
        load_kw["max_memory"] = max_mem
    if use_4bit:
        load_kw["quantization_config"] = _bnb_config()
    if LoraConfig is None:
        raise RuntimeError("peft is required for LoRA training")
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            num_labels=n_labels,
            pad_token_id=tok.pad_token_id,
            **load_kw,
        )
        model.config.pad_token_id = tok.pad_token_id
        model.config.problem_type = "single_label_classification"
        if use_4bit and prepare_model_for_kbit_training is not None:
            model = prepare_model_for_kbit_training(model)
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            target_modules=targets,
            modules_to_save=["score", "classifier"],
        )
        model = get_peft_model(model, lora)
    except Exception as exc:
        print(f"SEQ_CLS path failed ({exc}); using mean-pool head", flush=True)
        backbone = AutoModel.from_pretrained(str(model_path), **load_kw)
        if use_4bit and prepare_model_for_kbit_training is not None:
            backbone = prepare_model_for_kbit_training(backbone)
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=targets,
        )
        backbone = get_peft_model(backbone, lora)
        hidden = int(AutoConfig.from_pretrained(str(model_path), trust_remote_code=True).hidden_size)
        model = _PooledHead(backbone, hidden, n_labels)
        if torch.cuda.is_available():
            model.score = model.score.to(next(backbone.parameters()).device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model, tok, use_4bit


def _class_weights(y_ids: Sequence[int], n_labels: int = 18) -> torch.Tensor:
    counts = np.bincount(list(y_ids), minlength=n_labels).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_labels * counts)
    return torch.tensor(w, dtype=torch.float32)


def _move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _model_device(model) -> torch.device:
    if hasattr(model, "backbone"):
        return next(model.backbone.parameters()).device
    return next(model.parameters()).device


def train_epochs(
    model,
    tok,
    texts: Sequence[str],
    labels: Sequence[str],
    epochs: int,
    lr: float,
    batch_size: int = 1,
    accum: int = 16,
    class_weight: torch.Tensor | None = None,
) -> None:
    y_ids = _label_ids(labels)
    ds = PassageDataset(texts, y_ids, tok)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    device = _model_device(model)
    if class_weight is not None:
        class_weight = class_weight.to(device)
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    steps = 0
    opt.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        running = 0.0
        n = 0
        for batch in loader:
            batch = _move(batch, device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = out.logits
                loss = F.cross_entropy(logits, batch["labels"], weight=class_weight)
                loss = loss / accum
            scaler.scale(loss).backward()
            running += float(loss.item()) * accum
            n += 1
            steps += 1
            if steps % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        if steps % accum != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        print(f"epoch {epoch + 1}/{epochs} mean_loss={running / max(n, 1):.4f}", flush=True)


@torch.no_grad()
def predict_proba(model, tok, texts: Sequence[str], batch_size: int = 4) -> np.ndarray:
    model.eval()
    device = _model_device(model)
    ds = PassageDataset(texts, None, tok)
    # length sort for fewer pads on the hidden rerun
    order = np.argsort([len(t.split()) for t in texts])
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    ordered = [texts[i] for i in order]
    ds.texts = [str(t) for t in ordered]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    for batch in loader:
        batch = _move(batch, device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        chunks.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    probs = np.concatenate(chunks, axis=0)[inv]
    return probs


def two_stage_train(
    model_path: Path,
    train: pd.DataFrame,
    extra: pd.DataFrame | None = None,
    epochs_extra: int = 1,
    epochs_host: int = 2,
) -> tuple:
    model, tok, _ = build_model(model_path)
    y_host = train["model"].astype(str).tolist()
    weights = _class_weights(_label_ids(y_host))
    if extra is not None and len(extra):
        print(f"LoRA stage 1: extra n={len(extra)}", flush=True)
        train_epochs(
            model,
            tok,
            extra["text"].astype(str).tolist(),
            extra["model"].astype(str).tolist(),
            epochs=epochs_extra,
            lr=2.0e-4,
            class_weight=weights,
        )
    print(f"LoRA stage 2: host n={len(train)}", flush=True)
    train_epochs(
        model,
        tok,
        train["text"].astype(str).tolist(),
        y_host,
        epochs=epochs_host if extra is not None and len(extra) else max(epochs_host, 3),
        lr=8.0e-5,
        class_weight=weights,
    )
    return model, tok


def blend_and_penalize(
    p_lora: np.ndarray,
    p_tfidf: np.ndarray | None,
    texts: Sequence[str],
    train_texts: Sequence[str],
    train_y: Sequence[str],
    lora_weight: float = 0.78,
    rebalance: bool = True,
) -> np.ndarray:
    if p_tfidf is None:
        probs = np.asarray(p_lora, dtype=np.float64)
    else:
        w = lora_weight
        probs = w * np.asarray(p_lora, dtype=np.float64) + (1.0 - w) * np.asarray(p_tfidf, dtype=np.float64)
    illegal = fit_mark_illegal(train_texts, train_y)
    probs = apply_mark_penalties(probs, texts, list(LABELS), illegal, factor=0.18)
    if rebalance:
        probs = rebalance_probs(probs)
    return probs
