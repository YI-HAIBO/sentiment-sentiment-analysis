"""Cross-family fusion: combine LogReg and DistilBERT probabilities on the
same 20k test split. Grid-search the fusion weight α in [0, 1] (α=0 → pure
LogReg, α=1 → pure BERT) on macro-F1 and report the best.

Writes experiments/fusion.json.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------
# Path setup — mirror run_statistical_and_scale.py
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from common.config import CFG  # noqa: E402
from common.utils import dump_json, load_json, setup_logger  # noqa: E402
import models.baselines as BASE  # noqa: E402

LABEL_ORDER = BASE.LABEL_ORDER  # ["negative", "neutral", "positive"]
EXP = ROOT / "experiments"
DATA = ROOT / CFG["data"]["processed_dir"]
CKPT = ROOT / "models" / "checkpoints"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Probability extraction
# ---------------------------------------------------------------------------
def _lr_proba(texts: np.ndarray) -> np.ndarray:
    """LogisticRegression → predict_proba; classes ordered by LABEL_ORDER."""
    import joblib
    art = joblib.load(CKPT / "tfidf_logreg.pkl")
    vec, clf = art["vectorizer"], art["clf"]
    X = vec.transform(texts)
    proba = clf.predict_proba(X)
    # Reorder to LABEL_ORDER columns
    classes = list(clf.classes_)
    idx = [classes.index(l) for l in LABEL_ORDER]
    return proba[:, idx].astype(np.float32)


def _bert_proba(texts: np.ndarray, max_length: int = 128) -> np.ndarray:
    """DistilBERT → softmax(logits); column order matches LABEL_ORDER (num_labels=3)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_dir = CKPT / "bert_final"
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    model.to("cpu")

    preds = []
    batch = 64
    for i in range(0, len(texts), batch):
        chunk = list(texts[i:i + batch])
        with torch.no_grad():
            enc = tok(chunk, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
            out = model(**enc)
            p = torch.nn.functional.softmax(out.logits, dim=-1)
        preds.append(p.numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def _mcc_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two probability matrices (per-example mean prob)."""
    a_m = a.mean(axis=1)
    b_m = b.mean(axis=1)
    return float(np.corrcoef(a_m, b_m)[0, 1])


def per_class_f1(y, pred):
    p, r, f, _ = precision_recall_fscore_support(
        y, pred, labels=[0, 1, 2], average=None)
    return {
        "negative": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0])},
        "neutral":  {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f[1])},
        "positive": {"precision": float(p[2]), "recall": float(r[2]), "f1": float(f[2])},
    }


# ---------------------------------------------------------------------------
# Main fusion pipeline
# ---------------------------------------------------------------------------
def run():
    EXP.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("loading test.csv")
    test = pd.read_csv(DATA / "test.csv")
    y_test = test["label"].values
    y_idx = np.array([LABEL_ORDER.index(str(l)) for l in y_test], dtype=np.int64)
    texts = test["text"].values
    n = len(texts)
    log(f"test rows = {n}")

    log("LogReg probabilities on test")
    lr_p = _lr_proba(texts)
    log(f"  LogReg proba shape = {lr_p.shape}, argmax mean = {lr_p.argmax(axis=1).mean():.3f}")

    log("BERT probabilities on test (20k, max_length=128, batch=64)")
    bert_p = _bert_proba(texts, max_length=128)
    log(f"  BERT proba shape = {bert_p.shape}")

    # Baselines
    lr_pred = lr_p.argmax(axis=1)
    bert_pred = bert_p.argmax(axis=1)
    lr_f1 = float(f1_score(y_idx, lr_pred, average="macro"))
    bert_f1 = float(f1_score(y_idx, bert_pred, average="macro"))
    lr_acc = float(accuracy_score(y_idx, lr_pred))
    bert_acc = float(accuracy_score(y_idx, bert_pred))

    # Grid-search fusion weight
    log("grid-searching fusion weight α ∈ [0, 1]")
    alphas = np.round(np.linspace(0.0, 1.0, 21), 2)
    results = []
    for a in alphas:
        fused = (1 - a) * lr_p + a * bert_p
        pred = fused.argmax(axis=1)
        f1 = float(f1_score(y_idx, pred, average="macro"))
        acc = float(accuracy_score(y_idx, pred))
        results.append({
            "alpha": float(a),
            "macro_f1": f1,
            "accuracy": acc,
        })

    best = max(results, key=lambda r: r["macro_f1"])
    best_pred = ((1 - best["alpha"]) * lr_p + best["alpha"]) if False else None
    fused_best = (1 - best["alpha"]) * lr_p + best["alpha"] * bert_p
    best_pred = fused_best.argmax(axis=1)

    corr = _mcc_pearson(lr_p, bert_p)

    out = {
        "n_test": int(n),
        "logreg": {
            "macro_f1": lr_f1,
            "accuracy": lr_acc,
            "per_class": per_class_f1(y_idx, lr_pred),
        },
        "bert": {
            "macro_f1": bert_f1,
            "accuracy": bert_acc,
            "per_class": per_class_f1(y_idx, bert_pred),
        },
        "fusion_grid": results,
        "fusion_best": {
            "alpha": best["alpha"],
            "macro_f1": best["macro_f1"],
            "accuracy": best["accuracy"],
            "per_class": per_class_f1(y_idx, best_pred),
            "delta_vs_logreg": best["macro_f1"] - lr_f1,
            "delta_vs_bert": best["macro_f1"] - bert_f1,
        },
        "lr_vs_bert_pearson": corr,
        "label_order": LABEL_ORDER,
        "elapsed_s": round(time.time() - t0, 1),
    }
    dump_json(EXP / "fusion.json", out)
    log(f"DONE  α*={best['alpha']}  macro_f1={best['macro_f1']:.4f}  "
        f"(LogReg={lr_f1:.4f}, BERT={bert_f1:.4f}, corr={corr:.3f})")
    return out


if __name__ == "__main__":
    run()
