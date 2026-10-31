"""Revision experiment D1 (R1-6): re-select the fusion weight alpha on a
VALIDATION split instead of the test split.

The published protocol grid-searched alpha on the 20,000-row test split
(experiments/fusion.json).  This script:

  1. recomputes the published DistilBERT checkpoint's probabilities on the
     20,000-row validation split AND on the test split (cached as npy),
  2. computes TF-IDF+LogReg probabilities on the same splits from the released
     pickle,
  3. grid-searches alpha on VALIDATION only,
  4. evaluates the single val-selected alpha* once on the untouched test split,
  5. reports the old test-selected protocol alongside for transparency, and
     paired bootstrap CIs for the fusion gain.

Writes:
  experiments/bert_val_proba.npy      (20k, 3)
  experiments/bert_test_proba.npy     (20k, 3)
  experiments/fusion_val.json

Usage   python scripts/fusion_val.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from common.config import CFG  # noqa: E402

DATA = ROOT / CFG["data"]["processed_dir"]
EXP = ROOT / "experiments"
CKPT = ROOT / "models" / "checkpoints"
LABELS = ["negative", "neutral", "positive"]
N_BOOT = 2000

# the published checkpoint lives in the release bundle
RELEASE_CKPT = (ROOT.parent / "sentiment-system-release" / "models" /
                "checkpoints" / "bert_final")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def y_of(df: pd.DataFrame) -> np.ndarray:
    return np.array([LABELS.index(str(l)) for l in df["label"]], dtype=np.int64)


def lr_proba(texts) -> np.ndarray:
    import joblib
    art = joblib.load(CKPT / "tfidf_logreg.pkl")
    vec, clf = art["vectorizer"], art["clf"]
    X = vec.transform(texts)
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)
    idx = [classes.index(l) for l in LABELS]
    return proba[:, idx].astype(np.float64)


def bert_proba(texts, model, tok, batch: int = 128) -> np.ndarray:
    import torch
    preds = []
    for i in range(0, len(texts), batch):
        chunk = list(texts[i:i + batch])
        with torch.no_grad():
            enc = tok(chunk, padding=True, truncation=True,
                      max_length=128, return_tensors="pt")
            out = model(**enc)
            p = torch.nn.functional.softmax(out.logits, dim=-1)
        preds.append(p.numpy().astype(np.float64))
    return np.concatenate(preds, axis=0)


def macro_f1_from_conf(conf: np.ndarray) -> float:
    f1s = []
    for k in range(3):
        tp = conf[k, k]
        fp = conf[:, k].sum() - tp
        fn = conf[k, :].sum() - tp
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def grid_alpha(p1, p2, y):
    rows = []
    for a in np.round(np.arange(0.0, 1.01, 0.05), 2):
        pred = ((1 - a) * p1 + a * p2).argmax(1)
        rows.append({"alpha": float(a),
                     "macro_f1": float(f1_score(y, pred, average="macro",
                                                zero_division=0)),
                     "accuracy": float(accuracy_score(y, pred))})
    return rows


def paired_ci(y, preds: dict, n_boot: int = N_BOOT, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    codes = {k: (y * 3 + v) for k, v in preds.items()}
    draws = {k: np.empty(n_boot) for k in preds}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for k in preds:
            cnt = np.bincount(codes[k][idx], minlength=9).reshape(3, 3)
            draws[k][b] = macro_f1_from_conf(cnt)
    ref = draws["logreg"]
    out = {}
    for k in preds:
        gain = draws[k] - ref
        out[k] = {
            "macro_f1_mean": float(draws[k].mean()),
            "ci95": [float(np.percentile(draws[k], 2.5)),
                     float(np.percentile(draws[k], 97.5))],
            "gain_vs_logreg_mean": float(gain.mean()),
            "gain_ci95": [float(np.percentile(gain, 2.5)),
                          float(np.percentile(gain, 97.5))],
        }
    return out


def main() -> None:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    t0 = time.time()
    torch.set_num_threads(14)
    val = pd.read_csv(DATA / "val.csv")
    test = pd.read_csv(DATA / "test.csv")
    y_val, y_test = y_of(val), y_of(test)
    log(f"val={len(val)} test={len(test)}")

    # --- BERT probabilities (cached) ---
    val_path, test_path = EXP / "bert_val_proba.npy", EXP / "bert_test_proba.npy"
    if val_path.exists():
        bp_val = np.load(val_path)
        log("cached bert_val_proba.npy")
    else:
        ckpt = RELEASE_CKPT if RELEASE_CKPT.exists() else CKPT / "bert_final"
        log(f"loading BERT checkpoint: {ckpt}")
        tok = AutoTokenizer.from_pretrained(str(ckpt))
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
        model.eval()
        t1 = time.time()
        bp_val = bert_proba(val["text"].tolist(), model, tok)
        np.save(val_path, bp_val)
        log(f"BERT val proba done in {time.time()-t1:.0f}s")
    if test_path.exists():
        bp_test = np.load(test_path)
        log("cached bert_test_proba.npy")
    else:
        ckpt = RELEASE_CKPT if RELEASE_CKPT.exists() else CKPT / "bert_final"
        tok = AutoTokenizer.from_pretrained(str(ckpt))
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
        model.eval()
        t1 = time.time()
        bp_test = bert_proba(test["text"].tolist(), model, tok)
        np.save(test_path, bp_test)
        log(f"BERT test proba done in {time.time()-t1:.0f}s")

    # --- LogReg probabilities ---
    lp_val = lr_proba(val["text"].tolist())
    lp_test = lr_proba(test["text"].tolist())

    log(f"val  macro-F1: LogReg={f1_score(y_val, lp_val.argmax(1), average='macro'):.4f}"
        f"  BERT={f1_score(y_val, bp_val.argmax(1), average='macro'):.4f}")
    log(f"test macro-F1: LogReg={f1_score(y_test, lp_test.argmax(1), average='macro'):.4f}"
        f"  BERT={f1_score(y_test, bp_test.argmax(1), average='macro'):.4f}")

    # --- grid on validation, single evaluation on test ---
    val_grid = grid_alpha(lp_val, bp_val, y_val)
    test_grid = grid_alpha(lp_test, bp_test, y_test)
    best_val = max(val_grid, key=lambda r: r["macro_f1"])
    best_test = max(test_grid, key=lambda r: r["macro_f1"])
    a_star = best_val["alpha"]

    fused_test = ((1 - a_star) * lp_test + a_star * bp_test).argmax(1)
    fused_val = ((1 - a_star) * lp_val + a_star * bp_val).argmax(1)

    from sklearn.metrics import precision_recall_fscore_support
    pc = precision_recall_fscore_support(y_test, fused_test,
                                         labels=[0, 1, 2], average=None)

    # --- paired bootstrap on test ---
    ci = paired_ci(y_test, {
        "logreg": lp_test.argmax(1),
        "bert": bp_test.argmax(1),
        f"fused_alpha_val_{a_star:g}": fused_test,
    })

    out = {
        "n_val": int(len(val)), "n_test": int(len(test)),
        "protocol": ("alpha grid-searched on the 20k validation split, then the "
                     "single val-selected alpha* is evaluated once on the "
                     "untouched 20k test split (Reviewer 1 comment 6)."),
        "val_grid": val_grid,
        "test_grid": test_grid,
        "alpha_val_selected": a_star,
        "alpha_test_selected_OLD_PROTOCOL": best_test["alpha"],
        "test_at_alpha_val": {
            "alpha": a_star,
            "macro_f1": float(f1_score(y_test, fused_test, average="macro")),
            "accuracy": float(accuracy_score(y_test, fused_test)),
            "macro_f1_val": float(f1_score(y_val, fused_val, average="macro")),
            "per_class": {LABELS[i]: {"precision": float(pc[0][i]),
                                      "recall": float(pc[1][i]),
                                      "f1": float(pc[2][i])} for i in range(3)},
        },
        "bootstrap_paired_test": ci,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(EXP / "fusion_val.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"DONE alpha*={a_star} (val)  test macro-F1="
        f"{out['test_at_alpha_val']['macro_f1']:.4f}  "
        f"(old test-selected protocol: alpha={best_test['alpha']} -> "
        f"{best_test['macro_f1']:.4f})  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
