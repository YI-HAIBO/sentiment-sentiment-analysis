"""
Statistical + scale experiments on top of the baseline pipeline:

    bootstrap       -- Evaluate LogReg and BERT on the same 20k test set.
                       Report point macro-F1, paired bootstrap 95% CI on the
                       difference, and McNemar's exact test.
    bert-multiseed  -- Rerun DistilBERT @ 4k with 3 different random seeds.
                       Report per-seed macro-F1 and mean/std.
    scale-sweep     -- Train TF-IDF+LogReg and DistilBERT at n_train in
                       {4k, 8k, 16k} (32k optional), on the same 20k test.
                       Answers: does the crossover happen somewhere?

Usage:
    python scripts/run_statistical_and_scale.py bootstrap
    python scripts/run_statistical_and_scale.py bert-multiseed
    python scripts/run_statistical_and_scale.py scale-sweep tfidf
    python scripts/run_statistical_and_scale.py scale-sweep bert
    python scripts/run_statistical_and_scale.py all

Outputs land in experiments/:
    bootstrap.json
    bert_multiseed.json
    scale_sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from common.config import CFG  # noqa: E402
from common.utils import dump_json, load_json, setup_logger  # noqa: E402
import models.bert as BERT  # noqa: E402
import models.baselines as BASE  # noqa: E402


LABEL_ORDER = BASE.LABEL_ORDER  # ["negative", "neutral", "positive"]
EXP = ROOT / "experiments"
CKPT = ROOT / "models" / "checkpoints"
DATA = ROOT / CFG["data"]["processed_dir"]


# ============================================================
# 1. BOOTSTRAP CI + MCNEMAR
# ============================================================
def _bert_predict(model_dir: Path, texts: list[str], max_length: int = 128) -> np.ndarray:
    """Load the saved BERT model + tokenizer and return predicted class indices."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    n = len(texts)
    bs = 64
    preds = np.empty(n, dtype=np.int64)
    with torch.no_grad():
        for i in range(0, n, bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
            logits = model(**enc).logits
            preds[i:i + bs] = logits.argmax(dim=1).cpu().numpy()
    return preds


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_labels: int = 3) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=list(range(n_labels)), average="macro"))


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, n_labels: int = 3) -> np.ndarray:
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred, labels=list(range(n_labels)))


def run_bootstrap(n_boot: int = 5000, seed: int = 42) -> dict:
    """Paired bootstrap on the same 20k test set for LogReg vs BERT."""
    log = setup_logger("bootstrap")
    test = pd.read_csv(DATA / "test.csv")
    texts = test["text"].tolist()
    y_test = test["label"].values
    y_idx_arr = np.array([LABEL_ORDER.index(y) for y in y_test])
    n = len(test)

    # --- LogReg (loaded from pkl) ---
    # Note: sklearn clf.predict() returns whatever label format was used at fit.
    # Normalize to int class indices [0, 1, 2] so bootstrap math is consistent.
    art_lr = joblib.load(CKPT / "tfidf_logreg.pkl")
    X_te = art_lr["vectorizer"].transform(texts)
    lr_pred_raw = art_lr["clf"].predict(X_te)
    lr_pred_raw = np.asarray(lr_pred_raw)
    if lr_pred_raw.dtype.kind in "OUS":  # object / unicode / bytes — treat as strings
        lr_pred_idx = np.array([LABEL_ORDER.index(str(y)) for y in lr_pred_raw], dtype=np.int64)
    else:
        lr_pred_idx = lr_pred_raw.astype(np.int64)

    # --- BERT (loaded from bert_final/) ---
    # BERT was fit with class-weights keyed by integer indices, so argmax already returns 0..2.
    bert_dir = CKPT / "bert_final"
    if not bert_dir.exists():
        raise SystemExit("bert_final/ not found. Train BERT first.")
    bert_pred_idx = np.asarray(_bert_predict(bert_dir, texts, max_length=128), dtype=np.int64)

    # Point estimates
    lr_mf1 = _macro_f1(y_idx_arr, lr_pred_idx)
    bert_mf1 = _macro_f1(y_idx_arr, bert_pred_idx)
    diff = lr_mf1 - bert_mf1

    # Per-class F1 (use precision_recall_fscore_support for stability)
    from sklearn.metrics import precision_recall_fscore_support
    pr_lr, _, f_lr, _ = precision_recall_fscore_support(y_idx_arr, lr_pred_idx,
                                                          labels=list(range(3)), average=None)
    pr_bert, _, f_bert, _ = precision_recall_fscore_support(y_idx_arr, bert_pred_idx,
                                                              labels=list(range(3)), average=None)
    per_class = {
        lab: {"lr_f1": round(float(f_lr[i]), 4),
              "bert_f1": round(float(f_bert[i]), 4)}
        for i, lab in enumerate(LABEL_ORDER)
    }

    # --- McNemar's exact test on per-example correctness ---
    lr_correct = lr_pred_idx == y_idx_arr
    bert_correct = bert_pred_idx == y_idx_arr
    a = int(np.sum(lr_correct & bert_correct))
    b = int(np.sum(lr_correct & ~bert_correct))  # LR only correct
    c = int(np.sum(~lr_correct & bert_correct))  # BERT only correct
    d = int(np.sum(~lr_correct & ~bert_correct))
    from scipy.stats import fisher_exact
    odds, p_exact = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    # Also chi2 approximation (with continuity correction) for the classic McNemar
    from scipy.stats import chisquare
    n_diff = b + c
    if n_diff >= 1:
        stat = (abs(b - c) - 1) ** 2 / max(1, n_diff)
    else:
        stat = 0.0
    from scipy.stats import chi2 as chi2_dist
    p_chi2 = 1 - float(chi2_dist.cdf(stat, 1))

    # --- Paired bootstrap CI on the macro-F1 difference ---
    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_boot, dtype=np.float64)
    boot_lr = np.empty(n_boot)
    boot_bert = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_lr[i] = _macro_f1(y_idx_arr[idx], lr_pred_idx[idx])
        boot_bert[i] = _macro_f1(y_idx_arr[idx], bert_pred_idx[idx])
        boot_diffs[i] = boot_lr[i] - boot_bert[i]

    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    pct_pos = float(np.mean(boot_diffs > 0))

    log.info(
        f"n={n}  LogReg macro_F1={lr_mf1:.4f}  BERT macro_F1={bert_mf1:.4f}  "
        f"diff={diff:+.4f}  95% CI=[{ci_low:+.4f}, {ci_high:+.4f}]  "
        f"P(diff>0)={pct_pos:.3f}  McNemar exact p={p_exact:.3g}  chi2 p={p_chi2:.3g}"
    )

    results = {
        "n_test": n,
        "n_boot": n_boot,
        "seed": seed,
        "lr_macro_f1": round(lr_mf1, 4),
        "bert_macro_f1": round(bert_mf1, 4),
        "observed_diff": round(diff, 4),
        "bootstrap": {
            "diff_mean": round(float(boot_diffs.mean()), 4),
            "diff_std": round(float(boot_diffs.std(ddof=1)), 4),
            "ci_95_low": round(float(ci_low), 4),
            "ci_95_high": round(float(ci_high), 4),
            "diff_min": round(float(boot_diffs.min()), 4),
            "diff_max": round(float(boot_diffs.max()), 4),
            "pct_diff_positive": round(pct_pos, 4),
        },
        "bootstrap_lr": {
            "mean": round(float(boot_lr.mean()), 4),
            "std": round(float(boot_lr.std(ddof=1)), 4),
            "ci_95": [round(float(np.percentile(boot_lr, 2.5)), 4),
                      round(float(np.percentile(boot_lr, 97.5)), 4)],
        },
        "bootstrap_bert": {
            "mean": round(float(boot_bert.mean()), 4),
            "std": round(float(boot_bert.std(ddof=1)), 4),
            "ci_95": [round(float(np.percentile(boot_bert, 2.5)), 4),
                      round(float(np.percentile(boot_bert, 97.5)), 4)],
        },
        "mcnemar": {
            "both_correct": a,
            "only_lr_correct": b,
            "only_bert_correct": c,
            "both_wrong": d,
            "n_disagreeing": b + c,
            "fisher_exact_p": float(p_exact),
            "mcnemar_chi2_stat": float(stat),
            "mcnemar_chi2_p": float(p_chi2),
            "significant_at_005": (p_exact < 0.05),
        },
        "per_class": per_class,
        "confusion_lr": _confusion(y_idx_arr, lr_pred_idx).tolist(),
        "confusion_bert": _confusion(y_idx_arr, bert_pred_idx).tolist(),
        "label_order": LABEL_ORDER,
    }
    dump_json(EXP / "bootstrap.json", results)
    return results


# ============================================================
# 2. BERT MULTI-SEED
# ============================================================
def run_bert_multiseed(seeds: tuple[int, ...] = (42, 7, 123)) -> dict:
    """Rerun BERT at 4k training with different seeds.

    NOTE: BERT.train() reads its seed from CFG["bert"]["seed"], not CFG["seed"].
    Set both to be safe (top-level seed affects any downstream sampler
    fallbacks, the bert-level seed is what actually re-seeds RNG).
    """
    log = setup_logger("bert_multiseed")
    per_seed = []
    old_top_seed = CFG["seed"]
    old_bcfg = deepcopy(CFG["bert"])
    CFG["bert"]["max_length"] = 128
    CFG["bert"]["learning_rate"] = 2e-5
    CFG["bert"]["batch_size"] = 16

    class _A:
        quick = False

    try:
        for seed in seeds:
            CFG["seed"] = seed
            CFG["bert"]["seed"] = seed
            # Verify the seed is actually in place before training starts
            import torch as _torch
            log.info(
                f"Seed set: top={CFG['seed']} bert={CFG['bert']['seed']}  "
                f"torch.initial_seed()={_torch.initial_seed()}"
            )
            hist_path = EXP / f"bert_seed_{seed}_history.json"
            out_dir = CKPT / f"bert_seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            summary = BERT.train(_A(), "cpu", log, out_dir, hist_path)
            res = summary.get("test", {})
            entry = {
                "seed": seed,
                "n_train": summary.get("n_train", 0),
                "n_test": summary.get("n_test", 0),
                "accuracy": res.get("accuracy", 0.0),
                "macro_f1": res.get("macro_f1", 0.0),
                "weighted_f1": res.get("weighted_f1", 0.0),
                "neutral_f1": res.get("per_class", {}).get("neutral", {}).get("f1", 0.0),
                "elapsed_s": round(time.time() - t0, 1),
            }
            per_seed.append(entry)
            log.info(
                f"BERT seed={seed}  macro_f1={entry['macro_f1']:.4f}  "
                f"acc={entry['accuracy']:.4f}  ({entry['elapsed_s']}s)"
            )
    finally:
        CFG["seed"] = old_top_seed
        CFG["bert"] = old_bcfg

    f1s = [r["macro_f1"] for r in per_seed]
    accs = [r["accuracy"] for r in per_seed]
    neus = [r["neutral_f1"] for r in per_seed]
    results = {
        "seeds": list(seeds),
        "config": {"max_length": 128, "learning_rate": 2e-5, "batch_size": 16, "epochs": 2},
        "per_seed": per_seed,
        "summary": {
            "macro_f1_mean": round(statistics.mean(f1s), 4),
            "macro_f1_std": round(statistics.stdev(f1s), 4) if len(f1s) > 1 else 0.0,
            "macro_f1_min": round(min(f1s), 4),
            "macro_f1_max": round(max(f1s), 4),
            "accuracy_mean": round(statistics.mean(accs), 4),
            "neutral_f1_mean": round(statistics.mean(neus), 4),
        },
    }
    dump_json(EXP / "bert_multiseed.json", results)
    return results


# ============================================================
# 3. SCALE SWEEP
# ============================================================
def _subsample_train(train_full: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Stratified sample of n rows from the full training set."""
    from sklearn.utils import shuffle
    if n >= len(train_full):
        return train_full.reset_index(drop=True)
    # Stratified: keep class ratios roughly the same.
    parts = []
    for _, g in train_full.groupby("label"):
        k = max(1, int(round(len(g) * n / len(train_full))))
        parts.append(g.sample(n=min(k, len(g)), random_state=seed))
    return pd.concat(parts, axis=0).reset_index(drop=True)


def run_scale_sweep(model: str, scales: tuple[int, ...] = (4000, 8000, 16000),
                    epochs: int = 2) -> dict:
    """Train a single model at each data scale, on the fixed 20k test split."""
    log = setup_logger(f"scale_{model}")
    train_full = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y_test = test["label"].values
    n_test = len(test)

    entries = []
    for n_train in scales:
        if n_train > len(train_full):
            log.warning(f"n_train={n_train} > available {len(train_full)}, skip")
            continue
        t0 = time.time()
        train_s = _subsample_train(train_full, n_train, seed=42)
        # Split off a small val for early-stopping (BERT) / no need for LogReg
        from sklearn.model_selection import train_test_split
        tr, val = train_test_split(train_s, test_size=0.1,
                                   stratify=train_s["label"], random_state=42)
        log.info(f"{model} n_train={n_train} (train={len(tr)}, val={len(val)}, test={n_test})")

        if model == "tfidf_logreg":
            art = BASE._fit_tfidf_clf(tr, val, test, "logreg")
            X_te = art["vectorizer"].transform(test["text"].values)
            preds = art["clf"].predict(X_te)
            res = BASE._metrics(y_test, preds)
            entry = {
                "n_train": n_train,
                "accuracy": res["accuracy"],
                "macro_f1": res["macro_f1"],
                "weighted_f1": res["weighted_f1"],
                "neutral_f1": res["per_class"]["neutral"]["f1"],
                "elapsed_s": round(time.time() - t0, 1),
            }
        elif model == "bert":
            old_bcfg = deepcopy(CFG["bert"])
            CFG["bert"]["max_length"] = 128
            CFG["bert"]["learning_rate"] = 2e-5
            CFG["bert"]["batch_size"] = 16
            CFG["bert"]["epochs"] = epochs
            class _A:
                quick = False
                # Pass the subsampled dataframes directly to BERT.train()
                # so it doesn't read the default bert_train.csv (which is 3.2k).
                train_df = tr
                val_df = val
                test_df = test
            hist_path = EXP / f"scale_bert_{n_train}_hist.json"
            out_dir = CKPT / f"bert_scale_{n_train}"
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                summary = BERT.train(_A(), "cpu", log, out_dir, hist_path)
            finally:
                CFG["bert"] = old_bcfg
            m = summary.get("test", {})
            entry = {
                "n_train": n_train,
                "n_train_actual": summary.get("n_train", n_train),
                "n_val": summary.get("n_val", 0),
                "n_test": summary.get("n_test", n_test),
                "accuracy": m.get("accuracy", 0.0),
                "macro_f1": m.get("macro_f1", 0.0),
                "weighted_f1": m.get("weighted_f1", 0.0),
                "neutral_f1": m.get("per_class", {}).get("neutral", {}).get("f1", 0.0),
                "elapsed_s": round(time.time() - t0, 1),
            }
        else:
            raise ValueError(f"Unknown model {model}")

        entries.append(entry)
        log.info(
            f"{model} n_train={n_train}  macro_f1={entry['macro_f1']:.4f}  "
            f"acc={entry['accuracy']:.4f}  ({entry['elapsed_s']}s)"
        )
        # Flush to disk after every scale so a mid-run kill still saves progress.
        out_path = EXP / "scale_sweep.json"
        if out_path.exists():
            results = load_json(out_path)
        else:
            results = {"test_size": n_test, "models": {}}
        results["test_size"] = n_test
        results.setdefault("models", {})[model] = entries
        dump_json(out_path, results)

    return load_json(EXP / "scale_sweep.json")


# ============================================================
# CLI
# ============================================================
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["bootstrap", "bert-multiseed", "scale-sweep", "all"])
    p.add_argument("target", nargs="?", default=None,
                   help="For scale-sweep: 'tfidf_logreg' or 'bert'. "
                        "For bert-multiseed: comma-separated seeds like '42,7,123'. "
                        "For scale-sweep bert/tfidf_logreg with custom scales: use "
                        "'bert:8000' or 'tfidf_logreg:8000,16000' syntax.")
    p.add_argument("--epochs", type=int, default=2,
                   help="Number of epochs for BERT scale-sweep (default 2).")
    args = p.parse_args()

    EXP.mkdir(parents=True, exist_ok=True)

    if args.cmd == "bootstrap":
        run_bootstrap()
    elif args.cmd == "bert-multiseed":
        if args.target:
            seeds = tuple(int(s) for s in args.target.split(","))
        else:
            seeds = (42, 7, 123)
        run_bert_multiseed(seeds=seeds)
    elif args.cmd == "scale-sweep":
        if not args.target:
            print("Need target: 'tfidf_logreg' or 'bert', optionally with "
                  "custom scales like 'bert:8000,16000'.")
            sys.exit(2)
        if ":" in args.target:
            model, scale_str = args.target.split(":", 1)
            scales = tuple(int(s) for s in scale_str.split(","))
        else:
            model = args.target
            scales = (4000, 8000, 16000)
        run_scale_sweep(model, scales=scales, epochs=args.epochs)
    elif args.cmd == "all":
        run_bootstrap()
        run_bert_multiseed()
        run_scale_sweep("tfidf_logreg")
        run_scale_sweep("bert", scales=(8000, 16000))


if __name__ == "__main__":
    cli()
