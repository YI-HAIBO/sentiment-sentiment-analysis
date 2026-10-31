"""
Additional experiments beyond the baseline single-run pipeline:

    multiseed   — Rerun VADER / TF-IDF+LR / TF-IDF+SVC on 5 seeds, report mean±std
    ensemble    — TF-IDF+LR ⊕ TF-IDF+SVC (hard vote + soft prob average)
    bert-abl    — DistilBERT ablations (max_length / lr / batch_size), each 2 epochs

Usage:
    python -m scripts.run_experiments multiseed
    python -m scripts.run_experiments ensemble
    python -m scripts.run_experiments bert-abl <variant_name>
    python -m scripts.run_experiments all-fast
    python -m scripts.run_experiments all-abl        # runs all bert variants sequentially

Results land in experiments/:
    multiseed.json
    ensemble.json
    bert_ablation.json         (merged from bert_abl_<name>.json)
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from common.config import CFG  # noqa: E402
from common.utils import dump_json, load_json, setup_logger, seed_everything  # noqa: E402
import models.bert as BERT  # noqa: E402
import models.baselines as BASE  # noqa: E402


LABEL_ORDER = BASE.LABEL_ORDER
EXP = ROOT / "experiments"


# ============================================================
# 1. MULTI-SEED BASELINES
# ============================================================
def run_multiseed(seeds: list[int] = (42, 7, 123, 2024, 31415)) -> dict:
    log = setup_logger("multiseed")
    data_dir = ROOT / CFG["data"]["processed_dir"]
    out_dir = ROOT / "models" / "checkpoints"
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")

    results = {"seeds": list(seeds), "models": {}}

    for model_name in ["tfidf_logreg", "tfidf_svc", "vader"]:
        per_seed = []
        for seed in seeds:
            t0 = time.time()
            # Patch the CFG seed temporarily
            old_seed = CFG["seed"]
            CFG["seed"] = seed
            try:
                if model_name == "vader":
                    # VADER: use tuned thresholds on this seed
                    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                    analyzer = SentimentIntensityAnalyzer()
                    n_tune = min(20000, len(train))
                    tune_train = train.sample(n=n_tune, random_state=seed)
                    thr = BASE._tune_vader_thresholds(tune_train, val, analyzer)
                    preds = [BASE._pred_from_score(analyzer.polarity_scores(t)["compound"], *thr)
                             for t in test["text"]]
                else:
                    clf_name = "logreg" if "logreg" in model_name else "svc"
                    art = BASE._fit_tfidf_clf(train, val, test, clf_name)
                    vec = art["vectorizer"]
                    clf = art["clf"]
                    X_te = vec.transform(test["text"].values)
                    preds = list(clf.predict(X_te))
                res = BASE._metrics(test["label"].values, preds)
                per_seed.append({
                    "seed": seed,
                    "accuracy": res["accuracy"],
                    "macro_f1": res["macro_f1"],
                    "weighted_f1": res["weighted_f1"],
                    "neutral_f1": res["per_class"]["neutral"]["f1"],
                    "elapsed_s": round(time.time() - t0, 1),
                })
            finally:
                CFG["seed"] = old_seed
            log.info(f"{model_name} seed={seed}  macro_f1={per_seed[-1]['macro_f1']:.4f}  "
                     f"({per_seed[-1]['elapsed_s']}s)")
        # aggregate
        accs = [r["accuracy"] for r in per_seed]
        f1s = [r["macro_f1"] for r in per_seed]
        wf1s = [r["weighted_f1"] for r in per_seed]
        neu = [r["neutral_f1"] for r in per_seed]
        results["models"][model_name] = {
            "per_seed": per_seed,
            "summary": {
                "accuracy_mean": round(statistics.mean(accs), 4),
                "accuracy_std": round(statistics.stdev(accs), 4) if len(accs) > 1 else 0.0,
                "macro_f1_mean": round(statistics.mean(f1s), 4),
                "macro_f1_std": round(statistics.stdev(f1s), 4) if len(f1s) > 1 else 0.0,
                "weighted_f1_mean": round(statistics.mean(wf1s), 4),
                "weighted_f1_std": round(statistics.stdev(wf1s), 4) if len(wf1s) > 1 else 0.0,
                "neutral_f1_mean": round(statistics.mean(neu), 4),
                "neutral_f1_std": round(statistics.stdev(neu), 4) if len(neu) > 1 else 0.0,
            },
        }
    dump_json(EXP / "multiseed.json", results)
    return results


# ============================================================
# 2. ENSEMBLE (TF-IDF+LR ⊕ TF-IDF+SVC)
# ============================================================
def run_ensemble() -> dict:
    log = setup_logger("ensemble")
    data_dir = ROOT / CFG["data"]["processed_dir"]
    out_dir = ROOT / "models" / "checkpoints"
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y_test = test["label"].values

    # Fit both models independently
    lr_art = BASE._fit_tfidf_clf(train, val, test, "logreg")
    svc_art = BASE._fit_tfidf_clf(train, val, test, "svc")
    clf_lr = lr_art["clf"]
    clf_svc = svc_art["clf"]
    # Both share the same vectorizer (same TF-IDF config). Use LR's.
    X_te = lr_art["vectorizer"].transform(test["text"].values)

    # Predict on test
    lr_pred = clf_lr.predict(X_te)
    svc_pred = clf_svc.predict(X_te)

    # ---- Disagreement analysis ----
    disagree = sum(1 for a, b in zip(lr_pred, svc_pred) if a != b)
    disagree_rate = disagree / max(1, len(lr_pred))

    # ---- Hard majority vote (2 classifiers, ties broken toward the higher-precision class) ----
    # For 2-vote systems, majority is just "agreement wins, else tie-break".
    # We pick LogReg as the tie-breaker because it has the higher macro-F1.
    hard_vote = [lr if lr == svc else lr for lr, svc in zip(lr_pred, svc_pred)]

    # ---- Soft ensemble: LR probabilities + SVC softmaxed decision distances ----
    # LR exposes probabilities directly.
    lr_probs = clf_lr.predict_proba(X_te)  # shape (n, 3)
    # SVC exposes decision_function (signed distance to hyperplane).
    # For linear SVC with 3 classes, it returns 3 values (one-vs-rest).
    svc_decisions = clf_svc.decision_function(X_te)  # shape (n, 3)
    # Softmax with temperature to convert to pseudo-probabilities.
    # Higher temperature = flatter distribution (less confident).
    def _softmax(x, T=5.0):
        x = (x - x.max(axis=1, keepdims=True)) / T
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)
    svc_probs = _softmax(svc_decisions, T=5.0)

    # Average the two probability matrices and argmax.
    combined = 0.5 * (lr_probs + svc_probs)
    soft_pred = [LABEL_ORDER[i] for i in combined.argmax(axis=1).tolist()]

    results = {
        "vote_rule": "majority, tie-break -> logreg (higher macro-F1)",
        "soft_rule": "0.5 * (LR.predict_proba + softmax(SVC.decision_function, T=5))",
        "disagreement_rate": round(disagree_rate, 4),
        "hard_vote": BASE._metrics(y_test, hard_vote),
        "soft_ensemble": BASE._metrics(y_test, soft_pred),
        "logreg_solo": BASE._metrics(y_test, lr_pred),
        "svc_solo": BASE._metrics(y_test, svc_pred),
    }
    dump_json(EXP / "ensemble.json", results)
    log.info(
        f"disagreement_rate={disagree_rate:.3f}  "
        f"LR={results['logreg_solo']['macro_f1']:.4f}  "
        f"SVC={results['svc_solo']['macro_f1']:.4f}  "
        f"hard_vote={results['hard_vote']['macro_f1']:.4f}  "
        f"soft={results['soft_ensemble']['macro_f1']:.4f}"
    )
    return results


# ============================================================
# 3. BERT ABLATION
# ============================================================
ABLATION_VARIANTS = {
    # name         : (max_length, learning_rate, batch_size)
    "baseline"     : (128, 2e-5, 16),   # reference
    "ml64"         : (64,  2e-5, 16),   # shorter context
    "ml256"        : (256, 2e-5, 16),   # longer context
    "lr1e-5"       : (128, 1e-5, 16),   # lower LR
    "lr5e-5"       : (128, 5e-5, 16),   # higher LR
    "bs8"          : (128, 2e-5, 8),    # smaller batch
    "bs32"         : (128, 2e-5, 32),   # larger batch
}


def run_bert_ablation(variant: str, out_json: Path | None = None) -> dict:
    """Run one DistilBERT variant and dump a per-variant summary JSON."""
    name = variant
    if name not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown variant {name}. Available: {list(ABLATION_VARIANTS)}")
    ml, lr, bs = ABLATION_VARIANTS[name]

    log = setup_logger(f"bert_abl_{name}")
    log.info(f"Running BERT ablation variant '{name}'  max_length={ml}  lr={lr}  batch_size={bs}")

    # Save original CFG and override for this run
    old_bcfg = deepcopy(CFG["bert"])
    CFG["bert"]["max_length"] = ml
    CFG["bert"]["learning_rate"] = lr
    CFG["bert"]["batch_size"] = bs

    device = "cpu"  # CPU-only per project design
    out_dir = ROOT / "models" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = EXP / f"bert_abl_{name}_history.json"
    EXP.mkdir(parents=True, exist_ok=True)

    # Build a minimal argparse namespace matching models.bert.train()
    class _A:
        quick = False
    args = _A()

    t0 = time.time()
    try:
        summary = BERT.train(args, device, log, out_dir, hist_path)
    finally:
        # Restore CFG
        CFG["bert"] = old_bcfg

    # Save per-variant summary (do NOT overwrite the main bert_summary.json)
    summary["variant"] = name
    summary["config"] = {"max_length": ml, "learning_rate": lr, "batch_size": bs}
    summary["wall_time_s"] = round(time.time() - t0, 1)
    out_json = out_json or EXP / f"bert_abl_{name}.json"
    dump_json(out_json, summary)

    # Also merge into a running aggregate file
    agg_path = EXP / "bert_ablation.json"
    if agg_path.exists():
        agg = load_json(agg_path)
    else:
        agg = {"label_order": LABEL_ORDER, "variants": {}}
    agg["variants"][name] = summary
    dump_json(agg_path, agg)
    return summary


# ============================================================
# CLIs
# ============================================================
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["multiseed", "ensemble", "bert-abl", "all-fast", "all-abl"])
    p.add_argument("variant", nargs="?", default=None,
                   help="For bert-abl / all-abl: specific variant name or 'all'.")
    args = p.parse_args()

    if args.cmd == "multiseed":
        run_multiseed()
    elif args.cmd == "ensemble":
        run_ensemble()
    elif args.cmd == "bert-abl":
        if not args.variant:
            print("Need a variant name. Available:", list(ABLATION_VARIANTS))
            sys.exit(2)
        run_bert_ablation(args.variant)
    elif args.cmd == "all-fast":
        run_ensemble()
        run_multiseed()
    elif args.cmd == "all-abl":
        names = args.variant.split(",") if args.variant else list(ABLATION_VARIANTS)
        if "all" in names:
            names = list(ABLATION_VARIANTS)
        for n in names:
            print(f"\n=== BERT ablation: {n} ===")
            run_bert_ablation(n)


if __name__ == "__main__":
    cli()
