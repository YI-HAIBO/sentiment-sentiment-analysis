"""BERT training-duration sweep at fixed 4k training budget.

Vary the number of epochs {3, 5, 8} on the same 4k subsample and record
macro-F1 on the 20k test split. Answers: is 2 epochs an early stop that
leaves performance on the table, or has BERT plateaued?

Writes experiments/epoch_sweep.json.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from common.config import CFG  # noqa: E402
from common.utils import dump_json, load_json, setup_logger  # noqa: E402
import models.bert as BERT  # noqa: E402
import models.baselines as BASE  # noqa: E402

LABEL_ORDER = BASE.LABEL_ORDER
EXP = ROOT / "experiments"
DATA = ROOT / CFG["data"]["processed_dir"]
CKPT = ROOT / "models" / "checkpoints"


def run_epoch_sweep(epochs_grid: tuple[int, ...] = (3, 5, 8)) -> dict:
    log = setup_logger("epoch_sweep")
    train_full = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y_test = test["label"].values
    n_test = len(test)

    # Stratified subsample to 4k (same approach as scale_sweep)
    N_TARGET = 4000
    parts = []
    for _, g in train_full.groupby("label"):
        k = max(1, int(round(len(g) * N_TARGET / len(train_full))))
        parts.append(g.sample(n=min(k, len(g)), random_state=42))
    train_s = pd.concat(parts).reset_index(drop=True)
    from sklearn.model_selection import train_test_split
    tr, val = train_test_split(train_s, test_size=0.1,
                               stratify=train_s["label"], random_state=42)

    old_bcfg = deepcopy(CFG["bert"])
    CFG["bert"]["max_length"] = 128
    CFG["bert"]["learning_rate"] = 2e-5
    CFG["bert"]["batch_size"] = 16
    CFG["bert"]["seed"] = 42

    entries = []
    for n_ep in epochs_grid:
        t0 = time.time()
        CFG["bert"]["epochs"] = n_ep
        class _A:
            quick = False
            train_df = tr
            val_df = val
            test_df = test

        hist_path = EXP / f"epoch_{n_ep}_hist.json"
        out_dir = CKPT / f"bert_epoch_{n_ep}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            summary = BERT.train(_A(), "cpu", log, out_dir, hist_path)
        finally:
            pass
        m = summary.get("test", {})
        entry = {
            "epochs": n_ep,
            "n_train": int(len(tr)),
            "n_val": int(len(val)),
            "n_test": int(n_test),
            "accuracy": float(m.get("accuracy", 0.0)),
            "macro_f1": float(m.get("macro_f1", 0.0)),
            "weighted_f1": float(m.get("weighted_f1", 0.0)),
            "neutral_f1": float(m.get("per_class", {}).get("neutral", {}).get("f1", 0.0)),
            "best_epoch": summary.get("best_epoch"),
            "elapsed_s": round(time.time() - t0, 1),
        }
        entries.append(entry)
        log.info(f"epochs={n_ep}  macro_f1={entry['macro_f1']:.4f}  "
                 f"acc={entry['accuracy']:.4f}  best_epoch={entry['best_epoch']}  "
                 f"({entry['elapsed_s']}s)")
        # Flush after each epoch config so a kill still preserves progress.
        out_path = EXP / "epoch_sweep.json"
        if out_path.exists():
            results = load_json(out_path)
        else:
            results = {"test_size": n_test, "epochs": []}
        results["test_size"] = n_test
        results["epochs"] = entries
        dump_json(out_path, results)

    CFG["bert"] = old_bcfg
    return load_json(EXP / "epoch_sweep.json")


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["epoch-sweep"])
    p.add_argument("epochs", nargs="?", default="3,5,8",
                   help="Comma-separated epoch counts (default '3,5,8').")
    args = p.parse_args()

    EXP.mkdir(parents=True, exist_ok=True)
    if args.cmd == "epoch-sweep":
        epochs = tuple(int(x) for x in args.epochs.split(","))
        run_epoch_sweep(epochs_grid=epochs)


if __name__ == "__main__":
    cli()
