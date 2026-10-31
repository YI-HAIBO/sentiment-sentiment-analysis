"""Revision experiment C4 (R2.6 / R3.5): how much content does max_length=128
actually drop on the Yelp Review Full subsets?

Tokenizes the BERT train/val/test subsets (and the 20k baseline test split for
reference) with the project's local DistilBERT tokenizer and reports the
fraction of rows whose tokenized length exceeds 128 / 256, plus length
quantiles.

Writes experiments/truncation.json
Usage   python scripts/truncation.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from common.config import CFG  # noqa: E402

DATA = ROOT / CFG["data"]["processed_dir"]
EXP = ROOT / "experiments"


def length_stats(tok, texts, batch: int = 256) -> dict:
    lengths = []
    for i in range(0, len(texts), batch):
        enc = tok(list(texts[i:i + batch]), truncation=False,
                  padding=False, add_special_tokens=True)
        lengths.extend([len(x) for x in enc["input_ids"]])
    L = np.array(lengths)
    return {
        "n": int(len(L)),
        "mean": float(L.mean()),
        "median": float(np.median(L)),
        "p95": float(np.percentile(L, 95)),
        "p99": float(np.percentile(L, 99)),
        "frac_gt_128": float((L > 128).mean()),
        "frac_gt_160": float((L > 160).mean()),
        "frac_gt_256": float((L > 256).mean()),
        "frac_gt_512": float((L > 512).mean()),
        "bits_of_text_truncated_at_128": None,  # filled by caller
    }


def main() -> None:
    from transformers import AutoTokenizer

    model_name = CFG["bert"]["model_name"]
    model_path = ROOT / model_name
    tok_src = str(model_path) if model_path.exists() else model_name
    tok = AutoTokenizer.from_pretrained(tok_src)

    out = {"tokenizer": tok_src, "splits": {}, "_source": {}}
    for name in ["bert_train.csv", "bert_val.csv", "bert_test.csv", "test.csv"]:
        p = DATA / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        t0 = time.time()
        st = length_stats(tok, df["text"].tolist())
        st["elapsed_s"] = round(time.time() - t0, 1)
        out["splits"][name] = st
        out["_source"][name] = ("BERT fine-tuning subset" if name.startswith("bert_")
                                else "baseline (TF-IDF) test split, 20k")
        print(f"[rev] {name:16s} n={st['n']:6d} >128: {100*st['frac_gt_128']:5.2f}%  "
              f">256: {100*st['frac_gt_256']:5.2f}%  median={st['median']:.0f}")

    out["note"] = (
        "Answers Reviewer 2 comment 6 / Reviewer 3 comment 5. The ml=256 "
        "sensitivity run already exists (experiments/bert_abl_ml256.json, "
        "macro-F1 0.6906 vs 0.6432 at ml=128 on the 400-row BERT test split) "
        "and is reported alongside these rates in the revision."
    )

    with open(EXP / "truncation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[rev] wrote experiments/truncation.json")


if __name__ == "__main__":
    main()
