"""Revision experiment C1 (R1-12 / R2.10): per-class metrics for ALL four models
on ONE table so the 'neutral class is the persistent bottleneck' claim can be
verified directly by the reader.

Sources (no retraining):
  models/checkpoints/baselines_summary.json  -> VADER (default+tuned), LogReg, SVM on 20k test
  models/checkpoints/bert_summary.json       -> DistilBERT on its 400-row test split
  experiments/fusion.json                    -> DistilBERT on the 20k test (published checkpoint)

Writes experiments/perclass_table.json
Usage   python scripts/perclass_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXP = ROOT / "experiments"
CKPT = ROOT / "models" / "checkpoints"

LABELS = ["negative", "neutral", "positive"]


def load(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    base = load(CKPT / "baselines_summary.json")["models"]
    bert_small = load(CKPT / "bert_summary.json")
    fusion = load(EXP / "fusion.json")

    out: dict = {"label_order": LABELS, "test_split": {}, "note": ""}

    # --- TF-IDF family on the 20,000-row baseline test split ---
    for key, label in [("vader", "VADER (default thresholds)"),
                       ("tfidf_logreg", "TF-IDF + Logistic Regression"),
                       ("tfidf_svc", "TF-IDF + Linear SVM")]:
        m = base[key]
        if key == "vader":
            m = m["default"]
        out["test_split"][label] = {
            "n": m["test"]["n"],
            "macro_f1": m["test"]["macro_f1"],
            "accuracy": m["test"]["accuracy"],
            "per_class": m["test"]["per_class"],
        }
    # VADER tuned is a separate published row
    out["test_split"]["VADER (tuned thresholds)"] = {
        "n": base["vader"]["tuned"]["test"]["n"],
        "macro_f1": base["vader"]["tuned"]["test"]["macro_f1"],
        "accuracy": base["vader"]["tuned"]["test"]["accuracy"],
        "per_class": base["vader"]["tuned"]["test"]["per_class"],
    }

    # --- DistilBERT on its own 400-row test split ---
    out["bert_400row"] = {
        "n": bert_small["test"]["n"],
        "macro_f1": bert_small["test"]["macro_f1"],
        "accuracy": bert_small["test"]["accuracy"],
        "per_class": bert_small["test"]["per_class"],
    }

    # --- DistilBERT (published checkpoint) on the 20,000-row test split ---
    out["test_split"]["DistilBERT (published checkpoint, 20k test)"] = {
        "n": fusion["n_test"],
        "macro_f1": fusion["bert"]["macro_f1"],
        "accuracy": fusion["bert"]["accuracy"],
        "per_class": fusion["bert"]["per_class"],
    }

    # --- neutral-bottleneck verification ---
    check = {}
    for label, row in out["test_split"].items():
        f1s = {k: row["per_class"][k]["f1"] for k in LABELS}
        check[label] = {
            "neutral_is_lowest": min(f1s, key=f1s.get) == "neutral",
            "neutral_f1": f1s["neutral"],
            "min_f1": min(f1s.values()),
            "gap_to_best": max(f1s.values()) - f1s["neutral"],
        }
    out["neutral_bottleneck_check"] = check
    out["note"] = (
        "Assembled for the revision: replaces the prose-plus-caption argument "
        "with a single verifiable table (Reviewer 2 comment 10)."
    )

    with open(EXP / "perclass_table.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("[rev] wrote experiments/perclass_table.json")
    for label, c in check.items():
        print(f"  {label:42s} neutral lowest={c['neutral_is_lowest']}  "
              f"neutral F1={c['neutral_f1']:.4f}  gap={c['gap_to_best']:.4f}")


if __name__ == "__main__":
    main()
