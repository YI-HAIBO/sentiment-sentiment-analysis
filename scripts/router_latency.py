"""Revision experiment C3 (R1-9): measured end-to-end cost of the two-tier
router vs the analytical cost model E[c] = C1 + r*C2 used in Section 4.8.

Protocol mirrors the published inference benchmark (experiments/
inference_bench.json): 200 single requests, 8 warm-up calls, CPU-only,
wall-clock, inference only.  For each request the TF-IDF tier always runs;
the DistilBERT tier runs only when max-probability < tau (the B = 5 ms
budget-implied threshold).  Reported alongside the analytical estimate.

Writes experiments/router_latency.json
Usage   python scripts/router_latency.py
"""
from __future__ import annotations

import json
import os
import statistics
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
CKPT = ROOT / "models" / "checkpoints"
REL = ROOT.parent / "sentiment-system-release" / "models" / "checkpoints"
C1, C2 = 0.483, 44.81
N_REQ = 200
N_WARMUP = 8
BUDGET = 5.0

LABELS = ["negative", "neutral", "positive"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    import joblib
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    torch.set_num_threads(14)
    test = pd.read_csv(DATA / "test.csv")

    art = joblib.load(CKPT / "tfidf_logreg.pkl")
    vec, clf = art["vectorizer"], art["clf"]

    ckpt = REL / "bert_final" if (REL / "bert_final").exists() else CKPT / "bert_final"
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
    model.eval()

    # budget-implied threshold from the published test confidence distribution
    p1 = np.load(EXP / "tier1_test_proba.npy")
    s1 = p1.max(1)
    r = (BUDGET - C1) / (C2 - C1)
    tau = float(np.quantile(s1, r))
    log(f"tau={tau:.4f} (B={BUDGET} ms, r={100*r:.2f}%)")

    rng = np.random.default_rng(42)
    idx = rng.choice(len(test), size=N_REQ + N_WARMUP, replace=False)
    texts = test["text"].iloc[idx].tolist()

    # warm-up
    for i in range(N_WARMUP):
        X = vec.transform([texts[i]])
        _ = clf.predict_proba(X)
        with torch.no_grad():
            enc = tok([texts[i]], truncation=True, max_length=128,
                      padding=True, return_tensors="pt")
            model(**enc)

    lat, t1_ms, t2_ms, escalated = [], [], [], 0
    for i in range(N_WARMUP, N_REQ + N_WARMUP):
        X = vec.transform([texts[i]])
        t_a = time.perf_counter()
        pr = clf.predict_proba(X)
        t_b = time.perf_counter()
        s = float(pr.max())
        if s < tau:
            with torch.no_grad():
                enc = tok([texts[i]], truncation=True, max_length=128,
                          padding=True, return_tensors="pt")
                model(**enc)
            t_c = time.perf_counter()
            t2_ms.append((t_c - t_b) * 1e3)
            escalated += 1
        else:
            t_c = t_b
        lat.append((t_c - t_a) * 1e3)
        t1_ms.append((t_b - t_a) * 1e3)

    r_meas = escalated / N_REQ
    out = {
        "environment": {
            "torch_threads": 14,
            "protocol": (f"{N_REQ} single requests, {N_WARMUP} warm-up, "
                         "CPU-only, wall-clock, inference only"),
        },
        "budget_ms": BUDGET,
        "tau": round(tau, 4),
        "measured": {
            "mean_ms": round(float(np.mean(lat)), 3),
            "median_ms": round(float(np.median(lat)), 3),
            "p95_ms": round(float(np.percentile(lat, 95)), 3),
            "tier1_mean_ms": round(float(np.mean(t1_ms)), 3),
            "tier2_mean_ms_when_escalated": (round(float(np.mean(t2_ms)), 3)
                                             if t2_ms else None),
            "escalated_of_200": escalated,
            "measured_escalation_rate": round(r_meas, 4),
        },
        "analytical": {
            "c1_ms": C1, "c2_ms": C2,
            "expected_mean_ms": round(C1 + r * C2, 3),
        },
        "gap_measured_minus_analytical_ms": round(float(np.mean(lat)) - (C1 + r * C2), 3),
        "note": ("Answers R1-9: the routed deployment cost is MEASURED here and "
                 "compared with the analytical estimate used in Eq. (1)."),
    }
    with open(EXP / "router_latency.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"measured mean={out['measured']['mean_ms']:.3f} ms vs analytical "
        f"{out['analytical']['expected_mean_ms']:.3f} ms "
        f"(gap {out['gap_measured_minus_analytical_ms']:+.3f} ms); "
        f"escalated {escalated}/{N_REQ}")


if __name__ == "__main__":
    main()
