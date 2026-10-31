"""Revision experiment C2 (R1-10): does the budget-aware router actually beat
simpler routing rules at the same latency budget?

Compares, on the published 20,000-row test split with the same two tiers as
Table 12 (TF-IDF+LogReg tier-1 = 0.483 ms, tuned DistilBERT tier-2 = 44.81 ms):

  1. btr           : the published rule - quantile threshold on the TEST
                     confidence distribution ( ta Networks = Q(r) of test s1 )
  2. btr_val       : same budget-implied escalation rate r, but the threshold
                     is calibrated on the 20,000-row VALIDATION split instead
                     of the test split (removes test-set calibration bias)
  3. random        : escalate a uniformly random subset of the same size
  4. entropy       : escalate the r fraction with the highest tier-1 entropy
  5. margin        : escalate the r fraction with the smallest |p1-p2| margin
  6. oracle        : ceiling - escalate exactly the rows the tier-2 fixes
                     (labels used; upper bound only, not a deployable rule)

For every strategy we report macro-F1, accuracy, realized mean latency
E[c] = C1 + r*C2, and paired bootstrap 95% CI (2,000 resamples) of the routed
macro-F1 and of the gain over the tier-1-only baseline.

Writes experiments/routing_baselines.json
Usage   python scripts/routing_baselines.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXP = ROOT / "experiments"
DATA = ROOT / "data" / "processed"

C1 = 0.483          # published median latency of the TF-IDF tier (ms)
C2 = 44.81          # published mean latency of the DistilBERT tier (ms)
BUDGETS = [1.0, 2.0, 5.0, 10.0, 15.0]
N_BOOT = 2000
LABELS = ["negative", "neutral", "positive"]


def mf1(y: np.ndarray, p: np.ndarray) -> float:
    return float(f1_score(y, p, average="macro", zero_division=0))


def acc(y: np.ndarray, p: np.ndarray) -> float:
    return float(accuracy_score(y, p))


def routed_preds(pred1: np.ndarray, pred2: np.ndarray, esc: np.ndarray) -> np.ndarray:
    p = pred1.copy()
    p[esc] = pred2[esc]
    return p


def macro_f1_from_conf(conf: np.ndarray) -> float:
    """macro-F1 from a 3x3 confusion matrix (rows = true)."""
    f1s = []
    for k in range(3):
        tp = conf[k, k]
        fp = conf[:, k].sum() - tp
        fn = conf[k, :].sum() - tp
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def conf_of(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    m = np.zeros((3, 3), dtype=np.int64)
    np.add.at(m, (y, p), 1)
    return m


def main() -> None:
    t0 = time.time()
    rows = pd.read_csv(DATA / "test.csv")
    y = np.array([LABELS.index(str(l)) for l in rows["label"]], dtype=np.int64)

    p1 = np.load(EXP / "tier1_test_proba.npy")            # LogReg on 20k test
    p_t2 = np.load(EXP / "teacher_test_proba.npy")  # tuned DistilBERT on 20k test
    p_val = np.load(EXP / "tier1_val_proba.npy")          # LogReg on 20k validation

    s1, pred1 = p1.max(1), p1.argmax(1)
    pred2 = p_t2.argmax(1)
    s_val = p_val.max(1)
    ent = -(p1 * np.log(np.clip(p1, 1e-12, 1))).sum(1)   # tier-1 entropy

    base_f1 = mf1(y, pred1)
    print(f"[rev] tier1 macro-F1 = {base_f1:.4f}   tier2 macro-F1 = {mf1(y, pred2):.4f}")

    results = {}
    for B in BUDGETS:
        r = min(1.0, max(0.0, (B - C1) / (C2 - C1)))
        if r <= 0:
            continue
        k = max(1, int(round(r * len(y))))
        entry = {"budget_ms": B, "escalation_rate": round(r, 4),
                 "mean_latency_ms_analytic": round(C1 + r * C2, 3)}

        # 1. published BTR rule: quantile of TEST confidences
        tau_test = float(np.quantile(s1, r))
        entry["btr_test_quantile"] = {
            "tau": round(tau_test, 4),
            "macro_f1": mf1(y, routed_preds(pred1, pred2, s1 < tau_test)),
            "accuracy": acc(y, routed_preds(pred1, pred2, s1 < tau_test)),
        }

        # 2. validation-calibrated threshold (same r, calibrated off-test)
        tau_val = float(np.quantile(s_val, r))
        entry["btr_val_quantile"] = {
            "tau_on_validation": round(tau_val, 4),
            "tau_applied_to_test": round(tau_val, 4),
            "macro_f1": mf1(y, routed_preds(pred1, pred2, s1 < tau_val)),
            "accuracy": acc(y, routed_preds(pred1, pred2, s1 < tau_val)),
            "realized_escalation_rate": round(float((s1 < tau_val).mean()), 4),
        }

        # 3. random routing (20 draws)
        rng = np.random.default_rng(11)
        f1s, accs = [], []
        for _ in range(20):
            sel = np.zeros(len(y), dtype=bool)
            sel[rng.choice(len(y), size=k, replace=False)] = True
            f1s.append(mf1(y, routed_preds(pred1, pred2, sel)))
            accs.append(acc(y, routed_preds(pred1, pred2, sel)))
        entry["random"] = {
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
            "macro_f1_best_of_20": float(np.max(f1s)),
            "accuracy_mean": float(np.mean(accs)),
        }

        # 4. entropy routing: highest-entropy r fraction
        ent_order = np.argsort(-ent)[:k]
        sel_ent = np.zeros(len(y), dtype=bool)
        sel_ent[ent_order] = True
        entry["entropy"] = {
            "macro_f1": mf1(y, routed_preds(pred1, pred2, sel_ent)),
            "accuracy": acc(y, routed_preds(pred1, pred2, sel_ent)),
        }

        # 5. margin routing: smallest confidence margin
        marg = np.sort(p1, axis=1)
        margin = 1.0 - (marg[:, -1] - marg[:, -2])  # 1 - (top1 - top2)
        marg_order = np.argsort(-margin)[:k]
        sel_m = np.zeros(len(y), dtype=bool)
        sel_m[marg_order] = True
        entry["margin"] = {
            "macro_f1": mf1(y, routed_preds(pred1, pred2, sel_m)),
            "accuracy": acc(y, routed_preds(pred1, pred2, sel_m)),
        }

        # 6. oracle ceiling (label-aware, not deployable)
        fixed = (pred1 != y) & (pred2 == y)
        if fixed.sum() > 0:
            # escalate exactly the rows tier-2 fixes (bounded to k)
            cand = np.where(fixed)[0]
            sel_o = np.zeros(len(y), dtype=bool)
            sel_o[cand[:k]] = True
            entry["oracle_upper_bound"] = {
                "macro_f1": mf1(y, routed_preds(pred1, pred2, sel_o)),
                "n_fixable": int(fixed.sum()),
            }

        results[f"B_{B:g}"] = entry
        print(f"[rev] B={B:>5.1f} ms  r={100*r:5.2f}%  "
              f"btr={entry['btr_test_quantile']['macro_f1']:.4f}  "
              f"btr_val={entry['btr_val_quantile']['macro_f1']:.4f}  "
              f"rand={entry['random']['macro_f1_mean']:.4f}  "
              f"ent={entry['entropy']['macro_f1']:.4f}")

    # ---- paired bootstrap at the headline budget B = 5 ms ----
    B = 5.0
    r = (B - C1) / (C2 - C1)
    k = max(1, int(round(r * len(y))))
    tau_test = float(np.quantile(s1, r))
    tau_val = float(np.quantile(s_val, r))
    rng = np.random.default_rng(11)
    sel_rand = np.zeros(len(y), dtype=bool)
    sel_rand[rng.choice(len(y), size=k, replace=False)] = True
    ent_order = np.argsort(-ent)[:k]
    sel_ent = np.zeros(len(y), dtype=bool)
    sel_ent[ent_order] = True

    preds = {
        "tier1_only": pred1,
        "btr_test": routed_preds(pred1, pred2, s1 < tau_test),
        "btr_val": routed_preds(pred1, pred2, s1 < tau_val),
        "random": routed_preds(pred1, pred2, sel_rand),
        "entropy": routed_preds(pred1, pred2, sel_ent),
        "tier2_only": pred2,
    }
    ci = {}
    rng2 = np.random.default_rng(7)
    n = len(y)
    codes = {name: (y * 3 + preds[name]) for name in preds}
    draws = {name: np.empty(N_BOOT) for name in preds}
    for b in range(N_BOOT):
        idx = rng2.integers(0, n, size=n)
        for name in preds:
            cnt = np.bincount(codes[name][idx], minlength=9).reshape(3, 3)
            draws[name][b] = macro_f1_from_conf(conf=cnt)
    for name in preds:
        arr = draws[name]
        gain = arr - draws["tier1_only"]
        ci[name] = {
            "macro_f1_mean": float(arr.mean()),
            "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
            "gain_vs_tier1_mean": float(gain.mean()),
            "gain_ci95": [float(np.percentile(gain, 2.5)), float(np.percentile(gain, 97.5))],
        }

    out = {
        "n_test": int(len(y)),
        "c1_ms": C1, "c2_ms": C2,
        "tier1_macro_f1": base_f1,
        "tier2_macro_f1": mf1(y, pred2),
        "note": ("All strategies share the same two tiers, the same test split "
                 "and the same budget-implied escalation rate; only the routing "
                 "rule differs. Tier-2 = the tuned DistilBERT checkpoint behind "
                 "Table 12 (experiments/teacher_test_proba.npy)."),
        "budgets": results,
        "bootstrap_paired_B5": ci,
        "n_boot": N_BOOT,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(EXP / "routing_baselines.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[rev] wrote experiments/routing_baselines.json  "
          f"({time.time()-t0:.1f}s)")
    for name, c in ci.items():
        print(f"  {name:10s} mF1={c['macro_f1_mean']:.4f} "
              f"CI[{c['ci95'][0]:.4f},{c['ci95'][1]:.4f}]  "
              f"gain={c['gain_vs_tier1_mean']:+.4f} "
              f"CI[{c['gain_ci95'][0]:+.4f},{c['gain_ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
