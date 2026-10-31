"""Revision experiments D4/D5 (R1-8, R2.5, R3.2): statistical robustness.

Part A - paired bootstrap CIs for ALL FOUR models on the 20,000-row test split
         (VADER default + tuned, TF-IDF+LogReg, TF-IDF+SVM).  Per-sample
         predictions come from the released artifacts; VADER predictions are
         recomputed with the released tuned thresholds.

Part B - decomposition of the zero standard deviation in the published Table 7:
         (a) algorithm-seed variation: refit LogReg with different
             random_state on the SAME training rows -> std must be 0
             (deterministic convex problem);
         (b) training-data variation: refit on three independent bootstrap
             resamples of the 160k training rows -> std > 0, this is the
             sampling variability the reviewer asked for.

Writes experiments/bootstrap_ci.json
Usage   python scripts/bootstrap_partitions.py
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

from common.config import CFG  # noqa: E402

DATA = ROOT / CFG["data"]["processed_dir"]
CKPT = ROOT / "models" / "checkpoints"
REL = ROOT.parent / "sentiment-system-release" / "models" / "checkpoints"
EXP = ROOT / "experiments"
LABELS = ["negative", "neutral", "positive"]
N_BOOT = 5000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def y_of(df: pd.DataFrame) -> np.ndarray:
    return np.array([LABELS.index(str(l)) for l in df["label"]], dtype=np.int64)


def macro_f1_from_conf(conf: np.ndarray) -> float:
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


def paired_ci(y: np.ndarray, preds: dict, n_boot: int = N_BOOT, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    codes = {k: (y * 3 + v) for k, v in preds.items()}
    draws = {k: np.empty(n_boot) for k in preds}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for k in preds:
            cnt = np.bincount(codes[k][idx], minlength=9).reshape(3, 3)
            draws[k][b] = macro_f1_from_conf(cnt)
    out = {}
    for k in preds:
        arr = draws[k]
        out[k] = {
            "macro_f1_point": float(f1_score(y, preds[k], average="macro",
                                             zero_division=0)),
            "macro_f1_mean": float(arr.mean()),
            "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        }
    return out


def make_tfidf_clf(seed: int, svc: bool = False):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    vec = TfidfVectorizer(
        lowercase=True, strip_accents="unicode",
        min_df=CFG["baselines"]["tfidf"]["min_df"],
        max_df=CFG["baselines"]["tfidf"]["max_df"],
        ngram_range=tuple(CFG["baselines"]["tfidf"]["ngram_range"]),
        max_features=CFG["baselines"]["tfidf"]["max_features"],
        sublinear_tf=True,
    )
    if svc:
        clf = LinearSVC(C=0.5, class_weight="balanced", dual="auto",
                        random_state=seed)
    else:
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                                 solver="lbfgs", random_state=seed, n_jobs=-1)
    return vec, clf


def fit_eval(train_df, test_df, seed: int, svc: bool = False) -> dict:
    vec, clf = make_tfidf_clf(seed, svc=svc)
    X = vec.fit_transform(train_df["text"].values)
    X_te = vec.transform(test_df["text"].values)
    clf.fit(X, train_df["label"].values)
    pred = clf.predict(X_te)
    y = y_of(test_df)
    pi = np.array([LABELS.index(str(p)) for p in pred])
    return {
        "accuracy": float(accuracy_score(y, pi)),
        "macro_f1": float(f1_score(y, pi, average="macro")),
        "n_features": int(vec.get_feature_names_out().shape[0]),
    }


def main() -> None:
    t0 = time.time()
    import joblib

    test = pd.read_csv(DATA / "test.csv")
    y = y_of(test)
    log(f"test={len(test)}")

    preds: dict[str, np.ndarray] = {}

    # --- TF-IDF + LogReg (released artifact) ---
    art = joblib.load(CKPT / "tfidf_logreg.pkl")
    X = art["vectorizer"].transform(test["text"].values)
    raw = art["clf"].predict(X)
    preds["tfidf_logreg"] = np.array([LABELS.index(str(p)) for p in raw])

    # --- TF-IDF + Linear SVM (release artifact) ---
    svc_src = CKPT / "tfidf_svc.pkl"
    if not svc_src.exists():
        svc_src = REL / "tfidf_svc.pkl"
    art_s = joblib.load(svc_src)
    Xs = art_s["vectorizer"].transform(test["text"].values)
    raw_s = art_s["clf"].predict(Xs)
    preds["tfidf_svc"] = np.array([LABELS.index(str(p)) for p in raw_s])

    # --- VADER default + released tuned thresholds ---
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    thr_path = CKPT / "vader_thresholds.json"
    if not thr_path.exists():
        thr_path = REL / "vader_thresholds.json"
    with open(thr_path, encoding="utf-8") as f:
        thr = json.load(f)
    lo_t, hi_t = thr["tuned"]
    analyzer = SentimentIntensityAnalyzer()
    t1 = time.time()
    classes = np.empty((2, len(test)), dtype=np.int64)
    for i, txt in enumerate(test["text"].tolist()):
        s = analyzer.polarity_scores(txt)["compound"]
        classes[0, i] = 0 if s <= -0.05 else (2 if s >= 0.05 else 1)
        classes[1, i] = 0 if s <= lo_t else (2 if s >= hi_t else 1)
    preds["vader_default"] = classes[0]
    preds["vader_tuned"] = classes[1]
    log(f"VADER predictions in {time.time()-t1:.0f}s (tuned thresholds "
        f"lo={lo_t:.2f} hi={hi_t:.2f})")

    # --- BERT (20k test, cached by fusion_val.py if present) ---
    bp_path = EXP / "bert_test_proba.npy"
    if bp_path.exists():
        preds["distilbert"] = np.load(bp_path).argmax(1)
        log("BERT test predictions from bert_test_proba.npy")

    # --- paired bootstrap CIs ---
    log(f"paired bootstrap x{N_BOOT}")
    ci = paired_ci(y, preds)

    # --- Part B: zero-variance decomposition ---
    train = pd.read_csv(DATA / "train.csv")
    log(f"train={len(train)}  (zero-variance decomposition)")

    algo_seeds = []
    for sd in [7, 123, 2024]:
        r = fit_eval(train, test, seed=sd, svc=False)
        algo_seeds.append({"seed": sd, **r})
        log(f"  algorithm seed {sd}: macro_f1={r['macro_f1']:.4f}")

    data_resamples = []
    rng = np.random.default_rng(99)
    for rep, sd in enumerate([7, 123, 2024]):
        rs = np.random.default_rng(sd)
        idx = rs.integers(0, len(train), size=len(train))  # bootstrap resample
        r = fit_eval(train.iloc[idx], test, seed=42, svc=False)
        data_resamples.append({"rep": rep, "seed": sd, **r})
        log(f"  data resample {rep}: macro_f1={r['macro_f1']:.4f}")

    algo_f1 = [r["macro_f1"] for r in algo_seeds]
    data_f1 = [r["macro_f1"] for r in data_resamples]

    out = {
        "n_test": int(len(test)),
        "note": ("Answers R1-8 / R2.5 / R3-2: CIs for all four models on the "
                 "20k test split, and an explicit decomposition of the "
                 "published zero std (deterministic algorithm vs sampling "
                 "variability of the training rows)."),
        "part_A_bootstrap_ci": ci,
        "part_B_zero_variance_decomposition": {
            "algorithm_seed_variation": {
                "runs": algo_seeds,
                "macro_f1_mean": float(np.mean(algo_f1)),
                "macro_f1_std": float(np.std(algo_f1)),
                "explanation": ("LogisticRegression with lbfgs on identical "
                                "input rows converges to the same optimum, so "
                                "std = 0 is algorithmic determinism, not a "
                                "robustness claim."),
            },
            "training_data_variation": {
                "runs": data_resamples,
                "macro_f1_mean": float(np.mean(data_f1)),
                "macro_f1_std": float(np.std(data_f1)),
                "explanation": ("Three bootstrap resamples of the 160k "
                                "training rows, same algorithm seed: this is "
                                "the sampling variability of the headline "
                                "TF-IDF+LogReg number."),
            },
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(EXP / "bootstrap_ci.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log("wrote experiments/bootstrap_ci.json")
    for k, c in ci.items():
        print(f"  {k:14s} mF1={c['macro_f1_point']:.4f} "
              f"CI[{c['ci95'][0]:.4f},{c['ci95'][1]:.4f}]")


if __name__ == "__main__":
    main()
