"""Revision experiment D4 (editor: software-engineering scope / Section 3):
does the paper's finding pattern transfer to an app-store review corpus?

Third dataset: Google Play store reviews (lara-popovic/google_play_store_reviews)
- 376,325 valid reviews across 96 apps / 10 categories, each with a 1-5 star
  rating.  Subsampled to a star-balanced pool (cap 38,000 per star) so the
  class proportions are the SAME 2:1:2 negative:neutral:positive as the Yelp
  and Amazon studies, then mapped with the identical 1-2/3/4-5 rule.

Protocol is byte-for-byte the Amazon replication (amazon_eval.py):
  - VADER (default thresholds + thresholds re-tuned on this corpus's val)
  - TF-IDF + LogisticRegression / LinearSVC on a 160k stratified train subset
    (same vectorizer hyper-parameters, class_weight="balanced")
  - TF-IDF + LogReg at the MATCHED 3,200-row data budget
  - DistilBERT fine-tuned on 3,200 rows (2 epochs, lr 2e-5, max_length 128,
    batch 16 - the published reference configuration), evaluated on the same
    5,000-row test split and on its own 400-row held-out split
  - per-class F1 for every model to test whether the neutral-class bottleneck
    generalizes
  - paired bootstrap 95% CIs for the headline comparison

Writes data/processed/appstore_*.csv and experiments/appstore_eval.json
Usage   python scripts/appstore_eval.py
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
LABELS = ["negative", "neutral", "positive"]
STARS = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}   # same convention as the Yelp mapping
SEED = 42

N_BASELINE_TRAIN = 160_000
N_BERT_SUBSET = 4_000
CAP_PER_STAR = 38_000           # star-balanced pool -> 2:1:2 class proportions
N_VAL = 5_000
N_TEST = 5_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def macro_f1_from_conf(conf: np.ndarray) -> float:
    f1s = []
    for k in range(3):
        tp = conf[k, k]
        fp = conf[:, k].sum() - tp
        fn = conf[k, :].sum() - tp
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def per_class(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import precision_recall_fscore_support
    pr, rc, f, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2],
                                                   average=None, zero_division=0)
    return {LABELS[i]: {"precision": float(pr[i]), "recall": float(rc[i]),
                        "f1": float(f[i])} for i in range(3)}


def build_splits() -> dict:
    """Load the cached Google Play corpus and write the mirror splits."""
    from sklearn.model_selection import train_test_split

    raw = ROOT / "data" / "raw" / "appstore" / "all_reviews.csv"
    assert raw.exists(), f"missing {raw} (download first)"
    log(f"loading {raw} ({raw.stat().st_size/1e6:.1f} MB)")
    df = pd.read_csv(raw, on_bad_lines="skip", engine="c")
    sc = pd.to_numeric(df["score"], errors="coerce")
    mask = (df["content"].notna()
            & (df["content"].astype(str).str.strip() != "")
            & sc.between(1, 5))
    df = df[mask].copy()
    df["score"] = sc[mask].astype(int)
    log(f"valid rows = {len(df)}")

    # star-balanced pool: cap every star at CAP_PER_STAR -> 2:1:2 classes
    parts = []
    for s in range(1, 6):
        part = df[df["score"] == s]
        if len(part) > CAP_PER_STAR:
            part = part.sample(n=CAP_PER_STAR, random_state=SEED)
        parts.append(part)
    pool = pd.concat(parts, ignore_index=True)
    log(f"balanced pool = {len(pool)} rows, star counts = "
        f"{pool['score'].value_counts().sort_index().tolist()}")

    def lab(stars: pd.Series) -> pd.Series:
        return pd.Series([LABELS[STARS[int(s)]] for s in stars])

    # carve test / val / train-pool (stratified, disjoint)
    train_pool, test = train_test_split(
        pool, test_size=N_TEST, random_state=SEED, stratify=pool["score"].values)
    train_pool, val = train_test_split(
        train_pool, test_size=N_VAL, random_state=SEED,
        stratify=train_pool["score"].values)
    train_pool = train_pool.reset_index(drop=True)

    def frame(d: pd.DataFrame) -> pd.DataFrame:
        # assign positionally (numpy array) so that train_test_split's
        # non-contiguous indices can never misalign the label Series
        d = d.reset_index(drop=True)
        d["label"] = np.array([LABELS[STARS[int(s)]] for s in d["score"]])
        d["text"] = d["content"].astype(str).str.strip()
        return d[["text", "label", "score"]]

    train_all = frame(train_pool)
    dev = frame(val)
    test = frame(test)
    log(f"raw counts: train_pool={len(train_all)} val={len(dev)} test={len(test)}")

    strat = train_all["label"].values
    # disjoint 4k stratified BERT subset; the remainder feeds the TF-IDF
    # train split (160k stratified, mirroring the Yelp protocol exactly)
    base_idx, bert_idx = train_test_split(
        np.arange(len(train_all)), test_size=N_BERT_SUBSET,
        random_state=SEED, stratify=strat)
    rest = train_all.iloc[base_idx].reset_index(drop=True)
    train, _ = train_test_split(
        rest, train_size=N_BASELINE_TRAIN, random_state=SEED,
        stratify=rest["label"].values)
    train = train.reset_index(drop=True)
    assert len(train) == N_BASELINE_TRAIN
    bert_sub = train_all.iloc[bert_idx].reset_index(drop=True)
    # 4,000 -> 3,200 / 400 / 400 (mirrors the Yelp BERT subset protocol)
    bert_train, bert_rest = train_test_split(
        bert_sub, test_size=800, random_state=SEED,
        stratify=bert_sub["label"].values)
    bert_val, bert_test = train_test_split(
        bert_rest, test_size=400, random_state=SEED,
        stratify=bert_rest["label"].values)

    # matched-data-budget TF-IDF training rows (same 3,200 rows as BERT)
    mb_train, mb_val = train_test_split(
        bert_sub, test_size=800, random_state=SEED,
        stratify=bert_sub["label"].values)

    out = {
        "train": train, "val": dev, "test": test,
        "bert_train": bert_train.reset_index(drop=True),
        "bert_val": bert_val.reset_index(drop=True),
        "bert_test": bert_test.reset_index(drop=True),
        "matched_train": mb_train.reset_index(drop=True),
    }
    for name, dfx in out.items():
        dfx.to_csv(DATA / f"appstore_{name}.csv", index=False)
    log(f"wrote appstore splits: " +
        ", ".join(f"{k}={len(v)}" for k, v in out.items()))
    return out


def y_of(df: pd.DataFrame) -> np.ndarray:
    return np.array([LABELS.index(str(l)) for l in df["label"]], dtype=np.int64)


def make_tfidf(svc: bool = False, seed: int = SEED):
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
                                 solver="lbfgs", random_state=seed)
    return vec, clf


def fit_predict(train_df, test_df, svc: bool = False, seed: int = SEED):
    vec, clf = make_tfidf(svc=svc, seed=seed)
    X = vec.fit_transform(train_df["text"].values)
    X_te = vec.transform(test_df["text"].values)
    clf.fit(X, train_df["label"].values)
    raw = clf.predict(X_te)
    pi = np.array([LABELS.index(str(p)) for p in raw])
    n_feat = int(vec.get_feature_names_out().shape[0])
    return pi, n_feat


def main() -> None:
    import random
    import torch
    from sklearn.metrics import f1_score, accuracy_score

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(14)
    t0 = time.time()

    paths = {k: DATA / f"appstore_{k}.csv" for k in
             ["train", "val", "test", "bert_train", "bert_val", "bert_test",
              "matched_train"]}
    if all(p.exists() for p in paths.values()):
        splits = {k: pd.read_csv(v) for k, v in paths.items()}
        log("cached appstore splits")
    else:
        splits = build_splits()

    test, y_test = splits["test"], y_of(splits["test"])
    log(f"appstore test = {len(test)} rows, class counts = "
        f"{np.bincount(y_test, minlength=3).tolist()}")

    results: dict = {"dataset": "Google Play store reviews "
                     "(lara-popovic/google_play_store_reviews), "
                     "star-balanced subsample (38k/star)",
                     "mapping": "1-2 negative / 3 neutral / 4-5 positive",
                     "n_test": int(len(test)),
                     "n_apps": 96, "n_categories": 10,
                     "pool_size": int(len(pd.read_csv(DATA / "appstore_train.csv"))
                                      + len(pd.read_csv(DATA / "appstore_val.csv"))
                                      + len(pd.read_csv(DATA / "appstore_test.csv")))}

    # ---------- VADER ----------
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    t1 = time.time()
    val_scores = np.array([analyzer.polarity_scores(t)["compound"]
                           for t in splits["val"]["text"].tolist()])
    test_scores = np.array([analyzer.polarity_scores(t)["compound"]
                            for t in test["text"].tolist()])
    y_val = y_of(splits["val"])
    best, best_f1 = (-0.05, 0.05), -1.0
    for lo in np.arange(-0.30, 0.31, 0.02):
        for hi in np.arange(-0.30, 0.31, 0.02):
            if hi <= lo + 0.02:
                continue
            pv = np.full(len(val_scores), 1)
            pv[val_scores <= lo] = 0
            pv[val_scores >= hi] = 2
            f = f1_score(y_val, pv, average="macro", zero_division=0)
            if f > best_f1:
                best_f1, best = f, (float(lo), float(hi))
    def vpred(scores, thresholds):
        lo, hi = thresholds
        p = np.full(len(scores), 1)
        p[scores <= lo] = 0
        p[scores >= hi] = 2
        return p
    results["vader_default"] = {
        "macro_f1": float(f1_score(y_test, vpred(test_scores, (-0.05, 0.05)),
                                   average="macro", zero_division=0)),
        "tuned_thresholds_on_val": best, "elapsed_s": round(time.time() - t1, 1)}
    results["vader_tuned"] = {
        "macro_f1": float(f1_score(y_test, vpred(test_scores, best),
                                   average="macro", zero_division=0)),
        "per_class": per_class(y_test, vpred(test_scores, best))}
    log(f"VADER: default={results['vader_default']['macro_f1']:.4f} "
        f"tuned={results['vader_tuned']['macro_f1']:.4f} "
        f"({time.time()-t1:.0f}s, thresholds {best})")

    # ---------- TF-IDF family ----------
    t1 = time.time()
    pi, nf = fit_predict(splits["train"], test, svc=False)
    results["tfidf_logreg_160k"] = {
        "macro_f1": float(f1_score(y_test, pi, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_test, pi)),
        "n_features": nf,
        "per_class": per_class(y_test, pi)}
    log(f"TF-IDF+LogReg 160k: {results['tfidf_logreg_160k']['macro_f1']:.4f} "
        f"({time.time()-t1:.0f}s)")

    t1 = time.time()
    pi_s, nf_s = fit_predict(splits["train"], test, svc=True)
    results["tfidf_svc_160k"] = {
        "macro_f1": float(f1_score(y_test, pi_s, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_test, pi_s)),
        "n_features": nf_s}
    log(f"TF-IDF+SVM 160k: {results['tfidf_svc_160k']['macro_f1']:.4f} "
        f"({time.time()-t1:.0f}s)")

    # matched data budget: TF-IDF+LogReg trained on 3,200 rows
    t1 = time.time()
    pi_m, _ = fit_predict(splits["matched_train"].head(3200), test, svc=False)
    results["tfidf_logreg_matched_3200"] = {
        "macro_f1": float(f1_score(y_test, pi_m, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_test, pi_m)),
        "n_train": 3200,
        "per_class": per_class(y_test, pi_m)}
    log(f"TF-IDF+LogReg matched 3.2k: "
        f"{results['tfidf_logreg_matched_3200']['macro_f1']:.4f} "
        f"({time.time()-t1:.0f}s)")

    # ---------- DistilBERT (published reference configuration) ----------
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torch.optim.lr_scheduler import LinearLR, SequentialLR

    MAX_LEN, BS, LR = 128, 16, 2e-5
    EPOCHS = int(CFG["bert"]["epochs"])
    model_name = CFG["bert"]["model_name"]
    mp = ROOT / model_name
    src = str(mp) if mp.exists() else model_name
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=3)

    class DS(Dataset):
        def __init__(self, texts, labels):
            self.texts, self.labels = texts, labels
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, i):
            enc = tok(self.texts[i], truncation=True, padding="max_length",
                      max_length=MAX_LEN, return_attention_mask=False)
            return {"input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                    "labels": torch.tensor(LABELS.index(self.labels[i]), dtype=torch.long)}

    bt = splits["bert_train"]
    yb = np.array([LABELS.index(l) for l in bt["label"]])
    counts = np.bincount(yb, minlength=3)
    w = 1.0 / np.maximum(counts, 1)
    sampler = WeightedRandomSampler(torch.tensor([w[i] for i in yb], dtype=torch.double),
                                     num_samples=len(bt), replacement=True)
    loader = DataLoader(DS(bt["text"].tolist(), bt["label"].tolist()),
                        batch_size=BS, sampler=sampler)
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=float(CFG["bert"]["weight_decay"]))
    total = max(1, len(loader) * EPOCHS)
    warm = int(total * float(CFG["bert"]["warmup_ratio"]))
    sch = SequentialLR(opt, schedulers=[
        LinearLR(opt, start_factor=1e-8, end_factor=1.0, total_iters=max(1, warm)),
        LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=max(1, total - warm)),
    ], milestones=[warm])

    history = []
    for ep in range(EPOCHS):
        model.train()
        run_loss, corr, tot = 0.0, 0, 0
        te = time.time()
        for i, batch in enumerate(loader):
            out = model(input_ids=batch["input_ids"], labels=batch["labels"])
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            run_loss += out.loss.item()
            corr += int((out.logits.argmax(-1) == batch["labels"]).sum())
            tot += batch["labels"].size(0)
            if i % 20 == 0:
                log(f"  epoch {ep+1} step {i}/{len(loader)} loss={out.loss.item():.4f}")
        history.append({"epoch": ep + 1, "train_loss": run_loss / len(loader),
                        "train_acc": corr / max(1, tot)})
        log(f"epoch {ep+1}: train_acc={corr/max(1,tot):.4f} ({time.time()-te:.0f}s)")

    def predict(texts, batch_size=64):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                enc = tok(chunk, padding=True, truncation=True,
                          max_length=MAX_LEN, return_tensors="pt")
                preds.extend(model(**enc).logits.argmax(-1).tolist())
        return np.array(preds)

    yb_test = y_of(splits["bert_test"])
    pb_400 = predict(splits["bert_test"]["text"].tolist())
    results["distilbert_bertsplit_400"] = {
        "macro_f1": float(f1_score(yb_test, pb_400, average="macro",
                                   zero_division=0)),
        "accuracy": float(accuracy_score(yb_test, pb_400)),
        "n_train": int(len(bt)),
        "per_class": per_class(yb_test, pb_400)}

    t1 = time.time()
    pb = predict(test["text"].tolist(), batch_size=128)
    # persist the fine-tuned checkpoint and per-example predictions so the
    # paired bootstrap can be reproduced without re-running fine-tuning
    try:
        model.save_pretrained(ROOT / "models" / "distilbert_appstore")
        tok.save_pretrained(ROOT / "models" / "distilbert_appstore")
        np.save(DATA / "appstore_bert_test_preds.npy", pb)
    except Exception as e:  # persistence is best-effort, never fatal
        log(f"  (checkpoint persistence skipped: {e})")
    results["distilbert_test_5000"] = {
        "macro_f1": float(f1_score(y_test, pb, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_test, pb)),
        "n_train": int(len(bt)),
        "per_class": per_class(y_test, pb),
        "eval_elapsed_s": round(time.time() - t1, 1)}
    log(f"DistilBERT 3.2k -> 5k test: "
        f"{results['distilbert_test_5000']['macro_f1']:.4f}")

    # ---------- paired bootstrap for the matched comparison ----------
    preds = {"tfidf_matched_3200": pi_m, "distilbert_3200": pb,
             "tfidf_160k": pi}
    rng = np.random.default_rng(7)
    n = len(y_test)
    codes = {k: (y_test * 3 + v) for k, v in preds.items()}
    draws = {k: np.empty(2000) for k in preds}
    for b in range(2000):
        idx = rng.integers(0, n, size=n)
        for k in preds:
            cnt = np.bincount(codes[k][idx], minlength=9).reshape(3, 3)
            draws[k][b] = macro_f1_from_conf(cnt)
    results["bootstrap_paired"] = {
        k: {"macro_f1_mean": float(draws[k].mean()),
            "ci95": [float(np.percentile(draws[k], 2.5)),
                     float(np.percentile(draws[k], 97.5))]}
        for k in preds}

    results["elapsed_s"] = round(time.time() - t0, 1)
    results["tfidf_train_note"] = (
        "Stratified 160,000-row subset of the star-balanced Google Play "
        "pool, disjoint from the 4k BERT subset (mirrors the Yelp/Amazon "
        "160k/4k protocol).")
    results["note"] = ("App-store generalizability check (editor scope "
                       "comment / Section 3). Same three-class mapping, same "
                       "training budgets (160k TF-IDF / 3.2k BERT), same "
                       "reference DistilBERT configuration as the Yelp and "
                       "Amazon studies.")
    with open(EXP / "appstore_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    log("wrote experiments/appstore_eval.json  "
        f"[{time.time()-t0:.0f}s]")
    for k in ["vader_default", "vader_tuned", "tfidf_logreg_160k",
              "tfidf_svc_160k", "tfidf_logreg_matched_3200",
              "distilbert_bertsplit_400", "distilbert_test_5000"]:
        print(f"  {k:32s} mF1={results[k]['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
