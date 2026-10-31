"""
C7: Two-stage training — first converge (uniform), then difficulty-weight.

Hypothesis: the original difficulty-weighted experiment (gamma=1) failed because
the published checkpoint was undertrained. The class-budget effect (neutral F1
0.5038 -> 0.4234) may be an artifact of undertraining. If we first train to
convergence (uniform, 3 epochs -> 0.7336) and THEN apply difficulty weighting
(1 epoch), the weighting may work on a converged model.

Stage 1: Load difficulty_best_ctl.pt (uniform, converged, 0.7336 macro-F1)
Stage 2: Fine-tune with w = (1-s)^gamma for 1 epoch

Outputs: experiments/difficulty_best_c7.pt, difficulty_meta_c7.json
Usage:   python scripts/run_c7_twostage.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from transformers import AutoTokenizer, DistilBertForSequenceClassification  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments"
MODEL_DIR = str(ROOT / "models" / "distilbert_local")
LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}
SEED = 42
MAXLEN = 256
BS = 16
LR = 5e-5
WD = 0.01
GAMMA = 1.0
EPOCHS = 1  # 1 epoch of fine-tuning on top of converged ckpt

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(6)

def log(m):
    print("[c7] %s" % m, flush=True)

def mf1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))

def load_csv(name):
    df = pd.read_csv(ROOT / "data" / "processed" / name)
    return df["text"].astype(str).tolist(), np.array([L2I[l] for l in df["label"]])

def build_model_from_ctl():
    """Load the converged uniform control checkpoint (difficulty_best_ctl.pt)."""
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    ck = torch.load(EXP / "difficulty_best_ctl.pt", map_location="cpu")
    sd = ck["model_state_dict"]
    model.classifier = nn.Linear(768, int(sd["classifier.weight"].shape[0]))
    model.load_state_dict(sd, strict=False)
    log("loaded ctl ckpt: val_macro_f1=%.4f" % ck.get("val_macro_f1", 0))
    return model

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tr_t, tr_y = load_csv("bert_train.csv")
    va_t, va_y = load_csv("bert_val.csv")
    te_t, te_y = load_csv("test.csv")
    log("train=%d val=%d test=%d" % (len(tr_t), len(va_t), len(te_y)))

    def enc(texts):
        e = tok(texts, truncation=True, max_length=MAXLEN, padding="max_length")
        return np.array(e["input_ids"]), np.array(e["attention_mask"])

    tr_ids, tr_m = enc(tr_t)
    va_ids, va_m = enc(va_t)
    te_ids, te_m = enc(te_t)

    # Difficulty signal from Tier-1 on TRAIN rows
    p_tr = np.load(EXP / "tier1_train_proba.npy")
    s_tr = p_tr.max(1)
    w_raw = np.clip(1.0 - s_tr, 0.0, None) ** GAMMA
    w = w_raw / w_raw.mean()
    log("gamma=%.1f  weights min=%.3f med=%.3f max=%.3f" % (
        GAMMA, w.min(), np.median(w), w.max()))
    log("s<0.5 rows are %.1f%% of data but carry %.1f%% of gradient mass"
        % (100 * (s_tr < 0.5).mean(), 100 * w[s_tr < 0.5].sum() / w.sum()))

    model = build_model_from_ctl()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = len(tr_y) * EPOCHS // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                                pct_start=0.1)

    def batches(ids, mask, y, w=None):
        for s in range(0, len(y), BS):
            b = slice(s, min(s + BS, len(y)))
            yield (torch.from_numpy(ids[b]), torch.from_numpy(mask[b]).float(),
                   torch.from_numpy(y[b]),
                   torch.from_numpy(w[b]) if w is not None else None)

    best_v = -1.0
    for ep in range(EPOCHS):
        t0 = time.time()
        for ii, mm, yy, ww in batches(tr_ids, tr_m, tr_y, w):
            opt.zero_grad()
            ce = nn.functional.cross_entropy(model(input_ids=ii,
                                                   attention_mask=mm).logits,
                                             yy, reduction="none")
            loss = (ce * ww).mean() if ww is not None else ce.mean()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            v = mf1(va_y, model(input_ids=torch.from_numpy(va_ids),
                                attention_mask=torch.from_numpy(va_m)
                                ).logits.argmax(-1).numpy())
        if v > best_v:
            best_v = float(v)
            torch.save(dict(model_state_dict=model.state_dict(),
                            val_macro_f1=v, gamma=GAMMA, epoch=ep + 1,
                            init="difficulty_best_ctl.pt"),
                       EXP / "difficulty_best_c7.pt")
        log("epoch %d val mF1 %.4f (%.0fs)" % (ep + 1, v, time.time() - t0))

    # Test evaluation
    model.load_state_dict(torch.load(EXP / "difficulty_best_c7.pt",
                                      map_location="cpu")["model_state_dict"])
    model.eval()
    proba = np.zeros((len(te_y), 3))
    with torch.no_grad():
        for s in range(0, len(te_y), 64):
            b = slice(s, min(s + 64, len(te_y)))
            e = tok(te_t[b], truncation=True, max_length=MAXLEN,
                    padding="max_length", return_tensors="pt")
            proba[b] = torch.softmax(model(**e).logits, -1).numpy()
    pred = proba.argmax(1)
    p1 = np.load(EXP / "tier1_test_proba.npy")
    s_te, pred1 = p1.max(1), p1.argmax(1)
    pref = np.load(EXP / "teacher_test_proba.npy")
    pred_ref = pref.argmax(1)
    p_ctl = np.load(EXP / "difficulty_test_proba_ctl.npy")
    pred_ctl = p_ctl.argmax(1)

    res = dict(gamma=GAMMA, init="difficulty_best_ctl.pt", epochs=EPOCHS,
               val_macro_f1=best_v,
               test=dict(acc=float((pred == te_y).mean()),
                         macro_f1=mf1(te_y, pred),
                         neutral_f1=float(f1_score(te_y, pred, average=None,
                                                   labels=[1])[0])),
               reference_tuned=dict(macro_f1=mf1(te_y, pred_ref)),
               uniform_ctl=dict(macro_f1=mf1(te_y, pred_ctl),
                                neutral_f1=float(f1_score(te_y, pred_ctl,
                                                         average=None, labels=[1])[0])),
               original_difficulty=dict(macro_f1=0.7033, neutral_f1=0.4234),
               tier1=dict(macro_f1=mf1(te_y, pred1),
                          acc=float((pred1 == te_y).mean())))

    # Confidence bins
    bins, lo = [], 0.0
    for hi in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]:
        m = (s_te >= lo) & (s_te < hi)
        bin_lo = lo
        lo = hi
        if m.sum() == 0:
            continue
        bins.append(dict(lo=float(round(bin_lo, 2)), hi=float(hi), n=int(m.sum()),
                         tier1_mf1=mf1(te_y[m], pred1[m]),
                         tuned_mf1=mf1(te_y[m], pred_ref[m]),
                         c7_mf1=mf1(te_y[m], pred[m]),
                         c7_minus_tier1=mf1(te_y[m], pred[m]) - mf1(te_y[m], pred1[m]),
                         c7_minus_tuned=mf1(te_y[m], pred[m]) - mf1(te_y[m], pred_ref[m])))
    res["bins"] = bins

    np.save(EXP / "difficulty_test_proba_c7.npy", proba)
    with open(EXP / "difficulty_meta_c7.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    log("=== C7 TWO-STAGE RESULTS ===")
    log("C7 (two-stage):        test mF1 %.4f  neutral F1 %.4f" % (
        res["test"]["macro_f1"], res["test"]["neutral_f1"]))
    log("Uniform ctl (stage 1): test mF1 %.4f  neutral F1 %.4f" % (
        res["uniform_ctl"]["macro_f1"], res["uniform_ctl"]["neutral_f1"]))
    log("Original difficulty:   test mF1 %.4f  neutral F1 %.4f" % (
        0.7033, 0.4234))
    log("Published tuned:       test mF1 %.4f" % res["reference_tuned"]["macro_f1"])
    log("Tier-1:                test mF1 %.4f" % res["tier1"]["macro_f1"])
    log("wrote experiments/difficulty_meta_c7.json")


if __name__ == "__main__":
    main()
