"""
Difficulty-weighted fine-tuning of the (expensive) second tier.

Why: the cascade gain is rho * Delta, rho = fraction of traffic escalated to
the expensive tier, Delta = the repair it buys.  Measured, the repair region is
narrow (8.2% of traffic) because the tuned DistilBERT tier is not better than
the TF-IDF tier above s >= 0.5 and loses above it (level at [0.5,0.6)).  So the
binding constraint is not the router but how strongly the expensive tier is
conditioned on the cheap tier's difficulty signal.

We re-train the expensive tier with the cheap tier's posterior confidence on the
TRAIN rows, s(x) = max_k p_k(x), as a per-sample weight
    w(x) = (1 - s(x))^gamma,  normalised to mean 1,
so gradient mass moves to the reviews the cheap tier is unsure about.  This is
purely a training-side change: inference cost is untouched.

Reference for the comparison: the published tuned checkpoint's 20k probabilities
live in probe_teacher_test_proba.npy and are reused as the unweighted baseline.

Outputs  experiments/difficulty_weight.json, experiments/difficulty_test_proba.npy
Usage    python scripts/run_difficulty_weight_v2.py --gamma 1.0
"""
from __future__ import annotations

import argparse
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
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from transformers import AutoTokenizer, DistilBertForSequenceClassification  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED = 42
MAXLEN = 256
BS = 16
LR = 5e-5
WD = 0.01
EPOCHS = 3
HEAD_EPOCHS = 3
NL = 6
MODEL_DIR = "models/distilbert_local"
LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}
DEV = torch.device("cpu")
EXP = ROOT / "experiments"


def log(m):
    print("[dw] %s" % m, flush=True)


def mf1(y, p):
    return float(f1_score(y, p, average="macro"))


def load_csv(name):
    df = pd.read_csv(ROOT / "data" / "processed" / name)
    return df["text"].astype(str).tolist(), np.array([L2I[l] for l in df["label"]])


def tier1():
    """Load the cached strong Tier-1 probabilities (200k-row TF-IDF + LogReg).

    Re-fitting the cheap tier inside this script would reproduce the 0.6087
    three-thousand-row model, not the 0.7550 model the paper publishes, so the
    cached probabilities produced by scripts/_build_tier1.py are reused instead.
    """
    p_tr = np.load(EXP / "tier1_train_proba.npy")
    p_te = np.load(EXP / "tier1_test_proba.npy")
    return p_tr, p_te, p_tr.max(1), p_te.max(1), p_te.argmax(1)


def build_model():
    """published tuned checkpoint, 3-class head swapped in.

    models/distilbert_local ships a two-class classifier while the tuned
    checkpoint holds a three-class one; loading the checkpoint with strict=False
    after replacing the head reproduces the published tier exactly (the deepest
    early-exit exit measured 0.7089 macro-F1, matching the paper).
    """
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    raw = torch.load(EXP / "bert_tuned" / "best.pt", map_location="cpu")
    sd = raw["model_state_dict"] if "model_state_dict" in raw else raw
    head3 = nn.Linear(768, 3)
    head3.weight.data.copy_(sd["classifier.weight"])
    head3.bias.data.copy_(sd["classifier.bias"])
    model.classifier = head3
    model.load_state_dict({k: v for k, v in sd.items()
                           if not k.startswith("classifier")}, strict=False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--no-heads", action="store_true")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for the output files, so that a control run "
                         "(gamma=0, identical code path) can share the "
                         "experiments directory")
    args = ap.parse_args()
    TAG = args.tag
    suffix = ("_" + TAG) if TAG else ""
    CKPT_OUT = EXP / ("difficulty_best%s.pt" % suffix)
    PROBA_OUT = EXP / ("difficulty_test_proba%s.npy" % suffix)
    JSON_OUT = EXP / ("difficulty_weight%s.json" % suffix)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(6)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tr_t, tr_y = load_csv("bert_train.csv")
    va_t, va_y = load_csv("bert_val.csv")
    te_t, te_y = load_csv("test.csv")
    log("train=%d val=%d test=%d" % (len(tr_t), len(va_t), len(te_t)))

    def enc(texts):
        e = tok(texts, truncation=True, max_length=MAXLEN, padding="max_length")
        return np.array(e["input_ids"]), np.array(e["attention_mask"])

    tr_ids, tr_m = enc(tr_t)
    va_ids, va_m = enc(va_t)
    te_ids, te_m = enc(te_t)

    # ---------- difficulty signal from the cheap tier, on TRAIN rows ----------
    p_tr, p_te, s_tr, s_te, pred1 = tier1()
    assert len(p_tr) == len(tr_y), "cached Tier-1 train proba, row mismatch"
    assert len(p_te) == len(te_y), "cached Tier-1 test proba, row mismatch"
    w_raw = np.clip(1.0 - s_tr, 0.0, None) ** args.gamma
    w = w_raw / w_raw.mean()
    log("gamma=%.1f  weights min=%.3f med=%.3f max=%.3f" % (
        args.gamma, w.min(), np.median(w), w.max()))
    log("s<0.5 rows are %.1f%% of the data but carry %.1f%% of the gradient mass"
        % (100 * (s_tr < 0.5).mean(), 100 * w[s_tr < 0.5].sum() / w.sum()))

    model = build_model()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = len(tr_y) * args.epochs // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                                pct_start=0.1)

    def batches(ids, mask, y, w=None):
        for s in range(0, len(y), BS):
            b = slice(s, min(s + BS, len(y)))
            yield (torch.from_numpy(ids[b]), torch.from_numpy(mask[b]).float(),
                   torch.from_numpy(y[b]),
                   torch.from_numpy(w[b]) if w is not None else None)

    best_v, history = -1.0, []
    for ep in range(args.epochs):
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
        history.append(dict(epoch=ep + 1, val_macro_f1=float(v),
                            seconds=round(time.time() - t0, 1)))
        if v > best_v:
            best_v = float(v)
            torch.save(dict(model_state_dict=model.state_dict(), val_macro_f1=v,
                            gamma=args.gamma, epoch=ep + 1),
                       CKPT_OUT)
        log("epoch %d val mF1 %.4f (%.0fs)" % (ep + 1, v, time.time() - t0))

    # ---------- 20k test evaluation ----------
    proba = np.zeros((len(te_y), 3))
    model.eval()
    with torch.no_grad():
        for s in range(0, len(te_y), 64):
            b = slice(s, min(s + 64, len(te_y)))
            proba[b] = torch.softmax(
                model(input_ids=torch.from_numpy(te_ids[b]),
                      attention_mask=torch.from_numpy(te_m[b]).float()).logits,
                -1).numpy()
    pred = proba.argmax(1)
    res = dict(gamma=args.gamma, val_macro_f1=best_v,
               test=dict(acc=float((pred == te_y).mean()),
                         macro_f1=mf1(te_y, pred),
                         neutral_f1=float(f1_score(te_y, pred, average=None,
                                                   labels=[1])[0]),
                         per_class={k: float(v) for k, v in zip(
                             LABELS, f1_score(te_y, pred, average=None))}))
    log("difficulty-weighted test mF1 %.4f" % res["test"]["macro_f1"])

    # ---------- where does the difference sit? ----------
    ref_p = np.load(EXP / "teacher_test_proba.npy")
    ref_pred = ref_p.argmax(1)
    res["reference_tuned"] = dict(acc=float((ref_pred == te_y).mean()),
                                  macro_f1=mf1(te_y, ref_pred))
    res["tier1"] = dict(macro_f1=mf1(te_y, pred1),
                        acc=float((pred1 == te_y).mean()))
    bins, lo = [], 0.0
    for hi in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]:
        m = (s_te >= lo) & (s_te < hi)
        lo = hi
        if m.sum() == 0:
            continue
        bins.append(dict(
            lo=float(m_mask_min(s_te, m)), hi=float(hi), n=int(m.sum()),
            tier1_mf1=mf1(te_y[m], pred1[m]),
            tuned_mf1=mf1(te_y[m], ref_pred[m]),
            diff_mf1=mf1(te_y[m], pred[m]),
            diff_minus_tier1=mf1(te_y[m], pred[m]) - mf1(te_y[m], pred1[m]),
            diff_minus_tuned=mf1(te_y[m], pred[m]) - mf1(te_y[m], ref_pred[m])))
    res["bins"] = bins

    # ---------- optional: per-layer heads on the difficulty-weighted trunk ----
    if not args.no_heads:
        ck = torch.load(EXP / "difficulty_best.pt", map_location="cpu")
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        full_layers = list(model.distilbert.transformer.layer)

        def hidden_all(ids, mask, batch=32):
            ids = torch.as_tensor(ids)
            mask = torch.as_tensor(mask).float()
            out = [np.zeros((len(ids), 768)) for _ in range(NL)]
            with torch.no_grad():
                for s in range(0, len(ids), batch):
                    b = slice(s, min(s + batch, len(ids)))
                    db = model.distilbert(input_ids=ids[b], attention_mask=mask[b],
                                          output_hidden_states=True)
                    for li in range(NL):
                        out[li][b] = db.hidden_states[li + 1][:, 0].numpy()
            return out

        h_tr = hidden_all(tr_ids, tr_m)
        h_va = hidden_all(va_ids, va_m)
        heads = nn.ModuleList([nn.Sequential(nn.Dropout(0.1), nn.Linear(768, 3))
                               for _ in range(NL)]).to(DEV)
        for p in model.parameters():
            p.requires_grad = False
        opt = torch.optim.AdamW(heads.parameters(), lr=1e-3, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3,
                                                    total_steps=HEAD_EPOCHS,
                                                    pct_start=0.1)
        for ep in range(HEAD_EPOCHS):
            heads.train()
            opt.zero_grad()
            loss = sum(nn.functional.cross_entropy(
                heads[li](torch.from_numpy(h_tr[li]).float()),
                torch.from_numpy(tr_y)) for li in range(NL)) / NL
            loss.backward()
            opt.step()
            sched.step()
            heads.eval()
            with torch.no_grad():
                vals = [mf1(va_y, heads[li](
                    torch.from_numpy(h_va[li]).float()).argmax(1).numpy())
                    for li in range(NL)]
            log("head round %d val mF1 %s" % (ep + 1, [round(v, 4) for v in vals]))

        h_te = hidden_all(te_ids, te_m)
        layer_prob = [torch.softmax(heads[li](torch.from_numpy(h_te[li]).float()),
                                    -1).numpy() for li in range(NL)]
        per_layer = []
        for li in range(NL):
            p = layer_prob[li].argmax(1)
            med, _ = bench_depth(model, full_layers, li + 1, te_ids, te_m, n=20)
            per_layer.append(dict(layer=li + 1, latency_ms=med,
                                  macro_f1=mf1(te_y, p),
                                  acc=float((p == te_y).mean())))
            log("exit layer %d: %.2f ms mF1 %.4f" % (li + 1, med,
                                                     per_layer[-1]["macro_f1"]))
        res["per_layer"] = per_layer

        # cascade with the cheap exits
        c1, c2 = 0.483, per_layer[0]["latency_ms"]
        pareto = []
        for li in range(NL):
            for tau in np.round(np.arange(0.05, 1.0, 0.025), 3):
                esc = s_te < tau
                r = float(esc.mean())
                if r <= 0:
                    continue
                p = pred1.copy()
                p[esc] = layer_prob[li][esc].argmax(1)
                pareto.append(dict(layer=li + 1, tau=float(tau), r=r,
                                   latency=float(c1 + r * per_layer[li]["latency_ms"]),
                                   macro_f1=mf1(te_y, p),
                                   acc=float((p == te_y).mean())))
        for B in [1.0, 2.0, 5.0, 10.0]:
            feas = [x for x in pareto if x["latency"] <= B]
            if feas:
                b = max(feas, key=lambda x: x["macro_f1"])
                res.setdefault("cascade", {})["B=%.0fms" % B] = dict(
                    layer=b["layer"], tau=b["tau"], r=b["r"],
                    latency=b["latency"], macro_f1=b["macro_f1"])
        res["cascade"]["c2_used"] = c2

    np.save(PROBA_OUT, proba)
    res["history"] = history
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    log("test mF1 %.4f (published tuned %.4f, tier1 %.4f)"
        % (res["test"]["macro_f1"], res["reference_tuned"]["macro_f1"],
           res["tier1"]["macro_f1"]))
    for b in bins:
        log("s in [%.2f,%.2f) n=%5d  tier1 %.4f  tuned %.4f  diff %.4f  (vs T1 %+.4f)"
            % (b["lo"], b["hi"], b["n"], b["tier1_mf1"], b["tuned_mf1"],
               b["diff_mf1"], b["diff_minus_tier1"]))
    log("wrote experiments/difficulty_weight.json")


def m_mask_min(s, m):
    return float(s[m].min())


def bench_depth(model, full_layers, depth, ids, mask, n=20):
    """Single-request latency of an exit after `depth` layers (official API)."""
    model.eval()
    saved = model.distilbert.transformer.layer
    model.distilbert.transformer.layer = nn.ModuleList(list(full_layers)[:depth])
    times = []
    with torch.no_grad():
        for i in range(n):
            ii = torch.as_tensor(ids[i:i + 1])
            mm = torch.as_tensor(mask[i:i + 1]).float()
            t0 = time.perf_counter()
            model.distilbert(input_ids=ii, attention_mask=mm)
            times.append((time.perf_counter() - t0) * 1000.0)
    model.distilbert.transformer.layer = saved
    return float(np.median(times)), float(np.percentile(times, 95))


if __name__ == "__main__":
    main()
