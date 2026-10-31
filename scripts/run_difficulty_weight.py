"""
Difficulty-weighted fine-tuning of the (expensive) second tier.

Motivation (方案 b / item 2): the closed-form cascade analysis shows the whole
cascade gain is  rho * Delta, where rho is the fraction of traffic routed to
the expensive tier and Delta the repair it buys.  Measured on our data the
repair region is narrow (8.2% of traffic) because the tuned DistilBERT tier
is *not worse* than the TF-IDF tier above roughly s >= 0.5: at s in [0.5,0.6)
the two are level (0.5634 vs 0.5580) and above it the Transformer loses.
So the binding constraint is not the router but the *conditioning of the
second tier on the first tier's difficulty signal*.

We therefore re-train the second tier with the first tier's posterior
confidence s(x) = max_k p_k(x) as a per-sample difficulty weight
    w(x) = (1 - s(x))^gamma,   normalised to mean 1,
so that the gradient mass is put on the reviews the cheap tier is unsure
about, and the expensive tier stops spending capacity where it cannot help.

Outputs  experiments/difficulty_weight.json, experiments/difficulty_test_proba.npy
Usage    python scripts/run_difficulty_weight.py [--gamma 1.0] [--no-heads]
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
from transformers import AutoModel, AutoTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED = 42
MAXLEN = 256
BS = 16
LR = 5e-5
WD = 0.01
EPOCHS = 3
NL = 6
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


def tier1(train_text, train_y, test_text, test_y):
    vec = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2),
                          max_features=30000, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    clf.fit(vec.fit_transform(train_text), train_y)
    proba = clf.predict_proba(vec.transform(test_text))
    return proba, proba.max(1), proba.argmax(1), mf1(test_y, proba.argmax(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--no-heads", action="store_true")
    args = ap.parse_args()

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

    # ---- first-tier difficulty signal on the *training* rows
    _, s_tr, _, _ = tier1(tr_t, tr_y, tr_t, tr_y)
    w_raw = np.clip(1.0 - s_tr, 0.0, None) ** args.gamma
    w = w_raw / w_raw.mean()
    log("difficulty weights: gamma=%.1f  min=%.3f med=%.3f max=%.3f mean=%.3f"
        % (args.gamma, w.min(), np.median(w), w.max(), w.mean()))
    log("weight share on s<0.5 rows (%.1f%% of data): %.1f%% of gradient mass"
        % (100 * (s_tr < 0.5).mean(), 100 * w[s_tr < 0.5].sum() / w.sum()))

    base = AutoModel.from_pretrained("distilbert-base-uncased").to(DEV)
    head = nn.Linear(768, 3).to(DEV)

    def batches(ids, mask, y, w=None):
        for s in range(0, len(y), BS):
            b = slice(s, min(s + BS, len(y)))
            yield (torch.from_numpy(ids[b]), torch.from_numpy(mask[b]),
                   torch.from_numpy(y[b]),
                   torch.from_numpy(w[b]) if w is not None else None)

    model = nn.ModuleDict(dict(base=base, head=head))
    model.train()
    opt = torch.optim.AdamW(list(base.parameters()) + list(head.parameters()),
                            lr=LR, weight_decay=WD)
    steps = len(tr_y) * args.epochs // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                                pct_start=0.1)
    best_v = -1.0
    history = []
    for ep in range(args.epochs):
        t0 = time.time()
        for ii, mm, yy, ww in batches(tr_ids, tr_m, tr_y, w):
            opt.zero_grad()
            h = base(input_ids=ii, attention_mask=mm).last_hidden_state[:, 0]
            logits = head(h)
            ce = nn.functional.cross_entropy(logits, yy, reduction="none")
            loss = (ce * ww).mean() if ww is not None else ce.mean()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            h = base(input_ids=torch.from_numpy(va_ids),
                     attention_mask=torch.from_numpy(va_m)).last_hidden_state[:, 0]
            v = mf1(va_y, head(h).argmax(-1).numpy())
        history.append(dict(epoch=ep + 1, val_macro_f1=float(v),
                            seconds=round(time.time() - t0, 1)))
        if v > best_v:
            best_v = float(v)
            torch.save(dict(base=base.state_dict(), head=head.state_dict()),
                       EXP / "difficulty_best.pt")
        log("epoch %d val mF1 %.4f (%.0fs)" % (ep + 1, v, time.time() - t0))

    # ---- evaluate on the full 20k test set
    ck = torch.load(EXP / "difficulty_best.pt", map_location="cpu")
    base.load_state_dict(ck["base"])
    head.load_state_dict(ck["head"])
    base.eval()
    head.eval()
    proba = np.zeros((len(te_y), 3))
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(te_y), 64):
            b = slice(s, min(s + 64, len(te_y)))
            h = base(input_ids=torch.from_numpy(te_ids[b]),
                     attention_mask=torch.from_numpy(te_m[b])).last_hidden_state[:, 0]
            proba[b] = torch.softmax(head(h), -1).numpy()
    log("20k eval %.0fs" % (time.time() - t0))
    pred = proba.argmax(1)
    acc = float((pred == te_y).mean())
    res = dict(gamma=args.gamma, val_macro_f1=best_v,
               test=dict(acc=acc, macro_f1=mf1(te_y, pred),
                         neutral_f1=float(f1_score(te_y, pred, average=None,
                                                   labels=[1])[0]),
                         per_class={k: float(v) for k, v in zip(
                             LABELS, f1_score(te_y, pred, average=None))}))

    # ---- reference: the published tuned checkpoint's 20k probabilities
    ref_p = np.load(EXP / "probe_teacher_test_proba.npy")
    ref_pred = ref_p.argmax(1)
    res["reference_tuned"] = dict(acc=float((ref_pred == te_y).mean()),
                                  macro_f1=mf1(te_y, ref_pred))

    # ---- where does the difference sit?  bin the test rows by tier-1 confidence
    _, s_te, pred1, m1 = tier1(tr_t, tr_y, te_t, te_y)
    edges = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s_te >= lo) & (s_te < hi)
        if m.sum() == 0:
            continue
        bins.append(dict(
            band="[%.2f,%.2f)" % (lo, hi), n=int(m.sum()),
            tier1_mf1=mf1(te_y[m], pred1[m]),
            tuned_mf1=mf1(te_y[m], ref_pred[m]),
            diff_mf1=mf1(te_y[m], pred[m]),
            diff_minus_tier1=mf1(te_y[m], pred[m]) - mf1(te_y[m], pred1[m]),
            diff_minus_tuned=mf1(te_y[m], pred[m]) - mf1(te_y[m], ref_pred[m])))
    res["bins"] = bins

    # ---- optional: per-layer heads on top of the difficulty-weighted trunk
    if not args.no_heads:
        heads = nn.ModuleList([nn.Sequential(nn.Dropout(0.1), nn.Linear(768, 3))
                               for _ in range(NL)]).to(DEV)
        for p in base.parameters():
            p.requires_grad = False
        opt = torch.optim.AdamW(heads.parameters(), lr=1e-3, weight_decay=0.0)
        steps = len(tr_y) * 3 // BS
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3,
                                                    total_steps=steps, pct_start=0.1)
        base.eval()
        for ep in range(3):
            heads.train()
            t0 = time.time()
            for ii, mm, yy, ww in batches(tr_ids, tr_m, tr_y, w):
                opt.zero_grad()
                h = base(input_ids=ii, attention_mask=mm).last_hidden_state[:, 0]
                loss = sum(nn.functional.cross_entropy(heads[li](h), yy)
                           for li in range(NL)) / NL
                loss.backward()
                opt.step()
                sched.step()
            heads.eval()
            with torch.no_grad():
                h = base(input_ids=torch.from_numpy(va_ids),
                         attention_mask=torch.from_numpy(va_m)).last_hidden_state[:, 0]
                vals = [mf1(va_y, heads[li](h).argmax(-1).numpy()) for li in range(NL)]
            log("heads epoch %d val mF1 %s (%.0fs)"
                % (ep + 1, [round(v, 4) for v in vals], time.time() - t0))
        layer_test = []
        with torch.no_grad():
            h_all = [np.zeros((len(te_y), 3)) for _ in range(NL)]
            for s in range(0, len(te_y), 64):
                b = slice(s, min(s + 64, len(te_y)))
                ii = torch.from_numpy(te_ids[b])
                mm = torch.from_numpy(te_m[b])
                cur = base.embeddings(ii, mm)
                states = [cur[:, 0]]
                for layer in list(base.trunk.layer):
                    cur = layer(cur, mm)[0]
                    states.append(cur[:, 0])
                for li in range(NL):
                    h_all[li][b] = torch.softmax(heads[li](states[li]), -1).numpy()
        for li in range(NL):
            p = h_all[li].argmax(1)
            layer_test.append(dict(layer=li + 1, macro_f1=mf1(te_y, p),
                                   acc=float((p == te_y).mean())))
        res["per_layer"] = layer_test
        # cascade with the cheapest exit
        c1 = 0.483
        c2 = 7.5  # measured for a 2-layer exit in run_early_exit.py
        pareto = []
        for li in range(NL):
            for tau in np.round(np.arange(0.05, 1.0, 0.025), 3):
                esc = s_te < tau
                r = float(esc.mean())
                if r <= 0:
                    continue
                p = pred1.copy()
                p[esc] = h_all[li][esc].argmax(1)
                pareto.append(dict(layer=li + 1, tau=float(tau), r=r,
                                   latency=float(c1 + r * c2),
                                   macro_f1=mf1(te_y, p),
                                   acc=float((p == te_y).mean())))
        for B in [1.0, 2.0, 5.0, 10.0]:
            feas = [x for x in pareto if x["latency"] <= B]
            if feas:
                b = max(feas, key=lambda x: x["macro_f1"])
                res.setdefault("cascade_cheap", {})["B=%.0fms" % B] = dict(
                    layer=b["layer"], tau=b["tau"], r=b["r"], latency=b["latency"],
                    macro_f1=b["macro_f1"])

    np.save(EXP / "difficulty_test_proba.npy", proba)
    res["history"] = history
    res["tier1_macro_f1"] = float(m1)
    with open(EXP / "difficulty_weight.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    log("test mF1 %.4f (tuned ckpt %.4f, tier1 %.4f)"
        % (res["test"]["macro_f1"], res["reference_tuned"]["macro_f1"], m1))
    for b in bins:
        log("%s n=%5d  tier1 %.4f  tuned %.4f  diff %.4f  (vs T1 %+.4f)"
            % (b["band"], b["n"], b["tier1_mf1"], b["tuned_mf1"], b["diff_mf1"],
               b["diff_minus_tier1"]))
    log("wrote experiments/difficulty_weight.json")


if __name__ == "__main__":
    main()
