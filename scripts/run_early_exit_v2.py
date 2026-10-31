"""
Depth-adaptive second tier (DeeBERT-style per-layer exits) for the cascade.

Why: the cascade is dominated because the expensive tier costs c_2 = 44.81 ms
while the repair it buys is worth ~0.5 macro-F1 points.  The paper already
pins the admissible range of c_2 with one inequality, so the only lever that is
not saturated is c_2 itself.  We turn the expensive second tier into a stack of
per-layer classification heads: a review that is escalated exits at a middle
layer, so the depth -- and therefore the cost -- adapts to the latency budget.

Implementation note: transformers 5.x does not accept the attention mask in a
manually invoked per-layer call, so every trunk call below goes through the
official API -- quality is evaluated from one output_hidden_states=True forward,
and latency is measured by truncating the trunk layer list to the first l layers
and then calling the model normally.

Outputs  experiments/early_exit.json
Usage    python scripts/run_early_exit_v2.py
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
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from transformers import AutoTokenizer, DistilBertForSequenceClassification  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED = 42
MAXLEN = 256
BS = 16
HEAD_EPOCHS = 3
HEAD_LR = 1e-3
NL = 6
MODEL_DIR = "models/distilbert_local"
CKPT = "experiments/bert_tuned/best.pt"
LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}
DEV = torch.device("cpu")
EXP = ROOT / "experiments"


def log(m):
    print("[ee] %s" % m, flush=True)


def mf1(y, p):
    return float(f1_score(y, p, average="macro"))


def load_csv(name):
    df = pd.read_csv(ROOT / "data" / "processed" / name)
    return df["text"].astype(str).tolist(), np.array([L2I[l] for l in df["label"]])


def full_hidden(model, ids, mask, batch=32):
    """[CLS] hidden state after every transformer layer, one forward per batch."""
    ids = torch.as_tensor(ids)
    mask = torch.as_tensor(mask).float()
    out = [np.zeros((len(ids), 768), dtype=np.float32) for _ in range(NL)]
    with torch.no_grad():
        for s in range(0, len(ids), batch):
            b = slice(s, min(s + batch, len(ids)))
            db = model.distilbert(input_ids=ids[b], attention_mask=mask[b],
                                  output_hidden_states=True)
            for li in range(NL):
                out[li][b] = db.hidden_states[li + 1][:, 0].numpy()
    return out


def bench_depth(model, full_layers, depth, ids, mask, n=30):
    """Single-request latency of the tier exiting after `depth` layers."""
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


def make_exit_head():
    """A shallow classifier head attached to one layer's [CLS] state."""
    return nn.Sequential(nn.Dropout(0.1), nn.Linear(768, 768), nn.ReLU(),
                         nn.Dropout(0.1), nn.Linear(768, 3))


def main():
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

    # ---------- trunk: the published tuned checkpoint ----------
    ck = torch.load(ROOT / CKPT, map_location="cpu")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    sd = ck["model_state_dict"]
    # the checkpoint was trained for three classes while the on-disk config
    # still declares two -> drop only the head, keep pre_classifier
    model.load_state_dict({k: v for k, v in sd.items()
                           if not k.startswith("classifier")}, strict=False)
    head3 = nn.Linear(768, 3).to(DEV)
    head3.weight.data.copy_(sd["classifier.weight"])
    head3.bias.data.copy_(sd["classifier.bias"])
    model.classifier = head3
    model.eval()
    log("loaded trunk from %s (epoch %s, val mF1 %.4f)"
        % (CKPT, ck.get("epoch"), ck.get("val_macro_f1", float("nan"))))
    with torch.no_grad():
        pv = model(input_ids=torch.from_numpy(va_ids),
                   attention_mask=torch.from_numpy(va_m)).logits.argmax(1).numpy()
        log("trunk sanity: val mF1 %.4f, val acc %.4f"
            % (mf1(va_y, pv), (pv == va_y).mean()))

    full_layers = list(model.distilbert.transformer.layer)
    for p in model.parameters():
        p.requires_grad = False

    # ---------- stage 2: one head per layer, trunk frozen ----------
    h_tr = full_hidden(model, tr_ids, tr_m)
    h_va = full_hidden(model, va_ids, va_m)
    heads = nn.ModuleList([make_exit_head() for _ in range(NL)]).to(DEV)
    # the deepest exit is exactly the published classifier (pre_classifier + head)
    heads[NL - 1] = nn.Sequential(model.pre_classifier, nn.ReLU(),
                                  head3).to(DEV)
    opt = torch.optim.AdamW(heads.parameters(), lr=HEAD_LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=HEAD_LR,
                                                total_steps=HEAD_EPOCHS
                                                * ((len(tr_y) + BS - 1) // BS),
                                                pct_start=0.1)
    perm = np.random.RandomState(SEED).permutation(len(tr_y))
    for ep in range(HEAD_EPOCHS):
        heads.train()
        for s0 in range(0, len(perm), BS):
            bidx = torch.from_numpy(perm[s0:s0 + BS])
            yy = torch.from_numpy(tr_y[perm[s0:s0 + BS]])
            opt.zero_grad()
            loss = 0.0
            for li in range(NL):
                loss = loss + nn.functional.cross_entropy(
                    heads[li](torch.from_numpy(h_tr[li][perm[s0:s0 + BS]])), yy)
            (loss / NL).backward()
            opt.step()
            sched.step()
        heads.eval()
        with torch.no_grad():
            vals = [mf1(va_y, heads[li](torch.from_numpy(h_va[li]).float()).argmax(1).numpy())
                    for li in range(NL)]
        log("head round %d  val mF1 per layer %s" % (ep + 1,
                                                     [round(v, 4) for v in vals]))

    # ---------- quality of every exit on the 20k test set ----------
    h_te = full_hidden(model, te_ids, te_m)
    with torch.no_grad():
        proba_by_layer = [torch.softmax(
            heads[li](torch.from_numpy(h_te[li])), -1).numpy()
            for li in range(NL)]
    per_layer = []
    for li in range(NL):
        pred = proba_by_layer[li].argmax(1)
        med, p95 = bench_depth(model, full_layers, li + 1, te_ids, te_m, n=30)
        per_layer.append(dict(
            layer=li + 1, latency_ms=med, latency_p95_ms=p95,
            macro_f1=mf1(te_y, pred),
            acc=float((pred == te_y).mean()),
            neutral_f1=float(f1_score(te_y, pred, average=None, labels=[1])[0])))
        log("exit layer %d: %.2f ms  mF1 %.4f  acc %.4f  neutralF1 %.4f"
            % (li + 1, med, per_layer[-1]["macro_f1"], per_layer[-1]["acc"],
               per_layer[-1]["neutral_f1"]))

    # ---------- first tier (TF-IDF + LogReg) ----------
    vec = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2),
                          max_features=30000, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    clf.fit(vec.fit_transform(tr_t), tr_y)
    pt = clf.predict_proba(vec.transform(te_t))
    s1 = pt.max(1)
    pred1 = pt.argmax(1)
    tier1_mf1 = mf1(te_y, pred1)
    log("tier1 (TF-IDF+LogReg) test mF1 %.4f" % tier1_mf1)

    # ---------- cascade re-analysis with the cheap exits ----------
    c1 = 0.483
    pareto = []
    for li in range(NL):
        c2 = per_layer[li]["latency_ms"]
        for tau in np.round(np.arange(0.05, 1.0, 0.025), 3):
            esc = s1 < tau
            r = float(esc.mean())
            if r <= 0:
                continue
            p = pred1.copy()
            p[esc] = proba_by_layer[li][esc].argmax(1)
            pareto.append(dict(layer=li + 1, tau=float(tau), r=r, c1=c1, c2=c2,
                               latency=float(c1 + r * c2),
                               acc=float((p == te_y).mean()),
                               macro_f1=mf1(te_y, p)))
    budget = {}
    for B in [1.0, 2.0, 5.0, 10.0, 15.0]:
        feas = [x for x in pareto if x["latency"] <= B]
        if feas:
            b = max(feas, key=lambda x: x["macro_f1"])
            budget["B=%.0fms" % B] = dict(
                B=B, layer=b["layer"], tau=b["tau"], r=b["r"], c2=b["c2"],
                latency=b["latency"], acc=b["acc"], macro_f1=b["macro_f1"])
            log("B=%.0fms -> layer %d tau %.3f r %.3f %.2f ms mF1 %.4f"
                % (B, b["layer"], b["tau"], b["r"], b["latency"], b["macro_f1"]))

    li_cheap = min(range(NL), key=lambda i: per_layer[i]["latency_ms"])
    bands = []
    edges = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s1 >= lo) & (s1 < hi)
        if m.sum() == 0:
            continue
        row = dict(band="[%.2f,%.2f)" % (lo, hi), n=int(m.sum()),
                   tier1_mf1=mf1(te_y[m], pred1[m]),
                   full_mf1=mf1(te_y[m], proba_by_layer[NL - 1][m].argmax(1)),
                   cheap_mf1=mf1(te_y[m], proba_by_layer[li_cheap][m].argmax(1)))
        row["cheap_minus_tier1"] = row["cheap_mf1"] - row["tier1_mf1"]
        bands.append(row)

    np.save(EXP / "early_exit_test_proba.npy", np.stack(proba_by_layer))
    out = dict(
        config=dict(model=MODEL_DIR, ckpt=CKPT, max_length=MAXLEN,
                    head_rounds=HEAD_EPOCHS, head_lr=HEAD_LR,
                    n_train=len(tr_t), n_val=len(va_t), n_test=len(te_t)),
        trunk_val_macro_f1=float(ck.get("val_macro_f1", 0.0)),
        trunk_test=dict(macro_f1=mf1(te_y, proba_by_layer[NL - 1].argmax(1)),
                        acc=float((proba_by_layer[NL - 1].argmax(1) == te_y).mean())),
        per_layer=per_layer, tier1=dict(macro_f1=tier1_mf1, latency_ms=c1,
                                        acc=float((pred1 == te_y).mean())),
        bands=bands, cascade=dict(pareto=pareto, budget=budget))
    with open(EXP / "early_exit.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log("wrote experiments/early_exit.json")


if __name__ == "__main__":
    main()
