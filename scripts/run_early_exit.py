"""
Depth-adaptive second tier (DeeBERT-style per-layer exits) for the cascade.

为什么做这个（方案 b 第 1 项）：级联被支配的根因是后端成本 c_2 = 44.81 ms，
而它换来的修复只有 0.5 个 macro-F1 点。论文里已经把 c_2 的可行区间用
一条不等式刻死了，因此唯一没被榨干的杠杆就是 c_2 本身。做法是把昂贵的
第二层改成多层早退分支：每条被路由上来的评论只跑到某个中间层就出分类头，
深度（也就成本）随预算自适应。

步骤
  0. 直接载入已发表的调优 checkpoint（best.pt：完整主干 + 顶层分类头），
     它本身就是"跑满 6 层"的那一档。
  1. 冻结主干，在 6 个 transformer 层各接一个 3 类分类头并各自训练。
  2. 在 20k 测试集上一次性取全部层的 hidden state，评估每个出口的质量；
     再逐出口测单请求延迟。
  3. 用这些便宜出口重做论文里的预算分析（eq:cond / tab:cond 的同一套账）。

输出  experiments/early_exit.json
用法   python scripts/run_early_exit.py
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


def run_heads(model, heads, ids, mask, layer_idx, batch=64):
    """Predict with the head attached to transformer layer `layer_idx`."""
    n = len(ids)
    ids = torch.as_tensor(ids)
    mask = torch.as_tensor(mask).float()
    out = np.zeros((n, 3))
    layers = list(model.distilbert.transformer.layer)
    with torch.no_grad():
        for s in range(0, n, batch):
            b = slice(s, min(s + batch, n))
            h = model.distilbert.embeddings(ids[b])
            for i in range(layer_idx):
                h = layers[i](h, mask[b])[0]
            out[b] = torch.softmax(heads[layer_idx](h[:, 0]), -1).numpy()
    return out


def hidden_all(model, ids, mask, batch=64):
    """Hidden state after every layer, in one forward pass."""
    n = len(ids)
    ids = torch.as_tensor(ids)
    mask = torch.as_tensor(mask).float()
    out = [np.zeros((n, 768)) for _ in range(NL)]
    layers = list(model.distilbert.transformer.layer)
    with torch.no_grad():
        for s in range(0, n, batch):
            b = slice(s, min(s + batch, n))
            h = model.distilbert.embeddings(ids[b])
            states = [h[:, 0]]
            cur = h
            for layer in layers:
                cur = layer(cur, mask[b])[0]
                states.append(cur[:, 0])
            for li in range(NL):
                out[li][b] = states[li].numpy()
    return out


def bench_one(ids, mask, model, head, n=40, upto=None):
    """Single-request latency of an exit running `upto` layers."""
    times = []
    layers = list(model.distilbert.transformer.layer)
    with torch.no_grad():
        for i in range(n):
            ii = torch.as_tensor(ids[i:i + 1])
            mm = torch.as_tensor(mask[i:i + 1]).float()
            t0 = time.perf_counter()
            h = model.distilbert.embeddings(ii)
            if upto is None:
                for layer in layers:
                    h = layer(h, mm)[0]
            else:
                for i2 in range(upto):
                    h = layers[i2](h, mm)[0]
            head(h[:, 0])
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), float(np.percentile(times, 95))


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

    # ---------- trunk: the already published tuned checkpoint ----------
    ck = torch.load(ROOT / CKPT, map_location="cpu")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    sd = ck["model_state_dict"]
    # the local checkpoint was trained for three classes; the on-disk config
    # still declares two, so drop the (tiny) head before loading the trunk
    model.load_state_dict({k: v for k, v in sd.items()
                           if not k.startswith("classifier")},
                          strict=False)
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

    # ---------- stage 2: one head per layer, trunk frozen ----------
    heads = nn.ModuleList([nn.Sequential(nn.Dropout(0.1), nn.Linear(768, 3))
                           for _ in range(NL)]).to(DEV)
    heads[NL - 1][1].weight.data.copy_(model.classifier.weight.data)
    heads[NL - 1][1].bias.data.copy_(model.classifier.bias.data)
    for p in model.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(heads.parameters(), lr=HEAD_LR, weight_decay=0.0)
    steps = len(tr_y) * HEAD_EPOCHS // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=HEAD_LR,
                                                total_steps=steps, pct_start=0.1)
    for ep in range(HEAD_EPOCHS):
        heads.train()
        t0 = time.time()
        for s in range(0, len(tr_y), BS):
            b = slice(s, min(s + BS, len(tr_y)))
            ii = torch.from_numpy(tr_ids[b])
            mm = torch.from_numpy(tr_m[b]).float()
            yy = torch.from_numpy(tr_y[b])
            opt.zero_grad()
            h = model.distilbert.embeddings(ii)
            for layer in list(model.distilbert.transformer.layer):
                h = layer(h, mm)[0]
            loss = sum(nn.functional.cross_entropy(heads[li](h[:, 0]), yy)
                       for li in range(NL)) / NL
            loss.backward()
            opt.step()
            sched.step()
        heads.eval()
        with torch.no_grad():
            hv = hidden_all(model, va_ids, va_m)
            vals = [mf1(va_y, heads[li](hv[li]).argmax(-1).numpy())
                    for li in range(NL)]
        log("head epoch %d  val mF1 %s (%.0fs)"
            % (ep + 1, [round(v, 4) for v in vals], time.time() - t0))

    # ---------- quality of every exit on the 20k test set ----------
    t0 = time.time()
    per_layer = []
    H = hidden_all(model, te_ids, te_m)
    for li in range(NL):
        proba = torch.softmax(heads[li](torch.from_numpy(H[li])), -1).numpy()
        pred = proba.argmax(1)
        med, p95 = bench_one(te_ids, te_m, model, heads[li], n=40, upto=li + 1)
        per_layer.append(dict(
            layer=li + 1, latency_ms=med, latency_p95_ms=p95,
            macro_f1=mf1(te_y, pred),
            acc=float((pred == te_y).mean()),
            neutral_f1=float(f1_score(te_y, pred, average=None, labels=[1])[0])))
        log("exit layer %d: %.2f ms  mF1 %.4f  acc %.4f  neutralF1 %.4f"
            % (li + 1, med, per_layer[-1]["macro_f1"], per_layer[-1]["acc"],
               per_layer[-1]["neutral_f1"]))
    log("20k multi-exit eval %.0fs" % (time.time() - t0))

    # ---------- first tier (TF-IDF + LogReg) signals ----------
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
    proba_by_layer = []
    for li in range(NL):
        proba_by_layer.append(
            torch.softmax(heads[li](torch.from_numpy(H[li])), -1).numpy())
    res, pareto = [], []
    for li in range(NL):
        c2 = per_layer[li]["latency_ms"]
        for tau in np.round(np.arange(0.05, 1.0, 0.025), 3):
            esc = s1 < tau
            r = float(esc.mean())
            if r <= 0:
                continue
            p = pred1.copy()
            p[esc] = proba_by_layer[li][esc].argmax(1)
            row = dict(layer=li + 1, tau=float(tau), r=r, c1=c1, c2=c2,
                       latency=float(c1 + r * c2),
                       acc=float((p == te_y).mean()),
                       macro_f1=mf1(te_y, p))
            res.append(row)
            pareto.append(row)
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

    # per-band repair behaviour of the cheapest useful exit
    bands = []
    edges = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    li_best = min(range(NL), key=lambda i: per_layer[i]["latency_ms"])
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s1 >= lo) & (s1 < hi)
        if m.sum() == 0:
            continue
        band = dict(band="[%.2f,%.2f)" % (lo, hi), n=int(m.sum()),
                    tier1_mf1=mf1(te_y[m], pred1[m]),
                    full_mf1=mf1(te_y[m], proba_by_layer[NL - 1][m].argmax(1)),
                    exit_mf1=mf1(te_y[m], proba_by_layer[li_best][m].argmax(1)))
        band["exit_minus_tier1"] = band["exit_mf1"] - band["tier1_mf1"]
        bands.append(band)

    np.save(EXP / "early_exit_test_proba.npy", torch.stack(
        [torch.from_numpy(p) for p in proba_by_layer]).numpy())
    out = dict(
        config=dict(model=MODEL_DIR, ckpt=CKPT, max_length=MAXLEN,
                    head_epochs=HEAD_EPOCHS, head_lr=HEAD_LR,
                    n_train=len(tr_t), n_val=len(va_t), n_test=len(te_t)),
        trunk_val_macro_f1=float(ck.get("val_macro_f1", 0.0)),
        trunk_test=dict(macro_f1=mf1(te_y, proba_by_layer[NL - 1].argmax(1)),
                        acc=float((proba_by_layer[NL - 1].argmax(1) == te_y).mean())),
        per_layer=per_layer, tier1=dict(macro_f1=tier1_mf1, latency_ms=c1,
                                        acc=float((pred1 == te_y).mean())),
        bands=bands, cascade=dict(pareto=pareto, budget=budget),
    )
    with open(EXP / "early_exit.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log("wrote experiments/early_exit.json")


if __name__ == "__main__":
    main()
