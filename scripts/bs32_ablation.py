"""Revision experiment D2 (R1-m4): batch-size ablation row for Table 6.

The published ablation table covers max_length (64/256) and learning rate
(5e-5) but NOT batch size, although the text claims a three-knob ablation.
This script fine-tunes the reference DistilBERT configuration with
batch_size = 32 (reference = 16) and reports val/test metrics on the BERT
subset plus the 20,000-row test split, mirroring run_fusion-style evaluation.

Writes experiments/bs32_ablation.json + experiments/bs32_ablation_history.json
Usage   python scripts/bs32_ablation.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from common.config import CFG  # noqa: E402
from common.utils import seed_everything  # noqa: E402

DATA = ROOT / CFG["data"]["processed_dir"]
EXP = ROOT / "experiments"
LABELS = ["negative", "neutral", "positive"]
BATCH_SIZE = 32
EPOCHS = int(CFG["bert"]["epochs"])          # 2, reference configuration
LR = float(CFG["bert"]["learning_rate"])     # 2e-5
MAX_LEN = int(CFG["bert"]["max_length"])     # 128
SEED = int(CFG["bert"]["seed"])              # 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tok, max_length):
        self.texts, self.labels, self.tok, self.max_len = texts, labels, tok, max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding="max_length",
                       max_length=self.max_len, return_attention_mask=False)
        return {"input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "labels": torch.tensor(LABELS.index(self.labels[i]), dtype=torch.long)}


def evaluate(model, loader):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"]
            out = model(input_ids=ids, labels=batch["labels"])
            preds.extend(out.logits.argmax(-1).tolist())
            ys.extend(batch["labels"].tolist())
    y = np.array(ys)
    p = np.array(preds)
    return {"accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "n": int(len(y))}


def main() -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import WeightedRandomSampler

    torch.set_num_threads(14)
    seed_everything(SEED)
    t0 = time.time()

    train = pd.read_csv(DATA / "bert_train.csv")
    val = pd.read_csv(DATA / "bert_val.csv")
    bert_test = pd.read_csv(DATA / "bert_test.csv")
    test20k = pd.read_csv(DATA / "test.csv")
    log(f"train={len(train)} val={len(val)} bert_test={len(bert_test)} "
        f"test20k={len(test20k)}")

    model_name = CFG["bert"]["model_name"]
    model_path = ROOT / model_name
    tok = AutoTokenizer.from_pretrained(str(model_path) if model_path.exists() else model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path) if model_path.exists() else model_name, num_labels=3)

    train_ds = ReviewDataset(train["text"].tolist(), train["label"].tolist(), tok, MAX_LEN)
    val_ds = ReviewDataset(val["text"].tolist(), val["label"].tolist(), tok, MAX_LEN)
    btest_ds = ReviewDataset(bert_test["text"].tolist(), bert_test["label"].tolist(), tok, MAX_LEN)

    y_tr = np.array([LABELS.index(l) for l in train["label"]])
    counts = np.bincount(y_tr, minlength=3)
    w = 1.0 / np.maximum(counts, 1)
    sampler = WeightedRandomSampler(torch.tensor([w[i] for i in y_tr], dtype=torch.double),
                                     num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    btest_loader = DataLoader(btest_ds, batch_size=BATCH_SIZE, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=float(CFG["bert"]["weight_decay"]))
    total_steps = max(1, len(train_loader) * EPOCHS)
    warmup = int(total_steps * float(CFG["bert"]["warmup_ratio"]))
    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    s1 = LinearLR(opt, start_factor=1e-8, end_factor=1.0, total_iters=max(1, warmup))
    s2 = LinearLR(opt, start_factor=1.0, end_factor=0.0,
                  total_iters=max(1, total_steps - warmup))
    sched = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup])

    history = []
    for epoch in range(EPOCHS):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        te = time.time()
        for i, batch in enumerate(train_loader):
            out = model(input_ids=batch["input_ids"], labels=batch["labels"])
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            run_loss += out.loss.item()
            correct += int((out.logits.argmax(-1) == batch["labels"]).sum())
            total += batch["labels"].size(0)
            if i % 20 == 0:
                log(f"  epoch {epoch+1} step {i}/{len(train_loader)} "
                    f"loss={out.loss.item():.4f}")
        vm = evaluate(model, val_loader)
        history.append({"epoch": epoch + 1, "train_loss": run_loss / len(train_loader),
                        "train_acc": correct / max(1, total),
                        "val_acc": vm["accuracy"], "val_macro_f1": vm["macro_f1"]})
        log(f"epoch {epoch+1}: train_acc={correct/max(1,total):.4f} "
            f"val_macro_f1={vm['macro_f1']:.4f} ({time.time()-te:.0f}s)")

    bert_400 = evaluate(model, btest_loader)

    # --- 20k test split evaluation (matches the published 20k protocol) ---
    log("evaluating tuned model on the 20,000-row test split")
    preds, probs = [], []
    model.eval()
    t1 = time.time()
    for i in range(0, len(test20k), 128):
        chunk = test20k["text"].tolist()[i:i + 128]
        with torch.no_grad():
            enc = tok(chunk, padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt")
            out = model(**enc)
            probs.append(torch.nn.functional.softmax(out.logits, -1).numpy())
            preds.extend(out.logits.argmax(-1).tolist())
    probs = np.concatenate(probs)
    y20 = np.array([LABELS.index(str(l)) for l in test20k["label"]])
    p20 = np.array(preds)
    res20k = {"accuracy": float(accuracy_score(y20, p20)),
              "macro_f1": float(f1_score(y20, p20, average="macro", zero_division=0)),
              "n": int(len(y20))}
    log(f"20k macro_f1={res20k['macro_f1']:.4f} ({time.time()-t1:.0f}s)")

    np.save(EXP / "bs32_test_proba.npy", probs.astype(np.float32))

    out = {
        "variant": "batch_size_32",
        "config": {"batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR,
                   "max_length": MAX_LEN, "seed": SEED,
                   "reference_batch_size": int(CFG["bert"]["batch_size"])},
        "n_train": int(len(train)), "n_val": int(len(val)),
        "bert_test_400": bert_400,
        "test_20k": res20k,
        "history": history,
        "elapsed_s": round(time.time() - t0, 1),
        "note": ("Fills the missing batch-size row of the published ablation "
                 "table (Reviewer 1 minor comment 4)."),
    }
    with open(EXP / "bs32_ablation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"DONE bs=32  400-row test macro_f1={bert_400['macro_f1']:.4f}  "
        f"20k macro_f1={res20k['macro_f1']:.4f}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
