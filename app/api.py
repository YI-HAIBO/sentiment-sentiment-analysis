"""
FastAPI service for sentiment classification.

Endpoints:
    GET  /                - static UI
    GET  /health          - liveness probe
    POST /predict         - single text
    POST /predict/batch   - up to 500 texts
    GET  /models          - list available models + metadata

Run:
    python -m app.api                    # uses config.yaml
    python -m app.api --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import CFG  # noqa: E402
from common.utils import load_json, setup_logger  # noqa: E402
from models.baselines import predict_with_artifact  # noqa: E402

LABEL_ORDER = ["negative", "neutral", "positive"]

# ---------- Model manager ----------
class ModelManager:
    def __init__(self):
        self.log = setup_logger("serve")
        self.device = "cpu"
        self.bert_dir = ROOT / CFG["serving"]["model_path"]
        self.fallback_dir = ROOT / "models" / "checkpoints"
        self._bert_model = None
        self._bert_tok = None
        self._meta = {}

    def load_bert(self):
        if not self.bert_dir.exists():
            self.log.warning(f"BERT checkpoint not found at {self.bert_dir}; using TF-IDF fallback")
            return
        self.log.info(f"Loading BERT from {self.bert_dir}")
        self._bert_tok = AutoTokenizer.from_pretrained(str(self.bert_dir))
        self._bert_model = AutoModelForSequenceClassification.from_pretrained(str(self.bert_dir))
        self._bert_model.to(self.device)
        self._bert_model.eval()

    def predict(self, texts: List[str], model: str = "bert"):
        t0 = time.time()
        if model == "bert":
            if self._bert_model is None:
                raise HTTPException(status_code=503, detail="BERT model not loaded; use fallback_model=tfidf_logreg")
            preds, probs, labels = [], [], []
            with torch.no_grad():
                enc = self._bert_tok(texts, truncation=True, padding=True,
                                     max_length=CFG["bert"]["max_length"], return_tensors="pt")
                out = self._bert_model(**enc)
                probs = torch.softmax(out.logits, dim=-1).cpu().tolist()
                preds = [LABEL_ORDER[i] for i in out.logits.argmax(-1).tolist()]
        elif model in ("tfidf_logreg", "tfidf_svc"):
            preds, _ = predict_with_artifact(model, texts, self.fallback_dir)
            preds = list(preds)
            probs = [None] * len(preds)
        elif model == "vader":
            preds, extra = predict_with_artifact("vader", texts, self.fallback_dir)
            preds = list(preds)
            probs = []
            for ex in extra:
                probs.append({"compound": ex["compound"], "positive": ex["pos"],
                              "negative": ex["neg"], "neutral": ex["neu"]})
        else:
            raise HTTPException(status_code=400, detail=f"unknown model={model}")
        return preds, probs, time.time() - t0


MM = ModelManager()


# ---------- FastAPI ----------
app = FastAPI(
    title="NLP Sentiment Classification Service",
    version="1.0.0",
    description="Three-class (positive/neutral/negative) customer review sentiment classification. "
                "Powered by DistilBERT fine-tuning with TF-IDF + VADER fallbacks.",
)


class PredictRequest(BaseModel):
    text: str = Field(..., description="Single review text")
    model: str = Field("bert", description="One of: bert, tfidf_logreg, tfidf_svc, vader")


class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=500)
    model: str = Field("bert", description="One of: bert, tfidf_logreg, tfidf_svc, vader")


@app.on_event("startup")
def _startup():
    try:
        MM.load_bert()
    except Exception as e:
        MM.log.error(f"BERT load failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "bert_loaded": MM._bert_model is not None}


@app.get("/models")
def list_models():
    return {
        "label_order": LABEL_ORDER,
        "models": {
            "bert": {"loaded": MM._bert_model is not None, "path": str(MM.bert_dir),
                     "note": "DistilBERT fine-tuned on Yelp review subset"},
            "tfidf_logreg": {"path": str(MM.fallback_dir / "tfidf_logreg.pkl"),
                             "note": "TF-IDF + Logistic Regression"},
            "tfidf_svc": {"path": str(MM.fallback_dir / "tfidf_svc.pkl"),
                          "note": "TF-IDF + Linear SVC"},
            "vader": {"path": str(MM.fallback_dir / "vader_thresholds.json"),
                      "note": "Lexicon-based (VADER) with tuned thresholds"},
        },
    }


@app.post("/predict")
def predict(req: PredictRequest):
    preds, probs, dt = MM.predict([req.text], req.model)
    return {
        "text": req.text,
        "model": req.model,
        "prediction": preds[0],
        "probabilities": probs[0],
        "latency_ms": round(dt * 1000, 2),
        "label_order": LABEL_ORDER,
    }


@app.post("/predict/batch")
def predict_batch(req: PredictBatchRequest):
    preds, probs, dt = MM.predict(req.texts, req.model)
    return {
        "model": req.model,
        "count": len(preds),
        "latency_ms": round(dt * 1000, 2),
        "avg_ms": round(dt / max(1, len(preds)) * 1000, 2),
        "results": [
            {"text": t, "prediction": p, "probabilities": pr}
            for t, p, pr in zip(req.texts, preds, probs)
        ],
        "label_order": LABEL_ORDER,
    }


@app.get("/")
def index():
    static_dir = ROOT / "app" / "static"
    if not (static_dir / "index.html").exists():
        raise HTTPException(500, "static/index.html missing")
    return FileResponse(static_dir / "index.html")


@app.on_event("startup")
def _mount_static():
    static_dir = ROOT / "app" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    # Mount only after startup so index route above works as the "/" handler
    try:
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=CFG["serving"]["host"])
    p.add_argument("--port", type=int, default=CFG["serving"]["port"])
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    import uvicorn
    uvicorn.run("app.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
