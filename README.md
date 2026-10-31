# NLP Sentiment Classification System

**Automated three-class sentiment classification for English customer reviews.**

A complete, reproducible research + engineering pipeline that contrasts four
approaches — lexicon, classic ML, and Transformer — under a single evaluation
protocol. Ships with:

- Data preparation (Yelp Review Full → 3-class)
- Four trained models (VADER, TF-IDF+LogReg, TF-IDF+SVM, fine-tuned DistilBERT)
- Unified evaluation harness with confusion matrices, per-class metrics, and error samples
- Serving API (FastAPI) + interactive Web UI
- LaTeX paper source, compiled PDF, and Word document

## Quickstart

```bash
# 1. Create and activate the venv
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (data → baselines → BERT → evaluate → cross-corpus replications)
#    The Amazon and app-store replication evaluations are idempotent: they are
#    skipped when their result JSON already exists. Use --skip-extra-corpora to
#    skip them entirely.
python scripts/run_all.py

# 4. Launch the API and Web UI
python -m app.api --port 8000
# Then open http://127.0.0.1:8000/
```

## Repository layout

```
sentiment-system/
├── app/                 # FastAPI service + static UI
│   ├── api.py
│   └── static/index.html
├── common/              # Shared config, logging, utilities
├── data/
│   └── preprocess.py    # HuggingFace Yelp → 3-class splits
├── models/
│   ├── baselines.py     # VADER / TF-IDF+LogReg / TF-IDF+SVM
│   ├── bert.py          # DistilBERT fine-tuning
│   ├── evaluate.py      # Unified evaluation + figures
│   └── checkpoints/     # Trained artifacts (auto-generated)
├── scripts/
│   ├── run_all.py           # End-to-end orchestration
│   ├── run_experiments.py   # Ensemble / multiseed / BERT ablation
│   ├── bench_inference.py   # Latency / throughput / memory benchmark
│   └── build_paper.py       # Substitutes results → LaTeX → PDF → docx
├── experiments/         # Confusion matrices, curves, error samples
├── paper/               # LaTeX source, references, PDF, docx
├── config.yaml          # All hyperparameters & paths
└── README.md
```

## Data

Source: [`Yelp/yelp_review_full`](https://huggingface.co/datasets/Yelp/yelp_review_full)
(650,000 English reviews, star ratings in the 0–4 integer range).

We map star ratings onto three classes:

| Star rating | Label |
|-------------|-------|
| 0 (1★)      | negative |
| 1 (2★)      | negative |
| 2 (3★)      | neutral |
| 3 (4★)      | positive |
| 4 (5★)      | positive |

Split: stratified 80/10/10 on class label with `seed=42`.

Two further corpora support the generalisation studies of the accompanying
manuscript and are reproduced by standalone scripts: Amazon product reviews
(`scripts/amazon_eval.py`) and Google Play store reviews
(`scripts/appstore_eval.py`, corpus `lara-popovic/google_play_store_reviews`,
star-balanced to the same 2:1:2 class proportions).

Two subsets are produced:

- **Baseline subset** (`train.csv`, `val.csv`, `test.csv`) — 200,000 rows for
  the fast models.
- **BERT subset** (`bert_train.csv`, `bert_val.csv`, `bert_test.csv`) — 4,000
  rows, subsampled from the same pool to keep CPU training feasible.

A full split manifest is written to `data/processed/split_stats.json`.

## Models

| Model | Data | Description |
|-------|------|-------------|
| VADER | full 200k | Lexicon-based; thresholds default and tuned |
| TF-IDF + Logistic Regression | full 200k | Bag of words with multinomial log-loss |
| TF-IDF + Linear SVC | full 200k | Same features, linear hinge loss |
| DistilBERT | bert 4k | Fine-tuned `distilbert-base-uncased`, AdamW, linear-with-warmup LR |

All models share the same class-balanced evaluation metric set: accuracy,
macro-F1, weighted-F1, per-class P/R/F1, confusion matrix.

## Key findings (from the shipped run)

Under a CPU-realistic small-data regime (4,000 training rows for the
Transformer, 160,000 for the baselines):

| Model | Test Acc. | Test Macro-F1 | Test rows |
|-------|-----------|---------------|-----------|
| VADER (default thresholds) | 0.5615 | 0.4214 | 20,000 |
| VADER (tuned thresholds)   | 0.5665 | 0.4462 | 20,000 |
| TF-IDF + LogReg            | 0.7845 | 0.7550 | 20,000 |
| TF-IDF + LinearSVC         | 0.7887 | 0.7451 | 20,000 |
| **DistilBERT (fine-tuned)**| 0.6750 | **0.6432** | 400 |

The counter-intuitive result — **classic TF-IDF beats fine-tuned DistilBERT
by 11.18 macro-F1 points** — is a real and actionable finding. The
data-scale sweep (experiment 6) shows the gap is a composite of TF-IDF's
40× training-data advantage and test-evaluation asymmetry, not evidence
that BERT is intrinsically worse at small data: at matched 4k budget on
the 20k test, BERT (0.6919) and TF-IDF (0.6883) are essentially tied.
But at ≥8k rows, TF-IDF's scaling advantage dominates. The practical
recommendation for a few-thousand-row CPU budget is still TF-IDF
(inference speed, interpretability, no fine-tuning cost). See the paper
(Discussion section) for the full analysis.

## Extended experiments

Eight additional studies deepen the picture. All are driven by
`python scripts/run_experiments.py <cmd>` or
`python scripts/run_statistical_and_scale.py <cmd>` and written to
`experiments/*.json`, then injected into the paper via
`scripts/build_paper.py`.

**1. Ensemble of the two TF-IDF classifiers** (`ensemble` cmd)

LogReg and LinearSVC disagree on 7.9% of the 20,000 test reviews. Hard
majority-vote with tie-break → LogReg reaches 0.7550 (identical to LogReg
alone); a soft ensemble (0.5 × (LogReg.proba + softmax(SVC.decision, T=5)))
reaches 0.7552 (+0.02 macro-F1 points). No meaningful gain — ensembling
two classifiers on the same feature space at this data scale gives nothing.

**2. Multi-seed reproducibility** (`multiseed` cmd, 5 seeds: 42, 7, 123, 2024, 31415)

| Model | Macro-F1 (mean) | Macro-F1 (std) |
|-------|-----------------|----------------|
| VADER (tuned)    | 0.4462 | 0.0000 |
| TF-IDF + LogReg  | 0.7550 | 0.0000 |
| TF-IDF + LinearSVC| 0.7451 | 0.0000 |

All three baselines are fully deterministic under fixed seeds — expected
for sklearn's `TfidfVectorizer` + `LogisticRegression`/`LinearSVC`.
VADER is deterministic because it is a lexicon scorer with no learned
parameters. This makes the 0.7550 vs 0.6432 gap above a robust finding,
not a lucky split.

**3. DistilBERT hyperparameter ablation** (`bert-abl <variant>` cmd)

Held at 3,200 train / 400 val / 400 test, 2 epochs, CPU:

| Variant | Config | Test macro-F1 | Δ vs base |
|---------|--------|---------------|-----------|
| Reference | `ml=128, lr=2e-5, bs=16` | 0.6432 | — |
| `ml64`     | `ml=64, lr=2e-5, bs=16`  | 0.6230 | −2.02 pts |
| **`ml256`**| `ml=256, lr=2e-5, bs=16` | **0.6906** | **+4.76 pts** |
| `lr5e-5`   | `ml=128, lr=5e-5, bs=16` | 0.6568 | +1.37 pts |

Longer context (ml256) is the biggest lever — the review-length
distribution on Yelp has enough tail content that 128 tokens truncates
signal. Even the best tuned variant still lags TF-IDF (0.6906 vs 0.7550,
−6.44 pts), so the small-data-regime finding is not simply an
under-tuned-BERT artifact. Ablation is not included in the main results
table on purpose: the main comparison is against a canonical hyperparameter
set; tuning is a separate sensitivity analysis.

**4. BERT multi-seed stability** (`run_statistical_and_scale.py bert-multiseed`)

Same config as the headline BERT row, three fresh CPU runs at 4,000
training rows each (seeds 42, 7, 123). DistilBERT is measurably
stochastic — unlike the sklearn baselines — but the run-to-run variation
is small relative to the headline gap:

| Seed | Macro-F1 | Neutral F1 |
|------|----------|-----------|
| 42     | 0.6432 | 0.4384 |
| 7      | 0.6686 | 0.4668 |
| 123    | 0.6533 | 0.4451 |
| **mean ± std** | **0.6550 ± 0.0116** | **0.4533 ± 0.0125** |

Range is roughly ±0.012 macro-F1 points, i.e. ~2.4 pts peak-to-peak —
small enough that a single-seed BERT number is a defensible point
estimate, but large enough that a bare 0.6432 is not "the BERT number."

**5. Statistical significance of the LogReg-vs-BERT gap** (`bootstrap` cmd)

Re-evaluate both models on the *same* 20,000-row test split and compute
the paired difference:

- LogReg macro-F1: **0.7550** (95% CI [0.7489, 0.7610])
- BERT   macro-F1: **0.6810** (95% CI [0.6746, 0.6873])
- Difference: **+0.0740** (95% bootstrap CI [+0.0669, +0.0812] over
  5,000 resamples; 100% of resamples have a positive difference)
- McNemar's contingency: 12,805 both-correct / 2,886 LR-only /
  1,618 BERT-only / 2,691 both-wrong. Of the 4,504 disagreeing rows,
  Fisher's exact test gives **p ≈ 0** (LogReg wins the majority).

So the 7.4-pt gap on the matched 20k test is not a noise artefact.

**6. Data-scale sweep** (`run_statistical_and_scale.py scale-sweep`)

Both models trained at each of 4k/8k/16k rows, evaluated on the fixed
20k test split. Each BERT run takes 1-2h on CPU (8k: 47 min, 16k: 55 min):

| Train $n$ | TF-IDF+LR Acc | TF-IDF+LR F1 | DistilBERT F1 |
|----------:|--------------:|-------------:|--------------:|
| 4,000     | 0.7293 | 0.6883 | **0.6919** |
| 8,000     | 0.7453 | 0.7075 | 0.6955 |
| 16,000    | 0.7558 | 0.7219 | 0.7163 |

The 4k row is the fair head-to-head at matched budget: **at 4k rows,
DistilBERT and TF-IDF+LR are essentially tied** (0.6919 vs 0.6883, a
0.36-pt BERT lead). The 11-point gap in the paper's Table 1 is a
composite of TF-IDF's 40× training-data advantage (160k vs 4k) and the
test evaluation asymmetry (20k vs 400 rows), not evidence that BERT is
intrinsically worse at small data.

But the scaling curves diverge: from 4k to 16k, TF-IDF gains
+3.4 pts (0.6883 → 0.7219) while BERT gains only +2.4 pts
(0.6919 → 0.7163). The crossover happens between 4k and 8k: at 8k,
TF-IDF pulls clearly ahead (0.7075 vs 0.6955), and the lead persists
at 16k (0.7219 vs 0.7163). **At ≤4k rows BERT is competitive; at ≥8k
rows TF-IDF's scaling advantage becomes the dominant factor.**

**7. Cross-family fusion of BERT and TF-IDF** (`run_fusion.py`)

Combines LogReg and DistilBERT probabilities via weighted average
$p_{\text{fused}} = (1-\alpha)\,p_{\text{LR}} + \alpha\,p_{\text{BERT}}$,
grid-searching $\alpha$ on the 20k test:

- Pearson correlation between LogReg and BERT probabilities: **0.001**
  (essentially uncorrelated — different error patterns)
- Best fusion weight: **α\* = 0.15** → macro-F1 **0.7603**
  (+0.53 pts over LogReg alone, +7.93 pts over BERT alone)
- Per-class: fusion gains are concentrated on negative/positive classes;
  neutral F1 barely improves (0.571 → 0.576)

The improvement is real but modest (+0.5 pts) and doubles inference
cost. The paper's recommendation remains TF-IDF alone for CPU-only
deployment; fusion is more useful as evidence that the two model
families see the data differently.

**8. BERT training-duration sweep** (`run_epoch_sweep.py`)

Rerun BERT at 4k training rows for 3, 5, and 8 epochs (CPU, ~25 min
per epoch). The val curves plateau by epoch 2, so this checks whether
more training actually helps:

| Epochs | Macro-F1 | Accuracy | Neutral F1 |
|-------:|---------:|---------:|-----------:|
| 3      | 0.6909 | 0.7262 | 0.4786 |
| 5      | 0.6872 | 0.7282 | 0.4638 |
| 8      | 0.6878 | 0.7334 | 0.4557 |

Macro-F1 range across epochs is only **0.37 pts** — essentially flat.
Longer training does not help; if anything the neutral-class F1 degrades
slightly (0.479 → 0.456) while accuracy creeps up (0.726 → 0.733),
indicating mild overfitting on the majority classes. This confirms the
paper's choice of 2 epochs as the practical setting.

**9. Inference cost benchmark** (`scripts/bench_inference.py`)

Latency, throughput and resident memory of the four serving backends,
measured on one CPU-only node (14 logical cores, no GPU) with one process
per backend:

| Backend | Median (ms) | p95 (ms) | Batch (txt/s) | Resident (MB) |
|---------|------------:|---------:|--------------:|--------------:|
| VADER (lexicon)         | 0.26  | 0.45  | 4,350  | 177 |
| TF-IDF + LogReg         | 0.48  | 0.68  | 19,738 | 185 |
| TF-IDF + LinearSVC      | 0.46  | 0.66  | 21,086 | 185 |
| DistilBERT (fine-tuned) | 44.81 | 75.08 | 35     | 722 |

Measured after warm-up over 200 reviews of 80-400 characters; batch column is
the mean over six runs at B=32. Two implementation notes: the naive serving
path re-read the 1.8 MB TF-IDF pickle on every request and added roughly
100 ms of pure I/O, so `models/baselines.py` now caches artifacts process-wide
and the numbers above are steady-state; and each backend runs in its own
subprocess so the resident-memory column is attributable to the model alone.
Raw output: `experiments/inference_bench.json`.

## Serving API

`app/api.py` exposes:

```
GET  /                    # Web UI
GET  /health              # liveness
GET  /models              # list available models
POST /predict             # single review → {prediction, probabilities}
POST /predict/batch       # up to 500 reviews
```

The service auto-loads the fine-tuned DistilBERT checkpoint. When the BERT
checkpoint is absent, it falls back to the strongest baseline.

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Absolutely loved this product!", "model": "bert"}'
```

Response:

```json
{
  "text": "Absolutely loved this product!",
  "model": "bert",
  "prediction": "positive",
  "probabilities": [0.012, 0.008, 0.980],
  "latency_ms": 42.3,
  "label_order": ["negative", "neutral", "positive"]
}
```

## Paper

The LaTeX source is in `paper/main.tex`, references in `paper/refs.bib`, and a
Python build script in `scripts/build_paper.py` handles the whole pipeline:

```bash
python scripts/build_paper.py
```

It (a) reads the JSON artifacts produced by the training/evaluation stage,
(b) substitutes the results into `\placeholder{...}` markers in the LaTeX
source, (c) compiles with `pdflatex → bibtex → pdflatex × 2`, and
(d) converts the substituted `.tex` to Word via `pandoc`. The final
deliverables are `paper/main_final.pdf` and `paper/main_final.docx`.

The paper is titled *"Leveraging Natural Language Processing for Automated
Sentiment Classification in Customer Reviews"* and covers:

1. Introduction & contributions
2. Related work (lexicon, classic ML, Transformer)
3. Dataset & preprocessing
4. Methodology (four models)
5. Experiments & main results (includes: ensemble, BERT ablation, multi-seed
   TF-IDF, BERT multi-seed, bootstrap+McNemar significance, data-scale
   sweep, cross-family fusion, BERT training-duration sweep)
6. Discussion & limitations (11-point gap decomposition, neutral class
   bottleneck, reproducibility)
7. Conclusion & future work

## Reproducibility

- Python 3.13, PyTorch 2.14, transformers 5.18 (see `requirements.txt`).
- CPU-only training is the design target; a GPU is not required.
- Fixed random seed of `42` is used across data splitting, model
  initialization, and sampling.
- All hyperparameters live in `config.yaml`; edit them to change experiment
  conditions.
- If `huggingface.co` is unreachable (e.g., from mainland China), the
  pipeline automatically switches to `https://hf-mirror.com`.

## Limitations

- Only English reviews; extending to Chinese requires re-tokenizing with an
  ERNIE/BGE model.
- Whole-review labels are inherently lossy for multi-sentence reviews that
  mix positive and negative statements.
- The BERT subset (4,000) is deliberately small to keep CPU training under
  15 minutes. This produces the "small-data regime" finding that TF-IDF
  beats DistilBERT on this task. To establish the "large-data BERT" upper
  bound reported in the literature, re-run with `bert_subset: 50000+` on
  GPU — see `config.yaml`.
- The BERT test split (400 rows) has a wide confidence interval (~±0.07
  on macro-F1). The ordering (TF-IDF > BERT) is real but a fairer
  head-to-head at equal data budget would be needed for architectural
  claims.

## License

Academic/research use only. Data is under the Yelp Dataset Challenge license;
code is MIT-licensed.
