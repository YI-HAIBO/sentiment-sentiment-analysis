# Sentiment-Analysis Data — IET Software 8513297

Data for: *Leveraging Natural Language Processing for Automated Sentiment
Classification in Customer Reviews* (IET Software, submission 8513297).

## Contents
- `experiments/*.json` — evaluation results (metrics, bootstrap CIs, routing
  tables, per-class scores) backing every table in the manuscript.
- `experiments/*.npy` — model output probabilities on the 20,000-row test
  split (and the early-exit per-layer probabilities), one file per model.
- `data/processed/*.npy` — held-out Amazon / app-store BERT test predictions.
- `data/processed/split_stats.json` — split sizes and class distributions.
- `data/processed/{bert,amazon_bert,appstore_bert,matched}_*.csv` — the small
  fine-tuning / matched-budget subsets (regenerable, seed = 42).

## Reproducibility
- Three corpora: Yelp (main), Amazon, Google Play app-store.
- Three classes (1-2 / 3 / 4-5 stars -> negative / neutral / positive).
- Models: VADER, TF-IDF+LogReg / LinearSVM, DistilBERT (matched-budget).
- Seed = 42; see the manuscript's Data Availability statement for the
  experiment pipeline.

## Notes
- The large raw split corpora (`train/val/test.csv`, `amazon_*.csv`,
  `appstore_*.csv` full splits) are regenerable with `preprocess.py` /
  `amazon_eval.py` / `appstore_eval.py` and are intentionally not stored here
  (they exceed per-file size limits).
- Model weights (`.pt`, `.safetensors`) are excluded.
