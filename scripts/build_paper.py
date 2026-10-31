"""
Build the paper: substitute real numbers from experiment artifacts into
the LaTeX template, compile PDF, and convert to docx via pandoc.

Usage:
    python scripts/build_paper.py

Requires:
    - pdflatex (MiKTeX) on PATH
    - pandoc on PATH
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
EXP = ROOT / "experiments"


def load_json(p: Path, default=None):
    if not p.exists():
        return default if default is not None else {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_placeholders() -> dict:
    """Build a {placeholder_name: value} map from experiment artifacts."""
    base = load_json(ROOT / "models" / "checkpoints" / "baselines_summary.json", {})
    bert = load_json(ROOT / "models" / "checkpoints" / "bert_summary.json", {})
    splits = load_json(ROOT / "data" / "processed" / "split_stats.json", {})
    hp = {}

    # Class distribution
    if splits:
        bl = splits.get("baseline", {})
        bt = splits.get("bert", {})
        def _cls(d, k):
            return d.get("test", {}).get("class_counts", {}).get(k, 0)
        hp["NEG_FULL"] = f"{splits.get('star_counts', {}).get('0', 0) + splits.get('star_counts', {}).get('1', 0):,}"
        hp["NEU_FULL"] = f"{splits.get('star_counts', {}).get('2', 0):,}"
        hp["POS_FULL"] = f"{splits.get('star_counts', {}).get('3', 0) + splits.get('star_counts', {}).get('4', 0):,}"
        hp["NEG_B"] = _cls(bl, "negative")
        hp["NEU_B"] = _cls(bl, "neutral")
        hp["POS_B"] = _cls(bl, "positive")
        hp["NEG_BERT"] = _cls(bt, "negative")
        hp["NEU_BERT"] = _cls(bt, "neutral")
        hp["POS_BERT"] = _cls(bt, "positive")
        total = sum(_cls(bl, k) for k in ["negative", "neutral", "positive"])
        if total:
            hp["Y5_PCT"] = f"{100.0 * _cls(bl, 'positive') / total:.1f}"
            hp["Y1_PCT"] = f"{100.0 * _cls(bl, 'negative') / total:.1f}"
        else:
            hp["Y5_PCT"] = "--"
            hp["Y1_PCT"] = "--"

    # Baseline results (test split)
    def _get(models, key, metric="macro_f1"):
        return models.get(key, {}).get("test", {}).get(metric, 0.0)
    if base:
        m = base.get("models", {})
        v = m.get("vader", {}).get("default", {})
        vt = m.get("vader", {}).get("tuned", {})
        hp["V_ACC"] = f"{v.get('test', {}).get('accuracy', 0):.4f}"
        hp["V_MF1"] = f"{v.get('test', {}).get('macro_f1', 0):.4f}"
        hp["V_WF1"] = f"{v.get('test', {}).get('weighted_f1', 0):.4f}"
        hp["VADER_F1"] = hp["V_MF1"]
        hp["VT_ACC"] = f"{vt.get('test', {}).get('accuracy', 0):.4f}"
        hp["VT_MF1"] = f"{vt.get('test', {}).get('macro_f1', 0):.4f}"
        hp["VT_WF1"] = f"{vt.get('test', {}).get('weighted_f1', 0):.4f}"
        hp["LR_ACC"] = f"{m.get('tfidf_logreg', {}).get('test', {}).get('accuracy', 0):.4f}"
        hp["LR_MF1"] = f"{m.get('tfidf_logreg', {}).get('test', {}).get('macro_f1', 0):.4f}"
        hp["LR_WF1"] = f"{m.get('tfidf_logreg', {}).get('test', {}).get('weighted_f1', 0):.4f}"
        hp["LR_F1"] = hp["LR_MF1"]
        hp["SVM_ACC"] = f"{m.get('tfidf_svc', {}).get('test', {}).get('accuracy', 0):.4f}"
        hp["SVM_MF1"] = f"{m.get('tfidf_svc', {}).get('test', {}).get('macro_f1', 0):.4f}"
        hp["SVM_WF1"] = f"{m.get('tfidf_svc', {}).get('test', {}).get('weighted_f1', 0):.4f}"
        hp["SVM_F1"] = hp["SVM_MF1"]

        # Per-class from BERT
        bert_t = bert.get("test", {}) if bert else {}
        if bert_t:
            pc = bert_t.get("per_class", {})
            for lab, key in [("NEG", "negative"), ("NEU", "neutral"), ("POS", "positive")]:
                hp[f"BERT_{key[0].upper()}_P".replace("_", "_", 1)] = ""
                d = pc.get(key, {})
                hp[f"BERT_{lab.upper()}_P" if False else f"BERT_{key}_P".replace("_", "")] = ""
            # Simpler direct assignment:
            hp["BERT_NEG_P"] = f"{pc.get('negative', {}).get('precision', 0):.4f}"
            hp["BERT_NEG_R"] = f"{pc.get('negative', {}).get('recall', 0):.4f}"
            hp["BERT_NEG_F1"] = f"{pc.get('negative', {}).get('f1', 0):.4f}"
            hp["BERT_NEU_P"] = f"{pc.get('neutral', {}).get('precision', 0):.4f}"
            hp["BERT_NEU_R"] = f"{pc.get('neutral', {}).get('recall', 0):.4f}"
            hp["BERT_NEU_F1"] = f"{pc.get('neutral', {}).get('f1', 0):.4f}"
            hp["BERT_POS_P"] = f"{pc.get('positive', {}).get('precision', 0):.4f}"
            hp["BERT_POS_R"] = f"{pc.get('positive', {}).get('recall', 0):.4f}"
            hp["BERT_POS_F1"] = f"{pc.get('positive', {}).get('f1', 0):.4f}"

    # BERT summary
    if bert:
        hp["BERT_ACC"] = f"{bert.get('test', {}).get('accuracy', 0):.4f}"
        hp["BERT_MF1"] = f"{bert.get('test', {}).get('macro_f1', 0):.4f}"
        hp["BERT_WF1"] = f"{bert.get('test', {}).get('weighted_f1', 0):.4f}"
        hp["BERT_F1"] = hp["BERT_MF1"]

    # Relative improvement of BERT over best non-deep
    def _f(x):
        try:
            return float(str(x).split()[0])
        except Exception:
            return 0.0
    best_baseline = max(
        _f(hp.get("VT_MF1", 0)), _f(hp.get("LR_MF1", 0)), _f(hp.get("SVM_MF1", 0))
    )
    bert_f1 = _f(hp.get("BERT_MF1", 0))
    gap_signed = 100 * (bert_f1 - best_baseline)
    hp["BERT_UP"] = f"{gap_signed:+.2f}"
    # Absolute gap in macro-F1 percentage points, always non-negative.
    # Used by the paper to describe the small-data regime finding
    # where BERT is worse than TF-IDF.
    hp["BERT_GAP_ABS"] = f"{abs(gap_signed):.2f}"

    # ---------- Ensemble placeholders ----------
    ens = load_json(ROOT / "experiments" / "ensemble.json", {})
    if ens:
        hp["ENS_DISAGREE"] = f"{100 * ens.get('disagreement_rate', 0):.1f}"
        hp["ENS_LR"] = f"{ens.get('logreg_solo', {}).get('macro_f1', 0):.4f}"
        hp["ENS_SVC"] = f"{ens.get('svc_solo', {}).get('macro_f1', 0):.4f}"
        hp["ENS_HARD"] = f"{ens.get('hard_vote', {}).get('macro_f1', 0):.4f}"
        hp["ENS_SOFT"] = f"{ens.get('soft_ensemble', {}).get('macro_f1', 0):.4f}"
    else:
        for k in ["ENS_DISAGREE", "ENS_LR", "ENS_SVC", "ENS_HARD", "ENS_SOFT"]:
            hp.setdefault(k, "--")

    # ---------- BERT ablation placeholders ----------
    abl = load_json(ROOT / "experiments" / "bert_ablation.json", {})
    variants = abl.get("variants", {}) if abl else {}
    base_f1 = _f(hp.get("BERT_MF1", 0))
    for vname, key_prefix in [
        ("ml64", "ABL_64"),
        ("ml256", "ABL_256"),
        ("lr5e-5", "ABL_LR5"),
    ]:
        v = variants.get(vname, {})
        if v:
            f1 = _f(str(v.get("test", {}).get("macro_f1", 0)))
            cfg = v.get("config", {})
            hp[key_prefix] = f"{f1:.4f}"
            hp[key_prefix + "_D"] = f"{100 * (f1 - base_f1):+.2f}"
            hp[key_prefix + "_CFG"] = (
                f"ml={cfg.get('max_length')}, lr={cfg.get('learning_rate')}, "
                f"bs={cfg.get('batch_size')}"
            )
        else:
            hp.setdefault(key_prefix, "--")
            hp.setdefault(key_prefix + "_D", "--")
            hp.setdefault(key_prefix + "_CFG", "--")
    hp["ABL_BASE"] = f"{base_f1:.4f}"
    hp["ABL_BASE_CFG"] = "ml=128, lr=2e-5, bs=16"
    # Range and best-variant gap
    abl_f1s = [_f(str(v.get("test", {}).get("macro_f1", 0))) for v in variants.values() if v]
    if abl_f1s:
        hp["ABL_RANGE"] = f"{max(abl_f1s) - min(abl_f1s):.4f}"
        best_abl = max(abl_f1s)
        hp["ABL_BEST_GAP"] = f"{abs(best_abl - best_baseline):.2f}"
    else:
        hp["ABL_RANGE"] = "--"
        hp["ABL_BEST_GAP"] = "--"

    # ---------- Multi-seed placeholders ----------
    ms = load_json(ROOT / "experiments" / "multiseed.json", {})
    models_ms = ms.get("models", {}) if ms else {}
    for model_key, prefix in [
        ("vader", "SEED_VADER"),
        ("tfidf_logreg", "SEED_LR"),
        ("tfidf_svc", "SEED_SVC"),
    ]:
        m = models_ms.get(model_key, {})
        s = m.get("summary", {}) if m else {}
        if s:
            hp[f"{prefix}_MF1"] = f"{s.get('macro_f1_mean', 0):.4f}"
            hp[f"{prefix}_STD"] = f"{s.get('macro_f1_std', 0):.4f}"
        else:
            hp.setdefault(f"{prefix}_MF1", "--")
            hp.setdefault(f"{prefix}_STD", "--")
    stds = [float(str(hp.get(f"{p}_STD", "0")).split()[0])
            for p in ["SEED_VADER", "SEED_LR", "SEED_SVC"]
            if str(hp.get(f"{p}_STD", "--")) != "--"]
    hp["SEED_MAX_STD"] = f"{max(stds) if stds else 0:.4f}"

    # ---------- BERT multi-seed (4k training, 3 seeds) ----------
    bms = load_json(ROOT / "experiments" / "bert_multiseed.json", {})
    bms_summary = bms.get("summary", {}) if bms else {}
    if bms_summary:
        hp["BERT_SEED_MF1"] = f"{bms_summary.get('macro_f1_mean', 0):.4f}"
        hp["BERT_SEED_STD"] = f"{bms_summary.get('macro_f1_std', 0):.4f}"
        hp["BERT_SEED_MIN"] = f"{bms_summary.get('macro_f1_min', 0):.4f}"
        hp["BERT_SEED_MAX"] = f"{bms_summary.get('macro_f1_max', 0):.4f}"
        hp["BERT_SEED_NEU"] = f"{bms_summary.get('neutral_f1_mean', 0):.4f}"
    else:
        for k in ["BERT_SEED_MF1", "BERT_SEED_STD", "BERT_SEED_MIN",
                  "BERT_SEED_MAX", "BERT_SEED_NEU"]:
            hp.setdefault(k, "--")

    # ---------- Bootstrap CI + McNemar (statistical significance) ----------
    boot = load_json(ROOT / "experiments" / "bootstrap.json", {})
    if boot:
        b = boot.get("bootstrap", {})
        m = boot.get("mcnemar", {})
        hp["BOOT_N"] = f"{boot.get('n_test', 0):,}"
        hp["BOOT_NBOOT"] = f"{boot.get('n_boot', 0):,}"
        hp["BOOT_LR_MF1"] = f"{boot.get('lr_macro_f1', 0):.4f}"
        hp["BOOT_BERT_MF1"] = f"{boot.get('bert_macro_f1', 0):.4f}"
        hp["BOOT_DIFF"] = f"{100 * boot.get('observed_diff', 0):+.2f}"
        hp["BOOT_CI_LOW"] = f"{100 * b.get('ci_95_low', 0):+.2f}"
        hp["BOOT_CI_HIGH"] = f"{100 * b.get('ci_95_high', 0):+.2f}"
        hp["BOOT_PCT_POS"] = f"{100 * b.get('pct_diff_positive', 0):.1f}"
        hp["BOOT_DIFF_STD"] = f"{100 * b.get('diff_std', 0):.2f}"
        hp["BOOT_LR_CI"] = (
            f"[{boot.get('bootstrap_lr', {}).get('ci_95', [0, 0])[0]:.4f}, "
            f"{boot.get('bootstrap_lr', {}).get('ci_95', [0, 0])[1]:.4f}]"
        )
        hp["BOOT_BERT_CI"] = (
            f"[{boot.get('bootstrap_bert', {}).get('ci_95', [0, 0])[0]:.4f}, "
            f"{boot.get('bootstrap_bert', {}).get('ci_95', [0, 0])[1]:.4f}]"
        )
        hp["BOOT_MN_BOTH"] = f"{m.get('both_correct', 0):,}"
        hp["BOOT_MN_LR_ONLY"] = f"{m.get('only_lr_correct', 0):,}"
        hp["BOOT_MN_BERT_ONLY"] = f"{m.get('only_bert_correct', 0):,}"
        hp["BOOT_MN_BOTH_WRONG"] = f"{m.get('both_wrong', 0):,}"
        hp["BOOT_MN_DISAGREE"] = f"{m.get('n_disagreeing', 0):,}"
        hp["BOOT_P"] = f"{m.get('fisher_exact_p', 0):.3g}"
        hp["BOOT_SIG"] = "yes" if m.get("significant_at_005") else "no"
        # Per-class macro-F1 comparison
        for lab in ["negative", "neutral", "positive"]:
            key = lab.upper()[:3]
            pc = boot.get("per_class", {}).get(lab, {})
            hp[f"BOOT_PC_{key}_LR"] = f"{pc.get('lr_f1', 0):.4f}"
            hp[f"BOOT_PC_{key}_BERT"] = f"{pc.get('bert_f1', 0):.4f}"
    else:
        for k in ["BOOT_N", "BOOT_NBOOT", "BOOT_LR_MF1", "BOOT_BERT_MF1",
                  "BOOT_DIFF", "BOOT_CI_LOW", "BOOT_CI_HIGH", "BOOT_PCT_POS",
                  "BOOT_DIFF_STD", "BOOT_LR_CI", "BOOT_BERT_CI",
                  "BOOT_MN_BOTH", "BOOT_MN_LR_ONLY", "BOOT_MN_BERT_ONLY",
                  "BOOT_MN_BOTH_WRONG", "BOOT_MN_DISAGREE", "BOOT_P", "BOOT_SIG",
                  "BOOT_PC_NEG_LR", "BOOT_PC_NEG_BERT",
                  "BOOT_PC_NEU_LR", "BOOT_PC_NEU_BERT",
                  "BOOT_PC_POS_LR", "BOOT_PC_POS_BERT"]:
            hp.setdefault(k, "--")

    # ---------- Data-scale sweep ----------
    ss = load_json(ROOT / "experiments" / "scale_sweep.json", {})
    ss_models = ss.get("models", {}) if ss else {}
    # We always render 4k, 8k, 16k rows.
    for model_key, prefix in [("tfidf_logreg", "SCALE_LR"), ("bert", "SCALE_BERT")]:
        entries = ss_models.get(model_key, [])
        by_n = {int(e.get("n_train", -1)): e for e in entries}
        for n in [4000, 8000, 16000]:
            e = by_n.get(n, {})
            if e:
                hp[f"{prefix}_{n//1000}K_MF1"] = f"{e.get('macro_f1', 0):.4f}"
                hp[f"{prefix}_{n//1000}K_ACC"] = f"{e.get('accuracy', 0):.4f}"
            else:
                hp.setdefault(f"{prefix}_{n//1000}K_MF1", "--")
                hp.setdefault(f"{prefix}_{n//1000}K_ACC", "--")
    # Crossover analysis: where do BERT and LogReg cross?
    lr_series = sorted((int(e["n_train"]), float(e.get("macro_f1", 0)))
                       for e in ss_models.get("tfidf_logreg", []))
    bert_series = sorted((int(e["n_train"]), float(e.get("macro_f1", 0)))
                         for e in ss_models.get("bert", []))
    lr_f1_max = max([v for _, v in lr_series]) if lr_series else 0.0
    bert_f1_max = max([v for _, v in bert_series]) if bert_series else 0.0
    hp["SCALE_LR_MAX_MF1"] = f"{lr_f1_max:.4f}"
    hp["SCALE_BERT_MAX_MF1"] = f"{bert_f1_max:.4f}"
    if lr_series and bert_series:
        # Common-scale head-to-head at the largest scale where both ran.
        lr_by_n = dict(lr_series)
        bert_by_n = dict(bert_series)
        common_n = sorted(set(lr_by_n) & set(bert_by_n))
        if common_n:
            at_max = max(common_n)
            gap_pts = 100 * (lr_by_n[at_max] - bert_by_n[at_max])
            hp["SCALE_GAP_PTS"] = f"{gap_pts:+.2f}"
            hp["SCALE_MAX_COMMON_N"] = f"{at_max:,}"
            hp["SCALE_LR_AT_MAX"] = f"{lr_by_n[at_max]:.4f}"
            hp["SCALE_BERT_AT_MAX"] = f"{bert_by_n[at_max]:.4f}"
            # Also the absolute LR-vs-BERT gap in points (signed):
            # LR - BERT at the common max.
            hp["SCALE_LR_MINUS_BERT_PTS"] = f"{gap_pts:+.2f}"
        else:
            for k in ["SCALE_GAP_PTS", "SCALE_MAX_COMMON_N",
                      "SCALE_LR_AT_MAX", "SCALE_BERT_AT_MAX"]:
                hp.setdefault(k, "--")
    else:
        for k in ["SCALE_GAP_PTS", "SCALE_MAX_COMMON_N",
                  "SCALE_LR_AT_MAX", "SCALE_BERT_AT_MAX"]:
            hp.setdefault(k, "--")
    # TF-IDF's own gain from the smallest to the largest scale we measured.
    # (This is the number we cite in the paper when talking about TF-IDF's
    # monotonic scaling — the SCALE_GAP_PTS placeholder was overloaded.)
    if len(lr_series) >= 2:
        (n_min, f_min), (n_max, f_max) = lr_series[0], lr_series[-1]
        hp["SCALE_LR_GAIN_PTS"] = f"{100 * (f_max - f_min):+.2f}"
        hp["SCALE_LR_MIN_MF1"] = f"{f_min:.4f}"
        hp["SCALE_LR_MAX_N"] = f"{n_max:,}"
    else:
        for k in ["SCALE_LR_GAIN_PTS", "SCALE_LR_MIN_MF1", "SCALE_LR_MAX_N"]:
            hp.setdefault(k, "--")
    # BERT's own gain from smallest to largest scale (mirrors SCALE_LR_GAIN_PTS).
    if len(bert_series) >= 2:
        (bn_min, bf_min), (bn_max, bf_max) = bert_series[0], bert_series[-1]
        hp["SCALE_BERT_GAIN_PTS"] = f"{100 * (bf_max - bf_min):+.2f}"
        hp["SCALE_BERT_MIN_MF1"] = f"{bf_min:.4f}"
        hp["SCALE_BERT_MAX_N"] = f"{bn_max:,}"
    else:
        for k in ["SCALE_BERT_GAIN_PTS", "SCALE_BERT_MIN_MF1", "SCALE_BERT_MAX_N"]:
            hp.setdefault(k, "--")

    # ---------- Cross-family fusion ----------
    try:
        fus = json.load(open(os.path.join(EXP, 'fusion.json')))
        hb = fus['fusion_best']
        hp['FUS_ALPHA'] = f"{hb['alpha']:.2f}"
        hp['FUS_BEST_MF1'] = f"{hb['macro_f1']:.4f}"
        hp['FUS_BEST_ACC'] = f"{hb['accuracy']:.4f}"
        hp['FUS_DELTA_LR'] = f"{100 * hb['delta_vs_logreg']:+.2f}"
        hp['FUS_LR_MF1'] = f"{fus['logreg']['macro_f1']:.4f}"
        hp['FUS_LR_ACC'] = f"{fus['logreg']['accuracy']:.4f}"
        hp['FUS_BERT_MF1'] = f"{fus['bert']['macro_f1']:.4f}"
        hp['FUS_BERT_ACC'] = f"{fus['bert']['accuracy']:.4f}"
        hp['FUS_CORR'] = f"{fus['lr_vs_bert_pearson']:+.3f}"
        for cls, key in [('NEG', 'negative'), ('NEU', 'neutral'), ('POS', 'positive')]:
            hp[f'FUS_PC_{cls}_LR'] = f"{fus['logreg']['per_class'][key]['f1']:.4f}"
            hp[f'FUS_PC_{cls}_BERT'] = f"{fus['bert']['per_class'][key]['f1']:.4f}"
            hp[f'FUS_PC_{cls}_FUS'] = f"{hb['per_class'][key]['f1']:.4f}"
    except Exception as e:
        for k in ['FUS_ALPHA', 'FUS_BEST_MF1', 'FUS_BEST_ACC', 'FUS_DELTA_LR',
                  'FUS_LR_MF1', 'FUS_LR_ACC', 'FUS_BERT_MF1', 'FUS_BERT_ACC', 'FUS_CORR']:
            hp.setdefault(k, '--')

    # ---------- Epoch sweep ----------
    try:
        ep = json.load(open(os.path.join(EXP, 'epoch_sweep.json')))
        eps = ep['epochs']
        for e in eps:
            tag = f"EPOCH_{e['epochs']}"
            hp[f'{tag}_MF1'] = f"{e['macro_f1']:.4f}"
            hp[f'{tag}_ACC'] = f"{e['accuracy']:.4f}"
        f1s = [e['macro_f1'] for e in eps]
        hp['EPOCH_RANGE_PTS'] = f"{100 * (max(f1s) - min(f1s)):.2f}"
        if max(f1s) - min(f1s) < 0.01:
            hp['EPOCH_HAS_PLATEAU'] = 'no meaningful gain from more epochs'
        else:
            hp['EPOCH_HAS_PLATEAU'] = 'some variance but within noise'
    except Exception as e:
        for k in ['EPOCH_3_MF1', 'EPOCH_3_ACC', 'EPOCH_5_MF1', 'EPOCH_5_ACC',
                  'EPOCH_8_MF1', 'EPOCH_8_ACC', 'EPOCH_RANGE_PTS', 'EPOCH_HAS_PLATEAU']:
            hp.setdefault(k, '--')

        # Legacy ablation placeholders (kept for backward compat)
    for key in ["ABL_128", "ABL_128_D", "ABL_512", "ABL_512_D",
                "ABL_BS8", "ABL_BS8_D"]:
        hp.setdefault(key, "--")

    # ---------- Cross-family fusion (BERT + TF-IDF) ----------
    fus = load_json(ROOT / "experiments" / "fusion.json", {})
    if fus:
        b = fus.get("fusion_best", {})
        hp["FUS_N"] = f"{fus.get('n_test', 0):,}"
        hp["FUS_LR_MF1"] = f"{fus.get('logreg', {}).get('macro_f1', 0):.4f}"
        hp["FUS_BERT_MF1"] = f"{fus.get('bert', {}).get('macro_f1', 0):.4f}"
        hp["FUS_BEST_MF1"] = f"{b.get('macro_f1', 0):.4f}"
        hp["FUS_ALPHA"] = f"{b.get('alpha', 0):.2f}"
        hp["FUS_DELTA_LR"] = f"{100 * b.get('delta_vs_logreg', 0):+.2f}"
        hp["FUS_DELTA_BERT"] = f"{100 * b.get('delta_vs_bert', 0):+.2f}"
        hp["FUS_CORR"] = f"{fus.get('lr_vs_bert_pearson', 0):+.3f}"
        hp["FUS_LR_ACC"] = f"{fus.get('logreg', {}).get('accuracy', 0):.4f}"
        hp["FUS_BERT_ACC"] = f"{fus.get('bert', {}).get('accuracy', 0):.4f}"
        hp["FUS_BEST_ACC"] = f"{b.get('accuracy', 0):.4f}"
        # Per-class comparison at the best alpha
        for lab in ["negative", "neutral", "positive"]:
            key = lab.upper()[:3]
            lr_pc = fus.get("logreg", {}).get("per_class", {}).get(lab, {})
            bt_pc = fus.get("bert", {}).get("per_class", {}).get(lab, {})
            fu_pc = b.get("per_class", {}).get(lab, {})
            hp[f"FUS_PC_{key}_LR"] = f"{lr_pc.get('f1', 0):.4f}"
            hp[f"FUS_PC_{key}_BERT"] = f"{bt_pc.get('f1', 0):.4f}"
            hp[f"FUS_PC_{key}_FUS"] = f"{fu_pc.get('f1', 0):.4f}"
    else:
        for k in ["FUS_N", "FUS_LR_MF1", "FUS_BERT_MF1", "FUS_BEST_MF1",
                  "FUS_ALPHA", "FUS_DELTA_LR", "FUS_DELTA_BERT", "FUS_CORR",
                  "FUS_LR_ACC", "FUS_BERT_ACC", "FUS_BEST_ACC",
                  "FUS_PC_NEG_LR", "FUS_PC_NEG_BERT", "FUS_PC_NEG_FUS",
                  "FUS_PC_NEU_LR", "FUS_PC_NEU_BERT", "FUS_PC_NEU_FUS",
                  "FUS_PC_POS_LR", "FUS_PC_POS_BERT", "FUS_PC_POS_FUS"]:
            hp.setdefault(k, "--")

    # ---------- BERT training-duration sweep (3/5/8 epochs) ----------
    es = load_json(ROOT / "experiments" / "epoch_sweep.json", {})
    epochs_list = es.get("epochs", []) if es else []
    if epochs_list:
        by_ep = {int(e.get("epochs", -1)): e for e in epochs_list}
        for ep in [3, 5, 8]:
            e = by_ep.get(ep, {})
            if e:
                hp[f"EPOCH_{ep}_MF1"] = f"{e.get('macro_f1', 0):.4f}"
                hp[f"EPOCH_{ep}_ACC"] = f"{e.get('accuracy', 0):.4f}"
                hp[f"EPOCH_{ep}_NEU"] = f"{e.get('neutral_f1', 0):.4f}"
                hp[f"EPOCH_{ep}_BEST"] = f"{e.get('best_epoch', '--')}"
            else:
                for k in [f"EPOCH_{ep}_MF1", f"EPOCH_{ep}_ACC",
                          f"EPOCH_{ep}_NEU", f"EPOCH_{ep}_BEST"]:
                    hp.setdefault(k, "--")
        # Overall range
        f1s = [e.get("macro_f1", 0) for e in epochs_list if e.get("macro_f1", 0) > 0]
        if f1s:
            hp["EPOCH_MIN_MF1"] = f"{min(f1s):.4f}"
            hp["EPOCH_MAX_MF1"] = f"{max(f1s):.4f}"
            hp["EPOCH_RANGE_PTS"] = f"{100*(max(f1s)-min(f1s)):+.2f}"
            hp["EPOCH_HAS_PLATEAU"] = "yes" if (max(f1s) - min(f1s)) < 0.005 else "no"
    else:
        for k in ["EPOCH_3_MF1", "EPOCH_3_ACC", "EPOCH_3_NEU", "EPOCH_3_BEST",
                  "EPOCH_5_MF1", "EPOCH_5_ACC", "EPOCH_5_NEU", "EPOCH_5_BEST",
                  "EPOCH_8_MF1", "EPOCH_8_ACC", "EPOCH_8_NEU", "EPOCH_8_BEST",
                  "EPOCH_MIN_MF1", "EPOCH_MAX_MF1", "EPOCH_RANGE_PTS",
                  "EPOCH_HAS_PLATEAU"]:
            hp.setdefault(k, "--")

    return hp


def substitute_template(template: Path, hp: dict, out: Path) -> None:
    src = template.read_text(encoding="utf-8")
    for k, v in hp.items():
        src = re.sub(r"\\placeholder\{" + re.escape(k) + r"\}", str(v), src)
    out.write_text(src, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


def compile_pdf(tex_file: Path) -> Path:
    """Compile .tex to .pdf with pdflatex + bibtex + pdflatex x2."""
    paper_dir = tex_file.parent
    stem = tex_file.stem
    def run(cmd, timeout=180):
        r = subprocess.run(cmd, cwd=str(paper_dir),
                           capture_output=True, text=True, timeout=timeout)
        return r
    print("Compiling PDF (pdflatex → bibtex → pdflatex × 2) ...")
    # 1) First pdflatex pass — produces .aux with bib references
    r = run(["pdflatex", "-interaction=nonstopmode", f"-jobname={stem}",
             f"-output-directory={paper_dir}", str(tex_file)])
    if r.returncode != 0 and not (paper_dir / f"{stem}.aux").exists():
        print("pdflatex first pass failed:")
        print((r.stdout or "")[-2000:])
        print((r.stderr or "")[-2000:])
        raise SystemExit("pdflatex failed")
    # 2) bibtex — resolves citations
    if (paper_dir / f"{stem}.aux").exists():
        bibtex = shutil.which("bibtex") or shutil.which("bibtex.exe")
        if bibtex:
            r = run([bibtex, stem])
            if r.returncode != 0:
                print(f"bibtex warning (non-fatal): {(r.stdout or '')[-500:]}")
        else:
            print("bibtex not found on PATH; citations will be undefined.")
    # 3) Two more pdflatex passes to resolve cross-references
    for i in (1, 2):
        r = run(["pdflatex", "-interaction=nonstopmode", f"-jobname={stem}",
                 f"-output-directory={paper_dir}", str(tex_file)])
        if r.returncode != 0:
            print(f"pdflatex pass {i+1} exit {r.returncode} (may be recoverable warnings)")
    pdf = tex_file.with_suffix(".pdf")
    if not pdf.exists():
        print("pdflatex failed to produce PDF. Last stdout:")
        print((r.stdout or "")[-3000:])
        print((r.stderr or "")[-1500:])
        raise SystemExit("PDF missing")
    print(f"  → {pdf.relative_to(ROOT)}")
    return pdf


def find_pandoc():
    p = shutil.which("pandoc") or shutil.which("pandoc.exe")
    if p:
        return p
    # Common Windows install locations
    for cand in [
        r"C:\Program Files\Pandoc\pandoc.exe",
        r"C:\Program Files (x86)\Pandoc\pandoc.exe",
    ]:
        if Path(cand).exists():
            return cand
    return None


def convert_tex_to_docx(tex: Path, docx_out: Path) -> None:
    """Convert the substituted .tex source to .docx via pandoc."""
    pandoc = find_pandoc()
    if not pandoc:
        print("pandoc not found; skipping DOCX conversion.")
        return
    try:
        r = subprocess.run(
            [pandoc, str(tex), "-o", str(docx_out)],
            cwd=str(tex.parent),
            capture_output=True, text=True, timeout=120,
        )
        if docx_out.exists():
            print(f"wrote {docx_out.relative_to(ROOT)}")
        else:
            print(f"pandoc did not produce docx. stderr (last 500):")
            print((r.stderr or "")[-500:])
    except Exception as e:
        print(f"tex→DOCX conversion failed: {e}")


def make_curves_figure(hist_path: Path, out_path: Path) -> None:
    """Plot training/validation macro-F1 across epochs."""
    import matplotlib.pyplot as plt
    hist = load_json(hist_path, [])
    if not hist:
        print(f"No history at {hist_path}; skipping curves.png")
        return
    epochs = [h.get("epoch", i) for i, h in enumerate(hist)]
    val_f1 = [h.get("val_macro_f1", 0) for h in hist]
    tr_acc = [h.get("train_acc", 0) for h in hist]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, val_f1, marker="o", label="Val Macro-F1", color="#f4a261")
    ax1.plot(epochs, tr_acc, marker="s", linestyle="--", label="Train Accuracy", color="#2a9d8f")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Score")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax1.set_title("DistilBERT Training Curves")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(ROOT)}")


def main():
    hp = collect_placeholders()
    print(f"Collected {len(hp)} placeholders")

    # 1. Substitute into LaTeX
    tex_src = PAPER / "main.tex"
    tex_out = PAPER / "main_final.tex"
    substitute_template(tex_src, hp, tex_out)

    # 2. Generate curves figure from BERT history
    make_curves_figure(
        ROOT / "experiments" / "bert_history.json",
        PAPER / "figures" / "curves.png",
    )
    # Copy model comparison figure if it exists
    cmp_src = ROOT / "experiments" / "figures" / "model_comparison.png"
    cmp_dst = PAPER / "figures" / "model_comparison.png"
    if cmp_src.exists():
        cmp_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cmp_src, cmp_dst)
        print(f"copied {cmp_src.name}")

    # 3. Compile PDF
    try:
        compile_pdf(tex_out)
    except Exception as e:
        print(f"PDF compile failed: {e}")

    # 4. Convert tex source to docx via pandoc
    docx = PAPER / "main_final.docx"
    convert_tex_to_docx(tex_out, docx)


if __name__ == "__main__":
    main()
