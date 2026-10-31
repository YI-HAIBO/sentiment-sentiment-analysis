"""Latency / throughput / memory benchmark for the inference serving path.

Measures, for each deployed model (VADER, TF-IDF+LogReg, TF-IDF+LinearSVC,
fine-tuned DistilBERT):

  * artifact size on disk (MB)
  * resident memory of a process that serves *only* this model (MB)
  * per-request latency (median / p95 / mean, ms)
  * single-request throughput (req/s)
  * batch throughput at B=32 (texts/s)

Each model is benchmarked in its own child process so that the reported
resident memory is attributable to the model alone rather than to a shared
interpreter that already holds the other backends. A `__baseline__` child
records the cost of an empty interpreter, and the model footprint is
reported as the difference.

Run (parent, spawns children):
    python scripts/bench_inference.py
Run (single child, used by the parent):
    python scripts/bench_inference.py --model distilbert
Outputs:
    experiments/inference_bench.json
"""
from __future__ import annotations

import ctypes
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import argparse
import random
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MB = 1024 * 1024
BATCH = 32
WARMUP = 8
N_SINGLE = 200

if os.name == "nt":

    class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # GetCurrentProcess() returns the pseudo-handle -1; it must be read back
    # as a pointer-sized handle or 64-bit builds truncate it to 0 and every
    # subsequent handle-based query fails with ERROR_INVALID_HANDLE.
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel32.GetCurrentProcess.argtypes = []
    _psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                            ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
                                            wintypes.DWORD]

    def rss_bytes() -> int:
        h = _kernel32.GetCurrentProcess()
        pmc = _PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(pmc)
        if not _psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        # h is the pseudo-handle returned by GetCurrentProcess(); it must not
        # be closed, and c_void_p cannot be passed back to CloseHandle as int.
        return int(pmc.WorkingSetSize)

else:  # pragma: no cover - POSIX fallback

    def rss_bytes() -> int:
        with open("/proc/self/statm", "rb") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def rss_mb() -> float:
    return rss_bytes() / MB


def dir_size_mb(path: Path) -> float:
    total = 0
    for dp, _dn, fn in os.walk(path):
        for f in fn:
            try:
                total += (Path(dp) / f).stat().st_size
            except OSError:
                pass
    return total / MB


MAX_LEN = 256
CKPT = ROOT / "models" / "checkpoints"
MODEL_ORDER = ["vader", "tfidf_logreg", "tfidf_svc", "distilbert"]


def build_model(name: str):
    """Return fn(texts) -> list[str], matching the app/api.py serving path."""
    from models.baselines import predict_with_artifact

    if name == "vader":
        def fn(texts):
            p, _ = predict_with_artifact("vader", texts, CKPT)
            return list(p)
    elif name == "tfidf_logreg":
        def fn(texts):
            p, _ = predict_with_artifact("tfidf_logreg", texts, CKPT)
            return list(p)
    elif name == "tfidf_svc":
        def fn(texts):
            p, _ = predict_with_artifact("tfidf_svc", texts, CKPT)
            return list(p)
    elif name == "distilbert":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        d = CKPT / "bert_final"
        tok = AutoTokenizer.from_pretrained(str(d))
        mdl = AutoModelForSequenceClassification.from_pretrained(str(d))
        mdl.to("cpu")
        mdl.eval()

        def fn(texts):
            with torch.no_grad():
                enc = tok(texts, truncation=True, padding=True, max_length=MAX_LEN,
                          return_tensors="pt")
                out = mdl(**enc)
                return [["negative", "neutral", "positive"][i]
                        for i in out.logits.argmax(-1).tolist()]
    else:
        raise ValueError(name)
    return fn


def get_pool():
    import pandas as pd
    from common.config import CFG
    df = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    col = "text" if "text" in df.columns else df.columns[1]
    pool = [t for t in df[col].astype(str).tolist() if 80 <= len(t) <= 400]
    random.Random(42).shuffle(pool)
    return pool


def run_baseline() -> dict:
    """Empty-interpreter cost, measured in the same subprocess machinery."""
    gc.collect()
    time.sleep(0.3)
    return {"model": "__baseline__", "first_call_ms": 0.0,
            "latency_median_ms": 0.0, "latency_p95_ms": 0.0, "latency_mean_ms": 0.0,
            "throughput_rps": 0.0, "batch_size": BATCH, "batch_mean_ms": 0.0,
            "batch_throughput_tps": 0.0, "rss_after_load_mb": round(rss_mb(), 1),
            "rss_steady_mb": round(rss_mb(), 1), "artifact_mb": 0.0,
            "artifact_kind": "n/a", "n_samples": 0}


def run_single(name: str) -> dict:
    if name == "__baseline__":
        return run_baseline()
    fn = build_model(name)
    pool = get_pool()
    texts_single = pool[:N_SINGLE + WARMUP]
    texts_batch = pool[:BATCH]

    # cold first request (artifact page-in)
    t0 = time.perf_counter()
    fn(texts_single[:1])
    first_call_ms = (time.perf_counter() - t0) * 1000

    rss_after_load = rss_mb()

    for t in texts_single[:WARMUP]:
        fn([t])
    gc.collect()
    rss_steady = rss_mb()

    lags = []
    for t in texts_single[:N_SINGLE]:
        t0 = time.perf_counter()
        fn([t])
        lags.append((time.perf_counter() - t0) * 1000)

    s = sorted(lags)
    median_ms = statistics.median(lags)
    p95_ms = s[min(N_SINGLE - 1, int(0.95 * N_SINGLE))]
    mean_ms = statistics.fmean(lags)

    batch_lags = []
    for _ in range(6):
        t0 = time.perf_counter()
        fn(texts_batch)
        batch_lags.append((time.perf_counter() - t0) * 1000)
    batch_mean_ms = statistics.fmean(batch_lags)

    sizes = {
        "vader": CKPT.joinpath("vader_thresholds.json").stat().st_size / MB,
        "tfidf_logreg": CKPT.joinpath("tfidf_logreg.pkl").stat().st_size / MB,
        "tfidf_svc": CKPT.joinpath("tfidf_svc.pkl").stat().st_size / MB,
        "distilbert": dir_size_mb(CKPT / "bert_final"),
    }
    kinds = {"vader": "JSON thresholds",
             "tfidf_logreg": "pickle (vectorizer + classifier)",
             "tfidf_svc": "pickle (vectorizer + classifier)",
             "distilbert": "PyTorch weights"}

    return {
        "model": name,
        "first_call_ms": round(first_call_ms, 2),
        "latency_median_ms": round(median_ms, 3),
        "latency_p95_ms": round(p95_ms, 3),
        "latency_mean_ms": round(mean_ms, 3),
        "throughput_rps": round(1000.0 / median_ms, 1),
        "batch_size": BATCH,
        "batch_mean_ms": round(batch_mean_ms, 2),
        "batch_throughput_tps": round(BATCH * 1000.0 / batch_mean_ms, 1),
        "rss_after_load_mb": round(rss_after_load, 1),
        "rss_steady_mb": round(rss_steady, 1),
        "artifact_mb": round(sizes[name], 2),
        "artifact_kind": kinds[name],
        "n_samples": N_SINGLE,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="benchmark one model; omit to run all children")
    args = ap.parse_args()

    if args.model:
        row = run_single(args.model)
        tag = args.model.replace("__", "")
        child_out = ROOT / "experiments" / f"bench_{tag}.json"
        child_out.write_text(json.dumps(row), encoding="utf-8")
        return

    import torch
    print(f" CPU: {platform.processor()}")
    print(f" torch: {torch.__version__} | threads: {torch.get_num_threads()}")
    print(f" protocol: {N_SINGLE} single requests, batch B={BATCH}, "
          f"input window 80-400 chars")
    print("-" * 78)

    py = sys.executable
    script = str(Path(__file__).resolve())
    baseline = None
    rows = []
    for name in MODEL_ORDER + ["__baseline__"]:
        out = ROOT / "experiments" / ("bench_%s.json" % name.replace("__", ""))
        env = dict(os.environ)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        r = subprocess.run([py, script, "--model", name], capture_output=True,
                           text=True, env=env, cwd=str(ROOT))
        if r.returncode != 0:
            print(f" {name:14s} FAILED\n{r.stderr[-500:]}")
            continue
        row = json.loads(out.read_text(encoding="utf-8"))
        out.unlink(missing_ok=True)
        if name == "__baseline__":
            baseline = row["rss_steady_mb"]
            continue
        rows.append(row)
        for r_ in rows:
            pass
        print(f" {name:14s} med={row['latency_median_ms']:8.3f} ms  "
              f"p95={row['latency_p95_ms']:8.3f} ms  "
              f"rss={row['rss_steady_mb']:8.1f} MB  "
              f"batch={row['batch_throughput_tps']:9.1f} txt/s")

    for r in rows:
        r["rss_footprint_mb"] = round(max(0.0, r["rss_steady_mb"] - baseline), 1)
        r["speedup_vs_bert"] = None

    bert = next((r for r in rows if r["model"] == "distilbert"), None)
    if bert:
        for r in rows:
            r["speedup_vs_bert"] = round(bert["latency_median_ms"] / r["latency_median_ms"], 1)

    payload = {
        "environment": {
            "cpu": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "os": platform.platform(),
            "device": "CPU only (no GPU)",
        },
        "protocol": {
            "n_single_requests": N_SINGLE,
            "warmup": WARMUP,
            "batch_size": BATCH,
            "max_length_tokens": MAX_LEN,
            "input_window_chars": "80-400",
            "measurement": "wall-clock, one process per model, inference only "
                           "(no I/O, no HTTP overhead)",
            "memory_definition": "resident set of a process serving only this "
                                 "model, minus an empty-interpreter baseline",
            "empty_interpreter_rss_mb": baseline,
        },
        "results": rows,
    }
    out = ROOT / "experiments" / "inference_bench.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("-" * 78)
    print(f" baseline (empty interpreter) RSS: {baseline} MB")
    print(f" written: {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
