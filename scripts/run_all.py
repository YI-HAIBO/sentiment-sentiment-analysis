"""
End-to-end orchestration.

Usage:
    python scripts/run_all.py                    # full pipeline
    python scripts/run_all.py --skip-bert        # baselines only
    python scripts/run_all.py --quick-bert       # BERT with tiny subset
    python scripts/run_all.py --skip-data        # reuse existing data
    python scripts/run_all.py --serve            # after training, launch the API

Every stage is idempotent: it reuses prior outputs when possible.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PY = sys.executable  # current interpreter
STAGES = []


def run(cmd: list[str], label: str, extra_env=None):
    print(f"\n{'='*60}\n[{label}] {' '.join(cmd)}\n{'='*60}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=extra_env)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"Stage [{label}] failed with code {proc.returncode}")
    print(f"[{label}] done in {dt:.1f}s")
    return dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-data", action="store_true")
    p.add_argument("--skip-bert", action="store_true")
    p.add_argument("--quick-bert", action="store_true", help="Use tiny BERT subset for smoke test")
    p.add_argument("--skip-extra-corpora", action="store_true",
                   help="Skip the Amazon and app-store replication evaluations")
    p.add_argument("--serve", action="store_true", help="Launch FastAPI service after pipeline")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    # 1. Data
    if not args.skip_data:
        run([PY, "data/preprocess.py"], "data/preprocess")

    # 2. Baselines
    run([PY, "models/baselines.py"], "baselines")

    # 3. BERT
    if not args.skip_bert:
        cmd = [PY, "models/bert.py"]
        if args.quick_bert:
            cmd.append("--quick")
        run(cmd, "bert-finetune")

    # 4. Evaluation & figures
    run([PY, "models/evaluate.py"], "evaluate")

    # 5. Cross-corpus replications (Amazon and app-store), each idempotent:
    #    a stage is skipped when its result JSON already exists.
    if not args.skip_extra_corpora:
        for name, out in [("amazon_eval", "experiments/amazon_eval.json"),
                          ("appstore_eval", "experiments/appstore_eval.json")]:
            if Path(out).exists():
                print(f"[{name}] cached at {out}; skipping", flush=True)
            else:
                run([PY, f"scripts/{name}.py"], name)

    # 6. Server
    if args.serve:
        print(f"\nLaunching FastAPI on {args.host}:{args.port} ...")
        cmd = [PY, "-m", "app.api", "--host", args.host, "--port", str(args.port)]
        run(cmd, "serve")  # blocks until Ctrl-C


if __name__ == "__main__":
    main()
