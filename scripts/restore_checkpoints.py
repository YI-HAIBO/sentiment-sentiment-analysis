#!/usr/bin/env python3
"""Reassemble the published DistilBERT checkpoint from its gzip shards.

The checkpoint is split at a fixed byte budget so that every upload stays
below the host per-file size cap. Because the cut is fixed, no framing header
is needed: concatenating the decompressed chunks reproduces the original file
byte for byte. checkpoints/manifest.json records the SHA256 of the original
and of each compressed chunk, so a corrupted or partial download is detected
before any output is written, and the reassembled file is checked against the
original SHA256 after writing.

    python scripts/restore_checkpoints.py

The file is written to models/checkpoints/best.pt, the path app/api.py,
models/bert.py and models/evaluate.py already read. It is excluded from the
journal submission bundle, so restoring it does not affect what is shipped.

Experiments read from experiments/bert_tuned/best.pt can get that copy with:

    python -c "import shutil; shutil.copyfile('models/checkpoints/best.pt',
            'experiments/bert_tuned/best.pt')"
"""
import gzip
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MANIFEST = os.path.join(ROOT, "checkpoints", "manifest.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fail(out, msg):
    """Report msg and leave no bad artifact behind."""
    try:
        os.remove(out)
    except OSError:
        print("  (could not delete %s; remove it manually)" % out)
    sys.exit(msg)


def main():
    if not os.path.exists(MANIFEST):
        sys.exit("checkpoints/manifest.json not found; use a full clone")
    with open(MANIFEST, "r", encoding="utf-8") as f:
        man = json.load(f)

    # 1. verify every chunk before writing anything
    for c in man["chunks"]:
        p = os.path.join(ROOT, "checkpoints", c["file"])
        if not os.path.exists(p):
            sys.exit("missing shard: %s (incomplete download)" % c["file"])
        got = sha256(p)
        if got != c["sha256_gzip"]:
            sys.exit("sha256 mismatch for %s: got %s want %s"
                     % (c["file"], got, c["sha256_gzip"]))

    # 2. reassemble
    out = os.path.join(ROOT, man["restore_target"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        sys.exit("refusing to overwrite %s; move it aside first" % out)
    with open(out, "wb") as w:
        for c in man["chunks"]:
            with gzip.open(os.path.join(ROOT, "checkpoints", c["file"]),
                           "rb") as g:
                buf = g.read()
            if len(buf) != c["bytes"]:
                fail(out, "decompressed size mismatch for %s: got %d want %d"
                     % (c["file"], len(buf), c["bytes"]))
            w.write(buf)

    # 3. verify the result, and leave no artifact on failure
    got = sha256(out)
    if got != man["sha256_original"]:
        fail(out, "reconstructed file does not match the original sha256 "
                  "(got %s)" % got)

    print("restored %s  (%d bytes, %d chunks)"
          % (man["restore_target"], man["bytes"], len(man["chunks"])))
    print("sha256 %s  OK" % got)
    print("load: ckpt = torch.load(%r, map_location='cpu')"
          % (os.path.join(ROOT, man["restore_target"]),))
    print("       sd   = ckpt['model_state_dict']")


if __name__ == "__main__":
    main()
