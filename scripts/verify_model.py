#!/usr/bin/env python3
"""Check a downloaded model folder against the published manifest (standard library only).

Manifest (receipts/model-manifest-4bit-mtp.tsv): tab-separated columns
  repo, revision, path, size_bytes, folder_name, sha256 (empty for small non-LFS files).
For every row: the file must exist with exactly that size; when a SHA-256 is given it must match ("ok"),
otherwise the size alone is checked ("oksize"). Pass means: bad == 0 and ok + oksize == rows
(45 + 9 = 54 for TensorFold/GLM-5.3-Flash-MLX-4bit-MTP @ d2b17f52), plus explicit SHA-256 pins for
config.json, model.safetensors.index.json and tokenizer.json, plus the expected config fields
(text_config.num_hidden_layers 45, text_config.num_nextn_predict_layers 1, quantization 4-bit affine, group 64).

Hashing ~182 GB takes a while; --size-only skips the large-file hashes (not a full check).

Example:
  python3 scripts/verify_model.py --model-dir <models-dir>/TensorFold__GLM-5.3-Flash-MLX-4bit-MTP
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(HERE, "..", "receipts", "model-manifest-4bit-mtp.tsv")
PINS = {
    "config.json": "f80682bcc8ba5fc6ecdc5ce553700b02befe999bc4a6ba6fd47f644df4864adf",
    "model.safetensors.index.json": "6c030d5b58f4c264f00becf9dddcb8f8b1f69b3bbaaf35c93592b201b673208f",
    "tokenizer.json": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
}
CONFIG_EXPECT = {"num_hidden_layers": 45, "num_nextn_predict_layers": 1,
                 "quantization": {"bits": 4, "group_size": 64, "mode": "affine"}}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(8 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--size-only", action="store_true", help="skip SHA-256 of large files (quick, not a full check)")
    a = p.parse_args(argv)
    if not os.path.isdir(a.model_dir):
        print(f"error: not a folder: {a.model_dir}", file=sys.stderr)
        return 2
    try:
        with open(a.manifest, encoding="utf-8") as f:
            rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
    except OSError as exc:
        print(f"error: cannot read manifest: {exc}", file=sys.stderr)
        return 2
    ok = oksize = bad = 0
    for i, r in enumerate(rows, 1):
        if len(r) < 6:
            print(f"error: manifest line {i} has {len(r)} columns, expected 6", file=sys.stderr)
            return 2
        rel, size, want = r[2], int(r[3]), r[5].strip()
        path = os.path.join(a.model_dir, rel)
        if not os.path.isfile(path):
            print(f"BAD  missing  {rel}")
            bad += 1
            continue
        got_size = os.path.getsize(path)
        if got_size != size:
            print(f"BAD  size     {rel}: {got_size} != {size}")
            bad += 1
            continue
        if not want:
            oksize += 1
            continue
        if a.size_only and size > 64 << 20:
            oksize += 1
            continue
        print(f"hashing {i}/{len(rows)} {rel} ({size / 1e9:.1f} GB)", flush=True)
        if sha256_file(path) == want:
            ok += 1
        else:
            print(f"BAD  sha256   {rel}")
            bad += 1
    print(f"summary: ok={ok} oksize={oksize} bad={bad} rows={len(rows)}")
    pins_ok = True
    for rel, want in PINS.items():
        path = os.path.join(a.model_dir, rel)
        got = sha256_file(path) if os.path.isfile(path) else "missing"
        good = got == want
        pins_ok &= good
        print(f"pin  {rel}: {'OK' if good else 'MISMATCH ' + got}")
    cfg_ok = True
    try:
        with open(os.path.join(a.model_dir, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        text_cfg = cfg.get("text_config") or {}
        for k, v in CONFIG_EXPECT.items():
            got = cfg.get(k, text_cfg.get(k))  # layer counts sit under text_config in this pack
            if k == "quantization" and isinstance(got, dict):
                got = {kk: got.get(kk) for kk in v}
            good = got == v
            cfg_ok &= good
            print(f"config {k} = {json.dumps(got)} {'OK' if good else '(expected ' + json.dumps(v) + ')'}")
    except (OSError, ValueError) as exc:
        print(f"config.json unreadable: {exc}")
        cfg_ok = False
    passed = bad == 0 and ok + oksize == len(rows) and pins_ok and cfg_ok
    if a.size_only and passed:
        print("PASS (size-only: large files were not hashed)")
    else:
        print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
