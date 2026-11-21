#!/usr/bin/env python3
"""Interconnect micro-benchmark: what one mlx.core.distributed.all_sum costs between GPU matmuls.

Method (BENCHMARKS.md section 1.5, AGENTS.md Step A):
  * x has shape (rows, 4096), bfloat16 (1 row = 8 KiB); w is a 4096 x 4096 bfloat16 identity matrix;
  * one "step" is a chain of --nsum (default 101) operations, evaluated once with mx.eval:
      A  matmul only           x = x @ w
      B  matmul + all_sum      x = all_sum(x @ w) * 0.5
      C  all_sum only          x = all_sum(x) * 0.5
  * each arm runs --warmup (20) untimed steps then --steps (1000) timed steps; we report the median (p50)
    and p99 step time and p50 / nsum ("per-sum");
  * the headline is the ADDED COST per sum = (p50(B) - p50(A)) / nsum.
Our lab results on two M5 Ultra Macs over one Thunderbolt 5 cable (jaccl, quiet Macs, 1 row, 1000 steps):
  default fence 187.8 us/sum, MLX_METAL_FAST_SYNCH=1 19.0 us/sum.

Run it on both Macs at once with MLX's own launcher, from <mlx-env> (all on one line):
  <mlx-env>/bin/mlx.launch --backend jaccl --hostfile <hosts-jaccl.json> --python <mlx-env>/bin/python \\
      scripts/sync_bench.py --rows 1
  (add  --env MLX_METAL_FAST_SYNCH=1  for the fast-sync arm; never on 8-bit / FP8 / MXFP8 model runs).
Started without a launcher it runs as a single process (group size 1, all_sum does no communication);
use that only to check that the script works.

This file is our own code and uses only public mlx.core APIs (mx.distributed.init, all_sum, eval).
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=1, help="rows of 4096 bf16 values per all_sum (1 row = 8 KiB)")
    p.add_argument("--dim", type=int, default=4096)
    p.add_argument("--nsum", type=int, default=101, help="operations chained per step")
    p.add_argument("--steps", type=int, default=1000, help="timed steps per arm")
    p.add_argument("--warmup", type=int, default=20, help="untimed steps per arm")
    p.add_argument("--backend", default="any", help="mx.distributed.init backend (jaccl on the Macs; default: %(default)s)")
    p.add_argument("--json", help="rank 0 writes the numbers to this JSON file")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if min(a.rows, a.dim, a.nsum, a.steps) < 1 or a.warmup < 0:
        print("error: --rows, --dim, --nsum and --steps must be >= 1", file=sys.stderr)
        return 2
    try:
        import mlx.core as mx
    except ImportError:
        print("error: mlx is not installed in this Python; run with <mlx-env>/bin/python (see env/README.md)", file=sys.stderr)
        return 2
    try:
        group = mx.distributed.init(backend=a.backend)
    except Exception as exc:  # backend missing or misconfigured
        print(f"error: mx.distributed.init(backend={a.backend!r}) failed: {exc}", file=sys.stderr)
        return 3
    rank, size = group.rank(), group.size()
    w = mx.eye(a.dim, dtype=mx.bfloat16)

    def step(x, comm: bool, matmul: bool):
        for _ in range(a.nsum):
            if matmul:
                x = x @ w
            if comm:
                x = mx.distributed.all_sum(x, group=group) * 0.5
        return x

    def run_arm(comm: bool, matmul: bool) -> dict:
        mx.eval(mx.distributed.all_sum(mx.zeros(1), group=group))  # line the ranks up
        x = mx.ones((a.rows, a.dim), dtype=mx.bfloat16)
        times = []
        for i in range(a.warmup + a.steps):
            t = time.perf_counter()
            mx.eval(step(x, comm, matmul))
            dt = time.perf_counter() - t
            if i >= a.warmup:
                times.append(dt)
        times.sort()
        p50 = times[len(times) // 2]
        p99 = times[min(len(times) - 1, int(len(times) * 0.99))]
        return {"p50_s": p50, "p99_s": p99, "per_sum_p50_us": p50 / a.nsum * 1e6}

    arms = {}
    for name, comm, matmul in (("matmul only", False, True), ("matmul + all_sum", True, True), ("all_sum only", True, False)):
        arms[name] = r = run_arm(comm, matmul)
        if rank == 0:
            print(f"{name:18s} rows={a.rows} p50 {r['p50_s'] * 1e3:8.3f} ms  p99 {r['p99_s'] * 1e3:8.3f} ms  "
                  f"per-sum p50 {r['per_sum_p50_us']:8.1f} us", flush=True)
    added = (arms["matmul + all_sum"]["p50_s"] - arms["matmul only"]["p50_s"]) / a.nsum * 1e6
    if rank == 0:
        print(f"ADDED COST per sum (B - A) / nsum: {added:.1f} us   (ranks: {size})", flush=True)
        if size == 1:
            print("note: single process - all_sum did no communication; launch with mlx.launch on two Macs", flush=True)
        if a.json:
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump({"rows": a.rows, "dim": a.dim, "nsum": a.nsum, "steps": a.steps, "warmup": a.warmup,
                           "ranks": size, "arms": arms, "added_cost_per_sum_us": added}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
