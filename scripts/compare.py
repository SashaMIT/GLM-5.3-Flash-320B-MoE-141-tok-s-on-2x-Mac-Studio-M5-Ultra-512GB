#!/usr/bin/env python3
"""Compare a fresh results CSV with our receipts: median, 95% bootstrap interval and % difference per arm.

The CSV type is detected from its columns:
  single-user   (bench_single.py, receipts/r7-part1-requests.csv): per arm, E2E tok/s (the pass/fail metric),
                decode tok/s and TTFT. Only prompt ids present in BOTH files are compared, so a run on the
                16 published prompts is compared with the same 16 prompts in the receipt, not with all 24.
                The interval is a 10,000-resample bootstrap of the median over (restart, prompt) cells,
                each cell being the median of its passes (the method behind BENCHMARKS.md section 1.7).
  many-users    (bench_many.py --out, receipts/r7-part2-requests.csv): per system x users x prompt kind,
                per-user E2E (pass/fail) and TTFT.
  curve         (bench_many.py --summary-out, receipts/r7-crossover-curve.csv): per arm x users x prompt kind,
                total E2E over repeats (pass/fail) and TTFT p95.
  fairness      (bench_fairness.py, receipts/r7-part3-requests.csv): per arm x role, TTFT (pass/fail) and E2E.

Arms are matched by name; use --arm-map FRESH=RECEIPT (repeatable) when your labels differ.
Exit code: 0 when every pass/fail metric is within --tolerance (default 5%), 1 otherwise, 2 on bad input.
The tolerance is a guide, not a statistical test: see AGENTS.md, "Quick replication".

Example:
  python3 scripts/compare.py --fresh results/t1.csv --receipt receipts/r7-part1-requests.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oaistream as oa  # noqa: E402

KINDS = {
    # kind: (required columns, group-by columns, [(metric, label, is_pass_fail)], filter)
    "single-user": ({"arm", "prompt_id", "e2e_tok_s"}, ["arm"],
                    [("e2e_tok_s", "E2E tok/s", True), ("decode_tok_s", "decode tok/s", False), ("ttft_s", "TTFT s", False)]),
    "many-users": ({"system", "n_users", "prompt_kind", "e2e_tok_s"}, ["system", "n_users", "prompt_kind"],
                   [("e2e_tok_s", "per-user E2E tok/s", True), ("ttft_s", "TTFT s", False)]),
    "curve": ({"arm", "users", "prompt_len", "agg_e2e_tok_s"}, ["arm", "users", "prompt_len"],
              [("agg_e2e_tok_s", "total E2E tok/s", True), ("ttft_p95_s", "TTFT p95 s", False)]),
    "fairness": ({"arm", "role", "ttft_s"}, ["arm", "role"],
                 [("ttft_s", "TTFT s", True), ("e2e_tok_s", "E2E tok/s", False)]),
}


def read_csv(path: str) -> list:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError as exc:
        raise oa.UsageError(f"cannot read {path}: {exc}")


def detect(rows: list, path: str) -> str:
    if not rows:
        raise oa.UsageError(f"{path} has no rows")
    cols = set(rows[0].keys())
    for kind in ("fairness", "curve", "many-users", "single-user"):
        if KINDS[kind][0].issubset(cols):
            return kind
    raise oa.UsageError(f"{path}: columns not recognised ({', '.join(sorted(cols))})")


def usable(r: dict) -> bool:
    if (r.get("error") or "").strip():
        return False
    if r.get("block_status") not in (None, "", "ok"):
        return False
    return True


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def cell_values(rows: list, metric: str, kind: str) -> list:
    """Values that the bootstrap resamples: (restart, prompt) cell medians for single-user, else raw values."""
    if kind != "single-user":
        return [v for v in (num(r.get(metric)) for r in rows) if v is not None]
    cells = {}
    for r in rows:
        v = num(r.get(metric))
        if v is not None:
            cells.setdefault((r.get("restart", ""), r["prompt_id"]), []).append(v)
    return [statistics.median(v) for v in cells.values()]


def describe(rows: list, metric: str, kind: str, n_boot: int):
    vals = [v for v in (num(r.get(metric)) for r in rows) if v is not None]
    if not vals:
        return None, None, None, 0
    lo, hi = oa.bootstrap_median_ci(cell_values(rows, metric, kind), n_boot=n_boot)
    return statistics.median(vals), lo, hi, len(vals)


def fmt_med(m, lo, hi) -> str:
    if m is None:
        return "-"
    if lo is None:
        return f"{m:.2f}"
    return f"{m:.2f} ± {max(hi - m, m - lo):.2f} [{lo:.2f}-{hi:.2f}]"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fresh", required=True, help="your results CSV")
    p.add_argument("--receipt", required=True, help="our receipt CSV of the same kind (receipts/...)")
    p.add_argument("--arm-map", action="append", default=[], help="FRESH=RECEIPT arm/system label mapping")
    p.add_argument("--tolerance", type=float, default=0.05, help="allowed |difference| of the pass/fail median (default 0.05 = 5%%)")
    p.add_argument("--n-boot", type=int, default=10000)
    a = p.parse_args(argv)
    try:
        fresh, rec = read_csv(a.fresh), read_csv(a.receipt)
        kf, kr = detect(fresh, a.fresh), detect(rec, a.receipt)
        if kf != kr:
            raise oa.UsageError(f"different kinds: {a.fresh} is {kf}, {a.receipt} is {kr}")
        amap = {}
        for m in a.arm_map:
            if "=" not in m:
                raise oa.UsageError(f"--arm-map needs FRESH=RECEIPT, got {m!r}")
            k, v = m.split("=", 1)
            amap[k.strip()] = v.strip()
    except oa.UsageError as exc:
        oa.die(str(exc))
    _, keys, mets = KINDS[kf]
    fresh = [r for r in fresh if usable(r)]
    rec = [r for r in rec if usable(r)]

    def gkey(r, mapped):
        k = [r.get(c, "") for c in keys]
        if mapped:
            k[0] = amap.get(k[0], k[0])
        return tuple(k)

    fg, rg = {}, {}
    for r in fresh:
        fg.setdefault(gkey(r, True), []).append(r)
    for r in rec:
        rg.setdefault(gkey(r, False), []).append(r)

    print(f"kind: {kf}; tolerance ±{a.tolerance:.0%} on the first metric of each group; intervals: 95% bootstrap of the median")
    all_ok, compared = True, 0
    for g in sorted(fg):
        fr = fg[g]
        if g not in rg:
            print(f"\n{' / '.join(map(str, g))}: no matching group in the receipt "
                  f"(receipt groups: {', '.join(' / '.join(map(str, x)) for x in sorted(rg))}); use --arm-map")
            all_ok = False
            continue
        rr = rg[g]
        if kf == "single-user":
            common = set(r["prompt_id"] for r in fr) & set(r["prompt_id"] for r in rr)
            if not common:
                print(f"\n{g[0]}: no prompt ids in common")
                all_ok = False
                continue
            fr = [r for r in fr if r["prompt_id"] in common]
            rr = [r for r in rr if r["prompt_id"] in common]
            scope = f"{len(common)} prompt id(s) in common"
        else:
            scope = ""
        print(f"\n{' / '.join(map(str, g))}  {scope}")
        print(f"  {'metric':<20} {'fresh median ± [95% CI]':<34} {'receipt median ± [95% CI]':<34} {'diff':>8}  n fresh/receipt")
        for metric, label, pf in mets:
            fm, flo, fhi, fn = describe(fr, metric, kf, a.n_boot)
            rm, rlo, rhi, rn = describe(rr, metric, kf, a.n_boot)
            if fm is None or rm is None or rm == 0:
                print(f"  {label:<20} {'-':<34} {'-':<34} {'-':>8}")
                continue
            diff = fm / rm - 1.0
            verdict = ""
            if pf:
                compared += 1
                ok = abs(diff) <= a.tolerance
                all_ok &= ok
                verdict = "  PASS" if ok else "  OUTSIDE TOLERANCE"
            print(f"  {label:<20} {fmt_med(fm, flo, fhi):<34} {fmt_med(rm, rlo, rhi):<34} {diff * 100:+7.1f}%  {fn}/{rn}{verdict}")
    if compared == 0:
        print("\nnothing was compared")
        return 1
    print("\nresult: " + ("all pass/fail metrics within tolerance" if all_ok else "at least one metric outside tolerance or unmatched"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
