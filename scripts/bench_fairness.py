#!/usr/bin/env python3
"""Long-prompt fairness: one long prompt plus 8 short prompts arriving just after it (Round 7 Part 3).

Method:
  * long prompt: lc-32k from receipts/prompts/tf1-prompts-public.json (31,995 tokens of IETF RFC 9110),
    sent at t = 0;
  * 8 short prompts: the first 8 of `short_32` in receipts/prompts/r5-prompts.json, sent at
    0.20 s, 0.45 s, ... 1.95 s (0.25 s apart);
  * T=0, max_tokens 256, no reasoning_effort sent, 3 untimed warm-ups, 2 repeats, no pre-connect;
  * before the mixed runs each short prompt runs once alone ("solo") so we can say whether the
    short outputs under load are identical (sha256 of reasoning + NUL + content);
  * reported: short-prompt TTFT p50 / p95 / max, longest inter-token gap over all streams, long-prompt
    TTFT, total tok/s = sum(n_out) / (latest t_last - earliest t0), and solo matches.

Output: one CSV row per stream with the columns of receipts/r7-part3-requests.csv.

Example:
  python3 scripts/bench_fairness.py --target http://127.0.0.1:8020 --arm tf_default --engine tensorfold \\
      --out results/fairness.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oaistream as oa  # noqa: E402

PROMPTS_DIR = os.path.dirname(oa.WARMUP_FILE_DEFAULT)
FIELDS = ["arm", "engine", "rep", "role", "user_id", "error", "n_in", "n_out", "cached_tokens", "ttft_s",
          "time_to_answer_s", "e2e_tok_s", "decode_tok_s", "max_inter_tok_s", "solo_text_match"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True)
    p.add_argument("--model", default="default_model")
    p.add_argument("--long-prompts", default=os.path.join(PROMPTS_DIR, "tf1-prompts-public.json"))
    p.add_argument("--long-set", default="long_context")
    p.add_argument("--long-id", default="lc-32k")
    p.add_argument("--short-prompts", default=os.path.join(PROMPTS_DIR, "r5-prompts.json"))
    p.add_argument("--short-set", default="short_32")
    p.add_argument("--n-short", type=int, default=8)
    p.add_argument("--first-short-delay", type=float, default=0.20, help="seconds after the long prompt (default: %(default)s)")
    p.add_argument("--stagger", type=float, default=0.25, help="seconds between short prompts (default: %(default)s)")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--reasoning-effort")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--warmup-prompts", default=oa.WARMUP_FILE_DEFAULT)
    p.add_argument("--warmup-set", default="warmup_32")
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--no-solo", action="store_true", help="skip the solo runs (solo_text_match stays blank)")
    p.add_argument("--prefill-timeout", type=float, default=300.0)
    p.add_argument("--decode-watchdog", type=float, default=30.0)
    p.add_argument("--allow-cached", action="store_true")
    p.add_argument("--arm", default="arm")
    p.add_argument("--engine", default="engine")
    p.add_argument("--out", required=True)
    p.add_argument("--append", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        host, port, path = oa.parse_target(a.target)
        longs = [p for p in oa.load_prompts(a.long_prompts, a.long_set) if p["id"] == a.long_id]
        if not longs:
            raise oa.UsageError(f"{a.long_id!r} not found in {a.long_prompts} [{a.long_set}]")
        long_p = longs[0]
        if long_p["withheld"]:
            raise oa.UsageError(f"{a.long_id} is withheld in {a.long_prompts}; pick a published long prompt")
        shorts = [p for p in oa.load_prompts(a.short_prompts, a.short_set) if not p["withheld"]][: a.n_short]
        warm = oa.load_prompts(a.warmup_prompts, a.warmup_set)[: a.warmups] if a.warmups > 0 else []
    except oa.UsageError as exc:
        oa.die(str(exc))
    err = oa.check_server(host, port)
    if err:
        oa.die(err)

    def body(text, max_tok):
        return oa.build_body(a.model, text, max_tok, a.temperature, a.reasoning_effort)

    for w in warm:
        r = oa.stream_chat(host, port, path, body(w["prompt"], 16), 120.0, a.decode_watchdog)
        if r["error"]:
            oa.die(f"warm-up failed: {r['error']}", 3)

    solo = {}
    if not a.no_solo:
        for sp in shorts:
            r = oa.stream_chat(host, port, path, body(sp["prompt"], a.max_tokens), a.prefill_timeout, a.decode_watchdog)
            if r["error"]:
                oa.die(f"solo run of {sp['id']} failed: {r['error']}", 3)
            solo[sp["id"]] = oa.metrics(r)["text_sha256"]
        print(f"solo runs done for {len(solo)} short prompt(s)")

    rows, rc = [], 0
    for rep in range(1, a.repeats + 1):
        long_label = "long-32k" if long_p["id"] == "lc-32k" else f"long-{long_p['id']}"  # receipt naming
        jobs = [(long_label, long_label, long_p["prompt"], 0.0)]
        jobs += [("short", sp["id"], sp["prompt"], a.first_short_delay + i * a.stagger) for i, sp in enumerate(shorts)]
        results = [None] * len(jobs)
        barrier = threading.Barrier(len(jobs))

        def worker(i):
            role, uid, text, delay = jobs[i]
            barrier.wait(timeout=60)
            if delay > 0:
                time.sleep(delay)
            results[i] = oa.stream_chat(host, port, path, body(text, a.max_tokens), a.prefill_timeout, a.decode_watchdog)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(jobs))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        ok, short_ttft, gaps, long_ttft, matches = [], [], [], None, 0
        for (role, uid, _, _), r in zip(jobs, results):
            row = {"arm": a.arm, "engine": a.engine, "rep": rep, "role": role, "user_id": uid, "error": r["error"]}
            if not r["error"]:
                m = oa.metrics(r)
                row.update({k: m[k] for k in FIELDS if k in m})
                ok.append((r, m))
                gaps.append(m["max_inter_tok_s"])
                if m["cached_tokens"] and not a.allow_cached:
                    oa.die(f"cached_tokens={m['cached_tokens']} on {uid}: restart the server with caching off", 3)
                if role == "short":
                    short_ttft.append(m["ttft_s"])
                    if solo:
                        row["solo_text_match"] = m["text_sha256"] == solo.get(uid)
                        matches += bool(row["solo_text_match"])
                else:
                    long_ttft = m["ttft_s"]
            else:
                rc = 1
            rows.append(row)
        total = None
        if ok:
            span = max(r["t_last"] for r, _ in ok) - min(r["t0"] for r, _ in ok)
            total = sum(m["n_out"] for _, m in ok) / max(span, 1e-9)
        print(f"[{a.arm} rep {rep}] short TTFT p50 {oa.fmt(oa.median_or_none(short_ttft))} p95 {oa.fmt(oa.p95_rule(short_ttft))} "
              f"max {oa.fmt(max(short_ttft) if short_ttft else None)} s | longest gap {oa.fmt(max(gaps) if gaps else None)} s | "
              f"long TTFT {oa.fmt(long_ttft)} s | total {oa.fmt(total, 1)} tok/s | "
              f"solo match {matches}/{len(shorts) if solo else 0} | errors {len(jobs) - len(ok)}", flush=True)
    try:
        oa.write_csv(a.out, rows, FIELDS, append=a.append)
    except oa.UsageError as exc:
        oa.die(str(exc))
    print(f"wrote {len(rows)} row(s) to {a.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
