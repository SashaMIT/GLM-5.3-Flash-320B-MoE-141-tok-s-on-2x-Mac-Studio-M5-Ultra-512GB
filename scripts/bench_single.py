#!/usr/bin/env python3
"""Single-user benchmark: one request at a time against an OpenAI-compatible streaming server.

Method (Round 7 Part 1 in BENCHMARKS.md):
  * untimed warm-ups after the server starts (default: the first 3 `warmup_32` prompts of
    receipts/prompts/r5-prompts.json, max_tokens 32);
  * then every prompt in file order, repeated --repeats times (default 2), T=0, max_tokens 512,
    no reasoning_effort sent (the model's chat template default applies), one request at a time;
  * n_out comes from the server's `usage` block, never from counting stream chunks;
  * TTFT = t_first - t0; time-to-answer = t_answer - t0 (blank if the model was still thinking);
    decode tok/s = (n_out - 1) / (t_last - t_first); E2E tok/s = n_out / (t_last - t0);
  * cached_tokens must be 0 on every timed request (start a fresh server per block) - the run
    stops otherwise, unless --allow-cached;
  * watchdogs: no first token in 120 s (300 s for prompts of >= 6,000 tokens or type
    qa_doc_2k_8k), or no new token for 30 s -> the block stops and is marked stopped_watchdog.

Withheld prompts (prompt == null in the public files) are skipped and listed.
Output: one CSV row per request, with the same columns as receipts/r7-part1-requests.csv.

Example (TensorFold arm T1 on the published realistic prompts):
  python3 scripts/bench_single.py --target http://127.0.0.1:8020 \\
      --prompts receipts/prompts/r6-real-public.json --arm T1 --restart 1 \\
      --server-log <tf-log> --require-mtp --out results/t1.csv
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oaistream as oa  # noqa: E402

FIELDS = ["block_tag", "restart", "arm", "block_status", "pass", "prompt_idx", "prompt_id", "prompt_type",
          "error", "hang", "n_in", "n_out", "cached_tokens", "ttft_s", "time_to_answer_s", "decode_tok_s",
          "e2e_tok_s", "drafted", "draft_accepted", "draft_offered", "max_inter_tok_s", "finish_reason", "t0_wall"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True, help="server base URL, e.g. http://127.0.0.1:8020")
    p.add_argument("--model", default="default_model", help="value of the request's model field (default: %(default)s)")
    p.add_argument("--prompts", required=True, help="prompt file, e.g. receipts/prompts/r6-real-public.json")
    p.add_argument("--set", dest="set_name", help="prompt set inside the file when it has several (e.g. short_32)")
    p.add_argument("--ids", help="comma-separated prompt ids to keep (default: all published prompts)")
    p.add_argument("--repeats", type=int, default=2, help="passes over the prompt list (default: %(default)s)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--reasoning-effort", help="send reasoning_effort in the body (default: not sent)")
    p.add_argument("--seed", type=int, help="send seed in the body (default: not sent)")
    p.add_argument("--extra-body", help="JSON object merged into every request body")
    p.add_argument("--warmup-prompts", default=oa.WARMUP_FILE_DEFAULT, help="file with warm-up prompts (default: r5-prompts.json)")
    p.add_argument("--warmup-set", default="warmup_32")
    p.add_argument("--warmups", type=int, default=3, help="untimed warm-up requests (default: %(default)s)")
    p.add_argument("--warmup-max-tokens", type=int, default=32)
    p.add_argument("--prefill-timeout", type=float, default=120.0)
    p.add_argument("--long-prefill-timeout", type=float, default=300.0)
    p.add_argument("--decode-watchdog", type=float, default=30.0)
    p.add_argument("--allow-cached", action="store_true", help="record cached_tokens > 0 instead of stopping")
    p.add_argument("--server-log", help="server log file to read drafting counters from ('done ... accepted=A/O' lines)")
    p.add_argument("--require-mtp", action="store_true",
                   help="stop unless --server-log shows the TensorFold MTP-loaded lines ('... and the MTP layer', 'drafts: MTP')")
    p.add_argument("--arm", default="arm", help="arm label written to the CSV (e.g. T1)")
    p.add_argument("--restart", default="1", help="restart label written to the CSV")
    p.add_argument("--block-tag", help="block label (default: <arm>-r<restart>)")
    p.add_argument("--out", required=True, help="CSV file to write")
    p.add_argument("--append", action="store_true", help="append to --out instead of overwriting")
    return p.parse_args(argv)


def summarise(rows: list, arm: str) -> None:
    ok = [r for r in rows if not r["error"]]
    if not ok:
        print(f"[{arm}] no successful requests")
        return
    e2e = [r["e2e_tok_s"] for r in ok]
    dec = [r["decode_tok_s"] for r in ok]
    ttft = [r["ttft_s"] for r in ok]
    lo, hi = oa.bootstrap_median_ci(e2e, n_boot=2000)
    fr = collections.Counter(r["finish_reason"] for r in ok)
    acc = sum(r["draft_accepted"] for r in ok)
    off = sum(r["draft_offered"] for r in ok)
    drafted = f"yes ({acc}/{off})" if off > 0 else "no (0/0) or not reported"
    print(f"[{arm}] N={len(ok)}  E2E median {oa.fmt(oa.median_or_none(e2e), 1)} tok/s "
          f"[95% CI {oa.fmt(lo, 1)}-{oa.fmt(hi, 1)}]  decode median {oa.fmt(oa.median_or_none(dec), 1)} tok/s  "
          f"TTFT median {oa.fmt(oa.median_or_none(ttft))} s (p95 {oa.fmt(oa.p95_rule(ttft))} s)  "
          f"finish {dict(fr)}  drafted {drafted}  cached_max {max(r['cached_tokens'] for r in ok)}")


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        host, port, path = oa.parse_target(a.target)
        extra = oa.parse_extra_body(a.extra_body)
        prompts = oa.load_prompts(a.prompts, a.set_name)
        warm = oa.load_prompts(a.warmup_prompts, a.warmup_set)[: a.warmups] if a.warmups > 0 else []
        log = oa.LogTail(a.server_log) if a.server_log else None
    except oa.UsageError as exc:
        oa.die(str(exc))
    if a.require_mtp:
        if not a.server_log:
            oa.die("--require-mtp needs --server-log")
        ok, lines = oa.mtp_line_status(a.server_log)
        if not ok:
            oa.die("MTP-loaded lines not found in the server log; label this arm 'plain decode' or fix the launch", 4)
        print("MTP evidence: " + " | ".join(lines))
    if a.ids:
        keep = set(x.strip() for x in a.ids.split(",") if x.strip())
        prompts = [p for p in prompts if p["id"] in keep]
        missing = keep - set(p["id"] for p in prompts)
        if missing:
            oa.die(f"ids not in the prompt file: {', '.join(sorted(missing))}")
    withheld = [p["id"] for p in prompts if p["withheld"]]
    prompts = [p for p in prompts if not p["withheld"]]
    if withheld:
        print(f"skipping {len(withheld)} withheld prompt(s) (text not published): {', '.join(withheld)}")
    if not prompts:
        oa.die("no runnable prompts")
    err = oa.check_server(host, port)
    if err:
        oa.die(err)

    for i, w in enumerate(warm):
        body = oa.build_body(a.model, w["prompt"], a.warmup_max_tokens, 0.0, a.reasoning_effort, a.seed, extra)
        r = oa.stream_chat(host, port, path, body, a.prefill_timeout, a.decode_watchdog)
        if r["error"]:
            oa.die(f"warm-up {i} failed: {r['error']}", 3)
    if warm:
        print(f"{len(warm)} warm-up request(s) done")

    tag = a.block_tag or f"{a.arm}-r{a.restart}"
    rows, status = [], "ok"
    for pass_idx in range(1, a.repeats + 1):
        if status != "ok":
            break
        for idx, p in enumerate(prompts):
            long_prompt = p["expected_tokens"] >= 6000 or p["type"] == "qa_doc_2k_8k" or len(p["prompt"]) > 24000
            body = oa.build_body(a.model, p["prompt"], a.max_tokens, a.temperature, a.reasoning_effort, a.seed, extra)
            if log:
                log.mark()
            r = oa.stream_chat(host, port, path, body,
                               a.long_prefill_timeout if long_prompt else a.prefill_timeout, a.decode_watchdog)
            row = {"block_tag": tag, "restart": a.restart, "arm": a.arm, "pass": pass_idx, "prompt_idx": idx,
                   "prompt_id": p["id"], "prompt_type": p["type"], "error": r["error"], "hang": r["hang"]}
            if r["error"]:
                rows.append(row)
                status = "stopped_watchdog"
                print(f"[{tag}] pass {pass_idx} {p['id']}: {r['error']} -> block stopped", file=sys.stderr)
                break
            m = oa.metrics(r)
            if log:
                d = oa.drafts_from_log_text(log.new_text())
                if d is not None:
                    m["draft_accepted"], m["draft_offered"] = d
                    m["drafted"] = d[1] > 0
            row.update({k: m[k] for k in FIELDS if k in m})
            rows.append(row)
            print(f"[{tag}] pass {pass_idx} {idx + 1:2d}/{len(prompts)} {p['id']:<14} n_out={m['n_out']:<5} "
                  f"e2e={m['e2e_tok_s']:7.2f} dec={m['decode_tok_s']:7.2f} ttft={oa.fmt(m['ttft_s'])}s", flush=True)
            if m["cached_tokens"] and not a.allow_cached:
                status = "error"
                print(f"[{tag}] cached_tokens={m['cached_tokens']} on {p['id']}: restart the server with prompt "
                      f"caching off (or pass --allow-cached) -> block stopped", file=sys.stderr)
                break
    for row in rows:
        row["block_status"] = status
    try:
        oa.write_csv(a.out, rows, FIELDS, append=a.append)
    except oa.UsageError as exc:
        oa.die(str(exc))
    print(f"wrote {len(rows)} row(s) to {a.out} (block_status={status})")
    summarise(rows, a.arm)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
