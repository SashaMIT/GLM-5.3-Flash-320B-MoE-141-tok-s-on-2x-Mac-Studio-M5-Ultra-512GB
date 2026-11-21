#!/usr/bin/env python3
"""Many-users benchmark: N simultaneous users behind a start barrier (Round 7 Part 2 in BENCHMARKS.md).

Method:
  * prompts: receipts/prompts/r5-prompts.json, set short_32 (short) or conc_8k (~8k tokens); user i gets
    prompt i mod len(set) with a unique first line "[<step>-r<repeat>-u<i>-<nonce>]" so no request can
    reuse another's prompt cache (cached_tokens must be 0);
  * every user thread opens its TCP connection first (4 ms apart, with retries), then all wait on one
    barrier and send at the same moment ("pre-connected"); --raw-burst connects at release instead
    (shows servers whose listen backlog is too small);
  * T=0, max_tokens 256, 3 untimed warm-ups (max_tokens 16) before the first rung, 3 repeats per rung;
  * several --target servers split users round-robin (user 0 -> first server, user 1 -> second, ...);
  * total E2E = sum(n_out) / (latest t_last - earliest t0) over the users that finished;
    per-user E2E = each user's n_out / (t_last - t0), reported as median and minimum - never total / N;
    TTFT p50, p95 (sorted value at index ceil(0.95 (n - 1))) and max;
  * a rung with more than 10% failed users stops the ladder (not in --raw-burst mode).

Outputs: --out (one row per user, columns of receipts/r7-part2-requests.csv) and optionally
--summary-out (one row per repeat, columns of receipts/r7-crossover-curve.csv minus memory).

Example (one TensorFold per Mac, through two local ports):
  python3 scripts/bench_many.py --target http://127.0.0.1:8020 --target http://127.0.0.1:8021 \\
      --users 1,2,4,8,16,32 --set short_32 --system S3 --out results/s3.csv --summary-out results/s3-curve.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oaistream as oa  # noqa: E402

REQ_FIELDS = ["system", "step_name", "n_users", "prompt_kind", "repeat", "connect_before_barrier", "user_id",
              "target_tag", "error", "hang", "n_in", "n_out", "cached_tokens", "ttft_s", "time_to_answer_s",
              "e2e_tok_s", "decode_tok_s", "max_inter_tok_s", "drafted"]
SUM_FIELDS = ["arm", "users", "prompt_len", "repeat", "users_ok", "users_failed", "agg_e2e_tok_s",
              "per_user_e2e_med", "per_user_e2e_min", "per_stream_dec_med", "ttft_p50_s", "ttft_p95_s",
              "ttft_max_s", "drafted", "draft_accepted", "draft_offered", "label"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", action="append", required=True, help="server base URL; repeat for several servers")
    p.add_argument("--model", action="append", help="model field; repeat once per --target, or give once for all")
    p.add_argument("--users", default="1,2,4,8", help="comma-separated ladder of user counts (default: %(default)s)")
    p.add_argument("--prompts", default=os.path.join(os.path.dirname(oa.WARMUP_FILE_DEFAULT), "r5-prompts.json"))
    p.add_argument("--set", dest="set_name", default="short_32", help="short_32 or conc_8k (default: %(default)s)")
    p.add_argument("--prompt-kind", help="label for the CSV (default: 8k for conc_8k, else short)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--reasoning-effort", help="send reasoning_effort in the body (e.g. low for mlx-vlm; default: not sent)")
    p.add_argument("--extra-body", help="JSON object merged into every request body")
    p.add_argument("--no-prefix", action="store_true", help="do not add the unique per-user first line")
    p.add_argument("--raw-burst", action="store_true", help="connect at release instead of before the barrier")
    p.add_argument("--connect-stagger-ms", type=float, default=4.0)
    p.add_argument("--warmup-prompts", default=oa.WARMUP_FILE_DEFAULT)
    p.add_argument("--warmup-set", default="warmup_32")
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--prefill-timeout", type=float, help="first-token limit in s (default: 120 short, 300 for 8k)")
    p.add_argument("--decode-watchdog", type=float, default=30.0)
    p.add_argument("--max-error-frac", type=float, default=0.10)
    p.add_argument("--allow-cached", action="store_true")
    p.add_argument("--server-log", action="append", help="server log(s) to read 'done ... accepted=A/O' drafting lines from")
    p.add_argument("--system", default="S", help="system label written to the CSV (e.g. S3)")
    p.add_argument("--label", default="4-bit, lossy", help="label column of the summary CSV")
    p.add_argument("--out", required=True)
    p.add_argument("--summary-out")
    p.add_argument("--append", action="store_true")
    return p.parse_args(argv)


def run_burst(targets, models, prompts_text, a, prefill_s, max_tokens, preconnect, extra):
    """Send len(prompts_text) requests released together. Returns list of (target_idx, result)."""
    n = len(prompts_text)
    barrier = threading.Barrier(n)
    out = [None] * n

    def worker(i):
        ti = i % len(targets)
        host, port, path = targets[ti]
        body = oa.build_body(models[ti], prompts_text[i], max_tokens, a.temperature, a.reasoning_effort, None, extra)
        conn = None
        if preconnect:
            time.sleep(i * a.connect_stagger_ms / 1000.0)
            try:
                conn = oa.open_conn(host, port, prefill_s)
            except OSError as exc:
                try:
                    barrier.wait(timeout=60)
                except threading.BrokenBarrierError:
                    pass
                out[i] = (ti, {"error": f"pre_connect: {type(exc).__name__}: {exc}", "hang": False})
                return
        try:
            barrier.wait(timeout=60)
        except threading.BrokenBarrierError as exc:
            out[i] = (ti, {"error": f"barrier: {exc!r}", "hang": False})
            if conn:
                conn.close()
            return
        out[i] = (ti, oa.stream_chat(host, port, path, body, prefill_s, a.decode_watchdog, conn=conn))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        targets = [oa.parse_target(t) for t in a.target]
        models = a.model or ["default_model"]
        if len(models) == 1:
            models = models * len(targets)
        if len(models) != len(targets):
            raise oa.UsageError("give --model once, or once per --target")
        ladder = [int(x) for x in a.users.split(",") if x.strip()]
        if not ladder or min(ladder) < 1:
            raise oa.UsageError("--users must be positive integers")
        pool = [p for p in oa.load_prompts(a.prompts, a.set_name) if not p["withheld"]]
        warm = oa.load_prompts(a.warmup_prompts, a.warmup_set)[: a.warmups] if a.warmups > 0 else []
        extra = oa.parse_extra_body(a.extra_body)
        logs = [oa.LogTail(x) for x in (a.server_log or [])]
    except (oa.UsageError, ValueError) as exc:
        oa.die(str(exc))
    if not pool:
        oa.die("no runnable prompts in the chosen set")
    kind = a.prompt_kind or ("8k" if "8k" in a.set_name else "short")
    prefill_s = a.prefill_timeout or (300.0 if kind == "8k" else 120.0)
    for host, port, _ in targets:
        err = oa.check_server(host, port)
        if err:
            oa.die(err)

    def nonced(texts, tag):
        if a.no_prefix:
            return list(texts)
        return [f"[{tag}-u{i:02d}-{uuid.uuid4().hex[:8]}]\n{t}" for i, t in enumerate(texts)]

    if warm:
        w = run_burst(targets, models, nonced([x["prompt"] for x in warm], "warmup"), a, 120.0, 16, True, extra)
        bad = [r for _, r in w if r.get("error")]
        if bad:
            oa.die(f"{len(bad)} warm-up request(s) failed: {bad[0]['error']}", 3)
        print(f"{len(w)} warm-up request(s) done")

    req_rows, sum_rows, rc = [], [], 0
    for n_users in ladder:
        step = f"{a.system.lower()}-{kind}-u{n_users}"
        stop = False
        for rep in range(a.repeats):
            texts = nonced([pool[i % len(pool)]["prompt"] for i in range(n_users)], f"{step}-r{rep}")
            for lg in logs:
                lg.mark()
            res = run_burst(targets, models, texts, a, prefill_s, a.max_tokens, not a.raw_burst, extra)
            ok, failed = [], 0
            for i, (ti, r) in enumerate(res):
                host, port, _ = targets[ti]
                row = {"system": a.system, "step_name": step, "n_users": n_users, "prompt_kind": kind, "repeat": rep,
                       "connect_before_barrier": not a.raw_burst, "user_id": f"u{i:02d}",
                       "target_tag": f"{host}:{port}", "error": r.get("error"), "hang": r.get("hang")}
                if not r.get("error"):
                    m = oa.metrics(r)
                    if m["n_out"] <= 0:
                        row["error"] = "n_out==0"
                    else:
                        row.update({k: m[k] for k in REQ_FIELDS if k in m})
                        ok.append((r, m))
                        if m["cached_tokens"] and not a.allow_cached:
                            oa.die(f"cached_tokens={m['cached_tokens']} in {step} u{i:02d}: use fresh servers "
                                   f"with prompt caching off (or --allow-cached)", 3)
                if row["error"]:
                    failed += 1
                req_rows.append(row)
            s = {"arm": a.system, "users": n_users, "prompt_len": kind, "repeat": rep, "users_ok": len(ok),
                 "users_failed": failed, "label": a.label}
            if ok:
                t0 = min(r["t0"] for r, _ in ok)
                tl = max(r["t_last"] for r, _ in ok)
                ttfts = [m["ttft_s"] for _, m in ok]
                acc = sum(m["draft_accepted"] for _, m in ok)
                off = sum(m["draft_offered"] for _, m in ok)
                if off == 0 and logs:
                    tot = [oa.drafts_from_log_text(lg.new_text()) for lg in logs]
                    acc = sum(x[0] for x in tot if x)
                    off = sum(x[1] for x in tot if x)
                s.update({
                    "agg_e2e_tok_s": sum(m["n_out"] for _, m in ok) / max(tl - t0, 1e-9),
                    "per_user_e2e_med": oa.median_or_none([m["e2e_tok_s"] for _, m in ok]),
                    "per_user_e2e_min": min(m["e2e_tok_s"] for _, m in ok),
                    "per_stream_dec_med": oa.median_or_none([m["decode_tok_s"] for _, m in ok]),
                    "ttft_p50_s": oa.median_or_none(ttfts), "ttft_p95_s": oa.p95_rule(ttfts), "ttft_max_s": max(ttfts),
                    "drafted": off > 0, "draft_accepted": acc, "draft_offered": off,
                })
            sum_rows.append(s)
            print(f"[{step} rep {rep}] ok {len(ok)}/{n_users}  total E2E {oa.fmt(s.get('agg_e2e_tok_s'), 1)} tok/s  "
                  f"per-user median {oa.fmt(s.get('per_user_e2e_med'), 1)} / min {oa.fmt(s.get('per_user_e2e_min'), 1)}  "
                  f"TTFT p50 {oa.fmt(s.get('ttft_p50_s'))} p95 {oa.fmt(s.get('ttft_p95_s'))} max {oa.fmt(s.get('ttft_max_s'))} s",
                  flush=True)
            if not a.raw_burst and failed > a.max_error_frac * n_users:
                print(f"[{step}] {failed}/{n_users} failed (> {a.max_error_frac:.0%}): ladder stopped", file=sys.stderr)
                stop, rc = True, 1
                break
        if stop:
            break
    try:
        oa.write_csv(a.out, req_rows, REQ_FIELDS, append=a.append)
        if a.summary_out:
            oa.write_csv(a.summary_out, sum_rows, SUM_FIELDS, append=a.append)
    except oa.UsageError as exc:
        oa.die(str(exc))
    print(f"wrote {len(req_rows)} request row(s) to {a.out}" + (f" and {len(sum_rows)} summary row(s) to {a.summary_out}" if a.summary_out else ""))
    return rc


if __name__ == "__main__":
    sys.exit(main())
