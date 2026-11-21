#!/usr/bin/env python3
"""Sampled GSM8K: 5 answers per question at T=0.6, majority vote, exact McNemar test (Round 7 Part 5).

Sub-commands:
  run      ask a server: the first --first questions (default 100) of receipts/prompts/r6-gsm8k.json,
           --samples answers each (default 5), T=0.6, max_tokens 4096, no reasoning_effort sent,
           3 untimed warm-ups; writes one CSV row per answer (columns of receipts/r7-part5-requests.csv
           plus `error`).
  score    read one or more such CSVs (e.g. receipts/r7-part5-requests.csv) and print, per arm,
           majority-vote accuracy, mean per-sample accuracy and mean n_out; with two arms also the paired
           exact McNemar test on majority-vote correctness.
  extract  show how one model reply is parsed (for checking the rule).

The same `run` also scores the 200-question generated suite of Round 6 Part A (receipts/prompts/r6-quality.json):
  run --prompts receipts/prompts/r6-quality.json --first 200 --samples 1 --temperature 0 --max-tokens 4096
Its items already end with their own ANSWER instruction (no suffix is added) and carry an answer_type:
int / numeric -> number rule below; exact_str -> exact text match; exact_str_ci -> case-insensitive match.
For Round 6 Part A on GSM8K use --first 200 --samples 1 --temperature 0 (majority vote of one sample = the sample).

Prompt: the question, then
  "\\n\\nSolve this problem step by step. End your response with the final number on its own last line in
  the exact format:\\nANSWER: <number>"

Answer rule (re-implemented from the description of our lab scorer):
  1. Use only the visible answer (delta.content), not the reasoning text.
  2. Split into lines, drop empty ones. If the last line is "ANSWER: <value>" (upper-case ANSWER, optional
     spaces, optionally wrapped in up to two '*' on each side) take <value>; otherwise take the lowest
     earlier line of that form; otherwise there is no answer (counts as wrong).
  3. Remove one pair of surrounding backticks. Number clean-up: drop a single trailing '.', all ',',
     a leading '$' or '+', a trailing '%'; parse as a decimal. Correct when it equals the reference number.
     The canonical form (used for voting) is the decimal without trailing zeros ("-0" -> "0"); if the
     value is not a number, the stripped text itself.
  4. Majority vote per question: the most frequent canonical answer among the samples that have one
     (ties go to the answer seen first); the question is right if that answer is correct.
McNemar: b = questions right only in arm A, c = right only in arm B; two-sided exact binomial
  p = min(1, 2 * sum_{i <= min(b, c)} C(b + c, i) / 2^(b + c)); p = 1 when b + c = 0.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from decimal import Decimal, InvalidOperation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oaistream as oa  # noqa: E402

SUFFIX = ("\n\nSolve this problem step by step. End your response with the final number on its own last line "
          "in the exact format:\nANSWER: <number>")
FIELDS = ["arm", "id", "sample_idx", "expected", "canonical", "correct", "n_out", "content_len", "ttft_s",
          "e2e_tok_s", "decode_tok_s", "error"]
ANSWER_RE = re.compile(r"^\s*\*{0,2}\s*ANSWER\s*:\s*(.+?)\s*\*{0,2}\s*$")


def find_answer(content: str):
    """Returns (value or None, where) with where in last_line / earlier_line / none."""
    lines = [ln.strip() for ln in (content or "").splitlines() if ln.strip()]
    for pos, ln in enumerate(reversed(lines)):
        m = ANSWER_RE.match(ln)
        if m:
            val = m.group(1).strip()
            if len(val) >= 2 and val[0] == "`" and val[-1] == "`":
                val = val[1:-1].strip()
            return val, ("last_line" if pos == 0 else "earlier_line")
    return None, "none"


def to_decimal(text):
    if text is None:
        return None
    s = text.strip()
    if s.endswith(".") and s.count(".") == 1:
        s = s[:-1].strip()
    s = s.replace(",", "").strip()
    if s.startswith("$"):
        s = s[1:].strip()
    if s.endswith("%"):
        s = s[:-1].strip()
    if s.startswith("+"):
        s = s[1:]
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError):
        return None
    return d if d.is_finite() else None


def score(content: str, expected: str, answer_type: str = "gsm8k_numeric") -> dict:
    val, where = find_answer(content)
    if val is None:
        return {"canonical": None, "correct": False, "where": where}
    if answer_type == "exact_str_ci":
        return {"canonical": val.strip().lower(), "correct": val.strip().lower() == str(expected).strip().lower(), "where": where}
    if answer_type == "exact_str":
        return {"canonical": val.strip(), "correct": val.strip() == str(expected).strip(), "where": where}
    d, e = to_decimal(val), to_decimal(str(expected))
    if d is not None:
        canon = format(d.normalize(), "f")
        if "." in canon:
            canon = canon.rstrip("0").rstrip(".")
        if canon == "-0":
            canon = "0"
    else:
        canon = val.strip()
    return {"canonical": canon, "correct": d is not None and e is not None and d == e, "where": where}


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def majority(samples: list) -> bool:
    """samples: [(sample_idx, canonical or None, correct)] -> majority-vote correctness."""
    counts, first, right = {}, {}, {}
    for idx, canon, ok in sorted(samples, key=lambda s: s[0]):
        if canon is None:
            continue
        counts[canon] = counts.get(canon, 0) + 1
        first.setdefault(canon, idx)
        right[canon] = right.get(canon, False) or ok
    if not counts:
        return False
    top = max(counts, key=lambda k: (counts[k], -first[k]))
    return right[top]


# ---------------------------------------------------------------------------

def cmd_run(a) -> int:
    try:
        host, port, path = oa.parse_target(a.target)
        with open(a.prompts, "r", encoding="utf-8") as f:
            items = json.load(f)["items"][: a.first]
        warm = oa.load_prompts(a.warmup_prompts, a.warmup_set)[: a.warmups] if a.warmups > 0 else []
    except (OSError, ValueError, KeyError, oa.UsageError) as exc:
        oa.die(f"cannot load prompts: {exc}")
    err = oa.check_server(host, port)
    if err:
        oa.die(err)
    for w in warm:
        r = oa.stream_chat(host, port, path, oa.build_body(a.model, w["prompt"], 16, 0.0), 120.0, a.decode_watchdog)
        if r["error"]:
            oa.die(f"warm-up failed: {r['error']}", 3)
    rows = []
    jl = open(a.jsonl, "w", encoding="utf-8") if a.jsonl else None
    for q, it in enumerate(items):
        # GSM8K items carry `question` (the suffix is added); generated-suite items (r6-quality.json) carry a
        # complete `prompt` with its own ANSWER instruction and an `answer_type`
        text = it["prompt"] if "prompt" in it else it["question"] + SUFFIX
        atype = it.get("answer_type", "gsm8k_numeric")
        for s in range(a.samples):
            body = oa.build_body(a.model, text, a.max_tokens, a.temperature, a.reasoning_effort)
            r = oa.stream_chat(host, port, path, body, a.prefill_timeout, a.decode_watchdog)
            row = {"arm": a.arm, "id": it["id"], "sample_idx": s, "expected": str(it["answer"]), "error": r["error"]}
            if not r["error"]:
                m = oa.metrics(r)
                sc = score(r["content"], it["answer"], atype)
                row.update({"canonical": sc["canonical"], "correct": sc["correct"], "n_out": m["n_out"],
                            "content_len": len(r["content"]), "ttft_s": m["ttft_s"], "e2e_tok_s": m["e2e_tok_s"],
                            "decode_tok_s": m["decode_tok_s"]})
                if m["cached_tokens"] and not a.allow_cached:
                    oa.die(f"cached_tokens={m['cached_tokens']} on {it['id']}: turn prompt caching off", 3)
            rows.append(row)
            if jl:
                jl.write(json.dumps({**row, "content": r.get("content", "")}, ensure_ascii=False) + "\n")
        print(f"[{a.arm}] {q + 1}/{len(items)} {it['id']}: " +
              " ".join(("ok" if x.get("correct") else ("ERR" if x.get("error") else "x")) for x in rows[-a.samples:]),
              flush=True)
    if jl:
        jl.close()
    oa.write_csv(a.out, rows, FIELDS, append=a.append)
    print(f"wrote {len(rows)} row(s) to {a.out}")
    report(rows)
    return 0


def load_rows(paths: list) -> list:
    rows = []
    for p in paths:
        try:
            with open(p, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    rows.append(r)
        except OSError as exc:
            oa.die(f"cannot read {p}: {exc}")
    need = {"arm", "id", "sample_idx", "canonical", "correct"}
    if rows and not need.issubset(rows[0].keys()):
        oa.die(f"CSV must have columns {sorted(need)}")
    return rows


def _b(x) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes")


def report(rows: list, arm_a: str = None, arm_b: str = None) -> dict:
    by_arm = {}
    for r in rows:
        if r.get("error"):
            continue
        canon = r.get("canonical")
        canon = None if canon in (None, "") else str(canon)
        by_arm.setdefault(str(r["arm"]), {}).setdefault(r["id"], []).append(
            (int(r["sample_idx"]), canon, _b(r["correct"]), r.get("n_out")))
    out = {}
    for arm, qs in by_arm.items():
        maj = {q: majority([(i, c, ok) for i, c, ok, _ in s]) for q, s in qs.items()}
        samples = [x for s in qs.values() for x in s]
        n_out = [float(x[3]) for x in samples if x[3] not in (None, "")]
        out[arm] = {"maj": maj, "n_q": len(qs), "n_s": len(samples), "s_ok": sum(1 for x in samples if x[2])}
        print(f"arm {arm}: majority vote {sum(maj.values())}/{len(qs)} ({100.0 * sum(maj.values()) / max(len(qs), 1):.1f}%), "
              f"samples {out[arm]['s_ok']}/{len(samples)} ({100.0 * out[arm]['s_ok'] / max(len(samples), 1):.1f}%), "
              f"mean n_out {oa.fmt(sum(n_out) / len(n_out) if n_out else None, 1)}")
    arms = list(out)
    if arm_a or arm_b:
        arms = [x for x in (arm_a, arm_b) if x]
        for x in arms:
            if x not in out:
                oa.die(f"arm {x!r} not in the data (have: {', '.join(out)})")
    if len(arms) >= 2:
        A, B = out[arms[0]]["maj"], out[arms[1]]["maj"]
        common = sorted(set(A) & set(B))
        both = sum(1 for q in common if A[q] and B[q])
        neither = sum(1 for q in common if not A[q] and not B[q])
        b = sum(1 for q in common if A[q] and not B[q])
        c = sum(1 for q in common if not A[q] and B[q])
        p = mcnemar_exact(b, c)
        print(f"paired majority vote {arms[0]} vs {arms[1]} (n={len(common)}): both right {both}, both wrong {neither}, "
              f"{arms[0]} only {b}, {arms[1]} only {c}, exact McNemar p = {p:.4f}")
        out["_mcnemar"] = {"b": b, "c": c, "p": p, "n": len(common)}
    return out


def cmd_score(a) -> int:
    report(load_rows(a.csv), a.arm_a, a.arm_b)
    return 0


def cmd_extract(a) -> int:
    text = a.text if a.text is not None else sys.stdin.read()
    print(json.dumps(score(text, a.expected, a.answer_type), ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="ask a server and write per-answer rows")
    r.add_argument("--target", required=True)
    r.add_argument("--model", default="default_model")
    r.add_argument("--prompts", default=os.path.join(os.path.dirname(oa.WARMUP_FILE_DEFAULT), "r6-gsm8k.json"))
    r.add_argument("--first", type=int, default=100)
    r.add_argument("--samples", type=int, default=5)
    r.add_argument("--temperature", type=float, default=0.6)
    r.add_argument("--max-tokens", type=int, default=4096)
    r.add_argument("--reasoning-effort")
    r.add_argument("--warmup-prompts", default=oa.WARMUP_FILE_DEFAULT)
    r.add_argument("--warmup-set", default="warmup_32")
    r.add_argument("--warmups", type=int, default=3)
    r.add_argument("--prefill-timeout", type=float, default=120.0)
    r.add_argument("--decode-watchdog", type=float, default=30.0)
    r.add_argument("--allow-cached", action="store_true")
    r.add_argument("--arm", default="arm", help="arm label, e.g. off / on")
    r.add_argument("--out", required=True)
    r.add_argument("--append", action="store_true", help="append (run each arm into the same CSV)")
    r.add_argument("--jsonl", help="also save every reply text here")
    s = sub.add_parser("score", help="score CSV file(s)")
    s.add_argument("csv", nargs="+")
    s.add_argument("--arm-a")
    s.add_argument("--arm-b")
    e = sub.add_parser("extract", help="parse one reply (from --text or stdin)")
    e.add_argument("--text")
    e.add_argument("--expected", default="0")
    e.add_argument("--answer-type", default="gsm8k_numeric", choices=["gsm8k_numeric", "int", "numeric", "exact_str", "exact_str_ci"])
    a = p.parse_args(argv)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "score":
        return cmd_score(a)
    if a.cmd == "extract":
        return cmd_extract(a)
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
