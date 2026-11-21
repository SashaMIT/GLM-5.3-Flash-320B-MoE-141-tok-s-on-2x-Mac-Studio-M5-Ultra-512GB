#!/usr/bin/env python3
"""Minimal clean-room benchmark client for OpenAI-compatible streaming servers.

Measures one request (or N users in parallel) with the definitions used in BENCHMARKS.md:
  n_out        completion tokens from the server's `usage` field (never count stream chunks)
  TTFT         t_first - t0, first streamed token of any kind (reasoning or answer)
  decode tok/s (n_out - 1) / (t_last - t_first)
  E2E tok/s    n_out / (t_last - t0)
Many users: total E2E = sum(n_out) / (latest t_last - earliest t0).

Standard library only. Example:
  python3 bench_client.py --url http://127.0.0.1:8080/v1/chat/completions \
      --model MODEL --prompt "Explain HTTP caching." --max-tokens 256 --users 1
"""
import argparse, json, statistics, threading, time, urllib.request, urllib.parse


def one_request(url, model, prompt, max_tokens, temperature):
    body = json.dumps({
        "model": model, "stream": True, "max_tokens": max_tokens, "temperature": temperature,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = t_answer = t_last = None
    n_out = None
    cached = None
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                evt = json.loads(data)
                usage = evt.get("usage")
                if usage:
                    n_out = usage.get("completion_tokens")
                    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                for ch in evt.get("choices") or []:
                    d = ch.get("delta") or {}
                    now = time.perf_counter()
                    if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                        if t_first is None:
                            t_first = now
                        t_last = now
                    if d.get("content") and t_answer is None:
                        t_answer = now
    except Exception as exc:  # record, do not hide
        return {"error": repr(exc), "t0": t0}
    if t_first is None or not n_out:
        return {"error": "no tokens or no usage in stream", "t0": t0}
    return {
        "t0": t0, "t_last": t_last, "n_out": n_out, "cached_tokens": cached,
        "ttft_s": t_first - t0,
        "time_to_answer_s": (t_answer - t0) if t_answer else None,
        "decode_tok_s": (n_out - 1) / (t_last - t_first) if t_last > t_first else None,
        "e2e_tok_s": n_out / (t_last - t0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="full chat-completions URL (keep it on 127.0.0.1 or a trusted host)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--users", type=int, default=1, help="parallel streams")
    ap.add_argument("--out", help="write per-request results as JSON")
    a = ap.parse_args()
    if urllib.parse.urlparse(a.url).scheme not in ("http", "https"):
        ap.error("--url must be http or https")

    results = [None] * a.users
    def work(i):
        results[i] = one_request(a.url, a.model, a.prompt, a.max_tokens, a.temperature)
    threads = [threading.Thread(target=work, args=(i,)) for i in range(a.users)]
    for t in threads: t.start()
    for t in threads: t.join()

    ok = [r for r in results if "error" not in r]
    for i, r in enumerate(results):
        print(f"user {i}: " + (r["error"] if "error" in r else
              f"n_out={r['n_out']} cached={r['cached_tokens']} ttft={r['ttft_s']:.2f}s "
              f"decode={r['decode_tok_s']:.1f} e2e={r['e2e_tok_s']:.1f} tok/s"))
    if ok:
        total = sum(r["n_out"] for r in ok) / (max(r["t_last"] for r in ok) - min(r["t0"] for r in ok))
        per_user = [r["e2e_tok_s"] for r in ok]
        print(f"ok={len(ok)}/{a.users} total_e2e={total:.1f} tok/s "
              f"per-user e2e median={statistics.median(per_user):.1f} min={min(per_user):.1f}")
        print("reminder: cached_tokens must be 0 on every timed request; do 3 untimed warm-ups first.")
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
