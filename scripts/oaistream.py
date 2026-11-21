"""Shared helpers for the benchmark scripts in this folder (Python 3.9+, standard library only).

Contents:
  * stream_chat(): one streaming request to an OpenAI-compatible /v1/chat/completions endpoint,
    timed on the client with the definitions used in BENCHMARKS.md:
        t0       the moment the request is sent (after the TCP connection is open)
        t_first  first streamed token of any kind (delta.reasoning_content, delta.reasoning or delta.content)
        t_answer first non-empty delta.content
        t_last   last streamed token
  * metrics(): n_out from the server's `usage` block (never from counting chunks), TTFT,
    time-to-answer, decode tok/s = (n_out - 1) / (t_last - t_first), E2E tok/s = n_out / (t_last - t0).
  * load_prompts(): reads every prompt file in receipts/prompts/ (withheld prompts have prompt == null).
  * small statistics helpers (median, the p95 rule used in our tables, bootstrap CI) and CSV writing.

Written from scratch for this repository; it contains no code from any engine.
"""
from __future__ import annotations

import csv
import hashlib
import http.client
import json
import math
import os
import random
import re
import socket
import statistics
import sys
import time
import urllib.parse

CHAT_PATH = "/v1/chat/completions"
WARMUP_FILE_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "receipts", "prompts", "r5-prompts.json")


class UsageError(Exception):
    """Bad input from the user (file, flag or server address)."""


# ---------------------------------------------------------------------------
# Addresses and connections
# ---------------------------------------------------------------------------

def parse_target(url: str) -> tuple:
    """Accepts http://HOST:PORT, http://HOST:PORT/v1 or the full .../v1/chat/completions URL.

    Returns (host, port, chat_path)."""
    if "://" not in url:
        url = "http://" + url
    u = urllib.parse.urlparse(url)
    if u.scheme != "http":
        raise UsageError(f"only plain http is supported (local servers), got {url!r}")
    if not u.hostname:
        raise UsageError(f"no host in {url!r}")
    port = u.port or 80
    path = u.path.rstrip("/")
    if path.endswith("/chat/completions"):
        chat = path
    elif path.endswith("/v1"):
        chat = path + "/chat/completions"
    elif path == "":
        chat = CHAT_PATH
    else:
        chat = path + CHAT_PATH
    return u.hostname, port, chat


def open_conn(host: str, port: int, timeout_s: float, retries: int = 40) -> http.client.HTTPConnection:
    """Open a TCP connection with TCP_NODELAY; retry briefly (servers with a small listen backlog drop bursts)."""
    last = None
    for _ in range(max(1, retries)):
        conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
        try:
            conn.connect()
            if conn.sock is not None:
                conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return conn
        except OSError as exc:
            last = exc
            try:
                conn.close()
            except Exception:
                pass
            time.sleep(0.01)
    raise last  # type: ignore[misc]


def check_server(host: str, port: int, timeout_s: float = 5.0) -> str:
    """GET /v1/models. Returns '' when reachable, otherwise an error string."""
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        return "" if resp.status == 200 else f"GET /v1/models returned HTTP {resp.status}"
    except OSError as exc:
        return f"cannot reach http://{host}:{port}/v1/models: {exc}"


# ---------------------------------------------------------------------------
# One streaming request
# ---------------------------------------------------------------------------

def build_body(model: str, prompt: str, max_tokens: int, temperature: float,
               reasoning_effort: str = None, seed=None, extra: dict = None) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if seed is not None:
        body["seed"] = int(seed)
    if extra:
        body.update(extra)
    return body


def _draft_from_event(evt: dict):
    """Some engines attach drafting counters to stream events. Returns (accepted, offered) or None."""
    for key, acc_k, off_k in (("tensorfold", "draft_accepted", "draft_offered"),
                              ("speculative", "accepted", "offered")):
        meta = evt.get(key)
        if isinstance(meta, dict) and (meta.get(acc_k) is not None or meta.get(off_k) is not None):
            return int(meta.get(acc_k) or 0), int(meta.get(off_k) or 0)
    return None


def stream_chat(host: str, port: int, path: str, body: dict, prefill_timeout_s: float = 120.0,
                decode_watchdog_s: float = 30.0, conn: http.client.HTTPConnection = None) -> dict:
    """Send one streaming chat request and time it. Never raises for server-side problems.

    Watchdogs: no first token within prefill_timeout_s, or no new token for decode_watchdog_s after
    the first one (SSE ': keepalive' comment lines do not reset the timer) -> hang=True."""
    res = {"error": None, "hang": False, "hang_kind": None, "http_status": None,
           "t0": None, "t0_wall": None, "t_first": None, "t_answer": None, "t_last": None, "t_done": None,
           "usage": None, "finish_reason": None, "chunks": 0, "keepalives": 0, "max_inter_tok_s": 0.0,
           "content": "", "reasoning": "", "draft": None}
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    try:
        if conn is None:
            conn = open_conn(host, port, prefill_timeout_s, retries=1)
        res["t0"] = time.perf_counter()
        res["t0_wall"] = time.time()
        conn.request("POST", path, body=payload, headers={"Content-Type": "application/json"})
        sock = conn.sock  # keep it: http.client drops conn.sock when the server says "Connection: close"
        resp = conn.getresponse()
    except (OSError, http.client.HTTPException) as exc:
        res["error"] = f"connect_or_headers: {type(exc).__name__}: {exc}"
        res["hang"] = isinstance(exc, socket.timeout)
        res["hang_kind"] = "connect_or_headers" if res["hang"] else None
        if conn is not None:
            conn.close()
        return res

    t0 = res["t0"]
    res["http_status"] = resp.status
    if resp.status != 200:
        try:
            text = resp.read().decode("utf-8", "replace")
        except Exception:
            text = ""
        res["error"] = f"HTTP {resp.status}: {text[:300]}"
        conn.close()
        return res

    reasoning_parts, content_parts = [], []
    last_tok = t0
    while True:
        now = time.perf_counter()
        if res["t_first"] is None:
            left = prefill_timeout_s - (now - t0)
            kind = "prefill_timeout"
        else:
            left = decode_watchdog_s - (now - last_tok)
            kind = "decode_watchdog"
        if left <= 0:
            res["hang"], res["hang_kind"] = True, kind
            res["error"] = f"{kind}>{prefill_timeout_s if kind == 'prefill_timeout' else decode_watchdog_s}s"
            break
        if sock is not None:
            try:
                sock.settimeout(max(0.25, left))
            except OSError:
                pass
        try:
            raw = resp.fp.readline()
        except (OSError, ValueError) as exc:
            if isinstance(exc, socket.timeout):
                # the socket timeout was set to the time left on the watchdog, so it has fired
                # (a timed-out socket file cannot be read again)
                res["hang"], res["hang_kind"] = True, kind
                res["error"] = f"{kind}>{prefill_timeout_s if kind == 'prefill_timeout' else decode_watchdog_s}s"
            else:
                res["error"] = f"read_error: {type(exc).__name__}: {exc}"
            break
        if not raw:
            break
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        if line.startswith(":"):
            res["keepalives"] += 1
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            evt = json.loads(data)
        except ValueError:
            res["error"] = f"bad SSE JSON: {data[:120]}"
            break
        if evt.get("usage"):
            res["usage"] = evt["usage"]
        d = _draft_from_event(evt)
        if d is not None:
            res["draft"] = d
        for ch in evt.get("choices") or []:
            if ch.get("finish_reason") is not None:
                res["finish_reason"] = ch["finish_reason"]
            delta = ch.get("delta") or {}
            r_txt = delta.get("reasoning_content") or delta.get("reasoning") or ""
            c_txt = delta.get("content") or ""
            if not (r_txt or c_txt):
                continue
            t = time.perf_counter()
            if res["t_first"] is None:
                res["t_first"] = t
            else:
                res["max_inter_tok_s"] = max(res["max_inter_tok_s"], t - last_tok)
            if c_txt and res["t_answer"] is None:
                res["t_answer"] = t
            last_tok = t
            res["chunks"] += 1
            if r_txt:
                reasoning_parts.append(r_txt)
            if c_txt:
                content_parts.append(c_txt)
    res["t_done"] = time.perf_counter()
    try:
        conn.close()
    except Exception:
        pass
    res["t_last"] = last_tok if res["t_first"] is not None else None
    res["reasoning"] = "".join(reasoning_parts)
    res["content"] = "".join(content_parts)
    if res["error"] is None and res["t_first"] is None:
        res["error"] = "stream ended without any token"
    return res


def metrics(res: dict) -> dict:
    """Per-request numbers exactly as defined in BENCHMARKS.md."""
    usage = res.get("usage") or {}
    n_in = int(usage.get("prompt_tokens") or 0)
    n_out = int(usage.get("completion_tokens") or 0)
    cached = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
    t0, tf, ta, tl = res.get("t0"), res.get("t_first"), res.get("t_answer"), res.get("t_last")
    ttft = (tf - t0) if (tf is not None and t0 is not None) else None
    tta = (ta - t0) if (ta is not None and t0 is not None) else None
    dec_dt = (tl - tf) if (tl is not None and tf is not None) else None
    e2e_dt = (tl - t0) if (tl is not None and t0 is not None) else None
    if n_out > 1 and dec_dt and dec_dt > 0:
        dec = (n_out - 1) / dec_dt
    elif n_out == 1 and dec_dt and dec_dt > 0:
        dec = 1.0 / dec_dt
    else:
        dec = 0.0
    e2e = (n_out / e2e_dt) if (n_out > 0 and e2e_dt and e2e_dt > 0) else 0.0
    acc, off = res.get("draft") or (0, 0)
    text = (res.get("reasoning") or "") + "\x00" + (res.get("content") or "")
    return {
        "n_in": n_in, "n_out": n_out, "cached_tokens": cached,
        "ttft_s": ttft, "time_to_answer_s": tta,
        "decode_tok_s": dec, "e2e_tok_s": e2e,
        "max_inter_tok_s": res.get("max_inter_tok_s"),
        "finish_reason": res.get("finish_reason"),
        "drafted": off > 0, "draft_accepted": acc, "draft_offered": off,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "content_sha256": hashlib.sha256((res.get("content") or "").encode("utf-8")).hexdigest(),
        "t0_wall": res.get("t0_wall"),
    }


# ---------------------------------------------------------------------------
# Server logs (optional drafting counters and the MTP-loaded line)
# ---------------------------------------------------------------------------

DONE_LINE_RE = re.compile(r"\bdone\b.*?\baccepted=(\d+)/(\d+)")


class LogTail:
    """Reads only what a server log gained since the last mark()."""

    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise UsageError(f"server log not found: {path}")
        self.path = path
        self.pos = os.path.getsize(path)

    def mark(self) -> None:
        self.pos = os.path.getsize(self.path)

    def new_text(self) -> str:
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read()
        self.pos += len(data)
        return data.decode("utf-8", "replace")


def drafts_from_log_text(text: str):
    """Sum 'done ... accepted=A/O' lines (the per-request line TensorFold 1.2.10 writes). None if absent."""
    acc = off = 0
    found = False
    for line in text.splitlines():
        m = DONE_LINE_RE.search(line)
        if m:
            found = True
            acc += int(m.group(1))
            off += int(m.group(2))
    return (acc, off) if found else None


MTP_LINES = ("and the MTP layer", "drafts: MTP")


def mtp_line_status(log_path: str) -> tuple:
    """Returns (ok, lines) for the TensorFold MTP-loaded evidence in a startup log."""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    hits = [ln.strip() for ln in text.splitlines() if any(m in ln for m in MTP_LINES)]
    ok = all(any(m in ln for ln in hits) for m in MTP_LINES)
    return ok, hits


# ---------------------------------------------------------------------------
# Prompt files
# ---------------------------------------------------------------------------

def load_prompts(path: str, set_name: str = None) -> list:
    """Normalise any prompt file in receipts/prompts/ to [{id, type, prompt, expected_tokens, withheld, raw}]."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        raise UsageError(f"cannot read prompt file {path}: {exc}")
    except ValueError as exc:
        raise UsageError(f"{path} is not valid JSON: {exc}")
    if isinstance(data, list):
        items = data
    elif set_name:
        if set_name not in data:
            sets = [k for k, v in data.items() if isinstance(v, (list, dict))]
            raise UsageError(f"set {set_name!r} not in {path}; available: {', '.join(sets)}")
        items = data[set_name]
        if isinstance(items, dict):
            items = [items]
    elif "prompts" in data:
        items = data["prompts"]
    elif "items" in data:
        items = data["items"]
    else:
        sets = [k for k, v in data.items() if isinstance(v, (list, dict))]
        raise UsageError(f"{path} holds several prompt sets; pick one with --set ({', '.join(sets)})")
    out = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise UsageError(f"{path}: item {i} is not an object")
        pid = it.get("id") or it.get("name") or it.get("user_id") or it.get("slice_id") or f"p{i:03d}"
        text = it["prompt"] if "prompt" in it else it.get("question")
        out.append({
            "id": pid,
            "type": it.get("prompt_type") or it.get("type") or it.get("kind") or it.get("category") or "",
            "prompt": text,
            "expected_tokens": it.get("expected_tokens_default_effort") or it.get("expected_tokens") or 0,
            "withheld": bool(it.get("withheld")) or text is None,
            "raw": it,
        })
    if not out:
        raise UsageError(f"no prompts found in {path}")
    return out


# ---------------------------------------------------------------------------
# Statistics and output
# ---------------------------------------------------------------------------

def p95_rule(values: list):
    """p95 as used in our tables: the sorted value at index ceil(0.95 * (n - 1))."""
    v = sorted(values)
    if not v:
        return None
    return v[min(len(v) - 1, max(0, math.ceil(0.95 * (len(v) - 1))))]


def median_or_none(values: list):
    v = [x for x in values if x is not None]
    return statistics.median(v) if v else None


def bootstrap_median_ci(values: list, n_boot: int = 10000, seed: int = 20261120) -> tuple:
    """95% percentile-bootstrap interval of the median (indices 2.5% and 97.5% of n_boot sorted medians)."""
    v = [x for x in values if x is not None]
    if len(v) < 2:
        return (None, None)
    rng = random.Random(seed)
    n = len(v)
    meds = sorted(statistics.median(v[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    return meds[int(0.025 * (n_boot - 1))], meds[int(0.975 * (n_boot - 1))]


def fmt(x, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_csv(path: str, rows: list, fieldnames: list, append: bool = False) -> None:
    """Write rows; with append=True add to an existing file (its header must match)."""
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    if append and exists:
        with open(path, "r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != fieldnames:
            raise UsageError(f"{path} has a different header; write to a new file")
    mode = "a" if (append and exists) else "w"
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})


def die(msg: str, code: int = 2) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def parse_extra_body(text: str) -> dict:
    if not text:
        return {}
    try:
        val = json.loads(text)
    except ValueError as exc:
        raise UsageError(f"--extra-body is not valid JSON: {exc}")
    if not isinstance(val, dict):
        raise UsageError("--extra-body must be a JSON object")
    return val
