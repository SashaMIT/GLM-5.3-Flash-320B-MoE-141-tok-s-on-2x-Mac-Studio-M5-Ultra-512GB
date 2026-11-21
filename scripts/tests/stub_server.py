#!/usr/bin/env python3
"""Tiny fake OpenAI-compatible server for testing the benchmark scripts without a model.

POST /v1/chat/completions with "stream": true answers with Server-Sent Events:
  [optional ': keepalive' comment] -> wait --ttft seconds -> --chunks chunks, --interval seconds apart,
  the first --reasoning-chunks of them as delta.reasoning_content and the rest as delta.content, each
  chunk standing for --tokens-per-chunk tokens -> a final chunk with finish_reason and a `usage` block
  (prompt_tokens, completion_tokens, prompt_tokens_details.cached_tokens) -> 'data: [DONE]'.
completion_tokens = min(max_tokens, chunks * tokens_per_chunk), so a client that counts chunks
instead of reading `usage` gets the wrong n_out.
The text depends only on the prompt, so repeated prompts give identical outputs.
With --gsm8k FILE the reply to a known question ends with "ANSWER: <reference answer>".
GET /v1/models returns one model. Binds to 127.0.0.1 only.

  python3 scripts/tests/stub_server.py --port 18080 --ttft 0.05 --interval 0.005
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULTS = dict(ttft=0.05, interval=0.005, chunks=20, tokens_per_chunk=2, reasoning_chunks=4, keepalive=True,
                cached=0, stall_after=0, stall_s=0.0, draft=False, gsm8k=None)


class StubState:
    def __init__(self, **cfg):
        self.cfg = dict(DEFAULTS)
        self.cfg.update(cfg)
        self.answers = {}
        if self.cfg.get("gsm8k"):
            with open(self.cfg["gsm8k"], encoding="utf-8") as f:
                for it in json.load(f)["items"]:
                    self.answers[it["question"]] = str(it["answer"])
        self.lock = threading.Lock()
        self.first_lines = []  # first line of every prompt received (to check unique prefixes)
        self.requests = 0


def make_handler(state: StubState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep test output quiet
            pass

        def _json(self, code: int, obj: dict):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": "default_model", "object": "model"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": "not found"})
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n))
                prompt = body["messages"][-1]["content"]
            except (ValueError, KeyError, IndexError):
                self._json(400, {"error": "bad request"})
                return
            c = state.cfg
            with state.lock:
                state.requests += 1
                state.first_lines.append(prompt.split("\n", 1)[0])
            max_tokens = int(body.get("max_tokens") or 10 ** 9)
            n_out = min(max_tokens, c["chunks"] * c["tokens_per_chunk"])
            n_chunks = max(1, -(-n_out // c["tokens_per_chunk"]))
            seed = hashlib.sha256(prompt.encode()).hexdigest()
            answer = None
            for q, ans in state.answers.items():
                if q in prompt:
                    answer = ans
                    break
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def send(obj):
                self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                self.wfile.flush()

            try:
                if c["keepalive"]:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                time.sleep(c["ttft"])
                for i in range(n_chunks):
                    if i > 0:
                        time.sleep(c["interval"])
                    if c["stall_after"] and i == c["stall_after"]:
                        time.sleep(c["stall_s"])
                    word = f"w{seed[i % 60:(i % 60) + 4]} "
                    if i == n_chunks - 1 and answer is not None:
                        word = f"\nSo the total is {answer}.\nANSWER: {answer}"
                    key = "reasoning_content" if i < c["reasoning_chunks"] else "content"
                    if answer is not None and i == 0 and key == "reasoning_content":
                        word = "ANSWER: 0\n"  # decoy in the reasoning: scorers must read only the content
                    send({"id": "stub", "object": "chat.completion.chunk",
                          "choices": [{"index": 0, "delta": {key: word}, "finish_reason": None}]})
                final = {"id": "stub", "object": "chat.completion.chunk",
                         "choices": [{"index": 0, "delta": {}, "finish_reason": "length" if n_out >= max_tokens else "stop"}],
                         "usage": {"prompt_tokens": max(1, len(prompt) // 4), "completion_tokens": n_out,
                                   "total_tokens": max(1, len(prompt) // 4) + n_out,
                                   "prompt_tokens_details": {"cached_tokens": c["cached"]}}}
                if c["draft"]:
                    final["speculative"] = {"accepted": n_out // 2, "offered": n_out}
                send(final)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


def serve(port: int = 0, **cfg):
    """Start in a background thread. Returns (server, state); server.server_address[1] is the port."""
    state = StubState(**cfg)
    srv = _Server(("127.0.0.1", port), make_handler(state))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=0, help="0 = pick a free port and print it")
    p.add_argument("--ttft", type=float, default=DEFAULTS["ttft"])
    p.add_argument("--interval", type=float, default=DEFAULTS["interval"])
    p.add_argument("--chunks", type=int, default=DEFAULTS["chunks"])
    p.add_argument("--tokens-per-chunk", type=int, default=DEFAULTS["tokens_per_chunk"])
    p.add_argument("--reasoning-chunks", type=int, default=DEFAULTS["reasoning_chunks"])
    p.add_argument("--no-keepalive", action="store_true")
    p.add_argument("--cached", type=int, default=0, help="cached_tokens to report")
    p.add_argument("--stall-after", type=int, default=0, help="stall before this chunk index (0 = never)")
    p.add_argument("--stall-s", type=float, default=0.0)
    p.add_argument("--draft", action="store_true", help="attach speculative accepted/offered counters")
    p.add_argument("--gsm8k", help="r6-gsm8k.json: answer known questions correctly")
    a = p.parse_args(argv)
    srv, _ = serve(a.port, ttft=a.ttft, interval=a.interval, chunks=a.chunks, tokens_per_chunk=a.tokens_per_chunk,
                   reasoning_chunks=a.reasoning_chunks, keepalive=not a.no_keepalive, cached=a.cached,
                   stall_after=a.stall_after, stall_s=a.stall_s, draft=a.draft, gsm8k=a.gsm8k)
    print(f"stub listening on http://127.0.0.1:{srv.server_address[1]}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
