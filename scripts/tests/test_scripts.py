"""End-to-end tests of the benchmark scripts against the stub server (standard library only).

Run from the repository root:  python3 -m unittest discover -s scripts/tests -v
"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
PROMPTS = os.path.join(ROOT, "receipts", "prompts")
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)

import stub_server  # noqa: E402
import bench_single  # noqa: E402
import bench_many  # noqa: E402
import bench_fairness  # noqa: E402
import gsm8k_score  # noqa: E402
import compare  # noqa: E402
import oaistream as oa  # noqa: E402


def run(fn, args):
    """Call a script's main() and capture stdout; SystemExit codes are returned too."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = fn(args)
    except SystemExit as exc:
        rc = exc.code
    return rc, buf.getvalue()


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class StubCase(unittest.TestCase):
    cfg = {}

    def setUp(self):
        self.srv, self.state = stub_server.serve(0, **self.cfg)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.tmp.cleanup()

    def out(self, name):
        return os.path.join(self.tmp.name, name)


class TestSingleTiming(StubCase):
    # 10 chunks x 2 tokens: TTFT 0.20 s, then a chunk every 0.02 s; the first 4 chunks are reasoning
    cfg = dict(ttft=0.20, interval=0.02, chunks=10, tokens_per_chunk=2, reasoning_chunks=4)

    def test_metrics_and_columns(self):
        out = self.out("single.csv")
        rc, text = run(bench_single.main, ["--target", self.url, "--prompts", os.path.join(PROMPTS, "r6-real-public.json"),
                                           "--ids", "real-code-01,real-plan-01", "--repeats", "2", "--warmups", "1",
                                           "--arm", "T1", "--out", out])
        self.assertEqual(rc, 0, text)
        rows = read(out)
        with open(os.path.join(ROOT, "receipts", "r7-part1-requests.csv"), newline="") as f:
            self.assertEqual(next(csv.reader(f)), list(rows[0].keys()))
        self.assertEqual(len(rows), 4)
        for r in rows:
            self.assertEqual(r["block_status"], "ok")
            self.assertEqual(int(r["n_out"]), 20, "n_out must come from usage, not from the 10 chunks")
            self.assertEqual(int(r["cached_tokens"]), 0)
            ttft, tta = float(r["ttft_s"]), float(r["time_to_answer_s"])
            self.assertTrue(0.18 <= ttft <= 0.45, ttft)
            self.assertTrue(0.06 <= tta - ttft <= 0.20, tta - ttft)  # 4 reasoning chunks of 0.02 s
            dec, e2e = float(r["decode_tok_s"]), float(r["e2e_tok_s"])
            self.assertTrue(55 <= dec <= 110, dec)   # 19 / ~0.18 s = ~105 tok/s
            self.assertTrue(30 <= e2e <= 56, e2e)    # 20 / ~0.38 s = ~52 tok/s
            self.assertEqual(r["finish_reason"], "stop")
        self.assertEqual(self.state.requests, 1 + 4)

    def test_withheld_prompts_skipped(self):
        out = self.out("all.csv")
        rc, text = run(bench_single.main, ["--target", self.url, "--prompts", os.path.join(PROMPTS, "r6-real-public.json"),
                                           "--repeats", "1", "--warmups", "0", "--out", out])
        self.assertEqual(rc, 0, text)
        self.assertIn("skipping 8 withheld prompt(s)", text)
        self.assertEqual(len(read(out)), 16)


class TestCachedStops(StubCase):
    cfg = dict(ttft=0.01, interval=0.001, cached=7)

    def test_cached_tokens_stop_block(self):
        out = self.out("c.csv")
        rc, _ = run(bench_single.main, ["--target", self.url, "--prompts", os.path.join(PROMPTS, "r5-prompts.json"),
                                        "--set", "short_32", "--warmups", "0", "--out", out])
        self.assertEqual(rc, 1)
        rows = read(out)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["block_status"], "error")


class TestWatchdog(StubCase):
    cfg = dict(ttft=0.01, interval=0.001, chunks=6, stall_after=3, stall_s=2.0)

    def test_decode_watchdog(self):
        out = self.out("w.csv")
        rc, _ = run(bench_single.main, ["--target", self.url, "--prompts", os.path.join(PROMPTS, "r5-prompts.json"),
                                        "--set", "short_32", "--warmups", "0", "--decode-watchdog", "0.5", "--out", out])
        self.assertEqual(rc, 1)
        rows = read(out)
        self.assertEqual(rows[-1]["hang"], "True")
        self.assertIn("decode_watchdog", rows[-1]["error"])
        self.assertEqual(rows[-1]["block_status"], "stopped_watchdog")


class TestPrefillTimeout(StubCase):
    cfg = dict(ttft=2.0, keepalive=True)

    def test_first_token_limit(self):
        out = self.out("p.csv")
        rc, _ = run(bench_single.main, ["--target", self.url, "--prompts", os.path.join(PROMPTS, "r5-prompts.json"),
                                        "--set", "short_32", "--warmups", "0", "--prefill-timeout", "0.5", "--out", out])
        self.assertEqual(rc, 1)
        self.assertIn("prefill_timeout", read(out)[-1]["error"])


class TestMany(StubCase):
    cfg = dict(ttft=0.05, interval=0.005, chunks=16, tokens_per_chunk=2, draft=True)

    def test_ladder_two_targets(self):
        srv2, state2 = stub_server.serve(0, **self.cfg)
        try:
            url2 = f"http://127.0.0.1:{srv2.server_address[1]}"
            out, summ = self.out("m.csv"), self.out("m-sum.csv")
            rc, text = run(bench_many.main, ["--target", self.url, "--target", url2, "--users", "1,4,8", "--repeats", "2",
                                             "--system", "S3", "--out", out, "--summary-out", summ])
            self.assertEqual(rc, 0, text)
            rows, srows = read(out), read(summ)
            with open(os.path.join(ROOT, "receipts", "r7-part2-requests.csv"), newline="") as f:
                self.assertEqual(next(csv.reader(f)), list(rows[0].keys()))
            self.assertEqual(len(rows), 2 * (1 + 4 + 8))
            self.assertEqual(len(srows), 6)
            self.assertTrue(all(not r["error"] and int(r["n_out"]) == 32 for r in rows))
            self.assertEqual(set(r["target_tag"] for r in rows if r["n_users"] == "8"),
                             {f"127.0.0.1:{self.srv.server_address[1]}", f"127.0.0.1:{srv2.server_address[1]}"})
            for s in srows:
                agg, med = float(s["agg_e2e_tok_s"]), float(s["per_user_e2e_med"])
                n = int(s["users"])
                self.assertGreater(agg, 0)
                # total uses the union span, so it is at most n x the slowest user's rate and never total / n
                self.assertLessEqual(float(s["per_user_e2e_min"]), med)
                self.assertTrue(agg <= n * med * 1.5)
                self.assertEqual(s["drafted"], "True")
            # every timed request had a unique first line (no shared prompt cache)
            firsts = [x for x in self.state.first_lines + state2.first_lines if not x.startswith("[warmup")]
            self.assertEqual(len(firsts), len(set(firsts)))
        finally:
            srv2.shutdown()
            srv2.server_close()


class TestFairness(StubCase):
    cfg = dict(ttft=0.05, interval=0.002, chunks=12)

    def test_long_plus_eight(self):
        out = self.out("f.csv")
        rc, text = run(bench_fairness.main, ["--target", self.url, "--repeats", "1", "--arm", "tf_default",
                                             "--engine", "tensorfold", "--out", out])
        self.assertEqual(rc, 0, text)
        rows = read(out)
        with open(os.path.join(ROOT, "receipts", "r7-part3-requests.csv"), newline="") as f:
            self.assertEqual(next(csv.reader(f)), list(rows[0].keys()))
        self.assertEqual(len(rows), 9)
        self.assertEqual(sum(r["role"] == "long-32k" for r in rows), 1)
        self.assertTrue(all(r["solo_text_match"] == "True" for r in rows if r["role"] == "short"))
        self.assertIn("solo match 8/8", text)


class TestGsm8k(StubCase):
    cfg = dict(ttft=0.01, interval=0.001, chunks=20, gsm8k=os.path.join(PROMPTS, "r6-gsm8k.json"))

    def test_run_and_score(self):
        out = self.out("g.csv")
        rc, text = run(gsm8k_score.main, ["run", "--target", self.url, "--first", "3", "--samples", "5",
                                          "--warmups", "0", "--arm", "on", "--out", out])
        self.assertEqual(rc, 0, text)
        rows = read(out)
        self.assertEqual(len(rows), 15)
        self.assertTrue(all(r["correct"] == "True" for r in rows), "decoy ANSWER in the reasoning must be ignored")
        self.assertIn("majority vote 3/3", text)

    def test_receipt_numbers(self):
        rc, text = run(gsm8k_score.main, ["score", os.path.join(ROOT, "receipts", "r7-part5-requests.csv"),
                                          "--arm-a", "off", "--arm-b", "on"])
        self.assertEqual(rc, 0)
        self.assertIn("arm off: majority vote 100/100 (100.0%), samples 494/500 (98.8%), mean n_out 314.3", text)
        self.assertIn("arm on: majority vote 100/100 (100.0%), samples 496/500 (99.2%), mean n_out 297.4", text)
        self.assertIn("exact McNemar p = 1.0000", text)


class TestRules(unittest.TestCase):
    def test_answer_rule(self):
        s = gsm8k_score.score
        self.assertEqual(s("work\nANSWER: 18", "18")["correct"], True)
        self.assertEqual(s("**ANSWER: $1,200.**", "1200")["canonical"], "1200")
        self.assertEqual(s("ANSWER: `42`\nThanks!", "42")["where"], "earlier_line")
        self.assertEqual(s("Answer: 42", "42")["correct"], False)   # upper-case ANSWER only
        self.assertEqual(s("ANSWER: 7.50", "7.5")["canonical"], "7.5")
        self.assertEqual(s("ANSWER: -0", "0")["canonical"], "0")
        self.assertEqual(s("", "1")["canonical"], None)
        self.assertEqual(s("ANSWER: Blue", "blue", "exact_str_ci")["correct"], True)
        self.assertEqual(s("ANSWER: Blue", "blue", "exact_str")["correct"], False)
        self.assertTrue(gsm8k_score.majority([(0, "5", False), (1, "6", True), (2, "6", True), (3, None, False)]))
        self.assertFalse(gsm8k_score.majority([(0, "5", False), (1, "6", True)]))  # tie -> first seen ("5")

    def test_mcnemar(self):
        self.assertEqual(gsm8k_score.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(gsm8k_score.mcnemar_exact(0, 4), 0.125)   # value quoted in BENCHMARKS 3.1
        self.assertAlmostEqual(gsm8k_score.mcnemar_exact(1, 6), 0.125)

    def test_p95_rule(self):
        self.assertEqual(oa.p95_rule(list(range(1, 9))), 8)
        self.assertEqual(oa.p95_rule(list(range(1, 21))), 20)
        self.assertEqual(oa.p95_rule([3.0]), 3.0)


class TestCompare(StubCase):
    cfg = dict(ttft=0.02, interval=0.002)

    def test_subset_and_exit_codes(self):
        fresh = self.out("fresh.csv")
        rc, _ = run(bench_single.main, ["--target", self.url, "--prompts", os.path.join(PROMPTS, "r6-real-public.json"),
                                        "--repeats", "1", "--warmups", "0", "--arm", "T1", "--out", fresh])
        self.assertEqual(rc, 0)
        receipt = os.path.join(ROOT, "receipts", "r7-part1-requests.csv")
        rc, text = run(compare.main, ["--fresh", fresh, "--receipt", receipt, "--n-boot", "300"])
        self.assertEqual(rc, 1)  # a stub is not GLM-5.3-Flash
        self.assertIn("16 prompt id(s) in common", text)
        self.assertIn("79.57", text)  # receipt T1 median E2E over the 16 published prompts
        rc, text = run(compare.main, ["--fresh", receipt, "--receipt", receipt, "--n-boot", "300"])
        self.assertEqual(rc, 0, text)


class TestHelp(unittest.TestCase):
    def test_every_script_has_help(self):
        for name in ("bench_single.py", "bench_many.py", "bench_fairness.py", "gsm8k_score.py", "sync_bench.py",
                     "compare.py", "verify_model.py", os.path.join("tests", "stub_server.py")):
            cp = subprocess.run([sys.executable, os.path.join(SCRIPTS, name), "--help"], capture_output=True, text=True)
            self.assertEqual(cp.returncode, 0, name + cp.stderr)
            self.assertIn("usage:", cp.stdout)

    def test_bad_input_is_a_clear_error(self):
        cp = subprocess.run([sys.executable, os.path.join(SCRIPTS, "bench_single.py"), "--target", "http://127.0.0.1:9",
                             "--prompts", os.path.join(PROMPTS, "r5-prompts.json"), "--out", os.devnull],
                            capture_output=True, text=True)
        self.assertEqual(cp.returncode, 2)
        self.assertIn("pick one with --set", cp.stderr)


class TestVerifyModel(unittest.TestCase):
    def test_manifest_logic(self):
        import hashlib
        import verify_model
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.safetensors"), "wb") as f:
                f.write(b"x" * 100)
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"text_config": {"num_hidden_layers": 45, "num_nextn_predict_layers": 1},
                           "quantization": {"bits": 4, "group_size": 64, "mode": "affine"}}, f)
            man = os.path.join(d, "m.tsv")
            with open(man, "w") as f:
                f.write(f"r\trev\ta.safetensors\t100\tdir\t{hashlib.sha256(b'x' * 100).hexdigest()}\n")
                f.write(f"r\trev\tconfig.json\t{os.path.getsize(os.path.join(d, 'config.json'))}\tdir\t\n")
            rc, text = run(verify_model.main, ["--model-dir", d, "--manifest", man])
            self.assertIn("summary: ok=1 oksize=1 bad=0 rows=2", text)
            self.assertIn("config num_hidden_layers = 45 OK", text)
            self.assertEqual(rc, 1)  # the pinned SHA-256s of the real pack are absent here -> FAIL


if __name__ == "__main__":
    unittest.main()
