# Lessons Learned (Failures, Traps, and Fixes)

Every issue below happened directly during our 19 and 20 Nov 2026 runs on **Mac A** and **Mac B** (`Mac17,15`, M5 Ultra 512 GB, macOS 27.2 build `26B118`, stock `mlx 0.32.4`). Each entry states **what happened**, **why**, and **what to do**.

---

## 1. Fast sync (`MLX_METAL_FAST_SYNCH=1`) stalls on 8-bit and MXFP8 graphs

- **What happened:**
  - On the 4-bit MTP pack (`TensorFold/GLM-5.3-Flash-MLX-4bit-MTP`), `MLX_METAL_FAST_SYNCH=1` ran cleanly across every single-stream run, a 12,000-step `sync_bench` microbenchmark soak, a 30-minute / 309-turn soak (`0` stalls), and 32-user concurrency.
  - On the FP8 checkpoint requantized to MXFP8 (`dealignai/GLM-5.3-Flash-UNCENSORED-FP8` via `mlx-vlm`), enabling `MLX_METAL_FAST_SYNCH=1` loaded the model in 36.8 s and then **froze on prompt 1 with zero tokens emitted** (`twomac-fp8-fastsync-stall.log`).
  - Later, when testing an 8-bit `oQ8-MTP` conversion of the official weights, retry loops attempting `MLX_METAL_FAST_SYNCH=1` were caught and killed by our runtime guard (`exit 143`).
- **Why:**
  - Process stack sampling (`sample`) at the MXFP8 stall showed Mac B blocked inside `mlx::core::Fence::wait` while Mac A blocked in `jaccl::MeshImpl::all_reduce` waiting for Mac B. Under MLX 0.32.4 on this hardware (`mlx` issue class `#3142`), fast sync's GPU fence never signals across the RDMA collective on 8-bit/MXFP8 graphs.
- **What to do:**
  - **Never set `MLX_METAL_FAST_SYNCH=1` (or `MLX_FENCE_MODE=1`) on FP8, MXFP8, or any 8-bit graph over `jaccl`.**
  - Add an automatic pre-launch guard that refuses to start any fast-sync process when the checkpoint precision is FP8, MXFP8, or 8-bit (`oQ8`).

---

## 2. Two Macs on MXFP8 (default sync) reach less than one Mac

- **What happened:**
  - With default sync (`MLX_METAL_FAST_SYNCH` unset), two Macs running FP8-requantized-to-MXFP8 generated at **`21.67–21.78 tok/s`** (`twomac-quiet-1119.log`, `178.7 GB` MLX peak per Mac).
  - Running the **exact same FP8->MXFP8 build on one Mac (`Mac B`)** generated at **`26.3–26.5 tok/s`** (`334.9 GB` peak). Two Macs were slower than one Mac (`21.67–21.78` vs `26.3–26.5 tok/s`).
  - Even on 4-bit, disabling fast sync dropped two Macs from `49.8 tok/s` to **`27.3–27.4 tok/s`** (also below one Mac's `43.0 tok/s`).
- **Why:**
  - `glm5_next` executes ~90+ `all_sum` collectives per token across 45 layers. Under the default fence, each `all_sum` adds **`188–197 us`** (`~17 ms/tok` of pure sync overhead), versus **`18–19 us`** with fast sync (`sync-quiet-2026-11-19.log`).
  - Two-Mac tensor parallelism only gives more than one Mac if fast sync cuts that fence overhead by ~10x. Because fast sync deadlocks on MXFP8 (Lesson 1), splitting MXFP8 pays the ~17 ms/tok penalty without recovering it through parallelism.
- **What to do:**
  - If you need MXFP8 on 512 GB Studios and it fits below your RAM stop line (`334.9 GB < 338 GB`), run it on **one Mac**, not two.
  - For two-Mac speedups, use the **4-bit MTP pack** where `MLX_METAL_FAST_SYNCH=1` is safe and hang-free.

---

## 3. MTP drafting must be confirmed loaded (and verified per step)

- **What happened:**
  - Three different failure modes disguised whether speculative Multi-Token Prediction (MTP) was actually running:
    1. **`mlx-vlm 0.8.0` ignored MTP silently:** Loading `...-4bit-MTP` succeeded without warnings, yet ran plain decode at `43–50 tok/s`.
    2. **`GLM-Tandem` printed a misleading zero-tensor line:** `tpserve.py` logged `[r0] MTP head: loaded 0 tensors`, even though `oMLX`'s sanitizer had already bound the checkpoint's `layers.45.*` weights (`bound 1 nextn MTP layer(s) as mtp.*`) and MTP actually ran at `74–88%` acceptance.
    3. **`TensorFold --parallel 16` silently turned MTP off mid-run:** Even though the startup banner still said `drafts: MTP + copies`, the server logs showed `accepted=0/0` on all 16 concurrent streams (`result-r5-armC.json`).
  - Community issue `#3902` in TensorFold also documents cases on macOS `26B118` where MTP is left unloaded or stalls on the first decode.
- **Why:**
  - Different packs store MTP weights under different key prefixes (`model.language_model.layers.45.*` vs `mtp.*`), some engines do not implement MTP at all, and multi-stream schedulers can drop speculative drafting above a stream count threshold.
- **What to do:**
  - Check **both** startup logs (`45 of 45 layers and the MTP layer` + `drafts: MTP` in TensorFold; `bound 1 nextn MTP layer(s)` or `MTP head: loaded 59 tensors` in Tandem) **and** per-request telemetry (`acceptance_rate` / `[tp-mtp] head acc/off` / `accepted=0/0`).
  - If acceptance metrics are missing or `0/0`, label that run **`plain decode`** and never compare it as speculative decoding.
  - Run at least 5 repetitions (reporting median and max) to catch any first-decode `#3902` stall.

---

## 4. Busy Macs skew results badly

- **What happened:**
  - During an early two-Mac FP8->MXFP8 run (`result-fp8-r2.json`), `Mac A` started an unrelated CPU-heavy software test suite mid-run. Write speed dropped from `21.5 tok/s` to **`14.78 tok/s`**, and prompt read dropped from `171` to **`122 tok/s`**. Re-running once both Macs were quiet restored steady `21.67–21.78 tok/s` (`twomac-quiet-1119.log`).
- **Why:**
  - Tensor-parallel decoding synchronizes both Macs ~90 times per token. CPU scheduling delays, memory bandwidth contention, or background GPU/system activity on **either** Mac A or Mac B stalls the entire cluster at every collective barrier.
- **What to do:**
  - Acquire exclusive locks for both Macs before benchmarking, verify zero competing workloads (`pgrep`) and idle background orchestrators (e.g. `exo` state `0 0 0` on `127.0.0.1:52415/state`), and discard/repeat any run interrupted by background jobs.

---

## 5. Draft decoding is not word-for-word identical at `T=0` — report "same quality"

- **What happened:**
  - In Round 4 Gate 1 (`result-r4-gate1-identity.json`), running `GLM-Tandem` at `temperature=0` was 100% deterministic within each mode (`stream=True == stream=False`, `6/6` prompts). However, comparing `TP_MTP=0` (drafting OFF) against `TP_MTP=1` (drafting ON) diverged on **5 of 6 prompts** after `15–244` identical tokens (e.g. `"Use 8000 GB."` vs `"Use 8000."`, `"Gullwrack"` vs `"Gullwing"`).
  - Even with drafting completely OFF (`TP_MTP=0`), `GLM-Tandem` differed from `mlx-vlm 0.8.0 + overlay` on **3 of 3 prompts** after `14–75` tokens (`result-r4b-gate2-abccba.json`).
- **Why:**
  - Verifying multiple speculative draft tokens (`R >= 2` rows) in parallel—and using different Metal/NAX matmul/attention kernels or sharded `lm_head` reductions—computes identical mathematics in a different **`bfloat16` accumulation order**. At near-tie logit pairs (1 ULP apart), `bf16` rounding likely flips which token wins `argmax` (inferred from the Round 6/7 reports; not logged per token).
- **What to do:**
  - Never claim speculative decoding or cross-engine acceleration is "token-identical" or "bit-exact" at `T=0` on Apple Silicon `bf16` without verifying token hashes.
  - Verify internal determinism within each mode (`1` unique hash across repetitions, `stream == non-stream`), read outputs for factual/semantic parity, and report **`same quality, not bit-identical at T=0`**.

---

## 6. `listen(5)` socket backlog drops simultaneous connection bursts

- **What happened:**
  - Firing `8`, `16`, or `32` simultaneous client requests behind a `threading.Barrier` against `GLM-Tandem` (`mlx_lm.server`) immediately failed `2/8` (`u8`), `10/16` (`u16`), and `24/32` (`u32`) sockets with `ECONNRESET`, `EPIPE`, and `EINVAL`—even with `TP_CONC=32` set (`result-r5-armA.json`).
  - Under the exact same barrier test, `mlx-vlm 0.8.0` (`uvicorn`) and `TensorFold` handled `16` and `32` simultaneous bursts with **`0` errors**.
  - Round 7 repeated it on 20 Nov: a raw burst of `32` users against the `GLM-Tandem` pair completed `9` and dropped `23` (`s1-raw-burst-u32`), while dual `mlx-vlm` and dual `TensorFold` completed `32/32`.
- **Why:**
  - `mlx_lm.server` uses Python's standard-library `ThreadingHTTPServer`, whose default `request_queue_size` is **`5`** (`listen(5)`). When 8–32 TCP `SYN` packets land in the same millisecond, the macOS kernel drops connections beyond the backlog before Python's accept thread can drain them.
- **What to do:**
  - For benchmarking unmodified `mlx_lm.server` / `GLM-Tandem` at `N >= 8` concurrent users, either pre-connect the TCP sockets with a `5 ms` stagger + retry loop *before* triggering the request barrier (`result-r5-armA-preconn.json`, `0` errors at `32` users), or document the raw-burst `listen(5)` failure alongside it.
  - For production multi-user serving, place an async proxy / larger-backlog frontend in front of `mlx_lm.server`, or run independent `mlx-vlm` (`uvicorn`) servers per Mac.

---

## 7. `verify-manifest` pitfalls (LFS vs non-LFS rows, truncated manifests, and loadability)

- **What happened:**
  - In our Hugging Face manifests (`4-bit`: `54` rows; `FP8`: `74` rows), only LFS weights/large files have SHA-256 checksums (`45` in 4-bit; `63` in FP8). The remaining files (`9` in 4-bit; `11` in FP8—including `config.json`, `chat_template.jinja`, and in the official FP8 repo even `model.safetensors.index.json`) have an empty SHA-256 field and are checked only by byte size (`oksize=9` or `oksize=11`).
  - Meanwhile, passing manifest verification (`63 sha256 + 11 size, bad=0`) did **not** guarantee `GLM-Tandem` could load the official `zai-org` FP8 checkpoint directly: rank 1 crashed on `'Linear' object has no attribute 'scales'`, requiring an `oQ8-MTP` conversion into a new folder (`zai-org__GLM-5.3-Flash-oQ8-MTP`).
- **Why:**
  - Two common verifier bugs miss corruption: (a) asserting `ok == total_rows` rejects valid non-LFS JSON/config files, while (b) asserting only `bad == 0` silently passes a truncated/empty manifest or same-byte-length JSON corruption.
- **What to do:**
  - Enforce **all three** checks in preflight:
    1. `ok + oksize == expected_rows` (`45 + 9 = 54` for 4-bit; `63 + 11 = 74` for FP8) and `bad == 0`.
    2. Compute and pin SHA-256s explicitly for `config.json`, `model.safetensors.index.json`, and `tokenizer.json`, plus parse `config.json` to assert `num_hidden_layers == 45`, `num_nextn_predict_layers == 1` (for MTP packs), and expected `quantization`.
    3. Always write converted checkpoints (`oQ8-MTP`) to a **new directory**; never overwrite verified source weights.

---

## 8. Run one benchmark at a time (plus prompt-cache & persistent-service isolation)

- **What happened:**
  - One Mac held a persistent single-Mac `mlx_vlm.server` on port `8090` (~180 GB RAM). Leaving it alive during a two-Mac run risks OOM/contention, and reusing a warm server across benchmark repetitions activates prefix/LRU caching (`cached_tokens > 0`), inflating `read tok/s` by 10x–20x (e.g., `code-2k` jumped from `1,866.5 tok/s` cold to `34,205.5 tok/s` on rep 2).
  - Running `A` vs `B` in different hours or days also introduces thermal, OS, or background drift.
- **Why:**
  - Prefix cache hits skip prefill computation, and concurrent standing servers steal unified RAM and GPU queues.
- **What to do:**
  - Stop any standing inference service before benchmarking, verify ports are free, and restart the standing service only in post-run cleanup.
  - Launch a **fresh server per benchmark step/arm** (or send disjoint warmup topics) and record `cached_tokens` on every request so cold `cached_tokens == 0` prefill is never mixed with cache-hit TTFT.
  - Use interleaved **`A-B-B-A`** / **`A-B-C-C-B-A`** ordering on the same day on quiet Macs for engine comparisons.

---

## 9. Additional Operational Pitfalls Fixed Along the Way

- **Keep Thunderbolt Bridge (`bridge0`) OFF:** macOS's default Thunderbolt Bridge can create routing loops or degrade RDMA. Use point-to-point per-port Thunderbolt interfaces (`no bridge0`, `rdma_en4` `PORT_ACTIVE`), giving `~0.5 ms` ping and `18–19 us` fast-sync `all_sum`.
- **Chat template defaults to `Reasoning Effort: Max`:** `GLM-5.3-Flash`'s `chat_template.jinja` inserts `<|system|>Reasoning Effort: Max` even if a plain system message is passed unless `reasoning_effort="low"` is passed explicitly to `apply_chat_template` / `--chat-template-args`.
- **Watchdogs must count real tokens, not SSE keepalives:** Servers send `: keepalive` SSE comments during long prefills. Reset the decode watchdog only on non-empty `delta.content` or `delta.reasoning` chunks, with a separate bounded timeout for cold `14k–128k` prefill.
- **Full 36-field `rusage_info_v4` struct in macOS `ctypes`:** Declaring only 27 fields (`224` bytes) for `proc_pid_rusage(..., RUSAGE_INFO_V4, ...)` caused a `296`-byte write overflow and Python GC segfault during Arm A1 prep; always declare all 36 fields (`296` bytes).
- **`omlx==0.8.0rc1` is not on PyPI:** Installing from PyPI fails with HTTP `404`; download and SHA-256-verify the official GitHub Release wheel (`v0.8.0rc1`) into an isolated venv (`<tandem-env>`) so existing `mlx-vlm` environments stay untouched.

---

## 10. Drafting quietly switches off under load: count drafted tokens, not flags

- **What happened:**
  - In the Round 7 many-user ladder, `GLM-Tandem` with `TP_MTP=1` stopped drafting at 5 or more concurrent streams. At `8-48` users more than `99.5%` of completion tokens were un-drafted; the few drafted tokens (`36-58` over 3 repeats) came only from the 1-4 stream ramp-in and ramp-out edges.
  - `TensorFold 1.2.10` kept drafting at `8` active streams per Mac even with `--parallel 16`, but drafting was off (`accepted=0/0`) when `16` streams ran per Mac (`16` and `32` users, and the single-Mac `16`-user reference).
- **Why:**
  - The multi-stream schedulers drop speculative drafting above a stream-count threshold. The startup banner and the flag do not change.
- **What to do:**
  - Give every row a "Drafted? yes/no (accepted/offered)" column taken from that block's own telemetry, and call a block "drafts on" only when drafted tokens `> 0`. Never describe a many-user number as speculative decoding when it was plain decode.

---

## 11. A long prompt delays short users, and `mlx-vlm` can use most of a 512 GB Mac's memory

- **What happened:**
  - With one `31,995`-token prompt in flight, `8` short prompts waited `7.65 s` median (`37.04 s` for the 9th stream) on one-Mac `TensorFold`, `15.18 s` on the two-Mac `GLM-Tandem` pair, and `36.33 s` on one-Mac `mlx-vlm 0.8.0`.
  - All six `mlx-vlm` runs reached `485.3 GB` physical / `512.8 GB` system memory on a 512 GiB Mac. Our memory watchdog (`80%`, `410.0 GB`) fired `6` times across the three `mlx-vlm` set-ups x `2` repeats (the default set-up alone fired `2`); the server drained its in-flight replies cleanly when we stopped it.
  - Setting `mlx-vlm`'s `--prefill-step-size 512` gave a longer short-request wait (`50.23 s` vs `36.33 s`). `TensorFold`'s prefill environment variables (`TENSORFOLD_PREFILL_TOKENS`, `TENSORFOLD_PREFILL_ORDER`) are not in the `1.2.10` documentation, so we did not run them.
- **Why:**
  - `TensorFold` is best for this job: it splits a long prompt into `16,384`-token prefill chunks, so short requests slip in after the first chunk. `mlx-vlm`'s scheduler processes all active prompt chunks before steady decoding and keeps large temporary prefill buffers (our reading of the logs, `likely`, not proven). `GLM-Tandem` has no chunking setting at this pin.
- **What to do:**
  - Test short-request latency next to a long prompt before putting any of these engines behind several users, keep a memory watchdog on `mlx-vlm` with long prompts, do not rely on `--prefill-step-size 512`, and use documented flags only.

---

## 12. `mlx-vlm 0.8.0`'s built-in cluster mode does not work for this model

- **What happened:**
  - One attempt (`4.46 s`): the cluster endpoint reported `"state":"failed"` and chat requests returned HTTP `503` (`Startup preloading is in progress`). The server code raises `This model has no distributed shard implementation` because `glm5_next.Model` has no `shard` method.
- **Why:**
  - The model class never implemented tensor sharding in that release.
- **What to do:**
  - Do not claim a speed-up "over `mlx-vlm`'s two-Mac mode". The only working plain two-Mac `mlx-vlm` reference we have is our own clean-room shard overlay, and we label ratios against it as such.

---

## 13. The 4-bit vs 8-bit loss check could not run

- **What happened:**
  - `mlx_lm.perplexity` failed in `1.44 s` with `ValueError: Model type glm5_next not supported.` (`mlx_lm 0.32.2` has no `glm5_next` module). We stopped without writing our own scorer, so the quality loss of 4-bit versus 8-bit is still unmeasured.
- **Why:**
  - The stock `mlx_lm` release does not know this model type.
- **What to do:**
  - Keep every 4-bit result labelled `4-bit, lossy` and do not quote a quality gap. Our substitute evidence is task accuracy: `98.0%-99.0%` on GSM8K with and without drafts, and `100/100` majority vote on the sampled GSM8K check, none of which measures the 8-bit gap.

---

## 14. Comparing draft ON vs OFF accuracy: score strictly, and do not send a `seed` to `mlx_lm.server`

- **What happened:**
  - Over `400` questions, drafts off and on differed on `10` answers (`393/400` vs `390/400`, `p = 0.5078`). Two of the misses (`logic-008`, `logic-009`) had the right answer on the last line but without the required `ANSWER: ` prefix; we kept them as wrong rather than re-scoring.
  - Sending a JSON `"seed"` to `mlx_lm.server` disables `GLM-Tandem`'s batch-generator kernels, so `T=0.7` runs on it had to go unseeded and were not repeatable (`0/24`), while seeded `mlx-vlm` and `TensorFold` were (`24/24`).
- **Why:**
  - The answer differences are most likely the near-tie `bfloat16` effect of Lesson 5, but Part A did not log per-token `logprobs`, so that is inferred, not measured. The seed check is in `mlx_lm.server` itself.
- **What to do:**
  - Fix the scoring rule before the run and apply it to every arm. Log per-token `logprobs` if you want to prove the cause. Do not claim reproducible sampled (`T>0`) outputs on `GLM-Tandem`.

---

## 15. Cleanup and resume mistakes in our own harness

- **What happened:**
  - Round 6 Part B first ran on 20 Nov (`02:00-03:14`) and exited mid-visit `B2`. We waited for another job that was using the Macs to finish, then re-ran all `8` mirrored visits from scratch in one uninterrupted session (`20:30-22:07`) instead of resuming about `18` hours later.
  - Our post-run check script failed on one Mac because of an unescaped `\n` inside a `python3 -c` string, and because the `exo` status API was not listening. We fixed the snippet and re-ran the post-check, which then passed.
- **Why:**
  - Mirrored `A-B-C-D-D-C-B-A` orderings only cancel drift if they run back to back. Cleanup scripts are code and need a test run too.
- **What to do:**
  - Re-run a broken mirrored sequence from the start, treat "service not listening" as idle in an idle-check, and test the cleanup script before the real run.
