# Agent Reproduction Brief (`AGENTS.md`)

This brief gives an AI coding agent everything required to reproduce our **GLM-5.3-Flash** benchmarks on one or two **512 GB Apple Mac Studios (M5 Ultra, macOS 27.2 build `26B118`)** over Thunderbolt 5 RDMA (`jaccl`) using **stock MLX 0.32.4**.

Use **Mac A** (rank 0 / coordinator host) and **Mac B** (rank 1) placeholders everywhere. Replace `<MAC_A_TB_IP>`, `<MAC_B_TB_IP>`, `<MAC_A_HOST>`, `<MAC_B_HOST>`, and `<MODEL_DIR>` with your lab values, and **never commit or publish host names, IP addresses, user names, or temporary SSH keys**.

---

## 0. Hard Stop Rules (Non-Negotiable)

Stop immediately and abort the run if any of these are violated:

1. **NEVER enable fast sync on 8-bit / FP8 / MXFP8:** Never set `MLX_METAL_FAST_SYNCH=1` (or `MLX_FENCE_MODE=1`) when running FP8, MXFP8, or 8-bit (`oQ8`) models over `jaccl`. Enforce a pre-launch guard that kills any fast-sync run on non-4-bit graphs.
2. **Stock MLX only:** Use stock `mlx==0.32.4`; leave `MLX_FENCE_MODE` unset. Do not compile custom MLX C++ patches, install `.dmg` GUI bundles, run `sudo`, or change `sysctl iogpu.wired_limit_mb` (keep `0`).
3. **Bind `127.0.0.1` only:** Every HTTP server (`mlx_vlm.server`, `GLM-Tandem` `tpserve.py`, `tensorfold-native`) must listen exclusively on `127.0.0.1`. Before running `GLM-Tandem`, inspect its server entrypoint and ensure `--host` is `127.0.0.1` rather than `0.0.0.0`.
4. **Licence boundary:** `icryo/GLM-Tandem` (`https://github.com/icryo/GLM-Tandem`) has no licence. Never copy, paste, or redistribute `GLM-Tandem`, `oMLX`, `TensorFold`, `exo`, or `MLX` code or patches into public files.
5. **Remove temporary SSH keys on exit:** If a run creates a temporary ed25519 keypair on Mac A (`~/.ssh/<tmp_pair_key>`) to let rank 0 spawn rank 1 on Mac B over Thunderbolt, **always** delete both key files on Mac A, strip that key line from `~/.ssh/authorized_keys` on Mac B (`chmod 600`), and verify `ssh` returns `Permission denied`—even if the benchmark crashes.
6. **Memory ceilings:** Poll process `ri_phys_footprint` and `vm_stat` `sys_used` every `1.5 s` on both Macs. Abort immediately if either reaches **`>= 338.0 GB`** (66% line, plain 2-Mac / 1-Mac MXFP8 checks) or **`>= 410.0 GB`** (80% line, Tandem / TensorFold checks).
7. **One benchmark at a time (quiet Macs):** Stop any standing model server (e.g., port `8090`), wait for exclusive machine locks and zero competing CPU/GPU tasks (test suites, builds, other servers), confirm idle orchestrators (e.g., `exo` on `127.0.0.1:52415/state` returns empty `instances`, `runners`, `tasks`), and restore any stopped service afterwards.

8. **Documented flags only; no guessed prefill settings:** Use only flags that appear in the engine's own `--help`, `serve --help`, `capabilities --json` or runtime notes. Do **not** set `--prefill-step-size 512` on `mlx-vlm` (it gave a longer short-request TTFT: `50.23 s` vs `36.33 s`) and do **not** set `TENSORFOLD_PREFILL_TOKENS` / `TENSORFOLD_PREFILL_ORDER` on `TensorFold 1.2.10` (not documented there).
9. **Void, do not repair:** If an arm cannot run as specified (for example `mlx_lm.perplexity` raising `Model type glm5_next not supported`), record the failure text, mark the result void, and move on. Do not write a substitute scorer or guess a flag unless your instructions say so.

---

## 1. Exact Pins & SHA-256 Checksums

### 1.1 Hardware, OS, and Link
- **Hardware:** Two Mac Studios (`Mac17,15`), Apple M5 Ultra (36 CPU cores: 12P+24E, 512 GiB = `549,755,813,888` bytes unified memory each).
- **OS:** macOS `27.2`, build `26B118` (`sw_vers -buildVersion`), `iogpu.wired_limit_mb=0`.
- **Thunderbolt link:** One Thunderbolt 5 cable (`rdma_en4` <-> `rdma_en4`, `PORT_ACTIVE`, MTU `4096`), **Thunderbolt Bridge (`bridge0`) OFF**, RDMA enabled (`rdma_ctl` enabled).

### 1.2 Model Packs & Revisions
- **Primary 4-bit MTP pack:** `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` (MIT)
  - Revision: `d2b17f5253440224422b150705c1109a9a3a0368` (`54` files: `45` LFS SHA-256 + `9` size-checked, `0` bad)
  - Key SHA-256s:
    - `config.json`: `f80682bcc8ba5fc6ecdc5ce553700b02befe999bc4a6ba6fd47f644df4864adf`
    - `model.safetensors.index.json`: `6c030d5b58f4c264f00becf9dddcb8f8b1f69b3bbaaf35c93592b201b673208f`
    - `tokenizer.json`: `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`
  - Expected config fields: `num_hidden_layers=45`, `num_nextn_predict_layers=1`, `quantization={"bits": 4, "group_size": 64, "mode": "affine"}`.
- **Optional FP8 (run as MXFP8):** `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` (results for this pack are speed-only; check its model card licence before reuse)
  - Revision: `d5822b068da8ddb301665b48c58718e6e18a2c3a` (`74` files: `63` SHA-256 + `11` size-checked, `0` bad).
- **Optional Official FP8:** `zai-org/GLM-5.3-Flash` (MIT)
  - Revision: `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` (`74` files: `63` SHA-256 + `11` size-checked, `0` bad).
  - Note: Does not load directly in `GLM-Tandem` (rank-1 `Linear` `.scales` error). Optional `oQ8-MTP` conversion writes 67 shards (`319 GB`, `8-bit` affine `group_size=64`, `59` `mtp.*` keys) to a separate folder (`config.json` `20ad12d3c417a849e2c9e01bdbee67ada9ec29cf8592bcdda4aae90fadeabfb7`, `index.json` `5c9c3412a521e3f5a72f01657f190a1b24ad9d2136ff70fe4dbfb497169d91d0`).

### 1.3 Software Environments
- **Base MLX venv (`~/opt/mlx313`):**
  - Python `3.13.18`, `mlx==0.32.4`, `mlx-metal==0.32.4`, `mlx-vlm==0.8.0`, `mlx-lm==0.32.2`, `transformers==5.23.0`, `safetensors==0.8.0`.
  - Freeze SHA-256 (`uv pip freeze --python ~/opt/mlx313/bin/python | shasum -a 256`):
    `43474b7e63833fda26229d44e60d93d57a6de5f57e3d233d08a31b19c7f00dfa` (verify identical before & after).
  - Never edit `site-packages`; copy `mlx_vlm` into an overlay (`~/two-mac/overlay/mlx_vlm`) if applying a clean-room `glm5_next` `shard()` patch.
- **GLM-Tandem venv (`~/opt/tandem`) & checkout (`~/two-mac/tandem`):**
  - Repo: `https://github.com/icryo/GLM-Tandem` detached at `1a70342137f2650367b8ca2619e19ea854a7658b` (`git rev-parse HEAD`).
  - `oMLX` wheel (from `https://github.com/jundot/omlx/releases/tag/v0.8.0rc1`, not PyPI):
    `omlx-0.8.0rc1-cp313-cp313-macosx_15_0_universal2.whl`
    SHA-256: `a416a587d12116858bfbbe67530012ce1c368aae108cfee406b638373ebf7af0`
  - Install into isolated `~/opt/tandem` (`uv venv --python 3.13 ~/opt/tandem`); verify `mlx==0.32.4` (stock) and `omlx==0.8.0rc1`.
  - Bind-address check: verify `tpserve.py` binds `--host 127.0.0.1` instead of `0.0.0.0` (keep `git diff` minimal and verify its hash before every launch).
- **TensorFold (`~/opt/tensorfold-1.2.10`, Mac A only):**
  - Release `v1.2.10` (`https://github.com/ashhart/TensorFold`, Apache-2.0).
  - Archive `tensorfold-1.2.10-macos-arm64.tar.gz` SHA-256: `76eb33cce9e297f156cfe5cbb0cec664b0266f4ecc3d103176adadc6aee21e58`
  - Unpack and verify `shasum -a 256 -c SHA256SUMS` (`23/23 OK`).
  - Binary `bin/tensorfold-native` SHA-256: `ef0ab81eb11496956cf33eba9fd16d26b3500be54f6c93b08885d667c1af19b6`.
  - Read `RUNTIME.md`, `--help`, `serve --help`, `capabilities --json`, and `device --json` before running; use documented flags only.

---

## 2. Step-by-Step Preflight Checks

Run these on **both Mac A and Mac B** before every benchmark session:

1. **OS & GPU wired limit:** Check `sw_vers -buildVersion` == `26B118` and `sysctl -n iogpu.wired_limit_mb` == `0`.
2. **Thunderbolt RDMA & Bridge check:**
   - Run `ibv_devices` -> confirm `rdma_en4` is present and active (`PORT_ACTIVE`).
   - Run `ifconfig` -> confirm `bridge0` is **absent** and the Thunderbolt link-local interface has an active address on both Macs.
3. **Quiet-Mac check:**
   - Confirm zero competing compute/server processes (`tpserve`, `mlx_vlm`, `mlx_lm`, `tensorfold`, heavy builds).
   - If `exo` is running, `curl -sf http://127.0.0.1:52415/state` must report empty `instances`, `runners`, and `tasks`.
   - Check `lsof -iTCP:8010 -sTCP:LISTEN`, `:8020`, `:8090` are free (stop standing `:8090` service first if active, recording the exact restart command).
4. **Software & Manifest verification:**
   - Check `~/opt/mlx313` freeze SHA-256 (`43474b7e...0dfa`).
   - Check `GLM-Tandem` `HEAD` (`1a703421...`) and `127.0.0.1` bind diff (`git diff`).
   - Check `tensorfold-native` SHA-256 (`ef0ab81e...`) and `SHA256SUMS` on Mac A.
   - Verify model directory against manifest: assert **`ok + oksize == rows`** (`45 + 9 = 54` for 4-bit; `63 + 11 = 74` for FP8), **`bad == 0`**, and explicit SHA-256 match for `config.json`, `model.safetensors.index.json`, and `tokenizer.json`.

5. **Prompt and corpus hashes (Round 6/7 runs):** Check SHA-256 of the prompt files before use and record them: 24-prompt realistic set `825c8bc84f7709a44497acd491b47c2156079e4e8f980b3bfd2c458ee89aed10`; `short_32` / `conc_8k` set `4d3a622d61bbaedb669222fa6bbf22eb6cab670f4a5a382b1fb595172611c122`; RFC 9110 full text `21c1cdce6ab0e5509b04d84a28000836c7a087cf786efe6f04877ebfff47232a` (first `200,000` bytes `0c7d3f5fab58e6be647c5e979e0ddcb0a2526a4d627a14b93b52f880a9e08d55`); GSM8K test parquet `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59` (first-100-items file `7e5802845a8ea84a106506aa84134d95f847638d7fbb1364ce8f3079291a999e`). The 200 generated questions and the 200 GSM8K questions are in `receipts/prompts/`; for the 24-prompt realistic set and the `short_32` / `conc_8k` set only the hashes are published (see `receipts/README.md`).
6. **Reasoning effort:** The pack's chat template defaults to `Max` when `reasoning_effort` is omitted. Decide the effort per test (`Low` for the short-prompt, many-user and Round 4/5 runs; `Default`/`Max` for the Round 6/7 realistic runs), apply it to every engine, and record it.

---

## 3. Commands in Outline (End-to-End Suite)

Write host/RDMA config files with placeholders locally before running:
- `hosts-jaccl.json`: 2 entries (`<MAC_A_HOST>` with `<MAC_A_TB_IP>` and RDMA `[null, "rdma_en4"]`; `<MAC_B_HOST>` with `["rdma_en4", null]`).
- `tandem_ibv_devices.json`: `[[null, "rdma_en4"], ["rdma_en4", null]]`.

### Step A — Interconnect Microbenchmark (`sync_bench`)
1. Run `mlx.launch --backend jaccl --hostfile <HOSTS_JSON> --python ~/opt/mlx313/bin/python <SYNC_BENCH_SCRIPT>` with `MLX_METAL_FAST_SYNCH` unset (expect `~188–197 us/sum` at 1 row) and with `--env MLX_METAL_FAST_SYNCH=1` (expect `~18–19 us/sum` at 1 row).
2. Run a 12,000-step 1-row soak with `MLX_METAL_FAST_SYNCH=1` under a 15-minute watchdog; confirm `0` hangs before proceeding to model runs.

### Step B — One-Mac Baselines (`mlx-vlm 0.8.0`)
1. **4-bit baseline:** Run `~/opt/mlx313/bin/python -m mlx_vlm.generate --model <4BIT_MODEL_DIR> --temperature 0 --max-tokens 1024 --verbose` (and pass `reasoning_effort="low"` via `apply_chat_template` when matching short-prompt tests).
2. **FP8->MXFP8 baseline (Mac B):** Run the same on `<FP8_MODEL_DIR>` under a `< 338 GB` memory guard (`expect ~26.3–26.5 tok/s`, `~334.9 GB` peak).

### Step C — Two-Mac Plain Baseline (`mlx-vlm 0.8.0` + Clean-Room `glm5_next` Shard Overlay)
1. **Tiny-model verification first:** Run single-process reference logits vs `mlx.launch -n 2 --backend ring python test_shard.py --sharded` (`max_abs_diff <= 1e-3`, plus planted-break failure test).
2. **4-bit pair (`fast_synch=1` and `fast_synch=0` control):** Launch via `mlx.launch --backend jaccl --hostfile <HOSTS_JSON> --env PYTHONPATH=~/two-mac/overlay --env MLX_METAL_FAST_SYNCH=1 --python ~/opt/mlx313/bin/python <TWOMAC_RUNNER>` with `temperature=0`, `reasoning_effort="low"`, `max_tokens=512` (or `2048` for A-B-C-C-B-A), checking `< 338 GB` per Mac and rank token agreement.
3. **FP8->MXFP8 pair (`fast_synch=0` ONLY):** Run with `MLX_METAL_FAST_SYNCH` unset (`0`). Do NOT enable fast sync (Lesson 1).

### Step D — Two-Mac `GLM-Tandem` (Rounds 2, 3, 4, 4b, 5)
1. **Temporary TB SSH key:** Create ed25519 key `~/.ssh/<tmp_pair_key>` on Mac A, append `.pub` to Mac B's `~/.ssh/authorized_keys`, verify `ssh -i ~/.ssh/<tmp_pair_key> -o BatchMode=yes <USER>@<MAC_B_TB_IP> true`.
2. **Launch ranks (stock MLX `0.32.4`, `127.0.0.1` only):**
   - Rank 0 (Mac A) & Rank 1 (Mac B) inside `~/two-mac/tandem/server` using `~/opt/tandem/bin/python -u tpserve.py` with:
     `MODEL_PATH=<4BIT_MODEL_DIR>`, `MLX_IBV_DEVICES=<IBV_JSON>`, `MLX_RANK=<0|1>`, `MLX_WORLD_SIZE=2`, `MLX_JACCL_COORDINATOR=<MAC_A_TB_IP>:<PORT>`, `TP_PORT=8010`, `TP_MTP=1`, `TP_LOOKUP=1`, `TP_MTP_LOG=1`, `MLX_MAX_MB_PER_BUFFER=20000`, `MLX_MAX_OPS_PER_BUFFER=100000`.
   - Set `MLX_METAL_FAST_SYNCH=1` **ONLY** on 4-bit runs (`MLX_FENCE_MODE` unset). Leave `MLX_METAL_FAST_SYNCH` unset for default-sync controls and 8-bit `oQ8-MTP`.
   - Set `TP_CONC=1 TP_PROMPT_CONC=1` for single-stream/soak/long-context; set `TP_CONC=32 TP_PROMPT_CONC=1` for multi-user concurrency.
   - Confirm startup log shows `Starting httpd` on `127.0.0.1:8010`, rank 1 ready, and MTP bound (`bound 1 nextn MTP layer(s)` or `loaded 59 tensors` in `oQ8`).
3. **Workloads (client runs locally on Mac A against `http://127.0.0.1:8010/v1/chat/completions`, `temperature=0`, `stream=True`):**
   - **Round 2:** Default sync first, then `MLX_METAL_FAST_SYNCH=1` on `english-reasoning`, `arabic`, `prose-400` (`max_tokens=2048`).
   - **Round 3 soak & long context:** 30-min loop (`1800 s`) cycling 6 prompt types including unique ~8k RFC 9110 slices; fresh servers for `lc-14k` and `lc-54k` RFC 9110 prompts (`cached_tokens=0`). An optional official `oQ8-MTP` conversion (exact command not published here) was run with **default sync ONLY**.
   - **Round 4 Gate 1 (token identity):** Compare `TP_MTP=0 TP_LOOKUP=0` vs `TP_MTP=1 TP_LOOKUP=1` across 6 prompts (`english-reasoning`, `arabic`, `prose-400`, `code-refactor`, `summary-2k`, `tool-multi-turn`) with both `stream=True` and `stream=False, logprobs=True` (to hash raw token IDs).
   - **Round 4b Gate 2 (`A-B-C-C-B-A`):** Run fresh clusters in order `A1 -> B1 -> C1 -> C2 -> B2 -> A2` (`5` reps x `3` short prompts per run), capturing exact `token_ids` probes on warm `B`/`C` servers after timed reps.
   - **Round 5 concurrency:** Run `1, 2, 4, 8, 16, 32` short users + `8 x 8k` (`max_tokens=256`, fresh server + disjoint warmup per step so `cached_tokens=0`). For `N >= 8`, pre-connect client sockets (`5 ms` stagger + retry) before releasing the request barrier to bypass `ThreadingHTTPServer`'s `listen(5)` drop (and optionally record a raw un-staggered burst separately).
4. **Teardown:** Kill `tpserve.py` on both Macs, remove temporary SSH key files on Mac A and strip line from Mac B `authorized_keys`, assert TB SSH fails with `Permission denied`.

### Step E — One-Mac `TensorFold v1.2.10` (`Mac A` Only)
1. **Launch:** `TENSORFOLD_PHASE_METRICS=1 ~/opt/tensorfold-1.2.10/.../bin/tensorfold-native serve <4BIT_MODEL_DIR> --host 127.0.0.1 --port 8020 --no-update-check --metrics-open --reasoning-effort low --temperature 0 --max-tokens <2048|1024>`.
2. **Verify MTP loaded:** Startup log must contain `45 of 45 layers and the MTP layer` and `drafts: MTP + copies`. If absent, label `plain decode` and stop after single-stream.
3. **Workloads (TF1 & R5 Arm C):**
   - `Run 2`: 5 reps of `english-reasoning`, `arabic`, `prose-400`, `code-2k`.
   - `Run 3`: fresh server per prompt for `lc-32k` (RFC 9110) and `lc-128k` (Shakespeare).
   - `Run 4`: `4` concurrent `~8k` RFC 9110 users.
   - `Run 5`: same-day `A-B-B-A` (`TensorFold -> mlx-vlm+overlay -> mlx-vlm+overlay -> TensorFold`, 5 reps x 3 short prompts).
   - `Round 5 Arm C`: `1, 4, 8, 16` users + `8 x 8k` (`max_tokens=256`), plus `16` users with `--parallel 16 --max-inflight 32` (expect MTP auto-disabled `0/0` in logs).

### Step F — Dual Independent `mlx-vlm` Concurrency (`Round 5 Arm B`)
1. Launch stock `~/opt/mlx313/bin/python -m mlx_vlm.server --host 127.0.0.1 --port 8090 --model <4BIT_MODEL_DIR> --max-tokens 4096` independently on **Mac A** and **Mac B** (no draft model, no overlay).
2. Split `1 (1+0), 2 (1+1), 4 (2+2), 8 (4+4), 16 (8+8), 32 (16+16)` short users and `8 (4+4) x 8k` evenly across Mac A and Mac B behind a single client barrier (over local SSH tunnels or direct loopback).

### Step G — Round 6: Quality Score and Realistic Rerun (`4-bit, lossy`, `Default` effort)
1. **Part A (quality):** Three arms on fresh servers, `T=0`, max `4,096` tokens, one request at a time: `A` = `GLM-Tandem` 2 Macs `TP_MTP=0`; `B` = `GLM-Tandem` 2 Macs `TP_MTP=1`; `C` = `TensorFold 1.2.10`, 1 Mac. Two suites of `200` questions each: a generated suite (arithmetic, date/calendar, unit conversion, Python output, string tasks, logic ordering; `seed=20261120`) and a `200`-item GSM8K test sample (`seed=20261120`). Score strictly on a final non-empty line `ANSWER: <value>` (GSM8K: exact numeric match). Record per item: answer, `finish_reason`, tokens, and whether the output was empty or hit the length limit. Compare A vs B with an exact McNemar test.
2. **Part B (speed):** `24` varied prompts (`46-7,969` input tokens) x `T=0.7` and `T=0.0` x `2` visits, max `1,024` tokens, mirrored order `A1 -> B1 -> C1 -> D1 -> D2 -> C2 -> B2 -> A2` (A = `mlx-vlm 0.8.0` 1 Mac; B = `TensorFold 1.2.10` 1 Mac; C = `GLM-Tandem` drafts OFF; D = `GLM-Tandem` drafts ON). Run all eight visits in one uninterrupted session; if it breaks, start again rather than resuming later. Send `"seed"` to `mlx-vlm` and `TensorFold`; do **not** send it to `mlx_lm.server`/`GLM-Tandem` (it disables the batch kernels, Lesson 14).

### Step H — Round 7: Repeats, Many-User Curves, Long-Prompt Fairness, Loss Check, Sampled GSM8K
1. **Part 1 (headline repeat):** `24` prompts, `T=0`, `Default` effort, `max_tokens=512`, three rotated restarts (`P -> Q -> T1 -> M1 -> T0`, then the reverse, then `Q -> P -> PL -> M1 -> T1 -> T0`), each prompt twice per restart, `3` untimed warm-ups per server start, fresh server per block. Arms: `P` = `GLM-Tandem` `TP_MTP=1 TP_LOOKUP=1`; `Q` = `TP_MTP=0 TP_LOOKUP=0`; `PL` = `P` with `TP_LOOKUP=0` on the `12` document prompts; `T1` = `TensorFold` default; `T0` = `TensorFold --no-drafts --prompt-cache-gib 0`; `M1` = `mlx-vlm 0.8.0`. Report ratios with a `10,000`-resample bootstrap over prompts x restarts. Optionally attempt `mlx-vlm`'s built-in cluster mode once (expect it to fail, Lesson 12).
2. **Part 2 (many users):** `T=0`, `Reasoning Effort: Low`, `max_tokens=256`, `3` repeats per rung. Systems: `S1` = `GLM-Tandem` pair (`TP_CONC=max(32, n)`, `TP_PROMPT_CONC=1`, fast sync on); `S2` = one `mlx-vlm 0.8.0` per Mac with `--max-num-seqs 32 --max-tokens 4096`; `S3` = one `TensorFold 1.2.10` per Mac with `--parallel 8` (up to `16` users) or `--parallel 16 --max-inflight 32` (`16-64` users), `--reasoning-effort low`; `S4` = one `TensorFold` on one Mac. Rungs `1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64` short users (as far as each system allows) and `1, 2, 4, 8` users with `~8k` prompts. Pre-connect client sockets before the barrier and run one raw simultaneous-`connect()` burst at `32` users. Use a unique per-user prompt prefix above `32` users so `cached_tokens=0`. Fill in a "Drafted? yes/no (accepted/offered)" column from each block's own telemetry.
3. **Part 3 (fairness):** One `31,995`-token RFC 9110 prompt at `t=0`, then `8` short prompts staggered `0.25 s` apart, `max_tokens=256`, `T=0`, `2` repeats per arm. Arms: `TensorFold 1.2.10` default; `GLM-Tandem` default; `mlx-vlm 0.8.0` default. Record short-prompt TTFT p50/p95/max, longest inter-token gap, long-prompt TTFT, total tok/s, whether short outputs match their solo runs, and peak memory under the `80%` watchdog (expect it to fire on `mlx-vlm`).
4. **Part 4 (loss check, expected to fail):** Try `mlx_lm.perplexity --sequence-length 512 --num-samples 64` on the first `200,000` bytes of RFC 9110 with fast sync OFF. With `mlx_lm 0.32.2` it raises `Model type glm5_next not supported`; record that and stop (hard stop 9).
5. **Part 5 (sampled GSM8K):** First `100` GSM8K items, `5` samples each, `T=0.6`, `Default` effort, max `4,096` tokens, `GLM-Tandem` with drafts OFF (`TP_MTP=0 TP_LOOKUP=0`) and ON (`TP_MTP=1 TP_LOOKUP=1`), `TP_CONC=1`. Report majority-vote accuracy, mean per-sample accuracy, mean output tokens, wall time and an exact McNemar test.
6. **Not run in our rounds (skipped, optional):** an `exo` comparison, `HumanEval` (never run model-written code on the test Macs), newer `GLM-Tandem` commits (read the changes for security first), and wall-power readings.

---

## 4. Watchdogs & Probes

- **Decode no-token watchdog:** `30.0 s` (`60.0 s` in R2/R3/TF1). Reset the timer **only** when a streaming chunk carries non-empty `delta.content` or `delta.reasoning` (ignore SSE `: keepalive` lines).
- **Prefill timeout:** Separate `180–600 s` cap before the first token on cold `14k–128k` or `8 x 8k` batches; flag any stream with `ttft_s > 30 s`.
- **First-token wait:** `120 s` for short prompts, `300 s` for `8k` and longer prompts; kill the request, end the arm block, record. **Request errors:** stop climbing the ladder if more than `10%` of a block fails (except the deliberate raw-burst test). **Thermal:** if `pmset -g therm` shows any CPU/GPU speed limit below `100`, wait `5` minutes up to `3` times, otherwise skip the block as `thermal`. **Part time cap:** stop a part at `1.5x` its planned time and write the measured data.
- **Fast-sync guard in code:** the launcher must refuse `MLX_METAL_FAST_SYNCH=1` whenever the model path contains `fp8`, `8bit`, `oq8` or `mxfp8` (this also covers the `mlx_lm.perplexity` attempt: run it with the variable unset).
- **Mirrored-run break:** if a mirrored sequence stops partway, re-run it from the first visit.
- **Stall capture:** On watchdog timeout, immediately capture `sample <pid> 5` on both ranks into logs, terminate the cluster, and record `stall`.
- **Memory monitor:** Poll `vm_stat` + `proc_pid_rusage(..., RUSAGE_INFO_V4, ...)` using the **full 36-field (296-byte) `rusage_info_v4` structure** every `1.5 s`. Never use a truncated struct.

---

## 5. What to Record in Every Receipt

Save a machine-readable JSON receipt + scrubbed server/driver logs for every run:

- **Metadata:** `macos_build` (`26B118`), machine ID (`Mac A` / `Mac B`), `wired_limit`, engine name/version/commit/wheel SHA-256, `model_revision`, `fence_mode` (`MLX_METAL_FAST_SYNCH`, `MLX_FENCE_MODE`), label (`MEASURED`, precision, 1 or 2 Macs, MTP or plain).
- **Round 6/7 additions:** `reasoning_effort` in force, prompt-file SHA-256s, restart/visit order, `cached_max`, `max_inter_tok_s`, **"Drafted? yes/no (accepted/offered)"** per block (call a block "drafts on" only if drafted tokens `> 0`), `time_to_answer` (first non-empty `delta.content`; null if the model was still thinking), `n_out` from the server's `usage` field (never count stream chunks), the seed used or `none`, and the watchdog outcome. Many-user rows: total E2E (`sum(n_out) / (latest t_last - earliest t0)`) plus per-user median and minimum (never total divided by users).
- **MTP status:** `mtp_head_loaded` (bool + exact startup evidence line) and per-request `accept_rate` (`acc/drafted` or `head_acc/head_off` + `lookup_acc/lookup_off`).
- **Timing & cache hygiene:** `load_elapsed_s`, `prompt_tokens`, **`cached_tokens`** (must be `0` for cold measurements), `completion_tokens`, `finish_reason`, `ttft_s`, `decode_s`, `wall_s`, `max_inter_tok_s`, `client_read_tok_s`, `client_write_tok_s`. For multi-user batches: `completed`, `errors`, `batch_wall_s`, `batch_decode_span_s`, `agg_e2e_tok_s`, `agg_decode_span_tok_s`, per-user median & min `tok/s`, TTFT median & max.
- **Determinism & memory:** `token_sha256` (or `content_sha256` + `reasoning_sha256`), `peak_phys_gb`, `lifetime_max_phys_gb`, `peak_sys_used_gb`, `stalls`, `mem_abort`.
- **Post-run cleanup proof:** Zero stray processes, ports `8010`/`8020`/`8090` in expected state, `exo` idle, `~/opt/mlx313` freeze SHA-256 unchanged, temporary SSH keys removed and rejected, and any standing service restarted.
