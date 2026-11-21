# GLM-5.3-Flash at 141 tok/s on two Mac Studios

**A 320-billion-parameter Mixture-of-Experts model running locally on Apple Silicon — measured, reproducible, with open data.**

![Model](https://img.shields.io/badge/model-GLM--5.3--Flash%20320B%20MoE-111111)
![Active](https://img.shields.io/badge/active%20params-18B-555555)
![Hardware](https://img.shields.io/badge/hardware-2%C3%97%20Mac%20Studio%20M5%20Ultra%20512%20GB-555555)
![Framework](https://img.shields.io/badge/framework-Apple%20MLX-555555)
![Engines](https://img.shields.io/badge/engines-TensorFold%20%C2%B7%20GLM--Tandem%20%C2%B7%20mlx--vlm-555555)
![Precision](https://img.shields.io/badge/precision-4--bit-555555)
![License](https://img.shields.io/badge/license-Apache--2.0-555555)

## Thank you

This work stands on other people's generosity. Thank you to **Ash Hart and TensorFold**, whose 4-bit model pack every engine here ran on and whose engine leads on one Mac; to **Z.ai** for releasing GLM-5.3-Flash openly; to **icryo** for GLM-Tandem; to **Apple's MLX team**; to **Blaizzy** (mlx-vlm), **jundot** (oMLX) and **the exo team**; and to the community members whose posts taught us what to test. Every name, with what each gave us, is in [`CREDITS.md`](CREDITS.md).

## Why, how, what — in plain words

**Why.** Very large AI models usually live in data centres. We wanted to know whether one of the largest open models could run fast and privately on two desktop computers, and which free engine is best for which job.

**How.** We joined two Apple Mac Studios with a single cable, gave every engine the same model, and timed real requests: one person, many people at once, long documents, and a maths test. Each test was repeated with fresh restarts, and every number is measured and published with its raw data.

**What we found.**
- **One person:** 141.4 tokens per second of output with the model split across both Macs; 94.7 on a single Mac with TensorFold.
- **Reading prompts:** over 2,000 tokens per second on either set-up, with prompts up to 128,000 tokens.
- **Many people:** TensorFold, one per Mac, gives the best results from 2 to 32 users at once.
- **Accuracy:** the speed tricks cost nothing on a maths test — 100 out of 100 either way.

![Prompt reading and output speed](charts/00-speed.png)

## The numbers people ask for first

All `MEASURED`, `4-bit, lossy`, on the same TensorFold `GLM-5.3-Flash-MLX-4bit-MTP` model pack.

| | **GLM-Tandem** — 2 Macs, tensor-parallel over Thunderbolt 5 RDMA | **TensorFold 1.2.10** — 1 Mac | **mlx-vlm 0.8.0** — 1 Mac |
|---|---|---|---|
| **Prompt reading (prefill)** | **2,032.8 – 2,174.0 tok/s** (14k – 54k-token prompts) | **2,002.6 – 2,071.6 tok/s** (32k – 128k-token prompts) | not measured on public text |
| **Output (decode), 1 user, median** | **141.4 tok/s** | **94.7 tok/s** | **43.4 tok/s** |
| **End-to-end, 1 user, median** | 119.9 tok/s | 81.7 tok/s | 41.7 tok/s |
| **Many users, total output** | 248.3 tok/s at 32 users | 271.8 tok/s at 32 users (one per Mac) | 334.7 tok/s at 64 users (one per Mac) |
| **Peak memory per Mac** | 98.3 / 100.0 GB | 189.2 GB | 196.2 GB |
| **Longest prompt tested** | 54k tokens | 128k tokens | — |

Prefill: single long prompt, fresh server, no prompt cache ([`BENCHMARKS.md`](BENCHMARKS.md) §1.4). Single-user output: 144 timed requests per set-up across 3 fresh restarts, Round 7. Many users: median of 3 repeats ([`BENCHMARKS.md`](BENCHMARKS.md) §2.3).

## Which engine for which job

![Table 1: the right engine for each job](charts/0-at-a-glance.png)

Every engine below ran on the same **TensorFold `GLM-5.3-Flash-MLX-4bit-MTP` pack** (rev `d2b17f52`), so all results here build on TensorFold's work. All figures are `MEASURED`, `4-bit, lossy`; details and receipts follow.

| Job | Best choice | Measured result |
|---|---|---|
| Fastest for one user | **2 Macs `GLM-Tandem`** (by icryo) | `119.9 tok/s` end-to-end (Round 7), `1.47x` the best one-Mac result |
| Fastest on one Mac | **`TensorFold 1.2.10`** (with drafting) | `81.7 tok/s` end-to-end (Round 7) |
| Many users, 2 to 32 | **One `TensorFold 1.2.10` per Mac** | `243.0 tok/s` at 16 users, `271.8` at 32, first token under `5 s` (`3.07 s` max at 32) |
| Short requests next to a 32k document | **`TensorFold 1.2.10`** (one Mac) | shortest median wait: `7.65 s` (the 9th queued stream waited `37.04 s`) |
| Highest total at 48 to 64 users | **Two `mlx-vlm 0.8.0` servers** (one per Mac) | `334.7 tok/s` at 64 users, `0` errors |
| Multiple `~8k` prompts at once | **2 Macs `GLM-Tandem`** | `54.1 tok/s`, `18.75 s` median first token (8 x 8k) |

---

### What was tested

- **Model:** [GLM-5.3-Flash](https://huggingface.co/zai-org) by Z.ai — **320B total / 18B active** Mixture-of-Experts, 288 routed experts (8 active + 1 shared per token), a built-in **MTP head for speculative decoding**, 1M-token context window; run as TensorFold's **4-bit MLX** pack (169 GB on disk).
- **Devices:** **2 × Apple Mac Studio, M5 Ultra, 512 GB unified memory each**, macOS 27.2, linked by one **Thunderbolt 5** cable with **RDMA**.
- **Software:** stock **Apple MLX 0.32.4** (no patches), **TensorFold 1.2.10**, **GLM-Tandem** (two-Mac tensor parallelism), **mlx-vlm 0.8.0**; all served over **OpenAI-compatible** APIs.
- **Topics:** local LLM · on-device AI · Apple Silicon · MLX · MoE · speculative decoding · tensor parallelism · RDMA over Thunderbolt 5 · 4-bit quantization · multi-user serving · long context.

### About these results

- **TensorFold's pack underlies every result.** All engines tested here (`TensorFold`, `GLM-Tandem`, `mlx-vlm` and our shard overlay) loaded `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP`; thank you to TensorFold, Z.ai, icryo, Blaizzy, jundot and the MLX team ([`CREDITS.md`](CREDITS.md)).
- **All figures in this README are `MEASURED`** in our lab logs (rounds up to Round 7, 19-20 Nov 2026) (with exact receipt filenames in [`BENCHMARKS.md`](BENCHMARKS.md)).
- **Precision & determinism labels:** Native 4-bit MLX affine (`group_size=64`) is always labelled **`4-bit, lossy`**. Speculative draft decoding (`MTP`) and cross-engine comparisons produce **`same quality, not bit-identical at T=0`** due to `bfloat16` near-tie rounding.
- **Where things are:** scrubbed per-request data is in [`receipts/`](receipts/README.md), a small clean-room bench client is in [`scripts/bench_client.py`](scripts/bench_client.py), and [`BENCHMARKS.md`](BENCHMARKS.md) has all full tables, external `CLAIMED` numbers, and `ESTIMATE`s.

---

---

## Headline Numbers (`MEASURED` Only)

Hardware: Two identical **Apple Mac Studios (`Mac17,15`, M5 Ultra, 512 GiB RAM each — `Mac A` and `Mac B`)**, macOS `27.2` build `26B118`, `iogpu.wired_limit_mb=0`, stock `mlx==0.32.4` (`MLX_FENCE_MODE` unset). Model pack: `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` (`@ d2b17f5253440224422b150705c1109a9a3a0368`).

### One-User Speed (`T=0`, `Reasoning Effort: Low`, Quiet Macs)

![Figure 1: single-user speed](charts/1-single-user-speed.png)


| Setup & Precision | Speed (tok/s) | MTP Active? | Peak Mem / Mac | Key Receipt |
|---|---|---|---|---|
| **2 Macs `GLM-Tandem`** (`@ 1a703421`, `fast_synch=1`, `TP_MTP=1`, `4-bit, lossy`) — same-day `A-B-C-C-B-A` | **`137.1`** median (`116.7–144.0` per req; **`2.77x`** vs plain pair) | **Yes** (`74.2%` accept) | `97.0–99.4 GB` phys | `result-r4b-gate2-abccba.json` |
| **2 Macs `GLM-Tandem`** 30-min soak (`309` turns, `179,387` tok, `4-bit, lossy`) | **`136.0`** median (`129.6` agg, `0` stalls); `14k`/`54k` read `2,033`/`2,174`, write `132.4`/`128.0` | **Yes** (`63–93%`) | `100.4 GB` phys | `result-tandem-r3-soak.json`, `result-tandem-r3-long.json` |
| **2 Macs `GLM-Tandem`** (`TP_MTP=0`, `fast_synch=1`, plain greedy, `4-bit, lossy`) | **`104.6`** median (**`2.11x`** vs plain pair) | No (plain) | `93.8–95.4 GB` phys | `result-r4b-gate2-abccba.json` |
| **1 Mac `TensorFold v1.2.10`** (`Mac A`, `4-bit, lossy`) — same-day `A-B-B-A` | **`90.1`** mean of arm medians (`89.5` Run-2 short med, max `99.2`; **`2.10x`** vs 1-Mac `mlx-vlm`; `128k` write `87.8`) | **Yes** (`64.7–65.6%`) | `189.2–189.5 GB` phys | `tf1-run5-abba.json`, `tf1-run1-2.json`, `tf1-run3.json` |
| **2 Macs `mlx-vlm 0.8.0` + shard overlay** (`fast_synch=1`, `4-bit, lossy`) | **`49.5`** same-day median (`49.8` quiet pass) | No | `90.8 GB` phys (`94.7 GB` mlx) | `result-r4b-gate2-abccba.json`, `twomac-quiet-1119.log` |
| **1 Mac `mlx-vlm 0.8.0` + overlay** (`4-bit, lossy`) | **`43.0`** same-day `A-B-B-A` median | No | `178.5–184.9 GB` | `tf1-run5-abba.json` |
| **1 Mac vs 2 Macs FP8->MXFP8** (`dealignai` `@ d5822b06`, default sync) | **1 Mac `26.3–26.5`** vs **2 Macs `21.7–21.8`** (2 Macs reach `0.82x` of 1 Mac here; `fast_synch=1` stalls) | No | `334.9 GB` (1 Mac) / `178.7 GB` mlx/Mac (2 Macs) | `onemac-fp8-mac-b-2026-11-19.log`, `twomac-quiet-1119.log` |
| **2 Macs `GLM-Tandem`**, mixed 24 prompts, `Default` (`Max`) effort, `T=0.7`/`0.0`, `max_tokens=1,024` (`4-bit, lossy`, Round 6) | **`121.23`** end-to-end median (`93.49–141.99`); **`136.12`** write median; **`2.86x`** vs 1-Mac `mlx-vlm` (**`3.03x`** paired), **`1.47x`** vs 1-Mac `TensorFold` (**`1.49x`** paired) | **Yes** (`81.5%` head accept) | `99.07 / 98.21 GB` phys | `result-r6-partB.json` |
| **Same, repeated with 3 fresh restarts** (`T=0`, `max_tokens=512`, `4-bit, lossy`, Round 7) | **`119.9`** E2E / **`141.4`** decode median; **`P/T1` = `1.47x [1.43–1.49]`** over 1-Mac `TensorFold` (`81.7` E2E); `P/Q` = `1.25x [1.22–1.30]` (drafts off `96.2`); 1-Mac `mlx-vlm` `41.7` | **Yes** (`81.0%`, `41,616/51,402`) | `98.3 / 100.0 GB` phys | `result-r7-part1.json` |
| **1 Mac `TensorFold v1.2.10`**, same mixed 24 prompts (`Default` effort, Round 6) | **`82.32`** E2E median / `89.04` write; 1-Mac `mlx-vlm` `42.45` E2E | **Yes** (`69.7%`) | `190.98 GB` phys | `result-r6-partB.json` |

### Many-Users Total (`4-bit, lossy`, `T=0`, `max_tokens=256`, `cached=0`, Quiet Macs)

![Figure 2: many users at once](charts/2-many-users.png)


| Setup | Concurrency | **Aggregate E2E tok/s** (`Decode-Span`) | Per-User Med / Min tok/s | TTFT Med / Max (s) | Key Receipt |
|---|---|---|---|---|---|
| **Dual `mlx-vlm 0.8.0` servers** (`1` copy/Mac, `16+16`, no MTP) | `32` short users | **`265.7`** (`272.9`) — `0` errors | `9.19` / `8.76` | `2.03` / `3.54` | `result-r5-armB.json` |
| **2 Macs `GLM-Tandem`** (`TP_CONC=32`, pre-connected sockets, `fast_synch=1`) | `32` short users | **`259.6`** (`264.7`) — plateaus `8–16` (`205–212`); raw burst drops at `listen(5)` | `10.27` / `9.49` | `5.36` / `11.11` | `result-r5-armA-preconn.json`, `result-r5-armA.json` |
| **1 Mac `TensorFold v1.2.10`** (`--parallel 16 --max-inflight 32`, MTP auto-off) | `16` short users | **`137.9`** (`151.4`) — default 8 streams plateau at `121.4–122.5` | `9.47` / `9.47` | `2.66` / `2.66` | `result-r5-armC.json` |
| **2 Macs `GLM-Tandem`** (`8` concurrent `~8k` prompts) | `8 x ~8k` users | **`54.9`** (`64.5`) — vs `29.9` dual `mlx-vlm` & `24.7` 1-Mac `TensorFold` | `13.88` / `8.32` | `18.41` / `29.57` | `result-r5-armA-preconn.json` |
| **One `TensorFold v1.2.10` per Mac** (2 servers, `--parallel 8`, Round 7, `Effort: Low`) | `16` short users | **`243.0`** — `0` errors | `16.0` / `15.2` | `1.57` p50 / `1.59` max | `result-r7-part2.json` |
| **One `TensorFold v1.2.10` per Mac** (`--parallel 16 --max-inflight 32`, MTP off at this load) | `32` short users | **`271.8`** — `0` errors, raw burst `32/32` | `8.5` / `8.5` | `3.02` p50 / `3.07` max | `result-r7-part2.json` |
| **Dual `mlx-vlm 0.8.0`** (`--max-num-seqs 32`, 1 per Mac) | `64` short users | **`334.7`** — `0` errors, still climbing | `5.3` / `5.2` | `4.38` p50 / `7.69` max | `result-r7-part2.json` |
| **2 Macs `GLM-Tandem`** (`TP_MTP=1`, pre-connected sockets; drafts off above 4 streams) | `32` short users | **`248.3`** (crossover: from `20` users two independent `mlx-vlm` servers give the higher total) | `8.1` / `7.8` | `6.02` p50 / `12.31` max | `result-r7-part2.json` |
| **2 Macs `GLM-Tandem`** (`8 x ~8k`, Round 7) | `8 x ~8k` users | **`54.1`** — still the leader (no crossover up to `8 x 8k`) | `6.8` / `6.8` | `18.75` p50 / `30.09` max | `result-r7-part2.json` |

**Takeaway:** For **one user**, `GLM-Tandem` (`TP_MTP=1`, fast sync, stock `mlx 0.32.4`) on two Macs gives the fastest speed: **`137.1 tok/s`** on short prompts (`2.77x` the plain two-Mac pair; `104.6 tok/s` even with drafting off) and **`121.23 tok/s` end-to-end** on a realistic mixed set at `Default` effort, about **`1.5x` over the best one-Mac engine** (`TensorFold v1.2.10`, `1.47x [1.43–1.49]` over three fresh restarts, and not an artefact of prompt-lookup copying). For **many short-prompt users (2 to 32)**, one `TensorFold v1.2.10` per Mac is best (`243.0 tok/s` at 16 users, `271.8` at 32); from `20` users two independent `mlx-vlm` servers give a higher total than the two-Mac `GLM-Tandem` pair and keep scaling, reaching the highest total at 48 to 64 users (`334.7` at 64); for **multiple `8k` prompts**, two-Mac tensor-parallel `GLM-Tandem` leads (`54.1 tok/s`, `18.75 s` median TTFT).

### Quality Checks (`4-bit, lossy`, `MEASURED`)

![Figure 3: maths accuracy and speed](charts/4-gsm8k-accuracy-speed.png)


| Check | Result | Receipt |
|---|---|---|
| **Round 6**: generated 200-question suite, `T=0`, max `4,096` tokens: `GLM-Tandem` drafts off / drafts on / 1-Mac `TensorFold` | **`197/200` (`98.5%`)** / **`192/200` (`96.0%`)** / **`197/200` (`98.5%`)** | `result-r6-partA.json` |
| **Round 6**: public GSM8K, 200 test questions, same arms | **`196/200` (`98.0%`)** / **`198/200` (`99.0%`)** / **`197/200` (`98.5%`)** | `result-r6-partA.json` |
| **Round 6**: all `400` questions, drafts off vs on | `393/400` vs `390/400`; `10` answers differed (`6` A-right, `3` B-right); exact McNemar `p = 0.5078`: no significant difference | `result-r6-partA.json` |
| **Round 7**: sampled GSM8K, `T=0.6`, `5` samples x `100` questions, drafts off / on | Majority vote **`100/100`** both; per-sample mean **`98.8%`** vs **`99.2%`**; McNemar `p=1.0000`; drafts on `29.1%` faster | `result-r7-part5.json` |
| **Round 7**: 4-bit vs 8-bit loss (`mlx_lm.perplexity`) | **Could not run**: `mlx_lm` does not support model type `glm5_next`. The loss of 4-bit is **unmeasured** | `result-r7-part4.json` |

### A Long Prompt Delays Everyone Else (`MEASURED`, `4-bit, lossy`)

![Figure 4: waiting time beside a long prompt](charts/3-long-prompt-fairness.png)


One `31,995`-token prompt next to `8` short prompts delays short users by **`7.65 s` median (`37.04 s` for the 9th queued stream) on one-Mac `TensorFold`**, **`15.18 s` on two-Mac `GLM-Tandem`**, and **`36.33 s` on one-Mac `mlx-vlm 0.8.0`**, which also used up to `485.3 GB` physical memory (our `80%` memory watchdog fired `6` times in this test). Setting `mlx-vlm`'s `--prefill-step-size` to `512` gave `50.23 s`, so we do not recommend it. `TensorFold` has the shortest median wait because it splits long prompts into chunks; the `37.04 s` for the 9th queued stream is the worst case in its pool. Details: [`BENCHMARKS.md`](BENCHMARKS.md) Section 2.4.

### What We Could Not Measure or Do

- The 4-bit vs 8-bit quality loss (see above), HumanEval, `exo` on our Macs, and wall-power readings: skipped or impossible.
- `mlx-vlm 0.8.0`'s own cluster mode does not work for this model (no shard implementation), so our plain two-Mac reference is our own clean-room shard overlay.
- Prompt-lookup drafting is not what makes the headline fast on natural-language prompts: `P/PL = 1.01 [0.96–1.02]`.
- `T=0.7` outputs on `GLM-Tandem` were not reproducible between visits (`0/24`); `T=0` outputs were (`24/24`).
- Time-to-answer could not be measured for five of six prompt types at `max_tokens=512` (the model was still thinking).
- Two-Mac `MXFP8` reached `0.82x` of one Mac, and fast sync stalls on it; see [`LESSONS.md`](LESSONS.md).

---

## Repository Guide

- [`BENCHMARKS.md`](BENCHMARKS.md) — Every one-user and many-user table, exact macOS build/pins/pack revisions, `MEASURED` receipts, and separate `CLAIMED` / `ESTIMATE` context.
- [`LESSONS.md`](LESSONS.md) — Plain-language postmortems of every failure and fix (fast-sync stalls on 8-bit/MXFP8, MXFP8 pair result, confirming MTP loaded, busy-Mac skew, `bf16` near-tie non-identity, `listen(5)` drops, manifest pitfalls, isolation rules, drafting switching off under load, long-prompt starvation, `mlx-vlm` cluster mode, unmeasurable 4-bit loss).
- [`AGENTS.md`](AGENTS.md) — Step-by-step agent runbook: preflight checks, exact pins and SHA-256s, outlined commands, watchdogs, stop rules, and receipt schema.
- [`CREDITS.md`](CREDITS.md) — Full attribution for models, software, PRs/issues, community researchers, and team roles.

---

## How to Reproduce

Point your coding agent (or follow manually) at [`AGENTS.md`](AGENTS.md). In brief:

1. **Preflight both Macs:** Confirm macOS `27.2` (`26B118`), `iogpu.wired_limit_mb=0`, Thunderbolt Bridge (`bridge0`) OFF, `rdma_en4` active, exclusive quiet Macs, and verified model manifests (`TensorFold/GLM-5.3-Flash-MLX-4bit-MTP @ d2b17f52`, `45` SHA-256 + `9` size rows, `bad=0`, plus `config.json` SHA-256 check).
2. **Verify RDMA sync:** Run `sync_bench` over `jaccl` (`~19 us` fast-sync `all_sum` vs `~188 us` default) and a 12,000-step fast-sync soak (`0` hangs).
3. **Run benchmarks one at a time:** Keep stock `mlx==0.32.4` (`MLX_FENCE_MODE` unset), bind every HTTP server strictly to `127.0.0.1`, **never** enable `MLX_METAL_FAST_SYNCH=1` on 8-bit/FP8/MXFP8 graphs, verify MTP acceptance in both startup logs and per-request telemetry, enforce `30 s` real-token decode watchdogs + memory caps, and remove any temporary Mac-to-Mac SSH keys on exit.
4. **Compare same-day:** Use interleaved `A-B-B-A` / `A-B-C-C-B-A` passes on fresh servers (`cached_tokens=0`) and record the full receipt schema in [`AGENTS.md`](AGENTS.md).

---

## Credits

**Thank you** to everyone whose work made this possible. See [`CREDITS.md`](CREDITS.md) for the complete list of models, engines, datasets, issues, PRs, and community members. Special thanks to **TensorFold** (the pack every test ran on, and the fastest one-Mac engine), Z.ai, icryo, Blaizzy, jundot and Apple's MLX team.

In particular, the `117–143 tok/s` (short prompts) two-Mac MTP runtime uses [`icryo/GLM-Tandem`](https://github.com/icryo/GLM-Tandem) (`@ 1a703421`, with [`oMLX`](https://github.com/jundot/omlx) `0.8.0rc1` and stock `mlx 0.32.4`). **`icryo/GLM-Tandem` has no licence**: we credit and link to it only and include none of its code or patches. None of the credited upstream projects endorsed these results.

---

## Licence

This repository (documentation, charts, receipts and scripts) is released under the **Apache License 2.0**; see [`LICENSE`](LICENSE). Third-party models and software that we link or credit keep their own licences. `icryo/GLM-Tandem` has no licence and is not included in this repository. Check the model card licence of every model pack before reuse; the FP8 pack used for the MXFP8 runs (`dealignai/GLM-5.3-Flash-UNCENSORED-FP8`) is reported here for speed only.
