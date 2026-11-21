# Benchmarks: GLM-5.3-Flash on One and Two Mac Studios (M5 Ultra, 512 GB)

> Every 4-bit result in this document used the TensorFold `GLM-5.3-Flash-MLX-4bit-MTP` pack (the FP8 rows used the `dealignai` FP8 pack); see [`CREDITS.md`](CREDITS.md). The question answered here is which engine is best for which job.

All benchmarks were collected on **19 and 20 Nov 2026** (Sections 1.1-1.5 and 2.1-2.2 on 19 Nov; Round 6 and Round 7, Sections 1.6, 1.7, 2.3, 2.4 and 3, on 20 Nov) across two identical **Apple Mac Studios (`Mac17,15`, Apple M5 Ultra, 36 CPU cores: 12P+24E, 512 GiB / 549,755,813,888 bytes unified RAM each)** called **Mac A** and **Mac B**, running **macOS 27.2 (build `26B118`)** with `iogpu.wired_limit_mb=0` (OS default), connected over one **Thunderbolt 5** cable (`rdma_en4` <-> `rdma_en4`, Thunderbolt Bridge / `bridge0` off, RDMA enabled).

**The model:** GLM-5.3-Flash is a mixture-of-experts model with **320B total, 18B active (Hugging Face tensor count ~321B)** parameters (from the pack's own model card, rev `d2b17f52`): 45 layers, 288 routed experts (8 active per token plus 1 shared), 1 MTP (multi-token prediction) layer, 4-bit affine group 64, 1,048,576 max positions, 169 GiB on disk. Earlier internal notes called it "~355B"; that was wrong.

Every row is labelled:
- **`MEASURED`**: Directly measured in our lab logs, citing the receipt filename.
- **`CLAIMED`**: Someone else's public number, citing a link (never compared as an identical setup unless hardware, model size, and quantization match).
- **`ESTIMATE`**: Derived projection or analytical model, never measured directly.

> **Important notes on precision and determinism**
> - **4-bit is lossy.** Native 4-bit MLX affine (`group_size=64`) is always labelled `4-bit, lossy`.
> - **FP8 runs through `mlx-vlm` are requantized to MXFP8 at load.** They are labelled `loaded from FP8, run as MXFP8` (not native FP8).
> - **Draft decoding is same quality, not bit-identical at `T=0`.** Within any single mode, repeated runs and streamed vs non-streamed calls are deterministic. Across `TP_MTP=0` vs `TP_MTP=1` (and across engines `mlx-vlm` vs `GLM-Tandem`), `bfloat16` accumulation order likely flips near-tie `argmax` choices (inferred, not logged per token) after 14–244 tokens while preserving answer quality.
> - **Throughput definitions:** `Client read tok/s` = `prompt_tokens / TTFT`. `Client write tok/s` = `completion_tokens / (t_end - t_first_tok)`. For multi-user batches, `Aggregate E2E tok/s` = `sum(completion_tokens) / (max(t_end) - min(t_start))` and `Aggregate decode-span tok/s` = `sum(completion_tokens) / (max(t_end) - min(t_first))`.

---

## Common Software Pins & Model Pack Revisions

| Component | Exact Pin / Revision | SHA-256 / Verification | Notes |
|---|---|---|---|
| **macOS** | macOS `27.2`, build `26B118` | `iogpu.wired_limit_mb=0` | Same build on Mac A and Mac B |
| **Stock MLX** | `mlx==0.32.4`, `mlx-metal==0.32.4` | Stock wheels; `MLX_FENCE_MODE` unset | No custom C++ MLX patches used |
| **mlx-vlm env** | `mlx-vlm==0.8.0` (`Blaizzy/mlx-vlm`, MIT), `mlx-lm==0.32.2`, Python `3.13.18` | `uv pip freeze` `43474b7e63833fda26229d44e60d93d57a6de5f57e3d233d08a31b19c7f00dfa` | Repo: `Blaizzy/mlx-vlm` |
| **2-Mac shard overlay** | Clean-room `glm5_next` `shard()` overlay (`PATCH.diff`) | `b288a0288756a23696a39ad4926cffb92f5e9f78dd8bc78aecf8b916e4993f5f` | Applied to copy overlay only |
| **GLM-Tandem** | `https://github.com/icryo/GLM-Tandem` `@ 1a70342137f2650367b8ca2619e19ea854a7658b` | Bind `127.0.0.1` diff `af8d4ec83c73824c0468f5aa91d2c3b77c96679715ad8232be13c0f9cf2557e1` | No licence: linked only, no code copied |
| **oMLX wheel** | `omlx==0.8.0rc1` (`omlx-0.8.0rc1-cp313-cp313-macosx_15_0_universal2.whl`) | `a416a587d12116858bfbbe67530012ce1c368aae108cfee406b638373ebf7af0` | Pulls `mlx-lm @ 0d4d3d04` (`0.32.3.dev8+g0d4d3d04f`), `mlx-vlm @ 138d7617`, stock `mlx==0.32.4` |
| **TensorFold** | `tensorfold-native` `v1.2.10` (`https://github.com/ashhart/TensorFold`, Apache-2.0) | Tarball `76eb33cc...21e58`; `SHA256SUMS` 23/23; binary `ef0ab81eb11496956cf33eba9fd16d26b3500be54f6c93b08885d667c1af19b6` | Run on Mac A only (one-Mac) |
| **4-bit MTP pack** | `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` (MIT) | Rev `d2b17f5253440224422b150705c1109a9a3a0368`; 54 rows (`45` SHA-256 + `9` size, `0` bad) | 4-bit affine `group_size=64`, 45 layers + 1 MTP layer, 169 GiB / 43 shards |
| **Uncensored FP8 pack** | `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` | Rev `d5822b068da8ddb301665b48c58718e6e18a2c3a`; 74 rows (`63` SHA-256 + `11` size, `0` bad) | Requantized to MXFP8 at load; 306 GiB / 62 shards; speed-only results for this pack; check its model card licence before reuse |
| **Official FP8 pack** | `zai-org/GLM-5.3-Flash` (MIT) | Rev `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`; 74 rows (`63` SHA-256 + `11` size, `0` bad) | 328 GB / 62 shards |
| **Converted `oQ8-MTP`** | Local conversion from `zai-org/GLM-5.3-Flash` (`46a050be`) | 67 shards, 319 GB; `config.json` `20ad12d3...bfb7`, `index.json` `5c9c3412...91d0` | 8-bit affine `group_size=64`, 59 `mtp.*` tensors, `lm_head` unquantized |
| **Round 6/7 prompt sets** | `r6-real.json` (24 varied prompts, Round 6 Part B and Round 7 Part 1) | SHA-256 `825c8bc84f7709a44497acd491b47c2156079e4e8f980b3bfd2c458ee89aed10` | Input length 46-7,969 tokens |
| **Round 7 many-user prompts** | `r5-prompts.json` (`short_32`, `conc_8k`) | SHA-256 `4d3a622d61bbaedb669222fa6bbf22eb6cab670f4a5a382b1fb595172611c122` | Reused above 32 users with a unique per-user prefix so `cached_tokens=0` |
| **RFC 9110 corpus** | IETF RFC 9110 *HTTP Semantics* (`rfc9110.txt`, full file) | Full: `21c1cdce6ab0e5509b04d84a28000836c7a087cf786efe6f04877ebfff47232a`; first 200,000 bytes: `0c7d3f5fab58e6be647c5e979e0ddcb0a2526a4d627a14b93b52f880a9e08d55` | Long-prompt and perplexity-attempt text |
| **GSM8K** | `openai/gsm8k` test parquet (MIT) | `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59`; 200-item sample `seed=20261120` (Round 6); first 100 items file SHA-256 `7e5802845a8ea84a106506aa84134d95f847638d7fbb1364ce8f3079291a999e` (Round 7) | Public benchmark, never edited |

---

## Section 1: One-User Speed (Single-Stream)

### 1.1 Headline Same-Day Two-Mac `A-B-C-C-B-A` Comparison (`4-bit, lossy`, Quiet Macs)

- **macOS build:** `26B118` (Mac A & Mac B)
- **Pins:** Stock `mlx==0.32.4` (`MLX_METAL_FAST_SYNCH=1`, `MLX_FENCE_MODE` unset); Arm A = `mlx-vlm 0.8.0` + shard overlay (`b288a028...3f5f`); Arms B & C = `icryo/GLM-Tandem @ 1a70342137f2` + `omlx 0.8.0rc1` (`a416a587...7af0`), `127.0.0.1` only
- **Pack revision:** `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` `@ d2b17f5253440224422b150705c1109a9a3a0368`
- **Note:** Earlier long-context runs on one Mac used a private text and are not published. The long-context rows below use public RFC 9110 (or, for `lc-128k`, a Shakespeare text file that is not included here; check its licence before reuse).
- **Protocol:** Strict `A1 -> B1 -> C1 -> C2 -> B2 -> A2` order, fresh cluster per run, `T=0.0`, `Reasoning Effort: Low`, `max_tokens=2048`, 3 short prompts x 5 reps x 2 runs/arm (`30` timed requests per arm; `0` stalls; `4-bit, lossy, two Macs, fast_synch=1, stock MLX 0.32.4`).

| Arm & Engine | MTP Active (Accept Rate) | Mean of Run Medians (tok/s) | Pooled 30-Rep Median (IQR, Min–Max) | Token-Weighted Aggregate (tok/s) | Pair Spread | Ratio vs A (Medians) | Peak Mem / Mac (`phys` / `sys_used`) | Label | Receipt |
|---|---|---|---|---|---|---|---|---|---|
| **A:** `mlx-vlm 0.8.0` + shard overlay (plain greedy) | No (n/a) | **49.52** (`49.68` / `49.35`) | `49.42` (`0.39`, `48.90–49.99`) | `49.36` | `0.68%` | `1.00x` | `90.8 GB` / `116.3–119.2 GB` (`mlx 94.7 GB`) | `MEASURED` | `result-r4b-gate2-abccba.json` |
| **B:** `GLM-Tandem` (`TP_MTP=0, TP_LOOKUP=0`, plain greedy) | No (n/a) | **104.61** (`104.87` / `104.35`) | `104.54` (`0.64`, `103.88–105.90`) | `104.63` | `0.50%` | **`2.11x`** (`2.113x`) | `93.8–95.5 GB` / `119.9–122.1 GB` | `MEASURED` | `result-r4b-gate2-abccba.json` |
| **C:** `GLM-Tandem` (`TP_MTP=1, TP_LOOKUP=1`, draft decoding) | **Yes** (`74.17%`, `4710/6350`) | **137.13** (`137.21` / `137.05`) | `137.20` (`25.98`, `116.65–144.02`) | `127.09` | `0.12%` | **`2.77x`** (`2.769x` vs A; `1.311x` vs B) | `97.0–99.4 GB` / `125.1–125.9 GB` | `MEASURED` | `result-r4b-gate2-abccba.json` |

#### Per-Prompt Breakdown (`A-B-C-C-B-A`, 10 Timed Reps per Prompt per Arm)

| Prompt (`in_tok`) | Arm A (`out`, Med / Min–Max, Cold->Warm Read) | Arm B (`out`, Med / Min–Max, Cold->Warm Read) | Arm C (`out`, Med / Min–Max, Cold->Warm Read, MTP Acc) | `B/A` | `C/A` | Token Identity (`A==B` / `B==C`) | Label | Receipt |
|---|---|---|---|---|---|---|---|---|
| `english-reasoning` (`61`) | `189`, `49.56` (`48.97–49.99`, `97–100 -> 360`) | `253`, `104.32` (`103.88–105.29`, `136–142 -> 831`) | `261`, **`143.23`** (`142.43–144.02`, `129–135 -> 745`, `87.50%`) | `2.105x` | `2.890x` | No (`@14`) / No (`@18`); same quality | `MEASURED` | `result-r4b-gate2-abccba.json` |
| `arabic` (`36`) | `227`, `49.55` (`49.35–49.90`, `262–265 -> 263`) | `210`, `105.27` (`104.45–105.90`, `145–152 -> 431`) | `225`, **`137.13`** (`136.19–137.78`, `139–147 -> 318`, `85.16%`) | `2.123x` | `2.765x` | No (`@24`) / No (`@24`); same quality | `MEASURED` | `result-r4b-gate2-abccba.json` |
| `prose-400` (`24`) | `505`, `49.22` (`48.90–49.55`, `204 -> 204`) | `515`, `104.48` (`104.18–105.17`, `109–114 -> 274`) | `525`, **`116.85`** (`116.65–117.22`, `99–104 -> 219`, `63.42%`) | `2.123x` | `2.374x` | No (`@75`) / No (`@15`); same quality | `MEASURED` | `result-r4b-gate2-abccba.json` |

#### Per-Run Telemetry (`A1 -> B1 -> C1 -> C2 -> B2 -> A2`)

| Run | Arm & Config | Load (s) | MTP Head Loaded (Accept) | Write Med (`Min–Max`, `IQR`) | Aggregate | Peak Mem `Mac A` (`phys` / `lmax` / `sys`) | Peak Mem `Mac B` (`phys` / `lmax` / `sys`) | Label | Receipt |
|---|---|---|---|---|---|---|---|---|---|
| `A1` | `mlx-vlm 0.8.0+overlay` (plain) | `14.77` | No (n/a) | **`49.68`** (`49.22–49.99`, `0.37`) | `49.56` | `90.81` / `96.82` / `119.21 GB` | `90.80` / `96.87` / `116.40 GB` | `MEASURED` | `result-r4b-abba-A1.json` |
| `B1` | `GLM-Tandem` (`TP_MTP=0, TP_LOOKUP=0`) | `21.50` | No (n/a) | **`104.87`** (`104.23–105.90`, `0.83`) | `104.87` | `93.82` / `98.17` / `121.91 GB` | `95.45` / `98.16` / `121.02 GB` | `MEASURED` | `result-r4b-abba-B1.json` |
| `C1` | `GLM-Tandem` (`TP_MTP=1, TP_LOOKUP=1`) | `22.20` | **Yes** (`74.17%` head, `50.0%` lookup) | **`137.21`** (`116.69–144.02`, `26.17`) | `127.18` | `97.10` / `102.36` / `125.88 GB` | `99.38` / `102.35` / `125.52 GB` | `MEASURED` | `result-r4b-abba-C1.json` |
| `C2` | `GLM-Tandem` (`TP_MTP=1, TP_LOOKUP=1`) | `22.14` | **Yes** (`74.17%` head, `50.0%` lookup) | **`137.05`** (`116.65–143.57`, `26.09`) | `127.00` | `97.03` / `102.37` / `125.37 GB` | `99.44` / `102.36` / `125.13 GB` | `MEASURED` | `result-r4b-abba-C2.json` |
| `B2` | `GLM-Tandem` (`TP_MTP=0, TP_LOOKUP=0`) | `22.13` | No (n/a) | **`104.35`** (`103.88–105.25`, `0.62`) | `104.39` | `94.04` / `98.16` / `122.06 GB` | `93.99` / `98.14` / `119.87 GB` | `MEASURED` | `result-r4b-abba-B2.json` |
| `A2` | `mlx-vlm 0.8.0+overlay` (plain) | `14.85` | No (n/a) | **`49.35`** (`48.90–49.53`, `0.39`) | `49.17` | `90.81` / `96.86` / `119.12 GB` | `90.80` / `96.83` / `116.30 GB` | `MEASURED` | `result-r4b-abba-A2.json` |

---

### 1.2 Headline Same-Day One-Mac `A-B-B-A` & Single-Stream (`4-bit, lossy`, Mac A, Quiet)

- **macOS build:** `26B118` (Mac A)
- **Pins:** `TensorFold v1.2.10` (`ef0ab81e...19b6`, `--reasoning-effort low --temperature 0 --max-tokens 2048`) vs `mlx-vlm 0.8.0` + overlay (`b288a028...3f5f`), stock `mlx==0.32.4`
- **Pack revision:** `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` `@ d2b17f5253440224422b150705c1109a9a3a0368`

#### One-Mac Same-Day `A-B-B-A` (`TensorFold -> mlx-vlm -> mlx-vlm -> TensorFold`, 3 Short Prompts x 5 Reps/Arm)

| Run | Engine & Setup | MTP Active (Accept) | Load (s) | English Med / Max | Arabic Med / Max | Prose-400 Med / Max | Arm Median (tok/s) | Arm Aggregate | Peak Mem (`phys` / `sys` / `mlx`) | Label | Receipt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A1** | `TensorFold v1.2.10` (1 Mac, MTP) | **Yes** (`65.60%`) | `22.21` | `97.54` / `98.38` | `89.17` / `90.84` | `75.96` / `76.11` | **89.17** | `82.95` | `189.21` / `217.35` / — GB | `MEASURED` | `tf1-run5-abba.json` |
| **B1** | `mlx-vlm 0.8.0 + overlay` (1 Mac, plain) | No | `14.38` | `42.96` / `43.09` | `43.00` / `43.08` | `42.81` / `42.91` | **42.94** | `42.88` | `178.71` / `210.97` / `178.38` GB | `MEASURED` | `tf1-run5-abba.json` |
| **B2** | `mlx-vlm 0.8.0 + overlay` (1 Mac, plain) | No | `5.86` | `43.17` / `43.22` | `43.01` / `43.05` | `42.86` / `42.93` | **42.97** | `42.95` | `184.93` / `212.85` / `181.10` GB | `MEASURED` | `tf1-run5-abba.json` |
| **A2** | `TensorFold v1.2.10` (1 Mac, MTP) | **Yes** (`65.32%`) | `20.66` | `95.91` / `96.50` | `90.96` / `91.04` | `76.25` / `76.70` | **90.96** | `82.84` | `189.21` / `217.35` / — GB | `MEASURED` | `tf1-run5-abba.json` |
| **Summary** | `TensorFold` vs `mlx-vlm` (mean of medians) | `~65.5%` vs n/a | — | **`2.25x`** (`96.72` vs `43.07`) | **`2.09x`** (`90.06` vs `43.00`) | **`1.78x`** (`76.11` vs `42.83`) | **`2.10x`** (`90.06` vs `42.95`) | **`1.93x`** (`82.90` vs `42.91`) | Spread: `A=1.99%`, `B=0.08%` | `MEASURED` | `tf1-run5-abba.json` |

#### `TensorFold v1.2.10` One-Mac 5-Rep Single-Stream Detail (`Mac A`, MTP Loaded Confirmed)

| Prompt (`in_tok`, `cached`) | Out Tok (`finish`) | Read tok/s (Cold / Med / Max) | Write tok/s (Min / Med / Max Client) | Server tok/s Med / Max | MTP Accept Med (`acc/drafted` Sum) | Max Inter-Tok Gap | Label | Receipt |
|---|---|---|---|---|---|---|---|---|
| `english-reasoning` (`61`, `0`) | `241` (`stop`) | `366.7` / `377.2` / `380.0` | `95.74` / **96.21** / **99.18** | `95.81` / `98.77` | `73.13%` (`740/1000`) | `0.065 s` | `MEASURED` | `tf1-run1-2.json` |
| `arabic` (`36`, `0`) | `213` (`stop`) | `246.5` / `245.8` / `246.5` | `89.07` / **89.46** / **90.85** | `88.13` / `89.49` | `64.88%` (`548/839`) | `0.031 s` | `MEASURED` | `tf1-run1-2.json` |
| `prose-400` (`24`, `0`) | `547` (`stop`) | `181.7` / `182.0` / `182.9` | `73.69` / **76.06** / **76.37** | `75.66` / `75.97` | `59.29%` (`800/1363`) | `0.032 s` | `MEASURED` | `tf1-run1-2.json` |
| `code-2k` (`1,990`, `0` then `1,988`) | `1,367` (`stop`) | `1,866.5` (cold) / `34,205.5` (cached) | `85.77` / **86.15** / **86.50** | `85.94` / `86.29` | `64.47%` (`3608/5600`) | `0.056 s` | `MEASURED` | `tf1-run1-2.json` |
| **Short-15 / All-20 aggregate** | — | — | Short-15 med **`89.46`** (`89.5`), max **`99.18`**; All-20 med **`87.79`** | — | `64.71%` (`5696/8802`); `189.36 GB phys` | `0` stalls (#3902 OK) | `MEASURED` | `tf1-run1-2.json`, `tf1-run1-2-startup.log` |

---

### 1.3 Precision, Sync Mode & Engine Matrix (One Mac vs Two Macs: 4-bit, MXFP8, Official FP8, oQ8-MTP)

- **macOS build:** `26B118`
- **Pins:** Stock `mlx==0.32.4` (`MLX_FENCE_MODE` unset); `mlx-vlm==0.8.0` (+ overlay `b288a028...3f5f` on 2-Mac runs); `GLM-Tandem @ 1a70342137f2` + `omlx==0.8.0rc1` (`a416a587...7af0`).
- **Pack revisions:** 4-bit MTP `@ d2b17f52`; Uncensored FP8 `@ d5822b06`; Official FP8 `@ 46a050be`; converted `oQ8-MTP` from `46a050be`.

| Setup, Precision & Hardware | Sync Mode | Prompt (`in -> out`) | Read tok/s | Write tok/s | MTP Accept | Peak Mem / Mac | Notes | Label | Receipt |
|---|---|---|---|---|---|---|---|---|---|
| 1-Mac `mlx-vlm 0.8.0`, `4-bit, lossy`, Mac B (`Effort: Max`, quiet) | n/a | EN `61->1024` / AR `36->1017` | `198.7` / `143.2` | `43.01` / `43.01` | n/a (no MTP) | `178.5 GB` mlx | Earliest quiet baseline | `MEASURED` | `one-mac-4bit-mac-b-2026-11-19-r2.log` |
| 1-Mac `mlx-vlm 0.8.0` server, `4-bit, lossy`, Mac B (`Effort: Low`) | n/a | `24->200` / `258 tok` | — | `44–45` (`36.2` e2e) | n/a | — | Served HTTP path | `MEASURED` | `concurrency-4bit-mac-b-2026-11-19.log` |
| 1-Mac `mlx-vlm + overlay`, `loaded from FP8, run as MXFP8`, Mac B (quiet) | n/a | EN `61->308` / AR `36->214` | `45.5` / `121.2` | **`26.3`** / **`26.5`** | n/a | `334.9 GB` | Faster than 2-Mac MXFP8; rev `d5822b06` | `MEASURED` | `onemac-fp8-mac-b-2026-11-19.log` |
| 2-Mac `mlx-vlm + overlay`, `4-bit, lossy`, quiet (`r1`/`r2`/`quiet`) | `fast_synch=1` | EN `61->189` / AR `36->227` | `54–320` / `262–270` | **`49.79–49.92`** (`r1`), `49.39–49.45` (`r2`), **`49.80–49.81`** (`quiet`) | n/a (no MTP) | `94.7 GB` mlx (`~112 GB` sys) | Ranks agree token-for-token | `MEASURED` | `twomac-quiet-1119.log`, `result-4bit.json`, `result-4bit-r2.json` |
| 2-Mac `mlx-vlm + overlay`, `4-bit, lossy` (Mac A busy) | `fast_synch=0` | EN `61->189` / AR `36->227` | `239.9` / `194.5` | `27.43` / `27.32` | n/a | `94.7 GB` mlx | Default fence adds `~17 ms/tok` | `MEASURED` | `result-4bit-nofastsync.json` |
| 2-Mac `mlx-vlm + overlay`, `loaded from FP8, run as MXFP8` | `fast_synch=1` | Load only (`36.76 s`) | — | **STALL (`0` tok)** | n/a | `178.7 GB` mlx (`189.6 GB` sys) | Deadlock in `Fence::wait` / `all_reduce`; rev `d5822b06` | `MEASURED` | `twomac-fp8-fastsync-stall.log` |
| 2-Mac `mlx-vlm + overlay`, `loaded from FP8, run as MXFP8`, **quiet** | `fast_synch=0` | EN `61->251` / AR `36->207` | `173.6` / `149.8` | **`21.67`** / **`21.78`** | n/a | `178.7 GB` mlx (`190.1 GB` sys) | slower than 1 Mac (`26.4`); rev `d5822b06` | `MEASURED` | `twomac-quiet-1119.log` |
| 2-Mac `mlx-vlm + overlay`, `loaded from FP8, run as MXFP8`, Mac A busy | `fast_synch=0` | EN `61->251` / AR `36->207` | `170.9` / `146.1` (`r1`), `170.2` / `122.3` (`r2`) | `21.27` / `21.48` (`r1`), `20.85` / **`14.78`** (`r2`) | n/a | `178.7 GB` mlx | Dipped to `14.78` under background load | `MEASURED` | `result-fp8.json`, `result-fp8-r2.json` |
| 2-Mac `FP8 -> MXFP8` if fast sync did not stall | `fast_synch=1` | Hypothetical | — | `~35–40` | n/a | `~179 GB` | Scaled from 4-bit 1.8x fast-sync speedup (cannot run) | `ESTIMATE` | — |
| 2-Mac `GLM-Tandem` R2, `4-bit, lossy`, MTP | `fast_synch=0` | EN `61->261` / AR `36->225` / Prose `24->525` | `14.0` / `97.1` / `52.2` | `29.97` / `36.08` / `30.29` (agg `31.32`) | `74.17%` (`471/635`) | `98.3 GB` phys (`121.4 GB` sys) | Output identical to fast sync below | `MEASURED` | `result-tandem-default.json` |
| 2-Mac `GLM-Tandem` R2, `4-bit, lossy`, MTP | `fast_synch=1` | EN `61->261` / AR `36->225` / Prose `24->525` | `147.6` / `148.8` / `110.8` | **`143.34` / `134.75` / `117.20`** (agg `126.85`) | `74.17%` (`471/635`) | `93.7–97.1 GB` phys (`118.3 GB` sys) | First pass on stock MLX `0.32.4` | `MEASURED` | `result-tandem-fastsync.json` |
| 2-Mac `GLM-Tandem` R3, Official `zai-org` FP8 (`46a050be`) | `fast_synch=0` | Startup only | — | **FAILED to load** | n/a | `373.5 / 384.0 GB` phys (`405.4 GB` sys) | Rank-1 error: `'Linear' object has no attribute 'scales'` | `MEASURED` | `result-tandem-r3-official-raw-default.json`, `tandem-rank1-r3-official-raw-default.log` |
| 2-Mac `GLM-Tandem` R3, `oQ8-MTP` (8-bit converted from `46a050be`) | `fast_synch=0` | EN `61->187` / AR `36->230` / Prose `24->513` | `18.5` / `80.2` / `47.9` | `26.40` / `25.28` / `20.11` | `96.5%` / `86.4%` / `62.2%` | `181.3–185.2 GB` phys (`206.8 GB` sys) | MTP loaded (`59` tensors); default sync only | `MEASURED` | `result-tandem-r3-official-oq8-default.json` |
| 2-Mac `GLM-Tandem` R3, `oQ8-MTP` (8-bit converted from `46a050be`) | `fast_synch=1` | Blocked at start | — | **KILLED (`exit 143`)** | n/a | `169.4 / 180.1 GB` phys | Stopped by guard: never fast sync on 8-bit | `MEASURED` | `result-tandem-r3-official-oq8-fastsync.json` |

#### Gate 1 Token Identity & 6-Prompt Comparison (`GLM-Tandem @ 1a703421`, `fast_synch=1`, Stock `MLX 0.32.4`, `T=0`, `cached=0`)

- **macOS build:** `26B118` | **Pack revision:** `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP @ d2b17f52`
- **Overall aggregate MTP accept (`ON`):** `80.65%` head (`921/1142`), `85.49%` lookup (`701/820`); `stream=True == stream=False` on `6/6` prompts in both modes; peak memory `94.1–95.9 GB` (`OFF`) / `97.5–99.4 GB` (`ON`).

| Prompt (`kind`, `in_tok`) | `OFF` (`TP_MTP=0`) Out & Write tok/s | `ON` (`TP_MTP=1, TP_LOOKUP=1`) Out & Write tok/s | MTP Head / Lookup Accept | Token-Identical (`OFF == ON`)? | First Diff Index (Quality Note) | Label | Receipt |
|---|---|---|---|---|---|---|---|
| `english-reasoning` (`short`, `61`) | `253`, `104.69` | `261`, `143.41` | `87.50%` (`147/168`) / `0/1` | **No** (`stream == non-stream`) | `@ 18` (`' GB'` vs `'.'`; same quality) | `MEASURED` | `result-r4-gate1-identity.json` |
| `arabic` (`short`, `36`) | `210`, `105.93` | `225`, `138.35` | `85.16%` (`109/128`) / `1/1` | **No** (`stream == non-stream`) | `@ 24` (`' والأ'` vs `'**:'`; same quality) | `MEASURED` | `result-r4-gate1-identity.json` |
| `prose-400` (`short`, `24`) | `515`, `104.83` | `525`, `117.47` | `63.42%` (`215/339`) / `0/0` | **No** (`stream == non-stream`) | `@ 15` (`'wr'` vs `'wing'`; same quality) | `MEASURED` | `result-r4-gate1-identity.json` |
| `code-refactor` (`code`, `1,127`) | `1,236`, `100.32` | `1,224`, **`190.97`** | `92.08%` (`279/303`) / `86.40%` (`686/794`) | **No** (`stream == non-stream`) | `@ 244` (`' """'` vs `' def'`; same quality) | `MEASURED` | `result-r4-gate1-identity.json` |
| `summary-2k` (`summary-2k`, `1,987`) | `317`, `99.77` | `325`, `134.61` | `83.76%` (`165/197`) / `57.89%` (`11/19`) | **No** (`stream == non-stream`) | `@ 61` (`' —'` vs `','`; same quality) | `MEASURED` | `result-r4-gate1-identity.json` |
| `tool-multi-turn` (`tool`, `579`) | `16`, `812.59` | `16`, `802.99` | `85.71%` (`6/7`) / `60.0%` (`3/5`) | **Yes** (`16/16` exact match) | Exact match (`tool_calls`; same quality) | `MEASURED` | `result-r4-gate1-identity.json` |

---

### 1.4 Long Context & 30-Minute Soak (`4-bit, lossy`, Single-Stream)

- **macOS build:** `26B118`
- **Pack revision:** `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` `@ d2b17f5253440224422b150705c1109a9a3a0368`

| Setup & Test Case | Sync / MTP | Prompt Tokens (`cached`) | Out Tok | Read (Prefill) tok/s (`TTFT`) | Write tok/s | MTP Accept | Peak Mem / Mac | Notes | Label | Receipt |
|---|---|---|---|---|---|---|---|---|---|---|
| 2-Mac `GLM-Tandem` R3, `lc-14k` (RFC 9110) | `fast_synch=1` / **Yes** | `13,985` (`0`) | `477` | **`2,032.8`** (`6.88 s`) | **`132.35`** | `82.57%` (`251/304`) | `98.1 GB` phys (`121.6 GB` sys) | Fresh server, stock MLX `0.32.4` | `MEASURED` | `result-tandem-r3-long.json` |
| 2-Mac `GLM-Tandem` R3, `lc-54k` (RFC 9110) | `fast_synch=1` / **Yes** | `53,990` (`0`) | `515` | **`2,174.0`** (`24.83 s`) | **`127.98`** | `83.53%` (`279/334`) | `98.6 GB` phys (`121.9 GB` sys) | Fresh server, stock MLX `0.32.4` | `MEASURED` | `result-tandem-r3-long.json` |
| 1-Mac `TensorFold v1.2.10`, Mac A, `lc-32k` (RFC 9110) | n/a / **Yes** | `31,995` (`0`) | `542` | **`2,071.6`** (`15.44 s`) | **`95.88`** | `71.00%` (`328/462`) | `189.51 GB` phys (`217.4 GB` sys) | `+0.31 GB` over idle load | `MEASURED` | `tf1-run3.json` |
| 1-Mac `TensorFold v1.2.10`, Mac A, `lc-128k` (Shakespeare) | n/a / **Yes** | `127,996` (`0`) | `398` | **`2,002.6`** (`63.91 s`) | **`87.80`** | `67.67%` (`203/300`) | `189.53 GB` phys (`217.3 GB` sys) | `+0.33 GB` over idle load | `MEASURED` | `tf1-run3.json` |
| 2-Mac `GLM-Tandem` R3 **30-min soak** (`1,809.4 s`, `309` turns) | `fast_synch=1` / **Yes** | Mix (`24–8,000` tok) | `179,387` total | `1,398–2,169` (`8k` slices) | `min 104.45` / **med `136.02`** / `max 150.25` / `agg 129.59` | `63.4%–93.2%` med by prompt | `100.4 GB` phys (`123.9 GB` sys) | **`0` stalls**; 1 unique content hash per fixed prompt | `MEASURED` | `result-tandem-r3-soak.json` |

#### 30-Minute Soak Per-Prompt Breakdown (`result-tandem-r3-soak.json`)

| Prompt | Turns | Write Min / Med / Max (tok/s) | Read Med (tok/s) | MTP Accept Med | Max Inter-Tok Gap | Unique Hashes | Label | Receipt |
|---|---|---|---|---|---|---|---|---|
| `coding-lru` | `52` | `146.64` / **`149.45`** / `150.25` | `381.2` | `93.23%` | `0.109 s` | `1` | `MEASURED` | `result-tandem-r3-soak.json` |
| `english-reasoning` | `52` | `140.21` / **`143.46`** / `144.80` | `694.4` | `87.50%` | `0.053 s` | `1` | `MEASURED` | `result-tandem-r3-soak.json` |
| `arabic` | `51` | `133.64` / **`136.48`** / `138.46` | `306.2` | `85.16%` | `0.051 s` | `1` | `MEASURED` | `result-tandem-r3-soak.json` |
| `doc-summary-8k` (16 RFC 9110 slices) | `51` | `104.45` / **`132.66`** / `139.24` | `2,118.9` | `84.15%` | `0.294 s` | `16` (1 per slice) | `MEASURED` | `result-tandem-r3-soak.json` |
| `systems-rdma` | `52` | `118.75` / **`121.22`** / `121.77` | `403.7` | `69.12%` | `0.216 s` | `1` | `MEASURED` | `result-tandem-r3-soak.json` |
| `prose-400` | `51` | `115.30` / **`116.43`** / `117.47` | `199.0` | `63.42%` | `0.047 s` | `1` | `MEASURED` | `result-tandem-r3-soak.json` |

---

### 1.5 Interconnect (`sync_bench`) & Tiny-Model Sharding Checks

- **macOS build:** `26B118` | **Pins:** Stock `mlx==0.32.4`, `jaccl` (`rdma_en4` <-> `rdma_en4`) and `ring` (tiny test)

| Test | Configuration | Measured Result | Label | Receipt |
|---|---|---|---|---|
| `sync_bench` 1 row (8 KiB, quiet, 1,000 steps) | Default fence vs `MLX_METAL_FAST_SYNCH=1` | `187.8 us/sum` default vs **`19.0 us/sum`** fast (`18.2 us` `all_sum` alone) | `MEASURED` | `sync-quiet-2026-11-19.log` |
| `sync_bench` 1 row fast-sync soak (12,000 steps) | `MLX_METAL_FAST_SYNCH=1` (Macs busy) | `0` hangs in `226 s`; **`17.7 us/sum`** added cost (`p50 7.46 -> 9.25 ms`) | `MEASURED` | `soak-fast-12000-2026-11-19.log` |
| `sync_bench` payload sweep (200 steps, Macs busy) | 1 / 4 / 16 / 128 rows (`8 KiB`–`1 MiB`) | Default `189.1 / 196.7 / 195.8 / 603.6 us` vs Fast `17.5 / 23.3 / 46.3 / 173.9 us` | `MEASURED` | `sync-sweep-prelim-2026-11-19.log` |
| Tiny 4-layer `glm5_next` sharding proof (`ring` 2-rank, float32) | Single vs 2-rank sharded; broken `all_sum` plant | `max_abs_diff = 5.96e-08` (`<= 1e-3`, pass); planted break `0.8346` (fail, caught) | `MEASURED` | `single.log`, `sharded.log`, `plant.log` |

---

### 1.6 Round 6 Part B: Realistic Mixed-Prompt Rerun (`4-bit, lossy`, `Default` Reasoning Effort, `A-B-C-D-D-C-B-A`)

- **macOS build:** `26B118` (Mac A & Mac B), `iogpu.wired_limit_mb=0`
- **Pins:** Stock `mlx 0.32.4` (`MLX_FENCE_MODE` unset); `GLM-Tandem @ 1a70342137f2650367b8ca2619e19ea854a7658b` + `omlx 0.8.0rc1` (`a416a587...7af0`, bind diff `af8d4ec83c73...57e1`); `TensorFold v1.2.10` (`ef0ab81eb114...19b6`, Mac A only); `mlx-vlm 0.8.0` (freeze `43474b7e...0dfa`, Mac A only). Every server bound to `127.0.0.1`.
- **Pack revision:** `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` `@ d2b17f5253440224422b150705c1109a9a3a0368` (manifest `ok=45 oksize=9 bad=0 rows=54`).
- **Reasoning effort:** `Default`. The pack's `chat_template.jinja` sets `reasoning_effort = 'max'` when it is omitted, so this is `Max`, not `Low`. We omitted `reasoning_effort` on every engine. Prompt token counts were identical across engines and non-empty `<think>` reasoning appeared on `384/384` requests.
- **Protocol:** `24` varied prompts (`46-7,969` input tokens, six prompt types) at `T=0.7` and `T=0.0`, max `1,024` tokens, `384` requests, mirrored visits `A1 -> B1 -> C1 -> D1 -> D2 -> C2 -> B2 -> A2` in one uninterrupted `98`-minute run. `30.0 s` inter-token decode watchdog: `0` stalls or hangs, `0` errors, `cached_tokens = 0` on every timed request. Arms: **A** = `mlx-vlm 0.8.0` (1 Mac), **B** = `TensorFold 1.2.10` (1 Mac, drafting on), **C** = `GLM-Tandem` drafts OFF (2 Macs, `TP_MTP=0`), **D** = `GLM-Tandem` drafts ON (2 Macs, `TP_MTP=1`).
- **Label for every row below:** `4-bit, lossy`, stock MLX `0.32.4`, `Default` Reasoning Effort, `127.0.0.1` only, `MEASURED`. Receipts: `result-r6-partB.json`, `result-r6-summary.json`, `r6-tables.md`.
- All ratios here are against **one-Mac** `mlx-vlm 0.8.0` or one-Mac `TensorFold`, not against the two-Mac shard-overlay pair of Section 1.1.

#### Table 1.6.1: Summary by Arm & Temperature (24 prompts x 2 reps per arm per temperature = 48 requests per temperature)

| Arm | Temp | Reqs (`cached>0`) | TTFT med (`min–max`) s | Write med (`min–max`) tok/s | **E2E med (`min–max`) tok/s** | Draft Accept (agg) | Speed-up vs `mlx-vlm` (E2E / Write) | Speed-up vs `TensorFold` (E2E / Write) |
|---|---|---|---|---|---|---|---|---|
| mlx-vlm 0.8.0 (1 Mac) | all | 96 (`0`) | 0.579 (`0.242–7.899`) | 43.40 (`41.53–43.77`) | **42.45** (`29.82–43.28`) | n/a | 1.00x / 1.00x | 0.52x / 0.49x |
| mlx-vlm 0.8.0 (1 Mac) | T0.7 | 48 (`0`) | 0.579 (`0.244–7.899`) | 43.36 (`41.53–43.64`) | **42.38** (`29.82–43.01`) | n/a | 1.00x / 1.00x | 0.51x / 0.48x |
| mlx-vlm 0.8.0 (1 Mac) | T0.0 | 48 (`0`) | 0.621 (`0.242–7.889`) | 43.66 (`41.80–43.77`) | **42.60** (`30.78–43.28`) | n/a | 1.00x / 1.00x | 0.52x / 0.49x |
| TensorFold 1.2.10 (1 Mac) | all | 96 (`0`) | 0.332 (`0.140–4.272`) | 89.04 (`72.60–110.71`) | **82.32** (`71.10–97.62`) | 69.7% (52594/75428) | 1.94x / 2.05x | 1.00x / 1.00x |
| TensorFold 1.2.10 (1 Mac) | T0.7 | 48 (`0`) | 0.339 (`0.141–3.869`) | 89.50 (`72.60–110.71`) | **82.76** (`71.67–94.34`) | 69.7% (26384/37827) | 1.95x / 2.06x | 1.00x / 1.00x |
| TensorFold 1.2.10 (1 Mac) | T0.0 | 48 (`0`) | 0.332 (`0.140–4.272`) | 88.42 (`75.31–106.58`) | **82.14** (`71.10–97.62`) | 69.7% (26210/37601) | 1.93x / 2.03x | 1.00x / 1.00x |
| GLM-Tandem drafts OFF (2 Macs) | all | 96 (`0`) | 0.473 (`0.255–3.566`) | 102.27 (`96.68–103.09`) | **97.84** (`71.35–100.37`) | n/a | 2.30x / 2.36x | 1.19x / 1.15x |
| GLM-Tandem drafts OFF (2 Macs) | T0.7 | 48 (`0`) | 0.473 (`0.262–3.566`) | 102.22 (`96.68–102.61`) | **97.64** (`71.57–99.88`) | n/a | 2.30x / 2.36x | 1.18x / 1.14x |
| GLM-Tandem drafts OFF (2 Macs) | T0.0 | 48 (`0`) | 0.474 (`0.255–3.534`) | 102.68 (`97.06–103.09`) | **98.13** (`71.35–100.37`) | n/a | 2.30x / 2.35x | 1.19x / 1.16x |
| GLM-Tandem drafts ON (2 Macs) | all | 96 (`0`) | 0.493 (`0.263–3.638`) | 136.12 (`116.08–159.01`) | **121.23** (`93.49–141.99`) | 81.5% (46278/56792) | 2.86x / 3.14x | 1.47x / 1.53x |
| GLM-Tandem drafts ON (2 Macs) | T0.7 | 48 (`0`) | 0.493 (`0.263–3.614`) | 135.79 (`116.08–156.95`) | **120.36** (`94.72–141.99`) | 80.5% (23394/29050) | 2.84x / 3.13x | 1.45x / 1.52x |
| GLM-Tandem drafts ON (2 Macs) | T0.0 | 48 (`0`) | 0.494 (`0.263–3.638`) | 136.35 (`122.30–159.01`) | **121.62** (`93.49–141.95`) | 82.5% (22884/27742) | 2.85x / 3.12x | 1.48x / 1.54x |

#### Table 1.6.2: By Prompt Type (`all` = `T=0.7` + `T=0.0` pooled, 16 requests per type per arm)

| Prompt Type | Arm | TTFT med (min–max) s | Write med (min–max) tok/s | **E2E med (min–max) tok/s** | E2E vs `mlx-vlm` / `TensorFold` | Draft Accept med (min–max) |
|---|---|---|---|---|---|---|
| `summarise` | mlx-vlm 0.8.0 (1 Mac) | 1.653 (`0.769–4.419`) | 42.50 (`41.86–43.77`) | **39.77** (`35.46–42.22`) | 1.00x / 0.48x | n/a |
| `summarise` | TensorFold 1.2.10 (1 Mac) | 0.936 (`0.428–2.238`) | 89.17 (`86.03–100.03`) | **82.54** (`75.10–86.47`) | 2.08x / 1.00x | 70.4% (65.6–76.5%) |
| `summarise` | GLM-Tandem drafts OFF (2 Macs) | 0.821 (`0.567–2.066`) | 98.74 (`96.74–102.99`) | **91.51** (`80.97–97.14`) | 2.30x / 1.11x | n/a |
| `summarise` | GLM-Tandem drafts ON (2 Macs) | 0.848 (`0.606–2.140`) | 134.00 (`131.46–145.35`) | **120.41** (`103.77–133.44`) | 3.03x / 1.46x | 84.0% (80.4–87.4%) |
| `explain` | mlx-vlm 0.8.0 (1 Mac) | 1.295 (`0.284–5.789`) | 43.11 (`41.80–43.75`) | **40.89** (`33.85–43.23`) | 1.00x / 0.50x | n/a |
| `explain` | TensorFold 1.2.10 (1 Mac) | 0.749 (`0.164–2.908`) | 89.46 (`81.89–93.26`) | **80.97** (`72.75–89.28`) | 1.98x / 1.00x | 71.0% (65.7–73.0%) |
| `explain` | GLM-Tandem drafts OFF (2 Macs) | 0.676 (`0.301–2.616`) | 100.61 (`96.68–103.08`) | **94.35** (`77.53–100.01`) | 2.31x / 1.17x | n/a |
| `explain` | GLM-Tandem drafts ON (2 Macs) | 0.692 (`0.314–2.697`) | 130.87 (`124.81–140.78`) | **122.41** (`95.57–129.80`) | 2.99x / 1.51x | 81.1% (71.6–85.5%) |
| `write_code` | mlx-vlm 0.8.0 (1 Mac) | 0.340 (`0.279–0.356`) | 43.57 (`43.29–43.74`) | **42.96** (`42.68–43.17`) | 1.00x / 0.51x | n/a |
| `write_code` | TensorFold 1.2.10 (1 Mac) | 0.186 (`0.168–0.195`) | 85.06 (`72.60–88.66`) | **83.82** (`71.67–87.20`) | 1.95x / 1.00x | 68.6% (63.1–72.1%) |
| `write_code` | GLM-Tandem drafts OFF (2 Macs) | 0.325 (`0.292–0.352`) | 102.60 (`102.23–103.05`) | **99.42** (`99.00–99.96`) | 2.31x / 1.19x | n/a |
| `write_code` | GLM-Tandem drafts ON (2 Macs) | 0.322 (`0.288–0.351`) | 132.57 (`124.27–137.97`) | **127.41** (`119.94–131.97`) | 2.97x / 1.52x | 78.5% (71.9–83.2%) |
| `translate_arabic` | mlx-vlm 0.8.0 (1 Mac) | 0.358 (`0.279–0.389`) | 43.57 (`43.38–43.73`) | **42.96** (`42.72–43.20`) | 1.00x / 0.47x | n/a |
| `translate_arabic` | TensorFold 1.2.10 (1 Mac) | 0.190 (`0.165–0.236`) | 92.93 (`88.61–100.40`) | **91.47** (`87.36–97.62`) | 2.13x / 1.00x | 70.0% (63.5–72.6%) |
| `translate_arabic` | GLM-Tandem drafts OFF (2 Macs) | 0.343 (`0.292–0.379`) | 102.64 (`102.03–103.09`) | **99.19** (`98.54–100.14`) | 2.31x / 1.08x | n/a |
| `translate_arabic` | GLM-Tandem drafts ON (2 Macs) | 0.341 (`0.275–0.380`) | 143.98 (`139.76–149.37`) | **137.74** (`133.83–141.99`) | 3.21x / 1.51x | 86.4% (83.0–89.1%) |
| `plan` | mlx-vlm 0.8.0 (1 Mac) | 0.272 (`0.242–1.197`) | 43.45 (`43.08–43.73`) | **43.00** (`41.02–43.28`) | 1.00x / 0.56x | n/a |
| `plan` | TensorFold 1.2.10 (1 Mac) | 0.152 (`0.140–0.708`) | 77.48 (`73.76–89.27`) | **76.58** (`72.94–84.17`) | 1.78x / 1.00x | 62.0% (60.4–73.1%) |
| `plan` | GLM-Tandem drafts OFF (2 Macs) | 0.277 (`0.255–0.642`) | 102.47 (`100.55–102.99`) | **99.73** (`94.67–100.37`) | 2.32x / 1.30x | n/a |
| `plan` | GLM-Tandem drafts ON (2 Macs) | 0.282 (`0.263–0.654`) | 123.02 (`116.08–146.90`) | **119.12** (`112.61–134.30`) | 2.77x / 1.56x | 71.5% (63.6–88.2%) |
| `qa_doc_2k_8k` | mlx-vlm 0.8.0 (1 Mac) | 4.779 (`2.809–7.899`) | 41.90 (`41.53–42.23`) | **33.20** (`29.82–35.34`) | 1.00x / 0.43x | n/a |
| `qa_doc_2k_8k` | TensorFold 1.2.10 (1 Mac) | 2.469 (`1.467–4.272`) | 104.98 (`98.11–110.71`) | **77.38** (`71.10–88.64`) | 2.33x / 1.00x | 74.0% (70.3–78.8%) |
| `qa_doc_2k_8k` | GLM-Tandem drafts OFF (2 Macs) | 2.227 (`1.440–3.566`) | 97.24 (`96.80–97.65`) | **75.84** (`71.35–82.47`) | 2.28x / 0.98x | n/a |
| `qa_doc_2k_8k` | GLM-Tandem drafts ON (2 Macs) | 2.302 (`1.478–3.638`) | 152.02 (`139.97–159.01`) | **103.57** (`93.49–114.17`) | 3.12x / 1.34x | 91.3% (90.1–93.9%) |

#### Table 1.6.3: Visit-Pair Repeatability, Paired E2E Speed-ups & Peak Memory

| Arm | Visit 1 E2E med | Visit 2 E2E med | Pair Spread | Paired E2E vs `mlx-vlm` (med, min–max) | Paired E2E vs `TensorFold` (med, min–max) | Load med (s) | Peak `phys` S1 / S2 (GB) |
|---|---|---|---|---|---|---|---|
| mlx-vlm 0.8.0 (1 Mac) | 42.45 | 42.40 | 0.12% | 1.00x (`1.00–1.00x`) | 0.49x (`0.38–0.60x`) | 8.96 | 197.13 / n/a |
| TensorFold 1.2.10 (1 Mac) | 82.30 | 82.36 | 0.07% | 2.05x (`1.68–2.62x`) | 1.00x (`1.00–1.00x`) | 21.77 | 190.98 / n/a |
| GLM-Tandem drafts OFF (2 Macs) | 97.87 | 97.84 | 0.03% | 2.31x (`2.21–2.43x`) | 1.13x (`0.89–1.39x`) | 21.15 | 96.87 / 97.07 |
| GLM-Tandem drafts ON (2 Macs) | 121.23 | 120.90 | 0.27% | 3.03x (`2.62–3.31x`) | 1.49x (`1.21–1.69x`) | 22.76 | 99.07 / 98.21 |

- **Headline (`MEASURED`):** `GLM-Tandem` drafts ON (2 Macs) reached **`121.23 tok/s` end-to-end median** (`93.49-141.99`; `120.36` at `T=0.7`, `121.62` at `T=0.0`) and **`136.12 tok/s` writing median** (`116.08-159.01`), `0.493 s` median TTFT (`0.263-3.638 s`), `81.5%` aggregate MTP head acceptance (`46,278/56,792`) plus `62.5%` lookup acceptance (`6,561/10,500`).
  - vs one-Mac `mlx-vlm 0.8.0`: **`2.86x` E2E** (`3.14x` write) on arm medians; **`3.03x`** (`2.62-3.31x`) on paired per-prompt E2E ratios.
  - vs one-Mac `TensorFold 1.2.10`: **`1.47x` E2E** (`1.53x` write) on arm medians; **`1.49x`** (`1.21-1.69x`) on paired per-prompt E2E ratios.
- Drafts OFF (2 Macs): `97.84 tok/s` E2E (`71.35-100.37`), `102.27 tok/s` write, `0.473 s` TTFT (`2.30x` E2E vs one-Mac `mlx-vlm`, `1.19x` vs one-Mac `TensorFold`, arm medians). One-Mac `TensorFold`: `82.32 tok/s` E2E (`71.10-97.62`), `89.04 tok/s` write, `0.332 s` TTFT, `69.7%` draft acceptance (`52,594/75,428`). One-Mac `mlx-vlm`: `42.45 tok/s` E2E (`29.82-43.28`), `43.40 tok/s` write, `0.579 s` TTFT.
- Visit-pair drift across the `98`-minute run was `0.03%-0.27%` on every arm. Across all `384` requests `max_inter_tok_s` was `0.105 s`, `cached_tokens` was `0`, and `pmset -g therm` recorded no thermal or performance warning before or after the run.
- **Writing speed holds under `Default` effort and `T=0.7`** (`136.12 tok/s`, essentially the same as the `137.13` of Section 1.1), while **end-to-end shows prefill drag on long prompts**: on `2k-8k` document QA (`qa_doc_2k_8k`, `2,887-7,969` prompt tokens) a `2.30 s` median prefill lowers E2E to `103.57 tok/s` even though writing reaches `152.02 tok/s` (`91.3%` draft acceptance).
- **Honest detail:** `GLM-Tandem` drafts ON leads every prompt category in E2E (`103.57` to `137.74 tok/s`; `2.77x-3.21x` over one-Mac `mlx-vlm`, `1.34x-1.56x` over one-Mac `TensorFold`). But on `qa_doc_2k_8k`, one-Mac `TensorFold` (`77.38` E2E, `74.0%` accept) edges out `GLM-Tandem` drafts OFF (`75.84` E2E).
- **Seed limitation:** `mlx-vlm` and `TensorFold` received a JSON `"seed": 20261120` and gave `24/24` identical text hashes between visit 1 and visit 2 at both temperatures. On `GLM-Tandem` (`mlx_lm.server`) the seed was left out at `T=0.7` because sending `"seed"` disables its batch-generator kernels; those two arms gave `0/24` identical outputs at `T=0.7` and `24/24` at `T=0.0`. So `T=0.7` trajectories on `GLM-Tandem` were not reproducible.

---

### 1.7 Round 7 Part 1: Headline Repeat with Three Restarts (`4-bit, lossy`, `Default` Reasoning Effort, `max_tokens=512`)

- **macOS build:** `26B118`; same pins and pack revision as Section 1.6 (`mlx-lm==0.32.2` in the `mlx-vlm` environment, `TensorFold 1.2.10` tarball SHA-256 `76eb33cce9e297f156cfe5cbb0cec664b0266f4ecc3d103176adadc6aee21e58`).
- **Protocol:** the same `24` prompts (six types x four), `T=0`, `Default` effort (template default `Max`), `max_tokens=512`, one request at a time, `3` untimed warm-ups per server start (`cached_max=0`), `108` minutes, fresh server per block. Three rotated or mirrored restarts: `r1: P -> Q -> T1 -> M1 -> T0`, `r2: T0 -> M1 -> T1 -> Q -> P`, `r3: Q -> P -> PL -> M1 -> T1 -> T0`. Each prompt ran twice per restart (`144` timed requests per main arm; `24` for `PL`).
- **Arms:** `P` = `GLM-Tandem` 2 Macs (`TP_MTP=1`, `TP_LOOKUP=1`, `TP_CONC=1`, fast sync on); `Q` = same with `TP_MTP=0`, `TP_LOOKUP=0`; `PL` = `P` with `TP_LOOKUP=0` on the `12` document prompts (restart 3 only); `T1` = `TensorFold 1.2.10`, 1 Mac, default drafting (`MTP + copies`), `--prompt-cache-gib 0`; `T0` = `TensorFold` `--no-drafts`; `M1` = `mlx-vlm 0.8.0`, 1 Mac, no MTP.
- Receipts: `result-r7-part1.json`, `r7-part1-requests.csv`, `r7-part1-clock-drift.svg`.

#### Table 1.7.1: Overall Speed & Telemetry (`MEASURED`)

| Arm | N | Decode med tok/s | E2E med tok/s | TTFT med (p95) s | Time-to-Answer med s* | Drafted? yes/no (accepted/offered) | Cached Max | Peak Mac A/Mac B GB |
|---|---|---|---|---|---|---|---|---|
| `P` (`Tandem TP_MTP=1, TP_LOOKUP=1`) | 144 | **141.4** | **119.9** | 0.390 (3.037) | 3.607 | yes (`41,616/51,402`, `81.0%`) | 0 | `98.3 / 100.0` |
| `Q` (`Tandem TP_MTP=0`) | 144 | **103.3** | **96.2** | 0.383 (3.137) | 4.694 | no (`0/0`) | 0 | `95.8 / 95.6` |
| `T1` (`TensorFold 1.2.10`, 1 Mac) | 144 | **94.7** | **81.7** | 0.329 (3.162) | 4.746 | yes (`43,302/60,908`, `71.1%`) | 0 | `189.2 / n/a` |
| `T0` (`TensorFold --no-drafts`, 1 Mac) | 144 | **68.8** | **65.7** | 0.335 (3.159) | 6.451 | no (`0/0`) | 0 | `189.2 / n/a` |
| `M1` (`mlx-vlm 0.8.0`, 1 Mac) | 144 | **43.4** | **41.7** | 0.547 (6.409) | 10.320 | no (`0/0`) | 0 | `196.2 / n/a` |
| `PL` (`P` with `TP_LOOKUP=0`, r3 doc only) | 24 | **143.5** | **105.5** | 1.201 (3.649) | 3.792 | yes (`6,906/7,634`, `90.5%`) | 0 | `98.1 / 97.6` |

*\*Note on Time-to-Answer: under Default Reasoning Effort (`Max`) with `max_tokens=512`, `summarise`, `explain`, `write_code`, `translate_arabic`, and `plan` remained in `<think>` reasoning at 512 tokens (`t_answer` null), so `t_answer_med` reflects the `qa_doc_2k_8k` subset that exited thinking within 512 tokens.*

#### Table 1.7.2: Ratios with 95% Bootstrap Confidence Intervals (10,000 resamples over prompts x restarts, `MEASURED`)

| Ratio | N pairs | E2E Ratio of Meds [95% CI] | E2E Paired Med [95% CI] | Decode Ratio of Meds [95% CI] | Decode Paired Med [95% CI] |
|---|---|---|---|---|---|
| **`P/T1`** | 72 | **1.47 [1.43–1.49]** | **1.45 [1.42–1.46]** | **1.49 [1.44–1.53]** | **1.52 [1.51–1.54]** |
| `P/Q` | 72 | 1.25 [1.22–1.30] | 1.30 [1.27–1.34] | 1.37 [1.32–1.41] | 1.38 [1.36–1.40] |
| `Q/M1` | 72 | 2.31 [2.29–2.33] | 2.31 [2.30–2.31] | 2.38 [2.37–2.39] | 2.38 [2.36–2.39] |
| `T1/M1` | 72 | 1.96 [1.91–2.06] | 2.14 [2.06–2.19] | 2.18 [2.07–2.21] | 2.18 [2.08–2.22] |
| `T1/T0` | 72 | 1.24 [1.20–1.32] | 1.31 [1.29–1.35] | 1.38 [1.31–1.40] | 1.39 [1.36–1.41] |
| **`P/PL`** | 12 | **1.01 [0.96–1.02]** | **0.99 [0.98–1.02]** | **0.99 [0.95–1.02]** | **0.99 [0.97–1.02]** |

#### Table 1.7.3: Per Prompt Type (median E2E / decode tok/s, `MEASURED`)

| Arm | `summarise` | `explain` | `write_code` | `translate_arabic` | `plan` | `qa_doc_2k_8k` |
|---|---|---|---|---|---|---|
| `P` | 111.7 / 137.1 | 117.4 / 135.9 | 122.8 / 132.9 | 133.4 / 145.4 | 120.1 / 128.1 | 87.0 / 153.5 |
| `Q` | 86.5 / 100.5 | 90.6 / 102.5 | 97.7 / 103.9 | 97.4 / 103.8 | 98.5 / 103.9 | 66.8 / 97.8 |
| `T1` | 79.7 / 95.4 | 76.9 / 89.0 | 83.3 / 85.7 | 94.0 / 97.1 | 80.4 / 82.1 | 69.9 / 104.7 |
| `T0` | 60.0 / 68.1 | 61.1 / 67.7 | 67.6 / 69.2 | 67.5 / 69.1 | 67.9 / 69.1 | 52.1 / 68.4 |
| `M1` | 37.7 / 42.8 | 39.2 / 43.4 | 42.4 / 43.5 | 42.3 / 43.5 | 42.6 / 43.4 | 29.8 / 41.9 |
| `PL` | 114.0 / 141.6 | 116.8 / 137.7 | — | — | — | 87.8 / 151.6 |

#### Table 1.7.4: Per-Restart Stability (`MEASURED`)

| Arm | Restart 1 (E2E / Decode med) | Restart 2 (E2E / Decode med) | Restart 3 (E2E / Decode med) | Cross-Restart Spread (E2E) |
|---|---|---|---|---|
| `P` | 120.08 / 141.50 | 119.92 / 141.12 | 119.74 / 141.17 | `0.28%` |
| `Q` | 95.89 / 103.11 | 96.44 / 103.48 | 96.04 / 103.21 | `0.57%` |
| `T1` | 81.40 / 94.75 | 81.84 / 94.79 | 81.69 / 94.54 | `0.54%` |
| `T0` | 65.82 / 68.89 | 65.83 / 68.86 | 65.66 / 68.74 | `0.26%` |
| `M1` | 41.67 / 43.44 | 41.68 / 43.44 | 41.67 / 43.40 | `0.02%` |

- **Verdict (`MEASURED`):** `P/T1` E2E is **`1.47x [1.43-1.49]`** (paired `1.45 [1.42-1.46]`; decode `1.49 [1.44-1.53]`, paired `1.52 [1.51-1.54]`). The lower bound `1.43` is above the `1.3` threshold we set in advance, so **"about 1.5x over the best one-Mac engine" held on a second day across three fresh restarts.**
- **Is the speed-up just prompt copying?** No. `P/PL` on the `12` document prompts (`summarise`, `explain`, `qa_doc_2k_8k`) is **`1.01 [0.96-1.02]` E2E** (`0.99 [0.95-1.02]` decode), well under our `1.15` threshold: turning the prompt-lookup copier off does not change speed on these natural-language prompts. Prompt lookup does matter on code editing: the `code-refactor` prompt in Section 1.3 (Gate 1) had `86.40%` lookup acceptance (`686/794`).
- **Time-to-answer:** under `Default` (`Max`) effort with `max_tokens=512`, five of the six prompt types were still inside `<think>` reasoning at 512 tokens, so time-to-answer is only meaningful for `qa_doc_2k_8k`. We could not measure it for the other five.
- **`mlx-vlm 0.8.0` built-in cluster mode (arm `M`, one attempt, failed in `4.46 s`):** `GET /v1/cluster` returned `"state":"failed"` and chat requests returned HTTP `503`. Reading `mlx_vlm/server/distributed.py` showed it raises `This model has no distributed shard implementation` because `mlx_vlm.models.glm5_next.Model` has no `shard` method. So our own clean-room shard overlay (Section 1.1) is the only working plain two-Mac `mlx-vlm` reference, and ratios against it must not be described as "vs mlx-vlm's own cluster mode".
- Stability: cross-restart E2E spread `0.28%` (`P`), `0.57%` (`Q`), `0.54%` (`T1`), `0.26%` (`T0`), `0.02%` (`M1`).

---

## Section 2: Many-Users Total (Concurrent Aggregate Throughput)

### 2.1 Round 5 Multi-User Concurrency Ladder (`4-bit, lossy`, Quiet Macs, `T=0`, `max_tokens=256`, `cached=0`)

- **macOS build:** `26B118` (Mac A & Mac B)
- **Pins:** Stock `mlx==0.32.4` (`MLX_FENCE_MODE` unset); Arm A = `GLM-Tandem @ 1a70342137f2` + `omlx==0.8.0rc1` (`a416a587...7af0`), `fast_synch=1`, `TP_CONC=32`, `TP_PROMPT_CONC=1`; Arm B = stock `mlx-vlm 0.8.0` independent copy per Mac (`127.0.0.1:8090`, no draft model, requests split evenly); Arm C = `TensorFold v1.2.10` (`ef0ab81e...19b6`, Mac A only).
- **Pack revision:** `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` `@ d2b17f5253440224422b150705c1109a9a3a0368`.
- **Workload:** `32` unique short prompts (`69–89` input tok) and `8` unique `~8k` prompts (`7,996–8,005` input tok, RFC 9110), launched behind a barrier on fresh servers after untimed warmup (`cached_tokens=0` on all timed rows).

#### Arm A: `GLM-Tandem` Two-Mac TP (Pre-Connected Sockets, Zero SYN Drops)

| Step (`TP_MTP`) | Users (`ok/err`) | Total Out Tok | Wall (s) | **Agg E2E tok/s** | Agg Decode-Span tok/s | Per-User Med / Min tok/s | TTFT Med / Max (s) | Peak Mem `Mac A` / `Mac B` (`phys` / `sys`) | Label | Receipt |
|---|---|---|---|---|---|---|---|---|---|---|
| `1` user (`TP_MTP=1`) | `1` (`1/0`) | `256` | `2.22` | **115.16** | **137.28** | `137.28` / `137.28` | `0.36` / `0.36` | `97.1/125.5` / `98.6/124.1` GB | `MEASURED` | `result-r5-armA.json` |
| `2` users (`TP_MTP=1`) | `2` (`2/0`) | `512` | `3.46` | **147.93** | `175.34` | `91.81` / `87.67` | `0.57` / `0.60` | `97.0/125.1` / `98.7/124.0` GB | `MEASURED` | `result-r5-armA.json` |
| `4` users (`TP_MTP=1`) | `4` (`4/0`) | `1,024` | `5.65` | **181.34** | `200.45` | `56.08` / `52.24` | `0.94` / `1.15` | `97.1/125.2` / `98.4/123.9` GB | `MEASURED` | `result-r5-armA.json` |
| `8` users (`TP_MTP=1`, pre-conn) | `8` (`8/0`) | `2,048` | `9.98` | **205.28** | `217.51` | `30.93` / `28.44` | `1.55` / `2.28` | `99.6/127.8` / `99.6/124.9` GB | `MEASURED` | `result-r5-armA-preconn.json` |
| `16` users (`TP_MTP=1`, pre-conn) | `16` (`16/0`) | `4,096` | `19.31` | **212.13** | `218.80` | `15.80` / `14.69` | `2.74` / `5.03` | `113.3/141.5` / `113.3/138.6` GB | `MEASURED` | `result-r5-armA-preconn.json` |
| `32` users (`TP_MTP=1`, pre-conn) | `32` (`32/0`) | `8,192` | `31.56` | **259.59** | **264.71** | `10.27` / `9.49` | `5.36` / `11.11` | `171.0/199.3` / `171.2/196.7` GB | `MEASURED` | `result-r5-armA-preconn.json` |
| `8 x 8k` (`TP_MTP=1`, pre-conn) | `8` (`8/0`) | `2,048` | `37.28` | **54.94** | `64.50` | `13.88` / `8.32` | `18.41` / `29.57` | `110.0/138.2` / `110.1/135.4` GB | `MEASURED` | `result-r5-armA-preconn.json` |
| `1` user (`TP_MTP=0` control) | `1` (`1/0`) | `256` | `2.70` | `94.77` | `105.97` | `105.97` / `105.97` | `0.29` / `0.29` | `93.7/121.5` / `92.4/117.5` GB | `MEASURED` | `result-r5-armA.json` |
| `8` users (`TP_MTP=0`, pre-conn) | `8` (`8/0`) | `2,048` | `13.80` | `148.41` | `154.54` | `21.57` / `20.32` | `1.68` / `2.69` | `98.1/126.3` / `98.1/123.5` GB | `MEASURED` | `result-r5-armA-preconn.json` |

#### Arm A Failure Mode: Raw Simultaneous TCP `connect()` Burst (`listen(5)` Backlog Overflow)

| Step (`TP_CONC`) | Users (`ok/err`) | Errors (`ECONNRESET` / `EPIPE` / `EINVAL`) | Surviving Out Tok | Agg E2E tok/s (Subset) | Per-User Med / Min | Label | Receipt |
|---|---|---|---|---|---|---|---|
| `mtp1-u8` (`32`) | `8` (`6/2`) | `2` (`1` / `1` / `0`) | `1,536` | `194.01` (`208.33` dec) | `38.64` / `35.71` | `MEASURED` | `result-r5-armA.json` |
| `mtp1-u16` (`32`) | `16` (`6/10`) | `10` (`5` / `2` / `3`) | `1,536` | `195.17` (`210.90` dec) | `39.17` / `36.28` | `MEASURED` | `result-r5-armA.json` |
| `mtp1-u32` (`32`) | `32` (`8/24`) | `24` (`18` / `4` / `2`) | `2,048` | `203.75` (`216.07` dec) | `30.65` / `28.21` | `MEASURED` | `result-r5-armA.json` |
| `mtp1-conc8-u16` (`8`) | `16` (`6/10`) | `10` (`4` / `3` / `3`) | `1,536` | `195.64` (`211.08` dec) | `39.18` / `36.31` | `MEASURED` | `result-r5-armA.json` |
| `mtp1-conc8-u32` (`8`) | `32` (`8/24`) | `24` (`17` / `4` / `3`) | `2,048` | `205.64` (`218.26` dec) | `30.83` / `28.44` | `MEASURED` | `result-r5-armA.json` |
| `mtp1-8x8k` (`32`) | `8` (`6/2`) | `2` (`2` / `0` / `0`) | `1,536` | `54.05` (`66.77` dec) | `19.26` / `11.41` | `MEASURED` | `result-r5-armA.json` |
| `mtp0-u8` (`32`) | `8` (`6/2`) | `2` (`1` / `1` / `0`) | `1,536` | `129.91` (`136.27` dec) | `24.78` / `23.56` | `MEASURED` | `result-r5-armA.json` |

#### Arm B: Dual Independent `mlx-vlm 0.8.0` Servers (One Copy per Mac, Split Evenly, No MTP)

| Step (`split`) | Users (`ok/err`) | Total Out (`A+B`) | Wall (s) | **Combined E2E tok/s** | Combined Decode-Span tok/s | Per-User Med / Min tok/s | TTFT Med / Max (s) | Peak Mem `Mac A` / `Mac B` (`phys` / `sys`) | Label | Receipt |
|---|---|---|---|---|---|---|---|---|---|---|
| `1` (`1+0`) | `1` (`1/0`) | `256` (`256+0`) | `6.16` | **41.57** | `44.16` | `44.16` / `44.16` | `0.36` / `0.36` | `182.4/210.2` / `182.4/207.2` GB | `MEASURED` | `result-r5-armB.json` |
| `2` (`1+1`) | `2` (`2/0`) | `512` (`256+256`) | `6.21` | **82.41** | `87.37` | `43.92` / `43.90` | `0.37` / `0.38` | `182.3/210.1` / `182.4/207.2` GB | `MEASURED` | `result-r5-armB.json` |
| `4` (`2+2`) | `4` (`4/0`) | `1,024` (`512+512`) | `9.99` | **102.47** | `108.78` | `27.45` / `27.33` | `0.63` / `0.68` | `182.1/210.2` / `182.3/207.3` GB | `MEASURED` | `result-r5-armB.json` |
| `8` (`4+4`) | `8` (`8/0`) | `2,048` (`1024+1024`) | `13.76` | **148.83** | `156.77` | `19.87` / `19.75` | `0.84` / `0.87` | `182.4/210.0` / `182.5/207.2` GB | `MEASURED` | `result-r5-armB.json` |
| `16` (`8+8`) | `16` (`16/0`) | `4,096` (`2048+2048`) | `19.94` | **205.43** | `213.12` | `14.10` / `13.54` | `1.54` / `2.07` | `184.1/211.9` / `185.9/211.2` GB | `MEASURED` | `result-r5-armB.json` |
| `32` (`16+16`) | `32` (`32/0`) | `8,192` (`4096+4096`) | `30.83` | **265.71** | **272.94** | `9.19` / `8.76` | `2.03` / `3.54` | `195.2/222.9` / `198.9/223.9` GB | `MEASURED` | `result-r5-armB.json` |
| `8 x 8k` (`4+4`) | `8` (`8/0`) | `2,039` (`1024+1015`) | `68.24` | **29.88** | `34.01` | `19.09` / `4.66` | `54.26` / `54.77` | `219.0/247.1` / `220.8/245.7` GB | `MEASURED` | `result-r5-armB.json` |

#### Arm C: `TensorFold v1.2.10` on ONE Mac (`Mac A` Only)

| Step (Documented Flags) | Users (`ok/err`) | MTP Active? | Total Out Tok | Wall (s) | **Agg E2E tok/s** | Agg Decode-Span tok/s | Per-User Med / Min tok/s | TTFT Med / Max (s) | Peak Mem `Mac A` (`phys` / `sys`) | Label | Receipt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `1` (default 8 streams) | `1` (`1/0`) | **Yes** | `256` | `3.10` | **82.58** | `88.50` | `88.50` / `88.50` | `0.21` / `0.21` | `189.2` / `216.8` GB | `MEASURED` | `result-r5-armC.json` |
| `4` (default 8 streams) | `4` (`4/0`) | **Yes** | `1,024` | `9.42` | **108.67** | `117.55` | `32.93` / `29.39` | `0.71` / `0.71` | `189.2` / `216.7` GB | `MEASURED` | `result-r5-armC.json` |
| `8` (default 8 streams) | `8` (`8/0`) | **Yes** | `2,048` | `16.87` | **121.40** | `132.16` | `17.95` / `16.52` | `1.37` / `1.37` | `189.2` / `216.8` GB | `MEASURED` | `result-r5-armC.json` |
| `16` (default 8 streams; 8 queued) | `16` (`16/0`) | **Yes** | `4,096` | `33.43` | **122.54** | `127.76` | `18.01` / `12.80` | `7.91` / `21.62` | `189.2` / `217.1` GB | `MEASURED` | `result-r5-armC.json` |
| `16` (`--parallel 16 --max-inflight 32`) | `16` (`16/0`) | **No** (`0/0` drafted) | `4,096` | `29.70` | **137.89** | **151.44** | `9.47` / `9.47` | `2.66` / `2.66` | `191.6` / `219.4` GB | `MEASURED` | `result-r5-armC.json` |
| `8 x 8k` (default 8 streams) | `8` (`8/0`) | **Yes** | `2,048` | `83.01` | **24.67** | `25.88` | `12.20` / `6.55` | `59.74` / `74.23` | `190.4` / `217.9` GB | `MEASURED` | `result-r5-armC.json` |

---

### 2.2 Earlier One-Mac Multi-User Tests (`4-bit, lossy`, `26B118`, Rev `d2b17f52`)

| Setup & Workload | Total In (`cached`) / Out Tok | Wall (s) | **Agg E2E tok/s** | Decode-Span tok/s | Per-User tok/s | MTP Accept | Peak Mem | Label | Receipt |
|---|---|---|---|---|---|---|---|---|---|
| 1-Mac `mlx-vlm 0.8.0` server, Mac B (`8` short users, `max_tokens=300`, earlier prompt set) | — / `1,577` | `17.4` | **90.5** (superseded: Round 7 re-measured one Mac at `103.4`, see Section 2.3 Table 2.3.2) | — | `12.3–14.5` (`1` req `36.2` e2e) | n/a | — | `MEASURED` | `concurrency-4bit-mac-b-2026-11-19.log` |
| 1-Mac `TensorFold v1.2.10`, Mac A (`4` x `~8k` RFC 9110, `max_tokens=1024`) | `31,984` (`0`) / `1,453` | `45.19` | **32.15** | `35.46` | `9.91` / `12.21` / `25.70` / `39.44` (med `18.95`) | `76.67%` (`736/960`) | `189.81 GB` phys (`217.4 GB` sys) | `MEASURED` | `tf1-run4.json` |

---

### 2.3 Round 7 Many-Users Ladder and Crossover (`4-bit, lossy`, `T=0`, `Reasoning Effort: Low`, `max_tokens=256`, `cached=0`)

- **macOS build:** `26B118` (Mac A & Mac B). **Pack revision:** `d2b17f52`. Stock `mlx 0.32.4`, all servers on `127.0.0.1` only, never side by side. Receipts: `result-r7-part2.json`, `r7-part2-requests.csv`, `r7-crossover-curve.csv`, `r7-part2-crossover.svg`.
- **Setup:** `short_32` prompts reused cyclically above 32 users with a unique per-user prefix so `cached_tokens=0`; `conc_8k` for `~8k` prompts. `3` repeats per rung (median and `[min-max]`; single repeat for `S3` on `8k` and the raw-burst checks). Client sockets pre-connected before the barrier, plus one raw simultaneous `connect()` burst at `32` users per system. `3` untimed warm-ups after each server start.
- **Systems:** `S1` = `GLM-Tandem` pair (`TP_MTP=1`, `TP_LOOKUP=1`, `TP_CONC=max(32, n)`, `TP_PROMPT_CONC=1`, fast sync on). `S2` = one `mlx-vlm 0.8.0` per Mac, users split evenly, `--max-tokens 4096 --max-num-seqs 32` on every rung (plus a one-Mac control). `S3` = one `TensorFold 1.2.10` per Mac, users split evenly, `--reasoning-effort low`, `--parallel 8` (`2, 8, 16` users and `8k`) or `--parallel 16 --max-inflight 32` (`16, 32, 64` users and raw burst). `S4` = a single `TensorFold 1.2.10` on Mac A only.
- "Drafted?" means drafted tokens `> 0` in that block's own telemetry, not merely that a flag was set. `GLM-Tandem` at pin `1a70342` stops drafting at 5 or more concurrent streams even with `TP_MTP=1`; `TensorFold` stops drafting above 8 active streams per Mac.
- The optional open-loop pass was **skipped** (`15.0` minutes left, below the `20`-minute rule).

#### Table 2.3.1: `S1` `GLM-Tandem` Two-Mac Pair (`MEASURED`)

| Step | Users (`kind`) | Ok/Fail | Total E2E med [min-max] | Per-User E2E med / min | Per-Stream Decode med | TTFT p50 / p95 / max (s) | Drafted? yes/no (acc/off) | Peak Mac A/Mac B GB |
|---|---|---|---|---|---|---|---|---|
| `s1-short-u1` | 1 (`short`) | 1/0 | 115.9 [115.2-118.3] | 115.9 / 115.9 | 136.6 | 0.35 / 0.35 / 0.35 | yes (`394/493`, `79.9%`) | 97.0 / 99.6 |
| `s1-short-u2` | 2 (`short`) | 2/0 | 148.2 [142.8-148.9] | 75.1 / 74.1 | 91.9 | 0.64 / 0.66 / 0.66 | yes (`659/862`, `76.5%`) | 97.1 / 97.1 |
| `s1-short-u4` | 4 (`short`) | 4/0 | 172.2 [172.1-179.5] | 44.3 / 43.0 | 55.3 | 1.10 / 1.33 / 1.33 | yes (`1,292/1,751`, `73.8%`) | 97.0 / 102.1 |
| `s1-short-u8` | 8 (`short`) | 8/0 | 198.5 [198.5-198.7] | 25.1 / 24.8 | 30.0 | 1.70 / 2.55 / 2.55 | yes (`39/53`, `73.6%`)* | 99.1 / 100.5 |
| `s1-short-u12` | 12 (`short`) | 12/0 | 186.7 [186.3-187.4] | 15.8 / 15.6 | 18.4 | 2.32 / 3.97 / 3.97 | yes (`42/53`, `79.2%`)* | 105.3 / 105.3 |
| `s1-short-u16` | 16 (`short`) | 16/0 | 205.6 [205.0-205.9] | 13.1 / 12.9 | 15.5 | 3.06 / 5.47 / 5.47 | yes (`46/56`, `82.1%`)* | 113.2 / 113.2 |
| `s1-short-u20` | 20 (`short`) | 20/0 | 216.8 [215.7-217.2] | 11.1 / 10.8 | 13.2 | 3.77 / 7.14 / 7.14 | yes (`42/53`, `79.2%`)* | 124.5 / 124.5 |
| `s1-short-u24` | 24 (`short`) | 24/0 | 229.9 [229.4-230.3] | 9.9 / 9.6 | 11.9 | 4.54 / 8.61 / 8.71 | yes (`36/53`, `67.9%`)* | 137.8 / 137.8 |
| `s1-short-u32` | 32 (`short`) | 32/0 | 248.3 [248.0-249.4] | 8.1 / 7.8 | 9.9 | 6.02 / 12.10 / 12.31 | yes (`41/55`, `74.5%`)* | 172.1 / 172.2 |
| `s1-short-u48` | 48 (`short`) | 48/0 | 260.8 [259.3-263.9] | 5.7 / 5.4 | 7.3 | 9.55 / 19.70 / 20.62 | yes (`42/58`, `72.4%`)* | 264.1 / 264.0 |
| `s1-8k-u1` | 1 (`8k`) | 1/0 | 45.7 [44.7-46.0] | 45.7 / 45.7 | 136.1 | 3.72 / 3.72 / 3.72 | yes (`429/531`, `80.8%`) | 97.1 / 97.5 |
| `s1-8k-u2` | 2 (`8k`) | 2/0 | 49.5 [49.2-49.7] | 24.8 / 24.8 | 72.0 | 6.48 / 7.47 / 7.47 | yes (`703/811`, `86.7%`) | 99.2 / 99.5 |
| `s1-8k-u4` | 4 (`8k`) | 4/0 | 53.0 [52.3-53.0] | 13.3 / 13.2 | 34.5 | 11.84 / 14.90 / 14.90 | yes (`1,394/1,628`, `85.6%`) | 102.2 / 102.1 |
| `s1-8k-u8` | 8 (`8k`) | 8/0 | 54.1 [54.0-54.3] | 6.8 / 6.8 | 13.7 | 18.75 / 30.09 / 30.09 | yes (`168/201`, `83.6%`)* | 109.8 / 109.6 |
| `s1-raw-burst-u32` | 32 (`short`) | **9/23** | 167.1 [167.1-167.1] | 18.8 / 18.6 | 21.7 | 1.85 / 2.90 / 2.90 | yes (`16/17`, `94.1%`) | 100.3 / 100.1 |

*\*`S1` ramp-edge drafting note: At `8..48` users, `GLM-Tandem` server logs (`logs/r7-p2-s1-short-u16-rank0.log`) confirm steady-state `B >= 5` batching ran **zero** speculative drafts (`>99.5%` of completion tokens un-drafted); the `36–58` total drafted tokens over 3 repeats (`~12–19` per run) occurred exclusively during the brief 1–4 stream ramp-in / tail ramp-out cycles (`[tp-mspec] streams=1..4`). Under raw simultaneous `connect()` without pre-opened sockets (`s1-raw-burst-u32`), `23/32` sockets dropped on `ThreadingHTTPServer`'s `listen(5)` backlog.*

#### Table 2.3.2: `S2` Dual Independent `mlx-vlm 0.8.0` (`MEASURED`)

| Step | Users (`kind`) | Ok/Fail | Total E2E med [min-max] | Per-User E2E med / min | Per-Stream Decode med | TTFT p50 / p95 / max (s) | Drafted? yes/no (acc/off) | Peak Mac A/Mac B GB |
|---|---|---|---|---|---|---|---|---|
| `s2-short-u1` | 1 (`short`) | 1/0 | 41.3 [41.2-41.3] | 41.3 / 41.3 | 43.9 | 0.39 / 0.39 / 0.39 | no (`0/0`) | 182.4 / 182.4 |
| `s2-short-u2` | 2 (`short`) | 2/0 | 82.5 [82.5-82.6] | 41.3 / 41.2 | 43.9 | 0.38 / 0.40 / 0.40 | no (`0/0`) | 182.4 / 182.5 |
| `s2-short-u4` | 4 (`short`) | 4/0 | 101.6 [101.5-101.9] | 25.5 / 25.4 | 27.3 | 0.70 / 0.76 / 0.76 | no (`0/0`) | 182.7 / 182.4 |
| `s2-short-u8` | 8 (`short`) | 8/0 | 146.5 [146.3-146.7] | 18.4 / 18.3 | 19.7 | 1.03 / 1.06 / 1.06 | no (`0/0`) | 182.5 / 182.3 |
| `s2-short-u12` | 12 (`short`) | 12/0 | 176.8 [176.7-177.0] | 14.9 / 14.7 | 15.8 | 1.25 / 1.27 / 1.27 | no (`0/0`) | 182.4 / 182.3 |
| `s2-short-u16` | 16 (`short`) | 16/0 | 207.2 [207.1-208.3] | 13.1 / 13.0 | 14.1 | 1.59 / 1.62 / 1.62 | no (`0/0`) | 184.6 / 183.8 |
| `s2-short-u20` | 20 (`short`) | 20/0 | 223.3 [222.2-223.5] | 11.3 / 11.2 | 12.3 | 1.99 / 2.41 / 2.41 | no (`0/0`) | 185.7 / 185.6 |
| `s2-short-u24` | 24 (`short`) | 24/0 | 234.9 [234.5-235.0] | 9.9 / 9.8 | 10.8 | 2.22 / 2.74 / 2.74 | no (`0/0`) | 187.8 / 188.4 |
| `s2-short-u32` | 32 (`short`) | 32/0 | 266.3 [262.5-266.7] | 8.4 / 8.3 | 9.1 | 2.25 / 3.36 / 3.36 | no (`0/0`) | 192.2 / 193.0 |
| `s2-short-u48` | 48 (`short`) | 48/0 | 307.8 [300.3-307.8] | 6.5 / 6.4 | 7.1 | 3.68 / 5.06 / 5.06 | no (`0/0`) | 203.6 / 203.5 |
| `s2-short-u64` | 64 (`short`) | 64/0 | **334.7** [334.3-336.9] | 5.3 / 5.2 | 5.9 | 4.38 / 7.69 / 7.69 | no (`0/0`) | 218.6 / 218.0 |
| `s2-solo-1mac-u8` | 8 (`short`, 1 Mac) | 8/0 | 103.4 [103.3-103.5] | 12.9 / 12.9 | 14.0 | 1.60 / 1.60 / 1.60 | no (`0/0`) | 183.8 / n/a |
| `s2-8k-u1` | 1 (`8k`) | 1/0 | 18.3 [18.3-18.3] | 18.3 / 18.3 | 42.1 | 7.94 / 7.94 / 7.94 | no (`0/0`) | 197.5 / 182.4 |
| `s2-8k-u2` | 2 (`8k`) | 2/0 | 36.5 [35.8-36.5] | 18.3 / 18.3 | 42.0 | 7.92 / 7.94 / 7.94 | no (`0/0`) | 197.3 / 198.7 |
| `s2-8k-u4` | 4 (`8k`) | 4/0 | 32.7 [32.7-32.8] | 8.3 / 8.2 | 19.1 | 14.96 / 21.73 / 21.73 | no (`0/0`) | 203.6 / 199.8 |
| `s2-8k-u8` | 8 (`8k`) | 8/0 | 29.4 [28.1-29.4] | 3.8 / 3.7 | 19.2 | 54.40 / 56.26 / 56.26 | no (`0/0`) | 222.0 / 220.9 |
| `s2-raw-burst-u32` | 32 (`short`) | **32/0** | 263.7 [263.7-263.7] | 8.4 / 8.2 | 9.1 | 2.49 / 3.78 / 3.78 | no (`0/0`) | 192.2 / 192.2 |

*Settles the Round 5 `90.5` vs `148.8` puzzle: `148.8 tok/s` in Round 5 was **two** Macs (`4+4`, matched here by `s2-short-u8` at `146.5 tok/s`, `18.4/u`). On **one** Mac (`s2-solo-1mac-u8`), `mlx-vlm 0.8.0` with `--max-num-seqs 32 --max-tokens 4096` on `short_32` (`max_tokens=256`, Effort Low) measures **`103.4 tok/s`** (`12.9/u`, `0` errors), compared to `90.5 tok/s` (`12.3–14.5/u`) in the morning single-Mac log (`concurrency-4bit-mac-b-2026-11-19.log`) which used an earlier prompt set and `max_tokens=300`. Notice also `S2` is still climbing cleanly at **`64` users (`334.7 tok/s` aggregate, `0` errors, `7.69 s` p95/max TTFT)** — the measured ceiling is not yet saturated at `64`.*

#### Table 2.3.3: `S3` Dual `TensorFold 1.2.10`, One per Mac (`MEASURED`)

| Step | Users (`kind`) | Ok/Fail | Total E2E med [min-max] | Per-User E2E med / min | Per-Stream Decode med | TTFT p50 / p95 / max (s) | Drafted? yes/no (acc/off) | Peak Mac A/Mac B GB |
|---|---|---|---|---|---|---|---|---|
| `s3-p8-short-u2` | 2 (`short`, `--parallel 8`) | 2/0 | 151.1 [141.2-155.4] | 79.6 / 75.6 | 85.5 | 0.23 / 0.24 / 0.24 | yes (`682/1,031`, `66.1%`) | 189.2 / 189.2 |
| `s3-p8-short-u8` | 8 (`short`, `--parallel 8`) | 8/0 | 218.0 [206.6-218.5] | 28.8 / 27.3 | 31.1 | 0.81 / 0.82 / 0.82 | yes (`2,130/3,021`, `70.5%`) | 189.2 / 189.2 |
| **`s3-p8-short-u16`** | 16 (`short`, `--parallel 8`) | 16/0 | **243.0** [238.4-243.1] | **16.0** / 15.2 | 17.6 | 1.57 / **1.59** / 1.59 | yes (`3,135/4,329`, `72.4%`) | 189.2 / 189.2 |
| `s3-p16-short-u16` | 16 (`short`, `--parallel 16 --max-inflight 32`) | 16/0 | 240.1 [239.8-240.8] | 15.9 / 15.0 | 17.4 | 1.56 / 1.58 / 1.58 | yes (`3,591/5,014`, `71.6%`)* | 191.6 / 191.6 |
| **`s3-p16-short-u32`** | 32 (`short`, `--parallel 16 --max-inflight 32`) | 32/0 | **271.8** [271.4-272.0] | **8.5** / 8.5 | 9.4 | 3.02 / **3.06** / 3.07 | no (`0/0`) | 191.6 / 191.6 |
| `s3-p16-short-u64` | 64 (`short`, `--parallel 16 --max-inflight 32`) | 64/0 | 273.8 [272.5-273.9] | 6.4 / 4.3 | 9.5 | 16.72 / 33.00 / 33.00 | no (`0/0`) | 191.6 / 191.6 |
| `s3-p8-8k-u8` | 8 (`8k`, `--parallel 8`) | 8/0 | 51.5 [51.5-51.5] | 6.7 / 6.4 | 15.9 | 21.12 / 34.82 / 34.82 | yes (`1,161/1,583`, `73.3%`) | 189.8 / 189.8 |
| `s3-p16-raw-burst-u32` | 32 (`short`, `--parallel 16 --max-inflight 32`) | **32/0** | 271.3 [271.3-271.3] | 8.5 / 8.5 | 9.4 | 3.08 / 3.10 / 3.10 | no (`0/0`) | 191.6 / 191.6 |

*\*Note on `S3` `--parallel 16` drafting threshold: Server logs prove `TensorFold 1.2.10` keeps MTP active at `8` active streams per Mac even with `--parallel 16` (`logs/r7-p2-s3-p16-short-u16-mac-a-server.log`: `accepted=37..123`), and switches drafting off (`accepted=0/0`) when `16` streams run concurrently per Mac (`s3-p16-short-u32`, `s3-p16-short-u64`, `s4-short-u16`).*

#### Table 2.3.4: `S4` One `TensorFold 1.2.10` on One Mac, Reference (`MEASURED`)

| Step | Users (`kind`) | Ok/Fail | Total E2E med [min-max] | Per-User E2E med / min | Per-Stream Decode med | TTFT p50 / p95 / max (s) | Drafted? yes/no (acc/off) | Peak Mac A/Mac B GB |
|---|---|---|---|---|---|---|---|---|
| `s4-short-u8` | 8 (`short`, `--parallel 8`) | 8/0 | 120.3 [119.9-120.5] | 15.8 / 15.0 | 17.3 | 1.56 / 1.56 / 1.56 | yes (`1,666/2,319`, `71.8%`) | 189.2 / n/a |
| `s4-short-u16` | 16 (`short`, `--parallel 16 --max-inflight 32`) | 16/0 | 135.6 [135.5-136.0] | 8.5 / 8.5 | 9.4 | 3.02 / 3.02 / 3.02 | no (`0/0`) | 191.6 / n/a |

#### Crossover and Best-Setup Findings (`MEASURED`)

- **`S1` vs `S2` on short prompts:** `S1` leads from `1` through `12` users (per user `115.9` vs `41.3` at `1`; `25.1` vs `18.4` at `8`; `15.8` vs `14.9` at `12`). At `16` users they are a dead heat on per-user E2E (`13.09 [13.06-13.12]` vs `13.11 [13.09-13.15]`, ranges overlap), while `S2` already wins on TTFT p95 (`1.62 s` vs `5.47 s`). From **`20` users** (and at `24`, `32`, `48`) `S2` leads with non-overlapping 3-run ranges (`11.32 [11.20-11.34]` vs `11.11 [11.05-11.13]` at `20`; at `32`: `266.3` vs `248.3` total and `3.36 s` vs `12.10 s` p95 TTFT; at `48`: `307.8` vs `260.8` and `5.06 s` vs `19.70 s` p95 TTFT). **`CROSSOVER at 20 users (short)`.**
- **`S1` vs `S2` on `~8k` prompts (`1, 2, 4, 8` users):** `S1` leads at every rung (`45.7` vs `18.3` at `1`; `24.8` vs `18.3` at `2`; `13.3` vs `8.3` at `4`; `6.8` vs `3.8` at `8`, per user; TTFT p95 `30.09 s` vs `56.26 s` at `8`). **`none (8k)`**: no crossover up to `8 x 8k` users.
- **Best setup for `8`–`32` short users is `S3`, one `TensorFold 1.2.10` per Mac** (leads at `8`, `16` and `32` users; level with `S1` at `2`; not run at `4`, `12`, `20`, `24`): `243.0 tok/s` total at `16` users (`16.0` per user, `1.59 s` max TTFT) and `271.8 tok/s` at `32` users (`8.5` per user, `3.07 s` max TTFT, `32/32` on the raw burst). This replaces the Round 5 advice that two independent `mlx-vlm` servers were best at 32 users (Section 2.1, `265.7`).
- **At `48-64` short users** `S2` keeps scaling to **`334.7 tok/s`** at `64` users (`0` errors, `7.69 s` p95/max TTFT, not yet saturated), while `S3` at `--parallel 16` hits its 16-slot pool limit (`273.8 tok/s`, `33.00 s` p95 TTFT).
- **On `8 x 8k`** the `GLM-Tandem` pair (`S1`) still leads (`54.1 tok/s`, `18.75 s` p50 TTFT, `109.8 GB` per Mac vs `189.8 GB` per Mac for `S3`).
- **`S1` drafting above 4 users:** drafting ran only on the brief 1-4 stream ramp-in and ramp-out edges (see the footnote under Table 2.3.1). Steady state at `8-48` users was un-drafted (`>99.5%` of completion tokens), so `S1`'s many-user numbers are plain tensor-parallel decode.
- **`listen(5)` and raw bursts:** with pre-connected sockets `S1` served all users; under a raw simultaneous `connect()` burst at `32` users `S1` completed `9` and dropped `23` (`s1-raw-burst-u32`), while `S2` and `S3` completed `32/32`. (Round 5 raw bursts, Section 2.1, dropped `24/32`; different run, same cause.)
- **Round 5 `90.5` vs `148.8` puzzle, settled:** see the footnote under Table 2.3.2. One Mac `mlx-vlm` measures `103.4 tok/s` at `8` users on this prompt set; the earlier `90.5` came from an earlier prompt set and `max_tokens=300`; `148.8` was two Macs.

---

### 2.4 Round 7 Part 3: One 32k Prompt Next to Eight Short Prompts (Fairness, `4-bit, lossy`)

- **macOS build:** `26B118`. Workload `W1`: one `31,995`-token RFC 9110 prompt at `t=0`, then `8` short prompts staggered `0.25 s` apart (`0.20-1.95 s`), `max_tokens=256`, `T=0`, `Default` effort, `2` repeats per arm. Receipts: `result-r7-part3.json`, `r7-part3-requests.csv`, `r7-part3-flag-check.log`.

#### Table 2.4.1: Short-Prompt Delay and Stalls (`MEASURED`)

| Arm | Engine | Short TTFT p50 (s) | Short TTFT p95 (s) | Short TTFT max (s) | Longest Inter-Tok Gap (s) | Long 32k TTFT (s) | Total tok/s | Drafted? yes/no (acc/off) | Solo Hash Match | Peak Phys / Sys GB (Watchdog) |
|---|---|---|---|---|---|---|---|---|---|---|
| `tf_default` | `TensorFold 1.2.10` (1 Mac) | **7.649** | 37.039 | 37.039 | 1.330 | 32.153 | 52.5 | yes (`1,917/2,461`, `77.9%`) | **16/16** | `189.2 / 219.2` |
| `tandem_default` | `GLM-Tandem` (2 Macs) | 15.179 | **23.147** | **23.147** | **0.289** | **14.601** | **84.0** | yes (`276/322`, `85.7%`) | 0/16 | **`104.8 / 132.2`** |
| `mlxvlm_default` | `mlx-vlm 0.8.0` (1 Mac) | 36.331 | 37.407 | 37.407 | 5.538 | 36.109 | 34.7 | no (`0/0`) | 0/16 | `485.3 / 512.8` (**80% RAM watchdog!**) |
| `mlxvlm_step_512` | `mlx-vlm 0.8.0` (`--prefill-step-size 512`) | 50.229 | 51.145 | 51.145 | 4.401 | 50.397 | 28.9 | no (`0/0`) | 0/16 | `485.3 / 512.7` (**80% RAM watchdog!**) |
| `mlxvlm_step_2048` | `mlx-vlm 0.8.0` (`--prefill-step-size 2048`) | 36.123 | 37.011 | 37.011 | 4.376 | 35.935 | 35.2 | no (`0/0`) | 0/16 | `485.3 / 512.5` (**80% RAM watchdog!**) |
| `tf_prefill_1024` | `TensorFold 1.2.10` | flag not present | — | — | — | — | — | — | skipped per spec | — |
| `tf_prefill_4096` | `TensorFold 1.2.10` | flag not present | — | — | — | — | — | — | skipped per spec | — |
| `tf_prefill_srpt` | `TensorFold 1.2.10` | flag not present | — | — | — | — | — | — | skipped per spec | — |

- **Summary (`MEASURED`):** a single 32k prompt delays short users by **`7.65 s` median (`37.04 s` for the 9th queued stream) on one-Mac `TensorFold 1.2.10`**, **`15.18 s` on two-Mac `GLM-Tandem`**, and **`36.33 s` on one-Mac `mlx-vlm 0.8.0`**, which also spiked to `485.3 GB` physical / `512.8 GB` system memory on a 512 GiB Mac (past our `80%` / `410.0 GB` watchdog on all six `mlx-vlm` runs: three set-ups x `2` repeats; the default set-up alone fired `2`).
- `TensorFold` cut the 32k prompt into two `16,384`-token prefill chunks (its startup log says so), so short requests entered the stream pool after the first chunk and every short output matched its solo run (`16/16`); only the 9th stream queued behind the pool (`37.04 s`).
- `mlx-vlm`: setting `--prefill-step-size` to `512` gave a **longer** short-request wait (`50.23 s` vs `36.33 s` short TTFT p50).
- `GLM-Tandem` has no chunked-prefill setting at this pin. The 32k prompt prefilled across both Macs in `14.601 s` (about `2,190 tok/s`), delaying all short users until `15.18-23.15 s`, then decoded cleanly (`0.289 s` max inter-token gap, `104.8 GB` per Mac).
- The three `TensorFold` arms that would change prefill chunking (`TENSORFOLD_PREFILL_TOKENS` `1024` / `4096` and `TENSORFOLD_PREFILL_ORDER=srpt`) were **not run**: those settings do not appear in `1.2.10`'s documented `--help`, `serve --help`, `capabilities --json` or `RUNTIME.md`, so they were recorded as "flag not present". We do not recommend them.
- "Solo hash match" `0/16` for `GLM-Tandem` and `mlx-vlm` means their outputs next to a 32k prompt did not match their solo outputs token for token. The report gives no cause for this and we did not score quality in this part.
- **Watchdogs across all Round 7 parts:** **`6` fired**, all `mem_80_pct_abort` on `mlx-vlm` in this part (three `mlx-vlm` set-ups x `2` repeats; the default set-up alone fired `2`) (`485.3 GB` physical / `512.8 GB` system, above the `410.0 GB` line; the server drained its in-flight replies cleanly on shutdown). None crossed `66%` or `80%` in Parts 1, 2, 4 or 5. The `30 s` no-token watchdog fired `0` times (max observed inter-token gap `5.538 s`, `mlx-vlm` in this part); the first-token watchdog fired `0` times (max observed `56.26 s`, `S2` `8k-u8`); the thermal check fired `0` times.

---

## Section 3: Quality and Precision Checks (`4-bit, lossy`)

### 3.1 Round 6 Part A: Accuracy With and Without Drafts (`400` deterministic questions per arm, `T=0`, `Default` effort, max `4,096` tokens)

- **macOS build:** `26B118`; stock `mlx 0.32.4`; pack `d2b17f52`; same pins as Section 1.6. Part A ran `A -> B -> C` on 20 Nov (`00:43-02:00`). Receipts: `result-r6-partA.json`, `result-r6-summary.json`, `r6-tables.md`.
- **Arms:** **A** = `GLM-Tandem` 2 Macs, `TP_MTP=0`; **B** = `GLM-Tandem` 2 Macs, `TP_MTP=1`; **C** = `TensorFold 1.2.10`, 1 Mac.
- **Two suites of `200`:** a locally generated suite (arithmetic, date/calendar, unit conversion, Python output, string tasks, logic ordering; `seed=20261120`; the suite is ours) and a `200`-item sample of the public GSM8K test set (parquet SHA-256 `ee7b8da9...4f59`, `seed=20261120`). Answers are scored strictly on a final line `ANSWER: <value>`.

#### Table 3.1.1: Generated 200-Question Suite (`MEASURED`)

| Arm | Overall (200) | Arithmetic (60) | Date/Cal (30) | Unit Conv (30) | Python (30) | String (25) | Logic (25) | Prompt Tok | Out Tok | Wall (s) | Unparsed / MaxLen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A: GLM-Tandem (2 Macs, `TP_MTP=0`) | 197/200 (**98.5%**, 95% CI 95.7–99.5%) | 60/60 (100.0%) | 30/30 (100.0%) | 30/30 (100.0%) | 29/30 (96.7%) | 25/25 (100.0%) | 23/25 (92.0%) | 18,985 | 87,943 | 912.7 | 1 / 1 |
| B: GLM-Tandem (2 Macs, `TP_MTP=1`) | 192/200 (**96.0%**, 95% CI 92.3–98.0%) | 60/60 (100.0%) | 30/30 (100.0%) | 30/30 (100.0%) | 26/30 (86.7%) | 25/25 (100.0%) | 21/25 (84.0%) | 18,985 | 83,084 | 654.5 | 3 / 1 |
| C: TensorFold 1.2.10 (1 Mac) | 197/200 (**98.5%**, 95% CI 95.7–99.5%) | 60/60 (100.0%) | 30/30 (100.0%) | 30/30 (100.0%) | 29/30 (96.7%) | 25/25 (100.0%) | 23/25 (92.0%) | 18,985 | 88,703 | 986.9 | 2 / 2 |

#### Table 3.1.2: Public GSM8K Sample (`MEASURED`)

| Arm | Exact Numeric Match (200) | Prompt Tok | Out Tok | Wall (s) | Write Agg tok/s | Unparsed / MaxLen |
|---|---|---|---|---|---|---|
| A: GLM-Tandem (2 Macs, `TP_MTP=0`) | 196/200 (**98.0%**, 95% CI 95.0–99.2%) | 20,195 | 58,653 | 631.2 | 104.13 | 1 / 1 |
| B: GLM-Tandem (2 Macs, `TP_MTP=1`) | 198/200 (**99.0%**, 95% CI 96.4–99.7%) | 20,195 | 60,572 | 470.4 | 150.10 | 1 / 1 |
| C: TensorFold 1.2.10 (1 Mac) | 197/200 (**98.5%**, 95% CI 95.7–99.5%) | 20,195 | 54,873 | 564.6 | 104.21 | 0 / 0 |

#### Table 3.1.3: Paired A (`TP_MTP=0`) vs B (`TP_MTP=1`) Differences (`MEASURED`)

| Suite (200) | Parsed Answers Differ (canon / raw) | Both Right | Both Wrong (diff ans) | A Right, B Wrong | A Wrong, B Right | Net Correctness Delta (B−A) | Exact McNemar p |
|---|---|---|---|---|---|---|---|
| Generated (200) | 7 / 7 | 191 | 2 (0) | 6 | 1 | -5 | 0.1250 |
| GSM8K (200) | 3 / 4 | 196 | 2 (1) | 0 | 2 | +2 | 0.5000 |
| Combined (400) | 10 / 11 | 387 | 4 (1) | 6 | 3 | -3 | 0.5078 |

#### Table 3.1.4: Every Item Where the Answer or Correctness Differed Between A and B (`MEASURED`)

| Suite | ID | Category | Expected | A (`TP_MTP=0`) | B (`TP_MTP=1`) | Outcome |
|---|---|---|---|---|---|---|
| Generated | `logic-003` | `logic_ordering` | `Ebony,Amber,Coral,Bronze,Dusk` | `Ebony,Amber,Coral,Bronze,Dusk` (ok) | `Amber,Ebony,Coral,Bronze,Dusk` (last_line/stop) | A right, B wrong |
| Generated | `logic-005` | `logic_ordering` | `Alice,Bob,Carol,Dave,Eve` | `Alice,Dave,Carol,Bob,Eve` (last_line/stop) | `Alice,Bob,Carol,Dave,Eve` (ok) | A wrong, B right |
| Generated | `logic-008` | `logic_ordering` | `Delta,Alpha,Echo,Charlie,Bravo` | `Delta,Alpha,Echo,Charlie,Bravo` (ok) | `None` (no_answer_marker/stop) | A right, B wrong |
| Generated | `logic-009` | `logic_ordering` | `Bob,Eve,Dave,Carol,Alice` | `Bob,Eve,Dave,Carol,Alice` (ok) | `None` (no_answer_marker/stop) | A right, B wrong |
| Generated | `py-001` | `python_output` | `13,4,4` | `13,4,4` (ok) | `13,11,4` (last_line/stop) | A right, B wrong |
| Generated | `py-015` | `python_output` | `5:15` | `5:15` (ok) | `4:14` (last_line/stop) | A right, B wrong |
| Generated | `py-016` | `python_output` | `5,13` | `5,13` (ok) | `5,3` (last_line/stop) | A right, B wrong |
| GSM8K | `gsm8k-test-0340` | `gsm8k` | `4` | `6` (last_line/stop) | `None` (empty_content/length) | both wrong (diff ans) |
| GSM8K | `gsm8k-test-0754` | `gsm8k` | `89` | `107` (last_line/stop) | `89` (ok) | A wrong, B right |
| GSM8K | `gsm8k-test-1019` | `gsm8k` | `98` | `None` (empty_content/length) | `98` (ok) | A wrong, B right |

#### Table 3.1.5: Load Time, Peak Memory and Draft Telemetry (`MEASURED`)

| Arm | Load (s) | Mac A Peak `phys_footprint` (GB) | Mac B Peak `phys_footprint` (GB) | MTP Head / TensorFold Accept |
|---|---|---|---|---|
| A: GLM-Tandem (2 Macs, `TP_MTP=0`) | 19.57 | 98.52 | 98.00 | n/a |
| B: GLM-Tandem (2 Macs, `TP_MTP=1`) | 22.05 | 100.03 | 98.14 | 77744/87888 (88.5%) |
| C: TensorFold 1.2.10 (1 Mac) | 21.97 | 189.21 | n/a | 88250/114466 (77.1%) |

- **Combined over all `400` questions:** `A = 393/400 (98.25%, 95% CI 96.4-99.1%)`, `B = 390/400 (97.50%, 95% CI 95.5-98.6%)`, `C = 394/400 (98.50%, 95% CI 96.8-99.3%)`. Between `A` and `B`, `10` canonical answers (`11` raw strings) differed, flipping `9` correctness outcomes (`6` A-right/B-wrong, `3` A-wrong/B-right, net `-3`, exact McNemar `p = 0.5078`): **no statistically significant quality difference.** Accuracy across both suites and all three arms was `96.0%-99.0%`, with `100.0%` on arithmetic, date/calendar, unit conversion and string tasks in every arm.
- **Generated suite:** `A 197/200 (98.5%, CI 95.7-99.5%)`, `B 192/200 (96.0%, CI 92.3-98.0%)`, `C 197/200 (98.5%, CI 95.7-99.5%)`. Two of `B`'s misses (`logic-008`, `logic-009`) gave the exact answer string on the last line but without the required `ANSWER: ` prefix; we kept them scored as wrong.
- **GSM8K sample:** `A 196/200 (98.0%, CI 95.0-99.2%)`, `B 198/200 (99.0%, CI 96.4-99.7%)`, `C 197/200 (98.5%, CI 95.7-99.5%)`.
- **Why A and B differ on `10/400` answers (`167/400` had identical text):** most likely the multi-row verification step flips occasional near-tie `bfloat16` greedy tokens (Lesson 5). That cause is **inferred, not measured**: Part A streamed without per-token `logprobs`, so we did not log it per token. The flips went both ways.
- **Speed in Part A** (about `146k` generated tokens per arm): `B` finished the `400` questions in `1,124.9 s` (`140.80` write aggregate tok/s on the generated suite, `150.10` on GSM8K) vs `1,543.9 s` for `A` and `1,551.5 s` for `C`.
- **Limits:** `2/400` questions per arm hit `finish_reason=length` at `4,096` tokens. Quality and speed above `max_tokens=4,096` or under concurrent multi-user traffic with `Default` effort were not tested.

### 3.2 Round 7 Part 4: 4-Bit vs 8-Bit Loss Check (`mlx_lm.perplexity`), COULD NOT RUN

- **What we tried:** run `mlx_lm.perplexity --sequence-length 512 --num-samples 64` on the first `200,000` bytes of RFC 9110 (`sha256 0c7d3f5f...0d55`) on Mac A with fast sync off and `mlx_lm==0.32.2`, first on the 4-bit pack and then on the 8-bit `oQ8-MTP` pack, to measure how much quality 4-bit loses.
- **What happened:** it failed in `1.44 s` with `ValueError: Model type glm5_next not supported.` (`ModuleNotFoundError: No module named 'mlx_lm.models.glm5_next'`). We also checked that the 8-bit pack's `config.json` declares `"model_type": "glm5_next"` (`8` bits, group size `64`, affine). We stopped immediately, as planned, without writing a replacement scorer. No 8-bit model was loaded in Round 7.
- **Result:** the 4-bit versus 8-bit loss is still **unmeasured** (`NLL 4bit/8bit void/void`). Every 4-bit number in this repository stays labelled `4-bit, lossy`, with no quantified quality gap to 8-bit. Receipt: `result-r7-part4.json`.

### 3.3 Round 7 Part 5: Sampled GSM8K (`T=0.6`, `5` samples x first `100` questions, `Default` effort, max `4,096` tokens)

- **Setup:** `GLM-Tandem`, 2 Macs, `TP_CONC=1`, `500` requests per arm, `0` errors, `0` watchdogs, `3` untimed warm-ups (`cached_max=0`), `52.3` minutes. Receipts: `result-r7-part5.json`, `r7-part5-requests.csv`.

#### Table 3.3.1: Sampled GSM8K Accuracy and Length (`MEASURED`)

| Arm (`GLM-Tandem`, 2 Macs) | Questions | Samples | Majority Vote Acc | Mean Sample Acc | Mean Output Tok (`n_out`) | Mean Content Chars | Med E2E / Decode tok/s | Drafted? yes/no (`head + lookup`) | Wall (s) / Peak GB |
|---|---|---|---|---|---|---|---|---|---|
| `off` (`TP_MTP=0, TP_LOOKUP=0`) | 100 | 500 | **100/100 (100.0%)** | **494/500 (98.8%)** | 314.3 | 482.8 | 92.1 / 104.3 | no (`0/0`) | `1,835.4 s` / `96.1 / 97.2 GB` |
| `on` (`TP_MTP=1, TP_LOOKUP=1`) | 100 | 500 | **100/100 (100.0%)** | **496/500 (99.2%)** | 297.4 | 486.1 | 127.3 / 152.2 | yes (`91,177/105,596`, `86.3%`*) | `1,301.0 s` / `98.1 / 100.0 GB` |

*\*Draft breakdown on `on`: head `85,625/94,527` (`90.6%`) + lookup `5,552/11,069` (`50.2%`). Paired majority-vote (`n=100`): `both_right=100`, `both_wrong=0`, `off_right_on_wrong=0`, `off_wrong_on_right=0`, exact McNemar `p=1.0000`.*

- Sample-level misses: `off` missed `6/500` samples; `on` missed `4/500` samples. Under `5`-sample majority voting at `T=0.6`, both arms scored **`100/100 (100.0%)`**, while drafting ON finished **`29.1%` faster** (`21.7 min` vs `30.6 min`). This sits alongside Round 6's `400`-question `T=0` evaluation (Table 3.1.3).

### 3.4 Skipped on Purpose or Not Provable

- **Skipped, not approved for Round 7:** `exo` comparison, `HumanEval` (we never run model-written code on the test Macs), a newer `GLM-Tandem` commit (`7b58ce4`, needs a security read of the changes first), and wall-power readings (needs a wall-plug meter).
- **Not proved:** the 4-bit vs 8-bit loss (Section 3.2); the speed of `mlx-vlm`'s built-in distributed cluster mode (Section 1.7); whether `TensorFold` prefill chunking would cut the 32k short-request delay below `7.65 s` / `37.04 s` (undocumented in `1.2.10`); sustained open-loop arrival behaviour (`1` request per second); and identical `T=0.7` trajectories on `GLM-Tandem` (Section 1.6).

---

## Section 4: External Public Claims & Analytical Estimates (`CLAIMED` / `ESTIMATE`)

These rows are **not our measurements**. They are included strictly for honest context and differ in model size, hardware count, or custom patches.

| Source | Claimed / Estimated Setup | Figure | Why Not Directly Comparable | Label | Link |
|---|---|---|---|---|---|
| `icryo/GLM-Tandem` | 2x M5 Ultra, 4-bit MTP, **patched MLX (`MLX_FENCE_MODE=1`)**, 500-word essay mix (`77–79%` accept) | `141 tok/s` | Uses patched inline Metal fence (ours ran stock `mlx 0.32.4`) and higher-accept essay mix | `CLAIMED` | `https://github.com/icryo/GLM-Tandem` |
| `exo` (32-user benchmark) | **4x 256 GB M5 Ultra**, **`Qwen 3.8 27B`** (~15 GB dense 4-bit), 32 users | `~2,000 tok/s` aggregate | 4 Macs vs our 2 Macs; a 27B dense model (~15 GB) vs a 320B total / 18B active MoE (169 GiB on disk, 288 routed experts, 8 active per token); we have not seen their quantization or prompt lengths | `CLAIMED` | `https://github.com/exo-explore/exo` (project page only; we did not verify the original post, so treat the figure as unverified) |
| Analytical `sync_bench` model | 1-Mac vs 2-Mac plain native Flash (`s = 0.01–0.03 ms`) | `~32 tok/s` (1 Mac) / `~55–61 tok/s` (2 Macs) | Analytical projection from `sync_bench` cost (`0.018–0.019 ms`), not a model run | `ESTIMATE` | — |
