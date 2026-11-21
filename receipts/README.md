# Receipts

Scrubbed per-request data and summaries behind the Round 6 and Round 7 tables in [`BENCHMARKS.md`](../BENCHMARKS.md), the prompts we used, and the model manifest.

Scrubbing: the two Macs are named `mac_a` and `mac_b` (originally their host names). We removed absolute paths, user names, IP addresses, uptime and memory-pressure dumps, Python tracebacks, and one raw cluster-status blob that held hardware identifiers. No measured number was changed. Round 6 has no per-request CSV in our lab; its per-request data is inside the summary JSON.

## Results

| File | Backs | SHA-256 |
|---|---|---|
| `result-r6-summary.json` | Round 6 summary (BENCHMARKS Sections 1.6 and 3.1) | `bfff1322089c6b69f6da7a3a82c4fc86d268c27a3a9e98ac3dcef26f6ea815ee` |
| `result-r7-summary.json` | Round 7 summary, parts 1 to 5 (Sections 1.7, 2.3, 2.4, 3.2, 3.3) | `d4eccc4c82364770a6ae7cc09dcc189c5a520202ad502f43141276887519719b` |
| `r7-part1-requests.csv` | One row per request, Round 7 Part 1 (Tables 1.7.1 to 1.7.4) | `a2a4415d0b6720f004b25d86a880409c3369bc7f2635537841c51a0f6e15981f` |
| `r7-part2-requests.csv` | One row per request, Round 7 Part 2 many-users ladder (Section 2.3) | `102156ac85fdde509059f7ca2abdccf7fbf4add4203b5d6e2eb609a5114077a4` |
| `r7-crossover-curve.csv` | Per-step aggregates for the crossover curve (Section 2.3, Table 2.3.2) | `dcc667288e2307c362b265f9a5ab44fc4bf5ba3b8e68a1a5338d32b42f0d4bd2` |
| `r7-part3-requests.csv` | One row per request, one 32k prompt next to eight short prompts (Section 2.4) | `0c2f18bcada3c4ee066214f5d66ede774603b4a848c4068f2d2599d032022191` |
| `r7-part5-requests.csv` | One row per sampled GSM8K answer, drafts off/on (Section 3.3) | `0a313b95de222a1d317e7ba4cb5754e9f73bd22ea571f66fa501abdb41445a90` |
| `model-manifest-4bit-mtp.tsv` | Every file of `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` @ `d2b17f52` (path, size, LFS SHA-256); read by `scripts/verify_model.py` | `05a1abd48a650180add4fbe4ced4a5ba66865e29f5ec3866948bdbc4247bed22` |

The scripts in [`../scripts/`](../scripts/) write CSVs with the same columns as the request files above. `scripts/compare.py` compares your CSV with ours.

## Prompts (`prompts/`)

We read every prompt before publishing it. A prompt is published in full only if we wrote it ourselves (the seeded generators and the hand-written short prompts), or if it comes from a public source: IETF RFC 9110 excerpts reproduced verbatim, and GSM8K (MIT). Where a file mixes publishable and other prompts, the other prompts keep their id, type, token length (chat template), byte length and SHA-256, with `"prompt": null`. Their reason is given in the entry and in `withheld-prompts.csv`.

| File | Contents | Published | SHA-256 of the file in this repo |
|---|---|---|---|
| `r5-prompts.json` | Round 5 and Round 7 Part 2 many-users set: `short_32` (32 hand-written prompts), `warmup_32` (32 one-line warm-ups), `conc_8k` (8 RFC 9110 excerpts of ~8k tokens). Its first 8 `short_32` prompts are also Round 7 Part 3's short prompts | **all 72**, byte-identical to the file we ran | `4d3a622d61bbaedb669222fa6bbf22eb6cab670f4a5a382b1fb595172611c122` |
| `r6-real-public.json` | Round 6 Part B and Round 7 Part 1 realistic set (24 prompts, 6 types x 4) | **16 of 24** (7 built on RFC 9110 excerpts, 9 self-written); 8 withheld | `2b24d9954fed804c06ad19390dbb3cfdf984345476f8be11280a05e49d7a2443` |
| `r6-real-hashes.csv` | Per-prompt SHA-256 of all 24 realistic prompts | hashes | `6cffdc36343e9e1772bb2f0a181fef2867ce4e0c6227948a3e2f1c5d77319967` |
| `r6-quality.json` | 200 generated questions, Round 6 Part A (Section 3.1); written by a seeded script of ours | all 200 | `1a513893352736847ee4dd44ae7598b7526ff1c1e67cd219ead40e3bd633f7f8` |
| `r6-gsm8k.json` | 200 GSM8K test questions (MIT; seed `20261120`), Round 6 Part A; its first 100 are Round 7 Part 5 | all 200 | `7e5802845a8ea84a106506aa84134d95f847638d7fbb1364ce8f3079291a999e` |
| `tf1-prompts-public.json` | TensorFold round: 3 short prompts, `lc-32k` (31,995 tokens of RFC 9110; also the long prompt of Round 7 Part 3), 4 x 8k RFC 9110 excerpts | 8 of 10; `code-2k` and `lc-128k` withheld | `67aa584d6f69ad4cf6104cef1157659f5e7f16f2d2b11ffd705ae9620c1b7ec2` |
| `r4-prompts-public.json` | Round 4 Gate 1 / Round 2 prompts: 3 short prompts, `summary-2k` (RFC 9110) | 4 of 6 distinct; `code-refactor` and `tool-multi-turn` withheld | `d98e6698c663f89fbb23c31f19b7aa4c88f2c07b0797563ebccd8990abb6bd41` |
| `r3-prompts-public.json` | Round 3: `lc-14k`, `lc-54k` and 14 of 16 ~8k soak slices (all RFC 9110) | 16 of 18; two Shakespeare slices withheld | `029e6d2ad797d500a00756083b2e7c6dd96b35b0e6d7ff25201b09c7e6fe7c46` |
| `withheld-prompts.csv` | One row per withheld prompt: file, id, type, tokens, bytes, SHA-256, reason | — | `a98a8a8fc4749cc8b4f0f7ff8c6c9ab07714b8503cd8887581199bc9626245d4` |

**Check against the hashes quoted elsewhere:**

- `r5-prompts.json`: `4d3a622d...c122` is the hash given in `BENCHMARKS.md` (pins table) and `AGENTS.md` (preflight step 5). **Match:** the published file is byte-identical to the one we ran.
- `r6-gsm8k.json` (`7e580284...999e`) and `r6-quality.json` (`1a513893...f7f8`): **match** `AGENTS.md` and the earlier version of this table.
- `r6-real.json` (`825c8bc8...ed10` in `BENCHMARKS.md` and `AGENTS.md`): the full file cannot be published, so its hash cannot be reproduced from this repo. `r6-real-public.json` records it as `original_sha256`, and every one of its 24 per-prompt SHA-256s matches `r6-real-hashes.csv`.
- `tf1`, `r4` and `r3` files: no hash was quoted in `BENCHMARKS.md` or `AGENTS.md`. The public files differ from our lab files only by the withheld text, and each records the lab file's hash as `original_sha256` (`41d07f46...a9eb`, `b138f27d...1014`, `0866335b...e2d`).

**Why prompts were withheld** (14 in all):

| Reason | Prompts |
|---|---|
| Private or unrecorded-origin text (a product changelog; one tool-call conversation naming our lab machines) | `r6`: `real-sum-01`, `real-sum-02`, `real-ar-01`, `real-ar-03`, `real-plan-03`, `real-qa-01`, `real-qa-02`; `r4`: `tool-multi-turn` |
| Third-party source-code excerpt (an `mlx-vlm` model file; outside this repo's licence boundary) | `r6`: `real-exp-01`; `r4`: `code-refactor`; `tf1`: `code-2k` |
| Restricted electronic edition: our Shakespeare file is Project Gutenberg etext #100, which carries a World Library, Inc. 1990-1993 copyright notice with non-commercial terms | `tf1`: `lc-128k`; `r3`: `slice-14-shakespeare.txt`, `slice-15-shakespeare.txt` |

What this means for replication: Round 7 Part 2 (many users) and Part 3 (fairness) can be rerun with exactly our prompts. Part 1 and Round 6 Part B can be rerun on 16 of the 24 prompts. `scripts/bench_single.py` skips withheld prompts, and `scripts/compare.py` compares only the prompt ids both files share. Over those 16 prompts, our `T1` receipt has a median E2E of 79.6 tok/s; over all 24 it is 81.7.

**Public texts used elsewhere:** RFC 9110 full text `21c1cdce6ab0e5509b04d84a28000836c7a087cf786efe6f04877ebfff47232a` (first 200,000 bytes `0c7d3f5fab58e6be647c5e979e0ddcb0a2526a4d627a14b93b52f880a9e08d55`); GSM8K test parquet `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59`; Shakespeare etext #100 (not included) `c5601d266987ac885ca96522b1b4e439feb7eca39f0fe1111f7342b63b6468f3`. RFC 9110 is published by the IETF. Its copyright belongs to the IETF Trust and the document authors (BCP 78 and the IETF Trust Legal Provisions apply). The excerpts here are unmodified. The excerpts that start at the top of the RFC (in `r5`, `tf1`, `r4` and `r3`) include its header and copyright notice; the `r6` excerpts are sections from the middle of the document. Some excerpt descriptions inside the prompt files say "public-domain"; that wording is our harness's label, not a licence statement.
