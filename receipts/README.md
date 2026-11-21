# Receipts

Scrubbed per-request data and summaries behind the Round 6 and Round 7 tables in [`BENCHMARKS.md`](../BENCHMARKS.md).

Scrubbing: the two Macs are named `mac_a` and `mac_b` (originally their host names). Absolute paths, user names, IP addresses, uptime and memory-pressure dumps, Python tracebacks, and one raw cluster-status blob with hardware identifiers were removed. No measured number was changed. Round 6 has no per-request CSV in our lab; its per-request data is inside the summary JSON.

| File | Backs | SHA-256 |
|---|---|---|
| `result-r6-summary.json` | Round 6 summary (BENCHMARKS Sections 1.6 and 3.1) | `bfff1322089c6b69f6da7a3a82c4fc86d268c27a3a9e98ac3dcef26f6ea815ee` |
| `result-r7-summary.json` | Round 7 summary, parts 1 to 5 (Sections 1.7, 2.3, 2.4, 3.2, 3.3) | `d4eccc4c82364770a6ae7cc09dcc189c5a520202ad502f43141276887519719b` |
| `r7-part1-requests.csv` | One row per request, Round 7 Part 1 (Tables 1.7.1 to 1.7.4) | `a2a4415d0b6720f004b25d86a880409c3369bc7f2635537841c51a0f6e15981f` |
| `r7-part2-requests.csv` | One row per request, Round 7 Part 2 many-users ladder (Section 2.3) | `102156ac85fdde509059f7ca2abdccf7fbf4add4203b5d6e2eb609a5114077a4` |
| `r7-crossover-curve.csv` | Per-step aggregates for the crossover curve (Section 2.3, Table 2.3.2) | `dcc667288e2307c362b265f9a5ab44fc4bf5ba3b8e68a1a5338d32b42f0d4bd2` |
| `r7-part3-requests.csv` | One row per request, one 32k prompt next to eight short prompts (Section 2.4) | `0c2f18bcada3c4ee066214f5d66ede774603b4a848c4068f2d2599d032022191` |
| `r7-part5-requests.csv` | One row per sampled GSM8K answer, drafts off/on (Section 3.3) | `0a313b95de222a1d317e7ba4cb5754e9f73bd22ea571f66fa501abdb41445a90` |
| `prompts/r6-quality.json` | 200 generated questions, Round 6 Part A (Section 3.1); self-written by a seeded script | `1a513893352736847ee4dd44ae7598b7526ff1c1e67cd219ead40e3bd633f7f8` |
| `prompts/r6-gsm8k.json` | 200 public GSM8K test questions (MIT), Round 6 Part A (Section 3.1) | `7e5802845a8ea84a106506aa84134d95f847638d7fbb1364ce8f3079291a999e` |
| `prompts/r6-real-hashes.csv` | Per-prompt SHA-256 of the 24 realistic prompts, Round 6/7 (Sections 1.6, 1.7); text not published | `6cffdc36343e9e1772bb2f0a181fef2867ce4e0c6227948a3e2f1c5d77319967` |

## Prompt files not published

The 24 realistic prompts (`r6-real.json`) and the short/8k concurrency prompts (`r5-prompts.json`) contain text of mixed or unrecorded origin, so only their hashes are given. Verify your own copy against these.

- `r6-real.json` (24 realistic prompts): `825c8bc84f7709a44497acd491b47c2156079e4e8f980b3bfd2c458ee89aed10`; per-prompt hashes in `prompts/r6-real-hashes.csv`.
- `r5-prompts.json` (`short_32` / `conc_8k` set): `4d3a622d61bbaedb669222fa6bbf22eb6cab670f4a5a382b1fb595172611c122`.
- Public texts used elsewhere: RFC 9110 full text `21c1cdce6ab0e5509b04d84a28000836c7a087cf786efe6f04877ebfff47232a`; GSM8K test parquet `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59`.
