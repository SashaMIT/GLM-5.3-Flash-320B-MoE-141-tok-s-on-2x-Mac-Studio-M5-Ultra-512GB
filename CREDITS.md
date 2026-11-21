# Credits

## Thank you

This benchmark exists because other people did the hard work first and shared it openly. **TensorFold's `GLM-5.3-Flash-MLX-4bit-MTP` pack is the model every 4-bit engine in these tests ran on**, and its `tensorfold-native 1.2.10` engine is the fastest one-Mac engine we measured. Z.ai made the model; icryo built the two-Mac runtime that gives the fastest single-user speed; Blaizzy's `mlx-vlm` and jundot's `oMLX` are the foundation under several of our setups; Apple's MLX team built the stack all of it runs on; and a group of community members taught us what to test and how. Every one of them is listed below with what they gave us. Thank you all.

Rules we follow: credit everyone whose work we USED, and separately everyone we LEARNED FROM. We quote versions and commits we used. None of the credited projects or people endorsed these results, and our measured numbers are ours alone. `icryo/GLM-Tandem` has no licence: we credit and link it, and we never copied any of its code.

## Model packs (the weights every test ran on)
- **TensorFold, `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP`** (rev `d2b17f52`, MIT): the 4-bit pack with its MTP (multi-token prediction) layer. **Every engine in the 4-bit tests (`TensorFold`, `GLM-Tandem`, `mlx-vlm` and our shard overlay) ran on this pack**, so every 4-bit result in this repository stands on TensorFold's conversion work (the FP8 rows used the `dealignai` FP8 pack below). Verified with a 54-row manifest.
- **Z.ai (`zai-org`), `zai-org/GLM-5.3-Flash`** (MIT): the GLM-5.3-Flash model itself (`glm5_next`, 320B total / 18B active, MoE) and the official FP8 weights (rev `46a050be`), and the source of the official `oQ8` 8-bit reference pack (`zai-org/GLM-5.3-Flash-oQ8-MTP`) we prepared for the 4-bit vs 8-bit quality check (that check could not be run).
- **dealignai, `dealignai/GLM-5.3-Flash-UNCENSORED-FP8`** (rev `d5822b06`): the FP8 pack behind the MXFP8 runs. Results for it are speed-only; check its model card licence before any reuse.

## Engines and software we ran
- **TensorFold** by Ash Hart (`@ashxhart`), https://github.com/ashhart/TensorFold , Apache-2.0: `tensorfold-native 1.2.10` (release `tensorfold-1.2.10-macos-arm64`, `RUNTIME.md`). The fastest one-Mac engine in our tests, and, run one per Mac, our best many-user setup at 8, 16 and 32 users (level with GLM-Tandem at 2; not run at 4, 12, 20 or 24). Also credited for its documentation, the `RUNTIME.md` 70% RAM and MTP-load guidance, and the MCDMA multi-Mac fabric that we noted but did not test.
- **GLM-Tandem** by icryo, https://github.com/icryo/GLM-Tandem (pinned commit `1a703421`): the two-Mac, MTP, tensor-parallel runtime behind our fastest single-user results. No licence: link and credit only; none of its code is included here.
- **oMLX** by jundot and contributors (Apache-2.0), https://github.com/jundot/omlx , release `v0.8.0rc1` wheel (`omlx-0.8.0rc1`), which `GLM-Tandem` builds on. Its vendored `glm5_next` code (from Blaizzy/mlx-vlm PR #2030) also explained several of our results.
- **MLX** and **mlx-lm** by ml-explore (Apple), MIT: `mlx 0.32.4` (stock, no patches) and `mlx-lm 0.32.2`, the stack every run sits on, including the `jaccl` RDMA backend over Thunderbolt 5.
- **mlx-vlm** by Blaizzy (`Blaizzy/mlx-vlm`, MIT), `0.8.0`: the one-Mac and dual-server runs (it leads at 48 to 64 users), and the base of our plain two-Mac shard overlay. Its `glm5_next` support (#2127) and the open `glm5_next` shard PR #2684 were useful references.
- **exo** by exo-explore (Alex Cheema and team), Apache-2.0, https://github.com/exo-explore/exo : a reference we compared against and learned from (its main-branch PR #2517 for GLM-5.3-Flash was part of our engine research). We checked that it was idle before runs and did not benchmark it ourselves.
- **mlx-serve** (its `docs/cli.md`): read during engine research.

## Data and test material
- **GSM8K** by Cobbe et al. (OpenAI) (`openai/gsm8k`, MIT): the public maths test set. We used a fixed 200-question sample (seed 20261120) for the round 6 quality score and the first 100 questions (5 samples each at T=0.6) for the round 7 sampled check.
- **RFC 9110, HTTP Semantics** (IETF, public): the long-document text for the long-prompt, soak, fairness and perplexity-attempt prompts (rounds 3 to 7).
- **The Complete Works of William Shakespeare** (Project Gutenberg, public domain): a test corpus in the TensorFold one-Mac runs.

## Community members and write-ups we learned from
We credit these people for ideas, baselines, bug reports and recipes. We do not publish or rely on any number of theirs that we could not verify.
- **Mau Ledford (`@krunkosaurus`)**: TensorFold vs oMLX one-Mac comparison and the agent-loop observation, which shaped how we framed one-Mac baselines.
- **Kevin Gilbertson / gilby (`@unigilby`)**: TensorFold PRs #3903 and #3920 and their receipts.
- **jcscocca**: TensorFold issue #3902 (MTP left unloaded, first-decode stall on macOS `26B118`); it is why we verify the MTP-loaded line and use 5-repetition medians.
- **sethforprivacy**: the GLM-5.3-Flash M5 Ultra oMLX recipe (MIT).
- **BobClawblaw**: a community one-Mac recipe (study only).
- **0xSero** and **LazyLocalLLM**: public posts and baselines that taught us what readers want compared. We cite them for ideas only; we never quote their figures (unverified here).
- **Alex Cheema and the exo team**: a public multi-user claim we compared against (shown in `BENCHMARKS.md` as `CLAIMED`, not our number, unverified).
- **icryo**: the GLM-Tandem write-up (MTP on a pair, fast fence).
- **Ash Hart**: TensorFold docs, `RUNTIME.md`, MCDMA fabric numbers.
- **MLX maintainers (zcbenz and others)**: fast-sync guidance, issues #3142 and #4438, PR #4689.
- **Apple**: TN3205, RDMA over Thunderbolt.

## Our team (how the work was done)
- The project owner (this repository's account): idea, lab hardware, every go/no-go decision.
- Claude (Anthropic), CTO and orchestrator: plans, verification against raw logs, the reports, and the specs for rounds 5, 6 and 7.
- Gemini (via Cursor): ran the GLM-Tandem trials and the shard overlay, and ran, analysed and wrote up the round 7 tests.
- Grok bot, Ash Bot, Flint, Bella (the project owner's research agents): the cheat sheet, the security review of GLM-Tandem, the pins and the test gates. Fable and Ash Bot reviewed earlier results: Fable's review named the two evidence gaps that round 6 addresses (a quality score and a realistic rerun), and Ash Bot's feedback on an earlier report fed the round 7 plan.
- Codex and Kimi: reviews.
