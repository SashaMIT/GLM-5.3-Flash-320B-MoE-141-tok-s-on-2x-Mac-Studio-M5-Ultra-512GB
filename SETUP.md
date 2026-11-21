# Setup

Copy-pasteable setup for reproducing the results. Replace every `<placeholder>` with your own value. Do not publish your host names, IP addresses, user names or keys.

| Placeholder | Meaning |
|---|---|
| `<MAC_A>`, `<MAC_B>` | SSH names of the two Macs (Mac A = rank 0 / coordinator; Mac B = rank 1) |
| `<MAC_A_TB_IP>`, `<MAC_B_TB_IP>` | Each Mac's link-local (169.254.x.x) address on the Thunderbolt interface |
| `<TB_IF>` | The Thunderbolt network interface the cable is on (it was `en4` in our lab) |
| `<mlx-env>`, `<tandem-env>` | Python environments ([`env/README.md`](env/README.md)) |
| `<models-dir>` | Folder for model packs (about 182 GB for the 4-bit pack) |
| `<tools-dir>` | Folder for TensorFold and the GLM-Tandem checkout |

Steps marked **[admin]** need an administrator password. The step marked **[recovery mode]** also needs physical access to the Mac. Everything else runs as a normal user, with no `sudo`. Keep `sysctl iogpu.wired_limit_mb` at `0`. We never changed it.

Only steps 1, 3 and 4 are needed for the one-Mac [quick replication](AGENTS.md#quick-replication-one-mac-1-hour). Steps 5 and 6 are for the two-Mac runs.

---

## 1. Check the Mac

```bash
sw_vers                                  # we used macOS 27.2, build 26B118 (sw_vers -buildVersion)
sysctl -n hw.model hw.memsize            # we used Mac17,15 (M5 Ultra), 549755813888 bytes (512 GiB)
sysctl -n iogpu.wired_limit_mb           # must print 0
pmset -g therm                           # no CPU/GPU speed limit below 100
```

## 2. Python environments

Follow [`env/README.md`](env/README.md). The one-Mac quick replication needs no Python environment beyond a system `python3` (3.9 or newer) for the scripts. `<mlx-env>` is needed for the `mlx-vlm` arm, `sync_bench.py` and the `hf` download tool. If you skip `<mlx-env>`, any recent `huggingface_hub` gives you `hf`.

## 3. Model: download at the pinned revision and verify

The pack is [`TensorFold/GLM-5.3-Flash-MLX-4bit-MTP`](https://huggingface.co/TensorFold/GLM-5.3-Flash-MLX-4bit-MTP) (MIT). The pinned revision is `d2b17f5253440224422b150705c1109a9a3a0368`. It has 54 files: 45 large (LFS) files with SHA-256 and 9 small files checked by size.

```bash
<mlx-env>/bin/hf download TensorFold/GLM-5.3-Flash-MLX-4bit-MTP \
  --revision d2b17f5253440224422b150705c1109a9a3a0368 \
  --local-dir <models-dir>/TensorFold__GLM-5.3-Flash-MLX-4bit-MTP

# full check: every file's size, every LFS file's SHA-256, three pinned SHA-256s and the config fields
python3 scripts/verify_model.py --model-dir <models-dir>/TensorFold__GLM-5.3-Flash-MLX-4bit-MTP
# expect:  summary: ok=45 oksize=9 bad=0 rows=54   ...   PASS
```

`scripts/verify_model.py` reads [`receipts/model-manifest-4bit-mtp.tsv`](receipts/model-manifest-4bit-mtp.tsv), which lists every file of that revision. It passes only when `bad == 0`, `ok + oksize == rows`, and these hold:

| Check | Expected |
|---|---|
| `config.json` SHA-256 | `f80682bcc8ba5fc6ecdc5ce553700b02befe999bc4a6ba6fd47f644df4864adf` |
| `model.safetensors.index.json` SHA-256 | `6c030d5b58f4c264f00becf9dddcb8f8b1f69b3bbaaf35c93592b201b673208f` |
| `tokenizer.json` SHA-256 | `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` |
| `config.json` fields | `text_config.num_hidden_layers = 45`, `text_config.num_nextn_predict_layers = 1`, `quantization = {"bits": 4, "group_size": 64, "mode": "affine"}` |

You can also check the three pinned files by hand: `cd <models-dir>/TensorFold__GLM-5.3-Flash-MLX-4bit-MTP && shasum -a 256 config.json model.safetensors.index.json tokenizer.json`. On the two-Mac runs, both Macs need an identical, verified copy at the same path.

## 4. TensorFold 1.2.10 (one-Mac engine)

TensorFold by Ash Hart, Apache-2.0: <https://github.com/ashhart/TensorFold>. Use release tag **`v1.2.10`**. Our lab notes record the asset name and its hash, but not the download URL. Take the asset from the TensorFold release page for `v1.2.10`: <https://github.com/ashhart/TensorFold/releases/tag/v1.2.10>.

| Item | Value |
|---|---|
| Asset | `tensorfold-1.2.10-macos-arm64.tar.gz` (7,651,150 bytes) |
| Asset SHA-256 | `76eb33cce9e297f156cfe5cbb0cec664b0266f4ecc3d103176adadc6aee21e58` |
| Unpacked folder | `tensorfold-1.2.10-macos-arm64/` with `SHA256SUMS` (23 files), `RUNTIME.md`, `bin/tensorfold-native`, `lib/`, `LICENSE`, `LICENSES/`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, `VERSION` |
| `bin/tensorfold-native` SHA-256 | `ef0ab81eb11496956cf33eba9fd16d26b3500be54f6c93b08885d667c1af19b6` |

```bash
mkdir -p <tools-dir>/tensorfold-1.2.10 && cd <tools-dir>/tensorfold-1.2.10
# put tensorfold-1.2.10-macos-arm64.tar.gz from the v1.2.10 release page here, then:
echo "76eb33cce9e297f156cfe5cbb0cec664b0266f4ecc3d103176adadc6aee21e58  tensorfold-1.2.10-macos-arm64.tar.gz" | shasum -a 256 -c -
tar -xzf tensorfold-1.2.10-macos-arm64.tar.gz
cd tensorfold-1.2.10-macos-arm64
shasum -a 256 -c SHA256SUMS                       # expect 23 lines ending in OK
shasum -a 256 bin/tensorfold-native               # expect ef0ab81e...19b6
bin/tensorfold-native --version                   # tensorfold-native 1.2.10
```

Before the first run, read `RUNTIME.md`, `bin/tensorfold-native --help`, `serve --help`, `capabilities --json` and `device --json`. Use only flags documented there. Do not run any install or setup script from the package; the binary runs in place. `TENSORFOLD_PHASE_METRICS` is documented in `bin/tensorfold-native capabilities --json`; `serve --help` prints only usage. The quick replication in [`AGENTS.md`](AGENTS.md#quick-replication-one-mac-1-hour) gives the exact launch line.

## 5. GLM-Tandem (two-Mac engine) at the pinned commit

[`icryo/GLM-Tandem`](https://github.com/icryo/GLM-Tandem) **has no licence**. Clone it for your own use. This repository contains none of its code, and you should not redistribute it either.

```bash
# on both Macs
git clone https://github.com/icryo/GLM-Tandem <tools-dir>/tandem
cd <tools-dir>/tandem
git checkout --detach 1a70342137f2650367b8ca2619e19ea854a7658b
git rev-parse HEAD                     # must print 1a70342137f2650367b8ca2619e19ea854a7658b
```

**The one change we made (bind to loopback only).** At this commit, the HTTP server entry point `server/tpserve.py` starts the server by building an argument list for `mlx_lm.server`. That list passes `--host` with the value `0.0.0.0`, so the server listens on every network interface. In our checkout, the line was near line 1495. We replaced that single string value with `127.0.0.1` and changed nothing else.

```bash
grep -n '0\.0\.0\.0' server/tpserve.py   # find the --host value; edit it to 127.0.0.1 by hand
git diff --stat                          # must show exactly: server/tpserve.py | 2 +- (1 insertion, 1 deletion)
grep -rn '0\.0\.0\.0' server/ scripts/ || echo "no other 0.0.0.0 binds"
```

Our receipt for that diff was `git diff | shasum -a 256` = `af8d4ec83c73824c0468f5aa91d2c3b77c96679715ad8232be13c0f9cf2557e1`. A different git version may format the diff header differently, so the `--stat` check above is the one that matters. Check the diff again before every launch. The ranks talk over Thunderbolt RDMA (`jaccl`), not over this HTTP port, so loopback-only does not affect the two-Mac link. Install `<tandem-env>` as described in [`env/README.md`](env/README.md).

## 6. Thunderbolt 5 RDMA link between the two Macs

We used one Thunderbolt 5 cable between the two Macs, with no switch or hub.

**6.1 Enable RDMA on both Macs: [recovery mode] [admin].** On our Macs the owner enabled it in macOS Recovery; it cannot be done from a normal session. The tool is `/usr/bin/rdma_ctl`, and its usage line is `rdma_ctl <status|enable|disable>`. On each Mac: start up into macOS Recovery, open Terminal from the Utilities menu, run `rdma_ctl enable`, then restart normally. We did not record the exact keystrokes for entering Recovery; follow Apple's instructions for your Mac. Back in macOS, as a normal user:

```bash
rdma_ctl status                  # enabled
ibv_devices                      # lists rdma_en* devices (we saw rdma_en2, en3, en4, en5, en6, en13)
ibv_devinfo -d rdma_<TB_IF>      # state PORT_ACTIVE, active_mtu 4096, link_layer Thunderbolt
```

The cable's port shows up as the one `rdma_*` device whose state is `PORT_ACTIVE`. Its interface name (without `rdma_`) is your `<TB_IF>`.

**6.2 Turn Thunderbolt Bridge off: [admin].** With the bridge on, the Thunderbolt ports join `bridge0`, and the per-port link that RDMA uses is not available as a plain interface. In our lab, exo's network setup did this when the owner typed the admin password. It created a network location with one service per Thunderbolt port and no `bridge0`. To do it by hand, a standard macOS way (not the exact one we used) is System Settings > Network > Thunderbolt Bridge > disable it (or delete the service). The command-line equivalent is:

```bash
networksetup -listallnetworkservices                                  # find "Thunderbolt Bridge"
networksetup -setnetworkserviceenabled "Thunderbolt Bridge" off       # [admin]
```

**6.3 Check the link (normal user, both Macs):**

```bash
ifconfig bridge0                 # must fail: "interface bridge0 does not exist"
ifconfig <TB_IF>                 # status: active, inet 169.254.x.x (link-local; we set no static IPs)
ping -c 10 <MAC_B_TB_IP>         # from Mac A; we measured 0.447 / 0.563 / 0.653 ms min/avg/max
```

**6.4 Host file for `mlx.launch` (`hosts-jaccl.json`, keep it out of any public repo):**

```json
{"backend": "jaccl", "envs": [], "hosts": [
 {"ssh": "<MAC_A>", "ips": ["<MAC_A_TB_IP>"], "rdma": [null, "rdma_<TB_IF>"]},
 {"ssh": "<MAC_B>", "ips": ["<MAC_B_TB_IP>"], "rdma": ["rdma_<TB_IF>", null]}]}
```

GLM-Tandem takes the device map as `MLX_IBV_DEVICES` pointing to a JSON file containing `[[null, "rdma_<TB_IF>"], ["rdma_<TB_IF>", null]]`. Rank 0 must be able to SSH to Mac B over the Thunderbolt address. Create a temporary key for this, and remove it afterwards: [`AGENTS.md`](AGENTS.md), hard stop 5.

**6.5 Measure it:** [`scripts/sync_bench.py`](scripts/sync_bench.py) (AGENTS.md Step A). The repository must sit at the same path on both Macs, because `mlx.launch` runs the script on each one:

```bash
<mlx-env>/bin/mlx.launch --backend jaccl --hostfile hosts-jaccl.json --python <mlx-env>/bin/python scripts/sync_bench.py --rows 1
<mlx-env>/bin/mlx.launch --backend jaccl --hostfile hosts-jaccl.json --env MLX_METAL_FAST_SYNCH=1 --python <mlx-env>/bin/python scripts/sync_bench.py --rows 1
# ours (quiet Macs): ADDED COST per sum 187.8 us default, 19.0 us with MLX_METAL_FAST_SYNCH=1
```

Never set `MLX_METAL_FAST_SYNCH=1` for 8-bit, FP8 or MXFP8 model runs ([`LESSONS.md`](LESSONS.md), Lesson 1).
