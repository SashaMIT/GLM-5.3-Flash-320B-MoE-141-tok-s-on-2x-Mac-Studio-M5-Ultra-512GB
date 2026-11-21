# Python environments

Exact package lists for the two Python environments behind the results, taken read-only from both Macs with `uv pip freeze` (uv `0.13.9`). Both Macs had identical lists. Python was CPython `3.13.18` (arm64) in both environments (`python-version.txt`).

| File | Environment | What it is | SHA-256 of the file |
|---|---|---|---|
| `mlx-env.freeze.txt` | `<mlx-env>`: stock MLX, `mlx-vlm`, `mlx-lm`. Runs `mlx_vlm.server`, `sync_bench.py` and the tokenizer | Exact `uv pip freeze` output, unchanged | `43474b7e63833fda26229d44e60d93d57a6de5f57e3d233d08a31b19c7f00dfa` |
| `tandem-env.freeze.txt` | `<tandem-env>`: GLM-Tandem + oMLX `0.8.0rc1` | `uv pip freeze` output with one local path replaced by `<wheels-dir>` | `c7982861edb40bdeb1ede43227ad8f700928133485d4c54888bea2c782ac280e` |
| `tandem-env.requirements.txt` | `<tandem-env>` | The same list without the `omlx` line, so you can install it with `-r` after the wheel | `658be7b3c6e533f25469559031279eac7f07d75576bac065fc18d3b71ad167d7` |

TensorFold needs no Python environment: it ships as a native binary (see [`../SETUP.md`](../SETUP.md)). The benchmark scripts in [`../scripts/`](../scripts/) use only the Python standard library (any Python 3.9 or newer). The exception is `sync_bench.py`, which needs `<mlx-env>`.

## Hash check against `AGENTS.md`

- **`<mlx-env>`: matches.** `AGENTS.md` gives `43474b7e63833fda26229d44e60d93d57a6de5f57e3d233d08a31b19c7f00dfa` for `uv pip freeze --python <mlx-env>/bin/python | shasum -a 256`. The freeze taken for this kit hashes to exactly that value on both Macs. `mlx-env.freeze.txt` holds those bytes, so `shasum -a 256 env/mlx-env.freeze.txt` prints the same value.
- **`<tandem-env>`: no hash was recorded during the runs, so there is nothing to compare against.** The raw freeze differs between the two Macs in one line only: the `omlx @ file:///...` line, which holds each Mac's local wheel path (and so its user name). With that path replaced by `<wheels-dir>`, the two lists are byte-identical. The hash in the table is for the scrubbed file. A raw freeze of your own environment will not match it, because your wheel path will differ.
- In `<tandem-env>`, `mlx-lm`, `mlx-vlm`, `mlx-embeddings` and `mlx-audio` come from git commits. The `omlx 0.8.0rc1` wheel pins them; they are not PyPI releases. `transformers` (`5.22.2`) and `numpy` (`2.3.5`) also differ from `<mlx-env>`. Keep the two environments separate.

## Create `<mlx-env>`

```bash
# uv: https://docs.astral.sh/uv/ (we used 0.13.9)
uv venv --python 3.13.18 <mlx-env>
uv pip install --python <mlx-env>/bin/python -r env/mlx-env.freeze.txt

# verify: must print 43474b7e63833fda26229d44e60d93d57a6de5f57e3d233d08a31b19c7f00dfa
uv pip freeze --python <mlx-env>/bin/python | shasum -a 256
<mlx-env>/bin/python -c "import mlx.core as mx, mlx_vlm, mlx_lm; print(mx.__version__)"   # 0.32.4
```

If you prefer plain `venv` and `pip`, run `python3.13 -m venv <mlx-env>` and then `<mlx-env>/bin/pip install -r env/mlx-env.freeze.txt`. Then compare `<mlx-env>/bin/pip freeze` with the file by eye: `pip freeze` and `uv pip freeze` format some lines differently, so the hash check only works with `uv`.

## Create `<tandem-env>` (only for the two-Mac GLM-Tandem runs)

```bash
uv venv --python 3.13.18 <tandem-env>

# oMLX is not on PyPI: take the wheel from its official GitHub release and check it
mkdir -p <wheels-dir> && cd <wheels-dir>
curl -LO https://github.com/jundot/omlx/releases/download/v0.8.0rc1/omlx-0.8.0rc1-cp313-cp313-macosx_15_0_universal2.whl
echo "a416a587d12116858bfbbe67530012ce1c368aae108cfee406b638373ebf7af0  omlx-0.8.0rc1-cp313-cp313-macosx_15_0_universal2.whl" | shasum -a 256 -c -
cd -

uv pip install --python <tandem-env>/bin/python <wheels-dir>/omlx-0.8.0rc1-cp313-cp313-macosx_15_0_universal2.whl
uv pip install --python <tandem-env>/bin/python -r env/tandem-env.requirements.txt

# verify: stock MLX 0.32.4 and omlx 0.8.0rc1
uv pip freeze --python <tandem-env>/bin/python | grep -E '^(mlx|mlx-metal)==|^omlx'
# compare the whole list with env/tandem-env.freeze.txt (only the omlx path should differ)
diff <(uv pip freeze --python <tandem-env>/bin/python | sed -E 's#^omlx @ file://.*/omlx-#omlx @ file:///<wheels-dir>/omlx-#') env/tandem-env.freeze.txt
```

Never edit `site-packages` in either environment, and never install the oMLX `.dmg` app.
