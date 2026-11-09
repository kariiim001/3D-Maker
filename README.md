# image-to-3d

Turn a photo into a 3D model (`.glb` / `.obj`) without a local GPU. The heavy AI runs on free public
Hugging Face Spaces (Microsoft TRELLIS / TRELLIS.2, Tencent Hunyuan3D) through `gradio_client`; your
machine only does optional background removal (`rembg`, CPU) and uploads/downloads a few MB.

## Install (Python 3.10+)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[all]"          # or pick extras: .[ui]  .[rembg]  .[obj]
cp .env.example .env             # then put your HF token in the environment (see below)
```

## Run

```bash
image-to-3d                          # local web UI
image-to-3d --image photo.jpg        # command line
image-to-3d --pick                   # native file chooser
image-to-3d --inspect JeffreyXiang/TRELLIS   # print a Space's API (debugging)
python -m image_to_3d ...            # same thing
python run.py ...                    # same thing, without installing
```

As a library:

```python
from image_to_3d import convert_image_to_3d
result = convert_image_to_3d("photo.jpg", provider="auto", quality="balanced")
print(result.files)
```

## Hugging Face token

Spaces use ZeroGPU, which gives anonymous users a tiny quota. Create a free token at
https://huggingface.co/settings/tokens and export it:

```bash
export HF_TOKEN=hf_xxx                 # PowerShell: $env:HF_TOKEN = "hf_xxx"
```

Never commit a token to source control.

## Architecture

```
src/image_to_3d/
  config.py            constants, limits, provider -> Space ids, fallback order
  errors.py            exception hierarchy (str(error) is user-safe)
  context.py           RunContext: progress callback, cancellation, timeout
  utils.py             tiny helpers (short(), env_token())
  pipeline.py          convert_image_to_3d(): the one entry point both front ends call
  preprocess/          image_io.py (validate/EXIF/downscale) - background.py (rembg)
  remote/              compat, client, classify (error mapping), retry, files, session
  providers/           base - trellis (v1 + v2) - hunyuan - registry (fallback chain)
  export/              mesh.py (validate GLB, GLB -> OBJ)
  interfaces/          cli.py - web.py (Gradio, lazily imported)
tests/                 offline tests (fake backend; no network or GPU)
```

Dependency direction (no cycles): `interfaces -> pipeline -> providers -> remote -> context/errors/config/utils`.

**Add a new backend:** subclass `Provider` in `providers/`, register it in `providers/registry.py`,
add its Space id to `PROVIDER_SPACES` (and `AUTO_ORDER`) in `config.py`.
**A Space broke or moved:** change its id in `config.py` or pass `--space`; use `--inspect` to see its API.

## Tests

```bash
pip install -e ".[dev]" && pytest
```
