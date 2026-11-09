"""Central configuration: tunables, limits, and which Hugging Face Spaces back each provider."""
from __future__ import annotations

import os
from pathlib import Path

INSTALL_HINT = 'pip install "image-to-3d[all]"'

# Results go to ./outputs (relative to where you run the tool) unless overridden.
DEFAULT_OUTPUT_DIR = Path(os.environ.get("IMAGE_TO_3D_OUTPUT", Path.cwd() / "outputs")).expanduser()

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_FILE_MB = 30            # refuse absurdly large uploads
MAX_SIDE = 1024             # images are downscaled to this before upload (saves bandwidth)
MAX_SEED = 9_999_999        # every supported Space accepts seeds in this range
DEFAULT_REMBG_MODEL = os.environ.get("REMBG_MODEL", "silueta")  # 43 MB; "u2net"/"isnet-general-use" are bigger/better

CONNECT_TIMEOUT = 20.0      # seconds to open a connection to a Space
READ_TIMEOUT = 300.0        # a stalled stream is abandoned after this long (Gradio sends heartbeats)
CALL_TIMEOUT = 900.0        # max seconds for ONE remote call, queue wait included
CONNECT_RETRIES = 3         # attempts to reach a Space before giving up on it
POLL_SECONDS = 0.5          # how often the job is polled for queue/progress info
HEARTBEAT_SECONDS = 20      # "still working" line while a remote call runs

QUALITY_LEVELS = ("fast", "balanced", "high")
MESH_EXTS = (".glb", ".gltf", ".obj", ".ply", ".stl")

# Public Spaces used as backends.  Spaces get renamed or retired now and then: if one
# stops working, change its id here (or pass --space) -- nothing else needs to change.
PROVIDER_SPACES: dict[str, tuple[str, ...]] = {
    "trellis": ("JeffreyXiang/TRELLIS",),
    "trellis2": ("microsoft/TRELLIS.2",),
    "hunyuan3d": ("tencent/Hunyuan3D-2.1", "tencent/Hunyuan3D-2"),
}
# Order tried by --provider auto.  TRELLIS.2 is last because it asks the most GPU time.
AUTO_ORDER = ("trellis", "hunyuan3d", "trellis2")
