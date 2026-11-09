"""Optional CPU-only background removal with rembg."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from PIL import Image

from ..config import DEFAULT_REMBG_MODEL
from ..errors import PreprocessError
from ..utils import short

_REMBG_LOCK = threading.Lock()
_REMBG_SESSIONS: dict[str, Any] = {}  # model name -> loaded session (loaded once, reused)


def remove_background(image_path: Path, work_dir: Path, model_name: str = DEFAULT_REMBG_MODEL) -> Path:
    """
    Cut the subject out with rembg and return a PNG with an alpha channel.
    Raises PreprocessError (never anything else) so the caller can simply carry on without it.
    """
    try:
        from rembg import new_session, remove
    except (Exception, SystemExit) as exc:
        # Plain `pip install rembg` (no extra) installs without an onnxruntime backend, and rembg then
        # calls sys.exit(1) on import -- catch that too so it can't kill the whole app.
        reason = "it has no onnxruntime backend" if isinstance(exc, SystemExit) else short(exc, 100)
        raise PreprocessError(f'rembg is not available ({reason}). Install with: pip install "rembg[cpu]"') from exc

    with _REMBG_LOCK:
        session = _REMBG_SESSIONS.get(model_name)
        if session is None:
            try:
                session = new_session(model_name)  # downloads the ONNX model on first use
            except Exception as exc:
                raise PreprocessError(f"could not load rembg model '{model_name}': {short(exc)}") from exc
            _REMBG_SESSIONS[model_name] = session

    try:
        with Image.open(image_path) as im:
            im.load()
            cut = remove(im.convert("RGB"), session=session)
        cut = cut.convert("RGBA")
    except Exception as exc:
        raise PreprocessError(f"background removal failed: {short(exc)}") from exc

    # Sanity check: an almost empty mask means rembg found nothing -> better to use the original.
    hist = cut.getchannel("A").histogram()
    opaque_ratio = sum(hist[128:]) / max(1, cut.width * cut.height)
    if opaque_ratio < 0.01:
        raise PreprocessError("rembg found no foreground object in the image")

    out = work_dir / "cutout.png"
    cut.save(out, format="PNG")
    return out
