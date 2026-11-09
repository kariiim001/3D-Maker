from __future__ import annotations

import random
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .config import (CALL_TIMEOUT, DEFAULT_OUTPUT_DIR, DEFAULT_REMBG_MODEL, MAX_SEED, QUALITY_LEVELS)
from .context import RunContext
from .errors import AllBackendsFailed, ImageTo3DError, InputError, JobCancelled, PreprocessError
from .export.mesh import export_obj, validate_mesh_file
from .preprocess.background import remove_background
from .preprocess.image_io import prepare_image, validate_image_path
from .providers import GenerationParams, Provider, resolve_chain
from .remote.classify import classify_exception
from .remote.session import SpaceSession
from .utils import env_token, short


@dataclass
class ConversionResult:
    files: list[Path]                 
    preview: Path                     
    provider: str
    space_id: str
    seed: int
    seconds: float
    cutout: Optional[Path] = None     

def _temp_dir() -> "tempfile.TemporaryDirectory[str]":
    try:
        return tempfile.TemporaryDirectory(prefix="image_to_3d_", ignore_cleanup_errors=True)
    except TypeError:  
        return tempfile.TemporaryDirectory(prefix="image_to_3d_")


def convert_image_to_3d(
    image_path: Any,
    *,
    provider: str = "auto",
    remove_bg: bool = True,
    rembg_model: str = DEFAULT_REMBG_MODEL,
    quality: str = "balanced",
    seed: Optional[int] = None,
    textured: bool = True,
    out_format: str = "glb",
    space: Optional[str] = None,
    token: Optional[str] = None,
    timeout: float = CALL_TIMEOUT,
    output_dir: Optional[Path] = None,
    on_status: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    verbose: bool = False,
) -> ConversionResult:
    """
    The whole job: validate -> prepare -> (remove background) -> remote generation with
    fallback -> save into outputs/.  Used by both the web UI and the command line.
    Raises ImageTo3DError subclasses only.
    """
    ctx = RunContext(on_status=on_status or (lambda _m: None), cancel_event=cancel_event or threading.Event(),
                     call_timeout=timeout, verbose=verbose)
    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputError(f"Cannot create output folder {out_dir}: {exc}") from exc

    src = validate_image_path(image_path)
    quality = quality if quality in QUALITY_LEVELS else "balanced"
    out_format = (out_format or "glb").lower().lstrip(".")
    if out_format not in ("glb", "obj"):
        raise InputError("Output format must be 'glb' or 'obj'.")
    chain = resolve_chain(provider, space)
    token = token or env_token()
    seed = random.randint(0, MAX_SEED) if seed is None or seed < 0 else min(int(seed), MAX_SEED)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    t0 = time.monotonic()

    with _temp_dir() as tmp:
        work = Path(tmp)

        ctx.log(f"Preparing {src.name} ...")
        prepared, has_alpha = prepare_image(src, work)

        cutout_saved: Optional[Path] = None
        if has_alpha:
            ctx.log("Image already has transparency; skipping background removal.")
        elif remove_bg:
            ctx.log(f"Removing background with rembg ({rembg_model}, CPU) ...")
            try:
                cut = remove_background(prepared, work, rembg_model)
                prepared, has_alpha = cut, True
                cutout_saved = out_dir / f"{src.stem}_{stamp}_cutout.png"
                shutil.copy2(cut, cutout_saved)
            except PreprocessError as exc:
                ctx.log(f"Background removal skipped: {exc}. The 3D Space will try on its own.")
        else:
            ctx.log("Background removal disabled; the 3D Space will handle it.")

        params = GenerationParams(seed=seed, quality=quality, textured=textured, has_alpha=has_alpha)
        attempts: list[tuple[str, ImageTo3DError]] = []
        mesh: Optional[Path] = None
        used: Optional[tuple[type[Provider], str]] = None

        for provider_cls, space_id in chain:
            ctx.check_cancel()
            session: Optional[SpaceSession] = None
            try:
                session = SpaceSession.connect(space_id, token, work, ctx)
                ctx.log(f"Connected to {space_id} ({provider_cls.label}).")
                mesh = validate_mesh_file(provider_cls(session, ctx).generate(prepared, params))
                used = (provider_cls, space_id)
                break
            except JobCancelled:
                raise
            except Exception as exc:
                err = classify_exception(exc)
                attempts.append((space_id, err))
                ctx.log(f"{space_id} failed: {short(err, 300)}")
                mesh = None
            finally:
                if session is not None:
                    session.close()

        if mesh is None or used is None:
            raise AllBackendsFailed(attempts)

        provider_cls, space_id = used
        glb_dest = out_dir / f"{src.stem}_{provider_cls.key}_{stamp}{mesh.suffix.lower()}"
        shutil.copy2(mesh, glb_dest)
        files = [glb_dest]
        if out_format == "obj" and glb_dest.suffix.lower() != ".obj":
            try:
                files = export_obj(glb_dest) + [glb_dest]
            except ImageTo3DError as exc:
                ctx.log(f"{exc} Keeping the GLB instead.")

    return ConversionResult(files=files, preview=glb_dest, provider=provider_cls.key, space_id=space_id,
                            seed=seed, seconds=time.monotonic() - t0, cutout=cutout_saved)
