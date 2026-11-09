"""Input handling: validate, load (EXIF-aware), downscale and normalise the photo."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import MAX_FILE_MB, MAX_SIDE, SUPPORTED_EXTS
from ..errors import InputError
from ..utils import short

_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def validate_image_path(path: Any) -> Path:
    """Return a resolved Path to an existing, reasonably sized image file or raise InputError."""
    if not path:
        raise InputError("No image selected.")
    p = Path(str(path)).expanduser()
    if not p.exists():
        raise InputError(f"Image not found: {p}")
    if not p.is_file():
        raise InputError(f"Not a file: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTS:
        raise InputError(f"Unsupported file type '{p.suffix}'. Use one of: {', '.join(sorted(SUPPORTED_EXTS))}")
    size_mb = p.stat().st_size / 1_048_576
    if size_mb > MAX_FILE_MB:
        raise InputError(f"Image is {size_mb:.0f} MB; the limit is {MAX_FILE_MB} MB.")
    return p.resolve()


def load_image(path: Path) -> Image.Image:
    """Open an image fully into memory, honouring EXIF rotation (phone photos!)."""
    try:
        with Image.open(path) as im:
            return ImageOps.exif_transpose(im)  # returns a loaded copy
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InputError(f"Could not read '{path.name}' as an image: {short(exc)}") from exc


def prepare_image(src: Path, work_dir: Path) -> tuple[Path, bool]:
    """
    Normalise the input: fix rotation, downscale to MAX_SIDE, save as PNG in work_dir.
    Returns (png_path, has_alpha) where has_alpha means the image already has real transparency.
    """
    img = load_image(src)
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), _LANCZOS)
    if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        img = img.convert("RGBA")
    else:
        img = img.convert("RGB")
    has_alpha = img.mode == "RGBA" and img.getchannel("A").getextrema()[0] < 255
    if img.mode == "RGBA" and not has_alpha:
        img = img.convert("RGB")  # fully opaque alpha channel carries no information
    out = work_dir / "input.png"
    img.save(out, format="PNG")
    return out, has_alpha
