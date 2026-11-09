"""Output handling: validate the returned mesh and optionally convert GLB -> OBJ."""
from __future__ import annotations

from pathlib import Path

from ..errors import BackendError, ImageTo3DError
from ..utils import short


def validate_mesh_file(path: Path) -> Path:
    """Reject empty files and 'GLBs' that are really error pages."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size < 64:
        raise BackendError("The Space returned an empty or missing 3D file.")
    if p.suffix.lower() == ".glb":
        with p.open("rb") as fh:
            if fh.read(4) != b"glTF":
                raise BackendError("The Space returned a file that is not a valid GLB.")
    return p


def export_obj(glb_path: Path) -> list[Path]:
    """
    Convert a GLB to OBJ (+ .mtl and texture image) inside a sibling folder.
    Needs the optional `trimesh` package.  Returns the files, the .obj first.
    """
    try:
        import trimesh
    except ImportError as exc:
        raise ImageTo3DError("OBJ export needs the optional 'trimesh' package: pip install trimesh") from exc
    folder = glb_path.parent / f"{glb_path.stem}_obj"
    folder.mkdir(parents=True, exist_ok=True)
    obj_path = folder / f"{glb_path.stem}.obj"
    try:
        scene = trimesh.load(str(glb_path), force="scene")
        scene.export(str(obj_path))  # also writes the .mtl and texture images next to it
    except Exception as exc:
        raise ImageTo3DError(f"OBJ conversion failed: {short(exc)}") from exc
    if not obj_path.is_file():
        raise ImageTo3DError("OBJ conversion produced no file.")
    return sorted((f for f in folder.iterdir() if f.is_file()), key=lambda f: (f.suffix.lower() != ".obj", f.name))
