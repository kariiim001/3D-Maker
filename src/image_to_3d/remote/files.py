"""Locate local files inside the (variously shaped) results returned by a Space."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from ..config import MESH_EXTS
from ..errors import BackendError


def _iter_strings(obj: Any) -> Iterator[str]:
    """Yield every string / path found anywhere inside a nested API result."""
    if obj is None:
        return
    if isinstance(obj, (str, os.PathLike)):
        yield os.fspath(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_strings(value)
    elif isinstance(obj, (list, tuple, set)):
        for value in obj:
            yield from _iter_strings(value)


def find_files(result: Any, exts: Optional[Sequence[str]] = None) -> list[Path]:
    """
    Locate local files inside an API result, whatever its shape (path string, FileData dict,
    tuple of several outputs, ...).  Spaces change their return shapes now and then, so we
    search instead of indexing result[1].
    """
    seen: set[str] = set()
    found: list[Path] = []
    for text in _iter_strings(result):
        if len(text) > 1024 or "\n" in text:
            continue
        if exts and not text.lower().endswith(tuple(exts)):
            continue
        try:
            p = Path(text)
            if not p.is_file():
                continue
            key = str(p.resolve())
        except (OSError, ValueError):
            continue
        if key not in seen:
            seen.add(key)
            found.append(p)
    return found


def pick_mesh(result: Any, *, last: bool = False) -> Path:
    meshes = find_files(result, MESH_EXTS)
    if not meshes:
        raise BackendError("The Space finished but returned no 3D file.")
    return meshes[-1] if last else meshes[0]
