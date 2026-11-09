"""Provider base class and shared generation parameters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..context import RunContext
from ..remote.session import SpaceSession


@dataclass
class GenerationParams:
    seed: int
    quality: str = "balanced"
    textured: bool = True    # Hunyuan3D: texture the mesh (slower, more GPU quota)
    has_alpha: bool = False  # True when the input already has a clean cut-out


class Provider:
    """One family of Spaces that share an API.  Subclasses implement generate()."""

    key = ""
    label = ""

    def __init__(self, session: SpaceSession, ctx: RunContext):
        self.session = session
        self.ctx = ctx

    def generate(self, image_path: Path, params: GenerationParams) -> Path:  # pragma: no cover
        raise NotImplementedError
