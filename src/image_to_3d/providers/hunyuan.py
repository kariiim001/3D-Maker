"""Tencent Hunyuan3D provider."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from gradio_client import handle_file

from ..errors import BackendError, BackendTimeoutError
from ..remote.files import pick_mesh
from ..utils import short
from .base import GenerationParams, Provider


class Hunyuan3DProvider(Provider):
    """
    Tencent Hunyuan3D (2.1 / 2.0 Spaces).  One call does everything:
        /generation_all   -> (untextured mesh, textured mesh, ...)
        /shape_generation -> (untextured mesh, ...)
    """

    key = "hunyuan3d"
    label = "Hunyuan3D (Tencent)"
    OCTREE = {"fast": 192, "balanced": None, "high": 384}  # None = the Space's default (256)

    def _wanted(self, image_path: Path, params: GenerationParams) -> dict[str, Any]:
        wanted: dict[str, Any] = {
            "caption": "", "image": handle_file(str(image_path)),
            "mv_image_front": None, "mv_image_back": None, "mv_image_left": None, "mv_image_right": None,
            "seed": params.seed, "randomize_seed": False,
            # If we already cut the subject out, don't let the Space run its own remover again.
            "check_box_rembg": not params.has_alpha,
        }
        if self.OCTREE[params.quality]:
            wanted["octree_resolution"] = self.OCTREE[params.quality]
        return wanted

    def generate(self, image_path: Path, params: GenerationParams) -> Path:
        s, ctx = self.session, self.ctx
        use_texture = params.textured and s.has_endpoint("/generation_all")
        if params.textured and not use_texture:
            ctx.log("This Space has no texturing endpoint; generating an untextured shape.")
        if use_texture:
            try:
                res = s.call("/generation_all", self._wanted(image_path, params), label="Generating textured 3D (GPU)")
                return pick_mesh(res, last=True)  # the textured mesh comes after the plain one
            except BackendTimeoutError:
                raise
            except BackendError as exc:
                ctx.log(f"Textured generation failed ({short(exc, 120)}); retrying without texture.")
        res = s.call("/shape_generation", self._wanted(image_path, params), label="Generating 3D shape (GPU)")
        return pick_mesh(res)
