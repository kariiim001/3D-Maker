"""Microsoft TRELLIS (v1) and TRELLIS.2 providers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from gradio_client import handle_file

from ..errors import BackendError
from ..remote.files import find_files, pick_mesh
from ..utils import short
from .base import GenerationParams, Provider


class TrellisProvider(Provider):
    """
    Microsoft TRELLIS (v1) -- JeffreyXiang/TRELLIS.  Flow of the Space's own UI:
        /start_session -> /preprocess_image -> /image_to_3d -> /extract_glb
    The server keeps the generated latent per session, which is why the same client is reused.
    """

    key = "trellis"
    label = "TRELLIS (Microsoft)"
    # quality -> (kwargs for /image_to_3d, kwargs for /extract_glb)
    PRESETS = {
        "fast":     ({"ss_sampling_steps": 8,  "slat_sampling_steps": 8},  {"mesh_simplify": 0.97, "texture_size": 512}),
        "balanced": ({"ss_sampling_steps": 12, "slat_sampling_steps": 12}, {"mesh_simplify": 0.95, "texture_size": 1024}),
        "high":     ({"ss_sampling_steps": 20, "slat_sampling_steps": 20}, {"mesh_simplify": 0.92, "texture_size": 2048}),
    }
    BASE_3D = {"multiimages": [], "ss_guidance_strength": 7.5, "slat_guidance_strength": 3.0,
               "multiimage_algo": "stochastic"}

    def generate(self, image_path: Path, params: GenerationParams) -> Path:
        s, ctx = self.session, self.ctx
        gen_kwargs, glb_kwargs = self.PRESETS[params.quality]

        # 1. Open a session (creates the Space's per-session temp dir).  Best effort.
        if s.has_endpoint("/start_session"):
            try:
                s.call("/start_session", {}, label="Opening session", timeout=60)
            except BackendError as exc:
                ctx.log(f"Session start skipped: {short(exc, 120)}")

        # 2. Server-side preprocessing (crop/centre/resize; uses our alpha channel if present,
        #    otherwise the Space runs its own background removal).
        processed = image_path
        if s.has_endpoint("/preprocess_image"):
            try:
                res = s.call("/preprocess_image", {"image": handle_file(str(image_path))}, label="Preprocessing image")
                processed = (find_files(res) or [image_path])[0]
            except BackendError as exc:
                ctx.log(f"Server preprocessing skipped: {short(exc, 120)}")

        # 3. Generate the 3D asset on the GPU (result stays on the server).
        s.call("/image_to_3d",
               {"image": handle_file(str(processed)), "seed": params.seed, **self.BASE_3D, **gen_kwargs},
               label="Generating 3D asset (GPU)")

        # 4. Bake the GLB and download it.
        res = s.call("/extract_glb", dict(glb_kwargs), label="Exporting GLB")
        return pick_mesh(res)


class Trellis2Provider(TrellisProvider):
    """Microsoft TRELLIS.2 -- microsoft/TRELLIS.2.  Same four-step flow, PBR materials, heavier."""

    key = "trellis2"
    label = "TRELLIS.2 (Microsoft, highest quality)"
    PRESETS = {
        "fast":     ({"resolution": "512"},  {"decimation_target": 50_000,  "texture_size": 1024}),
        "balanced": ({"resolution": "512"},  {"decimation_target": 100_000, "texture_size": 2048}),
        "high":     ({"resolution": "1024"}, {"decimation_target": 200_000, "texture_size": 2048}),
    }
    BASE_3D: dict[str, Any] = {}  # per-stage sampler settings: keep the Space's defaults

