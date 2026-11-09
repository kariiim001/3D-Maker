"""Exception hierarchy.  str(error) is always safe to show to the end user."""
from __future__ import annotations

class ImageTo3DError(Exception):
    """Base class.  str(error) is always safe and useful to show to the end user."""


class InputError(ImageTo3DError):
    """Bad or missing input file / option."""


class PreprocessError(ImageTo3DError):
    """Local preprocessing (background removal) failed."""


class BackendError(ImageTo3DError):
    """A remote Space failed."""


class BackendUnavailableError(BackendError):
    """Space is offline, sleeping, moved, or the network is down."""


class BackendAPIChangedError(BackendError):
    """The Space no longer exposes the endpoint/parameters we expect."""


class QuotaError(BackendError):
    """ZeroGPU quota exhausted, or the Space's queue is full."""


class BackendTimeoutError(BackendError):
    """The remote call took longer than the configured timeout."""


class AuthError(BackendError):
    """Hugging Face rejected our token / access."""


class JobCancelled(ImageTo3DError):
    """The user cancelled the run."""


class AllBackendsFailed(ImageTo3DError):
    """Every candidate Space failed; carries a per-Space summary plus hints."""

    def __init__(self, attempts: list[tuple[str, ImageTo3DError]]):
        self.attempts = attempts
        kinds = {type(err) for _, err in attempts}
        lines = ["No 3D backend could produce a model:"]
        lines += [f"  - {space}: {err}" for space, err in attempts]
        hints = []
        if QuotaError in kinds:
            hints.append("GPU quota/queue limit: set HF_TOKEN to a free Hugging Face token (much bigger "
                          "ZeroGPU quota), wait for the daily reset, or retry later.")
        if AuthError in kinds:
            hints.append("Check that HF_TOKEN is valid (https://huggingface.co/settings/tokens).")
        if BackendAPIChangedError in kinds:
            hints.append("A Space changed its API: run `image-to-3d --inspect OWNER/SPACE`, "
                          "or pick another provider.")
        if BackendUnavailableError in kinds:
            hints.append("A Space looks offline, sleeping or moved: retry in a minute or edit PROVIDER_SPACES.")
        if BackendTimeoutError in kinds:
            hints.append("Timed out in the queue: retry later or raise --timeout.")
        if hints:
            lines.append("Hints:")
            lines += [f"  * {h}" for h in hints]
        super().__init__("\n".join(lines))
