"""Small shared helpers with no dependencies on the rest of the package."""
from __future__ import annotations

import os
from typing import Any, Optional


def short(obj: Any, limit: int = 200) -> str:
    text = " ".join(str(obj).split()) or type(obj).__name__
    return text if len(text) <= limit else text[: limit - 3] + "..."


def env_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
