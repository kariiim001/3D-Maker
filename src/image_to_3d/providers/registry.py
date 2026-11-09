"""Provider registry and the ordered fallback chain."""
from __future__ import annotations

from typing import Optional, Sequence

from ..config import AUTO_ORDER, PROVIDER_SPACES
from ..errors import InputError
from .base import Provider
from .hunyuan import Hunyuan3DProvider
from .trellis import Trellis2Provider, TrellisProvider


PROVIDERS: dict[str, type[Provider]] = {
    TrellisProvider.key: TrellisProvider,
    Trellis2Provider.key: Trellis2Provider,
    Hunyuan3DProvider.key: Hunyuan3DProvider,
}


def resolve_chain(provider: str, space: Optional[str]) -> list[tuple[type[Provider], str]]:
    """Expand a provider choice into the ordered list of (provider class, Space id) to try."""
    provider = (provider or "auto").lower()
    if provider == "auto":
        if space:
            raise InputError("--space needs an explicit --provider (not 'auto').")
        keys: Sequence[str] = AUTO_ORDER
    elif provider in PROVIDERS:
        keys = (provider,)
    else:
        raise InputError(f"Unknown provider '{provider}'. Choose from: auto, {', '.join(PROVIDERS)}.")
    chain = [(PROVIDERS[k], sp) for k in keys for sp in ((space,) if space else PROVIDER_SPACES.get(k, ()))]
    if not chain:
        raise InputError("No Space is configured for this provider. Check PROVIDER_SPACES near the top of the file.")
    return chain
