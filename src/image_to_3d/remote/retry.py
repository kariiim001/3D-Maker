"""Retry helper with exponential back-off for flaky connections."""
from __future__ import annotations

from typing import Any, Callable

from ..config import CONNECT_RETRIES
from ..context import RunContext
from ..errors import JobCancelled
from ..utils import short
from .classify import is_transient


def with_retries(fn: Callable[[], Any], ctx: RunContext, what: str, attempts: int = CONNECT_RETRIES,
                 base_delay: float = 3.0) -> Any:
    """Call fn(); retry transient failures with exponential back-off (3s, 6s, ...)."""
    for attempt in range(1, attempts + 1):
        ctx.check_cancel()
        try:
            return fn()
        except JobCancelled:
            raise
        except Exception as exc:
            if attempt >= attempts or not is_transient(exc):
                raise
            delay = base_delay * 2 ** (attempt - 1)
            ctx.log(f"{what}: {short(exc, 120)} -- retrying in {delay:.0f}s ({attempt}/{attempts - 1})")
            ctx.sleep(delay)
