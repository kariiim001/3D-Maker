"""Map whatever gradio_client / httpx / a Space raised onto our own exception types."""
from __future__ import annotations

import concurrent.futures
import re
from typing import Optional

import httpx

from ..errors import (AuthError, BackendAPIChangedError, BackendError, BackendTimeoutError,
                      BackendUnavailableError, ImageTo3DError, QuotaError)
from ..utils import short
from .compat import AuthenticationError

_QUOTA_RE = re.compile(r"gpu quota|zerogpu|no gpu was available|quota|queue (?:is )?full|too many requests|rate.?limit|\b429\b", re.I)
_API_RE = re.compile(r"cannot find a function|no value provided for required argument|unexpected keyword|expected \d+ (?:arguments|inputs)", re.I)
_NETWORK_RE = re.compile(r"connection|timed? ?out|name or service|temporary failure|network|ssl|reset by peer|bad gateway|service unavailable|gateway time-?out|\b50[234]\b", re.I)
_MISSING_RE = re.compile(r"could not (?:find|fetch|load)|not found|\b404\b|sleeping|paused|runtime error|building|no running space|stopped", re.I)


def _http_status(exc: BaseException) -> Optional[int]:
    return getattr(getattr(exc, "response", None), "status_code", None)


def is_transient(exc: BaseException) -> bool:
    """True for failures that are worth retrying (flaky network, 5xx), not for 'Space not found'."""
    status = _http_status(exc)
    return (
        isinstance(exc, httpx.TransportError)
        or (status is not None and status >= 500)
        or bool(_NETWORK_RE.search(str(exc)))
    )


def classify_exception(exc: BaseException) -> ImageTo3DError:
    """Map whatever gradio_client / httpx / the Space raised onto our own exception types."""
    if isinstance(exc, ImageTo3DError):
        return exc
    detail = short(exc, 400)
    low = detail.lower()
    status = _http_status(exc)
    if isinstance(exc, AuthenticationError) or status in (401, 403) or "invalid user token" in low:
        return AuthError("Hugging Face rejected the request (401/403). Check HF_TOKEN, or whether the Space is private/gated.")
    if status == 429 or _QUOTA_RE.search(detail):
        return QuotaError(detail)
    if _API_RE.search(detail):
        return BackendAPIChangedError(detail)
    if isinstance(exc, (TimeoutError, concurrent.futures.TimeoutError)):
        return BackendTimeoutError(f"Timed out: {detail}")
    if "has not enabled verbose error reporting" in low:
        return BackendError("The Space crashed while processing the request and gave no details (usually out of "
                            "GPU memory or time). Try quality 'fast', another provider, or retry later.")
    if isinstance(exc, httpx.TransportError) or (status is not None and (status >= 500 or status == 404)) \
            or _NETWORK_RE.search(detail) or _MISSING_RE.search(detail):
        return BackendUnavailableError(detail)
    return BackendError(detail)

