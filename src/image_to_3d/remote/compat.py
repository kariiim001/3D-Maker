"""gradio_client moved these exception classes around between releases; import them defensively."""
from __future__ import annotations

try:
    from gradio_client.exceptions import AppError
except ImportError:  # pragma: no cover
    class AppError(Exception):  # type: ignore[no-redef]
        """Placeholder so isinstance() checks keep working on old versions."""

try:
    from gradio_client.exceptions import AuthenticationError
except ImportError:  # pragma: no cover
    class AuthenticationError(Exception):  # type: ignore[no-redef]
        """Placeholder so isinstance() checks keep working on old versions."""

__all__ = ["AppError", "AuthenticationError"]
