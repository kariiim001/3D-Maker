"""Per-run shared state: progress sink, cancel flag, timeouts."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .config import CALL_TIMEOUT
from .errors import JobCancelled

LEAKED_JOB = threading.Event()


@dataclass
class RunContext:
    """Everything a running conversion shares: progress sink, cancel flag, timeout."""

    on_status: Callable[[str], None] = lambda _msg: None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    call_timeout: float = CALL_TIMEOUT
    verbose: bool = False

    def log(self, msg: str) -> None:
        self.on_status(msg)

    def debug(self, msg: str) -> None:
        if self.verbose:
            self.on_status(f"(debug) {msg}")

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled("Cancelled.")

    def sleep(self, seconds: float) -> None:
        """sleep() that reacts to cancellation within a quarter of a second."""
        end = time.monotonic() + seconds
        while (remaining := end - time.monotonic()) > 0:
            self.check_cancel()
            time.sleep(min(0.25, remaining))
