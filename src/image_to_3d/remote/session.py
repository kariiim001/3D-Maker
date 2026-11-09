"""SpaceSession: one connection to one Space, with name-based parameter binding,
queue/progress reporting, timeouts and cancellation."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from gradio_client import Client

from ..config import HEARTBEAT_SECONDS, POLL_SECONDS
from ..context import LEAKED_JOB, RunContext
from ..errors import (BackendAPIChangedError, BackendError, BackendTimeoutError, ImageTo3DError,
                      JobCancelled)
from ..utils import short
from .classify import classify_exception
from .client import make_client
from .retry import with_retries


def _neutral_value(param: dict) -> Any:
    """A harmless value for a required parameter we have no opinion about."""
    type_info = param.get("type")
    enum = type_info.get("enum") if isinstance(type_info, dict) else None
    if enum:
        return enum[0]
    py = str((param.get("python_type") or {}).get("type") or "").lower()
    if py.startswith(("list", "tuple")):
        return []
    if py.startswith("bool"):
        return False
    if py.startswith(("int", "float")):
        return 0
    if py.startswith("str"):
        return ""
    return None


def _value_allowed(param: dict, value: Any) -> bool:
    """For Radio/Dropdown parameters: is `value` one of the choices the Space accepts?"""
    type_info = param.get("type")
    enum = type_info.get("enum") if isinstance(type_info, dict) else None
    if not enum or value is None:
        return True
    return value in enum or str(value) in [str(e) for e in enum]


class SpaceSession:
    """
    One connection to one Space.  Wraps gradio_client with:
      * parameter binding by *name* against the Space's live API description, so added/removed
        optional parameters don't break us,
      * queue-position / progress reporting, a hard timeout and cancellation.
    A session must be used for a whole generation: the Spaces keep the intermediate result
    server-side, keyed by the client's session id.
    """

    def __init__(self, space_id: str, client: Client, ctx: RunContext):
        self.space_id = space_id
        self.client = client
        self.ctx = ctx
        self._endpoints: Optional[dict[str, dict]] = None

    @classmethod
    def connect(cls, space_id: str, token: Optional[str], work_dir: Path, ctx: RunContext) -> "SpaceSession":
        try:
            client = with_retries(lambda: make_client(space_id, token, work_dir), ctx, f"Connecting to {space_id}")
        except JobCancelled:
            raise
        except Exception as exc:
            raise classify_exception(exc) from exc
        return cls(space_id, client, ctx)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        try:
            if callable(close):
                close()
            elif getattr(self.client, "executor", None) is not None:
                self.client.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    # ---- API description -----------------------------------------------------

    def endpoints(self) -> dict[str, dict]:
        if self._endpoints is None:
            try:
                info = self.client.view_api(print_info=False, return_format="dict")
            except Exception as exc:
                raise BackendError(f"Could not read the API description of {self.space_id}: {short(exc)}") from exc
            named = (info or {}).get("named_endpoints") or {}
            self._endpoints = {"/" + str(name).lstrip("/"): spec for name, spec in named.items()}
        return self._endpoints

    def has_endpoint(self, api_name: str) -> bool:
        return api_name in self.endpoints()

    def _bind(self, api_name: str, wanted: dict[str, Any]) -> dict[str, Any]:
        """Turn {name: value} into kwargs the endpoint really accepts."""
        eps = self.endpoints()
        if api_name not in eps:
            raise BackendAPIChangedError(
                f"{self.space_id} has no endpoint {api_name}. It offers: {', '.join(sorted(eps)) or 'nothing'}")
        params = eps[api_name].get("parameters") or []
        known = set()
        kwargs: dict[str, Any] = {}
        for p in params:
            name = p.get("parameter_name") or str(p.get("label", "")).lower().replace(" ", "_")
            known.add(name)
            if name in wanted and _value_allowed(p, wanted[name]):
                kwargs[name] = wanted[name]
            elif p.get("parameter_has_default", False):
                continue  # let the client fill in the Space's own default
            else:
                kwargs[name] = _neutral_value(p)
        ignored = sorted(set(wanted) - known)
        if ignored:
            self.ctx.debug(f"{api_name}: parameters not offered by this Space were skipped: {', '.join(ignored)}")
        return kwargs

    # ---- calling -------------------------------------------------------------

    def call(self, api_name: str, wanted: dict[str, Any], *, label: str, timeout: Optional[float] = None) -> Any:
        """Run one endpoint to completion; returns gradio_client's (already downloaded) result."""
        kwargs = self._bind(api_name, wanted)
        return self._run_job(api_name, kwargs, label, timeout or self.ctx.call_timeout)

    def _run_job(self, api_name: str, kwargs: dict[str, Any], label: str, timeout: float) -> Any:
        ctx = self.ctx
        ctx.check_cancel()
        ctx.log(f"{label} ...")
        try:
            job = self.client.submit(api_name=api_name, **kwargs)
        except Exception as exc:
            raise classify_exception(exc) from exc

        started = last_beat = time.monotonic()
        last_key = None
        try:
            while not job.done():
                if ctx.cancel_event.is_set():
                    job.cancel()
                    raise JobCancelled("Cancelled.")
                now = time.monotonic()
                if now - started > timeout:
                    job.cancel()
                    LEAKED_JOB.set()
                    raise BackendTimeoutError(f"{label} did not finish within {timeout:.0f}s (long queue?).")
                key, text = self._status_text(job)
                if key != last_key or now - last_beat >= HEARTBEAT_SECONDS:
                    ctx.log(f"{label}: {text} ({now - started:.0f}s)")
                    last_key, last_beat = key, now
                time.sleep(POLL_SECONDS)
            return job.result()  # re-raises whatever the Space raised
        except ImageTo3DError:
            raise
        except KeyboardInterrupt:
            job.cancel()
            LEAKED_JOB.set()
            raise
        except Exception as exc:
            raise classify_exception(exc) from exc

    @staticmethod
    def _status_text(job: Any) -> tuple[str, str]:
        """(key that changes when the state changes, human readable text)."""
        try:
            st = job.status()
        except Exception:
            return "unknown", "working"
        code_obj = getattr(st, "code", None)
        code = str(getattr(code_obj, "name", code_obj) or "UNKNOWN").upper()
        rank, size, eta = getattr(st, "rank", None), getattr(st, "queue_size", None), getattr(st, "eta", None)
        if code in ("IN_QUEUE", "JOINING_QUEUE") and rank is not None and size:
            text = f"waiting in queue, position {rank + 1} of {size}"
            if eta:
                text += f", about {eta:.0f}s"
            return f"queue-{rank}", text
        if code == "QUEUE_FULL":
            return code, "queue is full, waiting"
        if code in ("PROCESSING", "ITERATING", "PROGRESS"):
            text = "running on the remote GPU"
            progress = getattr(st, "progress_data", None)
            if progress:
                unit = progress[-1]
                if getattr(unit, "length", None):
                    text += f" ({getattr(unit, 'desc', None) or 'step'} {unit.index}/{unit.length})"
            return "running", text
        return code, code.replace("_", " ").lower()
