"""Construction of gradio_client.Client across client versions."""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Optional

import httpx
from gradio_client import Client

from ..config import CONNECT_TIMEOUT, READ_TIMEOUT


def make_client(space_id: str, token: Optional[str], download_dir: Optional[Path]) -> Client:
    """
    Build a gradio_client.Client, adapting to the installed version
    (the token argument was `hf_token` in gradio_client 1.x and `token` in 2.x).
    """
    accepted = inspect.signature(Client.__init__).parameters
    kwargs: dict[str, Any] = {}
    if token:
        if "token" in accepted:
            kwargs["token"] = token
        elif "hf_token" in accepted:
            kwargs["hf_token"] = token
    if "verbose" in accepted:
        kwargs["verbose"] = False
    if "analytics_enabled" in accepted:
        kwargs["analytics_enabled"] = False
    if download_dir is not None and "download_files" in accepted:
        kwargs["download_files"] = str(download_dir)  # results land in our temp dir
    if "httpx_kwargs" in accepted:
        kwargs["httpx_kwargs"] = {"timeout": httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT)}
    return Client(space_id, **kwargs)
