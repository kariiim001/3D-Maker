from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from ..config import CALL_TIMEOUT, DEFAULT_OUTPUT_DIR, DEFAULT_REMBG_MODEL, QUALITY_LEVELS, SUPPORTED_EXTS
from ..context import LEAKED_JOB
from ..errors import ImageTo3DError, JobCancelled
from ..pipeline import convert_image_to_3d
from ..providers import PROVIDERS
from ..remote.classify import classify_exception
from ..remote.client import make_client
from ..utils import env_token
from .web import launch_ui


def pick_image_dialog() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        patterns = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTS))
        chosen = filedialog.askopenfilename(title="Choose an image", filetypes=[("Images", patterns), ("All files", "*.*")])
        root.destroy()
        return chosen or None
    except Exception:
        try:
            return input("Path to the image: ").strip().strip('"') or None
        except EOFError:
            return None


def inspect_space(space_id: str, token: Optional[str]) -> int:
    """Print the API of a Space: the first thing to check when a backend stops working."""
    try:
        client = make_client(space_id, token or env_token(), None)
        client.view_api(print_info=True)
        return 0
    except Exception as exc:
        print(f"ERROR: {classify_exception(exc)}", file=sys.stderr)
        return 1


def run_cli(args: argparse.Namespace) -> int:
    path = args.image
    if args.pick:
        path = pick_image_dialog()
        if not path:
            print("No image chosen.", file=sys.stderr)
            return 2
    t0 = time.monotonic()

    def say(msg: str) -> None:
        print(f"[{time.monotonic() - t0:5.0f}s] {msg}", flush=True)

    try:
        result = convert_image_to_3d(
            path, provider=args.provider, remove_bg=not args.no_rembg, rembg_model=args.rembg_model,
            quality=args.quality, seed=args.seed, textured=not args.no_texture, out_format=args.out_format,
            space=args.space, token=args.token, timeout=args.timeout,
            output_dir=Path(args.output_dir) if args.output_dir else None, on_status=say, verbose=args.verbose)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except JobCancelled:
        print("Cancelled.", file=sys.stderr)
        return 130
    except ImageTo3DError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    print(f"\nDone in {result.seconds:.0f}s using {result.space_id} (seed {result.seed}). Files:")
    for f in result.files:
        print(f"  {f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="image-to-3d",
        description="Convert an image into a 3D model (.glb/.obj) using free Hugging Face Spaces. "
                    "Without --image/--pick it starts a local web UI.",
        epilog="Tip: export HF_TOKEN=hf_xxx (free token) to avoid the tiny anonymous GPU quota.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("-i", "--image", metavar="PATH", help="image to convert (command-line mode)")
    mode.add_argument("--pick", action="store_true", help="choose the image in a file dialog (command-line mode)")
    mode.add_argument("--inspect", metavar="OWNER/SPACE", help="print a Space's API and exit")

    p.add_argument("-p", "--provider", default="auto", choices=["auto", *PROVIDERS], help="backend (default: auto)")
    p.add_argument("-q", "--quality", default="balanced", choices=QUALITY_LEVELS)
    p.add_argument("-f", "--format", dest="out_format", default="glb", choices=["glb", "obj"],
                   help="obj needs `pip install trimesh` (a GLB is always kept as well)")
    p.add_argument("--no-rembg", action="store_true", help="skip local background removal")
    p.add_argument("--rembg-model", default=DEFAULT_REMBG_MODEL, help="rembg model (default: %(default)s)")
    p.add_argument("--no-texture", action="store_true", help="Hunyuan3D: skip texturing (faster, less quota)")
    p.add_argument("--seed", type=int, default=None, help="fixed seed (default: random)")
    p.add_argument("--space", metavar="OWNER/NAME", help="override the Space id (needs an explicit --provider)")
    p.add_argument("--token", default=None, help="Hugging Face token (prefer the HF_TOKEN environment variable)")
    p.add_argument("--timeout", type=float, default=CALL_TIMEOUT, help="max seconds per remote call (default: %(default)s)")
    p.add_argument("--output-dir", metavar="DIR", help=f"where to save results (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("-v", "--verbose", action="store_true", help="print extra diagnostics")

    ui = p.add_argument_group("web UI options")
    ui.add_argument("--host", default="127.0.0.1", help="interface to listen on (default: %(default)s)")
    ui.add_argument("--port", type=int, default=None, help="port (default: first free one from 7860)")
    ui.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    ui.add_argument("--share", action="store_true", help="create a temporary public gradio.live link")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # never crash on exotic characters in Windows consoles
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    if args.inspect:
        return inspect_space(args.inspect, args.token)
    if args.image or args.pick:
        return run_cli(args)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else DEFAULT_OUTPUT_DIR.resolve()
    launch_ui(output_dir, args.host, args.port, not args.no_browser, args.share)
    return 0


def run() -> None:
    """Console entry point (`image-to-3d` and `python -m image_to_3d`)."""
    exit_code = main()
    if LEAKED_JOB.is_set():
        # A timed-out or interrupted remote job may still be blocked in a worker thread that
        # Python would wait for at exit; leave right away instead of hanging.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    sys.exit(exit_code)
