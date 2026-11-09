#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from image_to_3d.interfaces.cli import run  # noqa: E402

if __name__ == "__main__":
    run()
