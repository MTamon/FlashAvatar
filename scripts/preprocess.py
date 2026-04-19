#!/usr/bin/env python3
"""Thin CLI entry that forwards to preprocess.cli.main."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/preprocess.py ...` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocess.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
