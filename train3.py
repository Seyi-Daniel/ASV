#!/usr/bin/env python3
"""Compatibility shim that delegates to the refactored training entry point."""
from __future__ import annotations

from scripts.train import main


if __name__ == "__main__":
    main()

