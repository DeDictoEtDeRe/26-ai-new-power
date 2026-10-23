"""Shared pytest configuration for AI4I 2020 tests.

Ensures the project root and the ``src/`` package are importable so that
``from src.data import ...`` and the pipeline modules work from any CWD.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
