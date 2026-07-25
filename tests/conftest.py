"""
pytest configuration file.

Purpose:
    Test files live in tests/, but the code being tested lives in src/.
    This file adds src/ to Python's search path so that test files can
    write "from text_utils import ..." instead of a longer, fragile
    relative import path. pytest automatically loads any conftest.py
    it finds.
"""

import sys
from pathlib import Path

SRC_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))