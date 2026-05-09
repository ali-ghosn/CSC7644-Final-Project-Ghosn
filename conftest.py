"""
conftest.py
-----------
Pytest root configuration for the AI Task Decomposition Copilot.

Placed at the project root so pytest discovers it for all test modules.

Provides:
    - sys.path configuration (ensures the project root is importable)
    - Shared fixtures used across test modules

Run all tests:
    pytest tests/ -v

Run with coverage:
    pytest tests/ -v --cov=. --cov-report=term-missing --cov-omit="tests/*,conftest.py"
"""

import sys
import os

# Ensure the project root is on sys.path so test imports resolve correctly
# regardless of how pytest is invoked (from root dir, IDE runner, CI, etc.)
sys.path.insert(0, os.path.dirname(__file__))
