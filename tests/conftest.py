"""
conftest.py — shared pytest configuration.
"""
import sys
from pathlib import Path

# Ensure the project root is on PYTHONPATH for all tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
