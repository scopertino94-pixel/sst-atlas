import sys
import os

# Add backend/ to sys.path so `from runners import ...` works inside backend/main.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from main import app  # noqa: F401  (re-export for uvicorn)
