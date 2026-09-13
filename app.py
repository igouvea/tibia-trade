"""Compatibility shim. Prefer: python -m uvicorn app.main:app --host 0.0.0.0 --port 8765"""
from app.main import app

__all__ = ["app"]
