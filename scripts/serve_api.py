#!/usr/bin/env python3
"""YC Analyzer - Serve FastAPI REST API."""

import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import uvicorn
from yc_analyzer.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "yc_analyzer.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        workers=1,
    )