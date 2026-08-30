#!/usr/bin/env python3
"""YC Analyzer - Start the FastAPI server."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import uvicorn
from yc_analyzer.config import settings

if __name__ == "__main__":
    print(f"Starting YC Analyzer API on {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        "yc_analyzer.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
