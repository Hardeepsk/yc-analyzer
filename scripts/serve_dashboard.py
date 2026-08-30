#!/usr/bin/env python3
"""YC Analyzer - Serve Streamlit Dashboard."""

import sys
import subprocess
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

if __name__ == "__main__":
    dashboard_path = Path(__file__).resolve().parent.parent / "src" / "yc_analyzer" / "dashboard" / "app.py"
    subprocess.run([
        "streamlit", "run", str(dashboard_path),
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
    ])