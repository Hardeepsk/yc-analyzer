#!/usr/bin/env python3
"""YC Analyzer - Start the Streamlit dashboard."""

import subprocess
import sys
from pathlib import Path

dashboard_path = Path(__file__).resolve().parent.parent / "src" / "yc_analyzer" / "dashboard" / "app.py"

print(f"Starting YC Analyzer Dashboard on port {8501}")
subprocess.run([
    sys.executable, "-m", "streamlit", "run",
    str(dashboard_path),
    "--server.port", "8501",
    "--server.headless", "true",
])
