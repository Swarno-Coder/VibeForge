#!/usr/bin/env python3
"""
Multi-Agent System — Unified Launcher

Usage:
    python run.py api    → Start FastAPI server (port 8000)
    python run.py ui     → Start Streamlit app (port 8501)
    python run.py cli    → Start CLI interface
    python run.py        → Show help
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def run_api():
    """Start the FastAPI API server."""
    print("🚀 Starting Multi-Agent API Server on http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    print("   Endpoints: POST /api/query, GET /api/health, GET /api/history")
    print("─" * 60)
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "api.server:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=str(PROJECT_ROOT),
    )


def run_ui():
    """Start the Streamlit web UI."""
    print("🎨 Starting VibeForge AI on http://localhost:8501")
    print("─" * 60)
    app_path = str(PROJECT_ROOT / "ui" / "app.py")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path,
         "--server.port", "8501",
         "--server.headless", "true",
         "--theme.base", "dark",
         "--theme.primaryColor", "#6366f1",
         "--theme.backgroundColor", "#0a0a0f",
         "--theme.secondaryBackgroundColor", "#12121a",
         "--theme.textColor", "#e2e8f0"],
        cwd=str(PROJECT_ROOT),
    )


def run_cli():
    """Start the CLI interface."""
    print("⌨️  Starting Multi-Agent CLI")
    print("─" * 60)
    # Add project root to path and run CLI
    sys.path.insert(0, str(PROJECT_ROOT))
    from cli.main import main
    main()


def show_help():
    print("""
╔══════════════════════════════════════════════════════════╗
║        🧠 Multi-Agent System — Unified Launcher         ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Usage:                                                  ║
║    python run.py api   → FastAPI server (port 8000)      ║
║    python run.py ui    → Streamlit app  (port 8501)      ║
║    python run.py cli   → CLI interface                   ║
║                                                          ║
║  API Docs:  http://localhost:8000/docs                   ║
║  Web UI:    http://localhost:8501                         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_help()
        sys.exit(0)

    command = sys.argv[1].lower().strip()

    if command == "api":
        run_api()
    elif command == "ui":
        run_ui()
    elif command == "cli":
        run_cli()
    else:
        print(f"❌ Unknown command: '{command}'")
        show_help()
        sys.exit(1)
