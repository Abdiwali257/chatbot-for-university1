"""Start one Django development server instance for the project."""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8000


def port_is_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.5)
        return connection.connect_ex((HOST, PORT)) == 0


def main() -> int:
    if port_is_open():
        print(f"SIMAD Django server is already running at http://{HOST}:{PORT}/")
        print("Stop the existing server before starting another one.")
        return 0

    migration = subprocess.run(
        [sys.executable, "manage.py", "migrate"],
        cwd=PROJECT_DIR,
        check=False,
    )
    if migration.returncode:
        return migration.returncode

    return subprocess.call(
        [sys.executable, "manage.py", "runserver", f"{HOST}:{PORT}", "--noreload"],
        cwd=PROJECT_DIR,
    )


if __name__ == "__main__":
    raise SystemExit(main())
