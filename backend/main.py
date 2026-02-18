"""Application entry point for NavispaceAI backend.

Run locally:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from backend.api import create_app


def _load_local_env() -> None:
    """Load key=value pairs from local .env files if present.

    Priority (first existing file wins per key if env var was unset):
    1) backend/.env
    2) .env
    """
    candidates = [Path("backend/.env"), Path(".env")]

    for env_path in candidates:
        if not env_path.exists():
            continue

        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip().strip("'").strip('"')


_load_local_env()
app = create_app()


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload_enabled = os.getenv("API_RELOAD", "true").lower() == "true"
    uvicorn.run("backend.main:app", host=host, port=port, reload=reload_enabled)
