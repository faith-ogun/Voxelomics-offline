"""
Loads local environment files for mdt-command-service.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_service_env() -> None:
    """
    Loads service-local `.env` and `.env.local` if present.
    Existing shell exports take precedence.
    """
    service_dir = Path(__file__).resolve().parent
    load_dotenv(service_dir / ".env", override=False)
    load_dotenv(service_dir / ".env.local", override=False)
