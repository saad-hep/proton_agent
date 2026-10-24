"""Locate installed resources separately from writable calculation results."""

import os
from pathlib import Path

import proton_agent_resources

SOURCE_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = Path(proton_agent_resources.__file__).resolve().parent
BENCHMARK_NAMES = {"nonsusy-SU5.txt", "nonsusy-SO10.txt", "susy-SU5.txt", "susy-SO10.txt"}


def project_root():
    """Retain checkout behavior; installed packages use the working directory."""
    configured = os.environ.get("PROTON_AGENT_WORKDIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if SOURCE_DIR.name == "main" and (SOURCE_DIR / "pyproject.toml").is_file():
        return SOURCE_DIR.parent
    return Path.cwd().resolve()


def resource_path(relative):
    """Editable installs read source prompts; wheels read packaged resources."""
    candidate = SOURCE_DIR / relative
    return candidate if candidate.is_file() else RESOURCE_DIR / relative


def bundled_benchmark_path(path):
    """Fallback only for the four named benchmark examples, never custom files."""
    candidate = Path(path)
    if candidate.name in BENCHMARK_NAMES and candidate.as_posix() in {
        f"input/{candidate.name}", f"/input/{candidate.name}"
    }:
        return RESOURCE_DIR / "input" / candidate.name
    return None
