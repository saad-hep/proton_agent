"""Stable entry point for the proton-decay agent.

The implementation lives in :mod:`proton_agent_runtime`; aliasing the module
keeps existing imports and monkeypatch-based integrations fully compatible.
"""

import sys

import proton_agent_runtime as _runtime


if __name__ == "__main__":
    raise SystemExit(_runtime.main())

sys.modules[__name__] = _runtime
