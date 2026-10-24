"""Explicit, computation-scoped hadronic inputs; never auto-load a file."""

import ast
import copy
import hashlib
import json
import math
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path


_ACTIVE_MATRIX_ELEMENTS = ContextVar("proton_matrix_elements", default=None)


def default_matrix_elements():
    # Lazy imports keep the formula modules independent during initialization.
    from nonsusy_su5_formula import DEFAULT_HADRONIC
    from susy_su5_formula import DEFAULT_HADRONIC_MATRIX_ELEMENTS
    from susy_so10_formula import DEFAULT_HADRONIC as SO10_HADRONIC
    return {
        "non_susy": dict(DEFAULT_HADRONIC),
        "susy_su5": {"|".join(key): value for key, value in DEFAULT_HADRONIC_MATRIX_ELEMENTS.items()},
        "susy_so10": dict(SO10_HADRONIC),
    }


def load_matrix_elements_file(filepath):
    path = Path(filepath).resolve()
    raw = path.read_bytes()
    # Both JSON and the commented Python dictionary used by benchmark files
    # are accepted. literal_eval never executes file contents.
    text = "\n".join(line.split("#", 1)[0] for line in raw.decode("utf-8").splitlines())
    try:
        supplied = json.loads(text)
    except ValueError:
        supplied = ast.literal_eval(text)
    defaults = default_matrix_elements()
    if not isinstance(supplied, dict):
        raise ValueError("Matrix-element file must contain a dictionary.")
    unknown = set(supplied) - {*defaults, "units"}
    if unknown:
        raise ValueError(f"Unknown matrix-element sections: {sorted(unknown)}")
    if supplied.get("units", "GeV^2") != "GeV^2":
        raise ValueError("Matrix elements must be supplied in GeV^2.")
    for section, values in supplied.items():
        if section == "units":
            continue
        if not isinstance(values, dict):
            raise ValueError(f"{section} must contain named matrix elements.")
        for key, value in values.items():
            if key not in defaults[section]:
                raise ValueError(f"Unknown matrix element {section}/{key}.")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{section}/{key} must be a finite signed number in GeV^2.")
            defaults[section][key] = float(value)
    return {"file": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "units": "GeV^2", "values": defaults}


def current_matrix_elements():
    return copy.deepcopy(_ACTIVE_MATRIX_ELEMENTS.get())


@contextmanager
def using_matrix_elements(configuration):
    token = _ACTIVE_MATRIX_ELEMENTS.set(configuration)
    try:
        yield
    finally:
        _ACTIVE_MATRIX_ELEMENTS.reset(token)


def call_with_matrix_elements(configuration, function, *args, **kwargs):
    """Pass an immutable snapshot explicitly to a process-pool task."""
    with using_matrix_elements(configuration):
        return function(*args, **kwargs)


def hadronic_elements(section, defaults):
    configuration = _ACTIVE_MATRIX_ELEMENTS.get()
    if configuration is None:
        return dict(defaults)
    supplied = configuration["values"][section]
    return {key: supplied["|".join(key) if isinstance(key, tuple) else key] for key in defaults}


def record_matrix_elements(result):
    configuration = current_matrix_elements()
    if isinstance(result, dict) and configuration is not None:
        result["matrix_elements"] = configuration
    return result
