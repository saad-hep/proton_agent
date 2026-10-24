"""Default proton-decay limits and explicitly selected, computation-scoped files."""

import ast
import copy
import hashlib
import json
import math
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

EXPERIMENTAL_LIMITS = {
    "p->pi0 e+": {"current": 2.4e34, "future": 7.8e34},
    "p->pi0 mu+": {"current": 1.6e34, "future": 7.7e34},
    "p->K0 e+": {"current": 1.1e33, "future": None},
    "p->K0 mu+": {"current": 3.6e33, "future": None},
    "p->eta e+": {"current": 1.0e34, "future": 4.3e34},
    "p->eta mu+": {"current": 4.7e33, "future": 4.9e34},
    "p->pi+ nu": {"current": 3.9e32, "future": None},
    "p->K+ nu": {"current": 6.6e33, "future": 3.2e34},
}


_ACTIVE_BOUNDS = ContextVar("proton_experimental_bounds", default=None)


def load_bounds_file(filepath):
    path = Path(filepath).resolve()
    raw = path.read_bytes()
    text = "\n".join(line.split("#", 1)[0] for line in raw.decode("utf-8").splitlines())
    try:
        supplied = json.loads(text)
    except ValueError:
        supplied = ast.literal_eval(text)
    if not isinstance(supplied, dict) or set(supplied) - {"units", "channels"}:
        raise ValueError("Bounds file must contain 'channels' and optional 'units'.")
    if supplied.get("units", "years") not in ("years", "yr"):
        raise ValueError("Experimental bounds and sensitivities must be in years.")
    channels = supplied.get("channels")
    if not isinstance(channels, dict):
        raise ValueError("Bounds file 'channels' must be a dictionary.")
    values = copy.deepcopy(EXPERIMENTAL_LIMITS)
    seen = set()
    for raw_channel, limits in channels.items():
        channel = str(raw_channel).replace("nubar", "nu")
        if channel not in values or channel in seen:
            raise ValueError(f"Unknown or duplicate bounds channel {raw_channel!r}.")
        seen.add(channel)
        if not isinstance(limits, dict) or set(limits) - {"current", "future"}:
            raise ValueError(f"Bounds for {channel} must use 'current' and/or 'future'.")
        for kind, value in limits.items():
            if kind == "future" and value is None:
                values[channel][kind] = None
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{channel}/{kind} must be positive and finite in years; only future may be null.")
            values[channel][kind] = float(value)
    return {"file": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "units": "years", "channels": values}


def current_bounds_configuration():
    return copy.deepcopy(_ACTIVE_BOUNDS.get())


@contextmanager
def using_bounds(configuration):
    token = _ACTIVE_BOUNDS.set(configuration)
    try:
        yield
    finally:
        _ACTIVE_BOUNDS.reset(token)


def call_with_bounds(configuration, function, *args, **kwargs):
    """Explicitly transport the selected limits to parallel worker tasks."""
    with using_bounds(configuration):
        return function(*args, **kwargs)


def record_bounds(result):
    configuration = current_bounds_configuration()
    if isinstance(result, dict) and configuration is not None:
        result["experimental_bounds"] = configuration
    return result


def get_experimental_limits():
    configuration = _ACTIVE_BOUNDS.get()
    return copy.deepcopy(EXPERIMENTAL_LIMITS if configuration is None else configuration["channels"])


def get_current_limits():
    """Return the current experimental lower limit for each channel."""
    return {
        channel: values["current"]
        for channel, values in get_experimental_limits().items()
    }


def get_future_limits():
    """Return projected sensitivities for channels that define one."""
    return {
        channel: values["future"]
        for channel, values in get_experimental_limits().items()
        if values["future"] is not None
    }


def get_experimental_limit(channel, kind="current"):
    """Return one current or future limit, or None for an unknown value."""
    if kind not in {"current", "future"}:
        raise ValueError("kind must be 'current' or 'future'")
    values = get_experimental_limits().get(channel.replace("nubar", "nu"))
    return values.get(kind) if values is not None else None


def get_extracted_limit_metadata(channel, kind="current"):
    """Return selected-file provenance; built-in values have no external metadata."""
    if kind not in {"current", "future"}:
        raise ValueError("kind must be 'current' or 'future'")
    configuration = current_bounds_configuration()
    if configuration is not None:
        return {key: configuration[key] for key in ("file", "sha256", "units")}
    return None

def check_exclusion(lifetimes):
    limits = get_experimental_limits()
    for ch, tau in lifetimes.items():
        if tau < limits[ch]["current"]:
            return True
    return False


def check_future_probe(lifetimes):
    limits = get_experimental_limits()
    for ch, tau in lifetimes.items():
        future = limits[ch]["future"]
        if future is not None and tau < future:
            return True
    return False


def violated_channels(lifetimes):
    limits = get_experimental_limits()
    return [
        ch for ch, tau in lifetimes.items()
        if tau < limits[ch]["current"]
    ]


def testable_channels(lifetimes):
    limits = get_experimental_limits()
    return [
        ch for ch, tau in lifetimes.items()
        if limits[ch]["future"] is not None
        and tau >= limits[ch]["current"]
        and tau < limits[ch]["future"]
    ]


def most_constraining_channel(lifetimes):
    limits = get_experimental_limits()
    ratios = {
        ch: tau / limits[ch]["current"]
        for ch, tau in lifetimes.items()
    }
    ch_min = min(ratios, key=ratios.get)
    return ch_min, ratios[ch_min]

def classify_model(lifetimes):
    if check_exclusion(lifetimes):
        return "excluded"

    if check_future_probe(lifetimes):
        return "testable"

    return "safe"
