"""Physics tools and LangGraph runtime for :mod:`proton_agent`."""

import sys
import builtins
import argparse
import json
import itertools
import re
import ast
import difflib
import subprocess
import hashlib
import pickle
import os
import httpx
import threading
import time
import unicodedata
from functools import wraps
from fractions import Fraction
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from typing import Annotated, TypedDict, Literal
MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from hadronic_inputs import (load_matrix_elements_file, using_matrix_elements,
    current_matrix_elements, call_with_matrix_elements, record_matrix_elements, hadronic_elements)
from runtime_paths import project_root, resource_path, bundled_benchmark_path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from ollama import ResponseError
from openai import OpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ── Physics modules ──────────────────────────────────────────────────
from nonsusy_su5_formula import (
    compute_decay_rates as compute_nonsusy_su5_decay_rates,
    PROTON_MASS,
    MASS_K_PLUS,
    MASS_K_ZERO,
    MASS_PI_ZERO,
    MASS_PI_PLUS,
    MASS_ETA,
    HADRON_PI0_UDU,
    HADRON_K0_USU,
    HADRON_ETA_UDU,
    HADRON_PI_PLUS_UDD,
    HADRON_K_PLUS_USD,
    HADRON_K_PLUS_UDS,
    ALPHA_LONG,
    ALPHA_SHORT,
)
from nonsusy_so10_formula import compute_decay_rates as compute_nonsusy_so10_decay_rates
from parameters import (
    generate_parameters,
    generate_parameters_with_ckm_pmns,
    build_phased_mixing_matrix,
)
from bounds import (classify_model, get_current_limits, get_future_limits, get_experimental_limit,
    get_extracted_limit_metadata, load_bounds_file, using_bounds, current_bounds_configuration,
    call_with_bounds, record_bounds)
from susy_su5_formula import (
    DEFAULT_M_T as DEFAULT_SUSY_SU5_M_T,
    compute_susy_su5_formula,
)
from susy_so10_formula import (
    SO10VEVParameters,
    compute_susy_so10_from_model_parameters,
)
from rge_coefficients import coefficient_summary, fraction_strings_to_float
from rge_runner import DEFAULT_MZ, DEFAULT_MZ_COUPLINGS, DEFAULT_MZ_YUKAWAS, run_gauge_couplings
from mssm_coefficients import coefficient_summary as mssm_coefficient_summary
from mssm_rge_runner import run_mssm_gauge_couplings
from extended_rge_runner import run_staged_gauge_rges, canonical_group
from extended_rge_coefficients import MODELS as EXTENDED_RGE_MODELS, coefficient_summary as extended_coefficient_summary
CURRENT_BOUNDS_YEARS = get_current_limits()

FUTURE_SENSITIVITY_YEARS = get_future_limits()

from scan_runner import (
    run_scan,
    evaluate_points_parallel,
    save_scan_results,
    results_to_dataframe,
    SCAN_SCHEMA_VERSION,
)

# Keep timing on the agent's emitted output without changing answer strings
# returned by the public API. Reset this after each user prompt so the prefix
# reports response latency rather than process uptime.
_RESPONSE_TIMER_START = time.monotonic()


def _reset_response_timer() -> None:
    global _RESPONSE_TIMER_START
    _RESPONSE_TIMER_START = time.monotonic()


def _timed_print(*args, **kwargs):
    elapsed = time.monotonic() - _RESPONSE_TIMER_START
    builtins.print(f"[{elapsed:.1f} sec]", *args, **kwargs)


print = _timed_print

SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = project_root()
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INTERPRETATION_CONTEXT_MODE = os.getenv("PROTON_AGENT_INTERPRETATION_CONTEXT", "grounded").strip().lower()
PLOT_INTENT_MODE = os.getenv("PROTON_AGENT_PLOT_INTENT_MODE", "llm_first").strip().lower()
INVALID_MCMC_CHI_SQ = 1.0e300


_DASH_TRANSLATION = str.maketrans({
    "\u00ad": "-", "\u2010": "-", "\u2011": "-", "\u2012": "-",
    "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
})


def _normalize_path_text(path: str | os.PathLike) -> str:
    """Normalize copied paths without changing their directory semantics."""
    return unicodedata.normalize("NFKC", str(path)).translate(_DASH_TRANSLATION).strip().strip("\"'")


def _matrix_elements_filepath_from_text(text: str) -> str | None:
    pattern = r"(?<![\w./-])((?:/[^\s\"'`|<>]+|[^\s\"'`|<>:=;,(){}\[\]]+)\.(?:txt|json))(?![\w.-])"
    for match in re.finditer(pattern, str(text or ""), re.I):
        path = _normalize_path_text(match.group(1))
        prefix = text[max(0, match.start() - 65):match.start()]
        if re.match(r"matrix[-_]elements", Path(path).name.lower()) or re.search(
            r"(?:matrix_elements_file|(?:hadronic\s+)?matrix[- ]elements?\s+file)\s*(?:=|:|is|at)?\s*[\"'`|]*$",
            prefix, re.I,
        ):
            return path
    return None


def _requested_matrix_elements_filepath(text: str) -> str | None:
    path = _matrix_elements_filepath_from_text(text)
    if path:
        position = text.find(path)
        prefix = text[max(0, position - 100):position]
        if re.search(r"(?:do not use|don't use|without(?: using)?|ignore|instead of)\s*"
                     r"(?:the\s+)?(?:matrix[- ]elements?\s+file\s*)?[\"'`|]*$", prefix, re.I):
            return None
    return path


def _bounds_filepath_from_text(text: str) -> str | None:
    pattern = r"(?<![\w./-])((?:/[^\s\"'`|<>]+|[^\s\"'`|<>:=;,(){}\[\]]+)\.(?:txt|json))(?![\w.-])"
    for match in re.finditer(pattern, str(text or ""), re.I):
        path = _normalize_path_text(match.group(1))
        prefix = text[max(0, match.start() - 65):match.start()]
        if re.match(r"bounds(?:[_.-]|$)", Path(path).name.lower()) or re.search(
            r"(?:bounds_file|(?:experimental\s+)?bounds?\s+file)\s*(?:=|:|is|at)?\s*[\"'`|]*$", prefix, re.I,
        ):
            return path
    return None


def _requested_bounds_filepath(text: str) -> str | None:
    path = _bounds_filepath_from_text(text)
    if path:
        position = text.find(path)
        prefix = text[max(0, position - 100):position]
        if re.search(r"(?:do not use|don't use|without(?: using)?|ignore|instead of)\s*"
                     r"(?:the\s+)?(?:experimental\s+)?(?:bounds?\s+file\s*)?[\"'`|]*$", prefix, re.I):
            return None
    return path


def _explicit_filepath_from_text(text: str) -> str | None:
    """Return the last explicit data/input filepath in free-form user text."""
    pattern = re.compile(
        # Labels such as "result:output/..." and Markdown/LaTeX wrappers
        # separate the prose from a path; they are not part of its filename.
        # Absolute POSIX paths may themselves contain colons.
        r"(?<![\w./-])((?:/[^\s\"'`|<>]+|[^\s\"'`|<>:=;,(){}\[\]]+)"
        r"\.(?:txt|json|ya?ml|h5|hdf5|csv|npz))(?![\w.-])",
        re.IGNORECASE,
    )
    matrix_file = _matrix_elements_filepath_from_text(text)
    bounds_file = _bounds_filepath_from_text(text)
    matches = [path for path in pattern.findall(str(text or ""))
               if _normalize_path_text(path) not in {matrix_file, bounds_file}]
    return _normalize_path_text(matches[-1]) if matches else None


def _project_path_candidates(path: str | os.PathLike) -> list[Path]:
    """Return stable candidates for user paths, including /output and /input aliases."""
    raw = _normalize_path_text(path)
    candidate = Path(os.path.expanduser(raw))
    candidates: list[Path] = []
    if candidate.is_absolute():
        # Web UIs commonly display project-relative files as /output/... or /input/....
        if candidate.parts and len(candidate.parts) > 1 and candidate.parts[1] in {"output", "input"}:
            candidates.append(PROJECT_ROOT / candidate.relative_to("/"))
        candidates.append(candidate)
    else:
        if candidate.parts and candidate.parts[0] in {"output", "input", "main"}:
            candidates.append(PROJECT_ROOT / candidate)
        candidates.extend([candidate, PROJECT_ROOT / candidate, SOURCE_DIR / candidate, Path.cwd() / candidate])

    bundled = bundled_benchmark_path(raw)
    if bundled is not None:
        candidates.append(bundled)

    # Recover a common copy/paste error in generated MCMC paths, for example
    # ``master56_mcmc_master56_mcmc_<stamp>.h5``.  Keep the literal path first
    # and only use the collapsed spelling when it actually resolves below.
    for item in list(candidates):
        duplicate = re.match(r"^(.+?_mcmc_)\1(.+)$", item.name, flags=re.IGNORECASE)
        if duplicate:
            candidates.append(item.with_name(duplicate.group(1) + duplicate.group(2)))

    unique: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        key = str(item)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _resolve_project_path(path: str | os.PathLike, *, must_exist: bool = False) -> Path:
    candidates = _project_path_candidates(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if must_exist:
        raise FileNotFoundError(
            f"Could not find {_normalize_path_text(path)!r}. Tried: "
            + ", ".join(str(candidate) for candidate in candidates)
        )
    return candidates[0] if candidates else Path(_normalize_path_text(path))


def _with_matrix_elements_file(function):
    """Scope optional physics-input files to one invocation, never a chat global."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        filepath = kwargs.get("matrix_elements_file")
        configuration = (load_matrix_elements_file(_resolve_project_path(filepath, must_exist=True))
                         if filepath else current_matrix_elements())
        bounds_path = kwargs.get("bounds_file")
        limits = (load_bounds_file(_resolve_project_path(bounds_path, must_exist=True))
                  if bounds_path else current_bounds_configuration())
        with using_matrix_elements(configuration), using_bounds(limits):
            result = record_bounds(record_matrix_elements(function(*args, **kwargs)))
            if (configuration or limits) and isinstance(result, dict):
                for key in ("scan_filepath", "grid_data_filepath"):
                    if result.get(key):
                        with pd.HDFStore(result[key], mode="a") as store:
                            if configuration:
                                store.get_storer("results").attrs.matrix_elements = json.dumps(configuration)
                            if limits:
                                store.get_storer("results").attrs.experimental_bounds = json.dumps(limits)
            return result
    return wrapped


def _output_path(path: str | os.PathLike) -> Path:
    candidate = Path(_normalize_path_text(path))
    hidden_parts = [
        part for part in candidate.parts
        if part.startswith(".") and part not in {".", ".."}
    ]
    if hidden_parts:
        raise ValueError(
            "Hidden output paths are not allowed. Rename these path components "
            f"without a leading dot: {', '.join(hidden_parts)}"
        )
    if candidate.is_absolute():
        if len(candidate.parts) > 1 and candidate.parts[1] == "output":
            return PROJECT_ROOT / candidate.relative_to("/")
        return candidate
    if candidate.parts and candidate.parts[0] == "output":
        return PROJECT_ROOT / candidate
    return OUTPUT_DIR / candidate


_RESULT_TIMESTAMP_RE = re.compile(r"_\d{8}T\d{6}(?:\d{6})?Z?$")


def _timestamped_result_path(path: str | os.PathLike) -> Path:
    """Return a collision-safe output path carrying a UTC timestamp.

    A caller may already have supplied a stable timestamp (for example an RGE
    plot tied to its source artifact). Preserve that name while it is unused;
    if it already exists, append a fresh microsecond timestamp instead of
    overwriting it.
    """
    candidate = _output_path(path)
    suffix = candidate.suffix or ".png"
    stem = candidate.stem if candidate.suffix else candidate.name
    if _RESULT_TIMESTAMP_RE.search(stem) and not candidate.exists():
        return candidate if candidate.suffix else candidate.with_suffix(suffix)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return candidate.with_name(f"{stem}_{stamp}{suffix}")

# ── Exact key names returned by evaluate_point in scan_runner.py ─────
KEY_MGUT      = "MGUT (GeV)"
KEY_GGUT      = "gGUT"
KEY_LIFETIMES = "lifetimes (yrs)"
KEY_BR        = "branching_ratios"
KEY_CLASS     = "classification"

CHANNELS = [
    "p->pi0 e+", "p->pi0 mu+",
    "p->K0 e+",  "p->K0 mu+",
    "p->eta e+", "p->eta mu+",
    "p->pi+ nu", "p->K+ nu",
]

CHANNEL_LATEX_LABELS = {
    "p->pi0 e+": r"$p\to\pi^{0}e^{+}$",
    "p->pi0 mu+": r"$p\to\pi^{0}\mu^{+}$",
    "p->K0 e+": r"$p\to K^{0}e^{+}$",
    "p->K0 mu+": r"$p\to K^{0}\mu^{+}$",
    "p->eta e+": r"$p\to\eta e^{+}$",
    "p->eta mu+": r"$p\to\eta\mu^{+}$",
    "p->pi+ nu": r"$p\to\pi^{+}\bar{\nu}$",
    "p->K+ nu": r"$p\to K^{+}\bar{\nu}$",
    "p->pi+ nubar": r"$p\to\pi^{+}\bar{\nu}$",
    "p->K+ nubar": r"$p\to K^{+}\bar{\nu}$",
}


def _channel_latex_label(channel: str) -> str:
    """Return one canonical Matplotlib mathtext label for a decay channel."""
    return CHANNEL_LATEX_LABELS.get(str(channel), str(channel))


def _is_total_lifetime_plot_column(column: str) -> bool:
    """Return True for a total-lifetime observable, which must never be plotted."""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")
    tokens = normalized.split("_")
    return "total" in tokens and "lifetime" in tokens

MATRIX_NAMES = ["UL", "UR", "DL", "DR", "EL", "ER", "NL"]
SO10_MATRIX_INPUT_NAMES = ["UL", "UR", "DL", "DR", "EL", "ER", "NL"]
SU5_MATRIX_INPUT_NAMES = ["U", "D", "E", "Uc", "Dc", "Ec", "N"]
SU5_TO_CORE_MATRIX_NAMES = {
    "U": "UL",
    "D": "DL",
    "E": "EL",
    "N": "NL",
}
FULL_MATRICES_SCHEMA = "full_matrices"
MATRIX_SCAN_PARAMETERS = {
    "theta12", "theta23", "theta13",
    "delta", "phi1", "phi2", "phi3", "phi4", "phi5",
}
MASTER56_SCHEMA = "master56"
SUSY_SU5_SCHEMA = "susy_su5"
SUSY_SO10_SCHEMA = "susy_so10"
NONSUSY_SU5_DEFAULT_INPUT_FILE = "/input/nonsusy-SU5.txt"
NONSUSY_SO10_DEFAULT_INPUT_FILE = "/input/nonsusy-SO10.txt"
SUSY_SU5_DEFAULT_INPUT_FILE = "/input/susy-SU5.txt"
SUSY_SO10_DEFAULT_INPUT_FILE = "/input/susy-SO10.txt"
SUSY_SU5_OPERATOR_CONTEXT = {
    "scenario": "SUSY SU5",
    "operator_dimension": 5,
    "mediator": "color-triplet Higgs/Higgsino",
    "dressing_particles": ["gluino", "chargino", "neutralino"],
    "included_contributions": ["dimension-five dressed operators"],
    "excluded_contributions": ["dimension-six X/Y gauge-boson exchange"],
    "triplet_mass_parameter": "M_T",
    "exact_fixed_dressed_coefficient_scaling": {
        "width": "M_T^-2",
        "lifetime": "M_T^2",
    },
    "available_nonflavor_parameters": ["m0", "AX", "tanbeta", "M_T", "mu", "M_half", "M1", "M2", "M3"],
    "available_phase_parameters": ["phi13", "phi23"],
    "fixed_flavor_parameters": ["fU", "fD", "fE", "P", "phi13", "phi23"],
    "unsupported_point_parameters": ["MGUT", "gGUT"],
}
SUSY_SO10_OPERATOR_CONTEXT = {
    "scenario": "SUSY SO10",
    "operator_dimension": 5,
    "mediator": "color-triplet Higgs/Higgsino",
    "dressing_particles": ["wino", "higgsino"],
    "included_contributions": ["dimension-five dressed operators"],
    "excluded_contributions": ["dimension-six X/Y gauge-boson exchange"],
    "triplet_mass_parameter": "M_T",
    "exact_fixed_dressed_coefficient_scaling": {
        "width": "M_T^-2",
        "lifetime": "M_T^2",
    },
    "available_nonflavor_parameters": ["M_T", "mtildeq", "mtildeW", "mu"],
    "parameter_aliases": {"m_sfermion": "mtildeq", "M_wino": "mtildeW"},
    "fixed_flavor_parameters": [
        "h", "f", "g", "x", "y", "tanbeta",
        "theta_u", "phi_u", "psi_u", "theta_d", "phi_d", "psi_d", "vL", "vR",
    ],
    "unsupported_point_parameters": ["MGUT", "gGUT", "m0"],
}
MASTER56_MGUT = 5.252731496508281e13
MASTER56_GGUT = 0.5576325595577221
MASTER56_CKM_FIXED = {
    "theta12": 0.227045,
    "theta23": 0.0419423,
    "theta13": 0.0037,
    "delta": 1.139,
}
MASTER56_PMNS_FIXED = {
    "theta12": 0.589202,
    "theta23": 0.75538,
    "theta13": 0.150535,
}
MASTER56_MATRIX_PREFIXES = ["UL", "EL", "UR", "DR", "ER"]
MASTER56_PARAMETER_ORDER = (
    [f"{prefix}_{angle}" for prefix in MASTER56_MATRIX_PREFIXES for angle in ("theta12", "theta23", "theta13")]
    + [f"{prefix}_{phase}" for prefix in MASTER56_MATRIX_PREFIXES for phase in ("delta", "phi1", "phi2", "phi3", "phi4", "phi5")]
    + [f"UCKM_phi{index}" for index in range(1, 6)]
    + ["UPMNS_delta"]
    + [f"UPMNS_phi{index}" for index in range(1, 6)]
)
NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_NUMBER_RE = NUMBER_PATTERN

PARAMETER_COLUMN_ALIASES = {
    "MGUT": {
        "mgut", "m_gut", "gutscale", "gut_scale", "unificationscale",
        "unification_scale", "gutmass", "gut_mass", "gutscale",
    },
    "gGUT": {
        "ggut", "g_gut", "gaugecoupling", "gauge_coupling",
        "unifiedcoupling", "unified_coupling", "unifiedgaugecoupling",
        "unified_gauge_coupling", "gutcoupling", "gut_coupling", "coupling",
    },
    "M_T": {
        "mt", "m_t", "m t", "triplet", "tripletmass", "triplet_mass",
        "mtriplet", "m_triplet", "higgstripletmass", "higgs_triplet_mass",
        "colortripletmass", "color_triplet_mass", "colourtripletmass",
        "colour_triplet_mass", "coloredtripletmass", "colored_triplet_mass",
    },
    "M_wino": {
        "m_wino", "mwino", "wino", "wino_mass", "winomass", "wino mass",
    },
    "M_sfermion": {
        "m_sfermion", "msfermion", "sfermion", "sfermion_mass",
        "sfermionmass", "sfermion mass", "sfermion scalar mass",
    },
    "total_width_GeV": {
        "totalwidth", "total_width", "totalwidthgev", "total_width_gev",
        "decaywidth", "decay_width",
    },
}

CHANNEL_ALIASES = {
    "p->pi0 e+": {
        "p->pi0e+", "p->pi0 e+", "p to pi0 e+", "p pi0 e+",
        "pi0e", "pi0 e", "pi zero e", "pi zero positron",
        "pion zero e", "pion zero positron", "neutral pion e",
        "neutral pion positron", "p->π0 e+", "π0 e+",
    },
    "p->pi0 mu+": {
        "p->pi0mu+", "p->pi0 mu+", "p to pi0 mu+", "p pi0 mu+",
        "pi0mu", "pi0 mu", "pi zero mu", "pi zero antimuon",
        "pion zero mu", "pion zero antimuon", "neutral pion mu",
        "neutral pion antimuon", "p->π0 mu+", "π0 mu+",
    },
    "p->K0 e+": {
        "p->k0e+", "p->k0 e+", "p to k0 e+", "p k0 e+",
        "k0e", "k0 e", "k zero e", "k zero positron",
        "kaon zero e", "kaon zero positron", "neutral kaon e",
        "neutral kaon positron",
    },
    "p->K0 mu+": {
        "p->k0mu+", "p->k0 mu+", "p to k0 mu+", "p k0 mu+",
        "k0mu", "k0 mu", "k zero mu", "k zero antimuon",
        "kaon zero mu", "kaon zero antimuon", "neutral kaon mu",
        "neutral kaon antimuon",
    },
    "p->eta e+": {
        "p->etae+", "p->eta e+", "p to eta e+", "p eta e+",
        "etae", "eta e", "eta positron", "eta e+",
    },
    "p->eta mu+": {
        "p->etamu+", "p->eta mu+", "p to eta mu+", "p eta mu+",
        "etamu", "eta mu", "eta antimuon", "eta mu+",
    },
    "p->pi+ nu": {
        "p->pi+nu", "p->pi+ nu", "p to pi+ nu", "p pi+ nu",
        "pi+nu", "pi+ nu", "piplusnu", "piplus nu", "pi plus nu",
        "charged pion neutrino", "charged pion nu", "p->π+ nu", "π+ nu",
    },
    "p->K+ nu": {
        "p->k+nu", "p->k+ nu", "p to k+ nu", "p k+ nu",
        "k+nu", "k+ nu", "kplusnu", "kplus nu", "k plus nu",
        "charged kaon neutrino", "charged kaon nu",
    },
}

PREFERRED_MODELS = [
    "qwen3:8b",
    "qwen3.6:35b-a3b",
    "qwen3.6:27b",
    "qwen3.5:35b",
]
REMOTE_MODEL_ALIASES = {
    "gpt-oss:20b",
    "remote:gpt-oss:20b",
    "openai:gpt-oss:20b",
}
TOOL_CACHE_VERSION = "v4"

CACHE_ROOT = OUTPUT_DIR / "agent_cache"
TOOL_CACHE_DIR = CACHE_ROOT / "tool_results"
ARTIFACT_DIR = CACHE_ROOT / "artifacts"
ARTIFACT_MANIFEST = CACHE_ROOT / "artifacts_manifest.json"
LATEST_PD_RESULT_PATH = CACHE_ROOT / "latest_proton_decay_result.pkl"
SCAN_DATA_DIR = CACHE_ROOT / "scan_data"
SCAN_RESULTS_DIR = OUTPUT_DIR / "scan_results"
MCMC_RESULTS_DIR = OUTPUT_DIR / "mcmc_results"
PARAMETER_POINT_DIR = OUTPUT_DIR / "parameter_points"
for _path in (CACHE_ROOT, TOOL_CACHE_DIR, ARTIFACT_DIR, SCAN_DATA_DIR, SCAN_RESULTS_DIR, MCMC_RESULTS_DIR, PARAMETER_POINT_DIR):
    _path.mkdir(parents=True, exist_ok=True)

ARTIFACT_REGISTRY = {}
LAST_ARTIFACT_ID = None
LAST_PD_RESULT = None
LAST_PD_RESULT_TEXT = None

CHANNEL_DETAILS = {
    "p->pi0 e+": {
        "meson_mass": MASS_PI_ZERO,
        "hadronic_factor": HADRON_PI0_UDU,
        "coefficient_family": "c20/c21",
        "flavor_multiplicity": 1,
    },
    "p->pi0 mu+": {
        "meson_mass": MASS_PI_ZERO,
        "hadronic_factor": HADRON_PI0_UDU,
        "coefficient_family": "c20/c21",
        "flavor_multiplicity": 1,
    },
    "p->K0 e+": {
        "meson_mass": MASS_K_ZERO,
        "hadronic_factor": HADRON_K0_USU,
        "coefficient_family": "c20/c21",
        "flavor_multiplicity": 1,
    },
    "p->K0 mu+": {
        "meson_mass": MASS_K_ZERO,
        "hadronic_factor": HADRON_K0_USU,
        "coefficient_family": "c20/c21",
        "flavor_multiplicity": 1,
    },
    "p->eta e+": {
        "meson_mass": MASS_ETA,
        "hadronic_factor": HADRON_ETA_UDU,
        "coefficient_family": "c20/c21",
        "flavor_multiplicity": 1,
    },
    "p->eta mu+": {
        "meson_mass": MASS_ETA,
        "hadronic_factor": HADRON_ETA_UDU,
        "coefficient_family": "c20/c21",
        "flavor_multiplicity": 1,
    },
    "p->pi+ nu": {
        "meson_mass": MASS_PI_PLUS,
        "hadronic_factor": HADRON_PI_PLUS_UDD,
        "coefficient_family": "c22 (summed over 3 neutrinos)",
        "flavor_multiplicity": 3,
    },
    "p->K+ nu": {
        "meson_mass": MASS_K_PLUS,
        "hadronic_factor": (HADRON_K_PLUS_USD, HADRON_K_PLUS_UDS),
        "coefficient_family": "mixed c22 combination (summed over 3 neutrinos)",
        "flavor_multiplicity": 3,
    },
}

GENERIC_PLOT_KINDS = {
    "plot_lifetime_vs_MGUT",
    "plot_all_channels_vs_MGUT",
    "plot_saved_scan_quantity",
    "plot_saved_columns",
    "plot_saved_3d_columns",
}

CLASSIFICATION_SCATTER_STYLES = {
    "safe": {"color": "black", "marker": "o", "size": 10, "alpha": 0.45},
    "excluded": {"color": "magenta", "marker": "o", "size": 7, "alpha": 0.52},
    "testable": {"color": "cyan", "marker": "*", "size": 42, "alpha": 0.9},
}
CLASSIFICATION_SCATTER_LABELS = {
    "safe": "allowed (not testable)",
    "excluded": "ruled out",
    "testable": "testable",
}
CLASSIFICATION_PLOT_STYLE_VERSION = "classification-black-magenta-cyan-layered-v2"


def _scatter_classification_points(ax, coordinates: tuple, classifications) -> None:
    """Draw safe, excluded, then testable points so testable stars stay visible."""
    classes = pd.Series(classifications).astype(str).to_numpy()
    values = [np.asarray(values) for values in coordinates]
    for zorder, (classification, style) in enumerate(CLASSIFICATION_SCATTER_STYLES.items(), start=1):
        mask = classes == classification
        if np.any(mask):
            ax.scatter(
                *(values_for_axis[mask] for values_for_axis in values),
                s=style["size"],
                alpha=style["alpha"],
                color=style["color"],
                marker=style["marker"],
                label=CLASSIFICATION_SCATTER_LABELS[classification],
                zorder=zorder,
            )


def _add_x_lifetime_limit_guides(ax, x_column: str, divisor: float = 1.0) -> dict:
    """Add vertical current/future guides when the x-axis is a lifetime channel."""
    channel = _channel_from_lifetime_column(x_column)
    if channel is None:
        return {"channel": None, "current": None, "future": None}
    current = get_current_limits().get(channel)
    future = get_future_limits().get(channel)
    if current:
        ax.axvline(
            current / divisor,
            color="crimson",
            lw=1.4,
            ls="--",
            alpha=0.8,
            label=f"x-axis current bound ({_channel_latex_label(channel)})",
            zorder=4,
        )
    if future:
        ax.axvline(
            future / divisor,
            color="darkorange",
            lw=1.4,
            ls=":",
            alpha=0.8,
            label=f"x-axis future sensitivity ({_channel_latex_label(channel)})",
            zorder=4,
        )
    return {"channel": channel, "current": current, "future": future}


def _channel_column(channel: str, prefix: str) -> str:
    return f"{prefix}_{channel.replace('->', '_to_').replace('+', 'plus').replace(' ', '_')}"

# ================================================================
# SYSTEM PROMPT
# ================================================================

with resource_path("system_prompt.txt").open("r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()
with resource_path("prompts/computation_report.txt").open("r", encoding="utf-8") as f:
    COMPUTATION_REPORT_PROMPT = f.read()
with resource_path("prompts/bound_evasion_discussion.txt").open("r", encoding="utf-8") as f:
    BOUND_EVASION_DISCUSSION_PROMPT = f.read()
with resource_path("prompts/scan_workflow.txt").open("r", encoding="utf-8") as f:
    SCAN_WORKFLOW_PROMPT = f.read()
with resource_path("prompts/mcmc_workflow.txt").open("r", encoding="utf-8") as f:
    MCMC_WORKFLOW_PROMPT = f.read()
with resource_path("prompts/rge_agent.txt").open("r", encoding="utf-8") as f:
    RGE_SYSTEM_PROMPT = f.read()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def _shorten(text: str, limit: int = 240) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _persist_artifact_registry():
    # The output cache may be cleaned while the long-running server remains
    # alive.  Recreate its parent at write time instead of relying only on the
    # directories created during module import.
    ARTIFACT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    records = list(ARTIFACT_REGISTRY.values())
    with ARTIFACT_MANIFEST.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "last_artifact_id": LAST_ARTIFACT_ID,
                "records": records,
            },
            fh,
            indent=2,
        )


def _load_artifact_registry():
    global LAST_ARTIFACT_ID

    if not ARTIFACT_MANIFEST.exists():
        return

    try:
        with ARTIFACT_MANIFEST.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return

    ARTIFACT_REGISTRY.clear()
    for record in payload.get("records", []):
        ARTIFACT_REGISTRY[record["artifact_id"]] = record
    LAST_ARTIFACT_ID = payload.get("last_artifact_id")


def _cache_key(tool_name: str, args: dict) -> str:
    payload = {"version": TOOL_CACHE_VERSION, "tool": tool_name, "args": _jsonable(args)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tool_cache_path(tool_name: str, args: dict) -> Path:
    return TOOL_CACHE_DIR / f"{tool_name}_{_cache_key(tool_name, args)}.pkl"


def _scan_filepath(kind: str, summary: dict) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    digest = _cache_key(kind, summary)[:8]
    return str(SCAN_DATA_DIR / f"{kind}_{stamp}_{digest}.h5")


def _tool_cache_args(tool_name: str, args: dict) -> dict | None:
    if tool_name in {
        "get_latest_proton_decay_result",
        "compute_nonsusy_su5_point",
        "compute_nonsusy_so10_point",
        "compute_nonsusy_su5_from_file",
        "compute_nonsusy_so10_from_file",
        "compute_susy_su5_point",
        "compute_susy_so10_point",
        "compute_susy_su5_from_file",
        "compute_susy_so10_from_file",
        "compute_susy_su5_from_file_with_overrides",
        "compute_susy_so10_from_file_with_overrides",
        "find_susy_so10_minimum_sfermion_mass_from_file",
        "list_session_artifacts",
        "describe_last_artifact",
        "list_saved_scan_columns",
        "run_random_scan",
        "run_nonsusy_proton_decay_mcmc",
        "run_susy_proton_decay_mcmc",
        "run_mcmc_around_last_point",
        "run_fixed_flavor_scan_from_file",
        "run_susy_su5_fixed_flavor_scan_from_file",
        "run_susy_so10_fixed_flavor_scan_from_file",
        "run_grid_scan",
        "plot_lifetime_vs_MGUT",
        "plot_all_channels_vs_MGUT",
        "plot_saved_scan_quantity",
        "plot_saved_columns",
        "plot_saved_3d_columns",
        "analyze_saved_columns_relationship",
        "plot_from_hdf5",
    }:
        return None

    cache_args = dict(args)
    if cache_args.get("matrix_elements_file"):
        path = _resolve_project_path(cache_args["matrix_elements_file"], must_exist=True)
        cache_args["_matrix_elements_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if cache_args.get("bounds_file"):
        path = _resolve_project_path(cache_args["bounds_file"], must_exist=True)
        cache_args["_bounds_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if tool_name in {
        "run_sm_gauge_rges", "run_nonsusy_gauge_rges", "run_susy_gauge_rges",
        "analyze_sm_unification_effect", "plot_rge_running",
    }:
        # Older cached calls may omit loops while having run at the former
        # two-loop default, or lack the diagnostic/plot loop metadata.
        cache_args["_rge_contract_version"] = 3 if tool_name == "analyze_sm_unification_effect" else 4
        if tool_name in {"run_sm_gauge_rges", "run_nonsusy_gauge_rges", "run_susy_gauge_rges"}:
            cache_args.setdefault("loops", 1)
        if tool_name != "plot_rge_running":
            for key, value in DEFAULT_MZ_COUPLINGS.items():
                cache_args.setdefault(key, value)
    if tool_name == "run_sm_gauge_rges":
        # Default boundary values must distinguish runs after a physics-input update.
        for key, value in DEFAULT_MZ_YUKAWAS.items():
            cache_args.setdefault(key, value)
    if tool_name in GENERIC_PLOT_KINDS:
        scan_artifact_id = cache_args.get("scan_artifact_id", "latest_scan")
        if scan_artifact_id in {"latest_scan", "last_scan", "", None}:
            latest_scan = _find_latest_artifact_by_kind({"random-scan", "fixed-flavor-scan", "susy-su5-fixed-flavor-scan", "susy-so10-fixed-flavor-scan", "grid-scan", "susy-su5-scan", "susy-so10-scan"})
            cache_args["scan_artifact_id"] = latest_scan["artifact_id"] if latest_scan else "__no_scan__"
    if tool_name == "plot_rge_running":
        cache_args.setdefault("quantity", "coupling")
        rge_artifact_id = cache_args.get("rge_artifact_id", "latest_rge")
        if rge_artifact_id in {"latest_rge", "latest", "last", "", None}:
            latest_rge = _find_latest_artifact_by_kind(RGE_RUN_ARTIFACT_KINDS)
            cache_args["rge_artifact_id"] = latest_rge["artifact_id"] if latest_rge else "__no_rge__"
    return cache_args


def _load_cached_tool_result(tool_name: str, args: dict):
    cache_args = _tool_cache_args(tool_name, args)
    if cache_args is None:
        return None
    cache_path = _tool_cache_path(tool_name, cache_args)
    if not cache_path.exists():
        return None
    with cache_path.open("rb") as fh:
        return pickle.load(fh)


def _save_cached_tool_result(tool_name: str, args: dict, result):
    cache_args = _tool_cache_args(tool_name, args)
    if cache_args is None:
        return
    cache_path = _tool_cache_path(tool_name, cache_args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(result, fh)


def _find_identical_artifact(kind: str, payload: dict):
    """Return a prior content-identical artifact created by idempotent workflows."""
    content_key = _cache_key(f"{kind}:artifact-payload", payload)
    for record in reversed(list(ARTIFACT_REGISTRY.values())):
        if record.get("kind") != kind or record.get("content_key") != content_key:
            continue
        if Path(record["path"]).exists():
            return record
    return None


def _register_artifact(
    kind: str,
    payload: dict,
    summary: dict,
    *,
    reuse_identical: bool = False,
) -> dict:
    global LAST_ARTIFACT_ID

    record_matrix_elements(payload)
    record_matrix_elements(summary)
    record_bounds(payload)
    record_bounds(summary)

    content_key = None
    if reuse_identical:
        existing = _find_identical_artifact(kind, payload)
        if existing is not None:
            LAST_ARTIFACT_ID = existing["artifact_id"]
            _persist_artifact_registry()
            return {**existing, "reused": True}
        content_key = _cache_key(f"{kind}:artifact-payload", payload)

    artifact_id = f"{kind}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{_cache_key(kind, summary)[:8]}"
    artifact_path = ARTIFACT_DIR / f"{artifact_id}.pkl"

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as fh:
        pickle.dump(payload, fh)

    record = {
        "artifact_id": artifact_id,
        "kind": kind,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(artifact_path),
        "summary": _jsonable(summary),
    }
    if content_key is not None:
        record["content_key"] = content_key
    ARTIFACT_REGISTRY[artifact_id] = record
    LAST_ARTIFACT_ID = artifact_id
    _persist_artifact_registry()
    return record


def _get_artifact_record(artifact_id: str | None = None):
    global LAST_ARTIFACT_ID

    if artifact_id in (None, "", "last"):
        artifact_id = LAST_ARTIFACT_ID
    if artifact_id is None:
        return None
    return ARTIFACT_REGISTRY.get(artifact_id)


def _find_latest_artifact_by_kind(kinds: set[str]):
    for record in reversed(list(ARTIFACT_REGISTRY.values())):
        if record["kind"] in kinds:
            return record
    return None


def _load_artifact_payload(artifact_id: str | None = None):
    record = _get_artifact_record(artifact_id)
    if record is None:
        return None, None

    path = Path(record["path"])
    if not path.exists():
        return record, None

    with path.open("rb") as fh:
        return record, pickle.load(fh)


SCAN_ARTIFACT_KINDS = {"random-scan", "fixed-flavor-scan", "susy-su5-fixed-flavor-scan", "susy-so10-fixed-flavor-scan", "grid-scan", "susy-su5-scan", "susy-so10-scan", "mcmc-scan"}


def _load_latest_scan_payload():
    record = _find_latest_artifact_by_kind(SCAN_ARTIFACT_KINDS)
    if record is None:
        return None, None
    return _load_artifact_payload(record["artifact_id"])


def _touch_artifact_from_result(result: dict):
    global LAST_ARTIFACT_ID

    artifact_id = result.get("artifact_id")
    if artifact_id and artifact_id in ARTIFACT_REGISTRY:
        LAST_ARTIFACT_ID = artifact_id
        _persist_artifact_registry()


def _phase_space_factor(meson_mass: float, gGUT: float, MGUT: float) -> float:
    return (
        (PROTON_MASS**2 - meson_mass**2)**2
        * gGUT**4
        * ALPHA_LONG**2
        * ALPHA_SHORT**2
        / (32 * np.pi * PROTON_MASS**3 * MGUT**4)
    )


def _message_summary(msg) -> str | None:
    if isinstance(msg, HumanMessage):
        return f"User: {_shorten(msg.content)}"
    if isinstance(msg, ToolMessage):
        try:
            data = json.loads(msg.content)
        except Exception:
            return "Tool returned a result."
        tool_name = data.get("tool_name")
        artifact_id = data.get("artifact_id")
        if tool_name and artifact_id:
            return f"Tool {tool_name} produced artifact {artifact_id}."
        if tool_name:
            return f"Tool {tool_name} returned cached/computed data."
        return "Tool returned a result."
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        names = ", ".join(call.get("name", "unknown") for call in msg.tool_calls)
        return f"Assistant requested tools: {names}."
    content = getattr(msg, "content", None)
    if content:
        return f"Assistant: {_shorten(content)}"
    return None


def _prepare_messages(conversation_history: list) -> list:
    if len(conversation_history) <= 12:
        return conversation_history

    older = conversation_history[:-12]
    recent = conversation_history[-12:]

    summary_lines = []
    for msg in older:
        line = _message_summary(msg)
        if line:
            summary_lines.append(line)

    if not summary_lines:
        return recent

    compressed = "\n".join(summary_lines[-12:])
    return [SystemMessage(content=f"Conversation summary:\n{compressed}")] + recent


_load_artifact_registry()


def _scan_dataframe_from_artifact(artifact_id: str | None = None):
    if str(artifact_id).startswith("filepath::"):
        return _scan_dataframe_from_filepath(str(artifact_id).removeprefix("filepath::"))
    if artifact_id in (None, "", "latest_scan", "last_scan"):
        record, payload = _load_latest_scan_payload()
    else:
        record, payload = _load_artifact_payload(artifact_id)

    if record is None or payload is None:
        return None, None

    if record["kind"] not in SCAN_ARTIFACT_KINDS:
        return record, None

    scan_filepath = payload.get("scan_filepath")
    if scan_filepath and Path(scan_filepath).exists():
        _, frame = _scan_dataframe_from_filepath(scan_filepath)
        return record, frame

    results = payload.get("results")
    if results is None:
        return record, None
    return record, _compare_saved_bounds(results_to_dataframe(results), bool(payload.get("experimental_bounds")))


def _scan_payload(artifact_id: str | None = None):
    if artifact_id in (None, "", "latest_scan", "last_scan"):
        return _load_latest_scan_payload()
    return _load_artifact_payload(artifact_id)


def _compare_saved_bounds(df: pd.DataFrame, previously_custom: bool = False) -> pd.DataFrame:
    """Recompare saved predictions in memory; never resample or overwrite data."""
    if current_bounds_configuration() is None and not previously_custom:
        return df
    columns = {_channel_column(channel, "lifetime"): channel for channel in get_current_limits()
               if _channel_column(channel, "lifetime") in df}
    if not columns:
        return df
    result = df.copy()
    lifetimes = [{channel: float(row[column]) for column, channel in columns.items()
                  if not pd.isna(row[column])} for _, row in result.iterrows()]
    result["classification"] = [classify_model(values) if values else "unknown" for values in lifetimes]
    if "chi_sq" in result:
        result["chi_sq"] = [_master56_chi_sq(values) for values in lifetimes]
    if "mcmc_allowed" in result:
        result["mcmc_allowed"] = [_master56_chi_sq(values) < 1.0 for values in lifetimes]
    for column, channel in columns.items():
        score_column = _channel_column(channel, "chi_sq")
        if score_column in result:
            result[score_column] = get_current_limits()[channel] / result[column].clip(lower=1e-300)
    ratios = [{channel: tau / get_current_limits()[channel] for channel, tau in values.items()}
              for values in lifetimes]
    if "most_constraining_bound_ratio" in result:
        result["most_constraining_bound_ratio"] = [min(values.values(), default=np.nan) for values in ratios]
    if "most_constraining_channel" in result:
        result["most_constraining_channel"] = [min(values, key=values.get) if values else "unknown" for values in ratios]
    return result


def _scan_dataframe_from_filepath(filepath: str):
    try:
        path = _resolve_project_path(filepath, must_exist=True)
    except FileNotFoundError:
        return None, None
    try:
        with pd.HDFStore(path, mode="r") as store:
            df = store["results"]
            previously_custom = getattr(store.get_storer("results").attrs, "experimental_bounds", None) is not None
        df = _compare_saved_bounds(df, previously_custom)
    except Exception:
        return None, None

    record = {
        "artifact_id": f"filepath::{path}",
        "kind": "scan-file",
        "path": str(path),
    }
    return record, df


def _saved_scan_columns(scan_artifact_id: str = "latest_scan") -> tuple[dict | None, pd.DataFrame | None, list[str]]:
    record, df = _scan_dataframe_from_artifact(scan_artifact_id)
    if df is None:
        return record, None, []

    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]
    return record, df, numeric_cols


def _parse_numeric_token(token: str) -> float | None:
    token = token.strip().lower().replace(",", "").replace("×", "x").replace("**", "^")
    token = token.replace(" ", "").replace("gev", "")
    if token.startswith("10^"):
        try:
            return 10 ** float(token[3:])
        except ValueError:
            return None
    if "x10^" in token:
        base, exp = token.split("x10^", 1)
        try:
            return float(base) * (10 ** float(exp))
        except ValueError:
            return None
    try:
        return float(token)
    except ValueError:
        return None


def _normalize_alias_text(text: str) -> str:
    text = text.lower().replace("→", "->")
    text = text.replace("π", "pi")
    text = re.sub(r"\banti[- ]?muon\b", "mu", text)
    text = re.sub(r"\bpositron\b", "e", text)
    text = re.sub(r"\bneutrino\b", "nu", text)
    text = re.sub(r"\bpion\b", "pi", text)
    text = re.sub(r"\bkaon\b", "k", text)
    text = re.sub(r"\bzero\b", "0", text)
    text = re.sub(r"\bplus\b", "+", text)
    text = re.sub(r"\bto\b", "->", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _alias_forms(text: str) -> set[str]:
    normalized = _normalize_alias_text(text)
    compact = re.sub(r"[^a-z0-9+>\-]+", "", normalized)
    spaced = re.sub(r"[^a-z0-9+>\-]+", " ", normalized).strip()
    return {normalized, compact, spaced}


def _extract_channel(question: str) -> str | None:
    forms = _alias_forms(question)
    for channel, aliases in CHANNEL_ALIASES.items():
        candidates = set()
        for alias in aliases | {channel}:
            candidates.update(_alias_forms(alias))
        if forms & candidates:
            return channel
        if any(candidate and candidate in form for form in forms for candidate in candidates):
            return channel
    return None


def _extract_x_column(question: str) -> str | None:
    lowered = question.lower()
    if "unification scale" in lowered or "gut scale" in lowered or "mgut" in lowered:
        return "MGUT"
    if "unified coupling" in lowered or "ggut" in lowered:
        return "gGUT"

    match = re.search(r"\b(ul|ur|dl|dr|el|er|nl)\s*\[\s*(\d)\s*,\s*(\d)\s*\](?:\s*(real|imag|abs))?", lowered)
    if not match:
        match = re.search(r"\b(ul|ur|dl|dr|el|er|nl)_(\d)(\d)_(real|imag|abs)\b", lowered)
    if match:
        matrix = match.group(1).upper()
        i = match.group(2)
        j = match.group(3)
        suffix = match.group(4) or "abs"
        return f"{matrix}_{i}{j}_{suffix}"
    return None



def _clean_plot_axis_text(text: str) -> str:
    text = " ".join(text.lower().split())
    text = re.sub(r"^(?:make|create|draw|show|plot|a|an|the|of|for|me|please)\b\s*", "", text)
    text = re.sub(r"\b(?:plot|graph|figure|scatter|scatterplot)\b", " ", text)
    text = re.sub(r"\b(?:on|the|axis|x-axis|y-axis|x axis|y axis)\b", " ", text)
    text = re.sub(r"\b(?:linear|log|logarithmic|logarithm|scaled)\b", " ", text)
    text = re.sub(r"\bof\b", " ", text)
    return " ".join(text.split()).strip()


def _split_generic_plot_axes(question: str) -> tuple[str | None, str | None]:
    lowered = " ".join(question.lower().split())
    lowered = re.sub(r"^(?:make|create|draw|show)\s+(?:me\s+)?(?:a\s+)?(?:plot|graph|figure)\s+(?:of\s+)?", "", lowered)
    function_match = re.search(r"\s+as\s+(?:a\s+)?function\s+of\s+", lowered)
    if function_match:
        y_axis = lowered[:function_match.start()].strip()
        x_axis = lowered[function_match.end():].strip()
        return y_axis or None, x_axis or None
    match = re.search(r"\s+(?:versus|verses|vs\.?|against)\s+", lowered)
    if not match:
        return None, None
    left = lowered[:match.start()].strip()
    right = lowered[match.end():].strip()
    return left or None, right or None


def _resolve_plot_column(axis_text: str | None, default: str | None = None) -> str | None:
    if not axis_text:
        return default

    text = _clean_plot_axis_text(axis_text)
    compact = text.replace(" ", "").replace("-", "_")

    alias_key = compact.replace("_", "")
    for column, aliases in PARAMETER_COLUMN_ALIASES.items():
        normalized_aliases = {alias.replace("_", "") for alias in aliases}
        if compact in aliases or alias_key in normalized_aliases:
            return column

    matrix_param_match = re.fullmatch(
        r"(?:(ul|ur|dl|dr|el|er|nl)_?)?(theta12|theta23|theta13|delta|phi1|phi2|phi3|phi4|phi5)",
        compact,
    )
    if matrix_param_match:
        matrix = matrix_param_match.group(1)
        param = matrix_param_match.group(2)
        return f"{matrix.upper()}_{param}" if matrix else param

    channel = _extract_channel(text)
    if channel:
        if "branch" in text or "br" in text:
            return _channel_column(channel, "branching")
        if "width" in text:
            return _channel_column(channel, "width")
        if "lifetime" in text or "tau" in text or "partial lifetime" in text:
            return _channel_column(channel, "lifetime")

    match = re.search(r"\b(lifetime|tau|branching|branching ratio|width)\s+(?:of\s+)?(.+)", text)
    if match:
        quantity = match.group(1)
        channel = _extract_channel(match.group(2))
        if channel:
            if "branch" in quantity:
                return _channel_column(channel, "branching")
            if "width" in quantity:
                return _channel_column(channel, "width")
            return _channel_column(channel, "lifetime")

    match = re.search(r"\b(ul|ur|dl|dr|el|er|nl)\s*\[\s*(\d)\s*,\s*(\d)\s*\](?:\s*(real|imag|abs))?", text)
    if not match:
        match = re.search(r"\b(ul|ur|dl|dr|el|er|nl)_(\d)(\d)_(real|imag|abs)\b", text)
    if match:
        matrix = match.group(1).upper()
        i = match.group(2)
        j = match.group(3)
        suffix = match.group(4) or "abs"
        return f"{matrix}_{i}{j}_{suffix}"

    if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", compact):
        return compact

    return default


def _generic_plot_filename(x_column: str, y_column: str, xscale: str, yscale: str) -> str:
    safe_x = re.sub(r"[^A-Za-z0-9_]+", "_", x_column).strip("_")
    safe_y = re.sub(r"[^A-Za-z0-9_]+", "_", y_column).strip("_")
    prefix = f"{yscale}y_" if yscale == "log" else ""
    prefix += f"{xscale}x_" if xscale == "log" else ""
    return f"{prefix}{safe_y}_vs_{safe_x}.png"



def _plot_axis_label(column: str) -> str:
    if column == "MGUT":
        return r"$M_{\rm GUT}$ [GeV]"
    if column == "gGUT":
        return r"$g_{\rm GUT}$"
    if column == "total_width_GeV":
        return r"Total width $\Gamma$ [GeV]"
    if column == "m0":
        return r"$m_0$ [GeV]"
    if column == "M_T":
        return r"Color-triplet mass $M_T$ [GeV]"
    if column == "M_wino":
        return r"Wino mass $M_{\widetilde W}$ [GeV]"
    if column in {"mtildeq", "M_sfermion", "m_sfermion"}:
        return r"Sfermion scale [GeV]"
    if column == "susy_scale_GeV":
        return r"SUSY scale [GeV]"

    for channel in CHANNELS:
        if column == _channel_column(channel, "lifetime"):
            return rf"$\tau/B$({channel}) [yr]"
        if column == _channel_column(channel, "width"):
            return rf"$\Gamma$({channel}) [GeV]"
        if column == _channel_column(channel, "branching"):
            return rf"BR({channel})"

    match = re.fullmatch(r"(UL|UR|DL|DR|EL|ER|NL)_(\d)(\d)_(real|imag|abs)", column)
    if match:
        matrix, i, j, component = match.groups()
        component_label = {"real": "Re", "imag": "Im", "abs": "Abs"}[component]
        return f"{component_label}({matrix}[{i},{j}])"

    derived_labels = {
        "derived_min_lifetime_years": r"Minimum $\tau/B$ [yr]",
        "derived_max_lifetime_years": r"Maximum $\tau/B$ [yr]",
        "derived_min_lifetime_1e34yr": r"Minimum $\tau/B$ [$10^{34}$ yr]",
        "derived_max_branching_ratio": "Dominant branching ratio",
        "derived_min_branching_ratio": "Smallest branching ratio",
        "derived_max_width_GeV": r"Maximum $\Gamma$ [GeV]",
        "derived_min_width_GeV": r"Minimum $\Gamma$ [GeV]",
        "derived_most_constraining_bound_ratio": r"Minimum $\tau/\tau_{\rm bound}$",
    }
    if column in derived_labels:
        return derived_labels[column]
    if column.startswith("derived_log10_"):
        return r"$\log_{10}$ " + _plot_axis_label(column.removeprefix("derived_log10_"))
    if column.startswith("derived_bound_ratio_"):
        return r"$\tau/\tau_{\rm bound}$"
    if column.startswith("derived_future_ratio_"):
        return r"$\tau/\tau_{\rm future}$"
    if column.startswith("derived_lifetime_") and column.endswith("_1e34yr"):
        return r"$\tau/B$ [$10^{34}$ yr]"

    return column


def _configure_plot_axis_ticks(ax, axis_name: str, scale: str, column: str) -> None:
    """Use readable powers on log axes and a shared 1eX factor on mass axes."""
    from matplotlib.ticker import (
        FixedLocator, LogFormatterMathtext, LogLocator, NullFormatter, ScalarFormatter,
    )

    axis = ax.xaxis if axis_name == "x" else ax.yaxis
    if scale == "log":
        lower, upper = sorted(float(value) for value in axis.get_view_interval())
        if lower > 0.0 and upper > lower and np.log10(upper) - np.log10(lower) < 1.0:
            # A decade-only locator can leave a narrow logarithmic mass range
            # with no labeled tick at all. Geometric ticks plus ScalarFormatter
            # preserve the log geometry while displaying one shared ×10^n.
            ticks = np.geomspace(lower, upper, 5)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            formatter.set_useOffset(False)
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_major_formatter(formatter)
            axis.set_minor_formatter(NullFormatter())
            return
        axis.set_major_locator(LogLocator(base=10.0, numticks=7))
        axis.set_major_formatter(LogFormatterMathtext(base=10.0))
        axis.set_minor_formatter(NullFormatter())
        return
    if column in {
        "MGUT", "M_T", "m0", "mtildeq", "M_sfermion", "m_sfermion", "M_wino"
    }:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        formatter.set_useOffset(True)
        axis.set_major_formatter(formatter)


def _configure_colorbar_ticks(colorbar, column: str) -> None:
    """Use one shared scientific multiplier for large mass color values."""
    if column not in {"MGUT", "M_T", "m0", "mtildeq", "M_sfermion", "m_sfermion", "M_wino"}:
        return
    from matplotlib.ticker import ScalarFormatter

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    formatter.set_useOffset(False)
    colorbar.formatter = formatter
    colorbar.update_ticks()




def _scan_columns_by_prefix(df: pd.DataFrame, prefix: str) -> list[str]:
    return [
        _channel_column(channel, prefix)
        for channel in CHANNELS
        if _channel_column(channel, prefix) in df.columns
    ]


def _materialize_derived_observable(requested: str, df: pd.DataFrame) -> tuple[str | None, str | None]:
    raw = " ".join(str(requested).lower().split())
    cleaned = _clean_plot_axis_text(requested)
    compact = re.sub(r"[^a-z0-9]+", "", cleaned)

    lifetime_cols = _scan_columns_by_prefix(df, "lifetime")
    width_cols = _scan_columns_by_prefix(df, "width")
    branching_cols = _scan_columns_by_prefix(df, "branching")

    def require_columns(cols: list[str], family: str) -> str | None:
        if cols:
            return None
        return f"Cannot derive '{requested}' because this saved scan has no {family} columns."

    if "log10" in raw:
        base_text = re.sub(r"\blog10\s*(?:of\s*)?", "", raw).strip()
        base_col = _resolve_plot_column(base_text, default=base_text)
        if base_col not in df.columns:
            return None, f"Cannot derive log10 observable; base column '{base_col}' is not available."
        if np.any(df[base_col].to_numpy() <= 0):
            return None, f"Cannot derive log10({base_col}) because it contains non-positive values."
        derived_col = f"derived_log10_{base_col}"
        df[derived_col] = np.log10(df[base_col].to_numpy())
        return derived_col, None

    if any(key in compact for key in ["minimumlifetime", "minlifetime", "shortestlifetime"]):
        error = require_columns(lifetime_cols, "lifetime")
        if error:
            return None, error
        col = "derived_min_lifetime_years"
        df[col] = df[lifetime_cols].min(axis=1)
        return col, None

    if any(key in compact for key in ["maximumlifetime", "maxlifetime", "longestlifetime"]):
        error = require_columns(lifetime_cols, "lifetime")
        if error:
            return None, error
        col = "derived_max_lifetime_years"
        df[col] = df[lifetime_cols].max(axis=1)
        return col, None

    if any(key in compact for key in ["dominantbranching", "largestbranching", "maxbranching", "dominantbranchingratio"]):
        error = require_columns(branching_cols, "branching-ratio")
        if error:
            return None, error
        col = "derived_max_branching_ratio"
        df[col] = df[branching_cols].max(axis=1)
        return col, None

    if any(key in compact for key in ["smallestbranching", "minimumbranching", "minbranching"]):
        error = require_columns(branching_cols, "branching-ratio")
        if error:
            return None, error
        col = "derived_min_branching_ratio"
        df[col] = df[branching_cols].min(axis=1)
        return col, None

    if any(key in compact for key in ["maximumwidth", "maxwidth", "largestwidth", "dominantwidth"]):
        error = require_columns(width_cols, "width")
        if error:
            return None, error
        col = "derived_max_width_GeV"
        df[col] = df[width_cols].max(axis=1)
        return col, None

    if any(key in compact for key in ["minimumwidth", "minwidth", "smallestwidth"]):
        error = require_columns(width_cols, "width")
        if error:
            return None, error
        col = "derived_min_width_GeV"
        df[col] = df[width_cols].min(axis=1)
        return col, None

    if "1e34" in compact or "1034" in compact:
        channel = _extract_channel(raw)
        if channel:
            base_col = _channel_column(channel, "lifetime")
            if base_col not in df.columns:
                return None, f"Cannot derive lifetime/1e34 for {channel}; column '{base_col}' is missing."
            col = f"derived_{base_col}_1e34yr"
            df[col] = df[base_col] / 1e34
            return col, None
        if "lifetime" in compact:
            error = require_columns(lifetime_cols, "lifetime")
            if error:
                return None, error
            col = "derived_min_lifetime_1e34yr"
            df[col] = df[lifetime_cols].min(axis=1) / 1e34
            return col, None

    if "bound" in raw or "constraint" in raw:
        channel = _extract_channel(raw)
        if channel:
            base_col = _channel_column(channel, "lifetime")
            bound = get_current_limits().get(channel)
            if base_col not in df.columns or bound is None:
                return None, f"Cannot derive bound ratio for {channel}; required lifetime column or bound is missing."
            col = f"derived_bound_ratio_{_channel_column(channel, 'channel')}"
            df[col] = df[base_col] / bound
            return col, None

        ratio_cols = []
        for channel_name in CHANNELS:
            base_col = _channel_column(channel_name, "lifetime")
            bound = get_current_limits().get(channel_name)
            if base_col in df.columns and bound is not None:
                ratio_col = f"derived_bound_ratio_{_channel_column(channel_name, 'channel')}"
                df[ratio_col] = df[base_col] / bound
                ratio_cols.append(ratio_col)
        if ratio_cols:
            col = "derived_most_constraining_bound_ratio"
            df[col] = df[ratio_cols].min(axis=1)
            return col, None
        return None, f"Cannot derive '{requested}' because no lifetime/bound pairs are available."

    if "future" in raw:
        channel = _extract_channel(raw)
        if channel:
            base_col = _channel_column(channel, "lifetime")
            future = get_future_limits().get(channel)
            if base_col not in df.columns or future is None:
                return None, f"Cannot derive future-sensitivity ratio for {channel}; required lifetime column or future sensitivity is missing."
            col = f"derived_future_ratio_{_channel_column(channel, 'channel')}"
            df[col] = df[base_col] / future
            return col, None

    return None, None

def _resolve_contextual_saved_column_alias(requested: str, numeric_cols: list[str]) -> str | None:
    cleaned = _clean_plot_axis_text(str(requested))
    compact = re.sub(r"[^a-z0-9]+", "", cleaned)

    channel = _extract_channel(cleaned)
    if channel:
        if "branch" in cleaned:
            candidate = _channel_column(channel, "branching")
        elif "width" in cleaned:
            candidate = _channel_column(channel, "width")
        elif "lifetime" in cleaned or "tau" in cleaned:
            candidate = _channel_column(channel, "lifetime")
        else:
            # A bare decay-channel label on a plot axis conventionally means
            # its lifetime. Width and branching-ratio requests remain explicit.
            candidate = _channel_column(channel, "lifetime")
        if candidate in numeric_cols:
            return candidate

    matrix_match = re.search(
        r"\b(ul|ur|dl|dr|el|er|nl)\s*(?:\[\s*)?(\d)\s*[, _-]?\s*(\d)(?:\s*\])?"
        r"(?:\s*(?:-|_)?\s*(real|imag|abs|absolute(?:\s+value)?))?",
        cleaned,
    )
    if not matrix_match:
        matrix_match = re.search(
            r"\b(?:absolute(?:\s+value)?|abs|real|imag)?\s*(?:of\s+)?(?:the\s+)?"
            r"(\d)(\d)(?:st|nd|rd|th)?(?:\s*[- ]?entry)?(?:\s+of)?(?:\s+(?:mixing\s+)?matrix)?\s+"
            r"(ul|ur|dl|dr|el|er|nl)\b",
            cleaned,
        )
        if matrix_match:
            i, j, matrix = matrix_match.group(1), matrix_match.group(2), matrix_match.group(3)
            component = "abs" if "absolute" in cleaned or "abs" in cleaned else "abs"
            candidate = f"{matrix.upper()}_{i}{j}_{component}"
            if candidate in numeric_cols:
                return candidate
    else:
        matrix, i, j, component = matrix_match.groups()
        suffix = "abs" if not component or component.startswith("absolute") else component
        candidate = f"{matrix.upper()}_{i}{j}_{suffix}"
        if candidate in numeric_cols:
            return candidate

    if compact in {
        "susyscale",
        "supersymmetryscale",
        "susymassscale",
        "mssmscale",
        "superpartnerscale",
        "superpartnermassscale",
    }:
        for candidate in ["susy_scale_GeV", "m0", "mtildeq", "M_sfermion", "M_SUSY"]:
            for column in numeric_cols:
                if str(column).casefold() == candidate.casefold():
                    return column
    return None


def _resolve_saved_scan_column(requested: str, df: pd.DataFrame, numeric_cols: list[str] | None = None) -> tuple[str | None, str | None]:
    if numeric_cols is None:
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    if requested in numeric_cols:
        return requested, None

    contextual = _resolve_contextual_saved_column_alias(requested, numeric_cols)
    if contextual is not None:
        return contextual, None

    # Scan columns preserve model-facing capitalization, while natural-language
    # requests are normalized to lowercase before resolution.
    requested_folded = str(requested).casefold()
    for column in numeric_cols:
        if str(column).casefold() == requested_folded:
            return column, None

    resolved = _resolve_plot_column(requested, default=requested)
    if resolved in numeric_cols:
        return resolved, None
    resolved_folded = str(resolved).casefold() if resolved is not None else None
    if resolved_folded is not None:
        for column in numeric_cols:
            if str(column).casefold() == resolved_folded:
                return column, None

    derived_col, derived_error = _materialize_derived_observable(requested, df)
    if derived_error:
        return None, derived_error
    if derived_col is not None:
        return derived_col, None

    if resolved in MATRIX_SCAN_PARAMETERS:
        matches = [f"{matrix}_{resolved}" for matrix in MATRIX_NAMES if f"{matrix}_{resolved}" in numeric_cols]
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, (
                f"'{resolved}' is matrix-specific in this scan. Specify one of: "
                + ", ".join(matches)
            )
        return None, (
            f"'{resolved}' is a matrix-generation parameter, but this saved scan does not contain "
            f"{', '.join(f'{matrix}_{resolved}' for matrix in MATRIX_NAMES)}. "
            "Run a fresh scan with the updated scan schema, then specify the matrix parameter to plot."
        )

    matrix_param_match = re.fullmatch(
        r"(UL|UR|DL|DR|EL|ER|NL)_(theta12|theta23|theta13|delta|phi1|phi2|phi3|phi4|phi5)",
        resolved,
    )
    if matrix_param_match:
        return None, (
            f"'{resolved}' is not present in this saved scan. Run a fresh scan with the updated scan schema first."
        )

    close = difflib.get_close_matches(resolved, numeric_cols, n=8, cutoff=0.45)
    hint = f" Closest numeric columns: {', '.join(close)}." if close else ""
    return None, f"Requested column '{requested}' resolved to '{resolved}', but it is not available.{hint}"

def _invoke_tool(name: str, args: dict) -> dict:
    if name not in TOOL_MAP:
        return {"error": f"Unknown tool: {name}"}
    cached = _load_cached_tool_result(name, args)
    if cached is not None:
        _touch_artifact_from_result(cached)
        return cached
    result = TOOL_MAP[name].invoke(args)
    _save_cached_tool_result(name, args, result)
    _touch_artifact_from_result(result)
    return result


def _plot_savefig_for_args(tool_name: str, args: dict) -> str:
    if tool_name == "plot_saved_columns":
        return _generic_plot_filename(
            str(args.get("x_column", "x")),
            str(args.get("y_column", "y")),
            str(args.get("xscale", "linear")),
            str(args.get("yscale", "linear")),
        )
    if tool_name == "plot_saved_3d_columns":
        safe_x = re.sub(r"[^A-Za-z0-9_]+", "_", str(args.get("x_column", "x"))).strip("_")
        safe_y = re.sub(r"[^A-Za-z0-9_]+", "_", str(args.get("y_column", "y"))).strip("_")
        safe_z = re.sub(r"[^A-Za-z0-9_]+", "_", str(args.get("z_column", "z"))).strip("_")
        return f"3d_{safe_z}_vs_{safe_x}_{safe_y}.png"
    quantity = str(args.get("quantity", "quantity"))
    mode = str(args.get("mode", "single"))
    x_column = str(args.get("x_column", "x"))
    xscale = str(args.get("xscale", "linear"))
    prefix = "logx_" if xscale == "log" else ""
    if mode == "all":
        target = "all"
    else:
        target = _channel_column(str(args.get("channel", "channel")), "plot")
    return f"{prefix}{quantity}_{target}_vs_{x_column}.png".replace("plot_", "")


def _message_user_text(message: object) -> str | None:
    if isinstance(message, HumanMessage):
        return str(message.content)
    if isinstance(message, dict) and str(message.get("role", "")).lower() in {"user", "human"}:
        return str(message.get("content", ""))
    return None


def _contextual_plot_request(question: str, history: list) -> str:
    """Restore an explicit saved-file plot request behind a short follow-up."""
    if _explicit_filepath_from_text(question):
        return question
    lowered = " ".join(str(question or "").lower().split())
    followup_markers = (
        "multiple plots", "each channel", "other channels", "where are the other",
        "both axes", "log scale", "logarithmic", "same plot", "that plot",
        "previous plot", "all channels", "all lifetimes",
    )
    if not any(marker in lowered for marker in followup_markers):
        return question
    for message in reversed(history or []):
        text = _message_user_text(message)
        if not text or not _explicit_filepath_from_text(text):
            continue
        prior = " ".join(text.lower().split())
        if any(word in prior for word in ("plot", "lifetime", "versus", "function of")):
            return f"{text}\nFollow-up clarification: {question}"
    return question


def _explicit_all_lifetimes_plot_spec(question: str) -> dict | None:
    """Parse the unambiguous all-channel lifetime-vs-channel request."""
    lowered = " ".join(str(question or "").lower().split())
    wants_all = any(
        phrase in lowered
        for phrase in (
            "all lifetimes", "all lifetime", "all channels", "each channel",
            "other channels", "where are the other",
        )
    )
    if not wants_all or not any(
        token in lowered for token in ("plot", "plots", "versus", "function of", "vs ")
    ):
        return None
    _y_axis, requested_x_axis = _split_generic_plot_axes(question)
    requested_x_axis = requested_x_axis or ""
    requested_x_axis = re.split(
        r"[;,]|\b(?:with|using)?\s*both\s+axes\b|\bin\s+(?:a\s+)?log(?:arithmic)?\s+scale\b|"
        r"\b(?:show|use)\s+(?:a\s+)?heatmap\b|\bcolou?r(?:ing|ed)\b|\bcolou?r\s+by\b",
        requested_x_axis,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    requested_x_axis = re.sub(r"^\s*of\s+", "", requested_x_axis, flags=re.IGNORECASE).strip()
    channel = _extract_channel(requested_x_axis) or _extract_channel(question)
    if channel is not None:
        x_column = channel
    else:
        x_column = _resolve_plot_column(
            requested_x_axis,
            default=_extract_x_column(question),
        )
    if not x_column:
        return None

    color_by = "classification"
    color_match = re.search(
        r"\b(?:colou?r(?:ing|ed)|colou?r\s+by)\b(?:\s+the\s+points?)?\s+(?:by\s+)?(?:the\s+)?"
        r"(?P<parameter>.+?)(?=$|[;,.])",
        question,
        flags=re.IGNORECASE,
    )
    if color_match:
        requested_color = color_match.group("parameter").strip()
        color_by = _resolve_plot_column(requested_color, default=requested_color)
    args = {
        "quantity": "lifetime",
        # The schema retains a channel argument in all-channel mode, although
        # all available channels are plotted.
        "channel": channel or "p->pi0 e+",
        "mode": "all",
        "x_column": x_column,
        "xscale": "log" if any(
            phrase in lowered
            for phrase in ("both axes", "log-log", "log log", "logarithmic", "log scale")
        ) else "linear",
        "color_by": color_by,
    }
    filepath = _explicit_filepath_from_text(question)
    if filepath:
        args["filepath"] = filepath
    args["savefig"] = _plot_savefig_for_args("plot_saved_scan_quantity", args)
    return args


def _explicit_mass_lifetime_heatmap_spec(question: str) -> dict | None:
    """Parse a two-mass scatter plot whose numeric color is a channel lifetime."""
    text = " ".join(str(question or "").split())
    lowered = text.lower()
    if not (
        "plot" in lowered
        and any(word in lowered for word in ("heatmap", "color", "colour"))
        and "lifetime" in lowered
    ):
        return None

    axes_match = re.search(
        r"\bplot\s+(?P<x>.+?)\s+(?:versus|verses|vs\.?|against)\s+"
        r"(?P<y>.+?)(?=\s*[,;]\s*(?:moreover|and|show|with|in)\b|"
        r"\s+\b(?:moreover|show)\b|$)",
        text,
        flags=re.IGNORECASE,
    )
    if not axes_match:
        return None
    x_column = _resolve_plot_column(axes_match.group("x").strip())
    y_column = _resolve_plot_column(axes_match.group("y").strip())
    mass_columns = {
        "MGUT", "M_T", "m0", "mtildeq", "M_sfermion", "m_sfermion", "M_wino",
    }
    if x_column not in mass_columns or y_column not in mass_columns:
        return None

    channel = _extract_channel(text)
    if channel is None and "kaon" in lowered and any(
        token in lowered for token in ("neutrino", "anti-neutrino", "antineutrino", " nu")
    ):
        channel = "p->K+ nu"
    if channel is None:
        return None

    args = {
        "x_column": x_column,
        "y_column": y_column,
        "xscale": "log",
        "yscale": "log",
        "color_by": _channel_column(channel, "lifetime"),
        "color_scale": "log",
    }
    filepath = _explicit_filepath_from_text(question)
    if filepath:
        args["filepath"] = filepath
    args["savefig"] = _plot_savefig_for_args("plot_saved_columns", args)
    return args


RGE_FRAMEWORKS = {"susy": "SUSY", "nonsusy": "nonSUSY", "non-susy": "nonSUSY", "non susy": "nonSUSY"}
SCALE_UNITS = {
    "gev": 1.0,
    "tev": 1.0e3,
    "pev": 1.0e6,
    "eev": 1.0e9,
}
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
RGE_GROUP_ALIASES = {
    "sm": "sm",
    "321": "sm",
    "3221": "3221",
    "lr": "3221",
    "left-right": "3221",
    "left_right": "3221",
    "422": "422",
    "ps": "422",
    "pati-salam": "422",
    "pati_salam": "422",
    "3211": "3211",
    "tto": "3211",
    "421": "421",
    "g214": "421",
}


def _parse_scale(text: str) -> float | None:
    cleaned = text.strip().lower().replace("×", "x").replace(r"\times", "x").replace(r"\cdot", "*")
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = re.sub(rf"({_NUMBER_RE})\s*(?:[x*]\s*|\s+)10\s*\^\s*([-+]?\d+)", r"\1e\2", cleaned)
    cleaned = re.sub(r"10\s*\^\s*([-+]?\d+)", r"1e\1", cleaned)
    match = re.search(rf"({_NUMBER_RE})\s*([a-zA-Z]*)", cleaned)
    if not match:
        return None
    value = _parse_numeric_token(match.group(1))
    if value is None:
        return None
    unit = match.group(2).lower()
    return float(value) * SCALE_UNITS.get(unit, 1.0)


def _normalize_rge_request_text(text: str) -> str:
    """Remove display-only LaTeX wrappers before reading physical inputs."""
    normalized = unicodedata.normalize("NFKC", text or "").translate(_DASH_TRANSLATION)
    normalized = re.sub(r"\\verb\*?([^\w\s])(.*?)\1", r"\2", normalized)
    normalized = normalized.replace("`", "").replace("$", "")
    normalized = normalized.replace(r"\times", "x").replace("×", "x").replace(r"\cdot", "*")
    return normalized.replace("{", "").replace("}", "")


def _parse_field_tuple(text: str) -> tuple[str, ...] | None:
    match = re.search(r"\(([^()]+)\)", text)
    if not match:
        return None
    parts = tuple(part.strip() for part in match.group(1).split(","))
    return parts if len(parts) >= 3 else None


def _parse_field_kind(text: str, *, susy: bool) -> str | None:
    lowered = text.lower()
    if susy and any(word in lowered for word in ["chiral", "superfield", "multiplet"]):
        return "chiral"
    if "scalar" in lowered:
        return "scalar"
    if "fermion" in lowered or "weyl" in lowered or "dirac" in lowered:
        return "fermion"
    return None


def _split_tuple_kind_marker(
    reps: tuple[str, ...],
    kind: str | None,
    expected_len: int,
) -> tuple[tuple[str, ...], str | None, bool]:
    if len(reps) == expected_len:
        return reps, kind, False
    if len(reps) != expected_len + 1:
        return reps, kind, False

    marker = reps[-1].strip().lower()
    marker_kind = None
    real_scalar = False
    if marker in {"r", "real"}:
        marker_kind = "scalar"
        real_scalar = True
    elif marker in {"c", "complex"}:
        marker_kind = "scalar"
    elif marker in {"f", "fermion", "weyl"}:
        marker_kind = "fermion"
    elif marker in {"d", "dirac"}:
        marker_kind = "fermion"
    else:
        raise ValueError("Tuple marker must be R, C, or F.")

    if kind is not None and kind != marker_kind:
        raise ValueError(f"Tuple marker {reps[-1]!r} conflicts with declared {kind} field.")
    return reps[:-1], marker_kind, real_scalar


def _parse_sm_or_mssm_field(text: str, scale: float, *, susy: bool) -> dict | None:
    reps = _parse_field_tuple(text)
    kind = _parse_field_kind(text, susy=susy)
    if reps is None:
        return None
    reps, kind, real_scalar = _split_tuple_kind_marker(reps, kind, 3)
    if kind is None or len(reps) != 3:
        return None
    su3, su2, hypercharge = reps
    field = {
        "name": re.sub(r"\W+", "_", text.strip()).strip("_")[:40] or "threshold",
        "kind": kind,
        "su3": su3,
        "su2": int(su2),
        "hypercharge": hypercharge,
        "mass": float(scale),
    }
    lowered = text.lower()
    marker = _parse_field_tuple(text)[-1].strip().lower() if len(_parse_field_tuple(text) or ()) == 4 else ""
    if kind == "fermion" and ("dirac" in lowered or marker in {"d", "dirac"}):
        field["fermion_type"] = "dirac"
    if kind == "scalar" and real_scalar:
        field["real_scalar"] = True
    return field


def _parse_nonsusy_field_for_group(text: str, scale: float, group: str) -> dict | None:
    if group == "sm":
        return _parse_sm_or_mssm_field(text, scale, susy=False)
    reps = _parse_field_tuple(text)
    kind = _parse_field_kind(text, susy=False)
    if reps is None:
        return None
    model = EXTENDED_RGE_MODELS[canonical_group(group)]
    reps, kind, real_scalar = _split_tuple_kind_marker(reps, kind, len(model.factors))
    if kind is None:
        return None
    if len(reps) != len(model.factors):
        expected = ", ".join(factor.label for factor in model.factors)
        raise ValueError(
            f"Field tuple for group {group} must have {len(model.factors)} entries in order: {expected}; add final R, C, or F."
        )
    field = {
        "name": re.sub(r"\W+", "_", text.strip()).strip("_")[:40] or "threshold",
        "kind": kind,
        "mass": float(scale),
    }
    for factor, value in zip(model.factors, reps):
        field[factor.key] = value
    lowered = text.lower()
    marker = _parse_field_tuple(text)[-1].strip().lower() if len(_parse_field_tuple(text) or ()) == len(model.factors) + 1 else ""
    if kind == "fermion" and ("dirac" in lowered or marker in {"d", "dirac"}):
        field["fermion_type"] = "dirac"
    if kind == "scalar" and real_scalar:
        field["real_scalar"] = True
    return field


def _singular_kind(kind_text: str) -> str:
    lowered = kind_text.lower()
    if "fermion" in lowered:
        return "fermion"
    if "scalar" in lowered:
        return "scalar"
    if "chiral" in lowered or "superfield" in lowered or "multiplet" in lowered:
        return "chiral"
    return kind_text.strip().lower()


def _tuple_has_kind_marker(tuple_text: str) -> bool:
    reps = _parse_field_tuple(tuple_text)
    return bool(reps and reps[-1].strip().lower() in {"r", "real", "c", "complex", "f", "fermion", "weyl", "d", "dirac"})


def _field_segments_from_step(text: str) -> list[str]:
    kind_match = re.search(r"\b(scalars?|fermions?|chiral\s+superfields?|superfields?|multiplets?)\b", text, flags=re.IGNORECASE)
    tuples = re.findall(r"\([^()]+\)", text)
    if not tuples:
        return [text]
    if len(tuples) > 1 and all(_tuple_has_kind_marker(tuple_text) for tuple_text in tuples):
        return list(tuples)
    if kind_match:
        kind = _singular_kind(kind_match.group(1))
        suffix = text[kind_match.end():]
        suffix = re.sub(r"\bat\b\s+.+$", "", suffix, flags=re.IGNORECASE).strip()
        scoped_tuples = re.findall(r"\([^()]+\)", suffix)
        if len(scoped_tuples) <= 1:
            return [text]
        return [f"{kind} {tuple_text}" for tuple_text in scoped_tuples]
    if len(tuples) <= 1:
        return [text]
    return list(tuples)


def _parse_sm_or_mssm_fields(text: str, scale: float, *, susy: bool) -> list[dict] | None:
    fields = []
    for segment in _field_segments_from_step(text):
        field = _parse_sm_or_mssm_field(segment, scale, susy=susy)
        if field is None:
            return None
        fields.append(field)
    return fields


_EXPLICIT_RGE_SCALE_RE = (
    r"(?:\d+(?:\.\d+)?(?:[eE][-+]?\d+)|"
    r"\d+(?:\.\d+)?\s*[x*×]\s*10\s*\^\s*[-+]?\d+|"
    r"\d+(?:\.\d+)?\s*(?:[gGtTpPeE][eE][vV]))"
)


def _explicit_staged_nonsusy_spec(text: str) -> dict | None:
    """Parse an explicit SM -> supported-group staged request as a safety guardrail."""
    group_pattern = re.compile(
        rf"(?P<scale>{_EXPLICIT_RGE_SCALE_RE})(?:\s*[gGtTpPeE][eE][vV])?"
        rf"[^.;]{{0,70}}?\bgroup\s+(?:is\s+)?changed\s+to\s+"
        rf"(?P<group>left[- ]right(?:\s+symmetry)?|pati[- ]salam|3221|422|421|3211|lr|ps|g214|tto)\b",
        flags=re.IGNORECASE,
    )
    changes: list[tuple[float, str]] = []
    for match in group_pattern.finditer(text or ""):
        scale = _parse_scale(match.group("scale"))
        if scale is None:
            return None
        group_text = " ".join(match.group("group").lower().replace("-", " ").split())
        group_alias = {
            "left right": "3221",
            "left right symmetry": "3221",
            "pati salam": "422",
        }.get(group_text, match.group("group"))
        changes.append((float(scale), canonical_group(group_alias)))
    changes.sort(key=lambda item: item[0])
    if not changes or any(right[0] <= left[0] for left, right in zip(changes, changes[1:])):
        return None

    scale_values = [
        value for value in (_parse_scale(match.group(0)) for match in re.finditer(_EXPLICIT_RGE_SCALE_RE, text or ""))
        if value is not None
    ]
    if not scale_values:
        return None
    final_scale = max(float(value) for value in scale_values)
    if final_scale <= changes[-1][0]:
        return None

    fields_by_group: dict[str, list[dict]] = {"sm": []}
    field_pattern = re.compile(
        rf"\bat\s+(?P<scale>{_EXPLICIT_RGE_SCALE_RE})(?:\s*[gGtTpPeE][eE][vV])?\s*,?\s*"
        rf"add\s+(?P<kind>scalars?|fermions?)\s+(?P<body>[^.;]+)",
        flags=re.IGNORECASE,
    )
    for match in field_pattern.finditer(text or ""):
        scale = _parse_scale(match.group("scale"))
        if scale is None:
            return None
        active_group = "sm"
        for change_scale, next_group in changes:
            if change_scale <= float(scale) * (1.0 + 1e-12):
                active_group = next_group
            else:
                break
        kind = _singular_kind(match.group("kind"))
        tuples = re.findall(r"\([^()]+\)", match.group("body"))
        if not tuples:
            return None
        for tuple_text in tuples:
            try:
                field = _parse_nonsusy_field_for_group(
                    f"{kind} {tuple_text}", float(scale), active_group
                )
            except ValueError:
                return None
            if field is None:
                return None
            field["name"] = f"{kind}_{active_group}_{len(fields_by_group.get(active_group, [])) + 1}"
            fields_by_group.setdefault(active_group, []).append(field)

    stages: list[dict] = []
    current_group = "sm"
    for change_scale, next_group in changes:
        stages.append({
            "group": current_group,
            "end": change_scale,
            "fields": fields_by_group.get(current_group, []),
            "include_defaults": current_group != "sm",
        })
        current_group = next_group
    stages.append({
        "group": current_group,
        "end": final_scale,
        "fields": fields_by_group.get(current_group, []),
        "include_defaults": current_group != "sm",
    })

    # A field introduced exactly when its group stage begins is active for the
    # whole stage.  Omitting its redundant mass avoids exp(log(M)) roundoff
    # placing the first solver evaluation infinitesimally below the threshold.
    stage_start = float(DEFAULT_MZ)
    for stage in stages:
        for field in stage["fields"]:
            mass = field.get("mass", field.get("mass_GeV"))
            if mass is not None and np.isclose(float(mass), stage_start, rtol=1e-12, atol=0.0):
                field.pop("mass", None)
                field.pop("mass_GeV", None)
        stage_start = float(stage["end"])

    lowered = " ".join((text or "").lower().split())
    loops = _explicit_rge_loop_order(lowered) or 1
    return {"stages": stages, "loops": loops}


def _explicit_sm_to_mssm_spec(text: str) -> dict | None:
    """Parse an explicit SM-to-MSSM transition and final scale."""
    lowered = " ".join((text or "").lower().split())
    if not re.search(r"\bstandard model\b|\bsm\b", lowered) or not re.search(
        r"\bmssm\b|\bsupersymmetr(?:y|ic)\b|\bsusy\b", lowered
    ):
        return None
    scales = [
        value for value in (_parse_scale(match.group(0)) for match in re.finditer(_EXPLICIT_RGE_SCALE_RE, text or ""))
        if value is not None
    ]
    if len(scales) < 2:
        return None
    susy_scale = float(scales[0])
    gut_scale = float(scales[-1])
    if not DEFAULT_MZ < susy_scale < gut_scale:
        return None
    spec = {"susy_scale": susy_scale, "gut_scale": gut_scale}
    spec["loops"] = _explicit_rge_loop_order(lowered) or 1
    return spec


PD_FRAMEWORKS = {"susy": "SUSY", "non": "nonSUSY", "nonsusy": "nonSUSY", "non-susy": "nonSUSY", "non susy": "nonSUSY"}
PD_GROUPS = {"su5": "SU5", "su(5)": "SU5", "so10": "SO10", "so(10)": "SO10"}


def _normalize_pd_group(value: str) -> str | None:
    return PD_GROUPS.get(value.strip().lower().replace(" ", ""))


def _compute_nonsusy_decay_rates(
    group: str,
    *,
    MGUT: float,
    gGUT: float,
    UL,
    UR,
    DL,
    DR,
    EL,
    ER,
    NL,
) -> dict:
    """Dispatch core matrices to a model-specific, convention-safe formula."""
    normalized_group = _normalize_pd_group(group) or str(group).upper()
    if normalized_group == "SU5":
        return compute_nonsusy_su5_decay_rates(
            MGUT=MGUT,
            gGUT=gGUT,
            U=UL,
            Uc=np.conjugate(UR),
            D=DL,
            Dc=np.conjugate(DR),
            E=EL,
            Ec=np.conjugate(ER),
            N=NL,
        )
    if normalized_group == "SO10":
        return compute_nonsusy_so10_decay_rates(
            MGUT=MGUT,
            gGUT=gGUT,
            UL=UL,
            UR=UR,
            DL=DL,
            DR=DR,
            EL=EL,
            ER=ER,
            NL=NL,
        )
    raise ValueError("Non-SUSY group must be 'SU5' or 'SO10'.")


def _pd_scenario_label(framework: str, group: str) -> str:
    return f"{framework} {group}"


def _uses_su5_file_convention(framework: str, group: str) -> bool:
    return framework == "nonSUSY" and group == "SU5"


def _resolve_input_file_path(filepath: str | os.PathLike) -> Path:
    raw = _normalize_path_text(filepath)
    if not raw:
        raise ValueError("Input file path is empty.")
    return _resolve_project_path(raw, must_exist=True)


def _strip_hash_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _normalize_benchmark_schema(schema: object) -> str:
    normalized = str(schema).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "master_56": MASTER56_SCHEMA,
        "master56": MASTER56_SCHEMA,
        "full": FULL_MATRICES_SCHEMA,
        "matrix": FULL_MATRICES_SCHEMA,
        "matrices": FULL_MATRICES_SCHEMA,
        "full_matrix": FULL_MATRICES_SCHEMA,
        "full_matrices": FULL_MATRICES_SCHEMA,
    }
    return aliases.get(normalized, normalized)


def _read_benchmark_payload(filepath: str | os.PathLike) -> dict:
    path = _resolve_input_file_path(filepath)
    raw_text = path.read_text(encoding="utf-8")
    cleaned = _strip_hash_comments(raw_text).strip()
    if not cleaned:
        raise ValueError(f"Input file is empty after removing comments: {path}")

    payload = None
    if path.suffix.lower() == ".json":
        payload = json.loads(cleaned)
    else:
        try:
            payload = ast.literal_eval(cleaned)
        except (SyntaxError, ValueError):
            values = [float(token) for token in re.findall(_NUMBER_RE, cleaned)]
            payload = values

    if isinstance(payload, dict):
        schema = _normalize_benchmark_schema(payload.get("schema", MASTER56_SCHEMA))
        values = payload.get("values", payload.get("vector", payload.get("parameters")))
        mgut = payload.get("MGUT", payload.get("mgut", MASTER56_MGUT))
        ggut = payload.get("gGUT", payload.get("ggut", MASTER56_GGUT))
    else:
        schema = MASTER56_SCHEMA
        values = payload
        mgut = MASTER56_MGUT
        ggut = MASTER56_GGUT

    if schema == FULL_MATRICES_SCHEMA:
        return {
            "path": path,
            "schema": schema,
            "raw": payload,
            "values": None,
            "n_file_values": None,
            "MGUT": float(mgut),
            "gGUT": float(ggut),
        }

    if values is None:
        raise ValueError(f"Benchmark file {path} does not contain a values/vector/parameters list.")

    value_list = [float(value) for value in values]
    n_file_values = len(value_list)
    if len(value_list) == len(MASTER56_PARAMETER_ORDER) + 2:
        mgut, ggut = value_list[:2]
        value_list = value_list[2:]

    output = {
        "path": path,
        "schema": schema,
        "values": value_list,
        "n_file_values": n_file_values,
        "MGUT": float(mgut),
        "gGUT": float(ggut),
    }
    return output


def _master56_matrix(prefix: str, params: dict[str, float]) -> np.ndarray:
    return build_phased_mixing_matrix(
        params[f"{prefix}_theta12"],
        params[f"{prefix}_theta23"],
        params[f"{prefix}_theta13"],
        params[f"{prefix}_delta"],
        left_phases=(params[f"{prefix}_phi1"], params[f"{prefix}_phi2"], params[f"{prefix}_phi3"]),
        right_phases=(params[f"{prefix}_phi4"], params[f"{prefix}_phi5"]),
    )


def _master56_params_from_values(values: list[float], MGUT: float, gGUT: float, framework: str, group: str) -> dict:
    if len(values) != len(MASTER56_PARAMETER_ORDER):
        order = ", ".join(MASTER56_PARAMETER_ORDER)
        raise ValueError(
            f"Schema {MASTER56_SCHEMA!r} expects {len(MASTER56_PARAMETER_ORDER)} values, "
            f"got {len(values)}. Required order: {order}"
        )

    parameter_values = dict(zip(MASTER56_PARAMETER_ORDER, values))
    UL = _master56_matrix("UL", parameter_values)
    right_up = _master56_matrix("UR", parameter_values)
    right_down = _master56_matrix("DR", parameter_values)
    EL = _master56_matrix("EL", parameter_values)
    right_charged_lepton = _master56_matrix("ER", parameter_values)

    if _uses_su5_file_convention(framework, group):
        # SU(5) benchmark files use Uc/Dc/Ec notation; the lifetime core uses UR/DR/ER.
        UR = np.conjugate(right_up)
        DR = np.conjugate(right_down)
        ER = np.conjugate(right_charged_lepton)
        input_convention = "SU(5): Uc/Dc/Ec inputs mapped with UR=conj(Uc), DR=conj(Dc), ER=conj(Ec)"
    else:
        UR = right_up
        DR = right_down
        ER = right_charged_lepton
        input_convention = "SO(10): UL/UR/DR/EL/ER inputs used directly"

    Uckm = build_phased_mixing_matrix(
        MASTER56_CKM_FIXED["theta12"],
        MASTER56_CKM_FIXED["theta23"],
        MASTER56_CKM_FIXED["theta13"],
        MASTER56_CKM_FIXED["delta"],
        left_phases=tuple(parameter_values[f"UCKM_phi{index}"] for index in range(1, 4)),
        right_phases=tuple(parameter_values[f"UCKM_phi{index}"] for index in range(4, 6)),
    )
    Upmns = build_phased_mixing_matrix(
        MASTER56_PMNS_FIXED["theta12"],
        MASTER56_PMNS_FIXED["theta23"],
        MASTER56_PMNS_FIXED["theta13"],
        parameter_values["UPMNS_delta"],
        left_phases=tuple(parameter_values[f"UPMNS_phi{index}"] for index in range(1, 4)),
        right_phases=tuple(parameter_values[f"UPMNS_phi{index}"] for index in range(4, 6)),
    )

    return {
        "MGUT": float(MGUT),
        "gGUT": float(gGUT),
        "UL": UL,
        "UR": UR,
        "DL": UL @ Uckm,
        "DR": DR,
        "EL": EL,
        "ER": ER,
        "NL": EL @ Upmns,
        "matrix_parameters": parameter_values,
        "derived_matrices": {"Uckm": Uckm, "Upmns": Upmns},
        "input_convention": input_convention,
    }


def _matrix_key_lookup(mapping: dict) -> dict[str, object]:
    return {str(key).strip().lower().replace("_", ""): value for key, value in mapping.items()}


def _parse_matrix_payload(value: object, name: str) -> np.ndarray:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.lower() in {"i", "id", "identity", "eye"}:
            return np.eye(3, dtype=complex)
        try:
            value = ast.literal_eval(cleaned)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Matrix {name} must be a 3x3 list or 'identity'.") from exc
    try:
        matrix = np.asarray(value, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Matrix {name} entries must be numeric or complex numbers.") from exc
    if matrix.shape != (3, 3):
        raise ValueError(f"Matrix {name} must be 3x3; got shape {matrix.shape}.")
    return matrix


def _full_matrix_params_from_payload(payload: dict, MGUT: float, gGUT: float, framework: str, group: str) -> dict:
    matrix_source = payload.get("matrices", payload)
    if not isinstance(matrix_source, dict):
        raise ValueError(f"Schema {FULL_MATRICES_SCHEMA!r} needs a matrix dictionary.")

    lookup = _matrix_key_lookup(matrix_source)

    if group == "SU5":
        required = SU5_MATRIX_INPUT_NAMES
        missing = [name for name in required if name.lower() not in lookup]
        if missing:
            raise ValueError(
                f"SU(5) {FULL_MATRICES_SCHEMA!r} input needs matrices: {', '.join(required)}. "
                f"Missing: {', '.join(missing)}."
            )
        U = _parse_matrix_payload(lookup["u"], "U")
        D = _parse_matrix_payload(lookup["d"], "D")
        E = _parse_matrix_payload(lookup["e"], "E")
        Uc = _parse_matrix_payload(lookup["uc"], "Uc")
        Dc = _parse_matrix_payload(lookup["dc"], "Dc")
        Ec = _parse_matrix_payload(lookup["ec"], "Ec")
        N = _parse_matrix_payload(lookup["n"], "N")
        return {
            "MGUT": float(MGUT),
            "gGUT": float(gGUT),
            "UL": U,
            "UR": np.conjugate(Uc),
            "DL": D,
            "DR": np.conjugate(Dc),
            "EL": E,
            "ER": np.conjugate(Ec),
            "NL": N,
            "matrix_parameters": {},
            "derived_matrices": {},
            "input_convention": "SU(5): full U/D/E/Uc/Dc/Ec/N inputs mapped to SO(10)-style core matrices",
            "provided_matrix_inputs": required,
            "derived_inputs": ["UR=conj(Uc)", "DR=conj(Dc)", "ER=conj(Ec)"],
        }

    required = SO10_MATRIX_INPUT_NAMES
    missing = [name for name in required if name.lower() not in lookup]
    if missing:
        raise ValueError(
            f"SO(10) {FULL_MATRICES_SCHEMA!r} input needs matrices: {', '.join(required)}. "
            f"Missing: {', '.join(missing)}."
        )
    return {
        "MGUT": float(MGUT),
        "gGUT": float(gGUT),
        "UL": _parse_matrix_payload(lookup["ul"], "UL"),
        "UR": _parse_matrix_payload(lookup["ur"], "UR"),
        "DL": _parse_matrix_payload(lookup["dl"], "DL"),
        "DR": _parse_matrix_payload(lookup["dr"], "DR"),
        "EL": _parse_matrix_payload(lookup["el"], "EL"),
        "ER": _parse_matrix_payload(lookup["er"], "ER"),
        "NL": _parse_matrix_payload(lookup["nl"], "NL"),
        "matrix_parameters": {},
        "derived_matrices": {},
        "input_convention": "SO(10): full UL/UR/DL/DR/EL/ER/NL inputs used directly",
        "provided_matrix_inputs": required,
        "derived_inputs": [],
    }


def _load_benchmark_parameters_from_file(
    filepath: str | os.PathLike,
    framework: str,
    group: str,
    MGUT: float | None = None,
    gGUT: float | None = None,
) -> tuple[dict, dict]:
    payload = _read_benchmark_payload(filepath)
    schema = payload["schema"]
    mgut = float(MGUT) if MGUT is not None else payload["MGUT"]
    ggut = float(gGUT) if gGUT is not None else payload["gGUT"]

    if schema == FULL_MATRICES_SCHEMA:
        params = _full_matrix_params_from_payload(payload["raw"], mgut, ggut, framework, group)
        metadata = {
            "input_file": str(payload["path"]),
            "schema": schema,
            "input_convention": params["input_convention"],
            "n_values": len(params["provided_matrix_inputs"]),
            "n_file_values": len(params["provided_matrix_inputs"]) + 2,
            "values": None,
            "parameter_order": list(params["provided_matrix_inputs"]),
            "provided_matrix_inputs": list(params["provided_matrix_inputs"]),
            "derived_inputs": list(params["derived_inputs"]),
        }
        return params, metadata

    if schema != MASTER56_SCHEMA:
        raise ValueError(f"Unknown benchmark schema {schema!r}. Supported schemas: {MASTER56_SCHEMA}, {FULL_MATRICES_SCHEMA}.")
    if not _uses_su5_file_convention(framework, group):
        raise ValueError(
            f"Schema {MASTER56_SCHEMA!r} is the reduced SU(5) CKM/PMNS reconstruction schema. "
            f"For SO(10), use schema {FULL_MATRICES_SCHEMA!r} with UL, UR, DL, DR, EL, ER, NL."
        )

    params = _master56_params_from_values(payload["values"], mgut, ggut, framework, group)
    metadata = {
        "input_file": str(payload["path"]),
        "schema": schema,
        "input_convention": params["input_convention"],
        "n_values": len(payload["values"]),
        "n_file_values": payload["n_file_values"],
        "values": list(payload["values"]),
        "parameter_order": list(MASTER56_PARAMETER_ORDER),
    }
    return params, metadata


def _master56_chi_sq_terms(lifetimes_years: dict[str, float]) -> dict[str, float]:
    limits = get_current_limits()
    return {
        channel: float(limits[channel]) / max(float(lifetime), 1.0e-300)
        for channel, lifetime in lifetimes_years.items()
        if channel in limits
    }


def _master56_chi_sq(lifetimes_years: dict[str, float]) -> float:
    terms = _master56_chi_sq_terms(lifetimes_years)
    return float(max(terms.values())) if terms else np.inf


def _core_matrices_from_params(params: dict) -> dict[str, np.ndarray]:
    return {name: np.asarray(params[name], dtype=complex).copy() for name in SO10_MATRIX_INPUT_NAMES}


def _reflect_into_range(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return float(lower)
    period = upper - lower
    reflected = (float(value) - lower) % (2.0 * period)
    if reflected > period:
        reflected = 2.0 * period - reflected
    return float(lower + reflected)


def _wrap_into_range(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return float(lower)
    return float(((float(value) - lower) % (upper - lower)) + lower)


def _proposal_for_master56_state(
    vector: np.ndarray,
    MGUT: float,
    gGUT: float,
    rng: np.random.Generator,
    vary: set[str],
    angle_width: float,
    phase_width: float,
    mgut_log_width: float,
    ggut_width: float,
    mgut_log_bounds: tuple[float, float],
    ggut_bounds: tuple[float, float],
) -> tuple[np.ndarray, float, float]:
    proposed = np.asarray(vector, dtype=float).copy()
    if "flavor" in vary:
        proposed[:15] += rng.normal(0.0, float(angle_width), size=15)
        proposed[15:] += rng.normal(0.0, float(phase_width), size=len(proposed) - 15)
        for index in range(15):
            proposed[index] = _reflect_into_range(proposed[index], 0.0, np.pi / 2.0)
        for index in range(15, len(proposed)):
            proposed[index] = _wrap_into_range(proposed[index], -np.pi, np.pi)

    proposed_mgut = float(MGUT)
    proposed_ggut = float(gGUT)
    if "MGUT" in vary:
        proposed_log_mgut = np.log10(max(proposed_mgut, 1.0e-300)) + rng.normal(0.0, float(mgut_log_width))
        proposed_log_mgut = _reflect_into_range(proposed_log_mgut, *mgut_log_bounds)
        proposed_mgut = float(10.0 ** proposed_log_mgut)
    if "gGUT" in vary:
        proposed_ggut = _reflect_into_range(
            proposed_ggut + rng.normal(0.0, float(ggut_width)),
            *ggut_bounds,
        )

    return proposed, proposed_mgut, proposed_ggut


def _evaluate_master56_state(
    framework: str,
    group: str,
    vector: np.ndarray,
    MGUT: float,
    gGUT: float,
) -> dict:
    params = _master56_params_from_values(list(np.asarray(vector, dtype=float)), MGUT, gGUT, framework, group)
    result = _compute_nonsusy_decay_rates(
        group,
        MGUT=params["MGUT"],
        gGUT=params["gGUT"],
        UL=params["UL"],
        UR=params["UR"],
        DL=params["DL"],
        DR=params["DR"],
        EL=params["EL"],
        ER=params["ER"],
        NL=params["NL"],
    )
    lifetimes = {channel: float(value) for channel, value in result["lifetimes_years"].items()}
    chi_sq_terms = _master56_chi_sq_terms(lifetimes)
    return {
        "MGUT": float(params["MGUT"]),
        "gGUT": float(params["gGUT"]),
        "input_convention": params["input_convention"],
        "vector": np.asarray(vector, dtype=float).copy(),
        "lifetimes_years": lifetimes,
        "widths_GeV": {channel: float(value) for channel, value in result["widths"].items()},
        "branching_ratios": {channel: float(value) for channel, value in result["branching_ratios"].items()},
        "total_width_GeV": float(result["total_width_GeV"]),
        "classification": classify_model(lifetimes),
        "chi_sq": _master56_chi_sq(lifetimes),
        "chi_sq_terms": chi_sq_terms,
    }


def _master56_mcmc_row(
    sample_index: int,
    step: int,
    accepted_moves: int,
    proposals: int,
    state: dict,
) -> dict:
    vector = state["vector"]
    lifetimes = state["lifetimes_years"]
    min_channel, min_lifetime = min(lifetimes.items(), key=lambda item: item[1])
    row = {
        "scan_schema_version": SCAN_SCHEMA_VERSION,
        "scan_index": int(sample_index),
        "mcmc_step": int(step),
        "mcmc_accepted_moves": int(accepted_moves),
        "mcmc_acceptance_rate": float(accepted_moves / max(proposals, 1)) if proposals else np.nan,
        "mcmc_allowed": bool(state["chi_sq"] < 1.0),
        "MGUT": float(state["MGUT"]),
        "gGUT": float(state["gGUT"]),
        "classification": state["classification"],
        "chi_sq": float(state["chi_sq"]),
        "total_width_GeV": float(state["total_width_GeV"]),
        "min_lifetime_years": float(min_lifetime),
        "min_lifetime_channel": str(min_channel),
    }
    for name, value in zip(MASTER56_PARAMETER_ORDER, vector):
        row[f"master56_{name}"] = float(value)
    for channel, value in lifetimes.items():
        row[_channel_column(channel, "lifetime")] = float(value)
        row[f"log10_{_channel_column(channel, 'lifetime')}"] = float(np.log10(max(float(value), 1.0e-300)))
    for channel, value in state["widths_GeV"].items():
        row[_channel_column(channel, "width")] = float(value)
    for channel, value in state["branching_ratios"].items():
        row[_channel_column(channel, "branching")] = float(value)
    for channel, value in state["chi_sq_terms"].items():
        row[_channel_column(channel, "chi_sq")] = float(value)
    return row


def _small_unitary_step(rng: np.random.Generator, width: float) -> np.ndarray:
    z = rng.normal(0.0, 1.0, size=(3, 3)) + 1j * rng.normal(0.0, 1.0, size=(3, 3))
    anti_hermitian = z - z.conj().T
    q, r = np.linalg.qr(np.eye(3, dtype=complex) + float(width) * anti_hermitian)
    diagonal = np.diag(r)
    phases = np.ones_like(diagonal, dtype=complex)
    nonzero = np.abs(diagonal) > 0
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    return q @ np.diag(np.conjugate(phases))


def _proposal_for_full_matrix_state(
    matrices: dict[str, np.ndarray],
    MGUT: float,
    gGUT: float,
    rng: np.random.Generator,
    vary: set[str],
    matrix_width: float,
    mgut_log_width: float,
    ggut_width: float,
    mgut_log_bounds: tuple[float, float],
    ggut_bounds: tuple[float, float],
) -> tuple[dict[str, np.ndarray], float, float]:
    proposed = {name: np.asarray(matrix, dtype=complex).copy() for name, matrix in matrices.items()}
    if "flavor" in vary:
        for name in SO10_MATRIX_INPUT_NAMES:
            proposed[name] = _small_unitary_step(rng, matrix_width) @ proposed[name]

    proposed_mgut = float(MGUT)
    proposed_ggut = float(gGUT)
    if "MGUT" in vary:
        proposed_log_mgut = np.log10(max(proposed_mgut, 1.0e-300)) + rng.normal(0.0, float(mgut_log_width))
        proposed_log_mgut = _reflect_into_range(proposed_log_mgut, *mgut_log_bounds)
        proposed_mgut = float(10.0 ** proposed_log_mgut)
    if "gGUT" in vary:
        proposed_ggut = _reflect_into_range(
            proposed_ggut + rng.normal(0.0, float(ggut_width)),
            *ggut_bounds,
        )

    return proposed, proposed_mgut, proposed_ggut


def _evaluate_full_matrix_state(
    framework: str,
    group: str,
    matrices: dict[str, np.ndarray],
    MGUT: float,
    gGUT: float,
    input_convention: str = "",
) -> dict:
    core = {name: np.asarray(matrices[name], dtype=complex).copy() for name in SO10_MATRIX_INPUT_NAMES}
    result = _compute_nonsusy_decay_rates(
        group,
        MGUT=float(MGUT),
        gGUT=float(gGUT),
        UL=core["UL"],
        UR=core["UR"],
        DL=core["DL"],
        DR=core["DR"],
        EL=core["EL"],
        ER=core["ER"],
        NL=core["NL"],
    )
    lifetimes = {channel: float(value) for channel, value in result["lifetimes_years"].items()}
    chi_sq_terms = _master56_chi_sq_terms(lifetimes)
    return {
        "MGUT": float(MGUT),
        "gGUT": float(gGUT),
        "input_convention": input_convention or "SO(10): full UL/UR/DL/DR/EL/ER/NL inputs used directly",
        "matrices": core,
        "lifetimes_years": lifetimes,
        "widths_GeV": {channel: float(value) for channel, value in result["widths"].items()},
        "branching_ratios": {channel: float(value) for channel, value in result["branching_ratios"].items()},
        "total_width_GeV": float(result["total_width_GeV"]),
        "classification": classify_model(lifetimes),
        "chi_sq": _master56_chi_sq(lifetimes),
        "chi_sq_terms": chi_sq_terms,
    }


def _full_matrix_mcmc_row(
    sample_index: int,
    step: int,
    accepted_moves: int,
    proposals: int,
    state: dict,
) -> dict:
    lifetimes = state["lifetimes_years"]
    min_channel, min_lifetime = min(lifetimes.items(), key=lambda item: item[1])
    row = {
        "scan_schema_version": SCAN_SCHEMA_VERSION,
        "scan_index": int(sample_index),
        "mcmc_step": int(step),
        "mcmc_accepted_moves": int(accepted_moves),
        "mcmc_acceptance_rate": float(accepted_moves / max(proposals, 1)) if proposals else np.nan,
        "mcmc_allowed": bool(state["chi_sq"] < 1.0),
        "MGUT": float(state["MGUT"]),
        "gGUT": float(state["gGUT"]),
        "classification": state["classification"],
        "chi_sq": float(state["chi_sq"]),
        "total_width_GeV": float(state["total_width_GeV"]),
        "min_lifetime_years": float(min_lifetime),
        "min_lifetime_channel": str(min_channel),
    }
    for matrix_name, matrix in state["matrices"].items():
        for i in range(3):
            for j in range(3):
                value = complex(matrix[i, j])
                prefix = f"matrix_{matrix_name}_{i + 1}{j + 1}"
                row[f"{prefix}_re"] = float(value.real)
                row[f"{prefix}_im"] = float(value.imag)
                row[f"{prefix}_abs"] = float(abs(value))
    for channel, value in lifetimes.items():
        row[_channel_column(channel, "lifetime")] = float(value)
        row[f"log10_{_channel_column(channel, 'lifetime')}"] = float(np.log10(max(float(value), 1.0e-300)))
    for channel, value in state["widths_GeV"].items():
        row[_channel_column(channel, "width")] = float(value)
    for channel, value in state["branching_ratios"].items():
        row[_channel_column(channel, "branching")] = float(value)
    for channel, value in state["chi_sq_terms"].items():
        row[_channel_column(channel, "chi_sq")] = float(value)
    return row


def _run_metropolis_mcmc(
    initial_state: dict,
    propose_fn,
    row_fn,
    n_samples: int,
    burn_in: int,
    thinning: int,
    rng: np.random.Generator,
) -> tuple[list[dict], int, int, int, dict, dict]:
    current = initial_state
    current_chi_sq = float(current["chi_sq"])
    best = dict(current)
    rows = []
    accepted_moves = 0
    proposals = 0
    invalid_proposals = 0
    if burn_in == 0:
        rows.append(row_fn(0, 0, 0, 0, current))

    total_iterations = burn_in + n_samples * thinning
    for iteration in range(total_iterations):
        proposals += 1
        proposed = None
        try:
            proposed = propose_fn(current, rng)
            proposed_chi_sq = float(proposed["chi_sq"])
            if not np.isfinite(proposed_chi_sq):
                proposed_chi_sq = INVALID_MCMC_CHI_SQ
                invalid_proposals += 1
        except Exception:
            proposed_chi_sq = INVALID_MCMC_CHI_SQ
            invalid_proposals += 1

        delta = proposed_chi_sq - current_chi_sq
        if proposed is not None and (delta <= 0.0 or np.log(rng.uniform()) < -0.5 * delta):
            current = proposed
            current_chi_sq = proposed_chi_sq
            accepted_moves += 1
            if current_chi_sq < float(best["chi_sq"]):
                best = dict(current)

        if iteration >= burn_in and (iteration - burn_in) % thinning == 0:
            if len(rows) < n_samples:
                rows.append(row_fn(len(rows), proposals, accepted_moves, proposals, current))
            if len(rows) >= n_samples:
                break

    return rows, accepted_moves, proposals, invalid_proposals, current, best




def _resolve_mcmc_n_jobs(n_jobs: int | None, n_samples: int) -> int:
    cpu_count = os.cpu_count() or 1
    requested = -1 if n_jobs is None else int(n_jobs)
    if requested == 0:
        raise ValueError("n_jobs must be nonzero.")
    if requested < 0:
        auto_limit = max(1, int(os.environ.get("PROTON_AGENT_MAX_MCMC_WORKERS", "8")))
        resolved = min(max(cpu_count + 1 + requested, 1), auto_limit)
    else:
        resolved = requested
    # Each chain should have at least one proposal when burn_in=0, so cap at n//2.
    max_useful_chains = max(1, int(n_samples) // 2)
    return max(1, min(resolved, cpu_count, max_useful_chains))


def _split_mcmc_sample_counts(n_samples: int, n_chains: int) -> list[int]:
    base, remainder = divmod(int(n_samples), int(n_chains))
    return [base + (1 if index < remainder else 0) for index in range(n_chains)]


def _run_metropolis_mcmc_chain(
    chain_index: int,
    chain_samples: int,
    chain_seed: int,
    initial_state: dict,
    propose_fn,
    row_fn,
    burn_in: int,
    thinning: int,
) -> tuple[list[dict], int, int, int, dict, dict]:
    print(
        f"[MCMC] chain {chain_index + 1} started: {chain_samples} retained samples "
        f"(burn-in={burn_in}, thinning={thinning})"
    )
    rows, accepted_moves, proposals, invalid_proposals, current, best = _run_metropolis_mcmc(
        initial_state,
        propose_fn,
        row_fn,
        chain_samples,
        burn_in,
        thinning,
        np.random.default_rng(int(chain_seed)),
    )
    for row in rows:
        row["mcmc_chain"] = int(chain_index)
        row["mcmc_chain_sample_index"] = int(row.get("scan_index", 0))
    print(
        f"[MCMC] chain {chain_index + 1} completed: {len(rows)} samples, "
        f"{proposals} proposals"
    )
    return rows, accepted_moves, proposals, invalid_proposals, current, best


def _run_parallel_metropolis_mcmc(
    initial_state: dict,
    propose_fn,
    row_fn,
    n_samples: int,
    burn_in: int,
    thinning: int,
    seed: int,
    n_jobs: int | None = -1,
) -> tuple[list[dict], int, int, int, dict, dict, int, list[int]]:
    n_chains = _resolve_mcmc_n_jobs(n_jobs, n_samples)
    sample_counts = _split_mcmc_sample_counts(n_samples, n_chains)
    seed_rng = np.random.default_rng(int(seed))
    chain_seeds = seed_rng.integers(0, np.iinfo(np.uint32).max, size=n_chains, dtype=np.uint32).astype(int).tolist()

    if n_chains == 1:
        result_items = [
            _run_metropolis_mcmc_chain(
                0,
                sample_counts[0],
                chain_seeds[0],
                initial_state,
                propose_fn,
                row_fn,
                burn_in,
                thinning,
            )
        ]
    else:
        with parallel_backend("loky", inner_max_num_threads=1):
            result_items = Parallel(n_jobs=n_chains)(
                delayed(call_with_bounds)(current_bounds_configuration(), call_with_matrix_elements,
                    current_matrix_elements(), _run_metropolis_mcmc_chain,
                    chain_index,
                    chain_samples,
                    chain_seeds[chain_index],
                    initial_state,
                    propose_fn,
                    row_fn,
                    burn_in,
                    thinning,
                )
                for chain_index, chain_samples in enumerate(sample_counts)
            )

    rows = []
    accepted_moves = 0
    proposals = 0
    invalid_proposals = 0
    current = result_items[0][4]
    best = result_items[0][5]
    for chain_rows, chain_accepted, chain_proposals, chain_invalid, chain_current, chain_best in result_items:
        rows.extend(chain_rows)
        accepted_moves += int(chain_accepted)
        proposals += int(chain_proposals)
        invalid_proposals += int(chain_invalid)
        current = chain_current
        if float(chain_best["chi_sq"]) < float(best["chi_sq"]):
            best = chain_best
    for index, row in enumerate(rows):
        row["scan_index"] = int(index)
    return rows, accepted_moves, proposals, invalid_proposals, current, best, n_chains, sample_counts

def _mcmc_dataframe_summary(df: pd.DataFrame) -> dict:
    classification_counts = df["classification"].value_counts().to_dict()
    n_allowed = int(df["mcmc_allowed"].sum())
    n_excluded = int((~df["mcmc_allowed"]).sum())
    best_idx = df["chi_sq"].idxmin()
    min_lifetime_idx = df["min_lifetime_years"].idxmin()
    chi_sq_cols = [col for col in df.columns if col.startswith("chi_sq_p_to_")]
    best_chi_sq_channel = None
    best_chi_sq_channel_ratio = None
    if chi_sq_cols:
        best_chi_col = df.loc[best_idx, chi_sq_cols].astype(float).idxmax()
        best_chi_sq_channel = _channel_from_lifetime_column(best_chi_col.replace("chi_sq_", "lifetime_")) or best_chi_col
        best_chi_sq_channel_ratio = float(df.loc[best_idx, best_chi_col])
    return {
        "n_allowed": n_allowed,
        "n_excluded": n_excluded,
        "classification_counts": {str(key): int(value) for key, value in classification_counts.items()},
        "initial_chi_sq": float(df.iloc[0]["chi_sq"]),
        "best_chi_sq": float(df.loc[best_idx, "chi_sq"]),
        "best_chi_sq_scan_index": int(df.loc[best_idx, "scan_index"]),
        "best_chi_sq_sample_allowed": bool(df.loc[best_idx, "mcmc_allowed"]),
        "best_chi_sq_most_constraining_channel": best_chi_sq_channel,
        "best_chi_sq_most_constraining_ratio": best_chi_sq_channel_ratio,
        "min_chi_sq": float(df["chi_sq"].min()),
        "max_chi_sq": float(df["chi_sq"].max()),
        "fraction_allowed": float(df["mcmc_allowed"].mean()),
        "minimum_lifetime_years": float(df.loc[min_lifetime_idx, "min_lifetime_years"]),
        "minimum_lifetime_channel": str(df.loc[min_lifetime_idx, "min_lifetime_channel"]),
        "minimum_lifetime_scan_index": int(df.loc[min_lifetime_idx, "scan_index"]),
        "minimum_lifetime_chi_sq": float(df.loc[min_lifetime_idx, "chi_sq"]),
        "minimum_lifetime_sample_allowed": bool(df.loc[min_lifetime_idx, "mcmc_allowed"]),
    }


def _matrix_digest(matrices: dict[str, np.ndarray]) -> str:
    payload = b"".join(
        np.asarray(matrices[name], dtype=complex).view(np.float64).tobytes()
        for name in SO10_MATRIX_INPUT_NAMES
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def _run_master56_mcmc(
    framework: str,
    group: str,
    initial_vector: list[float],
    MGUT: float,
    gGUT: float,
    n_samples: int = 500,
    vary: set[str] | None = None,
    seed: int = 24680,
    angle_width: float = 2.0e-4,
    phase_width: float = 2.0e-4,
    mgut_log_width: float = 2.0e-3,
    ggut_width: float = 1.0e-3,
    burn_in: int = 0,
    thinning: int = 1,
    n_jobs: int = -1,
) -> dict:
    n_samples = max(1, min(int(n_samples), 200000))
    burn_in = max(0, int(burn_in))
    thinning = max(1, int(thinning))
    vary = set(vary or {"flavor"})
    vector = np.asarray(initial_vector, dtype=float)
    if len(vector) != len(MASTER56_PARAMETER_ORDER):
        raise ValueError(f"MCMC needs {len(MASTER56_PARAMETER_ORDER)} master56 flavor values, got {len(vector)}.")

    initial_state = _evaluate_master56_state(framework, group, vector, float(MGUT), float(gGUT))

    def propose(current: dict, step_rng: np.random.Generator) -> dict:
        proposed_vector, proposed_mgut, proposed_ggut = _proposal_for_master56_state(
            current["vector"],
            current["MGUT"],
            current["gGUT"],
            step_rng,
            vary,
            angle_width,
            phase_width,
            mgut_log_width,
            ggut_width,
            (10.0, 19.0),
            (0.1, 1.5),
        )
        return _evaluate_master56_state(framework, group, proposed_vector, proposed_mgut, proposed_ggut)

    rows, accepted_moves, proposals, invalid_proposals, current, best, n_chains, chain_sample_counts = _run_parallel_metropolis_mcmc(
        initial_state,
        propose,
        _master56_mcmc_row,
        n_samples,
        burn_in,
        thinning,
        seed,
        n_jobs,
    )
    df = pd.DataFrame(rows)
    scan_summary = _mcmc_dataframe_summary(df)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    summary_seed = {
        "scenario": _pd_scenario_label(framework, group),
        "n_samples": n_samples,
        "vary": sorted(vary),
        "seed": int(seed),
        "initial_MGUT": float(MGUT),
        "initial_gGUT": float(gGUT),
        "initial_vector_digest": hashlib.sha256(np.asarray(initial_vector, dtype=float).tobytes()).hexdigest()[:16],
    }
    digest = _cache_key("master56_mcmc", summary_seed)[:8]
    scan_filepath = str(MCMC_RESULTS_DIR / f"master56_mcmc_{stamp}_{digest}.h5")
    df.to_hdf(scan_filepath, key="results", mode="w")

    summary = {
        **summary_seed,
        "scan_mode": "mcmc",
        "sampler": "unconstrained_metropolis",
        "schema": MASTER56_SCHEMA,
        "input_convention": str(current.get("input_convention", "")),
        "master56_values": [float(value) for value in initial_vector],
        "master56_parameter_order": list(MASTER56_PARAMETER_ORDER),
        "n_retained": int(len(df)),
        "n_requested": int(n_samples),
        "n_proposals": int(proposals),
        "n_invalid_proposals": int(invalid_proposals),
        "n_accepted_moves": int(accepted_moves),
        "n_jobs": int(n_chains),
        "n_chains": int(n_chains),
        "chain_sample_counts": [int(count) for count in chain_sample_counts],
        "acceptance_rate": float(accepted_moves / max(proposals, 1)),
        **scan_summary,
        "angle_width": float(angle_width),
        "phase_width": float(phase_width),
        "mgut_log_width": float(mgut_log_width),
        "ggut_width": float(ggut_width),
        "burn_in": int(burn_in),
        "thinning": int(thinning),
        "scan_filepath": scan_filepath,
    }
    artifact = _register_artifact(
        "mcmc-scan",
        payload={"scan_filepath": scan_filepath, "config": summary_seed, "summary": summary},
        summary=summary,
    )
    return {**summary, "artifact_id": artifact["artifact_id"]}


def _run_full_matrix_mcmc(
    framework: str,
    group: str,
    initial_matrices: dict[str, np.ndarray],
    MGUT: float,
    gGUT: float,
    n_samples: int = 500,
    vary: set[str] | None = None,
    seed: int = 24680,
    matrix_width: float = 2.0e-4,
    mgut_log_width: float = 2.0e-3,
    ggut_width: float = 1.0e-3,
    burn_in: int = 0,
    thinning: int = 1,
    input_convention: str = "",
    n_jobs: int = -1,
) -> dict:
    n_samples = max(1, min(int(n_samples), 200000))
    burn_in = max(0, int(burn_in))
    thinning = max(1, int(thinning))
    vary = set(vary or {"flavor"})
    matrices = {name: np.asarray(initial_matrices[name], dtype=complex).copy() for name in SO10_MATRIX_INPUT_NAMES}

    initial_state = _evaluate_full_matrix_state(framework, group, matrices, float(MGUT), float(gGUT), input_convention)

    def propose(current: dict, step_rng: np.random.Generator) -> dict:
        proposed_matrices, proposed_mgut, proposed_ggut = _proposal_for_full_matrix_state(
            current["matrices"],
            current["MGUT"],
            current["gGUT"],
            step_rng,
            vary,
            matrix_width,
            mgut_log_width,
            ggut_width,
            (10.0, 19.0),
            (0.1, 1.5),
        )
        return _evaluate_full_matrix_state(
            framework,
            group,
            proposed_matrices,
            proposed_mgut,
            proposed_ggut,
            input_convention,
        )

    rows, accepted_moves, proposals, invalid_proposals, current, best, n_chains, chain_sample_counts = _run_parallel_metropolis_mcmc(
        initial_state,
        propose,
        _full_matrix_mcmc_row,
        n_samples,
        burn_in,
        thinning,
        seed,
        n_jobs,
    )
    df = pd.DataFrame(rows)
    scan_summary = _mcmc_dataframe_summary(df)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    summary_seed = {
        "scenario": _pd_scenario_label(framework, group),
        "n_samples": n_samples,
        "vary": sorted(vary),
        "seed": int(seed),
        "initial_MGUT": float(MGUT),
        "initial_gGUT": float(gGUT),
        "initial_matrix_digest": _matrix_digest(matrices),
    }
    digest = _cache_key("full_matrix_mcmc", summary_seed)[:8]
    scan_filepath = str(MCMC_RESULTS_DIR / f"full_matrix_mcmc_{stamp}_{digest}.h5")
    df.to_hdf(scan_filepath, key="results", mode="w")

    summary = {
        **summary_seed,
        "scan_mode": "mcmc",
        "sampler": "metropolis_unitary_matrix_adapter",
        "schema": FULL_MATRICES_SCHEMA,
        "input_convention": input_convention or str(current.get("input_convention", "")),
        "matrix_parameterization": "left-multiplied small random unitary steps for UL/UR/DL/DR/EL/ER/NL",
        "matrix_names": list(SO10_MATRIX_INPUT_NAMES),
        "n_retained": int(len(df)),
        "n_requested": int(n_samples),
        "n_proposals": int(proposals),
        "n_invalid_proposals": int(invalid_proposals),
        "n_accepted_moves": int(accepted_moves),
        "n_jobs": int(n_chains),
        "n_chains": int(n_chains),
        "chain_sample_counts": [int(count) for count in chain_sample_counts],
        "acceptance_rate": float(accepted_moves / max(proposals, 1)),
        **scan_summary,
        "matrix_width": float(matrix_width),
        "mgut_log_width": float(mgut_log_width),
        "ggut_width": float(ggut_width),
        "burn_in": int(burn_in),
        "thinning": int(thinning),
        "scan_filepath": scan_filepath,
    }
    artifact = _register_artifact(
        "mcmc-scan",
        payload={"scan_filepath": scan_filepath, "config": summary_seed, "summary": summary},
        summary=summary,
    )
    return {**summary, "artifact_id": artifact["artifact_id"]}


def _format_mcmc_result(result: dict) -> str:
    if "error" in result:
        return result["error"]
    lines = [
        f"MCMC completed: {result['scenario']} ({result['schema']})",
        f"Sampler: {result['sampler']}; varied: {', '.join(result['vary'])}",
        f"Samples retained: {result['n_retained']} / {result['n_requested']}; proposals: {result['n_proposals']}",
        f"Invalid proposals assigned chi_sq={INVALID_MCMC_CHI_SQ:.1e}: {result.get('n_invalid_proposals', 0)}",
        f"Parallel chains: {result.get('n_chains', 1)} workers; chain sample counts: {result.get('chain_sample_counts', [result.get('n_retained', '?')])}",
        f"Sampler acceptance rate: {result['acceptance_rate']:.6g}",
        f"Retained-sample status: allowed = {result.get('n_allowed', '?')}, excluded = {result.get('n_excluded', '?')}, fraction allowed = {result['fraction_allowed']:.6g}",
        f"chi_sq: initial {result['initial_chi_sq']:.6g}, best {result['best_chi_sq']:.6g}, range [{result['min_chi_sq']:.6g}, {result['max_chi_sq']:.6g}]",
        f"Best chi_sq sample = {result.get('best_chi_sq_scan_index', '?')} (allowed = {result.get('best_chi_sq_sample_allowed', '?')})",
    ]
    if result.get("initial_MGUT") is not None and result.get("initial_gGUT") is not None:
        lines.append(
            f"Initial/fixed gauge inputs: MGUT = {float(result['initial_MGUT']):.15g} GeV; "
            f"gGUT = {float(result['initial_gGUT']):.15g}"
        )
    if result.get("fixed_parameters"):
        lines.append(f"Held fixed: {', '.join(result['fixed_parameters'])}")
    if result.get("baseline_input_file"):
        lines.append(f"Initial flavor point: {result['baseline_input_file']}")
    if result.get("input_convention"):
        lines.append(f"Input convention: {result['input_convention']}")
    if result.get("best_chi_sq_most_constraining_channel"):
        lines.append(
            f"Best chi_sq most-constraining channel: {result['best_chi_sq_most_constraining_channel']} "
            f"with bound/lifetime = {result['best_chi_sq_most_constraining_ratio']:.6g}"
        )
    lines.extend([
        f"Smallest lifetime sample = {result['minimum_lifetime_scan_index']} "
        f"(allowed = {result.get('minimum_lifetime_sample_allowed', '?')}, chi_sq = {result.get('minimum_lifetime_chi_sq', np.nan):.6g})",
        f"Smallest lifetime = {result['minimum_lifetime_years']:.6e} yr in {result['minimum_lifetime_channel']}",
        f"Saved MCMC table: {result['scan_filepath']}",
        f"Artifact: {result['artifact_id']}",
    ])
    return "\n".join(lines)


def _is_saved_plot_only_request(text: str) -> bool:
    """A reference to MCMC/scan data in a plot request does not authorize sampling."""
    lowered = " ".join(unicodedata.normalize("NFKC", text or "").translate(_DASH_TRANSLATION).lower().split())
    if not re.search(r"\b(?:plots?|plotting|graph|figure|visuali[sz]e|draw|histogram|scatter|contour|corner)\b", lowered):
        return False
    execution_verbs = r"(?:run|rerun|re-run|perform|start|launch|execute|repeat|redo|continue|resume|extend|create)"
    scan_kind = r"(?:mcmc|monte carlo|(?:random |grid |parameter )?scan|sampling)"
    lowered = re.sub(
        rf"\b(?:do not|don't|never|without|no need to)\s+{execution_verbs}\b.{{0,80}}?\b{scan_kind}\b",
        " ", lowered,
    )
    lowered = re.sub(rf"\b(?:previously|already|just|last)\s+{execution_verbs}\b", "saved", lowered)
    # A verb acting on a plot/analysis is not a verb acting on a scan.
    new_scan = re.search(
        rf"\b{execution_verbs}\b(?:(?!\b(?:plots?|plotting|graph|figure|analysis|correlation|results|data)\b).){{0,60}}\b{scan_kind}\b(?!\s+(?:plots?|graph|figure|results|data|analysis)\b)"
        rf"|\b(?:make|do)\s+(?:(?:a|an|the|another|new)\s+)*{scan_kind}\b"
        r"|\b(?:sample|generate|collect)\s+(?:\d+\s+)?(?:new\s+)?(?:points|samples)\b",
        lowered,
    )
    return new_scan is None


def _parse_mcmc_request(text: str) -> dict | None:
    lowered = " ".join(text.lower().split())
    if "mcmc" not in lowered and "monte carlo" not in lowered:
        return None
    if _is_saved_plot_only_request(text):
        return None

    n_samples = 500
    for pattern in [
        r"(?:sample size|n_samples|samples?|points?)\s*(?:of|=|:)?\s*(\d+)",
        r"(\d+)\s*[- ]?\s*(?:samples?|points?)",
    ]:
        match = re.search(pattern, lowered)
        if match:
            n_samples = int(match.group(1))
            break

    seed_match = re.search(r"seed\s*(?:=|:)?\s*(\d+)", lowered)
    burn_in_match = re.search(r"burn[-_ ]?in\s*(?:=|:)?\s*(\d+)", lowered)
    thinning_match = re.search(r"thinning\s*(?:=|:)?\s*(\d+)", lowered)
    n_jobs_match = re.search(
        r"(?:n_jobs|jobs?|workers?|cores?)\s*(?:=|:)?\s*(-?\d+)", lowered
    ) or re.search(r"(-?\d+)\s*(?:jobs?|workers?|cores?)\b", lowered)
    angle_width = _parse_scalar_assignment(text, (
        "angle_width", "angle width", "flavor_angle_width", "flavor angle width",
        "flavor angle variation",
    ))
    phase_width = _parse_scalar_assignment(text, (
        "phase_width", "phase width", "flavor_phase_width", "flavor phase width",
        "flavor phase variation",
    ))
    matrix_width = _parse_scalar_assignment(text, (
        "matrix_width", "matrix width", "flavor_matrix_width", "flavor matrix width",
        "flavor matrix variation",
    ))
    flavor_width = _parse_scalar_assignment(text, (
        "flavor variation", "flavour variation", "flavor parameter variation",
        "flavour parameter variation", "flavor mixing variation",
        "flavour mixing variation", "vary flavor by", "vary flavour by",
    ))
    if flavor_width is None:
        flavor_width_match = re.search(
            rf"\bvar(?:y|ying)\b.{{0,80}}?\bflavo(?:u)?r\b.{{0,80}}?\bby\b\s*({_NUMBER_RE})",
            text,
            flags=re.IGNORECASE,
        )
        if flavor_width_match:
            flavor_width = _parse_numeric_token(flavor_width_match.group(1))
    mgut_log_width = _parse_scalar_assignment(text, (
        "mgut_log_width", "mgut log width", "mgut log variation",
        "MGUT variation", "log_width", "log width",
    ))
    ggut_width = _parse_scalar_assignment(text, (
        "ggut_width", "ggut width", "gGUT variation", "gGUT proposal width",
    ))
    susy_linear_width = _parse_scalar_assignment(text, (
        "linear_width", "linear width", "linear variation",
        "scalar linear width", "angular variation",
    ))
    varied_parameter_width_match = re.search(
        rf"\bvar(?:y|ying|ied)\b.{{0,180}}?\bby\b\s*(?:a\s+)?({_NUMBER_RE})",
        text,
        flags=re.IGNORECASE,
    )
    varied_parameter_width = (
        _parse_numeric_token(varied_parameter_width_match.group(1))
        if varied_parameter_width_match else None
    )

    mentions_mgut = bool(re.search(r"\b(?:mgut|m_gut|gut scale)\b", lowered))
    mentions_ggut = bool(re.search(r"\b(?:ggut|g_gut|gauge coupling|gut coupling)\b", lowered))
    mentions_flavor = any(token in lowered for token in ["flavor", "flavour", "56", "angle", "phase", "master"])
    only = bool(re.search(r"\bonly\b|\bjust\b", lowered))
    all_parameters = bool(re.search(r"\ball\s+(?:parameters|58)\b|\ball\s+58\b", lowered))

    if all_parameters:
        vary = {"flavor", "MGUT", "gGUT"}
    elif only and (mentions_mgut or mentions_ggut) and not mentions_flavor:
        vary = set()
    else:
        vary = {"flavor"}

    if mentions_mgut:
        vary.add("MGUT")
    if mentions_ggut:
        vary.add("gGUT")
    fixed_clause_match = re.search(
        r"\b(?:fix|fixed)\b(?P<body>.{0,240}?)(?=\b(?:vary|perform|run|start|launch)\b|$)",
        lowered,
    )
    fixed_clause = fixed_clause_match.group("body") if fixed_clause_match else ""
    keep_fixed_match = re.search(
        r"\b(?:keep|keeping|kept|hold|holding|held|leave|leaving)\b"
        r"(?P<body>.{0,200}?)\bfixed\b",
        lowered,
    )
    if keep_fixed_match:
        fixed_clause = f"{fixed_clause} {keep_fixed_match.group('body')}"
    if (
        "fix mgut" in lowered or "fixed mgut" in lowered
        or re.search(r"\b(?:mgut|m_gut|gut scale)\b", fixed_clause)
    ):
        vary.discard("MGUT")
    if (
        "fix ggut" in lowered or "fixed ggut" in lowered
        or re.search(r"\b(?:ggut|g_gut|gut coupling)\b", fixed_clause)
    ):
        vary.discard("gGUT")
    if not vary:
        vary = {"flavor"}

    file_request = _parse_benchmark_file_request(text)

    return {
        "n_samples": n_samples,
        "seed": int(seed_match.group(1)) if seed_match else 24680,
        "burn_in": int(burn_in_match.group(1)) if burn_in_match else 0,
        "thinning": int(thinning_match.group(1)) if thinning_match else 1,
        "n_jobs": int(n_jobs_match.group(1)) if n_jobs_match else -1,
        "vary": vary,
        "filepath": file_request["filepath"] if file_request else None,
        "angle_width": angle_width if angle_width is not None else (flavor_width if flavor_width is not None else 2.0e-4),
        "phase_width": phase_width if phase_width is not None else (flavor_width if flavor_width is not None else 2.0e-4),
        "matrix_width": matrix_width if matrix_width is not None else (flavor_width if flavor_width is not None else 2.0e-4),
        "mgut_log_width": mgut_log_width if mgut_log_width is not None else 2.0e-3,
        "ggut_width": ggut_width if ggut_width is not None else 1.0e-3,
        "susy_log_width": (
            varied_parameter_width
            if varied_parameter_width is not None
            else (mgut_log_width if mgut_log_width is not None else 2.0e-3)
        ),
        "susy_linear_width": susy_linear_width if susy_linear_width is not None else 1.0e-3,
    }


def _explicit_nonsusy_mcmc_spec(text: str) -> dict | None:
    """Ground an unambiguous direct non-SUSY MCMC request without LLM arithmetic."""
    cfg = _parse_mcmc_request(text)
    if cfg is None:
        return None
    lowered = " ".join(text.lower().replace("-", " ").split())
    if not re.search(r"\bnon\s*susy\b|\bnon\s*supersymmetric\b", lowered):
        return None
    if re.search(r"\b(?:su\s*\(?5\)?|su5)(?!\w)", lowered):
        group = "SU5"
    elif re.search(r"\b(?:so\s*\(?10\)?|so10)(?!\w)", lowered):
        group = "SO10"
    else:
        return None

    mgut = _parse_scalar_assignment(text, ("MGUT", "M_GUT", "gut scale", "gut_scale"))
    ggut = _parse_scalar_assignment(text, ("gGUT", "g_GUT", "gut coupling", "coupling"))
    args = {
        "group": group,
        "n_samples": int(cfg["n_samples"]),
        "vary": sorted(cfg["vary"]),
        "seed": int(cfg["seed"]),
        "angle_width": float(cfg["angle_width"]),
        "phase_width": float(cfg["phase_width"]),
        "matrix_width": float(cfg["matrix_width"]),
        "mgut_log_width": float(cfg["mgut_log_width"]),
        "ggut_width": float(cfg["ggut_width"]),
        "burn_in": int(cfg["burn_in"]),
        "thinning": int(cfg["thinning"]),
        "n_jobs": int(cfg["n_jobs"]),
    }
    if mgut is not None:
        args["MGUT"] = float(mgut)
    if ggut is not None:
        args["gGUT"] = float(ggut)
    if cfg.get("filepath"):
        args["filepath"] = cfg["filepath"]
    return args


def _explicit_susy_mcmc_spec(text: str) -> dict | None:
    """Ground an unambiguous direct SUSY MCMC request without LLM arithmetic."""
    cfg = _parse_mcmc_request(text)
    if cfg is None:
        return None
    lowered = " ".join(text.lower().replace("-", " ").split())
    if re.search(r"\bnon\s*susy\b|\bnon\s*supersymmetric\b", lowered):
        return None
    if not re.search(r"\bsusy\b|\bsupersymmetric\b", lowered):
        return None
    if re.search(r"\b(?:su\s*\(?5\)?|su5)(?!\w)", lowered):
        group = "SU5"
    elif re.search(r"\b(?:so\s*\(?10\)?|so10)(?!\w)", lowered):
        group = "SO10"
    else:
        return None

    args = {
        "group": group,
        "n_samples": int(cfg["n_samples"]),
        "vary": _parse_susy_mcmc_vary(text, group),
        "seed": int(cfg["seed"]),
        "log_width": float(cfg["susy_log_width"]),
        "linear_width": float(cfg["susy_linear_width"]),
        "matrix_width": float(cfg["matrix_width"]),
        "phase_width": float(cfg["phase_width"]),
        "burn_in": int(cfg["burn_in"]),
        "thinning": int(cfg["thinning"]),
        "n_jobs": int(cfg["n_jobs"]),
    }
    if cfg.get("filepath"):
        args["filepath"] = cfg["filepath"]
    return args



def _normalize_susy_key(key: object) -> str:
    normalized = str(key).strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "MT": "M_T",
        "M_T": "M_T",
        "Mhalf": "M_half",
        "M_half": "M_half",
        "M1_2": "M_half",
        "Mg": "M_half",
        "M_g": "M_half",
        "M_1": "M1", "m1": "M1",
        "M_2": "M2", "m2": "M2",
        "M_3": "M3", "m3": "M3",
        "M_wino": "mtildeW",
        "m_wino": "mtildeW",
        "mwino": "mtildeW",
        "mtildeW": "mtildeW",
        "m_sfermion": "mtildeq",
        "M_sfermion": "mtildeq",
        "msfermion": "mtildeq",
        "mtildeq": "mtildeq",
        "fL": "fE",
        "fLepton": "fE",
        "phi_13": "phi13",
        "Phi13": "phi13",
        "phi_23": "phi23",
        "Phi23": "phi23",
    }
    return aliases.get(normalized, normalized)


def _read_susy_benchmark_dict(filepath: str | os.PathLike) -> tuple[Path, dict]:
    path = _resolve_input_file_path(filepath)
    cleaned = _strip_hash_comments(path.read_text(encoding="utf-8")).strip()
    if not cleaned:
        raise ValueError(f"Input file is empty after removing comments: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(cleaned)
    else:
        payload = ast.literal_eval(cleaned)
    if not isinstance(payload, dict):
        raise ValueError(f"SUSY benchmark file must contain a dictionary: {path}")
    return path, payload


def _normalize_susy_schema(schema: object, group: str) -> str:
    normalized = str(schema).strip().lower().replace("-", "_").replace(" ", "_").replace("(", "").replace(")", "")
    aliases = {
        "susy_su5": SUSY_SU5_SCHEMA,
        "su5_susy": SUSY_SU5_SCHEMA,
        "susy_su_5": SUSY_SU5_SCHEMA,
        "susy_so10": SUSY_SO10_SCHEMA,
        "so10_susy": SUSY_SO10_SCHEMA,
        "susy_so_10": SUSY_SO10_SCHEMA,
    }
    return aliases.get(normalized, SUSY_SU5_SCHEMA if group == "SU5" else SUSY_SO10_SCHEMA)


def _load_susy_benchmark_parameters_from_file(filepath: str | os.PathLike, group: str) -> tuple[dict, dict]:
    path, payload = _read_susy_benchmark_dict(filepath)
    schema = _normalize_susy_schema(payload.get("schema", ""), group)
    expected_schema = SUSY_SU5_SCHEMA if group == "SU5" else SUSY_SO10_SCHEMA
    if schema != expected_schema:
        raise ValueError(f"SUSY {group} expected schema {expected_schema!r}, got {schema!r}.")

    params = _default_susy_su5_inputs() if group == "SU5" else _default_susy_so10_inputs()
    matrix_names = set(SUSY_SU5_MATRIX_INPUTS if group == "SU5" else SUSY_SO10_MATRIX_INPUTS)
    provided = []
    for raw_key, raw_value in payload.items():
        key = _normalize_susy_key(raw_key)
        if key == "schema":
            continue
        if key in matrix_names:
            params[key] = _parse_matrix_payload(raw_value, key)
        elif group == "SO10" and key in {"x", "y"} and isinstance(raw_value, dict):
            params[key] = dict(params[key])
            allowed = set(SUSY_SO10_X_INPUTS if key == "x" else SUSY_SO10_Y_INPUTS)
            for entry_key, entry_value in raw_value.items():
                entry_name = _normalize_susy_key(entry_key)
                if entry_name not in allowed:
                    raise ValueError(f"Unknown SUSY {group} triplet-mixing input {entry_key!r} in {path}.")
                params[key][entry_name] = float(entry_value)
                provided.append(entry_name)
            continue
        elif group == "SO10" and key in SUSY_SO10_TRIPLET_INPUTS:
            container = "x" if key.startswith("x") else "y"
            params[container] = dict(params[container])
            params[container][key] = float(raw_value)
        elif group == "SU5" and key in {"M1", "M2", "M3"}:
            params[key] = float(raw_value)
        elif key in params:
            params[key] = float(raw_value)
        elif key in {"g2", "g3", "A_L", "A_S"}:
            params[key] = float(raw_value)
        else:
            raise ValueError(f"Unknown SUSY {group} input {raw_key!r} in {path}.")
        provided.append(key)

    return params, {
        "input_file": str(path),
        "schema": schema,
        "provided_inputs": provided,
    }


def _compute_susy_point_from_file(framework: str, group: str, filepath: str | os.PathLike) -> dict:
    if framework != "SUSY":
        raise ValueError("SUSY benchmark loader called for a non-SUSY scenario.")
    params, metadata = _load_susy_benchmark_parameters_from_file(filepath, group)
    result = _compute_susy_su5_point(params) if group == "SU5" else _compute_susy_so10_point(params)
    result["schema"] = metadata["schema"]
    result["input_file"] = metadata["input_file"]
    result["point_kind"] = f"file benchmark ({metadata['schema']})"
    result["provided_inputs"] = [_susy_mcmc_display_name(name) for name in metadata["provided_inputs"]]
    canonical_inputs = SUSY_SU5_INPUTS if group == "SU5" else SUSY_SO10_INPUTS
    result["defaulted_inputs"] = [
        _susy_mcmc_display_name(name)
        for name in canonical_inputs
        if name not in metadata["provided_inputs"]
    ]
    return result


def _susy_mcmc_default_file(group: str) -> str:
    return SUSY_SU5_DEFAULT_INPUT_FILE if group == "SU5" else SUSY_SO10_DEFAULT_INPUT_FILE


def _nonsusy_mcmc_default_file(group: str) -> str:
    return NONSUSY_SU5_DEFAULT_INPUT_FILE if group == "SU5" else NONSUSY_SO10_DEFAULT_INPUT_FILE


def _susy_mcmc_parameter_order(group: str) -> list[str]:
    if group == "SU5":
        return ["m0", "AX", "tanbeta", "M_T", "mu", "M_half", "M1", "M2", "M3", "phi13", "phi23"]
    return [
        "M_T", "mtildeW", "mu", "mtildeq", "tanbeta",
        "theta_u", "phi_u", "psi_u", "theta_d", "phi_d", "psi_d",
        "vL", "vR", "h", "f", "g",
        *SUSY_SO10_TRIPLET_INPUTS,
    ]


def _susy_mcmc_default_vary(group: str) -> list[str]:
    # Preserve the common-mass default. Independent masses vary when requested.
    return [name for name in _susy_mcmc_parameter_order(group)
            if group != "SU5" or name not in {"M1", "M2", "M3"}]


def _susy_mcmc_display_name(name: str) -> str:
    return {"mtildeW": "M_wino", "mtildeq": "m_sfermion"}.get(name, name)


def _parse_susy_mcmc_vary(text: str, group: str) -> list[str]:
    default_order = _susy_mcmc_parameter_order(group)
    default_vary = _susy_mcmc_default_vary(group)
    lowered = " ".join(text.lower().replace("-", "_").split())
    aliases = {
        "mt": "M_T",
        "m_t": "M_T",
        "triplet mass": "M_T",
        "color triplet mass": "M_T",
        "colour triplet mass": "M_T",
        "colored triplet mass": "M_T",
        "coloured triplet mass": "M_T",
        "m half": "M_half",
        "m_half": "M_half",
        "mhalf": "M_half",
        "m_1": "M1", "m_2": "M2", "m_3": "M3",
        "mg": "M_half",
        "ax": "AX",
        "phi_13": "phi13",
        "phi 13": "phi13",
        "phase13": "phi13",
        "phase 13": "phi13",
        "phi_23": "phi23",
        "phi 23": "phi23",
        "phase23": "phi23",
        "phase 23": "phi23",
        "tan beta": "tanbeta",
        "tan_beta": "tanbeta",
        "m_wino": "mtildeW",
        "mwino": "mtildeW",
        "wino": "mtildeW",
        "m_sfermion": "mtildeq",
        "msfermion": "mtildeq",
        "sfermion": "mtildeq",
        "vl": "vL",
        "vr": "vR",
    }
    if group == "SU5":
        aliases.update({
            "susy scale": "m0",
            "susy_scale": "m0",
            "supersymmetry scale": "m0",
            "supersymmetry_scale": "m0",
            "superpartner scale": "m0",
            "superpartner_scale": "m0",
        })
    else:
        aliases.update({
            "susy scale": "mtildeq",
            "susy_scale": "mtildeq",
            "supersymmetry scale": "mtildeq",
            "supersymmetry_scale": "mtildeq",
            "superpartner scale": "mtildeq",
            "superpartner_scale": "mtildeq",
            "sfermion mass": "mtildeq",
            "sfermion_mass": "mtildeq",
        })
    lookup = {name.lower(): name for name in default_order}
    lookup.update(aliases)
    mentioned = []
    if group == "SU5" and re.search(r"\bphases?\b", lowered):
        mentioned.extend(["phi13", "phi23"])
    for token, canonical in sorted(lookup.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(token)}\b", lowered) and canonical not in mentioned and canonical in default_order:
            mentioned.append(canonical)
    explicit_vary_clause = bool(re.search(r"\bvar(?:y|ying|ied)\b", lowered))
    only_specific = bool(re.search(r"\b(?:only|just)\b", lowered)) and mentioned
    if mentioned and (only_specific or explicit_vary_clause):
        selected = mentioned
    else:
        selected = list(default_vary)
    for token, canonical in sorted(lookup.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\bfix(?:ed)?\s+{re.escape(token)}\b", lowered) and canonical in selected:
            selected.remove(canonical)
    return selected or list(default_vary)


def _susy_scalar_bounds(group: str) -> dict[str, tuple[float, float]]:
    if group == "SU5":
        return {
            "m0": (100.0, 1.0e8),
            "M_half": (10.0, 1.0e8),
            "M1": (10.0, 1.0e8),
            "M2": (10.0, 1.0e8),
            "M3": (10.0, 1.0e8),
            "AX": (-10.0, 10.0),
            "tanbeta": (1.01, 100.0),
            "M_T": (1.0e14, 1.0e20),
            "mu": (10.0, 1.0e8),
            "phi13": (0.0, 2.0 * np.pi),
            "phi23": (0.0, 2.0 * np.pi),
        }
    return {
        "M_T": (1.0e14, 1.0e20),
        "mtildeW": (10.0, 1.0e8),
        "mu": (10.0, 1.0e8),
        "mtildeq": (10.0, 1.0e9),
        "tanbeta": (1.01, 100.0),
        "theta_u": (0.0, np.pi / 2.0),
        "phi_u": (0.0, np.pi / 2.0),
        "psi_u": (0.0, np.pi / 2.0),
        "theta_d": (0.0, np.pi / 2.0),
        "phi_d": (0.0, np.pi / 2.0),
        "psi_d": (0.0, np.pi / 2.0),
        "vL": (1.0e-18, 1.0e-3),
        "vR": (1.0e8, 1.0e18),
        **{name: (-1.0, 1.0) for name in SUSY_SO10_TRIPLET_INPUTS},
    }


def _perturb_susy_scalar(
    name: str,
    value: float,
    rng: np.random.Generator,
    log_width: float,
    linear_width: float,
    bounds: dict[str, tuple[float, float]],
    phase_width: float | None = None,
) -> float:
    lower, upper = bounds[name]
    if name in {"phi13", "phi23"}:
        width = linear_width if phase_width is None else phase_width
        span = upper - lower
        proposed = float(value) + rng.normal(0.0, width)
        return float(lower + ((proposed - lower) % span))
    positive_log_names = {"m0", "M_half", "M1", "M2", "M3", "M_T", "mu", "mtildeW", "mtildeq", "tanbeta", "vL", "vR"}
    if name in positive_log_names and value > 0.0 and lower > 0.0:
        proposed = 10.0 ** _reflect_into_range(np.log10(value) + rng.normal(0.0, log_width), np.log10(lower), np.log10(upper))
        return float(proposed)
    scale = max(abs(float(value)), 1.0)
    return _reflect_into_range(float(value) + rng.normal(0.0, linear_width * scale), lower, upper)


def _perturb_symmetric_matrix(matrix: np.ndarray, rng: np.random.Generator, width: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=complex).copy()
    scale = max(float(np.nanmax(np.abs(matrix))), 1.0)
    step = rng.normal(0.0, width * scale, size=(3, 3)) + 1j * rng.normal(0.0, width * scale, size=(3, 3))
    step = 0.5 * (step + step.T)
    return matrix + step


def _perturb_antisymmetric_matrix(matrix: np.ndarray, rng: np.random.Generator, width: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=complex).copy()
    scale = max(float(np.nanmax(np.abs(matrix))), 1.0)
    step = rng.normal(0.0, width * scale, size=(3, 3)) + 1j * rng.normal(0.0, width * scale, size=(3, 3))
    step = 0.5 * (step - step.T)
    np.fill_diagonal(step, 0.0)
    return matrix + step


def _perturb_susy_h_matrix(matrix: np.ndarray, rng: np.random.Generator, width: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=complex)
    proposed = np.zeros((3, 3), dtype=complex)
    diag = np.real(np.diag(matrix))
    scale = max(float(np.nanmax(np.abs(diag))), 1.0)
    proposed_diag = np.maximum(diag + rng.normal(0.0, width * scale, size=3), 0.0)
    np.fill_diagonal(proposed, proposed_diag)
    return proposed


def _proposal_for_susy_state(group: str, params: dict, rng: np.random.Generator, vary: list[str], log_width: float, linear_width: float, matrix_width: float, phase_width: float | None = None) -> dict:
    proposed = dict(params)
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            proposed[key] = value.copy()
        elif isinstance(value, dict):
            proposed[key] = dict(value)
    bounds = _susy_scalar_bounds(group)
    if group == "SU5":
        independent = set(params.get("_independent_gaugino_masses", ()))
        independent.update(set(vary) & {"M1", "M2", "M3"})
        proposed["_independent_gaugino_masses"] = sorted(independent)
    for name in vary:
        if group == "SO10" and name in SUSY_SO10_TRIPLET_INPUTS:
            container = "x" if name.startswith("x") else "y"
            proposed[container][name] = _perturb_susy_scalar(
                name,
                float(proposed[container].get(name, 0.0)),
                rng,
                log_width,
                linear_width,
                bounds,
            )
        elif name in bounds:
            proposed[name] = _perturb_susy_scalar(
                name,
                float(proposed[name]),
                rng,
                log_width,
                linear_width,
                bounds,
                phase_width=phase_width,
            )
        elif group == "SO10" and name == "h":
            proposed[name] = _perturb_susy_h_matrix(proposed[name], rng, matrix_width)
        elif group == "SO10" and name == "f":
            proposed[name] = _perturb_symmetric_matrix(proposed[name], rng, matrix_width)
        elif group == "SO10" and name == "g":
            proposed[name] = _perturb_antisymmetric_matrix(proposed[name], rng, matrix_width)
    return proposed


def _susy_mcmc_state(group: str, params: dict) -> dict:
    result = _compute_susy_su5_point(params) if group == "SU5" else _compute_susy_so10_point(params)
    canonical_lifetimes = _canonical_lifetimes_for_bounds(result)
    chi_sq_terms = _master56_chi_sq_terms(canonical_lifetimes)
    return {
        "params": result["inputs"],
        "result": result,
        "lifetimes_years": result["lifetimes_years"],
        "widths": result["widths"],
        "branching_ratios": result.get("branching_ratios", {}),
        "total_width_GeV": float(result["total_width_GeV"]),
        "classification": classify_model(canonical_lifetimes) if canonical_lifetimes else "unknown",
        "chi_sq": _master56_chi_sq(canonical_lifetimes),
        "chi_sq_terms": chi_sq_terms,
    }


def _susy_mcmc_row(group: str, schema: str, vary: list[str], input_file: str):
    def row_fn(sample_index: int, step: int, accepted_moves: int, proposals: int, state: dict) -> dict:
        canonical_lifetimes = _canonical_lifetimes_for_bounds(state["result"])
        min_channel, min_lifetime = min(canonical_lifetimes.items(), key=lambda item: item[1]) if canonical_lifetimes else ("unknown", np.nan)
        row = {
            "scan_schema_version": SCAN_SCHEMA_VERSION,
            "scan_index": int(sample_index),
            "scenario": f"SUSY {group}",
            "schema": schema,
            "input_file": input_file,
            "mcmc_step": int(step),
            "mcmc_accepted_moves": int(accepted_moves),
            "mcmc_acceptance_rate": float(accepted_moves / max(proposals, 1)) if proposals else np.nan,
            "mcmc_allowed": bool(state["chi_sq"] < 1.0),
            "classification": state["classification"],
            "chi_sq": float(state["chi_sq"]),
            "total_width_GeV": float(state["total_width_GeV"]),
            "min_lifetime_years": float(min_lifetime),
            "min_lifetime_channel": str(min_channel),
        }
        for name in _susy_mcmc_parameter_order(group):
            if group == "SO10" and name in SUSY_SO10_TRIPLET_INPUTS:
                container = "x" if name.startswith("x") else "y"
                value = state["params"].get(container, {}).get(name)
            else:
                value = state["params"].get(name)
            column_name = _susy_mcmc_display_name(name)
            if isinstance(value, np.ndarray):
                _flatten_scan_matrix(row, column_name, value)
            elif value is not None and np.isscalar(value):
                row[column_name] = float(value)
        for channel, lifetime in canonical_lifetimes.items():
            row[_channel_column(channel, "lifetime")] = float(lifetime)
            row[f"log10_{_channel_column(channel, 'lifetime')}"] = float(np.log10(max(float(lifetime), 1.0e-300)))
        for channel, width in state["widths"].items():
            canonical = _canonical_decay_channel(channel)
            row[_channel_column(canonical, "width")] = float(width)
        for channel, branching in state["branching_ratios"].items():
            canonical = _canonical_decay_channel(channel)
            row[_channel_column(canonical, "branching")] = float(branching)
        for channel, term in state["chi_sq_terms"].items():
            row[_channel_column(channel, "chi_sq")] = float(term)
        return row
    return row_fn


def _run_susy_mcmc(
    group: str,
    initial_params: dict,
    input_file: str,
    n_samples: int = 500,
    vary: list[str] | None = None,
    seed: int = 24680,
    log_width: float = 2.0e-3,
    linear_width: float = 1.0e-3,
    matrix_width: float = 2.0e-4,
    phase_width: float = 2.0e-4,
    burn_in: int = 0,
    thinning: int = 1,
    n_jobs: int = -1,
) -> dict:
    n_samples = max(1, min(int(n_samples), 200000))
    burn_in = max(0, int(burn_in))
    thinning = max(1, int(thinning))
    vary = list(vary or _susy_mcmc_default_vary(group))
    schema = SUSY_SU5_SCHEMA if group == "SU5" else SUSY_SO10_SCHEMA
    initial_state = _susy_mcmc_state(group, initial_params)

    def propose(current: dict, step_rng: np.random.Generator) -> dict:
        proposed_params = _proposal_for_susy_state(group, current["params"], step_rng, vary, log_width, linear_width, matrix_width, phase_width)
        return _susy_mcmc_state(group, proposed_params)

    rows, accepted_moves, proposals, invalid_proposals, current, best, n_chains, chain_sample_counts = _run_parallel_metropolis_mcmc(
        initial_state,
        propose,
        _susy_mcmc_row(group, schema, vary, input_file),
        n_samples,
        burn_in,
        thinning,
        seed,
        n_jobs,
    )
    df = pd.DataFrame(rows)
    scan_summary = _mcmc_dataframe_summary(df)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    summary_seed = {
        "scenario": f"SUSY {group}",
        "schema": schema,
        "n_samples": n_samples,
        "vary": [_susy_mcmc_display_name(name) for name in vary],
        "seed": int(seed),
        "input_file": str(input_file),
    }
    digest = _cache_key(f"{schema}_mcmc", summary_seed)[:8]
    scan_filepath = str(MCMC_RESULTS_DIR / f"{schema}_mcmc_{stamp}_{digest}.h5")
    df.to_hdf(scan_filepath, key="results", mode="w")
    summary = {
        **summary_seed,
        "scan_mode": "mcmc",
        "sampler": "metropolis_susy_parameter_perturbation",
        "n_retained": int(len(df)),
        "n_requested": int(n_samples),
        "n_proposals": int(proposals),
        "n_invalid_proposals": int(invalid_proposals),
        "n_accepted_moves": int(accepted_moves),
        "n_jobs": int(n_chains),
        "n_chains": int(n_chains),
        "chain_sample_counts": [int(count) for count in chain_sample_counts],
        "acceptance_rate": float(accepted_moves / max(proposals, 1)),
        **scan_summary,
        "log_width": float(log_width),
        "linear_width": float(linear_width),
        "matrix_width": float(matrix_width),
        "phase_width": float(phase_width),
        "burn_in": int(burn_in),
        "thinning": int(thinning),
        "scan_filepath": scan_filepath,
    }
    artifact = _register_artifact(
        "mcmc-scan",
        payload={"scan_filepath": scan_filepath, "config": summary_seed, "summary": summary},
        summary=summary,
    )
    return {**summary, "artifact_id": artifact["artifact_id"]}


def _parse_benchmark_file_request(text: str) -> dict | None:
    matches = re.finditer(
        r"(?P<path>(?:~|\.{1,2}|/)?[A-Za-z0-9_.\-/]+(?:\.txt|\.json))",
        text,
    )
    matrix_file = _matrix_elements_filepath_from_text(text)
    bounds_file = _bounds_filepath_from_text(text)
    match = next((item for item in matches if item.group("path") not in {matrix_file, bounds_file}), None)
    if not match:
        return None
    mgut = _parse_scalar_assignment(text, ("MGUT", "M_GUT", "gut scale", "gut_scale"))
    ggut = _parse_scalar_assignment(text, ("gGUT", "g_GUT", "gut coupling", "coupling"))
    return {"filepath": match.group("path"), "MGUT": mgut, "gGUT": ggut}


def _parse_scalar_assignment(text: str, names: tuple[str, ...]) -> float | None:
    name_pattern = "|".join(re.escape(name) for name in names)
    match = re.search(rf"(?:{name_pattern})\s*(?:=|:)?\s*({_NUMBER_RE})", text, flags=re.IGNORECASE)
    if not match:
        return None
    return _parse_numeric_token(match.group(1))


SUSY_INPUT_LABELS = {
    "m0": "m0 (GeV)",
    "M_half": "M_half (GeV)",
    "M1": "M1 (GeV)", "M2": "M2 (GeV)", "M3": "M3 (GeV)",
    "mu": "mu (GeV)",
    "M_T": "M_T (GeV)",
    "mtildeW": "mtildeW (GeV)",
    "mtildeq": "mtildeq (GeV)",
    "AX": "AX (dimensionless)",
    "tanbeta": "tanbeta",
    "theta_u": "theta_u",
    "theta_d": "theta_d",
    "phi_u": "phi_u",
    "phi_d": "phi_d",
    "psi_u": "psi_u",
    "psi_d": "psi_d",
}


def _compute_non_susy_pd_from_file(
    framework: str,
    group: str,
    filepath: str | os.PathLike,
    MGUT: float | None = None,
    gGUT: float | None = None,
) -> dict:
    params, metadata = _load_benchmark_parameters_from_file(filepath, framework, group, MGUT=MGUT, gGUT=gGUT)
    result = _compute_nonsusy_decay_rates(
        group,
        MGUT=params["MGUT"],
        gGUT=params["gGUT"],
        UL=params["UL"],
        UR=params["UR"],
        DL=params["DL"],
        DR=params["DR"],
        EL=params["EL"],
        ER=params["ER"],
        NL=params["NL"],
    )
    derived_inputs = list(metadata.get("derived_inputs") or [])
    if metadata["schema"] == MASTER56_SCHEMA:
        derived_inputs = [
            metadata["input_convention"],
            "DL = UL @ Uckm",
            "NL = EL @ Upmns",
        ]
        flavor_mode = "validated Master.nb 56-parameter flavor vector"
        point_kind = f"file benchmark ({metadata['schema']})"
    else:
        flavor_mode = metadata["input_convention"]
        point_kind = "file benchmark (full matrix inputs)"

    output = {
        "scenario": _pd_scenario_label(framework, group),
        "point_kind": point_kind,
        "input_file": metadata["input_file"],
        "schema": metadata["schema"],
        "n_values": metadata["n_values"],
        "n_file_values": metadata["n_file_values"],
        "MGUT": float(params["MGUT"]),
        "gGUT": float(params["gGUT"]),
        "core_matrices": _core_matrices_from_params(params),
        "input_convention": metadata["input_convention"],
        "second_gauge_boson": "absent" if group == "SU5" else "included",
        "classification": classify_model(result["lifetimes_years"]),
        "lifetimes_1e34yr": {ch: float(tau) / 1e34 for ch, tau in result["lifetimes_years"].items()},
        "lifetimes_years": {ch: float(tau) for ch, tau in result["lifetimes_years"].items()},
        "branching_ratios": {ch: float(br) for ch, br in result["branching_ratios"].items()},
        "total_width_GeV": float(result["total_width_GeV"]),
        "provided_inputs": list(metadata.get("provided_matrix_inputs") or ["benchmark file"]),
        "derived_inputs": derived_inputs,
        "flavor_mode": flavor_mode,
    }
    if group == "SO10":
        output["k2_sq"] = float(result["parameters"]["k2_sq"])
    if metadata["schema"] == MASTER56_SCHEMA:
        output["master56_values"] = [float(value) for value in metadata["values"]]
        output["master56_parameter_order"] = list(MASTER56_PARAMETER_ORDER)
    return output


def _format_non_susy_pd_result(result: dict) -> str:
    lines = [
        f"Scenario: {result['scenario']} ({result['point_kind']})",
        f"MGUT = {result['MGUT']:.6e} GeV, gGUT = {result['gGUT']:.6g}",
        f"Second gauge-boson contribution: {result['second_gauge_boson']}",
    ]
    if result.get("second_gauge_boson") == "included":
        lines.append(f"k2_sq = {result['k2_sq']:.6e}")
    if result.get("input_file"):
        lines.append(f"Input file: {result['input_file']}")
    if result.get("schema"):
        n_file_values = result.get("n_file_values", result.get("n_values", "?"))
        lines.append(
            f"Input schema: {result['schema']} "
            f"({n_file_values} file values; {result.get('n_values', '?')} flavor values)"
        )
    if result.get("flavor_mode"):
        lines.append(f"Flavor mode: {result['flavor_mode']}")
    if result.get("provided_inputs"):
        lines.append(f"Provided inputs: {', '.join(result['provided_inputs'])}")
    if result.get("randomized_inputs"):
        lines.append(f"Randomized matrices: {', '.join(result['randomized_inputs'])}")
    if result.get("derived_inputs"):
        lines.append(f"Derived inputs: {', '.join(result['derived_inputs'])}")
    lines.extend([
        f"Classification: {result['classification']}",
        "Lifetimes tau/B [1e34 yr]:",
    ])
    for channel, value in result["lifetimes_1e34yr"].items():
        lines.append(f"  {channel:12s} {value:.6e}")
    lines.append("Branching ratios:")
    for channel, value in result["branching_ratios"].items():
        lines.append(f"  {channel:12s} {value:.6e}")
    lines.append(f"Total width = {result['total_width_GeV']:.6e} GeV")
    return "\n".join(lines)


SUSY_SU5_SCALAR_INPUTS = ["m0", "AX", "tanbeta", "M_T", "mu", "M_half", "M1", "M2", "M3", "phi13", "phi23"]
SUSY_SU5_MATRIX_INPUTS = ["fU", "fD", "fE", "P"]
SUSY_SU5_INPUTS = SUSY_SU5_SCALAR_INPUTS + SUSY_SU5_MATRIX_INPUTS


def _susy_su5_p_from_phases(phi13: float, phi23: float) -> np.ndarray:
    return np.diag([np.exp(1j * float(phi13)), np.exp(1j * float(phi23)), 1.0]).astype(complex)


def _default_susy_su5_inputs() -> dict:
    fU = np.array(
        [
            [6.09e-6, 0.0, 0.0],
            [0.0, 3.04e-3, 0.0],
            [0.0, 0.0, 0.8472],
        ],
        dtype=complex,
    )
    fD_diag = np.diag([1.33e-5, 2.66e-4, 1.373e-2]).astype(complex)
    Vckm = np.array(
        [
            [0.97435, 0.22500, 0.00369],
            [-0.22486, 0.97349, 0.04182],
            [0.00565, -0.04161, 0.99912],
        ],
        dtype=complex,
    )
    return {
        "m0": 3000.0,
        "AX": 0.0,
        "tanbeta": 10.0,
        "M_T": DEFAULT_SUSY_SU5_M_T,
        "mu": 1000.0,
        "M_half": 2000.0,
        "phi13": 0.0,
        "phi23": 0.0,
        "g2": 0.637,
        "g3": 1.048,
        "fU": fU,
        "fD": Vckm.conj() @ fD_diag,
        "fE": np.array(
            [
                [2.828e-6, 0.0, 0.0],
                [0.0, 5.9576e-4, 0.0],
                [0.0, 0.0, 1.0119e-2],
            ],
            dtype=complex,
        ),
        "P": _susy_su5_p_from_phases(0.0, 0.0),
    }

def _compute_susy_su5_point(overrides: dict | None = None) -> dict:
    params = _default_susy_su5_inputs()
    override_values = dict(overrides or {})
    if "Mg" in override_values and "M_half" not in override_values:
        override_values["M_half"] = override_values.pop("Mg")
    params.update(override_values)
    if "P" not in override_values or "phi13" in override_values or "phi23" in override_values:
        params["P"] = _susy_su5_p_from_phases(params["phi13"], params["phi23"])
    m_half = float(params["M_half"])
    independent = set(override_values.get("_independent_gaugino_masses",
        [name for name in ("M1", "M2", "M3") if name in override_values]))
    masses = {}
    for name, factor in (("M1", 1.0), ("M2", 2.0), ("M3", 6.0)):
        masses[name] = float(params[name]) if name in independent else factor * m_half
        if not np.isfinite(masses[name]) or masses[name] <= 0.0:
            raise ValueError(f"{name} must be positive and finite in GeV.")
    M1, M2, M3 = (masses[name] for name in ("M1", "M2", "M3"))
    params["_independent_gaugino_masses"] = sorted(independent)
    result = compute_susy_su5_formula(
        params["m0"],
        params["AX"],
        params["tanbeta"],
        params["M_T"],
        params["mu"],
        M1,
        M2,
        M3,
        params["g2"],
        params["g3"],
        params["fU"],
        params["fD"],
        params["fE"],
        P=params["P"],
    )
    params["M_half"] = m_half
    params["M1"] = M1
    params["M2"] = M2
    params["M3"] = M3
    widths = result.widths
    lifetimes = result.lifetimes
    total_width = float(widths.get("total", 0.0))
    return {
        "scenario": "SUSY SU5",
        "operator_context": dict(SUSY_SU5_OPERATOR_CONTEXT),
        "inputs": params,
        "provided_inputs": sorted((overrides or {}).keys()),
        "defaulted_inputs": [name for name in SUSY_SU5_INPUTS if name not in (overrides or {})],
        "widths": widths,
        "lifetimes_years": lifetimes,
        "branching_ratios": result.branching_ratios,
        "experimental_comparison": result.experimental_comparison,
        "experimental_summary": result.experimental_summary,
        "classification": result.experimental_summary["classification"],
        "total_width_GeV": total_width,
        "chargino_masses": result.gaugino_higgsino_spectrum.chargino_masses,
        "neutralino_masses": result.gaugino_higgsino_spectrum.neutralino_masses,
    }


def _format_susy_su5_result(result: dict) -> str:
    params = result["inputs"]
    lines = [
        "Scenario: SUSY SU5 (interactive point)",
        f"m0 = {params['m0']:.6g} GeV, M_half = {params['M_half']:.6g} GeV, AX = {params['AX']:.6g}",
        f"mu = {params['mu']:.6g} GeV, tanbeta = {params['tanbeta']:.6g}",
        f"M_T = {params['M_T']:.6e} GeV",
    ]
    if result.get("provided_inputs"):
        lines.append(f"Provided inputs: {', '.join(result['provided_inputs'])}")
    if result.get("defaulted_inputs"):
        lines.append(f"Defaulted inputs: {', '.join(result['defaulted_inputs'])}")
    lines.append("Lifetimes tau/B [yr]:")
    for channel, value in result["lifetimes_years"].items():
        if channel == "total":
            continue
        lines.append(f"  {channel:16s} {float(value):.6e}")
    lines.append("Partial widths [GeV]:")
    for channel, value in result["widths"].items():
        lines.append(f"  {channel:16s} {float(value):.6e}")
    lines.append(f"Total width = {result['total_width_GeV']:.6e} GeV")
    return "\n".join(lines)


SUSY_SO10_SCALAR_INPUTS = [
    "M_T", "mtildeW", "mu", "mtildeq", "tanbeta",
    "theta_u", "phi_u", "psi_u", "theta_d", "phi_d", "psi_d",
    "vL", "vR",
]
SUSY_SO10_MATRIX_INPUTS = ["h", "f", "g"]
SUSY_SO10_X_INPUTS = [f"x{i}" for i in range(11)]
SUSY_SO10_Y_INPUTS = ["y5", "y7", "y9", "y10"]
SUSY_SO10_TRIPLET_INPUTS = SUSY_SO10_X_INPUTS + SUSY_SO10_Y_INPUTS
SUSY_SO10_INPUTS = SUSY_SO10_MATRIX_INPUTS + SUSY_SO10_SCALAR_INPUTS + SUSY_SO10_TRIPLET_INPUTS


def _default_susy_so10_inputs() -> dict:
    return {
        "M_T": 1.0e17,
        "mtildeW": 2000.0,
        "mu": 1000.0,
        "mtildeq": 3000.0,
        "g2": 0.6374,
        "A_L": 0.4,
        "A_S": 1.0,
        "tanbeta": 10.0,
        "theta_u": 0.20,
        "phi_u": 0.30,
        "psi_u": 0.40,
        "theta_d": 0.25,
        "phi_d": 0.35,
        "psi_d": 0.45,
        "vL": 1.0e-11,
        "vR": 1.0e15,
        "h": np.diag([0.0, 0.0, 1.6838]).astype(complex),
        "f": np.array(
            [
                [0.00019072, 0.00129401, 0.00749214],
                [0.00129401, 0.01356053, 0.00122855],
                [0.00749214, 0.00122855, 0.05112795],
            ],
            dtype=complex,
        ),
        "g": np.array(
            [
                [0.0, 0.02975586j, 0.02508341j],
                [-0.02975586j, 0.0, 0.11570158j],
                [-0.02508341j, -0.11570158j, 0.0],
            ],
            dtype=complex,
        ),
        "x": {
            "x0": 1.0,
            "x1": 0.0,
            "x2": 0.0,
            "x3": 0.0,
            "x4": 0.0,
            "x5": 0.0,
            "x6": 0.0,
            "x7": 0.0,
            "x8": 0.0,
            "x9": 0.0,
            "x10": 0.0,
        },
        "y": {"y5": 0.0, "y7": 0.0, "y9": 0.0, "y10": 0.0},
    }

def _compute_susy_so10_point(overrides: dict | None = None) -> dict:
    params = _default_susy_so10_inputs()
    params.update(overrides or {})
    vev_parameters = SO10VEVParameters(
        tanbeta=params["tanbeta"],
        theta_u=params["theta_u"],
        phi_u=params["phi_u"],
        psi_u=params["psi_u"],
        theta_d=params["theta_d"],
        phi_d=params["phi_d"],
        psi_d=params["psi_d"],
        vL=params["vL"],
        vR=params["vR"],
    )
    result = compute_susy_so10_from_model_parameters(
        params["h"],
        params["f"],
        params["g"],
        params["x"],
        params["y"],
        vev_parameters,
        M_T=params["M_T"],
        M_wino=params["mtildeW"],
        mu=params["mu"],
        m_sfermion=params["mtildeq"],
        g2=params["g2"],
        A_L=params["A_L"],
        A_S=params["A_S"],
    )
    total_width = float(result.widths.get("total", 0.0))
    return {
        "scenario": "SUSY SO10",
        "operator_context": dict(SUSY_SO10_OPERATOR_CONTEXT),
        "inputs": params,
        "provided_inputs": sorted((overrides or {}).keys()),
        "defaulted_inputs": [name for name in SUSY_SO10_INPUTS if name not in (overrides or {})],
        "widths": result.widths,
        "lifetimes_years": result.lifetimes,
        "branching_ratios": result.branching_ratios,
        "experimental_comparison": result.experimental_comparison,
        "experimental_summary": result.experimental_summary,
        "classification": result.experimental_summary["classification"],
        "total_width_GeV": total_width,
        "fermion_masses": result.fermion_spectrum.masses if result.fermion_spectrum else {},
    }


def _format_susy_so10_result(result: dict) -> str:
    params = result["inputs"]
    lines = [
        "Scenario: SUSY SO10 (interactive point)",
        f"M_wino = {params['mtildeW']:.6g} GeV, mu = {params['mu']:.6g} GeV, m_sfermion = {params['mtildeq']:.6g} GeV",
        f"tanbeta = {params['tanbeta']:.6g}, vL = {params['vL']:.6e} GeV, vR = {params['vR']:.6e} GeV",
        f"theta_u = {params['theta_u']:.6g}, theta_d = {params['theta_d']:.6g}",
        f"phi_u = {params['phi_u']:.6g}, phi_d = {params['phi_d']:.6g}",
        f"psi_u = {params['psi_u']:.6g}, psi_d = {params['psi_d']:.6g}",
        f"M_T = {params['M_T']:.6e} GeV",
    ]
    if result.get("provided_inputs"):
        lines.append(f"Provided inputs: {', '.join(result['provided_inputs'])}")
    if result.get("defaulted_inputs"):
        lines.append(f"Defaulted inputs: {', '.join(result['defaulted_inputs'])}")
    lines.append("Lifetimes tau/B [yr]:")
    for channel, value in result["lifetimes_years"].items():
        if channel == "total":
            continue
        lines.append(f"  {channel:16s} {float(value):.6e}")
    lines.append("Partial widths [GeV]:")
    for channel, value in result["widths"].items():
        lines.append(f"  {channel:16s} {float(value):.6e}")
    lines.append(f"Total width = {result['total_width_GeV']:.6e} GeV")
    return "\n".join(lines)


SUSY_YUKAWA_DEFAULT_RANGE = (1.0e-6, 1.0)
SUSY_ANGLE_DEFAULT_RANGE = (0.0, np.pi / 2.0)
SUSY_PHASE_DEFAULT_RANGE = (0.0, 2.0 * np.pi)
SUSY_SU5_SCAN_DEFAULT_RANGES = {
    "m0": (3000.0, 20000.0),
    "M_half": (500.0, 8000.0),
    "M1": (500.0, 8000.0),
    "M2": (1000.0, 16000.0),
    "M3": (3000.0, 48000.0),
    "AX": (-3.0, 3.0),
    "mu": (300.0, 8000.0),
    "tanbeta": (2.0, 60.0),
    "M_T": (1.0e16, 1.0e18),
    "phi13": SUSY_PHASE_DEFAULT_RANGE,
    "phi23": SUSY_PHASE_DEFAULT_RANGE,
    "fU": SUSY_YUKAWA_DEFAULT_RANGE,
    "fD": SUSY_YUKAWA_DEFAULT_RANGE,
    "fE": SUSY_YUKAWA_DEFAULT_RANGE,
}
SUSY_SO10_SCAN_DEFAULT_RANGES = {
    "M_T": (1.0e16, 1.0e18),
    "h": SUSY_YUKAWA_DEFAULT_RANGE,
    "f": SUSY_YUKAWA_DEFAULT_RANGE,
    "g": SUSY_YUKAWA_DEFAULT_RANGE,
    "mtildeW": (500.0, 8000.0),
    "mu": (300.0, 8000.0),
    "mtildeq": (3000.0, 20000.0),
    "tanbeta": (2.0, 60.0),
    "theta_u": SUSY_ANGLE_DEFAULT_RANGE,
    "theta_d": SUSY_ANGLE_DEFAULT_RANGE,
    "phi_u": SUSY_ANGLE_DEFAULT_RANGE,
    "phi_d": SUSY_ANGLE_DEFAULT_RANGE,
    "psi_u": SUSY_ANGLE_DEFAULT_RANGE,
    "psi_d": SUSY_ANGLE_DEFAULT_RANGE,
    "vL": (1.0e-12, 1.0e-6),
    "vR": (1.0e13, 1.0e17),
}


def _flatten_scan_matrix(row: dict, prefix: str, matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            base = f"{prefix}_{i+1}{j+1}"
            row[f"{base}_real"] = float(np.real(value))
            row[f"{base}_imag"] = float(np.imag(value))
            row[f"{base}_abs"] = float(np.abs(value))


def _flatten_numeric_payload(row: dict, prefix: str, value) -> None:
    if isinstance(value, dict):
        for key, subvalue in value.items():
            safe_key = re.sub(r"[^A-Za-z0-9_]+", "_", str(key)).strip("_")
            if safe_key:
                _flatten_numeric_payload(row, f"{prefix}_{safe_key}", subvalue)
        return
    arr = np.asarray(value)
    if arr.dtype == object:
        return
    if arr.ndim == 0:
        scalar = arr.item()
        if np.isscalar(scalar):
            if np.iscomplexobj(arr):
                row[f"{prefix}_real"] = float(np.real(scalar))
                row[f"{prefix}_imag"] = float(np.imag(scalar))
                row[f"{prefix}_abs"] = float(np.abs(scalar))
            else:
                row[prefix] = float(scalar)
        return
    for index in np.ndindex(arr.shape):
        scalar = arr[index]
        indexed = f"{prefix}_{'_'.join(str(i + 1) for i in index)}"
        if np.iscomplexobj(arr):
            row[f"{indexed}_real"] = float(np.real(scalar))
            row[f"{indexed}_imag"] = float(np.imag(scalar))
            row[f"{indexed}_abs"] = float(np.abs(scalar))
        else:
            row[indexed] = float(scalar)


def _canonical_decay_channel(channel: str) -> str:
    return channel.replace("nubar", "nu")


def _channel_from_lifetime_column(column: str) -> str | None:
    for channel in CHANNELS:
        if column == _channel_column(channel, "lifetime"):
            return channel
    return None


def _susy_scan_row(group: str, point: dict, result: dict, index: int) -> dict:
    row = {
        "scan_schema_version": SCAN_SCHEMA_VERSION,
        "scan_index": index,
        "scenario": f"SUSY {group}",
        "total_width_GeV": result["total_width_GeV"],
    }
    matrix_names = SUSY_SU5_MATRIX_INPUTS if group == "SU5" else SUSY_SO10_MATRIX_INPUTS
    for name, value in point.items():
        if name in matrix_names:
            _flatten_scan_matrix(row, name, value)
        elif group == "SO10" and name in {"x", "y"} and isinstance(value, dict):
            for coefficient, coefficient_value in sorted(value.items()):
                row[coefficient] = float(coefficient_value)
        elif np.isscalar(value):
            row[name] = float(value)

    canonical_lifetimes = {}
    for channel, width in result["widths"].items():
        canonical = _canonical_decay_channel(channel)
        row[_channel_column(canonical, "width")] = float(width)
    for channel, lifetime in result["lifetimes_years"].items():
        canonical = _canonical_decay_channel(channel)
        row[_channel_column(canonical, "lifetime")] = float(lifetime)
        if canonical in get_current_limits() and channel != "total":
            canonical_lifetimes[canonical] = float(lifetime)
    for channel, branching in result.get("branching_ratios", {}).items():
        canonical = _canonical_decay_channel(channel)
        row[_channel_column(canonical, "branching")] = float(branching)
    for output_name in ["chargino_masses", "neutralino_masses", "fermion_masses"]:
        if output_name in result:
            _flatten_numeric_payload(row, output_name, result[output_name])
    if canonical_lifetimes:
        row["classification"] = classify_model(canonical_lifetimes)
        channel, lifetime = min(canonical_lifetimes.items(), key=lambda item: item[1])
        row["min_lifetime_channel"] = channel
        row["min_lifetime_years"] = float(lifetime)
        ratios = {
            ch: tau / get_current_limits()[ch]
            for ch, tau in canonical_lifetimes.items()
            if get_current_limits().get(ch)
        }
        if ratios:
            bound_channel, ratio = min(ratios.items(), key=lambda item: item[1])
            row["most_constraining_channel"] = bound_channel
            row["most_constraining_bound_ratio"] = float(ratio)
    return row

def _susy_su5_flavor_digest(params: dict, names: list[str] | None = None) -> str:
    names = SUSY_SU5_OPERATOR_CONTEXT["fixed_flavor_parameters"] if names is None else names
    payload = b"".join(
        np.asarray(params[name], dtype=complex).view(np.float64).tobytes()
        if isinstance(params[name], np.ndarray)
        else np.asarray([float(params[name])], dtype=np.float64).tobytes()
        for name in names
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def _evaluate_susy_su5_fixed_flavor_point(index: int, point: dict) -> tuple[dict | None, dict | None]:
    try:
        result = _compute_susy_su5_point(point)
        return _susy_scan_row("SU5", point, result, index), None
    except Exception as exc:
        return None, {"index": int(index), "error": str(exc)}


@tool
@_with_matrix_elements_file
def run_susy_su5_fixed_flavor_scan_from_file(
    filepath: str,
    n_points: int = 200,
    vary: list[str] | None = None,
    ranges: dict[str, list[float]] | None = None,
    seed: int = 24680,
    n_jobs: int = -1,
    matrix_elements_file: str | None = None,
    bounds_file: str | None = None,
) -> dict:
    """Randomly scan SUSY SU(5) benchmark parameters, keeping fU, fD, and fE fixed.

    Supported varied parameters are m0, AX, tanbeta, M_T, mu, M_half, M1, M2, M3,
    phi13, and phi23. Explicitly requested phases are sampled uniformly
    over their ranges (default [0, 2*pi]); P is rebuilt for every point.
    Unrequested parameters remain at their file values. With no phase
    variation this retains the original fixed-flavor scan behavior.
    `ranges` maps each varied parameter to [minimum, maximum]. Parameters in
    `vary` without an explicit range use the documented SUSY SU(5) scan range.
    """
    n_points = min(int(n_points), 2000)
    if n_points < 1:
        raise ValueError("n_points must be at least 1.")

    params, metadata = _load_susy_benchmark_parameters_from_file(filepath, "SU5")
    allowed = set(SUSY_SU5_OPERATOR_CONTEXT["available_nonflavor_parameters"])
    allowed.update(SUSY_SU5_OPERATOR_CONTEXT["available_phase_parameters"])
    aliases = {
        "mt": "M_T", "m_t": "M_T", "triplet": "M_T", "triplet_mass": "M_T",
        "mhalf": "M_half", "m_half": "M_half", "mg": "M_half",
        "m1": "M1", "m2": "M2", "m3": "M3",
        "m_1": "M1", "m_2": "M2", "m_3": "M3",
        "a0": "AX", "ax": "AX", "tan_beta": "tanbeta", "tanb": "tanbeta",
        "phi_13": "phi13", "phi_23": "phi23",
    }

    requested_ranges = dict(ranges or {})
    requested_vary = list(vary or requested_ranges.keys() or ["M_T", "m0"])
    normalized_vary = []
    for raw_name in requested_vary:
        text_name = str(raw_name).strip()
        canonical = aliases.get(text_name.lower(), text_name)
        if canonical not in allowed:
            raise ValueError(
                f"Cannot vary {raw_name!r} in a SUSY SU(5) file scan. "
                f"Allowed: {', '.join(sorted(allowed))}."
            )
        if canonical not in normalized_vary:
            normalized_vary.append(canonical)

    normalized_ranges = {}
    for raw_name, pair in requested_ranges.items():
        text_name = str(raw_name).strip()
        canonical = aliases.get(text_name.lower(), text_name)
        if canonical not in allowed:
            raise ValueError(f"Range supplied for unsupported parameter {raw_name!r}.")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Range for {canonical} must be [minimum, maximum].")
        low, high = float(pair[0]), float(pair[1])
        if not np.isfinite(low) or not np.isfinite(high) or high < low:
            raise ValueError(f"Range for {canonical} must contain finite values with minimum <= maximum.")
        physical_low, physical_high = _susy_scalar_bounds("SU5")[canonical]
        if low < physical_low or high > physical_high:
            raise ValueError(
                f"Range for {canonical} must lie within [{physical_low}, {physical_high}]."
            )
        normalized_ranges[canonical] = (low, high)

    for name in normalized_vary:
        if name not in normalized_ranges:
            normalized_ranges[name] = tuple(float(v) for v in SUSY_SU5_SCAN_DEFAULT_RANGES[name])

    varied_phases = [name for name in normalized_vary if name in {"phi13", "phi23"}]
    fixed_flavor_parameters = [
        name for name in SUSY_SU5_OPERATOR_CONTEXT["fixed_flavor_parameters"]
        if name not in varied_phases and not (varied_phases and name == "P")
    ]
    flavor_digest = _susy_su5_flavor_digest(params, fixed_flavor_parameters)
    rng = np.random.default_rng(int(seed))
    points = []
    for _ in range(n_points):
        point = {
            key: (value.copy() if isinstance(value, np.ndarray) else value)
            for key, value in params.items()
        }
        for name in normalized_vary:
            low, high = normalized_ranges[name]
            point[name] = float(rng.uniform(low, high)) if high > low else float(low)
        if varied_phases:
            point["P"] = _susy_su5_p_from_phases(point["phi13"], point["phi23"])
        points.append(point)

    if any(_susy_su5_flavor_digest(point, fixed_flavor_parameters) != flavor_digest for point in points):
        raise RuntimeError("Internal error: fixed SUSY SU(5) flavor inputs changed while constructing the scan.")

    with parallel_backend("loky", inner_max_num_threads=1):
        evaluated = Parallel(n_jobs=int(n_jobs), batch_size=1)(
            delayed(call_with_bounds)(current_bounds_configuration(), call_with_matrix_elements,
                                      current_matrix_elements(), _evaluate_susy_su5_fixed_flavor_point, index, point)
            for index, point in enumerate(points)
        )
    rows = [row for row, error in evaluated if row is not None]
    failures = [error for row, error in evaluated if error is not None]
    if not rows:
        return {
            "error": f"SUSY SU(5) file scan failed for all {n_points} points.",
            "failures": failures[:10],
        }

    frame = pd.DataFrame(rows)
    classification_counts = {
        str(key): int(value) for key, value in frame["classification"].value_counts().items()
    }
    summary = {
        "scenario": "SUSY SU5",
        "scan_mode": "fixed-yukawa-phase-random" if varied_phases else "fixed-flavor-random",
        "operator_context": {**SUSY_SU5_OPERATOR_CONTEXT,
                             "fixed_flavor_parameters": fixed_flavor_parameters},
        "input_file": metadata["input_file"],
        "input_file_sha256": _file_sha256(metadata["input_file"]),
        "input_schema": metadata["schema"],
        "n_requested": n_points,
        "n_completed": len(rows),
        "n_failed": len(failures),
        "vary": normalized_vary,
        "ranges": {name: [float(pair[0]), float(pair[1])] for name, pair in normalized_ranges.items()},
        "benchmark_parameters": {name: float(params[name]) for name in normalized_vary},
        "range_extends_above_benchmark": {
            name: bool(normalized_ranges[name][1] > float(params[name]))
            for name in normalized_vary
        },
        "seed": int(seed),
        "flavor_structure_fixed": not bool(varied_phases),
        "yukawa_matrices_fixed": True,
        "fixed_flavor_parameters": fixed_flavor_parameters,
        "flavor_digest_parameters": fixed_flavor_parameters,
        "flavor_digest": flavor_digest,
        "classification_counts": classification_counts,
    }
    summary["range_advisories"] = [
        f"{name} scan maximum does not exceed the benchmark value {float(params[name]):.6e}."
        for name in normalized_vary
        if normalized_ranges[name][1] <= float(params[name])
    ]
    if "most_constraining_bound_ratio" in frame.columns:
        best_index = frame["most_constraining_bound_ratio"].idxmax()
        summary.update({
            "best_bound_ratio": float(frame.loc[best_index, "most_constraining_bound_ratio"]),
            "best_bound_ratio_channel": str(frame.loc[best_index, "most_constraining_channel"]),
            "best_scan_index": int(frame.loc[best_index, "scan_index"]),
            "best_classification": str(frame.loc[best_index, "classification"]),
        })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    scan_prefix = "susy_su5_phase_scan" if varied_phases else "susy_su5_fixed_flavor_scan"
    digest = _cache_key(scan_prefix, summary)[:8]
    scan_filepath = str(SCAN_RESULTS_DIR / f"{scan_prefix}_{stamp}_{digest}.h5")
    Path(scan_filepath).parent.mkdir(parents=True, exist_ok=True)
    frame.to_hdf(scan_filepath, key="results", mode="w")
    summary["scan_filepath"] = scan_filepath
    if failures:
        summary["failures_preview"] = failures[:5]

    artifact = _register_artifact(
        kind="susy-su5-scan" if varied_phases else "susy-su5-fixed-flavor-scan",
        payload={"scan_filepath": scan_filepath, "config": summary, "summary": summary},
        summary=summary,
    )
    output = {**summary, "artifact_id": artifact["artifact_id"], "artifact_kind": artifact["kind"]}
    _set_last_pd_result(output, _format_susy_scan_result(output))
    return output


def _susy_so10_flavor_digest(params: dict) -> str:
    payload = b"".join(
        np.asarray(params[name], dtype=complex).view(np.float64).tobytes()
        for name in ("h", "f", "g")
    )
    payload += np.asarray(
        [
            float(params[name])
            for name in (
                "tanbeta", "theta_u", "phi_u", "psi_u",
                "theta_d", "phi_d", "psi_d", "vL", "vR",
            )
        ],
        dtype=np.float64,
    ).tobytes()
    payload += np.asarray(
        [float(params[container][name]) for container in ("x", "y") for name in sorted(params[container])],
        dtype=np.float64,
    ).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _evaluate_susy_so10_fixed_flavor_point(index: int, point: dict) -> tuple[dict | None, dict | None]:
    try:
        result = _compute_susy_so10_point(point)
        return _susy_scan_row("SO10", point, result, index), None
    except Exception as exc:
        return None, {"index": int(index), "error": str(exc)}


@tool
@_with_matrix_elements_file
def run_susy_so10_fixed_flavor_scan_from_file(
    filepath: str,
    n_points: int = 200,
    vary: list[str] | None = None,
    ranges: dict[str, list[float]] | None = None,
    seed: int = 24680,
    n_jobs: int = -1,
    matrix_elements_file: str | None = None,
    bounds_file: str | None = None,
) -> dict:
    """Scan SUSY SO(10) mass parameters while preserving the complete flavor and triplet-mixing structure from a benchmark file.

    Supported varied parameters are M_T, mtildeq (alias m_sfermion), mtildeW
    (alias M_wino), and mu. By default, M_T and mtildeq are varied.
    """
    n_points = min(int(n_points), 2000)
    if n_points < 1:
        raise ValueError("n_points must be at least 1.")

    params, metadata = _load_susy_benchmark_parameters_from_file(filepath, "SO10")
    allowed = set(SUSY_SO10_OPERATOR_CONTEXT["available_nonflavor_parameters"])
    aliases = {
        "mt": "M_T", "m_t": "M_T", "triplet": "M_T", "triplet_mass": "M_T",
        "m_sfermion": "mtildeq", "msfermion": "mtildeq", "sfermion": "mtildeq",
        "mtilde_q": "mtildeq", "m_wino": "mtildeW", "mwino": "mtildeW",
        "wino": "mtildeW", "mtildew": "mtildeW",
    }

    requested_ranges = dict(ranges or {})
    requested_vary = list(vary or requested_ranges.keys() or ["M_T", "mtildeq"])
    normalized_vary = []
    for raw_name in requested_vary:
        text_name = str(raw_name).strip()
        canonical = aliases.get(text_name.lower(), text_name)
        if canonical not in allowed:
            raise ValueError(
                f"Cannot vary {raw_name!r} in a fixed-flavor SUSY SO(10) scan. "
                f"Allowed: {', '.join(SUSY_SO10_OPERATOR_CONTEXT['available_nonflavor_parameters'])}."
            )
        if canonical not in normalized_vary:
            normalized_vary.append(canonical)

    normalized_ranges = {}
    for raw_name, pair in requested_ranges.items():
        text_name = str(raw_name).strip()
        canonical = aliases.get(text_name.lower(), text_name)
        if canonical not in allowed:
            raise ValueError(f"Range supplied for unsupported or flavor parameter {raw_name!r}.")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Range for {canonical} must be [minimum, maximum].")
        low, high = float(pair[0]), float(pair[1])
        if not np.isfinite(low) or not np.isfinite(high) or high < low:
            raise ValueError(f"Range for {canonical} must contain finite values with minimum <= maximum.")
        physical_low, physical_high = _susy_scalar_bounds("SO10")[canonical]
        if low < physical_low or high > physical_high:
            raise ValueError(
                f"Range for {canonical} must lie within [{physical_low}, {physical_high}]."
            )
        normalized_ranges[canonical] = (low, high)

    for name in normalized_vary:
        if name not in normalized_ranges:
            normalized_ranges[name] = tuple(float(v) for v in SUSY_SO10_SCAN_DEFAULT_RANGES[name])

    flavor_digest = _susy_so10_flavor_digest(params)
    rng = np.random.default_rng(int(seed))
    points = []
    for _ in range(n_points):
        point = {
            key: (
                value.copy() if isinstance(value, np.ndarray)
                else dict(value) if isinstance(value, dict)
                else value
            )
            for key, value in params.items()
        }
        for name in normalized_vary:
            low, high = normalized_ranges[name]
            point[name] = float(rng.uniform(low, high)) if high > low else float(low)
        points.append(point)

    if any(_susy_so10_flavor_digest(point) != flavor_digest for point in points):
        raise RuntimeError("Internal error: SUSY SO(10) flavor structure changed while constructing the scan.")

    with parallel_backend("loky", inner_max_num_threads=1):
        evaluated = Parallel(n_jobs=int(n_jobs), batch_size=1)(
            delayed(call_with_bounds)(current_bounds_configuration(), call_with_matrix_elements,
                                      current_matrix_elements(), _evaluate_susy_so10_fixed_flavor_point, index, point)
            for index, point in enumerate(points)
        )
    rows = [row for row, error in evaluated if row is not None]
    failures = [error for row, error in evaluated if error is not None]
    if not rows:
        return {
            "error": f"Fixed-flavor SUSY SO(10) scan failed for all {n_points} points.",
            "failures": failures[:10],
        }

    frame = pd.DataFrame(rows)
    classification_counts = {
        str(key): int(value) for key, value in frame["classification"].value_counts().items()
    }
    summary = {
        "scenario": "SUSY SO10",
        "scan_mode": "fixed-flavor-random",
        "operator_context": dict(SUSY_SO10_OPERATOR_CONTEXT),
        "input_file": metadata["input_file"],
        "input_file_sha256": _file_sha256(metadata["input_file"]),
        "input_schema": metadata["schema"],
        "n_requested": n_points,
        "n_completed": len(rows),
        "n_failed": len(failures),
        "vary": normalized_vary,
        "ranges": {name: [float(pair[0]), float(pair[1])] for name, pair in normalized_ranges.items()},
        "benchmark_parameters": {name: float(params[name]) for name in normalized_vary},
        "range_extends_above_benchmark": {
            name: bool(normalized_ranges[name][1] > float(params[name]))
            for name in normalized_vary
        },
        "seed": int(seed),
        "flavor_structure_fixed": True,
        "fixed_flavor_parameters": list(SUSY_SO10_OPERATOR_CONTEXT["fixed_flavor_parameters"]),
        "flavor_digest": flavor_digest,
        "classification_counts": classification_counts,
    }
    summary["range_advisories"] = [
        f"{name} scan maximum does not exceed the benchmark value {float(params[name]):.6e}."
        for name in normalized_vary
        if normalized_ranges[name][1] <= float(params[name])
    ]
    if "most_constraining_bound_ratio" in frame.columns:
        best_index = frame["most_constraining_bound_ratio"].idxmax()
        summary.update({
            "best_bound_ratio": float(frame.loc[best_index, "most_constraining_bound_ratio"]),
            "best_bound_ratio_channel": str(frame.loc[best_index, "most_constraining_channel"]),
            "best_scan_index": int(frame.loc[best_index, "scan_index"]),
            "best_classification": str(frame.loc[best_index, "classification"]),
        })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    digest = _cache_key("susy_so10_fixed_flavor_scan", summary)[:8]
    scan_filepath = str(SCAN_RESULTS_DIR / f"susy_so10_fixed_flavor_scan_{stamp}_{digest}.h5")
    Path(scan_filepath).parent.mkdir(parents=True, exist_ok=True)
    frame.to_hdf(scan_filepath, key="results", mode="w")
    summary["scan_filepath"] = scan_filepath
    if failures:
        summary["failures_preview"] = failures[:5]

    artifact = _register_artifact(
        kind="susy-so10-fixed-flavor-scan",
        payload={"scan_filepath": scan_filepath, "config": summary, "summary": summary},
        summary=summary,
    )
    output = {**summary, "artifact_id": artifact["artifact_id"], "artifact_kind": artifact["kind"]}
    _set_last_pd_result(output, _format_susy_scan_result(output))
    return output


def _format_susy_scan_result(result: dict) -> str:
    if "error" in result:
        return result["error"]
    lines = [
        f"Scenario: {result['scenario']} scan",
        f"Completed {result['n_completed']} / {result['n_requested']} points; failures = {result['n_failed']}",
    ]
    if result.get("minimum_lifetime_years") is not None:
        lines.append(
            f"Minimum lifetime = {result['minimum_lifetime_years']:.6e} yr "
            f"in {result['minimum_lifetime_channel']} at scan_index {result['minimum_lifetime_scan_index']}"
        )
    lines.append(f"Saved scan: {result['scan_filepath']}")
    lines.append(f"Artifact: {result['artifact_id']}")
    return "\n".join(lines)


def _set_last_pd_result(result: dict, formatted: str) -> None:
    global LAST_PD_RESULT, LAST_PD_RESULT_TEXT
    record_matrix_elements(result)
    record_bounds(result)
    LAST_PD_RESULT = result
    LAST_PD_RESULT_TEXT = formatted
    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": result,
        "formatted": formatted,
    }
    temporary_path = LATEST_PD_RESULT_PATH.with_name(
        f"{LATEST_PD_RESULT_PATH.name}.tmp-{os.getpid()}-{threading.get_ident()}-{time.time_ns()}"
    )
    try:
        with temporary_path.open("wb") as handle:
            pickle.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, LATEST_PD_RESULT_PATH)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _load_last_pd_result_from_disk() -> tuple[dict | None, str | None, str | None]:
    if not LATEST_PD_RESULT_PATH.exists():
        return None, None, None
    try:
        with LATEST_PD_RESULT_PATH.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError):
        return None, None, None
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
        return None, None, None
    return payload["result"], payload.get("formatted"), payload.get("saved_at_utc")


def _canonical_lifetimes_for_bounds(result: dict) -> dict[str, float]:
    lifetimes = result.get("lifetimes_years") or {}
    if not lifetimes and result.get("lifetimes_1e34yr"):
        lifetimes = {ch: float(v) * 1e34 for ch, v in result["lifetimes_1e34yr"].items()}
    canonical = {}
    for channel, lifetime in lifetimes.items():
        if channel == "total":
            continue
        normalized_text = str(channel).replace("nubar", "nu").replace("anti-neutrino", "nu")
        normalized = _extract_channel(normalized_text) or normalized_text
        if normalized in get_current_limits():
            canonical[normalized] = float(lifetime)
    return canonical


def _ollama_num_ctx(model: str) -> int:
    """Choose a VRAM-conscious context window for the local model family."""
    override = os.environ.get("PROTON_AGENT_NUM_CTX")
    if override is not None:
        value = int(override)
        if value <= 0:
            raise ValueError("PROTON_AGENT_NUM_CTX must be a positive integer.")
        return value

    model_name = model.strip().lower()
    return 16384 if model_name.startswith(("qwen", "gpt-oss")) else 8192


def _ollama_client_kwargs() -> dict:
    timeout = float(os.environ.get("PROTON_AGENT_LLM_TIMEOUT_SECONDS", "120"))
    if timeout <= 0:
        raise ValueError("PROTON_AGENT_LLM_TIMEOUT_SECONDS must be positive.")
    return {"timeout": timeout}


def _build_llm_without_tools(target: dict):
    if target["provider"] == "ollama":
        return ChatOllama(
            model=target["model"],
            temperature=0,
            num_ctx=_ollama_num_ctx(target["model"]),
            client_kwargs=_ollama_client_kwargs(),
        )
    if target["provider"] == "openwebui":
        return OpenWebUIChatModel(
            model=target["model"],
            api_key=target.get("api_key"),
            email=target.get("api_email"),
            password=target.get("api_password"),
            base_url=target["api_base"],
            temperature=0,
        )
    if target["provider"] == "anthropic":
        return AnthropicChatModel(
            model=target["model"],
            api_key=target["api_key"],
            base_url=target["api_base"],
            temperature=0,
        )
    return OpenAICompatibleChatModel(
        model=target["model"],
        api_key=target["api_key"],
        base_url=_normalize_openai_base_url(target["api_base"]),
        temperature=0,
    )


def _scan_schema_status(df: pd.DataFrame) -> dict:
    has_matrix_cols = any(col.startswith("NL_") or col.startswith("UL_") for col in df.columns)
    has_width_cols = any(col.startswith("width_") for col in df.columns)
    schema_version = int(df["scan_schema_version"].iloc[0]) if "scan_schema_version" in df.columns else 1
    return {
        "schema_version": schema_version,
        "has_matrix_columns": has_matrix_cols,
        "has_width_columns": has_width_cols,
        "is_rich_scan": has_matrix_cols and has_width_cols and schema_version >= SCAN_SCHEMA_VERSION,
    }


def _requested_axis_range(scan_artifact_id: str, axis: str):
    record, payload = _scan_payload(scan_artifact_id)
    if payload is None:
        return None

    config = payload.get("config")
    if config:
        if axis == "MGUT" and {"MGUT_min", "MGUT_max"} <= set(config):
            return [float(config["MGUT_min"]), float(config["MGUT_max"])]
        if axis == "gGUT" and {"gGUT_min", "gGUT_max"} <= set(config):
            return [float(config["gGUT_min"]), float(config["gGUT_max"])]

    grid = payload.get("grid_config")
    if grid:
        if axis == "MGUT":
            return [float(grid["MGUT"][0]), float(grid["MGUT"][1])]
        if axis == "gGUT":
            return [float(grid["gGUT"][0]), float(grid["gGUT"][1])]
    return None


def _parse_tool_code_calls(content: str) -> list[dict]:
    if not content or "<tool_code>" not in content:
        return []

    match = re.search(r"<tool_code>\s*(.*?)\s*</tool_code>", content, re.DOTALL)
    if not match:
        return []

    block = match.group(1).strip()
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return []

    calls = []
    for index, stmt in enumerate(tree.body):
        node = None
        if isinstance(stmt, ast.Expr):
            node = stmt.value
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Call)
        ):
            node = node.args[0]

        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue

        tool_name = node.func.id
        if tool_name not in TOOL_MAP:
            continue

        args = {}
        convertible = True
        for kw in node.keywords:
            if kw.arg is None:
                convertible = False
                break
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                convertible = False
                break

        if node.args:
            convertible = False

        if not convertible:
            continue

        calls.append(
            {
                "name": tool_name,
                "args": args,
                "id": f"pseudo_{tool_name}_{index}",
            }
        )

    return calls


def _extract_tool_calls(message) -> list[dict]:
    if hasattr(message, "tool_calls") and message.tool_calls:
        return list(message.tool_calls)
    return _parse_tool_code_calls(getattr(message, "content", ""))


# ================================================================
# INTERNAL SCAN SUMMARIZER
# Uses exact key names from evaluate_point / scan_runner.py
# ================================================================

def _summarize_scan(results: list, mode: str, grid_config=None) -> dict:
    n_total    = len(results)
    n_excluded = sum(1 for r in results if r[KEY_CLASS] == "excluded")
    n_testable = sum(1 for r in results if r[KEY_CLASS] == "testable")
    n_safe     = sum(1 for r in results if r[KEY_CLASS] == "safe")

    # Which channel causes most exclusions
    exclusion_counts = {}
    for r in results:
        if r[KEY_CLASS] == "excluded":
            for ch, tau in r[KEY_LIFETIMES].items():
                bound = get_current_limits().get(ch)
                if bound and tau < bound:
                    exclusion_counts[ch] = exclusion_counts.get(ch, 0) + 1

    dominant_exclusion = (
        max(exclusion_counts, key=exclusion_counts.get)
        if exclusion_counts else None
    )

    all_MGUT      = [r[KEY_MGUT] for r in results]
    viable_MGUT   = [r[KEY_MGUT] for r in results if r[KEY_CLASS] != "excluded"]
    testable_MGUT = [r[KEY_MGUT] for r in results if r[KEY_CLASS] == "testable"]

    medians = {}
    for ch in CHANNELS:
        vals = [r[KEY_LIFETIMES][ch] for r in results if ch in r[KEY_LIFETIMES]]
        medians[ch] = float(np.median(vals)) if vals else 0.0

    anomalies = []
    for r in results:
        for ch in CHANNELS:
            tau = r[KEY_LIFETIMES].get(ch, 0)
            if medians[ch] > 0 and tau > 10 * medians[ch]:
                anomalies.append({
                    KEY_MGUT: r[KEY_MGUT],
                    KEY_GGUT: r[KEY_GGUT],
                    "channel": ch,
                    "lifetime_1e34yr": tau / 1e34,
                    "ratio_to_median": tau / medians[ch],
                })
                break

    return {
        "scan_mode": mode,
        "n_total": n_total,
        "n_excluded": n_excluded,
        "n_testable": n_testable,
        "n_safe": n_safe,
        "fraction_excluded": n_excluded / n_total,
        "fraction_testable": n_testable / n_total,
        "fraction_safe":     n_safe     / n_total,
        "dominant_exclusion_channel": dominant_exclusion,
        "exclusion_channel_counts": exclusion_counts,
        "MGUT_full_range":     [min(all_MGUT),      max(all_MGUT)],
        "MGUT_viable_range":   [min(viable_MGUT),   max(viable_MGUT)]   if viable_MGUT   else None,
        "MGUT_testable_range": [min(testable_MGUT), max(testable_MGUT)] if testable_MGUT else None,
        "lifetime_medians_1e34yr": {ch: medians[ch] / 1e34 for ch in CHANNELS},
        "n_anomalies": len(anomalies),
        "anomalies": anomalies[:10],
        "grid_config": grid_config,
    }


# ================================================================
# TOOLS
# ================================================================

BOUND_CHANNEL_ALIASES = {
    "p->pi+ nubar": "p->pi+ nu",
    "p->K+ nubar": "p->K+ nu",
}


def _nonsusy_tool_result(group: str, params: dict, *, point_kind: str, seed: int | None = None) -> dict:
    result = _compute_nonsusy_decay_rates(
        group,
        MGUT=float(params["MGUT"]),
        gGUT=float(params["gGUT"]),
        UL=params["UL"],
        UR=params["UR"],
        DL=params["DL"],
        DR=params["DR"],
        EL=params["EL"],
        ER=params["ER"],
        NL=params["NL"],
    )
    total_width = float(result["total_width_GeV"])
    total_lifetime = np.inf if total_width <= 0 else 6.58e-25 / (total_width * 365 * 24 * 3600)
    is_su5 = group == "SU5"
    comparisons = result["experimental_comparison"]
    experimental_summary = result["experimental_summary"]
    output = {
        "scenario": f"nonSUSY {group}",
        "point_kind": point_kind,
        "formula_module": "nonsusy_su5_formula" if is_su5 else "nonsusy_so10_formula",
        "matrix_convention": (
            "SU(5): U, Uc, D, Dc, E, Ec, N; charge-conjugate Uc/Dc/Ec are converted to right-handed rotations"
            if is_su5
            else "SO(10): UL, UR, DL, DR, EL, ER, NL are used directly"
        ),
        "second_gauge_boson_contribution": "absent" if is_su5 else "included",
        "MGUT": float(params["MGUT"]),
        "gGUT": float(params["gGUT"]),
        "seed": seed,
        "widths_GeV": {channel: float(value) for channel, value in result["widths"].items()},
        "total_width_GeV": total_width,
        "total_lifetime_years": float(total_lifetime),
        "lifetimes_years": {channel: float(value) for channel, value in result["lifetimes_years"].items()},
        "lifetimes_1e34yr": {channel: float(value) / 1e34 for channel, value in result["lifetimes_years"].items()},
        "branching_ratios": {channel: float(value) for channel, value in result["branching_ratios"].items()},
        "wilson_coefficients": result.get("wilson_coefficients", {}),
        "experimental_comparison": comparisons,
        "experimental_summary": experimental_summary,
        "classification": experimental_summary["classification"],
    }
    _set_last_pd_result(output, json.dumps(_jsonable(output), indent=2))
    return output


def _random_nonsusy_params(MGUT: float, gGUT: float, seed: int | None) -> dict:
    if seed is not None:
        np.random.seed(int(seed))
    params = next(generate_parameters_with_ckm_pmns(1))
    params["MGUT"] = float(MGUT)
    params["gGUT"] = float(gGUT)
    return params


def _file_sha256(filepath: str | os.PathLike) -> str:
    return hashlib.sha256(Path(filepath).read_bytes()).hexdigest()


def _nonsusy_file_tool_result(group: str, filepath: str) -> dict:
    params, metadata = _load_benchmark_parameters_from_file(filepath, "nonSUSY", group)
    output = _nonsusy_tool_result(group, params, point_kind=f"file benchmark ({metadata['schema']})")
    output.update({
        "input_file": metadata["input_file"],
        "input_file_sha256": _file_sha256(metadata["input_file"]),
        "input_schema": metadata["schema"],
        "input_convention": metadata["input_convention"],
        "n_file_values": metadata["n_file_values"],
    })
    state_result = _compute_non_susy_pd_from_file("nonSUSY", group, filepath)
    state_result.update(output)
    _set_last_pd_result(state_result, _format_non_susy_pd_result(state_result))
    return output


@tool
@_with_matrix_elements_file
def compute_nonsusy_su5_point(MGUT: float, gGUT: float, seed: int | None = None, matrix_elements_file: str | None = None, bounds_file: str | None = None) -> dict:
    """Compute a non-supersymmetric SU(5) point. Uses U/Uc/D/Dc/E/Ec/N conventions and has no second gauge-boson contribution."""
    params = _random_nonsusy_params(MGUT, gGUT, seed)
    return _nonsusy_tool_result("SU5", params, point_kind="custom point", seed=seed)


@tool
@_with_matrix_elements_file
def compute_nonsusy_so10_point(MGUT: float, gGUT: float, seed: int | None = None, matrix_elements_file: str | None = None, bounds_file: str | None = None) -> dict:
    """Compute a non-supersymmetric SO(10) point. Uses UL/UR/DL/DR/EL/ER/NL directly and includes the nonzero second gauge-boson contribution."""
    params = _random_nonsusy_params(MGUT, gGUT, seed)
    return _nonsusy_tool_result("SO10", params, point_kind="custom point", seed=seed)


@tool
@_with_matrix_elements_file
def compute_nonsusy_su5_from_file(filepath: str, matrix_elements_file: str | None = None, bounds_file: str | None = None) -> dict:
    """Read and validate a local non-SUSY SU(5) benchmark file, then compute its proton-decay observables with the SU(5)-only formula."""
    return _nonsusy_file_tool_result("SU5", filepath)


@tool
@_with_matrix_elements_file
def compute_nonsusy_so10_from_file(filepath: str, matrix_elements_file: str | None = None, bounds_file: str | None = None) -> dict:
    """Read and validate a local non-SUSY SO(10) full-matrix benchmark file, then compute its proton-decay observables with the SO(10)-only formula."""
    return _nonsusy_file_tool_result("SO10", filepath)


def _public_susy_result(group: str, result: dict, point_kind: str) -> dict:
    scalar_inputs = {
        key: float(value)
        for key, value in result["inputs"].items()
        if np.isscalar(value) and not isinstance(value, (str, bytes, complex))
    }
    lifetimes = {
        channel: float(value)
        for channel, value in result["lifetimes_years"].items()
    }
    comparisons = result["experimental_comparison"]
    experimental_summary = result["experimental_summary"]
    output = {
        "scenario": f"SUSY {group}",
        "point_kind": point_kind,
        "formula_module": "susy_su5_formula" if group == "SU5" else "susy_so10_formula",
        "inputs": scalar_inputs,
        "provided_inputs": list(result.get("provided_inputs", [])),
        "defaulted_inputs": list(result.get("defaulted_inputs", [])),
        "widths_GeV": {channel: float(value) for channel, value in result["widths"].items()},
        "total_width_GeV": float(result["total_width_GeV"]),
        "total_lifetime_years": float(lifetimes.get("total", np.inf)),
        "lifetimes_years": lifetimes,
        "branching_ratios": {
            channel: float(value) for channel, value in result["branching_ratios"].items()
        },
        "experimental_comparison": comparisons,
        "experimental_summary": experimental_summary,
        "classification": experimental_summary["classification"],
    }
    if group == "SU5":
        output["operator_context"] = dict(SUSY_SU5_OPERATOR_CONTEXT)
        output["chargino_masses_GeV"] = np.asarray(result["chargino_masses"]).real.tolist()
        output["neutralino_masses_GeV"] = np.asarray(result["neutralino_masses"]).real.tolist()
    else:
        output["operator_context"] = dict(SUSY_SO10_OPERATOR_CONTEXT)
        output["fermion_masses_GeV"] = {
            key: np.asarray(value).real.tolist()
            for key, value in result.get("fermion_masses", {}).items()
        }
    if result.get("input_file"):
        output["input_file"] = result["input_file"]
        output["input_file_sha256"] = result["input_file_sha256"]
        output["input_schema"] = result["schema"]
    return output


@tool
@_with_matrix_elements_file
def compute_susy_su5_point(
    m0: float | None = None,
    AX: float | None = None,
    tanbeta: float | None = None,
    M_T: float | None = None,
    mu: float | None = None,
    M_half: float | None = None,
    M1: float | None = None,
    M2: float | None = None,
    M3: float | None = None,
    phi13: float | None = None,
    phi23: float | None = None,
    fU: list[list[float]] | None = None,
    fD: list[list[float]] | None = None,
    fE: list[list[float]] | None = None,
    matrix_elements_file: str | None = None,
    bounds_file: str | None = None,
) -> dict:
    """Compute a supersymmetric SU(5) proton-decay point. Omitted inputs use the documented benchmark defaults."""
    values = {
        "m0": m0, "AX": AX, "tanbeta": tanbeta, "M_T": M_T,
        "mu": mu, "M_half": M_half, "M1": M1, "M2": M2, "M3": M3,
        "phi13": phi13, "phi23": phi23,
    }
    overrides = {key: float(value) for key, value in values.items() if value is not None}
    for key, value in {"fU": fU, "fD": fD, "fE": fE}.items():
        if value is not None:
            overrides[key] = np.asarray(value, dtype=complex)
    result = _compute_susy_su5_point(overrides)
    result["point_kind"] = "custom point"
    _set_last_pd_result(result, _format_susy_su5_result(result))
    return _public_susy_result("SU5", result, "custom point")


def _susy_file_tool_result(group: str, filepath: str, scalar_overrides: dict | None = None) -> dict:
    overrides = dict(scalar_overrides or {})
    if overrides:
        params, metadata = _load_susy_benchmark_parameters_from_file(filepath, group)
        params.update(overrides)
        result = _compute_susy_su5_point(params) if group == "SU5" else _compute_susy_so10_point(params)
        result["schema"] = metadata["schema"]
        result["input_file"] = metadata["input_file"]
        result["point_kind"] = f"file benchmark with scalar overrides ({metadata['schema']})"
        provided = list(dict.fromkeys([*metadata["provided_inputs"], *overrides.keys()]))
        result["provided_inputs"] = [_susy_mcmc_display_name(name) for name in provided]
        canonical_inputs = SUSY_SU5_INPUTS if group == "SU5" else SUSY_SO10_INPUTS
        result["defaulted_inputs"] = [
            _susy_mcmc_display_name(name)
            for name in canonical_inputs
            if name not in provided
        ]
    else:
        result = _compute_susy_point_from_file("SUSY", group, filepath)
    result["input_file_sha256"] = _file_sha256(result["input_file"])
    formatted = _format_susy_su5_result(result) if group == "SU5" else _format_susy_so10_result(result)
    _set_last_pd_result(result, formatted)
    return _public_susy_result(group, result, result["point_kind"])


@tool
@_with_matrix_elements_file
def compute_susy_su5_from_file(filepath: str, matrix_elements_file: str | None = None, bounds_file: str | None = None) -> dict:
    """Read and compute a SUSY SU(5) benchmark file exactly as supplied, without changing any input parameter."""
    return _susy_file_tool_result("SU5", filepath)


@tool
@_with_matrix_elements_file
def compute_susy_su5_from_file_with_overrides(
    filepath: str,
    m0: float | None = None,
    AX: float | None = None,
    tanbeta: float | None = None,
    M_T: float | None = None,
    mu: float | None = None,
    M_half: float | None = None,
    M1: float | None = None,
    M2: float | None = None,
    M3: float | None = None,
    matrix_elements_file: str | None = None,
    bounds_file: str | None = None,
) -> dict:
    """Compute an explicitly requested SUSY SU(5) candidate point by overriding user-supplied non-flavor scalars while retaining the file's Yukawa matrices and phases. Never use for qualitative advice or invent override values."""
    values = {
        "m0": m0, "AX": AX, "tanbeta": tanbeta,
        "M_T": M_T, "mu": mu, "M_half": M_half, "M1": M1, "M2": M2, "M3": M3,
    }
    overrides = {key: float(value) for key, value in values.items() if value is not None}
    if not overrides:
        raise ValueError("At least one explicitly requested scalar override is required.")
    bounds = _susy_scalar_bounds("SU5")
    for name, value in overrides.items():
        low, high = bounds[name]
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must lie within [{low}, {high}].")
    return _susy_file_tool_result("SU5", filepath, overrides)


@tool
@_with_matrix_elements_file
def compute_susy_so10_from_file(filepath: str, matrix_elements_file: str | None = None, bounds_file: str | None = None) -> dict:
    """Read and compute a SUSY SO(10) benchmark file exactly as supplied, without changing any input parameter."""
    return _susy_file_tool_result("SO10", filepath)


@tool
@_with_matrix_elements_file
def compute_susy_so10_from_file_with_overrides(
    filepath: str,
    M_T: float | None = None,
    M_wino: float | None = None,
    mu: float | None = None,
    m_sfermion: float | None = None,
    matrix_elements_file: str | None = None,
    bounds_file: str | None = None,
) -> dict:
    """Compute an explicitly requested SUSY SO(10) candidate point by overriding user-supplied non-flavor masses while retaining the file's complete flavor structure. Never use for qualitative advice or invent override values."""
    values = {
        "M_T": M_T, "mtildeW": M_wino,
        "mu": mu, "mtildeq": m_sfermion,
    }
    overrides = {key: float(value) for key, value in values.items() if value is not None}
    if not overrides:
        raise ValueError("At least one explicitly requested mass override is required.")
    bounds = _susy_scalar_bounds("SO10")
    for name, value in overrides.items():
        low, high = bounds[name]
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{_susy_mcmc_display_name(name)} must lie within [{low}, {high}].")
    return _susy_file_tool_result("SO10", filepath, overrides)


@tool
@_with_matrix_elements_file
def find_susy_so10_minimum_sfermion_mass_from_file(
    filepath: str,
    matrix_elements_file: str | None = None,
    bounds_file: str | None = None,
) -> dict:
    """Find and plot the minimum increased SUSY SO(10) sfermion mass that passes current bounds.

    All file inputs, including Yukawa/flavor tensors, M_T, M_wino, and mu, are
    held fixed. The tool adaptively extends a logarithmic mass grid until it
    brackets the first safe point, plots every partial lifetime with its
    experimental bound, and bisects the crossing. The physical calculator's
    allowed upper mass is used internally; the LLM cannot invent a search cap.
    """
    tolerance = 1.0e-6
    grid_points = 72

    params, metadata = _load_susy_benchmark_parameters_from_file(filepath, "SO10")
    benchmark_mass = float(params["mtildeq"])
    search_upper = float(_susy_scalar_bounds("SO10")["mtildeq"][1])

    evaluations = 0

    def evaluate(mass: float) -> tuple[float, dict]:
        nonlocal evaluations
        result = _susy_file_tool_result("SO10", filepath, {"mtildeq": float(mass)})
        evaluations += 1
        summary = result["experimental_summary"]
        return float(summary["most_constraining_ratio"]), result

    benchmark_ratio, benchmark_result = evaluate(benchmark_mass)
    if benchmark_ratio >= 1.0:
        summary = benchmark_result["experimental_summary"]
        return {
            "scenario": "SUSY SO10",
            "search_kind": "minimum-increased-sfermion-mass",
            "input_file": metadata["input_file"],
            "flavor_structure_fixed": True,
            "benchmark_sfermion_mass_GeV": benchmark_mass,
            "minimum_sfermion_mass_GeV": benchmark_mass,
            "most_constraining_channel": summary["most_constraining_channel"],
            "bound_ratio_at_minimum": benchmark_ratio,
            "classification_at_minimum": benchmark_result["classification"],
            "benchmark_already_passes": True,
            "evaluations": evaluations,
        }

    lower_mass = benchmark_mass
    lower_ratio = benchmark_ratio
    upper_mass = benchmark_mass
    upper_ratio = benchmark_ratio
    upper_result = benchmark_result
    while upper_ratio < 1.0 and upper_mass < search_upper:
        lower_mass, lower_ratio = upper_mass, upper_ratio
        upper_mass = min(upper_mass * 2.0, search_upper)
        upper_ratio, upper_result = evaluate(upper_mass)

    if upper_ratio < 1.0:
        return {
            "scenario": "SUSY SO10",
            "search_kind": "minimum-increased-sfermion-mass",
            "input_file": metadata["input_file"],
            "flavor_structure_fixed": True,
            "benchmark_sfermion_mass_GeV": benchmark_mass,
            "benchmark_bound_ratio": benchmark_ratio,
            "search_upper_mass_GeV": search_upper,
            "bound_ratio_at_search_upper": upper_ratio,
            "threshold_found": False,
            "evaluations": evaluations,
            "error": "No safe point was found within the calculator's allowed sfermion-mass range.",
        }

    while (upper_mass - lower_mass) / upper_mass > tolerance:
        midpoint = float(np.sqrt(lower_mass * upper_mass))
        midpoint_ratio, midpoint_result = evaluate(midpoint)
        if midpoint_ratio >= 1.0:
            upper_mass, upper_ratio, upper_result = midpoint, midpoint_ratio, midpoint_result
        else:
            lower_mass, lower_ratio = midpoint, midpoint_ratio

    plot_upper = min(search_upper, max(upper_mass * 1.5, benchmark_mass * 2.0))
    grid_masses = np.geomspace(benchmark_mass, plot_upper, grid_points)
    grid_results = []
    rows = []
    for mass in grid_masses:
        ratio, point_result = evaluate(float(mass))
        grid_results.append(point_result)
        row = {
            "m_sfermion_GeV": float(mass),
            "most_constraining_bound_ratio": float(ratio),
            "classification": str(point_result["classification"]),
        }
        for channel, lifetime in point_result["lifetimes_years"].items():
            row[f"lifetime::{channel}"] = float(lifetime)
        rows.append(row)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    digest = _cache_key(
        "susy_so10_sfermion_threshold_grid",
        {"filepath": metadata["input_file"], "threshold": upper_mass},
    )[:8]
    data_path = SCAN_RESULTS_DIR / f"susy_so10_sfermion_threshold_grid_{stamp}_{digest}.h5"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_hdf(data_path, key="results", mode="w")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    # The total lifetime is retained in numerical results, but it is not a
    # decay-channel observable and must never appear in a plot.
    channels = [
        channel
        for channel in grid_results[0]["lifetimes_years"]
        if str(channel).strip().lower() != "total"
    ]
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(channels), 1)))
    for color, channel in zip(colors, channels):
        lifetimes = [float(result["lifetimes_years"][channel]) for result in grid_results]
        ax.plot(
            grid_masses,
            lifetimes,
            lw=1.8,
            color=color,
            label=_channel_latex_label(channel),
        )
        comparison = grid_results[0].get("experimental_comparison", {}).get(channel)
        if comparison and comparison.get("current_bound_years") is not None:
            ax.axhline(
                float(comparison["current_bound_years"]),
                color=color,
                ls=":",
                lw=1.0,
                alpha=0.75,
            )

    ax.axvline(
        upper_mass,
        color="black",
        ls="--",
        lw=1.5,
        label=fr"first all-channel crossing: {upper_mass:.3e} GeV",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Sfermion mass $m_{\widetilde f}$ [GeV]")
    ax.set_ylabel(r"Partial lifetime $\tau/B$ [yr]")
    ax.set_title(r"SUSY $SO(10)$ fixed-flavor sfermion-mass threshold")
    ax.grid(True, which="both", ls=":", alpha=0.30)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="best")
    fig.tight_layout()
    plot_path = OUTPUT_DIR / f"susy_so10_sfermion_threshold_{stamp}_{digest}.png"
    fig.savefig(plot_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Restore the stored latest point to the threshold rather than the final
    # visualization-grid endpoint.
    upper_ratio, upper_result = evaluate(upper_mass)
    summary = upper_result["experimental_summary"]
    output = {
        "scenario": "SUSY SO10",
        "search_kind": "minimum-increased-sfermion-mass",
        "input_file": metadata["input_file"],
        "input_file_sha256": _file_sha256(metadata["input_file"]),
        "flavor_structure_fixed": True,
        "varied_parameter": "m_sfermion",
        "held_fixed": [
            "M_T", "M_wino", "mu", "h", "f", "g", "x", "y", "tanbeta",
            "theta_u", "phi_u", "psi_u", "theta_d", "phi_d", "psi_d", "vL", "vR",
        ],
        "benchmark_sfermion_mass_GeV": benchmark_mass,
        "benchmark_bound_ratio": benchmark_ratio,
        "minimum_sfermion_mass_GeV": upper_mass,
        "relative_tolerance": tolerance,
        "last_excluded_mass_GeV": lower_mass,
        "bound_ratio_below_threshold": lower_ratio,
        "bound_ratio_at_minimum": upper_ratio,
        "most_constraining_channel": summary["most_constraining_channel"],
        "classification_at_minimum": upper_result["classification"],
        "benchmark_already_passes": False,
        "threshold_found": True,
        "evaluations": evaluations,
        "adaptive_grid_points": grid_points,
        "adaptive_grid_range_GeV": [benchmark_mass, float(plot_upper)],
        "grid_data_filepath": str(data_path),
        "plot_filepath": str(plot_path),
        "threshold_scope": (
            "Minimum increased sfermion mass along the one-parameter trajectory "
            "with every other benchmark input fixed."
        ),
    }
    artifact = _register_artifact(
        kind="susy-so10-sfermion-threshold",
        payload={
            "plot_filepath": str(plot_path),
            "grid_data_filepath": str(data_path),
            "threshold_result": output,
        },
        summary=output,
    )
    output["artifact_id"] = artifact["artifact_id"]
    output["artifact_kind"] = artifact["kind"]
    _set_last_pd_result(
        output,
        (
            "SUSY SO(10) fixed-flavor sfermion threshold analysis: "
            f"minimum={upper_mass:.16e} GeV; "
            f"plot={plot_path}; grid={data_path}."
        ),
    )
    return output


@tool
@_with_matrix_elements_file
def compute_susy_so10_point(
    M_T: float | None = None,
    M_wino: float | None = None,
    mu: float | None = None,
    m_sfermion: float | None = None,
    tanbeta: float | None = None,
    theta_u: float | None = None,
    phi_u: float | None = None,
    psi_u: float | None = None,
    theta_d: float | None = None,
    phi_d: float | None = None,
    psi_d: float | None = None,
    vL: float | None = None,
    vR: float | None = None,
    h: list[list[float]] | None = None,
    f: list[list[float]] | None = None,
    g_real: list[list[float]] | None = None,
    g_imag: list[list[float]] | None = None,
    triplet_x: dict[str, float] | None = None,
    triplet_y: dict[str, float] | None = None,
    matrix_elements_file: str | None = None,
    bounds_file: str | None = None,
) -> dict:
    """Compute a supersymmetric SO(10) proton-decay point. Omitted inputs use the documented benchmark defaults."""
    values = {
        "M_T": M_T, "mtildeW": M_wino, "mu": mu, "mtildeq": m_sfermion,
        "tanbeta": tanbeta, "theta_u": theta_u, "phi_u": phi_u, "psi_u": psi_u,
        "theta_d": theta_d, "phi_d": phi_d, "psi_d": psi_d, "vL": vL, "vR": vR,
    }
    overrides = {key: float(value) for key, value in values.items() if value is not None}
    if h is not None:
        overrides["h"] = np.asarray(h, dtype=complex)
    if f is not None:
        overrides["f"] = np.asarray(f, dtype=complex)
    if g_real is not None or g_imag is not None:
        default_g = _default_susy_so10_inputs()["g"]
        real = np.asarray(g_real, dtype=float) if g_real is not None else default_g.real
        imag = np.asarray(g_imag, dtype=float) if g_imag is not None else default_g.imag
        overrides["g"] = real + 1j * imag
    defaults = _default_susy_so10_inputs()
    if triplet_x is not None:
        overrides["x"] = {**defaults["x"], **{str(k): float(v) for k, v in triplet_x.items()}}
    if triplet_y is not None:
        overrides["y"] = {**defaults["y"], **{str(k): float(v) for k, v in triplet_y.items()}}
    result = _compute_susy_so10_point(overrides)
    result["point_kind"] = "custom point"
    _set_last_pd_result(result, _format_susy_so10_result(result))
    return _public_susy_result("SO10", result, "custom point")


@tool
def get_latest_proton_decay_result() -> dict:
    """Return the exact latest proton-decay tool result for numerical follow-up explanations or reformatting."""
    global LAST_PD_RESULT, LAST_PD_RESULT_TEXT
    saved_at_utc = None
    if LAST_PD_RESULT is None:
        restored_result, restored_text, saved_at_utc = _load_last_pd_result_from_disk()
        LAST_PD_RESULT = restored_result
        LAST_PD_RESULT_TEXT = restored_text
    if LAST_PD_RESULT is None:
        return {"error": "No proton-decay result is stored. Run the requested computation first."}
    return {
        "instruction": "Copy numerical values exactly; do not replace, infer, rescale, or recompute them.",
        "state_source": "persisted" if saved_at_utc else "current_process",
        "saved_at_utc": saved_at_utc,
        "result": _jsonable(LAST_PD_RESULT),
    }


@tool
@_with_matrix_elements_file
def run_nonsusy_proton_decay_mcmc(
    group: Literal["SU5", "SO10"],
    n_samples: int = 500,
    MGUT: float | None = None,
    gGUT: float | None = None,
    filepath: str | None = None,
    vary: list[str] | None = None,
    seed: int = 24680,
    angle_width: float = 2.0e-4,
    phase_width: float = 2.0e-4,
    matrix_width: float = 2.0e-4,
    mgut_log_width: float = 2.0e-3,
    ggut_width: float = 1.0e-3,
    burn_in: int = 0,
    thinning: int = 1,
    n_jobs: int = -1,
    matrix_elements_file: str | None = None,
    bounds_file: str | None = None,
) -> dict:
    """Run a deterministic non-SUSY proton-decay MCMC from a file flavor point.

    The packaged input file for the selected group supplies the initial flavor
    point when filepath is omitted. Explicit MGUT and gGUT replace the file
    values. By default only flavor is varied; include MGUT or gGUT in ``vary``
    only when the user explicitly asks to sample them.
    """
    normalized_group = str(group).upper().replace("(", "").replace(")", "")
    if normalized_group not in {"SU5", "SO10"}:
        return {"error": "group must be SU5 or SO10."}
    requested_vary = set(vary or ["flavor"])
    allowed = {"flavor", "MGUT", "gGUT"}
    unknown = requested_vary - allowed
    if unknown:
        return {"error": f"Unknown MCMC vary entries: {', '.join(sorted(unknown))}. Use flavor, MGUT, gGUT."}

    input_file = filepath or _nonsusy_mcmc_default_file(normalized_group)
    try:
        params, metadata = _load_benchmark_parameters_from_file(input_file, "nonSUSY", normalized_group)
        schema = metadata["schema"]
        initial_mgut = float(MGUT if MGUT is not None else params["MGUT"])
        initial_ggut = float(gGUT if gGUT is not None else params["gGUT"])
        if initial_mgut <= 0.0 or initial_ggut <= 0.0:
            return {"error": "MGUT and gGUT must be positive."}

        common = {
            "n_samples": n_samples,
            "vary": requested_vary,
            "seed": seed,
            "mgut_log_width": mgut_log_width,
            "ggut_width": ggut_width,
            "burn_in": burn_in,
            "thinning": thinning,
            "n_jobs": n_jobs,
        }
        if schema == MASTER56_SCHEMA:
            initial_vector = [float(params["matrix_parameters"][name]) for name in MASTER56_PARAMETER_ORDER]
            result = _run_master56_mcmc(
                "nonSUSY", normalized_group, initial_vector, initial_mgut, initial_ggut,
                angle_width=angle_width, phase_width=phase_width, **common,
            )
        elif schema == FULL_MATRICES_SCHEMA:
            result = _run_full_matrix_mcmc(
                "nonSUSY", normalized_group, _core_matrices_from_params(params),
                initial_mgut, initial_ggut, matrix_width=matrix_width,
                input_convention=metadata["input_convention"], **common,
            )
        else:
            return {"error": f"MCMC cannot run for schema {schema!r}."}
    except Exception as exc:
        return {"error": f"Could not run MCMC: {exc}"}

    result["baseline_input_file"] = metadata["input_file"]
    result["fixed_parameters"] = [name for name in ("MGUT", "gGUT") if name not in requested_vary]
    answer = _format_mcmc_result(result)
    _set_last_pd_result(result, answer)
    return result


@tool
@_with_matrix_elements_file
def run_susy_proton_decay_mcmc(
    group: Literal["SU5", "SO10"],
    n_samples: int = 500,
    filepath: str | None = None,
    vary: list[str] | None = None,
    seed: int = 24680,
    log_width: float = 2.0e-3,
    linear_width: float = 1.0e-3,
    matrix_width: float = 2.0e-4,
    phase_width: float = 2.0e-4,
    burn_in: int = 0,
    thinning: int = 1,
    n_jobs: int = -1,
    matrix_elements_file: str | None = None,
    bounds_file: str | None = None,
) -> dict:
    """Run a deterministic SUSY SU(5) or SO(10) proton-decay MCMC.

    The packaged input file for the selected group supplies the initial point
    when filepath is omitted. By default the supported SUSY parameter set for
    that group is varied; ``vary`` may select a supported subset.
    """
    normalized_group = str(group).upper().replace("(", "").replace(")", "")
    if normalized_group not in {"SU5", "SO10"}:
        return {"error": "group must be SU5 or SO10."}

    requested_vary = [
        _normalize_susy_key(name)
        for name in (vary or _susy_mcmc_default_vary(normalized_group))
    ]
    allowed = set(_susy_mcmc_parameter_order(normalized_group))
    unknown = [name for name in requested_vary if name not in allowed]
    if unknown:
        return {"error": f"Unknown SUSY MCMC vary entries: {', '.join(unknown)}."}

    input_file = filepath or _susy_mcmc_default_file(normalized_group)
    try:
        params, metadata = _load_susy_benchmark_parameters_from_file(
            input_file, normalized_group
        )
        result = _run_susy_mcmc(
            normalized_group,
            params,
            metadata["input_file"],
            n_samples=n_samples,
            vary=requested_vary,
            seed=seed,
            log_width=log_width,
            linear_width=linear_width,
            matrix_width=matrix_width,
            phase_width=phase_width,
            burn_in=burn_in,
            thinning=thinning,
            n_jobs=n_jobs,
        )
    except Exception as exc:
        return {"error": f"Could not run MCMC: {exc}"}

    answer = _format_mcmc_result(result)
    _set_last_pd_result(result, answer)
    return result


@tool
@_with_matrix_elements_file
def run_mcmc_around_last_point(
    n_samples: int = 500,
    vary: list[str] | None = None,
    seed: int = 24680,
    angle_width: float = 2.0e-4,
    phase_width: float = 2.0e-4,
    matrix_width: float = 2.0e-4,
    mgut_log_width: float = 2.0e-3,
    ggut_width: float = 1.0e-3,
    burn_in: int = 0,
    thinning: int = 1,
    n_jobs: int = -1,
    matrix_elements_file: str | None = None,
    bounds_file: str | None = None,
) -> dict:
    """
    Run deterministic Metropolis MCMC around the most recent benchmark point.
    Supports non-SUSY master56/full_matrices points and SUSY SU5/SO10 input
    points. Non-SUSY defaults to flavor variation; SUSY defaults to the supported
    SUSY parameter list for the active group.

    Args:
        n_samples: Number of retained MCMC samples.
        vary: Variables to vary. Non-SUSY accepts "flavor", "MGUT", "gGUT". SUSY accepts its benchmark parameter names.
        seed: Random seed for reproducibility.
        angle_width: Gaussian proposal width for the first 15 flavor angles.
        phase_width: Gaussian proposal width for phase parameters; for SUSY SU5 this applies to phi13 and phi23.
        matrix_width: Small unitary proposal width for full-matrix points.
        mgut_log_width: Gaussian proposal width in log10(MGUT).
        ggut_width: Gaussian proposal width for gGUT.
        burn_in: Number of initial proposals to discard.
        thinning: Keep one sample every thinning proposals after burn-in.
        n_jobs: Parallel worker count. Defaults to -1, meaning all available CPU cores.
    """
    if not LAST_PD_RESULT:
        return {"error": "Run a non-SUSY input file first, e.g. 'file /input/nonsusy-SU5.txt' or 'file /input/nonsusy-SO10.txt'."}

    scenario = str(LAST_PD_RESULT.get("scenario", "nonSUSY SU5")).split()
    framework = scenario[0] if scenario and scenario[0] in {"SUSY", "nonSUSY"} else "nonSUSY"
    group = scenario[1] if len(scenario) > 1 and scenario[1] in {"SU5", "SO10"} else "SU5"
    if framework == "SUSY" and group in {"SU5", "SO10"}:
        if not LAST_PD_RESULT.get("inputs"):
            return {"error": "Run or load a SUSY benchmark point first."}
        requested_susy_vary = [_normalize_susy_key(name) for name in (vary or _susy_mcmc_default_vary(group))]
        allowed_susy = set(_susy_mcmc_parameter_order(group))
        unknown_susy = [name for name in requested_susy_vary if name not in allowed_susy]
        if unknown_susy:
            return {"error": f"Unknown SUSY MCMC vary entries: {', '.join(unknown_susy)}."}
        try:
            return _run_susy_mcmc(
                group,
                LAST_PD_RESULT["inputs"],
                str(LAST_PD_RESULT.get("input_file", "last SUSY point")),
                n_samples=n_samples,
                vary=requested_susy_vary,
                seed=seed,
                log_width=mgut_log_width,
                linear_width=ggut_width,
                matrix_width=matrix_width,
                phase_width=phase_width,
                burn_in=burn_in,
                thinning=thinning,
                n_jobs=n_jobs,
            )
        except Exception as exc:
            return {"error": f"Could not run MCMC: {exc}"}
    if framework != "nonSUSY":
        return {"error": "MCMC is currently implemented for non-SUSY benchmark points and SUSY SU5/SO10 parameter files."}

    requested_vary = set(vary or ["flavor"])
    allowed = {"flavor", "MGUT", "gGUT"}
    unknown = requested_vary - allowed
    if unknown:
        return {"error": f"Unknown MCMC vary entries: {', '.join(sorted(unknown))}. Use flavor, MGUT, gGUT."}

    try:
        if LAST_PD_RESULT.get("schema") == MASTER56_SCHEMA and LAST_PD_RESULT.get("master56_values"):
            return _run_master56_mcmc(
                framework,
                group,
                LAST_PD_RESULT["master56_values"],
                float(LAST_PD_RESULT["MGUT"]),
                float(LAST_PD_RESULT["gGUT"]),
                n_samples=n_samples,
                vary=requested_vary,
                seed=seed,
                angle_width=angle_width,
                phase_width=phase_width,
                mgut_log_width=mgut_log_width,
                ggut_width=ggut_width,
                burn_in=burn_in,
                thinning=thinning,
                n_jobs=n_jobs,
            )
        if LAST_PD_RESULT.get("schema") == FULL_MATRICES_SCHEMA and LAST_PD_RESULT.get("core_matrices"):
            return _run_full_matrix_mcmc(
                framework,
                group,
                LAST_PD_RESULT["core_matrices"],
                float(LAST_PD_RESULT["MGUT"]),
                float(LAST_PD_RESULT["gGUT"]),
                n_samples=n_samples,
                vary=requested_vary,
                seed=seed,
                matrix_width=matrix_width,
                mgut_log_width=mgut_log_width,
                ggut_width=ggut_width,
                burn_in=burn_in,
                thinning=thinning,
                input_convention=str(LAST_PD_RESULT.get("input_convention", "")),
                n_jobs=n_jobs,
            )
        return {"error": "The last point is not an MCMC-supported schema. Use master56, full_matrices, susy_su5, or susy_so10."}
    except Exception as exc:
        return {"error": f"Could not run MCMC: {exc}"}


@tool
@_with_matrix_elements_file
def run_random_scan(
    group: Literal["SU5", "SO10"],
    n_points: int = 200,
    n_jobs: int = -1,
    MGUT_min: float = 1e15,
    MGUT_max: float = 1e17,
    gGUT_min: float = 0.5,
    gGUT_max: float = 0.8,
    user_config: dict | None = None,
    matrix_elements_file: str | None = None,
    bounds_file: str | None = None,
) -> dict:
    """
    Scan one explicitly selected non-SUSY model by drawing random parameter sets.
    """
    n_points = min(int(n_points), 2000)

    base_config = {
        "nonsusy_model": group,
        "MGUT": (float(MGUT_min), float(MGUT_max)),
        "gGUT": (float(gGUT_min), float(gGUT_max)),
    }

    if user_config:
        base_config.update(user_config)
    # The explicit tool argument is authoritative; nested config cannot switch models.
    base_config["nonsusy_model"] = group

    scan_output = run_scan(
        n_points=n_points,
        n_jobs=n_jobs,
        user_config=base_config,
        save_path=_scan_filepath("random_scan", {
            "n_points": n_points,
            "MGUT_min": MGUT_min,
            "MGUT_max": MGUT_max,
            "gGUT_min": gGUT_min,
            "gGUT_max": gGUT_max,
            "user_config": user_config,
        }),
    )

    results = scan_output["results"]
    scan_filepath = scan_output["hdf5_file"]
    summary = _summarize_scan(results, mode="random")

    artifact = _register_artifact(
        kind="random-scan",
        payload={
            "results": results,
            "scan_filepath": scan_filepath,
            "config": {
                "n_points": n_points,
                "n_jobs": n_jobs,
                "MGUT_min": MGUT_min,
                "MGUT_max": MGUT_max,
                "gGUT_min": gGUT_min,
                "gGUT_max": gGUT_max,
                "user_config": user_config,
            },
        },
        summary=summary,
    )

    return {
        "scenario": f"nonSUSY {group}",
        "formula_module": "nonsusy_su5_formula" if group == "SU5" else "nonsusy_so10_formula",
        **summary,
        "scan_filepath": scan_filepath,
        "requested_MGUT_range_GeV": [float(MGUT_min), float(MGUT_max)],
        "requested_gGUT_range": [float(gGUT_min), float(gGUT_max)],
        "artifact_id": artifact["artifact_id"],
        "artifact_kind": artifact["kind"],
    }


@tool
@_with_matrix_elements_file
def run_fixed_flavor_scan_from_file(
    group: Literal["SU5", "SO10"],
    filepath: str,
    n_points: int = 200,
    n_jobs: int = -1,
    MGUT_min: float = 1e15,
    MGUT_max: float = 1e17,
    gGUT_min: float = 0.5,
    gGUT_max: float = 0.8,
    seed: int = 24680,
    matrix_elements_file: str | None = None,
    bounds_file: str | None = None,
) -> dict:
    """Scan MGUT and gGUT while keeping all flavor matrices from one non-SUSY benchmark file exactly fixed."""
    normalized_group = _normalize_pd_group(group) or str(group).upper()
    if normalized_group not in {"SU5", "SO10"}:
        raise ValueError("Fixed-flavor scan group must be 'SU5' or 'SO10'.")

    n_points = min(int(n_points), 2000)
    if n_points < 1:
        raise ValueError("n_points must be at least 1.")

    bounds = np.asarray([MGUT_min, MGUT_max, gGUT_min, gGUT_max], dtype=float)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("Scan bounds must be finite numbers.")
    if MGUT_min <= 0 or MGUT_max < MGUT_min:
        raise ValueError("Require 0 < MGUT_min <= MGUT_max.")
    if gGUT_min <= 0 or gGUT_max < gGUT_min:
        raise ValueError("Require 0 < gGUT_min <= gGUT_max.")

    params, metadata = _load_benchmark_parameters_from_file(
        filepath, "nonSUSY", normalized_group
    )
    matrices = {
        name: np.array(params[name], dtype=complex, copy=True)
        for name in SO10_MATRIX_INPUT_NAMES
    }
    flavor_digest = _matrix_digest(matrices)
    rng = np.random.default_rng(int(seed))
    mgut_values = rng.uniform(float(MGUT_min), float(MGUT_max), n_points)
    ggut_values = rng.uniform(float(gGUT_min), float(gGUT_max), n_points)

    points = []
    for mgut, ggut in zip(mgut_values, ggut_values):
        point = {
            "MGUT": float(mgut),
            "gGUT": float(ggut),
            "nonsusy_model": normalized_group,
            "matrix_parameters": {},
            "derived_matrices": {},
        }
        point.update({name: matrix.copy() for name, matrix in matrices.items()})
        points.append(point)

    # Guard the defining invariant before any expensive parallel evaluation.
    if any(
        _matrix_digest({name: point[name] for name in SO10_MATRIX_INPUT_NAMES}) != flavor_digest
        for point in points
    ):
        raise RuntimeError("Internal error: flavor matrices changed while constructing the scan.")

    results = evaluate_points_parallel(points, n_jobs=n_jobs)
    summary = _summarize_scan(results, mode="fixed-flavor-random")
    scan_filepath = _scan_filepath("fixed_flavor_scan", {
        "group": normalized_group,
        "input_file_sha256": _file_sha256(metadata["input_file"]),
        "flavor_matrix_digest": flavor_digest,
        "n_points": n_points,
        "MGUT_min": MGUT_min,
        "MGUT_max": MGUT_max,
        "gGUT_min": gGUT_min,
        "gGUT_max": gGUT_max,
        "seed": int(seed),
    })
    save_scan_results(results, scan_filepath)

    artifact = _register_artifact(
        kind="fixed-flavor-scan",
        payload={
            "results": results,
            "scan_filepath": scan_filepath,
            "config": {
                "group": normalized_group,
                "input_file": metadata["input_file"],
                "n_points": n_points,
                "n_jobs": n_jobs,
                "MGUT_min": float(MGUT_min),
                "MGUT_max": float(MGUT_max),
                "gGUT_min": float(gGUT_min),
                "gGUT_max": float(gGUT_max),
                "seed": int(seed),
                "flavor_structure_fixed": True,
                "flavor_matrix_digest": flavor_digest,
            },
        },
        summary=summary,
    )

    return {
        "scenario": f"nonSUSY {normalized_group}",
        "formula_module": "nonsusy_su5_formula" if normalized_group == "SU5" else "nonsusy_so10_formula",
        **summary,
        "scan_filepath": scan_filepath,
        "input_file": metadata["input_file"],
        "input_file_sha256": _file_sha256(metadata["input_file"]),
        "input_schema": metadata["schema"],
        "flavor_structure_fixed": True,
        "flavor_matrix_digest": flavor_digest,
        "sampled_parameters": ["MGUT", "gGUT"],
        "requested_MGUT_range_GeV": [float(MGUT_min), float(MGUT_max)],
        "requested_gGUT_range": [float(gGUT_min), float(gGUT_max)],
        "seed": int(seed),
        "artifact_id": artifact["artifact_id"],
        "artifact_kind": artifact["kind"],
    }


@tool
@_with_matrix_elements_file
def run_grid_scan(
    group: Literal["SU5", "SO10"],
    MGUT_min: float = 1e13, MGUT_max: float = 1e17, MGUT_n: int = 10,
    gGUT_min: float = 0.4,  gGUT_max: float = 0.8,  gGUT_n: int = 10,
    n_jobs: int = -1,
    user_config: dict | None = None,
    matrix_elements_file: str | None = None,
    bounds_file: str | None = None,
) -> dict:
    """
    Scan an explicitly selected non-SUSY SU(5) or SO(10) grid over MGUT and gGUT.
    Use when the user wants a structured (MGUT, gGUT) exclusion map.
    """
    MGUT_vals = np.linspace(float(MGUT_min), float(MGUT_max), int(MGUT_n))
    gGUT_vals = np.linspace(float(gGUT_min), float(gGUT_max), int(gGUT_n))

    base_config = dict(user_config or {})
    base_config["nonsusy_model"] = group
    parameter_generator = (
        generate_parameters_with_ckm_pmns
        if base_config.get("derive_dl_nl_from_ckm_pmns")
        else generate_parameters
    )

    points = []
    for MGUT_val, gGUT_val in itertools.product(MGUT_vals, gGUT_vals):
        params = next(parameter_generator(1, user_config=base_config))
        params["MGUT"] = MGUT_val
        params["gGUT"] = gGUT_val
        params["nonsusy_model"] = group
        points.append(params)

    results = evaluate_points_parallel(points, n_jobs=n_jobs)

    grid_config = {
        "MGUT": [MGUT_min, MGUT_max, MGUT_n],
        "gGUT": [gGUT_min, gGUT_max, gGUT_n],
        "n_jobs": n_jobs,
        "user_config": user_config,
    }

    summary = _summarize_scan(results, mode="grid", grid_config=grid_config)
    scan_filepath = _scan_filepath("grid_scan", summary)
    save_scan_results(results, scan_filepath)

    artifact = _register_artifact(
        kind="grid-scan",
        payload={
            "results": results,
            "grid_config": grid_config,
            "scan_filepath": scan_filepath,
        },
        summary=summary,
    )

    return {
        "scenario": f"nonSUSY {group}",
        "formula_module": "nonsusy_su5_formula" if group == "SU5" else "nonsusy_so10_formula",
        **summary,
        "scan_filepath": scan_filepath,
        "requested_MGUT_range_GeV": [float(MGUT_min), float(MGUT_max)],
        "requested_gGUT_range": [float(gGUT_min), float(gGUT_max)],
        "artifact_id": artifact["artifact_id"],
        "artifact_kind": artifact["kind"],
    }


@tool
@_with_matrix_elements_file
def retrieve_experimental_bound(channel: str = "p->pi0 e+", kind: str = "current", bounds_file: str | None = None) -> dict:
    """
    Retrieve a current bound or future sensitivity from the built-in table,
    or from an explicitly selected bounds_file, in years with source metadata.
    """
    value = get_experimental_limit(channel, kind)
    metadata = get_extracted_limit_metadata(channel, kind)
    if value is None:
        return {"error": f"No {kind} bound found for {channel}."}
    return {
        "channel": channel,
        "kind": kind,
        "value_years": float(value),
        "value_1e34yr": float(value) / 1.0e34,
        "source": "bounds_file" if current_bounds_configuration() else "static_table",
        "metadata": metadata,
    }


@tool
@_with_matrix_elements_file
def classify_point(lifetimes: dict, bounds_file: str | None = None) -> dict:
    """
    Classify a lifetime dict against experimental bounds.
    Returns per-channel status and most constrained channel.

    Args:
        lifetimes: Dict of channel -> lifetime in years.
                   Keys: 'p->pi0 e+', 'p->pi0 mu+', 'p->K0 e+', 'p->K0 mu+',
                         'p->eta e+', 'p->eta mu+', 'p->pi+ nu', 'p->K+ nu'
    """
    classification = classify_model(lifetimes)
    details = {}

    for channel, tau in lifetimes.items():
        bound = get_experimental_limit(channel, "current")
        future = get_experimental_limit(channel, "future")
        if bound and tau < bound:
            status = "EXCLUDED"
        elif future and tau < future:
            status = "testable"
        else:
            status = "safe"

        details[channel] = {
            "tau_1e34yr":            tau / 1e34,
            "current_bound_1e34yr":  bound  / 1e34 if bound  else None,
            "future_sens_1e34yr":    future / 1e34 if future else None,
            "ratio_tau_over_bound":  tau / bound   if bound  else None,
            "ratio_tau_over_future": tau / future  if future else None,
            "status": status,
        }

    ratios = {ch: v["ratio_tau_over_bound"]
              for ch, v in details.items()
              if v["ratio_tau_over_bound"] is not None}
    most_constrained = min(ratios, key=ratios.get) if ratios else None

    return {
        KEY_CLASS: classification,
        "most_constrained_channel":  most_constrained,
        "ratio_at_most_constrained": ratios.get(most_constrained),
        "per_channel": details,
    }


@tool
@_with_matrix_elements_file
def analyse_hdf5_scan(filepath: str = "proton_decay_scan.h5", bounds_file: str | None = None) -> dict:
    """
    Load a saved HDF5 scan file and produce a full statistical summary:
    classification fractions, dominant exclusion channel, viable MGUT range,
    discovery frontier, per-channel stats, and anomalies.
    Use when the user says 'analyse my scan', 'load the HDF5', or
    'what does the scan show'.

    Args:
        filepath: Path to HDF5 file (default 'proton_decay_scan.h5')
    """
    try:
        _, df = _scan_dataframe_from_filepath(filepath)
        if df is None:
            raise ValueError("No readable results table.")
    except Exception as e:
        return {"error": f"Could not load {filepath}: {e}"}

    n_total = len(df)

    # Classification
    class_counts = df[KEY_CLASS].value_counts().to_dict()
    n_excluded = class_counts.get("excluded", 0)
    n_testable = class_counts.get("testable", 0)
    n_safe     = class_counts.get("safe",     0)

    # MGUT column — handle both naming conventions
    mgut_col = KEY_MGUT if KEY_MGUT in df.columns else "MGUT"
    ggut_col = KEY_GGUT

    all_MGUT      = df[mgut_col].values
    viable_MGUT   = df.loc[df[KEY_CLASS] != "excluded", mgut_col].values
    testable_MGUT = df.loc[df[KEY_CLASS] == "testable", mgut_col].values

    # Per-channel columns — HDF5 stores them with '_to_' replacing '->'
    lifetime_cols = {}
    for ch in CHANNELS:
        col = _channel_column(ch, "lifetime")
        if col in df.columns:
            lifetime_cols[ch] = col

    # Per-channel statistics
    exclusion_counts = {}
    channel_stats    = {}

    for ch, col in lifetime_cols.items():
        bound  = get_current_limits().get(ch)
        future = get_future_limits().get(ch)
        taus   = df[col].values

        n_ch_excl = int(np.sum(taus < bound))   if bound  else 0
        n_ch_test = int(np.sum(
            (taus >= (bound or 0)) & (taus < future)
        )) if future and bound else 0

        channel_stats[ch] = {
            "median_1e34yr":         float(np.median(taus)) / 1e34,
            "min_1e34yr":            float(np.min(taus))    / 1e34,
            "max_1e34yr":            float(np.max(taus))    / 1e34,
            "current_bound_1e34yr":  bound  / 1e34 if bound  else None,
            "future_sens_1e34yr":    future / 1e34 if future else None,
            "n_excluded_by_channel": n_ch_excl,
            "n_testable_by_channel": n_ch_test,
        }

        if n_ch_excl > 0:
            exclusion_counts[ch] = n_ch_excl

    dominant_exclusion = (
        max(exclusion_counts, key=exclusion_counts.get)
        if exclusion_counts else "not determinable from stored columns"
    )

    # Anomalies
    anomalies = []
    if lifetime_cols:
        medians = {ch: np.median(df[col].values)
                   for ch, col in lifetime_cols.items()}
        for idx, row in df.iterrows():
            for ch, col in lifetime_cols.items():
                if row[col] > 10 * medians[ch]:
                    anomalies.append({
                        mgut_col: row[mgut_col],
                        ggut_col: row[ggut_col],
                        "channel": ch,
                        "lifetime_1e34yr": row[col] / 1e34,
                        "ratio_to_median": row[col] / medians[ch],
                    })
                    break
            if len(anomalies) >= 10:
                break

    summary = {
        "filepath":  filepath,
        "n_total":   n_total,
        "n_excluded": n_excluded,
        "n_testable": n_testable,
        "n_safe":     n_safe,
        "fraction_excluded": n_excluded / n_total,
        "fraction_testable": n_testable / n_total,
        "fraction_safe":     n_safe     / n_total,
        "MGUT_full_range_GeV":     [float(all_MGUT.min()),      float(all_MGUT.max())],
        "MGUT_viable_range_GeV":   [float(viable_MGUT.min()),   float(viable_MGUT.max())]   if len(viable_MGUT)   > 0 else None,
        "MGUT_testable_range_GeV": [float(testable_MGUT.min()), float(testable_MGUT.max())] if len(testable_MGUT) > 0 else None,
        "dominant_exclusion_channel":    dominant_exclusion,
        "exclusion_counts_per_channel":  exclusion_counts,
        "channel_stats":   channel_stats,
        "n_anomalies":     len(anomalies),
        "anomalies":       anomalies,
        "columns_in_file": list(df.columns),
    }
    artifact = _register_artifact(
        kind="hdf5-analysis",
        payload={"filepath": filepath, "summary": summary},
        summary=summary,
    )
    return {
        **summary,
        "artifact_id": artifact["artifact_id"],
        "artifact_kind": artifact["kind"],
    }


# ================================================================
# SMART FIGURE DISPLAY HELPER
# ================================================================

def _show_figure(fig, savefig: str) -> str:
    """
    Save and display a figure correctly in any environment:
      - Jupyter notebook  → IPython.display (renders inline in the cell)
      - Terminal / script → GUI window via default backend
    Always saves to disk regardless of environment.
    """
    import matplotlib.pyplot as plt

    output_path = _timestamped_result_path(savefig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    # An API server has no single interactive viewer. Opening GUI windows from
    # request/background threads makes one saved plot appear multiple times.
    # In server mode the verified file is the sole plot output.
    if os.environ.get("PROTON_AGENT_SERVER_MODE", "").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        plt.close(fig)
        return str(output_path)

    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None and hasattr(shell, "kernel"):
            # Inside Jupyter — render inline
            from IPython.display import display as ipy_display
            ipy_display(fig)
            plt.close(fig)
            return str(output_path)
    except ImportError:
        pass

    # Terminal — open GUI window if display is available
    try:
        plt.show(block=False)
        plt.pause(0.5)
    except Exception:
        pass   # headless server: file is already saved.
    plt.close(fig)
    return str(output_path)


# ================================================================
# PLOTTING TOOLS
# ================================================================

@tool
@_with_matrix_elements_file
def plot_lifetime_vs_MGUT(
    channel: str = "p->pi0 e+",
    scan_artifact_id: str = "latest_scan",
    savefig: str = "lifetime_vs_MGUT.png",
    bounds_file: str | None = None,
) -> dict:
    """
    Plot the partial lifetime tau/B for one channel as a function of MGUT
    using the latest saved scan data by default. It does not launch a new scan.

    Args:
        channel:   Decay channel to plot. One of: 'p->pi0 e+', 'p->pi0 mu+',
                   'p->K0 e+', 'p->K0 mu+', 'p->eta e+', 'p->eta mu+',
                   'p->pi+ nu', 'p->K+ nu'
        scan_artifact_id: Scan artifact to plot from. Default is latest saved scan.
        savefig:   Output filename (default 'lifetime_vs_MGUT.png')
    """
    import matplotlib.pyplot as plt

    if channel not in CHANNELS:
        return {"error": f"Unknown channel '{channel}'. Valid: {CHANNELS}"}

    scan_record, df = _scan_dataframe_from_artifact(scan_artifact_id)
    if df is None:
        return {
            "error": "No saved scan data is available for plotting. Run `run_random_scan` or `run_grid_scan` explicitly first.",
            "requested_channel": channel,
        }

    lifetime_col = _channel_column(channel, "lifetime")
    if lifetime_col not in df.columns:
        return {
            "error": f"Saved scan does not contain column '{lifetime_col}'.",
            "available_columns": list(df.columns),
        }

    MGUT_vals = df["MGUT"].to_numpy()
    lifetime_vals = df[lifetime_col].to_numpy()
    gGUT_values = df["gGUT"].to_numpy()
    requested_mgut_range = _requested_axis_range(scan_record["artifact_id"], "MGUT")

    # ── Experimental lines ───────────────────────────────────────────
    bound  = get_current_limits().get(channel)
    future = get_future_limits().get(channel)

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    plot_data = {"MGUT": MGUT_vals, "lifetime": lifetime_vals}
    if KEY_CLASS in df.columns:
        plot_data[KEY_CLASS] = df[KEY_CLASS].to_numpy()
    plot_frame = pd.DataFrame(plot_data).drop_duplicates()
    unique_mgut = int(plot_frame["MGUT"].nunique())
    single_mgut = unique_mgut == 1
    if KEY_CLASS in plot_frame.columns:
        _scatter_classification_points(
            ax,
            (
                plot_frame["MGUT"].to_numpy() / 1e15,
                plot_frame["lifetime"].to_numpy() / 1e34,
            ),
            plot_frame[KEY_CLASS],
        )
    else:
        ax.scatter(
            plot_frame["MGUT"].to_numpy() / 1e15,
            plot_frame["lifetime"].to_numpy() / 1e34,
            s=70 if len(plot_frame) == 1 else 10,
            alpha=0.85 if len(plot_frame) == 1 else 0.45,
            color="black",
            label="Saved MCMC point" if len(plot_frame) == 1 else "Saved MCMC samples",
            zorder=1,
        )

    if bound:
        ax.axhline(
            bound / 1e34,
            color="crimson", lw=1.8, ls="--",
            label=f"{'Selected current bound' if current_bounds_configuration() else 'Super-K bound'} ({bound/1e34:.1f} × 10³⁴ yr)",
        )
        # shade excluded region
        ax.axhspan(0, bound / 1e34, alpha=0.08, color="crimson")

    if future:
        ax.axhline(
            future / 1e34,
            color="darkorange", lw=1.8, ls=":",
            label=f"{'Selected future sensitivity' if current_bounds_configuration() else 'HK sensitivity'} ({future/1e34:.1f} × 10³⁴ yr)",
        )

    # ── Labels and formatting ────────────────────────────────────────
    ch_latex = _channel_latex_label(channel)

    ax.set_xlabel(r"$M_{\rm GUT}$  [$10^{15}$ GeV]", fontsize=13)
    ax.set_ylabel(r"$\tau/B$  [$10^{34}$ yr]", fontsize=13)
    ax.set_title(f"Partial lifetime from saved scan: {ch_latex}", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    if requested_mgut_range is not None:
        ax.set_xlim(requested_mgut_range[0] / 1e15, requested_mgut_range[1] / 1e15)
    elif single_mgut:
        center = float(plot_frame["MGUT"].iloc[0]) / 1e15
        padding = max(abs(center) * 0.05, 1.0e-6)
        ax.set_xlim(center - padding, center + padding)
        ax.annotate(
            "single sampled $M_{\\rm GUT}$",
            (center, float(np.median(plot_frame["lifetime"])) / 1e34),
            xytext=(8, 8), textcoords="offset points", fontsize=9,
        )
    fig.tight_layout()

    savefig = _show_figure(fig, savefig)

    summary = {
        "status":   "plot saved and displayed",
        "savefig":  savefig,
        "channel":  channel,
        "n_points": int(len(df)),
        "n_unique_points": int(len(plot_frame)),
        "n_unique_MGUT": unique_mgut,
        "single_MGUT": single_mgut,
        "plot_note": (
            "The saved data contain one MGUT value; the plot shows the available point(s) at that fixed scale."
            if single_mgut else
            "The saved data contain multiple MGUT values."
        ),
        "MGUT_range_GeV": requested_mgut_range or [float(MGUT_vals.min()), float(MGUT_vals.max())],
        "gGUT_range": [float(gGUT_values.min()), float(gGUT_values.max())],
        "source_scan_artifact_id": scan_record["artifact_id"],
        "current_bound_1e34yr":  bound  / 1e34 if bound  else None,
        "future_sens_1e34yr":    future / 1e34 if future else None,
        "lifetime_min_1e34yr":   float(lifetime_vals.min()) / 1e34,
        "lifetime_max_1e34yr":   float(lifetime_vals.max()) / 1e34,
        "lifetime_median_1e34yr":float(np.median(lifetime_vals)) / 1e34,
    }
    artifact = _register_artifact(
        kind="mgut-plot-single-channel",
        payload={
            "channel": channel,
            "MGUT_vals_GeV": MGUT_vals,
            "lifetimes_years": lifetime_vals,
            "gGUT_values": gGUT_values,
            "source_scan_artifact_id": scan_record["artifact_id"],
            "savefig": savefig,
        },
        summary=summary,
    )
    return {
        **summary,
        "artifact_id": artifact["artifact_id"],
        "artifact_kind": artifact["kind"],
    }


@tool
@_with_matrix_elements_file
def plot_saved_scan_quantity(
    quantity: str = "branching",
    channel: str = "p->pi+ nu",
    scan_artifact_id: str = "latest_scan",
    filepath: str | None = None,
    mode: str = "single",
    x_column: str = "MGUT",
    xscale: str = "linear",
    color_by: str = "classification",
    savefig: str = "saved_scan_plot.png",
    bounds_file: str | None = None,
) -> dict:
    """
    Generic plotting tool for quantities already stored in a saved scan.
    Supports lifetime and branching-ratio plots without launching a new scan.

    Args:
        quantity: 'lifetime' or 'branching'
        channel: Decay channel for single-channel plots
        scan_artifact_id: Scan artifact to plot from. Default is latest saved scan.
        filepath: Optional HDF5 scan file; takes precedence over scan_artifact_id.
        mode: 'single' for one channel or 'all' for all channels
        x_column: numeric x-axis column, e.g. 'MGUT', 'gGUT', 'NL_23_abs'
        xscale: 'linear' or 'log'
        color_by: 'classification', 'none', or any saved numeric column
        savefig: Output filename
    """
    import matplotlib.pyplot as plt

    quantity = quantity.strip().lower()
    mode = mode.strip().lower()
    color_by = color_by.strip().lower()

    if quantity not in {"lifetime", "branching"}:
        return {"error": "quantity must be 'lifetime' or 'branching'."}
    if mode not in {"single", "all"}:
        return {"error": "mode must be 'single' or 'all'."}
    if channel not in CHANNELS:
        return {"error": f"Unknown channel '{channel}'. Valid: {CHANNELS}"}
    if xscale not in {"linear", "log"}:
        return {"error": "xscale must be 'linear' or 'log'."}

    if filepath:
        scan_record, df = _scan_dataframe_from_filepath(filepath)
    else:
        scan_record, df = _scan_dataframe_from_artifact(scan_artifact_id)
    if df is None:
        return {
            "error": (f"Cannot read the requested saved scan file {filepath!r} with HDF5 key 'results'."
                      if filepath else "No saved scan data is available for plotting; specify an existing HDF5 file or scan artifact."),
            "filepath": filepath,
        }
    if KEY_CLASS in df.columns and color_by in {"", "none"}:
        color_by = "classification"

    if x_column == "MGUT" and "MGUT" not in df.columns and "scan_index" in df.columns:
        x_column = "scan_index"
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    resolved_x_column, column_error = _resolve_saved_scan_column(x_column, df, numeric_cols)
    if column_error:
        return {
            "error": column_error,
            "available_columns": numeric_cols,
        }
    x_column = resolved_x_column
    resolved_color_by = color_by
    color_values = None
    color_norm = None
    if color_by not in {"classification", "", "none"}:
        resolved_color_by, color_error = _resolve_saved_scan_column(color_by, df, numeric_cols)
        if color_error:
            return {"error": color_error, "available_columns": numeric_cols}
        color_values = df[resolved_color_by].to_numpy(dtype=float)
        finite_color_values = color_values[np.isfinite(color_values)]
        if finite_color_values.size == 0:
            return {"error": f"Color column '{resolved_color_by}' has no finite values."}
        from matplotlib.colors import Normalize
        color_min = float(np.min(finite_color_values))
        color_max = float(np.max(finite_color_values))
        if color_min == color_max:
            padding = max(abs(color_min) * 1.0e-9, 1.0e-12)
            color_min -= padding
            color_max += padding
        color_norm = Normalize(vmin=color_min, vmax=color_max)
    x_lifetime_channel = _channel_from_lifetime_column(x_column)
    x_current_bound = get_current_limits().get(x_lifetime_channel) if x_lifetime_channel else None
    x_future_sensitivity = get_future_limits().get(x_lifetime_channel) if x_lifetime_channel else None

    prefix = "lifetime" if quantity == "lifetime" else "branching"
    ylabel = r"$\tau/B$  [$10^{34}$ yr]" if quantity == "lifetime" else "Branching ratio"
    yscale = "log" if quantity == "lifetime" else "linear"

    xvals = df[x_column].to_numpy()
    if xscale == "log" and np.any(xvals <= 0):
        return {"error": f"x_column '{x_column}' contains non-positive values and cannot use a log scale."}
    mass_columns = {"M_T", "m0", "mtildeq", "M_sfermion", "m_sfermion", "M_wino"}
    x_common_exponent = None
    gGUT_vals = df["gGUT"].to_numpy() if "gGUT" in df.columns else None
    if x_column == "MGUT":
        xplot = xvals / 1e15
        xlabel = r"$M_{\rm GUT}$  [$10^{15}$ GeV]"
        title_x = r"$M_{\rm GUT}$"
        requested_x_range = _requested_axis_range(scan_record["artifact_id"], "MGUT")
    else:
        xplot = xvals
        xlabel = _plot_axis_label(x_column)
        title_x = _plot_axis_label(x_column)
        requested_x_range = _requested_axis_range(scan_record["artifact_id"], "gGUT") if x_column == "gGUT" else None
        finite_positive_x = xvals[np.isfinite(xvals) & (xvals > 0)]
        if (
            x_column in mass_columns
            and finite_positive_x.size
            and np.log10(np.max(finite_positive_x)) - np.log10(np.min(finite_positive_x)) < 1.0
        ):
            x_common_exponent = int(np.floor(np.log10(np.sqrt(
                float(np.min(finite_positive_x)) * float(np.max(finite_positive_x))
            ))))
            xplot = xvals / (10.0 ** x_common_exponent)
            if xlabel.endswith("[GeV]"):
                xlabel = xlabel[:-5].rstrip() + rf" [$10^{{{x_common_exponent}}}$ GeV]"

    plotted_channels = []
    if mode == "single":
        fig, ax = plt.subplots(figsize=(8, 5))
        col = _channel_column(channel, prefix)
        if col not in df.columns:
            return {
                "error": f"Saved scan does not contain column '{col}'.",
                "available_columns": list(df.columns),
            }

        yvals = df[col].to_numpy()
        yplot = yvals / 1e34 if quantity == "lifetime" else yvals

        if resolved_color_by == "classification" and KEY_CLASS in df.columns:
            _scatter_classification_points(ax, (xplot, yplot), df[KEY_CLASS])
        elif color_values is not None:
            scatter = ax.scatter(
                xplot, yplot, c=color_values, norm=color_norm, cmap="viridis",
                s=14, alpha=0.72,
            )
            colorbar = fig.colorbar(scatter, ax=ax)
            colorbar.set_label(_plot_axis_label(resolved_color_by))
            _configure_colorbar_ticks(colorbar, resolved_color_by)
        else:
            ax.scatter(
                xplot,
                yplot,
                s=10,
                alpha=0.5,
                color="black",
                label=_channel_latex_label(channel),
            )

        plotted_channels.append(channel)

        if quantity == "lifetime":
            bound = get_current_limits().get(channel)
            future = get_future_limits().get(channel)
            if bound:
                ax.axhline(bound / 1e34, color="crimson", lw=1.6, ls="--", alpha=0.8)
            if future:
                ax.axhline(future / 1e34, color="darkorange", lw=1.6, ls=":", alpha=0.8)
        _add_x_lifetime_limit_guides(ax, x_column)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"Saved scan {quantity} plot vs {title_x}", fontsize=13)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        _configure_plot_axis_ticks(ax, "x", xscale, x_column)
        _configure_plot_axis_ticks(ax, "y", yscale, col)
        ax.grid(True, which="both", ls=":", alpha=0.3)
        if requested_x_range is not None:
            limits = (
                (requested_x_range[0] / 1e15, requested_x_range[1] / 1e15)
                if x_column == "MGUT" else tuple(requested_x_range)
            )
            ax.set_xlim(*limits)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9, loc="best")
        fig.tight_layout()
    else:
        fig, axes = plt.subplots(
            2, 4, figsize=(16, 8), squeeze=False,
            layout="constrained" if color_values is not None else None,
        )
        flat_axes = list(axes.flat)
        for channel_index, ch in enumerate(CHANNELS):
            col = _channel_column(ch, prefix)
            if col not in df.columns:
                continue
            channel_ax = flat_axes[len(plotted_channels)]
            yvals = df[col].to_numpy()
            yplot = yvals / 1e34 if quantity == "lifetime" else yvals
            if resolved_color_by == "classification" and KEY_CLASS in df.columns:
                _scatter_classification_points(channel_ax, (xplot, yplot), df[KEY_CLASS])
            elif color_values is not None:
                channel_ax.scatter(
                    xplot, yplot, c=color_values, norm=color_norm, cmap="viridis",
                    s=14, alpha=0.72,
                )
            else:
                channel_ax.scatter(xplot, yplot, s=10, alpha=0.45, color="black")
            plotted_channels.append(ch)
            if quantity == "lifetime":
                bound = get_current_limits().get(ch)
                future = get_future_limits().get(ch)
                if bound:
                    channel_ax.axhline(bound / 1e34, color="crimson", lw=1.0, ls="--", alpha=0.65)
                if future:
                    channel_ax.axhline(future / 1e34, color="darkorange", lw=1.0, ls=":", alpha=0.65)
            _add_x_lifetime_limit_guides(channel_ax, x_column)
            channel_ax.set_title(_channel_latex_label(ch), fontsize=11)
            channel_ax.set_xscale(xscale)
            channel_ax.set_yscale(yscale)
            _configure_plot_axis_ticks(channel_ax, "x", xscale, x_column)
            _configure_plot_axis_ticks(channel_ax, "y", yscale, col)
            channel_ax.grid(True, which="both", ls=":", alpha=0.3)
            if requested_x_range is not None:
                limits = (
                    (requested_x_range[0] / 1e15, requested_x_range[1] / 1e15)
                    if x_column == "MGUT" else tuple(requested_x_range)
                )
                channel_ax.set_xlim(*limits)

        for unused_ax in flat_axes[len(plotted_channels):]:
            unused_ax.set_visible(False)
        fig.supxlabel(xlabel, fontsize=12)
        fig.supylabel(ylabel, fontsize=12)
        fig.suptitle(f"All saved-scan {quantity}s vs {title_x}", fontsize=14)
        if color_values is not None:
            from matplotlib.cm import ScalarMappable
            colorbar = fig.colorbar(
                ScalarMappable(norm=color_norm, cmap="viridis"),
                ax=flat_axes[:len(plotted_channels)],
                fraction=0.025,
                pad=0.02,
            )
            colorbar.set_label(_plot_axis_label(resolved_color_by))
            _configure_colorbar_ticks(colorbar, resolved_color_by)
        legend_handles = []
        legend_labels = []
        for channel_ax in flat_axes[:len(plotted_channels)]:
            handles, labels = channel_ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in legend_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)
        if legend_handles:
            fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3, fontsize=9,
                       bbox_to_anchor=(0.5, 0.965))
        if color_values is None:
            fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.93))

    savefig = _show_figure(fig, savefig)

    summary = {
        "status": "plot saved and displayed",
        "savefig": savefig,
        "quantity": quantity,
        "mode": mode,
        "x_column": x_column,
        "xscale": xscale,
        "x_common_exponent": x_common_exponent,
        "yscale": yscale,
        "color_by": resolved_color_by,
        "channels": plotted_channels,
        "n_points": int(len(df)),
        "x_min": float((requested_x_range or [np.min(xvals), np.max(xvals)])[0]),
        "x_max": float((requested_x_range or [np.min(xvals), np.max(xvals)])[1]),
        "source_scan_artifact_id": scan_record["artifact_id"],
        "x_lifetime_channel": x_lifetime_channel,
        "x_current_bound_years": x_current_bound,
        "x_future_sensitivity_years": x_future_sensitivity,
    }
    if "MGUT" in df.columns:
        summary["MGUT_range_GeV"] = [float(df["MGUT"].min()), float(df["MGUT"].max())]
    if gGUT_vals is not None:
        summary["gGUT_range"] = [float(gGUT_vals.min()), float(gGUT_vals.max())]

    artifact = _register_artifact(
        kind=f"saved-scan-{quantity}-plot",
        payload={
            "quantity": quantity,
            "mode": mode,
            "x_column": x_column,
            "xscale": xscale,
            "yscale": yscale,
            "color_by": resolved_color_by,
            "channels": plotted_channels,
            "x_values": xvals,
            "gGUT_values": gGUT_vals,
            "source_scan_artifact_id": scan_record["artifact_id"],
            "savefig": savefig,
        },
        summary=summary,
    )
    return {
        **summary,
        "artifact_id": artifact["artifact_id"],
        "artifact_kind": artifact["kind"],
    }


@tool
def list_saved_scan_columns(scan_artifact_id: str = "latest_scan", filepath: str | None = None) -> dict:
    """
    List all stored columns and identify numeric plotting columns in a saved scan.
    Use before generic plotting requests involving widths, branching ratios,
    lifetimes, total width, or matrix elements.
    """
    if filepath:
        scan_record, df = _scan_dataframe_from_filepath(filepath)
        numeric_cols = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
        ] if df is not None else []
    else:
        scan_record, df, numeric_cols = _saved_scan_columns(scan_artifact_id)
    if df is None:
        return {
            "error": (f"Cannot read saved scan file {filepath!r} with HDF5 key 'results'."
                      if filepath else "No saved scan data is available for the requested artifact."),
            "filepath": filepath,
        }
    status = _scan_schema_status(df)
    note = None
    if not status["is_rich_scan"]:
        note = (
            "This saved scan predates the rich schema and does not include full matrix-entry/width columns. "
            "Run a fresh scan with the updated code if you need plots against mixing-matrix elements."
        )

    physics_context = None
    if "nonsusy_model" in df.columns:
        models = sorted(str(value) for value in df["nonsusy_model"].dropna().unique())
        physics_context = {
            "framework": "non-supersymmetric",
            "models": models,
            "operator_dimension": 6,
            "wilson_coefficient_meaning": (
                "The wilson_* columns are coefficients of the implemented dimension-six "
                "gauge-boson-mediated proton-decay operators; they are not dimension-five operators."
            ),
        }

    return {
        "source_scan_artifact_id": scan_record["artifact_id"],
        "filepath": str(scan_record.get("path") or (scan_record.get("summary") or {}).get("scan_filepath") or filepath or ""),
        "n_rows": int(len(df)),
        "n_columns": len(df.columns),
        "n_numeric_columns": len(numeric_cols),
        "scan_schema": status,
        "physics_context": physics_context,
        "note": note,
        "columns": numeric_cols,
        "all_columns": list(df.columns),
        "non_numeric_columns": [col for col in df.columns if col not in numeric_cols],
    }


@tool
@_with_matrix_elements_file
def plot_saved_columns(
    x_column: str = "MGUT",
    y_column: str = "branching_p_to_pi0_eplus",
    scan_artifact_id: str = "latest_scan",
    filepath: str | None = None,
    color_by: str = "classification",
    color_scale: str = "auto",
    xscale: str = "linear",
    yscale: str = "linear",
    savefig: str = "saved_columns_plot.png",
    bounds_file: str | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
) -> dict:
    """
    Generic plotting tool for any numeric columns stored in a saved scan.
    Supports the four common templates: x vs y, log x vs y,
    x vs log y, and log x vs log y. Use for requests such as
    widths vs matrix elements, branching ratio vs MGUT, total width vs
    any saved input/output column, or direct parameter plots like MGUT vs gGUT.
    """
    import matplotlib.pyplot as plt

    if _is_total_lifetime_plot_column(x_column) or _is_total_lifetime_plot_column(y_column):
        return {
            "error": "Total lifetime is intentionally excluded from every plot. Plot a partial channel lifetime instead."
        }

    if filepath:
        scan_record, df = _scan_dataframe_from_filepath(filepath)
        numeric_cols = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
        ] if df is not None else []
    else:
        scan_record, df, numeric_cols = _saved_scan_columns(scan_artifact_id)
    if df is None:
        return {
            "error": "No saved scan data is available. Run `run_random_scan` or `run_grid_scan` explicitly first."
        }
    if KEY_CLASS in df.columns and color_by in {"", "none", None}:
        color_by = "classification"

    if x_column == "MGUT" and "MGUT" not in df.columns and "scan_index" in df.columns:
        x_column = "scan_index"
    resolved_x_column, x_error = _resolve_saved_scan_column(x_column, df, numeric_cols)
    resolved_y_column, y_error = _resolve_saved_scan_column(y_column, df, numeric_cols)
    if x_error or y_error:
        return {
            "error": "; ".join(err for err in [x_error, y_error] if err),
            "available_columns": numeric_cols,
        }
    x_column = resolved_x_column
    y_column = resolved_y_column

    if _is_total_lifetime_plot_column(x_column) or _is_total_lifetime_plot_column(y_column):
        return {
            "error": "Total lifetime is intentionally excluded from every plot. Plot a partial channel lifetime instead."
        }

    if xscale not in {"linear", "log"}:
        return {"error": "xscale must be 'linear' or 'log'."}
    if yscale not in {"linear", "log"}:
        return {"error": "yscale must be 'linear' or 'log'."}
    if color_scale not in {"auto", "linear", "log"}:
        return {"error": "color_scale must be 'auto', 'linear', or 'log'."}

    xvals = df[x_column].to_numpy()
    yvals = df[y_column].to_numpy()
    if xscale == "log" and np.any(xvals <= 0):
        return {"error": f"x_column '{x_column}' contains non-positive values and cannot use a log scale."}
    if yscale == "log" and np.any(yvals <= 0):
        return {"error": f"y_column '{y_column}' contains non-positive values and cannot use a log scale."}
    requested_x_range = _requested_axis_range(scan_record["artifact_id"], "MGUT") if x_column == "MGUT" else (
        _requested_axis_range(scan_record["artifact_id"], "gGUT") if x_column == "gGUT" else None
    )

    resolved_color_scale = (
        "log"
        if color_scale == "auto" and (
            _channel_from_lifetime_column(str(color_by)) is not None
            or "lifetime" in str(color_by).lower()
        )
        else ("linear" if color_scale == "auto" else color_scale)
    )
    source_scan_artifact_id = scan_record["artifact_id"]
    for existing in reversed(list(ARTIFACT_REGISTRY.values())):
        if existing.get("kind") != "saved-columns-plot":
            continue
        prior = existing.get("summary", {})
        same_request = (
            prior.get("source_scan_artifact_id") == source_scan_artifact_id
            and prior.get("x_column") == x_column
            and prior.get("y_column") == y_column
            and prior.get("color_by") == color_by
            and prior.get("xscale") == xscale
            and prior.get("yscale") == yscale
            and prior.get("color_scale") == resolved_color_scale
            and prior.get("style_version") == CLASSIFICATION_PLOT_STYLE_VERSION
            and prior.get("title") == title
            and prior.get("x_label") == x_label
            and prior.get("y_label") == y_label
            and (prior.get("experimental_bounds") or {}).get("sha256") == (
                current_bounds_configuration() or {}).get("sha256")
        )
        if same_request and Path(str(prior.get("savefig", ""))).exists():
            global LAST_ARTIFACT_ID
            LAST_ARTIFACT_ID = existing["artifact_id"]
            _persist_artifact_registry()
            return {
                **prior,
                "artifact_id": existing["artifact_id"],
                "artifact_kind": existing["kind"],
                "reused_existing_artifact": True,
            }

    fig, ax = plt.subplots(figsize=(8, 5))
    resolved_color_by = color_by
    color_values = None

    if color_by in {"", "none", None}:
        ax.scatter(xvals, yvals, s=10, alpha=0.45, color="black", label=y_column)
        resolved_color_by = "none"
    elif color_by == "classification" and KEY_CLASS in df.columns:
        _scatter_classification_points(ax, (xvals, yvals), df[KEY_CLASS])
    else:
        if color_by in df.columns:
            resolved_color_by = color_by
        else:
            resolved_color_by, color_error = _resolve_saved_scan_column(str(color_by), df, numeric_cols)
            if color_error:
                return {
                    "error": color_error,
                    "available_columns": numeric_cols,
                }
        color_values = df[resolved_color_by]
        if pd.api.types.is_numeric_dtype(color_values):
            numeric_colors = color_values.to_numpy(dtype=float)
            finite_colors = numeric_colors[np.isfinite(numeric_colors)]
            if finite_colors.size == 0:
                return {"error": f"Color column '{resolved_color_by}' has no finite values."}
            color_norm = None
            if resolved_color_scale == "log":
                positive_colors = finite_colors[finite_colors > 0]
                if positive_colors.size != finite_colors.size:
                    return {
                        "error": (
                            f"Color column '{resolved_color_by}' contains non-positive values "
                            "and cannot use a log color scale."
                        )
                    }
                from matplotlib.colors import LogNorm
                color_min = float(np.min(positive_colors))
                color_max = float(np.max(positive_colors))
                if color_min == color_max:
                    color_min *= 1.0 - 1.0e-9
                    color_max *= 1.0 + 1.0e-9
                color_norm = LogNorm(vmin=color_min, vmax=color_max)
            sc = ax.scatter(
                xvals, yvals, c=numeric_colors, norm=color_norm,
                s=12, alpha=0.72, cmap="viridis",
            )
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(_plot_axis_label(resolved_color_by))
            if resolved_color_scale == "log":
                from matplotlib.ticker import LogFormatterMathtext, LogLocator
                cbar.locator = LogLocator(base=10.0, numticks=7)
                cbar.formatter = LogFormatterMathtext(base=10.0)
                cbar.update_ticks()
            else:
                _configure_colorbar_ticks(cbar, resolved_color_by)
        else:
            categories = list(pd.Series(color_values).astype(str).dropna().unique())
            palette = plt.cm.tab10(np.linspace(0, 1, max(1, len(categories))))
            for category, color in zip(categories, palette):
                mask = pd.Series(color_values).astype(str) == category
                ax.scatter(xvals[mask], yvals[mask], s=10, alpha=0.5, color=color, label=category)

    channel_for_bounds = _channel_from_lifetime_column(y_column)
    if channel_for_bounds:
        bound = get_current_limits().get(channel_for_bounds)
        future = get_future_limits().get(channel_for_bounds)
        if bound:
            ax.axhline(bound, color="crimson", lw=1.6, ls="--", alpha=0.8, label="current bound")
            ax.axhspan(0, bound, alpha=0.08, color="crimson")
        if future:
            ax.axhline(future, color="darkorange", lw=1.6, ls=":", alpha=0.8, label="future sensitivity")

    ax.set_xlabel(x_label or _plot_axis_label(x_column), fontsize=12)
    ax.set_ylabel(y_label or _plot_axis_label(y_column), fontsize=12)
    ax.set_title(title or f"{y_column} vs {x_column} from saved scan", fontsize=13)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    _configure_plot_axis_ticks(ax, "x", xscale, x_column)
    _configure_plot_axis_ticks(ax, "y", yscale, y_column)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    if requested_x_range is not None:
        ax.set_xlim(requested_x_range[0], requested_x_range[1])
    x_limit_metadata = _add_x_lifetime_limit_guides(ax, x_column)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9, loc="best")
    fig.tight_layout()

    savefig = _show_figure(fig, savefig)

    summary = {
        "status": "plot saved and displayed",
        "savefig": savefig,
        "x_column": x_column,
        "y_column": y_column,
        "color_by": resolved_color_by,
        "color_scale": resolved_color_scale,
        "xscale": xscale,
        "yscale": yscale,
        "style_version": CLASSIFICATION_PLOT_STYLE_VERSION,
        "classification_styles": (
            CLASSIFICATION_SCATTER_STYLES if resolved_color_by == "classification" else None
        ),
        "n_points": int(len(df)),
        "source_scan_artifact_id": source_scan_artifact_id,
        "title": title, "x_label": x_label, "y_label": y_label,
        "x_lifetime_channel": x_limit_metadata["channel"],
        "x_current_bound_years": x_limit_metadata["current"],
        "x_future_sensitivity_years": x_limit_metadata["future"],
        "x_min": float((requested_x_range or [np.min(xvals), np.max(xvals)])[0]),
        "x_max": float((requested_x_range or [np.min(xvals), np.max(xvals)])[1]),
        "y_min": float(np.min(yvals)),
        "y_max": float(np.max(yvals)),
    }
    artifact = _register_artifact(
        kind="saved-columns-plot",
        payload={
            "x_column": x_column,
            "y_column": y_column,
            "x_values": xvals,
            "y_values": yvals,
            "color_by": resolved_color_by,
            "color_scale": resolved_color_scale,
            "color_values": color_values.to_numpy() if color_values is not None and pd.api.types.is_numeric_dtype(color_values) else None,
            "xscale": xscale,
            "yscale": yscale,
            "source_scan_artifact_id": source_scan_artifact_id,
            "title": title, "x_label": x_label, "y_label": y_label,
            "savefig": savefig,
            "style_version": CLASSIFICATION_PLOT_STYLE_VERSION,
        },
        summary=summary,
        reuse_identical=True,
    )
    return {
        **summary,
        "artifact_id": artifact["artifact_id"],
        "artifact_kind": artifact["kind"],
        "reused_existing_artifact": bool(artifact.get("reused", False)),
    }


def _finite_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def _relationship_statistics(df: pd.DataFrame, x_column: str, y_column: str, n_bins: int = 5) -> dict:
    x = pd.to_numeric(df[x_column], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 3:
        return {"error": "At least three finite paired values are required."}

    pearson = _finite_correlation(x, y)
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    spearman = _finite_correlation(x_rank, y_rank)
    log_y_pearson = _finite_correlation(x, np.log10(y)) if np.all(y > 0) else None
    log_x_log_y_pearson = (
        _finite_correlation(np.log10(x), np.log10(y))
        if np.all(x > 0) and np.all(y > 0) else None
    )

    order = np.argsort(x)
    bins = []
    for index, positions in enumerate(np.array_split(order, max(1, min(int(n_bins), len(order))))):
        if not len(positions):
            continue
        bin_x = x[positions]
        bin_y = y[positions]
        item = {
            "bin": index + 1,
            "n": int(len(positions)),
            "x_min": float(np.min(bin_x)),
            "x_max": float(np.max(bin_x)),
            "x_median": float(np.median(bin_x)),
            "y_median": float(np.median(bin_y)),
        }
        if np.all(bin_y > 0):
            item["log10_y_median"] = float(np.median(np.log10(bin_y)))
        bins.append(item)

    magnitude = abs(spearman) if spearman is not None else 0.0
    if magnitude < 0.1:
        strength = "negligible"
    elif magnitude < 0.3:
        strength = "weak"
    elif magnitude < 0.6:
        strength = "moderate"
    else:
        strength = "strong"
    direction = "none" if strength == "negligible" else ("positive" if spearman > 0 else "negative")

    result = {
        "n_finite_pairs": int(len(x)),
        "x_range": [float(np.min(x)), float(np.max(x))],
        "y_range": [float(np.min(y)), float(np.max(y))],
        "pearson_raw": pearson,
        "spearman_raw": spearman,
        "pearson_x_vs_log10_y": log_y_pearson,
        "pearson_log10_x_vs_log10_y": log_x_log_y_pearson,
        "relationship_strength": strength,
        "relationship_direction": direction,
        "supports_monotonic_trend": bool(magnitude >= 0.3),
        "quantile_bins": bins,
    }
    classification = df.loc[finite, KEY_CLASS] if KEY_CLASS in df.columns else None
    if classification is not None:
        result["classification_fractions"] = {
            str(label): float(fraction)
            for label, fraction in classification.astype(str).value_counts(normalize=True).items()
        }
    return result


@tool
@_with_matrix_elements_file
def analyze_saved_columns_relationship(
    x_column: str,
    y_column: str,
    scan_artifact_id: str = "latest_scan",
    filepath: str | None = None,
    n_bins: int = 5,
    bounds_file: str | None = None,
) -> dict:
    """Compute Pearson/Spearman correlations and binned summaries before describing a saved-scan plot."""
    if filepath:
        scan_record, df = _scan_dataframe_from_filepath(filepath)
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])] if df is not None else []
    else:
        plot_record, plot_payload = _load_artifact_payload(scan_artifact_id)
        if plot_record and plot_record.get("kind") in _RELATIONSHIP_PLOT_KINDS and isinstance(plot_payload, dict):
            axes = _saved_plot_relationship_axes(plot_record, plot_payload, _channel_from_lifetime_column(y_column))
            if axes is None:
                return {"error": "This plot contains multiple channels; specify the channel to analyze."}
            x_column, y_column = axes
            scan_artifact_id = plot_payload.get("source_scan_artifact_id", "latest_scan")
        if str(scan_artifact_id).startswith("filepath::"):
            scan_record, df = _scan_dataframe_from_filepath(str(scan_artifact_id).removeprefix("filepath::"))
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])] if df is not None else []
        else:
            scan_record, df, numeric_cols = _saved_scan_columns(scan_artifact_id)
    if df is None:
        return {"error": "No saved scan data is available for relationship analysis."}

    resolved_x, x_error = _resolve_saved_scan_column(x_column, df, numeric_cols)
    resolved_y, y_error = _resolve_saved_scan_column(y_column, df, numeric_cols)
    if x_error or y_error:
        return {"error": "; ".join(error for error in (x_error, y_error) if error)}

    result = _relationship_statistics(df, resolved_x, resolved_y, n_bins=n_bins)
    if "error" in result:
        return result
    result.update({
        "x_column": resolved_x,
        "y_column": resolved_y,
        "source_scan_artifact_id": scan_record["artifact_id"],
        "method": "Pearson linear correlation, Spearman rank correlation, and equal-count x-bin summaries",
        "interpretation_limit": "Pairwise scan correlation does not establish causation or isolate the effect of one parameter; negligible correlation does not prove parameter independence.",
    })
    varying_inputs = {}
    for name in ("MGUT", "gGUT", "M_T", "m0", "mtildeq", "M_sfermion", "M_wino", "mu", "tanbeta"):
        if name in df and pd.api.types.is_numeric_dtype(df[name]) and df[name].nunique() > 1:
            varying_inputs[name] = {"min": float(df[name].min()), "max": float(df[name].max())}
    result["other_varying_parameters"] = {name: values for name, values in varying_inputs.items() if name != resolved_x}
    result["other_varying_flavor_columns_count"] = sum(
        bool(re.fullmatch(r"(?:UL|UR|DL|DR|EL|ER|NL)_\d\d_(?:real|imag|abs)", str(name)))
        and name != resolved_x and df[name].nunique() > 1
        for name in df.columns
    )
    channel = _channel_from_lifetime_column(resolved_y)
    if channel:
        result["channel"] = channel
        result["current_bound_years"] = get_current_limits().get(channel)
        result["future_sensitivity_years"] = get_future_limits().get(channel)
    return result


@tool
@_with_matrix_elements_file
def plot_saved_3d_columns(
    x_column: str = "M_T",
    y_column: str = "mtildeq",
    z_column: str = "min_lifetime_years",
    scan_artifact_id: str = "latest_scan",
    filepath: str | None = None,
    color_by: str = "classification",
    xscale: str = "linear",
    yscale: str = "linear",
    zscale: str = "linear",
    savefig: str = "saved_columns_3d_plot.png",
    bounds_file: str | None = None,
) -> dict:
    """
    3D scatter plot for any three numeric columns stored in a saved scan.
    Use for requests like a 3D lifetime plot versus two scanned parameters.
    """
    import matplotlib.pyplot as plt

    if any(_is_total_lifetime_plot_column(column) for column in [x_column, y_column, z_column]):
        return {
            "error": "Total lifetime is intentionally excluded from every plot. Plot a partial channel lifetime instead."
        }

    if filepath:
        scan_record, df = _scan_dataframe_from_filepath(filepath)
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])] if df is not None else []
    else:
        scan_record, df, numeric_cols = _saved_scan_columns(scan_artifact_id)
    if df is None:
        return {"error": "No saved scan data is available. Run a scan first."}
    if KEY_CLASS in df.columns and color_by in {"", "none", None}:
        color_by = "classification"

    resolved = []
    errors = []
    for requested in [x_column, y_column, z_column]:
        col, err = _resolve_saved_scan_column(requested, df, numeric_cols)
        resolved.append(col)
        if err:
            errors.append(err)
    if errors:
        return {"error": "; ".join(errors), "available_columns": numeric_cols}
    x_column, y_column, z_column = resolved

    if any(_is_total_lifetime_plot_column(column) for column in [x_column, y_column, z_column]):
        return {
            "error": "Total lifetime is intentionally excluded from every plot. Plot a partial channel lifetime instead."
        }

    if xscale not in {"linear", "log"} or yscale not in {"linear", "log"} or zscale not in {"linear", "log"}:
        return {"error": "xscale, yscale, and zscale must be 'linear' or 'log'."}

    xvals = df[x_column].to_numpy(dtype=float)
    yvals = df[y_column].to_numpy(dtype=float)
    zvals = df[z_column].to_numpy(dtype=float)
    if xscale == "log" and np.any(xvals <= 0):
        return {"error": f"x_column '{x_column}' contains non-positive values and cannot use a log scale."}
    if yscale == "log" and np.any(yvals <= 0):
        return {"error": f"y_column '{y_column}' contains non-positive values and cannot use a log scale."}
    if zscale == "log" and np.any(zvals <= 0):
        return {"error": f"z_column '{z_column}' contains non-positive values and cannot use a log scale."}

    xplot = np.log10(xvals) if xscale == "log" else xvals
    yplot = np.log10(yvals) if yscale == "log" else yvals
    zplot = np.log10(zvals) if zscale == "log" else zvals

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    resolved_color_by = color_by
    color_values = None
    if color_by == "classification" and KEY_CLASS in df.columns:
        _scatter_classification_points(ax, (xplot, yplot, zplot), df[KEY_CLASS])
    elif color_by in {"", "none", None}:
        resolved_color_by = "none"
        ax.scatter(xplot, yplot, zplot, s=10, alpha=0.45, color="black")
    else:
        if color_by in df.columns:
            resolved_color_by = color_by
        else:
            resolved_color_by, color_error = _resolve_saved_scan_column(str(color_by), df, numeric_cols)
            if color_error:
                return {"error": color_error, "available_columns": numeric_cols}
        color_values = df[resolved_color_by]
        if pd.api.types.is_numeric_dtype(color_values):
            sc = ax.scatter(xplot, yplot, zplot, c=color_values.to_numpy(), s=14, alpha=0.65, cmap="viridis")
            cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
            cbar.set_label(_plot_axis_label(resolved_color_by))
        else:
            categories = list(pd.Series(color_values).astype(str).dropna().unique())
            palette = plt.cm.tab10(np.linspace(0, 1, max(1, len(categories))))
            for category, color in zip(categories, palette):
                mask = pd.Series(color_values).astype(str) == category
                ax.scatter(xplot[mask], yplot[mask], zplot[mask], s=12, alpha=0.55, color=color, label=category)

    ax.set_xlabel(("log10 " if xscale == "log" else "") + _plot_axis_label(x_column))
    ax.set_ylabel(("log10 " if yscale == "log" else "") + _plot_axis_label(y_column))
    ax.set_zlabel(("log10 " if zscale == "log" else "") + _plot_axis_label(z_column))
    ax.set_title(f"{z_column} vs {x_column}, {y_column} from saved scan")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    savefig = _show_figure(fig, savefig)
    summary = {
        "status": "plot saved and displayed",
        "savefig": savefig,
        "x_column": x_column,
        "y_column": y_column,
        "z_column": z_column,
        "color_by": resolved_color_by,
        "xscale": xscale,
        "yscale": yscale,
        "zscale": zscale,
        "n_points": int(len(df)),
        "source_scan_artifact_id": scan_record["artifact_id"],
        "x_min": float(np.min(xvals)),
        "x_max": float(np.max(xvals)),
        "y_min": float(np.min(yvals)),
        "y_max": float(np.max(yvals)),
        "z_min": float(np.min(zvals)),
        "z_max": float(np.max(zvals)),
    }
    artifact = _register_artifact(
        kind="saved-columns-3d-plot",
        payload={
            "x_column": x_column,
            "y_column": y_column,
            "z_column": z_column,
            "color_by": resolved_color_by,
            "source_scan_artifact_id": scan_record["artifact_id"],
            "savefig": savefig,
        },
        summary=summary,
    )
    return {**summary, "artifact_id": artifact["artifact_id"], "artifact_kind": artifact["kind"]}


@tool
@_with_matrix_elements_file
def plot_all_channels_vs_MGUT(
    scan_artifact_id: str = "latest_scan",
    savefig: str = "all_channels_vs_MGUT.png",
    bounds_file: str | None = None,
) -> dict:
    """
    Plot all 8 partial lifetimes as a function of MGUT from saved scan data.
    It does not launch a new scan.

    Args:
        scan_artifact_id: Scan artifact to plot from. Default is latest saved scan.
        savefig:   Output filename
    """
    import matplotlib.pyplot as plt

    scan_record, df = _scan_dataframe_from_artifact(scan_artifact_id)
    if df is None:
        return {
            "error": "No saved scan data is available for plotting. Run `run_random_scan` or `run_grid_scan` explicitly first."
        }

    MGUT_vals = df["MGUT"].to_numpy()
    gGUT_vals = df["gGUT"].to_numpy()
    requested_mgut_range = _requested_axis_range(scan_record["artifact_id"], "MGUT")
    lifetimes_by_channel = {}
    for ch in CHANNELS:
        col = _channel_column(ch, "lifetime")
        if col in df.columns:
            lifetimes_by_channel[ch] = df[col].to_numpy()

    colors = plt.cm.tab10(np.linspace(0, 1, len(CHANNELS)))

    fig, ax = plt.subplots(figsize=(10, 6))

    for ch, col in zip(CHANNELS, colors):
        if ch not in lifetimes_by_channel:
            continue
        taus = np.array(lifetimes_by_channel[ch]) / 1e34
        ax.scatter(MGUT_vals / 1e15, taus, s=5, alpha=0.3, color=col)
        # median trend line
        order = np.argsort(MGUT_vals)
        ax.plot(MGUT_vals[order] / 1e15,
                np.array([taus[i] for i in order]),
                color=col, lw=0.6, alpha=0.5, label=_channel_latex_label(ch))

    # Experimental bounds and future sensitivities
    for ch in CHANNELS:
        b = get_current_limits().get(ch)
        f = get_future_limits().get(ch)
        if b:
            ax.axhline(b / 1e34, color="crimson", lw=0.8, ls="--", alpha=0.5)
        if f:
            ax.axhline(f / 1e34, color="darkorange", lw=0.8, ls=":", alpha=0.5)

    ax.set_xlabel(r"$M_{\rm GUT}$  [$10^{15}$ GeV]", fontsize=12)
    ax.set_ylabel(r"$\tau/B$  [$10^{34}$ yr]", fontsize=12)
    ax.set_title(r"All channels vs $M_{\rm GUT}$ from saved scan", fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, which="both", ls=":", alpha=0.3)
    if requested_mgut_range is not None:
        ax.set_xlim(requested_mgut_range[0] / 1e15, requested_mgut_range[1] / 1e15)
    fig.tight_layout()

    savefig = _show_figure(fig, savefig)

    summary = {
        "status":  "plot saved and displayed",
        "savefig": savefig,
        "n_points": int(len(df)),
        "MGUT_range_GeV": requested_mgut_range or [float(MGUT_vals.min()), float(MGUT_vals.max())],
        "gGUT_range": [float(gGUT_vals.min()), float(gGUT_vals.max())],
        "source_scan_artifact_id": scan_record["artifact_id"],
    }
    artifact = _register_artifact(
        kind="mgut-plot-all-channels",
        payload={
            "MGUT_vals_GeV": MGUT_vals,
            "lifetimes_by_channel_years": lifetimes_by_channel,
            "gGUT_values": gGUT_vals,
            "source_scan_artifact_id": scan_record["artifact_id"],
            "savefig": savefig,
        },
        summary=summary,
    )
    return {
        **summary,
        "artifact_id": artifact["artifact_id"],
        "artifact_kind": artifact["kind"],
    }


@tool
@_with_matrix_elements_file
def plot_from_hdf5(
    channel: str = "p->pi0 e+",
    filepath: str | None = None,
    scan_artifact_id: str | None = None,
    x_column: str | None = None,
    y_column: str | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    xscale: str = "linear",
    yscale: str = "log",
    color_by: str = "classification",
    savefig: str = "scan_plot.png",
    bounds_file: str | None = None,
) -> dict:
    """
    Load an existing HDF5 scan file and plot from it.
    If x_column and y_column are provided, perform a generic saved-column plot.
    Otherwise plot lifetime vs MGUT for one channel, colored by classification.

    Args:
        channel:    Channel to plot (default 'p->pi0 e+')
        filepath:   Explicit HDF5 file; takes precedence over the artifact selector
        scan_artifact_id: Optional saved scan/MCMC artifact. With neither source,
                          the channel plot reads legacy 'proton_decay_scan.h5'.
        x_column:   Optional generic x column
        y_column:   Optional generic y column
        title:      Optional plot title for generic mode
        x_label:    Optional x label for generic mode
        y_label:    Optional y label for generic mode
        xscale:     'linear' or 'log' for generic mode
        yscale:     'linear' or 'log' for generic mode
        color_by:   Column to color by (default 'classification')
        savefig:    Output filename
    """
    import matplotlib.pyplot as plt

    if x_column and y_column:
        return plot_saved_columns.invoke(
            {
                "x_column": x_column,
                "y_column": y_column,
                "scan_artifact_id": scan_artifact_id or "latest_scan",
                "filepath": filepath,
                "color_by": color_by,
                "xscale": xscale,
                "yscale": yscale,
                "savefig": savefig,
                "title": title, "x_label": x_label, "y_label": y_label,
            }
        )

    try:
        if filepath:
            scan_record, df = _scan_dataframe_from_filepath(filepath)
        elif scan_artifact_id:
            scan_record, df = _scan_dataframe_from_artifact(scan_artifact_id)
        else:
            scan_record, df = _scan_dataframe_from_filepath("proton_decay_scan.h5")
        if df is None:
            raise ValueError("No readable saved results table was found.")
    except Exception as e:
        return {"error": f"Could not load {filepath}: {e}"}

    mgut_col = KEY_MGUT if KEY_MGUT in df.columns else "MGUT"
    source_filepath = scan_record.get("path") if scan_record.get("kind") == "scan-file" else None
    if source_filepath is None:
        _, source_payload = _load_artifact_payload(scan_record["artifact_id"])
        source_filepath = (source_payload or {}).get("scan_filepath")

    # Find lifetime column for requested channel
    lifetime_col = _channel_column(channel, "lifetime")

    if lifetime_col not in df.columns:
        return {
            "error": f"No column found for channel '{channel}'.",
            "available_columns": list(df.columns),
        }

    fig, ax = plt.subplots(figsize=(8, 5))
    if KEY_CLASS in df.columns:
        _scatter_classification_points(
            ax,
            (df[mgut_col].to_numpy() / 1e15, df[lifetime_col].to_numpy() / 1e34),
            df[KEY_CLASS],
        )
    else:
        ax.scatter(
            df[mgut_col].to_numpy() / 1e15,
            df[lifetime_col].to_numpy() / 1e34,
            s=10, alpha=0.45, color="black", label="saved scan",
        )

    bound  = get_current_limits().get(channel)
    future = get_future_limits().get(channel)

    if bound:
        ax.axhline(bound / 1e34, color="crimson", lw=1.8, ls="--",
                   label=f"{'Selected current bound' if current_bounds_configuration() else 'Super-K bound'} ({bound/1e34:.1f} × 10³⁴ yr)")
        ax.axhspan(0, bound / 1e34, alpha=0.08, color="crimson")
    if future:
        ax.axhline(future / 1e34, color="darkorange", lw=1.8, ls=":",
                   label=f"{'Selected future sensitivity' if current_bounds_configuration() else 'HK sensitivity'} ({future/1e34:.1f} × 10³⁴ yr)")

    ax.set_xlabel(x_label or r"$M_{\rm GUT}$  [$10^{15}$ GeV]", fontsize=12)
    ax.set_ylabel(y_label or r"$\tau/B$  [$10^{34}$ yr]", fontsize=12)
    source_label = Path(source_filepath).name if source_filepath else scan_record["artifact_id"]
    ax.set_title(title or f"Scan: {_channel_latex_label(channel)}  (from {source_label})", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()

    savefig = _show_figure(fig, savefig)

    summary = {
        "status":  "plot saved and displayed",
        "savefig": savefig,
        "channel": channel,
        "n_points": len(df),
        "current_bound_1e34yr": bound  / 1e34 if bound  else None,
        "future_sens_1e34yr":   future / 1e34 if future else None,
        "source_scan_artifact_id": scan_record["artifact_id"],
        "filepath": source_filepath,
    }
    artifact = _register_artifact(
        kind="hdf5-plot",
        payload={
            "filepath": source_filepath,
            "channel": channel,
            "MGUT_vals_GeV": df[mgut_col].to_numpy(),
            "lifetimes_years": df[lifetime_col].to_numpy(),
            "classification": df[KEY_CLASS].to_list() if KEY_CLASS in df else ["unknown"] * len(df),
            "source_scan_artifact_id": scan_record["artifact_id"],
            "savefig": savefig,
        },
        summary=summary,
    )
    return {
        **summary,
        "artifact_id": artifact["artifact_id"],
        "artifact_kind": artifact["kind"],
    }


@tool
def list_session_artifacts() -> dict:
    """
    List cached scan/plot/analysis artifacts produced in this session.
    Use for follow-up questions before launching new scans.
    """
    records = list(ARTIFACT_REGISTRY.values())
    return {
        "n_artifacts": len(records),
        "last_artifact_id": LAST_ARTIFACT_ID,
        "artifacts": [
            {
                "artifact_id": record["artifact_id"],
                "kind": record["kind"],
                "created_at_utc": record["created_at_utc"],
                "summary": record["summary"],
            }
            for record in records[-10:]
        ],
    }


@tool
def describe_last_artifact(artifact_id: str = "last") -> dict:
    """
    Describe the most recent cached artifact, or a specific artifact_id.
    Use for reasoning over an earlier scan/plot without recomputing it.
    """
    record, payload = _load_artifact_payload(artifact_id)
    if record is None:
        return {"error": "No cached artifact is available yet."}

    description = {
        "artifact_id": record["artifact_id"],
        "kind": record["kind"],
        "created_at_utc": record["created_at_utc"],
        "summary": record["summary"],
    }

    if payload is None:
        return description

    if record["kind"] in {"random-scan", "fixed-flavor-scan", "susy-su5-fixed-flavor-scan", "susy-so10-fixed-flavor-scan", "grid-scan", "susy-su5-scan", "susy-so10-scan"}:
        results = payload["results"]
        description["n_points"] = len(results)
        description["channels"] = CHANNELS
        if payload.get("scan_filepath"):
            description["scan_filepath"] = payload["scan_filepath"]
    elif record["kind"] == "mgut-plot-all-channels":
        lifetimes = payload["lifetimes_by_channel_years"]
        description["channels"] = list(lifetimes.keys())
        description["median_lifetimes_1e34yr"] = {
            ch: float(np.median(vals)) / 1e34 for ch, vals in lifetimes.items()
        }
    elif record["kind"] in {"mgut-plot-single-channel", "hdf5-plot"}:
        vals = np.asarray(payload["lifetimes_years"])
        description["channel"] = payload["channel"]
        description["lifetime_median_1e34yr"] = float(np.median(vals)) / 1e34
    elif record["kind"] == "saved-columns-plot":
        x_column = payload.get("x_column")
        y_column = payload.get("y_column")
        description.update({
            "x_column": x_column,
            "y_column": y_column,
            "xscale": payload.get("xscale"),
            "yscale": payload.get("yscale"),
            "savefig": payload.get("savefig"),
            "source_scan_artifact_id": payload.get("source_scan_artifact_id"),
        })
        x_values = np.asarray(payload.get("x_values", []), dtype=float)
        y_values = np.asarray(payload.get("y_values", []), dtype=float)
        if len(x_values) == len(y_values) and len(x_values) >= 3:
            frame = pd.DataFrame({x_column: x_values, y_column: y_values})
            description["relationship_analysis"] = _relationship_statistics(frame, x_column, y_column)

    return description


@tool
@_with_matrix_elements_file
def explain_channel_factors(
    channel_a: str = "p->pi+ nu",
    channel_b: str = "p->pi0 e+",
    filepath: str | None = None,
    MGUT: float | None = None,
    gGUT: float | None = None,
    group: Literal["SU5", "SO10"] = "SU5",
    matrix_elements_file: str | None = None,
    bounds_file: str | None = None,
) -> dict:
    """
    Compare two decay channels using a real non-SUSY input file. When filepath
    is omitted, use the packaged input file for the selected group. Optional
    MGUT and gGUT values override only those two file values.
    """
    if channel_a not in CHANNEL_DETAILS or channel_b not in CHANNEL_DETAILS:
        return {"error": f"Unknown channel. Valid channels: {CHANNELS}"}

    normalized_group = str(group).upper().replace("(", "").replace(")", "")
    if normalized_group not in {"SU5", "SO10"}:
        return {"error": "group must be SU5 or SO10."}
    input_file = filepath or _nonsusy_mcmc_default_file(normalized_group)
    try:
        params, metadata = _load_benchmark_parameters_from_file(
            input_file, "nonSUSY", normalized_group
        )
    except Exception as exc:
        return {"error": f"Could not load the channel-comparison input: {exc}"}
    if MGUT is not None:
        params["MGUT"] = float(MGUT)
    if gGUT is not None:
        params["gGUT"] = float(gGUT)
    resolved_mgut = float(params["MGUT"])
    resolved_ggut = float(params["gGUT"])
    result = _compute_nonsusy_decay_rates(
        normalized_group,
        MGUT=resolved_mgut,
        gGUT=resolved_ggut,
        UL=params["UL"],
        UR=params["UR"],
        DL=params["DL"],
        DR=params["DR"],
        EL=params["EL"],
        ER=params["ER"],
        NL=params["NL"],
    )

    def channel_snapshot(channel: str) -> dict:
        meta = CHANNEL_DETAILS[channel]
        from nonsusy_su5_formula import DEFAULT_HADRONIC
        elements = hadronic_elements("non_susy", DEFAULT_HADRONIC)
        key = ("pi0_udu" if "pi0" in channel else "K0_usu" if "K0" in channel
               else "eta_udu" if "eta" in channel else "piplus_udd" if "pi+" in channel else None)
        hadronic = elements[key] if key else (elements["Kplus_usd"], elements["Kplus_uds"])
        hadronic_sq = (
            float(hadronic**2)
            if not isinstance(hadronic, tuple)
            else float(sum(x**2 for x in hadronic))
        )
        phase = _phase_space_factor(
            meta["meson_mass"], gGUT=resolved_ggut, MGUT=resolved_mgut
        )
        return {
            "channel": channel,
            "hadronic_factor": hadronic,
            "effective_hadronic_sq": hadronic_sq,
            "meson_mass_GeV": meta["meson_mass"],
            "phase_space_prefactor": phase,
            "coefficient_family": meta["coefficient_family"],
            "flavor_multiplicity": meta["flavor_multiplicity"],
            "reference_width_GeV": float(result["widths"][channel]),
            "reference_branching_ratio": float(result["branching_ratios"][channel]),
            "reference_lifetime_years": float(result["lifetimes_years"][channel]),
        }

    a = channel_snapshot(channel_a)
    b = channel_snapshot(channel_b)

    interpretation = []
    if a["effective_hadronic_sq"] > b["effective_hadronic_sq"]:
        interpretation.append(
            f"{channel_a} has the larger effective hadronic factor, so hadronic physics does not suppress it relative to {channel_b}."
        )
    else:
        interpretation.append(
            f"{channel_a} has the smaller effective hadronic factor, so any dominance must come from flavor coefficients or multiplicity."
        )

    if a["flavor_multiplicity"] > b["flavor_multiplicity"]:
        interpretation.append(
            f"{channel_a} sums over {a['flavor_multiplicity']} neutrino flavors, whereas {channel_b} carries multiplicity {b['flavor_multiplicity']}."
        )

    interpretation.append(
        f"At MGUT={resolved_mgut:.3e} GeV and gGUT={resolved_ggut:.3f}, the width ratio Gamma({channel_a})/Gamma({channel_b}) is "
        f"{a['reference_width_GeV'] / b['reference_width_GeV']:.3e}."
    )

    return {
        "channel_a": a,
        "channel_b": b,
        "source_input_file": metadata["input_file"],
        "width_ratio_a_over_b": a["reference_width_GeV"] / b["reference_width_GeV"],
        "branching_ratio_a_over_b": a["reference_branching_ratio"] / b["reference_branching_ratio"],
        "lifetime_ratio_a_over_b": a["reference_lifetime_years"] / b["reference_lifetime_years"],
        "interpretation": interpretation,
    }


RGE_RUN_ARTIFACT_KINDS = {"sm-rge-run", "nonsusy-rge-run", "susy-rge-run"}


def _register_rge_run(kind: str, result: dict) -> dict:
    summary = {
        "runner": result.get("runner", result.get("model", kind)),
        "loops": result.get("loops"),
        "initial_scale_GeV": result.get("initial_scale_GeV"),
        "final_scale_GeV": result.get("final_scale_GeV", result.get("gut_scale_GeV")),
        "final_couplings": result.get("final_couplings"),
        "final_inverse_alpha": result.get("final_inverse_alpha"),
    }
    if "thresholds" in result:
        summary["thresholds"] = result["thresholds"]
    if "matching" in result:
        summary["matching"] = result["matching"]
    if "scheme_conversion" in result:
        summary["scheme_conversion"] = result["scheme_conversion"]
    if "stages" in result:
        summary["stages"] = [
            {
                "group": stage.get("group"),
                "start_scale_GeV": stage.get("start_scale_GeV"),
                "end_scale_GeV": stage.get("end_scale_GeV"),
                "gauge_order": stage.get("gauge_order"),
            }
            for stage in result["stages"]
        ]
    record = _register_artifact(
        kind=kind,
        payload=result,
        summary=summary,
        reuse_identical=True,
    )
    return {
        "status": "RGE running completed",
        **summary,
        "artifact_id": record["artifact_id"],
        "artifact_kind": record["kind"],
        "reused_existing_artifact": bool(record.get("reused", False)),
    }


def _load_rge_run(artifact_id: str = "latest_rge") -> tuple[dict | None, dict | None]:
    if artifact_id in {"", "latest", "last", "latest_rge"}:
        record = _find_latest_artifact_by_kind(RGE_RUN_ARTIFACT_KINDS)
        if record is None:
            return None, None
        return _load_artifact_payload(record["artifact_id"])
    record, payload = _load_artifact_payload(artifact_id)
    if record is None or record.get("kind") not in RGE_RUN_ARTIFACT_KINDS:
        return record, None
    return record, payload


@tool
def compute_sm_rge_beta_coefficients(fields: list[dict] | None = None) -> dict:
    """
    Compute one-loop and two-loop gauge beta coefficients for the SM gauge
    group U(1)_Y x SU(2)_L x SU(3)_C, optionally including arbitrary BSM
    scalars or Weyl/Dirac fermions.

    Args:
        fields: Optional list of field dictionaries. Each field supports:
                name, kind ('scalar' or 'fermion'), su3 ('1', '3', 'bar3',
                '6', '8', '10', '15', 'p15', or [p,q]), su2 dimension,
                hypercharge, generations/multiplicity, mass, real_scalar,
                and fermion_type ('weyl' or 'dirac'). Hypercharge may be a
                string fraction such as '5/6'. Scalars are complex by default;
                set real_scalar=true only when the user explicitly specifies a real scalar.
    """
    summary = coefficient_summary(fields or [])
    return {
        **summary,
        "scalar_reality_policy": "Scalars are complex by default; real_scalar=true is used only when explicitly requested.",
        "sm_one_loop_float": fraction_strings_to_float(summary["sm_one_loop"]),
        "sm_two_loop_float": fraction_strings_to_float(summary["sm_two_loop"]),
        "bsm_one_loop_delta_float": fraction_strings_to_float(summary["bsm_one_loop_delta"]),
        "bsm_two_loop_delta_float": fraction_strings_to_float(summary["bsm_two_loop_delta"]),
        "total_one_loop_float": fraction_strings_to_float(summary["total_one_loop"]),
        "total_two_loop_float": fraction_strings_to_float(summary["total_two_loop"]),
    }


@tool
def analyze_sm_unification_effect(
    fields: list[dict],
    g1: float = 0.461425,
    g2: float = 0.65184,
    g3: float = DEFAULT_MZ_COUPLINGS["g3"],
) -> dict:
    """Determine how SM-gauge-group fields affect one-loop gauge unification.

    This is a directional diagnostic, not a fixed-endpoint RGE run. It uses
    s_ij^(J) = Delta b_i^(J) - Delta b_j^(J). The light-field limit sets
    the threshold weights r_J = 1.

    Args:
        fields: SM field dictionaries, including explicit scalar reality.
        g1: Optional SU(5)-normalized coupling at M_Z; default 0.461425.
        g2: Optional SU(2) coupling at M_Z; default 0.65184.
        g3: Optional SU(3) coupling at M_Z; default 1.2143.

    The reference scale is M_Z = 91.1876 GeV. Omit unspecified couplings
    to use these defaults; do not ask the user to supply them.
    """
    if not fields:
        raise ValueError("At least one SM field is required for a unification-effect diagnostic.")

    summary = coefficient_summary(fields)
    sm_b = [Fraction(value) for value in summary["sm_one_loop"]]
    delta_b = [Fraction(value) for value in summary["bsm_one_loop_delta"]]
    total_b = [left + right for left, right in zip(sm_b, delta_b)]

    sm_s12, sm_s23 = sm_b[0] - sm_b[1], sm_b[1] - sm_b[2]
    delta_s12, delta_s23 = delta_b[0] - delta_b[1], delta_b[1] - delta_b[2]
    light_s12, light_s23 = total_b[0] - total_b[1], total_b[1] - total_b[2]

    inverse_alpha = [4.0 * np.pi / float(g) ** 2 for g in (g1, g2, g3)]
    target_ratio = (inverse_alpha[1] - inverse_alpha[2]) / (inverse_alpha[0] - inverse_alpha[1])
    scale_numerator = 2.0 * np.pi * (inverse_alpha[0] - inverse_alpha[1])
    sm_ratio = float(sm_s23 / sm_s12)
    light_ratio = None if light_s12 == 0 else float(light_s23 / light_s12)
    light_ratio_exact = None if light_s12 == 0 else str(light_s23 / light_s12)
    improves_ratio = light_ratio is not None and abs(light_ratio - target_ratio) < abs(sm_ratio - target_ratio)
    initial_ratio_derivative = (
        float(delta_s23 * sm_s12 - sm_s23 * delta_s12) / float(sm_s12 * sm_s12)
    )
    initially_moves_toward_target = (
        (target_ratio - sm_ratio) * initial_ratio_derivative > 0.0
    )

    threshold_solution = None
    required_threshold_weight = None
    threshold_status = "not_a_single_field_problem"
    denominator = float(delta_s23) - target_ratio * float(delta_s12)
    if len(summary["fields"]) == 1:
        threshold_status = "no_single_field_solution"
    if len(summary["fields"]) == 1 and denominator != 0.0:
        threshold_weight = (
            target_ratio * float(sm_s12) - float(sm_s23)
        ) / denominator
        required_threshold_weight = threshold_weight
        if threshold_weight > 1.0:
            threshold_status = "required_weight_exceeds_light_limit"
        elif threshold_weight < 0.0:
            threshold_status = "required_weight_is_negative"
        weighted_s12 = float(sm_s12) + threshold_weight * float(delta_s12)
        if 0.0 <= threshold_weight <= 1.0 and weighted_s12 > 0.0:
            threshold_status = "physical_single_field_solution"
            log_unification_ratio = scale_numerator / weighted_s12
            unification_scale = float(DEFAULT_MZ) * float(np.exp(log_unification_ratio))
            field_mass = unification_scale * float(np.exp(-threshold_weight * log_unification_ratio))
            threshold_solution = {
                "status": "physical_single_field_solution",
                "loops": 1,
                "interpretation": "Formal one-loop crossing only; this mass is not a verified two-loop solution or a complete GUT viability test.",
                "threshold_weight_r_J": threshold_weight,
                "weighted_S12": weighted_s12,
                "weighted_S23": float(sm_s23) + threshold_weight * float(delta_s23),
                "log_MU_over_MZ": log_unification_ratio,
                "unification_scale_GeV": unification_scale,
                "field_mass_GeV": field_mass,
                "display": {
                    "threshold_weight_r_J": f"{threshold_weight:.6g}",
                    "unification_scale_GeV": f"{unification_scale:.6g} GeV",
                    "field_mass_GeV": f"{field_mass:.6g} GeV",
                    "threshold_weight_r_J_3sf": f"{threshold_weight:.3g}",
                    "unification_scale_GeV_3sf": f"{unification_scale:.2e} GeV",
                    "field_mass_GeV_3sf": f"{field_mass:.2e} GeV",
                },
                "mass_conversion": "M_J = M_U*exp[-r_J*ln(M_U/M_Z)] = M_Z*(M_U/M_Z)^(1-r_J)",
            }

    field_effects = []
    for field in summary["fields"]:
        values = [Fraction(value) for value in field["one_loop_delta"]]
        field_effects.append({
            "name": field["name"], "kind": field["kind"],
            "su3": field["su3"], "su2": field["su2"],
            "hypercharge": field["hypercharge"],
            "real_scalar": field.get("real_scalar"),
            "fermion_type": field.get("fermion_type"),
            "one_loop_delta": field["one_loop_delta"],
            "s12": str(values[0] - values[1]),
            "s23": str(values[1] - values[2]),
        })

    return {
        "gauge_order": summary["gauge_order"],
        "loops": 1,
        "limitations": [
            "One-loop diagnostic with step-function thresholds and no finite matching corrections.",
            "The threshold mass must not be presented as a two-loop solution; two-loop verification requires an explicit two-loop run.",
            "A coupling crossing does not establish proton-decay consistency or complete GUT viability.",
        ],
        "initial_scale_GeV": float(DEFAULT_MZ),
        "initial_couplings": {"g1": float(g1), "g2": float(g2), "g3": float(g3)},
        "notation": {
            "field_slope_difference": "s_ij^(J) = Delta b_i^(J) - Delta b_j^(J)",
            "weighted_total": "S_ij = (b_i-b_j)_SM + sum_J s_ij^(J) r_J",
        },
        "reference_equations": {
            "target_ratio": target_ratio,
            "scale_numerator": scale_numerator,
        },
        "sm": {"s12": str(sm_s12), "s23": str(sm_s23), "ratio": sm_ratio},
        "field_effects": field_effects,
        "aggregate_field_effect": {
            "one_loop_delta": summary["bsm_one_loop_delta"],
            "s12": str(delta_s12), "s23": str(delta_s23),
        },
        "light_field_limit": {
            "meaning": "all supplied fields active from MZ, so r_J = 1",
            "s12": str(light_s12), "s23": str(light_s23),
            "ratio_definition": "S_23/S_12",
            "ratio_exact": light_ratio_exact,
            "ratio": light_ratio, "improves_over_sm": improves_ratio,
            "ratio_change_from_sm": (
                "undefined" if light_ratio is None else
                "increases" if light_ratio > sm_ratio else
                "decreases" if light_ratio < sm_ratio else "unchanged"
            ),
        },
        "diagnostic_flags": {
            "s12_is_negative": delta_s12 < 0,
            "s23_is_positive": delta_s23 > 0,
            "light_ratio_below_target": light_ratio is not None and light_ratio < target_ratio,
            "light_ratio_above_target": light_ratio is not None and light_ratio > target_ratio,
            "positive_unification_log_in_light_limit": light_s12 > 0,
            "initially_moves_ratio_toward_target": initially_moves_toward_target,
        },
        "threshold_direction": (
            "decrease_r_J_by_raising_the_field_mass"
            if light_ratio is not None and light_ratio > target_ratio
            else "no_mass_direction_inferred_from_light_limit_alone"
        ),
        "single_field_threshold_solution": threshold_solution,
        "single_field_threshold_status": threshold_status,
        "required_threshold_weight_r_J": required_threshold_weight,
    }


@tool
def run_sm_gauge_rges(
    fields: list[dict] | None = None,
    gut_scale: float = 1e16,
    initial_scale: float = 91.1876,
    g1: float = 0.461425,
    g2: float = 0.65184,
    g3: float = DEFAULT_MZ_COUPLINGS["g3"],
    loops: int = 1,
    n_points: int = 250,
    run_yukawas: bool = True,
    yt: float = DEFAULT_MZ_YUKAWAS["yt"],
    yb: float = DEFAULT_MZ_YUKAWAS["yb"],
    ytau: float = DEFAULT_MZ_YUKAWAS["ytau"],
    yu_trace: float = 0.0,
    yd_trace: float = 0.0,
    ye_trace: float = 0.0,
) -> dict:
    """
    Run SM gauge couplings with BSM threshold corrections up to a requested
    GUT scale. BSM fields become active at their mass scale.

    Args:
        fields: Optional BSM field dictionaries with the same schema as
                compute_sm_rge_beta_coefficients. Include mass in GeV for
                threshold running; omit mass for a field active over the full
                interval.
        gut_scale: Final scale in GeV.
        initial_scale: Initial scale in GeV. Default is MZ.
        g1, g2, g3: Initial gauge couplings. g1 uses SU(5)-normalized
                    hypercharge, matching the notebook convention.
        loops: 1 (default) or 2 when explicitly requested.
        n_points: Number of output samples.
        run_yukawas: If true, run diagonal third-generation yt, yb, ytau
                    from their low-scale values, matching the notebook bottom-up setup.
        yt, yb, ytau: Initial low-scale third-generation Yukawa couplings.
        yu_trace, yd_trace, ye_trace: Optional fixed Yukawa traces used only
                    when run_yukawas is false.
    """
    initial_yukawas = {"yt": float(yt), "yb": float(yb), "ytau": float(ytau)} if run_yukawas else None
    result = run_gauge_couplings(
        fields=fields or [],
        gut_scale=float(gut_scale),
        initial_scale=float(initial_scale),
        initial_couplings={"g1": float(g1), "g2": float(g2), "g3": float(g3)},
        loops=int(loops),
        n_points=min(int(n_points), 2000),
        yukawa_traces=(float(yu_trace), float(yd_trace), float(ye_trace)),
        initial_yukawas=initial_yukawas,
    )
    result["runner"] = "SM gauge and third-generation Yukawa RGE" if run_yukawas else "SM gauge RGE"
    return _register_rge_run("sm-rge-run", result)


@tool
def compute_nonsusy_rge_beta_coefficients(
    group: str = "SM",
    fields: list[dict] | None = None,
    include_model_defaults: bool = True,
) -> dict:
    """Compute non-SUSY one- and two-loop gauge beta coefficients for SM, 3221/LR, 422/Pati-Salam, 421, or 3211 groups."""
    normalized = canonical_group(group)
    if normalized == "sm":
        summary = coefficient_summary(fields or [])
        return {
            **summary,
            "model": "SM",
            "scalar_reality_policy": "Scalars are complex by default; real_scalar=true is used only when explicitly requested.",
            "total_one_loop_float": fraction_strings_to_float(summary["total_one_loop"]),
            "total_two_loop_float": fraction_strings_to_float(summary["total_two_loop"]),
        }
    model = EXTENDED_RGE_MODELS[normalized]
    selected_fields = list(model.default_fields) if include_model_defaults else []
    selected_fields.extend(fields or [])
    return extended_coefficient_summary(model, selected_fields)


@tool
def run_nonsusy_gauge_rges(
    stages: list[dict],
    initial_scale: float = 91.1876,
    g1: float = 0.461425,
    g2: float = 0.65184,
    g3: float = DEFAULT_MZ_COUPLINGS["g3"],
    loops: int = 1,
    n_points_per_stage: int = 160,
) -> dict:
    """Run staged non-SUSY gauge RGEs through SM, 3221/LR, 422/Pati-Salam, 421, or 3211 stages with tree-level matching. One loop by default; use loops=2 only when requested."""
    result = run_staged_gauge_rges(
        stages=stages,
        initial_scale=float(initial_scale),
        initial_couplings={"g1": float(g1), "g2": float(g2), "g3": float(g3)},
        loops=int(loops),
        n_points_per_stage=min(int(n_points_per_stage), 2000),
    )
    return _register_rge_run("nonsusy-rge-run", result)


@tool
def compute_susy_rge_beta_coefficients(fields: list[dict] | None = None) -> dict:
    """Compute MSSM one- and two-loop gauge beta coefficients, optionally with extra chiral superfields or component fields."""
    return mssm_coefficient_summary(fields or [])


@tool
def run_susy_gauge_rges(
    susy_scale: float = 1.0e3,
    gut_scale: float = 1.0e16,
    initial_scale: float = 91.1876,
    g1: float = 0.461425,
    g2: float = 0.65184,
    g3: float = DEFAULT_MZ_COUPLINGS["g3"],
    sm_fields: list[dict] | None = None,
    susy_fields: list[dict] | None = None,
    loops: int = 1,
    n_points: int = 250,
) -> dict:
    """Run SM MS-bar RGEs to the SUSY scale, convert to DR-bar, then run MSSM RGEs. One loop by default; use loops=2 only when requested."""
    initial_scale = float(initial_scale)
    susy_scale = float(susy_scale)
    gut_scale = float(gut_scale)
    if not initial_scale < susy_scale < gut_scale:
        raise ValueError("Require initial_scale < susy_scale < gut_scale.")
    points = min(int(n_points), 2000)
    sm_result = run_gauge_couplings(
        fields=sm_fields or [],
        gut_scale=susy_scale,
        initial_scale=initial_scale,
        initial_couplings={"g1": float(g1), "g2": float(g2), "g3": float(g3)},
        loops=int(loops),
        n_points=points,
        initial_yukawas=None,
    )
    mssm_result = run_mssm_gauge_couplings(
        fields=susy_fields or [],
        gut_scale=gut_scale,
        initial_scale=susy_scale,
        initial_couplings=sm_result["final_couplings"],
        loops=int(loops),
        n_points=points,
    )
    result = {
        "model": "SM-to-MSSM",
        "loops": int(loops),
        "initial_scale_GeV": initial_scale,
        "susy_scale_GeV": susy_scale,
        "gut_scale_GeV": gut_scale,
        "matching": (
            "one-loop MSbar-to-DRbar gauge-coupling conversion at the SUSY threshold; "
            "spectrum-dependent finite SUSY threshold corrections omitted"
        ),
        "scheme_conversion": mssm_result["scheme_conversion"],
        "sm_couplings_at_susy_scale_msbar": sm_result["final_couplings"],
        "mssm_couplings_at_susy_scale_drbar": mssm_result["initial_couplings"],
        "sm_stage": sm_result,
        "mssm_stage": mssm_result,
        "final_couplings": mssm_result["final_couplings"],
        "final_inverse_alpha": mssm_result["final_inverse_alpha"],
    }
    return _register_rge_run("susy-rge-run", result)


def _rge_plot_series(result: dict, quantity: str) -> dict[str, tuple[list[float], list[float]]]:
    collected: dict[str, tuple[list[float], list[float]]] = {}

    def add(label: str, scales, values) -> None:
        target_scales, target_values = collected.setdefault(label, ([], []))
        target_scales.extend(float(value) for value in scales)
        target_values.extend(float(value) for value in values)

    samples = result.get("samples")
    if isinstance(samples, dict):
        scales = samples.get("scale_GeV", [])
        if quantity == "coupling":
            keys = [("g1", "g1"), ("g2", "g2"), ("g3", "g3")]
        else:
            keys = [("alpha1_inv", "alpha1^-1"), ("alpha2_inv", "alpha2^-1"), ("alpha3_inv", "alpha3^-1")]
        for key, label in keys:
            if key in samples:
                add(label, scales, samples[key])
    elif isinstance(samples, list):
        for point in samples:
            scale = point.get("scale_GeV")
            values = point.get("couplings" if quantity == "coupling" else "inverse_alpha", {})
            if scale is None or not isinstance(values, dict):
                continue
            for label, value in values.items():
                add(str(label) if quantity == "coupling" else f"{label} alpha^-1", [scale], [value])

    for stage_key in ("sm_stage", "mssm_stage"):
        stage = result.get(stage_key)
        if isinstance(stage, dict):
            for label, (scales, values) in _rge_plot_series(stage, quantity).items():
                add(label, scales, values)
    return collected


def _legacy_sm_threshold_field(item: dict) -> dict:
    """Recover representations from older artifacts that only stored encoded names."""
    field = dict(item)
    match = re.fullmatch(r"([SF])_([^_]+)_(\d+)_([^_]+)(?:_(real|complex))?", str(item.get("name", "")))
    if match:
        field.setdefault("kind", "scalar" if match.group(1) == "S" else "fermion")
        field.setdefault("su3", match.group(2))
        field.setdefault("su2", int(match.group(3)))
        field.setdefault("hypercharge", match.group(4))
        field.setdefault("real_scalar", match.group(5) == "real")
    return field


def _rge_threshold_field_label(item: dict, counters: dict) -> str:
    field = _legacy_sm_threshold_field(item)
    kind = str(field.get("kind", "")).lower()
    if kind not in {"scalar", "fermion", "chiral"} or not all(key in field for key in ("su3", "su2", "hypercharge")):
        return str(field.get("name", "threshold")).replace("_", " ")
    counters[kind] = counters.get(kind, 0) + 1
    symbol = {"scalar": r"\phi", "fermion": r"\psi", "chiral": r"\Phi"}[kind]
    color = field["su3"]
    if isinstance(color, (list, tuple)):
        color_tex = "[" + ",".join(str(value) for value in color) + "]"
    else:
        color_tex = str(color).lower().replace(" ", "")
        conjugate = re.fullmatch(r"(?:bar(\d+)|(\d+)bar)", color_tex)
        if conjugate:
            color_tex = r"\overline{" + (conjugate.group(1) or conjugate.group(2)) + "}"
        elif color_tex in {"p15", "15p", "15prime", "15'"}:
            color_tex = r"15^{\prime}"
        else:
            color_tex = {"singlet": "1", "fundamental": "3", "adjoint": "8", "-3": r"\overline{3}"}.get(color_tex, color_tex)
    charge = Fraction(str(field["hypercharge"])).limit_denominator()
    charge_tex = _fraction_latex(str(charge))
    reality = r"^{\mathrm{R}}" if kind == "scalar" and field.get("real_scalar") else ""
    return rf"${symbol}_{{{counters[kind]}}}{reality}({color_tex},{int(field['su2'])},{charge_tex})$"


def _rge_threshold_scales(result: dict, counters: dict | None = None) -> list[tuple[float, str]]:
    counters = counters if counters is not None else {}
    thresholds: list[tuple[float, str]] = []
    for item in result.get("thresholds", []) or []:
        mass = item.get("mass_GeV")
        if mass is not None:
            thresholds.append((float(mass), _rge_threshold_field_label(item, counters)))
    if result.get("susy_scale_GeV") is not None:
        thresholds.append((float(result["susy_scale_GeV"]), "SUSY"))
    for stage in result.get("stages", []) or []:
        end = stage.get("end_scale_GeV")
        if end is not None:
            thresholds.append((float(end), str(stage.get("group", "stage"))))
        for field in stage.get("fields", []) or []:
            mass = field.get("mass", field.get("mass_GeV"))
            if mass is not None:
                thresholds.append((float(mass), _rge_threshold_field_label(field, counters)))
    for stage_key in ("sm_stage", "mssm_stage"):
        stage = result.get(stage_key)
        if isinstance(stage, dict):
            thresholds.extend(_rge_threshold_scales(stage, counters))
    unique = {}
    for scale, label in thresholds:
        unique.setdefault((scale, label), None)
    return list(unique)


def _rge_scale_latex(scale: float) -> str:
    mantissa, exponent = f"{float(scale):.6e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    power = int(exponent)
    return rf"10^{{{power}}}" if mantissa == "1" else rf"{mantissa}\times10^{{{power}}}"


def _rge_plot_label(label: str) -> str:
    """Render standard gauge-series names with publication-quality math text."""
    text = str(label).strip()
    match = re.fullmatch(r"alpha([123])(?:\^-1|_inv)", text)
    if match:
        return rf"$\alpha_{{{match.group(1)}}}^{{-1}}$"
    match = re.fullmatch(r"g([123])", text)
    if match:
        return rf"$g_{{{match.group(1)}}}$"
    inverse_labels = {
        "U1Y": r"$\alpha_{Y}^{-1}$",
        "SU2L": r"$\alpha_{2L}^{-1}$",
        "SU2R": r"$\alpha_{2R}^{-1}$",
        "SU3C": r"$\alpha_{3C}^{-1}$",
        "SU4C": r"$\alpha_{4C}^{-1}$",
        "U1BL": r"$\alpha_{B-L}^{-1}$",
        "U1R": r"$\alpha_{R}^{-1}$",
    }
    coupling_labels = {
        "U1Y": r"$g_{Y}$",
        "SU2L": r"$g_{2L}$",
        "SU2R": r"$g_{2R}$",
        "SU3C": r"$g_{3C}$",
        "SU4C": r"$g_{4C}$",
        "U1BL": r"$g_{B-L}$",
        "U1R": r"$g_{R}$",
    }
    inverse_match = re.fullmatch(r"(U1Y|SU2L|SU2R|SU3C|SU4C|U1BL|U1R)\s+alpha\^-1", text)
    if inverse_match:
        return inverse_labels[inverse_match.group(1)]
    if text in coupling_labels:
        return coupling_labels[text]
    return text


def _timestamped_rge_plot_name(savefig: str, source_artifact_id: str) -> str:
    """Give an RGE plot a stable UTC timestamp from its source artifact."""
    path = Path(savefig)
    suffix = path.suffix or ".png"
    stem = path.stem if path.suffix else path.name
    if re.search(r"_\d{8}T\d{6}Z$", stem):
        return str(path if path.suffix else path.with_suffix(suffix))
    match = re.search(r"-(\d{8}T\d{6})-", str(source_artifact_id))
    stamp = f"{match.group(1)}Z" if match else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(path.with_name(f"{stem}_{stamp}{suffix}"))


@tool
def plot_rge_running(
    rge_artifact_id: str = "latest_rge",
    quantity: Literal["inverse_alpha", "coupling"] = "coupling",
    savefig: str = "rge_running.png",
) -> dict:
    """Plot a computed RGE artifact: g_i by default, inverse_alpha only when requested.

    Use after an RGE running tool; never calculate curves in the LLM.
    """
    import matplotlib.pyplot as plt

    record, result = _load_rge_run(rge_artifact_id)
    if record is None or result is None:
        return {"error": f"No RGE running artifact was found for {rge_artifact_id!r}. Run an RGE tool first."}
    savefig = _timestamped_rge_plot_name(savefig, record["artifact_id"])
    series = _rge_plot_series(result, quantity)
    if not series:
        return {"error": "The RGE artifact contains no plottable coupling samples."}

    n_points = sum(len(scales) for scales, _values in series.values())
    thresholds = _rge_threshold_scales(result)
    sampled_scales = np.concatenate([np.asarray(scales, dtype=float) for scales, _ in series.values()])
    endpoint = float(result.get("final_scale_GeV") or result.get("gut_scale_GeV") or np.max(sampled_scales))
    scale_bounds = [float(result.get("initial_scale_GeV") or np.min(sampled_scales)), endpoint]
    scale_bounds.sort()
    resolved_savefig = str(_output_path(savefig))
    summary = {
        "status": "RGE plot saved and displayed",
        "savefig": resolved_savefig,
        "quantity": quantity,
        "loops": result.get("loops"),
        "scale_range_GeV": scale_bounds,
        "style_version": "latex-multiplets-exact-endpoint-v2",
        "source_rge_artifact_id": record["artifact_id"],
        "series": list(series),
        "n_plotted_samples": int(n_points),
    }
    plot_payload = {**summary, "thresholds": thresholds}
    existing_plot = _find_identical_artifact("rge-running-plot", plot_payload)
    if existing_plot is not None and Path(resolved_savefig).exists():
        return {
            **summary,
            "artifact_id": existing_plot["artifact_id"],
            "artifact_kind": existing_plot["kind"],
            "reused_existing_artifact": True,
        }

    grouped_thresholds: dict[float, list[str]] = {}
    for scale, label in thresholds:
        if not scale_bounds[0] <= scale <= scale_bounds[1]:
            continue
        labels = grouped_thresholds.setdefault(scale, [])
        if label not in labels:
            labels.append(label)
    ncols = max(1, min(3, len(grouped_thresholds)))
    nrows = int(np.ceil(len(grouped_thresholds) / ncols))
    max_lines = max((len(labels) + 1 for labels in grouped_thresholds.values()), default=0)
    legend_height = (0.4 + 0.18 * max_lines) * nrows if grouped_thresholds else 0.0
    figure_height = 5.5 + legend_height
    fig, ax = plt.subplots(figsize=(9, figure_height))
    for label, (scales, values) in series.items():
        order = np.argsort(np.asarray(scales, dtype=float))
        x = np.asarray(scales, dtype=float)[order]
        y = np.asarray(values, dtype=float)[order]
        ax.plot(x, y, lw=1.8, label=_rge_plot_label(label))
    threshold_legend_labels = []
    for scale, labels in grouped_thresholds.items():
        symbols = []
        for label in labels:
            match = re.match(r"\$(\\(?:phi|psi|Phi)_\{\d+\})", label)
            symbols.append("$" + match.group(1) + "$" if match else label)
        label = ", ".join(symbols)
        ax.axvline(scale, color="grey", lw=0.8, ls="--", alpha=0.45)
        ax.annotate(label, (scale, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -4),
                    textcoords="offset points", rotation=90, va="top", fontsize=7, color="dimgray")
        threshold_legend_labels.append(rf"$\mu={_rge_scale_latex(scale)}\,\mathrm{{GeV}}$" + "\n" + "\n".join(labels))
    ax.set_xscale("log")
    ax.set_xlim(*scale_bounds)
    ax.axvline(endpoint, color="purple", lw=1.1, ls="--", alpha=0.75,
               label=rf"End: ${_rge_scale_latex(endpoint)}\,\mathrm{{GeV}}$")
    ax.spines["right" if endpoint == scale_bounds[1] else "left"].set_visible(False)
    # Include the exact endpoint as a tick without extending the log axis.
    from matplotlib.ticker import FuncFormatter, LogFormatterMathtext
    normal_formatter = LogFormatterMathtext(base=10)
    ticks = [value for value in ax.get_xticks() if scale_bounds[0] <= value <= scale_bounds[1]
             and abs(np.log10(endpoint / value)) > 0.65]
    ax.set_xticks(ticks + [endpoint])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position:
        "$" + _rge_scale_latex(endpoint) + "$" if np.isclose(value, endpoint, rtol=1e-10)
        else normal_formatter(value, position)))
    ax.set_xlabel(r"Renormalization scale $\mu$ [GeV]")
    ax.set_ylabel(r"$\alpha_i^{-1}$" if quantity == "inverse_alpha" else r"Gauge coupling $g_i$")
    loop_note = f" ({result['loops']}-loop)" if result.get("loops") is not None else ""
    ax.set_title("Gauge-coupling RGE running" + loop_note)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    if threshold_legend_labels:
        from matplotlib.lines import Line2D
        fig.legend([Line2D([], [], color="grey", ls="--", lw=0.8)] * len(threshold_legend_labels),
                   threshold_legend_labels, loc="lower center", ncol=ncols, fontsize=8,
                   bbox_to_anchor=(0.5, 0.015), frameon=False, title="Mass thresholds (R: real scalar)", title_fontsize=8)
    fig.tight_layout(rect=(0, legend_height / figure_height, 1, 1))
    # The endpoint label contains a mantissa and can be wider than a decade
    # label. Use rendered widths so it stays readable for any running range.
    fig.canvas.draw()
    tick_values = ax.get_xticks()
    tick_labels = ax.get_xticklabels()
    endpoint_index = next(i for i, value in enumerate(tick_values)
                          if np.isclose(value, endpoint, rtol=1e-10))
    endpoint_box = tick_labels[endpoint_index].get_window_extent()
    kept_ticks = [value for i, value in enumerate(tick_values)
                  if i == endpoint_index or not
                  tick_labels[i].get_window_extent().expanded(1.15, 1).overlaps(endpoint_box)]
    ax.set_xticks(kept_ticks)
    saved = _show_figure(fig, savefig)

    summary["savefig"] = saved
    plot_payload = {**summary, "thresholds": thresholds}
    plot_record = _register_artifact(
        kind="rge-running-plot",
        payload=plot_payload,
        summary=summary,
        reuse_identical=True,
    )
    return {
        **summary,
        "artifact_id": plot_record["artifact_id"],
        "artifact_kind": plot_record["kind"],
        "reused_existing_artifact": bool(plot_record.get("reused", False)),
    }


# ── Tool registry ────────────────────────────────────────────────────

PROTON_TOOLS = [
    compute_nonsusy_su5_point,
    compute_nonsusy_so10_point,
    compute_nonsusy_su5_from_file,
    compute_nonsusy_so10_from_file,
    compute_susy_su5_point,
    compute_susy_so10_point,
    compute_susy_su5_from_file,
    compute_susy_so10_from_file,
    compute_susy_su5_from_file_with_overrides,
    compute_susy_so10_from_file_with_overrides,
    find_susy_so10_minimum_sfermion_mass_from_file,
    get_latest_proton_decay_result,
    run_nonsusy_proton_decay_mcmc,
    run_susy_proton_decay_mcmc,
    run_mcmc_around_last_point,
    run_random_scan,
    run_fixed_flavor_scan_from_file,
    run_susy_su5_fixed_flavor_scan_from_file,
    run_susy_so10_fixed_flavor_scan_from_file,
    run_grid_scan,
    classify_point,
    retrieve_experimental_bound,
    analyse_hdf5_scan,
    plot_lifetime_vs_MGUT,
    plot_saved_scan_quantity,
    list_saved_scan_columns,
    plot_saved_columns,
    analyze_saved_columns_relationship,
    plot_saved_3d_columns,
    plot_all_channels_vs_MGUT,
    plot_from_hdf5,
    list_session_artifacts,
    describe_last_artifact,
    explain_channel_factors,
]

RGE_TOOLS = [
    compute_sm_rge_beta_coefficients,
    analyze_sm_unification_effect,
    run_sm_gauge_rges,
    compute_nonsusy_rge_beta_coefficients,
    run_nonsusy_gauge_rges,
    compute_susy_rge_beta_coefficients,
    run_susy_gauge_rges,
    plot_rge_running,
    list_session_artifacts,
    describe_last_artifact,
]

TOOLS = PROTON_TOOLS + [tool_obj for tool_obj in RGE_TOOLS if tool_obj.name not in {item.name for item in PROTON_TOOLS}]
TOOL_MAP = {t.name: t for t in TOOLS}


# ================================================================
# LANGGRAPH STATE
# ================================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ================================================================
# LLM
# ================================================================

def _normalize_remote_base_url(base_url: str | None) -> str | None:
    if base_url is None:
        return None
    cleaned = base_url.strip().rstrip("/")
    if not cleaned:
        return None
    return cleaned


def _normalize_openai_base_url(base_url: str | None) -> str | None:
    cleaned = _normalize_remote_base_url(base_url)
    if cleaned and not cleaned.endswith("/v1"):
        cleaned = f"{cleaned}/v1"
    return cleaned


class OpenAICompatibleChatModel:
    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str,
        temperature: float = 0,
        tools=None,
    ):
        self.model = model
        self.temperature = temperature
        self.tools = list(tools or [])
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def bind_tools(self, tools):
        return OpenAICompatibleChatModel(
            self.model,
            api_key=self.client.api_key,
            base_url=str(self.client.base_url),
            temperature=self.temperature,
            tools=tools,
        )

    def _convert_message(self, message):
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        if isinstance(message, ToolMessage):
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        if isinstance(message, AIMessage):
            payload = {"role": "assistant", "content": message.content or ""}
            if getattr(message, "tool_calls", None):
                payload["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call.get("args", {})),
                        },
                    }
                    for call in message.tool_calls
                ]
            return payload
        return {"role": "user", "content": str(message)}

    def _convert_tools(self):
        converted = []
        for tool_obj in self.tools:
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_obj.name,
                        "description": tool_obj.description or "",
                        "parameters": tool_obj.tool_call_schema.model_json_schema(),
                    },
                }
            )
        return converted

    def invoke(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[self._convert_message(message) for message in messages],
            tools=self._convert_tools() or None,
        )
        message = response.choices[0].message
        tool_calls = []
        for tool_call in message.tool_calls or []:
            args = tool_call.function.arguments or "{}"
            tool_calls.append(
                {
                    "name": tool_call.function.name,
                    "args": json.loads(args),
                    "id": tool_call.id,
                    "type": "tool_call",
                }
            )
        return AIMessage(
            content=message.content or "",
            tool_calls=tool_calls,
        )


class OpenWebUIChatModel:
    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        email: str | None = None,
        password: str | None = None,
        temperature: float = 0,
        tools=None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.tools = list(tools or [])
        self.email = email
        self.password = password
        self.client = httpx.Client(base_url=self.base_url, timeout=120.0)
        self.api_key = api_key or self._login_for_token()

    def _login_for_token(self) -> str:
        if not self.email or not self.password:
            raise RuntimeError(
                "Open WebUI requires authentication. Provide an API token via "
                "PROTON_AGENT_OPENAI_API_KEY/--api-key, or provide "
                "PROTON_AGENT_OPENAI_EMAIL and PROTON_AGENT_OPENAI_PASSWORD."
            )

        response = self.client.post(
            "/api/v1/auths/signin",
            json={"email": self.email, "password": self.password},
        )
        response.raise_for_status()
        payload = response.json()
        token = payload.get("token") or payload.get("access_token")
        if not token and isinstance(payload.get("data"), dict):
            token = payload["data"].get("token") or payload["data"].get("access_token")
        if not token:
            raise RuntimeError("Open WebUI sign-in succeeded but no token was returned.")
        return token

    def bind_tools(self, tools):
        return OpenWebUIChatModel(
            self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            email=self.email,
            password=self.password,
            temperature=self.temperature,
            tools=tools,
        )

    def _convert_message(self, message):
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        if isinstance(message, ToolMessage):
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        if isinstance(message, AIMessage):
            payload = {"role": "assistant", "content": message.content or ""}
            if getattr(message, "tool_calls", None):
                payload["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call.get("args", {})),
                        },
                    }
                    for call in message.tool_calls
                ]
            return payload
        return {"role": "user", "content": str(message)}

    def _convert_tools(self):
        converted = []
        for tool_obj in self.tools:
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_obj.name,
                        "description": tool_obj.description or "",
                        "parameters": tool_obj.tool_call_schema.model_json_schema(),
                    },
                }
            )
        return converted

    def invoke(self, messages):
        response = self.client.post(
            "/api/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": self.temperature,
                "messages": [self._convert_message(message) for message in messages],
                "tools": self._convert_tools() or None,
            },
        )
        if response.status_code >= 400:
            detail = response.text.strip()
            if len(detail) > 400:
                detail = detail[:397] + "..."
            raise RuntimeError(
                f"Open WebUI request failed with HTTP {response.status_code}: {detail}"
            )

        payload = response.json()
        message = payload["choices"][0]["message"]
        tool_calls = []
        for tool_call in message.get("tool_calls") or []:
            function_payload = tool_call.get("function", {})
            args = function_payload.get("arguments") or "{}"
            tool_calls.append(
                {
                    "name": function_payload["name"],
                    "args": json.loads(args),
                    "id": tool_call["id"],
                    "type": "tool_call",
                }
            )
        return AIMessage(
            content=message.get("content") or "",
            tool_calls=tool_calls,
        )


class AnthropicChatModel:
    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        temperature: float = 0,
        tools=None,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.tools = list(tools or [])
        self.max_tokens = max_tokens
        self.client = httpx.Client(base_url=self.base_url, timeout=120.0)

    def bind_tools(self, tools):
        return AnthropicChatModel(
            self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            tools=tools,
            max_tokens=self.max_tokens,
        )

    def _convert_tools(self):
        return [
            {
                "name": tool_obj.name,
                "description": tool_obj.description or "",
                "input_schema": tool_obj.tool_call_schema.model_json_schema(),
            }
            for tool_obj in self.tools
        ]

    def _convert_message(self, message):
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        if isinstance(message, ToolMessage):
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content,
                    }
                ],
            }
        if isinstance(message, AIMessage):
            content = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            for call in getattr(message, "tool_calls", None) or []:
                content.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["name"],
                        "input": call.get("args", {}),
                    }
                )
            return {"role": "assistant", "content": content or ""}
        return {"role": "user", "content": str(message)}

    def invoke(self, messages):
        system_parts = []
        converted_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                system_parts.append(message.content)
            else:
                converted_messages.append(self._convert_message(message))

        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": converted_messages,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        tools = self._convert_tools()
        if tools:
            body["tools"] = tools

        response = self.client.post(
            "/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
        )
        if response.status_code >= 400:
            detail = response.text.strip()
            if len(detail) > 400:
                detail = detail[:397] + "..."
            raise RuntimeError(
                f"Anthropic request failed with HTTP {response.status_code}: {detail}"
            )

        payload = response.json()
        text_parts = []
        tool_calls = []
        for block in payload.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "name": block["name"],
                        "args": block.get("input") or {},
                        "id": block["id"],
                        "type": "tool_call",
                    }
                )
        return AIMessage(content="".join(text_parts), tool_calls=tool_calls)


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _resolve_openai_compatible_settings(
    api_base: str | None = None,
    api_key: str | None = None,
    api_email: str | None = None,
    api_password: str | None = None,
    *,
    default_base: str | None = None,
    require_base: bool = True,
) -> dict:
    resolved_base = _normalize_remote_base_url(
        api_base or _env_first("PROTON_AGENT_OPENAI_BASE_URL", "OPENAI_BASE_URL") or default_base
    )
    resolved_key = api_key or _env_first("PROTON_AGENT_OPENAI_API_KEY", "OPENAI_API_KEY")
    resolved_email = api_email or os.environ.get("PROTON_AGENT_OPENAI_EMAIL")
    resolved_password = api_password or os.environ.get("PROTON_AGENT_OPENAI_PASSWORD")

    if require_base and not resolved_base:
        raise RuntimeError(
            "Remote model selected but no API base URL was provided. "
            "Set PROTON_AGENT_OPENAI_BASE_URL/OPENAI_BASE_URL or pass --api-base."
        )
    if not resolved_key and not (resolved_email and resolved_password):
        raise RuntimeError(
            "Remote model selected but no authentication was provided. "
            "Set PROTON_AGENT_OPENAI_API_KEY/OPENAI_API_KEY, or provide "
            "PROTON_AGENT_OPENAI_EMAIL and PROTON_AGENT_OPENAI_PASSWORD."
        )
    return {
        "api_base": resolved_base,
        "api_key": resolved_key,
        "api_email": resolved_email,
        "api_password": resolved_password,
    }


def _resolve_anthropic_settings(
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict:
    resolved_base = _normalize_remote_base_url(
        api_base or _env_first("PROTON_AGENT_ANTHROPIC_BASE_URL", "ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
    )
    resolved_key = api_key or _env_first("PROTON_AGENT_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "Anthropic model selected but no API key was provided. "
            "Set PROTON_AGENT_ANTHROPIC_API_KEY/ANTHROPIC_API_KEY or pass --api-key."
        )
    return {"api_base": resolved_base, "api_key": resolved_key}


def _split_model_prefix(requested: str) -> tuple[str | None, str]:
    lowered = requested.lower()
    for prefix in ("ollama:", "local:", "remote:", "openwebui:", "openai:", "anthropic:", "antropic:"):
        if lowered.startswith(prefix):
            return prefix[:-1], requested[len(prefix):]
    return None, requested


def _looks_like_openai_official_model(model: str) -> bool:
    lowered = model.lower()
    return lowered.startswith(("gpt-", "o1", "o3", "o4"))


def _resolve_model_target(
    model: str | None,
    api_base: str | None = None,
    api_key: str | None = None,
    api_email: str | None = None,
    api_password: str | None = None,
) -> dict:
    installed = _list_installed_ollama_models()

    if model is None or not model.strip() or model.strip().lower() == "auto":
        selected = _pick_default_model(installed)
        print(f"[INFO] Auto-selected Ollama model '{selected}'.")
        return {"provider": "ollama", "model": selected}

    requested = model.strip()
    prefix, requested_model = _split_model_prefix(requested)
    normalized = _normalize_model_alias(requested_model)
    installed_map = {name.lower(): name for name in installed}

    if prefix in {"ollama", "local"}:
        return {"provider": "ollama", "model": normalized}

    if prefix in {"anthropic", "antropic"}:
        return {
            "provider": "anthropic",
            "model": requested_model,
            **_resolve_anthropic_settings(api_base, api_key),
        }

    if prefix in {"remote", "openwebui"}:
        return {
            "provider": "openwebui",
            "model": requested_model,
            **_resolve_openai_compatible_settings(
                api_base,
                api_key,
                api_email,
                api_password,
                require_base=True,
            ),
        }

    if prefix == "openai":
        default_base = "https://api.openai.com/v1" if _looks_like_openai_official_model(requested_model) else None
        return {
            "provider": "openai-compatible",
            "model": requested_model,
            **_resolve_openai_compatible_settings(
                api_base,
                api_key,
                api_email,
                api_password,
                default_base=default_base,
                require_base=default_base is None,
            ),
        }

    # Prefer explicitly installed Ollama models over remote aliases when names
    # overlap, e.g. a local gpt-oss:20b after ollama pull gpt-oss:20b.
    if requested_model in installed:
        return {"provider": "ollama", "model": requested_model}
    if requested_model.lower() in installed_map:
        return {"provider": "ollama", "model": installed_map[requested_model.lower()]}
    if normalized in installed:
        print(f"[INFO] Normalized model alias '{requested_model}' -> '{normalized}'.")
        return {"provider": "ollama", "model": normalized}
    if normalized.lower() in installed_map:
        resolved = installed_map[normalized.lower()]
        print(f"[INFO] Normalized model alias '{requested_model}' -> '{resolved}'.")
        return {"provider": "ollama", "model": resolved}

    return {"provider": "ollama", "model": normalized}


def build_llm(target: dict, tools=None):
    if target["provider"] == "ollama":
        llm = ChatOllama(
            model=target["model"],
            temperature=0,
            num_ctx=_ollama_num_ctx(target["model"]),
            client_kwargs=_ollama_client_kwargs(),
        )
    elif target["provider"] == "openwebui":
        llm = OpenWebUIChatModel(
            model=target["model"],
            api_key=target.get("api_key"),
            email=target.get("api_email"),
            password=target.get("api_password"),
            base_url=target["api_base"],
            temperature=0,
        )
    elif target["provider"] == "anthropic":
        llm = AnthropicChatModel(
            model=target["model"],
            api_key=target["api_key"],
            base_url=target["api_base"],
            temperature=0,
        )
    else:
        llm = OpenAICompatibleChatModel(
            model=target["model"],
            api_key=target["api_key"],
            base_url=_normalize_openai_base_url(target["api_base"]),
            temperature=0,
        )
    return llm.bind_tools(TOOLS if tools is None else tools)


def _list_installed_ollama_models() -> list[str]:
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []

    models = []
    for line in proc.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def _normalize_model_alias(model: str) -> str:
    model = model.strip()
    qwen_match = re.fullmatch(r"qwen:(\d+(?:\.\d+)?):(.*)", model)
    if qwen_match:
        return f"qwen{qwen_match.group(1)}:{qwen_match.group(2)}"
    return model


def _pick_default_model(installed: list[str]) -> str:
    for candidate in PREFERRED_MODELS:
        if candidate in installed:
            return candidate
    if installed:
        return installed[0]
    raise RuntimeError(
        "No Ollama models are installed. Install one first, e.g. "
        "'ollama pull qwen3:8b' or 'ollama pull qwen3.5:35b'."
    )


def _format_model_error(model: str | None) -> str:
    installed = _list_installed_ollama_models()
    if model is None or not model.strip():
        model = "auto"
    normalized = _normalize_model_alias(model)
    candidates = difflib.get_close_matches(normalized, installed, n=3, cutoff=0.4)

    parts = [f"Invalid Ollama model '{model}'."]
    if normalized != model:
        parts.append(f"Normalized form tried: '{normalized}'.")
    if installed:
        parts.append("Installed models: " + ", ".join(installed) + ".")
    parts.append(
        "Remote models can be selected with 'remote:<model>', "
        "'openai:<model>', or 'anthropic:<model>' plus provider credentials."
    )
    if candidates:
        parts.append("Closest matches: " + ", ".join(candidates) + ".")
    return " ".join(parts)


# ================================================================
# NODES
# ================================================================

PD_COMPUTATION_TOOLS = {
    "compute_nonsusy_su5_point",
    "compute_nonsusy_so10_point",
    "compute_nonsusy_su5_from_file",
    "compute_nonsusy_so10_from_file",
    "compute_susy_su5_point",
    "compute_susy_so10_point",
    "compute_susy_su5_from_file",
    "compute_susy_so10_from_file",
    "compute_susy_su5_from_file_with_overrides",
    "compute_susy_so10_from_file_with_overrides",
}


def _latest_human_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _tool_names_in_messages(messages: list) -> set[str]:
    names = set()
    for message in messages:
        for call in _extract_tool_calls(message):
            if call.get("name"):
                names.add(str(call["name"]))
        if isinstance(message, ToolMessage):
            try:
                payload = json.loads(message.content)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("tool_name"):
                names.add(str(payload["tool_name"]))
    return names


def _explicit_evasion_intent(text: str) -> bool:
    lowered = " ".join(text.lower().split())
    return any(
        phrase in lowered
        for phrase in (
            "how to evade", "how can i evade", "how can we evade",
            "evade the bound", "evade current", "evade experimental",
            "satisfy the bound", "satisfy current", "satisfy experimental",
            "avoid the bound", "avoid current", "pass the bound",
            "raise the proton lifetime", "increase the proton lifetime",
        )
    )


def _saved_scan_columns_intent(text: str) -> bool:
    lowered = " ".join((text or "").lower().split())
    wants_columns = bool(re.search(r"\b(?:columns?|column names|fields)\b", lowered)
                         and re.search(r"\b(?:available|list|show|what|which|names)\b", lowered))
    saved_source = bool(re.search(r"\b(?:scan|mcmc|dataset|hdf5)\b|\.h(?:5|df5)\b", lowered))
    additional_work = bool(re.search(
        r"\b(?:plot|graph|calculate|compute|correlation|analy[sz]e)\b|"
        r"\b(?:run|perform|start|launch)\s+(?:(?:a|the|new|another)\s+)*(?:scan|mcmc|monte carlo|calculation)\b",
        lowered))
    return wants_columns and saved_source and not additional_work


def _format_saved_scan_columns(payload: dict) -> str:
    columns = payload.get("all_columns", payload.get("columns", []))
    source = payload.get("filepath") or payload["source_scan_artifact_id"]
    return (f"Saved scan: `{source}`\n\n"
            f"{payload['n_rows']} rows; {len(columns)} columns "
            f"({payload['n_numeric_columns']} numeric). Available columns:\n\n"
            "```text\n" + "\n".join(str(column) for column in columns) + "\n```")


def _explicit_scan_intent(text: str) -> bool:
    """Recognize requests to execute a scan, excluding requests merely to plot one."""
    lowered = " ".join(text.lower().split())
    if _is_saved_plot_only_request(text) or _saved_scan_columns_intent(text):
        return False
    if re.search(r"\b(?:mcmc|monte carlo)\b", lowered):
        return True
    if re.search(r"\b(?:run|perform|start|launch|make|do)\b.{0,60}\bscan\b", lowered):
        return True
    if re.search(r"^\s*scan\b", lowered):
        return True
    if re.search(r"\b(?:random|randomized|randomised|grid|parameter|fixed[- ]flavo[u]?r)\s+scan\b", lowered):
        return True
    if re.search(r"\bscan\b.{0,60}\b\d+\s+(?:points|samples)\b", lowered):
        return True
    if re.search(r"\b(?:randomly\s+)?sampl(?:e|ing)\b.{0,80}\b\d+\s+(?:points|samples)\b", lowered):
        return True
    return False


def _computation_report_intent(text: str, tool_names: set[str]) -> bool:
    if _explicit_evasion_intent(text) or _explicit_scan_intent(text):
        return False
    if tool_names & PD_COMPUTATION_TOOLS:
        return True
    lowered = " ".join(text.lower().split())
    asks_to_compute = any(word in lowered for word in ("compute", "calculate", "evaluate", "predict"))
    mentions_decay = any(phrase in lowered for phrase in ("proton lifetime", "proton decay", "partial lifetime"))
    return asks_to_compute and mentions_decay


def _response_mode_prompt(messages: list) -> str | None:
    question = _latest_human_text(messages)
    if _is_saved_plot_only_request(question) or _saved_scan_columns_intent(question):
        return None
    tool_names = _tool_names_in_messages(messages)
    if _parse_mcmc_request(question) is not None:
        return MCMC_WORKFLOW_PROMPT
    if _explicit_scan_intent(question):
        return SCAN_WORKFLOW_PROMPT
    if _explicit_evasion_intent(question):
        return BOUND_EVASION_DISCUSSION_PROMPT
    if _computation_report_intent(question, tool_names):
        return COMPUTATION_REPORT_PROMPT
    return None


def _completed_plot_message(plot_payload: dict) -> str:
    """Return a concise final answer for a verified plot tool result."""
    plot_path = plot_payload.get("savefig") or plot_payload.get("plot_filepath")
    x_column = plot_payload.get("x_column")
    y_column = plot_payload.get("y_column")
    if x_column and y_column:
        return (
            f"Plotted `{y_column}` versus `{x_column}` from the saved data. "
            f"Verified plot file: `{plot_path}`"
        )
    order = plot_payload.get("loops")
    loop_note = f" using {order}-loop running" if order is not None else ""
    return f"Completed the requested plot{loop_note}. Verified plot file: `{plot_path}`"


def _plot_errors_since(messages: list, start: int) -> list[dict]:
    """Collect plot-tool failures in the current user turn."""
    failures = []
    for message in messages[start:]:
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload = json.loads(message.content)
        except (TypeError, json.JSONDecodeError):
            continue
        if payload.get("tool_name") in _PLOT_TOOL_NAMES and payload.get("error"):
            failures.append(payload)
    return failures


def agent_node(state: AgentState, llm, domain: str = "proton") -> AgentState:
    last_human = max(
        (index for index, message in enumerate(state["messages"]) if isinstance(message, HumanMessage)),
        default=-1,
    )
    if _saved_scan_columns_intent(_latest_human_text(state["messages"])):
        columns = _latest_named_tool_payload(state["messages"], "list_saved_scan_columns", last_human + 1)
        if columns is not None and columns.get("verified"):
            return {"messages": [AIMessage(content=_format_saved_scan_columns(columns))]}
        attempted_tools = [message for message in state["messages"][last_human + 1:] if isinstance(message, ToolMessage)]
        for message in reversed(attempted_tools):
            try:
                result = json.loads(message.content)
            except (TypeError, ValueError):
                continue
            if result.get("tool_name") == "list_saved_scan_columns" and result.get("error"):
                return {"messages": [AIMessage(content="The column listing failed: " + str(result["error"]))]}
        if attempted_tools:
            source = _saved_plot_scan_context(state["messages"]) or {}
            args = {key: value for key, value in source.items() if key in {"filepath", "scan_artifact_id"}}
            return {"messages": [AIMessage(content="", tool_calls=[{
                "name": "list_saved_scan_columns", "args": args, "id": "saved-columns-required",
            }])]}
    completed_plot = _latest_verified_plot_payload(state["messages"], last_human + 1)
    if completed_plot is not None:
        return {"messages": [AIMessage(content=_completed_plot_message(completed_plot))]}
    plot_errors = _plot_errors_since(state["messages"], last_human + 1)
    if len(plot_errors) >= 2:
        return {"messages": [AIMessage(content=(
            "The plot could not be completed after a correction attempt. "
            f"Plotting error: {plot_errors[-1]['error']}"
        ))]}
    system_messages = [SystemMessage(content=RGE_SYSTEM_PROMPT if domain == "rge" else SYSTEM_PROMPT)]
    if domain == "proton":
        mode_prompt = _response_mode_prompt(state["messages"])
        if mode_prompt:
            system_messages.append(SystemMessage(content=mode_prompt))
    messages = system_messages + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


_FILEPATH_TOOL_NAMES = {
    "compute_nonsusy_su5_from_file",
    "compute_nonsusy_so10_from_file",
    "compute_susy_su5_from_file",
    "compute_susy_so10_from_file",
    "compute_susy_su5_from_file_with_overrides",
    "compute_susy_so10_from_file_with_overrides",
    "find_susy_so10_minimum_sfermion_mass_from_file",
    "run_fixed_flavor_scan_from_file",
    "run_susy_su5_fixed_flavor_scan_from_file",
    "run_susy_so10_fixed_flavor_scan_from_file",
    "analyse_hdf5_scan",
    "list_saved_scan_columns",
    "plot_saved_columns",
    "plot_saved_3d_columns",
    "plot_saved_scan_quantity",
    "plot_lifetime_vs_MGUT",
    "plot_all_channels_vs_MGUT",
    "plot_from_hdf5",
    "analyze_saved_columns_relationship",
}


def _latest_explicit_input_filepath(messages: list) -> str | None:
    """Return the newest explicit benchmark path mentioned by the user."""
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        filepath = _explicit_filepath_from_text(str(message.content))
        if filepath:
            return filepath
    return None


def _explicit_file_computation_context(messages: list) -> str | None:
    """Give the LLM a short, local reminder when the user supplied a file."""
    question = _latest_human_text(messages)
    filepath = _latest_explicit_input_filepath(messages)
    if not filepath or not _computation_report_intent(question, set()):
        return None
    return (
        "EXPLICIT FILE COMPUTATION: The user already supplied the exact filepath "
        f"`{filepath}`. The filepath is not missing. Call the matching *_from_file "
        "calculation tool now with this exact filepath. Do not ask the user to provide "
        "or confirm it."
    )


def _asks_for_filepath_already_supplied(answer: str) -> bool:
    """Detect an invalid clarification after an explicit file computation request."""
    lowered = " ".join((answer or "").lower().split())
    asks = any(
        phrase in lowered
        for phrase in (
            "need the path", "need a path", "need the filepath", "need a filepath",
            "provide the exact path", "provide the exact filepath", "provide the path",
            "provide the filepath", "confirm the exact path", "confirm the filepath",
            "which file", "what file",
        )
    )
    return asks


def _repair_file_tool_args(name: str, args: dict, messages: list) -> dict:
    """Preserve the user's literal filepath even if the LLM omits or rewrites it."""
    repaired = dict(args or {})
    if name not in _FILEPATH_TOOL_NAMES:
        return repaired
    filepath = _latest_explicit_input_filepath(messages)
    if filepath:
        repaired["filepath"] = _normalize_path_text(filepath)
    return repaired


_RGE_COEFFICIENT_TOOL_NAMES = {
    "compute_sm_rge_beta_coefficients",
    "compute_nonsusy_rge_beta_coefficients",
}
_RGE_FIELD_TOOL_NAMES = _RGE_COEFFICIENT_TOOL_NAMES | {
    "analyze_sm_unification_effect",
    "run_sm_gauge_rges",
    "run_nonsusy_gauge_rges",
    "run_susy_gauge_rges",
}


def _canonical_sm_field_key(su3: object, su2: object, hypercharge: object) -> tuple[str, int, Fraction] | None:
    """Return a representation key stable across harmless LLM spelling variations."""
    color = unicodedata.normalize("NFKD", str(su3)).lower()
    color = re.sub(r"[\\{}_^\s]", "", color).replace("overline", "bar")
    match = re.fullmatch(r"(\d+)bar", color)
    if match:
        color = "bar" + match.group(1)
    try:
        weak = int(su2)
        if isinstance(hypercharge, float):
            charge = Fraction(str(hypercharge)).limit_denominator(1000)
        else:
            charge = Fraction(str(hypercharge).strip())
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return color, weak, charge


def _scalar_realities_in_text(text: str) -> dict[tuple[str, int, Fraction], bool]:
    """Parse real/complex declarations per representation, preserving clause scope."""
    realities: dict[tuple[str, int, Fraction], bool] = {}
    state: bool | None = None
    token_pattern = re.compile(
        r"\b(?:(real|complex)(?:-valued)?\s+)?scalars?\b|"
        r"\b(?:weyl\s+|dirac\s+)?fermions?\b|"
        r"\(([^()]+)\)",
        flags=re.IGNORECASE,
    )
    for match in token_pattern.finditer(text or ""):
        token = match.group(0)
        tuple_body = match.group(2)
        if tuple_body is None:
            if re.search(r"\bfermions?\b", token, flags=re.IGNORECASE):
                state = None
            else:
                qualifier = (match.group(1) or "").lower()
                state = qualifier == "real"  # bare or explicitly complex scalar => complex
            continue

        parts = [part.strip() for part in tuple_body.split(",")]
        marker = parts[3].lower() if len(parts) == 4 else ""
        tuple_reality = state
        if marker in {"r", "real"}:
            tuple_reality = True
            parts = parts[:3]
        elif marker in {"c", "complex"}:
            tuple_reality = False
            parts = parts[:3]
        if tuple_reality is None or len(parts) != 3:
            continue
        key = _canonical_sm_field_key(parts[0], parts[1], parts[2])
        if key is not None:
            realities[key] = tuple_reality

    # Also accept postfix forms such as "(8,1,0) is a real scalar".
    postfix_pattern = re.compile(
        r"\(([^()]+)\)\s*(?:is\s+(?:an?\s+)?)?(real|complex)(?:-valued)?\s+scalar",
        flags=re.IGNORECASE,
    )
    for match in postfix_pattern.finditer(text or ""):
        parts = [part.strip() for part in match.group(1).split(",")]
        if len(parts) == 3:
            key = _canonical_sm_field_key(parts[0], parts[1], parts[2])
            if key is not None:
                realities[key] = match.group(2).lower() == "real"
    return realities


def _scalar_reality_by_representation(messages: list) -> dict[tuple[str, int, Fraction], bool]:
    """Use the newest explicit declaration for each representation across follow-ups."""
    combined: dict[tuple[str, int, Fraction], bool] = {}
    human_texts = [str(message.content) for message in messages if isinstance(message, HumanMessage)]
    for text in reversed(human_texts):
        for key, reality in _scalar_realities_in_text(text).items():
            combined.setdefault(key, reality)
    return combined


def _repair_rge_scalar_reality_args(name: str, args: dict, messages: list) -> dict:
    """Apply real/complex choices per scalar representation, including across follow-ups."""
    repaired = dict(args or {})
    if name not in _RGE_FIELD_TOOL_NAMES:
        return repaired
    declared_realities = _scalar_reality_by_representation(messages)

    def repair_fields(raw_fields) -> list:
        fields = []
        for raw_field in raw_fields or []:
            field = dict(raw_field)
            if str(field.get("kind", "")).lower() == "scalar":
                key = _canonical_sm_field_key(
                    field.get("su3"), field.get("su2"), field.get("hypercharge")
                )
                # Unspecified non-SUSY scalars remain complex by policy.
                field["real_scalar"] = bool(declared_realities.get(key, False))
            fields.append(field)
        return fields

    if isinstance(repaired.get("fields"), list):
        repaired["fields"] = repair_fields(repaired["fields"])
    for key in ("sm_fields", "susy_fields"):
        if isinstance(repaired.get(key), list):
            repaired[key] = repair_fields(repaired[key])
    if isinstance(repaired.get("stages"), list):
        stages = []
        for raw_stage in repaired["stages"]:
            stage = dict(raw_stage)
            if isinstance(stage.get("fields"), list):
                stage["fields"] = repair_fields(stage["fields"])
            stages.append(stage)
        repaired["stages"] = stages
    return repaired


def _explicit_rge_scale_overrides(text: str) -> dict:
    """Read named or directional running bounds without treating masses as bounds."""
    normalized = _normalize_rge_request_text(text).lower()
    number = r"(?:\d+(?:\.\d+)?\s*(?:[x*]\s*)?10\s*\^\s*[+-]?\d+|10\s*\^\s*[+-]?\d+|\d+(?:\.\d+)?(?:e[+-]?\d+)?)"
    scale = rf"{number}(?:\s*(?:gev|tev|pev|eev))?"
    result = {}
    patterns = {
        "gut_scale": [
            rf"\b(?:gut_scale|final_scale|final scale|end scale|mgut|m_gut|m_u|unification scale)\s*(?:=|:|is|of|at)?\s*({scale})",
            # Collect all directional clauses together. An intermediate
            # "up to" cannot take precedence over a later final "to".
            rf"\b(?:requested|up\s*to|until|ending at|ends? at|stop(?:ping)? at|to)\s*(?:the\s+)?({scale})",
        ],
        "initial_scale": [
            rf"\b(?:initial_scale|initial scale|starting scale)\s*(?:=|:|is|of|at)?\s*({scale})",
            rf"\bfrom\s+({scale}|m_?z)\b",
        ],
    }
    for key, alternatives in patterns.items():
        for pattern in alternatives:
            matches = list(re.finditer(pattern, normalized))
            if matches:
                # A threshold sequence can contain several "to/up to" clauses.
                # Its last endpoint closes the full interval, not the first leg.
                match = matches[-1] if key == "gut_scale" else matches[0]
                value = DEFAULT_MZ if re.fullmatch(r"m_?z", match.group(1)) else _parse_scale(match.group(1))
                if value is not None and value > 0:
                    result[key] = float(value)
                break
    return result


def _rge_scale_overrides_from_messages(messages: list) -> dict:
    question = _latest_human_text(messages)
    result = _explicit_rge_scale_overrides(question)
    # Short requests to change loop order retain the previous running interval.
    followup = bool(re.search(r"\b(?:same|previous|again|instead|only)\b", question.lower()))
    followup |= _explicit_rge_loop_order(question) is not None and _parse_field_tuple(question) is None
    if followup and "gut_scale" not in result:
        humans = [message for message in messages if isinstance(message, HumanMessage)]
        for message in reversed(humans[:-1]):
            if _domain_signal(str(message.content)) == "proton":
                break
            previous = _explicit_rge_scale_overrides(str(message.content))
            if "gut_scale" in previous:
                return {**previous, **result}
    return result


def _repair_explicit_rge_scales(name: str, args: dict, messages: list) -> dict:
    repaired = dict(args)
    if name in {"run_sm_gauge_rges", "run_susy_gauge_rges"}:
        repaired.update(_rge_scale_overrides_from_messages(messages))
    return repaired


def _explicit_rge_loop_order(text: str) -> int | None:
    """Read an explicit loop order; omission selects the one-loop default."""
    normalized = unicodedata.normalize("NFKC", text or "").translate(_DASH_TRANSLATION).lower()
    match = re.search(
        r"\b(one|two|1|2)[- ]?loops?\b|\bloops?\s*(?:=|is|:)\s*([12])\b",
        normalized,
    )
    if match is None:
        return None
    value = match.group(1) or match.group(2)
    return 2 if value in {"two", "2"} else 1


def _repair_rge_unspecified_options(name: str, args: dict, messages: list) -> dict:
    """Prevent the LLM from inventing optional RGE numerical inputs."""
    repaired = dict(args or {})
    if name not in _RGE_FIELD_TOOL_NAMES:
        return repaired
    lowered = " ".join(_latest_human_text(messages).lower().split())

    for key in ("g1", "g2", "g3"):
        if not re.search(rf"\b{key}\s*(?:=|is|:)\s*[-+0-9.]", lowered):
            repaired.pop(key, None)
    loop_order = _explicit_rge_loop_order(lowered)
    if loop_order is None:
        repaired.pop("loops", None)
    elif name in {"run_sm_gauge_rges", "run_nonsusy_gauge_rges", "run_susy_gauge_rges"}:
        repaired["loops"] = loop_order
    if not re.search(r"\b\d+\s+(?:points|samples)\b|\bn_points\s*(?:=|is|:)", lowered):
        repaired.pop("n_points", None)
        repaired.pop("n_points_per_stage", None)
    if "yukawa" not in lowered:
        repaired.pop("run_yukawas", None)
        for key in ("yt", "yb", "ytau", "yu_trace", "yd_trace", "ye_trace"):
            repaired.pop(key, None)
    # "MZ" selects the documented initial-scale default; do not let the model
    # substitute a rounded or alternate value unless the user supplied a number.
    if re.search(r"\bfrom\s+mz\b|\bmz\s+scale\b", lowered) and not re.search(
        r"\b(?:initial_scale|mz)\s*(?:=|is|:)\s*[0-9]", lowered
    ):
        repaired.pop("initial_scale", None)
    return repaired


def _repair_explicit_staged_rge_args(name: str, args: dict, messages: list) -> dict:
    """Replace malformed staged JSON with the user's explicitly stated stages."""
    repaired = dict(args or {})
    if name not in {"run_nonsusy_gauge_rges", "run_susy_gauge_rges"}:
        return repaired
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        text = str(message.content)
        spec = (
            _explicit_staged_nonsusy_spec(text)
            if name == "run_nonsusy_gauge_rges"
            else _explicit_sm_to_mssm_spec(text)
        )
        if spec is not None:
            if name == "run_nonsusy_gauge_rges":
                repaired["stages"] = spec["stages"]
            else:
                repaired["susy_scale"] = spec["susy_scale"]
                repaired["gut_scale"] = spec["gut_scale"]
            if "loops" in spec:
                repaired["loops"] = spec["loops"]
            lowered = " ".join(text.lower().split())
            if not re.search(r"\binitial_scale\s*(?:=|is|:)\s*[0-9]", lowered):
                repaired.pop("initial_scale", None)
            break
    return repaired


def _repair_explicit_mcmc_args(name: str, args: dict, messages: list) -> dict:
    """Use user-stated independent widths and defaults; reject invented MCMC options."""
    if name not in {"run_nonsusy_proton_decay_mcmc", "run_susy_proton_decay_mcmc"}:
        return dict(args or {})
    question = _latest_human_text(messages)
    spec = (
        _explicit_nonsusy_mcmc_spec(question)
        if name == "run_nonsusy_proton_decay_mcmc"
        else _explicit_susy_mcmc_spec(question)
    )
    return dict(spec) if spec is not None else dict(args or {})


def _repair_tool_args(name: str, args: dict, messages: list) -> dict:
    repaired = _repair_file_tool_args(name, args, messages)
    current_scan = _latest_current_scan_result(messages) if name in _SAVED_SCAN_READ_TOOL_NAMES else None
    if current_scan is not None and _explicit_scan_intent(_latest_human_text(messages)):
        schema = TOOL_MAP[name].args
        if "filepath" in schema:
            repaired["filepath"] = current_scan["scan_filepath"]
        if "scan_artifact_id" in schema and current_scan.get("artifact_id"):
            repaired["scan_artifact_id"] = current_scan["artifact_id"]
    if name == "plot_rge_running":
        repaired["quantity"] = _rge_plot_quantity_from_messages(messages)
        current_run = _latest_current_rge_run(messages)
        if current_run is not None:
            # A global "latest" artifact may belong to another model or chat.
            repaired["rge_artifact_id"] = current_run["artifact_id"]
    if (_is_saved_plot_only_request(_latest_human_text(messages)) or
            _saved_scan_columns_intent(_latest_human_text(messages))) and name in _SAVED_SCAN_READ_TOOL_NAMES:
        source = _saved_plot_scan_context(messages)
        if source:
            schema = TOOL_MAP[name].args
            if "scan_artifact_id" in schema:
                repaired["scan_artifact_id"] = source["scan_artifact_id"]
            if "filepath" in schema:
                if source.get("filepath"):
                    repaired["filepath"] = source["filepath"]
                else:
                    repaired.pop("filepath", None)
    if name == "analyze_saved_columns_relationship":
        context = _plot_relationship_description_context(_latest_human_text(messages))
        if context:
            if context.get("dataset_filepath"):
                repaired["filepath"] = context["dataset_filepath"]
                repaired.update({key: context[key] for key in ("x_column", "y_column") if context.get(key)})
                repaired.pop("scan_artifact_id", None)
            else:
                repaired.update({"x_column": context["x_column"], "y_column": context["y_column"],
                                 "scan_artifact_id": context["plot_artifact_id"]})
                repaired.pop("filepath", None)
    repaired = _repair_explicit_mcmc_args(name, repaired, messages)
    repaired = _repair_explicit_staged_rge_args(name, repaired, messages)
    if name == "run_sm_gauge_rges":
        sm_spec = _explicit_single_sm_running_spec(_latest_human_text(messages))
        if sm_spec is not None:
            repaired.update({"gut_scale": 1e16, "initial_scale": DEFAULT_MZ, **sm_spec,
                             **_rge_scale_overrides_from_messages(messages)})
    repaired = _repair_rge_scalar_reality_args(name, repaired, messages)
    repaired = _repair_rge_unspecified_options(name, repaired, messages)
    if name in TOOL_MAP and "matrix_elements_file" in TOOL_MAP[name].args:
        matrix_file = _requested_matrix_elements_filepath(_latest_human_text(messages))
        if matrix_file:
            repaired["matrix_elements_file"] = matrix_file
        else:
            repaired.pop("matrix_elements_file", None)
    if name in TOOL_MAP and "bounds_file" in TOOL_MAP[name].args:
        bounds_file = _requested_bounds_filepath(_latest_human_text(messages))
        if bounds_file:
            repaired["bounds_file"] = bounds_file
        else:
            repaired.pop("bounds_file", None)
    return _repair_explicit_rge_scales(name, repaired, messages)


_PLOT_TOOL_NAMES = {
    "plot_lifetime_vs_MGUT", "plot_all_channels_vs_MGUT", "plot_saved_scan_quantity",
    "plot_saved_columns", "plot_saved_3d_columns", "plot_from_hdf5", "plot_rge_running",
}
_SCAN_EXECUTION_TOOL_NAMES = {
    name for name in TOOL_MAP if name.startswith("run_") and ("scan" in name or "mcmc" in name)
}
_SAVED_SCAN_READ_TOOL_NAMES = {
    "plot_lifetime_vs_MGUT", "plot_all_channels_vs_MGUT", "plot_saved_scan_quantity",
    "plot_saved_columns", "plot_saved_3d_columns", "plot_from_hdf5",
    "list_saved_scan_columns", "analyse_hdf5_scan", "analyze_saved_columns_relationship",
}


def _saved_plot_scan_context(messages: list) -> dict | None:
    """Prefer an explicitly named scan, then the conversation's saved result."""
    question = _latest_human_text(messages)
    current_path = _explicit_filepath_from_text(question)
    if current_path and current_path.lower().endswith((".h5", ".hdf5")):
        path = str(_resolve_project_path(current_path))
        return {"scan_artifact_id": f"filepath::{path}", "filepath": path}

    def source_for_record(record: dict) -> dict | None:
        if record.get("kind") not in SCAN_ARTIFACT_KINDS:
            return None
        source = {"scan_artifact_id": record["artifact_id"]}
        path = (record.get("summary") or {}).get("scan_filepath")
        if path:
            source["filepath"] = str(_resolve_project_path(path))
        return source

    for record in reversed(list(ARTIFACT_REGISTRY.values())):
        if record["artifact_id"] in question:
            source = source_for_record(record)
            if source:
                return source
        if current_path and str((record.get("summary") or {}).get("savefig", "")).endswith(current_path):
            _, payload = _load_artifact_payload(record["artifact_id"])
            scan_id = (payload or {}).get("source_scan_artifact_id")
            if str(scan_id).startswith("filepath::"):
                return {"scan_artifact_id": scan_id, "filepath": str(scan_id).removeprefix("filepath::")}
            if scan_id in ARTIFACT_REGISTRY:
                return source_for_record(ARTIFACT_REGISTRY[scan_id])
    if current_path:
        # Never silently replace an explicitly supplied non-scan filename.
        return None

    for message in reversed(messages):
        if not isinstance(message, (AIMessage, ToolMessage)):
            continue
        content = str(message.content)
        for record in reversed(list(ARTIFACT_REGISTRY.values())):
            if record["artifact_id"] in content:
                source = source_for_record(record)
                if source:
                    return source
        path = _explicit_filepath_from_text(content)
        if path and path.lower().endswith((".h5", ".hdf5")):
            resolved = _resolve_project_path(path)
            if resolved.exists():
                return {"scan_artifact_id": f"filepath::{resolved}", "filepath": str(resolved)}
    record = _find_latest_artifact_by_kind(SCAN_ARTIFACT_KINDS)
    return source_for_record(record) if record else None


def _tool_call_fingerprint(name: str, args: dict) -> str:
    return json.dumps({"name": name, "args": _jsonable(args)}, sort_keys=True, default=str)


def _current_request_tool_fingerprints(messages: list) -> set[str]:
    last_human = max(
        (index for index, message in enumerate(messages) if isinstance(message, HumanMessage)),
        default=-1,
    )
    fingerprints: set[str] = set()
    for message in messages[last_human + 1:-1]:
        for call in _extract_tool_calls(message):
            args = _repair_tool_args(call["name"], call.get("args") or {}, messages)
            fingerprints.add(_tool_call_fingerprint(call["name"], args))
    return fingerprints


def _verify_tool_result(name: str, args: dict, result: dict) -> dict:
    """Attach machine-checkable evidence and reject missing plot outputs."""
    if not isinstance(result, dict) or result.get("error"):
        return result
    verified = dict(result)
    checks: dict[str, object] = {}
    if name == "list_saved_scan_columns":
        if args.get("filepath"):
            expected_source = f"filepath::{_resolve_project_path(args['filepath'], must_exist=True)}"
            if verified.get("source_scan_artifact_id") != expected_source:
                return {"error": "Column listing used a different source from the explicitly requested scan file."}
            checks["requested_source_matches"] = True
        if not isinstance(verified.get("columns"), list) or "n_rows" not in verified:
            return {"error": "Column listing did not return a valid saved-file schema."}
    if name in {"run_sm_gauge_rges", "run_susy_gauge_rges"} and "gut_scale" in args:
        actual_end = verified.get("final_scale_GeV", verified.get("gut_scale_GeV"))
        if actual_end is None or not np.isclose(float(actual_end), float(args["gut_scale"]), rtol=1e-12, atol=0):
            return {"error": f"The running endpoint does not match the requested {args['gut_scale']} GeV."}
        checks["requested_endpoint_matches"] = True
    if name in _PLOT_TOOL_NAMES:
        savefig = verified.get("savefig") or verified.get("plot_filepath")
        if not savefig:
            return {"error": f"{name} did not return a plot filepath."}
        plot_path = _resolve_project_path(str(savefig))
        if not plot_path.exists():
            return {"error": f"{name} reported a plot that does not exist: {savefig}"}
        checks["plot_file_exists"] = True
        checks["plot_filepath"] = str(plot_path)
        verified["savefig"] = str(plot_path)
        if name == "plot_rge_running":
            source_record, _ = _load_rge_run(args.get("rge_artifact_id", "latest_rge"))
            if source_record is None or verified.get("source_rge_artifact_id") != source_record.get("artifact_id"):
                return {"error": "plot_rge_running returned a different source from the requested running artifact."}
            checks["requested_source_matches"] = True
        if name in {"plot_saved_columns", "plot_saved_scan_quantity"}:
            if name == "plot_saved_columns" and (not verified.get("x_column") or not verified.get("y_column")):
                return {"error": "plot_saved_columns did not report its resolved x/y columns."}
            requested_filepath = args.get("filepath")
            if requested_filepath:
                try:
                    expected_source = f"filepath::{_resolve_project_path(requested_filepath, must_exist=True)}"
                except FileNotFoundError as exc:
                    return {"error": str(exc)}
                actual_source = verified.get("source_scan_artifact_id")
                if actual_source != expected_source:
                    return {
                        "error": (
                            "plot_saved_columns used a different source than the explicitly requested file: "
                            f"expected {expected_source}, got {actual_source or 'no source identifier'}"
                        )
                    }
                checks["requested_source_matches"] = True
            if name == "plot_saved_columns":
                checks["resolved_axes"] = {
                    "x_column": verified["x_column"],
                    "y_column": verified["y_column"],
                }
    verified["verified"] = True
    if checks:
        verified["verification"] = checks
    return verified


def tool_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    tool_results = []
    seen_fingerprints = _current_request_tool_fingerprints(state["messages"])
    last_human = max(
        (index for index, message in enumerate(state["messages"]) if isinstance(message, HumanMessage)),
        default=-1,
    )
    completed_rge_plot = _current_request_rge_plot_payload(state["messages"], last_human + 1)
    tool_calls = _extract_tool_calls(last_message)
    for tool_call in tool_calls:
        name = tool_call["name"]
        working_messages = state["messages"] + tool_results
        args = _repair_tool_args(name, tool_call.get("args") or {}, working_messages)
        fingerprint = _tool_call_fingerprint(name, args)

        display_args = {k: v for k, v in args.items() if k != "lifetimes"}
        print(f"  [TOOL] {name}({json.dumps(display_args, default=str)})")
        plot_bound_error = None
        if name == "plot_rge_running":
            plot_bound_error = _rge_plot_request_error(args, working_messages)
        if (_is_saved_plot_only_request(_latest_human_text(state["messages"])) or
                _saved_scan_columns_intent(_latest_human_text(state["messages"]))) and name in _SCAN_EXECUTION_TOOL_NAMES:
            result = {"error": "This request only asks to inspect or plot saved data. A new scan/MCMC is not authorized; use the existing scan artifact with a read-only tool."}
        elif plot_bound_error:
            result = {"error": plot_bound_error}
        elif completed_rge_plot is not None and name in {
            "run_sm_gauge_rges", "run_nonsusy_gauge_rges", "run_susy_gauge_rges", "plot_rge_running"
        }:
            result = {
                "error": "RGE run-and-plot workflow already completed in this user turn; repeated call skipped."
            }
        elif fingerprint in seen_fingerprints:
            result = {
                "error": (
                    "Repeated identical tool call rejected. Use the existing result or change the arguments; "
                    "do not call the same tool again."
                )
            }
        else:
            seen_fingerprints.add(fingerprint)
            try:
                result = _verify_tool_result(name, args, _invoke_tool(name, args))
            except Exception as exc:
                result = {
                    "error": (
                        f"{type(exc).__name__}: {exc}. "
                        "Correct the tool arguments and try the tool call again."
                    )
                }

        if name == "plot_rge_running" and isinstance(result, dict) and not result.get("error"):
            completed_rge_plot = {"tool_name": name, **result}
        payload = {"tool_name": name, **result}

        tool_results.append(
            ToolMessage(
                content=json.dumps(payload, default=str),
                tool_call_id=tool_call["id"],
            )
        )

    if completed_rge_plot is not None:
        plot_path = completed_rge_plot.get("savefig") or completed_rge_plot.get("plot_filepath")
        order = completed_rge_plot.get("loops")
        loop_note = f" using {order}-loop running" if order is not None else ""
        tool_results.append(AIMessage(
            content=(
                f"Completed the RGE running and plot workflow{loop_note}. "
                f"Verified plot file: `{plot_path}`"
            ),
            additional_kwargs={"rge_workflow_complete": True},
        ))
    return {"messages": tool_results}


# ================================================================
# TOOL LOOP CONTROL
# ================================================================

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    if _extract_tool_calls(last):
        return "tools"
    return "end"


def should_continue_after_tools(state: AgentState) -> Literal["agent", "end"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.additional_kwargs.get("rge_workflow_complete"):
        return "end"
    return "agent"


# ================================================================
# GRAPH
# ================================================================

def build_graph(target: dict, domain: str = "proton"):
    tools = RGE_TOOLS if domain == "rge" else PROTON_TOOLS
    llm = build_llm(target, tools=tools)
    graph = StateGraph(AgentState)
    graph.add_node("agent", lambda state: agent_node(state, llm, domain=domain))
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue,
                                {"tools": "tools", "end": END})
    graph.add_conditional_edges("tools", should_continue_after_tools,
                                {"agent": "agent", "end": END})
    return graph.compile()


def _invoke_agent_graph(graph, messages: list) -> dict:
    """Invoke the agent with a hard ceiling on model/tool iterations."""
    recursion_limit = int(os.environ.get("PROTON_AGENT_RECURSION_LIMIT", "12"))
    if recursion_limit <= 0:
        raise ValueError("PROTON_AGENT_RECURSION_LIMIT must be a positive integer.")
    try:
        return graph.invoke(
            {"messages": messages},
            {"recursion_limit": recursion_limit},
        )
    except GraphRecursionError as exc:
        raise RuntimeError(
            f"Agent stopped after reaching its graph-iteration limit ({recursion_limit}); "
            "this usually indicates a repeated or malformed tool-call loop."
        ) from exc


@contextmanager
def _activity_indicator(label: str = "Agent is working"):
    stop_event = threading.Event()

    def spin():
        frames = "|/-\\"
        index = 0
        while not stop_event.is_set():
            frame = frames[index % len(frames)]
            print(f"\r{label} ... {frame}", end="", flush=True)
            index += 1
            time.sleep(0.12)
        print("\r" + " " * (len(label) + 10) + "\r", end="", flush=True)

    worker = threading.Thread(target=spin, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop_event.set()
        worker.join()


# ================================================================
# PUBLIC INTERFACE  
# ================================================================

_graphs: dict[tuple[str, str], object] = {}
_router_llms: dict[str, object] = {}


def _domain_signal(text: str) -> str | None:
    lowered = " ".join((text or "").lower().split())
    if _saved_scan_columns_intent(text):
        return "proton"
    # MCMC is currently a proton-decay flavor/parameter workflow. Route it
    # before generic GUT/unification language can send it to the RGE specialist.
    if re.search(r"\b(?:mcmc|monte carlo)\b", lowered):
        return "proton"
    rge = bool(re.search(
        r"\b(?:rge|renormalization group|renormalisation group|beta[- ]?(?:function )?coefficient|"
        r"gauge[- ]coupling running|run(?:ning)? (?:the )?gauge couplings?|mssm running|"
        r"pati[- ]salam|left[- ]right|unif(?:y|ies|ied|ication)|3221|3211|\b422\b|\b421\b)\b",
        lowered,
    ))
    proton = bool(re.search(
        r"\b(?:proton decay|proton lifetime|partial lifetime|branching ratio|decay width|"
        r"p->|p→|fixed[- ]flavo[u]?r|mixing matrices?)\b",
        lowered,
    ))
    if rge and not proton:
        return "rge"
    if proton and not rge:
        return "proton"
    return None


def _route_specialist(question: str, history: list, target: dict) -> str:
    direct = _domain_signal(question)
    if direct:
        return direct

    # Short follow-ups inherit the most recent unambiguous specialist domain.
    for message in reversed(history[-6:]):
        hint = _domain_signal(str(getattr(message, "content", "")))
        if hint:
            return hint

    target_key = json.dumps({k: v for k, v in target.items() if k not in {"api_key", "api_password"}}, sort_keys=True)
    router = _router_llms.get(target_key)
    if router is None:
        router = _build_llm_without_tools(target)
        _router_llms[target_key] = router
    recent = "\n".join(
        f"{type(message).__name__}: {_shorten(getattr(message, 'content', ''), 180)}"
        for message in history[-4:]
    )
    response = router.invoke([
        SystemMessage(content=(
            "Route the user's requested work to exactly one specialist. Reply with only RGE or PROTON. "
            "Choose RGE for beta coefficients, gauge-coupling evolution, thresholds, matching, or unification. "
            "Choose PROTON for proton-decay calculations, scans, bounds, decay plots, or flavor analysis. "
            "For a follow-up, use the recent context. Do not answer the scientific question."
        )),
        HumanMessage(content=f"Recent context:\n{recent or '(none)'}\n\nCurrent request:\n{question}"),
    ])
    choice = str(getattr(response, "content", response)).strip().lower()
    return "rge" if "rge" in choice and "proton" not in choice else "proton"


def _final_answer_from_state(final_state: dict) -> tuple[str, list]:
    all_messages = final_state["messages"]
    for msg in reversed(all_messages):
        if isinstance(msg, HumanMessage):
            continue
        if isinstance(msg, ToolMessage):
            continue
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            continue
        if hasattr(msg, "content") and msg.content:
            return msg.content, all_messages
    return "[No response generated]", all_messages


_NUMERIC_CLAIM_RE = re.compile(
    r"(?:\d+(?:\.\d+)?\s*(?:[×x]\s*10|[eE][+-]?\d+|%|GeV\b|yr\b|years?\b))"
    r"|(?:10[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)"
)


def _contains_numeric_claim(text: str) -> bool:
    """Detect quantitative prose that must be grounded by a tool call."""
    return bool(_NUMERIC_CLAIM_RE.search(text or ""))


def _current_turn_used_tool(messages: list, initial_count: int) -> bool:
    return any(isinstance(message, ToolMessage) for message in messages[initial_count:])


_UNSOLICITED_EVASION_PATTERNS = (
    r"\bhow to evade\b",
    r"\bways? to evade\b",
    r"\bevad(?:e|ing) (?:the |current |experimental )?bounds?\b",
    r"\brecommendation(?:s)?\b",
    r"\bincrease (?:the )?(?:color[- ]triplet|triplet mass|m_t|m0|m_0|sfermion)\b",
    r"\braise (?:the )?(?:color[- ]triplet|triplet mass|m_t|m0|m_0|sfermion)\b",
    r"\brun (?:a |another )?(?:fixed[- ]flavo[u]?r )?scan\b",
    r"\bif you (?:would like|want),? (?:i|we) can\b",
    r"\bto bring .{0,80} above (?:the )?(?:current )?bound\b",
)


def _contains_unsolicited_evasion_guidance(text: str) -> bool:
    lowered = " ".join((text or "").lower().split())
    return any(re.search(pattern, lowered) for pattern in _UNSOLICITED_EVASION_PATTERNS)


_UNREQUESTED_QUANTITATIVE_EVASION_PATTERNS = (
    r"\bwould need to be increased by\b",
    r"\b(?:required|needed|minimum)\s+(?:value|mass|factor|m_t|m0|m_0)\b",
    r"(?:√|sqrt\s*\(|square root of)",
    r"\bpractical guidance for a scan\b",
    r"\b(?:run|perform|define|use) (?:a )?(?:fixed[- ]flavo[u]?r )?scan\b",
    r"\bscan that varies\b",
    r"\be\.g\.,?\s*(?:m_t|m0|m_0)\b",
    r"\b(?:raising|increasing)\s+(?:m_half|mu|μ|ax|tanβ|tanbeta).{0,80}\b(?:lengthens|raises?|increases?)\s+(?:the )?(?:proton )?lifetime\b",
    r"\brenormalization[- ]group evolution used in the calculation\b",
)


def _quantitative_evasion_intent(text: str) -> bool:
    lowered = " ".join(text.lower().split())
    return any(
        phrase in lowered
        for phrase in (
            "how much", "what value", "which value", "minimum value", "minimum mass",
            "required value", "required mass", "find the threshold", "compute the threshold",
            "perform a scan", "run a scan", "scan ", "candidate point", "specific point",
            "parameter range", "what range", "by what factor",
        )
    )


def _contains_unrequested_quantitative_evasion(text: str) -> bool:
    lowered = " ".join((text or "").lower().split())
    return any(re.search(pattern, lowered) for pattern in _UNREQUESTED_QUANTITATIVE_EVASION_PATTERNS)


def _turn_response_mode(question: str, messages: list) -> str | None:
    tool_names = _tool_names_in_messages(messages)
    if _explicit_scan_intent(question):
        return "scan"
    if _explicit_evasion_intent(question):
        return "evasion"
    if _computation_report_intent(question, tool_names):
        return "computation"
    return None


_RELATIONSHIP_PLOT_KINDS = {
    "saved-columns-plot", "saved-scan-lifetime-plot", "saved-scan-width-plot", "saved-scan-branching-plot",
}


def _saved_plot_relationship_axes(record: dict, payload: dict, channel: str | None = None) -> tuple[str, str] | None:
    """Recover the actual plotted y column, including channel-specific scan plots."""
    x_column, y_column = payload.get("x_column"), payload.get("y_column")
    if record.get("kind") != "saved-columns-plot":
        channels = payload.get("channels") or []
        selected = channel if channel in channels else channels[0] if len(channels) == 1 else None
        quantity = payload.get("quantity")
        if selected is None or quantity not in {"lifetime", "width", "branching"}:
            return None
        y_column = _channel_column(selected, quantity)
    return (x_column, y_column) if x_column and y_column else None


def _plot_relationship_description_context(question: str) -> dict | None:
    """Resolve a plot or explicitly named dataset interpretation request."""
    lowered = " ".join(question.lower().split())
    if not any(word in lowered for word in ("describe", "explain", "interpret", "behaviour", "behavior", "trend", "correlation", "relationship")):
        return None
    filepath = _explicit_filepath_from_text(question)
    if filepath and filepath.lower().endswith((".h5", ".hdf5")) and ".png" not in lowered:
        channel = _extract_channel(question)
        return {
            "dataset_filepath": str(_resolve_project_path(filepath)),
            "x_column": "MGUT" if re.search(r"\bgut scale\b|\bm_?gut\b", lowered) else None,
            "y_column": _channel_column(channel, "lifetime") if channel and "lifetime" in lowered else None,
        }
    if not any(word in lowered for word in ("plot", "graph", "figure", ".png")):
        return None

    mentioned = re.findall(r"[^\s\"'`]+\.png", question, flags=re.IGNORECASE)
    selected = None
    for record in reversed(list(ARTIFACT_REGISTRY.values())):
        if record.get("kind") not in _RELATIONSHIP_PLOT_KINDS:
            continue
        savefig = str((record.get("summary") or {}).get("savefig", ""))
        if not mentioned:
            selected = record
            break
        normalized_savefig = _normalize_path_text(savefig)
        if any(
            normalized_savefig.endswith(_normalize_path_text(item).lstrip("/"))
            or Path(normalized_savefig).name == Path(_normalize_path_text(item)).name
            for item in mentioned
        ):
            selected = record
            break
    if selected is None:
        return None

    _, payload = _load_artifact_payload(selected["artifact_id"])
    if not isinstance(payload, dict):
        return None
    axes = _saved_plot_relationship_axes(selected, payload, _extract_channel(question))
    if axes is None:
        return None
    return {
        "plot_artifact_id": selected["artifact_id"],
        "plot_filepath": payload.get("savefig"),
        "x_column": axes[0],
        "y_column": axes[1],
        "source_scan_artifact_id": payload.get("source_scan_artifact_id"),
    }


def _relationship_grounding_instruction(context: dict) -> str:
    if context.get("dataset_filepath"):
        return (
            "SAVED DATASET RELATIONSHIP GROUNDING: Before interpreting this dataset, call "
            "analyze_saved_columns_relationship with filepath="
            f"{context['dataset_filepath']!r}. "
            f"The explicitly identified columns are x={context.get('x_column')!r}, "
            f"y={context.get('y_column')!r}; resolve any unspecified column from the saved data "
            "and conversation, and identify the analyzed channel in the answer. "
            "Use its returned statistics and varying-input metadata. Pearson correlation is "
            "not a power-law exponent. Scan-bin medians do not represent fixed-input sensitivity. "
            "For non-SUSY gauge-mediated decay at fixed coupling and flavor, lifetime scales "
            "as MGUT^4: a tenfold mass increase gives a 10,000-fold lifetime increase. "
            "Keep this theoretical scaling separate from the dataset's marginal correlation."
        )
    return (
        "PLOT RELATIONSHIP GROUNDING: This is a request to interpret a saved scatter plot. "
        "Before describing any direction, trend, correlation, threshold, or dependence, call "
        "analyze_saved_columns_relationship with "
        f"x_column={context['x_column']!r}, y_column={context['y_column']!r}, and "
        f"scan_artifact_id={context['plot_artifact_id']!r}. Base the conclusion on its Pearson, "
        "Spearman, and binned summaries. Do not infer an exponential law, threshold, or causation "
        "from the image alone. This is a marginal correlation across scan points, not a controlled "
        "variation of one parameter. Report the other varying inputs returned by the tool; do not "
        "equate negligible correlation with no physical dependence. Use the plotted channel's "
        "partial lifetime, not a total lifetime. Keep the explanation concise and leave speculation "
        "about flavor mechanisms to calculations that actually isolate them. State the exact plot filepath returned in this context: "
        f"{context.get('plot_filepath')}."
    )


def _latest_named_tool_payload(messages: list, name: str, start: int = 0) -> dict | None:
    for message in reversed(messages[start:]):
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload = json.loads(message.content)
        except Exception:
            continue
        if payload.get("tool_name") == name and not payload.get("error"):
            return payload
    return None


def _latest_current_rge_run(messages: list) -> dict | None:
    start = max((i for i, message in enumerate(messages) if isinstance(message, HumanMessage)), default=-1) + 1
    for message in reversed(messages[start:]):
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload = json.loads(message.content)
        except (TypeError, ValueError):
            continue
        if (payload.get("tool_name") in {"run_sm_gauge_rges", "run_nonsusy_gauge_rges", "run_susy_gauge_rges"}
                and not payload.get("error") and payload.get("artifact_id")):
            return payload
    return None


def _latest_current_scan_result(messages: list) -> dict | None:
    start = max((i for i, message in enumerate(messages) if isinstance(message, HumanMessage)), default=-1) + 1
    for message in reversed(messages[start:]):
        if not isinstance(message, ToolMessage):
            continue
        try:
            result = json.loads(message.content)
        except (TypeError, ValueError):
            continue
        if result.get("tool_name") in _SCAN_EXECUTION_TOOL_NAMES and result.get("scan_filepath") and not result.get("error"):
            return result
    return None


def _rge_plot_request_error(args: dict, messages: list) -> str | None:
    """Check the requested physics and current-turn provenance before plotting."""
    record, run = _load_rge_run(args.get("rge_artifact_id", "latest_rge"))
    if record is None or run is None:
        return "No valid running artifact exists. Call the requested RGE running tool before plotting."
    requested_end = _rge_scale_overrides_from_messages(messages).get("gut_scale")
    if requested_end is not None:
        actual_end = run.get("final_scale_GeV", run.get("gut_scale_GeV"))
        if actual_end is None or not np.isclose(float(actual_end), requested_end, rtol=1e-12, atol=0):
            return (f"The saved run ends at {actual_end} GeV, but the user requested {requested_end} GeV. "
                    "Run the same model with the requested gut_scale, then plot that returned artifact.")
    question = _latest_human_text(messages)
    current_run = _latest_current_rge_run(messages)
    # An explicit field declaration is a calculation request, not authorization
    # to reuse arbitrary globally saved physics.
    if _parse_field_tuple(question) is not None and current_run is None:
        return ("This request specifies a model, but no successful RGE running tool was called in this turn. "
                "Call the running tool with the requested fields and masses, then plot its returned artifact_id. "
                "Do not plot a previous model or use latest_rge as a substitute.")
    if current_run is not None and record.get("artifact_id") != current_run["artifact_id"]:
        return "The plot source differs from this turn's running result. Use its returned artifact_id."
    spec = _explicit_single_sm_running_spec(question)
    if spec is not None:
        expected = spec["fields"][0]
        thresholds = run.get("thresholds", [])
        if len(thresholds) != 1:
            return "The saved model has a different field content from the requested single multiplet. Run the requested model."
        actual = _legacy_sm_threshold_field(thresholds[0])
        same_rep = (_canonical_sm_field_key(actual.get("su3"), actual.get("su2"), actual.get("hypercharge"))
                    == _canonical_sm_field_key(expected["su3"], expected["su2"], expected["hypercharge"]))
        mass = actual.get("mass_GeV", actual.get("mass"))
        same_mass = mass is not None and np.isclose(float(mass), expected["mass"], rtol=1e-12, atol=0)
        same_kind = actual.get("kind") == expected["kind"]
        same_reality = bool(actual.get("real_scalar", False)) == bool(expected.get("real_scalar", False))
        same_fermion = actual.get("fermion_type", "weyl") == expected.get("fermion_type", "weyl")
        if not (same_rep and same_mass and same_kind and same_reality and same_fermion
                and int(actual.get("generations", 1)) == 1):
            return "The saved multiplet, mass, or reality differs from the current request. Run the requested model."
        if run.get("loops") != spec["loops"]:
            return "The saved loop order differs from the current request. Run with the requested loop order."
        expected_end = requested_end or spec.get("gut_scale", 1e16)
        actual_end = run.get("final_scale_GeV", run.get("gut_scale_GeV"))
        if actual_end is None or not np.isclose(float(actual_end), expected_end, rtol=1e-12, atol=0):
            return "The saved endpoint differs from the requested or default interval. Run the requested model."
        expected_start = _rge_scale_overrides_from_messages(messages).get("initial_scale", spec.get("initial_scale", DEFAULT_MZ))
        actual_start = run.get("initial_scale_GeV")
        if actual_start is None or not np.isclose(float(actual_start), expected_start, rtol=1e-12, atol=0):
            return "The saved starting scale differs from the requested or default interval. Run the requested model."
    if args.get("quantity", "coupling") != _rge_plot_quantity_from_messages(messages):
        return "The plotted quantity differs from the request. Use coupling for g_i and inverse_alpha for inverse couplings."
    return None


def _current_request_rge_plot_payload(messages: list, start: int = 0) -> dict | None:
    payload = _latest_named_tool_payload(messages, "plot_rge_running", start)
    if payload is None:
        return None
    args = {"rge_artifact_id": payload.get("source_rge_artifact_id", ""),
            "quantity": payload.get("quantity")}
    return payload if _rge_plot_request_error(args, messages) is None else None


def _latest_verified_plot_payload(messages: list, start: int = 0) -> dict | None:
    for message in reversed(messages[start:]):
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload = json.loads(message.content)
        except Exception:
            continue
        if payload.get("tool_name") in _PLOT_TOOL_NAMES and payload.get("verified") and not payload.get("error"):
            return payload
    return None


def _rge_coefficient_intent(question: str) -> bool:
    lowered = " ".join(question.lower().split())
    return bool(
        re.search(r"\b(?:rge|beta|β)[- ]?(?:function )?coefficients?\b", lowered)
        or ("beta" in lowered and "contribution" in lowered)
    )


def _rge_running_plot_intent(question: str) -> bool:
    lowered = " ".join(question.lower().split())
    wants_plot = bool(re.search(r"\b(?:plot|graph|figure|visuali[sz]e|show)\b", lowered))
    mentions_running = bool(re.search(r"\b(?:rge|running|run|gauge coupling|threshold|mssm)\b", lowered))
    return wants_plot and mentions_running


def _explicit_rge_plot_quantity(question: str) -> str | None:
    text = unicodedata.normalize("NFKC", question or "").translate(_DASH_TRANSLATION).lower()
    text = re.sub(r"\b(?:not|instead of|rather than)\s+(?:the\s+)?(?:inverse|reciprocal)\s+(?:gauge\s+)?couplings?", "", text)
    compact = re.sub(r"[\s{}\\]", "", text).replace("α", "alpha")
    if re.search(r"\b(?:inverse|reciprocal)\b", text) or re.search(
        r"alpha(?:_?[i123])?\^?-1|1/alpha|frac1alpha", compact
    ):
        return "inverse_alpha"
    if re.search(r"\bcouplings?\b|\bg_?i\b", text):
        return "coupling"
    return None


def _rge_plot_quantity_from_messages(messages: list) -> str:
    question = _latest_human_text(messages)
    explicit = _explicit_rge_plot_quantity(question)
    if explicit is not None:
        return explicit
    if re.search(r"\b(?:same|previous|again)\b", question.lower()):
        humans = [message for message in messages if isinstance(message, HumanMessage)]
        for message in reversed(humans[:-1]):
            if _domain_signal(str(message.content)) == "proton":
                break
            previous = _explicit_rge_plot_quantity(str(message.content))
            if previous is not None:
                return previous
    return "coupling"


def _explicit_single_sm_running_spec(question: str) -> dict | None:
    """Ground an unambiguous single SM multiplet and its activation mass.

    Other requests remain with the specialist; never guess missing masses or
    reinterpret a symmetry-breaking chain as an SM interval.
    """
    text = _normalize_rge_request_text(question)
    if not _rge_running_plot_intent(text) or len(re.findall(r"\([^()]+\)", text)) != 1:
        return None
    if re.search(r"\b(?:susy|mssm|superfield|3221|422|421|3211|pati|left.right|generations?|copies|multiplicity|yukawa)\b", text, re.I):
        return None
    # Boundary-input overrides need the specialist's general argument parser.
    if re.search(r"\b(?:g[123]|alpha[123]|yt|yb|ytau|n_points)\b|\b\d+\s+(?:points|samples)\b", text, re.I):
        return None
    representation = re.search(r"\([^()]+\)", text)
    number = r"(?:\d+(?:\.\d+)?\s*[x*]\s*10\s*\^\s*[+-]?\d+|10\s*\^\s*[+-]?\d+|\d+(?:\.\d+)?(?:e[+-]?\d+)?)"
    masses = list(re.finditer(rf"\bat\s+({number}(?:\s*(?:GeV|TeV|PeV|EeV))?)", text[representation.end():], re.I))
    if len(masses) != 1:
        return None
    mass = _parse_scale(masses[0].group(1))
    if mass is None or not np.isfinite(mass) or mass <= 0:
        return None
    try:
        field = _parse_sm_or_mssm_field(text, mass, susy=False)
        if field is None:
            return None
        field = _repair_rge_scalar_reality_args(
            "run_sm_gauge_rges", {"fields": [field]}, [HumanMessage(content=question)]
        )["fields"][0]
    except (TypeError, ValueError):
        return None
    return {"fields": [field], **_explicit_rge_scale_overrides(text),
            "loops": _explicit_rge_loop_order(text) or 1}


def _sm_unification_effect_intent(question: str) -> bool:
    """Recognize a qualitative SM field-unification follow-up."""
    lowered = " ".join(question.lower().split())
    mentions_unification = bool(re.search(r"\b(?:unif(?:y|ies|ied|ication)|gut)\b", lowered))
    asks_effect = bool(re.search(
        r"\b(?:help|improv(?:e|ed|ement)|effect|impact|useful|good|bad|closer|worse|"
        r"contribution|best|mass|scale|threshold|reside)\b", lowered
    ))
    return mentions_unification and asks_effect and not _rge_running_plot_intent(question)


def _sm_unification_threshold_scale_intent(question: str) -> bool:
    lowered = " ".join(question.lower().split())
    return bool(
        re.search(r"\b(?:unif(?:y|ies|ied|ication)|gut)\b", lowered)
        and re.search(r"\b(?:best|mass|scale|threshold|reside|where|heavy|heavier)\b", lowered)
    )


def _latest_sm_fields_for_unification(messages: list) -> list[dict] | None:
    """Recover the newest explicit SM field declaration across a follow-up."""
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        try:
            fields = _parse_sm_or_mssm_fields(str(message.content), 1.0, susy=False)
        except (TypeError, ValueError):
            continue
        if not fields:
            continue
        for field in fields:
            field.pop("mass", None)
        repaired = _repair_rge_scalar_reality_args(
            "analyze_sm_unification_effect", {"fields": fields}, messages
        )
        return repaired["fields"]
    return None


def _sm_unification_fallback_payload(messages: list) -> dict | None:
    fields = _latest_sm_fields_for_unification(messages)
    if not fields:
        return None
    result = analyze_sm_unification_effect.invoke({"fields": fields})
    return result if isinstance(result, dict) and not result.get("error") else None


def _latest_rge_coefficient_payload(messages: list, start: int = 0) -> dict | None:
    for name in _RGE_COEFFICIENT_TOOL_NAMES:
        payload = _latest_named_tool_payload(messages, name, start)
        if payload is not None and "sm_one_loop" in payload and "sm_two_loop" in payload:
            return payload
    return None


def _fraction_latex(value: object) -> str:
    text = str(value).strip()
    if "/" not in text:
        return text
    numerator, denominator = text.split("/", 1)
    sign = "-" if numerator.startswith("-") else ""
    numerator = numerator.lstrip("+-")
    return rf"{sign}\frac{{{numerator}}}{{{denominator}}}"


def _fraction_vector_latex(values: list[object]) -> str:
    return r"\left(" + r",\;".join(_fraction_latex(value) for value in values) + r"\right)"


def _fraction_matrix_latex(values: list[list[object]]) -> str:
    rows = [" & ".join(_fraction_latex(value) for value in row) for row in values]
    return r"\begin{pmatrix}" + r" \\ ".join(rows) + r"\end{pmatrix}"


def _format_rge_coefficient_answer(payload: dict, loops: int = 1) -> str:
    """Render exact tool output without asking the LLM to copy fractions or matrices."""
    gauge_order = payload.get("gauge_order", ["U1Y", "SU2L", "SU3C"])
    lines = [
        "**Gauge order:** " + ", ".join(str(item) for item in gauge_order),
        "",
        "**Convention**",
        "",
        (
            r"$$\beta_i=\frac{g_i^3}{16\pi^2}b_i+\frac{g_i^3}{(16\pi^2)^2}\sum_j B_{ij}g_j^2.$$"
            if loops == 2 else r"$$\beta_i^{(1)}=\frac{g_i^3}{16\pi^2}b_i.$$"
        ),
        "",
        f"Showing {loops}-loop gauge coefficients. All coefficients below are exact fractions; scalars are complex by default.",
        "",
        "**Standard Model coefficients**",
        "",
        r"$$b_i^{\mathrm{SM}}=" + _fraction_vector_latex(payload["sm_one_loop"]) + ".$$",
    ]
    if loops == 2:
        lines.extend(["", r"$$B_{ij}^{\mathrm{SM}}=" + _fraction_matrix_latex(payload["sm_two_loop"]) + ".$$"])

    for field in payload.get("fields", []):
        if field.get("kind") == "scalar":
            reality = "real (explicitly requested)" if field.get("real_scalar") else "complex (default)"
        else:
            reality = str(field.get("fermion_type") or "fermion")
        representation = f"({field.get('su3')}, {field.get('su2')}, {field.get('hypercharge')})"
        lines.extend([
            "",
            f"**New contribution: {field.get('name', 'field')} {representation} — {reality}**",
            "",
            r"$$\Delta b_i=" + _fraction_vector_latex(field["one_loop_delta"]) + ".$$",
        ])
        if loops == 2:
            lines.extend(["", r"$$\Delta B_{ij}=" + _fraction_matrix_latex(field["two_loop_delta"]) + ".$$"])

    if payload.get("fields"):
        lines.extend([
            "",
            "**Updated totals**",
            "",
            r"$$b_i=" + _fraction_vector_latex(payload["total_one_loop"]) + ".$$",
        ])
        if loops == 2:
            lines.extend(["", r"$$B_{ij}=" + _fraction_matrix_latex(payload["total_two_loop"]) + ".$$"])
    return "\n".join(lines)


def _answer_uses_field_only_unification_ratio(answer: str, payload: dict) -> bool:
    """Reject s23^(J)/s12^(J), which is not the total unification ratio."""
    text = unicodedata.normalize("NFKC", answer or "").translate(_DASH_TRANSLATION)
    # Match whole values in ratio discussions, not a numerator inside a
    # valid contribution such as -1/3, or an exponent in alpha^{-1}.
    text = re.sub(
        r"(-?)\\(?:d?frac)\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*(\d+)\s*\}",
        lambda match: f"{match.group(1)}{match.group(2)}/{match.group(3)}", text,
    )
    text = re.sub(r"\^\s*(?:\{\s*[+-]?\d+\s*\}|[+-]?\d+)", "", text)
    compact = re.sub(r"[\s{}_]", "", text.lower())
    contexts = [
        compact[match.end():match.end() + 100]
        for match in re.finditer(r"ratio|s23/s12|\\fracs23s12", compact)
    ]
    for effect in payload.get("field_effects", []):
        denominator = Fraction(str(effect["s12"]))
        if denominator == 0:
            continue
        field_ratio = Fraction(str(effect["s23"])) / denominator
        exact = str(field_ratio)
        forms = {
            exact,
            f"{float(field_ratio):.4f}",
        }
        if field_ratio.denominator == 1:
            forms.add(f"{field_ratio.numerator}.0")
        if any(
            re.search(r"(?<![\d./^])" + re.escape(form) + r"(?!\d|/|\.\d|[eE][-+]?\d)", context)
            for context in contexts for form in forms
        ):
            return True
    return False


def _answer_contradicts_unification_direction(answer: str, payload: dict) -> bool:
    """Catch explicit direction and threshold claims inconsistent with the tool."""
    text = unicodedata.normalize("NFKC", answer or "").translate(_DASH_TRANSLATION).lower()
    light = payload.get("light_field_limit") or {}
    change = light.get("ratio_change_from_sm")
    if change is None and light.get("ratio") is not None:
        sm_ratio = (payload.get("sm") or {}).get("ratio")
        if sm_ratio is not None:
            change = "increases" if light["ratio"] > sm_ratio else "decreases" if light["ratio"] < sm_ratio else "unchanged"
    wrong_direction = "reduces?|decreases?|lowers?" if change == "increases" else "raises?|increases?" if change == "decreases" else None
    if wrong_direction:
        for clause in re.split(r"\n|(?<=[.!?])\s+(?=[a-z])", text):
            if re.search(r"\b(?:mass|threshold|heavier|heavy)\b", clause):
                continue
            if re.search(rf"\b(?:{wrong_direction})\s+(?:the\s+)?(?:total\s+|unification\s+)?ratio\b", clause):
                return True
        if change == "increases" and re.search(
            r"light[- ]field(?: limit)? ratio\b.{0,100}\b(?:lower|less|smaller) than (?:the )?sm\b", text
        ):
            return True
    if payload.get("single_field_threshold_status") == "required_weight_exceeds_light_limit":
        if re.search(
            r"\b(?:higher|heavier|raised) (?:mass |field |scalar )?threshold\b.{0,90}"
            r"\b(?:required|needed|unif\w*|crossing)\b", text
        ):
            return True
    return False


def _answer_has_reversed_unification_ratio(answer: str) -> bool:
    compact = re.sub(r"[\s{}_]", "", (answer or "").lower())
    compact = compact.replace(r"\mathcal", "").replace(r"\left", "").replace(r"\right", "")
    return "s12/s23" in compact or r"\fracs12s23" in compact


def _answer_contains_number(answer: str, value: float, *, significant_digits: int) -> bool:
    """Accept numerically equivalent plain or LaTeX values with sensible rounding."""
    text = (answer or "").lower()
    candidates: list[float] = []
    scientific_pattern = re.compile(
        r"(?<![\w.])([+-]?\d+(?:\.\d+)?)\s*"
        r"(?:\\times|×|[x*])\s*10\s*\^?\s*\{?\s*([+-]?\d+)\s*\}?"
    )
    for match in scientific_pattern.finditer(text):
        candidates.append(float(match.group(1)) * 10.0 ** int(match.group(2)))
    for match in re.finditer(r"(?<![\w.])([+-]?\d+(?:\.\d+)?[eE][+-]?\d+)(?![\w.])", text):
        candidates.append(float(match.group(1)))
    for match in re.finditer(r"(?<![\w.])([+-]?\d*\.\d+)(?![\w.])", text):
        candidates.append(float(match.group(1)))

    # Local models commonly shorten a three-significant-figure tool value to
    # two significant figures.  Three percent accepts that presentation while
    # still rejecting a wrong order of magnitude or the old 2e8 result.
    return any(np.isclose(candidate, float(value), rtol=0.03, atol=0.0) for candidate in candidates)


def _unification_answer_needs_revision(answer: str, payload: dict, question: str = "") -> bool:
    """Reject contradictions and requests for boundary inputs already used."""
    normalized = " ".join(unicodedata.normalize("NFKC", answer or "").lower().split())
    asks_for_boundary = bool(
        re.search(
            r"\b(?:provide|supply|specify|confirm|what|which|need)\b[^.!?]{0,180}"
            r"(?:\bgauge couplings?\b|\binitial couplings?\b|\bg[123]\b|"
            r"\breference scale\b|\bstarting scale\b|\binitial scale\b)",
            normalized,
        )
        and (
            "?" in normalized
            or re.search(r"\b(?:please|i(?:['’]ll)? need|we need|you must)\b", normalized)
        )
    )
    scale_intent = _sm_unification_threshold_scale_intent(question)
    solution = payload.get("single_field_threshold_solution") or {}
    field_mass = solution.get("field_mass_GeV")
    missing_requested_mass = bool(
        scale_intent
        and field_mass is not None
        and not _answer_contains_number(answer, float(field_mass), significant_digits=3)
    )
    return bool(
        asks_for_boundary
        or missing_requested_mass
        or _answer_has_reversed_unification_ratio(answer)
        or _answer_uses_field_only_unification_ratio(answer, payload)
        or _answer_contradicts_unification_direction(answer, payload)
    )


def _interpret_sm_unification_payload(target: dict, question: str, payload: dict) -> str:
    """Let the LLM explain verified tool data without giving it calculation tools."""
    scale_intent = _sm_unification_threshold_scale_intent(question)
    scale_note = ""
    if scale_intent and payload.get("single_field_threshold_solution"):
        display = (payload.get("single_field_threshold_solution") or {}).get("display") or {}
        scale_note = (
            "The user requested a threshold scale, so include the tool-derived field mass "
            f"M_J={display.get('field_mass_GeV_3sf')}. Other verified quantities may be included when useful. "
        )
    llm = _build_llm_without_tools(target)
    response = llm.invoke([
        SystemMessage(content=(
            "You are interpreting a completed deterministic SM unification diagnostic. The JSON below is "
            "the sole source of numerical facts; do not perform new arithmetic. Answer the user's question "
            "in your own words, choosing the relevant facts, reasoning, structure, and level of detail. "
            "If you discuss a unification ratio, use the SM-plus-threshold totals S_23/S_12; never use the "
            "quotient of a field's contributions alone or transpose the ratio. "
            "The diagnostic has already run with the boundary inputs in the JSON. "
            "State those inputs and answer the question; do not ask the user to provide or confirm "
            "gauge couplings, a reference scale, or a mass for the light-field limit. "
            "Explicitly label any claimed crossing and threshold mass as one-loop. Do not claim "
            "this mass guarantees two-loop unification or establishes a viable complete GUT. "
            "Use ratio_change_from_sm to describe whether the total ratio rises or falls. "
            "If required_weight_exceeds_light_limit, this field alone is insufficient even when light; "
            "raising its mass weakens its contribution and cannot restore unification. "
            "If no physical threshold solution exists, state that rather than inventing a mass. "
            "Answer briefly; do not dump JSON or propose additional work the user did not request. "
            + scale_note
            + "Do not refer to these instructions.\n\n"
            f"Verified diagnostic: {json.dumps(payload, sort_keys=True)}"
        )),
        HumanMessage(content=question),
    ])
    return str(getattr(response, "content", response)).strip()


def _interpret_mcmc_payload(target: dict, question: str, payload: dict) -> str:
    """Let the specialist LLM explain a verified MCMC result without new tools."""
    llm = _build_llm_without_tools(target)
    response = llm.invoke([
        SystemMessage(content=(
            "You are interpreting a completed deterministic proton-decay MCMC. The JSON below is the "
            "sole source of numerical and file facts; do not perform new calculations or invent values. "
            "Answer the user's request in your own words. Explain what was varied and held fixed, what the "
            "acceptance rate says about the sampler, what fraction of retained samples satisfies the proton-"
            "decay bounds, what the chi-squared and most-constraining-channel information establishes, and "
            "any limitations that follow from the reported diagnostics. Do not confuse proposal acceptance "
            "with the allowed-sample fraction. Include the exact scan_filepath and artifact_id. Do not refer "
            "to these instructions. Never show programming code, reproduction snippets, or assumed HDF5 "
            "dataset layouts. Requested plots are made by plotting tools; do not replace them with instructions "
            "for the user. Do not claim a starting-value override was applied unless the tool explicitly reports it.\n\n"
            f"Verified MCMC result: {json.dumps(payload, sort_keys=True, default=str)}"
        )),
        HumanMessage(content=question),
    ])
    return str(getattr(response, "content", response)).strip()


def _ensure_mcmc_provenance(answer: str, payload: dict) -> str:
    """Append exact machine-verified MCMC identifiers if an LLM omitted them."""
    repaired = (answer or "").strip()
    scan_filepath = str(payload.get("scan_filepath") or "")
    artifact_id = str(payload.get("artifact_id") or "")
    additions = []
    if scan_filepath and scan_filepath not in repaired:
        additions.append(f"Verified MCMC table: `{scan_filepath}`")
    if artifact_id and artifact_id not in repaired:
        additions.append(f"Artifact: `{artifact_id}`")
    if additions:
        repaired = repaired.rstrip() + "\n\n" + "\n".join(additions)
    return repaired


def _mcmc_answer_needs_revision(answer: str, payload: dict) -> bool:
    """Require a substantive LLM answer with exact output provenance."""
    text = (answer or "").strip()
    if len(text.split()) < 20:
        return True
    return any(
        value and str(value) not in text
        for value in (payload.get("scan_filepath"), payload.get("artifact_id"))
    )


def _sm_coefficient_fallback_payload(question: str) -> dict | None:
    """Safely execute an explicit SM coefficient request if the LLM declines to call its tool."""
    lowered = " ".join((question or "").lower().split())
    if not re.search(r"\bstandard model\b|\bsm\b", lowered):
        return None
    try:
        fields = _parse_sm_or_mssm_fields(question, 1.0, susy=False)
    except (TypeError, ValueError):
        return None
    if not fields:
        return None
    for field in fields:
        field.pop("mass", None)
    result = compute_sm_rge_beta_coefficients.invoke({"fields": fields})
    return result if isinstance(result, dict) and not result.get("error") else None


def _repair_answer_plot_filepath(answer: str, plot_payload: dict | None) -> str:
    if not plot_payload:
        return answer
    exact = str(plot_payload.get("savefig") or plot_payload.get("plot_filepath") or "")
    if not exact:
        return answer
    repaired = re.sub(r"[^\s`\"'()<>]+\.png", exact, answer or "", flags=re.IGNORECASE)
    if exact not in repaired:
        repaired = repaired.rstrip() + f"\n\nVerified plot file: `{exact}`"
    return repaired


def _unsupported_relationship_claim(answer: str, analysis: dict) -> bool:
    lowered = " ".join((answer or "").lower().split())
    if analysis.get("x_column") == "MGUT" and str(analysis.get("y_column", "")).startswith("lifetime_"):
        # Reject the specific scaling contradiction without constraining how
        # the LLM discusses genuine marginal trends in the saved scan.
        for sentence in re.split(r"(?<=[.!?])\s+", lowered):
            if re.search(r"\b(?:not|cannot|isn't|doesn't)\b", sentence):
                continue
            if (re.search(r"\b(?:one|1|an) order of magnitude\b|\btenfold\b|\bfactor of 10\b", sentence)
                    and re.search(r"\b(?:two|2) orders of magnitude\b|\bhundredfold\b|\bfactor of 100\b", sentence)
                    and re.search(r"\bfixed\b|\bconsistent\b", sentence)):
                return True
    # Pairwise scan statistics cannot establish physical independence, even
    # when a strong correlation exists or a weak one is absent.
    x_name = re.escape(str(analysis.get("x_column") or "this parameter").lower())
    if re.search(
        rf"\b(?:{x_name}|this parameter)\b.{{0,40}}\b(?:has no (?:physical )?effect|does not affect|doesn't affect)\b"
        r"|\b(?:the )?lifetime\b.{0,30}\b(?:is|remains) independent of\b",
        lowered,
    ):
        return True
    if analysis.get("supports_monotonic_trend"):
        return False
    lowered = re.sub(
        r"\b(?:do(?:es)? not (?:support|show)|doesn't (?:support|show)|"
        r"no evidence (?:for|of)|cannot infer|no|not)\s+"
        r"(?:(?:a|an|any)\s+)?(?:(?:strong|clear|pronounced|positive|negative)\s+)*"
        r"(?:monotonic(?:ally)?(?:\s+(?:trend|relationship|dependence))?|"
        r"correlation|relationship|trend|exponential(?:\s+(?:law|dependence|relationship))?)\b",
        " ",
        lowered,
    )
    patterns = (
        r"\bmonotonic(?:ally)?\b",
        r"\b(?:strong|clear|pronounced)\s+(?:positive|negative|inverse|direct)?\s*(?:trend|correlation|relationship)\b",
        r"\binverse correlation\b",
        r"\bdirect correlation\b",
        r"\bas .{1,80} increases?,? .{1,80} (?:increases?|decreases?)\b",
        r"\bexponential(?:ly)?\b",
        r"\b(?:sharp|critical|parameter)\s+threshold\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def _verified_relationship_fallback(analysis: dict, context: dict) -> str:
    pearson = analysis.get("pearson_raw")
    spearman = analysis.get("spearman_raw")
    log_pearson = analysis.get("pearson_x_vs_log10_y")
    def format_correlation(value) -> str:
        return "undefined" if value is None else f"{float(value):.4f}"
    conclusion = (
        "The data do not support a monotonic relationship."
        if not analysis.get("supports_monotonic_trend")
        else f"The data support a {analysis.get('relationship_strength')} {analysis.get('relationship_direction')} relationship."
    )
    return (
        f"For `{analysis.get('y_column')}` versus `{analysis.get('x_column')}`, "
        f"Pearson r = {format_correlation(pearson)}, Spearman ρ = {format_correlation(spearman)}, and "
        f"Pearson r against log10(y) = {format_correlation(log_pearson)}. {conclusion} "
        "These correlations describe this scan; they do not establish a causal law or parameter independence. "
        + (f"The source dataset is `{context['dataset_filepath']}`."
           if context.get("dataset_filepath") else f"The verified plot is `{context.get('plot_filepath')}`.")
    )


def _interpret_saved_relationship_payload(target: dict, question: str, analysis: dict, context: dict) -> str:
    """Ask the LLM to correct an explanation using the verified scan statistics."""
    llm = _build_llm_without_tools(target)
    response = llm.invoke([
        SystemMessage(content=(
            "Explain the named saved plot or dataset using these verified correlation statistics. "
            "State the plotted partial-lifetime channel and distinguish raw Pearson correlation "
            "from rank correlation and Pearson correlation with log10(lifetime). "
            "Use the reported strength and binned summaries; do not invent a monotonic law, "
            "exponential dependence, or causal flavor mechanism. Other parameters vary across this "
            "scan, so this is not an isolated sensitivity test; weak correlation does not prove "
            "absence of physical dependence. Pearson correlation is not a power-law exponent. "
            "For non-SUSY gauge-mediated decay, at fixed coupling and flavor, tau scales as "
            "MGUT^4: a tenfold MGUT increase means a 10,000-fold lifetime increase, not 100-fold. "
            "Do not describe marginal scan-bin medians as a fixed-input calculation. "
            "Keep the answer concise, without raw JSON.\n"
            + json.dumps({"plot": context, "analysis": analysis}, default=str)
        )),
        HumanMessage(content=question),
    ])
    return str(getattr(response, "content", response)).strip()


def _existing_saved_plot_context(question: str) -> str | None:
    """Return compact context for a follow-up asking to show an existing plot."""
    lowered = " ".join(question.lower().split())
    if not any(word in lowered for word in ("plot", "graph", "figure")):
        return None

    global LAST_PD_RESULT, LAST_PD_RESULT_TEXT
    result = LAST_PD_RESULT
    if result is None:
        restored_result, restored_text, _ = _load_last_pd_result_from_disk()
        result = restored_result
        if restored_result is not None:
            LAST_PD_RESULT = restored_result
            LAST_PD_RESULT_TEXT = restored_text

    candidates = []
    if isinstance(result, dict):
        candidates.append(result)
    record, payload = _load_artifact_payload("last")
    if isinstance(record, dict) and isinstance(record.get("summary"), dict):
        candidates.append(record["summary"])
    if isinstance(payload, dict):
        threshold_result = payload.get("threshold_result")
        if isinstance(threshold_result, dict):
            candidates.append(threshold_result)

    for candidate in candidates:
        plot_filepath = candidate.get("plot_filepath")
        if not plot_filepath:
            continue
        plot_path = Path(str(plot_filepath))
        if not plot_path.exists():
            continue
        grid_path = candidate.get("grid_data_filepath")
        return (
            "EXISTING SAVED PLOT FOLLOW-UP: The requested plot already exists. "
            f"Return the existing plot file exactly: {plot_path}. "
            + (f"Its saved grid data are at {grid_path}. " if grid_path else "")
            + "Do not claim that no scan or plot exists, do not request MGUT/gGUT, "
            "and do not run a new scan."
        )
    return None


def _explicit_lifetime_vs_mgut_plot_spec(question: str) -> dict | None:
    """Resolve a direct saved-data lifetime-vs-MGUT plot request."""
    lowered = " ".join((question or "").lower().split())
    if not any(word in lowered for word in ("plot", "graph", "figure")):
        return None
    if "lifetime" not in lowered or not re.search(r"\b(?:mgut|m_gut|gut scale)\b", lowered):
        return None

    channel = _extract_channel(question)
    record, payload = _load_latest_scan_payload()
    if record is None or payload is None:
        return None
    if channel is None and "most constraining" in lowered:
        candidates = [
            payload.get("summary") if isinstance(payload, dict) else None,
            record.get("summary") if isinstance(record, dict) else None,
            LAST_PD_RESULT if isinstance(LAST_PD_RESULT, dict) else None,
        ]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            channel = candidate.get("best_chi_sq_most_constraining_channel")
            if channel:
                break
    if channel not in CHANNELS:
        return None
    return {
        "channel": channel,
        "scan_artifact_id": record["artifact_id"],
        "savefig": "output/plots/lifetime_vs_MGUT.png",
    }


def _is_tool_call_parse_error(error: BaseException) -> bool:
    """Recognize Ollama failures caused by malformed JSON tool-call output."""
    text = str(error).lower()
    return "error parsing tool call" in text or (
        "tool call" in text and "invalid character" in text
    )


def _run_routed_turn(
        question: str, conversation_history: list = None,
        verbose: bool = True, model: str | None = None,
        api_base: str | None = None, api_key: str | None = None,
        api_email: str | None = None, api_password: str | None = None,
        *, _target: dict | None = None,
        _domain: Literal["rge", "proton"] | None = None) -> tuple:
    """Execute one already-orchestrated turn inside a specialist branch."""

    if conversation_history is None:
        conversation_history = []

    target = _target or _resolve_model_target(
        model,
        api_base=api_base,
        api_key=api_key,
        api_email=api_email,
        api_password=api_password,
    )
    target_key = json.dumps({k: v for k, v in target.items() if k not in {"api_key", "api_password"}}, sort_keys=True)
    domain = _domain or _route_specialist(question, conversation_history, target)
    contextual_plot_question = _contextual_plot_request(question, conversation_history)
    explicit_all_lifetimes_spec = (
        _explicit_all_lifetimes_plot_spec(contextual_plot_question)
        if domain == "proton" and not _explicit_scan_intent(question) else None
    )
    explicit_mass_heatmap_spec = (
        _explicit_mass_lifetime_heatmap_spec(contextual_plot_question)
        if domain == "proton" and not _explicit_scan_intent(question) else None
    )
    if explicit_mass_heatmap_spec is not None:
        messages = _prepare_messages(conversation_history) + [HumanMessage(content=question)]
        explicit_mass_heatmap_spec = _repair_tool_args("plot_saved_columns", explicit_mass_heatmap_spec, messages)
        plot_payload = _verify_tool_result(
            "plot_saved_columns",
            explicit_mass_heatmap_spec,
            plot_saved_columns.invoke(explicit_mass_heatmap_spec),
        )
        if not isinstance(plot_payload, dict) or plot_payload.get("error"):
            detail = (
                plot_payload.get("error", "unknown plotting error")
                if isinstance(plot_payload, dict) else "invalid plot result"
            )
            failure = f"The mass-versus-mass lifetime heatmap did not complete: {detail}"
            return failure, messages + [AIMessage(content=failure)]
        answer = (
            f"Plotted {plot_payload['y_column']} versus {plot_payload['x_column']} with "
            f"{plot_payload['color_by']} as the colour scale. Both mass axes and the lifetime "
            f"colour scale are logarithmic. Verified plot file: `{plot_payload['savefig']}`"
        )
        return answer, messages + [AIMessage(content=answer)]
    if explicit_all_lifetimes_spec is not None:
        messages = _prepare_messages(conversation_history) + [HumanMessage(content=question)]
        explicit_all_lifetimes_spec = _repair_tool_args("plot_saved_scan_quantity", explicit_all_lifetimes_spec, messages)
        wants_separate_plots = any(
            phrase in " ".join(question.lower().split())
            for phrase in ("multiple plots", "each channel", "other channels", "where are the other")
        )
        if wants_separate_plots:
            plot_payloads = []
            requested_x = explicit_all_lifetimes_spec["x_column"]
            x_channel = _extract_channel(str(requested_x))
            x_column = (
                _channel_column(x_channel, "lifetime") if x_channel else requested_x
            )
            for plotted_channel in CHANNELS:
                y_column = _channel_column(plotted_channel, "lifetime")
                plot_args = {
                    "x_column": x_column,
                    "y_column": y_column,
                    "xscale": explicit_all_lifetimes_spec["xscale"],
                    "yscale": "log",
                    "color_by": explicit_all_lifetimes_spec["color_by"],
                    "savefig": f"{y_column}_vs_{x_column}.png",
                }
                if explicit_all_lifetimes_spec.get("filepath"):
                    plot_args["filepath"] = explicit_all_lifetimes_spec["filepath"]
                plot_args = _repair_tool_args("plot_saved_columns", plot_args, messages)
                payload = _verify_tool_result(
                    "plot_saved_columns", plot_args, plot_saved_columns.invoke(plot_args)
                )
                if not isinstance(payload, dict) or payload.get("error"):
                    detail = payload.get("error", "unknown plotting error") if isinstance(payload, dict) else "invalid plot result"
                    failure = f"The separate lifetime plots did not complete: {detail}"
                    return failure, messages + [AIMessage(content=failure)]
                plot_payloads.append(payload)
            paths = [payload["savefig"] for payload in plot_payloads]
            answer = (
                f"Plotted all {len(plot_payloads)} channel lifetimes separately versus {x_channel or x_column}. "
                "Verified plot files:\n" + "\n".join(f"- `{path}`" for path in paths)
            )
            return answer, messages + [AIMessage(content=answer)]

        plot_payload = _verify_tool_result(
            "plot_saved_scan_quantity",
            explicit_all_lifetimes_spec,
            plot_saved_scan_quantity.invoke(explicit_all_lifetimes_spec),
        )
        if not isinstance(plot_payload, dict) or plot_payload.get("error"):
            detail = (
                plot_payload.get("error", "unknown plotting error")
                if isinstance(plot_payload, dict) else "invalid plot result"
            )
            failure = f"The all-channel lifetime plot did not complete: {detail}"
            return failure, messages + [AIMessage(content=failure)]
        answer = (
            f"Plotted all {len(plot_payload.get('channels', []))} channel lifetimes versus "
            f"{plot_payload.get('x_lifetime_channel') or plot_payload.get('x_column')} from the saved data. "
            f"Verified plot file: `{plot_payload['savefig']}`"
        )
        return answer, messages + [AIMessage(content=answer)]

    graph_key = (target_key, domain)

    if graph_key not in _graphs:
        if target["provider"] == "ollama":
            print(f"[INFO] Loading {domain} specialist with local model '{target['model']}' via Ollama...")
        else:
            print(
                f"[INFO] Loading {domain} specialist with remote model '{target['model']}' via "
                f"{target['api_base']}..."
            )
        _graphs[graph_key] = build_graph(target, domain=domain)
        print(f"[INFO] {domain.upper()} specialist ready.\n")
    graph = _graphs[graph_key]
    print(f"[ROUTER] Selected {domain.upper()} specialist.")

    messages = _prepare_messages(conversation_history) + [HumanMessage(content=question)]
    if domain == "proton" and _saved_scan_columns_intent(question):
        source = _saved_plot_scan_context(messages) or {}
        messages.append(SystemMessage(content=(
            "SAVED-SCAN COLUMN REQUEST: Call list_saved_scan_columns once for this exact source. "
            "Report all_columns, including nonnumeric metadata. Do not sample, calculate lifetimes, "
            "plot, or repeat the column tool after it succeeds.\n" + json.dumps(source, sort_keys=True)
        )))
    if domain == "proton" and _is_saved_plot_only_request(question):
        source = _saved_plot_scan_context(messages)
        messages.append(SystemMessage(content=(
            "SAVED-DATA PLOT REQUEST: The user asks to plot existing results. Do not call any "
            "scan or MCMC tool, even if history contains an earlier sampling request. "
            "Choose a plotting tool and use this exact saved scan source; the benchmark input "
            "file is not the sampled results. If plotting fails, correct the plot arguments or "
            "report the error; never regenerate the scan to obtain a plot.\n"
            + json.dumps(source or {"status": "no_saved_scan_found"}, sort_keys=True)
        )))
    if domain == "rge":
        if _rge_running_plot_intent(question):
            messages.append(SystemMessage(content=(
                "RGE PLOT QUANTITY: Use plot_rge_running quantity="
                + json.dumps(_rge_plot_quantity_from_messages(messages))
                + ". Plot gauge couplings g_i by default; inverse_alpha requires an explicit inverse-coupling request."
            )))
        bounds = _rge_scale_overrides_from_messages(messages)
        if bounds:
            messages.append(SystemMessage(content=(
                "RGE RUNNING BOUNDS: Preserve these user-specified running bounds exactly. "
                "Do not substitute a default endpoint or confuse a particle mass threshold with "
                "the final running scale. For a changed loop order, retain this interval.\n"
                + json.dumps(bounds, sort_keys=True)
            )))
    explicit_lifetime_plot_spec = (
        _explicit_lifetime_vs_mgut_plot_spec(question)
        if domain == "proton" and not _explicit_scan_intent(question) else None
    )
    if explicit_lifetime_plot_spec is not None:
        explicit_lifetime_plot_spec = _repair_tool_args("plot_lifetime_vs_MGUT", explicit_lifetime_plot_spec, messages)
        messages.append(SystemMessage(content=(
            "SAVED LIFETIME PLOT GROUNDING: Call plot_lifetime_vs_MGUT exactly once with the "
            "validated arguments below. A saved MCMC artifact is valid plotting input even when it "
            "contains only one unique MGUT value; plot the available point(s) and do not refuse on "
            "that basis. Do not launch a new scan.\n"
            + json.dumps(explicit_lifetime_plot_spec, sort_keys=True)
        )))
    explicit_mcmc_tool_name = None
    explicit_mcmc_spec = None
    if domain == "proton":
        explicit_mcmc_spec = _explicit_nonsusy_mcmc_spec(question)
        if explicit_mcmc_spec is not None:
            explicit_mcmc_tool_name = "run_nonsusy_proton_decay_mcmc"
        else:
            explicit_mcmc_spec = _explicit_susy_mcmc_spec(question)
            if explicit_mcmc_spec is not None:
                explicit_mcmc_tool_name = "run_susy_proton_decay_mcmc"
    if explicit_mcmc_spec is not None:
        matrix_file = _requested_matrix_elements_filepath(question)
        if matrix_file:
            explicit_mcmc_spec["matrix_elements_file"] = matrix_file
        bounds_file = _requested_bounds_filepath(question)
        if bounds_file:
            explicit_mcmc_spec["bounds_file"] = bounds_file
        messages.append(SystemMessage(content=(
            "MCMC ARGUMENT GROUNDING: This is a complete proton-decay MCMC request. "
            f"Call {explicit_mcmc_tool_name} exactly once with the validated arguments below. "
            "Do not ask for fields, representations, thresholds, or any already supplied value. "
            "The tool can select the packaged input file for the requested framework and group. "
            "Do not calculate or estimate the result yourself.\n"
            + json.dumps(explicit_mcmc_spec, sort_keys=True)
        )))
    relationship_context = _plot_relationship_description_context(question) if domain == "proton" else None
    if relationship_context:
        messages.append(SystemMessage(content=_relationship_grounding_instruction(relationship_context)))
    saved_plot_context = _existing_saved_plot_context(question) if domain == "proton" else None
    if saved_plot_context:
        messages.append(SystemMessage(content=saved_plot_context))
    explicit_file_context = _explicit_file_computation_context(messages) if domain == "proton" else None
    if explicit_file_context:
        messages.append(SystemMessage(content=explicit_file_context))
    matrix_file = _requested_matrix_elements_filepath(question) if domain == "proton" else None
    if matrix_file:
        messages.append(SystemMessage(content=(
            f"EXPLICIT HADRONIC INPUT: Pass matrix_elements_file={matrix_file!r} to each "
            "proton-decay computation, scan, or MCMC tool for this request. This is separate "
            "from the model benchmark filepath. Never replace the benchmark filepath with it."
        )))
    bounds_file = _requested_bounds_filepath(question) if domain == "proton" else None
    if bounds_file:
        messages.append(SystemMessage(content=(
            f"EXPLICIT EXPERIMENTAL INPUT: Pass bounds_file={bounds_file!r} to each "
            "proton-decay calculation, scan, MCMC, experimental comparison, or analysis/plot "
            "tool used for this request. It is separate from the model benchmark filepath "
            "and matrix_elements_file. Without an explicit request, use built-in limits."
        )))
    explicit_staged_spec = _explicit_staged_nonsusy_spec(question) if domain == "rge" else None
    explicit_susy_spec = _explicit_sm_to_mssm_spec(question) if domain == "rge" else None
    explicit_sm_spec = _explicit_single_sm_running_spec(question) if domain == "rge" else None
    if explicit_sm_spec is not None:
        explicit_sm_spec.update(_rge_scale_overrides_from_messages(messages))
        messages.append(SystemMessage(content=(
            "SM RUNNING ARGUMENT GROUNDING: This request supplies a single SM multiplet and its "
            "mass threshold. Call run_sm_gauge_rges with these arguments, then plot_rge_running "
            "with the returned artifact_id. The field mass is not the final running scale. "
            "Omitted boundary inputs and endpoint use the documented tool defaults.\n"
            + json.dumps(explicit_sm_spec, sort_keys=True)
        )))
    explicit_rge_spec = explicit_staged_spec or explicit_susy_spec
    if explicit_rge_spec is not None:
        tool_name = "run_nonsusy_gauge_rges" if explicit_staged_spec is not None else "run_susy_gauge_rges"
        messages.append(SystemMessage(content=(
            "RGE ARGUMENT GROUNDING: The explicit transition scales parse to the following validated "
            f"{tool_name} arguments. Call that tool with these exact arguments, then "
            "call plot_rge_running using its artifact_id. Do not ask for Yukawa inputs; this runner is gauge-only.\n"
            + json.dumps(explicit_rge_spec, sort_keys=True)
        )))
    unification_effect_intent = domain == "rge" and _sm_unification_effect_intent(question)
    if unification_effect_intent:
        unification_fields = _latest_sm_fields_for_unification(messages)
        if unification_fields:
            messages.append(SystemMessage(content=(
                "SM UNIFICATION-EFFECT GROUNDING: Call analyze_sm_unification_effect with exactly the "
                "following fields. Omit unspecified g1/g2/g3 to use the documented M_Z defaults; "
                "do not ask the user to provide or confirm these boundary inputs. "
                "Use its structured result as the sole source of numerical physics, but "
                "choose the explanation and relevant details yourself. If you discuss a ratio, use the "
                "SM-plus-threshold total S_23/S_12; never form a ratio from one field's contributions or "
                "transpose it. For a requested scale, use single_field_threshold_solution rather than "
                "performing the logarithmic inversion yourself. Do not judge unification by visual "
                "closeness at an arbitrary endpoint. For a qualitative help question, answer in at most "
                "three short sentences: direction, total ratio versus target, and sufficiency alone. "
                "Do not show tables, JSON, internal flags, or unrequested proposals for new fields "
                "or higher-loop effects.\n"
                + json.dumps({"fields": unification_fields}, sort_keys=True)
            )))

    try:
        final_state = _invoke_agent_graph(graph, messages)
    except ResponseError as e:
        if "invalid model name" in str(e).lower():
            raise RuntimeError(_format_model_error(model)) from e
        if not _is_tool_call_parse_error(e):
            raise
        recovery_instruction = SystemMessage(content=(
            "TOOL-CALL FORMAT RECOVERY: The previous attempt failed before any tool executed because "
            "the tool-call arguments were not valid JSON. Re-evaluate the original user request and call "
            "the appropriate tool now. Emit one tool call whose arguments are a strict JSON object matching "
            "the tool schema. Put no reasoning, prose, Markdown, comments, or tool name inside the arguments. "
            "Copy every explicit filepath exactly from the user's request."
        ))
        try:
            final_state = _invoke_agent_graph(graph, messages + [recovery_instruction])
        except ResponseError as retry_error:
            if _is_tool_call_parse_error(retry_error):
                raise RuntimeError(
                    "The local model produced malformed JSON tool arguments twice; no calculation tool ran. "
                    "Please retry the same request."
                ) from retry_error
            raise

    answer, all_messages = _final_answer_from_state(final_state)
    if domain == "proton" and _saved_scan_columns_intent(question):
        columns = _latest_named_tool_payload(all_messages, "list_saved_scan_columns", len(messages))
        attempted = any(isinstance(message, ToolMessage) and
                        '"tool_name": "list_saved_scan_columns"' in str(message.content)
                        for message in all_messages[len(messages):])
        if columns is None and not attempted:
            source = _saved_plot_scan_context(messages) or {}
            args = _repair_tool_args("list_saved_scan_columns", source, messages)
            try:
                columns = _verify_tool_result("list_saved_scan_columns", args,
                                              _invoke_tool("list_saved_scan_columns", args))
            except Exception as exc:
                columns = {"error": f"{type(exc).__name__}: {exc}"}
            all_messages += [
                AIMessage(content="", tool_calls=[{"name": "list_saved_scan_columns", "args": args,
                                                   "id": "saved-columns-fallback"}]),
                ToolMessage(content=json.dumps({"tool_name": "list_saved_scan_columns", **columns}, default=str),
                            tool_call_id="saved-columns-fallback"),
            ]
        if columns is not None and not columns.get("error") and columns.get("verified"):
            answer = _format_saved_scan_columns(columns)
            return answer, all_messages + [AIMessage(content=answer)]
        if columns is not None and columns.get("error"):
            answer = "The column listing failed: " + str(columns["error"])
            return answer, all_messages + [AIMessage(content=answer)]
    if explicit_mcmc_spec is not None:
        original_answer = answer
        mcmc_payload = _latest_named_tool_payload(
            all_messages, explicit_mcmc_tool_name, len(messages)
        )
        used_deterministic_fallback = False
        if mcmc_payload is None:
            mcmc_payload = TOOL_MAP[explicit_mcmc_tool_name].invoke(explicit_mcmc_spec)
            if isinstance(mcmc_payload, dict) and not mcmc_payload.get("error"):
                print("[MCMC GUARDRAIL] Executed the validated deterministic MCMC after a declined tool call.")
                used_deterministic_fallback = True
                all_messages = all_messages + [ToolMessage(
                    content=json.dumps(
                        {"tool_name": explicit_mcmc_tool_name, **mcmc_payload},
                        default=str,
                    ),
                    tool_call_id=f"guardrail-{explicit_mcmc_tool_name}",
                )]
        if not isinstance(mcmc_payload, dict) or mcmc_payload.get("error"):
            detail = mcmc_payload.get("error", "unknown MCMC error") if isinstance(mcmc_payload, dict) else "invalid tool result"
            failure = f"The deterministic MCMC did not complete: {detail}"
            return failure, messages + [AIMessage(content=failure)]
        if used_deterministic_fallback or _mcmc_answer_needs_revision(answer, mcmc_payload):
            try:
                answer = _interpret_mcmc_payload(target, question, mcmc_payload)
            except Exception:
                answer = _format_mcmc_result(mcmc_payload)
        answer = _ensure_mcmc_provenance(answer, mcmc_payload)
        if answer != original_answer:
            all_messages = all_messages + [AIMessage(content=answer)]
        mcmc_plot_spec = _explicit_mass_lifetime_heatmap_spec(question)
        if mcmc_plot_spec is not None:
            plot_args = _repair_tool_args("plot_saved_columns", mcmc_plot_spec, all_messages)
            plot_args["filepath"] = mcmc_payload["scan_filepath"]
            plot_args["scan_artifact_id"] = mcmc_payload["artifact_id"]
            expected_source = f"filepath::{_resolve_project_path(plot_args['filepath'], must_exist=True)}"
            plot_payload = _latest_named_tool_payload(all_messages, "plot_saved_columns", len(messages))
            matches_requested_plot = (
                plot_payload is not None and plot_payload.get("verified")
                and plot_payload.get("source_scan_artifact_id") == expected_source
                and _resolve_plot_column(plot_payload.get("x_column")) == _resolve_plot_column(plot_args["x_column"])
                and _resolve_plot_column(plot_payload.get("y_column")) == _resolve_plot_column(plot_args["y_column"])
                and plot_payload.get("color_by") == plot_args["color_by"]
                and all(plot_payload.get(key) == plot_args[key] for key in ("xscale", "yscale", "color_scale"))
            )
            if not matches_requested_plot:
                try:
                    plot_payload = _verify_tool_result("plot_saved_columns", plot_args,
                                                       plot_saved_columns.invoke(plot_args))
                except Exception as exc:
                    plot_payload = {"error": f"{type(exc).__name__}: {exc}"}
                all_messages += [
                    AIMessage(content="", tool_calls=[{"name": "plot_saved_columns", "args": plot_args,
                                                       "id": "mcmc-requested-plot"}]),
                    ToolMessage(content=json.dumps({"tool_name": "plot_saved_columns", **plot_payload}, default=str),
                                tool_call_id="mcmc-requested-plot"),
                ]
            if plot_payload.get("error"):
                answer = _code_free_answer(answer) + "\n\nThe MCMC completed, but the requested plot failed: " + str(plot_payload["error"])
            else:
                answer = _code_free_answer(answer)
                answer = _repair_answer_plot_filepath(answer, plot_payload)
                answer += "\n\nThe mass plot uses these MCMC samples, with the requested channel lifetime shown by the colour scale."
            all_messages += [AIMessage(content=answer)]

    if explicit_lifetime_plot_spec is not None:
        plot_payload = _latest_named_tool_payload(
            all_messages, "plot_lifetime_vs_MGUT", len(messages)
        )
        if (
            plot_payload is None
            or plot_payload.get("channel") != explicit_lifetime_plot_spec["channel"]
        ):
            plot_payload = _verify_tool_result(
                "plot_lifetime_vs_MGUT",
                explicit_lifetime_plot_spec,
                plot_lifetime_vs_MGUT.invoke(explicit_lifetime_plot_spec),
            )
            if isinstance(plot_payload, dict) and not plot_payload.get("error"):
                print("[PLOT GUARDRAIL] Plotted the saved lifetime data after a declined or mismatched tool call.")
        if not isinstance(plot_payload, dict) or plot_payload.get("error"):
            detail = (
                plot_payload.get("error", "unknown plotting error")
                if isinstance(plot_payload, dict) else "invalid plot result"
            )
            failure = f"The saved lifetime plot did not complete: {detail}"
            return failure, messages + [AIMessage(content=failure)]
        answer = (
            f"Plotted {plot_payload['channel']} lifetime versus MGUT from the saved data. "
            f"{plot_payload.get('plot_note', '')} Verified plot file: `{plot_payload['savefig']}`"
        ).strip()
        all_messages = all_messages + [AIMessage(content=answer)]

    if (
        explicit_file_context
        and not _current_turn_used_tool(all_messages, len(messages))
        and _asks_for_filepath_already_supplied(answer)
    ):
        retry_instruction = SystemMessage(content=(
            f"FILEPATH CLARIFICATION REJECTED: `{_latest_explicit_input_filepath(messages)}` was already "
            "supplied explicitly by the user. Discard the clarification response. Call the matching "
            "*_from_file calculation tool now with that exact filepath, then report its result."
        ))
        retry_state = _invoke_agent_graph(graph, all_messages + [retry_instruction])
        retry_answer, retry_messages = _final_answer_from_state(retry_state)
        retry_suffix = retry_messages[len(all_messages) + 1:]
        if (
            _asks_for_filepath_already_supplied(retry_answer)
            and not any(isinstance(message, ToolMessage) for message in retry_suffix)
        ):
            failure = (
                "The model ignored an explicitly supplied filepath twice and no calculation tool ran. "
                "Please retry after selecting a more reliable tool-calling model."
            )
            return failure, messages + [AIMessage(content=failure)]
        answer, all_messages = retry_answer, retry_messages

    if relationship_context:
        relationship_analysis = _latest_named_tool_payload(
            all_messages, "analyze_saved_columns_relationship", len(messages)
        )
        if relationship_analysis is None:
            retry_instruction = SystemMessage(content=(
                "RELATIONSHIP INTERPRETATION REJECTED: No correlation tool was called. Call "
                "analyze_saved_columns_relationship now using the source and columns in the "
                "relationship grounding instruction, then answer from that result."
            ))
            retry_state = _invoke_agent_graph(graph, all_messages + [retry_instruction])
            answer, all_messages = _final_answer_from_state(retry_state)
            relationship_analysis = _latest_named_tool_payload(
                all_messages, "analyze_saved_columns_relationship", len(messages)
            )
        if relationship_analysis is None:
            failure = (
                "I cannot describe the relationship because the required correlation analysis did not run. "
                "The saved data are unchanged."
            )
            return failure, messages + [AIMessage(content=failure)]
        if _unsupported_relationship_claim(answer, relationship_analysis):
            answer = _interpret_saved_relationship_payload(target, question, relationship_analysis, relationship_context)
            if _unsupported_relationship_claim(answer, relationship_analysis):
                answer = _verified_relationship_fallback(relationship_analysis, relationship_context)
            all_messages = all_messages + [AIMessage(content=answer)]

    if _rge_coefficient_intent(question):
        coefficient_payload = _latest_rge_coefficient_payload(all_messages, len(messages))
        if coefficient_payload is None:
            retry_instruction = SystemMessage(content=(
                "RGE COEFFICIENT GROUNDING: Call compute_sm_rge_beta_coefficients or "
                "compute_nonsusy_rge_beta_coefficients now. Scalars are complex by default; set "
                "real_scalar=true only if the user explicitly requested a real scalar. Do not calculate, "
                "copy, or reformat any coefficient yourself."
            ))
            retry_state = _invoke_agent_graph(graph, all_messages + [retry_instruction])
            _, all_messages = _final_answer_from_state(retry_state)
            coefficient_payload = _latest_rge_coefficient_payload(all_messages, len(messages))
        if coefficient_payload is None:
            coefficient_payload = _sm_coefficient_fallback_payload(question)
            if coefficient_payload is not None:
                print("[RGE GUARDRAIL] Executed the deterministic SM coefficient tool after two declined tool calls.")
        if coefficient_payload is None:
            failure = "I cannot report RGE coefficients because the coefficient tool did not run successfully."
            return failure, messages + [AIMessage(content=failure)]
        answer = _format_rge_coefficient_answer(coefficient_payload, loops=_explicit_rge_loop_order(question) or 1)
        all_messages = all_messages + [AIMessage(content=answer)]

    if unification_effect_intent or (
        domain == "rge" and _latest_named_tool_payload(
            all_messages, "analyze_sm_unification_effect", len(messages)
        ) is not None
    ):
        used_unification_fallback = False
        unification_payload = _latest_named_tool_payload(
            all_messages, "analyze_sm_unification_effect", len(messages)
        )
        if unification_payload is None:
            retry_instruction = SystemMessage(content=(
                "UNIFICATION TOOL REQUIRED: The previous draft did not call the diagnostic. Call "
                "analyze_sm_unification_effect now using the exact field supplied in the grounding "
                "instruction, then explain its exact s_12 and s_23 results in your own words."
            ))
            retry_state = _invoke_agent_graph(graph, all_messages + [retry_instruction])
            answer, all_messages = _final_answer_from_state(retry_state)
            unification_payload = _latest_named_tool_payload(
                all_messages, "analyze_sm_unification_effect", len(messages)
            )
        if unification_payload is None:
            unification_payload = _sm_unification_fallback_payload(messages)
            if unification_payload is not None:
                used_unification_fallback = True
                all_messages = all_messages + [ToolMessage(
                    content=json.dumps({"tool_name": "analyze_sm_unification_effect", **unification_payload}),
                    tool_call_id="guardrail-analyze-sm-unification-effect",
                )]
                print("[RGE GUARDRAIL] Executed the diagnostic after declined tool calls; the LLM still writes the interpretation.")
        if unification_payload is None:
            failure = (
                "I cannot assess this field's unification effect because no explicit SM field "
                "representation could be recovered from the current conversation."
            )
            return failure, messages + [AIMessage(content=failure)]
        if used_unification_fallback or _unification_answer_needs_revision(answer, unification_payload, question):
            answer = _interpret_sm_unification_payload(target, question, unification_payload)
            all_messages = all_messages + [AIMessage(content=answer)]
        if _unification_answer_needs_revision(answer, unification_payload, question):
            failure = (
                "The diagnostic succeeded, but the language model did not preserve all required "
                "tool-grounded facts in the explanation. No unverified interpretation was returned."
            )
            return failure, all_messages + [AIMessage(content=failure)]

    if domain == "rge" and _rge_running_plot_intent(question):
        plot_payload = _current_request_rge_plot_payload(all_messages, len(messages))
        if plot_payload is None:
            run_used = any(
                _latest_named_tool_payload(all_messages, name, len(messages)) is not None
                for name in ("run_sm_gauge_rges", "run_nonsusy_gauge_rges", "run_susy_gauge_rges")
            )
            retry_instruction = SystemMessage(content=(
                "RGE PLOT ENFORCEMENT: The user explicitly requested a plot, but no successful plot_rge_running "
                "tool result exists. "
                + (
                    "The RGE calculation already ran; call plot_rge_running with its artifact_id now."
                    if run_used else
                    "Call the appropriate RGE running tool with the user's exact scales, fields, thresholds, and "
                    "real/complex choices; then call plot_rge_running with the returned artifact_id."
                )
                + " Do not claim completion until the plotting tool returns an existing file."
            ))
            retry_state = _invoke_agent_graph(graph, all_messages + [retry_instruction])
            answer, all_messages = _final_answer_from_state(retry_state)
            plot_payload = _current_request_rge_plot_payload(all_messages, len(messages))
        if plot_payload is None:
            fallback_spec = explicit_staged_spec or _explicit_staged_nonsusy_spec(question)
            fallback_susy_spec = explicit_susy_spec or _explicit_sm_to_mssm_spec(question)
            run_payload = None
            detail = None
            for run_name in ("run_sm_gauge_rges", "run_nonsusy_gauge_rges", "run_susy_gauge_rges"):
                run_payload = _latest_named_tool_payload(all_messages, run_name, len(messages))
                if run_payload is not None:
                    bounds = _rge_scale_overrides_from_messages(messages)
                    if "gut_scale" in bounds:
                        actual_end = run_payload.get("final_scale_GeV", run_payload.get("gut_scale_GeV"))
                        if actual_end is None or not np.isclose(float(actual_end), bounds["gut_scale"], rtol=1e-12, atol=0):
                            detail = "The completed run does not match the requested endpoint."
                            run_payload = None
                            continue
                    if explicit_sm_spec is not None:
                        source_error = _rge_plot_request_error(
                            {"rge_artifact_id": run_payload["artifact_id"],
                             "quantity": _rge_plot_quantity_from_messages(messages)}, all_messages
                        )
                        if source_error:
                            detail = source_error
                            run_payload = None
                            continue
                    break
            fallback_name, fallback_args = (
                ("run_nonsusy_gauge_rges", fallback_spec) if fallback_spec is not None else
                ("run_susy_gauge_rges", fallback_susy_spec) if fallback_susy_spec is not None else
                ("run_sm_gauge_rges", explicit_sm_spec)
            )
            try:
                if run_payload is None and fallback_args is not None:
                    run_payload = _verify_tool_result(
                        fallback_name, fallback_args, _invoke_tool(fallback_name, fallback_args)
                    )
                    all_messages += [
                        AIMessage(content="", tool_calls=[{"name": fallback_name, "args": fallback_args,
                                                           "id": "rge-fallback-run"}]),
                        ToolMessage(content=json.dumps({"tool_name": fallback_name, **run_payload}, default=str),
                                    tool_call_id="rge-fallback-run"),
                    ]
                if isinstance(run_payload, dict) and not run_payload.get("error"):
                    plot_args = {"rge_artifact_id": run_payload["artifact_id"],
                                 "quantity": _rge_plot_quantity_from_messages(messages), "savefig": "rge_running.png"}
                    source_error = _rge_plot_request_error(plot_args, all_messages)
                    if source_error:
                        raise ValueError(source_error)
                    plot_payload = _verify_tool_result(
                        "plot_rge_running", plot_args, _invoke_tool("plot_rge_running", plot_args)
                    )
                    all_messages += [
                        AIMessage(content="", tool_calls=[{"name": "plot_rge_running", "args": plot_args,
                                                           "id": "rge-fallback-plot"}]),
                        ToolMessage(content=json.dumps({"tool_name": "plot_rge_running", **plot_payload}, default=str),
                                    tool_call_id="rge-fallback-plot"),
                    ]
                    if not plot_payload.get("error") and _current_request_rge_plot_payload(all_messages, len(messages)) is not None:
                        quantity_label = "inverse gauge couplings" if plot_payload["quantity"] == "inverse_alpha" else "gauge couplings g_i"
                        answer = (
                            f"Plotted the {quantity_label} using {plot_payload.get('loops')}-loop running. "
                            f"Verified plot file: `{plot_payload['savefig']}`"
                        )
                        all_messages += [AIMessage(content=answer)]
                        print("[RGE GUARDRAIL] Completed verified run-and-plot tools after declined tool calls.")
                    else:
                        detail = plot_payload.get("error", "The plot source does not match the current request.")
                        plot_payload = None
                elif isinstance(run_payload, dict):
                    detail = run_payload.get("error")
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                plot_payload = None
            if plot_payload is None:
                failure = "I cannot report an RGE plot because the required running-and-plot tool workflow did not complete."
                if detail:
                    failure += f" Tool error: {detail}"
                return failure, all_messages + [AIMessage(content=failure)]

    if (
        domain == "proton"
        and
        _turn_response_mode(question, all_messages[len(messages):]) == "computation"
        and _contains_unsolicited_evasion_guidance(answer)
    ):
        rewrite_instruction = SystemMessage(content=(
            f"{COMPUTATION_REPORT_PROMPT}\n\n"
            "SCOPE VALIDATION FAILED: The previous draft included unsolicited bound-evasion guidance. "
            "Rewrite the answer using the existing tool result. Remove every recommendation, parameter-change "
            "strategy, scan suggestion, threshold estimate, and offer of further work. Do not call another tool."
        ))
        rewrite_state = _invoke_agent_graph(graph, all_messages + [rewrite_instruction])
        rewritten_answer, rewritten_messages = _final_answer_from_state(rewrite_state)
        if _contains_unsolicited_evasion_guidance(rewritten_answer):
            scope_failure = (
                "The model did not follow computation-report scope after a constrained rewrite. "
                "No unsolicited parameter guidance will be shown; please retry the calculation."
            )
            return scope_failure, messages + [AIMessage(content=scope_failure)]
        answer, all_messages = rewritten_answer, rewritten_messages

    if (
        domain == "proton"
        and
        _turn_response_mode(question, all_messages[len(messages):]) == "evasion"
        and not _quantitative_evasion_intent(question)
        and _contains_unrequested_quantitative_evasion(answer)
    ):
        rewrite_instruction = SystemMessage(content=(
            f"{BOUND_EVASION_DISCUSSION_PROMPT}\n\n"
            "EVASION SCOPE VALIDATION FAILED: The user requested qualitative guidance, but the previous draft "
            "derived an unrequested numerical threshold/range, proposed a scan, asserted an unsupported monotonic "
            "direction for nonlinear parameters, or mentioned an RGE analysis that was not run. Rewrite using the "
            "existing latest-result tool output. Do not call another tool and do not repeat the lifetime table."
        ))
        rewrite_state = _invoke_agent_graph(graph, all_messages + [rewrite_instruction])
        rewritten_answer, rewritten_messages = _final_answer_from_state(rewrite_state)
        if _contains_unrequested_quantitative_evasion(rewritten_answer):
            scope_failure = (
                "The model did not follow qualitative evasion scope after a constrained rewrite. "
                "No invented threshold, scan range, or unsupported parameter direction will be shown; please retry."
            )
            return scope_failure, messages + [AIMessage(content=scope_failure)]
        answer, all_messages = rewritten_answer, rewritten_messages

    repaired_plot_answer = _repair_answer_plot_filepath(
        answer,
        _latest_verified_plot_payload(all_messages, len(messages)),
    )
    if repaired_plot_answer != answer:
        answer = repaired_plot_answer
        all_messages = all_messages + [AIMessage(content=answer)]

    if _contains_numeric_claim(answer) and not _current_turn_used_tool(all_messages, len(messages)):
        if domain == "rge":
            grounding_text = (
                "NUMERICAL GROUNDING ENFORCEMENT: Your draft contained numerical RGE claims but no tool was "
                "called in this turn. Discard it and call the appropriate RGE coefficient or running tool. "
                "Use only values returned by that tool."
            )
        else:
            grounding_text = (
                "NUMERICAL GROUNDING ENFORCEMENT: Your draft contained numerical claims but no tool was "
                "called in this user turn. Discard that draft. Call the appropriate tool now. For a "
                "follow-up about an earlier proton-decay computation, call get_latest_proton_decay_result. "
                "Use only exact numbers returned by that tool, including experimental bounds and ratios."
            )
        grounding_instruction = SystemMessage(content=grounding_text)
        retry_messages = messages + [grounding_instruction]
        retry_state = _invoke_agent_graph(graph, retry_messages)
        retry_answer, retry_all_messages = _final_answer_from_state(retry_state)
        retry_suffix = retry_all_messages[len(retry_messages):]
        if _contains_numeric_claim(retry_answer) and not any(
            isinstance(message, ToolMessage) for message in retry_suffix
        ):
            refusal = (
                "I cannot provide numerical values because no grounding tool was called. "
                "Please rerun the requested calculation."
            )
            return refusal, messages + [AIMessage(content=refusal)]
        return retry_answer, messages + retry_suffix

    return answer, all_messages


# ================================================================
# TOP-LEVEL LANGGRAPH ORCHESTRATION
# ================================================================

class WorkflowState(TypedDict, total=False):
    """State owned by the parent graph for one complete user turn."""

    question: str
    conversation_history: list
    verbose: bool
    model: str | None
    api_base: str | None
    api_key: str | None
    api_email: str | None
    api_password: str | None
    target: dict
    domain: Literal["rge", "proton"]
    answer: str
    result_history: list
    audit: dict


def _prepare_turn_node(state: WorkflowState) -> WorkflowState:
    history = state.get("conversation_history") or []
    target = _resolve_model_target(
        state.get("model"),
        api_base=state.get("api_base"),
        api_key=state.get("api_key"),
        api_email=state.get("api_email"),
        api_password=state.get("api_password"),
    )
    return {"conversation_history": history, "target": target}


def _route_turn_node(state: WorkflowState) -> WorkflowState:
    return {
        "domain": _route_specialist(
            state["question"], state["conversation_history"], state["target"]
        )
    }


def _selected_domain(state: WorkflowState) -> Literal["rge", "proton"]:
    return state["domain"]


def _run_specialist_workflow(
    state: WorkflowState, domain: Literal["rge", "proton"]
) -> WorkflowState:
    answer, history = _run_routed_turn(
        state["question"],
        conversation_history=state["conversation_history"],
        verbose=state.get("verbose", True),
        model=state.get("model"),
        api_base=state.get("api_base"),
        api_key=state.get("api_key"),
        api_email=state.get("api_email"),
        api_password=state.get("api_password"),
        _target=state["target"],
        _domain=domain,
    )
    return {"answer": answer, "result_history": history}


def _rge_workflow_node(state: WorkflowState) -> WorkflowState:
    return _run_specialist_workflow(state, "rge")


def _proton_workflow_node(state: WorkflowState) -> WorkflowState:
    return _run_specialist_workflow(state, "proton")


def _audit_turn_node(state: WorkflowState) -> WorkflowState:
    """Record machine-readable provenance before the response can reach END."""
    history = state.get("result_history", [])
    tool_names = sorted(_tool_names_in_messages(history))
    artifact_ids: list[str] = []
    verified_plots: list[str] = []
    for message in history:
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload = json.loads(message.content)
        except (TypeError, json.JSONDecodeError):
            continue
        artifact_id = payload.get("artifact_id")
        if artifact_id and artifact_id not in artifact_ids:
            artifact_ids.append(artifact_id)
        verification = payload.get("verification") or {}
        plot_path = verification.get("plot_filepath")
        if plot_path and plot_path not in verified_plots:
            verified_plots.append(plot_path)
    return {
        "audit": {
            "domain": state["domain"],
            "tool_names": tool_names,
            "artifact_ids": artifact_ids,
            "verified_plot_paths": verified_plots,
        }
    }


def _finalize_turn_node(state: WorkflowState) -> WorkflowState:
    answer = state.get("answer")
    history = state.get("result_history")
    if not isinstance(answer, str) or not answer.strip():
        answer = "[No response generated]"
    if not isinstance(history, list):
        history = list(state.get("conversation_history") or [])
        history.append(AIMessage(content=answer))
    clean_answer = _code_free_answer(answer)
    if clean_answer != answer:
        start = max((i for i, message in enumerate(history) if isinstance(message, HumanMessage)), default=-1) + 1
        plot = _latest_verified_plot_payload(history, start)
        if plot is not None:
            clean_answer = _repair_answer_plot_filepath(clean_answer, plot)
        if history and isinstance(history[-1], AIMessage) and history[-1].content == answer and not history[-1].tool_calls:
            history = history[:-1] + [history[-1].model_copy(update={"content": clean_answer})]
        else:
            history = history + [AIMessage(content=clean_answer)]
    answer = clean_answer
    return {"answer": answer, "result_history": history}


def _looks_like_program(text: str) -> bool:
    """Recognize source snippets while preserving formulas, paths and column names."""
    text = text.strip()
    if re.match(r"(?:from\s+\S+\s+import\b|import\s+\w|(?:async\s+)?def\s+|class\s+|"
                r"(?:for|while|with|if|elif|except)\s+.+:|try\s*:|return\s+|#\!|%matplotlib)", text):
        return True
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return bool(re.search(r"\b(?:plt|pd|np|sns|h5py|numpy|matplotlib)\.[\w.]+\s*\(", text))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.For, ast.While, ast.With, ast.Try, ast.Lambda, ast.Call)):
            return True
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            value = node.value
            if not isinstance(value, (ast.Constant, ast.UnaryOp)) or (
                isinstance(value, ast.Constant) and isinstance(value.value, str)
            ):
                return True
    return False


def _code_free_answer(answer: str) -> str:
    """Last-mile output policy: generated programs never become chat responses."""
    text = str(answer or "")
    removed = False

    def fence(match):
        nonlocal removed
        language = match.group("language").strip().lower().split(" ", 1)[0]
        body = match.group("body")
        if language not in {"", "text", "plaintext", "json", "csv"} or _looks_like_program(body):
            removed = True
            return ""
        return match.group(0)

    text = re.sub(r"(?P<marker>`{3,}|~{3,})(?P<language>[^\n]*)\n(?P<body>.*?)"
                  r"(?:(?P=marker)|\Z)", fence, text, flags=re.S)
    text, count = re.subn(r"(?P<marker>`{3,}|~{3,})\s*(?:python|py|ipython|bash|shell|javascript|js)\b"
                         r"[^\n]*?(?P=marker)", "", text, flags=re.I)
    removed |= bool(count)
    text, count = re.subn(r"<(?P<tag>tool_code|python|pre|script)\b[^>]*>.*?(?:</(?P=tag)>|\Z)",
                         "", text, flags=re.S | re.I)
    removed |= bool(count)

    def html_code(match):
        nonlocal removed
        import html
        body = html.unescape(match.group("body"))
        if _looks_like_program(body) or re.search(r"language-(?:python|py|javascript|bash)", match.group("attrs"), re.I):
            removed = True
            return ""
        return match.group(0)

    text = re.sub(r"<code(?P<attrs>\b[^>]*)>(?P<body>.*?)</code>", html_code, text, flags=re.S | re.I)

    def inline(match):
        nonlocal removed
        if _looks_like_program(match.group(1)):
            removed = True
            return ""
        return match.group(0)

    text = re.sub(r"(?<!`)`([^`\n]+)`(?!`)", inline, text)
    lines = []
    in_program = False
    in_data_fence = False
    for line in text.splitlines():
        if re.match(r"\s*(?:`{3,}|~{3,})", line):
            in_data_fence = not in_data_fence
            lines.append(line)
            continue
        if not in_data_fence and _looks_like_program(line):
            removed = in_program = True
            continue
        if in_program and (not line.strip() or line.startswith((" ", "\t")) or line.lstrip().startswith("#")):
            continue
        in_program = False
        lines.append(line)
    if not removed:
        return str(answer or "")
    text = "\n".join(lines)
    paragraphs = re.split(r"\n\s*\n", text)
    text = "\n\n".join(paragraph for paragraph in paragraphs if not re.search(
        r"\b(?:python snippet|python code|minimal python|code below|following code|run this code|how to reproduce)\b",
        paragraph, re.I))
    text = text.strip()
    return text or "No verified result was available to report."


def build_workflow_graph():
    """Build the parent graph controlling routing through final validation."""
    graph = StateGraph(WorkflowState)
    graph.add_node("prepare_turn", _prepare_turn_node)
    graph.add_node("route_specialist", _route_turn_node)
    graph.add_node("rge_workflow", _rge_workflow_node)
    graph.add_node("proton_workflow", _proton_workflow_node)
    graph.add_node("audit_artifacts", _audit_turn_node)
    graph.add_node("finalize_answer", _finalize_turn_node)
    graph.set_entry_point("prepare_turn")
    graph.add_edge("prepare_turn", "route_specialist")
    graph.add_conditional_edges(
        "route_specialist",
        _selected_domain,
        {"rge": "rge_workflow", "proton": "proton_workflow"},
    )
    graph.add_edge("rge_workflow", "audit_artifacts")
    graph.add_edge("proton_workflow", "audit_artifacts")
    graph.add_edge("audit_artifacts", "finalize_answer")
    graph.add_edge("finalize_answer", END)
    return graph.compile()


_workflow_graph = None


def ask(question: str, conversation_history: list = None,
        verbose: bool = True, model: str | None = None,
        api_base: str | None = None, api_key: str | None = None,
        api_email: str | None = None, api_password: str | None = None) -> tuple:
    """Run one complete request through the parent LangGraph."""
    global _workflow_graph
    if _workflow_graph is None:
        _workflow_graph = build_workflow_graph()
    try:
        state = _workflow_graph.invoke(
            {
                "question": question,
                "conversation_history": conversation_history or [],
                "verbose": verbose,
                "model": model,
                "api_base": api_base,
                "api_key": api_key,
                "api_email": api_email,
                "api_password": api_password,
            },
            {"recursion_limit": 10},
        )
    except GraphRecursionError as exc:
        raise RuntimeError(
            "The top-level workflow exceeded its graph transition limit."
        ) from exc
    return state["answer"], state["result_history"]


def direct_chat(
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    api_email: str | None = None,
    api_password: str | None = None,
):
    """Run an open-ended chat in which the LLM chooses any needed tools."""
    print("\nProton Decay AI Agent")
    print("=" * 50)
    print(f"Model: {model or 'auto'}")
    print("Ask a question directly. Commands: 'quit' and 'reset'.\n")

    history = []

    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit"}:
            break

        if question.lower() == "reset":
            history = []
            _reset_response_timer()
            print("[Conversation reset]\n")
            continue

        _reset_response_timer()

        try:
            with _activity_indicator():
                answer, history = ask(
                    question,
                    conversation_history=history,
                    model=model,
                    api_base=api_base,
                    api_key=api_key,
                    api_email=api_email,
                    api_password=api_password,
                )
            print(f"\nAgent: {answer}\n")
        except RuntimeError as error:
            print(f"\n[ERROR] {error}\n")


# ================================================================
# ENTRY POINT
# ================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proton_agent",
        description="Interactive proton-decay and RGE agent.",
    )
    parser.add_argument(
        "-m",
        "--model",
        "-model",
        default=None,
        help=(
            "Model selector. Bare names use local Ollama, e.g. gpt-oss:20b or qwen3:8b. "
            "Use ollama:<model>, remote:<model>/openwebui:<model>, openai:<model>, "
            "or anthropic:<model>."
        ),
    )
    parser.add_argument("--api-base", help="OpenAI-compatible/OpenWebUI/Anthropic base URL override.")
    parser.add_argument("--api-key", help="API key/token for the selected remote provider.")
    parser.add_argument("--api-email", help="Open WebUI sign-in email, if not using an API token.")
    parser.add_argument("--api-password", help="Open WebUI sign-in password, if not using an API token.")
    parser.add_argument("question", nargs="*", help="Optional one-shot question. Omit for interactive chat.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    question = " ".join(args.question).strip()

    try:
        if question:
            _reset_response_timer()
            with _activity_indicator():
                answer, _ = ask(
                    question,
                    model=args.model,
                    api_base=args.api_base,
                    api_key=args.api_key,
                    api_email=args.api_email,
                    api_password=args.api_password,
                )
            print(f"\n{answer}")
        else:
            direct_chat(
                model=args.model,
                api_base=args.api_base,
                api_key=args.api_key,
                api_email=args.api_email,
                api_password=args.api_password,
            )
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
