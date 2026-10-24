import numpy as np
import builtins
import time
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from pathlib import Path
from runtime_paths import project_root

SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = project_root()
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
from nonsusy_su5_formula import compute_decay_rates as compute_nonsusy_su5_decay_rates
from nonsusy_so10_formula import compute_decay_rates as compute_nonsusy_so10_decay_rates
from parameters import generate_parameters, generate_parameters_with_ckm_pmns
from hadronic_inputs import current_matrix_elements, call_with_matrix_elements
from bounds import classify_model, current_bounds_configuration, call_with_bounds

_AGENT_START_TIME = time.monotonic()


def _timed_print(*args, **kwargs):
    elapsed = time.monotonic() - _AGENT_START_TIME
    builtins.print(f"[{elapsed:.1f} sec]", *args, **kwargs)


print = _timed_print


MATRIX_KEYS = ["UL", "UR", "DL", "DR", "EL", "ER", "NL"]
SCAN_SCHEMA_VERSION = 5


def _generate_parameter_points(n_points, user_config=None):
    if user_config and user_config.get("derive_dl_nl_from_ckm_pmns"):
        return generate_parameters_with_ckm_pmns(n_points, user_config=user_config)
    return generate_parameters(n_points, user_config=user_config)

def evaluate_point(params):
    model = str(params.get("nonsusy_model", "")).upper().replace("(", "").replace(")", "")
    if model == "SU5":
        result = compute_nonsusy_su5_decay_rates(
            MGUT=params["MGUT"],
            gGUT=params["gGUT"],
            U=params["UL"],
            Uc=np.conjugate(params["UR"]),
            D=params["DL"],
            Dc=np.conjugate(params["DR"]),
            E=params["EL"],
            Ec=np.conjugate(params["ER"]),
            N=params["NL"],
        )
    elif model == "SO10":
        result = compute_nonsusy_so10_decay_rates(
            MGUT=params["MGUT"],
            gGUT=params["gGUT"],
            **{key: params[key] for key in MATRIX_KEYS},
        )
    else:
        raise ValueError("Each scan point must specify nonsusy_model='SU5' or 'SO10'.")

    return {
        "MGUT (GeV)": params["MGUT"],
        "gGUT": params["gGUT"],
        "nonsusy_model": model,
        "input_scalars": {
            key: value
            for key, value in params.items()
            if key not in MATRIX_KEYS
            and key not in {"matrix_parameters", "derived_matrices", "nonsusy_model"}
            and np.isscalar(value)
        },
        "matrices": {key: params[key] for key in MATRIX_KEYS},
        "matrix_parameters": params.get("matrix_parameters", {}),
        "derived_matrices": params.get("derived_matrices", {}),
        "widths_GeV": result["widths"],
        "total_width_GeV": result["total_width_GeV"],
        "lifetimes (yrs)": result["lifetimes_years"],
        "branching_ratios": result["branching_ratios"],
        "wilson_coefficients": result.get("wilson_coefficients", {}),
        "neutrino_coefficients": result.get("neutrino_coefficients", {}),
        "classification": classify_model(result["lifetimes_years"]),
    }

def evaluate_chunk(chunk):
    return [evaluate_point(p) for p in chunk]

def evaluate_points_parallel(points, n_jobs=-1, n_chunks=200):
    if not points:
        return []

    n_chunks = min(n_chunks, len(points))
    chunks = np.array_split(points, n_chunks)

    print(f"[INFO] Running {len(chunks)} chunks on {n_jobs} cores...")

    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(
            n_jobs=n_jobs,
            batch_size=1,
            verbose=10
        )(
            delayed(call_with_bounds)(current_bounds_configuration(), call_with_matrix_elements,
                                      current_matrix_elements(), evaluate_chunk, chunk) for chunk in chunks
        )

    results = [item for sublist in results for item in sublist]
    return results

def _channel_to_column(channel):
    return (
        channel.replace("->", "_to_")
        .replace("+", "plus")
        .replace(" ", "_")
    )

def _flatten_matrix_entries(row, matrix_name, matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            entry = matrix[i, j]
            base = f"{matrix_name}_{i+1}{j+1}"
            row[f"{base}_real"] = float(np.real(entry))
            row[f"{base}_imag"] = float(np.imag(entry))
            row[f"{base}_abs"] = float(np.abs(entry))


def _flatten_complex_value(row, base, value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        scalar = arr.item()
        row[f"{base}_real"] = float(np.real(scalar))
        row[f"{base}_imag"] = float(np.imag(scalar))
        row[f"{base}_abs"] = float(np.abs(scalar))
        return

    for index in np.ndindex(arr.shape):
        suffix = "_".join(str(i) for i in index)
        indexed_base = f"{base}_{suffix}"
        scalar = arr[index]
        row[f"{indexed_base}_real"] = float(np.real(scalar))
        row[f"{indexed_base}_imag"] = float(np.imag(scalar))
        row[f"{indexed_base}_abs"] = float(np.abs(scalar))

def results_to_dataframe(results):
    rows = []
    for result in results:
        row = {
            "scan_schema_version": SCAN_SCHEMA_VERSION,
            "MGUT": result["MGUT (GeV)"],
            "gGUT": result["gGUT"],
            "nonsusy_model": result["nonsusy_model"],
            "classification": result["classification"],
            "total_width_GeV": result["total_width_GeV"],
        }
        for key, value in result.get("input_scalars", {}).items():
            if np.isscalar(value):
                row[str(key)] = float(value)
        for matrix_name, matrix in result.get("matrices", {}).items():
            _flatten_matrix_entries(row, matrix_name, matrix)
        for matrix_name, matrix in result.get("derived_matrices", {}).items():
            _flatten_matrix_entries(row, matrix_name, matrix)
        for matrix_name, sampled in result.get("matrix_parameters", {}).items():
            for param_name, value in sampled.items():
                row[f"{matrix_name}_{param_name}"] = float(value)
        for channel, value in result["widths_GeV"].items():
            row[f"width_{_channel_to_column(channel)}"] = value
        for channel, value in result["lifetimes (yrs)"].items():
            row[f"lifetime_{_channel_to_column(channel)}"] = value
        for channel, value in result["branching_ratios"].items():
            row[f"branching_{_channel_to_column(channel)}"] = value
        for key, value in result.get("wilson_coefficients", {}).items():
            _flatten_complex_value(row, f"wilson_{key}", value)
        for key, value in result.get("neutrino_coefficients", {}).items():
            _flatten_complex_value(row, f"neutrino_{key}", value)
        rows.append(row)
    return pd.DataFrame(rows)

def save_scan_results(results, filepath):
    filepath = Path(filepath)
    hidden_parts = [
        part for part in filepath.parts
        if part.startswith(".") and part not in {".", ".."}
    ]
    if hidden_parts:
        raise ValueError(
            "Hidden output paths are not allowed. Rename these path components "
            f"without a leading dot: {', '.join(hidden_parts)}"
        )
    if not filepath.is_absolute():
        filepath = OUTPUT_DIR / filepath
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df = results_to_dataframe(results)
    df.to_hdf(filepath, key="results", mode="w")
    return str(filepath)

def run_scan(
    n_points=1000,
    n_jobs=-1,
    n_chunks=200,
    user_config=None,
    save_path=OUTPUT_DIR / "scan_results" / "latest_scan.h5",
):
    print(f"[INFO] Generating {n_points} parameter points...")
    points = list(_generate_parameter_points(n_points, user_config=user_config))
    model = str((user_config or {}).get("nonsusy_model", ""))
    if not model:
        raise ValueError("run_scan requires user_config['nonsusy_model'] to be 'SU5' or 'SO10'.")
    for point in points:
        point["nonsusy_model"] = model

    results = evaluate_points_parallel(points, n_jobs=n_jobs, n_chunks=n_chunks)

    print(f"[INFO] Completed scan: {len(results)} points")

    hdf5_file = save_scan_results(results, save_path)
    print(f"[INFO] Saved scan to: {hdf5_file}")

    return {
        "results": results,
        "hdf5_file": hdf5_file,
    }
