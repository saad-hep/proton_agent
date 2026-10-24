from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable

import numpy as np
from scipy.integrate import solve_ivp

from rge_coefficients import (
    SM_ONE_LOOP,
    SM_TWO_LOOP,
    SM_YUKAWA_TRACE_COEFFICIENTS,
    SMField,
    field_one_loop_delta,
    field_two_loop_delta,
)


DEFAULT_MZ = 91.1876
DEFAULT_MZ_COUPLINGS = {
    "g1": 0.461425,
    "g2": 0.65184,
    "g3": 1.2143,
}
DEFAULT_MZ_YUKAWAS = {
    "yt": 0.9861,
    "yb": 0.01639,
    "ytau": 0.010029,
}


def _parse_fields(fields: Iterable[SMField | dict[str, Any]]) -> list[SMField]:
    return [field if isinstance(field, SMField) else SMField.from_dict(field) for field in fields]


def _active_threshold(theta_scale: float, threshold_scale: float) -> float:
    return 1.0 if theta_scale >= threshold_scale else 0.0


def active_coefficients(
    log_mu: float,
    fields: Iterable[SMField | dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([float(x) for x in SM_ONE_LOOP], dtype=float)
    b = np.array([[float(x) for x in row] for row in SM_TWO_LOOP], dtype=float)
    mu = float(np.exp(log_mu))

    for field in _parse_fields(fields):
        if field.mass is None:
            active = 1.0
        else:
            active = _active_threshold(mu, float(field.mass))
        if not active:
            continue
        a += np.array([float(x) for x in field_one_loop_delta(field)], dtype=float)
        b += np.array([[float(x) for x in row] for row in field_two_loop_delta(field)], dtype=float)

    return a, b


def gauge_beta(
    log_mu: float,
    couplings: np.ndarray,
    fields: Iterable[SMField | dict[str, Any]] = (),
    loops: int = 1,
    yukawa_traces: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    g = np.asarray(couplings, dtype=float)
    a, b = active_coefficients(log_mu, fields)
    beta = (g**3) * a / (16.0 * np.pi**2)

    if loops >= 2:
        c = np.array([[float(Fraction(x)) for x in row] for row in SM_YUKAWA_TRACE_COEFFICIENTS], dtype=float)
        traces = np.asarray(yukawa_traces, dtype=float)
        gauge_terms = b @ (g**2)
        yukawa_terms = c @ traces
        beta += (g**3) * (gauge_terms - yukawa_terms) / ((16.0 * np.pi**2) ** 2)

    return beta


def third_generation_yukawa_beta(couplings: np.ndarray, yukawas: np.ndarray) -> np.ndarray:
    """Notebook one-loop beta functions for diagonal yt, yb, ytau."""
    g1, g2, g3 = np.asarray(couplings, dtype=float)
    yt, yb, ytau = np.asarray(yukawas, dtype=float)
    loop = 16.0 * np.pi**2

    yt_beta = yt * (
        -17.0 / 20.0 * g1**2
        - 9.0 / 4.0 * g2**2
        - 8.0 * g3**2
        + 9.0 / 2.0 * yt**2
        + 3.0 / 2.0 * yb**2
        + ytau**2
    ) / loop
    yb_beta = yb * (
        -1.0 / 4.0 * g1**2
        - 9.0 / 4.0 * g2**2
        - 8.0 * g3**2
        + 3.0 / 2.0 * yt**2
        + 9.0 / 2.0 * yb**2
        + ytau**2
    ) / loop
    ytau_beta = ytau * (
        -9.0 / 4.0 * g1**2
        - 9.0 / 4.0 * g2**2
        + 3.0 * yt**2
        + 3.0 * yb**2
        + 5.0 / 2.0 * ytau**2
    ) / loop
    return np.array([yt_beta, yb_beta, ytau_beta], dtype=float)


def _initial_coupling_array(initial_couplings: dict[str, float] | Iterable[float] | None) -> np.ndarray:
    if initial_couplings is None:
        return np.array(
            [DEFAULT_MZ_COUPLINGS["g1"], DEFAULT_MZ_COUPLINGS["g2"], DEFAULT_MZ_COUPLINGS["g3"]],
            dtype=float,
        )
    if isinstance(initial_couplings, dict):
        return np.array([initial_couplings["g1"], initial_couplings["g2"], initial_couplings["g3"]], dtype=float)
    return np.array(list(initial_couplings), dtype=float)


def _initial_yukawa_array(initial_yukawas: dict[str, float] | Iterable[float] | None) -> np.ndarray | None:
    if initial_yukawas is None:
        return None
    if isinstance(initial_yukawas, dict):
        return np.array([initial_yukawas["yt"], initial_yukawas["yb"], initial_yukawas["ytau"]], dtype=float)
    return np.array(list(initial_yukawas), dtype=float)


def run_gauge_couplings(
    fields: Iterable[SMField | dict[str, Any]] = (),
    gut_scale: float = 1.0e16,
    initial_scale: float = DEFAULT_MZ,
    initial_couplings: dict[str, float] | Iterable[float] | None = None,
    loops: int = 1,
    n_points: int = 250,
    yukawa_traces: tuple[float, float, float] = (0.0, 0.0, 0.0),
    initial_yukawas: dict[str, float] | Iterable[float] | None = None,
) -> dict[str, Any]:
    if gut_scale <= 0 or initial_scale <= 0:
        raise ValueError("Scales must be positive.")
    if gut_scale == initial_scale:
        raise ValueError("gut_scale and initial_scale must be different.")
    if loops not in {1, 2}:
        raise ValueError("loops must be 1 or 2.")

    parsed_fields = _parse_fields(fields)
    g0 = _initial_coupling_array(initial_couplings)
    if g0.shape != (3,):
        raise ValueError("initial_couplings must provide g1, g2, g3.")

    y0 = _initial_yukawa_array(initial_yukawas)
    run_yukawas = y0 is not None
    if run_yukawas and y0.shape != (3,):
        raise ValueError("initial_yukawas must provide yt, yb, ytau.")

    t0 = float(np.log(initial_scale))
    t1 = float(np.log(gut_scale))
    t_eval = np.linspace(t0, t1, int(max(2, n_points)))

    if run_yukawas:
        state0 = np.concatenate([g0, y0])

        def rhs(t: float, state: np.ndarray) -> np.ndarray:
            g = state[:3]
            y = state[3:6]
            traces = (float(y[0] ** 2), float(y[1] ** 2), float(y[2] ** 2))
            dg = gauge_beta(t, g, parsed_fields, loops=loops, yukawa_traces=traces)
            dy = third_generation_yukawa_beta(g, y)
            return np.concatenate([dg, dy])
    else:
        state0 = g0

        def rhs(t: float, state: np.ndarray) -> np.ndarray:
            return gauge_beta(t, state, parsed_fields, loops=loops, yukawa_traces=yukawa_traces)

    sol = solve_ivp(rhs, (t0, t1), state0, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(sol.message)

    scales = np.exp(sol.t)
    state_values = sol.y.T
    values = state_values[:, :3]
    final = values[-1]
    inverse_alpha = 4.0 * np.pi / (values**2)

    samples = {
        "scale_GeV": scales.tolist(),
        "g1": values[:, 0].tolist(),
        "g2": values[:, 1].tolist(),
        "g3": values[:, 2].tolist(),
        "alpha1_inv": inverse_alpha[:, 0].tolist(),
        "alpha2_inv": inverse_alpha[:, 1].tolist(),
        "alpha3_inv": inverse_alpha[:, 2].tolist(),
    }
    output = {
        "gauge_order": ["g1", "g2", "g3"],
        "scale_variable": "natural log of mu/GeV",
        "loops": loops,
        "yukawa_running": run_yukawas,
        "initial_scale_GeV": float(initial_scale),
        "gut_scale_GeV": float(gut_scale),
        "initial_couplings": {"g1": float(g0[0]), "g2": float(g0[1]), "g3": float(g0[2])},
        "final_couplings": {"g1": float(final[0]), "g2": float(final[1]), "g3": float(final[2])},
        "final_inverse_alpha": {
            "alpha1_inv": float(4.0 * np.pi / final[0] ** 2),
            "alpha2_inv": float(4.0 * np.pi / final[1] ** 2),
            "alpha3_inv": float(4.0 * np.pi / final[2] ** 2),
        },
        "thresholds": [
            {
                "name": field.name,
                "kind": field.kind,
                "su3": field.su3,
                "su2": field.su2,
                "hypercharge": str(field.hypercharge),
                "real_scalar": field.real_scalar,
                "fermion_type": field.fermion_type,
                "generations": field.generations,
                "mass_GeV": field.mass,
                "active_at_initial": field.mass is None or initial_scale >= field.mass,
                "active_at_gut": field.mass is None or gut_scale >= field.mass,
            }
            for field in parsed_fields
        ],
        "samples": samples,
    }

    if run_yukawas:
        yukawa_values = state_values[:, 3:6]
        final_y = yukawa_values[-1]
        output["initial_yukawas"] = {"yt": float(y0[0]), "yb": float(y0[1]), "ytau": float(y0[2])}
        output["final_yukawas"] = {"yt": float(final_y[0]), "yb": float(final_y[1]), "ytau": float(final_y[2])}
        output["samples"].update(
            {
                "yt": yukawa_values[:, 0].tolist(),
                "yb": yukawa_values[:, 1].tolist(),
                "ytau": yukawa_values[:, 2].tolist(),
            }
        )
    else:
        output["fixed_yukawa_traces"] = {
            "TrYu": float(yukawa_traces[0]),
            "TrYd": float(yukawa_traces[1]),
            "TrYe": float(yukawa_traces[2]),
        }

    return output
