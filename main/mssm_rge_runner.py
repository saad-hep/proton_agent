from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable

import numpy as np
from scipy.integrate import solve_ivp

from mssm_coefficients import MSSMField, coefficient_summary
from rge_runner import DEFAULT_MZ_COUPLINGS


DEFAULT_MSSM_INITIAL_SCALE = 1.0e3
DEFAULT_MSSM_COUPLINGS = DEFAULT_MZ_COUPLINGS.copy()
MSBAR_TO_DRBAR_ADJOINT_CASIMIRS = np.array([0.0, 2.0, 3.0], dtype=float)


def _parse_fields(fields: Iterable[MSSMField | dict[str, Any]]) -> list[MSSMField]:
    return [field if isinstance(field, MSSMField) else MSSMField.from_dict(field) for field in fields]


def _active_fields(log_mu: float, fields: Iterable[MSSMField]) -> list[MSSMField]:
    mu = float(np.exp(log_mu))
    return [field for field in fields if field.mass is None or mu >= float(field.mass)]


def active_coefficients(log_mu: float, fields: Iterable[MSSMField | dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    active = _active_fields(log_mu, _parse_fields(fields))
    summary = coefficient_summary(active)
    a = np.array([float(Fraction(x)) for x in summary["total_one_loop"]], dtype=float)
    b = np.array([[float(Fraction(x)) for x in row] for row in summary["total_two_loop"]], dtype=float)
    return a, b


def gauge_beta(log_mu: float, couplings: np.ndarray, fields: Iterable[MSSMField | dict[str, Any]] = (), loops: int = 1) -> np.ndarray:
    g = np.asarray(couplings, dtype=float)
    a, b = active_coefficients(log_mu, fields)
    beta = (g**3) * a / (16.0 * np.pi**2)
    if loops >= 2:
        beta += (g**3) * (b @ (g**2)) / ((16.0 * np.pi**2) ** 2)
    return beta


def _initial_coupling_array(initial_couplings: dict[str, float] | Iterable[float] | None) -> np.ndarray:
    if initial_couplings is None:
        initial_couplings = DEFAULT_MSSM_COUPLINGS
    if isinstance(initial_couplings, dict):
        return np.array([initial_couplings["g1"], initial_couplings["g2"], initial_couplings["g3"]], dtype=float)
    return np.array(list(initial_couplings), dtype=float)


def msbar_to_drbar_gauge_couplings(
    couplings: dict[str, float] | Iterable[float],
) -> dict[str, float]:
    """Apply the one-loop MS-bar to DR-bar gauge-coupling conversion.

    This implements Eq. (2.4) of Martin and Vaughn, hep-ph/9308222,
    equivalently alpha_DR^-1 = alpha_MS^-1 - C2(G)/(12*pi), separately
    for G = U(1)_Y, SU(2)_L, SU(3)_C with C2(G) = (0, 2, 3).
    """
    g_msbar = _initial_coupling_array(couplings)
    if g_msbar.shape != (3,) or not np.all(np.isfinite(g_msbar)) or np.any(g_msbar <= 0.0):
        raise ValueError("MS-bar couplings must be three positive finite values: g1, g2, g3.")

    inverse_alpha_msbar = 4.0 * np.pi / (g_msbar**2)
    inverse_alpha_drbar = inverse_alpha_msbar - MSBAR_TO_DRBAR_ADJOINT_CASIMIRS / (12.0 * np.pi)
    if np.any(inverse_alpha_drbar <= 0.0):
        raise ValueError("MS-bar to DR-bar conversion produced a non-positive inverse coupling.")
    g_drbar = np.sqrt(4.0 * np.pi / inverse_alpha_drbar)
    return {"g1": float(g_drbar[0]), "g2": float(g_drbar[1]), "g3": float(g_drbar[2])}


def run_mssm_gauge_couplings(
    fields: Iterable[MSSMField | dict[str, Any]] = (),
    gut_scale: float = 1.0e16,
    initial_scale: float = DEFAULT_MSSM_INITIAL_SCALE,
    initial_couplings: dict[str, float] | Iterable[float] | None = None,
    loops: int = 1,
    n_points: int = 250,
) -> dict[str, Any]:
    if gut_scale <= 0 or initial_scale <= 0:
        raise ValueError("Scales must be positive.")
    if gut_scale == initial_scale:
        raise ValueError("gut_scale and initial_scale must be different.")
    if loops not in {1, 2}:
        raise ValueError("loops must be 1 or 2.")

    parsed_fields = _parse_fields(fields)
    g0_msbar = _initial_coupling_array(initial_couplings)
    if g0_msbar.shape != (3,):
        raise ValueError("initial_couplings must provide g1, g2, g3.")
    input_couplings_msbar = {
        "g1": float(g0_msbar[0]), "g2": float(g0_msbar[1]), "g3": float(g0_msbar[2])
    }
    initial_couplings_drbar = msbar_to_drbar_gauge_couplings(input_couplings_msbar)
    g0 = _initial_coupling_array(initial_couplings_drbar)

    t0 = float(np.log(initial_scale))
    t1 = float(np.log(gut_scale))
    t_eval = np.linspace(t0, t1, int(max(2, n_points)))

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        return gauge_beta(t, state, parsed_fields, loops=loops)

    sol = solve_ivp(rhs, (t0, t1), g0, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(sol.message)

    scales = np.exp(sol.t)
    values = sol.y.T
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
    return {
        "model": "MSSM",
        "runner": "MSSM gauge-only RGE",
        "gauge_order": ["U1Y", "SU2L", "SU3C"],
        "loops": loops,
        "scale_variable": "natural log of mu/GeV",
        "renormalization_scheme": "DRbar",
        "initial_scale_GeV": float(initial_scale),
        "gut_scale_GeV": float(gut_scale),
        "input_couplings_msbar": input_couplings_msbar,
        "initial_couplings": {"g1": float(g0[0]), "g2": float(g0[1]), "g3": float(g0[2])},
        "scheme_conversion": {
            "name": "MSbar-to-DRbar",
            "scale_GeV": float(initial_scale),
            "loop_order": 1,
            "formula": "alpha_DRbar^-1 = alpha_MSbar^-1 - C2(G)/(12*pi)",
            "adjoint_casimirs_U1Y_SU2L_SU3C": [0, 2, 3],
            "reference": "Martin and Vaughn, hep-ph/9308222, Eq. (2.4)",
        },
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
