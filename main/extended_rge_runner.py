from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

import numpy as np
from scipy.integrate import solve_ivp

from extended_rge_coefficients import ExtendedField, GaugeModel, MODELS, coefficient_summary
from rge_coefficients import SMField, coefficients_as_float
from rge_runner import DEFAULT_MZ, DEFAULT_MZ_COUPLINGS


SM_LABELS = ["U1Y", "SU2L", "SU3C"]
GROUP_ALIASES = {
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
    "421": "421",
    "g214": "421",
    "3211": "3211",
    "tto": "3211",
}
LABEL_ALIASES = {
    "g1": "U1Y",
    "u1y": "U1Y",
    "hypercharge": "U1Y",
    "y": "U1Y",
    "g2": "SU2L",
    "su2l": "SU2L",
    "2l": "SU2L",
    "g3": "SU3C",
    "su3c": "SU3C",
    "3c": "SU3C",
    "su2r": "SU2R",
    "2r": "SU2R",
    "r": "U1R",
    "u1r": "U1R",
    "bl": "U1BL",
    "u1bl": "U1BL",
    "b-l": "U1BL",
    "bminusl": "U1BL",
    "su4c": "SU4C",
    "4c": "SU4C",
}


@dataclass(frozen=True)
class StageSpec:
    group: str
    end: float
    fields: tuple[dict[str, Any], ...] = ()
    include_defaults: bool = True
    matching: dict[str, float] | None = None
    points: int | None = None


def canonical_group(group: str) -> str:
    key = str(group).strip().lower().replace(" ", "_")
    if key not in GROUP_ALIASES:
        raise ValueError(f"Unknown group {group!r}. Use sm, 3221, 422, 421, or 3211.")
    return GROUP_ALIASES[key]


def group_labels(group: str) -> list[str]:
    canonical = canonical_group(group)
    if canonical == "sm":
        return list(SM_LABELS)
    return [factor.label for factor in MODELS[canonical].factors]


def _label_key(label: str) -> str:
    key = str(label).strip().lower().replace("_", "").replace(" ", "")
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    upper = str(label).strip().upper()
    if upper in {"U1Y", "SU2L", "SU3C", "SU2R", "U1R", "U1BL", "SU4C"}:
        return upper
    raise ValueError(f"Unknown coupling label {label!r}")


def _normalize_coupling_dict(couplings: dict[str, float]) -> dict[str, float]:
    return {_label_key(key): float(value) for key, value in couplings.items()}


def _alpha_inv(g: float) -> float:
    return 4.0 * np.pi / float(g) ** 2


def _g_from_alpha_inv(alpha_inv: float) -> float:
    if alpha_inv <= 0:
        raise ValueError(f"Cannot construct a real coupling from alpha^-1={alpha_inv}.")
    return float(np.sqrt(4.0 * np.pi / alpha_inv))


def _array_to_couplings(labels: list[str], values: Iterable[float]) -> dict[str, float]:
    return {label: float(value) for label, value in zip(labels, values)}


def _couplings_to_array(labels: list[str], couplings: dict[str, float]) -> np.ndarray:
    missing = [label for label in labels if label not in couplings]
    if missing:
        raise ValueError(f"Missing couplings for labels: {missing}")
    return np.array([couplings[label] for label in labels], dtype=float)


def _stage_fields(stage: StageSpec, model: GaugeModel | None) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    if model is not None and stage.include_defaults:
        fields.extend(dict(field) for field in model.default_fields)
    fields.extend(dict(field) for field in stage.fields)
    return fields


def _field_mass(field: SMField | ExtendedField) -> float | None:
    return None if field.mass is None else float(field.mass)


def _active_sm_fields(log_mu: float, fields: Iterable[dict[str, Any] | SMField]) -> list[SMField]:
    mu = float(np.exp(log_mu))
    parsed = [field if isinstance(field, SMField) else SMField.from_dict(field) for field in fields]
    return [field for field in parsed if _field_mass(field) is None or mu >= float(_field_mass(field))]


def _active_extended_fields(log_mu: float, model: GaugeModel, fields: Iterable[dict[str, Any] | ExtendedField]) -> list[ExtendedField]:
    mu = float(np.exp(log_mu))
    parsed = [field if isinstance(field, ExtendedField) else ExtendedField.from_dict(field, model) for field in fields]
    return [field for field in parsed if _field_mass(field) is None or mu >= float(_field_mass(field))]


def active_stage_coefficients(log_mu: float, group: str, fields: Iterable[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    canonical = canonical_group(group)
    if canonical == "sm":
        return coefficients_as_float(_active_sm_fields(log_mu, fields))

    model = MODELS[canonical]
    active = _active_extended_fields(log_mu, model, fields)
    summary = coefficient_summary(model, active)
    a = np.array([float(Fraction(x)) for x in summary["one_loop"]], dtype=float)
    b = np.array([[float(Fraction(x)) for x in row] for row in summary["two_loop"]], dtype=float)
    return a, b


def gauge_only_beta(log_mu: float, couplings: np.ndarray, group: str, fields: Iterable[dict[str, Any]], loops: int = 1) -> np.ndarray:
    g = np.asarray(couplings, dtype=float)
    a, b = active_stage_coefficients(log_mu, group, fields)
    beta = (g**3) * a / (16.0 * np.pi**2)
    if loops >= 2:
        beta += (g**3) * (b @ (g**2)) / ((16.0 * np.pi**2) ** 2)
    return beta


def match_couplings(
    source_group: str,
    target_group: str,
    source_couplings: dict[str, float],
    overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    """Tree-level matching between supported effective gauge groups.

    The multiple-U(1) case is treated in the diagonal convention with no kinetic
    mixing. Bottom-up matches that split hypercharge require one extra input;
    if it is not supplied, conservative defaults are used: g2R=g2L for 3221/422
    and gR=g1 for 3211/421.
    """
    target = canonical_group(target_group)
    src = _normalize_coupling_dict(source_couplings)
    out: dict[str, float] = {}
    user = _normalize_coupling_dict(overrides or {})
    target_labels = group_labels(target)

    for label in target_labels:
        if label in src:
            out[label] = src[label]

    if "SU3C" in target_labels and "SU3C" not in out and "SU4C" in src:
        out["SU3C"] = src["SU4C"]
    if "SU4C" in target_labels and "SU4C" not in out and "SU3C" in src:
        out["SU4C"] = src["SU3C"]
    if "U1BL" in target_labels and "U1BL" not in out and "SU4C" in src:
        out["U1BL"] = src["SU4C"]

    if "U1R" in target_labels and "U1R" not in out and "SU2R" in src:
        out["U1R"] = src["SU2R"]
    if "SU2R" in target_labels and "SU2R" not in out and "U1R" in src:
        out["SU2R"] = src["U1R"]

    out.update(user)

    if "SU2R" in target_labels and "SU2R" not in out:
        if "SU2L" in src:
            out["SU2R"] = src["SU2L"]
        elif "U1Y" in src:
            out["SU2R"] = src["U1Y"]
    if "U1R" in target_labels and "U1R" not in out:
        if "U1Y" in src:
            out["U1R"] = src["U1Y"]
        elif "SU2R" in out:
            out["U1R"] = out["SU2R"]
        elif "SU2R" in src:
            out["U1R"] = src["SU2R"]

    if "U1BL" in target_labels and "U1BL" not in out and "U1Y" in src:
        r_label = "SU2R" if "SU2R" in target_labels else "U1R"
        if r_label not in out:
            raise ValueError(f"Cannot split U1Y into {r_label}+U1BL without {r_label}.")
        alpha1_inv = _alpha_inv(src["U1Y"])
        alpha_r_inv = _alpha_inv(out[r_label])
        alpha_bl_inv = (alpha1_inv - (3.0 / 5.0) * alpha_r_inv) / (2.0 / 5.0)
        out["U1BL"] = _g_from_alpha_inv(alpha_bl_inv)

    if "U1Y" in target_labels and "U1Y" not in out:
        r_value = None
        if "SU2R" in src:
            r_value = src["SU2R"]
        elif "U1R" in src:
            r_value = src["U1R"]
        elif "SU2R" in out:
            r_value = out["SU2R"]
        elif "U1R" in out:
            r_value = out["U1R"]
        bl_value = src.get("U1BL", out.get("U1BL"))
        if r_value is not None and bl_value is not None:
            alpha1_inv = (3.0 / 5.0) * _alpha_inv(r_value) + (2.0 / 5.0) * _alpha_inv(bl_value)
            out["U1Y"] = _g_from_alpha_inv(alpha1_inv)

    out.update(user)
    missing = [label for label in target_labels if label not in out]
    if missing:
        raise ValueError(
            f"Cannot match {source_group!r} to {target_group!r}; missing target couplings {missing}. "
            "Provide them in the stage matching dictionary."
        )
    return {label: out[label] for label in target_labels}


def _initial_couplings(initial_couplings: dict[str, float] | Iterable[float] | None) -> dict[str, float]:
    if initial_couplings is None:
        return {
            "U1Y": DEFAULT_MZ_COUPLINGS["g1"],
            "SU2L": DEFAULT_MZ_COUPLINGS["g2"],
            "SU3C": DEFAULT_MZ_COUPLINGS["g3"],
        }
    if isinstance(initial_couplings, dict):
        return _normalize_coupling_dict(initial_couplings)
    values = list(initial_couplings)
    if len(values) != 3:
        raise ValueError("Iterable initial_couplings must be [g1, g2, g3].")
    return {"U1Y": float(values[0]), "SU2L": float(values[1]), "SU3C": float(values[2])}


def _parse_stage(stage: dict[str, Any] | StageSpec) -> StageSpec:
    if isinstance(stage, StageSpec):
        return stage
    return StageSpec(
        group=str(stage["group"]),
        end=float(stage.get("end", stage.get("end_scale", stage.get("scale")))),
        fields=tuple(dict(field) for field in stage.get("fields", ())),
        include_defaults=bool(stage.get("include_defaults", canonical_group(str(stage["group"])) != "sm")),
        matching=None if stage.get("matching") is None else dict(stage["matching"]),
        points=None if stage.get("points") is None else int(stage["points"]),
    )


def run_staged_gauge_rges(
    stages: Iterable[dict[str, Any] | StageSpec],
    initial_scale: float = DEFAULT_MZ,
    initial_couplings: dict[str, float] | Iterable[float] | None = None,
    loops: int = 1,
    n_points_per_stage: int = 160,
) -> dict[str, Any]:
    if loops not in {1, 2}:
        raise ValueError("loops must be 1 or 2.")
    if initial_scale <= 0:
        raise ValueError("initial_scale must be positive.")

    parsed_stages = [_parse_stage(stage) for stage in stages]
    if not parsed_stages:
        raise ValueError("At least one stage is required.")

    current_scale = float(initial_scale)
    current_group = "sm"
    current_couplings = _initial_couplings(initial_couplings)
    all_samples: list[dict[str, Any]] = []
    stage_outputs: list[dict[str, Any]] = []

    for index, stage in enumerate(parsed_stages):
        group = canonical_group(stage.group)
        if stage.end <= 0:
            raise ValueError("All stage end scales must be positive.")
        if stage.end <= current_scale:
            raise ValueError("Stages must be ordered with increasing end scales.")

        matched = None
        if index == 0 and group != current_group:
            current_couplings = match_couplings(current_group, group, current_couplings, stage.matching)
            matched = dict(current_couplings)
        elif group != current_group:
            current_couplings = match_couplings(current_group, group, current_couplings, stage.matching)
            matched = dict(current_couplings)

        labels = group_labels(group)
        model = None if group == "sm" else MODELS[group]
        fields = _stage_fields(stage, model)
        y0 = _couplings_to_array(labels, current_couplings)
        t0 = float(np.log(current_scale))
        t1 = float(np.log(stage.end))
        points = int(max(2, stage.points or n_points_per_stage))
        t_eval = np.linspace(t0, t1, points)

        def rhs(t: float, state: np.ndarray) -> np.ndarray:
            return gauge_only_beta(t, state, group, fields, loops=loops)

        sol = solve_ivp(rhs, (t0, t1), y0, t_eval=t_eval, rtol=1e-8, atol=1e-10)
        if not sol.success:
            raise RuntimeError(sol.message)

        scales = np.exp(sol.t)
        values = sol.y.T
        for scale, row in zip(scales, values):
            couplings = _array_to_couplings(labels, row)
            all_samples.append(
                {
                    "scale_GeV": float(scale),
                    "group": group,
                    "couplings": couplings,
                    "inverse_alpha": {label: _alpha_inv(value) for label, value in couplings.items()},
                }
            )

        current_scale = float(stage.end)
        current_couplings = _array_to_couplings(labels, values[-1])
        current_group = group
        active_end_a, active_end_b = active_stage_coefficients(float(np.log(current_scale)), group, fields)
        stage_outputs.append(
            {
                "group": group,
                "start_scale_GeV": float(np.exp(t0)),
                "end_scale_GeV": float(stage.end),
                "gauge_order": labels,
                "matched_initial_couplings": matched,
                "initial_couplings": _array_to_couplings(labels, y0),
                "final_couplings": dict(current_couplings),
                "final_inverse_alpha": {label: _alpha_inv(value) for label, value in current_couplings.items()},
                "fields": fields,
                "active_one_loop_at_end": active_end_a.tolist(),
                "active_two_loop_at_end": active_end_b.tolist(),
            }
        )

    return {
        "runner": "staged gauge-only RGE",
        "loops": loops,
        "scale_variable": "natural log of mu/GeV",
        "kinetic_mixing": "ignored; multiple U(1) stages use diagonal-coupling running",
        "matching": "tree-level matching from RGE-calc/note.tex; finite threshold corrections are not included",
        "initial_scale_GeV": float(initial_scale),
        "initial_group": "sm",
        "final_group": current_group,
        "final_scale_GeV": current_scale,
        "final_couplings": dict(current_couplings),
        "final_inverse_alpha": {label: _alpha_inv(value) for label, value in current_couplings.items()},
        "stages": stage_outputs,
        "samples": all_samples,
    }
