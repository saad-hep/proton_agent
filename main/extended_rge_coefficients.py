from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

import numpy as np


U1_BL_NORMALIZATION = Fraction(3, 8)

_SU4_INDEX = {
    "1": Fraction(0),
    "singlet": Fraction(0),
    "4": Fraction(1, 2),
    "bar4": Fraction(1, 2),
    "4bar": Fraction(1, 2),
    "-4": Fraction(1, 2),
    "6": Fraction(1),
    "10": Fraction(3),
    "bar10": Fraction(3),
    "10bar": Fraction(3),
    "15": Fraction(4),
    "adjoint": Fraction(4),
    "20": Fraction(13, 2),
    "p20": Fraction(8),
    "20p": Fraction(8),
    "20prime": Fraction(8),
    "20'": Fraction(8),
}


def _fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction(value).limit_denominator()
    return Fraction(str(value).strip())


def _rep_key(rep: Any) -> str:
    return str(rep).strip().lower().replace(" ", "").replace("_", "")


def su2_dimension(rep: Any) -> int:
    dim = int(rep)
    if dim < 1:
        raise ValueError("SU(2) dimension must be >= 1")
    return dim


def su2_index(rep: Any) -> Fraction:
    n = su2_dimension(rep)
    return Fraction(n * (n * n - 1), 12)


def su2_casimir(rep: Any) -> Fraction:
    n = su2_dimension(rep)
    return Fraction(n * n - 1, 4)


def su3_index(rep: Any) -> Fraction:
    from rge_coefficients import su3_dynkin_index
    return su3_dynkin_index(rep)


def su3_dimension(rep: Any) -> int:
    from rge_coefficients import su3_dimension as dim
    return dim(rep)


def su3_casimir(rep: Any) -> Fraction:
    from rge_coefficients import su3_quadratic_casimir
    return su3_quadratic_casimir(rep)


def su4_dimension(rep: Any) -> int:
    key = _rep_key(rep)
    if key.startswith("bar"):
        key = key[3:]
    if key.endswith("bar"):
        key = key[:-3]
    if key in {"p20", "20p", "20prime", "20'"}:
        return 20
    if key == "adjoint":
        return 15
    try:
        return int(key)
    except ValueError as exc:
        raise ValueError(f"Unknown SU(4) representation {rep!r}") from exc


def su4_index(rep: Any) -> Fraction:
    key = _rep_key(rep)
    if key not in _SU4_INDEX:
        raise ValueError("Unknown SU(4) representation {!r}. Use 1, 4, bar4, 6, 10, bar10, 15, 20, p20.".format(rep))
    return _SU4_INDEX[key]


def su4_casimir(rep: Any) -> Fraction:
    dim = su4_dimension(rep)
    if dim == 0:
        return Fraction(0)
    return Fraction(15) * su4_index(rep) / dim


@dataclass(frozen=True)
class GaugeFactor:
    key: str
    label: str
    kind: str
    c2g: Fraction
    u1_norm: Fraction = Fraction(1)


@dataclass(frozen=True)
class ExtendedField:
    name: str
    kind: str
    reps: dict[str, Any]
    charges: dict[str, Any] | None = None
    generations: int = 1
    real_scalar: bool = False
    fermion_type: str = "weyl"
    mass: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], group: "GaugeModel") -> "ExtendedField":
        reps: dict[str, Any] = {}
        charges: dict[str, Any] = {}
        raw_reps = data.get("reps") or {}
        raw_charges = data.get("charges") or {}
        for factor in group.factors:
            if factor.kind == "u1":
                value = raw_charges.get(factor.key, data.get(factor.key, data.get(f"q_{factor.key}")))
                if value is None:
                    value = 0
                charges[factor.key] = value
            else:
                value = raw_reps.get(factor.key, data.get(factor.key))
                if value is None:
                    value = 1
                reps[factor.key] = value
        mass = data.get("mass", data.get("mass_GeV", data.get("mass_scale")))
        return cls(
            name=str(data.get("name", data.get("label", "BSM"))),
            kind=str(data["kind"]).lower(),
            reps=reps,
            charges=charges,
            generations=int(data.get("generations", data.get("multiplicity", 1))),
            real_scalar=bool(data.get("real_scalar", data.get("real", False))),
            fermion_type=str(data.get("fermion_type", data.get("chirality", "weyl"))).lower(),
            mass=None if mass is None else float(mass),
        )


@dataclass(frozen=True)
class GaugeModel:
    name: str
    factors: tuple[GaugeFactor, ...]
    default_fields: tuple[dict[str, Any], ...]


def factor_dimension(factor: GaugeFactor, field: ExtendedField) -> int:
    if factor.kind == "u1":
        return 1
    rep = field.reps.get(factor.key, 1)
    if factor.kind == "su2":
        return su2_dimension(rep)
    if factor.kind == "su3":
        return su3_dimension(rep)
    if factor.kind == "su4":
        return su4_dimension(rep)
    raise ValueError(f"Unknown gauge factor kind: {factor.kind}")


def factor_index(factor: GaugeFactor, field: ExtendedField) -> Fraction:
    if factor.kind == "u1":
        q = _fraction((field.charges or {}).get(factor.key, 0))
        return factor.u1_norm * q * q
    rep = field.reps.get(factor.key, 1)
    if factor.kind == "su2":
        return su2_index(rep)
    if factor.kind == "su3":
        return su3_index(rep)
    if factor.kind == "su4":
        return su4_index(rep)
    raise ValueError(f"Unknown gauge factor kind: {factor.kind}")


def factor_casimir(factor: GaugeFactor, field: ExtendedField) -> Fraction:
    if factor.kind == "u1":
        q = _fraction((field.charges or {}).get(factor.key, 0))
        return factor.u1_norm * q * q
    rep = field.reps.get(factor.key, 1)
    if factor.kind == "su2":
        return su2_casimir(rep)
    if factor.kind == "su3":
        return su3_casimir(rep)
    if factor.kind == "su4":
        return su4_casimir(rep)
    raise ValueError(f"Unknown gauge factor kind: {factor.kind}")


def field_factor(field: ExtendedField) -> Fraction:
    if field.kind == "scalar":
        return Fraction(1, 2) if field.real_scalar else Fraction(1)
    if field.kind == "fermion":
        return Fraction(2) if field.fermion_type == "dirac" else Fraction(1)
    raise ValueError("field.kind must be scalar or fermion")


def field_s_vector(group: GaugeModel, field: ExtendedField) -> np.ndarray:
    values = []
    ffac = field_factor(field)
    for factor in group.factors:
        if factor.kind == "u1":
            q = _fraction((field.charges or {}).get(factor.key, 0))
            raw = q * q
            for other in group.factors:
                if other.key != factor.key:
                    raw *= factor_dimension(other, field)
            values.append(ffac * factor.u1_norm * raw)
        else:
            raw = factor_index(factor, field)
            for other in group.factors:
                if other.key != factor.key:
                    raw *= factor_dimension(other, field)
            values.append(ffac * raw)
    return np.array(values, dtype=object)


def field_c_vector(group: GaugeModel, field: ExtendedField) -> np.ndarray:
    return np.array([factor_casimir(factor, field) for factor in group.factors], dtype=object)


def field_one_loop_delta(group: GaugeModel, field: ExtendedField) -> np.ndarray:
    s = field_s_vector(group, field)
    if field.kind == "scalar":
        delta = Fraction(1, 3) * s
    elif field.kind == "fermion":
        delta = Fraction(2, 3) * s
    else:
        raise ValueError("field.kind must be scalar or fermion")
    return field.generations * delta


def field_two_loop_delta(group: GaugeModel, field: ExtendedField) -> np.ndarray:
    s = field_s_vector(group, field)
    c = field_c_vector(group, field)
    c2g = np.array([factor.c2g for factor in group.factors], dtype=object)
    if field.kind == "scalar":
        delta = Fraction(2, 3) * np.diag(c2g * s) + 4 * np.outer(c, s)
    elif field.kind == "fermion":
        delta = Fraction(10, 3) * np.diag(c2g * s) + 2 * np.outer(c, s)
    else:
        raise ValueError("field.kind must be scalar or fermion")
    return field.generations * delta.T


def parse_fields(group: GaugeModel, fields: Iterable[ExtendedField | dict[str, Any]]) -> list[ExtendedField]:
    return [field if isinstance(field, ExtendedField) else ExtendedField.from_dict(field, group) for field in fields]


def coefficient_summary(group: GaugeModel, fields: Iterable[ExtendedField | dict[str, Any]] | None = None) -> dict[str, Any]:
    parsed = parse_fields(group, group.default_fields if fields is None else fields)
    a = np.array([-Fraction(11, 3) * factor.c2g for factor in group.factors], dtype=object)
    b = -Fraction(34, 3) * np.diag(np.array([factor.c2g * factor.c2g for factor in group.factors], dtype=object))
    per_field = []
    for field in parsed:
        da = field_one_loop_delta(group, field)
        db = field_two_loop_delta(group, field)
        a += da
        b += db
        per_field.append({
            "name": field.name,
            "kind": field.kind,
            "reps": field.reps,
            "charges": field.charges or {},
            "generations": field.generations,
            "real_scalar": field.real_scalar if field.kind == "scalar" else None,
            "fermion_type": field.fermion_type if field.kind == "fermion" else None,
            "mass": field.mass,
            "one_loop_delta": fraction_array_to_strings(da),
            "two_loop_delta": fraction_matrix_to_strings(db),
        })
    return {
        "model": group.name,
        "gauge_order": [factor.label for factor in group.factors],
        "two_loop_matrix_convention": "B_ij in beta_i = g_i^3/(16*pi^2)^2 sum_j B_ij g_j^2",
        "one_loop": fraction_array_to_strings(a),
        "two_loop": fraction_matrix_to_strings(b),
        "one_loop_float": [float(Fraction(x)) for x in fraction_array_to_strings(a)],
        "two_loop_float": [[float(Fraction(x)) for x in row] for row in fraction_matrix_to_strings(b)],
        "fields": per_field,
    }


def fraction_array_to_strings(values: np.ndarray) -> list[str]:
    return [str(Fraction(v)) for v in values.tolist()]


def fraction_matrix_to_strings(values: np.ndarray) -> list[list[str]]:
    return [[str(Fraction(v)) for v in row] for row in values.tolist()]


LR_MODEL = GaugeModel(
    name="LR",
    factors=(
        GaugeFactor("su3c", "SU3C", "su3", Fraction(3)),
        GaugeFactor("su2l", "SU2L", "su2", Fraction(2)),
        GaugeFactor("su2r", "SU2R", "su2", Fraction(2)),
        GaugeFactor("bl", "U1BL", "u1", Fraction(0), U1_BL_NORMALIZATION),
    ),
    default_fields=(
        {"name": "QL", "kind": "fermion", "su3c": "3", "su2l": 2, "su2r": 1, "bl": "1/3", "generations": 3},
        {"name": "QR", "kind": "fermion", "su3c": "3", "su2l": 1, "su2r": 2, "bl": "1/3", "generations": 3},
        {"name": "LL", "kind": "fermion", "su3c": "1", "su2l": 2, "su2r": 1, "bl": "-1", "generations": 3},
        {"name": "LR", "kind": "fermion", "su3c": "1", "su2l": 1, "su2r": 2, "bl": "-1", "generations": 3},
        {"name": "Phi", "kind": "scalar", "su3c": "1", "su2l": 2, "su2r": 2, "bl": "0"},
    ),
)

PS_MODEL = GaugeModel(
    name="PS",
    factors=(
        GaugeFactor("su4c", "SU4C", "su4", Fraction(4)),
        GaugeFactor("su2l", "SU2L", "su2", Fraction(2)),
        GaugeFactor("su2r", "SU2R", "su2", Fraction(2)),
    ),
    default_fields=(
        {"name": "FL", "kind": "fermion", "su4c": "4", "su2l": 2, "su2r": 1, "generations": 3},
        {"name": "FR", "kind": "fermion", "su4c": "4", "su2l": 1, "su2r": 2, "generations": 3},
        {"name": "Phi", "kind": "scalar", "su4c": "1", "su2l": 2, "su2r": 2},
    ),
)

G214_MODEL = GaugeModel(
    name="G214",
    factors=(
        GaugeFactor("su4c", "SU4C", "su4", Fraction(4)),
        GaugeFactor("su2l", "SU2L", "su2", Fraction(2)),
        GaugeFactor("r", "U1R", "u1", Fraction(0), Fraction(1)),
    ),
    default_fields=(
        {"name": "FL", "kind": "fermion", "su4c": "4", "su2l": 2, "r": "0", "generations": 3},
        {"name": "uR", "kind": "fermion", "su4c": "4", "su2l": 1, "r": "1/2", "generations": 3},
        {"name": "dR", "kind": "fermion", "su4c": "4", "su2l": 1, "r": "-1/2", "generations": 3},
        {"name": "H", "kind": "scalar", "su4c": "1", "su2l": 2, "r": "1/2"},
    ),
)

TTO_MODEL = GaugeModel(
    name="3211/TTO",
    factors=(
        GaugeFactor("su3c", "SU3C", "su3", Fraction(3)),
        GaugeFactor("su2l", "SU2L", "su2", Fraction(2)),
        GaugeFactor("r", "U1R", "u1", Fraction(0), Fraction(1)),
        GaugeFactor("bl", "U1BL", "u1", Fraction(0), Fraction(3, 2)),
    ),
    default_fields=(
        {"name": "QL", "kind": "fermion", "su3c": "3", "su2l": 2, "r": "0", "bl": "1/6", "generations": 3},
        {"name": "uR", "kind": "fermion", "su3c": "3", "su2l": 1, "r": "1/2", "bl": "1/6", "generations": 3},
        {"name": "dR", "kind": "fermion", "su3c": "3", "su2l": 1, "r": "-1/2", "bl": "1/6", "generations": 3},
        {"name": "LL", "kind": "fermion", "su3c": "1", "su2l": 2, "r": "0", "bl": "-1/2", "generations": 3},
        {"name": "eR", "kind": "fermion", "su3c": "1", "su2l": 1, "r": "1/2", "bl": "-1/2", "generations": 3},
        {"name": "nuR", "kind": "fermion", "su3c": "1", "su2l": 1, "r": "-1/2", "bl": "-1/2", "generations": 3},
        {"name": "H", "kind": "scalar", "su3c": "1", "su2l": 2, "r": "-1/2", "bl": "0"},
    ),
)

MODELS = {"lr": LR_MODEL, "3221": LR_MODEL, "ps": PS_MODEL, "422": PS_MODEL, "g214": G214_MODEL, "421": G214_MODEL, "tto": TTO_MODEL, "3211": TTO_MODEL}
