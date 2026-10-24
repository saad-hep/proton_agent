from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Literal

import numpy as np

from rge_coefficients import (
    su2_dimension,
    su2_dynkin_index,
    su2_quadratic_casimir,
    su3_dimension,
    su3_dynkin_index,
    su3_quadratic_casimir,
)


GaugeGroup = Literal["U1Y", "SU2L", "SU3C"]
MSSMFieldKind = Literal["chiral", "scalar", "fermion"]
FermionType = Literal["weyl", "dirac"]

U1_Y_NORMALIZATION = Fraction(3, 5)
GAUGE_ORDER = ["U1Y", "SU2L", "SU3C"]
C2_G = np.array([Fraction(0), Fraction(2), Fraction(3)], dtype=object)


def _fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction(value).limit_denominator()
    return Fraction(str(value).strip())


@dataclass(frozen=True)
class MSSMField:
    """Extra field under U(1)_Y x SU(2)_L x SU(3)_C.

    kind="chiral" means a full N=1 chiral superfield: one complex scalar plus
    one Weyl fermion in the same representation. This is the natural MSSM+BSM
    input and has a SUSY two-loop gauge contribution.
    """

    name: str
    kind: MSSMFieldKind
    su3: Any
    su2: int
    hypercharge: Fraction | int | float | str
    generations: int = 1
    real_scalar: bool = False
    fermion_type: FermionType = "weyl"
    mass: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MSSMField":
        mass = data.get("mass", data.get("mass_GeV", data.get("mass_scale")))
        kind = str(data.get("kind", data.get("superfield_type", "chiral"))).lower()
        if kind in {"superfield", "chiral_superfield", "chiral-superfield"}:
            kind = "chiral"
        return cls(
            name=str(data.get("name", data.get("label", "X"))),
            kind=kind,
            su3=data.get("su3", data.get("su3c", "1")),
            su2=int(data.get("su2", data.get("su2l", 1))),
            hypercharge=_fraction(data.get("hypercharge", data.get("y", 0))),
            generations=int(data.get("generations", data.get("multiplicity", 1))),
            real_scalar=bool(data.get("real_scalar", data.get("real", False))),
            fermion_type=str(data.get("fermion_type", data.get("chirality", "weyl"))).lower(),
            mass=None if mass is None else float(mass),
        )


def field_s_vector(field: MSSMField) -> np.ndarray:
    y = _fraction(field.hypercharge)
    d3 = su3_dimension(field.su3)
    d2 = su2_dimension(field.su2)
    t2 = su2_dynkin_index(field.su2)
    t3 = su3_dynkin_index(field.su3)
    factor = Fraction(1)
    if field.kind == "scalar" and field.real_scalar:
        factor = Fraction(1, 2)
    if field.kind == "fermion" and field.fermion_type == "dirac":
        factor = Fraction(2)
    return factor * np.array(
        [
            U1_Y_NORMALIZATION * d3 * d2 * y * y,
            d3 * t2,
            d2 * t3,
        ],
        dtype=object,
    )


def field_c_vector(field: MSSMField) -> np.ndarray:
    y = _fraction(field.hypercharge)
    return np.array(
        [
            U1_Y_NORMALIZATION * y * y,
            su2_quadratic_casimir(field.su2),
            su3_quadratic_casimir(field.su3),
        ],
        dtype=object,
    )


def field_one_loop_delta(field: MSSMField) -> np.ndarray:
    s = field_s_vector(field)
    if field.kind == "chiral":
        delta = s
    elif field.kind == "scalar":
        delta = Fraction(1, 3) * s
    elif field.kind == "fermion":
        delta = Fraction(2, 3) * s
    else:
        raise ValueError("kind must be chiral, scalar, or fermion")
    return field.generations * delta


def field_two_loop_delta(field: MSSMField) -> np.ndarray:
    s = field_s_vector(field)
    c = field_c_vector(field)
    if field.kind == "chiral":
        delta = 2 * np.diag(C2_G * s) + 4 * np.outer(s, c)
    elif field.kind == "scalar":
        delta = Fraction(2, 3) * np.diag(C2_G * s) + 4 * np.outer(c, s).T
    elif field.kind == "fermion":
        delta = Fraction(10, 3) * np.diag(C2_G * s) + 2 * np.outer(c, s).T
    else:
        raise ValueError("kind must be chiral, scalar, or fermion")
    return field.generations * delta


def mssm_chiral_fields() -> list[MSSMField]:
    return [
        MSSMField("Q", "chiral", "3", 2, Fraction(1, 6), generations=3),
        MSSMField("uR", "chiral", "3", 1, Fraction(2, 3), generations=3),
        MSSMField("dR", "chiral", "3", 1, Fraction(-1, 3), generations=3),
        MSSMField("L", "chiral", "1", 2, Fraction(-1, 2), generations=3),
        MSSMField("eR", "chiral", "1", 1, Fraction(-1, 1), generations=3),
        MSSMField("Hu", "chiral", "1", 2, Fraction(1, 2), generations=1),
        MSSMField("Hd", "chiral", "1", 2, Fraction(-1, 2), generations=1),
    ]


def _base_coefficients() -> tuple[np.ndarray, np.ndarray]:
    s_total = np.array([Fraction(0), Fraction(0), Fraction(0)], dtype=object)
    b_two = -6 * np.diag(C2_G * C2_G)
    for field in mssm_chiral_fields():
        s = field_s_vector(field) * field.generations
        c = field_c_vector(field)
        s_total += s
        b_two += 2 * np.diag(C2_G * s) + 4 * np.outer(s, c)
    b_one = s_total - 3 * C2_G
    return b_one, b_two


MSSM_ONE_LOOP, MSSM_TWO_LOOP = _base_coefficients()


def _parse_fields(fields: Iterable[MSSMField | dict[str, Any]]) -> list[MSSMField]:
    return [field if isinstance(field, MSSMField) else MSSMField.from_dict(field) for field in fields]


def fraction_array_to_strings(values: np.ndarray) -> list[str]:
    return [str(Fraction(v)) for v in values.tolist()]


def fraction_matrix_to_strings(values: np.ndarray) -> list[list[str]]:
    return [[str(Fraction(v)) for v in row] for row in values.tolist()]


def coefficient_summary(fields: Iterable[MSSMField | dict[str, Any]] = ()) -> dict[str, Any]:
    parsed = _parse_fields(fields)
    one_delta = np.array([Fraction(0), Fraction(0), Fraction(0)], dtype=object)
    two_delta = np.zeros((3, 3), dtype=object)
    per_field = []
    for field in parsed:
        da = field_one_loop_delta(field)
        db = field_two_loop_delta(field)
        one_delta += da
        two_delta += db
        per_field.append(
            {
                "name": field.name,
                "kind": field.kind,
                "su3": field.su3,
                "su2": field.su2,
                "hypercharge": str(_fraction(field.hypercharge)),
                "generations": field.generations,
                "fermion_type": field.fermion_type if field.kind == "fermion" else None,
                "real_scalar": field.real_scalar if field.kind == "scalar" else None,
                "mass": field.mass,
                "one_loop_delta": fraction_array_to_strings(da),
                "two_loop_delta": fraction_matrix_to_strings(db),
            }
        )
    total_one = MSSM_ONE_LOOP + one_delta
    total_two = MSSM_TWO_LOOP + two_delta
    return {
        "model": "MSSM",
        "gauge_order": GAUGE_ORDER,
        "field_convention": "kind='chiral' is a full N=1 chiral superfield; scalar/fermion are component fields.",
        "two_loop_matrix_convention": "B_ij in beta_i = g_i^3/(16*pi^2)^2 sum_j B_ij g_j^2; chiral fields use the SUSY gauge-only two-loop formula.",
        "mssm_one_loop": fraction_array_to_strings(MSSM_ONE_LOOP),
        "mssm_two_loop": fraction_matrix_to_strings(MSSM_TWO_LOOP),
        "bsm_one_loop_delta": fraction_array_to_strings(one_delta),
        "bsm_two_loop_delta": fraction_matrix_to_strings(two_delta),
        "total_one_loop": fraction_array_to_strings(total_one),
        "total_two_loop": fraction_matrix_to_strings(total_two),
        "mssm_one_loop_float": [float(x) for x in MSSM_ONE_LOOP],
        "mssm_two_loop_float": [[float(x) for x in row] for row in MSSM_TWO_LOOP.tolist()],
        "total_one_loop_float": [float(x) for x in total_one],
        "total_two_loop_float": [[float(x) for x in row] for row in total_two.tolist()],
        "fields": per_field,
    }


def coefficients_as_float(fields: Iterable[MSSMField | dict[str, Any]] = ()) -> tuple[np.ndarray, np.ndarray]:
    summary = coefficient_summary(fields)
    a = np.array([float(Fraction(x)) for x in summary["total_one_loop"]], dtype=float)
    b = np.array([[float(Fraction(x)) for x in row] for row in summary["total_two_loop"]], dtype=float)
    return a, b
