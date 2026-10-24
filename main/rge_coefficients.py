from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Literal

import numpy as np


GaugeGroup = Literal["U1Y", "SU2L", "SU3C"]
FieldKind = Literal["scalar", "fermion"]
FermionType = Literal["weyl", "dirac"]

U1_Y_NORMALIZATION = Fraction(3, 5)
SM_ONE_LOOP = np.array([Fraction(41, 10), Fraction(-19, 6), Fraction(-7, 1)], dtype=object)
SM_TWO_LOOP = np.array(
    [
        [Fraction(199, 50), Fraction(27, 10), Fraction(44, 5)],
        [Fraction(9, 10), Fraction(35, 6), Fraction(12, 1)],
        [Fraction(11, 10), Fraction(9, 2), Fraction(-26, 1)],
    ],
    dtype=object,
)

SM_YUKAWA_TRACE_COEFFICIENTS = np.array(
    [
        [Fraction(17, 10), Fraction(1, 2), Fraction(3, 2)],
        [Fraction(3, 2), Fraction(3, 2), Fraction(1, 2)],
        [Fraction(2, 1), Fraction(2, 1), Fraction(0, 1)],
    ],
    dtype=object,
)

_SU3_ALIASES = {
    "1": (0, 0),
    "singlet": (0, 0),
    "3": (1, 0),
    "fundamental": (1, 0),
    "bar3": (0, 1),
    "3bar": (0, 1),
    "-3": (0, 1),
    "6": (2, 0),
    "bar6": (0, 2),
    "6bar": (0, 2),
    "8": (1, 1),
    "adjoint": (1, 1),
    "10": (3, 0),
    "bar10": (0, 3),
    "10bar": (0, 3),
    "15": (2, 1),
    "bar15": (1, 2),
    "15bar": (1, 2),
    "p15": (4, 0),
    "15p": (4, 0),
    "15prime": (4, 0),
    "15'": (4, 0),
}


@dataclass(frozen=True)
class SMField:
    """A scalar or Weyl/Dirac fermion representation under SU(3)xSU(2)xU(1)."""

    name: str
    kind: FieldKind
    su3: Any
    su2: int
    hypercharge: Fraction | int | float | str
    generations: int = 1
    real_scalar: bool = False
    fermion_type: FermionType = "weyl"
    mass: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SMField":
        mass = data.get("mass", data.get("mass_GeV", data.get("mass_scale")))
        return cls(
            name=str(data.get("name", data.get("label", "BSM"))),
            kind=str(data["kind"]).lower(),
            su3=data.get("su3", data.get("su3c", "1")),
            su2=int(data.get("su2", data.get("su2l", 1))),
            hypercharge=_fraction(data.get("hypercharge", data.get("y", 0))),
            generations=int(data.get("generations", data.get("multiplicity", 1))),
            real_scalar=bool(data.get("real_scalar", data.get("real", False))),
            fermion_type=str(data.get("fermion_type", data.get("chirality", "weyl"))).lower(),
            mass=None if mass is None else float(mass),
        )


def _fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction(value).limit_denominator()
    return Fraction(str(value).strip())


def _su3_dynkin(rep: Any) -> tuple[int, int]:
    if isinstance(rep, (tuple, list)) and len(rep) == 2:
        return int(rep[0]), int(rep[1])
    key = str(rep).strip().lower().replace(" ", "").replace("_", "")
    if key in _SU3_ALIASES:
        return _SU3_ALIASES[key]
    raise ValueError(
        f"Unknown SU(3) representation {rep!r}. Use a common label such as "
        "'1', '3', 'bar3', '6', '8', '10', '15', 'p15', or Dynkin labels [p, q]."
    )


def su2_dimension(rep: int) -> int:
    if int(rep) < 1:
        raise ValueError("SU(2) representation dimension must be >= 1")
    return int(rep)


def su2_dynkin_index(rep: int) -> Fraction:
    n = su2_dimension(rep)
    return Fraction(n * (n * n - 1), 12)


def su2_quadratic_casimir(rep: int) -> Fraction:
    n = su2_dimension(rep)
    return Fraction(n * n - 1, 4)


def su3_dimension(rep: Any) -> int:
    p, q = _su3_dynkin(rep)
    return (p + 1) * (q + 1) * (p + q + 2) // 2


def su3_quadratic_casimir(rep: Any) -> Fraction:
    p, q = _su3_dynkin(rep)
    return Fraction(p * p + q * q + p * q + 3 * p + 3 * q, 3)


def su3_dynkin_index(rep: Any) -> Fraction:
    dim = su3_dimension(rep)
    casimir = su3_quadratic_casimir(rep)
    return Fraction(dim, 8) * casimir


def _field_factor(field: SMField) -> Fraction:
    if field.kind == "scalar":
        return Fraction(1, 2) if field.real_scalar else Fraction(1, 1)
    if field.kind == "fermion":
        return Fraction(2, 1) if field.fermion_type == "dirac" else Fraction(1, 1)
    raise ValueError("field.kind must be 'scalar' or 'fermion'")


def field_invariants(field: SMField) -> dict[str, Any]:
    y = _fraction(field.hypercharge)
    d3 = su3_dimension(field.su3)
    d2 = su2_dimension(field.su2)
    t2 = su2_dynkin_index(field.su2)
    t3 = su3_dynkin_index(field.su3)
    c1 = U1_Y_NORMALIZATION * y * y
    c2 = su2_quadratic_casimir(field.su2)
    c3 = su3_quadratic_casimir(field.su3)
    factor = _field_factor(field)

    s = np.array(
        [
            factor * U1_Y_NORMALIZATION * d3 * d2 * y * y,
            factor * d3 * t2,
            factor * d2 * t3,
        ],
        dtype=object,
    )
    c = np.array([c1, c2, c3], dtype=object)

    return {
        "dimension": {"su3": d3, "su2": d2},
        "dynkin_index": {"u1": c1, "su2": t2, "su3": t3},
        "quadratic_casimir": {"u1": c1, "su2": c2, "su3": c3},
        "S": s,
        "C": c,
        "field_factor": factor,
    }


def field_one_loop_delta(field: SMField) -> np.ndarray:
    inv = field_invariants(field)
    s = inv["S"]
    if field.kind == "scalar":
        delta = Fraction(1, 3) * s
    elif field.kind == "fermion":
        delta = Fraction(2, 3) * s
    else:
        raise ValueError("field.kind must be 'scalar' or 'fermion'")
    return field.generations * delta


def field_two_loop_delta(field: SMField) -> np.ndarray:
    inv = field_invariants(field)
    s = inv["S"]
    c = inv["C"]
    c2g = np.array([Fraction(0, 1), Fraction(2, 1), Fraction(3, 1)], dtype=object)

    if field.kind == "scalar":
        delta = Fraction(2, 3) * np.diag(c2g * s) + 4 * np.outer(c, s)
    elif field.kind == "fermion":
        delta = Fraction(10, 3) * np.diag(c2g * s) + 2 * np.outer(c, s)
    else:
        raise ValueError("field.kind must be 'scalar' or 'fermion'")

    return field.generations * delta.T


def sm_fields() -> list[SMField]:
    return [
        SMField("H", "scalar", "1", 2, Fraction(1, 2)),
        SMField("Q", "fermion", "3", 2, Fraction(1, 6), generations=3),
        SMField("uR", "fermion", "3", 1, Fraction(2, 3), generations=3),
        SMField("dR", "fermion", "3", 1, Fraction(-1, 3), generations=3),
        SMField("L", "fermion", "1", 2, Fraction(-1, 2), generations=3),
        SMField("eR", "fermion", "1", 1, Fraction(-1, 1), generations=3),
    ]


def coefficient_summary(fields: Iterable[SMField | dict[str, Any]] = ()) -> dict[str, Any]:
    parsed_fields = [f if isinstance(f, SMField) else SMField.from_dict(f) for f in fields]
    deltas = []
    one_loop_delta = np.array([Fraction(0), Fraction(0), Fraction(0)], dtype=object)
    two_loop_delta = np.zeros((3, 3), dtype=object)

    for field in parsed_fields:
        da = field_one_loop_delta(field)
        db = field_two_loop_delta(field)
        one_loop_delta += da
        two_loop_delta += db
        deltas.append(
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
                "one_loop_delta": _fraction_array_to_strings(da),
                "two_loop_delta": _fraction_matrix_to_strings(db),
            }
        )

    return {
        "gauge_order": ["U1Y", "SU2L", "SU3C"],
        "two_loop_matrix_convention": "B_ij in beta_i = g_i^3/(16*pi^2)^2 sum_j B_ij g_j^2",
        "sm_one_loop": _fraction_array_to_strings(SM_ONE_LOOP),
        "sm_two_loop": _fraction_matrix_to_strings(SM_TWO_LOOP),
        "bsm_one_loop_delta": _fraction_array_to_strings(one_loop_delta),
        "bsm_two_loop_delta": _fraction_matrix_to_strings(two_loop_delta),
        "total_one_loop": _fraction_array_to_strings(SM_ONE_LOOP + one_loop_delta),
        "total_two_loop": _fraction_matrix_to_strings(SM_TWO_LOOP + two_loop_delta),
        "fields": deltas,
    }


def coefficients_as_float(fields: Iterable[SMField | dict[str, Any]] = ()) -> tuple[np.ndarray, np.ndarray]:
    summary = coefficient_summary(fields)
    a = np.array([float(Fraction(x)) for x in summary["total_one_loop"]], dtype=float)
    b = np.array([[float(Fraction(x)) for x in row] for row in summary["total_two_loop"]], dtype=float)
    return a, b


def _fraction_array_to_strings(values: np.ndarray) -> list[str]:
    return [str(Fraction(v)) for v in values.tolist()]


def _fraction_matrix_to_strings(values: np.ndarray) -> list[list[str]]:
    return [[str(Fraction(v)) for v in row] for row in values.tolist()]


def fraction_strings_to_float(values: list[str] | list[list[str]]) -> Any:
    if values and isinstance(values[0], list):
        return [[float(Fraction(x)) for x in row] for row in values]
    return [float(Fraction(x)) for x in values]

