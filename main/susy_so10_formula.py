"""SUSY SO(10) dimension-five proton-decay formulas.

This module follows section 2 of ``formula.tex``.  It implements the SO(10)
triplet tensors C^L/C^R, their mass-basis rotations, the reduced Wino and
Higgsino coefficients for the channels listed in the note, and the common
partial-width formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from experimental_ratios import compare_lifetimes_to_bounds
from hadronic_inputs import hadronic_elements

PROTON_MASS = 0.93827
MASS_PI_ZERO = 0.13497
MASS_PI_PLUS = 0.13957
MASS_K_ZERO = 0.49761
MASS_K_PLUS = 0.49367
HBAR_GEV_SECONDS = 6.582119569e-25
SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0
DEFAULT_M_T = 1.0e17

DEFAULT_HADRONIC = {
    "p->pi+ nubar": 0.189,
    "p->K+ nubar": 0.139,
    "p->pi0 e+": 0.134,
    "p->pi0 mu+": 0.134,
    "p->K0 e+": 0.057,
    "p->K0 mu+": 0.057,
}

MESON_MASSES = {
    "p->pi+ nubar": MASS_PI_PLUS,
    "p->K+ nubar": MASS_K_PLUS,
    "p->pi0 e+": MASS_PI_ZERO,
    "p->pi0 mu+": MASS_PI_ZERO,
    "p->K0 e+": MASS_K_ZERO,
    "p->K0 mu+": MASS_K_ZERO,
}


def dagger(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=complex).conj().T


def _as_3x3(name: str, matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must be a 3x3 matrix, got {matrix.shape}")
    return matrix


def _as_3333(name: str, tensor: np.ndarray) -> np.ndarray:
    tensor = np.asarray(tensor, dtype=complex)
    if tensor.shape != (3, 3, 3, 3):
        raise ValueError(f"{name} must be a 3x3x3x3 tensor, got {tensor.shape}")
    return tensor


def loop_I(a: float, b: float) -> float:
    """I(a,b) from the SO(10) section.

    I(a,b) = a/(b^2-a^2) * [1 + a^2/(b^2-a^2) log(a/b)].
    The note uses positive masses.  The expression is singular at a=b as
    written, so this function raises there instead of inventing a limit.
    """
    a = float(a)
    b = float(b)
    if a <= 0.0 or b <= 0.0:
        raise ValueError("loop_I expects positive masses")
    denom = b * b - a * a
    if abs(denom) <= 1.0e-14 * max(a * a, b * b):
        raise ValueError("loop_I is singular for degenerate a and b in the note's convention")
    return float(a / denom * (1.0 + (a * a / denom) * np.log(a / b)))


@dataclass(frozen=True)
class SO10TripletCoefficients:
    CL: np.ndarray
    CR: np.ndarray


@dataclass(frozen=True)
class SO10MassBasisCoefficients:
    CL_udue: np.ndarray
    CL_uddn: np.ndarray
    CR_udue: np.ndarray


@dataclass(frozen=True)
class SO10MixingMatrices:
    UL: np.ndarray
    UR: np.ndarray
    DL: np.ndarray
    DR: np.ndarray
    EL: np.ndarray
    ER: np.ndarray
    NL: np.ndarray



@dataclass(frozen=True)
class SO10VEVParameters:
    tanbeta: float
    theta_u: float
    phi_u: float
    psi_u: float
    theta_d: float
    phi_d: float
    psi_d: float
    vL: float
    vR: float
    v: float = 174.0


@dataclass(frozen=True)
class SO10VEVComponents:
    vu: float
    vd: float
    v10u: float
    v126u: float
    v1201u: float
    v12015u: float
    v10d: float
    v126d: float
    v1201d: float
    v12015d: float


@dataclass(frozen=True)
class SO10FermionMassMatrices:
    MU: np.ndarray
    MD: np.ndarray
    ME: np.ndarray
    MnuD: np.ndarray
    Mnu: np.ndarray


@dataclass(frozen=True)
class SO10FermionSpectrum:
    masses: dict[str, np.ndarray]
    mixing: SO10MixingMatrices
    yukawas: dict[str, np.ndarray]
    mass_matrices: SO10FermionMassMatrices
    vevs: SO10VEVComponents


@dataclass(frozen=True)
class SO10ReducedCoefficients:
    Cw: dict[str, np.ndarray]
    Ch: dict[str, np.ndarray]


@dataclass(frozen=True)
class SO10Result:
    flavor: SO10TripletCoefficients
    mass_basis: SO10MassBasisCoefficients
    reduced: SO10ReducedCoefficients
    widths: dict[str, float]
    lifetimes: dict[str, float]
    branching_ratios: dict[str, float]
    experimental_comparison: dict
    experimental_summary: dict
    fermion_spectrum: SO10FermionSpectrum | None = None



def beta_functions(tanbeta: float) -> dict[str, float]:
    tanbeta = float(tanbeta)
    if tanbeta <= 0.0:
        raise ValueError("tanbeta must be positive")
    cb = 1.0 / np.sqrt(1.0 + tanbeta * tanbeta)
    sb = tanbeta * cb
    return {"s_beta": sb, "c_beta": cb}


def so10_vev_components(params: SO10VEVParameters) -> SO10VEVComponents:
    """VEV hyperspherical parametrization from the end of section 2."""
    beta = beta_functions(params.tanbeta)
    vu = params.v * beta["s_beta"]
    vd = params.v * beta["c_beta"]
    v10u = vu * np.cos(params.theta_u)
    v126u = vu * np.sin(params.theta_u) * np.cos(params.phi_u)
    v1201u = vu * np.sin(params.theta_u) * np.sin(params.phi_u) * np.cos(params.psi_u)
    v12015u = vu * np.sin(params.theta_u) * np.sin(params.phi_u) * np.sin(params.psi_u)
    v10d = vd * np.cos(params.theta_d)
    v126d = vd * np.sin(params.theta_d) * np.cos(params.phi_d)
    v1201d = vd * np.sin(params.theta_d) * np.sin(params.phi_d) * np.cos(params.psi_d)
    v12015d = vd * np.sin(params.theta_d) * np.sin(params.phi_d) * np.sin(params.psi_d)
    return SO10VEVComponents(
        vu=vu,
        vd=vd,
        v10u=float(v10u),
        v126u=float(v126u),
        v1201u=float(v1201u),
        v12015u=float(v12015u),
        v10d=float(v10d),
        v126d=float(v126d),
        v1201d=float(v1201d),
        v12015d=float(v12015d),
    )


def so10_fermion_mass_matrices(
    h: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    params: SO10VEVParameters,
) -> SO10FermionMassMatrices:
    """SM fermion mass matrices from the final SO(10) equations.

    h is expected real diagonal positive, f complex symmetric, and g complex
    antisymmetric.  The light-neutrino matrix includes both type-II and type-I
    terms: Mnu = vL f - MnuD^T (vR f)^-1 MnuD.
    """
    h = _as_3x3("h", h)
    f = _as_3x3("f", f)
    g = _as_3x3("g", g)
    vevs = so10_vev_components(params)
    MU = vevs.v10u * h + vevs.v126u * f + (vevs.v1201u + vevs.v12015u) * g
    MD = vevs.v10d * h + vevs.v126d * f + (vevs.v1201d + vevs.v12015d) * g
    ME = vevs.v10d * h - 3.0 * vevs.v126d * f + (vevs.v1201d - 3.0 * vevs.v12015d) * g
    MnuD = vevs.v10u * h - 3.0 * vevs.v126u * f + (vevs.v1201u - 3.0 * vevs.v12015u) * g
    Mnu = params.vL * f - MnuD.T @ np.linalg.inv(params.vR * f) @ MnuD
    return SO10FermionMassMatrices(MU=MU, MD=MD, ME=ME, MnuD=MnuD, Mnu=Mnu)


def _diagonalize_dirac_mass_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return masses, U_R, U_L satisfying U_R^dag M U_L = diag(m)."""
    u, values, vh = np.linalg.svd(np.asarray(matrix, dtype=complex))
    order = np.argsort(values)
    values = values[order]
    u = u[:, order]
    vh = vh[order, :]
    UR = u
    UL = dagger(vh)
    return values.real, UR, UL


def _takagi_symmetric(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Numerical Takagi factorization N^T M N = diag(m) for symmetric M."""
    matrix = np.asarray(matrix, dtype=complex)
    hermitian = dagger(matrix) @ matrix
    values2, vectors = np.linalg.eigh(hermitian)
    order = np.argsort(np.maximum(values2.real, 0.0))
    vectors = vectors[:, order]
    diag = vectors.T @ matrix @ vectors
    phases = np.exp(-0.5j * np.angle(np.diag(diag)))
    N = vectors * phases
    masses = np.abs(np.diag(N.T @ matrix @ N)).real
    order = np.argsort(masses)
    return masses[order], N[:, order]


def so10_fermion_spectrum(
    h: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    params: SO10VEVParameters,
) -> SO10FermionSpectrum:
    """Derive masses, mixing matrices, and Yukawas from h, f, g and VEVs."""
    vevs = so10_vev_components(params)
    masses_m = so10_fermion_mass_matrices(h, f, g, params)
    mu, UR, UL = _diagonalize_dirac_mass_matrix(masses_m.MU)
    md, DR, DL = _diagonalize_dirac_mass_matrix(masses_m.MD)
    me, ER, EL = _diagonalize_dirac_mass_matrix(masses_m.ME)
    mnu, NL = _takagi_symmetric(masses_m.Mnu)
    mixing = SO10MixingMatrices(UL=UL, UR=UR, DL=DL, DR=DR, EL=EL, ER=ER, NL=NL)
    yukawas = {
        "yu": masses_m.MU / vevs.vu,
        "yd": masses_m.MD / vevs.vd,
        "ye": masses_m.ME / vevs.vd,
    }
    return SO10FermionSpectrum(
        masses={"up": mu, "down": md, "charged_lepton": me, "light_neutrino": mnu},
        mixing=mixing,
        yukawas=yukawas,
        mass_matrices=masses_m,
        vevs=vevs,
    )


def so10_triplet_coefficients(
    h: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    x: Mapping[str, complex] | None = None,
    y: Mapping[str, complex] | None = None,
) -> SO10TripletCoefficients:
    """Build C^L and C^R from the h, f, g tensors in the note.

    Missing x_i/y_i coefficients default to zero, except x0=1.
    Keys are strings like ``"x1"`` and ``"y10"``.
    """
    h = _as_3x3("h", h)
    f = _as_3x3("f", f)
    g = _as_3x3("g", g)
    x = dict(x or {})
    y = dict(y or {})

    def xv(name: str, default: complex = 0.0) -> complex:
        return complex(x.get(name, default))

    def yv(name: str, default: complex = 0.0) -> complex:
        return complex(y.get(name, default))

    x0 = xv("x0", 1.0)
    CL = np.zeros((3, 3, 3, 3), dtype=complex)
    CR = np.zeros((3, 3, 3, 3), dtype=complex)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    CL[i, j, k, l] = (
                        x0 * h[i, j] * h[k, l]
                        + xv("x1") * f[i, j] * f[k, l]
                        - xv("x3") * h[i, j] * f[k, l]
                        - xv("x4") * f[i, j] * h[k, l]
                        + yv("y5") * f[i, j] * g[k, l]
                        + yv("y7") * h[i, j] * g[k, l]
                        + yv("y9") * g[i, k] * f[j, l]
                        + yv("y10") * g[i, k] * g[j, l]
                    )
                    CR[i, j, k, l] = (
                        x0 * h[i, j] * h[k, l]
                        + xv("x1") * f[i, j] * f[k, l]
                        + xv("x2") * g[i, j] * g[k, l]
                        + xv("x3") * h[i, j] * f[k, l]
                        + xv("x4") * f[i, j] * h[k, l]
                        + xv("x5") * f[i, j] * g[k, l]
                        + xv("x6") * g[i, j] * f[k, l]
                        + xv("x7") * h[i, j] * g[k, l]
                        + xv("x8") * g[i, j] * h[k, l]
                        + xv("x9") * f[i, l] * g[j, k]
                        + xv("x10") * g[i, l] * g[j, k]
                    )
    return SO10TripletCoefficients(CL=CL, CR=CR)


def rotate_to_mass_basis(
    coeffs: SO10TripletCoefficients,
    mixing: SO10MixingMatrices,
) -> SO10MassBasisCoefficients:
    """Rotate flavor-basis SO(10) coefficients to mass-basis tensors.

    Uses the field conventions in the note: u=U_L u_m, d=D_L d_m,
    e=E_L e_m, nu=N_L nu_m and charge-conjugate fields rotate with the
    complex-conjugated right matrices.
    """
    UL = _as_3x3("UL", mixing.UL)
    UR = _as_3x3("UR", mixing.UR)
    DL = _as_3x3("DL", mixing.DL)
    DR = _as_3x3("DR", mixing.DR)
    EL = _as_3x3("EL", mixing.EL)
    ER = _as_3x3("ER", mixing.ER)
    NL = _as_3x3("NL", mixing.NL)
    CL = _as_3333("CL", coeffs.CL)
    CR = _as_3333("CR", coeffs.CR)

    CL_udue = np.einsum("ijkl,ia,jb,kc,ld->abcd", CL, UL, DL, UL, EL, optimize=True)
    CL_uddn = np.einsum("ijkl,ia,jb,kc,ld->abcd", CL, UL, DL, DL, NL, optimize=True)
    CR_udue = np.einsum(
        "ijkl,ia,jb,kc,ld->abcd",
        CR,
        UR.conj(),
        DR.conj(),
        UR.conj(),
        ER.conj(),
        optimize=True,
    )
    return SO10MassBasisCoefficients(CL_udue=CL_udue, CL_uddn=CL_uddn, CR_udue=CR_udue)


def antisym_u(tensor: np.ndarray, i: int, j: int, k: int, l: int) -> complex:
    """C_[ijk]l = C_ijkl - C_kjil, antisymmetrizing U_i <-> U_k."""
    return tensor[i, j, k, l] - tensor[k, j, i, l]


def antisym_d(tensor: np.ndarray, i: int, j: int, k: int, l: int) -> complex:
    """C_i[jk]l = C_ijkl - C_ikjl, antisymmetrizing D_j <-> D_k."""
    return tensor[i, j, k, l] - tensor[i, k, j, l]


def reduced_channel_coefficients(
    mass_basis: SO10MassBasisCoefficients,
    mixing: SO10MixingMatrices,
    yu: np.ndarray,
    yd: np.ndarray,
    ye: np.ndarray,
) -> SO10ReducedCoefficients:
    """Reduced C_W^M and C_h^M for the channels listed in section 2.

    Charged-lepton channels are returned for e and mu only.  Neutrino
    channels are returned as three-component arrays and are summed in the
    width function.
    """
    UL = _as_3x3("UL", mixing.UL)
    DL = _as_3x3("DL", mixing.DL)
    EL = _as_3x3("EL", mixing.EL)
    NL = _as_3x3("NL", mixing.NL)
    yu = _as_3x3("yu", yu)
    yd = _as_3x3("yd", yd)
    ye = _as_3x3("ye", ye)
    yud = dagger(yu)
    ydd = dagger(yd)
    yed = dagger(ye)
    CLu = mass_basis.CL_udue
    CLd = mass_basis.CL_uddn
    CRu = mass_basis.CR_udue

    Cw: dict[str, np.ndarray] = {}
    Ch: dict[str, np.ndarray] = {}

    def neutrino_coefficients(q: int) -> tuple[np.ndarray, np.ndarray]:
        cw = np.zeros(3, dtype=complex)
        ch = np.zeros(3, dtype=complex)
        for n in range(3):
            for l in range(3):
                for i in range(3):
                    cw[n] += 0.5 * antisym_u(CLu, i, q, 0, l) * DL[i, q] * NL[l, n]
                    ch[n] += antisym_u(CRu, i, q, 0, l) * yu[i, q] * ye[l, n]
                    if q == 1:
                        cw[n] += 0.5 * antisym_u(CLu, i, 0, 0, l) * DL[i, 1] * NL[l, n]
                        ch[n] += antisym_u(CRu, i, 0, 0, l) * yu[i, 1] * ye[l, n]
                for k in range(3):
                    for i in range(3):
                        cw[n] += -0.5 * antisym_d(CLd, i, q, k, l) * DL[i, q] * UL[k, 0]
                        ch[n] += -antisym_d(CLd, i, q, k, l) * ydd[i, q] * yud[k, 0]
                        ch[n] += antisym_d(CLd, i, q, k, l) * yud[i, 0] * ydd[k, q]
                        if q == 1:
                            cw[n] += -0.5 * antisym_d(CLd, i, 0, k, l) * DL[i, 1] * UL[k, 0]
                            ch[n] += -antisym_d(CLd, i, 0, k, l) * ydd[i, 1] * yud[k, 0]
                            ch[n] += antisym_d(CLd, i, 0, k, l) * yud[i, 0] * ydd[k, 1]
        return cw, ch

    def charged_coefficients(q: int, lep: int) -> tuple[complex, complex]:
        cw = 0.0j
        ch = 0.0j
        for j in range(3):
            for k in range(3):
                cw += 0.5 * antisym_u(CLu, 0, j, k, lep) * DL[k, q] * UL[j, 0]
                ch += antisym_u(CLu, 0, j, k, lep) * ydd[k, q] * yud[j, 0]
                ch += -antisym_u(CLu, 0, j, k, lep) * ydd[k, q] * yud[j, 0]
                ch += antisym_u(CRu, 0, j, k, lep) * yu[k, q] * yd[j, 0]
                if q == 0:
                    ch += -antisym_u(CRu, 0, j, k, lep) * yu[k, 0] * yd[j, 0]
                else:
                    ch += -antisym_u(CRu, 0, j, k, lep) * yu[k, 0] * yd[j, q]
        for l in range(3):
            for i in range(3):
                ch += -antisym_u(CLu, i, 0, q, l) * yud[i, 0] * yed[l, lep]
                if q == 0:
                    ch += -antisym_u(CRu, i, 0, 0, l) * yu[i, 0] * ye[l, lep]
                else:
                    ch += -antisym_u(CRu, i, 1, 0, l) * yu[i, 0] * ye[l, lep]
            for j in range(3):
                cw += -0.5 * antisym_d(CLd, 0, j, q, l) * UL[j, 0] * EL[l, lep]
                ch += -antisym_d(CLd, 0, j, q, l) * yud[j, 0] * yed[l, lep]
        return cw, ch

    Cw["p->pi+ nubar"], Ch["p->pi+ nubar"] = neutrino_coefficients(0)
    Cw["p->K+ nubar"], Ch["p->K+ nubar"] = neutrino_coefficients(1)

    for lep, name in enumerate(("e+", "mu+")):
        cw, ch = charged_coefficients(0, lep)
        Cw[f"p->pi0 {name}"] = np.array([cw], dtype=complex)
        Ch[f"p->pi0 {name}"] = np.array([ch], dtype=complex)
        cw, ch = charged_coefficients(1, lep)
        Cw[f"p->K0 {name}"] = np.array([cw], dtype=complex)
        Ch[f"p->K0 {name}"] = np.array([ch], dtype=complex)

    return SO10ReducedCoefficients(Cw=Cw, Ch=Ch)


def partial_widths(
    reduced: SO10ReducedCoefficients,
    M_T: float,
    M_wino: float,
    mu: float,
    m_sfermion: float,
    g2: float,
    A_L: float = 1.0,
    A_S: float = 1.0,
    hadronic: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Partial widths from the SO(10) width formula in section 2."""
    hadronic = hadronic_elements("susy_so10", DEFAULT_HADRONIC) if hadronic is None else dict(hadronic)
    iw = loop_I(M_wino, m_sfermion)
    ih = loop_I(mu, m_sfermion)
    common_loop = (1.0 / (16.0 * np.pi * np.pi)) * float(A_L) * float(A_S) / float(M_T)
    widths: dict[str, float] = {}

    for channel, c_w in reduced.Cw.items():
        c_h = reduced.Ch[channel]
        meson_mass = MESON_MASSES[channel]
        hme = float(hadronic[channel])
        phase = (PROTON_MASS * PROTON_MASS - meson_mass * meson_mass) ** 2
        phase /= 32.0 * np.pi * PROTON_MASS**3
        prefactor = phase * (common_loop * hme) ** 2
        widths[channel] = float(
            prefactor
            * sum(g2 * g2 * abs(iw * cw) ** 2 + abs(ih * ch) ** 2 for cw, ch in zip(c_w, c_h))
        )
    widths["total"] = sum(widths.values())
    return widths


def lifetimes_from_widths(widths: Mapping[str, float]) -> dict[str, float]:
    lifetimes = {}
    for channel, width in widths.items():
        lifetimes[channel] = float("inf") if width <= 0.0 else HBAR_GEV_SECONDS / (width * SECONDS_PER_YEAR)
    return lifetimes



def compute_susy_so10_from_model_parameters(
    h: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    x: Mapping[str, complex],
    y: Mapping[str, complex],
    vev_parameters: SO10VEVParameters,
    M_T: float = DEFAULT_M_T,
    M_wino: float = 2000.0,
    mu: float = 1000.0,
    m_sfermion: float = 3000.0,
    g2: float = 0.65,
    A_L: float = 1.0,
    A_S: float = 1.0,
) -> SO10Result:
    """High-level SO(10) path using h, f, g and the final mass equations."""
    fermions = so10_fermion_spectrum(h, f, g, vev_parameters)
    flavor = so10_triplet_coefficients(h, f, g, x=x, y=y)
    mass_basis = rotate_to_mass_basis(flavor, fermions.mixing)
    reduced = reduced_channel_coefficients(
        mass_basis,
        fermions.mixing,
        fermions.yukawas["yu"],
        fermions.yukawas["yd"],
        fermions.yukawas["ye"],
    )
    widths = partial_widths(reduced, M_T, M_wino, mu, m_sfermion, g2, A_L=A_L, A_S=A_S)
    lifetimes = lifetimes_from_widths(widths)
    total_width = float(widths["total"])
    branching_ratios = {
        channel: float(width) / total_width
        for channel, width in widths.items()
        if channel != "total" and total_width > 0.0
    }
    experimental_comparison, experimental_summary = compare_lifetimes_to_bounds(lifetimes)
    return SO10Result(
        flavor=flavor,
        mass_basis=mass_basis,
        reduced=reduced,
        widths=widths,
        lifetimes=lifetimes,
        branching_ratios=branching_ratios,
        experimental_comparison=experimental_comparison,
        experimental_summary=experimental_summary,
        fermion_spectrum=fermions,
    )


def compute_susy_so10_formula(
    h: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    x: Mapping[str, complex],
    y: Mapping[str, complex],
    mixing: SO10MixingMatrices,
    yu: np.ndarray,
    yd: np.ndarray,
    ye: np.ndarray,
    M_T: float = DEFAULT_M_T,
    M_wino: float = 2000.0,
    mu: float = 1000.0,
    m_sfermion: float = 3000.0,
    g2: float = 0.65,
    A_L: float = 1.0,
    A_S: float = 1.0,
) -> SO10Result:
    flavor = so10_triplet_coefficients(h, f, g, x=x, y=y)
    mass_basis = rotate_to_mass_basis(flavor, mixing)
    reduced = reduced_channel_coefficients(mass_basis, mixing, yu, yd, ye)
    widths = partial_widths(reduced, M_T, M_wino, mu, m_sfermion, g2, A_L=A_L, A_S=A_S)
    lifetimes = lifetimes_from_widths(widths)
    total_width = float(widths["total"])
    branching_ratios = {
        channel: float(width) / total_width
        for channel, width in widths.items()
        if channel != "total" and total_width > 0.0
    }
    experimental_comparison, experimental_summary = compare_lifetimes_to_bounds(lifetimes)
    return SO10Result(
        flavor=flavor,
        mass_basis=mass_basis,
        reduced=reduced,
        widths=widths,
        lifetimes=lifetimes,
        branching_ratios=branching_ratios,
        experimental_comparison=experimental_comparison,
        experimental_summary=experimental_summary,
    )
