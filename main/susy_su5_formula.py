"""SUSY SU(5) dimension-five proton-decay formulas.

This module transcribes the SUSY SU(5) section of ``formula.tex`` into
NumPy code.  The main entry point keeps the input list requested by the
user:

    m0, AX, tanbeta, M_T, fU, fD, fL

Those inputs determine the GUT-scale C5L/C5R tensors, the universal
GUT-scale scalar mass matrices, and the color-triplet normalization in the
TeX note.  The physical decay widths also need dressed four-fermion
coefficients, ``tilde C``; use ``partial_widths_from_tilde_coefficients``
for that final step when those quantities are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from experimental_ratios import compare_lifetimes_to_bounds
from hadronic_inputs import hadronic_elements

V_HIGGS = 174.0
S_W = np.sqrt(0.23122)
M_Z = 91.1876
DEFAULT_M_T = 1.0e17
PROTON_MASS = 0.93827
MASS_PI_ZERO = 0.13497
MASS_PI_PLUS = 0.13957
MASS_K_ZERO = 0.49761
MASS_K_PLUS = 0.49367
MASS_ETA = 0.54786

HBAR_GEV_SECONDS = 6.582119569e-25
SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


DEFAULT_HADRONIC_MATRIX_ELEMENTS = {
    # Values in GeV^2, copied from the SU(5) table in formula.tex.
    ("pi0", "ud", "u", "R", "L"): -0.131,
    ("pi0", "ud", "u", "L", "L"): 0.134,
    ("pi+", "ud", "d", "R", "L"): -0.186,
    ("pi+", "ud", "d", "L", "L"): 0.189,
    ("K0", "us", "u", "R", "L"): 0.103,
    ("K0", "us", "u", "L", "L"): 0.057,
    ("K+", "us", "d", "R", "L"): -0.049,
    ("K+", "us", "d", "L", "L"): 0.041,
    ("K+", "ud", "s", "R", "L"): -0.134,
    ("K+", "ud", "s", "L", "L"): 0.139,
    ("K+", "ds", "u", "R", "L"): -0.054,
    ("K+", "ds", "u", "L", "L"): -0.098,
    ("eta", "ud", "u", "R", "L"): 0.006,
    ("eta", "ud", "u", "L", "L"): 0.113,
}


def dagger(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=complex).conj().T


def _as_3x3(name: str, matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must be a 3x3 matrix, got shape {matrix.shape}")
    return matrix


def _identity3() -> np.ndarray:
    return np.eye(3, dtype=complex)


def _block_matrix(
    ll: np.ndarray,
    lr: np.ndarray,
    rl: np.ndarray,
    rr: np.ndarray,
) -> np.ndarray:
    return np.block([[ll, lr], [rl, rr]]).astype(complex)


def beta_functions(tanbeta: float) -> dict[str, float]:
    """Return s_beta, c_beta and c_2beta from tan(beta)."""
    tanbeta = float(tanbeta)
    if tanbeta <= 0.0:
        raise ValueError("tanbeta must be positive")
    c_beta = 1.0 / np.sqrt(1.0 + tanbeta * tanbeta)
    s_beta = tanbeta * c_beta
    c_2beta = c_beta * c_beta - s_beta * s_beta
    return {"s_beta": s_beta, "c_beta": c_beta, "c_2beta": c_2beta}


def loop_H(x: float, y: float) -> float:
    """H(x, y) from formula.tex.

    H(x,y) = 1/(x-y) * (x log(x)/(x-1) - y log(y)/(y-1)).
    The equal-mass limit is evaluated numerically from the derivative.
    """
    x = float(x)
    y = float(y)
    if x <= 0.0 or y <= 0.0:
        raise ValueError("loop_H expects positive mass-squared ratios")

    def f(z: float) -> float:
        if abs(z - 1.0) < 1.0e-10:
            return 1.0
        return z * np.log(z) / (z - 1.0)

    if abs(x - y) >= 1.0e-10:
        return float((f(x) - f(y)) / (x - y))

    if abs(x - 1.0) < 1.0e-8:
        return 0.5
    return float((x - 1.0 - np.log(x)) / ((x - 1.0) ** 2))


@dataclass(frozen=True)
class YukawaRotations:
    YU: np.ndarray
    YD: np.ndarray
    YL: np.ndarray
    UQ_u: np.ndarray
    UQ_d: np.ndarray
    UU: np.ndarray
    UD: np.ndarray
    UE: np.ndarray
    UL: np.ndarray
    P: np.ndarray
    VKM: np.ndarray
    VQU: np.ndarray
    VQE: np.ndarray
    VDL: np.ndarray


@dataclass(frozen=True)
class SfermionSpectrum:
    mass_matrices: dict[str, np.ndarray]
    mass_squared: dict[str, np.ndarray]
    rotations: dict[str, np.ndarray]
    fermion_masses: dict[str, np.ndarray]


@dataclass(frozen=True)
class SU5FormulaResult:
    M_T: float
    C5L: np.ndarray
    C5R: np.ndarray
    tilde_coefficients: dict[str, np.ndarray]
    yukawa_rotations: YukawaRotations
    sfermion_spectrum: SfermionSpectrum
    gaugino_higgsino_spectrum: GauginoHiggsinoSpectrum
    dressing_vertices: DressingVertices
    widths: dict[str, float]
    lifetimes: dict[str, float]
    branching_ratios: dict[str, float]
    experimental_comparison: dict
    experimental_summary: dict


def _diagonalize_yukawa(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagonalize f by the convention U_left^* f U_right^dagger = Y_diag."""
    u, values, vh = np.linalg.svd(matrix)
    order = np.argsort(values)
    values = values[order]
    u = u[:, order]
    vh = vh[order, :]
    y_diag = np.diag(values.astype(complex))
    u_left = u.T
    u_right = vh
    return y_diag, u_left, u_right


def yukawa_rotations(
    fU: np.ndarray,
    fD: np.ndarray,
    fL: np.ndarray,
    phi13: float = 0.0,
    phi23: float = 0.0,
    P_matrix: np.ndarray | None = None,
) -> YukawaRotations:
    """Build the flavor matrices in the SU(5) note.

    The TeX note defines:
      U_Q^(u)* f_U U_U^dagger = Y_U,
      U_Q^(d)* f_D U_D^dagger = Y_D,
      U_E* f_L U_L^dagger = Y_L,
      V_QU = U_Q^(u)dagger Pdagger U_U,
      V_QE = U_Q^(d)dagger U_E,
      V_DL = U_Ddagger U_L,
      V_KM = U_Q^(u) U_Q^(d)dagger.
    """
    fU = _as_3x3("fU", fU)
    fD = _as_3x3("fD", fD)
    fL = _as_3x3("fL", fL)

    YU, UQ_u, UU = _diagonalize_yukawa(fU)
    YD, UQ_d, UD = _diagonalize_yukawa(fD)
    YL, UE, UL = _diagonalize_yukawa(fL)

    if P_matrix is None:
        P = np.diag([np.exp(1j * float(phi13)), np.exp(1j * float(phi23)), 1.0])
    else:
        P = _as_3x3("P", P_matrix)

    VKM = UQ_u @ dagger(UQ_d)
    VQU = dagger(UQ_u) @ dagger(P) @ UU
    VQE = dagger(UQ_d) @ UE
    VDL = dagger(UD) @ UL

    return YukawaRotations(
        YU=YU,
        YD=YD,
        YL=YL,
        UQ_u=UQ_u,
        UQ_d=UQ_d,
        UU=UU,
        UD=UD,
        UE=UE,
        UL=UL,
        P=P,
        VKM=VKM,
        VQU=VQU,
        VQE=VQE,
        VDL=VDL,
    )


def gut_scale_c5_coefficients(
    fU: np.ndarray,
    fD: np.ndarray,
    rotations: YukawaRotations,
) -> tuple[np.ndarray, np.ndarray]:
    """C5L and C5R at M_X from Eq. (C5LRatMX) in formula.tex."""
    fU = _as_3x3("fU", fU)
    fD = _as_3x3("fD", fD)
    C5L = np.einsum(
        "im,mj,kn,nl->ijkl",
        fD,
        rotations.VDL,
        fU,
        dagger(rotations.VQU),
        optimize=True,
    )
    C5R = np.einsum(
        "mj,mi,nl,nk->ijkl",
        fD,
        rotations.VQU,
        fU,
        rotations.VQE,
        optimize=True,
    )
    return C5L, C5R


def sfermion_mass_matrices(
    m0: float,
    AX: float,
    tanbeta: float,
    fU: np.ndarray,
    fD: np.ndarray,
    fL: np.ndarray,
    mu: complex = 0.0,
) -> dict[str, np.ndarray]:
    """Universal GUT-scale sfermion mass matrices from formula.tex.

    The requested input list does not include the MSSM ``mu`` parameter,
    although it appears in the TeX LR/RL blocks.  The default keeps the
    requested API by setting mu=0; pass it explicitly if it is known.
    """
    fU = _as_3x3("fU", fU)
    fD = _as_3x3("fD", fD)
    fL = _as_3x3("fL", fL)

    m0 = float(m0)
    AX = float(AX)
    beta = beta_functions(tanbeta)
    sb = beta["s_beta"]
    cb = beta["c_beta"]
    c2b = beta["c_2beta"]
    eye = _identity3()

    mQ2 = m0 * m0 * eye
    mU2 = m0 * m0 * eye
    mD2 = m0 * m0 * eye
    mL2 = m0 * m0 * eye
    mE2 = m0 * m0 * eye

    AU = AX * m0 * fU
    AD = AX * m0 * fD
    AL = AX * m0 * fL

    up_ll = V_HIGGS**2 * sb**2 * (fU @ dagger(fU)) + mQ2
    up_ll += M_Z**2 * c2b * (0.5 - (2.0 / 3.0) * S_W**2) * eye
    up_rr = V_HIGGS**2 * sb**2 * (dagger(fU) @ fU) + mU2
    up_rr += M_Z**2 * c2b * ((2.0 / 3.0) * S_W**2) * eye
    up_lr = np.conjugate(mu) * fU * V_HIGGS * cb + AU * V_HIGGS * sb
    up_rl = dagger(up_lr)

    down_ll = V_HIGGS**2 * cb**2 * (fD @ dagger(fD)) + mQ2
    down_ll += M_Z**2 * c2b * (-0.5 + (1.0 / 3.0) * S_W**2) * eye
    down_rr = V_HIGGS**2 * cb**2 * (dagger(fD) @ fD) + mD2
    down_rr += M_Z**2 * c2b * (-(1.0 / 3.0) * S_W**2) * eye
    down_lr = np.conjugate(mu) * fD * V_HIGGS * sb + AD * V_HIGGS * cb
    down_rl = dagger(down_lr)

    slepton_ll = V_HIGGS**2 * cb**2 * (dagger(fL) @ fL) + mL2
    slepton_ll += M_Z**2 * c2b * (-0.5 + S_W**2) * eye
    slepton_rr = V_HIGGS**2 * cb**2 * (fL @ dagger(fL)) + mE2
    slepton_rr += M_Z**2 * c2b * (-S_W**2) * eye
    slepton_rl = np.conjugate(mu) * fL * V_HIGGS * sb + AL * V_HIGGS * cb
    slepton_lr = dagger(slepton_rl)

    sneutrino = mL2 + M_Z**2 * cb**2 * 0.5 * eye

    return {
        "up_squark": _block_matrix(up_ll, up_lr, up_rl, up_rr),
        "down_squark": _block_matrix(down_ll, down_lr, down_rl, down_rr),
        "charged_slepton": _block_matrix(
            slepton_ll,
            slepton_lr,
            slepton_rl,
            slepton_rr,
        ),
        "sneutrino": sneutrino,
    }


def _diagonalize_mass_matrix(
    matrix: np.ndarray,
    convention: str,
) -> tuple[np.ndarray, np.ndarray]:
    if convention == "U_M_T_Udagger":
        target = matrix.T
        values, vectors = np.linalg.eigh(target)
        rotation = dagger(vectors)
    elif convention == "Udagger_M_U":
        values, vectors = np.linalg.eigh(matrix)
        rotation = vectors
    else:
        raise ValueError(f"unknown diagonalization convention {convention!r}")

    order = np.argsort(values.real)
    return values[order].real, rotation[order] if convention == "U_M_T_Udagger" else rotation[:, order]


def sfermion_spectrum(
    m0: float,
    AX: float,
    tanbeta: float,
    fU: np.ndarray,
    fD: np.ndarray,
    fL: np.ndarray,
    mu: complex = 0.0,
) -> SfermionSpectrum:
    """Mass matrices, eigenvalues and rotations using formula.tex conventions."""
    matrices = sfermion_mass_matrices(m0, AX, tanbeta, fU, fD, fL, mu=mu)

    up_values, up_rotation = _diagonalize_mass_matrix(
        matrices["up_squark"],
        "U_M_T_Udagger",
    )
    down_values, down_rotation = _diagonalize_mass_matrix(
        matrices["down_squark"],
        "U_M_T_Udagger",
    )
    slepton_values, slepton_rotation = _diagonalize_mass_matrix(
        matrices["charged_slepton"],
        "Udagger_M_U",
    )
    sneutrino_values, sneutrino_rotation = _diagonalize_mass_matrix(
        matrices["sneutrino"],
        "Udagger_M_U",
    )

    beta = beta_functions(tanbeta)
    y_u = np.sort(np.linalg.svd(_as_3x3("fU", fU), compute_uv=False))
    y_d = np.sort(np.linalg.svd(_as_3x3("fD", fD), compute_uv=False))
    y_l = np.sort(np.linalg.svd(_as_3x3("fL", fL), compute_uv=False))

    return SfermionSpectrum(
        mass_matrices=matrices,
        mass_squared={
            "up_squark": up_values,
            "down_squark": down_values,
            "charged_slepton": slepton_values,
            "sneutrino": sneutrino_values,
        },
        rotations={
            "U_tilde_U": up_rotation,
            "U_tilde_D": down_rotation,
            "U_tilde_L": slepton_rotation,
            "U_tilde_N": sneutrino_rotation,
        },
        fermion_masses={
            "up": V_HIGGS * beta["s_beta"] * y_u,
            "down": V_HIGGS * beta["c_beta"] * y_d,
            "charged_lepton": V_HIGGS * beta["c_beta"] * y_l,
        },
    )


@dataclass(frozen=True)
class GauginoHiggsinoSpectrum:
    chargino_masses: np.ndarray
    neutralino_masses: np.ndarray
    U_plus: np.ndarray
    U_minus: np.ndarray
    U_neutralino: np.ndarray


@dataclass(frozen=True)
class DressingVertices:
    gluino: dict[str, np.ndarray]
    chargino: dict[str, np.ndarray]
    neutralino: dict[str, np.ndarray]


def gaugino_higgsino_spectrum(
    M1: float,
    M2: float,
    mu: float,
    tanbeta: float,
) -> GauginoHiggsinoSpectrum:
    """Chargino and neutralino masses/mixings from the MSSM matrices.

    The neutralino implementation assumes real M1, M2 and mu, so the real
    symmetric matrix can be diagonalized by an orthogonal matrix and then
    rephased to positive masses.
    """
    M1 = float(M1)
    M2 = float(M2)
    mu = float(mu)
    beta = beta_functions(tanbeta)
    sb = beta["s_beta"]
    cb = beta["c_beta"]
    cW = np.sqrt(1.0 - S_W * S_W)
    mW = M_Z * cW

    chargino_matrix = np.array(
        [
            [-M2, np.sqrt(2.0) * mW * sb],
            [-np.sqrt(2.0) * mW * cb, -mu],
        ],
        dtype=complex,
    )
    u_minus_svd, chargino_masses, vh_plus = np.linalg.svd(chargino_matrix)
    order = np.argsort(chargino_masses)
    chargino_masses = chargino_masses[order]
    u_minus = -u_minus_svd[:, order]
    u_plus = dagger(vh_plus)[..., order]

    neutralino_matrix = np.array(
        [
            [-M1, 0.0, -M_Z * S_W * cb, M_Z * S_W * sb],
            [0.0, -M2, M_Z * cW * cb, -M_Z * cW * sb],
            [-M_Z * S_W * cb, M_Z * cW * cb, 0.0, mu],
            [M_Z * S_W * sb, -M_Z * cW * sb, mu, 0.0],
        ],
        dtype=float,
    )
    neutralino_eigenvalues, neutralino_vectors = np.linalg.eigh(neutralino_matrix)
    order = np.argsort(np.abs(neutralino_eigenvalues))
    neutralino_eigenvalues = neutralino_eigenvalues[order]
    neutralino_vectors = neutralino_vectors[:, order].astype(complex)
    phases = np.where(neutralino_eigenvalues >= 0.0, 1.0 + 0.0j, 0.0 + 1.0j)
    neutralino_masses = np.abs(neutralino_eigenvalues)
    u_neutralino = neutralino_vectors * phases

    return GauginoHiggsinoSpectrum(
        chargino_masses=chargino_masses,
        neutralino_masses=neutralino_masses,
        U_plus=u_plus,
        U_minus=u_minus,
        U_neutralino=u_neutralino,
    )


def dressed_operator_tensors(
    C5L: np.ndarray,
    C5R: np.ndarray,
    rotations: YukawaRotations,
    spectrum: SfermionSpectrum,
) -> dict[str, np.ndarray]:
    """The C(tilde fields...) tensors immediately below Eq. H(x,y)."""
    Uu = spectrum.rotations["U_tilde_U"]
    Ud = spectrum.rotations["U_tilde_D"]
    Ul = spectrum.rotations["U_tilde_L"]
    Unu = spectrum.rotations["U_tilde_N"]
    Uu_dag = dagger(Uu)
    Ud_dag = dagger(Ud)
    V = rotations.VKM

    C_udul_L = np.zeros((6, 6, 3, 3), dtype=complex)
    C_udul_R = np.zeros((6, 6, 3, 3), dtype=complex)
    C_uddnu_L = np.zeros((6, 6, 3, 3), dtype=complex)
    C_ddunu_L = np.zeros((6, 6, 3, 3), dtype=complex)
    C_ulud_L = np.zeros((6, 6, 3, 3), dtype=complex)
    C_ulud_R = np.zeros((6, 6, 3, 3), dtype=complex)
    C_dnuud_L = np.zeros((6, 3, 3, 3), dtype=complex)
    C_unudd_L = np.zeros((6, 3, 3, 3), dtype=complex)

    C5Rc = np.conjugate(C5R)

    for M in range(6):
        for N in range(6):
            for i in range(3):
                for j in range(3):
                    total_l = 0.0j
                    total_r = 0.0j
                    total_uddnu = 0.0j
                    total_ddunu = 0.0j
                    for k in range(3):
                        for l in range(3):
                            total_l += (C5L[i, j, k, l] - C5L[k, j, i, l]) * Uu_dag[k, M] * Ud_dag[l, N]
                            total_r += (C5Rc[k, l, j, i] - C5Rc[i, l, j, k]) * Uu_dag[k + 3, M] * Ud_dag[l + 3, N]
                            total_ddunu += (C5L[l, j, i, k] - C5L[k, j, i, l]) * Ud_dag[k, M] * Ud_dag[l, N]
                            for m in range(3):
                                total_uddnu += (C5L[m, j, k, l] - C5L[l, j, k, m]) * Uu_dag[k, M] * Ud_dag[l, N] * V[m, i]
                    C_udul_L[M, N, i, j] = total_l
                    C_udul_R[M, N, i, j] = total_r
                    C_uddnu_L[M, N, i, j] = total_uddnu
                    C_ddunu_L[M, N, i, j] = total_ddunu

    for I in range(6):
        for J in range(6):
            for k in range(3):
                for l in range(3):
                    total_l = 0.0j
                    total_r = 0.0j
                    for i in range(3):
                        for j in range(3):
                            total_r += (C5Rc[k, l, j, i] - C5Rc[i, l, j, k]) * Uu_dag[i + 3, I] * Ul[j + 3, J]
                            for m in range(3):
                                total_l += (C5L[i, j, k, m] - C5L[k, j, i, m]) * Uu_dag[i, I] * Ul[j, J] * V[m, l]
                    C_ulud_L[I, J, k, l] = total_l
                    C_ulud_R[I, J, k, l] = total_r

    for I in range(6):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    total_dnu = 0.0j
                    total_unu = 0.0j
                    for i in range(3):
                        for n in range(3):
                            for m in range(3):
                                total_dnu += (C5L[i, n, k, m] - C5L[m, n, k, i]) * Ud_dag[i, I] * Unu[n, j] * V[m, l]
                            for p in range(3):
                                for q in range(3):
                                    total_unu += (C5L[q, n, i, p] - C5L[p, n, i, q]) * Uu_dag[i, I] * Unu[n, j] * V[p, k] * V[q, l]
                    C_dnuud_L[I, j, k, l] = total_dnu
                    C_unudd_L[I, j, k, l] = total_unu

    return {
        "u_d_ul_L": C_udul_L,
        "u_d_ul_R": C_udul_R,
        "u_d_dnu_L": C_uddnu_L,
        "d_d_unu_L": C_ddunu_L,
        "u_l_ud_L": C_ulud_L,
        "u_l_ud_R": C_ulud_R,
        "d_nu_ud_L": C_dnuud_L,
        "u_nu_dd_L": C_unudd_L,
    }


def dressing_vertices(
    rotations: YukawaRotations,
    spectrum: SfermionSpectrum,
    inos: GauginoHiggsinoSpectrum,
    tanbeta: float,
) -> DressingVertices:
    """Gluino, chargino and neutralino Gamma couplings from formula.tex."""
    beta = beta_functions(tanbeta)
    sb = beta["s_beta"]
    cb = beta["c_beta"]
    cW = np.sqrt(1.0 - S_W * S_W)
    mW = M_Z * cW
    tW = S_W / cW
    sqrt2 = np.sqrt(2.0)

    V = rotations.VKM
    Vdag = dagger(V)
    Uu = spectrum.rotations["U_tilde_U"]
    Ud = spectrum.rotations["U_tilde_D"]
    Ul = spectrum.rotations["U_tilde_L"]
    Unu = spectrum.rotations["U_tilde_N"]
    Ul_dag = dagger(Ul)
    Unu_dag = dagger(Unu)
    Up = inos.U_plus
    Um = inos.U_minus
    Up_dag = dagger(Up)
    Um_dag = dagger(Um)
    Nmix = inos.U_neutralino
    Ndag = dagger(Nmix)
    mu_mass = spectrum.fermion_masses["up"]
    md_mass = spectrum.fermion_masses["down"]
    ml_mass = spectrum.fermion_masses["charged_lepton"]

    GGLd = np.zeros((6, 3), dtype=complex)
    GGRd = np.zeros((6, 3), dtype=complex)
    GGLu = np.zeros((6, 3), dtype=complex)
    GGRu = np.zeros((6, 3), dtype=complex)
    for I in range(6):
        for j in range(3):
            GGLd[I, j] = sum(Ud[I, k] * V[k, j] for k in range(3))
            GGRd[I, j] = Ud[I, j + 3]
            GGLu[I, j] = Uu[I, j]
            GGRu[I, j] = Uu[I, j + 3]

    GCLd = np.zeros((6, 2, 3), dtype=complex)
    GCRd = np.zeros((6, 2, 3), dtype=complex)
    GCLu = np.zeros((6, 2, 3), dtype=complex)
    GCRu = np.zeros((6, 2, 3), dtype=complex)
    GCLl = np.zeros((3, 2, 3), dtype=complex)
    GCRl = np.zeros((3, 2, 3), dtype=complex)
    GCLnu = np.zeros((6, 2, 3), dtype=complex)

    for I in range(6):
        for a in range(2):
            for j in range(3):
                GCLd[I, a, j] = sum(
                    (Uu[I, k] * Up[0, a] + Uu[I, k + 3] * mu_mass[k] / (sqrt2 * mW * sb) * Up[1, a]) * V[k, j]
                    for k in range(3)
                )
                GCRd[I, a, j] = -sum(
                    Uu[I, k] * V[k, j] * md_mass[j] / (sqrt2 * mW * cb) * Um[1, a]
                    for k in range(3)
                )
                GCLu[I, a, j] = Ud[I, j] * Um_dag[a, 0] - sum(
                    Ud[I, k + 3] * md_mass[k] / (sqrt2 * mW * cb) * Vdag[k, j] * Um_dag[a, 1]
                    for k in range(3)
                )
                GCRu[I, a, j] = Ud[I, j] * mu_mass[j] / (sqrt2 * mW * sb) * Up_dag[a, 1]
                GCLnu[I, a, j] = -Ul_dag[I, j] * Um_dag[a, 0] + ml_mass[j] / (sqrt2 * mW * cb) * Ul_dag[I, j + 3] * Um_dag[a, 1]

    for i in range(3):
        for a in range(2):
            for j in range(3):
                GCLl[i, a, j] = -Unu_dag[i, j] * Up[0, a]
                GCRl[i, a, j] = ml_mass[j] / (sqrt2 * mW * cb) * Unu_dag[i, j] * Um[1, a]

    GNLd = np.zeros((6, 4, 3), dtype=complex)
    GNRd = np.zeros((6, 4, 3), dtype=complex)
    GNLu = np.zeros((6, 4, 3), dtype=complex)
    GNRu = np.zeros((6, 4, 3), dtype=complex)
    GNLl = np.zeros((6, 4, 3), dtype=complex)
    GNRl = np.zeros((6, 4, 3), dtype=complex)
    GNLnu = np.zeros((3, 4, 3), dtype=complex)

    for I in range(6):
        for a in range(4):
            for j in range(3):
                d_left = sum(Ud[I, k] * V[k, j] for k in range(3))
                GNLd[I, a, j] = sqrt2 * (0.5 * Nmix[1, a] - (1.0 / 6.0) * tW * Nmix[0, a]) * d_left
                GNLd[I, a, j] -= md_mass[j] / (sqrt2 * mW * cb) * Nmix[2, a] * Ud[I, j + 3]
                GNRd[I, a, j] = sqrt2 * (-(1.0 / 3.0) * tW * Ndag[a, 0]) * Ud[I, j + 3]
                GNRd[I, a, j] -= md_mass[j] / (sqrt2 * mW * cb) * Ndag[a, 2] * d_left
                GNLu[I, a, j] = sqrt2 * (-0.5 * Nmix[1, a] - (1.0 / 6.0) * tW * Nmix[0, a]) * Uu[I, j]
                GNLu[I, a, j] -= mu_mass[j] / (sqrt2 * mW * sb) * Nmix[3, a] * Uu[I, j + 3]
                GNRu[I, a, j] = sqrt2 * ((2.0 / 3.0) * tW * Ndag[a, 1]) * Uu[I, j + 3]
                GNRu[I, a, j] -= mu_mass[j] / (sqrt2 * mW * sb) * Ndag[a, 3] * Uu[I, j]
                GNLl[I, a, j] = sqrt2 * (0.5 * Nmix[1, a] + 0.5 * tW * Nmix[0, a]) * Ul_dag[I, j]
                GNLl[I, a, j] -= ml_mass[j] / (sqrt2 * mW * cb) * Nmix[2, a] * Ul_dag[I, j + 3]
                GNRl[I, a, j] = sqrt2 * (-tW * Ndag[a, 0]) * Ul_dag[I, j + 3]
                GNRl[I, a, j] -= ml_mass[j] / (sqrt2 * mW * cb) * Ndag[a, 2] * Ul_dag[I, j]

    for i in range(3):
        for a in range(4):
            for j in range(3):
                GNLnu[i, a, j] = sqrt2 * (-0.5 * Nmix[1, a] + 0.5 * tW * Nmix[0, a]) * Unu_dag[i, j]

    return DressingVertices(
        gluino={"GLd": GGLd, "GRd": GGRd, "GLu": GGLu, "GRu": GGRu},
        chargino={"CLd": GCLd, "CRd": GCRd, "CLu": GCLu, "CRu": GCRu, "CLl": GCLl, "CRl": GCRl, "CLnu": GCLnu},
        neutralino={"NLd": GNLd, "NRd": GNRd, "NLu": GNLu, "NRu": GNRu, "NLl": GNLl, "NRl": GNRl, "NLnu": GNLnu},
    )


def compute_tilde_coefficients(
    C5L: np.ndarray,
    C5R: np.ndarray,
    rotations: YukawaRotations,
    sfermions: SfermionSpectrum,
    inos: GauginoHiggsinoSpectrum,
    vertices: DressingVertices,
    g2: float,
    g3: float,
    M3: float,
) -> dict[str, np.ndarray]:
    """Compute the Eq. 14-20 dressed coefficients in formula.tex.

    The C^(6) terms appearing in two equations are not included here because
    they are not dimension-five triplet dressing terms.
    """
    g2 = float(g2)
    g3 = float(g3)
    M3 = float(M3)
    if M3 <= 0.0:
        raise ValueError("M3 must be positive")

    C = dressed_operator_tensors(C5L, C5R, rotations, sfermions)
    G = vertices.gluino
    Ch = vertices.chargino
    Ne = vertices.neutralino
    m2u = sfermions.mass_squared["up_squark"]
    m2d = sfermions.mass_squared["down_squark"]
    m2l = sfermions.mass_squared["charged_slepton"]
    m2nu = sfermions.mass_squared["sneutrino"]
    MC = inos.chargino_masses
    MN = inos.neutralino_masses

    CLL_udul = np.zeros((3, 3), dtype=complex)
    CRL_udul = np.zeros((3, 3), dtype=complex)
    CLR_udul = np.zeros((3, 3), dtype=complex)
    CRR_udul = np.zeros((3, 3), dtype=complex)
    CLL_uddnu = np.zeros((3, 3, 3), dtype=complex)
    CRL_uddnu = np.zeros((3, 3, 3), dtype=complex)
    CRL_ddunu = np.zeros((3, 3, 3), dtype=complex)

    for i in range(3):
        for k in range(3):
            for M in range(6):
                for N in range(6):
                    h_g = loop_H(m2u[M] / M3**2, m2d[N] / M3**2)
                    CLL_udul[i, k] += (4.0 / 3.0) * g3**2 / M3 * C["u_d_ul_L"][M, N, 0, k] * G["GLu"][M, 0] * G["GLd"][N, i] * h_g
                    CRL_udul[i, k] += (4.0 / 3.0) * g3**2 / M3 * C["u_d_ul_L"][M, N, 0, k] * G["GRu"][M, 0] * G["GRd"][N, i] * h_g
                    CLR_udul[i, k] += (4.0 / 3.0) * g3**2 / M3 * C["u_d_ul_R"][M, N, 0, k] * G["GLu"][M, 0] * G["GLd"][N, i] * h_g
                    CRR_udul[i, k] += (4.0 / 3.0) * g3**2 / M3 * C["u_d_ul_R"][M, N, 0, k] * G["GRu"][M, 0] * G["GRd"][N, i] * h_g
                    for a in range(2):
                        h_c = loop_H(m2u[M] / MC[a]**2, m2d[N] / MC[a]**2)
                        CLL_udul[i, k] += g2**2 / MC[a] * (-C["u_d_ul_L"][M, N, 0, k] * Ch["CLu"][N, a, 0] * Ch["CLd"][M, a, i] * h_c)
                        CRL_udul[i, k] += -g2**2 / MC[a] * C["u_d_ul_L"][M, N, 0, k] * Ch["CRu"][N, a, 0] * Ch["CRd"][M, a, i] * h_c
                        CLR_udul[i, k] += g2**2 / MC[a] * (-C["u_d_ul_R"][M, N, 0, k] * Ch["CLu"][N, a, 0] * Ch["CLd"][M, a, i] * h_c)
                        CRR_udul[i, k] += -g2**2 / MC[a] * C["u_d_ul_R"][M, N, 0, k] * Ch["CRu"][N, a, 0] * Ch["CRd"][M, a, i] * h_c
                    for a in range(4):
                        h_n = loop_H(m2u[M] / MN[a]**2, m2d[N] / MN[a]**2)
                        CLL_udul[i, k] += g2**2 / MN[a] * C["u_d_ul_L"][M, N, 0, k] * Ne["NLu"][M, a, 0] * Ne["NLd"][N, a, i] * h_n
                        CRL_udul[i, k] += g2**2 / MN[a] * C["u_d_ul_L"][M, N, 0, k] * Ne["NRu"][M, a, 0] * Ne["NRd"][N, a, i] * h_n
                        CLR_udul[i, k] += g2**2 / MN[a] * C["u_d_ul_R"][M, N, 0, k] * Ne["NLu"][M, a, 0] * Ne["NLd"][N, a, i] * h_n
                        CRR_udul[i, k] += g2**2 / MN[a] * C["u_d_ul_R"][M, N, 0, k] * Ne["NRu"][M, a, 0] * Ne["NRd"][N, a, i] * h_n

            for N in range(6):
                for m in range(3):
                    for a in range(2):
                        h = loop_H(m2d[N] / MC[a]**2, m2nu[m] / MC[a]**2)
                        CLL_udul[i, k] += g2**2 / MC[a] * C["d_nu_ud_L"][N, m, 0, i] * Ch["CLu"][N, a, 0] * Ch["CLl"][m, a, k] * h
                        CLR_udul[i, k] += g2**2 / MC[a] * C["d_nu_ud_L"][N, m, 0, i] * Ch["CRu"][N, a, 0] * Ch["CRl"][m, a, k] * h

            for M in range(6):
                for N in range(6):
                    for a in range(4):
                        h = loop_H(m2u[M] / MN[a]**2, m2l[N] / MN[a]**2)
                        CLL_udul[i, k] += g2**2 / MN[a] * C["u_l_ud_L"][M, N, 0, i] * Ne["NLu"][M, a, 0] * Ne["NLl"][N, a, k] * h
                        CRL_udul[i, k] += g2**2 / MN[a] * C["u_l_ud_R"][M, N, 0, i] * Ne["NLu"][M, a, 0] * Ne["NLl"][N, a, k] * h
                        CLR_udul[i, k] += g2**2 / MN[a] * C["u_l_ud_L"][M, N, 0, i] * Ne["NRu"][M, a, 0] * Ne["NRl"][N, a, k] * h
                        CRR_udul[i, k] += g2**2 / MN[a] * C["u_l_ud_R"][M, N, 0, i] * Ne["NRu"][M, a, 0] * Ne["NRl"][N, a, k] * h

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for M in range(6):
                    for N in range(6):
                        h_g_ud = loop_H(m2u[M] / M3**2, m2d[N] / M3**2)
                        h_g_dd = loop_H(m2d[M] / M3**2, m2d[N] / M3**2)
                        CLL_uddnu[i, j, k] += (4.0 / 3.0) * g3**2 / M3 * (
                            C["u_d_dnu_L"][M, N, j, k] * G["GLu"][M, 0] * G["GLd"][N, i] * h_g_ud
                            + C["d_d_unu_L"][M, N, 0, k] * G["GLd"][M, j] * G["GLd"][N, i] * h_g_dd
                        )
                        CRL_uddnu[i, j, k] += (4.0 / 3.0) * g3**2 / M3 * C["u_d_dnu_L"][M, N, j, k] * G["GRu"][M, 0] * G["GRd"][N, i] * h_g_ud
                        CRL_ddunu[i, j, k] += (4.0 / 3.0) * g3**2 / M3 * C["d_d_unu_L"][M, N, 0, k] * G["GRd"][M, i] * G["GRd"][N, j] * h_g_dd
                        for a in range(2):
                            h_c = loop_H(m2u[M] / MC[a]**2, m2d[N] / MC[a]**2)
                            CLL_uddnu[i, j, k] += g2**2 / MC[a] * (-C["u_d_dnu_L"][M, N, j, k] * Ch["CLu"][N, a, 0] * Ch["CLd"][M, a, i] * h_c)
                            CRL_uddnu[i, j, k] += g2**2 / MC[a] * (-C["u_d_dnu_L"][M, N, j, k] * Ch["CRu"][N, a, 0] * Ch["CRd"][M, a, i] * h_c)
                        for a in range(4):
                            h_n_ud = loop_H(m2u[M] / MN[a]**2, m2d[N] / MN[a]**2)
                            h_n_dd = loop_H(m2d[M] / MN[a]**2, m2d[N] / MN[a]**2)
                            CLL_uddnu[i, j, k] += g2**2 / MN[a] * (
                                C["u_d_dnu_L"][M, N, j, k] * Ne["NLu"][M, a, 0] * Ne["NLd"][N, a, i] * h_n_ud
                                + C["d_d_unu_L"][M, N, 0, k] * Ne["NLd"][M, a, j] * Ne["NLd"][N, a, i] * h_n_dd
                            )
                            CRL_uddnu[i, j, k] += g2**2 / MN[a] * C["u_d_dnu_L"][M, N, j, k] * Ne["NRu"][M, a, 0] * Ne["NRd"][N, a, i] * h_n_ud
                            CRL_ddunu[i, j, k] += g2**2 / MN[a] * C["d_d_unu_L"][M, N, 0, k] * Ne["NRd"][M, a, i] * Ne["NRd"][N, a, j] * h_n_dd

                    for a in range(2):
                        for M in range(6):
                            for N in range(6):
                                h = loop_H(m2u[M] / MC[a]**2, m2l[N] / MC[a]**2)
                                CLL_uddnu[i, j, k] += g2**2 / MC[a] * C["u_l_ud_L"][M, N, 0, i] * Ch["CLd"][M, a, j] * Ch["CLnu"][N, a, k] * h
                                CRL_uddnu[i, j, k] += g2**2 / MC[a] * C["u_l_ud_R"][M, N, 0, i] * Ch["CLd"][M, a, j] * Ch["CLnu"][N, a, k] * h
                        for M in range(6):
                            for n in range(3):
                                h = loop_H(m2d[M] / MN[a % 4]**2, m2nu[n] / MN[a % 4]**2)
                    for a in range(4):
                        for M in range(6):
                            for n in range(3):
                                h = loop_H(m2d[M] / MN[a]**2, m2nu[n] / MN[a]**2)
                                CLL_uddnu[i, j, k] += g2**2 / MN[a] * C["d_nu_ud_L"][M, n, 0, i] * Ne["NLd"][M, a, j] * Ne["NLnu"][n, a, k] * h
                                h = loop_H(m2u[M] / MN[a]**2, m2nu[n] / MN[a]**2)
                                CLL_uddnu[i, j, k] += g2**2 / MN[a] * C["u_nu_dd_L"][M, n, j, i] * Ne["NLu"][M, a, 0] * Ne["NLnu"][n, a, k] * h

    return {
        "CLL_udul": CLL_udul,
        "CRL_udul": CRL_udul,
        "CLR_udul": CLR_udul,
        "CRR_udul": CRR_udul,
        "CLL_uddnu": CLL_uddnu,
        "CRL_uddnu": CRL_uddnu,
        "CRL_ddunu": CRL_ddunu,
    }


def compute_susy_su5_formula(
    m0: float,
    AX: float,
    tanbeta: float,
    M_T: float,
    mu: float,
    M1: float,
    M2: float,
    M3: float,
    g2: float,
    g3: float,
    fU: np.ndarray,
    fD: np.ndarray,
    fL: np.ndarray,
    P: np.ndarray | None = None,
) -> SU5FormulaResult:
    """Compute all SU(5) quantities fixed by the requested inputs."""
    M_T = float(M_T)
    if M_T <= 0.0:
        raise ValueError("M_T must be positive")
    fU = _as_3x3("fU", fU)
    fD = _as_3x3("fD", fD)
    fL = _as_3x3("fL", fL)
    rotations = yukawa_rotations(fU, fD, fL, P_matrix=P)
    C5L, C5R = gut_scale_c5_coefficients(fU, fD, rotations)
    spectrum = sfermion_spectrum(m0, AX, tanbeta, fU, fD, fL, mu=mu)
    inos = gaugino_higgsino_spectrum(M1, M2, mu, tanbeta)
    vertices = dressing_vertices(rotations, spectrum, inos, tanbeta)
    tilde_coefficients = compute_tilde_coefficients(
        C5L,
        C5R,
        rotations,
        spectrum,
        inos,
        vertices,
        g2,
        g3,
        M3,
    )
    widths = partial_widths_from_tilde_coefficients(M_T, tilde_coefficients)
    lifetimes = lifetimes_from_widths(widths)
    total_width = float(widths["total"])
    branching_ratios = {
        channel: float(width) / total_width
        for channel, width in widths.items()
        if channel != "total" and total_width > 0.0
    }
    experimental_comparison, experimental_summary = compare_lifetimes_to_bounds(lifetimes)
    return SU5FormulaResult(
        M_T=M_T,
        C5L=C5L,
        C5R=C5R,
        tilde_coefficients=tilde_coefficients,
        yukawa_rotations=rotations,
        sfermion_spectrum=spectrum,
        gaugino_higgsino_spectrum=inos,
        dressing_vertices=vertices,
        widths=widths,
        lifetimes=lifetimes,
        branching_ratios=branching_ratios,
        experimental_comparison=experimental_comparison,
        experimental_summary=experimental_summary,
    )


def _zeros(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=complex)


def _array_from_mapping(
    data: Mapping[str, np.ndarray],
    key: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if key not in data:
        return _zeros(shape)
    arr = np.asarray(data[key], dtype=complex)
    if arr.shape != shape:
        raise ValueError(f"{key} must have shape {shape}, got {arr.shape}")
    return arr


def _hadron(
    elements: Mapping[tuple[str, str, str, str, str], float],
    meson: str,
    diquark: str,
    spectator: str,
    diquark_chirality: str,
    spectator_chirality: str,
) -> float:
    key = (meson, diquark, spectator, diquark_chirality, spectator_chirality)
    if key in elements:
        return float(elements[key])

    # Explicit chirality-swap symmetry in formula.tex.  It is written below
    # the pi0 channel and used here for the same operator structure in the
    # charged-lepton amplitudes.
    swapped = (meson, diquark, spectator, spectator_chirality, diquark_chirality)
    if swapped in elements:
        return float(elements[swapped])

    # The note only tabulates mixed R/L and L/L entries.  For all-R charged
    # lepton amplitudes, use the corresponding L/L entry unless the caller
    # supplies a more specific hadronic element.
    parity_partner = (meson, diquark, spectator, "L", "L")
    if diquark_chirality == "R" and spectator_chirality == "R" and parity_partner in elements:
        return float(elements[parity_partner])

    raise KeyError(f"missing hadronic matrix element {key}")


def _partial_width(M_T: float, meson_mass: float, AL: complex, AR: complex = 0.0) -> float:
    prefactor = PROTON_MASS / (32.0 * np.pi)
    phase_space = (1.0 - (meson_mass * meson_mass) / (PROTON_MASS * PROTON_MASS)) ** 2
    triplet_loop = (1.0 / (float(M_T) * 16.0 * np.pi * np.pi)) ** 2
    return float(prefactor * phase_space * (abs(AL) ** 2 + abs(AR) ** 2) * triplet_loop)


def partial_widths_from_tilde_coefficients(
    M_T: float,
    tilde_coefficients: Mapping[str, np.ndarray],
    hadronic_matrix_elements: Mapping[tuple[str, str, str, str, str], float] | None = None,
) -> dict[str, float]:
    """Partial widths for the SU(5) channels in formula.tex.

    ``tilde_coefficients`` accepts these arrays with zero-based generation
    indices:

      CLL_udul, CRL_udul, CLR_udul, CRR_udul: shape (3, 3), indices i,k
      CLL_uddnu, CRL_uddnu: shape (3, 3, 3), indices i,j,k
      CRL_ddunu: shape (3, 3, 3), indices i,j,k
    """
    h = hadronic_matrix_elements or hadronic_elements("susy_su5", DEFAULT_HADRONIC_MATRIX_ELEMENTS)

    CLL_udul = _array_from_mapping(tilde_coefficients, "CLL_udul", (3, 3))
    CRL_udul = _array_from_mapping(tilde_coefficients, "CRL_udul", (3, 3))
    CLR_udul = _array_from_mapping(tilde_coefficients, "CLR_udul", (3, 3))
    CRR_udul = _array_from_mapping(tilde_coefficients, "CRR_udul", (3, 3))
    CLL_uddnu = _array_from_mapping(tilde_coefficients, "CLL_uddnu", (3, 3, 3))
    CRL_uddnu = _array_from_mapping(tilde_coefficients, "CRL_uddnu", (3, 3, 3))
    CRL_ddunu = _array_from_mapping(tilde_coefficients, "CRL_ddunu", (3, 3, 3))

    charged_labels = ("e+", "mu+")
    widths: dict[str, float] = {}

    for k, label in enumerate(charged_labels):
        AL = (
            _hadron(h, "pi0", "ud", "u", "R", "L") * CRL_udul[0, k]
            + _hadron(h, "pi0", "ud", "u", "L", "L") * CLL_udul[0, k]
        )
        AR = -(
            _hadron(h, "pi0", "ud", "u", "L", "R") * CLR_udul[0, k]
            + _hadron(h, "pi0", "ud", "u", "R", "R") * CRR_udul[0, k]
        )
        widths[f"p->pi0 {label}"] = _partial_width(M_T, MASS_PI_ZERO, AL, AR)

        AL = (
            _hadron(h, "eta", "ud", "u", "R", "L") * CRL_udul[0, k]
            + _hadron(h, "eta", "ud", "u", "L", "L") * CLL_udul[0, k]
        )
        AR = -(
            _hadron(h, "eta", "ud", "u", "L", "R") * CLR_udul[0, k]
            + _hadron(h, "eta", "ud", "u", "R", "R") * CRR_udul[0, k]
        )
        widths[f"p->eta {label}"] = _partial_width(M_T, MASS_ETA, AL, AR)

        AL = (
            _hadron(h, "K0", "us", "u", "R", "L") * CRL_udul[1, k]
            + _hadron(h, "K0", "us", "u", "L", "L") * CLL_udul[1, k]
        )
        AR = -(
            _hadron(h, "K0", "us", "u", "L", "R") * CLR_udul[1, k]
            + _hadron(h, "K0", "us", "u", "R", "R") * CRR_udul[1, k]
        )
        widths[f"p->K0 {label}"] = _partial_width(M_T, MASS_K_ZERO, AL, AR)

    pi_plus_nubar = 0.0
    k_plus_nubar = 0.0
    for k in range(3):
        AL = (
            _hadron(h, "pi+", "ud", "d", "R", "L") * CRL_uddnu[0, 0, k]
            + _hadron(h, "pi+", "ud", "d", "L", "L") * CLL_uddnu[0, 0, k]
        )
        pi_plus_nubar += _partial_width(M_T, MASS_PI_PLUS, AL)

        AL = (
            _hadron(h, "K+", "ds", "u", "R", "L") * CRL_ddunu[0, 1, k]
            + _hadron(h, "K+", "ud", "s", "R", "L") * CRL_uddnu[0, 1, k]
            + _hadron(h, "K+", "us", "d", "R", "L") * CRL_uddnu[1, 0, k]
            + _hadron(h, "K+", "ud", "s", "L", "L") * CLL_uddnu[0, 1, k]
            + _hadron(h, "K+", "us", "d", "L", "L") * CLL_uddnu[1, 0, k]
        )
        k_plus_nubar += _partial_width(M_T, MASS_K_PLUS, AL)

    widths["p->pi+ nubar"] = pi_plus_nubar
    widths["p->K+ nubar"] = k_plus_nubar
    widths["total"] = sum(value for key, value in widths.items() if key != "total")
    return widths


def lifetimes_from_widths(widths: Mapping[str, float]) -> dict[str, float]:
    """Convert widths in GeV to lifetimes in years."""
    lifetimes: dict[str, float] = {}
    for channel, width in widths.items():
        if width <= 0.0:
            lifetimes[channel] = float("inf")
        else:
            lifetimes[channel] = HBAR_GEV_SECONDS / (width * SECONDS_PER_YEAR)
    return lifetimes
