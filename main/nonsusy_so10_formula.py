import numpy as np

from experimental_ratios import compare_lifetimes_to_bounds
from hadronic_inputs import hadronic_elements
from parameters import (
    DEFAULT_CKM_PARAMETERS,
    DEFAULT_PMNS_PARAMETERS,
    build_phased_mixing_matrix,
)

# -----------------------------
# Constants (GeV units assumed)
# -----------------------------
PROTON_MASS = 0.93827
MASS_K_PLUS = 0.49367
MASS_K_ZERO = 0.49761
MASS_PI_ZERO = 0.13497
MASS_PI_PLUS = 0.13957
MASS_ETA = 0.54786

# Hadronic matrix elements
HADRON_PI0_UDU = -0.131
HADRON_K0_USU = 0.103
HADRON_ETA_UDU = 0.006
HADRON_PI_PLUS_UDD = -0.186
HADRON_K_PLUS_USD = -0.049
HADRON_K_PLUS_UDS = -0.134

DEFAULT_HADRONIC = {
    "pi0_udu": HADRON_PI0_UDU,
    "K0_usu": HADRON_K0_USU,
    "eta_udu": HADRON_ETA_UDU,
    "piplus_udd": HADRON_PI_PLUS_UDD,
    "Kplus_usd": HADRON_K_PLUS_USD,
    "Kplus_uds": HADRON_K_PLUS_UDS,
}

# Renormalization factors
ALPHA_LONG = 1.2
ALPHA_SHORT = 2.0


# -----------------------------
# Helper functions
# -----------------------------
def dagger(matrix):
    return np.conjugate(matrix.T)




def _build_mixing_from_defaults(defaults, left_phases, right_phases, overrides=None):
    values = dict(defaults)
    if overrides:
        values.update(overrides)
    matrix = build_phased_mixing_matrix(
        values["theta12"],
        values["theta23"],
        values["theta13"],
        values["delta"],
        left_phases=left_phases,
        right_phases=right_phases,
    )
    return matrix, values


def compute_decay_rates_with_ckm_pmns(
    MGUT,
    gGUT,
    UL, UR,
    DR,
    EL, ER,
    ckm_left_phases=(0.0, 0.0, 0.0),
    ckm_right_phases=(0.0, 0.0),
    pmns_left_phases=(0.0, 0.0, 0.0),
    pmns_right_phases=(0.0, 0.0),
    ckm_parameters=None,
    pmns_parameters=None,
):
    """
    Compute non-SUSY proton decay with DL and NL derived from CKM/PMNS inputs.

    The flavor relations are

        DL = UL @ Uckm,    Uckm = Pckm Vckm Qckm,
        NL = EL @ Upmns,   Upmns = Ppmns Vpmns Qpmns.

    By default Vckm and Vpmns use the constants requested in parameters.py.
    ckm_parameters/pmns_parameters may override theta12, theta23, theta13,
    or delta; the P and Q phases are passed separately.
    """
    Uckm, ckm_values = _build_mixing_from_defaults(
        DEFAULT_CKM_PARAMETERS,
        ckm_left_phases,
        ckm_right_phases,
        overrides=ckm_parameters,
    )
    Upmns, pmns_values = _build_mixing_from_defaults(
        DEFAULT_PMNS_PARAMETERS,
        pmns_left_phases,
        pmns_right_phases,
        overrides=pmns_parameters,
    )
    DL = np.asarray(UL) @ Uckm
    NL = np.asarray(EL) @ Upmns

    result = compute_decay_rates(
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
    result["flavor_relations"] = {
        "DL": "UL @ Uckm",
        "NL": "EL @ Upmns",
        "ckm_parameters": {
            **ckm_values,
            "left_phases": tuple(float(x) for x in ckm_left_phases),
            "right_phases": tuple(float(x) for x in ckm_right_phases),
        },
        "pmns_parameters": {
            **pmns_values,
            "left_phases": tuple(float(x) for x in pmns_left_phases),
            "right_phases": tuple(float(x) for x in pmns_right_phases),
        },
    }
    result["derived_matrices"] = {"Uckm": Uckm, "Upmns": Upmns, "DL": DL, "NL": NL}
    return result


# -----------------------------
# Main computation function
# -----------------------------
def compute_decay_rates(
    MGUT,
    gGUT,
    UL, UR,
    DL, DR,
    EL, ER,
    NL,
):
    """
    Compute proton decay partial widths for various channels.
    
    Parameters:
        MGUT : float
        gGUT : float
        UL, UR, DL, DR, EL, ER, NL : (3x3 complex numpy arrays)
    
    Returns:
        dict of decay rates
    """

    hadronic = hadronic_elements("non_susy", DEFAULT_HADRONIC)

    # -----------------------------
    # Heavy gauge boson masses
    # -----------------------------
    M_X = MGUT
    M_X_prime = MGUT

    k1_sq = (gGUT / (np.sqrt(2) * M_X))**2
    k2_sq = (gGUT / (np.sqrt(2) * M_X_prime))**2

    # -----------------------------
    # Flavor structures
    # -----------------------------
    V1 = UR.T @ UL
    V2 = ER.T @ DL
    V3 = DR.T @ EL
    V4 = DR.T @ DL

    V_UD = dagger(UL) @ DL
    V_EN = dagger(EL) @ NL

    # -----------------------------
    # Wilson coefficients
    # -----------------------------
    def C20(a, b):
        return k1_sq * (
            V1[0, 0] * V2[a, b]
            + (V1 @ V_UD)[0, b] * (V2 @ dagger(V_UD))[a, 0]
        )

    def C21(a, b):
        return (
            k1_sq * (V1[0, 0] * V3[b, a])
            + k2_sq * (
                (V4 @ dagger(V_UD))[b, 0]
                * (V1 @ dagger(V_UD) @ dagger(V4) @ V3)[0, a]
            )
        )

    def C22(l, a, b):
        return (
            k1_sq * (V1 @ V_UD)[0, a] * (V3 @ V_EN)[b, l]
            + k2_sq * V4[b, a]
            * (V1 @ V_UD @ dagger(V4) @ V3 @ V_EN)[0, l]
        )

    # Normalize (c20, c21, etc.)
    def c20(a, b): return C20(a, b) / k1_sq
    def c21(a, b): return C21(a, b) / k1_sq
    def c22(l, a, b): return C22(l, a, b) / k1_sq

    # -----------------------------
    # Common prefactor function
    # -----------------------------
    def prefactor(meson_mass):
        return (
            (PROTON_MASS**2 - meson_mass**2)**2
            * gGUT**4 * ALPHA_LONG**2 * ALPHA_SHORT**2
            / (32 * np.pi * PROTON_MASS**3 * M_X**4)
        )

    # -----------------------------
    # Decay channels
    # -----------------------------
    gamma_pi0_e = prefactor(MASS_PI_ZERO) * hadronic["pi0_udu"]**2 * (
        abs(c21(0, 0))**2 + abs(c20(0, 0))**2
    )

    gamma_pi0_mu = prefactor(MASS_PI_ZERO) * hadronic["pi0_udu"]**2 * (
        abs(c21(1, 0))**2 + abs(c20(1, 0))**2
    )

    gamma_K0_e = prefactor(MASS_K_ZERO) * hadronic["K0_usu"]**2 * (
        abs(c21(0, 1))**2 + abs(c20(0, 1))**2
    )

    gamma_K0_mu = prefactor(MASS_K_ZERO) * hadronic["K0_usu"]**2 * (
        abs(c21(1, 1))**2 + abs(c20(1, 1))**2
    )

    gamma_eta_e = prefactor(MASS_ETA) * hadronic["eta_udu"]**2 * (
        abs(c21(0, 0))**2 + abs(c20(0, 0))**2
    )

    gamma_eta_mu = prefactor(MASS_ETA) * hadronic["eta_udu"]**2 * (
        abs(c21(1, 0))**2 + abs(c20(1, 0))**2
    )

    gamma_pi_nu = prefactor(MASS_PI_PLUS) * hadronic["piplus_udd"]**2 * sum(
        abs(c22(i, 0, 0))**2 for i in range(3)
    )

    gamma_K_nu = prefactor(MASS_K_PLUS) * sum(
        abs(
            hadronic["Kplus_usd"] * c22(i, 0, 1)
            + hadronic["Kplus_uds"] * c22(i, 1, 0)
        )**2
        for i in range(3)
    )


    # -----------------------------
    # Total width
    # -----------------------------
    total_width = (
        gamma_pi0_e
        + gamma_pi0_mu
        + gamma_K0_e
        + gamma_K0_mu
        + gamma_eta_e
        + gamma_eta_mu
        + gamma_pi_nu
        + gamma_K_nu
    )

    # -----------------------------
    # Branching ratios
    # -----------------------------
    branching_ratios = {
        "p->pi0 e+": gamma_pi0_e / total_width,
        "p->pi0 mu+": gamma_pi0_mu / total_width,
        "p->K0 e+": gamma_K0_e / total_width,
        "p->K0 mu+": gamma_K0_mu / total_width,
        "p->eta e+": gamma_eta_e / total_width,
        "p->eta mu+": gamma_eta_mu / total_width,
        "p->pi+ nu": gamma_pi_nu / total_width,
        "p->K+ nu": gamma_K_nu / total_width,
    }


    # -----------------------------
    # Lifetimes (in years)
    # -----------------------------
    HBAR_IN_GEV_S = 6.58e-25
    SECONDS_PER_YEAR = 365 * 24 * 3600

    def lifetime_from_width(gamma):
        if gamma <= 0:
            return np.inf
        return HBAR_IN_GEV_S / (gamma * SECONDS_PER_YEAR)


    lifetimes = {
        "p->pi0 e+": lifetime_from_width(gamma_pi0_e),
        "p->pi0 mu+": lifetime_from_width(gamma_pi0_mu),
        "p->K0 e+": lifetime_from_width(gamma_K0_e),
        "p->K0 mu+": lifetime_from_width(gamma_K0_mu),
        "p->eta e+": lifetime_from_width(gamma_eta_e),
        "p->eta mu+": lifetime_from_width(gamma_eta_mu),
        "p->pi+ nu": lifetime_from_width(gamma_pi_nu),
        "p->K+ nu": lifetime_from_width(gamma_K_nu),
    }
    experimental_comparison, experimental_summary = compare_lifetimes_to_bounds(lifetimes)
    
    # -----------------------------
    # Output
    # -----------------------------
    return {
        "parameters": {
            "model": "nonSUSY SO10",
            "matrix_convention": "UL, UR, DL, DR, EL, ER, NL used directly",
            "MGUT": MGUT,
            "gGUT": gGUT,
            "k1_sq": k1_sq,
            "k2_sq": k2_sq,
        },
        "widths": {
            "p->pi0 e+": gamma_pi0_e,
            "p->pi0 mu+": gamma_pi0_mu,
            "p->K0 e+": gamma_K0_e,
            "p->K0 mu+": gamma_K0_mu,
            "p->eta e+": gamma_eta_e,
            "p->eta mu+": gamma_eta_mu,
            "p->pi+ nu": gamma_pi_nu,
            "p->K+ nu": gamma_K_nu,
        },
        "total_width_GeV": total_width,
        "branching_ratios": branching_ratios,
        "lifetimes_years": lifetimes,
        "experimental_comparison": experimental_comparison,
        "experimental_summary": experimental_summary,
        "classification": experimental_summary["classification"],
        "wilson_coefficients": {
            "c20_00": c20(0, 0),
            "c21_00": c21(0, 0),
            "c20_10": c20(1, 0),
            "c21_10": c21(1, 0),
            "c20_01": c20(0, 1),
            "c21_01": c21(0, 1),
            "c20_11": c20(1, 1),
            "c21_11": c21(1, 1),
            "c22_i00": [c22(i, 0, 0) for i in range(3)],
            "c22_i01": [c22(i, 0, 1) for i in range(3)],
            "c22_i10": [c22(i, 1, 0) for i in range(3)], 
        },
    }
