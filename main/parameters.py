import numpy as np
# ============================================================
# DEFAULT RANGES
# ============================================================
DEFAULT_RANGES = {
    "MGUT": (1e14, 1e17),
    "gGUT": (0.4, 0.8),

    "angles": (0, np.pi / 2),
    "phases": (0, 2 * np.pi),
}


MATRIX_LIST = ["UL", "UR", "DL", "DR", "EL", "ER", "NL"]
INDEPENDENT_CKM_PMNS_MATRIX_LIST = ["UL", "UR", "DR", "EL", "ER"]

DEFAULT_CKM_PARAMETERS = {
    "theta12": 0.2270,
    "theta23": 0.04194,
    "theta13": 0.00370,
    "delta": 1.139,
}

DEFAULT_PMNS_PARAMETERS = {
    "theta12": 0.587827,
    "theta23": 0.846485,
    "theta13": 0.148702,
    "delta": 3.08923,
}


# ============================================================
# UNITARY GENERATION
# ============================================================

def ckm_like(theta12, theta23, theta13, delta):
    s12, c12 = np.sin(theta12), np.cos(theta12)
    s13, c13 = np.sin(theta13), np.cos(theta13)
    s23, c23 = np.sin(theta23), np.cos(theta23)

    return np.array([
        [c12*c13, s12*c13, s13*np.exp(-1j*delta)],
        [-s12*c23 - c12*s23*s13*np.exp(1j*delta),
         c12*c23 - s12*s23*s13*np.exp(1j*delta),
         s23*c13],
        [s12*s23 - c12*c23*s13*np.exp(1j*delta),
         -c12*s23 - s12*c23*s13*np.exp(1j*delta),
         c23*c13]
    ])


def _phase_diagonal(phases, *, trailing_one=False):
    values = [np.exp(1j * float(phase)) for phase in phases]
    if trailing_one:
        values.append(1.0)
    return np.diag(values)


def build_phased_mixing_matrix(
    theta12,
    theta23,
    theta13,
    delta,
    left_phases=(0.0, 0.0, 0.0),
    right_phases=(0.0, 0.0),
):
    """
    Build U = P V Q, where V is the CKM-like three-angle matrix,
    P has three diagonal phases, and Q has two phases plus a fixed 1.
    """
    if len(left_phases) != 3:
        raise ValueError("left_phases must contain exactly 3 phases")
    if len(right_phases) != 2:
        raise ValueError("right_phases must contain exactly 2 phases")

    P = _phase_diagonal(left_phases)
    V = ckm_like(theta12, theta23, theta13, delta)
    Q = _phase_diagonal(right_phases, trailing_one=True)
    return P @ V @ Q


def build_ckm_matrix(left_phases=(0.0, 0.0, 0.0), right_phases=(0.0, 0.0)):
    return build_phased_mixing_matrix(
        **DEFAULT_CKM_PARAMETERS,
        left_phases=left_phases,
        right_phases=right_phases,
    )


def build_pmns_matrix(left_phases=(0.0, 0.0, 0.0), right_phases=(0.0, 0.0)):
    return build_phased_mixing_matrix(
        **DEFAULT_PMNS_PARAMETERS,
        left_phases=left_phases,
        right_phases=right_phases,
    )


def build_unitary(cfg, return_parameters=False):

    angle_default = cfg.get("angles", DEFAULT_RANGES["angles"])
    phase_default = cfg.get("phases", DEFAULT_RANGES["phases"])

    ang_12 = cfg.get("theta12", angle_default)
    ang_23 = cfg.get("theta23", angle_default)
    ang_13 = cfg.get("theta13", angle_default)
    ph_del = cfg.get("delta", phase_default)

    ph_phi1 = cfg.get("phi1", phase_default)
    ph_phi2 = cfg.get("phi2", phase_default)
    ph_phi3 = cfg.get("phi3", phase_default)
    ph_phi4 = cfg.get("phi4", phase_default)
    ph_phi5 = cfg.get("phi5", phase_default)

    sampled = {
        "theta12": float(np.random.uniform(*ang_12)),
        "theta23": float(np.random.uniform(*ang_23)),
        "theta13": float(np.random.uniform(*ang_13)),
        "delta": float(np.random.uniform(*ph_del)),
        "phi1": float(np.random.uniform(*ph_phi1)),
        "phi2": float(np.random.uniform(*ph_phi2)),
        "phi3": float(np.random.uniform(*ph_phi3)),
        "phi4": float(np.random.uniform(*ph_phi4)),
        "phi5": float(np.random.uniform(*ph_phi5)),
    }

    # ============================
    # D1: left diagonal phases
    # ============================
    D1 = np.diag([
        np.exp(1j * sampled["phi1"]),
        np.exp(1j * sampled["phi2"]),
        np.exp(1j * sampled["phi3"]),
    ])

    # ============================
    # CKM-like matrix
    # ============================
    V = ckm_like(
        sampled["theta12"],
        sampled["theta23"],
        sampled["theta13"],
        sampled["delta"],
    )

    # ============================
    # D2: right diagonal phases
    # ============================
    D2 = np.diag([
        np.exp(1j * sampled["phi4"]),
        np.exp(1j * sampled["phi5"]),
        1.0
    ])

    matrix = D1 @ V @ D2
    if return_parameters:
        return matrix, sampled
    return matrix


def _fixed_or_sampled_parameter(cfg, name, default_value=None, range_key=None):
    if name in cfg:
        value = cfg[name]
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return float(np.random.uniform(*value))
        return float(value)
    if default_value is not None:
        return float(default_value)
    low, high = cfg.get(range_key or name, DEFAULT_RANGES["phases"])
    return float(np.random.uniform(low, high))


def build_fixed_angle_mixing(cfg, defaults, return_parameters=False):
    """
    Build a phased CKM/PMNS-style matrix with fixed default angles and delta.
    The five external phases are sampled from cfg["phases"] unless fixed by
    phi1...phi5 or by per-parameter ranges.
    """
    cfg = cfg or {}
    sampled = {
        "theta12": _fixed_or_sampled_parameter(cfg, "theta12", defaults["theta12"]),
        "theta23": _fixed_or_sampled_parameter(cfg, "theta23", defaults["theta23"]),
        "theta13": _fixed_or_sampled_parameter(cfg, "theta13", defaults["theta13"]),
        "delta": _fixed_or_sampled_parameter(cfg, "delta", defaults["delta"]),
        "phi1": _fixed_or_sampled_parameter(cfg, "phi1", range_key="phases"),
        "phi2": _fixed_or_sampled_parameter(cfg, "phi2", range_key="phases"),
        "phi3": _fixed_or_sampled_parameter(cfg, "phi3", range_key="phases"),
        "phi4": _fixed_or_sampled_parameter(cfg, "phi4", range_key="phases"),
        "phi5": _fixed_or_sampled_parameter(cfg, "phi5", range_key="phases"),
    }
    matrix = build_phased_mixing_matrix(
        sampled["theta12"],
        sampled["theta23"],
        sampled["theta13"],
        sampled["delta"],
        left_phases=(sampled["phi1"], sampled["phi2"], sampled["phi3"]),
        right_phases=(sampled["phi4"], sampled["phi5"]),
    )
    if return_parameters:
        return matrix, sampled
    return matrix


def generate_parameters(n_points, user_config=None):

    cfg = DEFAULT_RANGES.copy()
    if user_config:
        cfg.update(user_config)

    MGUT_range = cfg["MGUT"]
    gGUT_range = cfg["gGUT"]

    base_matrix_cfg = {
        "angles": cfg.get("angles", DEFAULT_RANGES["angles"]),
        "phases": cfg.get("phases", DEFAULT_RANGES["phases"]),
    }

    for _ in range(n_points):

        params = {
            "MGUT": np.random.uniform(*MGUT_range),
            "gGUT": np.random.uniform(*gGUT_range),
        }

        params["matrix_parameters"] = {}
        for matrix in MATRIX_LIST:
            matrix_cfg = cfg.get("matrix_configs", {}).get(matrix, base_matrix_cfg)
            params[matrix], params["matrix_parameters"][matrix] = build_unitary(
                matrix_cfg,
                return_parameters=True,
            )

        yield params


def generate_parameters_with_ckm_pmns(n_points, user_config=None):
    """
    Generate non-SUSY proton-decay inputs with physical left-mixing relations

        DL = UL @ Uckm,    NL = EL @ Upmns,

    where Uckm = Pckm Vckm Qckm and Upmns = Ppmns Vpmns Qpmns.
    Vckm and Vpmns use the default angles/delta above unless overridden in
    user_config["ckm_config"] or user_config["pmns_config"].
    """
    cfg = DEFAULT_RANGES.copy()
    if user_config:
        cfg.update(user_config)

    MGUT_range = cfg["MGUT"]
    gGUT_range = cfg["gGUT"]

    base_matrix_cfg = {
        "angles": cfg.get("angles", DEFAULT_RANGES["angles"]),
        "phases": cfg.get("phases", DEFAULT_RANGES["phases"]),
    }
    ckm_cfg = {"phases": cfg.get("phases", DEFAULT_RANGES["phases"])}
    ckm_cfg.update(cfg.get("ckm_config", {}))
    pmns_cfg = {"phases": cfg.get("phases", DEFAULT_RANGES["phases"])}
    pmns_cfg.update(cfg.get("pmns_config", {}))

    for _ in range(n_points):
        params = {
            "MGUT": np.random.uniform(*MGUT_range),
            "gGUT": np.random.uniform(*gGUT_range),
            "matrix_parameters": {},
        }

        for matrix in INDEPENDENT_CKM_PMNS_MATRIX_LIST:
            matrix_cfg = cfg.get("matrix_configs", {}).get(matrix, base_matrix_cfg)
            params[matrix], params["matrix_parameters"][matrix] = build_unitary(
                matrix_cfg,
                return_parameters=True,
            )

        Uckm, params["matrix_parameters"]["Uckm"] = build_fixed_angle_mixing(
            ckm_cfg,
            DEFAULT_CKM_PARAMETERS,
            return_parameters=True,
        )
        Upmns, params["matrix_parameters"]["Upmns"] = build_fixed_angle_mixing(
            pmns_cfg,
            DEFAULT_PMNS_PARAMETERS,
            return_parameters=True,
        )
        params["DL"] = params["UL"] @ Uckm
        params["NL"] = params["EL"] @ Upmns
        params["derived_matrices"] = {"Uckm": Uckm, "Upmns": Upmns}

        yield params
