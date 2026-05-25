"""
Morales et al. 2018 dynamic leaf CO2 assimilation model.

Initial mxlpy implementation skeleton:
- 16 state variables
- 7 environmental forcings as temporary parameters
- 103 model parameters
- first derived light and temperature-response equations
"""

import numpy as np
from mxlpy import Model


R_GAS = 8314.0  # J kmol-1 K-1, consistent with Morales parameter scale
AIR_PRESSURE = 1.01e8
FLUX0 = 1e-22
TREF = 298.15

def _par(Ib, Ig, Ir):
    return Ib + Ig + Ir


def _para(Ib, Ig, Ir, alphab, alphag, alphared):
    return Ib * alphab + Ig * alphag + Ir * alphared


def _parap(Ib, Ig, Ir, alphabp, alphagp, alpharp):
    return Ib * alphabp + Ig * alphagp + Ir * alpharp


def _arrhenius(T, value25, Ha):
    return value25 * np.exp((Ha * (T - 298.15)) / (298.15 * R_GAS * T))


def _peaked_arrhenius(T, value25, Ha, Hd, S):
    activation = np.exp((Ha * (T - 298.15)) / (298.15 * R_GAS * T))
    deactivation_ref = 1.0 + np.exp((298.15 * S - Hd) / (298.15 * R_GAS))
    deactivation_t = 1.0 + np.exp((T * S - Hd) / (T * R_GAS))
    return value25 * activation * deactivation_ref / deactivation_t

def _phiqe(fP, fZ, gamma1, gamma2, gamma3, PhiqEmax):
    return (fP * gamma1 + fP * fZ * gamma2 + fZ * gamma3) * PhiqEmax

def _rp(PR, kPR):
    return 0.5 * PR * kPR

def _sc(Scm, falphaSc, alphar):
    return Scm * falphaSc * alphar

def _kmapp_rubp(KmRuBP, PGA, Vch, KiPGA):
    return KmRuBP * (1 + PGA / (Vch * KiPGA))

def _frca(Tl, DHdRCA, ToRCA, DHaRCA):
    return (DHdRCA * np.exp(((Tl - ToRCA) * DHaRCA) / (ToRCA * R_GAS * Tl))) / (
        DHdRCA - DHaRCA * (1 - np.exp((DHdRCA * (Tl - ToRCA)) / (ToRCA * R_GAS * Tl)))
    )

def _fmax(RCA, fRCA, KaRCA):
    return (RCA * fRCA) / (RCA * fRCA + KaRCA)

def _para_p2(sigma2, alphar, PARaP):
    return sigma2 * alphar * PARaP

def _frubp(RB, Vch, Kmapp_RuBP, RuBP):
    inner = (RB / Vch + Kmapp_RuBP + RuBP / Vch) ** 2 - (4 * RB * RuBP) / (Vch ** 2)
    return (1 / ((2 * RB) / Vch)) * (
        (RB / Vch + Kmapp_RuBP + RuBP / Vch) - np.sqrt(max(inner, 0.0))
    )

def _phi(Kmc, Ko, O2, Kmo, Kc, Cc):
    return (Kmc * Ko * O2) / (Kmo * Kc * Cc)

def _vc(fRB, fRuBP, Kc, RB, Cc, Kmc, O2, Kmo):
    return (fRB * fRuBP * Kc * RB * Cc) / (Cc + Kmc * (1.0 + O2 / Kmo))

def _vr_tpu(TPU, phi):
    return (3.0 * TPU * (2.0 + 1.5 * phi)) / (1 - 0.5 * phi)

def _vre(fR, Vrmax, PGA, KmPGA):
    return (fR * Vrmax * PGA) / (PGA + KmPGA)

def _a(Ci, Ccyt, gw):
    return (Ci - Ccyt) * gw

def _gm(A, Ci, Cc):
    return A / (Ci - Cc)

def _vr(VrJ, VrTPU, VrE):
    return min(VrJ, VrTPU, VrE)

def get_morales2018() -> Model:
    """Return Morales et al. 2018 dynamic photosynthesis model skeleton."""

    m: Model = Model()
    
    states = {
        "PGA": 0.00005, "RuBP": 0.00005, "fRB": 0.25,
        "fP": 0.0, "fZ": 0.0, "alphar": 1.0,
        "PSIId": 0.0, "fR": 0.0, "PR": 0.0,
        "Cc": 0.00038, "Ccyt": 0.00038, "Ci": 0.00038,
        "Ca": 0.00038, "H2OS": 0.02000, "gsw": 0.09000,
        "sumA": 0.0,
    }
    for name, value in states.items():
        m = m.add_variable(name, initial_value=value)

    forcings = {
        "Ib": 100.0, "Ig": 0.0, "Ir": 900.0,
        "Ta": 298.15, "Tl": 298.15,
        "H2OR": 0.015, "CO2R": 0.00038,
    }
    for name, value in forcings.items():
        m = m.add_parameter(name, value=value)

    parameters = {
        "sigma2": 5.0000e-01, "Jmax25": 1.3928e-04,
        "DHaJmax": 3.6210e07, "DHdJmax": 2.1590e08,
        "DsJmax": 6.9000e05, "kD0": 4.5500e08,
        "kDinh": 5.0000e09, "kf": 5.6000e07,
        "kp": 2.6540e09, "fcyc": 1.0000e-01,
        "fpseudo": 1.0000e-01, "theta": 7.0000e-01,
        "gamma1": 2.0000e-01, "gamma2": 6.0000e-01,
        "gamma3": 2.0000e-01, "PhiqEmax": 2.0000e-01,
        "KiqEp": 1.8700e-02, "KdqEp": 2.3900e-02,
        "KiqEz": 1.8700e-03, "KdqEz": 2.3900e-03,
        "Kinh0": 1.0000e-01, "fprot": 1.0000e-01,
        "Krep25": 1.9200e-04, "DHaKrep": 1.6080e08,
        "DHdKrep": 2.3323e08, "DsKrep": 7.8000e05,
        "alphar_alpha25": 6.5500e03, "DHaAlphar": 6.7320e07,
        "DHaKalpha": 9.0500e07, "DsKalpha": 1.0800e06,
        "DHdKalpha": 3.2800e08, "Iac": 1.6000e-06,
        "alpharac": 5.0000e-02, "alpharav": 2.5000e-01,
        "thetaalphar": 3.6000e-01, "Kialpha25": 1.4900e-03,
        "Kdalpha25": 1.8600e-03, "fR0": 4.0000e-02,
        "alphafR": 2.5000e03, "thetafR": 9.6000e-01,
        "KiR": 6.2800e-03, "KdR": 7.5000e-03,
        "Vrmax": 1.1865e-04, "KmPGA": 5.0000e-06,
        "RB": 1.5900e-05, "Kc25": 4.1600e00,
        "DHaKc": 4.1820e07, "Ko25": 1.2600e00,
        "DHaKo": 5.5150e07, "Kmc25": 2.6170e-04,
        "DHaKmc": 4.9430e07, "Kmo25": 1.9850e-01,
        "DHaKmo": 2.9080e07, "KaRCA": 1.0200e-02,
        "ac": 2.7000e-01, "bc": 1.4000e04,
        "KdRB": 6.8000e-04, "Krca": 8.6300e-02,
        "fRBmin": 4.8000e-01, "O2": 2.1000e-01,
        "RCA": 1.1737e-01, "DHdRCA": 2.9020e08,
        "ToRCA": 3.0040e02, "DHaRCA": 3.0000e07,
        "KmRuBP": 2.0000e-02, "Vch": 1.0000e-05,
        "KiPGA": 8.4000e-01, "TPU25": 7.4700e-06,
        "DHaTPU": 5.7500e07, "DHdTPU": 2.4670e08,
        "DsTPU": 7.9000e05, "DHaGc": 7.0200e07,
        "DHdGc": 9.4000e07, "DsGc": 3.2000e05,
        "DHaGw": 7.0200e07, "DHdGw": 9.4000e07,
        "DsGw": 3.2000e05, "Rm25": 9.9000e-07,
        "DHaRm": 5.6200e07, "kPR": 2.4000e-02,
        "Vref": 1.5500e-04, "Scm": 7.1000e00,
        "Sm": 9.8000e00, "falphaSc": 9.3000e-01,
        "gcm25": 3.9000e-01, "gw25": 7.5000e-01,
        "D0": 7.4000e05, "fI0": 3.9000e-01,
        "gswm": 4.8000e-01, "alphafI": 7.6700e02,
        "thetafI": 8.8000e-01, "Kgsi": 1.1400e-03,
        "Kgsd": 1.1400e-03, "gbw": 9.2000e00,
        "volume_chamber": 8.0000e-05, "leaf_surface": 2.0000e-04,
        "Flow": 5.0000e-04, "alphab": 9.2000e-01,
        "alphag": 7.2000e-01, "alphared": 8.3000e-01,
        "alphabp": 6.6000e-01, "alphagp": 6.0000e-01,
        "alpharp": 8.0000e-01,
    }
    for name, value in parameters.items():
        m = m.add_parameter(name, value=value)

    morales_readouts = [
        "Vr", "Vc", "fRuBP", "fqEss", "PhiqE", "PhiIIoss",
        "PhiIIo", "qP", "PhiII", "VrJ", "NPQ", "qI", "qE",
        "qM", "Rp", "A", "gss", "VPDleaf", "Sc", "Photo",
        "transpiration", "Trmmol", "Cond", "gm", "reg_limit",
    ]
    for readout in morales_readouts:
        m = m.add_parameter(f"readout_placeholder_{readout}", value=0.0)

    # Light-derived quantities
    m = m.add_derived("PAR", fn=_par, args=["Ib", "Ig", "Ir"])
    m = m.add_derived("PARa", fn=_para, args=["Ib", "Ig", "Ir", "alphab", "alphag", "alphared"])
    m = m.add_derived("PARaP", fn=_parap, args=["Ib", "Ig", "Ir", "alphabp", "alphagp", "alpharp"])

    # Temperature-dependent biochemical capacities
    m = m.add_derived("Jmax", fn=_peaked_arrhenius, args=["Tl", "Jmax25", "DHaJmax", "DHdJmax", "DsJmax"])
    m = m.add_derived("Krep", fn=_peaked_arrhenius, args=["Tl", "Krep25", "DHaKrep", "DHdKrep", "DsKrep"])
    m = m.add_derived("TPU", fn=_peaked_arrhenius, args=["Tl", "TPU25", "DHaTPU", "DHdTPU", "DsTPU"])

    m = m.add_derived("Kc", fn=_arrhenius, args=["Tl", "Kc25", "DHaKc"])
    m = m.add_derived("Ko", fn=_arrhenius, args=["Tl", "Ko25", "DHaKo"])
    m = m.add_derived("Kmc", fn=_arrhenius, args=["Tl", "Kmc25", "DHaKmc"])
    m = m.add_derived("Kmo", fn=_arrhenius, args=["Tl", "Kmo25", "DHaKmo"])
    m = m.add_derived("Rm", fn=_arrhenius, args=["Tl", "Rm25", "DHaRm"])
    m = m.add_derived("gcm", fn=_peaked_arrhenius, args=["Tl", "gcm25", "DHaGc", "DHdGc", "DsGc"])
    m = m.add_derived("gw", fn=_peaked_arrhenius, args=["Tl", "gw25", "DHaGw", "DHdGw", "DsGw"])
    
    # First physiological derived quantities from Morales C++ core
    m = m.add_derived("PhiqE", fn=_phiqe, args=["fP", "fZ", "gamma1", "gamma2", "gamma3", "PhiqEmax"])
    m = m.add_derived("Rp", fn=_rp, args=["PR", "kPR"])
    m = m.add_derived("Sc", fn=_sc, args=["Scm", "falphaSc", "alphar"])
    m = m.add_derived("Kmapp_RuBP", fn=_kmapp_rubp, args=["KmRuBP", "PGA", "Vch", "KiPGA"])
    m = m.add_derived("fRCA", fn=_frca, args=["Tl", "DHdRCA", "ToRCA", "DHaRCA"])
    m = m.add_derived("fRBmax_calc", fn=_fmax, args=["RCA", "fRCA", "KaRCA"])
    m = m.add_derived("PARaP2", fn=_para_p2, args=["sigma2", "alphar", "PARaP"])
    m = m.add_derived("fRuBP_calc", fn=_frubp, args=["RB", "Vch", "Kmapp_RuBP", "RuBP"])
    m = m.add_derived("phi", fn=_phi, args=["Kmc", "Ko", "O2", "Kmo", "Kc", "Cc"])
    m = m.add_derived("Vc_calc", fn=_vc, args=["fRB", "fRuBP_calc", "Kc", "RB", "Cc", "Kmc", "O2", "Kmo"])
    m = m.add_derived("VrTPU", fn=_vr_tpu, args=["TPU", "phi"])
    m = m.add_derived("VrE", fn=_vre, args=["fR", "Vrmax", "PGA", "KmPGA"])
    m = m.add_derived("A_calc", fn=_a, args=["Ci", "Ccyt", "gw"])
    m = m.add_derived("gm_calc", fn=_gm, args=["A_calc", "Ci", "Cc"])

    return m