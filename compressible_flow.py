"""
compressible_flow.py
====================

One-dimensional compressible-flow relations for a calorically perfect gas.

Contents
--------
* Isentropic flow            : static/stagnation ratios, A/A*, inverse A/A* -> M
* Normal shock               : downstream state, inverse relations, Pitot ratio
* Oblique shock              : theta-beta-M relation, weak/strong solutions,
                               maximum deflection, downstream state
* Prandtl-Meyer expansion    : nu(M), inverse, expansion-fan solver
* Rayleigh flow              : frictionless duct with heat addition
* Fanno flow                 : adiabatic duct with wall friction

All angles at the public interface are in DEGREES. All functions validate their
inputs and raise ``ValueError`` for physically meaningless requests (e.g. a
normal shock in subsonic flow, or a wedge angle larger than the detachment
limit). Forward relations accept scalars or NumPy arrays; inverse (root-finding)
functions accept scalars only.

References
----------
[1] Anderson, J.D., Modern Compressible Flow, 3rd ed., McGraw-Hill, 2003.
[2] Ames Research Staff, NACA Report 1135, "Equations, Tables, and Charts for
    Compressible Flow", 1953.
[3] Liepmann, H.W. and Roshko, A., Elements of Gasdynamics, Wiley, 1957.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq

DEFAULT_GAMMA = 1.4


# --------------------------------------------------------------------------- #
# Result containers (NamedTuples can still be unpacked like plain tuples)
# --------------------------------------------------------------------------- #
class IsentropicState(NamedTuple):
    T0_T: float
    p0_p: float
    rho0_rho: float


class NormalShockState(NamedTuple):
    M2: float
    p2_p1: float
    rho2_rho1: float
    T2_T1: float
    p02_p01: float


class ObliqueShockState(NamedTuple):
    beta_deg: float
    theta_deg: float
    Mn1: float
    Mn2: float
    M2: float
    p2_p1: float
    rho2_rho1: float
    T2_T1: float
    p02_p01: float


class ExpansionState(NamedTuple):
    M2: float
    nu1_deg: float
    nu2_deg: float
    p2_p1: float
    rho2_rho1: float
    T2_T1: float
    mu1_deg: float  # leading Mach wave angle
    mu2_deg: float  # trailing Mach wave angle


class RayleighState(NamedTuple):
    T_Tstar: float
    p_pstar: float
    rho_rhostar: float
    V_Vstar: float
    T0_T0star: float
    p0_p0star: float


class FannoState(NamedTuple):
    T_Tstar: float
    p_pstar: float
    rho_rhostar: float
    V_Vstar: float
    p0_p0star: float
    fLstar_D: float  # Darcy friction factor * L* / D  (= 4 * Fanning f * L*/D)


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _check_gamma(gamma):
    g = np.asarray(gamma, dtype=float)
    if g.ndim != 0 or not np.isfinite(g) or g <= 1.0:
        raise ValueError(f"gamma must be a finite scalar > 1, got {gamma!r}.")
    return float(g)


def _check_mach(M, *, minimum=0.0, inclusive=False, name="Mach number"):
    """Return M as a float array; require M > minimum (or >= if inclusive)."""
    arr = np.asarray(M, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {M!r}.")
    bad = np.any(arr < minimum) if inclusive else np.any(arr <= minimum)
    if bad:
        rel = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be {rel} {minimum}, got {M!r}.")
    return arr


def _scalar(x):
    """Collapse 0-d arrays to Python floats, leave real arrays alone."""
    x = np.asarray(x)
    return float(x) if x.ndim == 0 else x


# --------------------------------------------------------------------------- #
# 1. Isentropic flow
# --------------------------------------------------------------------------- #
def isentropic_ratios(M, gamma=DEFAULT_GAMMA) -> IsentropicState:
    """Stagnation-to-static ratios T0/T, p0/p, rho0/rho at Mach M (M >= 0)."""
    g = _check_gamma(gamma)
    M = _check_mach(M, inclusive=True)
    factor = 1.0 + 0.5 * (g - 1.0) * M**2
    return IsentropicState(
        _scalar(factor),
        _scalar(factor ** (g / (g - 1.0))),
        _scalar(factor ** (1.0 / (g - 1.0))),
    )


def area_mach_ratio(M, gamma=DEFAULT_GAMMA):
    """Area ratio A/A* for isentropic quasi-1D flow (M > 0)."""
    g = _check_gamma(gamma)
    M = _check_mach(M)
    term = (2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M**2)
    return _scalar(term ** ((g + 1.0) / (2.0 * (g - 1.0))) / M)


def mach_from_area_ratio(A_Astar, supersonic=True, gamma=DEFAULT_GAMMA):
    """Invert A/A* -> M. ``supersonic`` selects the branch (M>1 or M<1)."""
    g = _check_gamma(gamma)
    A = float(A_Astar)
    if not np.isfinite(A) or A < 1.0 - 1e-12:
        raise ValueError(f"A/A* must be >= 1, got {A_Astar!r}.")
    # NB: A/A* - 1 ~ (M-1)^2 near the sonic point, so the tolerance must be a pure
    # absolute one (np.isclose's default rtol would corrupt M by ~1e-3 here).
    if abs(A - 1.0) <= 1e-12:
        return 1.0
    f = lambda m: area_mach_ratio(m, g) - A
    if supersonic:
        hi = 2.0
        while f(hi) < 0.0:
            hi *= 2.0
        return brentq(f, 1.0, hi, xtol=1e-14)
    return brentq(f, 1e-9, 1.0, xtol=1e-14)


# --------------------------------------------------------------------------- #
# 2. Normal shock
# --------------------------------------------------------------------------- #
def normal_shock(M1, gamma=DEFAULT_GAMMA) -> NormalShockState:
    """Property ratios across a stationary normal shock (M1 >= 1)."""
    g = _check_gamma(gamma)
    M1 = _check_mach(M1, minimum=1.0, inclusive=True, name="Upstream Mach number M1")
    M1sq = M1**2
    M2 = np.sqrt((1 + 0.5 * (g - 1) * M1sq) / (g * M1sq - 0.5 * (g - 1)))
    p2_p1 = 1 + (2 * g / (g + 1)) * (M1sq - 1)
    rho2_rho1 = ((g + 1) * M1sq) / ((g - 1) * M1sq + 2)
    T2_T1 = p2_p1 / rho2_rho1
    p02_p01 = (rho2_rho1 ** (g / (g - 1))) * (
        ((g + 1) / (2 * g * M1sq - (g - 1))) ** (1 / (g - 1))
    )
    return NormalShockState(*(_scalar(v) for v in (M2, p2_p1, rho2_rho1, T2_T1, p02_p01)))


def shock_entropy_rise(M1, gamma=DEFAULT_GAMMA):
    """Dimensionless entropy rise (s2 - s1)/R = -ln(p02/p01) across a normal shock."""
    return _scalar(-np.log(normal_shock(M1, gamma).p02_p01))


def pitot_pressure_ratio(M1, gamma=DEFAULT_GAMMA):
    """Rayleigh Pitot formula: p02/p1 for a Pitot probe in supersonic flow (M1 >= 1)."""
    g = _check_gamma(gamma)
    M1 = _check_mach(M1, minimum=1.0, inclusive=True, name="Upstream Mach number M1")
    M1sq = M1**2
    a = (0.5 * (g + 1) * M1sq) ** (g / (g - 1))
    b = (2 * g * M1sq / (g + 1) - (g - 1) / (g + 1)) ** (1 / (g - 1))
    return _scalar(a / b)


def mach1_from_pressure_ratio(p2_p1, gamma=DEFAULT_GAMMA):
    """Upstream Mach number that produces a given static-pressure jump p2/p1 >= 1."""
    g = _check_gamma(gamma)
    r = float(p2_p1)
    if not np.isfinite(r) or r < 1.0:
        raise ValueError(f"p2/p1 across a shock must be >= 1, got {p2_p1!r}.")
    return float(np.sqrt(1.0 + (g + 1.0) / (2.0 * g) * (r - 1.0)))


def mach1_from_downstream_mach(M2, gamma=DEFAULT_GAMMA):
    """Upstream Mach number for a given post-shock Mach number M2."""
    g = _check_gamma(gamma)
    M2 = float(M2)
    lo = np.sqrt((g - 1.0) / (2.0 * g))
    if not (lo <= M2 <= 1.0):
        raise ValueError(
            f"Post-shock Mach number must lie in [{lo:.4f}, 1] for gamma={g}, got {M2!r}."
        )
    return float(np.sqrt((1 + 0.5 * (g - 1) * M2**2) / (g * M2**2 - 0.5 * (g - 1))))


# --------------------------------------------------------------------------- #
# 3. Oblique shock
# --------------------------------------------------------------------------- #
def mach_angle_deg(M):
    """Mach angle mu = arcsin(1/M) in degrees (M >= 1)."""
    M = _check_mach(M, minimum=1.0, inclusive=True)
    return _scalar(np.degrees(np.arcsin(1.0 / M)))


def theta_from_beta(M1, beta_deg, gamma=DEFAULT_GAMMA):
    """Flow-deflection angle theta (deg) for shock-wave angle beta (deg): the theta-beta-M relation."""
    g = _check_gamma(gamma)
    M1 = _check_mach(M1, minimum=1.0, inclusive=True, name="Upstream Mach number M1")
    beta = np.radians(np.asarray(beta_deg, dtype=float))
    tan_theta = (
        2.0
        / np.tan(beta)
        * (M1**2 * np.sin(beta) ** 2 - 1.0)
        / (M1**2 * (g + np.cos(2.0 * beta)) + 2.0)
    )
    return _scalar(np.degrees(np.arctan(tan_theta)))


def max_deflection_angle(M1, gamma=DEFAULT_GAMMA):
    """Return (theta_max_deg, beta_at_theta_max_deg) for an attached oblique shock."""
    g = _check_gamma(gamma)
    M1 = float(_check_mach(M1, minimum=1.0, name="Upstream Mach number M1 (must be > 1)"))
    M2sq = M1**2
    sin2b = (1.0 / (g * M2sq)) * (
        0.25 * (g + 1) * M2sq
        - 1.0
        + np.sqrt((g + 1) * (1 + 0.5 * (g - 1) * M2sq + (g + 1) / 16.0 * M2sq**2))
    )
    beta_max = np.arcsin(np.sqrt(sin2b))
    return float(theta_from_beta(M1, np.degrees(beta_max), g)), float(np.degrees(beta_max))


def oblique_shock(M1, theta_deg=None, beta_deg=None, weak=True, gamma=DEFAULT_GAMMA) -> ObliqueShockState:
    """
    Solve an oblique shock given M1 and EITHER the deflection angle ``theta_deg``
    or the wave angle ``beta_deg`` (exactly one must be supplied).

    ``weak`` chooses the weak (default) or strong solution when theta is given.
    Raises ValueError if theta exceeds the detachment limit theta_max(M1).
    """
    g = _check_gamma(gamma)
    M1 = float(_check_mach(M1, minimum=1.0, name="Upstream Mach number M1 (must be > 1)"))
    if (theta_deg is None) == (beta_deg is None):
        raise ValueError("Provide exactly one of theta_deg or beta_deg.")

    mu = float(mach_angle_deg(M1))
    if beta_deg is None:
        theta = float(theta_deg)
        if theta < 0.0:
            raise ValueError("theta must be >= 0 (mirror the geometry for expansions).")
        theta_max, beta_tmax = max_deflection_angle(M1, g)
        if theta > theta_max + 1e-12:
            raise ValueError(
                f"theta = {theta:.3f} deg exceeds theta_max = {theta_max:.3f} deg at "
                f"M1 = {M1:g}: the shock detaches (no attached oblique solution)."
            )
        if theta == 0.0:
            beta = mu if weak else 90.0
        else:
            f = lambda b: float(theta_from_beta(M1, b, g)) - theta
            if weak:
                beta = brentq(f, mu, beta_tmax, xtol=1e-13)
            else:
                beta = brentq(f, beta_tmax, 90.0, xtol=1e-13)
    else:
        beta = float(beta_deg)
        if not (mu - 1e-12 <= beta <= 90.0 + 1e-12):
            raise ValueError(
                f"beta must lie between the Mach angle ({mu:.3f} deg) and 90 deg, got {beta:g}."
            )
        theta = float(theta_from_beta(M1, beta, g))

    b = np.radians(beta)
    Mn1 = max(M1 * np.sin(b), 1.0)
    ns = normal_shock(Mn1, g)
    M2 = ns.M2 / np.sin(b - np.radians(theta))
    return ObliqueShockState(
        float(beta), float(theta), float(Mn1), float(ns.M2), float(M2),
        float(ns.p2_p1), float(ns.rho2_rho1), float(ns.T2_T1), float(ns.p02_p01),
    )


# --------------------------------------------------------------------------- #
# 4. Prandtl-Meyer expansion
# --------------------------------------------------------------------------- #
def prandtl_meyer(M, gamma=DEFAULT_GAMMA):
    """Prandtl-Meyer function nu(M) in degrees (M >= 1)."""
    g = _check_gamma(gamma)
    M = _check_mach(M, minimum=1.0, inclusive=True)
    a = np.sqrt((g + 1) / (g - 1))
    s = np.sqrt(M**2 - 1)
    nu = a * np.arctan(np.sqrt((g - 1) / (g + 1)) * s) - np.arctan(s)
    return _scalar(np.degrees(nu))


def prandtl_meyer_max(gamma=DEFAULT_GAMMA):
    """Limiting turn angle nu_max = (pi/2)(sqrt((g+1)/(g-1)) - 1), in degrees."""
    g = _check_gamma(gamma)
    return float(90.0 * (np.sqrt((g + 1) / (g - 1)) - 1.0))


def mach_from_pm_angle(nu_target_deg, gamma=DEFAULT_GAMMA):
    """Invert the Prandtl-Meyer function: Mach number whose nu(M) equals the target."""
    g = _check_gamma(gamma)
    nu = float(nu_target_deg)
    nu_max = prandtl_meyer_max(g)
    if not np.isfinite(nu) or nu < 0.0:
        raise ValueError(f"nu must be >= 0, got {nu_target_deg!r}.")
    if nu >= nu_max:
        raise ValueError(
            f"nu = {nu:g} deg is not reachable: the limit for gamma={g:g} is {nu_max:.4f} deg."
        )
    if nu == 0.0:
        return 1.0
    f = lambda m: float(prandtl_meyer(m, g)) - nu
    hi = 2.0
    while f(hi) < 0.0:
        hi *= 2.0
        if hi > 1e12:
            raise ValueError("Could not bracket the Mach number (nu too close to the limit).")
    return brentq(f, 1.0, hi, xtol=1e-13)


def prandtl_meyer_expansion(M1, turn_deg, gamma=DEFAULT_GAMMA) -> ExpansionState:
    """Isentropic expansion of supersonic flow M1 turned through ``turn_deg`` degrees."""
    g = _check_gamma(gamma)
    M1 = float(_check_mach(M1, minimum=1.0, inclusive=True, name="Upstream Mach number M1"))
    turn = float(turn_deg)
    if turn < 0.0:
        raise ValueError("Expansion turn angle must be >= 0.")
    nu1 = float(prandtl_meyer(M1, g))
    nu2 = nu1 + turn
    M2 = mach_from_pm_angle(nu2, g)
    t1, p1, r1 = isentropic_ratios(M1, g)
    t2, p2, r2 = isentropic_ratios(M2, g)
    return ExpansionState(
        M2, nu1, nu2,
        p1 / p2, r1 / r2, t1 / t2,  # static ratios: (p0/p)_1 / (p0/p)_2 etc.
        float(mach_angle_deg(M1)), float(mach_angle_deg(M2)),
    )


# --------------------------------------------------------------------------- #
# 5. Rayleigh flow (frictionless, constant-area, heat addition)
# --------------------------------------------------------------------------- #
def rayleigh_ratios(M, gamma=DEFAULT_GAMMA) -> RayleighState:
    """Rayleigh-flow ratios relative to the sonic (*) state, M > 0."""
    g = _check_gamma(gamma)
    M = _check_mach(M)
    d = 1.0 + g * M**2
    T_Ts = (g + 1) ** 2 * M**2 / d**2
    p_ps = (g + 1) / d
    rho_rs = d / ((g + 1) * M**2)
    T0_T0s = (g + 1) * M**2 * (2 + (g - 1) * M**2) / d**2
    p0_p0s = p_ps * ((2 + (g - 1) * M**2) / (g + 1)) ** (g / (g - 1))
    return RayleighState(*(_scalar(v) for v in (T_Ts, p_ps, rho_rs, 1.0 / rho_rs, T0_T0s, p0_p0s)))


def mach_from_rayleigh_T0_ratio(T0_T0star, supersonic, gamma=DEFAULT_GAMMA):
    """Invert T0/T0* -> M on the subsonic or supersonic branch."""
    g = _check_gamma(gamma)
    r = float(T0_T0star)
    if not (0.0 < r <= 1.0 + 1e-12):
        raise ValueError(f"T0/T0* must be in (0, 1], got {T0_T0star!r}.")
    if abs(r - 1.0) <= 1e-12:  # pure absolute tolerance: T0/T0* - 1 ~ (M-1)^2 near M = 1
        return 1.0
    f = lambda m: float(rayleigh_ratios(m, g).T0_T0star) - r
    if supersonic:
        floor = (g + 1) * (g - 1) / g**2  # limit as M -> infinity
        if r <= floor:
            raise ValueError(f"No supersonic solution: T0/T0* must exceed {floor:.4f}.")
        hi = 2.0
        while f(hi) > 0.0:
            hi *= 2.0
        return brentq(f, 1.0, hi, xtol=1e-14)
    return brentq(f, 1e-9, 1.0, xtol=1e-14)


def rayleigh_exit(M1, T02_T01, gamma=DEFAULT_GAMMA):
    """
    Exit Mach number after heating (T02/T01 > 1) or cooling (< 1) a Rayleigh duct.
    The flow stays on its original branch; heating beyond the sonic limit
    raises ValueError (thermal choking).
    """
    g = _check_gamma(gamma)
    M1 = float(_check_mach(M1))
    ratio = float(T02_T01)
    if ratio <= 0.0:
        raise ValueError("T02/T01 must be positive.")
    target = float(rayleigh_ratios(M1, g).T0_T0star) * ratio
    if target > 1.0 + 1e-12:
        max_ratio = 1.0 / float(rayleigh_ratios(M1, g).T0_T0star)
        raise ValueError(
            f"Thermally choked: maximum T02/T01 for M1={M1:g} is {max_ratio:.4f}, got {ratio:g}."
        )
    return mach_from_rayleigh_T0_ratio(min(target, 1.0), supersonic=M1 > 1.0, gamma=g)


# --------------------------------------------------------------------------- #
# 6. Fanno flow (adiabatic, constant-area, wall friction)
# --------------------------------------------------------------------------- #
def fanno_parameter(M, gamma=DEFAULT_GAMMA):
    """Friction parameter f*L*/D (Darcy f, i.e. 4 x Fanning) needed to reach M = 1."""
    g = _check_gamma(gamma)
    M = _check_mach(M)
    M2 = M**2
    val = (1 - M2) / (g * M2) + (g + 1) / (2 * g) * np.log((g + 1) * M2 / (2 + (g - 1) * M2))
    return _scalar(val)


def fanno_ratios(M, gamma=DEFAULT_GAMMA) -> FannoState:
    """Fanno-flow ratios relative to the sonic (*) state, M > 0."""
    g = _check_gamma(gamma)
    M = _check_mach(M)
    e = 2 + (g - 1) * M**2
    T_Ts = (g + 1) / e
    p_ps = np.sqrt((g + 1) / e) / M
    rho_rs = np.sqrt(e / (g + 1)) / M
    p0_p0s = (e / (g + 1)) ** ((g + 1) / (2 * (g - 1))) / M
    return FannoState(
        *(_scalar(v) for v in (T_Ts, p_ps, rho_rs, 1.0 / rho_rs, p0_p0s)),
        fanno_parameter(M, g),
    )


def mach_from_fanno_parameter(fLstar_D, supersonic, gamma=DEFAULT_GAMMA):
    """Invert f*L*/D -> M on the subsonic or supersonic branch."""
    g = _check_gamma(gamma)
    p = float(fLstar_D)
    if p < 0.0 or not np.isfinite(p):
        raise ValueError(f"f L*/D must be >= 0, got {fLstar_D!r}.")
    if p == 0.0:
        return 1.0
    f = lambda m: float(fanno_parameter(m, g)) - p
    if supersonic:
        pmax = (g + 1) / (2 * g) * np.log((g + 1) / (g - 1)) - 1.0 / g
        if p >= pmax:
            raise ValueError(f"No supersonic solution: f L*/D must be below {pmax:.4f}.")
        hi = 2.0
        while f(hi) < 0.0:
            hi *= 2.0
        return brentq(f, 1.0, hi, xtol=1e-14)
    return brentq(f, 1e-9, 1.0, xtol=1e-14)


def fanno_exit(M1, fL_D, gamma=DEFAULT_GAMMA):
    """
    Exit Mach number of a Fanno duct with friction parameter f*L/D (Darcy f).
    Raises ValueError if the duct is longer than L*, i.e. the flow chokes.
    """
    g = _check_gamma(gamma)
    M1 = float(_check_mach(M1))
    if fL_D < 0.0:
        raise ValueError("f L/D must be >= 0.")
    remaining = float(fanno_parameter(M1, g)) - float(fL_D)
    if remaining < -1e-12:
        raise ValueError(
            f"Choked: duct length f L/D = {fL_D:g} exceeds f L*/D = "
            f"{float(fanno_parameter(M1, g)):.4f} for M1 = {M1:g}."
        )
    return mach_from_fanno_parameter(max(remaining, 0.0), supersonic=M1 > 1.0, gamma=g)
