"""
Test-suite for compressible_flow.py

Reference values (gamma = 1.4) come from the published tables in
NACA Report 1135 and Anderson, "Modern Compressible Flow" (Appendices A-C).
Table values are quoted to 4-5 significant figures, so tolerances are set to the
last quoted digit.
"""

import numpy as np
import pytest

import compressible_flow as cf

G = 1.4


# ------------------------------------------------------------------ isentropic
class TestIsentropic:
    @pytest.mark.parametrize(
        "M, T0_T, p0_p, rho0_rho, A_Astar",
        [
            (0.5, 1.0500, 1.1862, 1.1297, 1.3398),
            (1.0, 1.2000, 1.8929, 1.5774, 1.0000),
            (2.0, 1.8000, 7.8244, 4.3469, 1.6875),
            (3.0, 2.8000, 36.733, 13.119, 4.2346),  # rho0/rho = 2.8**2.5
        ],
    )
    def test_table_values(self, M, T0_T, p0_p, rho0_rho, A_Astar):
        s = cf.isentropic_ratios(M)
        assert s.T0_T == pytest.approx(T0_T, abs=5e-5)
        assert s.p0_p == pytest.approx(p0_p, rel=2e-4)
        assert s.rho0_rho == pytest.approx(rho0_rho, rel=2e-4)
        assert cf.area_mach_ratio(M) == pytest.approx(A_Astar, rel=2e-4)

    def test_ideal_gas_consistency(self):
        # p0/p = (rho0/rho) * (T0/T) for an ideal gas
        s = cf.isentropic_ratios(1.7, 1.3)
        assert s.p0_p == pytest.approx(s.rho0_rho * s.T0_T)

    def test_zero_mach_is_stagnation(self):
        assert tuple(cf.isentropic_ratios(0.0)) == (1.0, 1.0, 1.0)

    @pytest.mark.parametrize("M", [0.3, 0.9, 1.5, 4.0, 12.0])
    def test_area_inverse_roundtrip(self, M):
        A = cf.area_mach_ratio(M)
        assert cf.mach_from_area_ratio(A, supersonic=M > 1) == pytest.approx(M, rel=1e-9)

    def test_area_inverse_sonic(self):
        assert cf.mach_from_area_ratio(1.0) == 1.0

    def test_vectorised(self):
        M = np.array([0.5, 1.0, 2.0])
        out = cf.isentropic_ratios(M)
        assert out.p0_p.shape == (3,)
        assert out.p0_p[2] == pytest.approx(7.8244, rel=2e-4)

    @pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
    def test_rejects_bad_mach(self, bad):
        with pytest.raises(ValueError):
            cf.isentropic_ratios(bad)

    @pytest.mark.parametrize("g", [1.0, 0.9, -1.4, np.nan])
    def test_rejects_bad_gamma(self, g):
        with pytest.raises(ValueError):
            cf.isentropic_ratios(2.0, g)

    def test_area_ratio_zero_mach_rejected(self):
        # the original code raised ZeroDivisionError here
        with pytest.raises(ValueError):
            cf.area_mach_ratio(0.0)

    def test_area_inverse_rejects_below_one(self):
        with pytest.raises(ValueError):
            cf.mach_from_area_ratio(0.9)


# ---------------------------------------------------------------- normal shock
class TestNormalShock:
    @pytest.mark.parametrize(
        "M1, M2, p, rho, T, p0",
        [
            (2.0, 0.5774, 4.5000, 2.6667, 1.6875, 0.7209),
            (3.0, 0.4752, 10.333, 3.8571, 2.6790, 0.3283),
            (5.0, 0.4152, 29.000, 5.0000, 5.8000, 0.06172),
        ],
    )
    def test_table_values(self, M1, M2, p, rho, T, p0):
        s = cf.normal_shock(M1)
        assert s.M2 == pytest.approx(M2, abs=1e-4)
        assert s.p2_p1 == pytest.approx(p, rel=1e-4)
        assert s.rho2_rho1 == pytest.approx(rho, rel=1e-4)
        assert s.T2_T1 == pytest.approx(T, rel=1e-4)
        assert s.p02_p01 == pytest.approx(p0, rel=2e-3)

    def test_sonic_limit_is_no_shock(self):
        assert all(v == pytest.approx(1.0) for v in cf.normal_shock(1.0))

    def test_subsonic_rejected(self):
        with pytest.raises(ValueError):
            cf.normal_shock(0.8)

    def test_downstream_subsonic_and_loses_pressure(self):
        for M1 in (1.1, 1.5, 2.5, 10.0):
            s = cf.normal_shock(M1)
            assert s.M2 < 1.0
            assert 0.0 < s.p02_p01 < 1.0

    def test_strong_shock_density_limit(self):
        # rho2/rho1 -> (g+1)/(g-1) = 6 as M1 -> infinity
        assert cf.normal_shock(1e3).rho2_rho1 == pytest.approx(6.0, rel=1e-3)

    def test_entropy_rise_positive(self):
        assert cf.shock_entropy_rise(2.0) == pytest.approx(-np.log(0.7209), rel=2e-3)
        assert cf.shock_entropy_rise(1.0) == pytest.approx(0.0, abs=1e-12)

    def test_pitot_ratio_table(self):
        assert cf.pitot_pressure_ratio(2.0) == pytest.approx(5.6404, rel=1e-4)

    def test_pitot_consistent_with_shock_plus_isentropic(self):
        M1 = 2.7
        s = cf.normal_shock(M1)
        p01_p1 = cf.isentropic_ratios(M1).p0_p
        assert cf.pitot_pressure_ratio(M1) == pytest.approx(s.p02_p01 * p01_p1)

    def test_inverse_relations(self):
        s = cf.normal_shock(3.3)
        assert cf.mach1_from_pressure_ratio(s.p2_p1) == pytest.approx(3.3)
        assert cf.mach1_from_downstream_mach(s.M2) == pytest.approx(3.3)

    def test_inverse_bounds(self):
        with pytest.raises(ValueError):
            cf.mach1_from_pressure_ratio(0.5)
        with pytest.raises(ValueError):
            cf.mach1_from_downstream_mach(0.2)  # below sqrt((g-1)/2g) = 0.378

    def test_vectorised_shock(self):
        # the original implementation failed on array input
        s = cf.normal_shock(np.array([2.0, 3.0]))
        assert s.p2_p1[1] == pytest.approx(10.333, rel=1e-4)


# ---------------------------------------------------------------- oblique shock
class TestObliqueShock:
    def test_table_M2_theta10(self):
        s = cf.oblique_shock(2.0, theta_deg=10.0)
        assert s.beta_deg == pytest.approx(39.31, abs=0.01)
        assert s.M2 == pytest.approx(1.641, abs=2e-3)
        assert s.p2_p1 == pytest.approx(1.707, abs=2e-3)

    def test_textbook_example_M2_theta20(self):
        # Anderson Example 4.1-style case: M1 = 2, theta = 20 deg -> beta ~ 53.4 deg
        s = cf.oblique_shock(2.0, theta_deg=20.0)
        assert s.beta_deg == pytest.approx(53.4, abs=0.1)
        assert s.Mn1 == pytest.approx(1.60, abs=0.01)
        assert s.M2 == pytest.approx(1.21, abs=0.01)
        # The textbook rounds Mn1 to 1.60 -> p2/p1 = 2.82; the exact Mn1 = 1.607 gives 2.843.
        assert s.p2_p1 == pytest.approx(2.82, abs=0.03)
        assert s.p2_p1 == pytest.approx(1 + 2.8 / 2.4 * (s.Mn1**2 - 1), rel=1e-12)

    def test_max_deflection_values(self):
        theta_max, _ = cf.max_deflection_angle(2.0)
        assert theta_max == pytest.approx(22.97, abs=0.02)
        assert cf.max_deflection_angle(1e4)[0] == pytest.approx(45.58, abs=0.05)

    def test_detached_shock_rejected(self):
        with pytest.raises(ValueError, match="detach"):
            cf.oblique_shock(2.0, theta_deg=25.0)

    def test_weak_vs_strong_branches(self):
        weak = cf.oblique_shock(3.0, theta_deg=15.0, weak=True)
        strong = cf.oblique_shock(3.0, theta_deg=15.0, weak=False)
        assert weak.beta_deg < strong.beta_deg
        assert weak.M2 > 1.0       # weak solution: supersonic downstream
        assert strong.M2 < 1.0     # strong solution: subsonic downstream
        assert strong.p2_p1 > weak.p2_p1
        # both must reproduce the same deflection angle
        for s in (weak, strong):
            assert cf.theta_from_beta(3.0, s.beta_deg) == pytest.approx(15.0, abs=1e-9)

    def test_beta_input_roundtrip(self):
        s1 = cf.oblique_shock(2.5, theta_deg=12.0)
        s2 = cf.oblique_shock(2.5, beta_deg=s1.beta_deg)
        assert s2.theta_deg == pytest.approx(12.0, abs=1e-9)
        assert s2.M2 == pytest.approx(s1.M2)

    def test_normal_shock_limit(self):
        # beta = 90 deg must reduce to the normal shock
        o = cf.oblique_shock(2.0, beta_deg=90.0)
        n = cf.normal_shock(2.0)
        assert o.theta_deg == pytest.approx(0.0, abs=1e-9)
        assert o.p2_p1 == pytest.approx(n.p2_p1)
        assert o.M2 == pytest.approx(n.M2)

    def test_zero_deflection_is_mach_wave(self):
        s = cf.oblique_shock(2.0, theta_deg=0.0)
        assert s.beta_deg == pytest.approx(30.0)  # arcsin(1/2)
        assert s.p2_p1 == pytest.approx(1.0)

    def test_requires_exactly_one_angle(self):
        with pytest.raises(ValueError):
            cf.oblique_shock(2.0)
        with pytest.raises(ValueError):
            cf.oblique_shock(2.0, theta_deg=10.0, beta_deg=40.0)

    def test_beta_below_mach_angle_rejected(self):
        with pytest.raises(ValueError):
            cf.oblique_shock(2.0, beta_deg=20.0)

    def test_tangential_velocity_conserved(self):
        # V_t is continuous across an oblique shock: check with temperatures (a ~ sqrt(T))
        M1 = 3.0
        s = cf.oblique_shock(M1, theta_deg=14.0)
        b, th = np.radians(s.beta_deg), np.radians(s.theta_deg)
        a2_a1 = np.sqrt(s.T2_T1)
        Vt1 = M1 * np.cos(b)
        Vt2 = s.M2 * a2_a1 * np.cos(b - th)
        assert Vt2 == pytest.approx(Vt1, rel=1e-9)

    def test_mach_angle(self):
        assert cf.mach_angle_deg(2.0) == pytest.approx(30.0)
        assert cf.mach_angle_deg(1.0) == pytest.approx(90.0)


# ---------------------------------------------------------------- Prandtl-Meyer
class TestPrandtlMeyer:
    @pytest.mark.parametrize("M, nu", [(1.0, 0.0), (2.0, 26.380), (3.0, 49.757), (5.0, 76.920)])
    def test_table_values(self, M, nu):
        assert cf.prandtl_meyer(M) == pytest.approx(nu, abs=2e-3)

    def test_limit(self):
        assert cf.prandtl_meyer_max() == pytest.approx(130.454, abs=1e-3)
        assert cf.prandtl_meyer(1e6) == pytest.approx(130.454, abs=1e-2)

    def test_monotonic(self):
        nu = cf.prandtl_meyer(np.linspace(1.0, 10.0, 200))
        assert np.all(np.diff(nu) > 0)

    @pytest.mark.parametrize("M", [1.0001, 1.5, 2.0, 5.0, 20.0])
    def test_inverse_roundtrip(self, M):
        assert cf.mach_from_pm_angle(cf.prandtl_meyer(M)) == pytest.approx(M, rel=1e-8)

    def test_inverse_zero_and_unreachable(self):
        assert cf.mach_from_pm_angle(0.0) == 1.0   # failed in the original code
        assert cf.mach_from_pm_angle(130.0) > 100  # close to the limit, still solvable
        with pytest.raises(ValueError):
            cf.mach_from_pm_angle(131.0)
        with pytest.raises(ValueError):
            cf.mach_from_pm_angle(-1.0)

    def test_subsonic_rejected(self):
        with pytest.raises(ValueError):
            cf.prandtl_meyer(0.9)

    def test_expansion_fan(self):
        e = cf.prandtl_meyer_expansion(2.0, 10.0)
        assert e.nu2_deg == pytest.approx(36.380, abs=2e-3)
        assert e.M2 == pytest.approx(2.386, abs=2e-3)
        assert e.p2_p1 < 1.0 and e.T2_T1 < 1.0 and e.rho2_rho1 < 1.0
        assert e.mu2_deg < e.mu1_deg
        # isentropic: stagnation conditions unchanged
        p0 = cf.isentropic_ratios(2.0).p0_p / cf.isentropic_ratios(e.M2).p0_p
        assert e.p2_p1 == pytest.approx(p0)

    def test_zero_turn_is_identity(self):
        e = cf.prandtl_meyer_expansion(3.0, 0.0)
        assert e.M2 == pytest.approx(3.0)
        assert e.p2_p1 == pytest.approx(1.0)


# --------------------------------------------------------------------- Rayleigh
class TestRayleigh:
    def test_sonic_reference(self):
        assert all(v == pytest.approx(1.0) for v in cf.rayleigh_ratios(1.0))

    @pytest.mark.parametrize(
        "M, T, p, T0, p0",
        [
            (0.5, 0.7901, 1.7778, 0.6914, 1.1141),
            (2.0, 0.5289, 0.3636, 0.7934, 1.5031),
        ],
    )
    def test_table_values(self, M, T, p, T0, p0):
        r = cf.rayleigh_ratios(M)
        assert r.T_Tstar == pytest.approx(T, abs=1e-4)
        assert r.p_pstar == pytest.approx(p, abs=1e-4)
        assert r.T0_T0star == pytest.approx(T0, abs=1e-4)
        assert r.p0_p0star == pytest.approx(p0, abs=1e-4)

    def test_velocity_density_reciprocal(self):
        r = cf.rayleigh_ratios(2.0)
        assert r.rho_rhostar * r.V_Vstar == pytest.approx(1.0)
        assert r.V_Vstar == pytest.approx(1.4545, abs=1e-4)

    @pytest.mark.parametrize("M", [0.2, 0.7, 1.8, 6.0])
    def test_inverse_roundtrip(self, M):
        r = cf.rayleigh_ratios(M).T0_T0star
        assert cf.mach_from_rayleigh_T0_ratio(r, supersonic=M > 1) == pytest.approx(M, rel=1e-8)

    def test_supersonic_floor(self):
        with pytest.raises(ValueError):
            cf.mach_from_rayleigh_T0_ratio(0.40, supersonic=True)  # floor ~0.4898

    def test_heating_accelerates_subsonic_flow(self):
        M2 = cf.rayleigh_exit(0.3, 2.0)
        assert 0.3 < M2 < 1.0

    def test_heating_decelerates_supersonic_flow(self):
        M2 = cf.rayleigh_exit(3.0, 1.5)
        assert 1.0 < M2 < 3.0

    def test_cooling_reverses_trend(self):
        assert cf.rayleigh_exit(0.5, 0.8) < 0.5

    def test_thermal_choking(self):
        ratio_choke = 1.0 / cf.rayleigh_ratios(0.5).T0_T0star
        assert cf.rayleigh_exit(0.5, ratio_choke * 0.999999) == pytest.approx(1.0, abs=1e-2)
        with pytest.raises(ValueError, match="choked"):
            cf.rayleigh_exit(0.5, ratio_choke * 1.01)


# ------------------------------------------------------------------------ Fanno
class TestFanno:
    def test_sonic_reference(self):
        f = cf.fanno_ratios(1.0)
        assert all(v == pytest.approx(1.0) for v in f[:5])
        assert f.fLstar_D == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize(
        "M, T, p, rho, V, p0, fL",
        [
            (0.5, 1.1429, 2.1381, 1.8708, 0.5345, 1.3398, 1.0691),
            (2.0, 0.6667, 0.4082, 0.6124, 1.6330, 1.6875, 0.3050),
        ],
    )
    def test_table_values(self, M, T, p, rho, V, p0, fL):
        f = cf.fanno_ratios(M)
        assert f.T_Tstar == pytest.approx(T, abs=1e-4)
        assert f.p_pstar == pytest.approx(p, abs=1e-4)
        assert f.rho_rhostar == pytest.approx(rho, abs=1e-4)
        assert f.V_Vstar == pytest.approx(V, abs=1e-4)
        assert f.p0_p0star == pytest.approx(p0, abs=1e-4)
        assert f.fLstar_D == pytest.approx(fL, abs=1e-4)

    def test_p0_ratio_equals_area_ratio(self):
        # For Fanno flow p0/p0* is identical to the isentropic A/A*
        for M in (0.3, 0.8, 1.5, 3.0):
            assert cf.fanno_ratios(M).p0_p0star == pytest.approx(cf.area_mach_ratio(M))

    def test_ideal_gas_consistency(self):
        # p = rho R T  =>  p/p* = (rho/rho*)(T/T*)
        f = cf.fanno_ratios(0.6)
        assert f.p_pstar == pytest.approx(f.rho_rhostar * f.T_Tstar)

    def test_supersonic_limit(self):
        assert cf.fanno_parameter(1e6) == pytest.approx(0.8215, abs=1e-3)

    @pytest.mark.parametrize("M", [0.1, 0.5, 0.9, 1.5, 3.0, 10.0])
    def test_inverse_roundtrip(self, M):
        p = cf.fanno_parameter(M)
        assert cf.mach_from_fanno_parameter(p, supersonic=M > 1) == pytest.approx(M, rel=1e-7)

    def test_inverse_rejects_beyond_limit(self):
        with pytest.raises(ValueError):
            cf.mach_from_fanno_parameter(0.9, supersonic=True)

    def test_subsonic_friction_accelerates_to_sonic(self):
        M2 = cf.fanno_exit(0.5, 0.5)
        assert 0.5 < M2 < 1.0
        assert cf.fanno_exit(0.5, cf.fanno_parameter(0.5)) == pytest.approx(1.0)

    def test_supersonic_friction_decelerates(self):
        M2 = cf.fanno_exit(3.0, 0.2)
        assert 1.0 < M2 < 3.0

    def test_choked_duct(self):
        with pytest.raises(ValueError, match="Choked"):
            cf.fanno_exit(0.5, 1.2)

    def test_exit_additivity(self):
        # marching in two legs must equal one long leg
        M_mid = cf.fanno_exit(0.3, 0.4)
        M_end_two = cf.fanno_exit(M_mid, 0.5)
        M_end_one = cf.fanno_exit(0.3, 0.9)
        assert M_end_two == pytest.approx(M_end_one, rel=1e-8)


# ------------------------------------------------- remaining input-validation paths
class TestInputValidation:
    def test_oblique_negative_theta(self):
        with pytest.raises(ValueError):
            cf.oblique_shock(2.0, theta_deg=-5.0)

    def test_negative_expansion_turn(self):
        with pytest.raises(ValueError):
            cf.prandtl_meyer_expansion(2.0, -1.0)

    def test_rayleigh_ratio_domain(self):
        with pytest.raises(ValueError):
            cf.mach_from_rayleigh_T0_ratio(1.5, supersonic=False)
        with pytest.raises(ValueError):
            cf.rayleigh_exit(0.5, -1.0)

    def test_fanno_negative_parameter(self):
        with pytest.raises(ValueError):
            cf.mach_from_fanno_parameter(-0.1, supersonic=False)
        with pytest.raises(ValueError):
            cf.fanno_exit(0.5, -0.1)

    def test_pm_inverse_extremely_close_to_limit(self):
        # bracket expansion must either succeed or fail with a clear ValueError, never hang
        try:
            M = cf.mach_from_pm_angle(cf.prandtl_meyer_max() - 1e-9)
            assert M > 1e3
        except ValueError:
            pass


# ------------------------------------------- regression: near-sonic inverse accuracy
class TestNearSonicRegression:
    """np.isclose(..., atol=1e-12) used the default rtol=1e-5 and returned M=1.0 for
    M = 1.001 (error 1e-3). Quantities such as A/A* - 1 are quadratic in (M - 1)."""

    @pytest.mark.parametrize("M", [1.0005, 1.001, 1.01, 0.999, 0.99])
    def test_area_inverse_near_sonic(self, M):
        A = cf.area_mach_ratio(M)
        assert cf.mach_from_area_ratio(A, supersonic=M > 1) == pytest.approx(M, rel=1e-6)

    @pytest.mark.parametrize("M", [1.0005, 1.001, 1.01, 0.999, 0.99])
    def test_rayleigh_inverse_near_sonic(self, M):
        r = cf.rayleigh_ratios(M).T0_T0star
        assert cf.mach_from_rayleigh_T0_ratio(r, supersonic=M > 1) == pytest.approx(M, rel=1e-6)

    def test_full_sweep_supersonic_area_inverse(self):
        worst = max(
            abs(cf.mach_from_area_ratio(cf.area_mach_ratio(m), True) - m) / m
            for m in np.linspace(1.001, 30, 120)
        )
        assert worst < 1e-8
