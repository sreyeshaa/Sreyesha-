from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import compressible_flow as cf

FIG_DIR = "figures"


# --------------------------------------------------------------------------- #
# Self-validation (asserts, so a regression fails loudly)
# --------------------------------------------------------------------------- #
def run_validation(verbose=True) -> bool:
    checks = []

    def check(label, computed, expected, tol):
        ok = abs(computed - expected) <= tol
        checks.append(ok)
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label:<34} computed={computed:.4f}  expected={expected:.4f}")

    if verbose:
        print("=" * 72)
        print("VALIDATION AGAINST PUBLISHED TABLES (gamma = 1.4)")
        print("=" * 72)

    s = cf.isentropic_ratios(2.0)
    check("Isentropic M=2   T0/T", s.T0_T, 1.8000, 5e-5)
    check("Isentropic M=2   p0/p", s.p0_p, 7.8244, 5e-4)
    check("Isentropic M=2   A/A*", cf.area_mach_ratio(2.0), 1.6875, 5e-5)

    n = cf.normal_shock(2.0)
    check("Normal shock M1=2  M2", n.M2, 0.5774, 5e-5)
    check("Normal shock M1=2  p2/p1", n.p2_p1, 4.5000, 5e-5)
    check("Normal shock M1=2  T2/T1", n.T2_T1, 1.6875, 5e-5)
    check("Normal shock M1=2  p02/p01", n.p02_p01, 0.7209, 5e-5)

    o = cf.oblique_shock(2.0, theta_deg=10.0)
    check("Oblique M=2,theta=10  beta", o.beta_deg, 39.31, 0.01)
    check("Oblique M=2,theta=10  M2", o.M2, 1.641, 2e-3)
    check("Oblique M=2,theta=10  p2/p1", o.p2_p1, 1.707, 2e-3)

    check("Prandtl-Meyer M=2  nu", cf.prandtl_meyer(2.0), 26.380, 2e-3)
    check("Inverse PM nu=26.38 -> M", cf.mach_from_pm_angle(26.380), 2.0, 1e-3)

    r = cf.rayleigh_ratios(2.0)
    check("Rayleigh M=2  T0/T0*", r.T0_T0star, 0.7934, 5e-5)
    check("Rayleigh M=2  p/p*", r.p_pstar, 0.3636, 5e-5)

    f = cf.fanno_ratios(2.0)
    check("Fanno M=2  fL*/D", f.fLstar_D, 0.3050, 5e-5)
    check("Fanno M=2  p/p*", f.p_pstar, 0.4082, 5e-5)

    ok = all(checks)
    if verbose:
        print("-" * 72)
        print(f"  {sum(checks)}/{len(checks)} checks passed")
        print("=" * 72 + "\n")
    return ok


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _finish(fig, plt, name, show):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def make_plots(gamma=1.4, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = dict(linestyle="--", alpha=0.5)

    # 1. isentropic ratios + A/A*
    M = np.linspace(0.05, 5, 400)
    s = cf.isentropic_ratios(M, gamma)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(M, s.T0_T, label=r"$T_0/T$")
    ax.plot(M, s.p0_p, label=r"$p_0/p$")
    ax.plot(M, s.rho0_rho, label=r"$\rho_0/\rho$")
    ax.plot(M, cf.area_mach_ratio(M, gamma), "k--", label=r"$A/A^*$")
    ax.set_yscale("log")
    ax.set(xlabel="Mach number", ylabel="Ratio (log scale)",
           title=f"Isentropic flow ratios vs. Mach number ($\\gamma$={gamma})")
    ax.legend(); ax.grid(True, which="both", **grid)
    _finish(fig, plt, "isentropic_curves.png", show)

    # 2. normal shock
    M1 = np.linspace(1, 5, 400)
    n = cf.normal_shock(M1, gamma)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(M1, n.M2, label=r"$M_2$")
    ax.plot(M1, n.p2_p1, label=r"$p_2/p_1$")
    ax.plot(M1, n.T2_T1, label=r"$T_2/T_1$")
    ax.plot(M1, n.rho2_rho1, label=r"$\rho_2/\rho_1$")
    ax.plot(M1, n.p02_p01, label=r"$p_{02}/p_{01}$")
    ax.set(xlabel="Upstream Mach number $M_1$", ylabel="Ratio / downstream Mach number",
           title="Normal-shock relations vs. upstream Mach number")
    ax.legend(); ax.grid(True, **grid)
    _finish(fig, plt, "normal_shock_curves.png", show)

    # 3. Prandtl-Meyer
    M = np.linspace(1, 10, 400)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(M, cf.prandtl_meyer(M, gamma), color="darkorange")
    ax.axhline(cf.prandtl_meyer_max(gamma), color="gray", ls=":",
               label=rf"$\nu_{{max}}$ = {cf.prandtl_meyer_max(gamma):.2f}$^\circ$")
    ax.set(xlabel="Mach number", ylabel=r"Prandtl-Meyer angle $\nu$ (deg)",
           title="Prandtl-Meyer function vs. Mach number")
    ax.legend(); ax.grid(True, **grid)
    _finish(fig, plt, "prandtl_meyer_curve.png", show)

    # 4. theta-beta-M chart
    fig, ax = plt.subplots(figsize=(8, 6))
    for Mach in (1.2, 1.5, 2, 3, 5, 10):
        mu = float(cf.mach_angle_deg(Mach))
        beta = np.linspace(mu, 90, 500)
        theta = cf.theta_from_beta(Mach, beta, gamma)
        line, = ax.plot(theta, beta, label=f"M$_1$={Mach}")
        tmax, bmax = cf.max_deflection_angle(Mach, gamma)
        ax.plot(tmax, bmax, "o", color=line.get_color(), ms=4)
    ax.set(xlabel=r"Deflection angle $\theta$ (deg)", ylabel=r"Wave angle $\beta$ (deg)",
           title=r"$\theta$-$\beta$-M chart (dots: $\theta_{max}$)", xlim=(0, 50), ylim=(0, 90))
    ax.legend(); ax.grid(True, **grid)
    _finish(fig, plt, "theta_beta_M_chart.png", show)

    # 5. Rayleigh
    M = np.linspace(0.05, 5, 400)
    r = cf.rayleigh_ratios(M, gamma)
    fig, ax = plt.subplots(figsize=(8, 6))
    for y, lab in ((r.T_Tstar, r"$T/T^*$"), (r.p_pstar, r"$p/p^*$"), (r.T0_T0star, r"$T_0/T_0^*$"),
                   (r.p0_p0star, r"$p_0/p_0^*$"), (r.V_Vstar, r"$V/V^*$")):
        ax.plot(M, y, label=lab)
    ax.set(xlabel="Mach number", ylabel="Ratio to sonic state", ylim=(0, 3),
           title="Rayleigh flow (heat addition, constant area)")
    ax.legend(); ax.grid(True, **grid)
    _finish(fig, plt, "rayleigh_curves.png", show)

    # 6. Fanno
    M = np.linspace(0.05, 5, 400)
    f = cf.fanno_ratios(M, gamma)
    fig, ax = plt.subplots(figsize=(8, 6))
    for y, lab in ((f.T_Tstar, r"$T/T^*$"), (f.p_pstar, r"$p/p^*$"), (f.p0_p0star, r"$p_0/p_0^*$"),
                   (f.V_Vstar, r"$V/V^*$"), (f.fLstar_D, r"$fL^*/D$")):
        ax.plot(M, y, label=lab)
    ax.set(xlabel="Mach number", ylabel="Ratio to sonic state", ylim=(0, 4),
           title="Fanno flow (adiabatic, constant area, friction)")
    ax.legend(); ax.grid(True, **grid)
    _finish(fig, plt, "fanno_curves.png", show)

    print(f"Saved 6 figures to ./{FIG_DIR}/")


# --------------------------------------------------------------------------- #
# Printing helpers
# --------------------------------------------------------------------------- #
def print_summary(M, gamma=1.4):
    print(f"\n--- Isentropic relations (M = {M:g}, gamma = {gamma:g}) ---")
    s = cf.isentropic_ratios(M, gamma)
    print(f"  T0/T = {s.T0_T:.4f}   p0/p = {s.p0_p:.4f}   rho0/rho = {s.rho0_rho:.4f}")
    if M > 0:
        print(f"  A/A* = {cf.area_mach_ratio(M, gamma):.4f}")
    if M >= 1:
        n = cf.normal_shock(M, gamma)
        print(f"\n--- Normal shock (M1 = {M:g}) ---")
        print(f"  M2 = {n.M2:.4f}   p2/p1 = {n.p2_p1:.4f}   rho2/rho1 = {n.rho2_rho1:.4f}")
        print(f"  T2/T1 = {n.T2_T1:.4f}   p02/p01 = {n.p02_p01:.4f}   "
              f"(s2-s1)/R = {cf.shock_entropy_rise(M, gamma):.4f}")
        print(f"  Pitot p02/p1 = {cf.pitot_pressure_ratio(M, gamma):.4f}")
        print(f"\n--- Prandtl-Meyer (M = {M:g}) ---")
        print(f"  nu = {cf.prandtl_meyer(M, gamma):.4f} deg   (nu_max = {cf.prandtl_meyer_max(gamma):.2f} deg)")
        if M > 1:
            print(f"  Mach angle mu = {cf.mach_angle_deg(M):.3f} deg   "
                  f"theta_max (oblique) = {cf.max_deflection_angle(M, gamma)[0]:.3f} deg")
    else:
        print("\n(Shock and Prandtl-Meyer relations need M >= 1 - skipped.)")
    if M > 0:
        r = cf.rayleigh_ratios(M, gamma)
        f = cf.fanno_ratios(M, gamma)
        print(f"\n--- Rayleigh flow (M = {M:g}) ---")
        print(f"  T/T* = {r.T_Tstar:.4f}   p/p* = {r.p_pstar:.4f}   T0/T0* = {r.T0_T0star:.4f}   p0/p0* = {r.p0_p0star:.4f}")
        print(f"\n--- Fanno flow (M = {M:g}) ---")
        print(f"  T/T* = {f.T_Tstar:.4f}   p/p* = {f.p_pstar:.4f}   p0/p0* = {f.p0_p0star:.4f}   fL*/D = {f.fLstar_D:.4f}")
    print()


def _ask_float(prompt, default=None):
    raw = input(prompt).strip()
    if not raw and default is not None:
        return default
    return float(raw)


def interactive_menu(gamma=1.4):
    menu = (
        "\n" + "=" * 60 + "\nCOMPRESSIBLE FLOW CALCULATOR\n" + "=" * 60 + "\n"
        "  1  Full summary at one Mach number\n"
        "  2  Oblique shock (M1 + wedge angle theta)\n"
        "  3  Prandtl-Meyer expansion (M1 + turn angle)\n"
        "  4  Isentropic: Mach number from A/A*\n"
        "  5  Rayleigh duct: exit Mach for T02/T01\n"
        "  6  Fanno duct: exit Mach for f L/D\n"
        "  7  Change gamma (currently {g:g})\n"
        "  q  Quit\n"
    )
    while True:
        print(menu.format(g=gamma))
        try:
            choice = input("Choose an option: ").strip().lower()
            if choice in ("q", "quit", "exit", ""):
                break
            if choice == "1":
                print_summary(_ask_float("Mach number: "), gamma)
            elif choice == "2":
                M1 = _ask_float("Upstream Mach number M1: ")
                th = _ask_float("Deflection angle theta (deg): ")
                branch = input("Solution branch - weak or strong [weak]: ").strip().lower() or "weak"
                o = cf.oblique_shock(M1, theta_deg=th, weak=(branch != "strong"), gamma=gamma)
                print(f"\n  beta = {o.beta_deg:.3f} deg   Mn1 = {o.Mn1:.4f}   M2 = {o.M2:.4f}")
                print(f"  p2/p1 = {o.p2_p1:.4f}   rho2/rho1 = {o.rho2_rho1:.4f}   "
                      f"T2/T1 = {o.T2_T1:.4f}   p02/p01 = {o.p02_p01:.4f}")
            elif choice == "3":
                M1 = _ask_float("Upstream Mach number M1: ")
                turn = _ask_float("Turn angle (deg): ")
                e = cf.prandtl_meyer_expansion(M1, turn, gamma)
                print(f"\n  M2 = {e.M2:.4f}   nu1 = {e.nu1_deg:.3f}   nu2 = {e.nu2_deg:.3f} deg")
                print(f"  p2/p1 = {e.p2_p1:.4f}   rho2/rho1 = {e.rho2_rho1:.4f}   T2/T1 = {e.T2_T1:.4f}")
                print(f"  fan between Mach waves {e.mu1_deg:.2f} deg and {e.mu2_deg:.2f} deg")
            elif choice == "4":
                A = _ask_float("A/A* (>= 1): ")
                print(f"\n  Subsonic   M = {cf.mach_from_area_ratio(A, supersonic=False, gamma=gamma):.4f}")
                print(f"  Supersonic M = {cf.mach_from_area_ratio(A, supersonic=True, gamma=gamma):.4f}")
            elif choice == "5":
                M1 = _ask_float("Inlet Mach number: ")
                ratio = _ask_float("Stagnation-temperature ratio T02/T01: ")
                print(f"\n  Exit Mach number = {cf.rayleigh_exit(M1, ratio, gamma):.4f}")
            elif choice == "6":
                M1 = _ask_float("Inlet Mach number: ")
                fld = _ask_float("Friction parameter f L/D (Darcy f): ")
                print(f"\n  Exit Mach number = {cf.fanno_exit(M1, fld, gamma):.4f}")
            elif choice == "7":
                new = _ask_float("New gamma (> 1): ")
                cf.isentropic_ratios(1.0, new)  # validates
                gamma = new
            else:
                print("Unknown option.")
        except ValueError as exc:
            print(f"\n  Input error: {exc}")
        except (EOFError, KeyboardInterrupt):
            print()
            break
    print("Exiting. Figures (if generated) are in ./" + FIG_DIR + "/")


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Compressible flow calculator (calorically perfect gas).")
    ap.add_argument("--mach", type=float, help="print a one-shot summary at this Mach number")
    ap.add_argument("--gamma", type=float, default=1.4, help="ratio of specific heats (default 1.4)")
    ap.add_argument("--validate", action="store_true", help="run validation only")
    ap.add_argument("--plots", action="store_true", help="generate figures only")
    ap.add_argument("--no-plots", action="store_true", help="skip figure generation")
    ap.add_argument("--no-interactive", action="store_true", help="skip the interactive menu")
    ap.add_argument("--show", action="store_true", help="display plot windows (blocks until closed)")
    a = ap.parse_args(argv)

    try:
        cf.isentropic_ratios(1.0, a.gamma)  # validate gamma early
    except ValueError as exc:
        ap.error(str(exc))

    if a.validate:
        return 0 if run_validation() else 1
    if a.plots:
        make_plots(a.gamma, a.show)
        return 0
    if a.mach is not None:
        try:
            print_summary(a.mach, a.gamma)
        except ValueError as exc:
            ap.error(str(exc))
        return 0

    ok = run_validation()
    if not a.no_plots:
        make_plots(a.gamma, a.show)
    if not a.no_interactive:
        interactive_menu(a.gamma)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
