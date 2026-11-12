# Compressible Flow Calculator

Python library + command-line tool for one-dimensional compressible-flow relations
of a calorically perfect gas (default gamma = 1.4).

| Module | Contents |
|---|---|
| `compressible_flow.py` | Isentropic flow and A/A* (with inverse), normal shock (with inverse, Pitot), oblique shock (theta-beta-M, weak/strong, theta_max), Prandtl-Meyer expansion (with inverse and fan solver), Rayleigh flow, Fanno flow |
| `main.py` | CLI: validation, plots, one-shot summary, interactive menu |
| `test_compressible_flow.py` | 92 pytest tests against NACA 1135 / Anderson tables and physical identities |

## Quick start
```bash
pip install -r requirements.txt
python main.py                    # validate + plots + interactive menu
python main.py --mach 2.5         # one-shot summary
python main.py --validate         # self-check only
python -m pytest -q               # run the test-suite
```

## Library example
```python
import compressible_flow as cf

cf.normal_shock(2.0)                              # M2, p2/p1, rho2/rho1, T2/T1, p02/p01
cf.oblique_shock(2.0, theta_deg=10)               # weak solution; weak=False for strong
cf.prandtl_meyer_expansion(2.0, 10)               # M2 and static ratios after a 10-deg turn
cf.mach_from_area_ratio(1.6875, supersonic=True)  # -> 2.0
cf.fanno_exit(0.5, 0.5)                           # exit Mach, raises ValueError if choked
```
All angles are in degrees. Invalid physics (e.g. a normal shock at M<1, a wedge angle beyond
detachment, a choked duct) raises `ValueError` with an explanatory message.

## References
Anderson, *Modern Compressible Flow* (3rd ed., 2003); NACA Report 1135 (1953);
Liepmann & Roshko, *Elements of Gasdynamics* (1957).
