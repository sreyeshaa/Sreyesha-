"""Smoke tests for the command-line front end (main.py)."""

import os

import pytest

import main as cli


def test_validation_passes(capsys):
    assert cli.run_validation(verbose=True) is True
    assert "16/16 checks passed" in capsys.readouterr().out


def test_one_shot_summary(capsys):
    assert cli.main(["--mach", "2.0", "--no-plots"]) == 0
    out = capsys.readouterr().out
    assert "p2/p1 = 4.5000" in out and "nu = 26.3798" in out


def test_subsonic_summary_skips_shock(capsys):
    cli.main(["--mach", "0.5"])
    assert "skipped" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["--gamma", "0.8", "--mach", "2"], ["--mach", "-1"]])
def test_bad_arguments_exit_cleanly(argv):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2


def test_plots_written(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "FIG_DIR", str(tmp_path))
    assert cli.main(["--plots"]) == 0
    names = set(os.listdir(tmp_path))
    assert {"isentropic_curves.png", "normal_shock_curves.png", "prandtl_meyer_curve.png",
            "theta_beta_M_chart.png", "rayleigh_curves.png", "fanno_curves.png"} <= names


def test_interactive_menu_handles_errors(monkeypatch, capsys):
    # oblique shock past detachment -> message, not a crash; then quit
    inputs = iter(["2", "2", "25", "weak", "4", "0.5", "q"])
    monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
    cli.interactive_menu()
    out = capsys.readouterr().out
    assert "detaches" in out and "A/A* must be >= 1" in out


def test_interactive_menu_survives_eof(monkeypatch):
    def boom(*_):
        raise EOFError
    monkeypatch.setattr("builtins.input", boom)
    cli.interactive_menu()  # must return, not raise
