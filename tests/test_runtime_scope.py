import importlib.util
from pathlib import Path

from dp3_topp import cli


def test_runtime_scope_excludes_report_tools_and_heavy_plot_dependencies():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "dp3-report" not in pyproject
    assert "dp3-plot" in pyproject
    assert "matplotlib" not in pyproject
    assert "pandas" not in pyproject
    assert not hasattr(cli, "report_main")
    assert importlib.util.find_spec("dp3_topp.reporting") is None
    assert importlib.util.find_spec("dp3_topp.plotting") is not None


def test_runtime_scope_declares_toppra_for_mvc_generation():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '"toppra"' in pyproject
