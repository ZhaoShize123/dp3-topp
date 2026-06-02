import pathlib

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


def test_pyproject_declares_build_backend_and_readme():
    root = pathlib.Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["build-system"]["build-backend"] == "setuptools.build_meta"
    assert "setuptools>=68" in pyproject["build-system"]["requires"]
    assert pyproject["project"]["readme"] == "README.md"
    assert (root / pyproject["project"]["readme"]).exists()
    assert pyproject["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]


def test_pyproject_keeps_console_scripts_for_cli_users():
    root = pathlib.Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    scripts = pyproject["project"]["scripts"]
    assert scripts["dp3-run"] == "dp3_topp.cli:run_main"
    assert scripts["dp3-plot"] == "dp3_topp.plotting:plot_main"
