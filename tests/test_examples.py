import importlib.util
from pathlib import Path


def test_library_call_demo_runs_and_writes_outputs(tmp_path):
    demo_path = Path("examples/library_call_demo.py")
    spec = importlib.util.spec_from_file_location("library_call_demo", demo_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    out_dir = tmp_path / "demo"
    status = module.main(["--out-dir", str(out_dir)])

    assert status == 0
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "trajectory.csv").exists()
    assert (out_dir / "quantities.csv").exists()
    assert (out_dir / "constraint_utilization.csv").exists()
