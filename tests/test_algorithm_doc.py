from pathlib import Path


def test_chinese_algorithm_flow_doc_covers_core_pipeline():
    root = Path(__file__).resolve().parents[1]
    doc_path = root / "docs" / "dp3_algorithm_flow_zh.md"

    assert doc_path.exists()
    doc = doc_path.read_text(encoding="utf-8")

    required_terms = [
        "optimize_dp3",
        "_optimize_grid",
        "z = s_dot^2",
        "C2LinearZ",
        "C3QuadraticSpeed",
        "C4CubicSpeed",
        "path_time_derivatives",
        "audit_constraints",
        "velocity-dependent torque constraint",
        "MujocoRobotDynamics",
        "run_dp3",
        "dp3-run",
        "TOPPRA",
        "DP2",
    ]
    for term in required_terms:
        assert term in doc
