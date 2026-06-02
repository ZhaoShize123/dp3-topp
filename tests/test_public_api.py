import json

import numpy as np

import dp3_topp as dp3


def _linear_path() -> dp3.PathData:
    s = np.linspace(0.0, 1.0, 5)
    q = s[:, None]
    return dp3.PathData(
        s=s,
        q=q,
        q_s=np.ones_like(q),
        q_ss=np.zeros_like(q),
        q_sss=np.zeros_like(q),
    )


def _wide_limits() -> dp3.ConstraintLimits:
    return dp3.ConstraintLimits(
        q_dot_abs=np.array([2.0]),
        q_ddot_abs=np.array([20.0]),
        q_jerk_abs=np.array([200.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([1000.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )


def test_root_package_exposes_low_level_algorithm_api():
    assert dp3.__version__ == "0.1.0"
    assert dp3.DP3Config(ns=6).ns == 6
    assert dp3.optimize_dp3 is not None
    assert dp3.optimize_dp2 is not None
    assert dp3.evaluate_trajectory_quantities is not None
    assert dp3.PathData is not None
    assert dp3.ConstraintLimits is not None


def test_run_dp3_high_level_api_returns_result_and_writes_artifacts(tmp_path):
    out_dir = tmp_path / "run"

    run = dp3.run_dp3(
        path=_linear_path(),
        limits=_wide_limits(),
        config=dp3.DP3Config(ns=5, nz=7, nch=5),
        out_dir=out_dir,
        constraint_check_points=11,
        time_samples=5,
    )

    assert run.status == 0
    assert run.result.feasible
    assert run.quantities.q.shape == (run.result.s.size, 1)
    assert run.summary["method"] == "DP3"
    assert run.summary["constraint_check_samples"] == 11
    assert run.summary_path == out_dir / "summary.json"
    assert json.loads(run.summary_path.read_text(encoding="utf-8"))["method"] == "DP3"
    assert (out_dir / "trajectory.csv").exists()
    assert (out_dir / "quantities.csv").exists()
    assert (out_dir / "time_quantities.csv").exists()
    assert (out_dir / "constraint_utilization.csv").exists()
    assert (out_dir / "constraint_violations.csv").exists()


def test_run_dp2_high_level_api_accepts_path_and_limits_files(tmp_path):
    path_csv = tmp_path / "path.csv"
    path_csv.write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.5,0.5,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [2.0]\n"
        "q_ddot_abs: [20.0]\n"
        "q_jerk_abs: [200.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [1000.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )

    run = dp3.run_dp2(
        path=path_csv,
        limits=limits_yaml,
        config=dp3.DP3Config(ns=5, nz=7, nch=5),
        jerk_limited="no",
    )

    assert run.status == 0
    assert run.result.method == "DP2"
    assert run.summary["reproduction_sources"]["path_source"] == str(path_csv)
    assert run.summary["reproduction_sources"]["limits"] == str(limits_yaml)
