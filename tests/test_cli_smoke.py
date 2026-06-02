import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import dp3_topp.cli as cli
from dp3_topp.cli import (
    _required_time_scale,
    _time_scale_covers_violations,
    build_path_main,
    default_dyn_dir,
    run_main,
    validate_data,
)
from dp3_topp.constraints import ConstraintAudit, ConstraintViolation
from dp3_topp.optimizer import TrajectoryResult
from dp3_topp.path_data import PathData


def test_paper_like_config_points_to_existing_t12a_assets():
    config_path = Path("configs/paper_like.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert (config_path.parent / config["input"]).resolve().exists()
    assert (config_path.parent / config["model"]).resolve().exists()
    assert (config_path.parent / config["limits"]).resolve().exists()
    assert config["ee_site"] == "tcp"
    assert config["require_ik_convergence"] is True
    assert "joint_path_out" in config
    assert (config_path.parent / config["joint_path_out"]).resolve().parent == (config_path.parent / config["out_dir"]).resolve()
    assert config["constraint_check_points"] == 600
    assert config["require_full_reproduction_data"] is True


def test_time_scale_report_treats_acceleration_as_scalable():
    max_utilization = {
        "q_dot": 0.5,
        "q_ddot": 4.0,
        "q_jerk": 0.5,
        "tau": 0.8,
        "tau_rate": 0.5,
        "mechanical_power": 0.5,
    }

    assert _required_time_scale(max_utilization) == pytest.approx(0.5)
    assert _time_scale_covers_violations(max_utilization)


def test_active_constraint_percentages_count_active_path_samples_not_axis_entries():
    quantities = SimpleNamespace(
        q_dot_utilization=np.array([[1.0, 0.0], [0.0, 0.0]]),
        q_ddot_utilization=np.zeros((2, 2)),
        q_jerk_utilization=np.zeros((2, 2)),
        tau_utilization=np.zeros((2, 2)),
        tau_rate_utilization=np.zeros((2, 2)),
        mechanical_power_utilization=np.zeros(2),
        q_position_utilization=None,
    )

    percentages = cli._active_constraint_percentages(quantities, threshold=0.99)

    assert percentages["q_dot"] == pytest.approx(50.0)


def test_most_restrictive_constraint_percentages_ignore_inactive_samples():
    quantities = SimpleNamespace(
        q_dot_utilization=np.array([0.1, 0.2, 0.3]),
        q_ddot_utilization=np.array([0.0, 0.1, 0.2]),
        q_jerk_utilization=np.zeros(3),
        tau_utilization=np.zeros(3),
        tau_rate_utilization=np.zeros(3),
        mechanical_power_utilization=np.zeros(3),
        q_position_utilization=None,
    )

    percentages = cli._most_restrictive_constraint_percentages(quantities)

    assert percentages == {
        "q_dot": 0.0,
        "q_ddot": 0.0,
        "q_jerk": 0.0,
        "tau": 0.0,
        "tau_rate": 0.0,
        "mechanical_power": 0.0,
    }


def test_run_main_omits_scaled_duration_for_non_time_scalable_dense_violation(tmp_path, monkeypatch):
    path_csv = tmp_path / "path.csv"
    path_csv.write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [10.0]\n"
        "q_ddot_abs: [10.0]\n"
        "q_jerk_abs: [10.0]\n"
        "tau_abs: [10.0]\n"
        "tau_rate_abs: [10.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    audit = ConstraintAudit(
        ok=False,
        max_utilization={"q_position": 1.5},
        violations=[
            ConstraintViolation(
                quantity="q_position",
                sample=1,
                axis=0,
                value=1.5,
                limit=1.0,
                utilization=1.5,
            )
        ],
    )
    result = TrajectoryResult(
        feasible=False,
        t=np.array([0.0, 1.0]),
        s=np.array([0.0, 1.0]),
        z=np.array([1.0, 1.0]),
        z_s=np.zeros(2),
        z_ss=np.zeros(2),
        total_time=1.0,
        audit=audit,
        grid_s=np.array([0.0, 1.0]),
        grid_z=np.zeros((2, 1)),
        policy=np.zeros((2, 1), dtype=np.int32),
        segment_kinds=["C2"],
        segment_profiles=None,
        objective_cost=1.0,
        objective_time_cost=1.0,
        objective_drive_power_cost=0.0,
    )
    quantities = SimpleNamespace(
        q=np.zeros((2, 1)),
        q_dot=np.zeros((2, 1)),
        q_ddot=np.zeros((2, 1)),
        q_jerk=np.zeros((2, 1)),
        tau=np.zeros((2, 1)),
        tau_rate=np.zeros((2, 1)),
        mechanical_power=np.zeros(2),
        q_dot_utilization=np.zeros((2, 1)),
        q_ddot_utilization=np.zeros((2, 1)),
        q_jerk_utilization=np.zeros((2, 1)),
        tau_utilization=np.zeros((2, 1)),
        tau_rate_utilization=np.zeros((2, 1)),
        mechanical_power_utilization=np.zeros(2),
        q_position_utilization=np.array([[0.0], [1.5]]),
        drive_power=None,
    )
    monkeypatch.setattr(cli, "optimize_dp3", lambda **_: result)
    monkeypatch.setattr(cli, "_constraint_check_result", lambda **_: (result, quantities, audit, "test_dense"))
    out_dir = tmp_path / "out"

    status = run_main(
        [
            "--path-csv",
            str(path_csv),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "3",
            "--nz",
            "3",
            "--nch",
            "2",
        ]
    )

    assert status == 1
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["executable_with_st"] == "no"
    assert summary["required_time_scale_st"] == "-"
    assert summary["te_scale"] == "-"
    assert any(violation["quantity"] == "q_position" for violation in summary["violations"])


def test_validate_data_accepts_selected_t12a_limits_without_inventing_values():
    report = validate_data(
        robot_dir=Path("dyn - 副本/models/T12A"),
        paths_dir=Path("dyn - 副本"),
        limits_path=Path("dyn - 副本/models/T12A/limits.yaml"),
    )

    assert report.ok
    assert report.missing == []
    assert report.full_reproduction_ready is True
    assert report.reproduction_sources["limits_dof"] == 6


def test_default_data_directory_points_to_existing_t12a_assets():
    dyn_dir = default_dyn_dir()

    assert (dyn_dir / "models" / "T12A" / "T12A-14.xml").exists()
    assert (dyn_dir / "Offline_Traj.txt").exists()


def test_build_path_main_writes_toppra_joint_path_csv(tmp_path):
    input_path = tmp_path / "Offline_Traj.txt"
    input_path.write_text(
        "100 100 100 0 0 0\n"
        "101 100 100 0 0 0\n"
        "102 100 100 0 0 0\n",
        encoding="utf-8",
    )
    out_csv = tmp_path / "joint_path.csv"
    summary_json = tmp_path / "joint_path_summary.json"

    status = build_path_main(
        [
            "--input",
            str(input_path),
            "--model",
            str(default_dyn_dir() / "models" / "T12A" / "T12A-14.xml"),
            "--out",
            str(out_csv),
            "--summary-out",
            str(summary_json),
            "--ik-max-iters",
            "1",
            "--ik-pos-tol",
            "0.5",
            "--ik-ori-tol",
            "0.25",
            "--ik-damping",
            "0.002",
            "--ik-step-scale",
            "0.4",
            "--ik-orientation-weight",
            "0.2",
            "--tcp-offset-x",
            "0.01",
            "--tcp-offset-y",
            "-0.02",
            "--tcp-offset-z",
            "0.03",
        ]
    )

    assert status == 0
    header = out_csv.read_text(encoding="utf-8").splitlines()[0]
    assert header == (
        "s,q1,q2,q3,q4,q5,q6,"
        "dq1_ds,dq2_ds,dq3_ds,dq4_ds,dq5_ds,dq6_ds,"
        "d2q1_ds2,d2q2_ds2,d2q3_ds2,d2q4_ds2,d2q5_ds2,d2q6_ds2,"
        "d3q1_ds3,d3q2_ds3,d3q3_ds3,d3q4_ds3,d3q5_ds3,d3q6_ds3"
    )
    path = PathData.from_csv(out_csv)
    assert path.samples == 3
    assert path.dof == 6
    assert np.all(np.diff(path.s) > 0.0)
    assert np.all(np.isfinite(path.q))
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["samples"] == 3
    assert summary["dof"] == 6
    assert summary["joint_path_csv"] == str(out_csv)
    sources = summary["reproduction_sources"]
    assert sources["path_source"] == str(input_path)
    assert sources["joint_path_source"] == "generated_from_cartesian"
    assert sources["joint_path_csv"] == str(out_csv)
    assert sources["model"] == str(default_dyn_dir() / "models" / "T12A" / "T12A-14.xml")
    assert sources["dynamics_backend"] == "mujoco"
    assert sources["path_dof"] == 6
    assert sources["model_dof"] == 6
    assert summary["ik"]["settings"]["tcp_offset"] == [0.01, -0.02, 0.03]


def test_build_path_main_accepts_mujoco_site_target(tmp_path):
    poses = tmp_path / "poses.txt"
    poses.write_text(
        "1.086442 0.226425 0.657657 0 0 0\n"
        "1.089498 0.275344 0.694959 0 0 0\n",
        encoding="utf-8",
    )
    out_csv = tmp_path / "joint_path.csv"
    summary_json = tmp_path / "summary.json"

    status = build_path_main(
        [
            "--input",
            str(poses),
            "--model",
            str(default_dyn_dir() / "models" / "T12A" / "T12A-14.xml"),
            "--ee-site",
            "tcp",
            "--out",
            str(out_csv),
            "--summary-out",
            str(summary_json),
            "--ik-orientation-weight",
            "0",
            "--ik-pos-tol",
            "1e-8",
        ]
    )

    assert status == 0
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["ik"]["settings"]["ee_site"] == "tcp"
    assert summary["ik"]["converged"] is True


def test_build_path_main_loads_yaml_config_for_paper_workflow(tmp_path):
    input_path = tmp_path / "Offline_Traj.txt"
    input_lines = (default_dyn_dir() / "Offline_Traj.txt").read_text(encoding="utf-8").splitlines()[:3]
    input_path.write_text("\n".join(input_lines) + "\n", encoding="utf-8")
    out_csv = tmp_path / "configured_joint_path.csv"
    summary_json = tmp_path / "configured_joint_path_summary.json"
    config_yaml = tmp_path / "build_path.yaml"
    config_yaml.write_text(
        f"input: {input_path.as_posix()}\n"
        f"model: {(default_dyn_dir() / 'models' / 'T12A' / 'T12A-14.xml').as_posix()}\n"
        "ee_body: link_6\n"
        "ee_site: tcp\n"
        f"joint_path_out: {out_csv.as_posix()}\n"
        f"summary_out: {summary_json.as_posix()}\n"
        "ik_max_iters: 200\n"
        "ik_pos_tol: 1.0e-4\n"
        "ik_ori_tol: 1.0e-3\n"
        "ik_damping: 1.0e-4\n"
        "ik_step_scale: 0.6\n"
        "ik_orientation_weight: 0.0\n"
        "tcp_offset_x: 0.0\n"
        "tcp_offset_y: 0.0\n"
        "tcp_offset_z: 0.0\n"
        "require_ik_convergence: true\n",
        encoding="utf-8",
    )

    status = build_path_main(["--config", str(config_yaml)])

    assert status == 0
    path = PathData.from_csv(out_csv)
    assert path.samples == 3
    assert path.dof == 6
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["input"] == str(input_path)
    assert summary["joint_path_csv"] == str(out_csv)
    assert summary["ik"]["converged"] is True
    assert summary["ik"]["settings"]["ee_site"] == "tcp"
    assert summary["ik"]["settings"]["orientation_weight"] == 0.0


def test_run_main_executes_synthetic_joint_path(tmp_path):
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
        "q_dot_abs: [1.0]\n"
        "q_ddot_abs: [5.0]\n"
        "q_jerk_abs: [50.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "run"

    status = run_main(
        [
            "--path-csv",
            str(path_csv),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "5",
            "--nz",
            "7",
            "--nch",
            "5",
            "--constraint-check-points",
            "31",
            "--time-samples",
            "9",
        ]
    )

    assert status == 0
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "trajectory.csv").exists()
    assert (out_dir / "quantities.csv").exists()
    assert (out_dir / "time_quantities.csv").exists()
    assert (out_dir / "constraint_utilization.csv").exists()
    assert (out_dir / "constraint_violations.csv").exists()
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert isinstance(summary["cpu_time_s"], float)
    assert summary["cpu_time_s"] > 0.0
    assert summary["segment_kinds"] == ["C2", "C4", "C3", "C2"]
    assert summary["segment_count"] == 4
    assert summary["constraint_check_points"] == 31
    assert summary["constraint_check_samples"] == 31
    assert summary["time_samples"] == 9
    assert summary["time_quantities_csv"] == "time_quantities.csv"
    assert summary["constraint_check_source"] == "dense_segment_profiles"
    assert summary["constraint_utilization_csv"] == "constraint_utilization.csv"
    assert summary["constraint_violations_csv"] == "constraint_violations.csv"
    assert summary["active_constraint_threshold"] == 0.99
    assert set(summary["active_constraint_percent"]) == {
        "q_dot",
        "q_ddot",
        "q_jerk",
        "tau",
        "tau_rate",
        "mechanical_power",
    }
    assert all(0.0 <= value <= 100.0 for value in summary["active_constraint_percent"].values())
    assert set(summary["most_restrictive_constraint_percent"]) == {
        "q_dot",
        "q_ddot",
        "q_jerk",
        "tau",
        "tau_rate",
        "mechanical_power",
    }
    assert max(summary["active_constraint_percent"].values()) == 0.0
    assert sum(summary["most_restrictive_constraint_percent"].values()) == 0.0
    trajectory_header = (out_dir / "trajectory.csv").read_text(encoding="utf-8").splitlines()[0]
    quantities_header = (out_dir / "quantities.csv").read_text(encoding="utf-8").splitlines()[0]
    time_rows = list(csv.DictReader((out_dir / "time_quantities.csv").open(newline="", encoding="utf-8")))
    utilization_rows = list(csv.DictReader((out_dir / "constraint_utilization.csv").open(newline="", encoding="utf-8")))
    violation_rows = list(csv.DictReader((out_dir / "constraint_violations.csv").open(newline="", encoding="utf-8")))
    assert trajectory_header == "t,s,z,z_s,z_ss"
    assert quantities_header == "t,s,z,z_s,z_ss,q1,q_dot1,q_ddot1,q_jerk1,tau1,tau_rate1,mechanical_power"
    assert len(time_rows) == 9
    assert len(utilization_rows) == 31
    assert set(utilization_rows[0]) == {
        "t",
        "s",
        "z",
        "q_dot",
        "q_ddot",
        "q_jerk",
        "tau",
        "tau_rate",
        "mechanical_power",
    }
    assert all(float(row["q_dot"]) >= 0.0 for row in utilization_rows)
    assert violation_rows == []
    assert float(time_rows[0]["t"]) == 0.0
    assert float(time_rows[-1]["t"]) == pytest.approx(summary["total_time"])


def test_run_main_path_csv_uses_configured_mujoco_dynamics(tmp_path):
    model_xml = tmp_path / "one_dof.xml"
    model_xml.write_text(
        '<mujoco model="one_dof">\n'
        '  <compiler angle="radian"/>\n'
        '  <option gravity="0 0 0"/>\n'
        '  <worldbody>\n'
        '    <body name="link1" pos="0 0 0">\n'
        '      <joint name="joint1" type="hinge" axis="0 0 1" limited="true" range="-10 10"/>\n'
        '      <geom type="capsule" fromto="0 0 0 1 0 0" size="0.05" density="1000"/>\n'
        "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n",
        encoding="utf-8",
    )
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
        "q_jerk_abs: [1000.0]\n"
        "tau_abs: [1000.0]\n"
        "tau_rate_abs: [10000.0]\n"
        "mechanical_power:\n"
        "  lower: -10000.0\n"
        "  upper: 10000.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "run"

    status = run_main(
        [
            "--path-csv",
            str(path_csv),
            "--model",
            str(model_xml),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "5",
            "--nz",
            "9",
            "--nch",
            "5",
        ]
    )

    assert status == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    sources = summary["reproduction_sources"]
    assert sources["path_source"] == str(path_csv)
    assert sources["joint_path_source"] == "path_csv"
    assert sources["joint_path_csv"] == str(path_csv)
    assert sources["limits"] == str(limits_yaml)
    assert sources["model"] == str(model_xml)
    assert sources["dynamics_backend"] == "mujoco"
    assert sources["path_dof"] == 1
    assert sources["limits_dof"] == 1
    assert sources["model_dof"] == 1
    rows = list(csv.DictReader((out_dir / "quantities.csv").open(newline="", encoding="utf-8")))
    tau = np.array([float(row["tau1"]) for row in rows])
    assert np.max(np.abs(tau)) > 1e-6


def test_run_main_uses_mujoco_joint_ranges_as_position_limits_when_yaml_omits_them(tmp_path):
    model_xml = tmp_path / "one_dof.xml"
    model_xml.write_text(
        '<mujoco model="one_dof">\n'
        '  <compiler angle="radian"/>\n'
        '  <option gravity="0 0 0"/>\n'
        '  <worldbody>\n'
        '    <body name="link1" pos="0 0 0">\n'
        '      <joint name="joint1" type="hinge" axis="0 0 1" limited="true" range="-0.25 0.25"/>\n'
        '      <geom type="capsule" fromto="0 0 0 1 0 0" size="0.05" density="1000"/>\n'
        "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n",
        encoding="utf-8",
    )
    path_csv = tmp_path / "path.csv"
    path_csv.write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,-0.2,0.4,0.0,0.0\n"
        "0.5,0.0,0.4,0.0,0.0\n"
        "1.0,0.2,0.4,0.0,0.0\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [2.0]\n"
        "q_ddot_abs: [20.0]\n"
        "q_jerk_abs: [1000.0]\n"
        "tau_abs: [1000.0]\n"
        "tau_rate_abs: [10000.0]\n"
        "mechanical_power:\n"
        "  lower: -10000.0\n"
        "  upper: 10000.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "run"

    status = run_main(
        [
            "--path-csv",
            str(path_csv),
            "--model",
            str(model_xml),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "5",
            "--nz",
            "9",
            "--nch",
            "5",
            "--constraint-check-points",
            "11",
        ]
    )

    assert status == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["max_utilization"]["q_position"] == pytest.approx(0.8)
    assert "q_position" in summary["active_constraint_percent"]
    utilization_header = (out_dir / "constraint_utilization.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "q_position" in utilization_header.split(",")


def test_run_main_rejects_q_position_limits_wider_than_mujoco_joint_ranges(tmp_path):
    model_xml = tmp_path / "one_axis.xml"
    model_xml.write_text(
        '<mujoco model="one_axis">\n'
        '  <compiler angle="radian"/>\n'
        '  <option gravity="0 0 0"/>\n'
        '  <worldbody>\n'
        '    <body name="link1" pos="0 0 0">\n'
        '      <joint name="joint1" type="hinge" axis="0 0 1" limited="true" range="-0.25 0.25"/>\n'
        '      <geom type="capsule" fromto="0 0 0 1 0 0" size="0.05" density="1000"/>\n'
        "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n",
        encoding="utf-8",
    )
    path_csv = tmp_path / "path.csv"
    path_csv.write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,0.0,0.0,0.0\n"
        "1.0,0.0,0.0,0.0,0.0\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_position:\n"
        "  lower: [-1.0]\n"
        "  upper: [1.0]\n"
        "q_dot_abs: [2.0]\n"
        "q_ddot_abs: [20.0]\n"
        "q_jerk_abs: [1000.0]\n"
        "tau_abs: [1000.0]\n"
        "tau_rate_abs: [10000.0]\n"
        "mechanical_power:\n"
        "  lower: -10000.0\n"
        "  upper: 10000.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "run"

    status = run_main(
        [
            "--path-csv",
            str(path_csv),
            "--model",
            str(model_xml),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "4",
            "--nz",
            "3",
            "--nch",
            "2",
        ]
    )

    assert status == 2
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["feasible"] is False
    assert "q_position" in summary["error"]
    assert "MuJoCo joint range" in summary["error"]


def test_run_main_quantities_include_drive_power_with_motor_model(tmp_path):
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
        "q_jerk_abs: [100.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n"
        "friction:\n"
        "  coulomb: [1.0]\n"
        "  viscous: [0.0]\n"
        "motor:\n"
        "  gear_ratio: [2.0]\n"
        "  torque_constant: [0.5]\n"
        "  stator_resistance: [1.0]\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "run"

    status = run_main(
        [
            "--path-csv",
            str(path_csv),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "5",
            "--nz",
            "7",
            "--nch",
            "5",
            "--k2",
            "1.0",
        ]
    )

    assert status == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["objective_time_cost"] == pytest.approx(summary["total_time"])
    assert summary["objective_drive_power_cost"] > 0.0
    assert summary["objective_cost"] == pytest.approx(summary["objective_time_cost"] + summary["objective_drive_power_cost"])
    rows = list(csv.DictReader((out_dir / "quantities.csv").open(newline="", encoding="utf-8")))
    assert "drive_power" in rows[0]
    assert any(float(row["drive_power"]) > 0.0 for row in rows)


def test_run_main_executes_t12a_offline_traj_prefix_with_mujoco_dynamics(tmp_path):
    input_path = tmp_path / "Offline_Traj_prefix.txt"
    input_lines = (default_dyn_dir() / "Offline_Traj.txt").read_text(encoding="utf-8").splitlines()[:5]
    input_path.write_text("\n".join(input_lines) + "\n", encoding="utf-8")
    repeated = ", ".join(["1e18"] * 6)
    limits_yaml = tmp_path / "wide_limits.yaml"
    limits_yaml.write_text(
        f"q_dot_abs: [{repeated}]\n"
        f"q_ddot_abs: [{repeated}]\n"
        f"q_jerk_abs: [{repeated}]\n"
        f"tau_abs: [{repeated}]\n"
        f"tau_rate_abs: [{repeated}]\n"
        "mechanical_power:\n"
        "  lower: -1e30\n"
        "  upper: 1e30\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "t12a-prefix-run"
    joint_path_out = tmp_path / "t12a_prefix_joint_path.csv"
    config_yaml = tmp_path / "t12a_prefix.yaml"
    config_yaml.write_text(
        "method: dp3\n"
        f"input: {input_path.as_posix()}\n"
        f"model: {(default_dyn_dir() / 'models' / 'T12A' / 'T12A-14.xml').as_posix()}\n"
        "ee_body: link_6\n"
        "ee_site: tcp\n"
        f"limits: {limits_yaml.as_posix()}\n"
        f"out_dir: {out_dir.as_posix()}\n"
        f"joint_path_out: {joint_path_out.as_posix()}\n"
        "ns: 4\n"
        "nz: 5\n"
        "nch: 3\n"
        "z_max: 1.0e-2\n"
        "constraint_check_points: 11\n"
        "time_samples: 5\n"
        "ik_max_iters: 200\n"
        "ik_pos_tol: 1.0e-4\n"
        "ik_ori_tol: 1.0e-3\n"
        "ik_damping: 1.0e-4\n"
        "ik_step_scale: 0.6\n"
        "ik_orientation_weight: 0.0\n"
        "require_ik_convergence: true\n",
        encoding="utf-8",
    )

    status = run_main(["--config", str(config_yaml)])

    assert status == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["method"] == "DP3"
    assert summary["feasible"] is True
    assert summary["ik"]["converged"] is True
    assert summary["ik"]["settings"]["ee_site"] == "tcp"
    assert summary["joint_path_source"] == "generated_from_cartesian"
    assert summary["joint_path_csv"] == str(joint_path_out)
    assert summary["constraint_check_source"] == "dense_segment_profiles"
    assert summary["constraint_check_samples"] == 11
    assert summary["time_quantities_csv"] == "time_quantities.csv"
    joint_path = PathData.from_csv(joint_path_out)
    assert joint_path.samples == 5
    assert joint_path.dof == 6
    rows = list(csv.DictReader((out_dir / "quantities.csv").open(newline="", encoding="utf-8")))
    tau_columns = [f"tau{axis}" for axis in range(1, 7)]
    tau = np.array([[float(row[column]) for column in tau_columns] for row in rows])
    assert np.max(np.abs(tau)) > 1e-9


def test_t12a_full_offline_traj_builds_and_runs_dp3_with_mujoco_dynamics(tmp_path):
    joint_path_out = tmp_path / "full_joint_path.csv"
    build_summary_out = tmp_path / "build_summary.json"

    build_status = build_path_main(
        [
            "--config",
            "configs/paper_like.yaml",
            "--out",
            str(joint_path_out),
            "--summary-out",
            str(build_summary_out),
        ]
    )

    assert build_status == 0
    build_summary = json.loads(build_summary_out.read_text(encoding="utf-8"))
    assert build_summary["samples"] == 400
    assert build_summary["dof"] == 6
    assert build_summary["s_end"] == pytest.approx(1.0)
    assert build_summary["ik"]["require_convergence"] is True
    assert build_summary["ik"]["converged"] is True
    assert build_summary["ik"]["settings"]["ee_site"] == "tcp"

    repeated = ", ".join(["1000000.0"] * 6)
    torque_speed = "\n".join(["  - [[0.0, 1000000.0], [100000.0, 1000000.0]]" for _ in range(6)])
    limits_yaml = tmp_path / "wide_complete_limits.yaml"
    limits_yaml.write_text(
        f"q_dot_abs: [{repeated}]\n"
        f"q_ddot_abs: [{repeated}]\n"
        f"q_jerk_abs: [{repeated}]\n"
        f"tau_abs: [{repeated}]\n"
        f"tau_rate_abs: [{repeated}]\n"
        "mechanical_power:\n"
        "  lower: -1000000000.0\n"
        "  upper: 1000000000.0\n"
        "torque_speed_breakpoints:\n"
        f"{torque_speed}\n"
        "friction:\n"
        "  coulomb: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n"
        "  viscous: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n"
        "motor:\n"
        "  gear_ratio: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]\n"
        "  torque_constant: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]\n"
        "  stator_resistance: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "full_dp3_run"

    run_status = run_main(
        [
            "--method",
            "dp3",
            "--path-csv",
            str(joint_path_out),
            "--model",
            str(default_dyn_dir() / "models" / "T12A" / "T12A-14.xml"),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "6",
            "--nz",
            "12",
            "--nch",
            "4",
            "--z-max",
            "0.01",
            "--constraint-check-points",
            "40",
            "--time-samples",
            "10",
        ]
    )

    assert run_status == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["method"] == "DP3"
    assert summary["feasible"] is True
    assert summary["segment_kinds"] == ["C2", "C4", "C3", "C3", "C2"]
    assert summary["constraint_check_source"] == "dense_segment_profiles"
    assert summary["constraint_check_samples"] == 40
    assert summary["time_quantities_csv"] == "time_quantities.csv"
    assert summary["constraint_utilization_csv"] == "constraint_utilization.csv"
    assert summary["constraint_violations_csv"] == "constraint_violations.csv"
    assert summary["violations"] == []
    joint_path = PathData.from_csv(joint_path_out)
    assert joint_path.samples == 400
    assert joint_path.dof == 6
    rows = list(csv.DictReader((out_dir / "quantities.csv").open(newline="", encoding="utf-8")))
    tau_columns = [f"tau{axis}" for axis in range(1, 7)]
    tau = np.array([[float(row[column]) for column in tau_columns] for row in rows])
    assert np.max(np.abs(tau)) > 1e-9


def test_run_main_rejects_path_csv_outside_mujoco_joint_ranges(tmp_path):
    model_xml = tmp_path / "one_dof.xml"
    model_xml.write_text(
        '<mujoco model="one_dof">\n'
        '  <compiler angle="radian"/>\n'
        '  <option gravity="0 0 0"/>\n'
        '  <worldbody>\n'
        '    <body name="link1" pos="0 0 0">\n'
        '      <joint name="joint1" type="hinge" axis="0 0 1" limited="true" range="-1 1"/>\n'
        '      <geom type="capsule" fromto="0 0 0 1 0 0" size="0.05" density="1000"/>\n'
        "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n",
        encoding="utf-8",
    )
    path_csv = tmp_path / "path.csv"
    path_csv.write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.5,1.2,1.0,0.0,0.0\n"
        "1.0,0.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [10.0]\n"
        "q_ddot_abs: [100.0]\n"
        "q_jerk_abs: [1000.0]\n"
        "tau_abs: [1000.0]\n"
        "tau_rate_abs: [10000.0]\n"
        "mechanical_power:\n"
        "  lower: -10000.0\n"
        "  upper: 10000.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "run"

    status = run_main(
        [
            "--path-csv",
            str(path_csv),
            "--model",
            str(model_xml),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "5",
            "--nz",
            "5",
            "--nch",
            "3",
        ]
    )

    assert status == 2
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "joint position limits" in summary["error"]
    assert "joint1" in summary["error"]


def test_run_main_path_csv_reports_data_failure_summary(tmp_path):
    missing_csv = tmp_path / "missing.csv"
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [2.0]\n"
        "q_ddot_abs: [20.0]\n"
        "q_jerk_abs: [1000.0]\n"
        "tau_abs: [1000.0]\n"
        "tau_rate_abs: [10000.0]\n"
        "mechanical_power:\n"
        "  lower: -10000.0\n"
        "  upper: 10000.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "failed-run"

    status = run_main(["--path-csv", str(missing_csv), "--limits", str(limits_yaml), "--out-dir", str(out_dir)])

    assert status == 2
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["feasible"] is False
    assert "missing.csv" in summary["error"]


def test_run_main_reports_invalid_limits_yaml_without_crashing(tmp_path):
    path_csv = tmp_path / "path.csv"
    path_csv.write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.5,0.5,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text("q_dot_abs: [1.0]\n", encoding="utf-8")
    out_dir = tmp_path / "failed-limits"

    status = run_main(["--path-csv", str(path_csv), "--limits", str(limits_yaml), "--out-dir", str(out_dir)])

    assert status == 2
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["feasible"] is False
    assert "missing required limit fields" in summary["error"]


def test_run_main_reports_malformed_limits_yaml_as_data_gap(tmp_path):
    path_csv = tmp_path / "path.csv"
    path_csv.write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.5,0.5,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text("q_dot_abs: [\n", encoding="utf-8")
    out_dir = tmp_path / "malformed-limits"

    status = run_main(["--path-csv", str(path_csv), "--limits", str(limits_yaml), "--out-dir", str(out_dir)])

    assert status == 2
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["feasible"] is False
    assert summary["data_readiness"] == "data_gaps"
    assert "limits YAML is invalid" in summary["error"]
    assert any("limits YAML is invalid" in gap for gap in summary["data_gaps"])


def test_run_main_reports_invalid_dp_config_without_crashing(tmp_path):
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
        "q_dot_abs: [1.0]\n"
        "q_ddot_abs: [5.0]\n"
        "q_jerk_abs: [50.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "failed-config"

    status = run_main(
        [
            "--path-csv",
            str(path_csv),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--z-start",
            "-0.1",
        ]
    )

    assert status == 2
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["feasible"] is False
    assert "z_start" in summary["error"]
    assert summary["path_source"] == str(path_csv)


def test_run_main_writes_json_safe_summary_for_nonfinite_dp_config(tmp_path):
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
        "q_dot_abs: [1.0]\n"
        "q_ddot_abs: [5.0]\n"
        "q_jerk_abs: [50.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "failed-nonfinite-config"

    status = run_main(
        [
            "--path-csv",
            str(path_csv),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--z-max",
            "inf",
        ]
    )

    assert status == 2
    summary_text = (out_dir / "summary.json").read_text(encoding="utf-8")
    assert "Infinity" not in summary_text
    assert "NaN" not in summary_text
    summary = json.loads(summary_text)
    assert summary["config"]["z_max"] is None
    assert "z_max" in summary["error"]


def test_run_main_reports_single_path_limits_dof_mismatch_with_summary(tmp_path):
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
        "q_dot_abs: [1.0, 1.0]\n"
        "q_ddot_abs: [5.0, 5.0]\n"
        "q_jerk_abs: [50.0, 50.0]\n"
        "tau_abs: [100.0, 100.0]\n"
        "tau_rate_abs: [100.0, 100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "failed-dof"

    status = run_main(["--path-csv", str(path_csv), "--limits", str(limits_yaml), "--out-dir", str(out_dir)])

    assert status == 2
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["feasible"] is False
    assert "limits DOF (2) does not match path DOF (1)" in summary["error"]
    assert summary["path_source"] == str(path_csv)
    sources = summary["reproduction_sources"]
    assert sources["path_source"] == str(path_csv)
    assert sources["limits"] == str(limits_yaml)
    assert sources["path_dof"] == 1
    assert sources["limits_dof"] == 2


def test_run_main_reports_limits_template_placeholders_without_crashing(tmp_path):
    path_csv = tmp_path / "path.csv"
    path_csv.write_text(
        "s,q1,dq1,d2q1,d3q1,q2,dq2,d2q2,d3q2,q3,dq3,d2q3,d3q3,q4,dq4,d2q4,d3q4,q5,dq5,d2q5,d3q5,q6,dq6,d2q6,d3q6\n"
        "0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    limits_template = default_dyn_dir() / "models" / "T12A" / "limits.template.yaml"
    out_dir = tmp_path / "failed-template-limits"

    status = run_main(["--path-csv", str(path_csv), "--limits", str(limits_template), "--out-dir", str(out_dir)])

    assert status == 2
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["feasible"] is False
    assert "must be finite" in summary["error"]


def test_run_main_reports_full_reproduction_data_gaps_when_required(tmp_path):
    path_csv = tmp_path / "path.csv"
    path_csv.write_text(
        "s,q1,dq1,d2q1,d3q1,q2,dq2,d2q2,d3q2,q3,dq3,d2q3,d3q3,q4,dq4,d2q4,d3q4,q5,dq5,d2q5,d3q5,q6,dq6,d2q6,d3q6\n"
        "0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "failed-full-repro-gaps"
    config_yaml = tmp_path / "run.yaml"
    limits_template = default_dyn_dir() / "models" / "T12A" / "limits.template.yaml"
    config_yaml.write_text(
        "method: dp3\n"
        f"path_csv: {path_csv.as_posix()}\n"
        f"limits: {limits_template.as_posix()}\n"
        f"out_dir: {out_dir.as_posix()}\n"
        "require_full_reproduction_data: true\n",
        encoding="utf-8",
    )

    status = run_main(["--config", str(config_yaml)])

    assert status == 2
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["feasible"] is False
    assert summary["data_readiness"] == "data_gaps"
    assert "incomplete full reproduction limits" in summary["error"]
    assert "q_dot_abs" in summary["data_gaps"]
    assert "torque_speed_breakpoints.axis_1.point_1" in summary["data_gaps"]
    assert summary["source_missing"] == []


def test_run_main_paper_batch_like_executes_selected_t12a_batch(tmp_path):
    out_dir = tmp_path / "paper-batch-selected-data"

    status = run_main(
        [
            "--config",
            "configs/paper_batch_like.yaml",
            "--out-dir",
            str(out_dir),
            "--ns",
            "6",
            "--nz",
            "12",
            "--nch",
            "4",
            "--constraint-check-points",
            "18",
        ]
    )

    assert status == 0
    comparison = json.loads((out_dir / "comparison_batch_summary.json").read_text(encoding="utf-8"))
    assert comparison["method"] == "compare"
    assert comparison["status"] == "ok"
    assert comparison["path_count"] == 14
    assert comparison["expected_path_count"] == 14
    assert comparison["data_readiness"] == "ready"
    assert comparison["source_missing"] == []
    assert comparison["data_gaps"] == []
    assert comparison["runs"]["dp3"]["status"] == 0
    assert comparison["runs"]["dp2"]["status"] == 0
    assert comparison["path_deltas"]
    assert comparison["path_deltas"][0]["dp3_constraint_check_source"] == "dense_segment_profiles"
    assert comparison["path_deltas"][0]["dp2_constraint_check_source"] == "dense_segment_profiles"
    sources = comparison["reproduction_sources"]
    assert sources["path_index"].endswith("path_index.yaml")
    assert sources["limits"].endswith("limits.yaml")
    assert sources["model"].endswith("T12A-14.xml")
    assert sources["dynamics_backend"] == "mujoco"
    assert not (out_dir / "summary.json").exists()


def test_run_main_full_reproduction_path_csv_requires_mujoco_model(tmp_path, monkeypatch):
    path_csv = tmp_path / "path.csv"
    path_csv.write_text(
        "s,q1,dq1,d2q1,d3q1,q2,dq2,d2q2,d3q2,q3,dq3,d2q3,d3q3,q4,dq4,d2q4,d3q4,q5,dq5,d2q5,d3q5,q6,dq6,d2q6,d3q6\n"
        "0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    repeated = ", ".join(["1000.0"] * 6)
    torque_speed = "\n".join(["  - [[0.0, 1000.0], [1000.0, 1000.0]]" for _ in range(6)])
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        f"q_dot_abs: [{repeated}]\n"
        f"q_ddot_abs: [{repeated}]\n"
        f"q_jerk_abs: [{repeated}]\n"
        f"tau_abs: [{repeated}]\n"
        f"tau_rate_abs: [{repeated}]\n"
        "mechanical_power:\n"
        "  lower: -1000000.0\n"
        "  upper: 1000000.0\n"
        "torque_speed_breakpoints:\n"
        f"{torque_speed}\n"
        "friction:\n"
        "  coulomb: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n"
        "  viscous: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n"
        "motor:\n"
        "  gear_ratio: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]\n"
        "  torque_constant: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]\n"
        "  stator_resistance: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "full-repro-no-mujoco"

    def fail_model_load(path):
        raise ValueError("MuJoCo model unavailable")

    monkeypatch.setattr(cli.MujocoRobotDynamics, "from_model_path", staticmethod(fail_model_load))

    status = run_main(
        [
            "--path-csv",
            str(path_csv),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--require-full-reproduction-data",
            "yes",
            "--ns",
            "4",
            "--nz",
            "3",
            "--nch",
            "2",
        ]
    )

    assert status == 2
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["feasible"] is False
    assert "MuJoCo model unavailable" in summary["error"]


def test_run_main_full_reproduction_path_index_requires_mujoco_model(tmp_path, monkeypatch):
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    path_csv = paths_dir / "path.csv"
    path_csv.write_text(
        "s,q1,dq1,d2q1,d3q1,q2,dq2,d2q2,d3q2,q3,dq3,d2q3,d3q3,q4,dq4,d2q4,d3q4,q5,dq5,d2q5,d3q5,q6,dq6,d2q6,d3q6\n"
        "0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: p1\n"
        "    csv: path.csv\n",
        encoding="utf-8",
    )
    repeated = ", ".join(["1000.0"] * 6)
    torque_speed = "\n".join(["  - [[0.0, 1000.0], [1000.0, 1000.0]]" for _ in range(6)])
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        f"q_dot_abs: [{repeated}]\n"
        f"q_ddot_abs: [{repeated}]\n"
        f"q_jerk_abs: [{repeated}]\n"
        f"tau_abs: [{repeated}]\n"
        f"tau_rate_abs: [{repeated}]\n"
        "mechanical_power:\n"
        "  lower: -1000000.0\n"
        "  upper: 1000000.0\n"
        "torque_speed_breakpoints:\n"
        f"{torque_speed}\n"
        "friction:\n"
        "  coulomb: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n"
        "  viscous: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n"
        "motor:\n"
        "  gear_ratio: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]\n"
        "  torque_constant: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]\n"
        "  stator_resistance: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "full-repro-batch-no-mujoco"

    def fail_model_load(path):
        raise ValueError("MuJoCo model unavailable")

    monkeypatch.setattr(cli.MujocoRobotDynamics, "from_model_path", staticmethod(fail_model_load))

    status = run_main(
        [
            "--path-index",
            str(path_index),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--require-full-reproduction-data",
            "yes",
            "--ns",
            "4",
            "--nz",
            "3",
            "--nch",
            "2",
        ]
    )

    assert status == 2
    batch_summary = json.loads((out_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert batch_summary["status"] == "failed"
    assert batch_summary["runs"][0]["status"] == 2
    assert "MuJoCo model unavailable" in batch_summary["runs"][0]["error"]

    compare_out_dir = tmp_path / "full-repro-batch-compare-no-mujoco"
    compare_status = run_main(
        [
            "--method",
            "compare",
            "--path-index",
            str(path_index),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(compare_out_dir),
            "--require-full-reproduction-data",
            "yes",
            "--ns",
            "4",
            "--nz",
            "3",
            "--nch",
            "2",
        ]
    )

    assert compare_status == 2
    comparison = json.loads((compare_out_dir / "comparison_batch_summary.json").read_text(encoding="utf-8"))
    assert comparison["status"] == "failed"
    assert comparison["runs"]["dp3"]["status"] == 2
    assert comparison["runs"]["dp2"]["status"] == 2
    assert all(value is None for value in comparison["delta"].values())
    assert all(
        value is None
        for key, value in comparison["path_deltas"][0].items()
        if key.endswith("_minus_dp3")
    )
    comparison_rows = list(csv.DictReader((compare_out_dir / "comparison_metrics.csv").open(newline="", encoding="utf-8")))
    assert comparison_rows[0]["total_time_dp2_minus_dp3"] == ""
    assert comparison_rows[0]["te_scale_dp2_minus_dp3"] == ""
    assert comparison_rows[0]["objective_cost_dp2_minus_dp3"] == ""


def test_run_main_reports_missing_limits_file_with_summary(tmp_path):
    path_csv = tmp_path / "path.csv"
    path_csv.write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.5,0.5,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    missing_limits = tmp_path / "limits.yaml"
    out_dir = tmp_path / "missing-limits"

    status = run_main(["--path-csv", str(path_csv), "--limits", str(missing_limits), "--out-dir", str(out_dir)])

    assert status == 2
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["feasible"] is False
    assert "missing limits file" in summary["error"]
    assert str(missing_limits) in summary["error"]
    assert summary["data_readiness"] == "source_missing"
    assert summary["source_missing"] == [str(missing_limits)]


def test_run_main_reports_missing_cartesian_input_with_summary(tmp_path):
    repeated = ", ".join(["100.0"] * 6)
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        f"q_dot_abs: [{repeated}]\n"
        f"q_ddot_abs: [{repeated}]\n"
        f"q_jerk_abs: [{repeated}]\n"
        f"tau_abs: [{repeated}]\n"
        f"tau_rate_abs: [{repeated}]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    missing_input = tmp_path / "Offline_Traj.txt"
    model = default_dyn_dir() / "models" / "T12A" / "T12A-14.xml"
    out_dir = tmp_path / "missing-input"

    status = run_main(["--input", str(missing_input), "--model", str(model), "--limits", str(limits_yaml), "--out-dir", str(out_dir)])

    assert status == 2
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["data_readiness"] == "source_missing"
    assert summary["source_missing"] == [str(missing_input)]
    assert "missing input path" in summary["error"]


def test_run_main_reports_missing_mujoco_model_with_summary(tmp_path):
    input_path = tmp_path / "Offline_Traj.txt"
    input_path.write_text(
        "0 0 0 0 0 0\n"
        "1 0 0 0 0 0\n",
        encoding="utf-8",
    )
    repeated = ", ".join(["100.0"] * 6)
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        f"q_dot_abs: [{repeated}]\n"
        f"q_ddot_abs: [{repeated}]\n"
        f"q_jerk_abs: [{repeated}]\n"
        f"tau_abs: [{repeated}]\n"
        f"tau_rate_abs: [{repeated}]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    missing_model = tmp_path / "T12A-14.xml"
    out_dir = tmp_path / "missing-model"

    status = run_main(["--input", str(input_path), "--model", str(missing_model), "--limits", str(limits_yaml), "--out-dir", str(out_dir)])

    assert status == 2
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["data_readiness"] == "source_missing"
    assert summary["source_missing"] == [str(missing_model)]
    assert "missing MuJoCo model" in summary["error"]
    sources = summary["reproduction_sources"]
    assert sources["path_source"] == str(input_path)
    assert sources["limits"] == str(limits_yaml)
    assert sources["model"] == str(missing_model)
    assert sources["dynamics_backend"] == "none"


def test_run_main_writes_json_safe_summary_for_infeasible_optimization(tmp_path):
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
        "q_dot_abs: [1e-12]\n"
        "q_ddot_abs: [1e-12]\n"
        "q_jerk_abs: [1e-12]\n"
        "tau_abs: [1e-12]\n"
        "tau_rate_abs: [1e-12]\n"
        "mechanical_power:\n"
        "  lower: -1e-12\n"
        "  upper: 1e-12\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "infeasible-run"

    status = run_main(
        [
            "--path-csv",
            str(path_csv),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "5",
            "--nz",
            "3",
            "--nch",
            "3",
        ]
    )

    assert status == 1
    summary_text = (out_dir / "summary.json").read_text(encoding="utf-8")
    assert "Infinity" not in summary_text
    assert "NaN" not in summary_text
    summary = json.loads(summary_text)
    assert summary["feasible"] is False
    assert summary["total_time"] is None
    assert summary["objective_cost"] is None
    assert summary["executable_with_st"] == "no"
    assert summary["te_scale"] == "-"


def test_run_main_rejects_fractional_grid_count_from_config_without_truncating(tmp_path):
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
        "q_dot_abs: [1.0]\n"
        "q_ddot_abs: [5.0]\n"
        "q_jerk_abs: [50.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "fractional-grid"
    config_yaml = tmp_path / "run.yaml"
    config_yaml.write_text(
        f"path_csv: {path_csv.as_posix()}\n"
        f"limits: {limits_yaml.as_posix()}\n"
        f"out_dir: {out_dir.as_posix()}\n"
        "ns: 6.5\n"
        "nz: 8\n"
        "nch: 7\n",
        encoding="utf-8",
    )

    status = run_main(["--config", str(config_yaml)])

    assert status == 2
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "ns" in summary["error"]
    assert summary["config"]["ns"] is None


def test_run_main_defaults_to_non_strict_cartesian_ik_and_reports_residuals(tmp_path):
    input_path = tmp_path / "Offline_Traj.txt"
    input_path.write_text(
        "100 100 100 0 0 0\n"
        "101 100 100 0 0 0\n",
        encoding="utf-8",
    )
    repeated = ", ".join(["1000"] * 6)
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        f"q_dot_abs: [{repeated}]\n"
        f"q_ddot_abs: [{repeated}]\n"
        f"q_jerk_abs: [{repeated}]\n"
        f"tau_abs: [{repeated}]\n"
        f"tau_rate_abs: [{repeated}]\n"
        "mechanical_power:\n"
        "  lower: -1000000\n"
        "  upper: 1000000\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "ik-nonstrict"

    status = run_main(
        [
            "--input",
            str(input_path),
            "--model",
            str(default_dyn_dir() / "models" / "T12A" / "T12A-14.xml"),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ik-max-iters",
            "1",
        ]
    )

    assert status in (0, 1)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["ik"]["require_convergence"] is False
    assert summary["ik"]["converged"] is False
    assert summary["ik"]["total_iterations"] == 2
    assert summary["ik"]["max_position_error"] > 1.0
    assert summary["ik"]["max_residual"] == max(
        summary["ik"]["max_position_error"],
        summary["ik"]["max_orientation_error"],
    )


def test_run_main_records_cartesian_ik_settings(tmp_path):
    input_path = tmp_path / "Offline_Traj.txt"
    input_path.write_text(
        "100 100 100 0 0 0\n"
        "101 100 100 0 0 0\n",
        encoding="utf-8",
    )
    repeated = ", ".join(["1000"] * 6)
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        f"q_dot_abs: [{repeated}]\n"
        f"q_ddot_abs: [{repeated}]\n"
        f"q_jerk_abs: [{repeated}]\n"
        f"tau_abs: [{repeated}]\n"
        f"tau_rate_abs: [{repeated}]\n"
        "mechanical_power:\n"
        "  lower: -1000000\n"
        "  upper: 1000000\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "ik-settings"

    status = run_main(
        [
            "--input",
            str(input_path),
            "--model",
            str(default_dyn_dir() / "models" / "T12A" / "T12A-14.xml"),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "5",
            "--nz",
            "3",
            "--nch",
            "3",
            "--ik-max-iters",
            "1",
            "--ik-pos-tol",
            "0.5",
            "--ik-ori-tol",
            "0.25",
            "--ik-damping",
            "0.002",
            "--ik-step-scale",
            "0.4",
            "--ik-orientation-weight",
            "0.2",
            "--tcp-offset-x",
            "0.01",
            "--tcp-offset-y",
            "-0.02",
            "--tcp-offset-z",
            "0.03",
        ]
    )

    assert status in (0, 1)
    settings = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))["ik"]["settings"]
    assert settings == {
        "max_iterations": 1,
        "position_tolerance": 0.5,
        "orientation_tolerance": 0.25,
        "damping": 0.002,
        "step_scale": 0.4,
        "orientation_weight": 0.2,
        "ee_body": "link_6",
        "ee_site": None,
        "tcp_offset": [0.01, -0.02, 0.03],
    }


def test_run_main_can_save_generated_cartesian_joint_path(tmp_path):
    input_path = tmp_path / "Offline_Traj.txt"
    input_path.write_text(
        "100 100 100 0 0 0\n"
        "101 100 100 0 0 0\n"
        "102 100 100 0 0 0\n",
        encoding="utf-8",
    )
    repeated = ", ".join(["1000"] * 6)
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        f"q_dot_abs: [{repeated}]\n"
        f"q_ddot_abs: [{repeated}]\n"
        f"q_jerk_abs: [{repeated}]\n"
        f"tau_abs: [{repeated}]\n"
        f"tau_rate_abs: [{repeated}]\n"
        "mechanical_power:\n"
        "  lower: -1000000\n"
        "  upper: 1000000\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "run"
    joint_path_out = tmp_path / "generated_joint_path.csv"

    status = run_main(
        [
            "--input",
            str(input_path),
            "--model",
            str(default_dyn_dir() / "models" / "T12A" / "T12A-14.xml"),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "5",
            "--nz",
            "3",
            "--nch",
            "3",
            "--ik-max-iters",
            "1",
            "--joint-path-out",
            str(joint_path_out),
        ]
    )

    assert status in (0, 1)
    generated = PathData.from_csv(joint_path_out)
    assert generated.samples == 3
    assert generated.dof == 6
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["joint_path_source"] == "generated_from_cartesian"
    assert summary["joint_path_csv"] == str(joint_path_out)


def test_run_main_can_require_cartesian_ik_convergence(tmp_path):
    input_path = tmp_path / "Offline_Traj.txt"
    input_path.write_text(
        "100 100 100 0 0 0\n"
        "101 100 100 0 0 0\n",
        encoding="utf-8",
    )
    repeated = ", ".join(["1000"] * 6)
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        f"q_dot_abs: [{repeated}]\n"
        f"q_ddot_abs: [{repeated}]\n"
        f"q_jerk_abs: [{repeated}]\n"
        f"tau_abs: [{repeated}]\n"
        f"tau_rate_abs: [{repeated}]\n"
        "mechanical_power:\n"
        "  lower: -1000000\n"
        "  upper: 1000000\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "strict-ik-failed"

    status = run_main(
        [
            "--input",
            str(input_path),
            "--model",
            str(default_dyn_dir() / "models" / "T12A" / "T12A-14.xml"),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ik-max-iters",
            "1",
            "--require-ik-convergence",
            "yes",
        ]
    )

    assert status == 2
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["feasible"] is False
    assert "IK failed to converge" in summary["error"]


def test_run_main_loads_yaml_config_for_paper_workflow(tmp_path):
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
        "q_dot_abs: [1.0]\n"
        "q_ddot_abs: [5.0]\n"
        "q_jerk_abs: [50.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "configured-run"
    config_yaml = tmp_path / "paper_like.yaml"
    config_yaml.write_text(
        f"method: dp3\n"
        f"path_csv: {path_csv.as_posix()}\n"
        f"limits: {limits_yaml.as_posix()}\n"
        f"out_dir: {out_dir.as_posix()}\n"
        "ns: 5\n"
        "nz: 7\n"
        "nch: 5\n"
        "k1: 2.0\n"
        "k2: 0.0\n"
        "z_start: 0.25\n"
        "z_end: 0.25\n"
        "z_max: 1.0\n"
        "tau_rate_dt: 0.0002\n",
        encoding="utf-8",
    )

    status = run_main(["--config", str(config_yaml)])

    assert status == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    trajectory_rows = list(csv.DictReader((out_dir / "trajectory.csv").open(newline="", encoding="utf-8")))
    assert float(trajectory_rows[0]["z"]) == 0.25
    assert float(trajectory_rows[-1]["z"]) == 0.25
    assert summary["config"]["z_start"] == 0.25
    assert summary["config"]["z_end"] == 0.25
    assert summary["config"]["z_max"] == 1.0
    assert summary["config"]["tau_rate_dt"] == 0.0002
    summary_text = (out_dir / "summary.json").read_text(encoding="utf-8")
    assert '"method": "DP3"' in summary_text
    assert '"ns": 5' in summary_text
    assert '"k1": 2.0' in summary_text
    assert '"objective_cost"' in summary_text


def test_run_main_supports_path_index_batch_with_per_path_endpoint_speeds(tmp_path):
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    for name, scale in (("path_a.csv", 1.0), ("path_b.csv", 0.5)):
        (paths_dir / name).write_text(
            "s,q1,dq1,d2q1,d3q1\n"
            f"0.0,0.0,{scale},0.0,0.0\n"
            f"0.5,{0.5 * scale},{scale},0.0,0.0\n"
            f"1.0,{scale},{scale},0.0,0.0\n",
            encoding="utf-8",
        )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: first\n"
        "    csv: path_a.csv\n"
        "    zs: 0.25\n"
        "    ze: 0.25\n"
        "  - id: second\n"
        "    csv: path_b.csv\n"
        "    z_start: 0.0\n"
        "    z_end: 0.0\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [2.0]\n"
        "q_ddot_abs: [10.0]\n"
        "q_jerk_abs: [100.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "batch"
    config_yaml = tmp_path / "batch.yaml"
    config_yaml.write_text(
        f"method: dp3\n"
        f"path_index: {path_index.as_posix()}\n"
        f"limits: {limits_yaml.as_posix()}\n"
        f"out_dir: {out_dir.as_posix()}\n"
        "ns: 5\n"
        "nz: 9\n"
        "nch: 5\n"
        "constraint_check_points: 17\n"
        "z_max: 1.0\n",
        encoding="utf-8",
    )

    status = run_main(["--config", str(config_yaml)])

    assert status == 0
    batch_summary = json.loads((out_dir / "batch_summary.json").read_text(encoding="utf-8"))
    batch_sources = batch_summary["reproduction_sources"]
    assert batch_sources["path_index"] == str(path_index)
    assert batch_sources["limits"] == str(limits_yaml)
    assert batch_sources["model"] is None
    assert batch_sources["dynamics_backend"] == "none"
    assert [entry["id"] for entry in batch_summary["runs"]] == ["first", "second"]
    assert batch_summary["status"] == "ok"
    assert batch_summary["feasible_count"] == 2
    assert batch_summary["infeasible_count"] == 0
    assert batch_summary["total_cpu_time_s"] > 0.0
    assert batch_summary["total_trajectory_time_s"] > 0.0
    assert batch_summary["objective_cost_total"] > 0.0
    assert batch_summary["total_te_scale_s"] > 0.0
    assert batch_summary["batch_metrics_csv"] == "batch_metrics.csv"
    assert batch_summary["violation_count_total"] == 0
    assert set(batch_summary["max_utilization"]) == {
        "q_dot",
        "q_ddot",
        "q_jerk",
        "tau",
        "tau_rate",
        "mechanical_power",
    }
    assert batch_summary["runs"][0]["required_time_scale_st"] == "100%"
    assert batch_summary["runs"][0]["executable_with_st"] == "yes"
    assert isinstance(batch_summary["runs"][0]["te_scale"], float)
    assert batch_summary["runs"][0]["violation_count"] == 0
    assert set(batch_summary["runs"][0]["max_utilization"]) == {
        "q_dot",
        "q_ddot",
        "q_jerk",
        "tau",
        "tau_rate",
        "mechanical_power",
    }
    assert batch_summary["runs"][0]["constraint_check_points"] == 17
    assert batch_summary["runs"][0]["constraint_check_samples"] == 17
    assert batch_summary["runs"][0]["constraint_check_source"] == "dense_segment_profiles"
    assert batch_summary["runs"][0]["constraint_utilization_csv"] == "constraint_utilization.csv"
    run_sources = batch_summary["runs"][0]["reproduction_sources"]
    assert run_sources["path_source"] == str(paths_dir / "path_a.csv")
    assert run_sources["path_index"] == str(path_index)
    assert run_sources["limits"] == str(limits_yaml)
    assert run_sources["dynamics_backend"] == "none"
    assert (out_dir / "first" / "constraint_utilization.csv").exists()
    metric_rows = list(csv.DictReader((out_dir / "batch_metrics.csv").open(newline="", encoding="utf-8")))
    assert [row["id"] for row in metric_rows] == ["first", "second"]
    assert metric_rows[0]["method"] == "DP3"
    assert metric_rows[0]["feasible"] == "yes"
    assert metric_rows[0]["path_index"] == str(path_index)
    assert metric_rows[0]["limits"] == str(limits_yaml)
    assert metric_rows[0]["model"] == ""
    assert metric_rows[0]["dynamics_backend"] == "none"
    assert float(metric_rows[0]["t_e_s"]) == pytest.approx(batch_summary["runs"][0]["total_time"])
    assert float(metric_rows[0]["t_cpu_s"]) == pytest.approx(batch_summary["runs"][0]["cpu_time_s"])
    assert int(metric_rows[0]["constraint_check_samples"]) == 17
    assert metric_rows[0]["constraint_check_source"] == "dense_segment_profiles"
    assert int(metric_rows[0]["violation_count"]) == 0
    assert metric_rows[0]["constraint_utilization_csv"] == "constraint_utilization.csv"
    assert float(metric_rows[0]["max_utilization_q_dot"]) == pytest.approx(batch_summary["runs"][0]["max_utilization"]["q_dot"])
    assert float(metric_rows[0]["active_constraint_percent_q_dot"]) == pytest.approx(
        batch_summary["runs"][0]["active_constraint_percent"]["q_dot"]
    )
    assert "active_constraint_percent" in batch_summary["runs"][0]
    assert set(batch_summary["runs"][0]["active_constraint_percent"]) == {
        "q_dot",
        "q_ddot",
        "q_jerk",
        "tau",
        "tau_rate",
        "mechanical_power",
    }
    assert "most_restrictive_constraint_percent" in batch_summary["runs"][0]
    assert max(batch_summary["runs"][0]["active_constraint_percent"].values()) == 0.0
    assert sum(batch_summary["runs"][0]["most_restrictive_constraint_percent"].values()) == 0.0
    first_summary = json.loads((out_dir / "first" / "summary.json").read_text(encoding="utf-8"))
    first_rows = list(csv.DictReader((out_dir / "first" / "trajectory.csv").open(newline="", encoding="utf-8")))
    assert first_summary["path_id"] == "first"
    assert first_summary["reproduction_sources"] == run_sources
    assert first_summary["config"]["z_start"] == 0.25
    assert first_summary["config"]["z_end"] == 0.25
    assert float(first_rows[0]["z"]) == 0.25
    assert float(first_rows[-1]["z"]) == 0.25
    assert (out_dir / "second" / "summary.json").exists()


def test_run_main_rejects_marked_path_index_template_before_batch_execution(tmp_path):
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    (paths_dir / "path_a.csv").write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.5,0.5,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "template: true\n"
        "paths:\n"
        "  - id: first\n"
        "    csv: path_a.csv\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [2.0]\n"
        "q_ddot_abs: [10.0]\n"
        "q_jerk_abs: [100.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "batch"

    status = run_main(
        [
            "--path-index",
            str(path_index),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "5",
            "--nz",
            "9",
            "--nch",
            "5",
        ]
    )

    assert status == 2
    batch_summary = json.loads((out_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert batch_summary["status"] == "failed"
    assert batch_summary["path_count"] == 0
    assert batch_summary["data_readiness"] == "data_gaps"
    assert batch_summary["source_missing"] == []
    assert any("path_index" in gap and "template" in gap for gap in batch_summary["data_gaps"])
    assert not (out_dir / "first" / "summary.json").exists()


def test_run_main_path_index_compare_reports_full_reproduction_limits_gaps_as_batch_summary(tmp_path):
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    (paths_dir / "path_a.csv").write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.5,0.5,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: first\n"
        "    csv: path_a.csv\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "partial_limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [2.0]\n"
        "q_ddot_abs: [10.0]\n"
        "q_jerk_abs: [100.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "batch-compare"

    status = run_main(
        [
            "--method",
            "compare",
            "--path-index",
            str(path_index),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--require-full-reproduction-data",
            "yes",
        ]
    )

    assert status == 2
    comparison = json.loads((out_dir / "comparison_batch_summary.json").read_text(encoding="utf-8"))
    assert comparison["method"] == "compare"
    assert comparison["status"] == "failed"
    assert comparison["path_count"] == 0
    assert comparison["data_readiness"] == "data_gaps"
    assert comparison["source_missing"] == []
    assert "friction.coulomb" in comparison["data_gaps"]
    assert "torque_speed_breakpoints" in comparison["data_gaps"]
    sources = comparison["reproduction_sources"]
    assert sources["path_index"] == str(path_index)
    assert sources["limits"] == str(limits_yaml)
    assert sources["dynamics_backend"] == "none"
    assert not (out_dir / "summary.json").exists()


def test_run_main_path_index_reports_invalid_limits_yaml_with_batch_summary(tmp_path):
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    (paths_dir / "path_a.csv").write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.5,0.5,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: first\n"
        "    csv: path_a.csv\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "invalid_limits.yaml"
    limits_yaml.write_text("q_dot_abs: [2.0]\n", encoding="utf-8")
    out_dir = tmp_path / "batch"

    status = run_main(
        [
            "--path-index",
            str(path_index),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert status == 2
    batch_summary = json.loads((out_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert batch_summary["status"] == "failed"
    assert batch_summary["path_count"] == 0
    assert "missing required limit fields" in batch_summary["error"]
    sources = batch_summary["reproduction_sources"]
    assert sources["path_index"] == str(path_index)
    assert sources["limits"] == str(limits_yaml)
    assert sources["dynamics_backend"] == "none"
    assert not (out_dir / "summary.json").exists()


def test_run_main_path_index_compare_reports_invalid_limits_yaml_with_comparison_batch_summary(tmp_path):
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    (paths_dir / "path_a.csv").write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.5,0.5,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: first\n"
        "    csv: path_a.csv\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "invalid_limits.yaml"
    limits_yaml.write_text("q_dot_abs: [2.0]\n", encoding="utf-8")
    out_dir = tmp_path / "batch-compare"

    status = run_main(
        [
            "--method",
            "compare",
            "--path-index",
            str(path_index),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert status == 2
    comparison = json.loads((out_dir / "comparison_batch_summary.json").read_text(encoding="utf-8"))
    assert comparison["status"] == "failed"
    assert comparison["path_count"] == 0
    assert "missing required limit fields" in comparison["error"]
    sources = comparison["reproduction_sources"]
    assert sources["path_index"] == str(path_index)
    assert sources["limits"] == str(limits_yaml)
    assert sources["dynamics_backend"] == "none"
    assert not (out_dir / "summary.json").exists()


def test_run_main_path_index_compare_exposes_batch_config_error_at_top_level(tmp_path):
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    (paths_dir / "path_a.csv").write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.5,0.5,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: first\n"
        "    csv: path_a.csv\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [2.0]\n"
        "q_ddot_abs: [10.0]\n"
        "q_jerk_abs: [100.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "batch-compare"

    status = run_main(
        [
            "--method",
            "compare",
            "--path-index",
            str(path_index),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--expected-path-count",
            "14",
        ]
    )

    assert status == 2
    comparison = json.loads((out_dir / "comparison_batch_summary.json").read_text(encoding="utf-8"))
    assert comparison["status"] == "failed"
    assert comparison["path_count"] == 1
    assert comparison["expected_path_count"] == 14
    assert "expected 14" in comparison["error"]
    assert comparison["data_readiness"] == "data_gaps"
    assert comparison["data_gaps"] == [comparison["error"]]


def test_run_main_supports_path_index_dp2_dp3_comparison(tmp_path):
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    for name, scale in (("path_a.csv", 1.0), ("path_b.csv", 0.5)):
        (paths_dir / name).write_text(
            "s,q1,dq1,d2q1,d3q1\n"
            f"0.0,0.0,{scale},0.0,0.0\n"
            f"0.5,{0.5 * scale},{scale},0.0,0.0\n"
            f"1.0,{scale},{scale},0.0,0.0\n",
            encoding="utf-8",
        )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: first\n"
        "    csv: path_a.csv\n"
        "  - id: second\n"
        "    csv: path_b.csv\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [2.0]\n"
        "q_ddot_abs: [10.0]\n"
        "q_jerk_abs: [100.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "batch-compare"

    status = run_main(
        [
            "--method",
            "compare",
            "--path-index",
            str(path_index),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "5",
            "--nz",
            "9",
            "--nch",
            "5",
            "--constraint-check-points",
            "17",
            "--z-max",
            "1.0",
        ]
    )

    assert status == 0
    assert (out_dir / "dp3" / "batch_summary.json").exists()
    assert (out_dir / "dp2" / "batch_summary.json").exists()
    dp3_batch = json.loads((out_dir / "dp3" / "batch_summary.json").read_text(encoding="utf-8"))
    dp2_batch = json.loads((out_dir / "dp2" / "batch_summary.json").read_text(encoding="utf-8"))
    comparison = json.loads((out_dir / "comparison_batch_summary.json").read_text(encoding="utf-8"))
    assert comparison["method"] == "compare"
    assert comparison["path_count"] == 2
    assert comparison["status"] == "ok"
    sources = comparison["reproduction_sources"]
    assert sources["path_index"] == str(path_index)
    assert sources["limits"] == str(limits_yaml)
    assert sources["model"] is None
    assert sources["dynamics_backend"] == "none"
    assert comparison["runs"]["dp3"]["method"] == "dp3"
    assert comparison["runs"]["dp2"]["method"] == "dp2"
    assert comparison["runs"]["dp3"]["status"] == 0
    assert comparison["runs"]["dp2"]["status"] == 0
    assert comparison["runs"]["dp3"]["batch_summary_json"] == str(out_dir / "dp3" / "batch_summary.json")
    assert comparison["runs"]["dp2"]["batch_summary_json"] == str(out_dir / "dp2" / "batch_summary.json")
    assert comparison["runs"]["dp3"]["violation_count_total"] == dp3_batch["violation_count_total"]
    assert comparison["runs"]["dp2"]["violation_count_total"] == dp2_batch["violation_count_total"]
    assert comparison["runs"]["dp3"]["max_utilization"]["q_dot"] == pytest.approx(
        dp3_batch["max_utilization"]["q_dot"]
    )
    assert comparison["runs"]["dp2"]["max_utilization"]["q_dot"] == pytest.approx(
        dp2_batch["max_utilization"]["q_dot"]
    )
    assert comparison["comparison_metrics_csv"] == "comparison_metrics.csv"
    comparison_rows = list(csv.DictReader((out_dir / "comparison_metrics.csv").open(newline="", encoding="utf-8")))
    assert [row["id"] for row in comparison_rows] == ["first", "second"]
    assert comparison_rows[0]["path_index"] == str(path_index)
    assert comparison_rows[0]["limits"] == str(limits_yaml)
    assert comparison_rows[0]["model"] == ""
    assert comparison_rows[0]["dynamics_backend"] == "none"
    assert comparison_rows[0]["dp3_constraint_utilization_csv"] == "constraint_utilization.csv"
    assert comparison_rows[0]["dp2_constraint_utilization_csv"] == "constraint_utilization.csv"
    assert comparison_rows[0]["dp3_constraint_violations_csv"] == "constraint_violations.csv"
    assert comparison_rows[0]["dp2_constraint_violations_csv"] == "constraint_violations.csv"
    assert comparison_rows[0]["dp3_constraint_check_source"] == "dense_segment_profiles"
    assert comparison_rows[0]["dp2_constraint_check_source"] == "dense_segment_profiles"
    assert int(comparison_rows[0]["dp3_constraint_check_samples"]) == 17
    assert int(comparison_rows[0]["dp2_constraint_check_samples"]) == 17
    assert int(comparison_rows[0]["constraint_check_samples_dp2_minus_dp3"]) == 0
    assert float(comparison_rows[0]["dp3_max_utilization_q_dot"]) >= 0.0
    assert float(comparison_rows[0]["dp2_max_utilization_q_dot"]) >= 0.0
    assert float(comparison_rows[0]["dp3_active_constraint_percent_q_dot"]) >= 0.0
    assert float(comparison_rows[0]["dp2_active_constraint_percent_q_dot"]) >= 0.0
    assert float(comparison_rows[0]["dp3_most_restrictive_constraint_percent_q_dot"]) >= 0.0
    assert float(comparison_rows[0]["dp2_most_restrictive_constraint_percent_q_dot"]) >= 0.0
    assert float(comparison_rows[0]["max_utilization_dp2_minus_dp3_q_dot"]) == pytest.approx(
        float(comparison_rows[0]["dp2_max_utilization_q_dot"])
        - float(comparison_rows[0]["dp3_max_utilization_q_dot"])
    )
    assert float(comparison_rows[0]["active_constraint_percent_dp2_minus_dp3_q_dot"]) == pytest.approx(
        float(comparison_rows[0]["dp2_active_constraint_percent_q_dot"])
        - float(comparison_rows[0]["dp3_active_constraint_percent_q_dot"])
    )
    assert float(comparison_rows[0]["most_restrictive_constraint_percent_dp2_minus_dp3_q_dot"]) == pytest.approx(
        float(comparison_rows[0]["dp2_most_restrictive_constraint_percent_q_dot"])
        - float(comparison_rows[0]["dp3_most_restrictive_constraint_percent_q_dot"])
    )
    assert comparison_rows[0]["dp3_required_time_scale_st"] == "100%"
    assert comparison_rows[0]["dp2_required_time_scale_st"] == "100%"
    assert comparison_rows[0]["dp3_executable_with_st"] == "yes"
    assert comparison_rows[0]["dp2_executable_with_st"] == "yes"
    assert float(comparison_rows[0]["dp3_t_e_s"]) > 0.0
    assert float(comparison_rows[0]["dp2_t_e_s"]) > 0.0
    assert float(comparison_rows[0]["dp3_te_scale"]) == pytest.approx(float(comparison_rows[0]["dp3_t_e_s"]))
    assert float(comparison_rows[0]["dp2_te_scale"]) == pytest.approx(float(comparison_rows[0]["dp2_t_e_s"]))
    assert float(comparison_rows[0]["te_scale_dp2_minus_dp3"]) == pytest.approx(
        float(comparison_rows[0]["dp2_te_scale"]) - float(comparison_rows[0]["dp3_te_scale"])
    )
    assert float(comparison_rows[0]["total_time_dp2_minus_dp3"]) == pytest.approx(
        float(comparison_rows[0]["dp2_t_e_s"]) - float(comparison_rows[0]["dp3_t_e_s"])
    )
    assert float(comparison_rows[0]["objective_time_cost_dp2_minus_dp3"]) == pytest.approx(
        float(comparison_rows[0]["dp2_objective_time_cost"])
        - float(comparison_rows[0]["dp3_objective_time_cost"])
    )
    assert float(comparison_rows[0]["objective_drive_power_cost_dp2_minus_dp3"]) == pytest.approx(
        float(comparison_rows[0]["dp2_objective_drive_power_cost"])
        - float(comparison_rows[0]["dp3_objective_drive_power_cost"])
    )
    assert int(comparison_rows[0]["dp3_violation_count"]) == 0
    assert int(comparison_rows[0]["dp2_violation_count"]) == 0
    assert [entry["id"] for entry in comparison["path_deltas"]] == ["first", "second"]
    assert set(comparison["path_deltas"][0]) == {
        "id",
        "dp3_run_dir",
        "dp2_run_dir",
        "dp3_status",
        "dp2_status",
        "dp3_data_readiness",
        "dp2_data_readiness",
        "dp3_source_missing",
        "dp2_source_missing",
        "dp3_data_gaps",
        "dp2_data_gaps",
        "dp3_error",
        "dp2_error",
        "dp3_constraint_utilization_csv",
        "dp2_constraint_utilization_csv",
        "dp3_constraint_violations_csv",
        "dp2_constraint_violations_csv",
        "dp3_constraint_check_source",
        "dp2_constraint_check_source",
        "dp3_constraint_check_samples",
        "dp2_constraint_check_samples",
        "constraint_check_samples_dp2_minus_dp3",
        "dp3_max_utilization",
        "dp2_max_utilization",
        "dp3_active_constraint_percent",
        "dp2_active_constraint_percent",
        "dp3_most_restrictive_constraint_percent",
        "dp2_most_restrictive_constraint_percent",
        "max_utilization_dp2_minus_dp3",
        "active_constraint_percent_dp2_minus_dp3",
        "most_restrictive_constraint_percent_dp2_minus_dp3",
        "dp3_required_time_scale_st",
        "dp2_required_time_scale_st",
        "dp3_executable_with_st",
        "dp2_executable_with_st",
        "dp3_te_scale",
        "dp2_te_scale",
        "te_scale_dp2_minus_dp3",
        "total_time_dp2_minus_dp3",
        "cpu_time_dp2_minus_dp3",
        "objective_cost_dp2_minus_dp3",
        "objective_time_cost_dp2_minus_dp3",
        "objective_drive_power_cost_dp2_minus_dp3",
    }
    assert comparison["path_deltas"][0]["dp3_data_readiness"] == "ready"
    assert comparison["path_deltas"][0]["dp2_data_readiness"] == "ready"
    assert comparison["path_deltas"][0]["dp3_source_missing"] == []
    assert comparison["path_deltas"][0]["dp2_source_missing"] == []
    assert comparison["path_deltas"][0]["dp3_data_gaps"] == []
    assert comparison["path_deltas"][0]["dp2_data_gaps"] == []
    assert comparison["path_deltas"][0]["dp3_run_dir"] == str(out_dir / "dp3" / "first")
    assert comparison["path_deltas"][0]["dp2_run_dir"] == str(out_dir / "dp2" / "first")
    assert comparison["path_deltas"][0]["dp3_constraint_utilization_csv"] == "constraint_utilization.csv"
    assert comparison["path_deltas"][0]["dp2_constraint_utilization_csv"] == "constraint_utilization.csv"
    assert comparison["path_deltas"][0]["dp3_constraint_violations_csv"] == "constraint_violations.csv"
    assert comparison["path_deltas"][0]["dp2_constraint_violations_csv"] == "constraint_violations.csv"
    assert comparison["path_deltas"][0]["dp3_constraint_check_source"] == "dense_segment_profiles"
    assert comparison["path_deltas"][0]["dp2_constraint_check_source"] == "dense_segment_profiles"
    assert comparison["path_deltas"][0]["dp3_constraint_check_samples"] == 17
    assert comparison["path_deltas"][0]["dp2_constraint_check_samples"] == 17
    assert comparison["path_deltas"][0]["constraint_check_samples_dp2_minus_dp3"] == 0
    assert comparison["path_deltas"][0]["dp3_max_utilization"]["q_dot"] == pytest.approx(
        float(comparison_rows[0]["dp3_max_utilization_q_dot"])
    )
    assert comparison["path_deltas"][0]["dp2_max_utilization"]["q_dot"] == pytest.approx(
        float(comparison_rows[0]["dp2_max_utilization_q_dot"])
    )
    assert comparison["path_deltas"][0]["dp3_active_constraint_percent"]["q_dot"] == pytest.approx(
        float(comparison_rows[0]["dp3_active_constraint_percent_q_dot"])
    )
    assert comparison["path_deltas"][0]["dp2_active_constraint_percent"]["q_dot"] == pytest.approx(
        float(comparison_rows[0]["dp2_active_constraint_percent_q_dot"])
    )
    assert comparison["path_deltas"][0]["max_utilization_dp2_minus_dp3"]["q_dot"] == pytest.approx(
        float(comparison_rows[0]["max_utilization_dp2_minus_dp3_q_dot"])
    )
    assert comparison["path_deltas"][0]["active_constraint_percent_dp2_minus_dp3"]["q_dot"] == pytest.approx(
        float(comparison_rows[0]["active_constraint_percent_dp2_minus_dp3_q_dot"])
    )
    assert comparison["path_deltas"][0]["most_restrictive_constraint_percent_dp2_minus_dp3"]["q_dot"] == pytest.approx(
        float(comparison_rows[0]["most_restrictive_constraint_percent_dp2_minus_dp3_q_dot"])
    )
    assert comparison["path_deltas"][0]["dp3_required_time_scale_st"] == "100%"
    assert comparison["path_deltas"][0]["dp2_required_time_scale_st"] == "100%"
    assert comparison["path_deltas"][0]["dp3_executable_with_st"] == "yes"
    assert comparison["path_deltas"][0]["dp2_executable_with_st"] == "yes"
    assert comparison["path_deltas"][0]["dp3_te_scale"] == pytest.approx(float(comparison_rows[0]["dp3_te_scale"]))
    assert comparison["path_deltas"][0]["dp2_te_scale"] == pytest.approx(float(comparison_rows[0]["dp2_te_scale"]))
    assert comparison["path_deltas"][0]["te_scale_dp2_minus_dp3"] == pytest.approx(
        float(comparison_rows[0]["dp2_te_scale"]) - float(comparison_rows[0]["dp3_te_scale"])
    )
    assert comparison["path_deltas"][0]["objective_time_cost_dp2_minus_dp3"] == pytest.approx(
        float(comparison_rows[0]["dp2_objective_time_cost"])
        - float(comparison_rows[0]["dp3_objective_time_cost"])
    )
    assert comparison["path_deltas"][0]["objective_drive_power_cost_dp2_minus_dp3"] == pytest.approx(
        float(comparison_rows[0]["dp2_objective_drive_power_cost"])
        - float(comparison_rows[0]["dp3_objective_drive_power_cost"])
    )
    assert set(comparison["delta"]) == {
        "total_trajectory_time_dp2_minus_dp3",
        "total_te_scale_dp2_minus_dp3",
        "total_cpu_time_dp2_minus_dp3",
        "violation_count_total_dp2_minus_dp3",
        "max_utilization_dp2_minus_dp3",
        "objective_cost_total_dp2_minus_dp3",
        "objective_time_cost_total_dp2_minus_dp3",
        "objective_drive_power_cost_total_dp2_minus_dp3",
    }
    assert comparison["delta"]["violation_count_total_dp2_minus_dp3"] == (
        dp2_batch["violation_count_total"] - dp3_batch["violation_count_total"]
    )
    assert comparison["delta"]["max_utilization_dp2_minus_dp3"]["q_dot"] == pytest.approx(
        dp2_batch["max_utilization"]["q_dot"] - dp3_batch["max_utilization"]["q_dot"]
    )
    assert comparison["delta"]["objective_time_cost_total_dp2_minus_dp3"] == pytest.approx(
        comparison["runs"]["dp2"]["objective_time_cost_total"]
        - comparison["runs"]["dp3"]["objective_time_cost_total"]
    )
    assert comparison["delta"]["objective_drive_power_cost_total_dp2_minus_dp3"] == pytest.approx(
        comparison["runs"]["dp2"]["objective_drive_power_cost_total"]
        - comparison["runs"]["dp3"]["objective_drive_power_cost_total"]
    )
    assert comparison["delta"]["total_te_scale_dp2_minus_dp3"] == pytest.approx(
        comparison["runs"]["dp2"]["total_te_scale_s"] - comparison["runs"]["dp3"]["total_te_scale_s"]
    )


def test_run_main_path_index_makes_duplicate_run_ids_unique(tmp_path):
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    for name, scale in (("path_a.csv", 1.0), ("path_b.csv", 0.5)):
        (paths_dir / name).write_text(
            "s,q1,dq1,d2q1,d3q1\n"
            f"0.0,0.0,{scale},0.0,0.0\n"
            f"0.5,{0.5 * scale},{scale},0.0,0.0\n"
            f"1.0,{scale},{scale},0.0,0.0\n",
            encoding="utf-8",
        )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: repeat\n"
        "    csv: path_a.csv\n"
        "  - id: repeat\n"
        "    csv: path_b.csv\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [2.0]\n"
        "q_ddot_abs: [10.0]\n"
        "q_jerk_abs: [100.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "batch"

    status = run_main(
        [
            "--path-index",
            str(path_index),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "5",
            "--nz",
            "9",
            "--nch",
            "5",
        ]
    )

    assert status == 0
    batch_summary = json.loads((out_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert [entry["id"] for entry in batch_summary["runs"]] == ["repeat", "repeat_2"]
    assert (out_dir / "repeat" / "summary.json").exists()
    assert (out_dir / "repeat_2" / "summary.json").exists()
    second_summary = json.loads((out_dir / "repeat_2" / "summary.json").read_text(encoding="utf-8"))
    assert second_summary["path_id"] == "repeat_2"
    assert second_summary["path_source"] == str(paths_dir / "path_b.csv")


def test_run_main_path_index_rejects_unexpected_path_count_from_config(tmp_path):
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    (paths_dir / "path_a.csv").write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.5,0.5,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: first\n"
        "    csv: path_a.csv\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [2.0]\n"
        "q_ddot_abs: [10.0]\n"
        "q_jerk_abs: [100.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "batch"
    config_yaml = tmp_path / "batch.yaml"
    config_yaml.write_text(
        f"path_index: {path_index.as_posix()}\n"
        f"limits: {limits_yaml.as_posix()}\n"
        f"out_dir: {out_dir.as_posix()}\n"
        "expected_path_count: 14\n"
        "ns: 5\n"
        "nz: 7\n"
        "nch: 3\n",
        encoding="utf-8",
    )

    status = run_main(["--config", str(config_yaml)])

    assert status == 2
    batch_summary = json.loads((out_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert batch_summary["status"] == "failed"
    assert batch_summary["path_count"] == 1
    assert batch_summary["expected_path_count"] == 14
    assert "expected 14" in batch_summary["error"]
    assert batch_summary["data_readiness"] == "data_gaps"
    assert batch_summary["data_gaps"] == [batch_summary["error"]]
    sources = batch_summary["reproduction_sources"]
    assert sources["path_index"] == str(path_index)
    assert sources["limits"] == str(limits_yaml)
    assert sources["dynamics_backend"] == "none"


def test_run_main_path_index_uses_configured_mujoco_dynamics(tmp_path):
    model_xml = tmp_path / "one_dof.xml"
    model_xml.write_text(
        '<mujoco model="one_dof">\n'
        '  <compiler angle="radian"/>\n'
        '  <option gravity="0 0 0"/>\n'
        '  <worldbody>\n'
        '    <body name="link1" pos="0 0 0">\n'
        '      <joint name="joint1" type="hinge" axis="0 0 1" limited="true" range="-10 10"/>\n'
        '      <geom type="capsule" fromto="0 0 0 1 0 0" size="0.05" density="1000"/>\n'
        "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n",
        encoding="utf-8",
    )
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    (paths_dir / "joint.csv").write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.5,0.5,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: joint_path\n"
        "    csv: joint.csv\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [2.0]\n"
        "q_ddot_abs: [20.0]\n"
        "q_jerk_abs: [1000.0]\n"
        "tau_abs: [1000.0]\n"
        "tau_rate_abs: [10000.0]\n"
        "mechanical_power:\n"
        "  lower: -10000.0\n"
        "  upper: 10000.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "batch"

    status = run_main(
        [
            "--path-index",
            str(path_index),
            "--model",
            str(model_xml),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "5",
            "--nz",
            "9",
            "--nch",
            "5",
        ]
    )

    assert status == 0
    rows = list(csv.DictReader((out_dir / "joint_path" / "quantities.csv").open(newline="", encoding="utf-8")))
    tau = np.array([float(row["tau1"]) for row in rows])
    assert np.max(np.abs(tau)) > 1e-6


def test_run_main_path_index_reports_per_path_data_failures(tmp_path):
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    (paths_dir / "valid.csv").write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.5,0.5,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    (paths_dir / "wrong_dof.csv").write_text(
        "s,q1,q2,dq1,dq2,d2q1,d2q2,d3q1,d3q2\n"
        "0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0\n"
        "0.5,0.5,0.5,1.0,1.0,0.0,0.0,0.0,0.0\n"
        "1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0\n",
        encoding="utf-8",
    )
    (paths_dir / "bad_schema.csv").write_text(
        "s,q1,dq1,d2q1\n"
        "0.0,0.0,1.0,0.0\n"
        "0.5,0.5,1.0,0.0\n"
        "1.0,1.0,1.0,0.0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: ok\n"
        "    csv: valid.csv\n"
        "  - id: bad_dof\n"
        "    csv: wrong_dof.csv\n"
        "  - id: bad_schema\n"
        "    csv: bad_schema.csv\n"
        "  - id: missing\n"
        "    csv: missing.csv\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [2.0]\n"
        "q_ddot_abs: [10.0]\n"
        "q_jerk_abs: [100.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "batch"

    status = run_main(
        [
            "--path-index",
            str(path_index),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "5",
            "--nz",
            "9",
            "--nch",
            "5",
        ]
    )

    assert status == 2
    batch_summary = json.loads((out_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert batch_summary["status"] == "failed"
    assert batch_summary["feasible_count"] == 1
    assert batch_summary["infeasible_count"] == 3
    assert [entry["id"] for entry in batch_summary["runs"]] == ["ok", "bad_dof", "bad_schema", "missing"]
    assert batch_summary["runs"][0]["status"] == 0
    assert batch_summary["runs"][1]["status"] == 2
    assert batch_summary["runs"][1]["feasible"] is False
    assert "DOF" in batch_summary["runs"][1]["error"]
    assert batch_summary["runs"][2]["status"] == 2
    assert batch_summary["runs"][2]["feasible"] is False
    assert "missing columns" in batch_summary["runs"][2]["error"]
    assert batch_summary["runs"][2]["data_readiness"] == "data_gaps"
    assert batch_summary["runs"][2]["source_missing"] == []
    assert any("missing columns" in gap for gap in batch_summary["runs"][2]["data_gaps"])
    bad_schema_sources = batch_summary["runs"][2]["reproduction_sources"]
    assert bad_schema_sources["path_source"] == str(paths_dir / "bad_schema.csv")
    assert bad_schema_sources["path_index"] == str(path_index)
    assert bad_schema_sources["limits"] == str(limits_yaml)
    assert bad_schema_sources["dynamics_backend"] == "none"
    bad_schema_summary = json.loads((out_dir / "bad_schema" / "summary.json").read_text(encoding="utf-8"))
    assert bad_schema_summary["reproduction_sources"] == bad_schema_sources
    assert bad_schema_summary["data_readiness"] == "data_gaps"
    assert bad_schema_summary["source_missing"] == []
    assert any("missing columns" in gap for gap in bad_schema_summary["data_gaps"])
    assert batch_summary["runs"][3]["status"] == 2
    assert batch_summary["runs"][3]["feasible"] is False
    assert "missing.csv" in batch_summary["runs"][3]["error"]
    assert batch_summary["runs"][3]["data_readiness"] == "source_missing"
    assert batch_summary["runs"][3]["source_missing"] == [str(paths_dir / "missing.csv")]
    assert batch_summary["runs"][3]["data_gaps"] == []
    missing_summary = json.loads((out_dir / "missing" / "summary.json").read_text(encoding="utf-8"))
    assert missing_summary["data_readiness"] == "source_missing"
    assert missing_summary["source_missing"] == [str(paths_dir / "missing.csv")]
    metric_rows = list(csv.DictReader((out_dir / "batch_metrics.csv").open(newline="", encoding="utf-8")))
    assert metric_rows[2]["data_readiness"] == "data_gaps"
    assert "missing columns" in metric_rows[2]["error"]
    assert metric_rows[2]["source_missing"] == ""
    assert "missing columns" in metric_rows[2]["data_gaps"]
    assert metric_rows[3]["data_readiness"] == "source_missing"
    assert metric_rows[3]["source_missing"] == str(paths_dir / "missing.csv")
    assert metric_rows[3]["data_gaps"] == ""
    assert (out_dir / "ok" / "summary.json").exists()


def test_run_main_path_index_compare_exposes_per_path_failures_at_top_level(tmp_path):
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    (paths_dir / "valid.csv").write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.5,0.5,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    (paths_dir / "bad_schema.csv").write_text(
        "s,q1,dq1,d2q1\n"
        "0.0,0.0,1.0,0.0\n"
        "0.5,0.5,1.0,0.0\n"
        "1.0,1.0,1.0,0.0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: ok\n"
        "    csv: valid.csv\n"
        "  - id: bad_schema\n"
        "    csv: bad_schema.csv\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [2.0]\n"
        "q_ddot_abs: [10.0]\n"
        "q_jerk_abs: [100.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "batch-compare"

    status = run_main(
        [
            "--method",
            "compare",
            "--path-index",
            str(path_index),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "5",
            "--nz",
            "9",
            "--nch",
            "5",
        ]
    )

    assert status == 2
    comparison = json.loads((out_dir / "comparison_batch_summary.json").read_text(encoding="utf-8"))
    bad_delta = next(entry for entry in comparison["path_deltas"] if entry["id"] == "bad_schema")
    assert bad_delta["dp3_status"] == 2
    assert bad_delta["dp2_status"] == 2
    assert bad_delta["dp3_data_readiness"] == "data_gaps"
    assert bad_delta["dp2_data_readiness"] == "data_gaps"
    assert bad_delta["dp3_source_missing"] == []
    assert bad_delta["dp2_source_missing"] == []
    assert any("missing columns" in gap for gap in bad_delta["dp3_data_gaps"])
    assert any("missing columns" in gap for gap in bad_delta["dp2_data_gaps"])
    assert "missing columns" in bad_delta["dp3_error"]
    assert "missing columns" in bad_delta["dp2_error"]
    rows = list(csv.DictReader((out_dir / "comparison_metrics.csv").open(newline="", encoding="utf-8")))
    bad_row = next(row for row in rows if row["id"] == "bad_schema")
    assert bad_row["dp3_status"] == "2"
    assert bad_row["dp2_status"] == "2"
    assert bad_row["dp3_data_readiness"] == "data_gaps"
    assert bad_row["dp2_data_readiness"] == "data_gaps"
    assert bad_row["dp3_source_missing"] == ""
    assert bad_row["dp2_source_missing"] == ""
    assert "missing columns" in bad_row["dp3_data_gaps"]
    assert "missing columns" in bad_row["dp2_data_gaps"]
    assert "missing columns" in bad_row["dp3_error"]
    assert "missing columns" in bad_row["dp2_error"]


def test_run_main_path_index_failed_run_entries_keep_stable_metric_schema(tmp_path):
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    (paths_dir / "bad_schema.csv").write_text(
        "s,q1,dq1,d2q1\n"
        "0.0,0.0,1.0,0.0\n"
        "1.0,1.0,1.0,0.0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: bad_schema\n"
        "    csv: bad_schema.csv\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [2.0]\n"
        "q_ddot_abs: [10.0]\n"
        "q_jerk_abs: [100.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "batch"

    status = run_main(
        [
            "--path-index",
            str(path_index),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--tau-rate-dt",
            "0.0002",
        ]
    )

    assert status == 2
    run_entry = json.loads((out_dir / "batch_summary.json").read_text(encoding="utf-8"))["runs"][0]
    assert run_entry["tau_rate_dt"] == 0.0002
    assert run_entry["constraint_check_points"] == 0
    assert run_entry["constraint_check_samples"] == 0
    assert run_entry["constraint_check_source"] == "not_run"
    assert run_entry["constraint_utilization_csv"] is None
    assert run_entry["violation_count"] == 0
    assert run_entry["max_utilization"] == {}
    assert run_entry["active_constraint_threshold"] == 0.99
    assert run_entry["active_constraint_percent"] == {}
    assert run_entry["most_restrictive_constraint_percent"] == {}


def test_run_main_path_index_failed_run_summary_preserves_json_safe_config(tmp_path):
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    (paths_dir / "joint.csv").write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.5,0.5,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: bad_config\n"
        "    csv: joint.csv\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [2.0]\n"
        "q_ddot_abs: [10.0]\n"
        "q_jerk_abs: [100.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "batch"

    status = run_main(
        [
            "--path-index",
            str(path_index),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--tau-rate-dt",
            "inf",
        ]
    )

    assert status == 2
    batch_text = (out_dir / "batch_summary.json").read_text(encoding="utf-8")
    assert "Infinity" not in batch_text
    assert "NaN" not in batch_text
    batch_summary = json.loads(batch_text)
    assert batch_summary["runs"][0]["tau_rate_dt"] is None
    run_summary = json.loads((out_dir / "bad_config" / "summary.json").read_text(encoding="utf-8"))
    assert run_summary["config"]["tau_rate_dt"] is None


def test_run_main_path_index_reports_invalid_index_schema_with_batch_summary(tmp_path):
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    (paths_dir / "joint.csv").write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.5,0.5,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: bad_endpoint\n"
        "    csv: joint.csv\n"
        "    zs: -0.1\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [2.0]\n"
        "q_ddot_abs: [10.0]\n"
        "q_jerk_abs: [100.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "batch"

    status = run_main(
        [
            "--path-index",
            str(path_index),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert status == 2
    batch_summary = json.loads((out_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert batch_summary["status"] == "failed"
    assert batch_summary["path_index"] == str(path_index)
    assert batch_summary["path_count"] == 0
    assert batch_summary["runs"] == []
    assert batch_summary["data_readiness"] == "data_gaps"
    assert batch_summary["source_missing"] == []
    assert "zs" in batch_summary["error"]
    assert "nonnegative" in batch_summary["error"]
    assert batch_summary["data_gaps"] == [batch_summary["error"]]


def test_run_main_path_index_reports_invalid_yaml_syntax_with_batch_summary(tmp_path):
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text("paths: [\n", encoding="utf-8")
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [2.0]\n"
        "q_ddot_abs: [10.0]\n"
        "q_jerk_abs: [100.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "batch"

    status = run_main(
        [
            "--path-index",
            str(path_index),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert status == 2
    batch_summary = json.loads((out_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert batch_summary["status"] == "failed"
    assert batch_summary["path_index"] == str(path_index)
    assert batch_summary["path_count"] == 0
    assert batch_summary["runs"] == []
    assert batch_summary["data_readiness"] == "data_gaps"
    assert batch_summary["source_missing"] == []
    assert "path_index YAML" in batch_summary["error"]
    assert batch_summary["data_gaps"] == [batch_summary["error"]]


def test_run_main_path_index_reports_global_endpoint_speed_above_z_max_as_batch_config_error(tmp_path):
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    (paths_dir / "joint.csv").write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.5,0.5,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: global_bad\n"
        "    csv: joint.csv\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [2.0]\n"
        "q_ddot_abs: [10.0]\n"
        "q_jerk_abs: [100.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "batch"

    status = run_main(
        [
            "--path-index",
            str(path_index),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--z-start",
            "1.0",
            "--z-max",
            "0.5",
        ]
    )

    assert status == 2
    batch_summary = json.loads((out_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert batch_summary["status"] == "failed"
    assert batch_summary["path_count"] == 1
    assert batch_summary["runs"] == []
    assert "z_start" in batch_summary["error"]
    assert "z_max" in batch_summary["error"]
    assert batch_summary["data_readiness"] == "data_gaps"
    assert batch_summary["data_gaps"] == [batch_summary["error"]]
    sources = batch_summary["reproduction_sources"]
    assert sources["path_index"] == str(path_index)
    assert sources["limits"] == str(limits_yaml)
    assert sources["dynamics_backend"] == "none"


def test_run_main_path_index_reports_infeasible_paths_without_nonfinite_totals(tmp_path):
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    (paths_dir / "too_tight.csv").write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.5,0.5,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: tight\n"
        "    csv: too_tight.csv\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [1e-12]\n"
        "q_ddot_abs: [1e-12]\n"
        "q_jerk_abs: [1e-12]\n"
        "tau_abs: [1e-12]\n"
        "tau_rate_abs: [1e-12]\n"
        "mechanical_power:\n"
        "  lower: -1e-12\n"
        "  upper: 1e-12\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "batch"

    status = run_main(
        [
            "--path-index",
            str(path_index),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "5",
            "--nz",
            "3",
            "--nch",
            "3",
        ]
    )

    assert status == 1
    batch_summary = json.loads((out_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert batch_summary["runs"][0]["status"] == 1
    assert batch_summary["runs"][0]["feasible"] is False
    assert batch_summary["runs"][0]["total_time"] == 0.0
    assert batch_summary["runs"][0]["te_scale"] == "-"
    assert batch_summary["runs"][0]["required_time_scale_st"] == "-"
    assert batch_summary["runs"][0]["executable_with_st"] == "no"
    assert batch_summary["total_trajectory_time_s"] == 0.0
    assert batch_summary["total_te_scale_s"] == 0.0


def test_run_main_path_index_preserves_non_time_scalable_te_scale_marker(tmp_path, monkeypatch):
    paths_dir = tmp_path / "paths"
    paths_dir.mkdir()
    path_csv = paths_dir / "path.csv"
    path_csv.write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "1.0,1.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: non_scalable\n"
        "    csv: path.csv\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [10.0]\n"
        "q_ddot_abs: [10.0]\n"
        "q_jerk_abs: [10.0]\n"
        "tau_abs: [10.0]\n"
        "tau_rate_abs: [10.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    audit = ConstraintAudit(
        ok=False,
        max_utilization={"q_position": 1.5},
        violations=[
            ConstraintViolation(
                quantity="q_position",
                sample=1,
                axis=0,
                value=1.5,
                limit=1.0,
                utilization=1.5,
            )
        ],
    )
    result = TrajectoryResult(
        feasible=True,
        t=np.array([0.0, 1.0]),
        s=np.array([0.0, 1.0]),
        z=np.array([1.0, 1.0]),
        z_s=np.zeros(2),
        z_ss=np.zeros(2),
        total_time=1.0,
        audit=audit,
        grid_s=np.array([0.0, 1.0]),
        grid_z=np.zeros((2, 1)),
        policy=np.zeros((2, 1), dtype=np.int32),
        segment_kinds=["C2"],
        segment_profiles=None,
        objective_cost=1.0,
        objective_time_cost=1.0,
        objective_drive_power_cost=0.0,
    )
    quantities = SimpleNamespace(
        q=np.zeros((2, 1)),
        q_dot=np.zeros((2, 1)),
        q_ddot=np.zeros((2, 1)),
        q_jerk=np.zeros((2, 1)),
        tau=np.zeros((2, 1)),
        tau_rate=np.zeros((2, 1)),
        mechanical_power=np.zeros(2),
        q_dot_utilization=np.zeros((2, 1)),
        q_ddot_utilization=np.zeros((2, 1)),
        q_jerk_utilization=np.zeros((2, 1)),
        tau_utilization=np.zeros((2, 1)),
        tau_rate_utilization=np.zeros((2, 1)),
        mechanical_power_utilization=np.zeros(2),
        q_position_utilization=np.array([[0.0], [1.5]]),
        drive_power=None,
    )
    monkeypatch.setattr(cli, "optimize_dp3", lambda **_: result)
    monkeypatch.setattr(cli, "_constraint_check_result", lambda **_: (result, quantities, audit, "test_dense"))
    out_dir = tmp_path / "batch"

    status = run_main(
        [
            "--path-index",
            str(path_index),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "3",
            "--nz",
            "3",
            "--nch",
            "2",
        ]
    )

    assert status == 1
    run_summary = json.loads((out_dir / "non_scalable" / "summary.json").read_text(encoding="utf-8"))
    assert run_summary["te_scale"] == "-"
    batch_summary = json.loads((out_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert batch_summary["runs"][0]["te_scale"] == "-"
    assert batch_summary["runs"][0]["executable_with_st"] == "no"
    metric_rows = list(csv.DictReader((out_dir / "batch_metrics.csv").open(newline="", encoding="utf-8")))
    assert metric_rows[0]["te_scale"] == ""


def test_run_main_supports_dp2_baseline(tmp_path):
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
        "q_dot_abs: [1.0]\n"
        "q_ddot_abs: [5.0]\n"
        "q_jerk_abs: [50.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "dp2"

    status = run_main(
        [
            "--method",
            "dp2",
            "--path-csv",
            str(path_csv),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "5",
            "--nz",
            "7",
            "--nch",
            "5",
        ]
    )

    assert status == 0
    assert '"method": "DP2"' in (out_dir / "summary.json").read_text(encoding="utf-8")
    summary_text = (out_dir / "summary.json").read_text(encoding="utf-8")
    assert '"required_time_scale_st"' in summary_text
    assert '"te_scale"' in summary_text


def test_run_main_supports_dp2_dp3_comparison(tmp_path):
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
        "q_dot_abs: [1.0]\n"
        "q_ddot_abs: [5.0]\n"
        "q_jerk_abs: [50.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "compare"

    status = run_main(
        [
            "--method",
            "compare",
            "--path-csv",
            str(path_csv),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "5",
            "--nz",
            "7",
            "--nch",
            "5",
            "--constraint-check-points",
            "17",
        ]
    )

    assert status == 0
    assert (out_dir / "dp3" / "summary.json").exists()
    assert (out_dir / "dp2" / "summary.json").exists()
    comparison = json.loads((out_dir / "comparison_summary.json").read_text(encoding="utf-8"))
    assert comparison["method"] == "compare"
    assert comparison["runs"]["dp3"]["method"] == "DP3"
    assert comparison["runs"]["dp2"]["method"] == "DP2"
    assert comparison["runs"]["dp3"]["status"] == 0
    assert comparison["runs"]["dp2"]["status"] == 0
    assert comparison["runs"]["dp3"]["summary_json"] == str(out_dir / "dp3" / "summary.json")
    assert comparison["runs"]["dp2"]["summary_json"] == str(out_dir / "dp2" / "summary.json")
    assert comparison["runs"]["dp3"]["constraint_check_source"] == "dense_segment_profiles"
    assert comparison["runs"]["dp2"]["constraint_check_source"] == "dense_segment_profiles"
    assert comparison["runs"]["dp3"]["constraint_check_samples"] == 17
    assert comparison["runs"]["dp2"]["constraint_check_samples"] == 17
    assert comparison["runs"]["dp3"]["max_utilization"]["q_dot"] >= 0.0
    assert comparison["runs"]["dp2"]["max_utilization"]["q_dot"] >= 0.0
    assert comparison["runs"]["dp3"]["active_constraint_percent"]["q_dot"] >= 0.0
    assert comparison["runs"]["dp2"]["active_constraint_percent"]["q_dot"] >= 0.0
    assert comparison["runs"]["dp3"]["most_restrictive_constraint_percent"]["q_dot"] >= 0.0
    assert comparison["runs"]["dp2"]["most_restrictive_constraint_percent"]["q_dot"] >= 0.0
    assert comparison["comparison_metrics_csv"] == "comparison_metrics.csv"
    comparison_rows = list(csv.DictReader((out_dir / "comparison_metrics.csv").open(newline="", encoding="utf-8")))
    assert [row["id"] for row in comparison_rows] == ["path"]
    assert comparison_rows[0]["dp3_run_dir"] == str(out_dir / "dp3")
    assert comparison_rows[0]["dp2_run_dir"] == str(out_dir / "dp2")
    assert comparison_rows[0]["dp3_constraint_utilization_csv"] == "constraint_utilization.csv"
    assert comparison_rows[0]["dp2_constraint_utilization_csv"] == "constraint_utilization.csv"
    assert comparison_rows[0]["dp3_constraint_violations_csv"] == "constraint_violations.csv"
    assert comparison_rows[0]["dp2_constraint_violations_csv"] == "constraint_violations.csv"
    assert comparison_rows[0]["dp3_constraint_check_source"] == "dense_segment_profiles"
    assert comparison_rows[0]["dp2_constraint_check_source"] == "dense_segment_profiles"
    assert int(comparison_rows[0]["dp3_constraint_check_samples"]) == 17
    assert int(comparison_rows[0]["dp2_constraint_check_samples"]) == 17
    assert int(comparison_rows[0]["constraint_check_samples_dp2_minus_dp3"]) == 0
    assert float(comparison_rows[0]["dp3_max_utilization_q_dot"]) == pytest.approx(
        comparison["runs"]["dp3"]["max_utilization"]["q_dot"]
    )
    assert float(comparison_rows[0]["dp2_max_utilization_q_dot"]) == pytest.approx(
        comparison["runs"]["dp2"]["max_utilization"]["q_dot"]
    )
    assert float(comparison_rows[0]["dp3_active_constraint_percent_q_dot"]) == pytest.approx(
        comparison["runs"]["dp3"]["active_constraint_percent"]["q_dot"]
    )
    assert float(comparison_rows[0]["dp2_active_constraint_percent_q_dot"]) == pytest.approx(
        comparison["runs"]["dp2"]["active_constraint_percent"]["q_dot"]
    )
    assert float(comparison_rows[0]["max_utilization_dp2_minus_dp3_q_dot"]) == pytest.approx(
        comparison["runs"]["dp2"]["max_utilization"]["q_dot"]
        - comparison["runs"]["dp3"]["max_utilization"]["q_dot"]
    )
    assert float(comparison_rows[0]["active_constraint_percent_dp2_minus_dp3_q_dot"]) == pytest.approx(
        comparison["runs"]["dp2"]["active_constraint_percent"]["q_dot"]
        - comparison["runs"]["dp3"]["active_constraint_percent"]["q_dot"]
    )
    assert float(comparison_rows[0]["most_restrictive_constraint_percent_dp2_minus_dp3_q_dot"]) == pytest.approx(
        comparison["runs"]["dp2"]["most_restrictive_constraint_percent"]["q_dot"]
        - comparison["runs"]["dp3"]["most_restrictive_constraint_percent"]["q_dot"]
    )
    assert comparison_rows[0]["dp3_required_time_scale_st"] == "100%"
    assert comparison_rows[0]["dp2_required_time_scale_st"] == "100%"
    assert comparison_rows[0]["dp3_executable_with_st"] == "yes"
    assert comparison_rows[0]["dp2_executable_with_st"] == "yes"
    assert float(comparison_rows[0]["dp3_t_e_s"]) > 0.0
    assert float(comparison_rows[0]["dp2_t_e_s"]) > 0.0
    assert float(comparison_rows[0]["dp3_te_scale"]) == pytest.approx(float(comparison_rows[0]["dp3_t_e_s"]))
    assert float(comparison_rows[0]["dp2_te_scale"]) == pytest.approx(float(comparison_rows[0]["dp2_t_e_s"]))
    assert float(comparison_rows[0]["te_scale_dp2_minus_dp3"]) == pytest.approx(
        float(comparison_rows[0]["dp2_te_scale"]) - float(comparison_rows[0]["dp3_te_scale"])
    )
    assert float(comparison_rows[0]["total_time_dp2_minus_dp3"]) == pytest.approx(
        float(comparison_rows[0]["dp2_t_e_s"]) - float(comparison_rows[0]["dp3_t_e_s"])
    )
    assert float(comparison_rows[0]["objective_time_cost_dp2_minus_dp3"]) == pytest.approx(
        float(comparison_rows[0]["dp2_objective_time_cost"])
        - float(comparison_rows[0]["dp3_objective_time_cost"])
    )
    assert float(comparison_rows[0]["objective_drive_power_cost_dp2_minus_dp3"]) == pytest.approx(
        float(comparison_rows[0]["dp2_objective_drive_power_cost"])
        - float(comparison_rows[0]["dp3_objective_drive_power_cost"])
    )
    assert int(comparison_rows[0]["dp3_violation_count"]) == 0
    assert int(comparison_rows[0]["dp2_violation_count"]) == 0
    assert set(comparison["delta"]) == {
        "total_time_dp2_minus_dp3",
        "te_scale_dp2_minus_dp3",
        "cpu_time_dp2_minus_dp3",
        "violation_count_dp2_minus_dp3",
        "max_utilization_dp2_minus_dp3",
        "active_constraint_percent_dp2_minus_dp3",
        "most_restrictive_constraint_percent_dp2_minus_dp3",
        "objective_cost_dp2_minus_dp3",
        "objective_time_cost_dp2_minus_dp3",
        "objective_drive_power_cost_dp2_minus_dp3",
    }
    assert comparison["delta"]["violation_count_dp2_minus_dp3"] == (
        comparison["runs"]["dp2"]["violation_count"] - comparison["runs"]["dp3"]["violation_count"]
    )
    assert comparison["delta"]["max_utilization_dp2_minus_dp3"]["q_dot"] == pytest.approx(
        comparison["runs"]["dp2"]["max_utilization"]["q_dot"]
        - comparison["runs"]["dp3"]["max_utilization"]["q_dot"]
    )
    assert comparison["delta"]["active_constraint_percent_dp2_minus_dp3"]["q_dot"] == pytest.approx(
        comparison["runs"]["dp2"]["active_constraint_percent"]["q_dot"]
        - comparison["runs"]["dp3"]["active_constraint_percent"]["q_dot"]
    )
    assert comparison["delta"]["most_restrictive_constraint_percent_dp2_minus_dp3"]["q_dot"] == pytest.approx(
        comparison["runs"]["dp2"]["most_restrictive_constraint_percent"]["q_dot"]
        - comparison["runs"]["dp3"]["most_restrictive_constraint_percent"]["q_dot"]
    )
    assert comparison["delta"]["objective_time_cost_dp2_minus_dp3"] == pytest.approx(
        comparison["runs"]["dp2"]["objective_time_cost"] - comparison["runs"]["dp3"]["objective_time_cost"]
    )
    assert comparison["delta"]["objective_drive_power_cost_dp2_minus_dp3"] == pytest.approx(
        comparison["runs"]["dp2"]["objective_drive_power_cost"]
        - comparison["runs"]["dp3"]["objective_drive_power_cost"]
    )
    assert comparison["delta"]["te_scale_dp2_minus_dp3"] == pytest.approx(
        comparison["runs"]["dp2"]["te_scale"] - comparison["runs"]["dp3"]["te_scale"]
    )
    assert comparison["config"]["constraint_check_points"] == 17


def test_run_main_dp2_reports_time_scale_for_dense_third_order_violations(tmp_path):
    path_csv = tmp_path / "curved_path.csv"
    path_csv.write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,10.0,0.0\n"
        "0.5,0.5,1.0,10.0,0.0\n"
        "1.0,1.0,1.0,10.0,0.0\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [10.0]\n"
        "q_ddot_abs: [1000.0]\n"
        "q_jerk_abs: [1e-6]\n"
        "tau_abs: [1000.0]\n"
        "tau_rate_abs: [1000.0]\n"
        "mechanical_power:\n"
        "  lower: -1000.0\n"
        "  upper: 1000.0\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "dp2-third-order"

    status = run_main(
        [
            "--method",
            "dp2",
            "--path-csv",
            str(path_csv),
            "--limits",
            str(limits_yaml),
            "--out-dir",
            str(out_dir),
            "--ns",
            "5",
            "--nz",
            "7",
            "--nch",
            "5",
            "--constraint-check-points",
            "31",
        ]
    )

    assert status == 1
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["feasible"] is False
    assert summary["constraint_violations_csv"] == "constraint_violations.csv"
    assert any(violation["quantity"] == "q_jerk" for violation in summary["violations"])
    violation_rows = list(csv.DictReader((out_dir / "constraint_violations.csv").open(newline="", encoding="utf-8")))
    assert any(row["quantity"] == "q_jerk" for row in violation_rows)
    q_jerk_rows = [row for row in violation_rows if row["quantity"] == "q_jerk"]
    assert q_jerk_rows[0]["axis"] == "1"
    assert float(q_jerk_rows[0]["utilization"]) > 1.0
    assert 0.0 <= float(q_jerk_rows[0]["s"]) <= 1.0
    assert summary["required_time_scale_st"] != "-"
    assert summary["executable_with_st"] == "yes"
    assert isinstance(summary["te_scale"], float)
    assert summary["te_scale"] > summary["total_time"]
