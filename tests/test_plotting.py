import json
import re
from pathlib import Path

from dp3_topp.plotting import plot_main


def test_plot_main_writes_constraint_utilization_svg(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "constraint_utilization.csv").write_text(
        "t,s,z,q_dot,q_ddot,q_jerk,tau,tau_rate,mechanical_power\n"
        "0.0,0.0,0.0,0.1,0.2,0.3,0.4,0.5,0.6\n"
        "1.0,0.5,0.2,0.8,0.1,0.4,0.2,0.3,0.5\n"
        "2.0,1.0,0.0,0.2,0.7,0.9,0.1,0.2,0.3\n",
        encoding="utf-8",
    )

    status = plot_main(["--run", str(run_dir)])

    assert status == 0
    svg = (run_dir / "plots" / "constraint_utilization.svg").read_text(encoding="utf-8")
    assert "<svg" in svg
    assert '<rect width="100%" height="100%" fill="#fff"/>' in svg
    assert "Constraint Utilization" in svg
    assert "q_dot" in svg
    assert "q_jerk" in svg
    assert "polyline" in svg
    manifest = json.loads((run_dir / "plots" / "plots_manifest.json").read_text(encoding="utf-8"))
    assert manifest["plots"] == ["constraint_utilization.svg"]


def test_plot_main_writes_joint_quantity_svgs(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "quantities.csv").write_text(
        "t,s,z,q1,q2,q_dot1,q_dot2,q_ddot1,q_ddot2,q_jerk1,q_jerk2,tau1,tau2,tau_rate1,tau_rate2,mechanical_power\n"
        "0.0,0.0,0.0,0.0,0.1,0.0,0.2,0.0,0.3,0.0,0.4,1.0,2.0,0.1,0.2,0.0\n"
        "1.0,0.5,0.2,0.4,0.2,0.5,0.3,0.6,0.2,0.7,0.1,1.5,2.5,0.3,0.4,0.0\n"
        "2.0,1.0,0.0,0.2,0.3,0.1,0.4,0.2,0.5,0.3,0.6,1.2,2.2,0.5,0.6,0.0\n",
        encoding="utf-8",
    )

    status = plot_main(["--run", str(run_dir)])

    assert status == 0
    expected = {
        "joint_position.svg": "Joint Position",
        "joint_velocity.svg": "Joint Velocity",
        "joint_acceleration.svg": "Joint Acceleration",
        "joint_jerk.svg": "Joint Jerk",
        "joint_torque.svg": "Joint Torque",
        "joint_torque_rate.svg": "Joint Torque Rate",
    }
    for filename, title in expected.items():
        svg = (run_dir / "plots" / filename).read_text(encoding="utf-8")
        assert title in svg
        assert "joint 1" in svg
        assert "joint 2" in svg


def test_plot_main_writes_mvc_svg_from_trajectory_and_config(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "trajectory.csv").write_text(
        "t,s,z,z_s,z_ss\n"
        "0.0,0.0,0.0,0.0,0.0\n"
        "1.0,0.5,0.25,0.0,0.0\n"
        "2.0,1.0,0.0,0.0,0.0\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(json.dumps({"config": {"z_max": 1.0}}), encoding="utf-8")

    status = plot_main(["--run", str(run_dir)])

    assert status == 0
    svg = (run_dir / "plots" / "mvc.svg").read_text(encoding="utf-8")
    assert "MVC and Path Speed" in svg
    assert "path speed" in svg
    assert "MVC" in svg


def test_plot_main_computes_mvc_with_toppra_from_path_and_limits(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path_csv = tmp_path / "path.csv"
    path_csv.write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.5,0.2,2.0,0.0,0.0\n"
        "1.0,1.0,0.5,0.0,0.0\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [1.0]\n"
        "q_ddot_abs: [10.0]\n"
        "q_jerk_abs: [100.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    (run_dir / "trajectory.csv").write_text(
        "t,s,z,z_s,z_ss\n"
        "0.0,0.0,0.25,0.0,0.0\n"
        "1.0,0.5,0.25,0.0,0.0\n"
        "2.0,1.0,0.25,0.0,0.0\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {"z_max": 0.01},
                "reproduction_sources": {
                    "path_source": str(path_csv),
                    "limits": str(limits_yaml),
                },
            }
        ),
        encoding="utf-8",
    )

    status = plot_main(["--run", str(run_dir)])

    assert status == 0
    svg = (run_dir / "plots" / "mvc.svg").read_text(encoding="utf-8")
    assert "MVC (TOPPRA)" in svg
    assert "TOPPRA profile" in svg
    assert "configured speed cap" in svg
    polylines = re.findall(r'<polyline points="([^"]+)"', svg)
    assert len(polylines) >= 4
    mvc_y_values = [point.split(",")[1] for point in polylines[2].split()]
    assert len(set(mvc_y_values)) > 1


def test_plot_main_draws_toppra_mvc_on_path_grid_instead_of_sparse_trajectory_grid(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path_csv = tmp_path / "path.csv"
    path_csv.write_text(
        "s,q1,dq1,d2q1,d3q1\n"
        "0.0,0.0,1.0,0.0,0.0\n"
        "0.25,0.15,1.4,0.0,0.0\n"
        "0.5,0.45,2.0,0.0,0.0\n"
        "0.75,0.7,1.2,0.0,0.0\n"
        "1.0,1.0,0.8,0.0,0.0\n",
        encoding="utf-8",
    )
    limits_yaml = tmp_path / "limits.yaml"
    limits_yaml.write_text(
        "q_dot_abs: [1.0]\n"
        "q_ddot_abs: [10.0]\n"
        "q_jerk_abs: [100.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    (run_dir / "trajectory.csv").write_text(
        "t,s,z,z_s,z_ss\n"
        "0.0,0.0,0.25,0.0,0.0\n"
        "1.0,0.5,0.25,0.0,0.0\n"
        "2.0,1.0,0.25,0.0,0.0\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"reproduction_sources": {"path_source": str(path_csv), "limits": str(limits_yaml)}}),
        encoding="utf-8",
    )

    status = plot_main(["--run", str(run_dir)])

    assert status == 0
    svg = (run_dir / "plots" / "mvc.svg").read_text(encoding="utf-8")
    polylines = re.findall(r'<polyline points="([^"]+)"', svg)
    assert len(polylines) >= 2
    assert len(polylines[1].split()) > 3


def test_plot_main_writes_end_effector_speed_svg_when_mujoco_model_is_available(tmp_path):
    model = tmp_path / "one_dof.xml"
    model.write_text(
        '<mujoco model="one_dof">\n'
        '  <compiler angle="radian"/>\n'
        '  <worldbody>\n'
        '    <body name="link" pos="0 0 0">\n'
        '      <joint name="joint1" type="hinge" axis="0 0 1" limited="true" range="-3.14 3.14"/>\n'
        '      <geom type="sphere" size="0.01"/>\n'
        '      <site name="tcp" pos="1 0 0" size="0.01"/>\n'
        "    </body>\n"
        "  </worldbody>\n"
        '  <actuator><position name="joint1_act" joint="joint1"/></actuator>\n'
        "</mujoco>\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "quantities.csv").write_text(
        "t,s,z,q1,q_dot1,q_ddot1,q_jerk1,tau1,tau_rate1,mechanical_power\n"
        "0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0\n"
        "1.0,0.5,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0\n"
        "2.0,1.0,0.0,2.0,1.0,0.0,0.0,0.0,0.0,0.0\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"reproduction_sources": {"model": str(model)}, "ik": {"settings": {"ee_site": "tcp"}}}),
        encoding="utf-8",
    )

    status = plot_main(["--run", str(run_dir)])

    assert status == 0
    svg = (run_dir / "plots" / "end_effector_speed.svg").read_text(encoding="utf-8")
    assert "End-Effector Speed" in svg
    assert "tcp speed" in svg
    assert "m/s" in svg


def test_plot_main_writes_toppra_mvc_and_end_effector_speed_overlay(tmp_path):
    model = tmp_path / "one_dof.xml"
    model.write_text(
        '<mujoco model="one_dof">\n'
        '  <compiler angle="radian"/>\n'
        '  <worldbody>\n'
        '    <body name="link" pos="0 0 0">\n'
        '      <joint name="joint1" type="hinge" axis="0 0 1" limited="true" range="-3.14 3.14"/>\n'
        '      <geom type="sphere" size="0.01"/>\n'
        '      <site name="tcp" pos="1 0 0" size="0.01"/>\n'
        "    </body>\n"
        "  </worldbody>\n"
        '  <actuator><position name="joint1_act" joint="joint1"/></actuator>\n'
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
        "q_dot_abs: [1.0]\n"
        "q_ddot_abs: [10.0]\n"
        "q_jerk_abs: [100.0]\n"
        "tau_abs: [100.0]\n"
        "tau_rate_abs: [100.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "trajectory.csv").write_text(
        "t,s,z,z_s,z_ss\n"
        "0.0,0.0,0.0,0.0,0.0\n"
        "1.0,0.5,0.25,0.0,0.0\n"
        "2.0,1.0,0.0,0.0,0.0\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "reproduction_sources": {
                    "path_source": str(path_csv),
                    "limits": str(limits_yaml),
                    "model": str(model),
                },
                "ik": {"settings": {"ee_site": "tcp"}},
            }
        ),
        encoding="utf-8",
    )

    status = plot_main(["--run", str(run_dir)])

    assert status == 0
    svg = (run_dir / "plots" / "mvc_end_effector_speed.svg").read_text(encoding="utf-8")
    assert "MVC + End-Effector Speed" in svg
    assert "TCP speed (DP3)" in svg
    assert "MVC (TOPPRA, TCP)" in svg
    assert "m/s" in svg


def test_plot_main_writes_batch_metric_svgs(tmp_path):
    run_dir = tmp_path / "batch"
    run_dir.mkdir()
    (run_dir / "batch_metrics.csv").write_text(
        "id,t_e_s,t_cpu_s,violation_count\n"
        "path_01,10.0,1.5,0\n"
        "path_02,12.0,1.8,3\n",
        encoding="utf-8",
    )

    status = plot_main(["--run", str(run_dir)])

    assert status == 0
    times = (run_dir / "plots" / "batch_times.svg").read_text(encoding="utf-8")
    violations = (run_dir / "plots" / "batch_violations.svg").read_text(encoding="utf-8")
    assert "Batch Times" in times
    assert "path_01" in times
    assert "t_e_s" in times
    assert "t_cpu_s" in times
    assert "Batch Violations" in violations
    assert "violation_count" in violations


def test_plot_main_writes_comparison_svgs(tmp_path):
    run_dir = tmp_path / "compare"
    run_dir.mkdir()
    (run_dir / "comparison_metrics.csv").write_text(
        "id,dp3_t_e_s,dp2_t_e_s,dp3_t_cpu_s,dp2_t_cpu_s,dp3_violation_count,dp2_violation_count\n"
        "path_01,10.0,11.0,2.0,1.5,0,4\n"
        "path_02,8.0,9.5,1.8,1.3,0,2\n",
        encoding="utf-8",
    )

    status = plot_main(["--run", str(run_dir), "--out-dir", str(tmp_path / "figures")])

    assert status == 0
    times = (tmp_path / "figures" / "comparison_times.svg").read_text(encoding="utf-8")
    cpu = (tmp_path / "figures" / "comparison_cpu.svg").read_text(encoding="utf-8")
    violations = (tmp_path / "figures" / "comparison_violations.svg").read_text(encoding="utf-8")
    assert "DP3 vs DP2 Trajectory Time" in times
    assert "dp3_t_e_s" in times
    assert "dp2_t_e_s" in times
    assert "DP3 vs DP2 CPU Time" in cpu
    assert "DP3 vs DP2 Violations" in violations
    assert "dp2_violation_count" in violations
