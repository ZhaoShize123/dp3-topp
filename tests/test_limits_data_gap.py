import json
from pathlib import Path

import yaml

from dp3_topp.cli import default_dyn_dir, validate_data, validate_data_main, write_path_index_template_main
from dp3_topp.limits_schema import full_reproduction_gaps, write_t12a_limits_template
from dp3_topp.path_data import PathData


def _write_complete_limits(path: Path, dof: int = 6) -> None:
    repeated = ", ".join(["1"] * dof)
    torque_speed = "\n".join(["  - [[0, 10], [1, 5]]" for _ in range(dof)])
    path.write_text(
        f"q_dot_abs: [{repeated}]\n"
        f"q_ddot_abs: [{repeated}]\n"
        f"q_jerk_abs: [{repeated}]\n"
        f"tau_abs: [{repeated}]\n"
        f"tau_rate_abs: [{repeated}]\n"
        "mechanical_power:\n"
        "  lower: -100\n"
        "  upper: 100\n"
        "torque_speed_breakpoints:\n"
        f"{torque_speed}\n"
        "friction:\n"
        f"  coulomb: [{repeated}]\n"
        f"  viscous: [{repeated}]\n"
        "motor:\n"
        f"  gear_ratio: [{repeated}]\n"
        f"  torque_constant: [{repeated}]\n"
        f"  stator_resistance: [{repeated}]\n",
        encoding="utf-8",
    )


def _write_path_csv(path: Path, dof: int = 6) -> None:
    headers = ["s"]
    headers.extend(f"q{i}" for i in range(1, dof + 1))
    headers.extend(f"dq{i}" for i in range(1, dof + 1))
    headers.extend(f"d2q{i}" for i in range(1, dof + 1))
    headers.extend(f"d3q{i}" for i in range(1, dof + 1))
    rows = []
    for s_value in (0.0, 1.0):
        rows.append(
            [s_value]
            + [s_value] * dof
            + [1.0] * dof
            + [0.0] * dof
            + [0.0] * dof
        )
    path.write_text(
        ",".join(headers) + "\n" + "\n".join(",".join(f"{value:g}" for value in row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_mujoco_model(path: Path, dof: int = 6) -> None:
    joints = []
    for axis in range(1, dof + 1):
        joints.append(
            f'      <joint name="joint{axis}" type="hinge" axis="0 0 1" limited="true" range="-10 10"/>\n'
            f'      <geom name="link{axis}" type="capsule" fromto="0 0 0 1 0 0" size="0.01" density="1000"/>\n'
        )
    path.write_text(
        '<mujoco model="test_robot">\n'
        '  <compiler angle="radian"/>\n'
        '  <option gravity="0 0 0"/>\n'
        "  <worldbody>\n"
        '    <body name="base" pos="0 0 0">\n'
        + "".join(joints)
        + "    </body>\n"
        + "  </worldbody>\n"
        + "</mujoco>\n",
        encoding="utf-8",
    )


def test_t12a_limits_template_contains_required_paper_fields(tmp_path):
    out = tmp_path / "limits.template.yaml"

    write_t12a_limits_template(out, dof=6)

    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["robot"] == "T12A"
    assert data["units"]["mechanical_power"] == "W"
    assert len(data["q_dot_abs"]) == 6
    assert data["torque_speed_breakpoints"][0][0] == [0.0, None]
    assert data["friction"]["coulomb"][0] is None
    assert data["motor"]["gear_ratio"][0] is None


def test_t12a_limits_template_can_embed_mujoco_joint_metadata(tmp_path):
    out = tmp_path / "limits.template.yaml"

    write_t12a_limits_template(
        out,
        dof=6,
        model_path=default_dyn_dir() / "models" / "T12A" / "T12A-14.xml",
    )

    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["joint_names"] == [f"joint_{axis}" for axis in range(1, 7)]
    assert data["units"]["q_position"] == "rad"
    assert data["q_position"]["lower"] == [-2.967, -1.745, -2.792, -3.316, -2.268, -6.283]
    assert data["q_position"]["upper"] == [2.967, 1.745, 2.792, 3.316, 2.268, 6.283]
    assert data["q_dot_abs"] == [None] * 6
    assert data["tau_abs"] == [None] * 6


def test_path_index_template_contains_14_placeholder_paths(tmp_path):
    out = tmp_path / "path_index.template.yaml"

    status = write_path_index_template_main(["--out", str(out), "--count", "14"])

    assert status == 0
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["template"] is True
    assert data["expected_path_count"] == 14
    assert len(data["paths"]) == 14
    assert data["paths"][0] == {"id": "path_01", "csv": "path_01.csv", "zs": 0.0, "ze": 0.0}
    assert data["paths"][-1] == {"id": "path_14", "csv": "path_14.csv", "zs": 0.0, "ze": 0.0}


def test_default_t12a_path_index_template_is_present_and_placeholder_only():
    template = default_dyn_dir() / "paths" / "path_index.template.yaml"

    data = yaml.safe_load(template.read_text(encoding="utf-8"))

    assert data["template"] is True
    assert data["expected_path_count"] == 14
    assert len(data["paths"]) == 14
    assert [entry["id"] for entry in data["paths"]] == [f"path_{index:02d}" for index in range(1, 15)]
    assert [entry["csv"] for entry in data["paths"]] == [f"path_{index:02d}.csv" for index in range(1, 15)]
    assert all(entry["zs"] == 0.0 and entry["ze"] == 0.0 for entry in data["paths"])


def test_selected_t12a_limits_and_14_path_dataset_are_complete():
    dyn_dir = default_dyn_dir()
    limits = dyn_dir / "models" / "T12A" / "limits.yaml"
    path_index = dyn_dir / "paths" / "path_index.yaml"

    assert limits.exists()
    assert full_reproduction_gaps(limits, expected_dof=6) == []
    data = yaml.safe_load(path_index.read_text(encoding="utf-8"))
    assert data.get("template") is not True
    assert data["expected_path_count"] == 14
    assert len(data["paths"]) == 14
    ids = [entry["id"] for entry in data["paths"]]
    assert len(ids) == len(set(ids))
    for entry in data["paths"]:
        csv_path = path_index.parent / entry["csv"]
        assert csv_path.exists()
        path = PathData.from_csv(csv_path)
        assert path.dof == 6
        assert path.samples >= 11
        assert entry.get("selected_by") == "codex"


def test_validate_data_rejects_marked_path_index_template_as_reproduction_input(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text((default_dyn_dir() / "paths" / "path_index.template.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    report = validate_data(
        robot_dir=robot_dir,
        paths_dir=paths_dir,
        limits_path=limits,
        path_index=path_index,
        expected_path_count=14,
    )

    assert not report.ok
    assert any("path_index" in gap and "template" in gap for gap in report.data_gaps)


def test_full_reproduction_gaps_require_friction_motor_and_torque_speed(tmp_path):
    limits = tmp_path / "limits.yaml"
    limits.write_text(
        "q_dot_abs: [1, 1]\n"
        "q_ddot_abs: [2, 2]\n"
        "q_jerk_abs: [3, 3]\n"
        "tau_abs: [4, 4]\n"
        "tau_rate_abs: [5, 5]\n"
        "mechanical_power:\n"
        "  lower: -1\n"
        "  upper: 1\n",
        encoding="utf-8",
    )

    gaps = full_reproduction_gaps(limits, expected_dof=2)

    assert "torque_speed_breakpoints" in gaps
    assert "friction.coulomb" in gaps
    assert "friction.viscous" in gaps
    assert "motor.gear_ratio" in gaps
    assert "motor.torque_constant" in gaps
    assert "motor.stator_resistance" in gaps


def test_full_reproduction_gaps_accepts_paper_style_axis_lower_upper_limits(tmp_path):
    limits = tmp_path / "limits.yaml"
    limits.write_text(
        "q_dot:\n"
        "  lower: [-1, -1]\n"
        "  upper: [1, 1]\n"
        "q_ddot:\n"
        "  lower: [-2, -2]\n"
        "  upper: [2, 2]\n"
        "q_jerk:\n"
        "  lower: [-3, -3]\n"
        "  upper: [3, 3]\n"
        "tau:\n"
        "  lower: [-4, -4]\n"
        "  upper: [4, 4]\n"
        "tau_rate:\n"
        "  lower: [-5, -5]\n"
        "  upper: [5, 5]\n"
        "mechanical_power:\n"
        "  lower: -10\n"
        "  upper: 10\n"
        "torque_speed_breakpoints:\n"
        "  - [[0, 4], [10, 2]]\n"
        "  - [[0, 4], [10, 2]]\n"
        "friction:\n"
        "  coulomb: [0.1, 0.1]\n"
        "  viscous: [0.01, 0.01]\n"
        "motor:\n"
        "  gear_ratio: [100, 100]\n"
        "  torque_constant: [0.2, 0.2]\n"
        "  stator_resistance: [1, 1]\n",
        encoding="utf-8",
    )

    gaps = full_reproduction_gaps(limits, expected_dof=2)

    assert gaps == []


def test_validate_data_reports_limits_data_gaps(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    (paths_dir / "Offline_Traj.txt").write_text("0 0 0 0 0 0\n", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    limits.write_text(
        "q_dot_abs: [1, 1]\n"
        "q_ddot_abs: [2, 2]\n"
        "q_jerk_abs: [3, 3]\n"
        "tau_abs: [4, 4]\n"
        "tau_rate_abs: [5, 5]\n"
        "mechanical_power:\n"
        "  lower: -1\n"
        "  upper: 1\n",
        encoding="utf-8",
    )

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits)

    assert not report.ok
    assert not report.full_reproduction_ready
    assert "friction.viscous" in report.data_gaps


def test_validate_data_reports_malformed_limits_yaml_as_data_gap(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    (paths_dir / "Offline_Traj.txt").write_text("0 0 0 0 0 0\n1 0 0 0 0 0\n", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    limits.write_text("q_dot_abs: [\n", encoding="utf-8")

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits)

    assert not report.ok
    assert not report.full_reproduction_ready
    assert any("limits YAML is invalid" in gap for gap in report.data_gaps)


def test_validate_data_does_not_require_urdf_when_mujoco_model_is_available(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    (paths_dir / "Offline_Traj.txt").write_text(
        "0 0 0 0 0 0\n"
        "1 0 0 0 0 0\n",
        encoding="utf-8",
    )
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits)

    assert report.ok
    assert report.full_reproduction_ready
    assert not any("T12A-14.urdf" in item for item in report.missing)
    assert str(robot_dir / "T12A-14.xml") in report.present


def test_validate_data_rejects_nonfinite_limits_values(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    (paths_dir / "Offline_Traj.txt").write_text("0 0 0 0 0 0\n1 0 0 0 0 0\n", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)
    text = limits.read_text(encoding="utf-8")
    limits.write_text(text.replace("q_dot_abs: [1, 1, 1, 1, 1, 1]", "q_dot_abs: [.nan, 1, 1, 1, 1, 1]"), encoding="utf-8")

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits)

    assert not report.ok
    assert not report.full_reproduction_ready
    assert any("limits.yaml" in gap and "q_dot_abs" in gap and "finite positive" in gap for gap in report.data_gaps)


def test_validate_data_rejects_unloadable_mujoco_model(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    (robot_dir / "T12A-14.xml").write_text("<mujoco><worldbody>", encoding="utf-8")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    (paths_dir / "Offline_Traj.txt").write_text("0 0 0 0 0 0\n", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits)

    assert not report.ok
    assert any("T12A-14.xml" in gap and "MuJoCo" in gap for gap in report.data_gaps)


def test_validate_data_rejects_mujoco_model_with_wrong_dof(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml", dof=1)
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    (paths_dir / "Offline_Traj.txt").write_text("0 0 0 0 0 0\n", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits)

    assert not report.ok
    assert any("T12A-14.xml" in gap and "DOF 1" in gap for gap in report.data_gaps)


def test_validate_data_rejects_malformed_offline_traj_pose_table(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    (paths_dir / "Offline_Traj.txt").write_text("0 0 0\n", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits)

    assert not report.ok
    assert any("Offline_Traj.txt" in gap and "fewer than 6 columns" in gap for gap in report.data_gaps)


def test_validate_data_rejects_nonfinite_offline_traj_pose_table(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    (paths_dir / "Offline_Traj.txt").write_text(
        "0 0 0 0 0 0\n"
        "1 inf 0 0 0 0\n",
        encoding="utf-8",
    )
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits)

    assert not report.ok
    assert any("Offline_Traj.txt" in gap and "finite" in gap for gap in report.data_gaps)


def test_validate_data_rejects_undersampled_offline_traj_pose_table(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    (paths_dir / "Offline_Traj.txt").write_text("0 0 0 0 0 0\n", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits)

    assert not report.ok
    assert any("Offline_Traj.txt" in gap and "at least two poses" in gap for gap in report.data_gaps)


def test_validate_data_can_check_t12a_cartesian_ik_even_when_limits_missing(tmp_path):
    missing_limits = tmp_path / "limits.yaml"

    report = validate_data(
        robot_dir=default_dyn_dir() / "models" / "T12A",
        paths_dir=default_dyn_dir(),
        limits_path=missing_limits,
        check_ik=True,
        ee_body="missing_body",
        ik_check_samples=2,
        ik_max_iters=1,
    )

    assert str(missing_limits) in report.missing
    assert any("IK check failed" in gap and "Body not found: missing_body" in gap for gap in report.data_gaps)


def test_validate_data_ik_check_accepts_mujoco_site_target_even_when_limits_missing(tmp_path):
    missing_limits = tmp_path / "limits.yaml"

    report = validate_data(
        robot_dir=default_dyn_dir() / "models" / "T12A",
        paths_dir=default_dyn_dir(),
        limits_path=missing_limits,
        check_ik=True,
        ee_body="link_6",
        ee_site="tcp",
        ik_check_samples=4,
        ik_max_iters=200,
        ik_orientation_weight=0.0,
    )

    assert str(missing_limits) in report.missing
    assert not any("IK check failed" in gap for gap in report.data_gaps)


def test_validate_data_strict_ik_uses_paper_like_tcp_site_settings_even_when_limits_missing(tmp_path):
    missing_limits = tmp_path / "limits.yaml"

    report = validate_data(
        robot_dir=default_dyn_dir() / "models" / "T12A",
        paths_dir=default_dyn_dir(),
        limits_path=missing_limits,
        check_ik=True,
        ee_body="link_6",
        ee_site="tcp",
        ik_check_samples=40,
        ik_max_iters=200,
        ik_orientation_weight=0.0,
        require_ik_convergence=True,
    )

    assert str(missing_limits) in report.missing
    assert not any("IK check failed" in gap for gap in report.data_gaps)


def test_validate_data_strict_ik_reports_nonconverged_tool_target(tmp_path):
    missing_limits = tmp_path / "limits.yaml"

    report = validate_data(
        robot_dir=default_dyn_dir() / "models" / "T12A",
        paths_dir=default_dyn_dir(),
        limits_path=missing_limits,
        check_ik=True,
        ee_body="link_6",
        ik_check_samples=4,
        ik_max_iters=200,
        ik_orientation_weight=0.0,
        require_ik_convergence=True,
    )

    assert str(missing_limits) in report.missing
    assert any("IK check failed" in gap and "IK failed to converge" in gap for gap in report.data_gaps)


def test_validate_data_main_fails_for_partial_limits_yaml(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    (paths_dir / "Offline_Traj.txt").write_text("0 0 0 0 0 0\n", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    limits.write_text(
        "q_dot_abs: [1, 1]\n"
        "q_ddot_abs: [2, 2]\n"
        "q_jerk_abs: [3, 3]\n"
        "tau_abs: [4, 4]\n"
        "tau_rate_abs: [5, 5]\n"
        "mechanical_power:\n"
        "  lower: -1\n"
        "  upper: 1\n",
        encoding="utf-8",
    )

    status = validate_data_main(["--robot", str(robot_dir), "--paths", str(paths_dir), "--limits", str(limits)])

    assert status == 2


def test_validate_data_main_writes_machine_readable_summary(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    (paths_dir / "Offline_Traj.txt").write_text("0 0 0 0 0 0\n", encoding="utf-8")
    missing_limits = robot_dir / "limits.yaml"
    summary_out = tmp_path / "validate-summary.json"

    status = validate_data_main(
        [
            "--robot",
            str(robot_dir),
            "--paths",
            str(paths_dir),
            "--limits",
            str(missing_limits),
            "--summary-out",
            str(summary_out),
        ]
    )

    assert status == 2
    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    assert summary["ok"] is False
    assert summary["full_reproduction_ready"] is False
    assert summary["data_readiness"] == "source_missing"
    assert str(missing_limits) in summary["missing"]
    assert str(robot_dir / "T12A-14.xml") in summary["present"]
    sources = summary["reproduction_sources"]
    assert sources["limits"] == str(missing_limits)
    assert sources["model"] == str(robot_dir / "T12A-14.xml")
    assert sources["path_index"] is None
    assert sources["dynamics_backend"] == "none"


def test_validate_data_main_summary_records_loaded_limits_and_model_dof(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)
    (paths_dir / "Offline_Traj.txt").write_text("0 0 0 0 0 0\n", encoding="utf-8")
    summary_out = tmp_path / "validate-summary.json"

    status = validate_data_main(
        [
            "--robot",
            str(robot_dir),
            "--paths",
            str(paths_dir),
            "--limits",
            str(limits),
            "--summary-out",
            str(summary_out),
        ]
    )

    assert status == 2
    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    assert summary["data_readiness"] == "data_gaps"
    sources = summary["reproduction_sources"]
    assert sources["limits"] == str(limits)
    assert sources["model"] == str(robot_dir / "T12A-14.xml")
    assert sources["dynamics_backend"] == "none"
    assert sources["limits_dof"] == 6
    assert sources["model_dof"] == 6


def test_validate_data_main_accepts_strict_paper_like_site_ik_options(tmp_path):
    missing_limits = tmp_path / "limits.yaml"
    summary_out = tmp_path / "strict-ik-summary.json"

    status = validate_data_main(
        [
            "--robot",
            str(default_dyn_dir() / "models" / "T12A"),
            "--paths",
            str(default_dyn_dir()),
            "--limits",
            str(missing_limits),
            "--check-ik",
            "--ee-body",
            "link_6",
            "--ee-site",
            "tcp",
            "--ik-check-samples",
            "40",
            "--ik-max-iters",
            "200",
            "--ik-orientation-weight",
            "0",
            "--require-ik-convergence",
            "yes",
            "--summary-out",
            str(summary_out),
        ]
    )

    assert status == 2
    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    assert str(missing_limits) in summary["missing"]
    assert not any("IK check failed" in gap for gap in summary["data_gaps"])


def test_validate_data_main_reads_paper_like_run_config(tmp_path):
    summary_out = tmp_path / "paper-like-validate-summary.json"

    status = validate_data_main(
        [
            "--config",
            str(Path("configs/paper_like.yaml")),
            "--summary-out",
            str(summary_out),
        ]
    )

    assert status == 0
    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    assert summary["data_readiness"] == "ready"
    assert summary["missing"] == []
    assert summary["data_gaps"] == []
    assert summary["full_reproduction_ready"] is True
    assert not any("IK check failed" in gap for gap in summary["data_gaps"])
    sources = summary["reproduction_sources"]
    assert sources["limits"].endswith("limits.yaml")
    assert sources["model"].endswith("T12A-14.xml")
    assert sources["path_index"] is None
    assert sources["dynamics_backend"] == "none"


def test_validate_data_main_reads_paper_batch_like_config_as_14_path_gate(tmp_path):
    summary_out = tmp_path / "paper-batch-like-validate-summary.json"

    status = validate_data_main(
        [
            "--config",
            str(Path("configs/paper_batch_like.yaml")),
            "--summary-out",
            str(summary_out),
        ]
    )

    assert status == 0
    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    assert summary["data_readiness"] == "ready"
    assert summary["missing"] == []
    assert summary["data_gaps"] == []
    assert summary["full_reproduction_ready"] is True
    sources = summary["reproduction_sources"]
    assert sources["limits"].endswith("limits.yaml")
    assert sources["model"].endswith("T12A-14.xml")
    assert sources["path_index"].endswith("path_index.yaml")
    assert sources["dynamics_backend"] == "none"


def test_validate_data_checks_path_index_referenced_csvs(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    (robot_dir / "T12A-14.xml").write_text("<mujoco/>", encoding="utf-8")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)
    _write_path_csv(paths_dir / "present.csv")
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: present\n"
        "    csv: present.csv\n"
        "  - id: missing\n"
        "    csv: missing.csv\n",
        encoding="utf-8",
    )

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits, path_index=path_index)

    assert not report.ok
    assert str(paths_dir / "missing.csv") in report.missing
    assert str(paths_dir / "present.csv") in report.present
    assert str(path_index) in report.present


def test_validate_data_reports_existing_path_index_csv_gaps_even_when_other_csvs_are_missing(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    (robot_dir / "T12A-14.xml").write_text("<mujoco/>", encoding="utf-8")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)
    malformed = paths_dir / "malformed.csv"
    malformed.write_text(
        "s,q1,dq1,d2q1\n"
        "0,0,1,0\n"
        "1,1,1,0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: missing\n"
        "    csv: missing.csv\n"
        "  - id: malformed\n"
        "    csv: malformed.csv\n",
        encoding="utf-8",
    )

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits, path_index=path_index)

    assert str(paths_dir / "missing.csv") in report.missing
    assert any("malformed.csv" in gap and "missing columns" in gap for gap in report.data_gaps)


def test_validate_data_rejects_path_index_with_wrong_expected_path_count(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)
    _write_path_csv(paths_dir / "only.csv")
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: only\n"
        "    csv: only.csv\n",
        encoding="utf-8",
    )

    report = validate_data(
        robot_dir=robot_dir,
        paths_dir=paths_dir,
        limits_path=limits,
        path_index=path_index,
        expected_path_count=14,
    )

    assert not report.ok
    assert any("path_index" in gap and "1 paths" in gap and "expected 14" in gap for gap in report.data_gaps)


def test_validate_data_rejects_invalid_path_index_endpoint_speeds(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)
    _write_path_csv(paths_dir / "present.csv")
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: bad_start\n"
        "    csv: present.csv\n"
        "    zs: -0.1\n",
        encoding="utf-8",
    )

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits, path_index=path_index)

    assert not report.ok
    assert any("path_index" in gap and "zs" in gap and "nonnegative" in gap for gap in report.data_gaps)


def test_validate_data_rejects_conflicting_path_index_endpoint_aliases(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)
    _write_path_csv(paths_dir / "present.csv")
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: conflicting\n"
        "    csv: present.csv\n"
        "    zs: 0.0\n"
        "    z_start: 0.25\n",
        encoding="utf-8",
    )

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits, path_index=path_index)

    assert not report.ok
    assert any("path_index" in gap and "zs" in gap and "z_start" in gap and "conflict" in gap for gap in report.data_gaps)


def test_validate_data_rejects_path_index_endpoint_speed_above_qdot_limits(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)
    path_csv = paths_dir / "fast_start.csv"
    path_csv.write_text(
        "s,q1,dq1,d2q1,d3q1,q2,dq2,d2q2,d3q2,q3,dq3,d2q3,d3q3,q4,dq4,d2q4,d3q4,q5,dq5,d2q5,d3q5,q6,dq6,d2q6,d3q6\n"
        "0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n"
        "1,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: too_fast\n"
        "    csv: fast_start.csv\n"
        "    zs: 1.0\n",
        encoding="utf-8",
    )

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits, path_index=path_index)

    assert not report.ok
    assert any("too_fast" in gap and "zs" in gap and "q_dot" in gap for gap in report.data_gaps)


def test_validate_data_rejects_path_index_terminal_speed_above_qdot_limits(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)
    path_csv = paths_dir / "fast_end.csv"
    path_csv.write_text(
        "s,q1,dq1,d2q1,d3q1,q2,dq2,d2q2,d3q2,q3,dq3,d2q3,d3q3,q4,dq4,d2q4,d3q4,q5,dq5,d2q5,d3q5,q6,dq6,d2q6,d3q6\n"
        "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n"
        "1,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: too_fast_end\n"
        "    csv: fast_end.csv\n"
        "    ze: 1.0\n",
        encoding="utf-8",
    )

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits, path_index=path_index)

    assert not report.ok
    assert any("too_fast_end" in gap and "ze" in gap and "q_dot" in gap for gap in report.data_gaps)


def test_validate_data_rejects_path_index_endpoint_speed_above_entry_z_max(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)
    _write_path_csv(paths_dir / "present.csv")
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: zmax_too_low\n"
        "    csv: present.csv\n"
        "    zs: 0.25\n"
        "    z_max: 0.1\n",
        encoding="utf-8",
    )

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits, path_index=path_index)

    assert not report.ok
    assert any("path_index" in gap and "zs" in gap and "z_max" in gap and "exceed" in gap for gap in report.data_gaps)


def test_validate_data_main_applies_config_default_endpoint_speed_to_path_index(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)
    path_csv = paths_dir / "fast_default.csv"
    path_csv.write_text(
        "s,q1,dq1,d2q1,d3q1,q2,dq2,d2q2,d3q2,q3,dq3,d2q3,d3q3,q4,dq4,d2q4,d3q4,q5,dq5,d2q5,d3q5,q6,dq6,d2q6,d3q6\n"
        "0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n"
        "1,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: default_too_fast\n"
        "    csv: fast_default.csv\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"robot: {robot_dir.as_posix()}\n"
        f"paths: {paths_dir.as_posix()}\n"
        f"limits: {limits.as_posix()}\n"
        f"path_index: {path_index.as_posix()}\n"
        "z_start: 1.0\n",
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"

    status = validate_data_main(["--config", str(config), "--summary-out", str(summary)])

    assert status == 2
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert any("default_too_fast" in gap and "z_start" in gap and "q_dot" in gap for gap in data["data_gaps"])


def test_validate_data_main_accepts_path_index_when_all_referenced_csvs_exist(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)
    _write_path_csv(paths_dir / "present.csv")
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: present\n"
        "    csv: present.csv\n",
        encoding="utf-8",
    )

    status = validate_data_main(["--robot", str(robot_dir), "--paths", str(paths_dir), "--limits", str(limits), "--path-index", str(path_index)])

    assert status == 0


def test_validate_data_rejects_malformed_path_index_csv(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    (robot_dir / "T12A-14.xml").write_text("<mujoco/>", encoding="utf-8")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)
    malformed = paths_dir / "malformed.csv"
    malformed.write_text(
        "s,q1,dq1,d2q1\n"
        "0,0,1,0\n"
        "1,1,1,0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: bad\n"
        "    csv: malformed.csv\n",
        encoding="utf-8",
    )

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits, path_index=path_index)

    assert not report.ok
    assert any("malformed.csv" in gap and "missing columns" in gap for gap in report.data_gaps)


def test_validate_data_rejects_path_index_csv_with_nonfinite_values(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    (robot_dir / "T12A-14.xml").write_text("<mujoco/>", encoding="utf-8")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)
    bad = paths_dir / "nonfinite.csv"
    bad.write_text(
        "s,q1,dq1,d2q1,d3q1,q2,dq2,d2q2,d3q2,q3,dq3,d2q3,d3q3,q4,dq4,d2q4,d3q4,q5,dq5,d2q5,d3q5,q6,dq6,d2q6,d3q6\n"
        "0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0\n"
        "1,nan,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0\n",
        encoding="utf-8",
    )
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: nonfinite\n"
        "    csv: nonfinite.csv\n",
        encoding="utf-8",
    )

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits, path_index=path_index)

    assert not report.ok
    assert any("nonfinite.csv" in gap and "finite" in gap for gap in report.data_gaps)


def test_validate_data_rejects_path_index_csv_with_wrong_dof(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    (robot_dir / "T12A-14.xml").write_text("<mujoco/>", encoding="utf-8")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)
    _write_path_csv(paths_dir / "one_axis.csv", dof=1)
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: one_axis\n"
        "    csv: one_axis.csv\n",
        encoding="utf-8",
    )

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits, path_index=path_index)

    assert not report.ok
    assert any("one_axis.csv" in gap and "DOF" in gap for gap in report.data_gaps)


def test_validate_data_rejects_path_index_csv_outside_mujoco_joint_ranges(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    (robot_dir / "T12A-14.urdf").write_text("<robot/>", encoding="utf-8")
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)
    bad_path = paths_dir / "outside_range.csv"
    _write_path_csv(bad_path)
    text = bad_path.read_text(encoding="utf-8")
    bad_path.write_text(text.replace("1,1,1,1,1,1,1", "1,11,1,1,1,1,1"), encoding="utf-8")
    path_index = paths_dir / "path_index.yaml"
    path_index.write_text(
        "paths:\n"
        "  - id: outside_range\n"
        "    csv: outside_range.csv\n",
        encoding="utf-8",
    )

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits, path_index=path_index)

    assert not report.ok
    assert any("outside_range.csv" in gap and "joint position limits" in gap and "joint1" in gap for gap in report.data_gaps)


def test_validate_data_rejects_q_position_limits_wider_than_mujoco_joint_ranges(tmp_path):
    robot_dir = tmp_path / "robot"
    paths_dir = tmp_path / "paths"
    robot_dir.mkdir()
    paths_dir.mkdir()
    _write_mujoco_model(robot_dir / "T12A-14.xml")
    (paths_dir / "Offline_Traj.txt").write_text(
        "0 0 0 0 0 0\n"
        "1 0 0 0 0 0\n",
        encoding="utf-8",
    )
    limits = robot_dir / "limits.yaml"
    _write_complete_limits(limits)
    text = limits.read_text(encoding="utf-8")
    limits.write_text(
        "q_position:\n"
        "  lower: [-20, -20, -20, -20, -20, -20]\n"
        "  upper: [20, 20, 20, 20, 20, 20]\n"
        + text,
        encoding="utf-8",
    )

    report = validate_data(robot_dir=robot_dir, paths_dir=paths_dir, limits_path=limits)

    assert not report.ok
    assert any("q_position" in gap and "MuJoCo joint range" in gap for gap in report.data_gaps)
