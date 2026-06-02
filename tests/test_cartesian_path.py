import numpy as np
import pytest
import mujoco
import yaml

from dp3_topp.cartesian_path import build_joint_path_from_cartesian, build_joint_path_result_from_cartesian, parse_pose_table
from dp3_topp.cli import default_dyn_dir
from dp3_topp.dynamics_mujoco import MujocoRobotDynamics


def test_parse_pose_table_reads_offline_traj_format():
    poses = parse_pose_table(default_dyn_dir() / "Offline_Traj.txt")

    assert poses.shape[1] == 6
    assert poses.shape[0] >= 10
    assert np.all(np.isfinite(poses[:3]))


def test_parse_pose_table_rejects_nonfinite_values(tmp_path):
    path = tmp_path / "Offline_Traj.txt"
    path.write_text(
        "0 0 0 0 0 0\n"
        "1 nan 0 0 0 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite"):
        parse_pose_table(path)


def test_build_joint_path_from_cartesian_t12a_smoke():
    poses = parse_pose_table(default_dyn_dir() / "Offline_Traj.txt")[:8]
    robot = MujocoRobotDynamics.from_model_path(default_dyn_dir() / "models" / "T12A" / "T12A-14.xml")

    path = build_joint_path_from_cartesian(
        robot=robot,
        pose_table=poses,
        ee_body="link_6",
        ee_site="tcp",
        max_iters=200,
        orientation_weight=0.0,
        require_convergence=True,
    )

    assert path.samples == 8
    assert path.dof == 6
    assert np.all(np.isfinite(path.q))
    assert np.all(np.isfinite(path.q_s))
    assert np.all(np.diff(path.s) > 0.0)
    assert path.s[0] == pytest.approx(0.0)
    assert path.s[-1] == pytest.approx(1.0)


def test_build_joint_path_from_cartesian_requires_at_least_two_poses():
    robot = MujocoRobotDynamics.from_model_path(default_dyn_dir() / "models" / "T12A" / "T12A-14.xml")
    pose = parse_pose_table(default_dyn_dir() / "Offline_Traj.txt")[:1]

    with pytest.raises(ValueError, match="at least two poses"):
        build_joint_path_from_cartesian(
            robot=robot,
            pose_table=pose,
            ee_body="link_6",
            max_iters=1,
        )


def test_build_joint_path_from_cartesian_rejects_zero_length_cartesian_segments():
    robot = MujocoRobotDynamics.from_model_path(default_dyn_dir() / "models" / "T12A" / "T12A-14.xml")
    body_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "link_6")
    data = mujoco.MjData(robot.model)
    data.qpos[:] = 0.0
    data.qpos[robot.qpos_indices] = np.array([0.2, -0.4, 0.3, 0.1, -0.2, 0.0], dtype=np.float64)
    mujoco.mj_forward(robot.model, data)
    pose_table = np.zeros((2, 6), dtype=np.float64)
    pose_table[:, :3] = data.xpos[body_id]

    with pytest.raises(ValueError, match="zero-length Cartesian segment"):
        build_joint_path_from_cartesian(
            robot=robot,
            pose_table=pose_table,
            ee_body="link_6",
            max_iters=80,
            orientation_weight=0.0,
            require_convergence=True,
        )


def test_build_joint_path_from_cartesian_rejects_nonfinite_pose_values():
    robot = MujocoRobotDynamics.from_model_path(default_dyn_dir() / "models" / "T12A" / "T12A-14.xml")
    poses = parse_pose_table(default_dyn_dir() / "Offline_Traj.txt")[:2].copy()
    poses[1, 0] = np.inf

    with pytest.raises(ValueError, match="pose_table.*finite"):
        build_joint_path_from_cartesian(
            robot=robot,
            pose_table=poses,
            ee_body="link_6",
            max_iters=1,
        )


def test_build_joint_path_from_cartesian_can_require_ik_convergence():
    robot = MujocoRobotDynamics.from_model_path(default_dyn_dir() / "models" / "T12A" / "T12A-14.xml")
    unreachable = np.array(
        [
            [100.0, 100.0, 100.0, 0.0, 0.0, 0.0],
            [101.0, 100.0, 100.0, 0.0, 0.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="IK failed to converge"):
        build_joint_path_from_cartesian(
            robot=robot,
            pose_table=unreachable,
            ee_body="link_6",
            max_iters=1,
            require_convergence=True,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"max_iters": 0}, "max_iters"),
        ({"pos_tol": 0.0}, "pos_tol"),
        ({"ori_tol": np.inf}, "ori_tol"),
        ({"damping": 0.0}, "damping"),
        ({"step_scale": np.nan}, "step_scale"),
        ({"orientation_weight": -0.1}, "orientation_weight"),
        ({"tcp_offset": np.array([0.0, np.nan, 0.0])}, "tcp_offset"),
    ),
)
def test_build_joint_path_from_cartesian_rejects_invalid_ik_parameters(kwargs, message):
    robot = MujocoRobotDynamics.from_model_path(default_dyn_dir() / "models" / "T12A" / "T12A-14.xml")
    poses = parse_pose_table(default_dyn_dir() / "Offline_Traj.txt")[:2]

    with pytest.raises(ValueError, match=message):
        build_joint_path_from_cartesian(
            robot=robot,
            pose_table=poses,
            ee_body="link_6",
            **kwargs,
        )


def test_build_joint_path_result_from_cartesian_reports_ik_diagnostics():
    robot = MujocoRobotDynamics.from_model_path(default_dyn_dir() / "models" / "T12A" / "T12A-14.xml")
    unreachable = np.array(
        [
            [100.0, 100.0, 100.0, 0.0, 0.0, 0.0],
            [101.0, 100.0, 100.0, 0.0, 0.0, 0.0],
        ]
    )

    result = build_joint_path_result_from_cartesian(
        robot=robot,
        pose_table=unreachable,
        ee_body="link_6",
        max_iters=1,
        require_convergence=False,
    )

    assert result.path.samples == 2
    assert result.ik.require_convergence is False
    assert result.ik.converged is False
    assert result.ik.total_iterations == 2
    assert np.isfinite(result.ik.max_position_error)
    assert result.ik.max_position_error > 1.0
    assert np.isfinite(result.ik.max_orientation_error)
    assert result.ik.max_residual == max(result.ik.max_position_error, result.ik.max_orientation_error)


def test_build_joint_path_result_reports_final_position_error_when_orientation_is_ignored():
    robot = MujocoRobotDynamics.from_model_path(default_dyn_dir() / "models" / "T12A" / "T12A-14.xml")
    body_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "link_6")
    data = mujoco.MjData(robot.model)
    q_targets = np.array(
        [
            [0.2, -0.4, 0.3, 0.1, -0.2, 0.0],
            [0.25, -0.35, 0.28, 0.1, -0.18, 0.0],
        ],
        dtype=np.float64,
    )
    pose_table = np.zeros((q_targets.shape[0], 6), dtype=np.float64)
    for index, q in enumerate(q_targets):
        data.qpos[:] = 0.0
        data.qpos[robot.qpos_indices] = q
        mujoco.mj_forward(robot.model, data)
        pose_table[index, :3] = data.xpos[body_id]

    result = build_joint_path_result_from_cartesian(
        robot=robot,
        pose_table=pose_table,
        ee_body="link_6",
        max_iters=80,
        pos_tol=1e-8,
        orientation_weight=0.0,
    )

    assert result.ik.converged is True
    assert result.ik.max_position_error <= 1e-8
    assert result.ik.max_residual == pytest.approx(result.ik.max_position_error)


def test_t12a_position_only_ik_does_not_diverge_on_offline_traj_prefix():
    robot = MujocoRobotDynamics.from_model_path(default_dyn_dir() / "models" / "T12A" / "T12A-14.xml")
    poses = parse_pose_table(default_dyn_dir() / "Offline_Traj.txt")[:40]

    result = build_joint_path_result_from_cartesian(
        robot=robot,
        pose_table=poses,
        ee_body="link_6",
        ee_site="tcp",
        max_iters=200,
        pos_tol=1e-4,
        orientation_weight=0.0,
        require_convergence=True,
    )

    assert result.ik.max_position_error <= 1e-4
    assert result.ik.max_residual == pytest.approx(result.ik.max_position_error)


def test_paper_like_t12a_ik_settings_converge_on_offline_traj_prefix():
    config_path = default_dyn_dir().parent / "configs" / "paper_like.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    robot = MujocoRobotDynamics.from_model_path((config_path.parent / config["model"]).resolve())
    poses = parse_pose_table((config_path.parent / config["input"]).resolve())[:40]

    result = build_joint_path_result_from_cartesian(
        robot=robot,
        pose_table=poses,
        ee_body=config["ee_body"],
        ee_site=config.get("ee_site"),
        max_iters=int(config["ik_max_iters"]),
        pos_tol=float(config["ik_pos_tol"]),
        ori_tol=float(config["ik_ori_tol"]),
        damping=float(config["ik_damping"]),
        step_scale=float(config["ik_step_scale"]),
        orientation_weight=float(config["ik_orientation_weight"]),
        tcp_offset=np.array(
            [
                float(config["tcp_offset_x"]),
                float(config["tcp_offset_y"]),
                float(config["tcp_offset_z"]),
            ],
            dtype=np.float64,
        ),
        require_convergence=bool(config["require_ik_convergence"]),
    )

    assert result.ik.converged is True
    assert result.ik.max_position_error <= float(config["ik_pos_tol"])


def test_build_joint_path_result_can_target_mujoco_site():
    robot = MujocoRobotDynamics.from_model_path(default_dyn_dir() / "models" / "T12A" / "T12A-14.xml")
    site_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    data = mujoco.MjData(robot.model)
    q_targets = np.array(
        [
            [0.2, -0.4, 0.3, 0.1, -0.2, 0.0],
            [0.25, -0.35, 0.28, 0.1, -0.18, 0.0],
        ],
        dtype=np.float64,
    )
    pose_table = np.zeros((q_targets.shape[0], 6), dtype=np.float64)
    for index, q in enumerate(q_targets):
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[robot.qpos_indices] = q
        mujoco.mj_forward(robot.model, data)
        pose_table[index, :3] = data.site_xpos[site_id]

    result = build_joint_path_result_from_cartesian(
        robot=robot,
        pose_table=pose_table,
        ee_body="link_6",
        ee_site="tcp",
        max_iters=80,
        pos_tol=1e-8,
        orientation_weight=0.0,
    )

    assert result.ik.converged is True
    assert result.ik.max_position_error <= 1e-8
    assert result.path.dof == 6


def test_build_joint_path_result_rejects_missing_mujoco_site():
    robot = MujocoRobotDynamics.from_model_path(default_dyn_dir() / "models" / "T12A" / "T12A-14.xml")
    poses = parse_pose_table(default_dyn_dir() / "Offline_Traj.txt")[:2]

    with pytest.raises(ValueError, match="Site not found"):
        build_joint_path_result_from_cartesian(
            robot=robot,
            pose_table=poses,
            ee_body="link_6",
            ee_site="missing_site",
        )
