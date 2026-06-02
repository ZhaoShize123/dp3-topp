from pathlib import Path

import mujoco
import numpy as np
import pytest

from dp3_topp.dynamics_mujoco import MujocoRobotDynamics


def test_t12a_mujoco_model_loads_and_inverse_dynamics_is_finite():
    model_path = Path("dyn - 副本/models/T12A/T12A-14.xml")
    robot = MujocoRobotDynamics.from_model_path(model_path)

    assert robot.dof == 6
    assert robot.joint_names == [f"joint_{i}" for i in range(1, 7)]

    q = np.zeros(robot.dof)
    qd = np.linspace(0.0, 0.5, robot.dof)
    qdd = np.linspace(0.1, 0.6, robot.dof)
    tau = robot.inverse_dynamics(q, qd, qdd)

    assert tau.shape == (6,)
    assert np.all(np.isfinite(tau))


def test_mujoco_model_uses_actuated_hinge_joints_for_robot_dof(tmp_path):
    model_path = tmp_path / "actuated_plus_passive.xml"
    model_path.write_text(
        '<mujoco model="actuated_plus_passive">'
        '  <compiler autolimits="true"/>'
        '  <option gravity="0 0 0"/>'
        '  <worldbody>'
        '    <body name="actuated_body">'
        '      <joint name="joint_actuated" type="hinge" axis="0 0 1" range="-1 1"/>'
        '      <geom type="capsule" fromto="0 0 0 1 0 0" size="0.01" density="1000"/>'
        '      <body name="passive_body" pos="1 0 0">'
        '        <joint name="joint_passive" type="hinge" axis="0 1 0" range="-2 2"/>'
        '        <geom type="capsule" fromto="0 0 0 0 1 0" size="0.01" density="1000"/>'
        '      </body>'
        '    </body>'
        '  </worldbody>'
        '  <actuator>'
        '    <position name="joint_actuated_position" joint="joint_actuated"/>'
        '  </actuator>'
        '</mujoco>',
        encoding="utf-8",
    )

    robot = MujocoRobotDynamics.from_model_path(model_path)

    assert robot.dof == 1
    assert robot.joint_names == ["joint_actuated"]
    np.testing.assert_array_equal(robot.dof_indices, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(robot.qpos_indices, np.array([0], dtype=np.int32))


def test_mujoco_model_requires_finite_joint_ranges_for_selected_robot_joints(tmp_path):
    model_path = tmp_path / "missing_range.xml"
    model_path.write_text(
        '<mujoco model="missing_range">'
        '  <option gravity="0 0 0"/>'
        '  <worldbody>'
        '    <body name="body">'
        '      <joint name="joint1" type="hinge" axis="0 0 1"/>'
        '      <geom type="capsule" fromto="0 0 0 1 0 0" size="0.01" density="1000"/>'
        '    </body>'
        '  </worldbody>'
        '</mujoco>',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite joint range.*joint1"):
        MujocoRobotDynamics.from_model_path(model_path)


def test_mujoco_model_requires_at_least_one_selected_hinge_joint(tmp_path):
    model_path = tmp_path / "no_hinge.xml"
    model_path.write_text(
        '<mujoco model="no_hinge">'
        '  <compiler autolimits="true"/>'
        '  <option gravity="0 0 0"/>'
        '  <worldbody>'
        '    <body name="body">'
        '      <joint name="slide1" type="slide" axis="1 0 0" range="-1 1"/>'
        '      <geom type="box" size="0.1 0.1 0.1" density="1000"/>'
        '    </body>'
        '  </worldbody>'
        '</mujoco>',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least one selected hinge joint"):
        MujocoRobotDynamics.from_model_path(model_path)


@pytest.mark.parametrize(
    ("q", "qd", "qdd"),
    (
        (np.array([np.nan]), np.array([0.0]), np.array([0.0])),
        (np.array([0.0]), np.array([np.inf]), np.array([0.0])),
        (np.array([0.0]), np.array([0.0]), np.array([np.nan])),
    ),
)
def test_inverse_dynamics_rejects_nonfinite_state(monkeypatch, q, qd, qdd):
    robot = MujocoRobotDynamics(
        model=None,
        joint_names=["joint1"],
        dof_indices=np.array([0], dtype=np.int32),
        qpos_indices=np.array([0], dtype=np.int32),
        lower=np.array([-10.0]),
        upper=np.array([10.0]),
    )

    with pytest.raises(ValueError, match="finite"):
        robot.inverse_dynamics(q, qd, qdd)


def test_inverse_dynamics_excludes_mujoco_passive_joint_damping():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco model="damped">'
        '  <option gravity="0 0 0"/>'
        '  <worldbody>'
        '    <body name="body">'
        '      <joint name="joint1" type="hinge" axis="0 0 1" damping="8"/>'
        '      <geom type="capsule" fromto="0 0 0 1 0 0" size="0.01" density="1000"/>'
        '    </body>'
        '  </worldbody>'
        '</mujoco>'
    )
    robot = MujocoRobotDynamics(
        model=model,
        joint_names=["joint1"],
        dof_indices=np.array([0], dtype=np.int32),
        qpos_indices=np.array([0], dtype=np.int32),
        lower=np.array([-10.0]),
        upper=np.array([10.0]),
    )

    tau = robot.inverse_dynamics(q=np.array([0.0]), qd=np.array([2.0]), qdd=np.array([0.0]))

    np.testing.assert_allclose(tau, np.array([0.0]), atol=1e-12)


def test_torque_rate_uses_centered_state_difference(monkeypatch):
    robot = MujocoRobotDynamics(
        model=None,
        joint_names=["joint1"],
        dof_indices=np.array([0], dtype=np.int32),
        qpos_indices=np.array([0], dtype=np.int32),
        lower=np.array([-10.0]),
        upper=np.array([10.0]),
    )

    def fake_inverse_dynamics(self, q, qd, qdd):
        return np.array([float(q[0] ** 2 + 2.0 * qd[0] + 3.0 * qdd[0])])

    monkeypatch.setattr(MujocoRobotDynamics, "inverse_dynamics", fake_inverse_dynamics)
    q = np.array([0.7])
    qd = np.array([0.5])
    qdd = np.array([0.2])
    qddd = np.array([0.1])
    dt = 0.1

    actual = robot.torque_rate_finite_difference(q, qd, qdd, qddd, dt=dt)

    half = 0.5 * dt
    q_plus = q + qd * half + 0.5 * qdd * half**2 + (qddd * half**3) / 6.0
    q_minus = q - qd * half + 0.5 * qdd * half**2 - (qddd * half**3) / 6.0
    qd_plus = qd + qdd * half + 0.5 * qddd * half**2
    qd_minus = qd - qdd * half + 0.5 * qddd * half**2
    qdd_plus = qdd + qddd * half
    qdd_minus = qdd - qddd * half
    expected = (fake_inverse_dynamics(robot, q_plus, qd_plus, qdd_plus) - fake_inverse_dynamics(robot, q_minus, qd_minus, qdd_minus)) / dt

    np.testing.assert_allclose(actual, expected)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(actual, np.array([2.0 * q[0] * qd[0] + 2.0 * qdd[0] + 3.0 * qddd[0]]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"q": np.array([np.nan])}, "q.*finite"),
        ({"qd": np.array([np.inf])}, "qd.*finite"),
        ({"qdd": np.array([np.nan])}, "qdd.*finite"),
        ({"qddd": np.array([np.inf])}, "qddd.*finite"),
        ({"dt": np.nan}, "dt"),
    ),
)
def test_torque_rate_rejects_nonfinite_state_and_dt(monkeypatch, kwargs, message):
    robot = MujocoRobotDynamics(
        model=None,
        joint_names=["joint1"],
        dof_indices=np.array([0], dtype=np.int32),
        qpos_indices=np.array([0], dtype=np.int32),
        lower=np.array([-10.0]),
        upper=np.array([10.0]),
    )

    def fake_inverse_dynamics(self, q, qd, qdd):
        return np.zeros(1)

    monkeypatch.setattr(MujocoRobotDynamics, "inverse_dynamics", fake_inverse_dynamics)
    args = {
        "q": np.array([0.0]),
        "qd": np.array([0.0]),
        "qdd": np.array([0.0]),
        "qddd": np.array([0.0]),
        "dt": 1e-5,
    }
    args.update(kwargs)

    with pytest.raises(ValueError, match=message):
        robot.torque_rate_finite_difference(**args)


def test_joint_position_range_check_rejects_nonfinite_positions():
    robot = MujocoRobotDynamics(
        model=None,
        joint_names=["joint1"],
        dof_indices=np.array([0], dtype=np.int32),
        qpos_indices=np.array([0], dtype=np.int32),
        lower=np.array([-1.0]),
        upper=np.array([1.0]),
    )

    with pytest.raises(ValueError, match="finite"):
        robot.assert_joint_positions_in_range(np.array([[np.nan]]))
