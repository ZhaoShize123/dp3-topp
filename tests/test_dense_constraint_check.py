import numpy as np

from dp3_topp.cli import _constraint_check_result
from dp3_topp.constraints import ConstraintLimits, audit_constraints
from dp3_topp.interpolation import C3QuadraticSpeed
from dp3_topp.optimizer import TrajectoryResult
from dp3_topp.path_data import PathData


def test_dense_constraint_check_resamples_stored_segment_profile_exactly():
    profile = C3QuadraticSpeed.from_interval(0.0, 1.0, z0=1.0, z1=4.0, z_s1=0.5)
    endpoints = profile.evaluate(np.array([0.0, 1.0]))
    path = PathData(
        s=np.array([0.0, 1.0]),
        q=np.array([[0.0], [1.0]]),
        q_s=np.array([[1.0], [1.0]]),
        q_ss=np.zeros((2, 1)),
        q_sss=np.zeros((2, 1)),
    )
    limits = ConstraintLimits(
        q_dot_abs=np.array([100.0]),
        q_ddot_abs=np.array([100.0]),
        q_jerk_abs=np.array([100.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([100.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )
    result = TrajectoryResult(
        feasible=True,
        t=np.array([0.0, profile.duration()]),
        s=endpoints.s,
        z=endpoints.z,
        z_s=endpoints.z_s,
        z_ss=endpoints.z_ss,
        total_time=profile.duration(),
        audit=audit_constraints(
            limits=limits,
            q_dot=np.zeros((1, 1)),
            q_ddot=np.zeros((1, 1)),
            q_jerk=np.zeros((1, 1)),
            tau=np.zeros((1, 1)),
            tau_rate=np.zeros((1, 1)),
            mechanical_power=np.zeros(1),
        ),
        grid_s=np.array([0.0, 1.0]),
        grid_z=np.array([[1.0], [4.0]]),
        policy=np.array([[0], [-1]], dtype=np.int32),
        segment_profiles=(profile,),
    )

    checked, _, _, source = _constraint_check_result(
        path=path,
        result=result,
        limits=limits,
        robot=None,
        points=5,
    )

    expected = profile.evaluate(checked.s)
    linearly_interpolated_z_s = np.interp(checked.s, result.s, result.z_s)
    assert source == "dense_segment_profiles"
    assert np.allclose(checked.z, expected.z)
    assert np.allclose(checked.z_s, expected.z_s)
    assert np.allclose(checked.z_ss, expected.z_ss)
    assert not np.allclose(checked.z_s[1:-1], linearly_interpolated_z_s[1:-1])


def test_dense_constraint_check_fallback_preserves_torque_rate_difference_step():
    class RecordingRobot:
        dof = 1

        def __init__(self):
            self.dt_values = []

        def inverse_dynamics(self, q, qd, qdd):
            return np.zeros(1)

        def torque_rate_finite_difference(self, q, qd, qdd, qddd, dt=1e-5):
            self.dt_values.append(float(dt))
            return np.zeros(1)

    path = PathData(
        s=np.array([0.0, 1.0]),
        q=np.array([[0.0], [1.0]]),
        q_s=np.array([[1.0], [1.0]]),
        q_ss=np.zeros((2, 1)),
        q_sss=np.zeros((2, 1)),
    )
    limits = ConstraintLimits(
        q_dot_abs=np.array([100.0]),
        q_ddot_abs=np.array([100.0]),
        q_jerk_abs=np.array([100.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([100.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )
    result = TrajectoryResult(
        feasible=True,
        t=np.array([0.0, 1.0]),
        s=np.array([0.0, 1.0]),
        z=np.array([1.0, 1.0]),
        z_s=np.array([0.0, 0.0]),
        z_ss=np.array([0.0, 0.0]),
        total_time=1.0,
        audit=audit_constraints(
            limits=limits,
            q_dot=np.zeros((1, 1)),
            q_ddot=np.zeros((1, 1)),
            q_jerk=np.zeros((1, 1)),
            tau=np.zeros((1, 1)),
            tau_rate=np.zeros((1, 1)),
            mechanical_power=np.zeros(1),
        ),
        grid_s=np.array([0.0, 1.0]),
        grid_z=np.array([[1.0], [1.0]]),
        policy=np.array([[0], [-1]], dtype=np.int32),
        tau_rate_dt=3e-4,
    )
    robot = RecordingRobot()

    _, _, _, source = _constraint_check_result(
        path=path,
        result=result,
        limits=limits,
        robot=robot,
        points=3,
    )

    assert source == "dense_s_grid"
    assert robot.dt_values == [3e-4, 3e-4, 3e-4]


def test_constraint_check_preserves_empty_result_audit_violations():
    path = PathData(
        s=np.array([0.0, 1.0]),
        q=np.array([[0.0], [2.0]]),
        q_s=np.array([[1.0], [1.0]]),
        q_ss=np.zeros((2, 1)),
        q_sss=np.zeros((2, 1)),
    )
    limits = ConstraintLimits(
        q_position_lower=np.array([-0.5]),
        q_position_upper=np.array([0.5]),
        q_dot_abs=np.array([100.0]),
        q_ddot_abs=np.array([100.0]),
        q_jerk_abs=np.array([100.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([100.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )
    stored_audit = audit_constraints(
        limits=limits,
        q_position=path.q,
        q_dot=np.zeros((2, 1)),
        q_ddot=np.zeros((2, 1)),
        q_jerk=np.zeros((2, 1)),
        tau=np.zeros((2, 1)),
        tau_rate=np.zeros((2, 1)),
        mechanical_power=np.zeros(2),
    )
    result = TrajectoryResult(
        feasible=False,
        t=np.array([], dtype=np.float64),
        s=np.array([], dtype=np.float64),
        z=np.array([], dtype=np.float64),
        z_s=np.array([], dtype=np.float64),
        z_ss=np.array([], dtype=np.float64),
        total_time=np.inf,
        audit=stored_audit,
        grid_s=np.array([0.0, 1.0]),
        grid_z=np.array([[0.0], [0.0]]),
        policy=np.array([[-1], [-1]], dtype=np.int32),
    )

    checked, _, audit, source = _constraint_check_result(
        path=path,
        result=result,
        limits=limits,
        robot=None,
        points=5,
    )

    assert checked.s.size == 0
    assert source == "trajectory_samples"
    assert audit.max_utilization["q_position"] > 1.0
    assert any(violation.quantity == "q_position" for violation in audit.violations)
