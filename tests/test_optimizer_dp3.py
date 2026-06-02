import numpy as np
import pytest

from dp3_topp.constraints import ConstraintAudit, ConstraintLimits
from dp3_topp.interpolation import C2LinearZ
from dp3_topp.optimizer import (
    DP3Config,
    TrajectoryResult,
    _make_segment_profile,
    evaluate_trajectory_quantities,
    optimize_dp2,
    optimize_dp3,
    resample_trajectory_by_time,
)
from dp3_topp.path_data import PathData


def test_optimize_dp3_finds_feasible_zero_to_zero_profile_for_linear_path():
    s = np.linspace(0.0, 1.0, 21)
    q = s[:, None]
    path = PathData(
        s=s,
        q=q,
        q_s=np.ones_like(q),
        q_ss=np.zeros_like(q),
        q_sss=np.zeros_like(q),
    )
    limits = ConstraintLimits(
        q_dot_abs=np.array([1.0]),
        q_ddot_abs=np.array([5.0]),
        q_jerk_abs=np.array([50.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([100.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )

    result = optimize_dp3(path=path, limits=limits, config=DP3Config(ns=6, nz=8, nch=7))

    assert result.feasible
    assert result.s[0] == 0.0
    assert result.s[-1] == 1.0
    assert result.z[0] == 0.0
    assert result.z[-1] == 0.0
    assert result.total_time > 0.0
    assert result.t[0] == 0.0
    assert result.t[-1] == pytest.approx(result.total_time)
    assert np.all(np.diff(result.t) > 0.0)
    assert result.audit.ok
    assert result.segment_kinds[:2] == ["C2", "C4"]


def test_optimize_dp3_audits_fixed_path_joint_position_limits():
    s = np.linspace(0.0, 1.0, 21)
    q = (2.0 * s)[:, None]
    path = PathData(
        s=s,
        q=q,
        q_s=np.full_like(q, 2.0),
        q_ss=np.zeros_like(q),
        q_sss=np.zeros_like(q),
    )
    limits = ConstraintLimits(
        q_dot_abs=np.array([10.0]),
        q_ddot_abs=np.array([50.0]),
        q_jerk_abs=np.array([500.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([100.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
        q_position_lower=np.array([-0.1]),
        q_position_upper=np.array([1.0]),
    )

    result = optimize_dp3(path=path, limits=limits, config=DP3Config(ns=6, nz=8, nch=7))

    assert not result.feasible
    assert any(violation.quantity == "q_position" for violation in result.audit.violations)


def test_trajectory_quantities_report_interval_centered_joint_position_utilization():
    s = np.array([0.0, 0.5, 1.0])
    q = np.array([[0.5], [2.0], [3.5]])
    path = PathData(
        s=s,
        q=q,
        q_s=np.ones_like(q),
        q_ss=np.zeros_like(q),
        q_sss=np.zeros_like(q),
    )
    limits = ConstraintLimits(
        q_position_lower=np.array([1.0]),
        q_position_upper=np.array([3.0]),
        q_dot_abs=np.array([5.0]),
        q_ddot_abs=np.array([5.0]),
        q_jerk_abs=np.array([5.0]),
        tau_abs=np.array([10.0]),
        tau_rate_abs=np.array([20.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )
    result = TrajectoryResult(
        feasible=False,
        t=s.copy(),
        s=s.copy(),
        z=np.zeros(3),
        z_s=np.zeros(3),
        z_ss=np.zeros(3),
        total_time=1.0,
        audit=ConstraintAudit(ok=False, max_utilization={}, violations=[]),
        grid_s=s.copy(),
        grid_z=np.zeros((3, 1)),
        policy=np.zeros((3, 1), dtype=np.int32),
    )

    quantities = evaluate_trajectory_quantities(path=path, result=result, limits=limits)

    np.testing.assert_allclose(quantities.q_position_utilization, np.array([[-1.5], [0.0], [1.5]]))


def test_optimize_dp3_uses_paper_c2_profile_on_terminal_segment_with_nonzero_end_speed():
    s = np.linspace(0.0, 1.0, 21)
    q = s[:, None]
    path = PathData(
        s=s,
        q=q,
        q_s=np.ones_like(q),
        q_ss=np.zeros_like(q),
        q_sss=np.zeros_like(q),
    )
    limits = ConstraintLimits(
        q_dot_abs=np.array([2.0]),
        q_ddot_abs=np.array([20.0]),
        q_jerk_abs=np.array([500.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([100.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )

    result = optimize_dp3(path=path, limits=limits, config=DP3Config(ns=6, nz=8, nch=7, z_end=0.25))

    assert result.feasible
    assert result.z[-1] == pytest.approx(0.25)
    assert result.segment_kinds[-1] == "C2"
    assert isinstance(result.segment_profiles[-1], C2LinearZ)


def test_dp3_internal_segment_profile_rejects_zero_speed_instead_of_falling_back_to_c2():
    assert (
        _make_segment_profile(
            s0=0.25,
            s1=0.5,
            z0=0.0,
            z1=0.25,
            next_z_s=0.0,
            use_dp3_profiles=True,
            force_kind=None,
        )
        is None
    )
    assert (
        _make_segment_profile(
            s0=0.25,
            s1=0.5,
            z0=0.25,
            z1=0.0,
            next_z_s=0.0,
            use_dp3_profiles=True,
            force_kind=None,
        )
        is None
    )
    assert isinstance(
        _make_segment_profile(
            s0=0.0,
            s1=0.25,
            z0=0.0,
            z1=0.25,
            next_z_s=0.0,
            use_dp3_profiles=True,
            force_kind="C2",
        ),
        C2LinearZ,
    )


@pytest.mark.parametrize(
    ("config", "message"),
    (
        (DP3Config(ns=6, nz=8, nch=7, z_start=-0.1), "z_start"),
        (DP3Config(ns=6, nz=8, nch=7, z_end=-0.1), "z_end"),
        (DP3Config(ns=6, nz=8, nch=7, z_max=0.0), "z_max"),
        (DP3Config(ns=6, nz=8, nch=7, z_max=np.inf), "z_max"),
    ),
)
def test_optimize_dp3_rejects_invalid_z_boundary_config(config, message):
    s = np.linspace(0.0, 1.0, 21)
    q = s[:, None]
    path = PathData(
        s=s,
        q=q,
        q_s=np.ones_like(q),
        q_ss=np.zeros_like(q),
        q_sss=np.zeros_like(q),
    )
    limits = ConstraintLimits(
        q_dot_abs=np.array([1.0]),
        q_ddot_abs=np.array([5.0]),
        q_jerk_abs=np.array([50.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([100.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )

    with pytest.raises(ValueError, match=message):
        optimize_dp3(path=path, limits=limits, config=config)


@pytest.mark.parametrize(
    "config",
    (
        DP3Config(ns=6, nz=8, nch=7, k1=np.nan, k2=0.0),
        DP3Config(ns=6, nz=8, nch=7, k1=np.inf, k2=0.0),
        DP3Config(ns=6, nz=8, nch=7, k1=1.0, k2=np.nan),
        DP3Config(ns=6, nz=8, nch=7, k1=1.0, k2=np.inf),
    ),
)
def test_optimize_dp3_rejects_nonfinite_objective_weights(config):
    s = np.linspace(0.0, 1.0, 21)
    q = s[:, None]
    path = PathData(
        s=s,
        q=q,
        q_s=np.ones_like(q),
        q_ss=np.zeros_like(q),
        q_sss=np.zeros_like(q),
    )
    limits = ConstraintLimits(
        q_dot_abs=np.array([1.0]),
        q_ddot_abs=np.array([5.0]),
        q_jerk_abs=np.array([50.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([100.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )

    with pytest.raises(ValueError, match="finite nonnegative k1/k2"):
        optimize_dp3(path=path, limits=limits, config=config)


@pytest.mark.parametrize(
    ("config", "message"),
    (
        (DP3Config(ns=6.5, nz=8, nch=7), "ns"),
        (DP3Config(ns=np.nan, nz=8, nch=7), "ns"),
        (DP3Config(ns=6, nz=8.5, nch=7), "nz"),
        (DP3Config(ns=6, nz=np.inf, nch=7), "nz"),
        (DP3Config(ns=6, nz=8, nch=7.5), "nch"),
        (DP3Config(ns=6, nz=8, nch=np.nan), "nch"),
    ),
)
def test_optimize_dp3_rejects_noninteger_or_nonfinite_grid_counts(config, message):
    s = np.linspace(0.0, 1.0, 21)
    q = s[:, None]
    path = PathData(
        s=s,
        q=q,
        q_s=np.ones_like(q),
        q_ss=np.zeros_like(q),
        q_sss=np.zeros_like(q),
    )
    limits = ConstraintLimits(
        q_dot_abs=np.array([1.0]),
        q_ddot_abs=np.array([5.0]),
        q_jerk_abs=np.array([50.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([100.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )

    with pytest.raises(ValueError, match=message):
        optimize_dp3(path=path, limits=limits, config=config)


def test_optimize_dp3_requires_enough_grid_points_for_paper_start_profiles():
    s = np.linspace(0.0, 1.0, 21)
    q = s[:, None]
    path = PathData(
        s=s,
        q=q,
        q_s=np.ones_like(q),
        q_ss=np.zeros_like(q),
        q_sss=np.zeros_like(q),
    )
    limits = ConstraintLimits(
        q_dot_abs=np.array([1.0]),
        q_ddot_abs=np.array([5.0]),
        q_jerk_abs=np.array([50.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([100.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )

    with pytest.raises(ValueError, match="DP3.*ns >= 4"):
        optimize_dp3(path=path, limits=limits, config=DP3Config(ns=3, nz=8, nch=7))


@pytest.mark.parametrize(
    ("config", "message"),
    (
        (DP3Config(ns=6, nz=8, nch=7, z_start=1.0, z_max=0.5), "z_start.*z_max"),
        (DP3Config(ns=6, nz=8, nch=7, z_end=1.0, z_max=0.5), "z_end.*z_max"),
    ),
)
def test_optimize_dp3_rejects_boundary_speed_above_configured_z_max(config, message):
    s = np.linspace(0.0, 1.0, 21)
    q = s[:, None]
    path = PathData(
        s=s,
        q=q,
        q_s=np.ones_like(q),
        q_ss=np.zeros_like(q),
        q_sss=np.zeros_like(q),
    )
    limits = ConstraintLimits(
        q_dot_abs=np.array([2.0]),
        q_ddot_abs=np.array([5.0]),
        q_jerk_abs=np.array([50.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([100.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )

    with pytest.raises(ValueError, match=message):
        optimize_dp3(path=path, limits=limits, config=config)


@pytest.mark.parametrize("optimizer", (optimize_dp3, optimize_dp2))
@pytest.mark.parametrize(
    ("config", "message"),
    (
        (DP3Config(ns=6, nz=8, nch=7, z_start=1.0), "z_start.*q_dot.*s=0"),
        (DP3Config(ns=6, nz=8, nch=7, z_end=1.0), "z_end.*q_dot.*s=1"),
    ),
)
def test_optimization_rejects_boundary_speed_above_q_dot_limits(optimizer, config, message):
    s = np.linspace(0.0, 1.0, 21)
    q = (2.0 * s)[:, None]
    path = PathData(
        s=s,
        q=q,
        q_s=np.full_like(q, 2.0),
        q_ss=np.zeros_like(q),
        q_sss=np.zeros_like(q),
    )
    limits = ConstraintLimits(
        q_dot_abs=np.array([1.0]),
        q_ddot_abs=np.array([50.0]),
        q_jerk_abs=np.array([500.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([100.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )

    with pytest.raises(ValueError, match=message):
        optimizer(path=path, limits=limits, config=config)


def test_optimize_dp3_z_ceiling_uses_velocity_lower_bound_for_negative_path_derivative():
    s = np.linspace(0.0, 1.0, 21)
    q = (1.0 - s)[:, None]
    path = PathData(
        s=s,
        q=q,
        q_s=-np.ones_like(q),
        q_ss=np.zeros_like(q),
        q_sss=np.zeros_like(q),
    )
    limits = ConstraintLimits(
        q_dot_abs=np.array([2.0]),
        q_ddot_abs=np.array([50.0]),
        q_jerk_abs=np.array([500.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([100.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
        q_dot_lower=np.array([-0.5]),
        q_dot_upper=np.array([2.0]),
    )

    result = optimize_dp3(path=path, limits=limits, config=DP3Config(ns=4, nz=5, nch=5))

    assert float(np.max(result.grid_z)) <= 0.25 + 1e-12


def test_trajectory_quantities_include_limits_yaml_friction_terms():
    path = PathData(
        s=np.array([0.0, 1.0]),
        q=np.array([[0.0], [1.0]]),
        q_s=np.array([[1.0], [1.0]]),
        q_ss=np.array([[0.0], [0.0]]),
        q_sss=np.array([[0.0], [0.0]]),
    )
    limits = ConstraintLimits(
        q_dot_abs=np.array([10.0]),
        q_ddot_abs=np.array([10.0]),
        q_jerk_abs=np.array([10.0]),
        tau_abs=np.array([10.0]),
        tau_rate_abs=np.array([10.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
        friction_coulomb=np.array([1.0]),
        friction_viscous=np.array([0.5]),
    )
    result = TrajectoryResult(
        feasible=True,
        t=np.array([0.25]),
        s=np.array([0.5]),
        z=np.array([4.0]),
        z_s=np.array([4.0]),
        z_ss=np.array([0.0]),
        total_time=1.0,
        audit=ConstraintAudit(ok=True, max_utilization={}, violations=[]),
        grid_s=np.array([0.0, 1.0]),
        grid_z=np.array([[0.0], [0.0]]),
        policy=np.array([[0], [0]]),
    )

    quantities = evaluate_trajectory_quantities(path=path, result=result, limits=limits)

    np.testing.assert_allclose(quantities.q, np.array([[0.5]]))
    np.testing.assert_allclose(quantities.q_dot, np.array([[2.0]]))
    np.testing.assert_allclose(quantities.q_ddot, np.array([[2.0]]))
    np.testing.assert_allclose(quantities.tau, np.array([[2.0]]))
    np.testing.assert_allclose(quantities.tau_rate, np.array([[1.0]]))
    np.testing.assert_allclose(quantities.mechanical_power, np.array([4.0]))
    np.testing.assert_allclose(quantities.q_ddot_utilization, np.array([[0.2]]))
    np.testing.assert_allclose(quantities.tau_rate_utilization, np.array([[0.1]]))


def test_trajectory_quantities_include_drive_power_when_motor_model_is_available():
    path = PathData(
        s=np.array([0.0, 1.0]),
        q=np.array([[0.0], [1.0]]),
        q_s=np.array([[1.0], [1.0]]),
        q_ss=np.array([[0.0], [0.0]]),
        q_sss=np.array([[0.0], [0.0]]),
    )
    limits = ConstraintLimits(
        q_dot_abs=np.array([10.0]),
        q_ddot_abs=np.array([10.0]),
        q_jerk_abs=np.array([10.0]),
        tau_abs=np.array([10.0]),
        tau_rate_abs=np.array([10.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
        friction_coulomb=np.array([1.0]),
        motor_gear_ratio=np.array([2.0]),
        motor_torque_constant=np.array([0.5]),
        motor_stator_resistance=np.array([1.0]),
    )
    result = TrajectoryResult(
        feasible=True,
        t=np.array([0.5]),
        s=np.array([0.5]),
        z=np.array([4.0]),
        z_s=np.array([0.0]),
        z_ss=np.array([0.0]),
        total_time=1.0,
        audit=ConstraintAudit(ok=True, max_utilization={}, violations=[]),
        grid_s=np.array([0.0, 1.0]),
        grid_z=np.array([[0.0], [0.0]]),
        policy=np.array([[0], [0]]),
    )

    quantities = evaluate_trajectory_quantities(path=path, result=result, limits=limits)

    np.testing.assert_allclose(quantities.tau, np.array([[1.0]]))
    np.testing.assert_allclose(quantities.mechanical_power, np.array([2.0]))
    np.testing.assert_allclose(quantities.drive_power, np.array([3.0]))


def test_trajectory_quantities_use_result_torque_rate_difference_step():
    class RecordingRobot:
        dof = 1

        def __init__(self):
            self.dt_values = []

        def inverse_dynamics(self, q, qd, qdd):
            return np.zeros(1)

        def torque_rate_finite_difference(self, q, qd, qdd, qddd, dt=1e-5):
            self.dt_values.append(float(dt))
            return np.array([float(dt)])

    path = PathData(
        s=np.array([0.0, 1.0]),
        q=np.array([[0.0], [1.0]]),
        q_s=np.array([[1.0], [1.0]]),
        q_ss=np.array([[0.0], [0.0]]),
        q_sss=np.array([[0.0], [0.0]]),
    )
    limits = ConstraintLimits(
        q_dot_abs=np.array([10.0]),
        q_ddot_abs=np.array([10.0]),
        q_jerk_abs=np.array([10.0]),
        tau_abs=np.array([10.0]),
        tau_rate_abs=np.array([10.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )
    result = TrajectoryResult(
        feasible=True,
        t=np.array([0.0]),
        s=np.array([0.5]),
        z=np.array([1.0]),
        z_s=np.array([0.0]),
        z_ss=np.array([0.0]),
        total_time=1.0,
        audit=ConstraintAudit(ok=True, max_utilization={}, violations=[]),
        grid_s=np.array([0.0, 1.0]),
        grid_z=np.array([[0.0], [0.0]]),
        policy=np.array([[0], [0]]),
        tau_rate_dt=2e-4,
    )
    robot = RecordingRobot()

    quantities = evaluate_trajectory_quantities(path=path, result=result, limits=limits, robot=robot)

    assert robot.dt_values == [2e-4]
    np.testing.assert_allclose(quantities.tau_rate, np.array([[2e-4]]))


def test_trajectory_quantity_evaluation_rejects_path_samples_outside_domain():
    path = PathData(
        s=np.array([0.0, 1.0]),
        q=np.array([[0.0], [1.0]]),
        q_s=np.array([[1.0], [1.0]]),
        q_ss=np.array([[0.0], [0.0]]),
        q_sss=np.array([[0.0], [0.0]]),
    )
    limits = ConstraintLimits(
        q_dot_abs=np.array([10.0]),
        q_ddot_abs=np.array([10.0]),
        q_jerk_abs=np.array([10.0]),
        tau_abs=np.array([10.0]),
        tau_rate_abs=np.array([10.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )
    result = TrajectoryResult(
        feasible=True,
        t=np.array([0.0, 1.0]),
        s=np.array([-0.1, 1.0]),
        z=np.array([1.0, 1.0]),
        z_s=np.array([0.0, 0.0]),
        z_ss=np.array([0.0, 0.0]),
        total_time=1.0,
        audit=ConstraintAudit(ok=True, max_utilization={}, violations=[]),
        grid_s=np.array([0.0, 1.0]),
        grid_z=np.array([[0.0], [0.0]]),
        policy=np.array([[0], [0]]),
    )

    with pytest.raises(ValueError, match="path domain"):
        evaluate_trajectory_quantities(path=path, result=result, limits=limits)


def test_resample_trajectory_by_time_inverts_stored_segment_profile():
    profile = C2LinearZ.from_interval(0.0, 1.0, z0=4.0, z1=4.0)
    result = TrajectoryResult(
        feasible=True,
        t=np.array([0.0, profile.duration()]),
        s=np.array([0.0, 1.0]),
        z=np.array([4.0, 4.0]),
        z_s=np.array([0.0, 0.0]),
        z_ss=np.array([0.0, 0.0]),
        total_time=profile.duration(),
        audit=ConstraintAudit(ok=True, max_utilization={}, violations=[]),
        grid_s=np.array([0.0, 1.0]),
        grid_z=np.array([[4.0], [4.0]]),
        policy=np.array([[0], [-1]], dtype=np.int32),
        segment_profiles=(profile,),
    )

    sampled = resample_trajectory_by_time(result, np.array([0.0, 0.25, 0.5]))

    np.testing.assert_allclose(sampled.t, np.array([0.0, 0.25, 0.5]))
    np.testing.assert_allclose(sampled.s, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(sampled.z, np.array([4.0, 4.0, 4.0]))
    np.testing.assert_allclose(sampled.z_s, np.zeros(3))
    np.testing.assert_allclose(sampled.z_ss, np.zeros(3))


def test_dp3_config_k1_scales_objective_without_changing_trajectory_time():
    s = np.linspace(0.0, 1.0, 21)
    q = s[:, None]
    path = PathData(
        s=s,
        q=q,
        q_s=np.ones_like(q),
        q_ss=np.zeros_like(q),
        q_sss=np.zeros_like(q),
    )
    limits = ConstraintLimits(
        q_dot_abs=np.array([1.0]),
        q_ddot_abs=np.array([5.0]),
        q_jerk_abs=np.array([50.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([100.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )

    result = optimize_dp3(path=path, limits=limits, config=DP3Config(ns=6, nz=8, nch=7, k1=2.0, k2=0.0))

    assert result.feasible
    assert result.objective_cost == pytest.approx(2.0 * result.total_time)
    assert result.objective_time_cost == pytest.approx(2.0 * result.total_time)
    assert result.objective_drive_power_cost == pytest.approx(0.0)


def test_dp3_objective_reports_time_and_drive_power_components():
    s = np.linspace(0.0, 1.0, 21)
    q = s[:, None]
    path = PathData(
        s=s,
        q=q,
        q_s=np.ones_like(q),
        q_ss=np.zeros_like(q),
        q_sss=np.zeros_like(q),
    )
    limits = ConstraintLimits(
        q_dot_abs=np.array([1.0]),
        q_ddot_abs=np.array([5.0]),
        q_jerk_abs=np.array([50.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([100.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
        friction_coulomb=np.array([1.0]),
        motor_gear_ratio=np.array([2.0]),
        motor_torque_constant=np.array([0.5]),
        motor_stator_resistance=np.array([1.0]),
    )

    result = optimize_dp3(path=path, limits=limits, config=DP3Config(ns=6, nz=8, nch=7, k1=1.0, k2=1.0))

    assert result.feasible
    assert result.objective_time_cost == pytest.approx(result.total_time)
    assert result.objective_drive_power_cost > 0.0
    assert result.objective_cost == pytest.approx(result.objective_time_cost + result.objective_drive_power_cost)


def test_dp3_energy_weight_requires_motor_power_model():
    s = np.linspace(0.0, 1.0, 21)
    q = s[:, None]
    path = PathData(
        s=s,
        q=q,
        q_s=np.ones_like(q),
        q_ss=np.zeros_like(q),
        q_sss=np.zeros_like(q),
    )
    limits = ConstraintLimits(
        q_dot_abs=np.array([1.0]),
        q_ddot_abs=np.array([5.0]),
        q_jerk_abs=np.array([50.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([100.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )

    with pytest.raises(ValueError, match="motor power model"):
        optimize_dp3(path=path, limits=limits, config=DP3Config(ns=6, nz=8, nch=7, k1=0.0, k2=1.0))
