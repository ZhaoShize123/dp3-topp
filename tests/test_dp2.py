import numpy as np

from dp3_topp.constraints import ConstraintLimits
from dp3_topp.optimizer import DP3Config, optimize_dp2
from dp3_topp.path_data import PathData


def _linear_path() -> PathData:
    s = np.linspace(0.0, 1.0, 21)
    q = s[:, None]
    return PathData(
        s=s,
        q=q,
        q_s=np.ones_like(q),
        q_ss=np.zeros_like(q),
        q_sss=np.zeros_like(q),
    )


def _curved_path() -> PathData:
    s = np.linspace(0.0, 1.0, 21)
    q = (s + 2.5 * s * s)[:, None]
    return PathData(
        s=s,
        q=q,
        q_s=(1.0 + 5.0 * s)[:, None],
        q_ss=np.full((s.size, 1), 5.0),
        q_sss=np.zeros((s.size, 1)),
    )


def _wide_limits() -> ConstraintLimits:
    return ConstraintLimits(
        q_dot_abs=np.array([1.0]),
        q_ddot_abs=np.array([5.0]),
        q_jerk_abs=np.array([50.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([100.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )


def test_optimize_dp2_baseline_finds_zero_to_zero_profile():
    result = optimize_dp2(path=_linear_path(), limits=_wide_limits(), config=DP3Config(ns=6, nz=8, nch=7))

    assert result.feasible
    assert result.method == "DP2"
    assert result.jerk_limited == "no"
    assert result.z[0] == 0.0
    assert result.z[-1] == 0.0


def test_optimize_dp2_only_start_end_applies_third_order_limits_at_zero_speed_regions():
    limits = ConstraintLimits(
        q_dot_abs=np.array([100.0]),
        q_ddot_abs=np.array([100.0]),
        q_jerk_abs=np.array([1e-9]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([1e-9]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )
    config = DP3Config(ns=4, nz=5, nch=5, z_max=1.0)

    unrestricted = optimize_dp2(path=_curved_path(), limits=limits, config=config, jerk_limited="no")
    start_end_limited = optimize_dp2(
        path=_curved_path(),
        limits=limits,
        config=config,
        jerk_limited="only start / end",
    )

    assert unrestricted.s.size > 0
    assert start_end_limited.s.size == 0


def test_optimize_dp2_preserves_fixed_path_joint_position_limits_when_relaxing_third_order_constraints():
    limits = ConstraintLimits(
        q_position_lower=np.array([-0.1]),
        q_position_upper=np.array([0.5]),
        q_dot_abs=np.array([100.0]),
        q_ddot_abs=np.array([100.0]),
        q_jerk_abs=np.array([1e-9]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([1e-9]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )

    result = optimize_dp2(path=_linear_path(), limits=limits, config=DP3Config(ns=6, nz=8, nch=7), jerk_limited="no")

    assert not result.feasible
    assert result.s.size == 0
    assert any(violation.quantity == "q_position" for violation in result.audit.violations)
