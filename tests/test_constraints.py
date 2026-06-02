import numpy as np
import pytest

from dp3_topp.constraints import ConstraintLimits, audit_constraints


def test_constraint_audit_reports_normalized_utilization_and_violations():
    limits = ConstraintLimits(
        q_dot_abs=np.array([2.0, 2.0]),
        q_ddot_abs=np.array([5.0, 5.0]),
        q_jerk_abs=np.array([10.0, 10.0]),
        tau_abs=np.array([4.0, 4.0]),
        tau_rate_abs=np.array([8.0, 8.0]),
        mechanical_power_lower=-3.0,
        mechanical_power_upper=6.0,
    )
    audit = audit_constraints(
        limits=limits,
        q_dot=np.array([[1.0, -3.0]]),
        q_ddot=np.array([[0.0, 1.0]]),
        q_jerk=np.array([[4.0, 2.0]]),
        tau=np.array([[2.0, 1.0]]),
        tau_rate=np.array([[0.0, 0.0]]),
        mechanical_power=np.array([2.0]),
    )

    assert not audit.ok
    assert audit.max_utilization["q_dot"] == pytest.approx(1.5)
    assert audit.violations[0].quantity == "q_dot"
    assert audit.violations[0].axis == 1


@pytest.mark.parametrize(
    ("field", "values"),
    (
        ("q_dot", np.array([[np.nan]])),
        ("q_ddot", np.array([[np.inf]])),
        ("q_jerk", np.array([[np.nan]])),
        ("tau", np.array([[np.inf]])),
        ("tau_rate", np.array([[np.nan]])),
        ("mechanical_power", np.array([np.inf])),
    ),
)
def test_constraint_audit_rejects_nonfinite_dynamic_quantities(field, values):
    limits = ConstraintLimits(
        q_dot_abs=np.array([2.0]),
        q_ddot_abs=np.array([5.0]),
        q_jerk_abs=np.array([10.0]),
        tau_abs=np.array([4.0]),
        tau_rate_abs=np.array([8.0]),
        mechanical_power_lower=-3.0,
        mechanical_power_upper=6.0,
    )
    args = {
        "q_dot": np.array([[0.0]]),
        "q_ddot": np.array([[0.0]]),
        "q_jerk": np.array([[0.0]]),
        "tau": np.array([[0.0]]),
        "tau_rate": np.array([[0.0]]),
        "mechanical_power": np.array([0.0]),
    }
    args[field] = values

    with pytest.raises(ValueError, match="finite"):
        audit_constraints(limits=limits, **args)


def test_constraint_audit_rejects_mismatched_sample_counts():
    limits = ConstraintLimits(
        q_dot_abs=np.array([2.0]),
        q_ddot_abs=np.array([5.0]),
        q_jerk_abs=np.array([10.0]),
        tau_abs=np.array([4.0]),
        tau_rate_abs=np.array([8.0]),
        mechanical_power_lower=-3.0,
        mechanical_power_upper=6.0,
    )

    with pytest.raises(ValueError, match="same number of samples"):
        audit_constraints(
            limits=limits,
            q_dot=np.zeros((2, 1)),
            q_ddot=np.zeros((1, 1)),
            q_jerk=np.zeros((2, 1)),
            tau=np.zeros((2, 1)),
            tau_rate=np.zeros((2, 1)),
            mechanical_power=np.zeros(2),
        )


@pytest.mark.parametrize(
    ("lower", "upper"),
    (
        (0.0, 10.0),
        (-10.0, 0.0),
        (1.0, 10.0),
        (-10.0, -1.0),
    ),
)
def test_mechanical_power_limits_must_straddle_zero(lower, upper):
    with pytest.raises(ValueError, match="mechanical power limits must straddle zero"):
        ConstraintLimits(
            q_dot_abs=np.array([2.0]),
            q_ddot_abs=np.array([5.0]),
            q_jerk_abs=np.array([10.0]),
            tau_abs=np.array([4.0]),
            tau_rate_abs=np.array([8.0]),
            mechanical_power_lower=lower,
            mechanical_power_upper=upper,
        )


def test_constraint_audit_uses_velocity_dependent_torque_limits():
    limits = ConstraintLimits(
        q_dot_abs=np.array([5.0]),
        q_ddot_abs=np.array([5.0]),
        q_jerk_abs=np.array([5.0]),
        tau_abs=np.array([10.0]),
        tau_rate_abs=np.array([5.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
        torque_speed_breakpoints=[
            np.array([[0.0, 10.0], [1.0, 5.0], [2.0, 2.0]], dtype=float)
        ],
    )

    audit = audit_constraints(
        limits=limits,
        q_dot=np.array([[1.5]]),
        q_ddot=np.array([[0.0]]),
        q_jerk=np.array([[0.0]]),
        tau=np.array([[3.0]]),
        tau_rate=np.array([[0.0]]),
        mechanical_power=np.array([0.0]),
    )

    assert audit.ok
    assert audit.max_utilization["tau"] == pytest.approx(3.0 / 3.5)


def test_limits_yaml_supports_paper_style_asymmetric_lower_upper_bounds(tmp_path):
    path = tmp_path / "limits.yaml"
    path.write_text(
        "q_dot:\n"
        "  lower: [-1.0]\n"
        "  upper: [2.0]\n"
        "q_ddot:\n"
        "  lower: [-3.0]\n"
        "  upper: [5.0]\n"
        "q_jerk:\n"
        "  lower: [-7.0]\n"
        "  upper: [11.0]\n"
        "tau:\n"
        "  lower: [-13.0]\n"
        "  upper: [17.0]\n"
        "tau_rate:\n"
        "  lower: [-19.0]\n"
        "  upper: [23.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )

    limits = ConstraintLimits.from_yaml(path)
    audit = audit_constraints(
        limits=limits,
        q_dot=np.array([[-1.5]]),
        q_ddot=np.array([[4.0]]),
        q_jerk=np.array([[-8.0]]),
        tau=np.array([[16.0]]),
        tau_rate=np.array([[24.0]]),
        mechanical_power=np.array([0.0]),
    )

    assert not audit.ok
    assert limits.q_dot_abs.tolist() == [2.0]
    assert [violation.quantity for violation in audit.violations] == ["q_dot", "q_jerk", "tau_rate"]
    assert audit.violations[0].limit == pytest.approx(-1.0)
    assert audit.max_utilization["q_dot"] == pytest.approx(1.5)
    assert audit.max_utilization["q_ddot"] == pytest.approx(4.0 / 5.0)
    assert audit.max_utilization["tau"] == pytest.approx(16.0 / 17.0)


def test_limits_yaml_supports_optional_joint_position_bounds(tmp_path):
    path = tmp_path / "limits.yaml"
    path.write_text(
        "q_position:\n"
        "  lower: [-1.0, -2.0]\n"
        "  upper: [1.0, 2.0]\n"
        "q_dot_abs: [5.0, 5.0]\n"
        "q_ddot_abs: [5.0, 5.0]\n"
        "q_jerk_abs: [5.0, 5.0]\n"
        "tau_abs: [10.0, 10.0]\n"
        "tau_rate_abs: [20.0, 20.0]\n"
        "mechanical_power:\n"
        "  lower: -100.0\n"
        "  upper: 100.0\n",
        encoding="utf-8",
    )

    limits = ConstraintLimits.from_yaml(path)
    audit = audit_constraints(
        limits=limits,
        q_position=np.array([[0.5, 2.5]]),
        q_dot=np.zeros((1, 2)),
        q_ddot=np.zeros((1, 2)),
        q_jerk=np.zeros((1, 2)),
        tau=np.zeros((1, 2)),
        tau_rate=np.zeros((1, 2)),
        mechanical_power=np.array([0.0]),
    )

    assert not audit.ok
    assert audit.max_utilization["q_position"] == pytest.approx(1.25)
    assert [violation.quantity for violation in audit.violations] == ["q_position"]
    assert audit.violations[0].axis == 1
    assert audit.violations[0].limit == pytest.approx(2.0)


def test_joint_position_bounds_use_interval_centered_utilization():
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

    audit = audit_constraints(
        limits=limits,
        q_position=np.array([[0.5], [2.0], [3.5]]),
        q_dot=np.zeros((3, 1)),
        q_ddot=np.zeros((3, 1)),
        q_jerk=np.zeros((3, 1)),
        tau=np.zeros((3, 1)),
        tau_rate=np.zeros((3, 1)),
        mechanical_power=np.zeros(3),
    )

    assert not audit.ok
    assert audit.max_utilization["q_position"] == pytest.approx(1.5)
    assert [violation.utilization for violation in audit.violations] == pytest.approx([1.5, 1.5])


def test_torque_speed_limit_uses_motor_side_speed_and_joint_side_torque_when_gear_ratio_is_available():
    limits = ConstraintLimits(
        q_dot_abs=np.array([5.0]),
        q_ddot_abs=np.array([5.0]),
        q_jerk_abs=np.array([5.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([5.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
        torque_speed_breakpoints=[
            np.array([[0.0, 10.0], [10.0, 5.0], [20.0, 2.0]], dtype=float)
        ],
        motor_gear_ratio=np.array([4.0]),
    )

    limit = limits.torque_abs_limit(np.array([[2.5]]))

    np.testing.assert_allclose(limit, np.array([[20.0]]))


def test_torque_speed_tables_reject_negative_speed_breakpoints():
    with pytest.raises(ValueError, match="nonnegative speeds"):
        ConstraintLimits(
            q_dot_abs=np.array([5.0]),
            q_ddot_abs=np.array([5.0]),
            q_jerk_abs=np.array([5.0]),
            tau_abs=np.array([100.0]),
            tau_rate_abs=np.array([5.0]),
            mechanical_power_lower=-100.0,
            mechanical_power_upper=100.0,
            torque_speed_breakpoints=[
                np.array([[-1.0, 10.0], [1.0, 5.0]], dtype=float)
            ],
        )


@pytest.mark.parametrize(
    ("method_name", "arg"),
    (
        ("torque_abs_limit", np.array([[np.nan]])),
        ("friction_torque", np.array([[np.inf]])),
        ("friction_torque_rate", np.array([[np.nan]])),
    ),
)
def test_constraint_limit_helpers_reject_nonfinite_inputs(method_name, arg):
    limits = ConstraintLimits(
        q_dot_abs=np.array([5.0]),
        q_ddot_abs=np.array([5.0]),
        q_jerk_abs=np.array([5.0]),
        tau_abs=np.array([100.0]),
        tau_rate_abs=np.array([5.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
        torque_speed_breakpoints=[
            np.array([[0.0, 10.0], [1.0, 5.0]], dtype=float)
        ],
    )

    with pytest.raises(ValueError, match="finite"):
        getattr(limits, method_name)(arg)


def test_limits_yaml_rejects_missing_required_fields(tmp_path):
    path = tmp_path / "limits.yaml"
    path.write_text("q_dot_abs: [1, 1]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required limit fields"):
        ConstraintLimits.from_yaml(path)


def test_limits_yaml_rejects_malformed_yaml_with_clear_error(tmp_path):
    path = tmp_path / "limits.yaml"
    path.write_text("q_dot_abs: [\n", encoding="utf-8")

    with pytest.raises(ValueError, match="limits YAML is invalid"):
        ConstraintLimits.from_yaml(path)


def test_limits_yaml_rejects_nonfinite_or_placeholder_values(tmp_path):
    path = tmp_path / "limits.yaml"
    path.write_text(
        "q_dot_abs: [nan]\n"
        "q_ddot_abs: [5]\n"
        "q_jerk_abs: [5]\n"
        "tau_abs: [20]\n"
        "tau_rate_abs: [20]\n"
        "mechanical_power:\n"
        "  lower: -100\n"
        "  upper: 100\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite positive"):
        ConstraintLimits.from_yaml(path)


def test_limits_yaml_parses_friction_terms(tmp_path):
    path = tmp_path / "limits.yaml"
    path.write_text(
        "q_dot_abs: [5, 5]\n"
        "q_ddot_abs: [5, 5]\n"
        "q_jerk_abs: [5, 5]\n"
        "tau_abs: [20, 20]\n"
        "tau_rate_abs: [20, 20]\n"
        "mechanical_power:\n"
        "  lower: -100\n"
        "  upper: 100\n"
        "friction:\n"
        "  coulomb: [1.0, 2.0]\n"
        "  viscous: [0.5, 0.25]\n",
        encoding="utf-8",
    )

    limits = ConstraintLimits.from_yaml(path)
    tau_friction = limits.friction_torque(np.array([[2.0, -4.0], [0.0, 0.0]]))
    tau_rate_friction = limits.friction_torque_rate(np.array([[3.0, -2.0]]))

    np.testing.assert_allclose(tau_friction, np.array([[2.0, -3.0], [0.0, 0.0]]))
    np.testing.assert_allclose(tau_rate_friction, np.array([[1.5, -0.5]]))


def test_limits_yaml_parses_motor_model_for_drive_power(tmp_path):
    path = tmp_path / "limits.yaml"
    path.write_text(
        "q_dot_abs: [5, 5]\n"
        "q_ddot_abs: [5, 5]\n"
        "q_jerk_abs: [5, 5]\n"
        "tau_abs: [20, 20]\n"
        "tau_rate_abs: [20, 20]\n"
        "mechanical_power:\n"
        "  lower: -100\n"
        "  upper: 100\n"
        "motor:\n"
        "  gear_ratio: [2.0, 4.0]\n"
        "  torque_constant: [0.5, 1.0]\n"
        "  stator_resistance: [1.0, 2.0]\n",
        encoding="utf-8",
    )

    limits = ConstraintLimits.from_yaml(path)
    power = limits.drive_power(
        tau=np.array([[2.0, -2.0]]),
        q_dot=np.array([[3.0, -4.0]]),
    )

    np.testing.assert_allclose(power, np.array([18.5]))


def test_drive_power_rejects_mismatched_sample_counts():
    limits = ConstraintLimits(
        q_dot_abs=np.array([5.0]),
        q_ddot_abs=np.array([5.0]),
        q_jerk_abs=np.array([5.0]),
        tau_abs=np.array([20.0]),
        tau_rate_abs=np.array([20.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
        motor_gear_ratio=np.array([2.0]),
        motor_torque_constant=np.array([0.5]),
        motor_stator_resistance=np.array([1.0]),
    )

    with pytest.raises(ValueError, match="same shape"):
        limits.drive_power(tau=np.zeros((2, 1)), q_dot=np.zeros((1, 1)))


def test_drive_power_requires_complete_motor_model():
    limits = ConstraintLimits(
        q_dot_abs=np.array([5.0]),
        q_ddot_abs=np.array([5.0]),
        q_jerk_abs=np.array([5.0]),
        tau_abs=np.array([20.0]),
        tau_rate_abs=np.array([20.0]),
        mechanical_power_lower=-100.0,
        mechanical_power_upper=100.0,
    )

    with pytest.raises(ValueError, match="motor power model"):
        limits.drive_power(tau=np.array([[1.0]]), q_dot=np.array([[1.0]]))
