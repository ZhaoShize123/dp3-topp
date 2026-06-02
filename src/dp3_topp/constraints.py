from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class ConstraintViolation:
    quantity: str
    sample: int
    axis: int | None
    value: float
    limit: float
    utilization: float


@dataclass(frozen=True)
class ConstraintAudit:
    ok: bool
    max_utilization: dict[str, float]
    violations: list[ConstraintViolation]


@dataclass(frozen=True)
class ConstraintLimits:
    q_dot_abs: np.ndarray
    q_ddot_abs: np.ndarray
    q_jerk_abs: np.ndarray
    tau_abs: np.ndarray
    tau_rate_abs: np.ndarray
    mechanical_power_lower: float
    mechanical_power_upper: float
    torque_speed_breakpoints: list[np.ndarray] | None = None
    friction_coulomb: np.ndarray | None = None
    friction_viscous: np.ndarray | None = None
    motor_gear_ratio: np.ndarray | None = None
    motor_torque_constant: np.ndarray | None = None
    motor_stator_resistance: np.ndarray | None = None
    q_dot_lower: np.ndarray | None = None
    q_dot_upper: np.ndarray | None = None
    q_ddot_lower: np.ndarray | None = None
    q_ddot_upper: np.ndarray | None = None
    q_jerk_lower: np.ndarray | None = None
    q_jerk_upper: np.ndarray | None = None
    tau_lower: np.ndarray | None = None
    tau_upper: np.ndarray | None = None
    tau_rate_lower: np.ndarray | None = None
    tau_rate_upper: np.ndarray | None = None
    q_position_lower: np.ndarray | None = None
    q_position_upper: np.ndarray | None = None

    def __post_init__(self) -> None:
        arrays = {
            "q_dot_abs": np.asarray(self.q_dot_abs, dtype=np.float64),
            "q_ddot_abs": np.asarray(self.q_ddot_abs, dtype=np.float64),
            "q_jerk_abs": np.asarray(self.q_jerk_abs, dtype=np.float64),
            "tau_abs": np.asarray(self.tau_abs, dtype=np.float64),
            "tau_rate_abs": np.asarray(self.tau_rate_abs, dtype=np.float64),
        }
        dof = None
        for name, values in arrays.items():
            if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(f"{name} must be a non-empty finite positive one-dimensional array")
            dof = values.size if dof is None else dof
            if values.size != dof:
                raise ValueError("all per-axis limit arrays must have the same length")
            object.__setattr__(self, name, values)
        for quantity, abs_name, lower_name, upper_name in (
            ("q_dot", "q_dot_abs", "q_dot_lower", "q_dot_upper"),
            ("q_ddot", "q_ddot_abs", "q_ddot_lower", "q_ddot_upper"),
            ("q_jerk", "q_jerk_abs", "q_jerk_lower", "q_jerk_upper"),
            ("tau", "tau_abs", "tau_lower", "tau_upper"),
            ("tau_rate", "tau_rate_abs", "tau_rate_lower", "tau_rate_upper"),
        ):
            lower, upper = _normalize_axis_bounds(
                quantity,
                getattr(self, lower_name),
                getattr(self, upper_name),
                arrays[abs_name],
                int(dof),
            )
            object.__setattr__(self, lower_name, lower)
            object.__setattr__(self, upper_name, upper)
        position_lower, position_upper = _normalize_position_bounds(
            self.q_position_lower,
            self.q_position_upper,
            int(dof),
        )
        object.__setattr__(self, "q_position_lower", position_lower)
        object.__setattr__(self, "q_position_upper", position_upper)
        if not np.isfinite(self.mechanical_power_lower) or not np.isfinite(self.mechanical_power_upper):
            raise ValueError("mechanical power limits must be finite")
        if not self.mechanical_power_lower < self.mechanical_power_upper:
            raise ValueError("mechanical power lower limit must be less than upper limit")
        if not self.mechanical_power_lower < 0.0 < self.mechanical_power_upper:
            raise ValueError("mechanical power limits must straddle zero")
        for name in ("friction_coulomb", "friction_viscous"):
            raw = getattr(self, name)
            values = np.zeros(dof, dtype=np.float64) if raw is None else np.asarray(raw, dtype=np.float64)
            if values.shape != (dof,) or np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{name} must be a finite nonnegative array with one value per axis")
            object.__setattr__(self, name, values)
        for name in ("motor_gear_ratio", "motor_torque_constant", "motor_stator_resistance"):
            raw = getattr(self, name)
            if raw is None:
                continue
            values = np.asarray(raw, dtype=np.float64)
            if values.shape != (dof,) or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(f"{name} must be a finite positive array with one value per axis")
            object.__setattr__(self, name, values)
        if self.torque_speed_breakpoints is not None:
            if len(self.torque_speed_breakpoints) != dof:
                raise ValueError("torque_speed_breakpoints must have one table per axis")
            normalized: list[np.ndarray] = []
            for table in self.torque_speed_breakpoints:
                arr = np.asarray(table, dtype=np.float64)
                if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
                    raise ValueError("each torque-speed table must be Nx2")
                if np.any(~np.isfinite(arr)) or np.any(arr[:, 0] < 0.0) or np.any(np.diff(arr[:, 0]) <= 0.0) or np.any(arr[:, 1] <= 0.0):
                    raise ValueError("torque-speed tables require finite nonnegative speeds, increasing speed breakpoints, and positive torque limits")
                normalized.append(arr)
            object.__setattr__(self, "torque_speed_breakpoints", normalized)

    @property
    def dof(self) -> int:
        return int(self.q_dot_abs.size)

    @classmethod
    def from_yaml(cls, path: Path) -> "ConstraintLimits":
        try:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"limits YAML is invalid: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("limits YAML must contain a mapping")
        axis_limits = {
            quantity: _parse_axis_limit(raw, abs_key, quantity)
            for quantity, abs_key in (
                ("q_dot", "q_dot_abs"),
                ("q_ddot", "q_ddot_abs"),
                ("q_jerk", "q_jerk_abs"),
                ("tau", "tau_abs"),
                ("tau_rate", "tau_rate_abs"),
            )
        }
        missing = [
            f"{abs_key} or {quantity}"
            for quantity, abs_key in (
                ("q_dot", "q_dot_abs"),
                ("q_ddot", "q_ddot_abs"),
                ("q_jerk", "q_jerk_abs"),
                ("tau", "tau_abs"),
                ("tau_rate", "tau_rate_abs"),
            )
            if axis_limits[quantity] is None
        ]
        if "mechanical_power" not in raw:
            missing.append("mechanical_power")
        if missing:
            raise ValueError(f"missing required limit fields: {', '.join(missing)}")
        power = raw["mechanical_power"]
        if not isinstance(power, dict) or "lower" not in power or "upper" not in power:
            raise ValueError("mechanical_power must define lower and upper")
        torque_speed = _parse_torque_speed(raw.get("torque_speed_breakpoints"))
        friction = raw.get("friction", {})
        if friction is None:
            friction = {}
        if not isinstance(friction, dict):
            raise ValueError("friction must be a mapping when provided")
        motor = raw.get("motor", {})
        if motor is None:
            motor = {}
        if not isinstance(motor, dict):
            raise ValueError("motor must be a mapping when provided")
        q_dot_abs, q_dot_lower, q_dot_upper = axis_limits["q_dot"]
        q_ddot_abs, q_ddot_lower, q_ddot_upper = axis_limits["q_ddot"]
        q_jerk_abs, q_jerk_lower, q_jerk_upper = axis_limits["q_jerk"]
        tau_abs, tau_lower, tau_upper = axis_limits["tau"]
        tau_rate_abs, tau_rate_lower, tau_rate_upper = axis_limits["tau_rate"]
        q_position_lower, q_position_upper = _parse_optional_position_bounds(raw.get("q_position"))
        return cls(
            q_dot_abs=q_dot_abs,
            q_ddot_abs=q_ddot_abs,
            q_jerk_abs=q_jerk_abs,
            tau_abs=tau_abs,
            tau_rate_abs=tau_rate_abs,
            mechanical_power_lower=_parse_float(power["lower"], "mechanical_power.lower"),
            mechanical_power_upper=_parse_float(power["upper"], "mechanical_power.upper"),
            torque_speed_breakpoints=torque_speed,
            friction_coulomb=_parse_optional_array(friction.get("coulomb")),
            friction_viscous=_parse_optional_array(friction.get("viscous")),
            motor_gear_ratio=_parse_optional_array(motor.get("gear_ratio")),
            motor_torque_constant=_parse_optional_array(motor.get("torque_constant")),
            motor_stator_resistance=_parse_optional_array(motor.get("stator_resistance")),
            q_dot_lower=q_dot_lower,
            q_dot_upper=q_dot_upper,
            q_ddot_lower=q_ddot_lower,
            q_ddot_upper=q_ddot_upper,
            q_jerk_lower=q_jerk_lower,
            q_jerk_upper=q_jerk_upper,
            tau_lower=tau_lower,
            tau_upper=tau_upper,
            tau_rate_lower=tau_rate_lower,
            tau_rate_upper=tau_rate_upper,
            q_position_lower=q_position_lower,
            q_position_upper=q_position_upper,
        )

    def q_dot_bounds(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        return _broadcast_bounds(self.q_dot_lower, self.q_dot_upper, shape)

    def q_ddot_bounds(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        return _broadcast_bounds(self.q_ddot_lower, self.q_ddot_upper, shape)

    def q_jerk_bounds(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        return _broadcast_bounds(self.q_jerk_lower, self.q_jerk_upper, shape)

    def tau_rate_bounds(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        return _broadcast_bounds(self.tau_rate_lower, self.tau_rate_upper, shape)

    @property
    def has_q_position_bounds(self) -> bool:
        return self.q_position_lower is not None and self.q_position_upper is not None

    def q_position_bounds(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        if not self.has_q_position_bounds:
            raise ValueError("q_position bounds are not configured")
        return _broadcast_bounds(self.q_position_lower, self.q_position_upper, shape)

    def torque_bounds(self, q_dot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_dot = _check_matrix("q_dot", q_dot, self.dof)
        static_lower, static_upper = _broadcast_bounds(self.tau_lower, self.tau_upper, q_dot.shape)
        speed_abs = self.torque_abs_limit(q_dot)
        return np.maximum(static_lower, -speed_abs), np.minimum(static_upper, speed_abs)

    def torque_abs_limit(self, q_dot: np.ndarray) -> np.ndarray:
        q_dot = _check_matrix("q_dot", q_dot, self.dof)
        if self.torque_speed_breakpoints is None:
            return np.broadcast_to(self.tau_abs, q_dot.shape).copy()
        out = np.zeros_like(q_dot)
        speed = np.abs(q_dot)
        for axis, table in enumerate(self.torque_speed_breakpoints):
            if self.motor_gear_ratio is None:
                lookup_speed = speed[:, axis]
                torque_scale = 1.0
            else:
                lookup_speed = speed[:, axis] * float(self.motor_gear_ratio[axis])
                torque_scale = float(self.motor_gear_ratio[axis])
            out[:, axis] = torque_scale * np.interp(
                lookup_speed,
                table[:, 0],
                table[:, 1],
                left=table[0, 1],
                right=table[-1, 1],
            )
        return np.minimum(out, np.broadcast_to(self.tau_abs, q_dot.shape))

    def friction_torque(self, q_dot: np.ndarray) -> np.ndarray:
        q_dot = _check_matrix("q_dot", q_dot, self.dof)
        return np.sign(q_dot) * self.friction_coulomb + q_dot * self.friction_viscous

    def friction_torque_rate(self, q_ddot: np.ndarray) -> np.ndarray:
        q_ddot = _check_matrix("q_ddot", q_ddot, self.dof)
        return q_ddot * self.friction_viscous

    @property
    def has_motor_power_model(self) -> bool:
        return (
            self.motor_gear_ratio is not None
            and self.motor_torque_constant is not None
            and self.motor_stator_resistance is not None
        )

    def drive_power(self, tau: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        if not self.has_motor_power_model:
            raise ValueError("motor power model requires gear_ratio, torque_constant, and stator_resistance")
        tau = _check_matrix("tau", tau, self.dof)
        q_dot = _check_matrix("q_dot", q_dot, self.dof)
        if tau.shape != q_dot.shape:
            raise ValueError("tau and q_dot must have the same shape")
        copper = np.sum(
            tau * tau * self.motor_stator_resistance / ((self.motor_gear_ratio * self.motor_torque_constant) ** 2),
            axis=1,
        )
        mechanical = np.sum(np.maximum(0.0, tau * q_dot), axis=1)
        return copper + mechanical


def _parse_torque_speed(raw: Any) -> list[np.ndarray] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("torque_speed_breakpoints must be a list of axis tables")
    return [np.asarray(item, dtype=np.float64) for item in raw]


def _parse_axis_limit(raw: dict[str, Any], abs_key: str, bounds_key: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None] | None:
    has_abs = abs_key in raw
    has_bounds = bounds_key in raw
    if not has_abs and not has_bounds:
        return None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    if has_bounds:
        lower, upper = _parse_axis_bounds_mapping(raw[bounds_key], bounds_key)
    if has_abs:
        abs_values = np.asarray(raw[abs_key], dtype=np.float64)
    else:
        abs_values = np.maximum(np.abs(lower), np.abs(upper))
    return abs_values, lower, upper


def _parse_axis_bounds_mapping(raw: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(raw, dict) or "lower" not in raw or "upper" not in raw:
        raise ValueError(f"{name} must define lower and upper")
    return np.asarray(raw["lower"], dtype=np.float64), np.asarray(raw["upper"], dtype=np.float64)


def _parse_optional_position_bounds(raw: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    if raw is None:
        return None, None
    return _parse_axis_bounds_mapping(raw, "q_position")


def _normalize_axis_bounds(
    name: str,
    lower_raw: np.ndarray | None,
    upper_raw: np.ndarray | None,
    fallback_abs: np.ndarray,
    dof: int,
) -> tuple[np.ndarray, np.ndarray]:
    if lower_raw is None and upper_raw is None:
        return -fallback_abs.copy(), fallback_abs.copy()
    if lower_raw is None or upper_raw is None:
        raise ValueError(f"{name} lower and upper limits must be provided together")
    lower = np.asarray(lower_raw, dtype=np.float64)
    upper = np.asarray(upper_raw, dtype=np.float64)
    if lower.shape != (dof,) or upper.shape != (dof,):
        raise ValueError(f"{name} lower and upper limits must have one value per axis")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError(f"{name} lower and upper limits must be finite")
    if np.any(lower >= 0.0) or np.any(upper <= 0.0) or np.any(lower >= upper):
        raise ValueError(f"{name} lower and upper limits must satisfy lower < 0 < upper")
    return lower, upper


def _normalize_position_bounds(
    lower_raw: np.ndarray | None,
    upper_raw: np.ndarray | None,
    dof: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if lower_raw is None and upper_raw is None:
        return None, None
    if lower_raw is None or upper_raw is None:
        raise ValueError("q_position lower and upper limits must be provided together")
    lower = np.asarray(lower_raw, dtype=np.float64)
    upper = np.asarray(upper_raw, dtype=np.float64)
    if lower.shape != (dof,) or upper.shape != (dof,):
        raise ValueError("q_position lower and upper limits must have one value per axis")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("q_position lower and upper limits must be finite")
    if np.any(lower >= upper):
        raise ValueError("q_position lower and upper limits must satisfy lower < upper")
    return lower, upper


def _broadcast_bounds(lower: np.ndarray, upper: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    return np.broadcast_to(lower, shape), np.broadcast_to(upper, shape)


def _parse_optional_array(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    return np.asarray(raw, dtype=np.float64)


def _parse_float(raw: Any, name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def audit_constraints(
    *,
    limits: ConstraintLimits,
    q_dot: np.ndarray,
    q_ddot: np.ndarray,
    q_jerk: np.ndarray,
    tau: np.ndarray,
    tau_rate: np.ndarray,
    mechanical_power: np.ndarray,
    q_position: np.ndarray | None = None,
) -> ConstraintAudit:
    q_dot = _check_matrix("q_dot", q_dot, limits.dof)
    q_ddot = _check_matrix("q_ddot", q_ddot, limits.dof)
    q_jerk = _check_matrix("q_jerk", q_jerk, limits.dof)
    tau = _check_matrix("tau", tau, limits.dof)
    tau_rate = _check_matrix("tau_rate", tau_rate, limits.dof)
    checked_position = None
    if limits.has_q_position_bounds:
        if q_position is None:
            raise ValueError("q_position must be provided when q_position limits are configured")
        checked_position = _check_matrix("q_position", q_position, limits.dof)
        _check_same_sample_count(q_dot, q_ddot, q_jerk, tau, tau_rate, checked_position)
    else:
        _check_same_sample_count(q_dot, q_ddot, q_jerk, tau, tau_rate)
    mechanical_power = np.asarray(mechanical_power, dtype=np.float64)
    if mechanical_power.shape != (q_dot.shape[0],):
        raise ValueError("mechanical_power must have one value per sample")
    if np.any(~np.isfinite(mechanical_power)):
        raise ValueError("mechanical_power must contain only finite values")

    max_util: dict[str, float] = {}
    violations: list[ConstraintViolation] = []
    if checked_position is not None:
        _audit_interval_bounded("q_position", checked_position, *limits.q_position_bounds(checked_position.shape), max_util, violations)
    _audit_bounded("q_dot", q_dot, *limits.q_dot_bounds(q_dot.shape), max_util, violations)
    _audit_bounded("q_ddot", q_ddot, *limits.q_ddot_bounds(q_ddot.shape), max_util, violations)
    _audit_bounded("q_jerk", q_jerk, *limits.q_jerk_bounds(q_jerk.shape), max_util, violations)
    _audit_bounded("tau", tau, *limits.torque_bounds(q_dot), max_util, violations)
    _audit_bounded("tau_rate", tau_rate, *limits.tau_rate_bounds(tau_rate.shape), max_util, violations)
    _audit_power(mechanical_power, limits.mechanical_power_lower, limits.mechanical_power_upper, max_util, violations)
    return ConstraintAudit(ok=not violations, max_utilization=max_util, violations=violations)


def _check_matrix(name: str, values: np.ndarray, dof: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != dof:
        raise ValueError(f"{name} must have shape (samples, {dof})")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _check_same_sample_count(*arrays: np.ndarray) -> None:
    sample_count = arrays[0].shape[0]
    if any(array.shape[0] != sample_count for array in arrays[1:]):
        raise ValueError("all constraint quantity matrices must have the same number of samples")


def signed_bound_utilization(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.where(values >= 0.0, values / upper, values / np.abs(lower))


def interval_bound_utilization(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    center = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    return (values - center) / half_width


def _audit_bounded(
    name: str,
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_util: dict[str, float],
    violations: list[ConstraintViolation],
) -> None:
    util = signed_bound_utilization(values, lower, upper)
    abs_util = np.abs(util)
    max_util[name] = float(np.max(abs_util)) if abs_util.size else 0.0
    for sample, axis in np.argwhere((values < lower - 1e-12) | (values > upper + 1e-12)):
        limit = upper[sample, axis] if values[sample, axis] > upper[sample, axis] else lower[sample, axis]
        violations.append(
            ConstraintViolation(
                quantity=name,
                sample=int(sample),
                axis=int(axis),
                value=float(values[sample, axis]),
                limit=float(limit),
                utilization=float(abs_util[sample, axis]),
            )
        )


def _audit_interval_bounded(
    name: str,
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_util: dict[str, float],
    violations: list[ConstraintViolation],
) -> None:
    util = interval_bound_utilization(values, lower, upper)
    abs_util = np.abs(util)
    max_util[name] = float(np.max(abs_util)) if abs_util.size else 0.0
    for sample, axis in np.argwhere((values < lower - 1e-12) | (values > upper + 1e-12)):
        limit = upper[sample, axis] if values[sample, axis] > upper[sample, axis] else lower[sample, axis]
        violations.append(
            ConstraintViolation(
                quantity=name,
                sample=int(sample),
                axis=int(axis),
                value=float(values[sample, axis]),
                limit=float(limit),
                utilization=float(abs_util[sample, axis]),
            )
        )


def _audit_power(
    values: np.ndarray,
    lower: float,
    upper: float,
    max_util: dict[str, float],
    violations: list[ConstraintViolation],
) -> None:
    positive_util = np.where(values >= 0.0, values / upper, values / abs(lower))
    max_util["mechanical_power"] = float(np.max(np.abs(positive_util))) if values.size else 0.0
    for sample in np.argwhere((values < lower - 1e-12) | (values > upper + 1e-12)).ravel():
        limit = upper if values[sample] > upper else lower
        denom = upper if values[sample] > upper else abs(lower)
        violations.append(
            ConstraintViolation(
                quantity="mechanical_power",
                sample=int(sample),
                axis=None,
                value=float(values[sample]),
                limit=float(limit),
                utilization=float(abs(values[sample]) / denom),
            )
        )
