from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dp3_topp.dynamics_mujoco import MujocoRobotDynamics


AXIS_LIMIT_FIELDS = (
    ("q_dot_abs", "q_dot"),
    ("q_ddot_abs", "q_ddot"),
    ("q_jerk_abs", "q_jerk"),
    ("tau_abs", "tau"),
    ("tau_rate_abs", "tau_rate"),
)

LIMIT_REQUIRED_FIELDS = tuple(abs_key for abs_key, _ in AXIS_LIMIT_FIELDS) + ("mechanical_power",)


FULL_REPRODUCTION_FIELDS = (
    "torque_speed_breakpoints",
    "friction.coulomb",
    "friction.viscous",
    "motor.gear_ratio",
    "motor.torque_constant",
    "motor.stator_resistance",
)


def write_t12a_limits_template(path: Path, dof: int = 6, model_path: Path | None = None) -> None:
    values = [None] * int(dof)
    data: dict[str, Any] = {
        "robot": "T12A",
        "units": {
            "q_position": "rad",
            "q_dot_abs": "rad/s",
            "q_ddot_abs": "rad/s^2",
            "q_jerk_abs": "rad/s^3",
            "tau_abs": "N*m",
            "tau_rate_abs": "N*m/s",
            "mechanical_power": "W",
        },
        "q_dot_abs": values.copy(),
        "q_ddot_abs": values.copy(),
        "q_jerk_abs": values.copy(),
        "tau_abs": values.copy(),
        "tau_rate_abs": values.copy(),
        "mechanical_power": {"lower": None, "upper": None},
        "torque_speed_breakpoints": [[[0.0, None], [None, None]] for _ in range(int(dof))],
        "friction": {
            "coulomb": values.copy(),
            "viscous": values.copy(),
        },
        "motor": {
            "gear_ratio": values.copy(),
            "torque_constant": values.copy(),
            "stator_resistance": values.copy(),
        },
    }
    if model_path is not None:
        robot = MujocoRobotDynamics.from_model_path(Path(model_path))
        if robot.dof != int(dof):
            raise ValueError(f"MuJoCo model DOF ({robot.dof}) does not match template DOF ({int(dof)})")
        data["joint_names"] = list(robot.joint_names)
        data["q_position"] = {
            "lower": [float(value) for value in robot.lower],
            "upper": [float(value) for value in robot.upper],
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")


def full_reproduction_gaps(path: Path, expected_dof: int | None = None) -> list[str]:
    limits_path = Path(path)
    if not limits_path.exists():
        return [str(limits_path)]
    try:
        raw = yaml.safe_load(limits_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [f"limits YAML is invalid: {exc}"]
    if not isinstance(raw, dict):
        return ["limits_yaml_mapping"]

    gaps: list[str] = []
    for abs_key, bounds_key in AXIS_LIMIT_FIELDS:
        if not _has_axis_limit(raw, abs_key, bounds_key):
            gaps.append(abs_key)
    if _is_missing(raw.get("mechanical_power")):
        gaps.append("mechanical_power")
    gaps.extend(_missing_full_fields(raw))

    if expected_dof is not None:
        gaps.extend(_dof_gaps(raw, int(expected_dof)))
    return sorted(set(gaps), key=gaps.index)


def _missing_full_fields(raw: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    torque_speed = raw.get("torque_speed_breakpoints")
    if torque_speed is None or (isinstance(torque_speed, list) and not torque_speed):
        gaps.append("torque_speed_breakpoints")
    elif isinstance(torque_speed, list):
        for axis, table in enumerate(torque_speed, start=1):
            if table is None or (isinstance(table, list) and not table):
                gaps.append(f"torque_speed_breakpoints.axis_{axis}")
                continue
            if not isinstance(table, list | tuple):
                gaps.append(f"torque_speed_breakpoints.axis_{axis}")
                continue
            for point_index, point in enumerate(table, start=1):
                if not isinstance(point, list | tuple) or len(point) != 2 or _is_missing(point[0]) or _is_missing(point[1]):
                    gaps.append(f"torque_speed_breakpoints.axis_{axis}.point_{point_index}")
    else:
        gaps.append("torque_speed_breakpoints")

    friction = raw.get("friction")
    motor = raw.get("motor")
    for name in ("coulomb", "viscous"):
        if not isinstance(friction, dict) or _is_missing(friction.get(name)):
            gaps.append(f"friction.{name}")
    for name in ("gear_ratio", "torque_constant", "stator_resistance"):
        if not isinstance(motor, dict) or _is_missing(motor.get(name)):
            gaps.append(f"motor.{name}")
    return gaps


def _dof_gaps(raw: dict[str, Any], dof: int) -> list[str]:
    gaps: list[str] = []
    for key in ("q_dot_abs", "q_ddot_abs", "q_jerk_abs", "tau_abs", "tau_rate_abs"):
        if key in raw and isinstance(raw[key], list) and len(raw[key]) != dof:
            gaps.append(f"{key}.dof")
    for _, bounds_key in AXIS_LIMIT_FIELDS:
        bounds = raw.get(bounds_key)
        if not isinstance(bounds, dict):
            continue
        for side in ("lower", "upper"):
            value = bounds.get(side)
            if isinstance(value, list) and len(value) != dof:
                gaps.append(f"{bounds_key}.{side}.dof")
    for parent, child in (("friction", "coulomb"), ("friction", "viscous"), ("motor", "gear_ratio"), ("motor", "torque_constant"), ("motor", "stator_resistance")):
        value = raw.get(parent, {}).get(child) if isinstance(raw.get(parent), dict) else None
        if isinstance(value, list) and len(value) != dof:
            gaps.append(f"{parent}.{child}.dof")
    torque_speed = raw.get("torque_speed_breakpoints")
    if isinstance(torque_speed, list) and len(torque_speed) != dof:
        gaps.append("torque_speed_breakpoints.dof")
    return gaps


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().lower() in {"todo", "tbd", "null", "none"}
    if isinstance(value, list):
        return not value or any(_is_missing(item) for item in value)
    if isinstance(value, dict):
        return not value or any(_is_missing(item) for item in value.values())
    return False


def _has_axis_limit(raw: dict[str, Any], abs_key: str, bounds_key: str) -> bool:
    if not _is_missing(raw.get(abs_key)):
        return True
    bounds = raw.get(bounds_key)
    if not isinstance(bounds, dict):
        return False
    return not _is_missing(bounds.get("lower")) and not _is_missing(bounds.get("upper"))
