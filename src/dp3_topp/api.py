from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dp3_topp.cli import (
    ACTIVE_CONSTRAINT_THRESHOLD,
    _active_constraint_percentages,
    _config_summary,
    _constraint_check_result,
    _finite_or_none,
    _format_time_scale,
    _most_restrictive_constraint_percentages,
    _reproduction_sources,
    _required_time_scale,
    _time_scale_covers_violations,
    _write_constraint_utilizations,
    _write_constraint_violations,
    _write_quantities,
    _write_trajectory,
)
from dp3_topp.constraints import ConstraintAudit, ConstraintLimits
from dp3_topp.dynamics_mujoco import MujocoRobotDynamics
from dp3_topp.optimizer import (
    DP3Config,
    TrajectoryQuantities,
    TrajectoryResult,
    evaluate_trajectory_quantities,
    optimize_dp2,
    optimize_dp3,
    resample_trajectory_by_time,
)
from dp3_topp.path_data import PathData

PathLike = str | Path
PathInput = PathData | PathLike
LimitsInput = ConstraintLimits | PathLike
RobotInput = MujocoRobotDynamics | PathLike | None


@dataclass(frozen=True)
class DPToppRunResult:
    status: int
    result: TrajectoryResult
    quantities: TrajectoryQuantities
    check_result: TrajectoryResult
    check_quantities: TrajectoryQuantities
    check_audit: ConstraintAudit
    check_source: str
    summary: dict
    out_dir: Path | None = None
    summary_path: Path | None = None


def run_dp3(
    *,
    path: PathInput,
    limits: LimitsInput,
    config: DP3Config | None = None,
    robot: RobotInput = None,
    model: PathLike | None = None,
    out_dir: PathLike | None = None,
    constraint_check_points: int = 0,
    time_samples: int = 0,
) -> DPToppRunResult:
    """Run the DP3 optimizer as a library workflow."""
    return _run_method(
        method="dp3",
        path=path,
        limits=limits,
        config=config,
        robot=robot,
        model=model,
        out_dir=out_dir,
        constraint_check_points=constraint_check_points,
        time_samples=time_samples,
    )


def run_dp2(
    *,
    path: PathInput,
    limits: LimitsInput,
    config: DP3Config | None = None,
    robot: RobotInput = None,
    model: PathLike | None = None,
    out_dir: PathLike | None = None,
    jerk_limited: str = "no",
    constraint_check_points: int = 0,
    time_samples: int = 0,
) -> DPToppRunResult:
    """Run the DP2 baseline optimizer as a library workflow."""
    return _run_method(
        method="dp2",
        path=path,
        limits=limits,
        config=config,
        robot=robot,
        model=model,
        out_dir=out_dir,
        jerk_limited=jerk_limited,
        constraint_check_points=constraint_check_points,
        time_samples=time_samples,
    )


def _run_method(
    *,
    method: str,
    path: PathInput,
    limits: LimitsInput,
    config: DP3Config | None,
    robot: RobotInput,
    model: PathLike | None,
    out_dir: PathLike | None,
    jerk_limited: str | None = None,
    constraint_check_points: int,
    time_samples: int,
) -> DPToppRunResult:
    config = DP3Config() if config is None else config
    path_data, path_source = _coerce_path(path)
    limit_data, limits_source = _coerce_limits(limits)
    robot_data, model_source = _coerce_robot(robot=robot, model=model)
    if limit_data.dof != path_data.dof:
        raise ValueError(f"limits DOF ({limit_data.dof}) does not match path DOF ({path_data.dof})")
    if robot_data is not None:
        if robot_data.dof != path_data.dof:
            raise ValueError(f"MuJoCo model DOF ({robot_data.dof}) does not match path DOF ({path_data.dof})")
        robot_data.assert_joint_positions_in_range(path_data.q)

    start = time.perf_counter()
    if method == "dp2":
        result = optimize_dp2(
            path=path_data,
            limits=limit_data,
            config=config,
            robot=robot_data,
            jerk_limited=jerk_limited or "no",
        )
    elif method == "dp3":
        result = optimize_dp3(path=path_data, limits=limit_data, config=config, robot=robot_data)
    else:
        raise ValueError("method must be 'dp2' or 'dp3'")
    cpu_time_s = time.perf_counter() - start

    quantities = evaluate_trajectory_quantities(path=path_data, result=result, limits=limit_data, robot=robot_data)
    check_result, check_quantities, check_audit, check_source = _constraint_check_result(
        path=path_data,
        result=result,
        limits=limit_data,
        robot=robot_data,
        points=int(constraint_check_points),
    )
    feasible = bool(result.feasible and check_audit.ok)
    summary = _run_summary(
        result=result,
        check_result=check_result,
        check_audit=check_audit,
        check_quantities=check_quantities,
        check_source=check_source,
        feasible=feasible,
        cpu_time_s=cpu_time_s,
        config=config,
        constraint_check_points=int(constraint_check_points),
        time_samples=int(time_samples),
        path=path_data,
        limits=limit_data,
        robot=robot_data,
        path_source=path_source,
        limits_source=limits_source,
        model_source=model_source,
    )

    output_dir = None if out_dir is None else Path(out_dir)
    summary_path = None
    if output_dir is not None:
        summary_path = _write_outputs(
            out_dir=output_dir,
            path=path_data,
            limits=limit_data,
            robot=robot_data,
            result=result,
            quantities=quantities,
            check_result=check_result,
            check_quantities=check_quantities,
            check_audit=check_audit,
            summary=summary,
            time_samples=int(time_samples),
        )

    return DPToppRunResult(
        status=0 if feasible else 1,
        result=result,
        quantities=quantities,
        check_result=check_result,
        check_quantities=check_quantities,
        check_audit=check_audit,
        check_source=check_source,
        summary=summary,
        out_dir=output_dir,
        summary_path=summary_path,
    )


def _coerce_path(path: PathInput) -> tuple[PathData, Path | None]:
    if isinstance(path, PathData):
        return path, None
    source = Path(path)
    return PathData.from_csv(source), source


def _coerce_limits(limits: LimitsInput) -> tuple[ConstraintLimits, Path | None]:
    if isinstance(limits, ConstraintLimits):
        return limits, None
    source = Path(limits)
    return ConstraintLimits.from_yaml(source), source


def _coerce_robot(*, robot: RobotInput, model: PathLike | None) -> tuple[MujocoRobotDynamics | None, Path | None]:
    if robot is not None and model is not None:
        raise ValueError("provide either robot or model, not both")
    if isinstance(robot, MujocoRobotDynamics):
        return robot, None
    if robot is not None:
        source = Path(robot)
        return MujocoRobotDynamics.from_model_path(source), source
    if model is not None:
        source = Path(model)
        return MujocoRobotDynamics.from_model_path(source), source
    return None, None


def _run_summary(
    *,
    result: TrajectoryResult,
    check_result: TrajectoryResult,
    check_audit: ConstraintAudit,
    check_quantities: TrajectoryQuantities,
    check_source: str,
    feasible: bool,
    cpu_time_s: float,
    config: DP3Config,
    constraint_check_points: int,
    time_samples: int,
    path: PathData,
    limits: ConstraintLimits,
    robot: MujocoRobotDynamics | None,
    path_source: Path | None,
    limits_source: Path | None,
    model_source: Path | None,
) -> dict:
    scale = _required_time_scale(check_audit.max_utilization)
    has_timing = _finite_or_none(result.total_time) is not None and scale > 0.0
    has_scalable_timing = has_timing and _time_scale_covers_violations(check_audit.max_utilization)
    return {
        "method": result.method,
        "jerk_limited": result.jerk_limited,
        "feasible": feasible,
        "samples": int(result.s.size),
        "cpu_time_s": float(cpu_time_s),
        "total_time": _finite_or_none(result.total_time),
        "objective_cost": _finite_or_none(result.objective_cost),
        "objective_time_cost": _finite_or_none(result.objective_time_cost),
        "objective_drive_power_cost": _finite_or_none(result.objective_drive_power_cost),
        "segment_kinds": list(result.segment_kinds or []),
        "segment_count": len(result.segment_kinds or []),
        "constraint_check_points": int(constraint_check_points),
        "constraint_check_samples": int(check_result.s.size),
        "constraint_check_source": check_source,
        "constraint_utilization_csv": "constraint_utilization.csv" if check_result.s.size > 0 else None,
        "constraint_violations_csv": "constraint_violations.csv" if check_result.s.size > 0 else None,
        "time_samples": int(time_samples),
        "time_quantities_csv": "time_quantities.csv"
        if int(time_samples) > 0 and result.s.size > 0 and _finite_or_none(result.total_time) is not None
        else None,
        "max_utilization": check_audit.max_utilization,
        "active_constraint_threshold": ACTIVE_CONSTRAINT_THRESHOLD,
        "active_constraint_percent": _active_constraint_percentages(check_quantities),
        "most_restrictive_constraint_percent": _most_restrictive_constraint_percentages(check_quantities),
        "violations": [violation.__dict__ for violation in check_audit.violations],
        "required_time_scale_st": _format_time_scale(scale) if has_scalable_timing else "-",
        "executable_with_st": "yes" if has_scalable_timing else "no",
        "te_scale": _finite_or_none(result.total_time / scale) if has_scalable_timing else "-",
        "config": {
            **_config_summary(config),
            "constraint_check_points": int(constraint_check_points),
        },
        "reproduction_sources": _reproduction_sources(
            path=path,
            limits=limits,
            robot=robot,
            path_source=path_source,
            joint_path_csv=path_source,
            joint_path_source="path_csv" if path_source is not None else None,
            limits_source=limits_source,
            model_source=model_source,
        ),
    }


def _write_outputs(
    *,
    out_dir: Path,
    path: PathData,
    limits: ConstraintLimits,
    robot: MujocoRobotDynamics | None,
    result: TrajectoryResult,
    quantities: TrajectoryQuantities,
    check_result: TrajectoryResult,
    check_quantities: TrajectoryQuantities,
    check_audit: ConstraintAudit,
    summary: dict,
    time_samples: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_trajectory(out_dir / "trajectory.csv", result)
    _write_quantities(out_dir / "quantities.csv", result, quantities)
    if summary["time_quantities_csv"] is not None:
        time_result = resample_trajectory_by_time(
            result,
            np.linspace(0.0, float(result.total_time), max(2, int(time_samples))),
        )
        time_quantities = evaluate_trajectory_quantities(path=path, result=time_result, limits=limits, robot=robot)
        _write_quantities(out_dir / summary["time_quantities_csv"], time_result, time_quantities)
    if check_result.s.size > 0:
        _write_constraint_utilizations(out_dir / "constraint_utilization.csv", check_result, check_quantities)
        _write_constraint_violations(out_dir / "constraint_violations.csv", check_result, check_audit.violations)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary_path
