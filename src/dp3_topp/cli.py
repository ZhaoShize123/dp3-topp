from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from dp3_topp.cartesian_path import build_joint_path_result_from_cartesian, parse_pose_table
from dp3_topp.constraints import ConstraintLimits, audit_constraints
from dp3_topp.dynamics_mujoco import MujocoRobotDynamics
from dp3_topp.limits_schema import full_reproduction_gaps, write_t12a_limits_template
from dp3_topp.optimizer import (
    DP3Config,
    TrajectoryResult,
    evaluate_trajectory_quantities,
    optimize_dp2,
    optimize_dp3,
    resample_trajectory_by_segments,
    resample_trajectory_by_time,
)
from dp3_topp.path_data import PathData


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_CONSTRAINT_THRESHOLD = 0.99
CONSTRAINT_QUANTITY_NAMES = (
    "q_dot",
    "q_ddot",
    "q_jerk",
    "tau",
    "tau_rate",
    "mechanical_power",
)
TIME_SCALE_EXPONENTS = {
    "q_dot": 1.0,
    "q_ddot": 2.0,
    "q_jerk": 3.0,
    "tau_rate": 3.0,
    "mechanical_power": 1.0,
}
BATCH_METRIC_BASE_FIELDS = (
    "id",
    "method",
    "status",
    "feasible",
    "data_readiness",
    "error",
    "source_missing",
    "data_gaps",
    "csv",
    "path_index",
    "limits",
    "model",
    "dynamics_backend",
    "run_dir",
    "t_e_s",
    "t_cpu_s",
    "objective_cost",
    "objective_time_cost",
    "objective_drive_power_cost",
    "required_time_scale_st",
    "executable_with_st",
    "te_scale",
    "constraint_check_samples",
    "constraint_check_source",
    "violation_count",
    "constraint_utilization_csv",
    "constraint_violations_csv",
)


def default_dyn_dir() -> Path:
    return PROJECT_ROOT / "dyn - \u526f\u672c"


@dataclass(frozen=True)
class DataValidationReport:
    ok: bool
    missing: list[str]
    present: list[str]
    data_gaps: list[str]
    full_reproduction_ready: bool
    reproduction_sources: dict

    @property
    def data_readiness(self) -> str:
        if self.ok and self.full_reproduction_ready:
            return "ready"
        if self.missing:
            return "source_missing"
        return "data_gaps"

    def as_dict(self) -> dict:
        return {
            "ok": bool(self.ok),
            "data_readiness": self.data_readiness,
            "full_reproduction_ready": bool(self.full_reproduction_ready),
            "missing": list(self.missing),
            "present": list(self.present),
            "data_gaps": list(self.data_gaps),
            "reproduction_sources": dict(self.reproduction_sources),
        }


def validate_data(
    robot_dir: Path,
    paths_dir: Path,
    limits_path: Path | None = None,
    path_index: Path | None = None,
    expected_path_count: int | None = None,
    check_ik: bool = False,
    ee_body: str = "link_6",
    ee_site: str | None = None,
    ik_check_samples: int = 8,
    ik_max_iters: int = 4,
    ik_pos_tol: float = 1e-4,
    ik_ori_tol: float = 1e-3,
    ik_damping: float = 1e-4,
    ik_step_scale: float = 0.6,
    ik_orientation_weight: float = 0.1,
    tcp_offset: np.ndarray | None = None,
    require_ik_convergence: bool = False,
    z_start: float | None = None,
    z_end: float | None = None,
    z_max: float | None = None,
) -> DataValidationReport:
    robot_dir = Path(robot_dir)
    paths_dir = Path(paths_dir)
    limits_path = Path(limits_path) if limits_path is not None else robot_dir / "limits.yaml"
    path_index = Path(path_index) if path_index is not None else None

    required = [
        robot_dir / "T12A-14.xml",
        limits_path,
    ]
    optional_reference_files = [robot_dir / "T12A-14.urdf"]
    data_gaps = []
    path_csvs: list[Path] = []
    offline_traj_path = paths_dir / "Offline_Traj.txt"
    if path_index is None:
        required.append(offline_traj_path)
    else:
        required.append(path_index)
        if path_index.exists():
            try:
                entries = _load_path_index(path_index)
            except ValueError as exc:
                data_gaps.append(f"path_index: {exc}")
            else:
                data_gaps.extend(_path_index_template_gaps(path_index))
                if expected_path_count is not None and len(entries) != int(expected_path_count):
                    data_gaps.append(f"path_index: contains {len(entries)} paths, expected {int(expected_path_count)}")
                path_csvs = [_resolve_index_path(path_index.parent, entry["csv"]) for entry in entries]
                required.extend(path_csvs)
    missing = [str(path) for path in required if not path.exists()]
    present = [str(path) for path in required if path.exists()]
    present.extend(str(path) for path in optional_reference_files if path.exists())
    model_path = robot_dir / "T12A-14.xml"
    validation_robot = None
    validation_limits = None
    if model_path.exists():
        validation_robot, model_gaps = _mujoco_model_for_validation(model_path, expected_dof=6)
        data_gaps.extend(model_gaps)
    if limits_path.exists():
        data_gaps.extend(full_reproduction_gaps(limits_path, expected_dof=6))
        try:
            validation_limits = ConstraintLimits.from_yaml(limits_path)
        except ValueError as exc:
            data_gaps.append(f"{limits_path}: {exc}")
        if validation_robot is not None and validation_limits is not None:
            try:
                _validate_q_position_within_mujoco_ranges(validation_limits, validation_robot)
            except ValueError as exc:
                data_gaps.append(f"{limits_path}: {exc}")
    if path_index is None and offline_traj_path.exists():
        data_gaps.extend(_offline_traj_gaps(offline_traj_path))
        if check_ik and model_path.exists():
            data_gaps.extend(
                _cartesian_ik_gaps(
                    model_path=model_path,
                    offline_traj_path=offline_traj_path,
                    ee_body=ee_body,
                    ee_site=ee_site,
                    samples=ik_check_samples,
                    max_iters=ik_max_iters,
                    pos_tol=ik_pos_tol,
                    ori_tol=ik_ori_tol,
                    damping=ik_damping,
                    step_scale=ik_step_scale,
                    orientation_weight=ik_orientation_weight,
                    tcp_offset=tcp_offset,
                    require_convergence=require_ik_convergence,
                )
            )
    if path_index is not None and path_csvs:
        existing_path_csvs = [path for path in path_csvs if path.exists()]
        data_gaps.extend(_path_csv_gaps(existing_path_csvs, expected_dof=6, robot=validation_robot))
        if validation_limits is not None and path_index.exists():
            try:
                entries = _load_path_index(path_index)
            except ValueError:
                entries = []
            data_gaps.extend(
                _path_index_endpoint_speed_gaps(
                    entries,
                    path_index.parent,
                    validation_limits,
                    default_z_start=z_start,
                    default_z_end=z_end,
                    default_z_max=z_max,
                )
            )
    reproduction_sources = _reproduction_sources(
        path_source=offline_traj_path if path_index is None else None,
        limits_source=limits_path,
        model_source=model_path,
        path_index_source=path_index,
    )
    if validation_limits is not None:
        reproduction_sources["limits_dof"] = int(validation_limits.dof)
    if validation_robot is not None:
        reproduction_sources["model_dof"] = int(validation_robot.dof)
    return DataValidationReport(
        ok=not missing and not data_gaps,
        missing=missing,
        present=present,
        data_gaps=data_gaps,
        full_reproduction_ready=not missing and not data_gaps,
        reproduction_sources=reproduction_sources,
    )


def validate_data_main(argv: list[str] | None = None) -> int:
    config_path, config = _load_cli_config(argv)
    model_default = _config_path(config, "model", config_path)
    input_default = _config_path(config, "input", config_path)
    robot_default = _config_path(config, "robot", config_path)
    if robot_default is None:
        robot_default = model_default.parent if model_default is not None else default_dyn_dir() / "models" / "T12A"
    paths_default = _config_path(config, "paths", config_path)
    if paths_default is None:
        paths_default = input_default.parent if input_default is not None else default_dyn_dir()

    parser = argparse.ArgumentParser(description="Validate DP3 reproduction data inputs.")
    parser.add_argument("--config", type=Path, default=config_path, help="Optional YAML run config to validate.")
    parser.add_argument("--robot", type=Path, default=robot_default)
    parser.add_argument("--paths", type=Path, default=paths_default)
    parser.add_argument("--limits", type=Path, default=_config_path(config, "limits", config_path))
    parser.add_argument("--path-index", type=Path, default=_config_path(config, "path_index", config_path))
    parser.add_argument("--expected-path-count", type=int, default=_optional_int(config.get("expected_path_count")))
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument(
        "--check-ik",
        action="store_true",
        default=_config_bool_choice(config, "check_ik", default=False) == "yes",
        help="Check T12A MuJoCo IK can build a joint path from Offline_Traj.txt.",
    )
    parser.add_argument("--ee-body", type=str, default=config.get("ee_body", "link_6"))
    parser.add_argument("--ee-site", type=str, default=config.get("ee_site"))
    parser.add_argument("--ik-check-samples", type=int, default=int(config.get("ik_check_samples", 8)))
    parser.add_argument("--ik-max-iters", type=int, default=int(config.get("ik_max_iters", 4)))
    parser.add_argument("--ik-pos-tol", type=float, default=float(config.get("ik_pos_tol", 1e-4)))
    parser.add_argument("--ik-ori-tol", type=float, default=float(config.get("ik_ori_tol", 1e-3)))
    parser.add_argument("--ik-damping", type=float, default=float(config.get("ik_damping", 1e-4)))
    parser.add_argument("--ik-step-scale", type=float, default=float(config.get("ik_step_scale", 0.6)))
    parser.add_argument("--ik-orientation-weight", type=float, default=float(config.get("ik_orientation_weight", 0.1)))
    parser.add_argument("--tcp-offset-x", type=float, default=float(config.get("tcp_offset_x", 0.0)))
    parser.add_argument("--tcp-offset-y", type=float, default=float(config.get("tcp_offset_y", 0.0)))
    parser.add_argument("--tcp-offset-z", type=float, default=float(config.get("tcp_offset_z", 0.0)))
    parser.add_argument("--z-start", type=float, default=float(config.get("z_start", 0.0)))
    parser.add_argument("--z-end", type=float, default=float(config.get("z_end", 0.0)))
    parser.add_argument("--z-max", type=float, default=_optional_float(config.get("z_max")))
    parser.add_argument(
        "--require-ik-convergence",
        choices=("yes", "no"),
        default=_config_bool_choice(config, "require_ik_convergence", default=False),
    )
    args = parser.parse_args(argv)
    report = validate_data(
        args.robot,
        args.paths,
        args.limits,
        args.path_index,
        expected_path_count=args.expected_path_count,
        check_ik=bool(args.check_ik),
        ee_body=args.ee_body,
        ee_site=args.ee_site,
        ik_check_samples=int(args.ik_check_samples),
        ik_max_iters=int(args.ik_max_iters),
        ik_pos_tol=float(args.ik_pos_tol),
        ik_ori_tol=float(args.ik_ori_tol),
        ik_damping=float(args.ik_damping),
        ik_step_scale=float(args.ik_step_scale),
        ik_orientation_weight=float(args.ik_orientation_weight),
        tcp_offset=np.array([args.tcp_offset_x, args.tcp_offset_y, args.tcp_offset_z], dtype=np.float64),
        require_ik_convergence=_yes_no(args.require_ik_convergence),
        z_start=float(args.z_start),
        z_end=float(args.z_end),
        z_max=args.z_max,
    )
    if report.present:
        print("present:")
        for item in report.present:
            print(f"  {item}")
    if report.missing:
        print("missing:")
        for item in report.missing:
            print(f"  {item}")
    if report.data_gaps:
        print("data gaps:")
        for item in report.data_gaps:
            print(f"  {item}")
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if report.ok else 2


def write_limits_template_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a T12A limits.yaml template for full paper reproduction.")
    parser.add_argument("--out", type=Path, default=default_dyn_dir() / "models" / "T12A" / "limits.template.yaml")
    parser.add_argument("--dof", type=int, default=6)
    parser.add_argument("--model", type=Path, default=default_dyn_dir() / "models" / "T12A" / "T12A-14.xml")
    args = parser.parse_args(argv)
    write_t12a_limits_template(args.out, dof=args.dof, model_path=args.model)
    print(args.out)
    return 0


def write_path_index_template_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a 14-path index template for full paper reproduction.")
    parser.add_argument("--out", type=Path, default=default_dyn_dir() / "paths" / "path_index.template.yaml")
    parser.add_argument("--count", type=int, default=14)
    args = parser.parse_args(argv)
    if int(args.count) <= 0:
        raise ValueError("count must be positive")
    paths = [
        {
            "id": f"path_{index:02d}",
            "csv": f"path_{index:02d}.csv",
            "zs": 0.0,
            "ze": 0.0,
        }
        for index in range(1, int(args.count) + 1)
    ]
    data = {
        "template": True,
        "expected_path_count": int(args.count),
        "paths": paths,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")
    print(args.out)
    return 0


def build_path_main(argv: list[str] | None = None) -> int:
    config_path, config = _load_cli_config(argv)
    out_default = _config_path(config, "joint_path_out", config_path)
    parser = argparse.ArgumentParser(description="Build a DP3/TOPPRA joint path CSV from a Cartesian pose table.")
    parser.add_argument("--config", type=Path, default=config_path, help="Optional YAML build-path config.")
    parser.add_argument("--input", type=Path, default=_config_path(config, "input", config_path) or default_dyn_dir() / "Offline_Traj.txt", help="Cartesian pose table.")
    parser.add_argument("--model", type=Path, default=_config_path(config, "model", config_path) or default_dyn_dir() / "models" / "T12A" / "T12A-14.xml", help="T12A MuJoCo XML.")
    parser.add_argument("--ee-body", type=str, default=config.get("ee_body", "link_6"))
    parser.add_argument("--ee-site", type=str, default=config.get("ee_site"), help="Optional MuJoCo site to use as the end-effector target.")
    parser.add_argument("--out", type=Path, required=out_default is None, default=out_default, help="Output joint path CSV.")
    parser.add_argument("--summary-out", type=Path, default=_config_path(config, "summary_out", config_path), help="Optional JSON summary with IK diagnostics.")
    parser.add_argument("--ik-max-iters", type=int, default=int(config.get("ik_max_iters", 200)))
    parser.add_argument("--ik-pos-tol", type=float, default=float(config.get("ik_pos_tol", 1e-4)))
    parser.add_argument("--ik-ori-tol", type=float, default=float(config.get("ik_ori_tol", 1e-3)))
    parser.add_argument("--ik-damping", type=float, default=float(config.get("ik_damping", 1e-4)))
    parser.add_argument("--ik-step-scale", type=float, default=float(config.get("ik_step_scale", 0.6)))
    parser.add_argument("--ik-orientation-weight", type=float, default=float(config.get("ik_orientation_weight", 0.1)))
    parser.add_argument("--tcp-offset-x", type=float, default=float(config.get("tcp_offset_x", 0.0)))
    parser.add_argument("--tcp-offset-y", type=float, default=float(config.get("tcp_offset_y", 0.0)))
    parser.add_argument("--tcp-offset-z", type=float, default=float(config.get("tcp_offset_z", 0.0)))
    parser.add_argument(
        "--require-ik-convergence",
        choices=("yes", "no"),
        default=_config_bool_choice(config, "require_ik_convergence", default=False),
    )
    args = parser.parse_args(argv)
    try:
        robot = MujocoRobotDynamics.from_model_path(args.model)
        path_result = build_joint_path_result_from_cartesian(
            robot=robot,
            pose_table=parse_pose_table(args.input),
            ee_body=args.ee_body,
            ee_site=args.ee_site,
            max_iters=int(args.ik_max_iters),
            pos_tol=float(args.ik_pos_tol),
            ori_tol=float(args.ik_ori_tol),
            damping=float(args.ik_damping),
            step_scale=float(args.ik_step_scale),
            orientation_weight=float(args.ik_orientation_weight),
            tcp_offset=np.array([args.tcp_offset_x, args.tcp_offset_y, args.tcp_offset_z], dtype=np.float64),
            require_convergence=_yes_no(args.require_ik_convergence),
        )
    except (OSError, ValueError) as exc:
        print(f"{args.input}: {exc}")
        return 2
    _write_path_csv(args.out, path_result.path)
    if args.summary_out is not None:
        summary = {
            "input": str(args.input),
            "model": str(args.model),
            "joint_path_csv": str(args.out),
            "samples": int(path_result.path.samples),
            "dof": int(path_result.path.dof),
            "s_end": float(path_result.path.s[-1]),
            "ik": path_result.ik.as_dict(),
            "reproduction_sources": _reproduction_sources(
                path=path_result.path,
                robot=robot,
                path_source=args.input,
                joint_path_csv=args.out,
                joint_path_source="generated_from_cartesian",
                model_source=args.model,
            ),
        }
        summary["ik"]["settings"] = _ik_settings_from_args(args)
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return 0


def run_main(argv: list[str] | None = None) -> int:
    config_path, config = _load_cli_config(argv)
    parser = argparse.ArgumentParser(description="Run a DP3 reproduction experiment.")
    parser.add_argument("--config", type=Path, default=config_path, help="YAML run config.")
    parser.add_argument("--method", choices=("dp2", "dp3", "compare"), default=config.get("method", "dp3"))
    parser.add_argument("--jerk-limited", choices=("no", "only start / end", "yes"), default=config.get("jerk_limited"))
    parser.add_argument("--path-csv", type=Path, default=_config_path(config, "path_csv", config_path), help="Joint path CSV with s,q,dq,d2q,d3q columns.")
    parser.add_argument("--path-index", type=Path, default=_config_path(config, "path_index", config_path), help="YAML list of joint path CSVs and per-path zs/ze values.")
    parser.add_argument("--expected-path-count", type=int, default=_optional_int(config.get("expected_path_count")))
    parser.add_argument("--joint-path-out", type=Path, default=_config_path(config, "joint_path_out", config_path), help="Optional CSV copy of the generated Cartesian joint path.")
    parser.add_argument("--input", type=Path, default=_config_path(config, "input", config_path) or default_dyn_dir() / "Offline_Traj.txt", help="Cartesian pose table.")
    parser.add_argument("--model", type=Path, default=_config_path(config, "model", config_path) or default_dyn_dir() / "models" / "T12A" / "T12A-14.xml", help="T12A MuJoCo XML.")
    parser.add_argument("--ee-body", type=str, default=config.get("ee_body", "link_6"))
    parser.add_argument("--ee-site", type=str, default=config.get("ee_site"))
    parser.add_argument("--limits", type=Path, default=_config_path(config, "limits", config_path), help="Limits YAML.")
    parser.add_argument("--out-dir", type=Path, default=_config_path(config, "out_dir", config_path) or Path("outputs/dp3-run"))
    parser.add_argument("--ns", type=int, default=config.get("ns", 40))
    parser.add_argument("--nz", type=int, default=config.get("nz", 500))
    parser.add_argument("--nch", type=int, default=config.get("nch", 20))
    parser.add_argument("--constraint-check-points", type=int, default=int(config.get("constraint_check_points", 0)))
    parser.add_argument("--time-samples", type=int, default=int(config.get("time_samples", 0)))
    parser.add_argument(
        "--require-full-reproduction-data",
        choices=("yes", "no"),
        default=_config_bool_choice(config, "require_full_reproduction_data", default=False),
    )
    parser.add_argument("--k1", type=float, default=float(config.get("k1", 1.0)))
    parser.add_argument("--k2", type=float, default=float(config.get("k2", 0.0)))
    parser.add_argument("--z-start", type=float, default=float(config.get("z_start", 0.0)))
    parser.add_argument("--z-end", type=float, default=float(config.get("z_end", 0.0)))
    parser.add_argument("--z-max", type=float, default=_optional_float(config.get("z_max")))
    parser.add_argument("--tau-rate-dt", type=float, default=float(config.get("tau_rate_dt", 1e-5)))
    parser.add_argument("--ik-max-iters", type=int, default=int(config.get("ik_max_iters", 200)))
    parser.add_argument("--ik-pos-tol", type=float, default=float(config.get("ik_pos_tol", 1e-4)))
    parser.add_argument("--ik-ori-tol", type=float, default=float(config.get("ik_ori_tol", 1e-3)))
    parser.add_argument("--ik-damping", type=float, default=float(config.get("ik_damping", 1e-4)))
    parser.add_argument("--ik-step-scale", type=float, default=float(config.get("ik_step_scale", 0.6)))
    parser.add_argument("--ik-orientation-weight", type=float, default=float(config.get("ik_orientation_weight", 0.1)))
    parser.add_argument("--tcp-offset-x", type=float, default=float(config.get("tcp_offset_x", 0.0)))
    parser.add_argument("--tcp-offset-y", type=float, default=float(config.get("tcp_offset_y", 0.0)))
    parser.add_argument("--tcp-offset-z", type=float, default=float(config.get("tcp_offset_z", 0.0)))
    parser.add_argument(
        "--require-ik-convergence",
        choices=("yes", "no"),
        default=_config_bool_choice(config, "require_ik_convergence", default=False),
    )
    args = parser.parse_args(argv)
    require_full_reproduction_data = _yes_no(args.require_full_reproduction_data)
    args.model_explicit = (
        _config_path(config, "model", config_path) is not None
        or _argv_has_option(argv, "--model")
        or require_full_reproduction_data
    )

    source_missing = _run_source_missing(args)
    if source_missing:
        if args.path_index is not None:
            error = "missing required sources: " + ", ".join(str(item) for item in source_missing)
            print(error)
            if args.method == "compare":
                _write_failed_comparison_batch_summary(
                    out_dir=args.out_dir,
                    path_index=args.path_index,
                    error=error,
                    source_missing=source_missing,
                    expected_path_count=args.expected_path_count,
                    limits_source=args.limits,
                    model_source=args.model if args.model_explicit else None,
                )
            else:
                _write_failed_batch_summary(
                    out_dir=args.out_dir,
                    path_index=args.path_index,
                    method=args.method,
                    error=error,
                    expected_path_count=args.expected_path_count,
                    source_missing=source_missing,
                    limits_source=args.limits,
                    model_source=args.model if args.model_explicit else None,
                )
            return 2
        error = (
            "missing required limits file: pass --limits or set limits in --config"
            if args.limits is None
            else f"missing limits file: {args.limits}"
        )
        print(error)
        _write_failed_run_summary(
            out_dir=args.out_dir,
            run_id=_input_run_id(args),
            csv_path=args.path_csv or args.input,
            config=_dp3_config_from_args(args),
            method=args.method,
            jerk_limited=args.jerk_limited,
            error=error,
            source_missing=source_missing,
            limits_source=args.limits,
            model_source=args.model if args.model_explicit else None,
        )
        return 2
    if args.path_index is not None and args.path_index.exists():
        template_gaps = _path_index_template_gaps(args.path_index)
        if template_gaps:
            error = "; ".join(template_gaps)
            print(error)
            if args.method == "compare":
                return _write_failed_comparison_batch_summary(
                    out_dir=args.out_dir,
                    path_index=args.path_index,
                    error=error,
                    data_gaps=template_gaps,
                    expected_path_count=args.expected_path_count,
                    limits_source=args.limits,
                    model_source=args.model if args.model_explicit else None,
                )
            return _write_failed_batch_summary(
                out_dir=args.out_dir,
                path_index=args.path_index,
                method=args.method,
                error=error,
                data_gaps=template_gaps,
                expected_path_count=args.expected_path_count,
                limits_source=args.limits,
                model_source=args.model if args.model_explicit else None,
            )
    if args.limits is None:
        error = "missing required limits file: pass --limits or set limits in --config"
        print(error)
        _write_failed_run_summary(
            out_dir=args.out_dir,
            run_id=_input_run_id(args),
            csv_path=args.path_csv or args.input,
            config=_dp3_config_from_args(args),
            method=args.method,
            jerk_limited=args.jerk_limited,
            error=error,
            source_missing=["limits"],
            limits_source=args.limits,
            model_source=args.model if args.model_explicit else None,
        )
        return 2
    if not args.limits.exists():
        error = f"missing limits file: {args.limits}"
        print(error)
        _write_failed_run_summary(
            out_dir=args.out_dir,
            run_id=_input_run_id(args),
            csv_path=args.path_csv or args.input,
            config=_dp3_config_from_args(args),
            method=args.method,
            jerk_limited=args.jerk_limited,
            error=error,
            source_missing=[args.limits],
            limits_source=args.limits,
            model_source=args.model if args.model_explicit else None,
        )
        return 2
    if require_full_reproduction_data:
        gaps = full_reproduction_gaps(args.limits, expected_dof=6)
        if gaps:
            error = f"incomplete full reproduction limits: {', '.join(gaps)}"
            print(error)
            if args.path_index is not None:
                if args.method == "compare":
                    return _write_failed_comparison_batch_summary(
                        out_dir=args.out_dir,
                        path_index=args.path_index,
                        error=error,
                        data_gaps=gaps,
                        expected_path_count=args.expected_path_count,
                        limits_source=args.limits,
                        model_source=args.model if args.model_explicit else None,
                    )
                return _write_failed_batch_summary(
                    out_dir=args.out_dir,
                    path_index=args.path_index,
                    method=args.method,
                    error=error,
                    data_gaps=gaps,
                    expected_path_count=args.expected_path_count,
                    limits_source=args.limits,
                    model_source=args.model if args.model_explicit else None,
                )
            _write_failed_run_summary(
                out_dir=args.out_dir,
                run_id=_input_run_id(args),
                csv_path=args.path_csv or args.input,
                config=_dp3_config_from_args(args),
                method=args.method,
                jerk_limited=args.jerk_limited,
                error=error,
                data_gaps=gaps,
                limits_source=args.limits,
                model_source=args.model if args.model_explicit else None,
            )
            return 2
    try:
        limits = ConstraintLimits.from_yaml(args.limits)
    except ValueError as exc:
        error = f"{args.limits}: {exc}"
        print(error)
        if args.path_index is not None:
            if args.method == "compare":
                return _write_failed_comparison_batch_summary(
                    out_dir=args.out_dir,
                    path_index=args.path_index,
                    error=error,
                    data_gaps=[error],
                    expected_path_count=args.expected_path_count,
                    limits_source=args.limits,
                    model_source=args.model if args.model_explicit else None,
                )
            return _write_failed_batch_summary(
                out_dir=args.out_dir,
                path_index=args.path_index,
                method=args.method,
                error=error,
                data_gaps=[error],
                expected_path_count=args.expected_path_count,
                limits_source=args.limits,
                model_source=args.model if args.model_explicit else None,
            )
        _write_failed_run_summary(
            out_dir=args.out_dir,
            run_id=(args.path_csv.stem if args.path_csv is not None else "run"),
            csv_path=args.path_csv or args.input,
            config=_dp3_config_from_args(args),
            method=args.method,
            jerk_limited=args.jerk_limited,
            error=str(exc),
            data_gaps=[str(exc)],
            limits_source=args.limits,
            model_source=args.model if args.model_explicit else None,
        )
        return 2

    if args.path_index is not None:
        if args.method == "compare":
            return _run_path_index_comparison(args, limits)
        return _run_path_index(args, limits)

    robot = None
    ik_summary = None
    joint_path_csv = None
    joint_path_source = None
    if args.path_csv is not None:
        try:
            path = PathData.from_csv(args.path_csv)
            joint_path_csv = args.path_csv
            joint_path_source = "path_csv"
            if args.model_explicit:
                if not args.model.exists():
                    raise FileNotFoundError(args.model)
                robot = MujocoRobotDynamics.from_model_path(args.model)
                if robot.dof != limits.dof:
                    raise ValueError(f"MuJoCo model DOF ({robot.dof}) does not match limits DOF ({limits.dof})")
            if robot is not None and robot.dof != path.dof:
                raise ValueError(f"MuJoCo model DOF ({robot.dof}) does not match path DOF ({path.dof})")
            if robot is not None:
                robot.assert_joint_positions_in_range(path.q)
        except (OSError, ValueError) as exc:
            print(f"{args.path_csv}: {exc}")
            _write_failed_run_summary(
                out_dir=args.out_dir,
                run_id=args.path_csv.stem or "path_csv",
                csv_path=args.path_csv,
                config=_dp3_config_from_args(args),
                method=args.method,
                jerk_limited=args.jerk_limited,
                error=str(exc),
                limits_source=args.limits,
                model_source=args.model if args.model_explicit else None,
            )
            return 2
    else:
        if not args.input.exists():
            error = f"missing input path: {args.input}"
            print(error)
            _write_failed_run_summary(
                out_dir=args.out_dir,
                run_id=args.input.stem or "cartesian_input",
                csv_path=args.input,
                config=_dp3_config_from_args(args),
                method=args.method,
                jerk_limited=args.jerk_limited,
                error=error,
                source_missing=[args.input],
                limits_source=args.limits,
                model_source=args.model,
            )
            return 2
        if not args.model.exists():
            error = f"missing MuJoCo model: {args.model}"
            print(error)
            _write_failed_run_summary(
                out_dir=args.out_dir,
                run_id=args.input.stem or "cartesian_input",
                csv_path=args.input,
                config=_dp3_config_from_args(args),
                method=args.method,
                jerk_limited=args.jerk_limited,
                error=error,
                source_missing=[args.model],
                limits_source=args.limits,
                model_source=args.model,
            )
            return 2
        try:
            robot = MujocoRobotDynamics.from_model_path(args.model)
            path_result = build_joint_path_result_from_cartesian(
                robot=robot,
                pose_table=parse_pose_table(args.input),
                ee_body=args.ee_body,
                ee_site=args.ee_site,
                max_iters=int(args.ik_max_iters),
                pos_tol=float(args.ik_pos_tol),
                ori_tol=float(args.ik_ori_tol),
                damping=float(args.ik_damping),
                step_scale=float(args.ik_step_scale),
                orientation_weight=float(args.ik_orientation_weight),
                tcp_offset=np.array([args.tcp_offset_x, args.tcp_offset_y, args.tcp_offset_z], dtype=np.float64),
                require_convergence=_yes_no(args.require_ik_convergence),
            )
            path = path_result.path
            ik_summary = path_result.ik.as_dict()
            ik_summary["settings"] = _ik_settings_from_args(args)
            joint_path_source = "generated_from_cartesian"
            if args.joint_path_out is not None:
                _write_path_csv(args.joint_path_out, path)
                joint_path_csv = args.joint_path_out
        except (OSError, ValueError) as exc:
            print(f"{args.input}: {exc}")
            _write_failed_run_summary(
                out_dir=args.out_dir,
                run_id=args.input.stem or "cartesian_input",
                csv_path=args.input,
                config=_dp3_config_from_args(args),
                method=args.method,
                jerk_limited=args.jerk_limited,
                error=str(exc),
                limits_source=args.limits,
                model_source=args.model,
            )
            return 2
    try:
        runtime_limits = _limits_with_mujoco_position_bounds(limits, robot)
    except ValueError as exc:
        print(str(exc))
        _write_failed_run_summary(
            out_dir=args.out_dir,
            run_id=_input_run_id(args),
            csv_path=args.path_csv or args.input,
            config=_dp3_config_from_args(args),
            method=args.method,
            jerk_limited=args.jerk_limited,
            error=str(exc),
            limits_source=args.limits,
            model_source=args.model if robot is not None else None,
        )
        return 2
    config_obj = _dp3_config_from_args(args)
    if args.method == "compare":
        return _run_comparison(
            path=path,
            limits=runtime_limits,
            config=config_obj,
            out_dir=args.out_dir,
            robot=robot,
            path_source=args.path_csv if args.path_csv is not None else args.input,
            ik_summary=ik_summary,
            joint_path_csv=joint_path_csv,
            joint_path_source=joint_path_source,
            limits_source=args.limits,
            model_source=args.model if robot is not None else None,
            constraint_check_points=int(args.constraint_check_points),
            time_samples=int(args.time_samples),
            dp2_jerk_limited=args.jerk_limited or "no",
        )
    return _run_single_path(
        path=path,
        limits=runtime_limits,
        config=config_obj,
        method=args.method,
        jerk_limited=args.jerk_limited,
        out_dir=args.out_dir,
        robot=robot,
        path_source=args.path_csv if args.path_csv is not None else args.input,
        ik_summary=ik_summary,
        joint_path_csv=joint_path_csv,
        joint_path_source=joint_path_source,
        limits_source=args.limits,
        model_source=args.model if robot is not None else None,
        constraint_check_points=int(args.constraint_check_points),
        time_samples=int(args.time_samples),
    )


def _run_comparison(
    *,
    path: PathData,
    limits: ConstraintLimits,
    config: DP3Config,
    out_dir: Path,
    robot: MujocoRobotDynamics | None = None,
    path_source: Path | None = None,
    ik_summary: dict | None = None,
    joint_path_csv: Path | None = None,
    joint_path_source: str | None = None,
    limits_source: Path | None = None,
    model_source: Path | None = None,
    path_index_source: Path | None = None,
    constraint_check_points: int = 0,
    time_samples: int = 0,
    dp2_jerk_limited: str = "no",
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    status_dp3 = _run_single_path(
        path=path,
        limits=limits,
        config=config,
        method="dp3",
        jerk_limited=None,
        out_dir=out_dir / "dp3",
        robot=robot,
        path_source=path_source,
        ik_summary=ik_summary,
        joint_path_csv=joint_path_csv,
        joint_path_source=joint_path_source,
        limits_source=limits_source,
        model_source=model_source,
        path_index_source=path_index_source,
        constraint_check_points=int(constraint_check_points),
        time_samples=int(time_samples),
    )
    status_dp2 = _run_single_path(
        path=path,
        limits=limits,
        config=config,
        method="dp2",
        jerk_limited=dp2_jerk_limited,
        out_dir=out_dir / "dp2",
        robot=robot,
        path_source=path_source,
        ik_summary=ik_summary,
        joint_path_csv=joint_path_csv,
        joint_path_source=joint_path_source,
        limits_source=limits_source,
        model_source=model_source,
        path_index_source=path_index_source,
        constraint_check_points=int(constraint_check_points),
        time_samples=int(time_samples),
    )
    dp3_summary_path = out_dir / "dp3" / "summary.json"
    dp2_summary_path = out_dir / "dp2" / "summary.json"
    dp3_summary = json.loads(dp3_summary_path.read_text(encoding="utf-8"))
    dp2_summary = json.loads(dp2_summary_path.read_text(encoding="utf-8"))
    dp3_entry = _comparison_run_entry(status_dp3, out_dir / "dp3", dp3_summary_path, dp3_summary)
    dp2_entry = _comparison_run_entry(status_dp2, out_dir / "dp2", dp2_summary_path, dp2_summary)
    comparison_metrics_csv = "comparison_metrics.csv"
    _write_single_comparison_metrics(
        out_dir / comparison_metrics_csv,
        run_id=path_source.stem if path_source is not None and path_source.stem else "run",
        dp3_run=dp3_entry,
        dp2_run=dp2_entry,
    )
    comparison = {
        "method": "compare",
        "feasible": bool(status_dp3 == 0 and status_dp2 == 0),
        "comparison_metrics_csv": comparison_metrics_csv,
        "runs": {
            "dp3": dp3_entry,
            "dp2": dp2_entry,
        },
        "delta": {
            "total_time_dp2_minus_dp3": _numeric_delta_for_statuses(
                status_dp2, status_dp3, dp2_summary.get("total_time"), dp3_summary.get("total_time")
            ),
            "te_scale_dp2_minus_dp3": _numeric_delta_for_statuses(
                status_dp2, status_dp3, dp2_summary.get("te_scale"), dp3_summary.get("te_scale")
            ),
            "cpu_time_dp2_minus_dp3": _numeric_delta_for_statuses(
                status_dp2, status_dp3, dp2_summary.get("cpu_time_s"), dp3_summary.get("cpu_time_s")
            ),
            "violation_count_dp2_minus_dp3": _count_delta_for_statuses(
                status_dp2,
                status_dp3,
                len(dp2_summary.get("violations", [])),
                len(dp3_summary.get("violations", [])),
            ),
            "max_utilization_dp2_minus_dp3": _max_utilization_delta_for_statuses(
                status_dp2,
                status_dp3,
                dp2_summary.get("max_utilization", {}),
                dp3_summary.get("max_utilization", {}),
            ),
            "active_constraint_percent_dp2_minus_dp3": _metric_map_delta_for_statuses(
                status_dp2,
                status_dp3,
                dp2_summary.get("active_constraint_percent", {}),
                dp3_summary.get("active_constraint_percent", {}),
            ),
            "most_restrictive_constraint_percent_dp2_minus_dp3": _metric_map_delta_for_statuses(
                status_dp2,
                status_dp3,
                dp2_summary.get("most_restrictive_constraint_percent", {}),
                dp3_summary.get("most_restrictive_constraint_percent", {}),
            ),
            "objective_cost_dp2_minus_dp3": _numeric_delta_for_statuses(
                status_dp2, status_dp3, dp2_summary.get("objective_cost"), dp3_summary.get("objective_cost")
            ),
            "objective_time_cost_dp2_minus_dp3": _numeric_delta_for_statuses(
                status_dp2, status_dp3, dp2_summary.get("objective_time_cost"), dp3_summary.get("objective_time_cost")
            ),
            "objective_drive_power_cost_dp2_minus_dp3": _numeric_delta_for_statuses(
                status_dp2,
                status_dp3,
                dp2_summary.get("objective_drive_power_cost"), dp3_summary.get("objective_drive_power_cost")
            ),
        },
        "config": {
            **_config_summary(config),
            "constraint_check_points": int(constraint_check_points),
            "time_samples": int(time_samples),
        },
        "reproduction_sources": _reproduction_sources(
            path=path,
            limits=limits,
            robot=robot,
            path_source=path_source,
            joint_path_csv=joint_path_csv,
            joint_path_source=joint_path_source,
            limits_source=limits_source,
            model_source=model_source,
            path_index_source=path_index_source,
        ),
    }
    if path_source is not None:
        comparison["path_source"] = str(path_source)
    if joint_path_csv is not None:
        comparison["joint_path_csv"] = str(joint_path_csv)
    (out_dir / "comparison_summary.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if status_dp3 == 2 or status_dp2 == 2:
        return 2
    return 0 if status_dp3 == 0 and status_dp2 == 0 else 1


def _comparison_run_entry(status: int, run_dir: Path, summary_path: Path, summary: dict) -> dict:
    return {
        "method": summary.get("method"),
        "status": int(status),
        "feasible": bool(summary.get("feasible", False)),
        "run_dir": str(run_dir),
        "summary_json": str(summary_path),
        "reproduction_sources": summary.get("reproduction_sources", {}),
        "total_time": _finite_or_none(summary.get("total_time")),
        "cpu_time_s": _finite_or_none(summary.get("cpu_time_s")),
        "objective_cost": _finite_or_none(summary.get("objective_cost")),
        "objective_time_cost": _finite_or_none(summary.get("objective_time_cost")),
        "objective_drive_power_cost": _finite_or_none(summary.get("objective_drive_power_cost")),
        "required_time_scale_st": str(summary.get("required_time_scale_st", "-")),
        "executable_with_st": str(summary.get("executable_with_st", "no")),
        "te_scale": _finite_or_none(summary.get("te_scale")),
        "segment_count": int(summary.get("segment_count", 0)),
        "constraint_check_samples": int(summary.get("constraint_check_samples", 0)),
        "constraint_check_source": str(summary.get("constraint_check_source", "not_run")),
        "violation_count": len(summary.get("violations", [])),
        "max_utilization": dict(summary.get("max_utilization") or {}),
        "active_constraint_percent": dict(summary.get("active_constraint_percent") or {}),
        "most_restrictive_constraint_percent": dict(summary.get("most_restrictive_constraint_percent") or {}),
        "constraint_utilization_csv": summary.get("constraint_utilization_csv"),
        "constraint_violations_csv": summary.get("constraint_violations_csv"),
    }


def _numeric_delta(left, right) -> float | None:
    left_value = _finite_or_none(left)
    right_value = _finite_or_none(right)
    if left_value is None or right_value is None:
        return None
    return float(left_value - right_value)


def _numeric_delta_for_statuses(left_status, right_status, left, right) -> float | None:
    if _status_or_failure(left_status) != 0 or _status_or_failure(right_status) != 0:
        return None
    return _numeric_delta(left, right)


def _count_delta_for_statuses(left_status, right_status, left, right) -> int | None:
    if _status_or_failure(left_status) != 0 or _status_or_failure(right_status) != 0:
        return None
    try:
        return int(left) - int(right)
    except (TypeError, ValueError):
        return None


def _max_utilization_delta_for_statuses(
    left_status,
    right_status,
    left: dict | None,
    right: dict | None,
) -> dict[str, float] | None:
    return _metric_map_delta_for_statuses(left_status, right_status, left, right)


def _metric_map_delta_for_statuses(
    left_status,
    right_status,
    left: dict | None,
    right: dict | None,
) -> dict[str, float] | None:
    if _status_or_failure(left_status) != 0 or _status_or_failure(right_status) != 0:
        return None
    left_values = left or {}
    right_values = right or {}
    names = list(CONSTRAINT_QUANTITY_NAMES)
    for source in (left_values, right_values):
        for name in source:
            if name not in names:
                names.append(str(name))
    delta: dict[str, float] = {}
    for name in names:
        value = _numeric_delta(left_values.get(name), right_values.get(name))
        if value is not None:
            delta[name] = value
    return delta


def _status_or_failure(status) -> int:
    try:
        return int(status)
    except (TypeError, ValueError):
        return 2


def _run_single_path(
    *,
    path: PathData,
    limits: ConstraintLimits,
    config: DP3Config,
    method: str,
    jerk_limited: str | None,
    out_dir: Path,
    robot: MujocoRobotDynamics | None = None,
    path_id: str | None = None,
    path_source: Path | None = None,
    ik_summary: dict | None = None,
    joint_path_csv: Path | None = None,
    joint_path_source: str | None = None,
    limits_source: Path | None = None,
    model_source: Path | None = None,
    path_index_source: Path | None = None,
    constraint_check_points: int = 0,
    time_samples: int = 0,
) -> int:
    if limits.dof != path.dof:
        error = f"limits DOF ({limits.dof}) does not match path DOF ({path.dof})"
        print(error)
        _write_failed_run_summary(
            out_dir=out_dir,
            run_id=path_id or (path_source.stem if path_source is not None and path_source.stem else "run"),
            csv_path=path_source or joint_path_csv or Path("path"),
            config=config,
            method=method,
            jerk_limited=jerk_limited,
            error=error,
            path=path,
            limits=limits,
            robot=robot,
            limits_source=limits_source,
            model_source=model_source,
            path_index_source=path_index_source,
            joint_path_csv=joint_path_csv,
            joint_path_source=joint_path_source,
        )
        return 2
    if robot is not None and robot.dof != path.dof:
        error = f"MuJoCo model DOF ({robot.dof}) does not match path DOF ({path.dof})"
        print(error)
        _write_failed_run_summary(
            out_dir=out_dir,
            run_id=path_id or (path_source.stem if path_source is not None and path_source.stem else "run"),
            csv_path=path_source or joint_path_csv or Path("path"),
            config=config,
            method=method,
            jerk_limited=jerk_limited,
            error=error,
            path=path,
            limits=limits,
            robot=robot,
            limits_source=limits_source,
            model_source=model_source,
            path_index_source=path_index_source,
            joint_path_csv=joint_path_csv,
            joint_path_source=joint_path_source,
        )
        return 2
    optimizer_start = time.perf_counter()
    try:
        if method == "dp2":
            result = optimize_dp2(
                path=path,
                limits=limits,
                config=config,
                robot=robot,
                jerk_limited=jerk_limited or "no",
            )
        else:
            result = optimize_dp3(path=path, limits=limits, config=config, robot=robot)
    except ValueError as exc:
        print(str(exc))
        _write_failed_run_summary(
            out_dir=out_dir,
            run_id=path_id or (path_source.stem if path_source is not None and path_source.stem else "run"),
            csv_path=path_source or joint_path_csv or Path("path"),
            config=config,
            method=method,
            jerk_limited=jerk_limited,
            error=str(exc),
            path=path,
            limits=limits,
            robot=robot,
            limits_source=limits_source,
            model_source=model_source,
            path_index_source=path_index_source,
            joint_path_csv=joint_path_csv,
            joint_path_source=joint_path_source,
        )
        return 2
    cpu_time_s = time.perf_counter() - optimizer_start
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_trajectory(out_dir / "trajectory.csv", result)
    output_quantities = evaluate_trajectory_quantities(path=path, result=result, limits=limits, robot=robot)
    _write_quantities(out_dir / "quantities.csv", result, output_quantities)
    time_quantities_csv = None
    if int(time_samples) > 0 and result.s.size > 0 and _finite_or_none(result.total_time) is not None:
        time_result = resample_trajectory_by_time(result, np.linspace(0.0, float(result.total_time), max(2, int(time_samples))))
        time_quantities = evaluate_trajectory_quantities(path=path, result=time_result, limits=limits, robot=robot)
        time_quantities_csv = "time_quantities.csv"
        _write_quantities(out_dir / time_quantities_csv, time_result, time_quantities)
    check_result, check_quantities, check_audit, check_source = _constraint_check_result(
        path=path,
        result=result,
        limits=limits,
        robot=robot,
        points=int(constraint_check_points),
    )
    constraint_utilization_csv = None
    if check_result.s.size > 0:
        constraint_utilization_csv = "constraint_utilization.csv"
        _write_constraint_utilizations(out_dir / constraint_utilization_csv, check_result, check_quantities)
    constraint_violations_csv = None
    if check_result.s.size > 0:
        constraint_violations_csv = "constraint_violations.csv"
        _write_constraint_violations(out_dir / constraint_violations_csv, check_result, check_audit.violations)
    active_constraint_percent = _active_constraint_percentages(check_quantities)
    most_restrictive_constraint_percent = _most_restrictive_constraint_percentages(check_quantities)
    scale = _required_time_scale(check_audit.max_utilization)
    has_timing = _finite_or_none(result.total_time) is not None and scale > 0.0
    scale_covers_violations = _time_scale_covers_violations(check_audit.max_utilization)
    has_scalable_timing = has_timing and scale_covers_violations
    feasible = bool(result.feasible and check_audit.ok)
    summary = {
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
        "constraint_utilization_csv": constraint_utilization_csv,
        "constraint_violations_csv": constraint_violations_csv,
        "time_samples": int(time_samples),
        "time_quantities_csv": time_quantities_csv,
        "max_utilization": check_audit.max_utilization,
        "active_constraint_threshold": ACTIVE_CONSTRAINT_THRESHOLD,
        "active_constraint_percent": active_constraint_percent,
        "most_restrictive_constraint_percent": most_restrictive_constraint_percent,
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
            joint_path_csv=joint_path_csv,
            joint_path_source=joint_path_source,
            limits_source=limits_source,
            model_source=model_source,
            path_index_source=path_index_source,
        ),
    }
    if path_id is not None:
        summary["path_id"] = path_id
    if path_source is not None:
        summary["path_source"] = str(path_source)
    if ik_summary is not None:
        summary["ik"] = ik_summary
    if joint_path_source is not None:
        summary["joint_path_source"] = joint_path_source
    if joint_path_csv is not None:
        summary["joint_path_csv"] = str(joint_path_csv)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if feasible else 1


def _write_failed_batch_summary(
    *,
    out_dir: Path,
    path_index: Path,
    method: str,
    error: str,
    expected_path_count: int | None = None,
    source_missing: list[Path | str] | None = None,
    data_gaps: list[str] | None = None,
    limits_source: Path | None = None,
    model_source: Path | None = None,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_missing_items = [str(item) for item in (source_missing or [])]
    data_gap_items = list(data_gaps or [])
    batch_summary = {
        "path_index": str(path_index),
        "method": method,
        "path_count": 0,
        "status": "failed",
        "data_readiness": _failure_data_readiness(source_missing_items, data_gap_items),
        "feasible_count": 0,
        "infeasible_count": 0,
        "total_cpu_time_s": 0.0,
        "total_trajectory_time_s": 0.0,
        "total_te_scale_s": 0.0,
        "objective_cost_total": 0.0,
        "runs": [],
        "error": error,
        "source_missing": source_missing_items,
        "data_gaps": data_gap_items,
        "reproduction_sources": _reproduction_sources(
            limits_source=limits_source,
            model_source=model_source,
            path_index_source=path_index,
        ),
    }
    if expected_path_count is not None:
        batch_summary["expected_path_count"] = int(expected_path_count)
    (out_dir / "batch_summary.json").write_text(
        json.dumps(batch_summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 2


def _write_failed_comparison_batch_summary(
    *,
    out_dir: Path,
    path_index: Path,
    error: str,
    source_missing: list[Path | str] | None = None,
    data_gaps: list[str] | None = None,
    expected_path_count: int | None = None,
    limits_source: Path | None = None,
    model_source: Path | None = None,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_missing_items = [str(item) for item in (source_missing or [])]
    data_gap_items = list(data_gaps or [])
    comparison = {
        "method": "compare",
        "path_index": str(path_index),
        "path_count": 0,
        "status": "failed",
        "data_readiness": _failure_data_readiness(source_missing_items, data_gap_items),
        "comparison_metrics_csv": None,
        "runs": {
            "dp3": _failed_comparison_batch_run_entry("DP3", out_dir / "dp3"),
            "dp2": _failed_comparison_batch_run_entry("DP2", out_dir / "dp2"),
        },
        "delta": {
            "total_trajectory_time_dp2_minus_dp3": None,
            "total_te_scale_dp2_minus_dp3": None,
            "total_cpu_time_dp2_minus_dp3": None,
            "violation_count_total_dp2_minus_dp3": None,
            "max_utilization_dp2_minus_dp3": None,
            "objective_cost_total_dp2_minus_dp3": None,
            "objective_time_cost_total_dp2_minus_dp3": None,
            "objective_drive_power_cost_total_dp2_minus_dp3": None,
        },
        "path_deltas": [],
        "error": error,
        "source_missing": source_missing_items,
        "data_gaps": data_gap_items,
        "reproduction_sources": _reproduction_sources(
            limits_source=limits_source,
            model_source=model_source,
            path_index_source=path_index,
        ),
    }
    if expected_path_count is not None:
        comparison["expected_path_count"] = int(expected_path_count)
    (out_dir / "comparison_batch_summary.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 2


def _failure_data_readiness(source_missing: list[str], data_gaps: list[str]) -> str:
    if source_missing:
        return "source_missing"
    if data_gaps:
        return "data_gaps"
    return "failed"


def _comparison_data_readiness(status: str, source_missing: list[str], data_gaps: list[str]) -> str:
    if source_missing:
        return "source_missing"
    if data_gaps:
        return "data_gaps"
    if status == "ok":
        return "ready"
    return "failed"


def _reproduction_sources(
    *,
    path: PathData | None = None,
    limits: ConstraintLimits | None = None,
    robot: MujocoRobotDynamics | None = None,
    path_source: Path | None = None,
    joint_path_csv: Path | None = None,
    joint_path_source: str | None = None,
    limits_source: Path | None = None,
    model_source: Path | None = None,
    path_index_source: Path | None = None,
) -> dict:
    return {
        "path_source": str(path_source) if path_source is not None else None,
        "path_index": str(path_index_source) if path_index_source is not None else None,
        "joint_path_source": joint_path_source,
        "joint_path_csv": str(joint_path_csv) if joint_path_csv is not None else None,
        "limits": str(limits_source) if limits_source is not None else None,
        "model": str(model_source) if model_source is not None else None,
        "dynamics_backend": "mujoco" if robot is not None else "none",
        "path_dof": int(path.dof) if path is not None else None,
        "limits_dof": int(limits.dof) if limits is not None else None,
        "model_dof": int(robot.dof) if robot is not None else None,
    }


def _failed_comparison_batch_run_entry(method: str, run_dir: Path) -> dict:
    return {
        "method": method,
        "status": 2,
        "batch_status": "failed",
        "run_dir": str(run_dir),
        "batch_summary_json": None,
        "path_count": 0,
        "feasible_count": 0,
        "infeasible_count": 0,
        "total_cpu_time_s": None,
        "total_trajectory_time_s": None,
        "violation_count_total": None,
        "max_utilization": {},
        "total_te_scale_s": None,
        "objective_cost_total": None,
        "objective_time_cost_total": None,
        "objective_drive_power_cost_total": None,
    }


def _run_path_index_comparison(args, limits: ConstraintLimits) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dp3_out = args.out_dir / "dp3"
    dp2_out = args.out_dir / "dp2"
    dp3_args = _copy_args(args, method="dp3", jerk_limited=None, out_dir=dp3_out)
    dp2_args = _copy_args(args, method="dp2", jerk_limited=args.jerk_limited or "no", out_dir=dp2_out)
    status_dp3 = _run_path_index(dp3_args, limits)
    status_dp2 = _run_path_index(dp2_args, limits)
    dp3_summary_path = dp3_out / "batch_summary.json"
    dp2_summary_path = dp2_out / "batch_summary.json"
    dp3_summary = json.loads(dp3_summary_path.read_text(encoding="utf-8"))
    dp2_summary = json.loads(dp2_summary_path.read_text(encoding="utf-8"))
    comparison_metrics_csv = "comparison_metrics.csv"
    _write_comparison_metrics(args.out_dir / comparison_metrics_csv, dp3_summary, dp2_summary)
    comparison = {
        "method": "compare",
        "path_index": str(args.path_index),
        "path_count": int(max(int(dp3_summary.get("path_count", 0)), int(dp2_summary.get("path_count", 0)))),
        "status": "ok" if status_dp3 == 0 and status_dp2 == 0 else "failed",
        "comparison_metrics_csv": comparison_metrics_csv,
        "runs": {
            "dp3": _comparison_batch_entry(status_dp3, dp3_out, dp3_summary_path, dp3_summary),
            "dp2": _comparison_batch_entry(status_dp2, dp2_out, dp2_summary_path, dp2_summary),
        },
        "delta": {
            "total_trajectory_time_dp2_minus_dp3": _numeric_delta_for_statuses(
                status_dp2, status_dp3, dp2_summary.get("total_trajectory_time_s"), dp3_summary.get("total_trajectory_time_s")
            ),
            "total_te_scale_dp2_minus_dp3": _numeric_delta_for_statuses(
                status_dp2, status_dp3, dp2_summary.get("total_te_scale_s"), dp3_summary.get("total_te_scale_s")
            ),
            "total_cpu_time_dp2_minus_dp3": _numeric_delta_for_statuses(
                status_dp2, status_dp3, dp2_summary.get("total_cpu_time_s"), dp3_summary.get("total_cpu_time_s")
            ),
            "violation_count_total_dp2_minus_dp3": _count_delta_for_statuses(
                status_dp2,
                status_dp3,
                dp2_summary.get("violation_count_total"),
                dp3_summary.get("violation_count_total"),
            ),
            "max_utilization_dp2_minus_dp3": _max_utilization_delta_for_statuses(
                status_dp2,
                status_dp3,
                dp2_summary.get("max_utilization", {}),
                dp3_summary.get("max_utilization", {}),
            ),
            "objective_cost_total_dp2_minus_dp3": _numeric_delta_for_statuses(
                status_dp2, status_dp3, dp2_summary.get("objective_cost_total"), dp3_summary.get("objective_cost_total")
            ),
            "objective_time_cost_total_dp2_minus_dp3": _numeric_delta_for_statuses(
                status_dp2,
                status_dp3,
                dp2_summary.get("objective_time_cost_total"),
                dp3_summary.get("objective_time_cost_total"),
            ),
            "objective_drive_power_cost_total_dp2_minus_dp3": _numeric_delta_for_statuses(
                status_dp2,
                status_dp3,
                dp2_summary.get("objective_drive_power_cost_total"),
                dp3_summary.get("objective_drive_power_cost_total"),
            ),
        },
        "path_deltas": _comparison_path_deltas(dp3_summary, dp2_summary),
        "reproduction_sources": dp3_summary.get(
            "reproduction_sources",
            _reproduction_sources(limits=limits, limits_source=args.limits, path_index_source=args.path_index),
        ),
        "config": {
            "constraint_check_points": int(args.constraint_check_points),
            "time_samples": int(args.time_samples),
            "z_start": _finite_or_none(args.z_start),
            "z_end": _finite_or_none(args.z_end),
            "z_max": _finite_or_none(args.z_max),
            "tau_rate_dt": _finite_or_none(args.tau_rate_dt),
        },
    }
    batch_error = _comparison_batch_error(dp3_summary, dp2_summary)
    if batch_error is not None:
        comparison["error"] = batch_error
    comparison["source_missing"] = _merged_string_list(dp3_summary.get("source_missing", []), dp2_summary.get("source_missing", []))
    comparison["data_gaps"] = _merged_string_list(dp3_summary.get("data_gaps", []), dp2_summary.get("data_gaps", []))
    comparison["data_readiness"] = _comparison_data_readiness(
        comparison["status"],
        comparison["source_missing"],
        comparison["data_gaps"],
    )
    if getattr(args, "expected_path_count", None) is not None:
        comparison["expected_path_count"] = int(args.expected_path_count)
    (args.out_dir / "comparison_batch_summary.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if status_dp3 == 2 or status_dp2 == 2:
        return 2
    return 0 if status_dp3 == 0 and status_dp2 == 0 else 1


def _copy_args(args, **updates):
    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def _comparison_batch_error(dp3_summary: dict, dp2_summary: dict) -> str | None:
    errors: list[tuple[str, str]] = []
    for method, summary in (("DP3", dp3_summary), ("DP2", dp2_summary)):
        error = summary.get("error")
        if error:
            errors.append((method, str(error)))
    if not errors:
        return None
    unique_errors = _merged_string_list([error for _, error in errors])
    if len(unique_errors) == 1:
        return unique_errors[0]
    return "; ".join(f"{method}: {error}" for method, error in errors)


def _merged_string_list(*values) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        iterable = value if isinstance(value, list | tuple) else [value]
        for item in iterable:
            text = str(item)
            if text in seen:
                continue
            seen.add(text)
            merged.append(text)
    return merged


def _run_source_missing(args) -> list[Path | str]:
    missing: list[Path | str] = []
    if args.limits is None:
        missing.append("limits")
    elif not args.limits.exists():
        missing.append(args.limits)
    if args.path_index is not None and not args.path_index.exists():
        missing.append(args.path_index)
    return missing


def _comparison_batch_entry(status: int, run_dir: Path, summary_path: Path, summary: dict) -> dict:
    return {
        "method": summary.get("method"),
        "status": int(status),
        "batch_status": summary.get("status"),
        "run_dir": str(run_dir),
        "batch_summary_json": str(summary_path),
        "reproduction_sources": summary.get("reproduction_sources", {}),
        "path_count": int(summary.get("path_count", 0)),
        "feasible_count": int(summary.get("feasible_count", 0)),
        "infeasible_count": int(summary.get("infeasible_count", 0)),
        "total_cpu_time_s": _finite_or_none(summary.get("total_cpu_time_s")),
        "total_trajectory_time_s": _finite_or_none(summary.get("total_trajectory_time_s")),
        "total_te_scale_s": _finite_or_none(summary.get("total_te_scale_s")),
        "violation_count_total": int(summary.get("violation_count_total", 0)),
        "max_utilization": dict(summary.get("max_utilization") or {}),
        "objective_cost_total": _finite_or_none(summary.get("objective_cost_total")),
        "objective_time_cost_total": _finite_or_none(summary.get("objective_time_cost_total")),
        "objective_drive_power_cost_total": _finite_or_none(summary.get("objective_drive_power_cost_total")),
    }


def _comparison_path_deltas(dp3_summary: dict, dp2_summary: dict) -> list[dict]:
    dp2_by_id = {str(run.get("id")): run for run in dp2_summary.get("runs", [])}
    deltas: list[dict] = []
    for dp3_run in dp3_summary.get("runs", []):
        run_id = str(dp3_run.get("id"))
        dp2_run = dp2_by_id.get(run_id, {})
        dp2_status = dp2_run.get("status", 2)
        dp3_status = dp3_run.get("status", 2)
        deltas.append(
            {
                "id": run_id,
                "dp3_run_dir": dp3_run.get("run_dir"),
                "dp2_run_dir": dp2_run.get("run_dir"),
                "dp3_status": int(dp3_status),
                "dp2_status": int(dp2_status),
                "dp3_data_readiness": _run_metric_data_readiness(dp3_run),
                "dp2_data_readiness": _run_metric_data_readiness(dp2_run),
                "dp3_source_missing": list(dp3_run.get("source_missing") or []),
                "dp2_source_missing": list(dp2_run.get("source_missing") or []),
                "dp3_data_gaps": list(dp3_run.get("data_gaps") or []),
                "dp2_data_gaps": list(dp2_run.get("data_gaps") or []),
                "dp3_error": dp3_run.get("error"),
                "dp2_error": dp2_run.get("error"),
                "dp3_constraint_utilization_csv": dp3_run.get("constraint_utilization_csv"),
                "dp2_constraint_utilization_csv": dp2_run.get("constraint_utilization_csv"),
                "dp3_constraint_violations_csv": dp3_run.get("constraint_violations_csv"),
                "dp2_constraint_violations_csv": dp2_run.get("constraint_violations_csv"),
                "dp3_constraint_check_source": dp3_run.get("constraint_check_source", ""),
                "dp2_constraint_check_source": dp2_run.get("constraint_check_source", ""),
                "dp3_constraint_check_samples": int(dp3_run.get("constraint_check_samples", 0)),
                "dp2_constraint_check_samples": int(dp2_run.get("constraint_check_samples", 0)),
                "constraint_check_samples_dp2_minus_dp3": _count_delta_for_statuses(
                    dp2_status,
                    dp3_status,
                    dp2_run.get("constraint_check_samples", 0),
                    dp3_run.get("constraint_check_samples", 0),
                ),
                "dp3_max_utilization": dict(dp3_run.get("max_utilization") or {}),
                "dp2_max_utilization": dict(dp2_run.get("max_utilization") or {}),
                "dp3_active_constraint_percent": dict(dp3_run.get("active_constraint_percent") or {}),
                "dp2_active_constraint_percent": dict(dp2_run.get("active_constraint_percent") or {}),
                "dp3_most_restrictive_constraint_percent": dict(
                    dp3_run.get("most_restrictive_constraint_percent") or {}
                ),
                "dp2_most_restrictive_constraint_percent": dict(
                    dp2_run.get("most_restrictive_constraint_percent") or {}
                ),
                "max_utilization_dp2_minus_dp3": _max_utilization_delta_for_statuses(
                    dp2_status,
                    dp3_status,
                    dp2_run.get("max_utilization", {}),
                    dp3_run.get("max_utilization", {}),
                ),
                "active_constraint_percent_dp2_minus_dp3": _metric_map_delta_for_statuses(
                    dp2_status,
                    dp3_status,
                    dp2_run.get("active_constraint_percent", {}),
                    dp3_run.get("active_constraint_percent", {}),
                ),
                "most_restrictive_constraint_percent_dp2_minus_dp3": _metric_map_delta_for_statuses(
                    dp2_status,
                    dp3_status,
                    dp2_run.get("most_restrictive_constraint_percent", {}),
                    dp3_run.get("most_restrictive_constraint_percent", {}),
                ),
                "dp3_required_time_scale_st": dp3_run.get("required_time_scale_st"),
                "dp2_required_time_scale_st": dp2_run.get("required_time_scale_st"),
                "dp3_executable_with_st": dp3_run.get("executable_with_st"),
                "dp2_executable_with_st": dp2_run.get("executable_with_st"),
                "dp3_te_scale": _finite_or_none(dp3_run.get("te_scale")),
                "dp2_te_scale": _finite_or_none(dp2_run.get("te_scale")),
                "te_scale_dp2_minus_dp3": _numeric_delta_for_statuses(
                    dp2_status, dp3_status, dp2_run.get("te_scale"), dp3_run.get("te_scale")
                ),
                "total_time_dp2_minus_dp3": _numeric_delta_for_statuses(
                    dp2_status, dp3_status, dp2_run.get("total_time"), dp3_run.get("total_time")
                ),
                "cpu_time_dp2_minus_dp3": _numeric_delta_for_statuses(
                    dp2_status, dp3_status, dp2_run.get("cpu_time_s"), dp3_run.get("cpu_time_s")
                ),
                "objective_cost_dp2_minus_dp3": _numeric_delta_for_statuses(
                    dp2_status, dp3_status, dp2_run.get("objective_cost"), dp3_run.get("objective_cost")
                ),
                "objective_time_cost_dp2_minus_dp3": _numeric_delta_for_statuses(
                    dp2_status, dp3_status, dp2_run.get("objective_time_cost"), dp3_run.get("objective_time_cost")
                ),
                "objective_drive_power_cost_dp2_minus_dp3": _numeric_delta_for_statuses(
                    dp2_status,
                    dp3_status,
                    dp2_run.get("objective_drive_power_cost"), dp3_run.get("objective_drive_power_cost")
                ),
            }
        )
    return deltas


def _write_comparison_metrics(path: Path, dp3_summary: dict, dp2_summary: dict) -> None:
    dp2_by_id = {str(run.get("id")): run for run in dp2_summary.get("runs", [])}
    constraint_names = _constraint_quantity_names_from_runs(
        list(dp3_summary.get("runs", [])) + list(dp2_summary.get("runs", []))
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_comparison_metrics_header(constraint_names))
        writer.writeheader()
        for dp3_run in dp3_summary.get("runs", []):
            run_id = str(dp3_run.get("id"))
            dp2_run = dp2_by_id.get(run_id, {})
            writer.writerow(_comparison_metrics_row(run_id, dp3_run, dp2_run, constraint_names))


def _write_single_comparison_metrics(path: Path, *, run_id: str, dp3_run: dict, dp2_run: dict) -> None:
    constraint_names = _constraint_quantity_names_from_runs([dp3_run, dp2_run])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_comparison_metrics_header(constraint_names))
        writer.writeheader()
        writer.writerow(_comparison_metrics_row(run_id, dp3_run, dp2_run, constraint_names))


def _comparison_metrics_header(constraint_names: list[str] | None = None) -> list[str]:
    constraint_names = list(CONSTRAINT_QUANTITY_NAMES if constraint_names is None else constraint_names)
    return [
        "id",
        "path_index",
        "limits",
        "model",
        "dynamics_backend",
        "dp3_run_dir",
        "dp2_run_dir",
        "dp3_status",
        "dp2_status",
        "dp3_data_readiness",
        "dp2_data_readiness",
        "dp3_source_missing",
        "dp2_source_missing",
        "dp3_data_gaps",
        "dp2_data_gaps",
        "dp3_error",
        "dp2_error",
        "dp3_t_e_s",
        "dp2_t_e_s",
        "total_time_dp2_minus_dp3",
        "dp3_required_time_scale_st",
        "dp2_required_time_scale_st",
        "dp3_executable_with_st",
        "dp2_executable_with_st",
        "dp3_te_scale",
        "dp2_te_scale",
        "te_scale_dp2_minus_dp3",
        "dp3_t_cpu_s",
        "dp2_t_cpu_s",
        "cpu_time_dp2_minus_dp3",
        "dp3_objective_cost",
        "dp2_objective_cost",
        "dp3_objective_time_cost",
        "dp2_objective_time_cost",
        "dp3_objective_drive_power_cost",
        "dp2_objective_drive_power_cost",
        "objective_time_cost_dp2_minus_dp3",
        "objective_drive_power_cost_dp2_minus_dp3",
        "objective_cost_dp2_minus_dp3",
        "dp3_violation_count",
        "dp2_violation_count",
        "dp3_constraint_utilization_csv",
        "dp2_constraint_utilization_csv",
        "dp3_constraint_violations_csv",
        "dp2_constraint_violations_csv",
        "dp3_constraint_check_source",
        "dp2_constraint_check_source",
        "dp3_constraint_check_samples",
        "dp2_constraint_check_samples",
        "constraint_check_samples_dp2_minus_dp3",
        *(f"dp3_max_utilization_{name}" for name in constraint_names),
        *(f"dp2_max_utilization_{name}" for name in constraint_names),
        *(f"max_utilization_dp2_minus_dp3_{name}" for name in constraint_names),
        *(f"dp3_active_constraint_percent_{name}" for name in constraint_names),
        *(f"dp2_active_constraint_percent_{name}" for name in constraint_names),
        *(f"active_constraint_percent_dp2_minus_dp3_{name}" for name in constraint_names),
        *(f"dp3_most_restrictive_constraint_percent_{name}" for name in constraint_names),
        *(f"dp2_most_restrictive_constraint_percent_{name}" for name in constraint_names),
        *(f"most_restrictive_constraint_percent_dp2_minus_dp3_{name}" for name in constraint_names),
    ]


def _comparison_metrics_row(
    run_id: str,
    dp3_run: dict,
    dp2_run: dict,
    constraint_names: list[str] | None = None,
) -> dict:
    dp2_status = dp2_run.get("status", 2)
    dp3_status = dp3_run.get("status", 2)
    sources = dp3_run.get("reproduction_sources", {}) or dp2_run.get("reproduction_sources", {}) or {}
    constraint_names = list(CONSTRAINT_QUANTITY_NAMES if constraint_names is None else constraint_names)
    row = {
        "id": run_id,
        "path_index": sources.get("path_index") or "",
        "limits": sources.get("limits") or "",
        "model": sources.get("model") or "",
        "dynamics_backend": sources.get("dynamics_backend") or "",
        "dp3_run_dir": dp3_run.get("run_dir", ""),
        "dp2_run_dir": dp2_run.get("run_dir", ""),
        "dp3_status": int(dp3_status),
        "dp2_status": int(dp2_status),
        "dp3_data_readiness": _run_metric_data_readiness(dp3_run),
        "dp2_data_readiness": _run_metric_data_readiness(dp2_run),
        "dp3_source_missing": _csv_list_value(dp3_run.get("source_missing")),
        "dp2_source_missing": _csv_list_value(dp2_run.get("source_missing")),
        "dp3_data_gaps": _csv_list_value(dp3_run.get("data_gaps")),
        "dp2_data_gaps": _csv_list_value(dp2_run.get("data_gaps")),
        "dp3_error": dp3_run.get("error") or "",
        "dp2_error": dp2_run.get("error") or "",
        "dp3_t_e_s": _csv_metric_value(dp3_run.get("total_time")),
        "dp2_t_e_s": _csv_metric_value(dp2_run.get("total_time")),
        "total_time_dp2_minus_dp3": _csv_metric_value(
            _numeric_delta_for_statuses(dp2_status, dp3_status, dp2_run.get("total_time"), dp3_run.get("total_time"))
        ),
        "dp3_required_time_scale_st": dp3_run.get("required_time_scale_st", "-"),
        "dp2_required_time_scale_st": dp2_run.get("required_time_scale_st", "-"),
        "dp3_executable_with_st": dp3_run.get("executable_with_st", "no"),
        "dp2_executable_with_st": dp2_run.get("executable_with_st", "no"),
        "dp3_te_scale": _csv_metric_value(dp3_run.get("te_scale")),
        "dp2_te_scale": _csv_metric_value(dp2_run.get("te_scale")),
        "te_scale_dp2_minus_dp3": _csv_metric_value(
            _numeric_delta_for_statuses(dp2_status, dp3_status, dp2_run.get("te_scale"), dp3_run.get("te_scale"))
        ),
        "dp3_t_cpu_s": _csv_metric_value(dp3_run.get("cpu_time_s")),
        "dp2_t_cpu_s": _csv_metric_value(dp2_run.get("cpu_time_s")),
        "cpu_time_dp2_minus_dp3": _csv_metric_value(
            _numeric_delta_for_statuses(dp2_status, dp3_status, dp2_run.get("cpu_time_s"), dp3_run.get("cpu_time_s"))
        ),
        "dp3_objective_cost": _csv_metric_value(dp3_run.get("objective_cost")),
        "dp2_objective_cost": _csv_metric_value(dp2_run.get("objective_cost")),
        "dp3_objective_time_cost": _csv_metric_value(dp3_run.get("objective_time_cost")),
        "dp2_objective_time_cost": _csv_metric_value(dp2_run.get("objective_time_cost")),
        "dp3_objective_drive_power_cost": _csv_metric_value(dp3_run.get("objective_drive_power_cost")),
        "dp2_objective_drive_power_cost": _csv_metric_value(dp2_run.get("objective_drive_power_cost")),
        "objective_time_cost_dp2_minus_dp3": _csv_metric_value(
            _numeric_delta_for_statuses(
                dp2_status, dp3_status, dp2_run.get("objective_time_cost"), dp3_run.get("objective_time_cost")
            )
        ),
        "objective_drive_power_cost_dp2_minus_dp3": _csv_metric_value(
            _numeric_delta_for_statuses(
                dp2_status,
                dp3_status,
                dp2_run.get("objective_drive_power_cost"),
                dp3_run.get("objective_drive_power_cost"),
            )
        ),
        "objective_cost_dp2_minus_dp3": _csv_metric_value(
            _numeric_delta_for_statuses(
                dp2_status, dp3_status, dp2_run.get("objective_cost"), dp3_run.get("objective_cost")
            )
        ),
        "dp3_violation_count": int(dp3_run.get("violation_count", 0)),
        "dp2_violation_count": int(dp2_run.get("violation_count", 0)),
        "dp3_constraint_utilization_csv": dp3_run.get("constraint_utilization_csv") or "",
        "dp2_constraint_utilization_csv": dp2_run.get("constraint_utilization_csv") or "",
        "dp3_constraint_violations_csv": dp3_run.get("constraint_violations_csv") or "",
        "dp2_constraint_violations_csv": dp2_run.get("constraint_violations_csv") or "",
        "dp3_constraint_check_source": dp3_run.get("constraint_check_source") or "",
        "dp2_constraint_check_source": dp2_run.get("constraint_check_source") or "",
        "dp3_constraint_check_samples": int(dp3_run.get("constraint_check_samples", 0)),
        "dp2_constraint_check_samples": int(dp2_run.get("constraint_check_samples", 0)),
        "constraint_check_samples_dp2_minus_dp3": _csv_metric_value(
            _count_delta_for_statuses(
                dp2_status,
                dp3_status,
                dp2_run.get("constraint_check_samples", 0),
                dp3_run.get("constraint_check_samples", 0),
            )
        ),
    }
    for name in constraint_names:
        row[f"dp3_max_utilization_{name}"] = _csv_metric_value((dp3_run.get("max_utilization", {}) or {}).get(name))
        row[f"dp2_max_utilization_{name}"] = _csv_metric_value((dp2_run.get("max_utilization", {}) or {}).get(name))
        row[f"max_utilization_dp2_minus_dp3_{name}"] = _csv_metric_value(
            _numeric_delta_for_statuses(
                dp2_status,
                dp3_status,
                (dp2_run.get("max_utilization", {}) or {}).get(name),
                (dp3_run.get("max_utilization", {}) or {}).get(name),
            )
        )
        row[f"dp3_active_constraint_percent_{name}"] = _csv_metric_value(
            (dp3_run.get("active_constraint_percent", {}) or {}).get(name)
        )
        row[f"dp2_active_constraint_percent_{name}"] = _csv_metric_value(
            (dp2_run.get("active_constraint_percent", {}) or {}).get(name)
        )
        row[f"active_constraint_percent_dp2_minus_dp3_{name}"] = _csv_metric_value(
            _numeric_delta_for_statuses(
                dp2_status,
                dp3_status,
                (dp2_run.get("active_constraint_percent", {}) or {}).get(name),
                (dp3_run.get("active_constraint_percent", {}) or {}).get(name),
            )
        )
        row[f"dp3_most_restrictive_constraint_percent_{name}"] = _csv_metric_value(
            (dp3_run.get("most_restrictive_constraint_percent", {}) or {}).get(name)
        )
        row[f"dp2_most_restrictive_constraint_percent_{name}"] = _csv_metric_value(
            (dp2_run.get("most_restrictive_constraint_percent", {}) or {}).get(name)
        )
        row[f"most_restrictive_constraint_percent_dp2_minus_dp3_{name}"] = _csv_metric_value(
            _numeric_delta_for_statuses(
                dp2_status,
                dp3_status,
                (dp2_run.get("most_restrictive_constraint_percent", {}) or {}).get(name),
                (dp3_run.get("most_restrictive_constraint_percent", {}) or {}).get(name),
            )
        )
    return row


def _run_path_index(args, limits: ConstraintLimits) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    expected_path_count = getattr(args, "expected_path_count", None)
    try:
        entries = _load_path_index(args.path_index)
    except (OSError, ValueError) as exc:
        error = str(exc)
        print(error)
        return _write_failed_batch_summary(
            out_dir=args.out_dir,
            path_index=args.path_index,
            method=args.method,
            error=error,
            data_gaps=[error],
            expected_path_count=expected_path_count,
            limits_source=args.limits,
            model_source=args.model if getattr(args, "model_explicit", False) else None,
        )
    if expected_path_count is not None and len(entries) != int(expected_path_count):
        error = f"path_index contains {len(entries)} paths, expected {int(expected_path_count)}"
        print(error)
        data_gaps = [error]
        batch_summary = {
            "path_index": str(args.path_index),
            "method": args.method,
            "path_count": len(entries),
            "expected_path_count": int(expected_path_count),
            "status": "failed",
            "data_readiness": _failure_data_readiness([], data_gaps),
            "feasible_count": 0,
            "infeasible_count": len(entries),
            "total_cpu_time_s": 0.0,
            "total_trajectory_time_s": 0.0,
            "total_te_scale_s": 0.0,
            "objective_cost_total": 0.0,
            "runs": [],
            "error": error,
            "source_missing": [],
            "data_gaps": data_gaps,
            "reproduction_sources": _reproduction_sources(
                limits=limits,
                limits_source=args.limits,
                model_source=args.model if getattr(args, "model_explicit", False) else None,
                path_index_source=args.path_index,
            ),
        }
        (args.out_dir / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return 2
    config_error = _path_index_batch_config_error(args, entries)
    if config_error is not None:
        print(config_error)
        data_gaps = [config_error]
        batch_summary = {
            "path_index": str(args.path_index),
            "method": args.method,
            "path_count": len(entries),
            "status": "failed",
            "data_readiness": _failure_data_readiness([], data_gaps),
            "feasible_count": 0,
            "infeasible_count": len(entries),
            "total_cpu_time_s": 0.0,
            "total_trajectory_time_s": 0.0,
            "total_te_scale_s": 0.0,
            "objective_cost_total": 0.0,
            "runs": [],
            "error": config_error,
            "source_missing": [],
            "data_gaps": data_gaps,
            "reproduction_sources": _reproduction_sources(
                limits=limits,
                limits_source=args.limits,
                model_source=args.model if getattr(args, "model_explicit", False) else None,
                path_index_source=args.path_index,
            ),
        }
        if expected_path_count is not None:
            batch_summary["expected_path_count"] = int(expected_path_count)
        (args.out_dir / "batch_summary.json").write_text(
            json.dumps(batch_summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return 2
    robot = None
    model_error = None
    if getattr(args, "model_explicit", False):
        try:
            if not args.model.exists():
                raise FileNotFoundError(args.model)
            robot = MujocoRobotDynamics.from_model_path(args.model)
            if robot.dof != limits.dof:
                raise ValueError(f"MuJoCo model DOF ({robot.dof}) does not match limits DOF ({limits.dof})")
        except (OSError, ValueError) as exc:
            model_error = str(exc)
    try:
        runtime_limits = _limits_with_mujoco_position_bounds(limits, robot)
    except ValueError as exc:
        error = str(exc)
        print(error)
        data_gaps = [error]
        batch_summary = {
            "path_index": str(args.path_index),
            "method": args.method,
            "path_count": len(entries),
            "status": "failed",
            "data_readiness": _failure_data_readiness([], data_gaps),
            "feasible_count": 0,
            "infeasible_count": len(entries),
            "total_cpu_time_s": 0.0,
            "total_trajectory_time_s": 0.0,
            "total_te_scale_s": 0.0,
            "objective_cost_total": 0.0,
            "runs": [],
            "error": error,
            "source_missing": [],
            "data_gaps": data_gaps,
        }
        if expected_path_count is not None:
            batch_summary["expected_path_count"] = int(expected_path_count)
        (args.out_dir / "batch_summary.json").write_text(
            json.dumps(batch_summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return 2
    batch_summary = {
        "path_index": str(args.path_index),
        "method": args.method,
        "path_count": len(entries),
        "reproduction_sources": _reproduction_sources(
            limits=runtime_limits,
            robot=robot,
            limits_source=args.limits,
            model_source=args.model if robot is not None else None,
            path_index_source=args.path_index,
        ),
        "runs": [],
    }
    statuses: list[int] = []
    used_run_ids: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        csv_path = _resolve_index_path(args.path_index.parent, entry["csv"])
        run_id = _unique_run_id(_path_index_run_id(entry, csv_path, index), used_run_ids)
        run_dir = args.out_dir / run_id
        config = _dp3_config_from_args(
            args,
            z_start=_entry_float(entry, ("z_start", "zs"), default=float(args.z_start)),
            z_end=_entry_float(entry, ("z_end", "ze"), default=float(args.z_end)),
            z_max=_entry_float(entry, ("z_max",), default=args.z_max),
        )
        try:
            path = PathData.from_csv(csv_path)
            if model_error is not None:
                raise ValueError(model_error)
            if runtime_limits.dof != path.dof:
                raise ValueError(f"limits DOF ({runtime_limits.dof}) does not match path DOF ({path.dof})")
            if robot is not None and robot.dof != path.dof:
                raise ValueError(f"MuJoCo model DOF ({robot.dof}) does not match path DOF ({path.dof})")
            if robot is not None:
                robot.assert_joint_positions_in_range(path.q)
            status = _run_single_path(
                path=path,
                limits=runtime_limits,
                config=config,
                method=args.method,
                jerk_limited=args.jerk_limited,
                out_dir=run_dir,
                robot=robot,
                path_id=run_id,
                path_source=csv_path,
                limits_source=args.limits,
                model_source=args.model if robot is not None else None,
                path_index_source=args.path_index,
                constraint_check_points=int(args.constraint_check_points),
                time_samples=int(args.time_samples),
            )
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                raise ValueError(f"{run_id} did not write summary.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            run_summary = {
                "id": run_id,
                "method": summary.get("method", args.method.upper()),
                "csv": str(csv_path),
                "run_dir": str(run_dir),
                "status": int(status),
                "feasible": status == 0,
                "data_readiness": "ready",
                "source_missing": [],
                "data_gaps": [],
                "cpu_time_s": _finite_or_zero(summary.get("cpu_time_s")),
                "total_time": _finite_or_zero(summary.get("total_time")),
                "max_utilization": summary.get("max_utilization", {}),
                "violation_count": len(summary.get("violations", [])),
                "required_time_scale_st": str(summary.get("required_time_scale_st", "-")),
                "executable_with_st": str(summary.get("executable_with_st", "no")),
                "te_scale": _finite_or_marker(summary.get("te_scale")),
                "objective_cost": _finite_or_zero(summary.get("objective_cost")),
                "objective_time_cost": _finite_or_zero(summary.get("objective_time_cost")),
                "objective_drive_power_cost": _finite_or_zero(summary.get("objective_drive_power_cost")),
                "active_constraint_threshold": _finite_or_zero(summary.get("active_constraint_threshold")),
                "constraint_check_points": int(summary.get("constraint_check_points", 0)),
                "constraint_check_samples": int(summary.get("constraint_check_samples", 0)),
                "constraint_check_source": summary.get("constraint_check_source", ""),
                "constraint_utilization_csv": summary.get("constraint_utilization_csv"),
                "constraint_violations_csv": summary.get("constraint_violations_csv"),
                "reproduction_sources": summary.get("reproduction_sources", {}),
                "active_constraint_percent": summary.get("active_constraint_percent", {}),
                "most_restrictive_constraint_percent": summary.get("most_restrictive_constraint_percent", {}),
                "z_start": _finite_or_none(config.z_start),
                "z_end": _finite_or_none(config.z_end),
                "z_max": _finite_or_none(config.z_max),
                "tau_rate_dt": _finite_or_none(config.tau_rate_dt),
            }
        except (OSError, ValueError) as exc:
            status = 2
            print(f"{run_id}: {exc}")
            failure_source_missing = [csv_path] if isinstance(exc, OSError) and not csv_path.exists() else []
            failure_data_gaps = [] if failure_source_missing else [str(exc)]
            failure_sources = _reproduction_sources(
                limits=runtime_limits,
                robot=robot,
                path_source=csv_path,
                limits_source=args.limits,
                model_source=args.model if robot is not None else None,
                path_index_source=args.path_index,
            )
            _write_failed_run_summary(
                out_dir=run_dir,
                run_id=run_id,
                csv_path=csv_path,
                config=config,
                method=args.method,
                jerk_limited=args.jerk_limited,
                error=str(exc),
                source_missing=failure_source_missing,
                data_gaps=failure_data_gaps,
                limits=runtime_limits,
                robot=robot,
                limits_source=args.limits,
                model_source=args.model if robot is not None else None,
                path_index_source=args.path_index,
            )
            run_summary = {
                "id": run_id,
                "method": args.method.upper(),
                "csv": str(csv_path),
                "run_dir": str(run_dir),
                "status": int(status),
                "feasible": False,
                "data_readiness": _failure_data_readiness(
                    [str(item) for item in failure_source_missing],
                    failure_data_gaps,
                ),
                "source_missing": [str(item) for item in failure_source_missing],
                "data_gaps": failure_data_gaps,
                "error": str(exc),
                "cpu_time_s": 0.0,
                "total_time": 0.0,
                "max_utilization": {},
                "violation_count": 0,
                "required_time_scale_st": "-",
                "executable_with_st": "no",
                "te_scale": "-",
                "objective_cost": 0.0,
                "objective_time_cost": 0.0,
                "objective_drive_power_cost": 0.0,
                "active_constraint_threshold": ACTIVE_CONSTRAINT_THRESHOLD,
                "constraint_check_points": 0,
                "constraint_check_samples": 0,
                "constraint_check_source": "not_run",
                "constraint_utilization_csv": None,
                "constraint_violations_csv": None,
                "reproduction_sources": failure_sources,
                "active_constraint_percent": {},
                "most_restrictive_constraint_percent": {},
                "z_start": _finite_or_none(config.z_start),
                "z_end": _finite_or_none(config.z_end),
                "z_max": _finite_or_none(config.z_max),
                "tau_rate_dt": _finite_or_none(config.tau_rate_dt),
            }
        statuses.append(status)
        batch_summary["runs"].append(run_summary)
    _add_batch_totals(batch_summary, statuses)
    batch_metrics_csv = "batch_metrics.csv"
    batch_summary["batch_metrics_csv"] = batch_metrics_csv
    _write_batch_metrics(args.out_dir / batch_metrics_csv, batch_summary["runs"])
    (args.out_dir / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if any(status == 2 for status in statuses):
        return 2
    return 0 if all(status == 0 for status in statuses) else 1


def _write_failed_run_summary(
    *,
    out_dir: Path,
    run_id: str,
    csv_path: Path,
    config: DP3Config,
    method: str,
    jerk_limited: str | None,
    error: str,
    source_missing: list[Path | str] | None = None,
    data_gaps: list[str] | None = None,
    path: PathData | None = None,
    limits: ConstraintLimits | None = None,
    robot: MujocoRobotDynamics | None = None,
    limits_source: Path | None = None,
    model_source: Path | None = None,
    path_index_source: Path | None = None,
    joint_path_csv: Path | None = None,
    joint_path_source: str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_missing_items = [str(path) for path in (source_missing or [])]
    data_gap_items = list(data_gaps or [])
    summary = {
        "method": method.upper(),
        "jerk_limited": jerk_limited or ("yes" if method == "dp3" else "no"),
        "feasible": False,
        "data_readiness": "source_missing" if source_missing_items else ("data_gaps" if data_gap_items else "failed"),
        "source_missing": source_missing_items,
        "data_gaps": data_gap_items,
        "samples": 0,
        "cpu_time_s": 0.0,
        "total_time": 0.0,
        "objective_cost": 0.0,
        "objective_time_cost": 0.0,
        "objective_drive_power_cost": 0.0,
        "segment_kinds": [],
        "segment_count": 0,
        "constraint_check_points": 0,
        "constraint_check_samples": 0,
        "constraint_check_source": "not_run",
        "constraint_utilization_csv": None,
        "constraint_violations_csv": None,
        "max_utilization": {},
        "active_constraint_threshold": ACTIVE_CONSTRAINT_THRESHOLD,
        "active_constraint_percent": {},
        "most_restrictive_constraint_percent": {},
        "violations": [],
        "required_time_scale_st": "-",
        "executable_with_st": "no",
        "te_scale": "-",
        "path_id": run_id,
        "path_source": str(csv_path),
        "reproduction_sources": _reproduction_sources(
            path=path,
            limits=limits,
            robot=robot,
            path_source=csv_path,
            joint_path_csv=joint_path_csv,
            joint_path_source=joint_path_source,
            limits_source=limits_source,
            model_source=model_source,
            path_index_source=path_index_source,
        ),
        "error": error,
        "config": _config_summary(config),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _path_index_batch_config_error(args, entries: list[dict]) -> str | None:
    for index, entry in enumerate(entries, start=1):
        z_start = _entry_float(entry, ("z_start", "zs"), default=float(args.z_start))
        z_end = _entry_float(entry, ("z_end", "ze"), default=float(args.z_end))
        z_max = _entry_float(entry, ("z_max",), default=args.z_max)
        for name, value in (("z_start", z_start), ("z_end", z_end)):
            if not np.isfinite(value) or value < 0.0:
                return f"path_index entry {index} {name} must be finite and nonnegative"
        if z_max is None:
            continue
        if not np.isfinite(z_max) or z_max <= 0.0:
            return f"path_index entry {index} z_max must be finite and positive"
        if z_start > z_max + 1e-12:
            return f"path_index entry {index} z_start must not exceed z_max"
        if z_end > z_max + 1e-12:
            return f"path_index entry {index} z_end must not exceed z_max"
    return None


def _add_batch_totals(batch_summary: dict, statuses: list[int]) -> None:
    runs = batch_summary["runs"]
    feasible_count = sum(1 for status in statuses if status == 0)
    failed_count = len(statuses) - feasible_count
    batch_summary["status"] = "ok" if failed_count == 0 else "failed"
    batch_summary["feasible_count"] = int(feasible_count)
    batch_summary["infeasible_count"] = int(failed_count)
    batch_summary["total_cpu_time_s"] = float(sum(float(run["cpu_time_s"]) for run in runs))
    batch_summary["total_trajectory_time_s"] = float(sum(float(run["total_time"]) for run in runs))
    batch_summary["total_te_scale_s"] = float(sum(_finite_or_zero(run.get("te_scale")) for run in runs))
    batch_summary["objective_cost_total"] = float(sum(float(run["objective_cost"]) for run in runs))
    batch_summary["objective_time_cost_total"] = float(sum(float(run.get("objective_time_cost", 0.0)) for run in runs))
    batch_summary["objective_drive_power_cost_total"] = float(
        sum(float(run.get("objective_drive_power_cost", 0.0)) for run in runs)
    )
    batch_summary["violation_count_total"] = int(sum(int(run.get("violation_count", 0)) for run in runs))
    batch_summary["max_utilization"] = _aggregate_max_utilization(runs)


def _aggregate_max_utilization(runs: list[dict]) -> dict[str, float]:
    names = _constraint_quantity_names_from_runs(runs)
    aggregate = {name: 0.0 for name in names}
    for run in runs:
        max_utilization = run.get("max_utilization", {}) or {}
        for name in names:
            value = _finite_or_none(max_utilization.get(name))
            if value is not None:
                aggregate[name] = max(aggregate[name], float(value))
    return aggregate


def _write_batch_metrics(path: Path, runs: list[dict]) -> None:
    constraint_names = _constraint_quantity_names_from_runs(runs)
    header = [
        *BATCH_METRIC_BASE_FIELDS,
        *(f"max_utilization_{name}" for name in constraint_names),
        *(f"active_constraint_percent_{name}" for name in constraint_names),
        *(f"most_restrictive_constraint_percent_{name}" for name in constraint_names),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for run in runs:
            sources = run.get("reproduction_sources", {}) or {}
            row = {
                "id": run.get("id", ""),
                "method": run.get("method", ""),
                "status": run.get("status", ""),
                "feasible": "yes" if run.get("feasible") else "no",
                "data_readiness": _run_metric_data_readiness(run),
                "error": run.get("error") or "",
                "source_missing": _csv_list_value(run.get("source_missing")),
                "data_gaps": _csv_list_value(run.get("data_gaps")),
                "csv": run.get("csv", ""),
                "path_index": sources.get("path_index") or "",
                "limits": sources.get("limits") or "",
                "model": sources.get("model") or "",
                "dynamics_backend": sources.get("dynamics_backend") or "",
                "run_dir": run.get("run_dir", ""),
                "t_e_s": _csv_metric_value(run.get("total_time")),
                "t_cpu_s": _csv_metric_value(run.get("cpu_time_s")),
                "objective_cost": _csv_metric_value(run.get("objective_cost")),
                "objective_time_cost": _csv_metric_value(run.get("objective_time_cost")),
                "objective_drive_power_cost": _csv_metric_value(run.get("objective_drive_power_cost")),
                "required_time_scale_st": run.get("required_time_scale_st", "-"),
                "executable_with_st": run.get("executable_with_st", "no"),
                "te_scale": _csv_metric_value(run.get("te_scale")),
                "constraint_check_samples": run.get("constraint_check_samples", 0),
                "constraint_check_source": run.get("constraint_check_source", ""),
                "violation_count": run.get("violation_count", 0),
                "constraint_utilization_csv": run.get("constraint_utilization_csv") or "",
                "constraint_violations_csv": run.get("constraint_violations_csv") or "",
            }
            for name in constraint_names:
                row[f"max_utilization_{name}"] = _csv_metric_value((run.get("max_utilization", {}) or {}).get(name))
                row[f"active_constraint_percent_{name}"] = _csv_metric_value(
                    (run.get("active_constraint_percent", {}) or {}).get(name)
                )
                row[f"most_restrictive_constraint_percent_{name}"] = _csv_metric_value(
                    (run.get("most_restrictive_constraint_percent", {}) or {}).get(name)
                )
            writer.writerow(row)


def _constraint_quantity_names_from_runs(runs: list[dict]) -> list[str]:
    names = list(CONSTRAINT_QUANTITY_NAMES)
    for run in runs:
        for section in ("max_utilization", "active_constraint_percent", "most_restrictive_constraint_percent"):
            for name in (run.get(section, {}) or {}):
                if name not in names:
                    names.append(str(name))
    return names


def _csv_metric_value(value) -> str:
    finite = _finite_or_none(value)
    return "" if finite is None else f"{finite:.12g}"


def _csv_list_value(value) -> str:
    if not value:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item) for item in value)
    return str(value)


def _run_metric_data_readiness(run: dict) -> str:
    readiness = run.get("data_readiness")
    if readiness:
        return str(readiness)
    try:
        status = int(run.get("status", 2))
    except (TypeError, ValueError):
        status = 2
    if status == 0:
        return "ready"
    return _failure_data_readiness(
        [str(item) for item in (run.get("source_missing") or [])],
        [str(item) for item in (run.get("data_gaps") or [])],
    )


def _write_trajectory(path: Path, result) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "s", "z", "z_s", "z_ss"])
        for row in zip(result.t, result.s, result.z, result.z_s, result.z_ss):
            writer.writerow([f"{float(value):.12g}" for value in row])


def _write_quantities(path: Path, result, quantities) -> None:
    dof = int(quantities.q.shape[1])
    header = ["t", "s", "z", "z_s", "z_ss"]
    header.extend(f"q{axis}" for axis in range(1, dof + 1))
    header.extend(f"q_dot{axis}" for axis in range(1, dof + 1))
    header.extend(f"q_ddot{axis}" for axis in range(1, dof + 1))
    header.extend(f"q_jerk{axis}" for axis in range(1, dof + 1))
    header.extend(f"tau{axis}" for axis in range(1, dof + 1))
    header.extend(f"tau_rate{axis}" for axis in range(1, dof + 1))
    header.append("mechanical_power")
    if quantities.drive_power is not None:
        header.append("drive_power")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for sample in range(result.s.size):
            row = [
                result.t[sample],
                result.s[sample],
                result.z[sample],
                result.z_s[sample],
                result.z_ss[sample],
                *quantities.q[sample],
                *quantities.q_dot[sample],
                *quantities.q_ddot[sample],
                *quantities.q_jerk[sample],
                *quantities.tau[sample],
                *quantities.tau_rate[sample],
                quantities.mechanical_power[sample],
            ]
            if quantities.drive_power is not None:
                row.append(quantities.drive_power[sample])
            writer.writerow([f"{float(value):.12g}" for value in row])


def _write_constraint_utilizations(path: Path, result, quantities) -> None:
    collapsed = {name: _collapse_utilization_to_samples(values) for name, values in _constraint_utilizations(quantities).items()}
    names = list(collapsed)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "s", "z", *names])
        for sample in range(result.s.size):
            row = [
                result.t[sample],
                result.s[sample],
                result.z[sample],
                *(collapsed[name][sample] for name in names),
            ]
            writer.writerow([f"{float(value):.12g}" for value in row])


def _write_constraint_violations(path: Path, result, violations) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "sample", "axis", "t", "s", "z", "value", "limit", "utilization"])
        for violation in violations:
            sample = int(violation.sample)
            axis = "" if violation.axis is None else int(violation.axis) + 1
            if 0 <= sample < result.s.size:
                t_value = result.t[sample]
                s_value = result.s[sample]
                z_value = result.z[sample]
            else:
                t_value = float("nan")
                s_value = float("nan")
                z_value = float("nan")
            row = [
                violation.quantity,
                sample,
                axis,
                t_value,
                s_value,
                z_value,
                violation.value,
                violation.limit,
                violation.utilization,
            ]
            writer.writerow([value if isinstance(value, str) else f"{float(value):.12g}" for value in row])


def _constraint_check_result(
    *,
    path: PathData,
    result: TrajectoryResult,
    limits: ConstraintLimits,
    robot: MujocoRobotDynamics | None,
    points: int,
):
    if int(points) > 0 and result.s.size > 0:
        sample_count = max(2, int(points))
        checked = resample_trajectory_by_segments(result, sample_count)
        if checked is not None:
            source = "dense_segment_profiles"
        else:
            s = np.linspace(float(result.s[0]), float(result.s[-1]), sample_count)
            checked = TrajectoryResult(
                feasible=result.feasible,
                t=np.interp(s, result.s, result.t),
                s=s,
                z=np.interp(s, result.s, result.z),
                z_s=np.interp(s, result.s, result.z_s),
                z_ss=np.interp(s, result.s, result.z_ss),
                total_time=result.total_time,
                audit=result.audit,
                grid_s=result.grid_s,
                grid_z=result.grid_z,
                policy=result.policy,
                method=result.method,
                jerk_limited=result.jerk_limited,
                segment_kinds=result.segment_kinds,
                segment_profiles=result.segment_profiles,
                tau_rate_dt=result.tau_rate_dt,
                objective_cost=result.objective_cost,
            )
            source = "dense_s_grid"
    else:
        checked = result
        source = "trajectory_samples"
    quantities = evaluate_trajectory_quantities(path=path, result=checked, limits=limits, robot=robot)
    if checked.s.size == 0:
        audit = result.audit
    else:
        audit = audit_constraints(
            limits=limits,
            q_position=quantities.q if limits.has_q_position_bounds else None,
            q_dot=quantities.q_dot,
            q_ddot=quantities.q_ddot,
            q_jerk=quantities.q_jerk,
            tau=quantities.tau,
            tau_rate=quantities.tau_rate,
            mechanical_power=quantities.mechanical_power,
        )
    return checked, quantities, audit, source


def _active_constraint_percentages(quantities, threshold: float = ACTIVE_CONSTRAINT_THRESHOLD) -> dict[str, float]:
    utilizations = _constraint_utilizations(quantities)
    return {name: _active_percent(_collapse_utilization_to_samples(values), threshold) for name, values in utilizations.items()}


def _most_restrictive_constraint_percentages(
    quantities,
    threshold: float = ACTIVE_CONSTRAINT_THRESHOLD,
) -> dict[str, float]:
    collapsed = {
        name: _collapse_utilization_to_samples(values)
        for name, values in _constraint_utilizations(quantities).items()
    }
    names = list(collapsed)
    if not collapsed:
        return {name: 0.0 for name in CONSTRAINT_QUANTITY_NAMES}
    sample_count = max((values.size for values in collapsed.values()), default=0)
    if sample_count == 0:
        return {name: 0.0 for name in CONSTRAINT_QUANTITY_NAMES}
    matrix = np.vstack(
        [
            _pad_or_zero(collapsed[name], sample_count)
            for name in names
        ]
    )
    matrix = np.nan_to_num(matrix, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    active_samples = np.max(matrix, axis=0) >= float(threshold)
    active_count = int(np.count_nonzero(active_samples))
    if active_count == 0:
        return {name: 0.0 for name in names}
    winners = np.argmax(matrix[:, active_samples], axis=0)
    counts = np.bincount(winners, minlength=len(names)).astype(np.float64)
    percentages = 100.0 * counts / float(active_count)
    if percentages.size:
        percentages[-1] += 100.0 - float(np.sum(percentages))
    return {name: float(percentages[index]) for index, name in enumerate(names)}


def _constraint_utilizations(quantities) -> dict[str, np.ndarray]:
    utilizations = {
        "q_dot": quantities.q_dot_utilization,
        "q_ddot": quantities.q_ddot_utilization,
        "q_jerk": quantities.q_jerk_utilization,
        "tau": quantities.tau_utilization,
        "tau_rate": quantities.tau_rate_utilization,
        "mechanical_power": quantities.mechanical_power_utilization,
    }
    if getattr(quantities, "q_position_utilization", None) is not None:
        utilizations["q_position"] = quantities.q_position_utilization
    return utilizations


def _collapse_utilization_to_samples(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        return np.abs(arr).reshape(1)
    if arr.ndim == 1:
        return np.abs(arr)
    if arr.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return np.max(np.abs(arr), axis=1)


def _pad_or_zero(values: np.ndarray, sample_count: int) -> np.ndarray:
    if values.size == sample_count:
        return values
    out = np.zeros(int(sample_count), dtype=np.float64)
    if values.size:
        out[: min(values.size, sample_count)] = values[:sample_count]
    return out


def _active_percent(values, threshold: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.abs(arr[np.isfinite(arr)])
    if finite.size == 0:
        return 0.0
    return float(100.0 * np.count_nonzero(finite >= float(threshold)) / finite.size)


def _write_path_csv(path: Path, path_data: PathData) -> None:
    dof = path_data.dof
    header = ["s"]
    header.extend(f"q{axis}" for axis in range(1, dof + 1))
    header.extend(f"dq{axis}_ds" for axis in range(1, dof + 1))
    header.extend(f"d2q{axis}_ds2" for axis in range(1, dof + 1))
    header.extend(f"d3q{axis}_ds3" for axis in range(1, dof + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for sample in range(path_data.samples):
            row = [
                path_data.s[sample],
                *path_data.q[sample],
                *path_data.q_s[sample],
                *path_data.q_ss[sample],
                *path_data.q_sss[sample],
            ]
            writer.writerow([f"{float(value):.12g}" for value in row])


def _load_cli_config(argv: list[str] | None) -> tuple[Path | None, dict]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    known, _ = pre_parser.parse_known_args(argv)
    if known.config is None:
        return None, {}
    path = Path(known.config)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return path, {}
    if not isinstance(raw, dict):
        raise ValueError("run config YAML must contain a mapping")
    return path, raw


def _config_path(config: dict, key: str, config_path: Path | None) -> Path | None:
    value = config.get(key)
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute() or config_path is None:
        return path
    return config_path.parent / path


def _config_bool_choice(config: dict, key: str, *, default: bool) -> str:
    value = config.get(key, default)
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return "yes"
    if text in {"0", "false", "no", "n", "off"}:
        return "no"
    raise ValueError(f"{key} must be a boolean or yes/no")


def _limits_with_mujoco_position_bounds(
    limits: ConstraintLimits,
    robot: MujocoRobotDynamics | None,
) -> ConstraintLimits:
    if robot is None:
        return limits
    if limits.has_q_position_bounds:
        _validate_q_position_within_mujoco_ranges(limits, robot)
        return limits
    return replace(
        limits,
        q_position_lower=np.asarray(robot.lower, dtype=np.float64),
        q_position_upper=np.asarray(robot.upper, dtype=np.float64),
    )


def _validate_q_position_within_mujoco_ranges(limits: ConstraintLimits, robot: MujocoRobotDynamics) -> None:
    lower = np.asarray(limits.q_position_lower, dtype=np.float64)
    upper = np.asarray(limits.q_position_upper, dtype=np.float64)
    model_lower = np.asarray(robot.lower, dtype=np.float64)
    model_upper = np.asarray(robot.upper, dtype=np.float64)
    bad = np.argwhere((lower < model_lower - 1e-12) | (upper > model_upper + 1e-12))
    if bad.size == 0:
        return
    axis = int(bad[0, 0])
    joint_name = robot.joint_names[axis] if axis < len(robot.joint_names) else f"joint_{axis + 1}"
    raise ValueError(
        "q_position limits must stay within MuJoCo joint range for "
        f"{joint_name}: YAML [{float(lower[axis]):.6g}, {float(upper[axis]):.6g}], "
        f"MuJoCo [{float(model_lower[axis]):.6g}, {float(model_upper[axis]):.6g}]"
    )


def _yes_no(value: str) -> bool:
    return str(value).strip().lower() == "yes"


def _argv_has_option(argv: list[str] | None, name: str) -> bool:
    if argv is None:
        return False
    prefix = f"{name}="
    return any(item == name or item.startswith(prefix) for item in argv)


def _dp3_config_from_args(
    args,
    *,
    z_start: float | None = None,
    z_end: float | None = None,
    z_max: float | None = None,
) -> DP3Config:
    return DP3Config(
        ns=args.ns,
        nz=args.nz,
        nch=args.nch,
        k1=float(args.k1),
        k2=float(args.k2),
        z_start=float(args.z_start if z_start is None else z_start),
        z_end=float(args.z_end if z_end is None else z_end),
        z_max=args.z_max if z_max is None else z_max,
        tau_rate_dt=float(args.tau_rate_dt),
    )


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value) -> int | None:
    if value is None:
        return None
    return int(value)


def _input_run_id(args) -> str:
    path = args.path_csv if args.path_csv is not None else args.input
    return path.stem if path is not None and path.stem else "run"


def _ik_settings_from_args(args) -> dict:
    return {
        "max_iterations": int(args.ik_max_iters),
        "position_tolerance": float(args.ik_pos_tol),
        "orientation_tolerance": float(args.ik_ori_tol),
        "damping": float(args.ik_damping),
        "step_scale": float(args.ik_step_scale),
        "orientation_weight": float(args.ik_orientation_weight),
        "ee_body": str(args.ee_body),
        "ee_site": None if getattr(args, "ee_site", None) is None else str(args.ee_site),
        "tcp_offset": [float(args.tcp_offset_x), float(args.tcp_offset_y), float(args.tcp_offset_z)],
    }


def _config_summary(config: DP3Config) -> dict:
    return {
        "ns": _integer_or_none(config.ns),
        "nz": _integer_or_none(config.nz),
        "nch": _integer_or_none(config.nch),
        "k1": _finite_or_none(config.k1),
        "k2": _finite_or_none(config.k2),
        "z_start": _finite_or_none(config.z_start),
        "z_end": _finite_or_none(config.z_end),
        "z_max": _finite_or_none(config.z_max),
        "tau_rate_dt": _finite_or_none(config.tau_rate_dt),
    }


def _integer_or_none(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    return int(value)


def _finite_or_none(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _finite_or_zero(value) -> float:
    out = _finite_or_none(value)
    return 0.0 if out is None else out


def _finite_or_marker(value):
    out = _finite_or_none(value)
    return "-" if out is None else out


def _entry_float(entry: dict, names: tuple[str, ...], default: float | None) -> float | None:
    for name in names:
        if name in entry and entry[name] is not None:
            return float(entry[name])
    return default


def _load_path_index(path: Path) -> list[dict]:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"path_index YAML is invalid: {exc}") from exc
    entries = raw.get("paths") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise ValueError("path_index YAML must contain a non-empty paths list")
    normalized: list[dict] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"path_index entry {index} must be a mapping")
        if "csv" not in entry or entry["csv"] is None:
            raise ValueError(f"path_index entry {index} is missing csv")
        _validate_path_index_entry_scalars(entry, index)
        normalized.append(entry)
    return normalized


def _path_index_template_gaps(path: Path) -> list[str]:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return []
    if isinstance(raw, dict) and raw.get("template") is True:
        return [f"{path}: path_index is marked as a template; replace placeholder entries with real path CSV data and remove template: true"]
    return []


def _validate_path_index_entry_scalars(entry: dict, index: int) -> None:
    endpoint_values: list[tuple[str, float]] = []
    for primary, alias in (("zs", "z_start"), ("ze", "z_end")):
        primary_value = _optional_entry_nonnegative_float(entry, primary, index)
        alias_value = _optional_entry_nonnegative_float(entry, alias, index)
        if primary_value is not None and alias_value is not None and abs(primary_value - alias_value) > 1e-12:
            raise ValueError(f"path_index entry {index} {primary} and {alias} conflict")
        endpoint_value = primary_value if primary_value is not None else alias_value
        if endpoint_value is not None:
            endpoint_values.append((primary if primary_value is not None else alias, endpoint_value))
    z_max = _optional_entry_float(entry, "z_max", index)
    if z_max is not None and z_max <= 0.0:
        raise ValueError(f"path_index entry {index} z_max must be finite and positive")
    if z_max is not None:
        for name, endpoint_value in endpoint_values:
            if endpoint_value > z_max + 1e-12:
                raise ValueError(f"path_index entry {index} {name} exceeds z_max")


def _optional_entry_nonnegative_float(entry: dict, name: str, index: int) -> float | None:
    value = _optional_entry_float(entry, name, index)
    if value is not None and value < 0.0:
        raise ValueError(f"path_index entry {index} {name} must be finite and nonnegative")
    return value


def _optional_entry_float(entry: dict, name: str, index: int) -> float | None:
    if name not in entry or entry[name] is None:
        return None
    try:
        value = float(entry[name])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"path_index entry {index} {name} must be finite") from exc
    if not np.isfinite(value):
        raise ValueError(f"path_index entry {index} {name} must be finite")
    return value


def _path_csv_gaps(paths: list[Path], expected_dof: int, robot: MujocoRobotDynamics | None = None) -> list[str]:
    gaps: list[str] = []
    for path in paths:
        try:
            path_data = PathData.from_csv(path)
        except ValueError as exc:
            gaps.append(f"{path}: {exc}")
            continue
        if path_data.dof != int(expected_dof):
            gaps.append(f"{path}: DOF {path_data.dof} does not match expected {int(expected_dof)}")
            continue
        if robot is not None:
            try:
                robot.assert_joint_positions_in_range(path_data.q)
            except ValueError as exc:
                gaps.append(f"{path}: {exc}")
    return gaps


def _path_index_endpoint_speed_gaps(
    entries: list[dict],
    base_dir: Path,
    limits: ConstraintLimits,
    *,
    default_z_start: float | None = None,
    default_z_end: float | None = None,
    default_z_max: float | None = None,
) -> list[str]:
    gaps: list[str] = []
    for index, entry in enumerate(entries, start=1):
        csv_path = _resolve_index_path(base_dir, entry["csv"])
        if not csv_path.exists():
            continue
        try:
            path_data = PathData.from_csv(csv_path)
        except ValueError:
            continue
        if path_data.dof != limits.dof:
            continue
        run_id = _path_index_run_id(entry, csv_path, index)
        entry_z_max = _entry_float(entry, ("z_max",), default=default_z_max)
        for field_names, default_z, sample, label in (
            (("z_start", "zs"), default_z_start, 0, "s=0"),
            (("z_end", "ze"), default_z_end, -1, "s=1"),
        ):
            z_value = _entry_float(entry, field_names, default=default_z)
            if z_value is None:
                continue
            if not np.isfinite(z_value) or z_value < 0.0:
                gaps.append(f"path_index entry {index} ({run_id}) {field_names[0]}/{field_names[1]} must be finite and nonnegative")
                continue
            if entry_z_max is not None and float(z_value) > float(entry_z_max) + 1e-12:
                gaps.append(f"path_index entry {index} ({run_id}) {field_names[0]}/{field_names[1]} exceeds z_max")
                continue
            q_dot = path_data.q_s[[sample]] * np.sqrt(float(z_value))
            lower, upper = limits.q_dot_bounds(q_dot.shape)
            bad = np.argwhere((q_dot < lower - 1e-12) | (q_dot > upper + 1e-12))
            if bad.size == 0:
                continue
            _, axis = bad[0]
            axis = int(axis)
            value = float(q_dot[0, axis])
            limit = float(upper[0, axis] if value > upper[0, axis] else lower[0, axis])
            gaps.append(
                f"path_index entry {index} ({run_id}) {field_names[0]}/{field_names[1]} violates q_dot limit at {label}: "
                f"axis {axis + 1} q_dot={value:.6g}, limit={limit:.6g}"
            )
    return gaps


def _mujoco_model_gaps(path: Path, expected_dof: int) -> list[str]:
    _, gaps = _mujoco_model_for_validation(path, expected_dof)
    return gaps


def _mujoco_model_for_validation(path: Path, expected_dof: int) -> tuple[MujocoRobotDynamics | None, list[str]]:
    try:
        robot = MujocoRobotDynamics.from_model_path(path)
    except (OSError, ValueError) as exc:
        return None, [f"{path}: MuJoCo model load failed: {exc}"]
    if robot.dof != int(expected_dof):
        return None, [f"{path}: MuJoCo model DOF {robot.dof} does not match expected {int(expected_dof)}"]
    return robot, []


def _offline_traj_gaps(path: Path) -> list[str]:
    try:
        poses = parse_pose_table(path)
    except ValueError as exc:
        return [f"{path}: {exc}"]
    if poses.shape[0] < 2:
        return [f"{path}: expected at least two poses"]
    return []


def _cartesian_ik_gaps(
    *,
    model_path: Path,
    offline_traj_path: Path,
    ee_body: str,
    ee_site: str | None,
    samples: int,
    max_iters: int,
    pos_tol: float,
    ori_tol: float,
    damping: float,
    step_scale: float,
    orientation_weight: float,
    tcp_offset: np.ndarray | None,
    require_convergence: bool,
) -> list[str]:
    try:
        poses = parse_pose_table(offline_traj_path)
        count = max(2, min(int(samples), poses.shape[0]))
        robot = MujocoRobotDynamics.from_model_path(model_path)
        result = build_joint_path_result_from_cartesian(
            robot=robot,
            pose_table=poses[:count],
            ee_body=ee_body,
            ee_site=ee_site,
            max_iters=max(1, int(max_iters)),
            pos_tol=float(pos_tol),
            ori_tol=float(ori_tol),
            damping=float(damping),
            step_scale=float(step_scale),
            orientation_weight=float(orientation_weight),
            tcp_offset=tcp_offset,
            require_convergence=bool(require_convergence),
        )
        if result.path.dof != 6:
            return [f"{offline_traj_path}: IK check produced DOF {result.path.dof}, expected 6"]
        if require_convergence and not result.ik.converged:
            return [f"{offline_traj_path}: IK check did not converge"]
    except (OSError, ValueError) as exc:
        return [f"{offline_traj_path}: IK check failed: {exc}"]
    return []


def _resolve_index_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def _path_index_run_id(entry: dict, csv_path: Path, index: int) -> str:
    raw = str(entry.get("id") or entry.get("name") or csv_path.stem or f"path_{index:02d}")
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in raw).strip("._")
    return safe or f"path_{index:02d}"


def _unique_run_id(base: str, used: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _required_time_scale(max_utilization: dict[str, float]) -> float:
    scale = 1.0
    for name, exponent in TIME_SCALE_EXPONENTS.items():
        utilization = float(max_utilization.get(name, 0.0))
        if utilization > 1.0:
            scale = min(scale, utilization ** (-1.0 / exponent))
    return float(scale)


def _time_scale_covers_violations(max_utilization: dict[str, float]) -> bool:
    for name, utilization in max_utilization.items():
        if float(utilization) > 1.0 + 1e-12 and name not in TIME_SCALE_EXPONENTS:
            return False
    return True


def _format_time_scale(scale: float) -> str:
    return f"{100.0 * float(scale):.6g}%"
