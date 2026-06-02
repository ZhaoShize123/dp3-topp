from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable


COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
    "#be123c",
)


@dataclass(frozen=True)
class ToppraCurve:
    s: list[float]
    mvc_speed: list[float]
    profile_speed: list[float] | None = None


def plot_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate SVG result plots from DP3/TOPPRA run outputs.")
    parser.add_argument("--run", type=Path, required=True, help="Run directory containing CSV/JSON outputs.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for SVG plots. Defaults to <run>/plots.")
    args = parser.parse_args(argv)
    try:
        plot_run(args.run, args.out_dir)
    except Exception as exc:  # pragma: no cover - argparse-style CLI guard
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def plot_run(run_dir: Path, out_dir: Path | None = None) -> list[Path]:
    run_dir = Path(run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    out_dir = run_dir / "plots" if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plots: list[Path] = []
    constraint_csv = run_dir / "constraint_utilization.csv"
    if constraint_csv.exists():
        plots.append(_write_constraint_utilization_plot(constraint_csv, out_dir / "constraint_utilization.svg"))

    quantities_csv = run_dir / "quantities.csv"
    if quantities_csv.exists():
        plots.extend(_write_joint_quantity_plots(quantities_csv, out_dir))
        ee_speed_plot = _write_end_effector_speed_plot(run_dir, quantities_csv, out_dir / "end_effector_speed.svg")
        if ee_speed_plot is not None:
            plots.append(ee_speed_plot)

    trajectory_csv = run_dir / "trajectory.csv"
    if trajectory_csv.exists():
        plots.append(_write_mvc_plot(run_dir, trajectory_csv, out_dir / "mvc.svg"))
        mvc_ee_plot = _write_mvc_end_effector_speed_plot(run_dir, trajectory_csv, out_dir / "mvc_end_effector_speed.svg")
        if mvc_ee_plot is not None:
            plots.append(mvc_ee_plot)

    batch_csv = run_dir / "batch_metrics.csv"
    if batch_csv.exists():
        plots.extend(_write_batch_metric_plots(batch_csv, out_dir))

    comparison_csv = run_dir / "comparison_metrics.csv"
    if comparison_csv.exists():
        plots.extend(_write_comparison_metric_plots(comparison_csv, out_dir))

    if not plots:
        raise ValueError(f"no supported result CSV files found in {run_dir}")

    manifest = {
        "run": str(run_dir),
        "plots": [path.name for path in plots],
    }
    (out_dir / "plots_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return plots


def _write_constraint_utilization_plot(csv_path: Path, out_path: Path) -> Path:
    rows = _read_csv(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} contains no rows")
    x_name = "s" if "s" in rows[0] else "t"
    xs = [_finite_float(row.get(x_name), default=float(index)) for index, row in enumerate(rows)]
    series = {
        name: [_finite_float(row.get(name), default=0.0) for row in rows]
        for name in rows[0]
        if name not in {"t", "s", "z"} and _column_has_numeric(rows, name)
    }
    if not series:
        raise ValueError(f"{csv_path} has no numeric utilization columns")
    out_path.write_text(
        _line_chart_svg(
            title="Constraint Utilization",
            x_label=x_name,
            y_label="normalized utilization",
            xs=xs,
            series=series,
            reference_y=1.0,
        ),
        encoding="utf-8",
    )
    return out_path


def _write_joint_quantity_plots(csv_path: Path, out_dir: Path) -> list[Path]:
    rows = _read_csv(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} contains no rows")
    x_name = "t" if "t" in rows[0] else "s"
    xs = [_finite_float(row.get(x_name), default=float(index)) for index, row in enumerate(rows)]
    specs = [
        ("joint_position.svg", "Joint Position", "rad", "q"),
        ("joint_velocity.svg", "Joint Velocity", "rad/s", "q_dot"),
        ("joint_acceleration.svg", "Joint Acceleration", "rad/s^2", "q_ddot"),
        ("joint_jerk.svg", "Joint Jerk", "rad/s^3", "q_jerk"),
        ("joint_torque.svg", "Joint Torque", "N*m", "tau"),
        ("joint_torque_rate.svg", "Joint Torque Rate", "N*m/s", "tau_rate"),
    ]
    outputs: list[Path] = []
    for filename, title, y_label, prefix in specs:
        columns = _axis_columns(rows[0], prefix)
        if not columns:
            continue
        series = {
            f"joint {axis}": [_finite_float(row.get(column), default=0.0) for row in rows]
            for axis, column in columns
        }
        out_path = out_dir / filename
        out_path.write_text(
            _line_chart_svg(title=title, x_label=x_name, y_label=y_label, xs=xs, series=series),
            encoding="utf-8",
        )
        outputs.append(out_path)
    return outputs


def _write_mvc_plot(run_dir: Path, csv_path: Path, out_path: Path) -> Path:
    rows = _read_csv(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} contains no rows")
    xs = [_finite_float(row.get("s"), default=float(index)) for index, row in enumerate(rows)]
    path_speed = [math.sqrt(max(0.0, _finite_float(row.get("z"), default=0.0))) for row in rows]
    summary = _load_summary(run_dir)
    chart_xs = xs
    series = {"path speed": path_speed}
    toppra_curve = _toppra_mvc_curve(summary, run_dir, xs)
    if toppra_curve is not None:
        chart_xs = toppra_curve.s
        series = {"path speed": _interpolated_values(xs, path_speed, chart_xs)}
        if toppra_curve.profile_speed is not None:
            series["TOPPRA profile"] = toppra_curve.profile_speed
        series["MVC (TOPPRA)"] = toppra_curve.mvc_speed
        z_max = _summary_z_max(summary)
        if z_max is not None:
            cap_speed = math.sqrt(max(0.0, z_max))
            if toppra_curve.mvc_speed and cap_speed < 0.98 * max(toppra_curve.mvc_speed):
                series["configured speed cap"] = [cap_speed for _ in chart_xs]
    else:
        mvc_speed = _mvc_speed_from_path_velocity_limits(summary, run_dir, xs)
        if mvc_speed is not None:
            series["MVC (q_dot limit)"] = mvc_speed
        else:
            z_max = _summary_z_max(summary)
            if z_max is not None:
                series["configured speed cap"] = [math.sqrt(max(0.0, z_max)) for _ in chart_xs]
    out_path.write_text(
        _line_chart_svg(title="MVC and Path Speed", x_label="s", y_label="path speed", xs=chart_xs, series=series),
        encoding="utf-8",
    )
    return out_path


def _write_mvc_end_effector_speed_plot(run_dir: Path, csv_path: Path, out_path: Path) -> Path | None:
    rows = _read_csv(csv_path)
    if len(rows) < 2:
        return None
    trajectory_s = [_finite_float(row.get("s"), default=float(index)) for index, row in enumerate(rows)]
    path_speed = [math.sqrt(max(0.0, _finite_float(row.get("z"), default=0.0))) for row in rows]
    summary = _load_summary(run_dir)
    toppra_curve = _toppra_mvc_curve(summary, run_dir, trajectory_s)
    if toppra_curve is None:
        return None
    chart_xs = toppra_curve.s
    tcp_speed_factor = _tcp_speed_factor_from_path(summary, run_dir, chart_xs)
    if tcp_speed_factor is None:
        return None
    actual_path_speed = _interpolated_values(trajectory_s, path_speed, chart_xs)
    series = {
        "TCP speed (DP3)": [speed * factor for speed, factor in zip(actual_path_speed, tcp_speed_factor)],
        "MVC (TOPPRA, TCP)": [speed * factor for speed, factor in zip(toppra_curve.mvc_speed, tcp_speed_factor)],
    }
    out_path.write_text(
        _line_chart_svg(
            title="MVC + End-Effector Speed",
            x_label="s",
            y_label="TCP speed (m/s)",
            xs=chart_xs,
            series=series,
        ),
        encoding="utf-8",
    )
    return out_path


def _write_end_effector_speed_plot(run_dir: Path, csv_path: Path, out_path: Path) -> Path | None:
    rows = _read_csv(csv_path)
    if len(rows) < 2:
        return None
    q_columns = _axis_columns(rows[0], "q")
    if not q_columns:
        return None
    summary = _load_summary(run_dir)
    model_path = _summary_model_path(summary, run_dir)
    if model_path is None:
        return None
    site_name = _summary_site_name(summary)
    try:
        speed = _mujoco_site_speed(model_path=model_path, site_name=site_name, rows=rows, q_columns=q_columns)
    except Exception:
        return None
    xs = [_finite_float(row.get("t"), default=float(index)) for index, row in enumerate(rows)]
    out_path.write_text(
        _line_chart_svg(
            title="End-Effector Speed",
            x_label="t",
            y_label="m/s",
            xs=xs,
            series={f"{site_name} speed": speed},
        ),
        encoding="utf-8",
    )
    return out_path


def _write_batch_metric_plots(csv_path: Path, out_dir: Path) -> list[Path]:
    rows = _read_csv(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} contains no rows")
    ids = _row_ids(rows)
    outputs: list[Path] = []
    times_path = out_dir / "batch_times.svg"
    times_path.write_text(
        _grouped_bar_chart_svg(
            title="Batch Times",
            y_label="seconds",
            categories=ids,
            groups=[
                ("t_e_s", [_finite_float(row.get("t_e_s"), default=0.0) for row in rows]),
                ("t_cpu_s", [_finite_float(row.get("t_cpu_s"), default=0.0) for row in rows]),
            ],
        ),
        encoding="utf-8",
    )
    outputs.append(times_path)
    violations_path = out_dir / "batch_violations.svg"
    violations_path.write_text(
        _grouped_bar_chart_svg(
            title="Batch Violations",
            y_label="count",
            categories=ids,
            groups=[("violation_count", [_finite_float(row.get("violation_count"), default=0.0) for row in rows])],
        ),
        encoding="utf-8",
    )
    outputs.append(violations_path)
    return outputs


def _write_comparison_metric_plots(csv_path: Path, out_dir: Path) -> list[Path]:
    rows = _read_csv(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} contains no rows")
    ids = _row_ids(rows)
    specs = [
        (
            "comparison_times.svg",
            "DP3 vs DP2 Trajectory Time",
            "seconds",
            [
                ("dp3_t_e_s", [_finite_float(row.get("dp3_t_e_s"), default=0.0) for row in rows]),
                ("dp2_t_e_s", [_finite_float(row.get("dp2_t_e_s"), default=0.0) for row in rows]),
            ],
        ),
        (
            "comparison_cpu.svg",
            "DP3 vs DP2 CPU Time",
            "seconds",
            [
                ("dp3_t_cpu_s", [_finite_float(row.get("dp3_t_cpu_s"), default=0.0) for row in rows]),
                ("dp2_t_cpu_s", [_finite_float(row.get("dp2_t_cpu_s"), default=0.0) for row in rows]),
            ],
        ),
        (
            "comparison_violations.svg",
            "DP3 vs DP2 Violations",
            "count",
            [
                ("dp3_violation_count", [_finite_float(row.get("dp3_violation_count"), default=0.0) for row in rows]),
                ("dp2_violation_count", [_finite_float(row.get("dp2_violation_count"), default=0.0) for row in rows]),
            ],
        ),
    ]
    outputs: list[Path] = []
    for filename, title, y_label, groups in specs:
        out_path = out_dir / filename
        out_path.write_text(
            _grouped_bar_chart_svg(title=title, y_label=y_label, categories=ids, groups=groups),
            encoding="utf-8",
        )
        outputs.append(out_path)
    return outputs


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _row_ids(rows: list[dict[str, str]]) -> list[str]:
    return [row.get("id") or f"row_{index + 1}" for index, row in enumerate(rows)]


def _axis_columns(header_row: dict[str, str], prefix: str) -> list[tuple[int, str]]:
    if prefix == "q":
        pattern = re.compile(r"^q([0-9]+)$")
    else:
        pattern = re.compile(rf"^{re.escape(prefix)}([0-9]+)$")
    columns: list[tuple[int, str]] = []
    for name in header_row:
        match = pattern.match(name)
        if match:
            columns.append((int(match.group(1)), name))
    return sorted(columns)


def _column_has_numeric(rows: list[dict[str, str]], name: str) -> bool:
    return any(_maybe_float(row.get(name)) is not None for row in rows)


def _finite_float(raw: str | None, *, default: float) -> float:
    value = _maybe_float(raw)
    return default if value is None else value


def _maybe_float(raw: str | None) -> float | None:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _load_summary(run_dir: Path) -> dict:
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _summary_z_max(summary: dict) -> float | None:
    config = summary.get("config", {}) if isinstance(summary.get("config"), dict) else {}
    return _maybe_float(config.get("z_max"))


def _summary_z_boundary(summary: dict, name: str) -> float:
    config = summary.get("config", {}) if isinstance(summary.get("config"), dict) else {}
    return max(0.0, _maybe_float(config.get(name)) or 0.0)


def _summary_site_name(summary: dict) -> str:
    ik = summary.get("ik", {}) if isinstance(summary.get("ik"), dict) else {}
    settings = ik.get("settings", {}) if isinstance(ik.get("settings"), dict) else {}
    raw = settings.get("ee_site") or summary.get("ee_site") or "tcp"
    return str(raw)


def _summary_model_path(summary: dict, run_dir: Path) -> Path | None:
    return _summary_source_path(summary, run_dir, "model")


def _summary_source_path(summary: dict, run_dir: Path, key: str) -> Path | None:
    sources = summary.get("reproduction_sources", {}) if isinstance(summary.get("reproduction_sources"), dict) else {}
    raw = sources.get(key) or summary.get(key)
    if not raw:
        return None
    source_path = Path(str(raw))
    candidates = [source_path]
    if not source_path.is_absolute():
        candidates.extend([run_dir / source_path, Path.cwd() / source_path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _interpolated_values(source_xs: list[float], source_ys: list[float], target_xs: list[float]) -> list[float]:
    try:
        import numpy as np

        source_x = np.asarray(source_xs, dtype=float)
        source_y = np.asarray(source_ys, dtype=float)
        target_x = np.asarray(target_xs, dtype=float)
        if (
            source_x.ndim != 1
            or source_y.shape != source_x.shape
            or target_x.ndim != 1
            or source_x.size < 2
            or np.any(~np.isfinite(source_x))
            or np.any(~np.isfinite(source_y))
            or np.any(~np.isfinite(target_x))
            or np.any(np.diff(source_x) <= 0.0)
        ):
            return list(source_ys)
        return [float(value) for value in np.interp(target_x, source_x, source_y)]
    except Exception:
        return list(source_ys)


def _mvc_speed_from_path_velocity_limits(summary: dict, run_dir: Path, samples_s: list[float]) -> list[float] | None:
    path_source = _summary_source_path(summary, run_dir, "path_source")
    limits_source = _summary_source_path(summary, run_dir, "limits")
    if path_source is None or limits_source is None:
        return None
    try:
        import numpy as np

        from dp3_topp.constraints import ConstraintLimits
        from dp3_topp.path_data import PathData

        path = PathData.from_csv(path_source)
        limits = ConstraintLimits.from_yaml(limits_source)
        if path.dof != limits.dof:
            return None
        sample_arr = np.asarray(samples_s, dtype=float)
        q_s = np.column_stack([np.interp(sample_arr, path.s, path.q_s[:, axis]) for axis in range(path.dof)])
        lower, upper = limits.q_dot_bounds(q_s.shape)
        speed_limits: list[float] = []
        for sample_index, row in enumerate(q_s):
            candidates: list[float] = []
            for axis, derivative in enumerate(row):
                derivative = float(derivative)
                if abs(derivative) <= 1e-12:
                    continue
                speed_limit = (
                    float(upper[sample_index, axis]) / derivative
                    if derivative > 0.0
                    else float(lower[sample_index, axis]) / derivative
                )
                if math.isfinite(speed_limit):
                    candidates.append(max(0.0, speed_limit))
            speed_limits.append(min(candidates) if candidates else 0.0)
        return speed_limits
    except Exception:
        return None


def _toppra_mvc_speed(summary: dict, run_dir: Path, samples_s: list[float]) -> list[float] | None:
    curve = _toppra_mvc_curve(summary, run_dir, samples_s)
    if curve is None:
        return None
    return _interpolated_values(curve.s, curve.mvc_speed, samples_s)


def _toppra_mvc_curve(summary: dict, run_dir: Path, samples_s: list[float]) -> ToppraCurve | None:
    path_source = _summary_source_path(summary, run_dir, "path_source")
    limits_source = _summary_source_path(summary, run_dir, "limits")
    if path_source is None or limits_source is None:
        return None
    try:
        import numpy as np
        import toppra as ta  # type: ignore
        import toppra.algorithm as toppra_algorithm  # type: ignore
        import toppra.constraint as toppra_constraint  # type: ignore

        from dp3_topp.constraints import ConstraintLimits
        from dp3_topp.path_data import PathData

        path = PathData.from_csv(path_source)
        limits = ConstraintLimits.from_yaml(limits_source)
        if path.dof != limits.dof:
            return None
        sample_arr = np.asarray(samples_s, dtype=float)
        if (
            sample_arr.ndim != 1
            or sample_arr.size < 2
            or np.any(~np.isfinite(sample_arr))
            or np.any(np.diff(sample_arr) <= 0.0)
        ):
            return None
        grid_count = min(max(int(path.s.size), int(sample_arr.size), 101), 501)
        gridpoints = np.linspace(float(path.s[0]), float(path.s[-1]), grid_count)
        if gridpoints.ndim != 1 or gridpoints.size < 2 or np.any(~np.isfinite(gridpoints)) or np.any(np.diff(gridpoints) <= 0.0):
            return None
        toppra_path = ta.SplineInterpolator(path.s, path.q)
        constraints = [
            toppra_constraint.JointVelocityConstraint(np.column_stack([limits.q_dot_lower, limits.q_dot_upper])),
            toppra_constraint.JointAccelerationConstraint(np.column_stack([limits.q_ddot_lower, limits.q_ddot_upper])),
        ]
        model_path = _summary_model_path(summary, run_dir)
        if model_path is not None:
            try:
                from dp3_topp.dynamics_mujoco import MujocoRobotDynamics

                robot = MujocoRobotDynamics.from_model_path(model_path)
                if robot.dof == path.dof:
                    constraints.append(
                        toppra_constraint.JointTorqueConstraint(
                            robot.inverse_dynamics,
                            np.column_stack([limits.tau_lower, limits.tau_upper]),
                            limits.friction_coulomb,
                        )
                    )
            except Exception:
                pass
        instance = toppra_algorithm.TOPPRA(constraints, toppra_path, gridpoints=gridpoints, solver_wrapper="seidel")
        feasible_sets = instance.compute_feasible_sets()
        upper = np.asarray(feasible_sets[:, 1], dtype=float)
        if upper.shape != gridpoints.shape or np.any(~np.isfinite(upper)):
            return None
        profile_speed = None
        start_speed = math.sqrt(max(0.0, _summary_z_boundary(summary, "z_start")))
        end_speed = math.sqrt(max(0.0, _summary_z_boundary(summary, "z_end")))
        try:
            _, sd_vec, _, _ = instance.compute_parameterization(start_speed, end_speed, return_data=True)
            if sd_vec is not None:
                sd_vec = np.asarray(sd_vec, dtype=float)
                if sd_vec.shape == gridpoints.shape and np.all(np.isfinite(sd_vec)):
                    profile_speed = [float(value) for value in sd_vec]
        except Exception:
            profile_speed = None
        return ToppraCurve(
            s=[float(value) for value in gridpoints],
            mvc_speed=[float(value) for value in np.sqrt(np.maximum(upper, 0.0))],
            profile_speed=profile_speed,
        )
    except Exception:
        return None


def _tcp_speed_factor_from_path(summary: dict, run_dir: Path, samples_s: list[float]) -> list[float] | None:
    path_source = _summary_source_path(summary, run_dir, "path_source")
    model_path = _summary_model_path(summary, run_dir)
    if path_source is None or model_path is None:
        return None
    try:
        import mujoco  # type: ignore
        import numpy as np

        from dp3_topp.dynamics_mujoco import MujocoRobotDynamics
        from dp3_topp.path_data import PathData

        path = PathData.from_csv(path_source)
        sample_arr = np.asarray(samples_s, dtype=float)
        if sample_arr.ndim != 1 or sample_arr.size < 2 or np.any(np.diff(sample_arr) <= 0.0):
            return None
        q_samples = np.column_stack([np.interp(sample_arr, path.s, path.q[:, axis]) for axis in range(path.dof)])
        robot = MujocoRobotDynamics.from_model_path(model_path)
        if q_samples.shape[1] != robot.dof:
            return None
        site_name = _summary_site_name(summary)
        site_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            return None
        data = mujoco.MjData(robot.model)
        positions = []
        for q in q_samples:
            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            for axis_index, value in enumerate(q):
                data.qpos[robot.qpos_indices[axis_index]] = float(value)
            mujoco.mj_forward(robot.model, data)
            positions.append(np.asarray(data.site_xpos[site_id], dtype=float).copy())
        positions_arr = np.vstack(positions)
        edge_order = 2 if sample_arr.size > 2 else 1
        dx_ds = np.gradient(positions_arr, sample_arr, axis=0, edge_order=edge_order)
        speed_factor = np.linalg.norm(dx_ds, axis=1)
        if speed_factor.shape != sample_arr.shape or np.any(~np.isfinite(speed_factor)):
            return None
        return [float(value) for value in speed_factor]
    except Exception:
        return None


def _mujoco_site_speed(
    *,
    model_path: Path,
    site_name: str,
    rows: list[dict[str, str]],
    q_columns: list[tuple[int, str]],
) -> list[float]:
    import mujoco  # type: ignore
    import numpy as np

    from dp3_topp.dynamics_mujoco import MujocoRobotDynamics

    robot = MujocoRobotDynamics.from_model_path(model_path)
    if len(q_columns) != robot.dof:
        raise ValueError("quantity DOF does not match MuJoCo model DOF")
    site_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"MuJoCo site not found: {site_name}")
    data = mujoco.MjData(robot.model)
    positions = []
    times = []
    for sample, row in enumerate(rows):
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        for axis_index, (_, column) in enumerate(q_columns):
            data.qpos[robot.qpos_indices[axis_index]] = _finite_float(row.get(column), default=0.0)
        mujoco.mj_forward(robot.model, data)
        positions.append(np.asarray(data.site_xpos[site_id], dtype=float).copy())
        times.append(_finite_float(row.get("t"), default=float(sample)))
    positions_arr = np.vstack(positions)
    times_arr = np.asarray(times, dtype=float)
    speeds = []
    for index in range(len(rows)):
        if index == 0:
            dt = times_arr[1] - times_arr[0]
            delta = positions_arr[1] - positions_arr[0]
        elif index == len(rows) - 1:
            dt = times_arr[-1] - times_arr[-2]
            delta = positions_arr[-1] - positions_arr[-2]
        else:
            dt = times_arr[index + 1] - times_arr[index - 1]
            delta = positions_arr[index + 1] - positions_arr[index - 1]
        speeds.append(0.0 if abs(float(dt)) < 1e-12 else float(np.linalg.norm(delta) / abs(float(dt))))
    return speeds


def _line_chart_svg(
    *,
    title: str,
    x_label: str,
    y_label: str,
    xs: list[float],
    series: dict[str, list[float]],
    reference_y: float | None = None,
) -> str:
    width = 960
    height = 540
    left = 72
    right = 190
    top = 54
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = _extent(xs, include_zero=False)
    values = [value for items in series.values() for value in items]
    y_min, y_max = _extent(values + ([reference_y] if reference_y is not None else []), include_zero=True)

    def px(value: float) -> float:
        return left + _ratio(value, x_min, x_max) * plot_w

    def py(value: float) -> float:
        return top + plot_h - _ratio(value, y_min, y_max) * plot_h

    parts = [_svg_header(width, height), _title(title, width)]
    parts.append(_axes(left, top, plot_w, plot_h, x_label, y_label))
    parts.extend(_y_ticks(left, top, plot_h, y_min, y_max))
    if reference_y is not None and y_min <= reference_y <= y_max:
        y_ref = py(reference_y)
        parts.append(f'<line x1="{left}" y1="{y_ref:.3f}" x2="{left + plot_w}" y2="{y_ref:.3f}" stroke="#777" stroke-dasharray="6 6"/>')
        parts.append(f'<text x="{left + plot_w + 8}" y="{y_ref + 4:.3f}" font-size="12" fill="#555">limit {reference_y:g}</text>')
    for index, (name, values_for_name) in enumerate(series.items()):
        color = COLORS[index % len(COLORS)]
        points = " ".join(
            f"{px(x):.3f},{py(y):.3f}"
            for x, y in zip(xs, values_for_name, strict=False)
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        parts.append(_legend_item(width - right + 22, top + 24 + index * 22, color, name))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _grouped_bar_chart_svg(
    *,
    title: str,
    y_label: str,
    categories: list[str],
    groups: list[tuple[str, list[float]]],
) -> str:
    category_count = max(1, len(categories))
    width = max(760, 150 + 74 * category_count + 190)
    height = 520
    left = 72
    right = 190
    top = 54
    bottom = 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [max(0.0, value) for _, group_values in groups for value in group_values]
    _, y_max = _extent(values, include_zero=True)
    y_max = y_max if y_max > 0.0 else 1.0

    def py(value: float) -> float:
        return top + plot_h - (max(0.0, value) / y_max) * plot_h

    parts = [_svg_header(width, height), _title(title, width)]
    parts.append(_axes(left, top, plot_w, plot_h, "path", y_label))
    parts.extend(_y_ticks(left, top, plot_h, 0.0, y_max))
    slot_w = plot_w / category_count
    group_count = max(1, len(groups))
    bar_w = min(22.0, slot_w / (group_count + 1.5))
    for category_index, category in enumerate(categories):
        center = left + slot_w * (category_index + 0.5)
        group_start = center - (group_count * bar_w) / 2.0
        for group_index, (label, group_values) in enumerate(groups):
            value = group_values[category_index] if category_index < len(group_values) else 0.0
            x = group_start + group_index * bar_w
            y = py(value)
            h = top + plot_h - y
            color = COLORS[group_index % len(COLORS)]
            parts.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{bar_w * 0.82:.3f}" height="{h:.3f}" fill="{color}"/>')
        parts.append(
            f'<text x="{center:.3f}" y="{height - 38}" font-size="11" text-anchor="end" '
            f'transform="rotate(-35 {center:.3f} {height - 38})">{escape(category)}</text>'
        )
    for index, (label, _) in enumerate(groups):
        parts.append(_legend_item(width - right + 22, top + 24 + index * 22, COLORS[index % len(COLORS)], label))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _extent(values: Iterable[float], *, include_zero: bool) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if include_zero:
        finite.append(0.0)
    if not finite:
        return 0.0, 1.0
    lo = min(finite)
    hi = max(finite)
    if math.isclose(lo, hi):
        pad = 1.0 if math.isclose(lo, 0.0) else abs(lo) * 0.1
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    if include_zero and lo >= 0.0:
        return 0.0, hi + pad
    return lo - pad, hi + pad


def _ratio(value: float, lo: float, hi: float) -> float:
    if math.isclose(lo, hi):
        return 0.5
    return (float(value) - lo) / (hi - lo)


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        '<rect width="100%" height="100%" fill="#fff"/>'
    )


def _title(title: str, width: int) -> str:
    return f'<text x="{width / 2:.1f}" y="28" font-size="20" font-weight="700" text-anchor="middle">{escape(title)}</text>'


def _axes(left: int, top: int, plot_w: int, plot_h: int, x_label: str, y_label: str) -> str:
    bottom_y = top + plot_h
    return "\n".join(
        [
            f'<line x1="{left}" y1="{bottom_y}" x2="{left + plot_w}" y2="{bottom_y}" stroke="#222"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom_y}" stroke="#222"/>',
            f'<text x="{left + plot_w / 2:.1f}" y="{bottom_y + 54}" font-size="13" text-anchor="middle">{escape(x_label)}</text>',
            f'<text x="18" y="{top + plot_h / 2:.1f}" font-size="13" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2:.1f})">{escape(y_label)}</text>',
        ]
    )


def _y_ticks(left: int, top: int, plot_h: int, y_min: float, y_max: float) -> list[str]:
    ticks = []
    for index in range(5):
        ratio = index / 4.0
        value = y_min + (y_max - y_min) * ratio
        y = top + plot_h - ratio * plot_h
        ticks.append(f'<line x1="{left - 4}" y1="{y:.3f}" x2="{left}" y2="{y:.3f}" stroke="#222"/>')
        ticks.append(f'<text x="{left - 8}" y="{y + 4:.3f}" font-size="11" text-anchor="end">{value:.3g}</text>')
        ticks.append(f'<line x1="{left}" y1="{y:.3f}" x2="{left + 8}" y2="{y:.3f}" stroke="#ddd"/>')
    return ticks


def _legend_item(x: float, y: float, color: str, label: str) -> str:
    return (
        f'<rect x="{x:.3f}" y="{y - 10:.3f}" width="12" height="12" fill="{color}"/>'
        f'<text x="{x + 18:.3f}" y="{y:.3f}" font-size="12">{escape(label)}</text>'
    )
