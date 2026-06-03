from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dp3_topp import ConstraintLimits, DP3Config, PathData, run_dp3
from dp3_topp.plotting import plot_run


QUANTITY_PLOTS = (
    "joint_velocity.svg",
    "joint_acceleration.svg",
    "joint_jerk.svg",
    "joint_torque.svg",
    "joint_torque_rate.svg",
    "constraint_utilization.svg",
)

PUBLISHED_PLOTS = (
    "joint_velocity.svg",
    "joint_acceleration.svg",
    "joint_jerk.svg",
    "joint_torque.svg",
    "joint_torque_rate.svg",
)


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    torque_scale: float = 1.0
    jerk_scale: float = 1.0
    velocity_dependent_torque: bool = False


@dataclass(frozen=True)
class PathCase:
    id: str
    csv_path: Path
    joint_length: float
    max_q_s: float
    max_q_ss: float
    max_q_sss: float


SCENARIOS = (
    Scenario(
        id="nominal",
        title="Nominal torque and jerk constraints",
    ),
    Scenario(
        id="torque_speed_drop",
        title="Velocity-dependent torque drop",
        torque_scale=0.70,
        velocity_dependent_torque=True,
    ),
    Scenario(
        id="tight_jerk",
        title="Tight jerk constraint",
        jerk_scale=0.09,
    ),
    Scenario(
        id="combined_torque_jerk",
        title="Combined torque drop and tight jerk",
        torque_scale=0.70,
        jerk_scale=0.12,
        velocity_dependent_torque=True,
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a multi-path DP3 torque/jerk constraint sweep and plot joint data.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/runs/dp3-complex-constraint-sweep"))
    parser.add_argument("--asset-dir", type=Path, default=Path("assets/complex-constraint-sweep"))
    parser.add_argument("--path-count", type=int, default=4)
    parser.add_argument("--constraint-check-points", type=int, default=240)
    parser.add_argument("--time-samples", type=int, default=180)
    parser.add_argument("--regular-ns", type=int, default=8)
    parser.add_argument("--regular-nz", type=int, default=25)
    parser.add_argument("--regular-nch", type=int, default=6)
    parser.add_argument("--long-ns", type=int, default=12)
    parser.add_argument("--long-nz", type=int, default=60)
    parser.add_argument("--long-nch", type=int, default=10)
    parser.add_argument("--z-max", type=float, default=1.0)
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing run directories when summary.json is present.")
    args = parser.parse_args(argv)

    root = Path.cwd()
    dyn_dir = _find_dyn_dir(root)
    paths_dir = dyn_dir / "paths"
    model_path = dyn_dir / "models" / "T12A" / "T12A-14.xml"
    limits_path = dyn_dir / "models" / "T12A" / "limits.yaml"
    base_limits = ConstraintLimits.from_yaml(limits_path)
    path_cases = select_complex_paths(paths_dir, max(1, int(args.path_count)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for path_case in path_cases:
        path = PathData.from_csv(path_case.csv_path)
        config = _config_for_path(path_case, args)
        for scenario in SCENARIOS:
            run_dir = args.out_dir / path_case.id / scenario.id
            if not (args.skip_existing and (run_dir / "summary.json").exists()):
                scenario_limits = build_scenario_limits(base_limits, scenario)
                result = run_dp3(
                    path=path,
                    limits=scenario_limits,
                    model=model_path,
                    config=config,
                    out_dir=run_dir,
                    constraint_check_points=int(args.constraint_check_points),
                    time_samples=int(args.time_samples),
                )
                summary = result.summary
            else:
                summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            plot_run(run_dir)
            rows.append(_summary_row(path_case, scenario, config, run_dir, summary))
            _publish_plots(args.asset_dir, run_dir, path_case.id, scenario.id)

    _write_summary_csv(args.out_dir / "sweep_summary.csv", rows)
    (args.out_dir / "sweep_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _write_gallery(args.out_dir / "gallery.html", rows, args.out_dir)
    _write_asset_index(args.asset_dir / "index.html", rows)
    print(f"summary: {args.out_dir / 'sweep_summary.csv'}")
    print(f"gallery: {args.out_dir / 'gallery.html'}")
    print(f"published assets: {args.asset_dir}")
    return 0


def _find_dyn_dir(root: Path) -> Path:
    candidates = [path for path in root.glob("dyn*") if path.is_dir() and (path / "paths").exists()]
    if not candidates:
        raise FileNotFoundError("could not find dyn data directory")
    return candidates[0]


def select_complex_paths(paths_dir: Path, count: int) -> list[PathCase]:
    cases = [_path_case(path) for path in sorted(paths_dir.glob("path_*.csv"))]
    long_path = paths_dir / "long_path_01.csv"
    selected: list[PathCase] = []
    if long_path.exists():
        selected.append(_path_case(long_path))
    regular = [case for case in cases if case.id != "long_path_01"]
    regular.sort(key=lambda item: (item.joint_length, item.max_q_sss, item.max_q_ss), reverse=True)
    selected.extend(regular[: max(0, count - len(selected))])
    return selected[:count]


def _path_case(path: Path) -> PathCase:
    data = PathData.from_csv(path)
    return PathCase(
        id=path.stem,
        csv_path=path,
        joint_length=float(np.sum(np.linalg.norm(np.diff(data.q, axis=0), axis=1))),
        max_q_s=float(np.max(np.abs(data.q_s))),
        max_q_ss=float(np.max(np.abs(data.q_ss))),
        max_q_sss=float(np.max(np.abs(data.q_sss))),
    )


def _config_for_path(path_case: PathCase, args: argparse.Namespace) -> DP3Config:
    if path_case.id == "long_path_01":
        return DP3Config(ns=int(args.long_ns), nz=int(args.long_nz), nch=int(args.long_nch), z_max=float(args.z_max))
    return DP3Config(ns=int(args.regular_ns), nz=int(args.regular_nz), nch=int(args.regular_nch), z_max=float(args.z_max))


def build_scenario_limits(base: ConstraintLimits, scenario: Scenario) -> ConstraintLimits:
    torque_abs = base.tau_abs * float(scenario.torque_scale)
    tau_lower = base.tau_lower * float(scenario.torque_scale)
    tau_upper = base.tau_upper * float(scenario.torque_scale)
    q_jerk_abs = base.q_jerk_abs * float(scenario.jerk_scale)
    q_jerk_lower = base.q_jerk_lower * float(scenario.jerk_scale)
    q_jerk_upper = base.q_jerk_upper * float(scenario.jerk_scale)
    torque_speed = (
        _velocity_dependent_torque_tables(torque_abs)
        if scenario.velocity_dependent_torque
        else base.torque_speed_breakpoints
    )
    return ConstraintLimits(
        q_dot_abs=base.q_dot_abs,
        q_ddot_abs=base.q_ddot_abs,
        q_jerk_abs=q_jerk_abs,
        tau_abs=torque_abs,
        tau_rate_abs=base.tau_rate_abs,
        mechanical_power_lower=base.mechanical_power_lower,
        mechanical_power_upper=base.mechanical_power_upper,
        torque_speed_breakpoints=torque_speed,
        friction_coulomb=base.friction_coulomb,
        friction_viscous=base.friction_viscous,
        motor_gear_ratio=base.motor_gear_ratio,
        motor_torque_constant=base.motor_torque_constant,
        motor_stator_resistance=base.motor_stator_resistance,
        q_dot_lower=base.q_dot_lower,
        q_dot_upper=base.q_dot_upper,
        q_ddot_lower=base.q_ddot_lower,
        q_ddot_upper=base.q_ddot_upper,
        q_jerk_lower=q_jerk_lower,
        q_jerk_upper=q_jerk_upper,
        tau_lower=tau_lower,
        tau_upper=tau_upper,
        tau_rate_lower=base.tau_rate_lower,
        tau_rate_upper=base.tau_rate_upper,
        q_position_lower=base.q_position_lower,
        q_position_upper=base.q_position_upper,
    )


def _velocity_dependent_torque_tables(torque_abs: np.ndarray) -> list[np.ndarray]:
    return [
        np.array(
            [
                [0.0, float(limit)],
                [2.5, 0.78 * float(limit)],
                [5.0, 0.58 * float(limit)],
                [8.0, 0.42 * float(limit)],
            ],
            dtype=np.float64,
        )
        for limit in torque_abs
    ]


def _summary_row(path_case: PathCase, scenario: Scenario, config: DP3Config, run_dir: Path, summary: dict) -> dict[str, object]:
    max_util = summary.get("max_utilization", {}) or {}
    return {
        "path_id": path_case.id,
        "scenario_id": scenario.id,
        "scenario": scenario.title,
        "run_dir": str(run_dir),
        "status": int(summary.get("feasible") is True),
        "feasible": bool(summary.get("feasible", False)),
        "total_time_s": _finite_or_blank(summary.get("total_time")),
        "segment_count": int(summary.get("segment_count", 0)),
        "joint_length": path_case.joint_length,
        "max_q_s": path_case.max_q_s,
        "max_q_ss": path_case.max_q_ss,
        "max_q_sss": path_case.max_q_sss,
        "ns": int(config.ns),
        "nz": int(config.nz),
        "nch": int(config.nch),
        "z_max": float(config.z_max or 0.0),
        "torque_scale": float(scenario.torque_scale),
        "jerk_scale": float(scenario.jerk_scale),
        "velocity_dependent_torque": bool(scenario.velocity_dependent_torque),
        "max_utilization_q_dot": _finite_or_blank(max_util.get("q_dot")),
        "max_utilization_q_ddot": _finite_or_blank(max_util.get("q_ddot")),
        "max_utilization_q_jerk": _finite_or_blank(max_util.get("q_jerk")),
        "max_utilization_tau": _finite_or_blank(max_util.get("tau")),
        "max_utilization_tau_rate": _finite_or_blank(max_util.get("tau_rate")),
        "max_utilization_mechanical_power": _finite_or_blank(max_util.get("mechanical_power")),
        "violation_count": len(summary.get("violations", []) or []),
    }


def _finite_or_blank(value: object) -> float | str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ""
    return out if np.isfinite(out) else ""


def _publish_plots(asset_dir: Path, run_dir: Path, path_id: str, scenario_id: str) -> None:
    plot_dir = run_dir / "plots"
    for filename in PUBLISHED_PLOTS:
        source = plot_dir / filename
        if not source.exists():
            continue
        target = asset_dir / f"{path_id}_{scenario_id}_{filename}"
        shutil.copyfile(source, target)


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_gallery(path: Path, rows: list[dict[str, object]], out_dir: Path) -> None:
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>DP3 complex constraint sweep</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#111827}table{border-collapse:collapse;margin-bottom:24px}td,th{border:1px solid #d1d5db;padding:6px 8px;font-size:12px}.run{margin:28px 0}.grid{display:grid;grid-template-columns:repeat(2,minmax(320px,1fr));gap:16px}.grid img{width:100%;border:1px solid #e5e7eb}h1,h2{margin-bottom:8px}</style>",
        "</head>",
        "<body>",
        "<h1>DP3 complex torque and jerk constraint sweep</h1>",
        "<table><thead><tr><th>path</th><th>scenario</th><th>feasible</th><th>time</th><th>tau util</th><th>jerk util</th><th>violations</th></tr></thead><tbody>",
    ]
    for row in rows:
        parts.append(
            "<tr>"
            f"<td>{row['path_id']}</td><td>{row['scenario_id']}</td><td>{row['feasible']}</td>"
            f"<td>{row['total_time_s']}</td><td>{row['max_utilization_tau']}</td>"
            f"<td>{row['max_utilization_q_jerk']}</td><td>{row['violation_count']}</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")
    for row in rows:
        run_dir = Path(str(row["run_dir"]))
        rel_run = _relpath(run_dir, path.parent)
        parts.append(f"<section class='run'><h2>{row['path_id']} / {row['scenario_id']}</h2>")
        parts.append(f"<p>Run directory: <code>{rel_run}</code></p><div class='grid'>")
        for filename in QUANTITY_PLOTS:
            plot_path = run_dir / "plots" / filename
            if plot_path.exists():
                parts.append(f"<figure><img src='{_relpath(plot_path, path.parent)}' alt='{filename}'><figcaption>{filename}</figcaption></figure>")
        parts.append("</div></section>")
    parts.extend(["</body>", "</html>"])
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _write_asset_index(path: Path, rows: list[dict[str, object]]) -> None:
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head><meta charset='utf-8'><title>DP3 sweep published joint plots</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#111827}.grid{display:grid;grid-template-columns:repeat(2,minmax(320px,1fr));gap:16px}.grid img{width:100%;border:1px solid #e5e7eb}</style></head>",
        "<body><h1>DP3 sweep published joint plots</h1><div class='grid'>",
    ]
    for row in rows:
        for filename in PUBLISHED_PLOTS:
            asset_name = f"{row['path_id']}_{row['scenario_id']}_{filename}"
            if (path.parent / asset_name).exists():
                parts.append(f"<figure><img src='{asset_name}' alt='{asset_name}'><figcaption>{asset_name}</figcaption></figure>")
    parts.extend(["</div></body>", "</html>"])
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _relpath(path: Path, start: Path) -> str:
    return Path(path).resolve().relative_to(Path(start).resolve()).as_posix() if _is_relative_to(Path(path).resolve(), Path(start).resolve()) else Path(path).as_posix()


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
