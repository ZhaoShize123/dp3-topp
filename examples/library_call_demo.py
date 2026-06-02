from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dp3_topp import ConstraintLimits, DP3Config, PathData, optimize_dp3, run_dp3


def build_demo_path() -> PathData:
    s = np.linspace(0.0, 1.0, 21)
    q = np.column_stack(
        [
            s,
            0.4 * np.sin(np.pi * s),
        ]
    )
    q_s = np.column_stack(
        [
            np.ones_like(s),
            0.4 * np.pi * np.cos(np.pi * s),
        ]
    )
    q_ss = np.column_stack(
        [
            np.zeros_like(s),
            -0.4 * np.pi**2 * np.sin(np.pi * s),
        ]
    )
    q_sss = np.column_stack(
        [
            np.zeros_like(s),
            -0.4 * np.pi**3 * np.cos(np.pi * s),
        ]
    )
    return PathData(s=s, q=q, q_s=q_s, q_ss=q_ss, q_sss=q_sss)


def build_demo_limits() -> ConstraintLimits:
    return ConstraintLimits(
        q_dot_abs=np.array([2.0, 2.0]),
        q_ddot_abs=np.array([20.0, 20.0]),
        q_jerk_abs=np.array([250.0, 250.0]),
        tau_abs=np.array([100.0, 100.0]),
        tau_rate_abs=np.array([1000.0, 1000.0]),
        mechanical_power_lower=-200.0,
        mechanical_power_upper=200.0,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a minimal dp3-topp library-call demo.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/demo_library_call"))
    args = parser.parse_args(argv)

    path = build_demo_path()
    limits = build_demo_limits()
    config = DP3Config(ns=8, nz=12, nch=7)

    low_level = optimize_dp3(path=path, limits=limits, config=config)
    print(f"low-level optimize_dp3: feasible={low_level.feasible} total_time={low_level.total_time:.6g}s")

    run = run_dp3(
        path=path,
        limits=limits,
        config=config,
        out_dir=args.out_dir,
        constraint_check_points=41,
        time_samples=21,
    )
    print(f"high-level run_dp3: status={run.status} feasible={run.summary['feasible']}")
    print(f"summary: {run.summary_path}")
    print(f"trajectory: {args.out_dir / 'trajectory.csv'}")
    return int(run.status)


if __name__ == "__main__":
    raise SystemExit(main())
