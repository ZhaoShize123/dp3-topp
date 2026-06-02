from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dp3_topp.constraints import (
    ConstraintAudit,
    ConstraintLimits,
    audit_constraints,
    interval_bound_utilization,
    signed_bound_utilization,
)
from dp3_topp.dynamics_mujoco import MujocoRobotDynamics
from dp3_topp.interpolation import C2LinearZ, C3QuadraticSpeed, C4CubicSpeed
from dp3_topp.kinematics import path_time_derivatives
from dp3_topp.path_data import PathData


@dataclass(frozen=True)
class DP3Config:
    ns: int = 40
    nz: int = 500
    nch: int = 20
    k1: float = 1.0
    k2: float = 0.0
    z_start: float = 0.0
    z_end: float = 0.0
    z_max: float | None = None
    tau_rate_dt: float = 1e-5


@dataclass(frozen=True)
class TrajectoryResult:
    feasible: bool
    t: np.ndarray
    s: np.ndarray
    z: np.ndarray
    z_s: np.ndarray
    z_ss: np.ndarray
    total_time: float
    audit: ConstraintAudit
    grid_s: np.ndarray
    grid_z: np.ndarray
    policy: np.ndarray
    method: str = "DP3"
    jerk_limited: str = "yes"
    segment_kinds: list[str] | None = None
    segment_profiles: tuple[object, ...] | None = None
    tau_rate_dt: float = 1e-5
    objective_cost: float = 0.0
    objective_time_cost: float = 0.0
    objective_drive_power_cost: float = 0.0


@dataclass(frozen=True)
class TrajectoryQuantities:
    q: np.ndarray
    q_dot: np.ndarray
    q_ddot: np.ndarray
    q_jerk: np.ndarray
    tau: np.ndarray
    tau_rate: np.ndarray
    mechanical_power: np.ndarray
    q_dot_utilization: np.ndarray
    q_ddot_utilization: np.ndarray
    q_jerk_utilization: np.ndarray
    tau_utilization: np.ndarray
    tau_rate_utilization: np.ndarray
    mechanical_power_utilization: np.ndarray
    q_position_utilization: np.ndarray | None = None
    drive_power: np.ndarray | None = None


def optimize_dp3(
    *,
    path: PathData,
    limits: ConstraintLimits,
    config: DP3Config,
    robot: MujocoRobotDynamics | None = None,
) -> TrajectoryResult:
    return _optimize_grid(
        path=path,
        limits=limits,
        config=config,
        robot=robot,
        method="DP3",
        jerk_limited="yes",
        use_dp3_profiles=True,
        optimization_limits=limits,
    )


def optimize_dp2(
    *,
    path: PathData,
    limits: ConstraintLimits,
    config: DP3Config,
    robot: MujocoRobotDynamics | None = None,
    jerk_limited: str = "no",
) -> TrajectoryResult:
    jerk_limited = jerk_limited or "no"
    if jerk_limited not in {"no", "only start / end", "yes"}:
        raise ValueError("jerk_limited must be one of: no, only start / end, yes")
    if jerk_limited == "yes":
        optimization_limits = limits
    else:
        optimization_limits = ConstraintLimits(
            q_dot_abs=limits.q_dot_abs,
            q_ddot_abs=limits.q_ddot_abs,
            q_jerk_abs=np.full(limits.dof, 1e12),
            tau_abs=limits.tau_abs,
            tau_rate_abs=np.full(limits.dof, 1e12),
            mechanical_power_lower=limits.mechanical_power_lower,
            mechanical_power_upper=limits.mechanical_power_upper,
            torque_speed_breakpoints=limits.torque_speed_breakpoints,
            friction_coulomb=limits.friction_coulomb,
            friction_viscous=limits.friction_viscous,
            motor_gear_ratio=limits.motor_gear_ratio,
            motor_torque_constant=limits.motor_torque_constant,
            motor_stator_resistance=limits.motor_stator_resistance,
            q_dot_lower=limits.q_dot_lower,
            q_dot_upper=limits.q_dot_upper,
            q_ddot_lower=limits.q_ddot_lower,
            q_ddot_upper=limits.q_ddot_upper,
            tau_lower=limits.tau_lower,
            tau_upper=limits.tau_upper,
        )
    return _optimize_grid(
        path=path,
        limits=limits,
        config=config,
        robot=robot,
        method="DP2",
        jerk_limited=jerk_limited,
        use_dp3_profiles=False,
        optimization_limits=optimization_limits,
    )


def evaluate_trajectory_quantities(
    *,
    path: PathData,
    result: TrajectoryResult,
    limits: ConstraintLimits,
    robot: MujocoRobotDynamics | None = None,
) -> TrajectoryQuantities:
    q = _sample_matrix(path.s, path.q, result.s)
    q_dot, q_ddot, q_jerk, tau, tau_rate, power = _evaluate_dynamic_quantities(
        path=path,
        limits=limits,
        s=result.s,
        z=result.z,
        z_s=result.z_s,
        z_ss=result.z_ss,
        robot=robot,
        tau_rate_dt=float(result.tau_rate_dt),
    )
    power_limits = np.where(power >= 0.0, limits.mechanical_power_upper, abs(limits.mechanical_power_lower))
    tau_lower, tau_upper = limits.torque_bounds(q_dot)
    drive_power = limits.drive_power(tau=tau, q_dot=q_dot) if limits.has_motor_power_model else None
    q_position_utilization = (
        interval_bound_utilization(q, *limits.q_position_bounds(q.shape))
        if limits.has_q_position_bounds
        else None
    )
    return TrajectoryQuantities(
        q=q,
        q_dot=q_dot,
        q_ddot=q_ddot,
        q_jerk=q_jerk,
        tau=tau,
        tau_rate=tau_rate,
        mechanical_power=power,
        drive_power=drive_power,
        q_dot_utilization=signed_bound_utilization(q_dot, *limits.q_dot_bounds(q_dot.shape)),
        q_ddot_utilization=signed_bound_utilization(q_ddot, *limits.q_ddot_bounds(q_ddot.shape)),
        q_jerk_utilization=signed_bound_utilization(q_jerk, *limits.q_jerk_bounds(q_jerk.shape)),
        tau_utilization=signed_bound_utilization(tau, tau_lower, tau_upper),
        tau_rate_utilization=signed_bound_utilization(tau_rate, *limits.tau_rate_bounds(tau_rate.shape)),
        mechanical_power_utilization=power / power_limits,
        q_position_utilization=q_position_utilization,
    )


def _require_grid_count(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"DP3Config {name} must be a finite integer >= {minimum}")
    count = int(value)
    if count < int(minimum):
        raise ValueError(f"DP3Config {name} must be >= {minimum}")
    return count


def _optimize_grid(
    *,
    path: PathData,
    limits: ConstraintLimits,
    config: DP3Config,
    robot: MujocoRobotDynamics | None,
    method: str,
    jerk_limited: str,
    use_dp3_profiles: bool,
    optimization_limits: ConstraintLimits,
) -> TrajectoryResult:
    ns = _require_grid_count("ns", config.ns, 3)
    nz = _require_grid_count("nz", config.nz, 2)
    nch = _require_grid_count("nch", config.nch, 2)
    if use_dp3_profiles and ns < 4:
        raise ValueError("DP3 profile construction requires ns >= 4")
    if not np.isfinite(config.k1) or not np.isfinite(config.k2) or config.k1 < 0.0 or config.k2 < 0.0 or (config.k1 == 0.0 and config.k2 == 0.0):
        raise ValueError("DP3Config requires finite nonnegative k1/k2 and at least one positive cost weight")
    if not np.isfinite(config.tau_rate_dt) or config.tau_rate_dt <= 0.0:
        raise ValueError("DP3Config requires positive finite tau_rate_dt")
    if not np.isfinite(config.z_start) or config.z_start < 0.0:
        raise ValueError("DP3Config requires finite nonnegative z_start")
    if not np.isfinite(config.z_end) or config.z_end < 0.0:
        raise ValueError("DP3Config requires finite nonnegative z_end")
    if config.z_max is not None and (not np.isfinite(config.z_max) or config.z_max <= 0.0):
        raise ValueError("DP3Config requires positive finite z_max when provided")
    if config.z_max is not None and config.z_start > float(config.z_max) + 1e-12:
        raise ValueError("DP3Config z_start must not exceed z_max")
    if config.z_max is not None and config.z_end > float(config.z_max) + 1e-12:
        raise ValueError("DP3Config z_end must not exceed z_max")
    if config.k2 > 0.0 and not limits.has_motor_power_model:
        raise ValueError("k2 > 0 requires a complete motor power model in limits")
    _validate_boundary_speed_limits(path, limits, config)
    grid_s = np.linspace(path.s[0], path.s[-1], ns)
    z_ceiling = _z_ceiling(path, limits, grid_s, config.z_max)
    grid_z = np.vstack([np.linspace(0.0, max(float(zi), 1e-12), nz) for zi in z_ceiling])
    grid_z[0, 0] = float(config.z_start)
    grid_z[-1, 0] = float(config.z_end)

    cost = np.full((ns, nz), np.inf, dtype=np.float64)
    policy = np.full((ns, nz), -1, dtype=np.int32)
    stored_z_s = np.zeros((ns, nz), dtype=np.float64)
    cost[-1, 0] = 0.0

    first_i = 2 if use_dp3_profiles else 0
    for i in range(ns - 2, first_i - 1, -1):
        for j in range(nz):
            zi = float(grid_z[i, j])
            if i == 0 and abs(zi - config.z_start) > 1e-12:
                continue
            best_cost = np.inf
            best_l = -1
            best_slope = 0.0
            for l in range(nz):
                if not np.isfinite(cost[i + 1, l]):
                    continue
                if i + 1 == ns - 1 and l != 0:
                    continue
                zn = float(grid_z[i + 1, l])
                force_kind = "C2" if i + 1 == ns - 1 else None
                profile = _make_segment_profile(
                    s0=float(grid_s[i]),
                    s1=float(grid_s[i + 1]),
                    z0=zi,
                    z1=zn,
                    next_z_s=float(stored_z_s[i + 1, l]),
                    use_dp3_profiles=use_dp3_profiles,
                    force_kind=force_kind,
                )
                if profile is None:
                    continue
                segment_limits = _optimization_limits_for_segment(
                    limits=limits,
                    default_limits=optimization_limits,
                    method=method,
                    jerk_limited=jerk_limited,
                    segment_index=i,
                    segment_count=ns - 1,
                    z0=zi,
                    z1=zn,
                )
                audit = _audit_profile_segment(path, segment_limits, profile, nch, robot, tau_rate_dt=config.tau_rate_dt)
                if not audit.ok:
                    continue
                try:
                    segment_cost = _segment_objective_cost(path, limits, profile, config, robot)
                except ValueError:
                    continue
                candidate = segment_cost + cost[i + 1, l]
                if candidate < best_cost:
                    best_cost = candidate
                    best_l = l
                    best_slope = float(profile.evaluate(np.array([grid_s[i]])).z_s[0])
            if best_l >= 0:
                cost[i, j] = best_cost
                policy[i, j] = best_l
                stored_z_s[i, j] = best_slope

    if use_dp3_profiles:
        _solve_dp3_algorithm3_start(
            path=path,
            limits=optimization_limits,
            config=config,
            robot=robot,
            grid_s=grid_s,
            grid_z=grid_z,
            cost=cost,
            policy=policy,
            stored_z_s=stored_z_s,
        )

    if not np.isfinite(cost[0, 0]):
        empty_sample_count = int(path.q.shape[0]) if limits.has_q_position_bounds else 1
        zero_quantities = np.zeros((empty_sample_count, limits.dof), dtype=np.float64)
        empty_audit = audit_constraints(
            limits=limits,
            q_position=path.q if limits.has_q_position_bounds else None,
            q_dot=zero_quantities,
            q_ddot=zero_quantities,
            q_jerk=zero_quantities,
            tau=zero_quantities,
            tau_rate=zero_quantities,
            mechanical_power=np.zeros(empty_sample_count, dtype=np.float64),
        )
        return TrajectoryResult(
            feasible=False,
            t=np.array([], dtype=np.float64),
            s=np.array([], dtype=np.float64),
            z=np.array([], dtype=np.float64),
            z_s=np.array([], dtype=np.float64),
            z_ss=np.array([], dtype=np.float64),
            total_time=np.inf,
            audit=empty_audit,
            grid_s=grid_s,
            grid_z=grid_z,
            policy=policy,
            method=method,
            jerk_limited=jerk_limited,
            segment_kinds=[],
            tau_rate_dt=float(config.tau_rate_dt),
            objective_cost=np.inf,
            objective_time_cost=np.inf,
            objective_drive_power_cost=np.inf,
        )

    t_out, s_out, z_out, z_s_out, z_ss_out, segment_kinds, segment_profiles, total_time = _reconstruct_profile(
        grid_s,
        grid_z,
        policy,
        stored_z_s,
        config,
        use_dp3_profiles=use_dp3_profiles,
    )
    audit = _audit_samples(path, limits, s_out, z_out, z_s_out, z_ss_out, robot, tau_rate_dt=config.tau_rate_dt)
    objective_time_cost = float(config.k1 * total_time)
    objective_drive_power_cost = float(cost[0, 0] - objective_time_cost)
    return TrajectoryResult(
        feasible=audit.ok,
        t=t_out,
        s=s_out,
        z=z_out,
        z_s=z_s_out,
        z_ss=z_ss_out,
        audit=audit,
        grid_s=grid_s,
        grid_z=grid_z,
        policy=policy,
        method=method,
        jerk_limited=jerk_limited,
        segment_kinds=segment_kinds,
        segment_profiles=segment_profiles,
        tau_rate_dt=float(config.tau_rate_dt),
        total_time=float(total_time),
        objective_cost=float(cost[0, 0]),
        objective_time_cost=objective_time_cost,
        objective_drive_power_cost=objective_drive_power_cost,
    )


def resample_trajectory_by_segments(result: TrajectoryResult, points: int) -> TrajectoryResult | None:
    profiles = tuple(result.segment_profiles or ())
    if not profiles or result.s.size == 0:
        return None
    sample_count = max(2, int(points))
    s = np.linspace(float(profiles[0].s0), float(profiles[-1].s1), sample_count)
    t = np.empty(sample_count, dtype=np.float64)
    z = np.empty(sample_count, dtype=np.float64)
    z_s = np.empty(sample_count, dtype=np.float64)
    z_ss = np.empty(sample_count, dtype=np.float64)
    assigned = np.zeros(sample_count, dtype=bool)
    segment_start_time = 0.0
    for index, profile in enumerate(profiles):
        duration = profile.duration()
        if index == len(profiles) - 1:
            mask = (s >= float(profile.s0) - 1e-12) & (s <= float(profile.s1) + 1e-12)
        else:
            mask = (s >= float(profile.s0) - 1e-12) & (s < float(profile.s1) - 1e-12)
        if np.any(mask):
            segment_s = s[mask]
            values = profile.evaluate(segment_s)
            t[mask] = _profile_elapsed_at_samples(profile, segment_s, duration=duration) + segment_start_time
            z[mask] = values.z
            z_s[mask] = values.z_s
            z_ss[mask] = values.z_ss
            assigned[mask] = True
        segment_start_time += float(duration)
    if not np.all(assigned):
        raise RuntimeError("Segment profiles did not cover the dense constraint-check grid")
    return TrajectoryResult(
        feasible=result.feasible,
        t=t,
        s=s,
        z=z,
        z_s=z_s,
        z_ss=z_ss,
        total_time=result.total_time,
        audit=result.audit,
        grid_s=result.grid_s,
        grid_z=result.grid_z,
        policy=result.policy,
        method=result.method,
        jerk_limited=result.jerk_limited,
        segment_kinds=result.segment_kinds,
        segment_profiles=profiles,
        tau_rate_dt=result.tau_rate_dt,
        objective_cost=result.objective_cost,
        objective_time_cost=result.objective_time_cost,
        objective_drive_power_cost=result.objective_drive_power_cost,
    )


def resample_trajectory_by_time(result: TrajectoryResult, t: np.ndarray) -> TrajectoryResult:
    times = np.asarray(t, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError("time samples must be a one-dimensional array")
    if times.size == 0:
        return TrajectoryResult(
            feasible=result.feasible,
            t=np.array([], dtype=np.float64),
            s=np.array([], dtype=np.float64),
            z=np.array([], dtype=np.float64),
            z_s=np.array([], dtype=np.float64),
            z_ss=np.array([], dtype=np.float64),
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
            objective_time_cost=result.objective_time_cost,
            objective_drive_power_cost=result.objective_drive_power_cost,
        )
    if result.s.size == 0:
        raise ValueError("cannot time-resample an empty trajectory")
    if np.any(~np.isfinite(times)):
        raise ValueError("time samples must be finite")
    total_time = float(result.total_time)
    if not np.isfinite(total_time) or total_time < 0.0:
        raise ValueError("trajectory has no finite duration")
    if np.any(times < -1e-12) or np.any(times > total_time + 1e-12):
        raise ValueError("time samples must lie within [0, total_time]")
    times = np.clip(times, 0.0, total_time)
    profiles = tuple(result.segment_profiles or ())
    if profiles:
        sampled = _resample_time_with_profiles(result, times, profiles)
    else:
        sampled = _resample_time_with_discrete_samples(result, times)
    return sampled


def _resample_time_with_profiles(
    result: TrajectoryResult,
    times: np.ndarray,
    profiles: tuple[object, ...],
) -> TrajectoryResult:
    s = np.empty(times.size, dtype=np.float64)
    z = np.empty(times.size, dtype=np.float64)
    z_s = np.empty(times.size, dtype=np.float64)
    z_ss = np.empty(times.size, dtype=np.float64)
    segment_starts: list[float] = []
    elapsed = 0.0
    for profile in profiles:
        segment_starts.append(elapsed)
        elapsed += float(profile.duration())
    for sample_idx, time_value in enumerate(times):
        segment_idx = _segment_index_for_time(time_value, segment_starts, profiles)
        profile = profiles[segment_idx]
        local_time = min(max(float(time_value) - segment_starts[segment_idx], 0.0), float(profile.duration()))
        s_value = _profile_s_at_elapsed(profile, local_time)
        values = profile.evaluate(np.array([s_value]))
        s[sample_idx] = values.s[0]
        z[sample_idx] = values.z[0]
        z_s[sample_idx] = values.z_s[0]
        z_ss[sample_idx] = values.z_ss[0]
    return TrajectoryResult(
        feasible=result.feasible,
        t=times.copy(),
        s=s,
        z=z,
        z_s=z_s,
        z_ss=z_ss,
        total_time=result.total_time,
        audit=result.audit,
        grid_s=result.grid_s,
        grid_z=result.grid_z,
        policy=result.policy,
        method=result.method,
        jerk_limited=result.jerk_limited,
        segment_kinds=result.segment_kinds,
        segment_profiles=profiles,
        tau_rate_dt=result.tau_rate_dt,
        objective_cost=result.objective_cost,
        objective_time_cost=result.objective_time_cost,
        objective_drive_power_cost=result.objective_drive_power_cost,
    )


def _resample_time_with_discrete_samples(result: TrajectoryResult, times: np.ndarray) -> TrajectoryResult:
    return TrajectoryResult(
        feasible=result.feasible,
        t=times.copy(),
        s=np.interp(times, result.t, result.s),
        z=np.interp(times, result.t, result.z),
        z_s=np.interp(times, result.t, result.z_s),
        z_ss=np.interp(times, result.t, result.z_ss),
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
        objective_time_cost=result.objective_time_cost,
        objective_drive_power_cost=result.objective_drive_power_cost,
    )


def _segment_index_for_time(time_value: float, segment_starts: list[float], profiles: tuple[object, ...]) -> int:
    for idx, start in enumerate(segment_starts):
        end = start + float(profiles[idx].duration())
        if idx == len(profiles) - 1 or float(time_value) <= end + 1e-12:
            return idx
    return len(profiles) - 1


def _profile_s_at_elapsed(profile, elapsed_time: float) -> float:
    duration = float(profile.duration())
    if elapsed_time <= 1e-12:
        return float(profile.s0)
    if elapsed_time >= duration - 1e-12:
        return float(profile.s1)
    lo = float(profile.s0)
    hi = float(profile.s1)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        elapsed_mid = float(_profile_elapsed_at_samples(profile, np.array([mid]), duration=duration)[0])
        if elapsed_mid < float(elapsed_time):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _validate_boundary_speed_limits(path: PathData, limits: ConstraintLimits, config: DP3Config) -> None:
    for field, z_value, sample, label in (
        ("z_start", config.z_start, 0, "s=0"),
        ("z_end", config.z_end, -1, "s=1"),
    ):
        q_dot = path.q_s[[sample]] * np.sqrt(float(z_value))
        lower, upper = limits.q_dot_bounds(q_dot.shape)
        bad = np.argwhere((q_dot < lower - 1e-12) | (q_dot > upper + 1e-12))
        if bad.size == 0:
            continue
        _, axis = bad[0]
        axis = int(axis)
        value = float(q_dot[0, axis])
        limit = float(upper[0, axis] if value > upper[0, axis] else lower[0, axis])
        raise ValueError(
            f"DP3Config {field} violates q_dot limit at {label}: "
            f"axis {axis + 1} q_dot={value:.6g}, limit={limit:.6g}"
        )


def _z_ceiling(path: PathData, limits: ConstraintLimits, grid_s: np.ndarray, z_max: float | None) -> np.ndarray:
    q_s = _sample_matrix(path.s, path.q_s, grid_s)
    q_dot_lower, q_dot_upper = limits.q_dot_bounds(q_s.shape)
    ceilings = []
    for sample, row in enumerate(q_s):
        candidates = []
        for axis in range(path.dof):
            derivative = float(row[axis])
            if abs(derivative) <= 1e-12:
                continue
            speed_limit = (
                float(q_dot_upper[sample, axis]) / derivative
                if derivative > 0.0
                else float(q_dot_lower[sample, axis]) / derivative
            )
            candidates.append(max(speed_limit, 0.0) ** 2)
        ceilings.append(min(candidates) if candidates else (1.0 if z_max is None else float(z_max)))
    out = np.asarray(ceilings, dtype=np.float64)
    if z_max is not None:
        out = np.minimum(out, float(z_max))
    return np.maximum(out, 1e-9)


def _optimization_limits_for_segment(
    *,
    limits: ConstraintLimits,
    default_limits: ConstraintLimits,
    method: str,
    jerk_limited: str,
    segment_index: int,
    segment_count: int,
    z0: float,
    z1: float,
) -> ConstraintLimits:
    if method != "DP2":
        return default_limits
    if jerk_limited == "yes":
        return limits
    if jerk_limited == "only start / end" and (
        segment_index == 0
        or segment_index == int(segment_count) - 1
        or float(z0) <= 1e-12
        or float(z1) <= 1e-12
    ):
        return limits
    return default_limits


def _make_segment_profile(
    s0: float,
    s1: float,
    z0: float,
    z1: float,
    next_z_s: float,
    use_dp3_profiles: bool,
    force_kind: str | None,
):
    try:
        if force_kind == "C2" or not use_dp3_profiles:
            return C2LinearZ.from_interval(s0, s1, max(z0, 0.0), max(z1, 0.0))
        if z0 <= 1e-12 or z1 <= 1e-12:
            return None
        if force_kind == "C4":
            raise ValueError("C4 requires explicit endpoint slopes")
        return C3QuadraticSpeed.from_interval(s0, s1, z0=z0, z1=z1, z_s1=next_z_s)
    except ValueError:
        return None


def _solve_dp3_algorithm3_start(
    *,
    path: PathData,
    limits: ConstraintLimits,
    config: DP3Config,
    robot: MujocoRobotDynamics | None,
    grid_s: np.ndarray,
    grid_z: np.ndarray,
    cost: np.ndarray,
    policy: np.ndarray,
    stored_z_s: np.ndarray,
) -> None:
    if config.ns < 4:
        return
    best_cost = np.inf
    best_j = -1
    best_l = -1
    best_left_slope = 0.0
    z_start = float(grid_z[0, 0])
    for j in range(1, config.nz):
        z_mid = float(grid_z[1, j])
        c2 = C2LinearZ.from_interval(float(grid_s[0]), float(grid_s[1]), z_start, z_mid)
        audit_c2 = _audit_profile_segment(path, limits, c2, config.nch, robot, tau_rate_dt=config.tau_rate_dt)
        if not audit_c2.ok:
            continue
        left_slope = float(c2.evaluate(np.array([grid_s[1]])).z_s[0])
        try:
            c2_cost = _segment_objective_cost(path, limits, c2, config, robot)
        except ValueError:
            continue
        for l in range(config.nz):
            if not np.isfinite(cost[2, l]):
                continue
            z_next = float(grid_z[2, l])
            if z_mid <= 1e-12 or z_next <= 1e-12:
                continue
            try:
                c4 = C4CubicSpeed.from_interval(
                    float(grid_s[1]),
                    float(grid_s[2]),
                    z0=z_mid,
                    z1=z_next,
                    z_s0=left_slope,
                    z_s1=float(stored_z_s[2, l]),
                )
            except ValueError:
                continue
            audit_c4 = _audit_profile_segment(path, limits, c4, config.nch, robot, tau_rate_dt=config.tau_rate_dt)
            if not audit_c4.ok:
                continue
            try:
                c4_cost = _segment_objective_cost(path, limits, c4, config, robot)
            except ValueError:
                continue
            candidate = c2_cost + c4_cost + cost[2, l]
            if candidate < best_cost:
                best_cost = candidate
                best_j = j
                best_l = l
                best_left_slope = left_slope
    if best_j >= 0:
        cost[0, 0] = best_cost
        policy[0, 0] = best_j
        policy[1, best_j] = best_l
        stored_z_s[0, 0] = best_left_slope
        stored_z_s[1, best_j] = best_left_slope


def _audit_profile_segment(
    path: PathData,
    limits: ConstraintLimits,
    profile,
    nch: int,
    robot: MujocoRobotDynamics | None,
    tau_rate_dt: float = 1e-5,
) -> ConstraintAudit:
    s = np.linspace(profile.s0, profile.s1, int(nch))
    values = profile.evaluate(s)
    return _audit_samples(path, limits, values.s, values.z, values.z_s, values.z_ss, robot, tau_rate_dt=tau_rate_dt)


def _audit_samples(
    path: PathData,
    limits: ConstraintLimits,
    s: np.ndarray,
    z: np.ndarray,
    z_s: np.ndarray,
    z_ss: np.ndarray,
    robot: MujocoRobotDynamics | None,
    tau_rate_dt: float = 1e-5,
) -> ConstraintAudit:
    q = _sample_matrix(path.s, path.q, s) if limits.has_q_position_bounds else None
    q_dot, q_ddot, q_jerk, tau, tau_rate, power = _evaluate_dynamic_quantities(
        path=path,
        limits=limits,
        s=s,
        z=z,
        z_s=z_s,
        z_ss=z_ss,
        robot=robot,
        tau_rate_dt=float(tau_rate_dt),
    )
    return audit_constraints(
        limits=limits,
        q_dot=q_dot,
        q_ddot=q_ddot,
        q_jerk=q_jerk,
        tau=tau,
        tau_rate=tau_rate,
        mechanical_power=power,
        q_position=q,
    )


def _evaluate_dynamic_quantities(
    *,
    path: PathData,
    limits: ConstraintLimits,
    s: np.ndarray,
    z: np.ndarray,
    z_s: np.ndarray,
    z_ss: np.ndarray,
    robot: MujocoRobotDynamics | None,
    tau_rate_dt: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if np.asarray(s).size == 0:
        empty = np.zeros((0, path.dof), dtype=np.float64)
        return empty, empty, empty, empty, empty, np.zeros(0, dtype=np.float64)
    q = _sample_matrix(path.s, path.q, s)
    q_s = _sample_matrix(path.s, path.q_s, s)
    q_ss = _sample_matrix(path.s, path.q_ss, s)
    q_sss = _sample_matrix(path.s, path.q_sss, s)
    kin = path_time_derivatives(q_s=q_s, q_ss=q_ss, q_sss=q_sss, z=z, z_s=z_s, z_ss=z_ss)
    if robot is None:
        tau = np.zeros_like(kin.q_dot)
        tau_rate = np.zeros_like(kin.q_dot)
    else:
        tau = np.vstack([robot.inverse_dynamics(qi, qdi, qddi) for qi, qdi, qddi in zip(q, kin.q_dot, kin.q_ddot)])
        tau_rate = np.vstack(
            [
                robot.torque_rate_finite_difference(qi, qdi, qddi, qdddi, dt=float(tau_rate_dt))
                for qi, qdi, qddi, qdddi in zip(q, kin.q_dot, kin.q_ddot, kin.q_jerk)
            ]
        )
    tau = tau + limits.friction_torque(kin.q_dot)
    tau_rate = tau_rate + limits.friction_torque_rate(kin.q_ddot)
    power = np.sum(tau * kin.q_dot, axis=1)
    return kin.q_dot, kin.q_ddot, kin.q_jerk, tau, tau_rate, power


def _segment_objective_cost(
    path: PathData,
    limits: ConstraintLimits,
    profile,
    config: DP3Config,
    robot: MujocoRobotDynamics | None,
) -> float:
    duration = profile.duration()
    if config.k2 == 0.0:
        return float(config.k1 * duration)
    nodes, weights = np.polynomial.legendre.leggauss(max(16, int(config.nch) * 2))
    midpoint = 0.5 * (profile.s0 + profile.s1)
    half_width = 0.5 * (profile.s1 - profile.s0)
    s = midpoint + half_width * nodes
    values = profile.evaluate(s)
    q_dot, _, _, tau, _, _ = _evaluate_dynamic_quantities(
        path=path,
        limits=limits,
        s=values.s,
        z=values.z,
        z_s=values.z_s,
        z_ss=values.z_ss,
        robot=robot,
        tau_rate_dt=float(config.tau_rate_dt),
    )
    drive_power = limits.drive_power(tau=tau, q_dot=q_dot)
    energy_cost = half_width * np.sum(weights * drive_power / values.s_dot)
    return float(config.k1 * duration + config.k2 * energy_cost)


def _sample_matrix(s_ref: np.ndarray, values: np.ndarray, s: np.ndarray) -> np.ndarray:
    samples = np.asarray(s, dtype=np.float64)
    if np.any(~np.isfinite(samples)):
        raise ValueError("path sample points must be finite")
    if np.any(samples < float(s_ref[0]) - 1e-12) or np.any(samples > float(s_ref[-1]) + 1e-12):
        raise ValueError("path sample points must lie within the path domain")
    samples = np.clip(samples, float(s_ref[0]), float(s_ref[-1]))
    return np.column_stack([np.interp(samples, s_ref, values[:, axis]) for axis in range(values.shape[1])])


def _reconstruct_profile(
    grid_s: np.ndarray,
    grid_z: np.ndarray,
    policy: np.ndarray,
    stored_z_s: np.ndarray,
    config: DP3Config,
    use_dp3_profiles: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], tuple[object, ...], float]:
    all_t: list[np.ndarray] = []
    all_s: list[np.ndarray] = []
    all_z: list[np.ndarray] = []
    all_z_s: list[np.ndarray] = []
    all_z_ss: list[np.ndarray] = []
    segment_kinds: list[str] = []
    segment_profiles: list[object] = []
    total_time = 0.0
    j = 0
    for i in range(config.ns - 1):
        l = int(policy[i, j])
        if l < 0:
            raise RuntimeError("Cannot reconstruct infeasible policy")
        kind = "C2"
        if use_dp3_profiles and i == 1:
            kind = "C4"
            profile = C4CubicSpeed.from_interval(
                float(grid_s[i]),
                float(grid_s[i + 1]),
                z0=float(grid_z[i, j]),
                z1=float(grid_z[i + 1, l]),
                z_s0=float(stored_z_s[i, j]),
                z_s1=float(stored_z_s[i + 1, l]),
            )
        else:
            profile = _make_segment_profile(
                s0=float(grid_s[i]),
                s1=float(grid_s[i + 1]),
                z0=float(grid_z[i, j]),
                z1=float(grid_z[i + 1, l]),
                next_z_s=float(stored_z_s[i + 1, l]),
                use_dp3_profiles=use_dp3_profiles,
                force_kind="C2" if i == 0 or i == config.ns - 2 else None,
            )
            if use_dp3_profiles and i > 1 and i < config.ns - 2 and float(grid_z[i, j]) > 1e-12 and float(grid_z[i + 1, l]) > 1e-12:
                kind = "C3"
        if profile is None:
            raise RuntimeError("Stored policy contains an invalid profile")
        segment_start_time = float(total_time)
        segment_duration = profile.duration()
        local_s = np.linspace(grid_s[i], grid_s[i + 1], int(config.nch))
        values = profile.evaluate(local_s)
        local_t = _profile_elapsed_at_samples(profile, local_s, duration=segment_duration) + segment_start_time
        if i > 0:
            values = type(values)(
                s=values.s[1:],
                z=values.z[1:],
                z_s=values.z_s[1:],
                z_ss=values.z_ss[1:],
                s_dot=values.s_dot[1:],
            )
            local_t = local_t[1:]
        all_t.append(local_t)
        all_s.append(values.s)
        all_z.append(values.z)
        all_z_s.append(values.z_s)
        all_z_ss.append(values.z_ss)
        segment_kinds.append(kind)
        segment_profiles.append(profile)
        total_time = segment_start_time + float(segment_duration)
        j = l
    return (
        np.concatenate(all_t),
        np.concatenate(all_s),
        np.concatenate(all_z),
        np.concatenate(all_z_s),
        np.concatenate(all_z_ss),
        segment_kinds,
        tuple(segment_profiles),
        float(total_time),
    )


def _profile_elapsed_at_samples(profile, samples: np.ndarray, duration: float) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    elapsed = np.zeros(samples.size, dtype=np.float64)
    nodes, weights = np.polynomial.legendre.leggauss(32)
    for idx, sample in enumerate(samples):
        if sample <= float(profile.s0) + 1e-12:
            elapsed[idx] = 0.0
            continue
        if sample >= float(profile.s1) - 1e-12:
            elapsed[idx] = float(duration)
            continue
        a = float(profile.s0)
        b = float(sample)
        midpoint = 0.5 * (a + b)
        half_width = 0.5 * (b - a)
        points = midpoint + half_width * nodes
        speed = profile.evaluate(points).s_dot
        elapsed[idx] = half_width * float(np.sum(weights / speed))
    return elapsed
