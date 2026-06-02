from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mujoco  # type: ignore
import numpy as np
from scipy.spatial.transform import Rotation

from dp3_topp.dynamics_mujoco import MujocoRobotDynamics
from dp3_topp.path_data import PathData


@dataclass(frozen=True)
class CartesianIKDiagnostics:
    require_convergence: bool
    converged: bool
    samples: int
    total_iterations: int
    max_position_error: float
    max_orientation_error: float
    max_residual: float

    def as_dict(self) -> dict[str, bool | int | float]:
        return {
            "require_convergence": bool(self.require_convergence),
            "converged": bool(self.converged),
            "samples": int(self.samples),
            "total_iterations": int(self.total_iterations),
            "max_position_error": float(self.max_position_error),
            "max_orientation_error": float(self.max_orientation_error),
            "max_residual": float(self.max_residual),
        }


@dataclass(frozen=True)
class CartesianPathResult:
    path: PathData
    ik: CartesianIKDiagnostics


@dataclass(frozen=True)
class _KinematicTarget:
    kind: str
    id: int
    name: str


def parse_pose_table(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip().lstrip("\ufeff")
        if not line:
            continue
        parts = re.split(r"[,\s;]+", line)
        if len(parts) < 6:
            raise ValueError(f"Line {lineno} has fewer than 6 columns: {line}")
        try:
            rows.append([float(item) for item in parts[:6]])
        except ValueError as exc:
            raise ValueError(f"Non-numeric value at line {lineno}: {line}") from exc
    if not rows:
        raise ValueError(f"Empty pose table: {path}")
    data = np.asarray(rows, dtype=np.float64)
    if np.any(~np.isfinite(data)):
        raise ValueError(f"Pose table must contain only finite values: {path}")
    return data


def build_joint_path_from_cartesian(
    *,
    robot: MujocoRobotDynamics,
    pose_table: np.ndarray,
    ee_body: str,
    ee_site: str | None = None,
    max_iters: int = 200,
    pos_tol: float = 1e-4,
    ori_tol: float = 1e-3,
    damping: float = 1e-4,
    step_scale: float = 0.6,
    orientation_weight: float = 0.1,
    tcp_offset: np.ndarray | None = None,
    require_convergence: bool = False,
) -> PathData:
    return build_joint_path_result_from_cartesian(
        robot=robot,
        pose_table=pose_table,
        ee_body=ee_body,
        ee_site=ee_site,
        max_iters=max_iters,
        pos_tol=pos_tol,
        ori_tol=ori_tol,
        damping=damping,
        step_scale=step_scale,
        orientation_weight=orientation_weight,
        tcp_offset=tcp_offset,
        require_convergence=require_convergence,
    ).path


def build_joint_path_result_from_cartesian(
    *,
    robot: MujocoRobotDynamics,
    pose_table: np.ndarray,
    ee_body: str,
    ee_site: str | None = None,
    max_iters: int = 200,
    pos_tol: float = 1e-4,
    ori_tol: float = 1e-3,
    damping: float = 1e-4,
    step_scale: float = 0.6,
    orientation_weight: float = 0.1,
    tcp_offset: np.ndarray | None = None,
    require_convergence: bool = False,
) -> CartesianPathResult:
    pose_table = np.asarray(pose_table, dtype=np.float64)
    if pose_table.ndim != 2 or pose_table.shape[1] < 6:
        raise ValueError("pose_table must have columns x y z rz ry rx")
    if np.any(~np.isfinite(pose_table[:, :6])):
        raise ValueError("pose_table must contain only finite values")
    if pose_table.shape[0] < 2:
        raise ValueError("pose_table must contain at least two poses")
    tcp = np.zeros(3, dtype=np.float64) if tcp_offset is None else np.asarray(tcp_offset, dtype=np.float64)
    _validate_ik_parameters(
        max_iters=max_iters,
        pos_tol=pos_tol,
        ori_tol=ori_tol,
        damping=damping,
        step_scale=step_scale,
        orientation_weight=orientation_weight,
        tcp_offset=tcp,
    )
    q_path, diagnostics = _solve_cartesian_ik(
        robot=robot,
        pose_table=pose_table[:, :6],
        ee_body=ee_body,
        ee_site=ee_site,
        tcp_offset=tcp,
        max_iters=max_iters,
        pos_tol=pos_tol,
        ori_tol=ori_tol,
        damping=damping,
        step_scale=step_scale,
        orientation_weight=orientation_weight,
        require_convergence=require_convergence,
    )
    xyz = _fk_positions(robot, q_path, ee_body, tcp, ee_site=ee_site)
    s = _cartesian_s(xyz)
    edge_order = 2 if q_path.shape[0] >= 3 else 1
    q_s = np.gradient(q_path, s, axis=0, edge_order=edge_order)
    q_ss = np.gradient(q_s, s, axis=0, edge_order=edge_order)
    q_sss = np.gradient(q_ss, s, axis=0, edge_order=edge_order)
    path = PathData(s=s, q=q_path, q_s=q_s, q_ss=q_ss, q_sss=q_sss)
    return CartesianPathResult(path=path, ik=diagnostics)


def _validate_ik_parameters(
    *,
    max_iters: int,
    pos_tol: float,
    ori_tol: float,
    damping: float,
    step_scale: float,
    orientation_weight: float,
    tcp_offset: np.ndarray,
) -> None:
    if int(max_iters) < 1:
        raise ValueError("max_iters must be at least 1")
    for name, value in (
        ("pos_tol", pos_tol),
        ("ori_tol", ori_tol),
        ("damping", damping),
        ("step_scale", step_scale),
    ):
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    orientation_weight = float(orientation_weight)
    if not np.isfinite(orientation_weight) or orientation_weight < 0.0:
        raise ValueError("orientation_weight must be finite and nonnegative")
    if tcp_offset.shape != (3,) or np.any(~np.isfinite(tcp_offset)):
        raise ValueError("tcp_offset must be a finite three-vector")


def _rot_error_vec(r_current: np.ndarray, r_target: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(r_target @ r_current.T).as_rotvec()


def _solve_cartesian_ik(
    *,
    robot: MujocoRobotDynamics,
    pose_table: np.ndarray,
    ee_body: str,
    ee_site: str | None,
    tcp_offset: np.ndarray,
    max_iters: int,
    pos_tol: float,
    ori_tol: float,
    damping: float,
    step_scale: float,
    orientation_weight: float,
    require_convergence: bool,
) -> tuple[np.ndarray, CartesianIKDiagnostics]:
    model = robot.model
    target = _resolve_kinematic_target(model, ee_body=ee_body, ee_site=ee_site)
    target_pos = pose_table[:, :3]
    target_rot = Rotation.from_euler("zyx", pose_table[:, 3:6], degrees=False).as_matrix()
    data = mujoco.MjData(model)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    q = np.clip(0.5 * (robot.lower + robot.upper), robot.lower, robot.upper)
    q_path = np.zeros((pose_table.shape[0], robot.dof), dtype=np.float64)
    lambda2 = float(damping) * float(damping)
    all_converged = True
    max_position_error = 0.0
    max_orientation_error = 0.0
    total_iterations = 0

    for idx in range(pose_table.shape[0]):
        converged = False
        for _ in range(max(1, int(max_iters))):
            total_iterations += 1
            residual_norm, pos_err, ori_err, pos_norm, ori_norm = _ik_pose_error(
                model=model,
                data=data,
                robot=robot,
                target=target,
                q=q,
                target_pos=target_pos[idx],
                target_rot=target_rot[idx],
                tcp_offset=tcp_offset,
                orientation_weight=orientation_weight,
            )
            if _ik_converged(pos_norm, ori_norm, pos_tol, ori_tol, orientation_weight):
                converged = True
                break

            _target_jacobian(model, data, target, jacp, jacr)
            j = np.vstack(
                (
                    jacp[:, robot.dof_indices],
                    float(orientation_weight) * jacr[:, robot.dof_indices],
                )
            )
            e = np.concatenate((pos_err, float(orientation_weight) * ori_err))
            dq = j.T @ np.linalg.solve(j @ j.T + lambda2 * np.eye(6), e)
            best_q: np.ndarray | None = None
            alpha = float(step_scale)
            for _ in range(12):
                candidate = np.clip(q + alpha * dq, robot.lower, robot.upper)
                candidate_norm, *_ = _ik_pose_error(
                    model=model,
                    data=data,
                    robot=robot,
                    target=target,
                    q=candidate,
                    target_pos=target_pos[idx],
                    target_rot=target_rot[idx],
                    tcp_offset=tcp_offset,
                    orientation_weight=orientation_weight,
                )
                if candidate_norm < residual_norm - 1e-12:
                    best_q = candidate
                    break
                alpha *= 0.5
            if best_q is None:
                break
            q = best_q
        _, _, _, pos_norm, ori_norm = _ik_pose_error(
            model=model,
            data=data,
            robot=robot,
            target=target,
            q=q,
            target_pos=target_pos[idx],
            target_rot=target_rot[idx],
            tcp_offset=tcp_offset,
            orientation_weight=orientation_weight,
        )
        converged = converged or _ik_converged(pos_norm, ori_norm, pos_tol, ori_tol, orientation_weight)
        max_position_error = max(max_position_error, pos_norm)
        max_orientation_error = max(max_orientation_error, ori_norm)
        all_converged = all_converged and converged
        if require_convergence and not converged:
            raise ValueError(
                f"IK failed to converge at pose {idx}: "
                f"position error {pos_norm:.6g}, orientation error {ori_norm:.6g}"
            )
        q_path[idx] = q
    diagnostics = CartesianIKDiagnostics(
        require_convergence=bool(require_convergence),
        converged=bool(all_converged),
        samples=int(pose_table.shape[0]),
        total_iterations=int(total_iterations),
        max_position_error=float(max_position_error),
        max_orientation_error=float(max_orientation_error),
        max_residual=float(max(max_position_error, max_orientation_error) if orientation_weight > 0.0 else max_position_error),
    )
    return q_path, diagnostics


def _ik_converged(
    pos_norm: float,
    ori_norm: float,
    pos_tol: float,
    ori_tol: float,
    orientation_weight: float,
) -> bool:
    return float(pos_norm) <= float(pos_tol) and (float(orientation_weight) <= 0.0 or float(ori_norm) <= float(ori_tol))


def _ik_pose_error(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot: MujocoRobotDynamics,
    target: _KinematicTarget,
    q: np.ndarray,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    tcp_offset: np.ndarray,
    orientation_weight: float,
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[robot.qpos_indices] = q
    mujoco.mj_forward(model, data)
    p_cur, r_cur = _target_pose(data, target, tcp_offset)
    pos_err = target_pos - p_cur
    ori_err = _rot_error_vec(r_cur, target_rot)
    pos_norm = float(np.linalg.norm(pos_err))
    ori_norm = float(np.linalg.norm(ori_err))
    residual = np.concatenate((pos_err, float(orientation_weight) * ori_err))
    return float(np.linalg.norm(residual)), pos_err, ori_err, pos_norm, ori_norm


def _resolve_kinematic_target(model: mujoco.MjModel, *, ee_body: str, ee_site: str | None) -> _KinematicTarget:
    if ee_site:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if site_id < 0:
            raise ValueError(f"Site not found: {ee_site}")
        return _KinematicTarget(kind="site", id=int(site_id), name=str(ee_site))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
    if body_id < 0:
        raise ValueError(f"Body not found: {ee_body}")
    return _KinematicTarget(kind="body", id=int(body_id), name=str(ee_body))


def _target_pose(data: mujoco.MjData, target: _KinematicTarget, tcp_offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if target.kind == "site":
        rot = data.site_xmat[target.id].reshape(3, 3)
        pos = data.site_xpos[target.id] + rot @ tcp_offset
        return pos, rot
    rot = data.xmat[target.id].reshape(3, 3)
    pos = data.xpos[target.id] + rot @ tcp_offset
    return pos, rot


def _target_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: _KinematicTarget,
    jacp: np.ndarray,
    jacr: np.ndarray,
) -> None:
    if target.kind == "site":
        mujoco.mj_jacSite(model, data, jacp, jacr, target.id)
    else:
        mujoco.mj_jacBody(model, data, jacp, jacr, target.id)


def _fk_positions(
    robot: MujocoRobotDynamics,
    q_path: np.ndarray,
    ee_body: str,
    tcp_offset: np.ndarray,
    *,
    ee_site: str | None = None,
) -> np.ndarray:
    model = robot.model
    target = _resolve_kinematic_target(model, ee_body=ee_body, ee_site=ee_site)
    data = mujoco.MjData(model)
    xyz = np.zeros((q_path.shape[0], 3), dtype=np.float64)
    for idx, q in enumerate(q_path):
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[robot.qpos_indices] = q
        mujoco.mj_forward(model, data)
        xyz[idx], _ = _target_pose(data, target, tcp_offset)
    return xyz


def _cartesian_s(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[0] < 2 or xyz.shape[1] != 3:
        raise ValueError("Cartesian path positions must be an Nx3 array with at least two samples")
    if np.any(~np.isfinite(xyz)):
        raise ValueError("Cartesian path positions must be finite")
    ds = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    zero_segments = np.nonzero(ds <= 1e-12)[0]
    if zero_segments.size:
        raise ValueError(f"zero-length Cartesian segment at sample {int(zero_segments[0])}")
    cumulative = np.concatenate(([0.0], np.cumsum(ds)))
    if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 0.0:
        raise ValueError("Cartesian path arc length must be positive")
    return cumulative / cumulative[-1]
