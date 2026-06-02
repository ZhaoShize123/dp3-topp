from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco  # type: ignore
import numpy as np


@dataclass(frozen=True)
class MujocoRobotDynamics:
    model: mujoco.MjModel
    joint_names: list[str]
    dof_indices: np.ndarray
    qpos_indices: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def from_model_path(cls, path: Path) -> "MujocoRobotDynamics":
        model_path = Path(path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        xml_text = model_path.read_text(encoding="utf-8")
        assets: dict[str, bytes] = {}
        for asset_path in model_path.parent.iterdir():
            if asset_path.is_file() and asset_path.name != model_path.name:
                assets[asset_path.name] = asset_path.read_bytes()
        model = mujoco.MjModel.from_xml_string(xml_text, assets)
        hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
        actuated_hinges = _actuated_hinge_joint_ids(model, hinge)
        joints: list[tuple[int, int, str, float, float]] = []
        for joint_id in range(model.njnt):
            if int(model.jnt_type[joint_id]) != hinge:
                continue
            if actuated_hinges and int(joint_id) not in actuated_hinges:
                continue
            dof = int(model.jnt_dofadr[joint_id])
            qpos = int(model.jnt_qposadr[joint_id])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
            lower, upper = map(float, model.jnt_range[joint_id])
            if not bool(model.jnt_limited[joint_id]) or not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError(f"selected MuJoCo hinge joint requires a finite joint range: {name}")
            joints.append((dof, qpos, name, lower, upper))
        joints.sort(key=lambda item: item[0])
        if not joints:
            raise ValueError("MuJoCo model must contain at least one selected hinge joint")
        return cls(
            model=model,
            joint_names=[item[2] for item in joints],
            dof_indices=np.asarray([item[0] for item in joints], dtype=np.int32),
            qpos_indices=np.asarray([item[1] for item in joints], dtype=np.int32),
            lower=np.asarray([item[3] for item in joints], dtype=np.float64),
            upper=np.asarray([item[4] for item in joints], dtype=np.float64),
        )

    @property
    def dof(self) -> int:
        return int(self.dof_indices.size)

    def inverse_dynamics(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        q = _check_state_vector("q", q, self.dof)
        qd = _check_state_vector("qd", qd, self.dof)
        qdd = _check_state_vector("qdd", qdd, self.dof)
        data = mujoco.MjData(self.model)
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.qpos[self.qpos_indices] = q
        data.qvel[self.dof_indices] = qd
        data.qacc[self.dof_indices] = qdd
        mujoco.mj_inverse(self.model, data)
        rigid_tau = data.qfrc_inverse + data.qfrc_passive
        return np.asarray(rigid_tau[self.dof_indices], dtype=np.float64).copy()

    def assert_joint_positions_in_range(self, q: np.ndarray, tol: float = 1e-9) -> None:
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != self.dof:
            raise ValueError(f"q must have shape (samples, {self.dof})")
        if np.any(~np.isfinite(q)):
            raise ValueError("q must contain only finite values")
        lower = self.lower[None, :] - float(tol)
        upper = self.upper[None, :] + float(tol)
        bad = np.argwhere((q < lower) | (q > upper))
        if bad.size == 0:
            return
        details = []
        for sample, axis in bad[:5]:
            details.append(
                f"{self.joint_names[int(axis)]} sample {int(sample)} "
                f"q={float(q[int(sample), int(axis)]):.6g} "
                f"outside [{float(self.lower[int(axis)]):.6g}, {float(self.upper[int(axis)]):.6g}]"
            )
        suffix = "" if bad.shape[0] <= 5 else f"; ... {bad.shape[0] - 5} more"
        raise ValueError("path violates MuJoCo joint position limits: " + "; ".join(details) + suffix)

    def torque_rate_finite_difference(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        qddd: np.ndarray,
        dt: float = 1e-5,
    ) -> np.ndarray:
        q = _check_state_vector("q", q, self.dof)
        qd = _check_state_vector("qd", qd, self.dof)
        qdd = _check_state_vector("qdd", qdd, self.dof)
        qddd = _check_state_vector("qddd", qddd, self.dof)
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        half = 0.5 * dt
        q_plus = q + qd * half + 0.5 * qdd * half**2 + (qddd * half**3) / 6.0
        q_minus = q - qd * half + 0.5 * qdd * half**2 - (qddd * half**3) / 6.0
        qd_plus = qd + qdd * half + 0.5 * qddd * half**2
        qd_minus = qd - qdd * half + 0.5 * qddd * half**2
        qdd_plus = qdd + qddd * half
        qdd_minus = qdd - qddd * half
        tau_plus = self.inverse_dynamics(q_plus, qd_plus, qdd_plus)
        tau_minus = self.inverse_dynamics(q_minus, qd_minus, qdd_minus)
        return (tau_plus - tau_minus) / dt


def _actuated_hinge_joint_ids(model: mujoco.MjModel, hinge_type: int) -> set[int]:
    joint_trn = int(mujoco.mjtTrn.mjTRN_JOINT)
    joint_ids: set[int] = set()
    for actuator_id in range(model.nu):
        if int(model.actuator_trntype[actuator_id]) != joint_trn:
            continue
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if 0 <= joint_id < model.njnt and int(model.jnt_type[joint_id]) == int(hinge_type):
            joint_ids.add(joint_id)
    return joint_ids


def _check_state_vector(name: str, values: np.ndarray, dof: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (int(dof),):
        raise ValueError(f"{name} must have shape ({int(dof)},)")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr
